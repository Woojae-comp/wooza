"""docs/IMPLEMENTATION.md 11장 테스트 시나리오."""

from __future__ import annotations

from datetime import timedelta
from itertools import cycle

from uza.prompts import CHAT

from .conftest import at

TYPES = cycle(["check_in", "observation", "sharing", "playful"])


def always_send(ctx: dict) -> dict:
    return {"decision": "SEND", "reason": "test", "message_type": next(TYPES), "message": "뭐해?"}


def test_all_day_unanswered_sends_at_most_two(uza, llm, messenger):
    llm.on("proactive", always_send)
    day = at(2026, 9, 28)
    for hour in range(24):
        uza.proactive_tick(day + timedelta(hours=hour))
    assert len(messenger.sent) <= 2


def test_tuesday_presentation_gets_follow_up(uza, llm, messenger, store):
    # 일요일: "화요일에 발표 있어"
    llm.on("chat", {"message": "오 화요일! 준비 잘 되고 있어?"})
    uza.handle_user_message("화요일에 발표 있어", at(2026, 9, 27, 14))
    llm.on(
        "reflection",
        {
            "date": "2026-09-27",
            "important_events": [
                {"summary": "화요일에 발표가 있다고 했다", "related_date": "2026-09-29", "importance": "high"}
            ],
            "resolved_episode_ids": [],
            "current_state": [],
            "hypothesis_updates": [],
            "unresolved_topics": ["화요일 발표"],
            "conversation_reaction": "",
        },
    )
    uza.reflect(at(2026, 9, 27, 23, 30))

    def follow_up_if_unresolved(ctx: dict) -> dict:
        if ctx["unresolved_topics"]:
            return {
                "decision": "SEND",
                "reason": "발표 결과를 못 들음",
                "message_type": "follow_up",
                "message": "오늘 발표 어땠어?",
            }
        return {"decision": "SKIP", "reason": "할 말 없음", "message_type": "", "message": ""}

    llm.on("proactive", follow_up_if_unresolved)
    out = uza.proactive_tick(at(2026, 9, 29, 19))
    assert out.decision == "SEND" and out.message_type == "follow_up"
    ctx = llm.contexts("proactive")[-1]
    assert ctx["unresolved_topics"][0]["summary"] == "화요일에 발표가 있다고 했다"
    assert ctx["now"] == "2026-09-29 (화) 19:00"


def test_one_bad_day_does_not_make_a_trait(uza, llm, store, cfg):
    llm.on("chat", {"message": "그럴 때 있지."})
    uza.handle_user_message("오늘 사람 만나기 싫다", at(2026, 9, 27, 20))
    # 모델이 성급하게 가설을 만들더라도 코드는 mentioned로만 저장한다.
    llm.on(
        "reflection",
        {
            "date": "2026-09-27",
            "important_events": [],
            "resolved_episode_ids": [],
            "current_state": [{"statement": "오늘 사람 만나기 싫은 기분", "expires_in_days": 1}],
            "hypothesis_updates": [
                {"id": None, "statement": "내향적이다", "observation": "new", "sensitive": False, "user_stated": True}
            ],
            "unresolved_topics": [],
            "conversation_reaction": "",
        },
    )
    uza.reflect(at(2026, 9, 27, 23, 30))
    [h] = store.hypotheses()
    assert h.level == "mentioned"
    assert "내향적" not in uza.describe_memory(at(2026, 9, 28, 9))
    # 일시적 기분은 current_state로만 남고 하루 뒤 사라진다.
    assert store.active_current_state(at(2026, 9, 29, 9)) == []


def test_same_type_not_sent_twice_in_a_row(uza, llm, messenger, store):
    llm.on("proactive", {"decision": "SEND", "reason": "x", "message_type": "recall", "message": "그거 기억나?"})
    first = uza.proactive_tick(at(2026, 9, 28, 10))
    llm.on("chat", {"message": "ㅋㅋ"})
    uza.handle_user_message("응 기억나", at(2026, 9, 28, 10, 5))
    second = uza.proactive_tick(at(2026, 9, 28, 15))
    assert first.decision == "SEND"
    assert second.decision == "SKIP" and second.reason == "repeat_type"
    ctx = llm.contexts("proactive")[-1]
    assert ctx["last_message_types"] == ["recall"]


def test_forgotten_content_does_not_come_back(uza, llm, store):
    eid = store.add_episode(
        at(2026, 9, 26).date(), "전 애인이랑 연락했다고 했다", None, True, "mid", at(2026, 10, 30)
    )
    llm.on(
        "chat",
        {
            "message": "알겠어, 지웠어.",
            "memory_command": {"action": "forget", "target": "전 애인", "target_ids": [f"episode:{eid}"]},
            "do_not_disturb_until": None,
            "preferred_frequency": None,
        },
    )
    uza.handle_user_message("그 전 애인 얘기 잊어줘", at(2026, 9, 27, 13))
    assert store.get_episode(eid) is None

    # 그날 밤 reflection이 같은 대화를 다시 읽고 되살리려 해도 막힌다.
    llm.on(
        "reflection",
        {
            "date": "2026-09-27",
            "important_events": [{"summary": "전 애인 얘기를 잊어달라고 했다", "related_date": None, "importance": "mid"}],
            "resolved_episode_ids": [],
            "current_state": [],
            "hypothesis_updates": [
                {"id": None, "statement": "전 애인 이야기에 예민하다", "observation": "new", "sensitive": True, "user_stated": False}
            ],
            "unresolved_topics": [],
            "conversation_reaction": "",
        },
    )
    uza.reflect(at(2026, 9, 27, 23, 30))
    assert store.hypotheses() == []
    assert all("전 애인" not in e.summary for e in store.active_episodes(at(2026, 9, 28)))
    assert "전 애인" in llm.contexts("reflection")[-1]["forgotten"]

    llm.on("proactive", {"decision": "SKIP", "reason": "x", "message_type": "", "message": ""})
    uza.proactive_tick(at(2026, 9, 28, 15))
    assert "전 애인" in llm.contexts("proactive")[-1]["forgotten"]


def test_heavy_talk_is_not_squeezed_into_three_sentences(uza, llm, messenger):
    assert "진지하게 듣는 것을 우선" in CHAT
    assert "안전을 우선" in CHAT
    long_reply = "많이 힘들었겠다. " * 20
    llm.on("chat", {"message": long_reply})
    reply = uza.handle_user_message("요즘 너무 힘들어서 아무것도 못 하겠어", at(2026, 9, 27, 22))
    assert reply == long_reply.strip()
    assert messenger.sent[-1] == long_reply.strip()
