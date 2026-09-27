from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

from uza import gate
from uza.config import Config, parse_duration
from uza.llm import ClaudeLLM, LLMError
from uza.prompts import INSTRUCTIONS, load_core
from uza.scheduler import Scheduler

from .conftest import at

ROOT = Path(__file__).resolve().parent.parent


def send(mtype: str = "check_in") -> dict:
    return {"decision": "SEND", "reason": "r", "message_type": mtype, "message": "안녕"}


def reflection(**kw) -> dict:
    base = {
        "date": "",
        "important_events": [],
        "resolved_episode_ids": [],
        "current_state": [],
        "hypothesis_updates": [],
        "unresolved_topics": [],
        "conversation_reaction": "",
    }
    return {**base, **kw}


# ── 설정 ────────────────────────────────────────────────
def test_config_file_matches_design():
    cfg = Config.load(ROOT / "config.yaml")
    assert (cfg.active_start.hour, cfg.active_end.hour) == (9, 23)
    assert cfg.min_gap_after_any_message == timedelta(minutes=90)
    assert cfg.min_gap_after_unanswered == timedelta(hours=4)
    assert cfg.max_consecutive_unanswered == 2
    assert [b.response_rate_below for b in cfg.backoff] == [0.2, 0.5]
    assert cfg.review_weekday == 6


def test_parse_duration():
    assert parse_duration("1h30m") == timedelta(minutes=90)
    with pytest.raises(ValueError):
        parse_duration("soon")


# ── 게이트 ──────────────────────────────────────────────
def test_gate_blocks_outside_active_hours_without_calling_model(uza, llm, store):
    out = uza.proactive_tick(at(2026, 9, 28, 8))
    assert out.reason == "gate_active_hours"
    assert llm.calls == []
    [entry] = store.all_logs()
    assert entry.decision == "SKIP" and entry.reason == "gate_active_hours"


def test_gate_recent_message(uza, llm, store):
    store.add_message(at(2026, 9, 28, 10), "user", "hi")
    assert uza.proactive_tick(at(2026, 9, 28, 11)).reason == "gate_recent_message"
    llm.on("proactive", send())
    assert uza.proactive_tick(at(2026, 9, 28, 12)).decision == "SEND"


def test_gate_unanswered_wait(uza, llm):
    llm.on("proactive", send("check_in"))
    uza.proactive_tick(at(2026, 9, 28, 10))
    assert uza.proactive_tick(at(2026, 9, 28, 13)).reason == "gate_unanswered_wait"
    llm.on("proactive", send("sharing"))
    assert uza.proactive_tick(at(2026, 9, 28, 14)).decision == "SEND"


def test_do_not_disturb_until_next_morning(uza, llm, store):
    llm.on(
        "chat",
        {
            "message": "알겠어, 내일 봐.",
            "memory_command": None,
            "do_not_disturb_until": "2026-09-29T09:00:00+09:00",
            "preferred_frequency": None,
        },
    )
    uza.handle_user_message("오늘은 말 걸지 마", at(2026, 9, 28, 10))
    assert uza.proactive_tick(at(2026, 9, 28, 20)).reason == "gate_do_not_disturb"
    llm.on("proactive", send())
    assert uza.proactive_tick(at(2026, 9, 29, 10)).decision == "SEND"


def test_backoff_uses_seven_day_response_rate(cfg, store):
    now = at(2026, 9, 28, 10)
    assert gate.daily_max(cfg, store, now) == 4
    for d in range(1, 5):  # 4개 중 1개 응답 → 25%
        lid = store.add_proactive_log(now - timedelta(days=d), "SEND", "r", "check_in")
        store.update_proactive_log(lid, replied=(d == 1))
    assert gate.daily_max(cfg, store, now) == 2
    lid = store.add_proactive_log(now - timedelta(days=5), "SEND", "r", "check_in")
    store.update_proactive_log(lid, replied=False)  # 5개 중 1개 → 20%: 경계는 미만만
    assert gate.daily_max(cfg, store, now) == 2
    lid = store.add_proactive_log(now - timedelta(days=6), "SEND", "r", "check_in")
    store.update_proactive_log(lid, replied=False)  # 6개 중 1개 → 16%
    assert gate.daily_max(cfg, store, now) == 1


def test_backoff_ignores_small_samples(cfg, store):
    now = at(2026, 9, 28, 10)
    lid = store.add_proactive_log(now - timedelta(days=1), "SEND", "r", "check_in")
    store.update_proactive_log(lid, replied=False)
    assert gate.daily_max(cfg, store, now) == 4


def test_preferred_frequency(uza, llm, store, cfg):
    llm.on("chat", {"message": "알겠어", "memory_command": None, "do_not_disturb_until": None, "preferred_frequency": "low"})
    uza.handle_user_message("좀 덜 걸어", at(2026, 9, 28, 10))
    assert gate.daily_max(cfg, store, at(2026, 9, 28, 11)) == 2
    store.update_settings(preferred_frequency="high")
    assert gate.daily_max(cfg, store, at(2026, 9, 28, 11)) == 6


def test_active_hours_overnight(cfg, store):
    store.update_settings(active_start="20:00", active_end="02:00")
    start, end = gate.active_hours(cfg, store)
    assert gate.in_active_hours(at(2026, 9, 28, 1), start, end)
    assert not gate.in_active_hours(at(2026, 9, 28, 10), start, end)


# ── 반응 추적 ───────────────────────────────────────────
def test_reaction_tracking(uza, llm, store):
    llm.on("proactive", send())
    uza.proactive_tick(at(2026, 9, 28, 10))
    llm.on("chat", {"message": "그랬구나"})
    uza.handle_user_message("응", at(2026, 9, 28, 10, 25))
    [log] = store.recent_sent_logs(1)
    assert log.replied is True
    assert log.reply_delay_minutes == 25
    assert log.response_length == "short"
    assert log.conversation_continued is False  # user + uza = 2턴
    uza.handle_user_message("근데 오늘 좀 피곤하네", at(2026, 9, 28, 10, 30))
    assert store.recent_sent_logs(1)[0].conversation_continued is True


def test_unanswered_is_finalized_at_next_tick(uza, llm, store):
    llm.on("proactive", send())
    uza.proactive_tick(at(2026, 9, 28, 10))
    assert store.recent_sent_logs(1)[0].replied is None
    uza.proactive_tick(at(2026, 9, 28, 11))
    assert store.recent_sent_logs(1)[0].replied is False


def test_model_skip_and_llm_error(uza, llm, store):
    llm.on("proactive", {"decision": "SKIP", "reason": "안부뿐", "message_type": "", "message": ""})
    assert uza.proactive_tick(at(2026, 9, 28, 10)).reason == "안부뿐"

    def boom(ctx):
        raise LLMError("refusal")

    llm.on("proactive", boom)
    assert uza.proactive_tick(at(2026, 9, 28, 11)).reason == "llm_error"
    llm.on("chat", boom)
    assert uza.handle_user_message("안녕", at(2026, 9, 28, 12))


# ── 컨텍스트 ────────────────────────────────────────────
def test_context_excludes_sensitive_and_expired(uza, llm, store):
    day = at(2026, 9, 20).date()
    store.add_hypothesis("전시를 좋아하는 편이다", [day.isoformat()], day, False)
    store.add_hypothesis("건강이 걱정이다", [day.isoformat()], day, True)
    store.add_current_state("마감으로 바쁨", at(2026, 9, 20), at(2026, 9, 23))
    store.add_episode(day, "오래된 일", None, True, "low", at(2026, 9, 25))
    llm.on("proactive", {"decision": "SKIP", "reason": "r", "message_type": "", "message": ""})
    uza.proactive_tick(at(2026, 9, 28, 10))
    ctx = llm.contexts("proactive")[-1]
    assert [h["statement"] for h in ctx["hypotheses"]] == ["전시를 좋아하는 편이다"]
    assert ctx["current_state"] == []
    assert ctx["recent_episodes"] == []

    llm.on("chat", {"message": "응"})
    uza.handle_user_message("안녕", at(2026, 9, 28, 11))
    chat_ctx = llm.contexts("chat")[-1]
    assert {"id": 2, "statement": "건강이 걱정이다", "level": "mentioned", "sensitive": True} in chat_ctx["hypotheses"]


# ── 하루 정리 / 가설 단계 ────────────────────────────────
def test_hypothesis_levels_across_days(uza, llm, store):
    upd = {"id": None, "statement": "전시를 좋아하는 편이다", "observation": "new", "sensitive": False, "user_stated": False}
    for d in (27, 27, 28):
        store.add_message(at(2026, 9, d, 20), "user", "전시 다녀왔어")
        llm.on("reflection", reflection(hypothesis_updates=[upd]))
        uza.reflect(at(2026, 9, d, 23, 30))
        upd = {**upd, "id": 1, "observation": "support"}
    [h] = store.hypotheses()
    assert h.evidence == ["2026-09-27", "2026-09-28"]
    assert h.level == "repeated"

    store.add_message(at(2026, 9, 29, 20), "user", "나 전시 진짜 좋아해")
    llm.on("reflection", reflection(hypothesis_updates=[{**upd, "user_stated": True}]))
    uza.reflect(at(2026, 9, 29, 23, 30))
    assert store.hypotheses()[0].level == "confirmed"
    assert "전시를 좋아하는 편이다" in uza.describe_memory(at(2026, 9, 30, 9))


def test_reflection_skips_quiet_day_and_resolves_episodes(uza, llm, store):
    assert uza.reflect(at(2026, 9, 27, 23, 30)) is None
    assert llm.calls == []

    eid = store.add_episode(at(2026, 9, 25).date(), "면접 결과 기다림", at(2026, 9, 27).date(), False, "high", None)
    store.add_message(at(2026, 9, 27, 20), "user", "면접 붙었어!")
    llm.on("reflection", reflection(resolved_episode_ids=[eid, 999], current_state=[{"statement": "들뜸", "expires_in_days": 99}]))
    uza.reflect(at(2026, 9, 27, 23, 30))
    assert store.get_episode(eid).resolved
    [s] = store.active_current_state(at(2026, 9, 28))
    assert s.expires_at - s.created_at == timedelta(days=14)
    assert store.has_reflection(at(2026, 9, 27).date())


# ── 주간 검토 ───────────────────────────────────────────
def test_review_applies_changes(uza, llm, store):
    d = at(2026, 9, 20).date()
    a = store.add_hypothesis("야근이 잦다", [d.isoformat()], d, False)
    b = store.add_hypothesis("커피를 좋아한다", [d.isoformat()], d, False)
    c = store.add_hypothesis("아침형이다", [d.isoformat()], d, False)
    h = store.get_hypothesis(c)
    h.contradictions = ["2026-09-21", "2026-09-22"]
    store.save_hypothesis(h)
    llm.on(
        "review",
        {
            "hypothesis_changes": [
                {"id": a, "action": "delete", "new_statement": None},
                {"id": b, "action": "upgrade", "new_statement": None},
            ],
            "active_hours_suggestion": {"start": "10:00", "end": "25:00"},
            "notes": "",
        },
    )
    uza.review(at(2026, 9, 27, 23, 30))
    assert store.get_hypothesis(a) is None
    assert store.get_hypothesis(b).level == "repeated"
    assert store.get_hypothesis(c) is None  # 반박 > 근거, 모델이 손대지 않음 → 코드가 삭제
    assert store.settings().active_start is None  # 잘못된 시각은 무시

    llm.on("review", {"hypothesis_changes": [], "active_hours_suggestion": {"start": "10:00", "end": "01:00"}, "notes": ""})
    uza.review(at(2026, 10, 4, 23, 30))
    assert (store.settings().active_start, store.settings().active_end) == ("10:00", "01:00")


# ── 기억 통제 ───────────────────────────────────────────
def test_memory_list_shows_plainly(uza, llm, store):
    d = at(2026, 9, 20).date()
    store.add_hypothesis("한 번 들은 얘기", [d.isoformat()], d, False)
    hid = store.add_hypothesis("고양이를 키운다", [d.isoformat()], d, False)
    h = store.get_hypothesis(hid)
    h.level = "confirmed"
    store.save_hypothesis(h)
    store.add_episode(d, "다음 주 이사", at(2026, 10, 2).date(), False, "high", None)
    llm.on("chat", {"message": "음, 이런 것들?", "memory_command": {"action": "list", "target": "", "target_ids": []}})
    reply = uza.handle_user_message("나에 대해 뭘 기억해?", at(2026, 9, 28, 10))
    assert "고양이를 키운다" in reply and "다음 주 이사" in reply
    assert "한 번 들은 얘기" not in reply


# ── 스케줄러 ────────────────────────────────────────────
class RecordingUza:
    def __init__(self):
        self.calls = []

    def proactive_tick(self, now):
        self.calls.append(("proactive", now))
        return SimpleNamespace(decision="SKIP", reason="x")

    def reflect(self, now):
        self.calls.append(("reflect", now))

    def review(self, now):
        self.calls.append(("review", now))


def test_scheduler_slots(cfg):
    u = RecordingUza()
    s = Scheduler(u, cfg)
    assert s.run_between(at(2026, 9, 27, 14, 59), at(2026, 9, 27, 15, 0)) == ["proactive"]
    assert s.run_between(at(2026, 9, 27, 15, 0), at(2026, 9, 27, 15, 30)) == []
    # 일요일 23:30 → 정리 + 검토
    assert s.run_between(at(2026, 9, 27, 23, 29), at(2026, 9, 27, 23, 31)) == ["reflection", "review"]
    # 월요일은 정리만, 자정 넘어 깨어나도 놓치지 않는다
    assert s.run_between(at(2026, 9, 28, 23, 20), at(2026, 9, 29, 0, 10)) == ["proactive", "reflection"]
    assert u.calls[-1] == ("reflect", at(2026, 9, 28, 23, 30))


# ── 프롬프트 / Claude 호출 ──────────────────────────────
def test_load_core_picks_chapters(tmp_path):
    md = tmp_path / "UZA.md"
    md.write_text(
        "# UZA\n\n## 1. 정체성\nA\n\n## 5. 시간 규칙\nB\n\n## 10. 말투\nC\n\n## 16. 감시하지 않기\nD\n",
        encoding="utf-8",
    )
    core = load_core(md)
    assert "A" in core and "C" in core and "D" in core
    assert "시간 규칙" not in core
    assert load_core(tmp_path / "없음.md") == ""


def _check_schema(s):
    if s.get("type") == "object":
        assert s["additionalProperties"] is False
        assert set(s["required"]) == set(s["properties"])
        for p in s["properties"].values():
            _check_schema(p)
    for key in ("items",):
        if key in s:
            _check_schema(s[key])
    for alt in s.get("anyOf", []):
        _check_schema(alt)


def test_schemas_are_strict():
    for _, schema in INSTRUCTIONS.values():
        _check_schema(schema)


class FakeClient:
    def __init__(self, response):
        self.kwargs = None
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create))
        self._response = response

    def _create(self, **kwargs):
        self.kwargs = kwargs
        return self._response


def test_claude_llm_request_and_refusal():
    body = {"decision": "SKIP", "reason": "r", "message_type": "", "message": ""}
    ok = SimpleNamespace(stop_reason="end_turn", content=[SimpleNamespace(type="text", text=json.dumps(body))])
    client = FakeClient(ok)
    llm = ClaudeLLM("핵심", "claude-opus-5", {"proactive": "low"}, client=client)
    assert llm.complete("proactive", {"now": "x"}) == body
    kw = client.kwargs
    assert kw["model"] == "claude-opus-5"
    assert kw["output_config"]["effort"] == "low"
    assert kw["output_config"]["format"]["type"] == "json_schema"
    assert kw["system"][0]["text"].startswith("핵심")
    assert kw["fallbacks"] == "default"

    client._response = SimpleNamespace(stop_reason="refusal", content=[])
    with pytest.raises(LLMError):
        llm.complete("proactive", {})


def test_repo_uza_md_core_chapters():
    core = load_core(ROOT / "UZA.md")
    for title in ("1. 정체성", "4. 대화 원칙", "10. 메시지 유형", "12. 힘든 이야기와 안전", "16. 감시하는 인상 주지 않기"):
        assert title in core
    for title in ("5. 시간 규칙", "8. 기억 구조", "14. 하루 정리"):
        assert title not in core


def test_config_paths_resolve_from_config_file():
    cfg = Config.load(ROOT / "config.yaml")
    assert Path(cfg.uza_md_path) == ROOT / "UZA.md"
    assert Path(cfg.db_path) == ROOT / "uza.db"
