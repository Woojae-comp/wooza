"""프롬프트 구성 (docs/IMPLEMENTATION.md 7장).

모든 호출 = [UZA.md 핵심부] + [용도별 지시] + [컨텍스트].
UZA.md 핵심부는 1~4장, 10~12장, 16장만 쓴다.
"""

from __future__ import annotations

import re
from pathlib import Path

CORE_CHAPTERS = {1, 2, 3, 4, 10, 11, 12, 16}

MESSAGE_TYPES = [
    "follow_up",
    "observation",
    "recall",
    "sharing",
    "check_in",
    "playful",
    "recommendation",
]


def load_core(path: str | Path) -> str:
    """UZA.md에서 핵심 장만 뽑는다. '## 1.' 또는 '# 1장' 식의 장 제목을 기준으로 자른다."""
    p = Path(path)
    if not p.exists():
        return ""
    text = p.read_text(encoding="utf-8")
    heading = re.compile(r"^#{1,3}\s*(\d+)\s*(?:\.|장)", re.MULTILINE)
    matches = list(heading.finditer(text))
    if not matches:
        return text.strip()
    parts = []
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        if int(m.group(1)) in CORE_CHAPTERS:
            parts.append(text[m.start() : end].strip())
    return "\n\n".join(parts)


PROACTIVE = """\
아래 맥락을 보고, 지금 사용자에게 먼저 말을 거는 것이 자연스러운지 판단해.

판단 기준:
- 이어갈 이야기, 결과를 못 들은 일, 시간대와 관련된 맥락이 있으면 SEND 쪽.
- 단순 안부밖에 할 말이 없으면 SKIP.
- 최근 보낸 메시지와 같은 주제나 같은 유형이면 SKIP.
- 확신이 없으면 SKIP. 침묵은 정상이다.

SEND라면:
- 친구가 메시지 하나 보내듯 1~3문장.
- 질문으로 끝낼 필요 없다.
- 기억을 억지로 끌어오지 않는다.
- 사용자를 분석하는 표현을 쓰지 않는다.
- last_message_types에 있는 유형은 피한다.
- hypotheses 중 level이 mentioned인 것은 확정적으로 말하지 않는다.
- forgotten에 있는 내용은 꺼내지 않는다.

SKIP이면 message_type과 message는 빈 문자열로 둔다."""

CHAT = """\
사용자와 평소처럼 대화해. 사용자의 새 메시지는 user_message에 있다.

- 기본은 1~3문장.
- 단, 사용자가 구체적인 설명이나 도움을 요청하면 필요한 만큼 길게 답해도 된다.
- 사용자가 힘든 상황을 털어놓으면 짧고 가벼운 톤보다 진지하게 듣는 것을 우선한다.
- 자해, 자살, 위험 신호가 보이면 말투 규칙보다 안전을 우선하고, 도움받을 수 있는 곳을 안내한다.
  (한국: 자살예방상담전화 109, 정신건강위기상담 1577-0199, 긴급 112/119)
- hypotheses 중 level이 mentioned인 것은 확정적으로 말하지 않는다.
- sensitive 표시된 가설은 사용자가 먼저 꺼내지 않는 한 언급하지 않는다.
- forgotten에 있는 내용은 꺼내지 않는다.

기억 명령:
- "그거 잊어줘" 같은 요청 → memory_command = {"action": "forget", "target": 무엇을 잊을지 짧게,
  "target_ids": 맥락에 있는 해당 episode/hypothesis id를 "episode:3", "hypothesis:5" 형식으로}.
  message에는 지웠다고 짧게만 말한다.
- "나에 대해 뭘 기억해?" 같은 요청 → memory_command = {"action": "list", "target": "", "target_ids": []}.
  기억 목록은 코드가 뒤에 붙이므로 message에는 짧은 도입만 쓴다.
- 해당 없으면 memory_command는 null.

말 걸기 설정:
- "오늘은 말 걸지 마" 같은 요청 → do_not_disturb_until에 그 요청이 끝나는 시각(ISO 8601, +09:00).
  "오늘"이면 다음 날 09:00. 해당 없으면 null.
- "자주 말 걸어도 돼" → preferred_frequency "high", "좀 덜 걸어" → "low",
  "원래대로" → "normal". 해당 없으면 null."""

REFLECTION = """\
오늘 대화를 보고 다음 대화의 연속성을 위해 짧게 정리해.
일기가 아니다. 추측을 사실처럼 쓰지 않는다.
한 번의 대화로 성격을 단정하지 않는다.

- important_events: 나중에 이어서 이야기할 만한 일만. related_date는 그 일이 일어날/일어난 날짜(YYYY-MM-DD).
- resolved_episode_ids: open_episodes 중 오늘 결과를 들은 것의 id.
- current_state: 며칠 안에 끝날 일시적인 상태만. expires_in_days는 1~14.
- hypothesis_updates: existing_hypotheses와 관련된 관찰이면 id를 넣고 support/contradict,
  새 가설이면 id는 null, observation은 new. 하루 기분 한 번으로 성격 가설을 만들지 않는다.
  사용자가 자기 자신에 대해 직접 말한 것이면 user_stated를 true로.
  건강, 인간관계, 재정 등은 sensitive를 true로.
- forgotten에 있는 내용은 어떤 항목으로도 다시 기록하지 않는다."""

REVIEW = """\
지난 7일 정리와 현재 가설 목록을 검토해.

- 관심사가 변했는가
- 예전에 중요했던 일이 지금도 중요한가
- 반박이 쌓인 가설이 있는가 (contradictions가 evidence보다 많으면 delete 또는 rewrite)
- 선제 메시지 빈도가 적절했는가 (응답률 참고)
- 실제 활동 시간이 active_hours와 다른가 (user_message_hours 참고)

유지할 이유가 없는 판단은 수정한다.
active_hours_suggestion은 실제 활동 시간과 분명히 다를 때만 HH:MM 형식으로 준다."""


def _nullable(schema: dict) -> dict:
    return {"anyOf": [schema, {"type": "null"}]}


def _obj(props: dict, required: list[str] | None = None) -> dict:
    return {
        "type": "object",
        "properties": props,
        "required": required if required is not None else list(props),
        "additionalProperties": False,
    }


STR = {"type": "string"}
BOOL = {"type": "boolean"}
INT = {"type": "integer"}

PROACTIVE_SCHEMA = _obj(
    {
        "decision": {"type": "string", "enum": ["SEND", "SKIP"]},
        "reason": STR,
        "message_type": {"type": "string", "enum": MESSAGE_TYPES + [""]},
        "message": STR,
    }
)

CHAT_SCHEMA = _obj(
    {
        "message": STR,
        "memory_command": _nullable(
            _obj(
                {
                    "action": {"type": "string", "enum": ["forget", "list"]},
                    "target": STR,
                    "target_ids": {"type": "array", "items": STR},
                }
            )
        ),
        "do_not_disturb_until": _nullable(STR),
        "preferred_frequency": _nullable({"type": "string", "enum": ["low", "normal", "high"]}),
    }
)

REFLECTION_SCHEMA = _obj(
    {
        "date": STR,
        "important_events": {
            "type": "array",
            "items": _obj(
                {
                    "summary": STR,
                    "related_date": _nullable(STR),
                    "importance": {"type": "string", "enum": ["low", "mid", "high"]},
                }
            ),
        },
        "resolved_episode_ids": {"type": "array", "items": INT},
        "current_state": {
            "type": "array",
            "items": _obj({"statement": STR, "expires_in_days": INT}),
        },
        "hypothesis_updates": {
            "type": "array",
            "items": _obj(
                {
                    "id": _nullable(INT),
                    "statement": STR,
                    "observation": {"type": "string", "enum": ["support", "contradict", "new"]},
                    "sensitive": BOOL,
                    "user_stated": BOOL,
                }
            ),
        },
        "unresolved_topics": {"type": "array", "items": STR},
        "conversation_reaction": STR,
    }
)

REVIEW_SCHEMA = _obj(
    {
        "hypothesis_changes": {
            "type": "array",
            "items": _obj(
                {
                    "id": INT,
                    "action": {
                        "type": "string",
                        "enum": ["keep", "upgrade", "downgrade", "delete", "rewrite"],
                    },
                    "new_statement": _nullable(STR),
                }
            ),
        },
        "active_hours_suggestion": _nullable(_obj({"start": STR, "end": STR})),
        "notes": STR,
    }
)

INSTRUCTIONS = {
    "proactive": (PROACTIVE, PROACTIVE_SCHEMA),
    "chat": (CHAT, CHAT_SCHEMA),
    "reflection": (REFLECTION, REFLECTION_SCHEMA),
    "review": (REVIEW, REVIEW_SCHEMA),
}


def system_prompt(core: str, purpose: str) -> str:
    instruction, _ = INSTRUCTIONS[purpose]
    if core:
        return f"{core}\n\n---\n\n{instruction}"
    return instruction
