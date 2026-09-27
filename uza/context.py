"""컨텍스트 빌더 (docs/IMPLEMENTATION.md 6장).

원칙: 많이 넣지 않는다. 넣은 만큼 "아는 척"이 늘어난다.
"""

from __future__ import annotations

from datetime import datetime, timedelta

from .config import Config
from .gate import day_start
from .store import Episode, Hypothesis, Store

WEEKDAY_KO = "월화수목금토일"
LEVEL_RANK = {"confirmed": 0, "repeated": 1, "mentioned": 2}
IMPORTANCE_RANK = {"high": 0, "mid": 1, "low": 2}
UNRESOLVED_WINDOW_DAYS = 2


def fmt_dt(dt: datetime) -> str:
    return f"{dt:%Y-%m-%d} ({WEEKDAY_KO[dt.weekday()]}) {dt:%H:%M}"


def _episode_dict(e: Episode) -> dict:
    d = {"id": e.id, "date": e.date.isoformat(), "summary": e.summary, "resolved": e.resolved}
    if e.related_date:
        d["related_date"] = f"{e.related_date.isoformat()} ({WEEKDAY_KO[e.related_date.weekday()]})"
    return d


def _hypothesis_dict(h: Hypothesis) -> dict:
    d = {"id": h.id, "statement": h.statement, "level": h.level}
    if h.sensitive:
        d["sensitive"] = True
    return d


def select_hypotheses(hyps: list[Hypothesis], limit: int, include_sensitive: bool) -> list[Hypothesis]:
    pool = [h for h in hyps if include_sensitive or not h.sensitive]
    pool.sort(key=lambda h: (LEVEL_RANK[h.level], -(h.last_seen.toordinal() if h.last_seen else 0)))
    return pool[:limit]


def build(cfg: Config, store: Store, now: datetime, purpose: str) -> dict:
    """purpose: 'proactive' | 'chat'."""
    today = now.date()
    last = store.last_message()

    today_msgs = store.messages_between(day_start(now), now + timedelta(seconds=1))
    today_msgs = today_msgs[-cfg.recent_messages :]

    episodes = store.active_episodes(now)
    episodes.sort(key=lambda e: (e.resolved, IMPORTANCE_RANK[e.importance], -e.date.toordinal()))

    unresolved = [
        e
        for e in episodes
        if not e.resolved
        and e.related_date
        and abs((e.related_date - today).days) <= UNRESOLVED_WINDOW_DAYS
    ]

    # sensitive 가설은 선제 메시지에 쓰지 않는다 (5.3). 대화에서는 표시만 해서 넘긴다.
    hyps = select_hypotheses(store.hypotheses(), cfg.max_hypotheses, include_sensitive=purpose == "chat")

    recent_logs = store.recent_sent_logs(5)
    today_count = len(store.sent_logs_since(day_start(now)))

    ctx = {
        "now": fmt_dt(now),
        "last_message_at": fmt_dt(last.timestamp) if last else None,
        "last_message_by": last.sender if last else None,
        "today_messages": [
            {"at": f"{m.timestamp:%H:%M}", "by": m.sender, "text": m.text} for m in today_msgs
        ],
        "recent_episodes": [_episode_dict(e) for e in episodes[: cfg.max_episodes]],
        "unresolved_topics": [_episode_dict(e) for e in unresolved],
        "current_state": [s.statement for s in store.active_current_state(now)],
        "hypotheses": [_hypothesis_dict(h) for h in hyps],
        "recent_proactive": [
            {"type": l.message_type, "at": fmt_dt(l.timestamp), "replied": l.replied} for l in recent_logs
        ],
        "today_proactive_count": today_count,
        "last_message_types": [l.message_type for l in recent_logs[-2:]],
    }
    forgotten = store.forgotten()
    if forgotten:
        ctx["forgotten"] = forgotten
    return ctx
