"""설정 로드 (docs/IMPLEMENTATION.md 3장)."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import yaml

WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]


def parse_duration(value: str | int) -> timedelta:
    """'90m', '4h', '1h30m', '45s' 또는 분 단위 정수."""
    if isinstance(value, int):
        return timedelta(minutes=value)
    total = timedelta()
    parts = re.findall(r"(\d+)\s*([hms])", value.strip())
    if not parts:
        raise ValueError(f"잘못된 기간 값: {value!r}")
    for num, unit in parts:
        n = int(num)
        total += {"h": timedelta(hours=n), "m": timedelta(minutes=n), "s": timedelta(seconds=n)}[unit]
    return total


def parse_hhmm(value: str) -> time:
    h, m = value.split(":")
    return time(int(h), int(m))


@dataclass
class BackoffRule:
    response_rate_below: float
    max_proactive_per_day: int


@dataclass
class Config:
    active_start: time = time(9, 0)
    active_end: time = time(23, 0)
    check_interval: timedelta = timedelta(hours=1)
    timezone: ZoneInfo = field(default_factory=lambda: ZoneInfo("Asia/Seoul"))

    max_proactive_per_day: int = 4
    min_gap_after_any_message: timedelta = timedelta(minutes=90)
    min_gap_after_unanswered: timedelta = timedelta(hours=4)
    max_consecutive_unanswered: int = 2

    backoff: list[BackoffRule] = field(
        default_factory=lambda: [BackoffRule(0.2, 1), BackoffRule(0.5, 2)]
    )
    backoff_min_samples: int = 3

    reflection_time: time = time(23, 30)
    review_weekday: int = 6  # sunday

    recent_messages: int = 20
    max_episodes: int = 5
    max_hypotheses: int = 8

    model: str = "claude-opus-5"
    effort: dict[str, str] = field(
        default_factory=lambda: {
            "proactive": "low",
            "chat": "medium",
            "reflection": "high",
            "review": "high",
        }
    )

    db_path: str = "uza.db"
    uza_md_path: str = "UZA.md"

    @classmethod
    def load(cls, path: str | Path) -> "Config":
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: dict) -> "Config":
        c = cls()
        if ah := raw.get("active_hours"):
            c.active_start = parse_hhmm(ah["start"])
            c.active_end = parse_hhmm(ah["end"])
        if "check_interval" in raw:
            c.check_interval = parse_duration(raw["check_interval"])
        if "timezone" in raw:
            c.timezone = ZoneInfo(raw["timezone"])
        lim = raw.get("limits", {})
        c.max_proactive_per_day = lim.get("max_proactive_per_day", c.max_proactive_per_day)
        if "min_gap_after_any_message" in lim:
            c.min_gap_after_any_message = parse_duration(lim["min_gap_after_any_message"])
        if "min_gap_after_unanswered" in lim:
            c.min_gap_after_unanswered = parse_duration(lim["min_gap_after_unanswered"])
        c.max_consecutive_unanswered = lim.get(
            "max_consecutive_unanswered", c.max_consecutive_unanswered
        )
        if "backoff" in raw:
            c.backoff = sorted(
                (BackoffRule(**b) for b in raw["backoff"]), key=lambda b: b.response_rate_below
            )
        if "reflection_time" in raw:
            c.reflection_time = parse_hhmm(raw["reflection_time"])
        if "review_day" in raw:
            c.review_weekday = WEEKDAYS.index(raw["review_day"].lower())
        ctx = raw.get("context", {})
        c.recent_messages = ctx.get("recent_messages", c.recent_messages)
        c.max_episodes = ctx.get("max_episodes", c.max_episodes)
        c.max_hypotheses = ctx.get("max_hypotheses", c.max_hypotheses)
        model = raw.get("model", {})
        c.model = model.get("name", c.model)
        c.effort = {**c.effort, **model.get("effort", {})}
        c.db_path = raw.get("db_path", c.db_path)
        c.uza_md_path = raw.get("uza_md_path", c.uza_md_path)
        return c
