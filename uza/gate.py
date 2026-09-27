"""게이트: 모델을 부르기 전에 코드가 검사하는 하드 규칙 (docs/IMPLEMENTATION.md 4장)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time, timedelta

from .config import Config, parse_hhmm
from .store import Store

FREQUENCY_DELTA = {"low": None, "normal": 0, "high": 2}
LOW_FREQUENCY_MAX = 2


@dataclass
class GateResult:
    passed: bool
    reason: str | None = None


def day_start(now: datetime) -> datetime:
    return now.replace(hour=0, minute=0, second=0, microsecond=0)


def active_hours(cfg: Config, store: Store) -> tuple[time, time]:
    s = store.settings()
    start = parse_hhmm(s.active_start) if s.active_start else cfg.active_start
    end = parse_hhmm(s.active_end) if s.active_end else cfg.active_end
    return start, end


def in_active_hours(now: datetime, start: time, end: time) -> bool:
    t = now.time()
    if start <= end:
        return start <= t < end
    return t >= start or t < end  # 자정을 넘기는 구간


def response_rate_7d(store: Store, now: datetime) -> tuple[float | None, int]:
    """최근 7일 선제 메시지 응답률. 반응 대기 중인 것은 제외한다."""
    logs = [l for l in store.sent_logs_since(now - timedelta(days=7)) if l.replied is not None]
    if not logs:
        return None, 0
    return sum(1 for l in logs if l.replied) / len(logs), len(logs)


def daily_max(cfg: Config, store: Store, now: datetime) -> int:
    """하루 최대치 = 사용자 선호 반영 → backoff 상한 적용."""
    pref = store.settings().preferred_frequency
    if pref == "low":
        limit = min(LOW_FREQUENCY_MAX, cfg.max_proactive_per_day)
    else:
        limit = cfg.max_proactive_per_day + FREQUENCY_DELTA.get(pref, 0)

    rate, samples = response_rate_7d(store, now)
    if rate is not None and samples >= cfg.backoff_min_samples:
        for rule in cfg.backoff:  # response_rate_below 오름차순
            if rate < rule.response_rate_below:
                limit = min(limit, rule.max_proactive_per_day)
                break
    return limit


def consecutive_unanswered_today(store: Store, now: datetime) -> int:
    count = 0
    for log in reversed(store.sent_logs_since(day_start(now))):
        if log.replied:
            break
        count += 1
    return count


def check(cfg: Config, store: Store, now: datetime) -> GateResult:
    settings = store.settings()

    # 6. 사용자가 말 걸지 말라고 했다
    if settings.do_not_disturb_until and now < settings.do_not_disturb_until:
        return GateResult(False, "gate_do_not_disturb")

    # 1. active_hours 밖
    start, end = active_hours(cfg, store)
    if not in_active_hours(now, start, end):
        return GateResult(False, "gate_active_hours")

    # 2. 오늘 하루 최대치
    today_sent = store.sent_logs_since(day_start(now))
    if len(today_sent) >= daily_max(cfg, store, now):
        return GateResult(False, "gate_daily_max")

    # 3. 마지막 메시지(누가 보냈든) 이후 최소 간격
    last = store.last_message()
    if last and now - last.timestamp < cfg.min_gap_after_any_message:
        return GateResult(False, "gate_recent_message")

    # 4. 마지막 선제 메시지 무응답 + 대기 시간 미경과
    recent = store.recent_sent_logs(1)
    if recent and not recent[0].replied and now - recent[0].timestamp < cfg.min_gap_after_unanswered:
        return GateResult(False, "gate_unanswered_wait")

    # 5. 오늘 연속 무응답
    if consecutive_unanswered_today(store, now) >= cfg.max_consecutive_unanswered:
        return GateResult(False, "gate_consecutive_unanswered")

    return GateResult(True)
