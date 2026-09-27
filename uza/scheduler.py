"""스케줄러: 매시간 선제 판단, 매일 하루 정리, 매주 검토 (docs/IMPLEMENTATION.md 2장)."""

from __future__ import annotations

import logging
import threading
from datetime import datetime, timedelta
from typing import Callable

from .config import Config
from .engine import Uza

log = logging.getLogger(__name__)


class Scheduler:
    def __init__(self, uza: Uza, cfg: Config, clock: Callable[[], datetime] | None = None):
        self.uza = uza
        self.cfg = cfg
        self.clock = clock or (lambda: datetime.now(cfg.timezone))

    def _crossed(self, prev: datetime, now: datetime, at: datetime) -> bool:
        return prev < at <= now

    def run_between(self, prev: datetime, now: datetime) -> list[str]:
        """(prev, now] 구간에 걸린 작업을 실행한다. 여러 슬롯을 건너뛰었어도 각 작업은 한 번만."""
        ran: list[str] = []

        # 선제 판단: 자정 기준 check_interval 배수 시각
        midnight = now.replace(hour=0, minute=0, second=0, microsecond=0)
        slot = midnight
        while slot + self.cfg.check_interval <= now:
            slot += self.cfg.check_interval
        if prev < slot <= now:
            outcome = self.uza.proactive_tick(now)
            log.info("proactive %s (%s)", outcome.decision, outcome.reason)
            ran.append("proactive")

        # 하루 정리 → (일요일이면) 주간 검토
        for day_offset in (1, 0):
            d = (now - timedelta(days=day_offset)).date()
            at = datetime.combine(d, self.cfg.reflection_time, self.cfg.timezone)
            if self._crossed(prev, now, at):
                self.uza.reflect(at)
                ran.append("reflection")
                if at.weekday() == self.cfg.review_weekday:
                    self.uza.review(at)
                    ran.append("review")
        return ran

    def run_forever(self, stop: threading.Event, poll_seconds: float = 30.0) -> None:
        prev = self.clock()
        while not stop.wait(poll_seconds):
            now = self.clock()
            try:
                self.run_between(prev, now)
            except Exception:  # 한 번의 실패로 스케줄러가 멈추지 않게
                log.exception("스케줄 작업 실패")
            prev = now
