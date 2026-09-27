"""콘솔에서 UZA와 대화하기.

    python -m uza            # 대화 + 백그라운드 스케줄러
    python -m uza --no-scheduler

대화 중 명령: /tick (선제 판단 즉시 실행), /reflect, /review, /memory, /quit
"""

from __future__ import annotations

import argparse
import logging
import threading
from datetime import datetime

from .config import Config
from .engine import Uza
from .llm import ClaudeLLM
from .prompts import load_core
from .scheduler import Scheduler
from .store import Store


class ConsoleMessenger:
    def __init__(self) -> None:
        self._lock = threading.Lock()

    def send(self, text: str) -> None:
        with self._lock:
            print(f"\nUZA> {text}\n", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(prog="uza")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--no-scheduler", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)
    cfg = Config.load(args.config)
    core = load_core(cfg.uza_md_path)
    if not core:
        logging.warning("%s 를 찾지 못했다. 성격 원칙 없이 용도별 지시만으로 동작한다.", cfg.uza_md_path)

    store = Store(cfg.db_path)
    uza = Uza(cfg, store, ClaudeLLM(core, cfg.model, cfg.effort), ConsoleMessenger())
    now = lambda: datetime.now(cfg.timezone)  # noqa: E731

    stop = threading.Event()
    if not args.no_scheduler:
        sched = Scheduler(uza, cfg, now)
        threading.Thread(target=sched.run_forever, args=(stop,), daemon=True).start()

    try:
        while True:
            line = input("나> ").strip()
            if not line:
                continue
            if line == "/quit":
                break
            if line == "/tick":
                o = uza.proactive_tick(now())
                print(f"[{o.decision}] {o.reason}")
            elif line == "/reflect":
                print(uza.reflect(now()))
            elif line == "/review":
                print(uza.review(now()))
            elif line == "/memory":
                print(uza.describe_memory(now()))
            else:
                uza.handle_user_message(line, now())
    except (EOFError, KeyboardInterrupt):
        pass
    finally:
        stop.set()


if __name__ == "__main__":
    main()
