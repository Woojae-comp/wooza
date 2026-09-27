"""콘솔에서 UZA와 대화하기.

    python -m uza            # 대화 + 백그라운드 스케줄러
    python -m uza --no-scheduler
    python -m uza --env ../other-project/.env   # 다른 프로젝트의 .env 사용

API 키는 .env의 ANTHROPIC_API_KEY에서 읽는다. 이미 설정된 환경 변수가 우선한다.

대화 중 명령: /tick (선제 판단 즉시 실행), /reflect, /review, /memory, /quit
"""

from __future__ import annotations

import argparse
import json
import logging
import threading
from datetime import datetime

from dotenv import load_dotenv

from .config import Config
from .engine import Uza
from .llm import ClaudeLLM
from .prompts import load_core
from .scheduler import Scheduler
from .store import Store


def _print_result(name: str, result: dict | None) -> None:
    if result is None:
        # 정리할 대화가 없거나 모델 호출이 실패하면 엔진은 None을 돌려준다.
        print(f"[{name}] 결과 없음 (정리할 대화가 없거나 모델 호출 실패, -v로 로그 확인)")
    else:
        print(json.dumps(result, ensure_ascii=False, indent=2))


class ConsoleMessenger:
    def __init__(self) -> None:
        self._lock = threading.Lock()

    def send(self, text: str) -> None:
        with self._lock:
            print(f"\nUZA> {text}\n", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(prog="uza")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--env", default=".env", help="API 키를 읽을 .env 파일 경로")
    parser.add_argument("--no-scheduler", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)
    if not load_dotenv(args.env):
        logging.warning("%s 를 찾지 못했다. 환경 변수에서 API 키를 찾는다.", args.env)
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
                _print_result("reflect", uza.reflect(now()))
            elif line == "/review":
                _print_result("review", uza.review(now()))
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
