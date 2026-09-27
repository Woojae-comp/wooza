from __future__ import annotations

from datetime import datetime
from typing import Callable

import pytest

from uza.config import Config
from uza.engine import Uza
from uza.store import Store

TZ = Config().timezone


def at(y: int, mo: int, d: int, h: int = 0, mi: int = 0) -> datetime:
    return datetime(y, mo, d, h, mi, tzinfo=TZ)


class FakeLLM:
    """용도별 응답을 함수나 dict로 지정한다. 받은 맥락은 calls에 남는다."""

    def __init__(self) -> None:
        self.handlers: dict[str, Callable[[dict], dict] | dict] = {}
        self.calls: list[tuple[str, dict]] = []

    def on(self, purpose: str, handler: Callable[[dict], dict] | dict) -> None:
        self.handlers[purpose] = handler

    def complete(self, purpose: str, context: dict) -> dict:
        self.calls.append((purpose, context))
        h = self.handlers.get(purpose)
        if h is None:
            raise AssertionError(f"예상하지 못한 호출: {purpose}")
        return h(context) if callable(h) else h

    def contexts(self, purpose: str) -> list[dict]:
        return [c for p, c in self.calls if p == purpose]


class FakeMessenger:
    def __init__(self) -> None:
        self.sent: list[str] = []

    def send(self, text: str) -> None:
        self.sent.append(text)


@pytest.fixture
def cfg() -> Config:
    return Config()


@pytest.fixture
def store() -> Store:
    return Store(":memory:")


@pytest.fixture
def llm() -> FakeLLM:
    return FakeLLM()


@pytest.fixture
def messenger() -> FakeMessenger:
    return FakeMessenger()


@pytest.fixture
def uza(cfg, store, llm, messenger) -> Uza:
    return Uza(cfg, store, llm, messenger)
