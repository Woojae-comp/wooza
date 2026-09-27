"""모델 호출. 엔진은 LLM 프로토콜만 알고, Claude 구현은 여기에 둔다."""

from __future__ import annotations

import json
from typing import Protocol

import anthropic

from .prompts import INSTRUCTIONS, system_prompt


class LLMError(RuntimeError):
    pass


class LLM(Protocol):
    def complete(self, purpose: str, context: dict) -> dict:
        """purpose: proactive | chat | reflection | review. JSON 객체를 돌려준다."""
        ...


class ClaudeLLM:
    def __init__(
        self,
        core: str,
        model: str = "claude-opus-5",
        effort: dict[str, str] | None = None,
        client: anthropic.Anthropic | None = None,
    ):
        self.core = core
        self.model = model
        self.effort = effort or {}
        self.client = client or anthropic.Anthropic()

    def complete(self, purpose: str, context: dict) -> dict:
        _, schema = INSTRUCTIONS[purpose]
        output_config: dict = {"format": {"type": "json_schema", "schema": schema}}
        if purpose in self.effort:
            output_config["effort"] = self.effort[purpose]
        try:
            response = self.client.beta.messages.create(
                model=self.model,
                max_tokens=16000,
                # 시스템 프롬프트는 용도별로 고정이므로 캐시한다. 맥락은 매번 바뀌므로 user 쪽에 둔다.
                system=[
                    {
                        "type": "text",
                        "text": system_prompt(self.core, purpose),
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
                messages=[
                    {"role": "user", "content": json.dumps(context, ensure_ascii=False, indent=1)}
                ],
                output_config=output_config,
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
            )
        except anthropic.RateLimitError as e:
            raise LLMError(f"rate_limited: {e}") from e
        except anthropic.APIStatusError as e:
            raise LLMError(f"api_error {e.status_code}: {e}") from e
        except anthropic.APIConnectionError as e:
            raise LLMError(f"connection_error: {e}") from e

        if response.stop_reason == "refusal":
            raise LLMError("refusal")
        if response.stop_reason == "max_tokens":
            raise LLMError("max_tokens")
        text = "".join(b.text for b in response.content if b.type == "text")
        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            raise LLMError(f"invalid_json: {text[:200]!r}") from e
        if not isinstance(data, dict):
            raise LLMError("json_not_object")
        return data
