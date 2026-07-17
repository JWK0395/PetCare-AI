from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from pydantic import BaseModel, ConfigDict

from petcare_agent.config import PetCareSettings
from petcare_agent.llm.client import LLMClient, LLMMessage


class SimpleOutput(BaseModel):
    value: str

    model_config = ConfigDict(extra="forbid")


class FakeProvider:
    def __init__(self, payload: Any = None, error: Exception | None = None) -> None:
        self.payload = payload
        self.error = error
        self.calls = 0
        self.last_model: str | None = None
        self.last_messages: Sequence[LLMMessage] | None = None

    def structured_output(
        self,
        *,
        model: str,
        messages: Sequence[LLMMessage],
        output_model: type[BaseModel],
    ) -> Any:
        self.calls += 1
        self.last_model = model
        self.last_messages = messages
        if self.error is not None:
            raise self.error
        return self.payload


def test_openai_model_default(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_MODEL", raising=False)

    settings = PetCareSettings(_env_file=None)

    assert settings.openai_model == "gpt-5.4-mini"


def test_openai_model_override(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_MODEL", "gpt-5.4")

    settings = PetCareSettings(_env_file=None)

    assert settings.openai_model == "gpt-5.4"


def test_structured_output_success_with_mock_provider() -> None:
    provider = FakeProvider(payload={"value": "ok"})
    client = LLMClient(
        settings=PetCareSettings(_env_file=None, OPENAI_MODEL="mock-model"),
        provider=provider,
    )

    result = client.structured_output(
        system_prompt="system",
        user_prompt="user",
        output_model=SimpleOutput,
        fallback=SimpleOutput(value="fallback"),
    )

    assert result.value == "ok"
    assert provider.calls == 1
    assert provider.last_model == "mock-model"
    assert provider.last_messages == [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "user"},
    ]


def test_invalid_structured_output_returns_fallback() -> None:
    provider = FakeProvider(payload={"unexpected": "field"})
    client = LLMClient(settings=PetCareSettings(_env_file=None), provider=provider)

    result = client.structured_output(
        system_prompt="system",
        user_prompt="user",
        output_model=SimpleOutput,
        fallback=SimpleOutput(value="fallback"),
    )

    assert result.value == "fallback"


def test_provider_error_returns_fallback() -> None:
    provider = FakeProvider(error=RuntimeError("provider failed"))
    client = LLMClient(settings=PetCareSettings(_env_file=None), provider=provider)

    result = client.structured_output(
        system_prompt="system",
        user_prompt="user",
        output_model=SimpleOutput,
        fallback=SimpleOutput(value="fallback"),
    )

    assert result.value == "fallback"
    assert provider.calls == 1
