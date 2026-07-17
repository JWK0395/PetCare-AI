"""OpenAI-backed structured output helper for Phase 5 nodes."""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any, Protocol, TypeVar

from pydantic import BaseModel, ValidationError

from petcare_agent.config import PetCareSettings, get_settings

TOutput = TypeVar("TOutput", bound=BaseModel)
LLMMessage = dict[str, str]


class LLMClientError(RuntimeError):
    """Raised when structured output cannot be produced and no fallback exists."""


class LLMProviderError(LLMClientError):
    """Raised when the configured provider cannot complete a request."""


class StructuredOutputProvider(Protocol):
    """Provider boundary used by tests and the OpenAI wrapper."""

    def structured_output(
        self,
        *,
        model: str,
        messages: Sequence[LLMMessage],
        output_model: type[TOutput],
    ) -> Any:
        """Return provider output for a pydantic schema."""


class StructuredOutputClient(Protocol):
    """Client protocol accepted by Phase 5 nodes."""

    def structured_output(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        output_model: type[TOutput],
        fallback: TOutput | None = None,
    ) -> TOutput:
        """Return validated structured output or the supplied fallback."""


class OpenAIProvider:
    """OpenAI provider wrapper.

    The OpenAI client is created lazily so unit tests can instantiate the
    wrapper without network calls or a real API key.
    """

    def __init__(self, settings: PetCareSettings | None = None) -> None:
        self.settings = settings or get_settings()
        self._client: Any | None = None

    def structured_output(
        self,
        *,
        model: str,
        messages: Sequence[LLMMessage],
        output_model: type[TOutput],
    ) -> Any:
        """Call OpenAI structured output and return the parsed payload."""

        if not self.settings.openai_api_key:
            raise LLMProviderError("OPENAI_API_KEY is not configured")

        try:
            response = self._get_client().beta.chat.completions.parse(
                model=model,
                messages=list(messages),
                response_format=output_model,
            )
            parsed = response.choices[0].message.parsed
        except Exception as exc:  # pragma: no cover - real provider is mocked in tests.
            raise LLMProviderError("OpenAI structured output call failed") from exc

        if parsed is None:
            raise LLMProviderError("OpenAI response did not include parsed output")
        return parsed

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                from openai import OpenAI
            except Exception as exc:  # pragma: no cover - environment-specific import.
                raise LLMProviderError("openai package is not available") from exc
            self._client = OpenAI(api_key=self.settings.openai_api_key)
        return self._client


class LLMClient:
    """Validate provider output against pydantic structured-output schemas."""

    def __init__(
        self,
        *,
        settings: PetCareSettings | None = None,
        provider: StructuredOutputProvider | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.provider = provider or OpenAIProvider(self.settings)

    @property
    def model(self) -> str:
        return self.settings.openai_model

    def structured_output(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        output_model: type[TOutput],
        fallback: TOutput | None = None,
    ) -> TOutput:
        """Return validated structured output, falling back on failures."""

        messages: list[LLMMessage] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        try:
            raw_output = self.provider.structured_output(
                model=self.model,
                messages=messages,
                output_model=output_model,
            )
            return _coerce_output(raw_output, output_model)
        except (ValidationError, json.JSONDecodeError, TypeError, ValueError) as exc:
            if fallback is not None:
                return fallback.model_copy(deep=True)
            raise LLMClientError("LLM structured output validation failed") from exc
        except Exception as exc:
            if fallback is not None:
                return fallback.model_copy(deep=True)
            raise LLMClientError("LLM structured output provider failed") from exc


def call_structured_output(
    *,
    system_prompt: str,
    user_prompt: str,
    output_model: type[TOutput],
    fallback: TOutput | None = None,
    client: StructuredOutputClient | None = None,
) -> TOutput:
    """Convenience helper used by nodes and tests."""

    llm_client = client or LLMClient()
    return llm_client.structured_output(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        output_model=output_model,
        fallback=fallback,
    )


def _coerce_output(raw_output: Any, output_model: type[TOutput]) -> TOutput:
    if isinstance(raw_output, output_model):
        return raw_output
    if isinstance(raw_output, str):
        return output_model.model_validate(json.loads(raw_output))
    return output_model.model_validate(raw_output)
