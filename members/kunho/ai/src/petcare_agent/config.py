"""Runtime configuration for the PetCare-AI assessment graph."""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class PetCareSettings(BaseSettings):
    """Environment-backed settings fixed by the Phase 0 contract."""

    openai_api_key: str | None = Field(default=None, alias="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-5.4-mini", alias="OPENAI_MODEL")
    langsmith_api_key: str | None = Field(default=None, alias="LANGSMITH_API_KEY")
    langsmith_tracing: bool = Field(default=False, alias="LANGSMITH_TRACING")
    petcare_api_base_url: str = Field(
        default="http://localhost:8000",
        alias="PETCARE_API_BASE_URL",
    )

    langsmith_project: str = Field(
        default="petcare-ai-assessment",
        alias="LANGSMITH_PROJECT",
    )
    langsmith_run_prefix: str = Field(
        default="assessment_graph",
        alias="LANGSMITH_RUN_PREFIX",
    )
    environment: Literal["local", "test", "staging", "production"] = Field(
        default="local",
        alias="PETCARE_ENV",
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
        case_sensitive=False,
    )


@lru_cache
def get_settings() -> PetCareSettings:
    """Return cached process settings."""

    return PetCareSettings()
