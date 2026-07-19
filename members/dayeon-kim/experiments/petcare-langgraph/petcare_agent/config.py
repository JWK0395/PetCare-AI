from __future__ import annotations

import os
from dataclasses import dataclass
from getpass import getpass


@dataclass(frozen=True)
class Settings:
    openai_api_key: str
    openai_model: str = "gpt-5.4-mini"

    @classmethod
    def from_env(
        cls,
        *,
        prompt_if_missing: bool = True,
    ) -> "Settings":
        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        model = os.getenv(
            "OPENAI_MODEL",
            "gpt-5.4-mini",
        ).strip()

        if not api_key and prompt_if_missing:
            api_key = getpass(
                "OPENAI_API_KEY를 입력하세요: "
            ).strip()

        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY가 필요합니다. "
                "환경변수로 설정하거나 실행 시 입력해 주세요."
            )

        return cls(
            openai_api_key=api_key,
            openai_model=model,
        )
