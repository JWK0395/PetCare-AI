"""Prompt templates used by Phase 5 structured-output nodes."""

from __future__ import annotations

from functools import lru_cache
from importlib import resources

PROMPT_PACKAGE = "petcare_agent.prompts"


@lru_cache(maxsize=None)
def load_prompt(name: str) -> str:
    """Load a packaged markdown prompt template by stem or filename."""

    file_name = name if name.endswith(".md") else f"{name}.md"
    prompt_path = resources.files(PROMPT_PACKAGE).joinpath(file_name)
    return prompt_path.read_text(encoding="utf-8").strip()
