from __future__ import annotations

import time
from typing import Any, Literal

from .models import PetCareState


MAX_CONVERSATION_MESSAGES = 20
MAX_PROMPT_MESSAGES = 8
MAX_PROMPT_HISTORY_CHARS = 6000


def trim_conversation_history(
    history: list[dict[str, str]],
    *,
    max_messages: int = MAX_CONVERSATION_MESSAGES,
) -> list[dict[str, str]]:
    return history[-max_messages:]


def append_conversation_message(
    state: PetCareState,
    *,
    role: Literal["user", "assistant"],
    content: str,
) -> list[dict[str, str]]:
    history = list(state.get("conversation_history", []))
    history.append(
        {
            "role": role,
            "content": content.strip(),
        }
    )
    return trim_conversation_history(history)


def format_conversation_history(
    history: list[dict[str, str]],
    *,
    exclude_last_user_message: bool = False,
    max_messages: int = MAX_PROMPT_MESSAGES,
    max_chars: int = MAX_PROMPT_HISTORY_CHARS,
) -> str:
    messages = list(history)

    if (
        exclude_last_user_message
        and messages
        and messages[-1].get("role") == "user"
    ):
        messages = messages[:-1]

    messages = messages[-max_messages:]

    if not messages:
        return "이전 대화 없음"

    lines: list[str] = []

    for item in messages:
        role = item.get("role", "unknown")
        label = "사용자" if role == "user" else "Assistant"
        lines.append(
            f"{label}: {item.get('content', '')}"
        )

    text = "\n".join(lines)

    if len(text) > max_chars:
        text = text[-max_chars:]
        first_newline = text.find("\n")
        if first_newline >= 0:
            text = text[first_newline + 1 :]

    return text


def node_result(
    state: PetCareState,
    *,
    node_name: str,
    started_at: float,
    updates: dict[str, Any],
) -> dict[str, Any]:
    latency = dict(state.get("latency_ms", {}))
    latency[node_name] = round(
        (time.perf_counter() - started_at) * 1000,
        2,
    )

    return {
        **updates,
        "latency_ms": latency,
    }


def add_warning(
    state: PetCareState,
    *,
    node_name: str,
    warning: str,
    started_at: float,
    updates: dict[str, Any] | None = None,
) -> dict[str, Any]:
    warnings = list(state.get("warnings", []))
    warnings.append(
        f"{node_name}: {warning}"
    )

    return node_result(
        state,
        node_name=node_name,
        started_at=started_at,
        updates={
            **(updates or {}),
            "warnings": warnings,
        },
    )


def add_error(
    state: PetCareState,
    *,
    node_name: str,
    error: Exception,
    started_at: float,
) -> dict[str, Any]:
    errors = list(state.get("errors", []))
    errors.append(
        f"{node_name}: {type(error).__name__}: {error}"
    )

    safe_answer = (
        "처리 중 오류가 발생했습니다. "
        "잠시 후 다시 시도해 주세요."
    )

    return node_result(
        state,
        node_name=node_name,
        started_at=started_at,
        updates={
            "errors": errors,
            "answer": safe_answer,
            "conversation_history": (
                append_conversation_message(
                    state,
                    role="assistant",
                    content=safe_answer,
                )
            ),
        },
    )
