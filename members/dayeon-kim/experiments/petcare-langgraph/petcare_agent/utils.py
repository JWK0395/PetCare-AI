from __future__ import annotations

import time
from typing import Any, Literal

from .models import PetCareState


MAX_CONVERSATION_MESSAGES = 20


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
) -> str:
    messages = list(history)

    if (
        exclude_last_user_message
        and messages
        and messages[-1].get("role") == "user"
    ):
        messages = messages[:-1]

    if not messages:
        return "이전 대화 없음"

    lines: list[str] = []

    for item in messages:
        role = item.get("role", "unknown")
        label = "사용자" if role == "user" else "Assistant"
        lines.append(
            f"{label}: {item.get('content', '')}"
        )

    return "\n".join(lines)


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
