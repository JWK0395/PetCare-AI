from __future__ import annotations

import time
from typing import Any

from ..models import BackendContextPayload, PetCareState
from ..utils import add_error, node_result


def summarize_daily_entries(context: dict[str, Any]) -> str:
    entries = context.get("daily_entries", [])
    if not entries:
        return "최근 일기 기록 없음"

    sorted_entries = sorted(
        entries,
        key=lambda item: item.get("record_date", ""),
    )

    lines: list[str] = []
    for entry in sorted_entries:
        values = [
            entry.get("food"),
            entry.get("water"),
            entry.get("activity"),
            entry.get("symptom"),
            entry.get("stool"),
            entry.get("vomit"),
            entry.get("notes"),
        ]
        details = " / ".join(str(value) for value in values if value)
        lines.append(
            f"{entry.get('record_date', '날짜 미상')}: {details}"
        )

    return "\n".join(lines)

def summarize_diagnoses(context: dict[str, Any]) -> str:
    diagnoses = context.get("diagnoses", [])
    if not diagnoses:
        return "등록된 진단서 없음"

    sorted_diagnoses = sorted(
        diagnoses,
        key=lambda item: item.get("date", ""),
    )

    lines: list[str] = []
    for item in sorted_diagnoses:
        lines.append(
            f"{item.get('date', '날짜 미상')} | "
            f"{item.get('hospital', '병원 미상')} | "
            f"{item.get('diagnosis', '진단명 미상')} | "
            f"{item.get('content', '')}"
        )

    return "\n".join(lines)

def prepare_backend_context(state: PetCareState) -> dict[str, Any]:
    started = time.perf_counter()

    try:
        context_model = BackendContextPayload.model_validate(
            state["backend_context"]
        )
        context = context_model.model_dump()

        return node_result(
            state,
            node_name="prepare_backend_context",
            started_at=started,
            updates={
                "backend_context": context,
                "diary_summary": summarize_daily_entries(context),
                "diagnosis_summary": summarize_diagnoses(context),
            },
        )
    except Exception as error:
        return add_error(
            state,
            node_name="prepare_backend_context",
            error=error,
            started_at=started,
        )
