from __future__ import annotations

from petcare_agent.nodes.baseline_builder import build_baseline_context
from petcare_agent.schemas.graph_state import PetCareContext, PetCareGraphState


def test_baseline_builder_marks_unavailable_when_no_recent_entries() -> None:
    state = PetCareGraphState(context=PetCareContext(recent_daily_entries=[]))

    result = build_baseline_context(state)

    assert result.baseline_context.baseline_available is False
    assert "recent_daily_entries" in result.baseline_context.missing_baseline_fields


def test_baseline_builder_summarizes_recent_daily_entries() -> None:
    state = PetCareGraphState(
        context=PetCareContext(
            recent_daily_entries=[
                {
                    "record_date": "2026-07-15",
                    "food": "normal",
                    "water": "normal",
                    "activity": "normal",
                    "stool": "normal",
                    "vomit": "none",
                    "symptom": "sneezing",
                },
                {
                    "record_date": "2026-07-14",
                    "food": "normal",
                    "water": "normal",
                    "activity": "normal",
                    "stool": "normal",
                    "vomit": "none",
                    "symptom": "none",
                },
                {
                    "record_date": "2026-07-13",
                    "food": "normal",
                    "water": "normal",
                    "activity": "normal",
                    "stool": "normal",
                    "vomit": "none",
                    "symptom": "none",
                },
            ]
        )
    )

    result = build_baseline_context(state)
    summary = result.baseline_context.baseline_summary

    assert result.baseline_context.baseline_available is True
    assert summary.appetite == "normal"
    assert summary.water == "normal"
    assert summary.activity == "normal"
    assert summary.stool == "normal"
    assert summary.vomit == "none"
    assert summary.symptoms == ["sneezing"]


def test_baseline_builder_records_missing_baseline_fields() -> None:
    state = PetCareGraphState(
        context=PetCareContext(
            recent_daily_entries=[
                {
                    "record_date": "2026-07-15",
                    "food": "normal",
                    "activity": "normal",
                    "stool": "normal",
                    "vomit": "none",
                    "symptom": "none",
                },
                {
                    "record_date": "2026-07-14",
                    "food": "normal",
                    "activity": "normal",
                    "stool": "normal",
                    "vomit": "none",
                    "symptom": "none",
                },
                {
                    "record_date": "2026-07-13",
                    "food": "normal",
                    "activity": "normal",
                    "stool": "normal",
                    "vomit": "none",
                    "symptom": "none",
                },
            ]
        )
    )

    result = build_baseline_context(state)

    assert result.baseline_context.baseline_available is False
    assert "water" in result.baseline_context.missing_baseline_fields
