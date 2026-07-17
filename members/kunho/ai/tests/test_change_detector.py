from __future__ import annotations

from petcare_agent.nodes.change_detector import detect_changes
from petcare_agent.schemas.graph_state import (
    BaselineContext,
    BaselineSummary,
    CurrentStatus,
    PetCareGraphState,
)


def _state_with_baseline(
    *,
    symptoms: list[str] | None = None,
    appetite: str = "normal",
    water: str = "normal",
    activity: str = "normal",
    current_status: CurrentStatus | None = None,
) -> PetCareGraphState:
    return PetCareGraphState(
        baseline_context=BaselineContext(
            window_days=3,
            baseline_available=True,
            baseline_summary=BaselineSummary(
                appetite=appetite,  # type: ignore[arg-type]
                water=water,  # type: ignore[arg-type]
                activity=activity,  # type: ignore[arg-type]
                stool="normal",
                vomit="none",
                symptoms=symptoms or [],
            ),
        ),
        current_status=current_status or CurrentStatus(),
    )


def test_change_detector_finds_new_symptoms_not_in_baseline() -> None:
    state = _state_with_baseline(
        symptoms=[],
        current_status=CurrentStatus(
            symptoms=["coughing"],
            appetite="normal",
            water="normal",
            activity="normal",
        ),
    )

    result = detect_changes(state)

    assert result.change_detection.new_symptoms == ["coughing"]
    assert result.change_detection.baseline_deviation is True


def test_change_detector_marks_decreased_appetite_as_worsened() -> None:
    state = _state_with_baseline(
        appetite="normal",
        current_status=CurrentStatus(
            symptoms=[],
            appetite="decreased",
            water="normal",
            activity="normal",
        ),
    )

    result = detect_changes(state)

    assert "appetite" in result.change_detection.worsened_fields
    assert result.change_detection.baseline_deviation is True


def test_change_detector_marks_unchanged_fields_without_baseline_deviation() -> None:
    state = _state_with_baseline(
        symptoms=["sneezing"],
        current_status=CurrentStatus(
            symptoms=["sneezing"],
            appetite="normal",
            water="normal",
            activity="normal",
        ),
    )

    result = detect_changes(state)

    assert set(result.change_detection.unchanged_fields) == {
        "appetite",
        "water",
        "activity",
    }
    assert result.change_detection.new_symptoms == []
    assert result.change_detection.baseline_deviation is False
