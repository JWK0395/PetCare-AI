"""Rule-based current status to baseline comparison."""

from __future__ import annotations

from petcare_agent.schemas.graph_state import ChangeDetection, PetCareGraphState

COMPARABLE_FIELDS = ("appetite", "water", "activity")

FIELD_SEVERITY: dict[str, dict[str, int]] = {
    "appetite": {"normal": 0, "increased": 1, "decreased": 2},
    "water": {"normal": 0, "decreased": 1, "increased": 2},
    "activity": {"normal": 0, "increased": 1, "decreased": 2},
}


def detect_changes(state: PetCareGraphState) -> PetCareGraphState:
    """Compare current structured status against the recent baseline."""

    next_state = state.model_copy(deep=True)
    baseline_context = next_state.baseline_context
    baseline_summary = baseline_context.baseline_summary

    if not baseline_context.baseline_available:
        next_state.change_detection = ChangeDetection(
            baseline_available=False,
            baseline_deviation=False,
            summary="Baseline is unavailable, so current status was not compared.",
        )
        return next_state

    new_symptoms = _new_symptoms(
        current_symptoms=next_state.current_status.symptoms,
        baseline_symptoms=baseline_summary.symptoms,
    )

    worsened_fields: list[str] = []
    improved_fields: list[str] = []
    unchanged_fields: list[str] = []
    for field in COMPARABLE_FIELDS:
        baseline_value = getattr(baseline_summary, field)
        current_value = getattr(next_state.current_status, field)
        comparison = _compare_field(field, baseline_value, current_value)
        if comparison == "worsened":
            worsened_fields.append(field)
        elif comparison == "improved":
            improved_fields.append(field)
        elif comparison == "unchanged":
            unchanged_fields.append(field)

    baseline_deviation = bool(new_symptoms or worsened_fields or improved_fields)
    next_state.change_detection = ChangeDetection(
        baseline_available=True,
        new_symptoms=new_symptoms,
        worsened_fields=worsened_fields,
        improved_fields=improved_fields,
        unchanged_fields=unchanged_fields,
        baseline_deviation=baseline_deviation,
        summary=_build_summary(
            new_symptoms=new_symptoms,
            worsened_fields=worsened_fields,
            improved_fields=improved_fields,
            unchanged_fields=unchanged_fields,
        ),
    )
    return next_state


def change_detector(state: PetCareGraphState) -> PetCareGraphState:
    """LangGraph-friendly alias for the change detector node."""

    return detect_changes(state)


def _compare_field(field: str, baseline_value: str, current_value: str) -> str:
    if baseline_value == "unknown" or current_value == "unknown":
        return "unknown"
    if baseline_value == current_value:
        return "unchanged"

    severity = FIELD_SEVERITY[field]
    baseline_score = severity.get(baseline_value)
    current_score = severity.get(current_value)
    if baseline_score is None or current_score is None:
        return "unknown"
    if current_score > baseline_score:
        return "worsened"
    if current_score < baseline_score:
        return "improved"
    return "worsened"


def _new_symptoms(
    *,
    current_symptoms: list[str],
    baseline_symptoms: list[str],
) -> list[str]:
    baseline_normalized = {_normalize_symptom(symptom) for symptom in baseline_symptoms}
    new_symptoms: list[str] = []
    seen: set[str] = set()
    for symptom in current_symptoms:
        normalized = _normalize_symptom(symptom)
        if not normalized or normalized in baseline_normalized or normalized in seen:
            continue
        new_symptoms.append(normalized)
        seen.add(normalized)
    return new_symptoms


def _normalize_symptom(symptom: str) -> str:
    return " ".join(symptom.strip().lower().split())


def _build_summary(
    *,
    new_symptoms: list[str],
    worsened_fields: list[str],
    improved_fields: list[str],
    unchanged_fields: list[str],
) -> str:
    parts: list[str] = []
    if new_symptoms:
        parts.append(f"New symptoms reported: {', '.join(new_symptoms)}.")
    if worsened_fields:
        parts.append(f"Worsened compared with baseline: {', '.join(worsened_fields)}.")
    if improved_fields:
        parts.append(f"Improved compared with baseline: {', '.join(improved_fields)}.")
    if unchanged_fields:
        parts.append(f"Unchanged fields: {', '.join(unchanged_fields)}.")
    if not parts:
        return "No baseline deviation detected."
    return " ".join(parts)
