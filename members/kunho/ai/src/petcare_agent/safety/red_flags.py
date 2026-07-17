"""Canonical red-flag names shared by internal triage and handoff JSON."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from petcare_agent.schemas.handoff import RedFlagInputs, RedFlagSummary, TriStateBool
from petcare_agent.schemas.triage import ChecklistItem

CANONICAL_RED_FLAG_IDS = (
    "open_mouth_breathing",
    "labored_breathing",
    "gum_color_abnormal",
    "collapse_or_fainting",
    "seizure",
    "severe_bleeding",
    "toxin_exposure_suspected",
)

RED_FLAG_ALIASES = {
    "open_mouth_breathing": "open_mouth_breathing",
    "labored_breathing": "labored_breathing",
    "gum_color_abnormal": "gum_color_abnormal",
    "collapse_or_fainting": "collapse_or_fainting",
    "active_seizure": "seizure",
    "seizure_over_5_min": "seizure",
    "repeated_seizures": "seizure",
    "severe_bleeding": "severe_bleeding",
    "known_toxin_ingestion": "toxin_exposure_suspected",
    "unknown_substance_ingestion": "toxin_exposure_suspected",
    "suspected_toxin": "toxin_exposure_suspected",
}

RED_FLAG_LABELS = {
    "open_mouth_breathing": "Open-mouth breathing",
    "labored_breathing": "Labored breathing",
    "gum_color_abnormal": "Pale, blue, purple, or gray gums or tongue",
    "collapse_or_fainting": "Collapse or fainting",
    "seizure": "Seizure",
    "severe_bleeding": "Severe bleeding",
    "toxin_exposure_suspected": "Suspected toxin exposure",
}


FORBIDDEN_HANDOFF_FIELDS = {
    "risk_level",
    "confidence",
    "missing_items",
    "triggered_rules",
    "decision_basis",
    "sources",
    "attachments",
}


def canonical_red_flag_id(item_id: str) -> str | None:
    """Return the canonical handoff/internal red-flag id for a checklist item."""

    return RED_FLAG_ALIASES.get(item_id)


def build_red_flag_inputs(items: Mapping[str, ChecklistItem]) -> RedFlagInputs:
    """Collapse checklist item values into the internal canonical red-flag shape."""

    values: dict[str, TriStateBool] = {
        red_flag_id: "unknown" for red_flag_id in CANONICAL_RED_FLAG_IDS
    }
    seen: set[str] = set()

    for item_id, item in items.items():
        canonical_id = canonical_red_flag_id(item_id)
        if canonical_id is None:
            continue
        seen.add(canonical_id)
        values[canonical_id] = _merge_tri_state(values[canonical_id], tri_state_from_item(item))

    for red_flag_id in seen:
        if values[red_flag_id] == "unknown":
            continue
        values[red_flag_id] = bool(values[red_flag_id])

    return RedFlagInputs.model_validate(values)


def build_red_flag_summaries(items: Mapping[str, ChecklistItem]) -> list[RedFlagSummary]:
    """Build veterinarian-facing red-flag rows from true checklist items."""

    summaries: list[RedFlagSummary] = []
    seen: set[str] = set()
    for item_id, item in items.items():
        canonical_id = canonical_red_flag_id(item_id)
        if canonical_id is None or canonical_id in seen:
            continue
        if tri_state_from_item(item) is not True:
            continue
        label = RED_FLAG_LABELS.get(canonical_id, item.label)
        summaries.append(
            RedFlagSummary(
                name=canonical_id,  # type: ignore[arg-type]
                label=label,
                summary=item.label or label,
            )
        )
        seen.add(canonical_id)
    return summaries


def tri_state_from_item(item: ChecklistItem) -> TriStateBool:
    """Convert checklist item value/confidence to true, false, or unknown."""

    if item.value is True:
        return True
    if item.value is False:
        return False
    if isinstance(item.value, str):
        normalized = item.value.strip().lower()
        if normalized in {"true", "yes", "y"}:
            return True
        if normalized in {"false", "no", "n"}:
            return False
    return "unknown"


def _merge_tri_state(current_value: TriStateBool, item_value: TriStateBool) -> TriStateBool:
    if item_value is True:
        return True
    if current_value is True:
        return True
    if item_value is False:
        return False if current_value == "unknown" else current_value
    return current_value


def is_forbidden_handoff_field(key: Any) -> bool:
    return key in FORBIDDEN_HANDOFF_FIELDS


__all__ = [
    "CANONICAL_RED_FLAG_IDS",
    "FORBIDDEN_HANDOFF_FIELDS",
    "RED_FLAG_ALIASES",
    "RED_FLAG_LABELS",
    "build_red_flag_inputs",
    "build_red_flag_summaries",
    "canonical_red_flag_id",
    "is_forbidden_handoff_field",
    "tri_state_from_item",
]
