"""Build a recent baseline summary from fixture-loaded daily entries."""

from __future__ import annotations

from collections import Counter
import re
from typing import Any, Iterable, Literal

from petcare_agent.schemas.graph_state import (
    BaselineContext,
    BaselineSummary,
    PetCareGraphState,
)

BaselineField = Literal["appetite", "water", "activity", "stool", "vomit"]

FIELD_SOURCE_KEYS: dict[BaselineField, tuple[str, ...]] = {
    "appetite": ("appetite", "food"),
    "water": ("water",),
    "activity": ("activity",),
    "stool": ("stool",),
    "vomit": ("vomit",),
}

STATUS_VALUES = {"normal", "decreased", "increased", "abnormal", "none", "present"}

APPETITE_DECREASED = (
    "decreased",
    "less",
    "reduced",
    "low",
    "poor",
    "half",
    "not eating",
    "ate little",
    "left food",
    "refused food",
    "\uc904",
    "\uac10\uc18c",
    "\uc800\ud558",
    "\uc808\ubc18",
    "\ub0a8\uacbc",
    "\uc548 \uba39",
    "\ubabb \uba39",
    "\uc801\uac8c",
)
APPETITE_INCREASED = (
    "increased",
    "more",
    "high",
    "much",
    "\ub9ce",
    "\ub298",
    "\uc99d\uac00",
)

WATER_DECREASED = (
    "decreased",
    "less",
    "reduced",
    "not drinking",
    "\uc904",
    "\uac10\uc18c",
    "\uc801\uac8c",
    "\uc548 \ub9c8",
    "\ubabb \ub9c8",
)
WATER_INCREASED = (
    "increased",
    "more",
    "excessive",
    "much",
    "\ub9ce",
    "\ub298",
    "\uc99d\uac00",
    "\uacfc\ub2e4",
)

ACTIVITY_DECREASED = (
    "decreased",
    "less active",
    "lethargy",
    "lethargic",
    "low energy",
    "tired",
    "sluggish",
    "\uae30\ub825 \uc800\ud558",
    "\ucc98\uc9d0",
    "\ucc98\uc838",
    "\ubb34\uae30\ub825",
    "\ud65c\ub3d9 \uac10\uc18c",
    "\uc0b0\ucc45 \uac70\ubd80",
)
ACTIVITY_INCREASED = (
    "increased",
    "more active",
    "restless",
    "hyper",
    "\ub9ce",
    "\ub298",
    "\uc99d\uac00",
    "\ud765\ubd84",
    "\uc548\uc808\ubd80\uc808",
)

NORMAL_KEYWORDS = (
    "normal",
    "usual",
    "same",
    "similar",
    "good",
    "routine",
    "\ube44\uc2b7",
    "\ud3c9\uc18c",
    "\uc815\uc0c1",
)

STOOL_NORMAL = (
    "normal",
    "formed",
    "solid",
    "no diarrhea",
    "\uc815\uc0c1",
    "\uc124\uc0ac \uc5c6",
)
STOOL_ABNORMAL = (
    "abnormal",
    "diarrhea",
    "loose",
    "watery",
    "bloody",
    "blood",
    "constipation",
    "\uc124\uc0ac",
    "\ud608\ubcc0",
    "\ubb3d",
    "\ubb34\ub978",
    "\ubcc0\ube44",
    "\uc774\uc0c1",
)

VOMIT_NONE = (
    "none",
    "no vomiting",
    "no vomit",
    "did not vomit",
    "normal",
    "\uc5c6",
    "\uc548 \ud568",
    "\ud1a0\ud558\uc9c0",
)
VOMIT_PRESENT = (
    "present",
    "vomit",
    "vomiting",
    "vomited",
    "threw up",
    "\uad6c\ud1a0",
    "\ud1a0",
)

SYMPTOM_NONE = {
    "",
    "none",
    "no symptoms",
    "normal",
    "nothing unusual",
    "\uc5c6\uc74c",
    "\uc99d\uc0c1 \uc5c6\uc74c",
}

SYMPTOM_CANONICAL: tuple[tuple[tuple[str, ...], str], ...] = (
    (("cough", "coughing", "\uae30\uce68"), "coughing"),
    (("vomit", "vomiting", "\uad6c\ud1a0", "\ud1a0"), "vomiting"),
    (("diarrhea", "\uc124\uc0ac"), "diarrhea"),
    (("lethargy", "lethargic", "\uae30\ub825 \uc800\ud558", "\ubb34\uae30\ub825"), "lethargy"),
    (("sneeze", "sneezing", "\uc7ac\ucc44\uae30"), "sneezing"),
    (("itch", "itching", "\uac00\ub824"), "itching"),
    (("limp", "limping", "\uc808\ub6b1", "\ud30c\ud589"), "limping"),
)


def build_baseline_context(
    state: PetCareGraphState,
    *,
    window_days: int = 3,
) -> PetCareGraphState:
    """Summarize recent daily entries into the graph state's baseline context."""

    next_state = state.model_copy(deep=True)
    entries = _recent_entries(next_state.context.recent_daily_entries, window_days)

    summary = BaselineSummary(
        appetite=_summarize_field(entries, "appetite"),
        water=_summarize_field(entries, "water"),
        activity=_summarize_field(entries, "activity"),
        stool=_summarize_field(entries, "stool"),
        vomit=_summarize_field(entries, "vomit"),
        symptoms=_summarize_symptoms(entries),
    )

    missing_fields = _missing_baseline_fields(
        entries=entries,
        total_entry_count=len(next_state.context.recent_daily_entries),
        summary=summary,
        window_days=window_days,
    )
    next_state.baseline_context = BaselineContext(
        window_days=window_days,
        baseline_available=bool(entries) and not missing_fields,
        baseline_summary=summary,
        missing_baseline_fields=missing_fields,
    )
    return next_state


def baseline_builder(
    state: PetCareGraphState,
    *,
    window_days: int = 3,
) -> PetCareGraphState:
    """LangGraph-friendly alias for the baseline builder node."""

    return build_baseline_context(state, window_days=window_days)


def _recent_entries(entries: list[dict[str, Any]], window_days: int) -> list[dict[str, Any]]:
    ordered_entries = sorted(
        entries,
        key=lambda entry: str(entry.get("record_date") or ""),
        reverse=True,
    )
    return ordered_entries[:window_days]


def _missing_baseline_fields(
    *,
    entries: list[dict[str, Any]],
    total_entry_count: int,
    summary: BaselineSummary,
    window_days: int,
) -> list[str]:
    missing_fields: list[str] = []
    if total_entry_count < window_days:
        missing_fields.append("recent_daily_entries")

    for field in FIELD_SOURCE_KEYS:
        if getattr(summary, field) == "unknown":
            missing_fields.append(field)

    if entries and not _has_any_source(entries, ("symptom", "symptoms")):
        missing_fields.append("symptoms")

    return missing_fields


def _summarize_field(entries: list[dict[str, Any]], field: BaselineField) -> str:
    values = [
        value
        for value in (_classify_field(field, _first_text(entry, FIELD_SOURCE_KEYS[field])) for entry in entries)
        if value != "unknown"
    ]
    if not values:
        return "unknown"

    counts = Counter(values)
    highest_count = max(counts.values())
    for value in values:
        if counts[value] == highest_count:
            return value
    return "unknown"


def _classify_field(field: BaselineField, text: str | None) -> str:
    if text is None:
        return "unknown"

    normalized = _normalize_text(text)
    if normalized in STATUS_VALUES:
        return _direct_status(field, normalized)

    if field == "appetite":
        return _classify_appetite(normalized)
    if field == "water":
        return _classify_water(normalized)
    if field == "activity":
        return _classify_activity(normalized)
    if field == "stool":
        return _classify_stool(normalized)
    if field == "vomit":
        return _classify_vomit(normalized)
    return "unknown"


def _direct_status(field: BaselineField, status: str) -> str:
    if field in {"appetite", "water", "activity"} and status in {
        "normal",
        "decreased",
        "increased",
    }:
        return status
    if field == "stool" and status in {"normal", "abnormal"}:
        return status
    if field == "vomit" and status in {"none", "present"}:
        return status
    return "unknown"


def _classify_appetite(text: str) -> str:
    if _contains_any(text, APPETITE_DECREASED):
        return "decreased"
    if _contains_any(text, APPETITE_INCREASED):
        return "increased"
    if _contains_any(text, NORMAL_KEYWORDS):
        return "normal"
    return "unknown"


def _classify_water(text: str) -> str:
    if _contains_any(text, WATER_DECREASED):
        return "decreased"
    if _contains_any(text, WATER_INCREASED):
        return "increased"
    if _contains_any(text, NORMAL_KEYWORDS):
        return "normal"
    return "unknown"


def _classify_activity(text: str) -> str:
    if _contains_any(text, ACTIVITY_DECREASED):
        return "decreased"
    if _contains_any(text, ACTIVITY_INCREASED):
        return "increased"
    if _contains_any(text, NORMAL_KEYWORDS):
        return "normal"
    return "unknown"


def _classify_stool(text: str) -> str:
    if _contains_any(text, STOOL_NORMAL):
        return "normal"
    if _contains_any(text, STOOL_ABNORMAL):
        return "abnormal"
    return "unknown"


def _classify_vomit(text: str) -> str:
    if _contains_any(text, VOMIT_NONE):
        return "none"
    if _contains_any(text, VOMIT_PRESENT):
        return "present"
    return "unknown"


def _summarize_symptoms(entries: list[dict[str, Any]]) -> list[str]:
    symptoms: list[str] = []
    seen: set[str] = set()
    for entry in entries:
        for symptom in _entry_symptoms(entry):
            if symptom not in seen:
                symptoms.append(symptom)
                seen.add(symptom)
    return symptoms


def _entry_symptoms(entry: dict[str, Any]) -> list[str]:
    value = entry.get("symptoms", entry.get("symptom"))
    if isinstance(value, list):
        raw_symptoms = [str(item) for item in value]
    elif value is None:
        return []
    else:
        raw_symptoms = re.split(r"[,;/]|\band\b", str(value), flags=re.IGNORECASE)

    symptoms: list[str] = []
    for raw_symptom in raw_symptoms:
        normalized = _normalize_text(raw_symptom)
        if normalized in SYMPTOM_NONE:
            continue
        canonical = _canonical_symptom(normalized)
        if canonical:
            symptoms.append(canonical)
    return symptoms


def _canonical_symptom(text: str) -> str:
    for keywords, canonical in SYMPTOM_CANONICAL:
        if _contains_any(text, keywords):
            return canonical
    return text


def _first_text(entry: dict[str, Any], keys: Iterable[str]) -> str | None:
    for key in keys:
        value = entry.get(key)
        if value is not None and str(value).strip():
            return str(value)
    return None


def _has_any_source(entries: list[dict[str, Any]], keys: Iterable[str]) -> bool:
    return any(any(key in entry for key in keys) for entry in entries)


def _contains_any(text: str, needles: Iterable[str]) -> bool:
    return any(needle in text for needle in needles)


def _normalize_text(value: str) -> str:
    return " ".join(value.strip().lower().split())
