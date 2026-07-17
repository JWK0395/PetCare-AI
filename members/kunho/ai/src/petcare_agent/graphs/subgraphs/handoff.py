"""Deterministic hospital handoff helper."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from petcare_agent.localization import (
    display_label,
    display_list,
    localize_change_summary,
    wants_korean,
)
from petcare_agent.safety.red_flags import build_red_flag_summaries
from petcare_agent.schemas.graph_state import PetCareGraphState
from petcare_agent.schemas.handoff import (
    AssociatedSymptomSummary,
    BaselineChange,
    BaselineComparisonSummary,
    ClinicalCourseSummary,
    HospitalHandoffSummary,
    MedicalBackgroundSummary,
    PatientSummary,
    TriageAssessmentSummary,
    VisitReasonSummary,
)

HANDOFF_RISK_LEVELS = {"urgent", "non_emergency", "unknown"}
FIELD_LABELS = {
    "appetite": "Appetite",
    "water": "Water intake",
    "activity": "Activity",
}
FIELD_LABELS_KO = {
    "appetite": "식욕",
    "water": "음수량",
    "activity": "활동량",
}


def should_build_non_emergency_handoff(state: PetCareGraphState) -> bool:
    """Return true only when the non-emergency handoff contract allows it."""

    return state.risk_level in HANDOFF_RISK_LEVELS and state.hospital_visit_intent == "yes"


def build_non_emergency_handoff(state: PetCareGraphState) -> PetCareGraphState:
    """Build a draft handoff JSON/email without sending anything."""

    next_state = state.model_copy(deep=True)

    if next_state.risk_level == "emergency":
        return next_state

    if not should_build_non_emergency_handoff(next_state):
        next_state.handoff.type = "none"
        next_state.handoff.required = False
        next_state.handoff.summary = ""
        next_state.handoff.summary_json = None
        next_state.handoff.email_draft = ""
        return next_state

    summary_json = build_hospital_handoff_summary(next_state)
    summary = _summary_text(summary_json, next_state.locale)
    next_state.handoff.type = "non_emergency"
    next_state.handoff.required = True
    next_state.handoff.summary_json = summary_json
    next_state.handoff.summary = summary
    next_state.handoff.email_draft = _email_draft(next_state, summary)
    next_state.next_route = "handoff"
    return next_state


def handoff_subgraph(state: PetCareGraphState) -> PetCareGraphState:
    """LangGraph-friendly alias for the non-emergency handoff helper."""

    return build_non_emergency_handoff(state)


def build_hospital_handoff_summary(state: PetCareGraphState) -> HospitalHandoffSummary:
    """Build the veterinarian-facing six-section handoff JSON."""

    return HospitalHandoffSummary(
        generated_at=_generated_at(state),
        patient=_patient_summary(state),
        visit_reason=_visit_reason_summary(state),
        clinical_course=_clinical_course_summary(state),
        baseline_comparison=_baseline_comparison_summary(state),
        triage_assessment=_triage_assessment_summary(state),
        medical_background=_medical_background_summary(state),
    )


def _generated_at(state: PetCareGraphState) -> datetime:
    try:
        tzinfo = ZoneInfo(state.timezone)
    except ZoneInfoNotFoundError:
        tzinfo = timezone.utc
    return datetime.now(tzinfo).replace(microsecond=0)


def _patient_summary(state: PetCareGraphState) -> PatientSummary:
    pet = state.context.pet
    return PatientSummary(
        pet_id=str(pet.get("pet_id") or pet.get("id") or state.pet_id or ""),
        name=str(pet.get("name") or ""),
        species=_species(pet.get("species") or state.species),
        breed=_string_or_none(pet.get("breed")),
        sex=_sex(pet.get("sex")),
        neutered=_bool_or_none(pet.get("neutered")),
        age_years=_float_or_none(pet.get("age_years") or pet.get("age")),
        age_display=_string_or_none(pet.get("age_display")),
        weight_kg=_float_or_none(pet.get("weight_kg") or pet.get("weight")),
    )


def _visit_reason_summary(state: PetCareGraphState) -> VisitReasonSummary:
    complaints = _chief_complaints(state)
    return VisitReasonSummary(
        chief_complaints=complaints,
        owner_summary=state.user_input.strip(),
        requested_goal="vet_summary" if state.intent == "handoff_request" else "hospital_visit",
    )


def _clinical_course_summary(state: PetCareGraphState) -> ClinicalCourseSummary:
    duration = state.assessment.duration
    timeline_parts = []
    if wants_korean(state.locale):
        if duration:
            timeline_parts.append(f"보호자가 보고한 시점: {duration}.")
        if state.user_input.strip():
            timeline_parts.append(f"보호자 설명: {state.user_input.strip()}")
        fallback = "경과 시간은 명확히 제공되지 않았습니다."
    else:
        if duration:
            timeline_parts.append(f"Timing reported as: {duration}.")
        if state.user_input.strip():
            timeline_parts.append(f"Caregiver description: {state.user_input.strip()}")
        fallback = "Timeline was not specified."
    return ClinicalCourseSummary(
        onset_text=duration,
        onset_at=None,
        duration_text=duration,
        course_pattern=state.assessment.course_pattern,
        timeline_summary=" ".join(timeline_parts) or fallback,
    )


def _baseline_comparison_summary(state: PetCareGraphState) -> BaselineComparisonSummary:
    fallback = (
        "최근 기준 비교를 사용할 수 없습니다."
        if wants_korean(state.locale)
        else "Baseline comparison was not available."
    )
    return BaselineComparisonSummary(
        window_days=3,
        summary=localize_change_summary(state.change_detection.summary, state.locale) or fallback,
        changes=_baseline_changes(state),
    )


def _baseline_changes(state: PetCareGraphState) -> list[BaselineChange]:
    changes: list[BaselineChange] = []
    for field in [*state.change_detection.worsened_fields, *state.change_detection.improved_fields]:
        if field not in FIELD_LABELS:
            continue
        current = getattr(state.current_status, field, "unknown")
        baseline = getattr(state.baseline_context.baseline_summary, field, "unknown")
        changes.append(
            BaselineChange(
                field=field,  # type: ignore[arg-type]
                label=_field_label(field, state.locale),
                current=_none_if_unknown(current),
                baseline=_none_if_unknown(baseline),
                change_summary=_change_summary(current, state.locale),
                delta_value=None,
                delta_unit=None,
            )
        )
    return changes


def _triage_assessment_summary(state: PetCareGraphState) -> TriageAssessmentSummary:
    return TriageAssessmentSummary(
        associated_symptoms=_associated_symptoms(state),
        red_flags=build_red_flag_summaries(state.emergency_screening.items),
    )


def _associated_symptoms(state: PetCareGraphState) -> list[AssociatedSymptomSummary]:
    symptoms = _clean_values([*state.assessment.symptoms, *state.current_status.symptoms])
    summaries: list[AssociatedSymptomSummary] = []
    seen: set[str] = set()
    for symptom in symptoms:
        name = _associated_symptom_name(symptom)
        key = f"{name}:{symptom}"
        if key in seen:
            continue
        summaries.append(
            AssociatedSymptomSummary(
                name=name,  # type: ignore[arg-type]
                label=_symptom_label(name, symptom, state.locale),
                summary=symptom,
                count=1,
                last_observed_at=None,
            )
        )
        seen.add(key)
    return summaries


def _medical_background_summary(state: PetCareGraphState) -> MedicalBackgroundSummary:
    background = state.context.medical_background
    return MedicalBackgroundSummary(
        conditions=_string_list(background.get("conditions")),
        medications_or_supplements=_string_list(
            background.get("medications_or_supplements") or background.get("medications")
        ),
        allergies=_string_list(background.get("allergies")),
    )


def _summary_text(summary_json: HospitalHandoffSummary, locale: str | None) -> str:
    if wants_korean(locale):
        pet_name = summary_json.patient.name or "반려동물"
        complaints = ", ".join(
            display_list(summary_json.visit_reason.chief_complaints, locale)
        ) or "명확히 특정되지 않은 걱정"
        parts = [
            "병원 전달용 요약 초안입니다.",
            f"환자: {pet_name}.",
            f"주요 증상/걱정: {complaints}.",
        ]
        if summary_json.clinical_course.timeline_summary:
            parts.append(f"경과: {summary_json.clinical_course.timeline_summary}")
        if summary_json.baseline_comparison.summary:
            parts.append(f"최근 3일 비교: {summary_json.baseline_comparison.summary}")
        red_flags = display_list(
            [flag.name for flag in summary_json.triage_assessment.red_flags],
            locale,
        )
        if red_flags:
            parts.append(f"진료 시 언급할 위험 신호: {', '.join(red_flags)}.")
        return " ".join(parts)

    pet_name = summary_json.patient.name or "the pet"
    complaints = ", ".join(summary_json.visit_reason.chief_complaints) or "unspecified concern"
    parts = [
        "Hospital handoff summary draft.",
        f"Patient: {pet_name}.",
        f"Main concerns: {complaints}.",
    ]
    if summary_json.clinical_course.timeline_summary:
        parts.append(f"Timeline: {summary_json.clinical_course.timeline_summary}")
    if summary_json.baseline_comparison.summary:
        parts.append(f"Recent 3-day comparison: {summary_json.baseline_comparison.summary}")
    red_flags = [flag.name for flag in summary_json.triage_assessment.red_flags]
    if red_flags:
        parts.append(f"Red flags to mention: {', '.join(red_flags)}.")
    return " ".join(parts)


def _email_draft(state: PetCareGraphState, summary: str) -> str:
    pet_name = str(state.context.pet.get("name") or "반려동물")
    if wants_korean(state.locale):
        return (
            "초안입니다 - 이메일은 전송되지 않았습니다.\n\n"
            f"제목: {pet_name} 병원 방문 요약\n\n"
            "안녕하세요.\n\n"
            "검토를 위해 PetCare-AI 병원 방문 요약을 공유드립니다:\n\n"
            f"{summary}\n\n"
            "병원에 보내기 전에 이 초안을 확인해 주세요."
        )

    pet_name = str(state.context.pet.get("name") or "the pet")
    return (
        "Draft only - no email has been sent.\n\n"
        f"Subject: Visit summary for {pet_name}\n\n"
        "Hello,\n\n"
        "I would like to share the following PetCare-AI visit summary for review:\n\n"
        f"{summary}\n\n"
        "Please review this draft before sending it to a clinic."
    )


def _chief_complaints(state: PetCareGraphState) -> list[str]:
    complaints = _clean_values([*state.assessment.symptoms, *state.current_status.symptoms])
    if state.emergency_screening.chief_complaint:
        complaints.append(state.emergency_screening.chief_complaint)
    return _clean_values(complaints)


def _associated_symptom_name(symptom: str) -> str:
    normalized = symptom.lower()
    if "vomit" in normalized or "throw" in normalized:
        return "vomiting"
    if "diarrhea" in normalized:
        return "diarrhea"
    if "letharg" in normalized or "weak" in normalized:
        return "lethargy"
    if "cough" in normalized:
        return "coughing"
    if "breath" in normalized or "resp" in normalized:
        return "breathing_issue"
    if "pain" in normalized:
        return "pain"
    if "limp" in normalized:
        return "limping"
    if "toxin" in normalized or "toxic" in normalized:
        return "toxin_exposure"
    return "other"


def _symptom_label(name: str, fallback: str, locale: str | None) -> str:
    if wants_korean(locale):
        return display_label(name, locale)
    labels = {
        "vomiting": "Vomiting",
        "diarrhea": "Diarrhea",
        "lethargy": "Lethargy",
        "coughing": "Coughing",
        "breathing_issue": "Breathing issue",
        "pain": "Pain",
        "limping": "Limping",
        "toxin_exposure": "Toxin exposure",
    }
    return labels.get(name, fallback)


def _field_label(field: str, locale: str | None) -> str:
    if wants_korean(locale):
        return FIELD_LABELS_KO[field]
    return FIELD_LABELS[field]


def _change_summary(current: str, locale: str | None) -> str:
    if current in {"normal", "decreased", "increased", "abnormal"}:
        return current
    return "unknown"


def _species(value: object) -> str:
    normalized = str(value or "unknown").strip().lower()
    return normalized if normalized in {"cat", "dog"} else "unknown"


def _sex(value: object) -> str:
    normalized = str(value or "unknown").strip().lower()
    return normalized if normalized in {"male", "female"} else "unknown"


def _bool_or_none(value: object) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "yes", "y", "neutered", "spayed"}:
            return True
        if normalized in {"false", "no", "n", "intact"}:
            return False
    return None


def _float_or_none(value: object) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _string_or_none(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _string_list(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return _clean_values([str(item) for item in value])
    return _clean_values([str(value)])


def _none_if_unknown(value: object) -> str | None:
    text = str(value or "").strip()
    if not text or text == "unknown":
        return None
    return text


def _clean_values(values: list[str]) -> list[str]:
    cleaned: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = " ".join(str(value).strip().split())
        if not normalized or normalized in seen:
            continue
        cleaned.append(normalized)
        seen.add(normalized)
    return cleaned


__all__ = [
    "HANDOFF_RISK_LEVELS",
    "build_hospital_handoff_summary",
    "build_non_emergency_handoff",
    "handoff_subgraph",
    "should_build_non_emergency_handoff",
]