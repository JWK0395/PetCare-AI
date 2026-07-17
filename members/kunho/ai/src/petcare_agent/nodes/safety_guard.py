"""Safety Guard node wiring checklist extraction to rule validation."""

from __future__ import annotations

from petcare_agent.llm.client import StructuredOutputClient
from petcare_agent.nodes.checklist_extractor import extract_checklist_updates
from petcare_agent.safety.checklist_loader import (
    ChecklistLoaderError,
    load_checklist_template,
    select_checklist_template,
)
from petcare_agent.safety.red_flags import build_red_flag_inputs, canonical_red_flag_id
from petcare_agent.safety.validator import validate_checklist
from petcare_agent.schemas.graph_state import PetCareGraphState
from petcare_agent.schemas.handoff import ClinicalInputs, InternalTriageAssessment
from petcare_agent.schemas.triage import ChecklistItem, ChecklistTemplate


def run_safety_guard(
    state: PetCareGraphState,
    *,
    llm_client: StructuredOutputClient | None = None,
) -> PetCareGraphState:
    """Select/fill a checklist and apply the Phase 3 rule validator."""

    next_state = state.model_copy(deep=True)
    template = _ensure_screening_template(next_state)
    next_state = extract_checklist_updates(next_state, llm_client=llm_client)

    validation_template = _template_with_screening_items(
        template,
        next_state.emergency_screening.items,
    )
    result = validate_checklist(
        validation_template,
        safety_question_turns=next_state.safety_question_turns,
        species=next_state.species,
    )

    next_state.risk_level = result.risk_level
    next_state.confidence = result.confidence
    next_state.assessment.missing_fields = list(result.missing_items)
    next_state.emergency_screening.triggered_rules = list(result.triggered_rules)
    next_state.emergency_screening.red_flags = _red_flag_item_ids(
        next_state.emergency_screening.items
    )
    next_state.internal_triage_assessment = InternalTriageAssessment(
        risk_level=result.risk_level,
        red_flag_inputs=build_red_flag_inputs(next_state.emergency_screening.items),
        clinical_inputs=_clinical_inputs(next_state),
        needs_followup=result.requires_more_info,
        followup_questions=_followup_question_texts(next_state),
    )

    if result.requires_more_info:
        next_state.emergency_screening.status = "in_progress"
        next_state.next_route = "question_manager"
    elif result.risk_level == "emergency":
        next_state.emergency_screening.status = "complete"
        next_state.next_route = "emergency"
    else:
        next_state.emergency_screening.status = "complete"
        next_state.next_route = "chat"

    return next_state


def safety_guard(
    state: PetCareGraphState,
    *,
    llm_client: StructuredOutputClient | None = None,
) -> PetCareGraphState:
    """LangGraph-friendly alias for the Safety Guard node."""

    return run_safety_guard(state, llm_client=llm_client)


def _ensure_screening_template(state: PetCareGraphState) -> ChecklistTemplate:
    template = _load_or_select_template(state)
    existing_items = state.emergency_screening.items
    state.emergency_screening.checklist_id = template.checklist_id
    state.emergency_screening.chief_complaint = (
        state.emergency_screening.chief_complaint or template.chief_complaint
    )
    state.emergency_screening.items = _merge_existing_items(template, existing_items)
    return template


def _load_or_select_template(state: PetCareGraphState) -> ChecklistTemplate:
    if state.emergency_screening.checklist_id:
        try:
            return load_checklist_template(state.emergency_screening.checklist_id)
        except ChecklistLoaderError:
            pass

    return select_checklist_template(
        state.species,
        state.emergency_screening.chief_complaint or None,
    )


def _merge_existing_items(
    template: ChecklistTemplate,
    existing_items: dict[str, ChecklistItem],
) -> dict[str, ChecklistItem]:
    merged: dict[str, ChecklistItem] = {}
    for template_item in [*template.required_items, *template.optional_items]:
        existing = existing_items.get(template_item.item_id)
        if existing is None:
            merged[template_item.item_id] = template_item.model_copy(deep=True)
            continue

        merged[template_item.item_id] = template_item.model_copy(
            update={
                "value": existing.value,
                "confidence": existing.confidence,
                "source": existing.source,
                "asked_count": existing.asked_count,
                "unit": existing.unit or template_item.unit,
                "question_text": existing.question_text or template_item.question_text,
                "priority": existing.priority or template_item.priority,
                "metadata": {**template_item.metadata, **existing.metadata},
            },
            deep=True,
        )
    return merged


def _template_with_screening_items(
    template: ChecklistTemplate,
    screening_items: dict[str, ChecklistItem],
) -> ChecklistTemplate:
    return template.model_copy(
        update={
            "required_items": [
                screening_items.get(item.item_id, item).model_copy(deep=True)
                for item in template.required_items
            ],
            "optional_items": [
                screening_items.get(item.item_id, item).model_copy(deep=True)
                for item in template.optional_items
            ],
        },
        deep=True,
    )


def _red_flag_item_ids(items: dict[str, ChecklistItem]) -> list[str]:
    red_flags: list[str] = []
    seen: set[str] = set()
    for item_id, item in items.items():
        if item.metadata.get("red_flag") is not True or not _is_true(item.value):
            continue
        canonical_id = canonical_red_flag_id(item_id)
        if canonical_id is None or canonical_id in seen:
            continue
        red_flags.append(canonical_id)
        seen.add(canonical_id)
    return red_flags


def _clinical_inputs(state: PetCareGraphState) -> ClinicalInputs:
    return ClinicalInputs(
        onset_known=bool(state.assessment.duration),
        course_pattern=state.assessment.course_pattern,
        baseline_change_known=state.change_detection.baseline_available,
        associated_symptom_count_known=_associated_symptom_count_known(state),
    )


def _associated_symptom_count_known(state: PetCareGraphState) -> bool:
    if state.assessment.symptoms or state.current_status.symptoms:
        return True
    counted_item_ids = {"repeated_vomiting", "repeated_diarrhea", "duration_hours", "duration_minutes"}
    return any(
        item_id in counted_item_ids and item.value is not None
        for item_id, item in state.emergency_screening.items.items()
    )


def _followup_question_texts(state: PetCareGraphState) -> list[str]:
    questions: list[str] = []
    for item_id in state.emergency_screening.missing_questions:
        item = state.emergency_screening.items.get(item_id)
        if item is not None and item.question_text:
            questions.append(item.question_text)
        if len(questions) >= 2:
            break
    return questions


def _is_true(value: object) -> bool:
    if value is True:
        return True
    if isinstance(value, str):
        return value.strip().lower() in {"true", "yes", "y"}
    return False


__all__ = ["run_safety_guard", "safety_guard"]

