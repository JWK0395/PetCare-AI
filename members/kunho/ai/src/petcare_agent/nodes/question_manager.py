"""Question selection node for missing emergency checklist details."""

from __future__ import annotations

from petcare_agent.schemas.graph_state import PetCareGraphState
from petcare_agent.schemas.triage import ChecklistItem

MAX_QUESTIONS_PER_TURN = 2
MAX_SAFETY_QUESTION_TURNS = 2
MISSING_INFO_FALLBACK_ROUTE = "chat"
USER_RESPONSE_REENTRY_ROUTE = "state_updater"


def manage_questions(state: PetCareGraphState) -> PetCareGraphState:
    """Select up to two missing required red-flag questions and update state."""

    next_state = state.model_copy(deep=True)
    selected_item_ids = select_missing_required_questions(next_state)

    if not selected_item_ids:
        next_state.internal_triage_assessment.needs_followup = False
        next_state.internal_triage_assessment.followup_questions = []
        next_state.next_route = MISSING_INFO_FALLBACK_ROUTE
        return next_state

    missing_questions = list(next_state.emergency_screening.missing_questions)
    for item_id in selected_item_ids:
        item = next_state.emergency_screening.items[item_id]
        item.asked_count += 1
        if item_id not in missing_questions:
            missing_questions.append(item_id)

    next_state.emergency_screening.missing_questions = missing_questions
    next_state.internal_triage_assessment.needs_followup = True
    next_state.internal_triage_assessment.followup_questions = _question_texts(next_state)
    next_state.emergency_screening.status = "in_progress"
    next_state.safety_question_turns = min(
        MAX_SAFETY_QUESTION_TURNS,
        next_state.safety_question_turns + 1,
    )
    next_state.next_route = USER_RESPONSE_REENTRY_ROUTE
    return next_state


def question_manager(state: PetCareGraphState) -> PetCareGraphState:
    """LangGraph-friendly alias for the question manager node."""

    return manage_questions(state)


def select_missing_required_questions(
    state: PetCareGraphState,
    *,
    max_questions: int = MAX_QUESTIONS_PER_TURN,
) -> list[str]:
    """Return eligible missing required red-flag item ids sorted by priority."""

    if max_questions <= 0 or state.safety_question_turns >= MAX_SAFETY_QUESTION_TURNS:
        return []

    answered_item_ids = _answered_item_ids(state)
    pending_item_ids = set(state.emergency_screening.missing_questions)
    candidates: list[tuple[int, int, str]] = []

    for position, (item_id, item) in enumerate(state.emergency_screening.items.items()):
        if item_id in answered_item_ids or item.item_id in answered_item_ids:
            continue
        if item_id in pending_item_ids or item.item_id in pending_item_ids:
            continue
        if item.asked_count > 0:
            continue
        if not _is_required_red_flag_item(item):
            continue
        if not _item_is_unknown(item):
            continue
        if not _has_question_text(item):
            continue

        candidates.append((_priority_sort_value(item), position, item_id))

    candidates.sort()
    return [item_id for _, _, item_id in candidates[:max_questions]]


def _question_texts(state: PetCareGraphState) -> list[str]:
    questions: list[str] = []
    for item_id in state.emergency_screening.missing_questions:
        item = state.emergency_screening.items.get(item_id)
        if item is not None and item.question_text:
            questions.append(item.question_text)
        if len(questions) >= MAX_QUESTIONS_PER_TURN:
            break
    return questions


def _answered_item_ids(state: PetCareGraphState) -> set[str]:
    answered = set(state.emergency_screening.answered_questions)
    for item_id in list(answered):
        if item_id.startswith("resp_"):
            answered.add(item_id.removeprefix("resp_"))
    return answered


def _is_required_red_flag_item(item: ChecklistItem) -> bool:
    return item.metadata.get("red_flag") is True


def _item_is_unknown(item: ChecklistItem) -> bool:
    if item.value is None:
        return True
    if isinstance(item.value, str) and item.value.strip().lower() == "unknown":
        return True
    return item.confidence == "unknown"


def _has_question_text(item: ChecklistItem) -> bool:
    return item.question_text is not None and bool(item.question_text.strip())


def _priority_sort_value(item: ChecklistItem) -> int:
    if item.priority is None:
        return 999
    return item.priority


__all__ = [
    "MAX_QUESTIONS_PER_TURN",
    "MAX_SAFETY_QUESTION_TURNS",
    "manage_questions",
    "question_manager",
    "select_missing_required_questions",
]

