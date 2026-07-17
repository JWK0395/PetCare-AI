from __future__ import annotations

from petcare_agent.nodes.question_manager import (
    manage_questions,
    question_manager,
    select_missing_required_questions,
)
from petcare_agent.safety.checklist_loader import load_checklist_template
from petcare_agent.schemas.graph_state import EmergencyScreening, PetCareGraphState
from petcare_agent.schemas.triage import ChecklistItem, ChecklistTemplate


def _items_by_id(template: ChecklistTemplate) -> dict[str, ChecklistItem]:
    return {item.item_id: item for item in [*template.required_items, *template.optional_items]}


def _item(
    item_id: str,
    *,
    priority: int,
    question_text: str | None = None,
    value: bool | str | None = None,
    confidence: str = "unknown",
    asked_count: int = 0,
    red_flag: bool = True,
) -> ChecklistItem:
    return ChecklistItem(
        item_id=item_id,
        label=item_id.replace("_", " ").title(),
        type="boolean",
        value=value,
        confidence=confidence,  # type: ignore[arg-type]
        source="user_input",
        asked_count=asked_count,
        question_text=question_text if question_text is not None else f"Question for {item_id}?",
        priority=priority,
        metadata={"red_flag": red_flag},
    )


def _state_with_items(
    items: list[ChecklistItem],
    *,
    answered_questions: dict[str, object] | None = None,
    missing_questions: list[str] | None = None,
    safety_question_turns: int = 0,
) -> PetCareGraphState:
    return PetCareGraphState(
        safety_question_turns=safety_question_turns,
        emergency_screening=EmergencyScreening(
            checklist_id="test_triage",
            chief_complaint="test",
            items={item.item_id: item for item in items},
            answered_questions=answered_questions or {},
            missing_questions=missing_questions or [],
        ),
    )


def test_priority_order_selects_high_priority_missing_items_first() -> None:
    state = _state_with_items(
        [
            _item("low_priority", priority=4),
            _item("highest_priority", priority=1),
            _item("middle_priority", priority=2),
        ]
    )

    result = manage_questions(state)

    assert result.emergency_screening.missing_questions == [
        "highest_priority",
        "middle_priority",
    ]


def test_question_manager_selects_at_most_two_questions_and_updates_route() -> None:
    template = load_checklist_template("cat_cough_triage")
    state = PetCareGraphState(
        emergency_screening=EmergencyScreening(
            checklist_id=template.checklist_id,
            chief_complaint=template.chief_complaint,
            items=_items_by_id(template),
        ),
    )

    result = question_manager(state)

    assert len(result.emergency_screening.missing_questions) == 2
    assert result.safety_question_turns == 1
    assert result.next_route == "state_updater"
    assert result.emergency_screening.status == "in_progress"


def test_safety_question_turns_never_exceeds_two() -> None:
    state = _state_with_items(
        [_item("open_mouth_breathing", priority=1)],
        safety_question_turns=2,
    )

    result = manage_questions(state)

    assert result.safety_question_turns == 2
    assert result.emergency_screening.missing_questions == []
    assert result.next_route == "chat"
    assert result.emergency_screening.items["open_mouth_breathing"].asked_count == 0


def test_answered_questions_are_not_asked_again() -> None:
    state = _state_with_items(
        [
            _item("already_answered", priority=1),
            _item("next_question", priority=2),
        ],
        answered_questions={"already_answered": False},
    )

    result = manage_questions(state)

    assert result.emergency_screening.missing_questions == ["next_question"]
    assert result.emergency_screening.items["already_answered"].asked_count == 0
    assert result.emergency_screening.items["next_question"].asked_count == 1


def test_item_without_question_text_is_skipped_safely() -> None:
    no_text_item = _item("no_question_text", priority=1)
    no_text_item.question_text = None
    state = _state_with_items(
        [
            no_text_item,
            _item("askable_question", priority=2),
        ]
    )

    result = manage_questions(state)

    assert result.emergency_screening.missing_questions == ["askable_question"]
    assert result.emergency_screening.items["no_question_text"].asked_count == 0


def test_selected_items_have_asked_count_incremented() -> None:
    state = _state_with_items(
        [
            _item("first_question", priority=1),
            _item("second_question", priority=1),
        ]
    )

    result = manage_questions(state)

    assert result.emergency_screening.items["first_question"].asked_count == 1
    assert result.emergency_screening.items["second_question"].asked_count == 1


def test_previously_asked_or_pending_items_are_not_repeated() -> None:
    state = _state_with_items(
        [
            _item("asked_before", priority=1, asked_count=1),
            _item("already_pending", priority=1),
            _item("fresh_question", priority=2),
        ],
        missing_questions=["already_pending"],
    )

    result = manage_questions(state)

    assert result.emergency_screening.missing_questions == [
        "already_pending",
        "fresh_question",
    ]
    assert result.emergency_screening.items["asked_before"].asked_count == 1
    assert result.emergency_screening.items["already_pending"].asked_count == 0
    assert result.emergency_screening.items["fresh_question"].asked_count == 1


def test_selection_ignores_answered_resp_prefixed_question_ids() -> None:
    state = _state_with_items(
        [
            _item("open_mouth_breathing", priority=1),
            _item("labored_breathing", priority=2),
        ],
        answered_questions={"resp_open_mouth_breathing": False},
    )

    selected = select_missing_required_questions(state)

    assert selected == ["labored_breathing"]

