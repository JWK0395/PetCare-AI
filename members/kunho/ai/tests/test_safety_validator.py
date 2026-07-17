from petcare_agent.safety.checklist_loader import load_checklist_template
from petcare_agent.safety.validator import validate_checklist
from petcare_agent.schemas.triage import ChecklistItem, ChecklistTemplate


def _all_items(template: ChecklistTemplate) -> list[ChecklistItem]:
    return [*template.required_items, *template.optional_items]


def _set_item(
    template: ChecklistTemplate,
    item_id: str,
    value: bool | int | float | str | None,
    *,
    confidence: str = "high",
) -> None:
    for item in _all_items(template):
        if item.item_id == item_id:
            item.value = value
            item.confidence = confidence  # type: ignore[assignment]
            return
    raise AssertionError(f"Unknown item_id in test template: {item_id}")


def _set_required_items_false(template: ChecklistTemplate) -> None:
    for item in template.required_items:
        item.value = False
        item.confidence = "high"


def _append_bool_item(
    template: ChecklistTemplate,
    item_id: str,
    value: bool,
    *,
    confidence: str = "high",
) -> None:
    template.optional_items.append(
        ChecklistItem(
            item_id=item_id,
            label=item_id.replace("_", " ").title(),
            type="boolean",
            value=value,
            confidence=confidence,  # type: ignore[arg-type]
            source="user_input",
            asked_count=0,
            metadata={"red_flag": False},
        )
    )


def test_open_mouth_breathing_true_for_cat_is_emergency() -> None:
    template = load_checklist_template("cat_cough_triage")
    _set_required_items_false(template)
    _set_item(template, "open_mouth_breathing", True)

    result = validate_checklist(template)

    assert result.risk_level == "emergency"
    assert result.action == "final"
    assert any(hit.rule_id == "E_RESP_001" for hit in result.triggered_rules)


def test_gum_color_abnormal_true_is_emergency() -> None:
    template = load_checklist_template("dog_cough_triage")
    _set_required_items_false(template)
    _set_item(template, "gum_color_abnormal", True)

    result = validate_checklist(template)

    assert result.risk_level == "emergency"
    assert any(hit.rule_id == "E_RESP_003" for hit in result.triggered_rules)


def test_required_item_missing_with_zero_questions_needs_more_info() -> None:
    template = load_checklist_template("cat_cough_triage")

    result = validate_checklist(template, safety_question_turns=0)

    assert result.risk_level == "unknown"
    assert result.action == "needs_more_info"
    assert result.requires_more_info is True
    assert "open_mouth_breathing" in result.missing_items
    assert any(hit.rule_id == "Q_MISSING_001" for hit in result.triggered_rules)


def test_required_item_missing_after_two_questions_is_unknown_after_max_questions() -> None:
    template = load_checklist_template("cat_cough_triage")

    result = validate_checklist(template, safety_question_turns=2)

    assert result.risk_level == "unknown"
    assert result.action == "unknown_after_max_questions"
    assert result.requires_more_info is False
    assert any(hit.rule_id == "Q_MISSING_002" for hit in result.triggered_rules)


def test_no_emergency_urgent_or_unknown_rules_is_non_emergency() -> None:
    template = load_checklist_template("cat_cough_triage")
    _set_required_items_false(template)

    result = validate_checklist(template)

    assert result.risk_level == "non_emergency"
    assert result.action == "final"
    assert result.missing_items == []
    assert [hit.rule_id for hit in result.triggered_rules] == ["N_NONE_001"]


def test_repeated_vomiting_true_is_urgent() -> None:
    template = load_checklist_template("vomiting_triage")
    _set_required_items_false(template)
    _set_item(template, "repeated_vomiting", True)

    result = validate_checklist(template)

    assert result.risk_level == "urgent"
    assert any(hit.rule_id == "U_GI_001" for hit in result.triggered_rules)


def test_emergency_priority_wins_over_urgent_rule() -> None:
    template = load_checklist_template("cat_cough_triage")
    _set_required_items_false(template)
    _set_item(template, "open_mouth_breathing", True)
    _append_bool_item(template, "rapid_breathing", True)

    result = validate_checklist(template)

    assert result.risk_level == "emergency"
    rule_ids = {hit.rule_id for hit in result.triggered_rules}
    assert "E_RESP_001" in rule_ids
    assert "U_RESP_001" in rule_ids


def test_low_confidence_emergency_item_returns_unknown_trace() -> None:
    template = load_checklist_template("cat_cough_triage")
    _set_required_items_false(template)
    _set_item(template, "open_mouth_breathing", True, confidence="low")

    result = validate_checklist(template)

    assert result.risk_level == "unknown"
    assert result.action == "unknown_due_to_low_confidence"
    assert result.requires_more_info is True
    rule_ids = {hit.rule_id for hit in result.triggered_rules}
    assert "Q_CONF_001" in rule_ids
    assert "E_RESP_001" not in rule_ids


def test_shared_template_can_use_species_override_for_cat_open_mouth_rule() -> None:
    template = load_checklist_template("breathing_triage")
    _set_required_items_false(template)
    _set_item(template, "open_mouth_breathing", True)

    result = validate_checklist(template, species="cat")

    assert result.risk_level == "emergency"
    assert any(hit.rule_id == "E_RESP_001" for hit in result.triggered_rules)


def test_validator_does_not_mutate_template() -> None:
    template = load_checklist_template("cat_cough_triage")
    before = template.model_dump(mode="json")

    validate_checklist(template)

    assert template.model_dump(mode="json") == before
