import pytest

from petcare_agent.contracts import load_json_schema
from petcare_agent.safety.checklist_loader import (
    FALLBACK_CHECKLIST_ID,
    checklist_template_as_contract_dict,
    load_all_checklist_templates,
    load_checklist_template,
    select_checklist_template,
)
from petcare_agent.schemas.triage import ChecklistItem, ChecklistTemplate

EXPECTED_TEMPLATE_IDS = {
    "cat_cough_triage",
    "dog_cough_triage",
    "vomiting_triage",
    "diarrhea_triage",
    "breathing_triage",
    "seizure_triage",
    "toxicity_triage",
    "urinary_triage",
}


def _all_items(template: ChecklistTemplate) -> list[ChecklistItem]:
    return [*template.required_items, *template.optional_items]


def test_loads_mvp_checklist_templates_with_pydantic_validation() -> None:
    templates = load_all_checklist_templates()

    assert set(templates) == EXPECTED_TEMPLATE_IDS
    assert all(isinstance(template, ChecklistTemplate) for template in templates.values())


def test_load_single_checklist_template_returns_copy() -> None:
    template = load_checklist_template("cat_cough_triage")
    template.required_items[0].asked_count = 2

    fresh_template = load_checklist_template("cat_cough_triage")
    assert fresh_template.required_items[0].asked_count == 0


@pytest.mark.parametrize(
    ("species", "chief_complaint", "expected_id"),
    [
        ("cat", "cough", "cat_cough_triage"),
        ("dog", "cough", "dog_cough_triage"),
        ("cat", "vomiting", "vomiting_triage"),
        ("dog", "vomiting", "vomiting_triage"),
        ("cat", "diarrhea", "diarrhea_triage"),
        ("dog", "diarrhea", "diarrhea_triage"),
        ("cat", "breathing", "breathing_triage"),
        ("dog", "breathing", "breathing_triage"),
        ("cat", "seizure", "seizure_triage"),
        ("dog", "seizure", "seizure_triage"),
        ("cat", "toxicity", "toxicity_triage"),
        ("dog", "toxicity", "toxicity_triage"),
        ("cat", "urinary", "urinary_triage"),
        ("dog", "urinary", "urinary_triage"),
    ],
)
def test_selects_template_by_species_and_chief_complaint(
    species: str,
    chief_complaint: str,
    expected_id: str,
) -> None:
    template = select_checklist_template(species, chief_complaint)

    assert template.checklist_id == expected_id


@pytest.mark.parametrize(
    ("chief_complaint", "expected_id"),
    [
        ("vomiting", "vomiting_triage"),
        ("diarrhea", "diarrhea_triage"),
        ("breathing", "breathing_triage"),
        ("seizure", "seizure_triage"),
        ("toxicity", "toxicity_triage"),
        ("urinary", "urinary_triage"),
    ],
)
def test_common_cat_dog_templates_handle_unknown_species(
    chief_complaint: str,
    expected_id: str,
) -> None:
    template = select_checklist_template("unknown", chief_complaint)

    assert template.checklist_id == expected_id
    assert template.species == "cat/dog"


@pytest.mark.parametrize(
    ("species", "chief_complaint"),
    [
        ("unknown", "cough"),
        ("cat", "not_a_supported_complaint"),
        (None, None),
    ],
)
def test_selection_uses_conservative_breathing_fallback(
    species: str | None,
    chief_complaint: str | None,
) -> None:
    template = select_checklist_template(species, chief_complaint)

    assert template.checklist_id == FALLBACK_CHECKLIST_ID


def test_required_red_flag_items_are_not_empty() -> None:
    for template in load_all_checklist_templates().values():
        assert template.required_items
        for item in template.required_items:
            assert item.item_id.strip()
            assert item.label.strip()
            assert item.metadata.get("red_flag") is True


def test_question_priorities_are_defined_and_in_range() -> None:
    for template in load_all_checklist_templates().values():
        for item in _all_items(template):
            assert item.question_text
            assert item.priority is not None
            assert 1 <= item.priority <= 5
            assert item.metadata.get("question_limit_contract") == (
                "max_2_total_safety_questions"
            )


def test_templates_do_not_conflict_with_json_schema_or_pydantic_contracts() -> None:
    schema = load_json_schema("triage_checklist")
    template_fields = set(schema["properties"])
    item_fields = set(schema["$defs"]["ChecklistItem"]["properties"])
    required_template_fields = set(schema["required"])
    required_item_fields = set(schema["$defs"]["ChecklistItem"]["required"])

    for template in load_all_checklist_templates().values():
        serialized = checklist_template_as_contract_dict(template)
        assert set(serialized) == template_fields
        assert required_template_fields <= set(serialized)
        assert ChecklistTemplate.model_validate(serialized) == template

        for item in [*serialized["required_items"], *serialized["optional_items"]]:
            assert set(item) == item_fields
            assert required_item_fields <= set(item)
            assert item["priority"] is None or 1 <= item["priority"] <= 5
            ChecklistItem.model_validate(item)

