from __future__ import annotations

from pydantic import BaseModel

from petcare_agent.nodes.checklist_extractor import extract_checklist_updates
from petcare_agent.safety.checklist_loader import load_checklist_template
from petcare_agent.schemas.graph_state import EmergencyScreening, PetCareGraphState
from petcare_agent.schemas.llm_outputs import ChecklistExtractionOutput
from petcare_agent.schemas.triage import ChecklistItem, ChecklistTemplate


class FakeLLMClient:
    def __init__(self, output: BaseModel) -> None:
        self.output = output

    def structured_output(self, **kwargs):
        return self.output


def _items_by_id(template: ChecklistTemplate) -> dict[str, ChecklistItem]:
    return {item.item_id: item for item in [*template.required_items, *template.optional_items]}


def test_checklist_extractor_updates_existing_items_and_evidence() -> None:
    template = load_checklist_template("cat_cough_triage")
    state = PetCareGraphState(
        user_input="Mouth is closed, but coughing is repeated.",
        risk_level="urgent",
        emergency_screening=EmergencyScreening(
            checklist_id=template.checklist_id,
            chief_complaint=template.chief_complaint,
            items=_items_by_id(template),
        ),
    )
    output = ChecklistExtractionOutput(
        checklist_id="cat_cough_triage",
        updates=[
            {
                "item_id": "open_mouth_breathing",
                "value": False,
                "confidence": "high",
                "evidence": "mouth is closed",
            },
            {
                "item_id": "lethargy",
                "value": True,
                "confidence": "medium",
                "evidence": "seems tired",
            },
        ],
    )

    result = extract_checklist_updates(state, llm_client=FakeLLMClient(output))

    open_mouth = result.emergency_screening.items["open_mouth_breathing"]
    assert open_mouth.value is False
    assert open_mouth.confidence == "high"
    assert open_mouth.metadata["evidence"] == "mouth is closed"
    lethargy = result.emergency_screening.items["lethargy"]
    assert lethargy.value is True
    assert lethargy.confidence == "medium"
    assert lethargy.metadata["evidence"] == "seems tired"


def test_checklist_extractor_ignores_unknown_item_id_and_preserves_risk_level() -> None:
    template = load_checklist_template("cat_cough_triage")
    state = PetCareGraphState(
        risk_level="urgent",
        emergency_screening=EmergencyScreening(
            checklist_id=template.checklist_id,
            items=_items_by_id(template),
        ),
    )
    output = ChecklistExtractionOutput(
        checklist_id="cat_cough_triage",
        updates=[
            {
                "item_id": "not_in_template",
                "value": True,
                "confidence": "high",
                "evidence": "ignored",
            }
        ],
    )

    result = extract_checklist_updates(state, llm_client=FakeLLMClient(output))

    assert "not_in_template" not in result.emergency_screening.items
    assert result.risk_level == "urgent"


def test_checklist_extractor_maps_short_affirmative_to_first_pending_question() -> None:
    template = load_checklist_template("cat_cough_triage")
    items = _items_by_id(template)
    items["open_mouth_breathing"].asked_count = 1
    state = PetCareGraphState(
        user_input="\ub124",
        emergency_screening=EmergencyScreening(
            checklist_id=template.checklist_id,
            chief_complaint=template.chief_complaint,
            items=items,
            missing_questions=["open_mouth_breathing", "labored_breathing"],
        ),
    )
    output = ChecklistExtractionOutput(checklist_id="cat_cough_triage", updates=[])

    result = extract_checklist_updates(state, llm_client=FakeLLMClient(output))

    open_mouth = result.emergency_screening.items["open_mouth_breathing"]
    assert open_mouth.value is True
    assert open_mouth.confidence == "high"
    assert open_mouth.metadata["evidence"] == "\ub124"
    assert result.emergency_screening.missing_questions == ["labored_breathing"]
    assert result.emergency_screening.answered_questions["open_mouth_breathing"] is True
    assert result.emergency_screening.answered_questions["resp_open_mouth_breathing"] is True


def test_checklist_extractor_prunes_llm_answered_pending_question() -> None:
    template = load_checklist_template("cat_cough_triage")
    state = PetCareGraphState(
        user_input="No, mouth is closed.",
        emergency_screening=EmergencyScreening(
            checklist_id=template.checklist_id,
            chief_complaint=template.chief_complaint,
            items=_items_by_id(template),
            missing_questions=["open_mouth_breathing", "labored_breathing"],
        ),
    )
    output = ChecklistExtractionOutput(
        checklist_id="cat_cough_triage",
        updates=[
            {
                "item_id": "open_mouth_breathing",
                "value": False,
                "confidence": "high",
                "evidence": "mouth is closed",
            }
        ],
    )

    result = extract_checklist_updates(state, llm_client=FakeLLMClient(output))

    assert result.emergency_screening.items["open_mouth_breathing"].value is False
    assert result.emergency_screening.missing_questions == ["labored_breathing"]
    assert result.emergency_screening.answered_questions["open_mouth_breathing"] is False
    assert result.emergency_screening.answered_questions["resp_open_mouth_breathing"] is False

