from __future__ import annotations

from pydantic import BaseModel

from petcare_agent.contracts import load_json_schema
from petcare_agent.nodes.question_manager import question_manager
from petcare_agent.nodes.safety_guard import run_safety_guard
from petcare_agent.schemas.graph_state import AssessmentState, EmergencyScreening, PetCareGraphState
from petcare_agent.schemas.handoff import HospitalHandoffSummary, InternalTriageAssessment
from petcare_agent.schemas.llm_outputs import ChecklistExtractionOutput


class FakeLLMClient:
    def __init__(self, output: BaseModel) -> None:
        self.output = output

    def structured_output(self, **kwargs):
        return self.output


def test_hospital_handoff_schema_has_six_sections_and_no_internal_risk_fields() -> None:
    schema = load_json_schema("hospital_handoff_summary")

    assert schema["properties"]["schema_version"]["const"] == "1.1"
    assert schema["required"] == [
        "schema_version",
        "generated_at",
        "patient",
        "visit_reason",
        "clinical_course",
        "baseline_comparison",
        "triage_assessment",
        "medical_background",
    ]
    assert schema["additionalProperties"] is False
    assert schema["properties"]["baseline_comparison"]["properties"]["window_days"]["const"] == 3

    top_level_fields = set(schema["properties"])
    assert "risk_level" not in top_level_fields
    assert "confidence" not in top_level_fields
    assert "missing_items" not in top_level_fields
    assert "triggered_rules" not in top_level_fields
    assert "decision_basis" not in top_level_fields

    triage_properties = set(schema["properties"]["triage_assessment"]["properties"])
    assert triage_properties == {"associated_symptoms", "red_flags"}


def test_internal_triage_schema_owns_risk_level_and_followup_limit() -> None:
    schema = load_json_schema("internal_triage_assessment")

    assert schema["properties"]["schema_version"]["const"] == "1.0"
    assert "risk_level" in schema["required"]
    assert schema["properties"]["risk_level"]["enum"] == [
        "emergency",
        "urgent",
        "non_emergency",
        "unknown",
    ]
    assert schema["properties"]["followup_questions"]["maxItems"] == 2


def test_pydantic_contract_models_match_schema_names() -> None:
    handoff_schema = load_json_schema("hospital_handoff_summary")
    internal_schema = load_json_schema("internal_triage_assessment")

    assert set(HospitalHandoffSummary.model_fields) == set(handoff_schema["properties"])
    assert set(InternalTriageAssessment.model_fields) == set(internal_schema["properties"])


def test_safety_guard_populates_internal_triage_assessment_for_emergency() -> None:
    state = PetCareGraphState(
        species="cat",
        emergency_screening=EmergencyScreening(
            checklist_id="cat_cough_triage",
            chief_complaint="cough",
        ),
    )
    output = ChecklistExtractionOutput(
        checklist_id="cat_cough_triage",
        updates=[
            {
                "item_id": "open_mouth_breathing",
                "value": True,
                "confidence": "high",
                "evidence": "open mouth breathing",
            }
        ],
    )

    result = run_safety_guard(state, llm_client=FakeLLMClient(output))

    internal = result.internal_triage_assessment
    assert internal.risk_level == "emergency"
    assert internal.red_flag_inputs.open_mouth_breathing is True
    assert internal.needs_followup is False
    assert internal.followup_questions == []
    assert result.emergency_screening.red_flags == ["open_mouth_breathing"]


def test_unknown_triage_uses_followup_questions_in_internal_assessment() -> None:
    state = PetCareGraphState(
        species="cat",
        assessment=AssessmentState(symptoms=["coughing"]),
        emergency_screening=EmergencyScreening(
            checklist_id="cat_cough_triage",
            chief_complaint="cough",
        ),
    )
    safety_result = run_safety_guard(
        state,
        llm_client=FakeLLMClient(ChecklistExtractionOutput(checklist_id="cat_cough_triage", updates=[])),
    )

    question_result = question_manager(safety_result)

    internal = question_result.internal_triage_assessment
    assert internal.risk_level == "unknown"
    assert internal.needs_followup is True
    assert 0 < len(internal.followup_questions) <= 2
    assert question_result.next_route == "state_updater"
