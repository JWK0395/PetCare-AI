from __future__ import annotations

from pydantic import BaseModel

from petcare_agent.nodes.state_updater import update_state_from_user_input
from petcare_agent.schemas.graph_state import (
    BaselineContext,
    ChangeDetection,
    CurrentStatus,
    EmergencyScreening,
    PetCareContext,
    PetCareGraphState,
)
from petcare_agent.schemas.llm_outputs import StateExtractionOutput


class FakeLLMClient:
    def __init__(self, output: BaseModel) -> None:
        self.output = output

    def structured_output(self, **kwargs):
        return self.output


def test_state_updater_updates_species_current_status_and_assessment() -> None:
    output = StateExtractionOutput(
        species="cat",
        symptoms=["coughing"],
        duration="since this morning",
        current_status=CurrentStatus(
            symptoms=["coughing"],
            appetite="decreased",
            water="unknown",
            activity="decreased",
        ),
        negated_findings=[],
        uncertain_findings=[],
    )
    state = PetCareGraphState(user_input="My cat has been coughing since this morning.")

    result = update_state_from_user_input(state, llm_client=FakeLLMClient(output))

    assert result.species == "cat"
    assert result.assessment.symptoms == ["coughing"]
    assert result.assessment.duration == "since this morning"
    assert result.current_status.symptoms == ["coughing"]
    assert result.current_status.appetite == "decreased"
    assert result.current_status.activity == "decreased"


def test_state_updater_preserves_context_baseline_change_and_screening() -> None:
    output = StateExtractionOutput(
        species="dog",
        symptoms=["vomiting"],
        duration="2 hours",
        current_status=CurrentStatus(symptoms=["vomiting"], appetite="unknown"),
        negated_findings=[],
        uncertain_findings=[],
    )
    context = PetCareContext(pet={"id": 1, "name": "Kongi"})
    baseline = BaselineContext(baseline_available=True)
    change = ChangeDetection(baseline_available=True, summary="existing")
    screening = EmergencyScreening(checklist_id="vomiting_triage")
    state = PetCareGraphState(
        context=context,
        baseline_context=baseline,
        change_detection=change,
        emergency_screening=screening,
    )

    result = update_state_from_user_input(state, llm_client=FakeLLMClient(output))

    assert result.context == context
    assert result.baseline_context == baseline
    assert result.change_detection == change
    assert result.emergency_screening == screening
