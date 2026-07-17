"""State extraction and update structured-output node."""

from __future__ import annotations

import json

from petcare_agent.llm.client import StructuredOutputClient, call_structured_output
from petcare_agent.prompts import load_prompt
from petcare_agent.schemas.graph_state import PetCareGraphState
from petcare_agent.schemas.llm_outputs import StateExtractionOutput


def update_state_from_user_input(
    state: PetCareGraphState,
    *,
    llm_client: StructuredOutputClient | None = None,
) -> PetCareGraphState:
    """Extract current pet state while preserving unrelated graph context."""

    next_state = state.model_copy(deep=True)
    if next_state.turn_state_extracted:
        return next_state

    fallback = StateExtractionOutput(
        species=next_state.species,
        symptoms=list(next_state.assessment.symptoms),
        duration=next_state.assessment.duration,
        course_pattern=next_state.assessment.course_pattern,
        current_status=next_state.current_status.model_copy(deep=True),
        negated_findings=[],
        uncertain_findings=[],
    )

    try:
        output = call_structured_output(
            system_prompt=load_prompt("state_extraction"),
            user_prompt=_state_prompt_payload(next_state),
            output_model=StateExtractionOutput,
            fallback=fallback,
            client=llm_client,
        )
    except Exception:
        output = fallback

    next_state.species = output.species
    next_state.assessment = next_state.assessment.model_copy(
        update={
            "symptoms": list(output.symptoms),
            "duration": output.duration,
            "course_pattern": output.course_pattern,
        },
        deep=True,
    )
    next_state.current_status = output.current_status.model_copy(deep=True)
    return next_state


def state_updater(
    state: PetCareGraphState,
    *,
    llm_client: StructuredOutputClient | None = None,
) -> PetCareGraphState:
    """LangGraph-friendly alias for the state updater node."""

    return update_state_from_user_input(state, llm_client=llm_client)


def _state_prompt_payload(state: PetCareGraphState) -> str:
    payload = {
        "user_input": state.user_input,
        "conversation_history": [
            message.model_dump(mode="json") for message in state.conversation_history[-6:]
        ],
        "existing_species": state.species,
        "existing_assessment": state.assessment.model_dump(mode="json"),
        "existing_current_status": state.current_status.model_dump(mode="json"),
        "locale": state.locale,
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)

