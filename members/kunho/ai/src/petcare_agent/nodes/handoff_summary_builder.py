"""Handoff summary structured-output node."""

from __future__ import annotations

import json

from petcare_agent.llm.client import StructuredOutputClient, call_structured_output
from petcare_agent.prompts import load_prompt
from petcare_agent.schemas.graph_state import PetCareGraphState
from petcare_agent.schemas.llm_outputs import HandoffSummaryOutput


def build_handoff_summary(
    state: PetCareGraphState,
    *,
    llm_client: StructuredOutputClient | None = None,
) -> PetCareGraphState:
    """Build handoff summary/email draft text without sending anything."""

    next_state = state.model_copy(deep=True)
    try:
        output = call_structured_output(
            system_prompt=load_prompt("handoff_summary"),
            user_prompt=_handoff_prompt_payload(next_state),
            output_model=HandoffSummaryOutput,
            client=llm_client,
        )
    except Exception:
        return next_state

    next_state.handoff.type = output.type
    next_state.handoff.summary = output.summary
    next_state.handoff.email_draft = output.email_draft or ""
    return next_state


def handoff_summary_builder(
    state: PetCareGraphState,
    *,
    llm_client: StructuredOutputClient | None = None,
) -> PetCareGraphState:
    """LangGraph-friendly alias for the handoff summary node."""

    return build_handoff_summary(state, llm_client=llm_client)


def _handoff_prompt_payload(state: PetCareGraphState) -> str:
    payload = {
        "user_input": state.user_input,
        "species": state.species,
        "risk_level": state.risk_level,
        "confidence": state.confidence,
        "assessment": state.assessment.model_dump(mode="json"),
        "current_status": state.current_status.model_dump(mode="json"),
        "baseline_context": state.baseline_context.model_dump(mode="json"),
        "change_detection": state.change_detection.model_dump(mode="json"),
        "emergency_screening": state.emergency_screening.model_dump(mode="json"),
        "handoff": state.handoff.model_dump(mode="json"),
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)
