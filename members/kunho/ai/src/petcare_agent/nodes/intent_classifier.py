"""Turn understanding structured-output node."""

from __future__ import annotations

import json

from petcare_agent.llm.client import StructuredOutputClient, call_structured_output
from petcare_agent.prompts import load_prompt
from petcare_agent.schemas.graph_state import PetCareGraphState
from petcare_agent.schemas.llm_outputs import (
    IntentClassificationOutput,
    StateExtractionOutput,
    TurnUnderstandingOutput,
)


INTENT_FALLBACK = IntentClassificationOutput(
    intent="unknown",
    confidence="low",
    chief_complaint=None,
    requires_db_context=False,
    requires_safety_screening=True,
    red_flag_mentioned=False,
)


def classify_intent(
    state: PetCareGraphState,
    *,
    llm_client: StructuredOutputClient | None = None,
) -> PetCareGraphState:
    """Understand one turn and prefill routing, state, and social response fields."""

    next_state = state.model_copy(deep=True)
    fallback = _turn_understanding_fallback(next_state)
    try:
        output = call_structured_output(
            system_prompt=load_prompt("turn_understanding"),
            user_prompt=_turn_understanding_prompt_payload(next_state),
            output_model=TurnUnderstandingOutput,
            fallback=fallback,
            client=llm_client,
        )
    except Exception:
        output = fallback

    _apply_turn_understanding(next_state, output)
    return next_state


def intent_classifier(
    state: PetCareGraphState,
    *,
    llm_client: StructuredOutputClient | None = None,
) -> PetCareGraphState:
    """LangGraph-friendly alias for the turn understanding node."""

    return classify_intent(state, llm_client=llm_client)


def _turn_understanding_fallback(state: PetCareGraphState) -> TurnUnderstandingOutput:
    return TurnUnderstandingOutput(
        intent=INTENT_FALLBACK.intent,
        confidence=INTENT_FALLBACK.confidence,
        chief_complaint=INTENT_FALLBACK.chief_complaint,
        requires_db_context=INTENT_FALLBACK.requires_db_context,
        requires_safety_screening=INTENT_FALLBACK.requires_safety_screening,
        red_flag_mentioned=INTENT_FALLBACK.red_flag_mentioned,
        state=StateExtractionOutput(
            species=state.species,
            symptoms=list(state.assessment.symptoms),
            duration=state.assessment.duration,
            course_pattern=state.assessment.course_pattern,
            current_status=state.current_status.model_copy(deep=True),
            negated_findings=[],
            uncertain_findings=[],
        ),
        social_chat=None,
    )


def _apply_turn_understanding(
    state: PetCareGraphState,
    output: TurnUnderstandingOutput,
) -> None:
    state.intent = output.intent
    state.confidence = output.confidence
    state.requires_db_context = output.requires_db_context
    state.requires_safety_screening = output.requires_safety_screening
    state.red_flag_mentioned = output.red_flag_mentioned

    if output.chief_complaint:
        state.emergency_screening.chief_complaint = output.chief_complaint

    state.species = output.state.species
    state.assessment = state.assessment.model_copy(
        update={
            "symptoms": list(output.state.symptoms),
            "duration": output.state.duration,
            "course_pattern": output.state.course_pattern,
        },
        deep=True,
    )
    state.current_status = output.state.current_status.model_copy(deep=True)
    state.turn_state_extracted = True

    state.social_response_ready = False
    if output.intent == "social_chat" and output.social_chat is not None:
        assistant_message = output.social_chat.assistant_message.strip()
        if assistant_message:
            state.chat_response = assistant_message
            state.social_response_ready = True


def _turn_understanding_prompt_payload(state: PetCareGraphState) -> str:
    payload = {
        "user_input": state.user_input,
        "conversation_history": [
            message.model_dump(mode="json") for message in state.conversation_history[-12:]
        ],
        "locale": state.locale,
        "existing_intent": state.intent,
        "existing_species": state.species,
        "existing_assessment": state.assessment.model_dump(mode="json"),
        "existing_current_status": state.current_status.model_dump(mode="json"),
        "pet_context": state.context.pet,
        "pending_question_item_ids": list(state.emergency_screening.missing_questions),
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def _intent_prompt_payload(state: PetCareGraphState) -> str:
    """Backward-compatible prompt payload helper for older direct tests."""

    return _turn_understanding_prompt_payload(state)