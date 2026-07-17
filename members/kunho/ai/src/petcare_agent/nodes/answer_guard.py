"""Answer guard structured-output node."""

from __future__ import annotations

import json

from petcare_agent.llm.client import StructuredOutputClient, call_structured_output
from petcare_agent.prompts import load_prompt
from petcare_agent.schemas.graph_state import PetCareGraphState
from petcare_agent.schemas.llm_outputs import AnswerGuardReviewOutput


def review_answer_guard(
    state: PetCareGraphState,
    *,
    llm_client: StructuredOutputClient | None = None,
) -> PetCareGraphState:
    """Review the current draft answer and apply safe revisions when provided."""

    next_state = state.model_copy(deep=True)
    fallback = AnswerGuardReviewOutput(
        status=next_state.answer_guard.status,
        unsafe_phrases=list(next_state.answer_guard.revisions),
        revised_answer=None,
    )

    try:
        output = call_structured_output(
            system_prompt=load_prompt("answer_guard"),
            user_prompt=_answer_guard_prompt_payload(next_state),
            output_model=AnswerGuardReviewOutput,
            fallback=fallback,
            client=llm_client,
        )
    except Exception:
        output = fallback

    next_state.answer_guard.status = output.status
    next_state.answer_guard.revisions = list(output.unsafe_phrases)
    if output.revised_answer:
        next_state.chat_response = output.revised_answer

    return next_state


def answer_guard(
    state: PetCareGraphState,
    *,
    llm_client: StructuredOutputClient | None = None,
) -> PetCareGraphState:
    """LangGraph-friendly alias for the answer guard node."""

    return review_answer_guard(state, llm_client=llm_client)


def _answer_guard_prompt_payload(state: PetCareGraphState) -> str:
    payload = {
        "chat_response": state.chat_response,
        "risk_level": state.risk_level,
        "confidence": state.confidence,
        "intent": state.intent,
        "assessment": state.assessment.model_dump(mode="json"),
        "emergency_screening": state.emergency_screening.model_dump(mode="json"),
        "change_detection": state.change_detection.model_dump(mode="json"),
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)
