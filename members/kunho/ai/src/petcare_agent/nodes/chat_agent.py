"""LLM-backed social chat node and evidence-planning compatibility wrapper."""

from __future__ import annotations

import json

from petcare_agent.llm.client import StructuredOutputClient, call_structured_output
from petcare_agent.nodes.evidence_planner import plan_evidence_context
from petcare_agent.prompts import load_prompt
from petcare_agent.schemas.graph_state import PetCareGraphState, RetrievalState
from petcare_agent.schemas.llm_outputs import SocialChatOutput

CHAT_RISK_LEVELS = {"urgent", "non_emergency", "unknown"}
SOCIAL_CHAT_FALLBACK = (
    "I can help with conversation and pet-care questions, but I could not fully process "
    "that social chat turn. Please tell me again."
)
SOCIAL_CHAT_FALLBACK_KO = (
    "\ub300\ud654\uc640 \ubc18\ub824\ub3d9\ubb3c \uad00\ub828 \uc9c8\ubb38\uc744 "
    "\ub3c4\uc6b8 \uc218 \uc788\uc9c0\ub9cc, \ubc29\uae08 \uc0ac\ud68c\uc801 "
    "\ub300\ud654 \uc694\uccad\uc744 \ucda9\ubd84\ud788 \ucc98\ub9ac\ud558\uc9c0 "
    "\ubabb\ud588\uc5b4\uc694. \ub2e4\uc2dc \ud55c \ubc88 \ub9d0\uc500\ud574 \uc8fc\uc138\uc694."
)


def generate_chat_response(state: PetCareGraphState) -> PetCareGraphState:
    """Backward-compatible alias that prepares evidence retrieval only."""

    return plan_evidence_context(state)


def generate_social_chat_response(
    state: PetCareGraphState,
    *,
    llm_client: StructuredOutputClient | None = None,
) -> PetCareGraphState:
    """Generate a conversational response with the LLM using history and profile context."""

    next_state = state.model_copy(deep=True)
    if next_state.social_response_ready and next_state.chat_response.strip():
        next_state.retrieval = RetrievalState()
        next_state.next_route = "chat"
        return next_state

    next_state.retrieval = RetrievalState()
    fallback = SocialChatOutput(assistant_message=_fallback_social_message(next_state))

    try:
        output = call_structured_output(
            system_prompt=load_prompt("social_chat"),
            user_prompt=_social_chat_prompt_payload(next_state),
            output_model=SocialChatOutput,
            fallback=fallback,
            client=llm_client,
        )
    except Exception:
        output = fallback

    next_state.chat_response = output.assistant_message.strip() or fallback.assistant_message
    next_state.next_route = "chat"
    return next_state


def chat_agent(
    state: PetCareGraphState,
    *,
    llm_client: StructuredOutputClient | None = None,
) -> PetCareGraphState:
    """LangGraph-friendly LLM-backed node for lightweight conversational chat."""

    return generate_social_chat_response(state, llm_client=llm_client)


def _social_chat_prompt_payload(state: PetCareGraphState) -> str:
    payload = {
        "user_input": state.user_input,
        "conversation_history": [
            message.model_dump(mode="json") for message in state.conversation_history[-12:]
        ],
        "locale": state.locale,
        "pet_context": state.context.pet,
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def _fallback_social_message(state: PetCareGraphState) -> str:
    if (state.locale or "").lower().startswith("ko"):
        return SOCIAL_CHAT_FALLBACK_KO
    return SOCIAL_CHAT_FALLBACK


__all__ = [
    "CHAT_RISK_LEVELS",
    "chat_agent",
    "generate_chat_response",
    "generate_social_chat_response",
]
