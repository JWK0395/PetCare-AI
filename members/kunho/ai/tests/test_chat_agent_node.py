from __future__ import annotations

import json

from pydantic import BaseModel

from petcare_agent.nodes.chat_agent import (
    generate_chat_response,
    generate_social_chat_response,
)
from petcare_agent.nodes.evidence_planner import plan_evidence_context
from petcare_agent.schemas.graph_state import (
    AssessmentState,
    ConversationMessage,
    EmergencyScreening,
    PetCareContext,
    PetCareGraphState,
    RetrievedChunk,
    RetrievalState,
)
from petcare_agent.schemas.llm_outputs import SocialChatOutput


class SocialChatFakeLLM:
    def __init__(self, output: BaseModel | None = None, error: Exception | None = None) -> None:
        self.output = output
        self.error = error
        self.calls: list[type[BaseModel]] = []
        self.user_prompts: list[str] = []

    def structured_output(self, **kwargs):
        self.calls.append(kwargs["output_model"])
        self.user_prompts.append(kwargs["user_prompt"])
        if self.error is not None:
            raise self.error
        return self.output


def test_evidence_planner_builds_retrieval_query_without_drafting_answer() -> None:
    state = PetCareGraphState(
        user_input="My dog vomited twice today.",
        risk_level="urgent",
        species="dog",
        assessment=AssessmentState(symptoms=["vomiting"]),
        emergency_screening=EmergencyScreening(chief_complaint="vomiting"),
    )

    result = plan_evidence_context(state)

    assert result.chat_response == ""
    assert result.retrieval.query == (
        "My dog vomited twice today. species:dog risk_level:urgent "
        "chief_complaint:vomiting symptoms:vomiting"
    )
    assert result.retrieval.chunks == []
    assert result.next_route == "evidence_planner"


def test_evidence_planner_preserves_existing_query_and_chunks() -> None:
    chunk = RetrievedChunk(
        chunk_id="chunk_existing",
        source_id="source_existing",
        title="Existing source",
        text="Existing source text.",
    )
    state = PetCareGraphState(
        retrieval=RetrievalState(query="existing query", chunks=[chunk]),
    )

    result = plan_evidence_context(state)

    assert result.retrieval.query == "existing query"
    assert result.retrieval.chunks == [chunk]


def test_chat_agent_import_path_remains_evidence_planner_compatible() -> None:
    state = PetCareGraphState(user_input="Can cats eat treats?", risk_level="unknown")

    result = generate_chat_response(state)

    assert result.chat_response == ""
    assert result.retrieval.query.startswith("Can cats eat treats?")
    assert result.next_route == "evidence_planner"


def test_social_chat_response_uses_llm_with_conversation_history_and_pet_context() -> None:
    client = SocialChatFakeLLM(
        SocialChatOutput(assistant_message="\ucd08\ucf54\ub77c\uace0 \uc54c\ub824\uc8fc\uc168\uc5b4\uc694.")
    )
    state = PetCareGraphState(
        user_input="\ub0b4 \uac15\uc544\uc9c0 \uc774\ub984\uc740 \ubb50\uc57c?",
        conversation_history=[
            ConversationMessage(role="user", content="\ub0b4 \uc774\ub984\uc740 \uc7a5\uac74\ud638\uc57c"),
            ConversationMessage(role="assistant", content="\ubc18\uac11\uc2b5\ub2c8\ub2e4, \uc7a5\uac74\ud638\ub2d8."),
        ],
        context=PetCareContext(pet={"id": 1, "name": "\ucd08\ucf54", "species": "dog"}),
        retrieval=RetrievalState(query="stale query"),
    )

    result = generate_social_chat_response(state, llm_client=client)
    prompt_payload = json.loads(client.user_prompts[0])

    assert client.calls == [SocialChatOutput]
    assert prompt_payload["conversation_history"][0]["content"] == "\ub0b4 \uc774\ub984\uc740 \uc7a5\uac74\ud638\uc57c"
    assert prompt_payload["pet_context"] == {"id": 1, "name": "\ucd08\ucf54", "species": "dog"}
    assert result.next_route == "chat"
    assert result.retrieval.query == ""
    assert result.chat_response == "\ucd08\ucf54\ub77c\uace0 \uc54c\ub824\uc8fc\uc168\uc5b4\uc694."
    assert "Cornell" not in result.chat_response


def test_social_chat_response_fallback_is_generic_when_llm_fails() -> None:
    state = PetCareGraphState(
        user_input="\ub0b4 \uac15\uc544\uc9c0 \uc774\ub984\uc740 \ubb50\uc57c?",
        retrieval=RetrievalState(query="stale query"),
    )

    result = generate_social_chat_response(
        state,
        llm_client=SocialChatFakeLLM(error=RuntimeError("mock")),
    )

    assert result.next_route == "chat"
    assert result.retrieval.query == ""
    assert "Cornell" not in result.chat_response
