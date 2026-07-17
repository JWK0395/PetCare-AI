from __future__ import annotations

import json

from pydantic import BaseModel

from petcare_agent.nodes.answer_composer import compose_answer
from petcare_agent.schemas.graph_state import (
    AssessmentState,
    ChangeDetection,
    CurrentStatus,
    PetCareGraphState,
    RAGCitation,
    RetrievedChunk,
    RetrievalState,
)
from petcare_agent.schemas.llm_outputs import GeneralPetCareAnswerOutput


class AnswerComposerFakeLLM:
    def __init__(
        self,
        output: BaseModel | None = None,
        error: Exception | None = None,
    ) -> None:
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


def test_answer_composer_uses_safety_context_and_cornell_citations() -> None:
    chunk = RetrievedChunk(
        chunk_id="cornell_dog_vomiting_001",
        source_id="cornell_dog_vomiting",
        title="Vomiting in dogs",
        text="Official Cornell source text.",
        score=0.88,
        metadata={
            "provider": "cornell",
            "canonical_url": "https://www.vet.cornell.edu/dog-vomiting",
            "section_path": ["Vomiting in dogs"],
        },
    )
    state = PetCareGraphState(
        intent="symptom_check",
        risk_level="non_emergency",
        assessment=AssessmentState(symptoms=["vomiting"]),
        current_status=CurrentStatus(symptoms=["vomiting"]),
        change_detection=ChangeDetection(summary="New symptoms reported: vomiting."),
        retrieval=RetrievalState(
            query="dog vomiting",
            chunks=[chunk],
            citations=[
                RAGCitation(
                    number=1,
                    title=chunk.title,
                    url="https://www.vet.cornell.edu/dog-vomiting",
                    chunk_id=chunk.chunk_id,
                    section_path=["Vomiting in dogs"],
                )
            ],
            provider="cornell",
        ),
    )

    result = compose_answer(state)

    assert "즉시 응급 신호" in result.chat_response
    assert "최근 기록 비교: 새로 보고된 증상: 구토." in result.chat_response
    assert "공식 자료 확인" in result.chat_response
    assert "현재 확인된 증상: 구토." in result.chat_response
    assert "Cornell 출처:" in result.chat_response
    assert "https://www.vet.cornell.edu/dog-vomiting" in result.chat_response
    assert result.next_route == "answer_composer"


def test_answer_composer_uses_llm_for_general_chat_answer() -> None:
    client = AnswerComposerFakeLLM(
        GeneralPetCareAnswerOutput(
            assistant_message="강아지는 보통 하루 1~2회 이상 산책하되 나이와 체력에 맞춰 조절하세요."
        )
    )
    state = PetCareGraphState(
        user_input="강아지 산책은 하루에 몇 번이나 해야돼?",
        intent="general_chat",
        risk_level="unknown",
        retrieval=RetrievalState(
            query="강아지 산책",
            insufficient_evidence=True,
        ),
    )

    result = compose_answer(state, llm_client=client)
    prompt_payload = json.loads(client.user_prompts[0])

    assert client.calls == [GeneralPetCareAnswerOutput]
    assert prompt_payload["user_input"] == "강아지 산책은 하루에 몇 번이나 해야돼?"
    assert result.chat_response.startswith("강아지는 보통 하루 1~2회")
    assert "병원 방문" not in result.chat_response
    assert "Cornell 출처:" not in result.chat_response
    assert result.next_route == "answer_composer"


def test_answer_composer_general_chat_falls_back_without_inventing_sources() -> None:
    state = PetCareGraphState(
        intent="general_chat",
        risk_level="unknown",
        retrieval=RetrievalState(
            query="very narrow unsupported question",
            insufficient_evidence=True,
        ),
    )

    result = compose_answer(
        state,
        llm_client=AnswerComposerFakeLLM(error=RuntimeError("mock")),
    )

    assert "반려동물 케어 질문" in result.chat_response
    assert "Cornell 출처:" not in result.chat_response