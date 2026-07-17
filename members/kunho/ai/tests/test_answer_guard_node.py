from __future__ import annotations

import pytest
from pydantic import BaseModel

from petcare_agent.nodes.answer_guard import review_answer_guard
from petcare_agent.schemas.graph_state import PetCareGraphState
from petcare_agent.schemas.llm_outputs import AnswerGuardReviewOutput


class FakeLLMClient:
    def __init__(self, output: BaseModel) -> None:
        self.output = output

    def structured_output(self, **kwargs):
        return self.output


@pytest.mark.parametrize("status", ["passed", "revised", "blocked"])
def test_answer_guard_applies_statuses(status: str) -> None:
    output = AnswerGuardReviewOutput(
        status=status,  # type: ignore[arg-type]
        unsafe_phrases=["unsafe phrase"] if status != "passed" else [],
        revised_answer=None,
    )
    state = PetCareGraphState(chat_response="Draft answer")

    result = review_answer_guard(state, llm_client=FakeLLMClient(output))

    assert result.answer_guard.status == status
    assert result.answer_guard.revisions == output.unsafe_phrases
    assert result.chat_response == "Draft answer"


def test_answer_guard_applies_revised_answer_to_chat_response() -> None:
    output = AnswerGuardReviewOutput(
        status="revised",
        unsafe_phrases=["it is fine"],
        revised_answer="Based on current information, no immediate emergency signal is clear.",
    )
    state = PetCareGraphState(chat_response="It is fine.")

    result = review_answer_guard(state, llm_client=FakeLLMClient(output))

    assert result.answer_guard.status == "revised"
    assert result.chat_response == output.revised_answer
