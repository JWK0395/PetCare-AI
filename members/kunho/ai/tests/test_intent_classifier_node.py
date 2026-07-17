from __future__ import annotations

from pydantic import BaseModel

from petcare_agent.nodes.intent_classifier import classify_intent
from petcare_agent.schemas.graph_state import CurrentStatus, PetCareGraphState
from petcare_agent.schemas.llm_outputs import (
    SocialChatOutput,
    StateExtractionOutput,
    TurnUnderstandingOutput,
)


class FakeLLMClient:
    def __init__(self, output: BaseModel | None = None, error: Exception | None = None) -> None:
        self.output = output
        self.error = error
        self.calls = 0

    def structured_output(self, **kwargs):
        self.calls += 1
        if self.error is not None:
            raise self.error
        return self.output


def _turn_output(
    *,
    intent: str,
    confidence: str = "high",
    chief_complaint: str | None = None,
    requires_db_context: bool = False,
    requires_safety_screening: bool = False,
    red_flag_mentioned: bool = False,
    state: StateExtractionOutput | None = None,
    social_chat: SocialChatOutput | None = None,
) -> TurnUnderstandingOutput:
    return TurnUnderstandingOutput(
        intent=intent,
        confidence=confidence,
        chief_complaint=chief_complaint,
        requires_db_context=requires_db_context,
        requires_safety_screening=requires_safety_screening,
        red_flag_mentioned=red_flag_mentioned,
        state=state
        or StateExtractionOutput(
            species="unknown",
            symptoms=[],
            duration=None,
            course_pattern="unknown",
            current_status=CurrentStatus(),
            negated_findings=[],
            uncertain_findings=[],
        ),
        social_chat=social_chat,
    )


def test_intent_classifier_applies_general_chat_mock_output() -> None:
    client = FakeLLMClient(
        _turn_output(
            intent="general_chat",
            requires_db_context=False,
            requires_safety_screening=False,
        )
    )
    state = PetCareGraphState(user_input="How often can cats eat treats?")

    result = classify_intent(state, llm_client=client)

    assert result.intent == "general_chat"
    assert result.confidence == "high"
    assert result.requires_db_context is False
    assert result.requires_safety_screening is False
    assert result.red_flag_mentioned is False
    assert result.turn_state_extracted is True
    assert client.calls == 1


def test_intent_classifier_applies_social_chat_mock_output() -> None:
    client = FakeLLMClient(
        _turn_output(
            intent="social_chat",
            requires_db_context=False,
            requires_safety_screening=False,
            social_chat=SocialChatOutput(assistant_message="장건호님이라고 알려주셨어요."),
        )
    )
    state = PetCareGraphState(user_input="내 이름은 장건호야")

    result = classify_intent(state, llm_client=client)

    assert result.intent == "social_chat"
    assert result.confidence == "high"
    assert result.requires_db_context is False
    assert result.requires_safety_screening is False
    assert result.red_flag_mentioned is False
    assert result.chat_response == "장건호님이라고 알려주셨어요."
    assert result.social_response_ready is True
    assert client.calls == 1


def test_intent_classifier_applies_symptom_check_safety_flags() -> None:
    client = FakeLLMClient(
        _turn_output(
            intent="symptom_check",
            chief_complaint="cough",
            requires_db_context=True,
            requires_safety_screening=True,
            state=StateExtractionOutput(
                species="cat",
                symptoms=["coughing"],
                duration="today",
                course_pattern="new",
                current_status=CurrentStatus(symptoms=["coughing"]),
                negated_findings=[],
                uncertain_findings=[],
            ),
        )
    )
    state = PetCareGraphState(user_input="My cat is coughing today.")

    result = classify_intent(state, llm_client=client)

    assert result.intent == "symptom_check"
    assert result.requires_db_context is True
    assert result.requires_safety_screening is True
    assert result.emergency_screening.chief_complaint == "cough"
    assert result.species == "cat"
    assert result.assessment.symptoms == ["coughing"]
    assert result.assessment.duration == "today"
    assert result.current_status.symptoms == ["coughing"]


def test_intent_classifier_uses_unknown_fallback_when_llm_fails() -> None:
    client = FakeLLMClient(error=RuntimeError("mock failure"))
    state = PetCareGraphState(user_input="Something is odd.")

    result = classify_intent(state, llm_client=client)

    assert result.intent == "unknown"
    assert result.confidence == "low"
    assert result.requires_db_context is False
    assert result.requires_safety_screening is True
    assert result.red_flag_mentioned is False