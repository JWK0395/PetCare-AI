from __future__ import annotations

import json

from pydantic import BaseModel

from petcare_agent.harness.adapter import AgentSessionConfig
from petcare_agent.harness.adapters.current_graph import CurrentAssessmentGraphAdapter
from petcare_agent.nodes.db_context_loader import StaticDBContextProvider
from petcare_agent.schemas.graph_state import CurrentStatus
from petcare_agent.schemas.llm_outputs import (
    AnswerGuardReviewOutput,
    GeneralPetCareAnswerOutput,
    SocialChatOutput,
    StateExtractionOutput,
    TurnUnderstandingOutput,
)


class HarnessFakeLLM:
    def __init__(self) -> None:
        self.calls: list[type[BaseModel]] = []

    def structured_output(self, **kwargs):
        output_model = kwargs["output_model"]
        self.calls.append(output_model)
        if output_model is TurnUnderstandingOutput:
            return _turn_output(intent="general_chat")
        if output_model is GeneralPetCareAnswerOutput:
            return GeneralPetCareAnswerOutput(
                assistant_message="General pet-care answer."
            )
        if output_model is AnswerGuardReviewOutput:
            return AnswerGuardReviewOutput(
                status="passed",
                unsafe_phrases=[],
                revised_answer=None,
            )
        raise AssertionError(f"Unexpected output model: {output_model}")


class HarnessSocialFakeLLM:
    def __init__(self) -> None:
        self.calls: list[type[BaseModel]] = []

    def structured_output(self, **kwargs):
        output_model = kwargs["output_model"]
        self.calls.append(output_model)
        if output_model is TurnUnderstandingOutput:
            payload = json.loads(kwargs["user_prompt"])
            user_input = payload["user_input"]
            known_name = _latest_name_from_payload(payload)
            if "뭐라고" in user_input and known_name:
                message = f"{known_name}님이라고 알려주셨어요."
            elif "장건호" in user_input:
                message = "반갑습니다, 장건호님."
            else:
                message = "안녕하세요."
            return _turn_output(
                intent="social_chat",
                social_chat=SocialChatOutput(assistant_message=message),
            )
        if output_model is AnswerGuardReviewOutput:
            raise AssertionError("AnswerGuardReviewOutput was not expected for social chat")
        raise AssertionError(f"Unexpected output model: {output_model}")


def _turn_output(
    *,
    intent: str,
    social_chat: SocialChatOutput | None = None,
) -> TurnUnderstandingOutput:
    return TurnUnderstandingOutput(
        intent=intent,
        confidence="high",
        chief_complaint=None,
        requires_db_context=False,
        requires_safety_screening=False,
        red_flag_mentioned=False,
        state=StateExtractionOutput(
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


def _latest_name_from_payload(payload: dict[str, object]) -> str:
    for message in reversed(payload.get("conversation_history", [])):
        if not isinstance(message, dict) or message.get("role") != "user":
            continue
        content = str(message.get("content") or "")
        if "장건호" in content:
            return "장건호"
    return ""


def test_current_graph_adapter_keeps_stateful_chat_contract() -> None:
    session = CurrentAssessmentGraphAdapter().start_session(
        config=AgentSessionConfig(
            pet_id=1,
            conversation_id="test_conv",
        ),
        context_provider=StaticDBContextProvider(),
        rag_adapter=None,
        llm_client=HarnessFakeLLM(),
    )

    first = session.handle_user_message("How often can cats eat treats?")
    second = session.handle_user_message("What about kittens?")

    assert first.response.route == "answer_guard"
    assert second.response.route == "answer_guard"
    assert first.trace_path == [
        "db_context_loader",
        "intent_classifier",
        "evidence_planner",
        "rag_agent",
        "answer_composer",
        "answer_guard",
    ]
    assert len(first.state.conversation_history) == 2
    assert len(second.state.conversation_history) == 4
    assert second.state.request_id == "req_test_conv_0002"


def test_current_graph_adapter_keeps_social_chat_context_across_turns() -> None:
    session = CurrentAssessmentGraphAdapter().start_session(
        config=AgentSessionConfig(
            pet_id=1,
            conversation_id="social_conv",
        ),
        context_provider=StaticDBContextProvider(),
        rag_adapter=None,
        llm_client=HarnessSocialFakeLLM(),
    )

    first = session.handle_user_message("안녕")
    second = session.handle_user_message("내 이름은 장건호야")
    third = session.handle_user_message("내 이름이 뭐라고?")

    assert first.response.route == "chat"
    assert second.response.route == "chat"
    assert third.response.route == "chat"
    assert third.response.assistant_message == "장건호님이라고 알려주셨어요."
    assert len(third.state.conversation_history) == 6
    assert third.state.conversation_history[-2].content == "내 이름이 뭐라고?"
    assert third.trace_path == ["db_context_loader", "intent_classifier", "chat_agent"]