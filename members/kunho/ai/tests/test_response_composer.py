from __future__ import annotations

from petcare_agent.graphs.response_composer import compose_graph_response
from petcare_agent.schemas.graph_state import (
    EmergencyScreening,
    HandoffState,
    PetCareGraphState,
)
from petcare_agent.schemas.triage import ChecklistItem, RuleHit


def test_response_composer_returns_graph_response_contract() -> None:
    state = PetCareGraphState(
        request_id="req_20260716_001",
        conversation_id="conv_abc",
        risk_level="non_emergency",
        next_route="chat",
        chat_response="현재 정보만으로는 즉시 응급 신호가 뚜렷하지 않습니다.",
    )

    response = compose_graph_response(state)

    assert response.response_id == "res_20260716_001"
    assert response.conversation_id == "conv_abc"
    assert response.route == "chat"
    assert response.assistant_message == state.chat_response
    assert response.model_dump(mode="json")["handoff"] == {
        "type": "none",
        "summary": None,
        "summary_json": None,
        "email_draft": None,
    }


def test_response_composer_question_manager_follow_up_fields() -> None:
    item = ChecklistItem(
        item_id="open_mouth_breathing",
        label="Open mouth breathing",
        type="boolean",
        value=None,
        confidence="unknown",
        source="user_input",
        asked_count=1,
        question_text="반려동물이 지금 입을 벌리고 숨을 쉬고 있나요?",
        priority=1,
        metadata={"red_flag": True},
    )
    state = PetCareGraphState(
        conversation_id="conv_question",
        next_route="question_manager",
        emergency_screening=EmergencyScreening(
            items={"open_mouth_breathing": item},
            missing_questions=["open_mouth_breathing"],
        ),
    )

    response = compose_graph_response(state)

    assert response.needs_user_response is True
    assert response.follow_up_question is not None
    assert response.follow_up_question.question_id == "resp_open_mouth_breathing"
    assert response.follow_up_question.text == item.question_text
    assert response.assistant_message == item.question_text


def test_response_composer_emergency_route_reflects_emergency_fields() -> None:
    state = PetCareGraphState(
        conversation_id="conv_emergency",
        risk_level="emergency",
        next_route="emergency",
        chat_response="응급 신호가 있을 수 있습니다.",
        emergency_screening=EmergencyScreening(
            triggered_rules=[
                RuleHit(
                    rule_id="E_RESP_001",
                    result="emergency",
                    condition="open_mouth_breathing == true and species == cat",
                )
            ]
        ),
    )

    response = compose_graph_response(state)

    assert response.route == "emergency"
    assert response.emergency.is_emergency is True
    assert response.emergency.triggered_rules == ["E_RESP_001"]


def test_response_composer_preserves_handoff_contract_fields() -> None:
    state = PetCareGraphState(
        conversation_id="conv_handoff",
        risk_level="urgent",
        next_route="handoff",
        chat_response="병원 전달용 요약 초안입니다.",
        handoff=HandoffState(
            type="non_emergency",
            required=True,
            summary="병원 전달용 요약 초안입니다.",
            email_draft="초안입니다 - 이메일은 전송되지 않았습니다.",
        ),
    )

    response = compose_graph_response(state)

    assert response.handoff.type == "non_emergency"
    assert response.handoff.summary == "병원 전달용 요약 초안입니다."
    assert response.handoff.summary_json is None
    assert response.handoff.email_draft == "초안입니다 - 이메일은 전송되지 않았습니다."

