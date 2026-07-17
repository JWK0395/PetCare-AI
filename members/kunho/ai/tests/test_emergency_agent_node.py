from __future__ import annotations

from petcare_agent.nodes.emergency_agent import generate_emergency_response
from petcare_agent.schemas.graph_state import (
    AssessmentState,
    ChangeDetection,
    EmergencyScreening,
    PetCareGraphState,
)
from petcare_agent.schemas.triage import RuleHit


def _emergency_state() -> PetCareGraphState:
    return PetCareGraphState(
        risk_level="emergency",
        assessment=AssessmentState(symptoms=["open mouth breathing"]),
        change_detection=ChangeDetection(summary="Worsened compared with baseline: activity."),
        emergency_screening=EmergencyScreening(
            red_flags=["open_mouth_breathing"],
            triggered_rules=[
                RuleHit(
                    rule_id="E_RESP_001",
                    result="emergency",
                    condition="open_mouth_breathing == true and species == cat",
                )
            ],
        ),
    )


def test_emergency_agent_generates_immediate_care_guidance() -> None:
    result = generate_emergency_response(_emergency_state())

    assert "응급 신호가 있을 수 있습니다" in result.chat_response
    assert "즉시 진료" in result.chat_response
    assert result.next_route == "emergency"


def test_emergency_agent_does_not_ask_hospital_visit_intent_question() -> None:
    result = generate_emergency_response(_emergency_state())
    message = result.chat_response

    assert "병원 방문을 고려" not in message
    assert "그렇다면" not in message
    assert "아직 결정" not in message
    assert result.hospital_visit_intent == "not_asked"


def test_emergency_agent_reflects_rules_red_flags_and_change_summary() -> None:
    result = generate_emergency_response(_emergency_state())

    assert "E_RESP_001" in result.chat_response
    assert "입을 벌리고 호흡" in result.chat_response
    assert "최근 기준보다 나빠진 항목: 활동량." in result.chat_response


def test_emergency_agent_does_not_perform_external_work() -> None:
    result = generate_emergency_response(_emergency_state())

    assert result.retrieval.query == ""
    assert result.retrieval.chunks == []
    assert result.handoff.email_draft == ""