"""Compose public graph responses from internal graph state."""

from __future__ import annotations

from petcare_agent.localization import wants_korean
from petcare_agent.schemas.common import NodeRoute
from petcare_agent.schemas.graph_state import (
    FollowUpQuestion,
    GraphResponse,
    HandoffResponse,
    PetCareGraphState,
)


def compose_graph_response(
    state: PetCareGraphState,
    *,
    route: NodeRoute | None = None,
) -> GraphResponse:
    """Convert graph state into the existing GraphResponse contract."""

    response_route = route or state.next_route
    follow_up_question = _follow_up_question(state, response_route)
    assistant_message = _assistant_message(state, response_route, follow_up_question)

    return GraphResponse(
        response_id=_response_id(state),
        conversation_id=state.conversation_id or "unknown_conversation",
        route=response_route,
        risk_level=state.risk_level,
        assistant_message=assistant_message,
        needs_user_response=response_route == "question_manager" and follow_up_question is not None,
        follow_up_question=follow_up_question,
        handoff=HandoffResponse(
            type=state.handoff.type,
            summary=state.handoff.summary or None,
            summary_json=state.handoff.summary_json,
            email_draft=state.handoff.email_draft or None,
        ),
        emergency={
            "is_emergency": state.risk_level == "emergency",
            "triggered_rules": [rule.rule_id for rule in state.emergency_screening.triggered_rules],
        },
    )


def _response_id(state: PetCareGraphState) -> str:
    if state.request_id:
        if state.request_id.startswith("req_"):
            return state.request_id.replace("req_", "res_", 1)
        return f"res_{state.request_id}"
    return "res_local"


def _follow_up_question(
    state: PetCareGraphState,
    route: NodeRoute,
) -> FollowUpQuestion | None:
    if route != "question_manager":
        return None

    for item_id in state.emergency_screening.missing_questions:
        item = state.emergency_screening.items.get(item_id)
        if item is None or not item.question_text:
            continue
        return FollowUpQuestion(question_id=f"resp_{item_id}", text=item.question_text)
    return None


def _assistant_message(
    state: PetCareGraphState,
    route: NodeRoute,
    follow_up_question: FollowUpQuestion | None,
) -> str:
    if state.chat_response.strip():
        return state.chat_response.strip()
    if route == "question_manager" and follow_up_question is not None:
        return follow_up_question.text

    korean = wants_korean(state.locale)
    if state.risk_level == "emergency":
        if korean:
            return "응급 신호가 있을 수 있습니다. 지금 바로 수의사 진료를 받아 주세요."
        return "Emergency signs may be present. Please seek immediate veterinary care now."
    if state.risk_level == "urgent":
        if korean:
            return "현재 정보상 빠른 수의사 진료 상담을 권합니다."
        return "Prompt veterinary care is recommended based on the current information."
    if state.risk_level == "non_emergency":
        if korean:
            return "현재 정보만으로는 즉시 응급 신호가 뚜렷하지 않습니다."
        return "No immediate emergency signal is clear from the current information."
    if korean:
        return "위험도를 판단하려면 정보가 더 필요합니다."
    return "More information is needed to judge the risk confidently."


__all__ = ["compose_graph_response"]