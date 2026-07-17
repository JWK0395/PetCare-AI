"""Emergency response skeleton node."""

from __future__ import annotations

from petcare_agent.localization import display_list, localize_change_summary, wants_korean
from petcare_agent.schemas.graph_state import PetCareGraphState
from petcare_agent.schemas.triage import RuleHit


EMERGENCY_GUIDANCE = (
    "Emergency signs may be present. Please contact an emergency veterinary clinic "
    "or go for immediate veterinary care now. Do not wait for chat confirmation before "
    "getting help."
)

EMERGENCY_GUIDANCE_KO = (
    "응급 신호가 있을 수 있습니다. 지금 바로 응급 동물병원에 연락하거나 "
    "즉시 진료를 받아 주세요. 채팅 답변을 더 기다리지 말고 도움을 받는 것이 우선입니다."
)


def generate_emergency_response(state: PetCareGraphState) -> PetCareGraphState:
    """Create immediate-care guidance for emergency cases without external calls."""

    next_state = state.model_copy(deep=True)
    if next_state.risk_level != "emergency":
        return next_state

    korean = wants_korean(next_state.locale)
    message_parts = [EMERGENCY_GUIDANCE_KO if korean else EMERGENCY_GUIDANCE]

    rule_ids = _triggered_rule_ids(next_state.emergency_screening.triggered_rules)
    if rule_ids:
        if korean:
            message_parts.append(f"작동한 안전 규칙: {', '.join(rule_ids)}.")
        else:
            message_parts.append(f"Triggered safety rules: {', '.join(rule_ids)}.")

    red_flags = display_list(
        _clean_values(next_state.emergency_screening.red_flags),
        next_state.locale,
    )
    if red_flags:
        if korean:
            message_parts.append(f"보고된 위험 신호: {', '.join(red_flags)}.")
        else:
            message_parts.append(f"Reported red flags: {', '.join(red_flags)}.")

    change_summary = localize_change_summary(
        next_state.change_detection.summary,
        next_state.locale,
    )
    if change_summary:
        if korean:
            message_parts.append(f"최근 기록 비교: {change_summary}")
        else:
            message_parts.append(f"Recent log comparison: {change_summary}")

    symptoms = display_list(
        _clean_values(next_state.assessment.symptoms or next_state.current_status.symptoms),
        next_state.locale,
    )
    if symptoms:
        if korean:
            message_parts.append(f"보고된 증상: {', '.join(symptoms)}.")
        else:
            message_parts.append(f"Reported symptoms: {', '.join(symptoms)}.")

    next_state.chat_response = "\n\n".join(message_parts)
    next_state.next_route = "emergency"
    return next_state


def emergency_agent(state: PetCareGraphState) -> PetCareGraphState:
    """LangGraph-friendly alias for the emergency agent node."""

    return generate_emergency_response(state)


def _triggered_rule_ids(triggered_rules: list[RuleHit]) -> list[str]:
    return _clean_values([rule.rule_id for rule in triggered_rules])


def _clean_values(values: list[str]) -> list[str]:
    cleaned: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = " ".join(value.strip().split())
        if not normalized or normalized in seen:
            continue
        cleaned.append(normalized)
        seen.add(normalized)
    return cleaned


__all__ = ["generate_emergency_response", "emergency_agent"]