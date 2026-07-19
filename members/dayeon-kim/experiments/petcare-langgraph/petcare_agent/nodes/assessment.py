from __future__ import annotations

import re
import time
from typing import Any

from langgraph.graph import END, START, StateGraph

from ..models import PetCareState
from ..utils import node_result
from .context import prepare_backend_context
from .safety import raw_keyword_hits
from .triage import detect_symptoms, should_open_new_triage


HANDOFF_REQUEST_PATTERNS = [
    r"병원\s*전달용",
    r"병원에\s*(?:전달|보여|제출)",
    r"병원용\s*(?:요약|정리|문서)",
    r"전달용\s*(?:요약|정리|문서)",
    r"진료용\s*(?:요약|정리|문서)",
    r"수의사(?:에게|한테)\s*(?:전달|보여)",
    r"병원에서\s*볼\s*수\s*있게",
]


POST_TRIAGE_VISIT_PATTERNS = [
    r"병원(?:에|으로)?\s*(?:갈|가|방문)",
    r"(?:그냥\s*)?(?:갈래|갈게|가야겠|가겠습니다)",
    r"(?:진료|검사)\s*(?:받으러|받을래|받을게)",
]


def detect_handoff_request(text: str) -> bool:
    normalized = text.strip()

    return any(
        re.search(pattern, normalized)
        for pattern in HANDOFF_REQUEST_PATTERNS
    )


def detect_post_triage_visit_request(
    state: PetCareState,
    text: str,
) -> bool:
    if not state.get("post_triage_mode", False):
        return False

    previous_route = state.get(
        "previous_triage",
        {},
    ).get("route")

    if previous_route != "non_emergency":
        return False

    normalized = text.strip()

    return any(
        re.search(pattern, normalized)
        for pattern in POST_TRIAGE_VISIT_PATTERNS
    )


def _symptom_payload(text: str) -> list[dict[str, Any]]:
    codes: list[str] = []

    for code in detect_symptoms(text):
        if code not in codes:
            codes.append(code)

    for code in sorted(raw_keyword_hits(text)):
        if code not in codes:
            codes.append(code)

    if not codes and should_open_new_triage(text):
        codes.append("other")

    return [
        {
            "code": code,
            "evidence": text.strip(),
            "negated": False,
        }
        for code in codes
    ]


def _is_active_health_context(state: PetCareState) -> bool:
    return (
        state.get("triage_status") == "collecting"
        or bool(state.get("follow_up_history"))
    )


def assess_input(
    state: PetCareState,
) -> dict[str, Any]:
    started = time.perf_counter()
    user_input = state.get("user_input", "").strip()

    post_triage_visit = detect_post_triage_visit_request(
        state,
        user_input,
    )
    handoff_requested = (
        detect_handoff_request(user_input)
        or post_triage_visit
    )
    symptoms = _symptom_payload(user_input)

    health_related = (
        _is_active_health_context(state)
        or bool(symptoms)
        or should_open_new_triage(user_input)
    )

    intent = (
        "health_related"
        if health_related
        else "general_chat"
    )

    if post_triage_visit:
        user_goal = "이전 상태 확인을 바탕으로 병원 전달용 문서 생성"
    elif handoff_requested:
        user_goal = "병원 전달용 문서 생성"
    elif health_related:
        user_goal = "반려동물 건강 상태 확인"
    else:
        user_goal = "일반 대화 또는 등록 기록 조회"

    assessment = {
        "intent": intent,
        "handoff_requested": handoff_requested,
        "user_goal": user_goal,
        "symptoms": symptoms,
    }

    updates: dict[str, Any] = {
        "assessment": assessment,
        "handoff_requested": handoff_requested,
    }

    if post_triage_visit:
        updates["visit_decision"] = "yes"

    if intent == "general_chat" or handoff_requested:
        updates["route"] = "general_chat"

    return node_result(
        state,
        node_name="assess_input",
        started_at=started,
        updates=updates,
    )


assessment_builder = StateGraph(PetCareState)
assessment_builder.add_node(
    "prepare_backend_context",
    prepare_backend_context,
)
assessment_builder.add_node(
    "assess_input",
    assess_input,
)
assessment_builder.add_edge(
    START,
    "prepare_backend_context",
)
assessment_builder.add_edge(
    "prepare_backend_context",
    "assess_input",
)
assessment_builder.add_edge(
    "assess_input",
    END,
)

assessment_graph = assessment_builder.compile()
