from __future__ import annotations

import re
import time
from typing import Any

from langgraph.graph import END, START, StateGraph

from ..models import (
    AssessmentOutput,
    PetCareState,
    SymptomItem,
)
from ..utils import node_result
from .context import prepare_backend_context
from .safety import raw_keyword_hits
from .triage import (
    detect_symptom_items,
    should_open_new_triage,
)


HANDOFF_REQUEST_PATTERNS = [
    r"병원\s*전달용",
    r"병원에\s*(?:전달|보여|제출)",
    r"병원용\s*(?:요약|정리|문서)",
    r"전달용\s*(?:요약|정리|문서)",
    r"진료용\s*(?:요약|정리|문서)",
    r"수의사(?:에게|한테)\s*(?:전달|보여)",
    r"병원에서\s*볼\s*수\s*있게",
]


def detect_handoff_request(text: str) -> bool:
    normalized = text.strip()
    return any(
        re.search(pattern, normalized)
        for pattern in HANDOFF_REQUEST_PATTERNS
    )


def _symptom_payload(text: str) -> list[SymptomItem]:
    items = detect_symptom_items(text)
    known_codes = {
        item.code
        for item in items
        if not item.negated
    }

    for code in sorted(raw_keyword_hits(text)):
        if code in known_codes:
            continue

        items.append(
            SymptomItem(
                code=code,
                evidence=text.strip(),
                negated=False,
            )
        )
        known_codes.add(code)

    if (
        not any(not item.negated for item in items)
        and should_open_new_triage(text)
    ):
        items.append(
            SymptomItem(
                code="other",
                evidence=text.strip(),
                negated=False,
            )
        )

    return items


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

    handoff_requested = detect_handoff_request(user_input)
    symptoms = _symptom_payload(user_input)
    positive_symptoms = [
        item
        for item in symptoms
        if not item.negated
    ]

    health_related = (
        _is_active_health_context(state)
        or bool(positive_symptoms)
        or should_open_new_triage(user_input)
    )

    intent = (
        "health_related"
        if health_related
        else "general_chat"
    )

    if handoff_requested:
        user_goal = "병원 전달용 문서 생성"
    elif health_related:
        user_goal = "반려동물 건강 상태 확인"
    else:
        user_goal = "일반 대화 또는 등록 기록 조회"

    assessment = AssessmentOutput(
        intent=intent,
        handoff_requested=handoff_requested,
        user_goal=user_goal,
        symptoms=symptoms,
    )

    updates: dict[str, Any] = {
        "assessment": assessment.model_dump(),
        "handoff_requested": handoff_requested,
    }

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
