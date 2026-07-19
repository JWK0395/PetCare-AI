from __future__ import annotations

import json
import re
import time
from functools import partial
from typing import Any

from langgraph.graph import END, START, StateGraph

from ..models import AssessmentOutput, PetCareState
from ..prompts import ASSESSMENT_SYSTEM_PROMPT
from ..services import (
    AgentDependencies,
    build_default_dependencies,
)
from ..utils import (
    add_error,
    format_conversation_history,
    node_result,
)
from .context import prepare_backend_context


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
    """LLM 결과와 별도로 명시적인 병원 전달 요청을 보완한다."""
    return any(
        re.search(pattern, text.strip())
        for pattern in HANDOFF_REQUEST_PATTERNS
    )


def _follow_up_payload(
    state: PetCareState,
) -> list[dict[str, Any]]:
    """현재 입력 해석에 필요한 문진 답변만 압축한다."""
    return [
        {
            "field": item.get("field"),
            "kind": item.get("kind"),
            "symptom": item.get("symptom"),
            "answer": item.get("answer"),
            "answer_status": item.get(
                "answer_status"
            ),
        }
        for item in state.get(
            "follow_up_history",
            [],
        )
    ]


def _user_history(
    state: PetCareState,
) -> list[dict[str, str]]:
    """의도 해석에 필요한 사용자 발화만 선택한다."""
    return [
        item
        for item in state.get(
            "conversation_history",
            [],
        )
        if item.get("role") == "user"
    ]


def _resolve_intent(
    state: PetCareState,
    intent: str,
) -> str:
    """진행 중인 문진의 짧은 응답을 일반 대화로 이탈시키지 않는다."""
    if (
        state.get("triage_status") == "collecting"
        or state.get("follow_up_history")
    ):
        return "health_related"

    return intent


def _assessment_prompt(state: PetCareState) -> str:
    """기존 LLM 기반 해석을 유지하되 불필요한 전체 기록은 제외한다."""
    conversation_text = format_conversation_history(
        _user_history(state),
        exclude_last_user_message=True,
    )

    return f"""
현재 사용자 입력:
{state.get("user_input", "")}

이전 사용자 대화:
{conversation_text}

문진 답변 기록:
{json.dumps(
    _follow_up_payload(state),
    ensure_ascii=False,
)}

현재 질문 전략:
{json.dumps(
    state.get("question_strategy", {}),
    ensure_ascii=False,
)}
    """.strip()


def assess_input(
    state: PetCareState,
    *,
    dependencies: AgentDependencies,
) -> dict[str, Any]:
    """LLM으로 사용자 의도와 증상 정보를 구조화한다."""
    started = time.perf_counter()

    try:
        assessment = dependencies.require_llm().parse(
            schema=AssessmentOutput,
            system_prompt=ASSESSMENT_SYSTEM_PROMPT,
            user_prompt=_assessment_prompt(state),
        )

        if not isinstance(
            assessment,
            AssessmentOutput,
        ):
            raise TypeError(
                "AssessmentOutput 형식이 아닙니다."
            )

        handoff_requested = (
            assessment.handoff_requested
            or detect_handoff_request(
                state.get("user_input", "")
            )
        )

        payload = assessment.model_dump()
        payload["handoff_requested"] = (
            handoff_requested
        )
        payload["intent"] = _resolve_intent(
            state,
            assessment.intent,
        )

        updates: dict[str, Any] = {
            "assessment": payload,
            "handoff_requested": handoff_requested,
        }

        if (
            payload["intent"] == "general_chat"
            or handoff_requested
        ):
            updates["route"] = "general_chat"

        return node_result(
            state,
            node_name="assess_input",
            started_at=started,
            updates=updates,
        )

    except Exception as error:
        return add_error(
            state,
            node_name="assess_input",
            error=error,
            started_at=started,
        )


def build_assessment_graph(
    dependencies: AgentDependencies | None = None,
) -> Any:
    """상위 Runtime과 동일한 의존성을 사용하는 Assessment Subgraph 생성."""
    deps = dependencies or build_default_dependencies()
    builder = StateGraph(PetCareState)
    builder.add_node(
        "prepare_backend_context",
        prepare_backend_context,
    )
    builder.add_node(
        "assess_input",
        partial(
            assess_input,
            dependencies=deps,
        ),
    )
    builder.add_edge(
        START,
        "prepare_backend_context",
    )
    builder.add_edge(
        "prepare_backend_context",
        "assess_input",
    )
    builder.add_edge(
        "assess_input",
        END,
    )
    return builder.compile()


# 개별 Subgraph 사용에 대한 기존 호환성 유지.
# 상위 PetCare Graph에서는 build_assessment_graph(deps)를 사용한다.
assessment_graph = build_assessment_graph()
