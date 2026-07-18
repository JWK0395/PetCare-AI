from __future__ import annotations

import json
import re
import time
from typing import Any

from langgraph.graph import END, START, StateGraph

from ..models import AssessmentOutput, PetCareState
from ..prompts import ASSESSMENT_SYSTEM_PROMPT
from ..services import get_llm_service
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
    return any(
        re.search(pattern, text.strip())
        for pattern in HANDOFF_REQUEST_PATTERNS
    )


def _follow_up_payload(
    state: PetCareState,
) -> list[dict[str, Any]]:
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
    if (
        state.get("triage_status")
        == "collecting"
        or state.get("follow_up_history")
    ):
        return "health_related"

    return intent


def assess_input(
    state: PetCareState,
) -> dict[str, Any]:
    started = time.perf_counter()

    try:
        conversation_text = (
            format_conversation_history(
                _user_history(state),
                exclude_last_user_message=True,
            )
        )

        prompt = f"""
현재 사용자 입력:
{state["user_input"]}

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

최근 일기 요약:
{state.get("diary_summary", "없음")}

진단서 요약:
{state.get("diagnosis_summary", "없음")}
        """.strip()

        assessment = get_llm_service().parse(
            schema=AssessmentOutput,
            system_prompt=ASSESSMENT_SYSTEM_PROMPT,
            user_prompt=prompt,
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
