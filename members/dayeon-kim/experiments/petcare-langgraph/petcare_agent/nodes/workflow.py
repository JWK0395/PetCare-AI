from __future__ import annotations

import re
import time
from typing import Any

from langgraph.types import interrupt

from ..handoff import (
    build_handoff_document,
    build_handoff_summary_from_state,
    format_handoff_text,
)
from ..models import PetCareState
from ..response import (
    build_emergency_complete_response,
    build_no_visit_response,
    build_visit_question,
)
from ..services import AgentDependencies
from ..utils import (
    add_error,
    append_conversation_message,
    node_result,
)


YES_PATTERNS = [
    r"^(예|네|응|그래|좋아|갈게|갈래|방문할게|가겠습니다)",
    r"병원.{0,8}(갈|방문)",
]

NO_PATTERNS = [
    r"^(아니오|아니요|아니|안\s*갈|못\s*갈|가지\s*않)",
]


def _visit_answer(text: str) -> str:
    normalized = text.strip()

    if any(
        re.search(pattern, normalized)
        for pattern in YES_PATTERNS
    ):
        return "yes"

    if any(
        re.search(pattern, normalized)
        for pattern in NO_PATTERNS
    ):
        return "no"

    return "unknown"


def hospital_visit_decision(
    state: PetCareState,
) -> dict[str, Any]:
    started = time.perf_counter()
    question = build_visit_question(
        state.get("answer", "")
    )

    answer = interrupt(
        {
            "question": question,
            "field": "hospital_visit",
            "kind": "visit_decision",
            "needs_user_response": True,
        }
    )
    decision = _visit_answer(str(answer))

    if decision == "unknown":
        answer = interrupt(
            {
                "question": (
                    "병원 방문 여부를 '예' 또는 '아니오'로 "
                    "답해 주세요."
                ),
                "field": "hospital_visit",
                "kind": "visit_decision",
                "needs_user_response": True,
            }
        )
        decision = _visit_answer(str(answer))

    if decision not in {"yes", "no"}:
        decision = "no"

    return node_result(
        state,
        node_name="hospital_visit_decision",
        started_at=started,
        updates={
            "visit_decision": decision,
            "handoff_requested": decision == "yes",
            "needs_user_response": False,
        },
    )


def close_non_emergency(
    state: PetCareState,
) -> dict[str, Any]:
    started = time.perf_counter()
    answer = build_no_visit_response()

    return node_result(
        state,
        node_name="close_non_emergency",
        started_at=started,
        updates={
            "triage_status": "completed",
            "answer": answer,
            "conversation_history": (
                append_conversation_message(
                    state,
                    role="assistant",
                    content=answer,
                )
            ),
        },
    )


def search_open_hospital(
    state: PetCareState,
    *,
    dependencies: AgentDependencies,
) -> dict[str, Any]:
    started = time.perf_counter()

    try:
        hospitals = dependencies.hospital_search.search_open(
            location=state.get("location"),
            limit=5,
        )
        open_hospitals = [
            hospital
            for hospital in hospitals
            if hospital.is_open
        ]

        if not open_hospitals:
            raise RuntimeError(
                "운영 중인 동물병원을 찾지 못했습니다."
            )

        selected = sorted(
            open_hospitals,
            key=lambda item: (
                item.distance_km
                if item.distance_km is not None
                else float("inf")
            ),
        )[0]

        return node_result(
            state,
            node_name="search_open_hospital",
            started_at=started,
            updates={
                "nearby_hospitals": [
                    item.model_dump()
                    for item in open_hospitals
                ],
                "selected_hospital": selected.model_dump(),
            },
        )

    except Exception as error:
        return add_error(
            state,
            node_name="search_open_hospital",
            error=error,
            started_at=started,
        )


def generate_emergency_email(
    state: PetCareState,
) -> dict[str, Any]:
    started = time.perf_counter()

    try:
        context = state.get("backend_context", {})
        hospital = state.get("selected_hospital", {})
        summary = build_handoff_summary_from_state(state)
        handoff = build_handoff_document(
            state,
            summary,
        )
        pet_name = str(
            context.get("pet", {}).get(
                "name",
                "반려동물",
            )
        )
        subject = (
            "[PetCare AI 응급 전달] "
            f"{pet_name} 상태 요약"
        )
        body = format_handoff_text(
            handoff,
            hospital_name=str(
                hospital.get("name", "미확인")
            ),
        )

        return node_result(
            state,
            node_name="generate_emergency_email",
            started_at=started,
            updates={
                "handoff": handoff.model_dump(),
                "email_subject": subject,
                "email_body": body,
            },
        )

    except Exception as error:
        return add_error(
            state,
            node_name="generate_emergency_email",
            error=error,
            started_at=started,
        )


def send_emergency_email(
    state: PetCareState,
    *,
    dependencies: AgentDependencies,
) -> dict[str, Any]:
    started = time.perf_counter()

    try:
        hospital = state.get("selected_hospital", {})
        recipient = hospital.get("email")

        if not recipient:
            raise ValueError(
                "선택된 병원의 이메일 주소가 없습니다."
            )

        delivery = dependencies.email.send(
            recipient=str(recipient),
            subject=state.get("email_subject", ""),
            body=state.get("email_body", ""),
        )
        state_with_delivery = {
            **state,
            "email_delivery": delivery.model_dump(),
        }
        answer = build_emergency_complete_response(
            state_with_delivery
        )

        return node_result(
            state,
            node_name="send_emergency_email",
            started_at=started,
            updates={
                "triage_status": "completed",
                "email_delivery": delivery.model_dump(),
                "answer": answer,
                "conversation_history": (
                    append_conversation_message(
                        state,
                        role="assistant",
                        content=answer,
                    )
                ),
            },
        )

    except Exception as error:
        failed = {
            "status": "failed",
            "recipient": state.get(
                "selected_hospital",
                {},
            ).get("email"),
            "error": str(error),
        }
        state_with_delivery = {
            **state,
            "email_delivery": failed,
        }
        answer = build_emergency_complete_response(
            state_with_delivery
        )
        result = add_error(
            state,
            node_name="send_emergency_email",
            error=error,
            started_at=started,
        )
        result.update(
            {
                "triage_status": "completed",
                "email_delivery": failed,
                "answer": answer,
                "conversation_history": (
                    append_conversation_message(
                        state,
                        role="assistant",
                        content=answer,
                    )
                ),
            }
        )
        return result
