from __future__ import annotations

import json
import time
from typing import Any

from langgraph.graph import END, START, StateGraph

from ..models import HandoffOutput, PetCareState
from ..prompts import (
    GENERAL_CHAT_SYSTEM_PROMPT,
    HANDOFF_SYSTEM_PROMPT,
    NON_EMERGENCY_SYSTEM_PROMPT,
    POST_TRIAGE_SYSTEM_PROMPT,
)
from ..response import (
    build_emergency_response,
    clean_agent_response,
    closed_triage_message,
    format_evidence,
    handoff_confirmation,
    json_text,
)
from ..services import (
    get_llm_service,
    get_rag_provider,
)
from ..utils import (
    add_error,
    append_conversation_message,
    format_conversation_history,
    node_result,
)
from .triage import (
    is_post_triage_acknowledgement,
)


def _answer_result(
    state: PetCareState,
    *,
    node_name: str,
    started_at: float,
    answer: str,
    triage_status: str | None = None,
    extra_updates: dict[str, Any] | None = None,
) -> dict[str, Any]:
    cleaned = clean_agent_response(answer)

    updates: dict[str, Any] = {
        "answer": cleaned,
        "conversation_history": (
            append_conversation_message(
                state,
                role="assistant",
                content=cleaned,
            )
        ),
    }

    if triage_status is not None:
        updates["triage_status"] = (
            triage_status
        )

    if extra_updates:
        updates.update(extra_updates)

    return node_result(
        state,
        node_name=node_name,
        started_at=started_at,
        updates=updates,
    )


def emergency_agent(
    state: PetCareState,
) -> dict[str, Any]:
    started = time.perf_counter()

    return _answer_result(
        state,
        node_name="emergency_agent",
        started_at=started,
        answer=build_emergency_response(
            state
        ),
        triage_status="completed",
    )


def rag_agent(
    state: PetCareState,
) -> dict[str, Any]:
    started = time.perf_counter()

    try:
        assessment = state.get(
            "assessment",
            {},
        )
        symptoms = [
            item.get("code", "")
            for item in assessment.get(
                "symptoms",
                [],
            )
            if not item.get(
                "negated",
                False,
            )
        ]

        query = "\n".join(
            [
                (
                    "사용자 질문: "
                    f"{state['user_input']}"
                ),
                (
                    "구조화 증상: "
                    f"{', '.join(symptoms) or '없음'}"
                ),
            ]
        )

        chunks = get_rag_provider().search(
            query=query,
            pet_context=state.get(
                "backend_context",
                {},
            ),
            limit=5,
        )

        return node_result(
            state,
            node_name="rag_agent",
            started_at=started,
            updates={
                "rag_query": query,
                "rag_chunks": [
                    chunk.model_dump()
                    for chunk in chunks
                ],
                "rag_done": True,
            },
        )

    except Exception as error:
        return add_error(
            state,
            node_name="rag_agent",
            error=error,
            started_at=started,
        )


def _general_prompt(
    state: PetCareState,
    conversation_text: str,
) -> str:
    context = state.get(
        "backend_context",
        {},
    )

    return f"""
이전 대화:
{conversation_text}

등록 프로필:
{json_text(context.get("pet", {}))}

등록 일기:
{json_text(context.get("daily_entries", []))}

일기 요약:
{state.get("diary_summary", "없음")}

진단서 요약:
{state.get("diagnosis_summary", "없음")}

현재 입력:
{state["user_input"]}
    """.strip()


def _post_triage_prompt(
    state: PetCareState,
    conversation_text: str,
) -> str:
    return f"""
이전에 완료된 상태 확인:
{json_text(state.get("previous_triage", {}))}

이전 대화:
{conversation_text}

현재 입력:
{state["user_input"]}

이전 결론을 설명하되 새로운 문진은 시작하지 않는다.
    """.strip()


def _non_emergency_prompt(
    state: PetCareState,
    conversation_text: str,
) -> str:
    context = state.get(
        "backend_context",
        {},
    )

    return f"""
이전 대화:
{conversation_text}

현재 보호자 입력:
{state["user_input"]}

반려동물 정보:
{json_text(context.get("pet", {}))}

문진 답변:
{json_text(state.get("follow_up_history", []))}

현재 증상 구조화 결과:
{json_text(state.get("assessment", {}))}

최근 일기 요약:
{state.get("diary_summary", "없음")}

진단서 요약:
{state.get("diagnosis_summary", "없음")}

추가 증상 확인 상태:
{json_text(state.get("question_strategy", {}))}

회복 표현:
{json_text(state.get("recovery_hits", []))}

검색 근거:
{format_evidence(state.get("rag_chunks", []))}

Safety Guard 결과:
non_emergency

이 답변으로 현재 상태 확인을 종료한다.
    """.strip()


def chat_agent(
    state: PetCareState,
) -> dict[str, Any]:
    started = time.perf_counter()

    try:
        if (
            state.get("errors")
            or state.get(
                "handoff_requested",
                False,
            )
        ):
            return node_result(
                state,
                node_name="chat_agent",
                started_at=started,
                updates={},
            )

        route = state.get("route")
        conversation_text = (
            format_conversation_history(
                state.get(
                    "conversation_history",
                    [],
                ),
                exclude_last_user_message=True,
            )
        )

        if route == "general_chat":
            if state.get(
                "post_triage_mode",
                False,
            ):
                if is_post_triage_acknowledgement(
                    state.get(
                        "user_input",
                        "",
                    )
                ):
                    answer = closed_triage_message()
                else:
                    answer = (
                        get_llm_service().text(
                            system_prompt=(
                                POST_TRIAGE_SYSTEM_PROMPT
                            ),
                            user_prompt=(
                                _post_triage_prompt(
                                    state,
                                    conversation_text,
                                )
                            ),
                        )
                    )

                return _answer_result(
                    state,
                    node_name="chat_agent",
                    started_at=started,
                    answer=answer,
                    triage_status="completed",
                )

            answer = get_llm_service().text(
                system_prompt=(
                    GENERAL_CHAT_SYSTEM_PROMPT
                ),
                user_prompt=_general_prompt(
                    state,
                    conversation_text,
                ),
            )

            return _answer_result(
                state,
                node_name="chat_agent",
                started_at=started,
                answer=answer,
            )

        if (
            route == "non_emergency"
            and not state.get(
                "rag_done",
                False,
            )
        ):
            return node_result(
                state,
                node_name="chat_agent",
                started_at=started,
                updates={},
            )

        answer = get_llm_service().text(
            system_prompt=(
                NON_EMERGENCY_SYSTEM_PROMPT
            ),
            user_prompt=_non_emergency_prompt(
                state,
                conversation_text,
            ),
        )

        return _answer_result(
            state,
            node_name="chat_agent",
            started_at=started,
            answer=answer,
            triage_status="completed",
        )

    except Exception as error:
        return add_error(
            state,
            node_name="chat_agent",
            error=error,
            started_at=started,
        )


def collect_handoff_context(
    state: PetCareState,
) -> dict[str, Any]:
    return node_result(
        state,
        node_name="collect_handoff_context",
        started_at=time.perf_counter(),
        updates={},
    )


def generate_handoff(
    state: PetCareState,
) -> dict[str, Any]:
    started = time.perf_counter()

    try:
        context = state.get(
            "backend_context",
            {},
        )

        prompt = f"""
현재 입력:
{state["user_input"]}

반려동물 프로필:
{json_text(context.get("pet", {}))}

최근 일기 요약:
{state.get("diary_summary", "없음")}

진단서 요약:
{state.get("diagnosis_summary", "없음")}

미확인 항목:
{json_text(context.get("unknown_items", []))}
        """.strip()

        handoff = get_llm_service().parse(
            schema=HandoffOutput,
            system_prompt=(
                HANDOFF_SYSTEM_PROMPT
            ),
            user_prompt=prompt,
        )

        if not isinstance(
            handoff,
            HandoffOutput,
        ):
            raise TypeError(
                "HandoffOutput 형식이 아닙니다."
            )

        return _answer_result(
            state,
            node_name="generate_handoff",
            started_at=started,
            answer=handoff_confirmation(),
            extra_updates={
                "handoff": handoff.model_dump(),
            },
        )

    except Exception as error:
        return add_error(
            state,
            node_name="generate_handoff",
            error=error,
            started_at=started,
        )


handoff_builder = StateGraph(PetCareState)
handoff_builder.add_node(
    "collect_handoff_context",
    collect_handoff_context,
)
handoff_builder.add_node(
    "generate_handoff",
    generate_handoff,
)
handoff_builder.add_edge(
    START,
    "collect_handoff_context",
)
handoff_builder.add_edge(
    "collect_handoff_context",
    "generate_handoff",
)
handoff_builder.add_edge(
    "generate_handoff",
    END,
)

handoff_subgraph = handoff_builder.compile()
