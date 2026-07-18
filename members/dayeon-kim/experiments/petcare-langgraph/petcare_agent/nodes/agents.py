from __future__ import annotations

import json
import time
from typing import Any

from langgraph.graph import (
    END,
    START,
    StateGraph,
)

from ..documents import (
    create_handoff_pdf,
)
from ..handoff import (
    build_handoff_document,
)
from ..models import (
    HandoffOutput,
    PetCareState,
)
from ..prompts import (
    GENERAL_CHAT_SYSTEM_PROMPT,
    HANDOFF_SYSTEM_PROMPT,
    NON_EMERGENCY_SYSTEM_PROMPT,
    POST_TRIAGE_SYSTEM_PROMPT,
)
from ..response import (
    build_emergency_response,
    build_pdf_complete_response,
    clean_agent_response,
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
    extra_updates: (
        dict[str, Any] | None
    ) = None,
) -> dict[str, Any]:
    cleaned = clean_agent_response(
        answer
    )

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

    if triage_status:
        updates["triage_status"] = (
            triage_status
        )

    if extra_updates:
        updates.update(
            extra_updates
        )

    return node_result(
        state,
        node_name=node_name,
        started_at=started_at,
        updates=updates,
    )


def emergency_agent(
    state: PetCareState,
) -> dict[str, Any]:
    return _answer_result(
        state,
        node_name="emergency_agent",
        started_at=time.perf_counter(),
        answer=build_emergency_response(
            state
        ),
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

        query = (
            f"사용자 질문: "
            f"{state['user_input']}\n"
            f"구조화 증상: "
            f"{', '.join(symptoms) or '없음'}"
        )

        chunks = (
            get_rag_provider().search(
                query=query,
                pet_context=state.get(
                    "backend_context",
                    {},
                ),
                limit=5,
            )
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
{json.dumps(
    context.get("pet", {}),
    ensure_ascii=False,
)}

등록 일기:
{json.dumps(
    context.get(
        "daily_entries",
        [],
    ),
    ensure_ascii=False,
)}

진단서 요약:
{state.get("diagnosis_summary", "없음")}

현재 입력:
{state["user_input"]}
    """.strip()


def _non_emergency_prompt(
    state: PetCareState,
    conversation_text: str,
) -> str:
    return f"""
이전 대화:
{conversation_text}

현재 보호자 입력:
{state["user_input"]}

문진 답변:
{json.dumps(
    state.get(
        "follow_up_history",
        [],
    ),
    ensure_ascii=False,
)}

현재 증상:
{json.dumps(
    state.get(
        "assessment",
        {},
    ),
    ensure_ascii=False,
)}

최근 일기 요약:
{state.get("diary_summary", "없음")}

진단서 요약:
{state.get("diagnosis_summary", "없음")}

내부 RAG 근거:
{json.dumps(
    state.get(
        "rag_chunks",
        [],
    ),
    ensure_ascii=False,
)}

Safety Guard:
non_emergency

사용자에게는 현재 확인된 내용과 현재 판단만 표시한다.
권장 행동과 근거 섹션은 만들지 않는다.
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
                if (
                    is_post_triage_acknowledgement(
                        state.get(
                            "user_input",
                            "",
                        )
                    )
                ):
                    answer = (
                        "이번 상태 확인은 "
                        "이미 종료되었습니다."
                    )
                else:
                    answer = (
                        get_llm_service().text(
                            system_prompt=(
                                POST_TRIAGE_SYSTEM_PROMPT
                            ),
                            user_prompt=(
                                _general_prompt(
                                    state,
                                    conversation_text,
                                )
                            ),
                        )
                    )
            else:
                answer = (
                    get_llm_service().text(
                        system_prompt=(
                            GENERAL_CHAT_SYSTEM_PROMPT
                        ),
                        user_prompt=(
                            _general_prompt(
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
            user_prompt=(
                _non_emergency_prompt(
                    state,
                    conversation_text,
                )
            ),
        )

        return _answer_result(
            state,
            node_name="chat_agent",
            started_at=started,
            answer=answer,
        )

    except Exception as error:
        return add_error(
            state,
            node_name="chat_agent",
            error=error,
            started_at=started,
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
현재 사용자 입력:
{state["user_input"]}

현재 증상 구조화 결과:
{json.dumps(
    state.get(
        "assessment",
        {},
    ),
    ensure_ascii=False,
)}

문진 전체 기록:
{json.dumps(
    state.get(
        "follow_up_history",
        [],
    ),
    ensure_ascii=False,
)}

최근 일기 요약:
{state.get("diary_summary", "없음")}

진단서 요약:
{state.get("diagnosis_summary", "없음")}

대화 기록:
{json.dumps(
    state.get(
        "conversation_history",
        [],
    ),
    ensure_ascii=False,
)}
        """.strip()

        summary = (
            get_llm_service().parse(
                schema=HandoffOutput,
                system_prompt=(
                    HANDOFF_SYSTEM_PROMPT
                ),
                user_prompt=prompt,
            )
        )

        if not isinstance(
            summary,
            HandoffOutput,
        ):
            raise TypeError(
                "HandoffOutput 형식이 "
                "아닙니다."
            )

        handoff = build_handoff_document(
            state,
            summary,
        )

        return node_result(
            state,
            node_name="generate_handoff",
            started_at=started,
            updates={
                "handoff": handoff
            },
        )

    except Exception as error:
        return add_error(
            state,
            node_name="generate_handoff",
            error=error,
            started_at=started,
        )


def render_handoff_pdf(
    state: PetCareState,
) -> dict[str, Any]:
    started = time.perf_counter()

    try:
        artifact_path = (
            create_handoff_pdf(
                handoff=state.get(
                    "handoff",
                    {},
                ),
                session_id=state[
                    "session_id"
                ],
            )
        )

        return _answer_result(
            state,
            node_name="render_handoff_pdf",
            started_at=started,
            answer=(
                build_pdf_complete_response(
                    artifact_path
                )
            ),
            triage_status="completed",
            extra_updates={
                "artifact_path": (
                    artifact_path
                )
            },
        )

    except Exception as error:
        return add_error(
            state,
            node_name="render_handoff_pdf",
            error=error,
            started_at=started,
        )


handoff_builder = StateGraph(
    PetCareState
)
handoff_builder.add_node(
    "generate_handoff",
    generate_handoff,
)
handoff_builder.add_node(
    "render_handoff_pdf",
    render_handoff_pdf,
)
handoff_builder.add_edge(
    START,
    "generate_handoff",
)
handoff_builder.add_edge(
    "generate_handoff",
    "render_handoff_pdf",
)
handoff_builder.add_edge(
    "render_handoff_pdf",
    END,
)

handoff_subgraph = (
    handoff_builder.compile()
)
