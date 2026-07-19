from __future__ import annotations

import json
import time
from typing import Any

from langgraph.graph import (
    END,
    START,
    StateGraph,
)

from ..documents import create_handoff_pdf
from ..handoff import (
    build_handoff_document,
    build_handoff_summary_from_state,
)
from ..models import PetCareState, PromptContext
from ..prompt_context import build_prompt_context
from ..prompts import (
    GENERAL_CHAT_SYSTEM_PROMPT,
    NON_EMERGENCY_SYSTEM_PROMPT,
    POST_TRIAGE_SYSTEM_PROMPT,
)
from ..response import (
    build_emergency_response,
    build_pdf_complete_response,
    clean_agent_response,
)
from ..services import AgentDependencies
from ..utils import (
    add_error,
    add_warning,
    append_conversation_message,
    format_conversation_history,
    node_result,
)
from .triage import is_post_triage_acknowledgement


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

    if triage_status:
        updates["triage_status"] = triage_status

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
    return _answer_result(
        state,
        node_name="emergency_agent",
        started_at=time.perf_counter(),
        answer=build_emergency_response(state),
    )


def rag_agent(
    state: PetCareState,
    *,
    dependencies: AgentDependencies,
) -> dict[str, Any]:
    started = time.perf_counter()

    assessment = state.get("assessment", {})
    symptoms = [
        item.get("code", "")
        for item in assessment.get("symptoms", [])
        if not item.get("negated", False)
    ]

    query = (
        f"사용자 질문: {state['user_input']}\n"
        f"구조화 증상: {', '.join(symptoms) or '없음'}"
    )

    try:
        chunks = dependencies.rag.search(
            query=query,
            pet_context=state.get(
                "backend_context",
                {},
            ),
            limit=5,
        )
        normalized_chunks = [
            chunk.model_dump()
            for chunk in chunks
        ]

        return node_result(
            state,
            node_name="rag_agent",
            started_at=started,
            updates={
                "rag_query": query,
                "rag_chunks": normalized_chunks,
                "rag_done": True,
                "rag_status": (
                    "completed"
                    if normalized_chunks
                    else "unavailable"
                ),
            },
        )

    except Exception as error:
        return add_warning(
            state,
            node_name="rag_agent",
            warning=(
                f"{type(error).__name__}: {error}. "
                "RAG 근거 없이 답변을 계속합니다."
            ),
            started_at=started,
            updates={
                "rag_query": query,
                "rag_chunks": [],
                "rag_done": True,
                "rag_status": "failed",
            },
        )


def _context_stats(
    prompt_context: PromptContext,
    conversation_text: str,
) -> dict[str, Any]:
    return {
        "daily_entries": len(
            prompt_context.daily_entries
        ),
        "diagnoses": len(
            prompt_context.diagnoses
        ),
        "conversation_chars": len(
            conversation_text
        ),
        "data_period": prompt_context.data_period,
    }


def _general_prompt(
    state: PetCareState,
    conversation_text: str,
    prompt_context: PromptContext,
) -> str:
    return f"""
이전 대화:
{conversation_text}

현재 질문에 필요한 등록 기록:
{json.dumps(
    prompt_context.model_dump(),
    ensure_ascii=False,
)}

현재 입력:
{state["user_input"]}
    """.strip()


def _non_emergency_prompt(
    state: PetCareState,
    conversation_text: str,
    prompt_context: PromptContext,
) -> str:
    return f"""
이전 대화:
{conversation_text}

현재 보호자 입력:
{state["user_input"]}

문진 답변:
{json.dumps(
    state.get("follow_up_history", []),
    ensure_ascii=False,
)}

현재 증상:
{json.dumps(
    state.get("assessment", {}),
    ensure_ascii=False,
)}

현재 질문과 증상에 관련된 등록 기록:
{json.dumps(
    prompt_context.model_dump(),
    ensure_ascii=False,
)}

내부 RAG 근거:
{json.dumps(
    state.get("rag_chunks", []),
    ensure_ascii=False,
)}

RAG 상태:
{state.get("rag_status", "not_started")}

Safety Guard:
non_emergency

사용자에게는 현재 확인된 내용과 현재 판단만 표시한다.
권장 행동과 근거 섹션은 만들지 않는다.
등록 기록의 기간과 현재 보호자 진술의 시점을 구분한다.
    """.strip()


def chat_agent(
    state: PetCareState,
    *,
    dependencies: AgentDependencies,
) -> dict[str, Any]:
    started = time.perf_counter()

    try:
        if (
            state.get("errors")
            or state.get("handoff_requested", False)
        ):
            return node_result(
                state,
                node_name="chat_agent",
                started_at=started,
                updates={},
            )

        route = state.get("route")
        conversation_text = format_conversation_history(
            state.get("conversation_history", []),
            exclude_last_user_message=True,
        )
        prompt_context = build_prompt_context(state)
        prompt_stats = _context_stats(
            prompt_context,
            conversation_text,
        )

        if route == "general_chat":
            if state.get("post_triage_mode", False):
                if is_post_triage_acknowledgement(
                    state.get("user_input", "")
                ):
                    answer = (
                        "이번 상태 확인은 이미 종료되었습니다."
                    )
                else:
                    answer = dependencies.require_llm().text(
                        system_prompt=(
                            POST_TRIAGE_SYSTEM_PROMPT
                        ),
                        user_prompt=_general_prompt(
                            state,
                            conversation_text,
                            prompt_context,
                        ),
                    )
            else:
                answer = dependencies.require_llm().text(
                    system_prompt=(
                        GENERAL_CHAT_SYSTEM_PROMPT
                    ),
                    user_prompt=_general_prompt(
                        state,
                        conversation_text,
                        prompt_context,
                    ),
                )

            return _answer_result(
                state,
                node_name="chat_agent",
                started_at=started,
                answer=answer,
                extra_updates={
                    "prompt_context_stats": prompt_stats,
                },
            )

        if (
            route == "non_emergency"
            and not state.get("rag_done", False)
        ):
            return node_result(
                state,
                node_name="chat_agent",
                started_at=started,
                updates={},
            )

        answer = dependencies.require_llm().text(
            system_prompt=(
                NON_EMERGENCY_SYSTEM_PROMPT
            ),
            user_prompt=_non_emergency_prompt(
                state,
                conversation_text,
                prompt_context,
            ),
        )

        return _answer_result(
            state,
            node_name="chat_agent",
            started_at=started,
            answer=answer,
            extra_updates={
                "prompt_context_stats": prompt_stats,
            },
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
        summary = build_handoff_summary_from_state(state)
        handoff = build_handoff_document(
            state,
            summary,
        )

        return node_result(
            state,
            node_name="generate_handoff",
            started_at=started,
            updates={
                "handoff": handoff.model_dump()
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
        artifact_path = create_handoff_pdf(
            handoff=state.get("handoff", {}),
            session_id=state["session_id"],
        )

        return _answer_result(
            state,
            node_name="render_handoff_pdf",
            started_at=started,
            answer=build_pdf_complete_response(
                artifact_path
            ),
            triage_status="completed",
            extra_updates={
                "artifact_path": artifact_path
            },
        )

    except Exception as error:
        return add_error(
            state,
            node_name="render_handoff_pdf",
            error=error,
            started_at=started,
        )


handoff_builder = StateGraph(PetCareState)
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

handoff_subgraph = handoff_builder.compile()
