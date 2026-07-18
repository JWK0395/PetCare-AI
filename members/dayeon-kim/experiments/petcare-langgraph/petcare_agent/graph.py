from __future__ import annotations

from typing import Literal

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from .models import PetCareState
from .nodes.agents import (
    chat_agent,
    emergency_agent,
    handoff_subgraph,
    rag_agent,
)
from .nodes.assessment import assessment_graph
from .nodes.safety import (
    current_priority_emergency_codes,
    safety_guard,
)
from .nodes.triage import (
    plan_question_cycle,
    question_manager,
)


def has_obvious_emergency_wording(
    state: PetCareState,
) -> bool:
    return bool(
        current_priority_emergency_codes(
            state
        )
    )


def route_after_assessment(
    state: PetCareState,
) -> Literal["question_manager", "safety_guard", "chat_agent"]:
    if state.get("errors"):
        return "chat_agent"

    assessment = state.get("assessment", {})

    # 완료된 Episode에 대한 후속 대화는 질문 Cycle을 다시 열지 않습니다.
    if state.get("post_triage_mode", False):
        return "chat_agent"

    # 원문에 명확한 응급 표현이 있으면 추가 질문 없이
    # 즉시 Safety Guard로 보냅니다.
    if has_obvious_emergency_wording(state):
        return "safety_guard"

    if assessment.get("intent") == "general_chat":
        return "chat_agent"

    if plan_question_cycle(state) is not None:
        return "question_manager"

    return "safety_guard"


def route_after_safety(
    state: PetCareState,
) -> Literal["emergency_agent", "chat_agent"]:
    if state.get("route") == "emergency":
        return "emergency_agent"
    return "chat_agent"


def route_after_chat(
    state: PetCareState,
) -> Literal["rag_agent", "handoff_subgraph", "__end__"]:
    if state.get("errors"):
        return END

    if state.get("handoff_requested", False):
        return "handoff_subgraph"

    if (
        state.get("route") == "non_emergency"
        and not state.get("rag_done", False)
    ):
        return "rag_agent"

    return END


builder = StateGraph(PetCareState)

builder.add_node("assessment_graph", assessment_graph)
builder.add_node("question_manager", question_manager)
builder.add_node("safety_guard", safety_guard)
builder.add_node("emergency_agent", emergency_agent)
builder.add_node("chat_agent", chat_agent)
builder.add_node("rag_agent", rag_agent)
builder.add_node("handoff_subgraph", handoff_subgraph)

builder.add_edge(START, "assessment_graph")

builder.add_conditional_edges(
    "assessment_graph",
    route_after_assessment,
    {
        "question_manager": "question_manager",
        "safety_guard": "safety_guard",
        "chat_agent": "chat_agent",
    },
)

# 답변을 받은 뒤 Assessment Graph가 추가 답변까지 다시 구조화합니다.
builder.add_edge("question_manager", "assessment_graph")

builder.add_conditional_edges(
    "safety_guard",
    route_after_safety,
    {
        "emergency_agent": "emergency_agent",
        "chat_agent": "chat_agent",
    },
)

builder.add_edge("emergency_agent", END)

builder.add_conditional_edges(
    "chat_agent",
    route_after_chat,
    {
        "rag_agent": "rag_agent",
        "handoff_subgraph": "handoff_subgraph",
        END: END,
    },
)

builder.add_edge("rag_agent", "chat_agent")
builder.add_edge("handoff_subgraph", END)

checkpointer = InMemorySaver()
petcare_graph = builder.compile(checkpointer=checkpointer)

print("LangGraph 컴파일 완료")
