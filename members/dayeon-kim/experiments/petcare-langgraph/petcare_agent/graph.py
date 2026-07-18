from __future__ import annotations

from typing import Literal

from langgraph.checkpoint.memory import (
    InMemorySaver,
)
from langgraph.graph import (
    END,
    START,
    StateGraph,
)

from .models import PetCareState
from .nodes.agents import (
    chat_agent,
    emergency_agent,
    handoff_subgraph,
    rag_agent,
)
from .nodes.assessment import (
    assessment_graph,
)
from .nodes.safety import (
    current_priority_emergency_codes,
    safety_guard,
)
from .nodes.triage import (
    plan_question_cycle,
    question_manager,
)
from .nodes.workflow import (
    close_non_emergency,
    generate_emergency_email,
    hospital_visit_decision,
    search_open_hospital,
    send_emergency_email,
)


def route_after_assessment(
    state: PetCareState,
) -> Literal[
    "question_manager",
    "safety_guard",
    "chat_agent",
]:
    if state.get("errors"):
        return "chat_agent"

    if state.get(
        "post_triage_mode",
        False,
    ):
        return "chat_agent"

    if current_priority_emergency_codes(
        state
    ):
        return "safety_guard"

    assessment = state.get(
        "assessment",
        {},
    )

    if (
        assessment.get("intent")
        == "general_chat"
    ):
        return "chat_agent"

    if plan_question_cycle(
        state
    ) is not None:
        return "question_manager"

    return "safety_guard"


def route_after_safety(
    state: PetCareState,
) -> Literal[
    "emergency_agent",
    "chat_agent",
]:
    if (
        state.get("route")
        == "emergency"
    ):
        return "emergency_agent"

    return "chat_agent"


def route_after_chat(
    state: PetCareState,
) -> Literal[
    "rag_agent",
    "handoff_subgraph",
    "hospital_visit_decision",
    "__end__",
]:
    if state.get("errors"):
        return END

    if state.get(
        "handoff_requested",
        False,
    ):
        return "handoff_subgraph"

    if (
        state.get("route")
        == "non_emergency"
        and not state.get(
            "rag_done",
            False,
        )
    ):
        return "rag_agent"

    if (
        state.get("route")
        == "non_emergency"
        and state.get(
            "visit_decision",
            "pending",
        )
        == "pending"
    ):
        return "hospital_visit_decision"

    return END


def route_after_visit(
    state: PetCareState,
) -> Literal[
    "handoff_subgraph",
    "close_non_emergency",
]:
    if (
        state.get("visit_decision")
        == "yes"
    ):
        return "handoff_subgraph"

    return "close_non_emergency"


builder = StateGraph(
    PetCareState
)

builder.add_node(
    "assessment_graph",
    assessment_graph,
)
builder.add_node(
    "question_manager",
    question_manager,
)
builder.add_node(
    "safety_guard",
    safety_guard,
)
builder.add_node(
    "emergency_agent",
    emergency_agent,
)
builder.add_node(
    "chat_agent",
    chat_agent,
)
builder.add_node(
    "rag_agent",
    rag_agent,
)
builder.add_node(
    "hospital_visit_decision",
    hospital_visit_decision,
)
builder.add_node(
    "close_non_emergency",
    close_non_emergency,
)
builder.add_node(
    "handoff_subgraph",
    handoff_subgraph,
)
builder.add_node(
    "search_open_hospital",
    search_open_hospital,
)
builder.add_node(
    "generate_emergency_email",
    generate_emergency_email,
)
builder.add_node(
    "send_emergency_email",
    send_emergency_email,
)

builder.add_edge(
    START,
    "assessment_graph",
)
builder.add_conditional_edges(
    "assessment_graph",
    route_after_assessment,
    {
        "question_manager": (
            "question_manager"
        ),
        "safety_guard": "safety_guard",
        "chat_agent": "chat_agent",
    },
)
builder.add_edge(
    "question_manager",
    "assessment_graph",
)
builder.add_conditional_edges(
    "safety_guard",
    route_after_safety,
    {
        "emergency_agent": (
            "emergency_agent"
        ),
        "chat_agent": "chat_agent",
    },
)

builder.add_edge(
    "emergency_agent",
    "search_open_hospital",
)
builder.add_edge(
    "search_open_hospital",
    "generate_emergency_email",
)
builder.add_edge(
    "generate_emergency_email",
    "send_emergency_email",
)
builder.add_edge(
    "send_emergency_email",
    END,
)

builder.add_conditional_edges(
    "chat_agent",
    route_after_chat,
    {
        "rag_agent": "rag_agent",
        "handoff_subgraph": (
            "handoff_subgraph"
        ),
        "hospital_visit_decision": (
            "hospital_visit_decision"
        ),
        END: END,
    },
)
builder.add_edge(
    "rag_agent",
    "chat_agent",
)
builder.add_conditional_edges(
    "hospital_visit_decision",
    route_after_visit,
    {
        "handoff_subgraph": (
            "handoff_subgraph"
        ),
        "close_non_emergency": (
            "close_non_emergency"
        ),
    },
)
builder.add_edge(
    "handoff_subgraph",
    END,
)
builder.add_edge(
    "close_non_emergency",
    END,
)

petcare_graph = builder.compile(
    checkpointer=InMemorySaver()
)
