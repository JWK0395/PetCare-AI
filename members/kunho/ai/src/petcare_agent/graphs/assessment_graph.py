"""LangGraph wiring and runner for the PetCare assessment graph."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import Any

from langgraph.graph import END, StateGraph

from petcare_agent.graphs.response_composer import compose_graph_response
from petcare_agent.graphs.subgraphs.handoff import (
    handoff_subgraph,
    should_build_non_emergency_handoff,
)
from petcare_agent.llm.client import StructuredOutputClient
from petcare_agent.nodes.answer_composer import answer_composer
from petcare_agent.nodes.answer_guard import answer_guard
from petcare_agent.nodes.baseline_builder import baseline_builder
from petcare_agent.nodes.chat_agent import chat_agent
from petcare_agent.nodes.change_detector import change_detector
from petcare_agent.nodes.evidence_planner import evidence_planner
from petcare_agent.nodes.db_context_loader import DBContextProvider, db_context_loader
from petcare_agent.nodes.emergency_agent import emergency_agent
from petcare_agent.nodes.intent_classifier import intent_classifier
from petcare_agent.nodes.question_manager import question_manager
from petcare_agent.nodes.rag_agent import rag_agent
from petcare_agent.nodes.safety_guard import safety_guard
from petcare_agent.nodes.state_updater import state_updater
from petcare_agent.rag.adapter import RAGAdapter
from petcare_agent.schemas.common import NodeRoute
from petcare_agent.schemas.graph_state import (
    GraphRequest,
    GraphResponse,
    PetCareGraphState,
)
from petcare_agent.tracing import (
    TraceContext,
    build_runnable_config,
    build_state_trace_metadata,
    trace_span,
)

TraceMetadataHook = Callable[["NodeTraceMetadata"], None]
NodeCallable = Callable[[PetCareGraphState], PetCareGraphState]


@dataclass(frozen=True)
class NodeTraceMetadata:
    """Route metadata captured after a graph node executes."""

    node_name: str
    route: NodeRoute
    intent: str
    risk_level: str
    triggered_rules: list[str] = field(default_factory=list)
    next_route: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AssessmentGraphDependencies:
    """Mockable graph boundaries for Phase 9 wiring tests."""

    llm_client: StructuredOutputClient | None = None
    db_context_provider: DBContextProvider | None = None
    rag_adapter: RAGAdapter | None = None
    trace_metadata_hook: TraceMetadataHook | None = None
    db_context_days: int = 3
    rag_top_k: int = 5


@dataclass(frozen=True)
class AssessmentGraphRunResult:
    """Result returned by the graph runner."""

    state: PetCareGraphState
    response: GraphResponse
    trace_events: list[NodeTraceMetadata]


_NODE_ROUTE_BY_NAME: dict[str, NodeRoute] = {
    "intent_classifier": "intent_classifier",
    "db_context_loader": "db_context_loader",
    "baseline_builder": "baseline_builder",
    "state_updater": "state_updater",
    "change_detector": "change_detector",
    "safety_guard": "safety_guard",
    "question_manager": "question_manager",
    "emergency_agent": "emergency",
    "chat_agent": "chat",
    "evidence_planner": "evidence_planner",
    "rag_agent": "rag",
    "answer_composer": "answer_composer",
    "answer_guard": "answer_guard",
    "handoff_subgraph": "handoff",
}


def compile_assessment_graph(
    dependencies: AssessmentGraphDependencies | None = None,
) -> Any:
    """Compile the Phase 9 Assessment Graph."""

    deps = dependencies or AssessmentGraphDependencies()
    builder = StateGraph(PetCareGraphState)

    builder.add_node(
        "intent_classifier",
        _wrap_node(
            "intent_classifier",
            lambda state: intent_classifier(state, llm_client=deps.llm_client),
            deps,
        ),
    )
    builder.add_node(
        "db_context_loader",
        _wrap_node(
            "db_context_loader",
            lambda state: db_context_loader(
                state,
                provider=deps.db_context_provider,
                days=deps.db_context_days,
            ),
            deps,
        ),
    )
    builder.add_node("baseline_builder", _wrap_node("baseline_builder", baseline_builder, deps))
    builder.add_node(
        "state_updater",
        _wrap_node(
            "state_updater",
            lambda state: state_updater(state, llm_client=deps.llm_client),
            deps,
        ),
    )
    builder.add_node("change_detector", _wrap_node("change_detector", change_detector, deps))
    builder.add_node(
        "safety_guard",
        _wrap_node(
            "safety_guard",
            lambda state: safety_guard(state, llm_client=deps.llm_client),
            deps,
        ),
    )
    builder.add_node("question_manager", _wrap_node("question_manager", question_manager, deps))
    builder.add_node("emergency_agent", _wrap_node("emergency_agent", emergency_agent, deps))
    builder.add_node(
        "chat_agent",
        _wrap_node(
            "chat_agent",
            lambda state: chat_agent(state, llm_client=deps.llm_client),
            deps,
        ),
    )
    builder.add_node("evidence_planner", _wrap_node("evidence_planner", evidence_planner, deps))
    builder.add_node(
        "rag_agent",
        _wrap_node(
            "rag_agent",
            lambda state: rag_agent(
                state,
                adapter=deps.rag_adapter,
                top_k=deps.rag_top_k,
            ),
            deps,
        ),
    )
    builder.add_node(
        "answer_composer",
        _wrap_node(
            "answer_composer",
            lambda state: answer_composer(state, llm_client=deps.llm_client),
            deps,
        ),
    )
    builder.add_node(
        "answer_guard",
        _wrap_node(
            "answer_guard",
            lambda state: answer_guard(state, llm_client=deps.llm_client),
            deps,
        ),
    )
    builder.add_node("handoff_subgraph", _wrap_node("handoff_subgraph", handoff_subgraph, deps))

    builder.set_entry_point("db_context_loader")
    builder.add_edge("db_context_loader", "intent_classifier")

    builder.add_edge("chat_agent", END)
    builder.add_conditional_edges(
        "intent_classifier",
        _route_after_db_context_loader,
        {
            "chat_agent": "chat_agent",
            "evidence_planner": "evidence_planner",
            "baseline_builder": "baseline_builder",
        },
    )
    builder.add_edge("baseline_builder", "state_updater")
    builder.add_edge("state_updater", "change_detector")
    builder.add_edge("change_detector", "safety_guard")
    builder.add_conditional_edges(
        "safety_guard",
        _route_after_safety_guard,
        {
            "question_manager": "question_manager",
            "emergency_agent": "emergency_agent",
            "evidence_planner": "evidence_planner",
        },
    )
    builder.add_edge("question_manager", END)
    builder.add_edge("emergency_agent", END)
    builder.add_edge("evidence_planner", "rag_agent")
    builder.add_edge("rag_agent", "answer_composer")
    builder.add_edge("answer_composer", "answer_guard")
    builder.add_conditional_edges(
        "answer_guard",
        _route_after_answer_guard,
        {
            "handoff_subgraph": "handoff_subgraph",
            "end": END,
        },
    )
    builder.add_edge("handoff_subgraph", END)

    return builder.compile()


def run_assessment_graph(
    request_or_state: GraphRequest | PetCareGraphState | dict[str, Any],
    *,
    dependencies: AssessmentGraphDependencies | None = None,
) -> AssessmentGraphRunResult:
    """Run the compiled graph and compose the public graph response."""

    initial_state = build_initial_state(request_or_state)
    trace_events: list[NodeTraceMetadata] = []
    base_deps = dependencies or AssessmentGraphDependencies()
    deps = replace(
        base_deps,
        trace_metadata_hook=_collector_hook(trace_events, base_deps.trace_metadata_hook),
    )
    graph = compile_assessment_graph(deps)
    output = graph.invoke(
        initial_state,
        config=build_runnable_config(
            TraceContext(
                request_id=initial_state.request_id,
                conversation_id=initial_state.conversation_id,
                route="assessment_graph",
                metadata=build_state_trace_metadata(
                    initial_state,
                    route="assessment_graph",
                ),
            )
        ),
    )
    final_state = _coerce_state(output)
    final_route = trace_events[-1].route if trace_events else final_state.next_route
    response = compose_graph_response(final_state, route=final_route)
    return AssessmentGraphRunResult(
        state=final_state,
        response=response,
        trace_events=trace_events,
    )


def build_initial_state(
    request_or_state: GraphRequest | PetCareGraphState | dict[str, Any],
) -> PetCareGraphState:
    """Convert a graph request or existing state into graph state."""

    if isinstance(request_or_state, PetCareGraphState):
        return request_or_state.model_copy(deep=True)

    if isinstance(request_or_state, GraphRequest):
        request = request_or_state
    else:
        try:
            request = GraphRequest.model_validate(request_or_state)
        except Exception:
            return PetCareGraphState.model_validate(request_or_state)

    return PetCareGraphState(
        user_input=request.user_input,
        conversation_history=[
            message.model_copy(deep=True) for message in request.conversation_history
        ],
        request_id=request.request_id,
        conversation_id=request.conversation_id,
        pet_id=request.pet_id,
        locale=request.locale,
        timezone=request.timezone,
        next_route="intent_classifier",
    )


def build_node_trace_metadata(
    node_name: str,
    state: PetCareGraphState,
) -> NodeTraceMetadata:
    """Build the per-node metadata recorded by graph runner hooks."""

    route = _NODE_ROUTE_BY_NAME[node_name]
    triggered_rules = [rule.rule_id for rule in state.emergency_screening.triggered_rules]
    return NodeTraceMetadata(
        node_name=node_name,
        route=route,
        intent=state.intent,
        risk_level=state.risk_level,
        triggered_rules=triggered_rules,
        next_route=state.next_route,
        metadata=build_state_trace_metadata(state, node_name=node_name, route=route),
    )


def _collector_hook(
    trace_events: list[NodeTraceMetadata],
    downstream_hook: TraceMetadataHook | None,
) -> TraceMetadataHook:
    def collect(event: NodeTraceMetadata) -> None:
        trace_events.append(event)
        if downstream_hook is not None:
            downstream_hook(event)

    return collect


def _wrap_node(
    node_name: str,
    node_func: NodeCallable,
    dependencies: AssessmentGraphDependencies,
) -> NodeCallable:
    def wrapped(state: PetCareGraphState) -> PetCareGraphState:
        input_state = _coerce_state(state)
        with trace_span(
            f"assessment_graph.{node_name}",
            inputs={"node_name": node_name},
            context=TraceContext(
                request_id=input_state.request_id,
                conversation_id=input_state.conversation_id,
                node_name=node_name,
                route=_NODE_ROUTE_BY_NAME[node_name],
                metadata=build_state_trace_metadata(
                    input_state,
                    node_name=node_name,
                    route=_NODE_ROUTE_BY_NAME[node_name],
                ),
            ),
        ):
            output_state = _coerce_state(node_func(input_state))

        event = build_node_trace_metadata(node_name, output_state)
        if dependencies.trace_metadata_hook is not None:
            dependencies.trace_metadata_hook(event)
        return output_state

    return wrapped



def _route_after_db_context_loader(state: PetCareGraphState) -> str:
    routed_state = _coerce_state(state)
    if (
        routed_state.intent == "social_chat"
        and not routed_state.requires_safety_screening
        and not routed_state.red_flag_mentioned
    ):
        return "chat_agent"
    if (
        routed_state.requires_safety_screening
        or routed_state.red_flag_mentioned
        or routed_state.intent in {"symptom_check", "followup", "handoff_request"}
    ):
        return "baseline_builder"
    return "evidence_planner"


def _route_after_safety_guard(state: PetCareGraphState) -> str:
    routed_state = _coerce_state(state)
    if routed_state.next_route == "question_manager":
        return "question_manager"
    if routed_state.risk_level == "emergency" or routed_state.next_route == "emergency":
        return "emergency_agent"
    return "evidence_planner"


def _route_after_answer_guard(state: PetCareGraphState) -> str:
    routed_state = _coerce_state(state)
    if should_build_non_emergency_handoff(routed_state):
        return "handoff_subgraph"
    return "end"



def _coerce_state(state: PetCareGraphState | dict[str, Any]) -> PetCareGraphState:
    if isinstance(state, PetCareGraphState):
        return state
    return PetCareGraphState.model_validate(state)


__all__ = [
    "AssessmentGraphDependencies",
    "AssessmentGraphRunResult",
    "NodeTraceMetadata",
    "build_initial_state",
    "build_node_trace_metadata",
    "compile_assessment_graph",
    "run_assessment_graph",
]



