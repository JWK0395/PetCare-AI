from __future__ import annotations

from typing import Any

from langgraph.types import Command

from .graph import petcare_graph
from .models import (
    GraphStartRequest,
    GraphStepResult,
    PetCareState,
)
from .nodes.triage import (
    default_question_strategy,
    should_open_new_triage,
)
from .utils import trim_conversation_history


def _namespace_names(namespace: tuple[str, ...]) -> list[str]:
    return [
        item.split(":", maxsplit=1)[0]
        for item in namespace
    ]


def _trace_label(
    namespace: tuple[str, ...],
    node_name: str,
) -> str:
    parents = _namespace_names(namespace)

    if parents:
        return ":".join([*parents, node_name])

    return node_name


def _append_trace(
    trace: list[str],
    *,
    namespace: tuple[str, ...],
    node_name: str,
) -> None:

                                                          
    if (
        not namespace
        and node_name in {"assessment_graph", "handoff_subgraph"}
    ):
        return

    label = _trace_label(namespace, node_name)

    if not trace or trace[-1] != label:
        trace.append(label)


def stream_graph_once(
    graph_input: Any,
    *,
    config: dict[str, Any],
) -> tuple[PetCareState, list[str], dict[str, Any] | None]:
    trace: list[str] = []
    latest_state: PetCareState = {}
    interrupt_payload: dict[str, Any] | None = None

    for part in petcare_graph.stream(
        graph_input,
        config=config,
        stream_mode=["updates", "values"],
        subgraphs=True,
        version="v2",
    ):
        part_type = part["type"]
        namespace = tuple(part.get("ns", ()))
        data = part["data"]

        if part_type == "updates":
            if "__interrupt__" in data:

                interrupt_items = data["__interrupt__"]
                raw_preview = (
                    getattr(
                        interrupt_items[0],
                        "value",
                        interrupt_items[0],
                    )
                    if interrupt_items
                    else {}
                )
                field = (
                    raw_preview.get("field")
                    if isinstance(raw_preview, dict)
                    else None
                )
                interrupt_node = (
                    "hospital_visit_decision"
                    if field == "hospital_visit"
                    else "question_manager"
                )
                _append_trace(
                    trace,
                    namespace=namespace,
                    node_name=interrupt_node,
                )

                interrupt_items = data["__interrupt__"]

                if interrupt_items:
                    raw_value = getattr(
                        interrupt_items[0],
                        "value",
                        interrupt_items[0],
                    )
                    interrupt_payload = (
                        raw_value
                        if isinstance(raw_value, dict)
                        else {"question": str(raw_value)}
                    )
                continue

            for node_name in data:
                _append_trace(
                    trace,
                    namespace=namespace,
                    node_name=node_name,
                )

        elif part_type == "values":
            if isinstance(data, dict):
                latest_state = data


                                      
    snapshot = petcare_graph.get_state(config)

    if snapshot.values:
        latest_state = dict(snapshot.values)

    return latest_state, trace, interrupt_payload


def load_previous_session_context(
    *,
    session_id: str,
) -> dict[str, Any]:
    config = {
        "configurable": {
            "thread_id": session_id,
        }
    }

    snapshot = petcare_graph.get_state(config)
    values = dict(snapshot.values or {})

    history = values.get(
        "conversation_history",
        [],
    )

    normalized_history = [
        {
            "role": str(item.get("role", "")),
            "content": str(item.get("content", "")),
        }
        for item in history
        if isinstance(item, dict)
    ]

    previous_triage: dict[str, Any] = {}

    if values.get("triage_status") == "completed":
        previous_triage = {
            "route": values.get("route"),
            "answer": values.get("answer"),
            "emergency_hits": values.get(
                "emergency_hits",
                [],
            ),
            "recovery_hits": values.get(
                "recovery_hits",
                [],
            ),
            "question_strategy": values.get(
                "question_strategy",
                {},
            ),
            "follow_up_history": values.get(
                "follow_up_history",
                [],
            ),
        }

    return {
        "conversation_history": (
            trim_conversation_history(
                normalized_history
            )
        ),
        "triage_status": values.get(
            "triage_status",
            "idle",
        ),
        "previous_triage": previous_triage,
    }


def request_to_initial_state(
    request_payload: dict[str, Any] | GraphStartRequest,
    *,
    previous_session: dict[str, Any],
) -> tuple[GraphStartRequest, PetCareState]:
    request = (
        request_payload
        if isinstance(request_payload, GraphStartRequest)
        else GraphStartRequest.model_validate(
            request_payload
        )
    )

    previous_history = previous_session.get(
        "conversation_history",
        [],
    )

    conversation_history = list(previous_history)
    conversation_history.append(
        {
            "role": "user",
            "content": request.user_input.strip(),
        }
    )

    previous_completed = (
        previous_session.get("triage_status")
        == "completed"
    )

    opens_new_triage = should_open_new_triage(
        request.user_input
    )

    post_triage_mode = (
        previous_completed
        and not opens_new_triage
    )

    initial_state: PetCareState = {
        "session_id": request.session_id,
        "pet_id": request.pet_id,
        "user_input": request.user_input,
        "backend_context": request.context.model_dump(),
        "diary_summary": "",
        "diagnosis_summary": "",
        "assessment": {},
        "handoff_requested": False,
        "route": (
            "general_chat"
            if post_triage_mode
            else None
        ),
        "conversation_history": (
            trim_conversation_history(
                conversation_history
            )
        ),
        "triage_status": (
            "completed"
            if post_triage_mode
            else "idle"
        ),
        "previous_triage": (
            previous_session.get(
                "previous_triage",
                {},
            )
            if post_triage_mode
            else {}
        ),
        "post_triage_mode": post_triage_mode,
        "question_strategy": default_question_strategy(),
        "follow_up_history": [],
        "needs_user_response": False,
        "emergency_hits": [],
        "recovery_hits": [],
        "rag_query": "",
        "rag_chunks": [],
        "rag_done": False,
        "visit_decision": "pending",
        "nearby_hospitals": [],
        "selected_hospital": {},
        "answer": "",
        "handoff": {},
        "artifact_path": None,
        "email_subject": "",
        "email_body": "",
        "email_delivery": {},
        "latency_ms": {},
        "errors": [],
    }

    return request, initial_state
def start_petcare(
    request_payload: dict[str, Any] | GraphStartRequest,
) -> GraphStepResult:
    request = (
        request_payload
        if isinstance(request_payload, GraphStartRequest)
        else GraphStartRequest.model_validate(
            request_payload
        )
    )

    previous_session = load_previous_session_context(
        session_id=request.session_id,
    )

    request, initial_state = request_to_initial_state(
        request,
        previous_session=previous_session,
    )

    config = {
        "configurable": {
            "thread_id": request.session_id,
        },
        "recursion_limit": 30,
    }

    state, trace, interrupt_payload = stream_graph_once(
        initial_state,
        config=config,
    )

    if interrupt_payload is not None:
        return {
            "status": "waiting_for_user",
            "session_id": request.session_id,
            "state": state,
            "trace": trace,
            "question": str(
                interrupt_payload.get(
                    "question",
                    "추가 확인이 필요합니다.",
                )
            ),
            "field": interrupt_payload.get("field"),
        }

    return {
        "status": "completed",
        "session_id": request.session_id,
        "state": state,
        "trace": trace,
        "question": None,
        "field": None,
    }


def resume_petcare(
    *,
    session_id: str,
    answer: str,
) -> GraphStepResult:
    config = {
        "configurable": {
            "thread_id": session_id,
        },
        "recursion_limit": 30,
    }

    state, trace, interrupt_payload = stream_graph_once(
        Command(resume=answer),
        config=config,
    )

    if interrupt_payload is not None:
        return {
            "status": "waiting_for_user",
            "session_id": session_id,
            "state": state,
            "trace": trace,
            "question": str(
                interrupt_payload.get(
                    "question",
                    "추가 확인이 필요합니다.",
                )
            ),
            "field": interrupt_payload.get("field"),
        }

    return {
        "status": "completed",
        "session_id": session_id,
        "state": state,
        "trace": trace,
        "question": None,
        "field": None,
    }


def run_petcare(
    request_payload: dict[str, Any] | GraphStartRequest,
) -> GraphStepResult:
    return start_petcare(request_payload)
