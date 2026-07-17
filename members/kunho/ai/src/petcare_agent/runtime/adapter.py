"""Backend-facing runtime adapter for the Assessment Graph."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from petcare_agent.api.handoff_context import (
    ExistingAPIHandoffContextProvider,
    HandoffContextClient,
)
from petcare_agent.graphs.assessment_graph import (
    AssessmentGraphDependencies,
    NodeTraceMetadata,
    run_assessment_graph,
)
from petcare_agent.llm.client import StructuredOutputClient
from petcare_agent.localization import wants_korean
from petcare_agent.nodes.db_context_loader import DBContextProvider
from petcare_agent.rag.adapter import RAGAdapter
from petcare_agent.rag.cornell import CornellRAGAdapter
from petcare_agent.runtime.validation import (
    validate_graph_request_payload,
    validate_graph_response_payload,
)
from petcare_agent.schemas.graph_state import (
    EmergencyResponse,
    GraphRequest,
    GraphResponse,
    HandoffResponse,
    PetCareGraphState,
)

SAFE_FALLBACK_MESSAGE = (
    "The assessment service could not complete this request safely. "
    "Please try again, and seek veterinary care promptly if serious symptoms are present."
)

SAFE_FALLBACK_MESSAGE_KO = (
    "\ud604\uc7ac \uc694\uccad\uc744 \uc548\uc804\ud558\uac8c \uc644\ub8cc\ud558\uc9c0 \ubabb\ud588\uc2b5\ub2c8\ub2e4. \ub2e4\uc2dc \uc2dc\ub3c4\ud574 \uc8fc\uc138\uc694. "
    "\uc2ec\uac01\ud55c \uc99d\uc0c1\uc774 \uc788\ub2e4\uba74 \uc9c0\uccb4\ud558\uc9c0 \ub9d0\uace0 \uc218\uc758\uc0ac \uc9c4\ub8cc\ub97c \ubc1b\uc544 \uc8fc\uc138\uc694."
)

@dataclass(frozen=True)
class GraphRuntimeResult:
    """Runtime adapter result for backend callers."""

    response: GraphResponse
    state: PetCareGraphState | None
    trace_events: list[NodeTraceMetadata]
    fallback_reason: str | None = None


@dataclass(frozen=True)
class GraphRuntimeAdapter:
    """Minimal callable boundary for B.E -> Assessment Graph execution."""

    dependencies: AssessmentGraphDependencies = AssessmentGraphDependencies()

    def run(self, payload: GraphRequest | Mapping[str, Any]) -> GraphRuntimeResult:
        return run_graph_request(payload, dependencies=self.dependencies)


def build_existing_api_runtime_adapter(
    *,
    api_client: HandoffContextClient | None = None,
    db_context_provider: DBContextProvider | None = None,
    llm_client: StructuredOutputClient | None = None,
    rag_adapter: RAGAdapter | None = None,
) -> GraphRuntimeAdapter:
    """Build a runtime adapter wired to the documented handoff-context API."""

    provider = db_context_provider or ExistingAPIHandoffContextProvider(client=api_client)
    official_source_rag = rag_adapter or CornellRAGAdapter()
    return GraphRuntimeAdapter(
        dependencies=AssessmentGraphDependencies(
            llm_client=llm_client,
            db_context_provider=provider,
            rag_adapter=official_source_rag,
            db_context_days=3,
        )
    )


def run_graph_request(
    payload: GraphRequest | Mapping[str, Any],
    *,
    dependencies: AssessmentGraphDependencies | None = None,
) -> GraphRuntimeResult:
    """Validate a GraphRequest payload, run the graph, and validate response."""

    request = validate_graph_request_payload(payload)
    try:
        result = run_assessment_graph(request, dependencies=dependencies)
        response = validate_graph_response_payload(result.response)
    except Exception as exc:
        response = safe_fallback_response(request)
        return GraphRuntimeResult(
            response=response,
            state=None,
            trace_events=[],
            fallback_reason=exc.__class__.__name__,
        )

    return GraphRuntimeResult(
        response=response,
        state=result.state,
        trace_events=result.trace_events,
    )


def safe_fallback_response(request: GraphRequest | Mapping[str, Any]) -> GraphResponse:
    """Return a schema-valid conservative response for runtime failures."""

    if isinstance(request, GraphRequest):
        request_id = request.request_id
        conversation_id = request.conversation_id
        locale = request.locale
    else:
        request_id = str(request.get("request_id") or "runtime_error")
        conversation_id = str(request.get("conversation_id") or "unknown_conversation")
        locale = str(request.get("locale") or "")

    response = GraphResponse(
        response_id=_response_id(request_id),
        conversation_id=conversation_id,
        route="end",
        risk_level="unknown",
        assistant_message=(
            SAFE_FALLBACK_MESSAGE_KO if wants_korean(locale) else SAFE_FALLBACK_MESSAGE
        ),
        needs_user_response=False,
        follow_up_question=None,
        handoff=HandoffResponse(type="none", summary=None, email_draft=None),
        emergency=EmergencyResponse(is_emergency=False, triggered_rules=[]),
    )
    return validate_graph_response_payload(response)


def _response_id(request_id: str) -> str:
    if request_id.startswith("req_"):
        return request_id.replace("req_", "res_", 1)
    return f"res_{request_id}"


__all__ = [
    "GraphRuntimeAdapter",
    "GraphRuntimeResult",
    "build_existing_api_runtime_adapter",
    "run_graph_request",
    "safe_fallback_response",
]