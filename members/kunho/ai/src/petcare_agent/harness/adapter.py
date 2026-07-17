"""Agent harness contracts.

The shared contract intentionally uses the current Assessment Graph public
models as the fixed boundary. New agent implementations can be swapped into the
same local test pipeline by returning these same result objects.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from petcare_agent.graphs.assessment_graph import NodeTraceMetadata
from petcare_agent.llm.client import StructuredOutputClient
from petcare_agent.nodes.db_context_loader import DBContextProvider
from petcare_agent.rag.adapter import RAGAdapter
from petcare_agent.schemas.graph_state import GraphResponse, PetCareGraphState


@dataclass(frozen=True)
class AgentSessionConfig:
    """Runtime options shared by all harness-compatible agents."""

    pet_id: int
    conversation_id: str
    locale: str = "ko-KR"
    timezone: str = "Asia/Seoul"
    db_context_days: int = 3
    rag_top_k: int = 5


@dataclass(frozen=True)
class AgentTurnResult:
    """Normalized single-turn output for console display and transcript files."""

    response: GraphResponse
    state: PetCareGraphState
    trace_events: list[NodeTraceMetadata] = field(default_factory=list)
    fallback_reason: str | None = None

    @property
    def trace_path(self) -> list[str]:
        """Return compact node names for human-readable debugging."""

        return [event.node_name for event in self.trace_events]


class AgentSession(Protocol):
    """Stateful chat session for one pet and one conversation."""

    def handle_user_message(self, user_input: str) -> AgentTurnResult:
        """Run one user message through the selected agent."""


class AgentAdapter(Protocol):
    """Factory boundary for a harness-compatible agent implementation."""

    name: str

    def start_session(
        self,
        *,
        config: AgentSessionConfig,
        context_provider: DBContextProvider,
        rag_adapter: RAGAdapter | None = None,
        llm_client: StructuredOutputClient | None = None,
    ) -> AgentSession:
        """Create a session wired to the fake backend and optional test doubles."""
