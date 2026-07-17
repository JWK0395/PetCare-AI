"""Harness adapter for the current LangGraph Assessment Graph."""

from __future__ import annotations

from dataclasses import dataclass

from petcare_agent.graphs.assessment_graph import (
    AssessmentGraphDependencies,
    run_assessment_graph,
)
from petcare_agent.harness.adapter import (
    AgentSessionConfig,
    AgentTurnResult,
)
from petcare_agent.llm.client import StructuredOutputClient
from petcare_agent.nodes.db_context_loader import DBContextProvider
from petcare_agent.rag.adapter import RAGAdapter
from petcare_agent.schemas.graph_state import (
    AnswerGuardState,
    ConversationMessage,
    HandoffState,
    PetCareGraphState,
    RetrievalState,
)


@dataclass(frozen=True)
class CurrentAssessmentGraphAdapter:
    """Adapter that fixes the harness contract to the current agent models."""

    name: str = "current-assessment-graph"

    def start_session(
        self,
        *,
        config: AgentSessionConfig,
        context_provider: DBContextProvider,
        rag_adapter: RAGAdapter | None = None,
        llm_client: StructuredOutputClient | None = None,
    ) -> "CurrentAssessmentGraphSession":
        return CurrentAssessmentGraphSession(
            config=config,
            context_provider=context_provider,
            rag_adapter=rag_adapter,
            llm_client=llm_client,
        )


class CurrentAssessmentGraphSession:
    """Stateful session for repeatedly invoking the current graph."""

    def __init__(
        self,
        *,
        config: AgentSessionConfig,
        context_provider: DBContextProvider,
        rag_adapter: RAGAdapter | None,
        llm_client: StructuredOutputClient | None,
    ) -> None:
        self.config = config
        self.context_provider = context_provider
        self.rag_adapter = rag_adapter
        self.llm_client = llm_client
        self._turn_index = 0
        self._state = PetCareGraphState(
            pet_id=config.pet_id,
            conversation_id=config.conversation_id,
            locale=config.locale,
            timezone=config.timezone,
            next_route="intent_classifier",
        )

    @property
    def state(self) -> PetCareGraphState:
        """Return a defensive copy of the current graph state."""

        return self._state.model_copy(deep=True)

    def set_hospital_visit_intent(self, intent: str) -> None:
        """Set visit intent for local handoff testing commands."""

        if intent not in {"yes", "no", "undecided", "not_asked"}:
            raise ValueError("intent must be one of yes, no, undecided, not_asked")
        self._state = self._state.model_copy(
            update={"hospital_visit_intent": intent},
            deep=True,
        )

    def handle_user_message(self, user_input: str) -> AgentTurnResult:
        clean_input = user_input.strip()
        if not clean_input:
            raise ValueError("user_input must not be empty")

        self._turn_index += 1
        prepared_state = self._prepare_turn_state(clean_input)
        result = run_assessment_graph(
            prepared_state,
            dependencies=AssessmentGraphDependencies(
                llm_client=self.llm_client,
                db_context_provider=self.context_provider,
                rag_adapter=self.rag_adapter,
                db_context_days=self.config.db_context_days,
                rag_top_k=self.config.rag_top_k,
            ),
        )

        conversation_history = [
            *self._state.conversation_history,
            ConversationMessage(role="user", content=clean_input),
            ConversationMessage(role="assistant", content=result.response.assistant_message),
        ]
        self._state = result.state.model_copy(
            update={"conversation_history": conversation_history},
            deep=True,
        )
        return AgentTurnResult(
            response=result.response,
            state=self._state.model_copy(deep=True),
            trace_events=result.trace_events,
        )

    def _prepare_turn_state(self, user_input: str) -> PetCareGraphState:
        return self._state.model_copy(
            update={
                "user_input": user_input,
                "request_id": self._request_id(),
                "conversation_id": self.config.conversation_id,
                "pet_id": self.config.pet_id,
                "locale": self.config.locale,
                "timezone": self.config.timezone,
                "next_route": "intent_classifier",
                "turn_state_extracted": False,
                "social_response_ready": False,
                "retrieval": RetrievalState(),
                "chat_response": "",
                "answer_guard": AnswerGuardState(),
                "handoff": HandoffState(),
            },
            deep=True,
        )

    def _request_id(self) -> str:
        safe_conversation_id = "".join(
            char if char.isalnum() else "_"
            for char in self.config.conversation_id
        ).strip("_")
        prefix = safe_conversation_id or "local"
        return f"req_{prefix}_{self._turn_index:04d}"


__all__ = ["CurrentAssessmentGraphAdapter", "CurrentAssessmentGraphSession"]
