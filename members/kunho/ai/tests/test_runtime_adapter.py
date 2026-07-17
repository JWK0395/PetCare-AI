from __future__ import annotations

from typing import Any, Mapping

from pydantic import BaseModel

from petcare_agent.api.handoff_context import HandoffContextAPIError
from petcare_agent.rag.adapter import RetrievedChunk
from petcare_agent.rag.cornell import CornellRAGAdapter
from petcare_agent.runtime.adapter import (
    build_existing_api_runtime_adapter,
    safe_fallback_response,
)
from petcare_agent.runtime.validation import (
    ContractValidationError,
    validate_graph_request_payload,
    validate_graph_response_payload,
)
from petcare_agent.safety.checklist_loader import load_checklist_template
from petcare_agent.schemas.graph_state import CurrentStatus
from petcare_agent.schemas.llm_outputs import (
    AnswerGuardReviewOutput,
    ChecklistExtractionOutput,
    IntentClassificationOutput,
    StateExtractionOutput,
    TurnUnderstandingOutput,
)


class RuntimeFakeLLM:
    def __init__(
        self,
        *,
        intent: IntentClassificationOutput,
        state: StateExtractionOutput,
        checklist: ChecklistExtractionOutput,
    ) -> None:
        self.intent = intent
        self.state = state
        self.checklist = checklist
        self.calls: list[type[BaseModel]] = []

    def structured_output(self, **kwargs):
        output_model = kwargs["output_model"]
        self.calls.append(output_model)
        if output_model is TurnUnderstandingOutput:
            return TurnUnderstandingOutput(
                intent=self.intent.intent,
                confidence=self.intent.confidence,
                chief_complaint=self.intent.chief_complaint,
                requires_db_context=self.intent.requires_db_context,
                requires_safety_screening=self.intent.requires_safety_screening,
                red_flag_mentioned=self.intent.red_flag_mentioned,
                state=self.state,
                social_chat=None,
            )
        if output_model is ChecklistExtractionOutput:
            return self.checklist
        if output_model is AnswerGuardReviewOutput:
            return AnswerGuardReviewOutput(
                status="passed",
                unsafe_phrases=[],
                revised_answer=None,
            )
        raise AssertionError(f"Unexpected output model: {output_model}")


class RuntimeRecordingAPIClient:
    def __init__(self, payload: Mapping[str, Any]) -> None:
        self.payload = payload
        self.calls: list[tuple[str, dict[str, int]]] = []

    def get_json(self, path: str, *, query: Mapping[str, int]) -> Mapping[str, Any]:
        self.calls.append((path, dict(query)))
        return self.payload


class RuntimeFailingAPIClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, int]]] = []

    def get_json(self, path: str, *, query: Mapping[str, int]) -> Mapping[str, Any]:
        self.calls.append((path, dict(query)))
        raise HandoffContextAPIError("mocked provider failure")


class RuntimeRAGAdapter:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, object], int]] = []

    def retrieve(
        self,
        query: str,
        filters: dict[str, object],
        top_k: int = 5,
    ) -> list[RetrievedChunk]:
        self.calls.append((query, filters, top_k))
        return []


def test_existing_api_runtime_adapter_uses_cornell_rag_by_default() -> None:
    adapter = build_existing_api_runtime_adapter(
        api_client=RuntimeRecordingAPIClient(_handoff_context_payload()),
        llm_client=_non_emergency_llm(),
    )

    assert isinstance(adapter.dependencies.rag_adapter, CornellRAGAdapter)

def test_graph_request_dict_runs_through_runtime_adapter_into_graph_state() -> None:
    api_client = RuntimeRecordingAPIClient(_handoff_context_payload())
    rag_adapter = RuntimeRAGAdapter()
    adapter = build_existing_api_runtime_adapter(
        api_client=api_client,
        llm_client=_non_emergency_llm(),
        rag_adapter=rag_adapter,
    )

    result = adapter.run(_graph_request_payload())

    assert result.fallback_reason is None
    assert result.state is not None
    assert result.state.request_id == "req_phase10_001"
    assert result.state.conversation_id == "conv_phase10"
    assert result.state.pet_id == 42
    assert result.state.user_input == "My cat is coughing today."
    assert result.state.context.pet["name"] == "Kongi"
    assert api_client.calls == [("/api/pets/42/handoff-context", {"days": 3})]
    assert [event.node_name for event in result.trace_events][:2] == [
        "db_context_loader",
        "intent_classifier",
    ]
    assert result.response.route == "answer_guard"
    assert validate_graph_response_payload(result.response) == result.response


def test_provider_failure_returns_safe_schema_valid_graph_response() -> None:
    api_client = RuntimeFailingAPIClient()
    adapter = build_existing_api_runtime_adapter(
        api_client=api_client,
        llm_client=_non_emergency_llm(),
        rag_adapter=RuntimeRAGAdapter(),
    )

    result = adapter.run(_graph_request_payload())

    assert result.fallback_reason is None
    assert result.state is not None
    assert "db_context_unavailable" in result.state.context.unknown_items
    assert api_client.calls == [("/api/pets/42/handoff-context", {"days": 3})]
    assert result.response.risk_level in {"unknown", "non_emergency"}
    assert validate_graph_response_payload(result.response) == result.response


def test_safe_fallback_response_matches_pydantic_and_json_schema_contracts() -> None:
    request = validate_graph_request_payload(_graph_request_payload())

    response = safe_fallback_response(request)

    assert response.route == "end"
    assert response.risk_level == "unknown"
    assert response.handoff.type == "none"
    assert response.emergency.is_emergency is False
    assert validate_graph_response_payload(response) == response


def test_graph_request_validation_rejects_extra_fields() -> None:
    payload = {**_graph_request_payload(), "new_field": "not allowed"}

    try:
        validate_graph_request_payload(payload)
    except ContractValidationError:
        pass
    else:  # pragma: no cover - explicit failure branch for clarity.
        raise AssertionError("extra GraphRequest fields must be rejected")


def _non_emergency_llm() -> RuntimeFakeLLM:
    return RuntimeFakeLLM(
        intent=IntentClassificationOutput(
            intent="symptom_check",
            confidence="high",
            chief_complaint="cough",
            requires_db_context=True,
            requires_safety_screening=True,
            red_flag_mentioned=False,
        ),
        state=StateExtractionOutput(
            species="cat",
            symptoms=["coughing"],
            duration=None,
            current_status=CurrentStatus(symptoms=["coughing"]),
            negated_findings=[],
            uncertain_findings=[],
        ),
        checklist=_all_required_updates("cat_cough_triage", False),
    )


def _all_required_updates(checklist_id: str, value: bool) -> ChecklistExtractionOutput:
    template = load_checklist_template(checklist_id)
    return ChecklistExtractionOutput(
        checklist_id=checklist_id,
        updates=[
            {
                "item_id": item.item_id,
                "value": value,
                "confidence": "high",
                "evidence": "mocked runtime test",
            }
            for item in template.required_items
        ],
    )


def _graph_request_payload() -> dict[str, Any]:
    return {
        "request_id": "req_phase10_001",
        "conversation_id": "conv_phase10",
        "pet_id": 42,
        "user_input": "My cat is coughing today.",
        "locale": "ko-KR",
        "timezone": "Asia/Seoul",
        "timestamp": "2026-07-16T09:00:00+09:00",
    }


def _handoff_context_payload() -> dict[str, Any]:
    return {
        "pet": {"id": 42, "name": "Kongi", "species": "cat"},
        "recent_daily_entries": [
            _daily_entry("2026-07-15"),
            _daily_entry("2026-07-14"),
            _daily_entry("2026-07-13"),
        ],
        "diagnoses": [],
        "unknown_items": [],
        "data_from": "2026-07-13",
        "data_to": "2026-07-15",
        "generated_at": "2026-07-15T16:35:00+09:00",
    }


def _daily_entry(record_date: str) -> dict[str, str]:
    return {
        "record_date": record_date,
        "food": "normal",
        "water": "normal",
        "activity": "normal",
        "symptom": "none",
        "stool": "normal",
        "vomit": "none",
    }
