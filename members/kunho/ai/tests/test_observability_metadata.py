from __future__ import annotations

import json
import sys
from types import SimpleNamespace
from typing import Any

from pydantic import BaseModel

from petcare_agent.config import PetCareSettings
from petcare_agent.graphs.assessment_graph import (
    AssessmentGraphDependencies,
    NodeTraceMetadata,
    run_assessment_graph,
)
from petcare_agent.rag.adapter import RetrievedChunk
from petcare_agent.safety.checklist_loader import load_checklist_template
from petcare_agent.schemas.graph_state import (
    AnswerGuardState,
    ChangeDetection,
    ConversationMessage,
    CurrentStatus,
    HandoffState,
    PetCareContext,
    PetCareGraphState,
)
from petcare_agent.schemas.llm_outputs import (
    AnswerGuardReviewOutput,
    ChecklistExtractionOutput,
    IntentClassificationOutput,
    StateExtractionOutput,
    TurnUnderstandingOutput,
)
from petcare_agent.schemas.triage import ChecklistItem, RuleHit
from petcare_agent.tracing import (
    TRACE_SCHEMA_VERSION,
    TraceContext,
    build_metadata,
    build_state_trace_metadata,
    sanitize_trace_metadata,
    trace_span,
)


RAW_USER_INPUT = "RAW_USER_INPUT_SECRET"
RAW_HISTORY = "RAW_CONVERSATION_HISTORY_SECRET"
RAW_DAILY = "RAW_DAILY_ENTRY_TEXT_SECRET"
RAW_DIAGNOSIS = "RAW_DIAGNOSIS_CONTENT_SECRET"
RAW_RAG_TEXT = "RAW_RAG_CHUNK_TEXT_SECRET"
RAW_EVIDENCE = "RAW_CHECKLIST_EVIDENCE_SECRET"
RAW_EMAIL = "RAW_EMAIL_DRAFT_SECRET"


class ObservabilityFakeLLM:
    def __init__(self) -> None:
        self.calls: list[type[BaseModel]] = []

    def structured_output(self, **kwargs: Any) -> BaseModel:
        output_model = kwargs["output_model"]
        self.calls.append(output_model)
        if output_model is TurnUnderstandingOutput:
            return TurnUnderstandingOutput(
                intent="symptom_check",
                confidence="high",
                chief_complaint="cough",
                requires_db_context=True,
                requires_safety_screening=True,
                red_flag_mentioned=False,
                state=StateExtractionOutput(
                    species="cat",
                    symptoms=["coughing"],
                    duration=None,
                    current_status=CurrentStatus(symptoms=["coughing"]),
                    negated_findings=[],
                    uncertain_findings=[],
                ),
                social_chat=None,
            )
        if output_model is ChecklistExtractionOutput:
            return _all_required_updates("cat_cough_triage", False)
        if output_model is AnswerGuardReviewOutput:
            return AnswerGuardReviewOutput(
                status="revised",
                unsafe_phrases=["unsafe raw phrase"],
                revised_answer="Safe revised response.",
            )
        raise AssertionError(f"Unexpected output model: {output_model}")


class ObservabilityDBProvider:
    def load_context(self, pet_id: int, *, days: int = 3) -> PetCareContext:
        return PetCareContext(
            pet={"id": pet_id, "name": "Private Pet Name", "species": "cat"},
            recent_daily_entries=[
                {
                    "record_date": "2026-07-15",
                    "raw_text": RAW_DAILY,
                    "food": "normal",
                    "water": "normal",
                    "activity": "normal",
                    "symptom": "none",
                    "stool": "normal",
                    "vomit": "none",
                },
                {
                    "record_date": "2026-07-14",
                    "food": "normal",
                    "water": "normal",
                    "activity": "normal",
                    "symptom": "none",
                    "stool": "normal",
                    "vomit": "none",
                },
                {
                    "record_date": "2026-07-13",
                    "food": "normal",
                    "water": "normal",
                    "activity": "normal",
                    "symptom": "none",
                    "stool": "normal",
                    "vomit": "none",
                },
            ],
            diagnoses=[{"id": 1, "diagnosis": "private diagnosis", "content": RAW_DIAGNOSIS}],
            data_from="2026-07-13",
            data_to="2026-07-15",
        )


class ObservabilityRAGAdapter:
    def retrieve(
        self,
        query: str,
        filters: dict[str, object],
        top_k: int = 5,
    ) -> list[RetrievedChunk]:
        return [
            RetrievedChunk(
                chunk_id="chunk_obs_001",
                source_id="care_guide_cough_cat",
                title="Cat cough care guide",
                text=RAW_RAG_TEXT,
                score=0.91,
                metadata={"species": "cat", "topic": "cough", "raw_text": RAW_DAILY},
            )
        ]


def test_state_trace_metadata_contains_phase11_fields_without_raw_text() -> None:
    state = _sensitive_state()

    metadata = build_state_trace_metadata(state, node_name="answer_guard", route="answer_guard")
    serialized = json.dumps(metadata, ensure_ascii=False, sort_keys=True)

    assert metadata["trace_schema_version"] == TRACE_SCHEMA_VERSION
    assert metadata["request_context"] == {
        "request_id": "req_obs_001",
        "conversation_id": "conv_obs_001",
        "pet_id": 777,
        "locale": "ko-KR",
        "timezone": "Asia/Seoul",
    }
    assert metadata["route"] == "answer_guard"
    assert metadata["intent"] == "symptom_check"
    assert metadata["risk_level"] == "urgent"
    assert metadata["checklist"]["checklist_id"] == "cat_cough_triage"
    assert metadata["checklist"]["value_counts"]["true"] == 1
    assert metadata["safety"]["triggered_rules"] == ["U_RESP_001"]
    assert metadata["change_detection"]["baseline_deviation"] is True
    assert metadata["rag"]["chunk_count"] == 1
    assert metadata["rag"]["chunks"][0]["chunk_id"] == "chunk_obs_001"
    assert metadata["rag"]["chunks"][0]["metadata"] == {"species": "cat", "topic": "cough"}
    assert metadata["answer_guard"] == {"status": "revised", "revision_count": 1}
    assert metadata["handoff"] == {
        "required": True,
        "type": "non_emergency",
        "summary_present": True,
        "email_draft_present": True,
    }

    for raw_value in [
        RAW_USER_INPUT,
        RAW_HISTORY,
        RAW_DAILY,
        RAW_DIAGNOSIS,
        RAW_RAG_TEXT,
        RAW_EVIDENCE,
        RAW_EMAIL,
    ]:
        assert raw_value not in serialized


def test_graph_runner_hook_receives_sanitized_node_metadata() -> None:
    hook_events: list[NodeTraceMetadata] = []

    result = run_assessment_graph(
        PetCareGraphState(
            request_id="req_obs_hook",
            conversation_id="conv_obs_hook",
            pet_id=42,
            user_input=RAW_USER_INPUT,
        ),
        dependencies=AssessmentGraphDependencies(
            llm_client=ObservabilityFakeLLM(),
            db_context_provider=ObservabilityDBProvider(),
            rag_adapter=ObservabilityRAGAdapter(),
            trace_metadata_hook=hook_events.append,
        ),
    )

    assert [event.node_name for event in hook_events] == [
        event.node_name for event in result.trace_events
    ]
    answer_event = hook_events[-1]
    metadata = answer_event.metadata
    serialized = json.dumps(metadata, ensure_ascii=False, sort_keys=True)

    assert answer_event.node_name == "answer_guard"
    assert metadata["request_context"]["request_id"] == "req_obs_hook"
    assert metadata["request_context"]["conversation_id"] == "conv_obs_hook"
    assert metadata["request_context"]["pet_id"] == 42
    assert metadata["route"] == "answer_guard"
    assert metadata["intent"] == "symptom_check"
    assert metadata["risk_level"] == "non_emergency"
    assert metadata["triggered_rules"]
    assert metadata["checklist"]["checklist_id"] == "cat_cough_triage"
    assert metadata["change_detection"]["baseline_available"] is True
    assert metadata["rag"]["chunk_count"] == 1
    assert metadata["rag"]["chunks"][0]["source_id"] == "care_guide_cough_cat"
    assert metadata["answer_guard"]["status"] == "revised"
    assert metadata["handoff"]["required"] is False

    for raw_value in [RAW_USER_INPUT, RAW_DAILY, RAW_DIAGNOSIS, RAW_RAG_TEXT]:
        assert raw_value not in serialized


def test_build_metadata_sanitizes_custom_context_metadata() -> None:
    metadata = build_metadata(
        TraceContext(
            request_id="req_safe",
            conversation_id="conv_safe",
            node_name="intent_classifier",
            route="intent_classifier",
            metadata={
                "user_input": RAW_USER_INPUT,
                "nested": {"raw_text": RAW_DAILY, "safe_count": 1},
            },
        )
    )

    assert metadata["request_id"] == "req_safe"
    assert metadata["nested"] == {"safe_count": 1}
    assert RAW_USER_INPUT not in json.dumps(metadata, ensure_ascii=False)
    assert RAW_DAILY not in json.dumps(metadata, ensure_ascii=False)


def test_langsmith_enabled_trace_span_can_use_mock_without_external_call(monkeypatch) -> None:
    calls: list[dict[str, Any]] = []

    class FakeTraceRun:
        def __enter__(self) -> str:
            return "fake-run"

        def __exit__(self, exc_type: object, exc: object, tb: object) -> bool:
            return False

    def fake_trace(*args: Any, **kwargs: Any) -> FakeTraceRun:
        calls.append({"args": args, **kwargs})
        return FakeTraceRun()

    monkeypatch.setitem(sys.modules, "langsmith", SimpleNamespace(trace=fake_trace))
    settings = PetCareSettings(
        _env_file=None,
        LANGSMITH_TRACING=True,
        LANGSMITH_API_KEY="test-key",
    )

    with trace_span(
        "assessment_graph.intent_classifier",
        inputs={"user_input": RAW_USER_INPUT, "safe": True},
        context=TraceContext(
            request_id="req_enabled",
            conversation_id="conv_enabled",
            node_name="intent_classifier",
            route="intent_classifier",
        ),
        settings=settings,
    ) as run:
        assert run == "fake-run"

    assert calls[0]["args"] == ("assessment_graph.intent_classifier",)
    assert calls[0]["inputs"] == {"safe": True}
    assert calls[0]["metadata"]["request_id"] == "req_enabled"


def test_trace_metadata_sanitizer_removes_sensitive_nested_keys() -> None:
    assert sanitize_trace_metadata(
        {
            "safe": "value",
            "content": RAW_DIAGNOSIS,
            "items": [{"text": RAW_RAG_TEXT, "source_id": "guide"}],
        }
    ) == {"safe": "value", "items": [{"source_id": "guide"}]}


def _sensitive_state() -> PetCareGraphState:
    state = PetCareGraphState(
        user_input=RAW_USER_INPUT,
        conversation_history=[ConversationMessage(role="user", content=RAW_HISTORY)],
        request_id="req_obs_001",
        conversation_id="conv_obs_001",
        pet_id=777,
        intent="symptom_check",
        species="cat",
        requires_db_context=True,
        requires_safety_screening=True,
        risk_level="urgent",
        confidence="medium",
        next_route="answer_guard",
        context=PetCareContext(
            recent_daily_entries=[{"record_date": "2026-07-15", "raw_text": RAW_DAILY}],
            diagnoses=[{"id": 1, "content": RAW_DIAGNOSIS}],
        ),
        change_detection=ChangeDetection(
            baseline_available=True,
            new_symptoms=["coughing"],
            worsened_fields=["activity"],
            baseline_deviation=True,
            summary="New symptoms reported: coughing.",
        ),
        answer_guard=AnswerGuardState(status="revised", revisions=["unsafe phrase"]),
        handoff=HandoffState(
            type="non_emergency",
            required=True,
            summary="Summary exists",
            email_draft=RAW_EMAIL,
        ),
    )
    state.emergency_screening.checklist_id = "cat_cough_triage"
    state.emergency_screening.chief_complaint = "cough"
    state.emergency_screening.status = "complete"
    state.emergency_screening.items = {
        "rapid_breathing": ChecklistItem(
            item_id="rapid_breathing",
            label="Rapid breathing",
            type="boolean",
            value=True,
            confidence="high",
            metadata={"evidence": RAW_EVIDENCE, "red_flag": True},
        ),
        "gum_color_abnormal": ChecklistItem(
            item_id="gum_color_abnormal",
            label="Abnormal gum color",
            type="boolean",
            value=None,
            confidence="unknown",
            metadata={"red_flag": True},
        ),
    }
    state.emergency_screening.red_flags = ["rapid_breathing"]
    state.emergency_screening.triggered_rules = [
        RuleHit(rule_id="U_RESP_001", result="urgent", condition="rapid_breathing == true")
    ]
    state.retrieval.query = f"query with {RAW_USER_INPUT}"
    state.retrieval.chunks = [
        RetrievedChunk(
            chunk_id="chunk_obs_001",
            source_id="care_guide_cough_cat",
            title="Cat cough care guide",
            text=RAW_RAG_TEXT,
            score=0.91,
            metadata={"species": "cat", "topic": "cough", "raw_text": RAW_DAILY},
        )
    ]
    return state


def _all_required_updates(checklist_id: str, value: bool) -> ChecklistExtractionOutput:
    template = load_checklist_template(checklist_id)
    return ChecklistExtractionOutput(
        checklist_id=checklist_id,
        updates=[
            {
                "item_id": item.item_id,
                "value": value,
                "confidence": "high",
                "evidence": RAW_EVIDENCE,
            }
            for item in template.required_items
        ],
    )

