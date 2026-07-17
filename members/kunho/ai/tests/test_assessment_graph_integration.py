from __future__ import annotations

import json

import pytest
from pydantic import BaseModel

from petcare_agent.graphs.assessment_graph import (
    AssessmentGraphDependencies,
    NodeTraceMetadata,
    build_initial_state,
    run_assessment_graph,
)
from petcare_agent.safety.checklist_loader import load_checklist_template
from petcare_agent.schemas.graph_state import (
    CurrentStatus,
    EmergencyScreening,
    PetCareContext,
    PetCareGraphState,
    RetrievedChunk,
)
from petcare_agent.schemas.llm_outputs import (
    AnswerGuardReviewOutput,
    ChecklistExtractionOutput,
    GeneralPetCareAnswerOutput,
    IntentClassificationOutput,
    SocialChatOutput,
    StateExtractionOutput,
    TurnUnderstandingOutput,
)


class GraphFakeLLM:
    def __init__(
        self,
        *,
        intent: IntentClassificationOutput,
        state: StateExtractionOutput | None = None,
        checklist: ChecklistExtractionOutput | None = None,
        answer_guard: AnswerGuardReviewOutput | None = None,
        social_chat: SocialChatOutput | None = None,
        general_petcare_answer: GeneralPetCareAnswerOutput | None = None,
    ) -> None:
        self.intent = intent
        self.state = state
        self.checklist = checklist
        self.answer_guard = answer_guard or AnswerGuardReviewOutput(
            status="passed",
            unsafe_phrases=[],
            revised_answer=None,
        )
        self.social_chat = social_chat or SocialChatOutput(
            assistant_message="Social chat response."
        )
        self.general_petcare_answer = general_petcare_answer or GeneralPetCareAnswerOutput(
            assistant_message="General pet-care answer."
        )
        self.calls: list[type[BaseModel]] = []
        self.user_prompts: list[str] = []

    def structured_output(self, **kwargs):
        output_model = kwargs["output_model"]
        self.calls.append(output_model)
        self.user_prompts.append(kwargs["user_prompt"])
        if output_model is TurnUnderstandingOutput:
            return TurnUnderstandingOutput(
                intent=self.intent.intent,
                confidence=self.intent.confidence,
                chief_complaint=self.intent.chief_complaint,
                requires_db_context=self.intent.requires_db_context,
                requires_safety_screening=self.intent.requires_safety_screening,
                red_flag_mentioned=self.intent.red_flag_mentioned,
                state=self.state
                or StateExtractionOutput(
                    species="unknown",
                    symptoms=[],
                    duration=None,
                    current_status=CurrentStatus(),
                    course_pattern="unknown",
                    negated_findings=[],
                    uncertain_findings=[],
                ),
                social_chat=self.social_chat if self.intent.intent == "social_chat" else None,
            )
        if output_model is ChecklistExtractionOutput:
            if self.checklist is None:
                raise AssertionError("ChecklistExtractionOutput was not expected")
            return self.checklist
        if output_model is AnswerGuardReviewOutput:
            return self.answer_guard
        if output_model is GeneralPetCareAnswerOutput:
            return self.general_petcare_answer
        raise AssertionError(f"Unexpected output model: {output_model}")


class RecordingDBContextProvider:
    def __init__(self) -> None:
        self.calls: list[tuple[int, int]] = []

    def load_context(self, pet_id: int, *, days: int = 3) -> PetCareContext:
        self.calls.append((pet_id, days))
        return PetCareContext(
            pet={"id": pet_id, "name": "Kongi", "species": "cat"},
            recent_daily_entries=[
                _daily_entry("2026-07-15"),
                _daily_entry("2026-07-14"),
                _daily_entry("2026-07-13"),
            ],
            diagnoses=[],
            data_from="2026-07-13",
            data_to="2026-07-15",
        )


class RecordingRAGAdapter:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def retrieve(
        self,
        query: str,
        filters: dict[str, object],
        top_k: int = 5,
    ) -> list[RetrievedChunk]:
        self.calls.append({"query": query, "filters": filters, "top_k": top_k})
        return [
            RetrievedChunk(
                chunk_id="chunk_001",
                source_id="care_guide",
                title="Care guide",
                text="Mocked care guidance.",
                score=0.8,
                metadata={
                    "provider": "cornell",
                    "mocked": True,
                    "canonical_url": "https://www.vet.cornell.edu/mock-care-guide",
                    "section_path": ["Care guide"],
                },
            )
        ]


def test_graph_request_conversation_history_is_copied_to_initial_state() -> None:
    state = build_initial_state(
        {
            "request_id": "req_history_001",
            "conversation_id": "conv_history",
            "pet_id": 1,
            "user_input": "\ub0b4 \uc774\ub984\uc774 \ubb50\ub77c\uace0?",
            "conversation_history": [
                {"role": "user", "content": "\ub0b4 \uc774\ub984\uc740 \uc7a5\uac74\ud638\uc57c"},
                {"role": "assistant", "content": "\ubc18\uac11\uc2b5\ub2c8\ub2e4, \uc7a5\uac74\ud638\ub2d8."},
            ],
            "locale": "ko-KR",
            "timezone": "Asia/Seoul",
            "timestamp": "2026-07-17T09:00:00+09:00",
        }
    )

    assert len(state.conversation_history) == 2
    assert state.conversation_history[0].content == "\ub0b4 \uc774\ub984\uc740 \uc7a5\uac74\ud638\uc57c"


def test_general_chat_routes_intent_to_chat_rag_answer_guard() -> None:
    db_provider = RecordingDBContextProvider()
    rag_adapter = RecordingRAGAdapter()
    hook_events: list[NodeTraceMetadata] = []

    result = run_assessment_graph(
        PetCareGraphState(user_input="How often can cats eat treats?", pet_id=1),
        dependencies=AssessmentGraphDependencies(
            llm_client=GraphFakeLLM(
                intent=IntentClassificationOutput(
                    intent="general_chat",
                    confidence="high",
                    chief_complaint=None,
                    requires_db_context=False,
                    requires_safety_screening=False,
                    red_flag_mentioned=False,
                ),
                general_petcare_answer=GeneralPetCareAnswerOutput(
                    assistant_message="Cats should eat treats only in small amounts."
                ),
            ),
            db_context_provider=db_provider,
            rag_adapter=rag_adapter,
            trace_metadata_hook=hook_events.append,
        ),
    )

    assert _path(result.trace_events) == [
        "db_context_loader",
        "intent_classifier",
        "evidence_planner",
        "rag_agent",
        "answer_composer",
        "answer_guard",
    ]
    assert _path(hook_events) == _path(result.trace_events)
    assert hook_events[1].metadata["node_name"] == "intent_classifier"
    assert "risk_level" in hook_events[1].metadata
    assert "triggered_rules" in hook_events[1].metadata
    assert db_provider.calls == [(1, 3)]
    assert len(rag_adapter.calls) == 1
    assert result.state.chat_response == "Cats should eat treats only in small amounts."
    assert result.response.assistant_message == "Cats should eat treats only in small amounts."
    assert result.response.route == "answer_guard"
    assert result.trace_events[-2].metadata["node_name"] == "answer_composer"


def test_llm_social_chat_routes_to_chat_without_rag_or_answer_guard() -> None:
    db_provider = RecordingDBContextProvider()
    rag_adapter = RecordingRAGAdapter()
    llm_client = GraphFakeLLM(
        intent=IntentClassificationOutput(
            intent="social_chat",
            confidence="high",
            chief_complaint=None,
            requires_db_context=False,
            requires_safety_screening=False,
            red_flag_mentioned=False,
        ),
        social_chat=SocialChatOutput(
            assistant_message="\uc7a5\uac74\ud638\ub2d8\uc774\ub77c\uace0 \uc54c\ub824\uc8fc\uc168\uc5b4\uc694."
        ),
    )

    result = run_assessment_graph(
        PetCareGraphState(user_input="\ub0b4 \uc774\ub984\uc774 \ubb50\ub77c\uace0?", pet_id=1, conversation_history=[
            {"role": "user", "content": "\ub0b4 \uc774\ub984\uc740 \uc7a5\uac74\ud638\uc57c"}
        ]),
        dependencies=AssessmentGraphDependencies(
            llm_client=llm_client,
            db_context_provider=db_provider,
            rag_adapter=rag_adapter,
        ),
    )

    assert _path(result.trace_events) == ["db_context_loader", "intent_classifier", "chat_agent"]
    assert llm_client.calls == [TurnUnderstandingOutput]
    assert db_provider.calls == [(1, 3)]
    assert rag_adapter.calls == []
    assert result.response.route == "chat"
    assert result.response.needs_user_response is False
    assert result.state.retrieval.query == ""
    assert result.response.assistant_message == "\uc7a5\uac74\ud638\ub2d8\uc774\ub77c\uace0 \uc54c\ub824\uc8fc\uc168\uc5b4\uc694."
    assert "Cornell" not in result.response.assistant_message



def test_llm_social_pet_name_question_loads_db_context_before_chat() -> None:
    db_provider = RecordingDBContextProvider()
    rag_adapter = RecordingRAGAdapter()
    llm_client = GraphFakeLLM(
        intent=IntentClassificationOutput(
            intent="social_chat",
            confidence="high",
            chief_complaint=None,
            requires_db_context=False,
            requires_safety_screening=False,
            red_flag_mentioned=False,
        ),
        social_chat=SocialChatOutput(
            assistant_message="Kongi\ub77c\uace0 \uc54c\ub824\uc8fc\uc168\uc5b4\uc694."
        ),
    )

    result = run_assessment_graph(
        PetCareGraphState(
            user_input="\ub0b4 \uac15\uc544\uc9c0 \uc774\ub984\uc774 \ubb50\uc57c?",
            pet_id=1,
        ),
        dependencies=AssessmentGraphDependencies(
            llm_client=llm_client,
            db_context_provider=db_provider,
            rag_adapter=rag_adapter,
        ),
    )

    prompt_payload = json.loads(llm_client.user_prompts[-1])

    assert _path(result.trace_events) == ["db_context_loader", "intent_classifier", "chat_agent"]
    assert llm_client.calls == [TurnUnderstandingOutput]
    assert db_provider.calls == [(1, 3)]
    assert rag_adapter.calls == []
    assert prompt_payload["pet_context"]["name"] == "Kongi"
    assert result.response.route == "chat"
    assert result.response.assistant_message == "Kongi\ub77c\uace0 \uc54c\ub824\uc8fc\uc168\uc5b4\uc694."
    assert "Cornell" not in result.response.assistant_message


def test_llm_social_pet_name_question_routes_to_chat_without_rag() -> None:
    db_provider = RecordingDBContextProvider()
    rag_adapter = RecordingRAGAdapter()
    llm_client = GraphFakeLLM(
        intent=IntentClassificationOutput(
            intent="social_chat",
            confidence="high",
            chief_complaint=None,
            requires_db_context=False,
            requires_safety_screening=False,
            red_flag_mentioned=False,
        ),
        social_chat=SocialChatOutput(
            assistant_message="Kongi\ub77c\uace0 \uc54c\ub824\uc8fc\uc168\uc5b4\uc694."
        ),
    )

    result = run_assessment_graph(
        PetCareGraphState(
            user_input="\ub0b4 \uac15\uc544\uc9c0 \uc774\ub984\uc740 \ubb50\uc57c?",
            pet_id=1,
            context=PetCareContext(pet={"id": 1, "name": "\ucd08\ucf54", "species": "dog"}),
        ),
        dependencies=AssessmentGraphDependencies(
            llm_client=llm_client,
            db_context_provider=db_provider,
            rag_adapter=rag_adapter,
        ),
    )

    assert _path(result.trace_events) == ["db_context_loader", "intent_classifier", "chat_agent"]
    assert llm_client.calls == [TurnUnderstandingOutput]
    assert db_provider.calls == [(1, 3)]
    assert rag_adapter.calls == []
    assert result.response.route == "chat"
    assert result.response.assistant_message == "Kongi\ub77c\uace0 \uc54c\ub824\uc8fc\uc168\uc5b4\uc694."
    assert "Cornell" not in result.response.assistant_message


def test_symptom_check_routes_through_context_baseline_state_change_and_safety() -> None:
    result = run_assessment_graph(
        PetCareGraphState(user_input="My cat is coughing.", pet_id=1),
        dependencies=_dependencies_for_safety_case(
            intent=_intent("symptom_check", "cough"),
            state=_state("cat", ["coughing"]),
            checklist=ChecklistExtractionOutput(checklist_id="cat_cough_triage", updates=[]),
        ),
    )

    assert _path(result.trace_events)[:6] == [
        "db_context_loader",
        "intent_classifier",
        "baseline_builder",
        "state_updater",
        "change_detector",
        "safety_guard",
    ]


def test_needs_more_info_routes_to_question_manager() -> None:
    result = run_assessment_graph(
        PetCareGraphState(user_input="My cat is coughing.", pet_id=1),
        dependencies=_dependencies_for_safety_case(
            intent=_intent("symptom_check", "cough"),
            state=_state("cat", ["coughing"]),
            checklist=ChecklistExtractionOutput(checklist_id="cat_cough_triage", updates=[]),
        ),
    )

    assert _path(result.trace_events)[-1] == "question_manager"
    assert result.response.route == "question_manager"
    assert result.response.needs_user_response is True
    assert result.state.risk_level == "unknown"



def test_short_followup_yes_answer_resolves_pending_question_to_emergency() -> None:
    template = load_checklist_template("cat_cough_triage")
    items = {
        item.item_id: item.model_copy(deep=True)
        for item in [*template.required_items, *template.optional_items]
    }
    items["open_mouth_breathing"].asked_count = 1

    result = run_assessment_graph(
        PetCareGraphState(
            user_input="\ub124",
            pet_id=1,
            species="cat",
            safety_question_turns=1,
            emergency_screening=EmergencyScreening(
                checklist_id=template.checklist_id,
                chief_complaint=template.chief_complaint,
                items=items,
                missing_questions=["open_mouth_breathing", "labored_breathing"],
            ),
        ),
        dependencies=_dependencies_for_safety_case(
            intent=_intent("followup", "cough"),
            state=_state("cat", ["coughing"]),
            checklist=ChecklistExtractionOutput(checklist_id="cat_cough_triage", updates=[]),
        ),
    )

    assert _path(result.trace_events)[-2:] == ["safety_guard", "emergency_agent"]
    assert result.response.route == "emergency"
    assert result.state.risk_level == "emergency"
    assert result.state.emergency_screening.items["open_mouth_breathing"].value is True
    assert "open_mouth_breathing" not in result.state.emergency_screening.missing_questions


def test_emergency_result_routes_to_emergency_agent() -> None:
    result = run_assessment_graph(
        PetCareGraphState(user_input="My cat is breathing with an open mouth.", pet_id=1),
        dependencies=_dependencies_for_safety_case(
            intent=_intent("symptom_check", "cough"),
            state=_state("cat", ["coughing"]),
            checklist=ChecklistExtractionOutput(
                checklist_id="cat_cough_triage",
                updates=[
                    {
                        "item_id": "open_mouth_breathing",
                        "value": True,
                        "confidence": "high",
                        "evidence": "open mouth breathing",
                    }
                ],
            ),
        ),
    )

    assert _path(result.trace_events)[-2:] == ["safety_guard", "emergency_agent"]
    assert result.state.risk_level == "emergency"
    assert result.response.route == "emergency"
    assert result.response.emergency.is_emergency is True


@pytest.mark.parametrize(
    ("risk_label", "intent_name", "chief_complaint", "species", "checklist_id", "mode", "initial_turns"),
    [
        ("urgent", "symptom_check", "vomiting", "dog", "vomiting_triage", "urgent", 0),
        ("non_emergency", "symptom_check", "cough", "cat", "cat_cough_triage", "non_emergency", 0),
        ("unknown", "symptom_check", "cough", "cat", "cat_cough_triage", "unknown", 2),
    ],
)
def test_urgent_non_emergency_and_unknown_results_continue_to_chat_rag_answer_guard(
    risk_label: str,
    intent_name: str,
    chief_complaint: str,
    species: str,
    checklist_id: str,
    mode: str,
    initial_turns: int,
) -> None:
    result = run_assessment_graph(
        PetCareGraphState(
            user_input=f"{risk_label} symptom case",
            pet_id=1,
            safety_question_turns=initial_turns,
        ),
        dependencies=_dependencies_for_safety_case(
            intent=_intent(intent_name, chief_complaint),
            state=_state(species, [chief_complaint if chief_complaint != "cough" else "coughing"]),
            checklist=_checklist_for_mode(checklist_id, mode),
        ),
    )

    assert result.state.risk_level == risk_label
    assert _path(result.trace_events)[-4:] == [
        "evidence_planner",
        "rag_agent",
        "answer_composer",
        "answer_guard",
    ]
    assert result.response.route == "answer_guard"


def test_non_emergency_hospital_visit_intent_yes_routes_to_handoff_helper() -> None:
    result = run_assessment_graph(
        PetCareGraphState(
            user_input="Please make a hospital visit summary.",
            pet_id=1,
            hospital_visit_intent="yes",
        ),
        dependencies=_dependencies_for_safety_case(
            intent=_intent("handoff_request", "cough"),
            state=_state("cat", ["coughing"]),
            checklist=_all_required_updates("cat_cough_triage", False),
        ),
    )

    assert _path(result.trace_events)[-5:] == [
        "evidence_planner",
        "rag_agent",
        "answer_composer",
        "answer_guard",
        "handoff_subgraph",
    ]
    assert result.response.route == "handoff"
    assert result.response.handoff.type == "non_emergency"
    assert result.response.handoff.summary_json is not None
    assert result.response.handoff.summary_json.baseline_comparison.window_days == 3
    assert result.response.handoff.email_draft is not None


def _dependencies_for_safety_case(
    *,
    intent: IntentClassificationOutput,
    state: StateExtractionOutput,
    checklist: ChecklistExtractionOutput,
) -> AssessmentGraphDependencies:
    return AssessmentGraphDependencies(
        llm_client=GraphFakeLLM(
            intent=intent,
            state=state,
            checklist=checklist,
        ),
        db_context_provider=RecordingDBContextProvider(),
        rag_adapter=RecordingRAGAdapter(),
    )


def _intent(intent: str, chief_complaint: str) -> IntentClassificationOutput:
    return IntentClassificationOutput(
        intent=intent,
        confidence="high",
        chief_complaint=chief_complaint,
        requires_db_context=True,
        requires_safety_screening=True,
        red_flag_mentioned=False,
    )


def _state(species: str, symptoms: list[str]) -> StateExtractionOutput:
    return StateExtractionOutput(
        species=species,
        symptoms=symptoms,
        duration=None,
        current_status=CurrentStatus(symptoms=symptoms),
        negated_findings=[],
        uncertain_findings=[],
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
                "evidence": "mocked integration test",
            }
            for item in template.required_items
        ],
    )


def _checklist_for_mode(checklist_id: str, mode: str) -> ChecklistExtractionOutput:
    if mode == "urgent":
        return ChecklistExtractionOutput(
            checklist_id=checklist_id,
            updates=[
                {
                    "item_id": "repeated_vomiting",
                    "value": True,
                    "confidence": "high",
                    "evidence": "repeated vomiting",
                }
            ],
        )
    if mode == "non_emergency":
        return _all_required_updates(checklist_id, False)
    if mode == "unknown":
        return ChecklistExtractionOutput(checklist_id=checklist_id, updates=[])
    raise AssertionError(f"Unexpected mode: {mode}")


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


def _path(events: list[NodeTraceMetadata]) -> list[str]:
    return [event.node_name for event in events]






