from __future__ import annotations

import pytest

from petcare_agent.graphs.subgraphs.handoff import (
    build_non_emergency_handoff,
    should_build_non_emergency_handoff,
)
from petcare_agent.safety.red_flags import FORBIDDEN_HANDOFF_FIELDS
from petcare_agent.schemas.graph_state import (
    AssessmentState,
    BaselineContext,
    BaselineSummary,
    ChangeDetection,
    CurrentStatus,
    PetCareContext,
    PetCareGraphState,
)


@pytest.mark.parametrize("risk_level", ["urgent", "non_emergency", "unknown"])
def test_handoff_required_only_when_visit_intent_is_yes(risk_level: str) -> None:
    state = PetCareGraphState(
        risk_level=risk_level,  # type: ignore[arg-type]
        hospital_visit_intent="yes",
        user_input="Please prepare a summary for the clinic.",
        context=PetCareContext(
            pet={"id": 1, "name": "Coco", "species": "dog"},
            medical_background={
                "conditions": ["stage 2 dental disease"],
                "medications_or_supplements": ["joint supplement"],
                "allergies": [],
            },
        ),
        baseline_context=BaselineContext(
            window_days=3,
            baseline_available=True,
            baseline_summary=BaselineSummary(
                appetite="normal",
                water="normal",
                activity="normal",
                stool="normal",
                vomit="none",
            ),
        ),
        current_status=CurrentStatus(symptoms=["coughing"], appetite="decreased", activity="decreased"),
        assessment=AssessmentState(
            symptoms=["coughing"],
            duration="2 hours",
            course_pattern="persistent",
            severity_signals=["mild lethargy"],
        ),
        change_detection=ChangeDetection(
            baseline_available=True,
            worsened_fields=["appetite", "activity"],
            summary="Worsened compared with baseline: appetite, activity.",
        ),
    )

    result = build_non_emergency_handoff(state)

    assert should_build_non_emergency_handoff(state) is True
    assert result.handoff.required is True
    assert result.handoff.type == "non_emergency"
    assert result.handoff.summary_json is not None
    assert result.handoff.summary_json.schema_version == "1.1"
    assert result.handoff.summary_json.patient.name == "Coco"
    assert result.handoff.summary_json.visit_reason.chief_complaints == ["coughing"]
    assert result.handoff.summary_json.clinical_course.course_pattern == "persistent"
    assert result.handoff.summary_json.baseline_comparison.window_days == 3
    assert {change.field for change in result.handoff.summary_json.baseline_comparison.changes} == {
        "appetite",
        "activity",
    }
    assert result.handoff.summary_json.medical_background.conditions == ["stage 2 dental disease"]
    assert not _contains_forbidden_handoff_field(result.handoff.summary_json.model_dump(mode="json"))
    assert "Risk level" not in result.handoff.summary
    assert result.next_route == "handoff"


@pytest.mark.parametrize("intent", ["no", "undecided", "not_asked"])
def test_handoff_not_required_for_non_yes_visit_intents(intent: str) -> None:
    state = PetCareGraphState(
        risk_level="urgent",
        hospital_visit_intent=intent,  # type: ignore[arg-type]
    )

    result = build_non_emergency_handoff(state)

    assert result.handoff.required is False
    assert result.handoff.type == "none"
    assert result.handoff.summary == ""
    assert result.handoff.summary_json is None
    assert result.handoff.email_draft == ""


def test_emergency_is_not_processed_by_non_emergency_handoff_subgraph() -> None:
    state = PetCareGraphState(risk_level="emergency", hospital_visit_intent="yes")

    result = build_non_emergency_handoff(state)

    assert should_build_non_emergency_handoff(state) is False
    assert result.handoff.required is False
    assert result.handoff.type == "none"
    assert result.handoff.summary_json is None
    assert result.next_route == state.next_route


def test_email_draft_is_draft_only_and_not_sent() -> None:
    state = PetCareGraphState(
        risk_level="non_emergency",
        hospital_visit_intent="yes",
        context=PetCareContext(pet={"name": "Momo"}),
    )

    result = build_non_emergency_handoff(state)

    assert result.handoff.email_draft.startswith("초안입니다")
    assert "이메일은 전송되지 않았습니다" in result.handoff.email_draft
    assert "제목: Momo 병원 방문 요약" in result.handoff.email_draft


def test_empty_medical_background_stays_empty_without_extra_questions() -> None:
    state = PetCareGraphState(
        risk_level="non_emergency",
        hospital_visit_intent="yes",
        context=PetCareContext(
            pet={"name": "Momo"},
            medical_background={
                "conditions": [],
                "medications_or_supplements": [],
                "allergies": [],
            },
        ),
    )

    result = build_non_emergency_handoff(state)

    assert result.handoff.summary_json is not None
    assert result.handoff.summary_json.medical_background.conditions == []
    assert result.handoff.summary_json.medical_background.medications_or_supplements == []
    assert result.handoff.summary_json.medical_background.allergies == []
    assert result.internal_triage_assessment.followup_questions == []


def _contains_forbidden_handoff_field(value: object) -> bool:
    if isinstance(value, dict):
        return any(
            key in FORBIDDEN_HANDOFF_FIELDS or _contains_forbidden_handoff_field(child)
            for key, child in value.items()
        )
    if isinstance(value, list):
        return any(_contains_forbidden_handoff_field(child) for child in value)
    return False
