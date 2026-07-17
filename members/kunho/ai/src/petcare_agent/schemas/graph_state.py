"""Graph request, response, and state schemas."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from petcare_agent.schemas.common import (
    AnswerGuardStatus,
    Confidence,
    CoursePattern,
    HandoffType,
    HospitalVisitIntent,
    Intent,
    NodeRoute,
    RiskLevel,
    Species,
)
from petcare_agent.schemas.handoff import HospitalHandoffSummary, InternalTriageAssessment
from petcare_agent.schemas.triage import ChecklistItem, RuleHit


class UserLocation(BaseModel):
    lat: float = Field(ge=-90, le=90)
    lng: float = Field(ge=-180, le=180)
    permission: Literal["granted", "denied", "prompt", "unknown"] = "unknown"

    model_config = ConfigDict(extra="forbid")


class ConversationMessage(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str

    model_config = ConfigDict(extra="forbid")


class GraphRequest(BaseModel):
    request_id: str = Field(min_length=1)
    conversation_id: str = Field(min_length=1)
    pet_id: int = Field(gt=0)
    user_input: str = Field(min_length=1)
    conversation_history: list[ConversationMessage] = Field(default_factory=list)
    locale: str = "ko-KR"
    timezone: str = "Asia/Seoul"
    timestamp: datetime
    user_location: UserLocation | None = None

    model_config = ConfigDict(extra="forbid")


class FollowUpQuestion(BaseModel):
    question_id: str = Field(min_length=1)
    text: str = Field(min_length=1)

    model_config = ConfigDict(extra="forbid")


class HandoffResponse(BaseModel):
    type: HandoffType = "none"
    summary: str | None = None
    summary_json: HospitalHandoffSummary | None = None
    email_draft: str | None = None

    model_config = ConfigDict(extra="forbid")


class EmergencyResponse(BaseModel):
    is_emergency: bool = False
    triggered_rules: list[str] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid")


class GraphResponse(BaseModel):
    response_id: str = Field(min_length=1)
    conversation_id: str = Field(min_length=1)
    route: NodeRoute
    risk_level: RiskLevel
    assistant_message: str = Field(min_length=1)
    needs_user_response: bool = False
    follow_up_question: FollowUpQuestion | None = None
    handoff: HandoffResponse = Field(default_factory=HandoffResponse)
    emergency: EmergencyResponse = Field(default_factory=EmergencyResponse)

    model_config = ConfigDict(extra="forbid")


class PetCareContext(BaseModel):
    pet: dict[str, Any] = Field(default_factory=dict)
    recent_daily_entries: list[dict[str, Any]] = Field(default_factory=list)
    diagnoses: list[dict[str, Any]] = Field(default_factory=list)
    medical_background: dict[str, Any] = Field(default_factory=dict)
    unknown_items: list[str] = Field(default_factory=list)
    data_from: str = ""
    data_to: str = ""

    model_config = ConfigDict(extra="forbid")


AppetiteStatus = Literal["normal", "decreased", "increased", "unknown"]
WaterStatus = Literal["normal", "decreased", "increased", "unknown"]
ActivityStatus = Literal["normal", "decreased", "increased", "unknown"]
StoolStatus = Literal["normal", "abnormal", "unknown"]
VomitStatus = Literal["none", "present", "unknown"]


class BaselineSummary(BaseModel):
    appetite: AppetiteStatus = "unknown"
    water: WaterStatus = "unknown"
    activity: ActivityStatus = "unknown"
    stool: StoolStatus = "unknown"
    vomit: VomitStatus = "unknown"
    symptoms: list[str] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid")


class BaselineContext(BaseModel):
    window_days: int = 3
    baseline_available: bool = False
    baseline_summary: BaselineSummary = Field(default_factory=BaselineSummary)
    missing_baseline_fields: list[str] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid")


class CurrentStatus(BaseModel):
    symptoms: list[str] = Field(default_factory=list)
    appetite: AppetiteStatus = "unknown"
    water: WaterStatus = "unknown"
    activity: ActivityStatus = "unknown"

    model_config = ConfigDict(extra="forbid")


class ChangeDetection(BaseModel):
    baseline_available: bool = False
    new_symptoms: list[str] = Field(default_factory=list)
    worsened_fields: list[str] = Field(default_factory=list)
    improved_fields: list[str] = Field(default_factory=list)
    unchanged_fields: list[str] = Field(default_factory=list)
    baseline_deviation: bool = False
    summary: str = ""

    model_config = ConfigDict(extra="forbid")


class AssessmentState(BaseModel):
    symptoms: list[str] = Field(default_factory=list)
    duration: str | None = None
    course_pattern: CoursePattern = "unknown"
    severity_signals: list[str] = Field(default_factory=list)
    missing_fields: list[str] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid")


class EmergencyScreening(BaseModel):
    status: Literal["not_started", "in_progress", "complete"] = "not_started"
    checklist_id: str = ""
    chief_complaint: str = ""
    items: dict[str, ChecklistItem] = Field(default_factory=dict)
    answered_questions: dict[str, Any] = Field(default_factory=dict)
    missing_questions: list[str] = Field(default_factory=list)
    red_flags: list[str] = Field(default_factory=list)
    triggered_rules: list[RuleHit] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid")


class RetrievedChunk(BaseModel):
    chunk_id: str = Field(min_length=1)
    source_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    text: str = Field(min_length=1)
    score: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(extra="forbid")


class RAGCitation(BaseModel):
    number: int = Field(ge=1)
    title: str = Field(min_length=1)
    url: str = ""
    chunk_id: str = Field(min_length=1)
    section_path: list[str] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid")


class RetrievalState(BaseModel):
    query: str = ""
    chunks: list[RetrievedChunk] = Field(default_factory=list)
    citations: list[RAGCitation] = Field(default_factory=list)
    provider: str = ""
    insufficient_evidence: bool = False
    errors: list[str] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid")


class AnswerGuardState(BaseModel):
    status: AnswerGuardStatus = "passed"
    revisions: list[str] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid")


class HandoffState(BaseModel):
    type: HandoffType = "none"
    required: bool = False
    summary: str = ""
    summary_json: HospitalHandoffSummary | None = None
    email_draft: str = ""

    model_config = ConfigDict(extra="forbid")


class PetCareGraphState(BaseModel):
    user_input: str = ""
    conversation_history: list[ConversationMessage] = Field(default_factory=list)

    intent: Intent = "unknown"
    species: Species = "unknown"
    requires_db_context: bool = False
    requires_safety_screening: bool = False
    red_flag_mentioned: bool = False
    turn_state_extracted: bool = False
    social_response_ready: bool = False

    context: PetCareContext = Field(default_factory=PetCareContext)
    baseline_context: BaselineContext = Field(default_factory=BaselineContext)
    current_status: CurrentStatus = Field(default_factory=CurrentStatus)
    change_detection: ChangeDetection = Field(default_factory=ChangeDetection)
    assessment: AssessmentState = Field(default_factory=AssessmentState)
    emergency_screening: EmergencyScreening = Field(default_factory=EmergencyScreening)
    internal_triage_assessment: InternalTriageAssessment = Field(default_factory=InternalTriageAssessment)

    risk_level: RiskLevel = "unknown"
    confidence: Confidence = "unknown"
    safety_question_turns: int = Field(default=0, ge=0, le=2)

    retrieval: RetrievalState = Field(default_factory=RetrievalState)
    chat_response: str = ""
    answer_guard: AnswerGuardState = Field(default_factory=AnswerGuardState)
    hospital_visit_intent: HospitalVisitIntent = "not_asked"
    handoff: HandoffState = Field(default_factory=HandoffState)

    request_id: str | None = None
    conversation_id: str | None = None
    pet_id: int | None = None
    locale: str = "ko-KR"
    timezone: str = "Asia/Seoul"
    next_route: NodeRoute = "intent_classifier"

    model_config = ConfigDict(extra="forbid")


