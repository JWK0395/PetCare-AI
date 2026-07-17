"""Pydantic schemas for LLM structured output calls."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from petcare_agent.schemas.common import (
    AnswerGuardStatus,
    Confidence,
    CoursePattern,
    HandoffType,
    Intent,
    Species,
)
from petcare_agent.schemas.graph_state import CurrentStatus
from petcare_agent.schemas.triage import ChecklistValue


class IntentClassificationOutput(BaseModel):
    intent: Intent
    confidence: Confidence
    chief_complaint: str | None = None
    requires_db_context: bool
    requires_safety_screening: bool
    red_flag_mentioned: bool = False

    model_config = ConfigDict(extra="forbid")


class SocialChatOutput(BaseModel):
    assistant_message: str = Field(min_length=1)

    model_config = ConfigDict(extra="forbid")


class GeneralPetCareAnswerOutput(BaseModel):
    assistant_message: str = Field(min_length=1)

    model_config = ConfigDict(extra="forbid")


class StateExtractionOutput(BaseModel):
    species: Species
    symptoms: list[str] = Field(default_factory=list)
    duration: str | None = None
    course_pattern: CoursePattern = "unknown"
    current_status: CurrentStatus = Field(default_factory=CurrentStatus)
    negated_findings: list[str] = Field(default_factory=list)
    uncertain_findings: list[str] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid")


class TurnUnderstandingOutput(BaseModel):
    intent: Intent
    confidence: Confidence
    chief_complaint: str | None = None
    requires_db_context: bool
    requires_safety_screening: bool
    red_flag_mentioned: bool = False
    state: StateExtractionOutput
    social_chat: SocialChatOutput | None = None

    model_config = ConfigDict(extra="forbid")


class ChecklistItemUpdate(BaseModel):
    item_id: str = Field(min_length=1)
    value: ChecklistValue = None
    confidence: Confidence
    evidence: str | None = None

    model_config = ConfigDict(extra="forbid")


class ChecklistExtractionOutput(BaseModel):
    checklist_id: str = Field(min_length=1)
    updates: list[ChecklistItemUpdate] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid")


class AnswerGuardReviewOutput(BaseModel):
    status: AnswerGuardStatus
    unsafe_phrases: list[str] = Field(default_factory=list)
    revised_answer: str | None = None

    model_config = ConfigDict(extra="forbid")


class HandoffSummaryOutput(BaseModel):
    type: HandoffType
    summary: str = Field(min_length=1)
    email_draft: str | None = None

    model_config = ConfigDict(extra="forbid")

