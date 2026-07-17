"""Triage checklist and rule result schemas."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from petcare_agent.schemas.common import Confidence, RiskAction, RiskLevel

ChecklistSpecies = Literal["cat", "dog", "cat/dog", "unknown"]
ChecklistItemType = Literal["boolean", "number", "string"]
ChecklistValue = bool | int | float | str | None


class ChecklistItem(BaseModel):
    """One structured item in an emergency screening checklist."""

    item_id: str = Field(min_length=1)
    label: str = Field(min_length=1)
    type: ChecklistItemType
    value: ChecklistValue = None
    confidence: Confidence = "unknown"
    source: str = "user_input"
    asked_count: int = Field(default=0, ge=0)
    unit: str | None = None
    question_text: str | None = None
    priority: int | None = Field(default=None, ge=1, le=5)
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(extra="forbid")


class ChecklistTemplate(BaseModel):
    """Checklist template selected by species and chief complaint."""

    checklist_id: str = Field(min_length=1)
    species: ChecklistSpecies
    chief_complaint: str = Field(min_length=1)
    required_items: list[ChecklistItem] = Field(min_length=1)
    optional_items: list[ChecklistItem] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid")


class RuleHit(BaseModel):
    """A rule that fired during checklist validation."""

    rule_id: str = Field(min_length=1)
    result: RiskLevel | RiskAction
    condition: str = Field(min_length=1)
    details: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(extra="forbid")


class RiskResult(BaseModel):
    """Final rule-based risk output for Safety Guard."""

    risk_level: RiskLevel
    confidence: Confidence
    action: RiskAction = "final"
    triggered_rules: list[RuleHit] = Field(default_factory=list)
    missing_items: list[str] = Field(default_factory=list)
    requires_more_info: bool = False

    model_config = ConfigDict(extra="forbid")
