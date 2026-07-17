"""Hospital handoff and internal triage assessment schemas."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from petcare_agent.schemas.common import CoursePattern, RiskLevel, Species

TriStateBool = bool | Literal["unknown"]
Sex = Literal["male", "female", "unknown"]
RequestedGoal = Literal["situation_assessment", "hospital_visit", "vet_summary"]
BaselineFieldName = Literal[
    "appetite",
    "water",
    "activity",
    "weight",
    "stool",
    "vomiting",
    "urination",
    "other",
]
BaselineChangeSummary = Literal["normal", "decreased", "increased", "abnormal", "unknown"]
DeltaUnit = Literal["percent", "kg", "count", "minutes", "text"] | None
AssociatedSymptomName = Literal[
    "vomiting",
    "diarrhea",
    "lethargy",
    "coughing",
    "breathing_issue",
    "pain",
    "limping",
    "toxin_exposure",
    "other",
]
RedFlagName = Literal[
    "open_mouth_breathing",
    "labored_breathing",
    "gum_color_abnormal",
    "collapse_or_fainting",
    "seizure",
    "severe_bleeding",
    "toxin_exposure_suspected",
]


class RedFlagInputs(BaseModel):
    open_mouth_breathing: TriStateBool = "unknown"
    labored_breathing: TriStateBool = "unknown"
    gum_color_abnormal: TriStateBool = "unknown"
    collapse_or_fainting: TriStateBool = "unknown"
    seizure: TriStateBool = "unknown"
    severe_bleeding: TriStateBool = "unknown"
    toxin_exposure_suspected: TriStateBool = "unknown"

    model_config = ConfigDict(extra="forbid")


class ClinicalInputs(BaseModel):
    onset_known: bool = False
    course_pattern: CoursePattern = "unknown"
    baseline_change_known: bool = False
    associated_symptom_count_known: bool = False

    model_config = ConfigDict(extra="forbid")


class InternalTriageAssessment(BaseModel):
    schema_version: Literal["1.0"] = "1.0"
    risk_level: RiskLevel = "unknown"
    red_flag_inputs: RedFlagInputs = Field(default_factory=RedFlagInputs)
    clinical_inputs: ClinicalInputs = Field(default_factory=ClinicalInputs)
    needs_followup: bool = False
    followup_questions: list[str] = Field(default_factory=list, max_length=2)

    model_config = ConfigDict(extra="forbid")


class PatientSummary(BaseModel):
    pet_id: str = ""
    name: str = ""
    species: Species = "unknown"
    breed: str | None = None
    sex: Sex = "unknown"
    neutered: bool | None = None
    age_years: float | None = Field(default=None, ge=0)
    age_display: str | None = None
    weight_kg: float | None = Field(default=None, ge=0)

    model_config = ConfigDict(extra="forbid")


class VisitReasonSummary(BaseModel):
    chief_complaints: list[str] = Field(default_factory=list)
    owner_summary: str = ""
    requested_goal: RequestedGoal = "situation_assessment"

    model_config = ConfigDict(extra="forbid")


class ClinicalCourseSummary(BaseModel):
    onset_text: str | None = None
    onset_at: datetime | None = None
    duration_text: str | None = None
    course_pattern: CoursePattern = "unknown"
    timeline_summary: str = ""

    model_config = ConfigDict(extra="forbid")


class BaselineChange(BaseModel):
    field: BaselineFieldName
    label: str
    current: str | None = None
    baseline: str | None = None
    change_summary: BaselineChangeSummary = "unknown"
    delta_value: float | None = None
    delta_unit: DeltaUnit = None

    model_config = ConfigDict(extra="forbid")


class BaselineComparisonSummary(BaseModel):
    window_days: Literal[3] = 3
    summary: str = ""
    changes: list[BaselineChange] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid")


class AssociatedSymptomSummary(BaseModel):
    name: AssociatedSymptomName
    label: str
    summary: str
    count: float | None = Field(default=None, ge=0)
    last_observed_at: datetime | None = None

    model_config = ConfigDict(extra="forbid")


class RedFlagSummary(BaseModel):
    name: RedFlagName
    label: str
    summary: str

    model_config = ConfigDict(extra="forbid")


class TriageAssessmentSummary(BaseModel):
    associated_symptoms: list[AssociatedSymptomSummary] = Field(default_factory=list)
    red_flags: list[RedFlagSummary] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid")


class MedicalBackgroundSummary(BaseModel):
    conditions: list[str] = Field(default_factory=list)
    medications_or_supplements: list[str] = Field(default_factory=list)
    allergies: list[str] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid")


class HospitalHandoffSummary(BaseModel):
    schema_version: Literal["1.1"] = "1.1"
    generated_at: datetime
    patient: PatientSummary
    visit_reason: VisitReasonSummary
    clinical_course: ClinicalCourseSummary
    baseline_comparison: BaselineComparisonSummary
    triage_assessment: TriageAssessmentSummary
    medical_background: MedicalBackgroundSummary

    model_config = ConfigDict(extra="forbid")


__all__ = [
    "AssociatedSymptomSummary",
    "BaselineChange",
    "BaselineComparisonSummary",
    "ClinicalCourseSummary",
    "ClinicalInputs",
    "HospitalHandoffSummary",
    "InternalTriageAssessment",
    "MedicalBackgroundSummary",
    "PatientSummary",
    "RedFlagInputs",
    "RedFlagSummary",
    "TriageAssessmentSummary",
    "VisitReasonSummary",
]
