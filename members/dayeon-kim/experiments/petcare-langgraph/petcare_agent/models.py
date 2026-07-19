from __future__ import annotations

from typing import Any, Literal, TypedDict

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)


Route = Literal[
    "general_chat",
    "emergency",
    "non_emergency",
]

AnswerStatus = Literal[
    "reported",
    "no",
    "unknown",
]


class BackendContextPayload(BaseModel):
    pet: dict[str, Any]
    daily_entries: list[dict[str, Any]] = Field(
        default_factory=list
    )
    diagnoses: list[dict[str, Any]] = Field(
        default_factory=list
    )
    unknown_items: list[str] = Field(
        default_factory=list
    )
    data_from: str | None = None
    data_to: str | None = None
    generated_at: str | None = None

    model_config = ConfigDict(
        extra="allow",
    )

    @model_validator(mode="before")
    @classmethod
    def normalize_daily_entries_key(
        cls,
        value: Any,
    ) -> Any:
        if not isinstance(value, dict):
            return value

        normalized = dict(value)

        if (
            "daily_entries" not in normalized
            and "recent_daily_entries" in normalized
        ):
            normalized["daily_entries"] = (
                normalized["recent_daily_entries"]
            )

        return normalized


class UserLocation(BaseModel):
    latitude: float | None = Field(
        default=None,
        ge=-90,
        le=90,
    )
    longitude: float | None = Field(
        default=None,
        ge=-180,
        le=180,
    )
    address: str | None = None

    model_config = ConfigDict(
        extra="forbid",
    )


class GraphStartRequest(BaseModel):
    session_id: str = Field(min_length=1)
    pet_id: int = Field(gt=0)
    user_input: str = Field(min_length=1)
    context: BackendContextPayload
    location: UserLocation | None = None

    model_config = ConfigDict(
        extra="forbid",
    )

    @model_validator(mode="after")
    def validate_pet_id(
        self,
    ) -> "GraphStartRequest":
        context_pet_id = self.context.pet.get("id")

        if context_pet_id is None:
            raise ValueError(
                "context.pet.id가 필요합니다."
            )

        if str(context_pet_id) != str(self.pet_id):
            raise ValueError(
                f"요청 pet_id={self.pet_id}와 "
                f"context.pet.id={context_pet_id}가 다릅니다."
            )

        mismatched = [
            item.get("id")
            for item in self.context.diagnoses
            if (
                item.get("pet_id") is not None
                and str(item.get("pet_id"))
                != str(self.pet_id)
            )
        ]

        if mismatched:
            raise ValueError(
                "다른 반려동물의 진단서가 "
                f"포함되어 있습니다: {mismatched}"
            )

        return self


class ConversationMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1)

    model_config = ConfigDict(
        extra="forbid",
    )


class SymptomItem(BaseModel):
    code: str = Field(min_length=1)
    evidence: str = Field(min_length=1)
    negated: bool = False

    model_config = ConfigDict(
        extra="forbid",
    )


class AssessmentOutput(BaseModel):
    intent: Literal[
        "general_chat",
        "health_related",
    ]
    handoff_requested: bool = False
    user_goal: str = Field(min_length=1)
    symptoms: list[SymptomItem] = Field(
        default_factory=list
    )

    model_config = ConfigDict(
        extra="forbid",
    )


class QuestionCycleRecord(BaseModel):
    symptom: str = Field(min_length=1)
    questions: list[str] = Field(
        default_factory=list
    )
    answer: str = ""
    answer_status: AnswerStatus = "unknown"

    model_config = ConfigDict(
        extra="forbid",
    )


class AdditionalCheckRecord(BaseModel):
    question: str = Field(min_length=1)
    answer: str = ""
    answer_status: AnswerStatus = "unknown"

    model_config = ConfigDict(
        extra="forbid",
    )


class QuestionStrategy(BaseModel):
    detected_symptoms: list[str] = Field(
        default_factory=list
    )
    completed_cycles: list[str] = Field(
        default_factory=list
    )
    active_symptom: str | None = None
    cycle_history: list[QuestionCycleRecord] = Field(
        default_factory=list
    )
    additional_checks: list[AdditionalCheckRecord] = Field(
        default_factory=list
    )
    additional_answer_status: AnswerStatus | None = None
    unknown_additional_retry_count: int = Field(
        default=0,
        ge=0,
    )
    unknown_additional_unresolved: bool = False
    awaiting_additional_check: bool = False
    finished: bool = False

    model_config = ConfigDict(
        extra="forbid",
    )


class FollowUpItem(BaseModel):
    field: str = Field(min_length=1)
    kind: Literal[
        "symptom_detail",
        "additional_symptoms",
    ]
    symptom: str | None = None
    questions: list[str] = Field(
        default_factory=list
    )
    question: str = Field(min_length=1)
    answer: str = ""
    answer_status: AnswerStatus = "unknown"

    model_config = ConfigDict(
        extra="forbid",
    )


class RAGChunk(BaseModel):
    source_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    organization: str = Field(min_length=1)
    version: str = Field(min_length=1)
    page: int | None = None
    text: str = Field(min_length=1)
    score: float | None = None
    metadata: dict[str, Any] = Field(
        default_factory=dict
    )

    model_config = ConfigDict(
        extra="forbid",
    )


class HandoffOutput(BaseModel):
    chief_complaints: list[str] = Field(
        default_factory=list
    )
    major_changes: list[str] = Field(
        default_factory=list
    )
    course: list[str] = Field(
        default_factory=list
    )

    model_config = ConfigDict(
        extra="forbid",
    )


class HandoffDocumentInfo(BaseModel):
    title: str = Field(min_length=1)
    generated_at: str = Field(min_length=1)
    data_period: str = Field(min_length=1)

    model_config = ConfigDict(extra="forbid")


class HandoffPetInfo(BaseModel):
    name: str = Field(min_length=1)
    species: str = Field(min_length=1)
    breed: str = Field(min_length=1)
    sex_neutered: str = Field(min_length=1)
    age: str = Field(min_length=1)
    weight: str = Field(min_length=1)
    medications: list[str] = Field(default_factory=list)
    allergies: list[str] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid")


class HandoffStatus(BaseModel):
    classification: str = Field(min_length=1)
    risk_signs: list[str] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid")


class HandoffClinicalSummary(BaseModel):
    chief_complaints: list[str] = Field(default_factory=list)
    major_changes: list[str] = Field(default_factory=list)
    course: list[str] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid")


class HandoffDocument(BaseModel):
    document_info: HandoffDocumentInfo
    pet_info: HandoffPetInfo
    status: HandoffStatus
    clinical_summary: HandoffClinicalSummary

    model_config = ConfigDict(extra="forbid")


class PromptContext(BaseModel):
    pet: dict[str, Any] = Field(default_factory=dict)
    daily_entries: list[dict[str, Any]] = Field(
        default_factory=list
    )
    diagnoses: list[dict[str, Any]] = Field(
        default_factory=list
    )
    data_period: str = "미확인"
    selection_note: str = ""

    model_config = ConfigDict(extra="forbid")


class HospitalInfo(BaseModel):
    hospital_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    address: str = Field(min_length=1)
    phone: str | None = None
    email: str | None = None
    distance_km: float | None = Field(
        default=None,
        ge=0,
    )
    is_open: bool = True
    open_status: str = "운영 중"
    source: str = "provider"

    model_config = ConfigDict(extra="forbid")


class EmailDeliveryResult(BaseModel):
    status: Literal[
        "sent",
        "saved",
        "failed",
    ]
    recipient: str | None = None
    message_id: str | None = None
    file_path: str | None = None
    preview_path: str | None = None
    error: str | None = None

    model_config = ConfigDict(extra="forbid")


class PetCareState(
    TypedDict,
    total=False,
):
    session_id: str
    pet_id: int
    user_input: str
    backend_context: dict[str, Any]
    location: dict[str, Any] | None

    diary_summary: str
    diagnosis_summary: str

    assessment: dict[str, Any]
    handoff_requested: bool
    route: Route | None

    conversation_history: list[dict[str, str]]

    triage_status: Literal[
        "idle",
        "collecting",
        "completed",
    ]
    previous_triage: dict[str, Any]
    post_triage_mode: bool

    question_strategy: dict[str, Any]
    follow_up_history: list[dict[str, Any]]
    needs_user_response: bool

    emergency_hits: list[dict[str, Any]]
    recovery_hits: list[dict[str, Any]]

    rag_query: str
    rag_chunks: list[dict[str, Any]]
    rag_done: bool
    rag_status: Literal[
        "not_started",
        "unavailable",
        "completed",
        "failed",
    ]

    visit_decision: Literal[
        "pending",
        "yes",
        "no",
    ]

    nearby_hospitals: list[dict[str, Any]]
    selected_hospital: dict[str, Any]

    answer: str
    handoff: dict[str, Any]
    artifact_path: str | None

    email_subject: str
    email_body: str
    email_delivery: dict[str, Any]

    prompt_context_stats: dict[str, Any]
    latency_ms: dict[str, float]
    warnings: list[str]
    errors: list[str]


class GraphStepResult(TypedDict):
    status: Literal[
        "completed",
        "waiting_for_user",
    ]
    session_id: str
    state: PetCareState
    trace: list[str]
    question: str | None
    field: str | None
