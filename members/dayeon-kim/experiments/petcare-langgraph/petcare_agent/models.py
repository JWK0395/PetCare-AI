from __future__ import annotations

from typing import Any, Literal, TypedDict

from pydantic import BaseModel, Field, model_validator


Route = Literal[
    "general_chat",
    "emergency",
    "non_emergency",
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
            and "recent_daily_entries"
            in normalized
        ):
            normalized["daily_entries"] = (
                normalized[
                    "recent_daily_entries"
                ]
            )

        return normalized


class UserLocation(BaseModel):
    latitude: float | None = None
    longitude: float | None = None
    address: str | None = None


class GraphStartRequest(BaseModel):
    session_id: str
    pet_id: int
    user_input: str
    context: BackendContextPayload
    location: UserLocation | None = None

    @model_validator(mode="after")
    def validate_pet_id(
        self,
    ) -> "GraphStartRequest":
        context_pet_id = (
            self.context.pet.get("id")
        )

        if context_pet_id is None:
            raise ValueError(
                "context.pet.id가 필요합니다."
            )

        if str(context_pet_id) != str(
            self.pet_id
        ):
            raise ValueError(
                f"요청 pet_id={self.pet_id}와 "
                f"context.pet.id="
                f"{context_pet_id}가 다릅니다."
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


class SymptomItem(BaseModel):
    code: str
    evidence: str
    negated: bool = False


class AssessmentOutput(BaseModel):
    intent: Literal[
        "general_chat",
        "health_related",
    ]
    handoff_requested: bool = False
    user_goal: str
    symptoms: list[SymptomItem] = Field(
        default_factory=list
    )


class RAGChunk(BaseModel):
    source_id: str
    title: str
    organization: str
    version: str
    page: int | None = None
    text: str
    score: float | None = None
    metadata: dict[str, Any] = Field(
        default_factory=dict
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


class HospitalInfo(BaseModel):
    hospital_id: str
    name: str
    address: str
    phone: str | None = None
    email: str | None = None
    distance_km: float | None = None
    is_open: bool = True
    open_status: str = "운영 중"
    source: str = "provider"


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

    conversation_history: list[
        dict[str, str]
    ]

    triage_status: Literal[
        "idle",
        "collecting",
        "completed",
    ]
    previous_triage: dict[str, Any]
    post_triage_mode: bool

    question_strategy: dict[str, Any]
    follow_up_history: list[
        dict[str, Any]
    ]
    needs_user_response: bool

    emergency_hits: list[
        dict[str, Any]
    ]
    recovery_hits: list[
        dict[str, Any]
    ]

    rag_query: str
    rag_chunks: list[
        dict[str, Any]
    ]
    rag_done: bool

    visit_decision: Literal[
        "pending",
        "yes",
        "no",
    ]

    nearby_hospitals: list[
        dict[str, Any]
    ]
    selected_hospital: dict[
        str,
        Any,
    ]

    answer: str
    handoff: dict[str, Any]
    artifact_path: str | None

    email_subject: str
    email_body: str
    email_delivery: dict[str, Any]

    latency_ms: dict[str, float]
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
