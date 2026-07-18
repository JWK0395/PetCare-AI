from __future__ import annotations

from typing import Any, Literal, TypedDict

from pydantic import BaseModel, Field, model_validator


Route = Literal["general_chat", "emergency", "non_emergency"]


class BackendContextPayload(BaseModel):
    pet: dict[str, Any]
    daily_entries: list[dict[str, Any]] = Field(default_factory=list)
    diagnoses: list[dict[str, Any]] = Field(default_factory=list)
    unknown_items: list[str] = Field(default_factory=list)
    data_from: str | None = None
    data_to: str | None = None
    generated_at: str | None = None

    @model_validator(mode="before")
    @classmethod
    def normalize_daily_entries_key(cls, value: Any) -> Any:
        # 기존 API의 recent_daily_entries 키도 허용합니다.
        if isinstance(value, dict):
            normalized = dict(value)
            if (
                "daily_entries" not in normalized
                and "recent_daily_entries" in normalized
            ):
                normalized["daily_entries"] = normalized[
                    "recent_daily_entries"
                ]
            return normalized
        return value


class GraphStartRequest(BaseModel):
    session_id: str
    pet_id: int
    user_input: str
    context: BackendContextPayload

    @model_validator(mode="after")
    def validate_pet_id(self) -> "GraphStartRequest":
        context_pet_id = self.context.pet.get("id")

        if context_pet_id is None:
            raise ValueError("context.pet.id가 필요합니다.")

        if str(context_pet_id) != str(self.pet_id):
            raise ValueError(
                f"요청 pet_id={self.pet_id}와 "
                f"context.pet.id={context_pet_id}가 다릅니다."
            )

        mismatched_diagnoses = [
            item.get("id")
            for item in self.context.diagnoses
            if (
                item.get("pet_id") is not None
                and str(item.get("pet_id")) != str(self.pet_id)
            )
        ]

        if mismatched_diagnoses:
            raise ValueError(
                "다른 반려동물의 진단서가 포함되어 있습니다: "
                f"{mismatched_diagnoses}"
            )

        return self


class SymptomItem(BaseModel):
    code: str = Field(
        description=(
            "정규화된 증상 코드. 가능한 경우 "
            "respiratory_distress, cyanosis, unconsciousness, "
            "seizure, severe_bleeding, urinary_obstruction, "
            "toxin_ingestion, vomiting, diarrhea, fever, lethargy, "
            "appetite_loss, pain, urinary_abnormality, "
            "respiratory_issue, severe_deterioration 등을 사용"
        )
    )
    evidence: str
    negated: bool = False


class AssessmentOutput(BaseModel):
    intent: Literal["general_chat", "health_related"]
    handoff_requested: bool = False
    user_goal: str
    symptoms: list[SymptomItem] = Field(default_factory=list)


class RAGChunk(BaseModel):
    source_id: str
    title: str
    organization: str
    version: str
    page: int | None = None
    text: str
    score: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class HandoffOutput(BaseModel):
    title: str
    pet_summary: str
    chief_complaint: list[str]
    onset_and_course: list[str]
    recent_daily_record_summary: list[str]
    diagnosis_and_medication_history: list[str]
    unknown_items: list[str]
    caution: str


class PetCareState(TypedDict, total=False):
    # 백엔드 입력
    session_id: str
    pet_id: int
    user_input: str
    backend_context: dict[str, Any]

    # Context 정리
    diary_summary: str
    diagnosis_summary: str

    # Assessment Graph
    assessment: dict[str, Any]
    handoff_requested: bool

    # 최종 상태는 세 가지뿐
    route: Route | None

    # 같은 session_id 안의 대화 기억
    conversation_history: list[dict[str, str]]

    # Triage Episode 상태
    triage_status: Literal["idle", "collecting", "completed"]
    previous_triage: dict[str, Any]
    post_triage_mode: bool

    # 증상별 질문 Cycle 상태
    question_strategy: dict[str, Any]

    # 추가 질문 이력
    follow_up_history: list[dict[str, Any]]
    needs_user_response: bool

    # Safety Guard
    emergency_hits: list[dict[str, Any]]
    recovery_hits: list[dict[str, Any]]

    # Chat Agent ↔ RAG Agent
    rag_query: str
    rag_chunks: list[dict[str, Any]]
    rag_done: bool

    # 출력
    answer: str
    handoff: dict[str, Any]

    # 실행 관측
    latency_ms: dict[str, float]
    errors: list[str]


class GraphStepResult(TypedDict):
    status: Literal["completed", "waiting_for_user"]
    session_id: str
    state: PetCareState
    trace: list[str]
    question: str | None
    field: str | None
