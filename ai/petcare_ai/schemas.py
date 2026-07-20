"""전 모듈이 공유하는 입출력 스키마.

이 파일이 RAG·LangGraph·PDF·Email 사이의 유일한 계약이다.
앱 연동 시 adapter 내부만 바뀌고 이 스키마는 그대로 유지된다.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

Species = Literal["dog", "cat"]
RiskLevel = Literal["normal", "visit", "emergency"]
EmergencyUrgency = Literal["none", "contact_ready", "critical_immediate"]
SufficiencyStatus = Literal["sufficient", "insufficient", "conflicting"]
#: 라우팅 의도.
#:
#: `general_knowledge` 는 **증상 호소가 없는 수의학 지식 질문**이다
#: ("강아지에게 뭐가 몸에 좋아?", "포도 먹여도 되나요?", "슬개골 탈구가 뭔가요?").
#: 명세 6절이 RAG 의 첫 역할로 규정한 '일반적인 수의학 정보 검색' 이 이 자리다.
#:
#: `general_chat` 과 다르다 — 그쪽은 인사·앱 사용법이라 RAG 를 쓰지 않는다.
#: `health_question` 과도 다르다 — 그쪽은 **이 아이의 상태**를 판단해야 해서 위험도
#: 평가와 증상 문진이 필요하다. 지식 질문에는 판단할 대상이 없으므로 문진을 물으면
#: 대답할 수 없는 것을 묻는 셈이 된다.
Intent = Literal[
    "general_chat",
    "general_knowledge",
    "health_question",
    "hospital_search",
    "unsupported",
]

RISK_PRIORITY: dict[str, int] = {"normal": 0, "visit": 1, "emergency": 2}


def merge_risk(*levels: str | None) -> RiskLevel:
    """여러 평가 결과를 병합한다 — 더 낮은 위험도로 덮어쓰지 않는다(명세 28절)."""
    best: RiskLevel = "normal"
    for level in levels:
        if level in RISK_PRIORITY and RISK_PRIORITY[level] > RISK_PRIORITY[best]:
            best = level  # type: ignore[assignment]
    return best


# ---------------------------------------------------------------------------
# RAG
# ---------------------------------------------------------------------------
class RetrievedEvidence(BaseModel):
    """vector store 검색 결과 1건 — 답변에서 출처로 사용할 수 있어야 한다."""

    chunk_id: str
    document_id: str
    title: str
    text: str
    species: Species
    source: str
    source_url: str
    categories: list[str] = Field(default_factory=list)
    score: float | None = None
    heading_path: list[str] = Field(default_factory=list)


class RagQuery(BaseModel):
    """Query Builder 출력 — 한국어·영어 query 를 모두 만든다(명세 12절)."""

    primary_query_ko: str
    primary_query_en: str
    required_topics: list[str] = Field(default_factory=list)
    species: Species = "dog"
    emergency_hint: bool = False


class KnowledgeSufficiencyResult(BaseModel):
    status: SufficiencyStatus
    covered_topics: list[str] = Field(default_factory=list)
    missing_topics: list[str] = Field(default_factory=list)
    requires_recent_information: bool = False
    reason: str = ""


class RetrievalResult(BaseModel):
    """RAG 서비스 최종 출력(명세 6절)."""

    query: str
    species: Species
    documents: list[RetrievedEvidence] = Field(default_factory=list)
    sufficiency: SufficiencyStatus = "insufficient"
    covered_topics: list[str] = Field(default_factory=list)
    missing_topics: list[str] = Field(default_factory=list)
    web_fallback_required: bool = False


class WebEvidence(BaseModel):
    """Tavily 수의학 검색 결과 1건 (검증 전/후 공용)."""

    title: str
    url: str
    content: str
    score: float | None = None
    domain: str = ""
    accepted: bool = False
    reject_reason: str = ""


class FinalEvidence(BaseModel):
    """RAG + 검증된 웹 근거를 병합한 최종 근거(명세 16절)."""

    evidence_id: str
    source_type: Literal["rag", "web"]
    title: str
    source_url: str
    text: str
    supported_topics: list[str] = Field(default_factory=list)


class EvidenceMergeResult(BaseModel):
    evidence: list[FinalEvidence] = Field(default_factory=list)
    conflicts: list[str] = Field(default_factory=list)
    has_reliable_evidence: bool = False


# ---------------------------------------------------------------------------
# 임상 Context (PET DB / 진단서 DB / 일기장 DB)
# ---------------------------------------------------------------------------
class ContextConflict(BaseModel):
    """정보 충돌을 숨기지 않고 기록한다(명세 20절)."""

    field: str
    selected_value: Any = None
    selected_source: str = ""
    conflicting_values: list[dict[str, Any]] = Field(default_factory=list)


class ContextProvenance(BaseModel):
    field: str
    value: Any = None
    source: str = ""
    recorded_at: str | None = None


class ClinicalContext(BaseModel):
    """Clinical Context Priority Agent 출력."""

    current_observation: dict[str, Any] = Field(default_factory=dict)
    priority_pet_context: dict[str, Any] = Field(default_factory=dict)
    related_diagnoses: list[dict[str, Any]] = Field(default_factory=list)
    supporting_daily_entries: list[dict[str, Any]] = Field(default_factory=list)
    context_conflicts: list[ContextConflict] = Field(default_factory=list)
    context_provenance: list[ContextProvenance] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# 라우팅 / 평가
# ---------------------------------------------------------------------------
class SupervisorResult(BaseModel):
    intent: Intent
    possible_emergency: bool = False
    needs_clinical_context: bool = False
    reason: str = ""


class AssessmentResult(BaseModel):
    risk_level: RiskLevel = "normal"
    emergency_urgency: EmergencyUrgency = "none"
    red_flags: list[str] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)
    missing_information: list[str] = Field(default_factory=list)
    rag_required: bool = True


class MissingInformationResult(BaseModel):
    required_fields: list[str] = Field(default_factory=list)
    missing_fields: list[str] = Field(default_factory=list)
    question: str = ""
    ready: bool = False


# ---------------------------------------------------------------------------
# 병원
# ---------------------------------------------------------------------------
class HospitalRequirements(BaseModel):
    required: list[str] = Field(default_factory=list)
    preferred: list[str] = Field(default_factory=list)
    specialty_keywords: list[str] = Field(default_factory=list)
    previous_hospital_names: list[str] = Field(default_factory=list)
    search_queries: list[str] = Field(default_factory=list)


class HospitalCandidate(BaseModel):
    name: str
    address: str | None = None
    phone: str | None = None
    website: str | None = None
    email: str | None = None
    emergency_mentioned: bool = False
    open_24h_mentioned: bool = False
    specialty_mentions: list[str] = Field(default_factory=list)
    source_url: str = ""
    # 검색 결과만으로 실시간 영업 여부를 확정하지 않는다(명세 34절).
    availability: str = "전화 확인 필요"


class HospitalSuitabilityResult(BaseModel):
    hospital: HospitalCandidate
    score: int = 0
    suitability: Literal["recommended", "possible", "low_information"] = "low_information"
    matched_reasons: list[str] = Field(default_factory=list)
    unmatched_preferences: list[str] = Field(default_factory=list)
    verification_required: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# 문서 / 이메일
# ---------------------------------------------------------------------------
class ConsultationPacket(BaseModel):
    document_type: Literal["visit_consultation", "emergency_consultation"]
    generated_at: str
    pet: dict[str, Any] = Field(default_factory=dict)
    medical_history: dict[str, Any] = Field(default_factory=dict)
    current_condition: dict[str, Any] = Field(default_factory=dict)
    related_diagnoses: list[dict[str, Any]] = Field(default_factory=list)
    supporting_daily_entries: list[dict[str, Any]] = Field(default_factory=list)
    risk_assessment: dict[str, Any] = Field(default_factory=dict)
    unknown_fields: list[str] = Field(default_factory=list)
    provenance: list[dict[str, Any]] = Field(default_factory=list)


class EmailDraft(BaseModel):
    """실제 전송하지 않는다 — Android Gmail compose 용 정보만 담는다."""

    to: str | None = None
    subject: str
    body: str
    attachment_path: str
    attachment_filename: str


# ---------------------------------------------------------------------------
# 최종 출력
# ---------------------------------------------------------------------------
class ChatGraphResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    message: str
    risk_level: RiskLevel = "normal"
    emergency_urgency: EmergencyUrgency = "none"
    missing_information: list[str] = Field(default_factory=list)
    hospitals: list[HospitalSuitabilityResult] = Field(default_factory=list)
    pdf_path: str | None = None
    email_draft: EmailDraft | None = None
    ui_actions: list[dict[str, Any]] = Field(default_factory=list)
    evidence: list[FinalEvidence] = Field(default_factory=list)
    trace_metadata: dict[str, Any] = Field(default_factory=dict)


class OutputCheckResult(BaseModel):
    valid: bool = True
    errors: list[str] = Field(default_factory=list)
    action: Literal["accept", "regenerate", "fallback"] = "accept"
