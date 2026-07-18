from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


def _reject_null(value, info):
    """부분 수정(Update) 스키마에서 NOT NULL 컬럼에 명시적 null 이 오면 422 로 거른다.

    (안 거르면 DB IntegrityError → 500. 필드를 생략하는 것은 그대로 허용된다.)
    """
    if value is None:
        raise ValueError("null 은 허용되지 않습니다 — 필드를 생략하거나 값을 보내세요")
    return value

# 진단서의 필드명이 `date` 라 datetime.date 타입과 이름이 겹친다. 별칭으로 회피.
DateType = date

RiskLevel = Literal["normal", "observe", "consult", "emergency"]

RISK_LABELS = {
    "normal": "정상",
    "observe": "관찰",
    "consult": "신속 상담",
    "emergency": "응급",
}


# ---------- Auth ----------
class AuthRequest(BaseModel):
    """회원가입/로그인 공용 — 이메일 + 비밀번호만 사용한다."""

    email: str
    password: str


class UserOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    email: str


class AuthResponse(BaseModel):
    token: str
    user: UserOut


# ---------- Pet ----------
class PetBase(BaseModel):
    name: str
    species: str = "강아지"
    breed: str = ""
    birth_date: date | None = None
    sex: str = "수컷"
    is_neutered: bool = False
    weight_kg: float | None = None
    size_class: str = ""
    diseases: str = ""  # 질병
    medications: str = ""  # 복용약
    supplement: str = ""  # 영양제
    allergies: str = ""  # 알레르기


class PetCreate(PetBase):
    pass


class PetUpdate(BaseModel):
    name: str | None = None
    species: str | None = None
    breed: str | None = None
    birth_date: date | None = None  # nullable 컬럼 — null 로 지우기 허용
    sex: str | None = None
    is_neutered: bool | None = None
    weight_kg: float | None = None  # nullable 컬럼 — null 로 지우기 허용
    size_class: str | None = None
    diseases: str | None = None
    medications: str | None = None
    supplement: str | None = None
    allergies: str | None = None

    _no_null = field_validator(
        "name", "species", "breed", "sex", "is_neutered", "size_class",
        "diseases", "medications", "supplement", "allergies",
        mode="before",
    )(_reject_null)


class PetOut(PetBase):
    model_config = ConfigDict(from_attributes=True)
    id: int
    updated_at: datetime  # 프로필 수정 일시
    age_label: str = ""


# ---------- Daily entry (일기장) ----------
class RecordFields(BaseModel):
    """일기장 항목 — 모두 텍스트 상태값. (DB 스펙: daily_entries)"""

    food: str = ""  # 식사 상태
    water: str = ""  # 음수 상태
    activity: str = ""  # 활동 상태
    symptom: str = ""  # 증상
    stool: str = ""  # 배변 및 설사 상태
    vomit: str = ""  # 구토 상태
    notes: str = ""  # 기타사항


class RecordCreate(RecordFields):
    record_date: date | None = None
    raw_text: str = ""


class RecordUpdate(RecordFields):
    record_date: date | None = None
    raw_text: str | None = None

    _no_null = field_validator("raw_text", mode="before")(_reject_null)


class RecordOut(RecordFields):
    model_config = ConfigDict(from_attributes=True)
    pet_id: int
    record_date: date
    raw_text: str
    created_at: datetime


class DiaryExtractRequest(BaseModel):
    text: str
    record_date: date | None = None


class ExtractedItem(BaseModel):
    category: str  # 식사 | 음수 | 활동 | 증상 | 배변 | 구토
    value: str
    field: str  # target RecordFields key hint


class DiaryExtractResponse(BaseModel):
    items: list[ExtractedItem]
    fields: RecordFields
    source: str = "mock"  # mock | agent


# ---------- Diagnosis ----------
class DiagnosisBase(BaseModel):
    date: DateType | None = None  # 진단서 발급일 또는 진료일
    hospital: str = ""  # 발급 병원
    diagnosis: str = ""  # 진단명
    content: str = ""  # 진단 내용 및 기타사항


class DiagnosisCreate(DiagnosisBase):
    original_file_ref: str = ""


class DiagnosisUpdate(BaseModel):
    date: DateType | None = None  # nullable 컬럼 — null 로 지우기 허용
    hospital: str | None = None
    diagnosis: str | None = None
    content: str | None = None

    _no_null = field_validator("hospital", "diagnosis", "content", mode="before")(
        _reject_null
    )


class DiagnosisOut(DiagnosisBase):
    model_config = ConfigDict(from_attributes=True)
    id: int
    pet_id: int
    original_file_ref: str
    created_at: datetime


class DiagnosisExtractResponse(BaseModel):
    fields: DiagnosisBase
    original_file_ref: str
    items_read: int
    source: str = "mock"


# ---------- AI check ----------
class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class TrendItem(BaseModel):
    metric: str  # 식사 | 활동 | 음수 | 체중 | 구토
    change_pct: float | None = None
    note: str = ""


class AICheckRequest(BaseModel):
    messages: list[ChatMessage]
    session_id: int | None = None  # 이어지는 대화면 기존 세션 id


class AgentAction(BaseModel):
    """LangGraph 에이전트가 대화 응답과 함께 요청하는 후속 동작.

    앱은 이 목록을 버튼으로 그린다. (예: 병원 요약 PDF 생성, 이메일 전송)
    """

    type: Literal[
        "generate_summary", "save_summary_pdf", "send_email", "save_record"
    ]
    label: str = ""
    payload: dict = Field(default_factory=dict)


class RagCitation(BaseModel):
    """RAG 근거 인용 (전문 건강정보 RAG)."""

    title: str = ""
    source: str = ""
    snippet: str = ""


class AICheckResponse(BaseModel):
    reply: str
    risk_level: RiskLevel
    risk_label: str = ""
    trend_summary: str = ""
    trends: list[TrendItem] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)
    evidence: str = ""
    followup_question: str | None = None
    can_generate_summary: bool = False
    show_hospitals: bool = False
    transit_guidance: list[str] = Field(default_factory=list)
    # AI 모델 연결용 확장 필드 (mock 은 빈 값)
    actions: list[AgentAction] = Field(default_factory=list)
    citations: list[RagCitation] = Field(default_factory=list)
    source: str = "mock"
    session_id: int | None = None


class StoredChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str
    meta: dict | None = None  # assistant 턴의 위험도/근거 등 렌더링 정보


class AISessionSummary(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    pet_id: int
    title: str
    last_risk_level: str
    message_count: int = 0
    updated_at: datetime


class AISessionDetail(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    pet_id: int
    title: str
    last_risk_level: str
    messages: list[StoredChatMessage]
    created_at: datetime
    updated_at: datetime


# ---------- Hospital ----------
class HospitalOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    name: str
    phone: str
    email: str
    distance_km: float | None
    status: str
    features: str
    is_emergency: bool
    open_24h: bool


# ---------- Summary ----------
class SummaryContent(BaseModel):
    """병원 전달용 상태 요약 — 문서 4개 섹션 구조."""

    # 1. 문서 정보
    title: str = "PetCare AI 병원 전달용 상태 요약"
    data_period: str = ""  # 사용 데이터 기간 (예: 2026.06.17 ~ 2026.07.16)
    # 2. 반려동물 정보
    pet_name: str = ""
    species: str = ""
    breed: str = ""
    sex_neuter: str = ""  # 수컷 / 중성화 완료
    age_label: str = ""  # 만 4세
    weight: str = ""  # 5.08kg
    medications: str = ""  # 현재 복용 중인 약
    allergies: str = ""  # 알레르기
    # 3. 상태
    risk_label: str = ""  # 상태 분류 (예: 응급 징후 가능성 · 신속 상담 권장)
    risk_signs: list[str] = Field(default_factory=list)  # 확인된 위험 징후
    # 4. 주호소 및 주요 변화
    chief_complaint: str = ""  # 주호소
    major_changes: str = ""  # 주요 변화
    progress: str = ""  # 경과
    owner_note: str = ""  # 보호자 메모 (선택)


class SummaryOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    pet_id: int
    risk_level: str
    content: dict
    created_at: datetime


class SummaryCreateRequest(BaseModel):
    risk_level: RiskLevel | None = None
    extra_note: str = ""  # 보호자 보완 입력


# ---------- Emergency email ----------
class EmergencyEmailCreate(BaseModel):
    hospital_id: int | None = None
    symptom_summary: str = ""  # e.g. 호흡곤란 · 청색증


class EmergencyEmailOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    pet_id: int
    hospital_id: int | None
    to_email: str
    subject: str
    body: str
    content: dict  # 병원 전달용 요약과 같은 4섹션 구조
    attachments: list
    status: str
    created_at: datetime
    sent_at: datetime | None


# ---------- Dashboard ----------
class DashboardOut(BaseModel):
    pet: PetOut
    today_record: RecordOut | None = None
    recent_food_note: str = ""
    recent_activity_note: str = ""
    record_count_30d: int = 0
    last_diagnosis: DiagnosisOut | None = None
