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

#: 위험도의 심각도 순서(낮음 → 높음). 한 대화 안에서 위험도를 비교할 때 쓴다.
#: 값 자체는 Agent 가 정하지만, "무엇이 더 심각한가" 는 서버가 알아야 한다.
RISK_ORDER: dict[str, int] = {"normal": 0, "observe": 1, "consult": 2, "emergency": 3}


def higher_risk(current: str | None, incoming: str | None) -> str:
    """둘 중 더 심각한 위험도를 돌려준다(모르는 값은 무시).

    한 대화 안에서 위험도가 **내려가지 않게** 하는 데 쓴다. 이유:

    보호자가 되묻는 질문에 답할 때마다 Agent 는 그 turn 을 다시 평가한다. 그런데
    "모름" 처럼 정보가 적은 답이 오면 직전에 판단했던 근거가 약해져 위험도가 도로
    내려간다. 실제로 한 대화에서 consult → normal → emergency → normal 로 널뛰었고,
    앱 화면은 그때마다 권고 카드와 응급 카드를 오갔다.

    사용자가 관찰한 증상은 사라지지 않았는데 표시만 바뀌는 것이므로, 대화가 이어지는
    동안에는 **가장 높았던 위험도를 유지**한다. 이는 LangGraph 내부의 `merge_risk`
    (상향 전용)와 같은 규칙을 세션 수준에도 적용하는 것이다.

    '새 체크' 를 누르면 새 세션이라 다시 normal 부터 시작한다 — 이 유지는 대화
    하나 안에서만 유효하다.
    """
    left = RISK_ORDER.get(str(current or ""), -1)
    right = RISK_ORDER.get(str(incoming or ""), -1)
    if right >= left:
        return str(incoming or "normal")
    return str(current)


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
    # 앱이 알고 있는 지역명(예: "서울 강남구"). AI 가 실시간 병원 검색어를 만들 때 쓴다.
    # 서버는 이 값을 해석하지 않고 Agent 로 그대로 넘긴다. 값이 없으면 AI 는 지역을
    # **추측하지 않고** 병원 검색을 건너뛴다 — 응급 상황에 엉뚱한 지역 병원을
    # 안내하는 것이 아무것도 안내하지 않는 것보다 위험하기 때문이다.
    region_name: str | None = None


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


class HospitalSuggestion(BaseModel):
    """AI 가 실시간 검색(Tavily)으로 찾아 적합도를 매긴 병원 1건.

    아래 `HospitalOut`(DB 에 시드된 데모 병원)과는 다른 개념이다. 이쪽은 DB 에
    저장되지 않은 검색 결과라 id·distance_km 가 없고, 대신 **왜 이 병원인지**
    (score/matched_reasons)와 **무엇을 전화로 확인해야 하는지**
    (verification_required)를 함께 전달한다.

    `availability` 기본값이 "전화 확인 필요" 인 이유: 검색 결과만으로 지금 문이
    열려 있는지 확정할 수 없다. 응급 상황에서 "영업 중"이라고 단정했다가 틀리면
    보호자가 헛걸음한다.

    **모든 필드에 기본값이 있어야 한다.** 서버는 Agent 응답을 이 스키마로 검증하고
    실패하면 502 로 답변까지 통째로 버리므로, 일부 값을 못 채운 Agent 때문에
    대화 자체가 사라지면 안 된다.
    """

    name: str
    phone: str | None = None
    address: str | None = None
    # 병원 페이지에서 찾은 이메일. 응급 이메일 초안의 수신 주소로 쓴다.
    # 웹 검색으로 이메일이 나오는 경우는 드물어 대부분 None 이다 — 그때는 앱이
    # 보호자에게 주소를 직접 입력받는다(초안 자체는 만들어진다).
    email: str | None = None
    source_url: str = ""
    score: int = 0
    # recommended | possible | low_information — Literal 로 굳히지 않는다.
    # AI 쪽 분류 값이 늘어날 때마다 서버가 502 를 내면 안 되기 때문.
    suitability: str = "low_information"
    matched_reasons: list[str] = Field(default_factory=list)
    verification_required: list[str] = Field(default_factory=list)
    emergency_mentioned: bool = False
    open_24h_mentioned: bool = False
    availability: str = "전화 확인 필요"


class AICheckResponse(BaseModel):
    reply: str
    risk_level: RiskLevel
    risk_label: str = ""
    trend_summary: str = ""
    trends: list[TrendItem] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)
    evidence: str = ""
    followup_question: str | None = None
    # AI 가 아직 되묻는 중인지. True 면 판정이 끝나지 않은 것이라 앱은 결과 카드를
    # 그리지 않는다(판정 전 화면에 위험도 뱃지·요약 버튼이 붙는 사고가 있었다).
    awaiting_more_info: bool = False
    # 이번 turn 이 새 판정인가(False 면 앞선 판정에 대한 설명·잡담이다).
    # 앱은 이 값이 False 면 결과 카드를 그리지 않고 대화 말풍선만 보여준다.
    assessment_turn: bool = True
    can_generate_summary: bool = False
    show_hospitals: bool = False
    transit_guidance: list[str] = Field(default_factory=list)
    # AI 모델 연결용 확장 필드 (mock 은 빈 값)
    actions: list[AgentAction] = Field(default_factory=list)
    citations: list[RagCitation] = Field(default_factory=list)
    # AI 가 찾은 병원. 비어 있으면 앱은 기존 /api/hospitals(시드 병원)로 대체한다.
    hospitals: list[HospitalSuggestion] = Field(default_factory=list)
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
    """응급 이메일 초안 요청.

    병원은 두 경로로 올 수 있다.

    1. **AI 가 실시간 검색으로 찾은 병원** — 앱이 `hospital_name`/`hospital_email`/
       `hospital_phone` 을 그대로 보낸다. DB 에 없는 병원이라 id 가 없다.
    2. **DB 에 등록된 병원** — `hospital_id` 만 보낸다.

    셋 다 비어 있어도 된다. 병원이 정해지지 않아도 초안은 만들어지고, 수신 주소는
    앱에서 보호자가 입력한다 (`routers/emergency.py` 참고).
    """

    hospital_id: int | None = None
    hospital_name: str | None = None
    hospital_email: str | None = None
    hospital_phone: str | None = None
    symptom_summary: str = ""  # e.g. 호흡곤란 · 청색증


class EmergencyEmailOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    pet_id: int
    hospital_id: int | None
    # 병원 이메일을 못 구한 초안은 None 이다. 앱이 보호자에게 주소를 입력받는다.
    to_email: str | None
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
