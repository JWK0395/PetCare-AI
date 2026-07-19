"""AI Agent 연결 레이어.

Agent(Safety/Context/Trend/RAG/Summary)는 별도로 구현된다(ai/ 폴더 참고).
이 모듈은 서버와 Agent 사이의 "연결"만 담당한다.

- AGENT_MODE=http : 외부 Agent 서비스로 HTTP 요청을 전달한다. (계약: ai/README.md)
- AGENT_MODE=mock : Agent 가 없을 때 쓰는 **미연결 응답**. 판단·추출을 하지 않고
  "AI 가 연결되지 않았다" 는 사실만 돌려준다 (`MockAgentClient` docstring 참고).

서버는 항상 동일한 payload(pet / messages / context)를 만들어 전달하므로,
실제 Agent는 이 계약만 맞추면 그대로 교체된다.

DB 스펙(daily_entries)에 따라 기록은 모두 텍스트 상태값이다. 수치 기반 기준선/추이는
서버가 계산하지 않는다(AI 가 텍스트로 판단한다).
"""

from __future__ import annotations

import logging
import re
from datetime import date, timedelta
from typing import Protocol

import httpx
from fastapi import HTTPException

from ..config import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 계약 (HTTP 모드에서 Agent 서비스가 구현해야 하는 엔드포인트)
# ---------------------------------------------------------------------------
DIARY_EXTRACT_PATH = "/agent/diary-extract"
DIAGNOSIS_EXTRACT_PATH = "/agent/diagnosis-extract"
HEALTH_CHECK_PATH = "/agent/health-check"
SUMMARY_PATH = "/agent/summary"


class AgentClient(Protocol):
    def diary_extract(self, pet: dict, text: str, record_date: str, context: dict) -> dict: ...

    def diagnosis_extract(self, pet: dict, file_name: str, file_text: str) -> dict: ...

    def health_check(
        self,
        pet: dict,
        messages: list[dict],
        context: dict,
        region_name: str | None = None,
        conversation_id: str | None = None,
    ) -> dict: ...

    def generate_summary(
        self, pet: dict, risk_level: str, extra_note: str, context: dict
    ) -> dict: ...


# ---------------------------------------------------------------------------
# HTTP 모드 — 별도 구현된 Agent 서비스로 전달
# ---------------------------------------------------------------------------
class HttpAgentClient:
    def __init__(self) -> None:
        headers = {}
        if settings.agent_api_key:
            headers["Authorization"] = f"Bearer {settings.agent_api_key}"
        self._client = httpx.Client(
            base_url=settings.agent_base_url,
            timeout=settings.agent_timeout_seconds,
            headers=headers,
        )

    def _post(self, path: str, payload: dict) -> dict:
        try:
            res = self._client.post(path, json=payload)
            res.raise_for_status()
            data = res.json()
        except httpx.HTTPError as exc:
            logger.error("Agent request failed: %s %s", path, exc)
            raise HTTPException(
                status_code=502,
                detail=f"Agent 서비스에 연결할 수 없습니다 ({settings.agent_base_url}{path}). "
                "ai 서비스가 실행 중인지 확인하세요 (AGENT_MODE=mock 으로 바꾸면 "
                "분석 없이 '미연결' 안내만 표시됩니다).",
            )
        except ValueError as exc:  # res.json() — 200 인데 본문이 JSON 이 아닌 경우
            logger.error("Agent returned non-JSON body: %s %s", path, exc)
            raise HTTPException(
                status_code=502,
                detail=f"Agent 응답이 JSON 이 아닙니다 ({path}). Agent 서비스 구현을 확인하세요.",
            )
        if not isinstance(data, dict):  # JSON 배열/문자열 등 — 계약 위반
            logger.error("Agent returned non-object JSON: %s %s", path, type(data).__name__)
            raise HTTPException(
                status_code=502,
                detail=f"Agent 응답이 JSON 객체가 아닙니다 ({path}). Agent 서비스 구현을 확인하세요.",
            )
        data.setdefault("source", "agent")
        return data

    def diary_extract(self, pet: dict, text: str, record_date: str, context: dict) -> dict:
        return self._post(
            DIARY_EXTRACT_PATH,
            {"pet": pet, "text": text, "record_date": record_date, "context": context},
        )

    def diagnosis_extract(self, pet: dict, file_name: str, file_text: str) -> dict:
        return self._post(
            DIAGNOSIS_EXTRACT_PATH,
            {"pet": pet, "file_name": file_name, "file_text": file_text},
        )

    def health_check(
        self,
        pet: dict,
        messages: list[dict],
        context: dict,
        region_name: str | None = None,
        conversation_id: str | None = None,
    ) -> dict:
        """지역명과 대화 식별자를 함께 보낸다.

        `region_name` — Agent 가 병원 검색어를 만들려면 지역이 필요하다. None 이어도
        키는 넣는다. 키를 빼면 Agent 쪽에서 "안 보낸 것"과 "모르는 것"을 구분할 수
        없고, 지역을 임의로 추측하게 될 여지가 생긴다.

        `conversation_id` — Agent 의 LangGraph 가 되묻기(interrupt)를 재개하려면
        같은 대화를 가리키는 키가 필요하다(명세 29절). 서버의 `ai_sessions.id` 를
        그대로 쓴다. 이 값이 없으면 Agent 는 매 요청을 새 대화로 처리하고, 되묻기
        라운드 제한이 영원히 발동하지 않아 같은 질문을 반복한다.
        """
        return self._post(
            HEALTH_CHECK_PATH,
            {
                "pet": pet,
                "messages": messages,
                "context": context,
                "region_name": region_name,
                "conversation_id": conversation_id,
            },
        )

    def generate_summary(
        self, pet: dict, risk_level: str, extra_note: str, context: dict
    ) -> dict:
        return self._post(
            SUMMARY_PATH,
            {
                "pet": pet,
                "risk_level": risk_level,
                "extra_note": extra_note,
                "context": context,
            },
        )


#: mock 모드에서 사용자에게 보여줄 유일한 문구. 상태를 판정한 척하지 않고
#: 무엇을 해야 연결되는지까지 알려 준다.
NOT_CONNECTED_REPLY = (
    "AI 가 연결되지 않아 상태 분석을 할 수 없어요. "
    "server/.env 의 AGENT_MODE 를 http 로 설정하고 ai 서비스를 실행해 주세요."
)


# ---------------------------------------------------------------------------
# 병원 전달용 요약 4섹션 — 서버가 DB 기록으로 직접 만든다 (mock/http 공용)
# ---------------------------------------------------------------------------
#: daily_entries 텍스트 항목 7개 (schemas.RecordFields 와 같은 키 집합).
RECORD_FIELD_KEYS: tuple[str, ...] = (
    "food", "water", "activity", "symptom", "stool", "vomit", "notes",
)

# 저장된 기록 텍스트에서 상태 변화를 읽는 패턴. **판단이 아니라 요약**이다 —
# 보호자가 직접 저장한 문장을 그대로 훑어 무엇이 적혀 있었는지만 모은다.
FOOD_LOW_RE = re.compile(r"감소|남김|적게|거의")
ACTIVITY_LOW_RE = re.compile(r"짧|거부|감소")
LETHARGY_RE = re.compile(r"축 처|기운이 없|기운 없|무기력|축 늘어|처져|기력")
DIARRHEA_STOOL_RE = re.compile(r"설사|묽은")
#: 구토 칸이 "비어 있음/없음" 인 값 — 이 값들은 구토 관찰로 세지 않는다.
NO_VOMIT_VALUES = frozenset({"", "없음"})


def build_summary_content(
    pet: dict, risk_level: str, extra_note: str, context: dict
) -> dict:
    """병원 전달용 상태 요약(문서 4섹션 구조)을 만든다.

    요약(summary) 생성과 응급 이메일이 공용으로 사용한다. 기록이 모두 텍스트이므로
    최근 기록의 상태값을 스캔해 위험 징후·주호소·변화를 구성한다.

    ## 빈 값을 문구로 채우지 않는 이유

    예전에는 값이 비면 "보호자 관찰 변화" · "특이 변화 관찰되지 않음" · "기록 부족"
    같은 기본 문구를 넣었다. 이 문서는 **병원으로 전달된다.** 수의사가 읽을 때
    "특이 변화 관찰되지 않음" 은 '보호자가 관찰했고 변화가 없었다' 는 사실 주장이
    되는데, 실제로는 그 기간에 기록 자체가 없었을 뿐이다. 그래서 없는 값은 빈
    문자열로 남기고, 채워진 칸은 전부 DB 에 실제로 저장된 기록에서만 나오게 한다.
    """
    records: list[dict] = context.get("records", [])
    window_days = context.get("window_days", 30)
    recent = records[-3:]

    # 1. 문서 정보 — 사용 데이터 기간
    end = date.today()
    start = end - timedelta(days=window_days - 1)
    data_period = f"{start:%Y.%m.%d} ~ {end:%Y.%m.%d}"

    # 2. 반려동물 정보
    sex_neuter = " / ".join(
        x for x in [
            pet.get("sex", ""),
            "중성화 완료" if pet.get("is_neutered") else "중성화 안 함",
        ] if x
    )
    weight = f"{pet.get('weight_kg')}kg" if pet.get("weight_kg") else ""

    # 3. 상태 — 분류 + 확인된 위험 징후
    risk_label = {
        "normal": "정상 범위",
        "observe": "관찰 권장",
        "consult": "신속 상담 권장",
        "emergency": "응급 징후 가능성",
    }.get(risk_level, "관찰 권장")

    food_low = any(FOOD_LOW_RE.search(r.get("food", "")) for r in recent)
    lethargy = any(LETHARGY_RE.search(r.get("symptom", "")) for r in recent)
    vomiting = any((r.get("vomit") or "") not in NO_VOMIT_VALUES for r in recent)
    diarrhea = any(DIARRHEA_STOOL_RE.search(r.get("stool", "")) for r in recent)
    activity_low = any(ACTIVITY_LOW_RE.search(r.get("activity", "")) for r in recent)

    signs: list[str] = []
    if food_low:
        signs.append("식사량 감소")
    if lethargy:
        signs.append("기력 저하")
    if vomiting:
        signs.append("구토 관찰")
    if diarrhea:
        signs.append("설사")
    if activity_low:
        signs.append("활동 감소")

    # 4. 주호소 및 주요 변화 — 값이 없으면 빈 문자열로 둔다(위 docstring 참고).
    complaints = []
    if food_low:
        complaints.append("식욕 감소")
    if lethargy:
        complaints.append("기력 저하")
    if vomiting:
        complaints.append("구토")
    chief = " · ".join(complaints)

    change_parts = []
    food_low_days = sum(1 for r in recent if FOOD_LOW_RE.search(r.get("food", "")))
    if food_low_days:
        change_parts.append(f"최근 {food_low_days}일 식사량 감소")
    if activity_low:
        change_parts.append("활동 감소")
    if vomiting:
        change_parts.append("구토 발생")
    major_changes = " · ".join(change_parts)

    progress_parts = []
    foods = [r.get("food", "") for r in recent if r.get("food")]
    if foods:
        progress_parts.append(f"식사: {foods[-1]}")
    acts = [r.get("activity", "") for r in recent if r.get("activity")]
    if acts:
        progress_parts.append(f"활동: {acts[-1]}")
    vomits = [
        r.get("vomit", "") for r in recent if (r.get("vomit") or "") not in NO_VOMIT_VALUES
    ]
    if vomits:
        progress_parts.append(f"구토: {vomits[-1]}")
    progress = " · ".join(progress_parts)

    return {
        # 1. 문서 정보
        "title": "PetCare AI 병원 전달용 상태 요약",
        "data_period": data_period,
        # 2. 반려동물 정보
        "pet_name": pet.get("name", ""),
        "species": pet.get("species", ""),
        "breed": pet.get("breed", ""),
        "sex_neuter": sex_neuter,
        "age_label": pet.get("age_label", ""),
        "weight": weight,
        # 프로필에 입력하지 않은 항목을 "없음" 으로 바꾸지 않는다 — 병원이 읽을 때
        # '복용약 없음' 은 확인된 사실로 보이지만 실제로는 미입력일 뿐이다.
        "medications": pet.get("medications", ""),
        "allergies": pet.get("allergies", ""),
        # 3. 상태
        "risk_label": risk_label,
        "risk_signs": signs,
        # 4. 주호소 및 주요 변화
        "chief_complaint": chief,
        "major_changes": major_changes,
        "progress": progress,
        "owner_note": extra_note or "",
    }


class MockAgentClient:
    """Agent 를 붙이지 않고 서버만 띄웠을 때 쓰는 **최소** 클라이언트.

    이 클래스는 아무것도 판단하지 않는다. 모든 응답은 "AI 가 연결되지 않았다" 는
    사실만 전달하고 값 자리는 비워 둔다.

    ## 왜 예시 응답을 전부 지웠나

    예전 mock 은 화면이 비어 보이지 않도록 하드코딩된 값을 돌려줬다 — 고정 근거 한 줄
    ("WSAVA 보호자 가이드 …"), 고정 추가 질문, 고정 이동 안내 3개, `demo-normal` /
    `demo-consult` / `demo-emergency` 강제 상태, 그리고 정규식으로 일기·진단서에서
    지어낸 "사료 반쯤 남김 · 평소보다 감소" 같은 값들이다.

    문제는 화면 어디에도 "이건 예시입니다" 가 없다는 것이다. 보호자는 그 값을 AI
    판독 결과로 읽고, 일기 추출 결과는 확인 버튼 한 번으로 **그대로 DB 에 저장**된다.
    존재하지 않는 근거와 아무도 읽지 않은 기록이 진짜 건강 기록으로 남는 셈이다.
    빈 화면은 불편할 뿐이지만 가짜 값은 잘못된 판단을 만든다. 그래서 값을 만들지
    않고 미연결 사실만 알린다.

    실제 분석은 `AGENT_MODE=http` + ai 서비스(LangGraph)가 담당한다.
    """

    #: 응답 출처 표시 — 앱·로그에서 "실제 AI 결과가 아님" 을 구분할 수 있어야 한다.
    source = "mock-not-connected"

    # ---- 일기 구조화 -------------------------------------------------------
    def diary_extract(self, pet: dict, text: str, record_date: str, context: dict) -> dict:
        """항상 빈 7항목을 돌려준다(계약 유지, 값 없음).

        추출 결과는 보호자 확인 후 daily_entries 에 저장되므로, 여기서 지어낸 값은
        곧바로 영구 기록이 된다. 그래서 추측하지 않는다.
        """
        return {
            "items": [],
            "fields": {key: "" for key in RECORD_FIELD_KEYS},
            "source": self.source,
        }

    # ---- 진단서 구조화 -----------------------------------------------------
    def diagnosis_extract(self, pet: dict, file_name: str, file_text: str) -> dict:
        """항상 빈 진단서 필드를 돌려준다(라벨 정규식 파싱 제거).

        `items_read=0` 이면 앱은 "읽은 항목 없음" 으로 표시하고 보호자가 직접
        입력하게 된다 — 잘못 읽은 진단명을 보여 주는 것보다 안전하다.
        """
        return {
            "fields": {"date": None, "hospital": "", "diagnosis": "", "content": ""},
            "items_read": 0,
            "source": self.source,
        }

    # ---- AI 상태 체크 ------------------------------------------------------
    def health_check(
        self,
        pet: dict,
        messages: list[dict],
        context: dict,
        region_name: str | None = None,
        conversation_id: str | None = None,
    ) -> dict:
        """위험도를 판정하지 않고 미연결 사실만 알린다.

        `region_name`·`conversation_id` 는 쓰지 않지만 **시그니처는 맞춰 둔다** —
        라우터가 두 클라이언트를 같은 방식으로 호출하므로, 여기서 빠지면 mock
        모드에서만 TypeError 가 난다(실제로 그렇게 깨졌다).

        `region_name` 은 계약을 맞추려고 받기만 한다 — mock 은 병원을 검색하지 않으므로
        `hospitals` 는 항상 비어 있다(없는 병원을 지어내면 보호자가 실제로 전화한다).

        risk_level 을 `normal` 로 두는 이유: 분석을 하지 않았으니 위험을 올릴 근거도
        없다. 대신 reply 로 "분석할 수 없다" 를 분명히 말하고, 근거·추가질문·이동
        안내·요약 버튼은 모두 비활성으로 둔다.
        """
        return {
            "reply": NOT_CONNECTED_REPLY,
            "risk_level": "normal",
            "trend_summary": "",
            "trends": [],
            "reasons": [],
            "evidence": "",
            "followup_question": None,
            "can_generate_summary": False,
            "show_hospitals": False,
            "transit_guidance": [],
            "actions": [],
            "citations": [],
            "hospitals": [],
            "source": self.source,
        }

    # ---- 병원 전달용 요약 (문서 4섹션 구조) ---------------------------------
    def generate_summary(
        self, pet: dict, risk_level: str, extra_note: str, context: dict
    ) -> dict:
        """DB 에 저장된 프로필·기록만으로 4섹션을 채운다.

        4섹션 키는 전부 채우되(앱·PDF 렌더링 계약), 값은 실제 데이터에서 온 것만
        담는다. 없는 값은 빈 문자열이다 — `build_summary_content` 참고.
        """
        return {
            "content": build_summary_content(pet, risk_level, extra_note, context),
            "source": self.source,
        }


# ---------------------------------------------------------------------------
_client: AgentClient | None = None


def get_agent_client() -> AgentClient:
    global _client
    if _client is None:
        _client = HttpAgentClient() if settings.agent_mode == "http" else MockAgentClient()
    return _client
