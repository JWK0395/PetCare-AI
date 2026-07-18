"""AI Agent 연결 레이어.

Agent(Safety/Context/Trend/RAG/Summary)는 별도로 구현된다(ai/ 폴더 참고).
이 모듈은 서버와 Agent 사이의 "연결"만 담당한다.

- AGENT_MODE=http : 외부 Agent 서비스로 HTTP 요청을 전달한다. (계약: ai/README.md)
- AGENT_MODE=mock : Agent 없이도 앱이 동작하도록 내장 규칙 기반 응답을 돌려준다.

서버는 항상 동일한 payload(pet / messages / context)를 만들어 전달하므로,
실제 Agent는 이 계약만 맞추면 그대로 교체된다.

DB 스펙(daily_entries)에 따라 기록은 모두 텍스트 상태값이다. mock 은 정규식 기반으로
텍스트를 구조화하며, 수치 기반 기준선/추이는 계산하지 않는다(AI 가 텍스트로 판단).
"""

from __future__ import annotations

import logging
import re
from datetime import date, timedelta
from typing import Any, Protocol

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

    def health_check(self, pet: dict, messages: list[dict], context: dict) -> dict: ...

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
                "AGENT_MODE=mock 으로 바꾸면 내장 응답으로 동작합니다.",
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

    def health_check(self, pet: dict, messages: list[dict], context: dict) -> dict:
        return self._post(
            HEALTH_CHECK_PATH, {"pet": pet, "messages": messages, "context": context}
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


# ---------------------------------------------------------------------------
# Mock 모드 — 규칙 기반 (Agent 없이 앱 전체 흐름 확인용)
# ---------------------------------------------------------------------------
EMERGENCY_KEYWORDS = [
    "호흡곤란", "숨을 가쁘", "가쁘게", "헐떡", "숨을 잘 못", "숨쉬기 힘들",
    "청색", "파래", "파랗", "보라색",
    "경련", "발작", "의식이 없", "의식을 잃", "쓰러", "축 늘어",
    "중독", "쥐약", "부동액", "초콜릿을 먹", "포도를 먹", "양파를 먹", "자일리톨",
    "주워 먹", "침을 많이 흘리", "피를 토", "토혈", "하혈",
]

TRANSIT_GUIDANCE = ["기도 확보", "최대한 안정 유지", "음식과 물은 주지 않기"]
EVIDENCE_LINE = "WSAVA 보호자 가이드 2024 v2 · 최근 30일 기록"
FOLLOWUP_QUESTION = "지난 24시간 동안 구토나 설사가 있었나요?"

LETHARGY_RE = re.compile(r"축 처|기운이 없|기운 없|무기력|축 늘어|처져|기력")
VOMIT_RE = re.compile(r"구토|토를 하|토했|토를 했|토함|구역질|노란 토")
DIARRHEA_RE = re.compile(r"설사|묽은 변|무른 변")


def _demo_gate(text: str) -> tuple[bool, str]:
    """데모 비밀번호 게이트.

    - demo_password 가 비어 있으면 항상 활성(하위 호환).
    - 입력에 비밀번호가 포함되면 활성 + 비밀번호를 제거한 텍스트를 돌려준다.
    - 그 외에는 비활성 → 하드코딩 예시 응답을 내지 않는다.
    """
    pw = (settings.demo_password or "").strip()
    if not pw:
        return True, text
    if pw.lower() in (text or "").lower():
        cleaned = re.sub(re.escape(pw), "", text, flags=re.IGNORECASE).strip()
        return True, cleaned
    return False, text


def _time_of_day(text: str) -> str:
    for word in ["새벽", "아침", "오전", "점심", "오후", "저녁", "밤"]:
        if word in text:
            return word
    return ""


def build_summary_content(
    pet: dict, risk_level: str, extra_note: str, context: dict
) -> dict:
    """병원 전달용 상태 요약(문서 4섹션 구조)을 만든다.

    요약(summary) 생성과 응급 이메일이 공용으로 사용한다. 기록이 모두 텍스트이므로
    최근 기록의 상태값을 스캔해 위험 징후·주호소·변화를 구성한다.
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

    food_low = any(re.search(r"감소|남김|적게|거의", r.get("food", "")) for r in recent)
    lethargy = any(LETHARGY_RE.search(r.get("symptom", "")) for r in recent)
    vomiting = any(r.get("vomit") and r["vomit"] not in ("", "없음") for r in recent)
    diarrhea = any(
        "설사" in r.get("stool", "") or "묽은" in r.get("stool", "") for r in recent
    )
    activity_low = any(re.search(r"짧|거부|감소", r.get("activity", "")) for r in recent)

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

    # 4. 주호소 및 주요 변화
    complaints = []
    if food_low:
        complaints.append("식욕 감소")
    if lethargy:
        complaints.append("기력 저하")
    if vomiting:
        complaints.append("구토")
    chief = " · ".join(complaints) if complaints else "보호자 관찰 변화"

    change_parts = []
    food_low_days = sum(
        1 for r in recent if re.search(r"감소|남김|적게|거의", r.get("food", ""))
    )
    if food_low_days:
        change_parts.append(f"최근 {food_low_days}일 식사량 감소")
    if activity_low:
        change_parts.append("활동 감소")
    if vomiting:
        change_parts.append("구토 발생")
    major_changes = " · ".join(change_parts) or "특이 변화 관찰되지 않음"

    progress_parts = []
    foods = [r.get("food", "") for r in recent if r.get("food")]
    if foods:
        progress_parts.append(f"식사: {foods[-1]}")
    acts = [r.get("activity", "") for r in recent if r.get("activity")]
    if acts:
        progress_parts.append(f"활동: {acts[-1]}")
    vomits = [
        r.get("vomit", "")
        for r in recent
        if r.get("vomit") and r["vomit"] not in ("", "없음")
    ]
    if vomits:
        progress_parts.append(f"구토: {vomits[-1]}")
    progress = " · ".join(progress_parts) or "기록 부족"

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
        "medications": pet.get("medications", "") or "없음",
        "allergies": pet.get("allergies", "") or "없음",
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
    source = "mock"

    # ---- 일기 구조화 -------------------------------------------------------
    def diary_extract(self, pet: dict, text: str, record_date: str, context: dict) -> dict:
        empty = {
            "food": "", "water": "", "activity": "",
            "symptom": "", "stool": "", "vomit": "", "notes": "",
        }
        active, text = _demo_gate(text)
        if not active:
            # 실제 AI 연결 전에는 임의 입력에 대해 가짜 추출 결과를 내지 않는다.
            return {"items": [], "fields": dict(empty), "source": self.source}

        fields: dict[str, Any] = dict(empty)
        items: list[dict] = []

        # 식사
        gram = re.search(r"(\d+(?:\.\d+)?)\s*(?:g|그램|그람)", text)
        if re.search(r"(다 먹|전부 먹|완식|남기지 않)", text):
            fields["food"] = "사료 잘 먹음"
        elif re.search(r"(반쯤 남|절반|반 정도 남|반만 먹|반쯤 먹)", text):
            fields["food"] = "사료 반쯤 남김 · 평소보다 감소"
        elif re.search(r"(거의 안 먹|거의 먹지 않|입도 대지 않)", text):
            fields["food"] = "거의 먹지 않음"
        elif re.search(r"(안 먹|먹지 않)", text):
            fields["food"] = "먹지 않음"
        elif re.search(r"(적게 먹|평소보다 적|많이 남)", text):
            fields["food"] = "평소보다 적게 먹음"
        if gram:
            g = f"사료 {round(float(gram.group(1)))}g"
            fields["food"] = f"{g} · {fields['food']}" if fields["food"] else g
        if fields["food"]:
            items.append({"category": "식사", "value": fields["food"], "field": "food"})

        # 음수
        if re.search(r"물.{0,6}(잘 마|평소|비슷)", text):
            fields["water"] = "정상 범위"
        elif re.search(r"물.{0,10}(많이|빨리 비)", text):
            fields["water"] = "평소보다 많음"
        elif re.search(r"물.{0,10}(거의 안|안 마|조금 마)", text):
            fields["water"] = "평소보다 적음"
        if fields["water"]:
            items.append({"category": "음수", "value": fields["water"], "field": "water"})

        # 활동
        walk = re.search(r"산책.{0,10}?(\d+)\s*분", text)
        if walk:
            minutes = int(walk.group(1))
            note = f"산책 {minutes}분"
            if re.search(r"(걷기 싫|평소보다 짧|짧게)", text) or minutes < 25:
                note += " · 평소보다 짧음"
            fields["activity"] = note
        elif re.search(r"산책.{0,8}(안 |않|거부|싫어)", text) or re.search(
            r"(하루 종일 누워|계속 누워|산책은 하지 않)", text
        ):
            fields["activity"] = "산책 거부 · 활동 감소"
        elif re.search(r"걷기 싫어", text):
            fields["activity"] = "걷기 싫어함 · 활동 감소"
        if fields["activity"]:
            items.append({"category": "활동", "value": fields["activity"], "field": "activity"})

        # 구토
        if VOMIT_RE.search(text) and not re.search(r"구토.{0,4}(없|안 했)", text):
            m = re.search(r"(\d+)\s*(?:회|번)", text)
            count = m.group(1) if m else "1"
            color = "노란색 " if re.search(r"노란|노랑", text) else ""
            when = _time_of_day(text)
            fields["vomit"] = " · ".join(
                x for x in [f"{color}구토 {count}회", when] if x
            )
        elif re.search(r"구토.{0,4}(없|안 했)|토하지 않", text):
            fields["vomit"] = "없음"
        if fields["vomit"]:
            items.append({"category": "구토", "value": fields["vomit"], "field": "vomit"})

        # 증상 (기력 저하 / 절뚝거림)
        symptoms: list[str] = []
        if LETHARGY_RE.search(text):
            symptoms.append("기력 저하")
        if re.search(r"절뚝|다리를 (들|절|드는)", text):
            symptoms.append("절뚝거림")
        if symptoms:
            fields["symptom"] = " · ".join(symptoms)
            items.append({"category": "증상", "value": fields["symptom"], "field": "symptom"})

        # 배변 (설사 포함)
        if DIARRHEA_RE.search(text) and not re.search(r"설사.{0,4}(없|안 했)", text):
            fields["stool"] = "묽은 변 · 설사"
        elif re.search(r"(정상 변|배변은 정상|배변도 정상|배변은 평소)", text):
            fields["stool"] = "정상"
        elif re.search(r"(배변을 하지 않|배변 없|배변은 아직)", text):
            fields["stool"] = "배변 없음"
        if fields["stool"]:
            items.append({"category": "배변", "value": fields["stool"], "field": "stool"})

        return {"items": items, "fields": fields, "source": self.source}

    # ---- 진단서 구조화 -----------------------------------------------------
    def diagnosis_extract(self, pet: dict, file_name: str, file_text: str) -> dict:
        fields = {"date": None, "hospital": "", "diagnosis": "", "content": ""}
        # 진단서 추출은 업로드한 실제 문서를 파싱하므로 데모 게이트를 두지 않는다.
        text = file_text or ""

        def label(*labels: str) -> str:
            for lab in labels:
                m = re.search(rf"{lab}\s*[:：]\s*([^\n]+)", text)
                if m:
                    return m.group(1).strip()
            return ""

        # 병원 — 라벨 우선, 없으면 '○○동물병원' 패턴
        fields["hospital"] = label(r"발급\s*병원", "병원")
        if not fields["hospital"]:
            hm = re.search(
                r"([가-힣A-Za-z0-9]+\s*동물(?:병원|의료센터|클리닉))",
                file_name + " " + text,
            )
            if hm:
                fields["hospital"] = hm.group(1).strip()

        # 날짜 — 라벨 값 또는 본문에서 YYYY-MM-DD 파싱, 없으면 파일명(_MMDD)
        date_src = label("발급일", r"발급\s*연월일", "날짜") or text
        dm = re.search(r"(20\d{2})\D{1,3}(\d{1,2})\D{1,3}(\d{1,2})", date_src)
        if dm:
            y, mo, d = (int(g) for g in dm.groups())
            fields["date"] = f"{y:04d}-{mo:02d}-{d:02d}"
        else:
            mmdd = re.search(r"_(\d{2})(\d{2})(?:\.|_|$)", file_name)
            if mmdd:
                today = date.today()
                fields["date"] = (
                    f"{today.year:04d}-{int(mmdd.group(1)):02d}-{int(mmdd.group(2)):02d}"
                )

        # 진단명
        fields["diagnosis"] = label("진단명", "병명", r"최종\s*진단")

        # 진단 내용 — 기타사항/소견(여러 줄) + 처방·몸무게를 함께 서술로 담는다.
        prescription = label("처방", "투약", "치료명칭", "치료")
        weight = label("몸무게", "체중")
        cm = re.search(
            r"(?:기타사항|소견|진단\s*내용|임상\s*소견)\s*[:：]\s*(.+?)(?:\n\s*\d+\s*[.)]|\Z)",
            text,
            re.DOTALL,
        )
        parts: list[str] = []
        if cm:
            parts.append(re.sub(r"\s+", " ", cm.group(1)).strip())
        elif fields["diagnosis"]:
            parts.append(f"{fields['diagnosis']} 소견")
        if prescription:
            parts.append(f"처방: {prescription}")
        if weight:
            wm = re.search(r"(\d+(?:\.\d+)?)", weight)
            if wm:
                parts.append(f"체중 {wm.group(1)}kg")
        fields["content"] = " · ".join(parts)

        items_read = sum(1 for v in fields.values() if v not in ("", None))
        return {"fields": fields, "items_read": items_read, "source": self.source}

    def _forced_state(self, state: str, name: str) -> dict:
        """디자인 확인용 강제 상태 응답 (demo-normal/observe/consult/emergency)."""
        base = {
            "trend_summary": "",
            "trends": [],
            "reasons": [],
            "evidence": EVIDENCE_LINE,
            "followup_question": None,
            "can_generate_summary": False,
            "show_hospitals": False,
            "transit_guidance": [],
            "source": self.source,
        }
        if state == "emergency":
            return {
                **base,
                "reply": "응급 신호가 의심됩니다. 지금 바로 가까운 24시 동물병원에 연락하세요. "
                f"이동 중에는 {' · '.join(TRANSIT_GUIDANCE)}를 지켜주세요.",
                "risk_level": "emergency",
                "reasons": ["응급 규칙(호흡곤란 · 청색증 · 경련 · 중독 등)에 해당하는 표현 감지"],
                "can_generate_summary": True,
                "show_hospitals": True,
                "transit_guidance": TRANSIT_GUIDANCE,
            }
        if state == "consult":
            return {
                **base,
                "reply": "오늘 안에 병원 상담을 권해요",
                "risk_level": "consult",
                "trend_summary": "식사 감소 · 기력 저하 · 구토 관찰",
                "reasons": [
                    "식사량이 최근 3일 연속 평소보다 줄어든 기록",
                    "기력 저하 + 구토 동반 — 복합 신호",
                ],
                "can_generate_summary": True,
            }
        return {
            **base,
            "reply": f"최근 30일 기록 기준으로 {name}는 평소 범위 안에 있어요. "
            "변화가 느껴지면 언제든 기록해 주세요.",
            "risk_level": "normal",
        }

    # ---- AI 상태 체크 (Safety → 텍스트 기록 판단) --------------------------
    def health_check(self, pet: dict, messages: list[dict], context: dict) -> dict:
        user_messages = [m["content"] for m in messages if m["role"] == "user"]
        all_user_text = " ".join(user_messages)
        name = pet.get("name", "아이")

        # 디자인 확인용 강제 상태: 마지막 메시지에 demo-normal / demo-consult /
        # demo-emergency 가 있으면 해당 상태 응답을 바로 낸다. (AI 체크는 3상태만)
        last_user = user_messages[-1].lower() if user_messages else ""
        for state in ("emergency", "consult", "normal"):
            if f"demo-{state}" in last_user or f"demo {state}" in last_user:
                return self._forced_state(state, name)

        # 데모 게이트: 비밀번호가 없으면 예시 분석 대신 안내만 한다.
        active, cleaned = _demo_gate(all_user_text)
        if not active:
            return {
                "reply": "지금은 데모 준비 상태예요. AI 모델을 연결하면 최근 30일 기록과 "
                "RAG 지식을 바탕으로 상태를 분석해 드려요. "
                f"(디자인 예시를 보려면 입력에 '{settings.demo_password}'를 포함해 주세요.)",
                "risk_level": "normal",
                "trend_summary": "",
                "trends": [],
                "reasons": [],
                "evidence": "",
                "followup_question": None,
                "can_generate_summary": False,
                "show_hospitals": False,
                "transit_guidance": [],
                "source": self.source,
            }
        all_user_text = cleaned

        # 1) Safety — 응급 신호 우선
        if any(k in all_user_text for k in EMERGENCY_KEYWORDS):
            return {
                "reply": f"응급 신호가 의심됩니다. 지금 바로 가까운 24시 동물병원에 연락하세요. "
                f"이동 중에는 {' · '.join(TRANSIT_GUIDANCE)}를 지켜주세요.",
                "risk_level": "emergency",
                "trend_summary": "",
                "trends": [],
                "reasons": ["응급 규칙(호흡곤란 · 청색증 · 경련 · 중독 등)에 해당하는 표현 감지"],
                "evidence": EVIDENCE_LINE,
                "followup_question": None,
                "can_generate_summary": True,
                "show_hospitals": True,
                "transit_guidance": TRANSIT_GUIDANCE,
                "source": self.source,
            }

        # 2) 최근 기록(텍스트)에서 변화 신호 요약
        records = context.get("records", [])
        recent = records[-3:]
        food_low_days = sum(
            1 for r in recent if re.search(r"감소|남김|적게|거의", r.get("food", ""))
        )
        vomit_days = sum(
            1 for r in recent if r.get("vomit") and r["vomit"] not in ("", "없음")
        )
        record_lethargy = any(LETHARGY_RE.search(r.get("symptom", "")) for r in recent)

        trend_parts: list[str] = []
        trends: list[dict] = []
        if food_low_days:
            trend_parts.append(f"식사 감소 {food_low_days}일")
            trends.append(
                {"metric": "식사", "change_pct": None, "note": f"최근 {food_low_days}일 식사 감소 기록"}
            )
        if vomit_days:
            trend_parts.append("구토 관찰")
            trends.append({"metric": "구토", "change_pct": None, "note": f"구토 {vomit_days}일"})
        if record_lethargy:
            trend_parts.append("기력 저하")
        trend_summary = " · ".join(trend_parts)

        mentioned_vomit = bool(VOMIT_RE.search(all_user_text))
        mentioned_diarrhea = bool(DIARRHEA_RE.search(all_user_text))
        lethargy = bool(LETHARGY_RE.search(all_user_text)) or record_lethargy

        # 3) 정보가 부족하면 추가 질문 (첫 턴 & 구토/설사 언급 없음) — 상태는 normal
        if len(user_messages) <= 1 and not (mentioned_vomit or mentioned_diarrhea):
            reply = "기록을 확인했어요. "
            if trend_summary:
                reply += f"최근 기록에서 변화가 보여요 ({trend_summary}). "
            reply += "몇 가지만 더 확인할게요."
            return {
                "reply": reply,
                "risk_level": "normal",
                "trend_summary": trend_summary,
                "trends": trends,
                "reasons": [],
                "evidence": "",
                "followup_question": FOLLOWUP_QUESTION,
                "can_generate_summary": False,
                "show_hospitals": False,
                "transit_guidance": [],
                "source": self.source,
            }

        # 4) 최종 판단 — normal / consult 만 (observe 없음)
        reasons: list[str] = []
        if food_low_days >= 3:
            reasons.append("식사량이 최근 3일 연속 평소보다 줄어든 기록")
        elif food_low_days >= 1:
            reasons.append("최근 식사량 감소 기록")
        if lethargy and (mentioned_vomit or vomit_days):
            reasons.append("기력 저하 + 구토 동반 — 복합 신호")
        elif mentioned_vomit and mentioned_diarrhea:
            reasons.append("구토 + 설사 동반 — 복합 신호")

        has_signal = bool(reasons) or mentioned_vomit or mentioned_diarrhea or lethargy or trend_summary
        if has_signal:
            return {
                "reply": "오늘 안에 병원 상담을 권해요",
                "risk_level": "consult",
                "trend_summary": trend_summary,
                "trends": trends,
                "reasons": reasons or ["최근 기록·증상에서 변화 관찰"],
                "evidence": EVIDENCE_LINE,
                "followup_question": None,
                "can_generate_summary": True,
                "show_hospitals": False,
                "transit_guidance": [],
                "source": self.source,
            }

        return {
            "reply": f"최근 30일 기록 기준으로 {name}는 평소 범위 안에 있어요. 변화가 느껴지면 언제든 기록해 주세요.",
            "risk_level": "normal",
            "trend_summary": trend_summary,
            "trends": trends,
            "reasons": [],
            "evidence": EVIDENCE_LINE,
            "followup_question": None,
            "can_generate_summary": False,
            "show_hospitals": False,
            "transit_guidance": [],
            "source": self.source,
        }

    # ---- 병원 전달용 요약 (문서 4섹션 구조) ---------------------------------
    def generate_summary(
        self, pet: dict, risk_level: str, extra_note: str, context: dict
    ) -> dict:
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
