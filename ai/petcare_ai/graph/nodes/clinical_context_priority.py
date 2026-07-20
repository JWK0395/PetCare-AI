"""Clinical Context Priority Agent — 현재 질문에 필요한 자료만 골라낸다(명세 27절).

우선순위(명세 20절, 전 그래프 공통):

    현재 사용자 입력 > PET DB > 진단서 DB > 일기장 DB

수행 작업은 딱 이것뿐이다.
  - 현재 증상 구조화
  - PET DB 필수정보 선택
  - 현재 증상과 관련된 진단서 선택
  - 최근 관련 일기 선택
  - 날짜순 정렬
  - source provenance 기록
  - 충돌 필드 기록(ContextConflict)

**하지 않는 것**: 일기와 진단서를 다시 요약하거나 파싱하지 않는다. 선택된 레코드는
원문 dict 그대로 넘긴다. 여기서 다시 요약하면 (a) 이미 처리된 원문과 다른 두 번째
버전이 생기고 (b) PDF 에 넣을 원문 보존이 깨진다.

유일한 예외는 **충돌 검출용 단일 필드 조회**다(명세 43절: PET DB 5.2kg / 진단서
4.8kg / 사용자 4.5kg). 체중처럼 여러 출처에 같은 필드가 있으면 값을 비교해야만
충돌을 기록할 수 있다. 이것은 레코드 재해석이 아니라 지정된 한 필드를 읽는 것이며,
임의로 하나를 정답으로 확정하지 않고 **모든 값과 출처를 함께 남긴다**.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import TYPE_CHECKING, Any, Callable

from pydantic import BaseModel, Field

from ...schemas import ClinicalContext, ContextConflict, ContextProvenance

if TYPE_CHECKING:  # state.py 는 동시 작성 중이다.
    from ..state import PetCareState  # noqa: F401

logger = logging.getLogger(__name__)

__all__ = [
    "SOURCE_USER",
    "SOURCE_PET",
    "SOURCE_DIAGNOSIS",
    "SOURCE_DAILY",
    "SOURCE_PRIORITY",
    "SymptomTerm",
    "SYMPTOM_TAXONOMY",
    "PET_ESSENTIAL_FIELDS",
    "CurrentObservationDraft",
    "extract_current_observation",
    "select_related_diagnoses",
    "select_supporting_daily_entries",
    "detect_context_conflicts",
    "build_clinical_context",
    "make_clinical_context_priority_node",
    "clinical_context_priority_node",
]

# ---------------------------------------------------------------------------
# 출처 우선순위 (명세 20절)
# ---------------------------------------------------------------------------
SOURCE_USER = "user_input"
SOURCE_PET = "pet_db"
SOURCE_DIAGNOSIS = "diagnosis_db"
SOURCE_DAILY = "daily_entry_db"

SOURCE_PRIORITY: dict[str, int] = {
    SOURCE_USER: 0,
    SOURCE_PET: 1,
    SOURCE_DIAGNOSIS: 2,
    SOURCE_DAILY: 3,
}

# PET DB 필수정보 — LLM prompt 와 PDF 에 항상 실리는 항목(명세 20절 용도 정의)
PET_ESSENTIAL_FIELDS: tuple[str, ...] = (
    "id", "name", "species", "breed", "birth_date", "age_years", "sex",
    "is_neutered", "weight_kg", "diseases", "medications", "supplement", "allergies",
)

# 관련 자료 선택 범위
_DIAGNOSIS_MAX = 5
_DAILY_MAX = 7
_DAILY_WINDOW_DAYS = 21
_RECENT_DIAGNOSIS_DAYS = 180


# ---------------------------------------------------------------------------
# 증상 사전
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SymptomTerm:
    """증상 1종. 모든 패턴은 **공백 제거 형태**로 적는다(띄어쓰기 편차 대응).

    `patterns` 는 사용자 입력에서 증상을 찾을 때 쓰고,
    `related_terms` 는 그 증상과 관련된 진단서·일기를 고를 때 쓴다.
    관련어에 질환명을 넣는 이유: 보호자는 "기침" 이라고 말하지만 진단서에는
    "이첨판 폐쇄부전증" 이라고 적혀 있어서 표면 단어만으로는 연결되지 않는다.
    """

    code: str
    label: str
    patterns: tuple[str, ...]
    related_terms: tuple[str, ...]


SYMPTOM_TAXONOMY: tuple[SymptomTerm, ...] = (
    SymptomTerm(
        code="cough", label="기침",
        patterns=("기침", "콜록", "켁켁", "캑캑"),
        related_terms=(
            "기침", "기관", "기관지", "기관허탈", "폐", "심장", "심잡음", "심부전",
            "이첨판", "판막", "mitral", "acvim", "심장약",
        ),
    ),
    SymptomTerm(
        code="respiratory", label="호흡 이상",
        patterns=("숨", "호흡", "헥헥", "헐떡", "가쁘"),
        related_terms=(
            "호흡", "숨", "폐", "흉부", "심장", "심부전", "기관", "산소", "청진",
        ),
    ),
    SymptomTerm(
        code="vomiting", label="구토",
        patterns=("토해", "토했", "토함", "구토", "게워", "웩"),
        related_terms=("구토", "위", "위염", "췌장", "장염", "이물", "소화기"),
    ),
    SymptomTerm(
        code="diarrhea", label="설사·무른 변",
        patterns=("설사", "무른변", "변이무름", "묽은변", "물똥"),
        related_terms=("설사", "장염", "대장", "소화기", "변"),
    ),
    SymptomTerm(
        code="anorexia", label="식욕 저하",
        patterns=("안먹", "못먹", "덜먹", "식욕", "밥을거부", "사료를거부"),
        related_terms=("식욕", "체중", "영양", "거식"),
    ),
    SymptomTerm(
        code="lethargy", label="기력 저하",
        patterns=("기운이없", "기력", "무기력", "축처", "누워만", "활동량이줄", "지침"),
        related_terms=("기력", "활동", "쇠약", "운동불내성", "무기력"),
    ),
    SymptomTerm(
        code="skin", label="피부·가려움",
        patterns=("가려", "긁", "피부", "발진", "탈모", "각질", "붉어"),
        related_terms=("피부", "아토피", "알레르기", "외이염", "농피증", "곰팡이"),
    ),
    SymptomTerm(
        code="lameness", label="보행 이상",
        patterns=("절뚝", "다리를절", "걷지못", "다리를들", "잘못걷"),
        related_terms=("슬개골", "관절", "십자인대", "디스크", "골절", "파행", "정형"),
    ),
    SymptomTerm(
        code="urinary", label="배뇨 이상",
        patterns=("소변", "오줌", "혈뇨", "화장실을자주"),
        related_terms=("방광", "요로", "결석", "신장", "배뇨", "요도"),
    ),
    SymptomTerm(
        code="polydipsia", label="음수량 변화",
        patterns=("물을많이", "물을안", "음수량", "물을적게"),
        related_terms=("신장", "당뇨", "쿠싱", "음수", "탈수"),
    ),
    SymptomTerm(
        code="weight", label="체중 변화",
        patterns=("살이빠", "체중이줄", "체중이늘", "말라", "야위", "살이쪘"),
        related_terms=("체중", "비만", "저체중", "영양"),
    ),
    SymptomTerm(
        code="eye", label="눈 증상",
        patterns=("눈곱", "충혈", "눈물", "눈을못뜨", "눈이"),
        related_terms=("각막", "결막", "포도막", "안과", "눈"),
    ),
    SymptomTerm(
        code="ear", label="귀 증상",
        patterns=("귀를", "귀에서", "귀가", "귀를긁"),
        related_terms=("외이염", "중이염", "귀", "이도"),
    ),
    SymptomTerm(
        code="oral", label="구강 증상",
        patterns=("잇몸", "이빨", "치아", "입냄새", "침을흘"),
        related_terms=("치주", "치석", "구내염", "스케일링", "발치", "치은염"),
    ),
    SymptomTerm(
        code="neuro", label="신경 증상",
        patterns=("경련", "발작", "쓰러", "떨어", "떨고", "비틀", "빙빙"),
        related_terms=("신경", "뇌", "간질", "전정", "발작", "경련"),
    ),
    SymptomTerm(
        code="fever", label="발열·체온 이상",
        patterns=("열이", "체온", "뜨거"),
        related_terms=("발열", "감염", "염증", "체온"),
    ),
    SymptomTerm(
        code="lump", label="종괴·부기",
        patterns=("혹이", "멍울", "부었", "붓기", "덩어리"),
        related_terms=("종양", "종괴", "림프", "부종", "낭종"),
    ),
)

# 일기에서 '이상 소견' 으로 볼 표현 — 증상어가 없어도 참고 자료로 올린다.
_DAILY_ABNORMAL_MARKERS: tuple[str, ...] = (
    "절반", "1/3", "3분의1", "거의먹지", "안먹", "못먹", "적게", "줄었", "감소",
    "무름", "설사", "거부", "지침", "힘들", "잦아", "증가", "심함", "이상",
)

# 증상 지속기간·발생 시점 표현
_DURATION_RE = re.compile(r"(\d+)\s*(일|주|개월|달|시간|분)\s*(?:째|전|동안|간|정도)?")
_ONSET_WORDS: tuple[str, ...] = (
    "오늘", "어제", "그제", "그저께", "방금", "조금전", "아까", "새벽", "아침",
    "점심", "저녁", "밤", "지난주", "이번주", "며칠전", "얼마전",
)
_SEVERITY_WORDS: tuple[str, ...] = (
    "심하", "심해", "매우", "너무", "많이", "급격", "갑자기", "계속", "지속",
    "반복", "악화", "점점", "약간", "살짝", "조금",
)

# 충돌 검출용 단일 필드 조회 정규식(레코드 재해석이 아니다)
_WEIGHT_RE = re.compile(r"(?:체중|몸무게|weight)[^0-9]{0,8}(\d{1,3}(?:\.\d{1,2})?)\s*(?:kg|킬로)")
_WEIGHT_PLAIN_RE = re.compile(r"(\d{1,3}(?:\.\d{1,2})?)\s*(?:kg|킬로그램|킬로)")
_TEMPERATURE_RE = re.compile(r"(\d{2}(?:\.\d)?)\s*(?:도|℃|°c)")
_RESP_RATE_RE = re.compile(r"(?:호흡수|분당호흡)[^0-9]{0,6}(\d{1,3})")
_MED_STOP_RE = re.compile(r"약을?(?:\s*)(?:끊|중단|그만|안\s*먹|안\s*줬|안\s*줍|빼)")


class CurrentObservationDraft(BaseModel):
    """LLM structured output 스키마 — 서술형 답변이 아니라 구조만 받는다.

    측정값(체중·체온 등)은 여기 두지 않는다. 숫자는 규칙 추출만 신뢰한다 —
    LLM 이 단위를 바꾸거나 값을 만들어내면 그대로 PDF 에 실리기 때문이다.
    """

    symptoms: list[str] = Field(default_factory=list)
    duration: str = ""
    onset: str = ""
    severity: str = ""
    notes: str = ""


# ---------------------------------------------------------------------------
# 공통 유틸
# ---------------------------------------------------------------------------
def _compact(text: Any) -> str:
    """공백·구두점을 제거한 매칭용 문자열.

    **소수점은 보존한다.** 이 문자열에서 체중을 읽기 때문에, 마침표를 무조건
    지우면 '4.5kg' 가 '45kg' 가 되어 충돌 검출이 통째로 틀어진다.
    숫자 사이에 있지 않은 마침표만 제거한다.
    """
    if not text:
        return ""
    lowered = re.sub(r"(?<!\d)\.|\.(?!\d)", "", str(text).lower())
    return re.sub(r"[\s,!?~·…\-_'\"()\[\]{}/]+", "", lowered)


def _parse_date(value: Any) -> date | None:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if not value:
        return None
    text = str(value).strip()
    for candidate in (text, text[:10]):
        try:
            return date.fromisoformat(candidate)
        except ValueError:
            continue
    return None


def _record_text(record: dict[str, Any], fields: tuple[str, ...]) -> str:
    return " ".join(str(record.get(field) or "") for field in fields)


def _sort_by_date(items: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    """날짜 오름차순(마지막이 최신). 날짜 불명 레코드는 버리지 않고 앞에 둔다."""
    return sorted(items, key=lambda item: (_parse_date(item.get(key)) or date.min))


def _to_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


# ---------------------------------------------------------------------------
# 1) 현재 증상 구조화
# ---------------------------------------------------------------------------
def extract_current_observation(
    user_message: str | None,
    collected_information: dict[str, Any] | None = None,
    llm: Any | None = None,
    today: date | None = None,
) -> dict[str, Any]:
    """현재 사용자 입력을 구조화한다 — 최우선 정보원이다(명세 20절).

    규칙 추출이 기본이고 LLM 은 증상 라벨·경과 서술을 보강하는 용도로만 쓴다.
    LLM 이 없어도(키 없음) 전부 채워진다.
    """
    raw = (user_message or "").strip()
    compact = _compact(raw)

    symptoms: list[str] = []
    matched_terms: list[str] = []
    for term in SYMPTOM_TAXONOMY:
        if any(pattern in compact for pattern in term.patterns):
            symptoms.append(term.label)
            matched_terms.append(term.code)

    duration = ""
    match = _DURATION_RE.search(raw)
    if match:
        duration = f"{match.group(1)}{match.group(2)}"
    onset = next((word for word in _ONSET_WORDS if word in raw), "")
    severity = next((word for word in _SEVERITY_WORDS if word in raw), "")

    observation: dict[str, Any] = {
        "raw_text": raw,
        "symptoms": symptoms,
        "symptom_codes": matched_terms,
        "duration": duration,
        "onset": onset,
        "severity": severity,
        "measurements": _extract_measurements(raw),
        "reported_at": (today or date.today()).isoformat(),
        "source": SOURCE_USER,
    }

    # Missing Information Agent 가 되물어 받은 값도 '현재 사용자 입력' 이다.
    if collected_information:
        observation["collected_information"] = dict(collected_information)

    if llm is not None and raw:
        observation = _enrich_with_llm(observation, raw, llm)

    return observation


def _extract_measurements(raw: str) -> dict[str, Any]:
    """사용자가 말한 수치를 뽑는다(체중·체온·호흡수). 없으면 빈 dict."""
    measurements: dict[str, Any] = {}
    text = raw.lower()
    compact = _compact(raw)

    # 단위만 보고 체중이라 단정하지 않는다("사료 5kg 샀어요" 오인 방지).
    # 문장에 체중 문맥이 있을 때만 단위 단독 표현을 체중으로 받는다.
    has_weight_context = any(word in compact for word in ("몸무게", "체중", "weight"))
    weight_match = _WEIGHT_RE.search(compact)
    if weight_match is None and has_weight_context:
        weight_match = _WEIGHT_PLAIN_RE.search(text)
    if weight_match:
        weight = _to_float(weight_match.group(1))
        if weight is not None:
            measurements["weight_kg"] = weight

    temperature_match = _TEMPERATURE_RE.search(text)
    if temperature_match:
        temperature = _to_float(temperature_match.group(1))
        # 상온·나이 등 다른 숫자를 체온으로 오인하지 않도록 생리 범위만 받는다.
        if temperature is not None and 30.0 <= temperature <= 45.0:
            measurements["temperature_c"] = temperature

    resp_match = _RESP_RATE_RE.search(_compact(raw))
    if resp_match:
        rate = _to_float(resp_match.group(1))
        if rate is not None:
            measurements["respiratory_rate"] = int(rate)

    return measurements


_OBSERVATION_SYSTEM_PROMPT = (
    "너는 보호자의 메시지에서 '현재 관찰된 증상' 만 구조화한다.\n"
    "금지: 진단명 추정, 처방, 약 변경 안내, 위험도 판정, 조언 문장 생성.\n"
    "메시지에 실제로 적힌 내용만 쓰고, 없으면 빈 문자열로 둔다. 숫자(체중·체온 등)는 "
    "쓰지 않는다. 한국어로 간결하게 답한다."
)


def _enrich_with_llm(observation: dict[str, Any], raw: str, llm: Any) -> dict[str, Any]:
    """LLM 으로 증상 라벨을 보강한다. 실패해도 규칙 결과가 그대로 남는다."""
    from ...llm import safe_structured_invoke  # 지연 import

    default = CurrentObservationDraft(
        symptoms=list(observation["symptoms"]),
        duration=observation["duration"],
        onset=observation["onset"],
        severity=observation["severity"],
    )
    draft = safe_structured_invoke(
        llm,
        [
            {"role": "system", "content": _OBSERVATION_SYSTEM_PROMPT},
            {"role": "user", "content": raw},
        ],
        CurrentObservationDraft,
        default,
    )

    merged = list(observation["symptoms"])
    for symptom in draft.symptoms:
        label = str(symptom).strip()
        if label and label not in merged:
            merged.append(label)
    observation["symptoms"] = merged
    observation["duration"] = observation["duration"] or draft.duration.strip()
    observation["onset"] = observation["onset"] or draft.onset.strip()
    observation["severity"] = observation["severity"] or draft.severity.strip()
    if draft.notes.strip():
        observation["notes"] = draft.notes.strip()
    return observation


# ---------------------------------------------------------------------------
# 2) PET DB 필수정보 선택
# ---------------------------------------------------------------------------
def _select_pet_context(pet_profile: dict[str, Any]) -> dict[str, Any]:
    """PET DB 에서 상담에 필요한 필드만 고른다(값은 가공하지 않는다)."""
    return {
        field: pet_profile.get(field)
        for field in PET_ESSENTIAL_FIELDS
        if pet_profile.get(field) not in (None, "")
    }


def _disease_terms(pet_profile: dict[str, Any]) -> list[str]:
    """기저질환 문자열을 매칭용 토큰으로 나눈다(요약이 아니라 분리다)."""
    raw = str(pet_profile.get("diseases") or "")
    tokens = re.split(r"[,/·;()\[\]]+", raw)
    return [_compact(token) for token in tokens if len(_compact(token)) >= 2]


# ---------------------------------------------------------------------------
# 3) 관련 진단서 선택
# ---------------------------------------------------------------------------
def select_related_diagnoses(
    diagnoses: list[dict[str, Any]],
    observation: dict[str, Any],
    pet_profile: dict[str, Any],
    today: date | None = None,
) -> tuple[list[dict[str, Any]], list[str]]:
    """현재 증상과 관련된 진단서를 고른다. 레코드는 **원문 그대로** 넘긴다.

    반환: (선택된 진단서(날짜 오름차순), 선택 이유 목록)

    최신 진단서 1건은 관련어가 없어도 항상 포함한다. 현재 몸무게·복용약·최근
    치료 경과의 기준값이 거기 있고, 이를 빼면 PDF 와 병원 요구사항 생성에서
    기준 정보가 사라진다.
    """
    if not diagnoses:
        return [], []

    ref = today or date.today()
    codes = set(observation.get("symptom_codes") or [])
    terms = [term for term in SYMPTOM_TAXONOMY if term.code in codes]
    disease_tokens = _disease_terms(pet_profile)

    scored: list[tuple[int, int, dict[str, Any], list[str]]] = []
    for index, diagnosis in enumerate(diagnoses):
        text = _compact(_record_text(diagnosis, ("diagnosis", "content", "hospital")))
        score = 0
        reasons: list[str] = []

        for term in terms:
            if any(related in text for related in term.related_terms):
                score += 3
                reasons.append(f"현재 증상 '{term.label}' 관련")

        for token in disease_tokens:
            if token in text:
                score += 2
                reasons.append("PET DB 기저질환 관련")
                break

        recorded = _parse_date(diagnosis.get("date"))
        if recorded and (ref - recorded).days <= _RECENT_DIAGNOSIS_DAYS:
            score += 1
            reasons.append(f"최근 {_RECENT_DIAGNOSIS_DAYS}일 이내 기록")

        if score > 0:
            scored.append((score, index, diagnosis, sorted(set(reasons))))

    scored.sort(key=lambda item: (-item[0], -item[1]))
    selected = [item[2] for item in scored[:_DIAGNOSIS_MAX]]
    reason_map = {id(item[2]): item[3] for item in scored[:_DIAGNOSIS_MAX]}

    latest = _sort_by_date(list(diagnoses), "date")[-1]
    if all(item is not latest for item in selected):
        selected.append(latest)
        reason_map[id(latest)] = ["최신 진단서(기준 정보)"]

    ordered = _sort_by_date(selected, "date")
    reasons_out = [
        f"{item.get('date', '날짜미상')} {item.get('diagnosis', '')}: "
        + ", ".join(reason_map.get(id(item), ["기준 정보"]))
        for item in ordered
    ]
    return ordered, reasons_out


# ---------------------------------------------------------------------------
# 4) 관련 일기 선택 (보조자료)
# ---------------------------------------------------------------------------
def select_supporting_daily_entries(
    daily_entries: list[dict[str, Any]],
    observation: dict[str, Any],
    today: date | None = None,
    window_days: int = _DAILY_WINDOW_DAYS,
) -> list[dict[str, Any]]:
    """최근 관련 일기를 고른다. 일기장은 **보조자료**이며 요약하지 않는다.

    선택 기준
      1) 최근 `window_days` 이내
      2) 현재 증상 관련어가 있거나, 식사·활동·배변에 이상 표현이 있는 기록
      3) 하나도 없으면 최근 기록 3건(경과 비교용 기준선)
    """
    if not daily_entries:
        return []

    ref = today or date.today()
    ordered = _sort_by_date(list(daily_entries), "record_date")
    recent = [
        entry
        for entry in ordered
        if (_parse_date(entry.get("record_date")) or date.min) >= ref - timedelta(days=window_days)
    ] or ordered[-_DAILY_MAX:]

    codes = set(observation.get("symptom_codes") or [])
    terms = [term for term in SYMPTOM_TAXONOMY if term.code in codes]

    matched: list[dict[str, Any]] = []
    for entry in recent:
        text = _compact(
            _record_text(
                entry,
                ("raw_text", "symptom", "notes", "food", "water", "activity", "stool", "vomit"),
            )
        )
        related = any(
            any(word in text for word in term.related_terms) or
            any(word in text for word in term.patterns)
            for term in terms
        )
        abnormal = any(marker in text for marker in _DAILY_ABNORMAL_MARKERS)
        if related or abnormal:
            matched.append(entry)

    if not matched:
        matched = recent[-3:]

    return matched[-_DAILY_MAX:]


# ---------------------------------------------------------------------------
# 5) 충돌 검출
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class _Candidate:
    """한 필드에 대한 출처별 후보값."""

    value: Any
    source: str
    recorded_at: str | None = None
    note: str = ""


def _normalized(value: Any) -> Any:
    """비교용 정규화 — 부동소수 오차로 가짜 충돌이 생기지 않게 한다."""
    if isinstance(value, float):
        return round(value, 2)
    if isinstance(value, str):
        return value.strip()
    return value


def _resolve_field(
    field: str, candidates: list[_Candidate]
) -> tuple[_Candidate | None, ContextConflict | None]:
    """우선순위대로 값을 고르고, 값이 갈리면 충돌로 기록한다.

    선택은 하되 **다른 값을 버리지 않는다**. 임의로 하나만 확정하면 안 되고
    (명세 43절), PDF 에 모든 값과 출처가 보존되어야 한다.
    """
    usable = [c for c in candidates if c.value not in (None, "")]
    if not usable:
        return None, None

    usable.sort(key=lambda c: SOURCE_PRIORITY.get(c.source, 99))
    selected = usable[0]

    distinct = {_normalized(c.value) for c in usable}
    if len(distinct) < 2:
        return selected, None

    conflict = ContextConflict(
        field=field,
        selected_value=selected.value,
        selected_source=selected.source,
        conflicting_values=[
            {
                "value": c.value,
                "source": c.source,
                "recorded_at": c.recorded_at,
                "note": c.note,
            }
            for c in usable
        ],
    )
    logger.info(
        "context 충돌 기록 — %s: %s",
        field,
        ", ".join(f"{c.source}={c.value}" for c in usable),
    )
    return selected, conflict


def _weight_from_record(record: dict[str, Any], text_fields: tuple[str, ...]) -> float | None:
    """레코드에서 체중 한 필드만 읽는다(요약·재파싱이 아니다).

    구조화된 `weight_kg` 필드가 있으면 그것을 쓰고, 없을 때만 본문에서 '체중
    4.8kg' 형태를 찾는다. 명세 43절 충돌 fixture 가 정확히 이 형태다.
    """
    direct = _to_float(record.get("weight_kg"))
    if direct is not None:
        return direct
    match = _WEIGHT_RE.search(_compact(_record_text(record, text_fields)))
    return _to_float(match.group(1)) if match else None


def detect_context_conflicts(
    observation: dict[str, Any],
    pet_profile: dict[str, Any],
    diagnoses: list[dict[str, Any]],
    daily_entries: list[dict[str, Any]],
) -> tuple[list[ContextConflict], dict[str, _Candidate]]:
    """출처별로 값이 다른 필드를 찾는다. 반환: (충돌 목록, 필드별 선택값).

    현재 검사 필드
      - weight_kg : PET DB / 최신 진단서 / 사용자 현재 입력 (명세 43절)
      - medications : PET DB 기록 vs 보호자의 '약 중단' 발화
                      (복용 여부가 어긋난 채 답하면 위험하므로 반드시 남긴다)
    """
    conflicts: list[ContextConflict] = []
    selected: dict[str, _Candidate] = {}

    # --- 체중 --------------------------------------------------------------
    weight_candidates: list[_Candidate] = []
    user_weight = _to_float((observation.get("measurements") or {}).get("weight_kg"))
    if user_weight is not None:
        weight_candidates.append(
            _Candidate(user_weight, SOURCE_USER, observation.get("reported_at"), "보호자 현재 보고")
        )

    pet_weight = _to_float(pet_profile.get("weight_kg"))
    if pet_weight is not None:
        weight_candidates.append(_Candidate(pet_weight, SOURCE_PET, None, "PET DB 등록 체중"))

    for diagnosis in reversed(_sort_by_date(list(diagnoses), "date")):
        diagnosis_weight = _weight_from_record(diagnosis, ("content", "diagnosis"))
        if diagnosis_weight is not None:
            weight_candidates.append(
                _Candidate(
                    diagnosis_weight,
                    SOURCE_DIAGNOSIS,
                    str(diagnosis.get("date") or ""),
                    f"{diagnosis.get('hospital') or '병원'} 진단서 기록",
                )
            )
            break  # 가장 최신 진단서 1건만 비교 대상으로 삼는다.

    for entry in reversed(_sort_by_date(list(daily_entries), "record_date")):
        entry_weight = _weight_from_record(entry, ("raw_text", "notes"))
        if entry_weight is not None:
            weight_candidates.append(
                _Candidate(
                    entry_weight, SOURCE_DAILY, str(entry.get("record_date") or ""), "일기장 기록"
                )
            )
            break

    weight_selected, weight_conflict = _resolve_field("weight_kg", weight_candidates)
    if weight_selected is not None:
        selected["weight_kg"] = weight_selected
    if weight_conflict is not None:
        conflicts.append(weight_conflict)

    # --- 복용약 ------------------------------------------------------------
    pet_medications = str(pet_profile.get("medications") or "").strip()
    if pet_medications and _MED_STOP_RE.search(_compact(observation.get("raw_text"))):
        med_candidates = [
            _Candidate(
                "보호자 보고: 현재 복용 중단",
                SOURCE_USER,
                observation.get("reported_at"),
                "대화에서 복용 중단 언급",
            ),
            _Candidate(pet_medications, SOURCE_PET, None, "PET DB 등록 복용약"),
        ]
        med_selected, med_conflict = _resolve_field("medications", med_candidates)
        if med_selected is not None:
            selected["medications"] = med_selected
        if med_conflict is not None:
            conflicts.append(med_conflict)

    return conflicts, selected


# ---------------------------------------------------------------------------
# 6) provenance
# ---------------------------------------------------------------------------
def _build_provenance(
    observation: dict[str, Any],
    pet_context: dict[str, Any],
    related_diagnoses: list[dict[str, Any]],
    daily_entries: list[dict[str, Any]],
    selected_fields: dict[str, _Candidate],
) -> list[ContextProvenance]:
    """어떤 값이 어디서 왔는지 기록한다 — PDF·답변의 출처 표기 근거가 된다."""
    records: list[ContextProvenance] = []

    if observation.get("symptoms"):
        records.append(
            ContextProvenance(
                field="current_symptoms",
                value=observation["symptoms"],
                source=SOURCE_USER,
                recorded_at=observation.get("reported_at"),
            )
        )
    for name, value in (observation.get("measurements") or {}).items():
        records.append(
            ContextProvenance(
                field=f"current_{name}",
                value=value,
                source=SOURCE_USER,
                recorded_at=observation.get("reported_at"),
            )
        )

    for field, value in pet_context.items():
        candidate = selected_fields.get(field)
        records.append(
            ContextProvenance(
                field=field,
                value=candidate.value if candidate else value,
                source=candidate.source if candidate else SOURCE_PET,
                recorded_at=candidate.recorded_at if candidate else None,
            )
        )

    for diagnosis in related_diagnoses:
        records.append(
            ContextProvenance(
                field="related_diagnosis",
                value=diagnosis.get("diagnosis"),
                source=SOURCE_DIAGNOSIS,
                recorded_at=str(diagnosis.get("date") or "") or None,
            )
        )

    for entry in daily_entries:
        records.append(
            ContextProvenance(
                field="supporting_daily_entry",
                value=entry.get("symptom") or entry.get("raw_text"),
                source=SOURCE_DAILY,
                recorded_at=str(entry.get("record_date") or "") or None,
            )
        )

    return records


# ---------------------------------------------------------------------------
# 조립
# ---------------------------------------------------------------------------
def build_clinical_context(
    state: dict,
    llm: Any | None = None,
    today: date | None = None,
) -> ClinicalContext:
    """State 의 전체 임상 데이터에서 현재 질문에 필요한 부분만 골라낸다.

    State 에는 전체가 남고, 여기서 고른 것만 LLM prompt 와 PDF 로 간다(명세 21절).
    """
    ref = today or date.today()
    pet_profile = dict(state.get("pet_profile") or {})
    diagnoses = list(state.get("diagnoses") or [])
    daily_entries = list(state.get("daily_entries") or [])

    observation = extract_current_observation(
        state.get("user_message"),
        collected_information=state.get("collected_information"),
        llm=llm,
        today=ref,
    )

    pet_context = _select_pet_context(pet_profile)
    related, reasons = select_related_diagnoses(diagnoses, observation, pet_profile, today=ref)
    supporting = select_supporting_daily_entries(daily_entries, observation, today=ref)
    conflicts, selected_fields = detect_context_conflicts(
        observation, pet_profile, diagnoses, daily_entries
    )

    if reasons:
        # 선택 근거는 trace 확인용이며, 진단서 원문을 바꾸지 않는다.
        observation["diagnosis_selection_reasons"] = reasons

    # 충돌이 있어도 하나로 확정하지 않는다. 선택값은 우선순위 결과일 뿐이며,
    # 원본 PET DB 값은 `*_pet_db` 로 함께 남겨 PDF 에서 모두 보존한다.
    for field, candidate in selected_fields.items():
        if field in pet_context and _normalized(pet_context[field]) != _normalized(candidate.value):
            pet_context[f"{field}_pet_db"] = pet_context[field]
        pet_context[field] = candidate.value
        pet_context[f"{field}_source"] = candidate.source

    provenance = _build_provenance(
        observation, pet_context, related, supporting, selected_fields
    )

    return ClinicalContext(
        current_observation=observation,
        priority_pet_context=pet_context,
        related_diagnoses=related,
        supporting_daily_entries=supporting,
        context_conflicts=conflicts,
        context_provenance=provenance,
    )


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------
def make_clinical_context_priority_node(llm: Any | None = None) -> Callable[[dict], dict]:
    """LLM 을 주입한 노드를 만든다(테스트가 mock LLM 을 넣을 수 있게)."""

    def _node(state: dict) -> dict:
        return _prioritize(state, llm)

    return _node


def clinical_context_priority_node(state: dict) -> dict:
    """현재 질문에 필요한 임상 context 를 골라 State 에 쓴다(명세 27절)."""
    from ...llm import build_llm  # 지연 import

    return _prioritize(state, build_llm())


def _replace(items: list[Any]) -> Any:
    """선택 결과 리스트를 **교체**로 표시한다.

    state.py 의 `related_diagnoses` 등은 `merge_records` 누적 reducer 를 쓴다.
    누적은 RAG 근거처럼 여러 경로가 합류하는 필드에는 맞지만, 이 노드의 출력에는
    맞지 않는다. 이 노드는 **매 turn 현재 질문 기준으로 선택을 다시 계산**하기
    때문이다. 누적하면
      - 지난 turn 의 다른 증상 때문에 고른 진단서가 계속 남고,
      - 체중 충돌 기록이 turn 마다 쌓여 PDF 에 서로 모순된 항목이 실리며,
      - 같은 thread 에서 pet 이 바뀌면 다른 아이의 기록이 섞인다.
    그래서 `Replace` 로 감싸 통째로 갈아끼운다.

    state.py 가 아직 없거나 `Replace` 가 없으면 평범한 list 로 돌려준다
    (동작은 하되 누적된다) — import 실패로 그래프가 죽지 않게 한다.
    """
    try:
        from ..state import Replace  # 지연 import — 모듈 최상단 결합을 피한다.
    except (ImportError, AttributeError):  # pragma: no cover
        logger.debug("state.Replace 를 찾지 못해 누적 reducer 를 그대로 사용합니다.")
        return items
    return Replace(items)


def _prioritize(state: dict, llm: Any | None) -> dict:
    context = build_clinical_context(state, llm=llm)
    logger.info(
        "Clinical Context — 증상 %d개, 진단서 %d건, 일기 %d건, 충돌 %d건",
        len(context.current_observation.get("symptoms") or []),
        len(context.related_diagnoses),
        len(context.supporting_daily_entries),
        len(context.context_conflicts),
    )
    return {
        "current_observation": context.current_observation,
        "priority_pet_context": context.priority_pet_context,
        "related_diagnoses": _replace(context.related_diagnoses),
        "supporting_daily_entries": _replace(context.supporting_daily_entries),
        "context_conflicts": _replace([c.model_dump() for c in context.context_conflicts]),
        "context_provenance": _replace([p.model_dump() for p in context.context_provenance]),
    }
