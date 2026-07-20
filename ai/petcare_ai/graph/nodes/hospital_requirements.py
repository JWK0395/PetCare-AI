"""Hospital Requirement Builder (명세 33절) — 검색 전에 '어떤 병원을 찾을지' 정한다.

이 node 는 **LLM 을 쓰지 않는 일반 Python node** 다(명세 19/23절). 하는 일이
"보호자 발화·PET DB·진단서에서 조건 문자열과 검색어를 조립"하는 결정론적 매핑이라
LLM 을 끼워 넣을 이유가 없다. LLM 이 개입하면 오히려 없는 진료과를 지어낼 위험만
생긴다.

## 설계에서 가장 중요한 두 가지

1. **필수조건과 우대조건을 엄격히 나눈다.**
   명세 33절은 "특정 진료과나 장비가 반드시 필요하다고 단정하지 않는다" 를 명시한다.
   심장 질환 이력이 있다고 해서 "심장 전문 진료과가 있는 병원이어야 한다" 고 하면,
   응급 상황에서 갈 수 있는 병원을 스스로 지워버린다. 그래서 진료과는 **항상
   `preferred` 와 `specialty_keywords` 로만** 들어간다. `required` 는 명세가 정한
   세 가지(동물병원 / 지역 확인 / 연락수단)로 고정한다.

2. **입력 우선순위(명세 20/33절)를 그대로 지킨다.**
   현재 증상 > PET DB > 진단서 DB > 일기장 DB > RAG·검증된 웹 지식.
   앞 순위에서 이미 잡힌 진료과 키워드가 있으면 뒤 순위는 그 목록을 **덧붙이기만**
   하고 덮어쓰지 않는다. 오래된 진단서가 현재 증상보다 앞에 오면 안 된다.

`region_name` 이 없으면 검색어를 만들지 않는다. 지역 없는 "24시 응급 동물병원"
검색은 전국 광고 페이지를 긁어오며, 그 결과로 병원을 추천하면 보호자가 갈 수 없는
곳에 전화하게 된다. 지역 요청은 `hospital_search.request_location_node` 가 담당한다.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from ...schemas import HospitalRequirements

logger = logging.getLogger(__name__)

__all__ = [
    "REQUIRED_CONDITIONS",
    "PREFERRED_24H",
    "PREFERRED_EMERGENCY",
    "PREFERRED_INPATIENT",
    "PREFERRED_PREVIOUS_HOSPITAL",
    "SpecialtyTerm",
    "SPECIALTY_TAXONOMY",
    "MAX_SEARCH_QUERIES",
    "specialty_label",
    "collect_specialty_keywords",
    "collect_previous_hospital_names",
    "build_search_queries",
    "build_hospital_requirements",
    "hospital_requirements_node",
]


# ---------------------------------------------------------------------------
# 조건 문구 (명세 33절 예시 그대로)
# ---------------------------------------------------------------------------
#: 필수조건. **여기에 진료과·장비를 추가하지 않는다.** 명세 33절 금지 사항이다.
REQUIRED_CONDITIONS: tuple[str, ...] = (
    "동물병원",
    "지역 확인 가능",
    "전화번호 또는 연락수단 존재",
)

PREFERRED_24H = "24시간 또는 야간 진료 안내"
PREFERRED_EMERGENCY = "응급 진료 안내"
PREFERRED_INPATIENT = "입원 안내"
PREFERRED_PREVIOUS_HOSPITAL = "기존 진료 병원"

#: 검색어 상한. Tavily 호출 비용도 있지만, 더 큰 이유는 검색어가 많아질수록
#: 지역과 무관한 결과가 섞여 후보 품질이 떨어지기 때문이다.
MAX_SEARCH_QUERIES = 5

#: 진료과 우대조건 문구를 만들 때 쓰는 접미사. "필요" 가 아니라 "있으면 좋음" 이라는
#: 사실이 문자열 자체에 드러나야 한다(그대로 사용자에게 노출될 수 있다).
_PREFERRED_SPECIALTY_SUFFIX = " 관련 진료 안내(우대)"


# ---------------------------------------------------------------------------
# 진료과 사전
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SpecialtyTerm:
    """진료 분야 1종.

    `patterns` 는 **공백을 제거한 형태**로 적는다. 보호자는 "숨을 잘 못 쉬어요",
    진단서는 "이첨판 폐쇄부전증" 처럼 띄어쓰기가 제각각이라, 원문 그대로 매칭하면
    거의 걸리지 않는다.

    `query_term` 은 Tavily 검색어에 넣을 짧은 한국어다. 병원 홈페이지가 실제로 쓰는
    표현("심장 진료", "정형외과")이어야 검색이 걸린다.
    """

    name: str
    query_term: str
    patterns: tuple[str, ...]


#: 증상·기저질환 → 진료 분야. 어디까지나 **검색 힌트**이며 진단이 아니다.
SPECIALTY_TAXONOMY: tuple[SpecialtyTerm, ...] = (
    SpecialtyTerm(
        "심장",
        "심장 진료",
        (
            "심장", "심부전", "심잡음", "이첨판", "판막", "심비대", "심낭",
            "부정맥", "청색증", "heart", "cardiac", "cardio",
        ),
    ),
    SpecialtyTerm(
        "호흡기",
        "호흡기 진료",
        (
            "호흡", "숨", "기침", "폐수종", "폐렴", "기관지", "기관허탈", "천식",
            "헐떡", "그렁", "respiratory",
        ),
    ),
    SpecialtyTerm(
        "신경",
        "신경과 진료",
        (
            "경련", "발작", "마비", "뇌", "신경", "간질", "디스크", "사지마비",
            "머리기울", "seizure", "neuro",
        ),
    ),
    SpecialtyTerm(
        "내과",
        "내과 진료",
        (
            "구토", "설사", "췌장", "위염", "장염", "간수치", "간질환", "담낭",
            "이물", "혈변", "식욕부진", "당뇨", "갑상선", "복수",
        ),
    ),
    SpecialtyTerm(
        "비뇨기",
        "비뇨기 진료",
        (
            "소변", "배뇨", "방광", "요도", "신장", "신부전", "결석", "혈뇨",
            "요독", "urinary", "kidney",
        ),
    ),
    SpecialtyTerm(
        "안과",
        "안과 진료",
        ("눈", "각막", "결막", "안구", "백내장", "녹내장", "눈곱", "실명", "안검"),
    ),
    SpecialtyTerm(
        "정형외과",
        "정형외과 진료",
        (
            "절뚝", "파행", "골절", "슬개골", "관절", "탈구", "인대", "십자인대",
            "다리", "보행", "ortho",
        ),
    ),
    SpecialtyTerm(
        "피부",
        "피부과 진료",
        ("피부", "가려", "긁", "발진", "탈모", "농피", "귀", "외이염", "아토피"),
    ),
    SpecialtyTerm(
        "종양",
        "종양 진료",
        ("종양", "암", "혹", "멍울", "덩어리", "림프종", "전이", "oncol"),
    ),
    SpecialtyTerm(
        "치과",
        "치과 진료",
        ("치아", "이빨", "잇몸", "치석", "구취", "발치", "구내염"),
    ),
    SpecialtyTerm(
        "중독",
        "중독 응급 진료",
        (
            "중독", "삼켰", "먹었", "초콜릿", "포도", "양파", "자일리톨", "쥐약",
            "살충제", "세제", "이물질", "toxic", "poison",
        ),
    ),
    SpecialtyTerm(
        "외과",
        "외과 진료",
        ("수술", "출혈", "상처", "찢어", "물렸", "교통사고", "외상", "봉합"),
    ),
)

#: 증상·대화에서 '입원 안내' 우대조건을 켜는 신호.
_INPATIENT_SIGNALS: tuple[str, ...] = (
    "입원", "중환자", "산소", "수액", "집중치료", "수술", "장기간", "24시간관찰",
)

#: 진단서·PET DB 에서 병원 이름이 들어 있는 필드 후보.
_HOSPITAL_NAME_KEYS: tuple[str, ...] = (
    "hospital", "hospital_name", "clinic", "clinic_name", "병원", "병원명",
)

#: 증상 텍스트를 모을 때 훑는 레코드 필드.
_DIAGNOSIS_TEXT_KEYS: tuple[str, ...] = (
    "diagnosis", "diagnosis_name", "content", "summary", "treatment", "note", "notes",
)
_DAILY_TEXT_KEYS: tuple[str, ...] = ("raw_text", "content", "summary", "notes", "note")


def _compact(text: Any) -> str:
    """공백을 제거한 매칭용 소문자 문자열.

    `clinical_context_priority._compact` 와 같은 목적이지만, 그 모듈을 import 하면
    무거운 증상 사전까지 딸려 오므로 필요한 최소 동작만 여기에 둔다.
    """
    return "".join(str(text or "").split()).lower()


def specialty_label(name: str) -> str:
    """진료과 이름을 **우대조건 문구**로 바꾼다.

    문자열에 '우대' 를 박아 두는 이유: 이 값이 그대로 PDF·답변·로그로 흘러가는데,
    "심장" 만 적혀 있으면 읽는 사람이 '심장 진료과가 필수' 로 오해할 수 있다.
    """
    return f"{name}{_PREFERRED_SPECIALTY_SUFFIX}"


def _record_text(record: dict[str, Any], keys: tuple[str, ...]) -> str:
    """레코드에서 텍스트 필드만 이어 붙인다(요약·재파싱이 아니다)."""
    if not isinstance(record, dict):
        return ""
    return " ".join(str(record.get(key) or "") for key in keys)


def collect_specialty_keywords(
    current_observation: dict[str, Any] | None,
    pet_profile: dict[str, Any] | None,
    diagnoses: list[dict[str, Any]] | None,
    daily_entries: list[dict[str, Any]] | None,
    evidence_texts: list[str] | None = None,
) -> list[str]:
    """관련 진료 분야를 우선순위 순서로 모은다(명세 33절 입력 우선순위).

    앞 순위에서 나온 분야가 리스트 앞에 오고, 뒤 순위는 **없는 것만 덧붙인다.**
    검색어를 만들 때 앞에서부터 자르므로, 순서가 곧 우선순위다.

    반환값은 진료과 '이름'(예: "심장") 이며, 우대조건 문구로 바꾸는 것은
    `specialty_label()` 의 몫이다. 여기서 나온 값은 어떤 경우에도 필수조건이
    되지 않는다.
    """
    layers: list[str] = []

    observation = current_observation or {}
    # 1순위: 현재 사용자 입력 — 원문과 규칙 추출 결과를 모두 본다.
    layers.append(
        " ".join(
            [
                str(observation.get("raw_text") or ""),
                " ".join(str(item) for item in (observation.get("symptoms") or [])),
                " ".join(
                    str(value) for value in (observation.get("collected_information") or {}).values()
                ),
            ]
        )
    )
    # 2순위: PET DB 기저질환·복용약
    profile = pet_profile or {}
    layers.append(
        " ".join(
            str(profile.get(key) or "")
            for key in ("diseases", "medications", "supplement", "allergies", "note")
        )
    )
    # 3순위: 진단서
    layers.append(" ".join(_record_text(item, _DIAGNOSIS_TEXT_KEYS) for item in (diagnoses or [])))
    # 4순위: 일기장(보조)
    layers.append(" ".join(_record_text(item, _DAILY_TEXT_KEYS) for item in (daily_entries or [])))
    # 5순위: RAG·검증된 웹 근거(보조) — 근거 본문에 병명이 있으면 검색어 힌트로만 쓴다.
    layers.append(" ".join(str(text or "") for text in (evidence_texts or [])))

    found: list[str] = []
    for layer in layers:
        compact = _compact(layer)
        if not compact:
            continue
        for term in SPECIALTY_TAXONOMY:
            if term.name in found:
                continue
            if any(pattern in compact for pattern in term.patterns):
                found.append(term.name)
    return found


def collect_previous_hospital_names(
    diagnoses: list[dict[str, Any]] | None,
    pet_profile: dict[str, Any] | None = None,
) -> list[str]:
    """기존 진료 병원 이름을 모은다(최신 기록이 앞).

    이 값은 **가산점 대상일 뿐**이다(명세 35절). 여기서 1순위를 정하지 않는다.
    지금 응급 진료가 가능한지는 아무도 모르기 때문이다.
    """
    names: list[str] = []
    seen: set[str] = set()

    def _add(raw: Any) -> None:
        text = str(raw or "").strip()
        key = _compact(text)
        if not text or key in seen:
            return
        seen.add(key)
        names.append(text)

    # 진단서는 보통 오래된 것부터 저장되므로 뒤에서부터(최신부터) 읽는다.
    for record in reversed(list(diagnoses or [])):
        if not isinstance(record, dict):
            continue
        for key in _HOSPITAL_NAME_KEYS:
            if record.get(key):
                _add(record[key])
                break

    for key in _HOSPITAL_NAME_KEYS:
        if (pet_profile or {}).get(key):
            _add((pet_profile or {})[key])
            break

    return names


def _needs_inpatient(texts: str) -> bool:
    """입원 안내를 우대조건에 넣을지 판단한다."""
    compact = _compact(texts)
    return any(signal in compact for signal in _INPATIENT_SIGNALS)


def build_search_queries(
    region_name: str | None,
    specialties: list[str],
    *,
    emergency: bool,
    previous_hospital_names: list[str] | None = None,
) -> list[str]:
    """Tavily 병원 검색어를 만든다(명세 34절 예시 형태).

    지역이 없으면 **빈 목록**을 돌려준다. 전국 검색 결과로 병원을 안내하는 것은
    안내하지 않는 것보다 나쁘다.

    응급이면 '24시 응급' 검색어를 맨 앞에 둔다. 검색 결과는 앞 검색어일수록 많이
    남으므로(중복 제거가 먼저 들어온 URL 을 유지한다) 순서 자체가 우선순위다.
    """
    region = str(region_name or "").strip()
    if not region:
        logger.info("region_name 이 없어 병원 검색어를 만들지 않습니다(지역 요청 필요).")
        return []

    queries: list[str] = []
    if emergency:
        queries.append(f"{region} 24시 응급 동물병원 전화번호")
        queries.append(f"{region} 야간 동물병원 응급 진료")
    else:
        queries.append(f"{region} 동물병원 전화번호")
        queries.append(f"{region} 야간 동물병원 진료 시간")

    term_by_name = {term.name: term.query_term for term in SPECIALTY_TAXONOMY}
    for name in specialties[:2]:
        query_term = term_by_name.get(name, name)
        prefix = "야간 동물병원" if emergency else "동물병원"
        queries.append(f"{region} {prefix} {query_term}")

    for name in (previous_hospital_names or [])[:1]:
        queries.append(f"{region} {name} 전화번호")

    # 순서를 유지한 중복 제거 후 상한까지만.
    unique: list[str] = []
    seen: set[str] = set()
    for query in queries:
        key = _compact(query)
        if key in seen:
            continue
        seen.add(key)
        unique.append(query)
    return unique[:MAX_SEARCH_QUERIES]


def build_hospital_requirements(state: dict[str, Any]) -> HospitalRequirements:
    """State 에서 병원 검색 조건을 만든다(순수 함수 — 단독 테스트 가능).

    `required` 는 항상 명세 33절의 세 항목으로 고정한다. 증상이 무엇이든 필수조건이
    늘어나지 않는다는 뜻이며, 이것이 "특정 진료과를 필수로 단정하지 않는다" 규칙의
    구현이다.
    """
    observation = state.get("current_observation") or {}
    profile = state.get("priority_pet_context") or state.get("pet_profile") or {}
    diagnoses = state.get("related_diagnoses") or state.get("diagnoses") or []
    daily_entries = state.get("supporting_daily_entries") or state.get("daily_entries") or []

    evidence_texts = [
        str(item.get("text") or "")
        for item in (state.get("merged_evidence") or [])
        if isinstance(item, dict)
    ]

    specialties = collect_specialty_keywords(
        observation, profile, diagnoses, daily_entries, evidence_texts
    )
    previous_names = collect_previous_hospital_names(diagnoses, profile)

    risk = str(state.get("final_risk") or "normal")
    urgency = str(state.get("emergency_urgency") or "none")
    emergency = risk == "emergency" or urgency != "none" or bool(state.get("possible_emergency"))

    preferred: list[str] = []
    if emergency:
        # 응급일 때는 순서를 뒤집는다 — 지금 필요한 것은 '지금 받아주는 곳' 이다.
        preferred.append(PREFERRED_EMERGENCY)
        preferred.append(PREFERRED_24H)
    else:
        preferred.append(PREFERRED_24H)
        preferred.append(PREFERRED_EMERGENCY)

    preferred.extend(specialty_label(name) for name in specialties)

    combined_text = " ".join(
        [
            str(observation.get("raw_text") or ""),
            " ".join(str(item) for item in (observation.get("symptoms") or [])),
            " ".join(str(flag) for flag in (state.get("red_flags") or [])),
        ]
    )
    if emergency or _needs_inpatient(combined_text):
        preferred.append(PREFERRED_INPATIENT)
    if previous_names:
        preferred.append(PREFERRED_PREVIOUS_HOSPITAL)

    requirements = HospitalRequirements(
        required=list(REQUIRED_CONDITIONS),
        preferred=preferred,
        specialty_keywords=specialties,
        previous_hospital_names=previous_names,
        search_queries=build_search_queries(
            state.get("region_name"),
            specialties,
            emergency=emergency,
            previous_hospital_names=previous_names,
        ),
    )
    logger.info(
        "병원 요구사항 생성 — 응급=%s, 진료분야=%s, 기존병원=%d건, 검색어=%d개",
        emergency,
        ",".join(specialties) or "없음",
        len(previous_names),
        len(requirements.search_queries),
    )
    return requirements


def hospital_requirements_node(state: dict) -> dict:
    """Hospital Requirement Builder node (명세 33절).

    LLM 을 쓰지 않는다. 조건 조립은 결정론적이어야 같은 입력에 같은 병원 후보가
    나오고, 그래야 명세 43절 '병원 적합도' 테스트가 값을 단언할 수 있다.
    """
    requirements = build_hospital_requirements(state)
    return {
        "hospital_requirements": requirements.model_dump(),
        "hospital_search_queries": requirements.search_queries,
    }
