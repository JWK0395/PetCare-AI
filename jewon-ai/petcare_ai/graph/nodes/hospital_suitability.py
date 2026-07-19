"""Hospital Suitability Agent (명세 35절) — 특징 추출은 LLM, 점수는 Python.

## 역할 분담이 이 파일의 전부다

- **LLM**: 병원 페이지 텍스트에서 *사실*만 뽑는다(이름·전화·주소·응급/24시 언급·
  진료 분야). `prompts.HOSPITAL_FEATURE_PROMPT` 가 그 계약이다. 순위나 점수는
  절대 만들지 않는다.
- **Python**: 뽑힌 사실에 `config.HospitalScoreWeights` 가중치를 곱해 score 를
  계산하고 등급을 매긴다. 가중치를 코드에 하드코딩하지 않는 이유는 명세 35절이
  "점수는 config 로 분리한다" 를 요구하기 때문이며, 실무적으로도 병원 후보 품질은
  튜닝 대상이라 코드 수정 없이 조정할 수 있어야 한다.

LLM 이 없으면(키 없음) 규칙 파싱 결과만으로 그대로 동작한다. LLM 은 파싱이 놓친
빈 필드를 **채우기만** 하고, 이미 정규식으로 찾은 값을 덮어쓰지 않는다. 정규식은
텍스트에 실제로 있는 문자열만 반환하지만 LLM 은 형식을 그럴듯하게 지어낼 수 있다.

## 안전 규칙 (명세 34/35/40절)

- 모든 결과의 `verification_required` 에 "방문 전에 전화로 현재 진료 및 응급 접수
  가능 여부를 확인하세요." 를 반드시 넣는다.
- `availability` 는 언제나 "전화 확인 필요" 로 되돌린다. LLM 이 무엇을 반환하든
  실시간 상태를 확정하지 않는다.
- **기존 진료 병원은 가산점만.** 정렬을 강제하지 않는다. 지금 응급 진료가 불가능할
  수 있기 때문이다.
- 연락 수단이 전혀 없는 후보는 점수가 높아도 `low_information` 으로 낮춘다.
  필수조건(연락수단 존재)을 못 채운 병원을 "추천" 이라고 부를 수 없다.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

from pydantic import BaseModel, Field

from ...config import HospitalScoreWeights, Settings, get_settings
from ...llm import safe_structured_invoke
from ...schemas import HospitalCandidate, HospitalSuitabilityResult
from ..prompts import HOSPITAL_FEATURE_PROMPT, HOSPITAL_VERIFICATION_NOTICE, wrap_untrusted_block
from .hospital_search import (
    AVAILABILITY_UNCONFIRMED,
    detect_specialty_mentions,
    extract_email,
    extract_phone,
    looks_like_animal_hospital,
)

logger = logging.getLogger(__name__)

__all__ = [
    "HospitalFeatureExtraction",
    "MAX_EVALUATED_CANDIDATES",
    "MAX_LLM_EXTRACTIONS",
    "extract_features_with_llm",
    "match_specialties",
    "is_previous_hospital",
    "score_hospital",
    "grade_suitability",
    "evaluate_hospital",
    "evaluate_hospitals",
    "make_hospital_suitability_node",
    "hospital_suitability_node",
    "prepare_call_hospital_action_node",
]


#: 평가할 후보 상한. 사용자에게 20개를 보여줄 일은 없고, 응급 상황에서 목록이 길면
#: 오히려 판단이 늦어진다.
MAX_EVALUATED_CANDIDATES = 5

#: LLM 특징 추출을 시도할 후보 수. 상위 후보만 보강한다(응급 상황의 지연 최소화).
MAX_LLM_EXTRACTIONS = 5


class HospitalFeatureExtraction(BaseModel):
    """LLM structured output 스키마 — **사실 추출 전용**.

    score·suitability·순위 필드를 일부러 넣지 않았다. 스키마에 없으면 LLM 이 점수를
    만들 방법 자체가 없다. 명세 35절의 "최종 score 는 Python 이 계산한다" 를
    프롬프트 문구가 아니라 타입으로 강제하는 편이 안전하다.
    """

    name: str = ""
    address: str = ""
    phone: str = ""
    website: str = ""
    email: str = ""
    emergency_mentioned: bool = False
    open_24h_mentioned: bool = False
    specialty_mentions: list[str] = Field(default_factory=list)


def _compact(text: Any) -> str:
    """공백 제거 + 소문자."""
    return "".join(str(text or "").split()).lower()


def _as_candidate(record: dict[str, Any] | HospitalCandidate) -> HospitalCandidate:
    """dict 를 `HospitalCandidate` 로 되돌린다(여분 키는 무시된다)."""
    if isinstance(record, HospitalCandidate):
        return record
    return HospitalCandidate.model_validate(record)


# ---------------------------------------------------------------------------
# 1) LLM 특징 추출 — 빈 값 보강만
# ---------------------------------------------------------------------------
def extract_features_with_llm(
    candidate: HospitalCandidate,
    snippet: str,
    llm: Any | None,
) -> HospitalCandidate:
    """병원 페이지 텍스트에서 특징을 보강한다. LLM 이 없으면 원본 그대로.

    보강 규칙(어기면 안 되는 순서다):

    1. 전화·이메일 같은 **연락처는 LLM 값을 그대로 믿지 않는다.** LLM 이 준 문자열도
       원문 텍스트에 실제로 존재하는지 정규식으로 다시 확인하고, 확인되면 그때만
       채운다. 잘못된 번호로 응급 전화를 걸게 하는 것이 최악의 결과다.
    2. 이미 값이 있는 필드는 덮어쓰지 않는다.
    3. `emergency_mentioned` / `open_24h_mentioned` 는 **올리는 방향으로만** 반영한다.
       규칙이 찾은 '응급 언급' 을 LLM 이 지우지 못하게 한다.
    4. `availability` 는 어떤 경우에도 "전화 확인 필요" 다.
    """
    if llm is None or not str(snippet or "").strip():
        return candidate

    messages = [
        {"role": "system", "content": HOSPITAL_FEATURE_PROMPT},
        {
            "role": "user",
            "content": wrap_untrusted_block("병원 검색 결과", snippet, max_chars=2500),
        },
    ]
    extracted = safe_structured_invoke(
        llm, messages, HospitalFeatureExtraction, HospitalFeatureExtraction()
    )

    data = candidate.model_dump()

    if not data.get("name") and extracted.name.strip():
        data["name"] = extracted.name.strip()
    if not data.get("address") and extracted.address.strip():
        # 주소는 원문에 그 표현이 있는지 확인할 수 없어도, 잘못돼도 전화보다 피해가
        # 작다. 다만 지나치게 짧은 값은 버린다(예: "서울").
        if len(extracted.address.strip()) >= 6:
            data["address"] = extracted.address.strip()

    if not data.get("phone") and extracted.phone.strip():
        verified = extract_phone(snippet)
        llm_digits = "".join(ch for ch in extracted.phone if ch.isdigit())
        if verified and llm_digits and "".join(ch for ch in verified if ch.isdigit()) == llm_digits:
            data["phone"] = verified
        elif verified:
            logger.info("LLM 전화번호가 원문과 달라 정규식 결과를 사용합니다.")
            data["phone"] = verified
        else:
            logger.info("LLM 이 준 전화번호를 원문에서 확인하지 못해 버립니다.")

    if not data.get("email") and extracted.email.strip():
        verified_email = extract_email(snippet)
        if verified_email:
            data["email"] = verified_email

    if not data.get("website") and extracted.website.strip():
        data["website"] = extracted.website.strip()

    data["emergency_mentioned"] = bool(data.get("emergency_mentioned")) or bool(
        extracted.emergency_mentioned
    )
    data["open_24h_mentioned"] = bool(data.get("open_24h_mentioned")) or bool(
        extracted.open_24h_mentioned
    )

    merged_specialties = list(data.get("specialty_mentions") or [])
    for name in extracted.specialty_mentions:
        text = str(name).strip()
        if text and text not in merged_specialties:
            merged_specialties.append(text)
    data["specialty_mentions"] = merged_specialties

    data["availability"] = AVAILABILITY_UNCONFIRMED  # 실시간 상태는 확정하지 않는다.
    return HospitalCandidate.model_validate(data)


# ---------------------------------------------------------------------------
# 2) 매칭 판정
# ---------------------------------------------------------------------------
def match_specialties(
    candidate: HospitalCandidate,
    specialty_keywords: list[str],
    extra_text: str = "",
) -> list[str]:
    """요구된 진료 분야 중 이 병원이 **언급한** 것들을 고른다.

    '언급' 이상을 주장하지 않는다. 홈페이지에 심장 진료가 적혀 있다고 오늘 심장
    전문의가 있는 것은 아니다. 그래서 이 값은 가산점에만 쓰고 답변 문구에서는
    "안내되어 있습니다" 수준으로만 표현한다.
    """
    if not specialty_keywords:
        return []
    mentioned = set(candidate.specialty_mentions) | set(
        detect_specialty_mentions(f"{candidate.name} {extra_text}")
    )
    return [name for name in specialty_keywords if name in mentioned]


def is_previous_hospital(candidate: HospitalCandidate, previous_names: list[str]) -> bool:
    """기존 진료 병원인지 이름으로 판정한다(부분 일치).

    진단서의 '행복동물병원' 과 검색 결과의 '행복동물병원 강남점' 을 같은 곳으로 볼지가
    문제인데, **가산점 10점만 주는 판정**이라 다소 느슨해도 위험이 작다. 대신 이
    판정이 순위를 확정하지는 않는다(명세 35절).
    """
    name_key = _compact(candidate.name)
    if not name_key:
        return False
    for previous in previous_names or []:
        previous_key = _compact(previous)
        if not previous_key:
            continue
        if previous_key in name_key or name_key in previous_key:
            return True
    return False


# ---------------------------------------------------------------------------
# 3) 점수 계산 — 전부 Python (명세 35절)
# ---------------------------------------------------------------------------
def score_hospital(
    candidate: HospitalCandidate,
    *,
    is_animal_hospital: bool,
    specialty_matches: list[str],
    previous_hospital: bool,
    weights: HospitalScoreWeights,
) -> tuple[int, list[str], list[str]]:
    """가중치를 적용해 (score, 충족 이유, 미충족 우대조건) 을 계산한다.

    가중치 값은 전부 `HospitalScoreWeights` 에서 온다 — 이 함수 안에 숫자 리터럴이
    없어야 한다(명세 35절 "점수는 config 로 분리").
    """
    score = 0
    matched: list[str] = []
    unmatched: list[str] = []

    if is_animal_hospital:
        score += weights.is_animal_hospital
        matched.append("동물병원으로 확인됨")
    else:
        unmatched.append("동물병원 여부가 확인되지 않음")

    if candidate.phone:
        score += weights.has_phone
        matched.append(f"전화번호 확인: {candidate.phone}")
    else:
        unmatched.append("전화번호가 검색 결과에 없음")

    if candidate.emergency_mentioned:
        score += weights.emergency_mentioned
        matched.append("응급 진료 안내가 페이지에 언급됨")
    else:
        unmatched.append("응급 진료 안내 언급 없음")

    if candidate.open_24h_mentioned:
        score += weights.open_24h_mentioned
        matched.append("24시간 또는 야간 진료 안내가 페이지에 언급됨")
    else:
        unmatched.append("24시간·야간 진료 안내 언급 없음")

    if specialty_matches:
        score += weights.specialty_matches
        matched.append(f"관련 진료 안내 언급: {', '.join(specialty_matches)}")

    if previous_hospital:
        # 가산점만. 정렬을 강제하지 않는다(명세 35절).
        score += weights.is_previous_hospital
        matched.append("기존 진료 기록이 있는 병원 (가산점)")

    return score, matched, unmatched


def grade_suitability(
    score: int,
    candidate: HospitalCandidate,
    weights: HospitalScoreWeights,
) -> str:
    """점수 경계값으로 등급을 정한다(경계값도 config 에서 온다).

    한 가지 예외를 둔다. **연락 수단이 하나도 없으면 점수와 무관하게
    `low_information`** 이다. 명세 33절 필수조건이 '전화번호 또는 연락수단 존재'
    이므로, 연락할 방법이 없는 곳을 "추천" 으로 부르면 필수조건을 스스로 어기는
    셈이 된다.
    """
    if not (candidate.phone or candidate.email or candidate.website):
        return "low_information"
    if score >= weights.recommended_min:
        return "recommended"
    if score >= weights.possible_min:
        return "possible"
    return "low_information"


def evaluate_hospital(
    record: dict[str, Any] | HospitalCandidate,
    *,
    specialty_keywords: list[str] | None = None,
    previous_names: list[str] | None = None,
    weights: HospitalScoreWeights | None = None,
    llm: Any | None = None,
) -> HospitalSuitabilityResult:
    """후보 1건을 평가한다(순수 함수에 가까움 — llm 만 외부 의존).

    `verification_required` 에는 명세 35절이 요구한 확인 문구가 **항상** 들어간다.
    정보가 부족한 항목이 있으면 무엇을 더 확인해야 하는지도 함께 적는다.
    """
    resolved_weights = weights or get_settings().hospital_score
    raw = record if isinstance(record, dict) else {}
    # `source_query` 를 **일부러 제외한다.** 검색어에는 "24시 응급 동물병원" 같은
    # 단어가 들어 있어서, 그대로 넣으면 블로그·쇼핑 페이지가 '동물병원 언급' 으로
    # 판정되어 가산점을 받는다(실제로 그렇게 새어 들어갔다). 판정 근거는 병원
    # 페이지 자신이 말한 내용이어야 한다.
    snippet = " ".join(
        str(raw.get(key) or "") for key in ("search_title", "search_snippet")
    ).strip()

    candidate = _as_candidate(record)
    candidate = extract_features_with_llm(candidate, snippet, llm)

    animal_hospital = bool(raw.get("is_animal_hospital")) or looks_like_animal_hospital(
        candidate.name, snippet, candidate.source_url
    )
    matches = match_specialties(candidate, list(specialty_keywords or []), snippet)
    previous = is_previous_hospital(candidate, list(previous_names or []))

    score, matched, unmatched = score_hospital(
        candidate,
        is_animal_hospital=animal_hospital,
        specialty_matches=matches,
        previous_hospital=previous,
        weights=resolved_weights,
    )
    suitability = grade_suitability(score, candidate, resolved_weights)

    verification: list[str] = [HOSPITAL_VERIFICATION_NOTICE]
    if not candidate.phone:
        verification.append("전화번호가 확인되지 않아 병원 홈페이지에서 연락처 확인이 필요합니다.")
    if not candidate.address:
        verification.append("주소가 확인되지 않아 위치 확인이 필요합니다.")
    if candidate.emergency_mentioned or candidate.open_24h_mentioned:
        verification.append(
            "24시간·응급 진료 안내는 페이지에 적힌 내용이며, 현재 접수 가능 여부는 다를 수 있습니다."
        )
    if previous:
        verification.append(
            "기존 진료 병원이라도 현재 응급 접수가 가능한지는 전화로 확인해야 합니다."
        )
    if not matches and specialty_keywords:
        unmatched.append(f"요청한 진료 분야 언급 없음: {', '.join(specialty_keywords)}")

    return HospitalSuitabilityResult(
        hospital=candidate,
        score=score,
        suitability=suitability,  # type: ignore[arg-type]
        matched_reasons=matched,
        unmatched_preferences=unmatched,
        verification_required=verification,
    )


def evaluate_hospitals(
    records: list[dict[str, Any]],
    *,
    specialty_keywords: list[str] | None = None,
    previous_names: list[str] | None = None,
    settings: Settings | None = None,
    llm: Any | None = None,
    limit: int = MAX_EVALUATED_CANDIDATES,
) -> list[HospitalSuitabilityResult]:
    """후보 목록을 평가하고 점수순으로 정렬한다.

    정렬 기준은 (score 내림차순 → 전화번호 있음 → 이름). **기존 진료 병원이라는
    이유만으로 앞으로 끌어올리지 않는다**(명세 35절). 동점이면 전화번호가 있는 쪽을
    앞에 두는데, 응급 상황에서 바로 연락 가능한 정보가 더 가치 있기 때문이다.

    LLM 특징 추출은 앞쪽 후보 몇 건에만 적용한다(`MAX_LLM_EXTRACTIONS`). 응급
    경로에서 후보마다 LLM 을 호출하면 안내가 그만큼 늦어진다.
    """
    weights = (settings or get_settings()).hospital_score
    results: list[HospitalSuitabilityResult] = []

    for index, record in enumerate(records or []):
        if not isinstance(record, (dict, HospitalCandidate)):
            logger.warning("병원 후보 형식이 올바르지 않아 제외합니다: %r", record)
            continue
        try:
            results.append(
                evaluate_hospital(
                    record,
                    specialty_keywords=specialty_keywords,
                    previous_names=previous_names,
                    weights=weights,
                    llm=llm if index < MAX_LLM_EXTRACTIONS else None,
                )
            )
        except Exception as exc:  # 한 후보가 깨졌다고 전체 안내를 포기하지 않는다.
            logger.warning("병원 후보 평가에 실패해 제외합니다: %s", exc)

    results.sort(key=lambda item: (-item.score, item.hospital.phone is None, item.hospital.name))
    return results[: max(1, int(limit))] if results else []


# ---------------------------------------------------------------------------
# 4) Node
# ---------------------------------------------------------------------------
def make_hospital_suitability_node(
    llm: Any | None = None,
    settings: Settings | None = None,
) -> Callable[[dict], dict]:
    """LLM·설정을 주입할 수 있는 node factory (테스트가 mock 을 넣는다)."""

    def _node(state: dict) -> dict:
        return hospital_suitability_node(state, llm=llm, settings=settings)

    return _node


def hospital_suitability_node(
    state: dict,
    llm: Any | None = None,
    settings: Settings | None = None,
) -> dict:
    """Hospital Suitability Agent (명세 35절).

    `hospital_results` 는 reducer 없이 **통째로 교체**한다(state.py 주석 참고).
    점수순 정렬 결과가 목록 전체의 의미이므로 누적하면 순서가 깨진다.

    LLM 인자를 주지 않으면 `build_llm()` 으로 만들어 보고, 키가 없으면 None 이 되어
    규칙 파싱 결과만으로 평가한다 — 이것이 정상 경로다.
    """
    records = list(state.get("raw_hospital_results") or [])
    if not records:
        logger.info("평가할 병원 후보가 없습니다.")
        return {"hospital_results": []}

    if llm is None:
        from ...llm import build_llm  # 지연 import: langchain provider 미설치 대응

        llm = build_llm(settings)

    requirements = state.get("hospital_requirements") or {}
    specialty_keywords = list(requirements.get("specialty_keywords") or []) if isinstance(
        requirements, dict
    ) else []
    previous_names = list(requirements.get("previous_hospital_names") or []) if isinstance(
        requirements, dict
    ) else []

    results = evaluate_hospitals(
        records,
        specialty_keywords=specialty_keywords,
        previous_names=previous_names,
        settings=settings,
        llm=llm,
    )

    update: dict[str, Any] = {"hospital_results": [item.model_dump() for item in results]}

    top = results[0] if results else None
    if top is not None:
        update["selected_hospital"] = top.model_dump()
        if top.hospital.phone:
            # 응급 여부와 무관하게 '전화 확인' 이 권장 행동이므로 action 을 남긴다.
            update["ui_actions"] = [
                {
                    "type": "CALL_HOSPITAL",
                    "hospital_name": top.hospital.name,
                    "phone": top.hospital.phone,
                    "availability": AVAILABILITY_UNCONFIRMED,
                    "notice": HOSPITAL_VERIFICATION_NOTICE,
                }
            ]

    logger.info(
        "병원 적합도 평가 완료 — %d건 (최고점 %s)",
        len(results),
        results[0].score if results else "-",
    )
    return update


def prepare_call_hospital_action_node(state: dict) -> dict:
    """`Prepare CALL_HOSPITAL action` node (명세 32절 즉시 위급 경로).

    적합도 평가를 기다리지 않는다. 명세 43절 '즉시 위급' 기대는 **정보가 부족해도
    전화 action 이 존재하는 것**이다. 병원 검색이 실패했거나 아직 끝나지 않았다면
    번호 없는 안내형 action 이라도 남겨 사용자가 다음 행동을 알 수 있게 한다.

    번호를 아는 경우와 모르는 경우의 action 내용이 다르므로, `merge_ui_actions`
    reducer 가 두 action 을 모두 유지할 수 있다. 중복처럼 보여도 지우지 않는다 —
    응급 상황에서 버튼이 하나 더 있는 것이 하나도 없는 것보다 낫다.
    """
    results = state.get("hospital_results") or []
    top = results[0] if results and isinstance(results[0], dict) else None
    phone = ((top or {}).get("hospital") or {}).get("phone") if top else None

    if phone:
        action = {
            "type": "CALL_HOSPITAL",
            "hospital_name": ((top or {}).get("hospital") or {}).get("name") or "",
            "phone": phone,
            "availability": AVAILABILITY_UNCONFIRMED,
            "notice": HOSPITAL_VERIFICATION_NOTICE,
        }
    else:
        action = {
            "type": "CALL_HOSPITAL",
            "hospital_name": "",
            "phone": None,
            "availability": AVAILABILITY_UNCONFIRMED,
            "notice": (
                "지금 바로 평소 다니던 동물병원이나 가까운 24시 동물병원에 전화해 "
                "현재 진료 및 응급 접수가 가능한지 확인해 주세요."
            ),
        }

    return {"ui_actions": [action]}
