"""병원 적합도 평가 테스트 (명세 33·34·35·43절).

명세 43절 '병원 적합도' 는 네 종류의 후보에 대해 **score 와 이유를 assert** 하라고
요구한다.

    1. 일반 동물병원
    2. 24시 응급 동물병원
    3. 기존 진료 병원
    4. 전문진료 정보가 있으나 전화번호가 없는 병원

점수는 `config.HospitalScoreWeights` 에서만 온다(명세 35절 "점수는 config 로
분리"). 그래서 기대값도 숫자 리터럴이 아니라 **가중치 합**으로 적는다. 가중치를
조정해도 테스트가 함께 따라가고, 그러면서도 "어떤 항목이 몇 점을 만드는가" 는
정확히 고정된다.

함께 확인하는 안전 규칙(명세 34·35·47절)
  - `availability` 는 언제나 "전화 확인 필요" — 실시간 진료 가능 여부를 단정하지 않는다.
  - 모든 결과의 `verification_required` 에 전화 확인 안내가 들어간다.
  - 기존 진료 병원은 **가산점만** — 순위를 강제하지 않는다.
  - 연락 수단이 하나도 없으면 점수와 무관하게 `low_information`.
"""

from __future__ import annotations

from typing import Any

import pytest

from petcare_ai.adapters.clinical_data_adapter import FixtureClinicalDataAdapter
from petcare_ai.config import get_settings
from petcare_ai.graph.nodes.hospital_requirements import (
    REQUIRED_CONDITIONS,
    build_hospital_requirements,
    collect_previous_hospital_names,
)
from petcare_ai.graph.nodes.hospital_search import (
    AVAILABILITY_UNCONFIRMED,
    parse_hospital_results,
)
from petcare_ai.graph.nodes.hospital_suitability import (
    evaluate_hospitals,
    grade_suitability,
    hospital_suitability_node,
    is_previous_hospital,
    prepare_call_hospital_action_node,
)
from petcare_ai.graph.prompts import HOSPITAL_VERIFICATION_NOTICE
from petcare_ai.schemas import HospitalCandidate

DOG_PET_ID = 1
REGION = "서울 강남구"

#: 이번 평가에서 요청하는 진료 분야(경련 → 신경 / 기저 심장질환 → 심장).
SPECIALTY_KEYWORDS = ["신경", "심장"]

#: 기존 진료 병원 이름 — 진단서에서 온다(명세 33절).
PREVIOUS_HOSPITALS = ["서울동물메디컬센터"]


# 명세 43절이 지정한 네 종류의 후보(Tavily 원시 결과 형태).
RAW_RESULTS: list[dict[str, Any]] = [
    {
        # (1) 일반 동물병원 — 전화번호는 있으나 응급/24시/전문진료 언급 없음
        "title": "튼튼동물병원 진료 안내",
        "url": "https://tuntunvet.example.com",
        "content": "진료 안내 전화 02-123-4567. 서울 강남구 테헤란로 12.",
        "score": 0.55,
    },
    {
        # (2) 24시 응급 동물병원
        "title": "강남24시동물병원 | 24시간 응급 진료",
        "url": "https://gangnam24vet.example.com/emergency",
        "content": "24시간 응급 진료 안내입니다. 전화 02-987-6543. 서울 강남구 언주로 100.",
        "score": 0.83,
    },
    {
        # (3) 기존 진료 병원 — 진단서에 기록된 병원이며 요청 진료 분야도 언급
        "title": "서울동물메디컬센터 심장 진료 안내",
        "url": "https://seoulamc.example.com",
        "content": "심장 진료 전담팀 운영. 전화 02-555-1234. 서울 강남구 도산대로 45.",
        "score": 0.74,
    },
    {
        # (4) 전문진료 정보는 있으나 전화번호가 없는 병원
        "title": "미래동물의료센터 신경과 안내",
        "url": "https://miraeamc.example.com/neurology",
        "content": "신경과 진료와 MRI 검사를 안내합니다. 상담은 홈페이지 문의 양식을 이용해 주세요.",
        "score": 0.69,
    },
]


@pytest.fixture(autouse=True)
def _no_api_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    """LLM 없이 규칙 파싱만으로 평가되는 경로를 강제한다."""
    for name in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "TAVILY_API_KEY"):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture()
def results() -> dict[str, Any]:
    """네 후보를 평가하고 이름으로 찾을 수 있게 정리한다."""
    parsed = parse_hospital_results(RAW_RESULTS)
    evaluated = evaluate_hospitals(
        parsed,
        specialty_keywords=SPECIALTY_KEYWORDS,
        previous_names=PREVIOUS_HOSPITALS,
        llm=None,
    )
    return {"ordered": evaluated, "by_name": {item.hospital.name: item for item in evaluated}}


# ---------------------------------------------------------------------------
# 파싱 — 없는 값을 만들지 않는다 (명세 34절)
# ---------------------------------------------------------------------------
def test_parsing_extracts_only_values_present_in_text() -> None:
    parsed = {item["name"]: item for item in parse_hospital_results(RAW_RESULTS)}

    general = parsed["튼튼동물병원"]
    assert general["phone"] == "02-123-4567"
    assert general["address"] == "서울 강남구 테헤란로 12"
    assert general["emergency_mentioned"] is False
    assert general["open_24h_mentioned"] is False

    emergency = parsed["강남24시동물병원"]
    assert emergency["emergency_mentioned"] is True
    assert emergency["open_24h_mentioned"] is True

    no_phone = parsed["미래동물의료센터"]
    assert no_phone["phone"] is None, "없는 전화번호를 만들어 내면 안 된다."
    assert no_phone["address"] is None
    assert "신경" in no_phone["specialty_mentions"]

    # 검색 결과만으로 실시간 진료 가능 여부를 확정하지 않는다.
    assert all(item["availability"] == AVAILABILITY_UNCONFIRMED for item in parsed.values())


# ---------------------------------------------------------------------------
# 명세 43절 — 네 후보의 score 와 이유
# ---------------------------------------------------------------------------
def test_general_hospital_score_and_reasons(results: dict[str, Any]) -> None:
    """(1) 일반 동물병원 — 동물병원 + 전화번호만 충족한다."""
    weights = get_settings().hospital_score
    item = results["by_name"]["튼튼동물병원"]

    assert item.score == weights.is_animal_hospital + weights.has_phone
    assert item.suitability == "possible"
    assert "동물병원으로 확인됨" in item.matched_reasons
    assert any("전화번호 확인" in reason for reason in item.matched_reasons)
    assert "응급 진료 안내 언급 없음" in item.unmatched_preferences
    assert "24시간·야간 진료 안내 언급 없음" in item.unmatched_preferences
    assert any("요청한 진료 분야 언급 없음" in item_ for item_ in item.unmatched_preferences)


def test_emergency_24h_hospital_score_and_reasons(results: dict[str, Any]) -> None:
    """(2) 24시 응급 동물병원 — 응급·24시 언급 가산점이 모두 붙어 recommended."""
    weights = get_settings().hospital_score
    item = results["by_name"]["강남24시동물병원"]

    assert item.score == (
        weights.is_animal_hospital
        + weights.has_phone
        + weights.emergency_mentioned
        + weights.open_24h_mentioned
    )
    assert item.suitability == "recommended"
    assert "응급 진료 안내가 페이지에 언급됨" in item.matched_reasons
    assert "24시간 또는 야간 진료 안내가 페이지에 언급됨" in item.matched_reasons
    # '언급' 이상을 주장하지 않는다 — 지금 접수 가능한지는 알 수 없다.
    assert any("현재 접수 가능 여부는 다를 수 있습니다" in note for note in item.verification_required)


def test_previous_hospital_score_and_reasons(results: dict[str, Any]) -> None:
    """(3) 기존 진료 병원 — 진료 분야 일치 + 기존 병원 가산점."""
    weights = get_settings().hospital_score
    item = results["by_name"]["서울동물메디컬센터"]

    assert item.score == (
        weights.is_animal_hospital
        + weights.has_phone
        + weights.specialty_matches
        + weights.is_previous_hospital
    )
    assert item.suitability == "recommended"
    assert "기존 진료 기록이 있는 병원 (가산점)" in item.matched_reasons
    assert any("심장" in reason for reason in item.matched_reasons)
    assert any(
        "기존 진료 병원이라도" in note for note in item.verification_required
    ), "기존 병원이라도 지금 응급 접수가 되는지는 전화로 확인해야 한다."


def test_specialty_hospital_without_phone_score_and_reasons(
    results: dict[str, Any],
) -> None:
    """(4) 전문진료는 있으나 전화번호가 없는 병원 — 전화 가산점을 받지 못한다."""
    weights = get_settings().hospital_score
    item = results["by_name"]["미래동물의료센터"]

    assert item.score == weights.is_animal_hospital + weights.specialty_matches
    assert item.suitability == "possible"
    assert item.hospital.phone is None
    assert "전화번호가 검색 결과에 없음" in item.unmatched_preferences
    assert any("신경" in reason for reason in item.matched_reasons)
    assert any("전화번호가 확인되지 않아" in note for note in item.verification_required)


def test_ranking_is_by_score_and_previous_hospital_is_not_forced_first(
    results: dict[str, Any],
) -> None:
    """기존 진료 병원이라는 이유로 1순위를 강제하지 않는다(명세 35절)."""
    names = [item.hospital.name for item in results["ordered"]]
    scores = [item.score for item in results["ordered"]]

    assert scores == sorted(scores, reverse=True)
    assert names[0] == "강남24시동물병원"
    assert names.index("서울동물메디컬센터") > 0


def test_every_result_carries_verification_notice_and_unconfirmed_availability(
    results: dict[str, Any],
) -> None:
    """모든 후보에 '방문 전 전화 확인' 문구가 붙고 실시간 상태는 확정하지 않는다."""
    for item in results["ordered"]:
        assert HOSPITAL_VERIFICATION_NOTICE in item.verification_required
        assert item.hospital.availability == AVAILABILITY_UNCONFIRMED


# ---------------------------------------------------------------------------
# 등급 경계 / 필수조건
# ---------------------------------------------------------------------------
def test_no_contact_channel_is_low_information_regardless_of_score() -> None:
    """연락 수단이 없으면 점수가 높아도 '추천' 이라고 부르지 않는다(명세 33절 필수조건)."""
    weights = get_settings().hospital_score
    unreachable = HospitalCandidate(
        name="연락처없는동물병원",
        phone=None,
        email=None,
        website=None,
        emergency_mentioned=True,
        open_24h_mentioned=True,
    )
    assert grade_suitability(999, unreachable, weights) == "low_information"

    reachable = unreachable.model_copy(update={"phone": "02-000-0000"})
    assert grade_suitability(weights.recommended_min, reachable, weights) == "recommended"
    assert grade_suitability(weights.possible_min, reachable, weights) == "possible"
    assert grade_suitability(weights.possible_min - 1, reachable, weights) == "low_information"


def test_previous_hospital_name_matching_is_partial() -> None:
    """'서울동물메디컬센터' 와 '서울동물메디컬센터 강남점' 을 같은 곳으로 본다(가산점 판정)."""
    candidate = HospitalCandidate(name="서울동물메디컬센터 강남점")
    assert is_previous_hospital(candidate, PREVIOUS_HOSPITALS) is True
    assert is_previous_hospital(HospitalCandidate(name="전혀다른병원"), PREVIOUS_HOSPITALS) is False
    assert is_previous_hospital(HospitalCandidate(name=""), PREVIOUS_HOSPITALS) is False


def test_previous_hospital_names_come_from_diagnoses_latest_first() -> None:
    """기존 진료 병원 목록은 진단서에서 최신 순으로 모은다(명세 33절)."""
    adapter = FixtureClinicalDataAdapter()
    names = collect_previous_hospital_names(adapter.load_diagnoses(DOG_PET_ID))

    assert names[0] == "서울동물메디컬센터"  # 최신 진단서
    assert "행복동물병원" in names


def test_requirements_never_promote_specialty_to_required() -> None:
    """진료과는 어떤 경우에도 필수조건이 되지 않는다(명세 33절 금지 사항)."""
    adapter = FixtureClinicalDataAdapter()
    state = {
        "user_message": "지금 계속 경련해요",
        "current_observation": {"raw_text": "지금 계속 경련해요", "symptoms": ["신경 증상"]},
        "pet_profile": adapter.load_pet_profile(DOG_PET_ID),
        "related_diagnoses": adapter.load_diagnoses(DOG_PET_ID),
        "daily_entries": adapter.load_daily_entries(DOG_PET_ID),
        "region_name": REGION,
        "final_risk": "emergency",
        "emergency_urgency": "contact_ready",
    }
    requirements = build_hospital_requirements(state)

    assert requirements.required == list(REQUIRED_CONDITIONS)
    assert "신경" in requirements.specialty_keywords
    assert any("신경" in item and "우대" in item for item in requirements.preferred)
    assert all(REGION in query for query in requirements.search_queries)


# ---------------------------------------------------------------------------
# Node 계약
# ---------------------------------------------------------------------------
def test_hospital_suitability_node_sorts_and_prepares_call_action() -> None:
    """node 는 점수순 목록·최상위 선택·전화 action 을 State 에 남긴다."""
    state = {
        "raw_hospital_results": parse_hospital_results(RAW_RESULTS),
        "hospital_requirements": {
            "specialty_keywords": SPECIALTY_KEYWORDS,
            "previous_hospital_names": PREVIOUS_HOSPITALS,
        },
    }
    update = hospital_suitability_node(state, llm=None)

    scores = [item["score"] for item in update["hospital_results"]]
    assert scores == sorted(scores, reverse=True)
    assert update["selected_hospital"]["hospital"]["name"] == "강남24시동물병원"

    call_actions = [a for a in update["ui_actions"] if a["type"] == "CALL_HOSPITAL"]
    assert len(call_actions) == 1
    assert call_actions[0]["phone"] == "02-987-6543"
    assert call_actions[0]["availability"] == AVAILABILITY_UNCONFIRMED
    assert call_actions[0]["notice"] == HOSPITAL_VERIFICATION_NOTICE


def test_hospital_suitability_node_with_no_candidates() -> None:
    """후보가 없으면 빈 목록을 확정한다 — 이전 turn 의 병원이 남으면 안 된다."""
    assert hospital_suitability_node({"raw_hospital_results": []}, llm=None) == {
        "hospital_results": []
    }


def test_prepare_call_action_falls_back_to_guidance_without_phone() -> None:
    """번호를 모르면 임의의 번호(119 등)를 넣지 않고 안내형 action 을 남긴다."""
    update = prepare_call_hospital_action_node({"hospital_results": []})
    action = update["ui_actions"][0]

    assert action["type"] == "CALL_HOSPITAL"
    assert action["phone"] is None
    assert action["availability"] == AVAILABILITY_UNCONFIRMED
    assert "전화" in action["notice"]


def test_prepare_call_action_uses_top_hospital_phone() -> None:
    """적합도 상위 병원의 번호를 그대로 쓴다(재조립하지 않는다)."""
    evaluated = evaluate_hospitals(
        parse_hospital_results(RAW_RESULTS),
        specialty_keywords=SPECIALTY_KEYWORDS,
        previous_names=PREVIOUS_HOSPITALS,
        llm=None,
    )
    state = {"hospital_results": [item.model_dump() for item in evaluated]}
    action = prepare_call_hospital_action_node(state)["ui_actions"][0]

    assert action["hospital_name"] == "강남24시동물병원"
    assert action["phone"] == "02-987-6543"
