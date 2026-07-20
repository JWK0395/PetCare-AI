"""Emergency Subgraph 테스트 (명세 24·29·32·33·34·43절).

명세 43절의 두 응급 시나리오를 그대로 재현한다.

    응급      : "지금 계속 경련하고 반응이 둔해요" + region_name="서울 강남구"
                → emergency / 병원 검색 / 요구사항 생성 / 적합도 평가 /
                  최소정보 부족 시 interrupt / PDF·email draft
    즉시 위급 : "숨을 거의 쉬지 못하고 의식이 없어요"
                → critical_immediate / 정보가 부족해도 CALL_HOSPITAL action /
                  PDF 미확인 필드 허용

이 서브그래프에서 가장 위험한 실패는 **없는 병원을 안내하는 것**이다. 그래서
지역 정보가 없을 때 병원을 지어내지 않고 `REQUEST_LOCATION` 을 돌려주는지,
그리고 모든 병원 안내에 '전화 확인' 문구가 붙는지를 함께 확인한다(명세 34·47절).

원칙
  * LLM 주입 없음(None), Tavily 없음 — 병원 검색 서비스는 mock 주입.
  * PDF 는 실제로 만들되 출력 위치만 tmp 로 돌린다.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest

from petcare_ai.adapters.clinical_data_adapter import FixtureClinicalDataAdapter
from petcare_ai.config import configure, get_settings
from petcare_ai.graph.nodes.fast_emergency_guard import (
    detect_emergency_signals,
    fast_emergency_guard_node,
)
from petcare_ai.graph.nodes.hospital_requirements import REQUIRED_CONDITIONS
from petcare_ai.graph.nodes.hospital_search import AVAILABILITY_UNCONFIRMED, check_region_node
from petcare_ai.graph.prompts import HOSPITAL_VERIFICATION_NOTICE
from petcare_ai.graph.routers import route_after_fast_guard, route_region
from petcare_ai.graph.state import make_initial_state
from petcare_ai.graph.subgraphs import SubgraphDeps
from petcare_ai.graph.subgraphs.emergency import build_emergency_subgraph

DOG_PET_ID = 1
REGION = "서울 강남구"

EMERGENCY_MESSAGE = "지금 계속 경련하고 반응이 둔해요"
CRITICAL_MESSAGE = "숨을 거의 쉬지 못하고 의식이 없어요"

#: 응급 최소정보 8항목(명세 29절) — resume 시 라벨로 답한다.
EMERGENCY_ANSWERS = {
    "현재 가장 심한 증상": "경련이 반복되고 반응이 둔해요",
    "증상 시작 시각": "30분 전부터",
    "현재도 진행 중인지": "네, 지금도 계속돼요",
    "의식 또는 반응 상태": "불러도 반응이 거의 없어요",
    "호흡 상태": "숨이 빠르고 거칠어요",
    "움직일 수 있는지": "혼자 일어서지 못해요",
    "외상 또는 위험물질 섭취 가능성": "모름",
    "대략적인 횟수": "3번 정도요",
}


# ---------------------------------------------------------------------------
# mock 병원 검색
# ---------------------------------------------------------------------------
HOSPITAL_RAW_RESULTS: list[dict[str, Any]] = [
    {
        "title": "강남24시동물병원 | 24시간 응급 진료",
        "url": "https://gangnam24vet.example.com/emergency",
        "content": (
            "24시간 응급 진료 안내입니다. 전화 02-987-6543 으로 문의해 주세요. "
            "서울 강남구 언주로 100. 신경과 진료와 입원 시설을 운영합니다."
        ),
        "score": 0.82,
        "availability": AVAILABILITY_UNCONFIRMED,
    },
    {
        "title": "행복동물병원 진료 안내",
        "url": "https://happyvet.example.com",
        "content": "진료 안내 전화 02-123-4567. 서울 강남구 테헤란로 12.",
        "score": 0.61,
        "availability": AVAILABILITY_UNCONFIRMED,
    },
]


class FakeHospitalSearch:
    """`HospitalSearchService` 대체 — 검색어와 호출 횟수를 기록한다.

    `search(queries, max_results)` 시그니처를 그대로 지킨다. 실제 서비스가
    키 없음/실패를 빈 리스트로 흡수하는 것과 같도록 `results=[]` 도 지원한다.
    """

    def __init__(self, results: list[dict[str, Any]] | None = None) -> None:
        self._results = list(results if results is not None else HOSPITAL_RAW_RESULTS)
        self.calls: list[list[str]] = []

    def search(self, queries: list[str], max_results: int = 5) -> list[dict[str, Any]]:
        self.calls.append(list(queries))
        return [dict(item) for item in self._results]

    @property
    def all_queries(self) -> list[str]:
        return [query for batch in self.calls for query in batch]


# ---------------------------------------------------------------------------
# 공용 fixture / 헬퍼
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def _no_api_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "TAVILY_API_KEY"):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture()
def pdf_output_dir(tmp_path: Path) -> Any:
    original = get_settings().output_dir
    configure(output_dir=tmp_path)
    yield tmp_path
    configure(output_dir=original)


@pytest.fixture()
def adapter() -> FixtureClinicalDataAdapter:
    return FixtureClinicalDataAdapter()


def _emergency_state(
    adapter: FixtureClinicalDataAdapter,
    message: str,
    region_name: str | None,
    **overrides: Any,
) -> dict:
    state = make_initial_state(
        pet_id=DOG_PET_ID,
        user_message=message,
        region_name=region_name,
        pet_profile=adapter.load_pet_profile(DOG_PET_ID),
        diagnoses=adapter.load_diagnoses(DOG_PET_ID),
        daily_entries=adapter.load_daily_entries(DOG_PET_ID),
    )
    state.update(overrides)
    return dict(state)


def _build_app(hospital_search: FakeHospitalSearch, checkpointer: Any = None) -> Any:
    return build_emergency_subgraph(
        SubgraphDeps(
            settings=get_settings(),
            llm=None,
            hospital_search=hospital_search,
            checkpointer=checkpointer,
        )
    )


def _action_types(state: dict) -> set[str]:
    return {str(action.get("type")) for action in (state.get("ui_actions") or [])}


# ---------------------------------------------------------------------------
# 명세 24·32절 — 지역 확인 분기
# ---------------------------------------------------------------------------
def test_check_region_only_judges_and_does_not_write_state() -> None:
    """`Check Region Input` 은 판단만 한다 — 병렬 branch 와 같은 key 를 건드리지 않는다.

    실제 그래프가 쓰는 구현은 `nodes/hospital_search.check_region_node` 다
    (`resolve_optional_node` 가 전용 node 를 먼저 찾는다). 라우팅은
    `route_region()` 이 State 를 읽어 결정하므로 이 node 는 빈 dict 를 돌려준다.
    """
    assert check_region_node({"region_name": "   "}) == {}
    assert check_region_node({"region_name": REGION}) == {}


def test_route_region_treats_blank_region_as_missing() -> None:
    """공백만 있는 지역명은 '지역 없음' 이다 — 빈 검색어로 Tavily 를 부르면 안 된다."""
    assert route_region({"region_name": REGION}) == "hospital_search"
    assert route_region({"region_name": "   "}) == "request_location"
    assert route_region({"region_name": None}) == "request_location"


def test_emergency_fallback_check_region_normalizes_blank_input() -> None:
    """서브그래프 예비 구현은 지역명을 정규화한다(`" "` 가 '지역 있음' 으로 새지 않도록)."""
    from petcare_ai.graph.subgraphs.emergency import (
        check_region_node as fallback_check_region,
    )

    assert fallback_check_region({"region_name": "   "})["region_name"] is None
    assert fallback_check_region({"region_name": " 서울 강남구 "})["region_name"] == REGION


# ---------------------------------------------------------------------------
# 명세 43절 — 응급 (지역 있음)
# ---------------------------------------------------------------------------
def test_emergency_flow_searches_hospitals_and_builds_documents(
    adapter: FixtureClinicalDataAdapter, pdf_output_dir: Path
) -> None:
    """경련·반응 저하 → 응급 / 병원 검색 / 요구사항 / 적합도 / interrupt / PDF·email."""
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.types import Command

    hospital_search = FakeHospitalSearch()
    app = _build_app(hospital_search, InMemorySaver())
    config = {"configurable": {"thread_id": "emergency-full"}}

    paused = app.invoke(_emergency_state(adapter, EMERGENCY_MESSAGE, REGION), config)

    # --- 최소정보 부족 → interrupt (명세 32절 M/N) ---
    assert "__interrupt__" in paused
    payload = paused["__interrupt__"][0].value
    assert payload["type"] == "missing_information"
    assert payload["missing_fields"], "응급 연락 최소정보가 부족하면 되물어야 한다."
    assert set(payload["missing_fields"]) <= set(EMERGENCY_ANSWERS)
    assert payload["allow_unknown"] is True

    final = app.invoke(Command(resume=dict(EMERGENCY_ANSWERS)), config)

    # --- 위험도 (명세 28·32절) ---
    assert final["final_risk"] == "emergency"
    assert final["emergency_urgency"] in ("contact_ready", "critical_immediate")
    assert final["document_type"] == "emergency_consultation"
    assert final["missing_fields"] == []

    # --- 병원 요구사항 (명세 33절) ---
    requirements = final["hospital_requirements"]
    assert requirements["required"] == list(REQUIRED_CONDITIONS)
    # 진료과는 절대 필수조건이 되지 않는다 — 갈 수 있는 병원을 스스로 지우게 된다.
    assert all("우대" not in item for item in requirements["required"])
    assert "신경" in requirements["specialty_keywords"], "경련은 신경과 힌트로 이어져야 한다."
    assert "심장" in requirements["specialty_keywords"], "PET DB 기저질환도 반영되어야 한다."
    assert "서울동물메디컬센터" in requirements["previous_hospital_names"]

    # --- 병원 검색 (명세 34절) ---
    assert hospital_search.calls, "지역이 있으면 병원 검색을 호출해야 한다."
    assert all(REGION in query for query in hospital_search.all_queries)
    assert any("응급" in query for query in hospital_search.all_queries)

    # --- 병원 적합도 (명세 35절) ---
    results = final["hospital_results"]
    assert results, "검색 결과가 있으면 적합도 평가 결과가 있어야 한다."
    scores = [item["score"] for item in results]
    assert scores == sorted(scores, reverse=True), "점수 내림차순으로 정렬되어야 한다."
    for item in results:
        assert item["hospital"]["availability"] == AVAILABILITY_UNCONFIRMED
        assert HOSPITAL_VERIFICATION_NOTICE in item["verification_required"]
    assert final["selected_hospital"]["hospital"]["name"] == results[0]["hospital"]["name"]

    # --- 문서 (명세 36~38절) ---
    assert final["pdf_path"] and os.path.exists(final["pdf_path"])
    assert os.path.getsize(final["pdf_path"]) > 0
    assert final["email_draft"]["attachment_path"] == final["pdf_path"]
    assert final["email_draft"]["subject"].startswith("[응급 상담자료]")
    assert final["consultation_packet"]["document_type"] == "emergency_consultation"

    # --- 안내 문구 / UI action (명세 32·40절) ---
    message = final["draft_response"]
    assert "즉시" in message and "병원" in message
    assert HOSPITAL_VERIFICATION_NOTICE in message
    assert "CALL_HOSPITAL" in _action_types(final)
    assert {"OPEN_PDF_PREVIEW", "OPEN_GMAIL_COMPOSE"} <= _action_types(final)


def test_emergency_without_hospital_results_does_not_invent_one(
    adapter: FixtureClinicalDataAdapter, pdf_output_dir: Path
) -> None:
    """검색이 빈 결과여도(키 없음·실패) 병원을 지어내지 않는다."""
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.types import Command

    hospital_search = FakeHospitalSearch(results=[])
    app = _build_app(hospital_search, InMemorySaver())
    config = {"configurable": {"thread_id": "emergency-no-hospital"}}

    app.invoke(_emergency_state(adapter, EMERGENCY_MESSAGE, REGION), config)
    final = app.invoke(Command(resume=dict(EMERGENCY_ANSWERS)), config)

    assert hospital_search.calls, "검색 자체는 시도한다."
    assert final["hospital_results"] == []
    assert "확인하지 못했습니다" in final["draft_response"]
    assert final["draft_response"].count("동물병원") >= 1
    assert final["pdf_path"], "병원을 못 찾아도 상담 자료는 만든다."


# ---------------------------------------------------------------------------
# 명세 43절 — 즉시 위급
# ---------------------------------------------------------------------------
def test_fast_guard_marks_critical_immediate() -> None:
    """'숨을 거의 쉬지 못하고 의식이 없어요' 는 규칙만으로 즉시 위급 판정된다."""
    state = make_initial_state(pet_id=DOG_PET_ID, user_message=CRITICAL_MESSAGE)
    state.update(fast_emergency_guard_node(state))

    assert state["emergency_urgency"] == "critical_immediate"
    assert state["rule_risk"] == "emergency"
    assert route_after_fast_guard(state) == "emergency"
    assert state["red_flags"], "판정 근거가 red flag 로 남아야 한다."
    assert "의식" in " ".join(state["red_flags"])


@pytest.mark.parametrize(
    ("message", "expected_label"),
    [
        ("숨을 거의 쉬지 못해요", "호흡곤란"),
        ("계속 경련해요", "경련·발작"),
        ("잇몸이 파랗게 변했어요", "청색증"),
        ("초콜릿을 먹었어요", "중독 의심 섭취"),
    ],
)
def test_critical_signals_are_detected_by_rules_only(
    message: str, expected_label: str
) -> None:
    """즉시 위급 신호는 LLM 없이 규칙 사전만으로 잡힌다(명세 24절).

    LLM 왕복 지연·실패 자체가 위험인 상황이라 이 경로에는 네트워크가 없어야 한다.
    """
    hits = detect_emergency_signals(message)
    assert any(hit.signal.label == expected_label for hit in hits), (
        f"'{message}' 에서 '{expected_label}' 신호를 놓쳤습니다: "
        f"{[hit.red_flag for hit in hits]}"
    )


def test_critical_case_prepares_call_action_without_waiting_for_information(
    adapter: FixtureClinicalDataAdapter, pdf_output_dir: Path
) -> None:
    """즉시 위급이면 정보가 부족해도 되묻지 않고 CALL_HOSPITAL 을 준비한다(명세 29·32절)."""
    hospital_search = FakeHospitalSearch()
    app = _build_app(hospital_search)

    final = app.invoke(
        _emergency_state(
            adapter,
            CRITICAL_MESSAGE,
            REGION,
            emergency_urgency="critical_immediate",
            final_risk="emergency",
            rule_risk="emergency",
        )
    )

    # 정보 수집이 전화 action 을 막지 않는다 — interrupt 자체가 없어야 한다.
    assert "__interrupt__" not in final
    assert final["emergency_urgency"] == "critical_immediate"
    assert final["minimum_information_ready"] is True
    assert final["missing_fields"], "부족한 항목은 기록만 하고 진행한다."

    assert "CALL_HOSPITAL" in _action_types(final)

    # PDF 는 미확인 필드를 허용한다(추측으로 채우지 않는다).
    packet = final["consultation_packet"]
    assert packet["unknown_fields"], "미확인 항목이 PDF 자료에 남아야 한다."
    for label in final["missing_fields"]:
        assert label in packet["unknown_fields"]
    assert final["pdf_path"] and os.path.getsize(final["pdf_path"]) > 0


def test_critical_case_without_region_requests_location_instead_of_guessing(
    adapter: FixtureClinicalDataAdapter, pdf_output_dir: Path
) -> None:
    """지역을 모르면 병원을 추측하지 않고 위치를 요청한다(명세 32절 F)."""
    hospital_search = FakeHospitalSearch()
    app = _build_app(hospital_search)

    final = app.invoke(
        _emergency_state(
            adapter,
            CRITICAL_MESSAGE,
            None,
            emergency_urgency="critical_immediate",
            final_risk="emergency",
        )
    )

    assert hospital_search.calls == [], "지역이 없으면 병원 검색을 호출하지 않는다."
    assert final["hospital_results"] == []
    assert "REQUEST_LOCATION" in _action_types(final)
    # 명세 40절: critical_immediate 면 연락 action 이 반드시 하나는 있어야 한다.
    assert _action_types(final) & {"CALL_HOSPITAL", "REQUEST_LOCATION"}
    assert "지역" in final["draft_response"] or "위치" in final["draft_response"]
    assert "즉시" in final["draft_response"]


def test_immediate_message_comes_before_search_results(
    adapter: FixtureClinicalDataAdapter, pdf_output_dir: Path
) -> None:
    """응급 안내 문구가 검색·문서 결과보다 앞에 온다(명세 32절 B).

    뒤 단계가 느리거나 실패해도 보호자가 지금 무엇을 해야 하는지는 이미
    답변 맨 앞에 들어가 있어야 한다.
    """
    app = _build_app(FakeHospitalSearch())
    final = app.invoke(
        _emergency_state(
            adapter,
            CRITICAL_MESSAGE,
            REGION,
            emergency_urgency="critical_immediate",
            final_risk="emergency",
        )
    )

    lines = [line for line in final["draft_response"].splitlines() if line.strip()]
    assert "응급" in lines[0]
    # 확정 진단·처방 문구가 섞이면 안 된다.
    from petcare_ai.graph.nodes.output_check import find_forbidden_expressions

    assert find_forbidden_expressions(final["draft_response"]) == []
