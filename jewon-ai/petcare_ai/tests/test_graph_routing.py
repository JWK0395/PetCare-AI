"""전체 Graph 라우팅·공통 State 계약 테스트 (명세 24·25·26·28·43절).

검증하는 것은 **분기 규칙 자체**다. 명세 19절이 "분기는 LangGraph 가 한다" 를
요구하므로, 각 node 가 무슨 문장을 만드는지가 아니라 `routers.py` 가 어떤 라벨을
돌려주는지를 단언한다. 라우터가 흔들리면 답변 품질과 무관하게 안전 경로 자체가
무너지기 때문이다.

명세 43절 '공통' 항목을 그대로 담는다.
  - 최초 thread 에서 PET / 진단서 / 일기장 Context 가 로드된다
  - 두 번째 turn 에서 같은 `thread_id` State 가 유지된다
  - pet 별 thread 가 섞이지 않는다
  - DB 전체는 State 에 있지만 prompt 에는 선택된 Context 만 들어간다
그리고 '일반 대화' 항목의 **RAG·Tavily 미호출**을 호출 카운터로 직접 센다.

원칙
  * 네트워크 호출 없음 — LLM 은 주입하지 않고(None) 규칙 경로만 돈다.
  * 임상 데이터는 `FixtureClinicalDataAdapter` 만 사용한다.
  * 파일 산출물 없음.
"""

from __future__ import annotations

import importlib
from typing import Any

import pytest

from petcare_ai.adapters.clinical_data_adapter import FixtureClinicalDataAdapter
from petcare_ai.graph import routers
from petcare_ai.graph.nodes.assessment import build_assessment_prompt, evaluate_rules
from petcare_ai.graph.nodes.clinical_context_priority import clinical_context_priority_node
from petcare_ai.graph.nodes.db_context import (
    make_db_context_node,
    needs_db_context,
    route_context_loaded,
)
from petcare_ai.graph.nodes.fast_emergency_guard import (
    detect_emergency_signals,
    fast_emergency_guard_node,
    is_critical_immediate,
)
from petcare_ai.graph.nodes.general_chat import general_chat_node, unsupported_response_node
from petcare_ai.graph.nodes.supervisor import classify_intent_rule_based, make_supervisor_node
from petcare_ai.graph.state import (
    URGENCY_PRIORITY,
    Replace,
    escalate_risk,
    escalate_urgency,
    make_initial_state,
    make_message,
    merge_records,
    merge_ui_actions,
    merge_unique_strings,
)
from petcare_ai.schemas import RISK_PRIORITY, merge_risk

DOG_PET_ID = 1
CAT_PET_ID = 2


# ---------------------------------------------------------------------------
# 공용 fixture
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def _no_api_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    """키가 없는 환경을 강제한다 — `build_llm()` 이 None 을 돌려주는 정상 경로."""
    for name in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "TAVILY_API_KEY"):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture()
def adapter() -> FixtureClinicalDataAdapter:
    return FixtureClinicalDataAdapter()


def _loaded_state(adapter: FixtureClinicalDataAdapter, pet_id: int, message: str) -> dict:
    """DB Context Agent 를 한 번 돌려 임상 데이터가 실린 State 를 만든다."""
    state = make_initial_state(pet_id=pet_id, user_message=message)
    state.update(make_db_context_node(adapter)(state))
    return state


# ---------------------------------------------------------------------------
# 명세 43절 공통 — Context 로드
# ---------------------------------------------------------------------------
def test_first_thread_loads_pet_diagnosis_and_daily_context(
    adapter: FixtureClinicalDataAdapter,
) -> None:
    """최초 thread 에서 PET / 진단서 / 일기장이 모두 State 에 실린다."""
    state = make_initial_state(pet_id=DOG_PET_ID, user_message="안녕")
    assert state["context_loaded"] is False
    assert route_context_loaded(state) == "db_context"

    update = make_db_context_node(adapter)(state)

    assert update["context_loaded"] is True
    assert update["pet_profile"]["id"] == DOG_PET_ID
    assert update["pet_profile"]["species"] == "dog"
    assert len(update["diagnoses"]) == len(adapter.load_diagnoses(DOG_PET_ID))
    assert len(update["daily_entries"]) == len(adapter.load_daily_entries(DOG_PET_ID))
    # 정렬 계약: 마지막이 최신이어야 이후 노드의 '최근 N건' 선택이 성립한다.
    dates = [entry["record_date"] for entry in update["daily_entries"]]
    assert dates == sorted(dates)


def test_context_not_reloaded_on_second_turn(adapter: FixtureClinicalDataAdapter) -> None:
    """같은 pet 이면 두 번째 turn 에서 DB 를 다시 읽지 않는다."""
    state = _loaded_state(adapter, DOG_PET_ID, "안녕")

    assert needs_db_context(state) is False
    assert route_context_loaded(state) == "message_ingest"
    assert make_db_context_node(adapter)(state) == {}


def test_context_reloaded_when_pet_changes(adapter: FixtureClinicalDataAdapter) -> None:
    """pet 이 바뀌면 재로드한다 — 다른 아이의 기록이 섞이면 안 된다."""
    state = _loaded_state(adapter, DOG_PET_ID, "안녕")
    state["pet_id"] = CAT_PET_ID

    assert needs_db_context(state) is True
    assert route_context_loaded(state) == "db_context"

    update = make_db_context_node(adapter)(state)
    assert update["pet_profile"]["name"] == "나비"
    assert update["pet_profile"]["species"] == "cat"


# ---------------------------------------------------------------------------
# 명세 43절 공통 — thread 유지 / pet 별 thread 분리
# ---------------------------------------------------------------------------
def _mini_graph(adapter: FixtureClinicalDataAdapter) -> Any:
    """명세 24절 앞부분(START → Context loaded? → Fast Emergency Guard)만 조립한다.

    전체 그래프 대신 앞부분만 쓰는 이유: 여기서 검증하려는 것은 checkpointer 가
    `thread_id` 단위로 State 를 유지하는지와 pet 별 thread 가 섞이지 않는지이며,
    그 성질은 그래프 길이와 무관하기 때문이다.
    """
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.graph import END, START, StateGraph

    from petcare_ai.graph.state import PetCareState

    graph = StateGraph(PetCareState)
    graph.add_node("db_context", make_db_context_node(adapter))
    graph.add_node("fast_emergency_guard", fast_emergency_guard_node)
    graph.add_conditional_edges(
        START,
        route_context_loaded,
        {"db_context": "db_context", "message_ingest": "fast_emergency_guard"},
    )
    graph.add_edge("db_context", "fast_emergency_guard")
    graph.add_edge("fast_emergency_guard", END)
    return graph.compile(checkpointer=InMemorySaver())


def test_same_thread_id_keeps_state_across_turns(
    adapter: FixtureClinicalDataAdapter,
) -> None:
    """두 번째 turn 에서 같은 `thread_id` State(임상 데이터·대화)가 유지된다."""
    app = _mini_graph(adapter)
    config = {"configurable": {"thread_id": "thread-dog"}}

    first = app.invoke(
        make_initial_state(pet_id=DOG_PET_ID, user_message="안녕", thread_id="thread-dog"),
        config,
    )
    assert first["pet_profile"]["name"] == "초코"
    assert len(first["messages"]) == 1

    second = app.invoke(
        {
            "user_message": "오늘 한 번 토했어요",
            "messages": [make_message("user", "오늘 한 번 토했어요")],
        },
        config,
    )

    # 임상 데이터는 다시 읽지 않아도 그대로 남아 있다.
    assert second["pet_profile"]["name"] == "초코"
    assert len(second["daily_entries"]) == len(first["daily_entries"])
    # 대화는 누적된다(append_messages reducer).
    assert len(second["messages"]) == 2
    assert second["messages"][-1]["content"] == "오늘 한 번 토했어요"


def test_threads_are_isolated_per_pet(adapter: FixtureClinicalDataAdapter) -> None:
    """pet 별 thread 가 섞이지 않는다(명세 21·43절)."""
    app = _mini_graph(adapter)
    dog_config = {"configurable": {"thread_id": "thread-dog"}}
    cat_config = {"configurable": {"thread_id": "thread-cat"}}

    dog = app.invoke(
        make_initial_state(pet_id=DOG_PET_ID, user_message="안녕", thread_id="thread-dog"),
        dog_config,
    )
    cat = app.invoke(
        make_initial_state(pet_id=CAT_PET_ID, user_message="안녕", thread_id="thread-cat"),
        cat_config,
    )

    assert dog["pet_profile"]["name"] == "초코"
    assert cat["pet_profile"]["name"] == "나비"
    # species 는 RAG index 선택 키다 — thread 간에 섞이면 종이 다른 문서를 근거로 쓴다.
    assert dog["pet_profile"]["species"] == "dog"
    assert cat["pet_profile"]["species"] == "cat"

    # 고양이 thread 를 돌린 뒤에도 강아지 thread State 는 그대로다.
    dog_again = app.get_state(dog_config).values
    assert dog_again["pet_profile"]["name"] == "초코"
    assert dog_again["pet_id"] == DOG_PET_ID
    cat_dates = {entry["record_date"] for entry in cat["daily_entries"]}
    dog_dates = {entry["record_date"] for entry in dog_again["daily_entries"]}
    assert len(dog_dates) != len(cat_dates)


# ---------------------------------------------------------------------------
# 명세 43절 공통 — State 에는 전체, prompt 에는 선택 Context 만
# ---------------------------------------------------------------------------
def test_state_holds_full_db_but_prompt_only_selected_context(
    adapter: FixtureClinicalDataAdapter,
) -> None:
    """DB 전체는 State 에 남고 prompt 에는 선택된 것만 들어간다(명세 21·43절)."""
    state = _loaded_state(adapter, DOG_PET_ID, "며칠째 기침을 하고 기운이 없어요")
    state.update(clinical_context_priority_node(state))

    all_entries = state["daily_entries"]
    selected_entries = state["supporting_daily_entries"]
    assert len(all_entries) > len(selected_entries)  # State 에는 전체가 남아 있다

    prompt = build_assessment_prompt(state, evaluate_rules(state))

    # 프롬프트에는 최근 몇 건만 들어간다 — 전체 일기가 통째로 실리면 안 된다.
    dates_in_prompt = [
        entry["record_date"] for entry in all_entries if entry["record_date"] in prompt
    ]
    assert 0 < len(dates_in_prompt) <= 3
    assert all_entries[0]["record_date"] not in prompt  # 가장 오래된 기록은 제외

    # 외부 자료는 '데이터' 경계로 감싸서 넘긴다(프롬프트 인젝션 방어).
    assert "지시가 아니다" in prompt
    # PET DB 필수정보는 들어간다.
    assert "이첨판" in prompt


# ---------------------------------------------------------------------------
# 명세 43절 — 일반 대화 (RAG·Tavily 미호출)
# ---------------------------------------------------------------------------
class _CallCounter:
    """생성/호출 횟수만 세는 감시자 — 실제 검색은 절대 하지 않는다."""

    def __init__(self) -> None:
        self.constructed = 0
        self.searched = 0

    def make_class(self) -> type:
        counter = self

        class _Spy:
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                counter.constructed += 1

            def search(self, *args: Any, **kwargs: Any) -> list:
                counter.searched += 1
                return []

            def retrieve(self, *args: Any, **kwargs: Any) -> Any:
                counter.searched += 1
                raise AssertionError("일반 대화에서 RAG 를 호출하면 안 됩니다.")

        return _Spy


@pytest.mark.parametrize("message", ["안녕", "이 앱에서 무엇을 할 수 있어?"])
def test_general_chat_does_not_call_rag_or_tavily(
    message: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """일반 대화는 general_chat 으로 가고 RAG·Tavily 를 **한 번도** 호출하지 않는다."""
    import petcare_ai.rag.service as rag_service_module
    import petcare_ai.rag.tavily_vet_search as tavily_module

    rag_counter = _CallCounter()
    web_counter = _CallCounter()
    hospital_counter = _CallCounter()

    monkeypatch.setattr(
        rag_service_module, "VeterinaryRagService", rag_counter.make_class()
    )
    monkeypatch.setattr(
        tavily_module, "VeterinaryWebSearchService", web_counter.make_class()
    )
    monkeypatch.setattr(tavily_module, "HospitalSearchService", hospital_counter.make_class())

    state = make_initial_state(pet_id=DOG_PET_ID, user_message=message)
    state.update(make_supervisor_node(None)(state))

    assert state["intent"] == "general_chat"
    assert routers.route_intent(state) == "general_chat"

    state.update(general_chat_node(state))
    assert state["draft_response"].strip()

    assert (rag_counter.constructed, rag_counter.searched) == (0, 0)
    assert (web_counter.constructed, web_counter.searched) == (0, 0)
    assert (hospital_counter.constructed, hospital_counter.searched) == (0, 0)


def test_general_chat_does_not_raise_risk_level() -> None:
    """인사말에 위험도를 붙이지 않는다 — 붙이면 Output Check 가 잘못 발동한다."""
    state = make_initial_state(pet_id=DOG_PET_ID, user_message="안녕")
    update = general_chat_node(state)

    assert "final_risk" not in update
    assert "emergency_urgency" not in update


def test_unsupported_intent_routes_to_unsupported_response() -> None:
    """반려동물 건강과 무관한 요청은 unsupported 로 간다."""
    result = classify_intent_rule_based("비트코인 시세 알려줘")
    assert result.intent == "unsupported"

    state = make_initial_state(pet_id=DOG_PET_ID, user_message="비트코인 시세 알려줘")
    state["intent"] = result.intent
    assert routers.route_intent(state) == "unsupported"
    assert unsupported_response_node(state)["draft_response"].strip()


# ---------------------------------------------------------------------------
# 명세 24절 분기 — Fast Emergency Guard / Supervisor / Final Risk
# ---------------------------------------------------------------------------
def test_fast_guard_sends_critical_case_straight_to_emergency() -> None:
    """즉시 위급이면 Supervisor 를 건너뛰고 Emergency 로 직행한다(명세 24절)."""
    state = make_initial_state(
        pet_id=DOG_PET_ID, user_message="숨을 거의 쉬지 못하고 의식이 없어요"
    )
    state.update(fast_emergency_guard_node(state))

    assert is_critical_immediate(state) is True
    assert state["emergency_urgency"] == "critical_immediate"
    assert state["rule_risk"] == "emergency"
    assert routers.route_after_fast_guard(state) == "emergency"
    assert routers.route_after_fast_emergency_guard(state) == "emergency"


def test_fast_guard_keeps_warning_case_on_supervisor_path() -> None:
    """경고 등급 신호는 red flag 만 남기고 Supervisor 경로를 유지한다."""
    state = make_initial_state(pet_id=DOG_PET_ID, user_message="며칠째 안 먹고 기운이 없어요")
    state.update(fast_emergency_guard_node(state))

    assert state["emergency_urgency"] == "none"
    assert state["red_flags"], "경고 신호는 red flag 로 남아야 downstream 이 참고할 수 있다."
    assert routers.route_after_fast_guard(state) == "supervisor"


def test_fast_guard_ignores_negated_signal() -> None:
    """'경련은 없어요' 처럼 바로 뒤에 부정이 붙은 표현은 신호로 세지 않는다."""
    assert detect_emergency_signals("경련은 없어요") == []


def test_route_intent_falls_back_to_unsupported_on_unknown_value() -> None:
    """알 수 없는 intent 는 건강 경로가 아니라 unsupported 로 보낸다(안전 기본값)."""
    assert routers.route_intent({"intent": "???"}) == "unsupported"
    assert routers.route_intent({}) == "unsupported"


@pytest.mark.parametrize(
    ("intent", "expected"),
    [
        ("general_chat", "general_chat"),
        ("health_question", "health_question"),
        ("hospital_search", "hospital_search"),
        ("unsupported", "unsupported"),
    ],
)
def test_route_intent_passes_known_labels(intent: str, expected: str) -> None:
    assert routers.route_intent({"intent": intent}) == expected
    assert expected in routers.INTENT_LABELS


def test_route_final_risk_never_downgrades_below_evaluators() -> None:
    """`final_risk` 가 낮게 덮어써져도 평가자 원본값으로 되돌린다(명세 28·47절)."""
    state = {
        "final_risk": "normal",
        "rule_risk": "normal",
        "assessment_risk": "visit",
        "double_check_risk": "emergency",
        "emergency_urgency": "none",
    }
    assert routers.route_final_risk(state) == "emergency"


def test_route_final_risk_follows_critical_urgency() -> None:
    """긴급도가 critical_immediate 면 위험도 값과 무관하게 emergency 다."""
    state = {
        "final_risk": "normal",
        "rule_risk": "normal",
        "assessment_risk": "normal",
        "double_check_risk": "normal",
        "emergency_urgency": "critical_immediate",
    }
    assert routers.route_final_risk(state) == "emergency"


@pytest.mark.parametrize("risk", ["normal", "visit", "emergency"])
def test_route_final_risk_labels_are_graph_branches(risk: str) -> None:
    state = {"final_risk": risk, "emergency_urgency": "none"}
    assert routers.route_final_risk(state) == risk
    assert risk in routers.FINAL_RISK_LABELS


# ---------------------------------------------------------------------------
# 명세 24·41절 — 대화 요약 분기
# ---------------------------------------------------------------------------
def test_route_needs_summary_only_for_long_conversation() -> None:
    """짧은 대화에서는 요약 node 를 타지 않는다."""
    from petcare_ai.config import get_settings

    trigger = get_settings().summary_trigger_message_count
    short = {"messages": [make_message("user", "안녕")] * trigger}
    long = {"messages": [make_message("user", "안녕")] * (trigger + 1)}

    assert routers.route_needs_summary(short) == "fast_emergency_guard"
    assert routers.route_needs_summary(long) == "conversation_summary"


# ---------------------------------------------------------------------------
# 명세 29·30·32절 — Missing Information / 응급 연락 분기
# ---------------------------------------------------------------------------
def test_route_missing_info_asks_then_gives_up_after_max_rounds() -> None:
    """되묻기 한도를 넘으면 남은 항목은 '모름' 으로 두고 진행한다(무한 interrupt 방지)."""
    asking = {"missing_fields": ["빈도"], "minimum_information_ready": False}
    assert routers.route_missing_info(asking) == "ask"

    exhausted = dict(asking)
    exhausted["collected_information"] = {
        routers.MISSING_INFO_ROUNDS_KEY: routers.MAX_MISSING_INFORMATION_ROUNDS
    }
    assert routers.missing_information_rounds(exhausted) == (
        routers.MAX_MISSING_INFORMATION_ROUNDS
    )
    assert routers.route_missing_info(exhausted) == "ready"
    # 항목 자체는 지우지 않는다 — PDF 의 '미확인' 으로 이어져야 한다.
    assert exhausted["missing_fields"] == ["빈도"]


def test_route_missing_info_never_blocks_critical_case() -> None:
    """즉시 위급이면 정보가 부족해도 진행한다(명세 29절)."""
    state = {
        "missing_fields": ["호흡 상태", "의식 또는 반응 상태"],
        "minimum_information_ready": False,
        "emergency_urgency": "critical_immediate",
    }
    assert routers.route_missing_info(state) == "ready"
    assert routers.route_emergency_contact(state) == "call_hospital"


def test_route_emergency_contact_labels() -> None:
    ready = {"minimum_information_ready": True, "emergency_urgency": "contact_ready"}
    ask = {
        "missing_fields": ["호흡 상태"],
        "minimum_information_ready": False,
        "emergency_urgency": "contact_ready",
    }
    assert routers.route_emergency_contact(ready) == "ready"
    assert routers.route_emergency_contact(ask) == "ask"
    assert set(routers.EMERGENCY_CONTACT_LABELS) == {"call_hospital", "ready", "ask"}


# ---------------------------------------------------------------------------
# 명세 30·32·40절 — RAG 상태 / 지역 / 출력 검사 분기
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("status", "expected"),
    [
        ("sufficient", "sufficient"),
        ("insufficient", "insufficient"),
        ("conflicting", "conflicting"),
        (None, "insufficient"),
        ("알수없음", "insufficient"),
    ],
)
def test_route_rag_status_defaults_to_insufficient(status: Any, expected: str) -> None:
    """알 수 없는 값은 '근거 충분' 이 아니라 '부족' 으로 본다(안전 기본값)."""
    assert routers.route_rag_status({"rag_sufficiency": status}) == expected


def test_route_region_requires_region_name() -> None:
    """지역을 모르면 병원을 지어내지 않고 위치를 요청한다(명세 32절)."""
    assert routers.route_region({"region_name": "서울 강남구"}) == "hospital_search"
    assert routers.route_region({"region_name": "   "}) == "request_location"
    assert routers.route_region({}) == "request_location"
    # 좌표만으로는 지역명을 만들지 않는다(지오코딩 미구현).
    assert routers.route_region({"latitude": 37.5, "longitude": 127.0}) == "request_location"


@pytest.mark.parametrize(
    ("errors", "retry", "expected"),
    [
        ([], 0, "accept"),
        (["출처 누락"], 0, "regenerate"),
        (["출처 누락"], 1, "fallback"),
        (["[치명] schema 오류"], 0, "fallback"),
    ],
)
def test_route_output_check(errors: list[str], retry: int, expected: str) -> None:
    """명세 40절: 재생성은 최대 1회, 구조적 오류는 곧바로 fallback."""
    state = {"validation_errors": errors, "retry_count": retry}
    assert routers.route_output_check(state) == expected
    assert expected in routers.OUTPUT_CHECK_LABELS


# ---------------------------------------------------------------------------
# 명세 25·28절 — State reducer 계약
# ---------------------------------------------------------------------------
def test_risk_reducers_are_escalate_only() -> None:
    """위험도·긴급도 채널은 올라가기만 한다."""
    assert escalate_risk("emergency", "normal") == "emergency"
    assert escalate_risk("normal", "visit") == "visit"
    assert escalate_urgency("critical_immediate", "none") == "critical_immediate"
    assert escalate_urgency("none", "contact_ready") == "contact_ready"
    assert merge_risk("normal", "visit", None, "emergency") == "emergency"
    assert RISK_PRIORITY["emergency"] > RISK_PRIORITY["visit"] > RISK_PRIORITY["normal"]
    assert URGENCY_PRIORITY["critical_immediate"] > URGENCY_PRIORITY["contact_ready"]


def test_list_reducers_accumulate_and_deduplicate() -> None:
    """병렬 node 가 같은 신호를 내도 사용자에게 중복 노출되지 않는다."""
    assert merge_unique_strings(["호흡곤란"], ["호흡곤란", "기력 저하"]) == [
        "호흡곤란",
        "기력 저하",
    ]
    # Replace 로 감싸면 누적 대신 교체된다(해결된 항목을 지울 수 있어야 한다).
    assert merge_unique_strings(["호흡곤란"], Replace(["기력 저하"])) == ["기력 저하"]

    merged = merge_records(
        [{"evidence_id": "a", "title": "A"}],
        [{"evidence_id": "a", "title": "A"}, {"evidence_id": "b", "title": "B"}],
    )
    assert [item["evidence_id"] for item in merged] == ["a", "b"]

    actions = merge_ui_actions(
        [{"type": "CALL_HOSPITAL", "phone": "02-1-2"}],
        [{"type": "CALL_HOSPITAL", "phone": "02-1-2"}, {"type": "OPEN_PDF_PREVIEW"}],
    )
    assert [action["type"] for action in actions] == ["CALL_HOSPITAL", "OPEN_PDF_PREVIEW"]


# ---------------------------------------------------------------------------
# 전체 그래프(builder) 스모크 — 아직 없으면 건너뛴다
# ---------------------------------------------------------------------------
def _load_graph_factory() -> Any:
    """`graph/builder.py` 의 그래프 생성 함수를 찾는다. 없으면 skip."""
    try:
        module = importlib.import_module("petcare_ai.graph.builder")
    except ImportError as exc:  # pragma: no cover - builder 미작성 환경
        pytest.skip(f"graph/builder.py 를 불러오지 못해 건너뜁니다: {exc}")

    for name in ("build_graph", "build_chat_graph", "compile_graph"):
        factory = getattr(module, name, None)
        if callable(factory):
            return factory
    pytest.skip("graph/builder.py 에 그래프 생성 함수가 없습니다.")  # pragma: no cover


def test_full_graph_contains_spec24_branches() -> None:
    """전체 그래프에 명세 24절의 주요 node 가 모두 들어 있는지 확인한다."""
    factory = _load_graph_factory()
    try:
        app = factory()
    except TypeError:  # pragma: no cover - 인자가 필요한 시그니처
        app = factory(None)

    node_names = set(app.get_graph().nodes)
    for expected in ("supervisor", "fast_emergency_guard", "output_check"):
        assert any(expected in name for name in node_names), (
            f"명세 24절 node '{expected}' 를 그래프에서 찾지 못했습니다: {sorted(node_names)}"
        )
