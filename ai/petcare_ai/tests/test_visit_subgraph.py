"""Visit Subgraph 테스트 (명세 29·31·36·37·38·43절).

명세 43절 '병원 상담 권고' 시나리오를 그대로 재현한다.

    입력: "며칠째 식사를 거의 안 하고 활동량이 계속 줄었어요"
    기대: visit 분기 / Missing Information interrupt / resume 후 PDF 생성 /
          email draft 생성

여기서 특히 신경 써서 확인하는 것 두 가지.

1. **되묻기 순환이 끝난다.** mermaid 의 `Missing Info → Interrupt → Missing Info`
   순환은 보호자가 계속 모호하게 답하면 무한 루프가 된다. 라운드 한도를 넘으면
   남은 항목을 '미확인' 으로 두고 진행하는지 확인한다(명세 29절).
2. **PDF 경로와 이메일 첨부 경로가 같다.** 어긋나면 Output Check 가 치명 오류로
   잡는다(명세 40절 8번). 경로를 두 번 조립하지 않는다는 계약의 테스트다.

원칙
  * LLM 주입 없음(None).
  * PDF 는 실제로 만든다 — 파일 크기 검증까지가 명세 37절 요구사항이라
    mock 으로 대체하면 검증할 것이 남지 않는다. 출력 위치만 tmp 로 돌린다.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest

from petcare_ai.adapters.clinical_data_adapter import FixtureClinicalDataAdapter
from petcare_ai.config import configure, get_settings
from petcare_ai.graph.nodes.assessment import evaluate_rules
from petcare_ai.graph.nodes.clinical_context_priority import clinical_context_priority_node
from petcare_ai.graph.nodes.email_draft import SUBJECT_PREFIX
from petcare_ai.graph.nodes.missing_information import UNKNOWN_VALUE
from petcare_ai.graph.nodes.output_check import check_output, find_forbidden_expressions
from petcare_ai.graph.prompts import MEDICAL_DISCLAIMER
from petcare_ai.graph.routers import (
    MAX_MISSING_INFORMATION_ROUNDS,
    MISSING_INFO_ROUNDS_KEY,
    route_final_risk,
)
from petcare_ai.graph.state import make_initial_state
from petcare_ai.graph.subgraphs import SubgraphDeps
from petcare_ai.graph.subgraphs.visit import build_visit_subgraph

DOG_PET_ID = 1
VISIT_MESSAGE = "며칠째 식사를 거의 안 하고 활동량이 계속 줄었어요"


# ---------------------------------------------------------------------------
# 공용 fixture
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def _no_api_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "TAVILY_API_KEY"):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture()
def pdf_output_dir(tmp_path: Path) -> Any:
    """PDF 산출물을 사용자 작업 디렉터리가 아니라 tmp 로 보낸다."""
    original = get_settings().output_dir
    configure(output_dir=tmp_path)
    yield tmp_path
    configure(output_dir=original)


@pytest.fixture()
def adapter() -> FixtureClinicalDataAdapter:
    return FixtureClinicalDataAdapter()


def _visit_state(adapter: FixtureClinicalDataAdapter) -> dict:
    """Clinical Context 까지 끝낸 visit 진입 State 를 만든다."""
    state = make_initial_state(
        pet_id=DOG_PET_ID,
        user_message=VISIT_MESSAGE,
        pet_profile=adapter.load_pet_profile(DOG_PET_ID),
        diagnoses=adapter.load_diagnoses(DOG_PET_ID),
        daily_entries=adapter.load_daily_entries(DOG_PET_ID),
    )
    state.update(clinical_context_priority_node(state))
    state["final_risk"] = "visit"
    state["rule_risk"] = "visit"
    return dict(state)


def _build_app(checkpointer: Any) -> Any:
    return build_visit_subgraph(
        SubgraphDeps(settings=get_settings(), llm=None, checkpointer=checkpointer)
    )


def _action_types(state: dict) -> set[str]:
    return {str(action.get("type")) for action in (state.get("ui_actions") or [])}


# ---------------------------------------------------------------------------
# 명세 43절 — visit 분기 판정
# ---------------------------------------------------------------------------
def test_persistent_anorexia_and_lethargy_route_to_visit(
    adapter: FixtureClinicalDataAdapter,
) -> None:
    """'며칠째 식사를 거의 안 하고 활동량이 계속 줄었어요' 는 visit 로 간다."""
    state = _visit_state(adapter)
    state["final_risk"] = "normal"  # 라우터가 스스로 올려야 한다

    result = evaluate_rules(state)
    assert result.risk_level == "visit"
    assert result.emergency_urgency == "none"
    labels = " ".join(result.red_flags)
    assert "식욕" in labels or "식사" in labels
    assert "활동" in labels or "기력" in labels

    state["rule_risk"] = result.risk_level
    assert route_final_risk(state) == "visit"


# ---------------------------------------------------------------------------
# 명세 43절 — interrupt → resume → PDF / email draft
# ---------------------------------------------------------------------------
def test_visit_flow_interrupts_then_produces_pdf_and_email_draft(
    adapter: FixtureClinicalDataAdapter, pdf_output_dir: Path
) -> None:
    """부족한 정보를 되묻고, resume 후 PDF·이메일 초안까지 만든다."""
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.types import Command

    app = _build_app(InMemorySaver())
    config = {"configurable": {"thread_id": "visit-happy"}}

    paused = app.invoke(_visit_state(adapter), config)

    payload = paused["__interrupt__"][0].value
    assert payload["type"] == "missing_information"
    assert payload["risk_level"] == "visit"
    # 진료 의뢰서에 들어갈 항목이라 자유 서술로 추론하지 않고 명시적으로 받는다.
    assert set(payload["missing_fields"]) == {"빈도", "증상 변화", "현재 식사·음수·활동 상태"}
    assert payload["allow_unknown"] is True
    assert paused.get("pdf_path") is None

    final = app.invoke(
        Command(
            resume={
                "빈도": "하루 2~3회",
                "증상 변화": "처음보다 점점 심해지고 있어요",
                "현재 식사·음수·활동 상태": "사료는 1/3만 먹고 산책을 거부해요",
            }
        ),
        config,
    )

    # --- 되묻기 종료 ---
    assert final["missing_fields"] == []
    assert final["minimum_information_ready"] is True

    # --- PDF (명세 37절) ---
    pdf_path = final["pdf_path"]
    assert pdf_path, "resume 후에는 PDF 가 생성되어야 한다."
    assert os.path.exists(pdf_path)
    assert os.path.getsize(pdf_path) > 0
    assert Path(pdf_path).parent == Path(pdf_output_dir)
    assert final["pdf_filename"].endswith(".pdf")
    assert "초코" in final["pdf_filename"]

    # --- Email draft (명세 38절) — 첨부 경로는 PDF 경로와 반드시 같다 ---
    draft = final["email_draft"]
    assert draft["attachment_path"] == pdf_path
    assert draft["attachment_filename"] == final["pdf_filename"]
    assert draft["to"] is None, "병원 이메일이 없으면 지어내지 않고 None 으로 둔다."
    assert draft["subject"].startswith(SUBJECT_PREFIX["visit_consultation"])
    assert "초코" in draft["subject"]

    # --- packet (명세 36절) ---
    packet = final["consultation_packet"]
    assert packet["document_type"] == "visit_consultation"
    assert packet["pet"]["name"] == "초코"
    assert packet["current_condition"]["frequency"] == "하루 2~3회"
    assert packet["related_diagnoses"], "선택된 진단서가 원문 그대로 실려야 한다."

    # --- 결과 메시지 / UI action (명세 31·40절) ---
    assert "병원" in final["draft_response"]
    assert MEDICAL_DISCLAIMER in final["draft_response"]
    assert {"OPEN_PDF_PREVIEW", "OPEN_GMAIL_COMPOSE"} <= _action_types(final)
    assert find_forbidden_expressions(final["draft_response"]) == []

    # 상담이 끝나면 내부 카운터는 남지 않는다.
    assert MISSING_INFO_ROUNDS_KEY not in final["collected_information"]


def test_unknown_answers_are_preserved_as_unconfirmed_fields(
    adapter: FixtureClinicalDataAdapter, pdf_output_dir: Path
) -> None:
    """'모름' 답변은 추측으로 채우지 않고 PDF 항목에 그대로 남는다(명세 29·36절)."""
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.types import Command

    app = _build_app(InMemorySaver())
    config = {"configurable": {"thread_id": "visit-unknown"}}

    app.invoke(_visit_state(adapter), config)
    final = app.invoke(
        Command(
            resume={
                "빈도": UNKNOWN_VALUE,
                "증상 변화": UNKNOWN_VALUE,
                "현재 식사·음수·활동 상태": UNKNOWN_VALUE,
            }
        ),
        config,
    )

    assert final["missing_fields"] == []  # '모름' 은 유효한 답변이다
    condition = final["consultation_packet"]["current_condition"]
    assert condition["frequency"] == UNKNOWN_VALUE
    assert condition["change"] == UNKNOWN_VALUE
    assert final["pdf_path"] and os.path.exists(final["pdf_path"])


def test_vague_answers_stop_after_round_limit_and_keep_unknown_fields(
    adapter: FixtureClinicalDataAdapter, pdf_output_dir: Path
) -> None:
    """모호한 답이 반복돼도 되묻기가 무한히 돌지 않는다(명세 29절 / recursion 방지).

    한도를 넘기면 남은 항목은 `missing_fields` 에 **남긴 채로** 진행하고, 그 항목이
    PDF 의 `unknown_fields` 로 이어진다. 조용히 지워 버리면 수의사가 '확인됨' 과
    '미확인' 을 구분할 수 없게 된다.
    """
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.types import Command

    app = _build_app(InMemorySaver())
    config = {"configurable": {"thread_id": "visit-vague"}}

    app.invoke(_visit_state(adapter), config)
    # 항목을 특정할 수 없는 자유 서술 — 어느 항목의 답인지 단정하지 않는다.
    final = app.invoke(Command(resume="그냥 좀 안 좋아 보여요"), config)

    assert "__interrupt__" not in final, (
        f"되묻기 한도({MAX_MISSING_INFORMATION_ROUNDS}회)를 넘으면 진행해야 한다."
    )
    assert final["missing_fields"], "남은 항목은 지우지 않고 미확인으로 남긴다."

    packet = final["consultation_packet"]
    for label in final["missing_fields"]:
        assert label in packet["unknown_fields"]
    # 자유 서술은 임의 항목에 배정하지 않고 별도 자리에 원문 보존한다.
    assert packet["current_condition"]["보호자 추가 서술"] == "그냥 좀 안 좋아 보여요"
    # 내부 관리용 key 가 진료 자료에 인쇄되면 안 된다.
    assert MISSING_INFO_ROUNDS_KEY not in packet["current_condition"]
    assert all(not key.startswith("__") for key in packet["current_condition"])

    assert final["pdf_path"] and os.path.getsize(final["pdf_path"]) > 0
    assert "미확인" in final["draft_response"]


# ---------------------------------------------------------------------------
# 명세 40절 — 만들어진 결과가 출력 검사를 통과하는가
# ---------------------------------------------------------------------------
def test_visit_result_passes_output_check(
    adapter: FixtureClinicalDataAdapter, pdf_output_dir: Path
) -> None:
    """생성된 visit 결과가 Output Check 9개 항목을 그대로 통과한다."""
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.types import Command

    app = _build_app(InMemorySaver())
    config = {"configurable": {"thread_id": "visit-output-check"}}

    app.invoke(_visit_state(adapter), config)
    final = app.invoke(
        Command(
            resume={
                "빈도": "하루 2회",
                "증상 변화": "비슷해요",
                "현재 식사·음수·활동 상태": "사료를 절반만 먹어요",
            }
        ),
        config,
    )

    result = check_output(dict(final))
    assert result.action == "accept", f"예상치 못한 출력 오류: {result.errors}"
    assert result.valid is True
