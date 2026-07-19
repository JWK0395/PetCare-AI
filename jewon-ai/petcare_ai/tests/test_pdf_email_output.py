"""Context 우선순위·PDF·Email Draft·Output Safety 테스트 (명세 20·36~40·43절).

명세 43절의 남은 두 항목을 담는다.

    Context 우선순위 : PET DB 5.2kg / 최신 진단서 4.8kg / 사용자 4.5kg
                       → 충돌이 기록되고, 값과 출처가 모두 PDF 에 보존되며,
                         임의로 하나만 확정하지 않는다.
    Output Safety    : 확정 진단 문장 / 약 복용 중단 지시 / '현재 진료 가능' 단정이
                       각각 차단되는지.

PDF 본문을 텍스트로 되읽는 대신 **PDF 가 실제로 인쇄하는 자료 구조**
(`_normalize_provenance` / `_conflict_note`)를 검사한다. 이 두 함수의 출력이 곧
표에 찍히는 내용이라, 값과 출처가 보존되는지를 가장 가깝게 확인할 수 있다.
그리고 파일 자체는 실제로 생성해 크기까지 확인한다(명세 37절).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest

from petcare_ai.adapters.clinical_data_adapter import FixtureClinicalDataAdapter
from petcare_ai.config import configure, get_settings
from petcare_ai.graph.nodes.clinical_context_priority import (
    SOURCE_DIAGNOSIS,
    SOURCE_PET,
    SOURCE_USER,
    build_clinical_context,
    clinical_context_priority_node,
)
from petcare_ai.graph.nodes.document_agent import (
    build_consultation_packet,
    document_agent_node,
    pdf_generator_node,
)
from petcare_ai.graph.nodes.email_draft import build_email_draft, email_draft_node
from petcare_ai.graph.nodes.final_safety import (
    build_chat_graph_result,
    final_safety_node,
    safe_fallback_message,
)
from petcare_ai.graph.nodes.output_check import (
    FATAL_PREFIX,
    check_output,
    decide_action,
    find_forbidden_expressions,
    output_check_node,
)
from petcare_ai.graph.state import make_initial_state
from petcare_ai.pdf.consultation_pdf import (
    UNKNOWN_LABEL,
    _conflict_note,  # PDF 표에 실제로 찍히는 문자열을 만드는 함수
    _normalize_provenance,  # PDF 표의 입력 자료 구조
    generate_consultation_pdf,
)
from petcare_ai.schemas import ConsultationPacket

DOG_PET_ID = 1
WEIGHT_MESSAGE = "요즘 살이 빠진 것 같아요. 집 저울로 재보니 몸무게가 4.5kg 정도였어요."


# ---------------------------------------------------------------------------
# 공용 fixture
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


@pytest.fixture()
def conflict_state(adapter: FixtureClinicalDataAdapter) -> dict:
    """명세 43절 충돌 fixture(5.2 / 4.8 / 4.5)를 그대로 만든 State."""
    state = make_initial_state(
        pet_id=DOG_PET_ID,
        user_message=WEIGHT_MESSAGE,
        pet_profile=adapter.load_pet_profile(DOG_PET_ID),
        diagnoses=adapter.load_diagnoses(DOG_PET_ID),
        daily_entries=adapter.load_daily_entries(DOG_PET_ID),
        final_risk="visit",
    )
    state.update(clinical_context_priority_node(state))
    return dict(state)


# ---------------------------------------------------------------------------
# 명세 20·43절 — Context 우선순위 충돌
# ---------------------------------------------------------------------------
def test_weight_conflict_records_all_three_sources(
    adapter: FixtureClinicalDataAdapter,
) -> None:
    """세 출처의 몸무게가 다르면 충돌로 기록하고 하나만 남기지 않는다."""
    state = make_initial_state(
        pet_id=DOG_PET_ID,
        user_message=WEIGHT_MESSAGE,
        pet_profile=adapter.load_pet_profile(DOG_PET_ID),
        diagnoses=adapter.load_diagnoses(DOG_PET_ID),
        daily_entries=adapter.load_daily_entries(DOG_PET_ID),
    )
    context = build_clinical_context(dict(state))

    conflicts = {conflict.field: conflict for conflict in context.context_conflicts}
    assert "weight_kg" in conflicts, "체중 충돌이 기록되어야 한다."

    weight = conflicts["weight_kg"]
    values = {float(item["value"]) for item in weight.conflicting_values}
    sources = {item["source"] for item in weight.conflicting_values}

    assert values == {4.5, 5.2, 4.8}, "세 값이 모두 보존되어야 한다."
    assert sources == {SOURCE_USER, SOURCE_PET, SOURCE_DIAGNOSIS}
    # 우선순위: 현재 사용자 입력 > PET DB > 진단서 DB (명세 20절)
    assert weight.selected_source == SOURCE_USER
    assert float(weight.selected_value) == 4.5


def test_pet_context_keeps_original_pet_db_value(conflict_state: dict) -> None:
    """선택값을 쓰되 PET DB 원본을 지우지 않는다(임의 확정 금지)."""
    pet_context = conflict_state["priority_pet_context"]

    assert float(pet_context["weight_kg"]) == 4.5
    assert float(pet_context["weight_kg_pet_db"]) == 5.2
    assert pet_context["weight_kg_source"] == SOURCE_USER


def test_packet_preserves_conflicting_values_and_sources(conflict_state: dict) -> None:
    """충돌은 요약되지 않고 packet provenance 로 그대로 넘어간다(명세 36절)."""
    packet = build_consultation_packet(conflict_state)

    conflict_records = [
        record for record in packet.provenance if record.get("conflicting_values")
    ]
    assert conflict_records, "충돌 기록이 packet 에 실려야 한다."

    values = {
        str(item["value"])
        for record in conflict_records
        for item in record["conflicting_values"]
    }
    assert {"4.5", "5.2", "4.8"} <= values


def test_pdf_layer_prints_every_value_with_its_source(conflict_state: dict) -> None:
    """PDF 표에 값과 출처가 모두 남고 '확정하지 않음' 이 명시된다(명세 43절)."""
    packet = build_consultation_packet(conflict_state)
    records = _normalize_provenance(packet)

    weight_record = next(record for record in records if record["field"] == "weight_kg")
    assert weight_record["has_conflict"] is True

    printed_values = {str(entry["value"]) for entry in weight_record["entries"]}
    printed_sources = {entry["source"] for entry in weight_record["entries"] if entry["source"]}
    assert {"4.5", "5.2", "4.8"} <= printed_values
    assert {SOURCE_USER, SOURCE_PET, SOURCE_DIAGNOSIS} <= printed_sources

    note = _conflict_note(records, "weight_kg", "몸무게")
    assert note is not None
    assert "확정하지 않음" in note
    for value in ("4.5", "5.2", "4.8"):
        assert value in note


def test_conflict_case_generates_real_pdf(
    conflict_state: dict, pdf_output_dir: Path
) -> None:
    """충돌이 있어도 PDF 생성은 정상적으로 끝난다(파일 크기까지 확인)."""
    state = dict(conflict_state)
    state.update(document_agent_node(state))
    update = pdf_generator_node(state)

    pdf_path = update["pdf_path"]
    assert os.path.exists(pdf_path)
    assert os.path.getsize(pdf_path) > 0
    assert Path(pdf_path).parent == Path(pdf_output_dir)
    assert update["ui_actions"][0]["type"] == "OPEN_PDF_PREVIEW"
    assert UNKNOWN_LABEL in update["ui_actions"][0]["notice"]


# ---------------------------------------------------------------------------
# 명세 36·37절 — 미확인 항목 보존
# ---------------------------------------------------------------------------
def test_unknown_fields_are_listed_not_guessed(adapter: FixtureClinicalDataAdapter) -> None:
    """비어 있는 항목은 추측으로 채우지 않고 미확인 목록에 남긴다."""
    profile = adapter.load_pet_profile(DOG_PET_ID)
    profile["allergies"] = ""  # 알레르기 미기재
    state = {
        "pet_profile": profile,
        "priority_pet_context": profile,
        "missing_fields": ["빈도", "증상 변화"],
        "final_risk": "visit",
        "emergency_urgency": "none",
    }
    packet = build_consultation_packet(state)

    assert "빈도" in packet.unknown_fields
    assert "증상 변화" in packet.unknown_fields
    assert "알레르기" in packet.unknown_fields
    assert "allergies" not in packet.medical_history


# ---------------------------------------------------------------------------
# 명세 38절 — Email Draft (전송하지 않는다)
# ---------------------------------------------------------------------------
def test_email_draft_requires_pdf() -> None:
    """PDF 가 없으면 초안 자체를 만들지 않는다(첨부 없는 '자료 첨부' 메일 방지)."""
    assert build_email_draft({"pdf_path": ""}) is None
    assert email_draft_node({"pdf_path": None}) == {"email_draft": None}


def test_email_draft_reuses_pdf_path_exactly(
    conflict_state: dict, pdf_output_dir: Path
) -> None:
    """첨부 경로는 State 의 `pdf_path` 를 그대로 쓴다(명세 40절 8번 검사 대상)."""
    state = dict(conflict_state)
    state.update(document_agent_node(state))
    state.update(pdf_generator_node(state))
    update = email_draft_node(state)

    draft = update["email_draft"]
    assert draft["attachment_path"] == state["pdf_path"]
    assert draft["attachment_filename"] == state["pdf_filename"]
    assert draft["to"] is None, "병원 이메일이 없으면 주소를 지어내지 않는다."
    assert "초코" in draft["subject"]
    assert "AI 참고 분류" in draft["body"]
    assert UNKNOWN_LABEL in draft["body"]
    assert find_forbidden_expressions(draft["body"]) == []

    action = update["ui_actions"][0]
    assert action["type"] == "OPEN_GMAIL_COMPOSE"
    assert action["auto_send"] is False, "compose 화면만 열고 전송하지 않는다."


def test_email_draft_uses_hospital_email_when_available(
    conflict_state: dict, pdf_output_dir: Path
) -> None:
    """검색된 병원 이메일이 있으면 그 값을 그대로 수신자로 쓴다."""
    state = dict(conflict_state)
    state.update(document_agent_node(state))
    state.update(pdf_generator_node(state))
    state["selected_hospital"] = {"hospital": {"name": "강남24시동물병원", "email": "er@example.com"}}

    draft = build_email_draft(state)
    assert draft is not None
    assert draft.to == "er@example.com"


# ---------------------------------------------------------------------------
# 명세 40·43절 — Output Safety
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("text", "expected_marker"),
    [
        ("검사 결과 이 아이는 이첨판 폐쇄부전증입니다.", "확정 진단"),
        ("증상을 보니 심장병으로 진단됩니다.", "확정 진단"),
        ("진단명은 췌장염입니다.", "확정 진단"),
        ("지금 드시던 심장약을 중단하세요.", "중단"),
        ("심장약 복용을 중단하고 지켜보세요.", "중단"),
        ("이 병원은 지금 바로 진료 가능합니다.", "실시간 진료"),
        ("해당 병원은 현재 진료 가능하니 바로 가시면 됩니다.", "실시간 진료"),
    ],
)
def test_forbidden_expressions_are_detected(text: str, expected_marker: str) -> None:
    """확정 진단 / 약 중단 지시 / '현재 진료 가능' 단정은 각각 차단된다(명세 43절)."""
    violations = find_forbidden_expressions(text)
    assert violations, f"차단되어야 할 문장이 통과했습니다: {text}"
    assert any(expected_marker in violation for violation in violations)


@pytest.mark.parametrize(
    "text",
    [
        "이첨판 폐쇄부전증일 가능성이 있어 수의사 확인이 필요합니다.",
        "복용 중인 약은 임의로 중단하지 마시고 수의사와 상의해 주세요.",
        "방문 전에 전화로 현재 진료 및 응급 접수 가능 여부를 확인하세요.",
        "증상이 반복되면 동물병원에서 진료 상담을 받아보시길 권해드립니다.",
    ],
)
def test_safe_expressions_are_not_blocked(text: str) -> None:
    """가능성 표현·투약 금지 안내·전화 확인 안내는 정상 문장이다(오탐 방지)."""
    assert find_forbidden_expressions(text) == []


def test_check_output_regenerates_once_then_falls_back() -> None:
    """금지 표현이 있으면 재생성하고, 두 번째에는 fallback 이다(명세 40절)."""
    state = {
        "draft_response": "검사 결과 이 아이는 췌장염입니다. 심장약을 중단하세요.",
        "final_risk": "visit",
        "emergency_urgency": "none",
        "retry_count": 0,
    }
    first = check_output(state)
    assert first.valid is False
    assert first.action == "regenerate"

    update = output_check_node(state)
    assert update["retry_count"] == 1
    assert update["output_check_action"] == "regenerate"

    state["retry_count"] = 1
    assert check_output(state).action == "fallback"
    assert decide_action(["금지 표현"], 1) == "fallback"


def test_attachment_path_mismatch_is_fatal() -> None:
    """PDF 경로와 이메일 첨부 경로가 다르면 문장을 다시 써도 소용없으므로 fallback."""
    state = {
        "draft_response": "동물병원에서 진료 상담을 받아보시길 권해드립니다.",
        "final_risk": "visit",
        "emergency_urgency": "none",
        "pdf_path": "/tmp/a.pdf",
        "pdf_filename": "a.pdf",
        "email_draft": {
            "to": None,
            "subject": "[병원 상담자료] 초코 / 2026-07-19",
            "body": "본문",
            "attachment_path": "/tmp/b.pdf",
            "attachment_filename": "b.pdf",
        },
        "retry_count": 0,
    }
    result = check_output(state)

    assert result.action == "fallback"
    assert any(error.startswith(FATAL_PREFIX) for error in result.errors)


def test_missing_hospital_verification_notice_is_flagged() -> None:
    """병원을 안내했으면 '전화 확인' 안내가 반드시 함께 있어야 한다(명세 34·40절)."""
    state = {
        "draft_response": "근처 동물병원에서 진료 상담을 받아보세요.",
        "final_risk": "visit",
        "emergency_urgency": "none",
        "hospital_results": [
            {"hospital": {"name": "강남24시동물병원", "phone": "02-987-6543"}, "score": 65}
        ],
        "retry_count": 0,
    }
    errors = check_output(state).errors
    assert any("전화 확인" in error for error in errors)


def test_critical_case_without_contact_action_is_flagged() -> None:
    """즉시 위급인데 연락 action 이 없으면 오류로 잡는다(명세 40절)."""
    state = {
        "draft_response": "지금 즉시 동물병원에 연락해 주세요.",
        "final_risk": "emergency",
        "emergency_urgency": "critical_immediate",
        "ui_actions": [],
        "retry_count": 0,
    }
    errors = check_output(state).errors
    assert any("연락 action" in error for error in errors)


# ---------------------------------------------------------------------------
# 명세 39·40절 — Final Safety / 결과 조립
# ---------------------------------------------------------------------------
def test_final_safety_replaces_unsafe_answer_but_keeps_actions() -> None:
    """위험한 문장은 통째로 안전 문구로 바꾸되 응급 action 은 지우지 않는다."""
    state = {
        "draft_response": "심장병으로 진단됩니다. 심장약을 중단하세요.",
        "final_risk": "emergency",
        "emergency_urgency": "critical_immediate",
        "ui_actions": [{"type": "CALL_HOSPITAL", "phone": "02-987-6543"}],
    }
    update = final_safety_node(state)

    assert update["fallback_used"] is True
    assert update["final_response"] == safe_fallback_message("emergency")
    assert "ui_actions" not in update, "action 채널을 건드리면 전화 버튼이 사라질 수 있다."
    assert find_forbidden_expressions(update["final_response"]) == []
    # fallback 이어도 행동 안내는 남긴다.
    assert "병원" in update["final_response"]
    assert "즉시" in update["final_response"]


def test_final_safety_passes_safe_answer_with_disclaimer() -> None:
    """안전한 답변은 그대로 통과시키되 의료 고지를 반드시 붙인다."""
    from petcare_ai.graph.prompts import MEDICAL_DISCLAIMER

    state = {
        "draft_response": "증상이 반복되면 동물병원에서 진료 상담을 받아보시길 권해드립니다.",
        "final_risk": "visit",
        "emergency_urgency": "none",
    }
    update = final_safety_node(state)

    assert update["fallback_used"] is False
    assert update["final_response"].startswith(state["draft_response"])
    assert MEDICAL_DISCLAIMER in update["final_response"]


def test_build_chat_graph_result_maps_state_without_leaking_personal_data(
    conflict_state: dict, pdf_output_dir: Path
) -> None:
    """최종 결과 조립과 trace metadata 계약(명세 39·42절)."""
    state = dict(conflict_state)
    state.update(document_agent_node(state))
    state.update(pdf_generator_node(state))
    state.update(email_draft_node(state))
    state["final_response"] = "동물병원에서 진료 상담을 받아보시길 권해드립니다."
    state["missing_fields"] = ["빈도"]

    result = build_chat_graph_result(state)

    assert result.risk_level == "visit"
    assert result.pdf_path == state["pdf_path"]
    assert result.email_draft is not None
    assert result.email_draft.attachment_path == state["pdf_path"]
    assert result.missing_information == ["빈도"]
    assert all(action.get("type") for action in result.ui_actions)

    # trace 에는 원문·개인 식별정보를 넣지 않는다.
    metadata = result.trace_metadata
    assert metadata["final_risk"] == "visit"
    blob = " ".join(str(value) for value in metadata.values())
    assert "초코" not in blob
    assert WEIGHT_MESSAGE not in blob


def test_generate_consultation_pdf_rejects_non_packet(pdf_output_dir: Path) -> None:
    """packet 타입 계약을 지키지 않으면 조용히 넘어가지 않는다."""
    with pytest.raises(TypeError):
        generate_consultation_pdf({"document_type": "visit_consultation"})  # type: ignore[arg-type]

    packet = ConsultationPacket(
        document_type="visit_consultation",
        generated_at="2026-07-19T09:00:00",
        pet={"name": "초코", "species": "dog"},
    )
    pdf_path, filename = generate_consultation_pdf(packet)
    assert os.path.getsize(pdf_path) > 0
    assert filename.endswith(".pdf")
