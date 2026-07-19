"""Visit Subgraph — 명세 31절 mermaid 를 그대로 구현한다.

```mermaid
flowchart TD
    A[Visit] --> B[Select Relevant Clinical Records]
    B --> C[Missing Information Agent]
    C --> D{Enough for consultation packet?}
    D -->|No| E[Interrupt]
    E --> C
    D -->|Yes| F[Document Agent]
    F --> G[Packet Validator]
    G --> H[PDF Generator]
    H --> I[Email Draft Agent]
    I --> J[Visit Result]
```

## 설계 메모

**문서 체인(Document → Packet Validator → PDF → Email)은 응급 서브그래프와 완전히
같다(명세 31·32절).** 그래서 이 파일이 그 네 node 의 팩토리를 소유하고
`emergency.py` 가 가져다 쓴다. 같은 체인을 두 번 구현하면 한쪽만 고쳐지는
사고가 난다 — 특히 PDF 경로와 이메일 첨부 경로가 어긋나면 Output Check 가
치명 오류로 잡는다(명세 40절 8번).

**Packet Validator / PDF Generator / Email Draft 는 `resolve_optional_node` 로
해석한다.** 명세 22절 node 모듈 목록에 `packet_validator.py` 와
`pdf_generator.py` 가 없어서, 전용 node 가 있으면 그쪽을 쓰고 없으면 이 파일의
기본 구현을 쓴다. 나중에 전용 node 가 생기면 자동으로 그쪽이 이긴다.

**Document Agent 는 필수(`resolve_node`)다.** 일기장·진단서를 packet 으로
매핑하는 일은 도메인 지식이 필요해 여기서 대충 대체할 수 없다.

**LLM 은 이 서브그래프의 어디에도 필수가 아니다.** 기록 선택은 규칙 기반이고,
PDF 는 고정 template 이며(명세 37절 "LLM 이 자유롭게 전체 PDF 문장을 작성하지
않음"), 이메일 본문도 짧은 고정 template 이다(명세 38절).
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Any

from ..routers import route_missing_info, route_rag_status
from ..state import Replace, ReplaceDict
from . import (
    NodeFn,
    SubgraphDeps,
    resolve_node,
    resolve_optional_node,
    strip_internal_keys,
    with_sanitized_collected,
)

logger = logging.getLogger(__name__)

__all__ = [
    "build_visit_subgraph",
    "NODE_SELECT_CLINICAL_RECORDS",
    "NODE_MISSING_INFORMATION",
    "NODE_MISSING_INFORMATION_INTERRUPT",
    "NODE_DOCUMENT_AGENT",
    "NODE_PACKET_VALIDATOR",
    "NODE_PDF_GENERATOR",
    "NODE_EMAIL_DRAFT",
    "NODE_VISIT_RESULT",
    "make_select_clinical_records_node",
    "make_document_chain",
    "packet_validator_node",
    "make_pdf_generator_node",
    "make_email_draft_node",
    "visit_result_node",
]

NODE_SELECT_CLINICAL_RECORDS = "select_clinical_records"
NODE_MISSING_INFORMATION = "missing_information"
NODE_MISSING_INFORMATION_INTERRUPT = "missing_information_interrupt"
NODE_DOCUMENT_AGENT = "document_agent"
NODE_PACKET_VALIDATOR = "packet_validator"
NODE_PDF_GENERATOR = "pdf_generator"
NODE_EMAIL_DRAFT = "email_draft"
NODE_VISIT_RESULT = "visit_result"

#: 명세 36절: 없는 정보는 추측하지 않고 이 문구로 표시한다.
UNKNOWN_MARK = "미확인"

#: ConsultationPacket 에 반드시 있어야 하는 상위 항목(명세 36절).
REQUIRED_PACKET_SECTIONS: tuple[str, ...] = (
    "pet",
    "medical_history",
    "current_condition",
)


# ---------------------------------------------------------------------------
# B. Select Relevant Clinical Records
# ---------------------------------------------------------------------------
def make_select_clinical_records_node(deps: SubgraphDeps) -> NodeFn:
    """진료 자료에 넣을 진단서·일기 기록을 고르는 node 를 만든다(명세 31절 B).

    선택 로직은 `nodes/clinical_context_priority.py` 의 함수를 그대로 쓴다.
    거기가 명세 20절 우선순위(PET DB > 진단서 DB > 일기장 DB)를 이미 구현하고
    있고, 같은 규칙이 두 벌 존재하면 PDF 와 답변이 서로 다른 근거를 인용하게 된다.

    **레코드를 요약하거나 다시 파싱하지 않는다**(명세 36절). 원문을 그대로
    골라서 넘긴다.
    """

    def _node(state: dict) -> dict:
        from ..nodes.clinical_context_priority import (  # noqa: PLC0415
            select_related_diagnoses,
            select_supporting_daily_entries,
        )

        observation = dict(state.get("current_observation") or {})
        pet_profile = dict(state.get("pet_profile") or {})
        diagnoses = list(state.get("diagnoses") or [])
        daily_entries = list(state.get("daily_entries") or [])

        selected_diagnoses, reasons = select_related_diagnoses(
            diagnoses, observation, pet_profile
        )
        selected_entries = select_supporting_daily_entries(daily_entries, observation)

        updates: dict[str, Any] = {"document_type": "visit_consultation"}
        # 아무것도 못 골랐으면 State 를 건드리지 않는다 — Clinical Context Priority
        # 가 앞서 넣어 둔 값을 빈 리스트로 덮어쓰면 정보가 사라진다.
        if selected_diagnoses:
            updates["related_diagnoses"] = Replace(selected_diagnoses)
        if selected_entries:
            updates["supporting_daily_entries"] = Replace(selected_entries)

        logger.info(
            "Select Clinical Records: 진단서 %d건 / 일기 %d건 (%s)",
            len(selected_diagnoses),
            len(selected_entries),
            "; ".join(reasons[:3]),
        )
        return updates

    return _node


# ---------------------------------------------------------------------------
# G. Packet Validator (기본 구현)
# ---------------------------------------------------------------------------
def packet_validator_node(state: dict) -> dict:
    """Consultation Packet 의 구조를 점검하고 **빈 항목을 `미확인` 으로 채운다**.

    이 node 는 값을 지어내지 않는다. 하는 일은 두 가지뿐이다.

    1. 명세 36절이 요구한 상위 항목이 빠졌으면 빈 값으로 채워 PDF 생성이
       구조적으로 실패하지 않게 한다.
    2. 아직 못 받은 정보를 `unknown_fields` 에 모은다. `missing_fields` 는
       Missing Information Agent 가 되묻고도 못 받은 항목이라, 그대로 두면
       PDF 에서 조용히 사라진다. 명세 36절은 이것을 `미확인` 으로 **표시하라고**
       요구한다.

    검증 실패를 `validation_errors` 에 넣지 않는 이유: 그 채널은 Output Check 가
    매 검사마다 통째로 교체한다(`Replace`). 여기서 넣어도 최종 검사 때 지워지므로
    로그로 남기고 packet 자체를 고치는 편이 실제로 도움이 된다.
    """
    packet = state.get("consultation_packet")
    if not isinstance(packet, dict) or not packet:
        logger.error("Consultation Packet 이 비어 있습니다. Document Agent 출력을 확인하세요.")
        return {}

    fixed = dict(packet)
    for section in REQUIRED_PACKET_SECTIONS:
        value = fixed.get(section)
        if not isinstance(value, dict):
            logger.warning("packet 에 '%s' 항목이 없어 빈 값으로 채웁니다.", section)
            fixed[section] = {}

    if not fixed.get("document_type"):
        fixed["document_type"] = state.get("document_type") or "visit_consultation"
    if not fixed.get("generated_at"):
        from datetime import datetime  # noqa: PLC0415

        fixed["generated_at"] = datetime.now().isoformat(timespec="seconds")

    unknown = [str(item) for item in (fixed.get("unknown_fields") or [])]
    for label in state.get("missing_fields") or []:
        text = str(label).strip()
        if text and text not in unknown:
            unknown.append(text)
    fixed["unknown_fields"] = unknown

    logger.info(
        "Packet Validator: type=%s 미확인 %d건",
        fixed.get("document_type"),
        len(unknown),
    )
    return {"consultation_packet": fixed}


# ---------------------------------------------------------------------------
# H. PDF Generator (기본 구현)
# ---------------------------------------------------------------------------
def _packet_model(state: dict) -> Any | None:
    """State 의 packet dict 를 `ConsultationPacket` 으로 되돌린다(실패하면 None)."""
    from ...schemas import ConsultationPacket  # noqa: PLC0415

    raw = state.get("consultation_packet")
    if not isinstance(raw, dict) or not raw:
        return None
    try:
        return ConsultationPacket(**raw)
    except Exception as exc:
        logger.error("Consultation Packet 스키마가 올바르지 않습니다: %s", exc)
        return None


def make_pdf_generator_node(deps: SubgraphDeps) -> NodeFn:
    """PDF 생성 node 를 만든다(명세 37절).

    실제 생성은 `pdf/consultation_pdf.generate_consultation_pdf()` 가 한다 —
    고정 template, 한국어 폰트 탐색, 파일 크기 검증까지 거기 있다. 이 node 는
    State ↔ 스키마 변환과 UI action 생성만 담당하는 얇은 껍데기다.

    생성 실패는 예외로 올리지 않는다. PDF 가 없어도 상담 답변 자체는 나가야 하고,
    Output Check 가 "첨부 경로 불일치" 로 잡아 준다(명세 40절 8번).
    """

    def _node(state: dict) -> dict:
        packet = _packet_model(state)
        if packet is None:
            return {}

        from ...pdf.consultation_pdf import generate_consultation_pdf  # noqa: PLC0415

        try:
            pdf_path, filename = generate_consultation_pdf(
                packet, settings=deps.resolved_settings()
            )
        except Exception as exc:
            logger.error("PDF 생성 실패 — PDF 없이 진행합니다: %s", exc)
            return {}

        logger.info("PDF Generator: %s", pdf_path)
        return {
            "pdf_path": pdf_path,
            "pdf_filename": filename,
            "ui_actions": [{"type": "OPEN_PDF_PREVIEW", "pdf_path": pdf_path}],
        }

    return _node


# ---------------------------------------------------------------------------
# I. Email Draft Agent (기본 구현)
# ---------------------------------------------------------------------------
def _pet_name(state: dict) -> str:
    """PDF·이메일 제목에 쓸 반려동물 이름(없으면 '반려동물')."""
    profile = state.get("pet_profile") or {}
    name = str(profile.get("name") or profile.get("pet_name") or "").strip()
    return name or "반려동물"


def _hospital_email(state: dict) -> str | None:
    """병원 이메일을 찾는다 — 없으면 `None`(명세 38절: 지어내지 않는다)."""
    selected = state.get("selected_hospital")
    if isinstance(selected, dict) and selected.get("email"):
        return str(selected["email"])
    for item in state.get("hospital_results") or []:
        if not isinstance(item, dict):
            continue
        hospital = item.get("hospital") if isinstance(item.get("hospital"), dict) else item
        email = hospital.get("email") if isinstance(hospital, dict) else None
        if email:
            return str(email)
    return None


def make_email_draft_node(deps: SubgraphDeps) -> NodeFn:
    """이메일 초안 node 를 만든다(명세 38절) — **실제로 보내지 않는다.**

    Android 가 Gmail compose 화면을 열 때 채워 넣을 정보만 만든다. 본문은 짧은
    고정 template 이다. LLM 으로 본문을 생성하지 않는 이유: 이메일은 사람이 읽고
    보내기 전에 수정하는 문서인데, 생성문이 섞이면 사실과 다른 문장이 보호자 이름
    으로 병원에 나갈 수 있다.

    첨부 경로는 반드시 State 의 `pdf_path` 를 그대로 쓴다. 다시 조립하면 Output
    Check 의 경로 일치 검사(명세 40절 8번)에서 치명 오류가 난다.
    """

    def _node(state: dict) -> dict:
        pdf_path = state.get("pdf_path")
        pdf_filename = state.get("pdf_filename")
        if not pdf_path or not pdf_filename:
            logger.warning("PDF 가 없어 이메일 초안을 만들지 않습니다.")
            return {}

        from ...schemas import EmailDraft  # noqa: PLC0415

        emergency = str(state.get("document_type") or "") == "emergency_consultation"
        label = "응급 상담자료" if emergency else "진료 상담자료"
        name = _pet_name(state)
        today = date.today().isoformat()

        body_lines = [
            "안녕하세요.",
            f"{name} 보호자입니다. 진료 상담을 위해 정리한 자료를 첨부합니다.",
            "",
            "첨부한 자료에는 현재 증상, 증상 시작·빈도·변화, 기존 질병과 복용약,",
            "관련 진단 기록과 최근 생활 기록이 정리되어 있습니다.",
            "확인되지 않은 항목은 '미확인' 으로 표시했습니다.",
            "",
            "자료는 보호자가 정리한 참고자료이며 진단이 아닙니다.",
            "진료 가능 여부와 방문 시간은 전화로 확인하겠습니다.",
            "",
            "감사합니다.",
        ]

        draft = EmailDraft(
            to=_hospital_email(state),
            subject=f"[{label}] {name} / {today}",
            body="\n".join(body_lines),
            attachment_path=str(pdf_path),
            attachment_filename=str(pdf_filename),
        )
        logger.info("Email Draft: to=%s subject=%s", draft.to, draft.subject)
        return {
            "email_draft": draft.model_dump(),
            "ui_actions": [{"type": "OPEN_GMAIL_COMPOSE"}],
        }

    return _node


# ---------------------------------------------------------------------------
# 문서 체인 — visit / emergency 공용
# ---------------------------------------------------------------------------
def make_document_chain(deps: SubgraphDeps) -> dict[str, NodeFn]:
    """Document → Packet Validator → PDF → Email 네 node 를 해석해 돌려준다.

    반환 dict 의 key 는 graph node 이름이므로 호출자가
    `for name, fn in chain.items(): graph.add_node(name, fn)` 로 그대로 붙이면 된다.
    응급 서브그래프(명세 32절)도 같은 체인을 쓴다.
    """
    return {
        # Document Agent 는 `collected_information` 의 모든 항목을 진료 자료로 옮겨
        # 담는다. 되묻기 횟수 같은 내부 key 가 PDF 에 인쇄되지 않도록 입력을 정리해
        # 넘긴다.
        NODE_DOCUMENT_AGENT: with_sanitized_collected(
            resolve_node(
                deps,
                NODE_DOCUMENT_AGENT,
                ("document_agent",),
                ("document_agent_node",),
                factories=("make_document_agent_node",),
            )
        ),
        NODE_PACKET_VALIDATOR: resolve_optional_node(
            deps,
            NODE_PACKET_VALIDATOR,
            ("document_agent",),
            ("packet_validator_node", "validate_packet_node"),
            fallback=packet_validator_node,
        ),
        NODE_PDF_GENERATOR: resolve_optional_node(
            deps,
            NODE_PDF_GENERATOR,
            ("document_agent", "email_draft"),
            ("pdf_generator_node", "generate_pdf_node"),
            fallback=make_pdf_generator_node(deps),
        ),
        NODE_EMAIL_DRAFT: resolve_optional_node(
            deps,
            NODE_EMAIL_DRAFT,
            ("email_draft",),
            ("email_draft_node",),
            fallback=make_email_draft_node(deps),
            factories=("make_email_draft_node",),
        ),
    }


# ---------------------------------------------------------------------------
# J. Visit Result
# ---------------------------------------------------------------------------
def visit_result_node(state: dict) -> dict:
    """병원 상담 권고 결과 메시지를 만든다(명세 31절 J).

    고정 template 으로 조립한다 — 이 메시지는 **판단 결과의 전달**이지 생성이
    아니다. 확정 진단·처방 문구를 넣지 않고, Output Check(명세 40절 6번)가
    visit 답변에서 요구하는 '병원 / 진료' 안내를 반드시 포함한다.

    수집하지 못한 정보는 숨기지 않고 그대로 알린다. 보호자가 병원에서 그 항목을
    질문받을 것이기 때문이다.
    """
    from ..nodes.health_response import RISK_GUIDANCE  # noqa: PLC0415
    from ..prompts import MEDICAL_DISCLAIMER  # noqa: PLC0415

    lines: list[str] = ["정리해 드린 내용을 확인해 주세요.", ""]

    reasons = [str(item) for item in (state.get("risk_reasons") or [])][:4]
    red_flags = [str(item) for item in (state.get("red_flags") or [])][:4]
    if red_flags:
        lines.append("확인된 신호: " + ", ".join(red_flags))
    for reason in reasons:
        lines.append(f"- {reason}")
    if red_flags or reasons:
        lines.append("")

    lines.append(RISK_GUIDANCE.get(str(state.get("final_risk") or "visit"), RISK_GUIDANCE["visit"]))
    lines.append("")

    unknown = [str(item) for item in (state.get("missing_fields") or [])]
    if unknown:
        lines.append(
            "아직 확인되지 않은 항목(" + ", ".join(unknown) + ")은 '미확인' 으로 표시했습니다."
        )

    if state.get("pdf_path"):
        lines.append(
            f"병원에 보여 드릴 상담 자료를 PDF 로 정리했습니다: {state.get('pdf_filename')}"
        )
    if state.get("email_draft"):
        lines.append("이메일로 미리 보내실 수 있도록 초안도 함께 준비했습니다.")

    lines.append("")
    lines.append(MEDICAL_DISCLAIMER)

    message = "\n".join(lines).strip()
    return {
        "draft_response": message,
        "missing_information_question": "",
        # `ReplaceDict` 로 감싸야 얕은 병합 reducer 를 우회해 내부 key(되묻기 횟수)가
        # 실제로 사라진다. 상담이 끝난 시점이라 라운드 카운터를 남길 이유가 없고,
        # 남겨 두면 다음 turn 의 되묻기 한도가 이미 소진된 상태로 시작한다.
        "collected_information": ReplaceDict(
            strip_internal_keys(state.get("collected_information"))
        ),
    }


# ---------------------------------------------------------------------------
# 서브그래프 조립
# ---------------------------------------------------------------------------
def build_visit_subgraph(deps: SubgraphDeps | None = None) -> Any:
    """명세 31절 Visit Subgraph 를 compile 해서 돌려준다.

    Args:
        deps: 주입 컨테이너. `None` 이면 LLM 없이 규칙 기반으로 동작한다.
    """
    from langgraph.graph import END, START, StateGraph  # noqa: PLC0415

    from ..state import PetCareState  # noqa: PLC0415

    resolved = deps or SubgraphDeps()
    graph = StateGraph(PetCareState)

    from . import make_missing_information_gate  # noqa: PLC0415

    graph.add_node(NODE_SELECT_CLINICAL_RECORDS, make_select_clinical_records_node(resolved))
    graph.add_node(
        NODE_MISSING_INFORMATION,
        make_missing_information_gate(
            resolved,
            NODE_MISSING_INFORMATION,
            ("missing_information",),
            ("missing_information_node",),
        ),
    )
    graph.add_node(
        NODE_MISSING_INFORMATION_INTERRUPT,
        resolve_node(
            resolved,
            NODE_MISSING_INFORMATION_INTERRUPT,
            ("missing_information",),
            ("missing_information_interrupt_node",),
        ),
    )
    for name, node in make_document_chain(resolved).items():
        graph.add_node(name, node)
    graph.add_node(NODE_VISIT_RESULT, visit_result_node)

    # RAG 근거 수집 — health 서브그래프의 체인을 **그대로 재사용**한다.
    #
    # 원래 명세 23절은 RAG 계열 노드를 '일반 상담' 그룹에만 배치했다. 그런데 명세
    # 39절 `ChatGraphResult` 는 모든 경로에서 `evidence` 를 돌려주게 돼 있고, 33절
    # Hospital Requirement Builder 는 입력으로 'RAG 또는 검증된 웹 지식 보조' 를
    # 명시한다. 즉 명세 안에서 앞뒤가 맞지 않는다.
    #
    # 실제 결과도 그랬다 — 병원 상담 권고 문서에 "왜 가야 하는지" 의 근거가 늘
    # 비어 있었다(citations 가 항상 0건). 병원에 전달하는 자료일수록 근거가 필요하다.
    # 새 노드를 만들지 않고 같은 함수를 재사용하므로 판정 기준이 갈라지지 않는다.
    from .health import (  # noqa: PLC0415  — 순환 import 를 피해 지연 로드
        NODE_CONFLICT_HANDLER,
        NODE_EVIDENCE_MERGE,
        NODE_KNOWLEDGE_SUFFICIENCY,
        NODE_TAVILY_VET_SEARCH,
        NODE_VETERINARY_RAG,
        NODE_WEB_SOURCE_VALIDATOR,
        conflict_handler_node,
        evidence_merge_node,
        make_knowledge_sufficiency_node,
        make_tavily_vet_search_node,
        make_veterinary_rag_node,
        make_web_source_validator_node,
    )

    graph.add_node(NODE_VETERINARY_RAG, make_veterinary_rag_node(resolved))
    graph.add_node(NODE_KNOWLEDGE_SUFFICIENCY, make_knowledge_sufficiency_node(resolved))
    graph.add_node(NODE_TAVILY_VET_SEARCH, make_tavily_vet_search_node(resolved))
    graph.add_node(NODE_WEB_SOURCE_VALIDATOR, make_web_source_validator_node(resolved))
    graph.add_node(NODE_CONFLICT_HANDLER, conflict_handler_node)
    graph.add_node(NODE_EVIDENCE_MERGE, evidence_merge_node)

    graph.add_edge(START, NODE_SELECT_CLINICAL_RECORDS)
    graph.add_edge(NODE_SELECT_CLINICAL_RECORDS, NODE_MISSING_INFORMATION)

    # D: Enough for consultation packet?
    graph.add_conditional_edges(
        NODE_MISSING_INFORMATION,
        route_missing_info,
        {
            "ask": NODE_MISSING_INFORMATION_INTERRUPT,
            # 문서 작성 전에 근거를 모은다 — 상담 자료에 "왜" 가 들어가야 한다.
            "ready": NODE_VETERINARY_RAG,
        },
    )
    graph.add_edge(NODE_MISSING_INFORMATION_INTERRUPT, NODE_MISSING_INFORMATION)

    # RAG status 분기 — health 서브그래프와 동일한 판정 함수를 쓴다.
    graph.add_edge(NODE_VETERINARY_RAG, NODE_KNOWLEDGE_SUFFICIENCY)
    graph.add_conditional_edges(
        NODE_KNOWLEDGE_SUFFICIENCY,
        route_rag_status,
        {
            "sufficient": NODE_EVIDENCE_MERGE,
            "insufficient": NODE_TAVILY_VET_SEARCH,
            "conflicting": NODE_CONFLICT_HANDLER,
        },
    )
    graph.add_edge(NODE_TAVILY_VET_SEARCH, NODE_WEB_SOURCE_VALIDATOR)
    graph.add_edge(NODE_WEB_SOURCE_VALIDATOR, NODE_EVIDENCE_MERGE)
    graph.add_edge(NODE_CONFLICT_HANDLER, NODE_EVIDENCE_MERGE)
    graph.add_edge(NODE_EVIDENCE_MERGE, NODE_DOCUMENT_AGENT)

    graph.add_edge(NODE_DOCUMENT_AGENT, NODE_PACKET_VALIDATOR)
    graph.add_edge(NODE_PACKET_VALIDATOR, NODE_PDF_GENERATOR)
    graph.add_edge(NODE_PDF_GENERATOR, NODE_EMAIL_DRAFT)
    graph.add_edge(NODE_EMAIL_DRAFT, NODE_VISIT_RESULT)
    graph.add_edge(NODE_VISIT_RESULT, END)

    try:
        return graph.compile(name="visit_subgraph", **resolved.compile_kwargs())
    except TypeError:  # 구버전 langgraph 호환
        return graph.compile(**resolved.compile_kwargs())
