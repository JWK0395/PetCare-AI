"""Emergency Subgraph — 명세 32절 mermaid 를 그대로 구현한다.

```mermaid
flowchart TD
    A[Emergency Entry] --> B[Immediate Emergency Message]
    B --> C1[Clinical Context Priority]
    B --> C2[Check Region Input]
    C1 --> D[Hospital Requirement Builder]
    C2 --> E{Region exists?}
    E -->|No| F[REQUEST_LOCATION result]
    E -->|Yes| G[Hospital Search Agent]
    D --> G
    G --> H[Parse Hospital Results]
    H --> I[Hospital Suitability Agent]
    C1 --> J[Contact Minimum Information]
    J --> K{Critical immediate?}
    K -->|Yes| L[Prepare CALL_HOSPITAL action]
    K -->|No| M{Minimum info ready?}
    M -->|No| N[Interrupt]
    N --> J
    M -->|Yes| O[Document Agent]
    L --> O
    O --> P[PDF Generator]
    P --> Q[Email Draft Agent]
    I --> R[Build Emergency Result]
    Q --> R
```

## mermaid 를 그래프로 옮기면서 조정한 두 가지 (이유 포함)

**1) `Region exists?` 분기를 Hospital Requirement Builder 뒤로 옮겼다.**
원본대로 `C2 → G` 와 `D → G` 를 둘 다 그리면, `G`(Hospital Search)가 서로 다른
superstep 에 **두 번 트리거된다**(C2 는 깊이 2, D 는 깊이 3). Tavily 병원 검색이
두 번 호출되고 결과가 중복된다. 그래서 `C1`·`C2` 를 Requirement Builder 로 합류
시킨 뒤 거기서 한 번만 분기한다. 판정 시점만 한 칸 늦어질 뿐, "지역이 없으면
병원을 검색하지 않고 `REQUEST_LOCATION` 을 돌려준다" 는 규칙은 그대로다.

**2) `Build Emergency Result` 를 멱등(idempotent)하게 만들었다.**
병원 branch 와 문서 branch 가 fan-in 하는데, 최소정보 interrupt 가 끼면 두 branch
가 서로 다른 superstep 에 도착해 이 node 가 두 번 실행될 수 있다. 그래서 이
node 는 **State 만 읽어 결과를 다시 조립**한다. 몇 번 실행돼도 같은 결과가 되고,
중간 결과를 누적하지 않는다.

## 안전 규칙 (명세 32·34·47절)

- 지역을 모르면 병원을 **추측해서 만들지 않는다.** `REQUEST_LOCATION` 을 돌려준다.
- 검색 결과만으로 실시간 진료 가능 여부를 단정하지 않는다. 모든 병원 안내에
  "방문 전 전화 확인" 문구를 붙인다.
- `critical_immediate` 면 정보 수집이 전화 action 을 막지 못한다.
- 확정 진단·처방 문구를 만들지 않는다. 이 서브그래프의 메시지는 전부 고정
  template 조립이다.
"""

from __future__ import annotations

import logging
from typing import Any

from ..routers import route_emergency_contact, route_region
from . import SubgraphDeps, resolve_node, resolve_optional_node
from .visit import (
    NODE_DOCUMENT_AGENT,
    NODE_EMAIL_DRAFT,
    NODE_PACKET_VALIDATOR,
    NODE_PDF_GENERATOR,
    make_document_chain,
)

logger = logging.getLogger(__name__)

__all__ = [
    "build_emergency_subgraph",
    "NODE_IMMEDIATE_MESSAGE",
    "NODE_CLINICAL_CONTEXT",
    "NODE_CHECK_REGION",
    "NODE_HOSPITAL_REQUIREMENTS",
    "NODE_HOSPITAL_SEARCH",
    "NODE_PARSE_HOSPITAL_RESULTS",
    "NODE_HOSPITAL_SUITABILITY",
    "NODE_REQUEST_LOCATION",
    "NODE_CONTACT_MINIMUM_INFORMATION",
    "NODE_CONTACT_INTERRUPT",
    "NODE_PREPARE_CALL_HOSPITAL",
    "NODE_BUILD_EMERGENCY_RESULT",
    "immediate_emergency_message_node",
    "check_region_node",
    "request_location_node",
    "prepare_call_hospital_node",
    "build_emergency_result_node",
]

NODE_IMMEDIATE_MESSAGE = "immediate_emergency_message"
NODE_CLINICAL_CONTEXT = "clinical_context_priority"
NODE_CHECK_REGION = "check_region"
NODE_HOSPITAL_REQUIREMENTS = "hospital_requirements"
NODE_HOSPITAL_SEARCH = "hospital_search"
NODE_PARSE_HOSPITAL_RESULTS = "parse_hospital_results"
NODE_HOSPITAL_SUITABILITY = "hospital_suitability"
NODE_REQUEST_LOCATION = "request_location"
NODE_CONTACT_MINIMUM_INFORMATION = "contact_minimum_information"
NODE_CONTACT_INTERRUPT = "contact_minimum_information_interrupt"
NODE_PREPARE_CALL_HOSPITAL = "prepare_call_hospital"
NODE_BUILD_EMERGENCY_RESULT = "build_emergency_result"

#: 최종 안내에 노출할 병원 최대 개수. 응급 상황에서 목록이 길면 오히려 선택이 늦어진다.
MAX_HOSPITALS_IN_MESSAGE = 3

#: 명세 32절: 즉시 안내 문구는 생성하지 않고 고정 template 을 쓴다.
#: Output Check(명세 40절 6번)가 emergency 답변에서 요구하는 '병원' 과
#: '즉시' 를 반드시 포함한다.
IMMEDIATE_EMERGENCY_MESSAGE = (
    "지금 상황은 응급일 수 있습니다. 다른 설명보다 먼저 안내드립니다.\n"
    "즉시 가까운 동물병원 또는 24시간 동물병원에 전화해 현재 상태를 설명하고\n"
    "지금 바로 갈 수 있는지 확인해 주세요.\n"
    "\n"
    "이동 준비를 하시면서 아이의 호흡, 의식·반응 상태, 움직임을 계속 지켜봐 주세요.\n"
    "보호자 판단으로 약을 먹이거나 토하게 하지 마세요. 수의사 지시를 먼저 받으셔야 합니다."
)

#: 지역 정보가 없을 때의 안내. 병원을 지어내는 대신 위치를 요청한다.
REQUEST_LOCATION_MESSAGE = (
    "주변 동물병원을 찾아드리려면 현재 지역 정보가 필요합니다.\n"
    "위치 정보를 허용해 주시거나 지역명(예: 서울 강남구)을 알려 주세요.\n"
    "기다리는 동안에도 가까운 동물병원에 먼저 전화해 주시는 것이 가장 빠릅니다."
)


# ---------------------------------------------------------------------------
# B. Immediate Emergency Message
# ---------------------------------------------------------------------------
def immediate_emergency_message_node(state: dict) -> dict:
    """응급 서브그래프에 들어오자마자 **먼저** 안내 문구를 확정한다(명세 32절 B).

    병원 검색·문서 생성보다 앞에 두는 이유: 뒤 단계가 실패하거나 오래 걸려도
    보호자가 지금 무엇을 해야 하는지는 이미 State 에 들어가 있어야 하기 때문이다.
    `draft_response` 를 여기서 채워 두면 최악의 경우에도 이 문구가 남는다.

    위험도와 긴급도는 State reducer 가 상향 전용이라(명세 28절) 여기서 올린 값이
    뒤 node 에 의해 낮아지지 않는다.
    """
    red_flags = [str(item) for item in (state.get("red_flags") or [])][:5]
    lines = [IMMEDIATE_EMERGENCY_MESSAGE]
    if red_flags:
        lines += ["", "확인된 응급 신호: " + ", ".join(red_flags)]

    logger.warning("Emergency Subgraph 진입: red_flags=%s", red_flags)
    return {
        "draft_response": "\n".join(lines),
        "final_risk": "emergency",
        "emergency_urgency": "contact_ready",
        "document_type": "emergency_consultation",
    }


# ---------------------------------------------------------------------------
# C2. Check Region Input
# ---------------------------------------------------------------------------
def check_region_node(state: dict) -> dict:
    """지역 입력을 정규화한다(명세 32절 C2).

    실제 위치 획득은 구현하지 않는다(명세 32절: "실제 Android 위치 획득 ... 은
    구현하지 않는다"). Colab 입력의 `region_name` 앞뒤 공백만 정리하고, 비어 있는
    문자열은 `None` 으로 통일한다 — `" "` 같은 값이 "지역 있음" 으로 통과하면
    빈 검색어로 Tavily 를 호출하게 된다.
    """
    raw = state.get("region_name")
    region = raw.strip() if isinstance(raw, str) else None
    normalized = region or None

    if normalized is None:
        logger.info("지역 정보 없음 — 병원 검색 대신 위치 요청 경로로 갑니다.")
    return {"region_name": normalized}


# ---------------------------------------------------------------------------
# F. REQUEST_LOCATION result
# ---------------------------------------------------------------------------
def request_location_node(state: dict) -> dict:
    """지역을 모를 때 위치 요청 결과를 만든다(명세 32절 F).

    병원 이름·전화번호를 추측해서 채우지 않는다. 응급 상황에서 존재하지 않는
    병원으로 전화를 걸게 만드는 것이 이 시스템의 최악 실패다.

    `REQUEST_LOCATION` action 은 `critical_immediate` 일 때 Output Check 가 요구하는
    '연락 action' 조건도 함께 만족시킨다(명세 40절).
    """
    logger.info("REQUEST_LOCATION 결과를 반환합니다(지역 정보 없음).")
    return {
        "ui_actions": [{"type": "REQUEST_LOCATION", "reason": "주변 동물병원 검색"}],
        "hospital_results": [],
    }


# ---------------------------------------------------------------------------
# L. Prepare CALL_HOSPITAL action
# ---------------------------------------------------------------------------
def _hospital_records(state: dict) -> list[dict[str, Any]]:
    """`hospital_results` 를 병원 dict 목록으로 평탄화한다.

    Suitability node 는 `{"hospital": {...}, "score": ...}` 형태를 돌려주지만,
    검색 직후의 raw 결과는 병원 dict 그대로일 수 있다. 두 형태를 모두 받는다.
    """
    records: list[dict[str, Any]] = []
    for item in state.get("hospital_results") or []:
        if not isinstance(item, dict):
            continue
        inner = item.get("hospital")
        records.append(inner if isinstance(inner, dict) else item)
    return records


def _best_phone(state: dict) -> tuple[str | None, str]:
    """전화할 병원의 (전화번호, 이름)을 고른다. 없으면 `(None, "")`.

    `hospital_results` 는 Suitability node 가 점수순으로 정렬해 둔다. 그래서
    여기서는 재정렬하지 않고 **전화번호가 있는 첫 번째** 병원을 쓴다.
    """
    selected = state.get("selected_hospital")
    if isinstance(selected, dict):
        inner = selected.get("hospital") if isinstance(selected.get("hospital"), dict) else selected
        if inner.get("phone"):
            return str(inner["phone"]), str(inner.get("name") or "")
    for record in _hospital_records(state):
        if record.get("phone"):
            return str(record["phone"]), str(record.get("name") or "")
    return None, ""


def prepare_call_hospital_node(state: dict) -> dict:
    """즉시 위급일 때 전화 action 을 준비한다(명세 32절 L).

    이 node 는 **정보 수집을 기다리지 않고** 실행된다(명세 29절). 병원 검색은
    다른 branch 에서 병렬로 돌고 있어 아직 전화번호가 없을 수 있는데, 그때도
    action 자체는 만들어 둔다. 번호는 `Build Emergency Result` 가 검색 결과를
    보고 채워 넣는다. 여기서 임의의 번호(119, 대표번호 등)를 넣지 않는다.

    실제 전화 실행은 구현하지 않는다 — Android 가 처리할 action JSON 만 만든다.
    """
    phone, name = _best_phone(state)
    action: dict[str, Any] = {"type": "CALL_HOSPITAL", "phone": phone}
    if name:
        action["hospital_name"] = name
    if phone is None:
        action["status"] = "pending_hospital_search"

    logger.warning("critical_immediate — CALL_HOSPITAL action 준비(phone=%s)", phone)
    return {
        "ui_actions": [action],
        "risk_reasons": ["[응급] 즉시 위급 신호가 있어 정보 수집보다 병원 연락을 우선했습니다."],
    }


# ---------------------------------------------------------------------------
# H. Parse Hospital Results (기본 구현)
# ---------------------------------------------------------------------------
def parse_hospital_results_node(state: dict) -> dict:
    """검색 원문 → 병원 후보 파싱 단계의 기본 구현(명세 32절 H).

    전용 node(`nodes/hospital_search.py`)가 파싱까지 끝냈다면 여기서 할 일이 없다.
    그래서 **아무것도 바꾸지 않고 통과**시키되, 원문은 있는데 후보가 하나도 나오지
    않은 경우를 로그로 드러낸다. 여기서 텍스트를 자체 파싱해 병원 정보를 만들면
    전용 node 와 규칙이 갈라지고, 무엇보다 검색 결과를 근거 없이 해석하게 된다.
    """
    raw_count = len(state.get("raw_hospital_results") or [])
    parsed_count = len(state.get("hospital_results") or [])
    if raw_count and not parsed_count:
        logger.warning(
            "병원 검색 원문 %d건이 있는데 파싱된 후보가 없습니다. "
            "nodes/hospital_search.py 의 파싱 결과를 확인하세요.",
            raw_count,
        )
    else:
        logger.info("Parse Hospital Results: 원문 %d건 / 후보 %d건", raw_count, parsed_count)
    return {}


# ---------------------------------------------------------------------------
# R. Build Emergency Result
# ---------------------------------------------------------------------------
def _ensure_call_hospital_actions(state: dict) -> list[dict[str, Any]]:
    """**부족한** 연락 action 만 추가로 만들어 돌려준다(기존 action 은 건드리지 않는다).

    `Prepare CALL_HOSPITAL action` node 는 병원 검색이 끝나기 전에 실행되므로
    번호 없는 안내형 action 을 먼저 남긴다. 검색이 끝나 번호가 확인되면 번호가 든
    action 이 하나는 있어야 사용자가 바로 전화를 걸 수 있다.

    **기존 action 을 지우지 않는 이유**: 번호 없는 action 에는 "평소 다니던 병원에
    전화하라" 는 안내가 들어 있다. 응급 상황에서 안내가 하나 더 있는 것이 없는 것보다
    낫다는 판단은 `nodes/hospital_suitability.py` 와 동일하게 유지한다.

    추가분만 돌려주므로 `merge_ui_actions` reducer 가 중복을 걸러 준다. 즉 이
    node 가 두 번 실행돼도 action 이 두 배가 되지 않는다.
    """
    actions = [item for item in (state.get("ui_actions") or []) if isinstance(item, dict)]
    types = {str(item.get("type")) for item in actions}
    has_call_with_phone = any(
        item.get("type") == "CALL_HOSPITAL" and item.get("phone") for item in actions
    )
    critical = str(state.get("emergency_urgency") or "none") == "critical_immediate"
    phone, name = _best_phone(state)

    extra: list[dict[str, Any]] = []

    # 1) 전화번호를 알아냈는데 걸 수 있는 action 이 없다면 만들어 준다.
    if phone and not has_call_with_phone and ("CALL_HOSPITAL" in types or critical):
        from ..prompts import HOSPITAL_VERIFICATION_NOTICE  # noqa: PLC0415

        extra.append(
            {
                "type": "CALL_HOSPITAL",
                "hospital_name": name,
                "phone": phone,
                "notice": HOSPITAL_VERIFICATION_NOTICE,
            }
        )

    # 2) 즉시 위급인데 연락 수단 action 이 아예 없는 경우의 최후 방어선.
    #    명세 40절: critical_immediate 면 CALL_HOSPITAL 또는 REQUEST_LOCATION 이
    #    반드시 있어야 한다. 없으면 Output Check 가 오류로 잡는다.
    if critical and not extra and not (types & {"CALL_HOSPITAL", "REQUEST_LOCATION"}):
        logger.error("즉시 위급인데 연락 action 이 없어 안내형 action 을 추가합니다.")
        extra.append(
            {
                "type": "CALL_HOSPITAL",
                "hospital_name": "",
                "phone": None,
                "notice": (
                    "지금 바로 평소 다니던 동물병원이나 가까운 24시 동물병원에 전화해 "
                    "현재 진료 및 응급 접수가 가능한지 확인해 주세요."
                ),
            }
        )

    return extra


def _hospital_lines(state: dict) -> list[str]:
    """안내 메시지에 넣을 병원 목록 문장을 만든다(검색된 사실만 그대로 적는다)."""
    lines: list[str] = []
    for index, item in enumerate(state.get("hospital_results") or [], start=1):
        if index > MAX_HOSPITALS_IN_MESSAGE or not isinstance(item, dict):
            break
        inner = item.get("hospital") if isinstance(item.get("hospital"), dict) else item
        name = str(inner.get("name") or "이름 미확인")
        phone = str(inner.get("phone") or "전화번호 미확인")
        address = str(inner.get("address") or "").strip()
        suitability = str(item.get("suitability") or "")
        tail = f" [{suitability}]" if suitability else ""
        lines.append(f"{index}. {name} / {phone}" + (f" / {address}" if address else "") + tail)
    return lines


def build_emergency_result_node(state: dict) -> dict:
    """응급 결과를 조립한다(명세 32절 R). **여러 번 실행돼도 같은 결과가 된다.**

    병원 branch 와 문서 branch 가 fan-in 하는 지점이라, 두 branch 가 서로 다른
    superstep 에 도착하면 이 node 가 두 번 실행될 수 있다. 그래서 누적하지 않고
    State 만 읽어 매번 전체를 다시 만든다.

    메시지는 고정 template 조립이다. 병원을 안내할 때는 명세 34·35·40절이 요구하는
    "방문 전 전화 확인" 문구를 반드시 붙인다 — 검색 결과만으로 실시간 진료 가능
    여부를 알 수 없기 때문이다.
    """
    from ..prompts import HOSPITAL_VERIFICATION_NOTICE, MEDICAL_DISCLAIMER  # noqa: PLC0415

    lines: list[str] = [IMMEDIATE_EMERGENCY_MESSAGE]

    red_flags = [str(item) for item in (state.get("red_flags") or [])][:5]
    if red_flags:
        lines += ["", "확인된 응급 신호: " + ", ".join(red_flags)]

    hospital_lines = _hospital_lines(state)
    if hospital_lines:
        lines += [
            "",
            "주변에서 확인된 동물병원입니다.",
            *hospital_lines,
            "",
            HOSPITAL_VERIFICATION_NOTICE,
        ]
    elif route_region(state) == "request_location":
        # 지역을 모르면 병원을 지어내지 않고 위치를 요청한다(명세 32절 F).
        lines += ["", REQUEST_LOCATION_MESSAGE]
    else:
        lines += [
            "",
            "지금 검색으로는 조건에 맞는 동물병원을 확인하지 못했습니다.",
            "평소 다니시던 병원 또는 24시간 동물병원에 직접 전화해 주세요.",
        ]

    unknown = [str(item) for item in (state.get("missing_fields") or [])]
    if unknown:
        lines += [
            "",
            "병원에 전화하실 때 아래 항목을 함께 말씀해 주시면 판단에 도움이 됩니다: "
            + ", ".join(unknown),
        ]

    if state.get("pdf_path"):
        lines += ["", f"병원에 보여 드릴 상담 자료를 PDF 로 정리했습니다: {state.get('pdf_filename')}"]
    if state.get("email_draft"):
        lines.append("이메일로 미리 보내실 수 있도록 초안도 준비했습니다.")

    lines += ["", MEDICAL_DISCLAIMER]

    message = "\n".join(lines).strip()
    extra_actions = _ensure_call_hospital_actions(state)
    logger.info(
        "Build Emergency Result: 병원 %d건 / action 추가 %s",
        len(state.get("hospital_results") or []),
        [action.get("type") for action in extra_actions],
    )
    updates: dict[str, Any] = {"draft_response": message}
    if extra_actions:
        # 누적 reducer(`merge_ui_actions`)가 중복을 걸러 주므로 추가분만 넘긴다.
        # `Replace` 로 통째로 바꾸면 다른 branch 가 넣은 action 을 지울 위험이 있다.
        updates["ui_actions"] = extra_actions
    return updates


# ---------------------------------------------------------------------------
# 서브그래프 조립
# ---------------------------------------------------------------------------
def build_emergency_subgraph(deps: SubgraphDeps | None = None) -> Any:
    """명세 32절 Emergency Subgraph 를 compile 해서 돌려준다.

    Args:
        deps: 주입 컨테이너. `None` 이면 LLM·Tavily 키 없이도 조립되며, 그때
            병원 검색은 빈 결과가 되어 "직접 전화" 안내로 이어진다.
    """
    from langgraph.graph import END, START, StateGraph  # noqa: PLC0415

    from ..state import PetCareState  # noqa: PLC0415

    resolved = deps or SubgraphDeps()
    graph = StateGraph(PetCareState)

    from . import make_missing_information_gate  # noqa: PLC0415

    # -- B / C1 / C2 -------------------------------------------------------
    graph.add_node(NODE_IMMEDIATE_MESSAGE, immediate_emergency_message_node)
    graph.add_node(
        NODE_CLINICAL_CONTEXT,
        resolve_node(
            resolved,
            NODE_CLINICAL_CONTEXT,
            ("clinical_context_priority",),
            ("clinical_context_priority_node",),
            factories=("make_clinical_context_priority_node",),
        ),
    )
    # 지역 확인·위치 요청·병원 파싱·전화 action 은 `nodes/hospital_*.py` 에 전용
    # 구현이 있으면 그쪽을 쓴다. 이 파일의 함수는 그 모듈이 없을 때의 예비 구현이다.
    graph.add_node(
        NODE_CHECK_REGION,
        resolve_optional_node(
            resolved,
            NODE_CHECK_REGION,
            ("hospital_search",),
            ("check_region_node",),
            fallback=check_region_node,
        ),
    )

    # -- 병원 branch (D → G → H → I) ---------------------------------------
    graph.add_node(
        NODE_HOSPITAL_REQUIREMENTS,
        resolve_node(
            resolved,
            NODE_HOSPITAL_REQUIREMENTS,
            ("hospital_requirements",),
            ("hospital_requirements_node", "hospital_requirement_builder_node"),
            factories=("make_hospital_requirements_node",),
        ),
    )
    graph.add_node(
        NODE_HOSPITAL_SEARCH,
        resolve_node(
            resolved,
            NODE_HOSPITAL_SEARCH,
            ("hospital_search",),
            ("hospital_search_node",),
            factories=("make_hospital_search_node",),
            # 이 factory 가 받는 것은 LLM 이 아니라 **병원 검색 서비스**다(명세 34절:
            # 병원 검색 Tavily 와 수의학 지식 Tavily 는 별도 class).
            factory_arg=resolved.resolved_hospital_search(),
        ),
    )
    graph.add_node(
        NODE_PARSE_HOSPITAL_RESULTS,
        resolve_optional_node(
            resolved,
            NODE_PARSE_HOSPITAL_RESULTS,
            ("hospital_search",),
            ("parse_hospital_results_node", "parse_hospital_node"),
            fallback=parse_hospital_results_node,
        ),
    )
    graph.add_node(
        NODE_HOSPITAL_SUITABILITY,
        resolve_node(
            resolved,
            NODE_HOSPITAL_SUITABILITY,
            ("hospital_suitability",),
            ("hospital_suitability_node",),
            factories=("make_hospital_suitability_node",),
        ),
    )
    graph.add_node(
        NODE_REQUEST_LOCATION,
        resolve_optional_node(
            resolved,
            NODE_REQUEST_LOCATION,
            ("hospital_search",),
            ("request_location_node",),
            fallback=request_location_node,
        ),
    )

    # -- 연락 최소정보 branch (J → K → L/M/N) -------------------------------
    graph.add_node(
        NODE_CONTACT_MINIMUM_INFORMATION,
        make_missing_information_gate(
            resolved,
            NODE_CONTACT_MINIMUM_INFORMATION,
            ("missing_information",),
            ("contact_minimum_information_node",),
        ),
    )
    graph.add_node(
        NODE_CONTACT_INTERRUPT,
        resolve_node(
            resolved,
            NODE_CONTACT_INTERRUPT,
            ("missing_information",),
            ("missing_information_interrupt_node",),
        ),
    )
    graph.add_node(
        NODE_PREPARE_CALL_HOSPITAL,
        resolve_optional_node(
            resolved,
            NODE_PREPARE_CALL_HOSPITAL,
            ("hospital_suitability",),
            ("prepare_call_hospital_action_node", "prepare_call_hospital_node"),
            fallback=prepare_call_hospital_node,
        ),
    )

    # -- 문서 branch (O → P → Q) — Visit 과 동일한 체인을 재사용한다 ---------
    for name, node in make_document_chain(resolved).items():
        graph.add_node(name, node)

    graph.add_node(NODE_BUILD_EMERGENCY_RESULT, build_emergency_result_node)

    # -- edges -------------------------------------------------------------
    graph.add_edge(START, NODE_IMMEDIATE_MESSAGE)
    # B → C1, C2 (병렬)
    graph.add_edge(NODE_IMMEDIATE_MESSAGE, NODE_CLINICAL_CONTEXT)
    graph.add_edge(NODE_IMMEDIATE_MESSAGE, NODE_CHECK_REGION)

    # C1, C2 → D. 두 edge 가 같은 superstep 에 도착하므로 D 는 한 번만 실행된다.
    graph.add_edge(NODE_CLINICAL_CONTEXT, NODE_HOSPITAL_REQUIREMENTS)
    graph.add_edge(NODE_CHECK_REGION, NODE_HOSPITAL_REQUIREMENTS)

    # E: Region exists?  — 없으면 병원을 검색하지 않는다.
    graph.add_conditional_edges(
        NODE_HOSPITAL_REQUIREMENTS,
        route_region,
        {
            "hospital_search": NODE_HOSPITAL_SEARCH,
            "request_location": NODE_REQUEST_LOCATION,
        },
    )
    graph.add_edge(NODE_HOSPITAL_SEARCH, NODE_PARSE_HOSPITAL_RESULTS)
    graph.add_edge(NODE_PARSE_HOSPITAL_RESULTS, NODE_HOSPITAL_SUITABILITY)
    graph.add_edge(NODE_HOSPITAL_SUITABILITY, NODE_BUILD_EMERGENCY_RESULT)
    graph.add_edge(NODE_REQUEST_LOCATION, NODE_BUILD_EMERGENCY_RESULT)

    # C1 → J: 연락 최소정보
    graph.add_edge(NODE_CLINICAL_CONTEXT, NODE_CONTACT_MINIMUM_INFORMATION)
    graph.add_conditional_edges(
        NODE_CONTACT_MINIMUM_INFORMATION,
        route_emergency_contact,
        {
            "call_hospital": NODE_PREPARE_CALL_HOSPITAL,
            "ready": NODE_DOCUMENT_AGENT,
            "ask": NODE_CONTACT_INTERRUPT,
        },
    )
    graph.add_edge(NODE_CONTACT_INTERRUPT, NODE_CONTACT_MINIMUM_INFORMATION)
    graph.add_edge(NODE_PREPARE_CALL_HOSPITAL, NODE_DOCUMENT_AGENT)

    graph.add_edge(NODE_DOCUMENT_AGENT, NODE_PACKET_VALIDATOR)
    graph.add_edge(NODE_PACKET_VALIDATOR, NODE_PDF_GENERATOR)
    graph.add_edge(NODE_PDF_GENERATOR, NODE_EMAIL_DRAFT)
    graph.add_edge(NODE_EMAIL_DRAFT, NODE_BUILD_EMERGENCY_RESULT)

    graph.add_edge(NODE_BUILD_EMERGENCY_RESULT, END)

    try:
        return graph.compile(name="emergency_subgraph", **resolved.compile_kwargs())
    except TypeError:  # 구버전 langgraph 호환
        return graph.compile(**resolved.compile_kwargs())
