"""조건부 분기 함수 모음 (명세 19절 — **분기는 LangGraph 가 한다**).

## 이 파일이 따로 있는 이유

`add_conditional_edges` 에 넘길 함수가 node 파일 여기저기 흩어져 있으면, 분기
규칙 하나를 확인하려고 10개 모듈을 열어야 한다. 더 나쁜 것은 같은 분기를 두 곳에서
조금 다르게 구현해 두는 것이다(예: "응급이면 어디로 가는가"). 그래서 **모든 분기
판정을 이 파일 한 곳에서 읽을 수 있게** 모았다.

이미 node 모듈이 제공하는 router(`route_after_fast_emergency_guard` 등)는 여기서
**다시 구현하지 않고 위임(delegate)한다.** 로직을 복제하면 한쪽만 고쳐지는 사고가
난다.

## 계약

- 모든 router 는 `state: dict` 하나만 받고 **문자열 하나**를 돌려준다.
  그대로 `graph.add_conditional_edges(node, router, path_map)` 에 넘길 수 있다.
- router 는 **State 를 수정하지 않는다.** 부수효과가 있는 판정은 node 의 일이다.
- 반환값은 두 종류가 섞여 있다(기존 node 구현과의 호환 때문이다).
    - *node 이름* 을 돌려주는 router: `route_context_loaded`, `route_needs_summary`,
      `route_after_fast_guard`, `route_region`
    - *분기 라벨* 을 돌려주는 router: `route_intent`, `route_final_risk`,
      `route_rag_status`, `route_missing_info`, `route_emergency_contact`,
      `route_output_check`
  라벨형은 builder 가 자기 node 이름으로 path_map 을 만들어 쓰면 된다. 이 파일이
  builder 의 node 이름을 알고 있으면 안 되기 때문에 라벨을 기본으로 둔다.

## 안전 기본값

판정할 수 없는 값이 들어오면 **항상 더 안전한 쪽**으로 보낸다.
- intent 를 모르면 `unsupported`(검증 안 된 건강 답변을 내보내지 않는다)
- RAG 충분성을 모르면 `insufficient`(근거가 충분한 척하지 않는다)
- 위험도는 평가자들의 값을 `merge_risk()` 로 다시 합쳐서 **낮은 쪽으로 새지 않게**
  한다(명세 28·47절).
"""

from __future__ import annotations

import logging
from typing import Any, Literal

from ..schemas import Intent, RiskLevel, SufficiencyStatus, merge_risk

logger = logging.getLogger(__name__)

__all__ = [
    "MAX_MISSING_INFORMATION_ROUNDS",
    "MISSING_INFO_ROUNDS_KEY",
    "FINAL_RISK_LABELS",
    "INTENT_LABELS",
    "RAG_STATUS_LABELS",
    "MISSING_INFO_LABELS",
    "EMERGENCY_CONTACT_LABELS",
    "OUTPUT_CHECK_LABELS",
    "route_context_loaded",
    "route_needs_summary",
    "route_after_fast_guard",
    "route_after_fast_emergency_guard",
    "route_intent",
    "route_after_supervisor",
    "route_final_risk",
    "route_missing_info",
    "route_rag_status",
    "route_region",
    "route_emergency_contact",
    "route_output_check",
    "missing_information_rounds",
]


# ---------------------------------------------------------------------------
# 상수
# ---------------------------------------------------------------------------
#: 같은 정보를 되묻는 최대 횟수. 이 횟수를 넘으면 남은 항목은 '모름' 으로 두고
#: 진행한다(명세 29절 "모든 항목은 `모름` 을 유효값으로 허용한다").
#: 무한 되묻기는 사용자를 지치게 하고, LangGraph `recursion_limit` 에도 걸린다.
#:
#: 이 값이 실제로 발동하려면 `MISSING_INFO_ROUNDS_KEY` 카운터가 **올라가야** 한다.
#: 오랫동안 그 증가 코드가 없어 카운터가 늘 0 이었고, 한도가 한 번도 발동하지 않아
#: 같은 질문이 무한 반복됐다(`missing_information_interrupt_node` 참고).
MAX_MISSING_INFORMATION_ROUNDS: int = 2

#: 되묻기 횟수를 세는 내부 key. `collected_information`(dict 병합 reducer) 안에
#: 둔다 — State 스키마를 건드리지 않고 turn 을 넘겨 살아남는 유일한 자리다.
#: 앞에 `__` 를 붙여 **사용자 답변이 아니라 내부 값**임을 표시한다. PDF·프롬프트에
#: 넣기 전에 `strip_internal_keys()`(subgraphs 패키지)로 걸러야 한다.
MISSING_INFO_ROUNDS_KEY: str = "__missing_info_rounds"

FINAL_RISK_LABELS: tuple[str, ...] = ("normal", "visit", "emergency")
INTENT_LABELS: tuple[str, ...] = (
    "general_chat",
    "health_question",
    "hospital_search",
    "unsupported",
)
RAG_STATUS_LABELS: tuple[str, ...] = ("sufficient", "insufficient", "conflicting")
MISSING_INFO_LABELS: tuple[str, ...] = ("ready", "ask")
EMERGENCY_CONTACT_LABELS: tuple[str, ...] = ("call_hospital", "ready", "ask")
OUTPUT_CHECK_LABELS: tuple[str, ...] = ("accept", "regenerate", "fallback")


# ---------------------------------------------------------------------------
# 중앙 흐름 (명세 24절) — 기존 node 구현에 위임한다
# ---------------------------------------------------------------------------
def route_context_loaded(state: dict) -> str:
    """START → `Context loaded?` 분기 (명세 24절).

    반환: `"db_context"` 또는 `"message_ingest"` (node 이름).
    판정 로직은 `nodes/db_context.py` 가 소유한다 — pet 이 바뀐 경우의 재로드
    규칙까지 거기 있으므로 여기서 다시 쓰면 두 규칙이 갈라진다.
    """
    from .nodes.db_context import route_context_loaded as _impl

    return _impl(state)


def route_needs_summary(state: dict) -> str:
    """`Need summary?` 분기 (명세 24·41절).

    반환: `"conversation_summary"` 또는 `"fast_emergency_guard"` (node 이름).
    임계값(`summary_trigger_message_count`)은 Settings 가 소유하므로 node 위임.
    """
    from .nodes.conversation_summary import route_needs_summary as _impl

    return _impl(state)


def route_after_fast_guard(state: dict) -> str:
    """`Critical emergency?` 분기 (명세 24절).

    반환: `"emergency"`(Emergency Subgraph 직행) 또는 `"supervisor"` (node 이름).

    Fast Emergency Guard 는 LLM 없이 키워드로만 판정한다. 여기서 Supervisor 를
    건너뛰는 이유는 명세 24절대로 **즉시 위급이면 의도 분류를 기다리지 않기**
    위해서다.
    """
    from .nodes.fast_emergency_guard import route_after_fast_emergency_guard as _impl

    return _impl(state)


#: 긴 이름을 쓰는 기존 호출부와의 호환 별칭.
route_after_fast_emergency_guard = route_after_fast_guard


def route_intent(state: dict) -> Intent:
    """Supervisor 결과의 `Intent` 분기 (명세 24·26절).

    반환: `general_chat` / `health_question` / `hospital_search` / `unsupported`.
    알 수 없는 값은 `unsupported` 로 보낸다 — 임의로 건강 경로를 태우면 검증되지
    않은 의료 답변이 나간다.
    """
    from .nodes.supervisor import route_after_supervisor as _impl

    return _impl(state)


#: 기존 호출부 호환 별칭.
route_after_supervisor = route_intent


def route_final_risk(state: dict) -> RiskLevel:
    """Merge Risk 이후 `Final Risk` 분기 (명세 24·28절).

    반환: `"normal"` / `"visit"` / `"emergency"`.

    **`final_risk` 를 그대로 믿지 않고 평가자별 원본값과 다시 합친다.**
    `rule_risk` / `assessment_risk` / `double_check_risk` 는 State 에 원본 그대로
    남아 있으므로(state.py 는 여기에 reducer 를 붙이지 않았다), 어떤 node 가
    `final_risk` 를 실수로 낮게 덮어써도 이 router 가 가장 높은 위험도로 되돌린다.
    병합은 `schemas.merge_risk()` 한 곳만 쓴다 — 규칙을 복제하지 않는다.

    `emergency_urgency` 가 `critical_immediate` 면 위험도 값과 무관하게 emergency 다.
    """
    if str(state.get("emergency_urgency") or "none") == "critical_immediate":
        return "emergency"

    merged = merge_risk(
        state.get("final_risk"),
        state.get("rule_risk"),
        state.get("assessment_risk"),
        state.get("double_check_risk"),
    )
    if merged != state.get("final_risk"):
        logger.warning(
            "final_risk(%r) 보다 높은 평가가 있어 %r 로 라우팅합니다 "
            "(rule=%r assessment=%r double=%r).",
            state.get("final_risk"),
            merged,
            state.get("rule_risk"),
            state.get("assessment_risk"),
            state.get("double_check_risk"),
        )
    return merged


# ---------------------------------------------------------------------------
# Missing Information (명세 29·30·31·32절)
# ---------------------------------------------------------------------------
def missing_information_rounds(state: dict) -> int:
    """지금까지 되물은 횟수를 읽는다(없으면 0)."""
    collected = state.get("collected_information") or {}
    if not isinstance(collected, dict):
        return 0
    try:
        return int(collected.get(MISSING_INFO_ROUNDS_KEY, 0) or 0)
    except (TypeError, ValueError):
        return 0


def route_missing_info(state: dict) -> Literal["ready", "ask"]:
    """`Enough user info?` 분기 (명세 30·31절 mermaid 의 C/D 판정).

    반환: `"ready"`(다음 단계 진행) 또는 `"ask"`(interrupt 로 되묻기).

    `"ready"` 로 보내는 조건은 다섯 가지이며, 앞의 것이 우선한다.
      1. `critical_immediate` — 명세 29절: 정보 수집이 전화 action 을 막지 않는다.
      1-1. `intent=general_knowledge` — 증상 호소가 없는 지식 질문이라 물을 것이 없다.
      2. node 가 `minimum_information_ready=True` 로 판정했다.
      3. 부족한 항목이 아예 없다.
      4. 이미 `MAX_MISSING_INFORMATION_ROUNDS` 번 물었다 — 남은 항목은 '모름' 으로
         두고 진행한다. 이때 `missing_fields` 는 지우지 않으므로 PDF 의
         `unknown_fields` 와 답변의 미확인 안내에 그대로 반영된다(추측 금지).
    """
    if str(state.get("emergency_urgency") or "none") == "critical_immediate":
        return "ready"
    # 지식 질문(general_knowledge)은 되묻지 않는다. "강아지에게 뭐가 몸에 좋아?" 에
    # "가장 신경 쓰이는 증상이 무엇인가요?" 를 물으면 답할 수 없는 것을 묻는 셈이고,
    # 실제로 그 자리에서 대화가 멈춰 RAG 검색까지 도달하지 못했다.
    if str(state.get("intent") or "") == "general_knowledge":
        return "ready"
    if state.get("minimum_information_ready"):
        return "ready"
    if not (state.get("missing_fields") or []):
        return "ready"
    rounds = missing_information_rounds(state)
    if rounds >= MAX_MISSING_INFORMATION_ROUNDS:
        logger.info(
            "되묻기 %d회를 채워 남은 항목(%s)은 '모름' 으로 두고 진행합니다.",
            rounds,
            list(state.get("missing_fields") or []),
        )
        return "ready"
    return "ask"


def route_emergency_contact(state: dict) -> Literal["call_hospital", "ready", "ask"]:
    """응급 서브그래프의 `Critical immediate?` → `Minimum info ready?` 분기(명세 32절).

    반환
      - `"call_hospital"`: 즉시 위급 — 정보 수집을 기다리지 않고 전화 action 준비
      - `"ready"`: 최소정보가 모였으므로 Document Agent 로
      - `"ask"`: interrupt 로 최소정보를 되묻는다

    두 판정을 한 함수로 합친 이유: mermaid 의 K 와 M 이 연속 분기라 라우터를 둘로
    나누면 "critical 인데 ask 로 갔다" 같은 조합 실수가 생길 수 있다. 여기서
    critical 을 **가장 먼저** 확인해 그 조합 자체를 만들 수 없게 한다.
    """
    if str(state.get("emergency_urgency") or "none") == "critical_immediate":
        return "call_hospital"
    return "ready" if route_missing_info(state) == "ready" else "ask"


# ---------------------------------------------------------------------------
# Health Subgraph (명세 30절)
# ---------------------------------------------------------------------------
def route_rag_status(state: dict) -> SufficiencyStatus:
    """`RAG status` 분기 (명세 30절 mermaid 의 G).

    반환: `"sufficient"` / `"insufficient"` / `"conflicting"`.

    알 수 없는 값은 `"insufficient"` 로 본다. 근거가 충분한 척하고 답변을 만드는
    것보다, 웹 fallback 을 한 번 더 도는 쪽이 항상 안전하다. Tavily 가 없거나
    실패해도 그 경로는 빈 결과로 정상 종료된다(명세 15절).
    """
    status = state.get("rag_sufficiency")
    if status in RAG_STATUS_LABELS:
        return status  # type: ignore[return-value]
    logger.warning("rag_sufficiency 값이 %r 라 insufficient 로 처리합니다.", status)
    return "insufficient"


# ---------------------------------------------------------------------------
# Emergency Subgraph (명세 32절)
# ---------------------------------------------------------------------------
def _has_region_fallback(state: dict) -> bool:
    """지역 판정의 예비 구현 — `nodes/hospital_search.py` 를 못 불러올 때만 쓴다.

    `region_name` 문자열이 우선이다(Colab 테스트 입력이 이것이다). 위도·경도는
    Android 연동 대비 필드이며, 둘 다 있을 때만 유효로 본다 — 하나만 있으면
    좌표로서 의미가 없다.
    """
    region = state.get("region_name")
    if isinstance(region, str) and region.strip():
        return True
    latitude, longitude = state.get("latitude"), state.get("longitude")
    return isinstance(latitude, (int, float)) and isinstance(longitude, (int, float))


def route_region(state: dict) -> str:
    """`Region exists?` 분기 (명세 32절 mermaid 의 E).

    반환: `"hospital_search"` 또는 `"request_location"` (node 이름).

    지역을 모르면 병원을 **추측해서 만들지 않는다.** 존재하지 않는 병원 이름과
    전화번호를 응급 상황에 안내하는 것이 이 시스템에서 가장 위험한 실패다.
    대신 Android 가 위치 권한을 요청하도록 `REQUEST_LOCATION` 결과를 돌려준다.

    판정 자체는 `nodes/hospital_search.py` 가 소유한다(같은 규칙이 검색 node 안에서도
    쓰이기 때문이다). 그 모듈을 못 불러오는 환경에서만 예비 구현으로 떨어진다.
    """
    try:
        from .nodes.hospital_search import route_region_available as _impl
    except ImportError as exc:  # pragma: no cover - 모듈 미작성/미설치 환경
        logger.debug("hospital_search router 를 못 불러와 예비 판정을 씁니다: %s", exc)
        return "hospital_search" if _has_region_fallback(state) else "request_location"
    return _impl(state)


# ---------------------------------------------------------------------------
# 출력 검증 (명세 40절)
# ---------------------------------------------------------------------------
def route_output_check(state: dict) -> Literal["accept", "regenerate", "fallback"]:
    """`Valid?` 분기 (명세 24·40절). 판정 로직은 `nodes/output_check.py` 소유."""
    from .nodes.output_check import route_output_check as _impl

    return _impl(state)


def __getattr__(name: str) -> Any:  # pragma: no cover - 오타 진단용
    raise AttributeError(
        f"petcare_ai.graph.routers 에 '{name}' 이(가) 없습니다. "
        f"사용 가능한 router: {', '.join(n for n in __all__ if n.startswith('route_'))}"
    )
