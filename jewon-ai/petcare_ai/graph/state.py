"""LangGraph 공유 State (명세 25절) — 필드 정의와 reducer.

이 파일은 graph 하위 모든 node 가 읽고 쓰는 **유일한 계약**이다. node 는 여기에
없는 key 를 반환하면 안 된다(LangGraph 는 스키마에 없는 channel 쓰기를 거부한다).

## reducer 를 붙인 이유

LangGraph 는 **같은 super-step 에서 두 node 가 같은 key 를 쓰면** reducer 가 없는 한
`InvalidUpdateError` 를 낸다. 명세 24/30/32절 graph 에는 다음 병렬 구간이 있다.

- 24절: `Clinical Context` → `Rule Assessment` + `Assessment Agent` 동시 실행
  → 두 node 가 `red_flags`, `risk_reasons`, `missing_fields` 를 동시에 쓴다.
- 32절(Emergency): `Immediate Message` → `Clinical Context` + `Check Region` 동시,
  그리고 `Hospital Suitability` + `Email Draft` 가 `Build Emergency Result` 로 fan-in
  → `ui_actions`(REQUEST_LOCATION / CALL_HOSPITAL / OPEN_PDF_PREVIEW) 가 여러
  branch 에서 동시에 쌓인다.
- 30절(Health): RAG 경로와 웹 fallback 경로가 `Evidence Merge` 로 합류
  → `merged_evidence`, `validation_errors` 가 fan-in 된다.

그래서 위 리스트 필드에는 **누적 + 중복 제거** reducer 를 붙였다. 단순 덮어쓰기가
아니라 누적이므로, 뒤에 실행되는 node 가 앞 node 의 red flag 를 지울 수 없다.

`final_risk` 와 `emergency_urgency` 에는 **상향 전용(escalate) reducer** 를 붙였다.
명세 28절 "낮은 위험도로 덮어쓰지 않는다" 를 타입 수준에서 강제하기 위해서다.
병합 계산 자체는 `schemas.merge_risk()` 를 그대로 재사용한다(로직 이중화 금지).

리스트를 **줄여야 하는** 정상 상황(41절 대화 요약이 오래된 messages 를 잘라내는 등)
을 위해 `Replace` / `ReplaceDict` 탈출구를 둔다. node 가 `Replace([...])` 로 감싸
반환하면 reducer 가 누적 대신 교체한다.

reducer 를 **일부러 붙이지 않은** 필드:
- `rule_risk` / `assessment_risk` / `double_check_risk`: 평가자별 원본 기록이다.
  각각 다른 key 이므로 병렬 충돌이 없고, 원본을 덮어써서 보존해야 한다.
- `retry_count`: Output Check 단일 node 만 갱신한다(명세 40절, 재생성 최대 1회).
  누적 reducer 를 붙이면 절대값 쓰기가 깨진다.
- `hospital_results`: Suitability node 가 점수순 정렬 후 통째로 교체한다.
- `final_response` / `draft_response` 등 스칼라: 마지막 쓰기가 정답이다.
"""

from __future__ import annotations

import json
import uuid
from typing import Annotated, Any

try:  # 명세 25절이 지정한 형태. langgraph 가 typing_extensions 를 요구하므로 보통 존재한다.
    from typing_extensions import TypedDict
except ImportError:  # pragma: no cover - 3.11+ 표준 라이브러리 fallback
    from typing import TypedDict  # type: ignore[assignment]

from ..schemas import (
    RISK_PRIORITY,
    EmergencyUrgency,
    Intent,
    RiskLevel,
    Species,
    SufficiencyStatus,
    merge_risk,
)

__all__ = [
    "PetCareState",
    "Replace",
    "ReplaceDict",
    "URGENCY_PRIORITY",
    "append_messages",
    "merge_unique_strings",
    "merge_records",
    "merge_ui_actions",
    "merge_dicts",
    "escalate_risk",
    "escalate_urgency",
    "make_initial_state",
    "make_message",
    "build_trace_metadata",
]


# ---------------------------------------------------------------------------
# 누적 reducer 탈출구
# ---------------------------------------------------------------------------
class Replace(list):
    """reducer 를 우회해 리스트를 통째로 교체하고 싶을 때 감싸는 표식.

    예) 대화 요약 node 가 오래된 messages 를 잘라낼 때:
        `return {"messages": Replace(recent_messages)}`

    list 서브클래스이므로 그대로 state 에 저장돼도 동작에 문제가 없다.
    """


class ReplaceDict(dict):
    """`merge_dicts` reducer 를 우회해 dict 를 통째로 교체할 때 쓰는 표식."""


URGENCY_PRIORITY: dict[str, int] = {
    "none": 0,
    "contact_ready": 1,
    "critical_immediate": 2,
}

# 레코드 동일성 판단에 쓰는 key 후보(앞에서부터 먼저 발견된 것을 사용).
# `source_url` 은 일부러 제외한다 — 같은 페이지에서 추출한 서로 다른 병원이
# 하나로 합쳐지는 사고를 막기 위해서다.
_IDENTITY_KEYS: tuple[str, ...] = ("evidence_id", "chunk_id", "url", "id")


def _as_list(value: Any) -> list[Any]:
    """단일 값으로 반환된 update 도 허용한다(node 작성 실수 방어)."""
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _identity(item: Any) -> str:
    """dict 레코드의 중복 판단 key. 식별자가 없으면 내용 전체를 canonical 화한다."""
    if isinstance(item, dict):
        for key in _IDENTITY_KEYS:
            value = item.get(key)
            if value:
                return f"{key}={value}"
        try:
            return json.dumps(item, sort_keys=True, ensure_ascii=False, default=str)
        except Exception:  # 직렬화 불가한 값이 섞인 경우
            return repr(item)
    return repr(item)


# ---------------------------------------------------------------------------
# reducer 구현
# ---------------------------------------------------------------------------
def append_messages(left: list[Any] | None, right: Any) -> list[Any]:
    """대화 메시지를 누적한다.

    중복 제거를 하지 않는 이유: 사용자가 같은 문장("네", "모름")을 여러 번 보내는
    것은 정상이며, 지워지면 대화 흐름이 왜곡된다.
    요약 node 가 앞부분을 잘라낼 때만 `Replace` 로 교체한다.
    """
    if isinstance(right, Replace):
        return list(right)
    return list(left or []) + _as_list(right)


def merge_unique_strings(left: list[str] | None, right: Any) -> list[str]:
    """문자열 리스트를 순서 유지 + 중복 제거로 누적한다.

    red_flags / risk_reasons 는 Rule·Assessment·Double Check 세 평가자가 같은
    문구를 낼 수 있는데(예: "호흡곤란"), 사용자에게 같은 근거를 세 번 보여줄 수는
    없다. 반대로 뒤 node 가 앞 node 의 red flag 를 **삭제하는 것은 금지**이므로
    덮어쓰기가 아니라 누적으로 둔다.
    """
    if isinstance(right, Replace):
        return [str(item) for item in right]
    merged: list[str] = []
    seen: set[str] = set()
    for item in list(left or []) + _as_list(right):
        text = str(item).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        merged.append(text)
    return merged


def merge_records(left: list[dict[str, Any]] | None, right: Any) -> list[dict[str, Any]]:
    """dict 레코드 리스트를 누적하되 동일 레코드는 한 번만 남긴다.

    근거(evidence)·문서·provenance 는 RAG 경로와 웹 fallback 경로가 합류하면서
    같은 chunk 나 URL 이 두 번 들어올 수 있다. 근거가 중복되면 답변에서 같은 출처가
    반복 인용되므로 `evidence_id`/`chunk_id`/`url`/`id` 기준으로 정리한다.
    """
    if isinstance(right, Replace):
        return [item for item in right]
    merged: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in list(left or []) + _as_list(right):
        key = _identity(item)
        if key in seen:
            continue
        seen.add(key)
        merged.append(item)
    return merged


def merge_ui_actions(left: list[dict[str, Any]] | None, right: Any) -> list[dict[str, Any]]:
    """UI action 을 누적하되 완전히 동일한 action 은 한 번만 남긴다.

    응급 subgraph(명세 32절)에서 병원 branch 와 문서 branch 가 동시에 fan-in 하므로
    `CALL_HOSPITAL`, `OPEN_PDF_PREVIEW`, `OPEN_GMAIL_COMPOSE`, `REQUEST_LOCATION`
    이 서로 다른 node 에서 들어온다. 같은 전화번호로 두 번 거는 버튼을 보여주지
    않기 위해 action 전체 내용을 기준으로 중복을 제거한다.
    """
    if isinstance(right, Replace):
        return [item for item in right]
    merged: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in list(left or []) + _as_list(right):
        try:
            key = json.dumps(item, sort_keys=True, ensure_ascii=False, default=str)
        except Exception:
            key = repr(item)
        if key in seen:
            continue
        seen.add(key)
        merged.append(item)
    return merged


def merge_dicts(left: dict[str, Any] | None, right: Any) -> dict[str, Any]:
    """dict 를 얕게 병합한다(새 값 우선).

    `collected_information` 은 명세 29절 multi-turn interrupt 로 여러 turn 에 걸쳐
    조금씩 채워진다. 매번 통째로 덮어쓰면 이전 turn 에 받은 답이 사라진다.
    """
    if isinstance(right, ReplaceDict):
        return dict(right)
    if not isinstance(right, dict):
        return dict(left or {})
    merged = dict(left or {})
    merged.update(right)
    return merged


def escalate_risk(left: str | None, right: str | None) -> RiskLevel:
    """위험도는 올릴 수만 있다(명세 28절 / 47절).

    병합 로직은 `schemas.merge_risk()` 한 곳에만 둔다. 여기서 다시 구현하면
    normal < visit < emergency 규칙이 두 군데로 갈라진다.
    """
    return merge_risk(left, right)


def escalate_urgency(left: str | None, right: str | None) -> EmergencyUrgency:
    """응급 긴급도도 상향 전용이다.

    none < contact_ready < critical_immediate. Fast Emergency Guard 가
    `critical_immediate` 로 올린 뒤 뒤쪽 LLM node 가 `none` 으로 되돌리는 일을
    구조적으로 막는다.
    """
    best: str = left if left in URGENCY_PRIORITY else "none"
    if right in URGENCY_PRIORITY and URGENCY_PRIORITY[right] > URGENCY_PRIORITY[best]:
        best = right  # type: ignore[assignment]
    return best  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# State 스키마 (명세 25절 전 필드 + 노드 간 합의된 보조 필드)
# ---------------------------------------------------------------------------
class PetCareState(TypedDict, total=False):
    """PetCare 상담 graph 의 공유 State.

    `total=False` 이므로 모든 필드가 선택적이다. node 는 반드시 `state.get(...)` 로
    읽어 KeyError 를 피한다. 명세 21절에 따라 **현재 pet_id 의 데이터만** 담는다.
    """

    # ---- conversation ----------------------------------------------------
    thread_id: str
    request_id: str
    pet_id: int
    user_message: str
    messages: Annotated[list[Any], append_messages]
    conversation_summary: str
    context_loaded: bool

    # ---- future Android input (Colab 에서는 region_name 만 사용) -----------
    latitude: float | None
    longitude: float | None
    region_name: str | None

    # ---- full clinical data (명세 21절: 선택된 반려동물 전체 데이터) --------
    pet_profile: dict[str, Any]
    diagnoses: list[dict[str, Any]]
    daily_entries: list[dict[str, Any]]

    # ---- prioritized context (명세 27절) ----------------------------------
    current_observation: dict[str, Any]
    priority_pet_context: dict[str, Any]
    related_diagnoses: Annotated[list[dict[str, Any]], merge_records]
    supporting_daily_entries: Annotated[list[dict[str, Any]], merge_records]
    context_conflicts: Annotated[list[dict[str, Any]], merge_records]
    context_provenance: Annotated[list[dict[str, Any]], merge_records]

    # ---- routing / risk (명세 26·28절) ------------------------------------
    intent: Intent
    #: Fast Guard 가 탐지했으나 **질문 문장일 수 있어 판단을 보류한** 위급 신호.
    #: Supervisor 가 사건인지 질문인지 판정한다. LLM 이 없으면 응급으로 확정된다.
    pending_emergency_signals: Annotated[list[str], merge_unique_strings]
    rule_risk: RiskLevel
    assessment_risk: RiskLevel
    double_check_risk: RiskLevel
    final_risk: Annotated[RiskLevel, escalate_risk]
    emergency_urgency: Annotated[EmergencyUrgency, escalate_urgency]
    red_flags: Annotated[list[str], merge_unique_strings]
    risk_reasons: Annotated[list[str], merge_unique_strings]

    # ---- missing information (명세 29절) ----------------------------------
    required_fields: Annotated[list[str], merge_unique_strings]
    missing_fields: Annotated[list[str], merge_unique_strings]
    collected_information: Annotated[dict[str, Any], merge_dicts]
    minimum_information_ready: bool

    # ---- RAG and web evidence (명세 30절) ---------------------------------
    rag_query: str
    rag_documents: Annotated[list[dict[str, Any]], merge_records]
    rag_sufficiency: SufficiencyStatus
    rag_missing_topics: Annotated[list[str], merge_unique_strings]
    web_fallback_required: bool
    validated_web_evidence: Annotated[list[dict[str, Any]], merge_records]
    merged_evidence: Annotated[list[dict[str, Any]], merge_records]

    # ---- hospital (명세 33~35절) ------------------------------------------
    hospital_requirements: dict[str, Any]
    hospital_search_queries: Annotated[list[str], merge_unique_strings]
    raw_hospital_results: Annotated[list[dict[str, Any]], merge_records]
    hospital_results: list[dict[str, Any]]
    selected_hospital: dict[str, Any] | None

    # ---- document and email (명세 36~38절) --------------------------------
    consultation_packet: dict[str, Any]
    pdf_path: str | None
    pdf_filename: str | None
    email_draft: dict[str, Any] | None

    # ---- final output (명세 39·40절) --------------------------------------
    draft_response: str
    final_response: str
    ui_actions: Annotated[list[dict[str, Any]], merge_ui_actions]
    validation_errors: Annotated[list[str], merge_unique_strings]
    retry_count: int

    # ---- 보조 필드 (명세 25절 "최소한" 이상. node 간 이름 드리프트 방지용) --
    # Supervisor(26절) 부가 출력 — router 가 intent 외에 참고한다.
    species: Species
    possible_emergency: bool
    needs_clinical_context: bool
    supervisor_reason: str
    # RAG Query Builder(12절)는 ko/en 두 query 를 만든다. rag_query 는 대표 문자열.
    rag_query_ko: str
    rag_query_en: str
    rag_required_topics: Annotated[list[str], merge_unique_strings]
    # Evidence Merge(16절) 결과 부가 정보
    evidence_conflicts: Annotated[list[str], merge_unique_strings]
    has_reliable_evidence: bool
    # Missing Information Agent(29절)가 사용자에게 물을 질문 문장
    missing_information_question: str
    # Document Agent(36절): visit_consultation / emergency_consultation
    document_type: str
    # Output Check(40절): accept / regenerate / fallback
    output_check_action: str
    fallback_used: bool
    # LangSmith metadata(42절) — 개인 식별정보·원문은 넣지 않는다.
    trace_metadata: dict[str, Any]


# ---------------------------------------------------------------------------
# 초기 State 생성
# ---------------------------------------------------------------------------
def make_message(role: str, content: str, **meta: Any) -> dict[str, Any]:
    """대화 메시지 1건을 만든다.

    LangChain `BaseMessage` 대신 dict 를 쓰는 이유: State 가 그대로 JSON 직렬화되어
    checkpoint·LangSmith·Android 응답으로 나가야 하고, langchain 미설치 환경에서도
    graph 가 돌아가야 하기 때문이다. `{"role", "content"}` 형태는 LangChain 이
    그대로 받아들인다.
    """
    message: dict[str, Any] = {"role": role, "content": content}
    message.update(meta)
    return message


def make_initial_state(
    pet_id: int,
    user_message: str,
    thread_id: str | None = None,
    region_name: str | None = None,
    *,
    request_id: str | None = None,
    latitude: float | None = None,
    longitude: float | None = None,
    conversation_summary: str = "",
    messages: list[Any] | None = None,
    pet_profile: dict[str, Any] | None = None,
    diagnoses: list[dict[str, Any]] | None = None,
    daily_entries: list[dict[str, Any]] | None = None,
    collected_information: dict[str, Any] | None = None,
    **overrides: Any,
) -> PetCareState:
    """graph 실행에 넣을 초기 State 를 만든다.

    누적 필드를 빈 리스트/딕셔너리로 미리 채우는 이유: reducer 는 update 에만
    적용되므로 초기값이 없으면 첫 node 의 `state.get("red_flags")` 가 None 이 되어
    node 마다 방어 코드를 중복 작성하게 된다.

    `pet_profile` / `diagnoses` / `daily_entries` 를 주지 않으면 비워 두고
    `context_loaded=False` 로 남긴다 — DB Context Agent 가 채운다(명세 24절 분기).
    반대로 fixture 를 직접 주입하면 `context_loaded=True` 로 시작한다.

    `intent` 는 일부러 넣지 않는다. Supervisor 가 정하기 전에 기본값이 있으면
    router 가 잘못된 branch 를 탈 수 있다.
    """
    context_loaded = pet_profile is not None

    initial_messages: list[Any] = list(messages) if messages else []
    if user_message:
        initial_messages.append(make_message("user", user_message))

    state: PetCareState = {
        # conversation
        "thread_id": thread_id or f"thread-{uuid.uuid4().hex[:12]}",
        "request_id": request_id or f"req-{uuid.uuid4().hex[:12]}",
        "pet_id": pet_id,
        "user_message": user_message,
        "messages": initial_messages,
        "conversation_summary": conversation_summary,
        "context_loaded": context_loaded,
        # android input
        "latitude": latitude,
        "longitude": longitude,
        "region_name": region_name,
        # clinical data
        "pet_profile": dict(pet_profile) if pet_profile else {},
        "diagnoses": list(diagnoses) if diagnoses else [],
        "daily_entries": list(daily_entries) if daily_entries else [],
        # prioritized context
        "current_observation": {},
        "priority_pet_context": {},
        "related_diagnoses": [],
        "supporting_daily_entries": [],
        "context_conflicts": [],
        "context_provenance": [],
        # routing / risk
        "rule_risk": "normal",
        "assessment_risk": "normal",
        "double_check_risk": "normal",
        "final_risk": "normal",
        "emergency_urgency": "none",
        "red_flags": [],
        "pending_emergency_signals": [],
        "risk_reasons": [],
        # missing information
        "required_fields": [],
        "missing_fields": [],
        "collected_information": dict(collected_information) if collected_information else {},
        "minimum_information_ready": False,
        # RAG
        "rag_query": "",
        "rag_documents": [],
        "rag_sufficiency": "insufficient",
        "rag_missing_topics": [],
        "web_fallback_required": False,
        "validated_web_evidence": [],
        "merged_evidence": [],
        # hospital
        "hospital_requirements": {},
        "hospital_search_queries": [],
        "raw_hospital_results": [],
        "hospital_results": [],
        "selected_hospital": None,
        # document / email
        "consultation_packet": {},
        "pdf_path": None,
        "pdf_filename": None,
        "email_draft": None,
        # final output
        "draft_response": "",
        "final_response": "",
        "ui_actions": [],
        "validation_errors": [],
        "retry_count": 0,
        # 보조 필드
        "possible_emergency": False,
        "needs_clinical_context": False,
        "supervisor_reason": "",
        "rag_query_ko": "",
        "rag_query_en": "",
        "rag_required_topics": [],
        "evidence_conflicts": [],
        "has_reliable_evidence": False,
        "missing_information_question": "",
        "document_type": "",
        "output_check_action": "accept",
        "fallback_used": False,
        "trace_metadata": {},
    }

    species = _species_from_profile(state["pet_profile"])
    if species is not None:
        state["species"] = species

    for key, value in overrides.items():
        state[key] = value  # type: ignore[literal-required]

    return state


def _species_from_profile(pet_profile: dict[str, Any]) -> Species | None:
    """PET DB 값에서 dog/cat 을 판정한다. 판정 불가면 None(추측하지 않는다).

    RAG index 가 species 별로 분리되어 있어(명세 11절) 잘못 넘기면 고양이에게
    개 문서를 붙이게 된다. 그래서 애매하면 비워 두고 상위 node 가 다시 판단한다.
    """
    raw = str(pet_profile.get("species") or pet_profile.get("animal_type") or "").strip().lower()
    if raw in ("dog", "canine", "개", "강아지", "犬"):
        return "dog"
    if raw in ("cat", "feline", "고양이", "猫"):
        return "cat"
    return None


# ---------------------------------------------------------------------------
# LangSmith metadata (명세 42절)
# ---------------------------------------------------------------------------
def build_trace_metadata(state: PetCareState, environment: str = "colab") -> dict[str, Any]:
    """trace 용 metadata 를 만든다 — **원문·개인 식별정보는 넣지 않는다**.

    명세 42절이 요구한 항목만 담는다. user_message, 반려동물 이름, 주소, 전화번호는
    tag 로 나가면 LangSmith 에 그대로 남으므로 제외하고, 분기 판단에 필요한
    범주형 값과 개수만 넣는다.
    """
    return {
        "environment": environment,
        "pet_id": str(state.get("pet_id", "")),
        "thread_id": state.get("thread_id", ""),
        "request_id": state.get("request_id", ""),
        "intent": state.get("intent"),
        "final_risk": state.get("final_risk"),
        "emergency_urgency": state.get("emergency_urgency"),
        "rag_sufficiency": state.get("rag_sufficiency"),
        "web_fallback_triggered": bool(state.get("web_fallback_required")),
        "hospital_search_triggered": bool(state.get("hospital_search_queries")),
        "evidence_count": len(state.get("merged_evidence") or []),
        "retry_count": int(state.get("retry_count", 0) or 0),
        "risk_priority": RISK_PRIORITY.get(str(state.get("final_risk", "normal")), 0),
    }
