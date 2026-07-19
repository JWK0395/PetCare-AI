"""메인 그래프 조립 — 명세 24절 전체 flowchart 를 그대로 옮긴다.

구조(명세 24절):

    START → Context loaded? → DB Context → Message Ingest
          → Need summary? → Summary → Fast Emergency Guard
          → Critical? → Emergency Subgraph / Supervisor
          → Intent 분기(General / Health / Hospital Search / Unsupported)
          → Health 경로: Clinical Context Priority
                       → Rule Assessment ∥ Assessment Agent
                       → Risk Double Check → Merge Risk
          → Final Risk 분기(Normal→Health / Visit→Visit / Emergency→Emergency)
          → 전부 Output Check → Valid?(accept→Final Safety / regenerate→1회 재시도
                                        / fallback→Safe Fallback)
          → Build Result → END

설계 원칙:

- **분기는 LangGraph 가 한다.** 라우터는 `routers.py` 의 순수 함수이며 여기서는
  `add_conditional_edges` 로 연결만 한다(명세 19절).
- **의존성은 전부 주입**한다. `GraphDependencies` 하나만 바꿔 끼우면 Colab fixture →
  FastAPI/SQLite 로 이동한다(명세 2절 "adapter 만 교체").
- **LLM 이 없어도 끝까지 돈다.** `deps.llm` 이 None 이면 각 node 가 규칙 기반으로
  동작하므로 그래프 구조는 동일하다.
- 서브그래프는 `subgraphs/` 의 팩토리가 만든 compile 된 그래프를 **node 로 끼운다**.
  부모와 같은 `PetCareState` 를 공유하므로 상태 변환 래퍼가 필요 없다.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable

from ..schemas import ChatGraphResult
from .routers import (
    route_after_fast_guard,
    route_context_loaded,
    route_final_risk,
    route_intent,
    route_needs_summary,
    route_output_check,
)
from .subgraphs import SubgraphDeps, resolve_node

logger = logging.getLogger(__name__)

NodeFn = Callable[[dict], dict]

# ---------------------------------------------------------------------------
# graph node 이름 — 라우터 반환값과 1:1 로 맞춘다.
# ---------------------------------------------------------------------------
N_DB_CONTEXT = "db_context"
N_INGEST = "message_ingest"
N_SUMMARY = "conversation_summary"
N_FAST_GUARD = "fast_emergency_guard"
N_SUPERVISOR = "supervisor"
N_GENERAL = "general_chat"
N_UNSUPPORTED = "unsupported_response"
N_CLINICAL = "clinical_context_priority"
N_RULE = "rule_assessment"
N_MODEL = "assessment_agent"
N_DOUBLE = "risk_double_check"
N_MERGE = "merge_risk"
N_HEALTH_SUB = "health_subgraph"
N_VISIT_SUB = "visit_subgraph"
N_EMERGENCY_SUB = "emergency_subgraph"
N_HOSPITAL_FLOW = "hospital_search_flow"
N_OUTPUT_CHECK = "output_check"
N_FINAL_SAFETY = "final_safety"
N_FALLBACK = "safe_fallback"
N_RESULT = "build_result"


# ---------------------------------------------------------------------------
# 주입 컨테이너
# ---------------------------------------------------------------------------
@dataclass
class GraphDependencies:
    """메인 그래프가 쓰는 외부 의존성.

    전부 비워 두면(`GraphDependencies()`) LLM·API 키 없이 규칙 기반으로 도는
    기본 경로가 된다. 앱 연동 시에는 `clinical_adapter` 와 `checkpointer` 만
    교체하면 되고 node 코드는 손대지 않는다(명세 2절).

    Attributes:
        settings: 전역 설정. 없으면 `config.get_settings()`.
        llm: `llm.build_llm()` 결과. None 이면 규칙 기반.
        rag_service: `VeterinaryRagService`(mock 주입용).
        hospital_search: `HospitalSearchService` — 수의학 검색과 별도 class(명세 34절).
        clinical_adapter: PET/진단서/일기 DB adapter. Colab 은 Fixture, 앱은 SQLite.
        node_overrides: node 이름 → 함수. 테스트가 특정 node 만 갈아끼울 때.
        allow_missing_nodes: True 면 없는 node 를 빈 node 로 대체(진단용).
            기본 False — 반쪽짜리 그래프가 조용히 만들어지는 것이 가장 위험하다.
    """

    settings: Any = None
    llm: Any = None
    rag_service: Any = None
    hospital_search: Any = None
    clinical_adapter: Any = None
    node_overrides: dict[str, NodeFn] = field(default_factory=dict)
    allow_missing_nodes: bool = False

    def resolved_settings(self) -> Any:
        if self.settings is None:
            from ..config import get_settings  # noqa: PLC0415

            self.settings = get_settings()
        return self.settings

    def to_subgraph_deps(self, checkpointer: Any = None) -> SubgraphDeps:
        """서브그래프 팩토리에 넘길 컨테이너로 변환한다.

        checkpointer 는 기본적으로 넘기지 않는다 — 부모 그래프의 checkpointer 를
        상속하는 편이 thread_id 관리가 단순하기 때문이다(명세 29절 resume).
        """
        return SubgraphDeps(
            settings=self.resolved_settings(),
            llm=self.llm,
            rag_service=self.rag_service,
            hospital_search=self.hospital_search,
            node_overrides=self.node_overrides,
            allow_missing_nodes=self.allow_missing_nodes,
            checkpointer=checkpointer,
        )


# ---------------------------------------------------------------------------
# 자체 node
# ---------------------------------------------------------------------------
def message_ingest_node(state: dict) -> dict:
    """사용자 메시지를 대화 이력에 편입한다.

    Context 적재(DB Context) 직후, 요약·응급 판단 앞에 둔다. 같은 메시지가 두 번
    쌓이지 않도록 마지막 항목과 비교한다(재개(resume) 시 같은 turn 이 다시 들어온다).
    """
    message = (state.get("user_message") or "").strip()
    if not message:
        return {}

    history = state.get("messages") or []
    if history:
        last = history[-1]
        last_content = last.get("content") if isinstance(last, dict) else getattr(last, "content", None)
        last_role = last.get("role") if isinstance(last, dict) else getattr(last, "type", None)
        if last_role in ("user", "human") and last_content == message:
            return {}

    return {"messages": [{"role": "user", "content": message}]}


def _make_hospital_flow_node(deps: GraphDependencies) -> NodeFn:
    """병원 검색 직접 요청(intent=hospital_search) 경로.

    응급 서브그래프를 타지 않고 요구사항 → 검색 → 적합도만 수행한다.
    응급이 아니므로 PDF·이메일·최소정보 수집은 하지 않는다.
    """
    sub = deps.to_subgraph_deps()
    requirements = resolve_node(
        sub, "hospital_requirements", ("hospital_requirements",),
        ("hospital_requirements_node",), ("make_hospital_requirements_node",),
    )
    search = resolve_node(
        sub, "hospital_search", ("hospital_search",),
        ("hospital_search_node",), ("make_hospital_search_node",),
        factory_arg=sub.resolved_hospital_search(),
    )
    suitability = resolve_node(
        sub, "hospital_suitability", ("hospital_suitability",),
        ("hospital_suitability_node",), ("make_hospital_suitability_node",),
    )

    def hospital_search_flow_node(state: dict) -> dict:
        merged: dict[str, Any] = {}
        working = dict(state)
        for node in (requirements, search, suitability):
            update = node(working) or {}
            merged.update(update)
            working.update(update)
        if not merged.get("draft_response"):
            merged["draft_response"] = _hospital_flow_message(merged, working)
        return merged

    return hospital_search_flow_node


def _hospital_flow_message(merged: dict, state: dict) -> str:
    """병원 검색 결과 안내문 — 실시간 진료 가능 여부를 단정하지 않는다(명세 34절)."""
    results = merged.get("hospital_results") or state.get("hospital_results") or []
    if not results:
        return (
            "지금은 조건에 맞는 병원 정보를 찾지 못했어요. "
            "지역명을 알려주시면 다시 찾아볼게요."
        )
    lines = ["가까운 동물병원 정보를 정리했어요."]
    for item in results[:3]:
        hospital = item.get("hospital", item) if isinstance(item, dict) else {}
        name = hospital.get("name", "이름 미확인")
        phone = hospital.get("phone") or "전화번호 미확인"
        lines.append(f"· {name} — {phone}")
    lines.append("방문 전에 전화로 현재 진료 및 응급 접수 가능 여부를 확인하세요.")
    return "\n".join(lines)


def _make_regenerate_node() -> NodeFn:
    """재생성 1회 — 재시도 횟수만 올리고 답변 생성 node 로 되돌린다(명세 40절)."""

    def regenerate_node(state: dict) -> dict:
        return {
            "retry_count": int(state.get("retry_count") or 0) + 1,
            "draft_response": "",
        }

    return regenerate_node


# ---------------------------------------------------------------------------
# 그래프 조립
# ---------------------------------------------------------------------------
def build_petcare_graph(
    deps: GraphDependencies | None = None,
    checkpointer: Any = None,
) -> Any:
    """명세 24절 메인 그래프를 조립해 compile 된 그래프를 돌려준다.

    Args:
        deps: 주입 컨테이너. None 이면 기본값(LLM 없음, fixture adapter).
        checkpointer: multi-turn interrupt/resume 용. None 이면 `InMemorySaver`
            를 자동 생성한다 — 명세 29절이 Colab 에서 resume 테스트를 요구하므로
            checkpointer 없는 그래프가 기본이 되면 안 된다.

    Returns:
        compile 된 LangGraph. `run_chat()` 으로 실행한다.
    """
    from langgraph.graph import END, START, StateGraph  # noqa: PLC0415

    deps = deps or GraphDependencies()
    if checkpointer is None:
        from langgraph.checkpoint.memory import InMemorySaver  # noqa: PLC0415

        checkpointer = InMemorySaver()

    if deps.clinical_adapter is not None:
        from .nodes import set_clinical_adapter  # noqa: PLC0415

        set_clinical_adapter(deps.clinical_adapter)

    sub = deps.to_subgraph_deps()
    from .state import PetCareState  # noqa: PLC0415

    graph = StateGraph(PetCareState)

    # ---- 중앙 흐름 node -------------------------------------------------
    graph.add_node(N_DB_CONTEXT, resolve_node(
        sub, N_DB_CONTEXT, ("db_context",), ("db_context_node",), ("make_db_context_node",),
        factory_arg=deps.clinical_adapter,
    ))
    graph.add_node(N_INGEST, deps.node_overrides.get(N_INGEST, message_ingest_node))
    graph.add_node(N_SUMMARY, resolve_node(
        sub, N_SUMMARY, ("conversation_summary",),
        ("conversation_summary_node",), ("make_conversation_summary_node",),
    ))
    graph.add_node(N_FAST_GUARD, resolve_node(
        sub, N_FAST_GUARD, ("fast_emergency_guard",), ("fast_emergency_guard_node",),
    ))
    graph.add_node(N_SUPERVISOR, resolve_node(
        sub, N_SUPERVISOR, ("supervisor",), ("supervisor_node",), ("make_supervisor_node",),
    ))
    graph.add_node(N_GENERAL, resolve_node(
        sub, N_GENERAL, ("general_chat",), ("general_chat_node",), ("make_general_chat_node",),
    ))
    graph.add_node(N_UNSUPPORTED, resolve_node(
        sub, N_UNSUPPORTED, ("general_chat",), ("unsupported_response_node",),
    ))
    graph.add_node(N_CLINICAL, resolve_node(
        sub, N_CLINICAL, ("clinical_context_priority",),
        ("clinical_context_priority_node",), ("make_clinical_context_priority_node",),
    ))
    graph.add_node(N_RULE, resolve_node(
        sub, N_RULE, ("assessment",), ("rule_assessment_node",),
    ))
    graph.add_node(N_MODEL, resolve_node(
        sub, N_MODEL, ("assessment",), ("assessment_agent_node",), ("make_assessment_agent_node",),
    ))
    graph.add_node(N_DOUBLE, resolve_node(
        sub, N_DOUBLE, ("risk_double_check",),
        ("risk_double_check_node",), ("make_risk_double_check_node",),
    ))
    graph.add_node(N_MERGE, resolve_node(
        sub, N_MERGE, ("risk_double_check",), ("merge_risk_node",),
    ))
    graph.add_node(N_HOSPITAL_FLOW, deps.node_overrides.get(
        N_HOSPITAL_FLOW, _make_hospital_flow_node(deps)
    ))

    # ---- 서브그래프 (부모와 같은 State 를 공유하므로 그대로 node 로 끼운다) ----
    from .subgraphs import (  # noqa: PLC0415
        build_emergency_subgraph,
        build_health_subgraph,
        build_visit_subgraph,
    )

    graph.add_node(N_HEALTH_SUB, build_health_subgraph(sub))
    graph.add_node(N_VISIT_SUB, build_visit_subgraph(sub))
    graph.add_node(N_EMERGENCY_SUB, build_emergency_subgraph(sub))

    # ---- 출력 검증 ------------------------------------------------------
    graph.add_node(N_OUTPUT_CHECK, resolve_node(
        sub, N_OUTPUT_CHECK, ("output_check",), ("output_check_node",), ("make_output_check_node",),
    ))
    graph.add_node("regenerate_once", _make_regenerate_node())
    graph.add_node(N_FINAL_SAFETY, resolve_node(
        sub, N_FINAL_SAFETY, ("final_safety",), ("final_safety_node",), ("make_final_safety_node",),
    ))
    graph.add_node(N_FALLBACK, resolve_node(
        sub, N_FALLBACK, ("final_safety",), ("safe_fallback_node",),
    ))
    graph.add_node(N_RESULT, resolve_node(
        sub, N_RESULT, ("final_safety",), ("build_result_node",),
    ))

    # ---- 간선 -----------------------------------------------------------
    # START → Context loaded?
    # 라우터는 **node 이름을 그대로 반환**하는 규약이다(routers.py docstring).
    # 별칭 매핑을 끼우면 이름이 갈라지므로 반환값 = key 로 둔다.
    graph.add_conditional_edges(
        START, route_context_loaded,
        {N_DB_CONTEXT: N_DB_CONTEXT, N_INGEST: N_INGEST, "ingest": N_INGEST},
    )
    graph.add_edge(N_DB_CONTEXT, N_INGEST)

    # Message Ingest → Need summary?
    graph.add_conditional_edges(
        N_INGEST, route_needs_summary,
        {N_SUMMARY: N_SUMMARY, N_FAST_GUARD: N_FAST_GUARD},
    )
    graph.add_edge(N_SUMMARY, N_FAST_GUARD)

    # Fast Emergency Guard → Critical?
    graph.add_conditional_edges(
        N_FAST_GUARD, route_after_fast_guard,
        {
            "emergency": N_EMERGENCY_SUB,
            N_EMERGENCY_SUB: N_EMERGENCY_SUB,
            "supervisor": N_SUPERVISOR,
            N_SUPERVISOR: N_SUPERVISOR,
        },
    )

    # Supervisor → Intent
    #
    # `general_knowledge` 는 **N_CLINICAL(위험도 평가)을 건너뛰고** health 서브그래프로
    # 곧장 간다. 지식 질문에는 판단할 대상(이 아이의 현재 상태)이 없어서 위험도를
    # 매기는 것 자체가 범주 오류이고, 일기 추세가 질문과 무관하게 위험도를 올려
    # visit 경로로 새는 일도 막는다.
    #
    # 서브그래프를 새로 만들지 않는 이유: health 서브그래프가 이미
    # `veterinary_rag → knowledge_sufficiency → (web) → evidence_merge → response`
    # 라는 RAG 체인을 갖고 있다. 지식 질문에 필요한 것이 정확히 그 체인이다.
    # 증상 문진만 건너뛰면 되고, 그 판단은 `route_missing_info` 가 한다.
    graph.add_conditional_edges(
        N_SUPERVISOR, route_intent,
        {
            "general_chat": N_GENERAL,
            "general_knowledge": N_HEALTH_SUB,
            "health_question": N_CLINICAL,
            "hospital_search": N_HOSPITAL_FLOW,
            "unsupported": N_UNSUPPORTED,
        },
    )

    # Clinical Context → Rule ∥ Model (병렬 fan-out → fan-in)
    graph.add_edge(N_CLINICAL, N_RULE)
    graph.add_edge(N_CLINICAL, N_MODEL)
    graph.add_edge(N_RULE, N_DOUBLE)
    graph.add_edge(N_MODEL, N_DOUBLE)
    graph.add_edge(N_DOUBLE, N_MERGE)

    # Merge Risk → Final Risk 분기
    graph.add_conditional_edges(
        N_MERGE, route_final_risk,
        {"normal": N_HEALTH_SUB, "visit": N_VISIT_SUB, "emergency": N_EMERGENCY_SUB},
    )

    # 모든 응답 경로 → Output Check
    for node in (N_GENERAL, N_UNSUPPORTED, N_HOSPITAL_FLOW,
                 N_HEALTH_SUB, N_VISIT_SUB, N_EMERGENCY_SUB):
        graph.add_edge(node, N_OUTPUT_CHECK)

    # Output Check → accept / regenerate(1회) / fallback
    graph.add_conditional_edges(
        N_OUTPUT_CHECK, route_output_check,
        {"accept": N_FINAL_SAFETY, "regenerate": "regenerate_once", "fallback": N_FALLBACK},
    )
    graph.add_edge("regenerate_once", N_OUTPUT_CHECK)
    graph.add_edge(N_FINAL_SAFETY, N_RESULT)
    graph.add_edge(N_FALLBACK, N_RESULT)
    graph.add_edge(N_RESULT, END)

    return graph.compile(checkpointer=checkpointer)


# ---------------------------------------------------------------------------
# LangSmith metadata (명세 42절)
# ---------------------------------------------------------------------------
def build_trace_metadata(state: dict) -> dict[str, Any]:
    """trace 에 남길 metadata.

    **민감한 원문·개인 식별정보를 넣지 않는다**(명세 42절). 사용자 메시지,
    반려동물 이름, 진단 내용은 제외하고 분기 결과와 boolean 지표만 남긴다.
    """
    settings = None
    try:
        from ..config import get_settings  # noqa: PLC0415

        settings = get_settings()
    except Exception:  # pragma: no cover - 설정 로드 실패는 trace 를 막지 않는다
        pass

    return {
        "environment": getattr(settings, "environment", "colab"),
        # pet_id 는 내부 식별자이므로 그대로 두되 이름·연락처는 넣지 않는다.
        "pet_id": state.get("pet_id"),
        "intent": state.get("intent"),
        "final_risk": state.get("final_risk"),
        "emergency_urgency": state.get("emergency_urgency"),
        "rag_sufficiency": state.get("rag_sufficiency"),
        "web_fallback_triggered": bool(state.get("web_fallback_required")),
        "hospital_search_triggered": bool(state.get("hospital_search_queries")),
        "red_flag_count": len(state.get("red_flags") or []),
        "missing_field_count": len(state.get("missing_fields") or []),
        "pdf_generated": bool(state.get("pdf_path")),
        "email_draft_created": bool(state.get("email_draft")),
        "retry_count": int(state.get("retry_count") or 0),
        "validation_error_count": len(state.get("validation_errors") or []),
    }


# ---------------------------------------------------------------------------
# 실행 헬퍼
# ---------------------------------------------------------------------------
def run_chat(
    graph: Any,
    pet_id: int | None = None,
    user_message: str | None = None,
    thread_id: str | None = None,
    region_name: str | None = None,
    resume: Any = None,
    *,
    config: dict | None = None,
    **state_kwargs: Any,
) -> ChatGraphResult:
    """그래프를 1턴 실행하고 최종 결과를 돌려준다.

    최초 호출과 재개(resume)를 모두 처리한다. 재개는 같은 `thread_id` 로
    `Command(resume=...)` 를 넣는 방식이다(명세 29절).

    Args:
        graph: `build_petcare_graph()` 결과.
        pet_id/user_message: 최초 호출 시 필수.
        thread_id: 대화 스레드. 없으면 새로 만든다. **pet 별로 분리해야**
            다른 아이의 대화가 섞이지 않는다(명세 43절).
        region_name: Android 위치 대신 쓰는 테스트 입력(명세 32절).
        resume: interrupt 에 대한 사용자 응답. 주면 재개 실행이 된다.
        config: LangGraph config 를 직접 넘길 때(예: LangSmith metadata 추가).

    Returns:
        ChatGraphResult. `interrupt` 로 멈춘 경우에는 `missing_information` 에
        물어볼 항목이 담기고 `message` 에 질문이 들어간다.
    """
    thread_id = thread_id or f"thread-{uuid.uuid4().hex[:12]}"
    run_config = dict(config or {})
    configurable = dict(run_config.get("configurable") or {})
    configurable.setdefault("thread_id", thread_id)
    run_config["configurable"] = configurable

    if resume is not None:
        from langgraph.types import Command  # noqa: PLC0415

        payload: Any = Command(resume=resume)
    else:
        if pet_id is None or user_message is None:
            raise ValueError("최초 호출에는 pet_id 와 user_message 가 필요합니다.")
        from .state import make_initial_state  # noqa: PLC0415

        payload = make_initial_state(
            pet_id=pet_id,
            user_message=user_message,
            thread_id=thread_id,
            region_name=region_name,
            **state_kwargs,
        )

    result_state = graph.invoke(payload, config=run_config)
    return _to_result(graph, result_state, run_config, thread_id)


def _to_result(graph: Any, state: dict, run_config: dict, thread_id: str) -> ChatGraphResult:
    """최종 State 를 ChatGraphResult 로 만든다.

    interrupt 로 멈췄으면 그래프가 최종 결과를 만들지 않으므로, 남은 질문을
    담은 결과를 대신 구성한다(호출자가 항상 같은 타입을 받도록).
    """
    built = state.get("__result__")
    if isinstance(built, ChatGraphResult):
        result = built
    elif isinstance(built, dict):
        result = ChatGraphResult(**built)
    else:
        result = _result_from_state(state)

    metadata = dict(result.trace_metadata or {})
    metadata.update(build_trace_metadata(state))
    metadata["thread_id"] = thread_id

    interrupts = _pending_interrupts(graph, run_config)
    if interrupts:
        metadata["interrupted"] = True
        if not result.message:
            result.message = interrupts[0]
    result.trace_metadata = metadata
    return result


def _result_from_state(state: dict) -> ChatGraphResult:
    """State 만으로 결과를 조립한다(build_result node 를 타지 못한 경우)."""
    from ..schemas import EmailDraft, FinalEvidence, HospitalSuitabilityResult  # noqa: PLC0415

    def _coerce(items: Any, model: Any) -> list:
        out = []
        for item in items or []:
            if isinstance(item, model):
                out.append(item)
            elif isinstance(item, dict):
                try:
                    out.append(model(**item))
                except Exception:  # pragma: no cover - 부분 데이터는 건너뛴다
                    continue
        return out

    email = state.get("email_draft")
    if isinstance(email, dict):
        try:
            email = EmailDraft(**email)
        except Exception:  # pragma: no cover
            email = None

    return ChatGraphResult(
        message=state.get("final_response") or state.get("draft_response") or "",
        risk_level=state.get("final_risk") or "normal",
        emergency_urgency=state.get("emergency_urgency") or "none",
        missing_information=list(state.get("missing_fields") or []),
        hospitals=_coerce(state.get("hospital_results"), HospitalSuitabilityResult),
        pdf_path=state.get("pdf_path"),
        email_draft=email if isinstance(email, EmailDraft) else None,
        ui_actions=list(state.get("ui_actions") or []),
        evidence=_coerce(state.get("merged_evidence"), FinalEvidence),
        trace_metadata={},
    )


def _pending_interrupts(graph: Any, run_config: dict) -> list[str]:
    """대기 중인 interrupt 질문 목록을 읽는다(없으면 빈 리스트)."""
    try:
        snapshot = graph.get_state(run_config)
    except Exception:  # pragma: no cover - checkpointer 미사용 등
        return []

    questions: list[str] = []
    for task in getattr(snapshot, "tasks", ()) or ():
        for interrupt in getattr(task, "interrupts", ()) or ():
            value = getattr(interrupt, "value", None)
            if isinstance(value, dict):
                text = value.get("question") or value.get("message")
            else:
                text = value
            if text:
                questions.append(str(text))
    return questions


def describe_graph(graph: Any = None) -> str:
    """그래프 구조를 텍스트로 요약한다(노트북 확인용).

    Mermaid 렌더가 실패해도(그래프 객체 없음, 환경 제약) 항상 읽을 수 있는
    고정 설명을 돌려주므로 노트북 셀이 죽지 않는다.
    """
    lines = [
        "PetCare 메인 그래프 (명세 24절)",
        "",
        "  START → [Context loaded?]",
        f"      ├─ No  → {N_DB_CONTEXT}",
        f"      └─ Yes → {N_INGEST}",
        f"  {N_INGEST} → [Need summary?] → {N_SUMMARY} → {N_FAST_GUARD}",
        f"  {N_FAST_GUARD} → [Critical?]",
        f"      ├─ Yes → {N_EMERGENCY_SUB}",
        f"      └─ No  → {N_SUPERVISOR}",
        f"  {N_SUPERVISOR} → [Intent]",
        f"      ├─ general_chat    → {N_GENERAL}",
        f"      ├─ health_question → {N_CLINICAL}",
        f"      ├─ hospital_search → {N_HOSPITAL_FLOW}",
        f"      └─ unsupported     → {N_UNSUPPORTED}",
        f"  {N_CLINICAL} → ({N_RULE} ∥ {N_MODEL}) → {N_DOUBLE} → {N_MERGE}",
        f"  {N_MERGE} → [Final Risk]",
        f"      ├─ normal    → {N_HEALTH_SUB}",
        f"      ├─ visit     → {N_VISIT_SUB}",
        f"      └─ emergency → {N_EMERGENCY_SUB}",
        f"  모든 응답 경로 → {N_OUTPUT_CHECK} → [Valid?]",
        f"      ├─ accept     → {N_FINAL_SAFETY}",
        "      ├─ regenerate → regenerate_once → output_check (최대 1회)",
        f"      └─ fallback   → {N_FALLBACK}",
        f"  → {N_RESULT} → END",
    ]
    if graph is not None:
        try:
            mermaid = graph.get_graph().draw_mermaid()
            lines += ["", "--- Mermaid ---", mermaid]
        except Exception as exc:  # pragma: no cover - 렌더 실패는 무시
            lines += ["", f"(Mermaid 렌더 불가: {exc})"]
    return "\n".join(lines)


__all__ = [
    "GraphDependencies",
    "build_petcare_graph",
    "build_trace_metadata",
    "run_chat",
    "describe_graph",
    "message_ingest_node",
]
