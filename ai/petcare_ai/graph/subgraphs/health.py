"""Health Subgraph — 명세 30절 mermaid 를 그대로 구현한다.

```mermaid
flowchart TD
    A[Health] --> B[Missing Information Agent]
    B --> C{Enough user info?}
    C -->|No| D[Interrupt]
    D --> B
    C -->|Yes| E[Veterinary RAG]
    E --> F[Knowledge Sufficiency]
    F --> G{RAG status}
    G -->|Sufficient| H[Evidence Merge]
    G -->|Insufficient| I[Tavily Vet Search]
    G -->|Conflicting| J[Conflict Handler]
    I --> K[Web Source Validator]
    K --> H
    J --> H
    H --> L[Health Response Agent]
```

## 설계 메모

**왜 `VeterinaryRagService.retrieve_with_fallback()` 한 방으로 끝내지 않는가**
그 메서드는 검색·충분성·웹 fallback·병합을 한 함수 안에서 끝낸다. 편하지만
mermaid 의 E·F·G·I·K·H 가 전부 한 node 로 뭉개져서 **LangSmith trace 에 분기가
보이지 않고**(명세 19·42절), 조건부 분기를 LangGraph 가 하는 것도 아니게 된다.
그래서 여기서는 RAG 계층의 개별 함수(`build_rag_query` / `retrieve` /
`KnowledgeSufficiencyEvaluator` / `VeterinaryWebSearchService` /
`WebSourceValidator` / `merge_evidence`)를 node 로 나눠 붙인다. 로직은 전부 RAG
계층 것을 그대로 쓰며 여기서 재구현하지 않는다.

**Tavily 는 부족할 때만 부른다(명세 15·30절).** 호출 여부를 판단하는 코드는 이
파일에 없다 — `routers.route_rag_status()` 의 분기 결과가 곧 호출 여부다. 즉
"모든 질문에 Tavily 호출" 이 구조적으로 불가능하다.

**LLM 이 없어도 전 경로가 돈다.** query builder·충분성 평가기는 `llm=None` 이면
규칙 기반으로 동작하고, Tavily 키가 없으면 빈 리스트를 돌려주며(예외 아님),
근거가 하나도 없으면 `has_reliable_evidence=False` 로 Health Response 가
"확실하지 않음 + 병원 상담 권고" 를 답한다.

**FAISS index 가 없어도 죽지 않는다.** 검색 예외는 빈 결과로 흡수되어
`insufficient` → 웹 fallback 경로로 이어진다. 오프라인 테스트가 여기서 막히면
안 되기 때문이다.
"""

from __future__ import annotations

import logging
from typing import Any

from ..routers import route_missing_info, route_rag_status
from ..state import Replace
from . import NodeFn, SubgraphDeps, resolve_node, with_sanitized_collected

logger = logging.getLogger(__name__)

__all__ = [
    "build_health_subgraph",
    "NODE_MISSING_INFORMATION",
    "NODE_MISSING_INFORMATION_INTERRUPT",
    "NODE_VETERINARY_RAG",
    "NODE_KNOWLEDGE_SUFFICIENCY",
    "NODE_TAVILY_VET_SEARCH",
    "NODE_WEB_SOURCE_VALIDATOR",
    "NODE_CONFLICT_HANDLER",
    "NODE_EVIDENCE_MERGE",
    "NODE_HEALTH_RESPONSE",
    "make_veterinary_rag_node",
    "make_knowledge_sufficiency_node",
    "make_tavily_vet_search_node",
    "make_web_source_validator_node",
    "conflict_handler_node",
    "evidence_merge_node",
]

# graph 안에서 쓰는 node 이름 — 테스트와 builder 가 문자열을 하드코딩하지 않도록 노출한다.
NODE_MISSING_INFORMATION = "missing_information"
NODE_MISSING_INFORMATION_INTERRUPT = "missing_information_interrupt"
NODE_VETERINARY_RAG = "veterinary_rag"
NODE_KNOWLEDGE_SUFFICIENCY = "knowledge_sufficiency"
NODE_TAVILY_VET_SEARCH = "tavily_vet_search"
NODE_WEB_SOURCE_VALIDATOR = "web_source_validator"
NODE_CONFLICT_HANDLER = "conflict_handler"
NODE_EVIDENCE_MERGE = "evidence_merge"
NODE_HEALTH_RESPONSE = "health_response"

#: 웹 검색 최대 건수 — 검증에서 대부분 떨어지므로 넉넉히 5건을 받는다(명세 15절).
WEB_SEARCH_MAX_RESULTS = 5


# ---------------------------------------------------------------------------
# State ↔ RAG 스키마 변환
# ---------------------------------------------------------------------------
def _species(state: dict) -> str:
    """검색 대상 종을 정한다 — 확실하지 않으면 `dog` 로 두지 않고 프로필을 다시 본다.

    species 를 잘못 넘기면 고양이 질문에 개 문서를 붙이게 된다(명세 11절 index 분리).
    State 의 `species` → PET DB 순으로 확인하고, 그래도 모르면 `dog` 를 쓰되
    경고를 남긴다(RAG index 는 둘 중 하나를 반드시 골라야 한다).
    """
    value = state.get("species")
    if value in ("dog", "cat"):
        return value

    from ..state import _species_from_profile  # noqa: PLC0415

    resolved = _species_from_profile(state.get("pet_profile") or {})
    if resolved is not None:
        return resolved
    logger.warning("PET DB 에서 종을 확정하지 못해 'dog' index 로 검색합니다.")
    return "dog"


def _rag_query_from_state(state: dict) -> Any:
    """State 에 저장된 query 필드로 `RagQuery` 를 복원한다.

    node 사이에 pydantic 객체를 그대로 들고 다닐 수 없어서(State 는 JSON 직렬화
    되어 checkpoint·LangSmith 로 나간다) 문자열 필드로 저장했다가 필요할 때 다시
    조립한다.
    """
    from ...schemas import RagQuery  # noqa: PLC0415

    ko = str(state.get("rag_query_ko") or state.get("rag_query") or "")
    en = str(state.get("rag_query_en") or ko)
    return RagQuery(
        primary_query_ko=ko,
        primary_query_en=en,
        required_topics=[str(t) for t in (state.get("rag_required_topics") or [])],
        species=_species(state),  # type: ignore[arg-type]
        emergency_hint=str(state.get("final_risk") or "normal") != "normal",
    )


def _retrieved_from_state(state: dict) -> list[Any]:
    """State 의 `rag_documents`(dict) 를 `RetrievedEvidence` 로 되돌린다."""
    from ...schemas import RetrievedEvidence  # noqa: PLC0415

    documents: list[Any] = []
    for raw in state.get("rag_documents") or []:
        if not isinstance(raw, dict):
            continue
        try:
            documents.append(RetrievedEvidence(**raw))
        except Exception as exc:  # 스키마가 깨진 항목 하나로 전체를 죽이지 않는다
            logger.warning("rag_documents 항목을 복원하지 못했습니다: %s", exc)
    return documents


def _web_from_state(state: dict, accepted_only_flag: bool) -> list[Any]:
    """State 의 `validated_web_evidence`(dict) 를 `WebEvidence` 로 되돌린다."""
    from ...schemas import WebEvidence  # noqa: PLC0415

    items: list[Any] = []
    for raw in state.get("validated_web_evidence") or []:
        if not isinstance(raw, dict):
            continue
        if accepted_only_flag and not raw.get("accepted"):
            continue
        try:
            items.append(WebEvidence(**raw))
        except Exception as exc:
            logger.warning("validated_web_evidence 항목을 복원하지 못했습니다: %s", exc)
    return items


# ---------------------------------------------------------------------------
# E. Veterinary RAG
# ---------------------------------------------------------------------------
def make_veterinary_rag_node(deps: SubgraphDeps) -> NodeFn:
    """Query Builder(명세 12절) + Retriever(13절)를 수행하는 node 를 만든다.

    충분성 판정은 **여기서 하지 않는다.** 다음 node(F)가 담당해야 mermaid 의
    E → F → G 분기가 trace 에 그대로 보인다.
    """

    def _node(state: dict) -> dict:
        from ...rag.query_builder import build_rag_query  # noqa: PLC0415
        from ...rag.retriever import retrieve  # noqa: PLC0415

        service = deps.resolved_rag_service()
        settings = deps.resolved_settings()

        query = build_rag_query(
            user_message=str(state.get("user_message") or ""),
            pet_profile=dict(state.get("pet_profile") or {}),
            related_diagnoses=list(state.get("related_diagnoses") or []),
            supporting_daily_entries=list(state.get("supporting_daily_entries") or []),
            llm=deps.llm,
        )
        # 종은 PET DB 판정이 이긴다(명세 11절: index 를 고르는 값이다).
        resolved_species = _species(state)
        if query.species != resolved_species:
            query = query.model_copy(update={"species": resolved_species})

        try:
            documents = retrieve(service.store, query, settings)
        except Exception as exc:
            # index 미생성·faiss 미설치·손상된 index 는 모두 "근거 없음" 으로 흡수한다.
            # 여기서 예외를 올리면 오프라인 환경에서 상담 자체가 불가능해진다.
            logger.warning("RAG 검색 실패 — 근거 없음으로 진행합니다: %s", exc)
            documents = []

        logger.info(
            "Veterinary RAG: species=%s ko=%r docs=%d",
            query.species,
            query.primary_query_ko[:40],
            len(documents),
        )
        return {
            "rag_query": query.primary_query_ko,
            "rag_query_ko": query.primary_query_ko,
            "rag_query_en": query.primary_query_en,
            "rag_required_topics": Replace(query.required_topics),
            "species": query.species,
            "rag_documents": Replace([doc.model_dump() for doc in documents]),
        }

    return _node


# ---------------------------------------------------------------------------
# F. Knowledge Sufficiency
# ---------------------------------------------------------------------------
def make_knowledge_sufficiency_node(deps: SubgraphDeps) -> NodeFn:
    """충분성 판정 node(명세 14절)를 만든다.

    `KnowledgeSufficiencyEvaluator` 는 deterministic 검사를 먼저 하고 애매할 때만
    LLM 을 부른다. LLM 이 `sufficient` 라고 해도 빈 결과·species 불일치는
    `enforce_hard_guards()` 가 `insufficient` 로 되돌린다 — 그 규칙을 여기서
    다시 쓰지 않고 평가기를 그대로 신뢰한다.
    """

    def _node(state: dict) -> dict:
        service = deps.resolved_rag_service()
        query = _rag_query_from_state(state)
        documents = _retrieved_from_state(state)

        try:
            result = service.evaluator.evaluate(query, documents)
        except Exception as exc:
            logger.warning("충분성 판정 실패 — insufficient 로 처리합니다: %s", exc)
            from ...schemas import KnowledgeSufficiencyResult  # noqa: PLC0415

            result = KnowledgeSufficiencyResult(
                status="insufficient",
                missing_topics=list(query.required_topics),
                reason=f"충분성 판정 중 오류가 발생해 부족으로 처리했습니다: {exc}",
            )

        # 명세 15절: 부족하거나 충돌하거나, 최신 정보가 필요한 질문이면 웹 fallback.
        web_fallback = result.status != "sufficient" or result.requires_recent_information

        logger.info(
            "Knowledge Sufficiency: status=%s missing=%s web_fallback=%s (%s)",
            result.status,
            result.missing_topics,
            web_fallback,
            result.reason[:80],
        )
        return {
            "rag_sufficiency": result.status,
            "rag_missing_topics": Replace(result.missing_topics),
            "web_fallback_required": bool(web_fallback),
        }

    return _node


# ---------------------------------------------------------------------------
# I. Tavily Vet Search  /  K. Web Source Validator
# ---------------------------------------------------------------------------
def make_tavily_vet_search_node(deps: SubgraphDeps) -> NodeFn:
    """수의학 웹 검색 node(명세 15절)를 만든다 — **RAG 가 부족할 때만 실행된다.**

    검색 결과는 아직 근거가 아니다. `accepted=False` 상태로 State 에 넣어 두고,
    다음 node(Web Source Validator)가 allowlist·관련도·광고성 검사를 통과시킨
    것만 `accepted=True` 로 바꾼다. Evidence Merge 는 `accepted` 만 읽으므로
    **검증을 건너뛴 웹 정보가 답변에 들어갈 경로가 없다.**

    키 없음·호출 실패·결과 없음은 전부 정상 fallback 실패이며 빈 리스트가 된다.
    """

    def _node(state: dict) -> dict:
        service = deps.resolved_rag_service()
        query = _rag_query_from_state(state)
        species = query.species

        try:
            items = service.web_search.search(
                query.primary_query_en, species, max_results=WEB_SEARCH_MAX_RESULTS
            )
            # 영어 query 가 비었거나 결과가 없으면 한국어 query 로 한 번 더 시도한다.
            if not items and query.primary_query_ko != query.primary_query_en:
                items = service.web_search.search(
                    query.primary_query_ko, species, max_results=WEB_SEARCH_MAX_RESULTS
                )
        except Exception as exc:
            logger.warning("Tavily 수의학 검색 실패 — 웹 근거 없이 진행합니다: %s", exc)
            items = []

        logger.info("Tavily Vet Search: %d건 수집(검증 전)", len(items))
        return {"validated_web_evidence": Replace([item.model_dump() for item in items])}

    return _node


def make_web_source_validator_node(deps: SubgraphDeps) -> NodeFn:
    """웹 근거 검증 node(명세 15절)를 만든다.

    거절된 항목도 `reject_reason` 과 함께 State 에 **남겨 둔다.** 지우면
    "왜 이 근거를 안 썼는가" 를 trace 에서 확인할 수 없다. 답변에 쓰이는 것은
    Evidence Merge 가 고르는 `accepted=True` 항목뿐이다.
    """

    def _node(state: dict) -> dict:
        service = deps.resolved_rag_service()
        query = _rag_query_from_state(state)
        items = _web_from_state(state, accepted_only_flag=False)
        if not items:
            return {}

        try:
            validated = service.validator.validate(items, query.species, query)
        except Exception as exc:
            # 검증기가 죽으면 웹 근거를 **전부 버린다.** 검증 없이 쓰는 것보다 안전하다.
            logger.warning("웹 근거 검증 실패 — 웹 근거를 모두 버립니다: %s", exc)
            return {"validated_web_evidence": Replace([])}

        accepted = sum(1 for item in validated if item.accepted)
        logger.info("Web Source Validator: %d건 중 %d건 채택", len(validated), accepted)
        return {"validated_web_evidence": Replace([item.model_dump() for item in validated])}

    return _node


# ---------------------------------------------------------------------------
# J. Conflict Handler
# ---------------------------------------------------------------------------
def conflict_handler_node(state: dict) -> dict:
    """근거끼리 충돌할 때의 처리 node(명세 30절 J).

    **충돌을 조용히 한쪽으로 정리하지 않는다.** 어느 쪽이 맞는지는 이 시스템이
    판단할 수 없는 영역이라(수의사의 몫), 충돌 사실을 그대로 기록하고 답변이
    단정하지 않도록 근거 목록에 남긴다. 명세 30절 mermaid 대로 웹 검색을 거치지
    않고 곧바로 Evidence Merge 로 간다 — 충돌은 정보 부족이 아니라 정보 해석의
    문제라 웹을 더 뒤진다고 해결되지 않기 때문이다.

    위험도는 여기서 올리지 않는다. 위험도는 명세 28절의 평가자들만 정한다.
    """
    missing = [str(topic) for topic in (state.get("rag_missing_topics") or [])]
    detail = f" (쟁점: {', '.join(missing)})" if missing else ""
    note = (
        "내부 수의학 자료 사이에 서로 다른 설명이 있어 한 가지로 단정하지 않았습니다."
        f"{detail}"
    )
    logger.info("Conflict Handler: %s", note)
    return {
        "evidence_conflicts": [note],
        "risk_reasons": [f"[근거충돌] {note}"],
        "web_fallback_required": False,
    }


# ---------------------------------------------------------------------------
# H. Evidence Merge
# ---------------------------------------------------------------------------
def evidence_merge_node(state: dict) -> dict:
    """RAG 근거 + **검증된** 웹 근거를 병합한다(명세 16절).

    병합 순서 자체가 우선순위다(RAG 먼저). 검증을 통과하지 못한 웹 항목은
    `accepted=False` 라 여기서 걸러진다.

    `has_reliable_evidence=False` 는 오류가 아니다 — Health Response 가 추측하지
    않고 "확실하지 않음 + 병원 상담 권고" 로 답해야 한다는 신호다.
    """
    from ...rag.evidence_merger import merge_evidence  # noqa: PLC0415

    rag_docs = _retrieved_from_state(state)
    web_docs = _web_from_state(state, accepted_only_flag=True)
    topics = [str(t) for t in (state.get("rag_required_topics") or [])]

    merged = merge_evidence(rag_docs, web_docs, topics)
    logger.info(
        "Evidence Merge: rag=%d web=%d -> %d건 (reliable=%s)",
        len(rag_docs),
        len(web_docs),
        len(merged.evidence),
        merged.has_reliable_evidence,
    )
    return {
        "merged_evidence": Replace([item.model_dump() for item in merged.evidence]),
        "evidence_conflicts": list(merged.conflicts),
        "has_reliable_evidence": bool(merged.has_reliable_evidence),
    }


# ---------------------------------------------------------------------------
# 서브그래프 조립
# ---------------------------------------------------------------------------
def build_health_subgraph(deps: SubgraphDeps | None = None) -> Any:
    """명세 30절 Health Subgraph 를 compile 해서 돌려준다.

    반환값은 compile 된 LangGraph 그래프라 부모 그래프에
    `builder.add_node("health_subgraph", build_health_subgraph(deps))` 로
    그대로 끼울 수 있다. State 스키마가 부모와 같으므로 값이 자동으로 오간다.

    Args:
        deps: 주입 컨테이너. `None` 이면 LLM·키 없이 규칙 기반으로 동작한다.
    """
    from langgraph.graph import END, START, StateGraph  # noqa: PLC0415

    from ..state import PetCareState  # noqa: PLC0415

    resolved = deps or SubgraphDeps()
    graph = StateGraph(PetCareState)

    # B: Missing Information Agent — 되묻기 횟수를 세는 wrapper 로 감싼다.
    from . import make_missing_information_gate  # noqa: PLC0415

    graph.add_node(
        NODE_MISSING_INFORMATION,
        make_missing_information_gate(
            resolved,
            NODE_MISSING_INFORMATION,
            ("missing_information",),
            ("missing_information_node",),
        ),
    )
    # D: Interrupt — langgraph.types.interrupt() 로 멈추고 Command(resume=)로 재개
    graph.add_node(
        NODE_MISSING_INFORMATION_INTERRUPT,
        resolve_node(
            resolved,
            NODE_MISSING_INFORMATION_INTERRUPT,
            ("missing_information",),
            ("missing_information_interrupt_node",),
        ),
    )
    graph.add_node(NODE_VETERINARY_RAG, make_veterinary_rag_node(resolved))
    graph.add_node(NODE_KNOWLEDGE_SUFFICIENCY, make_knowledge_sufficiency_node(resolved))
    graph.add_node(NODE_TAVILY_VET_SEARCH, make_tavily_vet_search_node(resolved))
    graph.add_node(NODE_WEB_SOURCE_VALIDATOR, make_web_source_validator_node(resolved))
    graph.add_node(NODE_CONFLICT_HANDLER, conflict_handler_node)
    graph.add_node(NODE_EVIDENCE_MERGE, evidence_merge_node)
    # Health Response 는 `collected_information` 을 프롬프트에 통째로 넣는다.
    # 내부 관리용 key 가 LLM 프롬프트에 섞이지 않도록 입력을 정리해 넘긴다.
    graph.add_node(
        NODE_HEALTH_RESPONSE,
        with_sanitized_collected(
            resolve_node(
                resolved,
                NODE_HEALTH_RESPONSE,
                ("health_response",),
                ("health_response_node",),
                factories=("make_health_response_node",),
            )
        ),
    )

    graph.add_edge(START, NODE_MISSING_INFORMATION)

    # C: Enough user info?  — 분기 판정은 routers 가 소유한다(명세 19절).
    graph.add_conditional_edges(
        NODE_MISSING_INFORMATION,
        route_missing_info,
        {
            "ask": NODE_MISSING_INFORMATION_INTERRUPT,
            "ready": NODE_VETERINARY_RAG,
        },
    )
    # D → B: 답변을 받은 뒤 다시 판정한다(mermaid 의 순환).
    graph.add_edge(NODE_MISSING_INFORMATION_INTERRUPT, NODE_MISSING_INFORMATION)

    graph.add_edge(NODE_VETERINARY_RAG, NODE_KNOWLEDGE_SUFFICIENCY)

    # G: RAG status — Tavily 는 이 분기에서만 도달할 수 있다.
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

    graph.add_edge(NODE_EVIDENCE_MERGE, NODE_HEALTH_RESPONSE)
    graph.add_edge(NODE_HEALTH_RESPONSE, END)

    try:
        return graph.compile(name="health_subgraph", **resolved.compile_kwargs())
    except TypeError:  # 구버전 langgraph 는 name 인자를 받지 않는다
        return graph.compile(**resolved.compile_kwargs())
