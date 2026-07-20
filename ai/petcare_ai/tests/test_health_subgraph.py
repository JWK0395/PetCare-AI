"""Health Subgraph 테스트 (명세 30·43절).

명세 30절 mermaid 의 분기가 **실제로 갈라지는지**를 본다.

    Missing Info → (interrupt) → Veterinary RAG → Knowledge Sufficiency
      → sufficient   : Tavily 를 **호출하지 않고** Evidence Merge
      → insufficient : Tavily → Web Source Validator → Evidence Merge
      → conflicting  : Conflict Handler → Evidence Merge

명세 43절 항목 대응
  - 일반 건강 상담('오늘 한 번 토했는데 지금은 잘 놀아요') : normal 가능 / RAG 호출 /
    충분하면 Tavily 미호출
  - RAG fallback : `rag_sufficiency=insufficient` / Tavily 호출 / **검증된 출처만** evidence
  - 종 분리 : 고양이 fixture 는 cat index 만 조회하고 최종 evidence 에 dog 문서가 없다

원칙
  * LLM 주입 없음(None) — 규칙 경로만 돈다.
  * Tavily·임베딩·FAISS 없음 — store / evaluator / web_search 는 전부 mock 주입.
  * `WebSourceValidator` 는 **진짜를 쓴다.** "검증된 출처만" 이 이 테스트의 핵심이라
    검증기를 mock 으로 바꾸면 아무것도 검증하지 못한다.
"""

from __future__ import annotations

from typing import Any

import pytest

from petcare_ai.adapters.clinical_data_adapter import FixtureClinicalDataAdapter
from petcare_ai.config import get_settings
from petcare_ai.graph.nodes.assessment import evaluate_rules
from petcare_ai.graph.nodes.health_response import NO_EVIDENCE_MESSAGE
from petcare_ai.graph.nodes.output_check import find_forbidden_expressions
from petcare_ai.graph.prompts import MEDICAL_DISCLAIMER
from petcare_ai.graph.state import make_initial_state
from petcare_ai.graph.subgraphs import SubgraphDeps
from petcare_ai.graph.subgraphs.health import build_health_subgraph
from petcare_ai.rag.source_validator import WebSourceValidator
from petcare_ai.schemas import KnowledgeSufficiencyResult, RetrievedEvidence, WebEvidence

DOG_PET_ID = 1
CAT_PET_ID = 2

VOMIT_MESSAGE = "강아지가 오늘 한 번 토했는데 지금은 잘 놀아요"
CAT_MESSAGE = "고양이가 오늘 아침에 헤어볼을 한 번 토했어요"


# ---------------------------------------------------------------------------
# mock RAG 계층
# ---------------------------------------------------------------------------
def _evidence(chunk_id: str, species: str, title: str, text: str) -> RetrievedEvidence:
    return RetrievedEvidence(
        chunk_id=chunk_id,
        document_id=f"doc-{chunk_id}",
        title=title,
        text=text,
        species=species,  # type: ignore[arg-type]
        source="Cornell Riney Canine Health Center"
        if species == "dog"
        else "Cornell Feline Health Center",
        source_url=f"https://vet.cornell.edu/{species}/{chunk_id}",
        categories=["Digestive system"],
        score=0.82,
        heading_path=["Vomiting", "When to call"],
    )


DOG_DOCS = [
    _evidence(
        "dog-1",
        "dog",
        "Vomiting in dogs",
        "A single episode of vomiting in an otherwise bright and playful dog is often "
        "self-limiting. Contact your veterinarian if vomiting repeats or blood appears.",
    ),
    _evidence(
        "dog-2",
        "dog",
        "Monitoring appetite in dogs",
        "Reduced appetite together with reduced activity in a dog warrants veterinary review.",
    ),
]

CAT_DOCS = [
    _evidence(
        "cat-1",
        "cat",
        "Hairballs in cats",
        "Occasional hairballs are common in cats. Frequent retching or vomiting in a cat "
        "should be evaluated by a veterinarian.",
    ),
]


class FakeVectorStore:
    """`rag.retriever.retrieve()` 가 요구하는 최소 인터페이스만 구현한 store.

    종별 index 분리를 흉내 내되, `cat` 버킷에 dog 문서를 **일부러 하나 섞어 둔다.**
    retriever 의 2차 방어선(`deduplicate_evidence(species=...)`)이 교차 오염을
    실제로 걷어내는지 확인하기 위해서다.
    """

    def __init__(self, buckets: dict[str, list[RetrievedEvidence]]) -> None:
        self._buckets = buckets
        self.calls: list[tuple[str, str]] = []

    @property
    def loaded_species(self) -> set[str]:
        return set(self._buckets)

    def search(
        self, query: str, species: str, k: int = 6, fetch_k: int = 20
    ) -> list[RetrievedEvidence]:
        self.calls.append((query, species))
        return list(self._buckets.get(species, []))[:k]

    @property
    def searched_species(self) -> set[str]:
        return {species for _, species in self.calls}


class FakeSufficiencyEvaluator:
    """충분성 판정을 고정한다 — 분기 자체를 검증하려면 판정이 결정론적이어야 한다."""

    def __init__(
        self,
        status: str,
        missing_topics: tuple[str, ...] = (),
        requires_recent_information: bool = False,
    ) -> None:
        self.status = status
        self.missing_topics = list(missing_topics)
        self.requires_recent_information = requires_recent_information
        self.calls: list[tuple[Any, list[RetrievedEvidence]]] = []

    def evaluate(self, query: Any, documents: list[RetrievedEvidence]) -> KnowledgeSufficiencyResult:
        self.calls.append((query, list(documents)))
        return KnowledgeSufficiencyResult(
            status=self.status,  # type: ignore[arg-type]
            covered_topics=[],
            missing_topics=self.missing_topics,
            requires_recent_information=self.requires_recent_information,
            reason="테스트 고정 판정",
        )


class FakeWebSearch:
    """Tavily 대체 — 호출 여부를 세는 것이 이 mock 의 존재 이유다."""

    def __init__(self, results: list[WebEvidence] | None = None) -> None:
        self._results = list(results or [])
        self.calls: list[tuple[str, str, int]] = []

    def search(self, query: str, species: str, max_results: int = 5) -> list[WebEvidence]:
        self.calls.append((query, species, max_results))
        return [item.model_copy() for item in self._results]


class FakeRagService:
    """`VeterinaryRagService` 중 health subgraph 가 실제로 쓰는 속성만 노출한다."""

    def __init__(
        self,
        store: FakeVectorStore,
        evaluator: FakeSufficiencyEvaluator,
        web_search: FakeWebSearch,
        validator: Any,
    ) -> None:
        self.store = store
        self.evaluator = evaluator
        self.web_search = web_search
        self.validator = validator


# 검증을 통과해야 하는 웹 근거(수의학 기관 allowlist 도메인).
TRUSTED_WEB = WebEvidence(
    title="Vomiting in dogs — Cornell Riney Canine Health Center",
    url="https://vet.cornell.edu/riney-canine-health-center/canine-health-information/vomiting",
    content=(
        "Vomiting in a dog can have many causes. A single episode in a dog that is otherwise "
        "bright and eating normally is often self-limiting. Contact your veterinarian if the "
        "vomiting repeats. 강아지 구토가 반복되면 수의사 상담이 필요합니다."
    ),
    score=0.71,
)

# 검증에서 반드시 떨어져야 하는 웹 근거(커뮤니티 블로그).
UNTRUSTED_WEB = WebEvidence(
    title="강아지 구토 이럴 땐 이렇게! 집에서 하는 관리법",
    url="https://vetblog.tistory.com/12",
    content=(
        "강아지가 구토를 하면 집에서 이렇게 해보세요. 구토가 반복되어도 지켜보면 됩니다. "
        "우리 아이도 이렇게 나았어요."
    ),
    score=0.66,
)


# ---------------------------------------------------------------------------
# 공용 fixture / 헬퍼
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def _no_api_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "TAVILY_API_KEY"):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture()
def adapter() -> FixtureClinicalDataAdapter:
    return FixtureClinicalDataAdapter()


def _make_deps(
    store: FakeVectorStore,
    evaluator: FakeSufficiencyEvaluator,
    web_search: FakeWebSearch,
    checkpointer: Any = None,
) -> SubgraphDeps:
    return SubgraphDeps(
        settings=get_settings(),
        llm=None,  # 키 없음 = 규칙 기반 경로(구현 가이드 0-1절)
        rag_service=FakeRagService(store, evaluator, web_search, WebSourceValidator()),
        checkpointer=checkpointer,
    )


def _health_state(
    adapter: FixtureClinicalDataAdapter,
    pet_id: int,
    message: str,
    **overrides: Any,
) -> dict:
    """임상 데이터를 실은 뒤 Health Subgraph 입력 State 를 만든다."""
    state = make_initial_state(
        pet_id=pet_id,
        user_message=message,
        pet_profile=adapter.load_pet_profile(pet_id),
        diagnoses=adapter.load_diagnoses(pet_id),
        daily_entries=adapter.load_daily_entries(pet_id),
    )
    state.update(overrides)
    return dict(state)


# ---------------------------------------------------------------------------
# 명세 43절 — 일반 건강 상담
# ---------------------------------------------------------------------------
def test_single_vomit_case_stays_normal_by_rules() -> None:
    """'오늘 한 번 토했는데 지금은 잘 놀아요' 는 normal 로 남는다(명세 43절).

    구토는 `escalation="trend"` 규칙이라 1회성 + 완화 표현이면 경과 관찰 범위다.
    이 판정이 무너지면 Health Subgraph 가 아니라 Visit Subgraph 로 잘못 간다.
    """
    result = evaluate_rules({"user_message": VOMIT_MESSAGE})

    assert result.risk_level == "normal"
    assert result.emergency_urgency == "none"
    assert result.rag_required is True  # 근거 검색은 여전히 필요하다


def test_sufficient_rag_calls_retriever_but_not_tavily(
    adapter: FixtureClinicalDataAdapter,
) -> None:
    """근거가 충분하면 RAG 만 호출하고 Tavily 는 **한 번도** 호출하지 않는다."""
    store = FakeVectorStore({"dog": DOG_DOCS, "cat": CAT_DOCS})
    evaluator = FakeSufficiencyEvaluator("sufficient")
    web_search = FakeWebSearch([TRUSTED_WEB])

    app = build_health_subgraph(_make_deps(store, evaluator, web_search))
    final = app.invoke(_health_state(adapter, DOG_PET_ID, VOMIT_MESSAGE))

    assert store.calls, "RAG 검색이 호출되어야 한다."
    assert evaluator.calls, "충분성 판정이 호출되어야 한다."
    assert web_search.calls == [], "충분한데 Tavily 를 호출하면 명세 47절 위반이다."

    assert final["rag_sufficiency"] == "sufficient"
    assert final["web_fallback_required"] is False
    assert final["validated_web_evidence"] == []
    assert final["merged_evidence"], "RAG 근거가 최종 근거로 넘어와야 한다."
    assert all(item["source_type"] == "rag" for item in final["merged_evidence"])


def test_health_response_cites_sources_and_avoids_forbidden_expressions(
    adapter: FixtureClinicalDataAdapter,
) -> None:
    """근거 기반 답변에는 출처가 붙고 금지 표현이 없다(명세 40절 2·4·5번)."""
    store = FakeVectorStore({"dog": DOG_DOCS})
    app = build_health_subgraph(
        _make_deps(store, FakeSufficiencyEvaluator("sufficient"), FakeWebSearch())
    )
    final = app.invoke(_health_state(adapter, DOG_PET_ID, VOMIT_MESSAGE))

    answer = final["draft_response"]
    assert "참고한 자료" in answer
    assert "Vomiting in dogs" in answer
    assert MEDICAL_DISCLAIMER in answer
    assert find_forbidden_expressions(answer) == []


def test_no_evidence_answer_says_it_does_not_know(
    adapter: FixtureClinicalDataAdapter,
) -> None:
    """근거가 하나도 없으면 추측하지 않고 '모른다 + 병원 상담' 으로 답한다."""
    store = FakeVectorStore({"dog": []})
    app = build_health_subgraph(
        _make_deps(store, FakeSufficiencyEvaluator("sufficient"), FakeWebSearch())
    )
    final = app.invoke(_health_state(adapter, DOG_PET_ID, VOMIT_MESSAGE))

    assert final["merged_evidence"] == []
    assert final["has_reliable_evidence"] is False
    assert NO_EVIDENCE_MESSAGE.splitlines()[0] in final["draft_response"]


# ---------------------------------------------------------------------------
# 명세 43절 — RAG fallback
# ---------------------------------------------------------------------------
def test_insufficient_rag_falls_back_to_tavily_and_keeps_only_validated_sources(
    adapter: FixtureClinicalDataAdapter,
) -> None:
    """부족하면 Tavily 를 부르고, **검증을 통과한 출처만** 최종 근거가 된다."""
    store = FakeVectorStore({"dog": DOG_DOCS})
    evaluator = FakeSufficiencyEvaluator("insufficient", missing_topics=("응급 판단 기준",))
    web_search = FakeWebSearch([TRUSTED_WEB, UNTRUSTED_WEB])

    app = build_health_subgraph(_make_deps(store, evaluator, web_search))
    final = app.invoke(_health_state(adapter, DOG_PET_ID, VOMIT_MESSAGE))

    assert final["rag_sufficiency"] == "insufficient"
    assert final["web_fallback_required"] is True
    assert len(web_search.calls) >= 1, "부족 판정이면 Tavily 를 호출해야 한다."

    # 거절된 항목도 사유와 함께 State 에 남는다(왜 안 썼는지 설명할 수 있어야 한다).
    by_url = {item["url"]: item for item in final["validated_web_evidence"]}
    assert by_url[TRUSTED_WEB.url]["accepted"] is True
    assert by_url[UNTRUSTED_WEB.url]["accepted"] is False
    assert by_url[UNTRUSTED_WEB.url]["reject_reason"]

    # 최종 근거에는 검증 통과분만 들어간다.
    evidence_urls = {item["source_url"] for item in final["merged_evidence"]}
    assert UNTRUSTED_WEB.url not in evidence_urls
    assert TRUSTED_WEB.url in evidence_urls


def test_all_web_results_rejected_is_a_normal_path(
    adapter: FixtureClinicalDataAdapter,
) -> None:
    """웹 결과가 전량 거절돼도 예외 없이 RAG 근거만으로 답한다(정상 fallback 실패)."""
    store = FakeVectorStore({"dog": DOG_DOCS})
    web_search = FakeWebSearch([UNTRUSTED_WEB])

    app = build_health_subgraph(
        _make_deps(store, FakeSufficiencyEvaluator("insufficient"), web_search)
    )
    final = app.invoke(_health_state(adapter, DOG_PET_ID, VOMIT_MESSAGE))

    assert len(web_search.calls) >= 1
    assert all(item["source_type"] == "rag" for item in final["merged_evidence"])
    assert final["draft_response"].strip()


def test_empty_tavily_result_does_not_raise(adapter: FixtureClinicalDataAdapter) -> None:
    """Tavily 키가 없어 빈 결과가 와도(가장 흔한 오프라인 상황) 그래프가 끝까지 돈다."""
    store = FakeVectorStore({"dog": DOG_DOCS})
    web_search = FakeWebSearch([])

    app = build_health_subgraph(
        _make_deps(store, FakeSufficiencyEvaluator("insufficient"), web_search)
    )
    final = app.invoke(_health_state(adapter, DOG_PET_ID, VOMIT_MESSAGE))

    assert final["validated_web_evidence"] == []
    assert final["draft_response"].strip()


# ---------------------------------------------------------------------------
# 명세 30절 J — 근거 충돌
# ---------------------------------------------------------------------------
def test_conflicting_status_uses_conflict_handler_without_tavily(
    adapter: FixtureClinicalDataAdapter,
) -> None:
    """충돌은 정보 부족이 아니므로 웹을 더 뒤지지 않고 충돌 사실을 기록한다."""
    store = FakeVectorStore({"dog": DOG_DOCS})
    web_search = FakeWebSearch([TRUSTED_WEB])

    app = build_health_subgraph(
        _make_deps(store, FakeSufficiencyEvaluator("conflicting"), web_search)
    )
    final = app.invoke(_health_state(adapter, DOG_PET_ID, VOMIT_MESSAGE))

    assert final["rag_sufficiency"] == "conflicting"
    assert web_search.calls == [], "충돌 경로에서는 Tavily 를 호출하지 않는다."
    assert final["evidence_conflicts"], "충돌 사실이 기록되어야 한다."
    assert any("단정하지 않" in note for note in final["evidence_conflicts"])


# ---------------------------------------------------------------------------
# 명세 43절 — 종 분리
# ---------------------------------------------------------------------------
def test_cat_fixture_searches_cat_index_only(adapter: FixtureClinicalDataAdapter) -> None:
    """고양이 fixture 는 cat index 만 조회하고 dog 문서는 최종 근거에 없다(명세 11·43절)."""
    contaminated = _evidence(
        "dog-leak", "dog", "Canine only document", "This document is about dogs only."
    )
    store = FakeVectorStore({"dog": DOG_DOCS, "cat": [*CAT_DOCS, contaminated]})

    app = build_health_subgraph(
        _make_deps(store, FakeSufficiencyEvaluator("sufficient"), FakeWebSearch())
    )
    final = app.invoke(_health_state(adapter, CAT_PET_ID, CAT_MESSAGE))

    assert store.searched_species == {"cat"}, "dog index 를 건드리면 안 된다."
    assert final["species"] == "cat"

    # metadata 가 오염돼 dog 문서가 섞여 들어와도 retriever 가 걷어낸다(2차 방어선).
    assert all(doc["species"] == "cat" for doc in final["rag_documents"])
    evidence_urls = {item["source_url"] for item in final["merged_evidence"]}
    assert contaminated.source_url not in evidence_urls
    assert all("/dog/" not in url for url in evidence_urls)


# ---------------------------------------------------------------------------
# 명세 29·30절 — Missing Information interrupt
# ---------------------------------------------------------------------------
def test_missing_onset_triggers_interrupt_and_resume_continues(
    adapter: FixtureClinicalDataAdapter,
) -> None:
    """증상 시작 시점이 없으면 interrupt 로 되묻고, resume 하면 그대로 이어진다."""
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.types import Command

    store = FakeVectorStore({"dog": DOG_DOCS})
    app = build_health_subgraph(
        _make_deps(
            store,
            FakeSufficiencyEvaluator("sufficient"),
            FakeWebSearch(),
            checkpointer=InMemorySaver(),
        )
    )
    config = {"configurable": {"thread_id": "health-interrupt"}}

    paused = app.invoke(_health_state(adapter, DOG_PET_ID, "토했어요"), config)

    interrupts = paused["__interrupt__"]
    payload = interrupts[0].value
    assert payload["type"] == "missing_information"
    assert "증상 시작 시점" in payload["missing_fields"]
    # 명세 29절: '모름' 은 유효한 답변이라는 계약을 호출자에게 알린다.
    assert payload["allow_unknown"] is True
    assert payload["unknown_value"] == "모름"
    assert store.calls == [], "정보가 모이기 전에 RAG 를 호출하면 안 된다."

    final = app.invoke(Command(resume={"symptom_onset": "오늘 아침"}), config)

    assert final["collected_information"]["symptom_onset"] == "오늘 아침"
    assert final["minimum_information_ready"] is True
    assert store.calls, "resume 후에는 RAG 가 호출되어야 한다."
    assert final["draft_response"].strip()


def test_unknown_answer_is_accepted_and_stops_asking(
    adapter: FixtureClinicalDataAdapter,
) -> None:
    """'모름' 은 미응답이 아니라 유효한 답변이라 되묻기가 멈춘다(명세 29절)."""
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.types import Command

    store = FakeVectorStore({"dog": DOG_DOCS})
    app = build_health_subgraph(
        _make_deps(
            store,
            FakeSufficiencyEvaluator("sufficient"),
            FakeWebSearch(),
            checkpointer=InMemorySaver(),
        )
    )
    config = {"configurable": {"thread_id": "health-unknown"}}

    app.invoke(_health_state(adapter, DOG_PET_ID, "토했어요"), config)
    final = app.invoke(Command(resume={"symptom_onset": "모름"}), config)

    assert final["missing_fields"] == []
    assert final["minimum_information_ready"] is True
    assert final["draft_response"].strip()
