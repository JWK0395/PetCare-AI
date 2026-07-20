"""RAG 충분성 판정 · 근거 병합 테스트 (명세 14·16·17절).

검증 대상은 두 모듈이 함께 지키는 **하나의 안전 계약**이다.

- `rag/sufficiency.py` — "이 근거로 답해도 되는가" 를 답변 생성 전에 판정한다.
  가장 중요한 불변식은 **LLM 이 sufficient 라고 우겨도 species 불일치와 빈 검색
  결과는 절대 sufficient 가 될 수 없다**는 것이다(강아지 질문에 고양이 문서로
  답하는 사고를 코드로 차단한다).
- `rag/evidence_merger.py` — Cornell RAG 근거를 웹 근거보다 항상 앞에 두고,
  충돌은 감추지 않고 기록하며, 신뢰 가능한 근거가 하나도 없으면 그 사실을
  `has_reliable_evidence=False` 로 분명히 알린다.

원칙:
  * 임베딩 모델 로드 없음 — 전부 손으로 만든 더미 `RetrievedEvidence`/`WebEvidence`.
  * 네트워크 호출 없음 — LLM 은 stub 주입.
  * 전역 설정을 건드리지 않는다 — 매 테스트가 새 `RagSettings` 를 만들어 주입한다.
    (`RagSettings` 는 mutable dataclass 라 전역을 고치면 테스트 간에 샌다.)
"""

from __future__ import annotations

from typing import Any

import pytest

from petcare_ai.config import RagSettings
from petcare_ai.rag.evidence_merger import merge_evidence
from petcare_ai.rag.sufficiency import (
    UNCALIBRATED_WARNING,
    KnowledgeSufficiencyEvaluator,
    apply_calibration,
    calibrate_threshold,
)
from petcare_ai.schemas import (
    KnowledgeSufficiencyResult,
    RagQuery,
    RetrievedEvidence,
    WebEvidence,
)

# ---------------------------------------------------------------------------
# 더미 본문 상수
# ---------------------------------------------------------------------------
#: 구토 원인 문서. required_topic "vomiting causes" 를 커버한다.
#: 충돌 신호어(즉시 진료 / 가정 관찰)를 의도적으로 하나도 넣지 않았다 —
#: sufficient 판정 테스트가 충돌 탐지에 걸려 넘어지면 안 되기 때문이다.
DOG_VOMIT_CAUSES = (
    "Vomiting in dogs has many causes, including eating something unusual, "
    "infection, or intestinal obstruction. Owners should record how often the "
    "vomiting happens and whether the dog still drinks water."
)
#: 경고 징후 문서. required_topic "warning signs" 를 커버한다.
DOG_VOMIT_WARNING = (
    "Warning signs that need attention include repeated vomiting over several "
    "hours, blood in the vomit, and marked lethargy. These warning signs mean "
    "the dog should be examined by a veterinarian."
)
#: 충돌 쌍 A — "즉시 진료" 신호어만 들어 있다.
CONFLICT_URGENT = (
    "Repeated vomiting with blood requires urgent evaluation. Seek immediate "
    "veterinary care for any dog that cannot keep water down."
)
#: 충돌 쌍 B — "가정 관찰" 신호어만 들어 있다. 위 문서와 서로 다른 document_id
#: 여야 충돌로 인정된다(한 문서 안의 조건부 서술은 충돌이 아니다).
CONFLICT_WATCHFUL = (
    "Mild vomiting in an otherwise bright dog usually resolves on its own, "
    "and owners can monitor at home for twenty four hours."
)


# ---------------------------------------------------------------------------
# 팩토리
# ---------------------------------------------------------------------------
def make_evidence(
    chunk_id: str,
    text: str,
    *,
    title: str = "Cornell Article",
    species: str = "dog",
    score: float | None = 0.90,
    document_id: str | None = None,
    categories: list[str] | None = None,
    heading_path: list[str] | None = None,
) -> RetrievedEvidence:
    """검색 결과 1건을 만든다 — vector store 없이 판정 로직만 시험하기 위함."""
    return RetrievedEvidence(
        chunk_id=chunk_id,
        document_id=document_id or chunk_id.rsplit("#", 1)[0],
        title=title,
        text=text,
        species=species,  # type: ignore[arg-type]
        source="Cornell Riney Canine Health Center",
        source_url=f"https://vet.cornell.edu/{chunk_id}",
        categories=categories or [],
        score=score,
        heading_path=heading_path or [],
    )


def make_web(
    url: str,
    content: str,
    *,
    title: str = "Veterinary reference",
    accepted: bool = True,
    score: float | None = 0.80,
    domain: str = "vet.cornell.edu",
) -> WebEvidence:
    """검증(validator)을 통과한 웹 근거 1건. accepted=False 면 병합에서 버려진다."""
    return WebEvidence(
        title=title,
        url=url,
        content=content,
        score=score,
        domain=domain,
        accepted=accepted,
    )


def make_query(
    ko: str = "강아지가 이틀째 구토하고 기운이 없어요",
    en: str = "dog vomiting lethargy causes and treatment options",
    *,
    topics: list[str] | None = None,
    species: str = "dog",
) -> RagQuery:
    """RagQuery 를 만든다. 기본값은 최신성 신호어가 없는 '평범한' 질문이다."""
    return RagQuery(
        primary_query_ko=ko,
        primary_query_en=en,
        required_topics=topics if topics is not None else ["vomiting causes", "warning signs"],
        species=species,  # type: ignore[arg-type]
    )


def rag_settings(**overrides: Any) -> RagSettings:
    """테스트 전용 RagSettings — 전역 설정을 오염시키지 않는다."""
    return RagSettings(**overrides)


class StubLLM:
    """structured output LLM 흉내 — 호출 여부까지 기록한다.

    `safe_structured_invoke()` 는 `with_structured_output` 이 있으면 적용하고
    `invoke()` 를 부른다. 호출 횟수를 세는 이유: hard block 상황에서는 LLM 을
    **아예 부르지 않아야** 하고(비용·지연 절감), 그 사실 자체가 계약이다.
    """

    def __init__(self, result: KnowledgeSufficiencyResult) -> None:
        self.result = result
        self.calls: list[Any] = []
        self.schema: Any = None

    def with_structured_output(self, schema: Any) -> "StubLLM":
        self.schema = schema
        return self

    def invoke(self, messages: Any) -> KnowledgeSufficiencyResult:
        self.calls.append(messages)
        return self.result


def sufficient_docs() -> list[RetrievedEvidence]:
    """기본 sufficient 시나리오용 문서 2건(기준 min_documents_for_sufficient=2)."""
    return [
        make_evidence("dog-vomit-1#0", DOG_VOMIT_CAUSES, title="Vomiting in Dogs", score=0.92),
        make_evidence("dog-vomit-2#0", DOG_VOMIT_WARNING, title="Warning Signs", score=0.88),
    ]


# ---------------------------------------------------------------------------
# 1. 충분한 결과
# ---------------------------------------------------------------------------
def test_sufficient_when_topics_and_documents_cover_the_question() -> None:
    """근거가 충분하면 sufficient 로 판정되고 웹 fallback 이 불필요해야 한다.

    문서 수(2건 이상)·topic 커버율·핵심 증상어가 모두 충족되고 충돌이 없으면
    내부 코퍼스만으로 답할 수 있다. 이때 서비스 계층은 Tavily 를 호출하지
    않는다(명세 47절 "모든 질문에 Tavily 호출 금지"). 그 판단의 입력이 되는
    두 값(status / requires_recent_information)을 직접 확인한다.
    """
    evaluator = KnowledgeSufficiencyEvaluator(settings=rag_settings())
    result = evaluator.evaluate(make_query(), sufficient_docs())

    assert result.status == "sufficient"
    assert result.requires_recent_information is False
    assert result.missing_topics == []
    # 서비스 계층의 fallback 규칙과 동일한 식 — 둘 다 거짓이어야 웹 검색을 건너뛴다.
    assert not (result.status != "sufficient" or result.requires_recent_information)


def test_clean_sufficient_does_not_call_the_llm() -> None:
    """흠 없는 sufficient 는 LLM 을 아예 부르지 않는다.

    LLM 은 품질 향상 옵션이지 필수 경로가 아니다. deterministic 단계가 이미
    확신할 때까지 비용과 지연을 쓰면 안 된다.
    """
    llm = StubLLM(KnowledgeSufficiencyResult(status="insufficient", reason="부르면 안 됨"))
    evaluator = KnowledgeSufficiencyEvaluator(settings=rag_settings(), llm=llm)

    result = evaluator.evaluate(make_query(), sufficient_docs())

    assert result.status == "sufficient"
    assert llm.calls == []


# ---------------------------------------------------------------------------
# 2. 부족한 결과
# ---------------------------------------------------------------------------
def test_empty_documents_are_insufficient() -> None:
    """검색 결과가 비어 있으면 근거가 아예 없으므로 insufficient 다.

    이 경우 답변 생성기는 추측하지 말고 "확실하지 않다 + 병원 상담" 으로
    가야 하므로, 판정 결과에 사유가 남아 있어야 한다.
    """
    evaluator = KnowledgeSufficiencyEvaluator(settings=rag_settings())
    result = evaluator.evaluate(make_query(), [])

    assert result.status == "insufficient"
    assert "검색 결과가 비어 있습니다" in result.reason
    # required_topics 는 전부 미충족으로 남아야 후속 단계가 무엇이 없는지 안다.
    assert result.missing_topics == ["vomiting causes", "warning signs"]
    assert result.covered_topics == []


def test_single_document_is_insufficient_by_document_count() -> None:
    """관련 문서가 기준(min_documents_for_sufficient=2)보다 적으면 insufficient.

    한 건짜리 근거로 건강 판단을 내리면 그 문서의 편향이 그대로 답변이 된다.
    """
    evaluator = KnowledgeSufficiencyEvaluator(settings=rag_settings())
    result = evaluator.evaluate(
        make_query(topics=["vomiting causes"]),
        [make_evidence("dog-vomit-1#0", DOG_VOMIT_CAUSES, score=0.95)],
    )

    assert result.status == "insufficient"
    assert "관련 문서 수 부족" in result.reason


def test_score_margin_shortfall_makes_documents_irrelevant() -> None:
    """코퍼스 평균 대비 margin 을 못 넘긴 문서는 '관련 없음' 으로 떨어진다.

    임베딩 코사인 값은 좁은 밴드에 몰려서 절대 임계값이 무의미하다. 그래서
    코퍼스 평균이 보정돼 있으면 상대 margin 을 기준으로 쓴다. 여기서는
    평균 0.77 + margin 0.05 = 0.82 인데 문서 score 가 0.78/0.79 라 전부 탈락한다.
    """
    settings = rag_settings(corpus_score_mean=0.77, min_relevance_margin=0.05)
    evaluator = KnowledgeSufficiencyEvaluator(settings=settings)

    result = evaluator.evaluate(
        make_query(),
        [
            make_evidence("dog-vomit-1#0", DOG_VOMIT_CAUSES, score=0.78),
            make_evidence("dog-vomit-2#0", DOG_VOMIT_WARNING, score=0.79),
        ],
    )

    assert result.status == "insufficient"
    assert "관련 문서 수 부족" in result.reason
    # 같은 문서라도 margin 을 넘기면 sufficient 가 되어야 한다(기준 자체의 문제가 아님을 확인).
    passing = evaluator.evaluate(
        make_query(),
        [
            make_evidence("dog-vomit-1#0", DOG_VOMIT_CAUSES, score=0.90),
            make_evidence("dog-vomit-2#0", DOG_VOMIT_WARNING, score=0.91),
        ],
    )
    assert passing.status == "sufficient"


def test_documents_with_empty_body_are_insufficient() -> None:
    """본문이 빈 문서는 개수만 채워도 근거가 될 수 없다.

    score 와 문서 수는 통과하더라도 topic 커버율과 핵심 증상어가 0 이므로
    insufficient 여야 한다. "문서가 2건 있으니 충분" 이라는 개수 물신주의를 막는다.
    """
    evaluator = KnowledgeSufficiencyEvaluator(settings=rag_settings())
    result = evaluator.evaluate(
        make_query(),
        [
            make_evidence("empty-1#0", "", title="", score=0.95),
            make_evidence("empty-2#0", "", title="", score=0.94),
        ],
    )

    assert result.status == "insufficient"
    assert result.covered_topics == []
    assert "required_topics 커버율 부족" in result.reason
    assert "질문 핵심 증상이 문서에 없음" in result.reason


# ---------------------------------------------------------------------------
# 3. species 불일치
# ---------------------------------------------------------------------------
def test_species_mismatch_is_insufficient() -> None:
    """고양이 질문에 강아지 문서가 오면 내용과 무관하게 insufficient 다.

    species 별 index 를 분리했는데도 다른 종 문서가 섞였다는 것은 잘못된
    index 를 탄 것이다. 종이 다르면 용량·독성·정상치가 전부 달라서 그대로
    답하면 위험하다.
    """
    evaluator = KnowledgeSufficiencyEvaluator(settings=rag_settings())
    result = evaluator.evaluate(make_query(species="cat"), sufficient_docs())

    assert result.status == "insufficient"
    assert "species 불일치" in result.reason
    assert "질의=cat" in result.reason and "문서=dog" in result.reason


# ---------------------------------------------------------------------------
# 4. 충돌하는 근거
# ---------------------------------------------------------------------------
def test_conflicting_recommendations_across_documents() -> None:
    """서로 다른 문서가 상반된 권고를 하면 conflicting 이다.

    "즉시 병원" 과 "집에서 관찰" 이 다른 문서에서 각각 나오면 그 차이를 감추고
    한쪽만 인용하면 안 된다. 사용자에게 불확실성을 드러내야 한다.
    """
    evaluator = KnowledgeSufficiencyEvaluator(settings=rag_settings())
    result = evaluator.evaluate(
        make_query(topics=["vomiting causes"]),
        [
            make_evidence("conf-a#0", CONFLICT_URGENT, document_id="conf-a", score=0.91),
            make_evidence("conf-b#0", CONFLICT_WATCHFUL, document_id="conf-b", score=0.90),
        ],
    )

    assert result.status == "conflicting"
    assert "즉시 진료 권고 vs 가정 관찰 권고" in result.reason


def test_single_document_with_both_signals_is_not_a_conflict() -> None:
    """한 문서 안에 두 신호가 같이 있는 건 조건부 서술이지 충돌이 아니다.

    "보통은 가정 관찰, 그러나 다음 증상이면 즉시 내원" 같은 정상적인 임상 서술을
    충돌로 오탐하면 거의 모든 답변이 conflicting 이 되어 판정이 무의미해진다.
    """
    conditional = CONFLICT_WATCHFUL + " " + CONFLICT_URGENT
    evaluator = KnowledgeSufficiencyEvaluator(settings=rag_settings())
    result = evaluator.evaluate(
        make_query(topics=["vomiting causes"]),
        [
            make_evidence("cond-a#0", conditional, document_id="cond-a", score=0.91),
            make_evidence("cond-a#1", conditional, document_id="cond-a", score=0.90),
        ],
    )

    assert result.status != "conflicting"


# ---------------------------------------------------------------------------
# 5. **핵심** — LLM 이 hard guard 를 덮어쓸 수 없다
# ---------------------------------------------------------------------------
def test_llm_cannot_override_species_mismatch() -> None:
    """LLM 이 sufficient 를 반환해도 species 불일치는 insufficient 로 남는다.

    이 프로젝트에서 가장 중요한 가드다. LLM 은 문서를 읽고 "내용이 충분하다" 고
    말할 수 있지만, 애초에 다른 종의 문서라는 사실은 판단 대상이 아니다.
    게다가 hard block 이면 LLM 호출 자체를 생략해야 한다(비용·지연 절감).
    """
    llm = StubLLM(
        KnowledgeSufficiencyResult(
            status="sufficient",
            covered_topics=["vomiting causes", "warning signs"],
            reason="문서가 충분합니다",
        )
    )
    evaluator = KnowledgeSufficiencyEvaluator(settings=rag_settings(), llm=llm)

    result = evaluator.evaluate(make_query(species="cat"), sufficient_docs())

    assert result.status == "insufficient"
    assert "species 불일치" in result.reason
    assert llm.calls == [], "hard block 상황에서는 LLM 을 호출하지 않아야 한다"


def test_llm_cannot_override_empty_results() -> None:
    """LLM 이 sufficient 를 반환해도 빈 검색 결과는 insufficient 로 남는다.

    근거가 0건인데 충분하다고 말하는 것은 곧 환각이다. 이것 역시 LLM 호출 없이
    차단되어야 한다.
    """
    llm = StubLLM(KnowledgeSufficiencyResult(status="sufficient", reason="충분합니다"))
    evaluator = KnowledgeSufficiencyEvaluator(settings=rag_settings(), llm=llm)

    result = evaluator.evaluate(make_query(), [])

    assert result.status == "insufficient"
    assert "검색 결과가 비어 있습니다" in result.reason
    assert llm.calls == []


@pytest.mark.parametrize(
    ("species", "documents_factory", "expected_fragment"),
    [
        ("cat", sufficient_docs, "species 불일치"),
        ("dog", list, "검색 결과가 비어 있습니다"),
    ],
    ids=["species_mismatch", "empty_documents"],
)
def test_enforce_hard_guards_rejects_any_fabricated_sufficient_result(
    species: str,
    documents_factory: Any,
    expected_fragment: str,
) -> None:
    """임의로 만든 sufficient 결과를 넣어도 hard guard 를 지나면 insufficient 가 된다.

    `evaluate()` 내부 흐름과 무관하게 가드 자체가 성립하는지 직접 검증한다.
    LLM 응답이든 사람이 손으로 만든 결과든, 이 함수를 통과한 값은 절대
    sufficient/conflicting 일 수 없다.
    """
    evaluator = KnowledgeSufficiencyEvaluator(settings=rag_settings())
    fabricated = KnowledgeSufficiencyResult(
        status="sufficient",
        covered_topics=["vomiting causes"],
        reason="LLM 이 충분하다고 판단했습니다",
    )

    guarded = evaluator.enforce_hard_guards(
        fabricated, make_query(species=species), documents_factory()
    )

    assert guarded.status == "insufficient"
    assert "[강제]" in guarded.reason
    assert expected_fragment in guarded.reason
    # 원본 LLM 사유를 지우지 않는다 — 왜 뒤집혔는지 추적할 수 있어야 한다.
    assert "LLM 이 충분하다고 판단했습니다" in guarded.reason


def test_enforce_hard_guards_is_a_no_op_when_nothing_is_blocked() -> None:
    """막을 사유가 없으면 가드는 결과를 그대로 통과시킨다.

    가드가 정상 판정까지 깎아내리면 sufficient 가 영원히 나오지 않는다.
    """
    evaluator = KnowledgeSufficiencyEvaluator(settings=rag_settings())
    original = KnowledgeSufficiencyResult(status="sufficient", reason="정상")

    guarded = evaluator.enforce_hard_guards(original, make_query(), sufficient_docs())

    assert guarded is original


def test_llm_cannot_override_detected_conflict() -> None:
    """문서 간 충돌이 신호어로 확인되면 LLM 이 sufficient 라 해도 conflicting 유지.

    충돌은 추론이 아니라 관측된 사실이다. LLM 이 "종합하면 괜찮다" 고 뭉개는
    방향(=위험을 낮추는 방향)의 병합은 허용하지 않는다.
    """
    llm = StubLLM(KnowledgeSufficiencyResult(status="sufficient", reason="종합하면 문제없습니다"))
    evaluator = KnowledgeSufficiencyEvaluator(settings=rag_settings(), llm=llm)

    result = evaluator.evaluate(
        make_query(topics=["vomiting causes"]),
        [
            make_evidence("conf-a#0", CONFLICT_URGENT, document_id="conf-a", score=0.91),
            make_evidence("conf-b#0", CONFLICT_WATCHFUL, document_id="conf-b", score=0.90),
        ],
    )

    assert result.status == "conflicting"
    assert len(llm.calls) == 1, "hard block 이 아니므로 LLM 2차 판정은 수행된다"


def test_llm_may_upgrade_soft_insufficiency() -> None:
    """반대로 '문서 수 부족' 같은 soft 조건은 LLM 이 뒤집을 수 있다.

    가드의 경계를 명확히 하기 위한 대조군이다. 모든 것을 막아버리면 LLM 2차
    판정 자체가 죽은 코드가 되므로, hard block 이 아닌 사유는 승격 가능해야 한다.
    """
    llm = StubLLM(
        KnowledgeSufficiencyResult(
            status="sufficient",
            covered_topics=["vomiting causes"],
            reason="한 건이지만 질문에 정확히 답합니다",
        )
    )
    evaluator = KnowledgeSufficiencyEvaluator(settings=rag_settings(), llm=llm)

    result = evaluator.evaluate(
        make_query(topics=["vomiting causes"]),
        [make_evidence("dog-vomit-1#0", DOG_VOMIT_CAUSES, score=0.95)],
    )

    assert result.status == "sufficient"
    assert len(llm.calls) == 1
    assert "한 건이지만 질문에 정확히 답합니다" in result.reason


def test_llm_failure_falls_back_to_deterministic_verdict() -> None:
    """LLM 이 예외를 던져도 판정은 deterministic 결과로 계속된다.

    LLM 은 없어도 되는 부품이다. 타임아웃·rate limit 이 사용자 답변 전체를
    실패시키면 안 된다(`safe_structured_invoke` 규약).
    """

    class BoomLLM:
        def invoke(self, messages: Any) -> Any:
            raise RuntimeError("rate limit")

    evaluator = KnowledgeSufficiencyEvaluator(settings=rag_settings(), llm=BoomLLM())
    result = evaluator.evaluate(
        make_query(topics=["vomiting causes"]),
        [make_evidence("dog-vomit-1#0", DOG_VOMIT_CAUSES, score=0.95)],
    )

    assert result.status == "insufficient"
    assert "관련 문서 수 부족" in result.reason


# ---------------------------------------------------------------------------
# 6. 최신성 신호
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("ko", "en"),
    [
        ("요즘 사료 리콜 소식 있나요?", "is there a dog food recall advisory"),
        ("올해 개 인플루엔자 유행 상황이 궁금해요", "canine influenza outbreak status"),
        ("최신 예방접종 권고안이 바뀌었나요?", "updated guideline for vaccination"),
    ],
    ids=["recall", "outbreak", "updated_guideline"],
)
def test_recency_signals_require_recent_information(ko: str, en: str) -> None:
    """리콜·유행·권고안 변경 같은 질문은 최신 정보가 필요하다고 표시한다.

    내부 Cornell 코퍼스는 특정 시점 스냅샷이라 이런 질문에는 원리적으로 답할 수
    없다. 이 판단은 LLM 없이도 동작해야 하므로 신호어 규칙으로 처리한다.
    """
    evaluator = KnowledgeSufficiencyEvaluator(settings=rag_settings())
    result = evaluator.evaluate(make_query(ko=ko, en=en), sufficient_docs())

    assert result.requires_recent_information is True
    # 서비스 계층은 이 값만으로도 웹 fallback 을 켠다.
    assert result.status != "sufficient" or result.requires_recent_information


def test_recency_signal_detected_in_required_topics() -> None:
    """질문 문장이 아니라 required_topics 에만 신호어가 있어도 잡아야 한다.

    query builder 가 최신성 요구를 topic 으로 옮겨 적는 경우가 있어서, 탐지
    대상에 required_topics 도 포함되어야 누락이 없다.
    """
    evaluator = KnowledgeSufficiencyEvaluator(settings=rag_settings())
    result = evaluator.evaluate(
        make_query(topics=["canine influenza outbreak"]), sufficient_docs()
    )

    assert result.requires_recent_information is True


def test_ordinary_question_does_not_require_recent_information() -> None:
    """평범한 증상 질문은 최신성 요구가 없어야 한다(오탐 방지 대조군)."""
    evaluator = KnowledgeSufficiencyEvaluator(settings=rag_settings())
    result = evaluator.evaluate(make_query(), sufficient_docs())

    assert result.requires_recent_information is False


# ---------------------------------------------------------------------------
# 7. covered_topics / missing_topics
# ---------------------------------------------------------------------------
def test_covered_and_missing_topics_are_split_by_required_topics() -> None:
    """covered/missing 은 반드시 required_topics 안에서만 나뉘어야 한다.

    후속 단계(웹 fallback query 생성, 사용자 안내)가 "무엇이 빠졌는지" 를
    이 목록으로 읽는다. 임의의 문자열이 섞이면 fallback query 가 오염된다.
    """
    topics = ["vomiting causes", "kidney disease diet"]
    evaluator = KnowledgeSufficiencyEvaluator(settings=rag_settings())
    result = evaluator.evaluate(
        make_query(topics=topics),
        [
            make_evidence("dog-vomit-1#0", DOG_VOMIT_CAUSES, score=0.92),
            make_evidence("dog-vomit-2#0", DOG_VOMIT_WARNING, score=0.88),
        ],
    )

    assert result.covered_topics == ["vomiting causes"]
    assert result.missing_topics == ["kidney disease diet"]
    # 두 목록의 합집합이 곧 required_topics 여야 한다(빠뜨림·중복 없음).
    assert sorted(result.covered_topics + result.missing_topics) == sorted(topics)


def test_topic_matched_through_title_and_heading_path() -> None:
    """topic 이 본문에 없고 제목/소제목/카테고리에만 있어도 커버로 인정한다.

    Cornell 문서는 주제어가 소제목에만 등장하는 경우가 흔해서, 본문만 보면
    실제로는 다루는 주제를 missing 으로 잘못 보고하게 된다.
    """
    evaluator = KnowledgeSufficiencyEvaluator(settings=rag_settings())
    result = evaluator.evaluate(
        make_query(topics=["kidney disease diet"]),
        [
            make_evidence(
                "dog-kidney-1#0",
                DOG_VOMIT_CAUSES,
                title="Kidney disease diet for dogs",
                score=0.92,
            ),
            make_evidence(
                "dog-kidney-2#0",
                DOG_VOMIT_WARNING,
                heading_path=["Nutrition", "Kidney disease diet"],
                score=0.90,
            ),
        ],
    )

    assert result.covered_topics == ["kidney disease diet"]
    assert result.missing_topics == []


def test_no_required_topics_does_not_fail_coverage() -> None:
    """required_topics 가 비어 있으면 커버율 조건으로 탈락시키지 않는다.

    일반 대화성 질문은 topic 이 없을 수 있는데, 이때 커버율 0/0 을 실패로
    처리하면 어떤 근거도 절대 충분해지지 않는다.
    """
    evaluator = KnowledgeSufficiencyEvaluator(settings=rag_settings())
    result = evaluator.evaluate(make_query(topics=[]), sufficient_docs())

    assert result.status == "sufficient"
    assert result.covered_topics == []
    assert result.missing_topics == []


# ---------------------------------------------------------------------------
# 8~9. score 임계값 calibration
# ---------------------------------------------------------------------------
def test_calibrate_threshold_reports_score_distribution() -> None:
    """calibrate_threshold() 는 실제 score 분포 통계를 돌려준다.

    임베딩 모델을 바꾸면 코사인 값 밴드가 통째로 이동하므로 임계값을 코드에
    박아두면 안 된다(명세 14절). 대신 이 통계를 뽑아 config 에 반영한다.
    """
    stats = calibrate_threshold(
        {
            "vomiting": [
                make_evidence("a#0", DOG_VOMIT_CAUSES, score=0.70),
                make_evidence("a#1", DOG_VOMIT_CAUSES, score=0.75),
                make_evidence("a#2", DOG_VOMIT_CAUSES, score=0.80),
            ],
            "kidney": [
                make_evidence("b#0", DOG_VOMIT_WARNING, score=0.60),
                make_evidence("b#1", DOG_VOMIT_WARNING, score=0.90),
            ],
        }
    )

    assert stats["query_count"] == 2
    assert stats["count"] == 5
    assert stats["min"] == pytest.approx(0.60)
    assert stats["max"] == pytest.approx(0.90)
    assert stats["mean"] == pytest.approx(0.75)
    # 분위수는 선형 보간으로 계산한다(numpy 없이).
    assert stats["p25"] == pytest.approx(0.70)
    assert stats["p50"] == pytest.approx(0.75)
    assert stats["p75"] == pytest.approx(0.80)
    assert stats["median"] == pytest.approx(stats["p50"])
    assert stats["stdev"] is not None and stats["stdev"] > 0
    # query 별 1등 score 평균 = 관련 문서의 대표 수준.
    assert stats["top1_mean"] == pytest.approx(0.85)
    assert stats["suggested_min_relevance_score"] == pytest.approx(0.70)
    assert stats["suggested_min_relevance_margin"] == pytest.approx(0.10)
    assert set(stats["per_query"]) == {"vomiting", "kidney"}


def test_calibrate_threshold_counts_documents_without_score() -> None:
    """score 없는 문서는 통계에서 빼되 개수는 보고한다.

    조용히 버리면 "왜 표본이 적지?" 를 알 수 없다. 보정 신뢰도를 사람이
    판단할 수 있어야 한다.
    """
    stats = calibrate_threshold(
        [[make_evidence("a#0", DOG_VOMIT_CAUSES, score=None), make_evidence("a#1", DOG_VOMIT_CAUSES, score=0.8)]]
    )

    assert stats["count"] == 1
    assert stats["missing_score_count"] == 1


def test_calibrate_threshold_with_no_scores_suggests_nothing() -> None:
    """표본이 없으면 제안값을 만들지 않는다 — 근거 없는 임계값이 더 위험하다."""
    stats = calibrate_threshold({})

    assert stats["count"] == 0
    assert stats["min"] is None
    assert stats["suggested_min_relevance_score"] is None
    assert stats["suggested_min_relevance_margin"] is None


def test_apply_calibration_marks_settings_as_calibrated() -> None:
    """calibration 결과를 RagSettings 에 반영하면 미보정 경고가 사라진다.

    노트북에서 값을 손으로 옮겨 적다가 틀리는 일을 막는 경로다.
    """
    settings = rag_settings()
    stats = calibrate_threshold(
        {
            "q": [
                make_evidence("a#0", DOG_VOMIT_CAUSES, score=0.70),
                make_evidence("a#1", DOG_VOMIT_CAUSES, score=0.80),
            ]
        }
    )

    updated = apply_calibration(stats, settings)

    assert updated is settings
    assert settings.score_threshold_calibrated is True
    assert settings.corpus_score_mean == pytest.approx(0.75)


def test_apply_calibration_refuses_to_mark_empty_stats() -> None:
    """통계가 비었으면 보정됨으로 표시하지 않는다.

    보정되지 않은 상태를 보정됨으로 속이면 경고 문구가 사라져 훨씬 위험하다.
    """
    settings = rag_settings()

    apply_calibration(calibrate_threshold({}), settings)

    assert settings.score_threshold_calibrated is False


def test_uncalibrated_threshold_warning_is_present_in_reason() -> None:
    """score_threshold_calibrated=False 면 판정 사유에 미보정 경고가 붙는다.

    임계값을 보정하지 않은 채 score 기준 판정을 신뢰하면 안 된다는 사실을
    trace 를 읽는 사람이 알 수 있어야 한다.
    """
    evaluator = KnowledgeSufficiencyEvaluator(settings=rag_settings(score_threshold_calibrated=False))
    result = evaluator.evaluate(make_query(), sufficient_docs())

    assert "임계값 미보정" in result.reason
    assert UNCALIBRATED_WARNING in result.reason


def test_calibrated_settings_drop_the_warning() -> None:
    """보정이 끝나면 경고가 사라져야 한다 — 항상 붙어 있으면 경고가 무뎌진다."""
    settings = rag_settings(
        score_threshold_calibrated=True, corpus_score_mean=0.77, min_relevance_margin=0.05
    )
    evaluator = KnowledgeSufficiencyEvaluator(settings=settings)
    result = evaluator.evaluate(
        make_query(),
        [
            make_evidence("dog-vomit-1#0", DOG_VOMIT_CAUSES, score=0.92),
            make_evidence("dog-vomit-2#0", DOG_VOMIT_WARNING, score=0.90),
        ],
    )

    assert "임계값 미보정" not in result.reason


# ---------------------------------------------------------------------------
# 10. evidence_merger — 우선순위 / 강화 / 보조 / 충돌
# ---------------------------------------------------------------------------
#: 초콜릿 독성 — RAG 와 웹이 같은 방향으로 말하는 쌍.
RAG_CHOCOLATE = (
    "Chocolate toxicity in dogs is caused by theobromine, which is toxic to "
    "dogs even in modest amounts and can cause tremors."
)
WEB_CHOCOLATE = (
    "Chocolate toxicity should be treated as a veterinary emergency because "
    "theobromine is toxic to dogs."
)
#: 웹에만 있는 최신 정보 — RAG 스냅샷에는 없다.
WEB_RECALL = (
    "A pet food product recall was announced after elevated vitamin D levels "
    "were found in several lots."
)
#: 백합 독성 — RAG 와 웹이 정반대로 말하는 쌍.
RAG_LILY = (
    "Lily toxicity in cats is severe; lilies are toxic and can cause acute "
    "kidney failure after even small exposures."
)
WEB_LILY = (
    "Lily toxicity is often overstated online — this variety is generally safe "
    "for cats and harmless in small amounts."
)


def test_rag_evidence_always_precedes_web_evidence() -> None:
    """반환 순서가 곧 우선순위다 — Cornell RAG 근거가 항상 웹보다 앞에 온다.

    답변 생성기는 앞에서부터 인용하고, `final_evidence_max` 로 잘릴 때도 웹
    근거가 먼저 떨어져야 한다. 웹 score(0.99)가 RAG 보다 높아도 순서는 불변이다.
    """
    result = merge_evidence(
        [
            make_evidence("rag-low#0", DOG_VOMIT_CAUSES, score=0.50),
            make_evidence("rag-high#0", DOG_VOMIT_WARNING, score=0.90),
        ],
        [make_web("https://vet.cornell.edu/web-a", WEB_CHOCOLATE, score=0.99)],
        required_topics=["vomiting causes"],
    )

    assert [item.source_type for item in result.evidence] == ["rag", "rag", "web"]
    # RAG 내부에서는 score 내림차순.
    assert [item.evidence_id for item in result.evidence[:2]] == ["rag-high#0", "rag-low#0"]
    assert result.has_reliable_evidence is True


def test_same_claim_from_rag_and_web_is_kept_as_reinforcement() -> None:
    """같은 주장이면 한쪽을 버리지 않고 둘 다 남겨 근거를 강화한다.

    웹이 RAG 를 반복한다고 지우면 "여러 출처가 같은 말을 한다" 는 신뢰 신호가
    사라진다. 방향이 같으므로 conflicts 는 비어 있어야 한다.
    """
    result = merge_evidence(
        [make_evidence("rag-choco#0", RAG_CHOCOLATE, title="Chocolate Toxicity in Dogs")],
        [make_web("https://vet.cornell.edu/chocolate", WEB_CHOCOLATE, title="Chocolate toxicity")],
        required_topics=["chocolate toxicity"],
    )

    assert len(result.evidence) == 2
    assert [item.source_type for item in result.evidence] == ["rag", "web"]
    assert all("chocolate toxicity" in item.supported_topics for item in result.evidence)
    assert result.conflicts == []
    assert result.has_reliable_evidence is True


def test_web_only_topic_is_included_as_supporting_evidence() -> None:
    """RAG 에 없고 웹에만 있는 내용은 충돌이 아니라 보조 근거다.

    내부 코퍼스는 스냅샷이라 리콜 같은 최신 정보가 없다. 이걸 충돌로 처리하면
    가장 필요한 정보를 버리게 된다.
    """
    result = merge_evidence(
        [make_evidence("rag-choco#0", RAG_CHOCOLATE, title="Chocolate Toxicity in Dogs")],
        [make_web("https://vet.cornell.edu/recall", WEB_RECALL, title="Pet food product recall")],
        required_topics=["chocolate toxicity", "product recall"],
    )

    web_items = [item for item in result.evidence if item.source_type == "web"]
    assert len(web_items) == 1
    assert "product recall" in web_items[0].supported_topics
    # RAG 는 이 주제를 다루지 않으므로 비교 대상이 없다 = 충돌 아님.
    assert result.conflicts == []
    assert result.has_reliable_evidence is True


def test_conflicting_stances_are_recorded_not_hidden() -> None:
    """RAG 와 웹의 입장이 정반대면 conflicts 에 기록하고 보수적으로 안내한다.

    "독성 있음" vs "안전함" 은 가장 위험한 충돌이다. 감추고 한쪽만 인용하면
    사용자가 위험한 결정을 내릴 수 있다.
    """
    result = merge_evidence(
        [make_evidence("rag-lily#0", RAG_LILY, title="Lily toxicity", species="cat")],
        [make_web("https://vet.cornell.edu/lily", WEB_LILY, title="Lily toxicity myths")],
        required_topics=["lily toxicity"],
    )

    assert len(result.conflicts) == 1
    message = result.conflicts[0]
    assert "lily toxicity" in message
    assert "엇갈립니다" in message
    assert "수의사 확인" in message
    # 충돌이 있어도 근거 자체는 남는다 — 사용자에게 불확실성을 보여줘야 한다.
    assert len(result.evidence) == 2


def test_unaccepted_web_evidence_is_dropped() -> None:
    """검증을 통과하지 못한(accepted=False) 웹 결과는 무조건 버린다.

    validator 를 우회한 웹 근거가 조용히 섞이는 경로를 만들지 않는다.
    """
    result = merge_evidence(
        [make_evidence("rag-choco#0", RAG_CHOCOLATE)],
        [
            make_web("https://blog.example.com/x", WEB_CHOCOLATE, accepted=False),
            make_web("https://vet.cornell.edu/ok", WEB_CHOCOLATE, accepted=True),
        ],
        required_topics=["chocolate toxicity"],
    )

    urls = [item.source_url for item in result.evidence]
    assert "https://blog.example.com/x" not in urls
    assert "https://vet.cornell.edu/ok" in urls


def test_duplicate_rag_chunks_are_deduplicated() -> None:
    """같은 chunk 가 두 번 들어오면 한 번만 인용한다(ko/en 이중 검색의 결과)."""
    result = merge_evidence(
        [
            make_evidence("rag-choco#0", RAG_CHOCOLATE, score=0.9),
            make_evidence("rag-choco#0", RAG_CHOCOLATE, score=0.7),
        ],
        [],
        required_topics=["chocolate toxicity"],
    )

    assert len(result.evidence) == 1


# ---------------------------------------------------------------------------
# 11. 신뢰 가능한 근거 없음
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("rag_docs", "web_docs"),
    [
        ([], []),
        ([], [make_web("https://cafe.example.com/post", WEB_CHOCOLATE, accepted=False)]),
    ],
    ids=["nothing_at_all", "only_rejected_web"],
)
def test_has_reliable_evidence_is_false_without_usable_sources(
    rag_docs: list[RetrievedEvidence],
    web_docs: list[WebEvidence],
) -> None:
    """쓸 수 있는 근거가 하나도 없으면 has_reliable_evidence=False 여야 한다.

    이 값이 False 면 호출자는 추측하지 말고 "확실하지 않다 + 병원 상담 권고" 로
    답해야 한다. 거절된 웹 결과를 '근거 있음' 으로 세면 그 안전장치가 무너진다.
    """
    result = merge_evidence(rag_docs, web_docs, required_topics=["chocolate toxicity"])

    assert result.evidence == []
    assert result.has_reliable_evidence is False
    assert result.conflicts == []


# ---------------------------------------------------------------------------
# 12. FinalEvidence id 안정성 / source_type
# ---------------------------------------------------------------------------
def test_evidence_ids_are_stable_across_runs() -> None:
    """같은 입력이면 실행마다 같은 evidence_id 가 나와야 한다.

    답변 본문이 근거를 id 로 인용하고, PDF/trace 도 같은 id 로 대조한다.
    id 가 실행마다 바뀌면 인용이 전부 깨진다.
    """
    def run() -> list[str]:
        merged = merge_evidence(
            [make_evidence("rag-choco#0", RAG_CHOCOLATE)],
            [make_web("https://vet.cornell.edu/chocolate", WEB_CHOCOLATE)],
            required_topics=["chocolate toxicity"],
        )
        return [item.evidence_id for item in merged.evidence]

    assert run() == run()


def test_source_type_and_id_shape_per_source() -> None:
    """source_type 과 id 형태가 출처를 정확히 반영해야 한다.

    RAG 는 이미 안정적인 chunk_id 를 그대로 쓰고, 웹은 URL 해시 기반 `web:` id 를
    쓴다. trace 에서 눈으로 출처를 구분할 수 있어야 한다.
    """
    result = merge_evidence(
        [make_evidence("rag-choco#0", RAG_CHOCOLATE)],
        [make_web("https://vet.cornell.edu/chocolate", WEB_CHOCOLATE)],
        required_topics=["chocolate toxicity"],
    )
    rag_item, web_item = result.evidence

    assert rag_item.source_type == "rag"
    assert rag_item.evidence_id == "rag-choco#0"
    assert rag_item.source_url == "https://vet.cornell.edu/rag-choco#0"

    assert web_item.source_type == "web"
    assert web_item.evidence_id.startswith("web:vet.cornell.edu:")
    assert web_item.source_url == "https://vet.cornell.edu/chocolate"


def test_web_evidence_id_normalizes_url_case_and_trailing_slash() -> None:
    """대소문자·끝 슬래시만 다른 URL 은 같은 근거로 보고 같은 id 를 준다.

    같은 페이지가 표기 차이 때문에 서로 다른 근거처럼 두 번 인용되는 것을 막는다.
    """
    first = merge_evidence([], [make_web("https://VET.Cornell.edu/Chocolate/", WEB_CHOCOLATE)])
    second = merge_evidence([], [make_web("https://vet.cornell.edu/chocolate", WEB_CHOCOLATE)])

    assert first.evidence[0].evidence_id == second.evidence[0].evidence_id


def test_different_web_urls_get_different_ids() -> None:
    """다른 페이지는 반드시 다른 id 를 받아야 한다(인용 충돌 방지)."""
    result = merge_evidence(
        [],
        [
            make_web("https://vet.cornell.edu/chocolate", WEB_CHOCOLATE),
            make_web("https://vet.cornell.edu/recall", WEB_RECALL),
        ],
        required_topics=["chocolate toxicity", "product recall"],
    )

    ids = [item.evidence_id for item in result.evidence]
    assert len(set(ids)) == 2
