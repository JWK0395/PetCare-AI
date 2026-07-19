"""Retriever — 명세 13절. 한국어·영어 query 를 함께 던지고 결과를 합친다.

**왜 두 번 검색하는가**
Cornell 문서는 전부 영어라 영어 query 가 본문 표현과 직접 맞아떨어진다. 반면
한국어 query 는 다국어 임베딩(bge-m3 / multilingual-e5)이 교차언어로 이어 주면서
영어 query 가 놓친 문서를 잡아 준다. 두 검색은 서로를 보완하므로 합집합을 쓴다.

**중복 제거**
같은 chunk 가 양쪽 검색에 다 걸리는 일이 흔하다(그게 정상이고, 오히려 관련도가
높다는 신호다). `chunk_id` 기준으로 하나만 남기되 **더 높은 점수를 유지한다.**
낮은 점수를 남기면 그 chunk 가 순위에서 부당하게 밀리고, 충분성 판단(명세 14절)의
임계값 비교도 실제보다 박하게 나온다.

**점수 정규화·정렬 기준**
`VeterinaryVectorStore.search()` 가 돌려주는 `score` 는 질의 벡터와 chunk 벡터의
**코사인 유사도를 0~1 로 클램프한 값**이다(문서·질의 모두 L2 정규화 → FAISS
IndexFlatIP 의 내적 = 코사인, 음수는 0). ko/en 두 검색이 **같은 임베딩 공간·같은
척도**를 쓰므로 별도 재정규화 없이 점수를 직접 비교·정렬할 수 있다. 거리가 아니라
유사도이므로 **내림차순**(1.0 에 가까울수록 관련)으로 정렬하고, 점수가 같으면
먼저 검색된 순서(ko → en, 각 검색 내 원래 순위)를 유지하는 안정 정렬을 쓴다.
`score` 가 None 인 결과는 0.0 으로 보고 맨 뒤로 보낸다.

**species 고정**
검색 종은 `query.species` 하나로 고정한다. store 에 dog/cat index 가 둘 다 올라와
있어도 다른 종 index 는 건드리지 않고, 혹시 metadata 가 오염돼 다른 종 evidence 가
섞여 나오면 여기서 한 번 더 버린다(2차 방어선).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from ..config import Settings, Species, get_settings
from ..schemas import RagQuery, RetrievedEvidence

if TYPE_CHECKING:  # faiss 를 끌고 오지 않도록 런타임 import 를 피한다.
    from .vector_store import VeterinaryVectorStore

logger = logging.getLogger(__name__)

__all__ = [
    "MIN_FINAL_EVIDENCE",
    "MAX_FINAL_EVIDENCE",
    "retrieve",
    "retrieve_multi",
    "deduplicate_evidence",
    "resolve_final_evidence_limit",
    "retrieval_stats",
]

#: 명세 13절 "최종 상위 4~8개". 설정값이 벗어나면 이 범위로 잘라 쓴다.
MIN_FINAL_EVIDENCE: int = 4
MAX_FINAL_EVIDENCE: int = 8


def resolve_final_evidence_limit(settings: Settings) -> int:
    """최종 반환 개수를 4~8 범위로 확정한다.

    `RagSettings.final_evidence_max` 를 노트북에서 자유롭게 바꿀 수 있게 열어 뒀지만,
    3개 이하면 근거가 빈약해지고 9개 이상이면 프롬프트가 길어져 답변 품질이 떨어진다.
    명세가 정한 범위를 코드에서 강제한다.
    """
    raw = int(getattr(settings.rag, "final_evidence_max", MAX_FINAL_EVIDENCE))
    clamped = max(MIN_FINAL_EVIDENCE, min(MAX_FINAL_EVIDENCE, raw))
    if clamped != raw:
        logger.warning(
            "final_evidence_max=%d 는 명세 범위(%d~%d)를 벗어나 %d 로 조정했습니다.",
            raw,
            MIN_FINAL_EVIDENCE,
            MAX_FINAL_EVIDENCE,
            clamped,
        )
    return clamped


def _score_of(evidence: RetrievedEvidence) -> float:
    """정렬용 점수 — None 은 0.0(무관)으로 본다."""
    return float(evidence.score) if evidence.score is not None else 0.0


def deduplicate_evidence(
    batches: list[list[RetrievedEvidence]],
    species: Species | None = None,
) -> list[RetrievedEvidence]:
    """여러 검색 결과를 합쳐 `chunk_id` 중복을 제거하고 점수 내림차순으로 정렬한다.

    - 같은 `chunk_id` 가 여러 번 나오면 **점수가 더 높은 쪽을 남긴다.**
    - `species` 를 주면 그 종의 evidence 만 통과시킨다(교차 오염 차단).
    - `chunk_id` 가 비어 있는 결과는 중복 판정을 할 수 없으므로 등장 순서를 키로 삼아
      각각 별개 근거로 취급한다(버리지 않는다 — 근거 손실이 더 나쁘다).

    정렬은 (점수 내림차순, 최초 등장 순서 오름차순)이라 점수가 같으면 먼저 검색된
    query(=한국어 query) 결과가 앞에 온다. 결정적이라 테스트가 재현 가능하다.
    """
    best: dict[str, tuple[int, RetrievedEvidence]] = {}
    order = 0
    dropped_species = 0
    duplicates = 0

    for batch in batches:
        for evidence in batch:
            if species is not None and evidence.species != species:
                dropped_species += 1
                continue
            key = evidence.chunk_id or f"__anonymous__{order}"
            existing = best.get(key)
            if existing is None:
                best[key] = (order, evidence)
            else:
                duplicates += 1
                if _score_of(evidence) > _score_of(existing[1]):
                    # 순서는 최초 등장 시점을 유지하고 내용/점수만 더 좋은 것으로 바꾼다.
                    best[key] = (existing[0], evidence)
            order += 1

    if dropped_species:
        logger.warning(
            "species 가 %r 이 아닌 검색 결과 %d건을 제외했습니다(교차 오염 방지).",
            species,
            dropped_species,
        )
    if duplicates:
        logger.debug("chunk_id 중복 %d건을 제거했습니다(더 높은 점수 유지).", duplicates)

    ranked = sorted(best.values(), key=lambda item: (-_score_of(item[1]), item[0]))
    return [evidence for _, evidence in ranked]


def retrieve_multi(
    store: "VeterinaryVectorStore",
    queries: list[str],
    species: Species,
    settings: Settings | None = None,
    limit: int | None = None,
) -> list[RetrievedEvidence]:
    """여러 검색어를 같은 species index 에 던지고 병합 결과를 돌려준다.

    `retrieve()` 의 일반형이다. 웹 fallback 이후 재검색이나 노트북 실험에서
    임의의 query 조합을 시험할 때 쓰라고 분리해 뒀다.

    검색 중 예외가 나면 그 query 만 건너뛰고 나머지로 진행한다. 근거 검색 실패는
    파이프라인상 "근거 부족 → 웹 fallback" 으로 흡수되는 정상 경로라, 한쪽 query 의
    실패로 전체 답변을 죽이지 않는다. 다만 `ImportError`(faiss 미설치 등)는 환경
    설정 문제이므로 삼키지 않고 그대로 올려 보낸다.
    """
    resolved = settings or get_settings()
    rag = resolved.rag
    final_max = limit if limit is not None else resolve_final_evidence_limit(resolved)

    cleaned = []
    seen: set[str] = set()
    for text in queries:
        stripped = (text or "").strip()
        if stripped and stripped not in seen:
            seen.add(stripped)
            cleaned.append(stripped)

    if not cleaned:
        logger.warning("검색어가 비어 있어 빈 결과를 돌려줍니다.")
        return []

    # query 하나당 top_k 를 뽑는다. 합쳐서 중복 제거한 뒤 final_max 로 자르므로
    # top_k 는 final_max 보다 작아도 되지만, 최소한 final_max 는 확보해 둬야
    # 두 query 결과가 완전히 겹칠 때도 반환 개수가 모자라지 않는다.
    per_query_k = max(int(rag.top_k), final_max)
    fetch_k = max(int(rag.fetch_k), per_query_k)

    batches: list[list[RetrievedEvidence]] = []
    for text in cleaned:
        try:
            batches.append(store.search(text, species, k=per_query_k, fetch_k=fetch_k))
        except ImportError:
            raise
        except Exception as exc:  # 한 query 의 실패가 전체를 막지 않는다.
            logger.warning("검색 실패 — 이 query 는 건너뜁니다 (%r): %s", text[:40], exc)
            batches.append([])

    merged = deduplicate_evidence(batches, species=species)
    logger.info(
        "검색 %d건 → 병합 후 %d건 → 상위 %d건 반환 (species=%s)",
        sum(len(batch) for batch in batches),
        len(merged),
        min(len(merged), final_max),
        species,
    )
    return merged[:final_max]


def retrieve(
    store: "VeterinaryVectorStore",
    query: RagQuery,
    settings: Settings | None = None,
) -> list[RetrievedEvidence]:
    """`RagQuery` 의 한국어·영어 query 로 검색해 최종 근거 4~8건을 돌려준다(명세 13절).

    처리 순서
      1. `query.species` 로 검색 대상 index 를 **고정한다**(다른 종은 절대 섞지 않는다).
      2. 한국어 query 검색 → 영어 query 검색.
      3. 두 결과를 합치고 `chunk_id` 중복을 제거한다(더 높은 score 유지).
      4. score(0~1 로 클램프된 코사인 유사도) 내림차순으로 정렬한다.
         동점이면 먼저 검색된 순서를 유지한다.
      5. 상위 `final_evidence_max`(4~8 로 클램프) 건을 돌려준다.

    index 가 없거나 검색 결과가 없으면 예외 대신 빈 리스트를 돌려준다. 그 빈 결과가
    충분성 판단에서 `insufficient` → 웹 fallback 으로 이어지는 정상 경로다.
    """
    return retrieve_multi(
        store,
        [query.primary_query_ko, query.primary_query_en],
        query.species,
        settings=settings,
    )


def retrieval_stats(evidence: list[RetrievedEvidence]) -> dict[str, Any]:
    """검색 결과 요약 — 노트북에서 품질을 한눈에 볼 때 쓴다."""
    scores = [_score_of(item) for item in evidence]
    return {
        "count": len(evidence),
        "documents": len({item.document_id for item in evidence}),
        "species": sorted({item.species for item in evidence}),
        "max_score": max(scores) if scores else None,
        "min_score": min(scores) if scores else None,
        "mean_score": (sum(scores) / len(scores)) if scores else None,
    }
