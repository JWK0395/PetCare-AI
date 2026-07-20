"""RAG 파사드 — 앞선 모듈들을 명세 순서대로 조립한다(명세 6·13~16·44절).

이 파일은 **새 로직을 만들지 않는다.** loader → normalizer → chunker →
vector_store 로 index 를 만들고, query_builder → retriever → sufficiency 로
근거를 뽑고, 필요할 때만 tavily → source_validator → evidence_merger 를 태운다.
판단 규칙은 전부 각 모듈 안에 있고 여기서는 **호출 순서와 호출 조건만** 책임진다.

설계상 반드시 지키는 것 3가지.

1. **Tavily 는 조건부 호출이다(명세 15절 / 47절 금지사항).**
   `sufficiency` 가 `insufficient` / `conflicting` 이거나
   `requires_recent_information=True` 일 때만 부른다. `sufficient` 인데도
   웹을 뒤지면 비용·지연이 늘고, 검증된 내부 근거보다 약한 자료가 섞인다.

2. **웹 결과는 `WebSourceValidator` 를 통과한 것만 병합한다(명세 15절).**
   검증 전 결과가 `merge_evidence` 로 흘러가는 경로 자체를 만들지 않는다.

3. **모든 외부 의존은 주입 가능하다.**
   store / llm / web_search / validator 를 생성자로 받으므로, 테스트는
   `DeterministicEmbeddings` 와 mock Tavily client 만으로 전 분기를 검증할 수 있다.
   주입하지 않으면 실제로 쓰이는 시점에 지연 생성한다 — 서비스 객체를 만들기만
   하고 쓰지 않는 경우에 모델 로딩·API 키 조회가 일어나지 않게 하기 위해서다.

무거운 하위 모듈(faiss 를 쓰는 vector_store 등)은 메서드 안에서 지연 import 한다.
`petcare_ai.rag.service` 를 import 하는 것만으로 faiss/tavily 가 끌려오면
loader 만 쓰려는 테스트까지 함께 죽기 때문이다.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from ..config import Settings, Species, get_settings
from ..schemas import (
    EvidenceMergeResult,
    KnowledgeSufficiencyResult,
    RagQuery,
    RetrievalResult,
    RetrievedEvidence,
    WebEvidence,
)
from .loader import DEFAULT_ZIP_MEMBER, LoadReport

if TYPE_CHECKING:  # 런타임에 faiss/chunker 를 끌고 오지 않기 위한 타입 전용 import
    from .chunker import Chunk
    from .vector_store import VeterinaryVectorStore

logger = logging.getLogger(__name__)

__all__ = [
    "IndexBuildReport",
    "VeterinaryRagService",
    "WEB_FALLBACK_STATUSES",
]

#: 이 충분성 상태일 때만 웹 fallback 을 허용한다(명세 15절).
#: `sufficient` 는 포함되지 않는다 — Tavily 무조건 호출은 금지 사항이다.
WEB_FALLBACK_STATUSES: frozenset[str] = frozenset({"insufficient", "conflicting"})


class IndexBuildReport(BaseModel):
    """`build_index()` 결과 — 노트북에서 index 생성/재사용 여부를 눈으로 확인한다."""

    reused: bool = False
    reason: str = ""
    index_dir: str = ""
    document_count: int = 0
    chunk_count: int = 0
    index_counts: dict[str, int] = Field(default_factory=dict)
    chunk_stats: dict[str, Any] = Field(default_factory=dict)
    load_report: LoadReport | None = None

    def summary(self) -> str:
        """사람이 읽는 한 줄 요약."""
        counts = ", ".join(f"{k}={v}" for k, v in sorted(self.index_counts.items())) or "없음"
        if self.reused:
            # 재사용 경로는 문서를 아예 읽지 않는다 — "문서 0건" 으로 적으면
            # 데이터 로드가 실패한 것처럼 보인다.
            return f"기존 index 재사용 — 문서 로드 생략 / chunk {self.chunk_count}건 / {counts}"
        return (
            f"index 재생성 — 문서 {self.document_count}건 / "
            f"chunk {self.chunk_count}건 / {counts}"
        )


def _summarize_documents(documents: list[RetrievedEvidence]) -> list[dict[str, Any]]:
    """근거를 표로 찍기 좋은 얕은 dict 로 줄인다 — 본문 전체는 담지 않는다."""
    return [
        {
            "chunk_id": doc.chunk_id,
            "title": doc.title,
            "score": doc.score,
            "heading_path": " > ".join(doc.heading_path),
            "preview": doc.text[:80].replace("\n", " "),
        }
        for doc in documents
    ]


def _summarize_web(items: list[WebEvidence]) -> list[dict[str, Any]]:
    """웹 결과를 검증 사유까지 포함해 요약한다 — 왜 떨어졌는지 설명할 수 있어야 한다."""
    return [
        {
            "domain": item.domain,
            "url": item.url,
            "accepted": item.accepted,
            "reject_reason": item.reject_reason,
            "score": item.score,
        }
        for item in items
    ]


class VeterinaryRagService:
    """Cornell RAG + 조건부 웹 fallback 을 하나로 묶은 파사드.

    노트북(명세 44절)과 LangGraph 노드가 공통으로 쓰는 진입점이다.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        store: "VeterinaryVectorStore | None" = None,
        llm: Any = None,
        web_search: Any = None,
        validator: Any = None,
    ) -> None:
        """의존 객체를 전부 주입 가능하게 받는다.

        모두 `None` 이어도 되고, 그때는 실제로 필요해지는 시점에 설정대로
        만든다. `llm=None` 이면 query_builder / sufficiency 가 규칙 기반으로
        동작하므로 API 키 없이도 파이프라인 전체가 끝까지 돈다.
        """
        self.settings: Settings = settings or get_settings()
        self._store = store
        self.llm = llm
        self._web_search = web_search
        self._validator = validator
        self._evaluator: Any = None
        #: store 를 지연 생성할 때 디스크 index 를 로드했는지. None 이면 아직 시도 전.
        #: 서비스 기동 점검(/health 등)에서 RAG 사용 가능 여부를 확인하는 데 쓴다.
        self.index_autoload_ok: bool | None = None
        #: 마지막 호출의 단계별 결과. 노트북 표 출력·trace 용이다.
        self.last_debug: dict[str, Any] = {}

    # -- 지연 생성되는 협력 객체 ------------------------------------------
    @property
    def store(self) -> "VeterinaryVectorStore":
        """vector store — 주입되지 않았으면 설정대로 지연 생성하고 **디스크 index 를 로드한다.**

        로드까지 여기서 하는 이유(회귀 방지):
        노트북은 `build_index()` 를 거치므로 그 안에서 `store.load()` 가 호출되지만,
        서비스(`ai/app/*`)는 이미 만들어 둔 index 를 쓰기만 하므로 `build_index()` 를
        호출하지 않는다. 그래서 예전에는 `PETCARE_INDEX_DIR` 이 정확해도 index 가
        **로드되지 않은 채** `search()` 가 빈 리스트를 돌려주었다.
        `search()` 는 index 가 없을 때 예외 대신 빈 결과를 주도록 설계돼 있어
        (호출자가 죽지 않게 하려는 의도) **오류 없이 RAG 근거만 0건**이 되고,
        답변과 /health 는 정상이라 배포 후에도 발견되지 않는다.

        따라서 store 를 만들 때 1회 `load()` 를 시도한다. index 가 없거나 설정
        지문이 다르면 `load()` 가 False 를 돌려주고, 그 사실을 경고 로그로 남긴다
        (조용한 실패를 만들지 않는다). 주입된 store 는 호출자가 관리하므로 건드리지 않는다.
        """
        if self._store is None:
            from .vector_store import VeterinaryVectorStore  # noqa: PLC0415

            self._store = VeterinaryVectorStore(self.settings)
            self._autoload_store(self._store)
        return self._store

    def _autoload_store(self, store: "VeterinaryVectorStore") -> None:
        """디스크에 저장된 index 를 1회 로드한다(실패해도 서비스는 계속 동작)."""
        index_dir = getattr(self.settings, "index_dir", None)
        try:
            loaded = store.load()
        except Exception as exc:  # pragma: no cover - 손상된 index 등
            logger.warning("index 자동 로드 실패(%s): %s", index_dir, exc)
            self.index_autoload_ok = False
            return

        self.index_autoload_ok = bool(loaded)
        if loaded:
            logger.info(
                "index 자동 로드 완료: %s (species=%s)",
                index_dir, sorted(store.loaded_species),
            )
        else:
            logger.warning(
                "index 를 로드하지 못했습니다: %s. "
                "RAG 근거 없이 동작합니다(웹 fallback 경로). "
                "index 를 생성했는지, 생성 시 embedding_backend/모델 설정이 "
                "현재 설정과 같은지 확인하세요.",
                index_dir,
            )

    @property
    def web_search(self) -> Any:
        """수의학 웹 검색 서비스(명세 15절). 병원 검색과는 별개 class 다."""
        if self._web_search is None:
            from .tavily_vet_search import VeterinaryWebSearchService  # noqa: PLC0415

            self._web_search = VeterinaryWebSearchService(self.settings)
        return self._web_search

    @property
    def validator(self) -> Any:
        """웹 근거 검증기 — 웹 결과가 근거가 되는 유일한 관문이다."""
        if self._validator is None:
            from .source_validator import WebSourceValidator  # noqa: PLC0415

            self._validator = WebSourceValidator(self.settings)
        return self._validator

    @property
    def evaluator(self) -> Any:
        """충분성 평가기 — llm 을 그대로 물려준다(없으면 deterministic 판정만)."""
        if self._evaluator is None:
            from .sufficiency import KnowledgeSufficiencyEvaluator  # noqa: PLC0415

            self._evaluator = KnowledgeSufficiencyEvaluator(self.settings, llm=self.llm)
        return self._evaluator

    # -- index 구축 --------------------------------------------------------
    def build_index(
        self,
        documents_path: str | Path | None = None,
        zip_path: str | Path | None = None,
        index_dir: str | Path | None = None,
        force_rebuild: bool = False,
        zip_member: str = DEFAULT_ZIP_MEMBER,
        limit: int | None = None,
    ) -> IndexBuildReport:
        """문서를 읽어 species 별 FAISS index 를 만든다(없거나 설정이 바뀐 경우에만).

        명세 44절: "노트북 재실행 시 기존 FAISS index 가 있으면 로드하고, 설정이
        달라졌을 때만 재생성한다." 그래서 **가장 먼저 `store.load()` 를 시도한다.**
        `VeterinaryVectorStore.load()` 는 저장된 `meta.json` 의 설정 지문(임베딩
        백엔드/모델/정규화, chunk 파라미터, 차원)이 지금과 다르면 `False` 를
        돌려주므로, "설정이 같을 때만 재사용" 이 여기서 자동으로 성립한다.
        지문 비교를 이 파일에서 다시 구현하면 두 곳이 어긋나므로 하지 않는다.

        인자
          - `documents_path`: JSON 경로. 없으면 `settings.documents_path`.
          - `zip_path`: 주면 압축을 풀지 않고 zip 안에서 바로 읽는다(Colab 용).
          - `index_dir`: 저장 위치. `None` 이면 `settings.index_path(species)`.
          - `force_rebuild`: 지문이 같아도 무조건 다시 만든다.
          - `limit`: 스모크 테스트용 문서 수 상한. dog/cat 을 번갈아 담아
            한쪽 index 만 생기는 상황을 피한다(그러면 반대 종 검색이 통째로 빈다).

        문서를 한 건도 읽지 못하면 조용히 빈 index 를 만들지 않고 예외를 던진다.
        빈 index 는 이후 모든 질문을 "근거 없음" 으로 만들어 원인 추적이 어렵다.
        """
        self.last_debug = {"stage": "build_index"}
        report = IndexBuildReport(index_dir=str(index_dir or self.settings.index_dir))

        # (1) 재사용 가능한 index 가 있는가
        if not force_rebuild and self.store.load(index_dir):
            report.reused = True
            report.reason = "저장된 index 의 설정 지문이 현재 설정과 같아 그대로 로드했습니다."
            report.index_counts = {
                species: self.store.index_size(species)  # type: ignore[arg-type]
                for species in sorted(self.store.loaded_species)
            }
            report.chunk_count = sum(report.index_counts.values())
            self.last_debug["index"] = report.model_dump(exclude={"load_report"})
            logger.info("[build_index] %s", report.summary())
            return report

        # (2) 로드 → 정규화 → chunking
        from .chunker import chunk_documents, chunk_stats  # noqa: PLC0415
        from .loader import load_documents, load_documents_from_zip  # noqa: PLC0415
        from .normalizer import normalize_documents  # noqa: PLC0415

        if zip_path is not None:
            documents, load_report = load_documents_from_zip(zip_path, zip_member)
            origin = f"{zip_path}!{zip_member}"
        else:
            resolved_path = Path(documents_path) if documents_path else self.settings.documents_path
            documents, load_report = load_documents(resolved_path)
            origin = str(resolved_path)

        report.load_report = load_report
        if not documents:
            head = "; ".join(load_report.errors[:3]) or "원인 불명"
            raise RuntimeError(
                f"문서를 한 건도 로드하지 못해 index 를 만들 수 없습니다(경로: {origin}). 원인: {head}"
            )

        if limit is not None:
            documents = self._balanced_slice(documents, limit)

        normalized = normalize_documents(documents)
        if not normalized:
            raise RuntimeError(
                "정규화 후 본문이 남은 문서가 없어 index 를 만들 수 없습니다. "
                "원문의 content_markdown / content_text 를 확인하세요."
            )

        chunks: list["Chunk"] = chunk_documents(normalized, self.settings.rag)
        if not chunks:
            raise RuntimeError(
                "chunk 가 생성되지 않아 index 를 만들 수 없습니다. "
                f"min_chunk_length={self.settings.rag.min_chunk_length} 설정을 확인하세요."
            )

        # (3) build → save
        self.settings.ensure_dirs()
        index_counts = self.store.build_all(chunks)
        self.store.save(index_dir)

        report.reused = False
        report.reason = (
            "저장된 index 가 없거나 설정 지문이 달라 재생성했습니다."
            if not force_rebuild
            else "force_rebuild=True 로 재생성했습니다."
        )
        report.document_count = len(normalized)
        report.chunk_count = len(chunks)
        report.index_counts = index_counts
        report.chunk_stats = chunk_stats(chunks)

        self.last_debug["index"] = report.model_dump(exclude={"load_report"})
        logger.info("[build_index] %s", report.summary())
        return report

    @staticmethod
    def _balanced_slice(documents: list[dict], limit: int) -> list[dict]:
        """species 를 번갈아 뽑아 `limit` 건으로 줄인다(스모크 테스트용).

        앞에서부터 그냥 자르면 원본 정렬 때문에 dog 문서만 남아 cat index 가
        아예 만들어지지 않는다. 그 상태로 고양이 질문을 던지면 파이프라인
        버그와 데이터 슬라이싱 문제를 구분할 수 없다.
        """
        if limit <= 0 or limit >= len(documents):
            return documents

        buckets: dict[str, list[dict]] = {}
        for doc in documents:
            buckets.setdefault(str(doc.get("species") or "unknown"), []).append(doc)

        selected: list[dict] = []
        order = sorted(buckets)
        position = 0
        while len(selected) < limit and any(position < len(buckets[s]) for s in order):
            for species in order:
                if position < len(buckets[species]) and len(selected) < limit:
                    selected.append(buckets[species][position])
            position += 1
        return selected

    # -- 검색 --------------------------------------------------------------
    def retrieve(
        self,
        user_message: str,
        pet_profile: dict,
        related_diagnoses: list[dict] | None = None,
        supporting_daily_entries: list[dict] | None = None,
    ) -> RetrievalResult:
        """query 생성 → 검색 → 충분성 판정까지 수행한다(명세 6·12~14절).

        `web_fallback_required` 는 여기서 정해지고, `retrieve_with_fallback()` 은
        그 값만 보고 Tavily 호출 여부를 결정한다. 판단 지점을 한 곳으로 모아야
        "모든 질문에 Tavily 호출" 같은 사고가 구조적으로 막힌다.
        """
        self.last_debug = {"stage": "retrieve"}
        result, _query, _sufficiency = self._retrieve_stage(
            user_message, pet_profile, related_diagnoses, supporting_daily_entries
        )
        return result

    def retrieve_with_fallback(
        self,
        user_message: str,
        pet_profile: dict,
        related_diagnoses: list[dict] | None = None,
        supporting_daily_entries: list[dict] | None = None,
        max_web_results: int = 5,
    ) -> tuple[RetrievalResult, EvidenceMergeResult]:
        """검색 + (필요할 때만) 웹 fallback + 근거 병합까지 끝낸다(명세 15·16절).

        Tavily 호출 조건은 명세 15절 그대로다.
          - 충분성이 `insufficient` 또는 `conflicting`
          - 또는 `requires_recent_information=True`(리콜·최신 권고처럼 내부
            문서가 구조적으로 못 따라가는 질문)

        그 외에는 **웹 검색을 아예 호출하지 않는다.** 검색된 결과도 반드시
        `WebSourceValidator` 를 통과한 것만 `merge_evidence` 로 넘긴다.
        Tavily 실패·키 없음·전량 거절은 예외가 아니라 정상 경로이며, 그때는
        RAG 근거만으로 병합한다(근거가 하나도 없으면
        `EvidenceMergeResult.has_reliable_evidence=False` 가 되어 호출자가
        추측 대신 "확실하지 않음 + 병원 상담 권고"로 답해야 한다).
        """
        from .evidence_merger import merge_evidence  # noqa: PLC0415

        self.last_debug = {"stage": "retrieve_with_fallback"}
        result, query, sufficiency = self._retrieve_stage(
            user_message, pet_profile, related_diagnoses, supporting_daily_entries
        )

        accepted: list[WebEvidence] = []
        if result.web_fallback_required:
            accepted = self._web_fallback(query, sufficiency, max_web_results)
        else:
            self.last_debug["web_search_called"] = False
            self.last_debug["web_skip_reason"] = (
                f"충분성이 '{sufficiency.status}' 이고 최신 정보도 필요하지 않아 "
                "웹 검색을 호출하지 않았습니다."
            )

        rag_docs = self._reserve_web_slots(result.documents, accepted)
        merged = merge_evidence(rag_docs, accepted, query.required_topics)
        self.last_debug["merge"] = {
            "evidence_count": len(merged.evidence),
            "rag_count": sum(1 for e in merged.evidence if e.source_type == "rag"),
            "web_count": sum(1 for e in merged.evidence if e.source_type == "web"),
            "conflicts": merged.conflicts,
            "has_reliable_evidence": merged.has_reliable_evidence,
        }
        return result, merged

    def _reserve_web_slots(
        self,
        rag_docs: list[RetrievedEvidence],
        accepted: list[WebEvidence],
    ) -> list[RetrievedEvidence]:
        """검증된 웹 근거가 최종 목록에 들어갈 자리를 확보한다.

        `merge_evidence()` 는 RAG 를 항상 앞에 놓고 `final_evidence_max` 로 잘라낸다
        (명세 16절 우선순위). 그런데 retriever 는 상한(기본 8건)까지 꽉 채워 돌려주므로,
        **RAG 만으로 상한이 차서 웹 근거가 100% 잘려 나간다.** 웹 fallback 은 애초에
        "RAG 가 부족하다(insufficient/conflicting)"고 판정됐을 때만 도는 경로라,
        그 상황에서 관련도가 가장 낮은 RAG chunk 를 남기고 검증까지 통과한 웹 근거를
        버리는 것은 fallback 을 무력화하는 것과 같다.

        그래서 채택된 웹 근거가 있을 때만 RAG 목록의 **꼬리(=최저 점수)** 를 잘라
        자리를 만든다. 우선순위 자체는 그대로다 — RAG 는 여전히 앞에 오고, 점수가
        높은 RAG 근거는 하나도 밀려나지 않는다.
        """
        if not accepted or not rag_docs:
            return rag_docs

        # merge_evidence 가 실제로 적용하는 상한과 같은 값을 봐야 계산이 어긋나지 않는다.
        limit = int(get_settings().rag.final_evidence_max)
        if limit <= 0 or len(rag_docs) + len(accepted) <= limit:
            return rag_docs

        # 웹에 넘길 자리는 최대 limit 의 1/4 로 제한한다. 내부 수의학 문서가
        # 주 근거라는 원칙을 유지하면서 보조 근거 한두 건만 통과시키기 위해서다.
        reserved = min(len(accepted), max(1, limit // 4))
        keep = max(1, limit - reserved)
        if keep >= len(rag_docs):
            return rag_docs

        logger.info(
            "[merge] 검증된 웹 근거 %d건의 자리를 위해 RAG 근거를 %d건 → %d건으로 줄입니다.",
            len(accepted),
            len(rag_docs),
            keep,
        )
        self.last_debug["rag_trimmed_for_web"] = {"from": len(rag_docs), "to": keep}
        return rag_docs[:keep]

    # -- 내부 단계 ---------------------------------------------------------
    def _retrieve_stage(
        self,
        user_message: str,
        pet_profile: dict,
        related_diagnoses: list[dict] | None,
        supporting_daily_entries: list[dict] | None,
    ) -> tuple[RetrievalResult, RagQuery, KnowledgeSufficiencyResult]:
        """`retrieve()` 와 `retrieve_with_fallback()` 이 공유하는 본체.

        `RagQuery` 와 `KnowledgeSufficiencyResult` 원본을 함께 돌려주는 이유:
        fallback 단계에서 검색어·required_topics·`requires_recent_information`
        이 필요한데, `RetrievalResult` 로 축약하면 그 정보가 사라진다.
        """
        from .query_builder import build_rag_query  # noqa: PLC0415
        from .retriever import retrieval_stats  # noqa: PLC0415
        from .retriever import retrieve as retrieve_evidence  # noqa: PLC0415

        # (1) query 생성 — llm 이 없으면 규칙 기반으로 동작한다.
        query = build_rag_query(
            user_message,
            pet_profile or {},
            list(related_diagnoses or []),
            list(supporting_daily_entries or []),
            self.llm,
        )
        self.last_debug["query"] = query.model_dump()

        # (2) 검색 — species 는 query 가 고정한다(교차 오염 방지).
        self.last_debug["index_loaded_species"] = sorted(self.store.loaded_species)
        documents = retrieve_evidence(self.store, query, self.settings)
        self.last_debug["documents"] = _summarize_documents(documents)
        self.last_debug["retrieval_stats"] = retrieval_stats(documents)

        # (3) 충분성 판정
        sufficiency = self.evaluator.evaluate(query, documents)
        self.last_debug["sufficiency"] = sufficiency.model_dump()

        web_required = (
            sufficiency.status in WEB_FALLBACK_STATUSES
            or bool(sufficiency.requires_recent_information)
        )
        self.last_debug["web_fallback_required"] = web_required

        result = RetrievalResult(
            query=query.primary_query_ko,
            species=query.species,
            documents=documents,
            sufficiency=sufficiency.status,
            covered_topics=sufficiency.covered_topics,
            missing_topics=sufficiency.missing_topics,
            web_fallback_required=web_required,
        )
        return result, query, sufficiency

    def _web_fallback(
        self,
        query: RagQuery,
        sufficiency: KnowledgeSufficiencyResult,
        max_web_results: int,
    ) -> list[WebEvidence]:
        """웹 검색 → 검증 → 통과분만 반환한다. 실패는 전부 빈 리스트로 흡수한다."""
        from .source_validator import accepted_only  # noqa: PLC0415

        # Cornell/Merck 등 allowlist 문서는 영어라 영어 query 를 우선한다.
        search_text = query.primary_query_en.strip() or query.primary_query_ko.strip()
        self.last_debug["web_search_called"] = True
        self.last_debug["web_search_reason"] = (
            f"충분성='{sufficiency.status}', "
            f"requires_recent_information={sufficiency.requires_recent_information}"
        )
        self.last_debug["web_query"] = search_text

        try:
            raw = self.web_search.search(search_text, query.species, max_web_results)
        except Exception as exc:  # 웹 fallback 실패는 정상 경로다(명세 15절).
            logger.warning("[web-fallback] 웹 검색 호출에 실패했습니다: %s", exc)
            self.last_debug["web_error"] = str(exc)
            raw = []

        validated: list[WebEvidence] = self.validator.validate(raw, query.species, query)
        accepted = accepted_only(validated)

        self.last_debug["web_raw_count"] = len(raw)
        self.last_debug["web_accepted_count"] = len(accepted)
        self.last_debug["web_results"] = _summarize_web(validated)
        if raw and not accepted:
            logger.info("[web-fallback] 웹 결과 %d건이 모두 검증에서 거절됐습니다.", len(raw))
        return accepted

    # -- 노트북 편의 --------------------------------------------------------
    def debug_rows(self) -> list[dict[str, str]]:
        """`last_debug` 를 표 한 장으로 찍기 좋은 행 목록으로 바꾼다(명세 44절 데모용).

        노트북 셀에서 `pd.DataFrame(service.debug_rows())` 로 바로 출력할 수 있게
        값은 전부 문자열로 평탄화한다.
        """
        debug = self.last_debug
        if not debug:
            return [{"단계": "-", "결과": "아직 호출되지 않았습니다.", "비고": ""}]

        rows: list[dict[str, str]] = []

        index = debug.get("index")
        if isinstance(index, dict):
            rows.append(
                {
                    "단계": "0. index",
                    "결과": "재사용" if index.get("reused") else "재생성",
                    "비고": f"{index.get('index_counts')} / {index.get('reason', '')}",
                }
            )

        query = debug.get("query")
        if isinstance(query, dict):
            rows.append(
                {
                    "단계": "1. query_builder",
                    "결과": str(query.get("primary_query_ko", "")),
                    "비고": f"en={query.get('primary_query_en', '')} / "
                    f"species={query.get('species')} / "
                    f"emergency_hint={query.get('emergency_hint')}",
                }
            )

        documents = debug.get("documents")
        if isinstance(documents, list):
            stats = debug.get("retrieval_stats") or {}
            rows.append(
                {
                    "단계": "2. retriever",
                    "결과": f"{len(documents)}건",
                    "비고": f"문서 {stats.get('documents')}종 / "
                    f"max_score={stats.get('max_score')} / "
                    f"index={debug.get('index_loaded_species')}",
                }
            )

        sufficiency = debug.get("sufficiency")
        if isinstance(sufficiency, dict):
            rows.append(
                {
                    "단계": "3. sufficiency",
                    "결과": str(sufficiency.get("status")),
                    "비고": f"covered={sufficiency.get('covered_topics')} / "
                    f"missing={sufficiency.get('missing_topics')} / "
                    f"recent={sufficiency.get('requires_recent_information')}",
                }
            )

        if "web_fallback_required" in debug:
            if "web_search_called" not in debug:
                # `retrieve()` 단독 호출은 fallback 단계 자체를 실행하지 않는다.
                # "호출 안 함" 으로 적으면 조건 판정 결과로 오해된다.
                outcome = "미실행"
                note = (
                    f"retrieve() 단독 호출 — "
                    f"web_fallback_required={debug.get('web_fallback_required')}"
                )
            elif debug.get("web_search_called"):
                outcome = "호출함"
                note = str(
                    debug.get("web_error")
                    or f"raw={debug.get('web_raw_count')} / "
                    f"accepted={debug.get('web_accepted_count')} / "
                    f"{debug.get('web_search_reason', '')}"
                )
            else:
                outcome = "호출 안 함"
                note = str(debug.get("web_skip_reason") or "")
            rows.append({"단계": "4. web fallback", "결과": outcome, "비고": note})

        merge = debug.get("merge")
        if isinstance(merge, dict):
            rows.append(
                {
                    "단계": "5. evidence_merge",
                    "결과": f"{merge.get('evidence_count')}건 "
                    f"(rag={merge.get('rag_count')}, web={merge.get('web_count')})",
                    "비고": f"신뢰 근거={merge.get('has_reliable_evidence')} / "
                    f"충돌={len(merge.get('conflicts') or [])}건",
                }
            )
        return rows

    def index_ready(self, species: Species | None = None) -> bool:
        """검색 가능한 index 가 올라와 있는지 — 노트북/그래프가 미리 분기할 때 쓴다."""
        loaded = self.store.loaded_species
        return bool(loaded) if species is None else species in loaded

    def __repr__(self) -> str:  # pragma: no cover - 디버깅 편의
        return (
            f"VeterinaryRagService(index={sorted(self.store.loaded_species)}, "
            f"llm={'있음' if self.llm is not None else '없음'})"
        )
