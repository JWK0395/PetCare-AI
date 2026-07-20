"""species 별로 분리된 FAISS vector store — 명세 11절.

핵심 요구사항 3가지를 코드 구조로 강제한다.

1. **강아지/고양이 문서를 절대 섞지 않는다.**
   index 자체를 species 별로 분리하고(`faiss_dog/`, `faiss_cat/`),
   `build()` 는 넘어온 chunk 중 해당 species 것만 다시 걸러 담는다.
   검색도 species index 안에서만 일어나므로 교차 오염이 구조적으로 불가능하다.

2. **index 를 매번 다시 만들지 않는다.**
   `save()` 가 설정 지문(embedding backend/model/normalize, chunk 파라미터, 차원)을
   `meta.json` 으로 함께 남기고, `load()` 는 지문이 다르면 `False` 를 반환한다.
   설정만 바꾸고 옛 index 를 그대로 재사용해 조용히 틀린 검색을 하는 사고를 막는다.

3. **외부 의존 최소화.**
   `langchain-community` 는 설치돼 있지 않을 수 있으므로 LangChain FAISS 래퍼에
   의존하지 않고 faiss + numpy 만으로 직접 구현했다. faiss/numpy 는 지연 import 한다.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence

from ..config import Settings, Species, get_settings
from ..schemas import RetrievedEvidence

if TYPE_CHECKING:  # 순환/부재 방지 — chunker 는 런타임에 import 하지 않는다.
    from .chunker import Chunk

logger = logging.getLogger(__name__)

#: 저장 포맷 버전. 구조가 바뀌면 올려서 옛 index 를 무효화한다.
INDEX_FORMAT_VERSION: int = 1

SUPPORTED_SPECIES: tuple[str, ...] = ("dog", "cat")

_INDEX_FILENAME = "index.faiss"
_RECORDS_FILENAME = "records.json"
_META_FILENAME = "meta.json"


# ---------------------------------------------------------------------------
# 지연 import 헬퍼
# ---------------------------------------------------------------------------
def _import_faiss() -> Any:
    """faiss 를 지연 import 한다 — 미설치 시 한국어로 안내한다."""
    try:
        import faiss  # noqa: PLC0415

        return faiss
    except ImportError as exc:
        raise ImportError(
            "FAISS 가 설치돼 있지 않습니다. `pip install faiss-cpu` (GPU 환경은 faiss-gpu) "
            f"로 설치한 뒤 다시 실행하세요. (원인: {exc})"
        ) from exc


def _import_numpy() -> Any:
    """numpy 를 지연 import 한다 — faiss 와 짝이므로 오류 메시지도 함께 안내한다."""
    try:
        import numpy  # noqa: PLC0415

        return numpy
    except ImportError as exc:
        raise ImportError(
            f"numpy 가 설치돼 있지 않습니다. `pip install numpy` 로 설치하세요. (원인: {exc})"
        ) from exc


# ---------------------------------------------------------------------------
# chunk 접근 헬퍼 — chunker.Chunk 가 아직 없거나 형태가 달라도 견디게 한다.
# ---------------------------------------------------------------------------
def _chunk_field(chunk: Any, name: str, default: Any = None) -> Any:
    """pydantic 모델·dataclass·dict 어느 형태의 chunk 에서도 필드를 꺼낸다.

    chunker.py 는 다른 모듈에서 작성되고 테스트는 동일 필드의 임시 객체를 쓸 수 있어,
    속성 접근을 가정하지 않고 dict 접근까지 함께 시도한다.
    """
    if isinstance(chunk, dict):
        return chunk.get(name, default)
    value = getattr(chunk, name, None)
    return default if value is None else value


def _chunk_to_record(chunk: Any) -> dict[str, Any]:
    """chunk 를 저장/검색용 record dict 로 변환한다."""
    metadata = _chunk_field(chunk, "metadata", {}) or {}
    if not isinstance(metadata, dict):
        metadata = dict(metadata)
    chunk_id = str(_chunk_field(chunk, "chunk_id", "") or metadata.get("chunk_id", ""))
    document_id = str(
        _chunk_field(chunk, "document_id", "") or metadata.get("document_id", "")
    )
    return {
        "chunk_id": chunk_id,
        "document_id": document_id,
        "text": str(_chunk_field(chunk, "text", "") or ""),
        "metadata": metadata,
    }


def _record_species(record: dict[str, Any]) -> str | None:
    """record 의 species 를 읽는다 — metadata 에만 존재한다."""
    value = (record.get("metadata") or {}).get("species")
    return str(value) if value else None


# ---------------------------------------------------------------------------
# faiss index 파일 입출력
# ---------------------------------------------------------------------------
# faiss 의 write_index/read_index 는 C 런타임 `fopen` 에 경로를 **ANSI 코드페이지**로
# 넘긴다. 그래서 경로에 한글이 하나라도 있으면(예: 한국어 Windows 사용자 폴더,
# 한글이 포함된 프로젝트 디렉터리) "Illegal byte sequence" 로 저장/로드가 통째로
# 실패한다. 직렬화 자체는 메모리에서 하고 파일 입출력만 파이썬으로 처리하면
# 경로 인코딩 문제가 사라진다. 바이트 포맷은 write_index 와 동일하므로 기존에
# 저장된 index 와도 호환된다.
def _write_index_file(faiss: Any, index: Any, path: Path) -> None:
    """faiss index 를 파일로 저장한다 — 경로 인코딩에 의존하지 않는다."""
    path.write_bytes(faiss.serialize_index(index).tobytes())


def _read_index_file(faiss: Any, np: Any, path: Path) -> Any:
    """저장된 faiss index 를 읽어온다 — 경로 인코딩에 의존하지 않는다."""
    return faiss.deserialize_index(np.frombuffer(path.read_bytes(), dtype="uint8"))


# ---------------------------------------------------------------------------
# species 1개분 index
# ---------------------------------------------------------------------------
class _SpeciesIndex:
    """species 하나의 FAISS index + record 목록 + 정규화 벡터 행렬 묶음."""

    def __init__(self, species: str, index: Any, records: list[dict[str, Any]], vectors: Any) -> None:
        self.species = species
        self.index = index
        self.records = records
        self.vectors = vectors  # numpy float32 (n, dim), L2 정규화 완료

    @property
    def size(self) -> int:
        return len(self.records)

    @property
    def dim(self) -> int:
        return int(self.vectors.shape[1]) if self.size else 0


# ---------------------------------------------------------------------------
# 본체
# ---------------------------------------------------------------------------
class VeterinaryVectorStore:
    """Cornell 수의학 문서 검색용 vector store.

    embeddings 를 주입하면(테스트는 `DeterministicEmbeddings`) 모델 다운로드 없이
    build→save→load→search 왕복을 그대로 검증할 수 있다.
    """

    def __init__(self, settings: Settings | None = None, embeddings: Any = None) -> None:
        self.settings: Settings = settings or get_settings()
        self._embeddings: Any = embeddings
        self._indexes: dict[str, _SpeciesIndex] = {}

    # -- 임베딩 ------------------------------------------------------------
    @property
    def embeddings(self) -> Any:
        """임베딩 객체 — 주입되지 않았으면 설정대로 지연 생성한다.

        생성 시점을 늦춰야 `VeterinaryVectorStore()` 만 만들어 두고 쓰지 않는 경우에
        불필요한 모델 로드가 일어나지 않는다.
        """
        if self._embeddings is None:
            from .embeddings import build_embeddings  # noqa: PLC0415

            self._embeddings = build_embeddings(self.settings.rag)
        return self._embeddings

    @property
    def loaded_species(self) -> set[str]:
        """현재 메모리에 올라온(=검색 가능한) species 집합."""
        return set(self._indexes.keys())

    # -- 내부: 벡터 유틸 ---------------------------------------------------
    def _normalize(self, matrix: Any) -> Any:
        """행 단위 L2 정규화 — 내적을 코사인 유사도와 같게 만든다.

        임베딩 백엔드가 정규화를 껐더라도 index 안에서는 항상 정규화된 벡터를 쓴다.
        그래야 IndexFlatIP 의 내적 점수를 코사인으로 해석하는 계약이 깨지지 않는다.
        """
        np = _import_numpy()
        matrix = np.asarray(matrix, dtype="float32")
        if matrix.ndim == 1:
            matrix = matrix.reshape(1, -1)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        return (matrix / norms).astype("float32")

    def _embed_texts(self, texts: Sequence[str]) -> Any:
        """문서 배치를 임베딩해 정규화 행렬로 만든다."""
        vectors = self.embeddings.embed_documents(list(texts))
        return self._normalize(vectors)

    def _embed_query(self, query: str) -> Any:
        """질의를 임베딩해 (1, dim) 정규화 행렬로 만든다."""
        return self._normalize(self.embeddings.embed_query(query))

    # -- 내부: 설정 지문 ---------------------------------------------------
    def _active_embedding_model(self) -> str:
        """지금 선택된 backend 가 **실제로 쓰는** 모델 이름.

        backend 별로 모델 설정이 따로 있다 — huggingface 는 `embedding_model`,
        openai 는 `embedding_openai_model`(embeddings.py 의 build_embeddings 참고).
        지문에는 반드시 이쪽을 넣어야 한다. 예전에는 backend 와 무관하게
        `embedding_model` 만 기록했는데, 그러면 운영 경로(openai)에서
        `embedding_openai_model` 을 text-embedding-3-small → 3-large 로 바꿔도
        지문이 그대로여서 **벡터 공간이 다른 index 가 조용히 로드된다.**
        차원(1024)은 dimensions 파라미터로 고정되므로 차원 검사에도 걸리지 않는다.
        검색 품질만 소리 없이 나빠져 원인을 찾기 어렵다.
        """
        rag = self.settings.rag
        if rag.embedding_backend == "openai":
            return str(getattr(rag, "embedding_openai_model", "") or "")
        return str(rag.embedding_model)

    def _fingerprint(self, dim: int) -> dict[str, Any]:
        """index 재사용 가능 여부를 판정할 설정 지문.

        임베딩이 바뀌면 벡터 공간이 달라지고, chunk 파라미터가 바뀌면 record 경계가
        달라진다. 둘 중 하나만 바뀌어도 옛 index 는 무효이므로 모두 지문에 넣는다.
        """
        rag = self.settings.rag
        return {
            "format_version": INDEX_FORMAT_VERSION,
            "embedding_backend": rag.embedding_backend,
            "embedding_model": self._active_embedding_model(),
            "embedding_normalize": bool(rag.embedding_normalize),
            "embeddings_class": type(self.embeddings).__name__,
            "chunk_size": rag.chunk_size,
            "chunk_overlap": rag.chunk_overlap,
            "min_chunk_length": rag.min_chunk_length,
            "dimension": int(dim),
        }

    def _species_dir(self, directory: str | Path | None, species: str) -> Path:
        """species index 디렉터리 경로 — 설정의 `index_path()` 규칙을 그대로 따른다."""
        if directory is None:
            return Path(self.settings.index_path(species))  # type: ignore[arg-type]
        return Path(directory) / f"faiss_{species}"

    # -- build -------------------------------------------------------------
    def build(self, chunks: list["Chunk"], species: Species) -> None:
        """해당 species 의 index 를 새로 만든다.

        넘어온 chunk 중 `metadata["species"]` 가 일치하는 것만 담는다.
        호출자가 실수로 전체 chunk 를 넘겨도 dog/cat 이 섞이지 않게 하기 위한
        2차 방어선이다(species 미표기 chunk 는 제외하고 경고를 남긴다).
        """
        if species not in SUPPORTED_SPECIES:
            raise ValueError(
                f"지원하지 않는 species 입니다: {species!r}. {SUPPORTED_SPECIES} 중 하나여야 합니다."
            )

        records: list[dict[str, Any]] = []
        skipped = 0
        for chunk in chunks:
            record = _chunk_to_record(chunk)
            if _record_species(record) != species:
                skipped += 1
                continue
            if not record["text"].strip():
                skipped += 1
                continue
            records.append(record)

        if skipped:
            logger.info(
                "[%s] chunk %d건은 species 불일치 또는 빈 텍스트라 제외했습니다.",
                species,
                skipped,
            )
        if not records:
            raise ValueError(
                f"species='{species}' 에 해당하는 chunk 가 없어 index 를 만들 수 없습니다. "
                "chunk metadata 의 'species' 값을 확인하세요."
            )

        faiss = _import_faiss()
        vectors = self._embed_texts([record["text"] for record in records])
        index = faiss.IndexFlatIP(int(vectors.shape[1]))
        index.add(vectors)

        self._indexes[species] = _SpeciesIndex(species, index, records, vectors)
        logger.info("[%s] index 생성 완료 — chunk %d건, 차원 %d", species, len(records), vectors.shape[1])

    def build_all(self, chunks: list["Chunk"]) -> dict[str, int]:
        """dog/cat index 를 한 번에 만든다 — {species: chunk 수} 를 돌려준다.

        chunk 가 하나도 없는 species 는 조용히 건너뛴다(경고만). 예를 들어 dog 문서만
        로드한 상태에서도 파이프라인이 멈추지 않아야 하기 때문이다.
        """
        buckets: dict[str, list[Any]] = {name: [] for name in SUPPORTED_SPECIES}
        unknown = 0
        for chunk in chunks:
            record = _chunk_to_record(chunk)
            species = _record_species(record)
            if species in buckets:
                buckets[species].append(chunk)
            else:
                unknown += 1

        if unknown:
            logger.warning("species 를 알 수 없는 chunk %d건을 제외했습니다.", unknown)

        built: dict[str, int] = {}
        for species, species_chunks in buckets.items():
            if not species_chunks:
                logger.warning("species='%s' chunk 가 없어 index 를 건너뜁니다.", species)
                continue
            self.build(species_chunks, species)  # type: ignore[arg-type]
            built[species] = self._indexes[species].size
        return built

    # -- save / load -------------------------------------------------------
    def save(self, directory: str | Path | None = None) -> None:
        """메모리에 있는 모든 species index 를 디스크에 저장한다.

        species 별로 `index.faiss` / `records.json` / `meta.json` 3개를 남긴다.
        meta.json 의 지문 덕분에 다음 실행에서 재생성이 필요한지 즉시 판정할 수 있다.
        """
        if not self._indexes:
            logger.warning("저장할 index 가 없습니다. build() 를 먼저 호출하세요.")
            return

        faiss = _import_faiss()
        for species, bundle in self._indexes.items():
            target = self._species_dir(directory, species)
            target.mkdir(parents=True, exist_ok=True)

            _write_index_file(faiss, bundle.index, target / _INDEX_FILENAME)
            (target / _RECORDS_FILENAME).write_text(
                json.dumps(bundle.records, ensure_ascii=False),
                encoding="utf-8",
            )
            meta = {
                "species": species,
                "count": bundle.size,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "fingerprint": self._fingerprint(bundle.dim),
            }
            (target / _META_FILENAME).write_text(
                json.dumps(meta, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            logger.info("[%s] index 저장 완료 — %s", species, target)

    def load(self, directory: str | Path | None = None) -> bool:
        """저장된 index 를 읽어온다 — 하나라도 성공하면 True, 아니면 False.

        다음 경우 해당 species 를 건너뛴다(= 재생성이 필요하다는 신호).
          - 디렉터리/파일이 없다.
          - `meta.json` 의 지문이 현재 설정과 다르다(모델·정규화·chunk 파라미터 변경).
          - 저장된 벡터 수와 record 수가 어긋난다(손상).

        전부 실패하면 False 를 돌려주므로 호출자는 `if not store.load(): store.build_all(...)`
        한 줄로 재생성 여부를 처리할 수 있다.
        """
        faiss = _import_faiss()
        np = _import_numpy()
        loaded: dict[str, _SpeciesIndex] = {}

        for species in SUPPORTED_SPECIES:
            target = self._species_dir(directory, species)
            index_file = target / _INDEX_FILENAME
            records_file = target / _RECORDS_FILENAME
            meta_file = target / _META_FILENAME
            if not (index_file.exists() and records_file.exists() and meta_file.exists()):
                logger.debug("[%s] 저장된 index 가 없습니다: %s", species, target)
                continue

            try:
                meta = json.loads(meta_file.read_text(encoding="utf-8"))
                records = json.loads(records_file.read_text(encoding="utf-8"))
                index = _read_index_file(faiss, np, index_file)
            except Exception as exc:
                logger.warning("[%s] index 읽기에 실패해 건너뜁니다: %s", species, exc)
                continue

            # 저장된 index 의 실제 차원과 현재 임베딩의 차원을 먼저 맞춰본다.
            # 지문의 embeddings_class 만으로는 "같은 클래스, 다른 차원"(예: dim 을 바꿔
            # 주입한 DeterministicEmbeddings)을 걸러내지 못해 faiss 검색에서 뒤늦게
            # 터진다. 여기서 막아야 load()=False → 재생성이라는 정상 경로로 흡수된다.
            from .embeddings import embedding_dimension  # noqa: PLC0415

            actual_dim = embedding_dimension(self.embeddings)
            if int(index.d) != int(actual_dim):
                logger.warning(
                    "[%s] 저장된 index 차원(%d)과 현재 임베딩 차원(%d)이 달라 재생성이 필요합니다.",
                    species,
                    index.d,
                    actual_dim,
                )
                continue

            saved_fp = meta.get("fingerprint", {})
            current_fp = self._fingerprint(int(actual_dim))
            if saved_fp != current_fp:
                differences = [
                    key
                    for key in set(saved_fp) | set(current_fp)
                    if saved_fp.get(key) != current_fp.get(key)
                ]
                logger.warning(
                    "[%s] 설정 지문이 달라 저장된 index 를 쓰지 않습니다(재생성 필요). 변경 항목: %s",
                    species,
                    ", ".join(sorted(differences)) or "알 수 없음",
                )
                continue

            if index.ntotal != len(records):
                logger.warning(
                    "[%s] index 벡터 수(%d)와 record 수(%d)가 달라 손상으로 판단합니다.",
                    species,
                    index.ntotal,
                    len(records),
                )
                continue

            vectors = (
                np.asarray(index.reconstruct_n(0, index.ntotal), dtype="float32")
                if index.ntotal
                else np.zeros((0, index.d), dtype="float32")
            )
            loaded[species] = _SpeciesIndex(species, index, records, vectors)
            logger.info("[%s] index 로드 완료 — chunk %d건", species, len(records))

        if not loaded:
            return False
        self._indexes.update(loaded)
        return True

    # -- search ------------------------------------------------------------
    def _mmr_select(self, query_vec: Any, candidate_vecs: Any, k: int) -> list[int]:
        """MMR 로 candidate 중 k 개를 고른다 — 관련성과 다양성을 함께 본다.

        점수 = λ·sim(query, doc) − (1−λ)·max sim(doc, 이미 선택된 doc).
        같은 문서의 인접 chunk 가 상위를 독점해 근거가 한쪽으로 쏠리는 것을 막는다.
        모든 벡터가 정규화돼 있으므로 내적이 곧 코사인 유사도다.
        """
        np = _import_numpy()
        lambda_mult = float(self.settings.rag.mmr_lambda)
        query_sims = candidate_vecs @ query_vec.reshape(-1)
        pairwise = candidate_vecs @ candidate_vecs.T

        selected: list[int] = []
        remaining = list(range(candidate_vecs.shape[0]))
        while remaining and len(selected) < k:
            if not selected:
                best = int(max(remaining, key=lambda i: float(query_sims[i])))
            else:
                best = int(
                    max(
                        remaining,
                        key=lambda i: lambda_mult * float(query_sims[i])
                        - (1.0 - lambda_mult) * float(np.max(pairwise[i, selected])),
                    )
                )
            selected.append(best)
            remaining.remove(best)
        return selected

    def search(
        self,
        query: str,
        species: Species,
        k: int = 6,
        fetch_k: int = 20,
    ) -> list[RetrievedEvidence]:
        """species index 안에서만 검색해 `RetrievedEvidence` 목록을 돌려준다.

        `RagSettings.use_mmr` 가 True 면 먼저 fetch_k 개를 뽑은 뒤 MMR 로 k 개를 고르고,
        False 면 상위 k 개를 그대로 쓴다.

        **score 의 의미**: 질의 벡터와 chunk 벡터의 **코사인 유사도를 0~1 로 클램프한 값**
        (`max(0.0, min(1.0, cos))`)이다. 두 벡터 모두 L2 정규화돼 있어 FAISS
        `IndexFlatIP` 의 내적이 곧 코사인이고, 음수(=무관/반대 방향)는 0 으로 잘라
        `RagSettings.min_relevance_score` 같은 임계값과 직접 비교할 수 있게 했다.
        1.0 에 가까울수록 유사, 0 이면 무관하다. 거리(distance)가 아니라 유사도다.

        index 가 없는 species 로 호출하면 예외 대신 빈 리스트를 돌려준다.
        검색 실패는 파이프라인상 '근거 부족 → 웹 fallback' 으로 흡수되는 정상 경로이기 때문이다.
        """
        bundle = self._indexes.get(species)
        if bundle is None:
            logger.warning(
                "species='%s' index 가 없어 빈 결과를 돌려줍니다. build()/load() 를 먼저 호출하세요. "
                "(현재 로드된 species: %s)",
                species,
                sorted(self.loaded_species) or "없음",
            )
            return []
        if not query or not query.strip():
            logger.warning("빈 query 로 검색이 호출됐습니다 — 빈 결과를 돌려줍니다.")
            return []

        k = max(1, int(k))
        use_mmr = bool(self.settings.rag.use_mmr)
        pool = max(k, int(fetch_k)) if use_mmr else k
        pool = min(pool, bundle.size)

        query_vec = self._embed_query(query)
        scores, indices = bundle.index.search(query_vec, pool)
        raw_scores = [float(value) for value in scores[0]]
        candidate_ids = [int(value) for value in indices[0] if int(value) >= 0]
        raw_scores = raw_scores[: len(candidate_ids)]
        if not candidate_ids:
            return []

        if use_mmr and len(candidate_ids) > k:
            np = _import_numpy()
            candidate_vecs = bundle.vectors[np.asarray(candidate_ids, dtype="int64")]
            chosen = self._mmr_select(query_vec, candidate_vecs, k)
            ordered = [(candidate_ids[i], raw_scores[i]) for i in chosen]
        else:
            ordered = list(zip(candidate_ids, raw_scores))[:k]

        return [
            self._to_evidence(bundle.records[record_id], score, species)
            for record_id, score in ordered
        ]

    # -- 매핑 --------------------------------------------------------------
    def _to_evidence(
        self, record: dict[str, Any], raw_score: float, species: Species
    ) -> RetrievedEvidence:
        """record + 점수를 `RetrievedEvidence` 로 매핑한다.

        metadata 키가 비어 있어도 스키마 필수 필드가 깨지지 않도록 기본값을 채운다.
        species 는 index 의 species 를 신뢰한다 — metadata 가 오염돼 있어도 검색된
        index 가 곧 species 이므로 그쪽이 항상 정확하다.
        """
        metadata = record.get("metadata") or {}
        categories = metadata.get("categories") or []
        heading_path = metadata.get("heading_path") or []
        return RetrievedEvidence(
            chunk_id=str(record.get("chunk_id") or metadata.get("chunk_id") or ""),
            document_id=str(record.get("document_id") or metadata.get("document_id") or ""),
            title=str(metadata.get("title") or ""),
            text=str(record.get("text") or ""),
            species=species,
            source=str(metadata.get("source") or ""),
            source_url=str(metadata.get("source_url") or ""),
            categories=[str(item) for item in categories],
            score=max(0.0, min(1.0, float(raw_score))),
            heading_path=[str(item) for item in heading_path],
        )

    # -- 편의 --------------------------------------------------------------
    def index_size(self, species: Species) -> int:
        """해당 species index 의 chunk 수 — 없으면 0."""
        bundle = self._indexes.get(species)
        return bundle.size if bundle else 0

    def stats(self) -> dict[str, Any]:
        """진단용 요약 — 노트북에서 index 상태를 한 줄로 확인할 때 쓴다."""
        return {
            "loaded_species": sorted(self.loaded_species),
            "counts": {species: bundle.size for species, bundle in self._indexes.items()},
            "dimensions": {species: bundle.dim for species, bundle in self._indexes.items()},
            "use_mmr": bool(self.settings.rag.use_mmr),
            "mmr_lambda": float(self.settings.rag.mmr_lambda),
        }

    def __repr__(self) -> str:  # pragma: no cover - 디버깅 편의
        return f"VeterinaryVectorStore(loaded={sorted(self.loaded_species)})"
