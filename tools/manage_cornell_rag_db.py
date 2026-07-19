#!/usr/bin/env python3
"""Build and inspect a local ChromaDB index for the Cornell pet corpus.

OpenAI creates embeddings. ChromaDB only stores and searches the
explicit embeddings passed to it; Chroma's default embedding function is never
used. The module keeps third-party imports lazy so corpus validation and unit
tests can run before optional RAG dependencies are installed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence


MODEL = "text-embedding-3-small"
QUERY_REWRITE_MODEL = "gpt-5.4-mini"
DIMENSION = 1536
EXPECTED_CHUNKS = 732
DEFAULT_COLLECTION = "cornell_pet_health_text_embedding_3_small_1536"
DEFAULT_INPUT = Path("rag_data/chunks/cornell_pet_health_chunks.jsonl")
DEFAULT_DB_PATH = Path("rag_data/chroma")
DEFAULT_GOLD = Path("rag_data/evaluation/cornell_retrieval_gold.jsonl")
DEFAULT_RERANK_CANDIDATE_MULTIPLIER = 3
DEFAULT_MAX_RERANK_CANDIDATES = 20
HYBRID_DENSE_WEIGHT = 0.7
HYBRID_LEXICAL_WEIGHT = 0.3
TOKEN_RE = re.compile(r"[a-z0-9가-힣]+")
QUERY_REWRITE_INSTRUCTION = """You rewrite pet-health questions for retrieval.

Return one short English search query for Cornell veterinary health articles.
Do not answer the question.
Do not give medical advice.
Keep important animal species, disease, toxin, symptom, diagnosis, prevention, and emergency terms.
Return plain text only.
"""
QUERY_REWRITE_CACHE: dict[tuple[str, str], str] = {}
REQUIRED_FIELDS = {
    "chunk_id",
    "document_id",
    "title",
    "section_path",
    "species",
    "categories",
    "canonical_url",
    "last_updated",
    "source_institution",
    "source_center",
    "language",
    "medical_domain",
    "content_hash",
    "content",
}
RECORD_METADATA_FIELDS = (
    "document_id",
    "title",
    "section_path",
    "species",
    "categories",
    "canonical_url",
    "last_updated",
    "source_institution",
    "source_center",
    "language",
    "medical_domain",
    "content_hash",
)


class RagDbError(RuntimeError):
    """A user-actionable corpus, API, or database error."""


@dataclass(frozen=True)
class Corpus:
    rows: tuple[dict[str, Any], ...]
    sha256: str


@dataclass(frozen=True)
class SearchResult:
    rank: int
    distance: float
    chunk_id: str
    document: str
    metadata: dict[str, Any]

    @property
    def similarity(self) -> float:
        return 1.0 - self.distance


@dataclass(frozen=True)
class SearchRun:
    retrieval_query: str
    rewrite_failed: bool
    results: list[SearchResult]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise RagDbError(f"파일을 찾을 수 없습니다: {path}")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, raw_line in enumerate(stream, start=1):
            if not raw_line.strip():
                raise RagDbError(f"{path}의 {line_number}번째 줄이 비어 있습니다.")
            try:
                value = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise RagDbError(
                    f"{path}의 {line_number}번째 줄은 올바른 JSON이 아닙니다: {exc}"
                ) from exc
            if not isinstance(value, dict):
                raise RagDbError(f"{path}의 {line_number}번째 줄은 JSON 객체여야 합니다.")
            rows.append(value)
    return rows


def validate_corpus_rows(
    rows: Sequence[dict[str, Any]], expected_count: int | None = EXPECTED_CHUNKS
) -> None:
    if expected_count is not None and len(rows) != expected_count:
        raise RagDbError(
            f"청크 수가 예상과 다릅니다. 예상 {expected_count}개, 실제 {len(rows)}개입니다."
        )
    seen: set[str] = set()
    for index, row in enumerate(rows, start=1):
        missing = sorted(REQUIRED_FIELDS - row.keys())
        if missing:
            raise RagDbError(f"{index}번째 청크에 필수 필드가 없습니다: {', '.join(missing)}")
        chunk_id = row["chunk_id"]
        if not isinstance(chunk_id, str) or not chunk_id.strip():
            raise RagDbError(f"{index}번째 청크의 chunk_id가 비어 있습니다.")
        if chunk_id in seen:
            raise RagDbError(f"중복 chunk_id가 있습니다: {chunk_id}")
        seen.add(chunk_id)
        if not isinstance(row["content"], str) or not row["content"].strip():
            raise RagDbError(f"{chunk_id}의 content가 비어 있습니다.")
        species = row["species"]
        if not isinstance(species, list) or len(species) != 1 or species[0] not in {"dog", "cat"}:
            raise RagDbError(f"{chunk_id}의 species는 ['dog'] 또는 ['cat']이어야 합니다.")
        for field in ("section_path", "categories"):
            value = row[field]
            if not isinstance(value, list) or not value or not all(
                isinstance(item, str) and item for item in value
            ):
                raise RagDbError(f"{chunk_id}의 {field}는 비어 있지 않은 문자열 배열이어야 합니다.")
        url = row["canonical_url"]
        if not isinstance(url, str) or not url.startswith("https://www.vet.cornell.edu/"):
            raise RagDbError(f"{chunk_id}의 Cornell canonical_url이 올바르지 않습니다.")
        for field in (
            "document_id",
            "title",
            "source_institution",
            "source_center",
            "language",
            "medical_domain",
            "content_hash",
        ):
            if not isinstance(row[field], str) or not row[field].strip():
                raise RagDbError(f"{chunk_id}의 {field}가 비어 있습니다.")
        if row["last_updated"] is not None and not isinstance(row["last_updated"], str):
            raise RagDbError(f"{chunk_id}의 last_updated는 문자열 또는 null이어야 합니다.")


def read_corpus(path: Path, expected_count: int | None = EXPECTED_CHUNKS) -> Corpus:
    rows = load_jsonl(path)
    validate_corpus_rows(rows, expected_count)
    return Corpus(tuple(rows), sha256_file(path))


def document_embedding_text(row: dict[str, Any]) -> str:
    return f"title: {row['title']} | text: {row['content']}"


def query_embedding_text(query: str) -> str:
    query = query.strip()
    if not query:
        raise RagDbError("검색 질문은 비어 있을 수 없습니다.")
    return f"task: question answering | query: {query}"


def chroma_metadata(row: dict[str, Any]) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    for field in RECORD_METADATA_FIELDS:
        value = row[field]
        metadata[field] = "" if value is None else value
    metadata["embedding_model"] = MODEL
    metadata["embedding_dimension"] = DIMENSION
    return metadata


def validate_embeddings(embeddings: Sequence[Sequence[float]], expected: int) -> None:
    if len(embeddings) != expected:
        raise RagDbError(
            f"OpenAI가 {expected}개 대신 {len(embeddings)}개의 임베딩을 반환했습니다."
        )
    for position, embedding in enumerate(embeddings, start=1):
        if len(embedding) != DIMENSION:
            raise RagDbError(
                f"{position}번째 임베딩 차원이 {len(embedding)}입니다. 예상값은 {DIMENSION}입니다."
            )
        if not all(isinstance(value, (int, float)) and math.isfinite(value) for value in embedding):
            raise RagDbError(f"{position}번째 임베딩에 유효하지 않은 숫자가 있습니다.")


def _status_code(exc: BaseException) -> int | None:
    for candidate in (getattr(exc, "status_code", None), getattr(exc, "code", None)):
        if callable(candidate):
            try:
                candidate = candidate()
            except TypeError:
                continue
        if isinstance(candidate, int):
            return candidate
        value = getattr(candidate, "value", None)
        if isinstance(value, int):
            return value
    text = str(exc)
    for code in (429, 500, 502, 503, 504):
        if str(code) in text:
            return code
    return None


def is_retryable(exc: BaseException) -> bool:
    return _status_code(exc) in {429, 500, 502, 503, 504}


def retry_after_seconds(exc: BaseException) -> float | None:
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if headers:
        value = headers.get("retry-after") or headers.get("Retry-After")
        try:
            return max(0.0, float(value))
        except (TypeError, ValueError):
            return None
    return None


def call_with_retry(
    operation: Callable[[], Any],
    *,
    max_retries: int = 5,
    sleep: Callable[[float], None] = time.sleep,
    random_value: Callable[[], float] = random.random,
    on_retry: Callable[[int, float, BaseException], None] | None = None,
) -> Any:
    retries = 0
    while True:
        try:
            return operation()
        except Exception as exc:
            if not is_retryable(exc) or retries >= max_retries:
                raise
            retries += 1
            server_delay = retry_after_seconds(exc)
            delay = server_delay if server_delay is not None else min(30.0, 2 ** (retries - 1))
            delay += random_value() * 0.25
            if on_retry:
                on_retry(retries, delay, exc)
            sleep(delay)


def openai_client() -> Any:
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RagDbError(
            "OPENAI_API_KEY가 설정되지 않았습니다. PowerShell에서 "
            "$env:OPENAI_API_KEY=\"발급받은_키\"를 실행하세요."
        )
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RagDbError(
            "openai 패키지가 설치되지 않았습니다. "
            "python -m pip install -r requirements-rag.txt 를 실행하세요."
        ) from exc
    return OpenAI(api_key=api_key)


def embed_texts(client: Any, texts: Sequence[str]) -> list[list[float]]:
    if not texts:
        return []

    def request() -> Any:
        return client.embeddings.create(
            model=MODEL,
            input=list(texts),
            dimensions=DIMENSION,
        )

    try:
        response = call_with_retry(
            request,
            on_retry=lambda attempt, delay, exc: print(
                f"  일시적인 API 오류({_status_code(exc)})로 {delay:.1f}초 뒤 "
                f"재시도합니다 ({attempt}/5)."
            ),
        )
    except Exception as exc:
        raise RagDbError(f"OpenAI 임베딩 요청에 실패했습니다: {exc}") from exc
    objects = getattr(response, "data", None)
    if objects is None:
        raise RagDbError("OpenAI 응답에 data 필드가 없습니다.")
    embeddings = [list(item.embedding) for item in objects]
    validate_embeddings(embeddings, len(texts))
    return embeddings


def _response_text(response: Any) -> str:
    text = getattr(response, "output_text", None)
    if isinstance(text, str) and text.strip():
        return text
    text = getattr(response, "text", None)
    if isinstance(text, str) and text.strip():
        return text
    choices = getattr(response, "choices", None) or []
    if choices:
        message = getattr(choices[0], "message", None)
        content = getattr(message, "content", None)
        if isinstance(content, str) and content.strip():
            return content
    raise RagDbError("OpenAI query rewrite가 빈 응답을 반환했습니다.")


def sanitize_retrieval_query(text: str) -> str:
    query = text.strip()
    query = re.sub(r"^```(?:text)?\s*", "", query, flags=re.IGNORECASE)
    query = re.sub(r"\s*```$", "", query)
    query = " ".join(query.split())
    query = re.sub(r"^(retrieval query|query|english query)\s*:\s*", "", query, flags=re.IGNORECASE)
    query = query.strip(" \"'`")
    if not query:
        raise RagDbError("OpenAI query rewrite가 빈 검색 질의를 반환했습니다.")
    return query[:240]


def rewrite_retrieval_query(client: Any, question: str, species: str) -> str:
    question = question.strip()
    if not question:
        raise RagDbError("검색 질의로 바꿀 질문이 비어 있습니다.")
    if species not in {"dog", "cat"}:
        raise RagDbError("species는 dog 또는 cat이어야 합니다.")
    cache_key = (species, question)
    if cache_key in QUERY_REWRITE_CACHE:
        return QUERY_REWRITE_CACHE[cache_key]

    def request() -> Any:
        return client.responses.create(
            model=QUERY_REWRITE_MODEL,
            instructions=QUERY_REWRITE_INSTRUCTION,
            input=(
                f"Animal species: {species}\n"
                f"User question in Korean or English: {question}\n\n"
                "Rewrite as one English retrieval query."
            ),
            max_output_tokens=80,
        )

    response = call_with_retry(request)
    retrieval_query = sanitize_retrieval_query(_response_text(response))
    QUERY_REWRITE_CACHE[cache_key] = retrieval_query
    return retrieval_query


def chroma_client(db_path: Path) -> Any:
    try:
        import chromadb
    except ImportError as exc:
        raise RagDbError(
            "chromadb가 설치되지 않았습니다. "
            "python -m pip install -r requirements-rag.txt 를 실행하세요."
        ) from exc
    db_path.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(db_path))


def collection_names(client: Any) -> set[str]:
    names: set[str] = set()
    for item in client.list_collections():
        names.add(item if isinstance(item, str) else item.name)
    return names


def expected_collection_metadata(corpus: Corpus) -> dict[str, Any]:
    return {
        "embedding_model": MODEL,
        "embedding_dimension": DIMENSION,
        "corpus_sha256": corpus.sha256,
        "expected_chunks": len(corpus.rows),
    }


def validate_collection_compatibility(
    actual: dict[str, Any] | None, expected: dict[str, Any]
) -> None:
    actual = actual or {}
    mismatches = []
    for key in ("embedding_model", "embedding_dimension", "corpus_sha256", "expected_chunks"):
        if actual.get(key) != expected[key]:
            mismatches.append(f"{key}: DB={actual.get(key)!r}, 입력={expected[key]!r}")
    if mismatches:
        raise RagDbError(
            "기존 컬렉션과 현재 입력 설정이 다릅니다. --rebuild가 필요합니다. "
            + "; ".join(mismatches)
        )


def get_or_create_collection(
    client: Any, name: str, corpus: Corpus, rebuild: bool = False
) -> Any:
    names = collection_names(client)
    if rebuild and name in names:
        client.delete_collection(name)
        names.remove(name)
    expected = expected_collection_metadata(corpus)
    if name in names:
        collection = client.get_collection(name)
        validate_collection_compatibility(collection.metadata, expected)
        return collection
    return client.create_collection(
        name=name,
        metadata=expected,
        configuration={"hnsw": {"space": "cosine"}},
    )


def existing_hashes(collection: Any) -> dict[str, str]:
    result = collection.get(include=["metadatas"])
    ids = result.get("ids") or []
    metadatas = result.get("metadatas") or []
    return {
        chunk_id: metadata.get("content_hash", "")
        for chunk_id, metadata in zip(ids, metadatas)
    }


def pending_rows(
    rows: Sequence[dict[str, Any]], known_hashes: dict[str, str]
) -> list[dict[str, Any]]:
    return [
        row
        for row in rows
        if known_hashes.get(row["chunk_id"]) != row["content_hash"]
    ]


def batched(values: Sequence[Any], size: int) -> Iterable[Sequence[Any]]:
    if size < 1:
        raise RagDbError("batch-size는 1 이상이어야 합니다.")
    for start in range(0, len(values), size):
        yield values[start : start + size]


def run_check(args: argparse.Namespace) -> None:
    corpus = read_corpus(args.input, args.expected_count)
    print(f"[1/4] JSONL 검사 완료: {len(corpus.rows)}개 청크")
    print(f"[2/4] 입력 SHA-256: {corpus.sha256}")
    args.db_path.mkdir(parents=True, exist_ok=True)
    probe = args.db_path / ".write-test"
    try:
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
    except OSError as exc:
        raise RagDbError(f"DB 경로에 쓸 수 없습니다: {args.db_path}: {exc}") from exc
    print(f"[3/4] DB 경로 쓰기 가능: {args.db_path}")
    client = openai_client()
    embedding = embed_texts(client, [query_embedding_text("반려동물 건강 정보")])[0]
    print(f"[4/4] OpenAI API 연결 성공: {MODEL}, {len(embedding)}차원")
    print("사전 검사가 모두 끝났습니다. 이제 index 명령을 실행할 수 있습니다.")


def run_index(args: argparse.Namespace) -> None:
    corpus = read_corpus(args.input, args.expected_count)
    client = chroma_client(args.db_path)
    collection = get_or_create_collection(client, args.collection, corpus, args.rebuild)
    known = existing_hashes(collection)
    todo = pending_rows(corpus.rows, known)
    print(f"입력 {len(corpus.rows)}개 / 이미 완료 {len(corpus.rows) - len(todo)}개 / 처리할 청크 {len(todo)}개")
    if todo:
        openai = openai_client()
        completed = len(corpus.rows) - len(todo)
        for batch in batched(todo, args.batch_size):
            embeddings = embed_texts(
                openai, [document_embedding_text(row) for row in batch]
            )
            collection.upsert(
                ids=[row["chunk_id"] for row in batch],
                embeddings=embeddings,
                documents=[row["content"] for row in batch],
                metadatas=[chroma_metadata(row) for row in batch],
            )
            completed += len(batch)
            print(f"  저장 완료: {completed}/{len(corpus.rows)}")
    count = collection.count()
    if count != len(corpus.rows):
        raise RagDbError(
            f"색인 완료 후 DB 수가 다릅니다. 입력 {len(corpus.rows)}개, DB {count}개입니다."
        )
    print(f"색인 완료: {count}개 청크가 {args.collection} 컬렉션에 있습니다.")


def require_collection(client: Any, name: str) -> Any:
    if name not in collection_names(client):
        raise RagDbError(f"컬렉션이 없습니다: {name}. 먼저 index 명령을 실행하세요.")
    return client.get_collection(name)


def run_inspect(args: argparse.Namespace) -> None:
    collection = require_collection(chroma_client(args.db_path), args.collection)
    result = collection.get(include=["documents", "metadatas", "embeddings"])
    ids = result.get("ids") or []
    documents = result.get("documents") or []
    metadatas = result.get("metadatas") or []
    embeddings = result.get("embeddings")
    embeddings = [] if embeddings is None else embeddings
    document_ids = {m.get("document_id") for m in metadatas if m.get("document_id")}
    dog = sum("dog" in (m.get("species") or []) for m in metadatas)
    cat = sum("cat" in (m.get("species") or []) for m in metadatas)
    missing = sum(
        not all(m.get(field) not in (None, "") for field in ("document_id", "title", "canonical_url", "species"))
        for m in metadatas
    )
    wrong_dimension = sum(len(vector) != DIMENSION for vector in embeddings)
    print(f"전체 청크: {len(ids)}")
    print(f"고유 문서: {len(document_ids)}")
    print(f"종별 청크: dog={dog}, cat={cat}")
    print(f"컬렉션 설정: {collection.metadata}")
    print(f"필수 메타데이터 누락: {missing}")
    print(f"{DIMENSION}차원이 아닌 벡터: {wrong_dimension}")
    if ids:
        print("\n예시 검색 카드")
        print(f"  chunk_id: {ids[0]}")
        print(f"  title: {metadatas[0].get('title')}")
        print(f"  URL: {metadatas[0].get('canonical_url')}")
        preview = " ".join(documents[0].split())[:240]
        print(f"  content: {preview}...")


def query_collection(
    collection: Any,
    embedding: Sequence[float],
    species: str,
    top_k: int,
) -> list[SearchResult]:
    validate_embeddings([embedding], 1)
    if species not in {"dog", "cat"}:
        raise RagDbError("species는 dog 또는 cat이어야 합니다.")
    if top_k < 1:
        raise RagDbError("top-k는 1 이상이어야 합니다.")
    raw = collection.query(
        query_embeddings=[list(embedding)],
        n_results=top_k,
        where={"species": {"$contains": species}},
        include=["documents", "metadatas", "distances"],
    )
    ids = (raw.get("ids") or [[]])[0]
    documents = (raw.get("documents") or [[]])[0]
    metadatas = (raw.get("metadatas") or [[]])[0]
    distances = (raw.get("distances") or [[]])[0]
    return [
        SearchResult(rank, float(distance), chunk_id, document, metadata)
        for rank, (chunk_id, document, metadata, distance) in enumerate(
            zip(ids, documents, metadatas, distances), start=1
        )
    ]


def tokenize_for_rerank(text: str) -> list[str]:
    """Tokenize Korean/English text for the lightweight lexical reranker."""
    return TOKEN_RE.findall(text.lower())


def normalized_scores(values: Sequence[float]) -> list[float]:
    if not values:
        return []
    minimum = min(values)
    maximum = max(values)
    if math.isclose(minimum, maximum):
        return [1.0 for _ in values]
    return [(value - minimum) / (maximum - minimum) for value in values]


def dense_similarity_scores(results: Sequence[SearchResult]) -> list[float]:
    """Keep dense similarity magnitude so lexical scores can break close calls."""
    return [max(0.0, min(1.0, result.similarity)) for result in results]


def bm25_scores(query: str, results: Sequence[SearchResult]) -> list[float]:
    query_terms = tokenize_for_rerank(query)
    if not query_terms or not results:
        return [0.0 for _ in results]

    documents = [
        tokenize_for_rerank(
            f"{result.metadata.get('title', '')} {result.metadata.get('document_id', '')} {result.document}"
        )
        for result in results
    ]
    avgdl = sum(len(document) for document in documents) / max(len(documents), 1)
    k1 = 1.5
    b = 0.75
    scores: list[float] = []
    for document_terms in documents:
        score = 0.0
        doc_len = len(document_terms) or 1
        for term in query_terms:
            tf = document_terms.count(term)
            if tf == 0:
                continue
            containing = sum(1 for doc in documents if term in doc)
            idf = math.log(1 + (len(documents) - containing + 0.5) / (containing + 0.5))
            denominator = tf + k1 * (1 - b + b * doc_len / max(avgdl, 1e-9))
            score += idf * (tf * (k1 + 1)) / denominator
        scores.append(score)
    return scores


def hybrid_rerank_results(
    query: str,
    results: Sequence[SearchResult],
    top_k: int,
    *,
    dense_weight: float = HYBRID_DENSE_WEIGHT,
    lexical_weight: float = HYBRID_LEXICAL_WEIGHT,
) -> list[SearchResult]:
    if top_k < 1:
        raise RagDbError("top-k는 1 이상이어야 합니다.")
    dense = dense_similarity_scores(results)
    lexical = normalized_scores(bm25_scores(query, results))
    scored = [
        (dense_weight * dense_score + lexical_weight * lexical_score, index, result)
        for index, (result, dense_score, lexical_score) in enumerate(zip(results, dense, lexical))
    ]
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [replace(result, rank=rank) for rank, (_, _, result) in enumerate(scored[:top_k], start=1)]


def rerank_candidate_count(
    top_k: int,
    *,
    hybrid_rerank: bool,
    candidate_multiplier: int,
    max_candidates: int,
) -> int:
    if top_k < 1:
        raise RagDbError("top-k는 1 이상이어야 합니다.")
    if not hybrid_rerank:
        return top_k
    if candidate_multiplier < 1:
        raise RagDbError("--candidate-multiplier는 1 이상이어야 합니다.")
    if max_candidates < 1:
        raise RagDbError("--max-candidates는 1 이상이어야 합니다.")
    return min(max_candidates, max(top_k, top_k * candidate_multiplier))


def search(
    client: Any,
    collection: Any,
    query: str,
    species: str,
    top_k: int,
    *,
    hybrid_rerank: bool = True,
    candidate_multiplier: int = DEFAULT_RERANK_CANDIDATE_MULTIPLIER,
    max_candidates: int = DEFAULT_MAX_RERANK_CANDIDATES,
) -> list[SearchResult]:
    return run_search(
        client,
        collection,
        query,
        species,
        top_k,
        query_rewrite=False,
        hybrid_rerank=hybrid_rerank,
        candidate_multiplier=candidate_multiplier,
        max_candidates=max_candidates,
    ).results


def run_search(
    client: Any,
    collection: Any,
    query: str,
    species: str,
    top_k: int,
    *,
    query_rewrite: bool = True,
    hybrid_rerank: bool = True,
    candidate_multiplier: int = DEFAULT_RERANK_CANDIDATE_MULTIPLIER,
    max_candidates: int = DEFAULT_MAX_RERANK_CANDIDATES,
    query_rewriter: Callable[[str, str], str] | None = None,
) -> SearchRun:
    retrieval_query = query
    rewrite_failed = False
    if query_rewrite:
        try:
            cache_key = (species, query.strip())
            if query_rewriter is None and cache_key in QUERY_REWRITE_CACHE:
                retrieval_query = QUERY_REWRITE_CACHE[cache_key]
            else:
                raw_query = (
                    query_rewriter(query, species)
                    if query_rewriter is not None
                    else rewrite_retrieval_query(client, query, species)
                )
                retrieval_query = sanitize_retrieval_query(raw_query)
                if query_rewriter is None:
                    QUERY_REWRITE_CACHE[cache_key] = retrieval_query
        except Exception:
            rewrite_failed = True
            retrieval_query = query

    embedding = embed_texts(client, [query_embedding_text(retrieval_query)])[0]
    candidate_k = rerank_candidate_count(
        top_k,
        hybrid_rerank=hybrid_rerank,
        candidate_multiplier=candidate_multiplier,
        max_candidates=max_candidates,
    )
    candidates = query_collection(collection, embedding, species, candidate_k)
    if hybrid_rerank:
        results = hybrid_rerank_results(retrieval_query, candidates, top_k)
    else:
        results = candidates[:top_k]
    return SearchRun(retrieval_query, rewrite_failed, results)


def print_results(results: Sequence[SearchResult]) -> None:
    if not results:
        print("검색 결과가 없습니다.")
        return
    for result in results:
        metadata = result.metadata
        preview = " ".join(result.document.split())[:300]
        print(f"\n[{result.rank}] 유사도 {result.similarity:.4f}")
        print(f"제목: {metadata.get('title')}")
        print(f"섹션: {' > '.join(metadata.get('section_path') or [])}")
        print(f"종: {', '.join(metadata.get('species') or [])}")
        print(f"chunk_id: {result.chunk_id}")
        print(f"URL: {metadata.get('canonical_url')}")
        print(f"본문: {preview}...")


def run_query(args: argparse.Namespace) -> None:
    collection = require_collection(chroma_client(args.db_path), args.collection)
    metadata = collection.metadata or {}
    if metadata.get("embedding_model") != MODEL or metadata.get("embedding_dimension") != DIMENSION:
        raise RagDbError("컬렉션의 임베딩 모델 또는 차원이 현재 검색 설정과 다릅니다.")
    run = run_search(
        openai_client(),
        collection,
        args.query,
        args.species,
        args.top_k,
        query_rewrite=not args.no_query_rewrite,
        hybrid_rerank=args.hybrid_rerank,
        candidate_multiplier=args.candidate_multiplier,
        max_candidates=args.max_candidates,
    )
    print(f"원문 질문: {args.query}")
    print(f"검색 질의: {run.retrieval_query}")
    if run.rewrite_failed:
        print("주의: query rewrite에 실패해 원문 질문으로 검색했습니다.")
    print_results(run.results)


def validate_gold_cases(cases: Sequence[dict[str, Any]]) -> None:
    seen: set[str] = set()
    required = {"case_id", "query", "species", "expected_document_ids", "top_k"}
    for number, case in enumerate(cases, start=1):
        missing = required - case.keys()
        if missing:
            raise RagDbError(f"골든 질문 {number}에 필드가 없습니다: {sorted(missing)}")
        if case["case_id"] in seen:
            raise RagDbError(f"중복 case_id입니다: {case['case_id']}")
        seen.add(case["case_id"])
        if case["species"] not in {"dog", "cat"}:
            raise RagDbError(f"{case['case_id']}의 species가 올바르지 않습니다.")
        expected = case["expected_document_ids"]
        if not isinstance(expected, list) or not expected or not all(isinstance(x, str) for x in expected):
            raise RagDbError(f"{case['case_id']}의 expected_document_ids가 올바르지 않습니다.")
        if not isinstance(case["top_k"], int) or case["top_k"] < 1:
            raise RagDbError(f"{case['case_id']}의 top_k가 올바르지 않습니다.")


def score_case(case: dict[str, Any], results: Sequence[SearchResult]) -> tuple[bool, list[str]]:
    expected = set(case["expected_document_ids"])
    retrieved = [result.metadata.get("document_id", "") for result in results]
    correct_document = bool(expected.intersection(retrieved))
    correct_species = all(case["species"] in (r.metadata.get("species") or []) for r in results)
    complete = all(
        r.document.strip() and r.metadata.get("title") and r.metadata.get("canonical_url")
        for r in results
    )
    enough_results = bool(results)
    failures = []
    if not correct_document:
        failures.append("기대 문서가 top-k에 없음")
    if not correct_species:
        failures.append("다른 종 결과가 섞임")
    if not complete:
        failures.append("제목·URL·본문 누락")
    if not enough_results:
        failures.append("검색 결과 없음")
    return not failures, failures


def expected_document_rank(case: dict[str, Any], results: Sequence[SearchResult]) -> int | None:
    expected = set(case["expected_document_ids"])
    for rank, result in enumerate(results, start=1):
        if result.metadata.get("document_id") in expected:
            return rank
    return None


def run_evaluate(args: argparse.Namespace) -> None:
    cases = load_jsonl(args.gold)
    validate_gold_cases(cases)
    collection = require_collection(chroma_client(args.db_path), args.collection)
    openai = openai_client()
    passed = 0
    reciprocal_rank_total = 0.0
    total_seconds = 0.0
    rewrite_mode = "query-rewrite" if not args.no_query_rewrite else "original-query"
    retrieval_mode = "hybrid-rerank" if args.hybrid_rerank else "dense-only"
    mode = f"{rewrite_mode} + {retrieval_mode}"
    print(f"평가 모드: {mode}")
    if args.hybrid_rerank:
        print(
            "후보 확장: "
            f"candidate_multiplier={args.candidate_multiplier}, max_candidates={args.max_candidates}"
        )
    for case in cases:
        started = time.perf_counter()
        run = run_search(
            openai,
            collection,
            case["query"],
            case["species"],
            case["top_k"],
            query_rewrite=not args.no_query_rewrite,
            hybrid_rerank=args.hybrid_rerank,
            candidate_multiplier=args.candidate_multiplier,
            max_candidates=args.max_candidates,
        )
        elapsed = time.perf_counter() - started
        total_seconds += elapsed
        ok, failures = score_case(case, run.results)
        expected_rank = expected_document_rank(case, run.results)
        if expected_rank is not None:
            reciprocal_rank_total += 1.0 / expected_rank
        status = "PASS" if ok else "FAIL"
        rank_label = f"rank={expected_rank}" if expected_rank is not None else "rank=miss"
        print(f"[{status}] {case['case_id']} ({rank_label}, {elapsed * 1000:.0f}ms): {case['query']}")
        print(f"  retrieval_query: {run.retrieval_query}")
        if run.rewrite_failed:
            print("  rewrite_failed: true")
        if ok:
            passed += 1
        else:
            print(f"  사유: {', '.join(failures)}")
            for result in run.results:
                print(
                    f"  {result.rank}. {result.metadata.get('document_id')} "
                    f"(유사도 {result.similarity:.4f})"
                )
    total = len(cases)
    recall = passed / total if total else 0.0
    mrr = reciprocal_rank_total / total if total else 0.0
    average_latency_ms = (total_seconds / total * 1000) if total else 0.0
    print(f"\n평가 결과: {passed}/{total} 통과")
    print(f"Recall@k: {recall:.3f}")
    print(f"MRR: {mrr:.3f}")
    print(f"평균 검색 지연시간: {average_latency_ms:.0f}ms/case")
    if passed != len(cases):
        raise RagDbError("골든 질문 평가가 모두 통과하지 못했습니다.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="OpenAI 임베딩과 로컬 ChromaDB로 Cornell RAG 색인을 관리합니다."
    )
    parser.add_argument("command", choices=("check", "index", "inspect", "query", "evaluate"))
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--db-path", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--expected-count", type=int, default=EXPECTED_CHUNKS)
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--query")
    parser.add_argument("--species", choices=("dog", "cat"))
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--gold", type=Path, default=DEFAULT_GOLD)
    parser.set_defaults(hybrid_rerank=False)
    parser.add_argument(
        "--hybrid-rerank",
        dest="hybrid_rerank",
        action="store_true",
        help="Enable experimental hybrid rerank after dense retrieval.",
    )
    parser.add_argument(
        "--no-hybrid-rerank",
        dest="hybrid_rerank",
        action="store_false",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--no-query-rewrite",
        action="store_true",
        help="Use the original user question for retrieval instead of an English rewritten query.",
    )
    parser.add_argument(
        "--candidate-multiplier",
        type=int,
        default=DEFAULT_RERANK_CANDIDATE_MULTIPLIER,
        help="Dense candidate expansion factor used before hybrid rerank.",
    )
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=DEFAULT_MAX_RERANK_CANDIDATES,
        help="Maximum dense candidates fetched before hybrid rerank.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    handlers = {
        "check": run_check,
        "index": run_index,
        "inspect": run_inspect,
        "query": run_query,
        "evaluate": run_evaluate,
    }
    try:
        if args.command == "query" and (not args.query or not args.species):
            raise RagDbError("query 명령에는 --query와 --species가 모두 필요합니다.")
        handlers[args.command](args)
        return 0
    except RagDbError as exc:
        print(f"오류: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
