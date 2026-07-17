#!/usr/bin/env python3
"""Build and inspect a local ChromaDB index for the Cornell pet corpus.

OpenAI creates embeddings. ChromaDB only stores and searches the explicit
embeddings passed to it; Chroma's default embedding function is never used.
The module keeps third-party imports lazy so corpus validation and unit tests
can run before optional RAG dependencies are installed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
import time

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence


MODEL = "text-embedding-3-small"
DIMENSION = 1536
EXPECTED_CHUNKS = 732
DEFAULT_COLLECTION = "cornell_pet_health_text_embedding_3_small_1536"
DEFAULT_INPUT = Path("rag_data/chunks/cornell_pet_health_chunks.jsonl")
DEFAULT_DB_PATH = Path("rag_data/chroma")
DEFAULT_GOLD = Path("rag_data/evaluation/cornell_retrieval_gold.jsonl")
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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise RagDbError(f"File not found: {path}")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, raw_line in enumerate(stream, start=1):
            if not raw_line.strip():
                raise RagDbError(f"{path} line {line_number} is empty.")
            try:
                value = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise RagDbError(f"{path} line {line_number} is not valid JSON: {exc}") from exc
            if not isinstance(value, dict):
                raise RagDbError(f"{path} line {line_number} must be a JSON object.")
            rows.append(value)
    return rows


def validate_corpus_rows(
    rows: Sequence[dict[str, Any]], expected_count: int | None = EXPECTED_CHUNKS
) -> None:
    if expected_count is not None and len(rows) != expected_count:
        raise RagDbError(
            f"Chunk count mismatch. Expected {expected_count}, got {len(rows)}."
        )
    seen: set[str] = set()
    for index, row in enumerate(rows, start=1):
        missing = sorted(REQUIRED_FIELDS - row.keys())
        if missing:
            raise RagDbError(f"Chunk {index} is missing required fields: {', '.join(missing)}")
        chunk_id = row["chunk_id"]
        if not isinstance(chunk_id, str) or not chunk_id.strip():
            raise RagDbError(f"Chunk {index} has an empty chunk_id.")
        if chunk_id in seen:
            raise RagDbError(f"Duplicate chunk_id: {chunk_id}")
        seen.add(chunk_id)
        if not isinstance(row["content"], str) or not row["content"].strip():
            raise RagDbError(f"{chunk_id} has empty content.")
        species = row["species"]
        if not isinstance(species, list) or len(species) != 1 or species[0] not in {"dog", "cat"}:
            raise RagDbError(f"{chunk_id} species must be ['dog'] or ['cat'].")
        for field in ("section_path", "categories"):
            value = row[field]
            if not isinstance(value, list) or not value or not all(
                isinstance(item, str) and item for item in value
            ):
                raise RagDbError(f"{chunk_id} {field} must be a non-empty string array.")
        url = row["canonical_url"]
        if not isinstance(url, str) or not url.startswith("https://www.vet.cornell.edu/"):
            raise RagDbError(f"{chunk_id} has an invalid Cornell canonical_url.")
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
                raise RagDbError(f"{chunk_id} has an empty {field}.")
        if row["last_updated"] is not None and not isinstance(row["last_updated"], str):
            raise RagDbError(f"{chunk_id} last_updated must be a string or null.")


def read_corpus(path: Path, expected_count: int | None = EXPECTED_CHUNKS) -> Corpus:
    rows = load_jsonl(path)
    validate_corpus_rows(rows, expected_count)
    return Corpus(tuple(rows), sha256_file(path))


def document_embedding_text(row: dict[str, Any]) -> str:
    return f"title: {row['title']} | text: {row['content']}"


@dataclass(frozen=True)
class QueryExpansionEntry:
    triggers: tuple[str, ...]
    terms: tuple[str, ...]


@dataclass(frozen=True)
class NormalizedQuery:
    original: str
    species: str | None
    english_terms: tuple[str, ...]
    cross_lingual: bool


SPECIES_QUERY_TERMS: dict[str, tuple[str, ...]] = {
    "dog": ("dog", "canine", "canine health"),
    "cat": ("cat", "feline", "feline health"),
}

# Domain-level Korean/English veterinary lexicon for cross-lingual retrieval.
# Keep entries at the symptom, exposure, disease-family, and clinical-term level;
# do not encode gold case ids, chunk ids, or one-off expected documents here.
QUERY_EXPANSION_LEXICON: tuple[QueryExpansionEntry, ...] = (
    QueryExpansionEntry(
        triggers=("초콜릿", "카카오", "chocolate", "cocoa"),
        terms=("chocolate", "cocoa", "toxicity", "poisoning", "toxin ingestion", "methylxanthine", "theobromine"),
    ),
    QueryExpansionEntry(
        triggers=("자일리톨", "껌", "무설탕", "xylitol", "gum", "sugar free"),
        terms=("xylitol", "sugar-free gum", "toxicity", "poisoning", "toxin ingestion", "hypoglycemia", "liver injury"),
    ),
    QueryExpansionEntry(
        triggers=("열사병", "더운", "더위", "헐떡", "쓰러", "heatstroke", "heat stroke", "panting"),
        terms=("heatstroke", "heat stroke", "hyperthermia", "hot weather", "panting", "collapse", "emergency"),
    ),
    QueryExpansionEntry(
        triggers=("발작", "경련", "seizure", "convulsion"),
        terms=("seizure", "convulsion", "epilepsy", "neurologic signs", "emergency"),
    ),
    QueryExpansionEntry(
        triggers=("구토", "토해", "토하", "vomit", "vomiting", "nausea"),
        terms=("vomiting", "nausea", "regurgitation", "repeated vomiting", "clinical signs", "veterinary care"),
    ),
    QueryExpansionEntry(
        triggers=("심장사상충", "heartworm"),
        terms=("heartworm", "heartworm disease", "heartworms", "dirofilariasis", "Dirofilaria immitis", "mosquito-borne parasite", "mosquitoes", "monthly heartworm preventive", "routine prevention medication", "adult worms in heart and lungs"),
    ),
    QueryExpansionEntry(
        triggers=("신장", "콩팥", "kidney", "renal", "ckd"),
        terms=("kidney disease", "renal disease", "chronic kidney disease", "CKD", "increased thirst", "increased urination", "management"),
    ),
    QueryExpansionEntry(
        triggers=("갑상선", "hyperthyroid", "thyroid"),
        terms=("thyroid", "hyperthyroidism", "increased appetite", "weight loss", "older cat", "endocrine disease"),
    ),
    QueryExpansionEntry(
        triggers=("당뇨", "diabetes", "insulin", "혈당"),
        terms=("diabetes mellitus", "glucose", "insulin", "increased thirst", "increased urination", "weight loss", "endocrine disease"),
    ),
    QueryExpansionEntry(
        triggers=("요로", "소변", "오줌", "방광", "화장실", "urinary", "urinate", "bladder", "litter box"),
        terms=("lower urinary tract", "FLUTD", "LUTD", "LUTS", "bladder", "urethra", "straining to urinate", "frequent small urination", "litter box", "idiopathic cystitis", "urethral obstruction", "urinary tract infection", "UTI"),
    ),
    QueryExpansionEntry(
        triggers=("천식", "기침", "숨쉬", "호흡", "asthma", "cough", "breathing"),
        terms=("asthma", "coughing", "breathing difficulty", "respiratory signs", "airway inflammation", "wheezing"),
    ),
    QueryExpansionEntry(
        triggers=("췌장", "췌장염", "pancreas", "pancreatitis"),
        terms=("pancreatitis", "pancreas", "abdominal pain", "vomiting", "appetite loss", "diagnosis"),
    ),
    QueryExpansionEntry(
        triggers=("설사", "diarrhea"),
        terms=("diarrhea", "stool", "gastrointestinal signs", "dehydration", "veterinary care"),
    ),
    QueryExpansionEntry(
        triggers=("피부", "가려", "긁", "알레르", "skin", "itch", "allergy"),
        terms=("skin", "itching", "allergy", "dermatitis", "hair loss", "scratching"),
    ),
)


def has_hangul(text: str) -> bool:
    return any("\uac00" <= character <= "\ud7a3" for character in text)


def _dedupe_terms(values: Iterable[str]) -> tuple[str, ...]:
    terms: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = " ".join(value.split())
        key = normalized.lower()
        if normalized and key not in seen:
            terms.append(normalized)
            seen.add(key)
    return tuple(terms)


def expand_query_terms(query: str) -> list[str]:
    """Return reusable English veterinary terms for cross-lingual retrieval."""

    lowered = query.lower()
    matched_terms: list[str] = []
    for entry in QUERY_EXPANSION_LEXICON:
        if any(trigger.lower() in lowered for trigger in entry.triggers):
            matched_terms.extend(entry.terms)
    return list(_dedupe_terms(matched_terms))


def normalize_query_for_embedding(query: str, species: str | None = None) -> NormalizedQuery:
    query = query.strip()
    if not query:
        raise RagDbError("Search query must not be empty.")
    if species is not None and species not in SPECIES_QUERY_TERMS:
        raise RagDbError("species must be dog or cat.")
    return NormalizedQuery(
        original=query,
        species=species,
        english_terms=tuple(expand_query_terms(query)),
        cross_lingual=has_hangul(query),
    )


def query_embedding_text(query: str, species: str | None = None) -> str:
    normalized = normalize_query_for_embedding(query, species)
    lines = [
        "task: retrieve evidence from Cornell pet health corpus",
        f"user question: {normalized.original}",
    ]
    if normalized.species:
        lines.append(
            "species context: "
            + ", ".join(SPECIES_QUERY_TERMS[normalized.species])
        )
    if normalized.english_terms:
        lines.append(
            "normalized English veterinary concepts: "
            + ", ".join(normalized.english_terms)
        )
    if normalized.cross_lingual:
        lines.append("cross-lingual hint: user question may be Korean; indexed corpus language is English")
    return "\n".join(lines)
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
        raise RagDbError(f"Embedding API returned {len(embeddings)} embeddings; expected {expected}.")
    for position, embedding in enumerate(embeddings, start=1):
        if len(embedding) != DIMENSION:
            raise RagDbError(
                f"Embedding {position} has {len(embedding)} dimensions; expected {DIMENSION}."
            )
        if not all(isinstance(value, (int, float)) and math.isfinite(value) for value in embedding):
            raise RagDbError(f"Embedding {position} contains an invalid numeric value.")


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


def safe_provider_diagnostic(exc: BaseException) -> str:
    """Return provider error text with credential-looking values removed."""

    text = " ".join(str(exc).split())
    replacements = [
        (r"sk-[0-9A-Za-z_-]{10,}", "[REDACTED_API_KEY]"),
        (r"AIza[0-9A-Za-z_-]{10,}", "[REDACTED_API_KEY]"),
        (r"(?i)(Incorrect API key provided:\s*)[^.\s,;}]+", r"\1[REDACTED]"),
        (r"(?i)(x-goog-api-key|api[_ -]?key|authorization)(\s*[:=]\s*)([^\s,;}]+)", r"\1\2[REDACTED]"),
    ]
    import re

    for pattern, replacement in replacements:
        text = re.sub(pattern, replacement, text)
    if len(text) > 800:
        text = text[:800] + "..."
    status = _status_code(exc)
    prefix = f"HTTP {status}" if status is not None else type(exc).__name__
    return f"{prefix}: {text}" if text else prefix

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
            'OPENAI_API_KEY is not configured. In PowerShell run '
            '$env:OPENAI_API_KEY="your_key" or set it in .env.'
        )
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RagDbError(
            "openai is not installed. Run python -m pip install -r requirements-rag.txt."
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
                f"  Temporary embedding API error ({_status_code(exc)}); retrying in "
                f"{delay:.1f}s ({attempt}/5)."
            ),
        )
    except Exception as exc:
        raise RagDbError(f"OpenAI embedding request failed: {safe_provider_diagnostic(exc)}") from exc
    objects = getattr(response, "data", None)
    if objects is None:
        raise RagDbError("OpenAI response did not include a data field.")
    embeddings = [list(item.embedding) for item in objects]
    validate_embeddings(embeddings, len(texts))
    return embeddings


def chroma_client(db_path: Path) -> Any:
    try:
        import chromadb
    except ImportError as exc:
        raise RagDbError(
            "chromadb is not installed. Run python -m pip install -r requirements-rag.txt."
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
            mismatches.append(f"{key}: DB={actual.get(key)!r}, expected={expected[key]!r}")
    if mismatches:
        raise RagDbError(
            "Existing collection is incompatible with the current input/settings. "
            "Use --rebuild or choose a new collection. " + "; ".join(mismatches)
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
        raise RagDbError("batch-size must be at least 1.")
    for start in range(0, len(values), size):
        yield values[start : start + size]


def run_check(args: argparse.Namespace) -> None:
    corpus = read_corpus(args.input, args.expected_count)
    print(f"[1/4] JSONL check complete: {len(corpus.rows)} chunks")
    print(f"[2/4] Input SHA-256: {corpus.sha256}")
    args.db_path.mkdir(parents=True, exist_ok=True)
    probe = args.db_path / ".write-test"
    try:
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
    except OSError as exc:
        raise RagDbError(f"DB path is not writable: {args.db_path}: {exc}") from exc
    print(f"[3/4] DB path is writable: {args.db_path}")
    client = openai_client()
    embedding = embed_texts(client, [query_embedding_text("pet health information")])[0]
    print(f"[4/4] OpenAI embedding API connection successful: {MODEL}, {len(embedding)} dimensions")
    print("Pre-index checks passed. You can now run the index command.")


def run_index(args: argparse.Namespace) -> None:
    corpus = read_corpus(args.input, args.expected_count)
    client = chroma_client(args.db_path)
    collection = get_or_create_collection(client, args.collection, corpus, args.rebuild)
    known = existing_hashes(collection)
    todo = pending_rows(corpus.rows, known)
    print(f"Input {len(corpus.rows)} / already complete {len(corpus.rows) - len(todo)} / chunks to process {len(todo)}")
    if todo:
        embedding_client = openai_client()
        completed = len(corpus.rows) - len(todo)
        for batch in batched(todo, args.batch_size):
            embeddings = embed_texts(
                embedding_client, [document_embedding_text(row) for row in batch]
            )
            collection.upsert(
                ids=[row["chunk_id"] for row in batch],
                embeddings=embeddings,
                documents=[row["content"] for row in batch],
                metadatas=[chroma_metadata(row) for row in batch],
            )
            completed += len(batch)
            print(f"  Upsert complete: {completed}/{len(corpus.rows)}")
    count = collection.count()
    if count != len(corpus.rows):
        raise RagDbError(
            f"Index finished but DB count mismatched. Input {len(corpus.rows)}, DB {count}."
        )
    print(f"Index complete: {count} chunks are in collection {args.collection}.")


def require_collection(client: Any, name: str) -> Any:
    if name not in collection_names(client):
        raise RagDbError(f"Collection not found: {name}. Run the index command first.")
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
    print(f"Total chunks: {len(ids)}")
    print(f"Unique documents: {len(document_ids)}")
    print(f"Species chunks: dog={dog}, cat={cat}")
    print(f"Collection metadata: {collection.metadata}")
    print(f"Missing required metadata: {missing}")
    print(f"Vectors not {DIMENSION} dimensions: {wrong_dimension}")
    if ids:
        print("\nExample card")
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
        raise RagDbError("species must be dog or cat.")
    if top_k < 1:
        raise RagDbError("top-k must be at least 1.")
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


def search(client: Any, collection: Any, query: str, species: str, top_k: int) -> list[SearchResult]:
    embedding = embed_texts(client, [query_embedding_text(query, species=species)])[0]
    return query_collection(collection, embedding, species, top_k)


def print_results(results: Sequence[SearchResult]) -> None:
    if not results:
        print("No search results.")
        return
    for result in results:
        metadata = result.metadata
        preview = " ".join(result.document.split())[:300]
        print(f"\n[{result.rank}] similarity {result.similarity:.4f}")
        print(f"Title: {metadata.get('title')}")
        print(f"Section: {' > '.join(metadata.get('section_path') or [])}")
        print(f"Species: {', '.join(metadata.get('species') or [])}")
        print(f"chunk_id: {result.chunk_id}")
        print(f"URL: {metadata.get('canonical_url')}")
        print(f"Content: {preview}...")


def run_query(args: argparse.Namespace) -> None:
    collection = require_collection(chroma_client(args.db_path), args.collection)
    metadata = collection.metadata or {}
    if metadata.get("embedding_model") != MODEL or metadata.get("embedding_dimension") != DIMENSION:
        raise RagDbError("Collection embedding model or dimension does not match current settings.")
    results = search(openai_client(), collection, args.query, args.species, args.top_k)
    print_results(results)


def validate_gold_cases(cases: Sequence[dict[str, Any]]) -> None:
    seen: set[str] = set()
    required = {"case_id", "query", "species", "expected_document_ids", "top_k"}
    for number, case in enumerate(cases, start=1):
        missing = required - case.keys()
        if missing:
            raise RagDbError(f"Gold case {number} is missing fields: {sorted(missing)}")
        if case["case_id"] in seen:
            raise RagDbError(f"Duplicate case_id: {case['case_id']}")
        seen.add(case["case_id"])
        if case["species"] not in {"dog", "cat"}:
            raise RagDbError(f"{case['case_id']} has invalid species.")
        expected = case["expected_document_ids"]
        if not isinstance(expected, list) or not expected or not all(isinstance(x, str) for x in expected):
            raise RagDbError(f"{case['case_id']} has invalid expected_document_ids.")
        if not isinstance(case["top_k"], int) or case["top_k"] < 1:
            raise RagDbError(f"{case['case_id']} has invalid top_k.")


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
        failures.append("expected document missing from top-k")
    if not correct_species:
        failures.append("wrong species result returned")
    if not complete:
        failures.append("title, URL, or body missing")
    if not enough_results:
        failures.append("no results")
    return not failures, failures


def run_evaluate(args: argparse.Namespace) -> None:
    cases = load_jsonl(args.gold)
    validate_gold_cases(cases)
    collection = require_collection(chroma_client(args.db_path), args.collection)
    embedding_client = openai_client()
    passed = 0
    for case in cases:
        results = search(
            embedding_client,
            collection,
            case["query"],
            case["species"],
            case["top_k"],
        )
        ok, failures = score_case(case, results)
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] {case['case_id']}: {case['query']}")
        if ok:
            passed += 1
        else:
            print(f"  reasons: {', '.join(failures)}")
            for result in results:
                print(
                    f"  {result.rank}. {result.metadata.get('document_id')} "
                    f"(similarity {result.similarity:.4f})"
                )
    print(f"\nEvaluation result: {passed}/{len(cases)} passed")
    if passed != len(cases):
        raise RagDbError("Not all gold retrieval cases passed.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Manage a Cornell RAG index with OpenAI embeddings and local ChromaDB."
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
            raise RagDbError("query command requires both --query and --species.")
        handlers[args.command](args)
        return 0
    except RagDbError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())