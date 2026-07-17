from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from petcare_agent.config import get_settings
from petcare_rag import GENERATION_MODEL, RetrievedChunk, retrieve
from petcare_rag.manage_cornell_rag_db import (
    DEFAULT_COLLECTION,
    DIMENSION,
    EXPECTED_CHUNKS,
    MODEL,
    embed_texts,
    expand_query_terms,
    query_embedding_text,
    read_corpus,
)
from petcare_rag.pipeline import RagAnswerOutput, generate_answer


class _FakeEmbeddingItem:
    def __init__(self, embedding: list[float]) -> None:
        self.embedding = embedding


class _FakeEmbeddings:
    def __init__(self) -> None:
        self.kwargs = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        return SimpleNamespace(
            data=[_FakeEmbeddingItem([0.01] * DIMENSION) for _ in kwargs["input"]]
        )


class _FakeEmbeddingClient:
    def __init__(self) -> None:
        self.embeddings = _FakeEmbeddings()


class _FakeCompletions:
    def __init__(self) -> None:
        self.kwargs = None

    def parse(self, **kwargs):
        self.kwargs = kwargs
        parsed = RagAnswerOutput(
            answer="Use the cited Cornell evidence [1].",
            cited_source_numbers=[1],
            insufficient_evidence=False,
            disclaimer="general information only",
        )
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(parsed=parsed))]
        )


class _FakeOpenAIClient:
    def __init__(self) -> None:
        self.completions = _FakeCompletions()
        self.beta = SimpleNamespace(
            chat=SimpleNamespace(completions=self.completions)
        )


def _chunk() -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id="chunk-1",
        document_id="doc-1",
        title="Cornell Pet Health",
        section_path=["Health"],
        species=["dog"],
        canonical_url="https://www.vet.cornell.edu/example",
        content="Cornell source content.",
        distance=0.1,
    )


def test_embedding_contract_uses_openai_text_embedding_3_small() -> None:
    assert MODEL == "text-embedding-3-small"
    assert DIMENSION == 1536
    assert DEFAULT_COLLECTION == "cornell_pet_health_text_embedding_3_small_1536"


def test_embed_texts_calls_openai_embeddings_api() -> None:
    client = _FakeEmbeddingClient()

    vectors = embed_texts(client, ["first", "second"])

    assert client.embeddings.kwargs == {
        "model": "text-embedding-3-small",
        "input": ["first", "second"],
        "dimensions": 1536,
    }
    assert len(vectors) == 2
    assert all(len(vector) == 1536 for vector in vectors)


def test_korean_query_normalization_adds_domain_terms_and_species_context() -> None:
    query = "\uac15\uc544\uc9c0\uac00 \ucd08\ucf5c\ub9bf\uc744 \uba39\uc5c8\uc744 \ub54c \uc5b4\ub5bb\uac8c \ud574\uc57c \ud574?"
    expansions = expand_query_terms(query)
    embedding_text = query_embedding_text(query, species="dog")

    assert {"chocolate", "toxicity", "theobromine"}.issubset(set(expansions))
    assert all("cornell_" not in term for term in expansions)
    assert "species context: dog, canine, canine health" in embedding_text
    assert "normalized English veterinary concepts:" in embedding_text
    assert "cross-lingual hint:" in embedding_text

    urinary_query = "\uace0\uc591\uc774\uac00 \ud654\uc7a5\uc2e4\uc744 \uc790\uc8fc \uac00\uc9c0\ub9cc \uc18c\ubcc0\uc744 \uc798 \ubabb \ubcf4\uba74 \uc5b4\ub5a4 \uc694\ub85c \uc9c8\ud658\uc744 \uc758\uc2ec\ud558\ub098?"
    urinary_expansions = set(expand_query_terms(urinary_query))

    assert {"lower urinary tract", "FLUTD", "straining to urinate"}.issubset(urinary_expansions)

    heartworm_expansions = set(expand_query_terms("\uac15\uc544\uc9c0 \uc2ec\uc7a5\uc0ac\uc0c1\ucda9 \uc608\ubc29"))
    assert {"heartworm disease", "mosquito-borne parasite", "monthly heartworm preventive"}.issubset(heartworm_expansions)

def test_vendored_petcare_rag_imports_from_project_pythonpath() -> None:
    assert callable(retrieve)
    assert RetrievedChunk.__name__ == "RetrievedChunk"


def test_cornell_corpus_is_vendored_in_expected_rag_data_path() -> None:
    corpus = read_corpus(Path("rag_data/chunks/cornell_pet_health_chunks.jsonl"))

    assert len(corpus.rows) == EXPECTED_CHUNKS
    assert corpus.rows[0]["canonical_url"].startswith("https://www.vet.cornell.edu/")


def test_cornell_retrieval_gold_is_vendored() -> None:
    path = Path("rag_data/evaluation/cornell_retrieval_gold.jsonl")

    assert path.is_file()
    assert path.read_text(encoding="utf-8").strip()


def test_standalone_generation_uses_openai_model(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_MODEL", GENERATION_MODEL)
    get_settings.cache_clear()
    client = _FakeOpenAIClient()

    try:
        result = generate_answer(
            "What should I watch for?",
            "dog",
            [_chunk()],
            generation_client=client,
        )
    finally:
        get_settings.cache_clear()

    assert client.completions.kwargs["model"] == "gpt-5.4-mini"
    assert client.completions.kwargs["response_format"] is RagAnswerOutput
    assert client.completions.kwargs["messages"][0]["role"] == "system"
    assert result.cited_source_numbers == [1]