from __future__ import annotations

from typing import Any

from petcare_agent.rag.adapter import UnavailableRAGAdapter, retrieve
from petcare_agent.schemas.graph_state import RetrievedChunk


class MockRAGAdapter:
    def __init__(self, chunks: list[RetrievedChunk]) -> None:
        self.chunks = chunks
        self.calls: list[dict[str, Any]] = []

    def retrieve(
        self,
        query: str,
        filters: dict[str, Any],
        top_k: int = 5,
    ) -> list[RetrievedChunk]:
        self.calls.append({"query": query, "filters": filters, "top_k": top_k})
        return self.chunks


class FailingRAGAdapter:
    def retrieve(
        self,
        query: str,
        filters: dict[str, Any],
        top_k: int = 5,
    ) -> list[RetrievedChunk]:
        raise TimeoutError("rag provider timed out")


def test_default_rag_adapter_returns_empty_list_without_external_call() -> None:
    adapter = UnavailableRAGAdapter()

    assert adapter.retrieve("cat cough care", {"species": "cat"}, top_k=5) == []
    assert retrieve("cat cough care", {"species": "cat"}) == []


def test_mock_rag_adapter_can_return_retrieved_chunks() -> None:
    chunk = RetrievedChunk(
        chunk_id="chunk_001",
        source_id="care_guide_cough_cat",
        title="Cat cough observation",
        text="Monitor breathing, appetite, and energy.",
        score=0.82,
        metadata={"species": "cat", "topic": "cough"},
    )
    adapter = MockRAGAdapter(chunks=[chunk])

    result = retrieve(
        " cat cough care ",
        {"species": "cat", "chief_complaint": "cough"},
        top_k=3,
        adapter=adapter,
    )

    assert result == [chunk]
    assert adapter.calls == [
        {
            "query": "cat cough care",
            "filters": {"species": "cat", "chief_complaint": "cough"},
            "top_k": 3,
        }
    ]


def test_rag_adapter_error_falls_back_to_empty_chunks() -> None:
    result = retrieve(
        "cat cough care",
        {"species": "cat"},
        adapter=FailingRAGAdapter(),
    )

    assert result == []
