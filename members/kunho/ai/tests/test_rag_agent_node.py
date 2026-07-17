from __future__ import annotations

from typing import Any

from petcare_agent.nodes.rag_agent import retrieve_rag_context
from petcare_agent.schemas.graph_state import (
    EmergencyScreening,
    PetCareGraphState,
    RetrievedChunk,
    RetrievalState,
)


class RecordingRAGAdapter:
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
    def __init__(self) -> None:
        self.calls = 0

    def retrieve(
        self,
        query: str,
        filters: dict[str, Any],
        top_k: int = 5,
    ) -> list[RetrievedChunk]:
        self.calls += 1
        raise RuntimeError("rag backend unavailable")


def test_rag_agent_calls_adapter_with_query_and_filters() -> None:
    chunk = RetrievedChunk(
        chunk_id="chunk_001",
        source_id="care_guide_vomiting_dog",
        title="Dog vomiting care",
        text="Watch hydration, repeated vomiting, and energy.",
        score=0.77,
        metadata={
            "provider": "cornell",
            "species": "dog",
            "topic": "vomiting",
            "canonical_url": "https://www.vet.cornell.edu/dog-vomiting",
            "section_path": ["Dog vomiting care"],
        },
    )
    adapter = RecordingRAGAdapter(chunks=[chunk])
    state = PetCareGraphState(
        species="dog",
        risk_level="urgent",
        locale="ko-KR",
        emergency_screening=EmergencyScreening(chief_complaint="vomiting"),
        retrieval=RetrievalState(query="dog vomiting urgent care"),
    )

    result = retrieve_rag_context(state, adapter=adapter, top_k=4)

    assert adapter.calls == [
        {
            "query": "dog vomiting urgent care",
            "filters": {
                "species": "dog",
                "chief_complaint": "vomiting",
                "risk_level": "urgent",
                "locale": "ko-KR",
            },
            "top_k": 4,
        }
    ]
    assert result.retrieval.chunks == [chunk]
    assert result.retrieval.provider == "cornell"
    assert result.retrieval.insufficient_evidence is False
    assert result.retrieval.citations[0].url == "https://www.vet.cornell.edu/dog-vomiting"
    assert result.next_route == "rag"
    assert state.retrieval.chunks == []


def test_rag_agent_skips_adapter_when_query_is_empty() -> None:
    adapter = RecordingRAGAdapter(chunks=[])
    stale_chunk = RetrievedChunk(
        chunk_id="chunk_stale",
        source_id="source_old",
        title="Old chunk",
        text="This should not remain when there is no query.",
    )
    state = PetCareGraphState(
        retrieval=RetrievalState(query="  ", chunks=[stale_chunk]),
    )

    result = retrieve_rag_context(state, adapter=adapter)

    assert adapter.calls == []
    assert result.retrieval.chunks == []
    assert result.retrieval.citations == []
    assert result.retrieval.insufficient_evidence is False


def test_rag_agent_falls_back_to_empty_chunks_when_adapter_fails() -> None:
    adapter = FailingRAGAdapter()
    state = PetCareGraphState(
        species="cat",
        risk_level="unknown",
        emergency_screening=EmergencyScreening(chief_complaint="cough"),
        retrieval=RetrievalState(query="cat cough observation"),
    )

    result = retrieve_rag_context(state, adapter=adapter)

    assert adapter.calls == 1
    assert result.retrieval.chunks == []
    assert result.retrieval.citations == []
    assert result.retrieval.insufficient_evidence is True
    assert result.next_route == "rag"
