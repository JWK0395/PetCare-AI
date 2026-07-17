"""RAG Agent node for official-source retrieval."""

from __future__ import annotations

from typing import Any

from petcare_agent.rag.adapter import RAGAdapter, retrieve
from petcare_agent.schemas.graph_state import PetCareGraphState, RAGCitation, RetrievedChunk

RAG_SAFE_RISK_LEVELS = {"urgent", "non_emergency", "unknown"}


def retrieve_rag_context(
    state: PetCareGraphState,
    *,
    adapter: RAGAdapter | None = None,
    top_k: int = 5,
) -> PetCareGraphState:
    """Populate retrieval evidence from state.retrieval.query when available."""

    next_state = state.model_copy(deep=True)
    query = next_state.retrieval.query.strip()

    if not query:
        next_state.retrieval.chunks = []
        next_state.retrieval.citations = []
        next_state.retrieval.provider = ""
        next_state.retrieval.insufficient_evidence = False
        next_state.retrieval.errors = []
        return next_state

    chunks = retrieve(
        query,
        _build_filters(next_state),
        top_k=top_k,
        adapter=adapter,
    )
    next_state.retrieval.chunks = chunks
    next_state.retrieval.citations = _build_citations(chunks)
    next_state.retrieval.provider = _provider(chunks)
    next_state.retrieval.insufficient_evidence = not chunks
    next_state.retrieval.errors = []
    next_state.next_route = "rag"
    return next_state


def rag_agent(
    state: PetCareGraphState,
    *,
    adapter: RAGAdapter | None = None,
    top_k: int = 5,
) -> PetCareGraphState:
    """LangGraph-friendly alias for the RAG Agent node."""

    return retrieve_rag_context(state, adapter=adapter, top_k=top_k)


def _build_filters(state: PetCareGraphState) -> dict[str, Any]:
    chief_complaint = state.emergency_screening.chief_complaint.strip() or None
    risk_level = state.risk_level if state.risk_level in RAG_SAFE_RISK_LEVELS else None
    return {
        "species": state.species,
        "chief_complaint": chief_complaint,
        "risk_level": risk_level,
        "locale": state.locale,
    }


def _build_citations(chunks: list[RetrievedChunk]) -> list[RAGCitation]:
    citations: list[RAGCitation] = []
    for index, chunk in enumerate(chunks, start=1):
        metadata = chunk.metadata or {}
        url = str(metadata.get("canonical_url") or metadata.get("url") or "")
        section_path = metadata.get("section_path") or []
        citations.append(
            RAGCitation(
                number=index,
                title=chunk.title,
                url=url,
                chunk_id=chunk.chunk_id,
                section_path=[str(item) for item in section_path] if isinstance(section_path, list) else [],
            )
        )
    return citations


def _provider(chunks: list[RetrievedChunk]) -> str:
    for chunk in chunks:
        provider = str((chunk.metadata or {}).get("provider") or "").strip()
        if provider:
            return provider
    return ""


__all__ = ["RAG_SAFE_RISK_LEVELS", "rag_agent", "retrieve_rag_context"]
