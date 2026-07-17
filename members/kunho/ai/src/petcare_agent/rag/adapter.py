"""Safe RAG adapter interface.

Phase 8 defines only the boundary and fallback behavior. The default adapter
does not call a vector database, external API, database, or LLM.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping, Protocol

from pydantic import ValidationError

from petcare_agent.schemas.graph_state import RetrievedChunk

RAGFilters = Mapping[str, Any]


class RAGAdapter(Protocol):
    """Provider boundary for retrieving reference chunks."""

    def retrieve(
        self,
        query: str,
        filters: RAGFilters,
        top_k: int = 5,
    ) -> list[RetrievedChunk]:
        """Return chunks for the query and filters."""


class UnavailableRAGAdapter:
    """Default no-op RAG adapter used until a backend is connected."""

    def retrieve(
        self,
        query: str,
        filters: RAGFilters,
        top_k: int = 5,
    ) -> list[RetrievedChunk]:
        """Return no chunks without performing external I/O."""

        return []


def retrieve(
    query: str,
    filters: RAGFilters,
    top_k: int = 5,
    *,
    adapter: RAGAdapter | None = None,
) -> list[RetrievedChunk]:
    """Safely retrieve chunks through an adapter.

    Empty queries, invalid provider payloads, provider errors, and provider
    timeouts all fall back to an empty chunk list so the graph can continue.
    """

    normalized_query = query.strip()
    if not normalized_query or top_k <= 0:
        return []

    rag_adapter = adapter or UnavailableRAGAdapter()
    try:
        raw_chunks = rag_adapter.retrieve(
            normalized_query,
            deepcopy(dict(filters)),
            top_k=top_k,
        )
        return _coerce_chunks(raw_chunks)[:top_k]
    except (TimeoutError, ValidationError, TypeError, ValueError):
        return []
    except Exception:
        return []


def _coerce_chunks(raw_chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    return [
        chunk.model_copy(deep=True)
        if isinstance(chunk, RetrievedChunk)
        else RetrievedChunk.model_validate(chunk)
        for chunk in raw_chunks
    ]


__all__ = ["RAGAdapter", "RAGFilters", "UnavailableRAGAdapter", "retrieve"]
