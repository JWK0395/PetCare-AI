"""Cornell official-source RAG adapter."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from petcare_agent.rag.adapter import RAGFilters
from petcare_agent.schemas.graph_state import RetrievedChunk

RawCornellRetriever = Callable[..., Sequence[Any]]


class CornellRAGAdapter:
    """Adapt the Cornell RAG package to the project's RetrievedChunk contract."""

    def __init__(
        self,
        *,
        retriever: RawCornellRetriever | None = None,
        provider_name: str = "cornell",
        **retriever_kwargs: Any,
    ) -> None:
        self._retriever = retriever
        self.provider_name = provider_name
        self.retriever_kwargs = dict(retriever_kwargs)

    def retrieve(
        self,
        query: str,
        filters: RAGFilters,
        top_k: int = 5,
    ) -> list[RetrievedChunk]:
        """Return Cornell chunks for dog/cat queries only."""

        species = str(filters.get("species") or "").strip().lower()
        if species not in {"cat", "dog"} or not query.strip() or top_k <= 0:
            return []

        try:
            raw_chunks = self._call_retriever(query.strip(), species, top_k)
        except Exception:
            return []

        chunks = [
            chunk
            for raw_chunk in raw_chunks
            if (chunk := self._coerce_chunk(raw_chunk, species)) is not None
        ]
        return chunks[:top_k]

    def _call_retriever(self, query: str, species: str, top_k: int) -> Sequence[Any]:
        retriever = self._retriever or _load_default_retriever()
        try:
            return retriever(query, species, top_k, **self.retriever_kwargs)
        except TypeError:
            return retriever(
                question=query,
                species=species,
                top_k=top_k,
                **self.retriever_kwargs,
            )

    def _coerce_chunk(self, raw_chunk: Any, requested_species: str) -> RetrievedChunk | None:
        metadata = _raw_metadata(raw_chunk)
        species_values = _as_list(_read(raw_chunk, "species", metadata.get("species", [])))
        if requested_species not in species_values:
            return None

        chunk_id = str(_read(raw_chunk, "chunk_id", metadata.get("chunk_id", ""))).strip()
        source_id = str(
            _read(
                raw_chunk,
                "document_id",
                _read(raw_chunk, "source_id", metadata.get("document_id", chunk_id)),
            )
        ).strip()
        title = str(_read(raw_chunk, "title", metadata.get("title", ""))).strip()
        text = str(
            _read(raw_chunk, "content", _read(raw_chunk, "text", metadata.get("content", "")))
        ).strip()
        url = str(
            _read(raw_chunk, "canonical_url", metadata.get("canonical_url", metadata.get("url", "")))
        ).strip()
        if not chunk_id or not source_id or not title or not text:
            return None

        section_path = _as_list(_read(raw_chunk, "section_path", metadata.get("section_path", [])))
        score = _similarity(raw_chunk)
        return RetrievedChunk(
            chunk_id=chunk_id,
            source_id=source_id,
            title=title,
            text=text,
            score=score,
            metadata={
                "provider": self.provider_name,
                "species": species_values,
                "canonical_url": url,
                "section_path": section_path,
                "source_institution": "Cornell University College of Veterinary Medicine",
            },
        )


def _load_default_retriever() -> RawCornellRetriever:
    from petcare_rag import retrieve as cornell_retrieve

    return cornell_retrieve


def _raw_metadata(raw_chunk: Any) -> dict[str, Any]:
    metadata = _read(raw_chunk, "metadata", {})
    return dict(metadata) if isinstance(metadata, Mapping) else {}


def _read(raw_value: Any, field: str, default: Any = None) -> Any:
    if isinstance(raw_value, Mapping):
        return raw_value.get(field, default)
    return getattr(raw_value, field, default)


def _as_list(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value] if value else []
    if isinstance(value, Sequence):
        return [str(item) for item in value if str(item)]
    return []


def _similarity(raw_chunk: Any) -> float | None:
    value = _read(raw_chunk, "similarity", None)
    if isinstance(value, (int, float)):
        return float(value)
    distance = _read(raw_chunk, "distance", None)
    if isinstance(distance, (int, float)):
        return 1.0 - float(distance)
    return None


__all__ = ["CornellRAGAdapter", "RawCornellRetriever"]
