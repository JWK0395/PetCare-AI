"""Small, dependency-free public data types used by the RAG pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: str
    document_id: str
    title: str
    section_path: list[str]
    species: list[str]
    canonical_url: str
    content: str
    distance: float

    @property
    def similarity(self) -> float:
        """Cosine similarity derived from Chroma's cosine distance."""

        return 1.0 - self.distance

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RagAnswer:
    answer: str
    cited_source_numbers: list[int]
    insufficient_evidence: bool
    disclaimer: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Citation:
    number: int
    title: str
    section_path: list[str]
    url: str
    chunk_id: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RagResponse:
    question: str
    species: str
    answer: str
    insufficient_evidence: bool
    citations: list[Citation]
    disclaimer: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PipelineTrace:
    """Safe learning trace. It deliberately never contains API keys or vectors."""

    embedding_prompt: str
    retrieved_chunks: list[RetrievedChunk]
    context: str
    generation_prompt: str
    cited_source_numbers: list[int]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
