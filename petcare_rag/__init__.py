"""Cornell official-source RAG pipeline for PetCare AI."""

from .models import Citation, PipelineTrace, RagAnswer, RagResponse, RetrievedChunk
from .pipeline import (
    DEFAULT_DISCLAIMER,
    GENERATION_MODEL,
    RagPipelineError,
    answer_question,
    build_context,
    build_generation_prompt,
    embed_question,
    generate_answer,
    open_collection,
    retrieve,
    run_pipeline,
)

__all__ = [
    "Citation",
    "DEFAULT_DISCLAIMER",
    "GENERATION_MODEL",
    "PipelineTrace",
    "RagAnswer",
    "RagPipelineError",
    "RagResponse",
    "RetrievedChunk",
    "answer_question",
    "build_context",
    "build_generation_prompt",
    "embed_question",
    "generate_answer",
    "open_collection",
    "retrieve",
    "run_pipeline",
]
