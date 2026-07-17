"""Direct Chroma RAG pipeline for Cornell pet health information.

The module intentionally avoids LangChain. Each RAG step is a small function so
beginners can inspect retrieval, context construction, generation, and citation
validation independently.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Callable, Literal, Sequence

from pydantic import BaseModel, ConfigDict

from . import manage_cornell_rag_db as rag_db

from .models import Citation, PipelineTrace, RagAnswer, RagResponse, RetrievedChunk


Species = Literal["dog", "cat"]
Embedder = Callable[[str], Sequence[float]]
Generator = Callable[[str, str, dict[str, Any]], dict[str, Any] | RagAnswer]

GENERATION_MODEL = "gpt-5.4-mini"
DEFAULT_TOP_K = 5
DEFAULT_MAX_OUTPUT_TOKENS = 4096
DEFAULT_DISCLAIMER = "일반적인 공식 건강정보이며 진단이나 처방을 대신하지 않습니다."
INSUFFICIENT_ANSWER = (
    "검색된 Cornell 자료만으로는 이 질문에 충분히 답하기 어렵습니다. "
    "질문을 더 구체적으로 작성하거나 수의사에게 상담해 주세요."
)

SYSTEM_INSTRUCTION = """당신은 Cornell University College of Veterinary Medicine의 공식 자료를
보호자에게 쉽게 설명하는 검색 기반 도우미입니다.

반드시 지킬 규칙:
1. 제공된 SOURCE 내용만 근거로 사용합니다.
2. 한국어로 쉽고 간결하게 답합니다.
3. 확정 진단을 하지 않습니다.
4. 약물 처방, 복용량, 치료 변경을 제안하지 않습니다.
5. SOURCE에 없는 내용을 추측하지 않습니다.
6. 근거가 부족하면 insufficient_evidence를 true로 설정합니다.
7. 근거를 사용한 문장 끝에 [1], [2]처럼 SOURCE 번호를 표시합니다.
8. 요청한 동물 종과 다른 종의 정보를 사용하지 않습니다.
9. URL을 직접 작성하지 않습니다. URL은 프로그램이 별도로 붙입니다.
10. 이 모듈은 응급 판단 Agent가 아니므로 개인 상태에 대한 확정적인 행동 판정을 하지 않습니다.
"""

RAG_ANSWER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "cited_source_numbers": {
            "type": "array",
            "items": {"type": "integer"},
        },
        "insufficient_evidence": {"type": "boolean"},
        "disclaimer": {"type": "string"},
    },
    "required": [
        "answer",
        "cited_source_numbers",
        "insufficient_evidence",
        "disclaimer",
    ],
}


class RagPipelineError(RuntimeError):
    """Safe failure with an optional redacted diagnostic for ``--debug``."""

    def __init__(self, message: str, *, diagnostic: str | None = None) -> None:
        super().__init__(message)
        self.diagnostic = diagnostic


class RagAnswerOutput(BaseModel):
    """Structured answer shape requested from the OpenAI standalone path."""

    model_config = ConfigDict(extra="forbid")

    answer: str
    cited_source_numbers: list[int]
    insufficient_evidence: bool
    disclaimer: str


def _safe_provider_diagnostic(exc: BaseException) -> str:
    """Keep useful provider status text while removing common API-key patterns."""

    status = rag_db._status_code(exc)
    text = " ".join(str(exc).split())
    text = re.sub(r"sk-[0-9A-Za-z_-]{10,}", "[REDACTED_API_KEY]", text)
    text = re.sub(r"AIza[0-9A-Za-z_-]{10,}", "[REDACTED_API_KEY]", text)
    text = re.sub(
        r"(?i)(Incorrect API key provided:\s*)[^.\s,;}]+",
        r"\1[REDACTED]",
        text,
    )
    text = re.sub(
        r"(?i)(x-goog-api-key|api[_ -]?key|authorization)(\s*[:=]\s*)([^\s,;}]+)",
        r"\1\2[REDACTED]",
        text,
    )
    if len(text) > 800:
        text = text[:800] + "..."
    prefix = f"HTTP {status}" if status is not None else type(exc).__name__
    return f"{prefix}: {text}" if text else prefix

def _openai_generation_model() -> str:
    try:
        from petcare_agent.config import get_settings

        model = get_settings().openai_model
    except Exception:
        model = os.environ.get("OPENAI_MODEL", "")
    return model.strip() or GENERATION_MODEL


def _openai_client() -> Any:
    try:
        from petcare_agent.config import get_settings

        api_key = get_settings().openai_api_key
    except Exception:
        api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RagPipelineError("OPENAI_API_KEY is not configured.")
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RagPipelineError(
            "openai package is not installed. Install the project dependencies."
        ) from exc
    return OpenAI(api_key=api_key)


def validate_request(question: str, species: str, top_k: int) -> tuple[str, Species]:
    question = question.strip()
    if not question:
        raise RagPipelineError("질문은 비어 있을 수 없습니다.")
    if species not in {"dog", "cat"}:
        raise RagPipelineError("species는 dog 또는 cat이어야 합니다.")
    if top_k < 1:
        raise RagPipelineError("top-k는 1 이상이어야 합니다.")
    return question, species  # type: ignore[return-value]


def open_collection(
    db_path: Path = rag_db.DEFAULT_DB_PATH,
    collection_name: str = rag_db.DEFAULT_COLLECTION,
) -> Any:
    """Open the already-built collection and verify its embedding contract."""

    try:
        collection = rag_db.require_collection(
            rag_db.chroma_client(db_path), collection_name
        )
    except rag_db.RagDbError as exc:
        raise RagPipelineError(str(exc)) from exc
    except Exception as exc:
        raise RagPipelineError("ChromaDB를 열 수 없습니다. DB 경로와 파일 상태를 확인하세요.") from exc
    metadata = collection.metadata or {}
    if metadata.get("embedding_model") != rag_db.MODEL:
        raise RagPipelineError("DB의 임베딩 모델이 현재 질문 모델과 다릅니다.")
    if metadata.get("embedding_dimension") != rag_db.DIMENSION:
        raise RagPipelineError("DB embedding dimension does not match the current embedding model.")
    return collection


def embed_question(
    question: str,
    *,
    species: Species | None = None,
    embedding_client: Any | None = None,
    embedder: Embedder | None = None,
) -> list[float]:
    """Turn one question into a 1536-dimensional vector."""

    question = question.strip()
    if not question:
        raise RagPipelineError("질문은 비어 있을 수 없습니다.")
    prompt = rag_db.query_embedding_text(question, species=species)
    try:
        vector = (
            list(embedder(prompt))
            if embedder is not None
            else rag_db.embed_texts(
                embedding_client or rag_db.openai_client(), [prompt]
            )[0]
        )
        rag_db.validate_embeddings([vector], 1)
    except rag_db.RagDbError as exc:
        raise RagPipelineError(str(exc)) from exc
    except Exception as exc:
        raise RagPipelineError("질문 임베딩 생성에 실패했습니다.") from exc
    return vector


def retrieve(
    question: str,
    species: Species,
    top_k: int = DEFAULT_TOP_K,
    *,
    db_path: Path = rag_db.DEFAULT_DB_PATH,
    collection_name: str = rag_db.DEFAULT_COLLECTION,
    collection: Any | None = None,
    embedding_client: Any | None = None,
    embedder: Embedder | None = None,
) -> list[RetrievedChunk]:
    """Retrieve valid Cornell chunks after applying the mandatory species filter."""

    question, species = validate_request(question, species, top_k)
    collection = collection or open_collection(db_path, collection_name)
    vector = embed_question(
        question, species=species, embedding_client=embedding_client, embedder=embedder
    )
    try:
        raw_results = rag_db.query_collection(collection, vector, species, top_k)
    except rag_db.RagDbError as exc:
        raise RagPipelineError(str(exc)) from exc
    except Exception as exc:
        raise RagPipelineError("ChromaDB 검색에 실패했습니다.") from exc

    chunks: list[RetrievedChunk] = []
    for result in raw_results:
        metadata = result.metadata or {}
        url = metadata.get("canonical_url", "")
        result_species = metadata.get("species") or []
        content = result.document.strip() if isinstance(result.document, str) else ""
        if not content or not isinstance(url, str):
            continue
        if not url.startswith("https://www.vet.cornell.edu/"):
            continue
        if species not in result_species:
            continue
        chunks.append(
            RetrievedChunk(
                chunk_id=result.chunk_id,
                document_id=str(metadata.get("document_id", "")),
                title=str(metadata.get("title", "")),
                section_path=list(metadata.get("section_path") or []),
                species=list(result_species),
                canonical_url=url,
                content=content,
                distance=result.distance,
            )
        )
    return chunks


def build_context(chunks: Sequence[RetrievedChunk]) -> str:
    """Number retrieved chunks so the answer model can cite only known SOURCE IDs."""

    blocks = []
    for number, chunk in enumerate(chunks, start=1):
        section = " > ".join(chunk.section_path) or chunk.title
        blocks.append(
            "\n".join(
                [
                    f"[SOURCE {number}]",
                    f"Title: {chunk.title}",
                    f"Section: {section}",
                    f"Species: {', '.join(chunk.species)}",
                    f"URL: {chunk.canonical_url}",
                    "Content:",
                    chunk.content,
                ]
            )
        )
    return "\n\n".join(blocks)


def build_generation_prompt(
    question: str, species: Species, context: str
) -> str:
    return (
        f"대상 동물 종: {species}\n"
        f"보호자의 일반 건강정보 질문: {question}\n\n"
        "아래 SOURCE만 사용해 답하세요. SOURCE가 질문을 뒷받침하지 못하면 "
        "insufficient_evidence를 true로 설정하세요.\n\n"
        f"{context}"
    )


def _redacted_preview(text: str, limit: int = 500) -> str:
    preview = " ".join(text.split())
    preview = re.sub(r"AIza[0-9A-Za-z_-]{10,}", "[REDACTED_API_KEY]", preview)
    return preview[:limit] + ("..." if len(preview) > limit else "")


def _decode_json_object(text: str) -> dict[str, Any]:
    """Accept JSON with optional Markdown fences or harmless leading text."""

    decoder = json.JSONDecoder()
    candidates = [text.strip()]
    fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.IGNORECASE | re.DOTALL)
    if fenced:
        candidates.insert(0, fenced.group(1).strip())
    for candidate in candidates:
        for match in re.finditer(r"\{", candidate):
            try:
                payload, _ = decoder.raw_decode(candidate[match.start() :])
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                return payload
    raise RagPipelineError(
        "Answer model response is not the expected JSON shape.",
        diagnostic=f"응답 앞부분: {_redacted_preview(text)}",
    )


def _payload_from_response(response: Any) -> dict[str, Any]:
    parsed = getattr(response, "parsed", None)
    if isinstance(parsed, dict):
        return parsed
    if hasattr(parsed, "model_dump"):
        return parsed.model_dump()
    choices = getattr(response, "choices", None) or []
    if choices:
        message = getattr(choices[0], "message", None)
        parsed = getattr(message, "parsed", None)
        if isinstance(parsed, dict):
            return parsed
        if hasattr(parsed, "model_dump"):
            return parsed.model_dump()
        content = getattr(message, "content", None)
        if isinstance(content, str) and content.strip():
            return _decode_json_object(content)
    text = getattr(response, "text", None)
    if not isinstance(text, str) or not text.strip():
        raise RagPipelineError("Answer model returned an empty response.")
    try:
        return _decode_json_object(text)
    except RagPipelineError as exc:
        candidates = getattr(response, "candidates", None) or []
        finish_reason = getattr(candidates[0], "finish_reason", None) if candidates else None
        usage = getattr(response, "usage_metadata", None)
        answer_tokens = getattr(usage, "candidates_token_count", None)
        thought_tokens = getattr(usage, "thoughts_token_count", None)
        details = [exc.diagnostic or ""]
        if finish_reason is not None:
            details.append(f"finish_reason={finish_reason}")
        if answer_tokens is not None:
            details.append(f"answer_tokens={answer_tokens}")
        if thought_tokens is not None:
            details.append(f"thought_tokens={thought_tokens}")
        raise RagPipelineError(
            str(exc), diagnostic="; ".join(part for part in details if part)
        ) from exc


def validate_generated_answer(
    payload: dict[str, Any] | RagAnswer, source_count: int
) -> RagAnswer:
    """Reject hallucinated citations, URLs, and malformed model output."""

    if isinstance(payload, RagAnswer):
        raw = payload.to_dict()
    elif isinstance(payload, dict):
        raw = payload
    else:
        raise RagPipelineError("Answer model response shape is invalid.")

    answer = raw.get("answer")
    cited = raw.get("cited_source_numbers")
    insufficient = raw.get("insufficient_evidence")
    disclaimer = raw.get("disclaimer")
    if not isinstance(answer, str) or not isinstance(cited, list):
        raise RagPipelineError("Answer model response answer or citation array is invalid.")
    if not isinstance(insufficient, bool) or not isinstance(disclaimer, str):
        raise RagPipelineError("Answer model response insufficiency flag or disclaimer is invalid.")
    if any(isinstance(number, bool) or not isinstance(number, int) for number in cited):
        raise RagPipelineError("인용 번호는 정수여야 합니다.")

    if insufficient:
        return RagAnswer(
            answer=INSUFFICIENT_ANSWER,
            cited_source_numbers=[],
            insufficient_evidence=True,
            disclaimer=DEFAULT_DISCLAIMER,
        )

    answer = answer.strip()
    if not answer:
        raise RagPipelineError("근거가 충분하다고 표시했지만 답변이 비어 있습니다.")
    if re.search(r"https?://", answer, flags=re.IGNORECASE):
        raise RagPipelineError("Answer body included an arbitrary URL and was blocked.")
    invalid_declared = [number for number in cited if not 1 <= number <= source_count]
    if invalid_declared:
        raise RagPipelineError(
            f"존재하지 않는 SOURCE 번호를 인용했습니다: {invalid_declared}"
        )
    marker_sequence = [int(value) for value in re.findall(r"\[(\d+)\]", answer)]
    if not marker_sequence:
        raise RagPipelineError("답변 본문에 [번호] 형식의 인용이 없습니다.")
    invalid_markers = [
        number for number in marker_sequence if not 1 <= number <= source_count
    ]
    if invalid_markers:
        raise RagPipelineError(
            f"답변 본문이 존재하지 않는 SOURCE 번호를 인용했습니다: {invalid_markers}"
        )
    # The visible inline markers are the source of truth. The answer model occasionally
    # omits or duplicates the parallel JSON array even when its answer markers
    # are valid. Preserve first-appearance order and derive citations from it.
    normalized_citations = list(dict.fromkeys(marker_sequence))
    return RagAnswer(
        answer=answer,
        cited_source_numbers=normalized_citations,
        insufficient_evidence=False,
        disclaimer=DEFAULT_DISCLAIMER,
    )


def generate_answer(
    question: str,
    species: Species,
    chunks: Sequence[RetrievedChunk],
    *,
    generation_client: Any | None = None,
    generator: Generator | None = None,
) -> RagAnswer:
    """Generate a structured answer, or stop before OpenAI when evidence is empty."""

    question, species = validate_request(question, species, DEFAULT_TOP_K)
    if not chunks:
        return RagAnswer(
            answer=INSUFFICIENT_ANSWER,
            cited_source_numbers=[],
            insufficient_evidence=True,
            disclaimer=DEFAULT_DISCLAIMER,
        )
    context = build_context(chunks)
    prompt = build_generation_prompt(question, species, context)
    if generator is not None:
        return validate_generated_answer(
            generator(SYSTEM_INSTRUCTION, prompt, RAG_ANSWER_SCHEMA), len(chunks)
        )

    client = generation_client or _openai_client()
    model = _openai_generation_model()

    def request() -> Any:
        return client.beta.chat.completions.parse(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_INSTRUCTION},
                {"role": "user", "content": prompt},
            ],
            response_format=RagAnswerOutput,
        )

    try:
        response = rag_db.call_with_retry(request)
        payload = _payload_from_response(response)
        return validate_generated_answer(payload, len(chunks))
    except RagPipelineError:
        raise
    except Exception as exc:
        if rag_db.is_retryable(exc):
            raise RagPipelineError(
                "OpenAI answer generation failed temporarily. Please retry later.",
                diagnostic=_safe_provider_diagnostic(exc),
            ) from exc
        raise RagPipelineError(
            "OpenAI answer generation failed.",
            diagnostic=_safe_provider_diagnostic(exc),
        ) from exc


def build_response(
    question: str,
    species: Species,
    chunks: Sequence[RetrievedChunk],
    generated: RagAnswer,
) -> RagResponse:
    citations = [
        Citation(
            number=number,
            title=chunks[number - 1].title,
            section_path=chunks[number - 1].section_path,
            url=chunks[number - 1].canonical_url,
            chunk_id=chunks[number - 1].chunk_id,
        )
        for number in generated.cited_source_numbers
    ]
    return RagResponse(
        question=question,
        species=species,
        answer=generated.answer,
        insufficient_evidence=generated.insufficient_evidence,
        citations=citations,
        disclaimer=generated.disclaimer,
    )


def run_pipeline(
    question: str,
    species: Species,
    top_k: int = DEFAULT_TOP_K,
    *,
    db_path: Path = rag_db.DEFAULT_DB_PATH,
    collection_name: str = rag_db.DEFAULT_COLLECTION,
    collection: Any | None = None,
    embedding_client: Any | None = None,
    generation_client: Any | None = None,
    embedder: Embedder | None = None,
    generator: Generator | None = None,
) -> tuple[RagResponse, PipelineTrace]:
    question, species = validate_request(question, species, top_k)
    chunks = retrieve(
        question,
        species,
        top_k,
        db_path=db_path,
        collection_name=collection_name,
        collection=collection,
        embedding_client=embedding_client,
        embedder=embedder,
    )
    context = build_context(chunks)
    generation_prompt = build_generation_prompt(question, species, context)
    generated = generate_answer(
        question,
        species,
        chunks,
        generation_client=generation_client,
        generator=generator,
    )
    response = build_response(question, species, chunks, generated)
    trace = PipelineTrace(
        embedding_prompt=rag_db.query_embedding_text(question, species=species),
        retrieved_chunks=chunks,
        context=context,
        generation_prompt=generation_prompt,
        cited_source_numbers=generated.cited_source_numbers,
    )
    return response, trace


def answer_question(
    question: str,
    species: Species,
    top_k: int = DEFAULT_TOP_K,
    **kwargs: Any,
) -> RagResponse:
    """Public convenience API returning only the safe final response."""

    response, _ = run_pipeline(question, species, top_k, **kwargs)
    return response
