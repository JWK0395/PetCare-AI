"""Small FastAPI boundary around the Cornell RAG pipeline.

The HTTP layer deliberately accepts only a question and a species. Pet profiles,
medical records, and daily logs belong to the future Context/Trend components and
must not be sent to this official-source retrieval service.
"""

from __future__ import annotations

import hmac
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal

from fastapi import FastAPI, Header, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator

from tools import manage_cornell_rag_db as rag_db

from .models import RagResponse
from .pipeline import RagPipelineError, answer_question, open_collection


Answerer = Callable[..., RagResponse]
ReadinessChecker = Callable[[], "Readiness"]


class CitationPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    number: int
    title: str
    section_path: list[str]
    url: str
    chunk_id: str


class RagAnswerRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=1, max_length=2000)
    species: Literal["dog", "cat"]
    top_k: int = Field(default=5, ge=1, le=10)

    @field_validator("question")
    @classmethod
    def question_must_not_be_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("질문은 비어 있을 수 없습니다.")
        return value


class RagAnswerPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str
    species: Literal["dog", "cat"]
    answer: str
    insufficient_evidence: bool
    citations: list[CitationPayload]
    disclaimer: str


class HealthPayload(BaseModel):
    status: Literal["ok"] = "ok"


class ReadyPayload(BaseModel):
    status: Literal["ready", "not_ready"]
    checks: dict[str, bool]
    message: str


@dataclass(frozen=True)
class Readiness:
    ready: bool
    checks: dict[str, bool]
    message: str


def _error(status_code: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"error": {"code": code, "message": message}},
    )


def check_readiness(
    db_path: Path = rag_db.DEFAULT_DB_PATH,
    collection_name: str = rag_db.DEFAULT_COLLECTION,
) -> Readiness:
    """Check local configuration without spending an OpenAI API request."""

    checks = {
        "openai_api_key_configured": bool(os.environ.get("OPENAI_API_KEY", "").strip()),
        "service_token_configured": bool(
            os.environ.get("PETCARE_RAG_SERVICE_TOKEN", "").strip()
        ),
        "database_exists": db_path.exists(),
        "collection_compatible": False,
        "chunk_count_is_732": False,
    }
    if checks["database_exists"]:
        try:
            collection = open_collection(db_path, collection_name)
            checks["collection_compatible"] = True
            checks["chunk_count_is_732"] = collection.count() == rag_db.EXPECTED_CHUNKS
        except RagPipelineError:
            pass

    ready = all(checks.values())
    message = (
        "RAG API가 질문을 받을 준비가 되었습니다."
        if ready
        else "RAG API 설정이 아직 완전하지 않습니다. checks의 false 항목을 확인하세요."
    )
    return Readiness(ready=ready, checks=checks, message=message)


def create_app(
    *,
    answerer: Answerer = answer_question,
    readiness_checker: ReadinessChecker | None = None,
    db_path: Path | None = None,
    collection_name: str | None = None,
) -> FastAPI:
    resolved_db_path = db_path or Path(
        os.environ.get("PETCARE_RAG_DB_PATH", str(rag_db.DEFAULT_DB_PATH))
    )
    resolved_collection = collection_name or os.environ.get(
        "PETCARE_RAG_COLLECTION", rag_db.DEFAULT_COLLECTION
    )
    readiness_checker = readiness_checker or (
        lambda: check_readiness(resolved_db_path, resolved_collection)
    )

    app = FastAPI(
        title="PetCare AI Cornell RAG API",
        version="1.0.0",
        description=(
            "Cornell 공식 건강자료를 검색해 출처가 포함된 일반 건강정보를 반환합니다. "
            "응급 판단, 진단, 처방과 개인 기록 분석은 수행하지 않습니다."
        ),
    )
    app.state.answerer = answerer
    app.state.readiness_checker = readiness_checker
    app.state.db_path = resolved_db_path
    app.state.collection_name = resolved_collection

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(
        _request: Request, _exc: RequestValidationError
    ) -> JSONResponse:
        return _error(
            422,
            "invalid_request",
            "question, species(dog 또는 cat), top_k(1~10)를 확인하세요.",
        )

    @app.get("/health", response_model=HealthPayload, tags=["status"])
    def health() -> HealthPayload:
        return HealthPayload()

    @app.get("/ready", response_model=ReadyPayload, tags=["status"])
    def ready() -> ReadyPayload | JSONResponse:
        result = app.state.readiness_checker()
        payload = ReadyPayload(
            status="ready" if result.ready else "not_ready",
            checks=result.checks,
            message=result.message,
        )
        if result.ready:
            return payload
        return JSONResponse(status_code=503, content=payload.model_dump())

    @app.post(
        "/v1/rag/answer",
        response_model=RagAnswerPayload,
        tags=["rag"],
    )
    def answer(
        body: RagAnswerRequest,
        x_petcare_token: str | None = Header(default=None, alias="X-PetCare-Token"),
    ) -> RagAnswerPayload | JSONResponse:
        expected_token = os.environ.get("PETCARE_RAG_SERVICE_TOKEN", "").strip()
        if not expected_token:
            return _error(
                503,
                "service_not_configured",
                "서버의 PETCARE_RAG_SERVICE_TOKEN이 설정되지 않았습니다.",
            )
        if not x_petcare_token or not hmac.compare_digest(
            x_petcare_token, expected_token
        ):
            return _error(401, "unauthorized", "유효한 서비스 토큰이 필요합니다.")

        try:
            response = app.state.answerer(
                question=body.question,
                species=body.species,
                top_k=body.top_k,
                db_path=app.state.db_path,
                collection_name=app.state.collection_name,
            )
            return RagAnswerPayload.model_validate(response.to_dict())
        except RagPipelineError as exc:
            return _error(503, "rag_unavailable", str(exc))
        except Exception:
            return _error(
                500,
                "internal_error",
                "RAG 서버에서 예상하지 못한 오류가 발생했습니다.",
            )

    return app


app = create_app()
