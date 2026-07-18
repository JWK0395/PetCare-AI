from __future__ import annotations

from typing import Any, Callable, Protocol

from openai import OpenAI
from pydantic import BaseModel

from .config import Settings
from .models import RAGChunk


class OpenAIService:
    def __init__(self, api_key: str, model: str) -> None:
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def parse(
        self,
        *,
        schema: type[BaseModel],
        system_prompt: str,
        user_prompt: str,
    ) -> BaseModel:
        response = self.client.responses.parse(
            model=self.model,
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            text_format=schema,
        )
        if response.output_parsed is None:
            raise RuntimeError("OpenAI 구조화 출력이 비어 있습니다.")
        return response.output_parsed

    def text(self, *, system_prompt: str, user_prompt: str) -> str:
        response = self.client.responses.create(
            model=self.model,
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        if not response.output_text:
            raise RuntimeError("OpenAI 텍스트 출력이 비어 있습니다.")
        return response.output_text.strip()


_llm_service: OpenAIService | None = None


def get_llm_service() -> OpenAIService:
    global _llm_service

    if _llm_service is None:
        settings = Settings.from_env()
        _llm_service = OpenAIService(
            api_key=settings.openai_api_key,
            model=settings.openai_model,
        )

    return _llm_service


def set_llm_service(
    service: OpenAIService | Any,
) -> None:
    """테스트 또는 의존성 주입용 LLM 교체 함수."""
    global _llm_service
    _llm_service = service


class RAGProvider(Protocol):
    def search(
        self,
        *,
        query: str,
        pet_context: dict[str, Any],
        limit: int = 5,
    ) -> list[RAGChunk]:
        ...

class DemoRAGProvider:
    def __init__(self) -> None:
        self.documents = [
            RAGChunk(
                source_id="guide-001",
                title="검수 완료 반려동물 구토 보호자 안내",
                organization="PetCare AI Demo",
                version="2026.1",
                page=3,
                text=(
                    "구토가 반복되거나 기력 저하, 식욕 감소와 함께 나타나는 경우에는 "
                    "동물병원에 상담하는 것이 권장된다."
                ),
                score=0.95,
                metadata={"species": ["dog", "cat"], "topic": "vomiting"},
            ),
            RAGChunk(
                source_id="guide-002",
                title="검수 완료 반려동물 식욕 저하 안내",
                organization="PetCare AI Demo",
                version="2026.1",
                page=6,
                text=(
                    "평소보다 식사량이 줄었을 때는 지속 기간, 구토·설사 여부, "
                    "활동성 변화, 음수와 배뇨 상태를 함께 확인한다."
                ),
                score=0.91,
                metadata={"species": ["dog", "cat"], "topic": "appetite"},
            ),
        ]

    def search(
        self,
        *,
        query: str,
        pet_context: dict[str, Any],
        limit: int = 5,
    ) -> list[RAGChunk]:
        return self.documents[:limit]

class TeamRAGAdapter:
    def __init__(
        self,
        search_function: Callable[..., list[dict[str, Any]]],
    ) -> None:
        self.search_function = search_function

    def search(
        self,
        *,
        query: str,
        pet_context: dict[str, Any],
        limit: int = 5,
    ) -> list[RAGChunk]:
        raw_results = self.search_function(
            query=query,
            pet_context=pet_context,
            limit=limit,
        )
        return [RAGChunk.model_validate(item) for item in raw_results]

_rag_provider: RAGProvider = DemoRAGProvider()


def get_rag_provider() -> RAGProvider:
    return _rag_provider


def set_rag_provider(
    provider: RAGProvider,
) -> None:
    """팀 RAG 또는 테스트용 Provider 교체 함수."""
    global _rag_provider
    _rag_provider = provider
