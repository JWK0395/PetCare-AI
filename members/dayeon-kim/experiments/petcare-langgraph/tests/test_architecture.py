from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from petcare_agent.models import (
    HandoffDocument,
    QuestionStrategy,
)
from petcare_agent.nodes.agents import rag_agent
from petcare_agent.runtime import request_to_initial_state
from petcare_agent.services import (
    AgentDependencies,
    NullRAGProvider,
)


class FakeLLM:
    def text(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
    ) -> str:
        return f"{system_prompt[:8]}:{user_prompt[:8]}"


class FailingRAG:
    def search(
        self,
        *,
        query: str,
        pet_context: dict[str, Any],
        limit: int = 5,
    ) -> list[Any]:
        raise TimeoutError("timeout")


def _request() -> dict[str, Any]:
    return {
        "session_id": "location-test",
        "pet_id": 204,
        "user_input": "숨이 힘들어 보여",
        "context": {
            "pet": {
                "id": 204,
                "name": "보리",
                "species": "dog",
            },
            "daily_entries": [],
            "diagnoses": [],
        },
        "location": {
            "latitude": 35.1796,
            "longitude": 129.0756,
            "address": "부산광역시",
        },
    }


def test_location_is_copied_to_initial_state() -> None:
    _, state = request_to_initial_state(
        _request(),
        previous_session={},
    )
    assert state["location"] == {
        "latitude": 35.1796,
        "longitude": 129.0756,
        "address": "부산광역시",
    }


def test_null_rag_is_default_and_returns_no_fake_evidence() -> None:
    dependencies = AgentDependencies(llm=FakeLLM())
    assert isinstance(dependencies.rag, NullRAGProvider)
    assert dependencies.rag.search(
        query="설사",
        pet_context={},
    ) == []


def test_rag_failure_is_non_fatal() -> None:
    dependencies = AgentDependencies(
        llm=FakeLLM(),
        rag=FailingRAG(),
    )
    result = rag_agent(
        {
            "user_input": "설사를 해요",
            "assessment": {
                "symptoms": [
                    {
                        "code": "diarrhea",
                        "evidence": "설사",
                        "negated": False,
                    }
                ]
            },
            "backend_context": {},
            "latency_ms": {},
            "warnings": [],
            "errors": [],
        },
        dependencies=dependencies,
    )
    assert result["rag_done"] is True
    assert result["rag_status"] == "failed"
    assert result["rag_chunks"] == []
    assert result["warnings"]
    assert "errors" not in result


def test_internal_models_forbid_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        QuestionStrategy.model_validate(
            {
                "completed_cycle": [],
            }
        )

    with pytest.raises(ValidationError):
        HandoffDocument.model_validate(
            {
                "document_info": {},
                "pet_info": {},
                "status": {},
                "clinical_summary": {},
                "unexpected": True,
            }
        )


def test_dependencies_are_instance_scoped() -> None:
    first = AgentDependencies(llm=FakeLLM())
    second = AgentDependencies(llm=FakeLLM())
    first.rag = FailingRAG()

    assert isinstance(second.rag, NullRAGProvider)
    assert first.rag is not second.rag
