from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import pytest

from petcare_agent.api.handoff_context import (
    HANDOFF_CONTEXT_DAYS,
    HANDOFF_CONTEXT_PATH_TEMPLATE,
    ExistingAPIHandoffContextProvider,
    HandoffContextAPIError,
    build_handoff_context_path,
)
from petcare_agent.nodes.db_context_loader import load_db_context
from petcare_agent.schemas.graph_state import PetCareGraphState


class RecordingHandoffContextClient:
    def __init__(self, payload: Mapping[str, Any]) -> None:
        self.payload = payload
        self.calls: list[tuple[str, dict[str, int]]] = []

    def get_json(self, path: str, *, query: Mapping[str, int]) -> Mapping[str, Any]:
        self.calls.append((path, dict(query)))
        return self.payload


def test_existing_api_provider_uses_only_handoff_context_days_3_contract() -> None:
    payload = _handoff_context_payload()
    client = RecordingHandoffContextClient(payload)
    provider = ExistingAPIHandoffContextProvider(client=client)

    context = provider.load_context(42)

    assert client.calls == [("/api/pets/42/handoff-context", {"days": HANDOFF_CONTEXT_DAYS})]
    assert build_handoff_context_path(42) == "/api/pets/42/handoff-context"
    assert HANDOFF_CONTEXT_PATH_TEMPLATE == "/api/pets/{pet_id}/handoff-context"
    assert context.pet["name"] == "Kongi"
    assert len(context.recent_daily_entries) == 3
    assert context.diagnoses[0]["diagnosis"] == "patellar luxation stage 2"
    assert not hasattr(context, "generated_at")


def test_existing_api_provider_rejects_non_phase10_day_window() -> None:
    client = RecordingHandoffContextClient(_handoff_context_payload())
    provider = ExistingAPIHandoffContextProvider(client=client)

    with pytest.raises(HandoffContextAPIError):
        provider.load_context(42, days=30)

    assert client.calls == []


def test_db_context_loader_accepts_phase10_existing_api_provider_boundary() -> None:
    client = RecordingHandoffContextClient(_handoff_context_payload())
    provider = ExistingAPIHandoffContextProvider(client=client)
    state = PetCareGraphState(requires_db_context=True, pet_id=42)

    result = load_db_context(state, provider=provider)

    assert client.calls == [("/api/pets/42/handoff-context", {"days": 3})]
    assert result.context.pet["id"] == 42
    assert result.context.data_from == "2026-07-13"
    assert result.context.data_to == "2026-07-15"


def test_source_contains_no_undocumented_agent_api_endpoint_strings() -> None:
    source_root = Path(__file__).resolve().parents[1] / "src" / "petcare_agent"
    source_text = "\n".join(path.read_text(encoding="utf-8") for path in source_root.rglob("*.py"))

    assert "assessment-context" not in source_text
    assert "/daily-entries" not in source_text
    assert "/documents" not in source_text
    assert "/diagnoses" not in source_text
    assert "/api/pets/{pet_id}/handoff-context" in source_text


def _handoff_context_payload() -> dict[str, Any]:
    return {
        "pet": {
            "id": 42,
            "name": "Kongi",
            "species": "cat",
            "breed": "Korean shorthair",
            "birth_date": "2021-09-14",
            "sex": "male",
            "is_neutered": True,
            "weight_kg": 5.08,
            "size_class": "small",
            "diseases_medications_allergies": [],
        },
        "recent_daily_entries": [
            _daily_entry("2026-07-15"),
            _daily_entry("2026-07-14"),
            _daily_entry("2026-07-13"),
        ],
        "diagnoses": [
            {
                "id": 1,
                "date": "2026-07-02",
                "hospital": "Happy Animal Hospital",
                "diagnosis": "patellar luxation stage 2",
                "content": "Continue supplement.",
                "original_file_ref": "documents/diagnosis_001.pdf",
            }
        ],
        "unknown_items": [],
        "data_from": "2026-07-13",
        "data_to": "2026-07-15",
        "generated_at": "2026-07-15T16:35:00+09:00",
    }


def _daily_entry(record_date: str) -> dict[str, str]:
    return {
        "id": f"entry_{record_date}",
        "record_date": record_date,
        "food": "normal",
        "water": "normal",
        "activity": "normal",
        "symptom": "none",
        "stool": "normal",
        "vomit": "none",
        "notes": "Routine entry.",
    }
