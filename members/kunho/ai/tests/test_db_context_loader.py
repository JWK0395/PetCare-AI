from __future__ import annotations

from typing import Any

from petcare_agent.nodes.db_context_loader import load_db_context
from petcare_agent.schemas.graph_state import PetCareContext, PetCareGraphState


class CountingProvider:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload
        self.calls = 0

    def load_context(self, pet_id: int, *, days: int = 3) -> dict[str, Any]:
        self.calls += 1
        return self.payload


class FailingProvider:
    def __init__(self) -> None:
        self.calls = 0

    def load_context(self, pet_id: int, *, days: int = 3) -> dict[str, Any]:
        self.calls += 1
        raise RuntimeError("fixture provider failed")


def test_db_context_loader_loads_provider_even_when_context_not_required() -> None:
    provider = CountingProvider(payload={"pet": {"id": 1, "name": "Kongi"}})
    original_context = PetCareContext(pet={"id": 99, "name": "Existing"})
    state = PetCareGraphState(
        requires_db_context=False,
        pet_id=1,
        context=original_context,
    )

    result = load_db_context(state, provider=provider)

    assert provider.calls == 1
    assert result.context.pet == {"id": 1, "name": "Kongi"}


def test_db_context_loader_populates_state_context_from_fixture_provider() -> None:
    payload = {
        "pet": {"id": 1, "name": "Kongi", "species": "dog"},
        "recent_daily_entries": [
            {"record_date": "2026-07-15", "food": "normal"},
            {"record_date": "2026-07-14", "food": "normal"},
        ],
        "diagnoses": [{"id": 1, "diagnosis": "patellar luxation"}],
        "unknown_items": ["weight_history"],
        "data_from": "2026-07-13",
        "data_to": "2026-07-15",
    }
    provider = CountingProvider(payload=payload)
    state = PetCareGraphState(requires_db_context=True, pet_id=1)

    result = load_db_context(state, provider=provider)

    assert provider.calls == 1
    assert result.context.pet["name"] == "Kongi"
    assert result.context.recent_daily_entries == payload["recent_daily_entries"]
    assert result.context.diagnoses == payload["diagnoses"]
    assert result.context.unknown_items == ["weight_history"]
    assert result.context.data_from == "2026-07-13"
    assert result.context.data_to == "2026-07-15"


def test_db_context_loader_gracefully_falls_back_when_provider_fails() -> None:
    provider = FailingProvider()
    state = PetCareGraphState(requires_db_context=True, pet_id=1)

    result = load_db_context(state, provider=provider)

    assert provider.calls == 1
    assert result.context.pet == {}
    assert result.context.recent_daily_entries == []
    assert "db_context_unavailable" in result.context.unknown_items
