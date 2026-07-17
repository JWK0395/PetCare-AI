"""Temporary DB context loader node.

Phase 4 intentionally uses a fixture-backed provider instead of HTTP/API calls.
The provider boundary mirrors the future handoff-context adapter shape so it can
be replaced without changing the node contract.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping, Protocol

from petcare_agent.schemas.graph_state import PetCareContext, PetCareGraphState

ContextPayload = PetCareContext | Mapping[str, Any]


class DBContextProvider(Protocol):
    """Boundary for loading pet handoff context.

    The production implementation should eventually call the existing
    `GET /api/pets/{pet_id}/handoff-context?days=3` API. The Phase 4
    implementation must not perform any external I/O.
    """

    def load_context(self, pet_id: int, *, days: int = 3) -> ContextPayload:
        """Return context for one pet and recent day window."""


class StaticDBContextProvider:
    """Fixture-backed DB context provider for Phase 4 tests and local use."""

    def __init__(self, fixtures: Mapping[int, ContextPayload] | None = None) -> None:
        self._fixtures = dict(fixtures or {})

    def load_context(self, pet_id: int, *, days: int = 3) -> PetCareContext:
        payload = self._fixtures.get(pet_id)
        if payload is None:
            payload = _default_fixture(pet_id)

        context = _coerce_context(payload)
        return context.model_copy(
            update={
                "recent_daily_entries": context.recent_daily_entries[:days],
            },
            deep=True,
        )


def load_db_context(
    state: PetCareGraphState,
    *,
    provider: DBContextProvider | None = None,
    days: int = 3,
) -> PetCareGraphState:
    """Load pet context for every graph turn that has a pet id."""

    next_state = state.model_copy(deep=True)
    if next_state.pet_id is None:
        next_state.context = _fallback_context(next_state.context, "pet_id_missing")
        return next_state

    context_provider = provider or StaticDBContextProvider()
    try:
        next_state.context = _coerce_context(
            context_provider.load_context(next_state.pet_id, days=days)
        )
    except Exception:
        next_state.context = _fallback_context(next_state.context, "db_context_unavailable")

    return next_state


def db_context_loader(
    state: PetCareGraphState,
    *,
    provider: DBContextProvider | None = None,
    days: int = 3,
) -> PetCareGraphState:
    """LangGraph-friendly alias for the DB context loader node."""

    return load_db_context(state, provider=provider, days=days)


def _coerce_context(payload: ContextPayload) -> PetCareContext:
    if isinstance(payload, PetCareContext):
        return payload.model_copy(deep=True)
    return PetCareContext.model_validate(deepcopy(dict(payload)))


def _fallback_context(existing_context: PetCareContext, reason: str) -> PetCareContext:
    unknown_items = list(existing_context.unknown_items)
    if reason not in unknown_items:
        unknown_items.append(reason)
    return PetCareContext(unknown_items=unknown_items)


def _default_fixture(pet_id: int) -> dict[str, Any]:
    return {
        "pet": {
            "id": pet_id,
            "name": "Kongi",
            "species": "dog",
            "breed": "Maltese",
            "birth_date": "2021-09-14",
            "sex": "male",
            "is_neutered": True,
            "weight_kg": 5.08,
            "size_class": "small",
            "diseases_medications_allergies": [
                {"type": "disease", "name": "patellar luxation stage 2"},
                {"type": "medication", "name": "joint supplement", "details": "once daily"},
                {"type": "allergy", "name": "chicken"},
            ],
        },
        "recent_daily_entries": [
            {
                "id": 101,
                "record_date": "2026-07-15",
                "food": "normal",
                "water": "normal",
                "activity": "normal",
                "symptom": "none",
                "stool": "normal",
                "vomit": "none",
                "notes": "No unusual changes.",
            },
            {
                "id": 100,
                "record_date": "2026-07-14",
                "food": "normal",
                "water": "normal",
                "activity": "normal",
                "symptom": "none",
                "stool": "normal",
                "vomit": "none",
                "notes": "Routine daily entry.",
            },
            {
                "id": 99,
                "record_date": "2026-07-13",
                "food": "normal",
                "water": "normal",
                "activity": "normal",
                "symptom": "none",
                "stool": "normal",
                "vomit": "none",
                "notes": "Routine daily entry.",
            },
        ],
        "diagnoses": [
            {
                "id": 1,
                "date": "2026-07-02",
                "hospital": "Happy Animal Hospital",
                "diagnosis": "patellar luxation stage 2",
                "content": "Continue joint supplement and avoid excessive jumping.",
                "original_file_ref": "documents/diagnosis_001.pdf",
            }
        ],
        "unknown_items": [],
        "data_from": "2026-07-13",
        "data_to": "2026-07-15",
    }
