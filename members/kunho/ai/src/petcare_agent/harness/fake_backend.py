"""Fake backend adapters backed by a harness data bundle."""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any

from petcare_agent.harness.data_bundle import DataBundle
from petcare_agent.schemas.graph_state import PetCareContext, RetrievedChunk


class DataBundleBackendProvider:
    """Serve documented handoff-context payloads from raw DB-style JSON."""

    def __init__(self, bundle: DataBundle) -> None:
        self.bundle = bundle

    def load_context(self, pet_id: int, *, days: int = 3) -> PetCareContext:
        """Return ``GET /api/pets/{pet_id}/handoff-context`` shaped context."""

        direct_context = self._load_direct_handoff_context(pet_id)
        if direct_context is not None:
            return _slice_context(direct_context, days)

        pet = _find_by_id(self.bundle.pets, pet_id)
        unknown_items: list[str] = []
        if pet is None:
            unknown_items.append("pet_not_found")
            pet = {"id": pet_id}

        daily_entries = _recent_daily_entries(self.bundle.daily_entries, pet_id, days)
        diagnoses = _diagnoses_for_pet(self.bundle.diagnoses, pet_id)
        data_from, data_to = _data_window(daily_entries)

        return PetCareContext(
            pet=_handoff_pet(pet),
            recent_daily_entries=[_handoff_daily_entry(entry) for entry in daily_entries],
            diagnoses=[_handoff_diagnosis(diagnosis) for diagnosis in diagnoses],
            medical_background=_medical_background_from_pet(pet),
            unknown_items=unknown_items,
            data_from=data_from,
            data_to=data_to,
        )

    def _load_direct_handoff_context(self, pet_id: int) -> PetCareContext | None:
        contexts = self.bundle.handoff_contexts
        if contexts is None:
            return None

        payload: Any | None = None
        if isinstance(contexts, list):
            payload = _find_handoff_context(contexts, pet_id)
        elif isinstance(contexts, dict):
            if "handoff_contexts" in contexts and isinstance(contexts["handoff_contexts"], list):
                payload = _find_handoff_context(contexts["handoff_contexts"], pet_id)
            else:
                payload = contexts.get(str(pet_id)) or contexts.get(pet_id)
                if payload is None and _context_pet_id(contexts) == pet_id:
                    payload = contexts

        if not isinstance(payload, dict):
            return None
        return _context_from_endpoint_payload(payload)


class DataBundleRAGAdapter:
    """RAG adapter backed by optional ``rag/chunks.json`` fixture data."""

    def __init__(self, bundle: DataBundle) -> None:
        self.bundle = bundle

    def retrieve(
        self,
        query: str,
        filters: dict[str, Any],
        top_k: int = 5,
    ) -> list[RetrievedChunk]:
        chunks = [
            chunk
            for chunk in self.bundle.rag_chunks
            if _chunk_matches_filters(chunk, filters)
        ]
        chunks.sort(key=lambda chunk: float(chunk.get("score") or 0), reverse=True)
        return [RetrievedChunk.model_validate(chunk) for chunk in chunks[:top_k]]


def _context_from_endpoint_payload(payload: dict[str, Any]) -> PetCareContext:
    allowed_keys = {
        "pet",
        "recent_daily_entries",
        "diagnoses",
        "medical_background",
        "unknown_items",
        "data_from",
        "data_to",
    }
    return PetCareContext.model_validate(
        {key: value for key, value in payload.items() if key in allowed_keys}
    )


def _slice_context(context: PetCareContext, days: int) -> PetCareContext:
    if days <= 0:
        recent_daily_entries: list[dict[str, Any]] = []
    else:
        recent_daily_entries = context.recent_daily_entries[:days]
    return context.model_copy(
        update={"recent_daily_entries": recent_daily_entries},
        deep=True,
    )


def _find_by_id(records: list[dict[str, Any]], record_id: int) -> dict[str, Any] | None:
    for record in records:
        if _int_or_none(record.get("id") or record.get("pet_id")) == record_id:
            return dict(record)
    return None


def _find_handoff_context(contexts: list[Any], pet_id: int) -> dict[str, Any] | None:
    for context in contexts:
        if isinstance(context, dict) and _context_pet_id(context) == pet_id:
            return context
    return None


def _context_pet_id(context: dict[str, Any]) -> int | None:
    pet = context.get("pet")
    if isinstance(pet, dict):
        return _int_or_none(pet.get("id") or pet.get("pet_id"))
    return _int_or_none(context.get("pet_id"))


def _recent_daily_entries(
    entries: list[dict[str, Any]],
    pet_id: int,
    days: int,
) -> list[dict[str, Any]]:
    pet_entries = [
        dict(entry)
        for entry in entries
        if _int_or_none(entry.get("pet_id")) == pet_id
    ]
    pet_entries.sort(key=lambda entry: _date_sort_key(entry.get("record_date")), reverse=True)
    if days <= 0:
        return []
    if not pet_entries:
        return []

    latest = _parse_date(pet_entries[0].get("record_date"))
    if latest is None:
        return pet_entries[:days]

    start = latest - timedelta(days=days - 1)
    in_window = [
        entry
        for entry in pet_entries
        if (parsed := _parse_date(entry.get("record_date"))) is not None
        and start <= parsed <= latest
    ]
    return in_window[:days]


def _diagnoses_for_pet(
    diagnoses: list[dict[str, Any]],
    pet_id: int,
    *,
    limit: int = 20,
) -> list[dict[str, Any]]:
    pet_diagnoses = [
        dict(diagnosis)
        for diagnosis in diagnoses
        if _int_or_none(diagnosis.get("pet_id")) == pet_id
    ]
    pet_diagnoses.sort(key=lambda diagnosis: _date_sort_key(diagnosis.get("date")), reverse=True)
    return pet_diagnoses[:limit]


def _data_window(entries: list[dict[str, Any]]) -> tuple[str, str]:
    dates = [
        parsed
        for entry in entries
        if (parsed := _parse_date(entry.get("record_date"))) is not None
    ]
    if not dates:
        return "", ""
    return min(dates).isoformat(), max(dates).isoformat()


def _handoff_pet(pet: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "id",
        "name",
        "species",
        "breed",
        "birth_date",
        "sex",
        "is_neutered",
        "weight_kg",
        "size_class",
        "diseases_medications_allergies",
    ]
    return {key: pet[key] for key in keys if key in pet}


def _handoff_daily_entry(entry: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "id",
        "record_date",
        "food",
        "water",
        "activity",
        "symptom",
        "stool",
        "vomit",
        "notes",
    ]
    return {key: entry[key] for key in keys if key in entry}


def _handoff_diagnosis(diagnosis: dict[str, Any]) -> dict[str, Any]:
    keys = ["id", "date", "hospital", "diagnosis", "content", "original_file_ref"]
    return {key: diagnosis[key] for key in keys if key in diagnosis}


def _medical_background_from_pet(pet: dict[str, Any]) -> dict[str, Any]:
    items = pet.get("diseases_medications_allergies") or []
    if not isinstance(items, list):
        return {}

    conditions: list[str] = []
    medications_or_supplements: list[str] = []
    allergies: list[str] = []

    for item in items:
        if not isinstance(item, dict):
            continue
        item_type = str(item.get("type") or "").strip().lower()
        name = str(item.get("name") or "").strip()
        if not name:
            continue
        details = str(item.get("details") or "").strip()
        value = f"{name} ({details})" if details else name
        if item_type == "disease":
            conditions.append(value)
        elif item_type in {"medication", "supplement"}:
            medications_or_supplements.append(value)
        elif item_type == "allergy":
            allergies.append(value)

    return {
        "conditions": conditions,
        "medications_or_supplements": medications_or_supplements,
        "allergies": allergies,
    }


def _chunk_matches_filters(chunk: dict[str, Any], filters: dict[str, Any]) -> bool:
    metadata = chunk.get("metadata") if isinstance(chunk.get("metadata"), dict) else {}
    species_filter = str(filters.get("species") or "").strip().lower()
    chunk_species = str(metadata.get("species") or "").strip().lower()
    if species_filter in {"cat", "dog"} and chunk_species and chunk_species != species_filter:
        return False

    complaint_filter = str(filters.get("chief_complaint") or "").strip().lower()
    if complaint_filter:
        searchable = " ".join(
            [
                str(metadata.get("chief_complaint") or ""),
                str(metadata.get("topic") or ""),
                str(chunk.get("title") or ""),
                str(chunk.get("text") or ""),
            ]
        ).lower()
        if complaint_filter not in searchable:
            return False
    return True


def _date_sort_key(value: Any) -> tuple[int, str]:
    parsed = _parse_date(value)
    if parsed is None:
        return (0, str(value or ""))
    return (1, parsed.isoformat())


def _parse_date(value: Any) -> date | None:
    if not value:
        return None
    try:
        return date.fromisoformat(str(value)[:10])
    except ValueError:
        return None


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


__all__ = ["DataBundleBackendProvider", "DataBundleRAGAdapter"]
