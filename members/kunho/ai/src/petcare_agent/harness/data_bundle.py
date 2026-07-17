"""Load local data bundles used by the agent harness.

The bundle can be either a ``data.zip`` file or an unpacked directory. JSON
files are addressed by their path inside the bundle, so teams can review sample
fixtures as plain folders and zip the same contents for ad-hoc sharing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any
from zipfile import ZipFile


@dataclass(frozen=True)
class DataBundle:
    """Parsed harness data bundle."""

    source: Path
    manifest: dict[str, Any] = field(default_factory=dict)
    pets: list[dict[str, Any]] = field(default_factory=list)
    daily_entries: list[dict[str, Any]] = field(default_factory=list)
    diagnoses: list[dict[str, Any]] = field(default_factory=list)
    handoff_contexts: Any | None = None
    rag_chunks: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def load(cls, source: str | Path) -> "DataBundle":
        """Load a bundle from a zip file or directory."""

        source_path = Path(source)
        if not source_path.exists():
            raise FileNotFoundError(f"Data bundle does not exist: {source_path}")

        json_files = _load_json_files(source_path)
        manifest = _read_optional_object(json_files, ["manifest.json"])
        pets = _read_records(json_files, ["db/pets.json", "pets.json"], "pets")
        daily_entries = _read_records(
            json_files,
            ["db/daily_entries.json", "daily_entries.json"],
            "daily_entries",
        )
        diagnosis_records_path = "/".join(("db", "diagnoses.json"))
        diagnoses = _read_records(
            json_files,
            [diagnosis_records_path, "diagnoses.json"],
            "diagnoses",
        )
        handoff_contexts = _read_optional(
            json_files,
            ["api/handoff_contexts.json", "handoff_contexts.json"],
        )
        rag_chunks = _read_records(
            json_files,
            ["rag/chunks.json", "rag_chunks.json"],
            "chunks",
        )

        return cls(
            source=source_path,
            manifest=manifest,
            pets=pets,
            daily_entries=daily_entries,
            diagnoses=diagnoses,
            handoff_contexts=handoff_contexts,
            rag_chunks=rag_chunks,
        )

    @property
    def pet_ids(self) -> list[int]:
        """Return pet ids available from raw DB fixtures."""

        ids: list[int] = []
        for pet in self.pets:
            pet_id = _int_or_none(pet.get("id") or pet.get("pet_id"))
            if pet_id is not None:
                ids.append(pet_id)
        return sorted(set(ids))


def _load_json_files(source: Path) -> dict[str, Any]:
    if source.is_dir():
        return _load_json_files_from_directory(source)
    if source.is_file() and source.suffix.lower() == ".zip":
        return _load_json_files_from_zip(source)
    raise ValueError(f"Expected a data.zip file or directory: {source}")


def _load_json_files_from_directory(source: Path) -> dict[str, Any]:
    files: dict[str, Any] = {}
    for json_path in source.rglob("*.json"):
        relative_path = json_path.relative_to(source).as_posix()
        files[_normalize_path(relative_path)] = _parse_json(
            json_path.read_text(encoding="utf-8-sig"),
            relative_path,
        )
    return files


def _load_json_files_from_zip(source: Path) -> dict[str, Any]:
    files: dict[str, Any] = {}
    with ZipFile(source) as archive:
        for member in archive.infolist():
            if member.is_dir() or not member.filename.lower().endswith(".json"):
                continue
            name = _normalize_path(member.filename)
            with archive.open(member) as file_handle:
                text = file_handle.read().decode("utf-8-sig")
            files[name] = _parse_json(text, member.filename)
    return files


def _parse_json(text: str, label: str) -> Any:
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in data bundle file {label}") from exc


def _read_optional_object(
    files: dict[str, Any],
    candidates: list[str],
) -> dict[str, Any]:
    payload = _read_optional(files, candidates)
    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise ValueError(f"{candidates[0]} must contain a JSON object")
    return dict(payload)


def _read_records(
    files: dict[str, Any],
    candidates: list[str],
    key: str,
) -> list[dict[str, Any]]:
    payload = _read_optional(files, candidates)
    if payload is None:
        return []
    if isinstance(payload, list):
        return [_coerce_record(record, candidates[0]) for record in payload]
    if isinstance(payload, dict) and isinstance(payload.get(key), list):
        return [_coerce_record(record, candidates[0]) for record in payload[key]]
    raise ValueError(f"{candidates[0]} must contain a JSON array or a '{key}' array")


def _read_optional(files: dict[str, Any], candidates: list[str]) -> Any | None:
    for candidate in candidates:
        normalized = _normalize_path(candidate)
        if normalized in files:
            return files[normalized]

    for candidate in candidates:
        normalized = _normalize_path(candidate)
        matches = [
            payload
            for path, payload in files.items()
            if path.endswith(f"/{normalized}") or path == normalized
        ]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise ValueError(f"Multiple bundle files match {candidate}")
    return None


def _coerce_record(record: Any, label: str) -> dict[str, Any]:
    if not isinstance(record, dict):
        raise ValueError(f"{label} records must be JSON objects")
    return dict(record)


def _normalize_path(path: str) -> str:
    return path.replace("\\", "/").strip("/")


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
