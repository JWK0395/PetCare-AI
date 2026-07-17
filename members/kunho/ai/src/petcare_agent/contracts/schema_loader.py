"""Helpers for loading repository JSON schema contracts in tests/runtime."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Literal

ContractSchemaName = Literal[
    "agent_graph_request",
    "agent_graph_response",
    "triage_checklist",
    "llm_structured_outputs",
    "hospital_handoff_summary",
    "internal_triage_assessment",
]

CONTRACT_SCHEMA_FILES: dict[ContractSchemaName, str] = {
    "agent_graph_request": "agent-graph-request.schema.json",
    "agent_graph_response": "agent-graph-response.schema.json",
    "triage_checklist": "triage-checklist.schema.json",
    "llm_structured_outputs": "llm-structured-outputs.schema.json",
    "hospital_handoff_summary": "hospital-handoff-summary.schema.json",
    "internal_triage_assessment": "internal-triage-assessment.schema.json",
}


def contracts_root() -> Path:
    """Resolve the repository contracts directory.

    PETCARE_CONTRACTS_DIR may point either at contracts/ or at
    contracts/jsonschema/. Without it, walk up from this module to find the
    checked-in contracts/jsonschema directory.
    """

    override = os.getenv("PETCARE_CONTRACTS_DIR")
    if override:
        path = Path(override).expanduser().resolve()
        return path.parent if path.name == "jsonschema" else path

    current = Path(__file__).resolve()
    for parent in current.parents:
        candidate = parent / "contracts" / "jsonschema"
        if candidate.exists():
            return parent / "contracts"

    raise FileNotFoundError("Could not locate contracts/jsonschema")


def jsonschema_dir() -> Path:
    """Return the directory containing JSON schema contract files."""

    return contracts_root() / "jsonschema"


def load_json_schema(name: ContractSchemaName) -> dict[str, Any]:
    """Load one named JSON schema contract."""

    path = jsonschema_dir() / CONTRACT_SCHEMA_FILES[name]
    with path.open("r", encoding="utf-8-sig") as schema_file:
        schema = json.load(schema_file)
    if not isinstance(schema, dict):
        raise TypeError(f"JSON schema must be an object: {path}")
    return schema

