"""Load and select MVP safety checklist templates.

The Phase 2 contract is data-only: this module validates packaged checklist
templates and selects the safest matching template. It does not validate risk,
call an LLM, call RAG, or touch DB/API boundaries.
"""

from __future__ import annotations

import json
from functools import lru_cache
from importlib import resources
from typing import Any

from petcare_agent.schemas.common import Species
from petcare_agent.schemas.triage import ChecklistTemplate

CHECKLIST_PACKAGE = "petcare_agent.safety.checklists"
FALLBACK_CHECKLIST_ID = "breathing_triage"
COMMON_SPECIES = "cat/dog"

CHIEF_COMPLAINT_ALIASES = {
    "breath": "breathing",
    "breathing": "breathing",
    "breathing_issue": "breathing",
    "breathing_problem": "breathing",
    "cough": "cough",
    "coughing": "cough",
    "diarrhea": "diarrhea",
    "diarrhoea": "diarrhea",
    "dyspnea": "breathing",
    "ingestion": "toxicity",
    "pee": "urinary",
    "peeing": "urinary",
    "poison": "toxicity",
    "poisoning": "toxicity",
    "respiratory": "breathing",
    "seizure": "seizure",
    "seizures": "seizure",
    "toxin": "toxicity",
    "toxicity": "toxicity",
    "urinary": "urinary",
    "urination": "urinary",
    "vomit": "vomiting",
    "vomiting": "vomiting",
}

SPECIES_ALIASES = {
    "canine": "dog",
    "cat": "cat",
    "dog": "dog",
    "feline": "cat",
}


class ChecklistLoaderError(RuntimeError):
    """Raised when packaged checklist data cannot form a valid registry."""


def normalize_species(species: str | None) -> Species:
    """Normalize user/model species labels into the graph species contract."""

    if species is None:
        return "unknown"
    normalized = SPECIES_ALIASES.get(species.strip().lower())
    if normalized in {"cat", "dog"}:
        return normalized
    return "unknown"


def normalize_chief_complaint(chief_complaint: str | None) -> str | None:
    """Normalize a chief complaint label for checklist lookup."""

    if chief_complaint is None:
        return None
    key = chief_complaint.strip().lower().replace("-", "_").replace(" ", "_")
    return CHIEF_COMPLAINT_ALIASES.get(key)


@lru_cache(maxsize=1)
def _template_registry() -> dict[str, ChecklistTemplate]:
    registry: dict[str, ChecklistTemplate] = {}
    checklist_root = resources.files(CHECKLIST_PACKAGE)

    for template_file in sorted(checklist_root.iterdir(), key=lambda path: path.name):
        if template_file.name.startswith("_") or template_file.suffix != ".json":
            continue
        with template_file.open("r", encoding="utf-8") as file:
            payload = json.load(file)

        entries = payload if isinstance(payload, list) else [payload]
        for entry in entries:
            if not isinstance(entry, dict):
                raise ChecklistLoaderError(
                    f"Checklist template entry must be an object: {template_file.name}"
                )
            template = ChecklistTemplate.model_validate(entry)
            if template.checklist_id in registry:
                raise ChecklistLoaderError(f"Duplicate checklist_id: {template.checklist_id}")
            registry[template.checklist_id] = template

    if FALLBACK_CHECKLIST_ID not in registry:
        raise ChecklistLoaderError(f"Missing fallback checklist: {FALLBACK_CHECKLIST_ID}")
    return registry


def load_all_checklist_templates() -> dict[str, ChecklistTemplate]:
    """Return all validated checklist templates keyed by checklist_id."""

    return {
        checklist_id: template.model_copy(deep=True)
        for checklist_id, template in _template_registry().items()
    }


def load_checklist_template(checklist_id: str) -> ChecklistTemplate:
    """Return one validated checklist template by id."""

    try:
        return _template_registry()[checklist_id].model_copy(deep=True)
    except KeyError as exc:
        raise ChecklistLoaderError(f"Unknown checklist_id: {checklist_id}") from exc


def select_checklist_template(
    species: str | None,
    chief_complaint: str | None,
) -> ChecklistTemplate:
    """Select a checklist by species and chief complaint.

    Matching order:
    1. exact species and complaint, such as cat+cough or dog+cough
    2. common cat/dog complaint templates
    3. conservative fallback to breathing_triage

    The fallback is intentionally conservative because the breathing template
    contains the highest-priority immediate red flags shared by triage flows.
    """

    normalized_species = normalize_species(species)
    normalized_complaint = normalize_chief_complaint(chief_complaint)
    templates = _template_registry()

    if normalized_complaint is not None:
        exact_match = _find_template(
            templates,
            species=normalized_species,
            chief_complaint=normalized_complaint,
        )
        if exact_match is not None:
            return exact_match.model_copy(deep=True)

        common_match = _find_template(
            templates,
            species=COMMON_SPECIES,
            chief_complaint=normalized_complaint,
        )
        if common_match is not None:
            return common_match.model_copy(deep=True)

    return templates[FALLBACK_CHECKLIST_ID].model_copy(deep=True)


def _find_template(
    templates: dict[str, ChecklistTemplate],
    *,
    species: str,
    chief_complaint: str,
) -> ChecklistTemplate | None:
    for template in templates.values():
        if template.species == species and template.chief_complaint == chief_complaint:
            return template
    return None


def checklist_template_as_contract_dict(template: ChecklistTemplate) -> dict[str, Any]:
    """Serialize a template with the exact contract-facing field names."""

    return template.model_dump(mode="json")

