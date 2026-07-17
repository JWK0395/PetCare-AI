"""Pure Phase 3 safety rule definitions.

Rules in this module only inspect checklist item values, item confidence, the
resolved species, and the safety question turn count. They do not call DB/API,
RAG, LLMs, or graph nodes.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

from petcare_agent.schemas.common import RiskAction, RiskLevel, Species
from petcare_agent.schemas.triage import ChecklistItem, RuleHit

RuleResult = RiskLevel | RiskAction


@dataclass(frozen=True)
class RuleFacts:
    """Checklist-only facts available to Phase 3 safety rules."""

    species: Species
    items: Mapping[str, ChecklistItem]
    safety_question_turns: int


@dataclass(frozen=True)
class RuleDefinition:
    """A rule predicate with enough metadata to emit a traceable RuleHit."""

    rule_id: str
    result: RuleResult
    condition: str
    item_ids: tuple[str, ...]
    predicate: Callable[[RuleFacts], bool]

    def evaluate(self, facts: RuleFacts) -> RuleHit | None:
        if not self.predicate(facts):
            return None
        return RuleHit(
            rule_id=self.rule_id,
            result=self.result,
            condition=self.condition,
            details=build_rule_details(facts, self.item_ids),
        )


def build_rule_details(facts: RuleFacts, item_ids: tuple[str, ...]) -> dict[str, object]:
    """Build a stable, non-mutating RuleHit details payload."""

    items: dict[str, dict[str, object]] = {}
    missing_rule_items: list[str] = []

    for item_id in item_ids:
        item = facts.items.get(item_id)
        if item is None:
            missing_rule_items.append(item_id)
            continue
        items[item_id] = {
            "value": item.value,
            "confidence": item.confidence,
        }

    return {
        "species": facts.species,
        "safety_question_turns": facts.safety_question_turns,
        "items": items,
        "missing_rule_items": missing_rule_items,
    }


def evaluate_emergency_rules(facts: RuleFacts) -> list[RuleHit]:
    """Return emergency RuleHits in document order."""

    return _evaluate_rules(EMERGENCY_RULES, facts)


def evaluate_urgent_rules(facts: RuleFacts) -> list[RuleHit]:
    """Return urgent RuleHits in document order."""

    return _evaluate_rules(URGENT_RULES, facts)


def build_missing_required_rule_hit(
    facts: RuleFacts,
    missing_item_ids: list[str],
) -> RuleHit | None:
    """Build the missing required red-flag RuleHit, if any item is missing."""

    if not missing_item_ids:
        return None

    if facts.safety_question_turns < 2:
        return RuleHit(
            rule_id="Q_MISSING_001",
            result="needs_more_info",
            condition="required red flag item is unknown and safety_question_turns < 2",
            details={
                **build_rule_details(facts, tuple(missing_item_ids)),
                "missing_items": missing_item_ids,
            },
        )

    return RuleHit(
        rule_id="Q_MISSING_002",
        result="unknown_after_max_questions",
        condition="required red flag item is unknown and safety_question_turns >= 2",
        details={
            **build_rule_details(facts, tuple(missing_item_ids)),
            "missing_items": missing_item_ids,
        },
    )


def build_low_confidence_emergency_rule_hit(facts: RuleFacts) -> RuleHit | None:
    """Build an unknown RuleHit for true emergency items with low confidence."""

    item_ids = low_confidence_emergency_item_ids(facts)
    if not item_ids:
        return None

    return RuleHit(
        rule_id="Q_CONF_001",
        result="unknown_due_to_low_confidence",
        condition="any emergency item has confidence == low",
        details={
            **build_rule_details(facts, tuple(item_ids)),
            "low_confidence_items": item_ids,
        },
    )


def build_non_emergency_rule_hit(facts: RuleFacts) -> RuleHit:
    """Build the fallback non-emergency RuleHit."""

    return RuleHit(
        rule_id="N_NONE_001",
        result="non_emergency",
        condition="no emergency, urgent, or unknown rules fired",
        details={
            "species": facts.species,
            "safety_question_turns": facts.safety_question_turns,
        },
    )


def low_confidence_emergency_item_ids(facts: RuleFacts) -> list[str]:
    """Return true low-confidence items that would otherwise be emergency signals."""

    item_ids: list[str] = []

    if facts.species == "cat" and _is_low_confidence_true(facts, "open_mouth_breathing"):
        item_ids.append("open_mouth_breathing")

    for item_id in (
        "labored_breathing",
        "gum_color_abnormal",
        "collapse_or_fainting",
        "active_seizure",
        "seizure_over_5_min",
        "repeated_seizures",
        "known_toxin_ingestion",
        "suspected_toxin",
        "severe_bleeding",
        "unable_to_urinate",
    ):
        if _is_low_confidence_true(facts, item_id):
            item_ids.append(item_id)

    return item_ids


def _evaluate_rules(rules: tuple[RuleDefinition, ...], facts: RuleFacts) -> list[RuleHit]:
    return [hit for rule in rules if (hit := rule.evaluate(facts)) is not None]


def _is_confident_true(facts: RuleFacts, item_id: str) -> bool:
    item = facts.items.get(item_id)
    return item is not None and item.value is True and item.confidence in {"high", "medium"}


def _is_true(facts: RuleFacts, item_id: str) -> bool:
    item = facts.items.get(item_id)
    return item is not None and item.value is True


def _is_low_confidence_true(facts: RuleFacts, item_id: str) -> bool:
    item = facts.items.get(item_id)
    return item is not None and item.value is True and item.confidence == "low"


def _has_non_empty_value(facts: RuleFacts, item_id: str) -> bool:
    item = facts.items.get(item_id)
    if item is None or item.value is None:
        return False
    if isinstance(item.value, str):
        return item.value.strip() != ""
    return item.value is not False


def _number_at_least(facts: RuleFacts, item_id: str, minimum: float) -> bool:
    item = facts.items.get(item_id)
    if item is None or isinstance(item.value, bool):
        return False
    if isinstance(item.value, int | float):
        return float(item.value) >= minimum
    if isinstance(item.value, str):
        try:
            return float(item.value.strip()) >= minimum
        except ValueError:
            return False
    return False


EMERGENCY_RULES: tuple[RuleDefinition, ...] = (
    RuleDefinition(
        rule_id="E_RESP_001",
        result="emergency",
        condition="open_mouth_breathing == true and species == cat",
        item_ids=("open_mouth_breathing",),
        predicate=lambda facts: facts.species == "cat"
        and _is_confident_true(facts, "open_mouth_breathing"),
    ),
    RuleDefinition(
        rule_id="E_RESP_002",
        result="emergency",
        condition="labored_breathing == true",
        item_ids=("labored_breathing",),
        predicate=lambda facts: _is_confident_true(facts, "labored_breathing"),
    ),
    RuleDefinition(
        rule_id="E_RESP_003",
        result="emergency",
        condition="gum_color_abnormal == true",
        item_ids=("gum_color_abnormal",),
        predicate=lambda facts: _is_confident_true(facts, "gum_color_abnormal"),
    ),
    RuleDefinition(
        rule_id="E_GEN_001",
        result="emergency",
        condition="collapse_or_fainting == true",
        item_ids=("collapse_or_fainting",),
        predicate=lambda facts: _is_confident_true(facts, "collapse_or_fainting"),
    ),
    RuleDefinition(
        rule_id="E_SEIZ_001",
        result="emergency",
        condition="active_seizure == true",
        item_ids=("active_seizure",),
        predicate=lambda facts: _is_confident_true(facts, "active_seizure"),
    ),
    RuleDefinition(
        rule_id="E_SEIZ_002",
        result="emergency",
        condition="seizure_over_5_min == true or repeated_seizures == true",
        item_ids=("seizure_over_5_min", "repeated_seizures"),
        predicate=lambda facts: _is_confident_true(facts, "seizure_over_5_min")
        or _is_confident_true(facts, "repeated_seizures"),
    ),
    RuleDefinition(
        rule_id="E_TOX_001",
        result="emergency",
        condition="known_toxin_ingestion == true or suspected_toxin == true",
        item_ids=("known_toxin_ingestion", "suspected_toxin"),
        predicate=lambda facts: _is_confident_true(facts, "known_toxin_ingestion")
        or _is_confident_true(facts, "suspected_toxin"),
    ),
    RuleDefinition(
        rule_id="E_BLEED_001",
        result="emergency",
        condition="severe_bleeding == true",
        item_ids=("severe_bleeding",),
        predicate=lambda facts: _is_confident_true(facts, "severe_bleeding"),
    ),
    RuleDefinition(
        rule_id="E_URIN_001",
        result="emergency",
        condition="unable_to_urinate == true",
        item_ids=("unable_to_urinate",),
        predicate=lambda facts: _is_confident_true(facts, "unable_to_urinate"),
    ),
)

URGENT_RULES: tuple[RuleDefinition, ...] = (
    RuleDefinition(
        rule_id="U_RESP_001",
        result="urgent",
        condition="rapid_breathing == true",
        item_ids=("rapid_breathing",),
        predicate=lambda facts: _is_true(facts, "rapid_breathing"),
    ),
    RuleDefinition(
        rule_id="U_GEN_001",
        result="urgent",
        condition="lethargy == true and new_symptoms is not empty",
        item_ids=("lethargy", "new_symptoms"),
        predicate=lambda facts: _is_true(facts, "lethargy")
        and _has_non_empty_value(facts, "new_symptoms"),
    ),
    RuleDefinition(
        rule_id="U_GI_001",
        result="urgent",
        condition="repeated_vomiting == true or repeated_diarrhea == true",
        item_ids=("repeated_vomiting", "repeated_diarrhea"),
        predicate=lambda facts: _is_true(facts, "repeated_vomiting")
        or _is_true(facts, "repeated_diarrhea"),
    ),
    RuleDefinition(
        rule_id="U_GI_002",
        result="urgent",
        condition="appetite_decreased == true and duration_hours >= 24",
        item_ids=("appetite_decreased", "duration_hours"),
        predicate=lambda facts: _is_true(facts, "appetite_decreased")
        and _number_at_least(facts, "duration_hours", 24),
    ),
    RuleDefinition(
        rule_id="U_BASE_001",
        result="urgent",
        condition="baseline_deviation == true and worsened_fields is not empty",
        item_ids=("baseline_deviation", "worsened_fields"),
        predicate=lambda facts: _is_true(facts, "baseline_deviation")
        and _has_non_empty_value(facts, "worsened_fields"),
    ),
)

__all__ = [
    "EMERGENCY_RULES",
    "URGENT_RULES",
    "RuleDefinition",
    "RuleFacts",
    "build_low_confidence_emergency_rule_hit",
    "build_missing_required_rule_hit",
    "build_non_emergency_rule_hit",
    "evaluate_emergency_rules",
    "evaluate_urgent_rules",
    "low_confidence_emergency_item_ids",
]
