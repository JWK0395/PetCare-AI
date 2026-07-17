"""Rule-based checklist validator for Phase 3 Safety Guard."""

from __future__ import annotations

from petcare_agent.safety.rules import (
    RuleFacts,
    build_low_confidence_emergency_rule_hit,
    build_missing_required_rule_hit,
    build_non_emergency_rule_hit,
    evaluate_emergency_rules,
    evaluate_urgent_rules,
)
from petcare_agent.schemas.common import RiskAction, Species
from petcare_agent.schemas.triage import ChecklistItem, ChecklistTemplate, RiskResult, RuleHit


def validate_checklist(
    checklist: ChecklistTemplate,
    *,
    safety_question_turns: int = 0,
    species: Species | None = None,
) -> RiskResult:
    """Validate a filled checklist and return the final rule-based risk result.

    Phase 3 intentionally evaluates checklist state only. The optional species
    override preserves cat/dog-specific behavior for shared templates such as
    ``breathing_triage`` without requiring DB/API, graph wiring, LLM, or RAG.
    """

    if safety_question_turns < 0:
        raise ValueError("safety_question_turns must be >= 0")

    checklist_copy = checklist.model_copy(deep=True)
    facts = RuleFacts(
        species=_resolve_species(checklist_copy, species),
        items=_items_by_id(checklist_copy),
        safety_question_turns=safety_question_turns,
    )

    emergency_hits = evaluate_emergency_rules(facts)
    urgent_hits = evaluate_urgent_rules(facts)
    missing_items = _missing_required_red_flag_item_ids(checklist_copy, facts.items)

    unknown_hits = _build_unknown_hits(facts, missing_items)
    triggered_rules = [*emergency_hits, *urgent_hits, *unknown_hits]

    if emergency_hits:
        return RiskResult(
            risk_level="emergency",
            confidence="high",
            action="final",
            triggered_rules=triggered_rules,
            missing_items=missing_items,
            requires_more_info=False,
        )

    if urgent_hits:
        return RiskResult(
            risk_level="urgent",
            confidence="medium",
            action="final",
            triggered_rules=triggered_rules,
            missing_items=missing_items,
            requires_more_info=False,
        )

    if unknown_hits:
        action = _select_unknown_action(unknown_hits)
        return RiskResult(
            risk_level="unknown",
            confidence="low",
            action=action,
            triggered_rules=triggered_rules,
            missing_items=missing_items,
            requires_more_info=_requires_more_info(action, safety_question_turns),
        )

    return RiskResult(
        risk_level="non_emergency",
        confidence="medium",
        action="final",
        triggered_rules=[build_non_emergency_rule_hit(facts)],
        missing_items=[],
        requires_more_info=False,
    )


def _items_by_id(checklist: ChecklistTemplate) -> dict[str, ChecklistItem]:
    return {
        item.item_id: item
        for item in [*checklist.required_items, *checklist.optional_items]
    }


def _resolve_species(checklist: ChecklistTemplate, species: Species | None) -> Species:
    if species in {"cat", "dog"}:
        return species
    if checklist.species in {"cat", "dog"}:
        return checklist.species
    return "unknown"


def _missing_required_red_flag_item_ids(
    checklist: ChecklistTemplate,
    items: dict[str, ChecklistItem],
) -> list[str]:
    missing_items: list[str] = []

    for required_item in checklist.required_items:
        if required_item.metadata.get("red_flag") is False:
            continue

        item = items.get(required_item.item_id)
        if item is None or _item_is_unknown(item):
            missing_items.append(required_item.item_id)

    return missing_items


def _item_is_unknown(item: ChecklistItem) -> bool:
    if item.value is None:
        return True
    if isinstance(item.value, str) and item.value.strip().lower() == "unknown":
        return True
    return item.confidence == "unknown"


def _build_unknown_hits(facts: RuleFacts, missing_items: list[str]) -> list[RuleHit]:
    hits: list[RuleHit] = []

    low_confidence_hit = build_low_confidence_emergency_rule_hit(facts)
    if low_confidence_hit is not None:
        hits.append(low_confidence_hit)

    missing_hit = build_missing_required_rule_hit(facts, missing_items)
    if missing_hit is not None:
        hits.append(missing_hit)

    return hits


def _select_unknown_action(unknown_hits: list[RuleHit]) -> RiskAction:
    for preferred_action in (
        "needs_more_info",
        "unknown_after_max_questions",
        "unknown_due_to_low_confidence",
    ):
        if any(hit.result == preferred_action for hit in unknown_hits):
            return preferred_action
    return "unknown_due_to_low_confidence"


def _requires_more_info(action: RiskAction, safety_question_turns: int) -> bool:
    if action == "needs_more_info":
        return True
    return action == "unknown_due_to_low_confidence" and safety_question_turns < 2


__all__ = ["validate_checklist"]
