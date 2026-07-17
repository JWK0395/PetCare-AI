"""Checklist extraction structured-output node."""

from __future__ import annotations

import json
import re
from typing import Any

from petcare_agent.llm.client import StructuredOutputClient, call_structured_output
from petcare_agent.prompts import load_prompt
from petcare_agent.schemas.common import Confidence
from petcare_agent.schemas.graph_state import PetCareGraphState
from petcare_agent.schemas.llm_outputs import ChecklistExtractionOutput

_AFFIRMATIVE_REPLIES = {
    "yes",
    "y",
    "yeah",
    "yep",
    "true",
    "correct",
    "right",
    "\ub124",
    "\uc608",
    "\uc751",
    "\uc5b4",
    "\ub9de\uc544",
    "\ub9de\uc544\uc694",
    "\uadf8\ub798\uc694",
    "\uadf8\ub807\uc2b5\ub2c8\ub2e4",
    "\ub124\ub9de\uc544\uc694",
    "\uc608\ub9de\uc544\uc694",
    "\u3147\u3147",
}

_NEGATIVE_REPLIES = {
    "no",
    "n",
    "nope",
    "false",
    "incorrect",
    "\uc544\ub2c8\uc694",
    "\uc544\ub2c8",
    "\uc544\ub1e8",
    "\uc544\ub2d9\ub2c8\ub2e4",
    "\uc544\ub2cc\uac83\uac19\uc544\uc694",
    "\uc544\ub2cc\uac70\uac19\uc544\uc694",
    "\uc548\uadf8\ub798\uc694",
    "\uc5c6\uc5b4\uc694",
    "\uc5c6\uc2b5\ub2c8\ub2e4",
    "\u3134\u3134",
}


def extract_checklist_updates(
    state: PetCareGraphState,
    *,
    llm_client: StructuredOutputClient | None = None,
) -> PetCareGraphState:
    """Update existing emergency checklist item values from user text.

    Final risk determination intentionally remains with the Phase 3 validator.
    """

    next_state = state.model_copy(deep=True)
    fallback = ChecklistExtractionOutput(
        checklist_id=next_state.emergency_screening.checklist_id or "unknown",
        updates=[],
    )
    direct_followup_answer = _direct_followup_answer(next_state)

    try:
        output = call_structured_output(
            system_prompt=load_prompt("checklist_extraction"),
            user_prompt=_checklist_prompt_payload(next_state),
            output_model=ChecklistExtractionOutput,
            fallback=fallback,
            client=llm_client,
        )
    except Exception:
        output = fallback

    for update in output.updates:
        _apply_item_update(
            next_state,
            update.item_id,
            value=update.value,
            confidence=update.confidence,
            evidence=update.evidence,
        )

    if direct_followup_answer is not None:
        item_id, value, evidence = direct_followup_answer
        _apply_item_update(
            next_state,
            item_id,
            value=value,
            confidence="high",
            evidence=evidence,
        )

    _prune_answered_pending_questions(next_state)

    return next_state


def checklist_extractor(
    state: PetCareGraphState,
    *,
    llm_client: StructuredOutputClient | None = None,
) -> PetCareGraphState:
    """LangGraph-friendly alias for the checklist extractor node."""

    return extract_checklist_updates(state, llm_client=llm_client)


def _checklist_prompt_payload(state: PetCareGraphState) -> str:
    payload = {
        "user_input": state.user_input,
        "species": state.species,
        "chief_complaint": state.emergency_screening.chief_complaint,
        "checklist_id": state.emergency_screening.checklist_id,
        "items": {
            item_id: item.model_dump(mode="json")
            for item_id, item in sorted(state.emergency_screening.items.items())
        },
        "pending_question_item_ids": list(state.emergency_screening.missing_questions),
        "answered_questions": dict(state.emergency_screening.answered_questions),
        "assessment": state.assessment.model_dump(mode="json"),
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def _apply_item_update(
    state: PetCareGraphState,
    item_id: str,
    *,
    value: Any,
    confidence: Confidence,
    evidence: str | None,
) -> None:
    item = state.emergency_screening.items.get(item_id)
    if item is None:
        return

    item.value = value
    item.confidence = confidence
    if evidence is not None:
        item.metadata = {**item.metadata, "evidence": evidence}

    if not _item_is_unknown(item):
        _mark_question_answered(state, item_id, value)


def _direct_followup_answer(state: PetCareGraphState) -> tuple[str, bool, str] | None:
    answer = _parse_direct_boolean_reply(state.user_input)
    if answer is None:
        return None

    item_id = _first_pending_boolean_question_id(state)
    if item_id is None:
        return None

    return item_id, answer, state.user_input.strip()


def _parse_direct_boolean_reply(user_input: str) -> bool | None:
    normalized = _normalize_reply(user_input)
    if not normalized:
        return None
    if normalized in _AFFIRMATIVE_REPLIES:
        return True
    if normalized in _NEGATIVE_REPLIES:
        return False
    return None


def _normalize_reply(user_input: str) -> str:
    lowered = user_input.strip().casefold()
    normalized = re.sub(r"[\s.!?,;:~]+", " ", lowered).strip()
    return normalized.replace(" ", "")


def _first_pending_boolean_question_id(state: PetCareGraphState) -> str | None:
    for item_id in state.emergency_screening.missing_questions:
        item = state.emergency_screening.items.get(item_id)
        if item is None or item.type != "boolean":
            continue
        if _item_is_unknown(item):
            return item_id
    return None


def _mark_question_answered(
    state: PetCareGraphState,
    item_id: str,
    value: Any,
) -> None:
    state.emergency_screening.answered_questions = {
        **state.emergency_screening.answered_questions,
        item_id: value,
        f"resp_{item_id}": value,
    }
    state.emergency_screening.missing_questions = [
        missing_item_id
        for missing_item_id in state.emergency_screening.missing_questions
        if missing_item_id != item_id
    ]


def _prune_answered_pending_questions(state: PetCareGraphState) -> None:
    remaining: list[str] = []
    answered = dict(state.emergency_screening.answered_questions)

    for item_id in state.emergency_screening.missing_questions:
        item = state.emergency_screening.items.get(item_id)
        if item is not None and not _item_is_unknown(item):
            answered[item_id] = item.value
            answered[f"resp_{item_id}"] = item.value
            continue
        remaining.append(item_id)

    state.emergency_screening.answered_questions = answered
    state.emergency_screening.missing_questions = remaining


def _item_is_unknown(item: object) -> bool:
    value = getattr(item, "value", None)
    confidence = getattr(item, "confidence", "unknown")
    if value is None:
        return True
    if isinstance(value, str) and value.strip().lower() == "unknown":
        return True
    return confidence == "unknown"
