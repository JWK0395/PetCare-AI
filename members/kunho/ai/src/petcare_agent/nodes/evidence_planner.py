"""Plan official-source evidence retrieval before answer composition."""

from __future__ import annotations

from petcare_agent.schemas.graph_state import PetCareGraphState

EVIDENCE_SAFE_RISK_LEVELS = {"urgent", "non_emergency", "unknown"}


def plan_evidence_context(state: PetCareGraphState) -> PetCareGraphState:
    """Prepare a retrieval query without drafting the user-facing answer."""

    next_state = state.model_copy(deep=True)
    existing_query = next_state.retrieval.query.strip()
    planned_query = existing_query or _build_retrieval_query(next_state)

    if planned_query != existing_query:
        next_state.retrieval.chunks = []
        next_state.retrieval.citations = []
        next_state.retrieval.provider = ""
        next_state.retrieval.insufficient_evidence = False
        next_state.retrieval.errors = []

    next_state.retrieval.query = planned_query
    next_state.next_route = "evidence_planner"
    return next_state


def evidence_planner(state: PetCareGraphState) -> PetCareGraphState:
    """LangGraph-friendly alias for the evidence planning node."""

    return plan_evidence_context(state)


def _build_retrieval_query(state: PetCareGraphState) -> str:
    parts: list[str] = []

    user_input = " ".join(state.user_input.split())
    if user_input:
        parts.append(user_input)

    if state.species in {"cat", "dog"}:
        parts.append(f"species:{state.species}")

    if state.risk_level in EVIDENCE_SAFE_RISK_LEVELS:
        parts.append(f"risk_level:{state.risk_level}")

    chief_complaint = state.emergency_screening.chief_complaint.strip()
    if chief_complaint:
        parts.append(f"chief_complaint:{chief_complaint}")

    symptoms = _known_symptoms(state)
    if symptoms:
        parts.append(f"symptoms:{', '.join(symptoms)}")

    return " ".join(parts)


def _known_symptoms(state: PetCareGraphState) -> list[str]:
    symptoms: list[str] = []
    seen: set[str] = set()
    for raw_symptom in [*state.assessment.symptoms, *state.current_status.symptoms]:
        symptom = " ".join(raw_symptom.strip().lower().split())
        if not symptom or symptom in seen:
            continue
        symptoms.append(symptom)
        seen.add(symptom)
    return symptoms


__all__ = ["EVIDENCE_SAFE_RISK_LEVELS", "evidence_planner", "plan_evidence_context"]
