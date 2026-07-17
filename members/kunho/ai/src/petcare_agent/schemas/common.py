"""Shared type aliases for graph contracts."""

from __future__ import annotations

from typing import Literal

Species = Literal["cat", "dog", "unknown"]
Confidence = Literal["high", "medium", "low", "unknown"]
CoursePattern = Literal["new", "worsening", "improving", "persistent", "recurrent", "unknown"]
Intent = Literal[
    "social_chat",
    "general_chat",
    "symptom_check",
    "followup",
    "handoff_request",
    "document_request",
    "unknown",
]
RiskLevel = Literal["emergency", "urgent", "non_emergency", "unknown"]
RiskAction = Literal[
    "final",
    "needs_more_info",
    "unknown_after_max_questions",
    "unknown_due_to_low_confidence",
]
NodeRoute = Literal[
    "intent_classifier",
    "db_context_loader",
    "baseline_builder",
    "state_updater",
    "change_detector",
    "safety_guard",
    "question_manager",
    "emergency",
    "chat",
    "evidence_planner",
    "rag",
    "answer_composer",
    "answer_guard",
    "handoff",
    "end",
]
HospitalVisitIntent = Literal["yes", "no", "undecided", "not_asked"]
AnswerGuardStatus = Literal["passed", "revised", "blocked"]
HandoffType = Literal["non_emergency", "emergency", "none"]



