"""LangSmith tracing helpers for the assessment graph.

Phase 11 keeps LangSmith as an observability layer only. The helpers in this
module must not affect graph routing, validation, or response composition.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
import sys
from typing import Any, Iterator

from petcare_agent.config import PetCareSettings, get_settings

TRACE_SCHEMA_VERSION = "phase13.v1"
ASSESSMENT_GRAPH_VERSION = "assessment-graph.phase13"
TRACE_RUN_NAME_POLICY = "{LANGSMITH_RUN_PREFIX}[.{node_name}]"
TRACE_PRIVACY_POLICY = "safe_metadata_only_no_raw_user_or_medical_text"

SENSITIVE_METADATA_KEYS = frozenset(
    {
        "assistant_message",
        "chat_response",
        "content",
        "conversation_history",
        "daily_entries",
        "diagnoses",
        "diagnosis_content",
        "diseases_medications_allergies",
        "email_draft",
        "evidence",
        "hospital",
        "name",
        "notes",
        "raw_daily_entry_text",
        "raw_diagnosis_content",
        "raw_text",
        "revised_answer",
        "text",
        "unsafe_phrases",
        "user_input",
    }
)
MAX_METADATA_STRING_LENGTH = 240


@dataclass(frozen=True)
class TraceContext:
    request_id: str | None = None
    conversation_id: str | None = None
    node_name: str | None = None
    route: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def is_tracing_enabled(settings: PetCareSettings | None = None) -> bool:
    """Return whether LangSmith tracing should be active."""

    resolved = settings or get_settings()
    return bool(resolved.langsmith_tracing and resolved.langsmith_api_key)


def build_run_name(
    context: TraceContext | None = None,
    settings: PetCareSettings | None = None,
) -> str:
    """Build the standard LangSmith run name for graph and node traces.

    Naming policy:
    - graph run: ``LANGSMITH_RUN_PREFIX``
    - node run: ``LANGSMITH_RUN_PREFIX.node_name``
    - schema/graph versions live in metadata, not in the run name, so project
      dashboards can group the stable run names across releases.
    """

    resolved = settings or get_settings()
    if context and context.node_name:
        return f"{resolved.langsmith_run_prefix}.{context.node_name}"
    return resolved.langsmith_run_prefix


def build_metadata(context: TraceContext | None = None) -> dict[str, Any]:
    """Build trace metadata without letting tracing affect graph decisions."""

    metadata: dict[str, Any] = _base_metadata()
    if not context:
        return metadata

    if context.request_id:
        metadata["request_id"] = context.request_id
    if context.conversation_id:
        metadata["conversation_id"] = context.conversation_id
    if context.node_name:
        metadata["node_name"] = context.node_name
    if context.route:
        metadata["route"] = context.route
    metadata.update(sanitize_trace_metadata(context.metadata))
    return sanitize_trace_metadata(metadata)


def build_runnable_config(
    context: TraceContext | None = None,
    settings: PetCareSettings | None = None,
) -> dict[str, Any]:
    """Return a LangGraph RunnableConfig-compatible trace config."""

    resolved = settings or get_settings()
    metadata = build_metadata(context)
    metadata["langsmith_tracing_enabled"] = is_tracing_enabled(resolved)
    metadata["langsmith_project"] = resolved.langsmith_project
    return {
        "run_name": build_run_name(context, resolved),
        "metadata": sanitize_trace_metadata(metadata),
        "tags": [
            "petcare-ai",
            "assessment-graph",
            resolved.environment,
            TRACE_SCHEMA_VERSION,
        ],
    }


def build_state_trace_metadata(
    state: Any,
    *,
    node_name: str | None = None,
    route: str | None = None,
) -> dict[str, Any]:
    """Build sanitized, evaluation-friendly metadata from graph state.

    The metadata intentionally avoids raw user input, raw conversation history,
    raw daily-entry text, raw diagnosis content, RAG chunk text, assistant drafts,
    handoff drafts, and checklist evidence snippets. It records IDs, routes,
    counts, flags, rule ids, and bounded summaries that are useful for debugging
    and regression evaluation.
    """

    triggered_rules = _triggered_rule_ids(state)
    metadata: dict[str, Any] = {
        **_base_metadata(),
        "node_name": node_name,
        "route": route,
        "intent": _safe_attr(state, "intent", "unknown"),
        "risk_level": _safe_attr(state, "risk_level", "unknown"),
        "confidence": _safe_attr(state, "confidence", "unknown"),
        "next_route": _safe_attr(state, "next_route", ""),
        "triggered_rules": triggered_rules,
        "request_context": _request_context_metadata(state),
        "routing": _routing_metadata(state),
        "checklist": _checklist_metadata(state),
        "safety": _safety_metadata(state, triggered_rules),
        "change_detection": _change_detection_metadata(state),
        "rag": _rag_metadata(state),
        "answer_guard": _answer_guard_metadata(state),
        "handoff": _handoff_metadata(state),
    }
    return sanitize_trace_metadata(metadata)


def sanitize_trace_metadata(value: Any) -> Any:
    """Remove known sensitive raw-text fields from trace metadata."""

    if isinstance(value, dict):
        sanitized: dict[str, Any] = {}
        for raw_key, raw_item in value.items():
            key = str(raw_key)
            if key.lower() in SENSITIVE_METADATA_KEYS:
                continue
            sanitized[key] = sanitize_trace_metadata(raw_item)
        return sanitized
    if isinstance(value, (list, tuple, set)):
        return [sanitize_trace_metadata(item) for item in value]
    if isinstance(value, str):
        return _bounded_string(value)
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return _bounded_string(str(value))


@contextmanager
def trace_span(
    name: str,
    *,
    inputs: dict[str, Any] | None = None,
    context: TraceContext | None = None,
    settings: PetCareSettings | None = None,
) -> Iterator[Any | None]:
    """Open a LangSmith span when tracing is enabled.

    The import is intentionally lazy so disabled tracing remains import-safe in
    local tests. Any LangSmith client/import failure is fail-open and yields
    ``None`` so graph logic never depends on observability availability.
    """

    resolved = settings or get_settings()
    if not is_tracing_enabled(resolved):
        yield None
        return

    run_manager: Any | None = None
    try:
        from langsmith import trace  # type: ignore

        run_manager = trace(
            name,
            run_type="chain",
            inputs=sanitize_trace_metadata(inputs or {}),
            metadata=build_metadata(context),
            project_name=resolved.langsmith_project,
        )
        run = run_manager.__enter__()
    except Exception:
        yield None
        return

    exc_info: tuple[Any, Any, Any] = (None, None, None)
    try:
        try:
            yield run
        except BaseException:
            exc_info = sys.exc_info()
            raise
    finally:
        try:
            run_manager.__exit__(*exc_info)
        except Exception:
            pass


def _base_metadata() -> dict[str, Any]:
    return {
        "component": "assessment_graph",
        "trace_schema_version": TRACE_SCHEMA_VERSION,
        "graph_version": ASSESSMENT_GRAPH_VERSION,
        "run_name_policy": TRACE_RUN_NAME_POLICY,
        "privacy_policy": TRACE_PRIVACY_POLICY,
    }


def _request_context_metadata(state: Any) -> dict[str, Any]:
    return {
        "request_id": _safe_attr(state, "request_id", None),
        "conversation_id": _safe_attr(state, "conversation_id", None),
        "pet_id": _safe_attr(state, "pet_id", None),
        "locale": _safe_attr(state, "locale", None),
        "timezone": _safe_attr(state, "timezone", None),
    }


def _routing_metadata(state: Any) -> dict[str, Any]:
    return {
        "intent": _safe_attr(state, "intent", "unknown"),
        "requires_db_context": _safe_attr(state, "requires_db_context", False),
        "requires_safety_screening": _safe_attr(
            state,
            "requires_safety_screening",
            False,
        ),
        "red_flag_mentioned": _safe_attr(state, "red_flag_mentioned", False),
        "next_route": _safe_attr(state, "next_route", ""),
    }


def _checklist_metadata(state: Any) -> dict[str, Any]:
    screening = _safe_attr(state, "emergency_screening", None)
    items = dict(_safe_attr(screening, "items", {}) or {})
    known_item_ids: list[str] = []
    unknown_item_ids: list[str] = []
    confidence_counts: dict[str, int] = {}
    value_counts = {"true": 0, "false": 0, "known_other": 0, "unknown": 0}

    for item_id, item in sorted(items.items()):
        value = _safe_attr(item, "value", None)
        confidence = str(_safe_attr(item, "confidence", "unknown"))
        confidence_counts[confidence] = confidence_counts.get(confidence, 0) + 1
        if value is True:
            value_counts["true"] += 1
            known_item_ids.append(str(item_id))
        elif value is False:
            value_counts["false"] += 1
            known_item_ids.append(str(item_id))
        elif value is None or value == "unknown":
            value_counts["unknown"] += 1
            unknown_item_ids.append(str(item_id))
        else:
            value_counts["known_other"] += 1
            known_item_ids.append(str(item_id))

    assessment = _safe_attr(state, "assessment", None)
    return {
        "checklist_id": _safe_attr(screening, "checklist_id", ""),
        "chief_complaint": _safe_attr(screening, "chief_complaint", ""),
        "status": _safe_attr(screening, "status", "not_started"),
        "items_total": len(items),
        "known_item_ids": known_item_ids,
        "unknown_item_ids": unknown_item_ids,
        "confidence_counts": confidence_counts,
        "value_counts": value_counts,
        "answered_question_count": len(
            dict(_safe_attr(screening, "answered_questions", {}) or {})
        ),
        "missing_questions": list(_safe_attr(screening, "missing_questions", []) or []),
        "red_flags": list(_safe_attr(screening, "red_flags", []) or []),
        "missing_fields": list(_safe_attr(assessment, "missing_fields", []) or []),
    }


def _safety_metadata(state: Any, triggered_rules: list[str]) -> dict[str, Any]:
    screening = _safe_attr(state, "emergency_screening", None)
    return {
        "risk_level": _safe_attr(state, "risk_level", "unknown"),
        "confidence": _safe_attr(state, "confidence", "unknown"),
        "safety_question_turns": _safe_attr(state, "safety_question_turns", 0),
        "triggered_rules": triggered_rules,
        "triggered_rule_results": [
            {
                "rule_id": _safe_attr(rule, "rule_id", ""),
                "result": _safe_attr(rule, "result", ""),
            }
            for rule in list(_safe_attr(screening, "triggered_rules", []) or [])
        ],
        "red_flags": list(_safe_attr(screening, "red_flags", []) or []),
    }


def _change_detection_metadata(state: Any) -> dict[str, Any]:
    change = _safe_attr(state, "change_detection", None)
    return {
        "baseline_available": _safe_attr(change, "baseline_available", False),
        "baseline_deviation": _safe_attr(change, "baseline_deviation", False),
        "new_symptom_count": len(list(_safe_attr(change, "new_symptoms", []) or [])),
        "worsened_fields": list(_safe_attr(change, "worsened_fields", []) or []),
        "improved_fields": list(_safe_attr(change, "improved_fields", []) or []),
        "unchanged_fields": list(_safe_attr(change, "unchanged_fields", []) or []),
        "summary": _safe_attr(change, "summary", ""),
    }


def _rag_metadata(state: Any) -> dict[str, Any]:
    retrieval = _safe_attr(state, "retrieval", None)
    chunks = list(_safe_attr(retrieval, "chunks", []) or [])
    return {
        "has_query": bool(str(_safe_attr(retrieval, "query", "")).strip()),
        "query_length": len(str(_safe_attr(retrieval, "query", "")).strip()),
        "provider": _safe_attr(retrieval, "provider", ""),
        "insufficient_evidence": _safe_attr(retrieval, "insufficient_evidence", False),
        "citation_count": len(list(_safe_attr(retrieval, "citations", []) or [])),
        "error_count": len(list(_safe_attr(retrieval, "errors", []) or [])),
        "chunk_count": len(chunks),
        "chunks": [
            {
                "chunk_id": _safe_attr(chunk, "chunk_id", ""),
                "source_id": _safe_attr(chunk, "source_id", ""),
                "title": _safe_attr(chunk, "title", ""),
                "score": _safe_attr(chunk, "score", None),
                "metadata": sanitize_trace_metadata(
                    dict(_safe_attr(chunk, "metadata", {}) or {})
                ),
            }
            for chunk in chunks
        ],
    }


def _answer_guard_metadata(state: Any) -> dict[str, Any]:
    answer_guard = _safe_attr(state, "answer_guard", None)
    return {
        "status": _safe_attr(answer_guard, "status", "passed"),
        "revision_count": len(list(_safe_attr(answer_guard, "revisions", []) or [])),
    }


def _handoff_metadata(state: Any) -> dict[str, Any]:
    handoff = _safe_attr(state, "handoff", None)
    return {
        "required": _safe_attr(handoff, "required", False),
        "type": _safe_attr(handoff, "type", "none"),
        "summary_present": bool(str(_safe_attr(handoff, "summary", "")).strip()),
        "email_draft_present": bool(str(_safe_attr(handoff, "email_draft", "")).strip()),
    }


def _triggered_rule_ids(state: Any) -> list[str]:
    screening = _safe_attr(state, "emergency_screening", None)
    return [
        str(_safe_attr(rule, "rule_id", ""))
        for rule in list(_safe_attr(screening, "triggered_rules", []) or [])
        if _safe_attr(rule, "rule_id", "")
    ]


def _safe_attr(value: Any, name: str, default: Any) -> Any:
    if value is None:
        return default
    return getattr(value, name, default)


def _bounded_string(value: str) -> str:
    normalized = " ".join(value.split())
    if len(normalized) <= MAX_METADATA_STRING_LENGTH:
        return normalized
    return f"{normalized[:MAX_METADATA_STRING_LENGTH]}..."


__all__ = [
    "ASSESSMENT_GRAPH_VERSION",
    "TRACE_PRIVACY_POLICY",
    "TRACE_RUN_NAME_POLICY",
    "TRACE_SCHEMA_VERSION",
    "TraceContext",
    "build_metadata",
    "build_run_name",
    "build_runnable_config",
    "build_state_trace_metadata",
    "is_tracing_enabled",
    "sanitize_trace_metadata",
    "trace_span",
]

