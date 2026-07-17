# Agent Observability Policy

Date: 2026-07-17
Scope: Assessment Graph LangSmith observability, current implementation with trace schema `phase13.v1`

## Trace Naming And Versioning

LangSmith run names stay stable so dashboards can group graph and node runs across releases.

- Graph run name: `LANGSMITH_RUN_PREFIX`, default `assessment_graph`
- Node run name: `LANGSMITH_RUN_PREFIX.node_name`, for example `assessment_graph.safety_guard`
- Trace schema version is recorded in metadata as `trace_schema_version`
- Assessment graph version is recorded in metadata as `graph_version`
- Current metadata schema version: `phase13.v1`


## Current Implementation Notes

The active tracing implementation lives at `members/kunho/ai/src/petcare_agent/tracing.py`. The feature set has moved through Phase 17, but the trace metadata constants remain:

- `TRACE_SCHEMA_VERSION = "phase13.v1"`
- `ASSESSMENT_GRAPH_VERSION = "assessment-graph.phase13"`

Do not infer missing product features from those names. They are stable observability grouping labels, not a full implementation phase number.
## Safe Metadata Allowlist

Trace metadata may include operational identifiers and structured summaries needed for debugging and evaluation:

- `request_id`
- `conversation_id`
- `pet_id`
- locale and timezone
- node name, route, next route
- intent, risk level, confidence
- checklist id, chief complaint, item counts, known/unknown item ids, confidence counts, red flag item ids
- triggered rule ids and rule result labels
- change-detection summary fields such as baseline availability, deviation flag, field names, and bounded generated summary
- RAG query presence/length, chunk count, chunk ids, source ids, titles, scores, and sanitized chunk metadata
- answer guard status and revision count
- handoff required/type and whether summary or email draft exists

## Data That Must Not Be Logged

Trace metadata must not include raw user-authored or medical free text:

- raw `user_input`
- raw `conversation_history` content
- raw daily entry `raw_text`
- raw diagnosis `content`
- checklist extraction evidence snippets
- RAG chunk `text`
- assistant draft text, final assistant message text, answer-guard unsafe phrases, revised answer text
- handoff summary body or email draft body
- pet name, hospital name, or diseases/medications/allergies lists
- exact user location coordinates

The implementation enforces this with `sanitize_trace_metadata(...)` and by building metadata through `build_state_trace_metadata(...)` instead of dumping full graph state.

## Failure Behavior

LangSmith is fail-open. Disabled tracing yields no span. Enabled tracing attempts to open a LangSmith span, but import/client errors do not stop graph execution. Tests use fake LangSmith tracing and mocked LLM/RAG/API boundaries only.

## Phase 12 Handoff And Internal Triage Observability

Phase 12 adds a strict boundary between internal triage and hospital handoff JSON. Observability must preserve that boundary.

Allowed trace metadata:

- whether `internal_triage_assessment` exists
- internal triage `risk_level`
- `needs_followup`
- follow-up question count, not raw question text
- canonical red-flag ids and known/unknown counts
- whether `handoff.summary_json` exists
- handoff schema version
- handoff section-presence booleans for the six sections
- `baseline_comparison.window_days`
- whether forbidden internal fields were absent/present in validation checks

Disallowed trace metadata:

- full `internal_triage_assessment.followup_questions` text
- full `handoff.summary_json`
- `patient.name`, `pet_id` inside handoff JSON, conditions, medications, allergies, owner free text, timeline text, or symptom summary body
- email draft body
- rule details that include evidence snippets or raw user text

Operational rule:

- Trace metadata may record that a six-section handoff was generated.
- Trace metadata must not dump the six-section handoff payload.
- If future eval tracing needs more detail, add bounded derived booleans/counts rather than raw handoff content.

## Phase 13 Evidence And RAG Observability

Phase 13 adds evidence-first RAG composition. Trace metadata may include the new retrieval summary fields:

- `rag.provider`
- `rag.insufficient_evidence`
- `rag.citation_count`
- `rag.error_count`
- chunk ids, source ids, titles, scores, and sanitized metadata such as provider, species, canonical URL, and section path

Trace metadata must still exclude:

- `retrieval.query` text itself beyond presence/length
- raw `RetrievedChunk.text`
- generated `chat_response`
- personal daily-log or diagnosis text

`answer_composer` may create a user-facing message with Cornell citation titles and URLs, but observability records only counts and sanitized citation/chunk metadata.
## Phase 14 RAG Runtime Observability

Phase 14 adds local vector-store operations and an optional backend-only Cornell RAG API. These operations keep the same safe-metadata principle.

Allowed operational metadata:

- RAG provider name
- embedding model name and dimension
- Chroma collection name
- corpus chunk count
- whether `rag_data/chroma/` exists
- whether the collection is compatible
- readiness booleans from `/ready`
- bounded provider error class/status such as HTTP 429/5xx

Disallowed metadata:

- API key values or key prefixes
- raw user question text beyond presence/length
- raw Cornell chunk content
- raw generated answer text
- pet profile, daily log, diagnosis, owner note, or hospital text

Current operational state:

- Embeddings now use OpenAI `text-embedding-3-small`; credential values and provider request payloads must not be logged.
- The local Chroma collection has been generated and may be reported through bounded operational metadata such as collection name, chunk count, document count, embedding model, dimension, and evaluation pass/fail counts.
- Query normalization may be summarized through derived counts or feature flags, but raw user question text and raw expanded prompt text must not be logged.

## Phase 17 Social Chat Observability

Social/profile continuity turns route through `intent_classifier -> chat_agent -> END` when `intent=social_chat`, no safety screening is required, and no red flag is mentioned.

Allowed metadata for this path:

- route and node names
- `intent=social_chat`
- request and conversation ids
- locale and timezone
- whether RAG query/chunk counts are zero

Disallowed metadata remains unchanged:

- raw `conversation_history` content
- generated social response text
- pet name or owner name from conversation/profile context

The implementation may use recent conversation history and pet context in prompts, but observability must only record derived route/count/status metadata.