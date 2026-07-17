# PetCare-AI Agent Architecture

PetCare-AI uses a LangGraph Assessment Graph to turn a user message plus pet context into a safe, evidence-aware response. The graph separates internal triage decisions from user-facing and veterinarian-facing output.

The active AI backend lives under `members/kunho/ai/`: source packages are in `members/kunho/ai/src`, and tests are in `members/kunho/ai/tests`. Root `pyproject.toml` discovers packages from that member-owned source path.

## High-Level Goals

- Preserve existing backend API and database contracts.
- Keep emergency and risk routing deterministic.
- Use LLMs for structured understanding and language generation, not final safety authority.
- Retrieve official-source evidence before drafting pet-care information answers.
- Keep social/profile conversation separate from medical or evidence workflows.
- Expose trace metadata that can be audited without leaking raw sensitive content.


## Runtime And API Boundaries

The primary application boundary is Python runtime execution, not a graph HTTP route in this repository.

- `petcare_agent.runtime.adapter.run_graph_request(...)` validates `GraphRequest`, runs the graph, and validates `GraphResponse`.
- `build_existing_api_runtime_adapter(...)` wires DB context to the documented handoff-context endpoint and uses `CornellRAGAdapter` by default.
- `petcare_rag.api` is an optional FastAPI service for backend-only Cornell RAG diagnostics. The graph does not need this service for normal execution.
- The optional RAG API requires `PETCARE_RAG_SERVICE_TOKEN` only for `POST /v1/rag/answer`; graph-internal retrieval does not use that token.
## Assessment Graph

Current implemented routing shape:

```text
db_context_loader
  -> intent_classifier
      -> chat_agent
          when intent == social_chat
      -> evidence_planner
          -> rag_agent
          -> answer_composer
          -> answer_guard
          when intent == general_chat
      -> baseline_builder
          -> state_updater
          -> change_detector
          -> safety_guard
              -> question_manager
              -> emergency_agent
              -> evidence_planner
          -> rag_agent
          -> answer_composer
          -> answer_guard
          -> optional handoff_subgraph
          when intent needs safety screening
```

The graph starts by loading available context, then classifies the turn. Social turns are answered directly by `chat_agent`. Pet-care information turns use Cornell evidence retrieval before answer drafting. Symptom, follow-up, and handoff turns run through baseline comparison and deterministic safety validation.

## Implementation Contract

- Do not add new DB tables or new backend endpoints for the assessment flow.
- Use `GET /api/pets/{pet_id}/handoff-context?days=3` for backend context.
- Use graph request/response schemas as the agent runtime contract.
- Keep final risk decisions in Safety Guard and the rule validator.
- Keep hospital handoff JSON free of internal routing fields.

## Node Responsibilities

### DB Context Loader

Loads recent pet context from the documented backend handoff-context API. It is allowed to degrade gracefully when the backend is unavailable. It must not invent records or call undocumented endpoints.

### Intent Classifier

Classifies the turn and performs turn understanding. It returns structured intent, extracted current pet state when relevant, and an optional social response for social/profile turns.

Primary intents:

- `social_chat`
- `general_chat`
- `symptom_check`
- `followup_answer`
- `handoff_request`

### Baseline Builder

Summarizes recent daily entries into a compact baseline window. The current window is three days.

### State Updater

Updates graph state from structured turn understanding. It skips a second LLM call when the classifier already produced usable extracted state.

### Change Detector

Compares current user-reported status with baseline context and records new, worsened, improved, unchanged, or unknown fields.

### Safety Guard

Combines checklist extraction, current state, baseline comparison, and deterministic rules to produce internal triage state. It owns final risk routing.

Risk labels:

- `emergency`
- `urgent`
- `non_emergency`
- `unknown`

### Question Manager

Asks at most two follow-up questions for missing required safety information. It does not repeat already answered items.

### Emergency Agent

Produces immediate-care guidance when emergency rules fire. Emergency flow does not ask the user whether they want to visit a hospital before advising immediate care.

### Evidence Planner

Builds or preserves the retrieval query. It does not draft the final user-facing answer.

### RAG Agent

Retrieves official-source Cornell evidence through `petcare_agent.rag.cornell.CornellRAGAdapter`. It fills retrieval chunks, citations, provider metadata, insufficiency flags, and errors.

### Answer Composer

Combines safety guidance, baseline/change context, current symptoms, and Cornell citations into the user-facing draft.

### Answer Guard

Reviews the final draft for unsafe, overconfident, or disallowed medical language.

### Chat Agent

Handles `social_chat` turns. It may use current input, locale, pet context, and conversation history. It must not call Cornell RAG, attach citations, ask hospital-visit questions, or add medical safety boilerplate for social turns.

### Handoff Subgraph

Builds veterinarian-facing handoff output only when routing allows it. It creates a display summary, optional draft email text, and structured handoff JSON.

## Internal Triage Boundary

`internal_triage_assessment` is graph-internal state produced by Safety Guard and Question Manager. It supports routing and follow-up policy.

It may contain:

- `risk_level`
- `red_flag_inputs`
- `clinical_inputs`
- `needs_followup`
- `followup_questions`

This object is not the veterinarian-facing handoff payload.

## Hospital Handoff Boundary

`handoff.summary_json` is the veterinarian-facing payload. It must use exactly six top-level sections:

1. `patient`
2. `visit_reason`
3. `clinical_course`
4. `baseline_comparison`
5. `triage_assessment`
6. `medical_background`

The payload must not contain internal fields such as `risk_level`, `confidence`, `missing_items`, `triggered_rules`, `decision_basis`, `sources`, or `attachments`.

`baseline_comparison.window_days` is fixed at `3`.

## Hospital Visit Intent

Handoff generation runs only when:

```text
risk_level in ["urgent", "non_emergency", "unknown"]
AND hospital_visit_intent == "yes"
```

Emergency cases remain on the Emergency Agent path.

## Graph State

The graph state carries:

- request metadata and locale
- pet context
- conversation history
- extracted current state
- baseline summary
- change detection results
- checklist state
- internal triage assessment
- retrieval state
- draft response
- final response
- handoff output
- trace metadata

Retrieval state shape:

```json
{
  "retrieval": {
    "query": "...",
    "chunks": [],
    "citations": [],
    "provider": "cornell",
    "insufficient_evidence": false,
    "errors": []
  }
}
```

## Cornell RAG Runtime

The Cornell RAG runtime is vendored locally:

```text
members/kunho/ai/src/petcare_rag/
rag_data/chunks/cornell_pet_health_chunks.jsonl
rag_data/evaluation/cornell_retrieval_gold.jsonl
rag_data/chroma/
```

Generated Chroma files under `rag_data/chroma/` are local runtime state and are gitignored.

Provider/model split:

| Concern | Runtime |
| --- | --- |
| Corpus embeddings | OpenAI `text-embedding-3-small`, 1536 dimensions |
| Query embeddings | OpenAI `text-embedding-3-small` |
| Local vector store | Chroma collection `cornell_pet_health_text_embedding_3_small_1536` |
| Standalone RAG generation | OpenAI structured output, default `gpt-5.4-mini` |
| Graph answer composition | `answer_composer` plus `answer_guard` |

The graph consumes Cornell evidence through `CornellRAGAdapter`; frontend clients do not call the Cornell runtime directly.

## LangSmith Observability Layer

Trace metadata should include route, intent, node name, risk level, triggered rules, change detection summaries, retrieval provider/citation counts, insufficiency flags, and answer guard status.

Traces should avoid raw personal logs, raw diagnosis text, and raw retrieved chunk text when metadata is enough.

## Phase Addenda Summary

### Phase 12: Triage/Handoff Boundary

Internal triage and hospital handoff are separate contracts. Internal triage owns risk routing. Hospital handoff JSON owns veterinarian-facing content only.

The golden fixture lives at `members/kunho/ai/tests/fixtures/triage_handoff_golden.jsonl`.

### Phase 13: Evidence-First RAG

General pet-care and non-emergency symptom answers retrieve evidence before drafting. The implemented path is `evidence_planner -> rag_agent -> answer_composer -> answer_guard`.

### Phase 14/16: Local Cornell Vector Store

The local Cornell vector store uses OpenAI embeddings and Chroma. The generated collection contains 732 chunks across 282 Cornell documents and passes the current 12/12 retrieval gold set.

### Phase 17: LLM-Backed Social Chat

`social_chat` is a first-class route for greetings, user-name recall, pet-name recall, and other conversational continuity turns. These turns do not call Cornell RAG or the medical answer guard.