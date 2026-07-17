# PetCare-AI Agent Implementation Roadmap

This document defines the implementation sequence for the PetCare-AI LangGraph agent. It is based on `docs/agent-architecture.md`, `docs/agent-implementation-spec.md`, and `contracts/jsonschema/*`.

The active AI backend now lives under `members/kunho/ai/`: source packages are in `members/kunho/ai/src`, tests are in `members/kunho/ai/tests`, and root `pyproject.toml` points to those paths.

## Roadmap Principles

- Freeze contracts and pure logic before wiring LangGraph edges.
- Keep the final emergency decision in deterministic rule-based validation.
- Use LLM calls only for structured intent, state, checklist, answer, and handoff outputs.
- Treat RAG as an external provider boundary.
- Include LangSmith tracing from Phase 0, not as a later optional feature.
- Do not implement location-based hospital search or real email sending in the MVP.
- Do not change existing DB/API contracts; use only documented contracts in `docs/api-endpoints.md` and `docs/database-schema.md`.

## Phase 0. Project Foundation

Goal: Create the Python package foundation needed to implement the LangGraph agent.

Tasks:

- Initialize `members/kunho/ai/src/petcare_agent`.
- Define Python dependency configuration in `pyproject.toml`.
- Add LangGraph, LangChain/OpenAI client, LangSmith, pydantic, and pytest dependencies.
- Define environment variable contracts:
  - `OPENAI_API_KEY`
  - `LANGSMITH_API_KEY`
  - `LANGSMITH_TRACING`
  - `PETCARE_API_BASE_URL`
- Add JSON schema loading paths for tests.
- Define LangSmith project/run naming conventions.
- Add a tracing helper skeleton.

Deliverables:

- Python package skeleton
- Dependency config
- Basic import test

Completion criteria:

- `pytest` can import the empty package.
- JSON schemas parse in tests.
- LangSmith tracing helper imports in enabled and disabled modes.

## Phase 1. Domain Models And Contracts

Goal: Freeze graph state and request/response models in code.

Tasks:

- Add `schemas/graph_state.py`:
  - `PetCareGraphState`
  - `GraphRequest`
  - `GraphResponse`
- Add `schemas/triage.py`:
  - `ChecklistItem`
  - `ChecklistTemplate`
  - `RiskResult`
  - `RuleHit`
- Add `schemas/llm_outputs.py`:
  - `IntentClassificationOutput`
  - `StateExtractionOutput`
  - `ChecklistExtractionOutput`
  - `AnswerGuardReviewOutput`
  - `HandoffSummaryOutput`
- Validate field-name alignment between JSON schemas and pydantic models.

Deliverables:

- Pydantic schemas
- Contract validation tests

Completion criteria:

- Sample request/response payloads pass pydantic validation.
- JSON schema examples match pydantic field names.

## Phase 2. Checklist Templates

Goal: Implement MVP triage checklist templates as data.

Tasks:

- Add checklist templates under `safety/checklists/`:
  - `cat_cough_triage`
  - `dog_cough_triage`
  - `vomiting_triage`
  - `diarrhea_triage`
  - `breathing_triage`
  - `seizure_triage`
  - `toxicity_triage`
  - `urinary_triage`
- Add `safety/checklist_loader.py`.
- Select templates by species and chief complaint, with fallback behavior.
- Define checklist item priority and question text.

Deliverables:

- Checklist template data
- Checklist loader

Completion criteria:

- Each chief complaint selects the expected template.
- Required red-flag items are present.
- Question priorities stay within the expected range.

## Phase 3. Rule-Based Safety Validator

Goal: Determine risk level from checklist state without LLM judgment.

Tasks:

- Add `safety/rules.py` with emergency, urgent, unknown, and non-emergency rules.
- Add `safety/validator.py` with rule priority: `emergency > urgent > unknown > non_emergency`.
- Return `needs_more_info` when required data is missing and `safety_question_turns < 2`.
- Return `unknown_after_max_questions` when required data is missing and `safety_question_turns >= 2`.
- Emit traceable rule hits.

Deliverables:

- Rule-based validator
- Rule unit tests

Completion criteria:

- Cat open-mouth breathing is emergency.
- Abnormal gum color is emergency.
- Missing required items follow the question-count rules.
- Cases without emergency/urgent hits and without missing data become non-emergency.

## Phase 4. Baseline And Change Detection

Goal: Compare current status with recent historical pet records.

Tasks:

- Add `nodes/db_context_loader.py`.
- Only run DB context loading when intent requires it.
- Call only documented APIs, especially `GET /api/pets/{pet_id}/handoff-context?days=3`.
- Forbid new DB tables and new API endpoints.
- Add graceful API fallback behavior.
- Add `nodes/baseline_builder.py` to summarize recent daily entries.
- Add `nodes/change_detector.py` to compare current status against baseline.
- Add DB response fixtures.

Deliverables:

- DB context adapter
- Baseline builder
- Change detector

Completion criteria:

- Missing recent records set `baseline_available=false`.
- New current symptoms are surfaced as `new_symptoms`.
- Worsened fields such as appetite are detected from baseline comparison.

## Phase 5. LLM Adapter And Structured Nodes

Goal: Connect LLM structured-output generation behind stable node boundaries.

Tasks:

- Add `llm/client.py` with provider/model wrappers and structured-output helper calls.
- Add these nodes:
  - `nodes/intent_classifier.py`
  - `nodes/state_updater.py`
  - `nodes/checklist_extractor.py`
  - `nodes/answer_guard.py`
  - `nodes/handoff_summary_builder.py`
- Add prompt templates.
- Define fallback behavior for LLM failures.

Deliverables:

- LLM adapter
- Structured-output node implementations
- Prompt files

Completion criteria:

- Intent classification separates general questions from symptom questions.
- General chat avoids DB context loading and safety guard.
- Red-flag mentions route into symptom-check flow.
- Checklist extraction maps user answers to checklist item updates.
- Answer guard fixes disallowed statements.
- LLM behavior is testable with mocks.

## Phase 6. Question Manager

Goal: Ask at most two natural follow-up questions for missing required checklist items.

Tasks:

- Add `nodes/question_manager.py`.
- Select missing required items by priority.
- Ask no more than two follow-up questions.
- Avoid asking about already answered items.
- Increment `safety_question_turns`.
- Route follow-up answers back through `state_updater -> change_detector -> safety_guard`.

Deliverables:

- Question manager node
- Question selection tests

Completion criteria:

- No more than two questions are asked in one turn.
- Answered items are not re-asked.
- After two question turns, the graph moves to unknown fallback instead of asking again.

## Phase 7. Chat, Emergency, Handoff Nodes

Goal: Generate user-facing final answers and summaries.

Tasks:

- Add `nodes/chat_agent.py`.
- Add risk-level response skeletons.
- Confirm hospital-visit intent when needed.
- Generate RAG queries.
- Add `nodes/emergency_agent.py` for immediate-care guidance.
- Add `graphs/subgraphs/handoff.py` for non-emergency handoff summaries.
- Generate email drafts only; do not send real emails.
- Add response composer logic.

Deliverables:

- Chat agent
- Emergency agent
- Handoff subgraph
- Graph response composer

Completion criteria:

- Emergency answers do not ask whether the user wants to visit a hospital; they advise immediate care.
- Urgent and non-emergency answers can confirm hospital-visit intent.
- Non-emergency handoff runs only when hospital visit intent is yes.
- Emergency summaries are produced by the Emergency Agent, not the non-emergency handoff subgraph.

## Phase 8. RAG Adapter Integration

Goal: Call the RAG system through a stable provider boundary.

Tasks:

- Add `rag/adapter.py` with `retrieve(query, filters, top_k=5)`.
- Add timeout and error fallback handling.
- Add `nodes/rag_agent.py`.
- Pass species, chief complaint, and risk-level filters.
- Store retrieved chunks in graph state.
- Define safe fallback copy when RAG is unavailable.

Deliverables:

- RAG adapter
- RAG node
- Adapter mock tests

Completion criteria:

- Mocked RAG chunks are stored in `graph_state.retrieval.chunks`.
- RAG failure does not kill the graph.
- The graph continues with a safe general answer when evidence is unavailable.

## Phase 9. LangGraph Wiring With LangSmith Trace Hooks

Goal: Connect all nodes into the Assessment Graph.

Tasks:

- Add `graphs/assessment_graph.py`.
- Register all nodes.
- Set `intent_classifier` as the graph entrypoint.
- Add conditional edges for general-chat vs symptom-check paths.
- Add safety conditional edges.
- Add the needs-more-info loop.
- Add emergency and non-emergency branching.
- Add graph runner.
- Connect LangSmith trace metadata hooks.
- Record node name, risk level, triggered rules, and key route metadata.

Deliverables:

- Compiled LangGraph graph
- Graph runner
- LangSmith-instrumented graph runner
- Integration tests

Completion criteria:

- General-chat inputs follow `intent_classifier -> chat_agent -> rag_agent -> answer_guard`.
- Symptom inputs follow the DB/context/safety path.
- Follow-up answers re-enter safety validation.
- Emergency cases branch to Emergency Agent.
- Non-emergency visit-intent cases branch to Handoff Subgraph.
- General chat does not trigger emergency checklist questions.
- LangSmith traces show node transitions and Safety Guard output.

## Phase 10. Existing API Adapter Integration

Goal: Connect the graph to the backend flow without redesigning existing APIs.

Tasks:

- Freeze the list of callable APIs from `docs/api-endpoints.md`.
- Use `GET /api/pets/{pet_id}/handoff-context?days=3` for context.
- Interpret response fields from `docs/database-schema.md`.
- Forbid new endpoints, new DB tables, and existing API request/response changes.
- Treat graph request/response schemas as the agent runtime contract.
- Define adapter boundaries so backend callers can invoke the Assessment Graph runner.
- Add API error mapping.

Deliverables:

- Existing API adapter
- Graph runtime adapter
- Request/response validation

Completion criteria:

- Code does not call undocumented DB/API surfaces.
- DB Context Loader passes with `handoff-context?days=3` fixtures.
- Internal graph failures return safe fallback responses.

## Phase 11. LangSmith Observability Hardening

Goal: Make LangSmith traces usable for operations and evaluation.

Tasks:

- Add tracing config checks.
- Define trace naming and versioning.
- Record node metadata:
  - route
  - intent
  - checklist extraction result
  - triggered rules
  - change detection
  - RAG metadata
  - answer guard status
- Define personal/sensitive data logging policy.

Deliverables:

- LangSmith trace integration
- Trace metadata helper

Completion criteria:

- A single request can be inspected across the full graph.
- Safety Guard rule hits appear in traces.
- RAG chunk metadata appears in traces without raw sensitive content.

## Phase 12. Evaluation And Release Gates

Goal: Validate safety behavior and regression quality.

Tasks:

- Add synthetic evaluation data for:
  - emergency cases
  - urgent cases
  - non-emergency cases
  - unknown or missing-info cases
  - baseline-deviation cases
- Define release thresholds.
- Run unit, integration, and evaluation checks in CI.

Deliverables:

- Evaluation dataset
- Release gate config
- CI test command

Completion criteria:

- Emergency red-flag recall meets the target.
- Known non-emergency cases avoid over-escalation.
- Unknown cases do not fall directly to non-emergency.
- Follow-up question loops remain capped at two turns.

## Suggested Milestones

| Milestone | Scope | Completion criteria |
| --- | --- | --- |
| M1 Safety Core | Schemas, checklist, rule validator | Safety unit tests pass without LLM calls |
| M2 Context Core | DB loader, baseline, change detector | Baseline/change tests pass |
| M3 LLM Core | Structured-output nodes | LLM mock tests pass |
| M4 Graph MVP | Assessment Graph wiring, question loop, emergency/chat branching, LangSmith trace hooks | Graph integration tests pass and traces are inspectable |
| M5 Response MVP | Chat, answer guard, handoff, response composer | Request-to-response E2E tests pass |
| M6 Observability/Eval | LangSmith, eval dataset, release gates | Safety eval thresholds pass |

## Implementation Order Recommendation

1. Phase 1: domain models
2. Phase 2: checklist templates
3. Phase 3: rule validator
4. Phase 4: baseline/change detector
5. Phase 6: question manager
6. Phase 9: LangGraph wiring with mocked LLM/RAG/DB
7. Phase 5: real LLM structured nodes
8. Phase 8: RAG adapter integration
9. Phase 7: chat/emergency/handoff response generation
10. Phase 10-12: existing API adapter, observability, eval

## Open Decisions Before Coding

- Choose the Python dependency manager: pip/requirements, uv, or poetry.
- Choose how to handle conversation state without DB changes: existing storage, cache, or client replay.
- Confirm LLM model name and provider.
- Confirm whether the RAG adapter import path should be sync or async.
## Phase 12 Implemented Update. Triage/Handoff Contracts And Golden Set

Status: implemented on 2026-07-16.

Phase 12 was expanded from generic evaluation/release gates into a concrete triage/handoff contract separation plus golden scenario readiness pass.

### Delivered

- Added `contracts/jsonschema/internal-triage-assessment.schema.json`.
- Added `contracts/jsonschema/hospital-handoff-summary.schema.json`.
- Added Pydantic handoff/internal triage models in `members/kunho/ai/src/petcare_agent/schemas/handoff.py`.
- Added canonical red-flag mapping in `members/kunho/ai/src/petcare_agent/safety/red_flags.py`.
- Updated Safety Guard to populate `state.internal_triage_assessment`.
- Updated Question Manager to mirror selected follow-up question text into internal triage.
- Updated Handoff Subgraph so `handoff.summary_json` is a six-section hospital handoff payload.
- Updated Graph Response contract so `handoff.summary_json` is returned.
- Updated LLM structured-output contracts so `StateExtractionOutput` includes `course_pattern` and `HandoffSummaryOutput` no longer exposes `risk_level`, `triggered_rules`, or `metadata`.
- Added `members/kunho/ai/tests/fixtures/triage_handoff_golden.jsonl` with G01-G12 golden scenario rows.
- Added contract and fixture tests for the new separation.

### Completion Criteria Met

- Hospital handoff JSON has exactly six top-level content sections.
- Hospital handoff JSON excludes internal fields: `risk_level`, `confidence`, `missing_items`, `triggered_rules`, `decision_basis`, `sources`, and `attachments`.
- Internal triage schema owns `risk_level`.
- Follow-up questions are capped at two.
- `baseline_comparison.window_days` is fixed at `3`.
- Empty medical background stays as empty arrays and does not create extra medical-background questions.
- Existing graph routing behavior remains compatible.

### Verification

```powershell
python -m pytest members/kunho/ai/tests
```

Result:

```text
137 passed, 1 warning
```

### Remaining Phase 12/13 Work

- Build an executable eval runner that consumes `members/kunho/ai/tests/fixtures/triage_handoff_golden.jsonl` and maps each row to mocked structured LLM/API outputs.
- Add release thresholds for emergency recall, unknown handling, and over-escalation control.
- Decide whether emergency handoff summary should reuse `HospitalHandoffSummary` in production output or remain optional behind Emergency Agent UX.

## Phase 13 Implemented Update. Evidence-First Cornell RAG Composition

Phase 13 integrates the teammate Cornell RAG design as a native provider-shaped boundary instead of a bolt-on answer API.

Implemented code scope:

- Added `members/kunho/ai/src/petcare_agent/rag/cornell.py` with `CornellRAGAdapter`.
- Added `members/kunho/ai/src/petcare_agent/nodes/evidence_planner.py`.
- Added `members/kunho/ai/src/petcare_agent/nodes/answer_composer.py`.
- Rewired `assessment_graph` so non-emergency/general flows run `evidence_planner -> rag_agent -> answer_composer -> answer_guard`.
- Extended `RetrievalState` with citations, provider, insufficient evidence, and errors.
- Kept `chat_agent.py` as a compatibility wrapper only.

Acceptance checks:

- General chat runs through evidence planning, RAG retrieval, answer composition, and answer guard.
- Urgent, non-emergency, and unknown symptom cases compose answers after RAG retrieval.
- Cornell adapter rejects unsupported species before provider calls.
- RAG trace metadata includes provider/citation/evidence sufficiency counts without raw chunk text.

Resolved by Phase 14:

- The teammate `petcare_rag` package is vendored into this repository.
- Corpus files live under `rag_data/chunks/`.
- Retrieval gold data lives under `rag_data/evaluation/`.
- Generated Chroma state under `rag_data/chroma/` is excluded from git.

Still open:

- Add live integration tests gated by `OPENAI_API_KEY` and a local Chroma index.

## Phase 14 Implemented Update. Cornell Vector Store Readiness

Phase 14 vendors the teammate Cornell RAG runtime and corpus into this repository and prepares the generated Chroma vector-store location.

Implemented:

- `members/kunho/ai/src/petcare_rag/` runtime package
- `rag_data/chunks/cornell_pet_health_chunks.jsonl`
- `rag_data/evaluation/cornell_retrieval_gold.jsonl`
- `tools/_project_env.py`
- `tools/manage_cornell_rag_db.py`
- `tools/run_cornell_rag.py`
- `tools/run_cornell_rag_api.py`
- `requirements-rag.txt`
- optional `rag` dependencies in `pyproject.toml`
- `rag_data/chroma/` gitignore rule
- `docs/cornell-rag-vector-store.md`

Current provider/model contract:

- Embeddings: OpenAI `text-embedding-3-small`, 1536 dimensions.
- Vector DB: local Chroma collection `cornell_pet_health_text_embedding_3_small_1536`.
- Standalone answer generation: OpenAI structured output, default `gpt-5.4-mini`.
- Graph response composition: `answer_composer` and `answer_guard` inside `petcare_agent`.

Current status: embedding provider has been switched to OpenAI `text-embedding-3-small`; the local Chroma collection has been generated with 732 chunks across 282 Cornell documents, inspected successfully, and evaluated at 12/12 gold retrieval cases. Query embedding now includes reusable Korean/English veterinary term normalization plus species context before Chroma search.

<!-- BEGIN PHASE17_SOCIAL_CHAT_ROADMAP_ADDENDUM -->
## Phase 17 Implemented Update. LLM-backed Social Chat Routing

Phase 17 is implemented and supersedes earlier roadmap notes that described `chat_agent.py` as only a compatibility wrapper.

Implemented:

- Added `social_chat` as a first-class structured intent.
- Routed `social_chat` directly to `chat_agent`.
- Rebuilt `chat_agent` as an LLM-backed node that receives current input, recent conversation history, locale, and pet context.
- Added `SocialChatOutput` to the structured-output schema contract.
- Added optional `GraphRequest.conversation_history` for stateless clients.
- Kept `general_chat` on the evidence-first Cornell path for pet-care information requests.
- Removed production fallback logic that heuristically reclassifies `general_chat` as social chat after the classifier.

Acceptance checks:

- Greetings do not run Cornell RAG or `answer_guard`.
- User-name and pet-name recall questions are answered by the LLM-backed chat node using available history/context.
- Unknown remembered facts are handled honestly instead of fabricating.
- Pet-care information questions still run through `evidence_planner -> rag_agent -> answer_composer -> answer_guard`.

Latest local verification:

```text
155 passed, 1 warning
```
<!-- END PHASE17_SOCIAL_CHAT_ROADMAP_ADDENDUM -->

## Phase 18 Implemented Update. Documentation Sync With Current Code Layout

Phase 18 updates reader-facing documentation to match the current repository state after the member-owned implementation layout became the active package path.

Implemented documentation sync:

- Marked `members/kunho/ai/src` and `members/kunho/ai/tests` as the active package/test roots.
- Updated harness docs for `current-assessment-graph`, `current`, and `langgraph-v1` adapter names.
- Documented `--once`, `--replay`, `--list-pets`, transcript behavior, and fixture bundle compatibility.
- Updated graph request/response examples to match current JSON Schema required fields.
- Clarified that graph execution uses in-process `CornellRAGAdapter` by default.
- Documented optional `petcare_rag.api` endpoints as backend-only RAG diagnostics, not required app routes.
- Refreshed observability notes for Phase 17 social chat while preserving the stable `phase13.v1` trace schema label.

Recommended verification after this sync:

```powershell
python -m pytest
```