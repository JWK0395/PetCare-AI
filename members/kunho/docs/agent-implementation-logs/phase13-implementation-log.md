# Agent Phase 13 Implementation Log

Date: 2026-07-16
Repository: PetCare-AI
Scope: Evidence-first Cornell RAG integration and answer composition

## Reference Inputs

This phase follows the teammate Cornell RAG review and the decision to make it feel native to PetCare-AI rather than bolted on as a separate answer API.

Key design decisions:

- Keep Safety Guard as the owner of emergency/urgent/non-emergency routing.
- Use Cornell RAG only as official-source evidence, not emergency judgement.
- Do not send pet profile, daily logs, diagnosis text, or other personal records to the Cornell provider.
- Preserve Cornell source title, URL, section path, chunk id, and provider metadata.
- Compose the user-facing answer after retrieval, then pass it through Answer Guard.

## Implemented Scope

### Retrieval State

Updated `ai/src/petcare_agent/schemas/graph_state.py`:

- Added `RAGCitation`.
- Extended `RetrievalState` with:
  - `citations`
  - `provider`
  - `insufficient_evidence`
  - `errors`

This lets downstream nodes use structured retrieval evidence without inspecting provider-specific payloads.

### Cornell RAG Adapter

Added `ai/src/petcare_agent/rag/cornell.py`.

`CornellRAGAdapter` adapts the teammate RAG `retrieve(question, species, top_k)` style result into the project `RetrievedChunk` contract. It:

- accepts only `cat` and `dog`
- rejects unknown/unsupported species before provider calls
- maps `document_id` to `source_id`
- maps `content` to `text`
- maps `similarity` or `distance` to `score`
- preserves `canonical_url`, `section_path`, species, provider, and Cornell institution metadata
- fails closed with an empty result if the provider cannot be imported or called

Updated `ai/src/petcare_agent/rag/__init__.py` to export `CornellRAGAdapter`.

### Evidence Planner Node

Added `ai/src/petcare_agent/nodes/evidence_planner.py`.

The node prepares `state.retrieval.query` and does not draft a user-facing answer. It clears stale retrieval evidence when it creates a new query. This replaces the old behavior where `chat_agent` drafted `chat_response` before RAG retrieval.

Updated `ai/src/petcare_agent/nodes/chat_agent.py` to remain as a compatibility wrapper around `plan_evidence_context`.

### RAG Agent Node

Updated `ai/src/petcare_agent/nodes/rag_agent.py` so retrieval now fills:

- `retrieval.chunks`
- `retrieval.citations`
- `retrieval.provider`
- `retrieval.insufficient_evidence`
- `retrieval.errors`

The node still uses the existing safe adapter wrapper, so provider exceptions fall back to empty evidence and graph execution continues.

### Answer Composer Node

Added `ai/src/petcare_agent/nodes/answer_composer.py`.

The node builds `chat_response` from:

- risk guidance
- general-chat disclaimer when applicable
- change-detection summary
- known symptoms
- Cornell official-source evidence availability
- Cornell citation titles and URLs
- hospital visit intent prompt

Answer Guard remains responsible for the final safety-language review.

### Graph Wiring

Updated `ai/src/petcare_agent/graphs/assessment_graph.py`.

New general/non-emergency path:

```text
intent_classifier
 -> evidence_planner
 -> rag_agent
 -> answer_composer
 -> answer_guard
```

Symptom path after Safety Guard:

```text
safety_guard
 -> question_manager      # needs more info
 -> emergency_agent       # emergency
 -> evidence_planner      # urgent / non_emergency / unknown
 -> rag_agent
 -> answer_composer
 -> answer_guard
 -> optional handoff_subgraph
```

Updated `NodeRoute` and `agent-graph-response.schema.json` with `evidence_planner` and `answer_composer`.

### Observability

Updated `ai/src/petcare_agent/tracing.py` so RAG metadata includes:

- provider
- insufficient evidence flag
- citation count
- error count

Raw retrieval query text and raw chunk text remain excluded from trace metadata.

## Tests Added Or Updated

Added:

- `ai/tests/test_answer_composer_node.py`
- `ai/tests/test_cornell_rag_adapter.py`

Updated:

- `ai/tests/test_chat_agent_node.py`
- `ai/tests/test_rag_agent_node.py`
- `ai/tests/test_assessment_graph_integration.py`

Coverage includes:

- evidence planner builds a retrieval query without drafting `chat_response`
- old `chat_agent` import path remains compatible
- answer composer uses safety context, change summary, symptoms, Cornell evidence, and citations
- insufficient Cornell evidence does not invent source-backed claims
- Cornell adapter maps teammate chunks to project chunks
- Cornell adapter rejects unsupported species before provider calls
- graph paths include `evidence_planner` and `answer_composer`
- RAG node derives citations/provider/insufficient evidence state

## Verification

Focused command:

```powershell
python -m pytest ai\tests\test_chat_agent_node.py ai\tests\test_answer_composer_node.py ai\tests\test_rag_agent_node.py ai\tests\test_cornell_rag_adapter.py ai\tests\test_assessment_graph_integration.py ai\tests\test_contracts.py ai\tests\test_observability_metadata.py -q
```

Result:

```text
30 passed, 1 warning
```

Full-suite command before docs updates:

```powershell
python -m pytest ai\tests -q
```

Result:

```text
All tests passed, 1 warning
```

The warning is the existing LangGraph/LangChain `allowed_objects` pending deprecation warning and is unrelated to this change.

## Docs Updated

Updated:

- `docs/agent-architecture.md`
- `docs/agent-implementation-spec.md`
- `docs/agent-implementation-roadmap.md`
- `docs/agent-observability-policy.md`
- `docs/api-endpoints.md`

## Known Limitations And Next Work

- `CornellRAGAdapter` expects the teammate `petcare_rag` package to be importable or injected as a retriever callable. This phase does not vendor the full RAG package/corpus into the repository.
- The current `answer_composer` is deterministic. A future LLM-backed composer can be added behind a structured-output contract once prompt and citation rules are fixed.
- Live Gemini/Chroma tests remain future work and should be gated by `GEMINI_API_KEY`, a built local Chroma index, and an explicit integration-test environment flag.
- If the team chooses to vendor the corpus, place input JSONL under `rag_data/chunks/` and keep generated Chroma files out of git.
