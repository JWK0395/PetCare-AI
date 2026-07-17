# Current Implementation Status

Date: 2026-07-17
Repository: PetCare-AI

This document is the current reader-facing snapshot of the implemented agent/RAG structure. Historical phase logs remain under `docs/agent-implementation-logs/`.

## Implemented Backend Shape

The active AI backend lives under `members/kunho/ai/src` and is split into two packages:

```text
members/kunho/ai/src/petcare_agent/   # Assessment Graph, safety, contracts, tracing, native RAG adapter
members/kunho/ai/src/petcare_rag/     # Vendored Cornell RAG runtime and Chroma management
```

`pyproject.toml` is also aligned to this member-owned implementation path:

- package discovery: `members/kunho/ai/src`
- pytest pythonpath: `members/kunho/ai/src`
- pytest testpaths: `members/kunho/ai/tests`, plus optional root `tests`
- console script: `petcare-agent-playground = petcare_agent.harness.cli:main`

The Assessment Graph is the primary application path. The vendored Cornell runtime is used in-process through `CornellRAGAdapter`; no RAG server is required for graph execution.

## Assessment Graph Flow

Implemented graph path:

```text
db_context_loader
  -> intent_classifier  # turn understanding: intent + state extraction + optional social response
      -> chat_agent  # social_chat / lightweight conversational turns
      -> evidence_planner
          -> rag_agent
          -> answer_composer
          -> answer_guard
      -> baseline_builder
          -> state_updater  # skips LLM when turn understanding already extracted state
          -> change_detector
          -> safety_guard
              -> question_manager
              -> emergency_agent
              -> evidence_planner
          -> rag_agent
          -> answer_composer
          -> answer_guard
          -> optional handoff_subgraph
```

Key behavior:

- `evidence_planner` prepares `retrieval.query` only.
- `rag_agent` retrieves official-source evidence through `RAGAdapter` and fills chunks, citations, provider, insufficiency, and errors.
- `answer_composer` writes the user-facing draft after safety and retrieval.
- `answer_guard` remains the final response safety review.
- `GraphRequest.conversation_history` is optional and is copied into graph state when provided; stateful harness sessions append user/assistant turns automatically.
- `intent_classifier` now uses a single `TurnUnderstandingOutput` call to classify intent, extract current pet state, and optionally draft social-chat responses.
- `chat_agent.py` remains the compatibility import path for evidence planning and is also the active lightweight `chat` node for `social_chat` turns. It reuses the precomputed social response when available, avoiding a second LLM call.
- `state_updater` skips its LLM call when turn understanding has already extracted pet state.
- Runtime validation lives in `petcare_agent.runtime.validation` and checks both Pydantic models and the JSON Schema files under `contracts/jsonschema/`.
- `build_existing_api_runtime_adapter(...)` wires the graph to the documented `/api/pets/{pet_id}/handoff-context` boundary and uses `CornellRAGAdapter` by default.

## RAG Provider Split

Cornell evidence integration has two layers:

```text
petcare_agent.rag.cornell.CornellRAGAdapter
  -> petcare_rag.retrieve(...)
  -> local Chroma collection under rag_data/chroma/
```

The graph-internal Cornell RAG path uses:

| Purpose | Provider/model | Notes |
| --- | --- | --- |
| Corpus embeddings | OpenAI `text-embedding-3-small` | 1536 dimensions, stored in local Chroma |
| Query embeddings | OpenAI `text-embedding-3-small` | Korean/English domain normalization plus species context before retrieval |
| Assessment Graph response composition | Project graph nodes | `answer_composer` uses retrieved evidence; final pass is `answer_guard` |

The Chroma collection name is `cornell_pet_health_text_embedding_3_small_1536`.

## Current Vector Store State

Prepared and present:

- `rag_data/chunks/cornell_pet_health_chunks.jsonl`
- `rag_data/evaluation/cornell_retrieval_gold.jsonl`
- generated `rag_data/chroma/` local Chroma state, intentionally gitignored
- `requirements-rag.txt`
- `[project.optional-dependencies].rag`
- CLI wrappers in `tools/`, including the optional FastAPI runner

Generated collection:

- Collection: `cornell_pet_health_text_embedding_3_small_1536`
- Embedding model: `text-embedding-3-small`
- Embedding dimension: `1536`
- Corpus SHA-256: `21e3f445a63ccbf9d6c82b798a7aae2e0cd9cac4e54554cbb7d3cec77ad80ae6`
- Total chunks: `732`
- Unique documents: `282`
- Species chunks: `dog=418`, `cat=314`
- Metadata issues: `0`
- Wrong-dimension vectors: `0`

Latest implementation state:

- Embedding code calls OpenAI embeddings with `model="text-embedding-3-small"` and `dimensions=1536`.
- Gemini embedding code and `GEMINI_API_KEY` runtime requirement have been removed from current code/config.
- Query embedding now uses reusable Korean/English veterinary domain normalization and species context; it does not encode gold case IDs or expected document IDs.
- `python tools/manage_cornell_rag_db.py evaluate` passes `12/12` gold retrieval cases.

Optional local RAG HTTP boundary:

- `tools/run_cornell_rag_api.py` runs `petcare_rag.api:app` for backend-only diagnostics or integration smoke tests.
- `/health` does not require a token.
- `/ready` checks `OPENAI_API_KEY`, `PETCARE_RAG_SERVICE_TOKEN`, local Chroma presence, collection compatibility, and the expected 732 chunk count.
- `/v1/rag/answer` requires `X-PetCare-Token` to match `PETCARE_RAG_SERVICE_TOKEN`.
- The Assessment Graph still uses the in-process adapter path by default; the HTTP RAG API is not required for normal graph execution.

## Operational Commands

With a valid `OPENAI_API_KEY` configured:

```powershell
python tools/manage_cornell_rag_db.py check
python tools/manage_cornell_rag_db.py index
python tools/manage_cornell_rag_db.py inspect
python tools/manage_cornell_rag_db.py evaluate
```

A local graph-internal retrieval path can be spot-checked with:

```powershell
python tools/run_cornell_rag.py --species dog --question "My dog ate chocolate. What should I watch for?" --debug
```

Harness and tests use the member implementation path configured in `pyproject.toml`:

```powershell
python -m petcare_agent.harness --data-zip .\examples\data_bundles\petcare_db_v1_demo --pet-id 1 --once "My cat is coughing today."
python -m pytest
```
