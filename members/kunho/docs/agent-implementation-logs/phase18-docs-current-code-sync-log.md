# Agent Phase 18 Documentation Current-Code Sync Log

Date: 2026-07-17

## Reason

The repository documentation needed to match the current implementation layout and runtime behavior. The active agent code now lives under `members/kunho/ai/src`, tests live under `members/kunho/ai/tests`, and root `pyproject.toml` points package discovery and pytest configuration to those paths.

The docs also needed to distinguish normal graph execution from the optional Cornell RAG FastAPI boundary. The Assessment Graph uses in-process `CornellRAGAdapter` by default; `petcare_rag.api` exists for backend-only diagnostics and smoke tests.

## Updated Documentation

- `docs/current-implementation-status.md`
  - Added active package/test roots from `pyproject.toml`.
  - Added runtime validation and existing API adapter notes.
  - Documented optional local RAG HTTP boundary.
  - Added harness and pytest commands for the current member-owned implementation path.

- `docs/agent-harness.md`
  - Rewrote around the actual CLI options in `petcare_agent.harness.cli`.
  - Updated examples to use `examples/data_bundles/petcare_db_v1_demo`.
  - Documented `--once`, `--replay`, `--list-pets`, transcript options, visit-intent commands, built-in adapter names, and custom adapter loading.

- `docs/api-endpoints.md`
  - Added Assessment Graph runtime request/response boundary.
  - Updated conversation history guidance for social/profile continuity.
  - Added optional local Cornell RAG API endpoints: `/health`, `/ready`, and `/v1/rag/answer`.
  - Preserved the personal-data boundary for official-source RAG.

- `docs/cornell-rag-vector-store.md`
  - Added `api.py` and `tools/run_cornell_rag_api.py` to the runtime/tooling inventory.
  - Documented optional FastAPI usage and `PETCARE_RAG_SERVICE_TOKEN` scope.

- `docs/database-schema.md`
  - Corrected `daily_entries` to include an `id` primary key plus date `record_date`.
  - Added harness fixture compatibility paths and accepted wrapper keys.

- `docs/agent-observability-policy.md`
  - Updated date and scope.
  - Clarified that `phase13.v1` is the stable trace schema label even though the implementation has Phase 17 behavior.
  - Added social-chat observability rules.

- `docs/agent-architecture.md`
  - Clarified current source/test roots and runtime/API boundaries.

- `docs/agent-implementation-spec.md`
  - Updated GraphRequest and GraphResponse examples to match current JSON Schema contracts.
  - Updated RAG adapter contract to return `list[RetrievedChunk]`, with `rag_agent` filling retrieval state and citations.

- `docs/agent-implementation-roadmap.md`
  - Added this Phase 18 documentation sync as an implemented update.

## Current Technical Snapshot

- Active source root: `members/kunho/ai/src`
- Active test root: `members/kunho/ai/tests`
- Runtime adapter: `petcare_agent.runtime.adapter`
- Graph request/response contracts: `contracts/jsonschema/agent-graph-request.schema.json` and `contracts/jsonschema/agent-graph-response.schema.json`
- Harness built-ins: `current-assessment-graph`, `current`, `langgraph-v1`
- Default graph RAG path: in-process `CornellRAGAdapter`
- Optional RAG HTTP API: `petcare_rag.api`, launched by `tools/run_cornell_rag_api.py`
- RAG API token scope: only `POST /v1/rag/answer`

## Verification

Executed:

```powershell
python -m pytest
```

Result:

```text
161 passed, 1 warning in 4.20s
```