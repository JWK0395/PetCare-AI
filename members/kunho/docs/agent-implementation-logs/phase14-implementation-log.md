# Agent Phase 14 Implementation Log

Date: 2026-07-16
Repository: PetCare-AI
Scope: Vendored Cornell RAG runtime, corpus placement, and vector-store readiness

## Implemented Scope

This phase moves from adapter-only integration to local vector-store readiness.

Added runtime package:

- `ai/src/petcare_rag/__init__.py`
- `ai/src/petcare_rag/api.py`
- `ai/src/petcare_rag/manage_cornell_rag_db.py`
- `ai/src/petcare_rag/models.py`
- `ai/src/petcare_rag/pipeline.py`

Added corpus/evaluation files:

- `rag_data/chunks/cornell_pet_health_chunks.jsonl`
- `rag_data/evaluation/cornell_retrieval_gold.jsonl`

Added CLI wrappers:

- `tools/manage_cornell_rag_db.py`
- `tools/run_cornell_rag.py`
- `tools/run_cornell_rag_api.py`
- `tools/_project_env.py`

Updated configuration:

- Added `requirements-rag.txt`.
- Added `[project.optional-dependencies].rag` in `pyproject.toml`.
- Added Cornell RAG environment placeholders to `.env.example`.
- Added lightweight `.env` loading for RAG CLI/API wrappers without overriding shell environment variables.
- Added `rag_data/chroma/` to `.gitignore`.

## Dependency Installation

Installed local RAG dependencies with:

```powershell
python -m pip install -r requirements-rag.txt
```

Confirmed installed versions:

```text
chromadb 1.5.9
tiktoken 0.13.0
google-genai 2.9.0
```

## Pre-Index Check

Ran:

```powershell
python tools/manage_cornell_rag_db.py check
```

Observed result:

```text
[1/4] JSONL check complete: 732 chunks
[2/4] Input SHA-256: 21e3f445a63ccbf9d6c82b798a7aae2e0cd9cac4e54554cbb7d3cec77ad80ae6
[3/4] DB path is writable: rag_data\chroma
error: GEMINI_API_KEY is not configured
```

The local corpus and Chroma path are ready. At this point, vector embedding/index creation was blocked because `GEMINI_API_KEY` was not set in `.env` or in the shell environment.

## Tests Added Or Updated

Added:

- `ai/tests/test_cornell_rag_runtime.py`

Coverage includes:

- vendored `petcare_rag` imports from project pythonpath
- corpus exists at `rag_data/chunks/cornell_pet_health_chunks.jsonl`
- corpus validates to the expected 732 chunks
- retrieval gold file exists at `rag_data/evaluation/cornell_retrieval_gold.jsonl`

## Verification

Focused command:

```powershell
python -m pytest ai\tests\test_cornell_rag_runtime.py ai\tests\test_cornell_rag_adapter.py ai\tests\test_rag_agent_node.py ai\tests\test_assessment_graph_integration.py -q
```

Result:

```text
17 passed, 1 warning
```

Full-suite command:

```powershell
python -m pytest ai\tests --tb=short
```

Result:

```text
142 passed, 1 warning in 0.92s
```

The warning is the existing LangGraph/LangChain pending deprecation warning.

## Continuation Attempt

After `GEMINI_API_KEY` was added to `.env`, `python tools/manage_cornell_rag_db.py check` loaded the key and reached the Gemini embedding API. The network-enabled run failed with:

```text
429 RESOURCE_EXHAUSTED: prepayment credits are depleted
```

`rag_data/chroma/` remains empty, so no persistent Chroma collection has been generated yet.

## Current Blocker

`index`, `inspect`, and `evaluate` cannot complete until the configured Google AI project has active Gemini API billing/credits. Once billing/credits are available, run:

```powershell
$env:GEMINI_API_KEY="..."
python tools/manage_cornell_rag_db.py check
python tools/manage_cornell_rag_db.py index
python tools/manage_cornell_rag_db.py inspect
python tools/manage_cornell_rag_db.py evaluate
```
## Retry Attempt After Key Confirmation

Ran another network-enabled pre-index check after confirming `.env` contains exactly one `GEMINI_API_KEY` entry and no process-level `GEMINI_API_KEY` override is present.

Result:

```text
[1/4] JSONL check complete: 732 chunks
[2/4] Input SHA-256: 21e3f445a63ccbf9d6c82b798a7aae2e0cd9cac4e54554cbb7d3cec77ad80ae6
[3/4] DB path is writable: rag_data\chroma
429 RESOURCE_EXHAUSTED: prepayment credits are depleted
```

The retry did not create a Chroma collection. `rag_data/chroma/` remains empty.
