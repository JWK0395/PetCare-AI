# Cornell RAG Vector Store Setup

Date: 2026-07-17

## Current Repository State

The Cornell RAG runtime and corpus are vendored into this repository for graph-internal retrieval.

Runtime package:

```text
members/kunho/ai/src/petcare_rag/
  __init__.py
  api.py
  manage_cornell_rag_db.py
  models.py
  pipeline.py
```

Corpus, evaluation, and generated local index paths:

```text
rag_data/
  chunks/cornell_pet_health_chunks.jsonl
  evaluation/cornell_retrieval_gold.jsonl
  chroma/                  # generated local Chroma index, gitignored
```

Graph/RAG maintenance CLI wrappers load `.env` automatically without overriding existing shell environment variables:

```text
tools/_project_env.py
tools/manage_cornell_rag_db.py
tools/run_cornell_rag.py
tools/run_cornell_rag_api.py
```

## Provider And Model Contract

| Purpose | Provider/model | Required key | Notes |
| --- | --- | --- | --- |
| Corpus embeddings | OpenAI `text-embedding-3-small` | `OPENAI_API_KEY` | 1536-dimensional vectors stored in Chroma |
| Query embeddings | OpenAI `text-embedding-3-small` | `OPENAI_API_KEY` | Domain-normalized Korean/English query text plus species context before Chroma search |
| Graph answer composition | Assessment Graph nodes | project LLM settings as configured | `answer_composer` and `answer_guard` own graph responses |

Collection contract:

```text
name: cornell_pet_health_text_embedding_3_small_1536
embedding_model: text-embedding-3-small
embedding_dimension: 1536
expected_chunks: 732
default_path: rag_data/chroma
```

## Installed And Declared Dependencies

The project declares RAG dependencies in both `requirements-rag.txt` and the `rag` optional dependency group in `pyproject.toml`.

Current RAG dependency file:

```text
chromadb 1.5.9
openai >= 1.0.0
tiktoken 0.13.0
```

`google-genai` is no longer required by the current embedding path.

## Current Build Status

The code targets OpenAI `text-embedding-3-small`. The previous Gemini `429 RESOURCE_EXHAUSTED` blocker is superseded by this provider change.

Generated local Chroma state:

- Collection: `cornell_pet_health_text_embedding_3_small_1536`
- Total chunks: `732`
- Unique documents: `282`
- Species chunks: `dog=418`, `cat=314`
- Corpus SHA-256: `21e3f445a63ccbf9d6c82b798a7aae2e0cd9cac4e54554cbb7d3cec77ad80ae6`
- Collection metadata: `embedding_model=text-embedding-3-small`, `embedding_dimension=1536`, `expected_chunks=732`
- Metadata issues: `0`
- Wrong-dimension vectors: `0`
- Gold retrieval evaluation: `12/12 passed`

Query retrieval uses a reusable domain normalization layer for Korean user questions against the English Cornell corpus. The layer adds species context and broad veterinary English terms for exposures, symptoms, disease families, and clinical concepts; it intentionally does not encode gold case IDs, chunk IDs, or expected document IDs.

## Build Or Rebuild The Vector Store

Set `OPENAI_API_KEY` in `.env` or in the same PowerShell session:

```powershell
$env:OPENAI_API_KEY="..."
python tools/manage_cornell_rag_db.py check
python tools/manage_cornell_rag_db.py index --rebuild
python tools/manage_cornell_rag_db.py inspect
python tools/manage_cornell_rag_db.py evaluate
```

The `index` command creates the persistent Chroma collection under `rag_data/chroma/`. That directory is generated local state and is intentionally gitignored.

## Use The Vector Store

Run a local retrieval/answer after indexing:

```powershell
python tools/run_cornell_rag.py --species dog --question "My dog ate chocolate. What should I watch for?" --debug
```

The Assessment Graph/backend path uses `CornellRAGAdapter` directly in-process. No FastAPI server or `PETCARE_RAG_SERVICE_TOKEN` is required for graph-internal RAG.
## Optional Local RAG API

The Assessment Graph does not need an HTTP RAG service. For backend-only diagnostics or integration smoke tests, `petcare_rag.api` can be run with:

```powershell
python tools/run_cornell_rag_api.py --host 127.0.0.1 --port 8001
```

Endpoints:

| Endpoint | Purpose | Auth |
| --- | --- | --- |
| `GET /health` | Process liveness. | None |
| `GET /ready` | Checks OpenAI key, service token, Chroma path, collection compatibility, and expected chunk count. | None |
| `POST /v1/rag/answer` | Returns a citation-backed Cornell answer for `question`, `species`, and `top_k`. | `X-PetCare-Token` matching `PETCARE_RAG_SERVICE_TOKEN` |

The API accepts only official-source RAG inputs. Do not send pet profiles, daily logs, diagnoses, uploaded document text, owner notes, hospital names, or other personal records to this API.

## Safety Boundary

The Cornell vector store contains public Cornell official-source material only. Personal pet profiles, daily logs, diagnosis text, and owner free text must not be indexed into this store or sent to the OpenAI embedding API.
