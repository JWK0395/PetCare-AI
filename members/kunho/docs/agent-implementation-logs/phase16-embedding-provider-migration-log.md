# Agent Phase 16 Embedding Provider Migration Log

Date: 2026-07-17
Repository: PetCare-AI
Scope: Switch Cornell RAG embeddings from Gemini to OpenAI `text-embedding-3-small`, build the local Chroma collection, and harden Korean-to-English retrieval.

## Reason

The requested embedding model is OpenAI `text-embedding-3-small`. This supersedes the Phase 14/15 Gemini embedding setup and removes the previous Gemini billing blocker from the current implementation path.

## Code Changes

Updated `ai/src/petcare_rag/manage_cornell_rag_db.py`:

- `MODEL = "text-embedding-3-small"`
- `DIMENSION = 1536`
- `DEFAULT_COLLECTION = "cornell_pet_health_text_embedding_3_small_1536"`
- replaced Gemini client construction with OpenAI `OpenAI(api_key=OPENAI_API_KEY)`
- replaced `models.embed_content(...)` with `client.embeddings.create(...)`
- kept explicit embedding validation and Chroma metadata compatibility checks
- updated CLI output/error messages to reference OpenAI embeddings
- forced CLI stdout/stderr to UTF-8 with replacement fallback for Korean queries and Cornell punctuation on Windows consoles
- added reusable query normalization for cross-lingual retrieval:
  - species context (`dog/canine`, `cat/feline`)
  - Korean/English veterinary domain lexicon for exposures, symptoms, disease families, and clinical terms
  - no gold case ids, chunk ids, or expected document ids encoded in the query normalization layer

Updated `ai/src/petcare_rag/pipeline.py`:

- query embedding now calls `rag_db.openai_client()`
- query vector docstring updated to 1536 dimensions
- retrieval passes species into `query_embedding_text(...)` so app and CLI paths share the same normalization behavior

Updated `ai/src/petcare_rag/api.py`:

- readiness now checks `openai_api_key_configured`

Updated configuration:

- removed `GEMINI_API_KEY` from `.env.example`
- updated `PETCARE_RAG_COLLECTION` to `cornell_pet_health_text_embedding_3_small_1536`
- removed `google-genai` from RAG dependencies
- added `openai` to `requirements-rag.txt`

## Tests Updated

Updated `ai/tests/test_cornell_rag_runtime.py` with coverage for:

- embedding model constant
- embedding dimension
- default collection name
- OpenAI embeddings API call shape
- reusable Korean query normalization terms
- species context in query embedding text
- guard against document-id/gold-id based query expansion

## Earlier Blocked Attempts

Before a valid OpenAI key was available, `python tools/manage_cornell_rag_db.py check` reached the OpenAI embeddings API and returned sanitized `401 invalid_api_key` errors. No key value or key prefix was recorded in the repository, docs, or logs.

## Successful OpenAI Check

Ran after the key was corrected:

```powershell
python tools/manage_cornell_rag_db.py check
```

Result:

```text
[1/4] JSONL check complete: 732 chunks
[2/4] Input SHA-256: 21e3f445a63ccbf9d6c82b798a7aae2e0cd9cac4e54554cbb7d3cec77ad80ae6
[3/4] DB path is writable: rag_data\chroma
[4/4] OpenAI embedding API connection successful: text-embedding-3-small, 1536 dimensions
Pre-index checks passed. You can now run the index command.
```

## Index Build

Ran:

```powershell
python tools/manage_cornell_rag_db.py index
```

Result:

```text
Input 732 / already complete 0 / chunks to process 732
Upsert complete: 732/732
Index complete: 732 chunks are in collection cornell_pet_health_text_embedding_3_small_1536.
```

Generated local state under `rag_data/chroma/` is intentionally gitignored.

## Inspect Result

Ran:

```powershell
python tools/manage_cornell_rag_db.py inspect
```

Result:

```text
Total chunks: 732
Unique documents: 282
Species chunks: dog=418, cat=314
Collection metadata: {'embedding_dimension': 1536, 'expected_chunks': 732, 'corpus_sha256': '21e3f445a63ccbf9d6c82b798a7aae2e0cd9cac4e54554cbb7d3cec77ad80ae6', 'embedding_model': 'text-embedding-3-small'}
Missing required metadata: 0
Vectors not 1536 dimensions: 0
```

## Retrieval Evaluation

Initial evaluation after index creation exposed a cross-lingual retrieval gap: English Cornell documents did not always rank correctly for Korean user questions. The fix was not a per-case expected-document patch. The retrieval path now uses a general domain normalization layer that adds reusable English veterinary terms and species context before creating the query embedding.

Final run:

```powershell
python tools/manage_cornell_rag_db.py evaluate
```

Result:

```text
Evaluation result: 12/12 passed
```

## Full Verification

Ran:

```powershell
python -m pytest
```

Result:

```text
148 passed, 1 warning in 2.18s
```

## Docs Updated

Updated current docs to reflect the generated vector store and generalized query normalization:

- `docs/current-implementation-status.md`
- `docs/cornell-rag-vector-store.md`
- `docs/agent-architecture.md`
- `docs/agent-implementation-spec.md`
- `docs/agent-implementation-roadmap.md`
- `docs/agent-observability-policy.md`
- `docs/database-schema.md`