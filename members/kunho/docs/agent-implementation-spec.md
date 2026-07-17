# PetCare-AI Agent Implementation Spec

This spec fixes the implementation contract for the PetCare-AI agent runtime. The active AI backend lives under `members/kunho/ai/`.

## Fixed Decisions

| Area | Decision |
| --- | --- |
| Graph runtime | LangGraph Assessment Graph |
| Python package root | `members/kunho/ai/src` |
| Tests | `members/kunho/ai/tests` |
| Final safety routing | Rule-based Safety Guard |
| RAG provider | Local Cornell runtime through `CornellRAGAdapter` |
| DB/API surface | Existing documented backend APIs only |
| Hospital search/email sending | Out of MVP scope; draft only |
| Social chat | First-class `social_chat` route |

## LangGraph Node Plan

Implemented node sequence by route:

```text
social_chat:
  db_context_loader
  -> intent_classifier
  -> chat_agent

general_chat:
  db_context_loader
  -> intent_classifier
  -> evidence_planner
  -> rag_agent
  -> answer_composer
  -> answer_guard

symptom_check / followup_answer / handoff_request:
  db_context_loader
  -> intent_classifier
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
```

Rules:

- The classifier owns intent routing.
- LLM outputs may populate state, but final risk routing is rule-based.
- Evidence retrieval happens before answer composition.
- Social turns do not use RAG or medical answer guard.

## Graph Request Contract

Basic request shape:

```json
{
  "request_id": "req_demo_0001",
  "conversation_id": "conv_demo",
  "pet_id": 1,
  "user_input": "My cat is coughing today.",
  "conversation_history": [
    {"role": "user", "content": "My name is Kunho."},
    {"role": "assistant", "content": "Nice to meet you, Kunho."}
  ],
  "locale": "ko-KR",
  "timezone": "Asia/Seoul",
  "timestamp": "2026-07-17T09:00:00+09:00",
  "user_location": null
}
```

Required fields are `request_id`, `conversation_id`, `pet_id`, `user_input`, `locale`, `timezone`, and `timestamp`. `conversation_history` and `user_location` are optional in the Pydantic model and JSON Schema. Stateful harness sessions may maintain history internally; stateless API callers can provide recent turns explicitly.

## Graph Response Contract

Basic response shape:

```json
{
  "response_id": "res_demo_0001",
  "conversation_id": "conv_demo",
  "route": "answer_guard",
  "risk_level": "unknown",
  "assistant_message": "...",
  "needs_user_response": false,
  "follow_up_question": null,
  "handoff": {
    "type": "none",
    "summary": null,
    "summary_json": null,
    "email_draft": null
  },
  "emergency": {
    "is_emergency": false,
    "triggered_rules": []
  }
}
```

`risk_level` remains top-level for UI/routing compatibility. It must not be embedded inside `handoff.summary_json`. `route` must be one of the graph node routes defined in `contracts/jsonschema/agent-graph-response.schema.json`.

## Checklist Templates

MVP templates:

- `cat_cough_triage`
- `dog_cough_triage`
- `vomiting_triage`
- `diarrhea_triage`
- `breathing_triage`
- `seizure_triage`
- `toxicity_triage`
- `urinary_triage`

Each template must define required red-flag items, priority, and natural follow-up question text.

## Rule-Based Validator

Final rule priority:

```text
emergency > urgent > unknown > non_emergency
```

Emergency examples:

- open-mouth breathing in cats
- abnormal gum color
- collapse or fainting
- seizure
- severe bleeding
- suspected toxin exposure with concerning signs

Urgent examples:

- persistent worsening symptoms
- notable baseline deterioration
- repeated vomiting/diarrhea with risk factors
- breathing concern that does not meet emergency criteria

Unknown handling:

- Missing required information with available question turns returns `needs_more_info`.
- Missing required information after the question limit returns `unknown_after_max_questions`.
- Lack of information must not be treated as `non_emergency`.

Non-emergency is selected only when no emergency, urgent, or unknown rule applies.

## Follow-Up Policy

- Ask at most two safety follow-up questions.
- Do not ask about items already answered or negated.
- Prefer higher-priority missing red-flag items.
- After the cap is reached, move to unknown fallback instead of looping.

## LLM Structured Output Schemas

### Turn Understanding

The classifier may return a combined structured output with:

- intent
- requires safety screening flag
- extracted pet state
- red-flag mention flag
- optional social response

### State Extraction

State extraction includes:

```json
{
  "species": "cat",
  "symptoms": ["coughing"],
  "duration": null,
  "course_pattern": "unknown",
  "current_status": {
    "appetite": "unknown",
    "activity": "unknown",
    "water": "unknown"
  },
  "negated_findings": [],
  "uncertain_findings": []
}
```

Allowed `course_pattern` values:

- `new`
- `worsening`
- `improving`
- `persistent`
- `recurrent`
- `unknown`

### Checklist Extraction

Checklist extraction maps user text to checklist item values. It does not decide final risk.

### Social Chat Output

Social chat output is a lightweight assistant message:

```json
{
  "assistant_message": "You told me your name is Kunho."
}
```

The chat node should answer from known context and say it does not know when the information is absent.

### Answer Guard Review

Answer guard may approve, revise, or block unsafe drafts. It should reduce overconfidence, remove diagnosis certainty, and keep emergency language direct.

## RAG Adapter Contract

Provider interface:

```python
retrieve(query: str, filters: dict, top_k: int = 5) -> list[RetrievedChunk]
```

The adapter returns graph `RetrievedChunk` objects. `rag_agent` is responsible for converting those chunks into `RAGCitation` entries and for filling retrieval state:

- `retrieval.query`
- `retrieval.chunks`
- `retrieval.citations`
- `retrieval.provider`
- `retrieval.insufficient_evidence`
- `retrieval.errors`

The graph must continue safely if retrieval fails. The current `CornellRAGAdapter` returns an empty list on provider errors or invalid species/query input.
## Cornell Runtime Contract

The Cornell runtime is vendored as `petcare_rag` under `members/kunho/ai/src`.

| Contract | Value |
| --- | --- |
| Embedding model | `text-embedding-3-small` |
| Embedding dimensions | `1536` |
| Chroma collection | `cornell_pet_health_text_embedding_3_small_1536` |
| Expected corpus chunk count | `732` |
| Default corpus path | `rag_data/chunks/cornell_pet_health_chunks.jsonl` |
| Default retrieval gold path | `rag_data/evaluation/cornell_retrieval_gold.jsonl` |
| Standalone generation model default | `gpt-5.4-mini` |

Operational commands:

```powershell
python tools/manage_cornell_rag_db.py check
python tools/manage_cornell_rag_db.py index
python tools/manage_cornell_rag_db.py inspect
python tools/manage_cornell_rag_db.py evaluate
```

Generated `rag_data/chroma/` files are local runtime state and must not be committed.

## DB Context

Allowed backend context call:

```text
GET /api/pets/{pet_id}/handoff-context?days=3
```

Implementation rules:

- No new DB tables.
- No new application DB endpoints for graph execution. The optional `petcare_rag.api` HTTP service is backend-only diagnostics and is not required by the Assessment Graph.
- No undocumented request/response fields.
- Use fixtures to test DB Context Loader behavior.
- Return safe fallback responses when backend context is unavailable.

## Internal Triage Assessment

`internal_triage_assessment` owns graph-internal risk and follow-up policy state:

```json
{
  "schema_version": "1.0",
  "risk_level": "emergency | urgent | non_emergency | unknown",
  "red_flag_inputs": {},
  "clinical_inputs": {},
  "needs_followup": true,
  "followup_questions": []
}
```

Rules:

- Safety Guard sets final `risk_level`.
- LLMs may fill source values, but do not own risk routing.
- `followup_questions` is capped at two.

## Hospital Handoff Summary

Hospital handoff JSON must use exactly six content sections:

```json
{
  "schema_version": "1.1",
  "generated_at": "2026-07-16T09:00:00+09:00",
  "patient": {},
  "visit_reason": {},
  "clinical_course": {},
  "baseline_comparison": {},
  "triage_assessment": {},
  "medical_background": {}
}
```

Forbidden anywhere inside `handoff.summary_json`:

- `risk_level`
- `confidence`
- `missing_items`
- `triggered_rules`
- `decision_basis`
- `sources`
- `attachments`

`baseline_comparison.window_days` is fixed at `3`.

## Handoff Routing

Handoff generation runs only when:

```text
risk_level in ["urgent", "non_emergency", "unknown"]
AND hospital_visit_intent == "yes"
```

Emergency cases stay on the Emergency Agent path.

## Social Chat Contract

`social_chat` covers lightweight continuity turns such as:

```text
User: Hi.
User: What is my name?
User: What is my pet's name?
```

These turns should use `conversation_history` and `pet_context` if available. They should not fabricate remembered facts, run Cornell retrieval, attach citations, or add medical-safety boilerplate.

## Verification Contract

Primary local verification:

```powershell
python -m pytest
```

The root `pyproject.toml` sets:

```toml
[tool.setuptools.packages.find]
where = ["members/kunho/ai/src"]

[tool.pytest.ini_options]
pythonpath = ["members/kunho/ai/src"]
testpaths = ["members/kunho/ai/tests", "tests"]
```