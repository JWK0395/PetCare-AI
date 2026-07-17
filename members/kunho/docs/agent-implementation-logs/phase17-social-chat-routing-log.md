# Agent Phase 17 Social Chat Routing Log

Date: 2026-07-17
Repository: PetCare-AI
Scope: Let the LLM intent classifier separate conversational/meta turns from pet-care questions, then keep those turns out of Cornell RAG and the medical answer-guard path.

## Reason

A minimal greeting, and later a user name introduction, were classified as `general_chat`, then routed through:

```text
intent_classifier -> evidence_planner -> rag_agent -> answer_composer -> answer_guard
```

That made the assistant respond with generic veterinary disclaimers and Cornell source text even though the user had not asked a pet-care question. The fix is to make the intent contract more expressive: the LLM can now return `social_chat` for standalone conversational turns, while actual pet-care questions continue to use `general_chat` and Cornell evidence retrieval.

## Intent Contract Change

Updated `ai/src/petcare_agent/schemas/common.py` and `contracts/jsonschema/llm-structured-outputs.schema.json`:

- added `social_chat` to the `Intent` enum
- kept existing `general_chat` semantics for general pet-care questions

Updated `ai/src/petcare_agent/prompts/intent_classification.md`:

- `social_chat`: greetings, thanks, user name introductions, user-name recall questions, pet name/profile recall questions, assistant capability questions, and other conversational/meta turns without pet health guidance
- `general_chat`: general pet-care questions without current symptoms or handoff requests
- stored/profile information questions such as the user name or pet name should classify as `social_chat`, even when words like dog, cat, puppy, or pet appear
- mixed messages such as greeting plus pet health question should classify as the pet health intent, not `social_chat`
- `social_chat` does not require DB context or safety screening unless symptoms/red flags are also present

## Code Changes

Updated `ai/src/petcare_agent/graphs/assessment_graph.py`:

- routes `intent == "social_chat"` directly to `chat_agent`
- passes `deps.llm_client` into `chat_agent` so the node is LLM-backed
- keeps the Cornell RAG path for `general_chat` pet-care questions such as "How often can cats eat treats?"
- does not route `general_chat` through heuristic social-chat fallbacks; the LLM intent classifier owns that decision

Updated `ai/src/petcare_agent/nodes/chat_agent.py`:

- added LLM-backed `generate_social_chat_response(...)` using `SocialChatOutput` structured output
- sends current input, recent `conversation_history`, locale, and `pet_context` to the LLM
- uses only a generic deterministic fallback when the LLM call fails, avoiding profile-specific branch logic in fallback code
- clears stale `RetrievalState` before returning the social chat response
- kept `generate_chat_response(...)` as the backward-compatible evidence-planning alias for existing imports
- changed `chat_agent(...)` to be the LangGraph node for lightweight conversational chat

Updated request/history contracts:

- added optional `conversation_history` to `GraphRequest` and `contracts/jsonschema/agent-graph-request.schema.json`
- `build_initial_state(...)` now copies request history into `PetCareGraphState.conversation_history`
- stateful harness sessions already append each user/assistant turn and now have regression coverage for social-chat memory

Updated `ai/src/petcare_agent/schemas/llm_outputs.py` and `contracts/jsonschema/llm-structured-outputs.schema.json`:

- added `SocialChatOutput` with `assistant_message`

## Tests Updated

Updated `ai/tests/test_intent_classifier_node.py` with coverage for:

- LLM structured output applying `social_chat`
- `SocialChatOutput` contract alignment

Updated `ai/tests/test_chat_agent_node.py` with coverage for:

- LLM-backed social/profile/meta routing without production heuristic fallbacks
- not treating mixed greeting-plus-health-question text as standalone chat
- LLM-backed social chat receiving conversation history
- LLM-backed profile recall using conversation history and pet context
- social chat response clearing retrieval state and avoiding Cornell source text

Updated `ai/tests/test_assessment_graph_integration.py` with coverage for:

- LLM-returned `social_chat` routes as `intent_classifier -> chat_agent`
- DB context provider is not called
- RAG adapter is not called
- answer guard is not called
- public response route is `chat`
- response does not include Cornell source text
- `GraphRequest.conversation_history` is copied into initial graph state
- pet-name/profile questions route to `chat_agent` when classified as `social_chat`, without DB context, RAG, or answer guard
- stateful harness preserves social chat context across three turns: greeting, name introduction, name recall

Updated `ai/tests/test_contracts.py` with coverage for:

- `social_chat` is present in the structured-output JSON schema enum

## Manual Verification

Ran direct graph and harness checks with Korean user-name input and a fake LLM returning `social_chat`. The fake LLM raises if `answer_guard` is called for social chat.

Result:

```text
route: chat
trace: ['intent_classifier', 'chat_agent']
assistant_message: Korean response written by `chat_agent` from conversation history or pet context
```

## Full Verification

Ran:

```powershell
python -m pytest
```

Result:

```text
155 passed, 1 warning in 1.83s
```

The remaining warning is the existing LangGraph pending deprecation warning from `langgraph.cache.base`.
## Addendum: Turn Understanding Call Consolidation

Date: 2026-07-17

After measuring per-turn LLM calls, Phase 17 was extended to combine the two highest-impact call pairs:

- `intent_classifier + chat_agent` for `social_chat`
- `intent_classifier + state_updater` for symptom/safety paths

Implemented `TurnUnderstandingOutput` in `ai/src/petcare_agent/schemas/llm_outputs.py` and `contracts/jsonschema/llm-structured-outputs.schema.json`. The single structured output now carries:

- route fields formerly returned by `IntentClassificationOutput`
- pet-state fields formerly returned by `StateExtractionOutput`
- optional `SocialChatOutput` for social/profile turns

Updated `ai/src/petcare_agent/prompts/turn_understanding.md` so one LLM call classifies the turn, extracts current pet state, and drafts a social response only when `intent == "social_chat"`.

Updated graph wiring in `ai/src/petcare_agent/graphs/assessment_graph.py`:

```text
db_context_loader -> intent_classifier
```

This lets the turn-understanding call see `pet_context` before answering profile questions such as pet-name recall. The `chat_agent` now reuses `state.chat_response` when `social_response_ready` is true, and `state_updater` skips its LLM call when `turn_state_extracted` is true.

Resulting completion-call counts per user turn:

| Path | Before | After |
| --- | ---: | ---: |
| `social_chat` | 2 | 1 |
| symptom/emergency/follow-up question paths | 3 | 2 |
| final safety answer path | 4 | 3 |

Updated regression coverage in:

- `ai/tests/test_intent_classifier_node.py`
- `ai/tests/test_assessment_graph_integration.py`
- `ai/tests/test_harness_current_adapter.py`
- `ai/tests/test_runtime_adapter.py`
- `ai/tests/test_observability_metadata.py`
- `ai/tests/test_contracts.py`

Verification:

```powershell
python -m pytest
```

Result:

```text
160 passed, 1 warning in 3.95s
```

The remaining warning is the existing LangGraph pending deprecation warning from `langgraph.cache.base`.
