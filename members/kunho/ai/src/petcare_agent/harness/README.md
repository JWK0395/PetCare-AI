# PetCare Agent Harness

The harness runs a PetCare agent locally against a data bundle, without the
backend, frontend, or database. It is useful for smoke testing an agent,
replaying conversations, and comparing teammate implementations through one
shared CLI.

## Setup

From the repository root, install the package in editable mode:

```powershell
python -m pip install -e .
```

If Python cannot import `petcare_agent`, rerun the same command. This refreshes
the editable package path after the source tree moves.

## Run The Current Agent

Use the bundled demo data:

```powershell
python -m petcare_agent.harness `
  --data-zip examples\data_bundles\petcare_db_v1_demo.zip `
  --pet-id 2
```

List pets in a bundle:

```powershell
python -m petcare_agent.harness `
  --data-zip examples\data_bundles\petcare_db_v1_demo.zip `
  --pet-id 2 `
  --list-pets
```

Run one turn and exit:

```powershell
python -m petcare_agent.harness `
  --data-zip examples\data_bundles\petcare_db_v1_demo.zip `
  --pet-id 2 `
  --once "My dog vomited twice today."
```

## Run A Teammate Agent

Pass a custom adapter with `--agent module:attribute`:

```powershell
python -m petcare_agent.harness `
  --data-zip examples\data_bundles\petcare_db_v1_demo.zip `
  --pet-id 2 `
  --agent teammate_agent.harness_adapter:TeammateAgentAdapter
```

The module must be importable by Python. If the teammate source lives outside
the installed package, add it to `PYTHONPATH` first:

```powershell
$env:PYTHONPATH = "$PWD\members\teammate\ai\src;$env:PYTHONPATH"
```

## Adapter Contract

The adapter contract is defined in `adapter.py`.

An adapter implements `start_session(...)` and returns a stateful session:

```python
class TeammateAgentAdapter:
    name = "teammate-agent"

    def start_session(
        self,
        *,
        config,
        context_provider,
        rag_adapter=None,
        llm_client=None,
    ):
        return TeammateAgentSession(config, context_provider, rag_adapter)
```

The session implements `handle_user_message(user_input) -> AgentTurnResult`:

```python
from petcare_agent.harness.adapter import AgentTurnResult
from petcare_agent.schemas.graph_state import GraphResponse, PetCareGraphState


class TeammateAgentSession:
    def __init__(self, config, context_provider, rag_adapter=None):
        self.config = config
        self.context_provider = context_provider
        self.rag_adapter = rag_adapter
        self.turn_index = 0
        self.state = PetCareGraphState(
            pet_id=config.pet_id,
            conversation_id=config.conversation_id,
            locale=config.locale,
            timezone=config.timezone,
        )

    def handle_user_message(self, user_input: str) -> AgentTurnResult:
        self.turn_index += 1
        context = self.context_provider.load_context(
            self.config.pet_id,
            days=self.config.db_context_days,
        )

        # Call the teammate agent here and map its result to GraphResponse.
        assistant_message = "Mapped teammate agent response."

        response = GraphResponse(
            response_id=f"resp_{self.config.conversation_id}_{self.turn_index:04d}",
            conversation_id=self.config.conversation_id,
            route="chat",
            risk_level="unknown",
            assistant_message=assistant_message,
            needs_user_response=False,
        )

        return AgentTurnResult(response=response, state=self.state)
```

## Response Contract

The public response payload is `GraphResponse`.

- Python model: `petcare_agent.schemas.graph_state.GraphResponse`
- JSON Schema: `contracts/jsonschema/agent-graph-response.schema.json`

Required response fields include:

- `response_id`
- `conversation_id`
- `route`
- `risk_level`
- `assistant_message`
- `needs_user_response`
- `follow_up_question`
- `handoff`
- `emergency`

When constructing `GraphResponse` in Python, optional fields use model
defaults. When returning raw JSON across an API boundary, validate against the
JSON Schema.

## Data And RAG Hooks

The harness injects local test doubles into each adapter:

- `context_provider.load_context(pet_id, days=...)` returns fake backend context
  from the selected data bundle.
- `rag_adapter` retrieves fixture chunks when the bundle includes `rag/chunks.json`.
- `config` carries `pet_id`, `conversation_id`, locale, timezone, context days,
  and RAG top-k settings.

## Built-In Agent Names

These names use the current Assessment Graph adapter:

- `current-assessment-graph`
- `current`
- `langgraph-v1`

Example:

```powershell
python -m petcare_agent.harness `
  --data-zip examples\data_bundles\petcare_db_v1_demo.zip `
  --pet-id 2 `
  --agent current
```
