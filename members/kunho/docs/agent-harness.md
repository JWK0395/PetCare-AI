# Agent Harness

The harness lets teammates test interchangeable agent implementations without running the backend, frontend, or a database. The current implementation is packaged from `members/kunho/ai/src` and exposed through `python -m petcare_agent.harness` or the `petcare-agent-playground` console script after installation.

## Data Bundle

Provide either a `.zip` file or an unpacked directory. The loader accepts the canonical nested paths and simple root-level fallbacks for quick fixtures.

```text
data-bundle/
├─ manifest.json
├─ db/
│  ├─ pets.json
│  ├─ daily_entries.json
│  └─ diagnoses.json
├─ api/
│  └─ handoff_contexts.json       # optional direct endpoint-shaped contexts
└─ rag/
   └─ chunks.json                 # optional RetrievedChunk fixtures
```

The fake backend can build the same shape as `GET /api/pets/{pet_id}/handoff-context?days=3` from raw DB-style fixtures. If `api/handoff_contexts.json` is present, it is used as a direct endpoint payload and then sliced by `--days`.

Bundled examples:

- `examples/data_bundles/petcare_db_v1_demo` and `.zip`: three-pet demo with cat cough, dog vomiting, and cat urinary scenarios.
- `examples/data_bundles/cat_cough_minimal`: smaller fixture for targeted tests.

## Run

Install the package once from the repository root before using the module
entrypoint:

```powershell
python -m pip install -e .
```

If the source tree has moved since a previous editable install, rerun the same
command to refresh Python's package path.

Interactive demo bundle run:

```powershell
python -m petcare_agent.harness --data-zip .\examples\data_bundles\petcare_db_v1_demo --pet-id 1
```

Run the zipped demo bundle:

```powershell
python -m petcare_agent.harness --data-zip .\examples\data_bundles\petcare_db_v1_demo.zip --pet-id 1
```

One-turn smoke test:

```powershell
python -m petcare_agent.harness --data-zip .\examples\data_bundles\petcare_db_v1_demo --pet-id 1 --once "My cat is coughing today."
```

Useful CLI options:

| Option | Meaning |
| --- | --- |
| `--agent` | Built-in name or `module:attribute` adapter. Default: `current-assessment-graph`. |
| `--days` | Recent daily-entry window for fake handoff context. Default: `3`. |
| `--rag-top-k` | Max fixture RAG chunks retrieved per turn. Default: `5`. |
| `--locale` | Response locale. Default: `ko-KR`. |
| `--timezone` | Conversation timezone. Default: `Asia/Seoul`. |
| `--conversation-id` | Stable conversation id. Auto-generated when omitted. |
| `--transcript-dir` | JSONL transcript directory. Default: `.tmp/agent-harness`. |
| `--no-transcript` | Disable transcript writing. |
| `--once` | Run one message and exit. |
| `--replay` | Replay a text or JSONL transcript file. |
| `--list-pets` | Print pets from the bundle and exit. |

Console commands inside interactive mode:

```text
/help
/state
/handoff
/visit yes
/visit no
/visit undecided
/visit not_asked
/exit
```

The harness writes JSONL transcripts to `.tmp/agent-harness/` by default. Each event includes the user input, public `GraphResponse`, trace path, bounded state summary, and fallback reason.

## Swap Agents

The shared contract is fixed to the current Assessment Graph models:

- input session config: `AgentSessionConfig`
- turn output: `AgentTurnResult`
- public response: `GraphResponse`
- retained state: `PetCareGraphState`

Built-in adapter names:

- `current-assessment-graph`
- `current`
- `langgraph-v1`

Built-in current graph run:

```powershell
python -m petcare_agent.harness --data-zip .\examples\data_bundles\petcare_db_v1_demo --pet-id 1 --agent current-assessment-graph
```

Custom adapter:

```powershell
python -m petcare_agent.harness --data-zip .\examples\data_bundles\petcare_db_v1_demo --pet-id 1 --agent my_package.my_agent:Adapter
```

New agents only need to implement `AgentAdapter.start_session(...)` and `AgentSession.handle_user_message(...)` from `petcare_agent.harness.adapter`. The harness supplies a `DataBundleBackendProvider`, optional `DataBundleRAGAdapter`, and the selected `AgentSessionConfig`.