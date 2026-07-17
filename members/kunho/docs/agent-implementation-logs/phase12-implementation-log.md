# Agent Phase 12 Implementation Log

Date: 2026-07-16
Repository: PetCare-AI
Scope: Triage/Handoff contract separation and golden scenario readiness

## Reference Inputs

This phase was driven by the uploaded triage/handoff design notes and golden scenario eval set. The implementation translated the design into repository contracts and runtime behavior with these main principles:

- Hospital handoff JSON is veterinarian-facing and fixed to six sections.
- Internal risk routing belongs to `internal_triage_assessment`, not the hospital handoff payload.
- `risk_level`, `confidence`, `missing_items`, `triggered_rules`, `decision_basis`, `sources`, and `attachments` must not appear in the hospital handoff JSON.
- Safety Guard still owns final risk routing through the rule-based validator.
- LLM output remains useful for extraction/drafting, but it does not own the final risk decision.
- Follow-up questions are capped at two.
- Baseline comparison uses a three-day window.

## Implemented Scope

### JSON Schema Contracts

Added two contract files:

- `contracts/jsonschema/hospital-handoff-summary.schema.json`
- `contracts/jsonschema/internal-triage-assessment.schema.json`

`hospital-handoff-summary.schema.json` defines the six veterinarian-facing sections: `patient`, `visit_reason`, `clinical_course`, `baseline_comparison`, `triage_assessment`, and `medical_background`.

`internal-triage-assessment.schema.json` defines the internal routing payload: `risk_level`, `red_flag_inputs`, `clinical_inputs`, `needs_followup`, and `followup_questions`.

Updated `ai/src/petcare_agent/contracts/schema_loader.py` so the two new contracts are loaded alongside the existing graph, checklist, and LLM structured-output contracts.

### Pydantic Models

Added `ai/src/petcare_agent/schemas/handoff.py` with models for the six-section handoff JSON and internal triage payload:

- `HospitalHandoffSummary`
- `PatientSummary`
- `VisitReasonSummary`
- `ClinicalCourseSummary`
- `BaselineComparisonSummary`
- `BaselineChange`
- `TriageAssessmentSummary`
- `AssociatedSymptomSummary`
- `RedFlagSummary`
- `MedicalBackgroundSummary`
- `InternalTriageAssessment`
- `RedFlagInputs`
- `ClinicalInputs`

Updated `ai/src/petcare_agent/schemas/common.py` with `CoursePattern` so state extraction, internal triage, and handoff course fields use the same value set: `new | worsening | improving | persistent | recurrent | unknown`.

Updated `ai/src/petcare_agent/schemas/graph_state.py`:

- `HandoffResponse` now includes `summary_json`.
- `HandoffState` now includes `summary_json`.
- `PetCareGraphState` now includes `internal_triage_assessment`.
- `AssessmentState` now includes `course_pattern`.
- `PetCareContext` now includes `medical_background` so known empty conditions/medications/allergies can be preserved as empty arrays.

### Red Flag Canonicalization

Added `ai/src/petcare_agent/safety/red_flags.py`.

This module normalizes checklist item ids into the canonical red flag names used by internal triage and hospital handoff JSON. Examples:

- `open_mouth_breathing` -> `open_mouth_breathing`
- `labored_breathing` -> `labored_breathing`
- `gum_color_abnormal` -> `gum_color_abnormal`
- `collapse_or_fainting` -> `collapse_or_fainting`
- `active_seizure`, `seizure_over_5_min`, `repeated_seizures` -> `seizure`
- `known_toxin_ingestion`, `unknown_substance_ingestion`, `suspected_toxin` -> `toxin_exposure_suspected`

The module also centralizes `FORBIDDEN_HANDOFF_FIELDS` so tests can assert that internal routing fields do not leak into handoff JSON.

### Safety Guard Integration

Updated `ai/src/petcare_agent/nodes/safety_guard.py` so every validation result now populates `state.internal_triage_assessment`.

The internal assessment is built from:

- the rule validator result for `risk_level`
- canonical checklist values for `red_flag_inputs`
- current graph facts for `clinical_inputs`
- validator follow-up status for `needs_followup`
- selected/pending question text where available

The existing top-level `state.risk_level` and `state.confidence` remain for graph routing and public response compatibility. The important boundary is that these fields are not copied into the hospital handoff JSON.

### Question Manager Integration

Updated `ai/src/petcare_agent/nodes/question_manager.py` so selected follow-up question text is mirrored into `internal_triage_assessment.followup_questions`.

Behavior preserved:

- at most two questions per turn
- at most two safety-question turns
- answered or already pending questions are not repeated

### State Extraction Course Pattern

Updated:

- `ai/src/petcare_agent/schemas/llm_outputs.py`
- `ai/src/petcare_agent/nodes/state_updater.py`
- `ai/src/petcare_agent/prompts/state_extraction.md`
- `contracts/jsonschema/llm-structured-outputs.schema.json`

`StateExtractionOutput` now carries `course_pattern`, which is persisted into `AssessmentState` and then reused by internal triage and handoff clinical course generation.

### Handoff Subgraph Rewrite

Rewrote `ai/src/petcare_agent/graphs/subgraphs/handoff.py` so handoff generation is now six-section JSON first.

The non-emergency handoff path still only runs when:

```text
risk_level in ["urgent", "non_emergency", "unknown"]
AND hospital_visit_intent == "yes"
```

But instead of building a risk-bearing text summary, it now builds `HospitalHandoffSummary(patient=..., visit_reason=..., clinical_course=..., baseline_comparison=..., triage_assessment=..., medical_background=...)`.

The legacy `handoff.summary` string and `handoff.email_draft` are still generated for display and draft-email UX, but they are derived from the veterinarian-facing summary and do not include internal rule ids or risk labels.

### Response Contract Update

Updated `contracts/jsonschema/agent-graph-response.schema.json` so `handoff.summary_json` is part of the graph response handoff object.

Updated `ai/src/petcare_agent/graphs/response_composer.py` so public responses include `state.handoff.summary_json` when a handoff was generated.

Compatibility note:

- `GraphResponse.risk_level` remains top-level because routing/frontend logic may still need it.
- The new contract only forbids internal risk fields inside the hospital handoff JSON payload.

### LLM Handoff Output Cleanup

Updated `ai/src/petcare_agent/schemas/llm_outputs.py` and `contracts/jsonschema/llm-structured-outputs.schema.json` so `HandoffSummaryOutput` no longer includes `risk_level`, `triggered_rules`, or `metadata`.

Updated `ai/src/petcare_agent/prompts/handoff_summary.md` to say internal routing fields must not be included in the handoff summary.

### Golden Eval Fixture

Added `ai/tests/fixtures/triage_handoff_golden.jsonl` with 12 scenario rows corresponding to the uploaded golden set:

- G01 open-mouth/labored breathing emergency
- G02 collapse + abnormal gum color emergency
- G03 toxin exposure emergency
- G04 appetite/activity urgent
- G05 repeated vomiting urgent
- G06 vague input unknown
- G07 vague breathing concern unknown
- G08 mild change non-emergency
- G09 handoff request after safety
- G10 empty medical background
- G11 sparse red flag emergency
- G12 many fields without red flags non-emergency

Added `ai/tests/test_triage_handoff_golden_fixture.py` to validate fixture shape, risk labels, follow-up caps, and expected handoff assertion metadata.

The fixture is now ready for a future executable eval runner. Current tests validate the fixture contract and core runtime behaviors with mocked LLM/API boundaries.

## Tests Added Or Updated

Added:

- `ai/tests/test_hospital_handoff_contract.py`
- `ai/tests/test_triage_handoff_golden_fixture.py`
- `ai/tests/fixtures/triage_handoff_golden.jsonl`

Updated:

- `ai/tests/test_contracts.py`
- `ai/tests/test_handoff_subgraph.py`
- `ai/tests/test_response_composer.py`
- `ai/tests/test_handoff_summary_builder_node.py`
- `ai/tests/test_assessment_graph_integration.py`

New coverage includes:

- hospital handoff schema has exactly the six top-level sections
- hospital handoff schema excludes internal risk/routing fields
- internal triage schema owns `risk_level`
- internal follow-up questions are capped at two
- Pydantic model fields match the new schema properties
- Safety Guard populates `internal_triage_assessment`
- Question Manager writes selected follow-up question text into internal triage
- non-emergency handoff generation emits `summary_json`
- `baseline_comparison.window_days` is fixed at `3`
- empty medical background remains empty arrays
- handoff JSON does not contain forbidden internal fields recursively
- integration handoff route returns `handoff.summary_json`

## Verification

Full-suite command:

```powershell
python -m pytest ai/tests
```

Result:

```text
137 passed, 1 warning in 0.92s
```

Warning observed:

```text
LangChainPendingDeprecationWarning: The default value of allowed_objects will change in a future version.
```

The warning comes from installed LangGraph/LangChain dependencies and is unrelated to this contract change.

## Design Decisions

### Kept Top-Level Graph Risk Level

The design says hospital handoff JSON should not include `risk_level`. This phase follows that strictly for `HospitalHandoffSummary`.

However, `GraphResponse.risk_level` and `PetCareGraphState.risk_level` remain because they are routing and UI-facing graph outputs, not the veterinarian-facing hospital handoff JSON.

### Kept Legacy Text Summary

`HandoffState.summary` and `email_draft` remain to avoid breaking existing response surfaces. The new source of truth for hospital handoff content is `HandoffState.summary_json`.

### Deterministic Handoff Builder First

The implemented handoff subgraph is deterministic and state-based. This keeps the JSON contract stable for tests and evals. LLM handoff drafting still exists, but the six-section JSON does not depend on the LLM deciding risk or adding internal metadata.

### Red Flag Alias Mapping

The checklist templates use practical item ids, while the handoff/internal triage contract uses canonical red flag ids. A shared normalization module avoids repeating this mapping across Safety Guard and handoff generation.

## Known Limitations And Next Work

- The golden JSONL fixture is shape-validated, but there is not yet a full eval runner that executes every scenario through mocked LLM outputs.
- Natural-language extraction quality still depends on `StateExtractionOutput` and `ChecklistExtractionOutput` behavior. This phase hardens contracts and deterministic downstream behavior, not production NLU accuracy.
- `associated_symptoms.count` is deterministic and currently defaults to `1` per extracted symptom unless richer count extraction is added later.
- Some emergency concepts in existing checklist rules, such as urinary obstruction, are not part of the uploaded seven-field red flag schema and therefore are not represented in `RedFlagInputs` unless the contract is expanded later.
- Future work should add an executable eval harness that converts each JSONL row into mocked structured outputs, runs the graph, and checks route/risk/handoff assertions end to end.
