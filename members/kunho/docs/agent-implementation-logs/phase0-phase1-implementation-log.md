# Agent Phase 0/1 Implementation Log

Date: 2026-07-16
Repository: PetCare-AI
Scope: `docs/agent-implementation-roadmap.md` 기준 Phase 0 + Phase 1

## 기준 문서

구현 전 아래 문서를 먼저 확인하고, 문서에 존재하는 계약만 기준으로 작업했다.

- `docs/agent-architecture.md`
- `docs/agent-implementation-spec.md`
- `docs/agent-implementation-roadmap.md`
- `docs/api-endpoints.md`
- `docs/database-schema.md`
- `contracts/jsonschema/*.json`

작업 시작 시점에는 `contracts/jsonschema/` 경로가 존재하지 않아, 문서에서 명시한 4개 JSON schema 계약 파일을 추가했다.

## 구현 범위

### Phase 0. Project Foundation

- `ai/src/petcare_agent` Python package skeleton 추가
- 환경 변수 계약 정리
  - `OPENAI_API_KEY`
  - `LANGSMITH_API_KEY`
  - `LANGSMITH_TRACING`
  - `PETCARE_API_BASE_URL`
- LangSmith tracing helper skeleton 추가
  - graph 내부 의사결정 node가 아니라 외부 관측 layer로만 사용하도록 구성
  - disabled/enabled 설정 모두 import 가능하도록 lazy import 적용
- JSON schema contract loader 추가
  - `contracts/jsonschema` 위치 탐색
  - 테스트와 런타임에서 schema 이름으로 로드 가능
- pytest 설정 정리
  - sandbox 환경에서 cache permission warning이 나지 않도록 cache provider 비활성화

### Phase 1. Domain Models And Contracts

- graph request/response/state pydantic model 추가
  - `GraphRequest`
  - `GraphResponse`
  - `PetCareGraphState`
- triage pydantic model 추가
  - `ChecklistItem`
  - `ChecklistTemplate`
  - `RiskResult`
  - `RuleHit`
- LLM structured output pydantic model 추가
  - `IntentClassificationOutput`
  - `StateExtractionOutput`
  - `ChecklistExtractionOutput`
  - `AnswerGuardReviewOutput`
  - `HandoffSummaryOutput`
- JSON schema 계약 파일 추가
  - `contracts/jsonschema/agent-graph-request.schema.json`
  - `contracts/jsonschema/agent-graph-response.schema.json`
  - `contracts/jsonschema/triage-checklist.schema.json`
  - `contracts/jsonschema/llm-structured-outputs.schema.json`
- 계약 테스트 추가
  - package import
  - JSON schema 4개 parsing
  - sample request/response pydantic validation
  - JSON schema와 pydantic model 필드명 정합성 확인

## 준수한 제약

- LangGraph `StateGraph` 기반 구현을 위한 schema/foundation만 준비했다.
- DB/API 변경은 하지 않았다.
- 신규 endpoint는 만들지 않았다.
- `assessment-context` 같은 alias endpoint를 추가하지 않았다.
- DB context는 추후 `GET /api/pets/{pet_id}/handoff-context?days=3`만 사용하도록 문서 계약에 맞춰 모델 경계만 준비했다.
- RAG 내부 구현은 하지 않았다.
- 병원 검색 또는 이메일 실제 발송 구현은 하지 않았다.
- 응급 판단 rule validator, graph node, LLM adapter 실제 구현은 Phase 2 이후 범위로 남겼다.

## 추가/수정 파일

- `.env.example`
- `.gitignore`
- `pyproject.toml`
- `ai/src/petcare_agent/__init__.py`
- `ai/src/petcare_agent/config.py`
- `ai/src/petcare_agent/tracing.py`
- `ai/src/petcare_agent/py.typed`
- `ai/src/petcare_agent/contracts/__init__.py`
- `ai/src/petcare_agent/contracts/schema_loader.py`
- `ai/src/petcare_agent/schemas/__init__.py`
- `ai/src/petcare_agent/schemas/common.py`
- `ai/src/petcare_agent/schemas/graph_state.py`
- `ai/src/petcare_agent/schemas/triage.py`
- `ai/src/petcare_agent/schemas/llm_outputs.py`
- `contracts/jsonschema/agent-graph-request.schema.json`
- `contracts/jsonschema/agent-graph-response.schema.json`
- `contracts/jsonschema/triage-checklist.schema.json`
- `contracts/jsonschema/llm-structured-outputs.schema.json`
- `ai/tests/test_imports.py`
- `ai/tests/test_contracts.py`

## pytest 이슈와 해결

초기 pytest 실패 원인은 JSON schema 파일이 Windows PowerShell 저장 과정에서 UTF-8 BOM 포함으로 저장된 것이었다.

오류:

```text
JSONDecodeError: Unexpected UTF-8 BOM (decode using utf-8-sig)
```

해결:

- `petcare_agent.contracts.schema_loader.load_json_schema()`에서 schema 파일을 `utf-8-sig`로 읽도록 변경했다.

추가로 pytest cache provider가 sandbox 환경에서 임시 cache 디렉터리 생성 권한 경고를 발생시켰다.

해결:

- `pyproject.toml`의 pytest `addopts`에 `-p no:cacheprovider`를 추가했다.
- 이전 실패 중 생성된 `pytest-cache-files-*` 잔여 디렉터리는 권한 문제로 삭제되지 않아 `.gitignore`에 ignore 패턴을 추가했다.

## 검증 결과

실행 명령:

```powershell
python -m pytest ai/tests -q
```

결과:

```text
.........                                                                [100%]
```

총 9개 테스트가 통과했다.

## 다음 단계 후보

문서상 다음 구현 순서는 Phase 2부터다.

- MVP triage checklist 8종 데이터 추가
- checklist loader 구현
- rule-based safety validator 구현
- 이후 baseline/change detection, question manager, LangGraph wiring 순으로 진행