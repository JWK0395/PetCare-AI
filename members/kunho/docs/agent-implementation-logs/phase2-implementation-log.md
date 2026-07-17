# Agent Phase 2 Implementation Log

Date: 2026-07-16
Repository: PetCare-AI
Scope: `docs/agent-implementation-roadmap.md` 기준 Phase 2

## 기준 문서

구현 전에 아래 문서와 계약을 확인하고, Phase 2 범위 안에서만 작업했다.

- `docs/agent-architecture.md`
- `docs/agent-implementation-spec.md`
- `docs/agent-implementation-roadmap.md`
- `docs/api-endpoints.md`
- `docs/database-schema.md`
- `contracts/jsonschema/*.json`
- `docs/agent-implementation-logs/phase0-phase1-implementation-log.md`

## 구현 범위

### Checklist Template Data

`ai/src/petcare_agent/safety/checklists/mvp_triage_templates.json`에 MVP triage checklist 8종을 추가했다.

- `cat_cough_triage`
- `dog_cough_triage`
- `vomiting_triage`
- `diarrhea_triage`
- `breathing_triage`
- `seizure_triage`
- `toxicity_triage`
- `urinary_triage`

각 template은 기존 Phase 1의 `ChecklistTemplate`, `ChecklistItem` pydantic 모델과 `contracts/jsonschema/triage-checklist.schema.json` 계약에 맞게 작성했다.

### Checklist Loader

`ai/src/petcare_agent/safety/checklist_loader.py`를 추가했다.

주요 동작:

- packaged JSON checklist template 로드
- 로드 시 `ChecklistTemplate.model_validate()`로 pydantic 검증
- `species + chief_complaint` 기반 template 선택
- cat/dog 공용 template 처리
- unknown species 또는 unsupported chief complaint에 대한 보수적 fallback 처리

Fallback 정책:

- 정확한 species + chief complaint template이 있으면 우선 선택
- 공용 `cat/dog` chief complaint template이 있으면 선택
- 그래도 없으면 `breathing_triage`로 fallback

`breathing_triage` fallback은 즉시 확인해야 하는 호흡 관련 red flag를 포함하므로, unknown 입력을 낮은 위험으로 잘못 보내지 않는 보수적 선택이다.

## 질문 반복 제한 계약 반영

이번 Phase 2에서는 Question Manager나 validator를 구현하지 않았다.

다만 이후 Question Manager가 최대 2회 질문 제한을 지킬 수 있도록 각 checklist item에 다음 metadata를 포함했다.

- `question_text`
- `priority`
- `metadata.question_group`
- `metadata.question_limit_contract = "max_2_total_safety_questions"`

모든 question priority는 1~5 범위 안에 두었다.

## 지킨 제약

- DB schema 변경 없음
- API 변경 없음
- 신규 endpoint 추가 없음
- `assessment-context` endpoint 추가 없음
- LangGraph wiring 구현 없음
- LLM adapter 구현 없음
- 실제 RAG 호출 구현 없음
- DB context loader 구현 없음
- rule-based validator 구현 없음
- RAG는 이번 범위에서 연결하지 않음
- 응급 판단 구조는 "LLM이 checklist를 채우고, rule이 checklist를 판정" 계약을 유지하도록 data/loader까지만 구현

## 추가/수정 파일

- `ai/src/petcare_agent/safety/__init__.py`
- `ai/src/petcare_agent/safety/checklist_loader.py`
- `ai/src/petcare_agent/safety/checklists/__init__.py`
- `ai/src/petcare_agent/safety/checklists/mvp_triage_templates.json`
- `ai/tests/test_checklist_loader.py`
- `pyproject.toml`

## 테스트

추가한 테스트:

- checklist loader 로드 테스트
- checklist id별 단일 로드 테스트
- 각 chief complaint별 template 선택 테스트
- cat/dog 공용 template 선택 테스트
- 보수 fallback 선택 테스트
- required red flag item이 비어 있지 않은지 테스트
- question priority가 1~5 범위인지 테스트
- pydantic model과 JSON schema 필드 계약 충돌이 없는지 테스트

실행 명령:

```powershell
python -m pytest ai/tests -q
```

결과:

```text
.....................................                                    [100%]
```

총 37개 테스트가 통과했다.

## Phase 3 다음 작업

다음 단계는 `docs/agent-implementation-roadmap.md` 기준 Phase 3이다.

- `safety/rules.py` 추가
- `safety/validator.py` 추가
- emergency, urgent, unknown, non-emergency rule priority 구현
- `safety_question_turns < 2`이면 `needs_more_info`
- `safety_question_turns >= 2`이면 `unknown_after_max_questions`
- rule hit trace 데이터 생성
- validator unit test 추가

