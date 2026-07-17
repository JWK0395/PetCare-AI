<div align="center">

# 🐾 PetCare AI Agent

**반려동물 증상 평가 에이전트** — 보호자 입력과 반려동물 기록을 LangGraph로 평가해 상담 응답과 병원 인계 정보를 반환합니다.

<br>

[![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/) [![LangGraph](https://img.shields.io/badge/LangGraph-1.0%2B-1C3C3C?style=flat-square&logo=langchain&logoColor=white)](https://langchain-ai.github.io/langgraph/) [![OpenAI](https://img.shields.io/badge/OpenAI-structured_outputs-412991?style=flat-square&logo=openai&logoColor=white)](https://platform.openai.com/docs/) [![ChromaDB](https://img.shields.io/badge/ChromaDB-RAG-5B21B6?style=flat-square)](https://www.trychroma.com/) [![pytest](https://img.shields.io/badge/pytest-tests-0A9EDC?style=flat-square&logo=pytest&logoColor=white)](https://docs.pytest.org/)

<br>

[주요 기능](#주요-기능) · [빠른 시작](#빠른-시작) · [사용법](#사용법) · [설정](#설정) · [아키텍처](#아키텍처) · [AI 흐름](#ai-흐름) · [의존성](#의존성) · [라이선스](#라이선스)

</div>

---

## ✨ 주요 기능

- **증상 평가 그래프** — 사용자 입력을 의도 분류, 기록 조회, 변화 감지, 안전성 검사, 응답 생성 단계로 라우팅합니다.
- **안전성 우선 트리아지** — 체크리스트 템플릿과 규칙 기반 검증으로 응급 신호와 추가 질문을 결정합니다.
- **Cornell RAG 검색** — Cornell 반려동물 건강 자료를 ChromaDB로 검색하고 출처가 있는 일반 건강 정보를 구성합니다.
- **병원 인계 컨텍스트** — 기존 `GET /api/pets/{pet_id}/handoff-context?days=3` 응답을 그래프 상태로 변환합니다.
- **로컬 하네스** — 데모 데이터 번들로 대화형 테스트, 단일 턴 실행, JSONL 리플레이를 수행합니다.
- **민감정보 제한 추적** — LangSmith 메타데이터에 원문 대화와 의료 기록 텍스트를 남기지 않습니다.

---

## 🚀 빠른 시작

저장소 루트 `PetCare-AI`에서 실행합니다.

### 1. Environment setup

```powershell
python -m pip install -e .
```

### 2. Credentials / config

```powershell
Copy-Item .env.example .env
```

실제 OpenAI 구조화 출력과 RAG 임베딩을 사용하려면 `.env`의 `OPENAI_API_KEY`를 채웁니다. 로컬 데모 하네스는 키가 비어 있어도 보수적인 fallback 경로로 실행됩니다.

### 3. Run

```powershell
python -m petcare_agent.harness --data-zip examples\data_bundles\petcare_db_v1_demo --pet-id 1 --once "모찌가 기침을 해요"
```

---

## 📖 사용법

### 로컬 하네스

```powershell
python -m petcare_agent.harness --data-zip examples\data_bundles\petcare_db_v1_demo --list-pets
```

```powershell
python -m petcare_agent.harness --data-zip examples\data_bundles\petcare_db_v1_demo --pet-id 2
```

```powershell
python -m petcare_agent.harness --data-zip examples\data_bundles\petcare_db_v1_demo --pet-id 3 --replay members\kunho\ai\tests\fixtures\triage_handoff_golden.jsonl
```

> 하네스 안에서는 `/state`, `/handoff`, `/visit yes|no|undecided|not_asked` 명령으로 그래프 상태와 병원 방문 의도를 확인할 수 있습니다.

### Python 런타임

```python
from petcare_agent.runtime.adapter import build_existing_api_runtime_adapter

adapter = build_existing_api_runtime_adapter()
result = adapter.run(
    {
        "request_id": "req_demo_001",
        "conversation_id": "conv_demo",
        "pet_id": 1,
        "user_input": "고양이가 기침을 해요.",
        "locale": "ko-KR",
        "timezone": "Asia/Seoul",
        "conversation_history": [],
    }
)

print(result.response.assistant_message)
```

### Cornell RAG 관리

```powershell
python -m pip install -e ".[rag]"
```

```powershell
python -m petcare_rag.manage_cornell_rag_db check
python -m petcare_rag.manage_cornell_rag_db index
python -m petcare_rag.manage_cornell_rag_db query --species cat --query "cat coughing" --top-k 5
python -m petcare_rag.manage_cornell_rag_db evaluate
```

### 테스트

```powershell
python -m pytest members\kunho\ai\tests
```

---

## ⚙️ 설정

Everything is controlled through `.env` — no code changes needed to switch model, tracing, backend API, or RAG settings.

| Key | Default | Description |
|-----|---------|-------------|
| `OPENAI_API_KEY` | empty | OpenAI 구조화 출력과 임베딩 요청에 사용합니다. |
| `OPENAI_MODEL` | `gpt-5.4-mini` | 그래프 노드의 구조화 출력 모델을 지정합니다. |
| `PETCARE_API_BASE_URL` | `http://localhost:8000` | 기존 PetCare 백엔드 API의 기준 URL입니다. |
| `PETCARE_ENV` | `local` | 추적 태그와 런타임 환경 구분에 사용합니다. |
| `LANGSMITH_TRACING` | `false` | LangSmith 추적 활성화 여부를 제어합니다. |
| `LANGSMITH_API_KEY` | empty | LangSmith 추적을 보낼 때 사용하는 키입니다. |

<details>
<summary>Full list</summary>

| Key | Default | Description |
|-----|---------|-------------|
| `LANGSMITH_PROJECT` | `petcare-ai-assessment` | LangSmith 프로젝트 이름입니다. |
| `LANGSMITH_RUN_PREFIX` | `assessment_graph` | 그래프와 노드 trace run 이름의 접두사입니다. |
| `PETCARE_RAG_DB_PATH` | `rag_data/chroma` | Cornell RAG ChromaDB 저장 경로입니다. |
| `PETCARE_RAG_COLLECTION` | `cornell_pet_health_text_embedding_3_small_1536` | RAG 검색에 사용할 ChromaDB 컬렉션 이름입니다. |
| `PETCARE_RAG_SERVICE_TOKEN` | empty | RAG HTTP API의 `X-PetCare-Token` 검증에 사용합니다. |
| `PETCARE_CONTRACTS_DIR` | auto-detect | JSON Schema contract 경로를 수동 지정할 때 사용합니다. |

</details>

---

## 🏗️ 아키텍처

```
members/kunho/ai/
├── src/
│   ├── petcare_agent/
│   │   ├── graphs/          # LangGraph wiring
│   │   ├── nodes/           # graph node functions
│   │   ├── safety/          # checklist and triage rules
│   │   ├── rag/             # graph-facing RAG adapter
│   │   ├── runtime/         # backend adapter boundary
│   │   ├── harness/         # local console runner
│   │   ├── schemas/         # pydantic graph models
│   │   ├── prompts/         # structured-output prompts
│   │   └── tracing.py       # LangSmith metadata helpers
│   └── petcare_rag/
│       ├── manage_cornell_rag_db.py  # ChromaDB index CLI
│       ├── pipeline.py               # direct RAG pipeline
│       ├── api.py                    # FastAPI RAG boundary
│       └── models.py                 # RAG response models
└── tests/                            # graph, node, RAG, contract tests
```

```
User message
   │  GraphRequest or session turn
   ▼
db_context_loader ──▶ existing API or data bundle
   │  pet profile, daily entries, diagnoses
   ▼
intent_classifier ──▶ social_chat ──▶ chat_agent ──▶ GraphResponse
   │  symptom, followup, handoff, or general route
   ▼
baseline_builder ──▶ state_updater ──▶ change_detector
   │  current status and recent baseline
   ▼
safety_guard ──▶ question_manager ──▶ GraphResponse
   │  complete checklist or emergency signal
   ├────────────▶ emergency_agent ──▶ GraphResponse
   │  non-emergency or urgent guidance path
   ▼
evidence_planner ──▶ rag_agent ──▶ answer_composer ──▶ answer_guard
                                                        │  optional handoff
                                                        ▼
                                                handoff_subgraph ──▶ GraphResponse
```

> 핵심 결정은 그래프가 외부 I/O를 직접 소유하지 않고 `DBContextProvider`, `RAGAdapter`, `StructuredOutputClient` 경계로 주입받는 구조입니다.

---

## 🤖 AI 흐름

| Stage | Main files | Output |
|-------|------------|--------|
| Turn understanding | `nodes/intent_classifier.py`, `prompts/turn_understanding.md` | 의도, 종, 증상, 안전성 검사 필요 여부 |
| Context loading | `nodes/db_context_loader.py`, `api/handoff_context.py` | 반려동물 프로필, 최근 일지, 진단 기록 |
| Triage | `nodes/safety_guard.py`, `safety/checklists/mvp_triage_templates.json` | 위험도, 누락 질문, 응급 규칙 |
| Retrieval | `nodes/rag_agent.py`, `rag/cornell.py`, `petcare_rag/pipeline.py` | Cornell 근거 chunk와 citation |
| Response | `nodes/answer_composer.py`, `nodes/answer_guard.py` | 보호자 응답, 출처 요약, guard 결과 |
| Handoff | `graphs/subgraphs/handoff.py`, `nodes/handoff_summary_builder.py` | 병원 전달 요약과 이메일 초안 |

Cornell RAG는 `text-embedding-3-small` 1536차원 임베딩과 `cornell_pet_health_text_embedding_3_small_1536` 컬렉션을 기준으로 동작합니다. 기본 corpus 검증은 `732`개 chunk를 기대합니다.

---

## 📦 의존성

| Package | Role |
|---------|------|
| `langgraph` | 상태 그래프 컴파일과 노드 라우팅 |
| `langchain-core`, `langchain-openai` | LangGraph 생태계 통합 |
| `openai` | 구조화 출력과 RAG 임베딩 요청 |
| `pydantic`, `pydantic-settings` | 요청, 응답, 설정 검증 |
| `langsmith` | 선택적 관측성 추적 |
| `chromadb`, `tiktoken` | 선택적 Cornell RAG 인덱싱과 검색 |
| `pytest` | 노드, 그래프, contract 회귀 테스트 |

`petcare_rag.api`의 FastAPI 서버를 직접 띄우려면 `fastapi`와 `uvicorn`을 별도로 설치합니다.

---

## 📄 라이선스

현재 저장소 루트에 `LICENSE` 파일이 없습니다. 외부 배포 전에 라이선스를 추가하고 이 섹션을 갱신하세요.
