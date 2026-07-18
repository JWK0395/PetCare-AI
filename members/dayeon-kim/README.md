# PetCare AI LangGraph — Modular

기존 단일 Python 파일을 기능별 모듈로 분리한 버전입니다.

## 폴더 구조

```text
petcare-langgraph/
├─ main.py
├─ requirements.txt
├─ README.md
├─ data/
│  └─ README.md
├─ petcare_agent/
│  ├─ __init__.py
│  ├─ config.py
│  ├─ models.py
│  ├─ services.py
│  ├─ utils.py
│  ├─ local_data.py
│  ├─ graph.py
│  ├─ runtime.py
│  ├─ cli.py
│  └─ nodes/
│     ├─ __init__.py
│     ├─ context.py
│     ├─ assessment.py
│     ├─ safety.py
│     ├─ triage.py
│     └─ agents.py
└─ tests/
   └─ test_smoke.py
```

## 각 파일 역할

| 파일 | 역할 |
|---|---|
| `config.py` | 환경변수와 OpenAI 설정 |
| `models.py` | Pydantic 요청·응답 모델과 LangGraph State |
| `services.py` | OpenAI 서비스와 RAG Provider |
| `utils.py` | 대화 기록, latency, 오류 처리 공통 함수 |
| `nodes/context.py` | 백엔드 Context 검증 및 요약 |
| `nodes/assessment.py` | 일반 대화/건강 입력 분류 및 증상 구조화 |
| `nodes/safety.py` | 응급 규칙, 회복 표현, 악화 누적 판단 |
| `nodes/triage.py` | 증상별 질문 Cycle과 unknown 처리 |
| `nodes/agents.py` | 일반·비응급·응급·RAG·Handoff Agent |
| `graph.py` | 전체 LangGraph 조립 |
| `runtime.py` | 그래프 시작·재개·세션 처리 |
| `local_data.py` | 로컬 JSON 파일 로딩 |
| `cli.py` | 터미널 대화 실행기 |
| `main.py` | 실행 진입점 |

## 실행

```bash
pip install -r requirements.txt
```

`data/` 폴더에 아래 파일을 넣습니다.

```text
data/
├─ pet_profile.json
├─ daily_entries.json
└─ diagnoses.json
```

환경변수를 설정합니다.

```bash
# PowerShell
$env:OPENAI_API_KEY="YOUR_KEY"
```

실행합니다.

```bash
python main.py
```

## 팀 RAG 연결

`services.py`의 기본 Demo Provider 대신 아래처럼 교체합니다.

```python
from petcare_agent.services import (
    TeamRAGAdapter,
    set_rag_provider,
)

set_rag_provider(
    TeamRAGAdapter(team_search_function)
)
```

## 테스트용 LLM 교체

```python
from petcare_agent.services import set_llm_service

set_llm_service(fake_llm)
```

모듈 import 시 API 키 입력이나 JSON 로딩을 실행하지 않습니다.
실제 LLM 호출 또는 `main.py` 실행 시에만 필요한 설정과 데이터를 불러옵니다.