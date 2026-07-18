# PetCare AI LangGraph

반려동물 프로필, 일상 기록, 진단서와 RAG 근거를 바탕으로 일반 대화, 상태 확인, 응급 안내, 병원 전달용 요약을 처리하는 LangGraph 프로젝트입니다.

## 구조

```text
petcare-langgraph/
├─ main.py
├─ requirements.txt
├─ data/
├─ tests/
└─ petcare_agent/
   ├─ config.py
   ├─ models.py
   ├─ prompts.py
   ├─ response.py
   ├─ services.py
   ├─ local_data.py
   ├─ graph.py
   ├─ runtime.py
   ├─ cli.py
   └─ nodes/
      ├─ context.py
      ├─ assessment.py
      ├─ safety.py
      ├─ triage.py
      └─ agents.py
```

## 주요 파일

| 파일 | 역할 |
|---|---|
| `prompts.py` | 에이전트 역할과 답변 스타일 |
| `response.py` | 응급 답변 구성, 근거 표시, 이모지 제거 |
| `nodes/assessment.py` | 입력 분류와 증상 구조화 |
| `nodes/safety.py` | 응급 규칙과 악화 판단 |
| `nodes/triage.py` | 증상별 문진 흐름 |
| `nodes/agents.py` | 일반, 비응급, 응급, RAG, 병원 요약 응답 |
| `graph.py` | LangGraph 노드 연결 |
| `runtime.py` | 세션 시작과 문진 재개 |
| `cli.py` | 터미널 실행 화면 |

## 답변 스타일

사용자에게 보이는 답변은 다음 기준을 따릅니다.

- 이모지와 장식용 기호를 사용하지 않습니다.
- 단순 인사는 짧게 답합니다.
- 건강 상태 답변은 충분한 설명을 유지합니다.
- 같은 내용과 과거 기록을 불필요하게 반복하지 않습니다.
- 비응급 결과는 현재 상태, 판단, 권장 행동, 근거 순서로 정리합니다.
- 응급 결과는 고위험 신호와 즉시 해야 할 행동을 분명하게 제시합니다.
- 모델이 이모지를 생성하더라도 `response.py`에서 제거합니다.

## 실행

```bash
pip install -r requirements.txt
```

`data/` 폴더에 다음 파일을 넣습니다.

```text
pet_profile.json
daily_entries.json
diagnoses.json
```

PowerShell에서 API 키를 설정합니다.

```powershell
$env:OPENAI_API_KEY="YOUR_KEY"
```

실행합니다.

```bash
python main.py
```

기본 실행 화면에는 답변만 표시됩니다. 평가용 route와 trace를 확인하려면 대화창에서 `/debug`를 입력합니다.

## 명령어

```text
/help    명령어 확인
/debug   route와 trace 표시 전환
/state   현재 LangGraph State 확인
/memory  최근 대화 확인
/reload  JSON 다시 읽기
/quit    종료
```
