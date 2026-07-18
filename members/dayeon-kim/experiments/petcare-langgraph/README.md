# PetCare AI LangGraph

## 최종 사용자 흐름

### 비응급 건강 이상

```text
상태 확인
→ 현재 확인된 내용과 현재 판단
→ 병원 방문 여부 질문
→ 예: 병원 전달용 PDF 생성
→ 아니오: 상태 확인 종료
```

비응급 답변에는 `권장 행동`과 `근거` 섹션을 표시하지 않습니다. RAG 검색 결과는 내부 판단에만 사용합니다.

병원 전달용 PDF는 줄글이 아니라 아래 항목으로 구성됩니다.

```text
반려동물 기본 정보
현재 증상
발생 및 경과
최근 일상 기록
진단 및 복용 기록
미확인 항목
주의 사항
```

### 응급 건강 이상

```text
응급 신호 감지
→ 운영 중인 가까운 병원 탐색
→ 가장 가까운 병원 선택
→ 병원 전달 이메일 작성
→ 이메일 전송
```

## 실행

```powershell
pip install -r requirements.txt
$env:OPENAI_API_KEY="YOUR_KEY"
python main.py
```

## 병원 검색 연결

로컬 테스트에서는 `DemoHospitalSearchProvider`가 사용됩니다. 실제 앱에서는 팀의 병원 검색 함수를 연결합니다.

```python
from petcare_agent.services import (
    TeamHospitalSearchAdapter,
    set_hospital_search_provider,
)

set_hospital_search_provider(
    TeamHospitalSearchAdapter(
        search_function
    )
)
```

검색 함수 반환 형식:

```json
{
  "hospital_id": "hospital_001",
  "name": "24시 동물병원",
  "address": "부산광역시 ...",
  "phone": "051-000-0000",
  "email": "hospital@example.com",
  "distance_km": 1.2,
  "is_open": true,
  "open_status": "운영 중",
  "source": "team-api"
}
```

## 이메일 전송

SMTP 환경변수가 있으면 실제 이메일을 전송합니다.

```powershell
$env:SMTP_HOST="smtp.example.com"
$env:SMTP_PORT="587"
$env:SMTP_USERNAME="username"
$env:SMTP_PASSWORD="password"
$env:SMTP_SENDER="petcare@example.com"
```

SMTP 설정이 없으면 실제 전송 대신 `tmp/outbox/`에 `.eml` 파일을 저장합니다. 이는 로컬 테스트에서 실수로 이메일이 발송되는 것을 막기 위한 동작입니다.

## PDF 출력

병원 방문 여부에 `예`라고 답하면 다음 위치에 PDF가 생성됩니다.

```text
artifacts/{session_id}_hospital_handoff.pdf
```

## 주요 모듈

| 파일 | 역할 |
|---|---|
| `nodes/assessment.py` | 입력 분류와 증상 구조화 |
| `nodes/safety.py` | 응급 규칙 판정 |
| `nodes/triage.py` | 증상별 문진 |
| `nodes/workflow.py` | 병원 방문 결정, 병원 검색, 이메일 전송 |
| `nodes/agents.py` | 상태 요약과 전달 데이터 생성 |
| `documents.py` | 항목형 병원 전달 PDF 생성 |
| `services.py` | LLM, RAG, 병원 검색, 이메일 Provider |
| `graph.py` | 전체 LangGraph 흐름 |
