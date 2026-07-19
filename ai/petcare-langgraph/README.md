# PetCare AI LangGraph

## 실행 흐름

```text
사용자 입력
   ↓
백엔드 Context 검증 및 State 초기화
   ↓
assessment_graph
   ├─ 일반 대화·기록 질문
   │    └─ chat_agent → 최종 답변
   │
   └─ 건강 이상 입력
        ↓
     question_manager
        ↓
     safety_guard
        ├─ 응급
        │    ├─ emergency_agent
        │    ├─ search_open_hospital
        │    ├─ generate_emergency_email
        │    └─ send_emergency_email
        │
        └─ 비응급
             ├─ rag_agent
             ├─ chat_agent
             ├─ hospital_visit_decision
             ├─ 방문 yes → handoff_subgraph → PDF 생성
             └─ 방문 no → close_non_emergency
```

### 일반 대화·기록 질문

```text
질문 분석
→ 관련 기간 및 기록 선별
→ 축소 Context 구성
→ LLM 최종 답변 1회
```

### 증상 문진

```text
증상 코드 감지
→ 이미 확인된 정보 검사
→ 누락 항목 질문
→ 사용자 답변 State 누적
→ 추가 증상 확인
→ 응급 규칙 검사
```

### 응급 흐름

```text
응급 신호 감지
→ RAG 및 LLM 답변 생성 생략
→ 사용자 위치 기반 병원 검색
→ 병원 전달 정보 구성
→ 이메일 전송 또는 로컬 Outbox 저장
```

### 비응급 흐름

```text
문진 완료
→ RAG 검색
→ 현재 상태 설명 생성
→ 병원 방문 의사 확인
→ PDF 생성 또는 상태 확인 종료
```

---

## LLM 호출 정책

| 실행 사례 | LLM 호출 수 |
|---|---:|
| 일반 대화 및 기록 질문 | 1회 |
| 증상 문진 진행 | 0회 |
| 비응급 최종 상태 설명 | 1회 |
| 응급 판정 및 안내 | 0회 |
| 병원 전달 PDF 생성 | 0회 |
| 응급 이메일 생성 | 0회 |
| RAG 검색 | LLM 없음 / Embedding 호출 가능 |

### 규칙 기반 처리 범위

- 입력 유형 분류
- 증상 키워드 감지
- 부정 표현 판별
- 문진 질문 선택
- 응급 위험 징후 판정
- 병원 방문 의사 판별
- 이전 문진 연속성 처리
- 병원 전달 데이터 구성
- PDF 생성
- 응급 이메일 본문 구성

### LLM 사용 범위

- 일반 대화 및 등록 기록 기반 최종 답변 생성
- 비응급 문진 완료 후 현재 상태 설명 생성

---

## 주요 개선 사항

### 문진 상태 연속성

다음 표현 입력 시 이전 문진 State 유지 및 신규 정보 추가 처리.

```text
또 아파 보여
계속 안 좋아 보여
여전히 밥을 안 먹어
아직도 설사해
다시 토했어
전보다 더 안 좋아
```

### 병원 방문 결정 변경

```text
사용자: 아니오
→ 상태 확인 종료

사용자: 그냥 병원에 갈래
→ 이전 문진 State 복원
→ 방문 결정 yes 변경
→ 병원 전달용 PDF 생성
```

### 중복 질문 방지

```text
사용자 입력:
“일주일 전부터 평소의 70% 정도만 먹어요.”

확인 완료:
- 지속 기간 약 1주일
- 평소 대비 섭취량 약 70%

처리:
- 기간 및 섭취량 재질문 생략
- 다음 누락 항목 확인
```

### 부정 표현 오탐 방지

다음 표현에 대한 증상 미감지 처리.

```text
호흡은 괜찮아요
숨은 안 힘들어요
설사는 없어요
구토는 안 했어요
소변은 평소처럼 잘 봐요
다른 증상은 딱히 없어요
```

### 모호한 통증 표현 처리

`아파 보여요`만으로 통증 코드 확정 방지 및 직접적인 통증 근거 우선 처리.

```text
만지면 피함
만지면 소리를 냄
절뚝거림
특정 부위를 반복적으로 핥음
움직일 때 통증 반응
```

### Context 및 토큰 축소

- 최근 대화 최대 8개 메시지 사용
- 질문·증상 관련 일기 기본 최대 8개 선별
- 전체 일기 객체 대신 날짜 및 관련 필드 중심 압축
- 관련 진단 기록 기본 최대 3개 선별
- 기간 표현에 따른 검색 범위 조정
- 선택 결과의 `prompt_context_stats` 저장

---

## 디렉터리 구성

```text
petcare-langgraph/
├─ main.py
├─ requirements.txt
├─ requirements-dev.txt
├─ pytest.ini
├─ data/
│  ├─ pet_profile.json
│  ├─ daily_entries.json
│  └─ diagnoses.json
├─ petcare_agent/
│  ├─ __init__.py
│  ├─ config.py
│  ├─ models.py
│  ├─ services.py
│  ├─ graph.py
│  ├─ runtime.py
│  ├─ local_data.py
│  ├─ prompt_context.py
│  ├─ prompts.py
│  ├─ response.py
│  ├─ handoff.py
│  ├─ documents.py
│  ├─ utils.py
│  └─ nodes/
│     ├─ assessment.py
│     ├─ context.py
│     ├─ triage.py
│     ├─ safety.py
│     ├─ agents.py
│     └─ workflow.py
└─ tests/
```

### 주요 파일 역할

| 파일 | 역할 |
|---|---|
| `graph.py` | LangGraph 노드 및 조건부 분기 구성 |
| `runtime.py` | 그래프 실행, 체크포인터, `interrupt/resume`, 세션 State 관리 |
| `models.py` | 요청, State, RAG, 병원, 이메일, Handoff 모델 정의 |
| `services.py` | LLM·RAG·병원 검색·이메일 Provider 및 의존성 정의 |
| `nodes/assessment.py` | 규칙 기반 입력 분류 및 초기 문진 상태 구성 |
| `nodes/context.py` | Backend Context 검증 및 요약 구성 |
| `nodes/triage.py` | 증상 감지, 질문 계획, 문진 State 갱신 |
| `nodes/safety.py` | 응급 규칙 및 부정 표현 검사 |
| `nodes/agents.py` | 일반·비응급 답변, RAG 검색, 응급 안내 처리 |
| `nodes/workflow.py` | 방문 결정, 병원 검색, 이메일 생성·전송 처리 |
| `prompt_context.py` | 질문 관련 기록 선별 및 프롬프트 Context 축소 |
| `handoff.py` | 병원 전달용 구조화 데이터 생성 |
| `documents.py` | 병원 전달용 PDF 렌더링 |
| `local_data.py` | 로컬 JSON 로드 및 Graph 요청 생성 |

---

## 로컬 실행

본 모듈 디렉터리 기준 실행.

### 1. 패키지 설치

```powershell
python -m pip install -r requirements.txt
```

### 2. 환경변수 설정

```powershell
$env:OPENAI_API_KEY="YOUR_OPENAI_API_KEY"
$env:OPENAI_MODEL="gpt-5.4-mini"
```

`OPENAI_MODEL` 미설정 시 `gpt-5.4-mini` 기본 사용.

### 3. 샘플 데이터 배치

```text
data/pet_profile.json
data/daily_entries.json
data/diagnoses.json
```

### 4. 실행

```powershell
python main.py
```

### CLI 명령어

```text
/help     도움말
/debug    내부 디버그 표시 전환
/state    현재 Graph State 확인
/memory   최근 대화 확인
/reload   로컬 JSON 재로드
/outbox   로컬 이메일 발송함 열기
/quit     실행 종료
```

---

## 테스트

### 개발 의존성 설치

```powershell
python -m pip install -r requirements-dev.txt
```

### 전체 테스트

```powershell
python -m pytest -q
```

### 현재 검증 결과

```text
19 passed
```

### 주요 테스트 범위

- 일반 대화 및 건강 이상 분기
- 다중 턴 문진 및 State 유지
- 응급·비응급 판정
- 중복 질문 방지
- 증상 부정 표현 오탐 방지
- 모호한 통증 표현 처리
- 창백함 확인 질문
- 병원 방문 결정 변경
- 사용자 위치 State 전달
- RAG 미연결 Fallback
- 병원 전달용 PDF 생성
- 응급 이메일 생성
- Dependency Injection
- Pydantic 모델 검증
- 프롬프트 Context 축소

---