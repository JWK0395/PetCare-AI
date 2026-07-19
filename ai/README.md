# PetCare AI — Agent 서비스 (별도 구현)

이 폴더는 LLM Agent(Safety → Context → Trend → RAG → Summary)를 구현하는 곳이다.
**서버(server/)와의 연결부는 이미 구현되어 있으며**, 이 문서의 HTTP 계약만 맞추면
서버 코드 수정 없이 Agent 가 연결된다.

## 연결 방법

`server/.env` 에서:

```env
AGENT_MODE=http                       # mock → http 로 변경
AGENT_BASE_URL=http://127.0.0.1:8100  # Agent 서비스 주소
AGENT_API_KEY=                        # 설정 시 Authorization: Bearer <key> 로 전달됨
```

- `AGENT_MODE=mock`(기본값)이면 Agent 없이 서버 내장 규칙 기반 응답으로 동작한다.
- `AGENT_MODE=http`이면 아래 4개 엔드포인트로 요청이 전달된다.

## HTTP 계약 (Agent 서비스가 구현해야 하는 것)

모든 엔드포인트는 `POST`, 요청/응답 모두 JSON.
서버가 pet(프로필)과 context(최근 30일 기록·진단서)를 만들어 보내므로,
Agent 는 DB 접근 없이 payload 만으로 판단하면 된다.

### 공통 payload 구조

```jsonc
// pet — 반려동물 프로필
{
  "id": 1, "name": "콩이", "species": "강아지", "breed": "말티즈 · 순종",
  "birth_date": "2021-09-14", "age_label": "만 4세", "sex": "수컷",
  "is_neutered": true, "weight_kg": 5.08, "size_class": "소형",
  "diseases": "슬개골 탈구 2기",       // 문자열 (여러 개면 쉼표 등으로 이어진 자유 텍스트)
  "medications": "",                    // 문자열
  "supplement": "관절 영양제 1일 1회",  // 문자열
  "allergies": "닭고기 알레르기"        // 문자열
}

// context — 서버가 만들어 주는 개인 데이터 (DB 스펙: daily_entries / diagnoses)
{
  "window_days": 30,
  "records": [            // 최근 30일 일일 기록 (오래된 순) — 모두 텍스트 상태값
    { "record_date": "2026-07-11",
      "raw_text": "아침에 사료를 반쯤 남겼다. 산책은 20분 정도 했고 오후에 노란 토를 한 번 했다.",
      "food": "사료 반쯤 남김 · 평소보다 감소",
      "water": "정상 범위",
      "activity": "산책 20분 · 평소보다 짧음",
      "symptom": "기력 저하",
      "stool": "정상",
      "vomit": "노란색 구토 1회 · 오후",
      "notes": "" }
  ],
  "diagnoses": [          // 확정 저장된 진단서 (오래된 순)
    { "date": "2026-07-02", "hospital": "행복한동물병원",
      "diagnosis": "슬개골 탈구 2기",
      "content": "슬개골 탈구 2기 소견 · 처방: 관절 영양제 1일 1회 · 체중 5.28kg" }
  ]
}
```

> DB 스펙상 기록은 모두 텍스트 상태값이다. 수치 기반 개인 기준선/추이는 서버가 미리
> 계산하지 않으며, **Agent(LLM)가 최근 30일 기록 텍스트와 일기 원문(`raw_text`)을 읽고**
> 추이·이상 신호를 판단한다. (응답의 `trends`/`trend_summary` 는 Agent 가 채우는 선택 필드다.)

### 1) POST `/agent/diary-extract` — 일기 구조화

일기형 자연어 기록을 건강 항목별 데이터로 분류한다.

요청:
```jsonc
{ "pet": {...}, "text": "아침에 사료를 반쯤 남겼다. ...", "record_date": "2026-07-11", "context": {...} }
```

응답:
```jsonc
{
  "items": [   // 화면의 "AI 일기에서 N개 기록을 정리했어요" 목록
    { "category": "식사", "value": "사료 반쯤 남김 · 평소보다 감소", "field": "food" },
    { "category": "음수", "value": "정상 범위", "field": "water" },
    { "category": "활동", "value": "산책 20분 · 평소보다 짧음", "field": "activity" },
    { "category": "구토", "value": "노란색 구토 1회 · 오후", "field": "vomit" },
    { "category": "증상", "value": "기력 저하", "field": "symptom" }
  ],
  "fields": {  // DailyEntry(daily_entries) 저장용 텍스트 필드 (보호자 수정 후 저장됨)
    "food": "사료 반쯤 남김 · 평소보다 감소",
    "water": "정상 범위",
    "activity": "산책 20분 · 평소보다 짧음",
    "symptom": "기력 저하",
    "stool": "",
    "vomit": "노란색 구토 1회 · 오후",
    "notes": ""
  }
}
```

### 2) POST `/agent/diagnosis-extract` — 진단서 구조화

업로드된 진단서(PDF 텍스트)에서 항목을 추출한다.

요청:
```jsonc
{ "pet": {...}, "file_name": "진단서_행복한동물병원_0702.pdf", "file_text": "...PDF에서 추출한 텍스트..." }
```

응답:
```jsonc
{
  "fields": {  // Diagnosis(diagnoses) 저장용 — date · hospital · diagnosis · content
    "date": "2026-07-02",
    "hospital": "행복한동물병원",
    "diagnosis": "슬개골 탈구 2기",
    "content": "슬개골 탈구 2기 소견 · 처방: 관절 영양제 1일 1회 · 체중 5.28kg"
  },
  "items_read": 4   // 화면의 "AI 진단서에서 N개 항목을 읽었어요"
}
```

### 3) POST `/agent/health-check` — AI 상태 체크 (핵심)

Safety(응급 감지) → Context/Trend(기준선 비교) → 추가 질문 → RAG 근거 → 판단.

요청:
```jsonc
{
  "pet": {...},
  "messages": [   // 대화 전체 (멀티턴)
    { "role": "user", "content": "오늘 밥을 거의 안 먹고 하루 종일 축 처져 있어요" },
    { "role": "assistant", "content": "..." },
    { "role": "user", "content": "오후에 노란 토를 한 번 했어요" }
  ],
  "context": {...}
}
```

응답:
```jsonc
{
  "reply": "오늘 안에 병원 상담을 권해요",
  "risk_level": "consult",        // normal | observe | consult | emergency
  "trend_summary": "식사 ▼32% · 활동 ▼18% · 구토 1회",
  "trends": [ { "metric": "식사", "change_pct": -32.0, "note": "..." } ],
  "reasons": [
    "식사량 3일 연속 개인 기준선 30% 이상 미달",
    "기력 저하 + 구토 동반 — 복합 신호"
  ],
  "evidence": "WSAVA 보호자 가이드 2024 v2 · 개인 기준선 30일",  // RAG 출처
  "followup_question": null,      // 정보 부족 시 추가 질문 (문자열)
  "can_generate_summary": true,   // "병원 전달용 요약 만들기" 버튼 노출 여부
  "show_hospitals": false,        // 응급 시 true → 앱이 24시 병원 목록 표시
  "transit_guidance": [],         // 응급 시 이동 중 대처 ["기도 확보", ...]

  // --- AI 모델 연결용 확장 필드 (선택, mock 은 빈 값) ---
  "actions": [                    // 앱이 버튼으로 그릴 후속 동작
    { "type": "generate_summary", "label": "병원 전달용 요약 만들기", "payload": {} },
    { "type": "send_email",       "label": "상태 문서 이메일",        "payload": {} }
    // type: generate_summary | save_summary_pdf | send_email | save_record
  ],
  "citations": [                  // RAG 근거 인용
    { "title": "WSAVA 보호자 가이드", "source": "wsava_2024_v2", "snippet": "..." }
  ]
}
```

응급 판정 시 규칙:
- `risk_level: "emergency"`, `show_hospitals: true`, `transit_guidance` 포함
- 일반 답변은 생략하고 병원 안내를 우선한다.

`actions` / `citations` 는 선택 필드다. 안전을 위해 PDF 저장·이메일 전송 같은 동작은
`actions` 로 앱에 위임하고, 최종 실행은 사용자 확인을 거친다 (자동 전송 금지).

### 4) POST `/agent/summary` — 병원 전달용 요약

요청:
```jsonc
{ "pet": {...}, "risk_level": "consult", "extra_note": "보호자 보완 입력(선택)", "context": {...} }
```

응답 — 문서 4섹션 구조(문서 정보 / 반려동물 정보 / 상태 / 주호소·변화):
```jsonc
{
  "content": {
    // 1. 문서 정보
    "title": "PetCare AI 병원 전달용 상태 요약",
    "data_period": "2026.06.17 ~ 2026.07.16",   // 사용 데이터 기간
    // 2. 반려동물 정보
    "pet_name": "콩이", "species": "강아지", "breed": "말티즈 · 순종",
    "sex_neuter": "수컷 / 중성화 완료", "age_label": "만 4세", "weight": "5.08kg",
    "medications": "관절 영양제 1일 1회", "allergies": "닭고기 알레르기",
    // 3. 상태
    "risk_label": "신속 상담 권장",            // 상태 분류
    "risk_signs": ["식사량 감소", "기력 저하", "구토 관찰"],  // 확인된 위험 징후
    // 4. 주호소 및 주요 변화
    "chief_complaint": "식욕 감소 · 기력 저하 · 구토",
    "major_changes": "최근 3일 식사량 감소 · 활동 감소 · 구토 발생",
    "progress": "식사: 사료 반쯤 남김 · 활동: 산책 20분 · 구토: 노란색 구토 1회",
    "owner_note": ""
  }
}
```

- `created_at`(생성 일시)은 서버가 저장 시각으로 채워 넣는다(응답 `content` 에는 없음).
- 응급 이메일도 같은 `content` 구조를 사용한다. (`POST /api/pets/{id}/emergency-emails`)

## 안전 원칙 (Agent 구현 시 반드시 지킬 것)

- 확정 진단, 약물 처방·복용량 변경 안내를 하지 않는다.
- 응급 신호(호흡곤란·청색증·경련·중독 등) 감지 시 일반 답변을 생략하고 병원 안내를 우선한다.
- 근거가 부족하면 추정하지 말고 `followup_question` 으로 되묻거나 미확인 항목으로 명시한다.
- 병원 전송 전 보호자의 최종 확인을 전제로 한다 (서버/앱이 확인 단계를 강제한다).

## 스캐폴드 (이 폴더에 이미 있음 — 바로 실행 가능)

```
ai/
├── app/
│   ├── main.py               # FastAPI — 위 4개 엔드포인트
│   ├── io_schemas.py         # 공통 입력 스키마 (pet / context / message)
│   ├── health_check.py       # AI 체크 입출력 스켈레톤 — run_health_check() 에 구현 연결
│   ├── diary_extract.py      # 일기 구조화 입출력 스켈레톤 — run_diary_extract()
│   ├── diagnosis_extract.py  # 진단서 추출 입출력 스켈레톤 — run_diagnosis_extract()
│   ├── graph.py              # 요약(run_summary) + LangGraph 골격
│   ├── rag.py                # 전문 건강정보 RAG (RagStore.retrieve TODO)
│   ├── tools.py              # 요약 PDF / 이메일 초안 (메인 서버 API 호출 예시)
│   └── config.py
├── data/rag_docs/  # RAG 원본 문서 (분류해서 적재)
└── requirements.txt
```

실행 (stub — server/.venv 재사용 가능):

```powershell
cd ai
..\server\.venv\Scripts\python -m uvicorn app.main:app --port 8100
# → http://127.0.0.1:8100/health 로 확인
```

그다음 `server/.env` 에서 `AGENT_MODE=http` 로 두고 메인 서버를 켜면
앱의 AI 기능(일기 구조화·상태 체크·요약)이 이 서비스로 전달된다.

## 목표 아키텍처 (우리가 채울 것)

```
[앱] ──▶ [PetCare 서버] ──HTTP──▶ [AI Agent (LangGraph)]
                                     │
        DB context(30일 기록·진단서 · 텍스트) ┤  ← 서버가 만들어 전달
                                     ├─▶ safety_node  (응급 우선)
                                     ├─▶ rag_node     (전문 건강정보 RAG)
                                     ├─▶ reason_node  (LLM 추론/판단)
                                     └─▶ action_node  (요약 PDF·이메일 actions)
```

- **DB 정보**: 서버가 `context` 로 전달 (Agent 는 DB 직접 접근 불필요)
- **RAG**: `rag.py` 에서 벡터스토어 검색 → `citations` 로 근거 반환
- **PDF/이메일/저장**: `tools.py` 또는 응답 `actions` 로 위임
  - 요약 PDF: 메인 서버 `GET /api/summaries/{id}/pdf` 가 렌더링
  - 이메일: `POST /api/pets/{id}/emergency-emails` 초안 → 사용자 확인 후 전송
- **LLM 프로바이더/모델**: 우리가 선택 (`ai/.env` 의 `LLM_*`)

## 참고

- 서버 쪽 연결 코드: `server/app/services/agent_client.py` (`HttpAgentClient`)
- mock 응답: 같은 파일의 `MockAgentClient` — **판단·추출을 하지 않는다.** 위험도·근거·
  추가질문·이동 안내·병원을 전부 빈 값으로 두고 "AI 가 연결되지 않았다" 는 사실만 알린다.
  (예전의 하드코딩 예시 응답과 `DEMO_PASSWORD` 게이트는 제거됐다 — 화면에 예시 표시가
  없어 보호자가 AI 판독 결과로 읽을 수 있었기 때문이다.)
- 따라서 앱에서 실제 분석을 보려면 `AGENT_MODE=http` + 이 AI 서비스 기동이 필요하다.
