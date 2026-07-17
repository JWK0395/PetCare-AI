# PetCare-AI API Endpoints

PetCare-AI에서 사용할 수 있는 주요 API 명세를 정리한 문서입니다.

## 1. 반려동물 프로필 조회

특정 반려동물의 기본 프로필 정보를 조회합니다.

### Endpoint

```http
GET /api/pets/{pet_id}
```

### Path Parameter

| 이름 | 타입 | 설명 |
| --- | --- | --- |
| `pet_id` | int | 조회할 반려동물 ID |

### Response Example

```json
{
  "id": 1,
  "name": "콩이",
  "species": "dog",
  "breed": "말티즈",
  "birth_date": "2021-09-14",
  "sex": "male",
  "is_neutered": true,
  "weight_kg": 5.08,
  "size_class": "small",
  "diseases_medications_allergies": [
    {
      "type": "disease",
      "name": "슬개골 탈구 2기"
    },
    {
      "type": "medication",
      "name": "관절 영양제",
      "details": "1일 1회"
    },
    {
      "type": "allergy",
      "name": "닭고기"
    }
  ],
  "created_at": "2026-07-15T09:00:00+09:00"
}
```

## 2. 최근 건강 기록 조회

특정 기간의 일기 기반 건강 기록을 조회합니다.

### Endpoint

```http
GET /api/pets/{pet_id}/daily-entries
```

### Path Parameter

| 이름 | 타입 | 설명 |
| --- | --- | --- |
| `pet_id` | int | 조회할 반려동물 ID |

### Query Parameters

| 이름 | 타입 | 필수 여부 | 설명 |
| --- | --- | --- | --- |
| `from` | date | 선택 | 조회 시작일 |
| `to` | date | 선택 | 조회 종료일 |

### Query Example

```text
from=2026-06-15
to=2026-07-15
```

### Response Example

```json
{
  "pet_id": 1,
  "from": "2026-06-15",
  "to": "2026-07-15",
  "daily_entries": [
    {
      "id": 1,
      "pet_id": 1,
      "record_date": "2026-07-14",
      "raw_text": "아침에 밥을 반쯤 남겼고 오후에 노란 토를 한 번 했다.",
      "food": "사료를 평소의 절반 정도 먹음",
      "water": "평소와 비슷함",
      "activity": "산책 20분",
      "symptom": "기력 저하",
      "stool": "정상 배변, 설사 없음",
      "vomit": "노란색 구토 1회",
      "notes": "오후부터 조금 처져 있었음"
    }
  ]
}
```

## 3. 건강 기록 저장

사용자가 작성한 일기와 추출된 건강 상태 정보를 저장합니다.

### Endpoint

```http
POST /api/pets/{pet_id}/daily-entries
```

### Path Parameter

| 이름 | 타입 | 설명 |
| --- | --- | --- |
| `pet_id` | int | 건강 기록을 저장할 반려동물 ID |

### Request Example

```json
{
  "record_date": "2026-07-15",
  "raw_text": "아침에 밥을 반쯤 남겼고 오후에 노란 토를 한 번 했다.",
  "food": "사료를 평소의 절반 정도 먹음",
  "water": "평소와 비슷함",
  "activity": "산책 20분",
  "symptom": "기력 저하",
  "stool": "정상 배변, 설사 없음",
  "vomit": "노란색 구토 1회",
  "notes": "오후부터 조금 처져 있었음"
}
```

### Response Example

```json
{
  "id": 123,
  "pet_id": 1,
  "record_date": "2026-07-15",
  "status": "saved"
}
```

## 4. 진단서 업로드

진단서 원본 파일을 업로드합니다.

### Endpoint

```http
POST /api/pets/{pet_id}/documents
Content-Type: multipart/form-data
```

### Path Parameter

| 이름 | 타입 | 설명 |
| --- | --- | --- |
| `pet_id` | int | 진단서를 업로드할 반려동물 ID |

### Response Example

```json
{
  "original_file_ref": "documents/diagnosis_001.pdf",
  "upload_status": "uploaded"
}
```

## 5. 진단서 추출 결과 저장

업로드된 진단서에서 추출한 진단 정보를 저장합니다.

### Endpoint

```http
POST /api/pets/{pet_id}/diagnoses
```

### Path Parameter

| 이름 | 타입 | 설명 |
| --- | --- | --- |
| `pet_id` | int | 진단 정보를 저장할 반려동물 ID |

### Request Example

```json
{
  "date": "2026-07-02",
  "hospital": "행복한동물병원",
  "diagnosis": "슬개골 탈구 2기",
  "content": "관절 영양제 1일 1회, 30일 복용. 무리한 활동을 피하고 경과 관찰.",
  "original_file_ref": "documents/diagnosis_001.pdf"
}
```

### Response Example

```json
{
  "id": 1,
  "pet_id": 1,
  "status": "saved"
}
```

## 6. 진단서 목록 조회

특정 반려동물의 진단서 목록을 조회합니다.

### Endpoint

```http
GET /api/pets/{pet_id}/diagnoses
```

### Path Parameter

| 이름 | 타입 | 설명 |
| --- | --- | --- |
| `pet_id` | int | 진단서 목록을 조회할 반려동물 ID |

### Query Parameters

| 이름 | 타입 | 필수 여부 | 설명 |
| --- | --- | --- | --- |
| `limit` | int | 선택 | 조회할 진단서 최대 개수 |

### Query Example

```text
limit=20
```

### Response Example

```json
{
  "diagnoses": [
    {
      "id": 1,
      "pet_id": 1,
      "date": "2026-07-02",
      "hospital": "행복한동물병원",
      "diagnosis": "슬개골 탈구 2기",
      "content": "관절 영양제 1일 1회, 30일 복용. 무리한 활동을 피하고 경과 관찰.",
      "original_file_ref": "documents/diagnosis_001.pdf"
    }
  ]
}
```

## 7. 병원 전달용 Context 조회

병원 전달에 필요한 반려동물 프로필, 최근 건강 기록, 진단서 정보를 한 번에 조회합니다.

### Endpoint

```http
GET /api/pets/{pet_id}/handoff-context
```

### Path Parameter

| 이름 | 타입 | 설명 |
| --- | --- | --- |
| `pet_id` | int | 병원 전달용 Context를 조회할 반려동물 ID |

### Query Parameters

| 이름 | 타입 | 필수 여부 | 설명 |
| --- | --- | --- | --- |
| `days` | int | 선택 | 최근 조회 기간 |

### Query Example

```text
days=30
```

### Response Example

```json
{
  "pet": {
    "id": 1,
    "name": "콩이",
    "species": "dog",
    "breed": "말티즈",
    "birth_date": "2021-09-14",
    "sex": "male",
    "is_neutered": true,
    "weight_kg": 5.08,
    "size_class": "small",
    "diseases_medications_allergies": [
      {
        "type": "disease",
        "name": "슬개골 탈구 2기"
      },
      {
        "type": "medication",
        "name": "관절 영양제",
        "details": "1일 1회"
      },
      {
        "type": "allergy",
        "name": "닭고기"
      }
    ]
  },
  "recent_daily_entries": [
    {
      "id": 1,
      "record_date": "2026-07-14",
      "food": "사료를 평소의 절반 정도 먹음",
      "water": "평소와 비슷함",
      "activity": "산책 20분",
      "symptom": "기력 저하",
      "stool": "정상 배변, 설사 없음",
      "vomit": "노란색 구토 1회",
      "notes": "오후부터 조금 처져 있었음"
    }
  ],
  "diagnoses": [
    {
      "id": 1,
      "date": "2026-07-02",
      "hospital": "행복한동물병원",
      "diagnosis": "슬개골 탈구 2기",
      "content": "관절 영양제 1일 1회, 30일 복용.",
      "original_file_ref": "documents/diagnosis_001.pdf"
    }
  ],
  "unknown_items": [],
  "data_from": "2026-06-15",
  "data_to": "2026-07-15",
  "generated_at": "2026-07-15T16:35:00+09:00"
}
```


## 8. Assessment Graph Runtime Boundary

현재 저장소에는 Assessment Graph 자체를 노출하는 HTTP 라우터가 포함되어 있지 않습니다. 백엔드가 그래프를 호출할 때의 안정적인 Python 경계는 `petcare_agent.runtime.adapter`입니다.

주요 진입점:

```python
from petcare_agent.runtime.adapter import build_existing_api_runtime_adapter, run_graph_request
```

`build_existing_api_runtime_adapter(...)`는 다음 기본 의존성을 연결합니다.

- DB context: `GET /api/pets/{pet_id}/handoff-context?days=3` 형태의 기존 API provider
- RAG: in-process `CornellRAGAdapter`
- validation: Pydantic 모델과 `contracts/jsonschema/agent-graph-*.schema.json`

### Graph Request Contract

필수 필드:

| Field | Type | Notes |
| --- | --- | --- |
| `request_id` | string | 요청 추적용 ID. 비어 있으면 안 됩니다. |
| `conversation_id` | string | 대화 세션 ID. |
| `pet_id` | integer | 1 이상의 반려동물 ID. |
| `user_input` | string | 현재 사용자 발화. |
| `locale` | string | 기본값 `ko-KR`. |
| `timezone` | string | 기본값 `Asia/Seoul`. |
| `timestamp` | date-time string | 클라이언트 또는 서버 요청 시각. |

선택 필드:

| Field | Type | Notes |
| --- | --- | --- |
| `conversation_history` | array | `role=user|assistant|system`, `content`를 가진 이전 대화 목록. |
| `user_location` | object/null | `lat`, `lng`, `permission=granted|denied|prompt|unknown`. |

Example:

```json
{
  "request_id": "req_20260717_0001",
  "conversation_id": "conv_demo_001",
  "pet_id": 1,
  "user_input": "내 고양이가 오늘 기침을 해요.",
  "conversation_history": [
    {"role": "user", "content": "내 이름은 장건호야"},
    {"role": "assistant", "content": "반갑습니다, 장건호님."}
  ],
  "locale": "ko-KR",
  "timezone": "Asia/Seoul",
  "timestamp": "2026-07-17T09:00:00+09:00",
  "user_location": null
}
```

### Graph Response Contract

`GraphResponse`는 다음 필드를 항상 반환합니다.

| Field | Type | Notes |
| --- | --- | --- |
| `response_id` | string | `req_` prefix가 있으면 `res_`로 바꾼 ID. |
| `conversation_id` | string | 요청의 conversation id. |
| `route` | string | `chat`, `answer_guard`, `handoff`, `emergency`, `end` 등 graph route. |
| `risk_level` | string | `emergency`, `urgent`, `non_emergency`, `unknown`. |
| `assistant_message` | string | 사용자에게 보여줄 최종 메시지. |
| `needs_user_response` | boolean | follow-up 질문이 필요한지 여부. |
| `follow_up_question` | object/null | `question_id`, `text`. |
| `handoff` | object | `type`, `summary`, `summary_json`, `email_draft`. |
| `emergency` | object | `is_emergency`, `triggered_rules`. |

Conversation/profile continuity questions should route as `social_chat` and finish at `chat`. They should not call Cornell RAG, should not include Cornell citations, and should not run answer-guard medical boilerplate.

## 9. Internal Official-Source RAG Boundary

The Assessment Graph does not expose a browser/mobile RAG endpoint. Official-source retrieval is an internal backend dependency behind `petcare_agent.rag.adapter.RAGAdapter`.

Current native provider wrapper:

```python
from petcare_agent.rag.cornell import CornellRAGAdapter
```

The adapter maps the vendored Cornell RAG retriever into the graph's `RetrievedChunk` contract. It accepts only `species=dog` or `species=cat`, preserves Cornell title/URL/chunk metadata, and does not accept pet profile, daily-entry, diagnosis, or other personal-record payloads.

The Assessment Graph uses `CornellRAGAdapter` directly in-process by default; no RAG server or service token is required for normal graph execution.

## 10. Optional Local Cornell RAG API

`petcare_rag.api` provides a small FastAPI boundary for backend-only diagnostics or integration smoke tests. It is not required by the Assessment Graph path.

Run locally:

```powershell
python tools/run_cornell_rag_api.py --host 127.0.0.1 --port 8001
```

### Health

```http
GET /health
```

Response:

```json
{"status": "ok"}
```

### Readiness

```http
GET /ready
```

The readiness check reports whether these are configured:

- `OPENAI_API_KEY`
- `PETCARE_RAG_SERVICE_TOKEN`
- local Chroma database path
- compatible collection
- expected `732` chunks

If not ready, the endpoint returns HTTP `503` with `status="not_ready"` and per-check booleans.

### RAG Answer

```http
POST /v1/rag/answer
X-PetCare-Token: <PETCARE_RAG_SERVICE_TOKEN>
Content-Type: application/json
```

Request body:

```json
{
  "question": "My dog ate chocolate. What should I watch for?",
  "species": "dog",
  "top_k": 5
}
```

Response body:

```json
{
  "question": "My dog ate chocolate. What should I watch for?",
  "species": "dog",
  "answer": "...",
  "insufficient_evidence": false,
  "citations": [
    {
      "number": 1,
      "title": "...",
      "section_path": ["..."],
      "url": "https://...",
      "chunk_id": "..."
    }
  ],
  "disclaimer": "..."
}
```

Safety boundary: the RAG API accepts only `question`, `species`, and `top_k`. Do not send pet profiles, daily entries, diagnoses, uploaded document text, owner notes, or other personal records to this endpoint.