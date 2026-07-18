# PetCare AI Server (FastAPI + SQLite)

## 실행

```powershell
python -m venv .venv                                   # 최초 1회
.\.venv\Scripts\pip install -r requirements.txt        # 최초 1회
.\.venv\Scripts\python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

개발 중 자동 리로드: `--reload` 추가.
API 문서(Swagger): http://127.0.0.1:8000/docs

## 설정

`.env.example` 을 `.env` 로 복사해 수정한다.

| 변수 | 기본값 | 설명 |
| --- | --- | --- |
| `DATABASE_URL` | `sqlite:///<server>/data/petcare.db` (절대경로) | SQLite 경로 — 상대경로는 실행 위치에 따라 달라져 권장하지 않음 |
| `AGENT_MODE` | `mock` | `mock`(내장 규칙) 또는 `http`(외부 Agent) |
| `AGENT_BASE_URL` | `http://127.0.0.1:8100` | Agent 서비스 주소 |
| `AGENT_API_KEY` | (빈 값) | 설정 시 Bearer 헤더로 전달 |
| `SEED_DEMO_DATA` | `true` | DB가 비어 있으면 데모 데이터(콩이) 시드 |

데모 데이터를 초기화하려면 `data/petcare.db` 를 삭제하고 서버를 재시작한다.

## DB 구성 (DB_구성_및_API_설계 기반)

| 테이블 | 내용 |
| --- | --- |
| `users` | 사용자 계정 (이메일 + 비밀번호 해시 + 세션 토큰) |
| `pets` | PET DB — 이름, 견종, 생년월일, 성별, 중성화, 몸무게, 질병, 복용약, 영양제, 알레르기 |
| `daily_entries` | 일기장 DB — (pet_id, record_date) 복합 PK, 일기 원문 + 식사·음수·활동·증상·배변·구토·기타 (모두 텍스트 상태값) |
| `diagnoses` | 진단서 DB — 날짜, 병원, 진단명, 진단 내용, 원본 파일 참조 |
| `hospitals` | 응급 병원 정보 |
| `ai_sessions` | AI 상태 체크 대화 세션 (지난 대화 보기) |
| `summaries` | 병원 전달용 요약 (4섹션 content JSON) |
| `emergency_emails` | 응급 이메일 (초안 → 보호자 확인 후 전송) |

## API 요약

| 메서드 · 경로 | 설명 |
| --- | --- |
| `POST /api/auth/signup`, `POST /api/auth/login`, `POST /api/auth/logout`, `GET /api/auth/me` | 계정 가입/로그인/로그아웃/확인 |
| `GET/POST /api/pets`, `GET/PUT/DELETE /api/pets/{id}` | 반려동물 프로필 CRUD |
| `GET /api/pets/{id}/dashboard` | 홈 대시보드 (오늘 기록/최근 진단) |
| `GET /api/pets/{id}/records?days=30` | 일일 기록 목록 |
| `POST /api/pets/{id}/records/extract` | **일기 → AI 구조화** (저장 전 초안) |
| `POST /api/pets/{id}/records` | 확인된 기록 저장 (같은 날짜는 갱신) |
| `PUT/DELETE /api/pets/{id}/records/{record_date}` | 날짜별 기록 수정/삭제 |
| `GET /api/pets/{id}/diagnoses` | 진단서 목록 |
| `POST /api/pets/{id}/diagnoses/extract` | **진단서 파일 업로드 → AI 항목 추출** (multipart) |
| `POST /api/pets/{id}/diagnoses`, `PUT/DELETE /api/diagnoses/{id}` | 진단서 확정 저장/수정/삭제 |
| `POST /api/pets/{id}/ai-check` | **AI 상태 체크** (멀티턴 messages → 위험도/근거/추가질문) |
| `GET /api/pets/{id}/ai-sessions`, `GET/DELETE /api/ai-sessions/{id}` | 지난 대화 목록/상세/삭제 |
| `POST/GET /api/pets/{id}/summaries`, `GET /api/summaries/{id}[/pdf]` | **병원 전달용 요약** 생성/목록/조회/PDF |
| `GET /api/hospitals?emergency=true` | 응급 병원 목록 |
| `POST /api/pets/{id}/emergency-emails`, `GET /api/emergency-emails/{id}` | 응급 이메일 초안/조회 |
| `POST /api/emergency-emails/{id}/send` | 보호자 확인 후 전송 처리(로컬 기록) |

AI 관련 엔드포인트는 모두 `services/agent_client.py` 를 통해 Agent 로 위임된다
(`AGENT_MODE=mock` 이면 내장 규칙). Agent HTTP 계약은 [../ai/README.md](../ai/README.md).

## 구조

```
server/
├── app/
│   ├── main.py          # FastAPI 앱, 라우터 등록, 시드
│   ├── config.py        # 환경 설정 (.env)
│   ├── database.py      # SQLAlchemy 엔진/세션
│   ├── models.py        # 테이블 정의
│   ├── schemas.py       # 요청/응답 Pydantic 스키마
│   ├── routers/         # auth, pets, records, diagnoses, ai_check, summaries, hospitals, emergency
│   └── services/
│       ├── agent_client.py  # ★ Agent 연결 (mock/http)
│       ├── context.py       # Agent payload 구성
│       ├── pdf.py           # 요약 PDF 렌더링
│       └── seed.py          # 데모 데이터
└── data/                # SQLite DB, 업로드 파일 (자동 생성)
```
