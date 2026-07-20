# PetCare AI (멍냥케어)

보호자의 일상 기록과 진단서를 바탕으로 반려동물의 이상 신호를 감지하고,
**다음 행동**과 **병원 전달용 요약**을 제공하는 건강관리 AI Agent 서비스.

## 주요 기능

- **이메일/비밀번호 로그인** — 사용자별 반려동물 데이터 분리
- **홈 대시보드** — 프로필(기본/건강 정보), 다중 반려동물 전환
- **기록** — 자연어 일기 → AI 정리(식사·음수·활동·증상·배변·구토·기타) → 팝업 확인 후 저장, 지난 일기 달력
- **AI 체크** — 채팅으로 상태 확인, 위험도 3단계(normal/consult/emergency), 응급 시 24시 병원 안내 + 상태 문서 이메일
- **진료** — 진단서 PDF 업로드 → AI 항목 추출 → 확인 후 저장, 이전 진단서 보관함
- **병원 전달용 요약** — 4섹션 문서(문서 정보/반려동물 정보/상태/주호소·변화) + PDF

## 구조

| 폴더 | 역할 | 기술 |
| --- | --- | --- |
| `app/` | 모바일 앱 (Android) | React Native 0.86 · React Navigation 7 |
| `server/` | API 서버 + DB (포트 8000) | FastAPI · SQLAlchemy 2 · SQLite |
| `ai/` | AI Agent 서비스 (포트 8100, 선택) — RAG·LangGraph 파이프라인(`petcare_ai`) 포함 | FastAPI · LangGraph · FAISS · OpenAI · Tavily |
| `docs/` | 기술 문서 | — |

## 요구사항

- Python 3.11+
- Node.js 22.11+ / npm
- Android Studio (SDK + 에뮬레이터 또는 실기기)
- JDK 17+ (Android 빌드)

## 빠른 시작

### 1) 서버 (포트 8000)

```powershell
cd server
python -m venv .venv                                   # 최초 1회
.\.venv\Scripts\pip install -r requirements.txt        # 최초 1회
copy .env.example .env                                 # 선택 (기본값으로도 동작)
.\.venv\Scripts\python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

- 첫 실행 시 `server/data/petcare.db` 생성 + 데모 데이터 시드
- **데모 계정: `demo@petcare.ai` / `demo1234`** (`.env` 의 `DEMO_USER_*` 로 변경 가능)
- API 문서: http://127.0.0.1:8000/docs

### 2) 앱 (Android)

```powershell
cd app
npm install                    # 최초 1회
npm run bundle:android         # JS 번들 생성 (오프라인 번들 방식)
npm run apk:debug              # APK 빌드
adb install -r android\app\build\outputs\apk\debug\app-debug.apk
```

- 에뮬레이터는 자동으로 `http://10.0.2.2:8000`(호스트 PC 서버)에 접속
- 개발용 핫리로드(`npx react-native run-android`)도 가능 — 환경별 주의사항은 [dev-readme/local-development.md](dev-readme/local-development.md)
- 로그인 화면에서 데모 계정으로 로그인하면 시드 데이터(콩이)가 보인다

### 3) AI Agent 연결 (선택)

기본값 `AGENT_MODE=mock` — 외부 AI 없이 서버 내장 규칙으로 전체 흐름이 동작한다.
실제 AI(RAG + LangGraph)를 붙이려면 `ai/` 서비스를 띄우고 `server/.env` 에서
`AGENT_MODE=http` 로 전환한다. HTTP 계약과 실행 방법은 [ai/README.md](ai/README.md),
파이프라인 상세는 `ai/petcare_ai/` 를 참고한다.