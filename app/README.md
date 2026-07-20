# PetCare AI App (React Native)

멍냥케어 모바일 앱 — 블루화이트 v3 디자인 기반 7개 화면. Android 테스트 기준.

## 실행

서버(`../server`)를 먼저 8000 포트로 실행한 뒤:

```powershell
npm install                   # 최초 1회
npx react-native run-android  # 에뮬레이터 또는 USB 기기
```

Metro 만 따로 띄우려면 `npm start`.

### 서버 주소

[src/api/client.ts](src/api/client.ts) 의 `API_BASE_URL`:

- Android 에뮬레이터: `http://10.0.2.2:8000` (기본값, 호스트 PC)
- 실기기(USB): `adb reverse tcp:8000 tcp:8000` 실행 후 그대로 사용
- 실기기(Wi-Fi): PC 의 LAN IP 로 변경 (예: `http://192.168.0.10:8000`)

## 화면 ↔ 코드

| 디자인 화면 | 파일 |
| --- | --- |
| 01 홈 대시보드 | `src/screens/HomeScreen.tsx` |
| 02 건강기록 — 일기장 | `src/screens/RecordScreen.tsx` |
| 03 AI 상태 체크 — 일반 | `src/screens/AICheckScreen.tsx` (결과 카드) |
| 04 AI 상태 체크 — 응급 | `src/screens/AICheckScreen.tsx` (응급 카드) |
| 05 진단서 등록 (진료 탭) | `src/screens/DiagnosisUploadScreen.tsx` |
| 06 병원 전달용 요약 | `src/screens/SummaryScreen.tsx` |
| 07 응급 이메일 전송 | `src/screens/EmergencyEmailScreen.tsx` |
| (프로필 등록 — 최초 실행) | `src/screens/PetProfileScreen.tsx` |

추가 팝업 (요청 사항):

| 팝업 | 파일 | 여는 곳 |
| --- | --- | --- |
| 지난 일기 달력 (보기/수정) | `src/components/DiaryCalendarModal.tsx` | 기록 탭 우상단 "📅 지난 일기" |
| 이전 진단서 보관함 | `src/components/DiagnosisArchiveModal.tsx` | 진료 탭 우상단 "📂 이전 진단서" |
| AI 지난 대화 | `src/components/ChatHistoryModal.tsx` | AI 체크 우상단 "지난 대화" |

## 구조

```
src/
├── api/          # 서버 API 클라이언트 + 타입
├── components/   # 공용 UI (Card, Button, Badge ...)
├── navigation/   # 하단 탭(홈/기록/AI 체크/진료) + 스택
├── screens/      # 화면
├── state/        # PetContext (현재 반려동물)
├── theme.ts      # 블루화이트 팔레트
└── utils/        # 날짜/증상 유틸
```

## 검사

```powershell
npx tsc --noEmit   # 타입 체크
npm test           # Jest
npm run lint       # ESLint
```
