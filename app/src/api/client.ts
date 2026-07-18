import { Platform } from 'react-native';
import type {
  AICheckResponse,
  AISessionDetail,
  AISessionSummary,
  AuthResponse,
  AuthUser,
  ChatMessage,
  DailyRecord,
  Dashboard,
  Diagnosis,
  DiagnosisExtractResponse,
  DiagnosisFields,
  DiaryExtractResponse,
  EmergencyEmail,
  Hospital,
  Pet,
  PetInput,
  RecordFields,
  RiskLevel,
  Summary,
} from './types';

/**
 * 로컬 FastAPI 서버 주소.
 * - Android 에뮬레이터: 10.0.2.2 가 호스트 PC 를 가리킨다.
 * - 실기기 테스트: `adb reverse tcp:8000 tcp:8000` 실행 후 그대로 사용하거나,
 *   PC 의 LAN IP 로 바꿔주세요. (예: http://192.168.0.10:8000)
 */
export const API_BASE_URL =
  Platform.OS === 'android' ? 'http://10.0.2.2:8000' : 'http://127.0.0.1:8000';

// ---------- auth token ----------
// AuthContext 가 로그인/복원 시 설정한다. 모든 요청에 Bearer 헤더로 붙는다.
let authToken: string | null = null;
// 토큰 만료(401) 시 AuthContext 가 강제 로그아웃하도록 콜백을 등록한다.
let onUnauthorized: (() => void) | null = null;

export const setAuthToken = (token: string | null) => {
  authToken = token;
};
export const setOnUnauthorized = (handler: (() => void) | null) => {
  onUnauthorized = handler;
};

async function request<T>(path: string, options: RequestInit = {}): Promise<T> {
  // FormData 는 fetch 가 boundary 포함 Content-Type 을 직접 설정해야 한다
  const isForm = options.body instanceof FormData;
  const headers: Record<string, string> = {
    ...(isForm ? {} : { 'Content-Type': 'application/json' }),
    ...(authToken ? { Authorization: `Bearer ${authToken}` } : {}),
    ...((options.headers as Record<string, string>) || {}),
  };
  const res = await fetch(`${API_BASE_URL}${path}`, { ...options, headers });
  if (!res.ok) {
    // 로그인 상태에서 401 = 토큰 만료 → 로그인 화면으로 돌려보낸다
    if (res.status === 401 && authToken && !path.startsWith('/api/auth/')) {
      onUnauthorized?.();
    }
    let detail = `HTTP ${res.status}`;
    try {
      const body = await res.json();
      if (body?.detail) {
        detail =
          typeof body.detail === 'string'
            ? body.detail
            : JSON.stringify(body.detail);
      }
    } catch {}
    throw new Error(detail);
  }
  if (res.status === 204) {
    return undefined as T;
  }
  return res.json();
}

// ---------- auth ----------
export const signup = (email: string, password: string) =>
  request<AuthResponse>('/api/auth/signup', {
    method: 'POST',
    body: JSON.stringify({ email, password }),
  });
export const login = (email: string, password: string) =>
  request<AuthResponse>('/api/auth/login', {
    method: 'POST',
    body: JSON.stringify({ email, password }),
  });
export const logoutApi = () =>
  request<void>('/api/auth/logout', { method: 'POST' });
export const getMe = () => request<AuthUser>('/api/auth/me');

// ---------- pets ----------
export const getPets = () => request<Pet[]>('/api/pets');
export const createPet = (body: PetInput) =>
  request<Pet>('/api/pets', { method: 'POST', body: JSON.stringify(body) });
export const updatePet = (id: number, body: Partial<PetInput>) =>
  request<Pet>(`/api/pets/${id}`, { method: 'PUT', body: JSON.stringify(body) });
export const deletePet = (id: number) =>
  request<void>(`/api/pets/${id}`, { method: 'DELETE' });
export const getDashboard = (petId: number) =>
  request<Dashboard>(`/api/pets/${petId}/dashboard`);

// ---------- records ----------
export const getRecords = (petId: number, days = 30) =>
  request<DailyRecord[]>(`/api/pets/${petId}/records?days=${days}`);
export const extractDiary = (petId: number, text: string) =>
  request<DiaryExtractResponse>(`/api/pets/${petId}/records/extract`, {
    method: 'POST',
    body: JSON.stringify({ text }),
  });
// daily_entries 는 (pet_id, record_date) 가 PK 이므로 저장은 날짜 기준 upsert 다.
export const saveRecord = (
  petId: number,
  body: Partial<RecordFields> & { raw_text?: string; record_date?: string },
) =>
  request<DailyRecord>(`/api/pets/${petId}/records`, {
    method: 'POST',
    body: JSON.stringify(body),
  });

export const deleteRecord = (petId: number, recordDate: string) =>
  request<void>(`/api/pets/${petId}/records/${recordDate}`, { method: 'DELETE' });

// ---------- diagnoses ----------
export const getDiagnoses = (petId: number) =>
  request<Diagnosis[]>(`/api/pets/${petId}/diagnoses`);

export const extractDiagnosis = (
  petId: number,
  file: { uri: string; name: string; type: string },
) => {
  const form = new FormData();
  form.append('file', {
    uri: file.uri,
    name: file.name,
    type: file.type,
  } as unknown as Blob);
  return request<DiagnosisExtractResponse>(
    `/api/pets/${petId}/diagnoses/extract`,
    { method: 'POST', body: form },
  );
};

export const saveDiagnosis = (
  petId: number,
  body: Partial<DiagnosisFields> & { original_file_ref?: string },
) =>
  request<Diagnosis>(`/api/pets/${petId}/diagnoses`, {
    method: 'POST',
    body: JSON.stringify(body),
  });

export const deleteDiagnosis = (diagnosisId: number) =>
  request<void>(`/api/diagnoses/${diagnosisId}`, { method: 'DELETE' });

// ---------- AI check ----------
export const aiCheck = (
  petId: number,
  messages: ChatMessage[],
  sessionId: number | null = null,
) =>
  request<AICheckResponse>(`/api/pets/${petId}/ai-check`, {
    method: 'POST',
    body: JSON.stringify({ messages, session_id: sessionId }),
  });

export const getAISessions = (petId: number) =>
  request<AISessionSummary[]>(`/api/pets/${petId}/ai-sessions`);
export const getAISession = (sessionId: number) =>
  request<AISessionDetail>(`/api/ai-sessions/${sessionId}`);
export const deleteAISession = (sessionId: number) =>
  request<void>(`/api/ai-sessions/${sessionId}`, { method: 'DELETE' });

// ---------- hospitals ----------
export const getHospitals = (emergencyOnly = true) =>
  request<Hospital[]>(`/api/hospitals?emergency=${emergencyOnly}`);

// ---------- summaries ----------
export const createSummary = (
  petId: number,
  riskLevel: RiskLevel | null,
  extraNote = '',
) =>
  request<Summary>(`/api/pets/${petId}/summaries`, {
    method: 'POST',
    body: JSON.stringify({ risk_level: riskLevel, extra_note: extraNote }),
  });
export const getSummaries = (petId: number) =>
  request<Summary[]>(`/api/pets/${petId}/summaries`);
export const getSummary = (summaryId: number) =>
  request<Summary>(`/api/summaries/${summaryId}`);

// ---------- emergency email ----------
export const composeEmergencyEmail = (
  petId: number,
  hospitalId: number | null,
  symptomSummary: string,
) =>
  request<EmergencyEmail>(`/api/pets/${petId}/emergency-emails`, {
    method: 'POST',
    body: JSON.stringify({
      hospital_id: hospitalId,
      symptom_summary: symptomSummary,
    }),
  });
export const sendEmergencyEmail = (emailId: number) =>
  request<EmergencyEmail>(`/api/emergency-emails/${emailId}/send`, {
    method: 'POST',
  });
