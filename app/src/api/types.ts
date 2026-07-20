export type RiskLevel = 'normal' | 'observe' | 'consult' | 'emergency';

// ---------- auth ----------
export interface AuthUser {
  id: number;
  email: string;
}

export interface AuthResponse {
  token: string;
  user: AuthUser;
}

export interface Pet {
  id: number;
  name: string;
  species: string;
  breed: string;
  birth_date: string | null;
  sex: string;
  is_neutered: boolean;
  weight_kg: number | null;
  size_class: string;
  diseases: string;
  medications: string;
  supplement: string;
  allergies: string;
  updated_at: string; // 프로필 수정 일시
  age_label: string;
}

export interface PetInput {
  name: string;
  species: string;
  breed: string;
  birth_date: string | null;
  sex: string;
  is_neutered: boolean;
  weight_kg: number | null;
  size_class: string;
  diseases: string;
  medications: string;
  supplement: string;
  allergies: string;
}

// 일기장(daily_entries) — 모두 텍스트 상태값
export interface RecordFields {
  food: string; // 식사 상태
  water: string; // 음수 상태
  activity: string; // 활동 상태
  symptom: string; // 증상
  stool: string; // 배변 및 설사 상태
  vomit: string; // 구토 상태
  notes: string; // 기타사항
}

export interface DailyRecord extends RecordFields {
  pet_id: number;
  record_date: string; // "YYYY-MM-DD" — (pet_id, record_date) 가 PK
  raw_text: string;
  created_at: string;
}

export interface ExtractedItem {
  category: string;
  value: string;
  field: string;
}

export interface DiaryExtractResponse {
  items: ExtractedItem[];
  fields: RecordFields;
  source: string;
}

export interface DiagnosisFields {
  date: string | null; // 발급일/진료일
  hospital: string;
  diagnosis: string;
  content: string; // 진단 내용 및 기타사항
}

export interface Diagnosis extends DiagnosisFields {
  id: number;
  pet_id: number;
  original_file_ref: string;
  created_at: string;
}

export interface DiagnosisExtractResponse {
  fields: DiagnosisFields;
  original_file_ref: string;
  items_read: number;
  source: string;
}

export interface ChatMessage {
  role: 'user' | 'assistant';
  content: string;
}

export interface TrendItem {
  metric: string;
  change_pct: number | null;
  note: string;
}

export interface AgentAction {
  type: 'generate_summary' | 'save_summary_pdf' | 'send_email' | 'save_record';
  label: string;
  payload: Record<string, unknown>;
}

export interface RagCitation {
  title: string;
  source: string;
  snippet: string;
}

/**
 * AI 가 실시간 검색으로 찾아 적합도를 매긴 병원.
 *
 * 아래 `Hospital`(서버 DB 의 시드 병원)과는 다른 타입이다. 이쪽은 DB 에 없는
 * 검색 결과라 id·distance_km 이 없고, 대신 score/matched_reasons(왜 이 병원인지)와
 * verification_required(전화로 확인할 것)를 갖는다.
 * `availability` 가 기본 "전화 확인 필요" 인 이유: 검색 결과만으로 지금 문이 열려
 * 있는지 확정할 수 없다.
 *
 * mock 모드에서는 비어 있으므로 화면은 항상 빈 목록을 견뎌야 한다.
 */
export interface HospitalSuggestion {
  name: string;
  phone: string | null;
  address: string | null;
  /**
   * 병원 페이지에서 찾은 이메일 — 응급 이메일 초안의 수신 주소로 그대로 보낸다.
   * 웹 검색으로 이메일이 나오는 경우는 드물어 대부분 null 이다(정상 상황).
   * null 이면 초안은 만들어지되 수신 주소가 비므로 사용자가 직접 입력해야 한다.
   */
  email: string | null;
  source_url: string;
  score: number;
  suitability: string; // recommended | possible | low_information
  matched_reasons: string[];
  verification_required: string[];
  emergency_mentioned: boolean;
  open_24h_mentioned: boolean;
  availability: string;
}

/** 응급 이메일 초안의 수신 병원. DB 병원이면 hospitalId, AI 검색 병원이면 나머지. */
export interface EmailHospitalTarget {
  hospitalId?: number | null;
  name?: string | null;
  email?: string | null;
  phone?: string | null;
}

export interface AICheckResponse {
  reply: string;
  risk_level: RiskLevel;
  risk_label: string;
  trend_summary: string;
  trends: TrendItem[];
  reasons: string[];
  evidence: string;
  followup_question: string | null;
  /** AI 가 아직 되묻는 중 — 판정이 끝나지 않았다는 뜻 */
  awaiting_more_info?: boolean;
  /** 이번 턴이 새 판정인가 — false 면 앞선 판정에 대한 설명이라 카드를 다시 그리지 않는다 */
  assessment_turn?: boolean;
  can_generate_summary: boolean;
  show_hospitals: boolean;
  transit_guidance: string[];
  actions?: AgentAction[];
  citations?: RagCitation[];
  /** AI 가 찾은 병원. 비어 있으면 화면은 기존 /api/hospitals 목록으로 대체한다. */
  hospitals?: HospitalSuggestion[];
  source: string;
  session_id: number | null;
}

/** 지난 대화 저장용 — assistant 턴의 meta 에 결과 카드 정보가 담긴다 */
export interface StoredChatMessage {
  role: 'user' | 'assistant';
  content: string;
  meta?: {
    risk_level: RiskLevel;
    risk_label: string;
    trend_summary: string;
    reasons: string[];
    evidence: string;
    followup_question: string | null;
    can_generate_summary: boolean;
    show_hospitals: boolean;
    transit_guidance: string[];
    // AI(Agent) 연결 시에만 채워진다. mock 모드에서는 빈 목록.
    citations?: RagCitation[];
    // 지난 대화를 다시 열었을 때 "어느 병원에 연락하라고 했는지"가 남아 있어야 한다.
    hospitals?: HospitalSuggestion[];
  } | null;
}

export interface AISessionSummary {
  id: number;
  pet_id: number;
  title: string;
  last_risk_level: string;
  message_count: number;
  updated_at: string;
}

export interface AISessionDetail {
  id: number;
  pet_id: number;
  title: string;
  last_risk_level: string;
  messages: StoredChatMessage[];
  created_at: string;
  updated_at: string;
}

export interface Hospital {
  id: number;
  name: string;
  phone: string;
  email: string;
  distance_km: number | null;
  status: string;
  features: string;
  is_emergency: boolean;
  open_24h: boolean;
}

// 병원 전달용 상태 요약 — 문서 4섹션 구조
export interface SummaryContent {
  // 1. 문서 정보
  title: string;
  data_period: string;
  // 2. 반려동물 정보
  pet_name: string;
  species: string;
  breed: string;
  sex_neuter: string;
  age_label: string;
  weight: string;
  medications: string;
  allergies: string;
  // 3. 상태
  risk_label: string;
  risk_signs: string[];
  // 4. 주호소 및 주요 변화
  chief_complaint: string;
  major_changes: string;
  progress: string;
  owner_note?: string;
}

export interface Summary {
  id: number;
  pet_id: number;
  risk_level: string;
  content: SummaryContent;
  created_at: string;
}

export interface EmergencyEmail {
  id: number;
  pet_id: number;
  hospital_id: number | null;
  /**
   * 수신 주소. 병원 이메일을 못 구한 초안은 null 이다 — 404 로 막지 않고 초안을
   * 먼저 만들기 때문이다. 이 경우 화면에서 사용자에게 주소를 입력받아야 한다.
   */
  to_email: string | null;
  subject: string;
  body: string;
  content: SummaryContent; // 요약과 동일한 4섹션 구조
  attachments: { label: string; auto: boolean }[];
  status: 'draft' | 'sent';
  created_at: string;
  sent_at: string | null;
}

export interface Dashboard {
  pet: Pet;
  today_record: DailyRecord | null;
  recent_food_note: string;
  recent_activity_note: string;
  record_count_30d: number;
  last_diagnosis: Diagnosis | null;
}
