const DAY_NAMES = ['일', '월', '화', '수', '목', '금', '토'];

/** "7월 11일 금요일" */
export function koreanDate(d: Date = new Date()): string {
  return `${d.getMonth() + 1}월 ${d.getDate()}일 ${DAY_NAMES[d.getDay()]}요일`;
}

/** 오늘 날짜 "YYYY-MM-DD" — 기록(record_date) 비교/저장용 */
export function todayKey(d: Date = new Date()): string {
  const pad = (n: number) => String(n).padStart(2, '0');
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}`;
}

/**
 * 서버의 datetime 은 오프셋 없는 naive UTC ISO 문자열이다 (예: "2026-07-17T03:15:00").
 * JS 의 new Date() 는 오프셋이 없으면 로컬 시각으로 해석하므로 그대로 쓰면
 * 표시 시각이 UTC 오프셋만큼 밀린다 → UTC('Z')로 해석해 로컬 시각으로 변환한다.
 */
function parseServerDate(iso: string): Date {
  const hasOffset = /Z$|[+-]\d{2}:?\d{2}$/.test(iso);
  return new Date(iso.includes('T') && !hasOffset ? `${iso}Z` : iso);
}

/** "2021. 9. 14" — 순수 날짜(YYYY-MM-DD)와 datetime 문자열 모두 처리 */
export function dotDate(iso: string | null | undefined): string {
  if (!iso) {
    return '-';
  }
  if (iso.includes('T')) {
    const d = parseServerDate(iso);
    return `${d.getFullYear()}. ${d.getMonth() + 1}. ${d.getDate()}`;
  }
  const [y, m, d] = iso.slice(0, 10).split('-').map(Number);
  return `${y}. ${m}. ${d}`;
}

/** "2026.07.11 14:20" */
export function stampDateTime(iso: string): string {
  const d = parseServerDate(iso);
  const pad = (n: number) => String(n).padStart(2, '0');
  return `${d.getFullYear()}.${pad(d.getMonth() + 1)}.${pad(d.getDate())} ${pad(
    d.getHours(),
  )}:${pad(d.getMinutes())}`;
}

/** "7/11" */
export function shortDate(iso: string): string {
  const [, m, d] = iso.slice(0, 10).split('-').map(Number);
  return `${m}/${d}`;
}
