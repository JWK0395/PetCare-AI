import { deriveSymptomSummary } from '../src/utils/symptoms';
import { dotDate, shortDate, stampDateTime } from '../src/utils/date';

describe('date utils', () => {
  it('formats dot date', () => {
    expect(dotDate('2021-09-14')).toBe('2021. 9. 14');
    expect(dotDate(null)).toBe('-');
  });

  it('formats short date', () => {
    expect(shortDate('2026-07-11')).toBe('7/11');
  });

  it('parses server naive-UTC datetimes as UTC (회귀: 표시 시각 오프셋 밀림)', () => {
    // 서버는 오프셋 없는 UTC 를 보낸다 — 로컬 시각으로 변환되어야 한다
    const utc = new Date(Date.UTC(2026, 6, 17, 3, 15));
    const pad = (n: number) => String(n).padStart(2, '0');
    const expected = `${utc.getFullYear()}.${pad(utc.getMonth() + 1)}.${pad(
      utc.getDate(),
    )} ${pad(utc.getHours())}:${pad(utc.getMinutes())}`;
    expect(stampDateTime('2026-07-17T03:15:00')).toBe(expected);
    // 명시적 오프셋이 있으면 그대로 존중한다
    expect(stampDateTime('2026-07-17T03:15:00Z')).toBe(expected);
    // datetime 을 받은 dotDate 도 로컬 날짜 기준으로 변환한다
    expect(dotDate('2026-07-17T03:15:00')).toBe(
      `${utc.getFullYear()}. ${utc.getMonth() + 1}. ${utc.getDate()}`,
    );
  });
});

describe('deriveSymptomSummary', () => {
  it('detects respiratory + cyanosis', () => {
    expect(deriveSymptomSummary('숨을 가쁘게 몰아쉬고 혀 색이 파래요')).toBe(
      '호흡곤란 · 청색증 의심',
    );
  });

  it('detects poisoning', () => {
    expect(deriveSymptomSummary('산책 중에 뭔가를 주워 먹은 것 같아요')).toBe(
      '중독 의심',
    );
  });

  it('falls back to generic label', () => {
    expect(deriveSymptomSummary('상태가 이상해요')).toBe('응급 증상');
  });
});
