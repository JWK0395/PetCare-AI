/** 응급 입력에서 이메일 제목용 증상 요약을 만든다 */
export function deriveSymptomSummary(text: string): string {
  const labels: string[] = [];
  if (/호흡|가쁘|헐떡|숨/.test(text)) {
    labels.push('호흡곤란');
  }
  if (/파래|파랗|청색|보라/.test(text)) {
    labels.push('청색증 의심');
  }
  if (/경련|발작/.test(text)) {
    labels.push('경련');
  }
  if (/중독|주워 먹|쥐약|부동액|초콜릿|포도|양파|자일리톨/.test(text)) {
    labels.push('중독 의심');
  }
  if (/피를 토|토혈|하혈/.test(text)) {
    labels.push('출혈 증상');
  }
  return labels.join(' · ') || '응급 증상';
}
