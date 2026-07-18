/** 멍냥케어 블루화이트 v3 팔레트 */
export const colors = {
  primary: '#2F6BFF',
  primaryDark: '#1D4ED8',
  primarySoft: '#EAF1FF',
  background: '#F5F7FB',
  card: '#FFFFFF',
  text: '#111827',
  textSecondary: '#6B7280',
  textTertiary: '#9CA3AF',
  border: '#E5E7EB',
  danger: '#EF4444',
  dangerSoft: '#FEE2E2',
  dangerDark: '#B91C1C',
  warn: '#D97706',
  warnSoft: '#FEF3C7',
  success: '#10B981',
  successSoft: '#D1FAE5',
};

export const radius = {
  sm: 8,
  md: 12,
  lg: 16,
  xl: 20,
  full: 999,
};

export const spacing = (n: number) => n * 4;

export const shadow = {
  shadowColor: '#101828',
  shadowOpacity: 0.06,
  shadowRadius: 8,
  shadowOffset: { width: 0, height: 2 },
  elevation: 2,
};

export const riskColors: Record<
  string,
  { fg: string; bg: string; label: string }
> = {
  normal: { fg: colors.success, bg: colors.successSoft, label: '정상' },
  observe: { fg: colors.primary, bg: colors.primarySoft, label: '관찰' },
  consult: { fg: colors.warn, bg: colors.warnSoft, label: '신속 상담' },
  emergency: { fg: colors.danger, bg: colors.dangerSoft, label: '응급' },
};
