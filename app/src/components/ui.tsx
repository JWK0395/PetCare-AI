import React from 'react';
import {
  ActivityIndicator,
  StyleSheet,
  Text,
  TouchableOpacity,
  View,
  ViewStyle,
} from 'react-native';
import { colors, radius, shadow, spacing } from '../theme';

/** 흰색 라운드 카드 */
export function Card({
  children,
  style,
}: {
  children: React.ReactNode;
  style?: ViewStyle | ViewStyle[];
}) {
  return <View style={[styles.card, style]}>{children}</View>;
}

/** 섹션 제목 (작은 회색 라벨) */
export function SectionLabel({ children }: { children: React.ReactNode }) {
  return <Text style={styles.sectionLabel}>{children}</Text>;
}

/** 주요/보조/위험 버튼 */
export function Button({
  title,
  onPress,
  variant = 'primary',
  disabled,
  loading,
  style,
}: {
  title: string;
  onPress: () => void;
  variant?: 'primary' | 'secondary' | 'danger' | 'ghost';
  disabled?: boolean;
  loading?: boolean;
  style?: ViewStyle | ViewStyle[];
}) {
  const bg =
    variant === 'primary'
      ? colors.primary
      : variant === 'danger'
      ? colors.danger
      : variant === 'secondary'
      ? colors.primarySoft
      : 'transparent';
  const fg =
    variant === 'primary' || variant === 'danger'
      ? '#FFFFFF'
      : colors.primary;
  return (
    <TouchableOpacity
      style={[
        styles.button,
        { backgroundColor: bg },
        variant === 'ghost' && styles.ghostBorder,
        (disabled || loading) && styles.dimmed,
        style,
      ]}
      onPress={onPress}
      disabled={disabled || loading}
      activeOpacity={0.8}>
      {loading ? (
        <ActivityIndicator color={fg} />
      ) : (
        <Text style={[styles.buttonText, { color: fg }]}>{title}</Text>
      )}
    </TouchableOpacity>
  );
}

/** 상태 뱃지 (신속 상담 / 응급 등) */
export function Badge({
  label,
  fg,
  bg,
}: {
  label: string;
  fg: string;
  bg: string;
}) {
  return (
    <View style={[styles.badge, { backgroundColor: bg }]}>
      <Text style={[styles.badgeText, { color: fg }]}>{label}</Text>
    </View>
  );
}

/** 직관적인 삭제 아이콘 버튼 (빨간 원 안에 ✕) */
export function DeleteButton({
  onPress,
  style,
}: {
  onPress: () => void;
  style?: ViewStyle | ViewStyle[];
}) {
  return (
    <TouchableOpacity
      style={[styles.deleteButton, style]}
      onPress={onPress}
      hitSlop={{ top: 10, bottom: 10, left: 10, right: 10 }}
      accessibilityLabel="삭제">
      <Text style={styles.deleteButtonText}>✕</Text>
    </TouchableOpacity>
  );
}

/** 라벨-값 그리드 셀 (홈 프로필 카드) */
export function InfoCell({ label, value }: { label: string; value: string }) {
  return (
    <View style={styles.infoCell}>
      <Text style={styles.infoLabel}>{label}</Text>
      <Text style={styles.infoValue}>{value || '-'}</Text>
    </View>
  );
}

/** 화면 전체 로딩 */
export function FullLoading({ message }: { message?: string }) {
  return (
    <View style={styles.loadingWrap}>
      <ActivityIndicator size="large" color={colors.primary} />
      {message ? <Text style={styles.loadingText}>{message}</Text> : null}
    </View>
  );
}

const styles = StyleSheet.create({
  card: {
    backgroundColor: colors.card,
    borderRadius: radius.lg,
    padding: spacing(4),
    ...shadow,
  },
  sectionLabel: {
    fontSize: 13,
    fontWeight: '600',
    color: colors.textSecondary,
    marginBottom: spacing(2),
  },
  button: {
    borderRadius: radius.md,
    paddingVertical: spacing(3.5),
    alignItems: 'center',
    justifyContent: 'center',
  },
  ghostBorder: {
    borderWidth: 1,
    borderColor: colors.primary,
  },
  dimmed: { opacity: 0.5 },
  buttonText: {
    fontSize: 16,
    fontWeight: '700',
  },
  badge: {
    alignSelf: 'flex-start',
    paddingHorizontal: spacing(2.5),
    paddingVertical: spacing(1),
    borderRadius: radius.full,
  },
  badgeText: {
    fontSize: 12,
    fontWeight: '700',
  },
  deleteButton: {
    width: 26,
    height: 26,
    borderRadius: 13,
    backgroundColor: colors.danger,
    alignItems: 'center',
    justifyContent: 'center',
  },
  deleteButtonText: {
    color: '#FFFFFF',
    fontSize: 14,
    fontWeight: '800',
    lineHeight: 16,
  },
  infoCell: {
    width: '50%',
    paddingVertical: spacing(2),
  },
  infoLabel: {
    fontSize: 12,
    color: colors.textTertiary,
    marginBottom: 2,
  },
  infoValue: {
    fontSize: 15,
    fontWeight: '600',
    color: colors.text,
  },
  loadingWrap: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: colors.background,
  },
  loadingText: {
    marginTop: spacing(3),
    color: colors.textSecondary,
  },
});
