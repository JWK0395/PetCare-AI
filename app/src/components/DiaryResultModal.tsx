import React from 'react';
import { ScrollView, StyleSheet, Text, View } from 'react-native';
import { colors, radius, spacing } from '../theme';
import AppModal from './AppModal';
import { Button } from './ui';

const CATEGORY_COLORS: Record<string, string> = {
  식사: '#2F6BFF',
  음수: '#0EA5E9',
  활동: '#10B981',
  증상: '#EF4444',
  배변: '#D97706',
  구토: '#DB2777',
  기타사항: '#6B7280',
};

const CATEGORIES = [
  { key: 'food', label: '식사' },
  { key: 'water', label: '음수' },
  { key: 'activity', label: '활동' },
  { key: 'symptom', label: '증상' },
  { key: 'stool', label: '배변' },
  { key: 'vomit', label: '구토' },
  { key: 'notes', label: '기타사항' },
] as const;

/** DB(daily_entries) 정리 항목을 읽기 전용으로 보여준다. 상위 스크롤 영역 안에서 흐른다. */
export function DiaryFieldList({ fields }: { fields: Record<string, unknown> }) {
  return (
    <View>
      {CATEGORIES.map(({ key, label }) => {
        const raw = fields?.[key];
        const value = typeof raw === 'string' ? raw : '';
        return (
          <View key={key} style={styles.row}>
            <View
              style={[
                styles.chip,
                { backgroundColor: `${CATEGORY_COLORS[label]}18` },
              ]}>
              <Text style={[styles.chipText, { color: CATEGORY_COLORS[label] }]}>
                {label}
              </Text>
            </View>
            <Text style={[styles.value, !value && styles.empty]}>
              {value || '정보 없음'}
            </Text>
          </View>
        );
      })}
    </View>
  );
}

/**
 * AI 정리 결과 미리보기 팝업 (오늘 기록 · 지난 일기 공용).
 * 정리 항목 블럭은 스크롤되고, 저장/취소 버튼은 하단에 고정된다.
 */
export default function DiaryResultModal({
  visible,
  title,
  subtitle,
  foundCount,
  fields,
  onSave,
  saveLabel,
  saving,
  onClose,
}: {
  visible: boolean;
  title?: string;
  subtitle?: string;
  foundCount?: number;
  fields: Record<string, unknown>;
  onSave?: () => void;
  saveLabel?: string;
  saving?: boolean;
  onClose: () => void;
}) {
  return (
    <AppModal
      visible={visible}
      size="large"
      title={title || 'AI 정리 결과'}
      onClose={onClose}>
      <View style={styles.wrap}>
        <ScrollView
          style={styles.scroll}
          contentContainerStyle={styles.body}
          keyboardShouldPersistTaps="handled">
          {subtitle ? <Text style={styles.subtitle}>{subtitle}</Text> : null}
          {typeof foundCount === 'number' ? (
            <Text style={styles.count}>
              ✨ AI 일기에서 {foundCount}개 기록을 정리했어요
            </Text>
          ) : null}
          <DiaryFieldList fields={fields} />
        </ScrollView>
        <View style={styles.footer}>
          {onSave ? (
            <>
              <Button
                title={saveLabel || '저장'}
                onPress={onSave}
                loading={saving}
                style={styles.footerBtn}
              />
              <Button
                title="취소"
                variant="ghost"
                onPress={onClose}
                style={styles.footerBtn}
              />
            </>
          ) : (
            <Button title="닫기" variant="ghost" onPress={onClose} />
          )}
        </View>
      </View>
    </AppModal>
  );
}

const styles = StyleSheet.create({
  wrap: { flex: 1 },
  scroll: { flex: 1 },
  body: { paddingHorizontal: spacing(4), paddingBottom: spacing(4) },
  subtitle: {
    fontSize: 13,
    color: colors.textSecondary,
    marginBottom: spacing(2),
  },
  count: {
    fontSize: 14,
    fontWeight: '700',
    color: colors.primary,
    marginBottom: spacing(3),
  },
  row: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    paddingVertical: spacing(2.5),
    borderBottomWidth: StyleSheet.hairlineWidth,
    borderBottomColor: colors.border,
    gap: spacing(2.5),
  },
  chip: {
    paddingHorizontal: spacing(2.5),
    paddingVertical: spacing(1),
    borderRadius: radius.sm,
    minWidth: 56,
    alignItems: 'center',
  },
  chipText: { fontSize: 12, fontWeight: '700' },
  value: { flex: 1, fontSize: 14, color: colors.text, lineHeight: 20, paddingTop: 2 },
  empty: { color: colors.textTertiary },
  footer: {
    flexDirection: 'row',
    gap: spacing(2),
    paddingHorizontal: spacing(4),
    paddingTop: spacing(3),
    paddingBottom: spacing(2),
    borderTopWidth: StyleSheet.hairlineWidth,
    borderTopColor: colors.border,
    backgroundColor: colors.background,
  },
  footerBtn: { flex: 1 },
});
