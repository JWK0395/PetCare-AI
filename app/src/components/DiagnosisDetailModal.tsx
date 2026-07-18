import React from 'react';
import { ScrollView, StyleSheet, Text, View } from 'react-native';
import { colors, radius, shadow, spacing } from '../theme';
import { dotDate } from '../utils/date';
import AppModal from './AppModal';
import { Button } from './ui';

export interface DiagnosisFields {
  date: string | null;
  hospital: string;
  diagnosis: string;
  content: string;
  original_file_ref?: string;
}

/** 진단서 필드 목록 — 상위 스크롤 영역 안에서 그대로 흐른다(내부 스크롤 없음). */
export function DiagnosisFieldsView({ fields }: { fields: DiagnosisFields }) {
  const rows: [string, string][] = [
    ['날짜', fields.date ? dotDate(fields.date) : ''],
    ['병원', fields.hospital],
    ['진단명', fields.diagnosis],
    ['진단 내용', fields.content],
  ];
  if (fields.original_file_ref) {
    rows.push(['원본 파일', fields.original_file_ref]);
  }
  return (
    <View style={styles.card}>
      {rows.map(([label, value]) => (
        <View key={label} style={styles.row}>
          <Text style={styles.label}>{label}</Text>
          <Text style={[styles.value, !value && styles.unavailable]}>
            {value || '확인 불가'}
          </Text>
        </View>
      ))}
    </View>
  );
}

/**
 * 진단서 상세 팝업 (업로드 AI 정리 결과 확인용).
 * 필드 블럭은 스크롤되고, 저장/취소 버튼은 하단에 고정된다.
 */
export default function DiagnosisDetailModal({
  visible,
  title,
  fields,
  onSave,
  saveLabel,
  saving,
  onClose,
}: {
  visible: boolean;
  title?: string;
  fields: DiagnosisFields;
  onSave?: () => void;
  saveLabel?: string;
  saving?: boolean;
  onClose: () => void;
}) {
  return (
    <AppModal
      visible={visible}
      size="large"
      title={title || '진단서 상세'}
      onClose={onClose}>
      <View style={styles.wrap}>
        <ScrollView style={styles.scroll} contentContainerStyle={styles.body}>
          <DiagnosisFieldsView fields={fields} />
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
  card: {
    backgroundColor: colors.card,
    borderRadius: radius.lg,
    padding: spacing(4),
    ...shadow,
  },
  row: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    paddingVertical: spacing(2.5),
    borderBottomWidth: StyleSheet.hairlineWidth,
    borderBottomColor: colors.border,
    gap: spacing(3),
  },
  label: {
    width: 76,
    fontSize: 13,
    color: colors.textSecondary,
    fontWeight: '600',
  },
  value: { flex: 1, fontSize: 14, color: colors.text, fontWeight: '500', lineHeight: 20 },
  unavailable: { color: colors.textTertiary, fontWeight: '400' },
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
