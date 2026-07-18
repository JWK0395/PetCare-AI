import React, { useCallback, useEffect, useState } from 'react';
import {
  ScrollView,
  StyleSheet,
  Text,
  TouchableOpacity,
  View,
} from 'react-native';
import { deleteDiagnosis, getDiagnoses } from '../api/client';
import type { Diagnosis } from '../api/types';
import { colors, radius, shadow, spacing } from '../theme';
import { dotDate } from '../utils/date';
import { useAlert } from './AlertProvider';
import AppModal from './AppModal';
import { DiagnosisFieldsView } from './DiagnosisDetailModal';
import { DeleteButton } from './ui';

/** 진단서 등록 — 이전에 등록한 진단서를 보고 삭제하는 팝업 */
export default function DiagnosisArchiveModal({
  visible,
  petId,
  onClose,
  onChanged,
}: {
  visible: boolean;
  petId: number;
  onClose: () => void;
  onChanged?: () => void;
}) {
  const showAlert = useAlert();
  const [diagnoses, setDiagnoses] = useState<Diagnosis[]>([]);
  const [selected, setSelected] = useState<Diagnosis | null>(null);

  const load = useCallback(async () => {
    try {
      setDiagnoses(await getDiagnoses(petId));
    } catch {}
  }, [petId]);

  useEffect(() => {
    if (visible) {
      load();
      setSelected(null);
    }
  }, [visible, load]);

  const confirmDelete = (d: Diagnosis) => {
    showAlert('진단서 삭제', `"${d.diagnosis || '진단서'}" 진단서를 삭제할까요?`, [
      { text: '취소', style: 'cancel' },
      {
        text: '삭제',
        style: 'destructive',
        onPress: async () => {
          try {
            await deleteDiagnosis(d.id);
            setSelected(null);
            await load();
            onChanged?.();
          } catch (e) {
            showAlert('오류', e instanceof Error ? e.message : '삭제 실패');
          }
        },
      },
    ]);
  };

  return (
    <AppModal
      visible={visible}
      size="large"
      title={selected ? '진단서 상세' : '이전 진단서'}
      onClose={onClose}
      headerLeft={
        selected ? (
          <TouchableOpacity
            onPress={() => setSelected(null)}
            hitSlop={{ top: 8, bottom: 8, left: 8, right: 8 }}>
            <Text style={styles.backLink}>‹ 목록</Text>
          </TouchableOpacity>
        ) : undefined
      }>
      <ScrollView style={styles.scroll} contentContainerStyle={styles.body}>
        {selected ? (
          <DiagnosisFieldsView fields={selected} />
        ) : diagnoses.length === 0 ? (
          <Text style={styles.empty}>등록된 진단서가 없어요.</Text>
        ) : (
          diagnoses.map(d => (
            <View key={d.id} style={styles.listCard}>
              <TouchableOpacity
                style={styles.listMain}
                onPress={() => setSelected(d)}
                activeOpacity={0.7}>
                <Text style={styles.listTitle}>
                  {d.diagnosis || '진단서'}
                </Text>
                <Text style={styles.listMeta}>
                  {[d.hospital, dotDate(d.date)]
                    .filter(Boolean)
                    .join(' · ')}
                </Text>
              </TouchableOpacity>
              <DeleteButton onPress={() => confirmDelete(d)} />
            </View>
          ))
        )}
      </ScrollView>
    </AppModal>
  );
}

const styles = StyleSheet.create({
  scroll: { flex: 1 },
  body: { paddingHorizontal: spacing(4), paddingBottom: spacing(5) },
  backLink: { color: colors.primary, fontSize: 13, fontWeight: '700' },
  empty: {
    textAlign: 'center',
    color: colors.textTertiary,
    paddingVertical: spacing(10),
    fontSize: 13,
  },
  listCard: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: colors.card,
    borderRadius: radius.lg,
    padding: spacing(4),
    marginBottom: spacing(2.5),
    gap: spacing(2),
    ...shadow,
  },
  listMain: { flex: 1 },
  listTitle: { fontSize: 15, fontWeight: '700', color: colors.text },
  listMeta: { fontSize: 12, color: colors.textSecondary, marginTop: 2 },
  detailCard: {
    backgroundColor: colors.card,
    borderRadius: radius.lg,
    padding: spacing(4),
    ...shadow,
  },
  fieldRow: {
    flexDirection: 'row',
    paddingVertical: spacing(2.5),
    borderBottomWidth: StyleSheet.hairlineWidth,
    borderBottomColor: colors.border,
    gap: spacing(3),
  },
  fieldLabel: {
    width: 76,
    fontSize: 13,
    color: colors.textSecondary,
    fontWeight: '600',
  },
  fieldValue: { flex: 1, fontSize: 14, color: colors.text, fontWeight: '500' },
  unavailable: { color: colors.textTertiary, fontWeight: '400' },
  contentRow: { alignItems: 'flex-start' },
  contentScroll: {
    flex: 1,
    maxHeight: 150,
    borderWidth: 1,
    borderColor: colors.border,
    borderRadius: radius.sm,
    paddingHorizontal: spacing(2.5),
    paddingVertical: spacing(2),
    backgroundColor: colors.background,
  },
  deleteBtn: { marginTop: spacing(3) },
});
