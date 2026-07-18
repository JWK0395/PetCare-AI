import React from 'react';
import { ScrollView, StyleSheet, Text, TouchableOpacity, View } from 'react-native';
import { deletePet } from '../api/client';
import type { Pet } from '../api/types';
import { colors, radius, shadow, spacing } from '../theme';
import { useAlert } from './AlertProvider';
import AppModal from './AppModal';
import { DeleteButton } from './ui';

/** 반려동물 전환 + 새 등록 팝업 */
export default function PetSwitcherModal({
  visible,
  pets,
  selectedId,
  onSelect,
  onAddNew,
  onClose,
  onDeleted,
}: {
  visible: boolean;
  pets: Pet[];
  selectedId: number | null;
  onSelect: (id: number) => void;
  onAddNew: () => void;
  onClose: () => void;
  onDeleted: () => void;
}) {
  const showAlert = useAlert();
  const confirmDelete = (pet: Pet) => {
    if (pets.length <= 1) {
      showAlert('삭제 불가', '마지막 반려동물은 삭제할 수 없어요.');
      return;
    }
    showAlert(
      '반려동물 삭제',
      `"${pet.name}"과(와) 모든 기록·진단서·대화를 삭제할까요? 되돌릴 수 없어요.`,
      [
        { text: '취소', style: 'cancel' },
        {
          text: '삭제',
          style: 'destructive',
          onPress: async () => {
            try {
              await deletePet(pet.id);
              onDeleted();
            } catch (e) {
              showAlert('오류', e instanceof Error ? e.message : '삭제 실패');
            }
          },
        },
      ],
    );
  };

  return (
    <AppModal visible={visible} title="반려동물" onClose={onClose}>
      <ScrollView contentContainerStyle={styles.body}>
        {pets.map(pet => {
          const active = pet.id === selectedId;
          return (
            <View
              key={pet.id}
              style={[styles.row, active && styles.rowActive]}>
              <TouchableOpacity
                style={styles.rowMain}
                onPress={() => onSelect(pet.id)}
                activeOpacity={0.7}>
                <View style={[styles.avatar, active && styles.avatarActive]}>
                  <Text style={[styles.avatarText, active && styles.avatarTextActive]}>
                    {pet.name.slice(0, 1)}
                  </Text>
                </View>
                <View style={styles.info}>
                  <Text style={styles.name}>{pet.name}</Text>
                  <Text style={styles.meta} numberOfLines={1}>
                    {[pet.species, pet.breed].filter(Boolean).join(' · ')}
                  </Text>
                </View>
                {active ? <Text style={styles.check}>✓</Text> : null}
              </TouchableOpacity>
              <DeleteButton
                onPress={() => confirmDelete(pet)}
                style={styles.del}
              />
            </View>
          );
        })}

        <TouchableOpacity style={styles.addBtn} onPress={onAddNew} activeOpacity={0.8}>
          <Text style={styles.addPlus}>＋</Text>
          <Text style={styles.addText}>새 반려동물 등록</Text>
        </TouchableOpacity>
      </ScrollView>
    </AppModal>
  );
}

const styles = StyleSheet.create({
  body: { paddingHorizontal: spacing(4), paddingBottom: spacing(6) },
  row: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: colors.card,
    borderRadius: radius.lg,
    padding: spacing(3),
    marginBottom: spacing(2.5),
    gap: spacing(2),
    borderWidth: 1,
    borderColor: colors.card,
    ...shadow,
  },
  rowActive: { borderColor: colors.primary },
  rowMain: {
    flex: 1,
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing(3),
  },
  avatar: {
    width: 40,
    height: 40,
    borderRadius: 20,
    backgroundColor: colors.background,
    alignItems: 'center',
    justifyContent: 'center',
  },
  avatarActive: { backgroundColor: colors.primarySoft },
  avatarText: { fontSize: 16, fontWeight: '800', color: colors.textSecondary },
  avatarTextActive: { color: colors.primary },
  info: { flex: 1 },
  name: { fontSize: 15, fontWeight: '700', color: colors.text },
  meta: { fontSize: 12, color: colors.textSecondary, marginTop: 2 },
  check: { fontSize: 16, fontWeight: '800', color: colors.primary },
  del: { marginLeft: spacing(1) },
  addBtn: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: spacing(2),
    borderWidth: 1.5,
    borderColor: colors.primary,
    borderStyle: 'dashed',
    borderRadius: radius.lg,
    paddingVertical: spacing(3.5),
    marginTop: spacing(1),
  },
  addPlus: { fontSize: 18, color: colors.primary, fontWeight: '800' },
  addText: { fontSize: 14, color: colors.primary, fontWeight: '700' },
});
