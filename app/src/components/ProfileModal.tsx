import React, { useEffect, useState } from 'react';
import {
  ScrollView,
  StyleSheet,
  Text,
  TextInput,
  TouchableOpacity,
  View,
} from 'react-native';
import { createPet, updatePet } from '../api/client';
import type { Pet } from '../api/types';
import { colors, radius, spacing } from '../theme';
import { dotDate } from '../utils/date';
import { useAlert } from './AlertProvider';
import AppModal from './AppModal';
import { Button } from './ui';

const SIZE_OPTIONS = ['소형', '중형', '대형'];

/** 기존 값("소형견" 등)을 소형/중형/대형으로 정규화 */
function normalizeSize(value: string): string {
  if (value.includes('중형')) {
    return '중형';
  }
  if (value.includes('대형')) {
    return '대형';
  }
  return '소형';
}

/** 프로필 등록/수정 팝업 */
export default function ProfileModal({
  visible,
  mode,
  pet,
  onClose,
  onSaved,
}: {
  visible: boolean;
  mode: 'create' | 'edit';
  pet?: Pet | null;
  onClose: () => void;
  onSaved: (savedId: number) => void;
}) {
  const editing = mode === 'edit' && !!pet;
  const showAlert = useAlert();

  const [name, setName] = useState('');
  const [species, setSpecies] = useState('강아지');
  const [breed, setBreed] = useState('');
  const [birthDate, setBirthDate] = useState('');
  const [sex, setSex] = useState('수컷');
  const [neutered, setNeutered] = useState(false);
  const [weight, setWeight] = useState('');
  const [sizeClass, setSizeClass] = useState('소형');
  const [diseases, setDiseases] = useState('');
  const [medications, setMedications] = useState('');
  const [supplement, setSupplement] = useState('');
  const [allergies, setAllergies] = useState('');
  const [saving, setSaving] = useState(false);

  // 팝업이 열릴 때 값 초기화
  useEffect(() => {
    if (!visible) {
      return;
    }
    if (editing && pet) {
      setName(pet.name);
      setSpecies(pet.species);
      setBreed(pet.breed);
      setBirthDate(pet.birth_date || '');
      setSex(pet.sex);
      setNeutered(pet.is_neutered);
      setWeight(pet.weight_kg != null ? String(pet.weight_kg) : '');
      setSizeClass(normalizeSize(pet.size_class || '소형'));
      setDiseases(pet.diseases);
      setMedications(pet.medications);
      setSupplement(pet.supplement);
      setAllergies(pet.allergies);
    } else {
      setName('');
      setSpecies('강아지');
      setBreed('');
      setBirthDate('');
      setSex('수컷');
      setNeutered(false);
      setWeight('');
      setSizeClass('소형');
      setDiseases('');
      setMedications('');
      setSupplement('');
      setAllergies('');
    }
  }, [visible, editing, pet]);

  const onSave = async () => {
    if (!name.trim()) {
      showAlert('프로필', '이름을 입력해 주세요.');
      return;
    }
    if (birthDate && !/^\d{4}-\d{2}-\d{2}$/.test(birthDate)) {
      showAlert('프로필', '생년월일은 YYYY-MM-DD 형식으로 입력해 주세요.');
      return;
    }
    setSaving(true);
    try {
      const body = {
        name: name.trim(),
        species,
        breed: breed.trim(),
        birth_date: birthDate || null,
        sex,
        is_neutered: neutered,
        weight_kg: weight ? parseFloat(weight) : null,
        size_class: sizeClass,
        diseases: diseases.trim(),
        medications: medications.trim(),
        supplement: supplement.trim(),
        allergies: allergies.trim(),
      };
      const saved =
        editing && pet ? await updatePet(pet.id, body) : await createPet(body);
      onSaved(saved.id);
    } catch (e) {
      showAlert('오류', e instanceof Error ? e.message : '저장 실패');
    } finally {
      setSaving(false);
    }
  };

  return (
    <AppModal
      visible={visible}
      size="large"
      title={editing ? '프로필 수정' : '반려동물 등록'}
      onClose={onClose}>
      <View style={styles.wrap}>
        <ScrollView
          style={styles.scroll}
          contentContainerStyle={styles.body}
          keyboardShouldPersistTaps="handled">
          <Field label="이름 *" value={name} onChange={setName} placeholder="콩이" />
          <ChoiceField
            label="종"
            options={['강아지', '고양이']}
            value={species}
            onChange={setSpecies}
          />
          <Field
            label="견종 / 묘종"
            value={breed}
            onChange={setBreed}
            placeholder="말티즈 · 순종"
          />
          <Field
            label="생년월일 (YYYY-MM-DD)"
            value={birthDate}
            onChange={setBirthDate}
            placeholder="2021-09-14"
          />
          <ChoiceField
            label="성별"
            options={['수컷', '암컷']}
            value={sex}
            onChange={setSex}
          />
          <ChoiceField
            label="중성화 여부"
            options={['완료', '미완료']}
            value={neutered ? '완료' : '미완료'}
            onChange={v => setNeutered(v === '완료')}
          />
          <Field
            label="몸무게 (kg)"
            value={weight}
            onChange={setWeight}
            placeholder="5.08"
            keyboardType="decimal-pad"
          />
          <ChoiceField
            label="크기"
            options={SIZE_OPTIONS}
            value={sizeClass}
            onChange={setSizeClass}
          />
          <Field
            label="기존 질병"
            value={diseases}
            onChange={setDiseases}
            placeholder="슬개골 탈구 2기"
          />
          <Field
            label="복용약"
            value={medications}
            onChange={setMedications}
            placeholder="예: 항생제 1일 2회"
          />
          <Field
            label="영양제"
            value={supplement}
            onChange={setSupplement}
            placeholder="관절 영양제 1일 1회"
          />
          <Field
            label="알레르기"
            value={allergies}
            onChange={setAllergies}
            placeholder="닭고기 알레르기"
          />
          {editing && pet ? (
            <Text style={styles.updatedAt}>
              프로필 수정일 · {dotDate(pet.updated_at)}
            </Text>
          ) : null}
        </ScrollView>

        {/* 하단 고정 저장 버튼 */}
        <View style={styles.footer}>
          <Button
            title={editing ? '저장' : '등록하기'}
            onPress={onSave}
            loading={saving}
          />
        </View>
      </View>
    </AppModal>
  );
}

function Field({
  label,
  value,
  onChange,
  placeholder,
  keyboardType,
}: {
  label: string;
  value: string;
  onChange: (v: string) => void;
  placeholder?: string;
  keyboardType?: 'default' | 'decimal-pad';
}) {
  return (
    <View style={styles.field}>
      <Text style={styles.fieldLabel}>{label}</Text>
      <TextInput
        style={styles.fieldInput}
        value={value}
        onChangeText={onChange}
        placeholder={placeholder}
        placeholderTextColor={colors.textTertiary}
        keyboardType={keyboardType || 'default'}
      />
    </View>
  );
}

function ChoiceField({
  label,
  options,
  value,
  onChange,
}: {
  label: string;
  options: string[];
  value: string;
  onChange: (v: string) => void;
}) {
  return (
    <View style={styles.field}>
      <Text style={styles.fieldLabel}>{label}</Text>
      <View style={styles.choiceRow}>
        {options.map(option => (
          <TouchableOpacity
            key={option}
            style={[styles.choice, value === option && styles.choiceActive]}
            onPress={() => onChange(option)}>
            <Text
              style={[
                styles.choiceText,
                value === option && styles.choiceTextActive,
              ]}>
              {option}
            </Text>
          </TouchableOpacity>
        ))}
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  wrap: { flex: 1 },
  scroll: { flex: 1 },
  body: { paddingHorizontal: spacing(4), paddingBottom: spacing(4) },
  footer: {
    paddingHorizontal: spacing(4),
    paddingTop: spacing(3),
    paddingBottom: spacing(2),
    borderTopWidth: StyleSheet.hairlineWidth,
    borderTopColor: colors.border,
    backgroundColor: colors.background,
  },
  field: { marginBottom: spacing(3.5) },
  updatedAt: {
    fontSize: 12,
    color: colors.textTertiary,
    marginTop: spacing(1),
    marginBottom: spacing(2),
  },
  fieldLabel: {
    fontSize: 12,
    fontWeight: '600',
    color: colors.textSecondary,
    marginBottom: spacing(1.5),
  },
  fieldInput: {
    borderWidth: 1,
    borderColor: colors.border,
    borderRadius: radius.md,
    paddingHorizontal: spacing(3),
    paddingVertical: spacing(2.5),
    fontSize: 14,
    color: colors.text,
    backgroundColor: colors.card,
  },
  choiceRow: { flexDirection: 'row', flexWrap: 'wrap', gap: spacing(2) },
  choice: {
    borderWidth: 1,
    borderColor: colors.border,
    borderRadius: radius.full,
    paddingHorizontal: spacing(3.5),
    paddingVertical: spacing(1.5),
    backgroundColor: colors.card,
  },
  choiceActive: {
    backgroundColor: colors.primarySoft,
    borderColor: colors.primary,
  },
  choiceText: { fontSize: 13, color: colors.textSecondary },
  choiceTextActive: { color: colors.primary, fontWeight: '700' },
});
