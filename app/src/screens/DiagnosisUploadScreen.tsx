import { useFocusEffect } from '@react-navigation/native';
import React, { useCallback, useState } from 'react';
import { ScrollView, StyleSheet, Text, TouchableOpacity, View } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import {
  errorCodes,
  isErrorWithCode,
  pick,
  types,
} from '@react-native-documents/picker';
import { extractDiagnosis, saveDiagnosis } from '../api/client';
import type { DiagnosisExtractResponse } from '../api/types';
import { useAlert } from '../components/AlertProvider';
import DiagnosisArchiveModal from '../components/DiagnosisArchiveModal';
import DiagnosisDetailModal from '../components/DiagnosisDetailModal';
import { Button, Card } from '../components/ui';
import { usePet } from '../state/PetContext';
import { colors, radius, spacing } from '../theme';

export default function DiagnosisUploadScreen() {
  const { pet } = usePet();
  const showAlert = useAlert();
  const [uploading, setUploading] = useState(false);
  const [extraction, setExtraction] = useState<DiagnosisExtractResponse | null>(
    null,
  );
  const [saving, setSaving] = useState(false);
  const [archiveVisible, setArchiveVisible] = useState(false);

  // 화면을 벗어났다 돌아오면 초기 상태(업로드 전)로 되돌린다.
  useFocusEffect(
    useCallback(() => {
      return () => {
        setExtraction(null);
        setArchiveVisible(false);
      };
    }, []),
  );

  const onPickFile = async () => {
    if (!pet) {
      return;
    }
    try {
      const [file] = await pick({
        type: [types.pdf, types.images],
      });
      setUploading(true);
      setExtraction(null);
      const result = await extractDiagnosis(pet.id, {
        uri: file.uri,
        name: file.name || 'diagnosis.pdf',
        type: file.type || 'application/pdf',
      });
      setExtraction(result);
    } catch (e) {
      if (isErrorWithCode(e) && e.code === errorCodes.OPERATION_CANCELED) {
        return;
      }
      showAlert('오류', e instanceof Error ? e.message : '업로드 실패');
    } finally {
      setUploading(false);
    }
  };

  const onSave = async () => {
    if (!pet || !extraction) {
      return;
    }
    setSaving(true);
    try {
      const f = extraction.fields;
      await saveDiagnosis(pet.id, {
        date: f.date || null,
        hospital: f.hospital,
        diagnosis: f.diagnosis,
        content: f.content,
        original_file_ref: extraction.original_file_ref,
      });
      showAlert('진단서 저장', '진단서가 저장되었어요.', [
        { text: '확인', onPress: () => setExtraction(null) },
      ]);
    } catch (e) {
      showAlert('오류', e instanceof Error ? e.message : '저장 실패');
    } finally {
      setSaving(false);
    }
  };

  return (
    <SafeAreaView style={styles.safe} edges={['top']}>
      <ScrollView contentContainerStyle={styles.container}>
        <Text style={styles.subtitle}>
          {pet ? `${pet.name} · 문서 보관함` : ''}
        </Text>
        <View style={styles.titleRow}>
          <Text style={styles.title}>진단서 등록</Text>
          <TouchableOpacity
            style={styles.archiveButton}
            onPress={() => setArchiveVisible(true)}>
            <Text style={styles.archiveButtonText}>📂 이전 진단서</Text>
          </TouchableOpacity>
        </View>

        {/* 업로드 박스 */}
        <Card style={styles.uploadCard}>
          <Text style={styles.uploadEmoji}>📄</Text>
          <Text style={styles.uploadText}>동물 진단서 PDF를 올려주세요</Text>
          <Button
            title="PDF 파일 업로드"
            onPress={onPickFile}
            loading={uploading}
            style={styles.uploadButton}
          />
        </Card>

      </ScrollView>

      {/* AI 추출 결과 — 진단서 상세 팝업(읽기 전용) 으로 확인 후 저장 */}
      {extraction ? (
        <DiagnosisDetailModal
          visible={!!extraction}
          title="AI 정리 결과"
          fields={{
            ...extraction.fields,
            original_file_ref: extraction.original_file_ref,
          }}
          onSave={onSave}
          saveLabel="진단서 저장"
          saving={saving}
          onClose={() => setExtraction(null)}
        />
      ) : null}

      {/* 이전 진단서 팝업 */}
      {pet ? (
        <DiagnosisArchiveModal
          visible={archiveVisible}
          petId={pet.id}
          onClose={() => setArchiveVisible(false)}
        />
      ) : null}
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  safe: { flex: 1, backgroundColor: colors.background },
  container: { padding: spacing(5), paddingBottom: spacing(8) },
  subtitle: { fontSize: 12, color: colors.textSecondary },
  titleRow: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    marginTop: 2,
    marginBottom: spacing(4),
  },
  title: {
    fontSize: 22,
    fontWeight: '800',
    color: colors.text,
  },
  archiveButton: {
    backgroundColor: colors.primarySoft,
    borderRadius: radius.full,
    paddingHorizontal: spacing(3),
    paddingVertical: spacing(1.5),
  },
  archiveButtonText: { fontSize: 12, fontWeight: '700', color: colors.primary },
  uploadCard: {
    alignItems: 'center',
    paddingVertical: spacing(8),
    borderWidth: 1.5,
    borderColor: colors.primarySoft,
    borderStyle: 'dashed',
    marginBottom: spacing(4),
  },
  uploadEmoji: { fontSize: 34, marginBottom: spacing(2) },
  uploadText: {
    fontSize: 14,
    color: colors.textSecondary,
    marginBottom: spacing(4),
  },
  uploadButton: { paddingHorizontal: spacing(8) },
  resultCard: {},
  resultTitle: {
    fontSize: 14,
    fontWeight: '700',
    color: colors.primary,
    marginBottom: spacing(3),
  },
  fileChip: {
    backgroundColor: colors.successSoft,
    borderRadius: radius.sm,
    paddingHorizontal: spacing(3),
    paddingVertical: spacing(2),
    marginBottom: spacing(3),
  },
  fileChipText: { fontSize: 12, color: colors.success, fontWeight: '600' },
  fieldRow: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingVertical: spacing(3),
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
  fieldValue: { flex: 1, fontSize: 14, color: colors.text, fontWeight: '500', lineHeight: 20 },
  contentRow: { alignItems: 'flex-start' },
  contentScroll: {
    flex: 1,
    maxHeight: 130,
    borderWidth: 1,
    borderColor: colors.border,
    borderRadius: radius.sm,
    paddingHorizontal: spacing(2.5),
    paddingVertical: spacing(2),
    backgroundColor: colors.background,
  },
  saveButton: { marginTop: spacing(4) },
});
