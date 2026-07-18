import { useFocusEffect } from '@react-navigation/native';
import React, { useCallback, useState } from 'react';
import {
  KeyboardAvoidingView,
  ScrollView,
  StyleSheet,
  Text,
  TextInput,
  TouchableOpacity,
  View,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { extractDiary, getRecords, saveRecord } from '../api/client';
import type { DailyRecord, DiaryExtractResponse } from '../api/types';
import { useAlert } from '../components/AlertProvider';
import DiaryCalendarModal from '../components/DiaryCalendarModal';
import DiaryResultModal from '../components/DiaryResultModal';
import { Button, Card } from '../components/ui';
import { usePet } from '../state/PetContext';
import { colors, radius, spacing } from '../theme';
import { koreanDate, todayKey } from '../utils/date';

export default function RecordScreen() {
  const { pet } = usePet();
  const showAlert = useAlert();
  const [diary, setDiary] = useState('');
  const [extracting, setExtracting] = useState(false);
  const [extraction, setExtraction] = useState<DiaryExtractResponse | null>(null);
  const [resultVisible, setResultVisible] = useState(false);
  const [saving, setSaving] = useState(false);
  const [calendarVisible, setCalendarVisible] = useState(false);
  const [todayRecord, setTodayRecord] = useState<DailyRecord | null>(null);

  // 화면을 열면 오늘 일기가 있으면 불러와 보여주고, 없으면 빈 화면으로 둔다.
  const loadToday = useCallback(async () => {
    if (!pet) {
      return;
    }
    try {
      const list = await getRecords(pet.id, 1);
      const rec = list.find(r => r.record_date === todayKey()) ?? null;
      setTodayRecord(rec);
      setDiary(rec?.raw_text ?? '');
    } catch {}
  }, [pet]);

  // 화면에 들어올 때마다 초기 상태로: 열린 팝업을 닫고 오늘 기록을 다시 불러온다.
  useFocusEffect(
    useCallback(() => {
      setResultVisible(false);
      setExtraction(null);
      setCalendarVisible(false);
      loadToday();
    }, [loadToday]),
  );

  if (!pet) {
    return (
      <SafeAreaView style={styles.safe} edges={['top']}>
        <View style={styles.center}>
          <Text style={styles.hint}>먼저 홈에서 반려동물을 등록해 주세요.</Text>
        </View>
      </SafeAreaView>
    );
  }

  // 버튼 → AI 정리 결과 미리보기 팝업. 팝업의 '기록 저장' 을 눌러야 실제 저장된다.
  const onExtract = async () => {
    if (!diary.trim()) {
      showAlert('오늘의 기록', '일기 내용을 입력해 주세요.');
      return;
    }
    setExtracting(true);
    try {
      const result = await extractDiary(pet.id, diary);
      setExtraction(result);
      setResultVisible(true);
    } catch (e) {
      showAlert('오류', e instanceof Error ? e.message : '정리 실패');
    } finally {
      setExtracting(false);
    }
  };

  const onSave = async () => {
    if (!extraction) {
      return;
    }
    setSaving(true);
    try {
      await saveRecord(pet.id, { ...extraction.fields, raw_text: diary });
      setResultVisible(false);
      setExtraction(null);
      await loadToday();
      showAlert('기록 저장', '오늘의 기록이 저장되었어요.');
    } catch (e) {
      showAlert('오류', e instanceof Error ? e.message : '저장 실패');
    } finally {
      setSaving(false);
    }
  };

  return (
    <SafeAreaView style={styles.safe} edges={['top']}>
      {/* edge-to-edge(Android)에서 adjustResize 미동작 → padding 으로 키보드 회피 */}
      <KeyboardAvoidingView style={styles.flex} behavior="padding">
        <ScrollView
          contentContainerStyle={styles.container}
          keyboardShouldPersistTaps="handled">
          <Text style={styles.date}>{koreanDate()}</Text>
          <View style={styles.titleRow}>
            <Text style={styles.title}>오늘의 기록</Text>
            <TouchableOpacity
              style={styles.calendarButton}
              onPress={() => setCalendarVisible(true)}>
              <Text style={styles.calendarButtonText}>📅 지난 일기</Text>
            </TouchableOpacity>
          </View>

          {/* AI가 정리하는 항목 안내 (간단히 한두 줄) */}
          <View style={styles.hintBox}>
            <Text style={styles.hintDesc}>
              자연스럽게 적으면 AI가 식사·음수·활동·증상·배변·구토·기타로 자동
              정리해요.
            </Text>
          </View>

          {/* 일기 입력 */}
          <Card style={styles.diaryCard}>
            <Text style={styles.cardLabel}>일기로 남기기</Text>
            {todayRecord ? (
              <Text style={styles.todayHint}>
                오늘 일기가 저장돼 있어요. 수정한 뒤 다시 정리·저장할 수 있어요.
              </Text>
            ) : null}
            <TextInput
              style={styles.diaryInput}
              multiline
              placeholder={
                '아침에 사료를 반쯤 남겼다. 산책은 20분 정도 했는데 평소보다 걷기 싫어하는 느낌. 물은 잘 마셨고, 오후에 노란 토를 한 번 했다.'
              }
              placeholderTextColor={colors.textTertiary}
              value={diary}
              onChangeText={setDiary}
              textAlignVertical="top"
            />
            <Button
              title={todayRecord ? 'AI로 다시 정리하기' : 'AI로 정리하기'}
              onPress={onExtract}
              loading={extracting}
              variant="secondary"
            />
          </Card>

        </ScrollView>
      </KeyboardAvoidingView>

      {/* AI 정리 결과 미리보기 팝업 (저장/취소) */}
      <DiaryResultModal
        visible={resultVisible}
        title="AI 정리 결과"
        foundCount={extraction?.items.length ?? 0}
        fields={(extraction?.fields ?? {}) as Record<string, unknown>}
        onSave={onSave}
        saveLabel="기록 저장"
        saving={saving}
        onClose={() => {
          setResultVisible(false);
          setExtraction(null);
        }}
      />

      {/* 달력 팝업 — 이전 일기 보기/수정/작성 */}
      <DiaryCalendarModal
        visible={calendarVisible}
        petId={pet.id}
        onClose={() => setCalendarVisible(false)}
        onChanged={loadToday}
      />
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  safe: { flex: 1, backgroundColor: colors.background },
  flex: { flex: 1 },
  center: { flex: 1, alignItems: 'center', justifyContent: 'center' },
  hint: { color: colors.textSecondary },
  container: { padding: spacing(5), paddingBottom: spacing(8) },
  date: { fontSize: 13, color: colors.textSecondary, marginBottom: spacing(1) },
  titleRow: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    marginBottom: spacing(4),
  },
  title: { fontSize: 24, fontWeight: '800', color: colors.text },
  calendarButton: {
    backgroundColor: colors.primarySoft,
    borderRadius: radius.full,
    paddingHorizontal: spacing(3),
    paddingVertical: spacing(1.5),
  },
  calendarButtonText: { fontSize: 12, fontWeight: '700', color: colors.primary },
  hintBox: {
    backgroundColor: colors.primarySoft,
    borderRadius: radius.lg,
    paddingHorizontal: spacing(4),
    paddingVertical: spacing(3),
    marginBottom: spacing(4),
  },
  hintDesc: {
    fontSize: 13,
    color: colors.primaryDark,
    lineHeight: 19,
  },
  diaryCard: { marginBottom: spacing(4) },
  cardLabel: {
    fontSize: 14,
    fontWeight: '700',
    color: colors.text,
    marginBottom: spacing(2),
  },
  todayHint: {
    fontSize: 12,
    color: colors.textSecondary,
    marginBottom: spacing(2),
    lineHeight: 17,
  },
  diaryInput: {
    height: 170, // 고정 높이 — 내용이 길어지면 입력창 안에서 스크롤(레이아웃이 밀리지 않음)
    borderWidth: 1,
    borderColor: colors.border,
    borderRadius: radius.md,
    padding: spacing(3),
    fontSize: 14,
    lineHeight: 21,
    color: colors.text,
    marginBottom: spacing(3),
    backgroundColor: colors.background,
  },
});
