import React, { useCallback, useEffect, useMemo, useState } from 'react';
import {
  ScrollView,
  StyleSheet,
  Text,
  TextInput,
  TouchableOpacity,
  View,
} from 'react-native';
import { extractDiary, getRecords, saveRecord } from '../api/client';
import type { DailyRecord, DiaryExtractResponse } from '../api/types';
import { colors, radius, shadow, spacing } from '../theme';
import { todayKey } from '../utils/date';
import { useAlert } from './AlertProvider';
import AppModal from './AppModal';
import DiaryResultModal, { DiaryFieldList } from './DiaryResultModal';
import { Button } from './ui';

const WEEKDAYS = ['일', '월', '화', '수', '목', '금', '토'];

/** 달력 셀용 "YYYY-MM-DD" (month 는 0-based) */
function toKey(year: number, month: number, day: number): string {
  const pad = (n: number) => String(n).padStart(2, '0');
  return `${year}-${pad(month + 1)}-${pad(day)}`;
}

/** "2026-07-14" -> "2026년 7월 14일 화요일" */
function formatFullDate(key: string): string {
  const [y, m, d] = key.split('-').map(Number);
  const weekday = WEEKDAYS[new Date(y, m - 1, d).getDay()];
  return `${y}년 ${m}월 ${d}일 ${weekday}요일`;
}

/** 기록 탭 — 선택한 날짜의 일기를 중앙 팝업으로 보고 수정 */
export default function DiaryCalendarModal({
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
  const [records, setRecords] = useState<Map<string, DailyRecord>>(new Map());
  const [selectedKey, setSelectedKey] = useState<string>(todayKey());
  const [pickerVisible, setPickerVisible] = useState(false);
  const [rawText, setRawText] = useState('');
  const [extracting, setExtracting] = useState(false);
  const [extraction, setExtraction] = useState<DiaryExtractResponse | null>(null);
  const [resultVisible, setResultVisible] = useState(false);
  const [saving, setSaving] = useState(false);

  const load = useCallback(async () => {
    try {
      const list = await getRecords(petId, 365);
      setRecords(new Map(list.map(r => [r.record_date, r])));
    } catch {}
  }, [petId]);

  // 처음 열릴 때 오늘 날짜로 설정
  useEffect(() => {
    if (visible) {
      load();
      setSelectedKey(todayKey());
      setPickerVisible(false);
    }
  }, [visible, load]);

  const selected = records.get(selectedKey);

  // 선택 날짜가 바뀌면 일기 원문 갱신
  useEffect(() => {
    setRawText(selected?.raw_text || '');
  }, [selected]);

  // 버튼 → AI 정리 결과 미리보기 팝업 (저장 전 확인)
  const onExtract = async () => {
    if (!rawText.trim()) {
      return;
    }
    setExtracting(true);
    try {
      const result = await extractDiary(petId, rawText);
      setExtraction(result);
      setResultVisible(true);
    } catch (e) {
      showAlert('오류', e instanceof Error ? e.message : '정리 실패');
    } finally {
      setExtracting(false);
    }
  };

  // 팝업의 저장 → 날짜 기준 upsert
  const onConfirmSave = async () => {
    const wasEditing = !!selected;
    setSaving(true);
    try {
      await saveRecord(petId, {
        ...((extraction?.fields ?? {}) as Record<string, unknown>),
        raw_text: rawText,
        record_date: selectedKey,
      });
      setResultVisible(false);
      setExtraction(null);
      showAlert(
        wasEditing ? '일기 수정' : '일기 저장',
        wasEditing
          ? 'AI가 항목을 다시 정리했어요.'
          : '일기를 저장하고 AI가 정리했어요.',
      );
      await load();
      onChanged?.();
    } catch (e) {
      showAlert('오류', e instanceof Error ? e.message : '저장 실패');
    } finally {
      setSaving(false);
    }
  };

  return (
    <>
      <AppModal visible={visible} title="지난 일기" onClose={onClose}>
        <ScrollView
          style={styles.scroll}
          contentContainerStyle={styles.body}
          keyboardShouldPersistTaps="handled">
          {/* 날짜 선택 버튼 */}
          <TouchableOpacity
            style={styles.dateSelector}
            onPress={() => setPickerVisible(true)}
            activeOpacity={0.8}>
            <Text style={styles.dateIcon}>📅</Text>
            <Text style={styles.dateText}>{formatFullDate(selectedKey)}</Text>
            <Text style={styles.dateChevron}>▾</Text>
          </TouchableOpacity>

          <View style={styles.detailCard}>
            <Text style={styles.fieldLabel}>
              일기 원문 {selected ? '(수정 가능)' : '(새로 작성)'}
            </Text>
            {!selected ? (
              <Text style={styles.noRecordHint}>
                이 날은 기록이 없어요. 아래에 일기를 작성하면 새로 저장돼요.
              </Text>
            ) : null}
            <TextInput
              style={[styles.input, styles.inputMultiline]}
              multiline
              value={rawText}
              onChangeText={setRawText}
              placeholder={
                selected ? '일기 내용' : '이 날의 일기를 작성해 보세요.'
              }
              placeholderTextColor={colors.textTertiary}
              textAlignVertical="top"
            />

            {selected ? (
              <View style={styles.aiSection}>
                <Text style={styles.aiSectionTitle}>
                  AI 정리 항목 (저장된 기록)
                </Text>
                <ScrollView
                  style={styles.aiScroll}
                  nestedScrollEnabled
                  keyboardShouldPersistTaps="handled">
                  <DiaryFieldList
                    fields={selected as unknown as Record<string, unknown>}
                  />
                </ScrollView>
              </View>
            ) : null}

            <Button
              title={selected ? '저장하고 AI 다시 정리' : '저장하고 AI 정리'}
              onPress={onExtract}
              loading={extracting}
              disabled={!rawText.trim()}
              style={styles.saveBtn}
            />
          </View>
        </ScrollView>
      </AppModal>

      {/* AI 정리 결과 미리보기 팝업 (저장/취소) */}
      <DiaryResultModal
        visible={visible && resultVisible}
        title="AI 정리 결과"
        subtitle={`${formatFullDate(selectedKey)} · 저장 전 확인`}
        foundCount={extraction?.items.length ?? 0}
        fields={(extraction?.fields ?? {}) as Record<string, unknown>}
        onSave={onConfirmSave}
        saveLabel={selected ? '수정 저장' : '저장'}
        saving={saving}
        onClose={() => {
          setResultVisible(false);
          setExtraction(null);
        }}
      />

      {/* 날짜 선택 팝업 */}
      <DatePickerModal
        visible={visible && pickerVisible}
        selectedKey={selectedKey}
        records={records}
        onSelect={key => {
          setSelectedKey(key);
          setPickerVisible(false);
        }}
        onClose={() => setPickerVisible(false)}
      />
    </>
  );
}

/** 날짜 선택 팝업 (월 달력) */
function DatePickerModal({
  visible,
  selectedKey,
  records,
  onSelect,
  onClose,
}: {
  visible: boolean;
  selectedKey: string;
  records: Map<string, DailyRecord>;
  onSelect: (key: string) => void;
  onClose: () => void;
}) {
  const [y0, m0] = selectedKey.split('-').map(Number);
  const [year, setYear] = useState(y0);
  const [month, setMonth] = useState(m0 - 1);

  // 팝업을 열 때마다 선택 날짜의 달로 이동
  useEffect(() => {
    if (visible) {
      const [yy, mm] = selectedKey.split('-').map(Number);
      setYear(yy);
      setMonth(mm - 1);
    }
  }, [visible, selectedKey]);

  const weeks = useMemo(() => {
    const firstDay = new Date(year, month, 1).getDay();
    const daysInMonth = new Date(year, month + 1, 0).getDate();
    const cells: (number | null)[] = [
      ...Array(firstDay).fill(null),
      ...Array.from({ length: daysInMonth }, (_, i) => i + 1),
    ];
    while (cells.length % 7 !== 0) {
      cells.push(null);
    }
    const rows: (number | null)[][] = [];
    for (let i = 0; i < cells.length; i += 7) {
      rows.push(cells.slice(i, i + 7));
    }
    return rows;
  }, [year, month]);

  const moveMonth = (delta: number) => {
    const d = new Date(year, month + delta, 1);
    setYear(d.getFullYear());
    setMonth(d.getMonth());
  };

  const tKey = todayKey();

  return (
    <AppModal visible={visible} title="날짜 선택" onClose={onClose}>
      <View style={styles.pickerBody}>
        <View style={styles.monthRow}>
          <TouchableOpacity onPress={() => moveMonth(-1)} style={styles.navBtn}>
            <Text style={styles.navText}>‹</Text>
          </TouchableOpacity>
          <Text style={styles.monthTitle}>
            {year}년 {month + 1}월
          </Text>
          <TouchableOpacity onPress={() => moveMonth(1)} style={styles.navBtn}>
            <Text style={styles.navText}>›</Text>
          </TouchableOpacity>
        </View>

        <View style={styles.weekRow}>
          {WEEKDAYS.map((w, i) => (
            <Text
              key={w}
              style={[
                styles.weekday,
                i === 0 && styles.sun,
                i === 6 && styles.sat,
              ]}>
              {w}
            </Text>
          ))}
        </View>

        {weeks.map((week, wi) => (
          <View key={`w${wi}`} style={styles.weekRow}>
            {week.map((day, di) => {
              if (day === null) {
                return <View key={`e${di}`} style={styles.dayCell} />;
              }
              const key = toKey(year, month, day);
              const hasRecord = records.has(key);
              const isSelected = selectedKey === key;
              const isToday = key === tKey;
              const isFuture = key > tKey;
              return (
                <TouchableOpacity
                  key={key}
                  style={styles.dayCell}
                  disabled={isFuture}
                  onPress={() => onSelect(key)}>
                  <View
                    style={[
                      styles.dayInner,
                      isToday && styles.dayToday,
                      isSelected && styles.daySelected,
                    ]}>
                    <Text
                      style={[
                        styles.dayText,
                        isFuture && styles.dayFuture,
                        isSelected && styles.dayTextSelected,
                      ]}>
                      {day}
                    </Text>
                    <View
                      style={[
                        styles.dot,
                        hasRecord && !isSelected && styles.dotOn,
                        hasRecord && isSelected && styles.dotOnSelected,
                      ]}
                    />
                  </View>
                </TouchableOpacity>
              );
            })}
          </View>
        ))}
        <Text style={styles.pickerHint}>
          점이 있는 날짜에 기록이 있어요.
        </Text>
      </View>
    </AppModal>
  );
}

const styles = StyleSheet.create({
  scroll: { flexGrow: 0 },
  body: { paddingHorizontal: spacing(4), paddingBottom: spacing(5) },
  dateSelector: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: colors.primarySoft,
    borderRadius: radius.md,
    paddingHorizontal: spacing(4),
    paddingVertical: spacing(3.5),
    marginBottom: spacing(3),
    gap: spacing(2),
  },
  dateIcon: { fontSize: 16 },
  dateText: { flex: 1, fontSize: 15, fontWeight: '700', color: colors.primary },
  dateChevron: { fontSize: 14, color: colors.primary },
  detailCard: {
    backgroundColor: colors.card,
    borderRadius: radius.lg,
    padding: spacing(4),
    ...shadow,
  },
  fieldLabel: {
    fontSize: 12,
    fontWeight: '600',
    color: colors.textSecondary,
    marginBottom: spacing(1.5),
  },
  input: {
    borderWidth: 1,
    borderColor: colors.border,
    borderRadius: radius.md,
    paddingHorizontal: spacing(3),
    paddingVertical: spacing(2.5),
    fontSize: 14,
    color: colors.text,
    backgroundColor: colors.background,
    marginBottom: spacing(3),
  },
  inputMultiline: { height: 130 }, // 고정 높이 + 입력창 내부 스크롤
  aiSection: {
    backgroundColor: colors.primarySoft,
    borderRadius: radius.md,
    paddingHorizontal: spacing(3.5),
    paddingTop: spacing(2),
    paddingBottom: spacing(1),
  },
  aiSectionTitle: {
    fontSize: 12,
    fontWeight: '700',
    color: colors.primary,
    marginBottom: spacing(1),
  },
  aiScroll: { maxHeight: 200 },
  saveBtn: { marginTop: spacing(3) },
  noRecord: {
    textAlign: 'center',
    color: colors.textSecondary,
    fontSize: 14,
    fontWeight: '600',
    paddingTop: spacing(4),
  },
  noRecordHint: {
    textAlign: 'center',
    color: colors.textTertiary,
    fontSize: 12,
    marginTop: spacing(1.5),
    paddingBottom: spacing(4),
  },
  // date picker
  pickerBody: { paddingHorizontal: spacing(4), paddingBottom: spacing(4) },
  monthRow: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: spacing(2),
    gap: spacing(6),
  },
  navBtn: { padding: spacing(2) },
  navText: { fontSize: 22, color: colors.primary, fontWeight: '700' },
  monthTitle: { fontSize: 16, fontWeight: '800', color: colors.text },
  weekRow: { flexDirection: 'row' },
  weekday: {
    flex: 1,
    textAlign: 'center',
    fontSize: 12,
    fontWeight: '700',
    color: colors.textSecondary,
    paddingVertical: spacing(1.5),
  },
  sun: { color: colors.danger },
  sat: { color: colors.primary },
  dayCell: { flex: 1, alignItems: 'center', paddingVertical: spacing(1) },
  dayInner: {
    width: 38,
    height: 42,
    borderRadius: radius.md,
    alignItems: 'center',
    justifyContent: 'center',
  },
  dayToday: { borderWidth: 1, borderColor: colors.primary },
  daySelected: { backgroundColor: colors.primary },
  dayText: { fontSize: 14, color: colors.text, fontWeight: '600' },
  dayFuture: { color: colors.border },
  dayTextSelected: { color: '#fff' },
  dot: {
    width: 5,
    height: 5,
    borderRadius: 3,
    marginTop: 3,
    backgroundColor: 'transparent',
  },
  dotOn: { backgroundColor: colors.primary },
  dotOnSelected: { backgroundColor: '#fff' },
  pickerHint: {
    textAlign: 'center',
    color: colors.textTertiary,
    fontSize: 11,
    marginTop: spacing(3),
  },
});
