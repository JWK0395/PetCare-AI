import React from 'react';
import { StyleSheet, Text, View } from 'react-native';
import type { SummaryContent } from '../api/types';
import { colors, spacing } from '../theme';

/** 병원 전달용 상태 요약 — 문서 4섹션 렌더 (요약 화면 · 응급 이메일 공용) */
export default function SummaryDocumentView({
  content,
  createdAt,
}: {
  content: SummaryContent;
  createdAt: string;
}) {
  const signs = content.risk_signs ?? [];
  return (
    <View>
      <Section title="1. 문서 정보">
        <Row label="문서 제목" value={content.title} />
        <Row label="생성 일시" value={createdAt} />
        <Row label="사용 데이터 기간" value={content.data_period} />
      </Section>

      <Section title="2. 반려동물 정보">
        <Row label="이름" value={content.pet_name} />
        <Row label="종" value={content.species} />
        <Row label="품종" value={content.breed} />
        <Row label="성별/중성화" value={content.sex_neuter} />
        <Row label="나이" value={content.age_label} />
        <Row label="현재 체중" value={content.weight} />
        <Row label="현재 복용 중인 약" value={content.medications} />
        <Row label="알레르기" value={content.allergies} />
      </Section>

      <Section title="3. 상태">
        <Row label="상태 분류" value={content.risk_label} />
        <Text style={styles.subLabel}>확인된 위험 징후</Text>
        {signs.length > 0 ? (
          signs.map((s, i) => (
            <Text key={`${s}-${i}`} style={styles.bullet}>
              · {s}
            </Text>
          ))
        ) : (
          <Text style={styles.bulletEmpty}>· 특이 위험 징후 없음</Text>
        )}
      </Section>

      <Section title="4. 주호소 및 주요 변화">
        <Row label="주호소" value={content.chief_complaint} />
        <Row label="주요 변화" value={content.major_changes} />
        <Row label="경과" value={content.progress} />
        {content.owner_note ? (
          <Row label="보호자 메모" value={content.owner_note} />
        ) : null}
      </Section>
    </View>
  );
}

function Section({
  title,
  children,
}: {
  title: string;
  children: React.ReactNode;
}) {
  return (
    <View style={styles.section}>
      <Text style={styles.sectionTitle}>{title}</Text>
      {children}
    </View>
  );
}

function Row({ label, value }: { label: string; value: string }) {
  return (
    <View style={styles.row}>
      <Text style={styles.rowLabel}>{label}</Text>
      <Text style={styles.rowValue}>{value || '-'}</Text>
    </View>
  );
}

const styles = StyleSheet.create({
  section: {
    borderTopWidth: StyleSheet.hairlineWidth,
    borderTopColor: colors.border,
    paddingTop: spacing(3),
    paddingBottom: spacing(1),
    marginTop: spacing(2),
  },
  sectionTitle: {
    fontSize: 14,
    fontWeight: '800',
    color: colors.primary,
    marginBottom: spacing(2),
  },
  row: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    paddingVertical: spacing(1.5),
    gap: spacing(3),
  },
  rowLabel: {
    width: 108,
    fontSize: 13,
    color: colors.textSecondary,
    fontWeight: '600',
  },
  rowValue: { flex: 1, fontSize: 14, color: colors.text, lineHeight: 20 },
  subLabel: {
    fontSize: 13,
    color: colors.textSecondary,
    fontWeight: '600',
    marginTop: spacing(1.5),
    marginBottom: spacing(1),
  },
  bullet: {
    fontSize: 14,
    color: colors.text,
    lineHeight: 21,
    paddingLeft: spacing(2),
  },
  bulletEmpty: {
    fontSize: 14,
    color: colors.textTertiary,
    lineHeight: 21,
    paddingLeft: spacing(2),
  },
});
