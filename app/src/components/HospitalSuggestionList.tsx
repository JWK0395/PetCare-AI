import React from 'react';
import { Linking, StyleSheet, Text, TouchableOpacity, View } from 'react-native';
import type { HospitalSuggestion } from '../api/types';
import { colors, radius, shadow, spacing } from '../theme';

/**
 * AI 가 실시간 검색으로 찾은 병원 목록 (응급 카드 전용).
 *
 * 서버 DB 의 시드 병원(`Hospital`)과 다른 타입이다. 이쪽은 웹 검색 결과라
 * 거리·영업 상태를 확정할 수 없고, 대신 적합도 점수와 "왜 이 병원인지"가 있다.
 *
 * 비어 있으면 아무것도 그리지 않는다 — 그 경우 호출자가 기존 서버 병원 목록을
 * 대신 보여준다(지역 미확보·검색 실패·mock 모드).
 *
 * **표시 원칙**: 검색 결과만으로 "지금 진료 가능"이라고 단정하지 않는다.
 * 모든 항목에 전화 확인 안내를 붙이고, 전화번호가 없으면 전화 버튼을 숨긴다.
 */
export default function HospitalSuggestionList({
  hospitals,
  onSendEmail,
}: {
  hospitals?: HospitalSuggestion[] | null;
  /** 이 병원 앞으로 응급 이메일 초안을 만든다. 없으면 이메일 버튼을 숨긴다. */
  onSendEmail?: (hospital: HospitalSuggestion) => void;
}) {
  if (!hospitals || hospitals.length === 0) {
    return null;
  }

  const call = (phone: string) => {
    Linking.openURL(`tel:${phone}`).catch(() => {});
  };
  const openSource = (url: string) => {
    if (/^https?:\/\//.test(url)) {
      Linking.openURL(url).catch(() => {});
    }
  };

  return (
    <View>
      <Text style={styles.aiBadge}>AI가 지금 검색한 결과</Text>

      {hospitals.map((hospital, index) => (
        <View key={`${hospital.name}-${index}`} style={styles.card}>
          <View style={styles.headerRow}>
            <Text style={styles.name} numberOfLines={2}>
              {hospital.name}
            </Text>
            <SuitabilityTag suitability={hospital.suitability} />
          </View>

          {hospital.address ? (
            <Text style={styles.address} numberOfLines={2}>
              {hospital.address}
            </Text>
          ) : null}

          {/* 검색 결과에서 확인된 특징 — 근거 없는 단정을 피하려고 '언급됨'으로 표기 */}
          <View style={styles.tagRow}>
            {hospital.emergency_mentioned ? (
              <Text style={styles.tag}>응급 진료 언급</Text>
            ) : null}
            {hospital.open_24h_mentioned ? (
              <Text style={styles.tag}>24시간 언급</Text>
            ) : null}
          </View>

          {hospital.matched_reasons.length > 0 ? (
            <Text style={styles.reason} numberOfLines={2}>
              {hospital.matched_reasons.slice(0, 2).join(' · ')}
            </Text>
          ) : null}

          {/* 확인 문구는 앞의 2개만 — 응급 화면에서 읽을 시간이 없다.
              AI 쪽에서 결정에 가장 중요한 것(지역 불일치)을 1번 자리에 둔다. */}
          {(hospital.verification_required.length > 0
            ? hospital.verification_required.slice(0, 2)
            : ['방문 전에 전화로 현재 진료 및 응급 접수 가능 여부를 확인하세요.']
          ).map((notice, noticeIndex) => (
            <Text key={notice} style={noticeIndex === 0 ? styles.notice : styles.noticeStrong}>
              {notice}
            </Text>
          ))}

          <View style={styles.actions}>
            {onSendEmail ? (
              <TouchableOpacity
                style={styles.emailButton}
                onPress={() => onSendEmail(hospital)}>
                <Text style={styles.emailText}>이메일</Text>
              </TouchableOpacity>
            ) : null}
            {hospital.phone ? (
              <TouchableOpacity
                style={styles.callButton}
                onPress={() => call(hospital.phone as string)}>
                <Text style={styles.callText}>전화 {hospital.phone}</Text>
              </TouchableOpacity>
            ) : (
              <View style={[styles.callButton, styles.callDisabled]}>
                <Text style={styles.callDisabledText}>전화번호 미확인</Text>
              </View>
            )}
            {hospital.source_url ? (
              <TouchableOpacity
                style={styles.sourceButton}
                onPress={() => openSource(hospital.source_url)}>
                <Text style={styles.sourceText}>출처</Text>
              </TouchableOpacity>
            ) : null}
          </View>
        </View>
      ))}
    </View>
  );
}

/** 적합도 등급 뱃지 — 점수를 그대로 보여주면 정밀해 보여서 등급만 표시한다. */
function SuitabilityTag({ suitability }: { suitability: string }) {
  const map: Record<string, { label: string; fg: string; bg: string }> = {
    recommended: { label: '추천', fg: colors.success, bg: colors.successSoft },
    possible: { label: '가능', fg: colors.primary, bg: colors.primarySoft },
    low_information: {
      label: '정보 부족',
      fg: colors.textTertiary,
      bg: colors.background,
    },
  };
  const tag = map[suitability] || map.low_information;
  return (
    <View style={[styles.suitability, { backgroundColor: tag.bg }]}>
      <Text style={[styles.suitabilityText, { color: tag.fg }]}>{tag.label}</Text>
    </View>
  );
}

const styles = StyleSheet.create({
  aiBadge: {
    fontSize: 11,
    fontWeight: '700',
    color: colors.primary,
    marginBottom: spacing(1.5),
  },
  card: {
    backgroundColor: colors.card,
    borderRadius: radius.md,
    padding: spacing(3.5),
    marginBottom: spacing(2),
    ...shadow,
  },
  headerRow: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    justifyContent: 'space-between',
    gap: spacing(2),
  },
  name: { flex: 1, fontSize: 14, fontWeight: '700', color: colors.text },
  suitability: {
    paddingHorizontal: spacing(2),
    paddingVertical: 2,
    borderRadius: radius.full,
  },
  suitabilityText: { fontSize: 11, fontWeight: '700' },
  address: {
    fontSize: 12,
    color: colors.textSecondary,
    marginTop: 3,
    lineHeight: 17,
  },
  tagRow: { flexDirection: 'row', gap: spacing(1.5), marginTop: spacing(1.5) },
  tag: {
    fontSize: 11,
    fontWeight: '600',
    color: colors.primaryDark,
    backgroundColor: colors.primarySoft,
    paddingHorizontal: spacing(2),
    paddingVertical: 2,
    borderRadius: radius.sm,
    overflow: 'hidden',
  },
  reason: {
    fontSize: 11,
    color: colors.textTertiary,
    marginTop: spacing(1.5),
    lineHeight: 16,
  },
  notice: {
    fontSize: 11,
    color: colors.warn,
    marginTop: spacing(1.5),
    lineHeight: 16,
  },
  noticeStrong: {
    fontSize: 11,
    fontWeight: '700',
    color: colors.danger,
    marginTop: spacing(1),
    lineHeight: 16,
  },
  actions: { flexDirection: 'row', gap: spacing(2), marginTop: spacing(2.5) },
  callButton: {
    flex: 1,
    backgroundColor: colors.primary,
    borderRadius: radius.sm,
    paddingVertical: spacing(2.5),
    alignItems: 'center',
  },
  callText: { color: '#fff', fontWeight: '700', fontSize: 13 },
  emailButton: {
    paddingHorizontal: spacing(3.5),
    justifyContent: 'center',
    borderRadius: radius.sm,
    borderWidth: 1,
    borderColor: colors.primary,
  },
  emailText: { color: colors.primary, fontWeight: '700', fontSize: 13 },
  callDisabled: { backgroundColor: colors.background },
  callDisabledText: { color: colors.textTertiary, fontWeight: '600', fontSize: 13 },
  sourceButton: {
    paddingHorizontal: spacing(3.5),
    justifyContent: 'center',
    borderRadius: radius.sm,
    borderWidth: 1,
    borderColor: colors.border,
  },
  sourceText: { color: colors.textSecondary, fontWeight: '600', fontSize: 13 },
});
