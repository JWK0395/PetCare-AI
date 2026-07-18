import type { NativeStackScreenProps } from '@react-navigation/native-stack';
import React, { useCallback, useEffect, useState } from 'react';
import { ScrollView, StyleSheet, Text, View } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import {
  composeEmergencyEmail,
  getHospitals,
  sendEmergencyEmail,
} from '../api/client';
import type { EmergencyEmail, Hospital } from '../api/types';
import { useAlert } from '../components/AlertProvider';
import SummaryDocumentView from '../components/SummaryDocumentView';
import { Badge, Button, Card, FullLoading } from '../components/ui';
import type { RootStackParamList } from '../navigation/types';
import { usePet } from '../state/PetContext';
import { colors, radius, spacing } from '../theme';
import { stampDateTime } from '../utils/date';

type Props = NativeStackScreenProps<RootStackParamList, 'EmergencyEmail'>;

export default function EmergencyEmailScreen({ route }: Props) {
  const { pet } = usePet();
  const showAlert = useAlert();
  const symptomSummary = route.params?.symptomSummary || '응급 증상';
  const hospitalId = route.params?.hospitalId ?? null;
  const [hospitals, setHospitals] = useState<Hospital[]>([]);
  const [email, setEmail] = useState<EmergencyEmail | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [sending, setSending] = useState(false);

  const compose = useCallback(
    async (targetHospitalId: number | null) => {
      if (!pet) {
        return;
      }
      try {
        setEmail(
          await composeEmergencyEmail(pet.id, targetHospitalId, symptomSummary),
        );
      } catch (e) {
        setError(e instanceof Error ? e.message : '이메일 작성 실패');
      }
    },
    [pet, symptomSummary],
  );

  useEffect(() => {
    getHospitals(true)
      .then(setHospitals)
      .catch(() => {});
    compose(hospitalId);
  }, [compose, hospitalId]);

  if (error) {
    return (
      <SafeAreaView style={styles.safe} edges={['top']}>
        <View style={styles.center}>
          <Text style={styles.errorText}>{error}</Text>
        </View>
      </SafeAreaView>
    );
  }

  if (!email || !pet) {
    return <FullLoading message="상태 문서를 준비하는 중..." />;
  }

  const hospital = hospitals.find(h => h.id === email.hospital_id);
  const sent = email.status === 'sent';

  const onSend = () => {
    showAlert(
      '이메일 전송',
      `${hospital?.name || email.to_email} 으로 상태 문서를 전송할까요?\n\n전송 전 내용을 꼭 확인해 주세요.`,
      [
        { text: '취소', style: 'cancel' },
        {
          text: '전송',
          style: 'destructive',
          onPress: async () => {
            setSending(true);
            try {
              setEmail(await sendEmergencyEmail(email.id));
            } catch (e) {
              showAlert('오류', e instanceof Error ? e.message : '전송 실패');
            } finally {
              setSending(false);
            }
          },
        },
      ],
    );
  };

  return (
    <SafeAreaView style={styles.safe} edges={['top']}>
      <ScrollView style={styles.flex} contentContainerStyle={styles.container}>
        <View style={styles.headerRow}>
          <View>
            <Text style={styles.subtitle}>
              {pet.name} · {stampDateTime(email.created_at)}
            </Text>
            <Text style={styles.title}>응급 이메일 전송</Text>
          </View>
          <Badge label="응급" fg={colors.danger} bg={colors.dangerSoft} />
        </View>

        {/* 받는 병원 (전달용 문서는 이 화면에서 수정할 수 없습니다) */}
        <Card style={styles.block}>
          <Text style={styles.blockLabel}>받는 병원</Text>
          <Text style={styles.hospitalName}>
            {hospital?.name || '병원 미지정'}
          </Text>
          {hospital ? (
            <Text style={styles.hospitalMeta}>
              <Text style={styles.statusDot}>● </Text>
              {[
                hospital.status,
                hospital.distance_km != null ? `${hospital.distance_km}km` : null,
                hospital.features,
              ]
                .filter(Boolean)
                .join(' · ')}
            </Text>
          ) : null}
          <Text style={styles.emailAddr}>{email.to_email}</Text>
        </Card>

        {/* 제목 */}
        <Card style={styles.block}>
          <Text style={styles.blockLabel}>제목</Text>
          <Text style={styles.subjectText}>{email.subject}</Text>
        </Card>

        {/* 문서 내용 (병원 전달용 요약과 동일한 4섹션 구조 · PDF로 전송) */}
        <Card style={styles.block}>
          <Text style={styles.blockLabel}>문서 내용 (PDF 첨부)</Text>
          <SummaryDocumentView
            content={email.content}
            createdAt={stampDateTime(email.created_at)}
          />
        </Card>

      </ScrollView>

      {/* 하단 고정 액션 */}
      <View style={styles.footer}>
        {sent ? (
          <View style={styles.sentBanner}>
            <Text style={styles.sentText}>
              ✓ 전송 완료 ({email.sent_at ? stampDateTime(email.sent_at) : ''})
            </Text>
          </View>
        ) : (
          <Button
            title="이메일 전송"
            variant="danger"
            onPress={onSend}
            loading={sending}
          />
        )}
      </View>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  safe: { flex: 1, backgroundColor: colors.background },
  flex: { flex: 1 },
  center: { flex: 1, alignItems: 'center', justifyContent: 'center' },
  errorText: { color: colors.danger, padding: spacing(6), textAlign: 'center' },
  container: { padding: spacing(5), paddingBottom: spacing(4) },
  footer: {
    paddingHorizontal: spacing(5),
    paddingTop: spacing(3),
    paddingBottom: spacing(4),
    borderTopWidth: StyleSheet.hairlineWidth,
    borderTopColor: colors.border,
    backgroundColor: colors.background,
  },
  headerRow: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    justifyContent: 'space-between',
    marginBottom: spacing(4),
  },
  subtitle: { fontSize: 12, color: colors.textSecondary },
  title: {
    fontSize: 22,
    fontWeight: '800',
    color: colors.text,
    marginTop: 2,
  },
  block: { marginBottom: spacing(3) },
  blockLabel: {
    fontSize: 12,
    fontWeight: '700',
    color: colors.textSecondary,
    marginBottom: spacing(2),
  },
  hospitalName: { fontSize: 16, fontWeight: '700', color: colors.text },
  hospitalMeta: { fontSize: 12, color: colors.textSecondary, marginTop: 2 },
  statusDot: { color: colors.success },
  emailAddr: { fontSize: 13, color: colors.primary, marginTop: spacing(1) },
  subjectText: {
    fontSize: 14,
    fontWeight: '600',
    color: colors.text,
    lineHeight: 21,
  },
  sentBanner: {
    backgroundColor: colors.successSoft,
    borderRadius: radius.md,
    paddingVertical: spacing(3.5),
    alignItems: 'center',
  },
  sentText: { color: colors.success, fontWeight: '700' },
});
