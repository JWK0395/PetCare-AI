import type { NativeStackScreenProps } from '@react-navigation/native-stack';
import React, { useCallback, useEffect, useState } from 'react';
import { ScrollView, StyleSheet, Text, View } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import {
  composeEmergencyEmail,
  getHospitals,
  sendEmergencyEmail,
} from '../api/client';
import type {
  EmailHospitalTarget,
  EmergencyEmail,
  Hospital,
} from '../api/types';
import { openEmailApp } from '../native/share';
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
  // 병원은 두 경로로 온다: DB 등록 병원(id) 또는 AI 가 검색으로 찾은 병원(이름·연락처).
  // 둘 다 없어도 초안은 만들어진다 — 수신 주소는 메일 앱에서 보호자가 넣는다.
  const target: EmailHospitalTarget = {
    hospitalId: route.params?.hospitalId ?? null,
    name: route.params?.hospitalName ?? null,
    email: route.params?.hospitalEmail ?? null,
    phone: route.params?.hospitalPhone ?? null,
  };
  const [hospitals, setHospitals] = useState<Hospital[]>([]);
  const [email, setEmail] = useState<EmergencyEmail | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [sending, setSending] = useState(false);

  const compose = useCallback(
    async (hospital: EmailHospitalTarget) => {
      if (!pet) {
        return;
      }
      try {
        setEmail(await composeEmergencyEmail(pet.id, hospital, symptomSummary));
      } catch (e) {
        setError(e instanceof Error ? e.message : '이메일 작성 실패');
      }
    },
    [pet, symptomSummary],
  );

  // route.params 를 의존성으로 펼쳐 쓴다 — `target` 은 렌더마다 새 객체라
  // 그대로 넣으면 매 렌더 초안을 다시 만든다.
  const { hospitalId, name, email: targetEmail, phone } = target;
  useEffect(() => {
    getHospitals(true)
      .then(setHospitals)
      .catch(() => {});
    compose({ hospitalId, name, email: targetEmail, phone });
  }, [compose, hospitalId, name, targetEmail, phone]);

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
  const hospitalLabel = hospital?.name || name || '병원 미지정';
  const sent = email.status === 'sent';

  /**
   * 기본 메일 앱을 열어 **사용자가 직접** 전송하게 한다.
   *
   * 앱이 대신 보내지 않는 이유: 오발송이 곧 의료 정보 유출이고, 수신 주소가
   * 웹 검색으로 추정된 값일 수 있어 사람이 한 번 확인해야 한다.
   * 서버의 send 처리는 "전달함"을 남기는 기록일 뿐이라 메일 앱을 연 뒤에만 호출한다.
   */
  const onSend = () => {
    showAlert(
      '메일 앱으로 열기',
      email.to_email
        ? `${hospitalLabel} 앞으로 메일을 작성합니다.\n\n메일 앱에서 내용을 확인하고 직접 전송해 주세요.`
        : '병원 이메일 주소를 찾지 못했어요.\n\n메일 앱이 열리면 받는 사람을 직접 입력해 주세요.',
      [
        { text: '취소', style: 'cancel' },
        {
          text: '메일 앱 열기',
          onPress: async () => {
            setSending(true);
            try {
              const opened = await openEmailApp(
                email.to_email,
                email.subject,
                email.body,
              );
              if (!opened) {
                showAlert(
                  '메일 앱을 열 수 없어요',
                  '기기에 메일 앱이 없거나 열기에 실패했어요. 아래 문서 내용을 복사해 사용해 주세요.',
                );
                return;
              }
              setEmail(await sendEmergencyEmail(email.id));
            } catch (e) {
              showAlert('오류', e instanceof Error ? e.message : '메일 앱 열기 실패');
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
          <Text style={styles.hospitalName}>{hospitalLabel}</Text>
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
          {/* AI 검색 병원은 DB 에 없어 distance/status 가 없다. 대신 전화번호를 보여준다. */}
          {!hospital && phone ? (
            <Text style={styles.hospitalMeta}>{phone}</Text>
          ) : null}
          <Text style={styles.emailAddr}>
            {email.to_email || '이메일 주소 미확인 — 메일 앱에서 직접 입력'}
          </Text>
        </Card>

        {/* 제목 */}
        <Card style={styles.block}>
          <Text style={styles.blockLabel}>제목</Text>
          <Text style={styles.subjectText}>{email.subject}</Text>
        </Card>

        {/* 문서 내용 — 병원 전달용 요약과 같은 4섹션 구조를 **메일 본문 텍스트**로 보낸다.
            (PDF 첨부가 아니다: mailto: 는 첨부를 지원하지 않는다) */}
        <Card style={styles.block}>
          <Text style={styles.blockLabel}>메일 본문에 들어갈 내용</Text>
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
              ✓ 메일 앱으로 전달함 ({email.sent_at ? stampDateTime(email.sent_at) : ''})
            </Text>
          </View>
        ) : (
          <Button
            title="메일 앱으로 열기"
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
