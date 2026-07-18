import type { NativeStackScreenProps } from '@react-navigation/native-stack';
import React, { useCallback, useEffect, useState } from 'react';
import { ScrollView, Share, StyleSheet, Text, View } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { createSummary, getSummary } from '../api/client';
import type { Summary } from '../api/types';
import { useAlert } from '../components/AlertProvider';
import SummaryDocumentView from '../components/SummaryDocumentView';
import { Badge, Button, Card, FullLoading } from '../components/ui';
import type { RootStackParamList } from '../navigation/types';
import { usePet } from '../state/PetContext';
import { colors, riskColors, spacing } from '../theme';
import { stampDateTime } from '../utils/date';

type Props = NativeStackScreenProps<RootStackParamList, 'Summary'>;

export default function SummaryScreen({ route }: Props) {
  const { pet } = usePet();
  const showAlert = useAlert();
  const { riskLevel, summaryId } = route.params || {};
  const [summary, setSummary] = useState<Summary | null>(null);
  const [error, setError] = useState<string | null>(null);

  // pet 객체 자체를 의존성으로 쓰면 목록 새로고침마다 객체가 바뀌어
  // createSummary 가 중복 POST 되므로 primitive 인 id 만 의존한다.
  const petId = pet?.id ?? null;
  const load = useCallback(async () => {
    try {
      if (summaryId) {
        setSummary(await getSummary(summaryId));
      } else if (petId) {
        setSummary(await createSummary(petId, riskLevel || 'observe'));
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : '요약 생성 실패');
    }
  }, [petId, riskLevel, summaryId]);

  useEffect(() => {
    load();
  }, [load]);

  if (error) {
    return (
      <SafeAreaView style={styles.safe} edges={['top']}>
        <View style={styles.center}>
          <Text style={styles.errorText}>{error}</Text>
        </View>
      </SafeAreaView>
    );
  }

  if (!summary || !pet) {
    return <FullLoading message="병원 전달용 요약을 만드는 중..." />;
  }

  const content = summary.content;
  const created = stampDateTime(summary.created_at);
  const risk = riskColors[summary.risk_level] || riskColors.observe;

  const shareText = [
    content.title,
    '',
    '[1. 문서 정보]',
    `- 생성 일시: ${created}`,
    `- 사용 데이터 기간: ${content.data_period}`,
    '',
    '[2. 반려동물 정보]',
    `- 이름: ${content.pet_name}`,
    `- 종: ${content.species}`,
    `- 품종: ${content.breed}`,
    `- 성별/중성화: ${content.sex_neuter}`,
    `- 나이: ${content.age_label}`,
    `- 현재 체중: ${content.weight}`,
    `- 현재 복용 중인 약: ${content.medications}`,
    `- 알레르기: ${content.allergies}`,
    '',
    '[3. 상태]',
    `- 상태 분류: ${content.risk_label}`,
    '- 확인된 위험 징후:',
    ...(content.risk_signs?.length
      ? content.risk_signs.map(s => `  * ${s}`)
      : ['  * 특이 위험 징후 없음']),
    '',
    '[4. 주호소 및 주요 변화]',
    `- 주호소: ${content.chief_complaint}`,
    `- 주요 변화: ${content.major_changes}`,
    `- 경과: ${content.progress}`,
    content.owner_note ? `- 보호자 메모: ${content.owner_note}` : '',
    '',
    `PetCare AI · 생성 ${created}`,
  ]
    .filter(Boolean)
    .join('\n');

  const onShare = async () => {
    try {
      await Share.share({ message: shareText });
    } catch {
      showAlert('공유', '공유를 열 수 없어요.');
    }
  };

  return (
    <SafeAreaView style={styles.safe} edges={['top']}>
      <ScrollView style={styles.flex} contentContainerStyle={styles.container}>
        <Text style={styles.subtitle}>{pet.name} · {created} 생성</Text>
        <View style={styles.titleRow}>
          <Text style={styles.title}>병원 전달용 요약</Text>
          <Badge label={content.risk_label} fg={risk.fg} bg={risk.bg} />
        </View>

        <Card style={styles.summaryCard}>
          <SummaryDocumentView content={content} createdAt={created} />
        </Card>
      </ScrollView>

      {/* 하단 고정 액션 */}
      <View style={styles.footer}>
        <Button title="PDF 저장" onPress={onShare} />
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
  subtitle: { fontSize: 12, color: colors.textSecondary },
  titleRow: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    marginTop: 2,
    marginBottom: spacing(4),
  },
  title: { fontSize: 22, fontWeight: '800', color: colors.text },
  summaryCard: { marginBottom: spacing(4) },
});
