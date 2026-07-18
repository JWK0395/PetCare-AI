import React, { useCallback, useEffect, useState } from 'react';
import {
  ScrollView,
  StyleSheet,
  Text,
  TouchableOpacity,
  View,
} from 'react-native';
import {
  deleteAISession,
  getAISession,
  getAISessions,
} from '../api/client';
import type { AISessionDetail, AISessionSummary } from '../api/types';
import { colors, radius, riskColors, shadow, spacing } from '../theme';
import { stampDateTime } from '../utils/date';
import { useAlert } from './AlertProvider';
import AppModal from './AppModal';
import { Badge, DeleteButton } from './ui';

/** AI 체크 — 지난 대화를 보고 삭제하는 팝업 */
export default function ChatHistoryModal({
  visible,
  petId,
  onClose,
}: {
  visible: boolean;
  petId: number;
  onClose: () => void;
}) {
  const showAlert = useAlert();
  const [sessions, setSessions] = useState<AISessionSummary[]>([]);
  const [detail, setDetail] = useState<AISessionDetail | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);

  const load = useCallback(async () => {
    try {
      setSessions(await getAISessions(petId));
      setLoadError(null);
    } catch (e) {
      // 조용히 삼키면 "저장된 대화가 없어요" 로 오인된다 — 오류를 표시한다.
      setLoadError(e instanceof Error ? e.message : '대화 목록을 불러오지 못했어요');
    }
  }, [petId]);

  useEffect(() => {
    if (visible) {
      load();
      setDetail(null);
    }
  }, [visible, load]);

  const openSession = async (id: number) => {
    try {
      setDetail(await getAISession(id));
    } catch (e) {
      showAlert('오류', e instanceof Error ? e.message : '대화를 불러오지 못했어요');
    }
  };

  const confirmDelete = (id: number, title: string) => {
    showAlert('대화 삭제', `"${title || 'AI 상태 체크'}" 대화를 삭제할까요?`, [
      { text: '취소', style: 'cancel' },
      {
        text: '삭제',
        style: 'destructive',
        onPress: async () => {
          try {
            await deleteAISession(id);
            setDetail(null);
            await load();
          } catch (e) {
            showAlert('오류', e instanceof Error ? e.message : '삭제 실패');
          }
        },
      },
    ]);
  };

  return (
    <AppModal
      visible={visible}
      size="large"
      title={detail ? '대화 내용' : '지난 대화'}
      onClose={onClose}
      headerLeft={
        detail ? (
          <TouchableOpacity
            onPress={() => setDetail(null)}
            hitSlop={{ top: 8, bottom: 8, left: 8, right: 8 }}>
            <Text style={styles.backLink}>‹ 목록</Text>
          </TouchableOpacity>
        ) : undefined
      }>
      <ScrollView style={styles.scroll} contentContainerStyle={styles.body}>
        {detail ? (
          <Transcript detail={detail} />
        ) : loadError ? (
          <Text style={styles.empty}>{loadError}</Text>
        ) : sessions.length === 0 ? (
          <Text style={styles.empty}>저장된 대화가 없어요.</Text>
        ) : (
          sessions.map(s => (
            <View key={s.id} style={styles.listCard}>
              <TouchableOpacity
                style={styles.listMain}
                onPress={() => openSession(s.id)}
                activeOpacity={0.7}>
                <Text style={styles.listTitle} numberOfLines={1}>
                  {s.title || 'AI 상태 체크'}
                </Text>
                <Text style={styles.listMeta}>{stampDateTime(s.updated_at)}</Text>
              </TouchableOpacity>
              <DeleteButton onPress={() => confirmDelete(s.id, s.title)} />
            </View>
          ))
        )}
      </ScrollView>
    </AppModal>
  );
}

function Transcript({ detail }: { detail: AISessionDetail }) {
  return (
    <View>
      <Text style={styles.transcriptDate}>
        {stampDateTime(detail.created_at)}
      </Text>
      {detail.messages.map((message, index) => {
        if (message.role === 'user') {
          return (
            <View key={index} style={styles.userBubble}>
              <Text style={styles.userText}>{message.content}</Text>
            </View>
          );
        }
        const meta = message.meta;
        // 응급 턴은 배너로 재현
        if (meta?.risk_level === 'emergency') {
          return (
            <View key={index}>
              <View style={styles.emergencyBanner}>
                <Text style={styles.emergencyText}>
                  🚨 응급 징후 — 지금 병원에 연락하세요
                </Text>
              </View>
              {meta.transit_guidance.length > 0 ? (
                <View style={styles.guidanceBox}>
                  <Text style={styles.guidanceText}>
                    {meta.transit_guidance.join(' · ')}
                  </Text>
                </View>
              ) : null}
            </View>
          );
        }
        // 판단 결과 턴은 결과 카드로 재현
        if (meta && (meta.reasons.length > 0 || !meta.followup_question)) {
          const risk = riskColors[meta.risk_level] || riskColors.observe;
          return (
            <View key={index} style={styles.resultCard}>
              <Badge label={risk.label} fg={risk.fg} bg={risk.bg} />
              <Text style={styles.resultReply}>
                {message.content.split('\n')[0]}
              </Text>
              {meta.reasons.map(reason => (
                <Text key={reason} style={styles.reasonLine}>
                  · {reason}
                </Text>
              ))}
              {meta.evidence ? (
                <Text style={styles.evidence}>근거 · {meta.evidence}</Text>
              ) : null}
            </View>
          );
        }
        return (
          <View key={index} style={styles.aiBubble}>
            <Text style={styles.aiText}>{message.content}</Text>
          </View>
        );
      })}
    </View>
  );
}

const styles = StyleSheet.create({
  scroll: { flex: 1 },
  body: { paddingHorizontal: spacing(4), paddingBottom: spacing(8) },
  backLink: { color: colors.primary, fontSize: 13, fontWeight: '700' },
  empty: {
    textAlign: 'center',
    color: colors.textTertiary,
    paddingVertical: spacing(10),
    fontSize: 13,
  },
  listCard: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: colors.card,
    borderRadius: radius.lg,
    padding: spacing(4),
    marginBottom: spacing(2.5),
    gap: spacing(2),
    ...shadow,
  },
  listMain: { flex: 1 },
  listTitle: { fontSize: 14, fontWeight: '700', color: colors.text },
  listMeta: { fontSize: 12, color: colors.textSecondary, marginTop: 2 },
  transcriptDate: {
    textAlign: 'center',
    fontSize: 12,
    color: colors.textTertiary,
    marginBottom: spacing(3),
  },
  userBubble: {
    alignSelf: 'flex-end',
    backgroundColor: colors.primary,
    borderRadius: radius.lg,
    borderBottomRightRadius: radius.sm,
    paddingHorizontal: spacing(3.5),
    paddingVertical: spacing(2.5),
    marginBottom: spacing(2.5),
    maxWidth: '82%',
  },
  userText: { color: '#fff', fontSize: 14, lineHeight: 20 },
  aiBubble: {
    alignSelf: 'flex-start',
    backgroundColor: colors.card,
    borderRadius: radius.lg,
    borderBottomLeftRadius: radius.sm,
    paddingHorizontal: spacing(3.5),
    paddingVertical: spacing(2.5),
    marginBottom: spacing(2.5),
    maxWidth: '82%',
    ...shadow,
  },
  aiText: { color: colors.text, fontSize: 14, lineHeight: 20 },
  resultCard: {
    backgroundColor: colors.card,
    borderRadius: radius.lg,
    padding: spacing(4),
    marginBottom: spacing(2.5),
    ...shadow,
  },
  resultReply: {
    fontSize: 15,
    fontWeight: '700',
    color: colors.text,
    marginTop: spacing(2),
    marginBottom: spacing(1.5),
  },
  reasonLine: { fontSize: 13, color: colors.textSecondary, lineHeight: 20 },
  evidence: { fontSize: 11, color: colors.textTertiary, marginTop: spacing(2) },
  emergencyBanner: {
    backgroundColor: colors.danger,
    borderRadius: radius.md,
    padding: spacing(3.5),
    marginBottom: spacing(2),
  },
  emergencyText: { color: '#fff', fontWeight: '800', fontSize: 14 },
  guidanceBox: {
    backgroundColor: colors.dangerSoft,
    borderRadius: radius.md,
    padding: spacing(3),
    marginBottom: spacing(2.5),
  },
  guidanceText: { fontSize: 13, color: colors.dangerDark },
});
