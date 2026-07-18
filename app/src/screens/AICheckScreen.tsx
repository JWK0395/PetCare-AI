import { useNavigation } from '@react-navigation/native';
import type { NativeStackNavigationProp } from '@react-navigation/native-stack';
import React, { useEffect, useRef, useState } from 'react';
import {
  KeyboardAvoidingView,
  Linking,
  ScrollView,
  StyleSheet,
  Text,
  TextInput,
  TouchableOpacity,
  View,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { aiCheck, getHospitals } from '../api/client';
import type { AICheckResponse, ChatMessage, Hospital } from '../api/types';
import ChatHistoryModal from '../components/ChatHistoryModal';
import { Badge } from '../components/ui';
import type { RootStackParamList } from '../navigation/types';
import { useChat, type RenderItem } from '../state/ChatContext';
import { usePet } from '../state/PetContext';
import { colors, radius, riskColors, shadow, spacing } from '../theme';
import { deriveSymptomSummary } from '../utils/symptoms';

// 채팅 아이템 키 — 모듈 스코프 카운터라 화면이 언마운트돼도 충돌 없이 이어진다.
let keyCounter = 0;
const nextKey = () => `item-${keyCounter++}`;

export default function AICheckScreen() {
  const { pet } = usePet();
  const navigation =
    useNavigation<NativeStackNavigationProp<RootStackParamList>>();
  const { chat, setChat, resetChat } = useChat();
  const { messages, items, hospitals, sessionId } = chat;
  const [input, setInput] = useState('');
  const [sending, setSending] = useState(false);
  const [historyVisible, setHistoryVisible] = useState(false);
  const scrollRef = useRef<ScrollView>(null);

  // 반려동물을 바꾸면 이전 아이의 대화는 접고 새로 시작한다.
  useEffect(() => {
    if (pet && chat.petId !== null && chat.petId !== pet.id) {
      resetChat();
    }
  }, [pet, chat.petId, resetChat]);

  useEffect(() => {
    scrollRef.current?.scrollToEnd({ animated: true });
  }, [items]);

  const reset = () => resetChat();

  const onSend = async () => {
    if (!pet || !input.trim() || sending) {
      return;
    }
    const text = input.trim();
    setInput('');
    const petId = pet.id;
    const priorMessages = messages;
    const newMessages: ChatMessage[] = [
      ...priorMessages,
      { role: 'user', content: text },
    ];
    setChat(c => ({
      ...c,
      petId,
      messages: newMessages,
      items: [...c.items, { type: 'user', text, key: nextKey() }],
    }));
    setSending(true);
    // 요청 중 '새 체크'·펫 전환·로그아웃으로 리셋되면 늦게 도착한 응답을 버린다.
    const gen = chat.generation;

    try {
      const response = await aiCheck(petId, newMessages, sessionId);
      const newItems: RenderItem[] = [];

      if (response.risk_level === 'emergency') {
        newItems.push({ type: 'emergency', response, key: nextKey() });
        getHospitals(true)
          .then(list =>
            setChat(c => (c.generation !== gen ? c : { ...c, hospitals: list })),
          )
          .catch(() => {});
      } else {
        if (response.trend_summary && priorMessages.length === 0) {
          newItems.push({
            type: 'trend',
            text: `최근 30일 기록 조회 — ${response.trend_summary}`,
            key: nextKey(),
          });
        }
        if (response.followup_question) {
          newItems.push({
            type: 'assistant',
            text: response.reply,
            key: nextKey(),
          });
          newItems.push({
            type: 'assistant',
            text: response.followup_question,
            key: nextKey(),
          });
        } else {
          newItems.push({ type: 'result', response, key: nextKey() });
        }
      }

      const assistantContent = [response.reply, response.followup_question || '']
        .filter(Boolean)
        .join('\n');
      setChat(c =>
        c.generation !== gen
          ? c
          : {
              ...c,
              sessionId: response.session_id ?? c.sessionId,
              messages: [
                ...newMessages,
                { role: 'assistant', content: assistantContent },
              ],
              items: [...c.items, ...newItems],
            },
      );
    } catch (e) {
      setChat(c =>
        c.generation !== gen
          ? c
          : {
              ...c,
              items: [
                ...c.items,
                {
                  type: 'assistant',
                  text: `⚠️ ${e instanceof Error ? e.message : '연결 오류'}`,
                  key: nextKey(),
                },
              ],
            },
      );
    } finally {
      setSending(false);
    }
  };

  const lastUserText =
    [...messages].reverse().find(m => m.role === 'user')?.content || '';

  return (
    <SafeAreaView style={styles.safe} edges={['top']}>
      {/* 헤더 */}
      <View style={styles.header}>
        <View>
          <Text style={styles.title}>AI 상태 체크</Text>
          <Text style={styles.subtitle}>
            {pet ? `${pet.name} · 최근 30일 기록 기반` : '반려동물 미등록'}
          </Text>
        </View>
        <View style={styles.headerActions}>
          <TouchableOpacity onPress={() => setHistoryVisible(true)}>
            <Text style={styles.resetLink}>지난 대화</Text>
          </TouchableOpacity>
          {items.length > 0 ? (
            <TouchableOpacity onPress={reset}>
              <Text style={styles.resetLink}>새 체크</Text>
            </TouchableOpacity>
          ) : null}
        </View>
      </View>

      {/* edge-to-edge(Android)에서 adjustResize 미동작 → padding 으로 입력바를 키보드 위로 */}
      <KeyboardAvoidingView style={styles.flex} behavior="padding">
        <ScrollView
          ref={scrollRef}
          style={styles.flex}
          contentContainerStyle={styles.chatContainer}
          keyboardShouldPersistTaps="handled">
          {items.length === 0 ? (
            <View style={styles.emptyChat}>
              <Text style={styles.emptyEmoji}>🩺</Text>
              <Text style={styles.emptyText}>
                증상이나 걱정되는 상태를 입력하면{'\n'}최근 30일 기록과
                비교해서 알려드려요.
              </Text>
            </View>
          ) : null}

          {items.map(item => {
            switch (item.type) {
              case 'user':
                return (
                  <View key={item.key} style={styles.userBubble}>
                    <Text style={styles.userText}>{item.text}</Text>
                  </View>
                );
              case 'assistant':
                return (
                  <View key={item.key} style={styles.aiBubble}>
                    <Text style={styles.aiText}>{item.text}</Text>
                  </View>
                );
              case 'trend':
                return (
                  <View key={item.key} style={styles.trendChip}>
                    <Text style={styles.trendText}>{item.text}</Text>
                  </View>
                );
              case 'result':
                return (
                  <ResultCard
                    key={item.key}
                    response={item.response}
                    onCreateSummary={() =>
                      navigation.navigate('Summary', {
                        riskLevel: item.response.risk_level,
                      })
                    }
                  />
                );
              case 'emergency':
                return (
                  <EmergencyCard
                    key={item.key}
                    response={item.response}
                    hospitals={hospitals}
                    onSendEmail={hospitalId =>
                      navigation.navigate('EmergencyEmail', {
                        symptomSummary: deriveSymptomSummary(lastUserText),
                        hospitalId,
                      })
                    }
                  />
                );
            }
          })}
          {sending ? (
            <View style={styles.aiBubble}>
              <Text style={styles.aiText}>확인 중...</Text>
            </View>
          ) : null}
        </ScrollView>

        {/* 입력 바 */}
        <View style={styles.inputBar}>
          <TextInput
            style={styles.input}
            placeholder="증상이나 궁금한 점을 입력하세요"
            placeholderTextColor={colors.textTertiary}
            value={input}
            onChangeText={setInput}
            editable={!!pet && !sending}
            onSubmitEditing={onSend}
            returnKeyType="send"
          />
          <TouchableOpacity
            style={[styles.sendButton, (!input.trim() || sending) && styles.sendDisabled]}
            onPress={onSend}
            disabled={!input.trim() || sending}>
            <Text style={styles.sendIcon}>➤</Text>
          </TouchableOpacity>
        </View>
      </KeyboardAvoidingView>

      {/* 지난 대화 팝업 */}
      {pet ? (
        <ChatHistoryModal
          visible={historyVisible}
          petId={pet.id}
          onClose={() => setHistoryVisible(false)}
        />
      ) : null}
    </SafeAreaView>
  );
}

/** 일반 경로 결과 카드 — 위험도 + 근거 + 요약 CTA */
function ResultCard({
  response,
  onCreateSummary,
}: {
  response: AICheckResponse;
  onCreateSummary: () => void;
}) {
  const risk = riskColors[response.risk_level] || riskColors.observe;
  return (
    <View style={styles.resultCard}>
      <Badge
        label={`${risk.label}${
          response.risk_level === 'consult' ? ' 권장' : ''
        }`}
        fg={risk.fg}
        bg={risk.bg}
      />
      <Text style={styles.resultReply}>{response.reply}</Text>
      {response.reasons.map(reason => (
        <Text key={reason} style={styles.reasonLine}>
          · {reason}
        </Text>
      ))}
      {response.evidence ? (
        <Text style={styles.evidence}>근거 · {response.evidence}</Text>
      ) : null}
      {response.can_generate_summary ? (
        <TouchableOpacity style={styles.summaryCta} onPress={onCreateSummary}>
          <Text style={styles.summaryCtaText}>병원 전달용 요약 만들기 →</Text>
        </TouchableOpacity>
      ) : null}
    </View>
  );
}

/** 응급 경로 카드 — 병원 목록 + 이메일 + 이동 중 대처 */
function EmergencyCard({
  response,
  hospitals,
  onSendEmail,
}: {
  response: AICheckResponse;
  hospitals: Hospital[];
  onSendEmail: (hospitalId: number) => void;
}) {
  return (
    <View style={styles.emergencyWrap}>
      <View style={styles.emergencyBanner}>
        <Text style={styles.emergencyBannerText}>
          🚨 응급 징후 — 지금 병원에 연락하세요
        </Text>
      </View>

      <Text style={styles.emergencySectionTitle}>주변 24시 동물병원</Text>
      {hospitals.map(hospital => (
        <View key={hospital.id} style={styles.hospitalCard}>
          <View>
            <Text style={styles.hospitalName}>{hospital.name}</Text>
            <Text style={styles.hospitalMeta}>
              <Text style={styles.statusDot}>● </Text>
              {hospital.status}
              {hospital.distance_km ? ` · ${hospital.distance_km}km` : ''}
              {hospital.features ? ` · ${hospital.features}` : ''}
            </Text>
          </View>
          <View style={styles.hospitalActions}>
            <TouchableOpacity
              style={styles.emailButtonSmall}
              onPress={() => onSendEmail(hospital.id)}>
              <Text style={styles.emailButtonSmallText}>이메일</Text>
            </TouchableOpacity>
            <TouchableOpacity
              style={styles.callButton}
              onPress={() => Linking.openURL(`tel:${hospital.phone}`)}>
              <Text style={styles.callButtonText}>전화</Text>
            </TouchableOpacity>
          </View>
        </View>
      ))}

      {response.transit_guidance.length > 0 ? (
        <View style={styles.guidanceBox}>
          <Text style={styles.guidanceTitle}>이동 중에는 이렇게 해주세요</Text>
          <Text style={styles.guidanceText}>
            {response.transit_guidance.join(' · ')}
          </Text>
        </View>
      ) : null}
    </View>
  );
}

const styles = StyleSheet.create({
  safe: { flex: 1, backgroundColor: colors.background },
  flex: { flex: 1 },
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingHorizontal: spacing(5),
    paddingTop: spacing(3),
    paddingBottom: spacing(3),
    backgroundColor: colors.background,
  },
  title: { fontSize: 20, fontWeight: '800', color: colors.text },
  subtitle: { fontSize: 12, color: colors.textSecondary, marginTop: 2 },
  headerActions: { flexDirection: 'row', gap: spacing(4) },
  resetLink: { color: colors.primary, fontWeight: '600', fontSize: 13 },
  sendDisabled: { opacity: 0.4 },
  chatContainer: { padding: spacing(4), paddingBottom: spacing(6) },
  emptyChat: { alignItems: 'center', marginTop: spacing(16) },
  emptyEmoji: { fontSize: 40, marginBottom: spacing(3) },
  emptyText: {
    textAlign: 'center',
    color: colors.textSecondary,
    fontSize: 13,
    lineHeight: 20,
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
  trendChip: {
    alignSelf: 'center',
    backgroundColor: colors.primarySoft,
    borderRadius: radius.full,
    paddingHorizontal: spacing(3.5),
    paddingVertical: spacing(1.5),
    marginBottom: spacing(2.5),
  },
  trendText: { color: colors.primaryDark, fontSize: 12, fontWeight: '600' },
  resultCard: {
    backgroundColor: colors.card,
    borderRadius: radius.lg,
    padding: spacing(4),
    marginBottom: spacing(2.5),
    ...shadow,
  },
  resultReply: {
    fontSize: 16,
    fontWeight: '700',
    color: colors.text,
    marginTop: spacing(2.5),
    marginBottom: spacing(2),
  },
  reasonLine: {
    fontSize: 13,
    color: colors.textSecondary,
    lineHeight: 20,
  },
  evidence: {
    fontSize: 11,
    color: colors.textTertiary,
    marginTop: spacing(2.5),
  },
  summaryCta: {
    marginTop: spacing(3.5),
    backgroundColor: colors.primary,
    borderRadius: radius.md,
    paddingVertical: spacing(3),
    alignItems: 'center',
  },
  summaryCtaText: { color: '#fff', fontWeight: '700', fontSize: 14 },
  emergencyWrap: { marginBottom: spacing(2.5) },
  emergencyBanner: {
    backgroundColor: colors.danger,
    borderRadius: radius.md,
    padding: spacing(3.5),
    marginBottom: spacing(3),
  },
  emergencyBannerText: { color: '#fff', fontWeight: '800', fontSize: 15 },
  emergencySectionTitle: {
    fontSize: 13,
    fontWeight: '700',
    color: colors.text,
    marginBottom: spacing(2),
  },
  hospitalCard: {
    backgroundColor: colors.card,
    borderRadius: radius.md,
    padding: spacing(3.5),
    marginBottom: spacing(2),
    ...shadow,
  },
  hospitalName: { fontSize: 14, fontWeight: '700', color: colors.text },
  hospitalMeta: {
    fontSize: 12,
    color: colors.textSecondary,
    marginTop: 2,
  },
  statusDot: { color: colors.success },
  hospitalActions: {
    flexDirection: 'row',
    gap: spacing(2),
    marginTop: spacing(3),
  },
  callButton: {
    flex: 1,
    backgroundColor: colors.primary,
    borderRadius: radius.sm,
    paddingVertical: spacing(2.5),
    alignItems: 'center',
  },
  callButtonText: { color: '#fff', fontWeight: '700', fontSize: 13 },
  emailButtonSmall: {
    flex: 1,
    backgroundColor: colors.card,
    borderWidth: 1,
    borderColor: colors.danger,
    borderRadius: radius.sm,
    paddingVertical: spacing(2.5),
    alignItems: 'center',
  },
  emailButtonSmallText: { color: colors.danger, fontWeight: '700', fontSize: 13 },
  guidanceBox: {
    backgroundColor: colors.dangerSoft,
    borderRadius: radius.md,
    padding: spacing(3.5),
    marginTop: spacing(3),
  },
  guidanceTitle: {
    fontSize: 12,
    fontWeight: '700',
    color: colors.dangerDark,
    marginBottom: spacing(1),
  },
  guidanceText: { fontSize: 13, color: colors.dangerDark, lineHeight: 19 },
  inputBar: {
    flexDirection: 'row',
    alignItems: 'center',
    padding: spacing(3),
    backgroundColor: colors.card,
    borderTopWidth: StyleSheet.hairlineWidth,
    borderTopColor: colors.border,
    gap: spacing(2),
  },
  input: {
    flex: 1,
    backgroundColor: colors.background,
    borderRadius: radius.full,
    paddingHorizontal: spacing(4),
    paddingVertical: spacing(2.5),
    fontSize: 14,
    color: colors.text,
  },
  sendButton: {
    width: 40,
    height: 40,
    borderRadius: 20,
    backgroundColor: colors.primary,
    alignItems: 'center',
    justifyContent: 'center',
  },
  sendIcon: { color: '#fff', fontSize: 16 },
});
