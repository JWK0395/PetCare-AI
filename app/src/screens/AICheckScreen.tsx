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
import type {
  AICheckResponse,
  ChatMessage,
  Hospital,
  HospitalSuggestion,
} from '../api/types';
import ChatHistoryModal from '../components/ChatHistoryModal';
import CitationList from '../components/CitationList';
import HospitalSuggestionList from '../components/HospitalSuggestionList';
import { Badge } from '../components/ui';
import type { RootStackParamList } from '../navigation/types';
import { useChat, type RenderItem } from '../state/ChatContext';
import { usePet } from '../state/PetContext';
import { colors, radius, riskColors, shadow, spacing } from '../theme';
import { getCurrentRegion } from '../native/location';

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
      // 지역은 **대화당 1회만** 조회한다(ChatContext 캐시). 메시지마다 다시 읽으면
      // 전송이 매번 느려지고 권한 팝업이 반복되는데, 한 대화 중 구·동이 바뀔 일은 없다.
      // undefined = 아직 조회 전 / null = 조회했지만 알아내지 못함(재시도 안 함).
      let region = chat.regionName;
      if (region === undefined) {
        region = (await getCurrentRegion(true).catch(() => null))?.regionName ?? null;
        setChat(c => (c.generation !== gen ? c : { ...c, regionName: region }));
      }

      const response = await aiCheck(petId, newMessages, sessionId, region);
      const newItems: RenderItem[] = [];

      // 화면은 세 갈래뿐이다.
      //   1) 되묻는 중        → 대화 말풍선만
      //   2) 병원 권고(consult) → 요약(텍스트 내보내기)만
      //   3) 응급              → 병원 목록 + 이메일만
      //
      // 판정 완료 여부는 `awaiting_more_info` 로만 판단한다. followup_question 유무로
      // 가르면 안 된다 — 응급 판정이 **끝난** 응답도 병원에 전달할 항목을 함께
      // 물어보기 때문에, 진짜 응급 카드까지 질문으로 처리되어 버린다.
      // 앞선 판정을 되묻는 턴("왜 응급한거죠?")은 **카드를 다시 그리지 않는다.**
      // 카드는 이미 위에 있고, 같은 것이 반복되면 대화가 되지 않는다.
      const conversationalTurn =
        response.awaiting_more_info || response.assessment_turn === false;

      if (conversationalTurn) {
        newItems.push({
          type: 'assistant',
          text: response.followup_question || response.reply,
          key: nextKey(),
        });
      } else if (response.risk_level === 'emergency') {
        newItems.push({ type: 'emergency', response, key: nextKey() });
        getHospitals(true)
          .then(list =>
            setChat(c => (c.generation !== gen ? c : { ...c, hospitals: list })),
          )
          .catch(() => {});
      } else if (response.risk_level === 'consult') {
        newItems.push({ type: 'result', response, key: nextKey() });
      } else {
        // 정상 — 판정 카드를 붙이지 않는다. 걱정할 것이 없다는 답에 위험도 뱃지와
        // 버튼을 다는 것은 화면만 무겁게 한다.
        //
        // **답변과 질문을 같이 그리지 않는다.** 되묻는 중이 아니면 이 답변은 완결된
        // 것이고, 여기에 질문을 덧붙이면 "보양식 어떻게 만들까요?" 에 답을 다 해놓고
        // "언제부터 그런 모습이 나타났나요?" 를 되묻는 화면이 된다(실제로 그랬다).
        newItems.push({
          type: 'assistant',
          text: response.reply,
          key: nextKey(),
        });
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

  // 병원 전달 문서(요약·응급 이메일)에 실을 **대화 원문**.
  //
  // 예전에는 `deriveSymptomSummary()` 라는 정규식이 "호흡곤란 · 중독 의심" 같은
  // 라벨만 뽑아 넘겼다. 그러다 보니 "산책 갔다왔는데 발이 빨개" 처럼 사전에 없는
  // 증상은 통째로 사라지고, 문서의 주호소가 옛 진단명으로 채워졌다.
  //
  // 수의사가 볼 문서에는 보호자가 실제로 한 말이 들어가야 한다. 요약·판단은 AI 가
  // 하고, 여기서는 원문을 그대로 옮긴다.
  const conversationText = messages
    .filter(m => m.role === 'user')
    .map(m => m.content.trim())
    .filter(Boolean)
    .join('\n')
    .slice(0, 1500);

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
                        conversation: conversationText,
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
                        symptomSummary: conversationText,
                        hospitalId,
                      })
                    }
                    onSendEmailToSuggestion={suggestion =>
                      navigation.navigate('EmergencyEmail', {
                        symptomSummary: conversationText,
                        hospitalName: suggestion.name,
                        hospitalEmail: suggestion.email,
                        hospitalPhone: suggestion.phone,
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
      <CitationList citations={response.citations} />
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
  onSendEmailToSuggestion,
}: {
  response: AICheckResponse;
  hospitals: Hospital[];
  onSendEmail: (hospitalId: number) => void;
  onSendEmailToSuggestion: (hospital: HospitalSuggestion) => void;
}) {
  const aiHospitalCount = response.hospitals?.length ?? 0;

  return (
    <View style={styles.emergencyWrap}>
      <View style={styles.emergencyBanner}>
        <Text style={styles.emergencyBannerText}>
          🚨 응급 징후 — 지금 병원에 연락하세요
        </Text>
      </View>

      <Text style={styles.emergencySectionTitle}>주변 24시 동물병원</Text>

      {/* AI 가 실시간 검색으로 찾은 병원을 먼저 보여준다.
          없으면 사용자가 직접 등록해 둔 병원(서버 DB)으로 대체한다.
          둘 다 없을 수 있다 — 그때는 없는 병원을 지어내지 않고 안내만 한다. */}
      <HospitalSuggestionList
        hospitals={response.hospitals}
        onSendEmail={onSendEmailToSuggestion}
      />

      {aiHospitalCount === 0 && hospitals.length === 0 ? (
        <View style={styles.hospitalEmpty}>
          <Text style={styles.hospitalEmptyTitle}>
            주변 병원을 찾지 못했어요
          </Text>
          <Text style={styles.hospitalEmptyText}>
            위치 권한이 꺼져 있거나 검색이 되지 않았어요. 응급 상황이라면 지도
            앱에서 &apos;24시 동물병원&apos;을 검색하거나, 다니던 병원에 바로
            전화해 주세요.
          </Text>
        </View>
      ) : null}

      {aiHospitalCount === 0 &&
        hospitals.map(hospital => (
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
  hospitalEmpty: {
    backgroundColor: colors.card,
    borderRadius: radius.md,
    padding: spacing(3.5),
    marginBottom: spacing(2),
  },
  hospitalEmptyTitle: {
    fontSize: 13,
    fontWeight: '700',
    color: colors.text,
    marginBottom: spacing(1),
  },
  hospitalEmptyText: {
    fontSize: 12,
    color: colors.textSecondary,
    lineHeight: 18,
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
