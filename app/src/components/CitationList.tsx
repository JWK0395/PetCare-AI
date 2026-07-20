import React from 'react';
import { Linking, StyleSheet, Text, TouchableOpacity, View } from 'react-native';
import type { RagCitation } from '../api/types';
import { colors, spacing } from '../theme';

/**
 * RAG 근거 출처 목록 (AI 체크 결과 카드 · 지난 대화 공용).
 *
 * AI(Agent, AGENT_MODE=http)를 연결했을 때만 채워진다. 서버 내장 mock 은
 * citations 를 빈 목록으로 보내므로 이 컴포넌트는 아무것도 그리지 않는다
 * — 즉 AI 미연결 상태의 화면은 기존과 동일하게 유지된다.
 *
 * 출처를 탭하면 원문(Cornell 수의학 자료 등)을 브라우저로 연다. source 가
 * http(s) URL 이 아니면 열지 않는다(잘못된 스킴으로 앱이 죽지 않게).
 */
export default function CitationList({
  citations,
}: {
  citations?: RagCitation[] | null;
}) {
  if (!citations || citations.length === 0) {
    return null;
  }

  const openSource = (url: string) => {
    if (/^https?:\/\//.test(url)) {
      Linking.openURL(url).catch(() => {});
    }
  };

  return (
    <View style={styles.box}>
      <Text style={styles.title}>참고한 자료</Text>
      {citations.map((citation, index) => (
        <TouchableOpacity
          key={`${citation.source}-${index}`}
          activeOpacity={citation.source ? 0.6 : 1}
          onPress={() => openSource(citation.source)}>
          <Text style={styles.name} numberOfLines={2}>
            {index + 1}. {citation.title || '제목 없음'}
          </Text>
          {citation.snippet ? (
            <Text style={styles.snippet} numberOfLines={3}>
              {citation.snippet}
            </Text>
          ) : null}
        </TouchableOpacity>
      ))}
      <Text style={styles.note}>
        이 자료는 참고용이며 수의사의 진단을 대체하지 않습니다.
      </Text>
    </View>
  );
}

const styles = StyleSheet.create({
  box: {
    marginTop: spacing(3),
    paddingTop: spacing(2.5),
    borderTopWidth: StyleSheet.hairlineWidth,
    borderTopColor: colors.border,
    gap: spacing(1.5),
  },
  title: {
    fontSize: 12,
    fontWeight: '700',
    color: colors.textSecondary,
  },
  name: {
    fontSize: 12,
    fontWeight: '600',
    color: colors.primary,
    lineHeight: 18,
  },
  snippet: {
    fontSize: 11,
    color: colors.textTertiary,
    lineHeight: 16,
    marginTop: 2,
  },
  note: {
    fontSize: 10,
    color: colors.textTertiary,
    marginTop: spacing(1),
  },
});
