/**
 * PetCare AI — 멍냥케어
 * 반려동물 건강관리 AI Agent 서비스 (로컬 MVP)
 *
 * FE-PR-001 임시 버전 — FE-PR-003에서 Provider + 내비게이션이 연결된
 * 최종 구조로 교체된다.
 */

import React from 'react';
import { StatusBar, StyleSheet, Text, View } from 'react-native';
import { SafeAreaProvider } from 'react-native-safe-area-context';
import { colors } from './src/theme';

function App() {
  return (
    <SafeAreaProvider>
      <StatusBar barStyle="dark-content" backgroundColor={colors.background} />
      <View style={styles.center}>
        <Text style={styles.title}>멍냥케어</Text>
        <Text style={styles.caption}>
          FE 기반 설정 완료 — 기능 화면은 이후 PR에서 추가
        </Text>
      </View>
    </SafeAreaProvider>
  );
}

const styles = StyleSheet.create({
  center: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: colors.background,
  },
  title: { fontSize: 24, fontWeight: '800', color: colors.text },
  caption: { marginTop: 8, fontSize: 13, color: colors.textSecondary },
});

export default App;
