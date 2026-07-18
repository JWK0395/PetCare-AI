import React from 'react';
import { StyleSheet, Text, View } from 'react-native';
import { colors } from '../theme';

/** FE-PR-003 임시 화면 — FE-PR-006에서 실제 진료 화면으로 교체된다. */
export default function DiagnosisUploadScreen() {
  return (
    <View style={styles.center}>
      <Text style={styles.text}>진료 화면 준비 중 (FE-PR-006)</Text>
    </View>
  );
}

const styles = StyleSheet.create({
  center: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: colors.background,
  },
  text: { color: colors.textSecondary, fontSize: 14 },
});
