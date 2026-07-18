import React, { useState } from 'react';
import {
  KeyboardAvoidingView,
  ScrollView,
  StyleSheet,
  Text,
  TextInput,
  TouchableOpacity,
  View,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { useAlert } from '../components/AlertProvider';
import { Button } from '../components/ui';
import { useAuth } from '../state/AuthContext';
import { colors, radius, shadow, spacing } from '../theme';

/** 이메일/비밀번호 로그인 · 회원가입 (비밀번호 찾기 없음) */
export default function LoginScreen() {
  const { login, signup } = useAuth();
  const showAlert = useAlert();
  const [mode, setMode] = useState<'login' | 'signup'>('login');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [busy, setBusy] = useState(false);

  const isLogin = mode === 'login';

  const onSubmit = async () => {
    if (busy) {
      return; // 키보드 제출/버튼 연타로 인한 중복 요청 방지
    }
    if (!email.trim() || !password) {
      showAlert(
        isLogin ? '로그인' : '회원가입',
        '이메일과 비밀번호를 입력해 주세요.',
      );
      return;
    }
    setBusy(true);
    try {
      if (isLogin) {
        await login(email, password);
      } else {
        await signup(email, password);
      }
      // 성공하면 AuthContext 의 user 가 채워져 자동으로 메인 화면으로 전환된다.
    } catch (e) {
      showAlert(
        isLogin ? '로그인 실패' : '회원가입 실패',
        e instanceof Error ? e.message : '요청에 실패했어요.',
      );
    } finally {
      setBusy(false);
    }
  };

  return (
    <SafeAreaView style={styles.safe}>
      {/* edge-to-edge(Android)에서 adjustResize 미동작 → padding 으로 키보드 회피 */}
      <KeyboardAvoidingView style={styles.flex} behavior="padding">
        <ScrollView
          contentContainerStyle={styles.container}
          keyboardShouldPersistTaps="handled">
          {/* 브랜드 */}
          <View style={styles.brand}>
            <Text style={styles.logo}>🐾</Text>
            <Text style={styles.appName}>멍냥케어</Text>
            <Text style={styles.tagline}>
              반려동물 건강관리 AI 에이전트
            </Text>
          </View>

          {/* 로그인/회원가입 카드 */}
          <View style={styles.card}>
            <Text style={styles.cardTitle}>
              {isLogin ? '로그인' : '회원가입'}
            </Text>

            <Text style={styles.label}>이메일</Text>
            <TextInput
              style={styles.input}
              value={email}
              onChangeText={setEmail}
              placeholder="you@example.com"
              placeholderTextColor={colors.textTertiary}
              keyboardType="email-address"
              autoCapitalize="none"
              autoCorrect={false}
            />

            <Text style={styles.label}>비밀번호</Text>
            <TextInput
              style={styles.input}
              value={password}
              onChangeText={setPassword}
              placeholder={isLogin ? '비밀번호' : '4자 이상'}
              placeholderTextColor={colors.textTertiary}
              secureTextEntry
              autoCapitalize="none"
              onSubmitEditing={onSubmit}
              returnKeyType="done"
            />

            <Button
              title={isLogin ? '로그인' : '가입하기'}
              onPress={onSubmit}
              loading={busy}
              style={styles.submit}
            />

            {/* 모드 전환 */}
            <TouchableOpacity
              style={styles.switchRow}
              onPress={() => setMode(isLogin ? 'signup' : 'login')}>
              <Text style={styles.switchText}>
                {isLogin ? '계정이 없으신가요? ' : '이미 계정이 있으신가요? '}
                <Text style={styles.switchLink}>
                  {isLogin ? '회원가입' : '로그인'}
                </Text>
              </Text>
            </TouchableOpacity>
          </View>

          {/* 데모 계정 안내 */}
          <Text style={styles.demoHint}>
            데모 계정 · demo@petcare.ai / demo1234
          </Text>
        </ScrollView>
      </KeyboardAvoidingView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  safe: { flex: 1, backgroundColor: colors.background },
  flex: { flex: 1 },
  container: {
    flexGrow: 1,
    justifyContent: 'center',
    padding: spacing(6),
  },
  brand: { alignItems: 'center', marginBottom: spacing(6) },
  logo: { fontSize: 44 },
  appName: {
    fontSize: 26,
    fontWeight: '800',
    color: colors.text,
    marginTop: spacing(2),
  },
  tagline: {
    fontSize: 13,
    color: colors.textSecondary,
    marginTop: spacing(1),
  },
  card: {
    backgroundColor: colors.card,
    borderRadius: radius.lg,
    padding: spacing(5),
    ...shadow,
  },
  cardTitle: {
    fontSize: 18,
    fontWeight: '800',
    color: colors.text,
    marginBottom: spacing(4),
  },
  label: {
    fontSize: 12,
    fontWeight: '600',
    color: colors.textSecondary,
    marginBottom: spacing(1.5),
  },
  input: {
    borderWidth: 1,
    borderColor: colors.border,
    borderRadius: radius.md,
    paddingHorizontal: spacing(3),
    paddingVertical: spacing(2.5),
    fontSize: 14,
    color: colors.text,
    backgroundColor: colors.background,
    marginBottom: spacing(3.5),
  },
  submit: { marginTop: spacing(1) },
  switchRow: { alignItems: 'center', marginTop: spacing(4) },
  switchText: { fontSize: 13, color: colors.textSecondary },
  switchLink: { color: colors.primary, fontWeight: '700' },
  demoHint: {
    textAlign: 'center',
    fontSize: 12,
    color: colors.textTertiary,
    marginTop: spacing(4),
  },
});
