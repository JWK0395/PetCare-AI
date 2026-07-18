import React from 'react';
import {
  KeyboardAvoidingView,
  Modal,
  Pressable,
  StyleSheet,
  Text,
  TouchableOpacity,
  View,
} from 'react-native';
import { colors, radius, shadow, spacing } from '../theme';

/** 화면 중앙에 뜨는 공용 팝업.
 *  size="large" 는 고정 큰 크기(높이 80%), "auto" 는 내용에 맞춰 줄어든다. */
export default function AppModal({
  visible,
  title,
  onClose,
  children,
  headerLeft,
  size = 'auto',
}: {
  visible: boolean;
  title: string;
  onClose: () => void;
  children: React.ReactNode;
  headerLeft?: React.ReactNode;
  size?: 'large' | 'auto';
}) {
  return (
    <Modal
      visible={visible}
      transparent
      animationType="fade"
      statusBarTranslucent
      onRequestClose={onClose}>
      {/* edge-to-edge(Android)에서는 adjustResize 가 동작하지 않으므로
          두 플랫폼 모두 padding 으로 키보드 높이만큼 밀어 올린다. */}
      <KeyboardAvoidingView style={styles.flex} behavior="padding">
        <View style={styles.overlay}>
          <Pressable style={styles.backdrop} onPress={onClose} />
          <View style={[styles.card, size === 'large' && styles.cardLarge]}>
            <View style={styles.header}>
              <View style={styles.headerSide}>{headerLeft}</View>
              <Text style={styles.title} numberOfLines={1}>
                {title}
              </Text>
              <TouchableOpacity
                style={styles.headerSide}
                onPress={onClose}
                hitSlop={{ top: 8, bottom: 8, left: 8, right: 8 }}>
                <Text style={styles.close}>✕</Text>
              </TouchableOpacity>
            </View>
            <View style={[styles.body, size === 'large' && styles.bodyLarge]}>
              {children}
            </View>
          </View>
        </View>
      </KeyboardAvoidingView>
    </Modal>
  );
}

const styles = StyleSheet.create({
  flex: { flex: 1 },
  overlay: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    padding: spacing(5),
  },
  backdrop: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    backgroundColor: 'rgba(17, 24, 39, 0.5)',
  },
  card: {
    width: '100%',
    maxWidth: 460,
    maxHeight: '88%',
    backgroundColor: colors.background,
    borderRadius: radius.xl,
    overflow: 'hidden',
    ...shadow,
  },
  // 고정 큰 크기 — 목록 팝업(지난 대화/이전 진단서)에서 내용이 적어도 크기 유지
  cardLarge: {
    height: '80%',
    maxHeight: '88%',
  },
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: spacing(4),
    paddingTop: spacing(4),
    paddingBottom: spacing(3),
  },
  headerSide: { width: 48, alignItems: 'center' },
  title: {
    flex: 1,
    textAlign: 'center',
    fontSize: 16,
    fontWeight: '800',
    color: colors.text,
  },
  close: { fontSize: 16, color: colors.textSecondary, fontWeight: '700' },
  body: { flexShrink: 1 },
  bodyLarge: { flex: 1 }, // 고정 크기 카드를 채워, 내부 ScrollView 스크롤 + 하단 고정 푸터 가능
});
