import React, { createContext, useCallback, useContext, useState } from 'react';
import {
  Modal,
  Pressable,
  StyleSheet,
  Text,
  TouchableOpacity,
  View,
} from 'react-native';
import { colors, radius, shadow, spacing } from '../theme';

export interface AlertButton {
  text: string;
  style?: 'default' | 'cancel' | 'destructive';
  onPress?: () => void;
}

/** Alert.alert 와 같은 시그니처 — 앱 디자인(블루화이트)으로 통일된 커스텀 팝업 */
type ShowAlert = (
  title: string,
  message?: string,
  buttons?: AlertButton[],
) => void;

interface AlertState {
  title: string;
  message?: string;
  buttons?: AlertButton[];
}

const AlertContext = createContext<ShowAlert>(() => {});

export function AlertProvider({ children }: { children: React.ReactNode }) {
  const [state, setState] = useState<AlertState | null>(null);

  const show = useCallback<ShowAlert>((title, message, buttons) => {
    setState({ title, message, buttons });
  }, []);

  const close = () => setState(null);

  const buttons: AlertButton[] =
    state?.buttons && state.buttons.length > 0
      ? state.buttons
      : [{ text: '확인' }];
  const stacked = buttons.length > 2;

  return (
    <AlertContext.Provider value={show}>
      {children}
      <Modal
        visible={!!state}
        transparent
        animationType="fade"
        statusBarTranslucent
        onRequestClose={close}>
        <View style={styles.overlay}>
          <Pressable style={styles.backdrop} onPress={close} />
          <View style={styles.card}>
            {state?.title ? <Text style={styles.title}>{state.title}</Text> : null}
            {state?.message ? (
              <Text style={styles.message}>{state.message}</Text>
            ) : null}
            <View style={[styles.buttonRow, stacked && styles.buttonCol]}>
              {buttons.map((b, i) => {
                const isCancel = b.style === 'cancel';
                const isDestructive = b.style === 'destructive';
                return (
                  <TouchableOpacity
                    key={`${b.text}-${i}`}
                    activeOpacity={0.8}
                    style={[
                      styles.button,
                      isCancel && styles.buttonCancel,
                      isDestructive && styles.buttonDestructive,
                      !isCancel && !isDestructive && styles.buttonDefault,
                      stacked && styles.buttonStacked,
                    ]}
                    onPress={() => {
                      close();
                      b.onPress?.();
                    }}>
                    <Text
                      style={[
                        styles.buttonText,
                        isCancel ? styles.buttonTextCancel : styles.buttonTextFilled,
                      ]}>
                      {b.text}
                    </Text>
                  </TouchableOpacity>
                );
              })}
            </View>
          </View>
        </View>
      </Modal>
    </AlertContext.Provider>
  );
}

export const useAlert = () => useContext(AlertContext);

const styles = StyleSheet.create({
  overlay: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    padding: spacing(6),
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
    maxWidth: 380,
    backgroundColor: colors.card,
    borderRadius: radius.xl,
    padding: spacing(5),
    ...shadow,
  },
  title: {
    fontSize: 17,
    fontWeight: '800',
    color: colors.text,
    marginBottom: spacing(2),
  },
  message: {
    fontSize: 14,
    lineHeight: 21,
    color: colors.textSecondary,
    marginBottom: spacing(4),
  },
  buttonRow: { flexDirection: 'row', gap: spacing(2), marginTop: spacing(1) },
  buttonCol: { flexDirection: 'column' },
  button: {
    flex: 1,
    borderRadius: radius.md,
    paddingVertical: spacing(3),
    alignItems: 'center',
    justifyContent: 'center',
  },
  buttonStacked: { flex: 0, width: '100%' },
  buttonDefault: { backgroundColor: colors.primary },
  buttonDestructive: { backgroundColor: colors.danger },
  buttonCancel: { backgroundColor: colors.background },
  buttonText: { fontSize: 15, fontWeight: '700' },
  buttonTextFilled: { color: '#FFFFFF' },
  buttonTextCancel: { color: colors.textSecondary },
});
