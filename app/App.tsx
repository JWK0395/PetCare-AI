/**
 * PetCare AI — 멍냥케어
 * 반려동물 건강관리 AI Agent 서비스 (로컬 MVP)
 */

import React from 'react';
import { StatusBar } from 'react-native';
import { SafeAreaProvider } from 'react-native-safe-area-context';
import { AlertProvider } from './src/components/AlertProvider';
import AppNavigator from './src/navigation';
import { AuthProvider } from './src/state/AuthContext';
import { ChatProvider } from './src/state/ChatContext';
import { PetProvider } from './src/state/PetContext';
import { colors } from './src/theme';

function App() {
  return (
    <SafeAreaProvider>
      <StatusBar barStyle="dark-content" backgroundColor={colors.background} />
      <AlertProvider>
        <AuthProvider>
          <PetProvider>
            <ChatProvider>
              <AppNavigator />
            </ChatProvider>
          </PetProvider>
        </AuthProvider>
      </AlertProvider>
    </SafeAreaProvider>
  );
}

export default App;
