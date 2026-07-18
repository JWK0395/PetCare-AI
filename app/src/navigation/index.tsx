import { createBottomTabNavigator } from '@react-navigation/bottom-tabs';
import { NavigationContainer } from '@react-navigation/native';
import { createNativeStackNavigator } from '@react-navigation/native-stack';
import React from 'react';
import { StyleSheet, Text } from 'react-native';
import { FullLoading } from '../components/ui';
import AICheckScreen from '../screens/AICheckScreen';
import DiagnosisUploadScreen from '../screens/DiagnosisUploadScreen';
import EmergencyEmailScreen from '../screens/EmergencyEmailScreen';
import HomeScreen from '../screens/HomeScreen';
import LoginScreen from '../screens/LoginScreen';
import RecordScreen from '../screens/RecordScreen';
import SummaryScreen from '../screens/SummaryScreen';
import { useAuth } from '../state/AuthContext';
import { colors } from '../theme';
import type { RootStackParamList, TabParamList } from './types';

const Tab = createBottomTabNavigator<TabParamList>();
const Stack = createNativeStackNavigator<RootStackParamList>();

const TAB_ICONS: Record<keyof TabParamList, string> = {
  홈: '🏠',
  기록: '📝',
  'AI 체크': '💬',
  진료: '🏥',
};

function TabIcon({
  name,
  focused,
}: {
  name: keyof TabParamList;
  focused: boolean;
}) {
  return (
    <Text style={[styles.tabIcon, !focused && styles.tabIconInactive]}>
      {TAB_ICONS[name]}
    </Text>
  );
}

function Tabs() {
  return (
    <Tab.Navigator
      screenOptions={({ route }) => ({
        headerShown: false,
        tabBarActiveTintColor: colors.primary,
        tabBarInactiveTintColor: colors.textTertiary,
        tabBarStyle: {
          backgroundColor: colors.card,
          borderTopColor: colors.border,
        },
        tabBarLabelStyle: styles.tabLabel,
        // react-navigation 옵션 콜백 — 컴포넌트 정의가 아니라 렌더 함수라 안전하다
        // eslint-disable-next-line react/no-unstable-nested-components
        tabBarIcon: ({ focused }) => (
          <TabIcon
            name={route.name as keyof TabParamList}
            focused={focused}
          />
        ),
      })}>
      <Tab.Screen name="홈" component={HomeScreen} />
      <Tab.Screen name="AI 체크" component={AICheckScreen} />
      <Tab.Screen name="기록" component={RecordScreen} />
      <Tab.Screen name="진료" component={DiagnosisUploadScreen} />
    </Tab.Navigator>
  );
}

export default function AppNavigator() {
  const { user, initializing } = useAuth();

  // 저장된 토큰으로 세션 복원 중
  if (initializing) {
    return <FullLoading message="로그인 정보를 확인하는 중..." />;
  }

  return (
    <NavigationContainer>
      <Stack.Navigator
        screenOptions={{
          headerTintColor: colors.text,
          headerStyle: { backgroundColor: colors.background },
          headerShadowVisible: false,
          headerTitleStyle: { fontWeight: '700' },
        }}>
        {user ? (
          <>
            <Stack.Screen
              name="Tabs"
              component={Tabs}
              options={{ headerShown: false }}
            />
            <Stack.Screen
              name="Summary"
              component={SummaryScreen}
              options={{ title: '병원 전달용 요약' }}
            />
            <Stack.Screen
              name="EmergencyEmail"
              component={EmergencyEmailScreen}
              options={{ title: '응급 이메일' }}
            />
          </>
        ) : (
          <Stack.Screen
            name="Login"
            component={LoginScreen}
            options={{ headerShown: false }}
          />
        )}
      </Stack.Navigator>
    </NavigationContainer>
  );
}

const styles = StyleSheet.create({
  tabIcon: { fontSize: 18 },
  tabIconInactive: { opacity: 0.45 },
  tabLabel: { fontSize: 11, fontWeight: '600' },
});
