import type { NavigatorScreenParams } from '@react-navigation/native';
import type { RiskLevel } from '../api/types';

export type TabParamList = {
  홈: undefined;
  기록: undefined;
  'AI 체크': undefined;
  진료: undefined;
};

export type RootStackParamList = {
  Login: undefined;
  Tabs: NavigatorScreenParams<TabParamList> | undefined;
  Summary: { riskLevel?: RiskLevel; summaryId?: number } | undefined;
  EmergencyEmail: { symptomSummary?: string; hospitalId?: number } | undefined;
};
