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
  // conversation: 이 요약을 만든 대화 내용. 병원 전달 문서의 '주호소' 는 오늘
  // 무엇 때문에 왔는지이고, 그건 방금 나눈 대화다(옛 진단명이 아니다).
  Summary:
    | { riskLevel?: RiskLevel; summaryId?: number; conversation?: string }
    | undefined;
  // 병원은 DB 등록(hospitalId) 또는 AI 검색 결과(hospitalName/Email/Phone) 로 온다.
  EmergencyEmail:
    | {
        symptomSummary?: string;
        hospitalId?: number;
        hospitalName?: string;
        hospitalEmail?: string | null;
        hospitalPhone?: string | null;
      }
    | undefined;
};
