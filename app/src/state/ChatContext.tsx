import React, {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useState,
} from 'react';
import type { AICheckResponse, ChatMessage, Hospital } from '../api/types';
import { useAuth } from './AuthContext';

/** AI 체크 화면의 렌더 아이템 (채팅 버블/카드) */
export type RenderItem =
  | { type: 'user'; text: string; key: string }
  | { type: 'assistant'; text: string; key: string }
  | { type: 'trend'; text: string; key: string }
  | { type: 'result'; response: AICheckResponse; key: string }
  | { type: 'emergency'; response: AICheckResponse; key: string };

/**
 * 진행 중인 AI 체크 대화. 화면(AICheckScreen) 밖에 두어
 * 탭 이동·화면 언마운트에도 대화가 유지되고, "새 체크" 를 누를 때만 초기화된다.
 */
export interface ChatData {
  petId: number | null; // 이 대화가 속한 반려동물 (다른 아이로 바꾸면 새로 시작)
  messages: ChatMessage[];
  items: RenderItem[];
  hospitals: Hospital[];
  sessionId: number | null;
  /**
   * 이 대화에서 쓸 지역명(예: "서울특별시 강남구"). 응급 시 병원 검색어의 유일한 입력이다.
   *
   * **대화당 1회만 조회한다.** 메시지마다 위치를 다시 읽으면 매 전송이 수백 ms 씩
   * 느려지고 권한 팝업이 반복될 수 있는데, 한 대화 도중 구·동이 바뀔 일은 없다.
   * `undefined` = 아직 조회 전, `null` = 조회했지만 알아내지 못함(재시도하지 않는다).
   */
  regionName: string | null | undefined;
  /** 초기화 세대 — 진행 중 요청의 늦은 응답이 리셋된 대화를 되살리지 않게 한다 */
  generation: number;
}

const EMPTY: ChatData = {
  petId: null,
  messages: [],
  items: [],
  hospitals: [],
  sessionId: null,
  regionName: undefined,
  generation: 0,
};

interface ChatContextValue {
  chat: ChatData;
  setChat: React.Dispatch<React.SetStateAction<ChatData>>;
  resetChat: () => void;
}

const ChatContext = createContext<ChatContextValue>({
  chat: EMPTY,
  setChat: () => {},
  resetChat: () => {},
});

export function ChatProvider({ children }: { children: React.ReactNode }) {
  const { user } = useAuth();
  const [chat, setChat] = useState<ChatData>(EMPTY);
  const resetChat = useCallback(
    () => setChat(c => ({ ...EMPTY, generation: c.generation + 1 })),
    [],
  );

  // 로그아웃·계정 전환 시 이전 계정의 대화(건강 상담 내용)가 남지 않도록 초기화한다.
  useEffect(() => {
    resetChat();
  }, [user, resetChat]);

  return (
    <ChatContext.Provider value={{ chat, setChat, resetChat }}>
      {children}
    </ChatContext.Provider>
  );
}

export const useChat = () => useContext(ChatContext);
