import AsyncStorage from '@react-native-async-storage/async-storage';
import React, {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useState,
} from 'react';
import {
  getMe,
  login as loginApi,
  logoutApi,
  setAuthToken,
  setOnUnauthorized,
  signup as signupApi,
} from '../api/client';
import type { AuthUser } from '../api/types';

const TOKEN_KEY = 'petcare.auth_token';

interface AuthContextValue {
  user: AuthUser | null;
  initializing: boolean; // 저장된 토큰으로 세션 복원 중
  login: (email: string, password: string) => Promise<void>;
  signup: (email: string, password: string) => Promise<void>;
  logout: () => Promise<void>;
}

const AuthContext = createContext<AuthContextValue>({
  user: null,
  initializing: true,
  login: async () => {},
  signup: async () => {},
  logout: async () => {},
});

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const [user, setUser] = useState<AuthUser | null>(null);
  const [initializing, setInitializing] = useState(true);

  // 앱 시작 시 저장된 토큰으로 세션 복원
  useEffect(() => {
    (async () => {
      try {
        const token = await AsyncStorage.getItem(TOKEN_KEY);
        if (token) {
          setAuthToken(token);
          setUser(await getMe()); // 토큰이 유효하면 사용자 복원
        }
      } catch {
        // 만료/서버 미실행 등 — 로그인 화면에서 다시 시작
        setAuthToken(null);
        await AsyncStorage.removeItem(TOKEN_KEY).catch(() => {});
      } finally {
        setInitializing(false);
      }
    })();
  }, []);

  const applySession = useCallback(
    async (token: string, nextUser: AuthUser) => {
      setAuthToken(token);
      await AsyncStorage.setItem(TOKEN_KEY, token).catch(() => {});
      setUser(nextUser);
    },
    [],
  );

  const clearSession = useCallback(async () => {
    setAuthToken(null);
    await AsyncStorage.removeItem(TOKEN_KEY).catch(() => {});
    setUser(null);
  }, []);

  const login = useCallback(
    async (email: string, password: string) => {
      const res = await loginApi(email.trim(), password);
      await applySession(res.token, res.user);
    },
    [applySession],
  );

  const signup = useCallback(
    async (email: string, password: string) => {
      const res = await signupApi(email.trim(), password);
      await applySession(res.token, res.user); // 가입 즉시 로그인
    },
    [applySession],
  );

  const logout = useCallback(async () => {
    try {
      await logoutApi(); // 서버 토큰 무효화 (실패해도 로컬 세션은 지운다)
    } catch {}
    await clearSession();
  }, [clearSession]);

  // API 가 401 을 받으면 (토큰 만료) 강제 로그아웃 → 로그인 화면으로
  useEffect(() => {
    setOnUnauthorized(() => {
      clearSession();
    });
    return () => setOnUnauthorized(null);
  }, [clearSession]);

  return (
    <AuthContext.Provider value={{ user, initializing, login, signup, logout }}>
      {children}
    </AuthContext.Provider>
  );
}

export const useAuth = () => useContext(AuthContext);
