import React, {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useState,
} from 'react';
import { getPets } from '../api/client';
import type { Pet } from '../api/types';
import { useAuth } from './AuthContext';

interface PetContextValue {
  pets: Pet[];
  pet: Pet | null; // 현재 선택된 반려동물
  loading: boolean;
  error: string | null;
  refresh: (preferId?: number) => Promise<void>;
  selectPet: (id: number) => void;
}

const PetContext = createContext<PetContextValue>({
  pets: [],
  pet: null,
  loading: true,
  error: null,
  refresh: async () => {},
  selectPet: () => {},
});

export function PetProvider({ children }: { children: React.ReactNode }) {
  const { user } = useAuth();
  const [pets, setPets] = useState<Pet[]>([]);
  const [selectedId, setSelectedId] = useState<number | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(
    async (preferId?: number) => {
      setLoading(true);
      setError(null);
      try {
        const list = await getPets();
        setPets(list);
        setSelectedId(prev => {
          const wanted = preferId ?? prev;
          if (wanted != null && list.some(p => p.id === wanted)) {
            return wanted;
          }
          return list.length > 0 ? list[0].id : null;
        });
      } catch (e) {
        setError(
          e instanceof Error
            ? e.message
            : '서버에 연결할 수 없습니다. server 를 실행해 주세요.',
        );
      } finally {
        setLoading(false);
      }
    },
    [],
  );

  // 로그인한 계정의 반려동물만 불러온다. 로그아웃하면 목록을 비운다.
  useEffect(() => {
    if (user) {
      refresh();
    } else {
      setPets([]);
      setSelectedId(null);
      setError(null);
      setLoading(false);
    }
  }, [user, refresh]);

  const selectPet = useCallback((id: number) => setSelectedId(id), []);

  const pet = pets.find(p => p.id === selectedId) ?? null;

  return (
    <PetContext.Provider
      value={{ pets, pet, loading, error, refresh, selectPet }}>
      {children}
    </PetContext.Provider>
  );
}

export const usePet = () => useContext(PetContext);
