import { useFocusEffect } from '@react-navigation/native';
import React, { useCallback, useState } from 'react';
import {
  RefreshControl,
  ScrollView,
  StyleSheet,
  Text,
  TouchableOpacity,
  View,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { useAlert } from '../components/AlertProvider';
import PetSwitcherModal from '../components/PetSwitcherModal';
import ProfileModal from '../components/ProfileModal';
import { Card, FullLoading, InfoCell } from '../components/ui';
import { useAuth } from '../state/AuthContext';
import { usePet } from '../state/PetContext';
import { colors, radius, spacing } from '../theme';
import { dotDate, koreanDate } from '../utils/date';

export default function HomeScreen() {
  const { pets, pet, loading, error, refresh, selectPet } = usePet();
  const { user, logout } = useAuth();
  const showAlert = useAlert();
  const [refreshing, setRefreshing] = useState(false);
  const [switcherVisible, setSwitcherVisible] = useState(false);
  const [profileMode, setProfileMode] = useState<'create' | 'edit' | null>(
    null,
  );

  const onLogout = () => {
    showAlert('로그아웃', `${user?.email ?? ''}\n로그아웃할까요?`, [
      { text: '취소', style: 'cancel' },
      { text: '로그아웃', style: 'destructive', onPress: () => logout() },
    ]);
  };

  // 다른 화면에 갔다 오면 열려있던 팝업을 닫아 초기 상태로 되돌린다.
  useFocusEffect(
    useCallback(() => {
      return () => {
        setSwitcherVisible(false);
        setProfileMode(null);
      };
    }, []),
  );

  // 데이터가 이미 있으면(당겨서 새로고침 등) 전체 로더로 화면을 갈아치우지 않고
  // RefreshControl 스피너만 보여준다 — 최초 로딩에만 전체 로더.
  if (loading && pets.length === 0) {
    return <FullLoading message="서버에 연결하는 중..." />;
  }

  if (error || !pet) {
    return (
      <SafeAreaView style={styles.safe} edges={['top']}>
        {/* 로그아웃 (반려동물 미등록 상태에서도 접근 가능해야 한다) */}
        <View style={styles.emptyTopBar}>
          <TouchableOpacity onPress={onLogout}>
            <Text style={styles.logoutLink}>로그아웃</Text>
          </TouchableOpacity>
        </View>
        <View style={styles.emptyWrap}>
          <Text style={styles.emptyTitle}>
            {error ? '서버에 연결할 수 없어요' : '반려동물을 등록해 주세요'}
          </Text>
          <Text style={styles.emptyDesc}>
            {error
              ? `${error}\n\nserver 폴더에서 uvicorn 을 실행한 뒤 아래를 눌러 다시 시도해 주세요.`
              : '프로필을 만들면 기록과 AI 상태 체크를 시작할 수 있어요.'}
          </Text>
          <TouchableOpacity
            style={styles.emptyButton}
            onPress={() => (error ? refresh() : setProfileMode('create'))}>
            <Text style={styles.emptyButtonText}>
              {error ? '다시 연결' : '프로필 등록'}
            </Text>
          </TouchableOpacity>
        </View>

        <ProfileModal
          visible={profileMode !== null}
          mode={profileMode ?? 'create'}
          pet={null}
          onClose={() => setProfileMode(null)}
          onSaved={async savedId => {
            setProfileMode(null);
            await refresh(savedId);
          }}
        />
      </SafeAreaView>
    );
  }

  return (
    <SafeAreaView style={styles.safe} edges={['top']}>
      <ScrollView
        contentContainerStyle={styles.container}
        refreshControl={
          <RefreshControl
            refreshing={refreshing}
            onRefresh={async () => {
              setRefreshing(true);
              await refresh();
              setRefreshing(false);
            }}
          />
        }>
        <View style={styles.dateRow}>
          <Text style={styles.date}>{koreanDate()}</Text>
          <TouchableOpacity onPress={onLogout}>
            <Text style={styles.logoutLink}>로그아웃</Text>
          </TouchableOpacity>
        </View>
        <Text style={styles.title}>{pet.name}의 오늘</Text>

        {/* 블록 1 — 프로필 헤더 (아바타 + 이름 + 전환/수정) */}
        <Card style={styles.block}>
          <View style={styles.profileHeader}>
            <View style={styles.avatar}>
              <Text style={styles.avatarText}>{pet.name.slice(0, 1)}</Text>
            </View>
            <View style={styles.identity}>
              <Text style={styles.petName}>{pet.name}</Text>
            </View>
          </View>
          <View style={styles.headerActions}>
            <TouchableOpacity
              style={[styles.headerBtn, styles.headerBtnPrimary]}
              onPress={() => setProfileMode('edit')}
              activeOpacity={0.8}>
              <Text style={[styles.headerBtnText, styles.headerBtnTextPrimary]}>
                프로필 수정
              </Text>
            </TouchableOpacity>
            <TouchableOpacity
              style={styles.headerBtn}
              onPress={() => setSwitcherVisible(true)}
              activeOpacity={0.8}>
              <Text style={styles.headerBtnText}>
                전환{pets.length > 1 ? ` (${pets.length})` : ''}
              </Text>
            </TouchableOpacity>
          </View>
        </Card>

        {/* 블록 2 — 기본 정보 */}
        <Card style={styles.block}>
          <Text style={styles.blockTitle}>기본 정보</Text>
          <View style={styles.infoGrid}>
            <InfoCell label="견종" value={pet.breed} />
            <InfoCell label="생년월일" value={dotDate(pet.birth_date)} />
            <InfoCell label="성별" value={pet.sex} />
            <InfoCell
              label="중성화 여부"
              value={pet.is_neutered ? '완료' : '미완료'}
            />
            <InfoCell
              label="몸무게"
              value={pet.weight_kg ? `${pet.weight_kg}kg` : '-'}
            />
            <InfoCell label="크기" value={pet.size_class} />
          </View>
        </Card>

        {/* 블록 3 — 건강 정보 (기본 정보처럼 라벨-값 행) */}
        <Card style={styles.block}>
          <Text style={styles.blockTitle}>건강 정보</Text>
          <View style={styles.infoGrid}>
            <InfoCell label="기존 질병" value={pet.diseases || '없음'} />
            <InfoCell label="복용약" value={pet.medications || '없음'} />
            <InfoCell label="영양제" value={pet.supplement || '없음'} />
            <InfoCell label="알레르기" value={pet.allergies || '없음'} />
          </View>
        </Card>

        {/* 블록 4 — 프로필 수정일 (독립 블록) */}
        <Card style={styles.block}>
          <View style={styles.updatedRow}>
            <Text style={styles.updatedLabel}>프로필 수정일</Text>
            <Text style={styles.updatedValue}>{dotDate(pet.updated_at)}</Text>
          </View>
        </Card>
      </ScrollView>

      {/* 반려동물 전환 팝업 (프로필 팝업이 이 위에 겹쳐 뜬다) */}
      <PetSwitcherModal
        visible={switcherVisible}
        pets={pets}
        selectedId={pet.id}
        onSelect={id => {
          selectPet(id);
          setSwitcherVisible(false);
        }}
        onAddNew={() => setProfileMode('create')}
        onClose={() => setSwitcherVisible(false)}
        onDeleted={async () => {
          await refresh();
        }}
      />

      {/* 프로필 등록/수정 팝업 (전환 팝업 위에 겹쳐 뜬다) */}
      <ProfileModal
        visible={profileMode !== null}
        mode={profileMode ?? 'edit'}
        pet={profileMode === 'edit' ? pet : null}
        onClose={() => setProfileMode(null)}
        onSaved={async savedId => {
          setProfileMode(null);
          setSwitcherVisible(false);
          await refresh(savedId);
        }}
      />
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  safe: { flex: 1, backgroundColor: colors.background },
  container: { padding: spacing(5), paddingBottom: spacing(8) },
  dateRow: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    marginBottom: spacing(1),
  },
  date: { fontSize: 13, color: colors.textSecondary },
  logoutLink: { fontSize: 12, fontWeight: '600', color: colors.textTertiary },
  emptyTopBar: {
    flexDirection: 'row',
    justifyContent: 'flex-end',
    paddingHorizontal: spacing(5),
    paddingTop: spacing(3),
  },
  title: {
    fontSize: 24,
    fontWeight: '800',
    color: colors.text,
    marginBottom: spacing(4),
  },
  block: { marginBottom: spacing(3) },
  blockTitle: {
    fontSize: 13,
    fontWeight: '700',
    color: colors.textSecondary,
    marginBottom: spacing(3),
  },
  profileHeader: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  avatar: {
    width: 52,
    height: 52,
    borderRadius: 26,
    backgroundColor: colors.primarySoft,
    alignItems: 'center',
    justifyContent: 'center',
    marginRight: spacing(3),
  },
  avatarText: { fontSize: 22, fontWeight: '800', color: colors.primary },
  identity: { flex: 1 },
  petName: { fontSize: 20, fontWeight: '800', color: colors.text },
  headerActions: {
    flexDirection: 'row',
    gap: spacing(2),
    marginTop: spacing(3),
  },
  headerBtn: {
    flex: 1,
    alignItems: 'center',
    backgroundColor: colors.background,
    borderRadius: radius.md,
    paddingVertical: spacing(2.5),
    borderWidth: 1,
    borderColor: colors.border,
  },
  headerBtnPrimary: {
    backgroundColor: colors.primarySoft,
    borderColor: colors.primarySoft,
  },
  headerBtnText: { fontSize: 13, fontWeight: '700', color: colors.textSecondary },
  headerBtnTextPrimary: { color: colors.primary },
  infoGrid: { flexDirection: 'row', flexWrap: 'wrap' },
  updatedRow: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
  },
  updatedLabel: { fontSize: 13, fontWeight: '700', color: colors.textSecondary },
  updatedValue: { fontSize: 14, fontWeight: '600', color: colors.text },
  tag: {
    borderRadius: radius.full,
    paddingHorizontal: spacing(3),
    paddingVertical: spacing(1.5),
  },
  tagText: { fontSize: 12, fontWeight: '600' },
  emptyWrap: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    padding: spacing(8),
  },
  emptyTitle: { fontSize: 18, fontWeight: '700', color: colors.text },
  emptyDesc: {
    fontSize: 13,
    color: colors.textSecondary,
    textAlign: 'center',
    marginTop: spacing(2),
    marginBottom: spacing(5),
    lineHeight: 20,
  },
  emptyButton: {
    backgroundColor: colors.primary,
    borderRadius: radius.md,
    paddingHorizontal: spacing(6),
    paddingVertical: spacing(3),
  },
  emptyButtonText: { color: '#fff', fontWeight: '700' },
});
