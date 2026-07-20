import { NativeModules, PermissionsAndroid, Platform } from 'react-native';

/**
 * 응급 병원 검색용 위치·지역명 조회.
 *
 * 네이티브(LocationModule.kt)가 좌표를 얻고 Android 내장 Geocoder 로
 * "서울특별시 강남구" 같은 지역명까지 만들어 준다. 지역명이 필요한 이유는
 * 병원 검색이 웹 검색(Tavily) 기반이라 좌표를 이해하지 못하기 때문이다.
 *
 * 이 모듈은 **절대 throw 하지 않는다.** 위치를 못 얻는 것은 오류가 아니라
 * "지역을 모르는 상태"이며, 그 경우 앱은 사용자에게 지역을 직접 입력받는다.
 */

export type LocationReason =
  | 'ok'
  | 'permission_denied'
  | 'location_unavailable'
  | 'geocoder_failed'
  | 'unsupported';

export interface RegionResult {
  available: boolean;
  latitude: number | null;
  longitude: number | null;
  regionName: string | null;
  reason: LocationReason;
}

const UNAVAILABLE: RegionResult = {
  available: false,
  latitude: null,
  longitude: null,
  regionName: null,
  reason: 'unsupported',
};

/** 사용자에게 보여줄 안내 문구 — 실패 사유별로 다음 행동이 다르다. */
export const REASON_MESSAGE: Record<LocationReason, string> = {
  ok: '',
  permission_denied:
    '위치 권한이 없어 주변 병원을 찾지 못했어요. 지역을 직접 입력하시면 검색해 드릴게요.',
  location_unavailable:
    '현재 위치를 확인하지 못했어요. 기기의 위치 기능이 켜져 있는지 확인하거나 지역을 직접 입력해 주세요.',
  geocoder_failed:
    '위치는 확인했지만 지역명을 알아내지 못했어요. 지역을 직접 입력해 주세요.',
  unsupported: '이 기기에서는 위치 조회를 사용할 수 없어요. 지역을 직접 입력해 주세요.',
};

/**
 * 위치 권한을 요청한다.
 *
 * COARSE 만 요청하는 이유: 병원 검색은 동/구 단위면 충분하고, 정밀 위치를
 * 요구하면 사용자가 거부할 가능성이 높아진다. Android 12+ 에서는 사용자가
 * 대략적 위치만 허용할 수도 있는데 그 경우에도 정상 동작한다.
 */
export async function requestLocationPermission(): Promise<boolean> {
  if (Platform.OS !== 'android') {
    return false;
  }
  try {
    const granted = await PermissionsAndroid.request(
      PermissionsAndroid.PERMISSIONS.ACCESS_COARSE_LOCATION,
      {
        title: '주변 동물병원 찾기',
        message:
          '응급 상황에서 가까운 동물병원을 찾기 위해 위치를 사용합니다. 위치는 저장하지 않고 검색에만 사용해요.',
        buttonPositive: '허용',
        buttonNegative: '나중에',
      },
    );
    return granted === PermissionsAndroid.RESULTS.GRANTED;
  } catch {
    return false;
  }
}

/**
 * 현재 지역을 조회한다. 권한이 없으면 먼저 요청한다.
 *
 * @param askPermission false 면 권한 요청 팝업을 띄우지 않고 이미 허용된 경우만 조회한다.
 */
export async function getCurrentRegion(
  askPermission = true,
): Promise<RegionResult> {
  if (Platform.OS !== 'android') {
    return UNAVAILABLE;
  }

  const native = NativeModules.PetCareLocation as
    | { getCurrentRegion: () => Promise<RegionResult> }
    | undefined;
  if (!native?.getCurrentRegion) {
    return UNAVAILABLE;
  }

  try {
    let result = await native.getCurrentRegion();

    // 권한이 없어서 실패했으면 한 번 요청해 보고 재시도한다.
    if (result.reason === 'permission_denied' && askPermission) {
      const granted = await requestLocationPermission();
      if (granted) {
        result = await native.getCurrentRegion();
      }
    }
    return result;
  } catch {
    return { ...UNAVAILABLE, reason: 'location_unavailable' };
  }
}
