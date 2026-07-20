import { Linking, Platform, Share } from 'react-native';

/**
 * 문서 내보내기 — 이메일 앱 열기 / 요약 공유.
 *
 * **전송·저장은 항상 사용자가 한다.** 앱이 자동으로 보내지 않는다(오발송 방지).
 *
 * PDF 파일을 직접 저장·첨부하지 않는 이유:
 * 서버는 실제 PDF 를 만들지만(`GET /api/summaries/{id}/pdf`, reportlab) 그
 * 엔드포인트는 `Authorization` 헤더를 요구한다. 외부 뷰어·메일 앱은 헤더를 붙일
 * 수 없고, 앱이 직접 받아 저장하려면 파일시스템 라이브러리(네이티브 의존성)가
 * 필요하다. 그래서 지금은 **본문 텍스트**로 내보낸다.
 * 실제 PDF 첨부가 필요해지면 아래 둘 중 하나가 선행되어야 한다:
 *   1) 서버가 단기 서명 URL 을 발급(헤더 없이 열 수 있게)
 *   2) 앱에 파일 저장 라이브러리 추가 + FileProvider 로 ACTION_SEND 첨부
 */

/**
 * 기본 이메일 앱을 연다 (mailto:).
 *
 * 본문에 4섹션 요약 텍스트를 그대로 담는다. `to` 가 비어 있으면 수신자 없이 열어
 * 사용자가 직접 입력하게 한다 — 웹 검색으로는 병원 이메일이 잘 나오지 않으므로
 * 흔한 상황이며 오류가 아니다.
 *
 * @returns 메일 앱이 열렸으면 true
 */
export async function openEmailApp(
  to: string | null,
  subject: string,
  body: string,
): Promise<boolean> {
  const url =
    `mailto:${encodeURIComponent(to || '')}` +
    `?subject=${encodeURIComponent(subject)}` +
    `&body=${encodeURIComponent(body)}`;

  try {
    // Android 는 canOpenURL 이 mailto 에 false 를 돌려주는 경우가 있어 바로 시도한다.
    if (Platform.OS === 'android') {
      await Linking.openURL(url);
      return true;
    }
    if (await Linking.canOpenURL(url)) {
      await Linking.openURL(url);
      return true;
    }
    return false;
  } catch {
    return false;
  }
}

/**
 * 요약 텍스트를 공유 시트로 내보낸다(메모·드라이브·메신저 등에 저장).
 *
 * @returns 공유 시트가 열렸으면 true
 */
export async function shareSummaryText(text: string): Promise<boolean> {
  try {
    await Share.share({ message: text });
    return true;
  } catch {
    return false;
  }
}
