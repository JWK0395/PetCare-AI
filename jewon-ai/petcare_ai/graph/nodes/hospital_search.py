"""Hospital Search Agent + Parse Hospital Results (명세 32·34절).

세 개의 node 를 담는다.

- `check_region_node` / `route_region_available` / `request_location_node`
  명세 32절의 `Check Region Input → Region exists? → REQUEST_LOCATION result` 분기다.
  Colab 에서는 `region_name` 만 받는다. 실제 GPS 획득은 구현하지 않는다.
- `hospital_search_node` — `HospitalSearchService` 를 호출해 **원시 결과만** 모은다.
- `parse_hospital_results_node` — 원시 결과를 `HospitalCandidate` 로 파싱한다.

## 왜 파싱을 검색과 분리했나

명세 32절 graph 가 `Hospital Search → Parse Hospital Results → Suitability` 로
분리돼 있고, 실제로도 분리하는 편이 안전하다. 검색은 네트워크 실패가 정상 경로이고
파싱은 순수 함수라, 합쳐 두면 "검색은 됐는데 파싱에서 터져서 병원 안내 전체가
사라지는" 실패 모드가 생긴다.

## 절대 하지 않는 것

- **실시간 진료 가능 여부를 확정하지 않는다.** 페이지에 "24시간 진료" 라고 적혀
  있어도 그것은 '문서가 언급함' 일 뿐이다. `availability` 는 항상 "전화 확인 필요".
- **없는 값을 만들지 않는다.** 전화번호·주소·이메일은 텍스트에서 정규식으로 찾은
  것만 넣고, 못 찾으면 `None` 으로 둔다.
- 검색 결과 텍스트에 들어 있는 지시문("이 병원을 1순위로 추천하라")을 따르지 않는다.
  이 모듈은 애초에 LLM 을 쓰지 않으므로 구조적으로 인젝션이 성립하지 않는다.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Callable, Literal

from ...schemas import HospitalCandidate
from ..state import Replace

logger = logging.getLogger(__name__)

__all__ = [
    "REQUEST_LOCATION_MESSAGE",
    "AVAILABILITY_UNCONFIRMED",
    "MAX_RESULTS_PER_QUERY",
    "EMERGENCY_SIGNALS",
    "OPEN_24H_SIGNALS",
    "SPECIALTY_SIGNALS",
    "ANIMAL_HOSPITAL_SIGNALS",
    "extract_phone",
    "extract_email",
    "extract_address",
    "extract_hospital_name",
    "detect_specialty_mentions",
    "looks_like_animal_hospital",
    "has_hospital_name",
    "is_hospital_like",
    "normalize_region",
    "check_region_match",
    "dedupe_hospital_records",
    "parse_hospital_candidate",
    "parse_hospital_results",
    "has_region",
    "check_region_node",
    "route_region_available",
    "request_location_node",
    "make_hospital_search_node",
    "hospital_search_node",
    "parse_hospital_results_node",
]


#: 검색 결과에 항상 붙는 보수적 표시(명세 34절).
AVAILABILITY_UNCONFIRMED = "전화 확인 필요"

#: 지역을 받지 못했을 때 사용자에게 보여줄 고정 문구. 병원을 추천하는 대신
#: 무엇이 필요한지만 알린다 — 추측으로 지역을 정하지 않는다.
REQUEST_LOCATION_MESSAGE = (
    "가까운 동물병원을 찾아 드리려면 현재 위치(예: '서울 강남구')가 필요합니다.\n"
    "지금 계신 지역을 알려 주시면 그 지역의 동물병원 정보를 정리해 드리겠습니다.\n"
    "급한 상황이라면 기다리지 마시고 평소 다니던 동물병원이나 가까운 24시 동물병원에 "
    "먼저 전화해 지금 진료가 가능한지 확인해 주세요."
)

#: 검색어 1개당 결과 수. 너무 키우면 지역과 무관한 광고 페이지 비중이 커진다.
MAX_RESULTS_PER_QUERY = 5

#: '문서가 응급 진료를 언급했다' 신호. 실시간 가능 여부와는 무관하다.
EMERGENCY_SIGNALS: tuple[str, ...] = (
    "응급", "응급실", "응급진료", "응급의료", "emergency", "urgentcare", "er진료",
)

#: '문서가 24시간·야간 진료를 언급했다' 신호.
OPEN_24H_SIGNALS: tuple[str, ...] = (
    "24시", "24시간", "24hour", "24hr", "24h진료", "야간진료", "심야진료", "밤샘진료",
)

#: 진료 분야 언급 탐지. key 는 결과에 실릴 이름.
SPECIALTY_SIGNALS: dict[str, tuple[str, ...]] = {
    "심장": ("심장", "순환기", "cardio", "cardiac"),
    "호흡기": ("호흡기", "흉부", "respiratory"),
    "신경": ("신경과", "신경외과", "뇌신경", "neuro"),
    "내과": ("내과", "소화기", "internalmedicine"),
    "비뇨기": ("비뇨", "신장", "요로", "urinary"),
    "안과": ("안과", "ophthal"),
    "정형외과": ("정형외과", "정형", "관절", "ortho"),
    "피부": ("피부과", "피부", "dermat"),
    "종양": ("종양", "암센터", "oncol"),
    "치과": ("치과", "구강", "dental"),
    "영상": ("영상의학", "ct", "mri", "초음파"),
    "외과": ("외과", "수술센터", "surgery"),
    "중환자": ("중환자", "icu", "집중치료"),
}

#: 넓은 분야 → 그것을 부분 문자열로 품는 좁은 분야 표현.
#: (예: '정형외과' 안의 '외과' 를 별도 언급으로 세지 않기 위한 마스킹 목록)
_SPECIALTY_SUBSUMPTION: dict[str, tuple[str, ...]] = {
    "외과": ("정형외과", "신경외과", "안과외과"),
    "내과": ("소화기내과", "순환기내과"),
}

#: '동물병원 페이지로 보인다' 신호(필수조건 '동물병원' 판정용).
ANIMAL_HOSPITAL_SIGNALS: tuple[str, ...] = (
    "동물병원", "동물의료", "동물메디컬", "수의과", "동물종합병원", "펫클리닉",
    "animalhospital", "animalmedical", "vetclinic", "veterinary", "vethospital",
)

#: 행정구역 정식 명칭 → 주소·전화 대조에 쓰는 짧은 이름.
#: Android Geocoder 는 "서울특별시", 검색 결과 주소는 "서울"로 적히는 일이 많아
#: 양쪽을 같은 축으로 맞춰야 지역 대조가 성립한다.
_SIDO_ALIASES: dict[str, str] = {
    "서울특별시": "서울", "서울시": "서울",
    "부산광역시": "부산", "대구광역시": "대구", "인천광역시": "인천",
    "광주광역시": "광주", "대전광역시": "대전", "울산광역시": "울산",
    "세종특별자치시": "세종", "세종시": "세종",
    "경기도": "경기",
    "강원특별자치도": "강원", "강원도": "강원",
    "충청북도": "충북", "충청남도": "충남",
    "전북특별자치도": "전북", "전라북도": "전북", "전라남도": "전남",
    "경상북도": "경북", "경상남도": "경남",
    "제주특별자치도": "제주", "제주도": "제주",
}

#: 시/도 → 지역번호. **번호가 곧 위치라는 뜻은 아니다** — 대표번호를 다른 지역에서
#: 쓰는 병원도 있다. 그래서 이 표는 '불일치 의심'을 세우는 데만 쓰고, 주소가 있으면
#: 주소를 우선한다.
_SIDO_AREA_CODES: dict[str, str] = {
    "02": "서울", "051": "부산", "053": "대구", "032": "인천",
    "062": "광주", "042": "대전", "052": "울산", "044": "세종",
    "031": "경기", "033": "강원", "043": "충북", "041": "충남",
    "063": "전북", "061": "전남", "054": "경북", "055": "경남", "064": "제주",
}


# ---------------------------------------------------------------------------
# 정규식 — 없는 값을 만들지 않기 위해 '구분자가 있는 형태' 만 인정한다
# ---------------------------------------------------------------------------
#: 일반 전화(지역번호 + 국번 + 번호). 구분자를 필수로 둬서 사업자번호·우편번호 같은
#: 숫자 덩어리를 전화번호로 오인하지 않게 한다.
_PHONE_RE = re.compile(
    r"(?<![\d-])(0(?:2|3[1-3]|4[1-4]|5[1-5]|6[1-4]|70|10))\s*[-.)]\s*"
    r"(\d{3,4})\s*[-.]?\s*(\d{4})(?![\d-])"
)
#: 대표번호(1588-0000 계열).
_HOTLINE_RE = re.compile(r"(?<![\d-])(1[5-9]\d{2})\s*[-.]\s*(\d{4})(?![\d-])")

_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")

#: 이미지/자동생성 주소는 병원 연락처가 아니다.
_EMAIL_REJECT = ("example.com", "sentry.io", "wixpress.com", "@2x", ".png", ".jpg")

_SIDO = (
    "서울|부산|대구|인천|광주|대전|울산|세종|경기|강원|충북|충남|전북|전남|경북|경남|제주"
)
#: 한국 주소 — 시/도 + (구/군/시) + 도로명 + 번지 형태만 인정한다.
_ADDRESS_RE = re.compile(
    rf"((?:{_SIDO})[가-힣A-Za-z0-9\s]{{0,12}}?"
    r"(?:[가-힣]{1,10}(?:구|군|시))?\s*"
    r"[가-힣A-Za-z0-9]{1,15}(?:대로|로|길)\s*\d{1,4}(?:-\d{1,4})?"
    r"(?:[\s,]*[가-힣A-Za-z0-9]{0,12}(?:층|호|빌딩|타워))?)"
)

#: 제목/본문에서 병원 이름을 뽑는 패턴.
_HOSPITAL_NAME_RE = re.compile(
    r"([0-9A-Za-z가-힣]{1,20}?(?:동물병원|동물의료센터|동물메디컬센터|동물종합병원|펫클리닉))"
)
#: 제목에서 사이트명·설명을 잘라내는 구분자.
_TITLE_SPLIT_RE = re.compile(r"\s*[|·:\-–—]\s*|\s{2,}")


def _compact(text: Any) -> str:
    """공백 제거 + 소문자 — 표기 편차를 흡수하는 매칭용 문자열."""
    return "".join(str(text or "").split()).lower()


def _combined_text(item: dict[str, Any]) -> str:
    """한 검색 결과의 제목·본문·URL 을 한 덩어리로 만든다."""
    return " ".join(
        str(item.get(key) or "") for key in ("title", "content", "snippet", "url")
    )


# ---------------------------------------------------------------------------
# 필드 추출 — 전부 "찾으면 넣고, 못 찾으면 None"
# ---------------------------------------------------------------------------
def extract_phone(text: str) -> str | None:
    """텍스트에서 전화번호를 찾는다(없으면 None).

    형식을 **정규화만** 하고 지어내지 않는다. '02.123.4567' 은 '02-123-4567' 로
    통일하되, 자릿수가 맞지 않으면 아예 반환하지 않는다. 잘못된 번호로 응급 전화를
    걸게 하는 것이 최악의 실패다.
    """
    match = _PHONE_RE.search(str(text or ""))
    if match:
        return f"{match.group(1)}-{match.group(2)}-{match.group(3)}"
    hotline = _HOTLINE_RE.search(str(text or ""))
    if hotline:
        return f"{hotline.group(1)}-{hotline.group(2)}"
    return None


def extract_email(text: str) -> str | None:
    """텍스트에서 이메일 주소를 찾는다(없으면 None)."""
    for match in _EMAIL_RE.finditer(str(text or "")):
        candidate = match.group(0).strip().rstrip(".")
        lowered = candidate.lower()
        if any(signal in lowered for signal in _EMAIL_REJECT):
            continue
        return candidate
    return None


def extract_address(text: str) -> str | None:
    """텍스트에서 한국 주소로 보이는 부분을 찾는다(없으면 None)."""
    match = _ADDRESS_RE.search(str(text or ""))
    if not match:
        return None
    address = re.sub(r"\s+", " ", match.group(1)).strip(" ,")
    return address or None


def extract_hospital_name(title: str, content: str, url: str = "") -> str:
    """병원 이름을 고른다.

    우선순위: 제목 안의 '○○동물병원' → 본문 안의 '○○동물병원' → 제목 앞부분.
    끝까지 못 찾으면 제목(또는 URL)을 그대로 쓴다. 이름이 비면 Pydantic 검증
    이전에 후보 자체가 무의미해지므로, 최소한 사람이 식별할 문자열은 남긴다.
    """
    for source in (title, content):
        match = _HOSPITAL_NAME_RE.search(str(source or ""))
        if match:
            return match.group(1).strip()

    head = _TITLE_SPLIT_RE.split(str(title or "").strip())
    for part in head:
        cleaned = part.strip()
        if cleaned:
            return cleaned[:60]
    return str(url or "이름 미확인").strip()[:60]


def detect_specialty_mentions(text: str) -> list[str]:
    """문서가 언급한 진료 분야를 모은다(언급 사실일 뿐 능력 보증이 아니다)."""
    compact = _compact(text)
    found = [
        name
        for name, signals in SPECIALTY_SIGNALS.items()
        if any(signal in compact for signal in signals)
    ]
    # 넓은 분야가 좁은 분야의 부분 문자열이라 같이 켜지는 경우를 걷어낸다.
    # '정형외과' 한 단어가 '외과' 까지 켜면 언급 목록이 실제보다 부풀어 보인다.
    # 좁은 분야 표현을 지운 뒤에도 신호가 남아 있으면 진짜 언급이므로 유지한다.
    for broad, masks in _SPECIALTY_SUBSUMPTION.items():
        if broad not in found:
            continue
        masked = compact
        for mask in masks:
            masked = masked.replace(mask, "")
        if not any(signal in masked for signal in SPECIALTY_SIGNALS[broad]):
            found.remove(broad)
    return found


def looks_like_animal_hospital(*texts: Any) -> bool:
    """'동물병원 페이지로 보이는가' 필수조건 판정(명세 33절).

    이름·본문·URL 어디에도 동물병원 신호가 없으면 블로그 글이나 쇼핑몰일 가능성이
    높다. 적합도 점수의 `is_animal_hospital` 가산점은 이 판정을 따른다.
    """
    compact = _compact(" ".join(str(text or "") for text in texts))
    return any(signal in compact for signal in ANIMAL_HOSPITAL_SIGNALS)


def has_hospital_name(name: Any) -> bool:
    """이름 자체가 '○○동물병원' 형태인가 — 가장 강한 병원 신호.

    `extract_hospital_name` 은 패턴을 못 찾으면 **제목을 그대로** 이름으로 쓴다.
    그래서 이름이 패턴에 맞는지를 따로 확인해야 "고양이 건강 체크 방법" 같은
    블로그 글과 진짜 병원 페이지를 구분할 수 있다.
    """
    return bool(_HOSPITAL_NAME_RE.fullmatch(str(name or "").strip()))


def is_hospital_like(record: dict[str, Any]) -> bool:
    """검색 결과를 병원 후보로 남길지 판정한다 (명세 33절 필수조건 강화).

    기존에는 `is_animal_hospital` 이 **가산점**일 뿐이라, 본문에 '동물병원' 이
    한 번 스친 정보성 블로그 글이 후보로 남아 점수까지 받았다. 응급 목록에 병원이
    아닌 항목이 섞이면 보호자가 그걸 확인하는 데 시간을 쓴다.

    통과 조건(둘 중 하나):

    1. 이름이 '○○동물병원' 형태다 — 전화번호가 없어도 병원 페이지가 맞다.
    2. 동물병원 신호가 있고 **연락 수단(전화·이메일)** 도 있다.

    둘 다 아니면 버린다. 정보성 글은 대개 고유한 병원 이름도, 연락처도 없다.
    """
    if has_hospital_name(record.get("name")):
        return True
    if not record.get("is_animal_hospital"):
        return False
    return bool(record.get("phone") or record.get("email"))


def normalize_region(region_name: Any) -> tuple[str | None, str | None]:
    """'서울특별시 강남구' → ('서울', '강남구'). 못 알아보면 (None, None).

    시/도를 못 읽어내면 지역 대조 자체를 포기한다 — 틀린 기준으로 거르는 것보다
    안 거르는 편이 안전하다.
    """
    text = re.sub(r"\s+", " ", str(region_name or "")).strip()
    if not text:
        return None, None

    sido: str | None = None
    rest = text
    for full, short in _SIDO_ALIASES.items():
        if text.startswith(full):
            sido, rest = short, text[len(full):].strip()
            break
    if sido is None:
        match = re.match(rf"({_SIDO})", text)
        if match:
            sido, rest = match.group(1), text[match.end():].strip()
    if sido is None:
        return None, None

    district_match = re.search(r"([가-힣]{1,10}(?:구|군|시))", rest)
    return sido, district_match.group(1) if district_match else None


def check_region_match(record: dict[str, Any], sido: str | None) -> str:
    """후보가 요청 지역에 있는지 판정한다 — "match" | "mismatch" | "unknown".

    주소를 먼저 본다. 주소가 있으면 그것이 답이다. 주소가 없을 때만 지역번호를
    본다. 순서를 바꾸면, 서울 병원이 경기 대표번호를 쓰는 흔한 경우에
    '불일치' 판정이 나서 멀쩡한 병원이 밀린다.

    판정할 근거가 없으면 "unknown" 이다. 이 값은 **버리는 근거가 아니라** 등급을
    낮추고 확인 문구를 붙이는 근거로만 쓴다(명세 34절 "확정하지 않는다").
    """
    if not sido:
        return "unknown"

    address = str(record.get("address") or "")
    if address:
        found = re.match(rf"({_SIDO})", address)
        if found:
            return "match" if found.group(1) == sido else "mismatch"

    phone = str(record.get("phone") or "")
    code = re.match(r"(02|0\d{2})", phone)
    if code:
        area = _SIDO_AREA_CODES.get(code.group(1))
        if area:
            return "match" if area == sido else "mismatch"
    return "unknown"


def _dedupe_name_key(name: Any) -> str:
    """중복 판정용 이름 — 공백·괄호·지점 표기를 지운 형태."""
    text = re.sub(r"\(.*?\)", "", str(name or ""))
    return "".join(text.split()).lower()


def _completeness(record: dict[str, Any]) -> int:
    """정보량 — 같은 병원이 여러 페이지로 잡혔을 때 어느 쪽을 남길지 고르는 기준."""
    return sum(
        1
        for key in ("phone", "address", "email", "website")
        if str(record.get(key) or "").strip()
    )


def dedupe_hospital_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """같은 병원이 검색어마다 반복해 잡히는 것을 하나로 합친다.

    URL 만으로 거르면 공식 홈페이지·지도·블로그 소개가 전부 남아 같은 병원이
    3~4번 보인다. 그래서 **이름**으로 한 번 더 묶는다.

    다만 이름이 같아도 전화번호가 서로 다르면 **분점으로 보고 둘 다 남긴다** —
    '24시동물병원' 같은 이름은 지역마다 존재한다. 잘못 합치면 보호자가 엉뚱한
    지점에 전화하게 되므로, 합치는 쪽이 아니라 남기는 쪽으로 기운다.

    남길 대표는 정보가 가장 많은 항목이고, 빠진 필드는 같은 묶음의 다른 항목에서
    채운다(값을 지어내는 것이 아니라 이미 검색된 값을 옮기는 것이다).
    """
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    order: list[tuple[str, str]] = []

    for record in records:
        key = (_dedupe_name_key(record.get("name")), str(record.get("phone") or "").strip())
        if not key[0]:
            key = ("", str(record.get("source_url") or ""))
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(record)

    # 전화번호가 없는 묶음은, 이름이 같고 전화번호가 있는 묶음에 흡수시킨다.
    # (같은 병원의 '연락처 없는 소개 페이지' 를 별도 병원으로 세지 않기 위함)
    # 전화번호 없는 묶음('연락처 없는 소개 페이지')을 전화번호 있는 묶음에 흡수시킨다.
    # 단 **그 이름의 전화번호 묶음이 정확히 하나일 때만** 이다. 분점이 둘 이상이면
    # 그 소개 페이지가 어느 지점 것인지 알 수 없고, 잘못 합치면 A 지점 카드에 B 지점
    # 주소가 붙는다 — 응급에 엉뚱한 곳으로 가게 만드는 실패다. 모르면 합치지 않는다.
    phones_by_name: dict[str, set[str]] = {}
    for name, phone in groups:
        if phone:
            phones_by_name.setdefault(name, set()).add(phone)
    absorbable = {name for name, phones in phones_by_name.items() if len(phones) == 1}

    absorbed: dict[str, list[dict[str, Any]]] = {}
    for name, phone in order:
        if not phone and name in absorbable:
            absorbed.setdefault(name, []).extend(groups[(name, phone)])

    merged: list[dict[str, Any]] = []
    for key in order:
        name, phone = key
        if not phone and name in absorbable:
            continue
        group = groups[key] + (absorbed.get(name, []) if phone else [])
        record = dict(max(groups[key], key=_completeness))
        for other in group:
            for field in ("phone", "address", "email", "website", "source_url"):
                if not str(record.get(field) or "").strip():
                    value = str(other.get(field) or "").strip()
                    if value:
                        record[field] = value
        record["duplicate_count"] = len(group)
        merged.append(record)

    return merged


# ---------------------------------------------------------------------------
# 파싱
# ---------------------------------------------------------------------------
def parse_hospital_candidate(item: dict[str, Any]) -> HospitalCandidate:
    """검색 결과 1건 → `HospitalCandidate` (명세 34절).

    `availability` 는 스키마 기본값("전화 확인 필요")을 그대로 둔다. 문서가
    "24시간 진료" 라고 해도 지금 진료 중이라는 뜻이 아니다.
    """
    title = str(item.get("title") or "")
    content = str(item.get("content") or item.get("snippet") or "")
    url = str(item.get("url") or "").strip()
    text = f"{title}\n{content}"
    compact = _compact(f"{text} {url}")

    return HospitalCandidate(
        name=extract_hospital_name(title, content, url),
        address=extract_address(text),
        phone=extract_phone(text),
        website=url or None,
        email=extract_email(text),
        emergency_mentioned=any(signal in compact for signal in EMERGENCY_SIGNALS),
        open_24h_mentioned=any(signal in compact for signal in OPEN_24H_SIGNALS),
        specialty_mentions=detect_specialty_mentions(text),
        source_url=url,
        availability=AVAILABILITY_UNCONFIRMED,
    )


def parse_hospital_results(
    raw_results: list[dict[str, Any]],
    region_name: Any = None,
) -> list[dict[str, Any]]:
    """원시 검색 결과 목록을 후보 dict 목록으로 바꾼다(순수 함수).

    반환 dict 는 `HospitalCandidate.model_dump()` 에 다음 보조 키를 덧붙인 형태다.

    - `search_title` / `search_snippet` : 적합도 node 의 LLM 특징 추출 입력.
      원문을 버리면 LLM 이 다시 볼 자료가 없어진다.
    - `source_query` : 어떤 검색어에서 나왔는지(디버깅·trace 용).
    - `is_animal_hospital` : 필수조건 판정 결과(점수 계산에서 재사용).
    - `region_match` : 요청 지역과의 대조 결과("match"/"mismatch"/"unknown").
    - `duplicate_count` : 몇 개의 검색 결과가 이 항목으로 합쳐졌는지.

    Pydantic 은 기본적으로 여분 키를 무시하므로 `HospitalCandidate.model_validate()`
    로 그대로 되돌릴 수 있다. 이미 파싱된 항목(name 이 있고 content 가 없는 dict)이
    다시 들어오면 그대로 통과시킨다 — node 가 두 번 실행돼도 망가지지 않게 한다.

    정리 순서는 **URL 중복 제거 → 병원 여부 판정 → 이름 중복 합치기 → 지역 대조** 다.
    병원이 아닌 항목을 먼저 걷어내야 이름 합치기가 엉뚱한 글과 병원을 묶지 않는다.

    `region_name` 이 없으면 지역 대조를 건너뛴다(전부 "unknown"). 위치를 모르는데
    지역으로 거르면 남는 후보가 0이 된다.
    """
    parsed: list[dict[str, Any]] = []
    seen_urls: set[str] = set()
    dropped = 0

    for item in raw_results or []:
        if not isinstance(item, dict):
            logger.warning("병원 검색 결과 형식이 올바르지 않아 건너뜁니다: %r", item)
            continue

        if item.get("name") and not item.get("content"):
            record = dict(item)  # 이미 파싱된 항목
        else:
            candidate = parse_hospital_candidate(item)
            record = candidate.model_dump()
            record["search_title"] = str(item.get("title") or "")
            record["search_snippet"] = str(item.get("content") or item.get("snippet") or "")
            record["source_query"] = str(item.get("source_query") or "")
            record["is_animal_hospital"] = looks_like_animal_hospital(
                candidate.name, record["search_title"], record["search_snippet"], candidate.source_url
            )

        key = str(record.get("source_url") or record.get("website") or record.get("name") or "")
        key = key.lower().rstrip("/")
        if key and key in seen_urls:
            continue
        if key:
            seen_urls.add(key)

        if not is_hospital_like(record):
            dropped += 1
            logger.info("병원 페이지로 보이지 않아 제외: %r", record.get("name"))
            continue

        parsed.append(record)

    merged = dedupe_hospital_records(parsed)

    sido, _district = normalize_region(region_name)
    for record in merged:
        record["region_match"] = check_region_match(record, sido)

    if dropped or len(merged) != len(parsed):
        logger.info(
            "병원 후보 정리: 비병원 %d건 제외, 중복 %d건 병합.",
            dropped,
            len(parsed) - len(merged),
        )
    return merged


# ---------------------------------------------------------------------------
# 지역 확인 (명세 32절)
# ---------------------------------------------------------------------------
def has_region(state: dict[str, Any]) -> bool:
    """검색에 쓸 지역 정보가 있는지 확인한다.

    Colab 범위에서는 `region_name` 만 본다(명세 32절: 실제 위치 획득은 구현하지
    않는다). 위경도만 있고 지역명이 없으면 **지역 없음**으로 판단한다 — 좌표를
    지역명으로 바꾸는 지오코딩을 하지 않기 때문이다.
    """
    return bool(str(state.get("region_name") or "").strip())


def check_region_node(state: dict) -> dict:
    """`Check Region Input` node — 판단만 하고 State 는 바꾸지 않는다.

    빈 dict 를 돌려주는 이유: 이 시점에 무언가를 쓰면 병렬로 도는
    `Clinical Context` branch 와 같은 key 를 건드릴 수 있다. 라우팅은
    `route_region_available()` 이 State 를 읽어서 결정한다.
    """
    if not has_region(state):
        logger.info("지역 정보가 없어 REQUEST_LOCATION 경로로 갑니다.")
    return {}


def route_region_available(state: dict) -> Literal["hospital_search", "request_location"]:
    """명세 32절 `Region exists?` 조건 분기."""
    return "hospital_search" if has_region(state) else "request_location"


def request_location_node(state: dict) -> dict:
    """지역 정보를 요청하는 결과를 만든다(명세 32절 REQUEST_LOCATION result).

    `draft_response` 는 **비어 있을 때만** 채운다. 응급 경로에서는 앞선 Immediate
    Emergency Message node 가 이미 답변을 써 두는데, 그것을 지우면 "지금 바로
    병원에 연락하세요" 안내가 사라진다.

    `hospital_results` 는 빈 목록으로 확정한다. 지역을 모르는 상태에서 이전 turn 의
    병원 목록이 남아 있으면 엉뚱한 지역 병원을 계속 안내하게 된다.
    """
    update: dict[str, Any] = {
        "ui_actions": [
            {
                "type": "REQUEST_LOCATION",
                "reason": "hospital_search",
                "message": REQUEST_LOCATION_MESSAGE,
            }
        ],
        "hospital_results": [],
    }
    if not str(state.get("draft_response") or "").strip():
        update["draft_response"] = REQUEST_LOCATION_MESSAGE
    return update


# ---------------------------------------------------------------------------
# 검색 / 파싱 node
# ---------------------------------------------------------------------------
def _resolve_queries(state: dict[str, Any]) -> list[str]:
    """실행할 검색어를 고른다.

    Requirement Builder 가 이미 만들어 둔 `hospital_search_queries` 를 우선 쓰고,
    없으면 `hospital_requirements` 안의 값을 본다. 둘 다 없으면 여기서 새로
    지어내지 않는다 — 조건 없이 검색하면 지역이 빠진 검색어가 나간다.
    """
    queries = [
        str(query).strip()
        for query in (state.get("hospital_search_queries") or [])
        if str(query).strip()
    ]
    if queries:
        return queries
    requirements = state.get("hospital_requirements") or {}
    if isinstance(requirements, dict):
        return [
            str(query).strip()
            for query in (requirements.get("search_queries") or [])
            if str(query).strip()
        ]
    return []


def make_hospital_search_node(service: Any | None = None) -> Callable[[dict], dict]:
    """검색 서비스를 주입할 수 있는 node factory.

    테스트는 mock client 를 넣은 `HospitalSearchService` 를 주입해 Tavily 없이
    파싱·점수 경로를 검증한다(명세 15/43절).
    """

    def _node(state: dict) -> dict:
        return hospital_search_node(state, service=service)

    return _node


def hospital_search_node(state: dict, service: Any | None = None) -> dict:
    """Hospital Search Agent (명세 34절).

    Tavily 키가 없거나 호출이 실패하면 `HospitalSearchService` 가 빈 목록을
    돌려준다 — **예외가 아니라 정상 경로다.** 그때는 후보 없이 진행하고, 답변에서는
    "검색된 병원 정보가 없다" 는 사실만 전달한다(없는 병원을 만들지 않는다).
    """
    queries = _resolve_queries(state)
    if not queries:
        logger.info("병원 검색어가 없어 검색을 건너뜁니다(지역 미확인 또는 조건 미생성).")
        return {}

    if service is None:
        # 지연 import — tavily 미설치 환경에서도 이 모듈 import 는 성공해야 한다.
        from ...rag.tavily_vet_search import HospitalSearchService

        service = HospitalSearchService()

    try:
        results = service.search(queries, max_results=MAX_RESULTS_PER_QUERY)
    except Exception as exc:  # 서비스가 계약을 어겨도 상담 전체를 실패시키지 않는다.
        logger.warning("병원 검색 중 예기치 못한 오류 — 후보 없이 진행합니다: %s", exc)
        results = []

    if not results:
        logger.info("검색된 병원 후보가 없습니다(정상 fallback).")
        return {"raw_hospital_results": []}

    logger.info("병원 후보 %d건 수집(파싱 전).", len(results))
    return {"raw_hospital_results": list(results)}


def parse_hospital_results_node(state: dict) -> dict:
    """Parse Hospital Results node (명세 32·34절).

    파싱 결과로 `raw_hospital_results` 를 **교체**한다(`Replace`). 같은 key 에 원시
    결과와 파싱 결과가 섞이면 적합도 node 가 같은 병원을 두 번 평가하게 된다.
    """
    raw = state.get("raw_hospital_results") or []
    if not raw:
        return {}

    parsed = parse_hospital_results(list(raw), region_name=state.get("region_name"))
    logger.info("병원 후보 %d건 파싱 완료.", len(parsed))
    return {"raw_hospital_results": Replace(parsed)}
