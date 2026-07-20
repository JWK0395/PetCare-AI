"""웹 검색 결과 검증 (명세 15절).

Tavily 가 돌려준 결과는 **절대 그대로 근거로 쓰지 않는다.** 이 모듈이 유일한
관문이며, 통과하지 못한 항목은 `accepted=False` + `reject_reason` 으로 남는다.
버리지 않고 전부 돌려주는 이유는 "왜 웹 fallback 이 실패했는가"를 사용자와
trace 에 설명할 수 있어야 하기 때문이다(추측 답변 금지).

도메인 allowlist 와 거절 신호는 **오직 `config` 에서만** 읽는다. 여기에
도메인을 하드코딩하면 운영 중 목록을 조정할 때 두 곳이 어긋난다.

핵심 보안 포인트 — 도메인 매칭은 문자열 `in` 이 아니라 **호스트 라벨 경계**로
한다. `evil-vet.cornell.edu.attacker.com` 이나
`https://vet.cornell.edu@attacker.com/` 같은 위장 URL 이 통과하면
"Cornell 근거"라는 표시가 붙은 채로 임의의 문서가 답변에 들어간다.

표준 라이브러리 + pydantic + 같은 패키지만 사용한다.
"""

from __future__ import annotations

import re
from urllib.parse import urlsplit

from ..config import Settings, Species, get_settings
from ..schemas import RagQuery, WebEvidence

# ---------------------------------------------------------------------------
# 상수
# ---------------------------------------------------------------------------

#: 이 길이보다 짧은 본문은 근거로 인용할 수 없다(제목만 있는 링크 카드 등).
MIN_CONTENT_CHARS: int = 30

#: 허용 스킴 — data:/javascript: 등은 출처로 제시할 수 없다.
_ALLOWED_SCHEMES: frozenset[str] = frozenset({"http", "https"})

# 광고성 판정용 신호.
# config 의 WEB_REJECT_SIGNALS 는 "커뮤니티/쇼핑 도메인·경로" 판정용이라
# 문구 기반 광고 탐지는 여기서 따로 둔다(도메인 목록은 여전히 config 전용).
_AD_STRONG: tuple[str, ...] = (
    "buy now",
    "add to cart",
    "order now",
    "discount code",
    "free shipping",
    "sponsored content",
    "구매하기",
    "최저가",
    "무료배송",
    "지금 주문",
)
_AD_WEAK: tuple[str, ...] = (
    "advertisement",
    "coupon",
    "% off",
    "sale!",
    "promo",
    "affiliate",
    "광고",
    "쿠폰",
    "할인",
    "특가",
)
#: 약한 신호는 이 개수 이상 동시에 나올 때만 광고로 본다(오탐 방지).
_AD_WEAK_THRESHOLD: int = 2

# species 판정 — 영어는 단어 경계를 강제한다.
# ("cat" 을 substring 으로 찾으면 "catheter", "communicate" 가 전부 걸린다.)
_DOG_PATTERNS: tuple[str, ...] = (
    r"\bdogs?\b",
    r"\bcanines?\b",
    r"\bcanine\b",
    r"\bpupp(?:y|ies)\b",
    r"\bk9\b",
    "강아지",
    "반려견",
    "견종",
)
_CAT_PATTERNS: tuple[str, ...] = (
    r"\bcats?\b",
    r"\bfelines?\b",
    r"\bkittens?\b",
    "고양이",
    "반려묘",
    "묘종",
)
_DOG_RE = re.compile("|".join(_DOG_PATTERNS), re.IGNORECASE)
_CAT_RE = re.compile("|".join(_CAT_PATTERNS), re.IGNORECASE)

_SPECIES_PATTERNS: dict[str, re.Pattern[str]] = {"dog": _DOG_RE, "cat": _CAT_RE}
_OTHER_SPECIES: dict[str, str] = {"dog": "cat", "cat": "dog"}

# 질문 관련성 토큰화용
_EN_TOKEN_RE = re.compile(r"[a-z][a-z0-9\-]{2,}")
_KO_TOKEN_RE = re.compile(r"[가-힣]{2,}")
_STOPWORDS_EN: frozenset[str] = frozenset(
    {
        "the", "and", "for", "with", "that", "this", "from", "have", "has",
        "are", "was", "were", "what", "when", "how", "why", "can", "may",
        "should", "would", "could", "about", "into", "your", "you", "not",
        "but", "any", "all", "pet", "pets", "animal", "animals",
    }
)

#: 거절 사유 코드 — 앞부분이 안정적이라 테스트/trace 에서 문자열로 검사할 수 있다.
REJECT_MISSING_FIELD = "missing_field"
REJECT_INVALID_URL = "invalid_url"
REJECT_EMPTY_CONTENT = "empty_content"
REJECT_NOT_ALLOWLISTED = "not_allowlisted"
REJECT_SIGNAL = "reject_signal"
REJECT_ADVERTISEMENT = "advertisement"
REJECT_SPECIES_MISMATCH = "species_mismatch"
REJECT_IRRELEVANT = "irrelevant"


# ---------------------------------------------------------------------------
# 도메인 유틸 — 다른 모듈(tavily_vet_search)도 재사용한다.
# ---------------------------------------------------------------------------
def extract_domain(url: str) -> str:
    """URL 에서 호스트만 뽑는다. 실패하면 빈 문자열.

    `urlsplit().hostname` 을 쓰는 이유: `user@host` 형태의 userinfo 와 포트를
    표준 규칙대로 제거해 준다. 직접 문자열을 자르면
    `https://vet.cornell.edu@attacker.com/` 을 vet.cornell.edu 로 잘못 읽는다.
    """
    raw = (url or "").strip()
    if not raw:
        return ""
    try:
        parts = urlsplit(raw)
        # 스킴이 없으면 urlsplit 이 전체를 path 로 본다 → https 를 가정해 재파싱.
        if not parts.scheme and not raw.startswith("//"):
            parts = urlsplit("https://" + raw)
        if parts.scheme and parts.scheme.lower() not in _ALLOWED_SCHEMES:
            return ""
        host = parts.hostname or ""
    except ValueError:
        return ""
    return host.lower().strip().rstrip(".")


def match_allowed_domain(
    url_or_host: str, allowed_domains: tuple[str, ...] | list[str]
) -> str | None:
    """allowlist 에 걸리면 매칭된 허용 도메인을, 아니면 None 을 돌려준다.

    라벨 경계 규칙: `host == allowed` 이거나 `host` 가 `"." + allowed` 로 끝날 때만
    허용한다. 그래서
      - vet.cornell.edu            -> 허용
      - www.vet.cornell.edu        -> 허용(정상 서브도메인)
      - evil-vet.cornell.edu.attacker.com -> 거절(접미사가 아님)
      - notvet.cornell.edu         -> 거절(라벨 경계 불일치)
    """
    host = url_or_host if "/" not in url_or_host and ":" not in url_or_host else ""
    host = (host or extract_domain(url_or_host)).lower().rstrip(".")
    if not host:
        return None
    for allowed in allowed_domains:
        candidate = (allowed or "").strip().lower().strip(".")
        if not candidate:
            continue
        if host == candidate or host.endswith("." + candidate):
            return candidate
    return None


def accepted_only(items: list[WebEvidence]) -> list[WebEvidence]:
    """검증을 통과한 항목만 추린다 — 호출자가 매번 필터를 다시 쓰지 않도록."""
    return [item for item in items if item.accepted]


# ---------------------------------------------------------------------------
# 내부 헬퍼
# ---------------------------------------------------------------------------
def _query_tokens(query: RagQuery | None) -> tuple[set[str], set[str]]:
    """질문에서 (영어 토큰, 한국어 토큰)을 뽑는다.

    한국어 질문이어도 `primary_query_en` 이 있으므로 영어 문서와 겹칠 수 있다.
    """
    if query is None:
        return set(), set()
    blob = " ".join(
        [query.primary_query_ko or "", query.primary_query_en or "", *(query.required_topics or [])]
    ).lower()
    en = {tok for tok in _EN_TOKEN_RE.findall(blob) if tok not in _STOPWORDS_EN}
    ko = set(_KO_TOKEN_RE.findall(blob))
    return en, ko


def _relevance_hits(text: str, en_tokens: set[str], ko_tokens: set[str]) -> int:
    """질문 토큰이 문서에서 몇 개나 확인되는지 센다.

    한국어는 조사/어미가 붙어 정확히 일치하지 않으므로(구토하고 vs 구토)
    앞 2글자 substring 으로 느슨하게 본다. 형태소 분석기를 넣지 않는 이유는
    Colab 오프라인 동작을 유지하기 위해서다.
    """
    lowered = text.lower()
    doc_en = set(_EN_TOKEN_RE.findall(lowered))
    hits = len(en_tokens & doc_en)
    for token in ko_tokens:
        if token[:2] in lowered:
            hits += 1
    return hits


def _species_counts(text: str) -> dict[str, int]:
    """문서에 각 종이 몇 번 언급됐는지 센다."""
    return {
        "dog": len(_DOG_RE.findall(text)),
        "cat": len(_CAT_RE.findall(text)),
    }


def _ad_reason(text: str) -> str:
    """광고성으로 볼 근거가 있으면 사유 문자열, 없으면 빈 문자열."""
    lowered = text.lower()
    strong = [sig for sig in _AD_STRONG if sig in lowered]
    if strong:
        return f"판매 유도 문구({strong[0]})"
    weak = [sig for sig in _AD_WEAK if sig in lowered]
    if len(weak) >= _AD_WEAK_THRESHOLD:
        return f"광고성 문구 다수({', '.join(weak[:3])})"
    return ""


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------
class WebSourceValidator:
    """웹 근거 사용 여부를 결정하는 단일 관문(명세 15절).

    settings 를 주입받는 이유: 노트북에서 allowlist 를 바꿔 실험하거나
    테스트가 좁은 allowlist 로 경계 조건을 검증할 수 있어야 한다.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()

    @property
    def settings(self) -> Settings:
        return self._settings

    def validate(
        self,
        items: list[WebEvidence],
        species: Species,
        query: RagQuery | None = None,
    ) -> list[WebEvidence]:
        """전체 항목을 검사해 `accepted` / `reject_reason` 을 채워 **전부** 반환한다.

        원본 객체를 변형하지 않고 복사본을 만든다(호출자가 원본 응답을 그대로
        로그에 남길 수 있어야 한다). accepted 만 쓰는 것은 호출자 책임이며
        `accepted_only()` 헬퍼를 제공한다.
        """
        en_tokens, ko_tokens = _query_tokens(query)
        results: list[WebEvidence] = []
        for item in items or []:
            results.append(self._validate_one(item, species, en_tokens, ko_tokens))
        return results

    # -- 개별 검사 ---------------------------------------------------------
    def _validate_one(
        self,
        item: WebEvidence,
        species: Species,
        en_tokens: set[str],
        ko_tokens: set[str],
    ) -> WebEvidence:
        """검사 순서는 '싸고 확실한 것 먼저' — 위장 도메인은 내용을 보기 전에 끊는다."""
        title = (item.title or "").strip()
        url = (item.url or "").strip()
        content = (item.content or "").strip()
        domain = extract_domain(url)

        def reject(code: str, message: str) -> WebEvidence:
            return item.model_copy(
                update={"domain": domain, "accepted": False, "reject_reason": f"{code}: {message}"}
            )

        # (1) 출처 URL 과 제목이 존재하는가
        if not title or not url:
            return reject(REJECT_MISSING_FIELD, "제목 또는 URL 이 없어 출처로 제시할 수 없습니다.")
        if not domain:
            return reject(REJECT_INVALID_URL, f"해석할 수 없는 URL 입니다({url[:80]}).")

        # (2) 인용할 내용이 남아 있는가
        if len(content) < MIN_CONTENT_CHARS:
            return reject(
                REJECT_EMPTY_CONTENT,
                f"본문이 {len(content)}자로 너무 짧아 근거로 쓸 수 없습니다.",
            )

        # (3) 수의학 기관/대학 allowlist 도메인인가 (config 만 참조)
        matched = match_allowed_domain(domain, self._settings.allowed_web_domains)
        if matched is None:
            return reject(
                REJECT_NOT_ALLOWLISTED,
                f"수의학 기관 allowlist 에 없는 도메인입니다({domain}).",
            )

        # (4) 블로그/카페/SNS/쇼핑 신호 — allowlist 하위 경로에도 적용한다
        #     (예: blog.<허용도메인> 이나 /shop 경로).
        haystack = f"{url}\n{title}".lower()
        for signal in self._settings.web_reject_signals:
            token = (signal or "").strip().lower()
            if token and token in haystack:
                return reject(
                    REJECT_SIGNAL,
                    f"커뮤니티/쇼핑 신호가 포함된 페이지입니다({token}).",
                )

        # (5) 과도한 광고성 페이지가 아닌가
        ad_reason = _ad_reason(f"{title}\n{content}")
        if ad_reason:
            return reject(REJECT_ADVERTISEMENT, f"광고성 페이지로 판단됩니다 — {ad_reason}.")

        # (6) 현재 species 와 일치하는가
        #     반대 종만 언급된 문서는 거절한다. 둘 다 언급이 없으면(예: 일반 독성
        #     정보) 통과시킨다 — 없는 근거를 만들어내는 것보다 낫다.
        counts = _species_counts(f"{title}\n{content}")
        other = _OTHER_SPECIES.get(species, "")
        if counts.get(species, 0) == 0 and other and counts.get(other, 0) > 0:
            return reject(
                REJECT_SPECIES_MISMATCH,
                f"현재 종({species})에 대한 언급 없이 {other} 내용만 다룹니다.",
            )

        # (7) 질문과 직접 관련되는가 (query 가 없으면 검사 생략)
        if en_tokens or ko_tokens:
            if _relevance_hits(f"{title}\n{content}", en_tokens, ko_tokens) == 0:
                return reject(
                    REJECT_IRRELEVANT,
                    "질문 키워드가 문서에서 확인되지 않습니다.",
                )

        return item.model_copy(update={"domain": domain, "accepted": True, "reject_reason": ""})


__all__ = [
    "WebSourceValidator",
    "accepted_only",
    "extract_domain",
    "match_allowed_domain",
    "MIN_CONTENT_CHARS",
    "REJECT_ADVERTISEMENT",
    "REJECT_EMPTY_CONTENT",
    "REJECT_INVALID_URL",
    "REJECT_IRRELEVANT",
    "REJECT_MISSING_FIELD",
    "REJECT_NOT_ALLOWLISTED",
    "REJECT_SIGNAL",
    "REJECT_SPECIES_MISMATCH",
]
