"""Tavily 웹 검색 서비스 (명세 15절 수의학 지식 / 34절 병원 검색).

두 검색은 성격이 완전히 달라서 **별도 class** 로 나눈다.

- `VeterinaryWebSearchService` — 수의학 지식 보완. allowlist 도메인으로만
  검색을 제한하고(1차 방어), 결과는 다시 `WebSourceValidator` 를 통과해야
  한다(2차 방어). RAG 를 대체하지 않고 부족한 부분만 채운다.
- `HospitalSearchService` — 지역 동물병원 페이지가 대상이라 allowlist 를
  적용하지 않는다(적용하면 아무것도 못 찾는다). 대신 **결과에 확정 표현을
  넣지 않는다.** 검색 스니펫만으로 "24시간 진료 중"이나 "응급 접수 가능"을
  단정하면 보호자가 잘못된 판단을 한다. 파싱/판정은 graph 노드가 한다.

공통 설계 원칙:

1. `tavily` 는 **함수 안에서 지연 import** 한다. tavily-python 이 없는 환경에서도
   이 모듈을 import 하는 것만으로 죽으면 안 된다.
2. client 를 생성자로 주입할 수 있다 — Colab unit test 는 실제 API 를 호출하지
   않고 mock client 로 fallback 분기를 검증한다(명세 15절).
3. **API 키 없음 / 패키지 없음 / 호출 실패 / 결과 없음은 전부 "정상적인 fallback
   실패"** 다. 예외를 던지지 않고 빈 리스트를 돌려주며 사유만 로깅한다.
   호출자는 그때 추측하지 않고 제한된 답변을 만든다.
"""

from __future__ import annotations

import logging
from typing import Any, Sequence

from ..config import Settings, Species, get_settings
from ..schemas import WebEvidence
from .source_validator import extract_domain

logger = logging.getLogger(__name__)

#: Tavily 검색 깊이. 수의학 문서는 본문이 길어 basic 스니펫으로는 근거가 얇다.
DEFAULT_SEARCH_DEPTH: str = "advanced"

#: 병원 검색 결과에 항상 붙는 보수적 표시(명세 34절). 확정 표현 금지.
AVAILABILITY_UNCONFIRMED: str = "전화 확인 필요"

#: species 를 영어 검색어로 바꾼다 — Cornell/Merck 문서는 영어다.
_SPECIES_TERMS: dict[str, tuple[str, ...]] = {
    "dog": ("dog", "canine", "puppy", "강아지"),
    "cat": ("cat", "feline", "kitten", "고양이"),
}


# ---------------------------------------------------------------------------
# Tavily client 생성/호출 — 두 서비스가 공유하는 모듈 함수.
# (상속 대신 함수로 둔 이유: 명세가 요구한 대로 두 class 를 서로 독립적으로
#  유지하기 위해서다.)
# ---------------------------------------------------------------------------
def _create_client(settings: Settings, purpose: str) -> Any | None:
    """실 Tavily client 를 만든다. 불가능하면 사유를 로깅하고 None."""
    api_key = settings.tavily_api_key
    if not api_key:
        logger.info("[%s] TAVILY_API_KEY 가 없어 웹 검색을 건너뜁니다(정상 fallback).", purpose)
        return None
    try:
        from tavily import TavilyClient  # type: ignore[import-not-found]
    except ImportError:
        logger.info(
            "[%s] tavily-python 이 설치되어 있지 않아 웹 검색을 건너뜁니다(정상 fallback).",
            purpose,
        )
        return None
    except Exception as exc:  # pragma: no cover - 설치 손상 등 예외적 상황
        logger.warning("[%s] tavily import 실패: %s", purpose, exc)
        return None

    try:
        return TavilyClient(api_key=api_key)
    except Exception as exc:
        logger.warning("[%s] Tavily 클라이언트 생성 실패: %s", purpose, exc)
        return None


def _unwrap_results(raw: Any) -> list[dict[str, Any]]:
    """Tavily 응답에서 결과 리스트만 꺼낸다.

    실 client 는 `{"results": [...]}` 를 주지만 mock 이 리스트를 바로 주는 경우도
    받아준다. 형태가 예상과 다르면 조용히 빈 리스트로 처리한다.
    """
    if raw is None:
        return []
    if isinstance(raw, dict):
        items = raw.get("results") or []
    elif isinstance(raw, list):
        items = raw
    else:
        items = getattr(raw, "results", None) or []
    if not isinstance(items, list):
        return []
    return [item for item in items if isinstance(item, dict)]


def _invoke_search(
    client: Any,
    query: str,
    max_results: int,
    purpose: str,
    include_domains: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    """client.search 를 방어적으로 호출한다. 어떤 실패든 빈 리스트로 흡수한다.

    `include_domains` 는 절대 임의로 떨어뜨리지 않는다. 이 키워드를 못 받는
    client 라면 allowlist 제한 없는 검색이 되어버리므로, 차라리 검색을 포기한다.
    (`search_depth` 는 품질 옵션이라 미지원 시 빼고 재시도한다.)
    """
    base: dict[str, Any] = {"query": query, "max_results": max_results}
    if include_domains:
        base["include_domains"] = list(include_domains)

    attempts: tuple[dict[str, Any], ...] = (
        {**base, "search_depth": DEFAULT_SEARCH_DEPTH},
        base,
    )
    last_type_error: TypeError | None = None
    for kwargs in attempts:
        try:
            return _unwrap_results(client.search(**kwargs))
        except TypeError as exc:
            last_type_error = exc
            continue
        except Exception as exc:
            logger.warning("[%s] Tavily 검색 실패(query=%r): %s", purpose, query, exc)
            return []
    logger.warning(
        "[%s] Tavily client 가 요구 인자를 지원하지 않아 검색을 건너뜁니다: %s",
        purpose,
        last_type_error,
    )
    return []


def _as_float(value: Any) -> float | None:
    """score 는 client 마다 str/None 이 섞여 온다 — 실패하면 None."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _text_of(item: dict[str, Any]) -> str:
    """본문 후보를 고른다. raw_content 가 있으면 근거가 더 두껍다."""
    for key in ("content", "raw_content", "snippet", "description"):
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


# ---------------------------------------------------------------------------
# 1) 수의학 지식 검색
# ---------------------------------------------------------------------------
class VeterinaryWebSearchService:
    """RAG 가 부족할 때만 호출하는 수의학 지식 보완 검색(명세 15절).

    allowlist 를 Tavily `include_domains` 로 넘겨 애초에 커뮤니티/블로그가
    응답에 섞이지 않게 한다. 그래도 반환값은 `accepted=False` 상태이며,
    반드시 `WebSourceValidator` 를 거친 뒤에 근거로 쓴다.
    """

    def __init__(self, settings: Settings | None = None, client: Any | None = None) -> None:
        self._settings = settings or get_settings()
        self._client = client
        self._client_resolved = client is not None

    @property
    def settings(self) -> Settings:
        return self._settings

    def _get_client(self) -> Any | None:
        """주입 client 우선. 없으면 한 번만 생성 시도하고 결과를 캐싱한다."""
        if not self._client_resolved:
            self._client = _create_client(self._settings, "vet-search")
            self._client_resolved = True
        return self._client

    def is_available(self) -> bool:
        """검색이 가능한 상태인지 — 그래프 노드가 미리 분기할 때 쓴다."""
        return self._get_client() is not None

    def build_query(self, query: str, species: Species) -> str:
        """검색어에 종 정보가 없으면 붙인다.

        내부 문서가 영어라 종을 빼면 반대 종 문서가 상위에 올라온다.
        """
        text = (query or "").strip()
        terms = _SPECIES_TERMS.get(species, ())
        lowered = text.lower()
        if terms and not any(term.lower() in lowered for term in terms):
            return f"{text} {terms[0]}".strip()
        return text

    def search(
        self, query: str, species: Species, max_results: int = 5
    ) -> list[WebEvidence]:
        """수의학 allowlist 도메인에서만 검색한다.

        실패·키 없음·결과 없음은 전부 빈 리스트. 예외를 던지지 않는다.
        """
        text = (query or "").strip()
        if not text:
            logger.info("[vet-search] 빈 검색어라 호출하지 않습니다.")
            return []

        client = self._get_client()
        if client is None:
            return []

        allowed = tuple(self._settings.allowed_web_domains or ())
        if not allowed:
            # allowlist 가 비면 무제한 검색이 된다 — 검증되지 않은 근거를 쓰느니
            # 검색하지 않는다(명세 15절: 커뮤니티/블로그 근거 사용 금지).
            logger.warning("[vet-search] allowlist 가 비어 있어 검색을 중단합니다.")
            return []

        results = _invoke_search(
            client,
            self.build_query(text, species),
            max(1, int(max_results)),
            purpose="vet-search",
            include_domains=allowed,
        )
        if not results:
            logger.info("[vet-search] 유효한 검색 결과가 없습니다(정상 fallback).")
            return []

        evidence = [self._to_evidence(item) for item in results]
        logger.info("[vet-search] %d건 수집(검증 전).", len(evidence))
        return evidence

    def _to_evidence(self, item: dict[str, Any]) -> WebEvidence:
        """Tavily 결과 1건 → WebEvidence. 검증 전이므로 accepted 는 False 로 둔다."""
        url = str(item.get("url") or "").strip()
        return WebEvidence(
            title=str(item.get("title") or "").strip(),
            url=url,
            content=_text_of(item),
            score=_as_float(item.get("score")),
            domain=extract_domain(url),
            accepted=False,
            reject_reason="",
        )


# ---------------------------------------------------------------------------
# 2) 병원 검색
# ---------------------------------------------------------------------------
class HospitalSearchService:
    """지역 동물병원 검색(명세 34절) — 원시 결과만 돌려준다.

    allowlist 를 적용하지 않는 이유: 대상이 지역 병원 홈페이지·지도 페이지라
    수의학 기관 allowlist 로 거르면 결과가 0건이 된다. 대신 **검색 결과로
    실시간 영업/응급 접수 가능 여부를 확정하지 않는다.** 각 항목에
    `availability="전화 확인 필요"` 를 붙여 내려보내고, 이름/전화/특징 추출과
    적합도 판정은 graph 노드(Hospital Suitability Agent)가 담당한다.
    """

    def __init__(self, settings: Settings | None = None, client: Any | None = None) -> None:
        self._settings = settings or get_settings()
        self._client = client
        self._client_resolved = client is not None

    @property
    def settings(self) -> Settings:
        return self._settings

    def _get_client(self) -> Any | None:
        if not self._client_resolved:
            self._client = _create_client(self._settings, "hospital-search")
            self._client_resolved = True
        return self._client

    def is_available(self) -> bool:
        return self._get_client() is not None

    def search(self, queries: list[str], max_results: int = 5) -> list[dict]:
        """여러 검색어를 순회하며 원시 결과를 모은다(URL 기준 중복 제거).

        실패한 검색어는 건너뛰고 나머지를 계속 시도한다 — 검색어 하나가
        실패했다고 병원 안내 전체를 포기할 이유는 없다.
        """
        cleaned = [q.strip() for q in (queries or []) if isinstance(q, str) and q.strip()]
        if not cleaned:
            logger.info("[hospital-search] 검색어가 없어 호출하지 않습니다.")
            return []

        client = self._get_client()
        if client is None:
            return []

        limit = max(1, int(max_results))
        collected: list[dict] = []
        seen_urls: set[str] = set()

        for query in cleaned:
            for item in _invoke_search(client, query, limit, purpose="hospital-search"):
                url = str(item.get("url") or "").strip()
                key = url.lower().rstrip("/")
                if not url or key in seen_urls:
                    continue
                seen_urls.add(key)
                collected.append(
                    {
                        "title": str(item.get("title") or "").strip(),
                        "url": url,
                        "content": _text_of(item),
                        "score": _as_float(item.get("score")),
                        "domain": extract_domain(url),
                        "source_query": query,
                        # 확정 표현 금지 — 영업/응급 여부는 여기서 판단하지 않는다.
                        "availability": AVAILABILITY_UNCONFIRMED,
                    }
                )

        if not collected:
            logger.info("[hospital-search] 유효한 검색 결과가 없습니다(정상 fallback).")
        else:
            logger.info("[hospital-search] %d건 수집(파싱 전).", len(collected))
        return collected


__all__ = [
    "AVAILABILITY_UNCONFIRMED",
    "DEFAULT_SEARCH_DEPTH",
    "HospitalSearchService",
    "VeterinaryWebSearchService",
]
