"""Tavily 웹 fallback 계층 테스트 (명세 15·17·34·47절).

검증 대상은 세 모듈이 만드는 **하나의 안전 계약**이다.

- `rag/tavily_vet_search.py` — 검색 자체가 실패해도 예외를 던지지 않고,
  allowlist 를 절대 임의로 떨어뜨리지 않는다.
- `rag/source_validator.py` — 웹 결과가 근거가 되는 유일한 관문.
  커뮤니티/블로그/쇼핑/위장 도메인/종 불일치/빈 내용을 전부 막는다.
- `rag/service.py` — RAG 가 충분하면 Tavily 를 **아예 호출하지 않는다**
  (명세 47절 "Tavily 를 모든 질문에 호출하지 말 것").

원칙:
  * 실제 네트워크 호출 없음 — Tavily client 는 전부 mock 주입.
  * 실제 임베딩 모델 로드 없음 — `DeterministicEmbeddings` 만 사용.
  * 파일 산출물 없음 — 실데이터 index 도 메모리에만 만든다.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

from petcare_ai.config import RagSettings, Settings
from petcare_ai.rag.embeddings import DeterministicEmbeddings
from petcare_ai.rag.evidence_merger import merge_evidence
from petcare_ai.rag.service import VeterinaryRagService
from petcare_ai.rag.source_validator import (
    REJECT_ADVERTISEMENT,
    REJECT_EMPTY_CONTENT,
    REJECT_INVALID_URL,
    REJECT_MISSING_FIELD,
    REJECT_NOT_ALLOWLISTED,
    REJECT_SIGNAL,
    REJECT_SPECIES_MISMATCH,
    WebSourceValidator,
    accepted_only,
    extract_domain,
    match_allowed_domain,
)
from petcare_ai.rag.tavily_vet_search import (
    AVAILABILITY_UNCONFIRMED,
    HospitalSearchService,
    VeterinaryWebSearchService,
)
from petcare_ai.schemas import RetrievedEvidence, WebEvidence

# 실데이터(283문서) 압축 해제본 — 전체가 아니라 소량 subset 만 index 로 만든다.
REAL_DATA_PATH = Path(
    "C:/Users/user/AppData/Local/Temp/claude/"
    "E--user-JWK-project-PetCare-AI/e4b20802-3652-4b4b-a9cb-5d7c388b9380/"
    "scratchpad/raw/cornell_pet_health_documents.json"
)


# ---------------------------------------------------------------------------
# 공용 텍스트 상수
# ---------------------------------------------------------------------------
#: 정상 수의학 본문(강아지). 30자 이상 + 광고 문구 없음 + dog 언급 있음.
DOG_CONTENT = (
    "Vomiting in dogs can be caused by dietary indiscretion, infection, or "
    "obstruction. Red flags include repeated vomiting, blood in the vomit, "
    "and lethargy. Owner observations of frequency and appetite help the "
    "veterinarian decide when to seek emergency veterinary care."
)
#: 고양이 본문.
CAT_CONTENT = (
    "Vomiting in cats may indicate hairballs, kidney disease, or "
    "hyperthyroidism. Warning signs for a cat include repeated vomiting, "
    "weight loss, and hiding behavior. Owner observations of litter box use "
    "help the veterinarian decide when to seek emergency veterinary care."
)
#: 종 언급이 전혀 없는 일반 독성 정보 — validator 가 통과시키는 것이 정상이다.
NEUTRAL_CONTENT = (
    "Lilies are highly toxic and even a small amount of pollen can cause "
    "acute kidney injury. Immediate decontamination and veterinary treatment "
    "are required after any exposure."
)


# ---------------------------------------------------------------------------
# 테스트 대역 (mock) — 실제 Tavily / faiss 를 절대 쓰지 않는다.
# ---------------------------------------------------------------------------
class RecordingTavilyClient:
    """Tavily client 대역. 호출 인자를 전부 기록하고 정해진 결과를 돌려준다.

    `calls` 길이가 곧 "Tavily 를 몇 번 불렀는가" 이므로, 명세 47절의
    '조건부 호출' 을 호출 카운터로 직접 검증할 수 있다.
    """

    def __init__(self, results: list[dict] | None = None, error: Exception | None = None) -> None:
        self.results = list(results or [])
        self.error = error
        self.calls: list[dict[str, Any]] = []

    def search(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return {"results": list(self.results)}


class NoIncludeDomainsClient:
    """`include_domains` 를 지원하지 않는 구버전 client 대역.

    이 client 에 검색을 강행하면 allowlist 없는 무제한 검색이 되므로,
    모듈은 검색을 포기해야 한다(명세 15절).
    """

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def search(self, query: str, max_results: int = 5, search_depth: str | None = None) -> dict:
        self.calls.append({"query": query, "max_results": max_results})
        return {"results": [{"title": "무제한 검색 결과", "url": "https://anything.example/x"}]}


class FakeVectorStore:
    """faiss·임베딩 없이 검색 결과를 고정 주입하는 vector store 대역.

    service 의 Tavily 호출 조건만 검증하면 되므로 index 를 만들 이유가 없다.
    `retriever.retrieve()` 가 요구하는 `search()` / `loaded_species` 만 흉내낸다.
    """

    def __init__(self, documents: list[RetrievedEvidence] | None = None) -> None:
        self._documents = list(documents or [])
        self.queries: list[tuple[str, str]] = []

    @property
    def loaded_species(self) -> set[str]:
        return {"dog", "cat"}

    def search(
        self, query: str, species: str, k: int = 6, fetch_k: int = 20
    ) -> list[RetrievedEvidence]:
        self.queries.append((query, species))
        return [doc for doc in self._documents if doc.species == species][:k]


class StubWebSearch:
    """`VeterinaryWebSearchService` 대역 — 검증 단계만 따로 시험할 때 쓴다."""

    def __init__(self, results: list[WebEvidence] | None = None) -> None:
        self.results = list(results or [])
        self.calls: list[tuple[str, str, int]] = []

    def search(self, query: str, species: str, max_results: int = 5) -> list[WebEvidence]:
        self.calls.append((query, species, max_results))
        return list(self.results)


# ---------------------------------------------------------------------------
# 픽스처
# ---------------------------------------------------------------------------
@pytest.fixture()
def settings(tmp_path: Path) -> Settings:
    """테스트 전용 Settings.

    - 임베딩은 `deterministic` — bge-m3 다운로드/로드를 절대 하지 않는다.
    - 모든 경로는 tmp_path 아래로 돌려 저장소에 산출물을 남기지 않는다.
    - allowlist / reject signal 은 운영 기본값을 그대로 써서 실제 정책을 검증한다.
    """
    return Settings(
        data_dir=tmp_path / "raw",
        index_dir=tmp_path / "faiss_index",
        output_dir=tmp_path / "outputs",
        rag=RagSettings(embedding_backend="deterministic"),
    )


@pytest.fixture()
def validator(settings: Settings) -> WebSourceValidator:
    """운영 allowlist 를 그대로 쓰는 검증기."""
    return WebSourceValidator(settings)


def make_item(
    url: str,
    title: str = "Vomiting in Dogs",
    content: str = DOG_CONTENT,
    score: float | None = 0.9,
) -> WebEvidence:
    """검증 전 상태(accepted=False)의 웹 결과 1건을 만든다."""
    return WebEvidence(title=title, url=url, content=content, score=score)


def validate_one(
    validator: WebSourceValidator, item: WebEvidence, species: str = "dog"
) -> WebEvidence:
    """항목 1건만 검증해 결과를 돌려주는 헬퍼."""
    results = validator.validate([item], species)  # type: ignore[arg-type]
    assert len(results) == 1, "validator 는 거절 항목도 버리지 않고 전부 돌려줘야 한다"
    return results[0]


def dog_documents(count: int = 2, text: str = DOG_CONTENT) -> list[RetrievedEvidence]:
    """충분성 판정이 `sufficient` 가 되도록 만든 RAG 근거 목록.

    required_topics(red flags / owner observations / when to seek emergency
    veterinary care)를 본문이 모두 포함하고 score 도 임계값 위라, 충분성 평가기가
    웹 fallback 을 요구하지 않는 상태를 만든다.
    """
    return [
        RetrievedEvidence(
            chunk_id=f"cornell:dog:vomiting#{index}",
            document_id=f"cornell:dog:vomiting-{index}",
            title="Vomiting in Dogs",
            text=text,
            species="dog",
            source="Cornell Riney Canine Health Center",
            source_url=f"https://www.vet.cornell.edu/canine-health/vomiting-{index}",
            categories=["Digestive"],
            score=0.9,
        )
        for index in range(count)
    ]


def conflicting_documents() -> list[RetrievedEvidence]:
    """서로 다른 문서가 '즉시 진료' vs '가정 관찰' 로 엇갈리는 근거 묶음."""
    urgent, watchful = dog_documents(2)
    urgent = urgent.model_copy(
        update={
            "text": "Vomiting in dogs. Red flags and owner observations. "
            "Seek immediate care for any vomiting episode."
        }
    )
    watchful = watchful.model_copy(
        update={
            "text": "Vomiting in dogs. Red flags and owner observations. "
            "Vomiting usually resolves on its own, so you can monitor at home."
        }
    )
    return [urgent, watchful]


DOG_PROFILE = {"species": "강아지", "age_years": 5, "name": "코코"}


# ===========================================================================
# 1) allowlist 출처만 accepted (명세 17절-1)
# ===========================================================================
@pytest.mark.parametrize(
    ("url", "expected_domain"),
    [
        (
            "https://www.vet.cornell.edu/departments-centers-and-institutes/"
            "riney-canine-health-center/canine-health-information/vomiting",
            "www.vet.cornell.edu",
        ),
        ("https://www.merckvetmanual.com/digestive-system/vomiting-in-dogs", "www.merckvetmanual.com"),
        ("https://www.avma.org/resources/pet-owners/vomiting-in-dogs", "www.avma.org"),
        ("https://vetmed.ucdavis.edu/hospital/small-animal/vomiting", "vetmed.ucdavis.edu"),
        ("https://wsava.org/global-guidelines/vomiting", "wsava.org"),
    ],
)
def test_allowlist_domains_are_accepted(
    validator: WebSourceValidator, url: str, expected_domain: str
) -> None:
    """수의학 기관/대학 allowlist 도메인은 근거로 채택돼야 한다.

    이 관문이 과하게 막히면 웹 fallback 자체가 무의미해지므로, 정상 출처가
    확실히 통과하는지(그리고 domain 이 채워지는지)를 먼저 고정한다.
    """
    result = validate_one(validator, make_item(url))
    assert result.accepted is True, f"허용 도메인이 거절됐다: {result.reject_reason}"
    assert result.reject_reason == ""
    assert result.domain == expected_domain


def test_accepted_only_filters_rejected_items(validator: WebSourceValidator) -> None:
    """`accepted_only()` 는 채택분만 추린다 — 거절 항목이 근거로 새지 않는다."""
    items = [
        make_item("https://www.vet.cornell.edu/canine-health/vomiting"),
        make_item("https://www.reddit.com/r/dogs/comments/abc"),
    ]
    validated = validator.validate(items, "dog")
    assert len(validated) == 2, "검증 결과는 사유 설명을 위해 전부 반환돼야 한다"
    accepted = accepted_only(validated)
    assert [item.domain for item in accepted] == ["www.vet.cornell.edu"]


# ===========================================================================
# 2) 커뮤니티 / 블로그 / 쇼핑 결과 rejected (명세 17절-2)
# ===========================================================================
@pytest.mark.parametrize(
    ("url", "expected_code"),
    [
        # allowlist 밖 커뮤니티·쇼핑몰
        ("https://www.reddit.com/r/dogs/comments/abc/my_dog_vomits", REJECT_NOT_ALLOWLISTED),
        ("https://blog.naver.com/petlover/223456789", REJECT_NOT_ALLOWLISTED),
        ("https://cafe.naver.com/dogcafe/1234", REJECT_NOT_ALLOWLISTED),
        ("https://petlover.tistory.com/42", REJECT_NOT_ALLOWLISTED),
        ("https://www.coupang.com/vp/products/1234567", REJECT_NOT_ALLOWLISTED),
        ("https://www.amazon.com/dp/B0123456", REJECT_NOT_ALLOWLISTED),
        # allowlist 도메인이라도 블로그/쇼핑 경로면 거절(2차 방어선)
        ("https://blog.avma.org/2026/01/vomiting-in-dogs", REJECT_SIGNAL),
        ("https://www.aspca.org/shop/dog-supplements", REJECT_SIGNAL),
        ("https://www.merckvetmanual.com/product/vomiting-kit", REJECT_SIGNAL),
    ],
)
def test_community_blog_shopping_sources_are_rejected(
    validator: WebSourceValidator, url: str, expected_code: str
) -> None:
    """커뮤니티/블로그/쇼핑 결과는 거절되고 사유가 반드시 채워져야 한다.

    사유가 비면 "왜 웹 근거를 못 썼는지" 를 사용자·trace 에 설명할 수 없고,
    추측 답변을 막는 근거도 사라진다(명세 15절).
    """
    result = validate_one(validator, make_item(url))
    assert result.accepted is False
    assert result.reject_reason, "거절 항목은 reject_reason 이 비어 있으면 안 된다"
    assert result.reject_reason.startswith(expected_code), result.reject_reason


def test_advertisement_content_is_rejected(validator: WebSourceValidator) -> None:
    """허용 도메인이어도 판매 유도 문구가 있으면 광고성으로 거절한다."""
    result = validate_one(
        validator,
        make_item(
            "https://www.vet.cornell.edu/canine-health/supplement",
            title="구토 보조제 안내",
            content=(
                "Vomiting in dogs supplement. 지금 바로 구매하기 최저가 무료배송 "
                "이벤트를 진행합니다. 강아지 구토에 좋다고 알려진 제품입니다."
            ),
        ),
    )
    assert result.accepted is False
    assert result.reject_reason.startswith(REJECT_ADVERTISEMENT), result.reject_reason


# ===========================================================================
# 3) 유사 도메인 위장 차단 (명세 17절-3)
# ===========================================================================
@pytest.mark.parametrize(
    "url",
    [
        "https://evil.vet.cornell.edu.attacker.com/article",          # 접미사 위장
        "https://vet.cornell.edu.attacker.com/article",               # 접미사 위장
        "https://evil-vet.cornell.edu/article",                       # 라벨 경계 불일치
        "https://notmerckvetmanual.com/article",                      # 접두 위장
        "https://merckvetmanual.com.evil.io/article",                 # 접미사 위장
        "https://vet.cornell.edu@attacker.com/article",               # userinfo 위장
        "https://attacker.com/?ref=https://vet.cornell.edu",          # query 위장
    ],
)
def test_lookalike_domains_are_rejected(validator: WebSourceValidator, url: str) -> None:
    """접미사·접두·userinfo 위장 도메인은 전부 거절돼야 한다.

    이게 뚫리면 임의의 페이지가 'Cornell 근거' 라는 표시를 달고 답변에 들어간다.
    문자열 `in` 매칭이 아니라 **호스트 라벨 경계 매칭**이어야만 막힌다.
    """
    result = validate_one(validator, make_item(url))
    assert result.accepted is False, f"위장 도메인이 통과했다: {url}"
    assert result.reject_reason.startswith(REJECT_NOT_ALLOWLISTED), result.reject_reason


def test_real_subdomain_is_still_allowed(validator: WebSourceValidator) -> None:
    """정상 서브도메인(www.)은 막지 않는다 — 위장 차단이 과잉이면 안 된다."""
    result = validate_one(validator, make_item("https://www.vet.cornell.edu/canine-health/vomiting"))
    assert result.accepted is True


def test_domain_matching_helpers_use_label_boundary() -> None:
    """도메인 헬퍼가 라벨 경계 규칙을 그대로 따르는지 단위로 고정한다."""
    allowed = ("vet.cornell.edu", "merckvetmanual.com")
    assert match_allowed_domain("vet.cornell.edu", allowed) == "vet.cornell.edu"
    assert match_allowed_domain("www.vet.cornell.edu", allowed) == "vet.cornell.edu"
    assert match_allowed_domain("vet.cornell.edu.attacker.com", allowed) is None
    assert match_allowed_domain("evil-vet.cornell.edu", allowed) is None
    assert match_allowed_domain("notmerckvetmanual.com", allowed) is None
    # userinfo/포트가 섞여도 호스트만 정확히 뽑아야 한다.
    assert extract_domain("https://vet.cornell.edu@attacker.com/x") == "attacker.com"
    assert extract_domain("https://www.VET.cornell.edu:443/x") == "www.vet.cornell.edu"
    assert extract_domain("javascript:alert(1)") == ""


# ===========================================================================
# 4) species 불일치 거절 (명세 17절-4)
# ===========================================================================
def test_species_mismatch_is_rejected(validator: WebSourceValidator) -> None:
    """고양이 질문에 강아지 전용 내용이 오면 거절한다.

    종이 어긋난 근거로 답하면 그 자체가 잘못된 의학 정보가 된다(명세 11·14절과
    같은 원칙을 웹 근거에도 적용한다).
    """
    result = validate_one(
        validator,
        make_item("https://www.vet.cornell.edu/canine-health/vomiting", content=DOG_CONTENT),
        species="cat",
    )
    assert result.accepted is False
    assert result.reject_reason.startswith(REJECT_SPECIES_MISMATCH), result.reject_reason


def test_matching_species_content_is_accepted(validator: WebSourceValidator) -> None:
    """같은 고양이 질문이라도 고양이 내용이면 채택된다(위 거절이 도메인 탓이 아님을 증명)."""
    result = validate_one(
        validator,
        make_item(
            "https://www.vet.cornell.edu/feline-health/vomiting",
            title="Vomiting in Cats",
            content=CAT_CONTENT,
        ),
        species="cat",
    )
    assert result.accepted is True, result.reject_reason


def test_species_neutral_content_is_kept(validator: WebSourceValidator) -> None:
    """양쪽 종 언급이 모두 없는 일반 정보(독성 등)는 버리지 않는다.

    근거를 없애는 것보다 종 중립 정보를 남기는 편이 안전하다는 모듈 설계를 고정한다.
    """
    result = validate_one(
        validator,
        make_item(
            "https://www.petpoisonhelpline.com/poison/lilies",
            title="Lily Toxicity",
            content=NEUTRAL_CONTENT,
        ),
        species="cat",
    )
    assert result.accepted is True, result.reject_reason


# ===========================================================================
# 5) 빈 내용 / 제목 없음 / URL 없음 거절 (명세 17절-5)
# ===========================================================================
@pytest.mark.parametrize(
    ("item", "expected_code"),
    [
        (
            WebEvidence(title="Vomiting in Dogs", url="https://www.vet.cornell.edu/x", content=""),
            REJECT_EMPTY_CONTENT,
        ),
        (
            WebEvidence(
                title="Vomiting in Dogs", url="https://www.vet.cornell.edu/x", content="너무 짧다"
            ),
            REJECT_EMPTY_CONTENT,
        ),
        (
            WebEvidence(title="", url="https://www.vet.cornell.edu/x", content=DOG_CONTENT),
            REJECT_MISSING_FIELD,
        ),
        (WebEvidence(title="Vomiting in Dogs", url="", content=DOG_CONTENT), REJECT_MISSING_FIELD),
        (
            WebEvidence(title="Vomiting", url="javascript:alert(1)", content=DOG_CONTENT),
            REJECT_INVALID_URL,
        ),
        (
            WebEvidence(title="Vomiting", url="ftp://vet.cornell.edu/x", content=DOG_CONTENT),
            REJECT_INVALID_URL,
        ),
    ],
)
def test_incomplete_items_are_rejected(
    validator: WebSourceValidator, item: WebEvidence, expected_code: str
) -> None:
    """출처로 제시할 수 없는 결과(제목·URL·본문 결손)는 전부 거절한다.

    인용할 본문이 없는 링크 카드나 스킴이 이상한 URL 을 근거로 붙이면
    "출처 표기" 가 형식만 남고 검증 불가능해진다.
    """
    result = validate_one(validator, item)
    assert result.accepted is False
    assert result.reject_reason.startswith(expected_code), result.reject_reason


# ===========================================================================
# 6) API 오류 시 안전 실패 (명세 17절-6)
# ===========================================================================
@pytest.mark.parametrize(
    "error",
    [
        RuntimeError("Tavily 500 Internal Server Error"),
        TimeoutError("read timeout"),
        ValueError("invalid api key"),
        TypeError("unexpected keyword"),
    ],
)
def test_vet_search_absorbs_client_errors(settings: Settings, error: Exception) -> None:
    """Tavily client 가 어떤 예외를 던져도 밖으로 전파되지 않고 빈 리스트가 나온다.

    웹 fallback 실패는 예외가 아니라 정상 경로다. 여기서 예외가 새면 채팅 전체가
    죽고, 보호자는 아무 답도 받지 못한다(명세 15절).
    """
    client = RecordingTavilyClient(error=error)
    service = VeterinaryWebSearchService(settings, client=client)
    assert service.search("dog vomiting", "dog") == []
    assert client.calls, "예외 케이스라도 호출 자체는 시도돼야 한다"


@pytest.mark.parametrize(
    "raw",
    [None, {"results": None}, {"unexpected": "shape"}, "문자열 응답", 42, {"results": "not-a-list"}],
)
def test_vet_search_absorbs_malformed_responses(settings: Settings, raw: Any) -> None:
    """응답 형태가 예상과 달라도 예외 없이 빈 리스트로 처리한다."""

    class WeirdClient:
        def search(self, **kwargs: Any) -> Any:
            return raw

    service = VeterinaryWebSearchService(settings, client=WeirdClient())
    assert service.search("dog vomiting", "dog") == []


def test_hospital_search_absorbs_client_errors(settings: Settings) -> None:
    """병원 검색도 동일하게 안전 실패한다 — 예외 대신 빈 리스트."""
    client = RecordingTavilyClient(error=RuntimeError("Tavily down"))
    service = HospitalSearchService(settings, client=client)
    assert service.search(["서울 강남 24시 동물병원"]) == []


def test_hospital_search_continues_after_one_failed_query(settings: Settings) -> None:
    """검색어 하나가 실패해도 나머지 검색어로 병원 안내를 계속 시도한다."""

    class PartiallyFailingClient:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def search(self, **kwargs: Any) -> dict:
            query = kwargs["query"]
            self.calls.append(query)
            if "실패" in query:
                raise RuntimeError("이 검색어만 실패")
            return {
                "results": [
                    {"title": "행복동물병원", "url": "https://happy-vet.example/", "content": "24시"}
                ]
            }

    client = PartiallyFailingClient()
    results = HospitalSearchService(settings, client=client).search(["실패 검색어", "강남 동물병원"])
    assert [call for call in client.calls] == ["실패 검색어", "강남 동물병원"]
    assert len(results) == 1
    assert results[0]["title"] == "행복동물병원"


# ===========================================================================
# 7) API 키 부재 / 패키지 미설치 (명세 17절-7)
# ===========================================================================
def test_vet_search_without_api_key_returns_empty(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """TAVILY_API_KEY 가 없으면 예외 없이 빈 리스트 + is_available()=False."""
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    service = VeterinaryWebSearchService(settings)
    assert service.is_available() is False
    assert service.search("dog vomiting", "dog") == []


def test_vet_search_without_tavily_package_returns_empty(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """tavily-python 이 없어도(import 실패) 예외 없이 빈 리스트를 돌려준다.

    `sys.modules['tavily'] = None` 은 `import tavily` 를 ImportError 로 만드는
    표준 기법이라, 패키지 설치 여부와 무관하게 미설치 상황을 재현한다.
    """
    monkeypatch.setenv("TAVILY_API_KEY", "test-key-not-used")
    monkeypatch.setitem(sys.modules, "tavily", None)
    assert VeterinaryWebSearchService(settings).search("dog vomiting", "dog") == []
    assert HospitalSearchService(settings).search(["강남 동물병원"]) == []


def test_module_import_does_not_require_tavily(monkeypatch: pytest.MonkeyPatch) -> None:
    """모듈 최상단에서 tavily 를 import 하지 않는다(지연 import 계약).

    최상단 import 였다면 미설치 환경에서 이 파일 자체가 collect 되지 않는다.
    """
    monkeypatch.setitem(sys.modules, "tavily", None)
    monkeypatch.delitem(sys.modules, "petcare_ai.rag.tavily_vet_search", raising=False)
    import importlib

    module = importlib.import_module("petcare_ai.rag.tavily_vet_search")
    assert hasattr(module, "VeterinaryWebSearchService")


def test_empty_query_and_empty_allowlist_skip_the_call(settings: Settings) -> None:
    """빈 검색어이거나 allowlist 가 비면 client 를 아예 부르지 않는다.

    allowlist 가 빈 상태로 검색하면 무제한 웹 검색이 되어 검증 이전에
    커뮤니티 결과가 섞인다 — 그럴 바에는 검색하지 않는 편이 안전하다.
    """
    client = RecordingTavilyClient(results=[{"title": "x", "url": "https://x.example"}])
    assert VeterinaryWebSearchService(settings, client=client).search("   ", "dog") == []
    assert client.calls == []

    open_settings = Settings(
        data_dir=settings.data_dir,
        index_dir=settings.index_dir,
        output_dir=settings.output_dir,
        rag=settings.rag,
        allowed_web_domains=(),
    )
    assert VeterinaryWebSearchService(open_settings, client=client).search("dog vomiting", "dog") == []
    assert client.calls == [], "allowlist 가 비면 검색 자체를 포기해야 한다"


def test_include_domains_is_never_silently_dropped(settings: Settings) -> None:
    """client 가 include_domains 를 못 받으면 검색을 포기한다.

    이 인자를 떨어뜨리고 재시도하면 allowlist 제한이 사라진 검색이 되므로,
    결과가 0건이 되더라도 제한 없는 검색은 하지 않는 것이 옳다.
    """
    client = NoIncludeDomainsClient()
    assert VeterinaryWebSearchService(settings, client=client).search("dog vomiting", "dog") == []
    assert client.calls == [], "allowlist 없이 재시도해서는 안 된다"


# ===========================================================================
# 8) 유효 결과가 없으면 제한 답변 분기 (명세 17절-8 / 16절)
# ===========================================================================
def test_merge_without_valid_evidence_flags_unreliable() -> None:
    """RAG 도 없고 검증 통과 웹 근거도 없으면 has_reliable_evidence=False.

    이 플래그가 False 여야 호출자가 추측 답변 대신 "확실하지 않음 + 병원 상담
    권고" 분기로 간다(명세 16절).
    """
    rejected = WebEvidence(
        title="내 강아지도 그랬어요",
        url="https://www.reddit.com/r/dogs/comments/abc",
        content=DOG_CONTENT,
        accepted=False,
        reject_reason=f"{REJECT_NOT_ALLOWLISTED}: allowlist 밖",
    )
    merged = merge_evidence([], [rejected])
    assert merged.evidence == []
    assert merged.has_reliable_evidence is False


def test_service_all_web_results_rejected_results_in_limited_answer(settings: Settings) -> None:
    """검색 결과가 전량 거절되면 근거 0건 + has_reliable_evidence=False 로 끝난다.

    RAG 결과도 없고 웹도 전부 거절인 최악의 상황에서 파이프라인이 예외 없이
    '제한 답변' 상태를 만들어 내는지 확인한다.
    """
    client = RecordingTavilyClient(
        results=[
            {
                "title": "내 강아지 구토 후기",
                "url": "https://www.reddit.com/r/dogs/comments/abc",
                "content": DOG_CONTENT,
                "score": 0.7,
            },
            {
                "title": "강아지 영양제 최저가",
                "url": "https://www.coupang.com/vp/products/999",
                "content": DOG_CONTENT,
                "score": 0.6,
            },
        ]
    )
    service = VeterinaryRagService(
        settings,
        store=FakeVectorStore([]),  # RAG 근거 0건 → insufficient → 웹 fallback
        web_search=VeterinaryWebSearchService(settings, client=client),
    )
    result, merged = service.retrieve_with_fallback("강아지가 어제부터 계속 구토를 해요", DOG_PROFILE)

    assert result.sufficiency == "insufficient"
    assert result.web_fallback_required is True
    assert len(client.calls) == 1, "웹 fallback 은 필요할 때 한 번만 호출한다"
    assert merged.evidence == []
    assert merged.has_reliable_evidence is False
    assert service.last_debug["web_accepted_count"] == 0
    assert service.last_debug["web_raw_count"] == 2


# ===========================================================================
# 9) 수의학 검색 / 병원 검색은 별도 class (명세 15·34절)
# ===========================================================================
def test_vet_and_hospital_search_are_independent_classes() -> None:
    """두 서비스는 상속 관계가 아닌 별도 class 여야 한다.

    한쪽을 고치다 다른 쪽 정책(allowlist 적용 여부)이 딸려 바뀌는 사고를 막는다.
    """
    assert VeterinaryWebSearchService is not HospitalSearchService
    assert not issubclass(HospitalSearchService, VeterinaryWebSearchService)
    assert not issubclass(VeterinaryWebSearchService, HospitalSearchService)


def test_vet_search_passes_allowlist_to_client(settings: Settings) -> None:
    """수의학 검색은 allowlist 를 include_domains 로 넘긴다(1차 방어)."""
    client = RecordingTavilyClient(
        results=[
            {
                "title": "Vomiting in Dogs",
                "url": "https://www.vet.cornell.edu/canine-health/vomiting",
                "content": DOG_CONTENT,
                "score": 0.91,
            }
        ]
    )
    items = VeterinaryWebSearchService(settings, client=client).search("dog vomiting", "dog")

    assert len(client.calls) == 1
    assert client.calls[0]["include_domains"] == list(settings.allowed_web_domains)
    assert len(items) == 1
    # 검색 단계는 판정하지 않는다 — validator 를 거치기 전이므로 accepted=False.
    assert items[0].accepted is False
    assert items[0].domain == "www.vet.cornell.edu"


def test_hospital_search_does_not_apply_allowlist(settings: Settings) -> None:
    """병원 검색은 allowlist 를 적용하지 않는다.

    지역 병원 홈페이지·지도 페이지가 대상이라 수의학 기관 allowlist 로 거르면
    결과가 0건이 된다. 대신 확정 표현 없이 `availability` 를 보수적으로 붙인다.
    """
    client = RecordingTavilyClient(
        results=[
            {
                "title": "강남 24시 동물병원",
                "url": "https://blog.naver.com/gangnam-vet/1",
                "content": "야간 진료 안내",
                "score": 0.8,
            },
            {
                "title": "서울동물의료센터",
                "url": "https://seoul-animal.example/",
                "content": "응급 진료 문의",
                "score": 0.7,
            },
        ]
    )
    results = HospitalSearchService(settings, client=client).search(["강남 24시 동물병원"])

    assert len(client.calls) == 1
    assert "include_domains" not in client.calls[0], "병원 검색에 allowlist 를 걸면 결과가 0건이 된다"
    assert len(results) == 2, "allowlist 밖 도메인도 병원 후보로는 남아야 한다"
    assert {item["domain"] for item in results} == {"blog.naver.com", "seoul-animal.example"}
    assert all(item["availability"] == AVAILABILITY_UNCONFIRMED for item in results)


def test_hospital_search_deduplicates_urls_across_queries(settings: Settings) -> None:
    """여러 검색어에서 같은 병원이 나오면 URL 기준으로 한 번만 남긴다."""
    client = RecordingTavilyClient(
        results=[
            {"title": "행복동물병원", "url": "https://happy-vet.example/", "content": "24시"},
            {"title": "행복동물병원", "url": "https://happy-vet.example", "content": "24시"},
        ]
    )
    results = HospitalSearchService(settings, client=client).search(
        ["강남 동물병원", "강남 24시 동물병원"]
    )
    assert len(client.calls) == 2
    assert len(results) == 1


def test_vet_search_appends_species_term_to_query(settings: Settings) -> None:
    """검색어에 종 정보가 없으면 붙여서 반대 종 문서가 상위에 오지 않게 한다."""
    client = RecordingTavilyClient()
    service = VeterinaryWebSearchService(settings, client=client)
    assert service.build_query("vomiting warning signs", "cat") == "vomiting warning signs cat"
    # 이미 종이 들어 있으면 중복해서 붙이지 않는다.
    assert service.build_query("cat vomiting", "cat") == "cat vomiting"
    assert service.build_query("강아지 구토", "dog") == "강아지 구토"


# ===========================================================================
# 10) Tavily 조건부 호출 (명세 17절-10 / 47절 금지사항)
# ===========================================================================
@pytest.mark.parametrize(
    ("label", "message", "documents", "expected_status", "expected_calls"),
    [
        ("sufficient", "강아지가 어제부터 계속 구토를 해요", dog_documents(), "sufficient", 0),
        ("insufficient", "강아지가 어제부터 계속 구토를 해요", [], "insufficient", 1),
        ("conflicting", "강아지가 어제부터 계속 구토를 해요", conflicting_documents(), "conflicting", 1),
        ("requires_recent", "최근 리콜된 사료 공지가 있나요?", dog_documents(), "sufficient", 1),
    ],
)
def test_tavily_is_called_only_when_rag_is_not_enough(
    settings: Settings,
    label: str,
    message: str,
    documents: list[RetrievedEvidence],
    expected_status: str,
    expected_calls: int,
) -> None:
    """명세 47절 — Tavily 를 모든 질문에 호출하지 않는다.

    호출 카운터로 직접 센다.
      - `sufficient` (그리고 최신 정보 불필요) → **호출 0회**
      - `insufficient` / `conflicting` → 호출 1회
      - 충분하더라도 리콜·최신 권고 질문(requires_recent_information) → 호출 1회
    """
    client = RecordingTavilyClient()
    service = VeterinaryRagService(
        settings,
        store=FakeVectorStore(documents),
        web_search=VeterinaryWebSearchService(settings, client=client),
    )
    result, _merged = service.retrieve_with_fallback(message, DOG_PROFILE)

    assert result.sufficiency == expected_status, f"[{label}] 충분성 전제 조건이 깨졌다"
    assert len(client.calls) == expected_calls, f"[{label}] Tavily 호출 횟수가 다르다"
    assert result.web_fallback_required is bool(expected_calls)
    assert service.last_debug["web_search_called"] is bool(expected_calls)


def test_sufficient_case_records_skip_reason(settings: Settings) -> None:
    """호출하지 않은 경우에도 '왜 건너뛰었는지' 가 trace 에 남아야 한다."""
    client = RecordingTavilyClient()
    service = VeterinaryRagService(
        settings,
        store=FakeVectorStore(dog_documents()),
        web_search=VeterinaryWebSearchService(settings, client=client),
    )
    service.retrieve_with_fallback("강아지가 어제부터 계속 구토를 해요", DOG_PROFILE)

    assert client.calls == []
    assert service.last_debug["web_search_called"] is False
    assert "웹 검색을 호출하지 않았습니다" in service.last_debug["web_skip_reason"]
    rows = {row["단계"]: row for row in service.debug_rows()}
    assert rows["4. web fallback"]["결과"] == "호출 안 함"


def test_retrieve_alone_never_calls_tavily(settings: Settings) -> None:
    """`retrieve()` 는 판정만 한다 — 어떤 경우에도 웹을 부르지 않는다."""
    client = RecordingTavilyClient()
    service = VeterinaryRagService(
        settings,
        store=FakeVectorStore([]),  # insufficient 가 되는 상황
        web_search=VeterinaryWebSearchService(settings, client=client),
    )
    result = service.retrieve("강아지가 어제부터 계속 구토를 해요", DOG_PROFILE)
    assert result.web_fallback_required is True
    assert client.calls == [], "retrieve() 단독 호출은 Tavily 를 부르지 않는다"


def test_web_search_exception_does_not_break_pipeline(settings: Settings) -> None:
    """주입된 웹 검색 객체가 예외를 던져도 파이프라인은 RAG 근거로 마무리된다."""

    class ExplodingWebSearch:
        def search(self, query: str, species: str, max_results: int = 5) -> list[WebEvidence]:
            raise RuntimeError("웹 검색 객체 자체가 터진 경우")

    service = VeterinaryRagService(
        settings,
        store=FakeVectorStore(conflicting_documents()),
        web_search=ExplodingWebSearch(),
    )
    result, merged = service.retrieve_with_fallback("강아지가 어제부터 계속 구토를 해요", DOG_PROFILE)

    assert result.web_fallback_required is True
    assert "웹 검색 객체 자체가 터진 경우" in service.last_debug["web_error"]
    assert merged.has_reliable_evidence is True, "RAG 근거는 남아 있어야 한다"
    assert all(item.source_type == "rag" for item in merged.evidence)


# ===========================================================================
# 11) 검증 통과분만 최종 근거에 들어간다 (명세 17절-11 / 15절)
# ===========================================================================
def test_only_validated_web_results_reach_final_evidence(settings: Settings) -> None:
    """Tavily 가 결과를 줘도 validator 를 통과하지 못하면 근거가 되지 않는다.

    허용 출처 1건 + 커뮤니티 1건 + 본문 결손 1건 + 종 불일치 1건을 섞어 넣고,
    최종 근거에 허용 출처만 남는지 확인한다.
    """
    good = WebEvidence(
        title="Vomiting in Dogs",
        url="https://www.vet.cornell.edu/canine-health/vomiting",
        content=DOG_CONTENT,
        score=0.95,
    )
    community = WebEvidence(
        title="우리 강아지도 토했어요",
        url="https://www.reddit.com/r/dogs/comments/abc",
        content=DOG_CONTENT,
        score=0.94,
    )
    thin = WebEvidence(
        title="Vomiting",
        url="https://www.merckvetmanual.com/thin",
        content="짧음",
        score=0.93,
    )
    wrong_species = WebEvidence(
        title="Vomiting in Cats",
        url="https://www.merckvetmanual.com/feline-vomiting",
        content=CAT_CONTENT,
        score=0.92,
    )
    web = StubWebSearch([good, community, thin, wrong_species])

    service = VeterinaryRagService(settings, store=FakeVectorStore([]), web_search=web)
    _result, merged = service.retrieve_with_fallback("강아지가 어제부터 계속 구토를 해요", DOG_PROFILE)

    assert len(web.calls) == 1
    urls = {item.source_url for item in merged.evidence}
    assert urls == {good.url}, f"검증에 실패한 결과가 근거로 들어갔다: {urls}"
    assert merged.has_reliable_evidence is True
    assert service.last_debug["web_raw_count"] == 4
    assert service.last_debug["web_accepted_count"] == 1

    # trace 에는 거절 사유가 전부 남아 있어야 설명 가능한 답변을 만들 수 있다.
    reasons = {row["url"]: row["reject_reason"] for row in service.last_debug["web_results"]}
    assert reasons[good.url] == ""
    assert reasons[community.url].startswith(REJECT_NOT_ALLOWLISTED)
    assert reasons[thin.url].startswith(REJECT_EMPTY_CONTENT)
    assert reasons[wrong_species.url].startswith(REJECT_SPECIES_MISMATCH)


def test_validated_web_evidence_is_kept_behind_rag_evidence(settings: Settings) -> None:
    """검증 통과 웹 근거는 RAG 뒤에 보조 근거로 붙는다(명세 16절 우선순위)."""
    good = WebEvidence(
        title="Vomiting in Dogs",
        url="https://www.merckvetmanual.com/digestive-system/vomiting-in-dogs",
        content=DOG_CONTENT,
        score=0.95,
    )
    service = VeterinaryRagService(
        settings,
        store=FakeVectorStore(conflicting_documents()),  # conflicting → 웹 fallback 실행
        web_search=StubWebSearch([good]),
    )
    _result, merged = service.retrieve_with_fallback("강아지가 어제부터 계속 구토를 해요", DOG_PROFILE)

    types = [item.source_type for item in merged.evidence]
    assert types.count("web") == 1
    assert types.index("web") == len(types) - 1, "웹 근거는 RAG 근거 뒤에 와야 한다"
    assert merged.evidence[-1].source_url == good.url


# ===========================================================================
# 12) 실데이터 소량 subset 통합 확인
# ===========================================================================
@pytest.fixture(scope="module")
def cornell_subset() -> list[dict]:
    """실제 Cornell 문서에서 dog 15 + cat 15 만 뽑는다.

    전체 283문서를 chunk+index 하면 느리므로 subset 만 쓴다. 로드 자체는 빠르다.
    """
    from petcare_ai.rag.loader import load_documents

    if not REAL_DATA_PATH.exists():
        pytest.skip(f"실데이터가 없어 건너뜁니다: {REAL_DATA_PATH}")

    documents, report = load_documents(REAL_DATA_PATH)
    assert report.total_valid > 0
    dogs = [doc for doc in documents if doc.get("species") == "dog"][:15]
    cats = [doc for doc in documents if doc.get("species") == "cat"][:15]
    assert len(dogs) == 15 and len(cats) == 15
    return dogs + cats


def as_single_chunks(documents: list[dict]) -> list[Any]:
    """문서 1건 = chunk 1건으로 색인용 `Chunk` 를 만든다.

    실제 `chunk_documents()` 를 쓰지 않는 이유는 품질 문제가 아니라 속도다.
    chunker 는 `langchain_text_splitters` 를 지연 import 하는데, 그 패키지의
    `__init__` 이 `sentence_transformers`(→ torch)를 함께 끌고 와 import 만으로
    수십 초가 걸린다. 이 파일이 검증하는 것은 chunk 경계가 아니라 웹 fallback
    경로이므로, 본문을 그대로 한 chunk 로 색인해도 검증력이 떨어지지 않는다.
    (chunk 분할 규칙은 chunker 담당 테스트에서 검증한다.)
    """
    from petcare_ai.rag.chunker import Chunk  # 지연 import 없음 — 가벼운 모듈이다.
    from petcare_ai.rag.normalizer import select_body

    chunks: list[Any] = []
    for doc in documents:
        chunk_id = f"{doc['id']}::0001"
        chunks.append(
            Chunk(
                chunk_id=chunk_id,
                document_id=str(doc["id"]),
                text=select_body(doc),
                metadata={
                    "document_id": str(doc["id"]),
                    "chunk_id": chunk_id,
                    "species": doc["species"],
                    "title": doc.get("title") or "",
                    "source": doc.get("source") or "",
                    "source_url": doc.get("source_url") or "",
                    "categories": [str(c) for c in (doc.get("categories") or [])],
                    "heading_path": [],
                },
            )
        )
    return chunks


def test_real_index_falls_back_and_merges_validated_web_evidence(
    settings: Settings, cornell_subset: list[dict]
) -> None:
    """실제 Cornell 문서 index + mock Tavily 로 fallback 경로를 끝까지 확인한다.

    최신 정보를 요구하는 질문(리콜 공지)이라 RAG 가 아무리 잘 나와도 웹 fallback 이
    필요하다. 이때 (1) Tavily 가 정확히 한 번 호출되고, (2) 검증을 통과한 웹 근거가
    실제 RAG 근거 **뒤에** 보조 근거로 합쳐지는지 본다.
    임베딩은 DeterministicEmbeddings 라 모델 다운로드가 일어나지 않는다.
    """
    from petcare_ai.rag.normalizer import normalize_documents
    from petcare_ai.rag.vector_store import VeterinaryVectorStore

    chunks = as_single_chunks(normalize_documents(cornell_subset))
    store = VeterinaryVectorStore(settings, embeddings=DeterministicEmbeddings())
    counts = store.build_all(chunks)
    assert set(counts) == {"dog", "cat"}, "species 별 index 가 모두 만들어져야 한다"

    client = RecordingTavilyClient(
        results=[
            {
                "title": "Recall advisory for dog food",
                "url": "https://www.avma.org/resources/recall-advisory-dog-food",
                "content": (
                    "Warning signs and owner observations after a dog food recall. "
                    "A dog that vomits repeatedly or refuses food should be examined. "
                    "Red flags include lethargy and dehydration."
                ),
                "score": 0.88,
            },
            {
                "title": "사료 리콜 후기 모음",
                "url": "https://cafe.naver.com/dogfood/9999",
                "content": (
                    "우리 강아지도 리콜된 사료를 먹었어요. 커뮤니티 회원들의 후기와 "
                    "경험담을 모았습니다. 병원에 갔더니 괜찮다고 했어요."
                ),
                "score": 0.86,
            },
        ]
    )
    service = VeterinaryRagService(
        settings,
        store=store,
        web_search=VeterinaryWebSearchService(settings, client=client),
    )
    result, merged = service.retrieve_with_fallback("최근 리콜된 사료 공지가 있나요?", DOG_PROFILE)

    assert result.species == "dog"
    assert result.documents, "실데이터 index 에서 RAG 근거가 나와야 한다"
    assert all(doc.species == "dog" for doc in result.documents), "종 교차 오염 금지"
    assert result.web_fallback_required is True
    assert len(client.calls) == 1

    web_evidence = [item for item in merged.evidence if item.source_type == "web"]
    assert [item.source_url for item in web_evidence] == [
        "https://www.avma.org/resources/recall-advisory-dog-food"
    ], "커뮤니티 결과가 근거로 섞였다"
    assert any(item.source_type == "rag" for item in merged.evidence)
    assert merged.has_reliable_evidence is True
