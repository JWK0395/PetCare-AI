"""명세 17절 '데이터 테스트' — loader / normalizer / chunker 검증.

이 테스트가 지키려는 것은 "수의학 근거의 무결성"이다. 문서가 조용히 사라지거나
(중복 제거로 categories 가 날아가거나), chunk metadata 가 비면 답변에서 출처를
밝힐 수 없고 회귀를 추적할 수도 없다. 그래서 통계 숫자뿐 아니라
"왜 그 숫자여야 하는가"까지 함께 검증한다.

속도 원칙:
- 실제 임베딩 모델은 절대 로드하지 않는다(이 3개 모듈은 임베딩과 무관하다).
- 네트워크 호출 없음.
- 로드 통계는 전체 287건(파일 읽기 자체는 빠르다), chunking 은 dog 15 + cat 15
  subset 으로만 수행한다. 283건 전체 chunking 은 단위테스트에 넣기엔 느리다.
"""

from __future__ import annotations

import json
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

import pytest

from petcare_ai.rag.chunker import Chunk, chunk_document, chunk_documents, chunk_stats
from petcare_ai.rag.loader import (
    LoadReport,
    load_documents,
    load_documents_from_zip,
)
from petcare_ai.rag.normalizer import (
    normalize_document,
    normalize_documents,
    select_body,
)

# ---------------------------------------------------------------------------
# 실데이터 경로 (테스트용 압축 해제본 / 원본 zip)
# ---------------------------------------------------------------------------
_SCRATCH = Path(
    "C:/Users/user/AppData/Local/Temp/claude"
    "/E--user-JWK-project-PetCare-AI"
    "/e4b20802-3652-4b4b-a9cb-5d7c388b9380/scratchpad"
)
DOCUMENTS_JSON = _SCRATCH / "raw" / "cornell_pet_health_documents.json"
RAW_ZIP = Path("E:/user/Downloads/raw.zip")

# 명세 2절 실측 사실 — 이 숫자가 바뀌면 데이터 수집 파이프라인이 바뀐 것이다.
EXPECTED_TOTAL_RAW = 287
EXPECTED_TOTAL_VALID = 283
EXPECTED_DOG = 160
EXPECTED_CAT = 123
EXPECTED_DUPLICATES = 4
EXPECTED_NO_HEADING_DOCS = 14

# 명세 10절이 요구하는 chunk metadata 키 전부.
REQUIRED_METADATA_KEYS = {
    "document_id",
    "chunk_id",
    "species",
    "title",
    "source",
    "source_url",
    "categories",
    "last_updated",
    "medical_domain",
    "language",
    "content_hash",
    "heading_path",
}

# 중복 병합 검증용 — 같은 본문이 두 분류로 등재된 실제 문서.
PANCREATITIS_ID = "cornell:cat:feline-pancreatitis"
PANCREATITIS_CATEGORIES = {"Gastrointestinal Issues", "Pancreatitis"}


# ---------------------------------------------------------------------------
# 픽스처
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def raw_payload() -> list[dict[str, Any]]:
    """원본 JSON 을 그대로 읽는다 — 로더 동작과 무관한 '데이터 자체의 사실' 확인용."""
    if not DOCUMENTS_JSON.exists():
        pytest.skip(f"실데이터가 없습니다: {DOCUMENTS_JSON}")
    return json.loads(DOCUMENTS_JSON.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def loaded() -> tuple[list[dict], LoadReport]:
    """전체 287건 로드 결과. 로드는 빠르므로 통계 테스트는 전체를 쓴다."""
    if not DOCUMENTS_JSON.exists():
        pytest.skip(f"실데이터가 없습니다: {DOCUMENTS_JSON}")
    return load_documents(DOCUMENTS_JSON)


@pytest.fixture(scope="module")
def documents(loaded: tuple[list[dict], LoadReport]) -> list[dict]:
    """중복 제거를 마친 283건."""
    return loaded[0]


@pytest.fixture(scope="module")
def report(loaded: tuple[list[dict], LoadReport]) -> LoadReport:
    """로드 리포트."""
    return loaded[1]


@pytest.fixture(scope="module")
def subset_documents(documents: list[dict]) -> list[dict]:
    """chunking 용 소량 subset(dog 15 + cat 15) — 정규화까지 마친 상태.

    283건 전체 chunking 은 단위테스트에 넣기엔 느리다. chunk 규칙은 문서 수와
    무관하게 문서 단위로 결정되므로 30건이면 규칙 검증에 충분하다.
    """
    dogs = [d for d in documents if d.get("species") == "dog"][:15]
    cats = [d for d in documents if d.get("species") == "cat"][:15]
    return normalize_documents(dogs + cats)


@pytest.fixture(scope="module")
def subset_chunks(subset_documents: list[dict]) -> list[Chunk]:
    """subset 30건의 chunk 전체."""
    return chunk_documents(subset_documents)


def _write_json(tmp_path: Path, name: str, payload: Any) -> Path:
    """tmp_path 에 JSON 픽스처를 쓴다 — 저장소에 산출물을 남기지 않는다."""
    path = tmp_path / name
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


def _valid_doc(doc_id: str, **overrides: Any) -> dict[str, Any]:
    """검증을 통과하는 최소 문서 — 손상 입력 테스트의 기준선."""
    doc: dict[str, Any] = {
        "id": doc_id,
        "species": "dog",
        "source": "Cornell Riney Canine Health Center",
        "source_url": f"https://example.org/{doc_id}",
        "title": f"Title {doc_id}",
        "content_hash": f"hash-{doc_id}",
        "content_markdown": "# Title\n\n" + ("본문 내용입니다. " * 40),
        "content_text": "본문 내용입니다. " * 40,
        "categories": ["General"],
        "language": "en",
        "medical_domain": "canine_health",
        "last_updated": "2025-01-01",
        "headings": [{"level": 2, "text": "Overview"}],
    }
    doc.update(overrides)
    return doc


# ---------------------------------------------------------------------------
# 1. 전체 문서 로드
# ---------------------------------------------------------------------------
def test_total_raw_is_287(report: LoadReport, raw_payload: list[dict]) -> None:
    """287개 문서가 하나도 유실되지 않고 로드되는지 확인한다.

    `total_raw` 는 '읽어들인 원본 건수'이므로 중복 제거 전 값이어야 한다.
    이 값이 파일의 실제 항목 수와 다르면 로더가 조용히 문서를 버리고 있다는 뜻이고,
    그러면 어떤 근거가 빠졌는지 추적할 수 없게 된다.
    """
    assert report.total_raw == EXPECTED_TOTAL_RAW
    assert report.total_raw == len(raw_payload)


def test_all_287_documents_are_valid(report: LoadReport) -> None:
    """실데이터에는 검증 실패 문서가 없어야 한다(제외 0건).

    287 → 283 의 감소분은 전부 '중복 제거' 때문이어야지 '검증 실패' 때문이면 안 된다.
    둘을 구분해야 데이터 품질 저하를 조기에 감지할 수 있다.
    """
    assert report.excluded_count == 0
    assert report.total_valid + report.duplicate_count == report.total_raw


# ---------------------------------------------------------------------------
# 2. 중복 제거 후 분포
# ---------------------------------------------------------------------------
def test_dedupe_result_is_283_with_dog_160_cat_123(
    documents: list[dict], report: LoadReport
) -> None:
    """중복 제거 후 283건, dog 160 / cat 123 으로 분리되는지 확인한다.

    중복 4쌍은 모두 cat 문서이므로 dog 수는 그대로 160, cat 은 127 → 123 이 된다.
    species 는 vector store 분리 기준이라 이 분포가 틀어지면 검색 대상 자체가 바뀐다.
    """
    assert report.total_valid == EXPECTED_TOTAL_VALID
    assert len(documents) == EXPECTED_TOTAL_VALID
    assert report.dog_count == EXPECTED_DOG
    assert report.cat_count == EXPECTED_CAT

    counts = Counter(doc["species"] for doc in documents)
    assert counts == {"dog": EXPECTED_DOG, "cat": EXPECTED_CAT}
    assert report.dog_count + report.cat_count == report.total_valid


def test_average_content_length_matches_returned_documents(
    documents: list[dict], report: LoadReport
) -> None:
    """평균 본문 길이가 '반환된 문서'와 일치하는지 확인한다.

    통계를 중복 제거 전 문서로 계산하면 실제 인덱싱 대상과 어긋난 수치를 보고하게 된다.
    본문 선택은 명세 10절과 동일하게 markdown 우선이어야 한다.
    """
    lengths = [
        len(doc.get("content_markdown") or doc.get("content_text") or "")
        for doc in documents
    ]
    expected = round(sum(lengths) / len(lengths), 2)
    assert report.average_content_length == pytest.approx(expected, abs=0.01)
    assert 4000 < report.average_content_length < 8000


# ---------------------------------------------------------------------------
# 3. content_hash 기반 중복 제거
# ---------------------------------------------------------------------------
def test_duplicate_count_is_4_and_hashes_are_unique(
    documents: list[dict], raw_payload: list[dict], report: LoadReport
) -> None:
    """content_hash 중복 4건이 제거되고, 결과에 중복 해시가 남지 않는지 확인한다.

    원본에 content_hash 가 2번 등장하는 쌍이 정확히 4개 존재한다(모두 cat).
    같은 본문을 두 번 임베딩하면 검색 결과 상위가 같은 문서로 채워져
    근거 다양성이 떨어지므로 반드시 접혀야 한다.
    """
    raw_hash_counts = Counter(doc["content_hash"] for doc in raw_payload)
    assert sum(1 for count in raw_hash_counts.values() if count > 1) == EXPECTED_DUPLICATES

    assert report.duplicate_count == EXPECTED_DUPLICATES

    hashes = [doc["content_hash"] for doc in documents]
    assert len(set(hashes)) == len(hashes)
    ids = [doc["id"] for doc in documents]
    assert len(set(ids)) == len(ids)


def test_dedupe_priority_content_hash_before_id(tmp_path: Path) -> None:
    """id 가 달라도 content_hash 가 같으면 중복으로 접히는지 확인한다.

    명세 9절의 우선순위(content_hash → id → source_url)에서 content_hash 가
    가장 강한 동일성 근거인 이유는, 수집 스크립트가 URL 이나 slug 를 바꿔도
    본문이 같으면 같은 문서이기 때문이다.
    """
    payload = [
        _valid_doc("doc-a", content_hash="same-hash", source_url="https://a.example"),
        _valid_doc("doc-b", content_hash="same-hash", source_url="https://b.example"),
    ]
    path = _write_json(tmp_path, "dupe.json", payload)

    docs, report = load_documents(path)

    assert report.total_raw == 2
    assert report.duplicate_count == 1
    assert [d["id"] for d in docs] == ["doc-a"]  # 먼저 등장한 쪽이 살아남는다


# ---------------------------------------------------------------------------
# 4. 중복 병합 시 categories 합집합
# ---------------------------------------------------------------------------
def test_duplicate_merge_unions_categories(
    documents: list[dict], raw_payload: list[dict]
) -> None:
    """중복 병합 시 categories 가 합집합이 되는지 실데이터로 확인한다.

    `cornell:cat:feline-pancreatitis` 는 'Gastrointestinal Issues' 와 'Pancreatitis'
    두 분류로 각각 등재돼 있다. 한쪽을 그냥 버리면 "췌장염" 질의가 이 문서를
    영영 찾지 못하는 검색 품질 손실이 생기므로, 병합이 곧 근거 보존이다.
    """
    raw_categories: set[str] = set()
    for doc in raw_payload:
        if doc.get("id") == PANCREATITIS_ID:
            raw_categories.update(doc.get("categories") or [])
    assert raw_categories == PANCREATITIS_CATEGORIES  # 원본이 실제로 두 분류로 나뉜다

    merged = [doc for doc in documents if doc["id"] == PANCREATITIS_ID]
    assert len(merged) == 1, "중복 제거 후 1건만 남아야 한다"
    assert set(merged[0]["categories"]) == PANCREATITIS_CATEGORIES


def test_duplicate_merge_preserves_order_and_drops_no_category(tmp_path: Path) -> None:
    """categories 합집합이 '순서 유지 + 중복 없음'인지 확인한다.

    집합으로 합치면 순서가 실행마다 달라져 chunk metadata 가 흔들리고
    (=diff 로 회귀를 확인할 수 없다), 그대로 이어붙이면 같은 분류가 두 번 들어간다.
    """
    payload = [
        _valid_doc("doc-a", content_hash="h", categories=["Gastrointestinal Issues"]),
        _valid_doc("doc-b", content_hash="h", categories=["Pancreatitis", "Gastrointestinal Issues"]),
    ]
    path = _write_json(tmp_path, "cats.json", payload)

    docs, _ = load_documents(path)

    assert docs[0]["categories"] == ["Gastrointestinal Issues", "Pancreatitis"]


# ---------------------------------------------------------------------------
# 5. chunk metadata 완전성
# ---------------------------------------------------------------------------
def test_every_chunk_carries_full_source_metadata(
    subset_chunks: list[Chunk], subset_documents: list[dict]
) -> None:
    """모든 chunk 가 출처 metadata 를 빠짐없이 갖는지 확인한다.

    답변에서 "이 근거는 Cornell 의 어느 문서/어느 절에서 왔다"를 밝히려면
    metadata 가 chunk 단위로 완결돼 있어야 한다. 한 키라도 비면 인용 단계에서
    KeyError 가 나거나, 더 나쁘게는 출처 없는 의학 정보가 나간다.
    """
    assert subset_chunks, "subset 에서 chunk 가 하나도 나오지 않았다"
    by_id = {doc["id"]: doc for doc in subset_documents}

    for chunk in subset_chunks:
        meta = chunk.metadata
        assert REQUIRED_METADATA_KEYS <= set(meta), (
            f"{chunk.chunk_id}: metadata 키 누락 {REQUIRED_METADATA_KEYS - set(meta)}"
        )
        source = by_id[chunk.document_id]

        # 비어 있으면 인용이 불가능한 값들
        assert meta["document_id"] == source["id"]
        assert meta["chunk_id"] == chunk.chunk_id
        assert meta["species"] in {"dog", "cat"}
        assert meta["species"] == source["species"]
        assert meta["title"] == source["title"] and meta["title"]
        assert meta["source"] == source["source"] and meta["source"]
        assert meta["source_url"].startswith("http")
        assert meta["content_hash"] == source["content_hash"] and meta["content_hash"]
        assert meta["language"] == "en"
        assert meta["medical_domain"] in {"canine_health", "feline_health"}

        # 형태가 보장돼야 downstream 이 분기 없이 쓸 수 있다
        assert isinstance(meta["categories"], list)
        assert all(isinstance(c, str) for c in meta["categories"])
        assert isinstance(meta["heading_path"], list)
        assert all(isinstance(h, str) for h in meta["heading_path"])
        assert isinstance(meta["last_updated"], str)


def test_chunk_text_is_non_empty_and_within_size_policy(
    subset_chunks: list[Chunk],
) -> None:
    """chunk 본문이 비어 있지 않고 크기 정책 안에 있는지 확인한다.

    빈 chunk 는 임베딩 노이즈이고, 상한을 크게 넘는 chunk 는 임베딩 품질을 떨어뜨린다.
    heading 만 남은 조각을 인접 chunk 에 붙일 때 min_chunk_length 만큼의 초과는
    의도된 허용치이므로(chunker 3단계) 그만큼의 여유를 준다.
    """
    from petcare_ai.config import get_settings

    rag = get_settings().rag
    limit = rag.chunk_size + rag.min_chunk_length

    for chunk in subset_chunks:
        assert chunk.text.strip(), f"{chunk.chunk_id}: 빈 chunk"
        assert len(chunk.text) <= limit, f"{chunk.chunk_id}: {len(chunk.text)}자 초과"


# ---------------------------------------------------------------------------
# 6. 비정상 문서를 삭제하지 않고 errors 에 기록
# ---------------------------------------------------------------------------
def test_broken_documents_are_reported_not_silently_dropped(tmp_path: Path) -> None:
    """손상 문서를 조용히 버리지 않고 errors 에 사유를 남기는지 확인한다.

    근거 문서가 왜 빠졌는지 알 수 없으면 답변 품질 회귀를 진단할 수 없다.
    그래서 '제외'와 '기록'은 반드시 짝이어야 한다.
    """
    payload = [
        _valid_doc("ok-1"),
        _valid_doc("no-title", title=""),          # 필수 필드 빈 값
        {"id": "no-hash", "species": "cat", "source": "S",
         "source_url": "https://x", "title": "T",
         "content_markdown": "본문"},                # content_hash 자체가 없음
        "문자열은 문서가 아니다",                      # dict 가 아님
    ]
    path = _write_json(tmp_path, "broken.json", payload)

    docs, report = load_documents(path)

    assert report.total_raw == 4
    assert report.total_valid == 1
    assert report.excluded_count == 3
    assert [d["id"] for d in docs] == ["ok-1"]

    joined = "\n".join(report.errors)
    assert "no-title" in joined and "title" in joined
    assert "no-hash" in joined and "content_hash" in joined
    assert "dict" in joined  # 타입 오류 사유가 남는다
    # 제외된 문서 3건 모두 최소 1줄씩 사유가 있어야 한다
    assert len(report.errors) >= report.excluded_count


def test_missing_file_and_malformed_json_return_report_instead_of_raising(
    tmp_path: Path,
) -> None:
    """파일 없음 / JSON 깨짐 / 최상위 타입 오류에서 예외 대신 리포트를 주는지 확인한다.

    노트북 셀 하나가 예외로 죽어 전체 실행이 끊기는 것보다, 원인이 적힌 리포트를
    받아 다음 셀에서 진단하는 편이 낫다는 것이 loader 의 설계 전제다.
    """
    docs, report = load_documents(tmp_path / "없는파일.json")
    assert docs == [] and report.total_raw == 0
    assert any("찾을 수 없" in e for e in report.errors)

    broken = tmp_path / "broken.json"
    broken.write_text("{ not json", encoding="utf-8")
    docs, report = load_documents(broken)
    assert docs == [] and any("JSON" in e for e in report.errors)

    not_a_list = _write_json(tmp_path, "dict.json", {"documents": []})
    docs, report = load_documents(not_a_list)
    assert docs == [] and any("list" in e for e in report.errors)


# ---------------------------------------------------------------------------
# 7. species 이상값 / 본문 빈 문서
# ---------------------------------------------------------------------------
def test_invalid_species_document_is_excluded_with_reason(tmp_path: Path) -> None:
    """species 가 dog/cat 이 아닌 문서를 제외하고 사유를 남기는지 확인한다.

    species 는 FAISS index 분리 기준이다. 임의 값이 섞이면 어느 index 에도 들어가지
    못하거나(=조용한 유실) 잘못된 종의 근거가 답변에 인용된다.
    """
    payload = [
        _valid_doc("ok-1"),
        _valid_doc("hamster-1", species="hamster"),
        _valid_doc("empty-species", species=""),
        _valid_doc("upper-dog", species="Dog"),  # 대문자도 허용하지 않는다
    ]
    path = _write_json(tmp_path, "species.json", payload)

    docs, report = load_documents(path)

    assert [d["id"] for d in docs] == ["ok-1"]
    assert report.excluded_count == 3
    joined = "\n".join(report.errors)
    for bad in ("hamster-1", "empty-species", "upper-dog"):
        assert bad in joined
    assert "species" in joined


def test_document_without_body_is_excluded_with_reason(tmp_path: Path) -> None:
    """본문이 비어 있는 문서를 제외하고 사유를 남기는지 확인한다.

    본문 없는 문서는 chunk 도 임베딩도 만들 수 없으므로 인덱싱해 봐야 의미가 없다.
    다만 '왜 빠졌는지'는 남겨야 수집 스크립트를 고칠 수 있다.
    markdown 이 비어도 text 가 있으면 통과해야 한다(명세 10절 폴백).
    """
    payload = [
        _valid_doc("both-empty", content_markdown="", content_text=""),
        _valid_doc("whitespace-only", content_markdown="   \n\n  ", content_text="\t"),
        _valid_doc("text-only", content_markdown="", content_text="본문만 있습니다."),
    ]
    path = _write_json(tmp_path, "body.json", payload)

    docs, report = load_documents(path)

    assert [d["id"] for d in docs] == ["text-only"]
    assert report.excluded_count == 2
    joined = "\n".join(report.errors)
    assert "both-empty" in joined and "whitespace-only" in joined
    assert "content_markdown" in joined


def test_normalize_documents_drops_body_that_vanishes_after_cleaning() -> None:
    """정규화 후 본문이 사라지는 문서를 chunk 대상에서 빼는지 확인한다.

    제어문자/폭 없는 문자뿐이던 문서는 clean_text 후 빈 문자열이 된다.
    이런 문서를 chunk 로 만들면 벡터스토어에 의미 없는 벡터만 늘어난다.
    """
    # 폭 없는 공백(U+200B) + NUL 만 들어 있는 문서. 실제 문자를 소스에 넣으면
    # 눈에 보이지 않아 편집 중 소실되므로 반드시 이스케이프로 적는다.
    ghost = _valid_doc(
        "ghost", content_markdown="\u200b\u200b\x00", content_text="\u200b"
    )
    real = _valid_doc("real")

    result = normalize_documents([ghost, real])

    assert [d["id"] for d in result] == ["real"]
    assert chunk_document(normalize_document(ghost)) == []


# ---------------------------------------------------------------------------
# 8. headings 없는 문서(14건)
# ---------------------------------------------------------------------------
def test_documents_without_headings_still_produce_chunks(documents: list[dict]) -> None:
    """headings 가 없는 14개 문서도 chunk 가 생성되고 heading_path 가 []인지 확인한다.

    heading 기반 분리에만 의존하면 이 14건이 통째로 인덱싱에서 빠져 버린다.
    이 경우엔 splitter 로만 분할하고, 없는 절 이름을 지어내지 않기 위해
    heading_path 는 빈 리스트로 남겨야 한다.
    """
    without_headings = [doc for doc in documents if not doc.get("headings")]
    assert len(without_headings) == EXPECTED_NO_HEADING_DOCS

    for doc in without_headings:
        normalized = normalize_document(doc)
        chunks = chunk_document(normalized)
        assert chunks, f"{doc['id']}: heading 없는 문서에서 chunk 가 생성되지 않았다"
        for chunk in chunks:
            assert chunk.metadata["heading_path"] == []
            assert chunk.text.strip()


def test_heading_path_accumulates_parent_headings() -> None:
    """heading_path 가 상위 heading 을 누적하고 h1(문서 제목)은 제외하는지 확인한다.

    h1 은 metadata.title 과 중복이라 경로에 넣으면 인용 문구가 "제목 > 제목 > 절"이 된다.
    반대로 상위 h2 를 누적하지 않으면 "Treatment" 만 남아 어느 질환의 치료인지 알 수 없다.
    """
    body = (
        "# Feline Pancreatitis\n\n" + "개요 문단입니다. " * 30 + "\n\n"
        "## Symptoms\n\n" + "증상 문단입니다. " * 30 + "\n\n"
        "### Acute\n\n" + "급성 문단입니다. " * 30 + "\n\n"
        "## Treatment\n\n" + "치료 문단입니다. " * 30 + "\n"
    )
    doc = _valid_doc("cornell:cat:panc", species="cat", content_markdown=body)

    chunks = chunk_document(normalize_document(doc))
    paths = [tuple(c.metadata["heading_path"]) for c in chunks]

    assert () in paths or paths[0] == ()  # h1 아래 preamble 은 경로가 비어 있다
    assert ("Symptoms",) in paths
    assert ("Symptoms", "Acute") in paths  # 상위 heading 누적
    assert ("Treatment",) in paths
    assert all("Feline Pancreatitis" not in p for p in paths)  # h1 은 경로에 없다


# ---------------------------------------------------------------------------
# 9. chunk_id 유일성 / 형식
# ---------------------------------------------------------------------------
def test_chunk_ids_are_unique_and_well_formed(
    subset_chunks: list[Chunk], subset_documents: list[dict]
) -> None:
    """chunk_id 가 전역 유일하고 `{document_id}::0001` 형식인지 확인한다.

    chunk_id 는 retriever 의 중복 제거 키이자 인용 앵커다. 충돌하면 서로 다른 근거가
    하나로 접혀 사라지고, 형식이 흔들리면 id 만 보고 문서를 되짚을 수 없다.
    """
    ids = [chunk.chunk_id for chunk in subset_chunks]
    assert len(set(ids)) == len(ids), "chunk_id 중복 발생"

    per_document: dict[str, list[int]] = {}
    for chunk in subset_chunks:
        prefix, separator, sequence = chunk.chunk_id.rpartition("::")
        assert separator == "::"
        assert prefix == chunk.document_id
        assert len(sequence) == 4 and sequence.isdigit()
        per_document.setdefault(prefix, []).append(int(sequence))

    # 문서마다 1부터 빈틈 없이 증가해야 한다
    for document_id, sequences in per_document.items():
        assert sequences == list(range(1, len(sequences) + 1)), document_id

    assert set(per_document) <= {doc["id"] for doc in subset_documents}


def test_chunk_stats_reports_consistent_totals(subset_chunks: list[Chunk]) -> None:
    """chunk_stats 통계가 실제 chunk 목록과 일치하는지 확인한다.

    노트북에서 이 숫자만 보고 chunking 이상을 판단하므로, 통계가 실제와
    어긋나면 회귀를 놓친다. 빈 입력에서도 예외 없이 0 통계를 줘야 한다.
    """
    stats = chunk_stats(subset_chunks)

    assert stats["total_chunks"] == len(subset_chunks)
    assert stats["document_count"] == len({c.document_id for c in subset_chunks})
    assert stats["min_length"] == min(len(c.text) for c in subset_chunks)
    assert stats["max_length"] == max(len(c.text) for c in subset_chunks)
    assert 0.0 <= stats["heading_path_ratio"] <= 1.0
    assert stats["chunks_with_heading_path"] == sum(
        1 for c in subset_chunks if c.metadata["heading_path"]
    )

    empty = chunk_stats([])
    assert empty["total_chunks"] == 0 and empty["heading_path_ratio"] == 0.0


# ---------------------------------------------------------------------------
# 10. select_body 우선순위
# ---------------------------------------------------------------------------
def test_select_body_prefers_markdown_then_falls_back_to_text() -> None:
    """select_body 가 content_markdown 우선, 없으면 content_text 인지 확인한다.

    markdown 을 우선하는 이유는 heading 구조가 남아 있어야 heading_path 를 만들 수
    있기 때문이다. 순서가 뒤집히면 모든 chunk 의 heading_path 가 [] 로 무너진다.
    """
    assert select_body({"content_markdown": "# MD", "content_text": "TXT"}) == "# MD"
    assert select_body({"content_markdown": "", "content_text": "TXT"}) == "TXT"
    assert select_body({"content_markdown": "   \n ", "content_text": "TXT"}) == "TXT"
    assert select_body({"content_text": "TXT"}) == "TXT"
    assert select_body({"content_markdown": "", "content_text": ""}) == ""
    assert select_body({}) == ""
    assert select_body(None) == ""  # type: ignore[arg-type]

    # 타입이 문자열이 아니면 무시하고 다음 후보로 넘어간다
    assert select_body({"content_markdown": 123, "content_text": "TXT"}) == "TXT"

    # 이미 정규화된 문서는 확정된 body 를 그대로 쓴다(선택 로직 재실행 방지)
    assert select_body({"body": "확정본문", "content_markdown": "# MD"}) == "확정본문"


def test_chunker_uses_markdown_before_text() -> None:
    """chunker 가 실제로 markdown 을 본문으로 골라 heading 을 살리는지 확인한다.

    select_body 단위 동작이 맞아도 chunker 가 content_text 를 집어 오면
    heading 구조가 사라진다. 경로가 실제로 채워지는지까지 확인해야 의미가 있다.
    """
    doc = _valid_doc(
        "md-first",
        content_markdown="# T\n\n## Symptoms\n\n" + ("증상 " * 200),
        content_text="heading 없는 평문 " * 200,
    )

    chunks = chunk_document(normalize_document(doc))

    assert chunks
    assert any(c.metadata["heading_path"] == ["Symptoms"] for c in chunks)
    assert all("heading 없는 평문" not in c.text for c in chunks)


# ---------------------------------------------------------------------------
# 11. normalize_document 불변성
# ---------------------------------------------------------------------------
def test_normalize_document_does_not_mutate_input() -> None:
    """normalize_document 가 원본 dict 를 변형하지 않는지 확인한다.

    로더 리포트/디버깅에서 같은 원본 문서를 다시 봐야 하는데, 제자리 수정이면
    "정규화 전 값"을 영영 볼 수 없다. 중첩된 리스트(categories/headings)까지
    새 객체여야 나중에 정규화 결과를 고쳐도 원본이 오염되지 않는다.
    """
    original = _valid_doc(
        "immutable",
        title="  Spaced   Title  ",
        categories=["  A  ", "A", "B"],
        headings=[{"level": "3", "text": "  Sec  "}],
        content_markdown="# T\n\n\n\n본문\u200b   내용",
    )
    snapshot = json.loads(json.dumps(original))

    normalized = normalize_document(original)

    assert original == snapshot, "원본 dict 가 변형되었다"
    assert normalized is not original
    assert normalized["categories"] is not original["categories"]
    assert normalized["headings"] is not original["headings"]

    # 정규화 결과 자체는 기대대로 정리돼 있어야 한다
    assert normalized["title"] == "Spaced Title"
    assert normalized["categories"] == ["A", "B"]
    assert normalized["headings"] == [{"level": 3, "text": "Sec"}]
    assert "\u200b" not in normalized["content_markdown"]
    assert normalized["body"] == normalized["content_markdown"]

    # 결과를 수정해도 원본은 그대로여야 한다
    normalized["categories"].append("C")
    assert original["categories"] == snapshot["categories"]


def test_normalize_documents_does_not_mutate_loaded_documents(
    subset_documents: list[dict], documents: list[dict]
) -> None:
    """정규화 파이프라인이 loader 가 돌려준 문서 목록을 건드리지 않는지 확인한다.

    subset_documents 픽스처는 documents 픽스처를 입력으로 쓴다. 정규화가 제자리
    수정이면 뒤이어 실행되는 로더 통계 테스트가 오염된 데이터를 보게 된다.
    """
    assert subset_documents  # 픽스처가 이미 실행된 뒤여야 의미가 있다
    for doc in documents:
        assert "body" not in doc, f"{doc['id']}: 원본에 정규화 산출물이 새어 들어갔다"


def test_normalize_document_rejects_non_dict() -> None:
    """dict 가 아닌 입력에 TypeError 를 던지는지 확인한다.

    조용히 빈 문서를 만들어 넘기면 본문 없는 chunk 가 인덱싱되고,
    문제의 출처가 파이프라인 뒤쪽에서야 드러나 원인 추적이 어려워진다.
    """
    with pytest.raises(TypeError):
        normalize_document("문서가 아님")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# 12. zip 로드 동등성
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not RAW_ZIP.exists(), reason=f"raw.zip 이 없습니다: {RAW_ZIP}")
def test_load_documents_from_zip_matches_file_load(
    documents: list[dict], report: LoadReport
) -> None:
    """zip 에서 바로 읽어도 파일 로드와 완전히 동일한 결과인지 확인한다.

    Colab 에서는 압축 해제 단계를 건너뛰고 zip 을 그대로 읽는다. 두 경로의 결과가
    다르면 로컬에서 검증한 인덱스와 Colab 인덱스가 달라져 재현성이 깨진다.
    """
    zip_docs, zip_report = load_documents_from_zip(RAW_ZIP)

    assert zip_report.total_raw == report.total_raw
    assert zip_report.total_valid == report.total_valid
    assert zip_report.dog_count == report.dog_count
    assert zip_report.cat_count == report.cat_count
    assert zip_report.duplicate_count == report.duplicate_count
    assert zip_report.excluded_count == report.excluded_count
    assert zip_report.average_content_length == report.average_content_length

    assert [d["id"] for d in zip_docs] == [d["id"] for d in documents]
    assert [d["content_hash"] for d in zip_docs] == [d["content_hash"] for d in documents]
    assert [sorted(d["categories"]) for d in zip_docs] == [
        sorted(d["categories"]) for d in documents
    ]


def test_load_documents_from_zip_resolves_member_and_reports_errors(
    tmp_path: Path,
) -> None:
    """zip 내부 경로가 한 단계 어긋나도 찾아내고, 없으면 사유를 남기는지 확인한다.

    업로드 방식에 따라 최상위 폴더가 붙거나 빠지는데, 그때마다 로드가 실패하면
    Colab 에서 원인을 찾기 어렵다. 반대로 정말 없을 때는 조용히 빈 결과를 주지 말고
    zip 안 항목 목록을 오류에 담아야 한다.
    """
    payload = [_valid_doc("zip-1"), _valid_doc("zip-2", content_hash="hash-zip-2")]

    archive = tmp_path / "nested.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr(
            "bundle/raw/cornell_pet_health_documents.json",
            json.dumps(payload, ensure_ascii=False),
        )

    docs, report = load_documents_from_zip(archive)
    assert [d["id"] for d in docs] == ["zip-1", "zip-2"]
    assert report.total_raw == 2 and report.excluded_count == 0

    empty = tmp_path / "empty.zip"
    with zipfile.ZipFile(empty, "w") as zf:
        zf.writestr("readme.txt", "no documents here")
    docs, report = load_documents_from_zip(empty)
    assert docs == []
    assert any("찾을 수 없" in e for e in report.errors)

    missing = tmp_path / "없는.zip"
    docs, report = load_documents_from_zip(missing)
    assert docs == [] and report.errors

    not_a_zip = tmp_path / "fake.zip"
    not_a_zip.write_text("zip 이 아님", encoding="utf-8")
    docs, report = load_documents_from_zip(not_a_zip)
    assert docs == [] and any("zip" in e for e in report.errors)


# ---------------------------------------------------------------------------
# 보조 — 리포트 요약 출력
# ---------------------------------------------------------------------------
def test_report_summary_contains_spec_numbers(report: LoadReport) -> None:
    """summary() 가 명세 9절 출력 블록의 숫자를 담는지 확인한다.

    노트북 셀에서 사람이 눈으로 확인하는 유일한 출력이므로, 숫자가 빠지면
    데이터 이상을 알아챌 기회 자체가 없어진다.
    """
    text = report.summary()
    for value in (EXPECTED_TOTAL_RAW, EXPECTED_DOG, EXPECTED_CAT, EXPECTED_DUPLICATES):
        assert str(value) in text
