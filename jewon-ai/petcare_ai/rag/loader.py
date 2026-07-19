"""Cornell 반려동물 건강 문서 로더 (명세 9절).

수의학 근거의 출발점이므로 "조용히 고치는" 동작을 하지 않는다.
- 비정상 문서는 삭제하지 않고 `LoadReport.errors` 에 사유를 남긴다.
  (근거 문서가 왜 빠졌는지 추적할 수 없으면 답변 품질 회귀를 진단할 수 없다.)
- 중복은 `content_hash` -> `id` -> `source_url` 순으로 제거한다.
  같은 본문이 서로 다른 `categories` 로 2번 등재된 경우가 실제로 존재하므로
  (예: `cornell:cat:feline-pancreatitis` = Gastrointestinal Issues / Pancreatitis)
  중복을 버릴 때 **categories 는 합집합으로 병합**한다. 그냥 버리면
  "췌장염" 질의가 해당 문서를 찾지 못하는 검색 품질 손실이 생긴다.

표준 라이브러리 + pydantic 만 사용한다(지연 import 대상 없음).
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Any, Iterable

from pydantic import BaseModel, Field

# 문서 1건이 반드시 가져야 하는 필드. 이 중 하나라도 비면 근거로 인용할 수 없다.
REQUIRED_FIELDS: tuple[str, ...] = (
    "id",
    "species",
    "source",
    "source_url",
    "title",
    "content_hash",
)

# species 는 vector store 분리 기준이라 임의 값이 섞이면 안 된다.
VALID_SPECIES: frozenset[str] = frozenset({"dog", "cat"})

# 명세 9절 중복 제거 우선순위. 앞에 있는 키일수록 강한 동일성 근거다.
DEDUPE_KEY_PRIORITY: tuple[str, ...] = ("content_hash", "id", "source_url")

# raw.zip 내부의 기본 경로
DEFAULT_ZIP_MEMBER: str = "raw/cornell_pet_health_documents.json"


class LoadReport(BaseModel):
    """로딩 통계 — 명세 9절이 요구하는 출력 항목을 그대로 담는다."""

    total_raw: int = 0
    total_valid: int = 0
    dog_count: int = 0
    cat_count: int = 0
    excluded_count: int = 0
    duplicate_count: int = 0
    average_content_length: float = 0.0
    errors: list[str] = Field(default_factory=list)

    def summary(self) -> str:
        """명세 9절 출력 블록 형태의 사람이 읽는 요약."""
        return (
            f"전체 문서 수: {self.total_raw}\n"
            f"강아지 문서 수: {self.dog_count}\n"
            f"고양이 문서 수: {self.cat_count}\n"
            f"제외된 문서 수: {self.excluded_count}\n"
            f"중복 문서 수: {self.duplicate_count}\n"
            f"평균 본문 길이: {self.average_content_length:.1f}"
        )


# ---------------------------------------------------------------------------
# 내부 헬퍼
# ---------------------------------------------------------------------------
def _body_of(doc: dict[str, Any]) -> str:
    """본문 길이 통계용 본문 — 명세 10절과 동일하게 markdown 우선."""
    for key in ("content_markdown", "content_text"):
        value = doc.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return ""


def _validate_document(doc: Any, index: int) -> tuple[bool, list[str]]:
    """문서 1건을 검증한다.

    삭제 대신 사유를 반환하는 이유: 호출자가 `errors` 를 보고
    데이터 수집 스크립트를 고칠 수 있어야 하기 때문이다.
    """
    errors: list[str] = []

    if not isinstance(doc, dict):
        return False, [f"[{index}] 문서가 dict 가 아닙니다(type={type(doc).__name__})."]

    doc_id = doc.get("id") or f"<index {index}>"

    for field in REQUIRED_FIELDS:
        value = doc.get(field)
        if value is None or (isinstance(value, str) and not value.strip()):
            errors.append(f"[{index}] {doc_id}: 필수 필드 누락 또는 빈 값 — '{field}'")

    species = doc.get("species")
    if species not in VALID_SPECIES:
        errors.append(
            f"[{index}] {doc_id}: species 값이 올바르지 않습니다 — "
            f"{species!r} (허용: dog, cat)"
        )

    if not _body_of(doc):
        errors.append(
            f"[{index}] {doc_id}: content_markdown / content_text 가 모두 비어 있습니다."
        )

    return (not errors), errors


def _merge_duplicate(kept: dict[str, Any], duplicate: dict[str, Any]) -> None:
    """중복 문서를 이미 채택된 문서에 병합한다(제자리 수정).

    categories 는 합집합으로 합친다. 실제 데이터에서 동일 본문이
    서로 다른 분류로 2번 등재되어 있어, 한쪽을 버리면 그 분류로는
    영영 검색되지 않는다. 입력 순서를 유지해 결과를 재현 가능하게 만든다.
    """
    merged: list[str] = list(kept.get("categories") or [])
    seen = {c for c in merged}
    for category in duplicate.get("categories") or []:
        if category not in seen:
            seen.add(category)
            merged.append(category)
    kept["categories"] = merged

    # listing_urls 도 같은 이유로 합집합 병합한다(출처 추적용).
    merged_urls: list[str] = list(kept.get("listing_urls") or [])
    seen_urls = {u for u in merged_urls}
    for url in duplicate.get("listing_urls") or []:
        if url not in seen_urls:
            seen_urls.add(url)
            merged_urls.append(url)
    if merged_urls:
        kept["listing_urls"] = merged_urls


def _dedupe(
    docs: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], int, list[str]]:
    """명세 9절 우선순위대로 중복을 제거하고 categories 를 병합한다.

    `content_hash` 로 먼저 접은 뒤 남은 것을 `id`, `source_url` 순으로 접는다.
    반환: (중복 제거된 문서, 제거된 중복 수, 안내 메시지)
    """
    notes: list[str] = []
    current = [dict(doc) for doc in docs]  # 원본 dict 를 변형하지 않는다
    removed_total = 0

    for key in DEDUPE_KEY_PRIORITY:
        index_by_value: dict[str, int] = {}
        survivors: list[dict[str, Any]] = []
        for doc in current:
            value = doc.get(key)
            if not isinstance(value, str) or not value:
                survivors.append(doc)  # 키가 없으면 중복 판정 불가 — 그대로 통과
                continue
            existing = index_by_value.get(value)
            if existing is None:
                index_by_value[value] = len(survivors)
                survivors.append(doc)
                continue
            _merge_duplicate(survivors[existing], doc)
            removed_total += 1
            notes.append(
                f"중복 제거({key}): {doc.get('id')} — categories 병합 후 "
                f"{survivors[existing].get('categories')}"
            )
        current = survivors

    return current, removed_total, notes


def _build_report(
    total_raw: int,
    valid_docs: list[dict[str, Any]],
    excluded_count: int,
    duplicate_count: int,
    errors: list[str],
) -> LoadReport:
    """명세 9절 통계를 계산한다."""
    lengths = [len(_body_of(doc)) for doc in valid_docs]
    average = (sum(lengths) / len(lengths)) if lengths else 0.0
    return LoadReport(
        total_raw=total_raw,
        total_valid=len(valid_docs),
        dog_count=sum(1 for d in valid_docs if d.get("species") == "dog"),
        cat_count=sum(1 for d in valid_docs if d.get("species") == "cat"),
        excluded_count=excluded_count,
        duplicate_count=duplicate_count,
        average_content_length=round(average, 2),
        errors=errors,
    )


def _load_from_payload(payload: Any, origin: str) -> tuple[list[dict[str, Any]], LoadReport]:
    """파싱된 JSON payload 를 검증·중복 제거해 문서와 리포트를 만든다."""
    if not isinstance(payload, list):
        report = LoadReport(
            errors=[
                f"{origin}: JSON 최상위가 list 가 아닙니다"
                f"(type={type(payload).__name__}). 문서를 로드하지 않았습니다."
            ]
        )
        return [], report

    errors: list[str] = []
    valid: list[dict[str, Any]] = []
    excluded = 0

    for index, raw_doc in enumerate(payload):
        ok, doc_errors = _validate_document(raw_doc, index)
        if ok:
            valid.append(dict(raw_doc))
        else:
            excluded += 1
            errors.extend(doc_errors)  # 삭제하지 않고 사유를 남긴다

    deduped, duplicate_count, notes = _dedupe(valid)
    errors.extend(notes)

    return deduped, _build_report(
        total_raw=len(payload),
        valid_docs=deduped,
        excluded_count=excluded,
        duplicate_count=duplicate_count,
        errors=errors,
    )


# ---------------------------------------------------------------------------
# 공개 API
# ---------------------------------------------------------------------------
def load_documents(path: str | Path) -> tuple[list[dict], LoadReport]:
    """JSON 파일에서 문서를 로드한다.

    파일이 없거나 JSON 이 깨진 경우에도 예외를 던지지 않고 빈 목록 +
    사유가 담긴 리포트를 돌려준다. 노트북 셀 하나가 죽어서 전체 실행이
    끊기는 것보다, 리포트를 보고 원인을 파악하는 편이 낫다.
    """
    file_path = Path(path)
    if not file_path.exists():
        return [], LoadReport(
            errors=[f"문서 파일을 찾을 수 없습니다: {file_path}"]
        )

    try:
        payload = json.loads(file_path.read_text(encoding="utf-8"))
    except UnicodeDecodeError as exc:
        return [], LoadReport(
            errors=[f"{file_path}: UTF-8 로 읽을 수 없습니다 — {exc}"]
        )
    except json.JSONDecodeError as exc:
        return [], LoadReport(
            errors=[f"{file_path}: JSON 파싱에 실패했습니다 — {exc}"]
        )

    return _load_from_payload(payload, origin=str(file_path))


def load_documents_from_zip(
    zip_path: str | Path,
    member: str = DEFAULT_ZIP_MEMBER,
) -> tuple[list[dict], LoadReport]:
    """`raw.zip` 을 풀지 않고 안에서 바로 문서를 읽는다.

    Colab 에서 zip 업로드 후 압축 해제 단계를 생략할 수 있게 하려는 목적이다.
    member 가 없으면 zip 안의 후보 파일명을 오류 메시지에 담아 돌려준다.
    """
    archive_path = Path(zip_path)
    if not archive_path.exists():
        return [], LoadReport(errors=[f"zip 파일을 찾을 수 없습니다: {archive_path}"])

    try:
        with zipfile.ZipFile(archive_path) as archive:
            names = archive.namelist()
            target = member if member in names else _resolve_member(names, member)
            if target is None:
                return [], LoadReport(
                    errors=[
                        f"{archive_path}: zip 안에서 '{member}' 를 찾을 수 없습니다. "
                        f"포함된 항목: {names[:10]}"
                    ]
                )
            raw_bytes = archive.read(target)
    except zipfile.BadZipFile as exc:
        return [], LoadReport(errors=[f"{archive_path}: 올바른 zip 이 아닙니다 — {exc}"])

    try:
        payload = json.loads(raw_bytes.decode("utf-8"))
    except UnicodeDecodeError as exc:
        return [], LoadReport(
            errors=[f"{archive_path}!{member}: UTF-8 디코딩 실패 — {exc}"]
        )
    except json.JSONDecodeError as exc:
        return [], LoadReport(
            errors=[f"{archive_path}!{member}: JSON 파싱 실패 — {exc}"]
        )

    return _load_from_payload(payload, origin=f"{archive_path}!{member}")


def _resolve_member(names: Iterable[str], member: str) -> str | None:
    """zip 내부 경로가 한 단계 어긋나도(폴더 유무) 찾아준다."""
    wanted = Path(member).name
    for name in names:
        if name.endswith("/"):
            continue
        if Path(name).name == wanted:
            return name
    return None
