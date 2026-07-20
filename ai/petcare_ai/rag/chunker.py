"""문서 chunking — markdown heading 구조를 보존하면서 검색 단위를 만든다(명세 10절).

Cornell 문서는 heading 이 곧 임상적 주제 경계(증상/원인/치료/예방)이므로,
문자 수만으로 자르면 "증상" 문단과 "치료" 문단이 한 chunk 에 섞여 검색 정확도가 떨어진다.
그래서 다음 3단계를 순서대로 적용한다.

1. markdown heading(``#``~``####``) 기준으로 1차 분리하고, 상위 heading 을 누적해
   ``heading_path`` 로 남긴다. 이 값은 답변에서 "어느 절의 근거인지" 밝히는 데 쓴다.
2. heading 만으로는 여전히 긴 section 은 ``RecursiveCharacterTextSplitter`` 로 추가 분리한다.
3. ``min_chunk_length`` 미만의 짧은 chunk 는 인접 chunk 와 병합한다.
   heading 만 있고 본문이 거의 없는 절이 단독 chunk 가 되면 검색 노이즈가 되기 때문이다.

주의: 실제 데이터 287건 중 14건은 h2 이상 heading 이 전혀 없다(제목 h1 만 존재).
이 문서들도 반드시 chunk 가 나와야 하므로, heading 이 없으면 본문 전체를 splitter 로만
분할하고 ``heading_path`` 는 빈 리스트로 둔다.

chunk 크기 관련 상수(CHUNK_SIZE/CHUNK_OVERLAP/MIN_CHUNK_LENGTH)는 이 파일에 두지 않고
``RagSettings`` 에서만 읽는다(명세 10절: 숫자를 코드에 산재시키지 않는다).
"""

from __future__ import annotations

import re
from typing import Any, Iterable

from pydantic import BaseModel, Field

from ..config import RagSettings, get_settings

# ---------------------------------------------------------------------------
# 상수 — 크기 관련 숫자는 여기 두지 않는다(RagSettings 에서만 읽는다).
# ---------------------------------------------------------------------------

#: ``## Heading`` 형태의 ATX heading 을 인식한다(닫는 ``#`` 는 버린다).
_HEADING_RE = re.compile(r"^(#{1,6})[ \t]+(.+?)[ \t]*#*[ \t]*$")

#: 코드 펜스 — 펜스 안의 ``#`` 는 heading 이 아니다(현재 데이터에는 없지만 방어).
_FENCE_RE = re.compile(r"^(```|~~~)")

#: heading_path 에 포함할 최소 level.
#: level 1 은 문서 제목(metadata.title 과 중복)이므로 경로에 넣지 않는다.
#: 실측상 모든 문서가 h1 을 정확히 1개 가지며 그 값은 title 과 같다.
_MIN_PATH_LEVEL = 2

#: 3개 이상 연속된 빈 줄을 2개로 줄인다(chunk 길이를 공백이 잡아먹지 않게).
_BLANK_RUN_RE = re.compile(r"\n{3,}")

#: chunk_id 순번 자리수 — "{document_id}::0001" (1부터 시작).
_CHUNK_ID_DIGITS = 4


class Chunk(BaseModel):
    """검색 단위 1건. ``metadata`` 는 명세 10절 키를 전부 포함한다."""

    chunk_id: str
    document_id: str
    text: str
    metadata: dict = Field(default_factory=dict)

    @property
    def length(self) -> int:
        """chunk 본문 길이(문자 수) — 통계/디버깅 편의용."""
        return len(self.text)


class _Piece:
    """병합 전 중간 산출물. Chunk 로 확정되기 전까지 heading_path 를 들고 다닌다."""

    __slots__ = ("text", "heading_path")

    def __init__(self, text: str, heading_path: list[str]) -> None:
        self.text = text
        self.heading_path = heading_path


# ---------------------------------------------------------------------------
# 내부 헬퍼
# ---------------------------------------------------------------------------
def _resolve_settings(settings: RagSettings | None) -> RagSettings:
    """설정 주입이 없으면 전역 설정을 쓴다 — 크기 상수의 단일 출처를 지킨다."""
    if settings is not None:
        return settings
    return get_settings().rag


def _select_body(doc: dict) -> str:
    """본문 선택 — ``content_markdown`` 우선, 없으면 ``content_text``(명세 10절).

    normalizer 모듈이 있으면 그쪽 구현을 그대로 쓴다(동작 단일화).
    아직 없거나 import 에 실패하면 동일 규칙의 최소 구현으로 대체해,
    chunker 만 단독으로도 검증할 수 있게 한다.
    """
    try:
        from .normalizer import select_body  # 지연 import: 동시 작성 중일 수 있다.
    except Exception:  # pragma: no cover - normalizer 미존재/미완성 시 경로
        markdown = doc.get("content_markdown") or ""
        if isinstance(markdown, str) and markdown.strip():
            return markdown
        text = doc.get("content_text") or ""
        return text if isinstance(text, str) else ""
    return select_body(doc)


def _clean_text(text: str) -> str:
    """개행 정규화 + 과다 공백 줄 축약. 내용은 바꾸지 않는다."""
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    normalized = _BLANK_RUN_RE.sub("\n\n", normalized)
    return normalized.strip()


def _split_into_sections(body: str) -> list[_Piece]:
    """markdown heading 기준 1차 분리.

    heading 을 만나면 스택을 갱신해 상위 heading 을 누적한다.
    (``## A`` → ``### A-1`` 이면 heading_path = ["A", "A-1"])
    heading 줄 자체는 chunk 본문에 남긴다 — 임베딩에 주제어가 포함되는 편이 유리하다.
    heading 이 하나도 없으면 문서 전체가 heading_path=[] 인 section 1개가 된다.
    """
    sections: list[_Piece] = []
    stack: list[tuple[int, str]] = []
    current_lines: list[str] = []
    current_path: list[str] = []
    in_fence = False

    def flush() -> None:
        text = _clean_text("\n".join(current_lines))
        if text:
            sections.append(_Piece(text, list(current_path)))

    for line in body.split("\n"):
        if _FENCE_RE.match(line.strip()):
            in_fence = not in_fence
            current_lines.append(line)
            continue

        match = None if in_fence else _HEADING_RE.match(line)
        if match is None:
            current_lines.append(line)
            continue

        # 새 heading → 직전 section 을 확정한다.
        flush()
        current_lines = [line]

        level = len(match.group(1))
        title = match.group(2).strip()
        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, title))
        current_path = [text for lvl, text in stack if lvl >= _MIN_PATH_LEVEL]

    flush()
    return sections


def _split_long_text(text: str, settings: RagSettings) -> list[str]:
    """긴 section 을 RecursiveCharacterTextSplitter 로 추가 분리한다(2단계).

    langchain_text_splitters 는 무거운 선택 의존성이므로 지연 import 한다.
    """
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
    except ImportError as exc:  # pragma: no cover - 패키지 미설치 환경
        raise ImportError(
            "chunking 에 langchain-text-splitters 패키지가 필요합니다. "
            "`pip install langchain-text-splitters` 로 설치하세요."
        ) from exc

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        # 문단 → 줄 → 문장 → 단어 순으로 끊어야 의미 경계가 덜 깨진다.
        separators=["\n\n", "\n", ". ", "? ", "! ", " ", ""],
        length_function=len,
        keep_separator=True,
    )
    parts = [part.strip() for part in splitter.split_text(text)]
    return [part for part in parts if part]


def _is_heading_only(text: str) -> bool:
    """본문 없이 heading 줄로만 이루어진 조각인지 판단한다.

    ``# Pneumonia`` 처럼 제목만 남은 조각은 그 자체로 검색 근거가 될 수 없다
    (제목은 이미 모든 chunk 의 metadata.title 에 들어 있다).
    이런 조각은 단독 chunk 로 남기면 벡터스토어에 노이즈만 추가되므로
    크기 상한을 조금 넘기더라도 인접 chunk 에 붙인다.
    """
    lines = [line for line in text.split("\n") if line.strip()]
    return bool(lines) and all(_HEADING_RE.match(line) for line in lines)


def _common_heading_path(left: list[str], right: list[str]) -> list[str]:
    """병합된 chunk 의 heading_path — 두 경로의 공통 조상을 쓴다.

    한쪽이 비어 있으면(문서 앞머리 preamble 이 뒤 절과 합쳐진 경우) 비어 있지 않은
    쪽을 쓴다. 병합된 본문이 실제로 그 절에 속하므로 경로를 버릴 이유가 없다.
    공통 조상도 없고 양쪽 다 값이 있으면(서로 다른 대주제) chunk 가 시작되는 지점의
    경로를 남긴다 — 없는 맥락을 지어내지 않기 위함이다.
    """
    if not left:
        return list(right)
    if not right:
        return list(left)
    common: list[str] = []
    for a, b in zip(left, right):
        if a != b:
            break
        common.append(a)
    if common:
        return common
    return list(left)


def _merge_short_pieces(pieces: list[_Piece], settings: RagSettings) -> list[_Piece]:
    """3단계 — ``min_chunk_length`` 미만 chunk 를 인접 chunk 와 병합한다.

    앞/뒤 어느 쪽으로도 병합할 수 있다: 짧은 조각이 들어오면 직전 조각에 붙이고,
    직전 조각이 짧으면 다음 조각을 끌어와 붙인다(둘 다 이 루프의 같은 조건으로 처리된다).
    단, 합친 결과가 ``chunk_size`` 를 넘으면 병합하지 않는다 — 병합 때문에
    chunk 가 상한을 크게 초과하면 임베딩 품질이 떨어지기 때문이다.
    예외적으로 heading 뿐인 조각(본문 0줄)은 단독으로 두면 검색 노이즈가 되므로
    ``min_chunk_length`` 만큼의 초과는 허용하고 붙인다.
    문서 전체가 짧아 어느 쪽으로도 병합할 수 없는 조각은 버리지 않고 그대로 둔다.
    """
    merged: list[_Piece] = []
    for piece in pieces:
        if merged:
            previous = merged[-1]
            combined_length = len(previous.text) + 2 + len(piece.text)
            too_short = (
                len(previous.text) < settings.min_chunk_length
                or len(piece.text) < settings.min_chunk_length
            )
            limit = settings.chunk_size
            if _is_heading_only(previous.text) or _is_heading_only(piece.text):
                limit += settings.min_chunk_length
            if too_short and combined_length <= limit:
                previous.text = f"{previous.text}\n\n{piece.text}"
                previous.heading_path = _common_heading_path(
                    previous.heading_path, piece.heading_path
                )
                continue
        merged.append(piece)
    return merged


def _build_metadata(doc: dict, chunk_id: str, heading_path: list[str]) -> dict[str, Any]:
    """명세 10절 metadata 키를 전부 채운다.

    값이 없어도 키 자체는 빠뜨리지 않는다 — 벡터스토어/인용 단계에서
    ``KeyError`` 대신 빈 값으로 안전하게 처리하기 위함이다.
    """
    categories = doc.get("categories") or []
    if isinstance(categories, str):
        categories = [categories]
    return {
        "document_id": str(doc.get("id") or ""),
        "chunk_id": chunk_id,
        "species": doc.get("species") or "",
        "title": doc.get("title") or "",
        "source": doc.get("source") or "",
        "source_url": doc.get("source_url") or "",
        "categories": [str(c) for c in categories],
        "last_updated": doc.get("last_updated") or "",
        "medical_domain": doc.get("medical_domain") or "",
        "language": doc.get("language") or "en",
        "content_hash": doc.get("content_hash") or "",
        "heading_path": list(heading_path),
    }


# ---------------------------------------------------------------------------
# 공개 API
# ---------------------------------------------------------------------------
def chunk_document(doc: dict, settings: RagSettings | None = None) -> list[Chunk]:
    """문서 1건을 chunk 리스트로 변환한다(명세 10절 3단계 전략).

    heading 이 없는 문서(실측 14건)도 정상 처리된다 — 이 경우 splitter 로만 분할하고
    ``heading_path`` 는 빈 리스트가 된다.
    본문이 비어 있으면 빈 리스트를 반환한다(예외를 던지지 않는다. 로딩 단계에서
    이미 걸러졌어야 하는 문제이고, 여기서 파이프라인을 멈출 이유가 없다).
    """
    resolved = _resolve_settings(settings)
    document_id = str(doc.get("id") or "")

    body = _clean_text(_select_body(doc))
    if not body:
        return []

    # 1단계: heading 기준 분리. heading 이 없으면 section 1개(heading_path=[]).
    sections = _split_into_sections(body)

    # 2단계: 긴 section 추가 분리.
    pieces: list[_Piece] = []
    for section in sections:
        if len(section.text) <= resolved.chunk_size:
            pieces.append(section)
            continue
        for part in _split_long_text(section.text, resolved):
            pieces.append(_Piece(part, list(section.heading_path)))

    # 3단계: 짧은 chunk 병합.
    pieces = _merge_short_pieces(pieces, resolved)

    chunks: list[Chunk] = []
    for index, piece in enumerate(pieces, start=1):
        chunk_id = f"{document_id}::{index:0{_CHUNK_ID_DIGITS}d}"
        chunks.append(
            Chunk(
                chunk_id=chunk_id,
                document_id=document_id,
                text=piece.text,
                metadata=_build_metadata(doc, chunk_id, piece.heading_path),
            )
        )
    return chunks


def chunk_documents(
    docs: list[dict], settings: RagSettings | None = None
) -> list[Chunk]:
    """문서 리스트 전체를 chunking 한다.

    한 문서에서 예외가 나도 전체가 멈추지 않도록 하지는 않는다 — 데이터 이상은
    조용히 넘기지 말고 드러나야 하기 때문이다(loader 가 이미 검증을 끝낸 입력을 받는다).
    """
    resolved = _resolve_settings(settings)
    chunks: list[Chunk] = []
    for doc in docs:
        chunks.extend(chunk_document(doc, resolved))
    return chunks


def chunk_stats(chunks: list[Chunk]) -> dict:
    """chunking 결과 통계 — 노트북에서 분포를 눈으로 확인하기 위한 값들.

    ``heading_path_ratio`` 가 지나치게 낮으면 heading 파싱이 깨진 것이고,
    ``over_chunk_size`` 가 0 이 아니면 병합 로직이 상한을 넘긴 것이므로 즉시 알 수 있다.
    """
    if not chunks:
        return {
            "total_chunks": 0,
            "document_count": 0,
            "average_chunks_per_document": 0.0,
            "average_length": 0.0,
            "min_length": 0,
            "max_length": 0,
            "median_length": 0.0,
            "chunks_with_heading_path": 0,
            "heading_path_ratio": 0.0,
            "short_chunks": 0,
            "over_chunk_size": 0,
        }

    rag = get_settings().rag
    lengths = sorted(len(chunk.text) for chunk in chunks)
    document_ids = {chunk.document_id for chunk in chunks}
    with_path = sum(1 for chunk in chunks if chunk.metadata.get("heading_path"))
    total = len(chunks)
    middle = total // 2
    median = (
        float(lengths[middle])
        if total % 2
        else (lengths[middle - 1] + lengths[middle]) / 2
    )

    return {
        "total_chunks": total,
        "document_count": len(document_ids),
        "average_chunks_per_document": round(total / len(document_ids), 2),
        "average_length": round(sum(lengths) / total, 1),
        "min_length": lengths[0],
        "max_length": lengths[-1],
        "median_length": median,
        "chunks_with_heading_path": with_path,
        "heading_path_ratio": round(with_path / total, 3),
        "short_chunks": sum(1 for n in lengths if n < rag.min_chunk_length),
        "over_chunk_size": sum(1 for n in lengths if n > rag.chunk_size),
    }


def iter_chunk_texts(chunks: Iterable[Chunk]) -> list[str]:
    """임베딩 입력용 텍스트 리스트 — vector store 에서 그대로 쓸 수 있게 제공한다."""
    return [chunk.text for chunk in chunks]
