"""문서 본문 정규화 (명세 10절 '본문 선택').

chunker 가 markdown heading 기준으로 문서를 자르므로, 정규화 단계에서
**heading 구조를 절대 깨뜨리면 안 된다.** 그래서 공백 정리 규칙을
줄 단위로 적용하고, `#` 로 시작하는 줄의 마커와 코드 블록 내부는 건드리지 않는다.

또한 `normalize_document()` 는 입력 dict 를 변형하지 않고 새 dict 를 만든다.
같은 원본 문서를 로더 리포트/디버깅에서 다시 볼 수 있어야 하기 때문이다.

표준 라이브러리만 사용한다.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any

# 본문 후보 필드 — 명세 10절 순서(markdown 우선)
BODY_FIELDS: tuple[str, ...] = ("content_markdown", "content_text")

# 아래 문자 상수들은 반드시 이스케이프로 적는다.
# 실제 문자를 소스에 넣으면 눈에 보이지 않아 편집 중 소실되고,
# NUL 은 파이썬 소스 자체가 거부한다.

# 유지할 제어문자: 개행(\n)/탭(\t) 뿐. 나머지 C0/C1 은 임베딩에 잡음만 준다.
_CONTROL_CHARS = re.compile("[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")

# 폭 없는 문자(zero-width / 방향 제어 / word joiner / BOM)
_ZERO_WIDTH = re.compile("[​-‏‪-‮⁠-⁤﻿]")

# 유니코드 줄바꿈 문자 — 삭제하지 않고 일반 개행으로 바꾼다(문단 경계 보존).
_LINE_SEPARATORS: tuple[str, ...] = (" ", " ")

# 유니코드 공백류(NBSP, EM space, 전각 공백 등)를 일반 공백으로 통일한다.
_UNICODE_SPACE_CODEPOINTS: tuple[str, ...] = (
    "\xa0",    # NO-BREAK SPACE
    " ",  # OGHAM SPACE MARK
    " ",  # EN QUAD
    " ",  # EM QUAD
    " ",  # EN SPACE
    " ",  # EM SPACE
    " ",  # THREE-PER-EM SPACE
    " ",  # FOUR-PER-EM SPACE
    " ",  # SIX-PER-EM SPACE
    " ",  # FIGURE SPACE
    " ",  # PUNCTUATION SPACE
    " ",  # THIN SPACE
    " ",  # HAIR SPACE
    " ",  # NARROW NO-BREAK SPACE
    " ",  # MEDIUM MATHEMATICAL SPACE
    "　",  # IDEOGRAPHIC SPACE
)
_UNICODE_SPACES = str.maketrans({ch: " " for ch in _UNICODE_SPACE_CODEPOINTS})

# 줄 내부의 연속 공백/탭. 앞쪽 들여쓰기는 별도로 보존한다(목록 계층 유지).
_INNER_SPACES = re.compile(r"[ \t]{2,}")

# 3줄 이상 빈 줄 -> 빈 줄 1개(문단 구분은 남긴다)
_EXCESS_BLANK_LINES = re.compile(r"\n{3,}")

# 코드 펜스 (``` 또는 ~~~)
_CODE_FENCE = re.compile(r"^\s*(?:```|~~~)")


def select_body(doc: dict) -> str:
    """본문을 고른다 — `content_markdown` 우선, 없으면 `content_text`.

    markdown 을 우선하는 이유: heading 구조가 남아 있어야 chunk metadata 의
    `heading_path` 를 만들 수 있고, 그래야 근거를 인용할 때 문서의 어느
    section 에서 왔는지 밝힐 수 있다.
    이미 정규화된 문서라면 확정된 `body` 키를 먼저 사용한다.
    """
    if not isinstance(doc, dict):
        return ""

    body = doc.get("body")
    if isinstance(body, str) and body.strip():
        return body

    for field in BODY_FIELDS:
        value = doc.get(field)
        if isinstance(value, str) and value.strip():
            return value
    return ""


def clean_text(text: str) -> str:
    """제어문자·과도한 공백·중복 개행을 정리한다(markdown heading 구조 보존).

    줄 단위로 처리하는 이유: 문서 전체에 정규식을 한 번에 돌리면
    `#` 뒤 공백이나 목록 들여쓰기가 망가져 heading 기반 chunking 이 깨진다.
    """
    if not isinstance(text, str) or not text:
        return ""

    # 1) 유니코드 정규화 + 개행 통일
    normalized = unicodedata.normalize("NFC", text)
    normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
    for separator in _LINE_SEPARATORS:
        normalized = normalized.replace(separator, "\n")

    # 2) 보이지 않는 문자 제거 / 공백류 통일 / 제어문자 제거
    normalized = _ZERO_WIDTH.sub("", normalized)
    normalized = normalized.translate(_UNICODE_SPACES)
    normalized = _CONTROL_CHARS.sub("", normalized)

    # 3) 줄 단위 정리 — 코드 블록 안은 원문 그대로 둔다
    in_code_block = False
    lines: list[str] = []
    for line in normalized.split("\n"):
        if _CODE_FENCE.match(line):
            in_code_block = not in_code_block
            lines.append(line.rstrip())
            continue
        lines.append(line.rstrip() if in_code_block else _clean_line(line))

    cleaned = "\n".join(lines)

    # 4) 과도한 빈 줄 축약 후 전체 트림
    cleaned = _EXCESS_BLANK_LINES.sub("\n\n", cleaned)
    return cleaned.strip()


def _clean_line(line: str) -> str:
    """한 줄의 공백을 정리한다 — 들여쓰기와 heading 마커는 보존한다."""
    stripped_right = line.rstrip()
    if not stripped_right.strip():
        return ""

    # 들여쓰기 보존(목록 계층). 탭은 공백 4칸으로 통일한다.
    indent_len = len(stripped_right) - len(stripped_right.lstrip(" \t"))
    indent = stripped_right[:indent_len].replace("\t", "    ")
    body = stripped_right[indent_len:]

    # 줄 중간의 탭은 폭이 문맥에 따라 달라져 임베딩에 잡음이 되므로 공백 1칸으로 바꾼다.
    # (들여쓰기용 탭은 위에서 이미 분리해 4칸으로 보존했다.)
    body = body.replace("\t", " ")

    # heading 은 `#` 마커가 줄 맨 앞에 그대로 남고 뒤쪽 공백만 축약된다.
    return indent + _INNER_SPACES.sub(" ", body)


def _clean_str_list(values: Any) -> list[str]:
    """문자열 리스트 필드를 정리한다(공백 정리·빈 값 제거·순서 유지 중복 제거)."""
    if not isinstance(values, list):
        return []
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        if not isinstance(value, str):
            continue
        cleaned = clean_text(value).strip()
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            result.append(cleaned)
    return result


def _clean_headings(headings: Any) -> list[dict[str, Any]]:
    """headings 목록을 정리한다 — level(int) 과 text 만 남긴다.

    headings 가 아예 없는 문서가 실제로 존재하므로(14건) 결측을 오류로 보지 않는다.
    """
    if not isinstance(headings, list):
        return []
    result: list[dict[str, Any]] = []
    for item in headings:
        if not isinstance(item, dict):
            continue
        text = clean_text(str(item.get("text", ""))).strip()
        if not text:
            continue
        try:
            level = int(item.get("level", 2))
        except (TypeError, ValueError):
            level = 2
        result.append({"level": level, "text": text})
    return result


def normalize_document(doc: dict) -> dict:
    """문서 1건을 정규화한 **새 dict** 로 반환한다(원본 dict 불변).

    - 본문(`content_markdown` / `content_text`)은 `clean_text()` 로 정리
    - `body` 키에 명세 10절 선택 규칙을 적용한 최종 본문을 확정해 두어
      이후 chunker 가 선택 로직을 다시 구현하지 않게 한다
    - 결측 필드는 빈 값으로 채워 downstream 의 `KeyError` 를 없앤다
      (없는 값을 지어내지는 않는다)
    """
    if not isinstance(doc, dict):
        raise TypeError(f"문서는 dict 여야 합니다: {type(doc).__name__}")

    normalized: dict[str, Any] = dict(doc)  # 얕은 복사 후 필요한 키만 교체

    for field in BODY_FIELDS:
        value = doc.get(field)
        normalized[field] = clean_text(value) if isinstance(value, str) else ""

    normalized["id"] = str(doc.get("id", "")).strip()
    normalized["species"] = str(doc.get("species", "")).strip().lower()
    normalized["title"] = clean_text(str(doc.get("title", ""))).strip()
    normalized["source"] = clean_text(str(doc.get("source", ""))).strip()
    normalized["source_url"] = str(doc.get("source_url", "")).strip()
    normalized["language"] = str(doc.get("language", "") or "en").strip()
    normalized["medical_domain"] = str(doc.get("medical_domain", "")).strip()
    normalized["last_updated"] = str(doc.get("last_updated", "") or "").strip()
    normalized["content_hash"] = str(doc.get("content_hash", "")).strip()

    normalized["categories"] = _clean_str_list(doc.get("categories"))
    normalized["listing_urls"] = _clean_str_list(doc.get("listing_urls"))
    normalized["headings"] = _clean_headings(doc.get("headings"))

    # 명세 10절 본문 선택 결과를 확정한다(이전 값이 있어도 새로 계산).
    normalized.pop("body", None)
    normalized["body"] = select_body(normalized)

    return normalized


def normalize_documents(docs: list[dict]) -> list[dict]:
    """문서 목록을 정규화한다 — 정규화 후 본문이 빈 문서는 제외한다.

    본문이 사라졌다면 제어문자/공백뿐이던 문서라는 뜻이라,
    chunk 로 만들어도 검색 품질에 해만 된다.
    """
    result: list[dict] = []
    for doc in docs:
        normalized = normalize_document(doc)
        if normalized.get("body"):
            result.append(normalized)
    return result
