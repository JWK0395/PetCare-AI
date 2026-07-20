"""PDF 생성 패키지 — 수의사 상담용 고정 template 문서(명세 37절).

LangGraph 노드는 이 패키지의 `generate_consultation_pdf` 만 호출하면 되고,
reportlab 의존성은 `consultation_pdf` 내부에서 지연 import 로 처리한다.
"""

from __future__ import annotations

from .consultation_pdf import (
    KOREAN_FONT_CANDIDATES,
    UNKNOWN_LABEL,
    find_korean_font,
    generate_consultation_pdf,
)

__all__ = [
    "generate_consultation_pdf",
    "find_korean_font",
    "KOREAN_FONT_CANDIDATES",
    "UNKNOWN_LABEL",
]
