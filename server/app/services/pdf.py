"""병원 전달용 요약 PDF 생성.

reportlab 로 요약을 A4 문서로 렌더링한다. 한글 폰트가 필요하므로
시스템의 맑은 고딕(Windows) 등을 등록해서 사용한다.
"""

from io import BytesIO
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

# 한글 폰트 후보 (플랫폼별) — 처음 발견되는 것을 등록한다.
_FONT_CANDIDATES = [
    r"C:\Windows\Fonts\malgun.ttf",
    r"C:\Windows\Fonts\NanumGothic.ttf",
    "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
    "/System/Library/Fonts/AppleSDGothicNeo.ttc",
]

_FONT_NAME = "Helvetica"  # 폴백 (한글 미지원)


def _ensure_font() -> str:
    global _FONT_NAME
    if _FONT_NAME != "Helvetica":
        return _FONT_NAME
    for path in _FONT_CANDIDATES:
        if Path(path).exists():
            try:
                pdfmetrics.registerFont(TTFont("KR", path))
                _FONT_NAME = "KR"
                break
            except Exception:
                continue
    return _FONT_NAME


def build_summary_pdf(pet_name: str, content: dict, created_at: str) -> bytes:
    """병원 전달용 상태 요약을 문서 4섹션 구조로 렌더링한다."""
    font = _ensure_font()
    buf = BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4, topMargin=20 * mm, bottomMargin=20 * mm,
        leftMargin=18 * mm, rightMargin=18 * mm,
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "KRTitle", parent=styles["Title"], fontName=font, fontSize=17
    )
    section_style = ParagraphStyle(
        "KRSection", parent=styles["Normal"], fontName=font, fontSize=13,
        textColor="#2F6BFF", spaceBefore=8, spaceAfter=4, alignment=TA_LEFT,
    )
    row_style = ParagraphStyle(
        "KRRow", parent=styles["Normal"], fontName=font, fontSize=11, leading=17,
    )
    meta_style = ParagraphStyle(
        "KRMeta", parent=styles["Normal"], fontName=font, fontSize=9,
        textColor="#6B7280",
    )

    title = content.get("title") or "PetCare AI 병원 전달용 상태 요약"
    story = [
        Paragraph(escape(title), title_style),
        Spacer(1, 5 * mm),
    ]

    def section(name: str) -> None:
        story.append(Paragraph(name, section_style))

    def row(label: str, value: str) -> None:
        # Paragraph 는 <b> 같은 인라인 마크업을 해석하므로 사용자/Agent 텍스트는
        # 반드시 이스케이프한다 ('<1정/일>' 같은 입력이 렌더링 오류를 내지 않게).
        story.append(Paragraph(f"- {escape(label)}: {escape(str(value)) if value else '-'}", row_style))

    # 1. 문서 정보
    section("1. 문서 정보")
    row("문서 제목", title)
    row("생성 일시", created_at)
    row("사용 데이터 기간", content.get("data_period", ""))

    # 2. 반려동물 정보
    section("2. 반려동물 정보")
    row("이름", content.get("pet_name", pet_name))
    row("종", content.get("species", ""))
    row("품종", content.get("breed", ""))
    row("성별/중성화", content.get("sex_neuter", ""))
    row("나이", content.get("age_label", ""))
    row("현재 체중", content.get("weight", ""))
    row("현재 복용 중인 약", content.get("medications", ""))
    row("알레르기", content.get("allergies", ""))

    # 3. 상태
    section("3. 상태")
    row("상태 분류", content.get("risk_label", ""))
    story.append(Paragraph("- 확인된 위험 징후", row_style))
    signs = content.get("risk_signs") or []
    if signs:
        for s in signs:
            story.append(Paragraph(f"  * {escape(str(s))}", row_style))
    else:
        story.append(Paragraph("  * 특이 위험 징후 없음", row_style))

    # 4. 주호소 및 주요 변화
    section("4. 주호소 및 주요 변화")
    row("주호소", content.get("chief_complaint", ""))
    row("주요 변화", content.get("major_changes", ""))
    row("경과", content.get("progress", ""))
    if content.get("owner_note"):
        row("보호자 메모", content["owner_note"])

    story.append(Spacer(1, 8 * mm))
    story.append(
        Paragraph(
            "이 요약은 보호자 기록 기반 참고 자료이며, 수의사의 진단을 대체하지 않습니다.",
            meta_style,
        )
    )

    doc.build(story)
    return buf.getvalue()
