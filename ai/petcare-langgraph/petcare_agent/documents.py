from __future__ import annotations

from pathlib import Path
from typing import Any

from .models import HandoffDocument

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import (
    ParagraphStyle,
    getSampleStyleSheet,
)
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


FONT_CANDIDATES = [
    Path(
        "C:/Windows/Fonts/malgun.ttf"
    ),
    Path(
        "/usr/share/fonts/truetype/"
        "nanum/NanumGothic.ttf"
    ),
    Path(
        "/usr/share/fonts/truetype/"
        "unfonts-core/UnDotum.ttf"
    ),
]


def _register_font() -> str:
    for path in FONT_CANDIDATES:
        if not path.exists():
            continue

        font_name = "PetCareKorean"

        if font_name not in (
            pdfmetrics.getRegisteredFontNames()
        ):
            pdfmetrics.registerFont(
                TTFont(
                    font_name,
                    str(path),
                )
            )

        return font_name

    return "Helvetica"


def _escape(
    value: Any,
) -> str:
    return (
        str(value)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def _paragraph(
    value: Any,
    style: ParagraphStyle,
) -> Paragraph:
    return Paragraph(
        _escape(value).replace(
            "\n",
            "<br/>",
        ),
        style,
    )


def _bullet_text(
    values: list[str],
    *,
    empty_text: str = "기록 없음",
) -> str:
    if not values:
        values = [empty_text]

    return "\n".join(
        f"* {value}"
        for value in values
    )


def _section_table(
    rows: list[tuple[str, Any]],
    *,
    label_style: ParagraphStyle,
    body_style: ParagraphStyle,
) -> Table:
    data = [
        [
            _paragraph(
                label,
                label_style,
            ),
            _paragraph(
                value,
                body_style,
            ),
        ]
        for label, value in rows
    ]

    table = Table(
        data,
        colWidths=[
            42 * mm,
            128 * mm,
        ],
        hAlign="LEFT",
    )

    table.setStyle(
        TableStyle(
            [
                (
                    "VALIGN",
                    (0, 0),
                    (-1, -1),
                    "TOP",
                ),
                (
                    "GRID",
                    (0, 0),
                    (-1, -1),
                    0.4,
                    colors.HexColor(
                        "#C7CDD4"
                    ),
                ),
                (
                    "BACKGROUND",
                    (0, 0),
                    (0, -1),
                    colors.HexColor(
                        "#F3F5F7"
                    ),
                ),
                (
                    "LEFTPADDING",
                    (0, 0),
                    (-1, -1),
                    7,
                ),
                (
                    "RIGHTPADDING",
                    (0, 0),
                    (-1, -1),
                    7,
                ),
                (
                    "TOPPADDING",
                    (0, 0),
                    (-1, -1),
                    6,
                ),
                (
                    "BOTTOMPADDING",
                    (0, 0),
                    (-1, -1),
                    6,
                ),
            ]
        )
    )

    return table


def create_handoff_pdf(
    *,
    handoff: HandoffDocument | dict[str, Any],
    session_id: str,
    output_dir: str | Path = "artifacts",
) -> str:
    handoff = (
        handoff.model_dump()
        if isinstance(handoff, HandoffDocument)
        else HandoffDocument.model_validate(handoff).model_dump()
    )

    target_dir = Path(output_dir)
    target_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    safe_session = "".join(
        character
        for character in session_id
        if (
            character.isalnum()
            or character in {"-", "_"}
        )
    )

    path = (
        target_dir
        / f"{safe_session}_hospital_handoff.pdf"
    )

    font_name = _register_font()
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "PetCareTitle",
        parent=styles["Title"],
        fontName=font_name,
        fontSize=18,
        leading=24,
        textColor=colors.HexColor(
            "#202832"
        ),
        spaceAfter=10,
    )
    section_style = ParagraphStyle(
        "PetCareSection",
        parent=styles["Heading2"],
        fontName=font_name,
        fontSize=12,
        leading=17,
        textColor=colors.HexColor(
            "#263849"
        ),
        spaceBefore=10,
        spaceAfter=5,
    )
    label_style = ParagraphStyle(
        "PetCareLabel",
        parent=styles["BodyText"],
        fontName=font_name,
        fontSize=9.5,
        leading=14,
        textColor=colors.HexColor(
            "#27313B"
        ),
        wordWrap="CJK",
    )
    body_style = ParagraphStyle(
        "PetCareBody",
        parent=styles["BodyText"],
        fontName=font_name,
        fontSize=9.5,
        leading=14,
        textColor=colors.HexColor(
            "#151A20"
        ),
        wordWrap="CJK",
    )

    document = SimpleDocTemplate(
        str(path),
        pagesize=A4,
        rightMargin=18 * mm,
        leftMargin=18 * mm,
        topMargin=16 * mm,
        bottomMargin=16 * mm,
        title=handoff[
            "document_info"
        ]["title"],
        author="PetCare AI",
    )

    info = handoff["document_info"]
    pet = handoff["pet_info"]
    status = handoff["status"]
    clinical = handoff[
        "clinical_summary"
    ]

    story: list[Any] = [
        _paragraph(
            info["title"],
            title_style,
        ),
        Spacer(1, 2 * mm),
        _paragraph(
            "1. 문서 정보",
            section_style,
        ),
        _section_table(
            [
                (
                    "문서 제목",
                    info["title"],
                ),
                (
                    "생성 일시",
                    info["generated_at"],
                ),
                (
                    "사용 데이터 기간",
                    info["data_period"],
                ),
            ],
            label_style=label_style,
            body_style=body_style,
        ),
        _paragraph(
            "2. 반려동물 정보",
            section_style,
        ),
        _section_table(
            [
                (
                    "이름",
                    pet["name"],
                ),
                (
                    "종",
                    pet["species"],
                ),
                (
                    "품종",
                    pet["breed"],
                ),
                (
                    "성별/중성화",
                    pet["sex_neutered"],
                ),
                (
                    "나이",
                    pet["age"],
                ),
                (
                    "현재 체중",
                    pet["weight"],
                ),
                (
                    "현재 복용 중인 약",
                    _bullet_text(
                        pet["medications"]
                    ),
                ),
                (
                    "알레르기",
                    _bullet_text(
                        pet["allergies"]
                    ),
                ),
            ],
            label_style=label_style,
            body_style=body_style,
        ),
        _paragraph(
            "3. 상태",
            section_style,
        ),
        _section_table(
            [
                (
                    "상태 분류",
                    status[
                        "classification"
                    ],
                ),
                (
                    "확인된 위험 징후",
                    _bullet_text(
                        status["risk_signs"],
                        empty_text=(
                            "확인된 응급 위험 "
                            "징후 없음"
                        ),
                    ),
                ),
            ],
            label_style=label_style,
            body_style=body_style,
        ),
        _paragraph(
            "4. 주호소 및 주요 변화",
            section_style,
        ),
        _section_table(
            [
                (
                    "주호소",
                    (
                        ", ".join(
                            clinical[
                                "chief_complaints"
                            ]
                        )
                        or "기록 없음"
                    ),
                ),
                (
                    "주요 변화",
                    _bullet_text(
                        clinical[
                            "major_changes"
                        ]
                    ),
                ),
                (
                    "경과",
                    _bullet_text(
                        clinical["course"]
                    ),
                ),
            ],
            label_style=label_style,
            body_style=body_style,
        ),
    ]

    document.build(story)
    return str(path)
