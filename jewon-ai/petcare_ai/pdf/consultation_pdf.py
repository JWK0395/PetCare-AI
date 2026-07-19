"""수의사 상담용 PDF 생성기 (명세 37절).

설계 원칙:

- **고정 template** 이다. LLM 이 PDF 문장을 자유 생성하지 않는다. 이 모듈은
  `ConsultationPacket` 에 이미 들어있는 값만 정해진 자리에 배치한다.
- **확정 진단을 넣지 않는다.** 위험도는 항상 "AI 참고 분류"로 표기하고,
  문서 마지막에 수의사 확인 필요 문구를 고정 삽입한다.
- **모르는 값을 추측하지 않는다.** `packet.unknown_fields` 는 '미확인'으로
  명시하고, 비어있는 값도 '미확인'으로만 표기한다.
- **충돌을 숨기지 않는다(명세 20/43절).** 같은 필드에 서로 다른 값이 있으면
  임의로 하나를 확정하지 않고 값과 출처를 모두 보존해 표시한다.

reportlab 은 무거운 외부 패키지이므로 패키지 공통 규칙에 따라 모듈 최상단이
아닌 함수 내부에서 지연 import 한다(가이드 4절).
"""

from __future__ import annotations

import logging
import os
import re
import unicodedata
from datetime import datetime
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape as _xml_escape

from ..config import Settings, get_settings
from ..schemas import ConsultationPacket

logger = logging.getLogger(__name__)

__all__ = [
    "generate_consultation_pdf",
    "find_korean_font",
    "KOREAN_FONT_CANDIDATES",
    "UNKNOWN_LABEL",
]

#: 값이 없을 때 쓰는 고정 문구 — 추측 대신 항상 이 문자열을 쓴다.
UNKNOWN_LABEL = "미확인"

#: 한글 폰트 탐색 후보. Windows → Colab → 기타 리눅스 → macOS 순으로 본다.
#: (Colab 은 `apt-get install fonts-nanum` 시 nanum 경로에 설치된다.)
KOREAN_FONT_CANDIDATES: tuple[str, ...] = (
    # Windows
    "C:/Windows/Fonts/malgun.ttf",
    "C:/Windows/Fonts/malgunsl.ttf",
    "C:/Windows/Fonts/NanumGothic.ttf",
    "C:/Windows/Fonts/batang.ttc",
    "C:/Windows/Fonts/gulim.ttc",
    # Colab / Ubuntu (fonts-nanum)
    "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
    "/usr/share/fonts/truetype/nanum/NanumBarunGothic.ttf",
    "/usr/share/fonts/truetype/nanum/NanumMyeongjo.ttf",
    # 기타 리눅스 (noto CJK)
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJKkr-Regular.otf",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/unfonts-core/UnDotum.ttf",
    # macOS
    "/System/Library/Fonts/AppleSDGothicNeo.ttc",
    "/Library/Fonts/AppleGothic.ttf",
    "/System/Library/Fonts/Supplemental/AppleGothic.ttf",
)

#: 굵은 글꼴 후보 — 본문 폰트와 같은 계열이 있으면 제목에 쓴다.
_BOLD_FONT_CANDIDATES: dict[str, tuple[str, ...]] = {
    "malgun.ttf": ("C:/Windows/Fonts/malgunbd.ttf",),
    "NanumGothic.ttf": (
        "/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf",
        "C:/Windows/Fonts/NanumGothicBold.ttf",
    ),
    "NanumBarunGothic.ttf": ("/usr/share/fonts/truetype/nanum/NanumBarunGothicBold.ttf",),
}

_REGISTERED_FONT_NAME = "PetCareKR"
_REGISTERED_BOLD_NAME = "PetCareKR-Bold"

#: 필드명 → 한국어 라벨. 상류 모듈이 어떤 키를 쓰든 최대한 읽히게 만든다.
_LABELS: dict[str, str] = {
    # pet
    "name": "이름",
    "pet_name": "이름",
    "species": "종",
    "breed": "품종",
    "age": "나이",
    "age_years": "나이",
    "birth_date": "생년월일",
    "sex": "성별",
    "gender": "성별",
    "neutered": "중성화 여부",
    "weight": "몸무게",
    "weight_kg": "몸무게(kg)",
    "microchip": "동물등록번호",
    # medical history
    "conditions": "기존 질병",
    "existing_conditions": "기존 질병",
    "chronic_conditions": "기존 질병",
    "diseases": "기존 질병",
    "medications": "복용 중인 약",
    "current_medications": "복용 중인 약",
    "allergies": "알레르기",
    "surgeries": "수술 이력",
    "vaccinations": "예방접종",
    # current condition
    "main_symptoms": "주요 증상",
    "symptoms": "주요 증상",
    "chief_complaint": "주요 증상",
    "observation": "보호자 관찰 내용",
    "user_message": "보호자 진술",
    "onset": "증상 시작",
    "started_at": "증상 시작",
    "symptom_started_at": "증상 시작",
    "duration": "지속 기간",
    "frequency": "빈도",
    "change": "변화 추이",
    "progression": "변화 추이",
    "trend": "변화 추이",
    "severity": "정도",
    "note": "비고",
    "notes": "비고",
    # diagnoses / diary
    "date": "기록일",
    "recorded_at": "기록일",
    "diagnosed_at": "진단일",
    "entry_date": "기록일",
    "hospital": "병원",
    "hospital_name": "병원",
    "diagnosis": "진단명",
    "diagnosis_name": "진단명",
    "vet": "담당 수의사",
    "veterinarian": "담당 수의사",
    "treatment": "처치 내용",
    "prescription": "처방 기록",
    "content": "내용",
    "summary": "요약",
    "meal": "식사",
    "food": "식사",
    "appetite": "식욕",
    "water": "음수",
    "activity": "활동량",
    "stool": "배변",
    "urine": "배뇨",
    "vomit": "구토",
    "mood": "상태",
    # risk assessment
    "risk_level": "AI 참고 분류",
    "emergency_urgency": "응급 긴급도",
    "red_flags": "관찰된 주의 신호",
    "reasons": "분류 근거",
    "missing_information": "추가 확인 필요",
}

_RISK_LABELS: dict[str, str] = {
    "normal": "일상 관찰 권장 (normal)",
    "visit": "병원 진료 권장 (visit)",
    "emergency": "응급 대응 필요 (emergency)",
}

_URGENCY_LABELS: dict[str, str] = {
    "none": "해당 없음",
    "contact_ready": "병원 연락 준비 필요",
    "critical_immediate": "즉시 내원 필요",
}

_SPECIES_LABELS: dict[str, str] = {"dog": "강아지(dog)", "cat": "고양이(cat)"}

_DOC_TITLES: dict[str, str] = {
    "visit_consultation": "병원 상담 자료",
    "emergency_consultation": "응급 상담 자료",
}

#: 문서 하단 고정 안내 (9번 항목) — LLM 이 바꾸지 못하도록 상수로 둔다.
_FOOTER_NOTICE: tuple[str, ...] = (
    "본 문서는 보호자가 기록한 정보와 AI 참고 분류를 정리한 상담 보조 자료입니다.",
    "확정 진단·처방·치료 지시를 포함하지 않으며, 진단서로 사용할 수 없습니다.",
    "모든 내용은 수의사의 직접 진찰과 확인이 필요합니다.",
    f"'{UNKNOWN_LABEL}' 으로 표시된 항목은 정보가 없는 항목이며 추정하지 않았습니다. 진료 시 보호자에게 확인해 주세요.",
    "값이 충돌하는 항목은 어느 하나로 확정하지 않고 값과 출처를 모두 기록했습니다. 진료 시 확인이 필요합니다.",
)


# ---------------------------------------------------------------------------
# 폰트
# ---------------------------------------------------------------------------
def find_korean_font() -> str | None:
    """사용 가능한 한글 폰트 경로를 찾는다.

    Windows(맑은 고딕) → Colab/Ubuntu(나눔고딕) → 기타 리눅스(Noto CJK) →
    macOS 순으로 후보를 훑는다. 어디에도 없으면 ``None`` 을 돌려주고,
    호출자는 Helvetica 로 폴백한다(한글은 깨지지만 PDF 생성 자체는 성공시킨다).
    """
    for candidate in KOREAN_FONT_CANDIDATES:
        try:
            if os.path.isfile(candidate):
                return candidate
        except OSError:  # 권한 문제 등으로 stat 실패해도 탐색을 멈추지 않는다.
            continue
    return None


def _find_bold_font(regular_path: str) -> str | None:
    """본문 폰트와 같은 계열의 굵은 폰트를 찾는다(없으면 None)."""
    for key, candidates in _BOLD_FONT_CANDIDATES.items():
        if regular_path.replace("\\", "/").endswith(key):
            for candidate in candidates:
                if os.path.isfile(candidate):
                    return candidate
    return None


def _register_fonts() -> tuple[str, str, bool]:
    """reportlab 에 한글 폰트를 등록하고 (본문, 제목, 한글가능) 을 돌려준다.

    폰트를 찾지 못하면 Helvetica 로 폴백하되, 한글이 깨진다는 사실을
    경고 로그로 남긴다(조용히 실패하지 않는다).
    """
    from reportlab.pdfbase import pdfmetrics  # 지연 import
    from reportlab.pdfbase.ttfonts import TTFont

    font_path = find_korean_font()
    if font_path is None:
        logger.warning(
            "한글 폰트를 찾지 못해 Helvetica 로 폴백합니다 — PDF 안의 한글이 깨질 수 있습니다. "
            "Colab 에서는 `apt-get install -y fonts-nanum` 후 다시 실행하세요."
        )
        return "Helvetica", "Helvetica-Bold", False

    registered = set(pdfmetrics.getRegisteredFontNames())
    if _REGISTERED_FONT_NAME in registered:
        bold_name = (
            _REGISTERED_BOLD_NAME if _REGISTERED_BOLD_NAME in registered else _REGISTERED_FONT_NAME
        )
        return _REGISTERED_FONT_NAME, bold_name, True

    try:
        kwargs: dict[str, Any] = {}
        if font_path.lower().endswith(".ttc"):
            # .ttc 는 여러 폰트가 든 컬렉션이라 첫 번째 subfont 를 명시해야 한다.
            kwargs["subfontIndex"] = 0
        pdfmetrics.registerFont(TTFont(_REGISTERED_FONT_NAME, font_path, **kwargs))
    except Exception as exc:  # 폰트 파일이 손상된 경우까지 포함
        logger.warning(
            "한글 폰트 등록에 실패해 Helvetica 로 폴백합니다 (%s): %s — 한글이 깨질 수 있습니다.",
            font_path,
            exc,
        )
        return "Helvetica", "Helvetica-Bold", False

    bold_name = _REGISTERED_FONT_NAME
    bold_path = _find_bold_font(font_path)
    if bold_path:
        try:
            pdfmetrics.registerFont(TTFont(_REGISTERED_BOLD_NAME, bold_path))
            bold_name = _REGISTERED_BOLD_NAME
        except Exception:  # 굵은 글꼴은 없어도 문서 생성에 지장이 없다.
            bold_name = _REGISTERED_FONT_NAME

    # <b> 마크업이 Helvetica 로 대체되지 않도록 family 를 등록한다.
    pdfmetrics.registerFontFamily(
        _REGISTERED_FONT_NAME,
        normal=_REGISTERED_FONT_NAME,
        bold=bold_name,
        italic=_REGISTERED_FONT_NAME,
        boldItalic=bold_name,
    )
    logger.info("PDF 한글 폰트를 사용합니다: %s", font_path)
    return _REGISTERED_FONT_NAME, bold_name, True


# ---------------------------------------------------------------------------
# 텍스트 정규화 / escape
# ---------------------------------------------------------------------------
_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def _esc(value: Any) -> str:
    """reportlab Paragraph 용 안전 문자열로 변환한다.

    사용자·LLM 이 만든 텍스트에 ``<`` ``&`` 가 들어오면 Paragraph 가 마크업으로
    해석하다 예외를 던진다. 그래서 xml escape 는 선택이 아니라 필수다.
    줄바꿈은 escape 뒤에 ``<br/>`` 로 되살린다.
    """
    text = "" if value is None else str(value)
    text = _CONTROL_CHARS.sub("", text).replace("\r\n", "\n").replace("\r", "\n")
    return _xml_escape(text).replace("\n", "<br/>")


def _is_blank(value: Any) -> bool:
    """'값이 없음' 판정 — 빈 문자열/빈 컬렉션도 없음으로 본다."""
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() == ""
    if isinstance(value, (list, tuple, set, dict)):
        return len(value) == 0
    return False


def _fmt(value: Any, field: str = "") -> str:
    """값을 사람이 읽을 문자열로 만든다. 없으면 항상 '미확인'."""
    if _is_blank(value):
        return UNKNOWN_LABEL
    if isinstance(value, bool):
        return "예" if value else "아니오"
    if field in ("species", "pet_species") and isinstance(value, str):
        return _SPECIES_LABELS.get(value, value)
    if field == "risk_level" and isinstance(value, str):
        return _RISK_LABELS.get(value, value)
    if field == "emergency_urgency" and isinstance(value, str):
        return _URGENCY_LABELS.get(value, value)
    if isinstance(value, (list, tuple, set)):
        parts = [_fmt(item) for item in value if not _is_blank(item)]
        return ", ".join(parts) if parts else UNKNOWN_LABEL
    if isinstance(value, dict):
        parts = [f"{_label(k)}: {_fmt(v, k)}" for k, v in value.items() if not _is_blank(v)]
        return " / ".join(parts) if parts else UNKNOWN_LABEL
    return str(value).strip()


def _label(field: str) -> str:
    """필드명을 한국어 라벨로 바꾼다(모르는 키는 원문 유지)."""
    return _LABELS.get(field, field.replace("_", " "))


def _pick(source: dict[str, Any], *keys: str) -> Any:
    """여러 별칭 키 중 먼저 값이 있는 것을 고른다(상류 키 이름 변화 대비)."""
    for key in keys:
        if key in source and not _is_blank(source[key]):
            return source[key]
    return None


def _sanitize_filename_part(name: str) -> str:
    """파일명에 쓸 수 있게 정규화한다 — 한글은 살리고 위험 문자만 제거.

    Windows 예약 문자(``\\ / : * ? " < > |``), 제어문자, 앞뒤 점/공백을 없앤다.
    NFC 정규화로 자소 분리(맥에서 흔한 NFD)를 막고, 길이도 제한한다.
    """
    text = unicodedata.normalize("NFC", str(name or "")).strip()
    text = _CONTROL_CHARS.sub("", text)
    text = re.sub(r'[\\/:*?"<>|]', "", text)
    text = re.sub(r"\s+", "_", text)
    text = text.strip("._ ")
    if len(text) > 40:
        text = text[:40].strip("._ ")
    return text or "pet"


def _parse_generated_at(raw: str) -> datetime:
    """packet.generated_at 을 파싱한다. 실패하면 현재 시각을 쓴다."""
    text = (raw or "").strip()
    if text:
        candidate = text[:-1] + "+00:00" if text.endswith("Z") else text
        try:
            return datetime.fromisoformat(candidate)
        except ValueError:
            for pattern in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%Y/%m/%d %H:%M:%S"):
                try:
                    return datetime.strptime(text, pattern)
                except ValueError:
                    continue
    return datetime.now()


# ---------------------------------------------------------------------------
# provenance / 충돌 정리 (명세 20·43절)
# ---------------------------------------------------------------------------
def _normalize_provenance(packet: ConsultationPacket) -> list[dict[str, Any]]:
    """provenance 기록을 필드 단위로 묶어 충돌 여부까지 계산한다.

    상류(Clinical Context Priority Agent)는 두 가지 모양을 섞어 보낼 수 있다.

    - ``ContextProvenance``: ``{field, value, source, recorded_at}``
    - ``ContextConflict``:   ``{field, selected_value, selected_source, conflicting_values}``

    둘 다 받아 ``{field, entries[{value, source, recorded_at}], has_conflict,
    selected_value, selected_source}`` 형태로 통일한다. 같은 field 에 서로 다른
    값이 여러 번 기록된 경우도 충돌로 본다 — **하나로 확정하지 않는다.**
    """
    grouped: dict[str, dict[str, Any]] = {}
    order: list[str] = []

    for raw in packet.provenance or []:
        if not isinstance(raw, dict):
            continue
        field = str(raw.get("field") or "").strip() or "(필드 미상)"
        if field not in grouped:
            grouped[field] = {
                "field": field,
                "entries": [],
                "has_conflict": False,
                "selected_value": None,
                "selected_source": "",
            }
            order.append(field)
        bucket = grouped[field]

        conflicting = raw.get("conflicting_values") or []
        if "selected_value" in raw or conflicting:
            # ContextConflict 모양 — 선택값과 충돌값을 모두 보존한다.
            bucket["selected_value"] = raw.get("selected_value")
            bucket["selected_source"] = str(raw.get("selected_source") or "")
            bucket["entries"].append(
                {
                    "value": raw.get("selected_value"),
                    "source": str(raw.get("selected_source") or ""),
                    "recorded_at": raw.get("recorded_at"),
                    "selected": True,
                }
            )
            for item in conflicting:
                if not isinstance(item, dict):
                    bucket["entries"].append({"value": item, "source": "", "recorded_at": None})
                    continue
                bucket["entries"].append(
                    {
                        "value": item.get("value", item.get("selected_value")),
                        "source": str(item.get("source") or item.get("selected_source") or ""),
                        "recorded_at": item.get("recorded_at"),
                    }
                )
            if conflicting:
                bucket["has_conflict"] = True
        else:
            bucket["entries"].append(
                {
                    "value": raw.get("value"),
                    "source": str(raw.get("source") or ""),
                    "recorded_at": raw.get("recorded_at"),
                }
            )

    # 같은 필드에 서로 다른 값이 두 번 이상 기록됐으면 그것도 충돌이다.
    for bucket in grouped.values():
        distinct = {_fmt(entry.get("value"), bucket["field"]) for entry in bucket["entries"]}
        if len(distinct) > 1:
            bucket["has_conflict"] = True

    return [grouped[field] for field in order]


def _conflict_note(records: list[dict[str, Any]], *fields: str) -> str | None:
    """해당 필드에 충돌이 있으면 표 안에 붙일 한 줄 경고를 만든다."""
    wanted = {f.lower() for f in fields}
    for record in records:
        if not record["has_conflict"]:
            continue
        if record["field"].lower() not in wanted:
            continue
        pairs = [
            f"{_fmt(entry.get('value'), record['field'])}({entry.get('source') or '출처 미상'})"
            for entry in record["entries"]
        ]
        return "정보 충돌 — 확정하지 않음: " + " / ".join(pairs)
    return None


# ---------------------------------------------------------------------------
# 본문 조립
# ---------------------------------------------------------------------------
def _build_story(packet: ConsultationPacket, styles: dict[str, Any], table_style: Any) -> list[Any]:
    """명세 37절의 9개 항목을 순서 그대로 조립한다."""
    from reportlab.lib.units import mm
    from reportlab.platypus import KeepTogether, Paragraph, Spacer, Table

    story: list[Any] = []
    records = _normalize_provenance(packet)
    used_keys: dict[str, set[str]] = {}

    def heading(text: str) -> None:
        story.append(Spacer(1, 5 * mm))
        story.append(Paragraph(_esc(text), styles["h2"]))

    def note(text: str) -> None:
        story.append(Paragraph(_esc(text), styles["note"]))

    def rows_table(rows: list[tuple[str, str]]) -> None:
        if not rows:
            rows = [("내용", UNKNOWN_LABEL)]
        data = [
            [Paragraph(_esc(label), styles["th"]), Paragraph(_esc(value), styles["td"])]
            for label, value in rows
        ]
        table = Table(data, colWidths=[42 * mm, 123 * mm], hAlign="LEFT")
        table.setStyle(table_style)
        story.append(table)

    def field_rows(
        source: dict[str, Any],
        specs: list[tuple[str, tuple[str, ...]]],
        bucket: str,
    ) -> list[tuple[str, str]]:
        """(라벨, 별칭키들) 목록으로 표 행을 만들고 충돌 주석을 덧붙인다."""
        seen = used_keys.setdefault(bucket, set())
        rows: list[tuple[str, str]] = []
        for label, keys in specs:
            value = _pick(source, *keys)
            seen.update(keys)
            text = _fmt(value, keys[0])
            conflict = _conflict_note(records, *keys)
            if conflict:
                text = f"{text}\n{conflict}"
            rows.append((label, text))
        return rows

    def leftover_rows(source: dict[str, Any], bucket: str) -> list[tuple[str, str]]:
        """알려진 키 외에 남은 값도 버리지 않고 그대로 보여준다."""
        seen = used_keys.setdefault(bucket, set())
        rows: list[tuple[str, str]] = []
        for key, value in source.items():
            if key in seen or _is_blank(value):
                continue
            text = _fmt(value, key)
            conflict = _conflict_note(records, key)
            if conflict:
                text = f"{text}\n{conflict}"
            rows.append((_label(key), text))
        return rows

    def list_section(items: list[dict[str, Any]], empty_text: str) -> None:
        if not items:
            note(empty_text)
            return
        for index, item in enumerate(items, start=1):
            if not isinstance(item, dict):
                story.append(Paragraph(f"{index}) {_esc(_fmt(item))}", styles["body"]))
                continue
            rows = [
                (_label(key), _fmt(value, key))
                for key, value in item.items()
                if not _is_blank(value)
            ]
            block: list[Any] = [Paragraph(_esc(f"기록 {index}"), styles["h3"])]
            data = [
                [Paragraph(_esc(label), styles["th"]), Paragraph(_esc(value), styles["td"])]
                for label, value in (rows or [("내용", UNKNOWN_LABEL)])
            ]
            table = Table(data, colWidths=[42 * mm, 123 * mm], hAlign="LEFT")
            table.setStyle(table_style)
            block.append(table)
            block.append(Spacer(1, 2 * mm))
            story.append(KeepTogether(block))

    # ---- 표지 -------------------------------------------------------------
    pet_name = _fmt(_pick(packet.pet, "name", "pet_name"), "name")
    doc_title = _DOC_TITLES.get(packet.document_type, "상담 자료")
    story.append(Paragraph(_esc(f"반려동물 {doc_title}"), styles["h1"]))
    story.append(
        Paragraph(
            _esc(f"작성 시각: {packet.generated_at or UNKNOWN_LABEL}   |   대상: {pet_name}"),
            styles["meta"],
        )
    )
    note("이 문서는 AI 상담 보조 자료입니다. 확정 진단이 아니며 수의사의 확인이 필요합니다.")

    # ---- 1. 반려동물 기본정보 ---------------------------------------------
    heading("1. 반려동물 기본정보")
    rows = field_rows(
        packet.pet,
        [
            ("이름", ("name", "pet_name")),
            ("종", ("species", "pet_species")),
            ("품종", ("breed",)),
            ("나이", ("age", "age_years", "birth_date")),
            ("성별", ("sex", "gender")),
            ("중성화 여부", ("neutered",)),
            ("몸무게", ("weight_kg", "weight")),
        ],
        "pet",
    )
    rows += leftover_rows(packet.pet, "pet")
    rows_table(rows)

    # ---- 2. 현재 주요 증상 -------------------------------------------------
    heading("2. 현재 주요 증상")
    rows = field_rows(
        packet.current_condition,
        [
            ("주요 증상", ("main_symptoms", "symptoms", "chief_complaint")),
            ("보호자 관찰 내용", ("observation", "user_message", "description")),
        ],
        "condition",
    )
    rows_table(rows)

    # ---- 3. 증상 시작·빈도·변화 -------------------------------------------
    heading("3. 증상 시작 · 빈도 · 변화")
    rows = field_rows(
        packet.current_condition,
        [
            ("증상 시작", ("onset", "started_at", "symptom_started_at", "since")),
            ("지속 기간", ("duration",)),
            ("빈도", ("frequency",)),
            ("변화 추이", ("change", "progression", "trend")),
            ("정도", ("severity",)),
        ],
        "condition",
    )
    rows += leftover_rows(packet.current_condition, "condition")
    rows_table(rows)

    # ---- 4. 기존 질병·복용약·알레르기 --------------------------------------
    heading("4. 기존 질병 · 복용약 · 알레르기")
    rows = field_rows(
        packet.medical_history,
        [
            ("기존 질병", ("existing_conditions", "conditions", "chronic_conditions", "diseases")),
            ("복용 중인 약", ("medications", "current_medications")),
            ("알레르기", ("allergies",)),
        ],
        "history",
    )
    rows += leftover_rows(packet.medical_history, "history")
    rows_table(rows)

    # ---- 5. 관련 진단서 기록 ----------------------------------------------
    heading("5. 관련 진단서 기록")
    list_section(packet.related_diagnoses, f"관련 진단서 기록 {UNKNOWN_LABEL} (제공된 기록 없음)")

    # ---- 6. 최근 일기장 기록 ----------------------------------------------
    heading("6. 최근 일기장 기록")
    list_section(
        packet.supporting_daily_entries, f"최근 일기장 기록 {UNKNOWN_LABEL} (제공된 기록 없음)"
    )

    # ---- 7. AI 위험도 분류와 관찰 근거 -------------------------------------
    heading("7. AI 위험도 분류와 관찰 근거")
    note("아래 분류는 AI 참고 분류이며 확정 진단이 아닙니다. 최종 판단은 수의사가 합니다.")
    risk = packet.risk_assessment or {}
    rows = field_rows(
        risk,
        [
            ("AI 참고 분류", ("risk_level",)),
            ("응급 긴급도", ("emergency_urgency",)),
            ("관찰된 주의 신호", ("red_flags",)),
            ("분류 근거(관찰 사실)", ("reasons",)),
            ("추가 확인 필요", ("missing_information",)),
        ],
        "risk",
    )
    rows += leftover_rows(risk, "risk")
    rows_table(rows)

    # ---- 8. 미확인 정보 ----------------------------------------------------
    heading("8. 미확인 정보")
    if packet.unknown_fields:
        note("아래 항목은 정보가 없어 추정하지 않았습니다. 진료 시 보호자 확인이 필요합니다.")
        for field in packet.unknown_fields:
            story.append(
                Paragraph(f"- {_esc(_label(str(field)))}: {UNKNOWN_LABEL}", styles["body"])
            )
    else:
        note(f"별도로 표시된 {UNKNOWN_LABEL} 항목이 없습니다.")

    story.append(Spacer(1, 3 * mm))
    story.append(Paragraph(_esc("8-1. 정보 출처 및 값 충돌 기록"), styles["h3"]))
    if records:
        conflicted = [r for r in records if r["has_conflict"]]
        if conflicted:
            note(
                "아래 항목은 출처마다 값이 달라 하나로 확정하지 않았습니다. "
                "값과 출처를 모두 그대로 싣습니다."
            )
        prov_rows: list[tuple[str, str]] = []
        for record in records:
            lines = []
            for entry in record["entries"]:
                source = entry.get("source") or "출처 미상"
                recorded = entry.get("recorded_at")
                stamp = f", 기록 {recorded}" if recorded else ""
                marker = " [우선 선택]" if entry.get("selected") else ""
                lines.append(
                    f"- {_fmt(entry.get('value'), record['field'])} (출처: {source}{stamp}){marker}"
                )
            prefix = "[충돌] " if record["has_conflict"] else ""
            prov_rows.append((f"{prefix}{_label(record['field'])}", "\n".join(lines)))
        rows_table(prov_rows)
    else:
        note("기록된 출처 정보가 없습니다.")

    # ---- 9. 의료진 확인 필요 안내 (고정 문구) -------------------------------
    heading("9. 의료진 확인 필요 안내")
    for line in _FOOTER_NOTICE:
        story.append(Paragraph(f"- {_esc(line)}", styles["body"]))

    return story


def _build_styles(base_font: str, bold_font: str) -> tuple[dict[str, Any], Any]:
    """문서 스타일과 표 스타일을 만든다(폰트는 등록된 것을 그대로 쓴다)."""
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_LEFT
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.platypus import TableStyle

    accent = colors.HexColor("#1F4E79")
    styles = {
        "h1": ParagraphStyle(
            "h1", fontName=bold_font, fontSize=17, leading=22, spaceAfter=4, textColor=accent
        ),
        "h2": ParagraphStyle(
            "h2", fontName=bold_font, fontSize=12.5, leading=17, spaceAfter=3, textColor=accent
        ),
        "h3": ParagraphStyle(
            "h3", fontName=bold_font, fontSize=10.5, leading=14, spaceBefore=2, spaceAfter=2
        ),
        "meta": ParagraphStyle(
            "meta",
            fontName=base_font,
            fontSize=9,
            leading=13,
            textColor=colors.HexColor("#555555"),
            spaceAfter=2,
        ),
        "note": ParagraphStyle(
            "note",
            fontName=base_font,
            fontSize=8.8,
            leading=12.5,
            textColor=colors.HexColor("#8A3B00"),
            spaceAfter=3,
        ),
        "body": ParagraphStyle(
            "body", fontName=base_font, fontSize=9.5, leading=13.5, alignment=TA_LEFT, spaceAfter=1
        ),
        "th": ParagraphStyle("th", fontName=bold_font, fontSize=9.2, leading=13),
        "td": ParagraphStyle("td", fontName=base_font, fontSize=9.2, leading=13),
    }
    table_style = TableStyle(
        [
            ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#BBBBBB")),
            ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#EEF3F8")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 2 * mm),
            ("RIGHTPADDING", (0, 0), (-1, -1), 2 * mm),
            ("TOPPADDING", (0, 0), (-1, -1), 1.4 * mm),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 1.4 * mm),
        ]
    )
    return styles, table_style


# ---------------------------------------------------------------------------
# 공개 API
# ---------------------------------------------------------------------------
def generate_consultation_pdf(
    packet: ConsultationPacket,
    output_dir: str | Path | None = None,
    settings: Settings | None = None,
) -> tuple[str, str]:
    """상담 packet 을 고정 template PDF 로 만들고 ``(pdf_path, filename)`` 을 돌려준다.

    파일명은 ``consultation_<반려동물이름>_<YYYYmmdd_HHMMSS>.pdf`` 형태이며,
    이름은 파일시스템 안전 문자로 정규화한다(한글은 유지).

    생성 뒤 파일 존재와 크기(>0)를 검증한다. 검증에 실패하면 조용히 넘어가지
    않고 ``RuntimeError`` 를 던진다 — 빈 PDF 가 이메일 초안에 첨부되는 것을
    막기 위해서다.

    Raises:
        TypeError: packet 이 ConsultationPacket 이 아닐 때.
        RuntimeError: reportlab 미설치, 파일 생성 실패, 크기 0 인 경우.
    """
    if not isinstance(packet, ConsultationPacket):
        raise TypeError("generate_consultation_pdf 는 ConsultationPacket 을 받아야 합니다.")

    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.units import mm
        from reportlab.platypus import SimpleDocTemplate
    except ImportError as exc:  # 지연 import — 여기서만 실패한다.
        raise RuntimeError(
            "reportlab 이 설치돼 있지 않아 PDF 를 만들 수 없습니다. `pip install reportlab` 후 다시 실행하세요."
        ) from exc

    settings = settings or get_settings()
    target_dir = Path(output_dir) if output_dir is not None else Path(settings.output_dir)
    try:
        target_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise RuntimeError(f"PDF 출력 디렉터리를 만들 수 없습니다: {target_dir} ({exc})") from exc

    pet_name = _pick(packet.pet, "name", "pet_name") or "pet"
    stamp = _parse_generated_at(packet.generated_at).strftime("%Y%m%d_%H%M%S")
    base_name = f"consultation_{_sanitize_filename_part(str(pet_name))}_{stamp}"
    filename = f"{base_name}.pdf"
    pdf_path = target_dir / filename
    suffix = 1
    while pdf_path.exists():  # 같은 초에 두 번 생성돼도 덮어쓰지 않는다.
        filename = f"{base_name}_{suffix}.pdf"
        pdf_path = target_dir / filename
        suffix += 1

    base_font, bold_font, korean_ok = _register_fonts()
    styles, table_style = _build_styles(base_font, bold_font)

    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=A4,
        leftMargin=20 * mm,
        rightMargin=20 * mm,
        topMargin=18 * mm,
        bottomMargin=18 * mm,
        title=f"반려동물 상담 자료 - {_fmt(pet_name, 'name')}",
        author="PetCare AI",
        subject="AI 상담 보조 자료 (확정 진단 아님)",
    )

    def _page_footer(canvas: Any, doc_obj: Any) -> None:
        """모든 페이지 하단에 '확정 진단 아님' 고지와 쪽번호를 찍는다."""
        canvas.saveState()
        canvas.setFont(base_font, 7.5)
        canvas.setFillGray(0.35)
        canvas.drawString(
            20 * mm, 10 * mm, "AI 상담 보조 자료 — 확정 진단이 아니며 수의사 확인이 필요합니다."
        )
        canvas.drawRightString(A4[0] - 20 * mm, 10 * mm, f"{doc_obj.page} 쪽")
        canvas.restoreState()

    story = _build_story(packet, styles, table_style)
    try:
        doc.build(story, onFirstPage=_page_footer, onLaterPages=_page_footer)
    except Exception as exc:
        raise RuntimeError(f"PDF 생성 중 오류가 발생했습니다: {exc}") from exc

    if not pdf_path.exists():
        raise RuntimeError(f"PDF 파일이 생성되지 않았습니다: {pdf_path}")
    size = pdf_path.stat().st_size
    if size <= 0:
        raise RuntimeError(f"생성된 PDF 크기가 0 입니다: {pdf_path}")

    if not korean_ok:
        logger.warning("한글 폰트 없이 생성된 PDF 입니다 — 한글이 깨져 보일 수 있습니다: %s", pdf_path)
    logger.info("상담 PDF 생성 완료: %s (%d bytes)", pdf_path, size)
    return str(pdf_path), filename
