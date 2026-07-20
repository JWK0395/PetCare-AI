"""Email Draft Agent (명세 38절) — 초안만 만든다. **절대 보내지 않는다.**

이 node 는 Android 가 Gmail compose 화면을 열 때 채워 넣을 값만 반환한다. 전송
API 호출도, SMTP 연결도 없다. 이 파일에 네트워크 코드가 한 줄도 없다는 사실이
"실제로 보내지 않는다" 는 명세 요구를 지키는 가장 확실한 방법이다.

## 본문은 고정 template 이다

명세 38절: "본문은 짧은 고정 template 을 사용한다." LLM 자유 생성을 하지 않는 이유는
분량 문제가 아니라 **안전** 이다. 수의사에게 보내는 메일에 AI 가 만든 문장이 섞이면,
그 문장이 확정 진단처럼 읽히거나 없는 증상을 언급할 수 있다. 여기서는 State 에 이미
있는 값(반려동물 이름, AI 참고 분류)만 정해진 자리에 끼운다.

## 첨부 경로

`attachment_path` 는 반드시 `state["pdf_path"]` 와 **같아야 한다**(명세 40절 8번
검사). 그래서 경로를 새로 만들지 않고 State 값을 그대로 읽는다. PDF 가 없으면 초안
자체를 만들지 않는다 — 첨부 없는 '상담자료 첨부' 메일은 받는 쪽을 혼란스럽게 한다.
"""

from __future__ import annotations

import logging
import os
from datetime import date, datetime
from typing import Any

from ...schemas import EmailDraft

logger = logging.getLogger(__name__)

__all__ = [
    "SUBJECT_PREFIX",
    "BODY_TEMPLATE",
    "RISK_LABELS",
    "resolve_recipient_email",
    "resolve_pet_name",
    "build_subject",
    "build_body",
    "build_email_draft",
    "email_draft_node",
]


#: 문서 종류별 제목 접두어. 명세 38절 예시: `[응급 상담자료] 초코 / 2026-07-19`
SUBJECT_PREFIX: dict[str, str] = {
    "emergency_consultation": "[응급 상담자료]",
    "visit_consultation": "[병원 상담자료]",
}

#: 위험도 표기 — 항상 'AI 참고 분류' 라는 사실을 함께 적는다(확정 진단 금지).
RISK_LABELS: dict[str, str] = {
    "normal": "일상 관찰 권장 (AI 참고 분류)",
    "visit": "병원 진료 권장 (AI 참고 분류)",
    "emergency": "응급 대응 필요 (AI 참고 분류)",
}

#: 고정 본문 template. `{}` 자리에는 State 에 이미 존재하는 값만 들어간다.
#: 증상 서술·소견·권고 문장을 여기에 추가하지 않는다.
BODY_TEMPLATE = """안녕하세요. 반려동물 보호자입니다.

진료 상담을 위해 정리한 자료를 첨부드립니다.

- 반려동물: {pet_name}
- 작성일: {generated_date}
- AI 참고 분류: {risk_label}

첨부 파일: {attachment_filename}

첨부 자료는 보호자가 기록한 정보와 AI 참고 분류를 정리한 상담 보조 자료이며,
확정 진단·처방 내용을 포함하지 않습니다. 확인이 필요한 항목은 '미확인' 으로
표시해 두었습니다.

방문 전에 전화로 현재 진료 및 응급 접수 가능 여부를 확인하겠습니다.
감사합니다."""


def _first_nonblank(*values: Any) -> str:
    """앞에서부터 비어 있지 않은 첫 문자열을 고른다."""
    for value in values:
        text = str(value or "").strip()
        if text:
            return text
    return ""


def resolve_recipient_email(state: dict[str, Any]) -> str | None:
    """받는 사람 주소를 찾는다. 없으면 `None`(명세 38절).

    주소를 지어내지 않는다. 검색 결과에 이메일이 없으면 `to=None` 으로 두고,
    Android 가 사용자에게 직접 입력하도록 맡긴다. 대표 주소를 추측해서 넣으면
    엉뚱한 곳으로 반려동물 의료정보가 나간다.

    우선순위: 사용자가 선택한 병원 → 적합도 상위 병원 중 이메일이 있는 첫 후보.
    """
    selected = state.get("selected_hospital")
    if isinstance(selected, dict):
        hospital = selected.get("hospital") if isinstance(selected.get("hospital"), dict) else selected
        email = str((hospital or {}).get("email") or "").strip()
        if email:
            return email

    for item in state.get("hospital_results") or []:
        if not isinstance(item, dict):
            continue
        hospital = item.get("hospital") if isinstance(item.get("hospital"), dict) else item
        email = str((hospital or {}).get("email") or "").strip()
        if email:
            return email

    logger.info("병원 이메일을 찾지 못해 to=None 으로 둡니다(사용자가 직접 입력).")
    return None


def resolve_pet_name(state: dict[str, Any]) -> str:
    """반려동물 이름을 찾는다. 없으면 '반려동물'(추측하지 않는 중립 표현)."""
    packet = state.get("consultation_packet") or {}
    pet = packet.get("pet") if isinstance(packet, dict) else {}
    profile = state.get("priority_pet_context") or state.get("pet_profile") or {}
    return _first_nonblank(
        (pet or {}).get("name"),
        (profile or {}).get("name"),
    ) or "반려동물"


def _resolve_date(state: dict[str, Any]) -> str:
    """제목·본문에 쓸 날짜(YYYY-MM-DD).

    packet 의 `generated_at` 을 우선 쓴다 — PDF 파일명에 박힌 시각과 메일 날짜가
    어긋나면 나중에 어떤 파일 이야기인지 알 수 없게 된다. 파싱에 실패하면 오늘.
    """
    packet = state.get("consultation_packet") or {}
    raw = str((packet or {}).get("generated_at") or "").strip()
    if raw:
        candidate = raw[:-1] + "+00:00" if raw.endswith("Z") else raw
        try:
            return datetime.fromisoformat(candidate).date().isoformat()
        except ValueError:
            if len(raw) >= 10:
                try:
                    return date.fromisoformat(raw[:10]).isoformat()
                except ValueError:
                    pass
    return date.today().isoformat()


def build_subject(pet_name: str, generated_date: str, document_type: str) -> str:
    """제목을 만든다 — 명세 38절 예시 형식 그대로.

    `[응급 상담자료] 초코 / 2026-07-19`
    """
    prefix = SUBJECT_PREFIX.get(document_type, SUBJECT_PREFIX["visit_consultation"])
    return f"{prefix} {pet_name} / {generated_date}"


def build_body(
    pet_name: str,
    generated_date: str,
    risk_level: str,
    attachment_filename: str,
) -> str:
    """고정 template 에 값만 끼운다. **문장을 생성하지 않는다.**"""
    return BODY_TEMPLATE.format(
        pet_name=pet_name,
        generated_date=generated_date,
        risk_label=RISK_LABELS.get(risk_level, RISK_LABELS["normal"]),
        attachment_filename=attachment_filename,
    )


def build_email_draft(state: dict[str, Any]) -> EmailDraft | None:
    """State 로 이메일 초안을 만든다. PDF 가 없으면 `None`.

    `attachment_path` 는 State 의 `pdf_path` 를 **그대로** 쓴다. 여기서 경로를
    가공하면(정규화·절대경로 변환 등) Output Check 의 경로 일치 검사에 걸린다.
    """
    pdf_path = str(state.get("pdf_path") or "").strip()
    if not pdf_path:
        logger.info("생성된 PDF 가 없어 이메일 초안을 만들지 않습니다.")
        return None

    filename = _first_nonblank(state.get("pdf_filename"), os.path.basename(pdf_path))
    pet_name = resolve_pet_name(state)
    generated_date = _resolve_date(state)
    document_type = str(state.get("document_type") or "visit_consultation")
    risk_level = str(state.get("final_risk") or "normal")

    draft = EmailDraft(
        to=resolve_recipient_email(state),
        subject=build_subject(pet_name, generated_date, document_type),
        body=build_body(pet_name, generated_date, risk_level, filename),
        attachment_path=pdf_path,
        attachment_filename=filename,
    )
    logger.info(
        "이메일 초안 생성(전송하지 않음) — 수신자=%s, 첨부=%s",
        draft.to or "미지정",
        draft.attachment_filename,
    )
    return draft


def email_draft_node(state: dict) -> dict:
    """Email Draft Agent node (명세 38절).

    `OPEN_GMAIL_COMPOSE` action 을 함께 남긴다. 이름 그대로 **compose 화면을 여는**
    action 이며, 전송 action 이 아니다. 실제 전송은 사용자가 Gmail 앱에서 직접 한다.
    """
    draft = build_email_draft(state)
    if draft is None:
        return {"email_draft": None}

    return {
        "email_draft": draft.model_dump(),
        "ui_actions": [
            {
                "type": "OPEN_GMAIL_COMPOSE",
                "to": draft.to,
                "subject": draft.subject,
                "attachment_path": draft.attachment_path,
                "attachment_filename": draft.attachment_filename,
                # 전송은 사용자의 몫이라는 사실을 action payload 에도 남긴다.
                "auto_send": False,
            }
        ],
    }
