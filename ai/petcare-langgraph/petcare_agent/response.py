from __future__ import annotations

import re
from typing import Any


_EMOJI_PATTERN = re.compile(
    "["
    "\U0001F1E6-\U0001F1FF"
    "\U0001F300-\U0001FAFF"
    "\U00002600-\U000027BF"
    "\U00002300-\U000023FF"
    "]+",
    flags=re.UNICODE,
)

_VARIATION_PATTERN = re.compile(
    "[\u200d\ufe0e\ufe0f]"
)


def clean_agent_response(
    text: str,
) -> str:
    cleaned = _EMOJI_PATTERN.sub(
        "",
        text,
    )
    cleaned = _VARIATION_PATTERN.sub(
        "",
        cleaned,
    )
    cleaned = "\n".join(
        line.rstrip()
        for line in cleaned.splitlines()
    )
    cleaned = re.sub(
        r"\n{3,}",
        "\n\n",
        cleaned,
    )
    return cleaned.strip()


def pet_name_from_state(
    state: dict[str, Any],
) -> str:
    return str(
        state.get(
            "backend_context",
            {},
        )
        .get("pet", {})
        .get("name")
        or "반려동물"
    )


def build_emergency_response(
    state: dict[str, Any],
) -> str:
    reasons = [
        str(
            item.get(
                "message",
                "",
            )
        ).strip()
        for item in state.get(
            "emergency_hits",
            [],
        )
        if str(
            item.get(
                "message",
                "",
            )
        ).strip()
    ]

    reason_text = "\n".join(
        f"- {reason}"
        for reason in reasons
    )

    return clean_agent_response(
        f"""
현재 상태는 추가 문진보다 즉시 진료가 우선하는 고위험 상황입니다.

확인된 고위험 신호
{reason_text or "- 고위험 증상 확인"}

가까운 운영 중 동물병원을 확인하고 병원 전달용 이메일을 준비합니다.
        """
    )


def build_visit_question(
    assessment_text: str,
) -> str:
    return clean_agent_response(
        f"""
{assessment_text}

병원에 방문하시겠습니까?
방문할 예정이면 '예', 방문하지 않을 예정이면 '아니오'라고 답해 주세요.
        """
    )


def build_no_visit_response() -> str:
    return (
        "병원 방문을 선택하지 않아 "
        "이번 상태 확인을 종료합니다. "
        "새 증상이나 상태 악화가 생기면 "
        "새로운 상태 확인을 시작해 주세요."
    )


def build_pdf_complete_response(
    artifact_path: str,
) -> str:
    return clean_agent_response(
        f"""
병원 전달용 PDF를 생성했습니다.

파일
- {artifact_path}

문서는 줄글이 아니라 반려동물 정보, 현재 증상, 발생 경과, 최근 기록, 진단 및 복용 기록, 미확인 항목으로 구분되어 있습니다.
        """
    )


def build_emergency_complete_response(
    state: dict[str, Any],
) -> str:
    base = build_emergency_response(
        state
    )
    hospital = state.get(
        "selected_hospital",
        {},
    )
    delivery = state.get(
        "email_delivery",
        {},
    )

    hospital_text = "\n".join(
        [
            (
                f"- 병원: "
                f"{hospital.get('name', '미확인')}"
            ),
            (
                f"- 주소: "
                f"{hospital.get('address', '미확인')}"
            ),
            (
                f"- 전화: "
                f"{hospital.get('phone') or '미확인'}"
            ),
            (
                f"- 운영 상태: "
                f"{hospital.get('open_status', '미확인')}"
            ),
        ]
    )

    status = delivery.get(
        "status"
    )

    if status == "sent":
        email_text = (
            "병원 전달용 이메일을 "
            "전송했습니다."
        )
    elif status == "saved":
        email_text = (
            "SMTP 설정이 없어 이메일을 "
            "로컬 발송함에 저장했습니다."
        )
    else:
        email_text = (
            "이메일 전송에 실패했습니다. "
            "병원에 전화로 현재 상태를 "
            "먼저 전달하세요."
        )

    file_path = delivery.get(
        "file_path"
    )
    preview_path = delivery.get(
        "preview_path"
    )

    if file_path:
        email_text += (
            f"\n- EML 파일: {file_path}"
        )

    if preview_path:
        email_text += (
            f"\n- 내용 미리보기: {preview_path}"
        )

    return clean_agent_response(
        f"""
{base}

확인된 병원
{hospital_text}

이메일 처리
{email_text}
        """
    )
