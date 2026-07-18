from __future__ import annotations

import json
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


def clean_agent_response(text: str) -> str:
    cleaned = _EMOJI_PATTERN.sub("", text)
    cleaned = _VARIATION_PATTERN.sub("", cleaned)

    lines = [
        re.sub(r"[ \t]+$", "", line)
        for line in cleaned.splitlines()
    ]

    normalized = "\n".join(lines).strip()
    normalized = re.sub(
        r"\n{3,}",
        "\n\n",
        normalized,
    )
    normalized = re.sub(
        r" +([,.;:!?])",
        r"\1",
        normalized,
    )

    return normalized


def json_text(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
    )


def pet_name_from_state(
    state: dict[str, Any],
) -> str:
    pet = state.get(
        "backend_context",
        {},
    ).get("pet", {})

    return str(
        pet.get("name")
        or "반려동물"
    )


def format_evidence(
    chunks: list[dict[str, Any]],
) -> str:
    if not chunks:
        return "검색된 근거 없음"

    entries: list[str] = []

    for index, chunk in enumerate(
        chunks,
        start=1,
    ):
        page = chunk.get("page")
        page_text = (
            f"p.{page}"
            if page is not None
            else "페이지 정보 없음"
        )

        entries.append(
            "\n".join(
                [
                    (
                        f"[{index}] "
                        f"{chunk.get('organization')} | "
                        f"{chunk.get('title')} | "
                        f"v{chunk.get('version')} | "
                        f"{page_text}"
                    ),
                    str(chunk.get("text", "")),
                ]
            )
        )

    return "\n\n".join(entries)


def build_emergency_response(
    state: dict[str, Any],
) -> str:
    reasons = [
        str(item.get("message", "")).strip()
        for item in state.get(
            "emergency_hits",
            [],
        )
        if str(item.get("message", "")).strip()
    ]

    reason_lines = (
        "\n".join(
            f"- {reason}"
            for reason in reasons
        )
        if reasons
        else "- 즉시 진료가 필요한 고위험 표현이 확인됨"
    )

    pet_name = pet_name_from_state(state)

    return clean_agent_response(
        f"""
현재 상태는 추가 문진보다 즉시 진료가 우선하는 고위험 상황입니다.

확인된 고위험 신호
{reason_lines}

지금 해야 할 일
- 가까운 동물병원이나 응급 진료가 가능한 병원에 즉시 연락하세요.
- 이동 중에는 {pet_name}을 최대한 안정시키고 불필요한 움직임을 줄이세요.
- 병원에는 현재 증상, 시작 시점, 발생 횟수와 최근 변화를 그대로 전달하세요.

이 안내는 진단이 아니라 응급 가능성을 놓치지 않기 위한 행동 안내입니다.
        """
    )


def closed_triage_message() -> str:
    return (
        "이번 상태 확인은 이미 종료되었습니다. "
        "앞서 안내한 판단은 그대로 유지되며, "
        "동일한 문진은 반복하지 않겠습니다. "
        "새 증상이나 악화가 생기면 새로운 상태 확인을 시작해 주세요."
    )


def handoff_confirmation() -> str:
    return (
        "병원 전달용 요약 초안을 생성했습니다. "
        "날짜와 증상 내용이 정확한지 확인한 뒤 사용해 주세요."
    )
