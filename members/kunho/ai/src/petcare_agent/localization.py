"""Small locale helpers for user-facing PetCare-AI text."""

from __future__ import annotations

import re


FIELD_LABELS_KO = {
    "appetite": "식욕",
    "water": "음수량",
    "activity": "활동량",
    "stool": "대변",
    "vomit": "구토",
}

SYMPTOM_LABELS_KO = {
    "abdominal_pain": "복부 통증",
    "blood_in_urine": "혈뇨",
    "blood_in_vomit": "토사물의 혈액",
    "bloody_diarrhea": "혈변 또는 검은 변",
    "breathing": "호흡 이상",
    "breathing_issue": "호흡 이상",
    "collapse_or_fainting": "쓰러짐 또는 실신",
    "cough": "기침",
    "coughing": "기침",
    "dehydration_signs": "탈수 의심 신호",
    "diarrhea": "설사",
    "gum_color_abnormal": "잇몸 또는 혀 색 이상",
    "labored_breathing": "힘들어 보이는 호흡",
    "lethargy": "무기력",
    "open mouth breathing": "입을 벌리고 호흡",
    "open_mouth_breathing": "입을 벌리고 호흡",
    "pain": "통증",
    "rapid_breathing": "빠른 호흡",
    "repeated_coughing": "반복적인 기침",
    "repeated_diarrhea": "반복적인 설사",
    "repeated_vomiting": "반복적인 구토",
    "seizure": "경련",
    "suspected_toxin": "독성 물질 섭취 의심",
    "toxicity": "중독 의심",
    "toxin_exposure": "독성 물질 노출",
    "urinary": "배뇨 이상",
    "urinary_issue": "배뇨 이상",
    "vomiting": "구토",
}

VALUE_LABELS_KO = {
    "abnormal": "이상",
    "decreased": "감소",
    "increased": "증가",
    "normal": "평소와 비슷함",
    "none": "없음",
    "unknown": "확인되지 않음",
}


def wants_korean(locale: str | None) -> bool:
    """Return true when user-facing text should be Korean."""

    return (locale or "").lower().startswith("ko")


def display_label(value: str, locale: str | None) -> str:
    """Return a compact label for a structured symptom, field, or value."""

    normalized = " ".join(str(value).strip().split())
    if not normalized:
        return ""
    if not wants_korean(locale):
        return normalized

    key = normalized.lower()
    return (
        SYMPTOM_LABELS_KO.get(key)
        or FIELD_LABELS_KO.get(key)
        or VALUE_LABELS_KO.get(key)
        or normalized
    )


def display_list(values: list[str], locale: str | None) -> list[str]:
    """Normalize, deduplicate, and localize a list for display."""

    display_values: list[str] = []
    seen: set[str] = set()
    for value in values:
        label = display_label(value, locale)
        if not label or label in seen:
            continue
        display_values.append(label)
        seen.add(label)
    return display_values


def localize_change_summary(summary: str, locale: str | None) -> str:
    """Translate deterministic change-detector summaries for Korean output."""

    text = summary.strip()
    if not text or not wants_korean(locale):
        return text
    if _contains_korean(text):
        return text
    if text == "Baseline is unavailable, so current status was not compared.":
        return "최근 기준 데이터가 없어 현재 상태와 비교하지 못했습니다."
    if text == "No baseline deviation detected.":
        return "최근 기준과 비교해 뚜렷한 변화는 확인되지 않았습니다."

    translated_parts: list[str] = []
    patterns = [
        (r"New symptoms reported: ([^.]+)\.", "새로 보고된 증상: {}."),
        (r"Worsened compared with baseline: ([^.]+)\.", "최근 기준보다 나빠진 항목: {}."),
        (r"Improved compared with baseline: ([^.]+)\.", "최근 기준보다 좋아진 항목: {}."),
        (r"Unchanged fields: ([^.]+)\.", "최근 기준과 비슷한 항목: {}."),
    ]
    for pattern, template in patterns:
        match = re.search(pattern, text)
        if not match:
            continue
        labels = display_list(_split_csv(match.group(1)), locale)
        translated_parts.append(template.format(", ".join(labels)))

    return " ".join(translated_parts) if translated_parts else text


def localize_value(value: str, locale: str | None) -> str:
    """Localize a structured status value for user-facing summaries."""

    return display_label(value, locale)


def _split_csv(text: str) -> list[str]:
    return [part.strip() for part in text.split(",") if part.strip()]


def _contains_korean(text: str) -> bool:
    return any("\uac00" <= char <= "\ud7a3" for char in text)


__all__ = [
    "display_label",
    "display_list",
    "localize_change_summary",
    "localize_value",
    "wants_korean",
]
