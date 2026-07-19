from __future__ import annotations

import json
import re
from datetime import date
from typing import Any, Iterable

from .models import PetCareState, PromptContext


MAX_DEFAULT_DAILY_ENTRIES = 8
MAX_DEFAULT_DIAGNOSES = 3
MAX_RAW_TEXT_LENGTH = 180


TOPIC_KEYWORDS: dict[str, tuple[str, ...]] = {
    "appetite_loss": (
        "밥",
        "사료",
        "식사",
        "식욕",
        "먹",
        "간식",
    ),
    "vomiting": (
        "구토",
        "토",
        "토함",
    ),
    "diarrhea": (
        "설사",
        "묽은 변",
        "물변",
        "배변",
        "변",
    ),
    "fever": (
        "발열",
        "열",
        "체온",
    ),
    "lethargy": (
        "기력",
        "활동",
        "무기력",
        "축 처",
        "움직",
    ),
    "pain": (
        "통증",
        "아파",
        "낑낑",
        "절뚝",
    ),
    "urinary_abnormality": (
        "소변",
        "오줌",
        "배뇨",
    ),
    "respiratory_issue": (
        "기침",
        "호흡",
        "숨",
        "헐떡",
    ),
    "pallor": (
        "창백",
        "잇몸",
        "혀 색",
    ),
    "ear": (
        "귀",
        "외이염",
        "머리 흔",
        "긁",
    ),
}


PROFILE_DETAIL_KEYWORDS = (
    "약",
    "복용",
    "투약",
    "알레르기",
    "질환",
    "진단",
    "병력",
)

DIAGNOSIS_KEYWORDS = (
    "진단",
    "병원",
    "처방",
    "약",
    "복용",
    "치료",
    "병력",
    "외이염",
)


PROFILE_KEYS = (
    "id",
    "name",
    "species",
    "breed",
    "birth_date",
    "sex",
    "is_neutered",
    "neutered",
    "weight_kg",
    "size_class",
)


DATE_PATTERN = re.compile(r"(20\d{2})[-./년\s](\d{1,2})[-./월\s](\d{1,2})일?")

ANOMALY_QUERY_PATTERN = re.compile(
    r"줄|감소|저하|이상|문제|아팠|안\s*좋|나빴|악화|남겼|못\s*먹|적게"
)

APPETITE_ANOMALY_PATTERN = re.compile(
    r"남김|남겼|감소|저하|절반|거의\s*못|안\s*먹|"
    r"적게|평소보다\s*(?:덜|적)|(?:[1-9]\d?)\s*%"
)

GENERAL_ANOMALY_PATTERN = re.compile(
    r"감소|저하|악화|이상|남김|남겼|못\s*먹|안\s*먹|"
    r"설사|묽|구토|토함|피곤|무기력|통증|창백|기침|"
    r"호흡곤란|긁|분비물|냄새|혈변|피"
)


def _safe_text(value: Any) -> str:
    if value is None:
        return ""

    if isinstance(value, str):
        return value

    return json.dumps(
        value,
        ensure_ascii=False,
        default=str,
    )


def _unique(values: Iterable[str]) -> list[str]:
    result: list[str] = []

    for value in values:
        normalized = value.strip().lower()
        if normalized and normalized not in result:
            result.append(normalized)

    return result


def symptom_codes_from_state(
    state: PetCareState,
) -> list[str]:
    values: list[str] = []
    assessment = state.get("assessment", {})

    for item in assessment.get("symptoms", []):
        if item.get("negated", False):
            continue

        code = str(item.get("code", "")).strip()
        if code:
            values.append(code)

    strategy = state.get("question_strategy", {})
    values.extend(
        str(code)
        for code in strategy.get(
            "detected_symptoms",
            [],
        )
    )

    return _unique(values)


def _query_terms(
    query: str,
    symptom_codes: list[str],
) -> list[str]:
    terms: list[str] = []

    for code in symptom_codes:
        terms.extend(TOPIC_KEYWORDS.get(code, ()))

    for keywords in TOPIC_KEYWORDS.values():
        if any(keyword in query for keyword in keywords):
            terms.extend(keywords)

    raw_words = re.findall(
        r"[가-힣A-Za-z0-9]{2,}",
        query,
    )
    terms.extend(raw_words)

    return _unique(terms)


def _requested_entry_limit(query: str) -> int:
    if re.search(r"한\s*달|30\s*일|월간|전체\s*기록", query):
        return 31

    if re.search(r"2\s*주|14\s*일", query):
        return 14

    if re.search(r"일주일|7\s*일|이번\s*주|지난\s*주", query):
        return 7

    if re.search(r"최근", query):
        return 7

    return MAX_DEFAULT_DAILY_ENTRIES


def _requested_dates(query: str) -> set[str]:
    dates: set[str] = set()

    for year, month, day in DATE_PATTERN.findall(query):
        try:
            parsed = date(
                int(year),
                int(month),
                int(day),
            )
        except ValueError:
            continue

        dates.add(parsed.isoformat())

    return dates


def _compact_entry(
    entry: dict[str, Any],
    *,
    terms: list[str],
) -> dict[str, Any]:
    compact: dict[str, Any] = {
        "record_date": entry.get("record_date"),
    }

    field_topics: dict[str, tuple[str, ...]] = {
        "food": TOPIC_KEYWORDS["appetite_loss"],
        "water": ("물", "수분", "마시"),
        "activity": TOPIC_KEYWORDS["lethargy"],
        "symptom": tuple(
            keyword
            for values in TOPIC_KEYWORDS.values()
            for keyword in values
        ),
        "stool": TOPIC_KEYWORDS["diarrhea"],
        "vomit": TOPIC_KEYWORDS["vomiting"],
        "notes": tuple(terms),
    }

    selected_fields: list[str] = []

    for field_name, keywords in field_topics.items():
        value = entry.get(field_name)
        if value in (None, "", []):
            continue

        value_text = _safe_text(value).lower()
        if (
            not terms
            or any(term in value_text for term in terms)
            or any(keyword in terms for keyword in keywords)
        ):
            selected_fields.append(field_name)

    if not selected_fields:
        selected_fields = [
            field_name
            for field_name in (
                "food",
                "activity",
                "symptom",
                "stool",
                "vomit",
                "notes",
            )
            if entry.get(field_name) not in (None, "", [])
        ][:4]

    for field_name in selected_fields:
        compact[field_name] = entry.get(field_name)

    raw_text = str(entry.get("raw_text") or "").strip()
    if raw_text and any(term in raw_text.lower() for term in terms):
        compact["raw_text"] = raw_text[:MAX_RAW_TEXT_LENGTH]

    return compact


def _entry_anomaly_score(
    entry: dict[str, Any],
    *,
    query: str,
    symptom_codes: list[str],
) -> int:
    if not ANOMALY_QUERY_PATTERN.search(query):
        return 0

    score = 0
    full_text = _safe_text(entry).lower()

    appetite_focused = (
        "appetite_loss" in symptom_codes
        or any(
            keyword in query
            for keyword in TOPIC_KEYWORDS["appetite_loss"]
        )
    )

    if appetite_focused:
        food_text = _safe_text(entry.get("food")).lower()
        symptom_text = _safe_text(entry.get("symptom")).lower()

        if APPETITE_ANOMALY_PATTERN.search(food_text):
            score += 40

        if (
            "식사량 감소" in symptom_text
            or "식욕 저하" in symptom_text
        ):
            score += 20

        return score

    if (
        GENERAL_ANOMALY_PATTERN.search(full_text)
        and not re.search(
            r"특별한 이상 없음|증상 없음|모두 정상",
            full_text,
        )
    ):
        score += 10

    return score


def _entry_score(
    entry: dict[str, Any],
    *,
    query: str,
    symptom_codes: list[str],
    terms: list[str],
    recency_rank: int,
) -> tuple[int, int]:
    text = _safe_text(entry).lower()
    topic_score = sum(
        5
        for term in terms
        if term and term in text
    )
    anomaly_score = _entry_anomaly_score(
        entry,
        query=query,
        symptom_codes=symptom_codes,
    )
    recency_score = max(0, 3 - recency_rank)
    return topic_score + anomaly_score + recency_score, anomaly_score


def select_daily_entries(
    context: dict[str, Any],
    *,
    query: str,
    symptom_codes: list[str],
) -> list[dict[str, Any]]:
    entries = [
        item
        for item in context.get("daily_entries", [])
        if isinstance(item, dict)
    ]

    if not entries:
        return []

    sorted_entries = sorted(
        entries,
        key=lambda item: str(item.get("record_date", "")),
        reverse=True,
    )

    exact_dates = _requested_dates(query)
    terms = _query_terms(query, symptom_codes)
    limit = _requested_entry_limit(query)

    if exact_dates:
        selected = [
            entry
            for entry in sorted_entries
            if str(entry.get("record_date")) in exact_dates
        ]
    else:
        scored = [
            (
                *_entry_score(
                    entry,
                    query=query,
                    symptom_codes=symptom_codes,
                    terms=terms,
                    recency_rank=index,
                ),
                entry,
            )
            for index, entry in enumerate(sorted_entries)
        ]

        anomaly_matches = [
            pair
            for pair in scored
            if pair[1] > 0
        ]
        matched = [
            pair
            for pair in scored
            if pair[0] > 3
        ]

        candidates = anomaly_matches or matched
        if candidates:
            candidates.sort(
                key=lambda pair: (
                    pair[1],
                    pair[0],
                    str(pair[2].get("record_date", "")),
                ),
                reverse=True,
            )
            selected = [
                entry
                for _, _, entry in candidates[:limit]
            ]
        else:
            selected = sorted_entries[:limit]

    selected.sort(
        key=lambda item: str(item.get("record_date", ""))
    )

    return [
        _compact_entry(entry, terms=terms)
        for entry in selected
    ]


def _compact_diagnosis(
    item: dict[str, Any],
) -> dict[str, Any]:
    return {
        key: item.get(key)
        for key in (
            "date",
            "hospital",
            "diagnosis",
            "content",
        )
        if item.get(key) not in (None, "")
    }


def select_diagnoses(
    context: dict[str, Any],
    *,
    query: str,
    symptom_codes: list[str],
) -> list[dict[str, Any]]:
    diagnoses = [
        item
        for item in context.get("diagnoses", [])
        if isinstance(item, dict)
    ]

    if not diagnoses:
        return []

    terms = _query_terms(query, symptom_codes)
    include_by_intent = any(
        keyword in query
        for keyword in DIAGNOSIS_KEYWORDS
    )

    scored: list[tuple[int, dict[str, Any]]] = []
    for item in diagnoses:
        text = _safe_text(item).lower()
        score = sum(
            4
            for term in terms
            if term and term in text
        )
        if include_by_intent:
            score += 2
        scored.append((score, item))

    relevant = [
        pair
        for pair in scored
        if pair[0] > 0
    ]

    if not relevant:
        return []

    relevant.sort(
        key=lambda pair: (
            pair[0],
            str(pair[1].get("date", "")),
        ),
        reverse=True,
    )

    return [
        _compact_diagnosis(item)
        for _, item in relevant[:MAX_DEFAULT_DIAGNOSES]
    ]


def select_profile(
    pet: dict[str, Any],
    *,
    query: str,
) -> dict[str, Any]:
    selected = {
        key: pet.get(key)
        for key in PROFILE_KEYS
        if pet.get(key) is not None
    }

    if any(
        keyword in query
        for keyword in PROFILE_DETAIL_KEYWORDS
    ):
        for key in (
            "medications",
            "allergies",
            "chronic_conditions",
            "diseases_medications_allergies",
        ):
            if pet.get(key):
                selected[key] = pet.get(key)

    return selected


def build_prompt_context(
    state: PetCareState,
) -> PromptContext:
    context = state.get("backend_context", {})
    query = state.get("user_input", "").strip()
    symptom_codes = symptom_codes_from_state(state)

    daily_entries = select_daily_entries(
        context,
        query=query,
        symptom_codes=symptom_codes,
    )
    diagnoses = select_diagnoses(
        context,
        query=query,
        symptom_codes=symptom_codes,
    )

    data_from = context.get("data_from")
    data_to = context.get("data_to")
    if not data_from or not data_to:
        dates = sorted(
            str(item.get("record_date"))
            for item in context.get("daily_entries", [])
            if isinstance(item, dict)
            and item.get("record_date")
        )
        if dates:
            data_from = data_from or dates[0]
            data_to = data_to or dates[-1]

    data_period = (
        f"{data_from} ~ {data_to}"
        if data_from or data_to
        else "미확인"
    )

    return PromptContext(
        pet=select_profile(
            context.get("pet", {}),
            query=query,
        ),
        daily_entries=daily_entries,
        diagnoses=diagnoses,
        data_period=data_period,
        selection_note=(
            "현재 질문과 감지된 증상에 관련된 등록 기록만 선택함"
        ),
    )
