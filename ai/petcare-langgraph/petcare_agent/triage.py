from __future__ import annotations

import re
import time
from typing import Any, Literal

from ..models import PetCareState, Route
from ..utils import node_result


EMERGENCY_RULES: dict[str, dict[str, str]] = {
    "respiratory_distress": {
        "rule_id": "ER-RESP-001",
        "message": "호흡곤란 또는 숨을 제대로 쉬지 못하는 표현",
    },
    "cyanosis": {
        "rule_id": "ER-RESP-002",
        "message": "혀나 잇몸의 파랑·보라·회색 변화",
    },
    "unconsciousness": {
        "rule_id": "ER-NEURO-001",
        "message": "의식 소실 또는 반응 없음",
    },
    "seizure": {
        "rule_id": "ER-NEURO-002",
        "message": "경련 또는 발작",
    },
    "severe_bleeding": {
        "rule_id": "ER-BLEED-001",
        "message": "멈추지 않는 심한 출혈",
    },
    "urinary_obstruction": {
        "rule_id": "ER-URINE-001",
        "message": "소변을 전혀 보지 못함",
    },
    "toxin_ingestion": {
        "rule_id": "ER-TOXIN-001",
        "message": "독성물질 섭취",
    },
    "severe_deterioration": {
        "rule_id": "ER-GENERAL-001",
        "message": (
            "거의 움직이지 못하거나 상태가 급격히 매우 "
            "나빠 보인다는 고위험 악화 표현"
        ),
    },
    "owner_urgent_worsening": {
        "rule_id": "ER-GENERAL-002",
        "message": (
            "갑작스러운 악화가 반복 보고되었고 보호자가 "
            "상태를 확인하기 어려울 정도로 심각하게 우려함"
        ),
    },
}


RAW_EMERGENCY_PATTERNS: list[tuple[str, str]] = [
    (
        r"숨을\s*(못|안)\s*쉬|"
        r"숨(?:이|은|도)?\s*(?:많이\s*)?(?:힘들|어렵)|"
        r"숨\s*쉬(?:기|는\s*게)?(?:도|가|이)?\s*(?:많이\s*)?(?:힘들|어렵)|"
        r"숨쉬(?:기|는\s*게)?(?:도|가|이)?\s*(?:많이\s*)?(?:힘들|어렵)|"
        r"호흡(?:이|도|을)?\s*(?:많이\s*)?(?:힘들|어렵|곤란)|"
        r"호흡\s*곤란|숨이\s*막|숨이\s*(?:차|가쁘|가빠)|"
        r"숨을\s*가쁘게|호흡이\s*가빠|"
        r"입을\s*벌리고\s*숨|헐떡",
        "respiratory_distress",
    ),
    (
        r"혀.{0,8}(파래|파랗|보라|회색)|"
        r"잇몸.{0,8}(파래|파랗|보라|회색)",
        "cyanosis",
    ),
    (
        r"의식이\s*없|반응이\s*없|쓰러져서\s*안\s*일어",
        "unconsciousness",
    ),
    (r"경련|발작", "seizure"),
    (
        r"피가\s*안\s*멎|심한\s*출혈|피를\s*계속",
        "severe_bleeding",
    ),
    (
        r"소변을\s*(전혀\s*)?(못|안)\s*(봐|봄|본)",
        "urinary_obstruction",
    ),
    (
        r"초콜릿|자일리톨|살충제|쥐약|부동액|독성물질",
        "toxin_ingestion",
    ),
    (
        r"거의\s*(?:못|안)\s*움직|"
        r"아예\s*(?:못|안)\s*움직|"
        r"전혀\s*(?:못|안)\s*움직|"
        r"일어나(?:지)?\s*(?:못|않)|"
        r"걷(?:지)?\s*(?:못|않)|"
        r"서(?:지)?\s*(?:못|않)|"
        r"몸을\s*(?:못|안)\s*가누|"
        r"고개를\s*(?:못|안)\s*들|"
        r"축\s*늘어져.{0,10}반응.{0,6}(?:없|둔)|"
        r"반응이\s*(?:거의\s*)?(?:없|매우\s*둔)|"
        r"상태가.{0,12}(?:너무|매우|심하게)\s*안\s*좋|"
        r"상태가.{0,12}급격히\s*(?:나빠|악화)|"
        r"급격히\s*(?:나빠|악화)|"
        r"위급해\s*보|"
        r"죽을\s*것\s*같",
        "severe_deterioration",
    ),
]


PAST_SEVERE_MARKERS = [
    r"아까(?:는)?",
    r"전에는",
    r"이전에는",
    r"처음에는",
    r"방금\s*전(?:에는)?",
]


CURRENT_RECOVERY_PATTERNS = [
    r"지금(?:은|는)?.{0,18}(?:걸어|움직여|움직이|일어나|괜찮|정상|나아|회복)",
    r"현재(?:는|은)?.{0,18}(?:걸어|움직여|움직이|일어나|괜찮|정상|나아|회복)",
]


def is_recovered_severe_match(
    segment: str,
    match: re.Match[str],
) -> bool:
    before = segment[:match.start()]
    after = segment[match.end():]

    return (
        any(
            re.search(pattern, before)
            for pattern in PAST_SEVERE_MARKERS
        )
        and any(
            re.search(pattern, after)
            for pattern in CURRENT_RECOVERY_PATTERNS
        )
    )


def detect_recovery_hits(text: str) -> list[dict[str, Any]]:
    hits: list[dict[str, Any]] = []

    segments = [
        segment.strip()
        for segment in re.split(
            r"[\n\r.!?]+",
            text,
        )
        if segment.strip()
    ]

    severe_pattern = next(
        pattern
        for pattern, code in RAW_EMERGENCY_PATTERNS
        if code == "severe_deterioration"
    )

    for segment in segments:
        for match in re.finditer(severe_pattern, segment):
            if is_recovered_severe_match(
                segment,
                match,
            ):
                hits.append(
                    {
                        "symptom_code": "severe_deterioration",
                        "status": "recovered_now",
                        "evidence": segment,
                        "message": (
                            "과거 심한 기능 저하 후 현재 회복이 "
                            "명시된 표현"
                        ),
                    }
                )

    return hits


NEGATION_PATTERNS = [
    r"없(?:어|어요|음|습니다)",
    r"아니(?:야|에요|다|요)",
    r"그렇지\s*않",
    r"않(?:아|아요|음|습니다)",
    r"안\s*(?:해|했|먹|마셔|삼켜|보여|쉬|힘들|어렵|가쁘|빠르|차)",
    r"괜찮(?:아|아요|음|습니다)?",
    r"정상(?:이야|이에요|입니다)?",
]


YES_PATTERNS = [
    r"^(응|네|예|맞아|맞아요|그래|그래요|그렇다|그렇습니다)$",
    r"(있어|있어요|보여|보여요|그래|맞아)",
]


NO_PATTERNS = [
    r"^(아니|아니야|아니요|아뇨|없어|없어요|정상)$",
    r"(그렇지\s*않|그렇지는\s*않|없|아니|정상)",
]


def _emergency_match_negated(
    segment: str,
    match: re.Match[str],
) -> bool:
    before = segment[: match.start()]
    after = segment[match.end() :]

    contrast_pattern = re.compile(
        r"(?:하지만|그렇지만|반면|대신|다만|그런데|,)"
    )
    before_markers = list(
        contrast_pattern.finditer(before)
    )
    if before_markers:
        before = before[before_markers[-1].end() :]

    after_marker = contrast_pattern.search(after)
    if after_marker:
        after = after[: after_marker.start()]

    before = before[-18:]
    after = after[:18]

    before_negated = bool(
        re.search(
            r"(?:없|안|않|정상|괜찮)(?:고|지만|는데)?\s*$",
            before,
        )
    )
    after_negated = bool(
        re.search(
            r"^(?:은|는|이|가|도|을|를)?\s*"
            r"(?:안|않|없|정상|괜찮)",
            after,
        )
        or re.search(
            r"(?:지\s*않|지\s*안|아니)",
            after,
        )
    )

    return before_negated or after_negated


def raw_keyword_hits(text: str) -> set[str]:
    hits: set[str] = set()
    segments = [
        segment.strip()
        for segment in re.split(
            r"[\n\r.!?]+",
            text,
        )
        if segment.strip()
    ]

    for segment in segments:
        for pattern, code in RAW_EMERGENCY_PATTERNS:
            for match in re.finditer(pattern, segment):
                if (
                    code == "severe_deterioration"
                    and is_recovered_severe_match(
                        segment,
                        match,
                    )
                ):
                    continue

                if _emergency_match_negated(
                    segment,
                    match,
                ):
                    continue

                hits.add(code)

    return hits

def answer_polarity(answer: str) -> Literal["yes", "no", "unknown"]:
    normalized = answer.strip().lower()


    if any(re.search(pattern, normalized) for pattern in NO_PATTERNS):
        return "no"

    if any(re.search(pattern, normalized) for pattern in YES_PATTERNS):
        return "yes"

    return "unknown"


def collect_user_health_text(state: PetCareState) -> str:
    parts = [state.get("user_input", "")]

    for item in state.get("follow_up_history", []):
        answer = item.get("answer")
        if answer:
            parts.append(str(answer))

    return "\n".join(
        part.strip()
        for part in parts
        if part and part.strip()
    )


def follow_up_emergency_codes(
    history: list[dict[str, Any]],
) -> set[str]:
    codes: set[str] = set()

    for item in history:
        answer = str(item.get("answer", ""))
        codes |= raw_keyword_hits(answer)

    return codes


URGENT_WORSENING_PATTERNS = [
    r"갑자기.{0,20}(?:더\s*)?안\s*좋",
    r"갑자기.{0,20}(?:나빠|악화)",
    r"더\s*안\s*좋아졌",
    r"상태가.{0,15}더\s*나빠",
    r"급격히.{0,12}(?:나빠|악화)",
]


STRONG_OWNER_CONCERN_PATTERNS = [
    r"너무\s*안\s*좋아\s*보",
    r"너무\s*안\s*좋아보",
    r"상태가.{0,15}(?:너무|매우)\s*안\s*좋",
    r"어떡해|어떻게\s*해|어쩌지",
    r"급한\s*것\s*같|급해|위급",
]


UNCERTAIN_OBSERVATION_PATTERNS = [
    r"모르겠",
    r"잘\s*모르",
    r"확인\s*(?:못|안)\s*했",
    r"뭐가\s*문제인지\s*모르",
]


def detect_owner_urgent_worsening(
    state: PetCareState,
) -> bool:
    history = state.get(
        "follow_up_history",
        [],
    )


                                    

    if len(history) < 1:
        return False

    health_text = collect_user_health_text(state)

    has_worsening = any(
        re.search(pattern, health_text)
        for pattern in URGENT_WORSENING_PATTERNS
    )
    has_strong_concern = any(
        re.search(pattern, health_text)
        for pattern in STRONG_OWNER_CONCERN_PATTERNS
    )
    has_uncertainty = any(
        re.search(pattern, health_text)
        for pattern in UNCERTAIN_OBSERVATION_PATTERNS
    )

    return (
        has_worsening
        and has_strong_concern
        and has_uncertainty
    )


PRIORITY_EMERGENCY_CODES = set(EMERGENCY_RULES)


def assessment_emergency_codes(
    state: PetCareState,
) -> set[str]:
    assessment = state.get("assessment", {})

    return {
        str(item.get("code"))
        for item in assessment.get("symptoms", [])
        if (
            not item.get("negated", False)
            and item.get("code") in PRIORITY_EMERGENCY_CODES
        )
    }


def current_priority_emergency_codes(
    state: PetCareState,
) -> set[str]:
    health_text = collect_user_health_text(state)

    raw_codes = raw_keyword_hits(health_text)
    structured_codes = assessment_emergency_codes(
        state
    )
    recovery_hits = detect_recovery_hits(
        health_text
    )


    if "severe_deterioration" not in raw_codes:
        structured_codes.discard(
            "severe_deterioration"
        )

    if recovery_hits:
        structured_codes.discard(
            "severe_deterioration"
        )


                                 
    if detect_owner_urgent_worsening(state):
        raw_codes.add(
            "owner_urgent_worsening"
        )

    return raw_codes | structured_codes


def safety_guard(state: PetCareState) -> dict[str, Any]:
    started = time.perf_counter()

    health_text = collect_user_health_text(state)
    symptom_codes = current_priority_emergency_codes(
        state
    )
    recovery_hits = detect_recovery_hits(
        health_text
    )

    matched = [
        {
            "symptom_code": code,
            **EMERGENCY_RULES[code],
        }
        for code in sorted(symptom_codes)
        if code in EMERGENCY_RULES
    ]

    route: Route = (
        "emergency"
        if matched
        else "non_emergency"
    )

    return node_result(
        state,
        node_name="safety_guard",
        started_at=started,
        updates={
            "route": route,
            "emergency_hits": matched,
            "recovery_hits": recovery_hits,
            "needs_user_response": False,
        },
    )
