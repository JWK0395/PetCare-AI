from __future__ import annotations

import re
import time
from typing import Any

from langgraph.types import interrupt
from pydantic import ValidationError

from ..models import (
    FollowUpItem,
    PetCareState,
    QuestionStrategy,
    SymptomItem,
)
from ..utils import (
    node_result,
    trim_conversation_history,
)
from .safety import (
    collect_user_health_text,
    current_priority_emergency_codes,
    detect_recovery_hits,
    raw_keyword_hits,
)


MAX_TRIAGE_QUESTION_TURNS = 3


SYMPTOM_QUESTION_BANK: dict[str, list[str]] = {
    "appetite_loss": [
        "밥을 평소의 절반 정도는 먹었나요, 거의 못 먹었나요?",
        "이 변화는 언제부터 시작됐고, 계속 그런가요?",
    ],
    "vomiting": [
        "구토는 몇 번 했고, 색이나 내용물에 특이한 점이 있었나요?",
        "언제부터 시작됐고 물이나 음식도 계속 토하나요?",
    ],
    "diarrhea": [
        "설사는 몇 번 했고 물처럼 묽거나 피가 섞였나요?",
        "언제부터 시작됐고 계속 이어지고 있나요?",
    ],
    "fever": [
        "체온을 실제로 측정했다면 몇 도였나요?",
        "열감은 언제부터 있었고 계속되나요?",
    ],
    "lethargy": [
        "평소보다 덜 움직이는 정도인가요, 거의 움직이지 못하나요?",
        "기력 저하는 언제부터 시작됐고 식사나 반응도 함께 줄었나요?",
    ],
    "pain": [
        "어느 부위를 아파하는 것 같고 만지면 피하거나 소리를 내나요?",
        "통증은 언제부터였고 움직일 때 더 심해 보이나요?",
    ],
    "urinary_abnormality": [
        "소변을 보는 횟수와 양이 평소보다 어떻게 달라졌나요?",
        "소변을 보려고 자주 시도하거나 힘들어하는 모습이 있나요?",
    ],
    "respiratory_issue": [
        "기침이나 호흡 변화는 언제부터 시작됐고 계속되나요?",
        "가만히 있을 때도 숨이 빠르거나 힘들어 보이나요?",
    ],
    "pallor": [
        "창백해 보인다는 것이 잇몸이나 혀가 평소보다 하얗거나 회색으로 보인다는 뜻인가요?",
        "이 변화는 언제부터 보였고 지금도 같은가요?",
    ],
    "other": [
        "가장 눈에 띄는 증상을 한 가지로 설명해 주세요.",
        "그 변화는 언제부터 시작됐고 계속되고 있나요?",
    ],
}


SYMPTOM_LABELS: dict[str, str] = {
    "appetite_loss": "식욕저하",
    "vomiting": "구토",
    "diarrhea": "설사",
    "fever": "발열",
    "lethargy": "기력저하",
    "pain": "통증",
    "urinary_abnormality": "배뇨 이상",
    "respiratory_issue": "호흡 문제",
    "pallor": "창백함",
    "other": "기타 증상",
}


SYMPTOM_PATTERNS: list[tuple[str, list[str]]] = [
    (
        "appetite_loss",
        [
            r"밥.{0,10}(?:못|안|덜|남|줄|적게)",
            r"사료.{0,10}(?:못|안|덜|남|줄|적게)",
            r"식욕.{0,10}(?:없|저하|감소|떨어)",
            r"먹는\s*양.{0,10}(?:줄|감소|적)",
            r"거의\s*못\s*먹",
        ],
    ),
    (
        "vomiting",
        [
            r"구토",
            r"토를?\s*(?:했|해|함|하|했어|했어요)",
            r"토했",
        ],
    ),
    (
        "diarrhea",
        [
            r"설사",
            r"묽은\s*변",
            r"물\s*같은\s*변",
            r"물변",
        ],
    ),
    (
        "fever",
        [
            r"발열",
            r"열(?:이|도|가)?\s*(?:나|있|오르|높)",
            r"체온.{0,8}(?:높|도)",
        ],
    ),
    (
        "lethargy",
        [
            r"축\s*처",
            r"기력.{0,8}(?:저하|없|떨어)",
            r"무기력",
            r"활동량.{0,8}(?:줄|감소)",
            r"잘\s*안\s*움직",
        ],
    ),
    (
        "pain",
        [
            r"통증",
            r"낑낑",
            r"만지면.{0,8}(?:피|싫|울|소리)",
            r"절뚝|절름",
            r"아파서",
            r"움직일\s*때.{0,8}아파",
            r"(?:배|다리|허리|목|귀|입).{0,8}아파",
        ],
    ),
    (
        "urinary_abnormality",
        [
            r"(?:소변|오줌|배뇨).{0,12}(?:줄|늘|적|많|자주|힘들|못|안|통증|피|혈|이상|찔끔)",
            r"(?:자주|계속).{0,8}(?:소변|오줌)",
            r"(?:소변|오줌).{0,8}(?:조금씩|찔끔)",
        ],
    ),
    (
        "respiratory_issue",
        [
            r"기침",
            r"(?:호흡|숨).{0,12}(?:빠르|가쁘|힘들|이상|거칠|불규칙|소리|차)",
            r"(?:빠른|가쁜|거친).{0,8}(?:호흡|숨)",
            r"헐떡",
        ],
    ),
    (
        "pallor",
        [
            r"창백",
            r"(?:잇몸|혀).{0,10}(?:하얗|희|회색)",
        ],
    ),
]


ASSESSMENT_TO_CYCLE_SYMPTOM: dict[str, str] = {
    "appetite_loss": "appetite_loss",
    "vomiting": "vomiting",
    "diarrhea": "diarrhea",
    "fever": "fever",
    "lethargy": "lethargy",
    "pain": "pain",
    "urinary_abnormality": "urinary_abnormality",
    "urinary_obstruction": "urinary_abnormality",
    "respiratory_issue": "respiratory_issue",
    "respiratory_distress": "respiratory_issue",
    "pallor": "pallor",
}


ADDITIONAL_SYMPTOM_QUESTION = (
    "추가 증상을 확인하겠습니다. "
    "호흡이 힘들거나, 쓰러짐·반응 저하, 잇몸·혀 색 이상이 있으면 "
    "해당 증상만 바로 적어 주세요. "
    "구토, 설사, 통증, 배뇨 이상도 함께 적을 수 있습니다. "
    "없으면 '추가 증상 없음'이라고 답해 주세요."
)


UNKNOWN_ANSWER_PATTERNS = [
    r"모르겠",
    r"잘\s*모르",
    r"확인\s*(?:못|안)\s*(?:함|해|했|돼|됨)?",
    r"판단이\s*안",
    r"잘\s*안\s*보",
    r"애매해",
]


UNKNOWN_ADDITIONAL_QUESTION = (
    "추가 증상 여부가 확인되지 않았습니다. "
    "현재 보이는 범위에서 호흡곤란, 쓰러짐·반응 저하, "
    "잇몸·혀 색 이상, 구토, 설사, 통증, 배뇨 이상 중 "
    "해당되는 증상이 있나요? "
    "확인하기 어렵다면 '확인 못함'이라고 답해 주세요."
)


NO_ADDITIONAL_PATTERNS = [
    r"추가\s*증상\s*(?:없|없음|없어|없어요|없습니다)",
    r"딱히\s*(?:없|없어|없어요)",
    r"다른\s*(?:건|것|증상).{0,6}(?:딱히\s*)?(?:없|없는|없어|없어요|없습니다)",
    r"다른\s*(?:건|것).{0,4}딱히",
    r"그\s*외에는?\s*딱히",
    r"그게\s*다",
    r"더\s*이상\s*(?:없|없어|없어요)",
    r"^(?:없어|없어요|없습니다|없음)$",
]


POST_TRIAGE_ACK_PATTERNS = [
    r"추가\s*증상(?:은|이)?\s*(?:없|없음|없어|없어요|없습니다)",
    r"^(?:없어|없어요|없습니다|없음|없다고)$",
    r"아까\s*말했",
    r"이미\s*말했",
    r"말했잖",
    r"알겠어|알겠습니다|확인했어|확인했습니다",
    r"고마워|감사합니다",
]


NEW_TRIAGE_GENERIC_PATTERNS = [
    r"새로운?\s*증상",
    r"다른\s*증상",
    r"이번엔|이번에는",
    r"새로",
    r"갑자기",
    r"상태가.{0,8}(?:안\s*좋|나빠|이상)",
    r"몸\s*상태가.{0,8}(?:안\s*좋|이상)",
    r"아픈\s*것\s*같",
    r"아파\s*보",
    r"창백해\s*보",
]


NEGATION_AFTER_PATTERNS = [
    r"^(?:은|는|이|가|도|을|를)?\s*(?:없|안|않|정상|괜찮)",
    r"^(?:은|는|이|가|도|을|를)?\s*그렇지\s*않",
]

NEGATION_BEFORE_PATTERNS = [
    r"(?:없|안|않|정상|괜찮)(?:고|으며|지만|는데)?\s*$",
    r"없이\s*$",
]

CONTRAST_MARKERS = re.compile(
    r"(?:하지만|그렇지만|지만|으나|반면|대신|다만|그런데|,|\.|!|\?|\n)"
)

NORMAL_COLOR_PATTERN = re.compile(
    r"(?:잇몸|혀).{0,12}(?:분홍|핑크|정상)"
)
ABNORMAL_PALE_COLOR_PATTERN = re.compile(
    r"(?:잇몸|혀).{0,12}(?:하얗|희|회색|파랗|보라)"
)


def is_unknown_answer(text: str) -> bool:
    normalized = text.strip()
    return any(
        re.search(pattern, normalized)
        for pattern in UNKNOWN_ANSWER_PATTERNS
    )


def no_additional_symptoms(text: str) -> bool:
    normalized = text.strip()
    return any(
        re.search(pattern, normalized)
        for pattern in NO_ADDITIONAL_PATTERNS
    )


def _before_clause(text: str, start: int) -> str:
    before = text[:start]
    markers = list(CONTRAST_MARKERS.finditer(before))
    if markers:
        before = before[markers[-1].end() :]
    return before[-18:]


def _after_clause(text: str, end: int) -> str:
    after = text[end:]
    marker = CONTRAST_MARKERS.search(after)
    if marker:
        after = after[: marker.start()]
    return after[:18]


def is_negated_symptom_match(
    text: str,
    start: int,
    end: int,
    *,
    symptom: str | None = None,
) -> bool:
    before = _before_clause(text, start)
    after = _after_clause(text, end)
    matched = text[start:end]

    if symptom == "respiratory_issue" and re.search(
        r"안\s*(?:힘들|가쁘|빠르|차|이상)|"
        r"힘들지\s*않|괜찮",
        matched,
    ):
        return True

    if symptom in {"pain", "pallor", "fever"} and re.search(
        r"안\s*(?:아파|창백|뜨거|높)|"
        r"(?:아프|창백|뜨겁|높)지\s*않",
        matched,
    ):
        return True

    return (
        any(
            re.search(pattern, after)
            for pattern in NEGATION_AFTER_PATTERNS
        )
        or any(
            re.search(pattern, before)
            for pattern in NEGATION_BEFORE_PATTERNS
        )
    )


def _skip_pallor_due_to_normal_color(text: str) -> bool:
    return bool(
        NORMAL_COLOR_PATTERN.search(text)
        and not ABNORMAL_PALE_COLOR_PATTERN.search(text)
    )


def detect_symptom_items(text: str) -> list[SymptomItem]:
    matches: list[tuple[int, SymptomItem]] = []
    seen: set[tuple[str, bool]] = set()

    for symptom, patterns in SYMPTOM_PATTERNS:
        if symptom == "pallor" and _skip_pallor_due_to_normal_color(text):
            continue

        for pattern in patterns:
            for match in re.finditer(pattern, text):
                negated = is_negated_symptom_match(
                    text,
                    match.start(),
                    match.end(),
                    symptom=symptom,
                )
                key = (symptom, negated)
                if key in seen:
                    continue

                seen.add(key)
                matches.append(
                    (
                        match.start(),
                        SymptomItem(
                            code=symptom,
                            evidence=match.group(0).strip(),
                            negated=negated,
                        ),
                    )
                )

    matches.sort(key=lambda item: item[0])
    return [item for _, item in matches]


def detect_symptoms(text: str) -> list[str]:
    return [
        item.code
        for item in detect_symptom_items(text)
        if not item.negated
    ]


def is_post_triage_acknowledgement(text: str) -> bool:
    normalized = text.strip()
    return any(
        re.search(pattern, normalized)
        for pattern in POST_TRIAGE_ACK_PATTERNS
    )


def should_open_new_triage(text: str) -> bool:
    normalized = text.strip()

    if raw_keyword_hits(normalized):
        return True

    if detect_symptoms(normalized):
        return True

    return any(
        re.search(pattern, normalized)
        for pattern in NEW_TRIAGE_GENERIC_PATTERNS
    )


def default_question_strategy() -> dict[str, Any]:
    return QuestionStrategy().model_dump()


def normalize_question_strategy(
    state: PetCareState,
) -> dict[str, Any]:
    current = state.get("question_strategy", {})

    try:
        return QuestionStrategy.model_validate(
            current or {}
        ).model_dump()
    except ValidationError:
        allowed = set(QuestionStrategy.model_fields)
        sanitized = {
            key: value
            for key, value in (current or {}).items()
            if key in allowed
        }
        return QuestionStrategy.model_validate(
            sanitized
        ).model_dump()


def all_cycle_symptoms(
    state: PetCareState,
    strategy: dict[str, Any],
) -> list[str]:
    detected: list[str] = list(
        strategy.get("detected_symptoms", [])
    )

    raw_text = collect_user_health_text(state)

    for symptom in detect_symptoms(raw_text):
        if symptom not in detected:
            detected.append(symptom)

    assessment = state.get("assessment", {})

    for item in assessment.get("symptoms", []):
        if item.get("negated", False):
            continue

        mapped = ASSESSMENT_TO_CYCLE_SYMPTOM.get(
            item.get("code", "")
        )

        if mapped and mapped not in detected:
            detected.append(mapped)

    if (
        not detected
        and assessment.get("intent") == "health_related"
    ):
        detected.append("other")

    return detected


def plan_question_cycle(
    state: PetCareState,
) -> dict[str, Any] | None:
    if state.get("errors"):
        return None

    assessment = state.get("assessment", {})

    if assessment.get("intent") != "health_related":
        return None

    if current_priority_emergency_codes(state):
        return None

    if detect_recovery_hits(
        collect_user_health_text(state)
    ):
        return None

    strategy = normalize_question_strategy(state)

    if len(
        state.get("follow_up_history", [])
    ) >= MAX_TRIAGE_QUESTION_TURNS:
        return None

    if strategy.get("finished", False):
        return None

    if strategy.get("awaiting_additional_check", False):
        question_text = (
            UNKNOWN_ADDITIONAL_QUESTION
            if strategy.get(
                "unknown_additional_retry_count",
                0,
            ) > 0
            else ADDITIONAL_SYMPTOM_QUESTION
        )

        return {
            "kind": "additional_symptoms",
            "field": "additional_symptoms",
            "symptom": None,
            "questions": [question_text],
            "question_text": question_text,
        }

    detected = all_cycle_symptoms(
        state,
        strategy,
    )
    completed = set(
        strategy.get("completed_cycles", [])
    )

    pending = [
        symptom
        for symptom in detected
        if symptom not in completed
    ]

    if not pending:
        return None

    symptom = pending[0]
    questions = SYMPTOM_QUESTION_BANK.get(
        symptom,
        SYMPTOM_QUESTION_BANK["other"],
    )[:2]

    label = SYMPTOM_LABELS.get(
        symptom,
        "주증상",
    )

    question_text = (
        f"{label}에 대해 다음 두 가지를 확인하겠습니다.\n"
        + "\n".join(
            f"{index}. {question}"
            for index, question in enumerate(
                questions,
                start=1,
            )
        )
    )

    return {
        "kind": "symptom_detail",
        "field": f"cycle:{symptom}",
        "symptom": symptom,
        "questions": questions,
        "question_text": question_text,
    }


def question_manager(state: PetCareState) -> dict[str, Any]:
    started = time.perf_counter()
    plan = plan_question_cycle(state)

    if plan is None:
        return node_result(
            state,
            node_name="question_manager",
            started_at=started,
            updates={"needs_user_response": False},
        )

    answer = interrupt(
        {
            "question": plan["question_text"],
            "field": plan["field"],
            "kind": plan["kind"],
            "symptom": plan.get("symptom"),
            "questions": plan["questions"],
            "needs_user_response": True,
        }
    )

    answer_text = str(answer).strip()
    strategy = normalize_question_strategy(state)

    if plan["kind"] == "symptom_detail":
        answer_status = "reported"
    elif no_additional_symptoms(answer_text):
        answer_status = "no"
    elif is_unknown_answer(answer_text):
        answer_status = "unknown"
    else:
        answer_status = "reported"

    history = list(
        state.get("follow_up_history", [])
    )
    history_item = FollowUpItem(
        field=plan["field"],
        kind=plan["kind"],
        symptom=plan.get("symptom"),
        questions=plan["questions"],
        question=plan["question_text"],
        answer=answer_text,
        answer_status=answer_status,
    )
    history.append(history_item.model_dump())

    if plan["kind"] == "symptom_detail":
        symptom = str(plan["symptom"])

        if symptom not in strategy["detected_symptoms"]:
            strategy["detected_symptoms"].append(symptom)

        if symptom not in strategy["completed_cycles"]:
            strategy["completed_cycles"].append(symptom)

        strategy["active_symptom"] = symptom
        strategy["cycle_history"].append(
            {
                "symptom": symptom,
                "questions": plan["questions"],
                "answer": answer_text,
                "answer_status": answer_status,
            }
        )
        strategy["awaiting_additional_check"] = True
        strategy["finished"] = False

    else:
        strategy["additional_checks"].append(
            {
                "question": plan["question_text"],
                "answer": answer_text,
                "answer_status": answer_status,
            }
        )
        strategy["additional_answer_status"] = answer_status
        strategy["awaiting_additional_check"] = False
        strategy["active_symptom"] = None

        newly_detected = detect_symptoms(answer_text)

        for symptom in newly_detected:
            if symptom not in strategy["detected_symptoms"]:
                strategy["detected_symptoms"].append(symptom)

        state_with_answer = {
            **state,
            "follow_up_history": history,
        }

        if current_priority_emergency_codes(
            state_with_answer
        ):
            strategy["finished"] = True
        elif answer_status == "no":
            strategy["finished"] = True
        elif newly_detected:
            strategy["finished"] = False
        elif answer_status == "unknown":
            retry_count = int(
                strategy.get(
                    "unknown_additional_retry_count",
                    0,
                )
            ) + 1
            strategy[
                "unknown_additional_retry_count"
            ] = retry_count

            if retry_count == 1:
                strategy["awaiting_additional_check"] = True
                strategy["finished"] = False
            else:
                strategy[
                    "unknown_additional_unresolved"
                ] = True
                strategy["finished"] = True
        else:
            strategy["finished"] = False

    strategy = QuestionStrategy.model_validate(
        strategy
    ).model_dump()

    conversation_history = list(
        state.get("conversation_history", [])
    )
    conversation_history.extend(
        [
            {
                "role": "assistant",
                "content": plan["question_text"],
            },
            {
                "role": "user",
                "content": answer_text,
            },
        ]
    )

    return node_result(
        state,
        node_name="question_manager",
        started_at=started,
        updates={
            "triage_status": "collecting",
            "follow_up_history": history,
            "question_strategy": strategy,
            "conversation_history": (
                trim_conversation_history(
                    conversation_history
                )
            ),
            "needs_user_response": False,
        },
    )
