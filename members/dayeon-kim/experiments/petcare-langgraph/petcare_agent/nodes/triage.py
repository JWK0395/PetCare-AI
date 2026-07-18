from __future__ import annotations

import re
import time
from typing import Any

from langgraph.types import interrupt

from ..models import PetCareState
from ..utils import node_result, trim_conversation_history
from .safety import (
    NEGATION_PATTERNS,
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
    "other": "기타 증상",
}


SYMPTOM_PATTERNS: list[tuple[str, list[str]]] = [
    (
        "appetite_loss",
        [
            r"밥.{0,8}(못|안|덜|남)",
            r"사료.{0,8}(못|안|덜|남|줄)",
            r"식욕.{0,8}(없|저하|감소|떨어)",
            r"거의\s*못\s*먹",
        ],
    ),
    (
        "vomiting",
        [
            r"구토",
            r"토를?\s*(했|해|함|하|했어|했어요)",
            r"토했",
        ],
    ),
    (
        "diarrhea",
        [
            r"설사",
            r"묽은\s*변",
            r"물\s*같은\s*변",
        ],
    ),
    (
        "fever",
        [
            r"발열",
            r"열(이|도|가)?\s*(나|있|오르|높)",
            r"체온.{0,8}(높|도)",
        ],
    ),
    (
        "lethargy",
        [
            r"축\s*처",
            r"기력.{0,8}(저하|없|떨어)",
            r"무기력",
            r"활동량.{0,8}(줄|감소)",
            r"잘\s*안\s*움직",
        ],
    ),
    (
        "pain",
        [
            r"통증",
            r"아파",
            r"낑낑",
            r"만지면.{0,8}(피|싫|울)",
        ],
    ),
    (
        "urinary_abnormality",
        [
            r"소변",
            r"오줌",
            r"배뇨",
        ],
    ),
    (
        "respiratory_issue",
        [
            r"기침",
            r"호흡",
            r"숨",
            r"헐떡",
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
    # severe_deterioration은 Cycle로 변환하지 않습니다.
    # route_after_assessment에서 Safety Guard로 바로 보냅니다.
}


ADDITIONAL_SYMPTOM_QUESTION = (
    "추가로 확인되는 증상이 있나요? "
    "호흡이 힘듦, 쓰러짐·반응 저하, 잇몸·혀 색 이상이 있으면 "
    "그 증상만 바로 적어주세요. "
    "그 외 구토, 설사, 통증, 배뇨 이상도 적을 수 있어요. "
    "없으면 '추가 증상 없음'이라고 답해주세요."
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
    "추가 증상이 없는 것으로 확정하지 않고 한 번만 다시 확인할게요.\n"
    "현재 직접 보이는 범위에서 구토, 설사, 통증, 배뇨 이상, "
    "호흡 이상, 쓰러짐·반응 저하 중 확인되는 것이 있나요?\n"
    "정말 확인하기 어렵다면 '확인 못함'이라고 답해주세요."
)


def is_unknown_answer(text: str) -> bool:
    normalized = text.strip()

    return any(
        re.search(pattern, normalized)
        for pattern in UNKNOWN_ANSWER_PATTERNS
    )


NO_ADDITIONAL_PATTERNS = [
    r"추가\s*증상\s*(없|없음|없어|없어요|없습니다)",
    r"딱히\s*(없|없어|없어요)",
    r"다른\s*증상\s*(없|없어|없어요|없습니다)",
    r"그게\s*다",
    r"더\s*이상\s*(없|없어|없어요)",
    r"^(없어|없어요|없습니다|없음)$",
]


def is_negated_symptom_match(
    text: str,
    start: int,
    end: int,
) -> bool:
    window_start = max(0, start - 14)
    window_end = min(len(text), end + 14)
    window = text[window_start:window_end]

    return any(
        re.search(pattern, window)
        for pattern in NEGATION_PATTERNS
    )


def detect_symptoms(text: str) -> list[str]:
    detected_with_position: list[tuple[int, str]] = []

    for symptom, patterns in SYMPTOM_PATTERNS:
        first_position: int | None = None

        for pattern in patterns:
            for match in re.finditer(pattern, text):
                if is_negated_symptom_match(
                    text,
                    match.start(),
                    match.end(),
                ):
                    continue

                if (
                    first_position is None
                    or match.start() < first_position
                ):
                    first_position = match.start()

        if first_position is not None:
            detected_with_position.append(
                (first_position, symptom)
            )

    detected_with_position.sort(
        key=lambda item: item[0]
    )

    return [
        symptom
        for _, symptom in detected_with_position
    ]


def no_additional_symptoms(text: str) -> bool:
    normalized = text.strip()

    return any(
        re.search(pattern, normalized)
        for pattern in NO_ADDITIONAL_PATTERNS
    )



POST_TRIAGE_ACK_PATTERNS = [
    r"추가\s*증상(?:은|이)?\s*(없|없음|없어|없어요|없습니다)",
    r"^(없어|없어요|없습니다|없음|없다고)$",
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
    r"상태가.{0,8}(안\s*좋|나빠|이상)",
    r"아픈\s*것\s*같",
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
    return {
        "detected_symptoms": [],
        "completed_cycles": [],
        "active_symptom": None,
        "cycle_history": [],
        "additional_checks": [],
        "additional_answer_status": None,
        "unknown_additional_retry_count": 0,
        "unknown_additional_unresolved": False,
        "awaiting_additional_check": False,
        "finished": False,
    }


def normalize_question_strategy(
    state: PetCareState,
) -> dict[str, Any]:
    strategy = default_question_strategy()
    current = state.get("question_strategy", {})

    if isinstance(current, dict):
        strategy.update(current)

    for key in [
        "detected_symptoms",
        "completed_cycles",
        "cycle_history",
        "additional_checks",
    ]:
        if not isinstance(strategy.get(key), list):
            strategy[key] = []

    return strategy


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

    # 하나의 상태 체크에서 질문 응답이 3회에 도달하면
    # 새로운 Cycle을 열지 않고 Safety Guard에서 결론을 냅니다.
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
        f"{label} 상태를 조금 더 확인할게요.\n"
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
    history.append(
        {
            "field": plan["field"],
            "kind": plan["kind"],
            "symptom": plan.get("symptom"),
            "questions": plan["questions"],
            "question": plan["question_text"],
            "answer": answer_text,
            "answer_status": answer_status,
        }
    )

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
        strategy["additional_answer_status"] = (
            answer_status
        )
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
