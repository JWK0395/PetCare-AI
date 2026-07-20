"""Missing Information / Contact Minimum Information Agent (명세 29·30·32절).

명세 29절의 핵심은 **상황마다 필수정보 세트가 다르다**는 것이다.

  - 일반 건강 상담 : 답변에 꼭 필요한 최소 항목만 (질문을 늘리면 상담이 안 된다)
  - 병원 상담 권고 : 진료 의뢰서에 들어갈 항목 (증상·시작·빈도·변화·현재 상태)
  - 응급 연락      : 병원 전화 통화에서 즉시 답해야 하는 항목 (의식·호흡·이동성 등)

그리고 두 가지 안전 규칙이 있다.

  1. **모든 항목은 '모름' 을 유효값으로 허용한다.** 보호자가 모르는 것을 계속
     캐물으면 응급 상황에서 시간만 잃는다. '모름' 은 미응답이 아니라 답변이며,
     PDF 에는 `unknown_fields` 로 그대로 보존된다(추측해 채우지 않는다 — 명세 47절).
  2. **critical_immediate 에서는 정보 수집이 전화 action 을 막지 않는다.**
     그래서 `contact_minimum_information_node` 는 즉시 위급일 때 부족한 항목을
     기록만 하고 `minimum_information_ready=True` 로 통과시키며,
     `missing_information_interrupt_node` 는 아예 interrupt 를 걸지 않는다.

`missing_fields` / `required_fields` 에는 **한국어 label** 을 넣는다. 이 값이 그대로
`ChatGraphResult.missing_information` 으로 나가 사용자에게 보이기 때문이다. 반면
`collected_information` 은 프로그램이 다루는 dict 이므로 ascii key 를 쓴다.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from ..state import Replace

if TYPE_CHECKING:
    from ..state import PetCareState

logger = logging.getLogger(__name__)

__all__ = [
    "InformationField",
    "GENERAL_HEALTH_FIELDS",
    "VISIT_FIELDS",
    "EMERGENCY_CONTACT_FIELDS",
    "UNKNOWN_ANSWERS",
    "UNKNOWN_VALUE",
    "MAX_QUESTIONS_PER_TURN",
    "fields_for_risk",
    "field_by_key",
    "field_by_label",
    "is_unknown_answer",
    "detect_answered_fields",
    "evaluate_missing_information",
    "build_question",
    "missing_information_node",
    "contact_minimum_information_node",
    "missing_information_interrupt_node",
]


# ---------------------------------------------------------------------------
# 필수정보 정의
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class InformationField:
    """수집해야 할 정보 1개.

    `keywords` 는 "보호자가 이미 말했는가" 를 규칙으로 판정하기 위한 신호다.
    LLM 없이도 불필요한 재질문을 피하려면 이 판정이 결정론적이어야 한다.
    """

    key: str
    label: str
    question: str
    keywords: tuple[str, ...] = ()
    patterns: tuple[str, ...] = ()
    # 자유 서술만으로 "이미 답변됨" 으로 인정할지.
    #
    # False 인 항목은 반드시 명시적으로 받아야 한다. "활동량이 계속 줄었어요" 한
    # 문장에서 빈도·변화·현재 상태를 모두 읽어냈다고 치면 질문할 것이 없어지는데,
    # 정작 진료 의뢰서에 적을 구체값(하루 몇 회, 언제부터 어떻게 변했는지)은
    # 비어 있게 된다. 키워드 추론은 '무엇이 문제인가' 수준까지만 신뢰한다.
    inferable_from_text: bool = True


UNKNOWN_VALUE = "모름"

# '모름' 계열 답변 — 미응답이 아니라 **유효한 답변**으로 처리한다(명세 29절).
UNKNOWN_ANSWERS: tuple[str, ...] = (
    "모름",
    "모르겠",
    "몰라",
    "모르겠어요",
    "확인 못",
    "확인 불가",
    "확인이 어렵",
    "기억 안",
    "기억이 안",
    "잘 모르",
    "없음",
    "unknown",
    "n/a",
)

# 한 번에 물어보는 최대 질문 수. 8개를 한꺼번에 나열하면 보호자가 답을 포기한다.
MAX_QUESTIONS_PER_TURN = 4


# --- 일반 건강 상담 (명세 29절: "답변에 꼭 필요한 정보만 질문한다") --------------
GENERAL_HEALTH_FIELDS: tuple[InformationField, ...] = (
    InformationField(
        key="main_symptom",
        label="주요 증상",
        question="현재 가장 신경 쓰이는 증상이 무엇인가요?",
        keywords=(
            "토", "구토", "설사", "기침", "먹지", "안 먹", "기운", "가려", "긁",
            "절뚝", "숨", "호흡", "열", "소변", "대변", "변", "아파", "아픈",
            "부었", "떨", "경련", "피",
            # 보호자가 증상을 '상태 변화' 로 서술하는 경우도 답변으로 인정한다.
            # 이미 말한 것을 다시 묻는 것이 대화를 가장 크게 망가뜨린다.
            "식사", "밥", "사료", "식욕", "활동", "산책", "체중", "살이", "물을",
        ),
    ),
    InformationField(
        key="symptom_onset",
        label="증상 시작 시점",
        question="언제부터 그런 모습이 나타났나요? (예: 오늘 아침, 3일 전, 모름)",
        keywords=(
            "오늘", "어제", "그제", "아침", "저녁", "밤", "새벽", "방금", "조금 전",
            "일주일", "며칠", "이틀", "사흘", "달 전", "주 전", "시간 전", "분 전",
            "부터",
        ),
        patterns=(r"\d+\s*(?:일|주|달|개월|시간|분)\s*(?:전|째|동안)",),
    ),
)

# --- 병원 상담 권고 (명세 29절 예시 그대로) --------------------------------------
VISIT_FIELDS: tuple[InformationField, ...] = (
    GENERAL_HEALTH_FIELDS[0],  # 주요 증상
    GENERAL_HEALTH_FIELDS[1],  # 증상 시작 시점
    InformationField(
        key="frequency",
        label="빈도",
        question="그 증상이 하루에 몇 번 정도 나타나나요? (모르면 '모름')",
        keywords=("번", "회", "차례", "자주", "가끔", "계속", "하루에", "매일", "한 번", "여러 번"),
        patterns=(r"\d+\s*(?:번|회|차례)",),
        inferable_from_text=False,
    ),
    InformationField(
        key="symptom_change",
        label="증상 변화",
        question="처음보다 나아지고 있나요, 비슷한가요, 심해지고 있나요?",
        keywords=(
            "심해", "나아", "좋아", "비슷", "그대로", "악화", "호전", "덜해", "더해",
            "변화 없", "점점", "줄었", "줄어", "늘었", "늘어", "나빠", "심각해",
        ),
        inferable_from_text=False,
    ),
    InformationField(
        key="current_intake",
        label="현재 식사·음수·활동 상태",
        question="지금 식사와 물은 평소만큼 하고 있나요? 활동량은 어떤가요?",
        keywords=(
            "먹", "식사", "사료", "밥", "물", "음수", "마시", "산책", "활동", "놀",
            "움직", "누워", "자",
        ),
        inferable_from_text=False,
    ),
)

# --- 응급 연락 최소정보 (명세 29절 예시 그대로 8항목) ------------------------------
EMERGENCY_CONTACT_FIELDS: tuple[InformationField, ...] = (
    InformationField(
        key="worst_symptom",
        label="현재 가장 심한 증상",
        question="지금 가장 심한 증상이 무엇인가요?",
        keywords=(
            "숨", "호흡", "경련", "발작", "출혈", "피", "의식", "쓰러", "토", "구토",
            "부었", "떨", "아파", "못 움직",
        ),
    ),
    InformationField(
        key="onset_time",
        label="증상 시작 시각",
        question="언제부터 시작됐나요? (대략적인 시각이나 '모름' 도 괜찮습니다)",
        keywords=(
            "오늘", "어제", "아침", "저녁", "밤", "새벽", "방금", "조금 전", "부터",
            "시쯤", "시경",
        ),
        patterns=(r"\d+\s*(?:시|시간|분)\s*(?:전|쯤|경|부터)?",),
    ),
    InformationField(
        key="still_ongoing",
        label="현재도 진행 중인지",
        question="지금도 그 증상이 계속되고 있나요?",
        keywords=("계속", "지금도", "아직", "멈췄", "멎었", "끝났", "진행", "반복", "다시"),
    ),
    InformationField(
        key="consciousness",
        label="의식 또는 반응 상태",
        question="불렀을 때 반응이 있나요? 의식은 있어 보이나요?",
        keywords=("의식", "반응", "불러", "눈을", "멍", "깨어", "정신"),
    ),
    InformationField(
        key="breathing",
        label="호흡 상태",
        question="숨은 어떻게 쉬고 있나요? (빠른지, 힘들어하는지, 소리가 나는지)",
        keywords=("숨", "호흡", "헐떡", "가쁘", "그렁", "코골", "혀"),
    ),
    InformationField(
        key="mobility",
        label="움직일 수 있는지",
        question="스스로 일어서거나 걸을 수 있나요?",
        keywords=("일어", "걷", "움직", "서", "누워", "못 일어", "비틀", "절뚝"),
    ),
    InformationField(
        key="trauma_or_toxin",
        label="외상 또는 위험물질 섭취 가능성",
        question="다쳤거나, 삼키면 안 되는 것을 먹었을 가능성이 있나요?",
        keywords=(
            "다쳤", "부딪", "떨어", "차에", "사고", "물렸", "삼켰", "먹었", "이물",
            "약", "초콜릿", "포도", "양파", "세제", "살충", "쥐약", "없었", "없어요",
        ),
    ),
    InformationField(
        key="approximate_count",
        label="대략적인 횟수",
        question="그 증상이 대략 몇 번 있었나요? (모르면 '모름')",
        keywords=("번", "회", "차례", "한 번", "두 번", "여러 번", "계속", "한번"),
        patterns=(r"\d+\s*(?:번|회|차례)",),
    ),
)

_ALL_FIELDS: tuple[InformationField, ...] = (
    GENERAL_HEALTH_FIELDS + VISIT_FIELDS + EMERGENCY_CONTACT_FIELDS
)


def fields_for_risk(risk_level: str | None) -> tuple[InformationField, ...]:
    """위험도별 필수정보 세트를 고른다(명세 29절 — 세 세트는 서로 다르다)."""
    if risk_level == "emergency":
        return EMERGENCY_CONTACT_FIELDS
    if risk_level == "visit":
        return VISIT_FIELDS
    return GENERAL_HEALTH_FIELDS


def field_by_key(key: str) -> InformationField | None:
    for field in _ALL_FIELDS:
        if field.key == key:
            return field
    return None


def field_by_label(label: str) -> InformationField | None:
    for field in _ALL_FIELDS:
        if field.label == label:
            return field
    return None


# ---------------------------------------------------------------------------
# 답변 판정
# ---------------------------------------------------------------------------
def is_unknown_answer(value: Any) -> bool:
    """'모름' 계열 답변인지 판정한다 — 유효한 답변이므로 재질문 대상이 아니다."""
    text = str(value or "").strip().lower()
    if not text:
        return False
    return any(marker in text for marker in UNKNOWN_ANSWERS)


def _compact(text: str) -> str:
    return re.sub(r"\s+", "", text)


def _field_answered_in_text(field: InformationField, raw: str, compact: str) -> bool:
    """자유 서술에서 해당 항목이 이미 언급됐는지 규칙으로 본다."""
    for keyword in field.keywords:
        if keyword in raw or _compact(keyword) in compact:
            return True
    for pattern in field.patterns:
        if re.search(pattern, raw):
            return True
    return False


def _answer_text(state: dict[str, Any]) -> str:
    """판정 대상 텍스트 — 현재 입력과 지금까지 받은 자유 답변을 합친다."""
    parts: list[str] = [str(state.get("user_message") or "")]
    collected = state.get("collected_information") or {}
    if isinstance(collected, dict):
        parts.extend(str(value) for value in collected.values() if value)
    observation = state.get("current_observation") or {}
    if isinstance(observation, dict):
        parts.extend(str(value) for value in observation.values() if value)
    return " ".join(part for part in parts if part.strip())


def detect_answered_fields(
    state: dict[str, Any],
    fields: tuple[InformationField, ...],
) -> tuple[list[InformationField], list[InformationField]]:
    """(이미 답변된 항목, 아직 없는 항목) 으로 나눈다.

    답변으로 인정하는 경우는 세 가지다.
      1. `collected_information[key]` 에 값이 있음 (interrupt 로 명시적으로 받은 답)
      2. 그 값이 '모름' 계열임 — 명세 29절에 따라 유효한 답변이다
      3. 사용자의 자유 서술에 해당 항목의 신호 키워드가 있음
    """
    collected = state.get("collected_information") or {}
    raw = _answer_text(state)
    compact = _compact(raw)

    answered: list[InformationField] = []
    missing: list[InformationField] = []
    for field in fields:
        value = collected.get(field.key) if isinstance(collected, dict) else None
        if str(value or "").strip():
            answered.append(field)
            continue
        if field.inferable_from_text and _field_answered_in_text(field, raw, compact):
            answered.append(field)
            continue
        missing.append(field)
    return answered, missing


def build_question(missing: list[InformationField]) -> str:
    """부족한 항목을 하나의 한국어 질문으로 만든다.

    한 번에 최대 `MAX_QUESTIONS_PER_TURN` 개만 묻는다. 나머지는 다음 turn 에
    다시 계산되므로 정보가 유실되지 않는다. '모름' 이 유효하다는 안내를 반드시
    붙인다 — 이것이 없으면 보호자가 답을 못 해 대화가 멈춘다.
    """
    if not missing:
        return ""
    asked = missing[:MAX_QUESTIONS_PER_TURN]
    lines = [f"{index}. {field.question}" for index, field in enumerate(asked, start=1)]
    remaining = len(missing) - len(asked)
    tail = (
        f"\n(남은 확인 항목 {remaining}개는 답변해 주시면 이어서 여쭤볼게요.)"
        if remaining > 0
        else ""
    )
    return (
        "정확히 안내드리기 위해 몇 가지만 확인할게요. "
        f"모르시는 항목은 '{UNKNOWN_VALUE}' 이라고 답해 주셔도 괜찮습니다.\n"
        + "\n".join(lines)
        + tail
    )


def evaluate_missing_information(state: dict[str, Any]) -> dict[str, Any]:
    """필수정보 충족 여부를 계산한다(순수 함수 — 단독 테스트 가능).

    반환 dict 는 node 가 그대로 state 로 내보낼 수 있는 형태다.
    `missing_fields` 를 `Replace` 로 감싸는 이유: state.py 의 reducer 가 누적형이라
    감싸지 않으면 **이미 답변된 항목이 목록에서 사라지지 않는다.**
    """
    risk_level = str(state.get("final_risk") or "normal")
    urgency = str(state.get("emergency_urgency") or "none")
    fields = fields_for_risk(risk_level)

    _, missing = detect_answered_fields(state, fields)

    critical = urgency == "critical_immediate"
    ready = not missing or critical

    question = "" if ready else build_question(missing)
    if critical and missing:
        logger.info(
            "critical_immediate 이므로 부족한 정보(%d건)를 기록만 하고 진행합니다.",
            len(missing),
        )

    return {
        "required_fields": [field.label for field in fields],
        "missing_fields": Replace([field.label for field in missing]),
        "minimum_information_ready": ready,
        "missing_information_question": question,
    }


# ---------------------------------------------------------------------------
# Node — Missing Information Agent (일반 상담 / 병원 권고)
# ---------------------------------------------------------------------------
def missing_information_node(state: dict) -> dict:
    """현재 위험도에 맞는 필수정보 세트로 부족한 항목을 계산한다.

    LLM 을 쓰지 않는다. 어떤 항목이 필요한지는 상수 테이블이 정하고, 답변 여부는
    키워드 규칙이 정한다. 이 판정이 흔들리면 같은 질문을 반복하거나(무한 interrupt)
    필요한 정보 없이 PDF 를 만들게 된다.
    """
    result = evaluate_missing_information(state)
    logger.debug(
        "missing information: risk=%s missing=%s ready=%s",
        state.get("final_risk"),
        list(result["missing_fields"]),
        result["minimum_information_ready"],
    )
    return result


# ---------------------------------------------------------------------------
# Node — Contact Minimum Information Agent (응급 / 명세 32절)
# ---------------------------------------------------------------------------
def contact_minimum_information_node(state: dict) -> dict:
    """병원 전화 통화에 필요한 최소정보를 정리한다(응급 전용).

    명세 29·32절의 핵심 안전 규칙을 여기에 구현한다.
    **`critical_immediate` 이면 부족한 항목이 있어도 통과시킨다.** 숨을 못 쉬는
    상황에서 "증상이 하루 몇 번인가요" 를 먼저 묻는 것은 해가 된다. 부족한 항목은
    `missing_fields` 에 남아 PDF 의 `unknown_fields` 로 이어지며, 추측해 채우지 않는다.
    """
    fields = EMERGENCY_CONTACT_FIELDS
    _, missing = detect_answered_fields(state, fields)
    critical = str(state.get("emergency_urgency") or "none") == "critical_immediate"

    ready = not missing or critical
    question = "" if ready else build_question(missing)

    reasons: list[str] = []
    if critical and missing:
        reasons.append(
            "[정보수집] 즉시 위급 상황이라 정보 수집을 기다리지 않고 병원 연락을 우선합니다."
        )

    result: dict[str, Any] = {
        "required_fields": [field.label for field in fields],
        "missing_fields": Replace([field.label for field in missing]),
        "minimum_information_ready": ready,
        "missing_information_question": question,
    }
    if reasons:
        result["risk_reasons"] = reasons
    return result


# ---------------------------------------------------------------------------
# Node — Interrupt (명세 29절 multi-turn)
# ---------------------------------------------------------------------------
def missing_information_interrupt_node(state: dict) -> dict:
    """부족한 정보를 사용자에게 되묻고 답변을 State 에 반영한다.

    `langgraph.types.interrupt()` 는 graph 실행을 멈추고 payload 를 호출자에게
    돌려준다. 호출자가 `Command(resume=...)` 로 같은 `thread_id` 를 재개하면
    `interrupt()` 가 그 값을 반환하며 이 함수가 이어서 실행된다(명세 29절).

    **interrupt 를 걸지 않는 두 경우:**
      - 이미 필요한 정보가 다 모였을 때
      - `critical_immediate` 일 때 — 정보 수집이 전화 action 을 막으면 안 된다

    resume 값은 두 형태를 모두 받는다.
      - dict : {"symptom_onset": "어제 밤"} 처럼 항목별 답변
      - str  : 자유 서술. 항목을 추정하지 않고 원문을 보존한 뒤 키워드 규칙이
               다음 turn 에 다시 판정한다(임의로 항목에 배정하면 오답이 굳는다).
    """
    if str(state.get("emergency_urgency") or "none") == "critical_immediate":
        logger.info("critical_immediate — interrupt 없이 진행합니다(전화 action 우선).")
        return {"minimum_information_ready": True}

    missing_labels = [str(label) for label in (state.get("missing_fields") or [])]
    if not missing_labels:
        return {"minimum_information_ready": True}

    question = str(state.get("missing_information_question") or "")
    if not question:
        fields = [field for field in (field_by_label(label) for label in missing_labels) if field]
        question = build_question(fields)

    from langgraph.types import interrupt  # 지연 import — 모듈 최상단 무게를 줄인다

    answer = interrupt(
        {
            "type": "missing_information",
            "question": question,
            "required_fields": list(state.get("required_fields") or []),
            "missing_fields": missing_labels,
            "risk_level": state.get("final_risk", "normal"),
            "emergency_urgency": state.get("emergency_urgency", "none"),
            # 호출자(Android / Colab)에게 '모름' 이 허용된다는 계약을 명시한다.
            "allow_unknown": True,
            "unknown_value": UNKNOWN_VALUE,
        }
    )

    # 되물은 횟수를 하나 올린다.
    #
    # `routers.MAX_MISSING_INFORMATION_ROUNDS` 는 "이만큼 물었으면 남은 항목은
    # '모름' 으로 두고 진행한다" 는 탈출구다. 그런데 그 값을 **올리는 코드가 어디에도
    # 없어서** 카운터가 항상 0 이었고, 탈출구가 한 번도 발동하지 않았다. 키워드 규칙이
    # 인정하지 못하는 항목(빈도·증상 변화 등)은 보호자가 답해도 같은 질문이 무한
    # 반복됐다 — 실제로 그랬다.
    #
    # 여기가 맞는 자리다: interrupt 가 답을 받아 돌아온 시점이 곧 한 라운드의 끝이다.
    # `collected_information` 은 얕은 병합 reducer 라 이 키만 돌려주면 누적된다.
    from ..routers import MISSING_INFO_ROUNDS_KEY, missing_information_rounds  # noqa: PLC0415

    update = normalize_resume_answer(answer, missing_labels)
    update[MISSING_INFO_ROUNDS_KEY] = missing_information_rounds(state) + 1
    return {"collected_information": update}


def normalize_resume_answer(answer: Any, missing_labels: list[str]) -> dict[str, Any]:
    """`Command(resume=...)` 로 들어온 답변을 `collected_information` 형태로 만든다.

    dict 는 label 로 와도 key 로 와도 받아준다(호출자가 사람이 읽는 label 을 쓰기
    쉽다). 문자열 자유 서술은 어느 항목의 답인지 단정하지 않고 통째로 보관한다 —
    잘못 배정하면 그 오답이 PDF 와 병원 안내까지 그대로 흘러간다.

    `collected_information` 은 state.py 에서 얕은 병합 reducer 를 쓰므로 **새로 받은
    값만** 반환하면 이전 turn 의 답변이 유지된다.
    """
    if answer is None:
        return {}

    if isinstance(answer, dict):
        normalized: dict[str, Any] = {}
        for raw_key, value in answer.items():
            key = str(raw_key)
            field = field_by_key(key) or field_by_label(key)
            normalized[field.key if field else key] = value
        return normalized

    text = str(answer).strip()
    if not text:
        return {}

    # 자유 서술 1건이고 물어본 항목도 1개면 그 항목의 답으로 본다(모호함이 없다).
    if len(missing_labels) == 1:
        field = field_by_label(missing_labels[0])
        if field is not None:
            return {field.key: text}

    return {"free_text_answer": text}
