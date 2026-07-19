"""위험도 평가 — Rule Assessment Node 와 Assessment Agent (명세 23·28절).

명세 23절은 "모든 항목을 자유 행동형 agent 로 만들 필요는 없다. 규칙 계산은 일반
Python node 로 구현한다" 고 못 박았다. 그래서 이 파일의 **backbone 은 규칙 엔진**
(`evaluate_rules`)이고, LLM 은 그 결과를 못 낮추는 보조 평가자일 뿐이다.

두 node 는 명세 24절 graph 에서 `Clinical Context` 뒤에 **병렬로** 실행된다.
따라서 두 node 가 함께 쓰는 `red_flags` / `risk_reasons` / `emergency_urgency` 는
state.py 의 누적·상향 reducer 에 의존한다(같은 super-step 동시 쓰기 허용).
서로 다른 key 인 `rule_risk` / `assessment_risk` 에는 각자의 원본 판정을 남겨
LangSmith trace 에서 "누가 무엇을 올렸는지" 를 볼 수 있게 한다.

설계 원칙
  - 규칙은 **데이터(상수 테이블)** 다. if 문에 흩어 두면 테스트도 조정도 불가능하다.
  - 증상 1건이 곧 병원 방문은 아니다. 지속성(`trend`)과 완화 신호를 함께 본다.
  - 확정 진단·처방 문구는 만들지 않는다. red flag 는 '관찰된 신호' 이지 진단명이 아니다.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from ...llm import build_llm, safe_structured_invoke
from ...schemas import AssessmentResult, EmergencyUrgency, RiskLevel, merge_risk
from ..prompts import ASSESSMENT_PROMPT, wrap_untrusted_block
from ..state import URGENCY_PRIORITY

if TYPE_CHECKING:  # state.py 와의 순환 참조를 만들지 않기 위해 타입만 가져온다.
    from ..state import PetCareState

logger = logging.getLogger(__name__)

__all__ = [
    "RiskRule",
    "RED_FLAG_RULES",
    "DURATION_PATTERNS",
    "MITIGATING_PATTERNS",
    "DIARY_LOW_FOOD_SIGNALS",
    "DIARY_LOW_ACTIVITY_SIGNALS",
    "DIARY_TREND_WINDOW",
    "red_flag_urgency",
    "collect_risk_signals",
    "diary_trend_signals",
    "evaluate_rules",
    "build_assessment_prompt",
    "rule_assessment_node",
    "assessment_agent_node",
]


# ---------------------------------------------------------------------------
# 규칙 테이블
# ---------------------------------------------------------------------------
Escalation = Literal["single", "trend"]


@dataclass(frozen=True)
class RiskRule:
    """red flag 규칙 1건.

    `escalation` 의 의미:
      - "single": 한 번만 관찰돼도 병원 진료를 권할 만한 신호(식욕 폐절, 절뚝임 등)
      - "trend" : 1회성으로는 흔하고 **지속될 때** 의미가 생기는 신호(구토·기침·설사)

    이 구분이 없으면 "오늘 한 번 토했는데 지금은 잘 놀아요"(명세 43절 normal 기대
    케이스)가 병원 권고로 잘못 올라간다.
    """

    code: str
    label: str
    keywords: tuple[str, ...]
    risk_level: RiskLevel
    urgency: EmergencyUrgency = "none"
    escalation: Escalation = "single"
    # 사이에 다른 단어가 끼는 표현을 잡기 위한 정규식.
    # 예) "활동량이 **계속** 줄었어요" 는 '활동량이 줄' 키워드로는 잡히지 않는다.
    # 오탐을 막기 위해 사이 간격은 좁게(6자 이내) 제한한다.
    patterns: tuple[str, ...] = ()


# 응급 — 즉시 이동이 필요하며 정보 수집이 지연 사유가 되면 안 되는 신호.
# (명세 29절: critical_immediate 에서는 정보 수집이 전화 action 을 막지 않는다)
_CRITICAL_RULES: tuple[RiskRule, ...] = (
    RiskRule(
        "breathing_failure",
        "호흡곤란 또는 호흡 정지 의심",
        (
            "숨을 거의 쉬지",
            "숨을 못 쉬",
            "숨을 안 쉬",
            "숨을 쉬지 못",
            "호흡이 멈",
            "호흡을 안",
            "호흡곤란",
            "숨이 넘어",
        ),
        "emergency",
        "critical_immediate",
    ),
    RiskRule(
        "unconscious",
        "의식 소실 또는 무반응",
        (
            "의식이 없",
            "의식 없",
            "의식을 잃",
            "깨어나지 않",
            "깨어나지 못",
            "반응이 전혀 없",
            "불러도 반응이 없",
            "축 늘어져 반응",
        ),
        "emergency",
        "critical_immediate",
    ),
    RiskRule(
        "cyanosis",
        "잇몸·혀 청색증 또는 창백",
        ("혀가 파랗", "잇몸이 파랗", "혀가 보라", "청색증", "잇몸이 하얗", "잇몸이 창백"),
        "emergency",
        "critical_immediate",
    ),
    RiskRule(
        "uncontrolled_bleeding",
        "지혈되지 않는 출혈",
        ("피가 멈추지", "출혈이 멈추지", "대량 출혈", "피가 계속 나"),
        "emergency",
        "critical_immediate",
    ),
    RiskRule(
        "heatstroke",
        "열사병 의심",
        ("열사병", "차 안에 갇", "더위에 쓰러"),
        "emergency",
        "critical_immediate",
    ),
)

# 응급 — 즉시 병원 연락이 필요하지만 최소 정보를 물을 여유는 있는 신호.
_EMERGENCY_RULES: tuple[RiskRule, ...] = (
    RiskRule(
        "seizure",
        "경련 또는 발작",
        ("경련", "발작", "몸을 떨며 쓰러", "사지를 뻗고 떨"),
        "emergency",
        "contact_ready",
    ),
    RiskRule(
        "collapse",
        "허탈·기립 불능",
        ("쓰러졌", "쓰러져", "일어서지 못", "일어나지 못", "주저앉"),
        "emergency",
        "contact_ready",
    ),
    RiskRule(
        "altered_consciousness",
        "반응 저하·의식 수준 변화",
        ("반응이 둔", "반응이 느려", "멍하니", "정신이 없", "비틀거"),
        "emergency",
        "contact_ready",
    ),
    RiskRule(
        "toxin_ingestion",
        "중독 물질 섭취 가능성",
        (
            "초콜릿",
            "포도",
            "건포도",
            "양파",
            "마늘",
            "자일리톨",
            "살충제",
            "쥐약",
            "부동액",
            "농약",
            "세제를 먹",
            "사람 약을 먹",
            "약을 삼",
            "독",
        ),
        "emergency",
        "contact_ready",
    ),
    RiskRule(
        "foreign_body",
        "이물 섭취 가능성",
        ("이물", "장난감을 삼", "동전을 삼", "뼈를 삼", "실을 삼", "비닐을 먹"),
        "emergency",
        "contact_ready",
    ),
    RiskRule(
        "urinary_obstruction",
        "배뇨 곤란·요도 폐색 의심",
        ("소변을 못", "오줌을 못", "소변이 안 나", "화장실에서 힘", "배뇨 곤란", "소변을 못 봐"),
        "emergency",
        "contact_ready",
    ),
    RiskRule(
        "bloat",
        "복부 팽만·헛구역질(위확장 염전 의심)",
        ("배가 부풀", "복부 팽만", "배가 딱딱", "헛구역질", "토하려는데 안 나"),
        "emergency",
        "contact_ready",
    ),
    RiskRule(
        "trauma",
        "외상·사고",
        ("차에 치", "교통사고", "높은 곳에서 떨어", "떨어졌", "물렸", "밟혔"),
        "emergency",
        "contact_ready",
    ),
    RiskRule(
        "blood_in_excreta",
        "혈변·혈뇨·토혈",
        ("혈변", "혈뇨", "피 섞인", "피가 섞인", "피를 토", "검은 변"),
        "emergency",
        "contact_ready",
    ),
    RiskRule(
        "repeated_vomiting",
        "반복 구토",
        ("계속 토", "여러 번 토", "반복해서 토", "하루에 여러 번 토", "구토를 반복"),
        "emergency",
        "contact_ready",
    ),
    RiskRule(
        "labored_breathing",
        "호흡수 증가·호흡 노력 증가",
        ("숨이 가쁘", "호흡이 빠", "숨을 헐떡", "혀를 내밀고 헐떡", "배로 숨"),
        "emergency",
        "contact_ready",
    ),
)

# 병원 진료 권고 — 응급은 아니지만 수의사 확인이 필요한 신호.
_VISIT_RULES: tuple[RiskRule, ...] = (
    RiskRule(
        "anorexia",
        "식욕 저하 또는 식사 거부",
        ("먹지 않", "안 먹", "못 먹", "식사를 거의", "밥을 거부", "식욕이 없", "입맛이 없"),
        "visit",
        patterns=(
            r"(?:식사|밥|사료|식욕)[을를이가은는]?[^.]{0,6}(?:거의|안|못)\s*(?:먹|하)",
            r"(?:식사|밥|사료|식욕)[을를이가은는]?[^.]{0,6}(?:줄|감소|거부)",
        ),
    ),
    RiskRule(
        "lethargy",
        "기력 저하·활동량 감소",
        ("기운이 없", "기력이 없", "활동량이 줄", "축 처져", "무기력", "잠만 자", "놀지 않"),
        "visit",
        patterns=(
            r"활동(?:량)?[을를이가은는]?[^.]{0,6}(?:줄|감소|적어|덜)",
            r"(?:산책|놀이)[을를]?[^.]{0,6}(?:거부|안\s*하|싫어)",
            r"(?:기운|기력)[이가은는]?[^.]{0,6}(?:없|떨어|줄)",
        ),
    ),
    RiskRule(
        "weight_loss",
        "체중 감소",
        ("살이 빠", "체중이 줄", "마른 느낌", "야위"),
        "visit",
    ),
    RiskRule(
        "lameness",
        "파행·다리 통증",
        ("절뚝", "다리를 들고", "다리를 절", "걷기 힘들"),
        "visit",
    ),
    RiskRule(
        "fever",
        "발열 의심",
        ("열이 나", "몸이 뜨겁", "고열"),
        "visit",
    ),
    RiskRule(
        "polydipsia",
        "음수·배뇨량 변화",
        ("물을 너무 많이", "물을 많이 마", "소변량이 늘", "소변을 자주"),
        "visit",
    ),
    RiskRule(
        "vomiting",
        "구토",
        ("토했", "구토", "토함", "게워"),
        "visit",
        escalation="trend",
    ),
    RiskRule(
        "diarrhea",
        "설사·묽은 변",
        ("설사", "묽은 변", "변이 무름", "물똥"),
        "visit",
        escalation="trend",
    ),
    RiskRule(
        "cough",
        "기침",
        ("기침", "켁켁", "캑캑"),
        "visit",
        escalation="trend",
    ),
    RiskRule(
        "skin_problem",
        "피부·가려움 이상",
        ("계속 긁", "털이 빠", "발진", "피부가 빨"),
        "visit",
        escalation="trend",
    ),
    RiskRule(
        "eye_ear",
        "눈·귀 이상",
        ("눈이 빨", "눈곱", "눈을 못 뜨", "귀를 털", "귀에서 냄새"),
        "visit",
        escalation="trend",
    ),
)

RED_FLAG_RULES: tuple[RiskRule, ...] = _CRITICAL_RULES + _EMERGENCY_RULES + _VISIT_RULES

# 지속성 신호 — 'trend' 규칙을 병원 권고로 올리는 근거.
DURATION_PATTERNS: tuple[str, ...] = (
    "며칠째",
    "몇 일째",
    "몇일째",
    "이틀째",
    "사흘째",
    "이틀 동안",
    "일주일",
    "한 달",
    "계속",
    "지속",
    "자꾸",
    "반복",
    "여러 날",
    "여러 번",
    "매일",
    "오래",
    "낫지 않",
    "점점",
    "심해지",
)

# 완화 신호 — 단일 경증 신호를 normal 로 되돌릴 수 있는 표현.
# **응급 신호에는 절대 적용하지 않는다**(명세 47절: 낮은 위험도로 덮어쓰지 않는다).
MITIGATING_PATTERNS: tuple[str, ...] = (
    "지금은 잘",
    "지금은 괜찮",
    "지금은 멀쩡",
    "이후에는 괜찮",
    "다시 잘 먹",
    "평소처럼 잘",
    "평소와 같이",
    "잘 놀아",
    "잘 놀고",
    "활발해",
    "한 번만",
    "한번만",
    "금방 괜찮",
    "바로 회복",
)

# 일기장 추세 판정에 쓰는 신호(fixture·서버 payload 의 food/activity 자유서술 대응).
DIARY_LOW_FOOD_SIGNALS: tuple[str, ...] = (
    "절반",
    "1/3",
    "1/2",
    "조금",
    "거의 먹지",
    "안 먹",
    "먹지 않",
    "남김",
    "줄",
    "거부",
)
DIARY_LOW_ACTIVITY_SIGNALS: tuple[str, ...] = (
    "거부",
    "누워",
    "지침",
    "지쳐",
    "줄",
    "짧게",
    "멈춰",
    "쉬기만",
    "움직이지",
)
# 최근 며칠을 추세로 볼지. 3일이면 "일시적 컨디션" 과 "추세" 를 가르는 최소 창이다.
DIARY_TREND_WINDOW: int = 3
# 창 안에서 몇 건 이상이어야 추세로 인정할지(2/3 이상).
_DIARY_TREND_MIN_HITS: int = 2


def red_flag_urgency(label: str) -> EmergencyUrgency:
    """red flag 표기 문자열에서 응급 긴급도를 되찾는다.

    `merge_risk_node` 는 병렬 node 들이 누적한 `red_flags`(문자열 리스트)만 보고
    최종 긴급도를 다시 계산해야 한다. 규칙 테이블을 유일한 출처로 삼기 위해
    label → urgency 역참조를 여기서 제공한다.
    """
    for rule in RED_FLAG_RULES:
        if rule.label == label:
            return rule.urgency
    return "none"


# ---------------------------------------------------------------------------
# 텍스트 매칭 유틸
# ---------------------------------------------------------------------------
def _compact(text: str) -> str:
    """공백을 모두 제거한다 — "숨을못쉬어요" 같은 입력도 잡기 위해서다."""
    return re.sub(r"\s+", "", text)


def _contains(raw: str, compact: str, keyword: str) -> bool:
    """원문과 공백 제거본 양쪽에서 키워드를 찾는다."""
    return keyword in raw or _compact(keyword) in compact


def _any_contains(raw: str, compact: str, keywords: tuple[str, ...]) -> bool:
    return any(_contains(raw, compact, keyword) for keyword in keywords)


def _rule_matches(rule: RiskRule, raw: str, compact: str) -> bool:
    """키워드 또는 정규식 중 하나라도 걸리면 그 규칙이 매칭된 것으로 본다."""
    if _any_contains(raw, compact, rule.keywords):
        return True
    return any(re.search(pattern, raw) for pattern in rule.patterns)


def _state_text(state: dict[str, Any]) -> str:
    """규칙 매칭 대상 텍스트를 만든다.

    포함: 현재 사용자 입력(최우선), interrupt 로 수집한 추가 답변, 구조화된 현재 증상.
    **제외: 일기장·진단서 원문.** 일기장의 "며칠째" 같은 표현이 사용자의 현재 문장에
    섞이면 오늘 처음 생긴 증상을 만성으로 오판한다. 일기장은 `diary_trend_signals`
    가 별도로, 추세로만 본다(명세 20절 우선순위: 사용자 현재 입력 > DB).
    """
    parts: list[str] = [str(state.get("user_message") or "")]

    collected = state.get("collected_information") or {}
    if isinstance(collected, dict):
        parts.extend(str(value) for value in collected.values() if value)

    observation = state.get("current_observation") or {}
    if isinstance(observation, dict):
        parts.extend(str(value) for value in observation.values() if value)

    return " ".join(part for part in parts if part.strip())


def collect_risk_signals(text: str) -> list[RiskRule]:
    """문장에서 매칭되는 red flag 규칙을 모은다(테이블 순서 유지).

    같은 code 는 한 번만 담는다. 규칙 테이블이 유일한 판단 근거이므로 이 함수만
    테스트하면 위험도 분류 전체를 검증할 수 있다.
    """
    raw = text or ""
    compact = _compact(raw)
    matched: list[RiskRule] = []
    seen: set[str] = set()
    for rule in RED_FLAG_RULES:
        if rule.code in seen:
            continue
        if _rule_matches(rule, raw, compact):
            seen.add(rule.code)
            matched.append(rule)
    return matched


def has_duration_signal(text: str) -> bool:
    """증상이 지속·반복되고 있다는 표현이 있는지."""
    raw = text or ""
    return _any_contains(raw, _compact(raw), DURATION_PATTERNS)


def has_mitigating_signal(text: str) -> bool:
    """현재는 회복됐다는 표현이 있는지(응급 신호에는 적용하지 않는다)."""
    raw = text or ""
    return _any_contains(raw, _compact(raw), MITIGATING_PATTERNS)


# ---------------------------------------------------------------------------
# 일기장 추세 (명세 20절: 일기장은 보조자료 — 추세 판단에만 쓴다)
# ---------------------------------------------------------------------------
def _entry_text(entry: dict[str, Any], *fields: str) -> str:
    return " ".join(str(entry.get(field) or "") for field in fields)


def diary_trend_signals(
    entries: list[dict[str, Any]] | None,
    window: int = DIARY_TREND_WINDOW,
) -> list[tuple[str, str]]:
    """최근 일기에서 '추세' 를 뽑는다. 반환은 (red_flag 표기, 근거 문장) 목록.

    일기장 1건은 근거가 되지 못한다. `window` 일 중 `_DIARY_TREND_MIN_HITS` 건 이상
    같은 방향의 변화가 있을 때만 신호로 인정한다. 이것이 "어제 좀 덜 먹었다" 와
    "며칠째 식사량이 줄고 있다" 를 가르는 기준이다.

    entries 는 record_date 오름차순(마지막이 최신)을 가정한다(adapter 계약).
    """
    if not entries:
        return []

    recent = [entry for entry in entries[-window:] if isinstance(entry, dict)]
    if len(recent) < _DIARY_TREND_MIN_HITS:
        return []

    signals: list[tuple[str, str]] = []

    low_food = [
        entry
        for entry in recent
        if _any_contains(
            _entry_text(entry, "food", "raw_text"),
            _compact(_entry_text(entry, "food", "raw_text")),
            DIARY_LOW_FOOD_SIGNALS,
        )
    ]
    if len(low_food) >= _DIARY_TREND_MIN_HITS:
        signals.append(
            (
                "일기장 기준 최근 식사량 감소 추세",
                f"최근 {len(recent)}일 중 {len(low_food)}일에 식사량 감소 기록",
            )
        )

    low_activity = [
        entry
        for entry in recent
        if _any_contains(
            _entry_text(entry, "activity", "raw_text"),
            _compact(_entry_text(entry, "activity", "raw_text")),
            DIARY_LOW_ACTIVITY_SIGNALS,
        )
    ]
    if len(low_activity) >= _DIARY_TREND_MIN_HITS:
        signals.append(
            (
                "일기장 기준 최근 활동량 감소 추세",
                f"최근 {len(recent)}일 중 {len(low_activity)}일에 활동량 감소 기록",
            )
        )

    vomiting = [
        entry
        for entry in recent
        if str(entry.get("vomit") or "").strip() not in ("", "없음", "-", "0")
    ]
    if len(vomiting) >= _DIARY_TREND_MIN_HITS:
        signals.append(
            (
                "일기장 기준 구토 반복",
                f"최근 {len(recent)}일 중 {len(vomiting)}일에 구토 기록",
            )
        )

    symptomatic = [entry for entry in recent if str(entry.get("symptom") or "").strip()]
    if len(symptomatic) >= len(recent) and len(recent) >= _DIARY_TREND_MIN_HITS:
        signals.append(
            (
                "일기장 기준 증상 지속",
                f"최근 {len(recent)}일 연속으로 증상이 기록됨",
            )
        )

    return signals


# ---------------------------------------------------------------------------
# 기저질환 가중 (PET DB / 진단서)
# ---------------------------------------------------------------------------
# 해당 기저질환이 있으면 관련 증상을 한 단계 무겁게 본다. 진단명을 새로 만들지 않고
# **관찰 신호의 우선순위만 올린다**(확정 진단 생성 금지).
_CHRONIC_ESCALATION: tuple[tuple[tuple[str, ...], tuple[str, ...], str], ...] = (
    (
        ("심장", "심부전", "판막", "mitral", "cardiac", "heart"),
        ("cough", "labored_breathing", "lethargy", "collapse"),
        "심장 기저질환이 기록되어 있어 호흡기·활동 관련 신호를 더 무겁게 판단함",
    ),
    (
        ("신장", "신부전", "kidney", "renal"),
        ("anorexia", "vomiting", "polydipsia", "weight_loss"),
        "신장 기저질환이 기록되어 있어 식이·음수 관련 신호를 더 무겁게 판단함",
    ),
    (
        ("당뇨", "diabet"),
        ("polydipsia", "anorexia", "lethargy", "weight_loss"),
        "당뇨 기저질환이 기록되어 있어 음수·식이 관련 신호를 더 무겁게 판단함",
    ),
)


def _chronic_context_text(state: dict[str, Any]) -> str:
    """PET DB 의 기존 질병 + 관련 진단서 진단명을 합친다(원문 요약은 하지 않는다)."""
    profile = state.get("priority_pet_context") or state.get("pet_profile") or {}
    parts: list[str] = []
    if isinstance(profile, dict):
        parts.append(str(profile.get("diseases") or ""))
        parts.append(str(profile.get("medications") or ""))
    for diagnosis in state.get("related_diagnoses") or []:
        if isinstance(diagnosis, dict):
            parts.append(str(diagnosis.get("diagnosis") or ""))
    return " ".join(part for part in parts if part.strip()).lower()


def _chronic_escalations(
    state: dict[str, Any], matched_codes: set[str]
) -> list[str]:
    """기저질환과 현재 신호가 겹치는지 확인하고 사유 문장을 돌려준다."""
    context = _chronic_context_text(state)
    if not context:
        return []
    reasons: list[str] = []
    for disease_keywords, symptom_codes, reason in _CHRONIC_ESCALATION:
        if not any(keyword in context for keyword in disease_keywords):
            continue
        if matched_codes & set(symptom_codes):
            reasons.append(reason)
    return reasons


# ---------------------------------------------------------------------------
# 규칙 평가 (순수 함수)
# ---------------------------------------------------------------------------
def evaluate_rules(state: dict[str, Any]) -> AssessmentResult:
    """LLM 없이 위험도를 계산한다 — 이 프로젝트의 기본 경로다.

    판정 순서(위험한 쪽부터):
      1. 응급 규칙이 하나라도 걸리면 emergency (완화 표현으로 내리지 않는다)
      2. visit 규칙:
         - 'single' 규칙이 걸리거나
         - 'trend' 규칙 + 지속 표현이 있거나
         - 서로 다른 신호가 2건 이상이면 visit
      3. 일기장 추세 신호가 있으면 visit
      4. 기저질환이 관련 신호를 가중하면 visit 로 올린다
      5. 그 외 normal

    `rag_required` 는 "수의학 지식 검색이 필요한가" 이며, 신호가 하나라도 있으면 True 다.
    """
    text = _state_text(state)
    matched = collect_risk_signals(text)
    matched_codes = {rule.code for rule in matched}

    red_flags: list[str] = [rule.label for rule in matched]
    reasons: list[str] = []

    emergency_rules = [rule for rule in matched if rule.risk_level == "emergency"]
    visit_rules = [rule for rule in matched if rule.risk_level == "visit"]

    duration = has_duration_signal(text)
    mitigated = has_mitigating_signal(text)

    trend_signals = diary_trend_signals(
        state.get("supporting_daily_entries") or state.get("daily_entries")
    )
    for label, reason in trend_signals:
        red_flags.append(label)
        reasons.append(reason)

    risk: RiskLevel = "normal"
    urgency: EmergencyUrgency = "none"

    if emergency_rules:
        risk = "emergency"
        for rule in emergency_rules:
            if URGENCY_PRIORITY[rule.urgency] > URGENCY_PRIORITY[urgency]:
                urgency = rule.urgency
            reasons.append(f"응급 신호 감지: {rule.label}")
        # 응급에는 최소한 '연락 준비' 긴급도를 보장한다.
        if urgency == "none":
            urgency = "contact_ready"
    else:
        significant = [rule for rule in visit_rules if rule.escalation == "single"]
        trend_only = [rule for rule in visit_rules if rule.escalation == "trend"]

        if significant or (trend_only and duration) or len(visit_rules) >= 2:
            risk = "visit"
            for rule in visit_rules:
                reasons.append(f"진료 권고 신호: {rule.label}")
            if duration:
                reasons.append("증상이 지속·반복되고 있다는 표현이 있음")
        elif trend_only:
            reasons.append(
                f"관찰된 신호: {', '.join(rule.label for rule in trend_only)} "
                "(1회성으로 판단되어 경과 관찰 범위)"
            )

        # 완화 표현은 **경증 신호가 1건일 때만** 적용한다.
        if risk == "visit" and mitigated and len(visit_rules) <= 1 and not duration:
            risk = "normal"
            reasons.append("현재는 회복되었다는 보호자 관찰이 있어 경과 관찰로 판단함")

        # 일기 추세는 **현재 입력에 신호가 있을 때만** 위험도를 올린다.
        #
        # 명세 20절 우선순위는 '현재 사용자 입력 > PET DB > 진단서 > 일기장' 이고,
        # 47절 금지사항은 '사용자 현재 입력보다 오래된 DB 값을 우선하지 말 것' 이다.
        # 조건에서 `matched` 를 빼면 그 둘을 동시에 어긴다 — 보호자가 증상을 전혀
        # 말하지 않았는데 며칠 전 일기만으로 '진료 상담 권고' 가 나가고, 그 순간
        # visit 서브그래프로 확정되어 되돌릴 수 없다(merge_risk 는 상향 전용이다).
        #
        # 추세 자체는 버리지 않는다. 위험도만 올리지 않고 red_flags·reasons 에는
        # 그대로 남아 답변과 병원 문서에 실린다.
        if trend_signals and risk == "normal":
            if matched:
                risk = "visit"
                reasons.append("일기장에서 확인된 변화 추세가 있어 진료 상담을 권고함")
            else:
                reasons.append(
                    "일기장에서 변화 추세가 확인되었습니다. "
                    "지금 관찰되는 증상이 있으면 알려 주세요."
                )

        chronic_reasons = _chronic_escalations(state, matched_codes)
        if chronic_reasons and (matched or trend_signals):
            risk = merge_risk(risk, "visit")
            reasons.extend(chronic_reasons)

    if not red_flags and risk == "normal":
        reasons.append("현재 입력에서 응급·진료 권고 신호가 확인되지 않음")

    return AssessmentResult(
        risk_level=risk,
        emergency_urgency=urgency,
        red_flags=red_flags,
        reasons=reasons,
        missing_information=[],  # 필수 정보 산정은 missing_information node 담당
        rag_required=bool(matched or trend_signals or risk != "normal"),
    )


# ---------------------------------------------------------------------------
# Node 1 — Rule Assessment (LLM 없음)
# ---------------------------------------------------------------------------
def rule_assessment_node(state: dict) -> dict:
    """규칙만으로 위험도를 계산한다(명세 23절 '일반 Python node').

    이 node 는 절대 실패하지 않아야 한다. LLM·네트워크·RAG 어느 것에도 의존하지
    않으므로 키가 없는 환경에서도 graph 의 위험도 분기가 항상 동작한다.
    """
    result = evaluate_rules(state)
    logger.debug(
        "rule assessment: risk=%s urgency=%s flags=%s",
        result.risk_level,
        result.emergency_urgency,
        result.red_flags,
    )
    return {
        "rule_risk": result.risk_level,
        "emergency_urgency": result.emergency_urgency,
        "red_flags": result.red_flags,
        "risk_reasons": [f"[규칙] {reason}" for reason in result.reasons],
    }


# ---------------------------------------------------------------------------
# Node 2 — Assessment Agent (LLM, structured output)
# ---------------------------------------------------------------------------
def _brief_diagnoses(state: dict[str, Any], limit: int = 3) -> str:
    """진단서를 prompt 용으로 축약한다(명세 21절: 필요한 데이터만 전달)."""
    items = [d for d in (state.get("related_diagnoses") or []) if isinstance(d, dict)]
    if not items:
        return "없음"
    lines = [
        f"- {item.get('date', '날짜 미상')} {item.get('hospital', '')} : {item.get('diagnosis', '')}".strip()
        for item in items[-limit:]
    ]
    return "\n".join(lines)


def _brief_entries(state: dict[str, Any], limit: int = 3) -> str:
    """최근 일기를 prompt 용으로 축약한다(원문 전체를 넣지 않는다)."""
    items = [
        e
        for e in (state.get("supporting_daily_entries") or state.get("daily_entries") or [])
        if isinstance(e, dict)
    ]
    if not items:
        return "없음"
    lines = []
    for item in items[-limit:]:
        lines.append(
            f"- {item.get('record_date', '날짜 미상')} "
            f"식사:{item.get('food', '-')} / 활동:{item.get('activity', '-')} / "
            f"증상:{item.get('symptom') or '없음'} / 구토:{item.get('vomit', '-')}"
        )
    return "\n".join(lines)


def build_assessment_prompt(state: dict[str, Any], baseline: AssessmentResult) -> str:
    """평가 prompt 본문을 만든다 — Risk Double Check 도 이 함수를 재사용한다.

    명세 21절에 따라 State 에 있는 전체 DB 가 아니라 **현재 질문에 필요한 부분만**
    넣는다(진단서·일기장은 최근 몇 건만 축약).
    """
    profile = state.get("priority_pet_context") or state.get("pet_profile") or {}

    # 진단서·일기장·사용자 입력은 전부 '외부 데이터' 다. 그 안에 "확정 진단을 말해라"
    # 같은 문장이 있어도 지시로 읽히지 않도록 경계를 명시해 감싼다(prompts 인젝션 규칙).
    return f"""[반려동물]
종: {profile.get('species', '미상')} / 품종: {profile.get('breed', '미상')} / 나이: {profile.get('age_years', '미상')}세
기존 질병: {profile.get('diseases') or '없음'}
복용 중인 약: {profile.get('medications') or '없음'}
알레르기: {profile.get('allergies') or '없음'}

{wrap_untrusted_block('보호자의 현재 입력', str(state.get('user_message') or '(없음)'))}

{wrap_untrusted_block('추가로 수집된 답변', str(state.get('collected_information') or '없음'))}

{wrap_untrusted_block('관련 진단서', _brief_diagnoses(state))}

{wrap_untrusted_block('최근 일기장', _brief_entries(state))}

[규칙 기반 사전 판정 — 개발자가 계산한 값이며 신뢰할 수 있다]
risk_level={baseline.risk_level} / emergency_urgency={baseline.emergency_urgency}
red_flags={baseline.red_flags}

위 정보를 바탕으로 위험도를 재평가하라. 사전 판정보다 낮출 수 없다."""


def assessment_agent_node(state: dict) -> dict:
    """LLM 으로 위험도를 재평가한다. LLM 이 없으면 규칙 결과를 그대로 쓴다.

    `build_llm()` 이 None 을 돌려주는 것(키 없음 / 패키지 미설치)은 **정상 경로**이며,
    그 경우 규칙 결과가 그대로 `assessment_risk` 가 된다.

    LLM 결과는 `merge_risk()` 로 규칙 결과와 합쳐 **절대 낮아지지 않게** 한다.
    명세 47절 "낮은 위험도로 덮어쓰지 않는다" 를 node 안에서도 한 번 더 강제한다.
    """
    baseline = evaluate_rules(state)
    llm = build_llm()

    if llm is None:
        return {
            "assessment_risk": baseline.risk_level,
            "emergency_urgency": baseline.emergency_urgency,
            "red_flags": baseline.red_flags,
            "risk_reasons": ["[평가] LLM 없이 규칙 기반 판정을 사용함"],
        }

    result = safe_structured_invoke(
        llm,
        [
            ("system", ASSESSMENT_PROMPT),
            ("human", build_assessment_prompt(state, baseline)),
        ],
        AssessmentResult,
        baseline,
    )

    # LLM 이 위험도를 낮추려 해도 규칙 결과 아래로는 내려가지 않는다.
    risk = merge_risk(baseline.risk_level, result.risk_level)
    urgency: EmergencyUrgency = baseline.emergency_urgency
    if URGENCY_PRIORITY.get(result.emergency_urgency, 0) > URGENCY_PRIORITY[urgency]:
        urgency = result.emergency_urgency
    if risk == "emergency" and urgency == "none":
        urgency = "contact_ready"

    if result.risk_level != risk:
        logger.info(
            "Assessment Agent 가 규칙보다 낮은 위험도(%s)를 반환해 %s 로 유지합니다.",
            result.risk_level,
            risk,
        )

    return {
        "assessment_risk": risk,
        "emergency_urgency": urgency,
        "red_flags": list(baseline.red_flags) + list(result.red_flags),
        "risk_reasons": [f"[평가] {reason}" for reason in result.reasons],
    }
