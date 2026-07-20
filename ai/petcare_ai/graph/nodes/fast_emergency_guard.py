"""Fast Emergency Guard — LLM 호출 **전에** 규칙만으로 즉시 위급 신호를 잡는다.

왜 LLM 앞에 두는가(명세 24절):
  - 즉시 위급 상황에서는 LLM 왕복 지연(수 초)과 실패 가능성(타임아웃·rate limit·
    키 없음) 자체가 위험이다. 규칙 사전은 항상 있고 즉시 판정된다.
  - 판정이 서면 Supervisor·Clinical Context·RAG 를 모두 건너뛰고 Emergency
    서브그래프로 직행한다.

판정 성향(중요):
  **오탐(false positive)보다 미탐(false negative)이 훨씬 위험하다.** 따라서
  - 애매하면 위험 쪽으로 올린다.
  - 부정 표현은 매우 좁은 범위에서만 인정한다("경련은 없어요" 처럼 신호어 바로
    뒤에 부정 어미가 붙은 경우). 그 외에는 전부 신호로 취급한다.
  - "지금은 멈췄어요" 같은 종료 표현은 신호를 **지우지 않고** critical_immediate
    에서 경고 등급으로만 낮춘다(발작 후기·재발 위험 때문에 여전히 연락 대상이다).

이 노드는 진단하지 않는다. 답변 문장도 만들지 않는다. 오직 라우팅 신호만 만든다.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from ...schemas import merge_risk

if TYPE_CHECKING:  # 순환 import 방지 — state.py 는 동시 작성 중이다.
    from ..state import PetCareState  # noqa: F401

logger = logging.getLogger(__name__)

UrgencyTier = Literal["critical_immediate", "contact_ready"]

__all__ = [
    "EmergencySignal",
    "SignalHit",
    "CRITICAL_SIGNALS",
    "WARNING_SIGNALS",
    "NEGATION_MARKERS",
    "RESOLVED_MARKERS",
    "normalize_for_match",
    "detect_emergency_signals",
    "fast_emergency_guard_node",
    "is_critical_immediate",
    "route_after_fast_emergency_guard",
]


# ---------------------------------------------------------------------------
# 신호 사전
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class EmergencySignal:
    """위급 신호 1종. `patterns` 는 **공백을 모두 제거한 형태**로 적는다.

    한국어 입력은 띄어쓰기가 제멋대로다("숨을 못 쉬어요" / "숨을못쉬어요" /
    "숨 을 못쉬어요"). 그래서 매칭 직전에 입력의 공백을 전부 제거하고, 사전도
    같은 형태로 적어 띄어쓰기 차이로 미탐이 나지 않게 한다.
    """

    code: str
    label: str
    tier: UrgencyTier
    patterns: tuple[str, ...]
    #: 이 신호가 **질문 문장에도 그대로 나타날 수 있는가**.
    #:
    #: 관찰 서술("숨을 못 쉬어요", "경련해요")은 그 문장 자체가 사건을 단정한다.
    #: 반면 섭취 서술은 어간이 같아 구분되지 않는다:
    #:
    #:     "포도를 먹었어요"      → 사건 (응급)
    #:     "포도를 먹어도 되나요?" → 질문 (응급 아님)
    #:
    #: 둘 다 `포도를먹` 을 포함해서 규칙으로는 갈라낼 수 없다. 실제로 후자가 응급
    #: 화면을 띄웠다. 이런 신호는 **탐지만 하고 판단은 Supervisor(LLM)에게 넘긴다**.
    #: LLM 이 없거나 실패하면 지금까지처럼 응급으로 확정한다(안전 기본값 유지).
    revisable: bool = False


@dataclass(frozen=True)
class SignalHit:
    """탐지 결과 1건 — 어떤 표현이 걸렸는지까지 남겨 trace 에서 확인할 수 있게 한다."""

    signal: EmergencySignal
    matched: str
    tier: UrgencyTier
    downgraded: bool = False

    @property
    def revisable(self) -> bool:
        """LLM 이 '사건이 아니라 질문' 으로 뒤집을 수 있는 신호인가."""
        return self.signal.revisable

    @property
    def red_flag(self) -> str:
        return f"{self.signal.label}({self.matched})"


# 즉시 위급 — 판정 시 Emergency 서브그래프로 직행한다.
CRITICAL_SIGNALS: tuple[EmergencySignal, ...] = (
    EmergencySignal(
        code="respiratory_distress",
        label="호흡곤란",
        tier="critical_immediate",
        patterns=(
            "숨을못쉬", "숨을거의못", "숨을잘못쉬", "숨쉬기힘들", "숨쉬기어려",
            "숨을헐떡", "숨이넘어갈", "숨을안쉬", "숨을못쉬어", "호흡곤란",
            "호흡이곤란", "숨을가쁘게", "가쁘게숨", "목을빼고숨", "배로숨을쉬",
            "혀를내밀고숨", "입을벌리고숨", "개구호흡", "호흡을멈",
            "숨을거의쉬지못", "숨을쉬지못", "숨을못쉬고",
        ),
    ),
    EmergencySignal(
        code="cyanosis",
        label="청색증",
        tier="critical_immediate",
        patterns=(
            "청색증", "잇몸이파랗", "잇몸이보라", "잇몸이창백", "잇몸이하얗",
            "혀가파랗", "혀가보라", "혀가하얗", "혀가창백", "입술이파랗",
            "몸이파랗", "새파랗",
        ),
    ),
    EmergencySignal(
        code="seizure",
        label="경련·발작",
        tier="critical_immediate",
        patterns=(
            "경련", "발작", "경기를", "경기해", "부들부들떨면서", "거품을물",
            "몸이뻣뻣해지", "사지가뻣뻣", "눈이돌아가", "실룩",
        ),
    ),
    EmergencySignal(
        code="unconscious",
        label="의식 저하·무반응",
        tier="critical_immediate",
        patterns=(
            "의식이없", "의식없", "의식을잃", "의식이흐", "반응이없", "반응없",
            "불러도반응", "깨어나지않", "정신을잃", "혼수", "축늘어",
            "쓰러졌", "쓰러져", "기절", "실신", "반응이둔", "반응이없어",
        ),
    ),
    EmergencySignal(
        code="poisoning",
        label="중독 의심 섭취",
        tier="critical_immediate",
        # 유일하게 '먹여도 되나요' 형태의 질문과 어간을 공유한다.
        revisable=True,
        patterns=(
            "중독", "초콜릿을먹", "초콜렛을먹", "양파를먹", "마늘을먹",
            "포도를먹", "건포도를먹", "자일리톨", "살충제", "쥐약", "농약",
            "부동액", "제초제", "이부프로펜", "타이레놀", "아세트아미노펜",
            "사람약을먹", "사람약먹", "백합", "담배를먹", "니코틴",
            "세제를먹", "표백제", "곰팡이핀", "상한음식을먹", "독을먹",
            "이상한걸먹", "이물질을삼", "이물질을먹", "삼켰",
        ),
    ),
    EmergencySignal(
        code="hemorrhage",
        label="대량 출혈",
        tier="critical_immediate",
        patterns=(
            "피가멈추지않", "지혈이안", "피를많이", "출혈이심", "대량출혈",
            "피가계속", "피가철철", "피를토", "토혈", "각혈",
            "코피가멈추지", "상처에서피가",
        ),
    ),
    EmergencySignal(
        code="gdv_bloat",
        label="위확장·염전(산통) 의심",
        tier="critical_immediate",
        patterns=(
            "배가부풀", "배가빵빵", "배가딱딱", "복부팽만", "위염전", "위확장",
            "헛구역질", "구역질만", "토하려는데안나", "토하려고하는데",
            "구토시도", "웩웩거리",
        ),
    ),
    EmergencySignal(
        code="urinary_obstruction",
        label="요도 폐색 의심",
        tier="critical_immediate",
        patterns=(
            "소변을못보", "소변이안나", "소변을못누", "오줌을못", "오줌이안나",
            "화장실에서힘", "소변을보려고", "요도가막", "소변이막혔",
        ),
    ),
    EmergencySignal(
        code="heatstroke",
        label="열사병·고체온",
        tier="critical_immediate",
        patterns=("열사병", "일사병", "차에두고내렸", "체온이너무높", "몸이불덩"),
    ),
    EmergencySignal(
        code="trauma",
        label="중증 외상",
        tier="critical_immediate",
        patterns=(
            "차에치", "교통사고", "떨어졌는데", "추락", "밟혔", "물렸는데피",
            "크게다쳤", "뼈가보", "장기가", "눈이튀어나",
        ),
    ),
    EmergencySignal(
        code="dystocia",
        label="난산",
        tier="critical_immediate",
        patterns=("난산", "새끼가안나", "출산중인데", "분만중인데"),
    ),
    EmergencySignal(
        code="collapse",
        label="기립 불능·허탈",
        tier="critical_immediate",
        patterns=(
            "일어서지못", "일어나지못", "서지를못", "뒷다리를못쓰",
            "다리를전혀못", "몸이차가", "체온이너무낮", "저체온",
        ),
    ),
)

# 경고 — 즉시 위급은 아니지만 반드시 downstream 평가에 red flag 로 남긴다.
WARNING_SIGNALS: tuple[EmergencySignal, ...] = (
    EmergencySignal(
        code="tachypnea",
        label="호흡수 증가",
        tier="contact_ready",
        patterns=("호흡이빠", "호흡수가증가", "숨이가빠", "헥헥거림이심", "헐떡거리"),
    ),
    EmergencySignal(
        code="repeated_vomiting",
        label="반복 구토",
        tier="contact_ready",
        patterns=("계속토", "여러번토", "구토를반복", "하루에몇번씩토", "토를멈추지"),
    ),
    EmergencySignal(
        code="blood_in_excreta",
        label="혈변·혈뇨",
        tier="contact_ready",
        patterns=("혈변", "피가섞인변", "변에피", "혈뇨", "소변에피", "붉은소변", "흑색변"),
    ),
    EmergencySignal(
        code="anorexia",
        label="식이·음수 중단",
        tier="contact_ready",
        patterns=("아무것도안먹", "하나도안먹", "물도안마", "이틀째안먹", "며칠째안먹"),
    ),
    EmergencySignal(
        code="lethargy",
        label="기력 저하",
        tier="contact_ready",
        patterns=("기운이없", "기력이없", "계속누워", "움직이질않", "무기력"),
    ),
)

# 신호어 **바로 뒤**에서만 인정하는 부정 표현 — 오인 범위를 좁게 유지한다.
NEGATION_MARKERS: tuple[str, ...] = (
    "은없", "는없", "이없", "가없", "없어요", "없습니다", "없었", "안해요",
    "안했", "하지않", "하진않", "은아니", "는아니", "아니에요", "아닙니다",
)

# 신호는 있었으나 종료된 정황 — 지우지 않고 등급만 낮춘다.
RESOLVED_MARKERS: tuple[str, ...] = (
    "멈췄", "멎었", "그쳤", "괜찮아졌", "지금은괜찮", "이제괜찮", "돌아왔",
    "회복됐", "회복했", "진정됐", "가라앉았",
)

_NEGATION_WINDOW = 10
_RESOLVED_WINDOW = 24


# ---------------------------------------------------------------------------
# 탐지
# ---------------------------------------------------------------------------
def normalize_for_match(text: str | None) -> str:
    """매칭용 정규화 — 공백·문장부호를 제거하고 소문자로 만든다.

    띄어쓰기와 구두점 차이로 위급 신호를 놓치지 않기 위한 전처리다.
    """
    if not text:
        return ""
    lowered = str(text).lower()
    return re.sub(r"[\s,.!?~·…\-_'\"()\[\]{}]+", "", lowered)


def _is_negated(compact: str, index: int, pattern: str) -> bool:
    """신호어 직후 좁은 구간에 부정 표현이 있는지 본다(미탐을 막기 위해 좁게)."""
    tail = compact[index + len(pattern) : index + len(pattern) + _NEGATION_WINDOW]
    return any(marker in tail for marker in NEGATION_MARKERS)


def _is_resolved(compact: str, index: int, pattern: str) -> bool:
    """신호어 이후 구간에 '멈췄다·괜찮아졌다' 정황이 있는지 본다."""
    tail = compact[index + len(pattern) : index + len(pattern) + _RESOLVED_WINDOW]
    return any(marker in tail for marker in RESOLVED_MARKERS)


def detect_emergency_signals(text: str | None) -> list[SignalHit]:
    """규칙 사전으로 위급 신호를 찾는다. LLM 을 쓰지 않는다.

    같은 신호 code 는 한 번만 담는다(같은 증상을 여러 표현으로 말해도 red flag 가
    중복되지 않도록). 부정된 표현은 버리고, 종료된 표현은 등급만 낮춘다.
    """
    compact = normalize_for_match(text)
    if not compact:
        return []

    hits: list[SignalHit] = []
    seen: set[str] = set()

    for signal in (*CRITICAL_SIGNALS, *WARNING_SIGNALS):
        if signal.code in seen:
            continue
        for pattern in signal.patterns:
            index = compact.find(pattern)
            if index < 0:
                continue
            if _is_negated(compact, index, pattern):
                logger.debug("부정 표현으로 제외한 신호: %s(%s)", signal.code, pattern)
                continue
            downgraded = signal.tier == "critical_immediate" and _is_resolved(
                compact, index, pattern
            )
            hits.append(
                SignalHit(
                    signal=signal,
                    matched=pattern,
                    tier="contact_ready" if downgraded else signal.tier,
                    downgraded=downgraded,
                )
            )
            seen.add(signal.code)
            break

    return hits


def _latest_user_text(state: dict[str, Any]) -> str:
    """검사 대상 문장 — 현재 입력이 비어 있으면 마지막 user 메시지를 쓴다."""
    message = (state.get("user_message") or "").strip()
    if message:
        return message
    for item in reversed(state.get("messages") or []):
        role = _message_role(item)
        if role in {"user", "human"}:
            return _message_text(item)
    return ""


def _message_role(item: Any) -> str:
    """dict 메시지와 LangChain BaseMessage 를 모두 받는다."""
    if isinstance(item, dict):
        return str(item.get("role") or item.get("type") or "")
    return str(getattr(item, "type", "") or getattr(item, "role", ""))


def _message_text(item: Any) -> str:
    if isinstance(item, dict):
        return str(item.get("content") or "")
    return str(getattr(item, "content", "") or "")


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------
def fast_emergency_guard_node(state: dict) -> dict:
    """즉시 위급 신호를 규칙으로 판정한다(명세 23/24절).

    반환 키는 명세 25절 State 필드만 사용한다.

      - critical_immediate: `emergency_urgency` / `rule_risk` / `final_risk` 를
        emergency 로 올린다. 세 값 모두 `merge_risk()` 로 병합해 **더 낮은 값으로
        덮어쓰지 않는다**(명세 28절).
      - 경고 등급: `red_flags` / `risk_reasons` 만 남기고 `emergency_urgency` 는
        건드리지 않는다. risk_double_check 의 `resolve_final_risk()` 는 urgency 가
        contact_ready 이기만 해도 final_risk 를 emergency 로 올리므로, 여기서
        경고 신호에 contact_ready 를 붙이면 "기운이 없어요" 같은 입력까지 응급
        서브그래프로 끌려간다. 등급 판정은 Assessment 계열 노드의 몫이며,
        이 노드는 red flag 만 넘겨 그 판단 재료로 쓰이게 한다.

    red_flags / risk_reasons 는 **이번에 새로 찾은 것만** 반환한다. reducer 가
    누적형(add)이든 덮어쓰기형이든 같은 결과가 되도록 하기 위함이다.
    """
    text = _latest_user_text(state)
    hits = detect_emergency_signals(text)
    if not hits:
        return {}

    existing_flags = set(state.get("red_flags") or [])
    existing_reasons = set(state.get("risk_reasons") or [])

    new_flags = [h.red_flag for h in hits if h.red_flag not in existing_flags]
    critical = [h for h in hits if h.tier == "critical_immediate"]

    # **재판정 가능한 신호만 걸렸다면 여기서 응급을 확정하지 않는다.**
    #
    # 중독 섭취 신호는 "포도를 먹었어요"(사건)와 "포도를 먹어도 되나요?"(질문)가
    # 같은 어간을 공유해 규칙으로 갈라낼 수 없다. 여기서 확정해 버리면 Supervisor
    # 가 문장을 읽어 볼 기회조차 없이 응급 화면이 뜬다(실제로 그랬다).
    #
    # 그래서 red flag 는 남기되 응급 확정은 보류하고 Supervisor 로 보낸다.
    # 관찰 서술(호흡곤란·경련·의식저하 등)이 하나라도 같이 걸리면 보류하지 않는다 —
    # 그건 문장 자체가 사건을 단정하는 신호다.
    deferred = bool(critical) and all(h.revisable for h in critical)

    update: dict[str, Any] = {}
    if new_flags:
        update["red_flags"] = new_flags

    if deferred:
        labels = ", ".join(h.signal.label for h in critical)
        reason = (
            f"규칙이 위급 신호를 탐지했으나 질문 문장일 수 있어 판단을 보류했습니다: {labels}"
        )
        if reason not in existing_reasons:
            update["risk_reasons"] = [reason]
        # Supervisor 가 "사건인지 질문인지" 를 판정할 근거로 쓴다.
        # LLM 이 없으면 `evaluate_supervisor` 가 이 값을 보고 응급으로 확정한다.
        update["pending_emergency_signals"] = [h.signal.label for h in critical]
        logger.warning("Fast Emergency Guard — 보류(질문 가능성): %s", labels)
    elif critical:
        labels = ", ".join(h.signal.label for h in critical)
        reason = f"규칙 기반 즉시 위급 신호 탐지: {labels}"
        if reason not in existing_reasons:
            update["risk_reasons"] = [reason]
        update["emergency_urgency"] = "critical_immediate"
        update["rule_risk"] = merge_risk(state.get("rule_risk"), "emergency")
        update["final_risk"] = merge_risk(state.get("final_risk"), "emergency")
        logger.warning("Fast Emergency Guard — 즉시 위급 판정: %s", labels)
    else:
        labels = ", ".join(h.signal.label for h in hits)
        reason = f"규칙 기반 주의 신호 탐지(등급 판정은 평가 노드에서 수행): {labels}"
        if reason not in existing_reasons:
            update["risk_reasons"] = [reason]
        logger.info("Fast Emergency Guard — 주의 신호: %s", labels)

    return update


def is_critical_immediate(state: dict) -> bool:
    """Emergency 서브그래프 직행 여부 — router 가 쓰는 단일 판정 함수."""
    return state.get("emergency_urgency") == "critical_immediate"


def route_after_fast_emergency_guard(state: dict) -> str:
    """`add_conditional_edges` 용 분기 함수 — 분기는 LangGraph 가 한다(명세 19절)."""
    return "emergency" if is_critical_immediate(state) else "supervisor"
