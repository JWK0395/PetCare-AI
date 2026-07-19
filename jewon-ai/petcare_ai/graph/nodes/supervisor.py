"""Supervisor Agent — 의도(intent)만 분류한다(명세 26절).

계약:
  - 반환은 `SupervisorResult` 구조화 출력뿐이다.
  - **최종 의료 답변을 생성하지 않는다.** 답변은 General Chat / Health Response /
    Emergency 노드의 몫이고, Supervisor 가 답을 만들면 분기 전에 안전 검증을
    거치지 않은 문장이 새어 나간다.
  - 분기 자체는 LangGraph 가 한다(명세 19절). 이 노드는 `intent` 만 State 에 쓰고,
    `route_after_supervisor()` 가 그 값을 읽어 edge 를 고른다.

LLM 이 없으면(키 없음) 키워드 규칙으로 분류한다.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Callable

from ...schemas import Intent, SupervisorResult, merge_risk
from ..state import Replace
from .fast_emergency_guard import detect_emergency_signals, normalize_for_match

if TYPE_CHECKING:  # state.py 는 동시 작성 중이다.
    from ..state import PetCareState  # noqa: F401

logger = logging.getLogger(__name__)

__all__ = [
    "GREETING_KEYWORDS",
    "APP_FUNCTION_KEYWORDS",
    "HOSPITAL_KEYWORDS",
    "HOSPITAL_SEARCH_KEYWORDS",
    "SYMPTOM_KEYWORDS",
    "KNOWLEDGE_KEYWORDS",
    "OUT_OF_SCOPE_KEYWORDS",
    "classify_intent_rule_based",
    "evaluate_supervisor",
    "make_supervisor_node",
    "supervisor_node",
    "route_after_supervisor",
]


# ---------------------------------------------------------------------------
# 규칙 사전 — 공백을 제거한 형태로 적는다(fast_emergency_guard 와 동일 규약).
# ---------------------------------------------------------------------------
GREETING_KEYWORDS: tuple[str, ...] = (
    "안녕", "반가", "하이", "헬로", "고마워", "고맙습니다", "감사", "수고",
    "잘지내", "잘있었", "심심", "이름이뭐", "너는누구", "누구세요", "잘부탁",
)

APP_FUNCTION_KEYWORDS: tuple[str, ...] = (
    "무엇을할수있", "뭘할수있", "뭐할수있", "기능이뭐", "어떤기능", "사용법",
    "어떻게사용", "어떻게써", "어떻게쓰", "앱설정", "알림설정", "일기를어떻게",
    "기록하는법", "회원가입", "로그인", "비밀번호", "화면에서", "메뉴",
    "일기장기능", "이앱은",
)

HOSPITAL_KEYWORDS: tuple[str, ...] = (
    "병원", "동물병원", "응급실", "수의사", "24시", "이십사시", "야간진료", "진료시간",
)

# 병원을 '찾아 달라' 는 명시적 요청 신호
HOSPITAL_SEARCH_KEYWORDS: tuple[str, ...] = (
    "근처", "가까운", "주변", "찾아", "알려줘", "알려주세요", "추천", "어디로가",
    "어디가", "어디있", "검색", "리스트", "목록", "연락처", "전화번호", "예약",
)

# **증상 표현만** 넣는다 — 주제어를 넣으면 일상 대화가 문진으로 끌려간다.
#
# 예전에는 "식사"·"사료"·"산책"·"간식"·"이빨"·"알레르기" 같은 **주제어**가 들어 있었다.
# 그 결과 "산책은 얼마나 시켜야 해?", "간식 뭐가 좋아?" 같은 평범한 질문이 전부
# health_question 으로 분류되어 "가장 신경 쓰이는 증상이 무엇인가요?" 를 되물었다.
# 심지어 "밥 잘 먹었어" 라는 **좋은 소식**까지 증상 상담으로 갔다.
#
# 판단 기준: 그 단어가 **문제가 있다는 뜻을 담고 있는가**.
#   "산책"  → 주제. 산책 자체는 증상이 아니다.          → 뺀다
#   "안 걷" → 증상. 못 걷는다는 상태 서술이다.           → 남긴다
#
# 상태 변화형("식사량이 줄었") 은 변화를 나타내는 쪽('줄었'·'감소')만 남긴다.
# 주제어까지 넣으면 그 주제를 언급하기만 해도 걸린다.
#
# 이 사전은 **LLM 이 없을 때의 폴백**이다. LLM 이 있으면 Supervisor 가 문장을 읽고
# 판단하므로, 여기서 넓게 잡아 이득을 볼 일이 없고 오분류만 늘어난다.
SYMPTOM_KEYWORDS: tuple[str, ...] = (
    # 소화기 — 구토는 어형이 다양해 조사가 끼는 형태까지 넣는다.
    "토해", "토했", "토를", "토한", "토가", "토함", "게워", "구토",
    "설사", "혈변", "혈뇨", "묽은변", "무른변",
    # 호흡·순환
    "기침", "재채기", "콧물", "숨을", "숨이", "호흡곤란", "헥헥", "그렁",
    # 운동기
    "절뚝", "다리를절", "못걷", "안걷", "못일어", "비틀",
    # 피부·눈·귀
    "가려워", "긁어", "긁는", "발진", "탈모", "진물", "눈곱", "충혈",
    # 전신 상태 — '기운이 없다' 는 그 자체가 이상 서술이다.
    "무기력", "기운이없", "기력저하", "축처", "늘어져", "쳐져", "처져",
    "아파", "아픈", "아프", "통증", "부었", "붓기", "혹이", "멍울",
    "열이나", "고열", "떨고", "발작", "경련", "쓰러", "실신",
    # 식사·활동의 **변화** — 주제어가 아니라 변화 표현만.
    "안먹", "못먹", "덜먹", "남겼", "남김", "식욕이", "식욕부진",
    "물을많이", "살이빠", "살이쪘", "줄었", "줄어들",
    # 보호자가 이상함을 직접 표현하는 말
    "증상", "이상해", "이상한", "왜이럴", "정상인가", "괜찮을까",
)

# 증상 호소 없이 **수의학 지식**을 묻는 표현(명세 6절 '일반적인 수의학 정보 검색').
#
# 이 사전이 없던 동안 "강아지에게 뭐가 몸에 좋아?" 같은 질문은 어느 사전에도 걸리지
# 않아 `unsupported` 로 떨어졌고, LLM 분류기에서는 `health_question` 으로 가서
# 증상 문진에 막혔다. 둘 다 답을 못 주는 결과였다.
#
# `SYMPTOM_KEYWORDS` 보다 **뒤에서** 판정한다 — "구토할 때 뭐가 좋아?" 처럼 증상과
# 섞이면 증상 상담이 우선이다(미탐 방지).
KNOWLEDGE_KEYWORDS: tuple[str, ...] = (
    "몸에좋", "좋은가요", "좋을까", "괜찮나요", "괜찮을까", "먹여도", "먹어도",
    "줘도되", "급여", "영양", "영양제", "사료", "간식", "관리", "예방", "접종",
    "백신", "산책은", "목욕", "양치", "훈련", "추천", "차이", "무엇인가요",
    "뭔가요", "뭐야", "뭔지", "알려줘", "알려주세요", "방법",
)

# 반려동물 건강과 무관한 요청 — 지원 범위 밖임을 분명히 한다.
OUT_OF_SCOPE_KEYWORDS: tuple[str, ...] = (
    "주식", "코인", "비트코인", "환율", "부동산", "날씨", "번역", "코딩",
    "프로그램짜", "숙제", "레시피", "여행", "영화", "게임", "정치", "선거",
    "로또", "대출", "보험료계산",
)


def _hit(compact: str, keywords: tuple[str, ...]) -> str | None:
    """공백 제거한 문장에서 첫 매칭 키워드를 돌려준다."""
    for keyword in keywords:
        if keyword in compact:
            return keyword
    return None


# ---------------------------------------------------------------------------
# 규칙 기반 분류
# ---------------------------------------------------------------------------
def classify_intent_rule_based(text: str | None) -> SupervisorResult:
    """LLM 없이 의도를 분류한다(키 없는 환경의 정상 경로).

    판단 순서에는 이유가 있다.
      1. 위급 신호가 있으면 무조건 health_question 이다. 인사말과 섞여 있어도
         건강 경로로 보내야 한다(미탐 방지).
      2. 증상 표현이 있으면 병원 단어가 같이 있어도 health_question 이다.
         "기침하는데 병원 가야 할까요?" 는 위험도 평가가 먼저이고, 병원 추천은
         Visit/Emergency 서브그래프가 이어서 처리한다(명세 31/32절).
      3. 증상 없이 병원을 '찾아 달라' 고만 하면 hospital_search 다.
      4. 인사·앱 기능 질문은 general_chat.
      5. 증상 없이 수의학 지식을 물으면 general_knowledge — RAG 로 답한다(명세 6절).
      6. 명백히 범위 밖(주식·날씨 등)만 unsupported.
      7. 나머지는 general_chat — 판단이 안 된다고 거절하지 않는다.
    """
    raw = (text or "").strip()
    compact = normalize_for_match(raw)

    if not compact:
        return SupervisorResult(
            intent="unsupported",
            reason="입력이 비어 있어 의도를 판단할 수 없습니다.",
        )

    emergency_hits = detect_emergency_signals(raw)
    possible_emergency = any(h.tier == "critical_immediate" for h in emergency_hits)

    if emergency_hits:
        labels = ", ".join(h.signal.label for h in emergency_hits)
        return SupervisorResult(
            intent="health_question",
            possible_emergency=possible_emergency,
            needs_clinical_context=True,
            reason=f"위험 신호 표현이 포함되어 건강 상담으로 분류했습니다: {labels}",
        )

    symptom = _hit(compact, SYMPTOM_KEYWORDS)
    hospital = _hit(compact, HOSPITAL_KEYWORDS)
    search_intent = _hit(compact, HOSPITAL_SEARCH_KEYWORDS)

    if symptom:
        return SupervisorResult(
            intent="health_question",
            possible_emergency=False,
            needs_clinical_context=True,
            reason=f"증상·건강 관련 표현이 있어 건강 상담으로 분류했습니다: '{symptom}'",
        )

    if hospital and search_intent:
        return SupervisorResult(
            intent="hospital_search",
            possible_emergency=False,
            # 병원 요구사항 생성에 PET DB·진단서가 필요하다(명세 33절).
            needs_clinical_context=True,
            reason=f"병원 검색 요청으로 분류했습니다: '{hospital}' + '{search_intent}'",
        )

    if hospital:
        return SupervisorResult(
            intent="hospital_search",
            needs_clinical_context=True,
            reason=f"병원 관련 문의로 분류했습니다: '{hospital}'",
        )

    greeting = _hit(compact, GREETING_KEYWORDS)
    app_function = _hit(compact, APP_FUNCTION_KEYWORDS)
    if greeting or app_function:
        return SupervisorResult(
            intent="general_chat",
            needs_clinical_context=False,
            reason=f"인사 또는 앱 기능 질문으로 분류했습니다: '{greeting or app_function}'",
        )

    out_of_scope = _hit(compact, OUT_OF_SCOPE_KEYWORDS)
    if out_of_scope:
        return SupervisorResult(
            intent="unsupported",
            needs_clinical_context=False,
            reason=f"반려동물 건강과 무관한 요청입니다: '{out_of_scope}'",
        )

    knowledge = _hit(compact, KNOWLEDGE_KEYWORDS)
    if knowledge:
        return SupervisorResult(
            intent="general_knowledge",
            possible_emergency=False,
            # 지식 질문은 이 아이의 상태를 판단하지 않는다 — 종(species)만 있으면
            # RAG 를 돌릴 수 있고, 진단서·일기를 끌어오면 묻지 않은 것에 답하게 된다.
            needs_clinical_context=False,
            reason=f"증상 호소 없이 수의학 지식을 묻는 질문입니다: '{knowledge}'",
        )

    # **기본값은 거절이 아니라 대화다.**
    #
    # 예전 기본값은 `unsupported` 여서 "우리 애 이름은 콩이야", "오늘 산책 다녀왔어"
    # 같은 평범한 말에 "그 부분은 도와드리기 어려운 주제예요" 라고 답했다. 반려동물
    # 앱에서 반려동물 이야기를 거절한 셈이다.
    #
    # 범위 밖은 위에서 `OUT_OF_SCOPE_KEYWORDS` 가 이미 걸렀다. 거기 안 걸렸다면
    # 최소한 대화는 해야 한다 — 판단이 안 되는 것과 지원하지 않는 것은 다르다.
    return SupervisorResult(
        intent="general_chat",
        needs_clinical_context=False,
        reason="증상·병원·지식 질문 어디에도 해당하지 않아 일반 대화로 응답합니다.",
    )


# ---------------------------------------------------------------------------
# LLM 경로
# ---------------------------------------------------------------------------
_SYSTEM_PROMPT = (
    "너는 반려동물 건강 상담 그래프의 라우터다. 사용자의 마지막 메시지를 읽고 "
    "의도만 분류한다.\n"
    "절대 하지 말 것: 진단, 처방, 약 변경 안내, 어떤 형태의 의료 답변 생성. "
    "너는 답변을 만들지 않는다. 분류 결과만 낸다.\n"
    "intent 정의:\n"
    "- general_chat: 인사, 잡담, 앱 사용법·기능 질문, 그리고 **근황·소감 같은 평범한 말**. "
    "예) '오늘 산책 다녀왔어', '밥 잘 먹었어', '우리 강아지 귀여워'\n"
    "- general_knowledge: **지금 나타난 증상 호소 없이** 수의학 지식을 묻는 질문. "
    "예) 먹여도 되는 음식, 영양·사료·간식, 예방접종, 평소 관리 방법, "
    "특정 질환이 무엇인지. '우리 아이가 ~하다' 가 아니라 '~는 어떤가요' 형태다.\n"
    "- health_question: **지금 이 아이에게 이상이 있다**는 서술이나 그에 대한 질문. "
    "'밥 잘 먹었어'·'잘 놀아' 처럼 괜찮다는 말은 health_question 이 아니다(general_chat).\n"
    "- hospital_search: 병원·응급실을 찾아 달라는 요청(증상 서술 없이)\n"
    "- unsupported: 반려동물과 **명백히 무관한** 요청만(주식·날씨·번역 등). "
    "애매하면 unsupported 가 아니라 general_chat 으로 분류한다 — 판단이 안 되는 것과 "
    "지원하지 않는 것은 다르다.\n"
    "증상 서술과 병원 요청이 함께 있으면 health_question 으로 분류한다.\n"
    "지식 질문과 증상 서술이 함께 있으면 health_question 이 우선이다.\n"
    "이미 먹였다·삼켰다 처럼 **일어난 사건**을 말하면 health_question 이고, "
    "먹여도 되는지 **묻기만** 하면 general_knowledge 다.\n"
    "**[직전 대화] 를 반드시 읽고 판단한다.** 사용자의 마지막 문장은 앞 대화를 이어받는 "
    "경우가 많아, 그 문장만 보면 뜻이 통하지 않는다.\n"
    "**앞서 안내한 내용을 되묻는 말은 새 증상 신고가 아니다.** "
    "예) '그 정도로 심각해?', '꼭 가야 해?', '왜 그런 거야?', '얼마나 위험해?' 는 "
    "직전 답변에 대한 설명 요청이므로 general_knowledge 다. 새로 증상을 말한 것이 "
    "아니므로 증상 문진을 다시 시작하면 안 된다.\n"
    "반대로 **새로운 증상이나 상태 변화를 말하면** 그것은 health_question 이다.\n"
    "특히 **AI 가 직전에 '~하면 바로 병원에 연락하라' 고 안내한 그 일이 일어났다고 "
    "말하면, 증상이 가벼워 보여도 health_question + possible_emergency=true 다.** "
    "예) AI 가 '포도를 먹었다면 즉시 연락하라' 고 했고 보호자가 '조금 먹은 것 같다' 고 "
    "하면, 그것은 섭취 사실을 알린 것이므로 응급이다. '큰 반응은 없다' 는 말에 "
    "안심하면 안 된다 — 중독은 증상이 늦게 나타난다.\n"
    "[보류된 위급 신호] 가 주어지면, 그것이 **이미 일어난 일**인지 **묻기만 하는 것**인지 "
    "판정한다. 일어난 일이면 health_question + possible_emergency=true, 묻기만 하면 "
    "general_knowledge + possible_emergency=false 다. 확신이 없으면 일어난 일로 본다.\n"
    "possible_emergency 는 생명이 위험할 수 있는 표현이 있을 때만 true 로 한다.\n"
    "needs_clinical_context 는 반려동물의 프로필·진단서·일기 정보가 답변에 "
    "필요할 때 true 로 한다."
)


#: 분류에 참고할 직전 대화 turn 수(보호자·AI 합쳐서).
RECENT_TURNS_FOR_INTENT = 4


def _recent_turns(state: dict, limit: int = RECENT_TURNS_FOR_INTENT) -> str:
    """직전 대화를 분류 입력으로 만든다.

    **이게 없으면 이어지는 말을 판단할 수 없다.** 실제로 이런 일이 있었다.

        보호자: "강아지는 포도를 먹어도 되나요?"
        AI    : "먹이면 안 됩니다. 먹었다면 증상이 없어도 바로 병원에 연락하세요."
        보호자: "아주 조금 먹은거 같은데 큰 반응은 없긴해요"   ← 섭취 사실을 알림

    마지막 문장만 보면 '포도' 도 '중독' 도 없어서 어떤 규칙에도 걸리지 않는다.
    그래서 응급이 아니라 일반 증상 문진으로 갔다. AI 가 직전에 "먹었다면 바로
    연락하라" 고 해놓고, 먹었다고 하니 "언제부터 증상이 있었나요?" 를 되물은 것이다.

    사람은 대화를 보면 안다. 규칙은 문장 하나만 본다. 그래서 **대화를 판단하는 일은
    LLM 에게 맡기고** 그 재료로 직전 turn 을 넣어 준다.
    """
    turns: list[str] = []
    for message in (state.get("messages") or [])[-limit:]:
        if isinstance(message, dict):
            role, content = message.get("role"), message.get("content")
        else:
            role = getattr(message, "role", None) or getattr(message, "type", None)
            content = getattr(message, "content", None)
        body = str(content or "").strip()
        if not body:
            continue
        who = "보호자" if role in ("user", "human") else "AI"
        turns.append(f"{who}: {body[:300]}")
    return "\n".join(turns)


def _build_prompt(state: dict, text: str) -> list[dict[str, str]]:
    """분류에 필요한 최소 정보만 넣는다 — 임상 DB 원문은 넣지 않는다(명세 21절)."""
    summary = str(state.get("conversation_summary") or "").strip()
    profile = state.get("pet_profile") or {}
    species = profile.get("species") or "unknown"
    pending = [str(x) for x in (state.get("pending_emergency_signals") or []) if str(x).strip()]
    recent = _recent_turns(state)
    user = (
        f"[반려동물 종] {species}\n"
        + (f"[보류된 위급 신호] {', '.join(pending)}\n" if pending else "")
        + (f"[이전 대화 요약]\n{summary}\n" if summary else "")
        + (f"[직전 대화]\n{recent}\n" if recent else "")
        + f"[사용자 메시지]\n{text}"
    )
    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def _current_text(state: dict) -> str:
    """분류 대상 문장 — 현재 입력이 없으면 마지막 user 메시지를 쓴다."""
    message = str(state.get("user_message") or "").strip()
    if message:
        return message
    for item in reversed(state.get("messages") or []):
        role = (
            str(item.get("role") or item.get("type") or "")
            if isinstance(item, dict)
            else str(getattr(item, "type", "") or getattr(item, "role", ""))
        )
        if role in {"user", "human"}:
            content = (
                item.get("content") if isinstance(item, dict) else getattr(item, "content", "")
            )
            return str(content or "").strip()
    return ""


def evaluate_supervisor(state: dict, llm: Any | None = None) -> SupervisorResult:
    """의도 분류 결과 전체를 돌려준다(노드는 이 중 `intent` 만 State 에 쓴다).

    LLM 이 있어도 규칙 결과를 먼저 계산해 기본값으로 넘긴다. LLM 실패·스키마
    위반 시 그대로 규칙 결과가 쓰인다(`safe_structured_invoke` 는 예외를 밖으로
    내보내지 않는다).

    **안전 보정**: 규칙이 위급 신호를 찾았는데 LLM 이 general_chat/unsupported 로
    분류하면 규칙 결과를 채택한다. 위급 신호를 잡담으로 흘려보내는 미탐이 가장
    위험하기 때문이다.
    """
    text = _current_text(state)
    rule_result = classify_intent_rule_based(text)
    pending = [str(x) for x in (state.get("pending_emergency_signals") or []) if str(x).strip()]

    if llm is None:
        if pending:
            # 판정할 LLM 이 없다 → 보류된 신호를 **사건으로 간주**한다.
            # 규칙만으로는 "먹었다/먹어도 되나" 를 가를 수 없으므로, 미탐을 피하는
            # 쪽(응급)으로 확정하는 것이 안전 기본값이다.
            logger.warning("LLM 이 없어 보류된 위급 신호를 응급으로 확정합니다: %s", pending)
            return SupervisorResult(
                intent="health_question",
                possible_emergency=True,
                needs_clinical_context=True,
                reason=f"위급 신호를 판정할 LLM 이 없어 보수적으로 응급 처리: {', '.join(pending)}",
            )
        return rule_result

    from ...llm import safe_structured_invoke  # 지연 import

    result = safe_structured_invoke(
        llm, _build_prompt(state, text), SupervisorResult, rule_result
    )

    if rule_result.possible_emergency and result.intent in {"general_chat", "unsupported"}:
        logger.warning(
            "LLM 분류(%s)가 규칙의 위급 신호와 충돌해 규칙 결과를 채택합니다.",
            result.intent,
        )
        return rule_result
    return result


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------
def make_supervisor_node(llm: Any | None = None) -> Callable[[dict], dict]:
    """LLM 을 주입한 노드를 만든다(테스트가 mock LLM 을 넣을 수 있게)."""

    def _node(state: dict) -> dict:
        return _supervise(state, llm)

    return _node


def supervisor_node(state: dict) -> dict:
    """의도를 분류해 `intent` 만 갱신한다. 답변 문장은 만들지 않는다."""
    from ...llm import build_llm  # 지연 import

    return _supervise(state, build_llm())


def _supervise(state: dict, llm: Any | None) -> dict:
    result = evaluate_supervisor(state, llm)
    logger.info(
        "Supervisor intent=%s possible_emergency=%s reason=%s",
        result.intent,
        result.possible_emergency,
        result.reason,
    )

    update: dict[str, Any] = {"intent": result.intent}

    # 분류기가 응급이라고 보면 **State 에 실제로 반영한다.**
    #
    # `possible_emergency` 는 State 필드로 존재하는데 Supervisor 가 쓰지 않고 있었다.
    # 그래서 LLM 이 대화를 읽고 "포도를 먹었다고 하니 응급" 이라고 판정해도 라우팅에
    # 아무 영향이 없었고, 그대로 증상 문진으로 흘러갔다.
    #
    # 위험도를 emergency 로 올리는 것은 `merge_risk`(상향 전용)를 거치므로 이미
    # 더 높은 판정이 있으면 낮추지 않는다.
    if result.possible_emergency:
        logger.warning("Supervisor 가 응급으로 판정했습니다: %s", result.reason)
        update["possible_emergency"] = True
        update["rule_risk"] = merge_risk(state.get("rule_risk"), "emergency")
        update["final_risk"] = merge_risk(state.get("final_risk"), "emergency")
        # `critical_immediate` 로 올려 **문진을 건너뛰고 병원 연락을 먼저** 하게 한다
        # (명세 29절: 정보 수집이 전화 action 을 막지 않는다).
        #
        # 이게 없으면 "포도를 조금 먹은 것 같다" 에 응급 판정이 나고도 "지금 가장 심한
        # 증상이 무엇인가요?" 를 먼저 묻는다. 중독은 증상이 늦게 나타나므로 증상을
        # 기다리는 동안이 가장 위험한 시간이다. 실제로 그 화면이 나왔다.
        if str(state.get("emergency_urgency") or "none") != "critical_immediate":
            update["emergency_urgency"] = "critical_immediate"

    # 보류됐던 위급 신호를 **질문으로 판정했으면 red flag 를 거둔다.**
    #
    # Fast Guard 가 남긴 "중독 의심 섭취(포도를먹)" 가 State 에 그대로 있으면,
    # 답변 노드가 그걸 확인된 사실로 읽어 "포도를 먹은 것으로 확인되어..." 라고
    # 단정한다. 실제로 그렇게 나왔다 — 보호자는 먹여도 되는지 물었을 뿐인데
    # 먹였다고 답한 것이다. 분기를 바로잡았으면 그 근거도 함께 거둬야 한다.
    pending = [str(x) for x in (state.get("pending_emergency_signals") or []) if str(x).strip()]
    if pending and not result.possible_emergency:
        keep = [
            flag
            for flag in (state.get("red_flags") or [])
            if not any(str(flag).startswith(label) for label in pending)
        ]
        update["red_flags"] = Replace(keep)
        update["pending_emergency_signals"] = Replace([])
        logger.info("질문으로 판정되어 보류 신호를 해제합니다: %s", pending)

    return update


def route_after_supervisor(state: dict) -> Intent:
    """`add_conditional_edges` 용 분기 함수(명세 24절 INTENT 분기).

    알 수 없는 값이 들어오면 unsupported 로 보낸다 — 임의로 건강 경로를 태우면
    검증되지 않은 답변이 나갈 수 있다.
    """
    intent = state.get("intent")
    if intent in {
        "general_chat",
        "general_knowledge",
        "health_question",
        "hospital_search",
        "unsupported",
    }:
        return intent  # type: ignore[return-value]
    logger.warning("알 수 없는 intent 라 unsupported 로 보냅니다: %r", intent)
    return "unsupported"
