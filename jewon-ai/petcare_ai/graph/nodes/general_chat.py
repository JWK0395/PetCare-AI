"""General Chat Agent / Unsupported Response (명세 24·43절).

명세 43절 '일반 대화' 테스트의 기대는 명확하다.

    입력: "안녕", "이 앱에서 무엇을 할 수 있어?"
    기대: general_chat 분기 / **RAG 미호출** / **Tavily 미호출**

그래서 이 모듈은 `petcare_ai.rag` 를 **import 조차 하지 않는다.** 호출하지 않겠다는
약속을 주석으로 남기는 대신 의존성 자체를 끊어, 실수로라도 검색이 일어날 수 없게
한다(테스트는 rag 서비스에 mock 을 걸어 호출 0회를 확인하면 된다).

인사·기능 안내는 검색이 필요 없는 고정 정보이므로 규칙 기반 응답이 기본이다.
LLM 이 있으면 말투만 자연스럽게 다듬되, 금지 표현이 섞이면 규칙 응답으로 되돌린다.
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Any

from ...llm import build_llm
from ..prompts import GENERAL_CHAT_PROMPT, wrap_untrusted_block
from .output_check import find_forbidden_expressions

if TYPE_CHECKING:
    from ..state import PetCareState

logger = logging.getLogger(__name__)

__all__ = [
    "APP_CAPABILITY_MESSAGE",
    "GREETING_MESSAGE",
    "THANKS_MESSAGE",
    "FAREWELL_MESSAGE",
    "IDENTITY_MESSAGE",
    "DEFAULT_MESSAGE",
    "UNSUPPORTED_MESSAGE",
    "classify_small_talk",
    "rule_based_reply",
    "general_chat_node",
    "unsupported_response_node",
]


# ---------------------------------------------------------------------------
# 고정 응답 (검색 불필요)
# ---------------------------------------------------------------------------
APP_CAPABILITY_MESSAGE = """이 앱에서는 이런 것들을 도와드릴 수 있어요.

1. 반려동물 건강 상담
   보호자님이 관찰한 증상을 말씀해 주시면, 등록된 반려동물 정보와 수의학 자료를 함께 살펴보고
   지금 상황을 어떻게 보면 좋을지 정리해 드려요.

2. 일기장·진단서 기록 활용
   앱에 기록해 두신 식사·활동·증상 일기와 진단서를 상담에 함께 참고해요.
   같은 증상이라도 그동안의 흐름을 함께 봐야 판단이 정확해지거든요.

3. 병원 방문이 필요한지 함께 판단
   경과를 지켜봐도 되는 상황인지, 진료 상담이 필요한 상황인지, 응급으로 봐야 하는지 알려드려요.

4. 병원 찾기와 상담 준비
   방문이 필요하다면 근처 동물병원을 찾아보고, 수의사에게 전달할 상담 요약본(PDF)과
   이메일 초안을 만들어 드려요.

한 가지만 미리 말씀드릴게요. 저는 수의사가 아니라서 병을 확정하거나 약을 처방해 드릴 수 없어요.
진단과 처방은 반드시 병원에서 수의사에게 받으셔야 합니다."""

GREETING_MESSAGE = """안녕하세요! 반려동물 건강 상담을 도와드리는 AI예요.
오늘 아이 상태 중에 신경 쓰이는 부분이 있으신가요? 편하게 말씀해 주세요."""

THANKS_MESSAGE = """도움이 되었다니 다행이에요.
아이 상태가 달라지거나 새로 궁금한 점이 생기면 언제든 다시 말씀해 주세요."""

FAREWELL_MESSAGE = """네, 아이와 좋은 하루 보내세요.
상태가 달라지면 언제든 다시 찾아와 주세요."""

IDENTITY_MESSAGE = """저는 반려동물 건강 상담을 도와드리는 AI예요.
보호자님이 기록해 두신 일기장과 진단서를 함께 보면서, 지금 상황을 정리하고
병원에 가야 할 상황인지 함께 판단해 드려요. 다만 진단과 처방은 수의사만 할 수 있어요."""

# 증상을 **요구하지 않는다.** 예전 문구는 어떤 말에도 "신경 쓰이는 부분을 알려주세요"
# 로 되돌려서, 보호자가 그냥 근황을 얘기해도 문진을 재촉하는 것처럼 읽혔다.
DEFAULT_MESSAGE = """네, 잘 들었어요.
건강이나 병원에 대해 궁금한 점이 생기면 언제든 편하게 말씀해 주세요."""

UNSUPPORTED_MESSAGE = """죄송해요, 그 부분은 제가 도와드리기 어려운 주제예요.
저는 반려동물의 건강 상담과 병원 찾기를 도와드리는 역할을 하고 있어요.

아이의 증상, 식사나 활동의 변화, 병원 방문이 필요한지 같은 내용이라면 편하게 물어봐 주세요."""


# ---------------------------------------------------------------------------
# 규칙 기반 분류
# ---------------------------------------------------------------------------
_SMALL_TALK_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "capability",
        (
            "무엇을 할 수",
            "뭘 할 수",
            "뭐 할 수",
            "어떤 기능",
            "무슨 기능",
            "어떻게 쓰",
            "사용법",
            "사용 방법",
            "뭐 해줄",
            "무엇을 해줄",
            "도와줄 수 있",
            "할 수 있는 게",
            "할 수 있어",
        ),
    ),
    (
        "identity",
        ("너 누구", "누구세요", "누구야", "정체가", "이름이 뭐", "어떤 ai", "무슨 ai"),
    ),
    ("thanks", ("고마", "감사", "고맙", "땡큐", "thank")),
    ("farewell", ("잘 있어", "안녕히", "바이", "다음에 봐", "이만", "종료")),
    ("greeting", ("안녕", "하이", "헬로", "hello", "hi ", "반가", "좋은 아침", "처음 뵙")),
)

_REPLY_BY_TOPIC: dict[str, str] = {
    "capability": APP_CAPABILITY_MESSAGE,
    "identity": IDENTITY_MESSAGE,
    "thanks": THANKS_MESSAGE,
    "farewell": FAREWELL_MESSAGE,
    "greeting": GREETING_MESSAGE,
    "default": DEFAULT_MESSAGE,
}


def classify_small_talk(message: str) -> str:
    """일반 대화 유형을 규칙으로 분류한다.

    분류 순서가 중요하다. "안녕! 이 앱에서 뭘 할 수 있어?" 처럼 인사와 질문이 함께
    오면 **질문에 답하는 쪽이 더 유용**하므로 capability 를 인사보다 먼저 본다.
    """
    raw = (message or "").strip().lower()
    if not raw:
        return "default"
    compact = re.sub(r"\s+", "", raw)
    for topic, keywords in _SMALL_TALK_RULES:
        for keyword in keywords:
            if keyword in raw or re.sub(r"\s+", "", keyword) in compact:
                return topic
    return "default"


def rule_based_reply(message: str) -> str:
    """LLM 없이 만드는 일반 대화 응답 — 이 경로가 기본값이다."""
    return _REPLY_BY_TOPIC[classify_small_talk(message)]


# ---------------------------------------------------------------------------
# LLM (선택) — 말투만 다듬는다
# ---------------------------------------------------------------------------
# 공용 prompt(prompts.GENERAL_CHAT_PROMPT)에 이 node 전용 제약만 덧붙인다.
# 여기서는 LLM 이 사실을 새로 만들지 않고 **어조만** 다듬어야 하기 때문이다.
_POLISH_CONSTRAINT = """
[이 단계의 추가 제약]
- 아래 [기준 답변]의 내용과 사실을 바꾸지 마라. 어조만 자연스럽게 다듬어라.
- 앱에 없는 기능을 추가하지 마라.
- 기준 답변에 없는 반려동물 정보를 지어내지 마라."""


def _polish_with_llm(message: str, base_reply: str) -> str | None:
    """LLM 으로 어조만 다듬는다. 실패하거나 위험하면 None(규칙 응답 유지).

    사실 관계는 `base_reply` 가 이미 확정한 것이고 LLM 은 표현만 손본다. 그래서
    LLM 이 없어도, 실패해도, 이상한 문장을 만들어도 사용자 경험이 무너지지 않는다.
    """
    llm = build_llm()
    if llm is None:
        return None

    try:
        response = llm.invoke(
            [
                ("system", GENERAL_CHAT_PROMPT + _POLISH_CONSTRAINT),
                (
                    "human",
                    f"{wrap_untrusted_block('사용자 메시지', message)}\n\n"
                    f"[기준 답변 — 개발자가 작성한 것이며 사실 기준이다]\n{base_reply}\n\n"
                    "기준 답변의 내용을 유지한 채 자연스럽게 다듬어 주세요.",
                ),
            ]
        )
    except Exception as exc:
        logger.warning("일반 대화 LLM 호출 실패 — 규칙 응답을 사용합니다: %s", exc)
        return None

    text = getattr(response, "content", response)
    if not isinstance(text, str) or not text.strip():
        return None

    violations = find_forbidden_expressions(text)
    if violations:
        logger.warning("일반 대화 LLM 응답에 금지 표현이 있어 규칙 응답을 사용합니다: %s", violations)
        return None

    return text.strip()


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------
def general_chat_node(state: dict) -> dict:
    """일반 대화에 응답한다 — **RAG 와 Tavily 를 호출하지 않는다**(명세 43절).

    위험도는 건드리지 않는다. 일반 대화 분기까지 온 시점에서 Fast Emergency Guard
    와 Supervisor 가 이미 응급이 아니라고 판정했고, 인사말에 위험도를 부여하면
    Output Check 의 'visit/emergency 안내 누락' 검사가 잘못 발동한다.
    """
    message = str(state.get("user_message") or "")
    reply = rule_based_reply(message)

    # 재생성 경로에서는 LLM 을 쓰지 않는다 — 결정론적인 안전 응답으로 되돌린다.
    if int(state.get("retry_count", 0) or 0) == 0:
        polished = _polish_with_llm(message, reply)
        if polished:
            reply = polished

    return {"draft_response": reply}


def unsupported_response_node(state: dict) -> dict:
    """지원하지 않는 요청에 대한 응답(명세 24절 Unsupported Response).

    무엇을 할 수 있는지 함께 알려준다. 거절만 하면 사용자가 다음에 무엇을 물어야
    할지 알 수 없다.
    """
    return {"draft_response": UNSUPPORTED_MESSAGE}
