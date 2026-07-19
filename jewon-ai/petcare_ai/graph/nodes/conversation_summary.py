"""Conversation Summary Agent — 대화가 길어질 때만 이전 메시지를 요약한다(명세 41절).

규칙:
  - 대화가 길 때만 실행한다(`settings.summary_trigger_message_count` 초과).
  - 최근 `settings.summary_keep_recent_messages` 개는 **원문 그대로 유지**하고,
    그보다 이전 메시지만 요약 대상으로 삼는다.
  - **PET DB / 진단서 DB / 일기장 DB 내용을 요약문에 섞지 않는다.** 임상 DB
    context 는 별도 State 로 유지되며, 요약문에 섞이면 (a) 같은 정보가 두 경로로
    들어와 충돌하고 (b) provenance(출처 추적)가 끊긴다.
    → 그래서 이 노드는 `messages` 외의 State 키를 **읽지 않는다**.
  - LLM 이 없으면 규칙 기반으로 축약한다(키 없이도 동작해야 한다).

`messages` 자체는 건드리지 않는다. reducer(add_messages 등)가 무엇이든 안전하도록
`conversation_summary` 만 갱신하고, prompt 조립 시 "요약 + 최근 원문" 으로 쓰면 된다.
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Any, Callable

from pydantic import BaseModel, Field

from ...config import Settings, get_settings
from ...llm import safe_structured_invoke

if TYPE_CHECKING:  # state.py 는 동시 작성 중이다.
    from ..state import PetCareState  # noqa: F401

logger = logging.getLogger(__name__)

__all__ = [
    "ConversationSummaryDraft",
    "needs_conversation_summary",
    "route_needs_summary",
    "summarize_messages",
    "make_conversation_summary_node",
    "conversation_summary_node",
]

# 규칙 기반 요약 길이 상한 — 프롬프트가 무한정 길어지지 않게 한다.
_MAX_SUMMARY_CHARS = 900
_MAX_BULLETS = 12
_MAX_LINE_CHARS = 120


class ConversationSummaryDraft(BaseModel):
    """LLM structured output 스키마 — 자유 서술 대신 구조를 강제한다."""

    summary: str = Field(default="", description="이전 대화의 사실 위주 요약")


# ---------------------------------------------------------------------------
# 메시지 유틸 (dict / LangChain BaseMessage 모두 지원)
# ---------------------------------------------------------------------------
def _role(item: Any) -> str:
    if isinstance(item, dict):
        raw = str(item.get("role") or item.get("type") or "")
    else:
        raw = str(getattr(item, "type", "") or getattr(item, "role", ""))
    if raw in {"human", "user"}:
        return "user"
    if raw in {"ai", "assistant"}:
        return "assistant"
    return raw or "unknown"


def _text(item: Any) -> str:
    if isinstance(item, dict):
        content = item.get("content")
    else:
        content = getattr(item, "content", None)
    if isinstance(content, list):  # multimodal block 형태 방어
        parts = [
            str(block.get("text", ""))
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        ]
        content = " ".join(parts)
    return re.sub(r"\s+", " ", str(content or "")).strip()


def _label(role: str) -> str:
    return {"user": "보호자", "assistant": "AI"}.get(role, role)


# ---------------------------------------------------------------------------
# 실행 조건
# ---------------------------------------------------------------------------
def needs_conversation_summary(state: dict, settings: Settings | None = None) -> bool:
    """요약이 필요한지 판단한다 — 짧은 대화에서는 실행하지 않는다."""
    resolved = settings or get_settings()
    messages = state.get("messages") or []
    return len(messages) > resolved.summary_trigger_message_count


def route_needs_summary(state: dict) -> str:
    """`add_conditional_edges` 용 분기 함수(명세 24절 SUMMARY 분기)."""
    return "conversation_summary" if needs_conversation_summary(state) else "fast_emergency_guard"


# ---------------------------------------------------------------------------
# 요약
# ---------------------------------------------------------------------------
def _rule_based_summary(older: list[Any], previous_summary: str) -> str:
    """LLM 없이 축약한다 — 생성이 아니라 **발췌**이므로 사실을 지어내지 않는다.

    각 메시지의 첫 문장만 남기고 길이를 자른다. 보호자 발화를 우선 남기는 이유는
    증상·경과 정보가 거기 있고, AI 발화는 대개 안내 문구라 재사용 가치가 낮기
    때문이다.
    """
    bullets: list[str] = []
    for item in older:
        text = _text(item)
        if not text:
            continue
        first = re.split(r"(?<=[.!?。])\s+|\n", text)[0].strip() or text
        if len(first) > _MAX_LINE_CHARS:
            first = first[: _MAX_LINE_CHARS - 1].rstrip() + "…"
        line = f"- {_label(_role(item))}: {first}"
        if line not in bullets:
            bullets.append(line)

    if len(bullets) > _MAX_BULLETS:  # 앞부분 대신 최근 쪽을 남긴다.
        bullets = ["- (이전 대화 일부 생략)"] + bullets[-(_MAX_BULLETS - 1) :]

    body = "\n".join(bullets)
    if previous_summary:
        body = f"{previous_summary.strip()}\n{body}"
    if len(body) > _MAX_SUMMARY_CHARS:
        body = body[-_MAX_SUMMARY_CHARS:]
        body = body[body.find("\n") + 1 :] if "\n" in body else body
    return body.strip()


def _build_prompt(older: list[Any], previous_summary: str) -> list[dict[str, str]]:
    """요약 프롬프트 — 임상 DB 를 섞지 말라고 명시적으로 금지한다."""
    transcript = "\n".join(
        f"{_label(_role(item))}: {_text(item)}" for item in older if _text(item)
    )
    system = (
        "너는 반려동물 상담 대화의 '이전 대화 요약' 만 작성한다.\n"
        "규칙:\n"
        "1) 대화에 실제로 등장한 내용만 쓴다. 추측·진단·처방을 절대 추가하지 않는다.\n"
        "2) 반려동물 프로필(PET DB), 진단서, 일기장 내용은 요약에 넣지 않는다. "
        "그 정보는 별도 경로로 전달되며, 여기에 섞이면 출처 추적이 끊긴다.\n"
        "3) 보호자가 말한 증상·경과·시점, 그리고 이미 안내한 내용만 사실 위주로 적는다.\n"
        "4) 한국어 불릿 8줄 이내, 600자 이내."
    )
    user = (
        (f"[기존 요약]\n{previous_summary}\n\n" if previous_summary else "")
        + f"[요약할 이전 대화]\n{transcript}"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def summarize_messages(
    older: list[Any],
    previous_summary: str = "",
    llm: Any | None = None,
) -> str:
    """이전 메시지들을 요약문 1개로 만든다. llm=None 이면 규칙 기반."""
    fallback = _rule_based_summary(older, previous_summary)
    if llm is None or not older:
        return fallback

    draft = safe_structured_invoke(
        llm,
        _build_prompt(older, previous_summary),
        ConversationSummaryDraft,
        ConversationSummaryDraft(summary=fallback),
    )
    summary = (draft.summary or "").strip()
    if not summary:
        return fallback
    return summary[:_MAX_SUMMARY_CHARS].strip()


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------
def make_conversation_summary_node(
    llm: Any | None = None,
    settings: Settings | None = None,
) -> Callable[[dict], dict]:
    """LLM 을 주입한 노드를 만든다(테스트가 mock LLM 을 넣을 수 있게)."""

    def _node(state: dict) -> dict:
        return _summarize(state, llm=llm, settings=settings)

    return _node


def conversation_summary_node(state: dict) -> dict:
    """대화가 길 때만 이전 메시지를 요약해 `conversation_summary` 를 갱신한다.

    LLM 은 `build_llm()` 으로 만들고, 키가 없으면 None 이 와서 규칙 기반으로 간다.
    """
    from ...llm import build_llm  # 지연 import — provider 패키지가 없어도 동작한다.

    return _summarize(state, llm=build_llm(), settings=None)


def _summarize(
    state: dict,
    llm: Any | None,
    settings: Settings | None,
) -> dict:
    resolved = settings or get_settings()
    messages = list(state.get("messages") or [])

    if len(messages) <= resolved.summary_trigger_message_count:
        logger.debug("대화가 짧아 요약을 건너뜁니다(%d개).", len(messages))
        return {}

    keep = max(resolved.summary_keep_recent_messages, 0)
    older = messages[:-keep] if keep else messages
    if not older:
        return {}

    previous = str(state.get("conversation_summary") or "")
    summary = summarize_messages(older, previous_summary=previous, llm=llm)
    if not summary or summary == previous:
        return {}

    logger.info(
        "대화 요약 갱신 — 이전 메시지 %d개 요약, 최근 %d개 원문 유지",
        len(older),
        len(messages) - len(older),
    )
    return {"conversation_summary": summary}
