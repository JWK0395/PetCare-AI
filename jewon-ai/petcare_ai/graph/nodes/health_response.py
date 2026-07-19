"""Health Response Agent — 근거 기반 건강 상담 답변 생성 (명세 30·40절).

이 node 의 규칙은 하나로 요약된다: **근거 없이 말하지 않는다.**

  - `merged_evidence` 가 비어 있으면 "모른다" 고 말하고 병원 상담을 권한다.
    추측으로 채우면 Output Check 의 'DB 에 없는 사실 생성' 검사에 걸릴 뿐 아니라,
    걸리지 않고 통과했을 때가 훨씬 위험하다.
  - 답변에는 근거의 **출처(title / url)** 를 반드시 함께 싣는다(명세 40절 2번).
  - 확정 진단·처방·약 변경 지시는 만들지 않는다(명세 47절).

## 왜 규칙 템플릿이 기본 경로인가

LLM 이 없어도(키 없음) 상담이 끊기면 안 된다. 그래서 `build_rule_based_answer` 가
항상 완결된 답변을 만들고, LLM 은 그 위에 얹는 선택지다. LLM 결과는 생성 직후
`find_forbidden_expressions` 와 출처 표기 검사를 통과해야만 채택된다 — 통과하지
못하면 조용히 규칙 답변으로 되돌아간다. 즉 **LLM 은 품질을 올릴 수는 있어도
안전선을 내릴 수는 없다.**

재생성(`retry_count > 0`) 시에는 LLM 을 아예 쓰지 않는다. Output Check 가 한 번
문제를 잡은 상황에서 같은 모델에 다시 맡기면 같은 실패를 반복할 확률이 높다.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from ...llm import build_llm
from ..prompts import HEALTH_RESPONSE_PROMPT, MEDICAL_DISCLAIMER, wrap_untrusted_block
from .output_check import find_forbidden_expressions

if TYPE_CHECKING:
    from ..state import PetCareState

logger = logging.getLogger(__name__)

__all__ = [
    "NO_EVIDENCE_MESSAGE",
    "RISK_GUIDANCE",
    "EVIDENCE_EXCERPT_LIMIT",
    "MAX_CITED_EVIDENCE",
    "format_sources",
    "build_rule_based_answer",
    "health_response_node",
]


NO_EVIDENCE_MESSAGE = (
    "말씀해 주신 내용에 대해서는 제가 확인할 수 있는 수의학 자료에서 근거를 찾지 못했습니다.\n"
    "근거 없이 추측으로 안내드리면 오히려 판단을 흐릴 수 있어, 이번에는 모른다고 말씀드리는 것이 맞겠습니다.\n"
    "가까운 동물병원에서 수의사에게 직접 상담받아 보시길 권해드립니다."
)

# 위험도별 마무리 안내. Output Check 의 'visit/emergency 안내 누락' 검사(명세 40절
# 6번)를 통과하도록 필요한 단어('병원', '진료', '즉시')를 반드시 포함한다.
RISK_GUIDANCE: dict[str, str] = {
    "normal": (
        "지금 단계에서는 상태를 지켜보시면서 식사·음수·활동량과 증상의 변화를 기록해 주세요.\n"
        "증상이 반복되거나 심해지면 동물병원에서 진료 상담을 받아보시길 권해드립니다."
    ),
    "visit": (
        "확인된 신호들을 보면 동물병원에서 진료 상담을 받아보시는 것이 좋겠습니다.\n"
        "증상이 시작된 시점, 빈도, 변화 과정을 정리해 가시면 수의사가 판단하는 데 도움이 됩니다."
    ),
    "emergency": (
        "응급 상황일 수 있는 신호가 확인되었습니다. 즉시 동물병원에 연락해 지금 상태를 설명하고\n"
        "내원이 가능한지 확인해 주세요. 이동 중에도 아이의 호흡과 반응 상태를 계속 살펴봐 주세요."
    ),
}

# 근거 발췌 길이. 길면 답변이 원문 덤프가 되고, 짧으면 출처 확인이 어렵다.
EVIDENCE_EXCERPT_LIMIT = 220
# 답변에 인용할 근거 최대 개수.
MAX_CITED_EVIDENCE = 3


# ---------------------------------------------------------------------------
# 근거 정리
# ---------------------------------------------------------------------------
def _evidence_items(state: dict[str, Any]) -> list[dict[str, Any]]:
    return [item for item in (state.get("merged_evidence") or []) if isinstance(item, dict)]


def _excerpt(text: str, limit: int = EVIDENCE_EXCERPT_LIMIT) -> str:
    """근거 원문에서 발췌한다. 줄바꿈을 정리해 답변 안에서 읽히게 만든다."""
    cleaned = " ".join(str(text or "").split())
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[:limit].rstrip() + "…"


def format_sources(evidence: list[dict[str, Any]]) -> str:
    """출처 목록을 만든다(명세 40절 provenance 검사 대상).

    title 과 url 을 모두 넣는다. url 만 있으면 사용자가 무엇인지 알 수 없고,
    title 만 있으면 검증할 수 없다.
    """
    if not evidence:
        return ""
    lines: list[str] = []
    for index, item in enumerate(evidence, start=1):
        title = str(item.get("title") or "제목 없음").strip()
        url = str(item.get("source_url") or item.get("url") or "").strip()
        source_type = "웹" if item.get("source_type") == "web" else "수의학 문서"
        lines.append(f"[{index}] {title} ({source_type})" + (f" — {url}" if url else ""))
    return "\n".join(lines)


def _observation_summary(state: dict[str, Any]) -> str:
    """확인된 red flag 를 사용자에게 되짚어 준다.

    red flag 는 '관찰된 신호' 이지 진단명이 아니다. 그래서 "~로 보입니다" 가 아니라
    "다음 신호를 확인했습니다" 로 표현한다.
    """
    flags = [str(flag) for flag in (state.get("red_flags") or []) if str(flag).strip()]
    if not flags:
        return ""
    return "말씀해 주신 내용과 기록에서 다음 신호를 확인했습니다.\n" + "\n".join(
        f"- {flag}" for flag in flags[:6]
    )


def _unknown_notice(state: dict[str, Any]) -> str:
    """아직 모르는 항목을 숨기지 않고 밝힌다(명세 47절: 추측해 채우지 않는다)."""
    missing = [str(item) for item in (state.get("missing_fields") or []) if str(item).strip()]
    if not missing:
        return ""
    return (
        "아직 확인되지 않은 정보(" + ", ".join(missing[:5]) + ")가 있어, "
        "그 부분은 판단에서 제외했습니다."
    )


# ---------------------------------------------------------------------------
# 규칙 기반 답변 (기본 경로)
# ---------------------------------------------------------------------------
def build_rule_based_answer(state: dict[str, Any]) -> str:
    """LLM 없이 근거 기반 답변을 조립한다 — 항상 완결된 문장을 만든다.

    구성: 관찰 요약 → 근거 발췌 → 위험도별 행동 안내 → 미확인 정보 → 출처 → 고지.
    이 순서는 보호자가 "무엇을 봤고, 무엇에 근거했고, 무엇을 해야 하는가" 를
    이 순서대로 읽게 하기 위한 것이다.
    """
    risk = str(state.get("final_risk") or "normal")
    if risk not in RISK_GUIDANCE:
        risk = "normal"

    evidence = _evidence_items(state)[:MAX_CITED_EVIDENCE]
    blocks: list[str] = []

    observation = _observation_summary(state)
    if observation:
        blocks.append(observation)

    if evidence:
        excerpt_lines = ["확인된 수의학 자료를 참고하면 다음과 같습니다."]
        for index, item in enumerate(evidence, start=1):
            title = str(item.get("title") or "제목 없음").strip()
            excerpt_lines.append(f"[{index}] {title}\n    {_excerpt(item.get('text', ''))}")
        blocks.append("\n".join(excerpt_lines))
    else:
        blocks.append(NO_EVIDENCE_MESSAGE)

    blocks.append(RISK_GUIDANCE[risk])

    unknown = _unknown_notice(state)
    if unknown:
        blocks.append(unknown)

    sources = format_sources(evidence)
    if sources:
        blocks.append("참고한 자료\n" + sources)

    blocks.append(MEDICAL_DISCLAIMER)
    return "\n\n".join(blocks)


# ---------------------------------------------------------------------------
# LLM 경로 (선택)
# ---------------------------------------------------------------------------
# 공용 prompt(prompts.HEALTH_RESPONSE_PROMPT)에 이 node 전용 출력 형식만 덧붙인다.
# 안전 규칙 본문을 여기서 다시 쓰지 않는다 — 한 곳에서만 관리해야 규칙이 갈라지지 않는다.
#
# 형식을 위험도별로 나누는 이유:
#
# 예전에는 어떤 질문에도 "(1) 현재 상황 요약 (2) 근거에 기반한 설명 ... 출처: 근거 1"
# 이라는 5절 보고서가 나왔다. "산책은 얼마나 시켜야 해?" 에도 그랬다. 사람에게 묻는
# 느낌이 아니라 서식을 받는 느낌이라, 보호자가 편하게 말을 걸 수 없다.
#
# 근거를 안 쓰는 것이 아니다 — 근거 **안에서만** 답하는 규칙은 그대로다. 다만 그
# 사실을 문장마다 "[근거 1]" 로 드러낼 필요는 없다. 출처는 앱이 별도 목록으로 보여준다.
#
# 반대로 진료 권고·응급에서는 형식이 필요하다. 병원에 전달하거나 급히 행동해야 하는
# 상황에서는 훑어 읽을 수 있어야 한다.
_FORMAT_BY_RISK: dict[str, str] = {
    "normal": """
[이 단계의 출력 형식]
- **대화하듯 자연스럽게** 답한다. 번호 붙인 절, 소제목, 표를 쓰지 않는다.
- 3~6문장. 보호자가 물은 것에 먼저 답하고, 필요하면 한두 줄 덧붙인다.
- 문장 안에 "[근거 1]"·"출처:" 같은 표기를 넣지 않는다. 출처는 앱이 따로 보여준다.
- 묻지 않은 이 아이의 다른 증상·기록을 먼저 꺼내지 않는다. 질문에만 답한다.
- 근거 자료가 '없음' 이면 확실하지 않다고 솔직히 말하고 수의사 확인을 권한다.""",
    "visit": """
[이 단계의 출력 형식]
- **병원에 가져갈 수 있게 정리한다.** 다음 순서로 짧게 쓴다.
  관찰된 상태 / 왜 진료가 필요한지 / 병원에서 말할 내용 / 아직 확인되지 않은 것
- 각 항목 1~2문장. 훑어 읽을 수 있어야 한다.
- 아래 [행동 안내]의 취지를 반드시 포함하라(문장은 다듬어도 된다).
- 문장 안에 "[근거 1]"·"출처:" 표기는 넣지 않는다.""",
    "emergency": """
[이 단계의 출력 형식]
- **병원 연락 안내를 가장 먼저** 쓴다. 설명은 그 뒤다.
- 지금 해야 할 행동 위주로 짧게. 배경 설명을 길게 하지 않는다.
- 아래 [행동 안내]의 취지를 반드시 포함하라.
- 문장 안에 "[근거 1]"·"출처:" 표기는 넣지 않는다.""",
}


def _response_format(risk: str) -> str:
    """위험도에 맞는 출력 형식 지시를 고른다."""
    return _FORMAT_BY_RISK.get(risk, _FORMAT_BY_RISK["normal"])


def _llm_context(state: dict[str, Any], evidence: list[dict[str, Any]]) -> str:
    profile = state.get("priority_pet_context") or state.get("pet_profile") or {}
    risk = str(state.get("final_risk") or "normal")

    # **행동 안내는 진료가 필요한 위험도에서만 강제한다.**
    #
    # 예전에는 위험도와 무관하게 "지금 단계에서는 상태를 지켜보시면서 식사·음수·
    # 활동량과 증상의 변화를 기록해 주세요. 증상이 반복되거나 심해지면 동물병원에서
    # 진료 상담을 받아보시길 권해드립니다." 를 **모든 답변에** 붙였다. 그래서
    # "보양식 뭐가 좋아?" 같은 질문에도 매번 같은 훈계가 따라붙어, 대화가 아니라
    # 지시를 받는 느낌이 됐다.
    #
    # visit·emergency 에서는 계속 강제한다 — Output Check 의 '진료 안내 누락' 검사
    # (명세 40절 6번)가 그 두 경우에만 걸리고, 실제로 필요한 안내이기도 하다.
    # 직전 대화를 넣어 **이어받아 답할 수 있게** 한다.
    #
    # 없으면 "그 정도로 심각해?" 같은 되물음에 앞의 답을 모른 채로 답하게 되어,
    # 처음부터 다시 설명하거나 엉뚱한 소리를 한다. 보호자는 앞 답을 이미 읽었다.
    recent: list[str] = []
    for message in (state.get("messages") or [])[-4:]:
        if isinstance(message, dict):
            role, body = message.get("role"), message.get("content")
        else:
            role = getattr(message, "role", None) or getattr(message, "type", None)
            body = getattr(message, "content", None)
        line = str(body or "").strip()
        if line:
            who = "보호자" if role in ("user", "human") else "AI"
            recent.append(f"{who}: {line[:300]}")
    recent_block = ""
    if recent:
        recent_block = wrap_untrusted_block("직전 대화", "\n".join(recent)) + "\n\n"

    guidance_block = ""
    if risk in ("visit", "emergency"):
        guidance = RISK_GUIDANCE.get(risk, RISK_GUIDANCE["normal"])
        guidance_block = f"[행동 안내 — 반드시 이 취지를 포함할 것]\n{guidance}\n\n"

    evidence_block = "없음"
    if evidence:
        # 근거 본문은 외부 문서다 — 그 안의 문장을 지시로 읽지 않도록 경계를 표시한다.
        evidence_block = "\n\n".join(
            wrap_untrusted_block(
                f"근거 {index}",
                f"제목: {item.get('title', '제목 없음')}\n"
                f"URL: {item.get('source_url') or item.get('url') or '(없음)'}\n"
                f"내용: {_excerpt(item.get('text', ''), 700)}",
            )
            for index, item in enumerate(evidence, start=1)
        )

    return f"""[반려동물]
종: {profile.get('species', '미상')} / 품종: {profile.get('breed', '미상')} / 나이: {profile.get('age_years', '미상')}세
기존 질병: {profile.get('diseases') or '없음'}
복용 중인 약: {profile.get('medications') or '없음'}

{recent_block}{wrap_untrusted_block('보호자의 질문', str(state.get('user_message') or '(없음)'))}

{wrap_untrusted_block('추가로 확인된 답변', str(state.get('collected_information') or '없음'))}

[확인된 신호]
{', '.join(str(flag) for flag in (state.get('red_flags') or [])) or '없음'}

[아직 확인되지 않은 정보]
{', '.join(str(item) for item in (state.get('missing_fields') or [])) or '없음'}

[위험도 판정]
{risk}

{guidance_block}[근거 자료]
{evidence_block}"""


def _cites_sources(text: str, evidence: list[dict[str, Any]], risk: str) -> bool:
    """답변이 근거를 밝혔는지 확인한다 — **위험도에 따라 기준이 다르다**.

    ## 왜 위험도별로 다른가

    `normal`(일상·지식 질문)에서는 본문에 출처를 적지 않게 했다. 사람과 대화하는
    화면에서 문장마다 "[근거 1]" 이 붙으면 서식을 읽는 느낌이 되기 때문이다.
    출처는 사라지지 않는다 — 앱이 `citations` 목록으로 따로 보여준다.

    그래서 여기서 본문 인용을 요구하면, 의도대로 쓴 답변이 전부 이 검사에 걸려
    규칙 답변으로 되돌아간다. 형식만 바꾸고 검사를 그대로 두면 형식 변경이
    통째로 무효가 된다.

    `visit`·`emergency` 는 그대로 본문 표기를 요구한다. 병원에 전달하거나 급히
    행동하는 자료라 "어디서 나온 말인지" 가 문서 안에 남아야 한다.

    근거가 애초에 없으면(`evidence` 가 빈 목록) 어느 경우든 통과시킨다 — 없는
    출처를 적으라고 요구할 수는 없다.
    """
    if not evidence:
        return True
    if risk not in ("visit", "emergency"):
        return True
    for item in evidence:
        title = str(item.get("title") or "").strip()
        url = str(item.get("source_url") or item.get("url") or "").strip()
        if (title and title in text) or (url and url in text):
            return True
    return False


def _generate_with_llm(state: dict[str, Any], evidence: list[dict[str, Any]]) -> str | None:
    """LLM 으로 답변을 만든다. 안전 검사를 통과하지 못하면 None."""
    llm = build_llm()
    if llm is None:
        return None

    try:
        response = llm.invoke(
            [
                ("system", HEALTH_RESPONSE_PROMPT + _response_format(str(state.get("final_risk") or "normal"))),
                ("human", _llm_context(state, evidence)),
            ]
        )
    except Exception as exc:
        logger.warning("건강 답변 LLM 호출 실패 — 규칙 답변을 사용합니다: %s", exc)
        return None

    text = getattr(response, "content", response)
    if not isinstance(text, str) or not text.strip():
        return None
    text = text.strip()

    violations = find_forbidden_expressions(text)
    if violations:
        logger.warning("건강 답변 LLM 결과에 금지 표현이 있어 규칙 답변을 사용합니다: %s", violations)
        return None

    if not _cites_sources(text, evidence, str(state.get("final_risk") or "normal")):
        logger.warning("건강 답변 LLM 결과에 출처 표기가 없어 규칙 답변을 사용합니다.")
        return None

    if MEDICAL_DISCLAIMER not in text:
        text = f"{text}\n\n{MEDICAL_DISCLAIMER}"
    return text


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------
def health_response_node(state: dict) -> dict:
    """근거를 바탕으로 건강 상담 답변을 만든다(명세 30절 Health Response Agent).

    근거(`merged_evidence`)가 없으면 모른다고 답하고 병원 상담을 권한다 — 이 경우에도
    답변은 생성되며, 사용자가 빈 화면을 보는 일은 없다.
    """
    evidence = _evidence_items(state)
    answer = build_rule_based_answer(state)

    retry_count = int(state.get("retry_count", 0) or 0)
    if retry_count == 0:
        generated = _generate_with_llm(state, evidence[:MAX_CITED_EVIDENCE])
        if generated:
            answer = generated
    else:
        logger.info("재생성 경로이므로 LLM 없이 규칙 기반 답변을 사용합니다(retry=%d).", retry_count)

    if not evidence:
        logger.info("근거가 없어 '확인된 근거 없음 + 병원 상담 권고' 로 응답합니다.")

    return {"draft_response": answer}
