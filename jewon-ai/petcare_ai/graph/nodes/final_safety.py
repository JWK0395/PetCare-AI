"""Final Safety Agent · Safe Fallback · Build Result Node (명세 39·40절).

Output Check 가 "무엇이 잘못됐는지" 를 판정한다면, Final Safety 는 **사용자에게
실제로 나가는 문장**을 마지막으로 책임진다. 명세 40절은 특히 `visit`,
`emergency`, 재생성된 답변에 대해 한 번 더 확인할 것을 요구한다.

두 단계로 나눈 이유:
  - `final_safety_node` : 통과시킬지 / 안전 문구로 대체할지 결정하고 `final_response`
    를 확정한다. 검사에 걸리면 답변을 **버리고** 안전 문구로 바꾼다. 부분 수정은
    하지 않는다 — 위험한 문장을 정규식으로 고쳐 쓰는 것은 더 위험하다.
  - `safe_fallback_node` : Output Check 가 `fallback` 으로 라우팅했을 때의 종착점.

두 경로 모두에서 **응급 UI action 은 절대 지우지 않는다.** 문장이 부적절해도
`CALL_HOSPITAL` 버튼까지 사라지면 사용자가 잃는 것이 훨씬 크다.

`build_result_node` 는 State 를 `ChatGraphResult`(명세 39절)로 조립하는 순수
매핑 node 다. LLM 을 쓰지 않는다.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from ...schemas import (
    RISK_PRIORITY,
    ChatGraphResult,
    EmailDraft,
    FinalEvidence,
    HospitalSuitabilityResult,
)
from ..prompts import MEDICAL_DISCLAIMER
from ..state import URGENCY_PRIORITY, build_trace_metadata
from .output_check import find_forbidden_expressions

if TYPE_CHECKING:
    from ..state import PetCareState

logger = logging.getLogger(__name__)

__all__ = [
    "SAFE_FALLBACK_BASE",
    "SAFE_FALLBACK_BY_RISK",
    "safe_fallback_message",
    "needs_extra_safety_review",
    "final_safety_node",
    "safe_fallback_node",
    "build_chat_graph_result",
    "build_result_node",
]


# ---------------------------------------------------------------------------
# 안전 fallback 문구
# ---------------------------------------------------------------------------
SAFE_FALLBACK_BASE = (
    "죄송합니다. 지금 확인된 정보만으로는 안전하게 답변드리기 어렵습니다.\n"
    "정확하지 않은 안내가 반려동물에게 해가 될 수 있어, 추측으로 말씀드리지 않겠습니다."
)

# 위험도별로 뒤에 붙는 행동 안내. 문구는 Output Check 의 'visit/emergency 안내 누락'
# 검사(명세 40절 6번)를 그대로 통과하도록 '병원'·'진료'·'즉시' 를 포함한다.
SAFE_FALLBACK_BY_RISK: dict[str, str] = {
    "normal": (
        "증상이 이어지거나 심해지면 가까운 동물병원에서 진료 상담을 받아보시길 권해드립니다."
    ),
    "visit": (
        "확인된 신호로 보아 동물병원 진료 상담을 받아보시길 권해드립니다. "
        "증상이 시작된 시점과 변화 과정을 메모해 두었다가 수의사에게 전달해 주세요."
    ),
    "emergency": (
        "다만 응급 가능성이 있는 신호가 확인되었으므로, 즉시 동물병원에 연락해 "
        "지금 상태를 설명하고 내원 가능 여부를 확인해 주세요."
    ),
}


def safe_fallback_message(risk_level: str | None) -> str:
    """위험도에 맞는 안전 fallback 문구를 만든다.

    답변 품질을 포기하더라도 **행동 안내(병원 연락·진료 권고)는 남긴다.** 아무것도
    안내하지 않는 fallback 은 사용자를 방치하는 것과 같다.
    """
    risk = risk_level if risk_level in RISK_PRIORITY else "normal"
    return f"{SAFE_FALLBACK_BASE}\n\n{SAFE_FALLBACK_BY_RISK[risk]}\n\n{MEDICAL_DISCLAIMER}"


# ---------------------------------------------------------------------------
# Final Safety
# ---------------------------------------------------------------------------
def needs_extra_safety_review(state: dict[str, Any]) -> bool:
    """한 번 더 볼 대상인지 판정한다(명세 40절: visit·emergency·재생성 답변).

    나머지 답변도 검사는 하지만, 이 판정이 True 면 **판정 기준을 더 엄격하게**
    적용한다(안내 누락도 fallback 사유가 된다).
    """
    return (
        state.get("final_risk") in ("visit", "emergency")
        or int(state.get("retry_count", 0) or 0) > 0
        or str(state.get("emergency_urgency") or "none") != "none"
    )


def _missing_risk_guidance(state: dict[str, Any], text: str) -> list[str]:
    """위험도에 맞는 행동 안내가 문장에 남아 있는지 확인한다."""
    risk = str(state.get("final_risk") or "normal")
    problems: list[str] = []
    if risk == "visit" and not ("병원" in text and any(w in text for w in ("진료", "방문", "내원", "상담"))):
        problems.append("병원 진료 권고 문구 누락")
    if risk == "emergency" and not (
        "병원" in text and any(w in text for w in ("즉시", "지금 바로", "응급", "서둘러"))
    ):
        problems.append("응급 대응 안내 누락")
    return problems


def final_safety_node(state: dict) -> dict:
    """사용자에게 나갈 최종 문장을 확정한다.

    검사에 걸리면 답변 전체를 안전 문구로 **교체**한다. 문제가 된 문장만 지우면
    앞뒤 맥락이 깨진 채로 남아 오히려 오해를 부른다.

    `fallback_used=True` 를 남겨 LangSmith trace 와 테스트가 "안전 장치가 실제로
    작동했는지" 를 확인할 수 있게 한다(명세 42·43절).
    """
    draft = str(state.get("draft_response") or "")
    strict = needs_extra_safety_review(state)

    problems: list[str] = find_forbidden_expressions(draft)
    if not draft.strip():
        problems.append("답변 본문이 비어 있음")
    if strict:
        problems.extend(_missing_risk_guidance(state, draft))

    if problems:
        logger.warning("Final Safety 가 답변을 안전 문구로 대체합니다: %s", problems)
        return {
            "final_response": safe_fallback_message(state.get("final_risk")),
            "fallback_used": True,
            "validation_errors": [f"[최종안전] {problem}" for problem in problems],
        }

    final_text = draft

    # 면책 문구는 **건강 안내를 한 답변에만** 붙인다.
    #
    # 예전에는 모든 답변에 붙어서, "우리 강아지 귀여워" → "네, 잘 들었어요.
    # ... 이 안내는 수의사의 진료를 대체하지 않습니다. 최종 판단은 반드시 수의사가
    # 합니다." 가 됐다. 잡담에 의료 면책을 다는 것은 어색할 뿐 아니라, 매번 반복되면
    # 정작 필요한 자리에서 눈에 들어오지 않는다.
    #
    # 판단 기준은 intent 다 — 인사·잡담(general_chat)과 범위 밖(unsupported)은
    # 의료 안내가 아니다. 지식 질문·증상 상담·응급은 전부 붙인다.
    needs_disclaimer = str(state.get("intent") or "") not in ("general_chat", "unsupported")
    if needs_disclaimer and MEDICAL_DISCLAIMER not in final_text:
        final_text = f"{final_text}\n\n{MEDICAL_DISCLAIMER}"

    return {"final_response": final_text, "fallback_used": False}


def safe_fallback_node(state: dict) -> dict:
    """Output Check 가 `fallback` 으로 보냈을 때의 종착점.

    UI action 은 건드리지 않는다. state.py 의 `merge_ui_actions` reducer 가 누적형
    이므로 여기서 아무것도 쓰지 않으면 앞서 준비된 `CALL_HOSPITAL` 등이 그대로
    유지된다(명세 43절 '즉시 위급' 기대: 정보가 부족해도 전화 action 은 남는다).
    """
    logger.info(
        "안전 fallback 응답을 사용합니다 (risk=%s, errors=%d)",
        state.get("final_risk"),
        len(state.get("validation_errors") or []),
    )
    return {
        "final_response": safe_fallback_message(state.get("final_risk")),
        "fallback_used": True,
    }


# ---------------------------------------------------------------------------
# Build Result (명세 39절)
# ---------------------------------------------------------------------------
def _coerce_list(raw: Any, model: type, label: str) -> list[Any]:
    """dict 리스트를 Pydantic 모델 리스트로 변환한다 — 실패한 항목만 건너뛴다.

    한 건이 깨졌다고 전체 응답을 실패시키지 않는다. 대신 경고를 남겨 원인을
    추적할 수 있게 한다(조용한 데이터 손실 방지).
    """
    results: list[Any] = []
    for item in raw or []:
        if isinstance(item, model):
            results.append(item)
            continue
        if not isinstance(item, dict):
            logger.warning("%s 항목 형식이 올바르지 않아 제외합니다: %r", label, item)
            continue
        try:
            results.append(model.model_validate(item))
        except Exception as exc:
            logger.warning("%s 항목을 변환하지 못해 제외합니다: %s", label, exc)
    return results


def build_chat_graph_result(
    state: dict[str, Any],
    message: str | None = None,
) -> ChatGraphResult:
    """State 를 명세 39절 `ChatGraphResult` 로 조립한다(순수 함수).

    Output Check(9번 schema 검사)가 이 함수를 미리 호출해 조립 가능 여부를 확인하고,
    Build Result Node 가 실제 결과를 만들 때 다시 호출한다. 조립 규칙이 한 곳에만
    있어야 "검사는 통과했는데 조립에서 터지는" 상황이 생기지 않는다.

    `ui_actions` 는 type 이 없는 항목을 걸러낸다. Android 가 type 으로 분기하므로
    type 없는 action 은 조용히 무시되느니 여기서 제거하는 편이 안전하다.
    """
    text = message if message is not None else (
        state.get("final_response") or state.get("draft_response") or ""
    )
    if not str(text).strip():
        text = safe_fallback_message(state.get("final_risk"))

    risk = state.get("final_risk")
    if risk not in RISK_PRIORITY:
        risk = "normal"

    urgency = state.get("emergency_urgency")
    if urgency not in URGENCY_PRIORITY:
        urgency = "none"

    email_draft: EmailDraft | None = None
    raw_email = state.get("email_draft")
    if isinstance(raw_email, EmailDraft):
        email_draft = raw_email
    elif isinstance(raw_email, dict):
        try:
            email_draft = EmailDraft.model_validate(raw_email)
        except Exception as exc:
            logger.warning("email_draft 를 변환하지 못했습니다: %s", exc)

    ui_actions = [
        action
        for action in (state.get("ui_actions") or [])
        if isinstance(action, dict) and action.get("type")
    ]

    return ChatGraphResult(
        message=str(text),
        risk_level=risk,  # type: ignore[arg-type]
        emergency_urgency=urgency,  # type: ignore[arg-type]
        missing_information=[str(item) for item in (state.get("missing_fields") or [])],
        hospitals=_coerce_list(
            state.get("hospital_results"), HospitalSuitabilityResult, "hospital_results"
        ),
        pdf_path=state.get("pdf_path"),
        email_draft=email_draft,
        ui_actions=ui_actions,
        evidence=_coerce_list(state.get("merged_evidence"), FinalEvidence, "merged_evidence"),
        trace_metadata=build_trace_metadata(state),  # type: ignore[arg-type]
    )


def build_result_node(state: dict) -> dict:
    """최종 결과를 조립해 State 에 남긴다(명세 24절 Build Result → END).

    `final_response` 를 다시 쓰는 이유: fallback 경로에서도 이 값이 반드시 채워져
    있어야 호출자가 `state["final_response"]` 한 곳만 보면 되기 때문이다.
    `trace_metadata` 는 명세 42절 규칙대로 원문·개인 식별정보를 담지 않는다.
    """
    result = build_chat_graph_result(state)
    return {
        "final_response": result.message,
        "trace_metadata": result.trace_metadata,
    }
