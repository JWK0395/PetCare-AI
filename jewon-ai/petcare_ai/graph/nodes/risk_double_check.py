"""Risk Double Check Agent 와 Merge Risk Node (명세 23·28절).

명세 28절은 세 결과(Rule / Assessment / Double Check)를 병합하되 **최종 병합은
Python 코드로 수행**하고 낮은 위험도로 덮어쓰지 않을 것을 요구한다. 그래서:

- `risk_double_check_node` : 보수적 재확인 담당(LLM). 놓친 신호를 찾는 역할만 하며
  **위험도를 내릴 권한이 없다**. 기준선(rule ∨ assessment) 아래로는 계산 자체가
  불가능하도록 `merge_risk()` 를 통과시킨다.
- `merge_risk_node` : LLM 을 전혀 쓰지 않는 순수 Python node. `schemas.merge_risk()`
  로만 병합하고, 응급 긴급도는 누적된 red flag 를 규칙 테이블로 역산해 다시 구한다.

왜 긴급도를 '역산' 하는가: `red_flags` 는 여러 평가 node 가 병렬로 누적한 문자열
리스트라 어느 node 가 어떤 긴급도를 의도했는지가 남지 않는다. 규칙 테이블
(`assessment.RED_FLAG_RULES`)을 유일한 출처로 삼아 다시 계산하면, 평가 순서나
node 개수가 바뀌어도 같은 red flag 집합에서 항상 같은 긴급도가 나온다.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from ...llm import build_llm, safe_structured_invoke
from ...schemas import AssessmentResult, EmergencyUrgency, RiskLevel, merge_risk
from ..prompts import DOUBLE_CHECK_PROMPT
from ..state import URGENCY_PRIORITY
from .assessment import (
    build_assessment_prompt,  # 같은 context 요약을 재사용한다(프롬프트 이중화 금지)
    evaluate_rules,
    red_flag_urgency,
)

if TYPE_CHECKING:
    from ..state import PetCareState

logger = logging.getLogger(__name__)

__all__ = [
    "baseline_risk",
    "merge_urgency",
    "resolve_final_risk",
    "resolve_emergency_urgency",
    "risk_double_check_node",
    "merge_risk_node",
]


def baseline_risk(state: dict[str, Any]) -> RiskLevel:
    """재확인의 하한선 — 이 값보다 낮은 최종 위험도는 존재할 수 없다.

    Rule / Assessment 두 node 가 병렬이라 한쪽이 아직 state 에 반영되지 않았을
    가능성을 고려해 `final_risk`(누적 reducer 값)까지 함께 본다.
    """
    return merge_risk(
        state.get("rule_risk"),
        state.get("assessment_risk"),
        state.get("final_risk"),
    )


def merge_urgency(*levels: str | None) -> EmergencyUrgency:
    """응급 긴급도를 상향으로만 병합한다(none < contact_ready < critical_immediate).

    `merge_risk()` 의 긴급도 판이며, state.py 의 `escalate_urgency` reducer 와 같은
    우선순위 표(`URGENCY_PRIORITY`)를 공유해 두 곳의 규칙이 갈라지지 않게 한다.
    """
    best: str = "none"
    for level in levels:
        if level in URGENCY_PRIORITY and URGENCY_PRIORITY[level] > URGENCY_PRIORITY[best]:
            best = level
    return best  # type: ignore[return-value]


def resolve_final_risk(state: dict[str, Any]) -> RiskLevel:
    """세 평가자의 결과를 병합한다(명세 28절) — 순수 함수라 단독 테스트가 가능하다.

    `emergency_urgency` 가 이미 올라가 있으면(예: Fast Emergency Guard 가
    critical_immediate 로 판정) 위험도도 반드시 emergency 여야 한다. 긴급도와
    위험도가 어긋난 state 는 이후 분기를 통째로 망가뜨리기 때문이다.
    """
    risk = merge_risk(
        state.get("rule_risk"),
        state.get("assessment_risk"),
        state.get("double_check_risk"),
        state.get("final_risk"),
    )
    if state.get("emergency_urgency") in ("contact_ready", "critical_immediate"):
        risk = merge_risk(risk, "emergency")
    return risk


def resolve_emergency_urgency(state: dict[str, Any], final_risk: RiskLevel) -> EmergencyUrgency:
    """최종 긴급도를 정한다.

    근거는 세 가지이며 그중 가장 높은 값을 쓴다.
      1. state 에 이미 기록된 긴급도(Fast Emergency Guard·평가 node 들이 올린 값)
      2. 누적된 red flag 를 규칙 테이블로 역산한 긴급도
      3. 최종 위험도가 emergency 이면 최소 contact_ready
    """
    from_flags = merge_urgency(
        *[red_flag_urgency(str(flag)) for flag in (state.get("red_flags") or [])]
    )
    urgency = merge_urgency(state.get("emergency_urgency"), from_flags)
    if final_risk == "emergency" and urgency == "none":
        urgency = "contact_ready"
    if final_risk != "emergency":
        # 응급이 아닌데 긴급도만 남아 있으면 안내가 모순된다.
        # (위험도는 resolve_final_risk 에서 이미 emergency 로 올라갔으므로
        #  여기에 도달하는 경우는 긴급도가 none 인 경우뿐이다.)
        urgency = "none"
    return urgency


# ---------------------------------------------------------------------------
# Node — Risk Double Check Agent
# ---------------------------------------------------------------------------
def risk_double_check_node(state: dict) -> dict:
    """보수적으로 한 번 더 확인한다. LLM 이 없으면 기준선을 그대로 통과시킨다.

    LLM 없이도 이 node 는 의미가 있다. 규칙 재평가(`evaluate_rules`)를 다시 돌려
    앞 node 이후에 수집된 추가 답변(`collected_information`)까지 반영하기 때문이다.
    multi-turn interrupt 로 "사실 어제부터 계속 토했어요" 같은 답이 들어오면
    여기서 위험도가 올라간다.
    """
    baseline = baseline_risk(state)
    rules_now = evaluate_rules(state)  # interrupt 로 추가된 답변까지 반영
    baseline = merge_risk(baseline, rules_now.risk_level)

    llm = build_llm()
    if llm is None:
        return {
            "double_check_risk": baseline,
            "emergency_urgency": merge_urgency(
                state.get("emergency_urgency"), rules_now.emergency_urgency
            ),
            "red_flags": rules_now.red_flags,
            "risk_reasons": ["[재확인] LLM 없이 규칙 재평가로 확인함"],
        }

    default = AssessmentResult(
        risk_level=baseline,
        emergency_urgency=merge_urgency(
            state.get("emergency_urgency"), rules_now.emergency_urgency
        ),
        red_flags=[],
        reasons=["앞선 판정을 유지함"],
        rag_required=rules_now.rag_required,
    )

    result = safe_structured_invoke(
        llm,
        [
            ("system", DOUBLE_CHECK_PROMPT),
            ("human", build_assessment_prompt(state, rules_now)),
        ],
        AssessmentResult,
        default,
    )

    # 재확인은 올리기만 한다.
    risk = merge_risk(baseline, result.risk_level)
    urgency = merge_urgency(
        state.get("emergency_urgency"),
        rules_now.emergency_urgency,
        result.emergency_urgency,
    )
    if risk == "emergency" and urgency == "none":
        urgency = "contact_ready"

    if result.risk_level != risk:
        logger.info(
            "Risk Double Check 가 더 낮은 위험도(%s)를 반환해 %s 로 유지합니다.",
            result.risk_level,
            risk,
        )

    return {
        "double_check_risk": risk,
        "emergency_urgency": urgency,
        "red_flags": list(rules_now.red_flags) + list(result.red_flags),
        "risk_reasons": [f"[재확인] {reason}" for reason in result.reasons],
    }


# ---------------------------------------------------------------------------
# Node — Merge Risk (LLM 없음 / 명세 28절 "최종 병합은 Python 코드로")
# ---------------------------------------------------------------------------
def merge_risk_node(state: dict) -> dict:
    """세 평가 결과를 `schemas.merge_risk()` 로 병합해 최종 위험도를 확정한다.

    이 node 에는 LLM 이 없다. 위험도 병합은 판단이 아니라 **규칙 적용**이며,
    모델에 맡기면 같은 입력에 다른 분기가 나올 수 있다. 병합 결과는 graph 의
    Health / Visit / Emergency subgraph 분기를 직접 결정하므로 결정론적이어야 한다.

    state.py 의 `escalate_risk` / `escalate_urgency` reducer 덕분에 여기서 쓴 값이
    이후 어떤 node 에 의해서도 낮아지지 않는다(이중 안전장치).
    """
    final_risk = resolve_final_risk(state)
    urgency = resolve_emergency_urgency(state, final_risk)

    reason = (
        f"[병합] 규칙={state.get('rule_risk', 'normal')} / "
        f"평가={state.get('assessment_risk', 'normal')} / "
        f"재확인={state.get('double_check_risk', 'normal')} "
        f"→ 최종={final_risk} (긴급도={urgency})"
    )
    logger.debug(reason)

    return {
        "final_risk": final_risk,
        "emergency_urgency": urgency,
        "risk_reasons": [reason],
    }
