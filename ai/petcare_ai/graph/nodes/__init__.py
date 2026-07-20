"""LangGraph node 패키지 — 명세 22/23절 모듈 구조.

**지연 로딩(PEP 562 `__getattr__`)을 쓴다.** 이유 두 가지.

1. node 모듈은 서로 다른 시점에 채워진다. `__init__` 에서 전부 즉시 import 하면
   아직 없는 모듈 하나 때문에 `petcare_ai.graph.nodes` 전체가 죽는다.
2. 일부 node 는 reportlab·tavily 같은 선택 의존성을 쓴다. 실제로 그 node 를
   꺼내 쓸 때만 import 되어야 키·패키지가 없는 환경에서도 그래프 조립이 된다.

사용법은 평소와 같다.

    from petcare_ai.graph.nodes import supervisor_node          # 지연 import
    from petcare_ai.graph.nodes.supervisor import supervisor_node  # 직접 import
"""

from __future__ import annotations

import importlib
from typing import Any

# 명세 22절의 node 모듈 목록(작성 순서와 무관하게 이름만 고정한다).
NODE_MODULES: tuple[str, ...] = (
    "db_context",
    "conversation_summary",
    "fast_emergency_guard",
    "supervisor",
    "clinical_context_priority",
    "assessment",
    "risk_double_check",
    "missing_information",
    "general_chat",
    "health_response",
    "hospital_requirements",
    "hospital_search",
    "hospital_suitability",
    "document_agent",
    "email_draft",
    "output_check",
    "final_safety",
)

# 이 패키지가 확실히 제공하는 이름 → 모듈. 여기 없는 이름은 아래에서 탐색한다.
_EXPORT_MAP: dict[str, str] = {
    # db_context
    "db_context_node": "db_context",
    "make_db_context_node": "db_context",
    "set_clinical_adapter": "db_context",
    "get_clinical_adapter": "db_context",
    "needs_db_context": "db_context",
    "route_context_loaded": "db_context",
    # conversation_summary
    "conversation_summary_node": "conversation_summary",
    "make_conversation_summary_node": "conversation_summary",
    "needs_conversation_summary": "conversation_summary",
    "route_needs_summary": "conversation_summary",
    "summarize_messages": "conversation_summary",
    # fast_emergency_guard
    "fast_emergency_guard_node": "fast_emergency_guard",
    "detect_emergency_signals": "fast_emergency_guard",
    "is_critical_immediate": "fast_emergency_guard",
    "route_after_fast_emergency_guard": "fast_emergency_guard",
    "CRITICAL_SIGNALS": "fast_emergency_guard",
    "WARNING_SIGNALS": "fast_emergency_guard",
    # supervisor
    "supervisor_node": "supervisor",
    "make_supervisor_node": "supervisor",
    "evaluate_supervisor": "supervisor",
    "classify_intent_rule_based": "supervisor",
    "route_after_supervisor": "supervisor",
    # clinical_context_priority
    "clinical_context_priority_node": "clinical_context_priority",
    "make_clinical_context_priority_node": "clinical_context_priority",
    "build_clinical_context": "clinical_context_priority",
    "extract_current_observation": "clinical_context_priority",
    "select_related_diagnoses": "clinical_context_priority",
    "select_supporting_daily_entries": "clinical_context_priority",
    "detect_context_conflicts": "clinical_context_priority",
    "SOURCE_PRIORITY": "clinical_context_priority",
}

__all__ = ["NODE_MODULES", *sorted(_EXPORT_MAP)]


def __getattr__(name: str) -> Any:
    """이름을 실제로 꺼낼 때만 해당 모듈을 import 한다.

    매핑에 없는 이름(다른 node 모듈이 제공하는 함수)은 형제 모듈을 순회하며
    찾는다. 아직 작성되지 않았거나 선택 의존성이 없는 모듈은 조용히 건너뛴다 —
    그 모듈을 직접 import 하면 원래 오류를 그대로 볼 수 있다.
    """
    module_name = _EXPORT_MAP.get(name)
    if module_name is not None:
        module = importlib.import_module(f".{module_name}", __name__)
        return getattr(module, name)

    for candidate in NODE_MODULES:
        try:
            module = importlib.import_module(f".{candidate}", __name__)
        except ImportError:
            continue
        if hasattr(module, name):
            return getattr(module, name)

    raise AttributeError(
        f"petcare_ai.graph.nodes 에 '{name}' 이(가) 없습니다. "
        f"해당 node 모듈이 아직 작성되지 않았거나 이름이 다를 수 있습니다."
    )


def __dir__() -> list[str]:
    return sorted(__all__)
