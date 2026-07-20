"""LangGraph 상담 파이프라인 패키지.

## eager / lazy 를 나눈 이유

`state.py` 와 `prompts.py` 는 표준 라이브러리와 같은 패키지 내부만 import 하므로
어디서든 안전하게 불러올 수 있다. 그래서 여기서 바로 노출한다.

반면 `builder.py` / `routers.py` / `nodes/` / `subgraphs/` 는 `langgraph` 를 비롯해
RAG·PDF 계층까지 끌어온다. 이것들을 최상단에서 import 하면
`from petcare_ai.graph.state import PetCareState` 한 줄이 무거운 의존성 전체를
요구하게 되고, 그중 하나라도 미설치면 State 정의조차 못 읽는다.
그래서 PEP 562 `__getattr__` 로 **접근하는 순간에만** 불러온다.

사용 예:

    from petcare_ai.graph import PetCareState, make_initial_state   # 가벼움
    from petcare_ai.graph import builder                            # 이때 langgraph 로드
"""

from __future__ import annotations

import importlib
from typing import Any

from .prompts import (
    ALL_PROMPTS,
    HOSPITAL_VERIFICATION_NOTICE,
    MEDICAL_DISCLAIMER,
    SAFETY_RULES,
    wrap_untrusted_block,
)
from .state import (
    URGENCY_PRIORITY,
    PetCareState,
    Replace,
    ReplaceDict,
    build_trace_metadata,
    escalate_risk,
    escalate_urgency,
    make_initial_state,
    make_message,
)

# 속성 이름 -> 지연 로드할 하위 모듈
_LAZY_SUBMODULES: dict[str, str] = {
    "builder": ".builder",
    "routers": ".routers",
    "nodes": ".nodes",
    "subgraphs": ".subgraphs",
    "prompts": ".prompts",
    "state": ".state",
}

# 하위 모듈 안의 함수/클래스를 패키지 수준에서 바로 쓰고 싶을 때의 매핑.
# builder 작성자가 어떤 이름을 쓰든 하나는 걸리도록 후보를 넉넉히 둔다.
_LAZY_ATTRS: dict[str, str] = {
    "build_graph": ".builder",
    "build_chat_graph": ".builder",
    "compile_graph": ".builder",
    "run_chat": ".builder",
    "PetCareGraph": ".builder",
}


def __getattr__(name: str) -> Any:
    """접근 시점에 하위 모듈을 로드한다(PEP 562).

    실패해도 원인을 감추지 않는다 — langgraph 미설치인지, 아직 파일이 없는지
    한국어로 알려 준다. 조용히 None 을 돌려주면 graph 가 절반만 동작하는 상태로
    테스트를 통과해 버린다.
    """
    if name in _LAZY_SUBMODULES:
        return importlib.import_module(_LAZY_SUBMODULES[name], __name__)

    if name in _LAZY_ATTRS:
        module_name = _LAZY_ATTRS[name]
        try:
            module = importlib.import_module(module_name, __name__)
        except ImportError as exc:
            raise AttributeError(
                f"petcare_ai.graph.{name} 을 불러오지 못했습니다 "
                f"({module_name} 로드 실패: {exc}). langgraph 설치 여부와 "
                f"해당 모듈 존재 여부를 확인하세요."
            ) from exc
        try:
            return getattr(module, name)
        except AttributeError as exc:
            raise AttributeError(
                f"{module_name} 에 {name} 이(가) 정의되어 있지 않습니다."
            ) from exc

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(__all__) | set(_LAZY_SUBMODULES) | set(_LAZY_ATTRS))


__all__ = [
    # state
    "PetCareState",
    "make_initial_state",
    "make_message",
    "build_trace_metadata",
    "Replace",
    "ReplaceDict",
    "URGENCY_PRIORITY",
    "escalate_risk",
    "escalate_urgency",
    # prompts
    "SAFETY_RULES",
    "ALL_PROMPTS",
    "HOSPITAL_VERIFICATION_NOTICE",
    "MEDICAL_DISCLAIMER",
    "wrap_untrusted_block",
]
