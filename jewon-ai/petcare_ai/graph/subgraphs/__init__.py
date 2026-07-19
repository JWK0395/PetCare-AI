"""서브그래프 패키지 (명세 30·31·32절) — 공용 의존성 컨테이너와 node 해석기.

## 왜 팩토리(`build_xxx_subgraph(deps)`) 형태인가

LangGraph 는 **compile 된 그래프를 그대로 다른 그래프의 node 로 끼울 수 있다.**
(참고문서: subgraphs). 그래서 각 서브그래프는 모듈 최상단에서 즉시 compile 하지
않고 팩토리로 만든다. 이유 세 가지.

1. compile 시점에 node 함수가 전부 필요하다. 모듈 import 시점에 compile 하면
   아직 작성 중인 node 하나 때문에 패키지 전체가 죽는다.
2. LLM·RAG 서비스·Tavily client 를 **주입**해야 테스트가 mock 을 넣을 수 있다
   (구현 가이드 4절). 모듈 전역 compile 은 주입 지점을 없앤다.
3. 부모 그래프가 checkpointer 를 갖고 있으면 서브그래프는 그것을 상속한다.
   서브그래프를 미리 compile 해 두면 상속 관계를 부모가 제어할 수 없다.

## 이 파일이 담는 것

- `SubgraphDeps`: 세 서브그래프가 공유하는 주입 컨테이너
- `resolve_node()` / `resolve_optional_node()`: `nodes/` 패키지에서 node 함수를
  **지연 해석**한다. node 파일들이 동시에 작성되고 있으므로, 이름 후보를 여러 개
  받아 먼저 발견되는 것을 쓰고 못 찾으면 **한국어로 명확히** 실패한다.
- `make_missing_information_gate()`: 되묻기 횟수를 세는 공용 wrapper(명세 29절).
- `strip_internal_keys()`: `collected_information` 에 섞인 내부 key 를 걸러낸다.

세 서브그래프 모두 `graph/state.py` 의 `PetCareState` 를 그대로 쓴다. 서브그래프
전용 State 를 따로 만들지 않는 이유는, 부모와 같은 스키마여야 값이 자동으로
오가고 reducer(위험도 상향 전용, ui_actions 누적)가 서브그래프 안에서도 똑같이
동작하기 때문이다.
"""

from __future__ import annotations

import importlib
import logging
from dataclasses import dataclass, field
from typing import Any, Callable

from ..routers import (
    MAX_MISSING_INFORMATION_ROUNDS,
    MISSING_INFO_ROUNDS_KEY,
    missing_information_rounds,
)

logger = logging.getLogger(__name__)

NodeFn = Callable[[dict], dict]

__all__ = [
    "SubgraphDeps",
    "NodeFn",
    "resolve_node",
    "resolve_optional_node",
    "make_missing_information_gate",
    "with_sanitized_collected",
    "strip_internal_keys",
    "MAX_MISSING_INFORMATION_ROUNDS",
    "MISSING_INFO_ROUNDS_KEY",
    "build_health_subgraph",
    "build_visit_subgraph",
    "build_emergency_subgraph",
]


# ---------------------------------------------------------------------------
# 주입 컨테이너
# ---------------------------------------------------------------------------
@dataclass
class SubgraphDeps:
    """서브그래프가 필요로 하는 외부 의존성을 한 번에 넘기는 컨테이너.

    모든 필드가 선택적이며, 전부 비워 두면(`SubgraphDeps()`) **LLM 도 API 키도 없이**
    규칙 기반으로 끝까지 도는 기본 경로가 된다(구현 가이드 0-1절).

    Attributes:
        settings: 전역 설정. 없으면 `get_settings()`.
        llm: `llm.build_llm()` 결과. `None` 이면 규칙 기반 경로.
        rag_service: `rag.service.VeterinaryRagService`. 없으면 지연 생성한다.
        hospital_search: `rag.tavily_vet_search.HospitalSearchService`(mock 주입용).
        node_overrides: graph node 이름 → 함수. 테스트가 특정 node 만 바꿔 끼울 때.
        allow_missing_nodes: `True` 면 아직 없는 node 를 **경고만 남기는 빈 node**
            로 대체한다. 기본 `False` — 조용히 반쪽짜리 그래프가 만들어져 테스트를
            통과해 버리는 것이 가장 위험하기 때문이다.
        checkpointer: 서브그래프를 독립 실행할 때만 지정한다. `None` 이면 부모
            그래프의 checkpointer 를 상속한다(권장).
    """

    settings: Any = None
    llm: Any = None
    rag_service: Any = None
    hospital_search: Any = None
    node_overrides: dict[str, NodeFn] = field(default_factory=dict)
    allow_missing_nodes: bool = False
    checkpointer: Any = None

    # -- 지연 해석되는 협력 객체 -------------------------------------------
    def resolved_settings(self) -> Any:
        """Settings 를 확정한다(주입 없으면 전역 설정)."""
        if self.settings is None:
            from ...config import get_settings  # noqa: PLC0415

            self.settings = get_settings()
        return self.settings

    def resolved_rag_service(self) -> Any:
        """RAG 파사드를 확정한다.

        `VeterinaryRagService` 는 store·web_search·validator·evaluator 를 전부
        지연 생성하므로, 여기서 만들어 두어도 실제로 검색할 때까지 faiss 나
        tavily 를 import 하지 않는다.
        """
        if self.rag_service is None:
            from ...rag.service import VeterinaryRagService  # noqa: PLC0415

            self.rag_service = VeterinaryRagService(
                settings=self.resolved_settings(), llm=self.llm
            )
        return self.rag_service

    def resolved_hospital_search(self) -> Any:
        """병원 검색 서비스를 확정한다 — 수의학 지식 검색과 **별도 class** 다(명세 34절)."""
        if self.hospital_search is None:
            from ...rag.tavily_vet_search import HospitalSearchService  # noqa: PLC0415

            self.hospital_search = HospitalSearchService(self.resolved_settings())
        return self.hospital_search

    def compile_kwargs(self) -> dict[str, Any]:
        """`StateGraph.compile()` 에 넘길 인자를 만든다."""
        return {"checkpointer": self.checkpointer} if self.checkpointer is not None else {}


# ---------------------------------------------------------------------------
# node 해석
# ---------------------------------------------------------------------------
#: `factory_arg` 를 지정하지 않았을 때의 표식 — `deps.llm` 을 넘긴다는 뜻이다.
#: `None` 을 sentinel 로 쓸 수 없다. `factory_arg=None`(=주입 없음)은 유효한 값이다.
_USE_LLM: Any = object()


def _lookup(
    modules: tuple[str, ...],
    names: tuple[str, ...],
    factories: tuple[str, ...],
    factory_arg: Any,
) -> tuple[NodeFn | None, list[str]]:
    """`nodes/` 하위 모듈에서 node 함수를 찾는다. 반환: (함수, 실패 사유 목록).

    `factories` 는 `make_xxx_node(...)` 처럼 **함수를 만들어 주는** 이름이다.
    plain node 보다 factory 를 먼저 찾는다 — factory 가 있는 모듈은 의존성 주입을
    지원한다는 뜻이라 테스트가 mock 을 넣을 수 있는 상위 호환 경로이기 때문이다.

    `factory_arg` 를 **호출자가 명시**해야 하는 이유: factory 마다 받는 것이 다르다.
    `make_hospital_suitability_node(llm)` 은 LLM 을, `make_hospital_search_node(service)`
    는 검색 서비스를 받는다. 무조건 `llm` 을 넘기면 Tavily 자리에 LLM 객체가
    꽂히는 조용한 오류가 난다.
    """
    problems: list[str] = []
    for module_name in modules:
        try:
            module = importlib.import_module(f".{module_name}", "petcare_ai.graph.nodes")
        except ImportError as exc:
            problems.append(f"nodes.{module_name} import 실패: {exc}")
            continue
        for factory_name in factories:
            factory = getattr(module, factory_name, None)
            if callable(factory):
                return factory(factory_arg), problems
        for func_name in names:
            func = getattr(module, func_name, None)
            if callable(func):
                return func, problems
        problems.append(
            f"nodes.{module_name} 에 {names + factories} 중 어떤 이름도 없습니다."
        )
    return None, problems


def _placeholder(key: str, problems: list[str]) -> NodeFn:
    """아직 없는 node 자리를 채우는 빈 node — `allow_missing_nodes=True` 일 때만."""

    def _node(state: dict) -> dict:
        logger.error(
            "[미구현 node] '%s' 가 아직 없어 아무 일도 하지 않고 통과합니다. %s",
            key,
            " / ".join(problems),
        )
        return {}

    return _node


def resolve_node(
    deps: SubgraphDeps,
    key: str,
    modules: tuple[str, ...],
    names: tuple[str, ...],
    factories: tuple[str, ...] = (),
    factory_arg: Any = _USE_LLM,
) -> NodeFn:
    """graph node 하나를 해석한다. 순서: 주입 override → `nodes/` 탐색 → 실패.

    Args:
        deps: 주입 컨테이너. `node_overrides[key]` 가 있으면 그것이 최우선이다.
        key: graph 안에서 쓰는 node 이름(override key 이자 오류 메시지용).
        modules: 탐색할 `petcare_ai.graph.nodes` 하위 모듈 이름들.
        names: 찾을 node 함수 이름 후보.
        factories: `make_xxx_node(...)` 형태의 팩토리 이름 후보.
        factory_arg: 팩토리에 넘길 값. 생략하면 `deps.llm`.

    Raises:
        RuntimeError: 어디서도 찾지 못했고 `allow_missing_nodes=False` 인 경우.
            조용히 넘어가면 PDF 없는 응급 결과나 근거 없는 답변이 정상처럼
            만들어지므로, 여기서 반드시 크게 실패시킨다.
    """
    override = deps.node_overrides.get(key)
    if override is not None:
        return override

    resolved_arg = deps.llm if factory_arg is _USE_LLM else factory_arg
    func, problems = _lookup(modules, names, factories, resolved_arg)
    if func is not None:
        return func

    if deps.allow_missing_nodes:
        return _placeholder(key, problems)

    raise RuntimeError(
        f"graph node '{key}' 를 찾지 못했습니다. "
        f"후보 모듈={list(modules)}, 후보 이름={list(names + factories)}. "
        f"상세: {' / '.join(problems) or '없음'}. "
        f"해당 node 모듈을 작성했는지 확인하거나, "
        f"SubgraphDeps(node_overrides={{'{key}': 함수}}) 로 직접 주입하세요."
    )


def resolve_optional_node(
    deps: SubgraphDeps,
    key: str,
    modules: tuple[str, ...],
    names: tuple[str, ...],
    fallback: NodeFn,
    factories: tuple[str, ...] = (),
    factory_arg: Any = _USE_LLM,
) -> NodeFn:
    """`nodes/` 에 있으면 그것을, 없으면 `fallback` 을 쓴다.

    `resolve_node` 와 달리 실패해도 예외를 던지지 않는다. **서브그래프 안에
    안전한 기본 구현이 이미 있는 node** 에만 쓴다(Packet Validator, PDF Generator
    처럼 명세 22절 모듈 목록에 전용 파일이 없는 것들). 전용 node 파일이 나중에
    생기면 자동으로 그쪽이 이긴다 — 로직이 두 벌 살아 있는 상태를 막는다.
    """
    override = deps.node_overrides.get(key)
    if override is not None:
        return override

    resolved_arg = deps.llm if factory_arg is _USE_LLM else factory_arg
    func, problems = _lookup(modules, names, factories, resolved_arg)
    if func is not None:
        return func

    logger.debug("node '%s' 는 서브그래프 기본 구현을 사용합니다. (%s)", key, problems)
    return fallback


# ---------------------------------------------------------------------------
# Missing Information 공용 wrapper (명세 29절)
# ---------------------------------------------------------------------------
def strip_internal_keys(collected: dict[str, Any] | None) -> dict[str, Any]:
    """`collected_information` 에서 내부 관리용 key 를 걸러낸다.

    되묻기 횟수 같은 값은 사용자 답변이 아니다. PDF·프롬프트·최종 출력에 넣기
    전에 이 함수를 통과시켜야 "되묻기횟수: 2" 가 진료 자료에 인쇄되는 사고를
    막을 수 있다. 관례상 `__` 로 시작하는 key 를 내부 값으로 본다.
    """
    if not isinstance(collected, dict):
        return {}
    return {k: v for k, v in collected.items() if not str(k).startswith("__")}


def with_sanitized_collected(node: NodeFn) -> NodeFn:
    """`collected_information` 의 내부 key 를 가린 State 사본으로 node 를 호출한다.

    되묻기 횟수(`__missing_info_rounds`)는 이 패키지의 구현 세부사항인데,
    Document Agent 는 `collected_information` 의 **모든 항목을** 진료 자료의
    현재 상태 항목으로 옮겨 담고, Health Response 는 프롬프트에 통째로 넣는다.
    그대로 두면 PDF 에 "__missing_info_rounds: 2" 가 인쇄되고 LLM 프롬프트에도
    들어간다.

    node 를 고치는 대신 **입력만 정리해서 넘기는** 이유: 어떤 node 구현이 오든
    이 보호가 유효해야 하고, node 쪽에 이 패키지의 내부 규칙을 알게 만들면
    결합도가 올라가기 때문이다. State 자체는 바꾸지 않으므로 카운터는 다음
    turn 까지 그대로 살아남는다.
    """

    def _node(state: dict) -> dict:
        collected = state.get("collected_information")
        if not isinstance(collected, dict):
            return node(state)

        cleaned = strip_internal_keys(collected)

        # **지식 질문에는 이 아이의 기록을 넣지 않는다.**
        #
        # 서비스 계층이 일일 기록을 `collected_information` 으로 미리 채워 둔다
        # (되묻기를 줄이려고). 그런데 "예방접종은 언제 하나요?" 같은 지식 질문에
        # 그 값이 프롬프트로 들어가면 답변이 "최근 기력 저하와 구토가 있어..." 로
        # 시작한다 — 묻지 않은 것에 답하는 것이다. 실제로 그 화면이 나왔다.
        #
        # 지식 질문은 애초에 되묻지 않으므로(`route_missing_info`) 이 값이 필요 없다.
        if str(state.get("intent") or "") == "general_knowledge":
            cleaned = {}

        if cleaned == collected:
            return node(state)
        return node({**state, "collected_information": cleaned})

    return _node


def make_missing_information_gate(
    deps: SubgraphDeps,
    key: str,
    modules: tuple[str, ...],
    names: tuple[str, ...],
) -> NodeFn:
    """Missing Information node 를 감싸 **되묻기 횟수를 세는** node 를 만든다.

    왜 필요한가: 명세 30·31·32절 mermaid 는 모두 `Missing Info → Interrupt →
    Missing Info` 순환을 갖는다. 사용자가 계속 모호하게 답하면 이 순환이 끝나지
    않고, LangGraph `recursion_limit` 에 걸려 예외로 죽는다. 그래서 라운드 수를
    세어 `routers.route_missing_info` 가 `MAX_MISSING_INFORMATION_ROUNDS` 에서
    빠져나가게 한다.

    **부족한 항목을 지우지는 않는다.** `missing_fields` 는 그대로 남아
    PDF `unknown_fields` 와 답변의 미확인 안내로 이어진다(명세 36절 "없는 정보는
    추측하지 않고 `미확인` 으로 표시한다").
    """
    inner = resolve_node(deps, key, modules, names)

    def _node(state: dict) -> dict:
        updates = dict(inner(state) or {})
        rounds = missing_information_rounds(state) + 1

        collected = updates.get("collected_information")
        merged: dict[str, Any] = dict(collected) if isinstance(collected, dict) else {}
        merged[MISSING_INFO_ROUNDS_KEY] = rounds
        updates["collected_information"] = merged

        logger.debug(
            "[%s] round=%d missing=%s ready=%s",
            key,
            rounds,
            list(updates.get("missing_fields") or state.get("missing_fields") or []),
            updates.get("minimum_information_ready"),
        )
        return updates

    return _node


# ---------------------------------------------------------------------------
# 지연 노출 (PEP 562) — langgraph 를 실제로 쓸 때만 로드한다
# ---------------------------------------------------------------------------
_BUILDERS: dict[str, str] = {
    "build_health_subgraph": ".health",
    "build_visit_subgraph": ".visit",
    "build_emergency_subgraph": ".emergency",
}


def __getattr__(name: str) -> Any:
    """`from petcare_ai.graph.subgraphs import build_health_subgraph` 를 지연 처리한다.

    서브그래프 모듈은 `langgraph.graph.StateGraph` 를 import 하므로, 이 패키지를
    단순히 import 하는 것만으로 langgraph 가 필요해지면 안 된다.
    """
    module_name = _BUILDERS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = importlib.import_module(module_name, __name__)
    return getattr(module, name)


def __dir__() -> list[str]:
    return sorted(__all__)
