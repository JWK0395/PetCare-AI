"""Agent adapter registry for the shared harness CLI."""

from __future__ import annotations

from importlib import import_module
from typing import Any

from petcare_agent.harness.adapter import AgentAdapter
from petcare_agent.harness.adapters.current_graph import CurrentAssessmentGraphAdapter

CURRENT_AGENT_NAME = "current-assessment-graph"

_BUILT_IN_ADAPTERS: dict[str, AgentAdapter] = {
    CURRENT_AGENT_NAME: CurrentAssessmentGraphAdapter(),
    "current": CurrentAssessmentGraphAdapter(),
    "langgraph-v1": CurrentAssessmentGraphAdapter(),
}


def available_agent_names() -> list[str]:
    """Return built-in adapter names accepted by the harness."""

    return sorted(_BUILT_IN_ADAPTERS)


def load_agent_adapter(name: str) -> AgentAdapter:
    """Load a built-in adapter or a custom ``module:attribute`` adapter."""

    if name in _BUILT_IN_ADAPTERS:
        return _BUILT_IN_ADAPTERS[name]
    if ":" in name:
        return _load_dynamic_adapter(name)
    accepted = ", ".join(available_agent_names())
    raise ValueError(f"Unknown agent adapter '{name}'. Built-ins: {accepted}")


def _load_dynamic_adapter(spec: str) -> AgentAdapter:
    module_name, attribute_name = spec.split(":", 1)
    if not module_name or not attribute_name:
        raise ValueError("Custom agent adapter must use module:attribute syntax")

    module = import_module(module_name)
    value: Any = getattr(module, attribute_name)
    adapter = value() if isinstance(value, type) else value
    if not hasattr(adapter, "start_session"):
        raise TypeError(f"{spec} is not a harness AgentAdapter")
    return adapter


__all__ = [
    "CURRENT_AGENT_NAME",
    "available_agent_names",
    "load_agent_adapter",
]
