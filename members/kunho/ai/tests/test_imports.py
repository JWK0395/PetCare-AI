from petcare_agent import __version__
from petcare_agent.config import PetCareSettings
from petcare_agent.tracing import TraceContext, build_metadata, build_runnable_config


def test_package_imports() -> None:
    assert __version__ == "0.1.0"


def test_settings_contract_imports_without_env_file() -> None:
    settings = PetCareSettings(_env_file=None)

    assert settings.petcare_api_base_url == "http://localhost:8000"
    assert settings.langsmith_project == "petcare-ai-assessment"


def test_tracing_helper_imports_when_disabled_and_enabled() -> None:
    disabled = PetCareSettings(_env_file=None, LANGSMITH_TRACING=False)
    enabled = PetCareSettings(
        _env_file=None,
        LANGSMITH_TRACING=True,
        LANGSMITH_API_KEY="test-key",
    )
    context = TraceContext(
        request_id="req_1",
        conversation_id="conv_1",
        node_name="intent_classifier",
    )

    assert build_metadata(context)["node_name"] == "intent_classifier"
    assert build_runnable_config(context, disabled)["run_name"] == "assessment_graph.intent_classifier"
    assert build_runnable_config(context, enabled)["metadata"]["request_id"] == "req_1"
