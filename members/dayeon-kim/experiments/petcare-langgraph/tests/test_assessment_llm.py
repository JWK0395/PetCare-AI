from __future__ import annotations

from typing import Any

from petcare_agent.models import AssessmentOutput
from petcare_agent.nodes.assessment import assess_input
from petcare_agent.services import AgentDependencies


class RecordingLLM:
    def __init__(
        self,
        payload: dict[str, Any],
    ) -> None:
        self.payload = payload
        self.parse_calls: list[dict[str, Any]] = []

    def parse(
        self,
        *,
        schema: type[AssessmentOutput],
        system_prompt: str,
        user_prompt: str,
    ) -> AssessmentOutput:
        self.parse_calls.append(
            {
                "schema": schema,
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
            }
        )
        return schema.model_validate(self.payload)

    def text(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
    ) -> str:
        del system_prompt, user_prompt
        return "테스트 답변"


def _state(user_input: str) -> dict[str, Any]:
    return {
        "user_input": user_input,
        "conversation_history": [
            {
                "role": "user",
                "content": user_input,
            }
        ],
        "question_strategy": {},
        "follow_up_history": [],
        "triage_status": "idle",
        "diary_summary": "프롬프트에 포함되면 안 되는 전체 일기",
        "diagnosis_summary": "프롬프트에 포함되면 안 되는 전체 진단서",
        "latency_ms": {},
        "errors": [],
    }


def test_assessment_uses_llm_structured_output() -> None:
    llm = RecordingLLM(
        {
            "intent": "health_related",
            "handoff_requested": False,
            "user_goal": "복부 상태 확인",
            "symptoms": [
                {
                    "code": "pain",
                    "evidence": "배가 이상한 것 같아",
                    "negated": False,
                }
            ],
        }
    )
    state = _state("배가 평소와 다르게 이상한 것 같아")

    result = assess_input(
        state,
        dependencies=AgentDependencies(llm=llm),
    )

    assert len(llm.parse_calls) == 1
    assert result["assessment"]["intent"] == "health_related"
    assert result["assessment"]["symptoms"][0]["code"] == "pain"
    assert "errors" not in result


def test_assessment_prompt_keeps_token_optimization() -> None:
    llm = RecordingLLM(
        {
            "intent": "general_chat",
            "handoff_requested": False,
            "user_goal": "인사",
            "symptoms": [],
        }
    )
    state = _state("안녕하세요")

    assess_input(
        state,
        dependencies=AgentDependencies(llm=llm),
    )

    prompt = llm.parse_calls[0]["user_prompt"]
    assert "전체 일기" not in prompt
    assert "전체 진단서" not in prompt
    assert "안녕하세요" in prompt


def test_active_triage_keeps_health_context() -> None:
    llm = RecordingLLM(
        {
            "intent": "general_chat",
            "handoff_requested": False,
            "user_goal": "짧은 답변",
            "symptoms": [],
        }
    )
    state = _state("어제부터요")
    state["triage_status"] = "collecting"

    result = assess_input(
        state,
        dependencies=AgentDependencies(llm=llm),
    )

    assert result["assessment"]["intent"] == "health_related"


def test_explicit_handoff_request_has_deterministic_fallback() -> None:
    llm = RecordingLLM(
        {
            "intent": "general_chat",
            "handoff_requested": False,
            "user_goal": "기록 조회",
            "symptoms": [],
        }
    )
    state = _state("병원에 보여줄 전달용 문서 만들어 줘")

    result = assess_input(
        state,
        dependencies=AgentDependencies(llm=llm),
    )

    assert result["handoff_requested"] is True
    assert result["assessment"]["handoff_requested"] is True
    assert result["route"] == "general_chat"
