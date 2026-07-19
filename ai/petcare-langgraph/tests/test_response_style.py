from __future__ import annotations

from petcare_agent.nodes.agents import emergency_agent
from petcare_agent.response import clean_agent_response


def test_emoji_cleanup() -> None:
    result = clean_agent_response(
        "상태를 확인했습니다. 증상을 자세히 설명합니다."
    )
    assert "상태를 확인했습니다." in result
    assert "증상을 자세히 설명합니다." in result


def test_emergency_response_style() -> None:
    step = emergency_agent(
        {
            "backend_context": {
                "pet": {
                    "name": "모카",
                }
            },
            "emergency_hits": [
                {
                    "message": "호흡곤란",
                    "rule_id": "ER-RESP-001",
                }
            ],
            "conversation_history": [],
            "latency_ms": {},
            "errors": [],
        }
    )

    answer = step["answer"]

    assert "현재 상태는" in answer
    assert "즉시 진료가 우선" in answer
    assert "가까운 운영 중 동물병원" in answer
    assert "ER-RESP-001" not in answer
    assert "호흡곤란" in answer
