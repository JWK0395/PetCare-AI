from __future__ import annotations

from pydantic import BaseModel

from petcare_agent.nodes.handoff_summary_builder import build_handoff_summary
from petcare_agent.schemas.graph_state import PetCareGraphState
from petcare_agent.schemas.llm_outputs import HandoffSummaryOutput


class FakeLLMClient:
    def __init__(self, output: BaseModel) -> None:
        self.output = output
        self.calls = 0

    def structured_output(self, **kwargs):
        self.calls += 1
        return self.output


def test_handoff_summary_builder_applies_structured_summary_without_sending_email() -> None:
    output = HandoffSummaryOutput(
        type="non_emergency",
        summary="Dog has vomiting for 2 hours without reported collapse.",
        email_draft="Draft only: please review before sending to a clinic.",
    )
    client = FakeLLMClient(output)
    state = PetCareGraphState(user_input="Please make a hospital handoff summary.")

    result = build_handoff_summary(state, llm_client=client)

    assert client.calls == 1
    assert result.handoff.type == "non_emergency"
    assert result.handoff.summary == output.summary
    assert result.handoff.email_draft == output.email_draft

