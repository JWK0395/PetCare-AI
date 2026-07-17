from __future__ import annotations

import json
from pathlib import Path

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "triage_handoff_golden.jsonl"


def test_triage_handoff_golden_fixture_shape() -> None:
    rows = [json.loads(line) for line in FIXTURE_PATH.read_text(encoding="utf-8-sig").splitlines() if line]

    assert len(rows) == 12
    assert {row["id"][:3] for row in rows} == {f"G{index:02d}" for index in range(1, 13)}
    for row in rows:
        assert row["input"]
        assert "context" in row
        assert row["expected"]["risk_level"] in {
            "emergency",
            "urgent",
            "non_emergency",
            "unknown",
        }
        assert row["expected"]["max_followup_questions"] <= 2
        forbidden_fields = row["expected"].get("handoff_assertions", {}).get(
            "must_not_include_fields",
            [],
        )
        assert "risk_level" in forbidden_fields or row["expected"].get("route") != "handoff"
