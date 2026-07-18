from __future__ import annotations

import json
from pathlib import Path

from petcare_agent.local_data import (
    load_local_context,
    make_backend_request,
)
from petcare_agent.nodes.safety import raw_keyword_hits
from petcare_agent.nodes.triage import (
    default_question_strategy,
    is_unknown_answer,
)


def test_emergency_keyword() -> None:
    assert (
        "respiratory_distress"
        in raw_keyword_hits(
            "호흡이 힘들어 보여."
        )
    )


def test_unknown_answer() -> None:
    assert is_unknown_answer("모르겠어.")


def test_default_strategy() -> None:
    strategy = default_question_strategy()
    assert strategy["finished"] is False
    assert strategy["completed_cycles"] == []


def test_local_json_loader(
    tmp_path: Path,
) -> None:
    pet = {
        "id": 103,
        "name": "모카",
        "species": "dog",
    }
    daily = {
        "daily_entries": [
            {
                "record_date": "2026-07-18",
                "notes": "평소와 같음",
            }
        ]
    }
    diagnoses = {
        "diagnoses": []
    }

    (tmp_path / "pet_profile.json").write_text(
        json.dumps(pet, ensure_ascii=False),
        encoding="utf-8",
    )
    (tmp_path / "daily_entries.json").write_text(
        json.dumps(daily, ensure_ascii=False),
        encoding="utf-8",
    )
    (tmp_path / "diagnoses.json").write_text(
        json.dumps(diagnoses, ensure_ascii=False),
        encoding="utf-8",
    )

    context = load_local_context(tmp_path)
    request = make_backend_request(
        context,
        "안녕",
        session_id="test-session",
    )

    assert context["pet"]["name"] == "모카"
    assert request["pet_id"] == 103