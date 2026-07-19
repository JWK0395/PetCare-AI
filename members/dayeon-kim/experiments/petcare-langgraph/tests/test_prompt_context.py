from __future__ import annotations

from petcare_agent.prompt_context import build_prompt_context


def test_prompt_context_selects_relevant_records_only() -> None:
    entries = [
        {
            "record_date": f"2026-06-{day:02d}",
            "food": "정상 섭취",
            "activity": "정상",
            "notes": "특이사항 없음",
        }
        for day in range(1, 31)
    ]
    entries[5]["food"] = "아침 사료를 조금 남김"
    entries[16]["food"] = "평소의 약 80% 섭취"

    state = {
        "user_input": "보리 식사량이 줄었던 기록 알려줘",
        "backend_context": {
            "pet": {
                "id": 204,
                "name": "보리",
                "species": "dog",
                "medications": ["점이액"],
            },
            "daily_entries": entries,
            "diagnoses": [
                {
                    "date": "2026-06-18",
                    "diagnosis": "외이염",
                    "content": "점이액 사용",
                }
            ],
            "data_from": "2026-06-01",
            "data_to": "2026-06-30",
        },
        "assessment": {
            "symptoms": []
        },
        "question_strategy": {},
    }

    selected = build_prompt_context(state)

    assert len(selected.daily_entries) <= 8
    assert len(selected.daily_entries) < len(entries)
    selected_dates = {
        item.get("record_date")
        for item in selected.daily_entries
    }
    assert selected_dates == {
        "2026-06-06",
        "2026-06-17",
    }
    assert "medications" not in selected.pet
    assert selected.diagnoses == []
