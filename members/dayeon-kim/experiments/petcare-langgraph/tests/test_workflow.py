from __future__ import annotations

from pathlib import Path

from petcare_agent.documents import (
    create_handoff_pdf,
)
from petcare_agent.prompts import (
    NON_EMERGENCY_SYSTEM_PROMPT,
)


def test_non_emergency_prompt_sections() -> None:
    assert "권장 행동이라는 제목을 만들지 않는다" in NON_EMERGENCY_SYSTEM_PROMPT
    assert "근거라는 제목" in NON_EMERGENCY_SYSTEM_PROMPT


def test_structured_pdf(
    tmp_path: Path,
) -> None:
    path = create_handoff_pdf(
        handoff={
            "title": "보리 병원 전달용 상태 요약",
            "pet_summary": "말티즈, 4.2kg",
            "chief_complaint": [
                "구토 1회",
            ],
            "onset_and_course": [
                "오늘 저녁 식후 발생",
            ],
            "recent_daily_record_summary": [
                "전날까지 식욕과 활동 정상",
            ],
            "diagnosis_and_medication_history": [
                "외이염 치료 완료",
            ],
            "unknown_items": [
                "복통 여부 미확인",
            ],
            "caution": (
                "보호자가 내용을 확인한 "
                "초안입니다."
            ),
        },
        pet={
            "name": "보리",
            "species": "dog",
            "breed": "말티즈",
            "sex": "female",
            "is_neutered": True,
            "birth_date": "2021-03-12",
            "weight_kg": 4.2,
        },
        session_id="pdf-test",
        output_dir=tmp_path,
    )

    assert Path(path).exists()
    assert Path(path).stat().st_size > 1000
