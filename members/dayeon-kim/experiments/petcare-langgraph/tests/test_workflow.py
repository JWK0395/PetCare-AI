from __future__ import annotations

from pathlib import Path

from petcare_agent.documents import create_handoff_pdf
from petcare_agent.models import HandoffDocument
from petcare_agent.prompts import NON_EMERGENCY_SYSTEM_PROMPT


def test_non_emergency_prompt_sections() -> None:
    assert "권장 행동이라는 제목을 만들지 않는다" in NON_EMERGENCY_SYSTEM_PROMPT
    assert "근거라는 제목" in NON_EMERGENCY_SYSTEM_PROMPT


def test_structured_pdf(tmp_path: Path) -> None:
    handoff = HandoffDocument.model_validate(
        {
            "document_info": {
                "title": "PetCare AI 병원 전달용 상태 요약",
                "generated_at": "2026.07.19 15:30",
                "data_period": "2026.06.01 ~ 2026.06.30",
            },
            "pet_info": {
                "name": "보리",
                "species": "강아지",
                "breed": "말티즈",
                "sex_neutered": "암컷 / 중성화 완료",
                "age": "만 5세",
                "weight": "4.2kg",
                "medications": ["점이액"],
                "allergies": ["닭고기 의심"],
            },
            "status": {
                "classification": "비응급 건강 이상",
                "risk_signs": [],
            },
            "clinical_summary": {
                "chief_complaints": ["식욕 감소"],
                "major_changes": ["최근 일주일간 섭취량 감소"],
                "course": ["평소의 약 70% 섭취"],
            },
        }
    )

    path = create_handoff_pdf(
        handoff=handoff,
        session_id="pdf-test",
        output_dir=tmp_path,
    )

    assert Path(path).exists()
    assert Path(path).stat().st_size > 1000
