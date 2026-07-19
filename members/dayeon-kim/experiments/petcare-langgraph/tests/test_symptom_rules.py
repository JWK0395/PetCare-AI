from __future__ import annotations

from petcare_agent.nodes.safety import raw_keyword_hits
from petcare_agent.nodes.triage import detect_symptoms


def test_normal_statements_are_not_symptoms() -> None:
    assert detect_symptoms("소변은 평소처럼 잘 봐요") == []
    assert detect_symptoms("호흡은 괜찮아요") == []
    assert detect_symptoms("설사는 없고 밥은 잘 먹어요") == []


def test_ambiguous_sick_appearance_is_not_pain() -> None:
    assert "pain" not in detect_symptoms(
        "그냥 아파 보여요"
    )
    assert "pain" in detect_symptoms(
        "다리를 만지면 낑낑대요"
    )


def test_contrast_clause_negation_scope() -> None:
    assert detect_symptoms(
        "구토는 없지만 설사는 있어요"
    ) == ["diarrhea"]


def test_normal_gum_color_does_not_confirm_pallor() -> None:
    assert "pallor" not in detect_symptoms(
        "창백해 보이지만 잇몸은 분홍색이에요"
    )


def test_negated_breathing_distress_is_not_emergency() -> None:
    assert raw_keyword_hits("숨은 안 힘들어요") == set()
    assert "respiratory_distress" in raw_keyword_hits(
        "숨이 힘들어요"
    )
