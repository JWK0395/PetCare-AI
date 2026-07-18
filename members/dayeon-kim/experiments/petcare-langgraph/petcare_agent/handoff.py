from __future__ import annotations

from datetime import date, datetime
from typing import Any

from .models import HandoffOutput, PetCareState


SPECIES_LABELS = {
    "dog": "강아지",
    "cat": "고양이",
}

SEX_LABELS = {
    "male": "수컷",
    "female": "암컷",
}


def _format_date(
    value: str | None,
) -> str:
    if not value:
        return "미확인"

    normalized = value.strip()

    if (
        len(normalized) == 10
        and normalized[4] == "-"
        and normalized[7] == "-"
    ):
        try:
            parsed_date = date.fromisoformat(
                normalized
            )
            return parsed_date.strftime(
                "%Y.%m.%d"
            )
        except ValueError:
            return normalized

    try:
        parsed = datetime.fromisoformat(
            normalized.replace(
                "Z",
                "+00:00",
            )
        )
        return parsed.strftime(
            "%Y.%m.%d %H:%M"
        )
    except ValueError:
        pass

    try:
        parsed_date = date.fromisoformat(
            normalized[:10]
        )
        return parsed_date.strftime(
            "%Y.%m.%d"
        )
    except ValueError:
        return normalized


def _data_period(
    context: dict[str, Any],
) -> str:
    data_from = context.get("data_from")
    data_to = context.get("data_to")

    if not data_from or not data_to:
        dates = sorted(
            str(item.get("record_date"))
            for item in context.get(
                "daily_entries",
                [],
            )
            if item.get("record_date")
        )

        if dates:
            data_from = data_from or dates[0]
            data_to = data_to or dates[-1]

    if not data_from and not data_to:
        return "기록 없음"

    return (
        f"{_format_date(str(data_from))} "
        f"~ {_format_date(str(data_to))}"
    )


def _age_text(
    birth_date: str | None,
) -> str:
    if not birth_date:
        return "미확인"

    try:
        born = date.fromisoformat(
            birth_date[:10]
        )
    except ValueError:
        return "미확인"

    today = date.today()
    age = (
        today.year
        - born.year
        - (
            (today.month, today.day)
            < (born.month, born.day)
        )
    )

    return f"만 {max(age, 0)}세"


def _item_text(
    item: Any,
) -> str | None:
    if isinstance(item, str):
        value = item.strip()
        return value or None

    if not isinstance(item, dict):
        return None

    name = (
        item.get("name")
        or item.get("substance")
        or item.get("medication")
        or item.get("diagnosis")
    )

    details = (
        item.get("details")
        or item.get("dosage_original_text")
        or item.get("dosage")
        or item.get("reaction")
        or item.get("content")
    )

    if name and details:
        return f"{name}: {details}"

    if name:
        return str(name)

    if details:
        return str(details)

    return None


def _deduplicate(
    values: list[str],
) -> list[str]:
    result: list[str] = []

    for value in values:
        normalized = value.strip()

        if (
            normalized
            and normalized not in result
        ):
            result.append(normalized)

    return result


def _profile_items(
    pet: dict[str, Any],
    item_type: str,
) -> list[str]:
    values: list[str] = []

    direct_key = (
        "medications"
        if item_type == "medication"
        else "allergies"
    )

    direct_values = pet.get(
        direct_key,
        [],
    )

    if isinstance(
        direct_values,
        (str, dict),
    ):
        direct_values = [direct_values]

    if isinstance(direct_values, list):
        for item in direct_values:
            text = _item_text(item)

            if text:
                values.append(text)

    combined = pet.get(
        "diseases_medications_allergies",
        [],
    )

    if isinstance(combined, list):
        for item in combined:
            if not isinstance(item, dict):
                continue

            if (
                str(item.get("type", "")).lower()
                != item_type
            ):
                continue

            text = _item_text(item)

            if text:
                values.append(text)

    return _deduplicate(values)


def _sex_neutered_text(
    pet: dict[str, Any],
) -> str:
    sex = SEX_LABELS.get(
        str(pet.get("sex", "")).lower(),
        str(pet.get("sex") or "미확인"),
    )

    neutered = pet.get(
        "is_neutered",
        pet.get("neutered"),
    )

    if neutered is True:
        neutered_text = "중성화 완료"
    elif neutered is False:
        neutered_text = "중성화하지 않음"
    else:
        neutered_text = "중성화 여부 미확인"

    return f"{sex} / {neutered_text}"


def _weight_text(
    pet: dict[str, Any],
) -> str:
    weight = pet.get("weight_kg")

    if weight is None:
        return "미확인"

    return f"{weight}kg"


def _status_classification(
    state: PetCareState,
) -> str:
    if state.get("route") == "emergency":
        return "응급 징후 가능성"

    if state.get("route") == "non_emergency":
        return "비응급 건강 이상"

    return "상태 확인"


def _risk_signs(
    state: PetCareState,
) -> list[str]:
    values = [
        str(item.get("message", "")).strip()
        for item in state.get(
            "emergency_hits",
            [],
        )
        if str(
            item.get("message", "")
        ).strip()
    ]

    return _deduplicate(values)


def build_handoff_document(
    state: PetCareState,
    summary: HandoffOutput,
) -> dict[str, Any]:
    context = state.get(
        "backend_context",
        {},
    )
    pet = context.get("pet", {})

    generated_at = datetime.now().astimezone()

    return {
        "document_info": {
            "title": (
                "PetCare AI 병원 전달용 상태 요약"
            ),
            "generated_at": generated_at.strftime(
                "%Y.%m.%d %H:%M"
            ),
            "data_period": _data_period(
                context
            ),
        },
        "pet_info": {
            "name": pet.get(
                "name",
                "미확인",
            ),
            "species": SPECIES_LABELS.get(
                str(
                    pet.get(
                        "species",
                        "",
                    )
                ).lower(),
                str(
                    pet.get(
                        "species",
                        "미확인",
                    )
                ),
            ),
            "breed": pet.get(
                "breed",
                "미확인",
            ),
            "sex_neutered": (
                _sex_neutered_text(pet)
            ),
            "age": _age_text(
                pet.get("birth_date")
            ),
            "weight": _weight_text(
                pet
            ),
            "medications": (
                _profile_items(
                    pet,
                    "medication",
                )
            ),
            "allergies": (
                _profile_items(
                    pet,
                    "allergy",
                )
            ),
        },
        "status": {
            "classification": (
                _status_classification(
                    state
                )
            ),
            "risk_signs": (
                _risk_signs(state)
            ),
        },
        "clinical_summary": {
            "chief_complaints": (
                summary.chief_complaints
            ),
            "major_changes": (
                summary.major_changes
            ),
            "course": summary.course,
        },
    }


def _list_lines(
    values: list[str],
    *,
    empty_text: str = "기록 없음",
) -> list[str]:
    if not values:
        return [f"  * {empty_text}"]

    return [
        f"  * {value}"
        for value in values
    ]


def format_handoff_text(
    document: dict[str, Any],
    *,
    hospital_name: str | None = None,
) -> str:
    info = document["document_info"]
    pet = document["pet_info"]
    status = document["status"]
    clinical = document[
        "clinical_summary"
    ]

    lines: list[str] = []

    if hospital_name:
        lines.extend(
            [
                f"수신 병원: {hospital_name}",
                "",
            ]
        )

    lines.extend(
        [
            "1. 문서 정보",
            (
                f"- 문서 제목: "
                f"{info['title']}"
            ),
            (
                f"- 생성 일시: "
                f"{info['generated_at']}"
            ),
            (
                f"- 사용 데이터 기간: "
                f"{info['data_period']}"
            ),
            "",
            "2. 반려동물 정보",
            f"- 이름: {pet['name']}",
            f"- 종: {pet['species']}",
            f"- 품종: {pet['breed']}",
            (
                f"- 성별/중성화: "
                f"{pet['sex_neutered']}"
            ),
            f"- 나이: {pet['age']}",
            (
                f"- 현재 체중: "
                f"{pet['weight']}"
            ),
            "- 현재 복용 중인 약",
            *_list_lines(
                pet["medications"]
            ),
            "- 알레르기",
            *_list_lines(
                pet["allergies"]
            ),
            "",
            "3. 상태",
            (
                f"- 상태 분류: "
                f"{status['classification']}"
            ),
            "- 확인된 위험 징후",
            *_list_lines(
                status["risk_signs"],
                empty_text=(
                    "확인된 응급 위험 징후 없음"
                ),
            ),
            "",
            "4. 주호소 및 주요 변화",
            (
                "- 주호소: "
                + (
                    ", ".join(
                        clinical[
                            "chief_complaints"
                        ]
                    )
                    or "기록 없음"
                )
            ),
            "- 주요 변화",
            *_list_lines(
                clinical["major_changes"]
            ),
            "- 경과",
            *_list_lines(
                clinical["course"]
            ),
        ]
    )

    return "\n".join(lines).strip()
