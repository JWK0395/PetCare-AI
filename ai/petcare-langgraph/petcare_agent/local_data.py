from __future__ import annotations

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from .models import BackendContextPayload


DATA_DIR = Path("data")

PROFILE_CANDIDATES = [
    "pet_profile.json",
    "profile.json",
    "moka_profile.json",
]

DAILY_AGGREGATE_CANDIDATES = [
    "daily_entries.json",
    "pet_daily_entries.json",
    "moka_daily_entries.json",
    "today_diary.json",
]

DIAGNOSES_CANDIDATES = [
    "diagnoses.json",
    "pet_diagnoses.json",
    "diagnosis_list.json",
    "moka_diagnoses.json",
]


def read_json_file(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8") as file:
            return json.load(file)
    except json.JSONDecodeError as error:
        raise ValueError(
            f"JSON 문법 오류: {path}\n"
            f"line={error.lineno}, column={error.colno}"
        ) from error


def find_first_existing(
    data_dir: Path,
    candidates: list[str],
) -> Path | None:
    for filename in candidates:
        path = data_dir / filename
        if path.exists():
            return path
    return None


def unwrap_profile(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise TypeError("프로필 JSON 최상위 값은 객체여야 합니다.")

    if isinstance(payload.get("pet"), dict):
        return payload["pet"]

    if isinstance(payload.get("profile"), dict):
        return payload["profile"]

    return payload


def unwrap_daily_entries(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        entries = payload
    elif isinstance(payload, dict):
        if isinstance(payload.get("daily_entries"), list):
            entries = payload["daily_entries"]
        elif isinstance(payload.get("recent_daily_entries"), list):
            entries = payload["recent_daily_entries"]
        elif isinstance(payload.get("entries"), list):
            entries = payload["entries"]
        elif payload.get("record_date"):
            entries = [payload]
        else:
            raise ValueError(
                "일기 JSON에서 daily_entries, recent_daily_entries, "
                "entries 또는 record_date를 찾지 못했습니다."
            )
    else:
        raise TypeError("일기 JSON은 객체 또는 배열이어야 합니다.")

    return [
        item
        for item in entries
        if isinstance(item, dict)
    ]


def unwrap_diagnoses(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        diagnoses = payload
    elif isinstance(payload, dict):
        if isinstance(payload.get("diagnoses"), list):
            diagnoses = payload["diagnoses"]
        elif isinstance(payload.get("diagnosis_list"), list):
            diagnoses = payload["diagnosis_list"]
        elif payload.get("diagnosis"):
            diagnoses = [payload]
        else:
            raise ValueError(
                "진단서 JSON에서 diagnoses, diagnosis_list 또는 "
                "diagnosis를 찾지 못했습니다."
            )
    else:
        raise TypeError("진단서 JSON은 객체 또는 배열이어야 합니다.")

    return [
        item
        for item in diagnoses
        if isinstance(item, dict)
    ]


def load_daily_entries(
    data_dir: Path,
) -> tuple[list[dict[str, Any]], list[str]]:
    aggregate_path = find_first_existing(
        data_dir,
        DAILY_AGGREGATE_CANDIDATES,
    )

    if aggregate_path is not None:
        entries = unwrap_daily_entries(
            read_json_file(aggregate_path)
        )
        return entries, [str(aggregate_path)]

    daily_directories = [
        data_dir / "daily_entries",
        data_dir / "daily",
        data_dir / "diaries",
    ]

    daily_files: list[Path] = []

    for directory in daily_directories:
        if directory.exists() and directory.is_dir():
            daily_files.extend(
                sorted(directory.glob("*.json"))
            )

    if not daily_files:
        raise FileNotFoundError(
            "일기 파일을 찾지 못했습니다.\n"
            "daily_entries.json 한 파일 또는 "
            "daily_entries/*.json 파일을 넣어 주세요."
        )

    entries: list[dict[str, Any]] = []

    for path in daily_files:
        entries.extend(
            unwrap_daily_entries(
                read_json_file(path)
            )
        )

    return entries, [
        str(path)
        for path in daily_files
    ]


def infer_unknown_items(
    pet: dict[str, Any],
) -> list[str]:
    required_fields = {
        "name": "반려동물 이름",
        "species": "종",
        "breed": "품종",
        "weight_kg": "체중",
        "birth_date": "생년월일",
        "sex": "성별",
        "is_neutered": "중성화 여부",
    }

    unknown_items: list[str] = []

    for field_name, label in required_fields.items():
        if pet.get(field_name) is None:
            unknown_items.append(
                f"{label} 미입력"
            )

    return unknown_items


def load_local_context(
    data_dir: str | Path = DATA_DIR,
) -> dict[str, Any]:
    resolved_dir = Path(data_dir)

    if not resolved_dir.exists():
        raise FileNotFoundError(
            f"데이터 폴더가 없습니다: {resolved_dir}\n"
            "Colab 왼쪽 파일 메뉴에서 data 폴더를 만든 뒤 "
            "JSON 파일을 넣어 주세요."
        )

    profile_path = find_first_existing(
        resolved_dir,
        PROFILE_CANDIDATES,
    )
    diagnoses_path = find_first_existing(
        resolved_dir,
        DIAGNOSES_CANDIDATES,
    )

    if profile_path is None:
        raise FileNotFoundError(
            "프로필 파일을 찾지 못했습니다. "
            "pet_profile.json 파일을 넣어 주세요."
        )

    if diagnoses_path is None:
        raise FileNotFoundError(
            "진단서 목록 파일을 찾지 못했습니다. "
            "diagnoses.json 파일을 넣어 주세요."
        )

    pet = unwrap_profile(
        read_json_file(profile_path)
    )
    daily_entries, daily_source_files = (
        load_daily_entries(resolved_dir)
    )
    diagnoses = unwrap_diagnoses(
        read_json_file(diagnoses_path)
    )

    if pet.get("id") is None:
        raise ValueError(
            "프로필 JSON에 id 필드가 필요합니다."
        )

    daily_entries = sorted(
        daily_entries,
        key=lambda item: item.get(
            "record_date",
            "",
        ),
    )
    diagnoses = sorted(
        diagnoses,
        key=lambda item: item.get(
            "date",
            "",
        ),
    )

    pet_id = str(pet["id"])

    filtered_diagnoses = [
        item
        for item in diagnoses
        if (
            item.get("pet_id") is None
            or str(item.get("pet_id")) == pet_id
        )
    ]

    context = {
        "pet": pet,
        "daily_entries": daily_entries,
        "diagnoses": filtered_diagnoses,
        "unknown_items": infer_unknown_items(pet),
        "data_from": (
            daily_entries[0].get("record_date")
            if daily_entries
            else None
        ),
        "data_to": (
            daily_entries[-1].get("record_date")
            if daily_entries
            else None
        ),
        "generated_at": (
            datetime.now()
            .astimezone()
            .isoformat()
        ),
        "source_files": {
            "profile": str(profile_path),
            "daily_entries": daily_source_files,
            "diagnoses": str(diagnoses_path),
        },
    }

    BackendContextPayload.model_validate(context)
    return context


def make_backend_request(
    context: dict[str, Any],
    user_input: str,
    *,
    session_prefix: str = "local",
    session_id: str | None = None,
    location: dict[str, Any] | None = None,
) -> dict[str, Any]:
    pet_id = context.get("pet", {}).get("id")

    if pet_id is None:
        raise ValueError(
            "context.pet.id가 필요합니다."
        )

    resolved_session_id = (
        session_id
        or f"{session_prefix}-{uuid.uuid4().hex[:8]}"
    )

    request = {
        "session_id": resolved_session_id,
        "pet_id": int(pet_id),
        "user_input": user_input,
        "context": context,
    }

    if location is not None:
        request["location"] = location

    return request
