"""Agent 에 전달할 payload(pet / context) 를 만든다.

mock 이든 외부 Agent(http) 든 항상 같은 구조를 받는다. (ai/README.md 참고)

DB 스펙(daily_entries)에 따라 기록은 모두 텍스트 상태값이다. 수치 기반 기준선/추이는
저장하지 않으며, Agent(LLM)가 최근 기록 텍스트와 일기 원문을 읽고 판단한다.
"""

from datetime import date, timedelta

from sqlalchemy import select
from sqlalchemy.orm import Session

from .. import models


def age_label(birth_date: date | None) -> str:
    if not birth_date:
        return ""
    today = date.today()
    years = today.year - birth_date.year - (
        (today.month, today.day) < (birth_date.month, birth_date.day)
    )
    return f"만 {years}세"


def get_entries_in_window(
    db: Session, pet_id: int, window_days: int = 30, end: date | None = None
) -> list[models.DailyEntry]:
    end = end or date.today()
    start = end - timedelta(days=window_days - 1)
    stmt = (
        select(models.DailyEntry)
        .where(
            models.DailyEntry.pet_id == pet_id,
            models.DailyEntry.record_date >= start,
            models.DailyEntry.record_date <= end,
        )
        .order_by(models.DailyEntry.record_date)
    )
    return list(db.scalars(stmt))


def pet_payload(pet: models.Pet) -> dict:
    return {
        "id": pet.id,
        "name": pet.name,
        "species": pet.species,
        "breed": pet.breed,
        "birth_date": pet.birth_date.isoformat() if pet.birth_date else None,
        "age_label": age_label(pet.birth_date),
        "sex": pet.sex,
        "is_neutered": pet.is_neutered,
        "weight_kg": pet.weight_kg,
        "size_class": pet.size_class,
        "diseases": pet.diseases or "",
        "medications": pet.medications or "",
        "supplement": pet.supplement or "",
        "allergies": pet.allergies or "",
    }


def _entry_payload(e: models.DailyEntry) -> dict:
    return {
        "record_date": e.record_date.isoformat(),
        "raw_text": e.raw_text,
        "food": e.food,
        "water": e.water,
        "activity": e.activity,
        "symptom": e.symptom,
        "stool": e.stool,
        "vomit": e.vomit,
        "notes": e.notes,
    }


def build_context(db: Session, pet: models.Pet, window_days: int = 30) -> dict:
    entries = get_entries_in_window(db, pet.id, window_days)
    diagnoses = [
        {
            "date": d.date.isoformat() if d.date else None,
            "hospital": d.hospital,
            "diagnosis": d.diagnosis,
            "content": d.content,
        }
        for d in db.scalars(
            select(models.Diagnosis)
            .where(models.Diagnosis.pet_id == pet.id)
            .order_by(models.Diagnosis.date)
        )
    ]
    return {
        "window_days": window_days,
        "records": [_entry_payload(e) for e in entries],
        "diagnoses": diagnoses,
    }
