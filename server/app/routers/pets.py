from datetime import date

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from .. import models, schemas
from ..database import get_db
from ..services.context import age_label
from .auth import get_current_user

router = APIRouter(prefix="/api/pets", tags=["pets"])


def get_pet_or_404(db: Session, pet_id: int, user: models.User) -> models.Pet:
    """반려동물 조회 + 소유자 확인. 다른 계정의 반려동물은 404 로 감춘다."""
    pet = db.get(models.Pet, pet_id)
    if not pet or pet.owner_id != user.id:
        raise HTTPException(status_code=404, detail="반려동물을 찾을 수 없습니다")
    return pet


def _pet_out(pet: models.Pet) -> schemas.PetOut:
    out = schemas.PetOut.model_validate(pet)
    out.age_label = age_label(pet.birth_date)
    return out


@router.get("", response_model=list[schemas.PetOut])
def list_pets(
    db: Session = Depends(get_db),
    user: models.User = Depends(get_current_user),
):
    pets = db.scalars(
        select(models.Pet)
        .where(models.Pet.owner_id == user.id)
        .order_by(models.Pet.id)
    ).all()
    return [_pet_out(p) for p in pets]


@router.post("", response_model=schemas.PetOut, status_code=201)
def create_pet(
    body: schemas.PetCreate,
    db: Session = Depends(get_db),
    user: models.User = Depends(get_current_user),
):
    pet = models.Pet(owner_id=user.id, **body.model_dump())
    db.add(pet)
    db.commit()
    db.refresh(pet)
    return _pet_out(pet)


@router.get("/{pet_id}", response_model=schemas.PetOut)
def get_pet(
    pet_id: int,
    db: Session = Depends(get_db),
    user: models.User = Depends(get_current_user),
):
    return _pet_out(get_pet_or_404(db, pet_id, user))


@router.put("/{pet_id}", response_model=schemas.PetOut)
def update_pet(
    pet_id: int,
    body: schemas.PetUpdate,
    db: Session = Depends(get_db),
    user: models.User = Depends(get_current_user),
):
    pet = get_pet_or_404(db, pet_id, user)
    for key, value in body.model_dump(exclude_unset=True).items():
        setattr(pet, key, value)
    db.commit()
    db.refresh(pet)
    return _pet_out(pet)


@router.delete("/{pet_id}", status_code=204)
def delete_pet(
    pet_id: int,
    db: Session = Depends(get_db),
    user: models.User = Depends(get_current_user),
):
    pet = get_pet_or_404(db, pet_id, user)
    db.delete(pet)
    db.commit()


@router.get("/{pet_id}/dashboard", response_model=schemas.DashboardOut)
def dashboard(
    pet_id: int,
    db: Session = Depends(get_db),
    user: models.User = Depends(get_current_user),
):
    pet = get_pet_or_404(db, pet_id, user)
    today = date.today()

    today_record = db.scalar(
        select(models.DailyEntry).where(
            models.DailyEntry.pet_id == pet_id,
            models.DailyEntry.record_date == today,
        )
    )
    recent = db.scalars(
        select(models.DailyEntry)
        .where(models.DailyEntry.pet_id == pet_id)
        .order_by(models.DailyEntry.record_date.desc())
        .limit(30)
    ).all()

    latest_food = next((r.food for r in recent if r.food), "")
    latest_activity = next((r.activity for r in recent if r.activity), "")
    last_diagnosis = db.scalar(
        select(models.Diagnosis)
        .where(models.Diagnosis.pet_id == pet_id)
        .order_by(models.Diagnosis.date.desc())
        .limit(1)
    )

    return schemas.DashboardOut(
        pet=_pet_out(pet),
        today_record=schemas.RecordOut.model_validate(today_record) if today_record else None,
        recent_food_note=latest_food,
        recent_activity_note=latest_activity,
        record_count_30d=len(recent),
        last_diagnosis=schemas.DiagnosisOut.model_validate(last_diagnosis)
        if last_diagnosis
        else None,
    )
