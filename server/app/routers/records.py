from datetime import date, timedelta

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import ValidationError
from sqlalchemy import select
from sqlalchemy.orm import Session

from .. import models, schemas
from ..database import get_db
from ..services.agent_client import get_agent_client
from ..services.context import build_context, pet_payload
from .auth import get_current_user
from .pets import get_pet_or_404

router = APIRouter(prefix="/api", tags=["records"])


@router.get("/pets/{pet_id}/records", response_model=list[schemas.RecordOut])
def list_records(
    pet_id: int,
    days: int = Query(default=30, ge=1, le=365),
    db: Session = Depends(get_db),
    user: models.User = Depends(get_current_user),
):
    get_pet_or_404(db, pet_id, user)
    start = date.today() - timedelta(days=days - 1)
    return db.scalars(
        select(models.DailyEntry)
        .where(
            models.DailyEntry.pet_id == pet_id,
            models.DailyEntry.record_date >= start,
        )
        .order_by(models.DailyEntry.record_date.desc())
    ).all()


@router.post("/pets/{pet_id}/records/extract", response_model=schemas.DiaryExtractResponse)
def extract_diary(
    pet_id: int,
    body: schemas.DiaryExtractRequest,
    db: Session = Depends(get_db),
    user: models.User = Depends(get_current_user),
):
    """일기 원문 → AI(Agent)가 건강 항목별 구조화. 저장 전 보호자 확인/수정용 초안."""
    pet = get_pet_or_404(db, pet_id, user)
    if not body.text.strip():
        raise HTTPException(status_code=422, detail="일기 내용을 입력해 주세요")
    agent = get_agent_client()
    result = agent.diary_extract(
        pet=pet_payload(pet),
        text=body.text,
        record_date=(body.record_date or date.today()).isoformat(),
        context=build_context(db, pet),
    )
    try:
        return schemas.DiaryExtractResponse(**result)
    except ValidationError as exc:
        # http 모드에서 Agent 가 계약과 다른 응답을 보낸 경우 — 500 대신 502 로 변환
        raise HTTPException(
            status_code=502,
            detail=f"Agent 응답이 계약과 다릅니다 (diary-extract): {exc.error_count()}개 필드 오류",
        )


@router.post("/pets/{pet_id}/records", response_model=schemas.RecordOut, status_code=201)
def create_record(
    pet_id: int,
    body: schemas.RecordCreate,
    db: Session = Depends(get_db),
    user: models.User = Depends(get_current_user),
):
    """보호자가 확인한 일기장 기록을 저장한다.

    daily_entries 는 (pet_id, record_date) 가 PK 이므로 같은 날짜 기록이 있으면 갱신한다.
    """
    get_pet_or_404(db, pet_id, user)
    record_date = body.record_date or date.today()

    entry = db.get(models.DailyEntry, (pet_id, record_date))
    data = body.model_dump(exclude={"record_date"})
    if entry:
        for key, value in data.items():
            setattr(entry, key, value)
    else:
        entry = models.DailyEntry(pet_id=pet_id, record_date=record_date, **data)
        db.add(entry)

    db.commit()
    db.refresh(entry)
    return entry


@router.put("/pets/{pet_id}/records/{record_date}", response_model=schemas.RecordOut)
def update_record(
    pet_id: int,
    record_date: date,
    body: schemas.RecordUpdate,
    db: Session = Depends(get_db),
    user: models.User = Depends(get_current_user),
):
    """특정 날짜 기록을 부분 수정한다. 기록이 없으면 새로 만든다."""
    get_pet_or_404(db, pet_id, user)
    entry = db.get(models.DailyEntry, (pet_id, record_date))
    data = body.model_dump(exclude_unset=True, exclude={"record_date"})
    if entry:
        for key, value in data.items():
            setattr(entry, key, value)
    else:
        entry = models.DailyEntry(pet_id=pet_id, record_date=record_date, **data)
        db.add(entry)
    db.commit()
    db.refresh(entry)
    return entry


@router.delete("/pets/{pet_id}/records/{record_date}", status_code=204)
def delete_record(
    pet_id: int,
    record_date: date,
    db: Session = Depends(get_db),
    user: models.User = Depends(get_current_user),
):
    get_pet_or_404(db, pet_id, user)
    entry = db.get(models.DailyEntry, (pet_id, record_date))
    if not entry:
        raise HTTPException(status_code=404, detail="기록을 찾을 수 없습니다")
    db.delete(entry)
    db.commit()
