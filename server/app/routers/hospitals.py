from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.orm import Session

from .. import models, schemas
from ..database import get_db

router = APIRouter(prefix="/api", tags=["hospitals"])


@router.get("/hospitals", response_model=list[schemas.HospitalOut])
def list_hospitals(emergency: bool | None = None, db: Session = Depends(get_db)):
    stmt = select(models.Hospital).order_by(models.Hospital.distance_km)
    if emergency is not None:
        stmt = stmt.where(models.Hospital.is_emergency == emergency)
    return db.scalars(stmt).all()
