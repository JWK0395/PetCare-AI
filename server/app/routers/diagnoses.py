import re
import time
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from pydantic import ValidationError
from sqlalchemy import select
from sqlalchemy.orm import Session

from .. import models, schemas
from ..config import UPLOAD_DIR, settings
from ..database import get_db
from ..services.agent_client import get_agent_client
from ..services.context import pet_payload
from .auth import get_current_user
from .pets import get_pet_or_404

router = APIRouter(prefix="/api", tags=["diagnoses"])


def _get_owned_diagnosis(
    db: Session, diagnosis_id: int, user: models.User
) -> models.Diagnosis:
    diagnosis = db.get(models.Diagnosis, diagnosis_id)
    if not diagnosis:
        raise HTTPException(status_code=404, detail="진단서를 찾을 수 없습니다")
    get_pet_or_404(db, diagnosis.pet_id, user)  # 소유자 확인
    return diagnosis


def _read_pdf_text(path: Path) -> str:
    try:
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        return "\n".join((page.extract_text() or "") for page in reader.pages)
    except Exception:
        return ""


@router.get("/pets/{pet_id}/diagnoses", response_model=list[schemas.DiagnosisOut])
def list_diagnoses(
    pet_id: int,
    db: Session = Depends(get_db),
    user: models.User = Depends(get_current_user),
):
    get_pet_or_404(db, pet_id, user)
    # 최신순(발급일 내림차순, 같은 날짜/미기재는 등록 시각 내림차순)
    return db.scalars(
        select(models.Diagnosis)
        .where(models.Diagnosis.pet_id == pet_id)
        .order_by(models.Diagnosis.date.desc(), models.Diagnosis.created_at.desc())
    ).all()


@router.post(
    "/pets/{pet_id}/diagnoses/extract", response_model=schemas.DiagnosisExtractResponse
)
async def extract_diagnosis(
    pet_id: int,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    user: models.User = Depends(get_current_user),
):
    """진단서 PDF/이미지 업로드 → AI(Agent)가 항목 추출. 저장 전 보호자 확인/수정용 초안."""
    pet = get_pet_or_404(db, pet_id, user)

    # 파일명 정리 + 확장자 화이트리스트 + 크기 제한
    safe_name = re.sub(r"[^\w가-힣.\-]", "_", file.filename or "diagnosis.pdf")
    suffix = Path(safe_name).suffix.lower()
    if suffix not in settings.allowed_extensions:
        allowed = ", ".join(sorted(settings.allowed_extensions))
        raise HTTPException(
            status_code=422, detail=f"지원하지 않는 파일 형식입니다 ({allowed} 만 가능)"
        )
    content = await file.read()
    if not content:
        raise HTTPException(status_code=422, detail="빈 파일입니다")
    if len(content) > settings.max_upload_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"파일이 너무 큽니다 (최대 {settings.max_upload_mb}MB)",
        )
    stored_name = f"{int(time.time())}_{safe_name}"
    dest = UPLOAD_DIR / stored_name
    dest.write_bytes(content)

    file_text = ""
    if dest.suffix.lower() == ".pdf":
        file_text = _read_pdf_text(dest)

    agent = get_agent_client()
    result = agent.diagnosis_extract(
        pet=pet_payload(pet), file_name=safe_name, file_text=file_text
    )
    try:
        return schemas.DiagnosisExtractResponse(
            fields=schemas.DiagnosisBase(**result.get("fields", {})),
            original_file_ref=stored_name,
            items_read=result.get("items_read", 0),
            source=result.get("source", "mock"),
        )
    except ValidationError as exc:
        # http 모드에서 Agent 가 계약과 다른 응답을 보낸 경우 — 500 대신 502 로 변환
        raise HTTPException(
            status_code=502,
            detail=f"Agent 응답이 계약과 다릅니다 (diagnosis-extract): {exc.error_count()}개 필드 오류",
        )


@router.post("/pets/{pet_id}/diagnoses", response_model=schemas.DiagnosisOut, status_code=201)
def create_diagnosis(
    pet_id: int,
    body: schemas.DiagnosisCreate,
    db: Session = Depends(get_db),
    user: models.User = Depends(get_current_user),
):
    """보호자가 확인한 진단서 정보를 확정 저장한다."""
    get_pet_or_404(db, pet_id, user)
    diagnosis = models.Diagnosis(pet_id=pet_id, **body.model_dump())
    db.add(diagnosis)
    db.commit()
    db.refresh(diagnosis)
    return diagnosis


@router.put("/diagnoses/{diagnosis_id}", response_model=schemas.DiagnosisOut)
def update_diagnosis(
    diagnosis_id: int,
    body: schemas.DiagnosisUpdate,
    db: Session = Depends(get_db),
    user: models.User = Depends(get_current_user),
):
    diagnosis = _get_owned_diagnosis(db, diagnosis_id, user)
    for key, value in body.model_dump(exclude_unset=True).items():
        setattr(diagnosis, key, value)
    db.commit()
    db.refresh(diagnosis)
    return diagnosis


@router.delete("/diagnoses/{diagnosis_id}", status_code=204)
def delete_diagnosis(
    diagnosis_id: int,
    db: Session = Depends(get_db),
    user: models.User = Depends(get_current_user),
):
    diagnosis = _get_owned_diagnosis(db, diagnosis_id, user)
    db.delete(diagnosis)
    db.commit()
