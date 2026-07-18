from fastapi import APIRouter, Depends, HTTPException, Response
from pydantic import ValidationError
from sqlalchemy import select
from sqlalchemy.orm import Session

from .. import models, schemas
from ..database import get_db
from ..services.agent_client import get_agent_client
from ..services.context import build_context, pet_payload
from ..services.pdf import build_summary_pdf
from .auth import get_current_user
from .pets import get_pet_or_404

router = APIRouter(prefix="/api", tags=["summaries"])


def _get_owned_summary(
    db: Session, summary_id: int, user: models.User
) -> models.Summary:
    summary = db.get(models.Summary, summary_id)
    if not summary:
        raise HTTPException(status_code=404, detail="요약을 찾을 수 없습니다")
    get_pet_or_404(db, summary.pet_id, user)  # 소유자 확인
    return summary


@router.post("/pets/{pet_id}/summaries", response_model=schemas.SummaryOut, status_code=201)
def create_summary(
    pet_id: int,
    body: schemas.SummaryCreateRequest,
    db: Session = Depends(get_db),
    user: models.User = Depends(get_current_user),
):
    """병원 전달용 요약 생성 — Agent(Summary)가 최근 기록/기준선/진단서로 작성한다."""
    pet = get_pet_or_404(db, pet_id, user)
    agent = get_agent_client()
    risk_level = body.risk_level or "observe"
    result = agent.generate_summary(
        pet=pet_payload(pet),
        risk_level=risk_level,
        extra_note=body.extra_note,
        context=build_context(db, pet),
    )
    content = result.get("content", {})
    # http 모드에서 Agent 가 잘못된 content 를 보내면 그대로 저장될 경우
    # 이후 모든 조회/PDF 가 500 으로 오염되므로, 커밋 전에 계약을 검증한다.
    try:
        schemas.SummaryContent.model_validate(content)
    except ValidationError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Agent 응답이 계약과 다릅니다 (summary.content): {exc.error_count()}개 필드 오류",
        )
    summary = models.Summary(pet_id=pet_id, risk_level=risk_level, content=content)
    db.add(summary)
    db.commit()
    db.refresh(summary)
    return summary


@router.get("/pets/{pet_id}/summaries", response_model=list[schemas.SummaryOut])
def list_summaries(
    pet_id: int,
    db: Session = Depends(get_db),
    user: models.User = Depends(get_current_user),
):
    get_pet_or_404(db, pet_id, user)
    return db.scalars(
        select(models.Summary)
        .where(models.Summary.pet_id == pet_id)
        .order_by(models.Summary.created_at.desc())
    ).all()


@router.get("/summaries/{summary_id}", response_model=schemas.SummaryOut)
def get_summary(
    summary_id: int,
    db: Session = Depends(get_db),
    user: models.User = Depends(get_current_user),
):
    return _get_owned_summary(db, summary_id, user)


@router.get("/summaries/{summary_id}/pdf")
def get_summary_pdf(
    summary_id: int,
    db: Session = Depends(get_db),
    user: models.User = Depends(get_current_user),
):
    """병원 전달용 요약을 PDF 로 렌더링해 내려준다. (PDF 저장/이메일 첨부용)"""
    summary = _get_owned_summary(db, summary_id, user)
    pet = db.get(models.Pet, summary.pet_id)
    pet_name = pet.name if pet else "반려동물"
    pdf_bytes = build_summary_pdf(
        pet_name=pet_name,
        content=summary.content or {},
        created_at=summary.created_at.strftime("%Y.%m.%d %H:%M"),
    )
    filename = f"summary_{summary_id}.pdf"
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'inline; filename="{filename}"'},
    )
