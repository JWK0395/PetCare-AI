import re
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from .. import models, schemas
from ..database import get_db
from ..services.agent_client import build_summary_content
from ..services.context import age_label, build_context, pet_payload
from .auth import get_current_user
from .pets import get_pet_or_404

router = APIRouter(prefix="/api", tags=["emergency"])


def _get_owned_email(
    db: Session, email_id: int, user: models.User
) -> models.EmergencyEmail:
    email = db.get(models.EmergencyEmail, email_id)
    if not email:
        raise HTTPException(status_code=404, detail="이메일을 찾을 수 없습니다")
    get_pet_or_404(db, email.pet_id, user)  # 소유자 확인
    return email


def _format_email_body(content: dict, created_label: str) -> str:
    """4섹션 요약 구조를 이메일 본문 텍스트로 렌더링한다."""
    signs = content.get("risk_signs") or []
    lines = [
        content.get("title", "PetCare AI 병원 전달용 상태 요약"),
        "",
        "[1. 문서 정보]",
        f"- 생성 일시: {created_label}",
        f"- 사용 데이터 기간: {content.get('data_period', '')}",
        "",
        "[2. 반려동물 정보]",
        f"- 이름: {content.get('pet_name', '')}",
        f"- 종: {content.get('species', '')}",
        f"- 품종: {content.get('breed', '')}",
        f"- 성별/중성화: {content.get('sex_neuter', '')}",
        f"- 나이: {content.get('age_label', '')}",
        f"- 현재 체중: {content.get('weight', '')}",
        f"- 현재 복용 중인 약: {content.get('medications', '')}",
        f"- 알레르기: {content.get('allergies', '')}",
        "",
        "[3. 상태]",
        f"- 상태 분류: {content.get('risk_label', '')}",
        "- 확인된 위험 징후:",
        *([f"  * {s}" for s in signs] or ["  * 특이 위험 징후 없음"]),
        "",
        "[4. 주호소 및 주요 변화]",
        f"- 주호소: {content.get('chief_complaint', '')}",
        f"- 주요 변화: {content.get('major_changes', '')}",
        f"- 경과: {content.get('progress', '')}",
    ]
    return "\n".join(lines)


@router.post(
    "/pets/{pet_id}/emergency-emails",
    response_model=schemas.EmergencyEmailOut,
    status_code=201,
)
def compose_emergency_email(
    pet_id: int,
    body: schemas.EmergencyEmailCreate,
    db: Session = Depends(get_db),
    user: models.User = Depends(get_current_user),
):
    """응급 상태 문서 이메일 초안 생성 (자동 첨부 구성).

    실제 발송은 보호자가 앱에서 최종 확인 후 진행한다.
    """
    pet = get_pet_or_404(db, pet_id, user)

    hospital = None
    if body.hospital_id:
        hospital = db.get(models.Hospital, body.hospital_id)
    if hospital is None:
        hospital = db.scalar(
            select(models.Hospital)
            .where(models.Hospital.is_emergency == True)  # noqa: E712
            .order_by(models.Hospital.distance_km)
            .limit(1)
        )
    if hospital is None:
        raise HTTPException(status_code=404, detail="등록된 응급 병원이 없습니다")

    symptom = body.symptom_summary or "응급 증상"
    now = datetime.now()
    subject = f"[응급] {pet.name} ({pet.breed.split(' ·')[0]} · {age_label(pet.birth_date)}) — {symptom}"

    # 병원 전달용 요약과 동일한 4섹션 구조로 문서를 구성한다.
    context = build_context(db, pet)
    content = build_summary_content(pet_payload(pet), "emergency", "", context)

    # 대화에서 감지된 응급 증상을 '확인된 위험 징후' 맨 앞에 넣는다 (중복 제거).
    detected = [
        s.strip()
        for s in re.split(r"[·,]", symptom)
        if s.strip() and s.strip() != "응급 증상"
    ]
    seen: set[str] = set()
    content["risk_signs"] = [
        x
        for x in [*detected, *content.get("risk_signs", [])]
        if not (x in seen or seen.add(x))
    ]

    created_label = now.strftime("%Y.%m.%d %H:%M")
    email_body = _format_email_body(content, created_label)
    attachments = [
        {"label": "상태 요약 문서 (PDF)", "auto": True},
        {"label": f"최근 30일 건강 기록 ({content['data_period']})", "auto": True},
    ]

    email = models.EmergencyEmail(
        pet_id=pet_id,
        hospital_id=hospital.id,
        to_email=hospital.email,
        subject=subject,
        body=email_body,
        content=content,
        attachments=attachments,
        status="draft",
    )
    db.add(email)
    db.commit()
    db.refresh(email)
    return email


@router.post("/emergency-emails/{email_id}/send", response_model=schemas.EmergencyEmailOut)
def send_emergency_email(
    email_id: int,
    db: Session = Depends(get_db),
    user: models.User = Depends(get_current_user),
):
    """보호자가 확인을 마친 이메일을 발송 처리한다.

    로컬 MVP 에서는 실제 SMTP 발송 대신 발송 기록만 저장한다.
    (실제 발송이 필요하면 이 지점에 SMTP/이메일 API 연동을 추가한다.)
    """
    email = _get_owned_email(db, email_id, user)
    if email.status == "sent":
        return email
    email.status = "sent"
    email.sent_at = datetime.utcnow()
    db.commit()
    db.refresh(email)
    return email


@router.get("/emergency-emails/{email_id}", response_model=schemas.EmergencyEmailOut)
def get_emergency_email(
    email_id: int,
    db: Session = Depends(get_db),
    user: models.User = Depends(get_current_user),
):
    return _get_owned_email(db, email_id, user)
