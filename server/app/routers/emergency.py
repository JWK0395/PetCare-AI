import re
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException
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


def _format_email_body(content: dict, created_label: str, hospital_label: str = "") -> str:
    """4섹션 요약 구조를 이메일 본문 텍스트로 렌더링한다.

    `hospital_label`(병원 이름 · 전화번호)은 있을 때만 1섹션에 한 줄로 넣는다.
    AI 검색으로 찾은 병원은 DB 행이 없어 `hospital_id` 로 남길 수 없는데, 어느 병원
    앞으로 쓴 문서인지는 본문에 남아야 보호자가 초안을 다시 열었을 때 확인할 수 있다.
    """
    signs = content.get("risk_signs") or []
    lines = [
        content.get("title", "PetCare AI 병원 전달용 상태 요약"),
        "",
        "[1. 문서 정보]",
        f"- 생성 일시: {created_label}",
        f"- 사용 데이터 기간: {content.get('data_period', '')}",
        *([f"- 수신 병원: {hospital_label}"] if hospital_label else []),
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
        # 빈 목록일 때 "특이 위험 징후 없음" 이라고 쓰지 않는다 — 병원이 읽으면
        # '확인했고 없었다' 는 뜻이 되지만, 실제로는 기록에서 찾지 못했을 뿐이다.
        *([f"  * {s}" for s in signs] or ["  * 기록에서 확인된 항목 없음"]),
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
    """응급 상태 문서 이메일 초안 생성.

    실제 발송은 보호자가 앱에서 최종 확인 후 진행한다.

    ## 병원 결정 우선순위

    1. 요청으로 받은 `hospital_name`/`hospital_email` — AI 가 실시간 검색으로 찾은
       병원이다. DB 에 없으므로 `hospital_id` 는 남기지 않는다.
    2. `hospital_id` — 사용자가 직접 등록한 DB 병원.
    3. 없음 — 그래도 **404 를 내지 않는다.**

    ## 병원이 없어도 초안을 만드는 이유

    예전에는 DB 에서 응급 병원을 찾지 못하면 404 였다. 그 동작은 시드된 가짜 병원
    2건에 의존한 것이고, 시드를 지우자 응급 이메일 기능이 통째로 실패하게 됐다.
    게다가 웹 검색으로 병원 **이메일**이 나오는 경우는 드물다 — 주소를 못 구하는 것은
    오류가 아니라 정상 상황이다. 그래서 `to_email=None` 인 초안을 만들어 두고, 앱이
    보호자에게 수신 주소를 입력받는다. 응급 상황에서 "문서를 못 만든다" 로 막는 것이
    가장 나쁜 결과다.
    """
    pet = get_pet_or_404(db, pet_id, user)

    # 1순위: 요청에 실려 온 AI 검색 결과.
    hospital_name = (body.hospital_name or "").strip()
    hospital_phone = (body.hospital_phone or "").strip()
    to_email = (body.hospital_email or "").strip() or None
    hospital_id: int | None = None

    # 2순위: DB 병원. AI 가 이름을 주지 않았을 때만 조회한다(두 병원이 섞이면
    # 본문에는 A 병원, 수신 주소는 B 병원이 되는 사고가 난다).
    if not hospital_name and not to_email and body.hospital_id:
        hospital = db.get(models.Hospital, body.hospital_id)
        if hospital is not None:
            hospital_id = hospital.id
            hospital_name = hospital.name
            hospital_phone = (hospital.phone or "").strip()
            to_email = (hospital.email or "").strip() or None

    # 전화번호는 저장할 컬럼이 없다 — 이름과 함께 본문에 남긴다(보호자가 초안을
    # 다시 열었을 때 어디로 전화해야 하는지 알아야 한다).
    hospital_label = " · ".join(x for x in [hospital_name, hospital_phone] if x)

    # 제목에는 **한 줄만** 넣는다. 앱이 대화 원문을 그대로 보내므로 여러 줄이 올 수
    # 있는데, 메일 제목에 줄바꿈이 들어가면 헤더가 깨지거나 잘려 보인다.
    # 전체 대화는 본문에 그대로 실린다.
    raw_symptom = (body.symptom_summary or "").strip()
    first_line = next((ln.strip() for ln in raw_symptom.splitlines() if ln.strip()), "")
    symptom = (first_line[:60] + "…") if len(first_line) > 60 else (first_line or "응급 증상")
    now = datetime.now()
    # 품종·나이는 비어 있을 수 있다(직접 등록한 반려동물). 빈 값으로 "( · )" 같은
    # 껍데기를 만들지 않고 있는 항목만 넣는다.
    breed = pet.breed.split(" ·")[0].strip()
    profile = " · ".join(x for x in [breed, age_label(pet.birth_date)] if x)
    subject = f"[응급] {pet.name}{f' ({profile})' if profile else ''} — {symptom}"

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
    email_body = _format_email_body(content, created_label, hospital_label)

    email = models.EmergencyEmail(
        pet_id=pet_id,
        hospital_id=hospital_id,
        to_email=to_email,
        subject=subject,
        body=email_body,
        content=content,
        # 첨부는 비운다. 예전에는 "상태 요약 문서 (PDF)" · "최근 30일 건강 기록" 을
        # `auto: True` 로 표시했지만 실제로 붙는 파일은 하나도 없었다 — 보호자는
        # 첨부가 갔다고 믿고 발송하고, 병원은 본문만 받는다. 내용은 전부 본문
        # 4섹션 텍스트에 들어 있다(보호자가 "메일 본문 텍스트" 방식을 선택했다).
        attachments=[],
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
