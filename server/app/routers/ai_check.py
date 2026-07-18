from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException
from pydantic import ValidationError
from sqlalchemy import select
from sqlalchemy.orm import Session

from .. import models, schemas
from ..database import get_db
from ..services.agent_client import get_agent_client
from ..services.context import build_context, pet_payload
from .auth import get_current_user
from .pets import get_pet_or_404

router = APIRouter(prefix="/api", tags=["ai-check"])

# assistant meta 로 저장해 지난 대화에서 결과 카드를 다시 그릴 수 있게 한다
META_KEYS = [
    "risk_level",
    "risk_label",
    "trend_summary",
    "reasons",
    "evidence",
    "followup_question",
    "can_generate_summary",
    "show_hospitals",
    "transit_guidance",
]


@router.post("/pets/{pet_id}/ai-check", response_model=schemas.AICheckResponse)
def ai_check(
    pet_id: int,
    body: schemas.AICheckRequest,
    db: Session = Depends(get_db),
    user: models.User = Depends(get_current_user),
):
    """AI 상태 체크.

    서버는 최근 30일 기록·개인 기준선·진단서를 context 로 만들어 Agent 에 전달한다.
    Agent(mock 또는 외부 서비스)가 응급 감지 → 추이 분석 → 추가 질문/판단을 수행한다.
    대화는 ai_sessions 에 저장되어 "지난 대화" 팝업에서 다시 볼 수 있다.
    """
    pet = get_pet_or_404(db, pet_id, user)
    if not body.messages or body.messages[-1].role != "user":
        raise HTTPException(status_code=422, detail="마지막 메시지는 사용자 입력이어야 합니다")

    agent = get_agent_client()
    result = agent.health_check(
        pet=pet_payload(pet),
        messages=[m.model_dump() for m in body.messages],
        context=build_context(db, pet),
    )
    try:
        response = schemas.AICheckResponse(**result)
    except ValidationError as exc:
        # http 모드에서 Agent 가 계약과 다른 응답을 보낸 경우 — 500 대신 502 로 변환
        raise HTTPException(
            status_code=502,
            detail=f"Agent 응답이 계약과 다릅니다 (health-check): {exc.error_count()}개 필드 오류",
        )
    response.risk_label = schemas.RISK_LABELS.get(response.risk_level, response.risk_level)

    # ---- 세션 저장 ----
    session: models.AISession | None = None
    if body.session_id:
        session = db.get(models.AISession, body.session_id)
        if session and session.pet_id != pet_id:
            session = None
    if session is None:
        first_user = next((m.content for m in body.messages if m.role == "user"), "")
        session = models.AISession(pet_id=pet_id, title=first_user[:120])
        db.add(session)

    meta = {key: getattr(response, key) for key in META_KEYS}
    assistant_content = "\n".join(
        part for part in [response.reply, response.followup_question or ""] if part
    )
    # 클라이언트 히스토리(body.messages)에는 meta 가 없으므로, 그대로 덮어쓰면
    # 이전 assistant 턴의 결과 카드(meta)가 지워진다. 기존 저장본의 같은 턴
    # (index·role·content 일치)에서 meta 를 보존한다.
    stored = list(session.messages or [])
    incoming: list[dict] = []
    for i, m in enumerate(body.messages):
        msg = m.model_dump()
        if (
            i < len(stored)
            and stored[i].get("role") == msg["role"]
            and stored[i].get("content") == msg["content"]
            and "meta" in stored[i]
        ):
            msg["meta"] = stored[i]["meta"]
        incoming.append(msg)
    session.messages = [
        *incoming,
        {"role": "assistant", "content": assistant_content, "meta": meta},
    ]
    session.last_risk_level = response.risk_level
    session.updated_at = datetime.utcnow()
    db.commit()
    db.refresh(session)

    response.session_id = session.id
    return response


@router.get("/pets/{pet_id}/ai-sessions", response_model=list[schemas.AISessionSummary])
def list_ai_sessions(
    pet_id: int,
    db: Session = Depends(get_db),
    user: models.User = Depends(get_current_user),
):
    get_pet_or_404(db, pet_id, user)
    sessions = db.scalars(
        select(models.AISession)
        .where(models.AISession.pet_id == pet_id)
        .order_by(models.AISession.updated_at.desc())
    ).all()
    out = []
    for s in sessions:
        item = schemas.AISessionSummary.model_validate(s)
        item.message_count = len(s.messages or [])
        out.append(item)
    return out


def _get_owned_session(
    db: Session, session_id: int, user: models.User
) -> models.AISession:
    session = db.get(models.AISession, session_id)
    if not session:
        raise HTTPException(status_code=404, detail="대화를 찾을 수 없습니다")
    get_pet_or_404(db, session.pet_id, user)  # 소유자 확인
    return session


@router.get("/ai-sessions/{session_id}", response_model=schemas.AISessionDetail)
def get_ai_session(
    session_id: int,
    db: Session = Depends(get_db),
    user: models.User = Depends(get_current_user),
):
    return _get_owned_session(db, session_id, user)


@router.delete("/ai-sessions/{session_id}", status_code=204)
def delete_ai_session(
    session_id: int,
    db: Session = Depends(get_db),
    user: models.User = Depends(get_current_user),
):
    session = _get_owned_session(db, session_id, user)
    db.delete(session)
    db.commit()
