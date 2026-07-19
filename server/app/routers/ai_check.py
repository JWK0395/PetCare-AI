import logging
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

logger = logging.getLogger(__name__)

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
    # AI(Agent) 연결 시 RAG 근거 출처. 저장하지 않으면 "지난 대화"를 다시 열었을 때
    # 근거만 사라져 답변의 신뢰 근거를 확인할 수 없다. mock 모드에서는 빈 목록이다.
    "citations",
    # AI 가 찾은 병원도 같은 이유로 저장한다 — 응급 안내를 다시 열었을 때
    # "어느 병원에 연락하라고 했는지"가 사라지면 안 된다. mock 모드에서는 빈 목록.
    "hospitals",
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

    # ---- 세션 먼저 확보 ----
    #
    # **Agent 호출보다 먼저** 만들어야 한다. Agent 는 `conversation_id` 로 LangGraph
    # thread 를 잡아 되묻기를 재개하는데, 첫 턴에 이 값이 없으면 임의 thread 로
    # 시작하고 둘째 턴은 `session-{id}` 라는 **다른 thread** 를 쓴다. 그러면 매 턴이
    # 1턴째가 되어 이미 답한 질문을 계속 반복한다(실제로 그렇게 무한 반복했다).
    session: models.AISession | None = None
    if body.session_id:
        session = db.get(models.AISession, body.session_id)
        if session and session.pet_id != pet_id:
            session = None
    if session is None:
        first_user = next((m.content for m in body.messages if m.role == "user"), "")
        session = models.AISession(pet_id=pet_id, title=first_user[:120])
        db.add(session)
        db.flush()  # id 를 지금 받아야 Agent 에 넘길 수 있다(commit 은 뒤에서 한 번).

    agent = get_agent_client()
    result = agent.health_check(
        pet=pet_payload(pet),
        messages=[m.model_dump() for m in body.messages],
        context=build_context(db, pet),
        # 지역명은 서버가 만들 수 없다(DB 에 사용자 위치가 없다). 앱이 보낸 값을
        # 그대로 전달해야 AI 가 "○○ 24시 동물병원" 같은 검색어를 만들 수 있다.
        region_name=body.region_name,
        # LangGraph 되묻기 재개용 대화 키 — 첫 턴부터 값이 있어야 한다.
        conversation_id=str(session.id),
    )
    try:
        response = schemas.AICheckResponse(**result)
    except ValidationError as exc:
        # http 모드에서 Agent 가 계약과 다른 응답을 보낸 경우 — 500 대신 502 로 변환
        raise HTTPException(
            status_code=502,
            detail=f"Agent 응답이 계약과 다릅니다 (health-check): {exc.error_count()}개 필드 오류",
        )
    # ---- 대화 안에서 위험도를 안정시킨다 ----
    #
    # 되묻는 질문에 답할 때마다 Agent 는 그 turn 을 다시 평가한다. "모름" 처럼 정보가
    # 적은 답이 오면 직전 판단의 근거가 약해져 위험도가 도로 내려간다. 실제로 한
    # 대화에서 consult → consult → emergency → normal 로 널뛰었다. 보호자가 말한
    # 증상은 사라지지 않았는데 마지막 답만 보고 '정상' 으로 끝나는 셈이다.
    #
    # 그래서 **대화 전체의 최고 위험도를 추적**한다. LangGraph 내부의 `merge_risk`
    # (상향 전용)와 같은 규칙을 세션 수준에 적용하는 것이다. '새 체크' 는 새 세션이라
    # 다시 처음부터 시작한다.
    #
    # 되묻는 중의 판정도 **추적에는 포함**한다 — 그 turn 에도 그래프는 증상을 보고
    # 판단했다. 다만 화면에 덮어쓰는 것은 판정이 끝난 turn 에서만 한다.
    continuing = bool(session.messages)
    # 첫 turn 에는 비교할 이전 판정이 없다. `last_risk_level` 컬럼 기본값이
    # 'observe' 라 중립이 아니어서, 그대로 비교하면 첫 답변이 무조건 관찰 이상이 된다.
    previous = session.last_risk_level if continuing else None
    peak = schemas.higher_risk(previous, response.risk_level)
    session.last_risk_level = peak

    # 설명 turn 에는 덮어쓰지 않는다. 최고치 추적은 계속하되(위 `peak`), 화면에
    # 보이는 위험도는 이번 turn 의 판정 그대로 둔다 — 안 그러면 "왜 응급한거죠?"
    # 라는 질문에 응급 카드가 다시 뜬다.
    if response.assessment_turn and not response.awaiting_more_info and peak != response.risk_level:
        logger.info(
            "세션 %s 위험도 유지: 이번 turn %s → 대화 최고치 %s",
            session.id,
            response.risk_level,
            peak,
        )
        response.risk_level = peak  # type: ignore[assignment]

    response.risk_label = schemas.RISK_LABELS.get(response.risk_level, response.risk_level)

    # ---- 세션 저장 ----

    # model_dump(mode="json") 를 거치는 이유: citations/hospitals 는 중첩 Pydantic
    # 모델이라 getattr 로 꺼내 그대로 넣으면 messages(JSON 컬럼) 직렬화가
    # "Object of type HospitalSuggestion is not JSON serializable" 로 죽는다.
    # mock 모드는 두 목록이 항상 비어 있어 지금까지 드러나지 않았을 뿐이다.
    meta = response.model_dump(mode="json", include=set(META_KEYS))
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
