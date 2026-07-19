"""AI Agent 서비스 (stub) — PetCare 메인 서버의 AGENT_MODE=http 대상.

실행:
    cd ai
    # server/.venv 재사용 가능 (fastapi/uvicorn/httpx/pydantic 만 필요)
    ..\server\.venv\Scripts\python -m uvicorn app.main:app --port 8100

그러면 server/.env 에서:
    AGENT_MODE=http
    AGENT_BASE_URL=http://127.0.0.1:8100
로 두고 메인 서버를 켜면, 앱의 AI 기능이 이 서비스로 전달된다.

각 엔드포인트는 ai/README.md 의 계약을 따른다. 페이지별 입출력 스켈레톤은
각 파일로 분리되어 있으니, 직접 만든 AI 코드를 해당 파일의 run_* 에 연결하면 된다:
    - health_check.py       (AI 체크)  → run_health_check
    - diary_extract.py      (기록)     → run_diary_extract
    - diagnosis_extract.py  (진료)     → run_diagnosis_extract
    - graph.py              (요약)     → run_summary
"""

import secrets

from fastapi import Body, Depends, FastAPI, Header, HTTPException

from .config import settings
from .diagnosis_extract import run_diagnosis_extract
from .diary_extract import run_diary_extract
from .graph import run_summary
from .health_check import run_health_check

app = FastAPI(title="PetCare AI Agent (stub)", version="0.1.0")


def verify_agent_key(authorization: str = Header(default="")) -> None:
    """AGENT_API_KEY 가 설정된 경우 메인 서버가 보내는 Bearer 토큰을 검증한다.

    비어 있으면(기본값) 검증 없이 통과 — 로컬 개발 편의.
    """
    if not settings.agent_api_key:
        return
    expected = f"Bearer {settings.agent_api_key}"
    if not secrets.compare_digest(authorization, expected):
        raise HTTPException(status_code=401, detail="Authorization 헤더가 올바르지 않습니다")


@app.get("/health")
def health():
    return {"status": "ok", "service": "petcare-ai-agent-stub"}


@app.post("/agent/diary-extract", dependencies=[Depends(verify_agent_key)])
def diary_extract(body: dict = Body(...)):
    return run_diary_extract(
        pet=body.get("pet", {}),
        text=body.get("text", ""),
        record_date=body.get("record_date", ""),
        context=body.get("context", {}),
    )


@app.post("/agent/diagnosis-extract", dependencies=[Depends(verify_agent_key)])
def diagnosis_extract(body: dict = Body(...)):
    return run_diagnosis_extract(
        pet=body.get("pet", {}),
        file_name=body.get("file_name", ""),
        file_text=body.get("file_text", ""),
    )


@app.post("/agent/health-check", dependencies=[Depends(verify_agent_key)])
def health_check(body: dict = Body(...)):
    return run_health_check(
        pet=body.get("pet", {}),
        messages=body.get("messages", []),
        context=body.get("context", {}),
        # 지역명이 없으면 LangGraph 가 병원 검색을 건너뛴다 — 빠뜨리면 안 되는 값이다.
        region_name=body.get("region_name"),
        # 대화 식별자가 없으면 매 요청이 새 thread 가 되어 되묻기를 재개하지 못한다.
        # (이 줄이 빠져 있어서, 서버가 값을 보내는데도 AI 는 매번 1턴째로 시작했다.)
        conversation_id=body.get("conversation_id"),
    )


@app.post("/agent/summary", dependencies=[Depends(verify_agent_key)])
def summary(body: dict = Body(...)):
    return run_summary(
        pet=body.get("pet", {}),
        risk_level=body.get("risk_level", "observe"),
        extra_note=body.get("extra_note", ""),
        context=body.get("context", {}),
    )
