from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import settings
from .database import Base, engine
from .routers import auth


@asynccontextmanager
async def lifespan(app: FastAPI):
    Base.metadata.create_all(bind=engine)
    yield


app = FastAPI(
    title="PetCare AI Server",
    description="멍냥케어 — 반려동물 건강관리 AI Agent 서비스 로컬 서버",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS — RN 앱은 CORS 를 쓰지 않으므로 로컬 개발 기본값은 "*".
# 웹 클라이언트를 붙일 때는 CORS_ORIGINS 에 정확한 origin 을 나열할 것.
# (와일드카드 origin 과 credentials 동시 사용은 브라우저가 거부하는 무효 조합이라 분기한다)
_origins = settings.cors_origin_list
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_credentials="*" not in _origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router)


@app.get("/health")
def health():
    return {"status": "ok", "agent_mode": settings.agent_mode}
