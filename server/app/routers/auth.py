"""이메일/비밀번호 인증.

- 회원가입·로그인 모두 이메일 + 비밀번호만 사용한다 (비밀번호 찾기 없음).
- 로그인하면 불투명 토큰을 발급하고, 앱은 모든 요청에
  `Authorization: Bearer <token>` 헤더로 전달한다.
- 단일 세션: 재로그인하면 이전 토큰은 무효화된다.
- 토큰은 원문 대신 SHA-256 해시로 저장한다 (DB 파일이 유출돼도 세션 탈취 불가).
- 로그인/회원가입은 계정+IP 기준 시도 횟수를 제한한다 (brute-force 방어).
"""

from __future__ import annotations

import hashlib
import re
import secrets
import time
from collections import defaultdict, deque

from fastapi import APIRouter, Depends, Header, HTTPException, Request
from sqlalchemy import select
from sqlalchemy.orm import Session

from .. import models, schemas
from ..config import settings
from ..database import get_db

router = APIRouter(prefix="/api/auth", tags=["auth"])

EMAIL_RE = re.compile(r"^\S+@\S+\.\S+$")
PBKDF2_ITERATIONS = 120_000


# ---------------------------------------------------------------------------
# 비밀번호 해싱 (표준 라이브러리 pbkdf2 — 외부 의존성 없음)
# ---------------------------------------------------------------------------
def hash_password(password: str) -> str:
    salt = secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256", password.encode(), salt.encode(), PBKDF2_ITERATIONS
    ).hex()
    return f"pbkdf2${PBKDF2_ITERATIONS}${salt}${digest}"


def verify_password(password: str, stored: str) -> bool:
    try:
        _, iterations, salt, digest = stored.split("$")
        check = hashlib.pbkdf2_hmac(
            "sha256", password.encode(), salt.encode(), int(iterations)
        ).hex()
        return secrets.compare_digest(check, digest)
    except (ValueError, AttributeError):
        return False


def _hash_token(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


# ---------------------------------------------------------------------------
# 로그인 시도 제한 (in-memory sliding window — 단일 프로세스 로컬 서버 기준)
# ---------------------------------------------------------------------------
_attempts: dict[str, deque[float]] = defaultdict(deque)


def _check_rate_limit(key: str) -> None:
    window = settings.auth_rate_limit_window_seconds
    limit = settings.auth_rate_limit_attempts
    now = time.monotonic()
    bucket = _attempts[key]
    while bucket and now - bucket[0] > window:
        bucket.popleft()
    if len(bucket) >= limit:
        raise HTTPException(
            status_code=429,
            detail="시도가 너무 많습니다. 잠시 후 다시 시도해 주세요",
        )
    bucket.append(now)


def _client_key(request: Request, email: str) -> str:
    host = request.client.host if request.client else "unknown"
    return f"{host}:{email}"


# ---------------------------------------------------------------------------
# 인증 의존성 — 보호가 필요한 모든 라우터가 사용한다
# ---------------------------------------------------------------------------
def get_current_user(
    authorization: str | None = Header(default=None),
    db: Session = Depends(get_db),
) -> models.User:
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="로그인이 필요합니다")
    token = authorization.split(" ", 1)[1].strip()
    if not token:
        raise HTTPException(status_code=401, detail="로그인이 필요합니다")
    user = db.scalar(
        select(models.User).where(models.User.token == _hash_token(token))
    )
    if not user:
        raise HTTPException(
            status_code=401, detail="로그인이 만료되었습니다. 다시 로그인해 주세요"
        )
    return user


def _issue_token(user: models.User, db: Session) -> str:
    token = secrets.token_hex(32)
    user.token = _hash_token(token)  # 원문은 응답으로만 전달, DB 에는 해시만
    db.commit()
    db.refresh(user)
    return token


# ---------------------------------------------------------------------------
# 엔드포인트
# ---------------------------------------------------------------------------
@router.post("/signup", response_model=schemas.AuthResponse, status_code=201)
def signup(
    body: schemas.AuthRequest, request: Request, db: Session = Depends(get_db)
):
    email = body.email.strip().lower()
    _check_rate_limit(_client_key(request, email))
    if not EMAIL_RE.match(email):
        raise HTTPException(status_code=422, detail="올바른 이메일 형식이 아닙니다")
    if len(body.password) < 4:
        raise HTTPException(status_code=422, detail="비밀번호는 4자 이상이어야 합니다")
    if db.scalar(select(models.User).where(models.User.email == email)):
        raise HTTPException(status_code=409, detail="이미 가입된 이메일입니다")

    user = models.User(email=email, password_hash=hash_password(body.password))
    db.add(user)
    db.flush()
    token = _issue_token(user, db)
    return schemas.AuthResponse(token=token, user=schemas.UserOut.model_validate(user))


@router.post("/login", response_model=schemas.AuthResponse)
def login(
    body: schemas.AuthRequest, request: Request, db: Session = Depends(get_db)
):
    email = body.email.strip().lower()
    _check_rate_limit(_client_key(request, email))
    user = db.scalar(select(models.User).where(models.User.email == email))
    if not user or not verify_password(body.password, user.password_hash):
        raise HTTPException(status_code=401, detail="이메일 또는 비밀번호가 올바르지 않습니다")
    token = _issue_token(user, db)
    return schemas.AuthResponse(token=token, user=schemas.UserOut.model_validate(user))


@router.post("/logout", status_code=204)
def logout(
    user: models.User = Depends(get_current_user), db: Session = Depends(get_db)
):
    user.token = None
    db.commit()


@router.get("/me", response_model=schemas.UserOut)
def me(user: models.User = Depends(get_current_user)):
    return user
