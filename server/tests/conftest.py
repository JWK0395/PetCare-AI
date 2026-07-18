"""테스트 공통 설정.

임시 SQLite DB 를 쓰도록 환경 변수를 앱 import *이전에* 설정한다.
(app.config.Settings 는 모듈 import 시점에 읽히므로 순서가 중요하다)
실제 비밀값은 사용하지 않는다 — 모두 테스트 전용 값.
"""

import os
import tempfile
from pathlib import Path

_tmpdir = tempfile.mkdtemp(prefix="petcare-test-")
os.environ["DATABASE_URL"] = f"sqlite:///{(Path(_tmpdir) / 'test.db').as_posix()}"
os.environ["SEED_DEMO_DATA"] = "true"
os.environ["DEMO_USER_EMAIL"] = "demo@test.local"
os.environ["DEMO_USER_PASSWORD"] = "demo-test-1234"
os.environ["DEMO_PASSWORD"] = "demo"
os.environ["AGENT_MODE"] = "mock"
# rate limit 테스트가 다른 테스트를 막지 않도록 넉넉하게
os.environ["AUTH_RATE_LIMIT_ATTEMPTS"] = "50"
os.environ["AUTH_RATE_LIMIT_WINDOW_SECONDS"] = "60"

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402


@pytest.fixture(scope="session")
def client():
    # TestClient 컨텍스트로 lifespan(테이블 생성 + 시드)을 실행한다
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="session")
def demo_auth(client) -> dict:
    """시드된 데모 계정으로 로그인한 Authorization 헤더."""
    res = client.post(
        "/api/auth/login",
        json={"email": "demo@test.local", "password": "demo-test-1234"},
    )
    assert res.status_code == 200, res.text
    return {"Authorization": f"Bearer {res.json()['token']}"}


@pytest.fixture()
def second_user(client) -> dict:
    """데모와 분리된 두 번째 계정 (테스트마다 새로 생성)."""
    import uuid

    email = f"user-{uuid.uuid4().hex[:8]}@test.local"
    res = client.post(
        "/api/auth/signup", json={"email": email, "password": "pw-test-1234"}
    )
    assert res.status_code == 201, res.text
    return {"Authorization": f"Bearer {res.json()['token']}"}
