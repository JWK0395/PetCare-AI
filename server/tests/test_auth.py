"""인증 — 회원가입/로그인/로그아웃/토큰 검증."""


def test_signup_login_me_logout_flow(client):
    email = "flow@test.local"
    # 가입 → 즉시 로그인 토큰 발급
    res = client.post(
        "/api/auth/signup", json={"email": email, "password": "abcd1234"}
    )
    assert res.status_code == 201
    token = res.json()["token"]
    assert res.json()["user"]["email"] == email

    headers = {"Authorization": f"Bearer {token}"}
    assert client.get("/api/auth/me", headers=headers).json()["email"] == email

    # 로그아웃 → 토큰 무효화
    assert client.post("/api/auth/logout", headers=headers).status_code == 204
    assert client.get("/api/auth/me", headers=headers).status_code == 401

    # 재로그인 가능
    res = client.post(
        "/api/auth/login", json={"email": email, "password": "abcd1234"}
    )
    assert res.status_code == 200


def test_signup_validation(client):
    # 이메일 형식
    res = client.post(
        "/api/auth/signup", json={"email": "not-an-email", "password": "abcd1234"}
    )
    assert res.status_code == 422
    # 비밀번호 최소 길이(4)
    res = client.post(
        "/api/auth/signup", json={"email": "short@test.local", "password": "abc"}
    )
    assert res.status_code == 422
    # 중복 가입
    body = {"email": "dup@test.local", "password": "abcd1234"}
    assert client.post("/api/auth/signup", json=body).status_code == 201
    assert client.post("/api/auth/signup", json=body).status_code == 409


def test_login_wrong_password(client):
    res = client.post(
        "/api/auth/login",
        json={"email": "demo@test.local", "password": "wrong-password"},
    )
    assert res.status_code == 401


def test_relogin_invalidates_previous_token(client):
    """단일 세션: 재로그인하면 이전 토큰은 무효."""
    body = {"email": "single@test.local", "password": "abcd1234"}
    first = client.post("/api/auth/signup", json=body).json()["token"]
    second = client.post("/api/auth/login", json=body).json()["token"]
    assert first != second
    assert (
        client.get(
            "/api/auth/me", headers={"Authorization": f"Bearer {first}"}
        ).status_code
        == 401
    )
    assert (
        client.get(
            "/api/auth/me", headers={"Authorization": f"Bearer {second}"}
        ).status_code
        == 200
    )


def test_protected_routes_require_auth(client):
    assert client.get("/api/pets").status_code == 401
    assert client.get("/api/pets/1/records").status_code == 401
    assert client.post("/api/pets", json={"name": "x"}).status_code == 401


def test_token_stored_hashed(client):
    """DB 에는 토큰 원문이 아닌 해시가 저장된다 (유출 대비)."""
    from app import models
    from app.database import SessionLocal

    body = {"email": "hash@test.local", "password": "abcd1234"}
    token = client.post("/api/auth/signup", json=body).json()["token"]
    db = SessionLocal()
    try:
        user = (
            db.query(models.User).filter(models.User.email == body["email"]).one()
        )
        assert user.token != token  # 원문 미저장
        assert len(user.token) == 64  # sha256 hex
    finally:
        db.close()
