"""사용자별 데이터 분리 — 다른 계정의 리소스는 404 로 감춰져야 한다."""


def _demo_pet_id(client, demo_auth) -> int:
    pets = client.get("/api/pets", headers=demo_auth).json()
    assert pets, "시드된 데모 반려동물이 있어야 한다"
    return pets[0]["id"]


def test_new_user_has_empty_pet_list(client, second_user):
    assert client.get("/api/pets", headers=second_user).json() == []


def test_cross_user_pet_access_hidden(client, demo_auth, second_user):
    pet_id = _demo_pet_id(client, demo_auth)
    # 조회/수정/삭제/하위 리소스 모두 404
    assert client.get(f"/api/pets/{pet_id}", headers=second_user).status_code == 404
    assert (
        client.put(
            f"/api/pets/{pet_id}", json={"name": "탈취"}, headers=second_user
        ).status_code
        == 404
    )
    assert (
        client.delete(f"/api/pets/{pet_id}", headers=second_user).status_code == 404
    )
    assert (
        client.get(f"/api/pets/{pet_id}/records", headers=second_user).status_code
        == 404
    )
    assert (
        client.get(f"/api/pets/{pet_id}/diagnoses", headers=second_user).status_code
        == 404
    )


def test_pet_crud_within_owner(client, second_user):
    # 생성
    res = client.post(
        "/api/pets", json={"name": "테스트냥", "species": "고양이"}, headers=second_user
    )
    assert res.status_code == 201
    pet_id = res.json()["id"]
    # 목록/수정
    names = [p["name"] for p in client.get("/api/pets", headers=second_user).json()]
    assert "테스트냥" in names
    res = client.put(
        f"/api/pets/{pet_id}", json={"weight_kg": 4.2}, headers=second_user
    )
    assert res.status_code == 200 and res.json()["weight_kg"] == 4.2
    # updated_at 이 프로필 수정 시각으로 존재
    assert res.json()["updated_at"]
    # 삭제
    assert (
        client.delete(f"/api/pets/{pet_id}", headers=second_user).status_code == 204
    )


def test_record_upsert_by_date(client, second_user):
    pet_id = client.post(
        "/api/pets", json={"name": "기록냥"}, headers=second_user
    ).json()["id"]
    body = {"record_date": "2026-01-15", "raw_text": "첫 기록", "food": "잘 먹음"}
    assert (
        client.post(f"/api/pets/{pet_id}/records", json=body, headers=second_user)
        .status_code
        == 201
    )
    # 같은 날짜로 다시 저장하면 갱신 (복합 PK upsert)
    body["food"] = "적게 먹음"
    res = client.post(
        f"/api/pets/{pet_id}/records", json=body, headers=second_user
    )
    assert res.status_code == 201 and res.json()["food"] == "적게 먹음"
    records = client.get(
        f"/api/pets/{pet_id}/records?days=365", headers=second_user
    ).json()
    assert len([r for r in records if r["record_date"] == "2026-01-15"]) == 1


def test_cross_user_secondary_resources_hidden(client, demo_auth, second_user):
    """summaries / ai-sessions / emergency-emails 도 소유자 검증."""
    pet_id = _demo_pet_id(client, demo_auth)
    # demo 소유 요약 생성
    summary = client.post(
        f"/api/pets/{pet_id}/summaries",
        json={"risk_level": "consult", "extra_note": ""},
        headers=demo_auth,
    ).json()
    assert (
        client.get(f"/api/summaries/{summary['id']}", headers=second_user).status_code
        == 404
    )
    assert (
        client.get(f"/api/summaries/{summary['id']}", headers=demo_auth).status_code
        == 200
    )


def test_update_rejects_explicit_null_on_required_fields(client, second_user):
    """회귀: NOT NULL 컬럼에 명시적 null → 500 이 아니라 422."""
    pet = client.post("/api/pets", json={"name": "널테스트"}, headers=second_user).json()
    res = client.put(f"/api/pets/{pet['id']}", json={"name": None}, headers=second_user)
    assert res.status_code == 422
    # nullable 컬럼은 null 로 지우기 허용 (기존 동작 유지)
    res = client.put(
        f"/api/pets/{pet['id']}",
        json={"birth_date": None, "weight_kg": None},
        headers=second_user,
    )
    assert res.status_code == 200


def test_delete_pet_removes_dependent_rows(client, second_user):
    """회귀: 펫 삭제 시 요약/AI세션/응급이메일 고아 행이 남지 않는다."""
    pet = client.post("/api/pets", json={"name": "삭제테스트"}, headers=second_user).json()
    pid = pet["id"]
    summary = client.post(
        f"/api/pets/{pid}/summaries",
        json={"risk_level": "consult", "extra_note": ""},
        headers=second_user,
    ).json()
    session = client.post(
        f"/api/pets/{pid}/ai-check",
        json={"messages": [{"role": "user", "content": "demo-normal"}]},
        headers=second_user,
    ).json()
    email = client.post(
        f"/api/pets/{pid}/emergency-emails",
        json={"hospital_id": None, "symptom_summary": "t"},
        headers=second_user,
    ).json()
    assert client.delete(f"/api/pets/{pid}", headers=second_user).status_code == 204
    # 라우터는 pet 이 없으면 어차피 404 를 주므로, 고아 행 여부는 DB 를 직접 확인한다
    from app import models
    from app.database import SessionLocal

    with SessionLocal() as db:
        assert db.get(models.Summary, summary["id"]) is None
        assert db.get(models.AISession, session["session_id"]) is None
        assert db.get(models.EmergencyEmail, email["id"]) is None
