"""AI 체크(mock agent 계약) + 진단서 업로드 검증."""


def _pet_id(client, demo_auth) -> int:
    return client.get("/api/pets", headers=demo_auth).json()[0]["id"]


def test_ai_check_three_states(client, demo_auth):
    """디자인 강제 상태: normal / consult / emergency 3상태만 존재."""
    pet_id = _pet_id(client, demo_auth)
    expects = {
        "demo-normal": ("normal", False),
        "demo-consult": ("consult", False),
        "demo-emergency": ("emergency", True),
    }
    for keyword, (risk, show_hospitals) in expects.items():
        res = client.post(
            f"/api/pets/{pet_id}/ai-check",
            json={"messages": [{"role": "user", "content": keyword}]},
            headers=demo_auth,
        )
        assert res.status_code == 200, res.text
        body = res.json()
        assert body["risk_level"] == risk
        assert body["show_hospitals"] is show_hospitals
        assert body["session_id"]  # 대화가 세션으로 저장됨


def test_ai_check_requires_user_message(client, demo_auth):
    pet_id = _pet_id(client, demo_auth)
    res = client.post(
        f"/api/pets/{pet_id}/ai-check", json={"messages": []}, headers=demo_auth
    )
    assert res.status_code == 422


def test_diary_extract_contract(client, demo_auth):
    """일기 구조화 응답은 items + fields(7개 텍스트 항목) 계약을 지킨다."""
    pet_id = _pet_id(client, demo_auth)
    res = client.post(
        f"/api/pets/{pet_id}/records/extract",
        json={"text": "demo 아침에 사료를 반쯤 남겼다. 산책은 20분."},
        headers=demo_auth,
    )
    assert res.status_code == 200
    fields = res.json()["fields"]
    assert set(fields.keys()) == {
        "food", "water", "activity", "symptom", "stool", "vomit", "notes",
    }


def test_upload_rejects_bad_extension(client, demo_auth):
    pet_id = _pet_id(client, demo_auth)
    res = client.post(
        f"/api/pets/{pet_id}/diagnoses/extract",
        files={"file": ("malware.exe", b"MZ...", "application/octet-stream")},
        headers=demo_auth,
    )
    assert res.status_code == 422


def test_upload_rejects_oversize(client, demo_auth):
    from app.config import settings

    pet_id = _pet_id(client, demo_auth)
    big = b"0" * (settings.max_upload_bytes + 1)
    res = client.post(
        f"/api/pets/{pet_id}/diagnoses/extract",
        files={"file": ("big.pdf", big, "application/pdf")},
        headers=demo_auth,
    )
    assert res.status_code == 413


def test_upload_rejects_empty(client, demo_auth):
    pet_id = _pet_id(client, demo_auth)
    res = client.post(
        f"/api/pets/{pet_id}/diagnoses/extract",
        files={"file": ("empty.pdf", b"", "application/pdf")},
        headers=demo_auth,
    )
    assert res.status_code == 422


def test_summary_content_has_four_sections(client, demo_auth):
    """병원 전달용 요약 — 4섹션 구조 핵심 키."""
    pet_id = _pet_id(client, demo_auth)
    res = client.post(
        f"/api/pets/{pet_id}/summaries",
        json={"risk_level": "consult", "extra_note": ""},
        headers=demo_auth,
    )
    assert res.status_code == 201
    content = res.json()["content"]
    for key in (
        "title", "data_period",              # 1. 문서 정보
        "pet_name", "medications", "allergies",  # 2. 반려동물 정보
        "risk_label", "risk_signs",           # 3. 상태
        "chief_complaint", "major_changes", "progress",  # 4. 주호소·변화
    ):
        assert key in content, f"missing {key}"


def test_ai_check_multi_turn_preserves_meta(client, demo_auth):
    """회귀: 멀티턴 대화에서 이전 assistant 턴의 meta(결과 카드)가 보존된다."""
    pet_id = _pet_id(client, demo_auth)
    # 1턴 — consult 결과 카드(meta) 생성
    res1 = client.post(
        f"/api/pets/{pet_id}/ai-check",
        json={"messages": [{"role": "user", "content": "demo-consult"}]},
        headers=demo_auth,
    )
    assert res1.status_code == 200, res1.text
    body1 = res1.json()
    session_id = body1["session_id"]
    assistant1 = "\n".join(
        p for p in [body1["reply"], body1.get("followup_question") or ""] if p
    )
    # 2턴 — 앱과 동일하게 meta 없는 히스토리 + 새 사용자 메시지 전송
    res2 = client.post(
        f"/api/pets/{pet_id}/ai-check",
        json={
            "session_id": session_id,
            "messages": [
                {"role": "user", "content": "demo-consult"},
                {"role": "assistant", "content": assistant1},
                {"role": "user", "content": "demo-normal"},
            ],
        },
        headers=demo_auth,
    )
    assert res2.status_code == 200, res2.text
    # 저장된 세션에서 1턴 assistant meta 가 살아 있어야 한다
    detail = client.get(f"/api/ai-sessions/{session_id}", headers=demo_auth).json()
    assistants = [m for m in detail["messages"] if m["role"] == "assistant"]
    assert len(assistants) == 2
    assert assistants[0].get("meta"), "1턴 meta 가 지워졌다 (결과 카드 재렌더 불가)"
    assert assistants[0]["meta"]["risk_level"] == "consult"
    assert assistants[1]["meta"]["risk_level"] == "normal"


def test_summary_pdf_escapes_angle_brackets(client, demo_auth):
    """회귀: '<1정/일>' 같은 사용자 입력이 PDF 렌더링을 깨뜨리지 않는다."""
    pet_id = _pet_id(client, demo_auth)
    res = client.post(
        f"/api/pets/{pet_id}/summaries",
        json={"risk_level": "consult", "extra_note": "항생제 <1정/일> & <b>주의</b>"},
        headers=demo_auth,
    )
    assert res.status_code == 201, res.text
    summary_id = res.json()["id"]
    pdf = client.get(f"/api/summaries/{summary_id}/pdf", headers=demo_auth)
    assert pdf.status_code == 200, pdf.text
    assert pdf.content[:4] == b"%PDF"
