"""AI 체크(mock agent 계약) + 진단서 업로드 검증."""


def _pet_id(client, demo_auth) -> int:
    return client.get("/api/pets", headers=demo_auth).json()[0]["id"]


def test_ai_check_reports_not_connected(client, demo_auth):
    """mock 모드는 판단하지 않고 'AI 미연결' 사실만 돌려준다.

    예전에는 `demo-normal`/`demo-consult`/`demo-emergency` 로 3상태를 강제로 만들어
    화면을 채웠다. 그 응답의 근거·추가질문·이동 안내는 전부 하드코딩이었고 화면에는
    예시라는 표시가 없어, 보호자가 AI 판독 결과로 읽을 수 있었다. 지금은 값을 만들지
    않으므로 **어떤 입력에도 같은 미연결 응답**이 나와야 한다.
    """
    pet_id = _pet_id(client, demo_auth)
    for text in ("demo-emergency", "숨을 잘 못 쉬어요", "오늘 산책 잘 했어요"):
        res = client.post(
            f"/api/pets/{pet_id}/ai-check",
            json={"messages": [{"role": "user", "content": text}]},
            headers=demo_auth,
        )
        assert res.status_code == 200, res.text
        body = res.json()
        assert body["risk_level"] == "normal"
        assert "AI 가 연결되지 않아" in body["reply"]
        assert body["show_hospitals"] is False
        assert body["can_generate_summary"] is False
        # 지어낸 근거·질문·이동 안내·병원이 하나도 없어야 한다
        assert body["evidence"] == ""
        assert body["reasons"] == []
        assert body["trend_summary"] == ""
        assert body["followup_question"] is None
        assert body["transit_guidance"] == []
        assert body["citations"] == []
        assert body["hospitals"] == []
        assert body["session_id"]  # 대화는 그대로 세션으로 저장된다


def test_ai_check_requires_user_message(client, demo_auth):
    pet_id = _pet_id(client, demo_auth)
    res = client.post(
        f"/api/pets/{pet_id}/ai-check", json={"messages": []}, headers=demo_auth
    )
    assert res.status_code == 422


def test_diary_extract_contract(client, demo_auth):
    """일기 구조화 응답은 items + fields(7개 텍스트 항목) 계약을 지킨다.

    mock 은 값을 추출하지 않는다 — 추출 결과는 보호자 확인 한 번으로 daily_entries 에
    저장되므로, 정규식으로 지어낸 "사료 반쯤 남김" 이 진짜 건강 기록으로 남았었다.
    지금은 7개 키를 가진 **빈 fields** 만 돌려준다(계약은 유지, 값은 없음).
    """
    pet_id = _pet_id(client, demo_auth)
    res = client.post(
        f"/api/pets/{pet_id}/records/extract",
        json={"text": "아침에 사료를 반쯤 남겼다. 산책은 20분."},
        headers=demo_auth,
    )
    assert res.status_code == 200
    body = res.json()
    assert set(body["fields"].keys()) == {
        "food", "water", "activity", "symptom", "stool", "vomit", "notes",
    }
    assert all(value == "" for value in body["fields"].values())
    assert body["items"] == []


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
    """병원 전달용 요약 — 4섹션 구조 핵심 키.

    키는 전부 있어야 하고(앱·PDF 렌더링 계약), 값은 DB 에 실제로 있는 것만 담긴다.
    """
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
    # 실제 DB 값에서 온 항목
    assert content["pet_name"] == "콩이"
    assert content["data_period"]


def test_seed_has_no_hospitals(client):
    """시드는 병원을 만들지 않는다.

    예전에는 전화번호·이메일이 전부 가짜인 응급 병원 2건을 시드했고, 응급 화면이
    그것을 "주변 24시 동물병원" 으로 보여줬다. 테이블·API 는 사용자가 직접 등록할
    자리로 남기고, 데이터만 없앤다.
    """
    res = client.get("/api/hospitals")
    assert res.status_code == 200
    assert res.json() == []


def test_emergency_email_without_hospital_is_draft(client, demo_auth):
    """병원 정보가 하나도 없어도 404 가 아니라 201 초안이 만들어진다.

    웹 검색으로 병원 이메일을 못 구하는 것은 오류가 아니라 정상 상황이다. 초안을
    막으면 응급 상황에서 문서 자체를 못 만든다 — 주소는 앱에서 보호자가 입력한다.
    """
    pet_id = _pet_id(client, demo_auth)
    res = client.post(
        f"/api/pets/{pet_id}/emergency-emails",
        json={"symptom_summary": "호흡곤란 · 청색증"},
        headers=demo_auth,
    )
    assert res.status_code == 201, res.text
    email = res.json()
    assert email["to_email"] is None
    assert email["hospital_id"] is None
    # 실제로 붙는 파일이 없으므로 첨부 표기도 없어야 한다
    assert email["attachments"] == []
    assert "호흡곤란" in email["subject"]
    assert "호흡곤란" in email["body"]


def test_emergency_email_uses_ai_hospital(client, demo_auth):
    """AI 가 찾은 병원(hospital_name/email)이 DB 조회보다 우선한다."""
    pet_id = _pet_id(client, demo_auth)
    res = client.post(
        f"/api/pets/{pet_id}/emergency-emails",
        json={
            "hospital_name": "테스트동물의료센터",
            "hospital_email": "er@test.local",
            "hospital_phone": "02-000-0000",
            "symptom_summary": "경련",
        },
        headers=demo_auth,
    )
    assert res.status_code == 201, res.text
    email = res.json()
    assert email["to_email"] == "er@test.local"
    # DB 에 없는 병원이므로 hospital_id 는 비어 있고, 이름·전화는 본문에 남는다
    assert email["hospital_id"] is None
    assert "테스트동물의료센터" in email["body"]
    assert "02-000-0000" in email["body"]


def test_ai_check_multi_turn_preserves_meta(client, demo_auth):
    """회귀: 멀티턴 대화에서 이전 assistant 턴의 meta(결과 카드)가 보존된다."""
    pet_id = _pet_id(client, demo_auth)
    # 1턴 — 결과 카드(meta) 생성
    res1 = client.post(
        f"/api/pets/{pet_id}/ai-check",
        json={"messages": [{"role": "user", "content": "어제부터 밥을 잘 안 먹어요"}]},
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
                {"role": "user", "content": "어제부터 밥을 잘 안 먹어요"},
                {"role": "assistant", "content": assistant1},
                {"role": "user", "content": "물은 잘 마셔요"},
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
    assert assistants[0]["meta"]["risk_level"] == "normal"
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
