from petcare_agent.contracts import CONTRACT_SCHEMA_FILES, load_json_schema
from petcare_agent.schemas.graph_state import (
    GraphRequest,
    GraphResponse,
    PetCareGraphState,
)
from petcare_agent.schemas.llm_outputs import (
    AnswerGuardReviewOutput,
    ChecklistExtractionOutput,
    GeneralPetCareAnswerOutput,
    HandoffSummaryOutput,
    IntentClassificationOutput,
    SocialChatOutput,
    StateExtractionOutput,
    TurnUnderstandingOutput,
)
from petcare_agent.schemas.triage import ChecklistItem, ChecklistTemplate, RiskResult, RuleHit


def test_all_contract_json_schemas_parse() -> None:
    for schema_name in CONTRACT_SCHEMA_FILES:
        schema = load_json_schema(schema_name)
        assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
        assert schema["title"]


def test_graph_request_response_samples_validate_with_pydantic() -> None:
    request = GraphRequest(
        request_id="req_20260716_001",
        conversation_id="conv_abc",
        pet_id=1,
        user_input="cat is coughing",
        locale="ko-KR",
        timezone="Asia/Seoul",
        timestamp="2026-07-16T09:00:00+09:00",
        user_location={"lat": 37.5665, "lng": 126.9780, "permission": "granted"},
    )
    response = GraphResponse(
        response_id="res_20260716_001",
        conversation_id=request.conversation_id,
        route="question_manager",
        risk_level="unknown",
        assistant_message="?덉쟾 ?뺤씤???꾪빐 ??媛吏瑜?癒쇱? ?뺤씤?댁빞 ?⑸땲??",
        needs_user_response=True,
        follow_up_question={
            "question_id": "resp_open_mouth_breathing",
            "text": "諛섎젮?숇Ъ??吏湲??낆쓣 踰뚮━怨??⑥쓣 ?ш퀬 ?덈굹??",
        },
        handoff={"type": "none", "summary": None, "summary_json": None, "email_draft": None},
        emergency={"is_emergency": False, "triggered_rules": []},
    )

    assert request.pet_id == 1
    assert response.route == "question_manager"


def test_graph_contract_fields_match_models() -> None:
    request_schema = load_json_schema("agent_graph_request")
    response_schema = load_json_schema("agent_graph_response")

    assert set(GraphRequest.model_fields) == set(request_schema["properties"])
    assert set(GraphResponse.model_fields) == set(response_schema["properties"])


def test_pydantic_graph_state_defaults_match_documented_route() -> None:
    state = PetCareGraphState(user_input="hello")

    assert state.next_route == "intent_classifier"
    assert state.safety_question_turns == 0
    assert state.context.recent_daily_entries == []


def test_triage_models_and_schema_fields_align() -> None:
    schema = load_json_schema("triage_checklist")
    item = ChecklistItem(
        item_id="open_mouth_breathing",
        label="Open mouth breathing",
        type="boolean",
        value=None,
        confidence="unknown",
        source="user_input",
        asked_count=0,
    )
    template = ChecklistTemplate(
        checklist_id="cat_cough_triage",
        species="cat",
        chief_complaint="cough",
        required_items=[item],
    )
    result = RiskResult(
        risk_level="unknown",
        confidence="low",
        action="needs_more_info",
        triggered_rules=[
            RuleHit(
                rule_id="Q_MISSING_001",
                result="needs_more_info",
                condition="required item is unknown",
            )
        ],
        missing_items=["open_mouth_breathing"],
        requires_more_info=True,
    )

    assert set(ChecklistTemplate.model_fields) == set(schema["properties"])
    assert template.required_items[0].item_id == "open_mouth_breathing"
    assert result.triggered_rules[0].rule_id == "Q_MISSING_001"


def test_llm_output_contract_fields_align() -> None:
    schema = load_json_schema("llm_structured_outputs")
    defs = schema["$defs"]

    assert set(IntentClassificationOutput.model_fields) == set(
        defs["IntentClassificationOutput"]["properties"]
    )
    assert "social_chat" in defs["Intent"]["enum"]
    assert set(SocialChatOutput.model_fields) == set(
        defs["SocialChatOutput"]["properties"]
    )
    assert set(GeneralPetCareAnswerOutput.model_fields) == set(
        defs["GeneralPetCareAnswerOutput"]["properties"]
    )
    assert set(StateExtractionOutput.model_fields) == set(
        defs["StateExtractionOutput"]["properties"]
    )
    assert set(TurnUnderstandingOutput.model_fields) == set(
        defs["TurnUnderstandingOutput"]["properties"]
    )
    assert set(ChecklistExtractionOutput.model_fields) == set(
        defs["ChecklistExtractionOutput"]["properties"]
    )
    assert set(AnswerGuardReviewOutput.model_fields) == set(
        defs["AnswerGuardReviewOutput"]["properties"]
    )
    assert set(HandoffSummaryOutput.model_fields) == set(
        defs["HandoffSummaryOutput"]["properties"]
    )

    intent = IntentClassificationOutput(
        intent="general_chat",
        confidence="high",
        chief_complaint=None,
        requires_db_context=False,
        requires_safety_screening=False,
        red_flag_mentioned=False,
    )
    social_chat = SocialChatOutput(assistant_message="hello")
    general_answer = GeneralPetCareAnswerOutput(assistant_message="walk twice daily")
    state = StateExtractionOutput(
        species="cat",
        symptoms=["coughing"],
        duration=None,
        current_status={"appetite": "unknown", "activity": "unknown", "water": "unknown"},
        course_pattern="unknown",
        negated_findings=[],
        uncertain_findings=[],
    )
    turn = TurnUnderstandingOutput(
        intent="symptom_check",
        confidence="high",
        chief_complaint="cough",
        requires_db_context=True,
        requires_safety_screening=True,
        red_flag_mentioned=False,
        state=state,
        social_chat=None,
    )
    checklist = ChecklistExtractionOutput(
        checklist_id="cat_cough_triage",
        updates=[
            {
                "item_id": "open_mouth_breathing",
                "value": False,
                "confidence": "high",
                "evidence": "mouth is closed",
            }
        ],
    )
    review = AnswerGuardReviewOutput(
        status="revised",
        unsafe_phrases=["it is fine"],
        revised_answer="?꾩옱 ?뺣낫留뚯쑝濡쒕뒗 利됱떆 ?묎툒 ?좏샇媛 ?쒕졆?섏? ?딆뒿?덈떎.",
    )
    handoff = HandoffSummaryOutput(
        type="emergency",
        summary="?묎툒 ?꾪뿕 ?좏샇媛 蹂닿퀬?섏뿀?듬땲??",
        email_draft="珥덉븞?낅땲?? ?대찓?쇱? ?꾩넚?섏? ?딆븯?듬땲??",
    )

    assert intent.intent == "general_chat"
    assert social_chat.assistant_message == "hello"
    assert general_answer.assistant_message == "walk twice daily"
    assert state.species == "cat"
    assert turn.state.species == "cat"
    assert checklist.updates[0].item_id == "open_mouth_breathing"
    assert review.status == "revised"
    assert handoff.type == "emergency"


