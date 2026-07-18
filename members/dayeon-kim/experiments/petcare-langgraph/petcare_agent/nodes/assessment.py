from __future__ import annotations

import json
import re
import time
from typing import Any

from langgraph.graph import END, START, StateGraph

from ..models import (
    AssessmentOutput,
    HandoffOutput,
    PetCareState,
)
from ..services import get_llm_service
from ..utils import (
    add_error,
    format_conversation_history,
    node_result,
)
from .context import prepare_backend_context


HANDOFF_REQUEST_PATTERNS: list[str] = [
    r"병원\s*전달용",
    r"병원에\s*(?:전달|보여|제출)",
    r"병원용\s*(?:요약|정리|문서)",
    r"전달용\s*(?:요약|정리|문서)",
    r"진료용\s*(?:요약|정리|문서)",
    r"수의사(?:에게|한테)\s*(?:전달|보여)",
    r"병원에서\s*볼\s*수\s*있게",
]


def detect_handoff_request(text: str) -> bool:
    normalized = text.strip()

    return any(
        re.search(pattern, normalized)
        for pattern in HANDOFF_REQUEST_PATTERNS
    )


ASSESSMENT_SYSTEM_PROMPT = '''
당신은 PetCare AI의 Assessment Graph다.

역할:
1. 현재 사용자 입력을 일반 대화 또는 건강 상태 관련 입력으로 분류한다.
2. 건강 관련 입력이면 현재까지 언급된 증상을 구조화한다.
3. 증상별 세부질문과 사용자의 답변이 있으면 함께 해석한다.
4. 추가 증상 확인 답변에서 새 증상도 추출한다.
5. 사용자가 병원 전달용 요약을 요청했는지 확인한다.

분류 원칙:
- intent는 현재 사용자 입력을 중심으로 결정한다.
- 이전 대화 기록은 대명사, 이름, 생략된 표현을 이해하는 데 사용한다.
- 최근 일기와 진단서는 배경 정보일 뿐이며,
  그것만으로 현재 intent를 health_related로 바꾸지 않는다.
- 앞에서 알려 준 이름이나 일반 정보의 후속 질문은 general_chat이다.

절대 하지 말 것:
- 질병 확정
- 약물 또는 용량 추천
- 응급 여부 최종 결정

intent 기준:
- general_chat: 인사, 서비스 사용법, 단순 대화, 전달용 요약 요청
- health_related: 증상, 상태, 건강 질문, 이상 변화

표준 증상 코드:
respiratory_distress, cyanosis, unconsciousness, seizure,
severe_bleeding, urinary_obstruction, toxin_ingestion,
vomiting, diarrhea, fever, lethargy, appetite_loss,
pain, urinary_abnormality, respiratory_issue,
severe_deterioration.

질문 전략:
- 주증상에 맞는 세부정보를 추출한다.
- 식욕저하 입력을 호흡 문제로 바꾸지 않는다.
- 새로운 증상이 답변에 포함되면 별도 증상으로 추출한다.
- 거의 움직이지 못함, 일어나지 못함, 상태가 너무 안 좋아 보임,
  급격한 악화는 severe_deterioration으로 추출한다.
- Assistant 질문 문구에 포함된 증상 예시는 실제 사용자 증상으로 판단하지 않는다.
- 증상 판정에는 현재 사용자 입력과 Cycle 기록의 answer 필드만 사용한다.

사용자가 "아니요", "없어요", "그렇지 않아요"라고 답했다면
해당 증상을 negated=true로 표시한다.
'''


def assess_input(state: PetCareState) -> dict[str, Any]:
    started = time.perf_counter()

    try:
        # Assistant 질문에 들어 있는 응급 증상 예시가
        # 실제 사용자 증상으로 오인되지 않도록 사용자 답변만 전달합니다.
        follow_up_text = json.dumps(
            [
                {
                    "field": item.get("field"),
                    "kind": item.get("kind"),
                    "symptom": item.get("symptom"),
                    "answer": item.get("answer"),
                    "answer_status": item.get(
                        "answer_status"
                    ),
                }
                for item in state.get(
                    "follow_up_history",
                    [],
                )
            ],
            ensure_ascii=False,
        )

        # Assessment에서는 이전 Assistant 문진 문구를 제외합니다.
        # 대화 기억은 유지하되 증상 판정에는 사용자 발화만 사용합니다.
        assessment_history = [
            item
            for item in state.get(
                "conversation_history",
                [],
            )
            if item.get("role") == "user"
        ]

        conversation_text = format_conversation_history(
            assessment_history,
            exclude_last_user_message=True,
        )

        prompt = f'''
        현재 사용자 입력:
        {state["user_input"]}

        이전 대화 기록:
        {conversation_text}

        증상별 질문 Cycle 기록:
        {follow_up_text}

        현재 질문 전략 상태:
        {json.dumps(
            state.get("question_strategy", {}),
            ensure_ascii=False,
        )}

        최근 일기 요약:
        {state.get("diary_summary", "없음")}

        진단서 요약:
        {state.get("diagnosis_summary", "없음")}
        '''

        assessment = get_llm_service().parse(
            schema=AssessmentOutput,
            system_prompt=ASSESSMENT_SYSTEM_PROMPT,
            user_prompt=prompt,
        )
        assert isinstance(assessment, AssessmentOutput)

        # 명시적인 병원 전달용 요청은 LLM의 boolean 결과에만
        # 의존하지 않고 코드 규칙으로도 확인합니다.
        handoff_requested = (
            assessment.handoff_requested
            or detect_handoff_request(
                state.get("user_input", "")
            )
        )

        assessment_payload = assessment.model_dump()
        assessment_payload["handoff_requested"] = handoff_requested

        # 상태 체크 도중의 "모르겠어", "확인 못함" 같은 짧은 답변을
        # 일반 대화로 오분류하지 않도록 health_related를 유지합니다.
        resolved_intent = assessment.intent

        if (
            state.get("triage_status") == "collecting"
            or state.get("follow_up_history")
        ):
            resolved_intent = "health_related"

        assessment_payload["intent"] = resolved_intent

        updates: dict[str, Any] = {
            "assessment": assessment_payload,
            "handoff_requested": handoff_requested,
        }

        # Handoff 요청은 건강 위험 분류가 아니라 문서 생성 의도이므로
        # 일반 대화 경로에서 Chat Agent를 거쳐 Handoff Subgraph로 보냅니다.
        if (
            resolved_intent == "general_chat"
            or handoff_requested
        ):
            updates["route"] = "general_chat"

        return node_result(
            state,
            node_name="assess_input",
            started_at=started,
            updates=updates,
        )

    except Exception as error:
        return add_error(
            state,
            node_name="assess_input",
            error=error,
            started_at=started,
        )


assessment_builder = StateGraph(PetCareState)

assessment_builder.add_node(
    "prepare_backend_context",
    prepare_backend_context,
)
assessment_builder.add_node("assess_input", assess_input)

assessment_builder.add_edge(START, "prepare_backend_context")
assessment_builder.add_edge(
    "prepare_backend_context",
    "assess_input",
)
assessment_builder.add_edge("assess_input", END)

assessment_graph = assessment_builder.compile()
