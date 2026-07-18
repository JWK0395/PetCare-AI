from __future__ import annotations

import json
import time
from typing import Any

from langgraph.graph import END, START, StateGraph

from ..models import HandoffOutput, PetCareState
from ..services import (
    get_llm_service,
    get_rag_provider,
)
from ..utils import (
    add_error,
    append_conversation_message,
    format_conversation_history,
    node_result,
)
from .triage import is_post_triage_acknowledgement


def emergency_agent(state: PetCareState) -> dict[str, Any]:
    started = time.perf_counter()

    hits = state.get("emergency_hits", [])
    reasons = "\n".join(
        f"- {item['message']} ({item['rule_id']})"
        for item in hits
    )

    answer = f'''
🚨 지금은 질문을 더 이어가기보다 동물병원에 바로 연락하는 것이 우선이에요.

감지된 고위험 신호:
{reasons}

가까운 동물병원이나 응급 진료가 가능한 병원에 지금 연락하고,
이동 중에는 모카를 최대한 안정시켜 주세요.
병원에는 현재 증상과 시작 시점을 그대로 전달하면 됩니다.

이 안내는 진단이 아니라 응급 가능성을 놓치지 않기 위한 행동 안내입니다.
    '''.strip()

    return node_result(
        state,
        node_name="emergency_agent",
        started_at=started,
        updates={
            "triage_status": "completed",
            "answer": answer,
            "conversation_history": (
                append_conversation_message(
                    state,
                    role="assistant",
                    content=answer,
                )
            ),
        },
    )


def rag_agent(state: PetCareState) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        assessment = state.get("assessment", {})
        symptoms = [
            item.get("code", "")
            for item in assessment.get("symptoms", [])
            if not item.get("negated", False)
        ]
        query = (
            f"사용자 질문: {state['user_input']}\n"
            f"구조화 증상: {', '.join(symptoms) or '없음'}"
        )

        chunks = get_rag_provider().search(
            query=query,
            pet_context=state.get("backend_context", {}),
            limit=5,
        )

        return node_result(
            state,
            node_name="rag_agent",
            started_at=started,
            updates={
                "rag_query": query,
                "rag_chunks": [chunk.model_dump() for chunk in chunks],
                "rag_done": True,
            },
        )
    except Exception as error:
        return add_error(
            state,
            node_name="rag_agent",
            error=error,
            started_at=started,
        )


GENERAL_CHAT_SYSTEM_PROMPT = '''
당신은 PetCare AI의 일반 Chat Agent다.

같은 세션의 이전 대화 기록이 제공되면 자연스럽게 기억해 답한다.
사용자가 앞에서 이름이나 일반 정보를 알려 줬다면 후속 질문에 활용한다.

원칙:
- 제공된 '등록 반려동물 프로필', '등록 일기 원본',
  '등록 일기 정리', '등록 진단서 요약'은 현재 서비스에 저장된 데이터다.
- 사용자가 이름, 종, 품종, 체중, 등록 진단, 특정 날짜의 일기,
  기간별 기록을 물으면 제공된 등록 데이터를 사용해 답한다.
- 사용자가 5/7, 5월 7일처럼 연도를 생략하면 등록 일기의 날짜에서
  같은 월·일을 찾아 해석한다.
- 특정 기간을 물으면 해당 기간의 날짜만 골라 날짜별로 답한다.
- 등록 일기에 실제 기록이 있으면 '제공되지 않았다'고 답하지 않는다.
- 이전 대화와 등록 데이터에 없는 개인정보는 추측하지 않는다.
- 현재 세션 밖의 정보를 안다고 말하지 않는다.
- 건강 상태를 확정 진단하거나 약물과 용량을 권고하지 않는다.
- 친절하고 간결하게 답한다.
'''


NON_EMERGENCY_SYSTEM_PROMPT = '''
당신은 PetCare AI의 비응급 최종 답변 Agent다.

이 응답은 현재 Triage Episode를 종료하는 최종 답변이다.
제공된 반려동물 Context와 RAG 근거만 사용해 보호자에게 설명한다.
비응급 상황을 추가 위험 단계로 나누지 않는다.

반드시 지킬 것:
- 질병을 확정하지 않는다.
- 임의로 약물이나 용량을 권하지 않는다.
- 근거가 부족하면 부족하다고 말한다.
- 응급 규칙에 해당하지 않았다는 사실을
  '정상' 또는 '문제없음'으로 바꾸지 않는다.
- 근거 문서가 있으면 제목, 기관, 버전, 페이지를 표시한다.
- 현재까지 확인된 증상, 결론, 권장 행동, 악화 시 행동만 정리한다.
- 추가 질문을 하지 않는다.
- 번호를 붙여 새로운 질문 목록을 만들지 않는다.
- '알려주세요', '답해주세요', '확인해 주세요',
  '가능하면 바로 알려주세요'로 끝내지 않는다.
- `unknown_additional_unresolved=true`이면
  추가 증상이 없다고 단정하지 말고 '확인하지 못함'으로 표현한다.
- `recovery_hits`가 있으면 현재 회복과 과거 악화를 구분하고,
  과거의 심한 상태는 병원 상담 권고에 반영한다.
- 정보가 부족한 항목은 불확실성으로 표시하되 답변을 요구하지 않는다.
- 마지막 문장은
  '새 증상이나 악화가 생기면 새 상태 체크를 시작해 주세요.'로 끝낸다.
'''


def chat_agent(state: PetCareState) -> dict[str, Any]:
    started = time.perf_counter()

    try:
        if state.get("errors"):
            return node_result(
                state,
                node_name="chat_agent",
                started_at=started,
                updates={},
            )

        if state.get("handoff_requested", False):
            return node_result(
                state,
                node_name="chat_agent",
                started_at=started,
                updates={},
            )

        route = state.get("route")

        conversation_text = format_conversation_history(
            state.get("conversation_history", []),
            exclude_last_user_message=True,
        )

        if route == "general_chat":
            if state.get("post_triage_mode", False):
                previous_triage = state.get(
                    "previous_triage",
                    {},
                )

                if is_post_triage_acknowledgement(
                    state.get("user_input", "")
                ):
                    answer = (
                        "확인했어요. 이번 상태 체크는 이미 종료됐고, "
                        "앞서 안내한 결론은 그대로예요. "
                        "같은 확인 질문은 다시 하지 않을게요. "
                        "새 증상이나 악화가 생기면 새 상태 체크를 시작해 주세요."
                    )
                else:
                    prompt = f"""
                    이전에 완료된 상태 체크:
                    {json.dumps(
                        previous_triage,
                        ensure_ascii=False,
                    )}

                    이전 대화 기록:
                    {conversation_text}

                    현재 사용자 입력:
                    {state["user_input"]}

                    현재 입력은 완료된 상태 체크에 대한 후속 대화다.
                    새로운 증상 Cycle을 시작하지 말고,
                    앞서 내린 결론을 설명하거나 정리해서 답하라.
                    추가 문진 질문을 하지 마라.
                    """

                    answer = get_llm_service().text(
                        system_prompt=GENERAL_CHAT_SYSTEM_PROMPT,
                        user_prompt=prompt,
                    )

                return node_result(
                    state,
                    node_name="chat_agent",
                    started_at=started,
                    updates={
                        "triage_status": "completed",
                        "answer": answer,
                        "conversation_history": (
                            append_conversation_message(
                                state,
                                role="assistant",
                                content=answer,
                            )
                        ),
                    },
                )

            pet_profile = state.get(
                "backend_context",
                {},
            ).get("pet", {})

            prompt = f'''
            이전 대화 기록:
            {conversation_text}

            등록 반려동물 프로필:
            {json.dumps(
                pet_profile,
                ensure_ascii=False,
            )}

            등록 일기 원본(JSON):
            {json.dumps(
                state.get(
                    "backend_context",
                    {},
                ).get("daily_entries", []),
                ensure_ascii=False,
            )}

            등록 일기 정리:
            {state.get("diary_summary", "없음")}

            등록 진단서 요약:
            {state.get("diagnosis_summary", "없음")}

            현재 사용자 입력:
            {state["user_input"]}
            '''

            answer = get_llm_service().text(
                system_prompt=GENERAL_CHAT_SYSTEM_PROMPT,
                user_prompt=prompt,
            )

            return node_result(
                state,
                node_name="chat_agent",
                started_at=started,
                updates={
                    "answer": answer,
                    "conversation_history": (
                        append_conversation_message(
                            state,
                            role="assistant",
                            content=answer,
                        )
                    ),
                },
            )

        if (
            route == "non_emergency"
            and not state.get("rag_done", False)
        ):
            return node_result(
                state,
                node_name="chat_agent",
                started_at=started,
                updates={},
            )

        chunks = state.get("rag_chunks", [])

        evidence_text = "\n\n".join(
            (
                f"[{index}] {chunk.get('organization')} | "
                f"{chunk.get('title')} | "
                f"v{chunk.get('version')} | "
                f"p.{chunk.get('page')}\n"
                f"{chunk.get('text')}"
            )
            for index, chunk in enumerate(chunks, start=1)
        ) or "검색된 근거 없음"

        prompt = f'''
        이전 대화 기록:
        {conversation_text}

        현재 보호자 입력:
        {state["user_input"]}

        반려동물 정보:
        {json.dumps(
            state.get(
                "backend_context",
                {},
            ).get("pet", {}),
            ensure_ascii=False,
        )}

        최근 일기 요약:
        {state.get("diary_summary", "없음")}

        진단서 요약:
        {state.get("diagnosis_summary", "없음")}

        검색 근거:
        {evidence_text}

        현재 Triage Episode 상태:
        - 증상별 세부질문 완료
        - 추가 증상 확인 상태:
          {json.dumps(
              state.get(
                  "question_strategy",
                  {},
              ),
              ensure_ascii=False,
          )}
        - 회복 문맥:
          {json.dumps(
              state.get(
                  "recovery_hits",
                  [],
              ),
              ensure_ascii=False,
          )}
        - Safety Guard 결과: non_emergency
        - 이 답변을 끝으로 현재 상태 체크를 종료해야 함
        '''

        answer = get_llm_service().text(
            system_prompt=NON_EMERGENCY_SYSTEM_PROMPT,
            user_prompt=prompt,
        )

        return node_result(
            state,
            node_name="chat_agent",
            started_at=started,
            updates={
                "triage_status": "completed",
                "answer": answer,
                "conversation_history": (
                    append_conversation_message(
                        state,
                        role="assistant",
                        content=answer,
                    )
                ),
            },
        )

    except Exception as error:
        return add_error(
            state,
            node_name="chat_agent",
            error=error,
            started_at=started,
        )


HANDOFF_SYSTEM_PROMPT = '''
당신은 PetCare AI의 병원 전달용 요약 생성기다.

원칙:
- 입력된 사실만 사용한다.
- 새로운 진단명을 추론하지 않는다.
- 날짜와 변화 내용을 우선한다.
- 기록 누락은 unknown_items에 표시한다.
- 처방 내용은 기존 진단서 원문으로만 표시한다.
- 결과는 사용자가 검토해야 하는 초안이다.
'''

def collect_handoff_context(state: PetCareState) -> dict[str, Any]:
    started = time.perf_counter()
    return node_result(
        state,
        node_name="collect_handoff_context",
        started_at=started,
        updates={},
    )

def generate_handoff(state: PetCareState) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        prompt = f'''
        현재 사용자 입력:
        {state["user_input"]}

        반려동물 프로필:
        {json.dumps(
            state.get("backend_context", {}).get("pet", {}),
            ensure_ascii=False,
        )}

        최근 일기 요약:
        {state.get("diary_summary", "없음")}

        진단서 요약:
        {state.get("diagnosis_summary", "없음")}

        미확인 항목:
        {json.dumps(
            state.get("backend_context", {}).get("unknown_items", []),
            ensure_ascii=False,
        )}
        '''

        handoff = get_llm_service().parse(
            schema=HandoffOutput,
            system_prompt=HANDOFF_SYSTEM_PROMPT,
            user_prompt=prompt,
        )
        assert isinstance(handoff, HandoffOutput)

        answer = (
            "병원 전달용 요약 초안을 만들었어요. "
            "내용을 확인한 뒤 사용하세요."
        )

        return node_result(
            state,
            node_name="generate_handoff",
            started_at=started,
            updates={
                "handoff": handoff.model_dump(),
                "answer": answer,
                "conversation_history": (
                    append_conversation_message(
                        state,
                        role="assistant",
                        content=answer,
                    )
                ),
            },
        )
    except Exception as error:
        return add_error(
            state,
            node_name="generate_handoff",
            error=error,
            started_at=started,
        )

handoff_builder = StateGraph(PetCareState)
handoff_builder.add_node("collect_handoff_context", collect_handoff_context)
handoff_builder.add_node("generate_handoff", generate_handoff)
handoff_builder.add_edge(START, "collect_handoff_context")
handoff_builder.add_edge("collect_handoff_context", "generate_handoff")
handoff_builder.add_edge("generate_handoff", END)
handoff_subgraph = handoff_builder.compile()
