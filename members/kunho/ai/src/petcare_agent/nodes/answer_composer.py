"""Compose final user-facing answers after safety and RAG retrieval."""

from __future__ import annotations

import json
from typing import Any

from petcare_agent.llm.client import StructuredOutputClient, call_structured_output
from petcare_agent.localization import display_list, localize_change_summary, wants_korean
from petcare_agent.prompts import load_prompt
from petcare_agent.schemas.graph_state import (
    PetCareGraphState,
    RAGCitation,
    RetrievedChunk,
)
from petcare_agent.schemas.llm_outputs import GeneralPetCareAnswerOutput

ANSWER_SAFE_RISK_LEVELS = {"urgent", "non_emergency", "unknown"}

RISK_GUIDANCE: dict[str, str] = {
    "urgent": (
        "Some details suggest this should be checked by a veterinarian promptly. "
        "This is not the same as confirming an emergency, but it should not be ignored "
        "if signs continue or worsen."
    ),
    "non_emergency": (
        "Based on the current information, no immediate emergency signal is clear. "
        "Please keep monitoring closely and seek veterinary care if symptoms persist, "
        "worsen, or new red flags appear."
    ),
    "unknown": (
        "There is not enough reliable information to judge the risk confidently. "
        "Use a cautious approach, keep observing breathing, energy, appetite, and gum color, "
        "and consider veterinary advice if anything feels worse than usual."
    ),
}

RISK_GUIDANCE_KO: dict[str, str] = {
    "urgent": (
        "현재 정보 중 일부는 빠른 수의사 확인이 필요한 신호일 수 있습니다. "
        "응급이라고 단정하는 것은 아니지만, 증상이 계속되거나 나빠지면 "
        "가볍게 보지 말고 진료 상담을 권합니다."
    ),
    "non_emergency": (
        "현재 정보만으로는 즉시 응급 신호가 뚜렷하지 않습니다. "
        "다만 증상이 지속되거나 악화되거나 새 위험 신호가 보이면 "
        "수의사 진료를 받아 주세요."
    ),
    "unknown": (
        "위험도를 자신 있게 판단하기에는 정보가 부족합니다. "
        "호흡, 활력, 식욕, 잇몸 색을 조심스럽게 관찰하고 "
        "평소보다 나빠 보이면 수의사에게 상담하세요."
    ),
}

GENERAL_CHAT_GUIDANCE = (
    "I can help with general pet-care information, but this does not replace a veterinary "
    "exam or diagnosis."
)

GENERAL_CHAT_GUIDANCE_KO = (
    "일반적인 반려동물 건강 정보는 도울 수 있지만, 수의사의 진찰이나 진단을 "
    "대신할 수는 없습니다."
)

GENERAL_PETCARE_FALLBACK = (
    "I tried to answer your pet-care question directly, but I could not generate a "
    "complete response right now. For general care, tailor routines to your pet's age, "
    "health, breed, and energy level, and ask a veterinarian if symptoms or sudden "
    "changes are involved."
)

GENERAL_PETCARE_FALLBACK_KO = (
    "반려동물 케어 질문에 바로 답하려 했지만, 지금은 충분한 답변을 생성하지 "
    "못했어요. 일반 관리는 나이, 건강 상태, 품종, 활동량에 맞춰 조절하고, "
    "증상이나 갑작스러운 변화가 함께 있으면 수의사에게 확인해 주세요."
)

VISIT_INTENT_PROMPT = (
    "Are you considering a hospital visit? If yes, I can prepare a concise visit summary "
    "with the symptoms and recent changes. If no or undecided, I can keep the guidance focused "
    "on what to watch next."
)

VISIT_INTENT_PROMPT_KO = (
    "병원 방문을 고려하고 계신가요? 그렇다면 증상과 최근 변화를 정리한 "
    "병원 전달용 요약을 준비할 수 있어요. 아직 결정하지 않았다면, "
    "다음에 무엇을 관찰할지 중심으로 안내할게요."
)


def compose_answer(
    state: PetCareGraphState,
    *,
    llm_client: StructuredOutputClient | None = None,
) -> PetCareGraphState:
    """Build the final draft from safety status, context, and official-source evidence."""

    next_state = state.model_copy(deep=True)
    if next_state.risk_level not in ANSWER_SAFE_RISK_LEVELS:
        return next_state

    if _should_generate_general_petcare_answer(next_state):
        return _compose_general_petcare_answer(next_state, llm_client=llm_client)

    message_parts = [_opening_guidance(next_state)]

    change_summary = localize_change_summary(
        next_state.change_detection.summary,
        next_state.locale,
    )
    if change_summary:
        if wants_korean(next_state.locale):
            message_parts.append(f"최근 기록 비교: {change_summary}")
        else:
            message_parts.append(f"Recent log comparison: {change_summary}")

    symptoms = _known_symptoms(next_state)
    if symptoms:
        if wants_korean(next_state.locale):
            message_parts.append(f"현재 확인된 증상: {', '.join(symptoms)}.")
        else:
            message_parts.append(f"Reported symptoms to keep in view: {', '.join(symptoms)}.")

    evidence_summary = _evidence_summary(next_state)
    if evidence_summary:
        message_parts.append(evidence_summary)

    message_parts.append(_visit_intent_prompt(next_state))

    citation_summary = _citation_summary(next_state.retrieval.citations, next_state.locale)
    if citation_summary:
        message_parts.append(citation_summary)

    next_state.chat_response = "\n\n".join(message_parts)
    next_state.next_route = "answer_composer"
    return next_state


def answer_composer(
    state: PetCareGraphState,
    *,
    llm_client: StructuredOutputClient | None = None,
) -> PetCareGraphState:
    """LangGraph-friendly alias for the answer composition node."""

    return compose_answer(state, llm_client=llm_client)


def _should_generate_general_petcare_answer(state: PetCareGraphState) -> bool:
    return (
        state.intent == "general_chat"
        and not state.requires_safety_screening
        and not state.red_flag_mentioned
    )


def _compose_general_petcare_answer(
    state: PetCareGraphState,
    *,
    llm_client: StructuredOutputClient | None,
) -> PetCareGraphState:
    fallback = GeneralPetCareAnswerOutput(
        assistant_message=_fallback_general_petcare_message(state),
    )

    try:
        output = call_structured_output(
            system_prompt=load_prompt("general_petcare_answer"),
            user_prompt=_general_petcare_prompt_payload(state),
            output_model=GeneralPetCareAnswerOutput,
            fallback=fallback,
            client=llm_client,
        )
    except Exception:
        output = fallback

    state.chat_response = output.assistant_message.strip() or fallback.assistant_message
    state.next_route = "answer_composer"
    return state


def _general_petcare_prompt_payload(state: PetCareGraphState) -> str:
    payload = {
        "user_input": state.user_input,
        "conversation_history": [
            message.model_dump(mode="json") for message in state.conversation_history[-12:]
        ],
        "locale": state.locale,
        "pet_context": state.context.pet,
        "retrieval": {
            "query": state.retrieval.query,
            "provider": state.retrieval.provider,
            "insufficient_evidence": state.retrieval.insufficient_evidence,
            "chunks": [_retrieved_chunk_payload(chunk) for chunk in state.retrieval.chunks[:5]],
            "citations": [
                citation.model_dump(mode="json") for citation in state.retrieval.citations[:5]
            ],
        },
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def _retrieved_chunk_payload(chunk: RetrievedChunk) -> dict[str, Any]:
    metadata = chunk.metadata or {}
    return {
        "chunk_id": chunk.chunk_id,
        "source_id": chunk.source_id,
        "title": chunk.title,
        "text": _truncate_text(chunk.text),
        "score": chunk.score,
        "url": str(metadata.get("canonical_url") or metadata.get("url") or ""),
        "species": metadata.get("species"),
        "section_path": metadata.get("section_path") or [],
    }


def _truncate_text(text: str, limit: int = 1200) -> str:
    normalized = " ".join(text.split())
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 3].rstrip() + "..."


def _fallback_general_petcare_message(state: PetCareGraphState) -> str:
    if wants_korean(state.locale):
        return GENERAL_PETCARE_FALLBACK_KO
    return GENERAL_PETCARE_FALLBACK


def _opening_guidance(state: PetCareGraphState) -> str:
    if state.intent == "general_chat" and state.risk_level == "unknown":
        if wants_korean(state.locale):
            return GENERAL_CHAT_GUIDANCE_KO
        return GENERAL_CHAT_GUIDANCE
    if wants_korean(state.locale):
        return RISK_GUIDANCE_KO[state.risk_level]
    return RISK_GUIDANCE[state.risk_level]


def _visit_intent_prompt(state: PetCareGraphState) -> str:
    if wants_korean(state.locale):
        return VISIT_INTENT_PROMPT_KO
    return VISIT_INTENT_PROMPT


def _evidence_summary(state: PetCareGraphState) -> str:
    chunks = state.retrieval.chunks
    if chunks:
        titles = _unique_titles(chunks)
        joined_titles = "; ".join(titles)
        if wants_korean(state.locale):
            return (
                "공식 자료 확인: Cornell 수의학 자료에서 "
                f"{joined_titles} 관련 내용을 찾았습니다. 이 자료는 응급 여부를 "
                "단정하는 데 쓰지 않고, 일반적이고 출처가 있는 안내에만 참고합니다."
            )
        return (
            "Official-source context reviewed: Cornell veterinary material was found "
            f"for {joined_titles}. I am keeping the guidance general and citation-backed, "
            "not using it to make an emergency decision."
        )
    if state.retrieval.insufficient_evidence and state.retrieval.query.strip():
        if wants_korean(state.locale):
            return (
                "현재 질문에 대해 충분한 Cornell 공식 근거를 찾지 못해서, "
                "출처 있는 주장처럼 꾸며서 답하지 않겠습니다."
            )
        return (
            "I could not find enough Cornell official-source evidence for the current "
            "question, so I will not invent source-backed claims."
        )
    return ""


def _citation_summary(citations: list[RAGCitation], locale: str | None) -> str:
    if not citations:
        return ""
    lines = ["Cornell 출처:" if wants_korean(locale) else "Cornell sources:"]
    for citation in citations:
        suffix = f" - {citation.url}" if citation.url else ""
        lines.append(f"[{citation.number}] {citation.title}{suffix}")
    return "\n".join(lines)


def _unique_titles(chunks: list[RetrievedChunk]) -> list[str]:
    titles: list[str] = []
    seen: set[str] = set()
    for chunk in chunks:
        title = chunk.title.strip()
        if not title or title in seen:
            continue
        titles.append(title)
        seen.add(title)
        if len(titles) == 3:
            break
    return titles or ["the retrieved source"]


def _known_symptoms(state: PetCareGraphState) -> list[str]:
    symptoms: list[str] = []
    seen: set[str] = set()
    for raw_symptom in [*state.assessment.symptoms, *state.current_status.symptoms]:
        symptom = " ".join(raw_symptom.strip().lower().split())
        if not symptom or symptom in seen:
            continue
        symptoms.append(symptom)
        seen.add(symptom)
    return display_list(symptoms, state.locale)


__all__ = ["ANSWER_SAFE_RISK_LEVELS", "answer_composer", "compose_answer"]