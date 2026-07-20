"""petcare_ai ↔ 메인 서버 계약 어댑터.

`petcare_ai`(RAG + LangGraph)와 메인 서버(`server/app/schemas.py`)는 **서로 다른
스키마**를 쓴다. 이 파일이 유일한 변환 지점이며, 여기만 맞으면 서버·앱 코드를
고치지 않고 AI 를 붙일 수 있다.

어긋나는 지점 4가지 (실측 확인):

1. **위험도 값이 다르다**
   - petcare_ai : ``normal | visit | emergency``
   - 서버       : ``normal | observe | consult | emergency``
   → ``visit`` 을 ``consult`` 로 매핑한다. 서버의 ``observe`` 는 LangGraph 가
     만들지 않으므로 사용하지 않는다.

2. **action 종류가 다르다**
   - petcare_ai : ``CALL_HOSPITAL | OPEN_PDF_PREVIEW | OPEN_GMAIL_COMPOSE`` (Android UI 의도)
   - 서버       : ``generate_summary | save_summary_pdf | send_email | save_record`` (Literal 강제)
   → 매핑 가능한 것만 변환하고 나머지는 버린다. 서버 Literal 에 없는 값을 그대로
     보내면 Pydantic 검증 실패로 **502** 가 난다(서버가 Agent 응답을 검증한다).
     ``CALL_HOSPITAL`` 은 대응 action 이 없어 ``show_hospitals=True`` 로 표현한다.

3. **근거 스키마가 다르다**
   - petcare_ai : ``FinalEvidence(title, source_url, text, ...)``
   - 서버       : ``RagCitation(title, source, snippet)``
   → 필드명을 변환하고 snippet 길이를 제한한다.

4. **병원 스키마의 중첩 깊이가 다르다**
   - petcare_ai : ``HospitalSuitabilityResult(hospital=HospitalCandidate(...), score, ...)`` — 2단
   - 서버       : ``HospitalSuggestion(...)`` — 평평한 1단
   → ``to_server_hospitals()`` 로 평평하게 만든다. 이 매핑이 없던 동안에는 AI 가
     Tavily 로 찾은 병원이 담길 자리가 없어 그대로 버려졌고, 앱은 시드 데모 병원만
     보여줬다.

메인 서버가 Agent 응답을 Pydantic 으로 검증하므로, 여기서 계약을 깨면 사용자에게는
"Agent 응답이 계약과 다릅니다" 502 로 보인다. 값을 추가할 때는 서버 스키마를 먼저 확인할 것.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# petcare_ai import 경로
# ---------------------------------------------------------------------------
# petcare_ai 패키지는 이 폴더(ai/) 안에 함께 둔다(ai/petcare_ai). 서비스가 ai/ 에서
# 실행되면 그대로 import 되지만, 다른 위치에서 기동될 때를 대비해 ai/ 를 sys.path 에
# 넣어 준다. PETCARE_AI_PATH 환경 변수로 상위 폴더를 명시 지정할 수도 있다.
def _ensure_petcare_ai_importable() -> None:
    try:
        import petcare_ai  # noqa: F401,PLC0415

        return
    except ImportError:
        pass

    candidates = [
        Path(os.environ["PETCARE_AI_PATH"]) if os.environ.get("PETCARE_AI_PATH") else None,
        Path(__file__).resolve().parents[1],
        Path.cwd(),
        Path.cwd() / "ai",
    ]
    for candidate in candidates:
        if candidate and (candidate / "petcare_ai" / "__init__.py").exists():
            if str(candidate) not in sys.path:
                sys.path.insert(0, str(candidate))
            logger.info("petcare_ai 를 경로에서 로드합니다: %s", candidate)
            return

    raise RuntimeError(
        "petcare_ai 패키지를 찾을 수 없습니다.\n"
        "  · petcare_ai 는 ai/ 폴더 안에 있어야 합니다(ai/petcare_ai).\n"
        "  · 또는 PETCARE_AI_PATH 환경 변수에 petcare_ai 상위 폴더 경로를 지정하세요.\n"
        f"  · 탐색한 경로: {[str(c) for c in candidates if c]}"
    )


_ensure_petcare_ai_importable()


# ---------------------------------------------------------------------------
# 0. 공용 헬퍼
# ---------------------------------------------------------------------------
def _as_dict(item: Any) -> dict:
    """pydantic 모델이든 dict 든 dict 로 만든다.

    같은 값이 두 형태로 온다: LangGraph State 스냅샷에서 읽으면 dict,
    `ChatGraphResult` 에서 읽으면 pydantic 모델이다. 호출부마다 분기하면
    반드시 한쪽을 빠뜨린다.
    """
    if isinstance(item, dict):
        return item
    dump = getattr(item, "model_dump", None)
    if callable(dump):
        try:
            return dump()
        except Exception:  # pragma: no cover - 변환 실패가 답변을 막으면 안 된다
            return {}
    return {}


def _clean_str_list(value: Any, limit: int) -> list[str]:
    """문자열 목록만 남긴다(앱 카드에 그대로 노출되므로 개수를 제한한다)."""
    return [str(v).strip() for v in (value or []) if isinstance(v, str) and v.strip()][:limit]


# ---------------------------------------------------------------------------
# 1. 위험도 매핑
# ---------------------------------------------------------------------------
#: petcare_ai(3단계) → 서버(4단계). 서버의 observe 는 LangGraph 가 만들지 않는다.
RISK_LEVEL_TO_SERVER: dict[str, str] = {
    "normal": "normal",
    "visit": "consult",
    "emergency": "emergency",
}


def to_server_risk_level(risk: str | None) -> str:
    """LangGraph 위험도를 서버 enum 으로 바꾼다.

    모르는 값이 오면 ``normal`` 로 낮추지 않고 ``consult`` 로 올린다 —
    분류 실패를 '정상'으로 표시하는 것이 사용자에게 더 위험하기 때문이다.
    """
    if not risk:
        return "normal"
    mapped = RISK_LEVEL_TO_SERVER.get(str(risk).lower())
    if mapped:
        return mapped
    logger.warning("알 수 없는 risk_level=%r → consult 로 보수 처리합니다.", risk)
    return "consult"


# ---------------------------------------------------------------------------
# 2. action 매핑
# ---------------------------------------------------------------------------
#: petcare_ai ui_action → 서버 AgentAction.type. 없는 것은 버린다.
UI_ACTION_TO_SERVER: dict[str, str] = {
    "OPEN_PDF_PREVIEW": "save_summary_pdf",
    "OPEN_GMAIL_COMPOSE": "send_email",
    "GENERATE_SUMMARY": "generate_summary",
    "SAVE_RECORD": "save_record",
    # CALL_HOSPITAL 은 서버에 대응 action 이 없다 → show_hospitals 로 표현한다.
}

ACTION_LABELS: dict[str, str] = {
    "generate_summary": "병원 전달용 요약 만들기",
    "save_summary_pdf": "요약 PDF 저장",
    "send_email": "상태 문서 이메일",
    "save_record": "기록 저장",
}


def to_server_actions(ui_actions: list[dict] | None) -> list[dict]:
    """ui_actions 를 서버가 받는 AgentAction 목록으로 바꾼다.

    서버 스키마의 Literal 에 없는 type 은 **반드시 제거**한다. 그대로 보내면
    서버가 502 로 응답을 통째로 버려 답변까지 사라진다.
    """
    out: list[dict] = []
    seen: set[str] = set()
    for action in ui_actions or []:
        if not isinstance(action, dict):
            continue
        mapped = UI_ACTION_TO_SERVER.get(str(action.get("type", "")).upper())
        if not mapped or mapped in seen:
            continue
        seen.add(mapped)
        payload = {
            key: value
            for key, value in action.items()
            if key not in ("type", "label") and isinstance(value, (str, int, float, bool))
        }
        out.append({
            "type": mapped,
            "label": action.get("label") or ACTION_LABELS.get(mapped, ""),
            "payload": payload,
        })
    return out


def wants_hospitals(ui_actions: list[dict] | None) -> bool:
    """CALL_HOSPITAL action 이 있으면 앱이 병원 목록을 띄우도록 한다."""
    return any(
        isinstance(a, dict) and str(a.get("type", "")).upper() == "CALL_HOSPITAL"
        for a in ui_actions or []
    )


# ---------------------------------------------------------------------------
# 3. 근거 매핑
# ---------------------------------------------------------------------------
SNIPPET_MAX_CHARS = 300


def to_server_citations(evidence: list[Any] | None) -> list[dict]:
    """FinalEvidence → 서버 RagCitation(title/source/snippet).

    snippet 은 앱 화면에 그대로 노출되므로 길이를 제한한다.
    """
    citations: list[dict] = []
    for item in evidence or []:
        data = item if isinstance(item, dict) else getattr(item, "model_dump", lambda: {})()
        if not data:
            continue
        text = (data.get("text") or "").strip().replace("\n", " ")
        if len(text) > SNIPPET_MAX_CHARS:
            text = text[:SNIPPET_MAX_CHARS].rstrip() + "…"
        citations.append({
            "title": (data.get("title") or "").strip(),
            # 서버 RagCitation.source 는 출처 식별자다. URL 을 그대로 넣어야
            # 앱에서 어디서 온 근거인지 보여줄 수 있다.
            "source": (data.get("source_url") or data.get("source") or "").strip(),
            "snippet": text,
        })
    return citations


def evidence_line(evidence: list[Any] | None, fallback: str = "") -> str:
    """서버 AICheckResponse.evidence(한 줄 근거 문자열)를 만든다.

    앱의 결과 카드가 '근거 · {evidence}' 로 렌더하므로 출처 제목을 요약해 넣는다.
    """
    titles: list[str] = []
    for item in evidence or []:
        data = item if isinstance(item, dict) else getattr(item, "model_dump", lambda: {})()
        title = (data.get("title") or "").strip()
        if title and title not in titles:
            titles.append(title)
        if len(titles) >= 3:
            break
    if not titles:
        return fallback
    return " · ".join(titles) + " (Cornell 수의학 자료)"


# ---------------------------------------------------------------------------
# 4. 병원 매핑
# ---------------------------------------------------------------------------
#: 앱 카드에 보여줄 병원 수 상한. 응급 상황에 선택지를 늘리면 결정만 늦어진다.
MAX_HOSPITALS = 5
#: 이유/확인항목 bullet 상한.
MAX_HOSPITAL_REASONS = 4


def to_server_hospitals(hospitals: list[Any] | None) -> list[dict]:
    """petcare_ai `HospitalSuitabilityResult` → 서버 `HospitalSuggestion` dict.

    두 스키마는 **중첩 구조부터 다르다**:

    - petcare_ai : ``HospitalSuitabilityResult(hospital=HospitalCandidate(...),
      score, suitability, matched_reasons, verification_required)``
      — name/phone/address/email/source_url/emergency_mentioned/open_24h_mentioned/
      availability 는 안쪽 ``hospital`` 에, 평가 결과는 바깥에 있다.
    - 서버       : 평평한 `HospitalSuggestion` 1단 구조.

    → 안쪽과 바깥쪽을 합쳐 평평하게 만든다. 이 변환이 없으면 LangGraph 가 Tavily 로
    찾아 점수까지 매긴 병원이 서버 스키마에 담길 자리가 없어 그대로 버려진다
    (앱이 AI 결과 대신 시드 데모 병원을 보여주던 원인).

    ``name`` 이 없는 항목은 **버린다** — 이름 없는 병원 카드는 사용자가 전화 걸
    대상을 특정할 수 없어 화면만 어지럽힌다.
    """
    out: list[dict] = []
    seen: set[str] = set()
    for item in hospitals or []:
        data = _as_dict(item)
        if not data:
            continue
        # 평가 결과 없이 HospitalCandidate 만 올 수도 있어 자기 자신을 fallback 으로 둔다.
        candidate = _as_dict(data.get("hospital")) or data

        name = str(candidate.get("name") or "").strip()
        if not name:
            continue
        # 같은 병원이 서로 다른 검색어에서 중복으로 잡히는 일이 잦다.
        phone = str(candidate.get("phone") or "").strip()
        key = f"{name}|{phone}"
        if key in seen:
            continue
        seen.add(key)

        try:
            score = int(data.get("score") or 0)
        except (TypeError, ValueError):
            score = 0

        out.append({
            "name": name,
            # 빈 문자열 대신 None — 앱이 "전화 걸기" 버튼을 숨길지 판단해야 한다.
            "phone": phone or None,
            "address": str(candidate.get("address") or "").strip() or None,
            # 병원 이메일은 응급 이메일 초안의 수신 주소가 된다. 여기서 빠뜨리면
            # `extract_email()` 이 어렵게 찾아낸 주소가 서버 스키마에 담길 자리가
            # 없어 그대로 버려지고, 보호자가 매번 주소를 직접 입력해야 한다.
            "email": str(candidate.get("email") or "").strip() or None,
            "source_url": str(candidate.get("source_url") or "").strip(),
            "score": score,
            "suitability": str(data.get("suitability") or "low_information").strip(),
            "matched_reasons": _clean_str_list(
                data.get("matched_reasons"), MAX_HOSPITAL_REASONS
            ),
            "verification_required": _clean_str_list(
                data.get("verification_required"), MAX_HOSPITAL_REASONS
            ),
            "emergency_mentioned": bool(candidate.get("emergency_mentioned")),
            "open_24h_mentioned": bool(candidate.get("open_24h_mentioned")),
            # 검색 결과만으로 영업 여부를 확정하지 않는다(petcare_ai 명세 34절).
            "availability": str(candidate.get("availability") or "전화 확인 필요").strip(),
        })
        if len(out) >= MAX_HOSPITALS:
            break
    return out


# ---------------------------------------------------------------------------
# 최종 변환 — ChatGraphResult → 서버 AICheckResponse dict
# ---------------------------------------------------------------------------
def _app_redundant_prefixes() -> tuple[str, ...]:
    """앱이 **카드로 따로 그리는** 블록의 시작 문구.

    그래프의 message 는 Colab 처럼 화면이 없는 곳을 기준으로 조립돼서 병원 목록·
    확인 안내·첨부 안내가 본문에 다 들어 있다. 앱에서는 같은 정보가 병원 카드·
    이메일 버튼으로 또 나오므로 그대로 두면 한 화면에 두 번 보인다.

    **문구를 여기에 복사해 두지 않는다.** petcare_ai 가 실제로 쓰는 상수를 그대로
    가져와 쓴다 — 복사해 두면 그래프 쪽 문구가 바뀌는 순간 조용히 매칭이 실패해
    중복이 되살아난다. 상수로 존재하지 않고 노드 안에서 조립되는 몇 개만 문자열로
    적는다(그 자리는 emergency.py 의 조립부를 따라간다).

    지우는 것은 **구조화 필드로 전달되는 것**뿐이다. 응급 신호 요약·면책 문구처럼
    앱이 그리지 않는 내용은 남긴다 — 중복 제거가 목적이지 정보 축소가 아니다.
    """
    from petcare_ai.graph.prompts import HOSPITAL_VERIFICATION_NOTICE  # noqa: PLC0415

    return (
        HOSPITAL_VERIFICATION_NOTICE[:12],   # → 병원 카드마다 붙는 확인 문구
        "주변에서 확인된 동물병원",            # → hospitals 필드 (HospitalSuggestionList)
        "지금 검색으로는 조건에 맞는",          # → 앱의 '병원을 찾지 못했어요' 빈 상태
        "병원에 전화하실 때 아래 항목",         # → 앱이 쓰지 않는 안내
        "병원에 보여 드릴 상담 자료를 PDF",     # → PDF 는 앱에서 쓰지 않는다
        "이메일로 미리 보내실 수 있도록",       # → 이메일 버튼이 대신한다
    )


def to_app_reply(message: Any) -> str:
    """그래프 message 에서 앱이 따로 그리는 블록을 걷어낸 대화용 본문을 만든다.

    빈 줄로 나뉜 블록 단위로 판단한다. 한 블록이 여러 줄(병원 목록 등)이라
    줄 단위로 지우면 목록 항목만 남는 조각이 생긴다.
    """
    text = str(message or "").strip()
    if not text:
        return ""

    prefixes = _app_redundant_prefixes()
    separator = "\n\n"
    kept: list[str] = []
    for block in text.split(separator):
        head = block.strip()
        if not head:
            continue
        if head.startswith(prefixes):
            continue
        kept.append(head)
    return separator.join(kept).strip()


def _trace(data: dict) -> dict:
    """결과의 `trace_metadata` — 필드 이름이 `metadata` 가 아니라 `trace_metadata` 다."""
    trace = data.get("trace_metadata")
    return trace if isinstance(trace, dict) else {}


def chat_result_to_agent_response(result: Any, state: dict | None = None) -> dict:
    """LangGraph 결과를 서버가 그대로 받을 수 있는 dict 로 만든다.

    서버 `AICheckResponse` 필드만 담는다. 여분 필드는 서버가 무시하지만,
    Literal·타입이 어긋나면 502 가 되므로 값은 반드시 위 매핑을 거친다.
    """
    data = result if isinstance(result, dict) else getattr(result, "model_dump", lambda: {})()
    state = state or {}

    ui_actions = data.get("ui_actions") or []
    evidence = data.get("evidence") or []
    risk = to_server_risk_level(data.get("risk_level"))

    # 병원은 결과에 있으면 우선 쓰고, 없으면 State 의 hospital_results 를 본다.
    # (Suitability node 는 State 에 쓰고, 결과 객체에는 응급 경로에서만 실린다.)
    hospitals = to_server_hospitals(data.get("hospitals") or state.get("hospital_results"))

    # **답변과 되묻기는 배타적이다 — 둘을 같이 내보내지 않는다.**
    #
    # 그래프가 interrupt 로 멈춰 있으면 그 turn 은 '질문' 이고, 멈추지 않았으면
    # '답변' 이다. 그 사이는 없다.
    #
    # 예전에는 `missing_fields` 가 남아 있기만 하면 답변에 질문을 덧붙였다. 그래서
    # "보양식을 만들어 주고 싶은데 어떻게 하면 될까요?" 에 Cornell 근거로 답을 다 한
    # 뒤에 "언제부터 그런 모습이 나타났나요?" 를 또 물었다. 묻지도 않은 증상의 시작
    # 시점을 되묻는 셈이라 대화가 어긋난다.
    #
    # 미확인 항목이 사라지는 것은 아니다 — `missing_fields` 는 State 에 그대로 남아
    # 병원 전달 문서의 '아직 확인되지 않은 정보' 로 인쇄된다. 다만 그것을 **질문으로
    # 되돌려 보내지 않을** 뿐이다.
    interrupted = bool(_trace(data).get("interrupted"))
    if interrupted:
        # 되묻는 중 — message 에 질문이 들어 있다. `missing_information_question` 이
        # 비어 있을 수 있어(interrupt 로 멈춘 node 의 반환값은 State 에 반영되지 않는다)
        # message 를 폴백으로 쓴다.
        followup = (
            state.get("missing_information_question")
            or str(data.get("message") or "").strip()
            or None
        )
    else:
        followup = None

    return {
        "reply": to_app_reply(data.get("message")),
        "risk_level": risk,
        # risk_label 은 서버가 RISK_LABELS 로 덮어쓰므로 비워 보낸다.
        "risk_label": "",
        "trend_summary": state.get("trend_summary") or "",
        "trends": [],
        "reasons": [r for r in (state.get("risk_reasons") or []) if isinstance(r, str)][:5],
        "evidence": evidence_line(evidence),
        "followup_question": followup,
        # 아직 되묻는 중인가. 앱은 이 값으로 "질문 말풍선" 과 "판정 카드" 를 가른다.
        # risk_level 만 보면 안 된다 — 응급 판정이 **끝난** 응답도 병원에 전달할
        # 항목을 followup 으로 함께 물어보기 때문에, followup 유무로 가르면 진짜
        # 응급 카드까지 질문으로 처리된다.
        "awaiting_more_info": interrupted,
        # 이번 turn 이 **새 판정**인가, 아니면 앞선 판정에 대한 설명인가.
        #
        # "왜 응급한거죠?" 같은 되물음은 새 증상 신고가 아니라 설명 요청이다.
        # 그런데 서버가 대화의 최고 위험도를 유지하므로 이 turn 도 emergency 로
        # 표시되고, 앱은 위험도만 보고 응급 카드를 **또** 그렸다(실제로 그랬다).
        # 카드는 이미 위에 있는데 같은 것이 반복되면 대화가 되지 않는다.
        #
        # intent 로 가른다 — 지식 질문·잡담은 상태를 판정하는 turn 이 아니다.
        "assessment_turn": str(state.get("intent") or "health_question")
        not in ("general_knowledge", "general_chat"),
        # 요약 버튼은 **병원 권고**에서만. 응급 화면은 병원 목록·이메일만 쓴다.
        "can_generate_summary": risk == "consult" and not interrupted,
        # 병원을 실제로 찾았으면 위험도와 무관하게 카드를 띄운다 — 사용자가 병원을
        # 물어봤는데(hospital_search intent) 결과를 숨기면 검색이 무의미해진다.
        "show_hospitals": bool(hospitals) or risk == "emergency" or wants_hospitals(ui_actions),
        "transit_guidance": [
            g for g in (state.get("transit_guidance") or []) if isinstance(g, str)
        ],
        "actions": to_server_actions(ui_actions),
        "citations": to_server_citations(evidence),
        "hospitals": hospitals,
        "source": "agent",
    }


__all__ = [
    "RISK_LEVEL_TO_SERVER",
    "to_app_reply",
    "UI_ACTION_TO_SERVER",
    "to_server_risk_level",
    "to_server_actions",
    "to_server_citations",
    "to_server_hospitals",
    "wants_hospitals",
    "evidence_line",
    "chat_result_to_agent_response",
]
