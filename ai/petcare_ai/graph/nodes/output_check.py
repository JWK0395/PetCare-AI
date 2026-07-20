"""Output Check Agent — 답변을 사용자에게 보내기 전 마지막 검문 (명세 40절).

명세 40절이 요구한 **9가지 검사를 모두** 구현한다.

  1. DB 에 없는 사실 생성 여부        → `_check_fabricated_facts`
  2. source provenance 누락 여부      → `_check_provenance`
  3. RAG 근거와 답변 불일치           → `_check_evidence_alignment`
  4. 확정 진단 표현                   → `DEFINITIVE_DIAGNOSIS_PATTERNS`
  5. 약 처방 또는 변경 지시           → `PRESCRIPTION_PATTERNS`
  6. visit/emergency 안내 누락        → `_check_risk_guidance`
  7. 병원 실시간 진료 가능 여부 단정  → `REALTIME_AVAILABILITY_PATTERNS`
  8. PDF 와 email attachment path 불일치 → `_check_attachment_paths`
  9. 결과 schema 오류                 → `_check_schema`

## 왜 금지 표현을 정규식 상수로 모았는가

명세 43절 "Output Safety" 테스트가 확정 진단 문장·복용 중단 지시·"현재 진료 가능"
단정을 **각각** 차단하는지 검증한다. 검사 로직이 if 문으로 흩어져 있으면 테스트가
내부 구현을 알아야 하고, 표현을 추가할 때마다 코드를 고쳐야 한다. 여기서는
`(이름, 정규식)` 튜플 목록만 늘리면 검사와 테스트가 함께 확장된다.

## 재생성은 최대 1회 (명세 40절)

`retry_count` 로 셈한다. 두 번째도 실패하면 `regenerate` 가 아니라 `fallback` 이며,
Final Safety Agent 가 안전 문구로 대체한다. 무한 재생성 루프는 그 자체가 장애다.
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Any, Literal

from ...schemas import RISK_PRIORITY, OutputCheckResult
from ..prompts import HOSPITAL_VERIFICATION_NOTICE
from ..state import URGENCY_PRIORITY, Replace

if TYPE_CHECKING:
    from ..state import PetCareState

logger = logging.getLogger(__name__)

__all__ = [
    "DEFINITIVE_DIAGNOSIS_PATTERNS",
    "PRESCRIPTION_PATTERNS",
    "REALTIME_AVAILABILITY_PATTERNS",
    "SAFE_MEDICATION_CONTEXT",
    "GENERIC_HOSPITAL_WORDS",
    "FATAL_PREFIX",
    "MAX_REGENERATION",
    "find_forbidden_expressions",
    "has_forbidden_expression",
    "check_output",
    "decide_action",
    "output_check_node",
    "route_output_check",
]


# ---------------------------------------------------------------------------
# 금지 표현 정규식 (명세 40·43·47절)
# ---------------------------------------------------------------------------
# 4. 확정 진단 표현 — "가능성", "의심", "~일 수 있습니다" 는 허용하고
#    "~입니다", "~로 진단" 같은 **단정**만 잡는다.
DEFINITIVE_DIAGNOSIS_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "확정 진단 표현('~로 진단')",
        re.compile(r"(?:으로|로)\s*진단(?:됩니다|입니다|합니다|드립니다|되었|된\s*것)"),
    ),
    (
        "확정 진단 표현('진단 결과는 ~입니다')",
        re.compile(r"진단\s*(?:결과|명)\s*(?:은|는|이)?\s*[^\s.,]+\s*(?:입니다|이에요|예요)"),
    ),
    (
        "확정 진단 표현(질병명 단정)",
        re.compile(
            r"[가-힣A-Za-z]{1,10}\s*(?:증|염|병|질환|암|종양|부전|증후군)\s*"
            r"(?:입니다|이에요|예요|이며|이 확실|임이\s*확실)"
        ),
    ),
    (
        "확정 진단 표현(단정 부사)",
        re.compile(r"(?:확실히|분명히|틀림없이|100%)\s*\S{0,12}\s*(?:입니다|이에요|맞습니다)"),
    ),
)

# 5. 약 처방 또는 변경 지시 — 이 앱은 어떤 경우에도 투약을 지시하지 않는다.
PRESCRIPTION_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("복용 중단·변경 지시", re.compile(r"복용(?:을|를)?\s*(?:중단|중지|끊|줄이|늘리)")),
    ("약 중단·변경 지시", re.compile(r"약(?:을|를)?\s*(?:끊|중단|중지|줄이|늘리|바꾸|변경)")),
    ("투약 변경 지시", re.compile(r"투약(?:을|를)?\s*(?:중단|중지|변경|조절)")),
    ("용량 조절 지시", re.compile(r"용(?:량|법)(?:을|를)?\s*(?:줄이|늘리|조절|변경|바꾸)")),
    (
        "용량 지정 투약 지시",
        re.compile(r"\d+\s*(?:mg|밀리그램|ml|cc|정|알|포)\s*(?:씩)?\s*(?:복용|투여|먹이|급여)"),
    ),
    ("처방 행위", re.compile(r"처방\s*(?:해\s*드리|합니다|드립니다|해\s*줄|해\s*드립)")),
    (
        "투약 지시",
        re.compile(
            r"(?:약|항생제|진통제|해열제|소염제|스테로이드|구충제|영양제|지사제|제산제)"
            r"[가-힣]{0,4}\s*(?:을|를)?\s*(?:먹이|투여|복용|급여)[가-힣]{0,3}\s*"
            r"(?:세요|십시오|시면|하세요|해도)"
        ),
    ),
)

# 7. 병원 실시간 진료 가능 여부 단정 — 검색 결과로는 알 수 없다(명세 34·47절).
REALTIME_AVAILABILITY_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "실시간 진료 가능 단정",
        re.compile(r"(?:지금|현재)\s*(?:바로\s*)?(?:진료|접수|수술|영업)\s*(?:가|이)?\s*가능"),
    ),
    ("영업 중 단정", re.compile(r"(?:진료|영업)\s*중\s*(?:입니다|이에요|이며|이니)")),
    (
        "24시간 운영 단정",
        re.compile(r"24\s*시간?\s*(?:운영|영업|진료)\s*(?:합니다|해요|중|하고\s*있|하는)"),
    ),
    ("응급 접수 가능 단정", re.compile(r"응급\s*(?:접수|진료)\s*(?:가|이)?\s*가능(?:합니다|해요)")),
    ("즉시 방문 가능 단정", re.compile(r"바로\s*가시면\s*(?:됩니다|돼요|되세요)")),
    ("대기 없음 단정", re.compile(r"대기\s*(?:없이|없습니다|없어요)")),
)

# 투약 관련 표현이라도 "임의로 중단하지 마세요" 같은 **금지 안내**는 안전하다.
# 문장 단위로 이 신호가 있으면 위반으로 세지 않는다.
SAFE_MEDICATION_CONTEXT: tuple[str, ...] = (
    "하지 마",
    "하지 말",
    "마세요",
    "말아",
    "말고",
    "금물",
    "임의로",
    "상의 없이",
    "상담 없이",
    "수의사와 상의",
    "수의사와 상담",
    "수의사의 지시",
    "처방 없이",
)

# 고유명사가 아닌 일반명사 — "동물병원에 방문하세요" 를 지어낸 병원명으로 보면 안 된다.
GENERIC_HOSPITAL_WORDS: frozenset[str] = frozenset(
    {
        "동물병원",
        "반려동물병원",
        "24시동물병원",
        "24시간동물병원",
        "응급동물병원",
        "야간동물병원",
        "지역동물병원",
        "근처동물병원",
        "가까운동물병원",
        "종합병원",
        "대학병원",
    }
)

FATAL_PREFIX = "[치명]"
# 명세 40절: 재생성은 최대 1회.
MAX_REGENERATION = 1

_SENTENCE_SPLIT_RE = re.compile(r"[.!?\n]+")
_URL_RE = re.compile(r"https?://[^\s)>\]\"']+")
_WEIGHT_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(?:kg|킬로|㎏)")
_HOSPITAL_NAME_RE = re.compile(
    r"[가-힣A-Za-z0-9]{2,12}(?:동물병원|동물의료센터|동물메디컬센터|메디컬센터)"
)
_TOKEN_RE = re.compile(r"[가-힣]{2,}|[A-Za-z]{3,}")


# ---------------------------------------------------------------------------
# 금지 표현 탐지
# ---------------------------------------------------------------------------
def _sentences(text: str) -> list[str]:
    return [part.strip() for part in _SENTENCE_SPLIT_RE.split(text or "") if part.strip()]


def _is_safe_medication_sentence(sentence: str) -> bool:
    """투약 관련 문장이 '금지 안내' 인지 판정한다.

    "임의로 약을 중단하지 마세요" 는 올바른 안전 안내인데, 정규식만 보면
    '약을 중단' 이 걸린다. 이 예외가 없으면 안전한 문장이 재생성 루프를 유발한다.
    """
    return any(marker in sentence for marker in SAFE_MEDICATION_CONTEXT)


def find_forbidden_expressions(text: str) -> list[str]:
    """금지 표현을 모두 찾아 사람이 읽을 수 있는 사유 문자열로 돌려준다.

    문장 단위로 검사하는 이유: 예외(금지 안내 문맥)를 문장 범위로 판정해야
    "약을 중단하지 마세요" 와 "약을 중단하세요" 를 구분할 수 있기 때문이다.
    """
    if not text:
        return []

    found: list[str] = []
    for sentence in _sentences(text):
        for name, pattern in DEFINITIVE_DIAGNOSIS_PATTERNS:
            match = pattern.search(sentence)
            if match:
                found.append(f"{name}: '{match.group(0).strip()}'")

        if not _is_safe_medication_sentence(sentence):
            for name, pattern in PRESCRIPTION_PATTERNS:
                match = pattern.search(sentence)
                if match:
                    found.append(f"{name}: '{match.group(0).strip()}'")

        for name, pattern in REALTIME_AVAILABILITY_PATTERNS:
            match = pattern.search(sentence)
            if match:
                found.append(f"{name}: '{match.group(0).strip()}'")

    # 같은 표현이 여러 문장에 반복돼도 사유는 한 번만 보고한다.
    seen: set[str] = set()
    unique: list[str] = []
    for item in found:
        if item not in seen:
            seen.add(item)
            unique.append(item)
    return unique


def has_forbidden_expression(text: str) -> bool:
    """금지 표현이 하나라도 있는지(Final Safety 가 빠르게 쓰는 판정)."""
    return bool(find_forbidden_expressions(text))


# ---------------------------------------------------------------------------
# 개별 검사
# ---------------------------------------------------------------------------
def _known_fact_text(state: dict[str, Any]) -> str:
    """답변이 근거로 삼을 수 있는 모든 원천 텍스트를 합친다.

    사용자 입력 · PET DB · 진단서 · 일기장 · 검증된 근거 · 병원 검색 결과가 전부
    들어간다. 여기에 없는 구체적 사실(체중 수치, 병원 이름)이 답변에 나타나면
    모델이 지어낸 것이다(명세 40절 1번 항목).
    """
    parts: list[str] = [str(state.get("user_message") or "")]

    collected = state.get("collected_information") or {}
    if isinstance(collected, dict):
        parts.extend(str(value) for value in collected.values())

    for key in ("pet_profile", "priority_pet_context", "current_observation"):
        section = state.get(key) or {}
        if isinstance(section, dict):
            parts.extend(f"{name}: {value}" for name, value in section.items())

    for key in (
        "diagnoses",
        "related_diagnoses",
        "daily_entries",
        "supporting_daily_entries",
        "merged_evidence",
        "hospital_results",
        "raw_hospital_results",
    ):
        for record in state.get(key) or []:
            if isinstance(record, dict):
                parts.extend(str(value) for value in record.values())
            else:
                parts.append(str(record))

    return " ".join(part for part in parts if part)


def _evidence_urls(state: dict[str, Any]) -> set[str]:
    urls: set[str] = set()
    for evidence in state.get("merged_evidence") or []:
        if isinstance(evidence, dict):
            url = str(evidence.get("source_url") or evidence.get("url") or "").strip()
            if url:
                urls.add(url.rstrip("/"))
    for hospital in state.get("hospital_results") or []:
        if isinstance(hospital, dict):
            candidate = hospital.get("hospital") or hospital
            if isinstance(candidate, dict):
                url = str(candidate.get("source_url") or candidate.get("website") or "").strip()
                if url:
                    urls.add(url.rstrip("/"))
    return urls


def _check_fabricated_facts(state: dict[str, Any], draft: str) -> list[str]:
    """1. DB 에 없는 사실 생성 여부."""
    errors: list[str] = []
    known = _known_fact_text(state)
    known_compact = re.sub(r"\s+", "", known)

    # 체중 수치는 PDF 와 병원 안내에 그대로 실리므로 특히 위험하다.
    for value in set(_WEIGHT_RE.findall(draft)):
        if value not in known:
            errors.append(
                f"DB·입력 어디에도 없는 체중 수치를 생성했습니다: {value}kg"
            )

    for name in set(_HOSPITAL_NAME_RE.findall(draft)):
        if re.sub(r"\s+", "", name) in GENERIC_HOSPITAL_WORDS:
            continue
        if name not in known_compact and name not in known:
            errors.append(f"검색·진단서에 없는 병원 이름을 생성했습니다: {name}")

    return errors


def _check_provenance(state: dict[str, Any], draft: str) -> list[str]:
    """2. source provenance 누락 여부.

    **본문 표기를 요구하는 것은 `visit`·`emergency` 뿐이다.**

    일상·지식 질문(`normal`)의 답변은 대화체로 쓰게 했다. 문장마다 "[근거 1]" 이
    붙으면 서식을 읽는 느낌이 되어 편하게 말을 걸 수 없기 때문이다. 출처가 사라지는
    것이 아니라 앱이 `citations` 목록으로 따로 보여준다.

    여기서 본문 표기를 강제하면 의도대로 쓴 답변이 전부 반려되어 안전 폴백
    ("안전하게 답변드리기 어렵습니다")으로 떨어진다 — 실제로 그렇게 됐다.
    형식만 바꾸고 검사를 그대로 두면 형식 변경이 통째로 무효가 된다.

    반면 근거 목록에 없는 URL 을 지어낸 경우는 **위험도와 무관하게** 잡는다.
    그건 형식 문제가 아니라 없는 출처를 만든 것이다.
    """
    errors: list[str] = []
    evidence = [e for e in (state.get("merged_evidence") or []) if isinstance(e, dict)]
    cited_urls = {url.rstrip("/") for url in _URL_RE.findall(draft)}
    known_urls = _evidence_urls(state)
    risk = str(state.get("final_risk") or "normal")

    if evidence and risk in ("visit", "emergency"):
        titles = [str(e.get("title") or "").strip() for e in evidence]
        mentions_title = any(title and title in draft for title in titles)
        if not cited_urls and not mentions_title:
            errors.append(
                "근거 문서를 사용했는데 답변에 출처(제목 또는 URL)가 표시되지 않았습니다."
            )

    for url in cited_urls - known_urls:
        errors.append(f"근거 목록에 없는 출처 URL 을 답변에 넣었습니다: {url}")

    return errors


def _tokens(text: str) -> set[str]:
    return set(_TOKEN_RE.findall(text or ""))


def _check_evidence_alignment(state: dict[str, Any], draft: str) -> list[str]:
    """3. RAG 근거와 답변 불일치 — **언어가 같을 때만** 판정한다.

    어휘 교집합으로 "근거를 무시했는가" 를 보는 검사다. 그런데 이 저장소의 지식
    베이스는 **영문**(Cornell 수의학 문서)이고 답변은 **한국어**다. 번역해서 답하는
    것이 정상 동작인데, 토큰 교집합은 구조적으로 공집합이 된다:

        답변 토큰 : {'포도는', '강아지에게', '먹이면', ...}
        근거 토큰 : {'Grape', 'raisin', 'toxicity', ...}
        교집합    : 없음

    그래서 이 검사는 **정상 답변을 전부 반려**했다. 재생성 2회 후 안전 폴백으로
    떨어져 "안전하게 답변드리기 어렵습니다" 만 나갔다 — 근거를 8건이나 찾아 놓고도
    그랬다. 답변이 길고 근거 원문을 그대로 인용하던 시절에는 고유명사·숫자가 우연히
    겹쳐 가려져 있었을 뿐이다.

    그래서 **답변과 근거가 같은 문자 체계를 공유할 때만** 판정한다. 한국어 답변에
    한국어 근거가 하나도 없으면 대조할 방법이 없으므로 이 검사는 침묵한다.
    근거 없는 서술을 막는 일은 `_check_provenance`·`find_forbidden_expressions` 와
    "근거 안에서만 답한다" 는 프롬프트 규칙이 계속 담당한다.
    """
    evidence = [e for e in (state.get("merged_evidence") or []) if isinstance(e, dict)]
    if not evidence:
        return []

    evidence_tokens: set[str] = set()
    for item in evidence:
        evidence_tokens |= _tokens(str(item.get("title") or ""))
        evidence_tokens |= _tokens(str(item.get("text") or "")[:400])
        for topic in item.get("supported_topics") or []:
            evidence_tokens |= _tokens(str(topic))

    if not evidence_tokens:
        return []

    # 근거에 한글이 하나도 없는데 답변이 한국어라면 번역된 것이다 — 대조 불가.
    korean = re.compile(r"[가-힣]")
    if korean.search(draft) and not any(korean.search(token) for token in evidence_tokens):
        return []

    if not (_tokens(draft) & evidence_tokens):
        return ["검색된 근거와 답변 내용이 전혀 겹치지 않습니다(근거 미반영 의심)."]
    return []


def _check_risk_guidance(state: dict[str, Any], draft: str) -> list[str]:
    """6. visit / emergency 안내 누락 여부."""
    risk = str(state.get("final_risk") or "normal")
    errors: list[str] = []

    if risk == "visit":
        if "병원" not in draft:
            errors.append("병원 방문 권고(visit)인데 답변에 병원 안내가 없습니다.")
        elif not any(word in draft for word in ("진료", "방문", "내원", "상담", "검진")):
            errors.append("병원 방문 권고(visit)인데 진료·내원 안내 문구가 없습니다.")

    if risk == "emergency":
        if "병원" not in draft:
            errors.append("응급(emergency)인데 답변에 병원 안내가 없습니다.")
        if not any(word in draft for word in ("즉시", "지금 바로", "응급", "서둘러")):
            errors.append("응급(emergency)인데 즉시 대응을 알리는 안내가 없습니다.")

    # 병원을 안내했다면 "가기 전에 전화로 확인하라" 는 문구가 반드시 있어야 한다.
    # 검색 결과만으로는 실시간 진료 가능 여부를 알 수 없기 때문이다(명세 34·40절).
    if state.get("hospital_results"):
        has_notice = "전화" in draft or ("연락" in draft and "확인" in draft)
        if not has_notice:
            errors.append(
                "병원을 안내했는데 방문 전 전화 확인 안내가 없습니다"
                f"(권장 문구: '{HOSPITAL_VERIFICATION_NOTICE}')."
            )

    urgency = str(state.get("emergency_urgency") or "none")
    if urgency == "critical_immediate":
        actions = {
            str(action.get("type"))
            for action in (state.get("ui_actions") or [])
            if isinstance(action, dict)
        }
        if "CALL_HOSPITAL" not in actions and "REQUEST_LOCATION" not in actions:
            errors.append(
                "즉시 위급(critical_immediate)인데 병원 연락 action 이 준비되지 않았습니다."
            )

    return errors


def _check_attachment_paths(state: dict[str, Any]) -> list[str]:
    """8. PDF 와 email attachment path 불일치 (구조적 오류 → fatal)."""
    errors: list[str] = []
    email_draft = state.get("email_draft")
    pdf_path = state.get("pdf_path")

    if not isinstance(email_draft, dict):
        return errors

    attachment = str(email_draft.get("attachment_path") or "")
    if not pdf_path:
        errors.append(
            f"{FATAL_PREFIX} 이메일 초안에 첨부 경로가 있는데 생성된 PDF 경로가 없습니다."
        )
        return errors

    if attachment.replace("\\", "/") != str(pdf_path).replace("\\", "/"):
        errors.append(
            f"{FATAL_PREFIX} PDF 경로와 이메일 첨부 경로가 다릅니다: "
            f"pdf={pdf_path} / attachment={attachment}"
        )

    filename = str(email_draft.get("attachment_filename") or "")
    expected = str(state.get("pdf_filename") or "")
    if filename and expected and filename != expected:
        errors.append(
            f"{FATAL_PREFIX} PDF 파일명과 첨부 파일명이 다릅니다: {expected} / {filename}"
        )
    return errors


def _check_schema(state: dict[str, Any], draft: str) -> list[str]:
    """9. 결과 schema 오류."""
    errors: list[str] = []

    if not draft.strip():
        errors.append("답변 본문이 비어 있습니다.")

    risk = str(state.get("final_risk") or "normal")
    if risk not in RISK_PRIORITY:
        errors.append(f"{FATAL_PREFIX} 알 수 없는 risk_level 입니다: {risk!r}")

    urgency = str(state.get("emergency_urgency") or "none")
    if urgency not in URGENCY_PRIORITY:
        errors.append(f"{FATAL_PREFIX} 알 수 없는 emergency_urgency 입니다: {urgency!r}")

    for action in state.get("ui_actions") or []:
        if not isinstance(action, dict) or not action.get("type"):
            errors.append(f"{FATAL_PREFIX} ui_action 형식이 올바르지 않습니다: {action!r}")
            break

    # 실제로 조립까지 해 본다 — 여기서 걸러야 Android 응답 단계에서 터지지 않는다.
    from .final_safety import build_chat_graph_result  # 순환 import 회피용 지연 import

    try:
        build_chat_graph_result(state, message=draft or "(비어 있음)")
    except Exception as exc:
        errors.append(f"{FATAL_PREFIX} 최종 결과 schema 조립에 실패했습니다: {exc}")

    return errors


# ---------------------------------------------------------------------------
# 종합 판정
# ---------------------------------------------------------------------------
def check_output(state: dict[str, Any]) -> OutputCheckResult:
    """명세 40절 9개 항목을 모두 검사해 `OutputCheckResult` 를 만든다(순수 함수).

    `action` 결정 규칙:
      - 오류 없음                → accept
      - 구조적(치명) 오류 존재    → fallback (문장을 다시 써도 해결되지 않는다)
      - 재생성 기회가 남아 있음  → regenerate
      - 이미 한 번 재생성했음    → fallback (명세 40절: 재생성 최대 1회)
    """
    draft = str(state.get("draft_response") or "")
    errors: list[str] = []

    errors.extend(_check_schema(state, draft))
    errors.extend(find_forbidden_expressions(draft))
    errors.extend(_check_risk_guidance(state, draft))
    errors.extend(_check_provenance(state, draft))
    errors.extend(_check_evidence_alignment(state, draft))
    errors.extend(_check_fabricated_facts(state, draft))
    errors.extend(_check_attachment_paths(state))

    retry_count = int(state.get("retry_count", 0) or 0)
    action = decide_action(errors, retry_count)

    return OutputCheckResult(valid=not errors, errors=errors, action=action)


def decide_action(
    errors: list[str], retry_count: int
) -> Literal["accept", "regenerate", "fallback"]:
    """오류 목록과 재생성 횟수로 다음 행동을 정한다(테스트하기 쉬운 순수 함수)."""
    if not errors:
        return "accept"
    if any(error.startswith(FATAL_PREFIX) for error in errors):
        return "fallback"
    if retry_count >= MAX_REGENERATION:
        return "fallback"
    return "regenerate"


# ---------------------------------------------------------------------------
# Node / Router
# ---------------------------------------------------------------------------
def output_check_node(state: dict) -> dict:
    """답변을 검사하고 accept / regenerate / fallback 을 State 에 남긴다.

    `validation_errors` 를 `Replace` 로 감싸는 이유: state.py 의 reducer 가 누적형이라
    감싸지 않으면 재생성 후 **해결된 오류가 목록에 계속 남는다.** 매 검사는 그 시점
    답변에 대한 완전한 판정이어야 한다.
    """
    result = check_output(state)
    retry_count = int(state.get("retry_count", 0) or 0)

    updates: dict[str, Any] = {
        "validation_errors": Replace(result.errors),
        "output_check_action": result.action,
    }
    if result.action == "regenerate":
        updates["retry_count"] = retry_count + 1

    if result.errors:
        logger.warning(
            "Output Check %s (retry=%d): %s", result.action, retry_count, result.errors
        )
    return updates


def route_output_check(state: dict) -> Literal["accept", "regenerate", "fallback"]:
    """`add_conditional_edges` 용 router — 분기는 LangGraph 가 한다(명세 19절).

    node 가 남긴 `output_check_action` 을 그대로 읽되, 없으면 State 만으로 같은
    결론을 다시 계산한다(node 순서가 바뀌어도 분기가 흔들리지 않게).
    """
    action = state.get("output_check_action")
    if action in ("accept", "regenerate", "fallback"):
        return action  # type: ignore[return-value]
    return decide_action(
        list(state.get("validation_errors") or []),
        int(state.get("retry_count", 0) or 0),
    )
