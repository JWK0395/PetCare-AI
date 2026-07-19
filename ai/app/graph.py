"""페이지별 진입점 재노출 + 병원 전달용 요약(run_summary).

페이지별 입출력은 각 파일로 분리되어 있다(직접 만든 AI 코드가 붙는 자리):

    - health_check.py       (앱 'AI 체크')  → run_health_check   [petcare_ai LangGraph]
    - diary_extract.py      (앱 '기록')     → run_diary_extract  [OpenAI structured output]
    - diagnosis_extract.py  (앱 '진료')     → run_diagnosis_extract

이 파일은 위 3개를 재노출(backward-compat)하고, 병원 전달용 요약(run_summary)을 담는다.

## run_summary 가 LLM 을 쓰지 않는 이유

이 문서는 보호자가 **병원에 그대로 들고 가는 자료**다. LLM 이 문장을 새로 지어내면
기록에 없는 사실이 진료 현장에 들어갈 수 있다. 그래서 요약은
`petcare_ai.graph.nodes.document_agent.build_consultation_packet()` 과 같은 원칙 —
"이미 있는 값을 정해진 자리에 옮겨 담기만 한다" — 을 따르며, 실제로 그 함수와
`clinical_context_priority.build_clinical_context()` 를 재사용한다.

petcare_ai 를 불러오지 못해도(패키지 미설치 등) 서버 계약(SummaryContent 16개 키)은
빠짐없이 채운다. 키가 하나라도 빠지면 서버가 502 로 요약 저장을 거부한다.

계약 원문: ai/README.md · 서버 소비부: server/app/routers/summaries.py
"""

from __future__ import annotations

import logging

# 페이지별 진입점 재노출 — `from .graph import run_health_check` 형태 하위 호환용.
from .diagnosis_extract import run_diagnosis_extract  # noqa: F401
from .diary_extract import run_diary_extract  # noqa: F401

# 기록 텍스트 판정 규칙은 health_check 가 소유한다. 여기서 다시 쓰면 화면의 추이
# 요약과 병원에 제출하는 문서가 서로 다른 말을 하게 된다.
from .health_check import (  # noqa: F401  (run_health_check 는 재노출용)
    run_health_check,
)
from .io_schemas import AgentContext, PetProfile

logger = logging.getLogger(__name__)

SUMMARY_TITLE = "PetCare AI 병원 전달용 상태 요약"

#: 서버 위험도(4단계) → 문서에 인쇄할 라벨. 확정 진단이 아니라 **AI 참고 분류**다.
RISK_LABELS: dict[str, str] = {
    "normal": "정상 범위",
    "observe": "관찰 권장",
    "consult": "신속 상담 권장",
    "emergency": "응급 징후 가능성",
}

def _data_period(context: AgentContext) -> str:
    """문서에 인쇄할 '사용 데이터 기간' (예: 2026.06.17 ~ 2026.07.16).

    기록이 없으면 빈 문자열이다 — 기간을 지어내면 수의사가 없는 관찰을 있다고 읽는다.
    기록은 오래된 순으로 오므로 처음과 끝이 곧 기간이다.
    """
    records = list(context.records or [])
    if not records:
        return ""
    first = str(getattr(records[0], "record_date", "") or "").replace("-", ".")
    last = str(getattr(records[-1], "record_date", "") or "").replace("-", ".")
    if not first or not last:
        return ""
    return first if first == last else f"{first} ~ {last}"


#: 서버 위험도 → petcare_ai 위험도(3단계). document_agent 가 문서 종류를 고를 때 쓴다.
_RISK_TO_PETCARE: dict[str, str] = {
    "normal": "normal",
    "observe": "normal",
    "consult": "visit",
    "emergency": "emergency",
}


# --------------------------------------------------------------------------
# 기록 텍스트에서 상태 섹션 만들기 (LLM 없이 — 값을 지어내지 않는다)
# --------------------------------------------------------------------------
def _first_filled_list(*candidates: list[str]) -> list[str]:
    """앞에서부터 비어 있지 않은 첫 목록을 고른다."""
    for value in candidates:
        items = [str(v).strip() for v in (value or []) if str(v).strip()]
        if items:
            return items
    return []


#: 문서에 인쇄할 위험 징후 최대 개수. 수의사가 훑어보는 자리라 길면 오히려 안 읽힌다.
MAX_RISK_SIGNS = 5


def _dedupe_signs(signs: list[str]) -> list[str]:
    """위험 징후를 겹치는 것끼리 합치고 개수를 제한한다.

    평가 노드가 규칙·LLM·일기 추세에서 각각 신호를 올리다 보니 같은 사실이 다른
    문장으로 여러 번 들어온다("구토" / "구토가 있었음" / "구토가 동반된 날이 있음").
    12개가 나열되면 병원에서 무엇이 중요한지 알 수 없다.

    한쪽이 다른 쪽의 부분 문자열이면 **짧은 쪽을 남긴다** — 짧은 표현이 대개
    사실만 담고, 긴 쪽은 같은 사실에 설명을 덧붙인 것이다.
    """
    cleaned = [str(x).strip() for x in signs if str(x).strip()]
    kept: list[str] = []
    for sign in sorted(cleaned, key=len):
        core = sign.replace(" ", "")
        if any(core in k.replace(" ", "") or k.replace(" ", "") in core for k in kept):
            continue
        kept.append(sign)
        if len(kept) >= MAX_RISK_SIGNS:
            break
    return kept


def _first_filled(*candidates: str) -> str:
    """앞에서부터 비어 있지 않은 첫 값을 고른다."""
    for value in candidates:
        text = str(value or "").strip()
        if text and text != "-":
            return text
    return ""


def _latest_diagnosis(context: AgentContext) -> str:
    """가장 최근 진단서의 진단명(없으면 빈 문자열)."""
    for record in reversed(list(context.diagnoses or [])):
        name = str(getattr(record, "diagnosis", "") or "").strip()
        if name:
            stamp = str(getattr(record, "date", "") or "").strip()
            return f"{name} ({stamp} 진단)" if stamp else name
    return ""


def _diary_symptoms(context: AgentContext, limit: int = 3) -> str:
    """일기장에 보호자가 직접 적은 증상을 최근 순으로 모은다.

    **날짜를 함께 남긴다.** 언제 관찰된 것인지 없으면 수의사가 오늘 상태로 오해한다.
    """
    seen: list[str] = []
    for record in reversed(list(context.records or [])):
        symptom = str(getattr(record, "symptom", "") or "").strip()
        if not symptom or symptom in ("없음", "정상"):
            continue
        stamp = str(getattr(record, "record_date", "") or "").strip()
        line = f"{stamp} {symptom}" if stamp else symptom
        if line not in seen:
            seen.append(line)
        if len(seen) >= limit:
            break
    return " / ".join(seen)


def _diary_changes(context: AgentContext, limit: int = 3) -> str:
    """식사·활동에 변화가 적힌 날을 모은다(보호자가 쓴 문장 그대로)."""
    seen: list[str] = []
    for record in reversed(list(context.records or [])):
        parts = [
            str(getattr(record, field, "") or "").strip()
            for field in ("food", "activity")
        ]
        parts = [p for p in parts if p and p not in ("정상", "정상 범위", "없음")]
        if not parts:
            continue
        stamp = str(getattr(record, "record_date", "") or "").strip()
        line = f"{stamp} " + " · ".join(parts) if stamp else " · ".join(parts)
        if line not in seen:
            seen.append(line)
        if len(seen) >= limit:
            break
    return " / ".join(seen)


def _diary_period(context: AgentContext) -> str:
    """증상이 적힌 기간 — '경과' 자리의 최후 보루."""
    dates = [
        str(getattr(r, "record_date", "") or "").strip()
        for r in (context.records or [])
        if str(getattr(r, "symptom", "") or "").strip() not in ("", "없음", "정상")
    ]
    dates = [d for d in dates if d]
    if not dates:
        return ""
    if len(dates) == 1:
        return f"{dates[0]} 하루 기록"
    return f"{dates[0]} ~ {dates[-1]} ({len(dates)}일 기록)"


def run_summary(pet: dict, risk_level: str, extra_note: str, context: dict) -> dict:
    """병원 전달용 요약(문서 4섹션)을 만든다 — **LangGraph 의 문서 에이전트로**.

    ## 왜 그래프를 태우는가

    예전에는 이 함수가 정규식으로 기록을 훑어(식사 감소·구토 등) 문장을 조립했다.
    그것은 명세 20·36절의 Document Agent 를 규칙으로 다시 구현한 것이라, 같은
    데이터에서 그래프와 다른 결론이 나올 수 있었다. 병원에 전달하는 문서가
    화면 안내와 어긋나는 것은 그 자체로 위험하다.

    이제는 그래프를 한 번 태워 `ConsultationPacket` 을 만들고, 그 값만 서버 스키마로
    옮긴다. 판단은 전부 그래프가 한다 — 여기서는 자리만 맞춘다.

    반환 `content` 는 서버 `schemas.SummaryContent` 로 검증되므로 **16개 키를 항상
    모두 채운다**(값이 없으면 빈 문자열로 두되 키는 뺴지 않는다).
    `created_at` 은 서버가 저장 시각으로 넣으므로 여기서 만들지 않는다.
    """
    from .health_check import run_graph_for_packet

    profile = PetProfile(**(pet or {}))
    ctx = AgentContext(**(context or {}))
    note = (extra_note or "").strip()

    packet, source = run_graph_for_packet(profile, ctx, note, risk_level)
    condition = packet.get("current_condition") or {}
    risk = packet.get("risk_assessment") or {}

    sex_neuter = " / ".join(
        x
        for x in (profile.sex, "중성화 완료" if profile.is_neutered else "중성화 안 함")
        if x
    )

    def _joined(*keys: str) -> str:
        """패킷의 여러 자리를 한 줄로 합친다(없는 자리는 건너뛴다)."""
        parts = [str(condition.get(k) or "").strip() for k in keys]
        return " · ".join(p for p in parts if p)

    content = {
        # 1. 문서 정보
        "title": SUMMARY_TITLE,
        "data_period": _data_period(ctx),
        # 2. 반려동물 정보 — 프로필 원본을 그대로 옮긴다(판단이 필요 없는 자리)
        "pet_name": profile.name,
        "species": profile.species,
        "breed": profile.breed,
        "sex_neuter": sex_neuter,
        "age_label": profile.age_label,
        "weight": f"{profile.weight_kg}kg" if profile.weight_kg else "",
        "medications": profile.medications,
        "allergies": profile.allergies,
        # 3. 상태 — 위험도와 징후는 그래프의 판정을 그대로 쓴다
        "risk_label": RISK_LABELS.get(risk_level, ""),
        "risk_signs": _dedupe_signs(
            _first_filled_list(
                [str(x) for x in (risk.get("red_flags") or []) if str(x).strip()],
                [line for line in _diary_symptoms(ctx, limit=5).split(" / ") if line],
            )
        ),
        # 4. 주호소 및 주요 변화
        #
        # **채우는 순서: 그래프 판정 → 이번 대화 → 진단서 → 일기장.**
        # 앞에서 값을 못 찾으면 다음 자료로 내려간다. 어느 자리도 지어내지 않는다 —
        # 전부 보호자가 적었거나 병원이 발급한 원문이다.
        #
        # **주호소만 대화가 진단서보다 앞선다.** 주호소는 "오늘 무엇 때문에 왔는가"
        # 이고 그건 방금 나눈 대화다. 예전에는 진단서가 위에 있어서, 보호자가
        # "산책 갔다왔는데 발이 빨개" 라고 상담했는데 문서의 주호소가
        # "슬개골 탈구 2기 (2026-07-02 진단)" 로 찍혔다. 기존 질환은 이미 2번
        # 섹션(반려동물 정보)에 있으므로 주호소 자리를 차지하면 안 된다.
        #
        # 반대로 '주요 변화' 는 추세라서 일기장이 대화보다 정확하다(며칠치를 본다).
        "chief_complaint": _first_filled(
            _joined("main_symptom", "worst_symptom"),
            note,  # 이번 상담의 대화 원문 — '오늘 왜 왔는가' 는 여기에 있다
            _latest_diagnosis(ctx),
            _diary_symptoms(ctx),
        ),
        "major_changes": _first_filled(
            _joined("symptom_change", "current_intake"),
            _diary_changes(ctx),
            note,
        ),
        "progress": _first_filled(
            _joined("symptom_onset", "frequency", "still_ongoing"),
            _diary_period(ctx),
        ),
        "owner_note": note,
    }
    logger.info(
        "요약 생성(graph) — 위험도=%s, 기록 %d건, 징후 %d개, source=%s",
        risk_level,
        len(ctx.records),
        len(content["risk_signs"]),
        source,
    )
    return {"content": content, "source": source}
