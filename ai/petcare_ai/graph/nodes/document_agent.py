"""Document Agent + PDF Generator node (명세 36·37절).

## 이 node 가 하지 않는 일이 핵심이다

명세 36절: **"일기장과 진단서를 새로 정리하거나 파싱하지 않는다."**
그래서 이 파일에는 증상 사전도, 요약 LLM 도, 날짜 파서도 없다. 이미 만들어진 결과
(`current_observation`, `priority_pet_context`, `related_diagnoses`,
`supporting_daily_entries`, 위험도 평가)와 대화에서 확보한 값
(`collected_information`)을 `ConsultationPacket` 자리에 **옮겨 담기만** 한다.

같은 원문을 두 번 해석하면 두 결과가 갈라지고, 그때 PDF 에 적힌 내용과 화면 답변이
서로 다른 말을 하게 된다. 진료 의뢰서에서 그것보다 나쁜 결함은 없다.

## 없는 정보

추측해서 채우지 않는다. 비어 있는 항목은 `unknown_fields` 에 라벨로 남기고, PDF 는
그것을 '미확인' 으로 인쇄한다(`pdf/consultation_pdf.py` 의 `UNKNOWN_LABEL`).

## 충돌

`context_conflicts` 를 **그대로 `provenance` 에 실어 보낸다.** PDF 의
`_normalize_provenance()` 가 `ContextConflict` 모양(`selected_value` /
`conflicting_values`)과 `ContextProvenance` 모양을 모두 인식하도록 이미 만들어져
있으므로, 여기서 값을 하나로 확정하거나 요약하면 안 된다(명세 20/43절).
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from ...pdf.consultation_pdf import UNKNOWN_LABEL
from ...schemas import ConsultationPacket

logger = logging.getLogger(__name__)

__all__ = [
    "CONDITION_KEY_MAP",
    "PET_FIELD_LABELS",
    "MEDICAL_HISTORY_LABELS",
    "resolve_document_type",
    "map_pet_section",
    "map_medical_history_section",
    "map_current_condition_section",
    "map_risk_assessment_section",
    "collect_unknown_fields",
    "collect_provenance",
    "build_consultation_packet",
    "document_agent_node",
    "pdf_generator_node",
]


#: `collected_information` 의 키(=`missing_information.InformationField.key`) →
#: ConsultationPacket `current_condition` 의 필드명.
#:
#: 영어 키는 PDF `_LABELS` 가 아는 이름이라 한국어 라벨로 예쁘게 인쇄되고,
#: 한국어 키는 PDF 가 그대로 라벨로 쓴다(`_label()` 은 모르는 키를 원문 유지한다).
#: 응급 8항목처럼 PDF 라벨 사전에 없는 항목은 한국어 키를 그대로 쓰는 편이 읽기 좋다.
CONDITION_KEY_MAP: dict[str, str] = {
    # 일반·방문 상담
    "main_symptom": "main_symptoms",
    "symptom_onset": "onset",
    "frequency": "frequency",
    "symptom_change": "change",
    "current_intake": "현재 식사·음수·활동 상태",
    # 응급 최소정보(명세 29절 8항목)
    "worst_symptom": "main_symptoms",
    "onset_time": "onset",
    "approximate_count": "frequency",
    "still_ongoing": "현재도 진행 중인지",
    "consciousness": "의식 또는 반응 상태",
    "breathing": "호흡 상태",
    "mobility": "움직일 수 있는지",
    "trauma_or_toxin": "외상 또는 위험물질 섭취 가능성",
    # 항목을 특정하지 못한 자유 서술(명세 29절: 임의 배정하지 않는다)
    "free_text_answer": "보호자 추가 서술",
}

#: PET DB 필드 → PDF 라벨. `unknown_fields` 에 넣을 이름을 여기서 가져온다.
PET_FIELD_LABELS: dict[str, str] = {
    "name": "이름",
    "species": "종",
    "breed": "품종",
    "birth_date": "나이(생년월일)",
    "sex": "성별",
    "weight_kg": "몸무게",
}

MEDICAL_HISTORY_LABELS: dict[str, str] = {
    "diseases": "기존 질병",
    "medications": "복용 중인 약",
    "allergies": "알레르기",
}

#: PET DB 에서 medical_history 로 옮길 필드(이름 변환 포함).
_HISTORY_SOURCE_KEYS: tuple[tuple[str, str], ...] = (
    ("diseases", "existing_conditions"),
    ("medications", "medications"),
    ("supplement", "복용 중인 영양제"),
    ("allergies", "allergies"),
    ("surgeries", "surgeries"),
    ("vaccinations", "vaccinations"),
)

#: PET DB 에서 pet 섹션으로 옮길 필드.
_PET_SOURCE_KEYS: tuple[str, ...] = (
    "id", "name", "species", "breed", "birth_date", "age_years", "sex",
    "is_neutered", "weight_kg", "microchip",
)


def _is_blank(value: Any) -> bool:
    """'값 없음' 판정 — PDF 모듈과 같은 기준을 쓴다."""
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    if isinstance(value, (list, tuple, set, dict)):
        return len(value) == 0
    return False


def _clean(value: Any) -> Any:
    """문자열은 양끝 공백만 정리한다. **내용은 손대지 않는다**(요약 금지)."""
    return value.strip() if isinstance(value, str) else value


# ---------------------------------------------------------------------------
# 섹션별 매핑
# ---------------------------------------------------------------------------
def resolve_document_type(state: dict[str, Any]) -> str:
    """문서 종류를 정한다: `emergency_consultation` / `visit_consultation`.

    State 에 이미 값이 있으면 존중한다(상위 subgraph 가 명시적으로 정한 경우).
    없으면 위험도와 응급 긴급도로 판단하며, 둘 중 하나라도 응급이면 응급 문서다 —
    **낮은 쪽으로 내리지 않는다**(명세 28절 정신을 문서 종류에도 적용).
    """
    declared = str(state.get("document_type") or "")
    if declared in ("visit_consultation", "emergency_consultation"):
        return declared

    if (
        str(state.get("final_risk") or "normal") == "emergency"
        or str(state.get("emergency_urgency") or "none") != "none"
    ):
        return "emergency_consultation"
    return "visit_consultation"


def map_pet_section(state: dict[str, Any]) -> dict[str, Any]:
    """반려동물 기본정보를 옮긴다(PET DB 원본 값 그대로).

    `priority_pet_context`(Clinical Context Priority 결과)를 우선 쓰고, 없으면
    `pet_profile` 을 본다. 값을 가공하거나 나이를 계산하지 않는다 — 계산하면 그것이
    또 하나의 '새로 만든 사실' 이 된다.
    """
    source = state.get("priority_pet_context") or state.get("pet_profile") or {}
    if not isinstance(source, dict):
        return {}
    return {key: _clean(source[key]) for key in _PET_SOURCE_KEYS if not _is_blank(source.get(key))}


def map_medical_history_section(state: dict[str, Any]) -> dict[str, Any]:
    """기존 질병·복용약·알레르기를 옮긴다."""
    source = state.get("priority_pet_context") or state.get("pet_profile") or {}
    if not isinstance(source, dict):
        return {}
    history: dict[str, Any] = {}
    for source_key, target_key in _HISTORY_SOURCE_KEYS:
        if not _is_blank(source.get(source_key)):
            history[target_key] = _clean(source[source_key])
    return history


def map_current_condition_section(state: dict[str, Any]) -> dict[str, Any]:
    """현재 증상 섹션을 만든다.

    우선순위(명세 20절)를 자리 단위로 적용한다.

    1. `collected_information` — 되물어 받은 **명시적 답변**이 가장 정확하다.
    2. `current_observation` — 같은 사용자 입력에서 규칙으로 추출한 값. 1번이 채우지
       못한 자리만 메운다.

    같은 자리에 서로 다른 답이 두 번 들어오면(예: 응급 `worst_symptom` 과 일반
    `main_symptom`) 하나를 고르지 않고 ' / ' 로 **둘 다 남긴다.** 어느 쪽이 맞는지는
    수의사가 판단할 몫이다.
    """
    condition: dict[str, Any] = {}

    def _put(key: str, value: Any) -> None:
        if _is_blank(value):
            return
        cleaned = _clean(value)
        existing = condition.get(key)
        if _is_blank(existing):
            condition[key] = cleaned
        elif str(existing) != str(cleaned):
            condition[key] = f"{existing} / {cleaned}"

    collected = state.get("collected_information") or {}
    if isinstance(collected, dict):
        for raw_key, value in collected.items():
            target = CONDITION_KEY_MAP.get(str(raw_key), str(raw_key))
            _put(target, value)

    observation = state.get("current_observation") or {}
    if isinstance(observation, dict):
        symptoms = observation.get("symptoms") or []
        if symptoms:
            _put("main_symptoms", ", ".join(str(item) for item in symptoms))
        _put("onset", observation.get("onset"))
        _put("duration", observation.get("duration"))
        _put("severity", observation.get("severity"))
        _put("observation", observation.get("raw_text"))

        measurements = observation.get("measurements") or {}
        if isinstance(measurements, dict) and measurements:
            # 보호자가 말한 수치는 PET DB 값과 다를 수 있다. 덮어쓰지 않고 별도
            # 항목으로 남긴다(충돌 자체는 provenance 에 기록돼 있다).
            _put(
                "보호자가 보고한 수치",
                ", ".join(f"{key}={value}" for key, value in measurements.items()),
            )

    return condition


def map_risk_assessment_section(state: dict[str, Any]) -> dict[str, Any]:
    """AI 위험도 분류 섹션을 옮긴다.

    PDF 는 이 값을 항상 'AI 참고 분류' 로 인쇄한다. 여기서 진단명으로 바꾸거나
    문장을 만들지 않는다(확정 진단 금지).
    """
    return {
        "risk_level": str(state.get("final_risk") or "normal"),
        "emergency_urgency": str(state.get("emergency_urgency") or "none"),
        "red_flags": [str(item) for item in (state.get("red_flags") or [])],
        "reasons": [str(item) for item in (state.get("risk_reasons") or [])],
        "missing_information": [str(item) for item in (state.get("missing_fields") or [])],
    }


def collect_unknown_fields(
    state: dict[str, Any],
    pet: dict[str, Any],
    medical_history: dict[str, Any],
    current_condition: dict[str, Any],
) -> list[str]:
    """'미확인' 으로 표시할 항목을 모은다(순서 유지 + 중복 제거).

    포함 대상:
      - 아직 답을 못 받은 필수정보(`missing_fields`) — 이미 한국어 라벨이다.
      - PET DB 기본 항목 중 비어 있는 것.
      - 기존 질병·복용약·알레르기 중 비어 있는 것.

    **추측으로 채우지 않는다는 사실 자체가 정보** 다. 수의사는 '알레르기 없음' 과
    '알레르기 미확인' 을 완전히 다르게 다룬다.
    """
    unknown: list[str] = []
    seen: set[str] = set()

    def _add(label: str) -> None:
        text = str(label).strip()
        if text and text not in seen:
            seen.add(text)
            unknown.append(text)

    for label in state.get("missing_fields") or []:
        _add(str(label))

    for key, label in PET_FIELD_LABELS.items():
        if _is_blank(pet.get(key)):
            # 나이는 birth_date 또는 age_years 중 하나만 있어도 확인된 것으로 본다.
            if key == "birth_date" and not _is_blank(pet.get("age_years")):
                continue
            _add(label)

    history_keys = {
        "diseases": "existing_conditions",
        "medications": "medications",
        "allergies": "allergies",
    }
    for source_key, target_key in history_keys.items():
        if _is_blank(medical_history.get(target_key)):
            _add(MEDICAL_HISTORY_LABELS[source_key])

    if _is_blank(current_condition.get("main_symptoms")) and _is_blank(
        current_condition.get("observation")
    ):
        _add("주요 증상")

    return unknown


def collect_provenance(state: dict[str, Any]) -> list[dict[str, Any]]:
    """출처 기록과 **충돌 기록을 함께** 넘긴다(명세 20/36/43절).

    두 리스트를 하나로 합치는 이유: `ConsultationPacket` 에는 provenance 필드만
    있고, PDF 는 두 모양을 모두 이해하도록 이미 구현돼 있다. 충돌을 별도로 요약해
    넘기면 그 과정에서 값이 하나로 확정되어 버린다 — 명세가 금지하는 동작이다.
    """
    records: list[dict[str, Any]] = []
    for item in state.get("context_provenance") or []:
        if isinstance(item, dict):
            records.append(dict(item))
        else:
            logger.warning("provenance 항목 형식이 올바르지 않아 제외합니다: %r", item)

    for item in state.get("context_conflicts") or []:
        if isinstance(item, dict):
            records.append(dict(item))
        else:
            logger.warning("context_conflict 항목 형식이 올바르지 않아 제외합니다: %r", item)
    return records


# ---------------------------------------------------------------------------
# Packet 조립
# ---------------------------------------------------------------------------
def build_consultation_packet(state: dict[str, Any]) -> ConsultationPacket:
    """State → `ConsultationPacket` 매핑(순수 함수 — LLM 을 쓰지 않는다).

    진단서·일기장은 Clinical Context Priority 가 **이미 선택해 둔** 레코드를 원문
    그대로 싣는다. 여기서 다시 고르거나 요약하면 명세 36절 위반이다. 선택 결과가
    비어 있으면 비운 채로 둔다 — PDF 가 '제공된 기록 없음' 으로 인쇄한다.
    """
    pet = map_pet_section(state)
    medical_history = map_medical_history_section(state)
    current_condition = map_current_condition_section(state)

    related_diagnoses = [
        dict(item) for item in (state.get("related_diagnoses") or []) if isinstance(item, dict)
    ]
    supporting_daily_entries = [
        dict(item)
        for item in (state.get("supporting_daily_entries") or [])
        if isinstance(item, dict)
    ]

    packet = ConsultationPacket(
        document_type=resolve_document_type(state),  # type: ignore[arg-type]
        generated_at=datetime.now().isoformat(timespec="seconds"),
        pet=pet,
        medical_history=medical_history,
        current_condition=current_condition,
        related_diagnoses=related_diagnoses,
        supporting_daily_entries=supporting_daily_entries,
        risk_assessment=map_risk_assessment_section(state),
        unknown_fields=collect_unknown_fields(state, pet, medical_history, current_condition),
        provenance=collect_provenance(state),
    )
    logger.info(
        "상담 packet 생성 — 종류=%s, 진단서 %d건 / 일기 %d건, 미확인 %d항목, 출처기록 %d건",
        packet.document_type,
        len(packet.related_diagnoses),
        len(packet.supporting_daily_entries),
        len(packet.unknown_fields),
        len(packet.provenance),
    )
    return packet


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------
def document_agent_node(state: dict) -> dict:
    """Document Agent (명세 36절) — packet 만 만든다. PDF 생성은 다음 node.

    두 단계를 나눈 이유: packet 조립은 실패할 수 없는 순수 매핑이고, PDF 생성은
    폰트·파일시스템 문제로 실패할 수 있다. 합쳐 두면 PDF 실패가 packet 까지 없애
    "무엇을 담으려 했는지" 조차 남지 않는다.
    """
    packet = build_consultation_packet(state)
    return {
        "consultation_packet": packet.model_dump(),
        "document_type": packet.document_type,
    }


def pdf_generator_node(state: dict) -> dict:
    """PDF Generator node (명세 37절) — `pdf/consultation_pdf.py` 를 호출한다.

    PDF 생성에 실패해도 **상담 자체를 실패시키지 않는다.** 위험도 안내와 병원 정보는
    이미 답변에 담겨 있고, 그것이 첨부 파일보다 중요하다. 실패 사실은
    `validation_errors` 에 남겨 Output Check 와 trace 에서 확인할 수 있게 한다.

    `attachment_path` 불일치를 막기 위해 경로·파일명을 **여기서 확정하고** Email
    Draft node 는 State 값을 그대로 읽어 쓴다(명세 40절 8번 검사 대상).
    """
    raw_packet = state.get("consultation_packet") or {}
    if not raw_packet:
        logger.warning("consultation_packet 이 없어 PDF 를 만들지 않습니다.")
        return {}

    try:
        packet = ConsultationPacket.model_validate(raw_packet)
    except Exception as exc:
        logger.warning("consultation_packet 형식 오류로 PDF 를 만들지 못했습니다: %s", exc)
        return {"validation_errors": [f"[PDF] packet 형식 오류: {exc}"]}

    from ...pdf.consultation_pdf import generate_consultation_pdf  # 지연 import(reportlab)

    try:
        pdf_path, filename = generate_consultation_pdf(packet)
    except Exception as exc:  # reportlab 미설치·폰트·권한 문제 등
        logger.warning("PDF 생성 실패 — 첨부 없이 진행합니다: %s", exc)
        return {"validation_errors": [f"[PDF] 생성 실패: {exc}"]}

    logger.info("상담 PDF 생성 완료: %s", pdf_path)
    return {
        "pdf_path": pdf_path,
        "pdf_filename": filename,
        "ui_actions": [
            {
                "type": "OPEN_PDF_PREVIEW",
                "path": pdf_path,
                "filename": filename,
                "notice": f"확정 진단이 아닌 상담 보조 자료입니다. 값이 없는 항목은 '{UNKNOWN_LABEL}' 로 표기됩니다.",
            }
        ],
    }
