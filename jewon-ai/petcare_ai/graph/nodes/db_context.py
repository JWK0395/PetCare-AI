"""DB Context Agent — 대화 시작 시 현재 pet 의 임상 데이터 전체를 State 에 싣는다.

명세 21절:
  - 첫 대화 시작 시 선택된 반려동물의 데이터를 **모두** State 에 넣는다
    (pet_profile 1건 / diagnoses 전체 / daily_entries 전체).
  - **모든 반려동물이 아니라 현재 `pet_id` 데이터만** 넣는다.
  - State 에는 전체를 저장하되 LLM prompt 에는 필요한 것만 전달한다
    → 그 선별은 Clinical Context Priority Agent 가 한다. 여기서는 요약·가공하지
      않고 원본 그대로 싣는다.

이 노드는 임상 데이터를 해석하지 않는다. adapter 호출 + 정렬 + 종 정규화만 한다.
"""

from __future__ import annotations

import logging
from datetime import date, datetime
from typing import TYPE_CHECKING, Any, Callable

from ...adapters.clinical_data_adapter import (
    USE_EXISTING_PROCESSORS,
    ClinicalDataAdapter,
    get_adapter,
    normalize_species,
)

if TYPE_CHECKING:  # state.py 는 동시 작성 중이므로 런타임 import 하지 않는다.
    from ..state import PetCareState  # noqa: F401

logger = logging.getLogger(__name__)

__all__ = [
    "set_clinical_adapter",
    "get_clinical_adapter",
    "make_db_context_node",
    "db_context_node",
    "needs_db_context",
    "route_context_loaded",
]

# 노트북/테스트가 주입한 adapter. None 이면 get_adapter() 로 만든다.
_adapter: ClinicalDataAdapter | None = None


def set_clinical_adapter(adapter: ClinicalDataAdapter | None) -> None:
    """전역 adapter 를 주입한다(Colab 노트북·테스트용).

    운영에서는 `get_adapter(use_existing=True)` 가 기본이지만, Colab 처럼 서버
    코드가 없는 환경은 fixture adapter 를 **명시적으로** 주입해야 한다.
    명세 4절대로 "조용한 fixture 대체" 를 막기 위해 선택을 호출자에게 남긴다.
    """
    global _adapter
    _adapter = adapter


def get_clinical_adapter() -> ClinicalDataAdapter:
    """사용할 adapter 를 돌려준다. 주입값이 없으면 설정 기본값으로 만든다.

    `get_adapter()` 가 던지는 RuntimeError 는 잡지 않는다 — fixture 를 실제
    데이터로 착각한 채 그래프가 도는 것이 가장 위험하기 때문이다(명세 4절).
    """
    if _adapter is not None:
        return _adapter
    return get_adapter(USE_EXISTING_PROCESSORS)


# ---------------------------------------------------------------------------
# 정렬 유틸
# ---------------------------------------------------------------------------
def _parse_date(value: Any) -> date | None:
    """'YYYY-MM-DD' 문자열·date·datetime 을 date 로 바꾼다. 실패하면 None."""
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if not value:
        return None
    text = str(value).strip()
    for candidate in (text, text[:10]):
        try:
            return date.fromisoformat(candidate)
        except ValueError:
            continue
    logger.debug("날짜를 해석하지 못했습니다: %r", value)
    return None


def _sort_by_date(items: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    """날짜 오름차순 정렬 — 마지막이 최신이 되도록 통일한다.

    날짜를 읽을 수 없는 레코드는 **버리지 않고** 맨 앞으로 보낸다. 임상 기록을
    조용히 누락시키는 것보다 순서가 어긋나는 편이 안전하다.
    """
    return sorted(items, key=lambda item: (_parse_date(item.get(key)) or date.min))


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------
def make_db_context_node(
    adapter: ClinicalDataAdapter | None = None,
) -> Callable[[dict], dict]:
    """adapter 를 주입한 노드를 만든다(테스트가 mock adapter 를 넣을 수 있게)."""

    def _node(state: dict) -> dict:
        return _load_context(state, adapter or get_clinical_adapter())

    return _node


def db_context_node(state: dict) -> dict:
    """현재 pet_id 의 PET / 진단서 / 일기장 데이터를 State 에 적재한다.

    이미 같은 pet 의 context 가 로드돼 있으면 아무 것도 하지 않는다(두 번째 turn
    부터는 DB 를 다시 읽지 않는다). pet 이 바뀌면 다시 로드한다 — thread 간
    pet 데이터가 섞이면 안 된다(명세 43절 공통 테스트).
    """
    return _load_context(state, get_clinical_adapter())


def _load_context(state: dict, adapter: ClinicalDataAdapter) -> dict:
    if not needs_db_context(state):
        logger.debug("임상 context 가 이미 로드되어 DB Context 를 건너뜁니다.")
        return {}

    pet_id = state.get("pet_id")
    if pet_id is None:
        # 종(species)을 모르면 RAG index 선택부터 틀어진다. 조용히 빈 context 로
        # 진행하면 잘못된 근거로 답할 수 있으므로 여기서 멈춘다.
        raise ValueError(
            "pet_id 가 State 에 없어 임상 context 를 불러올 수 없습니다. "
            "그래프 invoke 시 pet_id 를 반드시 넣어 주세요."
        )

    profile = dict(adapter.load_pet_profile(int(pet_id)))
    profile["species"] = normalize_species(profile.get("species"))

    diagnoses = _sort_by_date(list(adapter.load_diagnoses(int(pet_id))), "date")
    daily_entries = _sort_by_date(
        list(adapter.load_daily_entries(int(pet_id))), "record_date"
    )

    logger.info(
        "임상 context 로드 완료 — pet_id=%s species=%s 진단서 %d건 일기 %d건",
        pet_id,
        profile.get("species"),
        len(diagnoses),
        len(daily_entries),
    )

    return {
        "pet_profile": profile,
        "diagnoses": diagnoses,
        "daily_entries": daily_entries,
        "context_loaded": True,
    }


def needs_db_context(state: dict) -> bool:
    """다시 로드해야 하는지 판단한다 — pet 이 바뀐 경우도 재로드 대상이다."""
    if not state.get("context_loaded"):
        return True
    profile = state.get("pet_profile") or {}
    if not profile:
        return True
    pet_id = state.get("pet_id")
    loaded_id = profile.get("id")
    if pet_id is None or loaded_id is None:
        return False
    return int(loaded_id) != int(pet_id)


def route_context_loaded(state: dict) -> str:
    """`add_conditional_edges` 용 분기 함수 (명세 24절 START → Context loaded?)."""
    return "db_context" if needs_db_context(state) else "message_ingest"
