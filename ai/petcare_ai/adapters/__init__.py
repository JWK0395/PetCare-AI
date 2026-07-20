"""외부 시스템(기존 앱 서버의 일기장·진단서 처리 코드)과 붙이는 얇은 adapter 계층.

LangGraph 노드는 이 계층의 반환 스키마에만 의존한다. 앱 연동 방식이 바뀌어도
adapter 내부만 고치면 되고 `petcare_ai/schemas.py` 계약은 그대로 유지된다.
"""

from __future__ import annotations

from .clinical_data_adapter import (
    USE_EXISTING_PROCESSORS,
    ClinicalDataAdapter,
    ExistingProcessorAdapter,
    FixtureClinicalDataAdapter,
    get_adapter,
    load_daily_entries_for_test,
    load_diagnoses_for_test,
    load_pet_profile_for_test,
)

__all__ = [
    "USE_EXISTING_PROCESSORS",
    "ClinicalDataAdapter",
    "ExistingProcessorAdapter",
    "FixtureClinicalDataAdapter",
    "get_adapter",
    "load_daily_entries_for_test",
    "load_diagnoses_for_test",
    "load_pet_profile_for_test",
]
