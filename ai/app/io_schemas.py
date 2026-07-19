"""AI Agent 공통 입출력 스키마.

메인 서버(server/)가 모든 엔드포인트에 공통으로 넣어 보내는 조각들 —
반려동물 프로필(pet), 개인 데이터 묶음(context), 대화 메시지(message).

각 페이지 파일이 이 조각들을 재사용한다:
    - health_check.py       (앱 'AI 체크')
    - diary_extract.py      (앱 '기록')
    - diagnosis_extract.py  (앱 '진료')

계약 원문: ai/README.md · 서버 연결부: server/app/services/context.py (pet_payload / build_context)
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class PetProfile(BaseModel):
    """반려동물 프로필. (server: context.pet_payload)"""

    id: int = 0
    name: str = ""
    species: str = ""  # 강아지 | 고양이
    breed: str = ""  # 견종/묘종 (예: "말티즈 · 순종")
    birth_date: Optional[str] = None  # "YYYY-MM-DD"
    age_label: str = ""  # 예: "만 4세"
    sex: str = ""  # 수컷 | 암컷
    is_neutered: bool = False
    weight_kg: Optional[float] = None
    size_class: str = ""  # 소형 | 중형 | 대형
    diseases: str = ""  # 기존 질병 (텍스트)
    medications: str = ""  # 복용약 (텍스트)
    supplement: str = ""  # 영양제 (텍스트)
    allergies: str = ""  # 알레르기 (텍스트)


class DailyRecord(BaseModel):
    """최근 30일 일일 기록 한 건 — 모두 텍스트 상태값. (DB: daily_entries)"""

    record_date: str  # "YYYY-MM-DD"
    raw_text: str = ""  # 보호자가 쓴 일기 원문
    food: str = ""  # 식사
    water: str = ""  # 음수
    activity: str = ""  # 활동
    symptom: str = ""  # 증상
    stool: str = ""  # 배변(설사 포함)
    vomit: str = ""  # 구토
    notes: str = ""  # 기타사항


class DiagnosisRecord(BaseModel):
    """확정 저장된 진단서. (DB: diagnoses)"""

    date: Optional[str] = None  # "YYYY-MM-DD"
    hospital: str = ""
    diagnosis: str = ""
    content: str = ""


class AgentContext(BaseModel):
    """서버가 만들어 주는 개인 데이터 묶음 (Agent 는 DB 직접 접근 불필요)."""

    window_days: int = 30
    records: list[DailyRecord] = Field(default_factory=list)  # 오래된 순
    diagnoses: list[DiagnosisRecord] = Field(default_factory=list)  # 오래된 순


class ChatMessage(BaseModel):
    role: str  # user | assistant
    content: str
