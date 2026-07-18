from __future__ import annotations

from datetime import date, datetime

from sqlalchemy import JSON, Boolean, Date, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .database import Base


def utcnow() -> datetime:
    return datetime.utcnow()


class User(Base):
    """사용자 계정 — 이메일 + 비밀번호 로그인 (비밀번호 찾기 없음)"""

    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    email: Mapped[str] = mapped_column(String(120), unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(String(200))
    # 로그인 세션 토큰 (단일 세션 — 재로그인하면 이전 토큰은 무효)
    token: Mapped[str | None] = mapped_column(String(80), nullable=True, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)

    pets: Mapped[list["Pet"]] = relationship(
        back_populates="owner", cascade="all, delete-orphan"
    )


class Pet(Base):
    """PET DB — 이름, 견종, 생년월일, 성별, 중성화 여부, 몸무게, 질병·복용약·알레르기"""

    __tablename__ = "pets"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    owner_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    name: Mapped[str] = mapped_column(String(50))
    species: Mapped[str] = mapped_column(String(20), default="강아지")  # 강아지 | 고양이
    breed: Mapped[str] = mapped_column(String(50), default="")
    birth_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    sex: Mapped[str] = mapped_column(String(10), default="수컷")  # 수컷 | 암컷
    is_neutered: Mapped[bool] = mapped_column(Boolean, default=False)
    weight_kg: Mapped[float | None] = mapped_column(Float, nullable=True)
    size_class: Mapped[str] = mapped_column(String(20), default="")  # 소형 | 중형 | 대형
    diseases: Mapped[str] = mapped_column(String(300), default="")  # 질병
    medications: Mapped[str] = mapped_column(String(300), default="")  # 복용약
    supplement: Mapped[str] = mapped_column(String(300), default="")  # 영양제
    allergies: Mapped[str] = mapped_column(String(300), default="")  # 알레르기
    # 프로필 수정 일시 — 생성 시 설정되고 프로필을 수정할 때마다 갱신된다.
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=utcnow, onupdate=utcnow
    )

    owner: Mapped["User"] = relationship(back_populates="pets")
    entries: Mapped[list["DailyEntry"]] = relationship(
        back_populates="pet", cascade="all, delete-orphan"
    )
    diagnoses: Mapped[list["Diagnosis"]] = relationship(
        back_populates="pet", cascade="all, delete-orphan"
    )
    # SQLite FK 미강제 환경에서 펫 삭제 시 고아 행이 남지 않도록 ORM cascade 로 정리
    summaries: Mapped[list["Summary"]] = relationship(cascade="all, delete-orphan")
    ai_sessions: Mapped[list["AISession"]] = relationship(cascade="all, delete-orphan")
    emergency_emails: Mapped[list["EmergencyEmail"]] = relationship(
        cascade="all, delete-orphan"
    )


class DailyEntry(Base):
    """일기장 DB (daily_entries) — 하루 한 기록.

    DB 스펙: record_date 를 PK 로, 반려동물별로 날짜당 오직 1개.
    식사·음수·활동·증상·배변·구토는 모두 텍스트 상태값이며, 원문은 raw_text 에 보관한다.
    (추이/기준선 같은 수치 분석은 저장하지 않고 AI 가 텍스트 기록으로 판단한다.)
    """

    __tablename__ = "daily_entries"

    # 복합 PK — (pet_id, record_date) 로 날짜당 1개를 보장 (다중 반려동물 지원)
    pet_id: Mapped[int] = mapped_column(
        ForeignKey("pets.id"), primary_key=True, index=True
    )
    record_date: Mapped[date] = mapped_column(Date, primary_key=True)
    raw_text: Mapped[str] = mapped_column(Text, default="")  # 사용자가 작성한 일기 원문

    food: Mapped[str] = mapped_column(String(200), default="")  # 식사 상태
    water: Mapped[str] = mapped_column(String(200), default="")  # 음수 상태
    activity: Mapped[str] = mapped_column(String(200), default="")  # 활동 상태
    symptom: Mapped[str] = mapped_column(String(300), default="")  # 증상
    stool: Mapped[str] = mapped_column(String(200), default="")  # 배변 및 설사 상태
    vomit: Mapped[str] = mapped_column(String(200), default="")  # 구토 상태
    notes: Mapped[str] = mapped_column(Text, default="")  # 기타사항

    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)

    pet: Mapped["Pet"] = relationship(back_populates="entries")


class Diagnosis(Base):
    """진단서 DB — 날짜, 병원, 진단명, 진단 내용(및 기타사항), 원본 파일"""

    __tablename__ = "diagnoses"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    pet_id: Mapped[int] = mapped_column(ForeignKey("pets.id"), index=True)
    date: Mapped[date | None] = mapped_column(Date, nullable=True)  # 발급일/진료일
    hospital: Mapped[str] = mapped_column(String(100), default="")  # 발급 병원
    diagnosis: Mapped[str] = mapped_column(String(300), default="")  # 진단명
    content: Mapped[str] = mapped_column(Text, default="")  # 진단 내용 및 기타사항
    original_file_ref: Mapped[str] = mapped_column(String(300), default="")  # 원본 파일명
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)

    pet: Mapped["Pet"] = relationship(back_populates="diagnoses")


class Hospital(Base):
    """응급 병원 안내용 병원 정보"""

    __tablename__ = "hospitals"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    name: Mapped[str] = mapped_column(String(100))
    phone: Mapped[str] = mapped_column(String(30), default="")
    email: Mapped[str] = mapped_column(String(100), default="")
    distance_km: Mapped[float | None] = mapped_column(Float, nullable=True)
    status: Mapped[str] = mapped_column(String(30), default="진료 중")
    features: Mapped[str] = mapped_column(String(200), default="")  # 응급실 운영 등
    is_emergency: Mapped[bool] = mapped_column(Boolean, default=True)
    open_24h: Mapped[bool] = mapped_column(Boolean, default=True)


class Summary(Base):
    """병원 전달용 요약"""

    __tablename__ = "summaries"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    pet_id: Mapped[int] = mapped_column(ForeignKey("pets.id"), index=True)
    risk_level: Mapped[str] = mapped_column(String(20), default="observe")  # observe | consult | emergency
    content: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


class AISession(Base):
    """AI 상태 체크 대화 세션 — 지난 대화 보기용"""

    __tablename__ = "ai_sessions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    pet_id: Mapped[int] = mapped_column(ForeignKey("pets.id"), index=True)
    title: Mapped[str] = mapped_column(String(120), default="")  # 첫 사용자 메시지 요약
    last_risk_level: Mapped[str] = mapped_column(String(20), default="observe")
    # [{role, content, meta?}] — assistant meta 에 위험도/근거 등 렌더링 정보 저장
    messages: Mapped[list] = mapped_column(JSON, default=list)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


class EmergencyEmail(Base):
    """응급 상태 문서 이메일 (전송 전 보호자 확인 필수 — 초안으로 생성)"""

    __tablename__ = "emergency_emails"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    pet_id: Mapped[int] = mapped_column(ForeignKey("pets.id"), index=True)
    hospital_id: Mapped[int | None] = mapped_column(ForeignKey("hospitals.id"), nullable=True)
    to_email: Mapped[str] = mapped_column(String(100), default="")
    subject: Mapped[str] = mapped_column(String(200), default="")
    body: Mapped[str] = mapped_column(Text, default="")
    content: Mapped[dict] = mapped_column(JSON, default=dict)  # 4섹션 요약 구조
    attachments: Mapped[list] = mapped_column(JSON, default=list)
    status: Mapped[str] = mapped_column(String(20), default="draft")  # draft | sent
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    sent_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
