"""데모 시드 데이터 — 디자인 시안(콩이)과 동일한 상태를 재현한다.

- 데모 계정: demo@petcare.ai / demo1234 — .env 의 DEMO_USER_EMAIL / DEMO_USER_PASSWORD 로 변경 가능
- 콩이: 말티즈, 2021-09-14생, 수컷(중성화), 5.08kg, 슬개골 탈구 2기
- 최근 30일 일일 기록(daily_entries): 평소엔 사료를 잘 먹고 산책 30분대,
  마지막 3일은 식사 감소 + 활동 감소, 어제 노란 구토 1회 (모두 텍스트 상태값)
- 병원: 24시 온누리동물의료센터 · 센트럴동물응급의료센터
- 진단서: 행복한동물병원 · 슬개골 탈구 2기
"""

from datetime import date, timedelta

from sqlalchemy import select
from sqlalchemy.orm import Session

from .. import models
from ..config import settings
from ..routers.auth import hash_password


def seed_if_empty(db: Session) -> None:
    if db.scalar(select(models.Pet).limit(1)):
        return

    today = date.today()

    # 데모 계정이 이미 있으면 재사용 — 펫이 모두 삭제된 DB 에서 재시작할 때
    # UNIQUE(email) 위반으로 서버 시작이 실패하지 않게 한다 (시드 재실행 안전).
    demo_user = db.scalar(
        select(models.User).where(models.User.email == settings.demo_user_email)
    )
    if demo_user is None:
        demo_user = models.User(
            email=settings.demo_user_email,
            password_hash=hash_password(settings.demo_user_password),
        )
        db.add(demo_user)
        db.flush()

    pet = models.Pet(
        owner_id=demo_user.id,
        name="콩이",
        species="강아지",
        breed="말티즈 · 순종",
        birth_date=date(2021, 9, 14),
        sex="수컷",
        is_neutered=True,
        weight_kg=5.08,
        size_class="소형",
        diseases="슬개골 탈구 2기",
        medications="",
        supplement="관절 영양제 1일 1회",
        allergies="닭고기 알레르기",
    )
    db.add(pet)
    db.flush()

    # ---- 최근 30일 기록 (텍스트 상태값) ----
    for i in range(29, -1, -1):
        d = today - timedelta(days=i)
        if i >= 3:
            # 평소: 사료 잘 먹고 산책 30분대, 특이 증상 없음
            entry = models.DailyEntry(
                pet_id=pet.id,
                record_date=d,
                raw_text="",
                food="사료 잘 먹음",
                water="정상 범위",
                activity="산책 30분",
                symptom="",
                stool="정상",
                vomit="없음",
                notes="",
            )
        else:
            # 마지막 3일: 식사 감소 + 활동 감소, 어제 노란 구토 1회
            vomit_day = i == 1  # 어제
            entry = models.DailyEntry(
                pet_id=pet.id,
                record_date=d,
                raw_text=(
                    "아침에 사료를 반쯤 남겼다. 산책은 20분 정도 했는데 평소보다 걷기 싫어하는 느낌. "
                    "물은 잘 마셨고, 오후에 노란 토를 한 번 했다."
                    if vomit_day
                    else "밥을 평소보다 많이 남겼다. 기운이 없어 보인다."
                ),
                food="사료 반쯤 남김 · 평소보다 감소",
                water="정상 범위" if i != 2 else "",  # 그저께 음수 기록 누락
                activity="산책 20분 · 평소보다 짧음",
                symptom="기력 저하" if not vomit_day else "기력 저하 · 구토",
                stool="정상",
                vomit="노란색 구토 1회 · 오후" if vomit_day else "없음",
                notes="",
            )
        db.add(entry)

    # ---- 진단서 ----
    db.add(
        models.Diagnosis(
            pet_id=pet.id,
            date=date(today.year, 7, 2),
            hospital="행복한동물병원",
            diagnosis="슬개골 탈구 2기",
            content=(
                "무릎뼈(슬개골)가 안쪽으로 탈구되는 2기 상태로, 촉진 시 정복과 재탈구가 "
                "반복됨. 간헐적 뒷다리 들기·보행 이상 관찰. 급성 통증 소견은 없음. "
                "처방: 관절 영양제 1일 1회 · 30일. 체중 5.28kg. 정기 재검 권장(4주 후)."
            ),
            original_file_ref="진단서_행복한동물병원_0702.pdf",
        )
    )

    # ---- 응급 병원 ----
    db.add_all(
        [
            models.Hospital(
                name="24시 온누리동물의료센터",
                phone="02-1234-5678",
                email="er@onnuri-amc.kr",
                distance_km=1.2,
                status="진료 중",
                features="응급실 운영",
                is_emergency=True,
                open_24h=True,
            ),
            models.Hospital(
                name="센트럴동물응급의료센터",
                phone="02-8765-4321",
                email="emergency@central-amc.kr",
                distance_km=2.8,
                status="진료 중",
                features="야간 수술 가능",
                is_emergency=True,
                open_24h=True,
            ),
        ]
    )

    db.commit()
