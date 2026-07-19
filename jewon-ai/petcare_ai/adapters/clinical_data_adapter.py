"""임상 데이터 adapter — PET / 진단서 / 일기장 을 LangGraph 형식으로 공급한다.

명세 4절 원칙:
  - 기존 일기장·진단서 처리 코드를 새로 작성하거나 복제하지 않는다.
  - LangGraph 에는 얇은 adapter 로만 전달한다.
  - `USE_EXISTING_PROCESSORS = True` 인데 기존 코드 import 에 실패하면
    **조용히 fixture 로 넘어가지 않고 명확한 오류를 낸다.**
    (fixture 를 실제 데이터로 착각한 채 테스트가 통과하는 것이 가장 위험하다.)

반환 dict 의 키는 이 저장소 서버(`server/app/models.py`,
`server/app/services/context.py`)의 payload 와 1:1로 맞춰 두었다. 따라서 fixture 로
개발한 노드가 실제 서버 데이터에서도 그대로 동작한다.

  pet_profile : id, name, species, breed, birth_date, age_years, sex, is_neutered,
                weight_kg, diseases, medications, supplement, allergies
  daily_entry : record_date, raw_text, food, water, activity, symptom, stool, vomit, notes
  diagnosis   : date, hospital, diagnosis, content

`species` 는 서버 DB 의 한국어 값('강아지'/'고양이')이 아니라 **반드시 'dog'/'cat'
영문**으로 정규화해서 내보낸다. RAG index 가 species 별로 분리되어 있어(명세 11절)
이 값이 그대로 index 선택 키로 쓰이기 때문이다.
"""

from __future__ import annotations

import copy
import logging
import os
import sys
from datetime import date, timedelta
from typing import Any, Protocol, runtime_checkable

logger = logging.getLogger(__name__)

# 명세 4절 — 기본값은 "기존 처리 코드를 쓴다".
USE_EXISTING_PROCESSORS: bool = True

# 기존 처리 코드 import 경로 후보. 이 저장소의 실제 위치는
# `PetCare-AI/server/app` 이며, Colab 에는 존재하지 않는다.
# (환경 변수 PETCARE_SERVER_PATH 로 server 디렉터리를 지정하면 sys.path 에 추가한다.)
EXISTING_MODULE_CANDIDATES: tuple[str, ...] = (
    "app.services.context",
    "server.app.services.context",
    "PetCare-AI.server.app.services.context",
    "petcare_server.app.services.context",
    "services.context",
)

SERVER_PATH_ENV = "PETCARE_SERVER_PATH"

__all__ = [
    "USE_EXISTING_PROCESSORS",
    "EXISTING_MODULE_CANDIDATES",
    "ClinicalDataAdapter",
    "FixtureClinicalDataAdapter",
    "ExistingProcessorAdapter",
    "get_adapter",
    "load_pet_profile_for_test",
    "load_daily_entries_for_test",
    "load_diagnoses_for_test",
    "normalize_species",
]


# ---------------------------------------------------------------------------
# 공통 유틸
# ---------------------------------------------------------------------------
def normalize_species(raw: str | None) -> str:
    """서버 DB 의 한국어 종 값을 RAG index 키('dog'/'cat')로 정규화한다.

    알 수 없는 값은 'dog' 로 두되 경고를 남긴다 — 조용히 틀린 index 를 쓰면
    고양이에게 강아지 문서를 근거로 답하게 되므로 로그가 반드시 필요하다.
    """
    text = (raw or "").strip().lower()
    if text in {"dog", "canine", "강아지", "개", "견"}:
        return "dog"
    if text in {"cat", "feline", "고양이", "묘"}:
        return "cat"
    logger.warning("알 수 없는 species 값이라 'dog' 로 처리합니다: %r", raw)
    return "dog"


def _age_years(birth_date: date | None, today: date | None = None) -> int | None:
    """생년월일로 만 나이를 계산한다(생일 전이면 1살 뺀다)."""
    if birth_date is None:
        return None
    ref = today or date.today()
    return (
        ref.year
        - birth_date.year
        - ((ref.month, ref.day) < (birth_date.month, birth_date.day))
    )


def _years_ago(years: int, today: date | None = None) -> date:
    """N년 전 같은 날짜(2월 29일은 28일로 보정) — fixture 나이를 고정하기 위함."""
    ref = today or date.today()
    try:
        return ref.replace(year=ref.year - years)
    except ValueError:  # 2/29 → 2/28
        return ref.replace(year=ref.year - years, day=28)


# ---------------------------------------------------------------------------
# 인터페이스
# ---------------------------------------------------------------------------
@runtime_checkable
class ClinicalDataAdapter(Protocol):
    """LangGraph 가 임상 데이터를 읽는 유일한 통로(명세 4절)."""

    def load_pet_profile(self, pet_id: int) -> dict[str, Any]:
        """PET DB 레코드 1건을 반환한다."""
        ...

    def load_daily_entries(self, pet_id: int) -> list[dict[str, Any]]:
        """일기장 기록 전체를 record_date 오름차순으로 반환한다."""
        ...

    def load_diagnoses(self, pet_id: int) -> list[dict[str, Any]]:
        """진단서 기록 전체를 date 오름차순(마지막이 최신)으로 반환한다."""
        ...


# ---------------------------------------------------------------------------
# Fixture 구현 (테스트 전용)
# ---------------------------------------------------------------------------
class FixtureClinicalDataAdapter:
    """**테스트 전용** 고정 데이터 adapter — 실제 처리 코드가 아니다.

    명세 4절이 허용한 "기존 모듈 import 가 불가능할 때만 쓰는 동일 스키마 fixture"
    이며, 운영 경로에서 이 클래스가 쓰이면 안 된다. 데이터는 명세 43절 테스트
    시나리오를 그대로 재현하도록 설계했다.

      - pet 1 '초코'(dog / poodle / 8세): 심장질환 보유, 심장약 복용,
        최근 3일 식사·활동 감소 → 위험도 상승·RAG·PDF 시나리오용
      - pet 1 몸무게 충돌(명세 43절): **PET DB 5.2kg vs 최신 진단서 4.8kg**
        (여기에 사용자가 대화에서 말하는 4.5kg 이 더해져 3중 충돌이 된다)
      - pet 2 '나비'(cat / korean shorthair / 3세): 기저질환 없음
        → 종 분리(cat index 만 사용) 검증용

    날짜는 오늘 기준 상대값으로 만든다. 언제 돌려도 "최근 14일", "만 8세" 가
    유지되어야 테스트가 시간이 지나도 깨지지 않는다.
    """

    def __init__(self, today: date | None = None) -> None:
        """`today` 를 주입하면 날짜를 고정할 수 있다(스냅샷 테스트용)."""
        self._today: date = today or date.today()

    # -- PET ---------------------------------------------------------------
    def load_pet_profile(self, pet_id: int) -> dict[str, Any]:
        profile = self._pet_profiles().get(pet_id)
        if profile is None:
            raise ValueError(
                f"fixture 에 없는 pet_id 입니다: {pet_id} "
                f"(사용 가능: {sorted(self._pet_profiles())})"
            )
        return copy.deepcopy(profile)

    def load_daily_entries(self, pet_id: int) -> list[dict[str, Any]]:
        if pet_id not in self._pet_profiles():
            raise ValueError(
                f"fixture 에 없는 pet_id 입니다: {pet_id} "
                f"(사용 가능: {sorted(self._pet_profiles())})"
            )
        entries = self._daily_entries().get(pet_id, [])
        return copy.deepcopy(entries)

    def load_diagnoses(self, pet_id: int) -> list[dict[str, Any]]:
        if pet_id not in self._pet_profiles():
            raise ValueError(
                f"fixture 에 없는 pet_id 입니다: {pet_id} "
                f"(사용 가능: {sorted(self._pet_profiles())})"
            )
        diagnoses = self._diagnoses().get(pet_id, [])
        return copy.deepcopy(diagnoses)

    # -- fixture 정의 -------------------------------------------------------
    def _pet_profiles(self) -> dict[int, dict[str, Any]]:
        """PET DB fixture — 몸무게 충돌의 기준값(5.2kg)이 여기 들어 있다."""
        choco_birth = _years_ago(8, self._today)
        nabi_birth = _years_ago(3, self._today)
        return {
            1: {
                "id": 1,
                "name": "초코",
                "species": "dog",
                "breed": "poodle",
                "birth_date": choco_birth.isoformat(),
                "age_years": _age_years(choco_birth, self._today),
                "sex": "수컷",
                "is_neutered": True,
                # 명세 43절 충돌 fixture: PET DB 5.2 / 최신 진단서 4.8 / 사용자 4.5
                "weight_kg": 5.2,
                "diseases": "이첨판 폐쇄부전증(mitral valve disease), 심장 질환",
                "medications": "심장약(heart medication) 1일 2회",
                "supplement": "오메가3",
                "allergies": "",
            },
            2: {
                "id": 2,
                "name": "나비",
                "species": "cat",
                "breed": "korean shorthair",
                "birth_date": nabi_birth.isoformat(),
                "age_years": _age_years(nabi_birth, self._today),
                "sex": "암컷",
                "is_neutered": True,
                "weight_kg": 4.1,
                "diseases": "",
                "medications": "",
                "supplement": "",
                "allergies": "",
            },
        }

    def _diagnoses(self) -> dict[int, list[dict[str, Any]]]:
        """진단서 fixture — date 오름차순, 마지막이 최신.

        pet 1 의 **최신 진단서에만 체중 4.8kg** 을 기록한다. 오래된 진단서에도
        체중을 넣으면 "최신 진단서 값" 을 고르는 우선순위 로직을 검증할 수 없다.
        """
        d = self._today
        return {
            1: [
                {
                    "date": (d - timedelta(days=210)).isoformat(),
                    "hospital": "행복동물병원",
                    "diagnosis": "정기 건강검진",
                    "content": (
                        "연간 정기 건강검진. 혈액검사 및 흉부 방사선 촬영 시행. "
                        "청진에서 좌측 심첨부 수축기 심잡음(grade 2/6)이 새로 확인되어 "
                        "심장 초음파 추적 검사를 권고함. 그 외 특이 소견 없음."
                    ),
                },
                {
                    # 최신 진단서 — 몸무게 4.8kg (PET DB 5.2kg 와 충돌)
                    "date": (d - timedelta(days=32)).isoformat(),
                    "hospital": "서울동물메디컬센터",
                    "diagnosis": "이첨판 폐쇄부전증(mitral valve disease) ACVIM stage B2",
                    "content": (
                        "심장 초음파 및 흉부 방사선 재평가. 좌심방 확장 진행 소견. "
                        "ACVIM stage B2 로 진단하여 심장약 처방 시작(1일 2회). "
                        "내원 시 측정 체중 4.8kg. 기침·호흡수 증가·운동 불내성 발생 시 "
                        "즉시 재내원하도록 보호자 교육 시행. 4~6주 후 재검 예정."
                    ),
                },
            ],
            2: [
                {
                    "date": (d - timedelta(days=400)).isoformat(),
                    "hospital": "나래동물병원",
                    "diagnosis": "종합백신 3차 접종",
                    "content": "종합백신 3차 접종 완료. 접종 후 이상 반응 없음.",
                },
                {
                    "date": (d - timedelta(days=180)).isoformat(),
                    "hospital": "나래동물병원",
                    "diagnosis": "정기 건강검진",
                    "content": (
                        "혈액검사·신체검사 정상 범위. 기저질환 없음. "
                        "체중 유지 양호, 별도 처방 없음."
                    ),
                },
                {
                    "date": (d - timedelta(days=60)).isoformat(),
                    "hospital": "나래동물병원",
                    "diagnosis": "치석 제거(스케일링)",
                    "content": (
                        "경도 치은염 동반 치석 제거 시행. 발치 없음. "
                        "마취 회복 양호, 3일간 부드러운 사료 급여 권고."
                    ),
                },
            ],
        }

    def _daily_entries(self) -> dict[int, list[dict[str, Any]]]:
        """일기장 fixture — record_date 오름차순(서버 context.py 와 동일 정렬).

        pet 1 은 최근 14일치이며 **마지막 3일에 식사량·활동량 감소와 기침 악화**가
        나타나도록 구성했다. 심장질환 기왕력과 결합해 '병원 방문 권고' 분기가
        타지는지 검증하기 위한 추세다.
        """
        choco = self._build_choco_entries()
        nabi = self._build_nabi_entries()
        return {1: choco, 2: nabi}

    def _build_choco_entries(self) -> list[dict[str, Any]]:
        # (days_ago, food, water, activity, symptom, stool, vomit, notes)
        rows: list[tuple[int, str, str, str, str, str, str, str]] = [
            (13, "사료 한 그릇 완식", "평소만큼", "산책 30분, 활발", "", "정상 변", "없음", ""),
            (12, "사료 한 그릇 완식", "평소만큼", "산책 30분, 활발", "", "정상 변", "없음", ""),
            (11, "사료 한 그릇 완식", "평소만큼", "산책 25분", "", "정상 변", "없음", ""),
            (10, "사료 한 그릇 완식", "평소만큼", "산책 30분, 공놀이", "", "정상 변", "없음", ""),
            (9, "사료 대부분 먹음", "평소만큼", "산책 25분", "", "정상 변", "없음", ""),
            (8, "사료 한 그릇 완식", "평소만큼", "산책 30분", "", "정상 변", "없음", ""),
            (7, "사료 한 그릇 완식", "평소만큼", "산책 25분", "", "정상 변", "없음", "심장약 복용 잘 함"),
            (6, "사료 대부분 먹음", "평소만큼", "산책 25분", "", "정상 변", "없음", ""),
            (5, "사료 한 그릇 완식", "평소만큼", "산책 30분", "", "정상 변", "없음", ""),
            (4, "사료 대부분 먹음", "평소만큼", "산책 20분", "아침에 마른기침 1회", "정상 변", "없음", ""),
            (3, "사료 대부분 먹음", "평소만큼", "산책 20분", "", "정상 변", "없음", ""),
            # --- 최근 3일: 식사 감소 + 활동 감소 추세 ---
            (
                2,
                "사료 절반만 먹음",
                "평소보다 적게 마심",
                "산책 15분, 자주 멈춰 섬",
                "간헐적 기침",
                "정상 변",
                "없음",
                "저녁에 헥헥거림이 평소보다 심함",
            ),
            (
                1,
                "사료 1/3만 먹음",
                "적게 마심",
                "산책 10분, 금방 지침",
                "기침 잦아짐, 호흡이 빠름",
                "변이 무름",
                "없음",
                "밤에 기침 때문에 여러 번 깸",
            ),
            (
                0,
                "거의 먹지 않음",
                "물만 조금 마심",
                "산책을 거부하고 계속 누워 있음",
                "기침 지속, 호흡수 증가, 기운 없음",
                "변이 무름",
                "1회",
                "평소보다 마른 느낌이 들어 걱정됨",
            ),
        ]
        return [self._entry(*row) for row in rows]

    def _build_nabi_entries(self) -> list[dict[str, Any]]:
        rows: list[tuple[int, str, str, str, str, str, str, str]] = [
            (4, "습식 사료 완식", "평소만큼", "캣타워 오르내리며 활발", "", "정상 변", "없음", ""),
            (3, "습식 사료 완식", "평소만큼", "장난감 사냥놀이 20분", "", "정상 변", "없음", ""),
            (2, "건사료 대부분 먹음", "평소만큼", "평소처럼 활발", "", "정상 변", "없음", ""),
            (1, "습식 사료 완식", "평소만큼", "창가에서 오래 쉼", "", "정상 변", "없음", ""),
            (
                0,
                "건사료 대부분 먹음",
                "평소만큼",
                "평소처럼 활발",
                "",
                "정상 변",
                "헤어볼 1회 뱉음",
                "그루밍 후 헤어볼 토함, 이후 식사 정상",
            ),
        ]
        return [self._entry(*row) for row in rows]

    def _entry(
        self,
        days_ago: int,
        food: str,
        water: str,
        activity: str,
        symptom: str,
        stool: str,
        vomit: str,
        notes: str,
    ) -> dict[str, Any]:
        """일기장 1건을 서버 payload 스키마로 만든다(raw_text 는 원문 재구성)."""
        parts = [
            f"식사: {food}",
            f"음수: {water}",
            f"활동: {activity}",
            f"증상: {symptom or '특이사항 없음'}",
            f"배변: {stool}",
            f"구토: {vomit}",
        ]
        if notes:
            parts.append(f"기타: {notes}")
        return {
            "record_date": (self._today - timedelta(days=days_ago)).isoformat(),
            "raw_text": " / ".join(parts),
            "food": food,
            "water": water,
            "activity": activity,
            "symptom": symptom,
            "stool": stool,
            "vomit": vomit,
            "notes": notes,
        }


# ---------------------------------------------------------------------------
# 기존 처리 코드 연동 구현
# ---------------------------------------------------------------------------
class ExistingProcessorAdapter:
    """기존 앱 서버의 일기장·진단서 처리 코드를 그대로 호출하는 adapter.

    처리 로직을 복제하지 않고 `server/app/services/context.py` 의
    `pet_payload` / `get_entries_in_window` / 진단서 조회를 재사용한다.

    import 에 실패하면 **fixture 로 조용히 대체하지 않고 RuntimeError** 를 낸다
    (명세 4절). Colab 처럼 서버 코드가 없는 환경에서는 호출자가
    `get_adapter(use_existing=False)` 로 명시적으로 fixture 를 골라야 한다.
    """

    def __init__(self, session_factory: Any | None = None) -> None:
        """생성 시점에 즉시 import 를 시도한다(실패를 늦게 발견하지 않기 위함).

        `session_factory` 를 주입하면 그 세션을 쓰고, 없으면 서버의
        `app.database.SessionLocal` 을 찾아 쓴다.
        """
        self._context, self._models, root = _import_existing_modules()
        self._session_factory = session_factory or _resolve_session_factory(root)

    def load_pet_profile(self, pet_id: int) -> dict[str, Any]:
        with self._session() as db:
            pet = self._get_pet(db, pet_id)
            payload = dict(self._context.pet_payload(pet))
        # 서버는 '강아지'/'고양이' 로 저장한다 — RAG index 키로 정규화한다.
        payload["species"] = normalize_species(payload.get("species"))
        birth_raw = payload.get("birth_date")
        payload["age_years"] = _age_years(
            date.fromisoformat(birth_raw) if birth_raw else None
        )
        return payload

    def load_daily_entries(self, pet_id: int) -> list[dict[str, Any]]:
        with self._session() as db:
            pet = self._get_pet(db, pet_id)
            # window_days 를 크게 잡아 명세 21절대로 '일기장 전체' 를 State 에 싣는다.
            context = self._context.build_context(db, pet, window_days=3650)
            return list(context.get("records", []))

    def load_diagnoses(self, pet_id: int) -> list[dict[str, Any]]:
        with self._session() as db:
            pet = self._get_pet(db, pet_id)
            context = self._context.build_context(db, pet, window_days=3650)
            return list(context.get("diagnoses", []))

    # -- 내부 ---------------------------------------------------------------
    def _session(self) -> Any:
        if self._session_factory is None:
            raise RuntimeError(
                "기존 처리 코드용 DB 세션 팩토리를 찾지 못했습니다. "
                "ExistingProcessorAdapter(session_factory=...) 로 주입하세요."
            )
        return self._session_factory()

    def _get_pet(self, db: Any, pet_id: int) -> Any:
        pet = db.get(self._models.Pet, pet_id)
        if pet is None:
            raise ValueError(f"서버 DB 에 없는 pet_id 입니다: {pet_id}")
        return pet


def _package_root(context_path: str) -> str:
    """'a.b.services.context' → 'a.b' (앞에 패키지가 없으면 빈 문자열)."""
    suffix = "services.context"
    trimmed = context_path[: -len(suffix)].rstrip(".")
    return trimmed if context_path.endswith(suffix) else context_path


def _sibling_module(root: str, name: str) -> str:
    """패키지 root 아래의 형제 모듈 경로를 만든다(root 가 비면 최상위 모듈)."""
    return f"{root}.{name}" if root else name


def _import_existing_modules() -> tuple[Any, Any, str]:
    """기존 처리 코드(context, models)를 import 한다. 실패하면 RuntimeError.

    여러 경로 후보를 시도하는 이유: 이 코드는 Colab / 서버 리포 루트 / jewon-ai
    하위 등 서로 다른 작업 디렉터리에서 실행되며, 각각 패키지 경로가 다르다.
    성공한 패키지 root 를 함께 반환해 database 모듈도 같은 root 에서 찾는다.
    """
    _extend_sys_path_from_env()

    import importlib

    tried: list[str] = []
    for context_path in EXISTING_MODULE_CANDIDATES:
        root = _package_root(context_path)
        models_path = _sibling_module(root, "models")
        try:
            context_module = importlib.import_module(context_path)
            models_module = importlib.import_module(models_path)
        except Exception as exc:  # ImportError 외에 설정 오류도 잡는다.
            tried.append(f"  - {context_path} / {models_path}: {exc}")
            continue
        logger.info("기존 처리 코드 import 성공: %s", context_path)
        return context_module, models_module, root

    raise RuntimeError(
        "USE_EXISTING_PROCESSORS=True 이지만 기존 일기장·진단서 처리 코드를 "
        "import 하지 못했습니다. 명세 4절에 따라 fixture 로 조용히 대체하지 않고 "
        "중단합니다.\n"
        "시도한 경로:\n" + "\n".join(tried) + "\n"
        "해결 방법:\n"
        "  1) 이 저장소의 실제 위치는 'PetCare-AI/server/app' 입니다. "
        f"환경 변수 {SERVER_PATH_ENV} 에 server 디렉터리 경로를 지정하세요.\n"
        "  2) Colab 처럼 서버 코드가 없는 환경이라면 fixture 사용을 "
        "명시적으로 선택하세요: get_adapter(use_existing=False)"
    )


def _extend_sys_path_from_env() -> None:
    """PETCARE_SERVER_PATH 가 있으면 sys.path 에 추가한다(경로 하드코딩 금지)."""
    server_path = os.environ.get(SERVER_PATH_ENV)
    if server_path and server_path not in sys.path and os.path.isdir(server_path):
        sys.path.insert(0, server_path)
        logger.info("%s 를 sys.path 에 추가했습니다: %s", SERVER_PATH_ENV, server_path)


def _resolve_session_factory(root: str) -> Any | None:
    """context 를 찾은 것과 같은 패키지 root 에서 SessionLocal 을 찾는다.

    못 찾으면 None 을 돌려주고, 실제 조회 시점에 "주입하라" 는 오류를 낸다.
    """
    import importlib

    database_path = _sibling_module(root, "database")
    try:
        database_module = importlib.import_module(database_path)
    except Exception as exc:
        logger.warning("%s import 실패: %s", database_path, exc)
        return None
    factory = getattr(database_module, "SessionLocal", None)
    if factory is None:
        logger.warning(
            "%s 에 SessionLocal 이 없습니다 — session_factory 주입이 필요합니다.",
            database_path,
        )
    return factory


# ---------------------------------------------------------------------------
# 팩토리 / 명세 4절 호환 함수
# ---------------------------------------------------------------------------
def get_adapter(use_existing: bool = USE_EXISTING_PROCESSORS) -> ClinicalDataAdapter:
    """adapter 를 고른다 — 실패를 숨기지 않는 것이 이 함수의 핵심 계약이다.

    use_existing=True  : 기존 처리 코드 사용. import 실패 시 RuntimeError.
    use_existing=False : fixture 사용(테스트 전용). 경고 로그를 남긴다.
    """
    if use_existing:
        return ExistingProcessorAdapter()
    logger.warning(
        "fixture 임상 데이터를 사용합니다 — 테스트 전용이며 실제 처리 코드가 아닙니다."
    )
    return FixtureClinicalDataAdapter()


def load_pet_profile_for_test(pet_id: int) -> dict[str, Any]:
    """테스트용 PET DB 레코드를 반환한다(명세 4절 인터페이스)."""
    return FixtureClinicalDataAdapter().load_pet_profile(pet_id)


def load_daily_entries_for_test(pet_id: int) -> list[dict[str, Any]]:
    """테스트용 일기장 기록을 LangGraph 형식으로 반환한다(명세 4절 인터페이스)."""
    return FixtureClinicalDataAdapter().load_daily_entries(pet_id)


def load_diagnoses_for_test(pet_id: int) -> list[dict[str, Any]]:
    """테스트용 진단서 기록을 LangGraph 형식으로 반환한다(명세 4절 인터페이스)."""
    return FixtureClinicalDataAdapter().load_diagnoses(pet_id)
