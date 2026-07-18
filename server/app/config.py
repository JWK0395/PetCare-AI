"""서버 설정 — 환경 변수를 한 곳에서 읽고 검증한다.

모든 코드는 os.environ 대신 이 모듈의 `settings` 객체를 사용한다.
값은 server/.env(.env.example 참고)에서 읽으며, 시작 시 형식을 검증해
잘못된 설정이면 서버가 즉시 명확한 오류로 실패한다.
"""

import sys
from pathlib import Path
from typing import Literal

from pydantic import ValidationError, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env", env_file_encoding="utf-8", extra="ignore"
    )

    # --- Database -----------------------------------------------------------
    # 기본값: server/data/petcare.db (절대경로 — 실행 위치와 무관하게 동일)
    database_url: str = f"sqlite:///{(DATA_DIR / 'petcare.db').as_posix()}"

    # --- AI Agent 연결 -------------------------------------------------------
    # mock : 내장 규칙 기반 응답 (외부 Agent 불필요)
    # http : ai/ 서비스로 전달 (계약: ai/README.md)
    agent_mode: Literal["mock", "http"] = "mock"
    agent_base_url: str = "http://127.0.0.1:8100"
    agent_timeout_seconds: float = 60.0
    agent_api_key: str = ""

    # --- CORS ----------------------------------------------------------------
    # 쉼표로 구분한 허용 origin 목록. "*" 는 모든 origin (credentials 미사용 전제).
    # RN 앱은 CORS 를 사용하지 않으므로 로컬 개발 기본값은 "*" 로 둔다.
    cors_origins: str = "*"

    # --- 업로드 ---------------------------------------------------------------
    max_upload_mb: int = 10
    allowed_upload_extensions: str = ".pdf,.png,.jpg,.jpeg"

    # --- 인증 -----------------------------------------------------------------
    # 로그인/회원가입 시도 제한 (계정+IP 기준, 창 단위)
    auth_rate_limit_attempts: int = 10
    auth_rate_limit_window_seconds: int = 300

    # --- 데모 데이터 -----------------------------------------------------------
    seed_demo_data: bool = True
    demo_user_email: str = "demo@petcare.ai"
    demo_user_password: str = "demo1234"

    # 데모 비밀번호 — mock 모드에서 하드코딩된 디자인 예시 응답을 "잠금 해제"하는 키워드.
    # 입력(일기/증상)에 이 값이 포함될 때만 예시 결과가 나온다.
    # 빈 문자열이면 항상 예시 응답(하위 호환). 실제 AI(agent_mode=http) 사용 시에는 무시된다.
    demo_password: str = "demo"

    # --- 검증 -----------------------------------------------------------------
    @field_validator("database_url")
    @classmethod
    def _database_url_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("DATABASE_URL 이 비어 있습니다")
        return v

    @field_validator("agent_base_url")
    @classmethod
    def _agent_base_url_format(cls, v: str) -> str:
        if v and not v.startswith(("http://", "https://")):
            raise ValueError("AGENT_BASE_URL 은 http:// 또는 https:// 로 시작해야 합니다")
        return v.rstrip("/")

    @field_validator("agent_timeout_seconds")
    @classmethod
    def _timeout_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("AGENT_TIMEOUT_SECONDS 는 0보다 커야 합니다")
        return v

    @field_validator("max_upload_mb")
    @classmethod
    def _upload_positive(cls, v: int) -> int:
        if not 1 <= v <= 100:
            raise ValueError("MAX_UPLOAD_MB 는 1~100 사이여야 합니다")
        return v

    @model_validator(mode="after")
    def _http_mode_requires_base_url(self) -> "Settings":
        if self.agent_mode == "http" and not self.agent_base_url:
            raise ValueError("AGENT_MODE=http 이면 AGENT_BASE_URL 이 필요합니다")
        if self.seed_demo_data and len(self.demo_user_password) < 4:
            raise ValueError("DEMO_USER_PASSWORD 는 4자 이상이어야 합니다")
        return self

    # --- 파생 값 ---------------------------------------------------------------
    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def allowed_extensions(self) -> set[str]:
        return {
            e.strip().lower() if e.strip().startswith(".") else f".{e.strip().lower()}"
            for e in self.allowed_upload_extensions.split(",")
            if e.strip()
        }

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024


try:
    settings = Settings()
except ValidationError as exc:
    # pydantic 기본 출력은 input_value= 로 설정값(비밀값 포함)을 에코하므로,
    # 변수명과 사유만 출력하고 종료한다. (환경 변수 문서의 "비밀값 미출력" 보장)
    _lines = [
        f"- {'.'.join(str(p) for p in err['loc']) or '(모델 검증)'}: {err['msg']}"
        for err in exc.errors(include_url=False, include_input=False)
    ]
    sys.stderr.write(
        "환경 변수 설정 오류 — 서버를 시작할 수 없습니다:\n" + "\n".join(_lines) + "\n"
    )
    raise SystemExit(1)

DATA_DIR.mkdir(parents=True, exist_ok=True)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
