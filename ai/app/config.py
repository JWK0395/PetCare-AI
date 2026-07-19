"""AI Agent 서비스 설정 — 환경 변수를 읽어 두 소비자에게 나눠 준다.

이 모듈이 하는 일은 두 가지뿐이다.

  1) `settings.agent_api_key` — 메인 서버가 보내는 Bearer 토큰 검증값(main.py).
  2) `load_provider_env()`   — ai/.env 의 키를 os.environ 으로 올린다. petcare_ai 가
     API 키·모델·index 경로를 os.environ 에서 **직접** 읽기 때문이다.

LLM 자체의 설정(provider/model/timeout)은 여기가 아니라 petcare_ai/config.py 가
갖는다. 한때 이 파일에도 같은 필드와 require_llm() 검증이 있었지만, 실제 LLM 생성은
petcare_ai/llm.py 가 os.environ 을 보고 하므로 한 번도 호출되지 않는 껍데기였다.
설정이 두 군데 있으면 어느 쪽이 실제로 쓰이는지 알 수 없어 전부 지웠다.

비밀값은 ai/.env 에만 두고 절대 커밋하지 않는다(루트 .gitignore 가 **/.env 차단).
"""

import logging
import os
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent

#: petcare_ai 가 **os.environ 에서 직접 읽는** 비밀값/설정 이름.
#: pydantic-settings 는 .env 를 Settings 필드로만 읽고 os.environ 에는 넣지 않는다.
#: 그래서 ai/.env 에 OPENAI_API_KEY 를 적어도 petcare_ai(llm.py / config.py)는
#: 보지 못한다 — 아래 load_provider_env() 가 그 간극을 메운다.
PROVIDER_ENV_KEYS: tuple[str, ...] = (
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "TAVILY_API_KEY",
    "LLM_PROVIDER",
    "LLM_MODEL",
    "EMBEDDING_BACKEND",
    "PETCARE_INDEX_DIR",
    "PETCARE_AI_PATH",
    "LANGSMITH_TRACING",
    "LANGSMITH_API_KEY",
    "LANGSMITH_PROJECT",
    "LANGSMITH_ENDPOINT",
)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env", env_file_encoding="utf-8", extra="ignore"
    )

    # 메인 서버의 AGENT_API_KEY 와 같은 값을 넣으면 Bearer 검증이 켜진다.
    # 비워 두면 검증 없이 통과한다 — 로컬 개발에서만 비워 둘 것(main.py 참고).
    agent_api_key: str = ""


settings = Settings()


_provider_env_loaded = False


def load_provider_env() -> None:
    """ai/.env 의 provider 키를 os.environ 으로 올린다 (1회만 수행).

    petcare_ai 는 API 키를 `os.environ.get("OPENAI_API_KEY")` 로 직접 읽는다
    (petcare_ai/config.py). 반면 pydantic-settings 는 .env 를 **Settings 필드로만**
    읽어 os.environ 을 건드리지 않으므로, 이 다리를 놓지 않으면 ai/.env 에 키를
    적어도 LLM 이 계속 None 으로 동작한다.

    OS 환경변수가 이미 있으면 덮어쓰지 않는다 — 컨테이너/systemd 가 주입한 값이
    파일보다 우선이어야 하기 때문이다. 값은 로그에 남기지 않는다.
    """
    global _provider_env_loaded
    if _provider_env_loaded:
        return
    _provider_env_loaded = True

    env_path = BASE_DIR / ".env"
    if not env_path.is_file():
        return
    try:
        from dotenv import dotenv_values  # 지연 import — pydantic-settings 의존성
    except ImportError:  # pragma: no cover - python-dotenv 미설치 환경
        logger.debug("python-dotenv 가 없어 ai/.env 의 provider 키를 올리지 못했습니다.")
        return

    values = dotenv_values(env_path) or {}

    loaded: list[str] = []
    for key, value in values.items():
        if key in PROVIDER_ENV_KEYS and value and not os.environ.get(key):
            os.environ[key] = value
            loaded.append(key)

    # LLM_API_KEY 별칭 처리.
    # 초기 스텁의 .env.example 이 provider 중립 이름인 LLM_API_KEY 를 안내했고
    # Settings.llm_api_key / require_llm() 도 그 이름을 그대로 쓴다. 그런데
    # petcare_ai 는 provider 별 표준 이름(OPENAI_API_KEY)을 os.environ 에서 직접 읽는다.
    # 다리를 놓지 않으면 사용자가 LLM_API_KEY 를 채워도 require_llm() 은 통과하는데
    # build_llm() 은 None 을 돌려줘 **조용히 규칙 기반으로 떨어진다** — 원인을 찾기 매우 어렵다.
    # 그래서 provider 에 맞는 표준 키가 비어 있을 때만 별칭을 승격시킨다.
    alias_value = values.get("LLM_API_KEY") or os.environ.get("LLM_API_KEY")
    if alias_value:
        provider = (
            values.get("LLM_PROVIDER") or os.environ.get("LLM_PROVIDER") or "openai"
        ).strip().lower()
        target = "ANTHROPIC_API_KEY" if provider == "anthropic" else "OPENAI_API_KEY"
        if not os.environ.get(target):
            os.environ[target] = alias_value
            loaded.append(f"LLM_API_KEY→{target}")

    if loaded:
        logger.info("ai/.env 에서 환경 변수를 적용했습니다: %s", ", ".join(sorted(loaded)))
