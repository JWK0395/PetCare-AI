"""LLM 팩토리 — provider 별 지연 import 와 "키 없으면 None" 규약.

이 프로젝트의 모든 Agent 는 LLM 없이도(=API 키 없음, 패키지 미설치) 규칙 기반으로
동작해야 한다. 그래서 여기서는 키가 없거나 langchain provider 패키지가 설치되지
않았을 때 **예외를 던지지 않고 None 을 반환**한다. 호출자는 `if llm is None:` 한
줄로 규칙 기반 경로를 타면 되고, Colab(키 있음)과 로컬 테스트(키 없음)가 같은
코드로 돌아간다.

무거운 provider 패키지(langchain_anthropic / langchain_openai)는 모듈 최상단에서
import 하지 않는다. 최상단 import 는 그 패키지가 없는 환경에서 `petcare_ai` 전체
import 를 깨뜨리기 때문이다.
"""

from __future__ import annotations

import logging
from typing import Any, TypeVar

from pydantic import BaseModel

from .config import Settings, get_settings

logger = logging.getLogger(__name__)

TSchema = TypeVar("TSchema", bound=BaseModel)

__all__ = [
    "build_llm",
    "llm_available",
    "safe_structured_invoke",
]


def llm_available(settings: Settings | None = None) -> bool:
    """LLM 을 실제로 만들 수 있는 환경인지 확인한다.

    키 유무만 보고 판단한다(패키지 설치 여부는 실제 생성 시점에 확인).
    Agent 가 "LLM 경로 / 규칙 경로" 를 미리 분기하거나 로그를 남길 때 쓴다.
    """
    resolved = settings or get_settings()
    return bool(resolved.has_llm_key)


def build_llm(
    settings: Settings | None = None,
    structured_output_schema: type[BaseModel] | None = None,
) -> Any | None:
    """설정에 맞는 LangChain chat model 을 만든다. 만들 수 없으면 None.

    None 을 반환하는 경우(모두 정상 동작이며 예외가 아니다):
      - provider 에 해당하는 API 키가 환경 변수에 없음
      - langchain_anthropic / langchain_openai 패키지 미설치
      - 알 수 없는 provider 값

    `structured_output_schema` 를 주면 `.with_structured_output()` 을 적용해
    Pydantic 인스턴스를 직접 받도록 한다. 재시도는 구조화 출력 적용 **전에**
    걸어 네트워크·rate limit 오류만 재시도하도록 한다.
    """
    resolved = settings or get_settings()
    provider = resolved.llm_provider

    if provider == "anthropic":
        api_key = resolved.anthropic_api_key
        if not api_key:
            logger.info("ANTHROPIC_API_KEY 가 없어 LLM 없이 규칙 기반으로 동작합니다.")
            return None
        try:
            from langchain_anthropic import ChatAnthropic  # 지연 import
        except ImportError:
            logger.warning(
                "langchain-anthropic 이 설치되어 있지 않아 LLM 없이 규칙 기반으로 동작합니다."
            )
            return None
        try:
            llm: Any = ChatAnthropic(
                model=resolved.anthropic_model,
                temperature=resolved.llm_temperature,
                timeout=resolved.llm_timeout_seconds,
                max_retries=0,  # 재시도는 아래 with_retry 로 일원화한다.
                api_key=api_key,
            )
        except Exception as exc:  # 잘못된 모델명·환경 문제로 죽지 않는다.
            logger.warning("ChatAnthropic 생성 실패 — 규칙 기반으로 동작합니다: %s", exc)
            return None

    elif provider == "openai":
        api_key = resolved.openai_api_key
        if not api_key:
            logger.info("OPENAI_API_KEY 가 없어 LLM 없이 규칙 기반으로 동작합니다.")
            return None
        try:
            from langchain_openai import ChatOpenAI  # 지연 import
        except ImportError:
            logger.warning(
                "langchain-openai 가 설치되어 있지 않아 LLM 없이 규칙 기반으로 동작합니다."
            )
            return None
        try:
            llm = ChatOpenAI(
                model=resolved.openai_model,
                temperature=resolved.llm_temperature,
                timeout=resolved.llm_timeout_seconds,
                max_retries=0,
                api_key=api_key,
            )
        except Exception as exc:
            logger.warning("ChatOpenAI 생성 실패 — 규칙 기반으로 동작합니다: %s", exc)
            return None

    else:  # Literal 을 벗어난 값이 configure() 로 들어온 경우
        logger.warning("알 수 없는 llm_provider 입니다: %r", provider)
        return None

    base_model = llm  # 래핑 전 원본 모델(구조화 출력 메서드를 가진 객체)

    # 순서가 중요하다: **structured output 을 먼저, retry 를 나중에** 적용한다.
    # with_retry() 는 모델을 RunnableRetry 로 감싸는데, RunnableRetry 에는
    # with_structured_output() 이 없다(그건 BaseChatModel 의 메서드다).
    # 먼저 retry 를 걸면 이후 구조화 출력 적용이 조용히 건너뛰어지고,
    # 노드는 schema 대신 AIMessage 를 받아 검증에 실패한 뒤 **전부 규칙 기반으로
    # 떨어진다**(로그에만 "스키마와 맞지 않아 기본값을 사용" 이 남아 발견이 늦다).
    if structured_output_schema is not None:
        llm = _apply_structured_output(llm, structured_output_schema)

    llm = _apply_retry(llm, resolved.llm_max_retries)

    # schema 없이 만든 경우에도 호출자가 나중에 with_structured_output 을 쓸 수 있도록
    # 원본 모델을 남겨 둔다(_apply_structured_output 이 이 참조를 사용한다).
    if structured_output_schema is None:
        try:
            llm._petcare_base_model = base_model  # type: ignore[attr-defined]
        except Exception:  # pragma: no cover - 속성 설정 불가한 래퍼
            pass

    return llm


def safe_structured_invoke(
    llm: Any | None,
    messages: Any,
    schema: type[TSchema],
    default: TSchema,
) -> TSchema:
    """LLM 구조화 호출을 감싸서 **절대 예외를 밖으로 내보내지 않는다**.

    LLM 은 품질 향상 옵션일 뿐이므로, 타임아웃·rate limit·스키마 위반 응답이
    사용자 답변 전체를 실패시키면 안 된다. 실패하면 호출자가 규칙 기반으로 계산해
    넘긴 `default` 를 그대로 돌려주고 경고만 남긴다.

    llm 이 이미 `with_structured_output` 이 적용된 객체여도 안전하게 동작한다
    (반환값이 schema 인스턴스면 그대로, dict 면 검증해서 변환).
    """
    if llm is None:
        return default

    try:
        runnable = _apply_structured_output(llm, schema)
        result = runnable.invoke(messages)
    except Exception as exc:
        logger.warning("LLM 구조화 호출 실패 — 규칙 기반 기본값을 사용합니다: %s", exc)
        return default

    return _coerce_to_schema(result, schema, default)


# ---------------------------------------------------------------------------
# 내부 헬퍼
# ---------------------------------------------------------------------------
def _apply_retry(llm: Any, max_retries: int) -> Any:
    """LangChain Runnable 의 with_retry 를 적용한다(미지원 객체면 원본 반환).

    테스트 mock 처럼 with_retry 가 없는 객체도 주입될 수 있으므로 방어한다.
    """
    if max_retries <= 0 or not hasattr(llm, "with_retry"):
        return llm
    try:
        return llm.with_retry(stop_after_attempt=max_retries + 1)
    except Exception as exc:
        logger.debug("with_retry 적용 실패 — 재시도 없이 사용합니다: %s", exc)
        return llm


def _apply_structured_output(llm: Any, schema: type[BaseModel]) -> Any:
    """with_structured_output 을 적용한다(미지원 객체면 원본 반환).

    mock LLM 이 이미 schema 인스턴스를 돌려주는 경우를 위해 실패해도 죽지 않는다.
    """
    target = llm
    if not hasattr(target, "with_structured_output"):
        # retry 등으로 래핑돼 메서드가 가려진 경우 원본 모델을 찾는다.
        base = getattr(llm, "_petcare_base_model", None)
        if base is not None and hasattr(base, "with_structured_output"):
            target = base
        else:
            logger.debug(
                "with_structured_output 을 지원하지 않는 객체입니다(%s) — 원본을 사용합니다.",
                type(llm).__name__,
            )
            return llm
    try:
        return target.with_structured_output(schema)
    except Exception as exc:
        logger.debug("with_structured_output 적용 실패 — 원본 LLM 을 사용합니다: %s", exc)
        return llm


def _coerce_to_schema(
    result: Any,
    schema: type[TSchema],
    default: TSchema,
) -> TSchema:
    """LLM 반환값을 schema 인스턴스로 정규화한다. 불가능하면 default."""
    if isinstance(result, schema):
        return result
    if isinstance(result, BaseModel):
        payload: Any = result.model_dump()
    elif isinstance(result, dict):
        payload = result
    else:
        logger.warning(
            "LLM 응답 형식을 해석할 수 없어 기본값을 사용합니다: %s", type(result).__name__
        )
        return default

    try:
        return schema.model_validate(payload)
    except Exception as exc:
        logger.warning("LLM 응답이 스키마를 만족하지 않아 기본값을 사용합니다: %s", exc)
        return default
