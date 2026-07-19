"""임베딩 백엔드 팩토리 — 명세 11절.

Cornell 문서는 영어지만 사용자 질문은 한국어로 들어온다. 따라서 운영 백엔드는
multilingual 모델(`BAAI/bge-m3`)을 쓰되, 모델 이름·정규화·device·batch_size 는
전부 `RagSettings` 로만 관리한다(코드에 산재 금지).

동시에 오프라인 단위테스트가 무거운 모델 다운로드 없이 돌아가야 하므로,
외부 의존이 전혀 없는 해시 기반 `DeterministicEmbeddings` 백엔드를 함께 제공한다.
`build_embeddings()` 는 두 백엔드를 같은 인터페이스(LangChain Embeddings 호환)로
돌려주므로 vector store 는 어느 쪽이 들어왔는지 몰라도 된다.
"""

from __future__ import annotations

import hashlib
import logging
import re
from typing import Any, Iterable, Sequence

from ..config import RagSettings, Settings, get_settings

logger = logging.getLogger(__name__)

#: DeterministicEmbeddings 의 고정 차원. 테스트 index 호환성을 위해 바꾸지 않는다.
DETERMINISTIC_DIM: int = 384

#: 단어 토큰 추출 정규식 — 한글/영문/숫자를 모두 하나의 토큰으로 본다.
_WORD_RE = re.compile(r"[0-9A-Za-z가-힣ㄱ-ㆎ]+", re.UNICODE)

#: 문자 n-gram 길이. 어미 변화가 심한 한국어에서 부분 일치를 잡아주기 위해 쓴다.
_CHAR_NGRAM: int = 3

# 단어 토큰이 부분 일치보다 강하게 반영되도록 가중치를 나눈다.
_WORD_WEIGHT: float = 1.0
_CHAR_WEIGHT: float = 0.5


# ---------------------------------------------------------------------------
# 설정 정규화
# ---------------------------------------------------------------------------
def _as_rag_settings(settings: RagSettings | Settings | None) -> RagSettings:
    """`RagSettings` / `Settings` / None 을 모두 받아 `RagSettings` 로 통일한다.

    호출자가 전체 `Settings` 를 그대로 넘겨도 깨지지 않게 하기 위한 방어 코드다.
    시그니처상 기대 타입은 `RagSettings` 지만, 실수로 상위 객체가 들어오는 상황이
    파이프라인에서 흔하므로 조용히 흡수한다.
    """
    if settings is None:
        return get_settings().rag
    rag = getattr(settings, "rag", None)
    if isinstance(rag, RagSettings):
        return rag
    if isinstance(settings, RagSettings):
        return settings
    # dataclass 가 아닌 임의 객체라도 필요한 속성만 있으면 그대로 쓴다.
    return settings  # type: ignore[return-value]


def detect_device(preferred: str | None = None) -> str:
    """임베딩 실행 device 를 결정한다 — 설정값 우선, 없으면 cuda > cpu 자동 감지.

    torch 는 지연 import 한다. Colab 이 아닌 로컬/CI 에는 torch 가 없을 수 있는데,
    그때 임베딩 팩토리 전체가 죽으면 안 되므로 어떤 예외가 나도 "cpu" 로 진행한다.
    """
    if preferred:
        return preferred
    try:
        import torch  # noqa: PLC0415  (지연 import: torch 미설치 환경 보호)

        if torch.cuda.is_available():
            return "cuda"
    except Exception as exc:  # pragma: no cover - 환경 의존
        logger.debug("torch device 감지 실패 — cpu 로 진행합니다: %s", exc)
    return "cpu"


# ---------------------------------------------------------------------------
# 오프라인 결정적 임베딩
# ---------------------------------------------------------------------------
class DeterministicEmbeddings:
    """외부 의존 없는 해시 기반 임베딩 — 오프라인 단위테스트 전용.

    signed hashing trick 을 쓴다. 텍스트를 단어 토큰과 문자 3-gram 으로 쪼갠 뒤
    각 토큰의 SHA-256 다이제스트에서 (차원 index, 부호) 를 뽑아 누적한다.
    부호를 섞기 때문에 해시 충돌이 한쪽으로 편향되지 않고, 어휘가 겹치는 두 문장은
    같은 차원을 공유하므로 코사인 유사도가 실제로 의미를 갖는다
    (= build→search 왕복 테스트가 성립한다).

    보장:
      - 같은 입력 → 항상 같은 출력(파이썬 `hash()` 의 실행별 salt 를 쓰지 않는다).
      - 항상 `DETERMINISTIC_DIM`(384) 차원 float 벡터.
      - `normalize=True` 면 L2 정규화되어 내적 == 코사인 유사도.

    주의: 의미(semantic) 유사도가 아니라 어휘 중첩 기반이다. 품질 평가용이 아니라
    파이프라인 배선 검증용이다.
    """

    def __init__(self, dim: int = DETERMINISTIC_DIM, normalize: bool = True) -> None:
        if dim <= 0:
            raise ValueError("임베딩 차원은 1 이상이어야 합니다.")
        self.dim: int = dim
        self.normalize: bool = normalize

    # -- 내부 유틸 ---------------------------------------------------------
    def _tokens(self, text: str) -> Iterable[tuple[str, float]]:
        """(토큰, 가중치) 스트림을 만든다 — 단어 토큰 + 문자 n-gram."""
        lowered = " ".join(text.lower().split())
        for word in _WORD_RE.findall(lowered):
            yield f"w:{word}", _WORD_WEIGHT
        if len(lowered) >= _CHAR_NGRAM:
            for i in range(len(lowered) - _CHAR_NGRAM + 1):
                yield f"c:{lowered[i : i + _CHAR_NGRAM]}", _CHAR_WEIGHT

    def _vector(self, text: str) -> list[float]:
        """텍스트 1건을 결정적 벡터로 변환한다."""
        vector = [0.0] * self.dim
        for token, weight in self._tokens(text or ""):
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self.dim
            sign = 1.0 if digest[4] & 1 else -1.0
            vector[index] += sign * weight

        if not self.normalize:
            return vector

        norm = sum(value * value for value in vector) ** 0.5
        if norm == 0.0:
            # 빈 문자열/토큰 없음. 0 벡터는 코사인 계산에서 0 나눗셈을 일으키므로
            # 결정적인 단위 벡터(첫 차원 1.0)로 대체한다.
            vector[0] = 1.0
            return vector
        return [value / norm for value in vector]

    # -- LangChain Embeddings 호환 인터페이스 ------------------------------
    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """문서 목록을 임베딩한다."""
        return [self._vector(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        """질의 1건을 임베딩한다. 같은 문자열이면 문서 임베딩과 동일한 벡터다."""
        return self._vector(text)

    def __repr__(self) -> str:  # pragma: no cover - 디버깅 편의
        return f"DeterministicEmbeddings(dim={self.dim}, normalize={self.normalize})"


# ---------------------------------------------------------------------------
# HuggingFace 백엔드
# ---------------------------------------------------------------------------
def _import_hf_embeddings() -> Any:
    """HuggingFaceEmbeddings 클래스를 지연 import 한다.

    `langchain_huggingface` 가 정식 경로이고, 구버전 환경을 위해
    `langchain_community` 경로도 시도한다. 둘 다 없으면 설치 방법을 한국어로 안내한다.
    """
    try:
        from langchain_huggingface import HuggingFaceEmbeddings  # noqa: PLC0415

        return HuggingFaceEmbeddings
    except ImportError as primary_exc:
        try:
            from langchain_community.embeddings import (  # noqa: PLC0415
                HuggingFaceEmbeddings,
            )

            logger.warning(
                "langchain_huggingface 가 없어 langchain_community 경로로 대체합니다."
            )
            return HuggingFaceEmbeddings
        except ImportError as fallback_exc:
            raise ImportError(
                "HuggingFace 임베딩을 쓰려면 langchain-huggingface 와 "
                "sentence-transformers 가 필요합니다. "
                "`pip install langchain-huggingface sentence-transformers` 로 설치하거나, "
                'RagSettings.embedding_backend 를 "deterministic" 으로 바꿔 '
                f"오프라인 모드로 실행하세요. (원인: {primary_exc} / {fallback_exc})"
            ) from fallback_exc


def _build_huggingface_embeddings(rag: RagSettings) -> Any:
    """설정대로 HuggingFace 임베딩을 만든다 — 실패 시 fallback 모델로 1회 재시도.

    bge-m3 는 Colab 무료 런타임에서 메모리 초과가 나기 쉬워, 명세 11절이 요구한 대로
    더 가벼운 `embedding_fallback_model` 로 한 번 더 시도한 뒤에야 포기한다.
    """
    embeddings_cls = _import_hf_embeddings()
    device = detect_device(rag.embedding_device)

    model_kwargs: dict[str, Any] = {"device": device}
    encode_kwargs: dict[str, Any] = {
        "normalize_embeddings": bool(rag.embedding_normalize),
        "batch_size": int(rag.embedding_batch_size),
    }

    primary = rag.embedding_model
    fallback = rag.embedding_fallback_model

    try:
        return embeddings_cls(
            model_name=primary,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs,
        )
    except Exception as primary_exc:
        logger.warning(
            "임베딩 모델 로드 실패(%s) — fallback 모델 %s 로 재시도합니다: %s",
            primary,
            fallback,
            primary_exc,
        )
        if not fallback or fallback == primary:
            raise RuntimeError(
                f"임베딩 모델 '{primary}' 로드에 실패했고 사용할 fallback 모델이 없습니다. "
                f"RagSettings.embedding_fallback_model 을 지정하거나 "
                f'embedding_backend 를 "deterministic" 으로 바꾸세요. (원인: {primary_exc})'
            ) from primary_exc
        try:
            return embeddings_cls(
                model_name=fallback,
                model_kwargs=model_kwargs,
                encode_kwargs=encode_kwargs,
            )
        except Exception as fallback_exc:
            raise RuntimeError(
                f"임베딩 모델 로드에 두 번 모두 실패했습니다. "
                f"기본 모델 '{primary}' 오류: {primary_exc} / "
                f"fallback 모델 '{fallback}' 오류: {fallback_exc}. "
                f"네트워크·디스크 용량·device({device}) 설정을 확인하거나, "
                'RagSettings.embedding_backend 를 "deterministic" 으로 바꿔 '
                "오프라인 모드로 실행하세요."
            ) from fallback_exc


# ---------------------------------------------------------------------------
# 팩토리
# ---------------------------------------------------------------------------
def build_embeddings(settings: RagSettings | None = None) -> Any:
    """설정의 `embedding_backend` 에 맞는 LangChain Embeddings 호환 객체를 만든다.

    - "openai": OpenAI Embeddings API(운영/AWS 프리티어 기본).
      로컬에 모델·torch 를 두지 않으므로 RAM 1GB 인스턴스에서도 동작한다.
    - "huggingface": 로컬 multilingual 모델(Colab/GPU 기본). 지연 import + fallback 재시도.
    - "deterministic": 해시 기반 오프라인 임베딩(단위테스트/CI).

    반환 객체는 항상 `embed_documents(list[str]) -> list[list[float]]` 와
    `embed_query(str) -> list[float]` 를 제공하므로 vector store 는 백엔드를 구분하지 않는다.
    """
    rag = _as_rag_settings(settings)
    backend = getattr(rag, "embedding_backend", "huggingface")

    if backend == "deterministic":
        return DeterministicEmbeddings(
            dim=DETERMINISTIC_DIM,
            normalize=bool(getattr(rag, "embedding_normalize", True)),
        )
    if backend == "huggingface":
        return _build_huggingface_embeddings(rag)
    if backend == "openai":
        return _build_openai_embeddings(rag)

    raise ValueError(
        f"지원하지 않는 embedding_backend 입니다: {backend!r}. "
        '"openai" / "huggingface" / "deterministic" 중 하나여야 합니다.'
    )


def _build_openai_embeddings(rag: RagSettings) -> Any:
    """OpenAI Embeddings API 백엔드.

    AWS 프리티어(t3.micro, RAM 1GB)에서 로컬 임베딩 모델을 올릴 수 없어 도입했다.
    bge-m3 는 모델 파일만 2.2GB 이고 torch 까지 더하면 인스턴스가 죽는다.
    API 방식은 프로세스 메모리가 수십 MB 수준이라 프리티어에서 동작한다.

    차원(`embedding_openai_dimensions`)을 지정하면 API 가 축소된 벡터를 돌려준다.
    text-embedding-3-small 은 1536 이 기본이며 축소해도 품질 저하가 작아,
    FAISS index 크기와 검색 시간을 줄이려면 512~1024 를 쓸 수 있다.
    """
    model = getattr(rag, "embedding_openai_model", "text-embedding-3-small")
    dimensions = getattr(rag, "embedding_openai_dimensions", None)

    try:
        from langchain_openai import OpenAIEmbeddings  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - 설치 안내 경로
        raise RuntimeError(
            "embedding_backend='openai' 이려면 langchain-openai 가 필요합니다. "
            "`pip install langchain-openai` 후 다시 시도하세요."
        ) from exc

    import os  # noqa: PLC0415

    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError(
            "embedding_backend='openai' 인데 OPENAI_API_KEY 가 없습니다. "
            "환경 변수를 설정하거나 embedding_backend 를 'huggingface'/'deterministic' 로 바꾸세요."
        )

    kwargs: dict[str, Any] = {"model": model}
    if dimensions:
        kwargs["dimensions"] = int(dimensions)
    logger.info("OpenAI 임베딩 사용: model=%s dimensions=%s", model, dimensions or "기본")
    return OpenAIEmbeddings(**kwargs)


def embedding_dimension(embeddings: Any) -> int:
    """임베딩 객체의 출력 차원을 실제 1회 호출로 알아낸다.

    모델마다 차원 메타데이터 노출 방식이 달라, 추측 대신 빈 질의를 한 번 임베딩해
    길이를 재는 쪽이 확실하다. index 차원 검증에 쓴다.
    """
    dim = getattr(embeddings, "dim", None)
    if isinstance(dim, int) and dim > 0:
        return dim
    return len(embeddings.embed_query("dimension probe"))
