"""전역 설정 — 경로·모델·임계값·점수 가중치를 한 곳에서 주입한다.

Colab 전역변수에 비즈니스 로직이 의존하지 않도록, 모든 모듈은 이 파일의
`Settings` 객체를 인자로 받거나 `get_settings()` 로 읽는다.
API 키는 코드에 두지 않고 환경 변수에서만 읽는다.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

Species = Literal["dog", "cat"]
LLMProvider = Literal["anthropic", "openai"]
EmbeddingBackend = Literal["openai", "huggingface", "deterministic"]


# ---------------------------------------------------------------------------
# 수의학 웹 검색 allowlist — 커뮤니티·블로그는 근거로 쓰지 않는다.
# ---------------------------------------------------------------------------
VETERINARY_ALLOWED_DOMAINS: tuple[str, ...] = (
    "vet.cornell.edu",
    "merckvetmanual.com",
    "avma.org",
    "wsava.org",
    "aaha.org",
    "ucdavis.edu",
    "vetmed.ucdavis.edu",
    "acvim.org",
    "aspca.org",
    "petpoisonhelpline.com",
)

# 광고성·커뮤니티 페이지 차단용 신호 (URL 또는 제목에 포함되면 거절)
WEB_REJECT_SIGNALS: tuple[str, ...] = (
    "blog.",
    "cafe.",
    "tistory",
    "naver.me",
    "reddit.com",
    "quora.com",
    "facebook.com",
    "instagram.com",
    "pinterest.",
    "coupang",
    "amazon.",
    "/shop", "/product", "/cart", "buy-now",
)


# ---------------------------------------------------------------------------
# 병원 적합도 점수 — 명세 35절. 코드에 산재시키지 않고 여기서만 조정한다.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class HospitalScoreWeights:
    is_animal_hospital: int = 20
    has_phone: int = 10
    emergency_mentioned: int = 20
    open_24h_mentioned: int = 15
    specialty_matches: int = 15
    is_previous_hospital: int = 10

    # suitability 등급 경계 (score 기준)
    recommended_min: int = 55
    possible_min: int = 30


@dataclass
class RagSettings:
    """RAG 파이프라인 설정.

    frozen 이 아닌 이유: 노트북에서 임베딩 백엔드를 교체하거나(huggingface ↔
    deterministic) calibration 결과를 되써야 하므로 런타임 변경이 필요하다.
    """

    # chunking — 문자 기준
    chunk_size: int = 1000
    chunk_overlap: int = 150
    min_chunk_length: int = 200

    # 임베딩
    # 기본값이 huggingface 인 이유: Colab(GPU)에서는 로컬 bge-m3 가 품질이 가장 좋다.
    # AWS 프리티어(RAM 1GB)에 배포할 때는 "openai" 로 바꾼다 — 로컬 모델·torch 를
    # 올릴 수 없기 때문이며, 그 전환은 환경 변수 하나로 끝나도록 설계했다.
    embedding_backend: EmbeddingBackend = "huggingface"
    embedding_model: str = "BAAI/bge-m3"
    # OpenAI 백엔드 전용 설정
    embedding_openai_model: str = "text-embedding-3-small"
    # None 이면 모델 기본 차원(1536). 512~1024 로 줄이면 index 가 작아진다.
    embedding_openai_dimensions: int | None = 1024
    embedding_fallback_model: str = "intfloat/multilingual-e5-base"
    embedding_normalize: bool = True
    embedding_device: str | None = None  # None 이면 자동 감지(cuda > cpu)
    embedding_batch_size: int = 16

    # 검색
    top_k: int = 6
    fetch_k: int = 20
    use_mmr: bool = True
    mmr_lambda: float = 0.5
    final_evidence_max: int = 8

    # --- 충분성 판단 임계값 (명세 14절: 임의 고정 금지, calibration 후 확정) ---
    # 실측 근거: multilingual-e5 계열은 코사인 값이 좁은 밴드에 몰린다.
    # (Cornell dog 160문서 기준 min 0.713 / mean 0.772 / max 0.808, std 0.013)
    # → 절대 임계값은 모델을 바꾸면 곧바로 무의미해지므로 **상대 지표를 기본으로 쓴다.**
    #   관련 문서는 코퍼스 평균 대비 +0.11 이상 벌어졌다(영어 query 기준 0.88~0.93).
    score_threshold_calibrated: bool = False
    # 코퍼스 평균 대비 최소 margin — 기본 판단 기준
    min_relevance_margin: float = 0.05
    # 절대 임계값 — margin 을 쓸 수 없을 때만의 보조 기준(모델 의존적)
    min_relevance_score: float = 0.30
    # calibrate_threshold() 가 채우는 코퍼스 score 통계
    corpus_score_mean: float | None = None
    corpus_score_std: float | None = None
    min_documents_for_sufficient: int = 2
    min_topic_coverage: float = 0.5
    use_llm_sufficiency: bool = True


@dataclass
class Settings:
    """전체 설정 컨테이너."""

    # 경로
    data_dir: Path = Path("raw")
    documents_filename: str = "cornell_pet_health_documents.json"
    index_dir: Path = Path("faiss_index")
    output_dir: Path = Path("outputs")

    # LLM — 이 프로젝트는 OpenAI gpt-5.4-mini 를 사용한다(기존 일기장·진단서
    # 노트북과 동일 모델). provider 는 교체 가능하게 두되 기본값은 openai 다.
    llm_provider: LLMProvider = "openai"
    openai_model: str = "gpt-5.4-mini"
    anthropic_model: str = "claude-sonnet-5"
    llm_temperature: float = 0.0
    llm_timeout_seconds: float = 60.0
    llm_max_retries: int = 2

    # 대화 요약 — 최근 N개 원문 유지 (명세 41절)
    summary_keep_recent_messages: int = 8
    summary_trigger_message_count: int = 12

    # 하위 설정
    rag: RagSettings = field(default_factory=RagSettings)
    hospital_score: HospitalScoreWeights = field(default_factory=HospitalScoreWeights)
    allowed_web_domains: tuple[str, ...] = VETERINARY_ALLOWED_DOMAINS
    web_reject_signals: tuple[str, ...] = WEB_REJECT_SIGNALS

    # 실행 환경 표시 (LangSmith metadata 용)
    environment: str = "colab"

    # ---- 파생 경로 -------------------------------------------------------
    @property
    def documents_path(self) -> Path:
        return self.data_dir / self.documents_filename

    def index_path(self, species: Species) -> Path:
        """species 별 FAISS index 경로 — 강아지/고양이 문서를 섞지 않는다."""
        return self.index_dir / f"faiss_{species}"

    # ---- API 키 (환경 변수에서만 읽는다) ----------------------------------
    @property
    def anthropic_api_key(self) -> str | None:
        return os.environ.get("ANTHROPIC_API_KEY")

    @property
    def openai_api_key(self) -> str | None:
        return os.environ.get("OPENAI_API_KEY")

    @property
    def tavily_api_key(self) -> str | None:
        return os.environ.get("TAVILY_API_KEY")

    @property
    def has_llm_key(self) -> bool:
        return bool(
            self.anthropic_api_key
            if self.llm_provider == "anthropic"
            else self.openai_api_key
        )

    @property
    def llm_model(self) -> str:
        return (
            self.anthropic_model
            if self.llm_provider == "anthropic"
            else self.openai_model
        )

    def ensure_dirs(self) -> None:
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)


_settings = Settings()


def get_settings() -> Settings:
    return _settings


def configure(**overrides) -> Settings:
    """노트북 셀에서 설정을 덮어쓸 때 사용한다."""
    global _settings
    for key, value in overrides.items():
        if not hasattr(_settings, key):
            raise AttributeError(f"알 수 없는 설정 항목입니다: {key}")
        setattr(_settings, key, value)
    return _settings
