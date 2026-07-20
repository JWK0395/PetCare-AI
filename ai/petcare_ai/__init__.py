"""PetCare AI — 수의학 RAG + LangGraph 상담 파이프라인.

패키지 최상단에서는 **설정과 스키마만** 노출한다. 하위 모듈(rag, adapters,
pdf, llm)은 faiss/langchain/reportlab 같은 무거운 의존성을 쓰므로, 여기서
eager import 하면 `import petcare_ai` 한 줄이 그 전부를 요구하게 된다.
필요한 모듈은 각자 경로로 직접 import 한다.
"""

from __future__ import annotations

from .config import (
    VETERINARY_ALLOWED_DOMAINS,
    WEB_REJECT_SIGNALS,
    HospitalScoreWeights,
    RagSettings,
    Settings,
    configure,
    get_settings,
)
from .schemas import (
    RISK_PRIORITY,
    AssessmentResult,
    ChatGraphResult,
    ClinicalContext,
    ConsultationPacket,
    EmailDraft,
    EvidenceMergeResult,
    FinalEvidence,
    HospitalCandidate,
    HospitalRequirements,
    HospitalSuitabilityResult,
    KnowledgeSufficiencyResult,
    MissingInformationResult,
    OutputCheckResult,
    RagQuery,
    RetrievalResult,
    RetrievedEvidence,
    SupervisorResult,
    WebEvidence,
    merge_risk,
)

__version__ = "0.1.0"

__all__ = [
    "__version__",
    # config
    "Settings",
    "RagSettings",
    "HospitalScoreWeights",
    "get_settings",
    "configure",
    "VETERINARY_ALLOWED_DOMAINS",
    "WEB_REJECT_SIGNALS",
    # schemas
    "RetrievedEvidence",
    "RagQuery",
    "KnowledgeSufficiencyResult",
    "RetrievalResult",
    "WebEvidence",
    "FinalEvidence",
    "EvidenceMergeResult",
    "ClinicalContext",
    "SupervisorResult",
    "AssessmentResult",
    "MissingInformationResult",
    "HospitalRequirements",
    "HospitalCandidate",
    "HospitalSuitabilityResult",
    "ConsultationPacket",
    "EmailDraft",
    "ChatGraphResult",
    "OutputCheckResult",
    "RISK_PRIORITY",
    "merge_risk",
]
