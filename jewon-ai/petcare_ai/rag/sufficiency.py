"""RAG 검색 결과의 충분성 판단 (명세 14절).

검색이 끝났다고 바로 답변을 만들지 않는다. "이 근거로 답해도 되는가" 를 먼저
판정하고, 부족하면 웹 fallback(15절)으로 넘긴다.

구조는 **2단계**다.

1. deterministic 검사 — 결과 존재/species 일치/핵심 증상 포함/required_topics
   커버율/문서 수/score 임계/문서 간 충돌. 여기서 나온 판정이 기본값이다.
2. LLM structured output 검사 — deterministic 이 애매할 때만(=needs_llm_review)
   호출한다. LLM 은 품질 향상 옵션이지 필수 경로가 아니다.

가장 중요한 규칙: **LLM 이 sufficient 라고 해도 species 불일치나 빈 검색 결과는
덮어쓸 수 없다.** 이 가드는 두 겹으로 강제한다.
  - hard block 이 걸리면 LLM 을 아예 호출하지 않는다(비용·지연도 아낀다).
  - LLM 결과를 병합한 뒤에도 `enforce_hard_guards()` 를 다시 통과시킨다.
`enforce_hard_guards()` 는 공개 메서드라서 테스트가 임의의 LLM 응답을 넣어 직접
검증할 수 있다.

score 임계값은 임의로 고정하지 않는다(명세 14절). 임베딩 모델을 바꾸면 코사인
값 분포가 통째로 이동하므로, `calibrate_threshold()` 로 실제 score 분포를 뽑아
config 에 반영해야 한다. `settings.rag.score_threshold_calibrated` 가 False 인
동안에는 판정 결과 `reason` 에 '임계값 미보정' 경고를 남긴다.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from ..config import RagSettings, Settings, get_settings
from ..llm import safe_structured_invoke
from ..schemas import (
    KnowledgeSufficiencyResult,
    RagQuery,
    RetrievedEvidence,
    SufficiencyStatus,
)

logger = logging.getLogger(__name__)

__all__ = [
    "KnowledgeSufficiencyEvaluator",
    "calibrate_threshold",
    "apply_calibration",
    "CONFLICT_SIGNAL_PAIRS",
    "RECENCY_SIGNALS",
    "UNCALIBRATED_WARNING",
]


# ---------------------------------------------------------------------------
# 상수 — 신호어는 코드 곳곳에 흩지 않고 여기서만 관리한다.
# ---------------------------------------------------------------------------

#: score_threshold_calibrated=False 일 때 reason 에 반드시 들어가는 문구.
#: 다른 모듈/테스트가 '임계값 미보정' 부분 문자열로 탐지한다.
UNCALIBRATED_WARNING: str = (
    "임계값 미보정 — calibrate_threshold() 결과를 config 에 반영하기 전까지 "
    "score 기준 판정은 참고값입니다"
)

#: 질문이 '최신 정보'를 요구하는지 판단하는 신호어(한국어/영어 혼용).
#: 내부 Cornell 코퍼스는 스냅샷이라 리콜·유행 같은 질문에는 답할 수 없다.
RECENCY_SIGNALS: tuple[str, ...] = (
    # 한국어
    "발병", "유행", "확산", "리콜", "회수", "경보", "속보", "최근", "최신",
    "올해", "요즘", "신종", "변이", "권고안 변경",
    # 영어
    "outbreak", "recall", "recalled", "advisory", "alert", "epidemic",
    "epizootic", "emerging", "latest", "recent", "current guideline",
    "updated guideline", "new variant",
    # 연도 — 스냅샷 이후를 명시적으로 묻는 경우
    "2025", "2026", "2027",
)

#: 서로 다른 문서가 '심각하게' 충돌하는지 보는 신호어 쌍.
#: (A 그룹 신호, B 그룹 신호, 충돌 라벨) — 서로 다른 document_id 에서 A 와 B 가
#: 각각 발견되면 충돌로 본다. 한 문서 안에 둘 다 있는 건 조건부 서술("보통은
#: 가정 관찰, 다음 증상이면 즉시 내원")이라 충돌이 아니다.
CONFLICT_SIGNAL_PAIRS: tuple[tuple[tuple[str, ...], tuple[str, ...], str], ...] = (
    (
        (
            "seek immediate care",
            "seek immediate veterinary",
            "immediate veterinary attention",
            "immediate veterinary care",
            "emergency veterinary care",
            "go to the emergency",
            "requires urgent",
            "urgent veterinary attention",
            "즉시 병원",
            "즉시 내원",
            "응급 진료",
            "지체 없이 병원",
        ),
        (
            "monitor at home only",
            "monitor at home",
            "manage at home",
            "no veterinary visit is needed",
            "does not require veterinary",
            "does not require treatment",
            "no treatment is needed",
            "usually resolves on its own",
            "resolves without treatment",
            "self-limiting",
            "가정에서 관찰",
            "집에서 지켜",
            "치료가 필요하지 않",
            "병원에 갈 필요는 없",
        ),
        "즉시 진료 권고 vs 가정 관찰 권고",
    ),
    (
        (
            "is toxic to",
            "are toxic to",
            "is poisonous",
            "are poisonous",
            "can be fatal",
            "독성이 있",
            "중독을 일으",
        ),
        (
            "is safe for",
            "are safe for",
            "is not toxic",
            "are not toxic",
            "is considered safe",
            "안전합니다",
            "독성이 없",
        ),
        "독성 있음 vs 안전함",
    ),
    (
        (
            "should be fasted",
            "withhold food",
            "do not feed",
            "금식",
        ),
        (
            "continue feeding",
            "should be fed",
            "keep feeding",
            "계속 급여",
            "평소대로 급여",
        ),
        "금식 권고 vs 급여 유지 권고",
    ),
)

#: 핵심 증상어 추출 시 걸러낼 영어 불용어(검색 query 에 흔히 섞이는 말).
_EN_STOPWORDS: frozenset[str] = frozenset(
    {
        "the", "and", "for", "with", "what", "when", "why", "how", "does",
        "did", "can", "could", "should", "would", "is", "are", "was", "were",
        "this", "that", "these", "those", "have", "has", "had", "about",
        "from", "into", "your", "you", "his", "her", "its", "their", "they",
        "there", "here", "then", "than", "also", "any", "some", "many",
        "much", "more", "most", "not", "but", "out", "off", "over", "under",
        "please", "help", "need", "want", "tell", "give", "make",
        "dog", "dogs", "cat", "cats", "pet", "pets", "animal", "animals",
        "vet", "veterinary", "veterinarian", "owner", "case", "cases",
    }
)

#: 핵심 증상어로 인정할 한국어 불용어 제외 목록.
_KO_STOPWORDS: frozenset[str] = frozenset(
    {
        "강아지", "고양이", "반려견", "반려묘", "우리", "저희", "지금", "요즘",
        "그런데", "그리고", "하지만", "어떻게", "무엇", "무슨", "어떤", "정도",
        "때문", "경우", "관련", "대해", "대한", "해야", "하나요", "인가요",
        "있나요", "없나요", "병원", "동물병원", "수의사",
    }
)

_TOKEN_RE = re.compile(r"[a-z0-9]+|[가-힣]+")
_WS_RE = re.compile(r"\s+")


# ---------------------------------------------------------------------------
# 내부 자료구조
# ---------------------------------------------------------------------------
@dataclass
class _DeterministicVerdict:
    """deterministic 단계의 판정 결과 묶음.

    LLM 병합 단계가 이 값을 기준으로 삼고, 판정 근거(reasons)도 여기에 모은다.
    """

    status: SufficiencyStatus = "insufficient"
    covered_topics: list[str] = field(default_factory=list)
    missing_topics: list[str] = field(default_factory=list)
    requires_recent_information: bool = False
    reasons: list[str] = field(default_factory=list)
    hard_block_reason: str | None = None
    conflict_detected: bool = False
    conflict_labels: list[str] = field(default_factory=list)
    topic_coverage: float = 1.0
    symptom_covered: bool = True
    relevant_count: int = 0
    needs_llm_review: bool = False


# ---------------------------------------------------------------------------
# 평가기
# ---------------------------------------------------------------------------
class KnowledgeSufficiencyEvaluator:
    """RAG 근거가 답변에 충분한지 판정한다.

    `llm` 은 주입 가능(테스트가 mock 을 넣는다)하고 None 이어도 된다.
    None 이면 deterministic 판정만으로 동작하므로 API 키 없는 환경에서도
    파이프라인 전체가 그대로 돌아간다.
    """

    def __init__(self, settings: Any = None, llm: Any = None) -> None:
        """settings 는 Settings / RagSettings / None 을 모두 허용한다.

        호출자가 어디서는 전역 Settings 를, 어디서는 rag 하위 설정만 들고
        있어서 둘 다 받아준다. None 이면 `get_settings()` 를 쓴다.
        """
        self.settings: Settings | RagSettings | None = settings
        self.llm = llm
        self._rag: RagSettings = _resolve_rag_settings(settings)

    # -- 공개 API ----------------------------------------------------------
    def evaluate(
        self,
        query: RagQuery,
        documents: list[RetrievedEvidence],
    ) -> KnowledgeSufficiencyResult:
        """검색 결과의 충분성을 판정한다.

        순서가 중요하다. deterministic 검사에서 hard block(빈 결과/species
        불일치)이 걸리면 LLM 을 호출하지 않고 즉시 insufficient 로 끝낸다.
        그 외의 애매한 경우에만 LLM 을 부르고, 병합 후 다시 hard guard 를
        통과시킨다.
        """
        docs = list(documents or [])
        verdict = self._deterministic(query, docs)
        base = self._to_result(verdict)

        if verdict.hard_block_reason is not None:
            # LLM 이 뒤집을 수 없는 조건이므로 호출 자체를 생략한다.
            return self.enforce_hard_guards(base, query, docs)

        if not self._should_consult_llm(verdict):
            return base

        llm_result = self._llm_review(query, docs, base)
        merged = self._merge_llm_result(verdict, base, llm_result)
        return self.enforce_hard_guards(merged, query, docs)

    def enforce_hard_guards(
        self,
        result: KnowledgeSufficiencyResult,
        query: RagQuery,
        documents: list[RetrievedEvidence],
    ) -> KnowledgeSufficiencyResult:
        """뒤집을 수 없는 조건을 최종 강제한다.

        빈 검색 결과와 species 불일치는 어떤 판정도 sufficient/conflicting 로
        만들 수 없다. LLM 응답이든 사람이 만든 결과든 이 함수를 지나면
        반드시 insufficient 가 된다. 테스트가 직접 호출해 검증하도록 공개한다.
        """
        block = _hard_block_reason(query, list(documents or []))
        if block is None:
            return result
        if result.status == "insufficient" and block in result.reason:
            return result
        reason = _join_reasons(
            [f"[강제] {block}", result.reason.strip()],
        )
        return result.model_copy(update={"status": "insufficient", "reason": reason})

    # -- 1단계: deterministic ----------------------------------------------
    def _deterministic(
        self,
        query: RagQuery,
        documents: list[RetrievedEvidence],
    ) -> _DeterministicVerdict:
        """규칙만으로 판정한다 — LLM 없이도 여기까지는 항상 동작한다."""
        rag = self._rag
        verdict = _DeterministicVerdict()
        verdict.requires_recent_information = _requires_recent_information(query)

        # (1) 결과 존재 / (2) species 일치 — 되돌릴 수 없는 조건
        block = _hard_block_reason(query, documents)
        if block is not None:
            verdict.hard_block_reason = block
            verdict.status = "insufficient"
            verdict.missing_topics = list(query.required_topics)
            verdict.reasons.append(block)
            verdict.needs_llm_review = False
            self._append_calibration_note(verdict)
            return verdict

        blobs = [_document_blob(doc) for doc in documents]

        # (3) required_topics 커버율
        covered, missing = _split_topics(query.required_topics, blobs)
        verdict.covered_topics = covered
        verdict.missing_topics = missing
        total_topics = len(query.required_topics)
        coverage = 1.0 if total_topics == 0 else len(covered) / total_topics
        verdict.topic_coverage = coverage
        if total_topics:
            verdict.reasons.append(
                f"topic 커버율 {coverage:.2f} ({len(covered)}/{total_topics},"
                f" 기준 {rag.min_topic_coverage:.2f})"
            )

        # (4) 질문 핵심 증상이 문서에 등장하는가
        key_terms = _key_terms(query)
        matched_terms = [term for term in key_terms if any(term in blob for blob in blobs)]
        if key_terms:
            ratio = len(matched_terms) / len(key_terms)
            # 핵심어가 하나도 안 걸리면 다른 주제의 문서를 가져온 것이다.
            verdict.symptom_covered = bool(matched_terms) and ratio >= 0.25
            verdict.reasons.append(
                f"핵심 증상어 매칭 {len(matched_terms)}/{len(key_terms)}"
                f" ({', '.join(matched_terms[:5]) or '없음'})"
            )
        else:
            # 추출할 핵심어가 없으면 이 항목으로 탈락시키지 않는다.
            verdict.symptom_covered = True

        # (5)(6) score 임계 + 문서 수
        relevant, score_note = _relevance_filter(documents, rag)
        verdict.relevant_count = len(relevant)
        verdict.reasons.append(
            f"관련 문서 {len(relevant)}/{len(documents)}건"
            f" (기준 {rag.min_documents_for_sufficient}건, {score_note})"
        )

        # (7) 서로 다른 문서 간 심각한 충돌
        conflicts = _detect_conflicts(documents)
        verdict.conflict_detected = bool(conflicts)
        verdict.conflict_labels = conflicts

        # 최종 status 결정
        enough_documents = len(relevant) >= rag.min_documents_for_sufficient
        enough_topics = coverage >= rag.min_topic_coverage
        if conflicts:
            verdict.status = "conflicting"
            verdict.reasons.append("문서 간 상반된 권고: " + "; ".join(conflicts))
        elif enough_documents and enough_topics and verdict.symptom_covered:
            verdict.status = "sufficient"
        else:
            verdict.status = "insufficient"
            if not enough_documents:
                verdict.reasons.append("관련 문서 수 부족")
            if not enough_topics:
                verdict.reasons.append("required_topics 커버율 부족")
            if not verdict.symptom_covered:
                verdict.reasons.append("질문 핵심 증상이 문서에 없음")

        if verdict.requires_recent_information:
            verdict.reasons.append("질문에 최신성 신호가 있어 내부 스냅샷만으로는 부족할 수 있음")

        # 완벽히 깔끔한 sufficient 가 아니면 LLM 에게 한 번 더 물어본다.
        verdict.needs_llm_review = not (
            verdict.status == "sufficient"
            and coverage >= 1.0
            and verdict.symptom_covered
            and not verdict.requires_recent_information
        )

        self._append_calibration_note(verdict)
        return verdict

    def _append_calibration_note(self, verdict: _DeterministicVerdict) -> None:
        """score 임계값이 보정되지 않았으면 경고를 남긴다(명세 14절)."""
        if not self._rag.score_threshold_calibrated:
            verdict.reasons.append(UNCALIBRATED_WARNING)

    # -- 2단계: LLM --------------------------------------------------------
    def _should_consult_llm(self, verdict: _DeterministicVerdict) -> bool:
        """LLM 을 부를지 결정한다 — '필요한 경우에만' 부른다."""
        if self.llm is None:
            return False
        if not self._rag.use_llm_sufficiency:
            return False
        return verdict.needs_llm_review

    def _llm_review(
        self,
        query: RagQuery,
        documents: list[RetrievedEvidence],
        default: KnowledgeSufficiencyResult,
    ) -> KnowledgeSufficiencyResult:
        """LLM structured output 으로 2차 판정을 받는다.

        실패(타임아웃/스키마 위반/패키지 없음)해도 예외를 내지 않고 deterministic
        결과를 그대로 돌려준다 — `safe_structured_invoke` 규약.
        """
        messages = _build_llm_messages(query, documents)
        return safe_structured_invoke(
            self.llm, messages, KnowledgeSufficiencyResult, default
        )

    def _merge_llm_result(
        self,
        verdict: _DeterministicVerdict,
        base: KnowledgeSufficiencyResult,
        llm_result: KnowledgeSufficiencyResult,
    ) -> KnowledgeSufficiencyResult:
        """deterministic 결과와 LLM 결과를 안전한 방향으로만 병합한다.

        병합 규칙(모두 '위험을 낮추는 쪽으로는 못 간다'는 원칙):
          - deterministic 이 문서 간 충돌을 실제로 찾아냈으면 LLM 이 sufficient
            라 해도 conflicting 을 유지한다. 충돌은 신호어로 확인된 사실이다.
          - covered/missing 은 required_topics 기준의 실측이므로 유지하되,
            LLM 이 "이건 사실 부족하다" 고 한 topic 은 missing 으로 내린다.
            (missing → covered 방향의 승격은 허용하지 않는다.)
          - requires_recent_information 은 OR 로 합친다.
        """
        if llm_result is base:
            return base

        status: SufficiencyStatus = llm_result.status
        notes: list[str] = []
        if verdict.conflict_detected and status == "sufficient":
            status = "conflicting"
            notes.append("LLM 은 sufficient 라 했으나 문서 간 충돌이 확인되어 conflicting 유지")

        covered = list(verdict.covered_topics)
        missing = list(verdict.missing_topics)
        demoted = [
            topic
            for topic in llm_result.missing_topics
            if topic in covered and topic not in missing
        ]
        if demoted:
            covered = [topic for topic in covered if topic not in demoted]
            missing = missing + demoted
            notes.append("LLM 판단으로 부족 처리된 topic: " + ", ".join(demoted))

        reason = _join_reasons(
            [base.reason, *notes, f"LLM: {llm_result.reason.strip()}" if llm_result.reason.strip() else ""]
        )
        return KnowledgeSufficiencyResult(
            status=status,
            covered_topics=covered,
            missing_topics=missing,
            requires_recent_information=(
                base.requires_recent_information or llm_result.requires_recent_information
            ),
            reason=reason,
        )

    # -- 변환 --------------------------------------------------------------
    @staticmethod
    def _to_result(verdict: _DeterministicVerdict) -> KnowledgeSufficiencyResult:
        """내부 판정 묶음을 계약 스키마로 옮긴다."""
        return KnowledgeSufficiencyResult(
            status=verdict.status,
            covered_topics=list(verdict.covered_topics),
            missing_topics=list(verdict.missing_topics),
            requires_recent_information=verdict.requires_recent_information,
            reason=_join_reasons(verdict.reasons),
        )


# ---------------------------------------------------------------------------
# score 분포 calibration (명세 14절 — 임계값을 임의로 고정하지 않는다)
# ---------------------------------------------------------------------------
def calibrate_threshold(documents_by_query: Any) -> dict[str, Any]:
    """테스트 query 들의 검색 score 분포를 계산해 돌려준다.

    임베딩 모델이 바뀌면 코사인 값 밴드가 통째로 이동하므로, 절대 임계값을
    코드에 박아두면 곧 무의미해진다. 그래서 이 함수로 실제 분포(min/max/평균/
    표준편차/분위수)를 먼저 뽑고, 그 결과를 config 에 넣은 뒤에야
    `score_threshold_calibrated=True` 로 올린다.

    인자는 `{query: [RetrievedEvidence, ...]}` 매핑을 기본으로 받고, 편의상
    `[[evidence, ...], ...]` 같은 시퀀스도 허용한다.

    반환 dict 의 주요 키:
      - count/query_count/missing_score_count
      - min/max/mean/median/stdev
      - p10/p25/p50/p75/p90
      - top1_mean: query 별 1등 score 의 평균(=관련 문서의 대표 수준)
      - suggested_min_relevance_score: 절대 임계 후보(p25)
      - suggested_min_relevance_margin: 코퍼스 평균 대비 margin 후보
      - per_query: query 별 요약
    """
    grouped = _as_query_groups(documents_by_query)

    all_scores: list[float] = []
    top1_scores: list[float] = []
    missing = 0
    per_query: dict[str, dict[str, Any]] = {}

    for label, docs in grouped:
        scores = [float(d.score) for d in docs if getattr(d, "score", None) is not None]
        missing += len(docs) - len(scores)
        if scores:
            all_scores.extend(scores)
            top1_scores.append(max(scores))
        per_query[label] = {
            "document_count": len(docs),
            "score_count": len(scores),
            "min": min(scores) if scores else None,
            "max": max(scores) if scores else None,
            "mean": _mean(scores),
        }

    stats: dict[str, Any] = {
        "query_count": len(grouped),
        "count": len(all_scores),
        "missing_score_count": missing,
        "min": min(all_scores) if all_scores else None,
        "max": max(all_scores) if all_scores else None,
        "mean": _mean(all_scores),
        "median": _percentile(sorted(all_scores), 50.0),
        "stdev": _stdev(all_scores),
        "p10": _percentile(sorted(all_scores), 10.0),
        "p25": _percentile(sorted(all_scores), 25.0),
        "p50": _percentile(sorted(all_scores), 50.0),
        "p75": _percentile(sorted(all_scores), 75.0),
        "p90": _percentile(sorted(all_scores), 90.0),
        "top1_mean": _mean(top1_scores),
        "per_query": per_query,
    }

    if all_scores:
        # 절대 임계 후보: 하위 25% 는 잡음으로 본다.
        stats["suggested_min_relevance_score"] = _round(stats["p25"])
        # margin 후보: 1등 score 평균이 전체 평균보다 얼마나 위인가.
        mean_all = stats["mean"] or 0.0
        top1_mean = stats["top1_mean"]
        stats["suggested_min_relevance_margin"] = (
            _round(max(top1_mean - mean_all, 0.0)) if top1_mean is not None else None
        )
        stats["note"] = (
            "제안값을 config.RagSettings 에 반영하고 score_threshold_calibrated=True 로 "
            "올린 뒤에만 임계값 판정을 신뢰하십시오."
        )
    else:
        stats["suggested_min_relevance_score"] = None
        stats["suggested_min_relevance_margin"] = None
        stats["note"] = "score 가 있는 검색 결과가 없어 임계값을 보정할 수 없습니다."

    return stats


def apply_calibration(
    stats: Mapping[str, Any],
    settings: Any = None,
    *,
    mark_calibrated: bool = True,
) -> RagSettings:
    """`calibrate_threshold()` 결과를 RagSettings 에 반영한다.

    노트북에서 calibration 셀을 돌린 뒤 그 값을 손으로 옮겨 적다가 틀리는 일을
    막기 위한 편의 함수다. 제안값이 없으면(=score 없음) 아무것도 바꾸지 않고
    `mark_calibrated` 도 무시한다 — 보정되지 않은 상태를 보정됨으로 표시하면
    경고 문구가 사라져 더 위험하다.
    """
    rag = _resolve_rag_settings(settings)
    if stats.get("mean") is None:
        logger.warning("score 통계가 비어 있어 calibration 을 적용하지 않았습니다.")
        return rag

    rag.corpus_score_mean = float(stats["mean"])
    if stats.get("stdev") is not None:
        rag.corpus_score_std = float(stats["stdev"])
    if stats.get("suggested_min_relevance_score") is not None:
        rag.min_relevance_score = float(stats["suggested_min_relevance_score"])
    if stats.get("suggested_min_relevance_margin"):
        rag.min_relevance_margin = float(stats["suggested_min_relevance_margin"])
    if mark_calibrated:
        rag.score_threshold_calibrated = True
    return rag


# ---------------------------------------------------------------------------
# 내부 헬퍼
# ---------------------------------------------------------------------------
def _resolve_rag_settings(settings: Any) -> RagSettings:
    """Settings / RagSettings / None 중 무엇이 와도 RagSettings 를 돌려준다."""
    if settings is None:
        return get_settings().rag
    rag = getattr(settings, "rag", None)
    if rag is not None:
        return rag
    if isinstance(settings, RagSettings):
        return settings
    # 덕타이핑으로 들어온 설정 객체(테스트용 stub)도 그대로 받아준다.
    if hasattr(settings, "min_topic_coverage"):
        return settings  # type: ignore[return-value]
    raise TypeError(
        "settings 는 Settings 또는 RagSettings 여야 합니다: " f"{type(settings).__name__}"
    )


def _hard_block_reason(
    query: RagQuery,
    documents: list[RetrievedEvidence],
) -> str | None:
    """LLM 이 절대 뒤집을 수 없는 실패 사유를 찾는다. 없으면 None.

    - 검색 결과가 비어 있으면 근거가 아예 없는 것이다.
    - species 가 다르면 잘못된 index 를 탄 것이므로 내용과 무관하게 못 쓴다.
      (강아지 질문에 고양이 문서로 답하는 사고를 코드로 차단한다.)
    """
    if not documents:
        return "검색 결과가 비어 있습니다"

    mismatched = [doc for doc in documents if doc.species != query.species]
    if mismatched:
        found = sorted({doc.species for doc in mismatched})
        return (
            f"species 불일치: 질의={query.species}, 문서={'/'.join(found)}"
            f" ({len(mismatched)}/{len(documents)}건)"
        )
    return None


def _normalize(text: str) -> str:
    """비교용 정규화 — 소문자 + 공백 축약."""
    return _WS_RE.sub(" ", (text or "").lower()).strip()


def _tokenize(text: str) -> list[str]:
    """영문/숫자/한글 토큰만 뽑는다."""
    return _TOKEN_RE.findall((text or "").lower())


def _document_blob(doc: RetrievedEvidence) -> str:
    """문서 1건을 검색 가능한 단일 문자열로 만든다.

    본문뿐 아니라 title/categories/heading_path 까지 넣는 이유: topic 이
    본문 문장에는 없고 소제목이나 카테고리에만 있는 경우가 흔하다.
    """
    parts = [
        doc.title or "",
        doc.text or "",
        " ".join(doc.categories or []),
        " ".join(doc.heading_path or []),
    ]
    return _normalize(" ".join(parts))


def _key_terms(query: RagQuery) -> list[str]:
    """질문의 핵심 증상어 후보를 뽑는다.

    코퍼스가 영어(Cornell)이므로 영어 query 토큰을 우선 쓰고, 영어 토큰이
    없을 때만 한국어 토큰으로 넘어간다. 종 이름(dog/cat)이나 일반 동사는
    어떤 문서에나 있어 변별력이 없으므로 불용어로 제거한다.
    """
    en_tokens = [
        token
        for token in _tokenize(query.primary_query_en)
        if len(token) >= 4 and token not in _EN_STOPWORDS and not token.isdigit()
    ]
    terms = _dedupe(en_tokens)
    if terms:
        return terms
    ko_tokens = [
        token
        for token in _tokenize(query.primary_query_ko)
        if len(token) >= 2 and token not in _KO_STOPWORDS
    ]
    return _dedupe(ko_tokens)


def _split_topics(
    required_topics: Sequence[str],
    blobs: Sequence[str],
) -> tuple[list[str], list[str]]:
    """required_topics 를 covered / missing 으로 나눈다."""
    covered: list[str] = []
    missing: list[str] = []
    for topic in required_topics:
        if _topic_covered(topic, blobs):
            covered.append(topic)
        else:
            missing.append(topic)
    return covered, missing


def _topic_covered(topic: str, blobs: Sequence[str]) -> bool:
    """topic 이 문서들에 실제로 등장하는지 본다.

    1) topic 문자열이 통째로 들어 있으면 커버.
    2) 아니면 topic 을 토큰으로 쪼개 한 문서 안에서 절반 이상이 발견되면 커버.
       ("kidney disease diet" 처럼 어순이 다르게 서술되는 경우를 잡는다.)
    """
    norm = _normalize(topic)
    if not norm:
        return False
    if any(norm in blob for blob in blobs):
        return True

    tokens = [
        token
        for token in _tokenize(norm)
        if len(token) >= 3 and token not in _EN_STOPWORDS
    ]
    if not tokens:
        return False
    for blob in blobs:
        hits = sum(1 for token in tokens if token in blob)
        if hits / len(tokens) >= 0.5:
            return True
    return False


def _relevance_filter(
    documents: Sequence[RetrievedEvidence],
    rag: RagSettings,
) -> tuple[list[RetrievedEvidence], str]:
    """score 기준으로 '관련 있다'고 볼 문서만 남긴다.

    score 는 값이 클수록 관련도가 높다고 가정한다(vector_store 가 코사인
    유사도를 넣는다). 판단 기준 우선순위:
      1) 코퍼스 평균이 보정돼 있으면 상대 margin — 모델 교체에 강하다.
      2) 없으면 절대 임계값 — 모델 의존적이므로 보조 수단이다.
      3) score 자체가 없으면 필터링하지 않는다(개수 기준만 적용).
    """
    scored = [doc for doc in documents if doc.score is not None]
    if not scored:
        return list(documents), "score 없음 — 개수 기준만 적용"

    if rag.corpus_score_mean is not None:
        threshold = rag.corpus_score_mean + rag.min_relevance_margin
        basis = f"코퍼스 평균 {rag.corpus_score_mean:.4f}+margin {rag.min_relevance_margin:.4f}"
    else:
        threshold = rag.min_relevance_score
        basis = f"절대 임계 {threshold:.4f}"

    # score 가 없는 문서는 판단 불가이므로 탈락시키지 않는다.
    relevant = [doc for doc in documents if doc.score is None or doc.score >= threshold]
    return relevant, basis


def _detect_conflicts(documents: Sequence[RetrievedEvidence]) -> list[str]:
    """서로 다른 문서가 상반된 권고를 하는지 신호어 쌍으로 탐지한다.

    같은 document_id 안에서 두 신호가 모두 나오는 건 조건부 서술이므로
    충돌로 보지 않는다. 문서가 1건이면 비교 대상이 없어 항상 빈 리스트다.
    """
    if len(documents) < 2:
        return []

    blobs = [(doc.document_id, _document_blob(doc)) for doc in documents]
    conflicts: list[str] = []

    for group_a, group_b, label in CONFLICT_SIGNAL_PAIRS:
        docs_a = {doc_id for doc_id, blob in blobs if any(sig in blob for sig in group_a)}
        docs_b = {doc_id for doc_id, blob in blobs if any(sig in blob for sig in group_b)}
        # 한쪽에만 있는 문서가 각각 존재해야 진짜 충돌이다.
        only_a = docs_a - docs_b
        only_b = docs_b - docs_a
        if only_a and only_b:
            conflicts.append(
                f"{label} (문서 {sorted(only_a)[0]} vs {sorted(only_b)[0]})"
            )
    return conflicts


def _requires_recent_information(query: RagQuery) -> bool:
    """질문이 최신 정보(리콜·유행·경보 등)를 요구하는지 신호어로 본다.

    내부 코퍼스는 특정 시점 스냅샷이라 이런 질문은 원칙적으로 웹 fallback 이
    필요하다. 판단을 LLM 에만 맡기지 않는 이유는 키 없는 환경에서도 이
    경로가 동작해야 하기 때문이다.
    """
    haystack = " ".join(
        [
            query.primary_query_ko or "",
            query.primary_query_en or "",
            " ".join(query.required_topics or []),
        ]
    ).lower()
    return any(signal in haystack for signal in RECENCY_SIGNALS)


def _build_llm_messages(
    query: RagQuery,
    documents: Sequence[RetrievedEvidence],
) -> list[tuple[str, str]]:
    """LLM 2차 판정용 메시지를 만든다(토큰 절약을 위해 본문은 잘라 넣는다)."""
    system = (
        "당신은 수의학 지식 검색 결과의 충분성을 판정하는 평가자입니다.\n"
        "주어진 문서만으로 질문에 안전하게 답할 수 있는지 판단하십시오.\n"
        "- 진단이나 처방을 하지 마십시오. 충분성만 판정합니다.\n"
        "- 문서가 서로 상반된 권고를 하면 status='conflicting' 입니다.\n"
        "- 근거가 부족하거나 핵심 주제가 빠졌으면 status='insufficient' 입니다.\n"
        "- covered_topics/missing_topics 는 반드시 주어진 required_topics 중에서만 고르십시오.\n"
        "- 리콜·유행·경보처럼 최신 정보가 필요한 질문이면 "
        "requires_recent_information=true 로 표시하십시오.\n"
        "- reason 은 한국어 한두 문장으로 씁니다."
    )
    doc_lines: list[str] = []
    for index, doc in enumerate(documents[:8], start=1):
        snippet = _normalize(doc.text)[:700]
        doc_lines.append(
            f"[{index}] title={doc.title} | species={doc.species} | "
            f"categories={', '.join(doc.categories[:5])}\n{snippet}"
        )
    human = (
        f"질문(ko): {query.primary_query_ko}\n"
        f"질문(en): {query.primary_query_en}\n"
        f"species: {query.species}\n"
        f"required_topics: {', '.join(query.required_topics) or '(없음)'}\n\n"
        "검색된 문서:\n" + "\n\n".join(doc_lines or ["(없음)"])
    )
    return [("system", system), ("human", human)]


def _join_reasons(reasons: Sequence[str]) -> str:
    """판정 근거를 하나의 문장으로 합친다(빈 문자열·중복 제거)."""
    cleaned = _dedupe([reason.strip() for reason in reasons if reason and reason.strip()])
    return " / ".join(cleaned)


def _dedupe(values: Sequence[str]) -> list[str]:
    """순서를 유지하며 중복을 제거한다."""
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result


def _as_query_groups(documents_by_query: Any) -> list[tuple[str, list[RetrievedEvidence]]]:
    """calibration 입력을 (라벨, 문서목록) 리스트로 정규화한다."""
    if documents_by_query is None:
        return []
    if isinstance(documents_by_query, Mapping):
        return [(str(key), list(value or [])) for key, value in documents_by_query.items()]
    if isinstance(documents_by_query, Sequence) and not isinstance(documents_by_query, (str, bytes)):
        groups: list[tuple[str, list[RetrievedEvidence]]] = []
        for index, item in enumerate(documents_by_query):
            if isinstance(item, RetrievedEvidence):
                # 평평한 evidence 리스트를 준 경우 — 단일 query 로 본다.
                return [("query_0", list(documents_by_query))]
            groups.append((f"query_{index}", list(item or [])))
        return groups
    raise TypeError(
        "documents_by_query 는 {query: [RetrievedEvidence]} 매핑이거나 "
        "[[RetrievedEvidence]] 시퀀스여야 합니다."
    )


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _stdev(values: Sequence[float]) -> float | None:
    """표본 표준편차 — 값이 2개 미만이면 None(0.0 으로 속이지 않는다)."""
    if len(values) < 2:
        return None
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return variance**0.5


def _percentile(sorted_values: Sequence[float], percent: float) -> float | None:
    """선형 보간 분위수 — numpy 없이 계산한다."""
    if not sorted_values:
        return None
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    position = (len(sorted_values) - 1) * (percent / 100.0)
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = position - lower
    return float(sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight)


def _round(value: float | None, digits: int = 4) -> float | None:
    return None if value is None else round(float(value), digits)
