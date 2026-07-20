"""Evidence Merge (명세 16절).

Cornell 내부 RAG 근거와 **검증을 통과한** 웹 근거를 하나의 최종 근거 목록으로
합친다. 규칙은 명세 그대로다.

1. 우선순위: Cornell RAG > 검증된 웹 자료.
2. 같은 주장: 근거를 버리지 않고 함께 남겨 근거를 강화한다.
3. RAG 에 없고 웹에만 있음: 보조 근거로 사용한다.
4. 충돌: 감추지 않고 `conflicts` 에 기록하고, 안내는 보수적인(더 위험한) 쪽으로
   기울인다.
5. 신뢰 가능한 자료가 하나도 없음: `has_reliable_evidence=False`. 호출자는
   이때 추측하지 말고 "확실하지 않다 + 병원 상담 권고"로 답해야 한다.

`accepted=True` 가 아닌 웹 항목은 **무조건 버린다.** 검증되지 않은 웹 근거를
쓰는 것은 금지 사항이라, validator 를 거치지 않고 들어온 결과가 조용히 섞이는
경로를 만들지 않는다.

표준 라이브러리 + pydantic + 같은 패키지만 사용한다.
"""

from __future__ import annotations

import hashlib
import logging
import re

from ..config import get_settings
from ..schemas import EvidenceMergeResult, FinalEvidence, RetrievedEvidence, WebEvidence
from .source_validator import extract_domain

logger = logging.getLogger(__name__)

#: required_topics 가 비었을 때 충돌 비교에 쓰는 가상 주제.
#: 주제가 없다고 충돌 탐지를 통째로 끄면, 가장 위험한 케이스(독성 여부가
#: 엇갈리는 상황)를 놓친다.
_WHOLE_TOPIC = "전체 내용"

_EN_TOKEN_RE = re.compile(r"[a-z][a-z0-9\-]{2,}")
_KO_TOKEN_RE = re.compile(r"[가-힣]{2,}")

#: topic 토큰 중 이 비율 이상이 문서에서 확인되면 그 topic 을 다룬다고 본다.
_TOPIC_MATCH_RATIO: float = 0.5

# ---------------------------------------------------------------------------
# 충돌 탐지용 stance 큐
# ---------------------------------------------------------------------------
# "non-toxic" 안에 "toxic" 이 들어 있어 그대로 두면 안전/위험 신호가 동시에
# 켜진다. 부정형을 먼저 안전 마커로 치환한다.
_NEGATED_HARM: tuple[str, ...] = (
    "non-toxic",
    "nontoxic",
    "not toxic",
    "non-poisonous",
    "not poisonous",
    "not dangerous",
    "not harmful",
    "무독성",
    "독성이 없",
    "위험하지 않",
)
_SAFE_MARKER = " __safe__ "

_STANCE_CUES: dict[str, tuple[str, ...]] = {
    "safe": (_SAFE_MARKER.strip(), "generally safe", "is safe", "harmless", "안전합니다", "안전하다", "무해"),
    "harmful": ("toxic", "poisonous", "harmful", "dangerous", "lethal", "fatal", "독성", "중독", "치명적", "위험합니다"),
    "urgent": (
        "emergency",
        "immediately",
        "seek veterinary care",
        "veterinary emergency",
        "즉시",
        "응급",
        "당장",
    ),
    "watchful": (
        "monitor at home",
        "home care",
        "wait and see",
        "usually resolves",
        "self-limiting",
        "경과를 지켜",
        "집에서 관찰",
        "대개 호전",
    ),
}

#: 서로 모순되는 stance 쌍
_CONFLICTING_PAIRS: tuple[tuple[str, str], ...] = (("safe", "harmful"), ("urgent", "watchful"))

_STANCE_LABELS: dict[str, str] = {
    "safe": "위험하지 않다는 설명",
    "harmful": "독성/위험이 있다는 설명",
    "urgent": "즉시 진료가 필요하다는 설명",
    "watchful": "경과 관찰로 충분하다는 설명",
}

#: 충돌 시 더 보수적인(안전한 쪽으로 안내해야 하는) stance
_CONSERVATIVE: dict[str, str] = {"harmful": "harmful", "urgent": "urgent"}


# ---------------------------------------------------------------------------
# 내부 헬퍼
# ---------------------------------------------------------------------------
def _significant_tokens(text: str) -> list[str]:
    """topic 문자열에서 비교에 쓸 토큰만 뽑는다(영어 3자+, 한글 2자+)."""
    lowered = (text or "").lower()
    return _EN_TOKEN_RE.findall(lowered) + _KO_TOKEN_RE.findall(lowered)


def _token_present(token: str, lowered_text: str) -> bool:
    """토큰이 문서에 있는지. 한국어는 조사/어미 때문에 앞 2글자로 느슨하게 본다."""
    if not token:
        return False
    if _KO_TOKEN_RE.fullmatch(token):
        return token[:2] in lowered_text
    return token in lowered_text


def _topic_supported(topic: str, lowered_text: str) -> bool:
    """이 근거 텍스트가 해당 topic 을 다루는지 판단한다."""
    normalized = (topic or "").strip().lower()
    if not normalized:
        return False
    if normalized in lowered_text:
        return True
    tokens = _significant_tokens(normalized)
    if not tokens:
        return False
    matched = sum(1 for token in tokens if _token_present(token, lowered_text))
    return (matched / len(tokens)) >= _TOPIC_MATCH_RATIO


def _supported_topics(text: str, topics: list[str]) -> list[str]:
    """required_topics 중 이 근거가 실제로 다루는 것만 남긴다."""
    if not topics:
        return []
    lowered = (text or "").lower()
    return [topic for topic in topics if _topic_supported(topic, lowered)]


def _stances(text: str) -> set[str]:
    """텍스트의 입장(안전/위험, 즉시진료/경과관찰)을 추정한다.

    LLM 없이 동작해야 하므로 큐 문자열 기반이다. 놓치는 충돌이 있을 수는 있어도
    **없는 충돌을 만들어내지 않는** 쪽으로 보수적으로 잡았다.
    """
    lowered = (text or "").lower()
    for negated in _NEGATED_HARM:
        lowered = lowered.replace(negated, _SAFE_MARKER)
    return {name for name, cues in _STANCE_CUES.items() if any(cue in lowered for cue in cues)}


def _web_evidence_id(url: str, domain: str, fallback: str) -> str:
    """웹 근거의 안정적인 id — 같은 URL 이면 실행마다 같은 값이 나와야 한다.

    URL 을 소문자/끝 슬래시 제거로 정규화한 뒤 sha256 앞 10자리를 쓴다.
    도메인을 앞에 붙여 trace 에서 눈으로 출처를 알아볼 수 있게 한다.
    """
    seed = (url or fallback or "").strip().lower().rstrip("/")
    digest = hashlib.sha256(seed.encode("utf-8")).hexdigest()[:10]
    return f"web:{domain or 'unknown'}:{digest}"


def _rag_sort_key(item: RetrievedEvidence) -> float:
    """score 가 없으면 뒤로 — 안정 정렬이라 retriever 순서는 그대로 유지된다."""
    return -(item.score if item.score is not None else 0.0)


# ---------------------------------------------------------------------------
# 공개 API
# ---------------------------------------------------------------------------
def merge_evidence(
    rag_docs: list[RetrievedEvidence],
    web_docs: list[WebEvidence],
    required_topics: list[str] | None = None,
) -> EvidenceMergeResult:
    """RAG 근거와 검증된 웹 근거를 병합해 최종 근거 목록을 만든다(명세 16절).

    반환 순서 자체가 우선순위다 — RAG 가 항상 앞에 온다. 답변 생성기는 앞에서부터
    인용하면 되고, `final_evidence_max` 로 잘릴 때도 웹 근거가 먼저 떨어진다.
    """
    topics = [t.strip() for t in (required_topics or []) if isinstance(t, str) and t.strip()]

    # --- 1) RAG 근거 (최우선) -------------------------------------------
    ordered_rag = sorted(list(rag_docs or []), key=_rag_sort_key)
    rag_evidence: list[FinalEvidence] = []
    rag_topic_texts: dict[str, list[str]] = {}
    seen_chunks: set[str] = set()

    for doc in ordered_rag:
        if doc.chunk_id in seen_chunks:
            continue
        seen_chunks.add(doc.chunk_id)
        supported = _supported_topics(f"{doc.title}\n{doc.text}", topics)
        rag_evidence.append(
            FinalEvidence(
                evidence_id=doc.chunk_id,  # chunk_id 자체가 이미 안정적인 식별자
                source_type="rag",
                title=doc.title,
                source_url=doc.source_url,
                text=doc.text,
                supported_topics=supported,
            )
        )
        for topic in supported:
            rag_topic_texts.setdefault(topic, []).append(doc.text)
        rag_topic_texts.setdefault(_WHOLE_TOPIC, []).append(doc.text)

    # --- 2) 검증된 웹 근거 (보조) ----------------------------------------
    candidates = [item for item in (web_docs or []) if item.accepted]
    dropped = len(web_docs or []) - len(candidates)
    if dropped:
        logger.info("[merge] 검증을 통과하지 못한 웹 결과 %d건을 제외했습니다.", dropped)

    ordered_web = sorted(candidates, key=lambda i: -(i.score if i.score is not None else 0.0))
    web_evidence: list[FinalEvidence] = []
    web_topic_texts: dict[str, list[str]] = {}
    seen_urls: set[str] = set()

    for item in ordered_web:
        key = (item.url or "").strip().lower().rstrip("/")
        if key and key in seen_urls:
            continue
        if key:
            seen_urls.add(key)
        domain = item.domain or extract_domain(item.url)
        supported = _supported_topics(f"{item.title}\n{item.content}", topics)
        web_evidence.append(
            FinalEvidence(
                evidence_id=_web_evidence_id(item.url, domain, item.title),
                source_type="web",
                title=item.title or domain,
                source_url=item.url,
                text=item.content,
                supported_topics=supported,
            )
        )
        for topic in supported:
            web_topic_texts.setdefault(topic, []).append(item.content)
        web_topic_texts.setdefault(_WHOLE_TOPIC, []).append(item.content)

    # --- 3) 충돌 기록 -----------------------------------------------------
    conflicts = _detect_conflicts(rag_topic_texts, web_topic_texts, topics)

    # --- 4) 최종 목록 -----------------------------------------------------
    evidence = rag_evidence + web_evidence
    limit = get_settings().rag.final_evidence_max
    if limit and len(evidence) > limit:
        logger.info("[merge] 근거 %d건 중 상위 %d건만 사용합니다.", len(evidence), limit)
        evidence = evidence[:limit]

    return EvidenceMergeResult(
        evidence=evidence,
        conflicts=conflicts,
        # RAG 든 검증된 웹이든 하나라도 남았을 때만 True.
        # False 면 호출자는 "모른다 + 병원 상담 권고"로 답해야 한다.
        has_reliable_evidence=bool(evidence),
    )


def _detect_conflicts(
    rag_topic_texts: dict[str, list[str]],
    web_topic_texts: dict[str, list[str]],
    topics: list[str],
) -> list[str]:
    """같은 주제에서 RAG 와 웹의 입장이 반대면 충돌로 기록한다.

    양쪽에 근거가 다 있을 때만 비교한다. 웹에만 있는 내용은 충돌이 아니라
    보조 근거다(명세 16절).

    topic 단위 충돌이 하나라도 잡히면 `_WHOLE_TOPIC`(전체 내용) 비교는 하지
    않는다. 같은 충돌을 주제별/전체로 두 번 적어 놓으면 답변 생성기가 충돌이
    여러 건인 것처럼 읽는다. 전체 비교는 topic 이 없거나 topic 단위로는 아무것도
    못 잡았을 때의 안전망이다.
    """
    def compare(topic: str) -> list[str]:
        rag_texts = rag_topic_texts.get(topic) or []
        web_texts = web_topic_texts.get(topic) or []
        if not rag_texts or not web_texts:
            return []
        rag_stances = _stances(" ".join(rag_texts))
        web_stances = _stances(" ".join(web_texts))
        found: list[str] = []
        for left, right in _CONFLICTING_PAIRS:
            for rag_side, web_side in ((left, right), (right, left)):
                # web_side 가 RAG 쪽에도 있으면 단순 서술 차이일 뿐 모순이 아니다.
                if rag_side in rag_stances and web_side in web_stances and web_side not in rag_stances:
                    conservative = (
                        _CONSERVATIVE.get(rag_side) or _CONSERVATIVE.get(web_side) or rag_side
                    )
                    found.append(
                        f"'{topic}' 에 대해 Cornell 자료({_STANCE_LABELS[rag_side]})와 "
                        f"웹 자료({_STANCE_LABELS[web_side]})가 엇갈립니다. "
                        f"보수적으로 {_STANCE_LABELS[conservative]}을 기준으로 안내하고 "
                        f"수의사 확인을 권고해야 합니다."
                    )
                    break
        return found

    conflicts: list[str] = []
    for topic in topics:
        for message in compare(topic):
            if message not in conflicts:
                conflicts.append(message)

    if not conflicts:
        conflicts = compare(_WHOLE_TOPIC)
    return conflicts


__all__ = ["merge_evidence"]
