"""보호자 기록을 **질문과 관련된 것만** 골라내는 검색.

## 왜 필요한가

서버는 최근 30일 일일 기록을 전부 보낸다(실측 26건 · 약 2,400 토큰). 예전에는
그걸 통째로 그래프에 넣었는데, 두 가지 문제가 있었다.

1. **묻지 않은 것에 답하게 된다.** "예방접종은 언제 하나요?" 라고 물어도 지난달
   구토 기록이 프롬프트에 들어가서, 답변에 "최근 기력 저하와 구토가 있어..." 가
   섞여 나왔다. 실제로 그 화면이 나왔다.
2. 질문과 무관한 기록이 위험도 판정 재료로도 쓰인다.

## 왜 키워드가 아니라 임베딩인가

같은 상태를 보호자와 시스템이 다르게 적는다.

    보호자 질문 : "밥을 잘 안 먹어요"
    기록 필드   : "사료 반쯤 남김 · 평소보다 감소"

겹치는 글자가 없어서 부분 문자열 매칭으로는 못 잡는다. 점수 사전("남김"=40점
식으로)을 만들면 잡히지만, 표현이 바뀔 때마다 사전을 고쳐야 하고 한국어는
조사·어미 변형이 많아 그 싸움을 이길 수 없다(이 저장소에서 실제로 세 번 뚫렸다).

임베딩은 **의미가 가까우면 표기가 달라도** 잡는다. 수의학 문서를 이미 그렇게
찾고 있으므로, 개인 기록만 다른 방식을 쓸 이유가 없다.

## 비용

`text-embedding-3-small` 기준 기록 30건이 약 1,500 토큰(≈ $0.00003)이다. 질의는
1건. 프롬프트에서 줄이는 양이 더 크다.

## 실패해도 답변을 막지 않는다

임베딩 호출이 실패하면 **최근 기록 N건**으로 물러선다. 검색이 안 된다고 상담이
멈추면 안 되고, 최근 기록은 어떤 질문에도 최악은 아니다.
"""

from __future__ import annotations

import logging
import math
import threading
from typing import Any, Sequence

from .io_schemas import DailyRecord

logger = logging.getLogger(__name__)

__all__ = [
    "MAX_SELECTED_RECORDS",
    "MIN_RELEVANCE_MARGIN",
    "record_to_text",
    "select_relevant_records",
]

#: 그래프에 넘길 기록 최대 건수.
#:
#: 너무 적으면 추세(며칠째 이어지는지)를 볼 수 없고, 너무 많으면 애초에 선별하는
#: 의미가 없다. 8건이면 최근 일주일 추세를 담으면서 프롬프트가 무거워지지 않는다.
MAX_SELECTED_RECORDS = 8

#: 검색 결과를 믿을 최소 '여유'(최고 유사도 − 중앙값).
#:
#: 이 값보다 작으면 어느 기록도 특별히 관련 있지 않다는 뜻이라 선별을 포기하고
#: 최근 기록으로 돌아간다. 절대 유사도 기준을 쓰지 않는 이유는
#: `select_relevant_records` 안의 주석을 보라.
MIN_RELEVANCE_MARGIN = 0.05

_embeddings_lock = threading.Lock()
_embeddings: Any = None
_embeddings_failed = False


def record_to_text(record: DailyRecord) -> str:
    """기록 1건을 검색용 문장으로 만든다.

    날짜는 넣지 않는다 — "2026-07-16" 이 질문과 의미적으로 가까울 리 없는데
    모든 기록에 공통으로 들어가면 유사도 차이만 흐려진다.
    """
    parts = [
        str(getattr(record, field, "") or "").strip()
        for field in ("raw_text", "food", "water", "activity", "symptom", "stool", "vomit", "notes")
    ]
    return " ".join(part for part in parts if part)


def _get_embeddings() -> Any | None:
    """임베딩 객체를 준비한다(프로세스당 1회, 실패하면 이후 재시도하지 않는다).

    매 요청마다 재시도하면 키가 없는 환경에서 요청마다 예외 비용을 문다.
    """
    global _embeddings, _embeddings_failed
    if _embeddings is not None or _embeddings_failed:
        return _embeddings
    with _embeddings_lock:
        if _embeddings is not None or _embeddings_failed:
            return _embeddings
        try:
            # ai/.env → os.environ 을 **먼저** 적용한다. `_apply_env_overrides` 는
            # os.environ 을 읽을 뿐이라, 이 호출이 없으면 EMBEDDING_BACKEND 를 못 보고
            # 기본값(huggingface)으로 떨어져 "torch 미설치" 로 실패한다. 그래프보다
            # 먼저 호출되는 경로가 있어서 여기서도 보장해야 한다.
            from .config import load_provider_env
            from .health_check import _apply_env_overrides
            from petcare_ai.config import get_settings
            from petcare_ai.rag.embeddings import build_embeddings

            load_provider_env()
            _embeddings = build_embeddings(_apply_env_overrides(get_settings()).rag)
            logger.info("기록 선별용 임베딩을 준비했습니다.")
        except Exception as exc:  # noqa: BLE001 — 검색 실패가 상담을 막으면 안 된다
            logger.warning("기록 선별용 임베딩을 만들지 못했습니다(최근 기록으로 대체): %s", exc)
            _embeddings_failed = True
    return _embeddings


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """코사인 유사도. 한쪽이라도 영벡터면 0."""
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def select_relevant_records(
    user_message: str,
    records: Sequence[DailyRecord],
    limit: int = MAX_SELECTED_RECORDS,
) -> list[DailyRecord]:
    """질문과 의미적으로 가까운 기록만 고른다(날짜 오름차순 유지).

    반환 순서를 **원래 시간 순서로 되돌리는** 이유: 뒤쪽 node 와 PDF 가 기록을
    시간 순으로 읽어 추세를 본다. 유사도 순으로 넘기면 "며칠째" 판단이 깨진다.

    기록이 limit 이하이면 그대로 돌려준다 — 고를 필요가 없는데 임베딩을 호출하면
    돈과 시간만 쓴다.
    """
    items = list(records or [])
    if len(items) <= limit:
        return items

    query = str(user_message or "").strip()
    embeddings = _get_embeddings() if query else None
    if embeddings is None:
        logger.info("임베딩 없이 최근 기록 %d건으로 진행합니다.", limit)
        return items[-limit:]

    texts = [record_to_text(item) for item in items]
    try:
        doc_vectors = embeddings.embed_documents(texts)
        query_vector = embeddings.embed_query(query)
    except Exception as exc:  # noqa: BLE001
        logger.warning("기록 임베딩에 실패해 최근 기록으로 대체합니다: %s", exc)
        return items[-limit:]

    scored = [
        (_cosine(query_vector, vector), index)
        for index, vector in enumerate(doc_vectors)
        if index < len(items)
    ]
    # 유사도 내림차순, 동점이면 최신 우선(index 내림차순).
    scored.sort(key=lambda pair: (pair[0], pair[1]), reverse=True)

    # **절대 유사도가 아니라 '여유' 로 판단한다.**
    #
    # 코사인 값의 절대 크기는 질의 길이에 따라 크게 흔들린다(실측: "토했어요" 0.18 /
    # "어제 오후에 노란 토를 한 번 했고..." 0.49). 그래서 "0.3 이상이면 관련 있다"
    # 같은 고정 기준은 성립하지 않는다.
    #
    # 대신 **최고점이 중앙값보다 뚜렷이 높은가**를 본다. 일일 기록은 대부분 같은
    # 문장이라(실측 30건 중 26건 동일) 관련 기록이 없으면 모든 점수가 붙어 여유가
    # 0 에 수렴한다. 그때 상위 N건을 고르는 것은 사실상 무작위 선택이다.
    #
    # 실측 여유: "토했어요" +0.000(실패) / "구토를 했어요" +0.111(1위 적중)
    #           "예방접종은 언제 하나요?" +0.000(관련 기록 자체가 없음 — 정상)
    scores = sorted((score for score, _ in scored), reverse=True)
    median = scores[len(scores) // 2]
    margin = scores[0] - median

    if margin < MIN_RELEVANCE_MARGIN:
        logger.info(
            "기록 선별 보류(여유 %.3f < %.3f) — 최근 %d건으로 진행합니다.",
            margin,
            MIN_RELEVANCE_MARGIN,
            limit,
        )
        return items[-limit:]

    chosen = sorted(index for _, index in scored[:limit])
    logger.info(
        "기록 선별: %d건 → %d건 (최고 %.3f, 여유 %.3f)",
        len(items),
        len(chosen),
        scores[0],
        margin,
    )
    return [items[index] for index in chosen]
