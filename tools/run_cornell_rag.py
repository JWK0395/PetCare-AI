#!/usr/bin/env python3
"""Beginner-friendly CLI for the Cornell RAG answer pipeline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from petcare_rag import RagPipelineError, run_pipeline  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Cornell 공식 자료를 검색하고, 검색된 근거만 사용해 한국어 답변을 만듭니다."
        )
    )
    parser.add_argument("--question", required=True, help="일반 건강정보 질문")
    parser.add_argument("--species", required=True, choices=("dog", "cat"))
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--db-path", type=Path, default=Path("rag_data/chroma"))
    parser.add_argument(
        "--collection", default="cornell_pet_health_text_embedding_3_small_1536"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="벡터 값이나 API 키를 제외한 중간 검색·컨텍스트 정보를 표시합니다.",
    )
    parser.add_argument(
        "--hybrid-rerank",
        action="store_true",
        help="dense 후보를 더 넓게 가져온 뒤 BM25 스타일 lexical score로 실험적 rerank를 적용합니다.",
    )
    parser.add_argument(
        "--json", action="store_true", help="최종 결과를 JSON으로 출력합니다."
    )
    return parser


def print_debug(trace: object) -> None:
    print("\n===== DEBUG: 원문 질문 =====", file=sys.stderr)
    print(trace.original_question, file=sys.stderr)
    print("\n===== DEBUG: 검색 질의 =====", file=sys.stderr)
    print(trace.retrieval_query, file=sys.stderr)
    if trace.query_rewrite_failed:
        print("query rewrite 실패: 원문 질문으로 검색했습니다.", file=sys.stderr)
    print("\n===== DEBUG: 질문 임베딩 프롬프트 =====", file=sys.stderr)
    print(trace.embedding_prompt, file=sys.stderr)
    print("\n===== DEBUG: 검색 순위 =====", file=sys.stderr)
    for rank, chunk in enumerate(trace.retrieved_chunks, start=1):
        print(
            f"{rank}. similarity={chunk.similarity:.4f} "
            f"document_id={chunk.document_id} chunk_id={chunk.chunk_id}",
            file=sys.stderr,
        )
    print("\n===== DEBUG: OpenAI에 전달되는 SOURCE 컨텍스트 =====", file=sys.stderr)
    print(trace.context or "(유효한 컨텍스트 없음)", file=sys.stderr)
    print("\n===== DEBUG: 최종 인용 번호 =====", file=sys.stderr)
    print(trace.cited_source_numbers, file=sys.stderr)


def print_human_response(response: object) -> None:
    print("\n답변")
    print(response.answer)
    if response.citations:
        print("\nCornell 공식 출처")
        for citation in response.citations:
            section = " > ".join(citation.section_path)
            print(f"[{citation.number}] {citation.title}")
            if section:
                print(f"    섹션: {section}")
            print(f"    {citation.url}")
            print(f"    chunk_id: {citation.chunk_id}")
    else:
        print("\n인용할 수 있는 충분한 근거가 없습니다.")
    print(f"\n안내: {response.disclaimer}")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        response, trace = run_pipeline(
            question=args.question,
            species=args.species,
            top_k=args.top_k,
            db_path=args.db_path,
            collection_name=args.collection,
            hybrid_rerank_enabled=args.hybrid_rerank,
        )
        if args.debug:
            print_debug(trace)
        if args.json:
            print(json.dumps(response.to_dict(), ensure_ascii=False, indent=2))
        else:
            print_human_response(response)
        return 0
    except RagPipelineError as exc:
        print(f"오류: {exc}", file=sys.stderr)
        if args.debug and exc.diagnostic:
            print(f"진단 정보: {exc.diagnostic}", file=sys.stderr)
        return 1
    except Exception:
        print(
            "오류: 예상하지 못한 문제가 발생했습니다. 설정과 DB 상태를 확인하세요.",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
