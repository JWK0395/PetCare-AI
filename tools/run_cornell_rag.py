#!/usr/bin/env python3
"""Run a Cornell RAG answer from the local vendored runtime."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Sequence

from _project_env import load_project_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_project_dotenv(ROOT)

SRC = ROOT / "ai" / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from petcare_rag import RagPipelineError, answer_question, run_pipeline
from petcare_rag.manage_cornell_rag_db import DEFAULT_COLLECTION, DEFAULT_DB_PATH


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Cornell official-source RAG.")
    parser.add_argument("--species", choices=("dog", "cat"), required=True)
    parser.add_argument("--question", required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--debug", action="store_true")
    return parser


def _runtime_kwargs() -> dict[str, object]:
    return {
        "db_path": Path(os.environ.get("PETCARE_RAG_DB_PATH", str(DEFAULT_DB_PATH))),
        "collection_name": os.environ.get("PETCARE_RAG_COLLECTION", DEFAULT_COLLECTION),
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    kwargs = _runtime_kwargs()
    try:
        if args.debug:
            response, trace = run_pipeline(args.question, args.species, args.top_k, **kwargs)
            print("===== DEBUG: retrieved chunks =====", file=sys.stderr)
            for chunk in trace.retrieved_chunks:
                print(f"{chunk.chunk_id} | {chunk.title} | {chunk.similarity:.4f}", file=sys.stderr)
        else:
            response = answer_question(args.question, args.species, args.top_k, **kwargs)
        if args.json:
            print(json.dumps(response.to_dict(), ensure_ascii=False, indent=2))
        else:
            print(response.answer)
            if response.citations:
                print("\nCornell sources")
                for citation in response.citations:
                    print(f"[{citation.number}] {citation.title} - {citation.url}")
            print(f"\nNotice: {response.disclaimer}")
        return 0
    except RagPipelineError as exc:
        print(f"error: {exc}", file=sys.stderr)
        diagnostic = getattr(exc, "diagnostic", None)
        if args.debug and diagnostic:
            print(f"diagnostic: {diagnostic}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())