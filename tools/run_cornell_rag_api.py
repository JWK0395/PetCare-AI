#!/usr/bin/env python3
"""Run the local Cornell RAG FastAPI app."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from _project_env import load_project_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_project_dotenv(ROOT)

SRC = ROOT / "ai" / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run PetCare Cornell RAG API.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--reload", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        import uvicorn
    except ImportError:
        print("error: uvicorn is not installed", file=sys.stderr)
        return 1
    uvicorn.run("petcare_rag.api:app", host=args.host, port=args.port, reload=args.reload)
    return 0


if __name__ == "__main__":
    sys.exit(main())