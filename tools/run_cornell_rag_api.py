#!/usr/bin/env python3
"""Run the shared Cornell RAG HTTP API."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="팀 앱이 공용으로 호출할 Cornell RAG API를 실행합니다."
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--reload",
        action="store_true",
        help="개발 중 코드 변경 시 서버를 자동 재시작합니다.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        import uvicorn
    except ImportError:
        print(
            "오류: FastAPI 서버 도구가 없습니다. "
            "python -m pip install -r requirements-rag.txt 를 실행하세요.",
            file=sys.stderr,
        )
        return 1

    uvicorn.run(
        "petcare_rag.api:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
