#!/usr/bin/env python3
"""Manage the local Cornell RAG Chroma index."""

from __future__ import annotations

import sys
from pathlib import Path

from _project_env import load_project_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_project_dotenv(ROOT)

SRC = ROOT / "ai" / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from petcare_rag.manage_cornell_rag_db import main


if __name__ == "__main__":
    sys.exit(main())