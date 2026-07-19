"""기존 .db 파일의 제약 보정 테스트 (`app.services.schema_fix`).

이 프로젝트는 Alembic 없이 `create_all` 로 스키마를 만든다. `create_all` 은 이미
있는 테이블을 고치지 않으므로, 모델에서 `nullable=True` 로 바꿔도 예전에 만들어진
DB 는 `NOT NULL` 인 채로 남는다.

실제로 그래서 **병원 없이 응급 이메일 초안을 만들면 500** 이 났다. 보정이 없으면
"기능은 고쳤는데 내 기기에서만 안 된다" 가 된다.

여기서 고정하는 계약은 세 가지다.

    1. NOT NULL 이 풀린다 — 그래야 NULL 을 쓸 수 있다.
    2. **행이 하나도 사라지지 않는다** — 사용자 데이터가 든 파일을 만지는 코드다.
    3. 여러 번 실행해도 같다(멱등) — 서버는 기동할 때마다 이걸 부른다.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from sqlalchemy import create_engine, inspect, text

from app.services.schema_fix import apply_schema_fixes, relax_not_null

#: 보정 대상과 같은 모양의 옛 스키마(to_email 이 NOT NULL).
_LEGACY_DDL = """
CREATE TABLE emergency_emails (
    id INTEGER NOT NULL PRIMARY KEY,
    pet_id INTEGER NOT NULL,
    hospital_id INTEGER,
    to_email VARCHAR(100) NOT NULL,
    subject VARCHAR(200) NOT NULL,
    body TEXT NOT NULL,
    status VARCHAR(20) NOT NULL
)
"""


@pytest.fixture()
def legacy_engine(tmp_path: Path):
    """옛 제약 + 데이터가 든 SQLite 파일을 만든다."""
    engine = create_engine(f"sqlite:///{tmp_path / 'legacy.db'}")
    with engine.begin() as conn:
        conn.exec_driver_sql(_LEGACY_DDL)
        for index in range(3):
            conn.exec_driver_sql(
                "INSERT INTO emergency_emails "
                "(pet_id, hospital_id, to_email, subject, body, status) "
                f"VALUES (1, 1, 'er{index}@example.com', 's', 'b', 'draft')"
            )
    return engine


def _nullable(engine, table: str, column: str) -> bool:
    return next(
        c["nullable"] for c in inspect(engine).get_columns(table) if c["name"] == column
    )


def _rows(engine) -> list[tuple]:
    with engine.connect() as conn:
        return list(conn.execute(text("SELECT * FROM emergency_emails ORDER BY id")))


def test_not_null_을_풀고_데이터를_그대로_보존한다(legacy_engine) -> None:
    before = _rows(legacy_engine)
    assert _nullable(legacy_engine, "emergency_emails", "to_email") is False

    assert relax_not_null(legacy_engine, "emergency_emails", "to_email") is True

    assert _nullable(legacy_engine, "emergency_emails", "to_email") is True
    assert _rows(legacy_engine) == before, "제약만 바꾼다 — 한 행도 잃지 않는다."


def test_보정_후에는_NULL_을_쓸_수_있다(legacy_engine) -> None:
    """이게 애초에 보정을 하는 이유다 — 병원 미지정 초안."""
    apply_schema_fixes(legacy_engine)

    with legacy_engine.begin() as conn:
        conn.exec_driver_sql(
            "INSERT INTO emergency_emails "
            "(pet_id, hospital_id, to_email, subject, body, status) "
            "VALUES (1, NULL, NULL, 's', 'b', 'draft')"
        )

    with legacy_engine.connect() as conn:
        stored = conn.execute(
            text("SELECT to_email FROM emergency_emails WHERE subject='s' AND body='b'")
        ).fetchall()
    assert (None,) in stored


def test_여러_번_실행해도_결과가_같다(legacy_engine) -> None:
    apply_schema_fixes(legacy_engine)
    after_first = _rows(legacy_engine)

    apply_schema_fixes(legacy_engine)
    apply_schema_fixes(legacy_engine)

    assert _nullable(legacy_engine, "emergency_emails", "to_email") is True
    assert _rows(legacy_engine) == after_first


def test_이미_nullable_이면_아무것도_하지_않는다(tmp_path: Path) -> None:
    engine = create_engine(f"sqlite:///{tmp_path / 'new.db'}")
    with engine.begin() as conn:
        conn.exec_driver_sql(
            "CREATE TABLE emergency_emails (id INTEGER PRIMARY KEY, to_email VARCHAR(100))"
        )
    assert relax_not_null(engine, "emergency_emails", "to_email") is False


def test_없는_테이블은_건너뛴다(tmp_path: Path) -> None:
    engine = create_engine(f"sqlite:///{tmp_path / 'empty.db'}")
    assert relax_not_null(engine, "emergency_emails", "to_email") is False
    apply_schema_fixes(engine)  # 예외 없이 지나가야 한다


def test_같은_이름으로_시작하는_다른_컬럼은_건드리지_않는다(tmp_path: Path) -> None:
    """`to_email` 을 찾다가 `to_email_backup` 의 NOT NULL 을 풀면 안 된다."""
    engine = create_engine(f"sqlite:///{tmp_path / 'similar.db'}")
    with engine.begin() as conn:
        conn.exec_driver_sql(
            "CREATE TABLE emergency_emails ("
            "id INTEGER PRIMARY KEY, "
            "to_email VARCHAR(100) NOT NULL, "
            "to_email_backup VARCHAR(100) NOT NULL)"
        )

    relax_not_null(engine, "emergency_emails", "to_email")

    assert _nullable(engine, "emergency_emails", "to_email") is True
    assert _nullable(engine, "emergency_emails", "to_email_backup") is False
