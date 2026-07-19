"""기존 SQLite 파일의 제약을 새 모델에 맞추는 최소 보정 (기동 시 1회).

## 왜 필요한가

이 프로젝트는 Alembic 을 쓰지 않고 `Base.metadata.create_all` 로 스키마를 만든다.
`create_all` 은 **없는 테이블만** 만들 뿐, 이미 있는 테이블의 컬럼 제약은 건드리지
않는다. 그래서 모델에서 `nullable=True` 로 바꿔도 예전에 만들어진 .db 파일은
`NOT NULL` 인 채로 남고, 그 컬럼에 NULL 을 쓰는 순간 IntegrityError(500) 가 난다.

실제로 그 일이 있었다. `emergency_emails.to_email` 은 병원 이메일을 못 구했을 때
NULL 이어야 하는데(응급 이메일은 병원 없이도 초안이 나와야 한다), 예전 DB 에서는
NOT NULL 이라 "병원 없이 초안 만들기" 가 500 으로 실패한다.

## 이 모듈이 하는 일과 하지 않는 일

- **한다**: NOT NULL → NULL 허용으로 **완화**. 데이터는 한 행도 잃지 않는다.
- **하지 않는다**: 행 삭제, 컬럼 삭제, 타입 변경, 제약 강화. 되돌릴 수 없거나
  데이터를 잃을 수 있는 변경은 여기서 하지 않는다. 그런 변경이 필요해지면 그때는
  사람이 백업을 확인하고 실행하는 별도 스크립트로 다뤄야 한다.

이미 맞는 스키마라면 아무 일도 하지 않는다(멱등). SQLite 가 아니면 건너뛴다.
"""

from __future__ import annotations

import logging

from sqlalchemy import inspect, text
from sqlalchemy.engine import Engine

logger = logging.getLogger(__name__)

__all__ = ["relax_not_null", "apply_schema_fixes"]

#: (테이블, 컬럼) — 모델에서 nullable 로 바뀌었으나 옛 DB 에는 NOT NULL 로 남아
#: 있을 수 있는 자리. 완화만 하므로 목록에 없던 항목을 빠뜨려도 손해는 없다.
_RELAXED_COLUMNS: tuple[tuple[str, str], ...] = (("emergency_emails", "to_email"),)


def relax_not_null(engine: Engine, table: str, column: str) -> bool:
    """`table.column` 의 NOT NULL 을 푼다. 바꿨으면 True.

    SQLite 는 `ALTER COLUMN` 이 없어 테이블을 다시 만들어야 한다. 순서가 중요하다.

        1. 새 정의로 임시 테이블 생성
        2. 데이터 복사
        3. 원본 삭제 → 임시 테이블 rename

    전부 한 트랜잭션 안에서 한다. 중간에 죽으면 롤백되어 원본이 남는다.
    `legacy_alter_table=ON` 은 rename 시 다른 객체의 참조가 따라 바뀌지 않게 한다.
    """
    inspector = inspect(engine)
    if table not in inspector.get_table_names():
        return False

    columns = inspector.get_columns(table)
    target = next((c for c in columns if c["name"] == column), None)
    if target is None or target.get("nullable", True):
        return False  # 없거나 이미 nullable — 할 일 없음

    # 원본 DDL 에서 해당 컬럼의 NOT NULL 만 지운 정의를 만든다.
    with engine.connect() as conn:
        ddl_row = conn.execute(
            text("SELECT sql FROM sqlite_master WHERE type='table' AND name=:name"),
            {"name": table},
        ).fetchone()
    if not ddl_row or not ddl_row[0]:
        logger.warning("%s 의 정의를 읽지 못해 제약 완화를 건너뜁니다.", table)
        return False

    original_ddl: str = ddl_row[0]
    temp_table = f"{table}__schema_fix"
    patched_ddl = _drop_not_null(original_ddl, column)
    if patched_ddl == original_ddl:
        logger.warning("%s.%s 의 NOT NULL 을 정의에서 찾지 못했습니다.", table, column)
        return False
    patched_ddl = patched_ddl.replace(f'"{table}"', f'"{temp_table}"', 1)
    patched_ddl = patched_ddl.replace(f" {table} ", f" {temp_table} ", 1)

    names = ", ".join(f'"{c["name"]}"' for c in columns)
    with engine.begin() as conn:
        conn.exec_driver_sql("PRAGMA legacy_alter_table=ON")
        conn.exec_driver_sql(patched_ddl)
        conn.exec_driver_sql(
            f'INSERT INTO "{temp_table}" ({names}) SELECT {names} FROM "{table}"'
        )
        conn.exec_driver_sql(f'DROP TABLE "{table}"')
        conn.exec_driver_sql(f'ALTER TABLE "{temp_table}" RENAME TO "{table}"')
        conn.exec_driver_sql("PRAGMA legacy_alter_table=OFF")

    logger.info("%s.%s 의 NOT NULL 제약을 해제했습니다(데이터 보존).", table, column)
    return True


def _drop_not_null(ddl: str, column: str) -> str:
    """CREATE TABLE 문에서 특정 컬럼의 `NOT NULL` 만 제거한다.

    컬럼 정의는 쉼표로 나뉘고 이름이 맨 앞에 온다. 이름이 정확히 일치하는 조각에서만
    `NOT NULL` 을 지운다 — `to_email` 을 찾다가 `to_email_backup` 을 건드리면 안 된다.
    """
    head, sep, body = ddl.partition("(")
    if not sep:
        return ddl

    parts = body.rsplit(")", 1)
    inner, tail = parts[0], (")" + parts[1] if len(parts) > 1 else ")")

    patched: list[str] = []
    for piece in inner.split(","):
        name = piece.strip().split()[0].strip('"`[]') if piece.strip() else ""
        if name == column:
            piece = piece.replace(" NOT NULL", "")
        patched.append(piece)
    return f"{head}{sep}{','.join(patched)}{tail}"


def apply_schema_fixes(engine: Engine) -> None:
    """기동 시 호출 — 실패해도 서버는 뜬다.

    보정에 실패하는 것보다 서버가 아예 안 뜨는 쪽이 나쁘다. 실패는 로그로 남기고
    넘어간다. 그 경우 해당 기능만 예전처럼 동작한다(500).
    """
    if engine.dialect.name != "sqlite":
        return
    for table, column in _RELAXED_COLUMNS:
        try:
            relax_not_null(engine, table, column)
        except Exception as exc:  # noqa: BLE001 — 기동을 막지 않는다
            logger.warning("%s.%s 제약 완화 실패(무시하고 진행): %s", table, column, exc)
