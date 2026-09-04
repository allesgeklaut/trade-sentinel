"""Tests for the SQLite engine factory and pragma tuning in app.db.

Covers the concurrency/durability pragmas (WAL, synchronous=NORMAL,
foreign_keys=ON, busy_timeout) applied by ``create_db_engine``.
"""

from __future__ import annotations

import pytest
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError

from app import db as db_mod
from app.db import _should_apply_sqlite_pragmas, create_db_engine


async def test_pragmas_on_memory_engine():
    """Pragmas are applied on connect for an in-memory engine.

    journal_mode on :memory: reports "memory" (WAL needs a file), so it is
    not asserted here.
    """
    engine = create_db_engine("sqlite+aiosqlite:///:memory:")
    try:
        async with engine.connect() as c:
            timeout = (await c.execute(text("PRAGMA busy_timeout"))).scalar_one()
            fks = (await c.execute(text("PRAGMA foreign_keys"))).scalar_one()
            sync = (await c.execute(text("PRAGMA synchronous"))).scalar_one()
    finally:
        await engine.dispose()

    assert timeout == 5000
    assert fks == 1
    assert sync == 1  # NORMAL


async def test_wal_on_file_engine(tmp_path):
    """WAL mode is active on a file-backed database."""
    engine = create_db_engine(f"sqlite+aiosqlite:///{tmp_path}/t.db")
    try:
        async with engine.connect() as c:
            mode = (await c.execute(text("PRAGMA journal_mode"))).scalar_one()
    finally:
        await engine.dispose()

    assert mode == "wal"


def test_should_apply_sqlite_pragmas():
    assert _should_apply_sqlite_pragmas("sqlite+aiosqlite:////data/trading.db")
    assert _should_apply_sqlite_pragmas("sqlite:///:memory:")
    assert not _should_apply_sqlite_pragmas("postgresql+asyncpg://user:pass@host/db")
    assert not _should_apply_sqlite_pragmas("")


async def test_fk_enforcement_active(tmp_path):
    """foreign_keys=ON means raw DDL FKs are actually enforced."""
    engine = create_db_engine(f"sqlite+aiosqlite:///{tmp_path}/fk.db")
    try:
        async with engine.begin() as c:
            await c.execute(text("CREATE TABLE parent (id INTEGER PRIMARY KEY)"))
            await c.execute(
                text(
                    "CREATE TABLE child ("
                    "id INTEGER PRIMARY KEY, "
                    "parent_id INTEGER REFERENCES parent(id))"
                )
            )
            with pytest.raises(IntegrityError):
                await c.execute(text("INSERT INTO child (id, parent_id) VALUES (1, 999)"))
    finally:
        await engine.dispose()


def test_app_engine_is_sqlite():
    """Import-level check only — do not connect to the real /data engine."""
    assert db_mod.engine.url.drivername == "sqlite+aiosqlite"
