"""Tests for the PROD/DEV database switch (app.db + app.main routing).

Covered here:

- the ``_active_db`` ContextVar (:func:`use_db`, :func:`current_db`)
- :func:`Session` dispatching on the contextvar
- the ``_require_prod_db`` / ``_require_dev_db`` guard helpers
- the ``ts_db`` cookie middleware routing real ASGI requests end-to-end
- :func:`clone_prod_to_dev` (backup, atomic swap, stale sidecars)
- per-DB daily-core state files
- :func:`init_db` migrating every configured engine

All DBs are file-backed SQLite files under ``tmp_path``; ``app.db`` module
attributes are monkeypatched, so nothing touches the real data volume.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from fastapi import HTTPException
from sqlalchemy.ext.asyncio import async_sessionmaker

from app import daily_core, main
from app import db as db_mod
from app.db import Base, Watchlist, current_db, use_db


def _engine_for(path: Path):
    """File-backed async engine built through the app's own factory."""
    return db_mod.create_db_engine(f"sqlite+aiosqlite:///{path}")


async def _create_schema(engine) -> None:
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


def _tickers(path: Path) -> set[str]:
    """All watchlist tickers in a SQLite file, via a raw fresh connection."""
    conn = sqlite3.connect(path)
    try:
        return {row[0] for row in conn.execute("SELECT ticker FROM watchlist")}
    finally:
        conn.close()  # `with sqlite3.connect` commits but does NOT close


def _has_table(path: Path, name: str) -> bool:
    conn = sqlite3.connect(path)
    try:
        row = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)
        ).fetchone()
    finally:
        conn.close()
    return row is not None


@pytest.fixture
async def two_file_dbs(monkeypatch, tmp_path):
    """Two file-backed DBs wired into app.db's engine + sessionmaker dict."""
    prod_path = tmp_path / "prod.sqlite"
    dev_path = tmp_path / "dev.sqlite"
    eng_prod, eng_dev = _engine_for(prod_path), _engine_for(dev_path)
    await _create_schema(eng_prod)
    await _create_schema(eng_dev)
    monkeypatch.setattr(db_mod, "engine", eng_prod)
    monkeypatch.setattr(db_mod, "engine_dev", eng_dev)
    monkeypatch.setitem(db_mod._sessionmakers, "prod",
                        async_sessionmaker(eng_prod, expire_on_commit=False))
    monkeypatch.setitem(db_mod._sessionmakers, "dev",
                        async_sessionmaker(eng_dev, expire_on_commit=False))
    yield SimpleNamespace(prod_path=prod_path, dev_path=dev_path,
                          eng_prod=eng_prod, eng_dev=eng_dev)
    await eng_prod.dispose()
    await eng_dev.dispose()


@pytest.fixture
async def dev_cookie_app(monkeypatch, tmp_path):
    """A live DEV engine with schema so the ts_db cookie middleware engages.

    ``app.main`` imported ``engine_dev`` by value, so both module names must be
    patched for the middleware/status endpoint to see a non-None dev engine.
    """
    dev_path = tmp_path / "dev.sqlite"
    eng_dev = _engine_for(dev_path)
    await _create_schema(eng_dev)
    monkeypatch.setattr(db_mod, "engine_dev", eng_dev)
    monkeypatch.setattr(main, "engine_dev", eng_dev)
    monkeypatch.setitem(db_mod._sessionmakers, "dev",
                        async_sessionmaker(eng_dev, expire_on_commit=False))
    yield SimpleNamespace(path=dev_path, engine=eng_dev)
    await eng_dev.dispose()


class TestContextVar:
    def test_default_is_prod(self):
        assert current_db() == "prod"

    def test_switches_and_resets_on_exit(self):
        with use_db("dev"):
            assert current_db() == "dev"
        assert current_db() == "prod"

    def test_resets_after_exception_inside_the_block(self):
        with pytest.raises(RuntimeError, match="boom"):
            with use_db("dev"):
                assert current_db() == "dev"
                raise RuntimeError("boom")
        assert current_db() == "prod"

    def test_unknown_name_raises_valueerror(self):
        with pytest.raises(ValueError, match="unknown db name"):
            with use_db("bogus"):
                pass
        assert current_db() == "prod"


class TestSessionDispatch:
    async def test_session_lands_in_the_active_db_file(self, two_file_dbs):
        with use_db("prod"):
            async with db_mod.Session() as s:
                s.add(Watchlist(ticker="PROD1"))
                await s.commit()
        with use_db("dev"):
            async with db_mod.Session() as s:
                s.add(Watchlist(ticker="DEV1"))
                await s.commit()

        # Cross-isolation: each row exists only in its own file.
        assert _tickers(two_file_dbs.prod_path) == {"PROD1"}
        assert _tickers(two_file_dbs.dev_path) == {"DEV1"}

    async def test_missing_factory_raises_runtime_error(self, monkeypatch):
        # A contextvar name without a sessionmaker entry (dev engine disabled).
        monkeypatch.delitem(db_mod._sessionmakers, "dev", raising=False)
        with use_db("dev"), pytest.raises(RuntimeError, match="no session factory"):
            db_mod.Session()


class TestGuardHelpers:
    def test_require_dev_db_403s_on_prod(self):
        with pytest.raises(HTTPException) as ei:
            main._require_dev_db("X")
        assert ei.value.status_code == 403
        with use_db("dev"):
            main._require_dev_db("X")  # no raise on dev

    def test_require_prod_db_403s_on_dev(self):
        main._require_prod_db("X")  # no raise on prod (default)
        with use_db("dev"):
            with pytest.raises(HTTPException) as ei:
                main._require_prod_db("X")
            assert ei.value.status_code == 403


class TestCookieRouting:
    async def test_status_reports_the_cookie_selected_db(self, dev_cookie_app):
        async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=main.app),
                base_url="http://test") as client:
            r = await client.get("/api/db/status")
            assert r.status_code == 200
            body = r.json()
            assert body["active"] == "prod"
            assert body["dev_enabled"] is True

            client.cookies.set("ts_db", "dev")
            r = await client.get("/api/db/status")
            assert r.status_code == 200
            body = r.json()
            assert body["active"] == "dev"
            assert body["dev_enabled"] is True
            assert body["dev_seeded"] is True

    async def test_guarded_endpoint_respects_the_cookie(self, dev_cookie_app):
        async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=main.app),
                base_url="http://test") as client:
            r = await client.post("/api/sim/reset")
            assert r.status_code == 403  # reset is DEV-only

            client.cookies.set("ts_db", "dev")
            r = await client.post("/api/sim/reset")
            # The guard passed (not 403) and the dev fixture has the schema,
            # so the reset actually ran on the DEV file.
            assert r.status_code == 200
            with sqlite3.connect(dev_cookie_app.path) as conn:
                count = conn.execute("SELECT COUNT(*) FROM sim_account").fetchone()[0]
            assert count == 1


class TestCloneProdToDev:
    async def test_clone_swaps_dev_and_prod_stays_independent(self, monkeypatch, tmp_path):
        prod_path = tmp_path / "prod.sqlite"
        dev_path = tmp_path / "dev.sqlite"
        eng_prod, eng_dev = _engine_for(prod_path), _engine_for(dev_path)
        await _create_schema(eng_prod)
        monkeypatch.setattr(db_mod, "engine", eng_prod)
        monkeypatch.setattr(db_mod, "engine_dev", eng_dev)
        sm_prod = async_sessionmaker(eng_prod, expire_on_commit=False)
        sm_dev = async_sessionmaker(eng_dev, expire_on_commit=False)

        async with sm_prod() as s:
            s.add(Watchlist(ticker="PROD1"))
            await s.commit()

        # Sidecars a crashed process could have left behind.
        Path(f"{dev_path}-wal").write_text("stale-wal")
        Path(f"{dev_path}-shm").write_text("stale-shm")

        r = await db_mod.clone_prod_to_dev()
        assert r["ok"] is True
        assert r["path"] == str(dev_path)
        assert r["size_bytes"] > 0
        datetime.fromisoformat(r["cloned_at"])  # parses as ISO
        assert not Path(f"{dev_path}-wal").exists()
        assert not Path(f"{dev_path}-shm").exists()
        assert _tickers(dev_path) == {"PROD1"}  # dev is now the prod copy

        # Prod keeps living independently: a row written after the clone must
        # not appear in dev until the next clone.
        async with sm_prod() as s:
            s.add(Watchlist(ticker="PROD2"))
            await s.commit()
        assert _tickers(prod_path) == {"PROD1", "PROD2"}
        assert _tickers(dev_path) == {"PROD1"}

        # A dev-only row is wiped by the next clone (dev is replaceable).
        async with sm_dev() as s:
            s.add(Watchlist(ticker="DEVONLY"))
            await s.commit()
        assert _tickers(dev_path) == {"PROD1", "DEVONLY"}
        r2 = await db_mod.clone_prod_to_dev()
        assert r2["ok"] is True
        assert _tickers(dev_path) == {"PROD1", "PROD2"}
        assert _tickers(prod_path) == {"PROD1", "PROD2"}

        # No dev engine configured -> refuse, nothing to clone into.
        monkeypatch.setattr(db_mod, "engine_dev", None)
        with pytest.raises(RuntimeError, match="DATABASE_URL_DEV"):
            await db_mod.clone_prod_to_dev()

        await eng_prod.dispose()
        await eng_dev.dispose()


class TestDailyCoreStateRouting:
    def test_state_files_are_per_db(self, monkeypatch, tmp_path):
        prod_state = tmp_path / "daily_core_state.json"
        dev_state = tmp_path / "daily_core_state_dev.json"
        monkeypatch.setattr(daily_core, "_STATE_FILE", prod_state)
        monkeypatch.setattr(daily_core, "_STATE_FILE_DEV", dev_state)

        with use_db("prod"):
            daily_core._save_state({"mom_variant": "raw"})
        assert prod_state.exists()
        assert not dev_state.exists()

        with use_db("dev"):
            assert daily_core._load_state() == {}  # dev file not written yet
            daily_core._save_state({"mom_variant": "residual"})
        assert json.loads(dev_state.read_text()) == {"mom_variant": "residual"}

        # The dev write never touched prod, and prod reads stay on prod.
        assert json.loads(prod_state.read_text()) == {"mom_variant": "raw"}
        assert daily_core._load_state() == {"mom_variant": "raw"}


class TestInitDb:
    async def test_creates_schema_on_every_configured_engine(self, monkeypatch, tmp_path):
        prod_path = tmp_path / "prod.sqlite"
        dev_path = tmp_path / "dev.sqlite"
        eng_prod, eng_dev = _engine_for(prod_path), _engine_for(dev_path)
        monkeypatch.setattr(db_mod, "engine", eng_prod)
        monkeypatch.setattr(db_mod, "engine_dev", eng_dev)

        await db_mod.init_db()

        assert _has_table(prod_path, "watchlist")
        assert _has_table(dev_path, "watchlist")

        await eng_prod.dispose()
        await eng_dev.dispose()
