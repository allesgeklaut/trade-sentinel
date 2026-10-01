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

import asyncio
import json
import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from fastapi import HTTPException
from sqlalchemy.ext.asyncio import async_sessionmaker

from app import daily_core, main, monthly, sim
from app import db as db_mod
from app.config import settings
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
    """A live DEV engine (and an isolated PROD file) so the ts_db cookie
    middleware engages.

    ``app.main`` imported ``engine_dev`` by value, so both module names must be
    patched for the middleware/status endpoint to see a non-None dev engine.
    PROD gets its own temp file too, so a DEV-routed write can be checked
    against a real PROD file instead of the host's /data volume.
    """
    prod_path = tmp_path / "prod.sqlite"
    dev_path = tmp_path / "dev.sqlite"
    eng_prod, eng_dev = _engine_for(prod_path), _engine_for(dev_path)
    await _create_schema(eng_prod)
    await _create_schema(eng_dev)
    monkeypatch.setattr(db_mod, "engine", eng_prod)
    monkeypatch.setattr(db_mod, "engine_dev", eng_dev)
    monkeypatch.setattr(main, "engine_dev", eng_dev)
    monkeypatch.setitem(db_mod._sessionmakers, "prod",
                        async_sessionmaker(eng_prod, expire_on_commit=False))
    monkeypatch.setitem(db_mod._sessionmakers, "dev",
                        async_sessionmaker(eng_dev, expire_on_commit=False))
    yield SimpleNamespace(prod_path=prod_path, path=dev_path, engine=eng_dev,
                          eng_prod=eng_prod)
    await eng_prod.dispose()
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


class TestDevEngineFactory:
    """`_create_dev_engine` refuses unsafe DEV configs (M3) instead of building
    an engine a clone could use to overwrite PROD."""

    def test_empty_url_disables_dev_silently(self):
        assert db_mod._create_dev_engine(db_mod.engine, "") is None

    async def test_non_file_url_disables_dev(self, tmp_path):
        prod = _engine_for(tmp_path / "prod.sqlite")
        try:
            assert db_mod._create_dev_engine(prod, "sqlite+aiosqlite:///:memory:") is None
        finally:
            await prod.dispose()

    async def test_dev_path_equal_to_prod_disables_dev(self, tmp_path):
        prod_path = tmp_path / "prod.sqlite"
        prod = _engine_for(prod_path)
        try:
            # Same file through a non-normalized path: resolve() must catch it.
            same = tmp_path / "sub" / ".." / "prod.sqlite"
            assert db_mod._create_dev_engine(prod, f"sqlite+aiosqlite:///{same}") is None
        finally:
            await prod.dispose()

    async def test_distinct_path_builds_a_real_engine(self, tmp_path):
        prod_path, dev_path = tmp_path / "prod.sqlite", tmp_path / "dev.sqlite"
        prod = _engine_for(prod_path)
        dev = db_mod._create_dev_engine(prod, f"sqlite+aiosqlite:///{dev_path}")
        try:
            assert dev is not None
            assert db_mod._sqlite_path(dev) == dev_path
        finally:
            await dev.dispose()
            await prod.dispose()


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
            # The fixture's DEV file exists but holds no watchlist rows, so it
            # is not "seeded" — dev_seeded tracks data, not bare file existence.
            assert body["dev_seeded"] is False

    async def test_cookie_dev_falls_through_when_dev_is_disabled(self, monkeypatch):
        """A ts_db=dev cookie with no dev engine must stay on PROD (the
        disabled fallthrough), not error out."""
        monkeypatch.setattr(db_mod, "engine_dev", None)
        monkeypatch.setattr(main, "engine_dev", None)
        async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=main.app),
                base_url="http://test") as client:
            client.cookies.set("ts_db", "dev")
            r = await client.get("/api/db/status")
            assert r.status_code == 200
            body = r.json()
            assert body["active"] == "prod"
            assert body["dev_enabled"] is False

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
            # The DEV write must never land in PROD: PROD's own sim_account
            # (schema exists in the fixture) still has no row.
            with sqlite3.connect(dev_cookie_app.prod_path) as conn:
                prod_count = conn.execute("SELECT COUNT(*) FROM sim_account").fetchone()[0]
            assert prod_count == 0


class TestDevIsEmpty:
    """`_dev_is_empty` (the auto-seed / dev_seeded signal) must query DEV, not
    the lifespan's default PROD context."""

    async def test_true_for_a_fresh_schema(self, dev_cookie_app):
        assert await main._dev_is_empty() is True

    async def test_false_once_dev_has_a_watchlist_row(self, dev_cookie_app):
        with use_db("dev"):
            async with db_mod.Session() as s:
                s.add(Watchlist(ticker="DEV1"))
                await s.commit()
        assert await main._dev_is_empty() is False

    async def test_status_dev_seeded_tracks_data_not_bare_existence(self, dev_cookie_app):
        async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=main.app),
                base_url="http://test") as client:
            client.cookies.set("ts_db", "dev")
            r = await client.get("/api/db/status")
            assert r.json()["dev_seeded"] is False  # file exists, but empty
            with use_db("dev"):
                async with db_mod.Session() as s:
                    s.add(Watchlist(ticker="DEV1"))
                    await s.commit()
            r = await client.get("/api/db/status")
            assert r.json()["dev_seeded"] is True


class TestCloneMutualExclusion:
    """M1: a running clone and the individual DEV backfills/runs exclude each
    other — both directions, DEV-routed only for the individual ops."""

    DEV_BACKFILLS = ("/api/sim/backfill", "/api/sim/backfill-benchmark",
                     "/api/monthly/backfill", "/api/dailycore/backfill")
    DEV_RUNS = ("/api/sim/run", "/api/monthly/run", "/api/dailycore/run")

    async def test_dev_ops_409_while_a_clone_runs(self, dev_cookie_app, monkeypatch):
        monkeypatch.setattr(main, "_db_cloning", True)
        async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=main.app),
                base_url="http://test") as client:
            client.cookies.set("ts_db", "dev")
            for path in self.DEV_BACKFILLS + self.DEV_RUNS:
                r = await client.post(path)
                assert r.status_code == 409, f"{path}: {r.text}"
                assert "clone is running" in r.json()["detail"]

    async def test_prod_requests_ignore_the_clone_flag(self, monkeypatch):
        """A clone swaps only the DEV file: PROD requests never hit the guard.
        (The runs are stubbed so the assertion is about the guard, not work.)"""
        monkeypatch.setattr(main, "_db_cloning", True)
        monkeypatch.setattr(sim, "start_run_cycle_background", lambda: {"started": True})

        async def fake_rebalance(force=False):
            return {"rebalanced": True}

        async def fake_daily_cycle(wait=True):
            return {"skipped": True, "reason": "stubbed"}

        monkeypatch.setattr(monthly, "run_rebalance", fake_rebalance)
        monkeypatch.setattr(daily_core, "run_daily_cycle", fake_daily_cycle)
        async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=main.app),
                base_url="http://test") as client:
            for path in self.DEV_BACKFILLS:
                r = await client.post(path)
                assert r.status_code == 403, f"{path}: {r.text}"  # DEV-only, not clone-409
            for path in self.DEV_RUNS:
                r = await client.post(path)
                assert r.status_code == 200, f"{path}: {r.text}"

    async def test_clone_409_when_a_portfolio_lock_is_held(self, dev_cookie_app):
        async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=main.app),
                base_url="http://test") as client:
            for lock in (sim._run_cycle_lock, monthly._rebalance_lock,
                         daily_core._cycle_lock):
                await lock.acquire()
                try:
                    r = await client.post("/api/db/clone")
                    assert r.status_code == 409
                    assert "backfill or run is already running" in r.json()["detail"]
                finally:
                    lock.release()

    async def test_clone_409_when_dev_is_disabled(self, monkeypatch):
        monkeypatch.setattr(db_mod, "engine_dev", None)
        monkeypatch.setattr(main, "engine_dev", None)
        async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=main.app),
                base_url="http://test") as client:
            r = await client.post("/api/db/clone")
            assert r.status_code == 409
            assert "DEV database is disabled" in r.json()["detail"]


class TestTickLockWait:
    """M2: a busy portfolio lock delays the scheduled cycle (bounded) instead
    of skipping the tick; only a holder wedged past the bound is skipped."""

    @pytest.fixture(autouse=True)
    def _fresh_locks(self, monkeypatch):
        """Fresh locks per test: an asyncio.Lock binds to the event loop of its
        first contended acquire, and pytest-asyncio gives every test a new
        loop — a later test waiting on a stale cross-loop lock raises."""
        monkeypatch.setattr(sim, "_run_cycle_lock", asyncio.Lock())
        monkeypatch.setattr(monthly, "_rebalance_lock", asyncio.Lock())
        monkeypatch.setattr(daily_core, "_cycle_lock", asyncio.Lock())

    async def test_run_cycle_waits_then_proceeds(self, monkeypatch):
        monkeypatch.setattr(sim, "_TICK_LOCK_WAIT_SECONDS", 10)
        ran: list[int] = []

        async def fake_locked_body():
            ran.append(1)
            return {"trades": []}

        monkeypatch.setattr(sim, "_run_cycle_locked", fake_locked_body)
        await sim._run_cycle_lock.acquire()
        try:
            task = asyncio.create_task(sim.run_cycle())
            await asyncio.sleep(0)  # let the task reach the held lock
            assert ran == []
            assert not task.done()
        finally:
            sim._run_cycle_lock.release()
        assert await asyncio.wait_for(task, 5) == {"trades": []}
        assert ran == [1]

    async def test_run_cycle_times_out_and_skips(self, monkeypatch, caplog):
        monkeypatch.setattr(sim, "_TICK_LOCK_WAIT_SECONDS", 0.05)
        ran: list[int] = []

        async def fake_locked_body():  # pragma: no cover — must not run
            ran.append(1)
            return {"trades": []}

        monkeypatch.setattr(sim, "_run_cycle_locked", fake_locked_body)
        await sim._run_cycle_lock.acquire()
        try:
            with caplog.at_level(logging.ERROR):
                result = await sim.run_cycle()
        finally:
            sim._run_cycle_lock.release()
        assert result == {"skipped": True, "reason": "already running"}
        assert ran == []
        assert "still held" in caplog.text

    async def test_daily_cycle_waits_then_proceeds(self, monkeypatch):
        monkeypatch.setattr(daily_core, "_TICK_LOCK_WAIT_SECONDS", 10)
        ran: list[int] = []

        async def fake_locked_body():
            ran.append(1)
            return {"snapshot": {}}

        monkeypatch.setattr(daily_core, "_run_daily_cycle_locked", fake_locked_body)
        await daily_core._cycle_lock.acquire()
        try:
            task = asyncio.create_task(daily_core.run_daily_cycle())
            await asyncio.sleep(0)
            assert ran == []
        finally:
            daily_core._cycle_lock.release()
        assert await asyncio.wait_for(task, 5) == {"snapshot": {}}
        assert ran == [1]

    async def test_daily_cycle_times_out_and_skips(self, monkeypatch, caplog):
        monkeypatch.setattr(daily_core, "_TICK_LOCK_WAIT_SECONDS", 0.05)
        ran: list[int] = []

        async def fake_locked_body():  # pragma: no cover — must not run
            ran.append(1)
            return {"snapshot": {}}

        monkeypatch.setattr(daily_core, "_run_daily_cycle_locked", fake_locked_body)
        await daily_core._cycle_lock.acquire()
        try:
            with caplog.at_level(logging.ERROR):
                result = await daily_core.run_daily_cycle()
        finally:
            daily_core._cycle_lock.release()
        assert result == {"skipped": True, "reason": "already running"}
        assert ran == []
        assert "still held" in caplog.text

    async def test_daily_cycle_manual_wait_false_skips_immediately(self):
        await daily_core._cycle_lock.acquire()
        try:
            r = await daily_core.run_daily_cycle(wait=False)
        finally:
            daily_core._cycle_lock.release()
        assert r == {"skipped": True, "reason": "already running"}

    async def test_monthly_cycle_waits_then_proceeds(self, monkeypatch):
        monkeypatch.setattr(settings, "sim_monthly_enabled", True)
        monkeypatch.setattr(monthly, "is_rebalance_day", lambda today=None: True)
        monkeypatch.setattr(monthly, "_TICK_LOCK_WAIT_SECONDS", 10)
        ran: list[int] = []

        async def fake_locked_body(force=False):
            ran.append(1)
            return {"rebalanced": True}

        monkeypatch.setattr(monthly, "_run_rebalance_locked", fake_locked_body)
        await monthly._rebalance_lock.acquire()
        try:
            task = asyncio.create_task(monthly.run_monthly_cycle())
            await asyncio.sleep(0)
            assert ran == []
        finally:
            monthly._rebalance_lock.release()
        assert await asyncio.wait_for(task, 5) == {"rebalanced": True}
        assert ran == [1]

    async def test_monthly_cycle_times_out_and_skips(self, monkeypatch, caplog):
        monkeypatch.setattr(settings, "sim_monthly_enabled", True)
        monkeypatch.setattr(monthly, "is_rebalance_day", lambda today=None: True)
        monkeypatch.setattr(monthly, "_TICK_LOCK_WAIT_SECONDS", 0.05)

        async def fail_body(force=False):  # pragma: no cover — must not run
            raise AssertionError("must not rebalance while the lock is held")

        monkeypatch.setattr(monthly, "_run_rebalance_locked", fail_body)
        await monthly._rebalance_lock.acquire()
        try:
            with caplog.at_level(logging.ERROR):
                result = await monthly.run_monthly_cycle()
        finally:
            monthly._rebalance_lock.release()
        assert result == {"skipped": True, "reason": "already running"}
        assert "still held" in caplog.text

    async def test_scheduler_tick_delays_the_sim_stage_then_continues(self, monkeypatch):
        """The tick WAITS for the sim lock (instead of skipping) and the
        daily-core stage still runs right after it."""
        monkeypatch.setattr(sim, "_TICK_LOCK_WAIT_SECONDS", 10)
        monkeypatch.setattr(sim, "_utcnow", lambda: datetime(2026, 9, 21, 22, 30))
        order: list[str] = []

        async def fake_locked_body():
            order.append("sim")
            return {"trades": []}

        async def fake_monthly():
            return {"skipped": True, "reason": "not month end"}

        async def fake_daily_core():
            order.append("daily_core")
            return {"deployment": {"trades": []}}

        monkeypatch.setattr(sim, "_run_cycle_locked", fake_locked_body)
        monkeypatch.setattr(monthly, "run_monthly_cycle", fake_monthly)
        monkeypatch.setattr(daily_core, "run_daily_cycle", fake_daily_core)
        await sim._run_cycle_lock.acquire()
        try:
            task = asyncio.create_task(sim._scheduler_tick())
            await asyncio.sleep(0)
            assert order == []  # waiting — neither skipped nor run yet
        finally:
            sim._run_cycle_lock.release()
        await asyncio.wait_for(task, 5)
        assert order == ["sim", "daily_core"]


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

    async def test_stale_tmp_from_a_crashed_clone_is_discarded(self, monkeypatch, tmp_path):
        """Low 4: a leftover `<dev>.tmp` must never poison the next backup."""
        prod_path, dev_path = tmp_path / "prod.sqlite", tmp_path / "dev.sqlite"
        eng_prod, eng_dev = _engine_for(prod_path), _engine_for(dev_path)
        await _create_schema(eng_prod)
        monkeypatch.setattr(db_mod, "engine", eng_prod)
        monkeypatch.setattr(db_mod, "engine_dev", eng_dev)
        async with async_sessionmaker(eng_prod, expire_on_commit=False)() as s:
            s.add(Watchlist(ticker="PROD1"))
            await s.commit()

        stale_tmp = Path(f"{dev_path}.tmp")
        stale_tmp.write_bytes(b"garbage from a crashed clone")

        r = await db_mod.clone_prod_to_dev()
        assert r["ok"] is True
        assert _tickers(dev_path) == {"PROD1"}
        assert not stale_tmp.exists()

        await eng_prod.dispose()
        await eng_dev.dispose()

    async def test_refuses_when_dev_path_resolves_to_prod(self, monkeypatch, tmp_path):
        """Defense in depth: even a programmatic engine mixup must not let the
        clone's os.replace land on PROD."""
        path = tmp_path / "prod.sqlite"
        eng = _engine_for(path)
        await _create_schema(eng)
        monkeypatch.setattr(db_mod, "engine", eng)
        monkeypatch.setattr(db_mod, "engine_dev", eng)
        with pytest.raises(RuntimeError, match="refusing to clone over PROD"):
            await db_mod.clone_prod_to_dev()
        await eng.dispose()


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
