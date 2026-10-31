"""Tests for the PROD/DEV database switch (app.db + app.main routing).

Covered here:

- the ``_active_db`` ContextVar (:func:`use_db`, :func:`current_db`) and the
  dispatching :func:`Session` factory (engine isolation, engine-swap rebuild)
- the FastAPI dependency guards: the 7 history-rewriting endpoints 403 on
  PROD (including with the whole feature disabled), the 20 savings write
  endpoints 403 on DEV; direct Python calls bypass the guards
- the ``ts_db`` cookie middleware routing real ASGI requests end-to-end
- the misconfigured-DEV refusals (``:memory:``, non-SQLite, dev == prod path)
- :func:`clone_prod_to_dev` (backup, atomic swap, stale sidecars, mutual
  exclusion both directions) and ``/api/db/clone`` / ``/api/db/status``
- the lifespan auto-seed signal (:func:`_dev_is_empty`) and dual-engine
  :func:`init_db`
- per-DB sim and daily-core state files (incl. monkeypatch patchability)
- the scheduler deferred-retry protection (a busy lock delays the scheduled
  cycle, never drops it; the deferred task never wakes inside a test)
- Phase 2 invariant: a dev-routed request's run-config/epoch reads hit DEV

All databases are file-backed SQLite files under ``tmp_path``; ``app.db``
module attributes are monkeypatched, so nothing touches /data or the network.
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

from app import daily_core, main, monthly, run_config, sim
from app import db as db_mod
from app.config import settings
from app.db import Base, DailyCoreRanking, Watchlist, current_db, use_db

# A pinned trading day (Monday) for the scheduler-tick tests: the tick returns
# early on weekends/NYSE holidays.
_TRADING_NOW = datetime(2026, 9, 21, 22, 30)

# Captured before any test can monkeypatch sqlite3.connect (clone-failure
# simulations patch it globally).
REAL_SQLITE_CONNECT = sqlite3.connect


def _engine_for(path: Path):
    """File-backed async engine built through the app's own factory."""
    return db_mod.create_db_engine(f"sqlite+aiosqlite:///{path}")


async def _create_schema(engine) -> None:
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


def _tickers(path: Path) -> set[str]:
    """All watchlist tickers in a SQLite file, via a raw fresh connection.

    Uses the connect captured at import time: some tests monkeypatch
    sqlite3.connect globally to simulate clone failures."""
    conn = REAL_SQLITE_CONNECT(path)
    try:
        return {row[0] for row in conn.execute("SELECT ticker FROM watchlist")}
    finally:
        conn.close()  # `with sqlite3.connect` commits but does not close


def _count_rows(path: Path, table: str) -> int:
    conn = REAL_SQLITE_CONNECT(path)
    try:
        return conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    finally:
        conn.close()


def _has_table(path: Path, name: str) -> bool:
    conn = REAL_SQLITE_CONNECT(path)
    try:
        row = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)
        ).fetchone()
    finally:
        conn.close()
    return row is not None


@pytest.fixture(autouse=True)
def _cleanup_deferred_retries():
    """Never leave a pending deferred-retry task behind: a sleeping task would
    only ever fire long after its test's loop closed."""
    yield
    for task in list(sim._deferred_retry_tasks):
        task.cancel()
    sim._deferred_retry_tasks.clear()


@pytest.fixture
async def two_file_dbs(monkeypatch, tmp_path):
    """Two file-backed DBs wired into app.db's engine globals.

    Only the ENGINE globals are patched: the dispatching Session() resolves
    the active engine at call time (the sessionmaker cache is engine-identity
    aware), so no sessionmaker patching is needed.
    """
    prod_path = tmp_path / "prod.sqlite"
    dev_path = tmp_path / "dev.sqlite"
    eng_prod, eng_dev = _engine_for(prod_path), _engine_for(dev_path)
    await _create_schema(eng_prod)
    await _create_schema(eng_dev)
    monkeypatch.setattr(db_mod, "engine", eng_prod)
    monkeypatch.setattr(db_mod, "engine_dev", eng_dev)
    # Keep the JSON state files off /data (the host has no /data).
    monkeypatch.setattr(sim, "_STATE_FILE", tmp_path / "sim_state.json")
    monkeypatch.setattr(sim, "_STATE_FILE_DEV", tmp_path / "sim_state_dev.json")
    monkeypatch.setattr(daily_core, "_STATE_FILE", tmp_path / "dc_state.json")
    monkeypatch.setattr(daily_core, "_STATE_FILE_DEV", tmp_path / "dc_state_dev.json")
    yield SimpleNamespace(prod_path=prod_path, dev_path=dev_path,
                         eng_prod=eng_prod, eng_dev=eng_dev)
    db_mod._sessionmakers.clear()
    await eng_prod.dispose()
    await eng_dev.dispose()


@pytest.fixture
async def client():
    async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=main.app),
            base_url="http://test") as c:
        yield c


def _stub_backfills(monkeypatch, calls: dict) -> dict:
    """Replace every engine-level backfill/reset with a recorder so the guard
    tests exercise the ROUTING, not the multi-minute replays."""
    def rec(name, result):
        async def _fn(*args, **kwargs):
            calls[name] = (args, kwargs)
            return result
        return _fn

    monkeypatch.setattr(sim, "backfill", rec("sim_backfill", {"ok": True}))
    monkeypatch.setattr(sim, "backfill_benchmark", rec("benchmark_backfill", {"ok": True}))
    monkeypatch.setattr(monthly, "backfill", rec("monthly_backfill", {"ok": True}))
    monkeypatch.setattr(monthly, "preload_backfill", rec("monthly_preload", object()))
    monkeypatch.setattr(daily_core, "backfill", rec("daily_core_backfill", {"ok": True}))
    monkeypatch.setattr(sim, "reset_sim", rec("reset_sim", {"ok": True}))
    monkeypatch.setattr(monthly, "reset_monthly", rec("reset_monthly", {"ok": True}))
    monkeypatch.setattr(daily_core, "reset_daily_core", rec("reset_daily_core", {"ok": True}))
    return calls


def _stub_runs(monkeypatch) -> None:
    def fake_start() -> dict:
        return {"started": True}

    async def fake_rebalance(force=False, now=None):
        return {"rebalanced": True}

    async def fake_dc(*args, **kwargs):
        return {"skipped": True, "reason": "stubbed"}

    monkeypatch.setattr(sim, "start_run_cycle_background", fake_start)
    monkeypatch.setattr(monthly, "run_rebalance", fake_rebalance)
    monkeypatch.setattr(daily_core, "run_daily_cycle", fake_dc)


# The 7 history-rewriting endpoints (403 on PROD, DEV-only).
DESTRUCTIVE = (
    ("/api/sim/backfill", {"params": {"start": "all"}}),
    ("/api/sim/backfill-benchmark", {"params": {"start": "all"}}),
    ("/api/backfill-all", {"params": {"start": "all"}}),
    ("/api/monthly/backfill", {"params": {"start": "all"}}),
    ("/api/dailycore/backfill", {"params": {"start": "all"}}),
    ("/api/sim/reset", {}),
    ("/api/sim/reset-all", {}),
)

# The 20 real-money (savings) write endpoints (PROD-only, 403 on DEV).
SAVINGS_WRITES = (
    ("POST", "/api/savings/initialize", {"json": {"cash": 100, "positions": []}}),
    ("POST", "/api/savings/reset", {}),
    ("POST", "/api/savings/trueup", {"json": {"actual_cash": 1}}),
    ("POST", "/api/savings/config", {"json": {"monthly_transfer": 1}}),
    ("POST", "/api/savings/deposit", {"json": {"amount": 1}}),
    ("POST", "/api/savings/withdraw", {"json": {"amount": 1}}),
    ("POST", "/api/savings/dividend", {"json": {"amount": 1, "ticker": "AAA"}}),
    ("POST", "/api/savings/fee", {"json": {"amount": 1}}),
    ("POST", "/api/savings/tax", {"json": {"amount": 1}}),
    ("POST", "/api/savings/settle", {"json": {"ticker": "AAA", "proceeds": 1}}),
    ("POST", "/api/savings/split", {"json": {"ticker": "AAA", "ratio": 2.0}}),
    ("POST", "/api/savings/correct", {"json": {"ticker": "AAA", "shares": 1}}),
    ("POST", "/api/savings/saveback", {"json": {"amount": 1, "ticker": "AAA"}}),
    ("POST", "/api/savings/buy", {"json": {"amount": 1, "ticker": "AAA"}}),
    ("POST", "/api/savings/plans", {"json": {"ticker": "AAA", "amount": 1,
                                            "day_of_month": 1}}),
    ("POST", "/api/savings/sell", {"json": {"ticker": "AAA", "shares": 1}}),
    ("POST", "/api/savings/refresh", {}),
    ("POST", "/api/savings/run", {}),
    ("DELETE", "/api/savings/plans/1", {}),
    ("DELETE", "/api/savings/positions/AAA", {}),
)


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
    async def test_sessions_land_in_the_active_db_file(self, two_file_dbs):
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

    async def test_dev_without_an_engine_raises(self, two_file_dbs, monkeypatch):
        """A dev selection with no dev engine (feature disabled mid-run) must
        fail loudly, never silently fall back to the live book."""
        monkeypatch.setattr(db_mod, "engine_dev", None)
        with use_db("dev"), pytest.raises(RuntimeError, match="not configured"):
            db_mod.Session()

    async def test_engine_swap_gets_a_fresh_sessionmaker(self, two_file_dbs,
                                                         monkeypatch, tmp_path):
        """The sessionmaker cache is engine-identity aware: patching db.engine
        (tests, maintenance) transparently rebuilds the factory instead of
        serving sessions on the old engine."""
        third_path = tmp_path / "third.sqlite"
        eng_third = _engine_for(third_path)
        await _create_schema(eng_third)
        try:
            with use_db("prod"):
                async with db_mod.Session() as s:
                    s.add(Watchlist(ticker="ORIGINAL"))
                    await s.commit()
            assert _tickers(two_file_dbs.prod_path) == {"ORIGINAL"}
            monkeypatch.setattr(db_mod, "engine", eng_third)
            # The very same use_db("prod") block now hits the new file.
            with use_db("prod"):
                async with db_mod.Session() as s:
                    s.add(Watchlist(ticker="SWAPPED"))
                    await s.commit()
            assert _tickers(third_path) == {"SWAPPED"}
            assert _tickers(two_file_dbs.prod_path) == {"ORIGINAL"}
        finally:
            await eng_third.dispose()


class TestDevEngineFactory:
    """_create_dev_engine refuses unsafe DEV configs instead of building an
    engine a clone could use to overwrite PROD."""

    def test_empty_url_disables_dev_silently(self):
        assert db_mod._create_dev_engine(db_mod.engine, "") is None

    async def test_non_file_url_disables_dev(self, tmp_path, caplog):
        prod = _engine_for(tmp_path / "prod.sqlite")
        try:
            with caplog.at_level(logging.ERROR):
                assert db_mod._create_dev_engine(prod, "sqlite+aiosqlite:///:memory:") is None
            assert "not usable" in caplog.text
        finally:
            await prod.dispose()

    async def test_dev_path_equal_to_prod_disables_dev(self, tmp_path, caplog):
        prod_path = tmp_path / "prod.sqlite"
        prod = _engine_for(prod_path)
        try:
            # Same file through a non-normalized path: resolve() must catch it.
            same = tmp_path / "sub" / ".." / "prod.sqlite"
            with caplog.at_level(logging.ERROR):
                assert db_mod._create_dev_engine(prod, f"sqlite+aiosqlite:///{same}") is None
            assert "PROD" in caplog.text
        finally:
            await prod.dispose()

    async def test_non_sqlite_url_disables_dev(self, tmp_path):
        prod = _engine_for(tmp_path / "prod.sqlite")
        try:
            assert db_mod._create_dev_engine(
                prod, "postgresql+asyncpg://user@host/db") is None
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


class TestGuardFunctions:
    """The dependency factories themselves (HTTP behaviour is covered below
    with real requests)."""

    async def test_require_dev_db_403s_on_prod(self):
        guard = main._require_dev_db("X")
        with pytest.raises(HTTPException) as ei:
            await guard()
        assert ei.value.status_code == 403
        with use_db("dev"):
            await guard()  # no raise on dev

    async def test_require_prod_db_403s_on_dev(self):
        guard = main._require_prod_db("X")
        await guard()  # no raise on prod (default)
        with use_db("dev"):
            with pytest.raises(HTTPException) as ei:
                await guard()
            assert ei.value.status_code == 403

    async def test_refuse_during_clone_fires_only_dev_routed(self, monkeypatch):
        monkeypatch.setattr(main, "_db_cloning", True)
        await main._refuse_during_clone()  # prod-routed: a clone swaps DEV only
        with use_db("dev"):
            with pytest.raises(HTTPException) as ei:
                await main._refuse_during_clone()
            assert ei.value.status_code == 409
        monkeypatch.setattr(main, "_db_cloning", False)
        with use_db("dev"):
            await main._refuse_during_clone()  # no clone running: no raise


class TestCookieRouting:
    async def test_status_reports_the_cookie_selected_db(self, two_file_dbs, client):
        r = await client.get("/api/db/status")
        assert r.status_code == 200
        body = r.json()
        assert body["active"] == "prod"
        assert body["dev_enabled"] is True
        assert body["dev_seeded"] is False  # schema-only DEV: no data yet
        assert body["cloning"] is False

        client.cookies.set("ts_db", "dev")
        r = await client.get("/api/db/status")
        body = r.json()
        assert body["active"] == "dev"
        assert body["dev_enabled"] is True

    async def test_unknown_cookie_falls_through_to_prod(self, two_file_dbs, client):
        client.cookies.set("ts_db", "bogus")
        r = await client.get("/api/db/status")
        assert r.json()["active"] == "prod"

    async def test_dev_cookie_falls_through_when_dev_is_disabled(
            self, two_file_dbs, monkeypatch, client):
        monkeypatch.setattr(db_mod, "engine_dev", None)
        client.cookies.set("ts_db", "dev")
        r = await client.get("/api/db/status")
        body = r.json()
        assert body["active"] == "prod"
        assert body["dev_enabled"] is False

    async def test_http_writes_follow_the_cookie(self, two_file_dbs, client, monkeypatch):
        """An unguarded write endpoint proves end-to-end engine isolation: the
        row lands in exactly the file the cookie selected."""
        async def fake_fetch(ticker, period):
            return []

        monkeypatch.setattr(main, "refresh_yfinance", fake_fetch)

        r = await client.post("/api/watchlist/AAA")
        assert r.status_code == 200
        assert _tickers(two_file_dbs.prod_path) == {"AAA"}
        assert _tickers(two_file_dbs.dev_path) == set()

        client.cookies.set("ts_db", "dev")
        r = await client.post("/api/watchlist/BBB")
        assert r.status_code == 200
        assert _tickers(two_file_dbs.dev_path) == {"BBB"}
        assert _tickers(two_file_dbs.prod_path) == {"AAA"}  # PROD untouched

        # Reads follow the cookie too.
        assert (await client.get("/api/watchlist")).json() == ["BBB"]
        client.cookies.delete("ts_db")
        assert (await client.get("/api/watchlist")).json() == ["AAA"]

    async def test_savings_reads_stay_unguarded_on_dev(self, two_file_dbs, client,
                                                       monkeypatch):
        from app import savings

        async def fake_plans():
            return ["P1"]

        monkeypatch.setattr(savings, "get_plans", fake_plans)
        client.cookies.set("ts_db", "dev")
        r = await client.get("/api/savings/plans")
        assert r.status_code == 200
        assert r.json() == ["P1"]


class TestDestructiveGuards:
    """Invariant 1: the 7 history-rewriting endpoints 403 unless the request
    is dev-routed — UNCONDITIONALLY, i.e. also with the feature disabled."""

    async def test_prod_requests_get_403_and_never_call_engines(self, two_file_dbs,
                                                                client):
        calls: dict = {}
        mp = pytest.MonkeyPatch()
        try:
            _stub_backfills(mp, calls)
            for path, kwargs in DESTRUCTIVE:
                r = await client.post(path, **kwargs)
                assert r.status_code == 403, f"{path}: {r.text}"
                assert "PROD" in r.json()["detail"]
            assert calls == {}  # the guard fired before any engine work ran
        finally:
            mp.undo()

    async def test_dev_cookie_acks_all_seven(self, two_file_dbs, client, monkeypatch):
        calls = _stub_backfills(monkeypatch, {})
        client.cookies.set("ts_db", "dev")
        for path, kwargs in DESTRUCTIVE:
            r = await client.post(path, **kwargs)
            assert r.status_code == 200, f"{path}: {r.text}"
        # Every stubbed engine ran exactly once, dev-routed.
        assert set(calls) == {"sim_backfill", "benchmark_backfill", "monthly_backfill",
                              "monthly_preload", "daily_core_backfill",
                              "reset_sim", "reset_monthly", "reset_daily_core"}

    async def test_feature_disabled_refuses_everything(
            self, two_file_dbs, monkeypatch, client):
        """Invariant 1's unconditional clause: with no DEV database at all,
        no database accepts the destructive endpoints — not even with the
        cookie set (the cookie falls through to PROD, and PROD refuses)."""
        monkeypatch.setattr(db_mod, "engine_dev", None)
        calls = _stub_backfills(monkeypatch, {})
        for cookie in (None, "prod", "dev"):
            if cookie is None:
                client.cookies.delete("ts_db")
            else:
                client.cookies.set("ts_db", cookie)
            for path, kwargs in DESTRUCTIVE:
                r = await client.post(path, **kwargs)
                assert r.status_code == 403, f"{path} (ts_db={cookie}): {r.text}"
        assert calls == {}

    async def test_reset_end_to_end_on_dev_file(self, two_file_dbs, client):
        """The REAL reset (no stub) with the dev cookie: the new sim account
        row must appear in the DEV file only — PROD stays untouched."""
        client.cookies.set("ts_db", "dev")
        r = await client.post("/api/sim/reset")
        assert r.status_code == 200
        assert _count_rows(two_file_dbs.dev_path, "sim_account") == 1
        assert _count_rows(two_file_dbs.prod_path, "sim_account") == 0

    async def test_direct_python_calls_bypass_the_guards(self, two_file_dbs, monkeypatch):
        """The guards are HTTP dependencies: maintenance/test callers of the
        endpoint functions keep the plain functions usable (tests/test_main.py
        depends on this)."""
        calls = _stub_backfills(monkeypatch, {})
        out = await main.sim_reset()
        assert out == {"ok": True}
        assert "reset_sim" in calls
        out = await main.backfill_all(start="all")
        assert out["ok"] is True


class TestSavingsGuards:
    """Invariant 2: real-money writes are PROD-only (403 when dev-routed);
    savings GETs stay unguarded."""

    @pytest.fixture(autouse=True)
    def savings_calls(self, monkeypatch):
        from app import savings
        calls: list[str] = []

        def recorder(name, result=None):
            async def _fn(*args, **kwargs):
                calls.append(name)
                return result if result is not None else {"ok": True}
            return _fn

        for fn in ("initialize", "reset", "trueup", "set_config", "add_deposit",
                   "withdraw", "add_dividend", "add_fee", "add_tax", "settle_sale",
                   "record_split", "correct_position", "add_saveback", "manual_buy",
                   "add_plan", "remove_plan", "remove_position"):
            monkeypatch.setattr(savings, fn, recorder(fn))
        monkeypatch.setattr(savings, "refresh_data",
                            recorder("refresh_data", {"refreshed": [], "errors": []}))
        monkeypatch.setattr(savings, "run_savings_cycle",
                            recorder("run_savings_cycle", {"ok": True}))
        return calls

    async def test_savings_writes_403_on_dev(self, two_file_dbs, client, savings_calls):
        client.cookies.set("ts_db", "dev")
        for method, path, kwargs in SAVINGS_WRITES:
            r = await client.request(method, path, **kwargs)
            assert r.status_code == 403, f"{path}: {r.text}"
            assert "PROD" in r.json()["detail"]
        assert savings_calls == []  # the guard fired before any savings code

    async def test_savings_writes_ack_on_prod(self, two_file_dbs, client, savings_calls):
        for method, path, kwargs in SAVINGS_WRITES:
            r = await client.request(method, path, **kwargs)
            assert r.status_code == 200, f"{path}: {r.text}"
        # every handler actually ran on PROD (the stubs were called)
        assert len(savings_calls) == len(SAVINGS_WRITES)


class TestUnguardedSurface:
    """The verified unguarded inventory: the manual runs follow the selected
    DB but are never refused by a PROD/DEV guard."""

    async def test_runs_work_on_both_dbs(self, two_file_dbs, client, monkeypatch):
        _stub_runs(monkeypatch)
        for cookie in (None, "dev"):
            if cookie is None:
                client.cookies.delete("ts_db")
            else:
                client.cookies.set("ts_db", cookie)
            for path in ("/api/sim/run", "/api/monthly/run", "/api/dailycore/run"):
                r = await client.post(path)
                assert r.status_code == 200, f"{path} (ts_db={cookie}): {r.text}"


class TestCloneMutualExclusion:
    DEV_BACKFILLS = ("/api/sim/backfill?start=all", "/api/sim/backfill-benchmark?start=all",
                     "/api/monthly/backfill?start=all", "/api/dailycore/backfill?start=all")
    DEV_RUNS = ("/api/sim/run", "/api/monthly/run", "/api/dailycore/run")

    async def test_dev_ops_409_while_a_clone_runs(self, two_file_dbs, client, monkeypatch):
        _stub_backfills(monkeypatch, {})
        _stub_runs(monkeypatch)
        monkeypatch.setattr(main, "_db_cloning", True)
        client.cookies.set("ts_db", "dev")
        for path in self.DEV_BACKFILLS + self.DEV_RUNS:
            r = await client.post(path)
            assert r.status_code == 409, f"{path}: {r.text}"
            assert "clone is running" in r.json()["detail"]

    async def test_prod_requests_ignore_the_clone_flag(self, two_file_dbs, client,
                                                       monkeypatch):
        """A clone swaps only the DEV file: PROD requests never hit the clone
        guard (the PROD-only backfill guard is what answers them)."""
        _stub_backfills(monkeypatch, {})
        _stub_runs(monkeypatch)
        monkeypatch.setattr(main, "_db_cloning", True)
        for path in self.DEV_BACKFILLS:
            r = await client.post(path)
            assert r.status_code == 403, f"{path}: {r.text}"  # DEV-only, not clone-409
        for path in self.DEV_RUNS:
            r = await client.post(path)
            assert r.status_code == 200, f"{path}: {r.text}"

    async def test_clone_409_while_backfill_all_lock_is_held(self, two_file_dbs, client):
        await main._backfill_all_lock.acquire()
        try:
            r = await client.post("/api/db/clone")
            assert r.status_code == 409
            assert "already running" in r.json()["detail"]
        finally:
            main._backfill_all_lock.release()

    async def test_clone_409_while_a_portfolio_lock_is_held(self, two_file_dbs, client):
        for lock in (sim._run_cycle_lock, monthly._rebalance_lock, daily_core._cycle_lock):
            await lock.acquire()
            try:
                r = await client.post("/api/db/clone")
                assert r.status_code == 409
                assert "backfill or run is already running" in r.json()["detail"]
            finally:
                lock.release()

    async def test_clone_409_when_dev_is_disabled(self, two_file_dbs, monkeypatch, client):
        monkeypatch.setattr(db_mod, "engine_dev", None)
        r = await client.post("/api/db/clone")
        assert r.status_code == 409
        assert "disabled" in r.json()["detail"]

    async def test_clone_flag_observable_in_status(self, two_file_dbs, client, monkeypatch):
        release = asyncio.Event()

        async def slow_clone():
            await release.wait()
            return {"ok": True, "path": "stub", "size_bytes": 1,
                    "cloned_at": "2026-10-30T00:00:00+00:00"}

        monkeypatch.setattr(main, "clone_prod_to_dev", slow_clone)
        task = asyncio.create_task(client.post("/api/db/clone"))
        for _ in range(500):  # wait for the flag to flip
            if main._db_cloning:
                break
            await asyncio.sleep(0.01)
        assert main._db_cloning is True
        r = await client.get("/api/db/status")
        assert r.json()["cloning"] is True
        release.set()
        r = await asyncio.wait_for(task, 5)
        assert r.status_code == 200
        assert r.json()["size_bytes"] == 1
        r = await client.get("/api/db/status")
        assert r.json()["cloning"] is False

    async def test_clone_endpoint_clones_for_real(self, two_file_dbs, client):
        with use_db("prod"):
            async with db_mod.Session() as s:
                s.add(Watchlist(ticker="PROD1"))
                await s.commit()
        r = await client.post("/api/db/clone")
        assert r.status_code == 200
        body = r.json()
        assert body["ok"] is True
        assert body["size_bytes"] > 0
        assert body["path"] == str(two_file_dbs.dev_path)
        assert _tickers(two_file_dbs.dev_path) == {"PROD1"}


class TestCloneProdToDev:
    async def test_clone_swaps_dev_and_prod_stays_independent(self, two_file_dbs,
                                                              monkeypatch):
        with use_db("prod"):
            async with db_mod.Session() as s:
                s.add(Watchlist(ticker="PROD1"))
                await s.commit()
        # Sidecars a crashed process could have left behind.
        Path(f"{two_file_dbs.dev_path}-wal").write_text("stale-wal")
        Path(f"{two_file_dbs.dev_path}-shm").write_text("stale-shm")

        r = await db_mod.clone_prod_to_dev()
        assert r["ok"] is True
        assert r["path"] == str(two_file_dbs.dev_path)
        assert r["size_bytes"] > 0
        datetime.fromisoformat(r["cloned_at"])  # parses as ISO
        assert not Path(f"{two_file_dbs.dev_path}-wal").exists()
        assert not Path(f"{two_file_dbs.dev_path}-shm").exists()
        assert _tickers(two_file_dbs.dev_path) == {"PROD1"}

        # Prod keeps living independently after the clone.
        with use_db("prod"):
            async with db_mod.Session() as s:
                s.add(Watchlist(ticker="PROD2"))
                await s.commit()
        assert _tickers(two_file_dbs.prod_path) == {"PROD1", "PROD2"}
        assert _tickers(two_file_dbs.dev_path) == {"PROD1"}

        # A dev-only row is wiped by the re-clone (dev is replaceable).
        with use_db("dev"):
            async with db_mod.Session() as s:
                s.add(Watchlist(ticker="DEVONLY"))
                await s.commit()
        r2 = await db_mod.clone_prod_to_dev()
        assert r2["ok"] is True
        assert _tickers(two_file_dbs.dev_path) == {"PROD1", "PROD2"}

        # No dev engine configured -> refuse, nothing to clone into.
        monkeypatch.setattr(db_mod, "engine_dev", None)
        with pytest.raises(RuntimeError, match="DATABASE_URL_DEV"):
            await db_mod.clone_prod_to_dev()

    async def test_stale_tmp_from_a_crashed_clone_is_discarded(self, two_file_dbs):
        with use_db("prod"):
            async with db_mod.Session() as s:
                s.add(Watchlist(ticker="PROD1"))
                await s.commit()
        stale_tmp = Path(f"{two_file_dbs.dev_path}.tmp")
        stale_tmp.write_bytes(b"garbage from a crashed clone")

        r = await db_mod.clone_prod_to_dev()
        assert r["ok"] is True
        assert _tickers(two_file_dbs.dev_path) == {"PROD1"}
        assert not stale_tmp.exists()

    async def test_refuses_when_dev_path_resolves_to_prod(self, two_file_dbs):
        """Defense in depth: even a programmatic engine mixup must not let
        the clone's os.replace land on PROD."""
        dev_backup = db_mod.engine_dev
        db_mod.engine_dev = two_file_dbs.eng_prod
        try:
            with pytest.raises(RuntimeError, match="refusing to clone over PROD"):
                await db_mod.clone_prod_to_dev()
        finally:
            db_mod.engine_dev = dev_backup

    async def test_failed_clone_leaves_the_old_dev_intact(self, two_file_dbs, monkeypatch):
        """Atomicity: a backup that dies mid-flight leaves the OLD dev file
        untouched, no half-written tmp behind."""
        with use_db("prod"):
            async with db_mod.Session() as s:
                s.add(Watchlist(ticker="PROD1"))
                await s.commit()
        with use_db("dev"):
            async with db_mod.Session() as s:
                s.add(Watchlist(ticker="DEVOLD"))
                await s.commit()

        def boom(*args, **kwargs):
            raise RuntimeError("backup exploded")

        monkeypatch.setattr(db_mod.sqlite3, "connect", boom)
        with pytest.raises(RuntimeError, match="backup exploded"):
            await db_mod.clone_prod_to_dev()
        assert _tickers(two_file_dbs.dev_path) == {"DEVOLD"}  # old dev intact
        assert not Path(f"{two_file_dbs.dev_path}.tmp").exists()


class TestDevIsEmpty:
    async def test_true_for_a_fresh_schema(self, two_file_dbs):
        assert await main._dev_is_empty() is True

    async def test_false_once_dev_has_a_watchlist_row(self, two_file_dbs):
        with use_db("dev"):
            async with db_mod.Session() as s:
                s.add(Watchlist(ticker="DEV1"))
                await s.commit()
        assert await main._dev_is_empty() is False

    async def test_status_dev_seeded_tracks_data_not_bare_existence(self, two_file_dbs,
                                                                    client):
        client.cookies.set("ts_db", "dev")
        r = await client.get("/api/db/status")
        assert r.json()["dev_seeded"] is False  # file exists, but empty
        with use_db("dev"):
            async with db_mod.Session() as s:
                s.add(Watchlist(ticker="DEV1"))
                await s.commit()
        r = await client.get("/api/db/status")
        assert r.json()["dev_seeded"] is True


class TestLifespanAutoSeed:
    async def _seed_and_maybe_clone(self):
        """The lifespan's seed block (init_db -> watchlist seed -> auto-seed
        DEV when missing/empty and PROD has data)."""
        await db_mod.init_db()
        async with db_mod.Session() as s:
            for t in ("AAA", "BBB"):
                if not await s.get(Watchlist, t):
                    s.add(Watchlist(ticker=t))
            await s.commit()
        if (db_mod.dev_enabled() and db_mod.prod_db_exists()
                and (not db_mod.dev_db_exists() or await main._dev_is_empty())):
            await db_mod.clone_prod_to_dev()

    async def test_lifespan_seeds_an_empty_dev_from_prod(self, two_file_dbs):
        """A greenfield boot: PROD receives the settings watchlist seed, then
        the still-empty DEV is auto-cloned so the sandbox starts with data."""
        await self._seed_and_maybe_clone()
        assert _tickers(two_file_dbs.prod_path) == {"AAA", "BBB"}
        assert _tickers(two_file_dbs.dev_path) == {"AAA", "BBB"}  # auto-seeded

        # A seeded DEV is NOT re-cloned on the next start: the operator's dev
        # experiments survive a restart.
        with use_db("dev"):
            async with db_mod.Session() as s:
                s.add(Watchlist(ticker="DEVEXPERIMENT"))
                await s.commit()
        await self._seed_and_maybe_clone()
        assert _tickers(two_file_dbs.dev_path) == {"AAA", "BBB", "DEVEXPERIMENT"}


class TestInitDb:
    async def test_creates_schema_on_every_configured_engine(self, two_file_dbs):
        await db_mod.init_db()
        assert _has_table(two_file_dbs.prod_path, "watchlist")
        assert _has_table(two_file_dbs.dev_path, "watchlist")
        # Phase 2 tables included on both sides.
        assert _has_table(two_file_dbs.prod_path, "run_configs")
        assert _has_table(two_file_dbs.dev_path, "run_epochs")

    async def test_explicit_engine_migrates_only_that_one(self, tmp_path):
        """``db_engine`` pins the migration to ONE engine: the module's
        configured PROD/DEV databases are not even created."""
        prod_path, dev_path = tmp_path / "prod2.sqlite", tmp_path / "dev2.sqlite"
        other_path = tmp_path / "other.sqlite"
        eng_prod, eng_dev = _engine_for(prod_path), _engine_for(dev_path)
        eng_other = _engine_for(other_path)
        original = db_mod.engine, db_mod.engine_dev
        db_mod.engine, db_mod.engine_dev = eng_prod, eng_dev
        try:
            await db_mod.init_db(db_engine=eng_other)
            assert _has_table(other_path, "watchlist")
            assert not prod_path.exists()  # untouched
            assert not dev_path.exists()  # untouched
        finally:
            db_mod.engine, db_mod.engine_dev = original
            db_mod._sessionmakers.clear()
            await eng_prod.dispose()
            await eng_dev.dispose()
            await eng_other.dispose()

    async def test_dev_failure_disables_dev_and_keeps_prod(self, two_file_dbs, monkeypatch):
        """A DEV database that cannot be migrated must degrade to "feature
        disabled" (error-logged) — never take PROD's startup down."""
        real_migrate = db_mod._migrate_engine
        eng_dev = db_mod.engine_dev

        async def selective_migrate(eng):
            if eng is eng_dev:
                raise RuntimeError("dev disk on fire")
            await real_migrate(eng)

        monkeypatch.setattr(db_mod, "_migrate_engine", selective_migrate)
        await db_mod.init_db()
        assert db_mod.engine_dev is None  # disabled, not crashed
        assert _has_table(two_file_dbs.prod_path, "watchlist")  # PROD migrated


class TestPerDbStateFiles:
    """The per-session fill guard / strategy state files are per-DB, so a DEV
    run can never make the PROD tick skip (and vice versa). The globals stay
    monkeypatchable — existing tests patch _STATE_FILE."""

    def test_sim_state_files_are_per_db(self, two_file_dbs, tmp_path, monkeypatch):
        prod_state = tmp_path / "s_prod.json"
        dev_state = tmp_path / "s_dev.json"
        monkeypatch.setattr(sim, "_STATE_FILE", prod_state)
        monkeypatch.setattr(sim, "_STATE_FILE_DEV", dev_state)

        with use_db("prod"):
            sim._save_state({"last_session": "2026-09-21"})
        assert json.loads(prod_state.read_text()) == {"last_session": "2026-09-21"}
        assert not dev_state.exists()

        with use_db("dev"):
            assert sim._load_state() == {}  # dev file not written yet
            sim._save_state({"last_session": "2026-10-01"})
        assert json.loads(dev_state.read_text()) == {"last_session": "2026-10-01"}
        assert json.loads(prod_state.read_text()) == {"last_session": "2026-09-21"}
        # The prod tick still sees its own (already processed) session...
        with use_db("prod"):
            assert sim._session_guard_passed("2026-09-21") is False
            assert sim._session_guard_passed("2026-09-22") is True
        # ...while DEV's guard is independent.
        with use_db("dev"):
            assert sim._session_guard_passed("2026-10-01") is False

    def test_sim_state_file_patchability(self, tmp_path, monkeypatch):
        """A test repointing sim._STATE_FILE mid-flight is honoured (the
        global is read at CALL time, not captured)."""
        first = tmp_path / "first.json"
        monkeypatch.setattr(sim, "_STATE_FILE", first)
        sim._save_state({"a": 1})
        assert first.exists()
        second = tmp_path / "second.json"
        monkeypatch.setattr(sim, "_STATE_FILE", second)
        sim._save_state({"b": 2})
        assert second.exists()
        assert json.loads(first.read_text()) == {"a": 1}

    def test_daily_core_state_files_are_per_db(self, tmp_path, monkeypatch):
        prod_state = tmp_path / "dc_prod.json"
        dev_state = tmp_path / "dc_dev.json"
        monkeypatch.setattr(daily_core, "_STATE_FILE", prod_state)
        monkeypatch.setattr(daily_core, "_STATE_FILE_DEV", dev_state)

        with use_db("prod"):
            daily_core._save_state({"mom_variant": "raw"})
        assert prod_state.exists()
        assert not dev_state.exists()

        with use_db("dev"):
            assert daily_core._load_state() == {}
            daily_core._save_state({"mom_variant": "residual"})
        assert json.loads(dev_state.read_text()) == {"mom_variant": "residual"}
        assert json.loads(prod_state.read_text()) == {"mom_variant": "raw"}
        with use_db("prod"):
            assert daily_core.current_variant() == "raw"
        with use_db("dev"):
            assert daily_core.current_variant() == "residual"

    async def test_dev_cookie_routes_strategy_state_to_dev(self, two_file_dbs, client):
        """End-to-end: a dev-routed /api/dailycore/strategy invalidates the
        DEV ranking row and writes the DEV state file only. PROD's ranking
        and state stay untouched (what the live book runs never changes)."""
        for db in ("prod", "dev"):
            with use_db(db):
                async with db_mod.Session() as s:
                    s.add(DailyCoreRanking(id=1,
                                           ranking_date=datetime(2026, 9, 1, 22, 0)))
                    await s.commit()
        client.cookies.set("ts_db", "dev")
        r = await client.post("/api/dailycore/strategy", json={"variant": "residual"})
        assert r.status_code == 200
        assert _count_rows(two_file_dbs.dev_path, "daily_core_ranking") == 0
        assert _count_rows(two_file_dbs.prod_path, "daily_core_ranking") == 1
        dev_state = Path(daily_core._STATE_FILE_DEV)
        assert json.loads(dev_state.read_text())["mom_variant"] == "residual"
        assert not Path(daily_core._STATE_FILE).exists()


class TestRunConfigRouting:
    """Invariant 6 (Phase 2): run-config/epoch reads and writes resolve
    db.Session at CALL time, so a dev-routed request's registry hits DEV."""

    async def test_dev_routed_run_config_and_epochs_hit_dev(self, two_file_dbs):
        with use_db("dev"):
            cid = await run_config.get_or_create("sim", "paper", {"a": 1})
            epoch = await run_config.record_boundary("sim", note="dev experiment")
            assert epoch == 2  # first boundary of the dev book
        # The dev file holds the config row AND the boundary; PROD has none.
        assert _count_rows(two_file_dbs.dev_path, "run_configs") == 1
        assert _count_rows(two_file_dbs.dev_path, "run_epochs") == 1
        assert _count_rows(two_file_dbs.prod_path, "run_configs") == 0
        assert _count_rows(two_file_dbs.prod_path, "run_epochs") == 0

        # Epoch reads are per-DB: prod is still in epoch 1.
        with use_db("prod"):
            assert await run_config.current_epoch("sim") == 1
            cid_prod = await run_config.get_or_create("sim", "paper", {"a": 1})
        with use_db("dev"):
            assert await run_config.current_epoch("sim") == 2
            # Content-hash dedupe stays per-DB: the same config resolves to
            # the same dev row (both calls returned the same id), and prod
            # registered its own copy in its own file. (Ids are per-table
            # autoincrement, so both books legitimately mint id 1.)
            assert await run_config.get_or_create("sim", "paper", {"a": 1}) == cid
        assert _count_rows(two_file_dbs.prod_path, "run_configs") == 1
        assert _count_rows(two_file_dbs.dev_path, "run_configs") == 1
        assert cid_prod == cid == 1


class TestDeferredRetry:
    """Invariant 3: a busy portfolio lock may DELAY a scheduled tick but must
    never silently drop it — the tick defers the sim stage to a fire-and-forget
    retry whose first attempt sleeps far longer than any test."""

    @pytest.fixture(autouse=True)
    def _trading_day(self, monkeypatch):
        monkeypatch.setattr(sim, "_utcnow", lambda: _TRADING_NOW)

    @pytest.fixture(autouse=True)
    def _fresh_locks(self, monkeypatch):
        """Fresh locks per test: a lock with waiters binds to its loop, and
        pytest-asyncio gives every test a new loop."""
        monkeypatch.setattr(sim, "_run_cycle_lock", asyncio.Lock())
        monkeypatch.setattr(monthly, "_rebalance_lock", asyncio.Lock())
        monkeypatch.setattr(daily_core, "_cycle_lock", asyncio.Lock())

    def _patch_stages(self, monkeypatch, called):
        async def fake_sim_cycle():
            called["sim"] += 1
            return {"skipped": False, "trades": []}

        async def fake_monthly():
            called["monthly"] += 1
            return {"skipped": True, "reason": "not month end"}

        async def fake_daily_core():
            called["daily_core"] += 1
            return {"deployment": {"trades": []}}

        monkeypatch.setattr(sim, "run_cycle", fake_sim_cycle)
        monkeypatch.setattr(monthly, "run_monthly_cycle", fake_monthly)
        monkeypatch.setattr(daily_core, "run_daily_cycle", fake_daily_core)

    async def test_tick_skips_this_wake_and_defers_never_drops(self, monkeypatch):
        called = {"sim": 0, "monthly": 0, "daily_core": 0}
        self._patch_stages(monkeypatch, called)
        await sim._run_cycle_lock.acquire()
        try:
            await sim._scheduler_tick()
            assert called["sim"] == 0  # skipped THIS wake...
            assert called["daily_core"] == 1  # ...but the later stages still ran
            pending = set(sim._deferred_retry_tasks)
            assert len(pending) == 1, "the scheduled cycle must have been deferred"
            task = pending.pop()
            # The deferred task never wakes inside a test (its initial delay
            # is far longer than any test).
            await asyncio.sleep(0.1)
            assert called["sim"] == 0
            assert not task.done()
        finally:
            sim._run_cycle_lock.release()

    async def test_deferred_retry_runs_once_the_lock_frees(self, monkeypatch):
        called = {"sim": 0, "monthly": 0, "daily_core": 0}
        self._patch_stages(monkeypatch, called)
        monkeypatch.setattr(sim, "_DEFERRED_RETRY_DELAY_SECONDS", 0.01)
        await sim._run_cycle_lock.acquire()
        try:
            await sim._scheduler_tick()
        finally:
            sim._run_cycle_lock.release()
        for _ in range(500):
            if called["sim"] == 1:
                break
            await asyncio.sleep(0.01)
        assert called["sim"] == 1  # delayed, not dropped

    async def test_retry_runs_the_reference_captured_at_schedule_time(self, monkeypatch):
        """The retry must call the run_cycle that was current when the tick
        scheduled it (a test's monkeypatched fake), not a later rebinding."""
        called = {"first": 0, "second": 0}

        async def first_fake():
            called["first"] += 1
            return {"trades": []}

        async def second_fake():
            called["second"] += 1
            return {"trades": []}

        async def noop_monthly():
            return {"skipped": True, "reason": "x"}

        async def noop_dc():
            return {"deployment": {}}

        monkeypatch.setattr(monthly, "run_monthly_cycle", noop_monthly)
        monkeypatch.setattr(daily_core, "run_daily_cycle", noop_dc)
        monkeypatch.setattr(sim, "_DEFERRED_RETRY_DELAY_SECONDS", 0.01)

        monkeypatch.setattr(sim, "run_cycle", first_fake)
        await sim._run_cycle_lock.acquire()
        try:
            await sim._scheduler_tick()
        finally:
            sim._run_cycle_lock.release()
        # Rebind the module global AFTER the schedule: the deferred retry
        # must still call the captured first_fake.
        monkeypatch.setattr(sim, "run_cycle", second_fake)
        for _ in range(500):
            if called["first"]:
                break
            await asyncio.sleep(0.01)
        assert called["first"] == 1
        assert called["second"] == 0

    async def test_retry_gives_up_bounded_and_logs(self, monkeypatch, caplog):
        called = {"sim": 0, "monthly": 0, "daily_core": 0}
        self._patch_stages(monkeypatch, called)
        monkeypatch.setattr(sim, "_DEFERRED_RETRY_DELAY_SECONDS", 0.01)
        monkeypatch.setattr(sim, "_DEFERRED_RETRY_WAIT_SECONDS", 0.03)
        await sim._run_cycle_lock.acquire()
        try:
            with caplog.at_level(logging.ERROR):
                await sim._scheduler_tick()
                for _ in range(100):  # let the bounded retry exhaust itself
                    await asyncio.sleep(0.01)
            assert called["sim"] == 0  # never ran into the lock holder
            assert not sim._deferred_retry_tasks  # the retry finished
            assert "giving up" in caplog.text  # ...and the loss is logged
        finally:
            sim._run_cycle_lock.release()

    async def test_stop_scheduler_cancels_pending_retries(self, monkeypatch):
        called = {"sim": 0, "monthly": 0, "daily_core": 0}
        self._patch_stages(monkeypatch, called)
        await sim._run_cycle_lock.acquire()
        try:
            await sim._scheduler_tick()
            assert sim._deferred_retry_tasks
        finally:
            sim._run_cycle_lock.release()
        sim.stop_scheduler()
        assert not sim._deferred_retry_tasks

    async def test_daily_mark_loop_defers_a_busy_snapshot(self, monkeypatch):
        """Decision F: the daily monthly-mark loop defers (not drops) a mark
        skipped because a rebalance/backfill holds the lock."""
        calls: list[str] = []
        lock = monthly._rebalance_lock
        monkeypatch.setattr(sim, "_DEFERRED_RETRY_DELAY_SECONDS", 0.01)

        async def fake_deposit():
            return {"deposited": False}

        async def fake_refresh():
            return {"refreshed": [], "errors": []}

        async def busy_snapshot():
            calls.append("snapshot")
            return {"skipped": True, "reason": "rebalance in progress"}

        monkeypatch.setattr(monthly, "deposit_allowance", fake_deposit)
        monkeypatch.setattr(monthly, "refresh_holdings", fake_refresh)
        monkeypatch.setattr(monthly, "take_snapshot", busy_snapshot)

        class _OnePass(Exception):
            pass

        real_sleep = asyncio.sleep
        big_sleeps = {"n": 0}

        async def fake_sleep(seconds):
            # Only the loop's own (long) schedule sleeps are faked; the
            # deferred retry's short delay passes through to the real sleep.
            if seconds > 1:
                big_sleeps["n"] += 1
                if big_sleeps["n"] >= 2:
                    raise _OnePass()
                return
            await real_sleep(seconds)

        monkeypatch.setattr(sim.asyncio, "sleep", fake_sleep)
        await lock.acquire()
        try:
            try:
                await sim._daily_monthly_snapshot_loop()
            except _OnePass:
                pass
            assert calls == ["snapshot"]
            assert sim._deferred_retry_tasks, "the busy mark must have been deferred"
        finally:
            lock.release()

        for _ in range(500):
            if len(calls) >= 2:
                break
            await real_sleep(0.01)
        assert len(calls) == 2  # the deferred mark made its attempt


class TestTickStaysOnProd:
    """Invariant 3: schedulers always operate on PROD, never redirected by a
    cookie — a dev-routed HTTP request leaves no trace on the tick's context."""

    @pytest.fixture(autouse=True)
    def _trading_day(self, monkeypatch):
        monkeypatch.setattr(sim, "_utcnow", lambda: _TRADING_NOW)

    async def test_tick_serves_prod_after_a_dev_request(self, two_file_dbs, client,
                                                        monkeypatch):
        # First, a dev-cookie request (the middleware sets use_db("dev") for
        # that request's task context only).
        client.cookies.set("ts_db", "dev")
        r = await client.get("/api/db/status")
        assert r.json()["active"] == "dev"

        seen: dict = {}

        async def fake_cycle():
            seen["db"] = db_mod.current_db()
            async with db_mod.Session() as s:
                s.add(Watchlist(ticker="SCHEDPROD"))
                await s.commit()
            return {"skipped": False, "trades": []}

        async def fake_monthly():
            return {"skipped": True, "reason": "x"}

        async def fake_dc():
            return {"deployment": {}}

        monkeypatch.setattr(sim, "run_cycle", fake_cycle)
        monkeypatch.setattr(monthly, "run_monthly_cycle", fake_monthly)
        monkeypatch.setattr(daily_core, "run_daily_cycle", fake_dc)

        await sim._scheduler_tick()
        assert seen["db"] == "prod"
        assert _tickers(two_file_dbs.prod_path) == {"SCHEDPROD"}
        assert _tickers(two_file_dbs.dev_path) == set()


class TestBoundedTickWaits:
    """The bounded lock waits: a busy lock delays the scheduled call, only a
    holder wedged past the bound skips (error-logged)."""

    @pytest.fixture(autouse=True)
    def _fresh_locks(self, monkeypatch):
        monkeypatch.setattr(sim, "_run_cycle_lock", asyncio.Lock())
        monkeypatch.setattr(monthly, "_rebalance_lock", asyncio.Lock())
        monkeypatch.setattr(daily_core, "_cycle_lock", asyncio.Lock())

    async def _seed_sim_account(self):
        from app.db import SimAccount
        async with db_mod.Session() as s:
            if not await s.get(SimAccount, 1):
                s.add(SimAccount(id=1, cash=settings.sim_start_cash,
                                last_allowance_month=None))
                await s.commit()

    async def test_sim_run_cycle_waits_then_proceeds(self, two_file_dbs, monkeypatch):
        """The REAL run_cycle body (Phase 2) runs after the bounded wait — no
        body extraction, so this exercises the actual function."""
        monkeypatch.setattr(sim, "_TICK_LOCK_WAIT_SECONDS", 10)
        monkeypatch.setattr(settings, "sim_benchmark_enabled", False)
        monkeypatch.setattr(settings, "sim_strategy", "deterministic")

        async def no_candidates():
            return []

        monkeypatch.setattr(sim, "_candidate_tickers", no_candidates)
        await self._seed_sim_account()

        await sim._run_cycle_lock.acquire()
        try:
            task = asyncio.create_task(sim.run_cycle())
            await asyncio.sleep(0)  # let the task reach the held lock
            assert not task.done()  # waiting, not skipped
        finally:
            sim._run_cycle_lock.release()
        result = await asyncio.wait_for(task, 30)
        assert "skipped" not in result, "the delayed cycle must still run"
        # The cycle wrote its snapshot into the PROD file it served.
        assert _count_rows(two_file_dbs.prod_path, "sim_snapshots") >= 1
        assert _count_rows(two_file_dbs.dev_path, "sim_snapshots") == 0

    async def test_sim_run_cycle_times_out_and_skips(self, two_file_dbs, monkeypatch,
                                                     caplog):
        monkeypatch.setattr(sim, "_TICK_LOCK_WAIT_SECONDS", 0.05)
        await sim._run_cycle_lock.acquire()
        try:
            with caplog.at_level(logging.ERROR):
                result = await sim.run_cycle()
        finally:
            sim._run_cycle_lock.release()
        assert result == {"skipped": True, "reason": "already running"}
        assert "still held" in caplog.text

    async def test_sim_run_cycle_release_survives_a_body_exception(self, monkeypatch):
        """The manual acquire is paired with a finally release: a failure
        must not wedge the lock.

        The injection point is current_config(), which runs inside the outer
        try but BEFORE the inner stage-try (whose catch-all would swallow the
        exception) and BEFORE the stamp token exists: the error escapes
        run_cycle entirely, and the outer finally must still release the lock.
        (Do NOT hold _run_cycle_lock across the call — the bounded acquire
        would then wait on the test's own lock: a self-deadlock.)
        """
        async def boom():
            raise AssertionError("unreachable")

        monkeypatch.setattr(sim, "current_config", boom)
        with pytest.raises(AssertionError):
            await sim.run_cycle()
        # The lock was released by the failing call itself (outer finally),
        # even though the exception escaped run_cycle.
        assert not sim._run_cycle_lock.locked()

    async def test_daily_core_manual_wait_false_skips_immediately(self, monkeypatch):
        await daily_core._cycle_lock.acquire()
        try:
            r = await daily_core.run_daily_cycle(wait=False)
        finally:
            daily_core._cycle_lock.release()
        assert r == {"skipped": True, "reason": "already running"}

    async def test_daily_core_waits_then_proceeds(self, monkeypatch):
        """wait=True (the scheduler's call) delays the cycle. The body's first
        gate returns 'disabled' — proving the body ran after the wait without
        needing market data."""
        monkeypatch.setattr(settings, "sim_daily_core_enabled", False)
        monkeypatch.setattr(daily_core, "_TICK_LOCK_WAIT_SECONDS", 10)
        await daily_core._cycle_lock.acquire()
        try:
            task = asyncio.create_task(daily_core.run_daily_cycle())
            await asyncio.sleep(0)
            assert not task.done()
        finally:
            daily_core._cycle_lock.release()
        r = await asyncio.wait_for(task, 5)
        assert r == {"skipped": True, "reason": "daily-core portfolio disabled"}

    async def test_daily_core_times_out_and_skips(self, monkeypatch, caplog):
        monkeypatch.setattr(daily_core, "_TICK_LOCK_WAIT_SECONDS", 0.05)
        await daily_core._cycle_lock.acquire()
        try:
            with caplog.at_level(logging.ERROR):
                result = await daily_core.run_daily_cycle()
        finally:
            daily_core._cycle_lock.release()
        assert result == {"skipped": True, "reason": "already running"}
        assert "still held" in caplog.text

    async def test_monthly_cycle_busy_path_waits_and_runs_locked_body(self, monkeypatch):
        monkeypatch.setattr(settings, "sim_monthly_enabled", True)
        monkeypatch.setattr(monthly, "_TICK_LOCK_WAIT_SECONDS", 10)
        ran: list[tuple] = []

        async def fake_locked_body(force, now=None):
            ran.append((force, now))
            return {"rebalanced": True}

        monkeypatch.setattr(monthly, "_run_rebalance_locked", fake_locked_body)
        await monthly._rebalance_lock.acquire()
        try:
            task = asyncio.create_task(monthly.run_monthly_cycle())
            await asyncio.sleep(0)
            assert ran == []
        finally:
            monthly._rebalance_lock.release()
        r = await asyncio.wait_for(task, 5)
        assert r == {"rebalanced": True}
        assert ran == [(False, None)]  # the (force, now) signature is preserved

    async def test_monthly_cycle_free_path_delegates_to_run_rebalance(self, monkeypatch):
        monkeypatch.setattr(settings, "sim_monthly_enabled", True)
        seen: dict = {}

        async def fake_rebalance(force=False, now=None):
            seen["force"], seen["now"] = force, now
            return {"rebalanced": True}

        monkeypatch.setattr(monthly, "run_rebalance", fake_rebalance)
        clock = datetime(2026, 8, 28, 22, 0)
        out = await monthly.run_monthly_cycle(now=clock)
        assert out == {"rebalanced": True}
        assert seen == {"force": False, "now": clock}

    async def test_monthly_rebalance_keeps_its_fail_fast_answer(self, monkeypatch):
        """The MANUAL run_rebalance (the /api/monthly/run path) must never
        queue behind a backfill: busy -> immediate skip.

        t1 parks inside _pending_rebuild — the FIRST await inside the locked
        body, reached before any DB work (current_config / _paper_stamp / the
        Session query all come later). The previous version parked t1 in
        refresh_data, which t1 can never reach without a database fixture:
        the task died on the first DB access, in_refresh never fired and the
        test hung forever. Parking here proves the same contract with no DB.
        """
        in_body = asyncio.Event()
        release = asyncio.Event()

        async def fake_pending(now):
            in_body.set()
            await release.wait()
            return None  # once released: "no pending month-end rebuild"

        monkeypatch.setattr(monthly, "_pending_rebuild", fake_pending)

        t1 = asyncio.create_task(monthly.run_rebalance())
        await in_body.wait()
        assert monthly._rebalance_lock.locked()  # t1 owns it, mid-body
        t2 = await monthly.run_rebalance()       # must fail fast, not queue
        assert t2.get("skipped") is True and "already running" in t2["reason"]
        release.set()
        out = await t1
        # The released holder completes its own pass (nothing pending today).
        assert out.get("skipped") is True

    async def test_monthly_cycle_times_out_and_skips(self, monkeypatch, caplog):
        monkeypatch.setattr(settings, "sim_monthly_enabled", True)
        monkeypatch.setattr(monthly, "_TICK_LOCK_WAIT_SECONDS", 0.05)

        async def fail_body(force, now=None):  # pragma: no cover — must not run
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
