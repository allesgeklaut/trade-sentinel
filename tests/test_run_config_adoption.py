"""Run-configuration ADOPTION tests (Phase 2, batch R2).

What this suite proves, per the audit's Phase 2 acceptance:

* every forward engine write (trade/allowance/snapshot/ranking/rebalance) is
  stamped with the run's ONE immutable configuration + epoch, so a paper run is
  reconstructable and its configuration is frozen at run start;
* the number of configurations tried is the DISTINCT RunConfig count, while
  executions append one RunExecution each (provenance summary, not an immutable
  data snapshot);
* a backfill cannot overwrite a forward paper record (refused before any
  mutation), and a re-backfill replaces only its own synthetic rows;
* a reset preserves history bit-for-bit, records an epoch boundary and re-seeds
  the book — the curve/metrics restart at the new epoch while the ledger
  survives, and a same-month deposit works again in the new epoch;
* epoch-scoped readers (curve, realized-vol returns, allowance sums) never
  splice a closed epoch into the fresh book;
* the seed capital (``*_start_cash``) counts as external contribution.

OFFLINE ONLY: every test uses an in-memory SQLite DB, no provider/network call,
no production DB and no repo writes (`_offline_guard` blocks and COUNTS any
AF_INET/AF_INET6 connect attempt; the counter must stay empty).
"""

from __future__ import annotations

import asyncio
import json
import socket
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock

import pandas as pd
import pytest
from sqlalchemy import StaticPool, func, inspect, select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app import daily_core, monthly, run_config, sim
from app.config import settings
from app.db import (
    Base,
    Candle,
    DailyCoreAccount,
    DailyCoreAllowance,
    DailyCorePosition,
    DailyCoreRanking,
    DailyCoreSnapshot,
    DailyCoreTrade,
    MonthlyAccount,
    MonthlyAllowance,
    MonthlyPosition,
    MonthlyRebalance,
    MonthlySnapshot,
    MonthlyTrade,
    RunConfig,
    RunEpoch,
    RunExecution,
    SimAccount,
    SimAllowance,
    SimPosition,
    SimSnapshot,
    SimTrade,
)

# ---------------------------------------------------------------------------
# Offline guard: block + COUNT every IP connect attempt
# ---------------------------------------------------------------------------

_NETWORK_ATTEMPTS: list[tuple] = []


@pytest.fixture(autouse=True)
def _offline_guard(monkeypatch):
    """Refuse AF_INET/AF_INET6 connects and record every attempt.

    A test that reaches the network must FAIL (and show the attempt), not
    silently depend on a provider. AF_UNIX/AF_LOCAL (e.g. nothing here) stays
    untouched.
    """
    _NETWORK_ATTEMPTS.clear()
    real_connect = socket.socket.connect
    real_connect_ex = socket.socket.connect_ex

    def _blocked(sock, address, *args, **kwargs):
        if sock.family in (socket.AF_INET, socket.AF_INET6):
            _NETWORK_ATTEMPTS.append((sock.family, address))
            raise OSError(f"network blocked by the offline test guard: {address!r}")
        return real_connect(sock, address, *args, **kwargs)

    def _blocked_ex(sock, address, *args, **kwargs):
        if sock.family in (socket.AF_INET, socket.AF_INET6):
            _NETWORK_ATTEMPTS.append((sock.family, address))
            return 111  # ECONNREFUSED, never reaches the stack
        return real_connect_ex(sock, address, *args, **kwargs)

    monkeypatch.setattr(socket.socket, "connect", _blocked)
    monkeypatch.setattr(socket.socket, "connect_ex", _blocked_ex)
    yield
    assert _NETWORK_ATTEMPTS == [], f"network attempted: {_NETWORK_ATTEMPTS}"


# ---------------------------------------------------------------------------
# DB fixture: one in-memory DB, every engine session factory patched onto it
# ---------------------------------------------------------------------------


@pytest.fixture
async def run_db(monkeypatch, tmp_path):
    """Replace the engines' Session factories with an in-memory SQLite DB.

    The engines resolve their run-config/provenance fallbacks through THEIR
    module-global ``Session`` (never ``db.Session`` directly), so patching
    these names is what keeps every standalone writer off an external DB.
    """
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    factory = async_sessionmaker(engine, expire_on_commit=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    import app.db as db_mod
    import app.edgar as edgar_mod
    import app.fundamentals as fund_mod

    monkeypatch.setattr(daily_core, "Session", factory)
    monkeypatch.setattr(monthly, "Session", factory)
    monkeypatch.setattr(sim, "Session", factory)
    monkeypatch.setattr(db_mod, "Session", factory)
    # Modules that bound ``Session`` at import time (they read the same DB).
    monkeypatch.setattr(fund_mod, "Session", factory)
    monkeypatch.setattr(edgar_mod, "Session", factory)
    monkeypatch.setattr(daily_core, "_STATE_FILE", tmp_path / "daily_core_state.json")
    monkeypatch.setattr(sim, "_STATE_FILE", tmp_path / "sim_state.json")
    sim._reset_session_guard()
    yield factory
    await engine.dispose()


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


async def _all(factory, model) -> list:
    async with factory() as s:
        return list((await s.scalars(select(model).order_by(model.id))).all())


async def _count(factory, model) -> int:
    async with factory() as s:
        return int((await s.scalar(select(func.count()).select_from(model))) or 0)


def _dump(rows) -> list[tuple]:
    """Every mapped column of every row, comparable bit-for-bit."""
    return [
        tuple((attr.key, getattr(row, attr.key)) for attr in inspect(row).mapper.column_attrs)
        for row in rows
    ]


def _assert_stamped(rows, *, mode: str, epoch: int, config_id: int | None = None):
    assert rows, "expected written rows"
    for row in rows:
        assert row.mode == mode, (row.id, row.mode)
        assert row.epoch == epoch, (row.id, row.epoch)
        assert row.config_id is not None, row.id
        if config_id is not None:
            assert row.config_id == config_id, (row.id, row.config_id)


# ---------------------------------------------------------------------------
# daily-core live-cycle environment (month-end band decision faked flat)
# ---------------------------------------------------------------------------

DC_NOW = datetime(2026, 10, 1, 21, 0, tzinfo=UTC)
DC_ME = pd.Timestamp("2026-09-30")
DC_FRAME = pd.DataFrame(
    {
        "roe": [0.30, 0.25, 0.20, 0.10],
        "p_fcf": [10.0, 15.0, 20.0, 25.0],
        "mcap": [1e10] * 4,
        "dollar_vol": [1e8] * 4,
        "mom": [0.5, 0.4, 0.3, 0.2],
        "price": [100.0, 50.0, 25.0, 10.0],
        "vol": [0.2] * 4,
        "eligible": [True] * 4,
    },
    index=pd.Index(["AAA", "BBB", "CCC", "DDD"], name="ticker"),
)


def _patch_dc_decision(monkeypatch) -> None:
    """Pin the month-end decision + flat $100 marks/fills (no DB candles)."""

    async def fake_stored_me(_now):
        return DC_ME

    async def fake_core_frame(_me):
        return DC_FRAME, ["AAA", "BBB", "CCC", "DDD"]

    async def fake_price_map(tickers, asof=None):
        return {t: 100.0 for t in tickers}

    async def fake_open_map(tickers, d):
        return {t: 100.0 for t in tickers}

    monkeypatch.setattr(daily_core, "_stored_month_end", fake_stored_me)
    monkeypatch.setattr(daily_core, "_core_frame", fake_core_frame)
    monkeypatch.setattr(monthly, "_price_usd_map", fake_price_map)
    monkeypatch.setattr(monthly, "_open_usd_map", fake_open_map)
    monkeypatch.setattr(daily_core, "_now", lambda: DC_NOW)
    monkeypatch.setattr(daily_core, "_current_month", lambda: "2026-10")
    # The runtime selections read the fixture's private state file (run_db),
    # so the real getters stay in play and a mid-test set_variant() is seen.


# ---------------------------------------------------------------------------
# 1. One live cycle per engine -> every writer stamped paper/epoch 1/config_id
# ---------------------------------------------------------------------------


async def test_daily_core_live_cycle_stamps_every_writer(run_db, monkeypatch):
    _patch_dc_decision(monkeypatch)
    assert (await daily_core.deposit_allowance())["deposited"] is True
    out = await daily_core.run_deployment(now=DC_NOW)
    assert out.get("deployed") is True, out
    assert [t["side"] for t in out["trades"]] == ["BUY"] * 4
    assert (await daily_core.take_snapshot())["snapshotted"] is True

    _assert_stamped(await _all(run_db, DailyCoreTrade), mode="paper", epoch=1)
    _assert_stamped(await _all(run_db, DailyCoreAllowance), mode="paper", epoch=1)
    _assert_stamped(await _all(run_db, DailyCoreSnapshot), mode="paper", epoch=1)
    _assert_stamped(await _all(run_db, DailyCoreRanking), mode="paper", epoch=1)

    configs = await _all(run_db, RunConfig)
    assert [(c.engine, c.mode) for c in configs] == [("daily_core", "paper")]
    config_id = configs[0].id
    stamped_rows = [
        row
        for model in (DailyCoreTrade, DailyCoreAllowance, DailyCoreSnapshot,
                      DailyCoreRanking)
        for row in await _all(run_db, model)
    ]
    assert stamped_rows and all(row.config_id == config_id for row in stamped_rows)

    executions = await _all(run_db, RunExecution)
    assert executions, "every logical run must append a RunExecution"
    assert all(e.engine == "daily_core" and e.mode == "paper" and e.epoch == 1
               for e in executions)
    # per-run provenance is the DATA SUMMARY (not an immutable data snapshot)
    summary = json.loads(executions[0].provenance)
    assert summary["scope"]["known"] is True
    # The provenance clock is the WALL clock, deliberately independent of the
    # injectable decision clock (recording must never perturb a frozen
    # decision/fill timestamp); production has the two coincide.
    recorded = datetime.fromisoformat(summary["asof"])
    assert abs((datetime.now(UTC) - recorded).total_seconds()) < 120
    assert executions[0].data_asof is not None
    assert summary["asof"] == executions[0].data_asof.isoformat()


async def _install_monthly_live(run_db, monkeypatch) -> datetime:
    """Seed a 4-name tape + facts and patch the network refresh; return F."""
    from app.screener import tickers as universe_tickers

    universe = [t for t in universe_tickers("diversified-plus") if "." not in t][:4]
    dates = pd.bdate_range(end="2026-09-30", periods=400)
    fill_session = pd.Timestamp("2026-10-01")
    prices = {t: 100.0 + i for i, t in enumerate(universe)}

    all_days = dates.union(pd.DatetimeIndex([fill_session]))
    async with run_db() as s:
        for t in universe:
            for day in all_days:
                px = prices[t]
                s.add(Candle(ticker=t, timestamp=day.to_pydatetime(), open=px,
                             high=px, low=px, close=px, volume=2e6))
            s.add_all(_fundamental_rows(t))
        await s.commit()

    async def fake_refresh_data(_tickers):
        return [], {}

    monkeypatch.setattr(monthly, "refresh_data", fake_refresh_data)
    monkeypatch.setattr(settings, "sim_monthly_target_n", 2)
    monkeypatch.setattr(settings, "sim_monthly_hold_band", 3)
    return datetime(2026, 10, 1, 22, 0, tzinfo=UTC)


async def test_monthly_live_cycle_stamps_every_writer(run_db, monkeypatch):
    clock = await _install_monthly_live(run_db, monkeypatch)
    out = await monthly.run_rebalance(now=clock)
    assert out.get("rebalanced") is True, out
    assert out["trades"], "the fixture must actually fill legs"

    _assert_stamped(await _all(run_db, MonthlyTrade), mode="paper", epoch=1)
    _assert_stamped(await _all(run_db, MonthlyAllowance), mode="paper", epoch=1)
    _assert_stamped(await _all(run_db, MonthlySnapshot), mode="paper", epoch=1)
    _assert_stamped(await _all(run_db, MonthlyRebalance), mode="paper", epoch=1)

    configs = await _all(run_db, RunConfig)
    assert [(c.engine, c.mode) for c in configs] == [("monthly", "paper")]


async def test_sim_live_cycle_and_standalone_buy_stamp_paper(run_db, monkeypatch):
    clock = datetime(2026, 10, 2, 23, 30, tzinfo=UTC)
    monkeypatch.setattr(sim, "_utcnow", lambda: clock)
    monkeypatch.setattr(sim.settings, "sim_benchmark_enabled", False)
    monkeypatch.setattr(sim, "refresh_many", AsyncMock(return_value=([], [])))
    monkeypatch.setattr(sim, "_benchmark_snapshot", AsyncMock())
    monkeypatch.setattr(sim, "_deterministic_decide", AsyncMock(return_value=[]))

    async def fake_inputs(tickers, held_entries=None, now=None):
        return sim._DecisionInputs(
            f_ts=pd.Timestamp(clock.date()), s_ts=pd.Timestamp(clock.date()) - timedelta(days=1),
            signals={}, prices={}, session_bars={}, ratchets={})

    async def fake_valuate():
        return {"cash": 1000.0, "positions": [], "positions_value": 0.0,
                "total_equity": 1000.0, "allowance_total": 1000.0}

    monkeypatch.setattr(sim, "_decision_inputs", fake_inputs)
    monkeypatch.setattr(sim, "valuate", fake_valuate)

    out = await sim.run_cycle()
    assert out.get("skipped") is not True, out
    _assert_stamped(await _all(run_db, SimAllowance), mode="paper", epoch=1)
    _assert_stamped(await _all(run_db, SimSnapshot), mode="paper", epoch=1)

    # A standalone executor is a small logical run of its own — still paper.
    async with run_db() as s:
        acc = await s.get(SimAccount, 1)
        acc.cash = 500.0
        await s.commit()
    trade = await sim._exec_buy("AAPL", 100.0, 500.0, "standalone")
    assert trade is not None
    _assert_stamped(await _all(run_db, SimTrade), mode="paper", epoch=1)
    configs = await _all(run_db, RunConfig)
    assert [(c.engine, c.mode) for c in configs] == [("sim", "paper")]


def _fundamental_rows(ticker: str) -> list:
    """A minimal point-in-time qv-mom fact set public before 2026-09-30."""
    from app.db import Fundamental

    quality = 10.0
    return [
        Fundamental(ticker=ticker, tag="StockholdersEquity", start=None,
                    end="2025-06-30", filed="2026-01-10", val=1e9, currency="USD"),
        Fundamental(ticker=ticker, tag="NetIncomeLoss", start="2024-07-01",
                    end="2025-06-30", filed="2026-01-10", val=1e8 * quality,
                    currency="USD"),
        Fundamental(ticker=ticker, tag="NetCashProvidedByUsedInOperatingActivities",
                    start="2024-07-01", end="2025-06-30", filed="2026-01-10",
                    val=5e8, currency="USD"),
        Fundamental(ticker=ticker, tag="PaymentsToAcquirePropertyPlantAndEquipment",
                    start="2024-07-01", end="2025-06-30", filed="2026-01-10",
                    val=-1e8, currency="USD"),
        Fundamental(ticker=ticker, tag="CommonStockSharesOutstanding", start=None,
                    end="2025-06-30", filed="2026-01-10", val=1e8, currency="USD"),
    ]


# ---------------------------------------------------------------------------
# 2. A configuration change mints a NEW config id; the count is DISTINCT configs
# ---------------------------------------------------------------------------


async def test_config_change_mints_a_new_config_id(run_db, monkeypatch):
    _patch_dc_decision(monkeypatch)
    await daily_core.deposit_allowance()
    first = await daily_core.run_deployment(now=DC_NOW)
    assert first.get("deployed") is True
    ranking = (await _all(run_db, DailyCoreRanking))[0]
    first_config = ranking.config_id

    # A runtime configuration change (the UI's momentum-variant dropdown).
    daily_core.set_variant("residual")
    monkeypatch.setattr(daily_core, "current_variant", lambda: "residual")

    second = await daily_core.run_deployment(now=DC_NOW)
    assert second.get("deployed") is True, second
    ranking = (await _all(run_db, DailyCoreRanking))[0]
    assert ranking.config_id != first_config, "the new decision must carry the new config"
    await daily_core.take_snapshot()

    configs = await _all(run_db, RunConfig)
    assert len(configs) == 2, (
        "distinct REGISTERED configurations — the configuration registry counts "
        "what was tried; RunExecution counts admitted runs, never read polls")
    snapshots = await _all(run_db, DailyCoreSnapshot)
    assert snapshots[-1].config_id == ranking.config_id

    # Executions are NOT deduped: the same config repeated is a new execution.
    executions = await _all(run_db, RunExecution)
    assert len(executions) >= 3
    assert len({e.config_id for e in executions}) == 2


async def test_cycle_provenance_is_recorded_after_the_refresh(run_db, monkeypatch):
    """The RunExecution must describe the cache the decision READ: a candle
    written by the cycle's own refresh has to appear in its provenance."""
    _patch_dc_decision(monkeypatch)
    member = daily_core.current_config()["universe_members"][0]

    async def refreshing():
        async with run_db() as s:
            s.add(Candle(
                ticker=member,
                timestamp=(DC_NOW - timedelta(days=1)).replace(tzinfo=None),
                open=100.0, high=100.0, low=100.0, close=100.0, volume=1e6,
                source="mock-provider", adjustment_basis="split_dividend",
                ingested_at=DC_NOW - timedelta(hours=1)))
            await s.commit()
        return [], []

    monkeypatch.setattr(daily_core, "refresh_data", refreshing)
    out = await daily_core.run_daily_cycle(now=DC_NOW)
    assert out.get("skipped") is not True, out

    executions = await _all(run_db, RunExecution)
    assert len(executions) == 1, "one logical cycle records exactly one execution"
    summary = json.loads(executions[0].provenance)
    sources = {row["source"]: row["count"] for row in summary["candles"]["by_source"]}
    assert sources.get("mock-provider") == 1


async def test_read_polls_and_skipped_passes_register_no_execution(run_db, monkeypatch):
    """Only ADMITTED runs append a RunExecution: a status poll that recomputes
    the ranking cache, and a pass skipped before its gates, are not runs. The
    RunConfig row they may create counts a REGISTERED configuration, never an
    execution."""
    _patch_dc_decision(monkeypatch)
    band, _picks, _frame = await daily_core.compute_targets()
    assert band == ["AAA", "BBB", "CCC", "DDD"]
    assert await _count(run_db, RunExecution) == 0
    assert await _count(run_db, RunConfig) == 1  # registered, not executed

    async def no_month_end(_now):
        return None

    monkeypatch.setattr(daily_core, "_stored_month_end", no_month_end)
    skipped = await daily_core.run_deployment(now=DC_NOW)
    assert skipped.get("skipped") is True
    assert await _count(run_db, RunExecution) == 0


async def test_unknown_config_ranking_row_is_recomputed(run_db, monkeypatch):
    """The singleton ranking is a DERIVED cache: a legacy (NULL config) row
    cannot be reused under a stamped run — it must be recomputed and re-stored
    with the current config, or the cache would rank under a provenance it
    does not have."""
    _patch_dc_decision(monkeypatch)
    async with run_db() as s:
        s.add(DailyCoreRanking(id=1, ranking_date=DC_ME.to_pydatetime(),
                               band="STALE,STALE2", picks="STALE", mode=None,
                               config_id=None, epoch=1))
        await s.commit()

    calls = {"n": 0}

    async def counting_frame(_me):
        calls["n"] += 1
        return DC_FRAME, ["AAA", "BBB", "CCC", "DDD"]

    monkeypatch.setattr(daily_core, "_core_frame", counting_frame)
    band, picks, _frame = await daily_core.compute_targets()
    assert calls["n"] == 1, "an unstamped cache row must be invalidated"
    assert band == ["AAA", "BBB", "CCC", "DDD"]

    row = (await _all(run_db, DailyCoreRanking))[0]
    config = (await _all(run_db, RunConfig))[0]
    assert row.mode == "paper" and row.config_id == config.id


# ---------------------------------------------------------------------------
# 3. Backfill in a fresh epoch: stamped rows; re-backfill replaces only its own
# ---------------------------------------------------------------------------

DC_BACKFILL_TICKERS = ["A01", "B02", "C03", "D04"]


def _install_dc_backfill(monkeypatch, tmp_path):
    monkeypatch.setattr(daily_core, "_STATE_FILE", tmp_path / "bf_state.json")
    monkeypatch.setattr(daily_core, "universe_tickers", lambda _u: list(DC_BACKFILL_TICKERS))
    monkeypatch.setattr(daily_core, "current_universe", lambda: "probe-universe")
    monkeypatch.setattr(daily_core, "_current_month", lambda: "2025-02")

    async def fake_fund(_tickers):
        return {t: {"x": []} for t in DC_BACKFILL_TICKERS}

    monkeypatch.setattr(daily_core.fundamentals_mod, "load_fundamentals", fake_fund)

    dates = pd.bdate_range("2024-01-02", "2025-02-28")
    close = pd.DataFrame(100.0, index=dates, columns=DC_BACKFILL_TICKERS)
    open_px = close.copy()
    vol = pd.DataFrame(1e6, index=dates, columns=DC_BACKFILL_TICKERS)
    frame = pd.DataFrame(
        {"eligible": [True] * len(DC_BACKFILL_TICKERS)},
        index=pd.Index(DC_BACKFILL_TICKERS, name="ticker"),
    )
    monkeypatch.setattr(monthly, "eligible_frame", lambda m, c, v, f: frame)
    monkeypatch.setattr(monthly, "_qv_order",
                        lambda fr: (fr, list(DC_BACKFILL_TICKERS)))
    monkeypatch.setattr(settings, "sim_monthly_target_n", 2)
    monkeypatch.setattr(settings, "sim_monthly_hold_band", 3)
    return monthly.BackfillPreload(
        tickers=list(DC_BACKFILL_TICKERS),
        fund={t: {"x": []} for t in DC_BACKFILL_TICKERS},
        open=open_px, close=close, vol=vol, frames={},
    )


async def test_backfill_stamps_and_rebackfill_replaces_only_its_own(run_db, monkeypatch, tmp_path):
    preload = _install_dc_backfill(monkeypatch, tmp_path)

    first = await daily_core.backfill(start="2025-01-02", preload=preload)
    assert first["ok"] is True, first
    trade_count = await _count(run_db, DailyCoreTrade)
    snap_count = await _count(run_db, DailyCoreSnapshot)
    allow_count = await _count(run_db, DailyCoreAllowance)
    assert trade_count > 0 and snap_count > 0 and allow_count > 0

    _assert_stamped(await _all(run_db, DailyCoreTrade), mode="backfill", epoch=1)
    _assert_stamped(await _all(run_db, DailyCoreSnapshot), mode="backfill", epoch=1)
    _assert_stamped(await _all(run_db, DailyCoreAllowance), mode="backfill", epoch=1)

    configs = await _all(run_db, RunConfig)
    assert [(c.engine, c.mode) for c in configs] == [("daily_core", "backfill")]

    # A re-backfill (same config) replaces its OWN rows — no duplication.
    second = await daily_core.backfill(start="2025-01-02", preload=preload)
    assert second["ok"] is True, second
    assert await _count(run_db, DailyCoreTrade) == trade_count
    assert await _count(run_db, DailyCoreSnapshot) == snap_count
    assert await _count(run_db, DailyCoreAllowance) == allow_count
    assert len(await _all(run_db, RunConfig)) == 1  # same configuration, one row
    assert len(await _all(run_db, RunExecution)) == 2  # two EXECUTIONS recorded


async def test_monthly_backfill_stamps_rows(run_db, monkeypatch, tmp_path):
    tickers = ["AAA", "BBB"]
    monkeypatch.setattr(monthly, "universe_tickers", lambda _u: list(tickers))
    monkeypatch.setattr(monthly, "_strategy_universe", lambda: "probe-universe")
    monkeypatch.setattr(monthly, "_current_month", lambda: "2025-02")

    async def fake_fund(_tickers):
        return {t: {"x": []} for t in tickers}

    monkeypatch.setattr(monthly.fundamentals_mod, "load_fundamentals", fake_fund)

    dates = pd.bdate_range("2024-01-02", "2025-02-28")
    close = pd.DataFrame(100.0, index=dates, columns=tickers)
    open_px = close.copy()
    vol = pd.DataFrame(1e6, index=dates, columns=tickers)
    frame = pd.DataFrame({"eligible": [True, True]},
                         index=pd.Index(tickers, name="ticker"))
    monkeypatch.setattr(monthly, "eligible_frame", lambda m, c, v, f: frame)
    monkeypatch.setattr(monthly, "_qv_order", lambda fr: (fr, list(tickers)))
    monkeypatch.setattr(monthly, "pick_portfolio",
                        lambda d, c, v, f, holdings=None, frame=None:
                        (sorted(tickers), frame))
    monkeypatch.setattr(settings, "sim_monthly_target_n", 2)
    monkeypatch.setattr(settings, "sim_monthly_hold_band", 3)

    preload = monthly.BackfillPreload(
        tickers=list(tickers), fund={t: {"x": []} for t in tickers},
        open=open_px, close=close, vol=vol, frames={},
    )
    out = await monthly.backfill(start="2025-01-02", preload=preload)
    assert out["ok"] is True, out

    _assert_stamped(await _all(run_db, MonthlyTrade), mode="backfill", epoch=1)
    _assert_stamped(await _all(run_db, MonthlySnapshot), mode="backfill", epoch=1)
    _assert_stamped(await _all(run_db, MonthlyAllowance), mode="backfill", epoch=1)
    _assert_stamped(await _all(run_db, MonthlyRebalance), mode="backfill", epoch=1)
    configs = await _all(run_db, RunConfig)
    assert [(c.engine, c.mode) for c in configs] == [("monthly", "backfill")]


async def test_sim_backfill_stamps_rows(run_db, monkeypatch, tmp_path):
    from app import optimize as opt
    from app.analysis import signal_series
    from tests.test_optimize import _gen_candles

    up = signal_series(_gen_candles(100.0, 0.01, n=300, seed=11))
    dn = signal_series(_gen_candles(100.0, -0.01, n=300, seed=12))

    async def fake_load_series(_tickers, start=None):
        return {"UP": up, "DN": dn}

    monkeypatch.setattr(opt, "_load_series", fake_load_series)
    monkeypatch.setattr(sim, "_utcnow", lambda: datetime(2026, 10, 2, 23, 30, tzinfo=UTC))

    out = await sim.backfill(start="2025-03-01")
    assert out["ok"] is True, out

    _assert_stamped(await _all(run_db, SimTrade), mode="backfill", epoch=1)
    _assert_stamped(await _all(run_db, SimSnapshot), mode="backfill", epoch=1)
    _assert_stamped(await _all(run_db, SimAllowance), mode="backfill", epoch=1)
    configs = await _all(run_db, RunConfig)
    assert [(c.engine, c.mode) for c in configs] == [("sim", "backfill")]


async def test_backfills_on_an_empty_db_record_no_execution(run_db, monkeypatch):
    """A backfill rejected by a DATA gate is not an executed run: no
    RunExecution row for any engine (the config row it resolves is only a
    registered configuration)."""
    async def no_fund(_tickers):
        return {}

    monkeypatch.setattr(daily_core, "universe_tickers", lambda _u: ["AAA"])
    monkeypatch.setattr(daily_core, "current_universe", lambda: "probe")
    monkeypatch.setattr(daily_core.fundamentals_mod, "load_fundamentals", no_fund)
    dc_out = await daily_core.backfill(start="all")
    assert dc_out["ok"] is False

    monkeypatch.setattr(monthly, "universe_tickers", lambda _u: ["AAA"])
    monkeypatch.setattr(monthly, "_strategy_universe", lambda: "probe")
    monkeypatch.setattr(monthly.fundamentals_mod, "load_fundamentals", no_fund)
    m_out = await monthly.backfill(start="all")
    assert m_out["ok"] is False

    from app import optimize as opt

    async def no_series(_tickers, start=None):
        return {}

    monkeypatch.setattr(opt, "_load_series", no_series)
    monkeypatch.setattr(sim, "_candidate_tickers", AsyncMock(return_value=[]))
    s_out = await sim.backfill(start="all")
    assert s_out["ok"] is False

    assert await _count(run_db, RunExecution) == 0


async def test_backfill_persist_fences_a_queued_standalone_writer(
        run_db, monkeypatch, tmp_path):
    """REGRESSION (backfill/stamp race): a standalone writer that commits
    during the replay must not be erased by the final persist.

    Deterministic interleaving, no polling: when the backfill's FINAL
    in-transaction forward guard runs (it holds ``_state_write_lock``), the
    test starts a real ``deposit_allowance`` task and yields ONCE so it reaches
    the lock and blocks there. The writer is then paused inside its own lock
    section; the backfill commits; the test reads the post-backfill state,
    releases the writer and verifies the paper row + its cash effect survive
    the backfill's rows bit-for-bit.
    """
    preload = _install_dc_backfill(monkeypatch, tmp_path)
    monkeypatch.setattr(daily_core, "_current_month", lambda: "2025-03")
    real_guard = run_config.has_paper_rows
    real_locked = daily_core._deposit_allowance_locked
    writer: dict = {}
    release = asyncio.Event()

    async def paused_deposit(month, *, mode=None, config_id=None, epoch=None):
        # Runs AFTER the writer acquired the state lock (i.e. after the
        # backfill released it): hold it until the test has read the
        # backfill's committed state.
        await release.wait()
        return await real_locked(month, mode=mode, config_id=config_id,
                                 epoch=epoch)

    async def hooked_guard(engine, cur, *, session=None, session_factory=None):
        result = await real_guard(engine, cur, session=session,
                                  session_factory=session_factory)
        if session is not None:
            # Inside the final persistence transaction (state lock held).
            writer["task"] = asyncio.create_task(daily_core.deposit_allowance())
            await asyncio.sleep(0)  # one scheduling turn: the writer blocks
            assert daily_core._state_write_lock.locked()
            assert not writer["task"].done(), "the writer must wait for the lock"
        return result

    monkeypatch.setattr(daily_core, "_deposit_allowance_locked", paused_deposit)
    monkeypatch.setattr(run_config, "has_paper_rows", hooked_guard)

    out = await daily_core.backfill(start="2025-01-02", preload=preload)
    assert out["ok"] is True, out
    # The writer is now blocked inside its lock section, before any write.
    assert not writer["task"].done()
    def backfill_dump(rows):
        return _dump([row for row in rows if row.mode == "backfill"])

    backfill_rows = {
        model: backfill_dump(await _all(run_db, model))
        for model in (DailyCoreTrade, DailyCoreSnapshot, DailyCoreAllowance)
    }
    assert all(backfill_rows.values())
    cash_after_backfill = (await daily_core.valuate(now=DC_NOW))["cash"]

    release.set()
    deposited = await asyncio.wait_for(writer["task"], 30)
    assert deposited["deposited"] is True

    # The backfill's rows were not rewritten ...
    for model, snapshot in backfill_rows.items():
        assert backfill_dump(await _all(run_db, model)) == snapshot, model.__name__
    # ... and the paper row + its cash landed on top of them.
    paper_rows = [r for r in await _all(run_db, DailyCoreAllowance)
                  if r.mode == "paper"]
    assert len(paper_rows) == 1
    assert paper_rows[0].month == "2025-03"
    assert paper_rows[0].config_id is not None
    assert (await daily_core.valuate(now=DC_NOW))["cash"] == pytest.approx(
        cash_after_backfill + settings.sim_monthly_contribution, abs=0.01)


# ---------------------------------------------------------------------------
# 4. THE PIN: cycle then backfill in one epoch -> REFUSED, nothing changed
# ---------------------------------------------------------------------------


async def test_backfill_after_a_paper_cycle_is_refused_and_changes_nothing(
        run_db, monkeypatch, tmp_path):
    preload = _install_dc_backfill(monkeypatch, tmp_path)
    _patch_dc_decision(monkeypatch)

    # A forward paper record exists in epoch 1 (a snapshot is enough).
    assert (await daily_core.take_snapshot())["snapshotted"] is True
    await daily_core._exec_buy("AAA", 100.0, 250.0, "paper buy")

    before = {
        model: _dump(await _all(run_db, model))
        for model in (DailyCoreTrade, DailyCoreSnapshot, DailyCoreAllowance,
                      DailyCorePosition, RunConfig, RunEpoch, RunExecution)
    }
    refused = await daily_core.backfill(start="2025-01-02", preload=preload)
    assert refused["ok"] is False
    assert refused["error"] == (
        "a forward paper record exists for this epoch — backfill would "
        "overwrite the live book's origin; reset first (reset preserves "
        "history as a closed epoch)")
    for model, snapshot in before.items():
        assert _dump(await _all(run_db, model)) == snapshot, model.__name__


async def test_backfill_refused_for_legacy_unstamped_rows(run_db, monkeypatch, tmp_path):
    """FAIL CLOSED: a legacy NULL-mode row cannot be proven synthetic, so a
    backfill must not overwrite it — reset once first."""
    preload = _install_dc_backfill(monkeypatch, tmp_path)
    async with run_db() as s:
        s.add(DailyCoreTrade(ticker="AAA", side="BUY", shares=1.0, price=10.0,
                             cash_after=990.0, reason="pre-stamping", mode=None,
                             config_id=None, epoch=1))
        await s.commit()

    refused = await daily_core.backfill(start="2025-01-02", preload=preload)
    assert refused["ok"] is False
    assert "reset first" in refused["error"]
    assert await _count(run_db, DailyCoreTrade) == 1
    assert (await _all(run_db, RunExecution)) == []  # a refusal is not executed


async def test_mid_run_config_change_refuses_before_any_write(run_db, monkeypatch):
    """A configuration change during the pass's awaits must not stamp a stale
    decision: the executor refuses BEFORE the first history write."""
    _patch_dc_decision(monkeypatch)
    await daily_core.deposit_allowance()

    async def mutating_frame(_me):
        daily_core.set_variant("residual")  # the operator edits the UI mid-pass
        return DC_FRAME, ["AAA", "BBB", "CCC", "DDD"]

    monkeypatch.setattr(daily_core, "_core_frame", mutating_frame)
    out = await daily_core.run_deployment(now=DC_NOW)
    assert out.get("skipped") is True
    assert "configuration changed mid-run" in out["reason"]
    assert await _all(run_db, DailyCoreTrade) == []
    assert await _all(run_db, DailyCoreRanking) == []


async def test_monthly_pending_rebuild_ignores_a_closed_epochs_audit_row(run_db, monkeypatch):
    """The same-month reset must not let the closed epoch's MonthlyRebalance
    row mark the fresh epoch's rebuild as done."""
    clock = datetime(2026, 10, 1, 22, 0, tzinfo=UTC)
    monkeypatch.setattr(monthly, "_prev_trading_day", lambda _now: datetime(2026, 9, 30))
    monkeypatch.setattr(monthly, "_candidate_month_end", lambda _s: datetime(2026, 9, 30))
    monkeypatch.setattr(monthly, "_candle_stored_on", AsyncMock(return_value=True))

    async with run_db() as s:
        s.add(MonthlyAccount(id=1, cash=0.0))
        s.add(MonthlyRebalance(rebal_month="2026-09",
                               rebal_date=datetime(2026, 9, 30), held_before="",
                               picked="", n_new=0, snapshot=""))
        await s.commit()
    # Epoch 1: the row + an empty book prove the rebuild was acted on.
    assert await monthly._pending_rebuild(clock) is None

    out = await monthly.reset_monthly()
    assert out["epoch"] == 2
    pending = await monthly._pending_rebuild(clock)
    assert pending is not None, "the closed epoch's row must not mark the new one"
    assert pending[0] == "2026-09"


async def test_monthly_config_change_refuses_the_same_month_row(run_db, monkeypatch):
    """A same-month pass under a DIFFERENT configuration must not rewrite the
    month's audit content under the new stamp (silent relabelling): it is
    refused before any decision/ledger write, and a reset re-decides it."""
    clock = await _install_monthly_live(run_db, monkeypatch)
    first = await monthly.run_rebalance(now=clock)
    assert first.get("rebalanced") is True, first
    row_before = _dump(await _all(run_db, MonthlyRebalance))
    trade_count = await _count(run_db, MonthlyTrade)
    allow_count = await _count(run_db, MonthlyAllowance)
    exec_count = await _count(run_db, RunExecution)

    monkeypatch.setattr(settings, "sim_monthly_target_n", 3)  # config change
    refused = await monthly.run_rebalance(force=True, now=clock)
    assert refused.get("skipped") is True
    assert "reset (new epoch)" in refused["reason"]
    assert _dump(await _all(run_db, MonthlyRebalance)) == row_before
    assert await _count(run_db, MonthlyTrade) == trade_count
    assert await _count(run_db, MonthlyAllowance) == allow_count
    assert await _count(run_db, RunExecution) == exec_count  # not an execution

    # A reset opens a new epoch: the month is re-decided under the new config.
    out = await monthly.reset_monthly()
    assert out["epoch"] == 2
    second = await monthly.run_rebalance(force=True, now=clock)
    assert second.get("rebalanced") is True, second
    rows = await _all(run_db, MonthlyRebalance)
    assert len(rows) == 2
    assert sorted(r.epoch for r in rows) == [1, 2]
    assert rows[-1].mode == "paper"


# ---------------------------------------------------------------------------
# 4b. THE STAMP/RESET RACE: a stale stamp must refuse without mutation
# ---------------------------------------------------------------------------


async def test_stale_stamp_deposit_and_buy_refuse_after_reset(run_db, monkeypatch):
    """Capture a stamp in epoch 1, reset to epoch 2, then use the OLD stamp:
    nothing may be credited/bought, and no allowance row may land in either
    epoch's book with the stale stamp."""
    _patch_dc_decision(monkeypatch)
    monkeypatch.setattr(daily_core, "_current_month", lambda: "2026-10")
    stale = await daily_core._paper_stamp()          # epoch 1 stamp, no writes
    assert stale["epoch"] == 1
    out = await daily_core.reset_daily_core()
    assert out["epoch"] == 2
    cash_before = (await daily_core.valuate(now=DC_NOW))["cash"]

    refused = await daily_core.deposit_allowance(**stale)
    assert refused["deposited"] is False
    assert refused["refused"] == "stale epoch stamp"
    assert (await daily_core.valuate(now=DC_NOW))["cash"] == cash_before

    trade = await daily_core._exec_buy("AAA", 100.0, 250.0, "stale", **stale)
    assert trade is None
    assert await _count(run_db, DailyCoreTrade) == 0
    assert await _count(run_db, DailyCoreAllowance) == 0
    assert (await daily_core.valuate(now=DC_NOW))["cash"] == cash_before


async def test_stale_stamp_snapshot_and_ranking_are_refused(run_db, monkeypatch):
    """A snapshot/ranking write with a stale stamp must not leak a point or a
    cache row into the new epoch's book."""
    _patch_dc_decision(monkeypatch)
    stale = await daily_core._paper_stamp()
    out = await daily_core.reset_daily_core()
    assert out["epoch"] == 2

    snap = await daily_core.take_snapshot(mode=stale["mode"],
                                         config_id=stale["config_id"],
                                         epoch=stale["epoch"])
    assert snap["snapshotted"] is False
    assert await _count(run_db, DailyCoreSnapshot) == 0

    await daily_core._store_ranking(["AAA"], ["AAA"], DC_ME.to_pydatetime(),
                                   mode=stale["mode"],
                                   config_id=stale["config_id"],
                                   epoch=stale["epoch"])
    assert await _count(run_db, DailyCoreRanking) == 0


async def test_monthly_and_sim_stale_stamps_refuse(run_db, monkeypatch):
    """Same fence on the other engines' standalone writers."""
    # monthly
    m_stale = await monthly._paper_stamp()
    assert (await monthly.reset_monthly())["epoch"] == 2
    m_out = await monthly.deposit_allowance(mode=m_stale["mode"],
                                            config_id=m_stale["config_id"],
                                            epoch=m_stale["epoch"])
    assert m_out["deposited"] is False and m_out["refused"] == "stale epoch stamp"
    assert await monthly._exec_buy("AAA", 10.0, 100.0, "stale",
                                  mode=m_stale["mode"],
                                  config_id=m_stale["config_id"],
                                  epoch=m_stale["epoch"]) is None
    assert await _count(run_db, MonthlyAllowance) == 0
    assert await _count(run_db, MonthlyTrade) == 0

    # sim
    monkeypatch.setattr(sim, "_current_month", lambda: "2026-10")
    s_stale = await sim._paper_stamp()
    assert (await sim.reset_sim())["epoch"] == 2
    s_out = await sim.deposit_allowance(mode=s_stale["mode"],
                                        config_id=s_stale["config_id"],
                                        epoch=s_stale["epoch"])
    assert s_out["deposited"] is False and s_out["refused"] == "stale epoch stamp"
    assert await sim._exec_buy("AAPL", 10.0, 100.0, "stale",
                               mode=s_stale["mode"],
                               config_id=s_stale["config_id"],
                               epoch=s_stale["epoch"]) is None
    await sim._write_snapshot({"cash": 0.0, "positions_value": 0.0,
                               "total_equity": 0.0, "allowance_total": 0.0},
                              mode=s_stale["mode"],
                              config_id=s_stale["config_id"],
                              epoch=s_stale["epoch"])
    assert await _count(run_db, SimAllowance) == 0
    assert await _count(run_db, SimTrade) == 0
    assert await _count(run_db, SimSnapshot) == 0


# ---------------------------------------------------------------------------
# 5. Reset preserves history bit-for-bit, records a boundary, re-seeds state
# ---------------------------------------------------------------------------


async def test_reset_preserves_history_and_records_a_boundary(run_db, monkeypatch):
    _patch_dc_decision(monkeypatch)
    await daily_core.deposit_allowance()
    await daily_core.run_deployment(now=DC_NOW)
    await daily_core.take_snapshot()

    history_models = (DailyCoreTrade, DailyCoreSnapshot, DailyCoreAllowance)
    before = {model: _dump(await _all(run_db, model)) for model in history_models}

    out = await daily_core.reset_daily_core()
    assert out["ok"] is True
    assert out["epoch"] == 2
    assert out["preserved"]["trades"] == len(before[DailyCoreTrade])
    assert out["preserved"]["snapshots"] == len(before[DailyCoreSnapshot])
    assert out["preserved"]["allowances"] == len(before[DailyCoreAllowance])

    for model in history_models:
        assert _dump(await _all(run_db, model)) == before[model], (
            f"{model.__name__} must be preserved bit-for-bit")

    boundaries = await _all(run_db, RunEpoch)
    assert [(b.engine, b.epoch) for b in boundaries] == [("daily_core", 2)]
    assert "preserved as epoch 1" in boundaries[0].note

    async with run_db() as s:
        acc = await s.get(DailyCoreAccount, 1)
        assert acc.cash == settings.sim_monthly_start_cash
        assert acc.last_allowance_month is None
    assert await _all(run_db, DailyCorePosition) == []
    assert await _all(run_db, DailyCoreRanking) == []  # current cache, invalidated


async def test_monthly_and_sim_resets_preserve_history(run_db, monkeypatch):
    # --- monthly: one of every history row, then reset ---
    async with run_db() as s:
        s.add(MonthlyAccount(id=1, cash=123.45, last_allowance_month="2026-09",
                             last_rebalance_month="2026-09"))
        s.add(MonthlyPosition(ticker="AAA", shares=1.0, avg_cost=10.0))
        s.add(MonthlyTrade(ticker="AAA", side="BUY", shares=1.0, price=10.0,
                           cash_after=0.0, reason="seed"))
        s.add(MonthlyAllowance(amount=1000.0, month="2026-09"))
        s.add(MonthlySnapshot(cash=0.0, positions_value=1.0, total_equity=1.0,
                              allowance_total=1000.0))
        s.add(MonthlyRebalance(rebal_month="2026-09",
                               rebal_date=datetime(2026, 9, 30), held_before="",
                               picked="AAA", n_new=1, snapshot=""))
        await s.commit()
    before = {
        model: _dump(await _all(run_db, model))
        for model in (MonthlyTrade, MonthlySnapshot, MonthlyAllowance, MonthlyRebalance)
    }
    out = await monthly.reset_monthly()
    assert out["epoch"] == 2
    assert out["preserved"] == {"trades": 1, "snapshots": 1, "allowances": 1,
                                "rebalances": 1}
    for model, snapshot in before.items():
        assert _dump(await _all(run_db, model)) == snapshot, model.__name__
    async with run_db() as s:
        acc = await s.get(MonthlyAccount, 1)
        assert acc.cash == settings.sim_monthly_start_cash
        assert acc.last_rebalance_month is None

    # --- sim: one trade + one snapshot, then reset (benchmark is derived) ---
    await sim.deposit_allowance()
    await sim._exec_buy("AAPL", 100.0, 500.0, "seed")
    await sim._write_snapshot({"cash": 500.0, "positions_value": 500.0,
                              "total_equity": 1000.0, "allowance_total": 1000.0})
    before = {model: _dump(await _all(run_db, model))
              for model in (SimTrade, SimSnapshot, SimAllowance)}
    out = await sim.reset_sim()
    assert out["epoch"] == 2
    for model, snapshot in before.items():
        assert _dump(await _all(run_db, model)) == snapshot, model.__name__
    async with run_db() as s:
        acc = await s.get(SimAccount, 1)
        assert acc.cash == settings.sim_start_cash
    assert await _all(run_db, SimPosition) == []


# ---------------------------------------------------------------------------
# 6. Post-reset same-month deposit works (one row per epoch) + curve clips
# ---------------------------------------------------------------------------


async def test_post_reset_same_month_deposit_and_epoch_scoped_curve(run_db, monkeypatch):
    _patch_dc_decision(monkeypatch)
    monkeypatch.setattr(daily_core, "_current_month", lambda: "2026-10")
    assert (await daily_core.deposit_allowance())["deposited"] is True
    await daily_core.take_snapshot()

    first = await daily_core.reset_daily_core()
    assert first["epoch"] == 2
    # The same month is NEW allocation in the new epoch's ledger, not a move.
    again = await daily_core.deposit_allowance()
    assert again["deposited"] is True
    await daily_core.take_snapshot()

    allowances = await _all(run_db, DailyCoreAllowance)
    assert len(allowances) == 2
    assert [a.month for a in allowances] == ["2026-10", "2026-10"]
    assert [a.epoch for a in allowances] == [1, 2]
    assert [a.config_id is not None for a in allowances] == [True, True]

    curve = await daily_core.get_equity_curve()
    assert len(curve) == 1 and curve[0]["epoch"] == 2
    assert await _count(run_db, DailyCoreSnapshot) == 2  # pre-reset row preserved

    # The current-epoch contributed total restarts (seed + this epoch's deposit).
    val = await daily_core.valuate(now=DC_NOW)
    assert val["allowance_total"] == pytest.approx(
        settings.sim_monthly_start_cash + settings.sim_monthly_contribution)


# ---------------------------------------------------------------------------
# 7. Epoch clip: mixed-epoch DB -> curve and realized-vol returns
# ---------------------------------------------------------------------------


async def test_epoch_clip_on_curve_and_recent_port_returns(run_db):
    from app import run_config

    async with run_db() as s:
        s.add(DailyCoreSnapshot(cash=0.0, positions_value=1000.0,
                                total_equity=1000.0, allowance_total=1000.0,
                                mode="paper", epoch=1,
                                created_at=datetime(2026, 9, 1, 16, tzinfo=UTC)))
        s.add(DailyCoreSnapshot(cash=0.0, positions_value=1000.0,
                                total_equity=1000.0, allowance_total=1000.0,
                                mode="paper", epoch=1,
                                created_at=datetime(2026, 9, 2, 16, tzinfo=UTC)))
        await s.commit()
    async with run_db() as s:
        new_epoch = await run_config.record_boundary("daily_core", "test boundary", session=s)
        assert new_epoch == 2
        for day, equity in ((3, 1000.0), (4, 1100.0), (5, 1210.0)):
            s.add(DailyCoreSnapshot(cash=0.0, positions_value=equity,
                                    total_equity=equity, allowance_total=1000.0,
                                    mode="paper", epoch=2,
                                    created_at=datetime(2026, 9, day, 16, tzinfo=UTC)))
        await s.commit()

    curve = await daily_core.get_equity_curve()
    assert [row["total_equity"] for row in curve] == [1000.0, 1100.0, 1210.0]
    assert {row["epoch"] for row in curve} == {2}

    # Realized-vol returns must not bridge the closed epoch into the new book.
    rets = await daily_core._recent_port_returns(21)
    assert len(rets) == 2  # only the three epoch-2 points: no epoch-1 bridge
    assert rets[0] == pytest.approx(0.09531017980432493, rel=1e-9)  # 1000 -> 1100
    assert rets[1] == pytest.approx(0.09531017980432493, rel=1e-9)  # 1100 -> 1210


# ---------------------------------------------------------------------------
# 8. Legacy pre-stamping rows (epoch 1/None-equivalent) read as epoch 1
# ---------------------------------------------------------------------------


async def test_legacy_rows_read_as_epoch_one_then_are_excluded_after_reset(run_db):
    async with run_db() as s:
        s.add(DailyCoreSnapshot(cash=0.0, positions_value=1000.0,
                                total_equity=1000.0, allowance_total=1000.0,
                                mode=None, config_id=None, epoch=1,
                                created_at=datetime(2026, 9, 1, 16, tzinfo=UTC)))
        s.add(DailyCoreTrade(ticker="AAA", side="BUY", shares=1.0, price=10.0,
                             cash_after=990.0, reason="legacy", mode=None,
                             config_id=None, epoch=1))
        await s.commit()

    assert (await _all(run_db, RunEpoch)) == []
    curve = await daily_core.get_equity_curve()
    assert len(curve) == 1
    assert (curve[0]["mode"], curve[0]["config_id"], curve[0]["epoch"]) == (None, None, 1)

    out = await daily_core.reset_daily_core()
    assert out["preserved"]["snapshots"] == 1 and out["preserved"]["trades"] == 1
    assert await daily_core.get_equity_curve() == []
    assert await _count(run_db, DailyCoreSnapshot) == 1  # still present
    assert await _count(run_db, DailyCoreTrade) == 1


# ---------------------------------------------------------------------------
# 9. Sim chat [[ACTION]] path is a forward paper record too
# ---------------------------------------------------------------------------


async def test_sim_chat_action_stamps_trade_and_snapshot_paper(run_db, monkeypatch):
    monkeypatch.setattr(sim, "_utcnow", lambda: datetime(2026, 10, 5, 12, 0, tzinfo=UTC))
    monkeypatch.setattr(sim, "_current_month", lambda: "2026-10")
    async with run_db() as s:
        s.add(SimAccount(id=1, cash=1000.0, last_allowance_month=None))
        await s.commit()

    async def fake_context():
        return "mock context"

    async def fake_reply(_messages):
        return {"text": 'Buying.\n[[ACTION]]\n'
                        '{"actions": [{"ticker": "AAPL", "action": "BUY", '
                        '"amount": 200.0, "reason": "test"}]}\n[[/ACTION]]'}

    async def fake_close(_ticker):
        return 100.0

    monkeypatch.setattr(sim, "_build_sim_chat_context", fake_context)
    monkeypatch.setattr(sim.llm_mod, "chat", fake_reply)
    monkeypatch.setattr(sim, "_latest_close", fake_close)

    out = await sim.sim_chat([{"role": "user", "content": "buy AAPL"}])
    assert out["actions_executed"] is True
    assert [(t["ticker"], t["side"]) for t in out["trades"]] == [("AAPL", "BUY")]

    _assert_stamped(await _all(run_db, SimTrade), mode="paper", epoch=1)
    _assert_stamped(await _all(run_db, SimSnapshot), mode="paper", epoch=1)
    executions = await _all(run_db, RunExecution)
    assert any(e.mode == "paper" and e.engine == "sim" for e in executions)


# ---------------------------------------------------------------------------
# 10. Seed capital is external contribution in the current-epoch basis
# ---------------------------------------------------------------------------


async def test_seed_capital_is_frozen_per_epoch(run_db, monkeypatch):
    """Changing the seed setting WITHOUT a reset must not rewrite the
    contributed basis of an account that already holds the old capital."""
    _patch_dc_decision(monkeypatch)
    monkeypatch.setattr(settings, "sim_monthly_start_cash", 100.0)
    monkeypatch.setattr(daily_core, "_current_month", lambda: "2026-10")

    await daily_core.deposit_allowance()  # opens the account with $100 + deposit
    assert (await daily_core.valuate(now=DC_NOW))["allowance_total"] == pytest.approx(1100.0)

    monkeypatch.setattr(settings, "sim_monthly_start_cash", 999.0)  # settings only
    assert (await daily_core.valuate(now=DC_NOW))["allowance_total"] == pytest.approx(1100.0)

    out = await daily_core.reset_daily_core()  # re-seeds with the CURRENT setting
    assert out["epoch"] == 2
    assert (await daily_core.valuate(now=DC_NOW))["allowance_total"] == pytest.approx(999.0)

    # Even a settings change BEFORE the new epoch's first run cannot rewrite
    # the basis: the seed is frozen on the ACCOUNT, not on an execution.
    monkeypatch.setattr(settings, "sim_monthly_start_cash", 5000.0)
    assert (await daily_core.valuate(now=DC_NOW))["allowance_total"] == pytest.approx(999.0)


async def test_factor_backfill_pins_the_seed_to_zero(run_db, monkeypatch, tmp_path):
    """A backfilled factor book has NO seed capital: the replay baseline is
    cash = 0 + its own deposits, and a later live pass must not retroactively
    claim the configured seed for the synthetic origin."""
    preload = _install_dc_backfill(monkeypatch, tmp_path)
    monkeypatch.setattr(daily_core, "_current_month", lambda: "2025-02")
    monkeypatch.setattr(settings, "sim_monthly_start_cash", 1000.0)

    out = await daily_core.backfill(start="2025-01-02", preload=preload)
    assert out["ok"] is True, out
    async with run_db() as s:
        acc = await s.get(DailyCoreAccount, 1)
        assert acc.opening_seed == 0.0
    assert (await daily_core.valuate(now=DC_NOW))["allowance_total"] == pytest.approx(
        out["contributed"])

    # A live paper pass in the same epoch must KEEP the frozen 0 seed.
    _patch_dc_decision(monkeypatch)
    await daily_core.take_snapshot()
    val = await daily_core.valuate(now=DC_NOW)
    assert val["allowance_total"] == pytest.approx(out["contributed"])


async def test_seed_capital_counts_in_the_epoch_contribution_basis(run_db, monkeypatch):
    _patch_dc_decision(monkeypatch)
    monkeypatch.setattr(settings, "sim_monthly_start_cash", 1000.0)
    monkeypatch.setattr(daily_core, "_current_month", lambda: "2026-10")

    assert (await daily_core.deposit_allowance())["deposited"] is True
    val = await daily_core.valuate(now=DC_NOW)
    assert val["allowance_total"] == pytest.approx(2000.0)  # seed + one deposit

    await daily_core.take_snapshot()
    snap = (await _all(run_db, DailyCoreSnapshot))[-1]
    assert snap.allowance_total == pytest.approx(2000.0)

    out = await daily_core.reset_daily_core()
    assert out["epoch"] == 2
    # Fresh epoch: the seed is the only contribution until it deposits again.
    assert (await daily_core.valuate(now=DC_NOW))["allowance_total"] == pytest.approx(1000.0)
    assert (await daily_core.deposit_allowance())["deposited"] is True
    assert (await daily_core.valuate(now=DC_NOW))["allowance_total"] == pytest.approx(2000.0)
