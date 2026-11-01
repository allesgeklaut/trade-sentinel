"""Fail-closed isfinite guards in the sim book (app/sim.py) + the Settings
hermeticity pin for tests/conftest.py.

SQLite stores inf as REAL, so a non-finite candle close is storable end to
end and used to flow into the book: ``_latest_close`` returned it raw,
``_decision_inputs`` put it into the decision prices, and ``_exec_buy``'s
``price <= 0`` guard let NaN through (NaN fails every comparison) so
floor()ed NaN shares poisoned cash inf→NaN. Now every close→book/snapshot
propagation treats a non-finite close as NO DATA (fail closed, the factor
books' isfinite convention); decision math for finite values is unchanged.

Also pins the tests/conftest.py autouse fixture: decision-input Settings
fields are at their CODE defaults under pytest even though the ambient
gitignored .env sets SIM_DAILY_CORE_MOM_VARIANT=residual — and a test's own
monkeypatch override still wins and is exercised.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime

import pandas as pd
import pytest
from sqlalchemy import StaticPool, func, select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app import daily_core, monthly, sim
from app.config import Settings, settings
from app.db import Base, Candle, SimAccount, SimPosition, SimSnapshot, SimTrade

INF = float("inf")
NAN = float("nan")


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


@pytest.fixture
async def mem_db(monkeypatch):
    """In-memory DB for the sim (and the session loader, which imports
    ``app.db.Session`` at call time — patching the module attr covers it)."""
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    import app.db as db_mod
    import app.market as market_mod

    monkeypatch.setattr(sim, "Session", session_factory)
    monkeypatch.setattr(db_mod, "Session", session_factory)
    # market.latest_close uses the market module's own Session import.
    monkeypatch.setattr(market_mod, "Session", session_factory)

    async with session_factory() as s:
        s.add(SimAccount(id=1, cash=10_000.0, last_allowance_month=None))
        await s.commit()
    yield session_factory
    await engine.dispose()


async def _seed_candle(session_factory, ticker: str, close: float,
                       when: datetime) -> None:
    async with session_factory() as s:
        s.add(Candle(ticker=ticker, timestamp=when,
                     open=close, high=close, low=close, close=close, volume=1e6))
        await s.commit()


async def _seed_candle_series(session_factory, ticker: str, n: int = 260,
                              start: str = "2025-01-06", last_close: float = 100.0,
                              ) -> list[pd.Timestamp]:
    """A flat-ish tape of n business-day bars; the LAST bar gets
    ``last_close`` (pass inf to poison the S close)."""
    dates = list(pd.bdate_range(start, periods=n))
    async with session_factory() as s:
        for i, d in enumerate(dates):
            px = last_close if i == n - 1 else 100.0 + (i % 7) * 0.5
            s.add(Candle(ticker=ticker, timestamp=d.to_pydatetime(),
                         open=px, high=px, low=px, close=px, volume=1e6))
        await s.commit()
    return dates


async def _trade_count(mem_db) -> int:
    async with mem_db() as s:
        return (await s.scalar(select(func.count()).select_from(SimTrade))) or 0


async def _snapshot_count(mem_db) -> int:
    async with mem_db() as s:
        return (await s.scalar(select(func.count()).select_from(SimSnapshot))) or 0


async def _cash(mem_db) -> float:
    async with mem_db() as s:
        return float((await s.get(SimAccount, 1)).cash)


# ---------------------------------------------------------------------------
# latest_close / valuate: non-finite close = no data (fail closed)
# ---------------------------------------------------------------------------


class TestLatestCloseGuard:
    async def test_nonfinite_stored_close_reads_as_no_data(self, mem_db):
        await _seed_candle(mem_db, "INF1", INF, datetime.now(UTC))
        await _seed_candle(mem_db, "OK1", 42.5, datetime.now(UTC))
        assert await sim._latest_close("INF1") is None
        assert await sim._latest_close("OK1") == 42.5
        assert await sim._latest_close("MISSING") is None

    async def test_valuate_falls_back_to_avg_cost_and_stays_finite(self, mem_db):
        await _seed_candle(mem_db, "AAA", INF, datetime.now(UTC))
        async with mem_db() as s:
            s.add(SimPosition(ticker="AAA", shares=2.0, avg_cost=100.0,
                              thesis="t", stop_price=85.0))
            await s.commit()
        v = await sim.valuate()
        pos = v["positions"][0]
        # inf close = no data -> existing no-data fallback (avg cost), never inf
        assert pos["current_price"] == 100.0
        assert pos["value"] == 200.0
        for key in ("cash", "positions_value", "total_equity", "allowance_total"):
            assert math.isfinite(v[key]), key


# ---------------------------------------------------------------------------
# The book: _exec_buy / _exec_sell refuse non-finite fills
# ---------------------------------------------------------------------------


class TestExecGuards:
    @pytest.mark.parametrize("bad_price", [INF, -INF, NAN])
    async def test_buy_refuses_nonfinite_price_and_budget(self, mem_db, bad_price):
        t = await sim._exec_buy("AAA", bad_price, 100.0, "test")
        assert t is None
        assert await _trade_count(mem_db) == 0
        assert await _cash(mem_db) == 10_000.0  # unchanged, not NaN

    async def test_buy_refuses_nonfinite_budget(self, mem_db):
        assert await sim._exec_buy("AAA", 100.0, NAN, "test") is None
        assert await sim._exec_buy("AAA", 100.0, INF, "test") is None
        assert await _trade_count(mem_db) == 0
        assert await _cash(mem_db) == 10_000.0

    async def test_buy_finite_happy_path(self, mem_db):
        t = await sim._exec_buy("AAA", 100.0, 500.0, "test")
        assert t is not None and t["side"] == "BUY"
        assert await _cash(mem_db) == 9_500.0

    @pytest.mark.parametrize("bad_price", [INF, NAN])
    async def test_sell_refuses_nonfinite_price(self, mem_db, bad_price):
        async with mem_db() as s:
            s.add(SimPosition(ticker="AAA", shares=2.0, avg_cost=100.0,
                              thesis="t", stop_price=85.0))
            await s.commit()
        t = await sim._exec_sell("AAA", bad_price, None, "test")
        assert t is None
        assert await _trade_count(mem_db) == 0
        async with mem_db() as s:
            pos = await s.scalar(select(SimPosition).where(SimPosition.ticker == "AAA"))
            assert pos is not None and pos.shares == 2.0  # position untouched
        assert await _cash(mem_db) == 10_000.0

    async def test_sell_finite_happy_path(self, mem_db):
        async with mem_db() as s:
            s.add(SimPosition(ticker="AAA", shares=2.0, avg_cost=100.0,
                              thesis="t", stop_price=85.0))
            await s.commit()
        t = await sim._exec_sell("AAA", 110.0, None, "test")
        assert t is not None and t["side"] == "SELL"
        assert await _cash(mem_db) == 10_220.0


# ---------------------------------------------------------------------------
# Decision inputs + snapshot: a non-finite close never lands anywhere
# ---------------------------------------------------------------------------


class TestDecisionAndSnapshotGuards:
    async def test_decision_inputs_skip_a_ticker_whose_s_close_is_nonfinite(self, mem_db):
        dates = await _seed_candle_series(mem_db, "AAA", last_close=INF)
        now = (dates[-1] + pd.Timedelta(days=3)).to_pydatetime().replace(tzinfo=UTC)
        dec = await sim._decision_inputs(["AAA"], now=now)
        # fail closed: no signal, no decision price — like a ticker that did
        # not trade on S — never an inf price that could reach a fill
        assert "AAA" not in dec.prices
        assert "AAA" not in dec.signals

    async def test_decision_inputs_finite_close_still_priced(self, mem_db):
        dates = await _seed_candle_series(mem_db, "AAA", last_close=123.0)
        now = (dates[-1] + pd.Timedelta(days=3)).to_pydatetime().replace(tzinfo=UTC)
        dec = await sim._decision_inputs(["AAA"], now=now)
        assert dec.prices.get("AAA") == 123.0
        assert "AAA" in dec.signals

    @pytest.mark.parametrize("poison", ["cash", "total_equity"])
    async def test_write_snapshot_refuses_nonfinite_valuation(self, mem_db, poison):
        post = {"cash": 100.0, "positions_value": 50.0,
                "total_equity": 150.0, "allowance_total": 1_000.0}
        post[poison] = INF
        await sim._write_snapshot(post)
        assert await _snapshot_count(mem_db) == 0  # nothing landed

    async def test_write_snapshot_finite_happy_path(self, mem_db):
        post = {"cash": 100.0, "positions_value": 50.0,
                "total_equity": 150.0, "allowance_total": 1_000.0}
        await sim._write_snapshot(post)
        assert await _snapshot_count(mem_db) == 1


# ---------------------------------------------------------------------------
# tests/conftest.py hermeticity pin
# ---------------------------------------------------------------------------


class TestSettingsHermeticity:
    def test_autouse_fixture_resets_variant_to_code_default(self):
        """The ambient gitignored .env on this host sets
        SIM_DAILY_CORE_MOM_VARIANT=residual; under pytest the autouse
        fixture in tests/conftest.py must have reset the field to the CODE
        default (read from pydantic model_fields)."""
        default = Settings.model_fields["sim_daily_core_mom_variant"].default
        assert default == "raw"
        assert settings.sim_daily_core_mom_variant == default

    def test_per_test_override_wins_and_is_exercised(self, monkeypatch):
        """The fixture must not fight per-test overrides: a test that sets
        residual deliberately gets residual — and it flows through the real
        read path (no /data state file exists under pytest, so
        daily_core.current_variant() falls back to the settings value)."""
        monkeypatch.setattr(settings, "sim_daily_core_mom_variant", "residual")
        assert settings.sim_daily_core_mom_variant == "residual"   # override applied
        assert daily_core.current_variant() == "residual"           # read path sees it
        assert monthly._strategy_variant() == "residual"            # engines resolve it
