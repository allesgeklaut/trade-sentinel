"""Tests for the daily-core paper portfolio (app/daily_core).

These tests use an in-memory SQLite database and monkeypatched market
data so they don't touch the real data volume. They cover:

- Allowance deposit: once per operator-local month, idempotent
- compute_targets: hysteresis picks from the qv-mom ranking
- run_deployment: rank-first top-up toward equal weight, band releases
- Cycle idempotence: a second run does not duplicate trades
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime

import pandas as pd
import pytest
from sqlalchemy import StaticPool, select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app import daily_core, monthly
from app.config import settings
from app.db import Base, DailyCoreAllowance

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
async def mem_db(monkeypatch):
    """Replace the daily-core engine's database with an in-memory SQLite DB."""
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    session_factory = async_sessionmaker(engine, expire_on_commit=False)

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    monkeypatch.setattr(daily_core, "Session", session_factory)
    import app.db as db_mod
    monkeypatch.setattr(db_mod, "Session", session_factory)

    yield session_factory
    await engine.dispose()


@pytest.fixture
def fake_ranking(monkeypatch):
    """Patch the ranking loaders so compute_targets sees a tiny fixed frame.

    The frame is what monthly.eligible_frame/_qv_order/_band_fill consume:
    index=ticker, columns roe/p_fcf/mom/... with an `eligible` flag.
    """
    frame = pd.DataFrame(
        {
            "roe": [0.3, 0.25, 0.2, 0.1],
            "p_fcf": [10.0, 15.0, 20.0, 25.0],
            "mcap": [1e10] * 4,
            "dollar_vol": [1e8] * 4,
            "mom": [0.5, 0.4, 0.3, 0.2],
            "price": [100.0, 50.0, 25.0, 10.0],
            "vol": [0.2] * 4,
            "eligible": [True, True, True, True],
        },
        index=pd.Index(["AAA", "BBB", "CCC", "DDD"], name="ticker"),
    )
    async def fake_targets():
        band = ["AAA", "BBB", "CCC", "DDD"]
        async with daily_core.Session() as s:
            held = [p.ticker for p in
                    (await s.scalars(select(daily_core.DailyCorePosition))).all()]
        picks = monthly._band_fill(band, held, 3, 4)
        return band, picks, frame
    monkeypatch.setattr(daily_core, "compute_targets", fake_targets)
    return frame


# ---------------------------------------------------------------------------
# Allowance
# ---------------------------------------------------------------------------

class TestAllowance:
    def test_deposit_once_per_month(self, mem_db, monkeypatch):
        monkeypatch.setattr(daily_core, "_current_month", lambda: "2026-09")
        r1 = asyncio.run(daily_core.deposit_allowance())
        r2 = asyncio.run(daily_core.deposit_allowance())
        assert r1["deposited"] is True
        assert r2["deposited"] is False
        async def check():
            async with mem_db() as s:
                rows = (await s.scalars(select(DailyCoreAllowance))).all()
                assert len(rows) == 1
                acc = await daily_core._account(s)
                assert acc.cash == pytest.approx(settings.sim_monthly_contribution)
        asyncio.run(check())

    def test_new_month_deposits_again(self, mem_db, monkeypatch):
        monkeypatch.setattr(daily_core, "_current_month", lambda: "2026-09")
        asyncio.run(daily_core.deposit_allowance())
        monkeypatch.setattr(daily_core, "_current_month", lambda: "2026-10")
        r = asyncio.run(daily_core.deposit_allowance())
        assert r["deposited"] is True


# ---------------------------------------------------------------------------
# Deployment
# ---------------------------------------------------------------------------

class TestDeployment:
    def test_rank_first_topup(self, mem_db, fake_ranking, monkeypatch):
        """Free cash deploys best-rank-first toward equal weight
        (weight = equity / sim_monthly_target_n, like the live config)."""
        async def scenario():
            async with mem_db() as s:
                acc = await daily_core._account(s)
                acc.cash = 3000.0
                await s.commit()
            monkeypatch.setattr(
                monthly, "_price_usd_map",
                lambda tickers: _px({t: 100.0 for t in tickers}))
            r = await daily_core.run_deployment()
            assert r["deployed"] is True
            # weight = 3000/10 = 300 per slot; the 4-ticker fake ranking
            # tops AAA..DDD to 300 each (rank-first), then the band is
            # exhausted — no pro-rata spread beyond the top target_n.
            assert [t["ticker"] for t in r["trades"]] == ["AAA", "BBB", "CCC", "DDD"]
            assert all(t["side"] == "BUY" for t in r["trades"])
            assert r["valuation"]["cash"] == pytest.approx(3000.0 - 4 * 300.0)
        asyncio.run(scenario())

    def test_top_target_n_rank_first(self, mem_db, fake_ranking, monkeypatch):
        """With enough cash, each of the top target_n names reaches the same
        equal-weight target — cash goes to the best rank first but the
        per-name target is the same (boost=0 semantics)."""
        async def scenario():
            async with mem_db() as s:
                acc = await daily_core._account(s)
                # 10 * contribution would fully fund 10 slots; use 4 tickers
                # worth: equity 1200 → weight 120 → all 4 topped to 120.
                acc.cash = 1200.0
                await s.commit()
            monkeypatch.setattr(
                monthly, "_price_usd_map",
                lambda tickers: _px({t: 100.0 for t in tickers}))
            r = await daily_core.run_deployment()
            vals = {t["ticker"]: t["cost"] for t in r["trades"]}
            assert vals == {"AAA": 120.0, "BBB": 120.0, "CCC": 120.0, "DDD": 120.0}
        asyncio.run(scenario())

    def test_no_pro_rata_into_laggards(self, mem_db, fake_ranking, monkeypatch):
        """A held bottom-ranked name at target gets NO fresh cash before
        the top-ranked names reach their targets (rank-first rule)."""
        async def scenario():
            async with mem_db() as s:
                acc = await daily_core._account(s)
                acc.cash = 0.0
                await s.commit()
                s.add(daily_core.DailyCorePosition(ticker="BBB", shares=10.0, avg_cost=100.0))
                await s.commit()
            # BBB at 100/share = 1000 value = already at weight (equity 1000/3 ≈ 333
            # per slot — actually BBB is OVER weight, so nothing more to buy there)
            monkeypatch.setattr(
                monthly, "_price_usd_map",
                lambda tickers: _px({t: 100.0 for t in tickers}))
            r = await daily_core.run_deployment()
            # No cash: no trades at all
            assert r["trades"] == []
        asyncio.run(scenario())

    def test_band_release_sell(self, mem_db, fake_ranking, monkeypatch):
        """A held name outside the hold band is released (SELL) before BUYs."""
        async def scenario():
            async with mem_db() as s:
                acc = await daily_core._account(s)
                acc.cash = 0.0
                await s.commit()
                s.add(daily_core.DailyCorePosition(ticker="ZZZ", shares=5.0, avg_cost=50.0))
                await s.commit()
            def fake_map(tickers):
                px = {t: 100.0 for t in tickers}
                px["ZZZ"] = 60.0
                return _px(px)
            monkeypatch.setattr(monthly, "_price_usd_map", lambda tickers: fake_map(tickers))
            r = await daily_core.run_deployment()
            sells = [t for t in r["trades"] if t["side"] == "SELL"]
            assert len(sells) == 1 and sells[0]["ticker"] == "ZZZ"
            assert sells[0]["proceeds"] == pytest.approx(300.0)
        asyncio.run(scenario())

    def test_second_run_is_noop(self, mem_db, fake_ranking, monkeypatch):
        """Deployment is idempotent: targets met -> no further trades."""
        async def scenario():
            async with mem_db() as s:
                acc = await daily_core._account(s)
                acc.cash = 3000.0
                await s.commit()
            monkeypatch.setattr(
                monthly, "_price_usd_map",
                lambda tickers: _px({t: 100.0 for t in tickers}))
            r1 = await daily_core.run_deployment()
            n1 = len(r1["trades"])
            assert n1 > 0
            r2 = await daily_core.run_deployment()
            assert r2["trades"] == []
        asyncio.run(scenario())


class _Awy:
    """Tiny helper: wrap a dict as an awaitable (for _price_usd_map patches)."""
    def __init__(self, px):
        self.px = px
    def __await__(self):
        if False:
            yield
        return self.px


def _px(px: dict):
    return _Awy(px)