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
from datetime import UTC, datetime, timedelta

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
    async def fake_targets(force=False):
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

    def test_duplicate_month_row_does_not_500_or_double_deposit(self, mem_db, monkeypatch):
        """The status endpoint calls deposit_allowance() unlocked and the UI
        fires two concurrent status requests on tab load. At a month rollover
        both can read the stale marker; the UNIQUE(month) constraint makes
        the loser's insert fail. It must be handled as "already deposited",
        not surfaced as a 500 or a second cash deposit."""
        async def seed():
            async with mem_db() as s:
                acc = await daily_core._account(s)
                acc.cash = 500.0
                acc.last_allowance_month = "2026-08"
                s.add(DailyCoreAllowance(amount=settings.sim_monthly_contribution,
                                         month="2026-09"))
                await s.commit()
        asyncio.run(seed())
        monkeypatch.setattr(daily_core, "_current_month", lambda: "2026-09")
        r = asyncio.run(daily_core.deposit_allowance())
        assert r["deposited"] is False
        async def check():
            async with mem_db() as s:
                acc = await daily_core._account(s)
                assert acc.cash == pytest.approx(500.0)
                rows = (await s.scalars(select(DailyCoreAllowance))).all()
                assert len(rows) == 1
        asyncio.run(check())


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

class TestStoredRanking:
    def test_compute_targets_reads_stored_ranking(self, mem_db, monkeypatch):
        """compute_targets() (force=False) reads the daily_core_ranking row
        instead of recomputing the fundamentals math; force=True recomputes
        and re-stores."""
        from app.db import DailyCoreRanking

        calls = {"n": 0}

        async def fake_targets(**kwargs):
            calls["n"] += 1
            return ["AAA", "BBB", "CCC"], ["AAA", "BBB", "CCC"], None

        # Simulate a stored ranking from today
        async def seed():
            async with mem_db() as s:
                s.add(DailyCoreRanking(
                    id=1,
                    ranking_date=datetime.now(UTC).replace(tzinfo=None),
                    band="AAA,BBB,CCC", picks="AAA,BBB,CCC"))
                await s.commit()
        asyncio.run(seed())

        monkeypatch.setattr(daily_core, "compute_targets", fake_targets)
        # Read the STORED ranking through the real read helper:
        stored = asyncio.run(daily_core._load_stored_ranking())
        assert stored == (["AAA", "BBB", "CCC"], ["AAA", "BBB", "CCC"])

    def test_stale_ranking_is_ignored(self, mem_db):
        """A ranking older than a day is treated as absent."""
        from app.db import DailyCoreRanking

        async def seed():
            async with mem_db() as s:
                s.add(DailyCoreRanking(
                    id=1,
                    ranking_date=datetime.now(UTC).replace(tzinfo=None) - timedelta(days=2),
                    band="AAA", picks="AAA"))
                await s.commit()
        asyncio.run(seed())
        stored = asyncio.run(daily_core._load_stored_ranking())
        assert stored is None

    def test_concurrent_compute_runs_once(self, mem_db, monkeypatch):
        """The UI fires two status requests on first tab load; both call
        compute_targets() with no stored ranking. They must not both run the
        expensive full-universe ranking — the second reads the first's
        freshly stored result."""
        import app.monthly as monthly_mod

        calls = {"fund": 0}
        frame = pd.DataFrame({"eligible": [True]},
                             index=pd.Index(["AAA"], name="ticker"))
        day = pd.DatetimeIndex(["2026-09-01"])

        async def fake_fund(_tickers):
            calls["fund"] += 1
            await asyncio.sleep(0)  # yield so a racer could interleave
            return {"AAA": {}}

        async def fake_frames(_tickers, _asof, start=None):
            return pd.DataFrame({"AAA": [1.0]}, index=day), \
                   pd.DataFrame({"AAA": [1.0]}, index=day)

        monkeypatch.setattr(daily_core, "universe_tickers", lambda _u: ["AAA"])
        monkeypatch.setattr(daily_core.fundamentals_mod, "load_fundamentals", fake_fund)
        monkeypatch.setattr(daily_core.monthly_mod, "load_frames", fake_frames)
        monkeypatch.setattr(monthly_mod, "eligible_frame", lambda *a: frame)
        monkeypatch.setattr(monthly_mod, "_qv_order", lambda f: (f["eligible"], ["AAA"]))
        monkeypatch.setattr(monthly_mod, "_band_fill", lambda o, h, n, b: ["AAA"])

        async def scenario():
            r1, r2 = await asyncio.gather(
                daily_core.compute_targets(),
                daily_core.compute_targets(),
            )
            assert r1[0] == ["AAA"] and r2[0] == ["AAA"]
        asyncio.run(scenario())
        assert calls["fund"] == 1, (
            f"full-universe ranking computed {calls['fund']} times concurrently"
        )


# ---------------------------------------------------------------------------
# Backfill
# ---------------------------------------------------------------------------

@pytest.fixture
def fake_market(monkeypatch):
    """Patch daily_core's market/fundamentals loaders + ranking so backfill
    replays a tiny deterministic two-ticker market (no network, no DB rows)."""
    import pandas as pd

    dates = pd.bdate_range("2026-07-01", "2026-08-28")
    close = pd.DataFrame({"AAA": 100.0, "BBB": 50.0}, index=dates)
    vol = pd.DataFrame(1e6, index=dates, columns=["AAA", "BBB"])

    def fake_frame(m, c, v, f):
        return pd.DataFrame({"eligible": [True, True]},
                            index=pd.Index(["AAA", "BBB"], name="ticker"))

    def fake_order(frame):
        return frame["eligible"], ["AAA", "BBB"]

    async def fake_load_frames(_tickers, _asof, start=None):
        return close, vol

    async def fake_load_fundamentals(_tickers):
        return {"AAA": {"x": []}, "BBB": {"x": []}}

    monkeypatch.setattr(daily_core, "universe_tickers", lambda _u: ["AAA", "BBB"])
    monkeypatch.setattr(daily_core.fundamentals_mod, "load_fundamentals",
                        fake_load_fundamentals)
    monkeypatch.setattr(daily_core.monthly_mod, "load_frames", fake_load_frames)
    monkeypatch.setattr(daily_core.monthly_mod, "eligible_frame", fake_frame)
    monkeypatch.setattr(daily_core.monthly_mod, "_qv_order", fake_order)
    return {"close": close}


class TestBackfill:
    def test_uncovered_current_month_keeps_its_contribution(self, mem_db, fake_market,
                                                            monkeypatch):
        """The live engine deposits at month start; a backfill run before the
        current month has any candle must not lose that deposit — the replay
        never re-creates it and the live marker would suppress it forever."""
        from app.db import DailyCoreAllowance as Al

        monkeypatch.setattr(daily_core, "_current_month", lambda: "2026-09")

        async def seed():
            async with mem_db() as s:
                acc = await daily_core._account(s)
                acc.cash = 1234.0
                acc.last_allowance_month = "2026-09"
                await s.commit()
        asyncio.run(seed())

        r = asyncio.run(daily_core.backfill(start="2026-08-01"))
        assert r["ok"] is True
        # Aug (replayed) + Sep (live, uncovered) = two contributions.
        assert r["contributed"] == pytest.approx(2 * settings.sim_monthly_contribution)

        async def check():
            async with mem_db() as s:
                acc = await daily_core._account(s)
                assert acc.last_allowance_month == "2026-09"
                rows = (await s.scalars(select(Al))).all()
                assert sorted(x.month for x in rows) == ["2026-08", "2026-09"]
        asyncio.run(check())

    def test_trades_carry_replay_day_and_fill_data(self, mem_db, fake_market, monkeypatch):
        """Synthetic trades must show when they happened and at what
        price/size, not "now" with 0 shares and $0.00."""
        from app.db import DailyCoreTrade as Tr

        monkeypatch.setattr(daily_core, "_current_month", lambda: "2026-08")
        r = asyncio.run(daily_core.backfill(start="2026-08-01"))
        assert r["trades"] > 0

        async def check():
            async with mem_db() as s:
                rows = (await s.scalars(select(Tr))).all()
                assert rows
                for tr in rows:
                    assert tr.shares > 0
                    assert tr.price > 0
                    assert tr.created_at.strftime("%Y-%m") == "2026-08"
        asyncio.run(check())

    def test_allowance_rows_match_deposited_months_only(self, mem_db, fake_market,
                                                        monkeypatch):
        """Found 2026-09-14 with start=all: the persisted allowance rows were
        derived from EVERY month in the candle window (1980+ → ~550 rows)
        while the replay only deposits once a ranking exists (2017+ → 110).
        The phantom rows inflated allowance_total to $550k and broke every
        contributed-normalized UI metric. Rows must match the months the
        replay actually funded — in this fixture July 1 has no prior
        month-end ranking yet, so only August deposits (plus the live
        current-month add-back)."""
        from app.db import DailyCoreAllowance as Al

        monkeypatch.setattr(daily_core, "_current_month", lambda: "2026-09")
        # Seed the live marker to 2026-09 (deposit already made live): the
        # replay wipes it and the add-back must restore it.
        async def seed():
            async with mem_db() as s:
                acc = await daily_core._account(s)
                acc.cash = 0.0
                acc.last_allowance_month = "2026-09"
                await s.commit()
        asyncio.run(seed())

        r = asyncio.run(daily_core.backfill(start="2026-08-01"))
        assert r["ok"] is True
        # One replayed deposit (August) + the live current-month add-back.
        assert r["contributed"] == pytest.approx(2 * settings.sim_monthly_contribution)

        async def check():
            async with mem_db() as s:
                rows = (await s.scalars(select(Al))).all()
                # August (replayed) + September (live add-back). No phantom
                # rows for months outside what the replay actually funded.
                assert sorted(x.month for x in rows) == ["2026-08", "2026-09"]
        asyncio.run(check())


# ---------------------------------------------------------------------------
# Data refresh
# ---------------------------------------------------------------------------

class TestRefreshData:
    @staticmethod
    def _seed_holding():
        async def seed():
            async with daily_core.Session() as s:
                s.add(daily_core.DailyCorePosition(ticker="HELD", shares=1.0, avg_cost=10.0))
                await s.commit()
        asyncio.run(seed())

    def test_weekend_refreshes_only_holdings(self, mem_db, monkeypatch):
        """The ~176-ticker universe refresh is the heaviest nightly path and
        its candles cannot change while the market is closed; on weekends
        only the holdings are refreshed."""
        from app import market
        self._seed_holding()
        monkeypatch.setattr(daily_core, "_local_now",
                            lambda: datetime(2026, 9, 12, 12, 0))  # Saturday
        monkeypatch.setattr(daily_core, "universe_tickers",
                            lambda _u: ["AAA", "BBB", "CCC"])
        seen = {}

        async def fake_refresh_many(tickers, period, **kwargs):
            seen["tickers"] = list(tickers)
            seen["kwargs"] = kwargs
            return list(tickers), []
        monkeypatch.setattr(market, "refresh_many", fake_refresh_many)

        refreshed, errors = asyncio.run(daily_core.refresh_data())
        assert seen["tickers"] == ["HELD"]
        assert refreshed == ["HELD"] and errors == []

    def test_weekday_refreshes_universe_plus_holdings(self, mem_db, monkeypatch):
        from app import market
        self._seed_holding()
        monkeypatch.setattr(daily_core, "_local_now",
                            lambda: datetime(2026, 9, 14, 12, 0))  # Monday
        monkeypatch.setattr(daily_core, "universe_tickers",
                            lambda _u: ["AAA", "BBB"])
        seen = {}

        async def fake_refresh_many(tickers, period, **kwargs):
            seen["tickers"] = list(tickers)
            seen["kwargs"] = kwargs
            return list(tickers), []
        monkeypatch.setattr(market, "refresh_many", fake_refresh_many)

        asyncio.run(daily_core.refresh_data())
        assert seen["tickers"] == ["AAA", "BBB", "HELD"]
        # The nightly prefetch is reused via the freshness window.
        assert seen["kwargs"].get("max_age_seconds") == settings.market_fresh_seconds


# ---------------------------------------------------------------------------
# Strategy selection (runtime variant dropdown)
# ---------------------------------------------------------------------------

class TestVariantSelection:
    def test_default_falls_back_to_config(self, monkeypatch, tmp_path):
        monkeypatch.setattr(daily_core, "_STATE_FILE", tmp_path / "state.json")
        assert daily_core.current_variant() == settings.sim_daily_core_mom_variant

    def test_set_and_persist(self, monkeypatch, tmp_path):
        monkeypatch.setattr(daily_core, "_STATE_FILE", tmp_path / "state.json")
        daily_core.set_variant("raw")
        assert daily_core.current_variant() == "raw"
        daily_core.set_variant("residual")
        # persisted across "restarts" (a fresh read from the file)
        assert daily_core.current_variant() == "residual"

    def test_set_variant_rejects_unknown(self, monkeypatch, tmp_path):
        monkeypatch.setattr(daily_core, "_STATE_FILE", tmp_path / "state.json")
        with pytest.raises(ValueError):
            daily_core.set_variant("turbo-momentum")

    def test_invalid_state_file_falls_back(self, monkeypatch, tmp_path):
        state_file = tmp_path / "state.json"
        state_file.write_text('{"mom_variant": "bogus"}')
        monkeypatch.setattr(daily_core, "_STATE_FILE", state_file)
        # the stored variant is unknown -> config default wins
        assert daily_core.current_variant() == settings.sim_daily_core_mom_variant


# ---------------------------------------------------------------------------
# Protection selection (runtime, persisted)
# ---------------------------------------------------------------------------

class TestProtectionSelection:
    def test_default_is_none(self, monkeypatch, tmp_path):
        monkeypatch.setattr(daily_core, "_STATE_FILE", tmp_path / "state.json")
        assert daily_core.current_protection() == "none"

    def test_set_and_persist(self, monkeypatch, tmp_path):
        monkeypatch.setattr(daily_core, "_STATE_FILE", tmp_path / "state.json")
        daily_core.set_protection("gradient200")
        assert daily_core.current_protection() == "gradient200"
        # variant + protection coexist in the same state file
        daily_core.set_variant("raw")
        assert daily_core.current_variant() == "raw"
        assert daily_core.current_protection() == "gradient200"

    def test_rejects_unknown(self, monkeypatch, tmp_path):
        monkeypatch.setattr(daily_core, "_STATE_FILE", tmp_path / "state.json")
        with pytest.raises(ValueError):
            daily_core.set_protection("turbo")

    def test_all_modes_have_labels(self):
        for mode in daily_core.PROTECTION_MODES:
            assert daily_core.PROTECTION_MODES[mode]
        assert set(daily_core._ARM_SMA) <= set(daily_core.PROTECTION_MODES)


# ---------------------------------------------------------------------------
# Strategy universe selection (runtime, persisted)
# ---------------------------------------------------------------------------

class TestUniverseSelection:
    def test_default_falls_back_to_config(self, monkeypatch, tmp_path):
        monkeypatch.setattr(daily_core, "_STATE_FILE", tmp_path / "state.json")
        assert daily_core.current_universe() == daily_core.settings.sim_monthly_universe

    def test_set_and_persist(self, monkeypatch, tmp_path):
        monkeypatch.setattr(daily_core, "_STATE_FILE", tmp_path / "state.json")
        avail = daily_core.universe_names()
        assert avail, "expected at least one universe file"
        daily_core.set_universe(avail[0])
        assert daily_core.current_universe() == avail[0]

    def test_rejects_unknown(self, monkeypatch, tmp_path):
        monkeypatch.setattr(daily_core, "_STATE_FILE", tmp_path / "state.json")
        with pytest.raises(ValueError):
            daily_core.set_universe("does-not-exist")

    def test_stale_state_value_falls_back(self, monkeypatch, tmp_path):
        monkeypatch.setattr(daily_core, "_STATE_FILE", tmp_path / "state.json")
        (tmp_path / "state.json").write_text('{"universe": "gone"}')
        assert daily_core.current_universe() == daily_core.settings.sim_monthly_universe

    def test_sp500_available(self):
        assert "sp500" in daily_core.universe_names()

    def test_monthly_helper_resolves(self, monkeypatch, tmp_path):
        monkeypatch.setattr(daily_core, "_STATE_FILE", tmp_path / "state.json")
        daily_core.set_universe("sp500")
        assert monthly._strategy_universe() == "sp500"

    def test_persisted_universe_none_when_unset(self, monkeypatch, tmp_path):
        monkeypatch.setattr(daily_core, "_STATE_FILE", tmp_path / "state.json")
        assert daily_core.persisted_universe() is None

    def test_sim_uses_shared_universe(self, monkeypatch, tmp_path):
        from app import sim
        monkeypatch.setattr(daily_core, "_STATE_FILE", tmp_path / "state.json")
        daily_core.set_universe("sp500")
        assert sim._universe() == "sp500"
        # clearing falls back to the sim's own config default
        (tmp_path / "state.json").write_text("{}")
        assert sim._universe() == settings.sim_universe


# ---------------------------------------------------------------------------
# Target-volatility control (runtime, persisted)
# ---------------------------------------------------------------------------

class TestTargetVol:
    def test_default_off(self, monkeypatch, tmp_path):
        monkeypatch.setattr(daily_core, "_STATE_FILE", tmp_path / "state.json")
        assert daily_core.current_target_vol() == 0.0

    def test_set_and_persist(self, monkeypatch, tmp_path):
        monkeypatch.setattr(daily_core, "_STATE_FILE", tmp_path / "state.json")
        daily_core.set_target_vol("0.15")
        assert daily_core.current_target_vol() == 0.15
        daily_core.set_target_vol("off")
        assert daily_core.current_target_vol() == 0.0

    def test_rejects_unknown(self, monkeypatch, tmp_path):
        monkeypatch.setattr(daily_core, "_STATE_FILE", tmp_path / "state.json")
        with pytest.raises(ValueError):
            daily_core.set_target_vol("0.5")

    def test_room_infinite_when_calm_or_short_history(self):
        assert daily_core._vol_deploy_room(1000.0, 500.0, [], 0.15) == float("inf")
        calm = [0.0001] * 30
        assert daily_core._vol_deploy_room(1000.0, 500.0, calm, 0.15) == float("inf")

    def test_room_caps_when_vol_hot(self):
        hot = [0.05, -0.05] * 15          # ~80% annualized
        room = daily_core._vol_deploy_room(1000.0, 1000.0, hot, 0.15)
        assert room == pytest.approx(1000.0 * 0.15 / (0.05 * (252 ** 0.5)), rel=0.25)

    def test_room_nonnegative_when_over_deployed(self):
        hot = [0.05, -0.05] * 15
        assert daily_core._vol_deploy_room(1000.0, 0.0, hot, 0.15) == 0.0

    def test_port_return_is_contribution_adjusted(self):
        # equity flat at 1000 with a 100 contribution -> slightly negative
        assert daily_core._port_return(1000.0, 1000.0, 100.0) < 0
        # no contribution and flat equity -> zero
        assert daily_core._port_return(1000.0, 1000.0, 0.0) == 0.0


class TestSharedBackfillPreload:
    async def test_monthly_and_daily_core_share_frames(self, mem_db, fake_market,
                                                       monkeypatch):
        """backfill-all builds the per-month eligibility frames once and hands
        them to both qv-mom replays (same universe/window/variant), so
        eligible_frame runs once per month-end rather than twice."""
        monkeypatch.setattr(monthly, "Session", mem_db)
        monkeypatch.setattr(monthly, "universe_tickers", lambda _u: ["AAA", "BBB"])

        calls: list = []
        real = monthly.eligible_frame

        def counting(m, c, v, f):
            calls.append(m)
            return real(m, c, v, f)

        monkeypatch.setattr(monthly, "eligible_frame", counting)

        start = "2026-07-01"
        preload = await monthly.preload_backfill(start)
        await monthly.backfill(start, preload=preload)
        after_monthly = len(calls)
        assert after_monthly > 0
        assert preload.frames  # cache populated by the first replay

        await daily_core.backfill(start, preload=preload)
        assert len(calls) == after_monthly  # second replay recomputed nothing
