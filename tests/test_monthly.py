"""Tests for the monthly qv-mom portfolio (app.monthly).

These tests use an in-memory SQLite database so they don't touch the real
data volume. They cover:

- Point-in-time fact helpers (_asof_entries, _latest_as_of, _ttm_as_of)
- Eligibility frame (filters, momentum, ROE, P/FCF math)
- Scoring order + explicit tie-breakers
- Hysteresis band fill (held name inside band keeps slot, outside released)
- Rebalance scheduling (last trading day detection)
- Allowance deposit (once per month)
- Rebalance execution: SELLs before BUYs, equal-weight sizing, monthly
  idempotence
"""

from __future__ import annotations

import asyncio
import urllib.error
from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import StaticPool, select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app import monthly
from app.config import settings
from app.db import Base, MonthlyAccount

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
async def mem_db(monkeypatch):
    """Replace the monthly engine's database with an in-memory SQLite DB."""
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    session_factory = async_sessionmaker(engine, expire_on_commit=False)

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    monkeypatch.setattr(monthly, "Session", session_factory)
    import app.db as db_mod
    monkeypatch.setattr(db_mod, "Session", session_factory)
    import app.fundamentals as fund_mod
    monkeypatch.setattr(fund_mod, "Session", session_factory)
    import app.edgar as edgar_mod
    monkeypatch.setattr(edgar_mod, "Session", session_factory)

    async with session_factory() as s:
        s.add(MonthlyAccount(id=1, cash=settings.sim_monthly_start_cash))
        await s.commit()
    yield session_factory
    # Dispose the engine AFTER the test's event loop closed: aiosqlite runs
    # every connection on a dedicated worker thread that schedules its result
    # back onto the loop. Without dispose, that outlives the loop and each
    # pending operation crashes with "RuntimeError: Event loop is closed"
    # (surfaced by pytest as PytestUnhandledThreadExceptionWarning).
    await engine.dispose()


def _make_close(n_days: int = 400, tickers=("AAA", "BBB", "CCC")) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    dates = pd.bdate_range("2024-06-03", periods=n_days)
    data = rng.normal(0.0004, 0.012, (n_days, len(tickers))).cumsum(axis=0) + 100
    return pd.DataFrame(data, index=dates, columns=list(tickers))


def _make_vol(close: pd.DataFrame, shares: float = 5e6) -> pd.DataFrame:
    return pd.DataFrame(shares, index=close.index, columns=close.columns)


def _fund_all_positive(tickers) -> dict:
    """Facts that make every ticker eligible: equity 5B, TTM NI 500M (ROE 0.1),
    1e8 shares, positive OCF/capex. All filed well before the last candle date
    of _make_close() (bdate_range 2024-06-03 + 400td ends 2025-12)."""
    return {
        t: {
            "StockholdersEquity": [{"start": None, "end": "2024-06-30", "filed": "2024-08-14", "val": 5e9}],
            "NetIncomeLoss": [{"start": "2024-01-01", "end": "2024-12-31", "filed": "2025-02-14", "val": 5e8}],
            "CommonStockSharesOutstanding": [{"start": None, "end": "2024-06-30", "filed": "2024-08-14", "val": 1e8}],
            "NetCashProvidedByUsedInOperatingActivities": [
                {"start": "2024-01-01", "end": "2024-12-31", "filed": "2025-02-14", "val": 1e9}],
            "PaymentsToAcquirePropertyPlantAndEquipment": [
                {"start": "2024-01-01", "end": "2024-12-31", "filed": "2025-02-14", "val": -2e8}],
        }
        for t in tickers
    }


# ---------------------------------------------------------------------------
# Point-in-time fact helpers
# ---------------------------------------------------------------------------

def test_asof_entries_filters_not_yet_public():
    ents = [
        {"start": "2025-01-01", "end": "2025-03-31", "filed": "2025-05-15", "val": 1.0},
        {"start": "2025-01-01", "end": "2025-03-31", "filed": "2025-05-01", "val": 2.0},
        {"start": "2025-04-01", "end": "2025-06-30", "filed": "2025-08-15", "val": 3.0},
    ]
    # before the Q1 filing: nothing public
    assert monthly._asof_entries(ents, pd.Timestamp("2025-04-30")) == []
    # after: restatement (later filed) replaces the original
    out = monthly._asof_entries(ents, pd.Timestamp("2025-06-30"))
    assert len(out) == 1 and out[0]["val"] == 1.0
    # after Q2 filing: both quarters
    out2 = monthly._asof_entries(ents, pd.Timestamp("2025-09-01"))
    assert {e["val"] for e in out2} == {1.0, 3.0}


def test_latest_as_of_instant_facts_only():
    ents = [
        {"start": "2024-01-01", "end": "2024-12-31", "filed": "2025-02-14", "val": 9.0},
        {"start": None, "end": "2024-12-31", "filed": "2025-02-14", "val": 5e9},
        {"start": None, "end": "2025-06-30", "filed": "2025-07-30", "val": 6e9},
    ]
    d = pd.Timestamp("2025-08-01")
    assert monthly._latest_as_of(ents, d) == ("2025-06-30", 6e9)
    # before the second filing: only the older instant fact
    assert monthly._latest_as_of(ents, pd.Timestamp("2025-03-01")) == ("2024-12-31", 5e9)


def test_ttm_four_quarter_chain():
    quarters = [
        ("2024-07-01", "2024-09-30"), ("2024-10-01", "2024-12-31"),
        ("2025-01-01", "2025-03-31"), ("2025-04-01", "2025-06-30"),
    ]
    ents = [{
        "start": s, "end": e,
        "filed": (pd.Timestamp(e) + pd.Timedelta(days=45)).strftime("%Y-%m-%d"),
        "val": 10.0,
    } for s, e in quarters]
    assert monthly._ttm_as_of(ents, pd.Timestamp("2025-08-28")) == 40.0
    # with only 3 quarters public: no TTM
    assert monthly._ttm_as_of(ents[:3], pd.Timestamp("2025-08-28")) is None


def test_ttm_prefers_fy_identity():
    ents = [
        # FY 2024
        {"start": "2024-01-01", "end": "2024-12-31", "filed": "2025-02-14", "val": 400.0},
        # Q1-Q3 2025 standalone quarters
        {"start": "2025-01-01", "end": "2025-03-31", "filed": "2025-05-15", "val": 100.0},
        {"start": "2025-04-01", "end": "2025-06-30", "filed": "2025-08-15", "val": 110.0},
        # 9-month YTD span
        {"start": "2025-01-01", "end": "2025-09-30", "filed": "2025-11-15", "val": 320.0},
        # prior-year 9-month YTD
        {"start": "2024-01-01", "end": "2024-09-30", "filed": "2024-11-15", "val": 300.0},
    ]
    # TTM = FY400 + YTD320 - prior300 = 420
    assert monthly._ttm_as_of(ents, pd.Timestamp("2025-12-01")) == 420.0


# ---------------------------------------------------------------------------
# Eligibility + scoring + hysteresis
# ---------------------------------------------------------------------------

def test_eligible_frame_filters_and_factors():
    close = _make_close()
    vol = _make_vol(close)
    fund = _fund_all_positive(close.columns)
    d = pd.Timestamp(close.index[-1])
    frame = monthly.eligible_frame(d, close, vol, fund)
    assert frame is not None
    assert list(frame.index) == list(close.columns)
    assert frame["eligible"].all()
    # ROE = 5e8 / 5e9 = 0.1; mcap = price * 1e8; P/FCF = mcap / 8e8
    assert np.allclose(frame["roe"], 0.1)
    assert (frame["mcap"] > 0).all()
    assert (frame["p_fcf"] > 0).all()
    # dollar volume = price * 5e6 > 1e7 threshold
    assert (frame["dollar_vol"] > settings.sim_monthly_min_dollar_vol).all()


def test_eligible_frame_missing_capex_is_missing_fcf():
    """Capex missing -> FCF missing: OCF alone is never substituted as FCF.

    The old fallback assumed capex == 0, which OVERSTATES FCF for every
    non-repiler (capex is a real cash outflow) and inflated the 1/P-FCF
    value factor (audit P0-A)."""
    close = _make_close()
    vol = _make_vol(close)
    fund = _fund_all_positive(close.columns)
    # drop capex everywhere
    for rec in fund.values():
        rec.pop("PaymentsToAcquirePropertyPlantAndEquipment", None)
    d = pd.Timestamp(close.index[-1])
    frame = monthly.eligible_frame(d, close, vol, fund)
    assert frame is not None
    assert frame["eligible"].all()          # missing p_fcf is not a filter
    assert frame["p_fcf"].isna().all()      # FCF missing, NOT ocf-as-FCF


def test_eligible_frame_missing_ocf_is_missing_fcf():
    close = _make_close()
    vol = _make_vol(close)
    fund = _fund_all_positive(close.columns)
    for rec in fund.values():
        rec.pop("NetCashProvidedByUsedInOperatingActivities", None)
    d = pd.Timestamp(close.index[-1])
    frame = monthly.eligible_frame(d, close, vol, fund)
    assert frame is not None
    assert frame["eligible"].all()
    assert frame["p_fcf"].isna().all()


def test_quality_snapshot_missing_capex_is_missing_fcf():
    """quality_snapshot mirrors eligible_frame's FCF contract, and the
    negative-capex arithmetic is unchanged: FCF = OCF + capex with capex
    canonical (a negative outflow)."""
    fund = _fund_all_positive(["AAA"])
    asof = pd.Timestamp("2025-06-30")
    # missing capex -> missing P/FCF
    no_capex = {"AAA": {k: v for k, v in fund["AAA"].items()
                        if k != "PaymentsToAcquirePropertyPlantAndEquipment"}}
    snap = monthly.quality_snapshot(no_capex, "AAA", asof, close=100.0)
    assert snap is not None
    assert snap["roe"] == pytest.approx(0.1)
    assert snap["p_fcf"] is None
    # capex PRESENT and negative: P/FCF = 100 * 1e8 / (1e9 + (-2e8)) = 12.5
    snap_ok = monthly.quality_snapshot(fund, "AAA", asof, close=100.0)
    assert snap_ok["p_fcf"] == pytest.approx(12.5)
    close = _make_close(n_days=100)
    vol = _make_vol(close)
    fund = _fund_all_positive(close.columns)
    d = pd.Timestamp(close.index[-1])
    assert monthly.eligible_frame(d, close, vol, fund) is None


def test_eligible_frame_mcap_filter():
    close = _make_close()
    vol = _make_vol(close)
    fund = _fund_all_positive(close.columns)
    # ~100 * 5e7 = $5B: at (not above) the $5B threshold -> not eligible.
    # Random-walk prices wobble around 100, so assert on the mcap column
    # instead of the boolean (a name drifting above $5B stays eligible).
    for t in fund:
        fund[t]["CommonStockSharesOutstanding"][0]["val"] = 5e7
    d = pd.Timestamp(close.index[-1])
    frame = monthly.eligible_frame(d, close, vol, fund)
    assert frame is not None
    for t, row in frame.iterrows():
        if row["mcap"] <= settings.sim_monthly_min_mcap:
            assert not row["eligible"], t


def test_qv_order_tie_breakers():
    rows = {
        "AAA": {"roe": 0.10, "p_fcf": 20.0, "mcap": 1e10, "dollar_vol": 5e7, "mom": 0.10, "price": 100, "vol": 0.2},
        "BBB": {"roe": 0.10, "p_fcf": 20.0, "mcap": 1e10, "dollar_vol": 5e7, "mom": 0.10, "price": 100, "vol": 0.2},
        "CCC": {"roe": 0.20, "p_fcf": 25.0, "mcap": 1e10, "dollar_vol": 5e7, "mom": 0.10, "price": 100, "vol": 0.2},
        "DDD": {"roe": 0.05, "p_fcf": None, "mcap": 1e10, "dollar_vol": 5e7, "mom": 0.10, "price": 100, "vol": 0.2},
    }
    frame = pd.DataFrame(rows).T
    frame["eligible"] = True
    elig, order = monthly._qv_order(frame)
    # CCC best (highest ROE); AAA before BBB (identical: ticker asc); DDD last (missing P/FCF)
    assert order[0] == "CCC"
    assert order.index("AAA") < order.index("BBB")
    assert order[-1] == "DDD"


def test_band_fill_hysteresis():
    order = [f"T{i:02d}" for i in range(30)]  # T00 best
    # held name at rank 15 (inside band 20, outside top10) keeps its slot
    picked = monthly._band_fill(order, ["T15"], 10, 20)
    assert "T15" in picked and len(picked) == 10
    # held name at rank 25 (outside band) is released
    picked2 = monthly._band_fill(order, ["T25"], 10, 20)
    assert "T25" not in picked2 and len(picked2) == 10
    # no holdings: pure top-10
    assert monthly._band_fill(order, [], 10, 20) == order[:10]


class TestDeployTargets:
    """The shared deployment-target contract (audit P0-D): the hysteresis
    picks in RANKING order — never band[:target_n], never alphabetical."""

    # deliberately NON-alphabetical ranking: Z01 is the best name, Z25 the
    # worst in the band
    ORDER = ["Z01", "M02", "C03", "P04", "B05", "A06", "Z07", "Q08", "K09",
             "J10", "I11", "H12", "G13", "F14", "E15", "D16", "C17", "B18",
             "A19", "Z20"]

    def test_rank_order_not_alphabetical(self):
        # 3 held names deep in the band: membership preserved, order is the
        # ranking's (deployment funds best rank first with limited cash)
        held = ["C17", "Z01", "H12"]
        got = monthly._deploy_targets(self.ORDER, held, 10, 20, refill=False)
        assert got == ["Z01", "H12", "C17"]  # rank order, NOT sorted(held)

    def test_refill_true_is_band_fill_membership_in_rank_order(self):
        held = ["C17", "H12"]
        got = monthly._deploy_targets(self.ORDER, held, 10, 20, refill=True)
        filled = monthly._band_fill(self.ORDER, held, 10, 20)
        assert set(got) == set(filled) and len(got) == 10
        assert got == [t for t in self.ORDER if t in set(got)]  # rank order
        # hysteresis: the held band names keep their slots over raw top-10
        assert "H12" in got and "C17" in got and "J10" not in got

    def test_refill_false_keeps_held_no_fills(self):
        # 10 held in-band + refill=False: exactly those held names, NO new
        # name (the deferred-release replay cannot open a transient 11th)
        held = ["Z01", "M02", "C03", "P04", "B05", "A06", "Z07", "Q08",
                "K09", "J10"]
        got = monthly._deploy_targets(self.ORDER, held, 10, 20, refill=False)
        assert got == held

    def test_preexisting_overflow_capped_no_fills(self):
        # 11 held in-band: deterministic retention — top-10 by rank keep
        # their slots, the excess is dropped from picks (no sell here, no
        # new name until a slot frees)
        held = ["Z01", "M02", "C03", "P04", "B05", "A06", "Z07", "Q08",
                "K09", "J10", "I11"]
        got = monthly._deploy_targets(self.ORDER, held, 10, 20, refill=True)
        assert got == held[:10] and "I11" not in got

    def test_out_of_band_pending_release_not_in_picks(self):
        # 10 held, one of them outside a 15-name band (Z20 at rank 19): the
        # pending release is not a deployment candidate between rebuilds
        held = ["Z01", "M02", "C03", "P04", "B05", "A06", "Z07", "Q08",
                "K09", "Z20"]
        got = monthly._deploy_targets(self.ORDER, held, 10, 15, refill=False)
        assert "Z20" not in got and len(got) == 9
        got2 = monthly._deploy_targets(self.ORDER, held, 10, 15, refill=True)
        assert "Z20" not in got2 and len(got2) == 10  # released + refilled

    def test_pending_release_plan_keeps_replacements_queued(self):
        # Review round 3, exact repro: 9 held (8 in-band + 1 out-of-band
        # pending release), target 10 — the PLAN still queues 2 replacements
        # (the release frees the slot), NOT 1. This is PLANNING semantics:
        # the live engine calls _deploy_targets BEFORE its SELL pass so
        # replacements are never deferred to the next day; the EXECUTORS
        # (live BUY loop, backfill do_buy, replay trade()/BUY blocks) each
        # enforce the ACTUAL post-SELL capacity with `len(shares) <
        # target_n` — capping the plan here instead would defer every
        # replacement after a successful release.
        held = ["Z01", "M02", "C03", "P04", "B05", "A06", "Z07", "Q08",
                "OLD_PENDING"]  # OLD_PENDING is stale/out-of-band, still held
        got = monthly._deploy_targets(self.ORDER, held, 10, 20, refill=True)
        assert len(got) == 10  # full post-release membership
        assert "K09" in got and "J10" in got  # both replacement fills queued
        assert "OLD_PENDING" not in got  # a pending release is never a pick


class TestCompletedMonthEnds:
    """The shared calendar rule: a month-end is the scheduler's last US
    trading day of the month, counted only when its candle is stored."""

    def test_good_friday_month_end_walks_back(self):
        # March 2024: last weekday is Good Friday 2024-03-29 (NYSE closed)
        # -> the scheduled month-end is 2024-03-28; it must be found WITHOUT
        # any April candle existing (no future-candle completion test).
        idx = pd.bdate_range("2024-03-01", "2024-03-28")
        assert monthly.completed_month_ends(idx, idx[0], idx[-1]) == \
            [pd.Timestamp("2024-03-28")]

    def test_truncation_at_month_end_still_counts(self):
        # A frame truncated EXACTLY at the scheduled month-end keeps its
        # month-end (the old later-candle rule lost it) — future-data
        # invariance at the endpoint.
        idx = pd.bdate_range("2024-01-02", "2024-01-31")
        assert monthly.completed_month_ends(idx, idx[0], idx[-1]) == \
            [pd.Timestamp("2024-01-31")]

    def test_last_mid_month_candle_is_not_a_month_end(self):
        # Data ends 2026-01-15: no fabricated month-end from the last bar.
        idx = pd.bdate_range("2026-01-02", "2026-01-15")
        assert monthly.completed_month_ends(idx, idx[0], idx[-1]) == []

    def test_window_bounds_and_dedup(self):
        idx = pd.bdate_range("2025-11-03", "2026-01-15")
        got = monthly.completed_month_ends(idx, pd.Timestamp("2025-11-15"),
                                           pd.Timestamp("2025-12-31"))
        assert got == [pd.Timestamp("2025-11-28"), pd.Timestamp("2025-12-31")]

    def test_month_end_candle_hole_not_papered_over(self):
        # The scheduled month-end's candle is missing (hole) while later
        # candles exist: no month-end — never the last mid-month bar.
        idx = pd.bdate_range("2025-01-02", "2025-02-10")
        hole = pd.Timestamp("2025-01-31")
        assert hole in idx
        idx2 = idx.delete(idx.get_loc(hole))
        got = monthly.completed_month_ends(idx2, idx2[0], idx2[-1])
        assert pd.Timestamp("2025-01-31") not in got


def test_pick_portfolio_end_to_end():
    close = _make_close()
    vol = _make_vol(close)
    # 12 tickers so hysteresis matters
    close["DDD"] = close["AAA"] * 1.01
    vol["DDD"] = vol["AAA"]
    fund = _fund_all_positive(close.columns)
    d = pd.Timestamp(close.index[-1])
    picks, frame = monthly.pick_portfolio(d, close, vol, fund, holdings=[])
    assert len(picks) == min(settings.sim_monthly_target_n, len(close.columns))
    assert frame is not None and frame["eligible"].all()


# ---------------------------------------------------------------------------
# Scheduling
# ---------------------------------------------------------------------------

def test_is_rebalance_day():
    tz = UTC
    # 2026-08-31 is a Monday and the last weekday of August
    assert monthly.is_rebalance_day(datetime(2026, 8, 31, 22, 0, tzinfo=tz))
    assert not monthly.is_rebalance_day(datetime(2026, 8, 30, 22, 0, tzinfo=tz))  # Sunday
    assert not monthly.is_rebalance_day(datetime(2026, 8, 28, 22, 0, tzinfo=tz))  # Friday, not last
    # weekend walked back: 2026-10-31 is a Saturday -> last trading day is Fri 10-30
    assert monthly.is_rebalance_day(datetime(2026, 10, 30, 22, 0, tzinfo=tz))
    assert not monthly.is_rebalance_day(datetime(2026, 10, 31, 22, 0, tzinfo=tz))


def test_month_last_trading_day():
    tz = UTC
    out = monthly.month_last_trading_day(datetime(2026, 8, 15, tzinfo=tz))
    assert out.date() == datetime(2026, 8, 31).date() and out.weekday() == 0


def test_rebalance_day_skips_holiday_month_end():
    """A month whose last weekday is a NYSE holiday (2024-03-29 Good Friday)
    must rebalance on the prior trading day, not be skipped entirely by the
    scheduler's trading-day guard."""
    tz = UTC
    assert monthly.month_last_trading_day(datetime(2024, 3, 15, tzinfo=tz)).date() \
        == datetime(2024, 3, 28).date()
    assert monthly.is_rebalance_day(datetime(2024, 3, 28, 22, 0, tzinfo=tz))
    assert not monthly.is_rebalance_day(datetime(2024, 3, 29, 22, 0, tzinfo=tz))


# ---------------------------------------------------------------------------
# Engine: allowance + rebalance execution
# ---------------------------------------------------------------------------

async def test_deposit_allowance_once_per_month(mem_db, monkeypatch):
    r1 = await monthly.deposit_allowance()
    assert r1["deposited"] is True
    r2 = await monthly.deposit_allowance()
    assert r2["deposited"] is False
    val = await monthly.monthly_valuate()
    assert val["cash"] == settings.sim_monthly_contribution
    assert val["allowance_total"] == settings.sim_monthly_contribution


async def test_exec_buy_sell(mem_db):
    await monthly.deposit_allowance()  # fund the account
    t1 = await monthly._exec_buy("AAA", 50.0, 1000.0, "test")
    assert t1 is not None and t1["cost"] == 1000.0
    val = await monthly.monthly_valuate()
    assert val["cash"] == 0.0
    assert val["positions"][0]["ticker"] == "AAA"
    # sell everything
    t2 = await monthly._exec_sell("AAA", 60.0, "test")
    assert t2 is not None and t2["proceeds"] > 1000.0
    val2 = await monthly.monthly_valuate()
    assert val2["positions"] == []
    assert val2["cash"] == pytest.approx(t2["proceeds"])


async def test_exec_buy_respects_cash(mem_db):
    t = await monthly._exec_buy("AAA", 50.0, 10_000.0, "test")  # budget > cash (1000)
    assert t is None
    val = await monthly.monthly_valuate()
    assert val["cash"] == settings.sim_monthly_start_cash


async def test_run_rebalance_executes_and_is_idempotent(mem_db, monkeypatch):
    """Full rebalance: seeds candles+fundamentals in the DB, monkeypatches the
    network fetches, runs run_rebalance, checks SELLs/BUYs and idempotence."""
    from app.db import Candle, Fundamental

    # --- seed candles: 12 universe tickers x 400 trading days
    from app.screener import tickers as universe_tickers
    uni = universe_tickers("diversified-plus")
    tickers = uni[:12]
    dates = pd.bdate_range("2024-06-03", periods=400)
    prices = {}
    for i, t in enumerate(tickers):
        # constant per-ticker price level: open == close on every session
        prices[t] = 100.0 + i
    close = pd.DataFrame(prices, index=dates)

    # --- seed fundamentals: T00..T09 high quality (score desc by index),
    #     T10/T11 worse so the top-10 excludes them
    async with mem_db() as s:
        for t in tickers:
            for ts, px in zip(dates, close[t], strict=False):
                s.add(Candle(ticker=t, timestamp=ts.to_pydatetime(), open=px, high=px,
                             low=px, close=px, volume=2e6))
        quality = {t: (10 - i) for i, t in enumerate(tickers)}  # T00 best
        for t in tickers:
            q = quality[t]
            s.add(Fundamental(ticker=t, tag="StockholdersEquity", start=None,
                              end="2024-06-30", filed="2024-08-14", val=1e9, currency="USD"))
            s.add(Fundamental(ticker=t, tag="NetIncomeLoss", start="2024-07-01",
                              end="2025-06-30", filed="2025-08-14", val=1e8 * q, currency="USD"))
            s.add(Fundamental(ticker=t, tag="NetCashProvidedByUsedInOperatingActivities",
                              start="2024-07-01", end="2025-06-30", filed="2025-08-14",
                              val=5e8, currency="USD"))
            s.add(Fundamental(ticker=t, tag="PaymentsToAcquirePropertyPlantAndEquipment",
                              start="2024-07-01", end="2025-06-30", filed="2025-08-14",
                              val=-1e8, currency="USD"))
            s.add(Fundamental(ticker=t, tag="CommonStockSharesOutstanding", start=None,
                              end="2024-06-30", filed="2024-08-14", val=1e8, currency="shares"))
        await s.commit()

    # --- monkeypatch data refresh (no network)
    async def fake_refresh_data(tickers_list):
        return [], {}

    monkeypatch.setattr(monthly, "refresh_data", fake_refresh_data)

    # Constant fixture prices: the fill session's OPEN equals its close, so
    # the equal-weight sizing (budget/OPEN) is exactly the post-trade mark
    # (shares*CLOSE) — rounding and float-sum order can't skew the check.
    # (The random-walk tape under the deferred-fill cadence marked at F's
    # close against fills at F's open drifted past the 2% tolerance.)
    # --- run 1: deposits allowance + buys top-10 at F's open
    clock = dates[-1].to_pydatetime().replace(hour=23, minute=30, tzinfo=UTC)
    r1 = await monthly.run_rebalance(now=clock)
    assert r1.get("rebalanced") is True
    assert len(r1["picks"]) == settings.sim_monthly_target_n
    assert "T10" not in r1["picks"] and "T11" not in r1["picks"]
    buys = [t for t in r1["trades"] if t["side"] == "BUY"]
    assert len(buys) == settings.sim_monthly_target_n
    val = await monthly.monthly_valuate(now=clock)
    assert len(val["positions"]) == settings.sim_monthly_target_n
    # equal weight: each position ~ 1/10 of equity (allowance deposited)
    expected = val["total_equity"] / settings.sim_monthly_target_n
    for p in val["positions"]:
        assert p["value"] == pytest.approx(expected, rel=0.02)

    # --- run 2 same month: skipped
    r2 = await monthly.run_rebalance(now=clock)
    assert r2.get("skipped") is True

    # --- forced re-run with unchanged data: no churn (hysteresis keeps picks)
    r3 = await monthly.run_rebalance(force=True, now=clock)
    assert r3.get("rebalanced") is True
    assert r3["picks"] == r1["picks"]
    assert r3["trades"] == []  # fully rebalanced portfolio -> no trades

    # --- audit log: one row per month, reused and updated by the forced re-run
    async with mem_db() as s:
        from app.db import MonthlyRebalance
        rows = (await s.scalars(select(MonthlyRebalance))).all()
        assert len(rows) == 1  # unique on rebal_month
        assert rows[0].picked == ",".join(r1["picks"])


async def test_run_monthly_cycle_gate(mem_db, monkeypatch):
    """run_monthly_cycle is the scheduler entry: disabled portfolio skips;
    enabled, it delegates to run_rebalance — which fills the PENDING
    month-end rebuild under the session fill contract (the trigger is not
    ``is_rebalance_day``; the fill lands on the first session after the
    completed month-end)."""
    seen: dict = {}

    async def fake_rebalance(force=False, now=None):
        seen["force"], seen["now"] = force, now
        return {"rebalanced": True}

    monkeypatch.setattr(monthly, "run_rebalance", fake_rebalance)
    clock = datetime(2026, 8, 28, 22, 0, tzinfo=UTC)
    out = await monthly.run_monthly_cycle(now=clock)
    assert out == {"rebalanced": True}
    assert seen["force"] is False and seen["now"] is clock

    monkeypatch.setattr(settings, "sim_monthly_enabled", False)
    out2 = await monthly.run_monthly_cycle(now=clock)
    assert out2 == {"skipped": True, "reason": "monthly portfolio disabled"}


# ---------------------------------------------------------------------------
# Daily snapshot + start-of-month deposit (curve-alignment with sim)
# ---------------------------------------------------------------------------

async def test_deposit_allowance_at_month_start(mem_db, monkeypatch):
    """The monthly portfolio must deposit at the START of the operator-local
    month (same as sim.deposit_allowance), not only at the month-end
    rebalance — otherwise its contributed total trails the sim's and the two
    equity curves are not comparable. It also must NOT depend on UTC."""
    from zoneinfo import ZoneInfo

    # Simulate an operator-local time early in a month whose UTC month differs
    # (2026-09-01 00:30 Vienna == 2026-08-31 22:30 UTC): the old UTC key would
    # have deposited for "2026-08" here, the shared anchor must use "2026-09".
    fake_now = datetime(2026, 8, 31, 22, 30, tzinfo=UTC)
    monkeypatch.setattr(monthly, "_current_month", lambda: fake_now.astimezone(
        ZoneInfo(monthly.settings.allowance_tz)).strftime("%Y-%m"))
    r1 = await monthly.deposit_allowance()
    assert r1["deposited"] is True
    assert r1["month"] == "2026-09"  # operator-local month, not UTC's 2026-08
    r2 = await monthly.deposit_allowance()  # same month -> idempotent
    assert r2["deposited"] is False
    val = await monthly.monthly_valuate()
    assert val["allowance_total"] == settings.sim_monthly_contribution


async def test_deposit_month_key_matches_sim_anchor(mem_db):
    """Both portfolios must compute the same month key at the same instant so
    contributed steps at the same boundary."""
    from app import sim as sim_mod
    assert monthly._current_month() == sim_mod._current_month()
    assert monthly._TZ == sim_mod._TZ


async def test_take_snapshot_records_daily_equity(mem_db):
    """take_snapshot() writes a MonthlySnapshot from the live valuation —
    the mechanism that makes the Monthly curve move every day."""
    from app.db import Candle

    # seed a price so valuation uses the fresh close, not the avg_cost fallback
    async with mem_db() as s:
        s.add(Candle(ticker="AAA", timestamp=datetime(2026, 9, 1, tzinfo=UTC),
                     open=60, high=60, low=60, close=60, volume=1000))
        await s.commit()
    await monthly.deposit_allowance()
    await monthly._exec_buy("AAA", 50.0, 500.0, "test")  # 10 shares @ 50
    r = await monthly.take_snapshot()
    assert r["snapshotted"] is True
    curve = await monthly.get_equity_curve()
    assert len(curve) == 1
    pt = curve[-1]
    assert pt["total_equity"] == pytest.approx(1100.0)  # 500 cash + 10*60
    assert pt["allowance_total"] == settings.sim_monthly_contribution


async def test_take_snapshot_skips_during_rebalance(mem_db, monkeypatch):
    """While a rebalance holds the lock, the daily mark must skip — the
    rebalance writes its own authoritative post-trade snapshot and a racing
    daily mark mid-execution would distort the curve."""

    lock_held = asyncio.Event()
    release = asyncio.Event()

    async def fake_locked_rebalance():
        # simulate the real run_rebalance: hold the lock while working
        async with monthly._rebalance_lock:
            lock_held.set()
            await release.wait()
        return {"rebalanced": True}

    task = asyncio.create_task(fake_locked_rebalance())
    await lock_held.wait()
    r = await monthly.take_snapshot()
    assert r.get("skipped") is True and "rebalance" in r["reason"]
    release.set()
    await task
    # after the lock frees, a snapshot is allowed again
    r2 = await monthly.take_snapshot()
    assert r2["snapshotted"] is True


async def test_refresh_holdings_no_rebalance(mem_db, monkeypatch):
    """refresh_holdings refreshes candles for held tickers (+ FX pairs) only —
    no deposits, no trades, no rebalance side effects."""
    from app.db import Candle

    async def fake_refresh(ticker, period):
        return []

    monkeypatch.setattr("app.market.refresh", fake_refresh)

    await monthly.deposit_allowance()
    await monthly._exec_buy("AAA", 50.0, 500.0, "test")
    val_before = await monthly.monthly_valuate()
    async with mem_db() as s:
        s.add(Candle(ticker="AAA", timestamp=datetime(2026, 9, 1, tzinfo=UTC),
                     open=60, high=60, low=60, close=60, volume=1000))
        await s.commit()

    out = await monthly.refresh_holdings()
    assert out["refreshed"] == ["AAA"] and out["errors"] == []
    val = await monthly.monthly_valuate()
    # valuation now prices AAA at the fresh 60 close (600) — refresh only
    # touched data, the position is unchanged
    assert len(val["positions"]) == 1
    assert val["positions"][0]["value"] == pytest.approx(600.0)
    assert val["cash"] == val_before["cash"]  # no deposit sneaked in
    assert val["allowance_total"] == settings.sim_monthly_contribution


async def test_daily_snapshot_loop_disabled_funds_nothing(mem_db, monkeypatch):
    """With sim_monthly_enabled=False the daily scheduler must not deposit,
    refresh or snapshot — the gate mirrors monthly.run_monthly_cycle()."""
    from app import sim as sim_mod

    calls: list[str] = []

    # Pin the clock to a normal weekday: the loop skips weekends/NYSE holidays,
    # so without this the test only passes Mon-Fri (CI failed on a weekend run).
    monkeypatch.setattr(sim_mod, "_utcnow", lambda: datetime(2026, 9, 21, 22, 30))

    async def fake_deposit():
        calls.append("deposit")
        return {"deposited": False}

    async def fake_refresh():
        calls.append("refresh")
        return {"refreshed": [], "errors": []}

    async def fake_snapshot():
        calls.append("snapshot")
        return {"snapshotted": True}

    monkeypatch.setattr(monthly, "deposit_allowance", fake_deposit)
    monkeypatch.setattr(monthly, "refresh_holdings", fake_refresh)
    monkeypatch.setattr(monthly, "take_snapshot", fake_snapshot)

    # Drive exactly two loop iterations: the first sleep returns so the loop
    # body runs once, the second raises the sentinel to end the task. (Raising
    # CancelledError from sleep would cancel the task BEFORE the body runs,
    # making the assertions vacuous.)
    class _TwoPasses(Exception):
        pass

    sleep_calls = {"n": 0}

    async def fake_sleep(_seconds):
        sleep_calls["n"] += 1
        if sleep_calls["n"] >= 2:
            raise _TwoPasses()

    monkeypatch.setattr(sim_mod.asyncio, "sleep", fake_sleep)

    async def run_two_passes():
        try:
            await sim_mod._daily_monthly_snapshot_loop()
        except _TwoPasses:
            pass

    # Disabled: the loop iterates but performs no deposits/refreshes/snapshots.
    monkeypatch.setattr(sim_mod.settings, "sim_monthly_enabled", False)
    await run_two_passes()
    assert calls == []
    assert sleep_calls["n"] == 2  # loop kept its schedule while disabled

    # Enabled: the body runs (deposit -> refresh -> snapshot).
    monkeypatch.setattr(sim_mod.settings, "sim_monthly_enabled", True)
    sleep_calls["n"] = 0  # reset pass counter for the second run
    await run_two_passes()
    assert calls == ["deposit", "refresh", "snapshot"]

# ---------------------------------------------------------------------------
# EDGAR source (app.edgar)
# ---------------------------------------------------------------------------

def _companyfacts_payload() -> dict:
    """Minimal companyfacts-shaped payload: NI with a restatement, shares
    under dei, equity incl. a fallback-tag hole, capex as negative."""
    return {
        "facts": {
            "us-gaap": {
                "StockholdersEquity": {"units": {"USD": [
                    {"start": None, "end": "2020-06-30", "filed": "2020-08-05", "val": 2e9},
                    {"start": None, "end": "2021-06-30", "filed": "2021-08-05", "val": 3e9},
                ]}},
                "NetIncomeLoss": {"units": {"USD": [
                    {"start": "2020-07-01", "end": "2021-06-30", "filed": "2021-08-05", "val": 5e8},
                ]}},
                # issuer switched tags in 2022 -> primary NI has a hole for FY2022
                "NetIncomeLossAvailableToCommonStockholdersBasic": {"units": {"USD": [
                    {"start": "2021-07-01", "end": "2022-06-30", "filed": "2022-08-04", "val": 6e8},
                ]}},
                "PaymentsToAcquirePropertyPlantAndEquipment": {"units": {"USD": [
                    {"start": "2021-07-01", "end": "2022-06-30", "filed": "2022-08-04", "val": -1e8},
                ]}},
                "NetCashProvidedByUsedInOperatingActivities": {"units": {"USD": [
                    {"start": "2021-07-01", "end": "2022-06-30", "filed": "2022-08-04", "val": 9e8},
                ]}},
            },
            "dei": {
                "EntityCommonStockSharesOutstanding": {"units": {"shares": [
                    {"start": None, "end": "2021-06-30", "filed": "2021-08-05", "val": 1e8},
                ]}},
            },
        }
    }


def test_parse_companyfacts_incl_fallbacks_and_dei():
    from app.edgar import _parse_companyfacts
    rec = _parse_companyfacts(_companyfacts_payload())
    # primary tags present
    assert "NetIncomeLoss" in rec and "StockholdersEquity" in rec
    # dei shares mapped under the us-gaap-style name the strategy reads
    assert rec["EntityCommonStockSharesOutstanding"][0]["val"] == 1e8
    # fallback gap fill: NI now includes the FY2022 period from the alt tag
    ni_ends = {e["end"] for e in rec["NetIncomeLoss"]}
    assert "2022-06-30" in ni_ends
    # fallback alt tags are consumed (not left as separate entries)
    assert "NetIncomeLossAvailableToCommonStockholdersBasic" not in rec


def test_edgar_facts_are_point_in_time_via_ttm():
    """End-to-end: parse -> store shape -> TTM as of a date sees only public facts."""
    from app import monthly
    from app.edgar import _parse_companyfacts
    rec = _parse_companyfacts(_companyfacts_payload())
    d_late = pd.Timestamp("2022-09-30")   # FY2022 filed 2022-08-04 => public
    assert monthly._ttm_as_of(rec["NetIncomeLoss"], d_late) == 6e8
    d_early = pd.Timestamp("2022-07-01")  # before the FY2022 filing
    assert monthly._ttm_as_of(rec["NetIncomeLoss"], d_early) == 5e8
    # ROE input: equity instant fact as of date
    eq = monthly._latest_as_of(rec["StockholdersEquity"], pd.Timestamp("2021-09-01"))
    assert eq == ("2021-06-30", 3e9)


async def test_load_prefers_edgar_over_yfinance(mem_db):
    """When both sources exist for a ticker, the loader must serve EDGAR only."""
    from app import fundamentals as fund_mod
    from app.db import Fundamental
    async with mem_db() as s:
        s.add(Fundamental(ticker="EEE", tag="StockholdersEquity", start=None,
                          end="2024-06-30", filed="2024-08-14", val=5e9,
                          currency="USD", source="yfinance"))
        s.add(Fundamental(ticker="EEE", tag="StockholdersEquity", start=None,
                          end="2024-06-30", filed="2024-07-30", val=4e9,
                          currency="USD", source="edgar"))
        # yfinance-only ticker keeps working
        s.add(Fundamental(ticker="FFF", tag="StockholdersEquity", start=None,
                          end="2024-06-30", filed="2024-08-14", val=7e9,
                          currency="USD", source="yfinance"))
        await s.commit()
    out = await fund_mod.load_fundamentals(["EEE", "FFF"])
    assert out["EEE"]["StockholdersEquity"] == [
        {"start": None, "end": "2024-06-30", "filed": "2024-07-30", "val": 4e9}]
    assert out["FFF"]["StockholdersEquity"][0]["val"] == 7e9


async def test_ensure_cik_map_uses_cache_and_handles_unknown(mem_db, monkeypatch):
    from app import edgar
    from app.db import SecCik
    # pre-seed the cache: AAA known, BBB known-no-CIK
    async with mem_db() as s:
        s.add(SecCik(ticker="AAA", cik=123456))
        s.add(SecCik(ticker="BBB", cik=0))
        await s.commit()
    calls = {"n": 0}

    def blocked(url):
        calls["n"] += 1
        raise urllib.error.HTTPError(url, 403, "Forbidden", None, None)  # type: ignore[arg-type]

    monkeypatch.setattr(edgar, "_http_get", blocked)

    # cached tickers resolve with no network hit
    m = await edgar.ensure_cik_map(["AAA", "BBB"])
    assert m == {"AAA": 123456, "BBB": 0}
    assert calls["n"] == 0

    # no contact email: the map download is skipped entirely, unknown → 0
    monkeypatch.setattr(settings, "sec_contact_email", "")
    m2 = await edgar.ensure_cik_map(["AAA", "CCC"])
    assert m2 == {"AAA": 123456, "CCC": 0}
    assert calls["n"] == 0

    # with a contact email one attempt is made; a blocked download degrades
    monkeypatch.setattr(settings, "sec_contact_email", "ops@example.com")
    m3 = await edgar.ensure_cik_map(["AAA", "DDD"])
    assert m3 == {"AAA": 123456, "DDD": 0}
    assert calls["n"] == 1

    # unknown tickers are never persisted as a bogus no-CIK 0 row
    async with mem_db() as s:
        rows = (await s.scalars(select(SecCik))).all()
        assert sorted(r.ticker for r in rows) == ["AAA", "BBB"]


# ---------------------------------------------------------------------------
# Regression tests: refresh scheduling, 404 classification, NULL-start upsert
# ---------------------------------------------------------------------------

async def test_instant_fact_upsert_dedupes(mem_db):
    """Instant facts (start=None) must upsert, not insert duplicates: NULLs
    are distinct in SQLite UNIQUE constraints, so the start column stores ''
    for instants (OptionalDateStr) and the conflict target matches."""
    from sqlalchemy.dialects.sqlite import insert as sqlite_insert

    from app.db import Fundamental

    async with mem_db() as s:
        for _ in range(2):  # same fact inserted twice — second must update
            stmt = sqlite_insert(Fundamental).values(
                ticker="QQQ", tag="StockholdersEquity", start=None,
                end="2024-06-30", filed="2024-08-14", val=5e9,
                currency="USD", source="edgar")
            stmt = stmt.on_conflict_do_update(
                index_elements=["ticker", "tag", "start", "end", "filed", "source"],
                set_={"val": stmt.excluded.val, "updated_at": stmt.excluded.updated_at})
            await s.execute(stmt)
        await s.commit()
        rows = (await s.scalars(select(Fundamental).where(Fundamental.ticker == "QQQ"))).all()
        assert len(rows) == 1
        assert rows[0].start is None  # '' round-trips as None
        assert rows[0].val == 5e9


async def test_fetch_companyfacts_404_vs_timeout(mem_db, monkeypatch):
    """A real HTTP 404 means no companyfacts; a network failure must not be
    misclassified as 404 just because the CIK contains '404' in its URL."""
    from app import edgar

    def get_404(url):
        raise urllib.error.HTTPError(url, 404, "Not Found", None, None)  # type: ignore[arg-type]

    monkeypatch.setattr(edgar, "_http_get", get_404)
    assert edgar.fetch_companyfacts(4040) == {}  # CIK contains "404" but IS a 404

    def get_timeout(url):
        raise TimeoutError("timed out")

    monkeypatch.setattr(edgar, "_http_get", get_timeout)
    with pytest.raises(TimeoutError):  # not swallowed as no-companyfacts
        edgar.fetch_companyfacts(4040)


def test_edgar_user_agent_includes_optional_contact(monkeypatch):
    """The configured SEC_CONTACT_EMAIL appears in the UA; with none set the UA
    still identifies the app (no empty parentheses). The CIK-map download then
    403s and degrades to the cached map — see ensure_cik_map."""
    from app import edgar

    monkeypatch.setattr(settings, "sec_contact_email", "ops@example.com")
    assert "ops@example.com" in edgar._user_agent()

    monkeypatch.setattr(settings, "sec_contact_email", "")
    ua = edgar._user_agent()
    assert "trade-sentinel" in ua and "()" not in ua


async def test_refresh_edgar_staleness_gate(mem_db, monkeypatch):
    """refresh_edgar must refetch tickers whose last refresh is older than
    STALE_AFTER_DAYS — the staleness cutoff exists precisely so quarterly
    filings are picked up; only fresh tickers are skipped."""
    from datetime import datetime, timedelta

    from app import edgar
    from app.db import Fundamental, SecCik

    now = datetime.now(UTC)
    async with mem_db() as s:
        s.add(SecCik(ticker="STALE", cik=111))
        s.add(SecCik(ticker="FRESH", cik=222))
        s.add(Fundamental(ticker="STALE", tag="StockholdersEquity", start=None,
                          end="2024-06-30", filed="2024-08-14", val=1e9,
                          currency="USD", source="edgar",
                          updated_at=now - timedelta(days=edgar.STALE_AFTER_DAYS + 10)))
        s.add(Fundamental(ticker="FRESH", tag="StockholdersEquity", start=None,
                          end="2024-06-30", filed="2024-08-14", val=1e9,
                          currency="USD", source="edgar",
                          updated_at=now - timedelta(days=1)))
        await s.commit()

    def fake_fetch(cik):
        return {"StockholdersEquity": [{"start": None, "end": "2024-06-30",
                                        "filed": "2024-08-14", "val": 2e9}]}

    monkeypatch.setattr(edgar, "fetch_companyfacts", fake_fetch)
    monkeypatch.setattr(edgar, "_http_get", lambda url: (_ for _ in ()).throw(
        AssertionError("CIK map download should not be needed")))
    monkeypatch.setattr(edgar.asyncio, "sleep", _no_sleep)

    statuses = await edgar.refresh_edgar(["STALE", "FRESH"])
    assert statuses["STALE"].startswith("ok(")  # stale -> refetched
    assert statuses["FRESH"] == "skipped"       # fresh -> skipped


async def _no_sleep(_):
    return None


async def test_price_usd_converts_foreign_listings(mem_db):
    """_price/_price_usd_map must convert non-USD closes to USD at the latest
    FX rate — the strategy decides on USD-converted closes, so booking the
    raw local-currency close as USD would misstate exposure ~1/FX."""
    from app.db import Candle

    dates = pd.bdate_range(end=pd.Timestamp.now().normalize(), periods=5)
    async with mem_db() as s:
        for d in dates[:-1]:  # EURUSD pair stops updating one day before...
            s.add(Candle(ticker="EURUSD=X", timestamp=d.to_pydatetime(),
                         open=1.1, high=1.1, low=1.1, close=1.1, volume=0))
        s.add(Candle(ticker="EURUSD=X", timestamp=dates[-1].to_pydatetime(),
                     open=1.2, high=1.2, low=1.2, close=1.2, volume=0))
        s.add(Candle(ticker="SAP.DE", timestamp=dates[-1].to_pydatetime(),
                     open=100, high=100, low=100, close=100, volume=1000))
        s.add(Candle(ticker="AAPL", timestamp=dates[-1].to_pydatetime(),
                     open=200, high=200, low=200, close=200, volume=1000))
        # USDCHF divides (local units per USD)
        s.add(Candle(ticker="USDCHF=X", timestamp=dates[-1].to_pydatetime(),
                     open=0.8, high=0.8, low=0.8, close=0.8, volume=0))
        s.add(Candle(ticker="NOVN.SW", timestamp=dates[-1].to_pydatetime(),
                     open=80, high=80, low=80, close=80, volume=1000))
        await s.commit()

    out = await monthly._price_usd_map(["SAP.DE", "AAPL", "NOVN.SW"])
    assert out["SAP.DE"] == pytest.approx(100 * 1.2)  # EUR close * EURUSD
    assert out["AAPL"] == pytest.approx(200.0)         # USD: unchanged
    assert out["NOVN.SW"] == pytest.approx(80 / 0.8)   # CHF close / USDCHF
    # single-ticker helper agrees
    assert await monthly._price("SAP.DE") == pytest.approx(120.0)


async def test_run_rebalance_lock_serializes(mem_db, monkeypatch):
    """Concurrent run_rebalance calls: the second must bail out instead of
    passing the month-idempotence gate while the first is mid-refresh (which
    would double-deposit the allowance and duplicate trades)."""

    in_refresh = asyncio.Event()
    release = asyncio.Event()

    async def fake_refresh_data(tickers_list):
        in_refresh.set()
        await release.wait()
        return [], {}

    monkeypatch.setattr(monthly, "refresh_data", fake_refresh_data)

    async def fake_pending(now):
        return "2026-08", pd.Timestamp("2026-08-31")

    monkeypatch.setattr(monthly, "_pending_rebuild", fake_pending)

    t1 = asyncio.create_task(monthly.run_rebalance())
    await in_refresh.wait()          # t1 is inside the locked region
    t2 = await monthly.run_rebalance()
    assert t2.get("skipped") is True and "already running" in t2["reason"]
    release.set()
    r1 = await t1
    assert r1.get("rebalanced") is True or r1.get("skipped") is True


# ---------------------------------------------------------------------------
# Deferred-fill cadence (LIVE engine, session fill contract rule 2/5):
# the month-end tick is the DECISION; the fill lands on the FIRST session
# after the month-end at THAT session's strict exact-date OPEN; a leg
# without an executable quote retries at the following sessions; a newer
# month-end supersedes a still-pending decision.
# ---------------------------------------------------------------------------

def _const_bars(start: str, end: str, o: float, c: float,
                drop: tuple[str, ...] = ()) -> list[tuple[str, float, float]]:
    """(YYYY-MM-DD, open, close) rows at a constant price over the business
    days of [start, end], minus `drop` dates (candle holes)."""
    return [(d.strftime("%Y-%m-%d"), o, c) for d in pd.bdate_range(start, end)
            if d.strftime("%Y-%m-%d") not in drop]


async def _seed_candle_bars(bars: dict[str, list[tuple[str, float, float]]]):
    """Seed Candle rows for bars[ticker] = [(date, open, close), ...]."""
    from app.db import Candle
    async with monthly.Session() as s:
        for t, series in bars.items():
            for d, o, c in series:
                s.add(Candle(ticker=t, timestamp=pd.Timestamp(d).to_pydatetime(),
                             open=o, high=max(o, c), low=min(o, c), close=c,
                             volume=1e6))
        await s.commit()


async def _seed_roe_fundamentals(quality: dict[str, float]):
    """Point-in-time facts public well before the decision month-ends:
    equity 1e9, 12-month NI = quality x 1e9 (ROE = quality), 1e8 shares.
    No OCF/capex on purpose: P/FCF is missing everywhere, so the score is
    the degenerate roe+mom path and the ranking is quality-determined.
    filed 2025-07-15 < the July month-end, so both frames see the same facts."""
    from app.db import Fundamental
    async with monthly.Session() as s:
        for t, q in quality.items():
            s.add(Fundamental(ticker=t, tag="StockholdersEquity", start=None,
                              end="2025-06-30", filed="2025-07-15", val=1e9,
                              currency="USD"))
            s.add(Fundamental(ticker=t, tag="NetIncomeLoss", start="2024-07-01",
                              end="2025-06-30", filed="2025-07-15",
                              val=1e9 * q, currency="USD"))
            s.add(Fundamental(ticker=t, tag="CommonStockSharesOutstanding",
                              start=None, end="2025-06-30", filed="2025-07-15",
                              val=1e8, currency="shares"))
        await s.commit()


def _deferred_env(mem_db, monkeypatch, bars, quality):
    """Common patches for the deferred-cadence tests: the fixture universe is
    the two tape tickers, data refresh is a no-op, and the operator-local
    month tracks the test's clock holder."""
    monkeypatch.setattr(monthly, "universe_tickers", lambda _u: list(bars))
    monkeypatch.setattr(monthly, "refresh_data", fake_refresh_noop)
    clock = {"month": "2025-08"}
    monkeypatch.setattr(monthly, "_current_month", lambda: clock["month"])
    return clock


async def fake_refresh_noop(_tickers):
    return [], {}


class TestDeferredFillCadence:
    async def test_decision_fills_first_session_after_month_end_at_that_open(
            self, mem_db, monkeypatch):
        """The July month-end (2025-07-31) is the decision; its fills land on
        the FIRST session after it (2025-08-01) at THAT session's OPEN. AAA's
        tape makes all three price anchors distinct: July-ME close 100 (the
        stale-close trap), fill-session open 110, fill-session close 120 —
        the booked fill price must be 110."""
        from app.db import MonthlyRebalance as Rb
        from app.db import MonthlyTrade as Tr

        bars = {
            "AAA": _const_bars("2024-06-03", "2025-07-31", 100.0, 100.0)
                   + _const_bars("2025-08-01", "2025-09-30", 110.0, 120.0),
            "BBB": _const_bars("2024-06-03", "2025-09-30", 100.0, 100.0),
        }
        await _seed_candle_bars(bars)
        await _seed_roe_fundamentals({"AAA": 0.5, "BBB": 0.1})
        _deferred_env(mem_db, monkeypatch, bars, None)

        r1 = await monthly.run_rebalance(
            now=datetime(2025, 8, 1, 23, 30, tzinfo=UTC))
        assert r1.get("rebalanced") is True
        assert r1["month"] == "2025-07"          # the DECISION month...
        assert r1["decision_date"] == "2025-07-31"   # ...decided at its ME
        assert r1["picks"] == ["AAA", "BBB"]

        async def check():
            async with mem_db() as s:
                trs = (await s.scalars(
                    select(Tr).order_by(Tr.created_at, Tr.ticker))).all()
                assert [(t.ticker, t.side, t.price) for t in trs] == \
                    [("AAA", "BUY", 110.0), ("BBB", "BUY", 100.0)]
                # (live trades stamp the real write time in created_at; the
                # FILL SESSION is pinned by the strict-open price: 110 is
                # the 2025-08-01 OPEN — no other session quotes it)
                rbs = (await s.scalars(select(Rb))).all()
                assert len(rbs) == 1
                # the audit row keeps the DECISION day distinct from the fill day
                assert rbs[0].rebal_date.strftime("%Y-%m-%d") == "2025-07-31"
                acc = await s.get(MonthlyAccount, 1)
                assert acc.last_rebalance_month == "2025-07"  # pass complete
        await check()

        # idempotent: the decision is complete, a second cycle is a no-op
        r2 = await monthly.run_rebalance(
            now=datetime(2025, 8, 1, 23, 30, tzinfo=UTC))
        assert r2.get("skipped") is True

    async def test_leg_without_a_quote_retries_next_session(
            self, mem_db, monkeypatch):
        """BBB has NO candle on the fill session (2025-08-01): its leg gets
        NO fill (a missing quote is not an executable price — never a stale
        or forward-filled close), the decision stays pending, and the leg is
        retried on the NEXT session (2025-08-04) at THAT session's open (90
        — not the month-end's 100 close)."""
        from app.db import MonthlyRebalance as Rb
        from app.db import MonthlyTrade as Tr

        bars = {
            "AAA": _const_bars("2024-06-03", "2025-09-30", 100.0, 100.0),
            "BBB": _const_bars("2024-06-03", "2025-07-31", 100.0, 100.0)
                   + _const_bars("2025-08-04", "2025-09-30", 90.0, 90.0),
        }
        await _seed_candle_bars(bars)
        await _seed_roe_fundamentals({"AAA": 0.5, "BBB": 0.1})
        _deferred_env(mem_db, monkeypatch, bars, None)

        r1 = await monthly.run_rebalance(
            now=datetime(2025, 8, 1, 23, 30, tzinfo=UTC))
        assert r1.get("rebalanced") is True
        assert r1["complete"] is False          # a leg is still pending
        assert [t["ticker"] for t in r1["trades"]] == ["AAA"]  # BBB got no fill

        async def check1():
            async with mem_db() as s:
                acc = await s.get(MonthlyAccount, 1)
                # the decision is NOT marked complete while a leg is open
                assert acc.last_rebalance_month is None
                trs = (await s.scalars(
                    select(Tr).order_by(Tr.created_at, Tr.ticker))).all()
                assert [(t.ticker, t.side, t.price) for t in trs] == \
                    [("AAA", "BUY", 100.0)]
        await check1()

        # catch-up: the next session re-takes the pending decision
        r2 = await monthly.run_rebalance(
            now=datetime(2025, 8, 4, 23, 30, tzinfo=UTC))
        assert r2.get("rebalanced") is True
        assert r2["complete"] is True

        async def check2():
            async with mem_db() as s:
                trs = (await s.scalars(
                    select(Tr).order_by(Tr.created_at, Tr.ticker))).all()
                # AAA filled on the 08-01 session; BBB's retry filled on the
                # 08-04 session at THAT session's strict open (90 — the
                # month-end's close 100 was never an executable price)
                assert [(t.ticker, t.side, t.price) for t in trs] == \
                    [("AAA", "BUY", 100.0), ("BBB", "BUY", 90.0)]
                # the retry re-used the first-pass audit row — no duplicate
                rbs = (await s.scalars(select(Rb))).all()
                assert len(rbs) == 1
                assert rbs[0].rebal_date.strftime("%Y-%m-%d") == "2025-07-31"
                acc = await s.get(MonthlyAccount, 1)
                assert acc.last_rebalance_month == "2025-07"
        await check2()

    async def test_newer_month_end_supersedes_a_pending_decision(
            self, mem_db, monkeypatch):
        """AAA's tape is holed until 2025-08-29, so the July decision's AAA
        leg stays pending. When the August month-end (2025-08-29) completes,
        the NEXT session (2025-09-02, after Labor Day) re-resolves the
        pending rebuild to the NEWER month-end: the decision is August's
        (rebal_date 2025-08-29), and the stuck July leg fills under it at the
        CURRENT session's strict open (130 — not the Aug-29 open 140 and not
        the July-ME close 100)."""
        from app.db import MonthlyRebalance as Rb
        from app.db import MonthlyTrade as Tr

        bars = {
            "AAA": _const_bars("2024-06-03", "2025-07-31", 100.0, 100.0)
                   + _const_bars("2025-08-29", "2025-08-29", 140.0, 140.0)
                   + _const_bars("2025-09-02", "2025-09-02", 130.0, 130.0),
            "BBB": _const_bars("2024-06-03", "2025-09-02", 100.0, 100.0),
        }
        await _seed_candle_bars(bars)
        await _seed_roe_fundamentals({"AAA": 0.5, "BBB": 0.1})
        clock = _deferred_env(mem_db, monkeypatch, bars, None)

        r1 = await monthly.run_rebalance(
            now=datetime(2025, 8, 1, 23, 30, tzinfo=UTC))
        assert r1.get("rebalanced") is True
        assert r1["complete"] is False          # AAA's leg: no quote on 08-01
        assert [t["ticker"] for t in r1["trades"]] == ["BBB"]

        # September begins: the newest completed month-end now ranks
        clock["month"] = "2025-09"
        r2 = await monthly.run_rebalance(
            now=datetime(2025, 9, 2, 23, 30, tzinfo=UTC))
        assert r2.get("rebalanced") is True
        # the PENDING decision was superseded by the newer month-end
        assert r2["month"] == "2025-08"
        assert r2["decision_date"] == "2025-08-29"

        async def check():
            async with mem_db() as s:
                rbs = (await s.scalars(
                    select(Rb).order_by(Rb.rebal_month))).all()
                assert [(r.rebal_month, r.rebal_date.strftime("%Y-%m-%d"))
                        for r in rbs] == \
                    [("2025-07", "2025-07-31"), ("2025-08", "2025-08-29")]
                trs = (await s.scalars(
                    select(Tr).order_by(Tr.created_at, Tr.ticker))).all()
                buys = [(t.ticker, t.side, t.price) for t in trs]
                # BBB filled under the July decision (100 = its open);
                # AAA's stuck July leg filled under the SUPERSEDING August
                # decision at the 09-02 session's strict open (130 — not the
                # 08-29 open 140 and not the July-ME close 100)
                assert ("BBB", "BUY", 100.0) in buys
                assert ("AAA", "BUY", 130.0) in buys
                assert not any(price == 140.0 for _, _, price in buys)
                acc = await s.get(MonthlyAccount, 1)
                assert acc.last_rebalance_month == "2025-08"
        await check()


async def test_refresh_data_reuses_the_nightly_prefetch(monkeypatch):
    """The rebalance must go through refresh_many with the shared freshness
    window, so it reuses the nightly prefetch/DB instead of re-pulling 10y of
    candles per ticker from Yahoo. Regression guard for the reported
    "monthly rebalance refreshes 10y and ignores the DB"."""
    from app import edgar, market

    seen: dict = {}

    async def fake_refresh_many(tickers, period, **kwargs):
        seen["tickers"] = list(tickers)
        seen["period"] = period
        seen["kwargs"] = kwargs
        return list(tickers), []

    async def fake_fundamentals(**_kwargs):
        return {}

    monkeypatch.setattr(market, "refresh_many", fake_refresh_many)
    monkeypatch.setattr(edgar, "refresh_universe_mixed", fake_fundamentals)
    monkeypatch.setattr(settings, "sim_monthly_fundamentals_source", "edgar")

    _, fund_status = await monthly.refresh_data(["AAA", "BBB"])

    assert seen["tickers"] == ["AAA", "BBB"]
    assert seen["period"] == "2y"
    assert seen["kwargs"].get("max_age_seconds") == settings.market_fresh_seconds
    assert fund_status == {}


# ---------------------------------------------------------------------------
# Risk overlays: residual momentum + low-vol tilt
# ---------------------------------------------------------------------------

def _trend_close(n_days: int = 400) -> pd.DataFrame:
    """Deterministic 5-ticker frame: LEAD (strong smooth uptrend), CHOP (same
    endpoint as LEAD but 5x the noise), WEAK (flat), LAG (downtrend), CRASH
    (uptrend with a mid-window -30% crash). Residual momentum must rank the
    smooth trend above the choppy one even at identical drift."""
    rng = np.random.default_rng(7)
    dates = pd.bdate_range("2024-06-03", periods=n_days)
    drift = 0.0009
    lead = 100.0 * np.exp(np.cumsum(rng.normal(drift, 0.003, n_days)))
    # CHOP: same drift but much noisier path
    chop = 100.0 * np.exp(np.cumsum(rng.normal(drift, 0.018, n_days)))
    weak = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.006, n_days)))
    lag = 100.0 * np.exp(np.cumsum(rng.normal(-drift, 0.006, n_days)))
    crash = 100.0 * np.exp(np.cumsum(rng.normal(drift, 0.004, n_days)))
    crash[n_days // 2] = crash[n_days // 2] * 0.70  # one-day -30% idio crash
    return pd.DataFrame({"LEAD": lead, "CHOP": chop, "WEAK": weak,
                         "LAG": lag, "CRASH": crash}, index=dates)


def test_residual_momentum_rewards_smooth_uptrend():
    """Blitz-Huij-Martens: momentum per unit of idiosyncratic vol. LEAD and
    CHOP share the same drift, but LEAD's path is cleaner, so its
    residual-momentum score must be higher. The crashed name is penalized;
    the downtrending name is last."""
    close = _trend_close()
    rm = monthly.residual_momentum(close, close.index[-1])
    assert rm["LEAD"] > rm["CHOP"]
    assert rm["LEAD"] > 0
    assert rm["LAG"] == rm.min()


def test_residual_momentum_short_history_nan():
    close = _trend_close(n_days=100)  # < 252+21 warmup
    rm = monthly.residual_momentum(close, close.index[-1])
    assert rm.isna().all()


def test_eligible_frame_residual_variant(monkeypatch):
    """The residual variant fills the same `mom` column so downstream
    scoring/eligibility work unchanged; the -0.99 raw-mom floor must not
    kick in (residual mom is a t-stat, not a return)."""
    monkeypatch.setattr(settings, "sim_daily_core_mom_variant", "residual")
    close = _trend_close()
    vol = pd.DataFrame(5e6, index=close.index, columns=close.columns)
    fund = _fund_all_positive(close.columns)
    frame = monthly.eligible_frame(close.index[-1], close, vol, fund)
    monkeypatch.setattr(settings, "sim_daily_core_mom_variant", "raw")
    assert frame is not None
    assert np.isfinite(frame["mom"]).all()
    assert float(frame["mom"]["LEAD"]) > float(frame["mom"]["CHOP"])


def test_qv_order_lowvol_tilt():
    """With the tilt on, the low-vol name gains one full rank point; the
    explicit tie-breakers still resolve identical scores."""
    rows = {
        "HIVOL": {"roe": 0.10, "p_fcf": 20.0, "mcap": 1e10, "dollar_vol": 5e7,
                  "mom": 0.10, "price": 100, "vol": 0.60},
        "LOVOL": {"roe": 0.10, "p_fcf": 20.0, "mcap": 1e10, "dollar_vol": 5e7,
                  "mom": 0.10, "price": 100, "vol": 0.15},
    }
    frame = pd.DataFrame(rows).T
    frame["eligible"] = True
    _e, order_plain = monthly._qv_order(frame, lowvol_tilt=False)
    _e2, order_tilt = monthly._qv_order(frame, lowvol_tilt=True)
    # identical fundamentals: the tilt is the only difference and must
    # promote the low-vol name from the ticker-asc fallback.
    assert order_plain == ["HIVOL", "LOVOL"]  # tie -> ticker asc
    assert order_tilt == ["LOVOL", "HIVOL"]


# ---------------------------------------------------------------------------
# Backfill (synthetic history)
# ---------------------------------------------------------------------------

@pytest.fixture
def fake_market(monkeypatch):
    """Patch monthly's market/fundamentals loaders so backfill replays a tiny
    deterministic two-ticker market (no network, no real DB rows)."""
    # Two COMPLETE months: 2026-07-31 and 2026-08-31 are the real US session
    # month-ends, so the replay rebalances on scheduled month-ends — a frame
    # ending inside a running month has none (see monthly.completed_month_ends).
    # Under the deferred-fill cadence the July decision's fills land on the
    # FIRST session after the month-end (2026-08-03) at that session's open;
    # August's decision fills on 2026-09-01 — OUTSIDE the replayed window, so
    # a window ending on the August month-end leaves August pending.
    dates = pd.bdate_range("2026-07-01", "2026-08-31")
    close = pd.DataFrame({"AAA": 100.0, "BBB": 50.0}, index=dates)
    open_px = pd.DataFrame({"AAA": 100.0, "BBB": 50.0}, index=dates)
    vol = pd.DataFrame(1e6, index=dates, columns=["AAA", "BBB"])

    def fake_frame(m, c, v, f):
        return pd.DataFrame({
            "roe": [0.1, 0.2], "p_fcf": [20.0, 15.0], "mcap": [1e10, 1e10],
            "dollar_vol": [5e7, 5e7], "mom": [0.1, 0.2], "price": [100.0, 50.0],
            "vol": [0.2, 0.25], "eligible": [True, True],
        }, index=pd.Index(["AAA", "BBB"], name="ticker"))

    async def fake_load_ohlcv_frames(_tickers, _asof, start=None):
        return open_px, close, vol

    async def fake_load_fundamentals(_tickers):
        return {"AAA": {"x": []}, "BBB": {"x": []}}

    monkeypatch.setattr(monthly, "universe_tickers", lambda _u: ["AAA", "BBB"])
    monkeypatch.setattr(monthly.fundamentals_mod, "load_fundamentals",
                        fake_load_fundamentals)
    monkeypatch.setattr(monthly, "load_ohlcv_frames", fake_load_ohlcv_frames)
    monkeypatch.setattr(monthly, "eligible_frame", fake_frame)
    return {"close": close}


class TestBackfill:
    def test_replaces_state_and_persists_rows(self, mem_db, fake_market, monkeypatch):
        """The replay wipes the portfolio and persists allowance/trade/
        rebalance/snapshot rows with historical stamps, converging to the
        picks the strategy would hold today."""
        from app.db import MonthlyAllowance as Al
        from app.db import MonthlyPosition as Pos
        from app.db import MonthlyRebalance as Rb
        from app.db import MonthlySnapshot as Sn
        from app.db import MonthlyTrade as Tr

        monkeypatch.setattr(monthly, "_current_month", lambda: "2026-08")
        r = asyncio.run(monthly.backfill(start="2026-07-01"))
        assert r["ok"] is True
        assert r["contributed"] == pytest.approx(2 * settings.sim_monthly_contribution)

        async def check():
            async with mem_db() as s:
                snaps = (await s.scalars(select(Sn))).all()
                assert len(snaps) == r["snapshots"] > 0
                # one snapshot per trading day, dated by the replay day
                days = sorted(x.created_at.strftime("%Y-%m-%d") for x in snaps)
                assert days[0] == "2026-07-01" and days[-1] == "2026-08-31"
                # allowance rows: exactly the deposited months
                months = sorted(x.month for x in (await s.scalars(select(Al))).all())
                assert months == ["2026-07", "2026-08"]
                # the end state holds the strategy's picks
                pos = (await s.scalars(select(Pos))).all()
                assert len(pos) == 2
                for p in pos:
                    assert p.shares > 0 and p.avg_cost > 0
                # rebalance audit rows: one row per decision whose FILLS
                # completed inside the window — July's decision fills on
                # 2026-08-03; August's would fill 2026-09-01 (outside the
                # window), so August stays pending under the deferred cadence
                rbs = (await s.scalars(select(Rb))).all()
                assert sorted(x.rebal_month for x in rbs) == ["2026-07"]
                assert all(x.picked for x in rbs)
                # trades carry the replay day
                trs = (await s.scalars(select(Tr))).all()
                for t in trs:
                    assert t.created_at.strftime("%Y-%m") in ("2026-07", "2026-08")
                    assert t.shares > 0 and t.price > 0
        asyncio.run(check())

    def test_funds_months_without_an_eligible_frame(self, mem_db, fake_market,
                                                    monkeypatch):
        """A window month with no eligible ranking is still funded (the live
        engine deposits before the frame check); only its rebalance is skipped."""
        import pandas as pd

        from app.db import MonthlyAllowance as Al
        from app.db import MonthlyRebalance as Rb

        real = monthly.eligible_frame

        def frame_with_gap(m, c, v, f):
            fr = real(m, c, v, f)
            if fr is not None and pd.Timestamp(m).month == 8:
                fr = fr.assign(eligible=False)
            return fr
        monkeypatch.setattr(monthly, "eligible_frame", frame_with_gap)
        monkeypatch.setattr(monthly, "_current_month", lambda: "2026-09")

        r = asyncio.run(monthly.backfill(start="2026-07-01"))
        assert r["ok"] is True
        # Jul + Aug (Aug has no ranking but is still funded)
        assert r["contributed"] == pytest.approx(2 * settings.sim_monthly_contribution)

        async def check():
            async with mem_db() as s:
                months = sorted(x.month for x in (await s.scalars(select(Al))).all())
                assert months == ["2026-07", "2026-08"]
                rows = (await s.scalars(select(Rb))).all()
                assert sorted(x.rebal_month for x in rows) == ["2026-07"]
        asyncio.run(check())

    def test_uncovered_current_month_keeps_its_contribution(self, mem_db, fake_market,
                                                            monkeypatch):
        """A backfill run before the current month has candles must not lose
        the live deposit (same rule as daily_core.backfill)."""
        from app.db import MonthlyAllowance as Al

        monkeypatch.setattr(monthly, "_current_month", lambda: "2026-09")

        async def seed():
            async with mem_db() as s:
                acc = await s.get(MonthlyAccount, 1)
                acc.last_allowance_month = "2026-09"
                await s.commit()
        asyncio.run(seed())

        r = asyncio.run(monthly.backfill(start="2026-07-01"))
        assert r["ok"] is True
        # Jul + Aug replayed + Sep (live, uncovered) = 3 contributions
        assert r["contributed"] == pytest.approx(3 * settings.sim_monthly_contribution)

        async def check():
            async with mem_db() as s:
                acc = await s.get(MonthlyAccount, 1)
                assert acc.last_allowance_month == "2026-09"
                rows = (await s.scalars(select(Al))).all()
                assert sorted(x.month for x in rows) == \
                    ["2026-07", "2026-08", "2026-09"]
        asyncio.run(check())

    def test_backfill_is_fee_free_like_the_live_engine(self, mem_db, fake_market,
                                                       monkeypatch):
        """The replay mirrors the live fee-free engine: with constant fixture
        prices the end state's value equals the contributed total, with no
        phantom 10 bps cost drag. Regression guard for the daily-core 999."""
        from app.db import MonthlyPosition as Pos

        monkeypatch.setattr(monthly, "_current_month", lambda: "2026-08")
        r = asyncio.run(monthly.backfill(start="2026-07-01"))
        assert r["ok"] is True

        async def check():
            async with mem_db() as s:
                acc = await s.get(MonthlyAccount, 1)
                pos = (await s.scalars(select(Pos))).all()
                value = acc.cash + sum(p.shares * p.avg_cost for p in pos)
                assert value == pytest.approx(r["contributed"], abs=1e-6)
        asyncio.run(check())

    def test_partial_month_is_walked_but_never_rebalanced(self, mem_db, fake_market,
                                                          monkeypatch):
        """A window ending INSIDE a running month (e.g. Oct-21's partial
        candle vs the actual Oct-30 month-end) is walked — funded and
        snapshotted — but NEVER rebalanced: no partial month-end is treated
        as a completed month-end (persisted-backfill boundary fix)."""
        from app.db import MonthlyRebalance as Rb
        from app.db import MonthlySnapshot as Sn

        # truncate the market to a mid-August last candle
        truncated = fake_market["close"].loc[:"2026-08-21"]

        async def fake_load_ohlcv(_tickers, _asof, start=None):
            return (pd.DataFrame({"AAA": 100.0, "BBB": 50.0}, index=truncated.index),
                    truncated,
                    pd.DataFrame(1e6, index=truncated.index, columns=["AAA", "BBB"]))

        monkeypatch.setattr(monthly, "load_ohlcv_frames", fake_load_ohlcv)
        monkeypatch.setattr(monthly, "_current_month", lambda: "2026-08")

        r = asyncio.run(monthly.backfill(start="2026-07-01"))
        assert r["ok"] is True
        # Jul + Aug funded (the partial month still receives the deposit)
        assert r["contributed"] == pytest.approx(2 * settings.sim_monthly_contribution)

        async def check():
            async with mem_db() as s:
                rbs = (await s.scalars(select(Rb))).all()
                assert sorted(x.rebal_month for x in rbs) == ["2026-07"]
                snaps = (await s.scalars(select(Sn))).all()
                last = max(x.created_at for x in snaps)
                assert last.strftime("%Y-%m-%d") == "2026-08-21"
        asyncio.run(check())

    def test_interior_month_end_hole_funds_months_in_order(
            self, mem_db, monkeypatch):
        """A month whose true calendar-end candle is MISSING (February 2026:
        its scheduled month-end Feb-27 was never stored) is still funded and
        marked in order. The replay walks ONE chronological daily loop, so
        February's snapshots must NOT carry March's deposit — the old
        book-all-deposits-then-snapshot ordering applied FUTURE money to
        earlier months' marks whenever one span covered several calendar
        months. February is not fabricated into a rebalance either."""
        from app.db import MonthlyAllowance as Al
        from app.db import MonthlyRebalance as Rb
        from app.db import MonthlySnapshot as Sn

        dates = pd.bdate_range("2026-01-02", "2026-04-30")
        hole = pd.Timestamp("2026-02-27")  # February's scheduled month-end
        assert hole in dates
        dates = dates.delete(dates.get_loc(hole))
        close = pd.DataFrame({"AAA": 100.0, "BBB": 50.0}, index=dates)
        open_px = pd.DataFrame({"AAA": 100.0, "BBB": 50.0}, index=dates)
        vol = pd.DataFrame(1e6, index=dates, columns=["AAA", "BBB"])

        def fake_frame(m, c, v, f):
            return pd.DataFrame({
                "roe": [0.1, 0.2], "p_fcf": [20.0, 15.0], "mcap": [1e10, 1e10],
                "dollar_vol": [5e7, 5e7], "mom": [0.1, 0.2],
                "price": [100.0, 50.0], "vol": [0.2, 0.25],
                "eligible": [True, True],
            }, index=pd.Index(["AAA", "BBB"], name="ticker"))

        async def fake_load_ohlcv(_tickers, _asof, start=None):
            return open_px, close, vol

        async def fake_load_fundamentals(_tickers):
            return {"AAA": {"x": []}, "BBB": {"x": []}}

        monkeypatch.setattr(monthly, "universe_tickers", lambda _u: ["AAA", "BBB"])
        monkeypatch.setattr(monthly.fundamentals_mod, "load_fundamentals",
                            fake_load_fundamentals)
        monkeypatch.setattr(monthly, "load_ohlcv_frames", fake_load_ohlcv)
        monkeypatch.setattr(monthly, "eligible_frame", fake_frame)
        monkeypatch.setattr(monthly, "_current_month", lambda: "2026-03")

        r = asyncio.run(monthly.backfill(start="2026-01-02"))
        assert r["ok"] is True
        # Jan + Feb + Mar + Apr all funded despite February's missing month-end
        assert r["contributed"] == \
            pytest.approx(4 * settings.sim_monthly_contribution)

        async def check():
            async with mem_db() as s:
                al = sorted(x.month for x in (await s.scalars(select(Al))).all())
                assert al == ["2026-01", "2026-02", "2026-03", "2026-04"]
                rbs = sorted(x.rebal_month
                             for x in (await s.scalars(select(Rb))).all())
                # only months with a STORED month-end decision AND completed
                # fills — no fabricated February rebuild; March's fills land
                # on Apr-1 (the first session after its month-end), inside
                # the now-extended window; April's would fill May-1 (outside)
                assert rbs == ["2026-01", "2026-03"]
                snaps = (await s.scalars(
                    select(Sn).order_by(Sn.created_at))).all()
                by_day = {x.created_at.strftime("%Y-%m-%d"): x.allowance_total
                          for x in snaps}
                # February's marks exclude March's money (March's deposit
                # lands on Mar-2, never inside February). Feb-27 itself has
                # no snapshot — its candle is the hole.
                assert by_day["2026-02-02"] == pytest.approx(2000.0)
                assert by_day["2026-02-26"] == pytest.approx(2000.0)
                assert "2026-02-27" not in by_day
                assert by_day["2026-03-02"] == pytest.approx(3000.0)
                assert by_day["2026-04-01"] == pytest.approx(4000.0)
        asyncio.run(check())


class TestReset:
    def test_reset_clears_state_and_restores_start_cash(self, mem_db):
        from app.db import MonthlyAllowance as Al
        from app.db import MonthlyPosition as Pos
        from app.db import MonthlyRebalance as Rb
        from app.db import MonthlySnapshot as Sn
        from app.db import MonthlyTrade as Tr

        async def seed():
            async with mem_db() as s:
                acc = await s.get(MonthlyAccount, 1)
                if acc is None:
                    acc = MonthlyAccount(id=1, cash=0.0)
                    s.add(acc)
                acc.cash = 123.45
                acc.last_allowance_month = "2026-09"
                acc.last_rebalance_month = "2026-09"
                s.add(Pos(ticker="AAA", shares=1.0, avg_cost=10.0))
                s.add(Tr(ticker="AAA", side="BUY", shares=1.0, price=10.0,
                         cash_after=0.0, reason="x"))
                s.add(Al(amount=1000.0, month="2026-09"))
                s.add(Sn(cash=0.0, positions_value=1.0, total_equity=1.0,
                         allowance_total=1000.0))
                s.add(Rb(rebal_month="2026-09", rebal_date=datetime(2026, 9, 30),
                         held_before="", picked="AAA", n_new=1, snapshot=""))
                await s.commit()
        asyncio.run(seed())

        r = asyncio.run(monthly.reset_monthly())
        assert r["ok"] is True
        assert r["cash"] == settings.sim_monthly_start_cash

        async def check():
            async with mem_db() as s:
                for model in (Pos, Tr, Al, Sn, Rb):
                    assert (await s.scalars(select(model))).all() == []
                acc = await s.get(MonthlyAccount, 1)
                assert acc.cash == settings.sim_monthly_start_cash
                assert acc.last_allowance_month is None
                assert acc.last_rebalance_month is None
        asyncio.run(check())


async def test_load_frames_filters_tickers_and_bounds_history(mem_db):
    """load_frames must query ONLY the requested tickers (the unfiltered
    full-table scan made every call O(all candles) once the broad universes
    were added) and honor the optional history bound."""
    from app.db import Candle

    old = datetime(2020, 1, 2)
    recent = datetime.now(UTC).replace(tzinfo=None) - timedelta(days=5)
    async with mem_db() as s:
        for i in range(3):
            s.add(Candle(ticker="AAA", timestamp=old + timedelta(days=i),
                         open=1, high=1, low=1, close=10, volume=1))
            s.add(Candle(ticker="UNRELATED", timestamp=old + timedelta(days=i),
                         open=1, high=1, low=1, close=99, volume=1))
        s.add(Candle(ticker="AAA", timestamp=recent, open=1, high=1, low=1, close=20, volume=1))
        await s.commit()

    close, _vol = await monthly.load_frames(["AAA"], None)
    assert list(close.columns) == ["AAA"]  # UNRELATED never loaded
    assert len(close) == 4

    # history bound drops the 2020 rows, keeping only the recent bar
    close2, _ = await monthly.load_frames(["AAA"], None, start=recent - timedelta(days=1))
    assert len(close2) == 1
    assert float(close2["AAA"].iloc[-1]) == 20


async def test_first_snapshot_is_parked_contribution(mem_db, monkeypatch):
    """The monthly engine parks the contribution until month-end, so the FIRST
    snapshot must equal what has been contributed (100%) — not the value of the
    month-end picks marked at the month's earlier (lower) prices. Regression:
    the replay used to build the month-end portfolio and then snapshot the
    whole month, so a 2025 backfill's first point read 91% instead of 100%."""
    from app.db import MonthlySnapshot as Sn

    dates = pd.bdate_range("2026-07-01", "2026-08-31")
    rising = [100.0 * (1.01 ** i) for i in range(len(dates))]
    close = pd.DataFrame({"AAA": rising, "BBB": [x / 2 for x in rising]}, index=dates)
    vol = pd.DataFrame(1e6, index=dates, columns=["AAA", "BBB"])

    def fake_frame(m, c, v, f):
        return pd.DataFrame({
            "roe": [0.1, 0.2], "p_fcf": [20.0, 15.0], "mcap": [1e10, 1e10],
            "dollar_vol": [5e7, 5e7], "mom": [0.1, 0.2], "price": [100.0, 50.0],
            "vol": [0.2, 0.25], "eligible": [True, True],
        }, index=pd.Index(["AAA", "BBB"], name="ticker"))

    async def fake_load_ohlcv(_t, _a, start=None):
        return (pd.DataFrame({"AAA": rising, "BBB": [x / 2 for x in rising]},
                             index=dates),
                close,
                vol)

    async def fake_load_fundamentals(_t):
        return {"AAA": {"x": []}, "BBB": {"x": []}}

    monkeypatch.setattr(monthly, "universe_tickers", lambda _u: ["AAA", "BBB"])
    monkeypatch.setattr(monthly, "load_ohlcv_frames", fake_load_ohlcv)
    monkeypatch.setattr(monthly.fundamentals_mod, "load_fundamentals", fake_load_fundamentals)
    monkeypatch.setattr(monthly, "eligible_frame", fake_frame)

    await monthly.backfill("2026-07-01")

    async with mem_db() as s:
        rows = (await s.scalars(select(Sn).order_by(Sn.created_at))).all()
    assert rows, "backfill produced snapshots"
    assert rows[0].total_equity == pytest.approx(1000.0)      # parked, 100%
    assert rows[0].allowance_total == pytest.approx(1000.0)
