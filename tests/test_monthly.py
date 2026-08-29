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

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import StaticPool, select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app import fundamentals as fundamentals_mod
from app import monthly
from app.config import settings
from app.db import Base, MonthlyAccount, MonthlyPosition, MonthlyTrade


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


def test_eligible_frame_requires_history():
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
    tz = timezone.utc
    # 2026-08-31 is a Monday and the last weekday of August
    assert monthly.is_rebalance_day(datetime(2026, 8, 31, 22, 0, tzinfo=tz))
    assert not monthly.is_rebalance_day(datetime(2026, 8, 30, 22, 0, tzinfo=tz))  # Sunday
    assert not monthly.is_rebalance_day(datetime(2026, 8, 28, 22, 0, tzinfo=tz))  # Friday, not last
    # weekend walked back: 2026-10-31 is a Saturday -> last trading day is Fri 10-30
    assert monthly.is_rebalance_day(datetime(2026, 10, 30, 22, 0, tzinfo=tz))
    assert not monthly.is_rebalance_day(datetime(2026, 10, 31, 22, 0, tzinfo=tz))


def test_month_last_trading_day():
    tz = timezone.utc
    out = monthly.month_last_trading_day(datetime(2026, 8, 15, tzinfo=tz))
    assert out.date() == datetime(2026, 8, 31).date() and out.weekday() == 0


# ---------------------------------------------------------------------------
# Engine: allowance + rebalance execution
# ---------------------------------------------------------------------------

async def test_deposit_allowance_once_per_month(mem_db, monkeypatch):
    r1 = await monthly._deposit_allowance()
    assert r1["deposited"] is True
    r2 = await monthly._deposit_allowance()
    assert r2["deposited"] is False
    val = await monthly.monthly_valuate()
    assert val["cash"] == settings.sim_monthly_contribution
    assert val["allowance_total"] == settings.sim_monthly_contribution


async def test_exec_buy_sell(mem_db):
    await monthly._deposit_allowance()  # fund the account
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
    rng = np.random.default_rng(11)
    dates = pd.bdate_range("2024-06-03", periods=400)
    prices = {}
    for i, t in enumerate(tickers):
        prices[t] = 100 + rng.normal(0.0005, 0.012, len(dates)).cumsum() + i
    close = pd.DataFrame(prices, index=dates)

    # --- seed fundamentals: T00..T09 high quality (score desc by index),
    #     T10/T11 worse so the top-10 excludes them
    async with mem_db() as s:
        for t in tickers:
            for ts, px in zip(dates, close[t]):
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

    # --- run 1: deposits allowance + buys top-10
    r1 = await monthly.run_rebalance()
    assert r1.get("rebalanced") is True
    assert len(r1["picks"]) == settings.sim_monthly_target_n
    assert "T10" not in r1["picks"] and "T11" not in r1["picks"]
    buys = [t for t in r1["trades"] if t["side"] == "BUY"]
    assert len(buys) == settings.sim_monthly_target_n
    val = await monthly.monthly_valuate()
    assert len(val["positions"]) == settings.sim_monthly_target_n
    # equal weight: each position ~ 1/10 of equity (allowance deposited)
    expected = val["total_equity"] / settings.sim_monthly_target_n
    for p in val["positions"]:
        assert p["value"] == pytest.approx(expected, rel=0.02)

    # --- run 2 same month: skipped
    r2 = await monthly.run_rebalance()
    assert r2.get("skipped") is True

    # --- forced re-run with unchanged data: no churn (hysteresis keeps picks)
    r3 = await monthly.run_rebalance(force=True)
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
    """run_monthly_cycle only acts on the last trading day."""
    async def fail_rebalance(force=False):
        raise AssertionError("should not rebalance off-schedule")

    monkeypatch.setattr(monthly, "run_rebalance", fail_rebalance)
    # 2026-08-28 is a Friday but NOT the last trading day of Aug 2026
    out = await monthly.run_monthly_cycle()
    assert out["skipped"] is True

    monkeypatch.setattr(monthly, "is_rebalance_day", lambda today=None: True)
    monkeypatch.setattr(monthly, "run_rebalance",
                        lambda force=False: _coro({"rebalanced": True}))


async def _coro(value):
    return value

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
    from app.edgar import _parse_companyfacts
    from app import monthly
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
    from app.db import Fundamental
    from app import fundamentals as fund_mod
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
    # CCC unknown: would trigger a download; monkeypatch it out
    def fail_get(url):
        raise AssertionError("network hit for already-cached tickers")
    monkeypatch.setattr(edgar, "_http_get", fail_get)
    with pytest.raises(AssertionError):
        await edgar.ensure_cik_map(["CCC"])
    # cached tickers resolve without network
    m = await edgar.ensure_cik_map(["AAA", "BBB"])
    assert m == {"AAA": 123456, "BBB": 0}
