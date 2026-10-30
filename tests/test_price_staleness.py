"""Price-age policy (``app/price_freshness.py``) — one staleness definition.

The audit finding (docs/ui-strategy-audit-2026-10-21.md, P1-C): forward-filled
or average-cost fallback prices need stale flags and age limits. A missing
quote is not a valid executable price, and an indefinitely stale listing must
not remain silently eligible.

Every test here is synthetic and offline: temp/in-memory SQLite seeded with
hand-written candle rows (or plain DataFrames for the pure helpers), no
provider or network call, no production DB.

Covered:

1. ``is_fresh`` boundary grid (exact max age, tz-aware/naive, future print).
2. ``freshness_mask`` on a small frame (fresh / NaN / ffilled-past-limit).
3. ``monthly.eligible_frame``: only the fresh name is eligible; the stale
   names carry ``stale=True`` and are excluded even with strong mcap/mom.
4. ``monthly._price_usd_map``: an old close -> None, fresh -> the exact
   converted price; an ancient FX print cannot convert a fresh close.
5. Valuation stale flags (``daily_core.valuate`` / ``monthly.monthly_valuate``)
   — cost-basis mark, ``"stale": True``, sorted ``stale_marks``.
6. The persisted replay (``daily_core.backfill``) and the research replay
   (``optimize._daily_core_backtest``) mark a stale ffilled name at cost basis
   while a fresh name marks at its close.
7. ``daily_core._exec_buy``'s hard slot backstop (the third fill path).
8. The config default and an end-to-end override of the limit.
"""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import StaticPool, select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

import app.db as db_mod
from app import daily_core, monthly, optimize, price_freshness
from app.config import settings
from app.db import (
    Base,
    Candle,
    DailyCoreAccount,
    DailyCorePosition,
    DailyCoreTrade,
    MonthlyAccount,
    MonthlyPosition,
)

# The synthetic valuation clock: inside ``_price_usd_map``'s 400-day candle
# lookback, so every fixture date below is query-visible.
CLOCK = datetime(2025, 3, 20, 21, 0, tzinfo=UTC)
CLOCK_DAY = pd.Timestamp("2025-03-20")


@pytest.fixture
async def mem_db(monkeypatch):
    """In-memory SQLite for the monthly price map AND the daily-core engine."""
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    monkeypatch.setattr(monthly, "Session", session_factory)
    monkeypatch.setattr(daily_core, "Session", session_factory)
    monkeypatch.setattr(db_mod, "Session", session_factory)
    monkeypatch.setattr(monthly, "_now", lambda: CLOCK)
    yield session_factory
    await engine.dispose()


def _candle(ticker: str, d, close: float, *, open_: float | None = None) -> Candle:
    o = close if open_ is None else open_
    return Candle(
        ticker=ticker,
        timestamp=pd.Timestamp(d).to_pydatetime(),
        open=float(o), high=float(max(o, close)), low=float(min(o, close)),
        close=float(close), volume=1e6,
    )


async def _seed(session_factory, rows: list[Candle]) -> None:
    async with session_factory() as s:
        s.add_all(rows)
        await s.commit()


# ---------------------------------------------------------------------------
# 1. is_fresh: the boundary grid
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(("print_ts", "asof_ts", "max_age", "expected"), [
    # same instant -> the newest print is fresh
    ("2025-03-20T00:00", "2025-03-20T00:00", 10, True),
    # a print exactly AT the limit is still fresh ...
    ("2025-03-10T00:00", "2025-03-20T00:00", 10, True),
    # ... one second past it is not
    ("2025-03-10T00:00", "2025-03-20T00:00:01", 10, False),
    # comfortably inside / outside
    ("2025-03-11T23:00", "2025-03-20T00:00", 10, True),
    ("2025-03-09T00:00", "2025-03-20T00:00", 10, False),
    # a FUTURE print is not a known-as-of mark
    ("2025-03-21T00:00", "2025-03-20T00:00", 10, False),
    # tz-aware print vs naive asof (drop-tz convention: same wall clock)
    (pd.Timestamp("2025-03-15T00:00", tz="UTC"), pd.Timestamp("2025-03-20T00:00"), 10, True),
    # naive print vs tz-aware asof
    (pd.Timestamp("2025-03-15T00:00"), pd.Timestamp("2025-03-20T00:00", tz="UTC"), 10, True),
    # tz-aware both, different offsets, same wall clock
    (pd.Timestamp("2025-03-15T00:00+02:00"), pd.Timestamp("2025-03-20T00:00-05:00"), 10, True),
    # max_age 0: only the print itself
    ("2025-03-20T00:00", "2025-03-20T00:00", 0, True),
    ("2025-03-19T23:59", "2025-03-20T00:00", 0, False),
    # missing inputs are never fresh
    (None, "2025-03-20T00:00", 10, False),
    ("2025-03-20T00:00", None, 10, False),
    (pd.NaT, "2025-03-20T00:00", 10, False),
])
def test_is_fresh_grid(print_ts, asof_ts, max_age, expected):
    assert price_freshness.is_fresh(print_ts, asof_ts, max_age) is expected


def test_is_fresh_default_limit_is_the_settings_default():
    """The callable default is the one config setting (10 calendar days)."""
    base = pd.Timestamp("2025-03-10T00:00")
    assert price_freshness.is_fresh(base, base + pd.Timedelta(days=10)) is True
    assert price_freshness.is_fresh(base, base + pd.Timedelta(days=10, seconds=1)) is False


# ---------------------------------------------------------------------------
# 2. freshness_mask: ffill cells only while the print behind them is fresh
# ---------------------------------------------------------------------------

def test_freshness_mask_fresh_nan_and_ffilled_past_limit():
    idx = pd.DatetimeIndex(["2025-03-03", "2025-03-10", "2025-03-14",
                            "2025-03-17", "2025-03-20", "2025-03-25"])
    close = pd.DataFrame({
        # prints on 03-03, then nothing: ffilled cells stay fresh until 03-13
        "FFILL": [100.0, np.nan, np.nan, np.nan, np.nan, np.nan],
        # never prints at all
        "NOPRINT": [np.nan] * 6,
        # prints every row: always fresh
        "DENSE": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        # prints late: NaN cells before the first print have no source
        "LATE": [np.nan, np.nan, np.nan, 7.0, np.nan, np.nan],
    }, index=idx)

    mask = price_freshness.freshness_mask(close, None, 10)

    assert mask.index.equals(close.index) and mask.columns.equals(close.columns)
    assert all(dt == np.dtype(bool) for dt in mask.dtypes)
    # 03-03 -> 03-10 is 7 days (fresh), 03-14 is 11 (stale), and every later
    # row stays stale: a NaN cell is True only while its source print is young.
    assert mask["FFILL"].tolist() == [True, True, False, False, False, False]
    # no print at all -> never a usable mark
    assert mask["NOPRINT"].tolist() == [False] * 6
    assert mask["DENSE"].tolist() == [True] * 6
    # 03-17's own print is fresh through 03-27; before it there is nothing
    assert mask["LATE"].tolist() == [False, False, False, True, True, True]


def test_freshness_mask_asof_cap_and_unsorted_input():
    idx = pd.DatetimeIndex(["2025-03-17", "2025-03-18", "2025-03-19"])
    close = pd.DataFrame({"A": [1.0, 2.0, 3.0]}, index=idx)
    capped = price_freshness.freshness_mask(close, pd.Timestamp("2025-03-18"), 10)
    # rows after the caller's clock never rule on anything
    assert capped["A"].tolist() == [True, True, False]

    unsorted = close.iloc[::-1]
    out = price_freshness.freshness_mask(unsorted, None, 10)
    # returned in the INPUT row order, with the per-row answer intact
    assert out.index.tolist() == unsorted.index.tolist()
    assert out["A"].tolist() == [True, True, True]


def test_freshness_mask_empty_frame():
    out = price_freshness.freshness_mask(pd.DataFrame(), None, 10)
    assert out.empty


# ---------------------------------------------------------------------------
# 3. eligible_frame: the age gate excludes a stale name from the ranking
# ---------------------------------------------------------------------------

RANK_DAY = pd.Timestamp("2025-03-20")
TAPE = pd.bdate_range("2023-10-02", RANK_DAY)


def _tape():
    """Three names with an identical, strongly-rising price path; only their
    LAST PRINT differs:

    * FRESH     — prints on every session through 2025-03-17 (3 days before
                  the ranking date) -> fresh;
    * MIDSTALE  — prints through 2025-03-07 (13 days before) -> stale, yet
                  inside the last 20 sessions, so its 20d dollar volume and
                  momentum stay finite and strong: the age gate is the ONLY
                  reason it is ineligible;
    * OLD       — dense history through 2024-12-31 (79 days before) and
                  nothing since -> stale AND no recent dollar volume.
    """
    prices = pd.Series(100.0 + 0.05 * np.arange(len(TAPE)), index=TAPE)
    close = pd.DataFrame({"FRESH": prices, "MIDSTALE": prices, "OLD": prices},
                         index=TAPE)
    close.loc[close.index > pd.Timestamp("2025-03-17"), "FRESH"] = np.nan
    close.loc[close.index > pd.Timestamp("2025-03-07"), "MIDSTALE"] = np.nan
    close.loc[close.index > pd.Timestamp("2024-12-31"), "OLD"] = np.nan
    vol = pd.DataFrame(1e7, index=TAPE, columns=close.columns)
    return close, vol


def _fund(tickers):
    """Strong fundamentals public well before RANK_DAY (same shape as the
    monthly fixture builders: a filed/end pair per fact)."""
    return {
        t: {
            "StockholdersEquity": [{"start": None, "end": "2024-09-30",
                                    "filed": "2024-11-01", "val": 5e9}],
            "NetIncomeLoss": [{"start": "2024-01-01", "end": "2024-12-31",
                               "filed": "2025-02-15", "val": 5e8}],
            "CommonStockSharesOutstanding": [{"start": None, "end": "2024-09-30",
                                              "filed": "2024-11-01", "val": 1e8}],
            "NetCashProvidedByUsedInOperatingActivities": [
                {"start": "2024-01-01", "end": "2024-12-31",
                 "filed": "2025-02-15", "val": 1e9}],
            "PaymentsToAcquirePropertyPlantAndEquipment": [
                {"start": "2024-01-01", "end": "2024-12-31",
                 "filed": "2025-02-15", "val": -2e8}],
        }
        for t in tickers
    }


def test_eligible_frame_stale_names_are_excluded_not_silently_eligible():
    close, vol = _tape()
    frame = monthly.eligible_frame(RANK_DAY, close, vol, _fund(close.columns))
    assert frame is not None
    assert list(frame.index) == ["FRESH", "MIDSTALE", "OLD"]

    # the fresh name is eligible, and carries no stale flag
    assert not bool(frame.loc["FRESH", "stale"])
    assert bool(frame.loc["FRESH", "eligible"])

    # the 13-day-old print is STALE and therefore ineligible ...
    assert bool(frame.loc["MIDSTALE", "stale"])
    assert not bool(frame.loc["MIDSTALE", "eligible"])
    # ... despite a strong market cap, a finite momentum and a live 20d
    # dollar volume — the age gate is the only thing excluding it
    assert frame.loc["MIDSTALE", "mcap"] > settings.sim_monthly_min_mcap
    assert frame.loc["MIDSTALE", "dollar_vol"] > settings.sim_monthly_min_dollar_vol
    assert np.isfinite(frame.loc["MIDSTALE", "mom"])

    # the months-old listing is stale and excluded as well
    assert bool(frame.loc["OLD", "stale"])
    assert not bool(frame.loc["OLD", "eligible"])

    # the age gate reads the last ACTUAL print, not the ffilled frame
    assert frame.loc["FRESH", "price"] == pytest.approx(
        float(close.loc[close.index <= pd.Timestamp("2025-03-17"), "FRESH"].iloc[-1]))


def test_eligible_frame_honours_an_overridden_age_limit(monkeypatch):
    """Moving the setting moves the boundary end-to-end (13-day print)."""
    close, vol = _tape()
    fund = _fund(close.columns)
    monkeypatch.setattr(settings, "factor_price_max_age_days", 20)
    wider = monthly.eligible_frame(RANK_DAY, close, vol, fund)
    assert not bool(wider.loc["MIDSTALE", "stale"])
    assert bool(wider.loc["MIDSTALE", "eligible"])   # 13 days < 20
    assert bool(wider.loc["OLD", "stale"])           # 79 days > 20

    monkeypatch.setattr(settings, "factor_price_max_age_days", 10)
    narrow = monthly.eligible_frame(RANK_DAY, close, vol, fund)
    assert bool(narrow.loc["MIDSTALE", "stale"])
    assert not bool(narrow.loc["MIDSTALE", "eligible"])


# ---------------------------------------------------------------------------
# 4. _price_usd_map: the stale close (and the stale rate) fail closed
# ---------------------------------------------------------------------------

async def test_price_map_stale_close_is_none_and_fresh_close_is_exact(mem_db):
    await _seed(mem_db, [
        _candle("FRESH", CLOCK_DAY - pd.Timedelta(days=3), 101.0),
        _candle("STALE", CLOCK_DAY - pd.Timedelta(days=30), 555.0),
    ])
    out = await monthly._price_usd_map(["FRESH", "STALE"], asof=CLOCK)
    assert out["FRESH"] == pytest.approx(101.0)
    assert out["STALE"] is None      # a 30-day-old print is not a mark


async def test_price_map_uses_the_newest_row_as_clock_when_asof_is_none(mem_db):
    await _seed(mem_db, [
        _candle("FRESH", CLOCK_DAY - pd.Timedelta(days=2), 101.0),
        _candle("STALE", CLOCK_DAY - pd.Timedelta(days=30), 555.0),
    ])
    out = await monthly._price_usd_map(["FRESH", "STALE"], asof=None)
    assert out["FRESH"] == pytest.approx(101.0)
    assert out["STALE"] is None


async def test_price_map_ancient_fx_print_cannot_convert_a_fresh_close(mem_db):
    px_day = CLOCK_DAY - pd.Timedelta(days=3)
    await _seed(mem_db, [
        _candle("EURUSD=X", px_day - pd.Timedelta(days=40), 1.10),  # ancient
        _candle("SAP.DE", px_day, 200.0),
    ])
    assert (await monthly._price_usd_map(["SAP.DE"], asof=CLOCK))["SAP.DE"] is None

    # A fresh rate converts exactly as before (close x pair close).
    await _seed(mem_db, [_candle("EURUSD=X", px_day, 1.40)])
    assert (await monthly._price_usd_map(["SAP.DE"], asof=CLOCK))["SAP.DE"] == \
        pytest.approx(200.0 * 1.40)


async def test_price_map_fx_age_is_measured_against_the_price_timestamp(mem_db):
    """The rate's age is measured from the PRICE's ts, not from ``asof``: a
    rate printed 12 days before a fresh close is stale even though it is
    recent relative to the valuation clock."""
    px_day = CLOCK_DAY - pd.Timedelta(days=1)
    await _seed(mem_db, [
        _candle("EURUSD=X", px_day - pd.Timedelta(days=12), 1.10),
        _candle("SAP.DE", px_day, 200.0),
    ])
    assert (await monthly._price_usd_map(["SAP.DE"], asof=CLOCK))["SAP.DE"] is None


# ---------------------------------------------------------------------------
# 5. Valuation stale flags (live daily-core + monthly shape parity)
# ---------------------------------------------------------------------------

async def test_valuate_flags_a_stale_position_and_marks_it_at_cost(mem_db):
    await _seed(mem_db, [
        _candle("FRESH", CLOCK_DAY - pd.Timedelta(days=3), 60.0),
        _candle("STALE", CLOCK_DAY - pd.Timedelta(days=30), 90.0),
    ])
    async with mem_db() as s:
        s.add(DailyCoreAccount(id=1, cash=100.0, last_allowance_month="2025-03"))
        s.add(DailyCorePosition(ticker="FRESH", shares=10.0, avg_cost=50.0,
                                opened_at=datetime(2025, 1, 15)))
        s.add(DailyCorePosition(ticker="STALE", shares=10.0, avg_cost=50.0,
                                opened_at=datetime(2025, 1, 15)))
        await s.commit()

    val = await daily_core.valuate(now=CLOCK)
    by = {p["ticker"]: p for p in val["positions"]}

    # the fresh name marks at its close, unflagged
    assert by["FRESH"]["current_price"] == pytest.approx(60.0)
    assert by["FRESH"]["stale"] is False
    assert by["FRESH"]["value"] == pytest.approx(600.0)

    # the stale name marks at its cost basis, flagged, and never at the
    # 30-day-old 90 print
    assert by["STALE"]["current_price"] == pytest.approx(50.0)
    assert by["STALE"]["stale"] is True
    assert by["STALE"]["value"] == pytest.approx(500.0)
    assert val["stale_marks"] == ["STALE"]
    assert val["positions_value"] == pytest.approx(1100.0)
    assert val["total_equity"] == pytest.approx(1200.0)

    # additive keys only: the pre-existing shape is untouched
    assert set(by["FRESH"]) == {"ticker", "shares", "avg_cost", "current_price",
                                "value", "pnl_pct", "opened_at", "stale"}
    assert set(val) == {"cash", "positions", "positions_value", "total_equity",
                        "allowance_total", "stale_marks"}


async def test_monthly_valuate_mirrors_the_stale_flags(mem_db):
    await _seed(mem_db, [
        _candle("FRESH", CLOCK_DAY - pd.Timedelta(days=3), 60.0),
        _candle("STALE", CLOCK_DAY - pd.Timedelta(days=30), 90.0),
    ])
    async with mem_db() as s:
        s.add(MonthlyAccount(id=1, cash=0.0, last_allowance_month="2025-03"))
        s.add(MonthlyPosition(ticker="FRESH", shares=5.0, avg_cost=50.0,
                              opened_at=datetime(2025, 1, 15)))
        s.add(MonthlyPosition(ticker="STALE", shares=5.0, avg_cost=50.0,
                              opened_at=datetime(2025, 1, 15)))
        await s.commit()

    val = await monthly.monthly_valuate(now=CLOCK)
    by = {p["ticker"]: p for p in val["positions"]}
    assert by["FRESH"]["stale"] is False
    assert by["FRESH"]["current_price"] == pytest.approx(60.0)
    assert by["STALE"]["stale"] is True
    assert by["STALE"]["current_price"] == pytest.approx(50.0)
    assert val["stale_marks"] == ["STALE"]
    assert set(by["STALE"]) == {"ticker", "shares", "avg_cost", "current_price",
                                "value", "pnl_pct", "opened_at", "stale"}
    assert set(val) == {"cash", "positions", "positions_value", "total_equity",
                        "allowance_total", "stale_marks"}


# ---------------------------------------------------------------------------
# 6. Replays: the stale ffilled name marks at cost basis, not the old close
# ---------------------------------------------------------------------------

REPLAY_TICKERS = ["FRESH", "STALE"]
REPLAY_START = "2025-12-01"
REPLAY_END = "2026-01-30"
WARMUP_START = "2025-01-02"       # >= 253 sessions of momentum warmup
STALE_LAST_PRINT = pd.Timestamp("2026-01-15")


def _sessions(start: str = WARMUP_START):
    """TRUE NYSE sessions in the range (holidays dropped, so the first
    session of January and the month-end rebuild are the real ones)."""
    from app import sim as sim_mod

    return pd.DatetimeIndex([
        d for d in pd.bdate_range(start, REPLAY_END)
        if sim_mod.is_trading_day(d.to_pydatetime())])


def _replay_frames(*, stale_quote: bool):
    """Both names are bought at 100 on the 2026-01-02 rebuild, then:

    * FRESH closes at 120 from 2026-01-05 on (a live mark to track);
    * STALE closes at 300 from 2026-01-05 through its LAST PRINT on
      2026-01-15, then has no quote at all (``stale_quote=True``).
      Its ffilled 300 is fresh until 2026-01-25 and stale afterwards, so the
      mark must fall back to the 100 cost basis — never the ancient 300.
    """
    days = _sessions()
    close = pd.DataFrame({"FRESH": 100.0, "STALE": 100.0}, index=days)
    close.loc[close.index >= pd.Timestamp("2026-01-05"), "FRESH"] = 120.0
    close.loc[close.index >= pd.Timestamp("2026-01-05"), "STALE"] = 300.0
    if stale_quote:
        close.loc[close.index > STALE_LAST_PRINT, "STALE"] = np.nan
    open_px = pd.DataFrame({"FRESH": 100.0, "STALE": 100.0}, index=days)
    vol = pd.DataFrame(1e6, index=days, columns=REPLAY_TICKERS)
    return open_px, close, vol


def _canned_frame(order):
    return pd.DataFrame(
        {t: {"roe": 0.5 - 0.01 * i, "p_fcf": 20.0, "mcap": 1e10,
             "dollar_vol": 1e8, "mom": 0.1, "price": 100.0, "vol": 0.2,
             "eligible": True}
         for i, t in enumerate(order)}).T


@pytest.fixture
def replay_env(mem_db, monkeypatch, tmp_path):
    """Patch the persisted replay's loaders/ranking onto the synthetic tape."""
    monkeypatch.setattr(daily_core, "_STATE_FILE", tmp_path / "state.json")
    monkeypatch.setattr(daily_core, "universe_tickers", lambda _u: list(REPLAY_TICKERS))
    monkeypatch.setattr(daily_core, "_current_month", lambda: "2026-02")
    monkeypatch.setattr(settings, "sim_monthly_target_n", 2)
    monkeypatch.setattr(settings, "sim_monthly_hold_band", 2)
    monkeypatch.setattr(settings, "sim_monthly_contribution", 1000.0)

    async def fake_fund(_tickers):
        return {t: {"x": []} for t in REPLAY_TICKERS}

    monkeypatch.setattr(daily_core.fundamentals_mod, "load_fundamentals", fake_fund)

    frame = _canned_frame(REPLAY_TICKERS)

    def fake_frame(m, _close, _vol, _fund):
        return frame if pd.Timestamp(m).normalize() in {
            pd.Timestamp("2025-12-31"), pd.Timestamp("2026-01-30")} else None

    def fake_order(f):
        return f["eligible"], list(f.index)

    monkeypatch.setattr(monthly, "eligible_frame", fake_frame)
    monkeypatch.setattr(monthly, "_qv_order", fake_order)
    return mem_db


async def _run_backfill(mem_db, monkeypatch, *, stale_quote: bool):
    open_px, close, vol = _replay_frames(stale_quote=stale_quote)

    async def fake_load_ohlcv(_tickers, _asof, start=None):
        return open_px, close, vol

    monkeypatch.setattr(monthly, "load_ohlcv_frames", fake_load_ohlcv)
    out = await daily_core.backfill(start=REPLAY_START)
    assert out["ok"] is True, out
    async with mem_db() as s:
        trades = (await s.scalars(
            select(DailyCoreTrade).order_by(DailyCoreTrade.id))).all()
        trades = [(t.created_at.strftime("%Y-%m-%d"), t.side, t.ticker,
                   round(float(t.shares), 6)) for t in trades]
        snaps = (await s.scalars(
            select(daily_core.DailyCoreSnapshot)
            .order_by(daily_core.DailyCoreSnapshot.created_at))).all()
        equity = {sn.created_at.strftime("%Y-%m-%d"): round(float(sn.total_equity), 2)
                  for sn in snaps}
    return trades, equity


async def test_backfill_marks_a_stale_print_at_cost_basis(replay_env, monkeypatch):
    trades, equity = await _run_backfill(replay_env, monkeypatch, stale_quote=True)
    baseline_trades, baseline_equity = await _run_backfill(
        replay_env, monkeypatch, stale_quote=False)

    # The tape differs only AFTER every fill happened: both runs hold
    # 10 FRESH + 10 STALE at a 100 cost basis.
    assert trades == baseline_trades
    assert trades == [
        ("2026-01-02", "BUY", "FRESH", 10.0),
        ("2026-01-02", "BUY", "STALE", 10.0),
    ]

    # While STALE's 300 print is still young it IS the mark (both runs agree).
    assert equity["2026-01-23"] == pytest.approx(4200.0)      # 1200 + 3000
    assert baseline_equity["2026-01-23"] == pytest.approx(4200.0)

    # Past the limit the live-quote run keeps marking at STALE's close ...
    assert baseline_equity["2026-01-30"] == pytest.approx(4200.0)
    # ... while the stale run reports STALE at its 100 cost basis, never at
    # the ancient 300: 10 * 120 + 10 * 100.
    assert equity["2026-01-30"] == pytest.approx(2200.0)


async def _run_research_replay(monkeypatch, *, stale_quote: bool):
    open_px, close, vol = _replay_frames(stale_quote=stale_quote)
    frame = _canned_frame(REPLAY_TICKERS)
    monkeypatch.setattr(optimize, "universe_tickers", lambda _u: list(REPLAY_TICKERS))
    return await optimize._daily_core_backtest(
        REPLAY_START, REPLAY_END, "diversified-plus", 1000.0, False,
        "none", "daily", False, "rank",
        frame_cache={pd.Timestamp("2025-12-31"): frame,
                     pd.Timestamp("2026-01-30"): frame},
        close_preloaded=(close, vol),
        fund_preloaded={},
        open_preloaded=open_px,
        cost_oneway=0.0,
        with_baseline=False,
    )


async def test_research_replay_marks_a_stale_print_at_cost_basis(
        replay_env, monkeypatch):
    base = await _run_research_replay(monkeypatch, stale_quote=True)
    live = await _run_research_replay(monkeypatch, stale_quote=False)

    # Identical legs (the tape diverges only after the cash is deployed).
    assert [(t["date"], t["side"], t["ticker"], round(t["shares"], 6))
            for t in base.trades] == \
        [(t["date"], t["side"], t["ticker"], round(t["shares"], 6))
         for t in live.trades] == \
        [("2026-01-02", "BUY", "FRESH", 10.0),
         ("2026-01-02", "BUY", "STALE", 10.0)]

    # Same final-valuation policy as the persisted replay: the live-quote run
    # ends at 1200 + 3000, the stale-quote run at 1200 + the 1000 cost basis.
    assert live.final_value == pytest.approx(4200.0, abs=0.02)
    assert base.final_value == pytest.approx(2200.0, abs=0.02)


# ---------------------------------------------------------------------------
# 7. _exec_buy: the hard slot backstop every fill path now carries
# ---------------------------------------------------------------------------

async def test_exec_buy_refuses_a_new_name_at_capacity(mem_db, monkeypatch):
    monkeypatch.setattr(settings, "sim_monthly_target_n", 2)
    async with mem_db() as s:
        s.add(DailyCoreAccount(id=1, cash=1000.0, last_allowance_month="2025-03"))
        s.add(DailyCorePosition(ticker="AAA", shares=1.0, avg_cost=100.0,
                                opened_at=datetime(2025, 1, 15)))
        s.add(DailyCorePosition(ticker="BBB", shares=1.0, avg_cost=100.0,
                                opened_at=datetime(2025, 1, 15)))
        await s.commit()

    r = await daily_core._exec_buy("CCC", 10.0, 100.0, "test")
    assert r is None                      # no fill ...
    async with mem_db() as s:             # ... and no mutation at all
        assert (await s.get(DailyCoreAccount, 1)).cash == pytest.approx(1000.0)
        held = (await s.scalars(
            select(DailyCorePosition).order_by(DailyCorePosition.ticker))).all()
        assert [p.ticker for p in held] == ["AAA", "BBB"]
        assert (await s.scalars(select(DailyCoreTrade))).all() == []

    # A top-up of an EXISTING name still fills at capacity.
    r2 = await daily_core._exec_buy("AAA", 10.0, 100.0, "test")
    assert r2 is not None and r2["shares"] == pytest.approx(10.0)
    async with mem_db() as s:
        assert (await s.get(DailyCoreAccount, 1)).cash == pytest.approx(900.0)
        pos = await s.scalar(
            select(DailyCorePosition).where(DailyCorePosition.ticker == "AAA"))
        assert pos.shares == pytest.approx(11.0)


# ---------------------------------------------------------------------------
# 8. The setting itself
# ---------------------------------------------------------------------------

def test_factor_price_max_age_days_default_is_ten():
    assert settings.factor_price_max_age_days == 10
    assert monthly.settings is settings
    # the pure helper's default IS that setting (one definition, no copy)
    assert price_freshness.is_fresh("2025-03-10", "2025-03-20") is True
    assert price_freshness.is_fresh("2025-03-09", "2025-03-20") is False


async def test_the_limit_is_honoured_end_to_end(mem_db, monkeypatch):
    """A 7-day-old print is fresh at the default 10 and stale at 5."""
    await _seed(mem_db, [_candle("AAA", CLOCK_DAY - pd.Timedelta(days=7), 42.0)])
    assert (await monthly._price_usd_map(["AAA"], asof=CLOCK))["AAA"] == \
        pytest.approx(42.0)

    monkeypatch.setattr(settings, "factor_price_max_age_days", 5)
    assert (await monthly._price_usd_map(["AAA"], asof=CLOCK))["AAA"] is None
