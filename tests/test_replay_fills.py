"""Session fill contract (app/session.py) in the RESEARCH replays.

The live/persisted engines were wired to the contract first (decision on data
<= S, fills at F's strict exact-date open, standing-order stops); this file
pins the same rules in the research replay, ``optimize._daily_core_backtest``:

(a) GOLDEN research-vs-live parity: on ONE shared synthetic tape, the research
    daily-core backtest books exactly the same legs (date/side/ticker/shares/
    fill price) as ``daily_core.backfill`` — the fee-free live replay. This is
    what makes "the research number models the live engine" a testable claim
    instead of a hope (run at ``cost_oneway=0.0``: the live paper portfolios
    are fee-free by design, the research 10 bps is a documented cost model).
(b) The fill price is the session's STRICT exact-date OPEN, never its close:
    on a tape whose open != close every booked price is the open.
(c) The decision is made on data <= S: mutating F's OWN bar cannot change a
    single fill on F — the change is only visible from the NEXT session (the
    standing-order/overlay levels that govern F are struck at S's close).
(d) A ticker without a stored bar on the fill session gets NO fill (a missing
    quote is not an executable price) and the leg retries at the next session;
    the still-occupied slot never lets the book grow past the target.
(e) The TECHNICAL replay (``optimize._replay``) runs the same contract when
    its series carry session bars: fills at F's strict open, decisions on S,
    and the standing-order stop rule (open through the level -> the open;
    else the low -> the level; else nothing).

The golden tape builder is copied from ``tests/test_cadence_parity.py`` (test
modules must not import each other); the same all-positive fundamentals keep
every month-end frame eligible after it turns live, which is what makes the
"newest eligible month-end" rule agree on both sides.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import StaticPool, select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app import daily_core, optimize
from app import fundamentals as fundamentals_mod
from app import monthly as monthly_mod
from app.config import settings
from app.db import Base, Candle, DailyCoreTrade

TICKERS = ["T01", "T02", "T03", "T04", "T05", "T06", "T07", "T08"]

TAPE_START = "2024-06-03"
TAPE_END = "2026-10-02"


# ---------------------------------------------------------------------------
# Shared tape (copied from test_cadence_parity.py — do not import it)
# ---------------------------------------------------------------------------

def _tape() -> tuple[pd.DatetimeIndex, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Sessions, strict OPEN frame, CLOSE frame, volume frame.

    Piecewise paths create REAL rank flips at month-ends (T07 surges from
    2025-12-31, T08 sinks from 2026-01-30, T03 from 2026-04-01 → releases),
    and every session closes with a ±3% per-ticker phase-shifted wiggle so the
    daily path and the month-end frame disagree — the fill parity cannot be an
    artifact of a static ranking. ``open != close`` on every bar.
    """
    from app import sim as sim_mod

    dates = pd.bdate_range(TAPE_START, TAPE_END)
    sessions = pd.DatetimeIndex(
        [d for d in dates if sim_mod.is_trading_day(d.to_pydatetime())])
    base = pd.DataFrame(100.0, index=sessions, columns=TICKERS)
    n = len(sessions)

    def _regime(ticker: str, start: str, factor: float) -> None:
        mask = sessions >= pd.Timestamp(start)
        steps = np.arange(1, int(mask.sum()) + 1)
        base.loc[mask, ticker] = 100.0 * factor ** steps

    _regime("T07", "2025-12-31", 1.005)
    _regime("T08", "2026-01-30", 0.995)
    _regime("T03", "2026-04-01", 0.99)

    phase = 0.7 * np.arange(len(TICKERS))
    wig = 1.0 + 0.03 * np.sin(
        2 * np.pi * np.arange(n)[:, None] / 9.0 + phase[None, :])
    close = base.mul(pd.DataFrame(wig, index=sessions, columns=TICKERS))
    open_px = close.mul(1.0 - 0.002 * (np.arange(n) % 5) / 5.0, axis=0)
    vol = pd.DataFrame(1e6, index=sessions, columns=TICKERS)
    return sessions, open_px, close, vol


def _fund_all_positive() -> dict[str, dict[str, list[dict]]]:
    """Facts public well before the first live month-end (all names stay
    eligible once the FY2024 filing lands — no interior dead month-end)."""

    def _roe(t: str) -> float:
        return 0.5 - 0.01 * TICKERS.index(t) * 10

    return {
        t: {
            "StockholdersEquity": [{"start": None, "end": "2024-06-30",
                                    "filed": "2024-08-14", "val": 5e9}],
            "NetIncomeLoss": [{"start": "2024-01-01", "end": "2024-12-31",
                               "filed": "2025-02-14",
                               "val": 5e9 * _roe(t)}],
            "CommonStockSharesOutstanding": [{"start": None,
                                              "end": "2024-06-30",
                                              "filed": "2024-08-14",
                                              "val": 1e8}],
            "NetCashProvidedByUsedInOperatingActivities": [
                {"start": "2024-01-01", "end": "2024-12-31",
                 "filed": "2025-02-14", "val": 1e9}],
            "PaymentsToAcquirePropertyPlantAndEquipment": [
                {"start": "2024-01-01", "end": "2024-12-31",
                 "filed": "2025-02-14", "val": -2e8}],
        }
        for t in TICKERS
    }


@pytest.fixture
async def parity_env(mem_db, monkeypatch, tmp_path):
    """Seed the tape into the temp DB; returns (sessions, open, close, vol, fund)."""
    sessions, open_px, close, vol = _tape()
    fund = _fund_all_positive()

    async with mem_db() as s:
        await s.execute(
            Candle.__table__.insert(),
            [{"ticker": t, "timestamp": d.to_pydatetime(),
              "open": float(open_px.at[d, t]),
              "high": float(max(open_px.at[d, t], close.at[d, t])),
              "low": float(min(open_px.at[d, t], close.at[d, t])),
              "close": float(close.at[d, t]), "volume": 1e6}
             for d in sessions for t in TICKERS])
        await s.commit()

    async def fake_fund(_tickers):
        return fund

    monkeypatch.setattr(fundamentals_mod, "load_fundamentals", fake_fund)
    monkeypatch.setattr(monthly_mod, "universe_tickers", lambda _u: list(TICKERS))
    monkeypatch.setattr(daily_core, "universe_tickers", lambda _u: list(TICKERS))
    monkeypatch.setattr(settings, "sim_monthly_target_n", 3)
    monkeypatch.setattr(settings, "sim_monthly_hold_band", 6)
    # Both engines must rank the same variant: `backfill` pins the runtime
    # variant (`current_variant()`), so point that at an empty state file and
    # pin the ambient default.
    monkeypatch.setattr(settings, "sim_daily_core_mom_variant", "raw")
    monkeypatch.setattr(daily_core, "_STATE_FILE", tmp_path / "state.json")
    return sessions, open_px, close, vol, fund


@pytest.fixture
async def mem_db(monkeypatch):
    """In-memory SQLite DB shared by ``daily_core.backfill`` and the loaders."""
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    session_factory = async_sessionmaker(engine, expire_on_commit=False)

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    monkeypatch.setattr(daily_core, "Session", session_factory)
    monkeypatch.setattr(monthly_mod, "Session", session_factory)
    import app.db as db_mod
    monkeypatch.setattr(db_mod, "Session", session_factory)

    yield session_factory
    await engine.dispose()


async def _backfill_rows() -> list[tuple]:
    """``daily_core.backfill``'s booked legs, in stream order."""
    async with daily_core.Session() as s:
        rows = (await s.scalars(
            select(DailyCoreTrade).order_by(DailyCoreTrade.id))).all()
    return [(r.created_at.strftime("%Y-%m-%d"), r.side, r.ticker,
             round(float(r.shares), 6), round(float(r.price), 6)) for r in rows]


def _research_rows(summary) -> list[tuple]:
    """The research replay's booked legs, in stream order."""
    return [(t["date"], t["side"], t["ticker"],
             round(float(t["shares"]), 6), round(float(t["price"]), 6))
            for t in summary.trades]


async def _run_research(tape, *, start, end=None, **kw):
    _, open_px, close, vol, fund = tape
    return await optimize._daily_core_backtest(
        start, end, "diversified-plus", settings.sim_monthly_contribution,
        False, "none", "daily", False, "rank",
        frame_cache=None, close_preloaded=(close, vol), fund_preloaded=fund,
        open_preloaded=kw.pop("open_preloaded", open_px),
        cost_oneway=kw.pop("cost_oneway", 0.0),
        with_baseline=False, **kw)


# ---------------------------------------------------------------------------
# (a) GOLDEN: research daily-core replay == live persisted backfill
# ---------------------------------------------------------------------------

async def test_research_replay_matches_the_live_backfill(parity_env):
    """One tape, two engines, the same booked legs: the research daily-core
    backtest is a faithful model of ``daily_core.backfill`` (the fee-free
    replay of the live engine). Compared on date/side/ticker/shares/price —
    a fill-convention, sizing or decision-cadence bug breaks it."""
    sessions, _open_px, close, vol, fund = parity_env
    start = sessions[0].strftime("%Y-%m-%d")

    replay = await daily_core.backfill(start=start)
    assert replay["ok"] is True, replay
    live_rows = await _backfill_rows()

    res = await _run_research(parity_env, start=start)
    assert res is not None
    research_rows = _research_rows(res)

    # the tape exercises the contract: both sides trade, release and re-buy
    assert live_rows, "the parity tape produced no live legs"
    assert {r[1] for r in live_rows} == {"BUY", "SELL"}
    assert research_rows == live_rows, (
        "research daily-core replay diverges from daily_core.backfill — the "
        "session fill contract is not wired identically")

    # the same money: no cost model at cost_oneway=0 on either side.
    # `backfill` persists its equity rounded to cents; the research summary
    # keeps the unrounded mark.
    assert round(res.final_value, 2) == replay["final_equity"]
    assert res.n_trades == replay["trades"]


# ---------------------------------------------------------------------------
# Canned-frame helpers for the contract units (b)-(d)
# ---------------------------------------------------------------------------

CTICKERS = ["A01", "B02", "C03", "D04", "E05", "F06", "G07", "H08"]


def _canned_frame(order: list[str]) -> pd.DataFrame:
    """A frame whose ``_qv_order`` equals `order` (roe decays with rank)."""
    roe = {t: 0.5 - 0.01 * i for i, t in enumerate(order)}
    rows = {t: {"roe": roe.get(t, 0.001), "p_fcf": 20.0, "mcap": 1e10,
                "dollar_vol": 1e8, "mom": 0.1, "price": 100.0,
                "vol": 0.2, "eligible": True}
            for t in CTICKERS}
    return pd.DataFrame(rows).T


def _flat_tape(last_day: str,
               ) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Long flat history so the 253-session momentum warmup passes.
    dates = pd.bdate_range("2023-06-01", last_day)
    close = pd.DataFrame(100.0, index=dates, columns=CTICKERS)
    vol = pd.DataFrame(1e6, index=dates, columns=CTICKERS)
    return close, vol


async def _run_canned(monkeypatch, close, vol, frames, *, start, end=None,
                      open_px=None, **kw):
    monkeypatch.setattr(
        monthly_mod, "eligible_frame",
        lambda d, c, v, f: frames.get(pd.Timestamp(d).normalize()))
    return await optimize._daily_core_backtest(
        start, end, "diversified-plus", 1000.0, False,
        kw.pop("entry_gate", "none"), kw.pop("exit_cadence", "daily"),
        kw.pop("crash_pause", False), kw.pop("dca", "rank"),
        frame_cache=None, close_preloaded=(close, vol), fund_preloaded={},
        open_preloaded=close if open_px is None else open_px,
        cost_oneway=kw.pop("cost_oneway", 0.0),
        with_baseline=False, **kw)


@pytest.fixture
def small_slots(monkeypatch):
    monkeypatch.setattr(settings, "sim_monthly_target_n", 3)
    monkeypatch.setattr(settings, "sim_monthly_hold_band", 6)


STABLE = CTICKERS


# ---------------------------------------------------------------------------
# (b) The fill price is the session's strict OPEN, not its close
# ---------------------------------------------------------------------------

async def test_fill_price_is_the_session_open_not_the_close(
        small_slots, monkeypatch):
    """Every booked leg prices at that SESSION's open. The tape's opens gap
    0.1-0.5% below its closes, so a close fill (or a previous session's open)
    cannot pass."""
    close, vol = _flat_tape("2025-02-28")
    # a distinct open per session: 0.1%..0.5% below the (flat) close
    gaps = 1.0 - 0.001 * ((np.arange(len(close)) % 5) + 1)
    open_px = close.mul(gaps, axis=0)
    frames = {pd.Timestamp("2024-12-31"): _canned_frame(STABLE),
              pd.Timestamp("2025-01-31"): _canned_frame(STABLE),
              pd.Timestamp("2025-02-28"): _canned_frame(STABLE)}

    res = await _run_canned(monkeypatch, close, vol, frames,
                            start="2025-01-15", open_px=open_px)
    assert res is not None
    buys = [t for t in res.trades if t["side"] == "BUY"]
    assert buys, "no fills booked"
    for t in buys:
        d = pd.Timestamp(t["date"])
        assert t["price"] == pytest.approx(float(open_px.at[d, t["ticker"]]))
        assert t["price"] == pytest.approx(t["notional"] / t["shares"])
        # ... and NOT the session's close (nor the previous session's open)
        assert t["price"] != pytest.approx(float(close.at[d, t["ticker"]]))
        prev_d = close.index[close.index.get_loc(d) - 1]
        assert t["price"] != pytest.approx(float(open_px.at[prev_d, t["ticker"]]))

    # the first fill is the window's first session — it deploys against the
    # PRE-window (Dec-31) frame — priced at that session's open
    assert buys[0]["date"] == "2025-01-15"
    assert buys[0]["price"] == pytest.approx(float(open_px.at["2025-01-15", "A01"]))


# ---------------------------------------------------------------------------
# (c) The decision reads data <= S: F's own bar changes nothing until next day
# ---------------------------------------------------------------------------

async def test_mutating_fs_own_bar_changes_nothing_until_the_next_session(
        small_slots, monkeypatch):
    """A -20% print on session X in every held name breaches a 5%
    portfolio-stop on the marks. Measured on S's close (the contract's
    stop-style overlay rule), it must NOT cash out on X: the standing order
    fills at X+1's open. If the replay read F's own bar, X would show the
    sell."""
    close, vol = _flat_tape("2025-02-28")
    frames = {pd.Timestamp("2024-12-31"): _canned_frame(STABLE),
              pd.Timestamp("2025-01-31"): _canned_frame(STABLE),
              pd.Timestamp("2025-02-28"): _canned_frame(STABLE)}
    x = "2025-02-10"

    base = await _run_canned(monkeypatch, close, vol, frames,
                             start="2025-01-15", portfolio_stop_pct=0.05)
    assert base is not None

    # Mutated tape: the close CRASHES on X (the open on X is untouched, so the
    # fill price on X cannot explain anything), and the next session gaps down.
    close_m = close.copy()
    close_m.loc[x:, :] = 80.0
    open_m = close.copy()          # opens stay at 100 on X ...
    open_m.loc["2025-02-11":, :] = 80.0   # ... and gap down from X+1
    mut = await _run_canned(monkeypatch, close_m, vol, frames,
                            start="2025-01-15", open_px=open_m,
                            portfolio_stop_pct=0.05)
    assert mut is not None

    # nothing changes on or before X (the decision at X is taken on S's data)
    assert [t for t in mut.trades if t["date"] <= x] == \
        [t for t in base.trades if t["date"] <= x]
    assert not any(t["side"] == "SELL" and t["date"] == x for t in mut.trades)
    # ... and the crash becomes visible exactly one session later
    assert not any(t["side"] == "SELL" for t in base.trades)
    sells = [t for t in mut.trades if t["side"] == "SELL"]
    assert sells, "the drawdown never cashed out"
    assert {t["date"] for t in sells} == {"2025-02-11"}
    assert {t["ticker"] for t in sells} == {"A01", "B02", "C03"}
    for t in sells:
        assert t["price"] == pytest.approx(80.0)  # X+1's open


# ---------------------------------------------------------------------------
# (d) A missing session bar is no fill; the leg retries next session
# ---------------------------------------------------------------------------

async def test_missing_session_bar_skips_the_fill_and_retries(
        small_slots, monkeypatch):
    """B02 has no stored bar on the deferred rebuild session (a delisted /
    missing tape segment): its release cannot fill — a missing quote is not
    an executable price — so B02 keeps occupying its slot (no replacement
    grows the book past 3) and the release retries on the NEXT session."""
    close, vol = _flat_tape("2025-02-14")
    rotated = ["D04", "E05", "F06", "G07", "H08", "A01", "B02", "C03"]
    frames = {pd.Timestamp("2024-12-31"): _canned_frame(STABLE),
              pd.Timestamp("2025-01-31"): _canned_frame(rotated)}
    fill_day = pd.Timestamp("2025-02-03")    # first session after Jan-31
    # the missing segment: no bar at all for B02 on the fill session
    close.loc[fill_day, "B02"] = np.nan
    open_px = close.copy()

    res = await _run_canned(monkeypatch, close, vol, frames,
                            start="2025-01-15", end="2025-02-14",
                            open_px=open_px)
    assert res is not None
    rows = [(t["date"], t["side"], t["ticker"]) for t in res.trades]

    day1 = [r for r in rows if r[0] == "2025-02-03"]
    assert ("2025-02-03", "SELL", "C03") in day1     # the other release fills
    assert ("2025-02-03", "SELL", "B02") not in day1  # no bar -> no fill
    # B02 still occupies its slot: the plan's two replacement fills cannot
    # both land, and the book never exceeds the 3-slot target
    assert ("2025-02-03", "BUY", "D04") in day1
    assert ("2025-02-03", "BUY", "E05") not in day1
    held: set[str] = set()
    for (_, side, t) in rows:
        if side == "BUY":
            held.add(t)
        else:
            held.discard(t)
        assert len(held) <= 3
    # the retry lands on the next session, at THAT session's open, and the
    # freed slot is then filled
    assert ("2025-02-04", "SELL", "B02") in rows
    assert ("2025-02-04", "BUY", "E05") in rows
    b02_sell = next(t for t in res.trades
                    if t["side"] == "SELL" and t["ticker"] == "B02")
    assert b02_sell["date"] == "2025-02-04"
    assert b02_sell["price"] == pytest.approx(
        float(open_px.at["2025-02-04", "B02"]))


# ---------------------------------------------------------------------------
# (e) The technical replay (optimize._replay) runs the same contract
# ---------------------------------------------------------------------------

def _trend_candles(n: int = 300, drift: float = 0.01,
                   seed: int = 7) -> list[dict]:
    """A clean single-name uptrend with a visible open/close gap on every
    bar (so a close fill can never be mistaken for an open fill)."""
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    price = 100.0
    start = pd.Timestamp("2025-01-02")
    for i in range(n):
        close = price * (1.0 + drift) * (1.0 + float(rng.normal(0.0, 0.002)))
        open_ = price * 0.998
        high = max(open_, close) * 1.001
        low = min(open_, close) * 0.999
        rows.append({
            "time": (start + pd.Timedelta(days=i)).strftime("%Y-%m-%d"),
            "open": round(open_, 4), "high": round(high, 4),
            "low": round(low, 4), "close": round(close, 4),
            "volume": 1_000_000.0,
        })
        price = close
    return rows


def _contract_series(rows_by_ticker: dict[str, list[dict]]) -> dict[str, pd.DataFrame]:
    """The CONTRACT-mode series shape ``optimize._load_series`` produces:
    the indicator frame plus the strict session bars (open/high/low)."""
    series: dict[str, pd.DataFrame] = {}
    for t, rows in rows_by_ticker.items():
        ser = optimize._signal_series(rows).copy()
        for col in ("open", "high", "low"):
            ser[col] = [r[col] for r in rows]
        series[t] = ser
    return series


def _replay_trades(rows: list[dict]) -> optimize.ReplayResult:
    params = optimize.ReplayParams(start_cash=10_000.0, monthly_allowance=0.0,
                                   use_atr_stop=False, stop_type="none",
                                   trailing_stop_pct=0.0)
    return optimize._replay(_contract_series({"UP": rows}), params,
                            start=rows[60]["time"], end=rows[-1]["time"])


def test_replay_contract_mode_fills_at_the_strict_open():
    """``_replay`` on series that carry session bars fills at F's strict
    exact-date OPEN — the same rule the live sim and the daily-core replay
    use (the close would be the pre-contract convention)."""
    rows = _trend_candles()
    res = _replay_trades(rows)
    buys = [t for t in res.trades if t["side"] == "BUY"]
    assert buys, "the uptrend produced no BUY signal"
    by_date = {r["time"]: r for r in rows}
    for t in buys:
        bar = by_date[t["date"]]
        assert t["price"] == pytest.approx(bar["open"])
        assert t["price"] != pytest.approx(bar["close"])


def test_replay_decision_ignores_fs_own_bar():
    """Rule 1 in the technical replay: the indicator row read on session F is
    S's, so a wild move in F's own bar changes the marks (and therefore the
    equity curve) but not a single trade."""
    rows = _trend_candles()
    mutated = [dict(r) for r in rows]
    mutated[-1]["close"] *= 1.6
    mutated[-1]["high"] *= 1.6

    base = _replay_trades(rows)
    mut = _replay_trades(mutated)

    assert base.trades == mut.trades
    assert base.equity_curve[:-1] == mut.equity_curve[:-1]
    # the mutation IS material — it moves the final mark, not the decision
    assert base.position_count_curve[-1] > 0
    assert mut.equity_curve[-1]["equity"] > base.equity_curve[-1]["equity"]


def test_standing_stop_fill_rule():
    """Contract rule 4: a level known at S's close is a standing order for F
    — fill at F's open when the open is already through it, else AT the level
    when F's low breaches it, else no fill. No F bar = no fill."""
    assert optimize._standing_stop_fill({"open": 80.0, "low": 79.0}, 90.0,
                                        None) == 80.0
    assert optimize._standing_stop_fill({"open": 95.0, "low": 85.0}, 90.0,
                                        None) == 90.0
    assert optimize._standing_stop_fill({"open": 95.0, "low": 92.0}, 90.0,
                                        None) is None
    assert optimize._standing_stop_fill(None, 90.0, None) is None
