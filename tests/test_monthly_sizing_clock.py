"""Monthly order-SIZING clock causality (session fill contract rules 1/3).

A monthly rebuild must size its orders on marks known at S — the previous
COMPLETED session — never on F's own close. F's close is an end-of-session
REPORTING input only (the post-trade valuation/snapshot).

Every test here pins that with a mutation on a tape where the F open and all
data <= S are fixed:

  * the ONLY mutated cell is the F close (and F FX close) of a HELD name;
  * the F-day trade tuples must be identical — live AND persisted replay —
    including an out-of-band SELL, a top-up and its replacements;
  * the end-of-session valuation/snapshot MUST differ (reporting keeps using
    F's close, so the mutation is visible exactly where it belongs);
  * mutating the S close DOES change the sizing, and mutating the F close
    does not: S is the sizing clock — not the month-end decision date and not
    F (the catch-up tests drive a session where S != the month-end);
  * no future FX/data is read: an F-only FX-rate + close jump changes neither
    the sizing nor the fill price.

OHLC stays legal in every variant: ``high = max(open, close)`` and
``low = min(open, close)`` per cell, so a mutated close is a valid bar.
"""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import StaticPool, select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

import app.db as db_mod
from app import monthly
from app.db import Base, Candle, MonthlyAccount, MonthlyPosition, MonthlySnapshot, MonthlyTrade

# ---------------------------------------------------------------------------
# Fixture tape / scenario constants
# ---------------------------------------------------------------------------

STOCKS = ["A01", "B02", "C03", "D04"]
PAIR = "EURUSD=X"
DAYS = pd.bdate_range("2024-10-01", "2025-02-28")

DEC_ME = pd.Timestamp("2024-12-31")   # January's decision month-end
FEB_ME = pd.Timestamp("2025-01-31")   # February's decision month-end
F1 = pd.Timestamp("2025-02-03")       # first session after FEB_ME; S == FEB_ME
F2 = pd.Timestamp("2025-02-04")       # catch-up session; S == F1 != FEB_ME

CLOCK1 = datetime(2025, 2, 3, 12, 0, tzinfo=UTC)
CLOCK2 = datetime(2025, 2, 4, 12, 0, tzinfo=UTC)
CLOCK_BACKFILL = datetime(2025, 2, 4, 12, 0, tzinfo=UTC)

JAN_PICKS = ["A01", "D04"]                     # bought at 2025-01-02
FEB_PICKS = ["A01", "B02", "C03"]              # top-up A01, SELL D04, add B02/C03
PICKS_BY_MONTH = {12: JAN_PICKS, 1: FEB_PICKS}

CASH = 1000.0                                   # funded account (Feb deposited)
BOOK = [("A01", 5.0, 100.0), ("D04", 5.0, 100.0)]   # in-band top-up + out-of-band
MUT = 140.0                                     # the F-close-only mutation

# The S-sized legacy trade set for F1/F2: SELL the out-of-band name, top up
# A01 to the 1/N weight, open the two replacements. Values come from
# total equity 2000 / 3 = 666.666..., with A01 already worth 500.
F1_TRADES = [
    ("SELL", "D04", 5.0, 100.0),
    ("BUY", "A01", 1.6666, 100.0),
    ("BUY", "B02", 6.6666, 100.0),
    ("BUY", "C03", 6.6666, 100.0),
]


def _canned_frame(tickers=STOCKS) -> pd.DataFrame:
    return pd.DataFrame(
        {t: {"roe": 0.5 - 0.01 * i, "p_fcf": 20.0, "mcap": 1e10,
             "dollar_vol": 1e8, "mom": 0.1, "price": 100.0, "vol": 0.2,
             "eligible": True}
         for i, t in enumerate(tickers)}).T


def _bars(days, tickers=STOCKS, *, base=None, closes=None, opens=None):
    """(open, close, high, low, vol) frames with legal OHLC.

    ``base`` maps ticker -> constant price (default 100); ``closes``/``opens``
    are {(date, ticker): price} overrides. high/low are recomputed from the
    final open/close, so the mutated bar is valid too.
    """
    base = base or {}
    close = pd.DataFrame({t: float(base.get(t, 100.0)) for t in tickers}, index=days)
    open_px = close.copy()
    for (d, t), v in (closes or {}).items():
        close.loc[pd.Timestamp(d), t] = v
    for (d, t), v in (opens or {}).items():
        open_px.loc[pd.Timestamp(d), t] = v
    high = pd.DataFrame(np.maximum(open_px.to_numpy(), close.to_numpy()),
                        index=days, columns=tickers)
    low = pd.DataFrame(np.minimum(open_px.to_numpy(), close.to_numpy()),
                       index=days, columns=tickers)
    vol = pd.DataFrame(1e6, index=days, columns=tickers)
    return open_px, close, high, low, vol


async def _new_db(monkeypatch):
    """Fresh in-memory SQLite for ONE engine run (a real async DB, no files)."""
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    sf = async_sessionmaker(engine, expire_on_commit=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    monkeypatch.setattr(monthly, "Session", sf)
    monkeypatch.setattr(db_mod, "Session", sf)
    return engine, sf


async def _seed_candles(sf, bars, tickers=STOCKS) -> None:
    open_px, close, high, low, vol = bars
    async with sf() as s:
        await s.execute(
            Candle.__table__.insert(),
            [{"ticker": t, "timestamp": d.to_pydatetime(),
              "open": float(open_px.at[d, t]), "high": float(high.at[d, t]),
              "low": float(low.at[d, t]), "close": float(close.at[d, t]),
              "volume": float(vol.at[d, t])}
             for d in open_px.index for t in tickers])
        await s.commit()


def _patch_engine(monkeypatch, *, bars, picks_by_month=None,
                  tickers=STOCKS, frames=None) -> None:
    """Patch the non-decision I/O: no network, canned ranking, fixed frames."""
    open_px, close, _high, _low, vol = bars
    picks_by_month = picks_by_month or PICKS_BY_MONTH
    frames = frames if frames is not None else {FEB_ME: _canned_frame(tickers)}

    async def fake_refresh(_tickers):
        return [], {}

    async def fake_load_frames(_tickers, _asof=None, start=None):
        return close, vol

    async def fake_fund(_tickers):
        return {"Synthetic": {"tag": []}}

    def fake_pick(rebal_date, _close, _vol, _fund, holdings=None, frame=None):
        return picks_by_month[pd.Timestamp(rebal_date).month], frame

    monkeypatch.setattr(monthly, "refresh_data", fake_refresh)
    monkeypatch.setattr(monthly, "load_frames", fake_load_frames)
    monkeypatch.setattr(monthly.fundamentals_mod, "load_fundamentals", fake_fund)
    monkeypatch.setattr(
        monthly, "eligible_frame",
        lambda d, _c, _v, _f: frames.get(pd.Timestamp(d).normalize()))
    monkeypatch.setattr(monthly, "pick_portfolio", fake_pick)
    monkeypatch.setattr(monthly, "universe_tickers", lambda _u: list(tickers))
    monkeypatch.setattr(monthly, "_strategy_universe", lambda: "synthetic")


async def _read_book(sf):
    async with sf() as s:
        trades = (await s.scalars(
            select(MonthlyTrade).order_by(MonthlyTrade.id))).all()
        snaps = (await s.scalars(
            select(MonthlySnapshot).order_by(MonthlySnapshot.id))).all()
        cash = (await s.get(MonthlyAccount, 1)).cash
    tuples = [(r.created_at.strftime("%Y-%m-%d"), r.side, r.ticker,
               round(float(r.shares), 6), round(float(r.price), 6))
              for r in trades]
    return tuples, snaps, cash


async def _live_pass(monkeypatch, *, clock, bars, force=True,
                     positions=BOOK, cash=CASH, picks_by_month=None,
                     tickers=STOCKS):
    """One LIVE monthly rebuild pass on a fresh temp DB."""
    engine, sf = await _new_db(monkeypatch)
    try:
        await _seed_candles(sf, bars, tickers)
        async with sf() as s:
            s.add(MonthlyAccount(id=1, cash=cash, last_allowance_month="2025-02"))
            for t, shares, px in positions:
                s.add(MonthlyPosition(ticker=t, shares=shares, avg_cost=px,
                                      opened_at=datetime(2025, 1, 15)))
            await s.commit()
        monkeypatch.setattr(monthly, "_now", lambda: clock)
        _patch_engine(monkeypatch, bars=bars, picks_by_month=picks_by_month,
                      tickers=tickers)
        out = await monthly._run_rebalance_locked(force=force, now=clock)
        dated, snaps, cash_after = await _read_book(sf)
    finally:
        await engine.dispose()
    # live trades stamp the real write clock, so they carry no useful session
    # date; the DB is fresh per run, so these ARE this pass's trades.
    return out, [t[1:] for t in dated], snaps, cash_after


async def _backfill_run(monkeypatch, *, bars, start="2025-01-02",
                        frames=None, picks_by_month=None, tickers=STOCKS):
    """One persisted monthly replay on a fresh temp DB (preloaded frames)."""
    engine, sf = await _new_db(monkeypatch)
    try:
        monkeypatch.setattr(monthly, "_now", lambda: CLOCK_BACKFILL)
        _patch_engine(monkeypatch, bars=bars, picks_by_month=picks_by_month,
                      tickers=tickers, frames=frames)
        preload = monthly.BackfillPreload(
            tickers=list(tickers), fund={"Synthetic": {"tag": []}},
            open=bars[0], close=bars[1], vol=bars[4],
            frames=dict(frames or {DEC_ME: _canned_frame(tickers),
                                   FEB_ME: _canned_frame(tickers)}))
        out = await monthly.backfill(start=start, preload=preload)
        trades, snaps, cash_after = await _read_book(sf)
    finally:
        await engine.dispose()
    return out, trades, snaps, cash_after


def _replay_day(trades, day: pd.Timestamp):
    """The replay's trade tuples of one fill session (dated by created_at)."""
    return [t for t in trades if t[0] == day.strftime("%Y-%m-%d")]


def _snap_on(snaps, day: pd.Timestamp):
    """The one persisted snapshot of a session (the replay stamps history)."""
    hits = [s for s in snaps if s.created_at.date() == day.date()]
    assert len(hits) == 1, f"expected one snapshot on {day.date()}, got {len(hits)}"
    return hits[0]


def _only_snap(snaps):
    """The single snapshot a live pass writes (live stamps the real clock)."""
    assert len(snaps) == 1, f"expected one live snapshot, got {len(snaps)}"
    return snaps[0]


# ---------------------------------------------------------------------------
# LIVE: the sizing marks are S-known, F's close only reports
# ---------------------------------------------------------------------------

async def test_live_sizing_ignores_the_f_close_of_a_held_name(monkeypatch):
    """F-close-only mutation of a HELD name (100 -> 140) at an unchanged F
    open: the whole F-day trade set — SELL, top-up, replacements — must be
    identical; the end-of-session valuation and snapshot must differ."""
    days = DAYS[DAYS <= F1]                       # the session stream ends at F
    base = _bars(days)
    mut = _bars(days, closes={(F1, "A01"): MUT})  # ONLY F's close changes
    # the mutated tape is still a legal OHLC series (high >= open/close,
    # low <= open/close for EVERY bar, including the 140 close)
    assert (mut[2].to_numpy() >= np.maximum(mut[0].to_numpy(),
                                            mut[1].to_numpy())).all()
    assert (mut[3].to_numpy() <= np.minimum(mut[0].to_numpy(),
                                            mut[1].to_numpy())).all()
    assert mut[2].max().max() >= MUT

    base_out, base_tr, base_snaps, base_cash = await _live_pass(
        monkeypatch, clock=CLOCK1, bars=base)
    mut_out, mut_tr, mut_snaps, mut_cash = await _live_pass(
        monkeypatch, clock=CLOCK1, bars=mut)

    assert base_tr == F1_TRADES
    assert mut_tr == base_tr, "F's close leaked into the order sizing"
    assert base_cash == mut_cash

    # ...while the END-OF-SESSION report books F's close (rule 3)
    base_val = base_out["valuation"]
    mut_val = mut_out["valuation"]
    assert mut_val["positions_value"] > base_val["positions_value"]
    base_snap = _only_snap(base_snaps)
    mut_snap = _only_snap(mut_snaps)
    assert mut_snap.total_equity > base_snap.total_equity


async def test_live_normal_path_sizes_on_previous_session_not_month_end(monkeypatch):
    """The normal (_pending_rebuild) path at a CATCH-UP session: F = 2025-02-04,
    S = 2025-02-03, decision month-end = 2025-01-31. Mutating F's close must
    not move a single share; mutating S's close MUST (S is the sizing clock —
    neither the month-end nor F is)."""
    days = DAYS[DAYS <= F2]
    base = _bars(days)
    f_mut = _bars(days, closes={(F2, "A01"): MUT})   # F-only mutation
    s_mut = _bars(days, closes={(F1, "A01"): MUT})   # S-known mutation

    base_out, base_tr, _base_snaps, _ = await _live_pass(
        monkeypatch, clock=CLOCK2, bars=base, force=False)
    assert base_out.get("rebalanced") is True
    assert base_out["month"] == "2025-01"        # decision: Jan month-end
    assert base_tr == F1_TRADES

    _out, f_tr, _snaps, _ = await _live_pass(
        monkeypatch, clock=CLOCK2, bars=f_mut, force=False)
    assert f_tr == base_tr, "F's close leaked into the order sizing"

    _out, s_tr, _snaps, _ = await _live_pass(
        monkeypatch, clock=CLOCK2, bars=s_mut, force=False)
    assert s_tr != base_tr, "sizing ignored the S-known close"
    # the fill prices never move: only the SIZING clock is under test
    assert {t[3] for t in s_tr} == {100.0}


async def test_live_sizing_ignores_f_close_and_f_fx(monkeypatch):
    """A foreign holding marked through its FX pair: an F-only jump of BOTH
    the local close and the pair close must leave the sizing untouched (no
    future FX/data), while the end-of-session valuation reflects it."""
    tickers = ["SAP.DE", "B02", "C03", "D04", PAIR]
    days = DAYS[DAYS <= F1]
    base_kw = {"base": {PAIR: 1.1}}
    base = _bars(days, tickers, **base_kw)
    mut = _bars(days, tickers,
                closes={(F1, "SAP.DE"): MUT, (F1, PAIR): 2.0}, **base_kw)
    picks = {12: ["SAP.DE", "D04"], 1: ["SAP.DE", "B02", "C03"]}
    book = [("SAP.DE", 5.0, 110.0), ("D04", 5.0, 100.0)]

    base_out, base_tr, _s, _ = await _live_pass(
        monkeypatch, clock=CLOCK1, bars=base, positions=book, tickers=tickers,
        picks_by_month=picks)
    mut_out, mut_tr, _s, _ = await _live_pass(
        monkeypatch, clock=CLOCK1, bars=mut, positions=book, tickers=tickers,
        picks_by_month=picks)

    # S-known USD marks: SAP.DE 100 * 1.10 = 110 (both variants)
    assert base_tr == [
        ("SELL", "D04", 5.0, 100.0),
        ("BUY", "SAP.DE", 1.2121, 110.0),
        ("BUY", "B02", 6.8333, 100.0),
        ("BUY", "C03", 6.8333, 100.0),
    ]
    assert mut_tr == base_tr, "F's FX/close leaked into the order sizing"
    # ...and the F-close mark (140 * 2.0 = 280) shows up ONLY in the report
    assert mut_out["valuation"]["positions_value"] > \
        base_out["valuation"]["positions_value"]


async def test_mark_lookup_at_s_cannot_see_f(monkeypatch):
    """Direct clock check on the live mark loader: the S probe is byte-blind
    to F's close and F's FX rate; the F probe sees both."""
    tickers = ["SAP.DE", PAIR]
    days = DAYS[DAYS <= F1]
    mut = _bars(days, tickers,
                base={PAIR: 1.1},
                closes={(F1, "SAP.DE"): MUT, (F1, PAIR): 2.0})
    monkeypatch.setattr(monthly, "_now", lambda: CLOCK1)
    engine, sf = await _new_db(monkeypatch)
    try:
        await _seed_candles(sf, mut, tickers)
        at_s = await monthly._price_usd_map(["SAP.DE"], asof=F1 - pd.Timedelta(days=1))
        at_f = await monthly._price_usd_map(["SAP.DE"], asof=F1)
    finally:
        await engine.dispose()
    assert at_s["SAP.DE"] == pytest.approx(110.0)   # 100 * 1.10, S-known
    assert at_f["SAP.DE"] == pytest.approx(280.0)   # 140 * 2.00, F report


# ---------------------------------------------------------------------------
# Persisted replay: the same clock discipline
# ---------------------------------------------------------------------------

async def test_backfill_sizing_ignores_the_f_close_of_a_held_name(monkeypatch):
    """The persisted replay sizes on the previous completed stored session's
    close: an F-only close mutation of a held name must not change the F-day
    trades (SELL D04 + top-up A01 + replacements), while the F-day snapshot
    is still valued at F's close."""
    days = DAYS[DAYS <= F1]
    base = _bars(days)
    mut = _bars(days, closes={(F1, "A01"): MUT})

    base_out, base_all, base_snaps, _ = await _backfill_run(monkeypatch, bars=base)
    mut_out, mut_all, mut_snaps, _ = await _backfill_run(monkeypatch, bars=mut)
    assert base_out["ok"] is True and mut_out["ok"] is True

    # January's decision bought the seed book: A01 5 + D04 5 at 100.
    jan = [t for t in base_all if t[0] == "2025-01-02"]
    assert jan == [("2025-01-02", "BUY", "A01", 5.0, 100.0),
                   ("2025-01-02", "BUY", "D04", 5.0, 100.0)]

    base_f = [t[1:] for t in _replay_day(base_all, F1)]
    mut_f = [t[1:] for t in _replay_day(mut_all, F1)]
    assert base_f == F1_TRADES
    assert mut_f == base_f, "F's close leaked into the replay's order sizing"

    # the replay's END-OF-SESSION snapshot still marks F's close
    assert _snap_on(mut_snaps, F1).total_equity > \
        _snap_on(base_snaps, F1).total_equity


async def test_backfill_catch_up_sizes_on_previous_session_not_month_end(monkeypatch):
    """Replay catch-up: B02 has no executable F1 open, so the Feb decision
    retries on F2 (S = F1, month-end = 2025-01-31). An F2-only close mutation
    must not move the retry's trades; an F1 (S-known) mutation MUST."""
    days = DAYS[DAYS <= F2]
    missing_f1_open = {(F1, "B02"): np.nan}   # leg unfillable on F1 -> retry
    base = _bars(days, opens=missing_f1_open)
    f_mut = _bars(days, opens=missing_f1_open, closes={(F2, "A01"): MUT})
    s_mut = _bars(days, opens=missing_f1_open, closes={(F1, "A01"): MUT})

    _out, base_all, _s, _ = await _backfill_run(monkeypatch, bars=base)
    _out, f_all, _s, _ = await _backfill_run(monkeypatch, bars=f_mut)
    _out, s_all, _s, _ = await _backfill_run(monkeypatch, bars=s_mut)

    base_f1 = [t[1:] for t in _replay_day(base_all, F1)]
    base_f2 = [t[1:] for t in _replay_day(base_all, F2)]
    # F1 is incomplete: B02 could not fill, so only D04/A01/C03 traded then.
    assert [t[0] for t in base_f1] == ["SELL", "BUY", "BUY"]
    assert set(t[1] for t in base_f1) == {"D04", "A01", "C03"}
    # ...and the retry on F2 completes the membership.
    assert base_f2 and all(t[0] == "BUY" and t[1] == "B02" for t in base_f2)

    assert [t[1:] for t in _replay_day(f_all, F2)] == base_f2
    assert [t[1:] for t in _replay_day(s_all, F2)] != base_f2


# ---------------------------------------------------------------------------
# Cross-path: live and the persisted replay agree on the S-sized trade set
# ---------------------------------------------------------------------------

async def test_live_and_backfill_agree_under_the_f_close_mutation(monkeypatch):
    """Seed the live book to exactly the replay's end-of-January state, then
    drive both engines through the same F1 pass. Their F-day trade tuples and
    end cash must match — in the base tape AND under the F-close mutation."""
    days = DAYS[DAYS <= F1]
    base = _bars(days)
    mut = _bars(days, closes={(F1, "A01"): MUT})

    for bars in (base, mut):
        live_out, live_tr, live_snaps, live_cash = await _live_pass(
            monkeypatch, clock=CLOCK1, bars=bars)
        rep_out, rep_all, rep_snaps, rep_cash = await _backfill_run(
            monkeypatch, bars=bars)
        rep_f = [t[1:] for t in _replay_day(rep_all, F1)]

        assert live_tr == rep_f == F1_TRADES
        assert live_cash == rep_cash           # identical zero-fee cash trail
        # live reports the post-trade valuation to the cent; the replay stores
        # the raw end-of-session equity — same book, same F-close mark.
        assert _only_snap(live_snaps).total_equity == pytest.approx(
            _snap_on(rep_snaps, F1).total_equity, abs=0.01)
        assert live_out["valuation"]["total_equity"] == pytest.approx(
            _snap_on(rep_snaps, F1).total_equity, abs=0.01)
        assert rep_out["ok"] is True
