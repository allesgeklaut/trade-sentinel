"""GOLDEN cadence parity (app/session.py contract): the LIVE deployment
(injectable ``now``) vs the persisted backfill replay must produce identical
trade sets on the identical ordered session stream, fee-free.

The test seeds ONE synthetic 8-ticker tape (daily candles, monthly
month-ends on the true US session calendar) into a temp SQLite DB, then:

1. drives the LIVE engines session by session — ``monthly.deposit_allowance``
   + ``monthly.run_monthly_cycle`` + ``daily_core.run_daily_cycle`` — with
   ``monthly._now()``/``daily_core._now()`` pinned to each session's
   post-close instant (21:00 UTC; the Vienna allowance anchor keeps the
   session's own calendar date, so deposit months align with the replay's),
   and captures every live trade row in stream order;
2. proves the FORWARD-RECORD GUARD: a replay inside the live (epoch 1) book
   is REFUSED before any mutation, and the live trade/position rows are
   unchanged, bit-for-bit, by the refusal;
3. explicitly resets BOTH engines (the documented "reset first" path): each
   epoch closes at 2, the live paper rows survive in epoch 1 untouched and
   only the current positions/state are cleared;
4. replays the SAME candles through ``monthly.backfill`` /
   ``daily_core.backfill`` IN EPOCH 2 — which stamp the historical fill days
   — captures ONLY the epoch-2 replay trade rows, and verifies the closed
   epoch-1 rows are still bit-for-bit intact after the replay;
5. asserts the live and replay trade sets are identical, the end-of-stream
   positions match (ticker, shares, avg_cost), and the replay's ``opened_at``
   equals each position's actual first-buy date.

The tape's intra-month wiggle makes the DAILY-ranked portfolio differ from
the month-end frame's picks — the parity itself proves the month-end band
policy: the replay ranks ONLY on completed month-ends, so a live engine that
leaked daily candles into its ranking could not match it. The test also
asserts directly that a mid-month frame's ranking differs from the
month-end frame's, and that both engines' band releases land on the FIRST
session after the month-end that changed the band.
"""

from __future__ import annotations

from datetime import UTC

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import StaticPool, select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app import daily_core, monthly, run_config
from app import fundamentals as fundamentals_mod
from app import sim as sim_mod
from app.config import settings
from app.db import Base, Candle, MonthlyAccount

TICKERS = ["T01", "T02", "T03", "T04", "T05", "T06", "T07", "T08"]

TAPE_START = "2024-06-03"
TAPE_END = "2026-10-02"

# per-session driver clock: post-US-close, but early enough that the Vienna
# allowance anchor (UTC+1/+2) still names the SESSION's own date — otherwise
# the live engine would deposit one session before the replay does.
DRIVE_HOUR = 21


# ---------------------------------------------------------------------------
# Deterministic tape
# ---------------------------------------------------------------------------

def _tape() -> tuple[pd.DatetimeIndex, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Sessions, strict OPEN frame, CLOSE frame, volume frame.

    Piecewise paths (regime shifts at fixed dates) create REAL rank flips at
    month-ends: T07 surges from 2025-12-31 (overtakes the flat names),
    T08 sinks from 2026-01-30 and T03 from 2026-04-01 (fall out of the
    band -> releases). Every session closes with a ±3% sine wiggle so the
    DAILY-ranked data would order differently than the month-end frame.
    """
    dates = pd.bdate_range(TAPE_START, TAPE_END)
    sessions = pd.DatetimeIndex(
        [d for d in dates if sim_mod.is_trading_day(d.to_pydatetime())])
    base = pd.DataFrame(100.0, index=sessions, columns=TICKERS)
    n = len(sessions)

    def _regime(ticker: str, start: str, factor: float) -> None:
        mask = sessions >= pd.Timestamp(start)
        steps = np.arange(1, int(mask.sum()) + 1)
        base.loc[mask, ticker] = 100.0 * factor ** steps

    _regime("T07", "2025-12-31", 1.005)   # momentum takeover
    _regime("T08", "2026-01-30", 0.995)   # drops below the flat names
    _regime("T03", "2026-04-01", 0.99)    # held name falls out of the band

    # Per-ticker phase shift: a COMMON wiggle would multiply every ticker by
    # the same daily factor, cancelling out of 12-1 momentum comparisons —
    # the flat names would stay roe-ordered all month and a mid-month frame
    # could never rank differently than the month-end frame (the fixture
    # sanity check below would fail). The phase shift makes each ticker's
    # own daily path (and thus its sliding-window momentum) distinct.
    phase = 0.7 * np.arange(len(TICKERS))
    wig = 1.0 + 0.03 * np.sin(
        2 * np.pi * np.arange(n)[:, None] / 9.0 + phase[None, :])
    close = base.mul(pd.DataFrame(wig, index=sessions, columns=TICKERS))
    open_px = close.mul(1.0 - 0.002 * (np.arange(n) % 5) / 5.0, axis=0)
    vol = pd.DataFrame(1e6, index=sessions, columns=TICKERS)
    return sessions, open_px, close, vol


def _fund_all_positive() -> dict[str, dict[str, list[dict]]]:
    """Facts public well before the first decision month-end: equity 5e9,
    FY2024 NI decaying with the ticker index (roe_rank decides the flat
    phase), 1e8 shares. mcap 1e10 > min, dollar_vol 1e8 > min."""

    def _roe(t: str) -> float:
        return 0.5 - 0.01 * TICKERS.index(t) * 10  # T01 0.50 ... T08 0.43
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
    """Seed the tape into the shared temp DB and pin every clock to the
    session driver. Returns (clock, sessions, open_px, close, vol, fund)."""
    sessions, open_px, close, vol = _tape()
    fund = _fund_all_positive()

    async with mem_db() as s:
        await s.execute(
            Candle.__table__.insert(),
            [{"ticker": t, "timestamp": d.to_pydatetime(), "open": float(open_px.at[d, t]),
              "high": float(max(open_px.at[d, t], close.at[d, t])),
              "low": float(min(open_px.at[d, t], close.at[d, t])),
              "close": float(close.at[d, t]), "volume": 1e6}
             for d in sessions for t in TICKERS])
        await s.commit()

    clock = {"now": sessions[0].to_pydatetime().replace(hour=DRIVE_HOUR,
                                                        tzinfo=UTC)}

    def fake_now():
        return clock["now"]

    async def monthly_refresh(_tickers):
        return [], {}

    async def daily_refresh():
        return [], []

    async def fake_fund(_tickers):
        return fund

    monkeypatch.setattr(monthly, "_now", fake_now)
    monkeypatch.setattr(daily_core, "_now", fake_now)
    monkeypatch.setattr(monthly, "refresh_data", monthly_refresh)
    monkeypatch.setattr(daily_core, "refresh_data", daily_refresh)
    monkeypatch.setattr(fundamentals_mod, "load_fundamentals", fake_fund)
    monkeypatch.setattr(monthly, "universe_tickers", lambda _u: list(TICKERS))
    monkeypatch.setattr(daily_core, "universe_tickers", lambda _u: list(TICKERS))
    monkeypatch.setattr(settings, "sim_monthly_target_n", 3)
    monkeypatch.setattr(settings, "sim_monthly_hold_band", 6)
    monkeypatch.setattr(daily_core, "_STATE_FILE", tmp_path / "state.json")
    return clock, sessions, open_px, close, vol, fund


@pytest.fixture
async def mem_db(monkeypatch):
    """Shared in-memory SQLite DB for BOTH engines (one engine, one cache)."""
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    session_factory = async_sessionmaker(engine, expire_on_commit=False)

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    monkeypatch.setattr(daily_core, "Session", session_factory)
    monkeypatch.setattr(monthly, "Session", session_factory)
    import app.db as db_mod
    monkeypatch.setattr(db_mod, "Session", session_factory)

    async with session_factory() as s:
        s.add(MonthlyAccount(id=1))
        await s.commit()

    yield session_factory
    await engine.dispose()


# ---------------------------------------------------------------------------
# Capture helpers
# ---------------------------------------------------------------------------

async def _trade_rows(model, epoch: int | None = None) -> list[dict]:
    """Trade rows in stream order; ``epoch`` optionally scopes the dump (the
    default is the full history: closed epochs + the current one)."""
    async with daily_core.Session() as s:
        stmt = select(model).order_by(model.id)
        if epoch is not None:
            stmt = stmt.where(model.epoch == epoch)
        rows = (await s.scalars(stmt)).all()
    return [{"side": r.side, "ticker": r.ticker,
             "shares": round(r.shares, 6), "price": round(r.price, 6),
             "date": r.created_at.strftime("%Y-%m-%d")} for r in rows]


async def _table_rows(model, epoch: int | None = None) -> list[dict]:
    """Every mapped column of every *model* row, in id order.

    ``epoch`` scopes the dump to one epoch of a history table; the default is
    the full history. Used for bit-for-bit preservation checks: a reset or a
    replay may not rewrite (or drop) a single field of a CLOSED epoch's rows.
    """
    async with daily_core.Session() as s:
        stmt = select(model).order_by(model.id)
        if epoch is not None:
            stmt = stmt.where(model.epoch == epoch)
        rows = (await s.scalars(stmt)).all()
    cols = [c.name for c in model.__table__.columns]
    return [{c: getattr(r, c) for c in cols} for r in rows]


async def _position_rows(model) -> list[dict]:
    async with daily_core.Session() as s:
        rows = (await s.scalars(
            select(model).order_by(model.ticker))).all()
    return sorted(({"ticker": p.ticker, "shares": round(p.shares, 6),
                    "avg_cost": round(p.avg_cost, 6),
                    "opened_at": p.opened_at.strftime("%Y-%m-%d")
                    if p.opened_at else ""} for p in rows),
                  key=lambda d: d["ticker"])


def _key(tr: dict) -> tuple:
    return (tr["side"], tr["ticker"], tr["shares"], tr["price"])


# ---------------------------------------------------------------------------
# The golden test
# ---------------------------------------------------------------------------

async def test_live_deployment_matches_persisted_backfill(parity_env):

    clock, sessions, open_px, close, vol, fund = parity_env
    start = sessions[0].strftime("%Y-%m-%d")

    # --- 1) drive the LIVE engines over the ordered session stream ---
    for d in sessions:
        clock["now"] = d.to_pydatetime().replace(hour=DRIVE_HOUR, tzinfo=UTC)
        await monthly.deposit_allowance()
        await monthly.run_monthly_cycle(now=clock["now"])
        await daily_core.run_daily_cycle(force=False, now=clock["now"])

    live_monthly = await _trade_rows(monthly.MonthlyTrade)
    live_daily = await _trade_rows(daily_core.DailyCoreTrade)
    live_monthly_pos = await _position_rows(monthly.MonthlyPosition)
    live_daily_pos = await _position_rows(daily_core.DailyCorePosition)

    # the tape must actually exercise the contract: both engines traded,
    # releases happened (a held name fell out of the month-end band)
    assert live_monthly and live_daily
    assert any(t["side"] == "SELL" for t in live_monthly)
    assert any(t["side"] == "SELL" for t in live_daily)

    # The live book is the current (epoch 1) FORWARD record. These full-column
    # dumps are its baseline: neither the refused replay, nor the reset, nor
    # the epoch-2 replay may rewrite one field of it. Ranking is excluded on
    # purpose — it is a current cache (singleton), not history, and the reset
    # drops it by contract.
    history = {
        "monthly_trades": monthly.MonthlyTrade,
        "monthly_allowances": monthly.MonthlyAllowance,
        "monthly_snapshots": monthly.MonthlySnapshot,
        "monthly_rebalances": monthly.MonthlyRebalance,
        "daily_trades": daily_core.DailyCoreTrade,
        "daily_allowances": daily_core.DailyCoreAllowance,
        "daily_snapshots": daily_core.DailyCoreSnapshot,
    }
    live_hist = {name: await _table_rows(model, epoch=1)
                 for name, model in history.items()}
    positions = (monthly.MonthlyPosition, daily_core.DailyCorePosition)
    live_pos_dump = {model.__name__: await _table_rows(model)
                     for model in positions}

    # --- 2) the forward-record guard REFUSES the destructive replay ---
    # A replay regenerates the book's ORIGIN, so it must not run inside the
    # live paper epoch — and it must refuse BEFORE mutating anything.
    refused_m = await monthly.backfill(start=start)
    refused_d = await daily_core.backfill(start=start)
    for refused, engine in ((refused_m, "monthly"), (refused_d, "daily-core")):
        assert refused["ok"] is False, (engine, refused)
        assert "forward paper record" in refused["error"], (engine, refused)
    # the refusal is inert: not one live row changed (no wipe, no re-stamp)...
    for name, model in history.items():
        assert await _table_rows(model, epoch=1) == live_hist[name], \
            f"{name}: a refused replay mutated the live book"
    for model in positions:
        assert await _table_rows(model) == live_pos_dump[model.__name__], \
            f"{model.__name__}: a refused replay mutated the live book"
    # ...and no epoch boundary was appended by a refused replay
    assert await run_config.current_epoch("monthly") == 1
    assert await run_config.current_epoch("daily_core") == 1

    # --- 3) explicit reset closes both epochs; the paper book survives ---
    reset_m = await monthly.reset_monthly()
    reset_d = await daily_core.reset_daily_core()
    assert reset_m["ok"] is True and reset_m["epoch"] == 2, reset_m
    assert reset_d["ok"] is True and reset_d["epoch"] == 2, reset_d
    # every live row was COUNTED as preserved, not discarded
    assert reset_m["preserved"]["trades"] == len(live_monthly)
    assert reset_m["preserved"]["snapshots"] == len(live_hist["monthly_snapshots"])
    assert reset_m["preserved"]["allowances"] == len(live_hist["monthly_allowances"])
    assert reset_m["preserved"]["rebalances"] == len(live_hist["monthly_rebalances"])
    assert reset_d["preserved"] == {
        "trades": len(live_daily),
        "snapshots": len(live_hist["daily_snapshots"]),
        "allowances": len(live_hist["daily_allowances"]),
    }
    assert await run_config.current_epoch("monthly") == 2
    assert await run_config.current_epoch("daily_core") == 2
    # the closed epoch is bit-for-bit intact; only CURRENT state was cleared
    for name, model in history.items():
        assert await _table_rows(model, epoch=1) == live_hist[name], \
            f"{name}: the reset rewrote the closed epoch"
    for model in positions:
        assert await _table_rows(model) == [], model.__name__
    # both equity curves restart EMPTY in epoch 2 (no closed-epoch leak)
    assert await monthly.get_equity_curve() == []
    assert await daily_core.get_equity_curve() == []

    # --- 4) replay the SAME candles in the fresh epoch (SAME explicit start;
    #        the pinned clock sits at the last session and is never consulted
    #        for the replay's window) ---
    rm = await monthly.backfill(start=start)
    rd_ = await daily_core.backfill(start=start)
    assert rm["ok"] is True, rm
    assert rd_["ok"] is True, rd_

    replay_monthly = await _trade_rows(monthly.MonthlyTrade, epoch=2)
    replay_daily = await _trade_rows(daily_core.DailyCoreTrade, epoch=2)
    replay_monthly_pos = await _position_rows(monthly.MonthlyPosition)
    replay_daily_pos = await _position_rows(daily_core.DailyCorePosition)
    assert replay_monthly and replay_daily

    # the replay wrote epoch-2 rows ONLY: after it, the closed epoch is still
    # bit-for-bit identical to what the live engines produced ...
    for name, model in history.items():
        assert await _table_rows(model, epoch=1) == live_hist[name], \
            f"{name}: the replay rewrote the closed epoch"
    # ... and the old paper trades are exactly the live stream, not a concat
    assert await _trade_rows(monthly.MonthlyTrade, epoch=1) == live_monthly
    assert await _trade_rows(daily_core.DailyCoreTrade, epoch=1) == live_daily
    # the full history is live epoch 1 + replay epoch 2, nothing lost/duplicated
    assert len(await _table_rows(monthly.MonthlyTrade)) == \
        len(live_monthly) + len(replay_monthly)
    assert len(await _table_rows(daily_core.DailyCoreTrade)) == \
        len(live_daily) + len(replay_daily)

    # --- 5) GOLDEN: identical trade sets on the identical stream ---
    assert [_key(t) for t in live_monthly] == [_key(t) for t in replay_monthly], \
        "monthly: live vs replay trade streams diverge"
    assert [_key(t) for t in live_daily] == [_key(t) for t in replay_daily], \
        "daily-core: live vs replay trade streams diverge"

    # ...and the end-of-stream books match exactly (shares + avg_cost)
    for a, b, name in ((live_monthly_pos, replay_monthly_pos, "monthly"),
                       (live_daily_pos, replay_daily_pos, "daily-core")):
        assert [{k: p[k] for k in ("ticker", "shares", "avg_cost")} for p in a] == \
            [{k: p[k] for k in ("ticker", "shares", "avg_cost")} for p in b], \
            f"{name}: end-of-stream positions diverge"

    # --- 6) replay bookkeeping: opened_at = the first-buy date of the
    #        CURRENT (still-open) stint — a full release + re-buy opens a
    #        NEW stint, so the stream's very first buy does not anchor it ---
    for replay_pos, replay_tr in ((replay_monthly_pos, replay_monthly),
                                  (replay_daily_pos, replay_daily)):
        stints: dict[str, str] = {}
        for t in replay_tr:
            if t["side"] == "BUY":
                stints.setdefault(t["ticker"], t["date"])
            else:
                stints.pop(t["ticker"], None)
        for p in replay_pos:
            assert p["opened_at"] == stints[p["ticker"]], \
                f"opened_at {p['ticker']}: {p['opened_at']} != stint start {stints.get(p['ticker'])}"

    # --- 7) month-end band policy: releases only at the first session
    #        after the month-end that changed the band ---
    mes = monthly.completed_month_ends(close.index, sessions[0], sessions[-1])
    first_after_me = set()
    for me in mes:
        later = sessions[sessions > me]
        if len(later):
            first_after_me.add(later[0])
    for tr in replay_monthly:
        if tr["side"] == "SELL":
            assert pd.Timestamp(tr["date"]) in first_after_me, tr
    for tr in replay_daily:
        if tr["side"] == "SELL":
            assert pd.Timestamp(tr["date"]) in first_after_me, tr

    # --- 8) the fixture proves the policy is doing real work: a MID-MONTH
    #        frame ranks differently than the month-end frame, yet both
    #        engines matched each other (they both ranked the month-end) ---
    me_frame = monthly.eligible_frame(pd.Timestamp("2026-02-27"), close, vol, fund)
    mid_frame = monthly.eligible_frame(pd.Timestamp("2026-03-13"), close, vol, fund)
    assert me_frame is not None and mid_frame is not None
    _me, me_order = monthly._qv_order(me_frame)
    _mid, mid_order = monthly._qv_order(mid_frame)
    assert me_order[:3] != mid_order[:3], (
        "fixture regression: daily facts no longer differ from the month-end "
        "ranking — the parity no longer proves the month-end band policy")

    # --- 9) contributions aligned: the SAME months funded on both sides ---
    async def allowance_months(model, epoch: int | None = None):
        async with daily_core.Session() as s:
            stmt = select(model)
            if epoch is not None:
                stmt = stmt.where(model.epoch == epoch)
            rows = (await s.scalars(stmt)).all()
        return sorted(r.month for r in rows)

    from app.db import DailyCoreAllowance, MonthlyAllowance
    expected_months = sorted({d.strftime("%Y-%m") for d in sessions})
    # the epoch-2 replay deposits the tape's month sequence on BOTH engines...
    replay_months = await allowance_months(MonthlyAllowance, epoch=2)
    assert replay_months == expected_months
    assert replay_months == await allowance_months(DailyCoreAllowance, epoch=2)
    # ...and exactly the same months the live book (epoch 1) funded
    assert await allowance_months(MonthlyAllowance, epoch=1) == expected_months
    assert await allowance_months(DailyCoreAllowance, epoch=1) == expected_months
    # the full-history listings agree too (epoch 1 + epoch 2, per engine)
    assert await allowance_months(MonthlyAllowance) == \
        await allowance_months(DailyCoreAllowance)


async def test_deposit_months_align_with_the_replay_stream(parity_env):
    """Deposit timing: the live driver deposits each month on the month's
    FIRST session (the 21:00 UTC clock keeps Vienna's calendar date on the
    session date, so no off-by-one deposit); the replay deposits on the
    month's first STORED session. Same sessions -> same month keys."""
    clock, sessions, _open, _close, _vol, _fund = parity_env
    for d in sessions:
        clock["now"] = d.to_pydatetime().replace(hour=DRIVE_HOUR, tzinfo=UTC)
        await monthly.deposit_allowance()
    live = {d.strftime("%Y-%m") for d in sessions}
    from app.db import MonthlyAllowance
    async with daily_core.Session() as s:
        rows = (await s.scalars(select(MonthlyAllowance))).all()
    assert sorted(r.month for r in rows) == sorted(live)
