"""Daily-Core's adoption of the shared zero-fee fill primitives.

The three inline arithmetic sites in ``app/daily_core.py`` now call
``app/factor_fills.py`` — the SAME primitives Monthly's ``_exec_buy`` /
``_exec_sell`` and its persisted backfill already use:

    _exec_buy                  -> factor_fills.buy_fill(acc.cash, budget, price)
    _exec_sell                 -> factor_fills.sell_fill(acc.cash, pos.shares, price)
    backfill do_buy / do_sell  -> the same two primitives

This is arithmetic CONSOLIDATION, not a strategy change. On valid inputs the
float operation order is unchanged (``budget / price * 10000``, ``shares *
price``, ``cash - notional`` / ``cash + notional``), so every booked share
count, notional and cash balance is bit-identical to the pre-adoption code —
pinned by tests/test_daily_core.py, tests/test_dailycore_sizing_clock.py and
tests/test_cadence_parity.py, which this file deliberately leaves alone.

What CHANGED is the invalid-input contract, and that is what this file locks
down. Measured on the pre-adoption code (synthetic probe, same fixtures):

    _exec_buy(price=NaN)         ValueError: cannot convert float NaN to integer
                                 -> None, no mutation
    _exec_buy(budget=NaN)        ValueError: cannot convert float NaN to integer
                                 -> None, no mutation
    _exec_buy(budget=inf)        OverflowError: cannot convert float infinity to integer
                                 -> None, no mutation
    _exec_buy(price=1e-320)      OverflowError (shares overflow the 4dp scale)
                                 -> None, no mutation
    _exec_sell(price=NaN)        cash := NaN, commit dies with
                                 IntegrityError (NOT NULL on daily_core_account.cash)
                                 -> None, no mutation
    _exec_sell(price=inf)        cash := inf, position DELETED, a $inf trade booked
                                 -> None, no mutation

The backfill's ``do_buy`` / ``do_sell`` already tolerated bad inputs (they
sized through ``monthly.share_size``, itself a ``factor_fills.share_size``
wrapper, and pre-checked ``p is None or p <= 0``), so for them the adoption is
pure arithmetic consolidation: invalid input still books nothing and moves
nothing — the same behaviour, one formula.

Everything here is synthetic and offline: in-memory SQLite, hand-built price
frames, no provider calls and no production database.
"""

from __future__ import annotations

import asyncio

import pandas as pd
import pytest
from sqlalchemy import StaticPool, delete, select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app import daily_core, factor_fills, monthly
from app.config import settings
from app.db import Base

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


async def _seed(cash: float, positions: list[tuple[str, float, float]] = ()) -> None:
    """Seed the account cash and (ticker, shares, avg_cost) positions."""
    async with daily_core.Session() as s:
        acc = await daily_core._account(s)
        acc.cash = cash
        await s.commit()
        for t, sh, avg in positions:
            s.add(daily_core.DailyCorePosition(ticker=t, shares=sh, avg_cost=avg))
        await s.commit()


async def _reset_state(cash: float, positions: list[tuple[str, float, float]] = ()) -> None:
    """Wipe positions + trades, then re-seed at `cash`."""
    async with daily_core.Session() as s:
        await s.execute(delete(daily_core.DailyCoreTrade))
        await s.execute(delete(daily_core.DailyCorePosition))
        acc = await daily_core._account(s)
        acc.cash = cash
        await s.commit()
        for t, sh, avg in positions:
            s.add(daily_core.DailyCorePosition(ticker=t, shares=sh, avg_cost=avg))
        await s.commit()


async def _state() -> dict:
    """The whole portfolio state a fill could have touched."""
    async with daily_core.Session() as s:
        acc = await s.get(daily_core.DailyCoreAccount, 1)
        positions = (await s.scalars(
            select(daily_core.DailyCorePosition)
            .order_by(daily_core.DailyCorePosition.ticker))).all()
        trades = (await s.scalars(
            select(daily_core.DailyCoreTrade)
            .order_by(daily_core.DailyCoreTrade.id))).all()
        return {
            "cash": None if acc is None else acc.cash,
            "positions": [(p.ticker, p.shares, p.avg_cost) for p in positions],
            "trades": [(t.side, t.ticker, t.shares, t.price, t.cash_after) for t in trades],
        }


def _flat_maps(monkeypatch, prices: dict[str, float]) -> None:
    """Stub both USD price maps (marks at S, fills at F's strict open) flat."""
    async def fake_price_map(tickers, asof=None):
        return {t: prices[t] for t in tickers if t in prices}

    async def fake_open_map(tickers, d):
        return {t: prices[t] for t in tickers if t in prices}

    monkeypatch.setattr(monthly, "_price_usd_map", fake_price_map)
    monkeypatch.setattr(monthly, "_open_usd_map", fake_open_map)


@pytest.fixture
def deployment_env(monkeypatch, tmp_path):
    """A four-name ranking behind the live decision pipeline, priced flat at
    $100 so every expected number is exact."""
    frame = pd.DataFrame(
        {
            "roe": [0.3, 0.25, 0.2, 0.1],
            "p_fcf": [10.0, 15.0, 20.0, 25.0],
            "mcap": [1e10] * 4,
            "dollar_vol": [1e8] * 4,
            "mom": [0.5, 0.4, 0.3, 0.2],
            "price": [100.0] * 4,
            "vol": [0.2] * 4,
            "eligible": [True] * 4,
        },
        index=pd.Index(["AAA", "BBB", "CCC", "DDD"], name="ticker"),
    )
    me = pd.Timestamp("2026-09-30")

    async def fake_stored_me(_now):
        return me

    async def fake_core_frame(_me):
        return frame, ["AAA", "BBB", "CCC", "DDD"]

    monkeypatch.setattr(daily_core, "_STATE_FILE", tmp_path / "state.json")
    monkeypatch.setattr(daily_core, "_stored_month_end", fake_stored_me)
    monkeypatch.setattr(daily_core, "_core_frame", fake_core_frame)
    _flat_maps(monkeypatch, {t: 100.0 for t in ["AAA", "BBB", "CCC", "DDD"]})
    return frame


def _install_backfill_env(monkeypatch, tmp_path, *, open_px: pd.DataFrame,
                          close: pd.DataFrame, vol: pd.DataFrame, order_of) -> None:
    """Patch the backfill loaders so the replay walks a hand-built tape.

    ``order_of(month_end)`` returns the ranking order for that month-end, so a
    test can script a band change (and therefore a release) between months.
    """
    def fake_frame(m, c, v, f):
        names = order_of(pd.Timestamp(m))
        frame = pd.DataFrame({"eligible": [True] * len(names)},
                             index=pd.Index(names, name="ticker"))
        frame.attrs["me"] = pd.Timestamp(m)
        return frame

    def fake_order(frame):
        return frame["eligible"], order_of(frame.attrs["me"])

    async def fake_load_ohlcv_frames(_tickers, _asof, start=None):
        return open_px, close, vol

    async def fake_load_fundamentals(tickers):
        return {t: {"x": []} for t in tickers}

    monkeypatch.setattr(daily_core, "_STATE_FILE", tmp_path / "state.json")
    monkeypatch.setattr(daily_core, "_current_month", lambda: "2026-10")
    monkeypatch.setattr(daily_core, "universe_tickers", lambda _u: list(close.columns))
    monkeypatch.setattr(daily_core.fundamentals_mod, "load_fundamentals",
                        fake_load_fundamentals)
    monkeypatch.setattr(monthly, "load_ohlcv_frames", fake_load_ohlcv_frames)
    monkeypatch.setattr(monthly, "eligible_frame", fake_frame)
    monkeypatch.setattr(monthly, "_qv_order", fake_order)


def _one_ticker_tape(bad_open: float | None):
    """Jul 1 - Aug 3 tape, AAA flat at $100, with `bad_open` written into the
    ONLY fill session (Aug 3, the first session after the July month-end).

    The fill session is the tape's LAST session on purpose: the deployment has
    no second chance to retry the leg, so a rejected price leaves the replay
    with no position and no trade at all."""
    dates = pd.bdate_range("2026-07-01", "2026-08-03")
    close = pd.DataFrame({"AAA": 100.0}, index=dates)
    open_px = close.copy()
    open_px.at[pd.Timestamp("2026-08-03"), "AAA"] = bad_open
    vol = pd.DataFrame(1e6, index=dates, columns=["AAA"])
    return open_px, close, vol


# ---------------------------------------------------------------------------
# _exec_buy: invalid inputs are "no fill, no mutation"
# ---------------------------------------------------------------------------

class TestExecBuyInvalidInput:
    @pytest.mark.parametrize("price", [0.0, -5.0, float("nan"), float("inf"),
                                       float("-inf"), 1e-320])
    def test_invalid_price_books_nothing(self, mem_db, price):
        """NaN/inf/overflow used to raise out of ``math.floor``; a subnormal
        price overflowed the 4dp scale. All of them are now "no fill"."""
        asyncio.run(_seed(1000.0, [("AAA", 5.0, 100.0)]))
        out = asyncio.run(daily_core._exec_buy("AAA", price, 100.0, "fills: invalid price"))
        assert out is None
        state = asyncio.run(_state())
        assert state["cash"] == 1000.0            # no debit
        assert state["positions"] == [("AAA", 5.0, 100.0)]   # no top-up
        assert state["trades"] == []              # nothing booked

    @pytest.mark.parametrize("budget", [0.0, 0.99, -50.0, float("nan"), float("inf")])
    def test_budget_guards_book_nothing(self, mem_db, budget):
        """A sub-dollar budget is not executable; a NaN budget used to raise
        ValueError and an inf budget OverflowError. Both are now no-fill."""
        asyncio.run(_seed(1000.0, [("AAA", 5.0, 100.0)]))
        out = asyncio.run(daily_core._exec_buy("AAA", 100.0, budget, "fills: invalid budget"))
        assert out is None
        state = asyncio.run(_state())
        assert state["cash"] == 1000.0
        assert state["positions"] == [("AAA", 5.0, 100.0)]
        assert state["trades"] == []

    def test_sub_quantum_size_books_nothing(self, mem_db):
        """$100 at $1e9 is 1e-7 shares: below the 1e-4 quantum, no fill."""
        asyncio.run(_seed(1000.0))
        out = asyncio.run(daily_core._exec_buy("AAA", 1e9, 100.0, "fills: dust"))
        assert out is None
        state = asyncio.run(_state())
        assert state["cash"] == 1000.0
        assert state["positions"] == []
        assert state["trades"] == []
        # ... and the primitive agrees this leg is not executable.
        assert factor_fills.share_size(100.0, 1e9) == 0.0

    def test_exact_cash_razor_edge_is_skipped_never_resized(self, mem_db):
        """The live BUY loop clamps its budget to ``round(cash, 2)``, which can
        be a hair ABOVE the exact cash. When the 4dp share floor then books a
        notional above that exact cash, the leg is SKIPPED — never re-sized
        down to the cash (a shrunken trade is a trade the live executor would
        have refused)."""
        cash = 99.99995
        budget = round(cash, 2)
        assert budget == 100.0 and budget > cash

        asyncio.run(_seed(cash))
        out = asyncio.run(daily_core._exec_buy("AAA", 1.0, budget, "fills: razor edge"))
        assert out is None
        state = asyncio.run(_state())
        assert state["cash"] == cash              # untouched, not debited to 0
        assert state["positions"] == []           # no clamped/resized position
        assert state["trades"] == []

        # The two outcomes the guard has to distinguish: the budgeted size
        # costs MORE than the cash on hand, while a clamped size WOULD have
        # been executable. The guard picks "skip", not "clamp".
        assert factor_fills.share_size(budget, 1.0) * 1.0 > cash
        assert factor_fills.share_size(cash, 1.0) == 99.9999

    def test_exact_cash_razor_edge_on_a_fractional_price(self, mem_db):
        """Same razor edge one ULP under a round number, at a fractional
        price: budget $3.00 sizes 2.0000 shares = $3.00 > $2.9999999999999996."""
        cash = 3.0 - 5e-16
        budget = round(cash, 2)
        assert budget == 3.0 and budget > cash

        asyncio.run(_seed(cash))
        out = asyncio.run(daily_core._exec_buy("AAA", 1.5, budget, "fills: razor edge"))
        assert out is None
        state = asyncio.run(_state())
        assert state["cash"] == cash
        assert state["positions"] == []
        assert state["trades"] == []
        assert factor_fills.share_size(cash, 1.5) == 1.9999   # a clamp would fill


# ---------------------------------------------------------------------------
# _exec_sell: invalid inputs are "no fill, no mutation"
# ---------------------------------------------------------------------------

class TestExecSellInvalidInput:
    @pytest.mark.parametrize("price", [0.0, -5.0, float("nan"), float("inf")])
    def test_invalid_price_keeps_the_position(self, mem_db, price):
        """Before the adoption a NaN quote set cash to NaN and died on commit
        (IntegrityError), and an inf quote DELETED the position for $inf. Now
        the primitive rejects the leg before any state change."""
        asyncio.run(_seed(1000.0, [("AAA", 5.0, 100.0)]))
        out = asyncio.run(daily_core._exec_sell("AAA", price, "fills: invalid price"))
        assert out is None
        state = asyncio.run(_state())
        assert state["cash"] == 1000.0
        assert state["positions"] == [("AAA", 5.0, 100.0)]   # not deleted
        assert state["trades"] == []

    def test_empty_position_is_not_a_fill(self, mem_db):
        """A zero-share position is not liquidatable: no trade, no delete,
        cash untouched (the primitive's non-positive-share guard)."""
        asyncio.run(_seed(1000.0, [("AAA", 0.0, 100.0)]))
        out = asyncio.run(daily_core._exec_sell("AAA", 100.0, "fills: empty"))
        assert out is None
        state = asyncio.run(_state())
        assert state["cash"] == 1000.0
        assert state["positions"] == [("AAA", 0.0, 100.0)]
        assert state["trades"] == []


# ---------------------------------------------------------------------------
# Exact valid-input pins (the historical arithmetic, bit for bit)
# ---------------------------------------------------------------------------

class TestValidFillPins:
    def test_floor4_size_and_exact_cash_after(self, mem_db):
        """$100 at $3 -> 33.3333 shares, notional 99.9999, cash 200 - notional
        exactly (the historical operation order: shares * price, then cash -)."""
        asyncio.run(_seed(200.0))
        out = asyncio.run(daily_core._exec_buy("AAA", 3.0, 100.0, "fills: pin"))
        assert out is not None
        assert out["shares"] == 33.3333
        assert out["price"] == 3.0
        assert out["cost"] == round(33.3333 * 3.0, 2)      # 99.9999 -> 100.0
        state = asyncio.run(_state())
        assert state["cash"] == 200.0 - 33.3333 * 3.0
        assert state["positions"] == [("AAA", 33.3333, 3.0)]
        assert state["trades"] == [("BUY", "AAA", 33.3333, 3.0,
                                    200.0 - 33.3333 * 3.0)]

    def test_full_cash_buy_leaves_the_exact_remainder(self, mem_db):
        """A full-cash budget at $3 buys 333.3333 shares ($999.9999), leaving
        the exact float remainder — not a rounded-to-zero cash balance."""
        asyncio.run(_seed(1000.0))
        out = asyncio.run(daily_core._exec_buy("AAA", 3.0, 1000.0, "fills: full cash"))
        assert out is not None
        assert out["shares"] == 333.3333
        state = asyncio.run(_state())
        assert state["cash"] == 1000.0 - 333.3333 * 3.0
        assert state["cash"] > 0.0
        assert state["cash"] == pytest.approx(1e-4, abs=1e-12)

    def test_topup_keeps_the_weighted_avg_cost(self, mem_db):
        """A second buy averages into the existing position with the same
        formula as before (cost_basis weighted by shares)."""
        asyncio.run(_seed(300.0, [("AAA", 10.0, 5.0)]))
        out = asyncio.run(daily_core._exec_buy("AAA", 5.0, 50.0, "fills: top-up"))
        assert out is not None
        assert out["shares"] == 10.0
        state = asyncio.run(_state())
        # (10 * 5.0 + 50.0) / 20 — unchanged price, so the average stays put
        assert state["positions"] == [("AAA", 20.0, (10.0 * 5.0 + 50.0) / 20.0)]
        assert state["cash"] == 300.0 - 50.0

    def test_full_liquidation_sells_the_position_shares_as_is(self, mem_db):
        """A SELL liquidates the position's OWN share count — never re-floored
        to the 4dp quantum — for exactly shares * price."""
        shares = 1.23456789                      # deliberately not 4dp
        asyncio.run(_seed(100.0, [("AAA", shares, 10.0)]))
        out = asyncio.run(daily_core._exec_sell("AAA", 10.0, "fills: liquidate"))
        assert out is not None
        assert out["shares"] == shares           # as-is, no re-flooring
        assert out["proceeds"] == round(shares * 10.0, 2)   # 12.3456789 -> 12.35
        state = asyncio.run(_state())
        assert state["cash"] == 100.0 + shares * 10.0
        assert state["positions"] == []          # position closed
        assert state["trades"] == [("SELL", "AAA", shares, 10.0, 100.0 + shares * 10.0)]

    def test_four_dp_liquidation_proceeds(self, mem_db):
        """The 4dp case: 33.3333 shares at $3 -> $99.9999 proceeds exactly."""
        asyncio.run(_seed(0.0001, [("AAA", 33.3333, 3.0)]))
        out = asyncio.run(daily_core._exec_sell("AAA", 3.0, "fills: liquidate"))
        assert out is not None
        assert out["proceeds"] == 100.0          # round(99.9999, 2)
        state = asyncio.run(_state())
        assert state["cash"] == 0.0001 + 33.3333 * 3.0


# ---------------------------------------------------------------------------
# Backfill do_buy / do_sell: invalid prices book nothing
# ---------------------------------------------------------------------------

class TestBackfillInvalidPrice:
    def test_control_valid_price_books_the_leg(self, mem_db, monkeypatch, tmp_path):
        """The tape shape used by the invalid-price cases really does fill on
        the last session when the price is usable — otherwise those cases
        would be vacuous."""
        open_px, close, vol = _one_ticker_tape(100.0)
        _install_backfill_env(monkeypatch, tmp_path, open_px=open_px, close=close,
                              vol=vol, order_of=lambda _me: ["AAA"])

        r = asyncio.run(daily_core.backfill(start="2026-08-01"))
        assert r["ok"] is True
        assert r["trades"] == 1
        state = asyncio.run(_state())
        # $1000 deposit, weight $1000/10 = $100 -> 1 share at $100
        assert state["cash"] == settings.sim_monthly_contribution - 100.0
        assert [(t, sh) for t, sh, _avg in state["positions"]] == [("AAA", 1.0)]
        assert [(side, ticker) for side, ticker, *_ in state["trades"]] == [("BUY", "AAA")]

    @pytest.mark.parametrize("bad_open", [None, float("nan"), float("inf"),
                                          float("-inf"), 0.0, -5.0])
    def test_invalid_price_books_nothing(self, mem_db, monkeypatch, tmp_path, bad_open):
        """do_buy sees a NaN (px_at -> None), inf, zero or negative open on the
        only fill session: no trade, no position, and the deposit is NOT spent.
        Before and after the adoption this is "no fill" — the adoption just
        moves the rejection into the shared primitive."""
        open_px, close, vol = _one_ticker_tape(bad_open)
        _install_backfill_env(monkeypatch, tmp_path, open_px=open_px, close=close,
                              vol=vol, order_of=lambda _me: ["AAA"])

        r = asyncio.run(daily_core.backfill(start="2026-08-01"))
        assert r["ok"] is True
        assert r["trades"] == 0
        state = asyncio.run(_state())
        assert state["cash"] == settings.sim_monthly_contribution   # nothing spent
        assert state["positions"] == []
        assert state["trades"] == []


# ---------------------------------------------------------------------------
# Mini-parity: the backfill books exactly what the live executor books
# ---------------------------------------------------------------------------

class TestBackfillLiveParity:
    """The replay's do_buy/do_sell and the live _exec_buy/_exec_sell must book
    identical shares / notional / cash for identical (cash, budget|shares,
    price) inputs — otherwise a backfilled book diverges from the live one.

    The tape scripts a real band change so BOTH leg kinds fire:

        2026-08-03  July month-end band {AAA, BBB}: buy 1 AAA @ $100, buy 2 BBB @ $50
        2026-09-01  August month-end band {AAA}:    release BBB @ $50, top up AAA @ $100

    The spy on the shared primitives records the exact arguments the replay
    hands them, so the live executor can be replayed against the SAME inputs.
    """

    @pytest.fixture
    def parity_tape(self, monkeypatch, tmp_path):
        dates = pd.bdate_range("2026-07-01", "2026-09-30")
        close = pd.DataFrame({"AAA": 100.0, "BBB": 50.0}, index=dates)
        open_px = close.copy()
        vol = pd.DataFrame(1e6, index=dates, columns=["AAA", "BBB"])

        def order_of(me):
            return ["AAA", "BBB"] if me.month == 7 else ["AAA"]

        _install_backfill_env(monkeypatch, tmp_path, open_px=open_px, close=close,
                              vol=vol, order_of=order_of)
        return {"open": open_px, "close": close}

    def test_backfill_and_live_executor_book_identical_fills(self, mem_db, parity_tape,
                                                             monkeypatch):
        real_buy, real_sell = factor_fills.buy_fill, factor_fills.sell_fill
        calls: list[tuple] = []
        recording = [True]  # the live loop below reaches the same primitives

        def spy_buy(cash, budget, price):
            fill = real_buy(cash, budget, price)
            if recording[0]:
                calls.append(("BUY", cash, budget, price, fill))
            return fill

        def spy_sell(cash, shares, price):
            fill = real_sell(cash, shares, price)
            if recording[0]:
                calls.append(("SELL", cash, shares, price, fill))
            return fill

        monkeypatch.setattr(factor_fills, "buy_fill", spy_buy)
        monkeypatch.setattr(factor_fills, "sell_fill", spy_sell)

        r = asyncio.run(daily_core.backfill(start="2026-08-01"))
        assert r["ok"] is True
        recorded = list(calls)   # the replay's legs, in booking order
        recording[0] = False     # stop recording before replaying them live

        # The replay's exact call sequence: deposit $1000 on Aug 3, spend it,
        # then the Sept deposit lands before the release (cash 800 -> 1800).
        assert [(kind, cash, arg, price) for kind, cash, arg, price, _f in recorded] == [
            ("BUY", 1000.0, 100.0, 100.0),
            ("BUY", 900.0, 100.0, 50.0),
            ("SELL", 1800.0, 2.0, 50.0),
            ("BUY", 1900.0, 100.0, 100.0),
        ]
        assert all(f is not None for *_args, f in recorded)
        assert [(f.shares, f.notional, f.cash_after) for *_a, f in recorded] == [
            (1.0, 100.0, 900.0),
            (2.0, 100.0, 800.0),
            (2.0, 100.0, 1900.0),
            (1.0, 100.0, 1800.0),
        ]

        # The replay BOOKED those values: the trade log carries them in order
        # and the end state is the fold of the fills.
        state = asyncio.run(_state())
        assert state["trades"] == [
            ("BUY", "AAA", 1.0, 100.0, 0.0),
            ("BUY", "BBB", 2.0, 50.0, 0.0),
            ("SELL", "BBB", 2.0, 50.0, 0.0),
            ("BUY", "AAA", 1.0, 100.0, 0.0),
        ]
        assert state["cash"] == 1800.0
        assert state["positions"] == [("AAA", 2.0, 100.0)]

        # ... and the LIVE executor, fed the same args, books the same fill.
        for kind, cash, arg, price, fill in recorded:
            asyncio.run(_reset_state(cash, []))
            if kind == "BUY":
                out = asyncio.run(daily_core._exec_buy("PAR", price, arg, "parity"))
                assert out is not None
                assert out["shares"] == fill.shares
                assert out["cost"] == round(fill.notional, 2)
            else:
                asyncio.run(_reset_state(cash, [("PAR", arg, price)]))
                out = asyncio.run(daily_core._exec_sell("PAR", price, "parity"))
                assert out is not None
                assert out["shares"] == fill.shares
                assert out["proceeds"] == round(fill.notional, 2)
            assert out["price"] == price
            live = asyncio.run(_state())
            assert live["cash"] == fill.cash_after
            assert live["trades"] == [(kind, "PAR", fill.shares, price, fill.cash_after)]
            if kind == "BUY":
                assert live["positions"] == [("PAR", fill.shares, price)]
            else:
                assert live["positions"] == []


# ---------------------------------------------------------------------------
# Negative-cash reachability: the primitives reject cash < 0
# ---------------------------------------------------------------------------

class TestCashNeverNegative:
    def test_deployment_never_books_negative_cash(self, mem_db, deployment_env):
        """``buy_fill``/``sell_fill`` reject negative cash. Daily-Core cannot
        reach that branch: buys are exact-cash guarded (a leg is skipped, not
        clamped) and deposits/sell proceeds only add. Here a $50 account
        buying four $100 names books four $5 legs and never overdraws."""
        asyncio.run(_seed(50.0))

        r = asyncio.run(daily_core.run_deployment())
        assert r["deployed"] is True
        assert [t["ticker"] for t in r["trades"]] == ["AAA", "BBB", "CCC", "DDD"]
        assert [t["shares"] for t in r["trades"]] == [0.05] * 4
        assert [t["cost"] for t in r["trades"]] == [5.0] * 4

        state = asyncio.run(_state())
        # every intermediate balance is >= 0: 50 -> 45 -> 40 -> 35 -> 30
        assert [t[4] for t in state["trades"]] == [45.0, 40.0, 35.0, 30.0]
        assert state["cash"] == 30.0
        assert state["cash"] >= 0.0
        assert all(cash >= 0.0 for _side, _t, _sh, _px, cash in state["trades"])
