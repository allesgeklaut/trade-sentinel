"""Session fill contract — engine-level tests (contract: app/session.py docstring).

Pins the contract end to end at the sim-engine level (app/sim.py +
app/session.py), on top of the proposal-level tests in
tests/test_strategy_safety.py:

  * decision inputs exclude F's bar entirely (rule 1) — mutating F's candle
    cannot change a decision, and a ticker that did not trade on S gets no
    fresh signal (its stop standing orders still apply through F's bar);
  * session primitives: strict exact-date lookups (``open_at`` — no ffill),
    ``prev_session`` from the stored index with the capped calendar-walk
    fallback, and the aligned five-field loader round-trip (rules 1-2);
  * the monotone ATR ratchet (rule 6) is anchored on the position's entry
    and never loosens as the price declines;
  * the percent stop is frozen at entry and survives top-ups (rule 5);
  * the per-session guard (rule 2) makes a second cycle in one session a
    no-op, and backfill/reset_sim re-arm it.

All tests run on in-memory SQLite; nothing external is contacted, and the
guard state file is pointed at a tmp_path so /data is never touched.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from unittest.mock import AsyncMock

import pandas as pd
import pytest
from sqlalchemy import StaticPool, select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app import session as session_mod
from app import sim
from app.config import settings
from app.db import Base, Candle, SimAccount, SimPosition

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

# Cycle dates used throughout: S = Friday 2026-03-06, F = Monday 2026-03-09.
S_STR = "2026-03-06"
F_STR = "2026-03-09"
NOW_F = datetime(2026, 3, 9, 23, 30, tzinfo=UTC)


@pytest.fixture
async def mem_db(monkeypatch):
    """Replace the sim's database engine with an in-memory SQLite DB."""
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    session_factory = async_sessionmaker(engine, expire_on_commit=False)

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    monkeypatch.setattr(sim, "Session", session_factory)
    # session.load_open_frames imports db.Session at call time — patch it too.
    import app.db as db_mod
    monkeypatch.setattr(db_mod, "Session", session_factory)

    # Seed the singleton account
    async with session_factory() as s:
        acc = SimAccount(id=1, cash=settings.sim_start_cash, last_allowance_month=None)
        s.add(acc)
        await s.commit()

    yield session_factory

    await engine.dispose()


@pytest.fixture
async def with_cash(mem_db):
    """Give the sim account some starting cash (default is 0.0)."""
    async with mem_db() as s:
        acc = await s.get(SimAccount, 1)
        acc.cash = 10_000.0
        await s.commit()
    return mem_db


def _candle(ticker: str, d: str, o: float, h: float, l: float, c: float,
            v: float = 1000.0) -> Candle:
    ts = datetime.strptime(d, "%Y-%m-%d")
    return Candle(ticker=ticker, timestamp=ts, open=o, high=h, low=l, close=c,
                  volume=v)


def _mk_frames(ticker: str, dates, opens, highs, lows, closes) -> session_mod.SessionFrames:
    """Build aligned SessionFrames for one ticker directly (no DB)."""
    idx = pd.DatetimeIndex(dates)

    def f(vals):
        return pd.DataFrame({ticker: pd.Series(vals, index=idx, dtype=float)})

    return session_mod.SessionFrames(
        open=f(opens), high=f(highs), low=f(lows), close=f(closes),
        volume=f([1000.0] * len(dates)))


# ---------------------------------------------------------------------------
# Rule 1 — decision inputs are strictly data <= S
# ---------------------------------------------------------------------------

class TestSessionDecisionInputs:
    async def test_f_bar_never_informs_the_decision(self, mem_db):
        """Mutating F's candle cannot change a decision: signals and decision
        prices are computed from candles <= S only. F's bar shows up solely
        in ``session_bars`` (the strict fill reference)."""
        from app.db import Session

        s_close = 100.0 + 0.1 * 209  # the drift's last close (210th day)
        f_open, f_low, f_close = 200.0, 199.0, 500.0  # absurd values on purpose
        async with Session() as s:
            end = datetime.strptime(S_STR, "%Y-%m-%d").date()
            for i in range(210):
                d = end - pd.Timedelta(days=209 - i)
                c = 100.0 + 0.1 * i
                s.add(_candle("X", d.strftime("%Y-%m-%d"), c - 0.2, c + 0.5,
                              c - 0.5, c))
            s.add(_candle("X", F_STR, f_open, 501.0, f_low, f_close))
            await s.commit()

        d1 = await sim._decision_inputs(["X"], now=NOW_F)
        assert d1.s_ts == pd.Timestamp(S_STR)
        assert d1.signals["X"]["snapshot"]["close"] == pytest.approx(s_close)
        assert d1.prices["X"] == pytest.approx(s_close)
        assert d1.session_bars["X"] == {"open": f_open, "low": f_low,
                                        "close": f_close}
        sig1 = d1.signals["X"]

        # Mutate F's bar drastically — the decision must not move.
        async with Session() as s:
            row = (await s.scalars(
                select(Candle).where(Candle.ticker == "X",
                                     Candle.timestamp == datetime(2026, 3, 9)),
            )).one()
            row.open, row.low, row.close = 1.0, 0.5, 2.0
            await s.commit()

        d2 = await sim._decision_inputs(["X"], now=NOW_F)
        assert d2.signals["X"] == sig1, "F's bar must never inform the decision"
        assert d2.prices == d1.prices
        # Only the strict fill reference picked up the mutation.
        assert d2.session_bars["X"] == {"open": 1.0, "low": 0.5, "close": 2.0}

    async def test_no_f_bar_means_unfillable(self, mem_db):
        """No F bar → empty session_bars → nothing can fill this cycle, while
        the decision view (signals on S) still stands for the next cycle."""
        from app.db import Session

        async with Session() as s:
            end = datetime.strptime(S_STR, "%Y-%m-%d").date()
            for i in range(210):
                d = end - pd.Timedelta(days=209 - i)
                c = 100.0 + 0.1 * i
                s.add(_candle("X", d.strftime("%Y-%m-%d"), c - 0.2, c + 0.5,
                              c - 0.5, c))
            await s.commit()

        d = await sim._decision_inputs(["X"], now=NOW_F)
        assert d.session_bars == {}
        assert "X" in d.signals  # the decision is still taken on S's data
        assert d.prices["X"] > 0

    async def test_stale_ticker_gets_no_signal_but_f_bar_fills(self, mem_db):
        """A ticker whose last candle predates S has no fresh decision view
        (no signal, no decision price) — but F's bar still feeds its stop
        standing orders through session_bars. Y keeps trading through S so
        the stored index (and thus S) advances past X's last bar."""
        from app.db import Session

        async with Session() as s:
            # X: 210 calendar-day candles ending 2026-03-03 — stale by S.
            end = datetime.strptime("2026-03-03", "%Y-%m-%d").date()
            for i in range(210):
                d = end - pd.Timedelta(days=209 - i)
                c = 100.0 + 0.1 * i
                s.add(_candle("X", d.strftime("%Y-%m-%d"), c - 0.2, c + 0.5,
                              c - 0.5, c))
            s.add(_candle("X", F_STR, 80.0, 81.0, 79.0, 80.5))
            # Y: 210 candles ending exactly on S — the fresh one.
            end_y = datetime.strptime(S_STR, "%Y-%m-%d").date()
            for i in range(210):
                d = end_y - pd.Timedelta(days=209 - i)
                c = 50.0 + 0.05 * i
                s.add(_candle("Y", d.strftime("%Y-%m-%d"), c - 0.1, c + 0.3,
                              c - 0.3, c))
            await s.commit()

        d = await sim._decision_inputs(["X", "Y"], now=NOW_F)
        assert d.s_ts == pd.Timestamp(S_STR)
        assert "X" not in d.signals, "stale decision views are excluded"
        assert "X" not in d.prices
        assert "Y" in d.signals and "Y" in d.prices
        assert d.session_bars["X"] == {"open": 80.0, "low": 79.0,
                                       "close": 80.5}, \
            "the stale ticker's stop standing orders still apply via F's bar"
        assert "Y" not in d.session_bars, "Y has no F bar — unfillable"


# ---------------------------------------------------------------------------
# Session primitives (strict lookups, calendar fallback, loader round-trip)
# ---------------------------------------------------------------------------

class TestSessionPrimitives:
    def test_prev_session_from_stored_index(self):
        idx = pd.DatetimeIndex(["2026-03-02", "2026-03-03", "2026-03-05",
                                "2026-03-06"])
        assert session_mod.prev_session(idx, F_STR) == pd.Timestamp(S_STR)
        # Strictly before: S itself resolves to the session before it.
        assert session_mod.prev_session(idx, S_STR) == pd.Timestamp("2026-03-05")

    def test_prev_session_calendar_fallback(self, monkeypatch):
        """No stored bars (e.g. warmup-bounded load over an empty cache) →
        walk back over the trading calendar, skipping weekends/holidays."""
        def fake_trading(dt):
            return dt.weekday() < 5

        monkeypatch.setattr(sim, "is_trading_day", fake_trading)
        # Monday 2026-03-09 → Friday 2026-03-06 (Sat/Sun skipped).
        assert session_mod.prev_session(
            pd.DatetimeIndex([]), F_STR) == pd.Timestamp(S_STR)

    def test_prev_session_fallback_capped_at_two_weeks(self, monkeypatch):
        """An unusable calendar can never spin: after the 14-day cap the
        caller gets None and must take no decision."""
        monkeypatch.setattr(sim, "is_trading_day", lambda dt: False)
        assert session_mod.prev_session(
            pd.DatetimeIndex([]), F_STR) is None

    def test_open_at_strict_exact_date(self):
        frame = pd.DataFrame(
            {"X": [10.0, 11.0]},
            index=pd.DatetimeIndex(["2026-03-05", "2026-03-06"]))
        assert session_mod.open_at(frame, S_STR, "X") == 11.0
        assert session_mod.open_at(frame, "2026-03-07", "X") is None, \
            "absent date is never forward-filled"
        assert session_mod.open_at(frame, S_STR, "Y") is None, \
            "absent ticker is not a price"
        assert session_mod.open_at(None, S_STR, "X") is None

    def test_open_at_nan_is_not_a_price(self):
        frame = pd.DataFrame(
            {"X": [10.0, float("nan")]},
            index=pd.DatetimeIndex(["2026-03-05", "2026-03-06"]))
        assert session_mod.open_at(frame, S_STR, "X") is None, \
            "a missing quote is not an executable price"

    async def test_load_open_frames_round_trip(self, mem_db):
        """One query → aligned five-field frames; rows_for drops missing
        bars per ticker (never forward-fills) and rebuilds compute-shaped
        rows."""
        from app.db import Session

        async with Session() as s:
            s.add(_candle("X", "2026-03-02", 9.5, 10.5, 9.0, 10.0, v=500.0))
            s.add(_candle("X", "2026-03-03", 10.5, 11.5, 10.0, 11.0, v=600.0))
            # gap: X does not trade on 03-04 / 03-05
            s.add(_candle("X", S_STR, 12.5, 13.5, 12.0, 13.0, v=700.0))
            for d, c in (("2026-03-02", 20.0), ("2026-03-03", 21.0),
                         ("2026-03-04", 22.0), ("2026-03-05", 23.0),
                         (S_STR, 24.0)):
                s.add(_candle("Y", d, c - 0.5, c + 0.5, c - 1.0, c, v=800.0))
            await s.commit()

        frames = await session_mod.load_open_frames(["X", "Y"],
                                                    asof=pd.Timestamp(F_STR))
        assert list(frames.close.index) == [
            pd.Timestamp(d) for d in
            ("2026-03-02", "2026-03-03", "2026-03-04", "2026-03-05", S_STR)]
        assert session_mod.open_at(frames.close, S_STR, "X") == 13.0
        assert session_mod.open_at(frames.open, S_STR, "X") == 12.5
        assert session_mod.open_at(frames.low, S_STR, "X") == 12.0
        assert session_mod.open_at(frames.high, S_STR, "X") == 13.5
        assert session_mod.open_at(frames.close, "2026-03-04", "X") is None, \
            "no ffill across X's gap"

        xr = session_mod.rows_for(frames, "X")
        assert [r["time"] for r in xr] == [
            "2026-03-02", "2026-03-03", S_STR]
        assert xr[-1] == {"time": S_STR, "open": 12.5, "high": 13.5,
                          "low": 12.0, "close": 13.0, "volume": 700.0}
        yr = session_mod.rows_for(frames, "Y")
        assert len(yr) == 5

    def test_session_fill_open(self):
        """_session_fill_open: only the strict open; missing ticker or a
        non-executable open is no fill."""
        bars = {"X": {"open": 98.0, "low": 97.0, "close": 99.0}}
        assert sim._session_fill_open(bars, "X") == 98.0
        assert sim._session_fill_open(bars, "NOPE") is None
        assert sim._session_fill_open({"X": {"open": None, "low": 90.0,
                                             "close": 99.0}}, "X") is None
        assert sim._session_fill_open({"X": {"open": 0.0, "low": 90.0,
                                             "close": 99.0}}, "X") is None
        # Case tolerance: lowercase ticker arg finds uppercase keys.
        assert sim._session_fill_open({"X": {"open": 5.0}}, "x") == 5.0


# ---------------------------------------------------------------------------
# Rule 6 — the monotone ATR ratchet (anchored on the entry, never loosens)
# ---------------------------------------------------------------------------

def _rise_then_decline_dates(n_rise: int = 40, n_fall: int = 30):
    end = datetime.strptime(S_STR, "%Y-%m-%d").date()
    dates = []
    for i in range(n_rise + n_fall):
        d = end - pd.Timedelta(days=n_rise + n_fall - 1 - i)
        dates.append(pd.Timestamp(d))
    return dates


class TestRatchetMonotone:
    def _scenario_frames(self):
        """Rise 100→139 over 40 sessions, then decline ~2.5/day over 30.

        Returns (frames, dates, closes): the classic pattern where the
        per-bar ATR candidate level peaks during the rise and collapses
        during the decline — the ratchet must keep the historical max.
        """
        dates = _rise_then_decline_dates()
        closes = []
        for i in range(40):
            closes.append(100.0 + i)          # 100 .. 139
        for j in range(1, 31):
            closes.append(139.0 - 2.5 * j)    # 136.5 .. 63.5
        opens = [c - 0.5 for c in closes]
        highs = [c + 1.0 for c in closes]
        lows = [c - 1.0 for c in closes]
        frames = _mk_frames("X", dates, opens, highs, lows, closes)
        return frames, dates, closes

    def test_ratchet_holds_historical_max_through_the_decline(self):
        frames, dates, closes = self._scenario_frames()
        s_str = dates[-1].strftime("%Y-%m-%d")
        peak_str = dates[39].strftime("%Y-%m-%d")
        entry = dates[0].strftime("%Y-%m-%d")

        # The per-bar candidate at the peak vs at the end of the decline.
        rows = session_mod.rows_for(frames, "X")
        from app.analysis import signal_series
        series = signal_series(rows)["atr_stop"]
        rows_by_time = {r["time"]: float(v) for r, v in zip(rows, series,
                                                            strict=True)
                        if pd.notna(v)}
        cand_at_peak = rows_by_time[peak_str]
        cand_at_end = rows_by_time[s_str]
        assert cand_at_end < cand_at_peak, \
            "scenario must actually decline past the peak candidate"

        # Ratchet anchored at entry = max over ALL bars since entry.
        ratchet = sim._atr_ratchets(frames, {"X": entry}, s_str)["X"]
        assert ratchet is not None
        assert ratchet >= cand_at_peak - 1e-9, \
            "the ratchet reaches at least the peak-era candidate"
        assert ratchet > cand_at_end, \
            "the ratchet NEVER loosens to the declining current level"

    def test_ratchet_monotone_across_cycle_boundaries(self):
        """Growing window → non-decreasing level: the value at a later S is
        never below the value at an earlier S (same entry anchor)."""
        frames, dates, _ = self._scenario_frames()
        entry = dates[0].strftime("%Y-%m-%d")
        levels = []
        for s_i in (35, 45, 55, 69):
            s_str = dates[s_i].strftime("%Y-%m-%d")
            levels.append(sim._atr_ratchets(frames, {"X": entry}, s_str)["X"])
        assert all(lv is not None for lv in levels)
        assert all(b >= a - 1e-9
                   for a, b in zip(levels, levels[1:], strict=False)), levels

    def test_ratchet_anchor_is_the_entry_session(self):
        """A late entry only sees bars since ITS entry: its ratchet is
        strictly lower than an early entry's on the same declining tape."""
        frames, dates, _ = self._scenario_frames()
        s_str = dates[-1].strftime("%Y-%m-%d")
        early = sim._atr_ratchets(
            frames, {"X": dates[0].strftime("%Y-%m-%d")}, s_str)["X"]
        late = sim._atr_ratchets(
            frames, {"X": dates[50].strftime("%Y-%m-%d")}, s_str)["X"]
        assert early is not None and late is not None
        assert late < early, "the entry anchor bounds the ratchet window"

    def test_no_entry_anchor_no_ratchet(self):
        frames, dates, _ = self._scenario_frames()
        s_str = dates[-1].strftime("%Y-%m-%d")
        out = sim._atr_ratchets(frames, {"X": None, "Y": None}, s_str)
        assert out == {"X": None, "Y": None}

    async def test_decision_inputs_ratchet_anchored_on_opened_at(self, mem_db):
        """End to end: a DB position's opened_at anchors the ratchet that
        _decision_inputs hands the engine. The tape: 200 quiet base days,
        then the rise to ~140 and the decline to ~65, all ending on S."""
        from app.db import Session

        end = datetime.strptime(S_STR, "%Y-%m-%d").date()
        dates = [end - pd.Timedelta(days=269 - i) for i in range(270)]

        def close_at(i: int) -> float:
            if i < 200:
                return 100.0                      # quiet base
            if i < 240:
                return 100.0 + (i - 199)          # rise 101 .. 140
            return 140.0 - 2.5 * (i - 239)        # decline 137.5 .. 65

        async with Session() as s:
            for i, d in enumerate(dates):
                c = close_at(i)
                s.add(_candle("X", d.strftime("%Y-%m-%d"), c - 0.5,
                              c + 1.0, c - 1.0, c))
            s.add(SimPosition(ticker="X", shares=10.0, avg_cost=105.0,
                              opened_at=datetime(2026, 3, 2),
                              stop_price=88.0))
            await s.commit()

        entry = dates[200].strftime("%Y-%m-%d")  # the rise's first session
        d = await sim._decision_inputs(["X"], held_entries={"X": entry},
                                       now=NOW_F)
        assert d.prices["X"] == pytest.approx(close_at(269)), \
            "the decision close is S's close (the declined price)"
        expected = sim._atr_ratchets(
            await session_mod.load_open_frames(["X"], asof=pd.Timestamp(F_STR)),
            {"X": entry}, S_STR)["X"]
        assert d.ratchets["X"] == expected
        # The ratchet keeps the rise-era peak level, far above the tape's
        # declined close (which is why it must never loosen).
        assert d.ratchets["X"] is not None
        assert d.ratchets["X"] > d.prices["X"], \
            "peak-era ratchet level above the declined decision close"


# ---------------------------------------------------------------------------
# Rule 5 — the percent stop is frozen at entry, top-ups cannot loosen it
# ---------------------------------------------------------------------------

class TestFrozenPercentStopTopUp:
    async def _pos(self, session_factory):
        async with session_factory() as s:
            return await s.scalar(
                select(SimPosition).where(SimPosition.ticker == "X"))

    async def test_topup_never_loosens_frozen_stop(self, with_cash):
        """A cheaper top-up blends the average cost down but must NOT
        re-anchor the stop to that lower blend (85.0 stays 85.0)."""
        await sim._exec_buy("X", 100.0, 1000.0, "entry", stop_price=85.0)
        pos = await self._pos(with_cash)
        assert pos.stop_price == pytest.approx(85.0)
        first_shares = pos.shares

        await sim._exec_buy("X", 50.0, 1000.0, "top-up at a much lower price")
        pos = await self._pos(with_cash)
        assert pos.shares > first_shares
        assert pos.avg_cost < 100.0, "the top-up blended the avg cost down"
        assert pos.stop_price == pytest.approx(85.0), \
            "the frozen stop survives the top-up untouched"
        assert pos.stop_price != pytest.approx(pos.avg_cost * 0.85, abs=0.01), \
            "never re-anchored to the blended average cost"

    async def test_new_position_default_stop_from_entry_fill(self, with_cash,
                                                             monkeypatch):
        monkeypatch.setattr(sim.settings, "sim_stop_pct", 15.0)
        t = await sim._exec_buy("Y", 120.0, 1000.0, "entry")
        assert t is not None
        async with with_cash() as s:
            pos = await s.scalar(select(SimPosition)
                                 .where(SimPosition.ticker == "Y"))
        assert pos.stop_price == pytest.approx(120.0 * 0.85), \
            "the default freeze anchors on the entry fill price"

    async def test_execute_proposed_freezes_the_proposals_entry_stop(
            self, with_cash, monkeypatch):
        """_execute_proposed passes the proposal's entry_stop through, so the
        contract BUY's entry-anchored freeze (F's open) wins over the
        settings-based default."""
        monkeypatch.setattr(sim.settings, "sim_stop_pct", 10.0)
        monkeypatch.setattr(sim.settings, "sim_max_positions", 10)
        proposal = {"ticker": "N", "side": "BUY", "price": 99.5,
                    "budget": 500.0, "reason": "contract BUY at F's open",
                    "entry_stop": 99.5 * 0.85}
        executed = await sim._execute_proposed([proposal])
        assert len(executed) == 1
        async with with_cash() as s:
            pos = await s.scalar(select(SimPosition)
                                 .where(SimPosition.ticker == "N"))
        assert pos.stop_price == pytest.approx(99.5 * 0.85), \
            "the proposal's entry stop is persisted, not the settings default"
        assert pos.stop_price != pytest.approx(99.5 * 0.90, abs=0.01)

    async def test_engine_stop_out_uses_frozen_level(self, mem_db):
        """End to end through _deterministic_propose: a held position with a
        frozen stop produces a SELL whose price comes from the standing-order
        rule (F's open after a gap, the level on an intrabar breach)."""
        held = SimPosition(ticker="X", shares=10.0, avg_cost=100.0,
                           opened_at=datetime(2026, 3, 2), stop_price=85.0,
                           thesis="entry")
        async with mem_db() as s:
            s.add(held)
            await s.commit()

        valuation = {"cash": 1000.0, "positions": [], "positions_value": 0.0,
                     "total_equity": 1000.0, "allowance_total": 0.0}
        hold = {"action": "HOLD", "strength": 30, "reason": "HOLD",
                "snapshot": {"close": 100.0, "atr_stop": None, "atr14": 2.0,
                             "net_score": 25, "trend_up": True,
                             "trend_down": False, "run_5d": 0.0,
                             "dist_above": 5.0, "weekly_trend_up": True}}
        decision = sim._DecisionInputs(
            f_ts=pd.Timestamp(F_STR), s_ts=pd.Timestamp(S_STR),
            signals={"X": hold}, prices={"X": 100.0}, session_bars={},
            ratchets={"X": None})

        # Gap through the level: open 80 <= 85 → fill AT THE OPEN.
        decision.session_bars = {"X": {"open": 80.0, "low": 79.0,
                                       "close": 81.0}}
        out = await sim._deterministic_propose(valuation, decision=decision)
        sells = [p for p in out if p["side"] == "SELL"]
        assert len(sells) == 1
        assert sells[0]["price"] == 80.0
        assert sells[0]["reason"].lower().startswith("initial stop")
        assert "filled at the open" in sells[0]["reason"]

        # Intrabar breach: open above the level, low reaches it → fill AT
        # THE LEVEL.
        decision.session_bars = {"X": {"open": 90.0, "low": 84.0,
                                       "close": 95.0}}
        out = await sim._deterministic_propose(valuation, decision=decision)
        sells = [p for p in out if p["side"] == "SELL"]
        assert len(sells) == 1
        assert sells[0]["price"] == 85.0
        assert "filled at the level" in sells[0]["reason"]

        # No breach: no fill, the position rides.
        decision.session_bars = {"X": {"open": 90.0, "low": 86.0,
                                       "close": 95.0}}
        out = await sim._deterministic_propose(valuation, decision=decision)
        assert out == []


# ---------------------------------------------------------------------------
# Rule 2 — the per-session idempotence guard
# ---------------------------------------------------------------------------

class TestSessionGuardIdempotent:
    def _patch_cycle(self, monkeypatch, tmp_path, fixed):
        """Point the guard state file at tmp_path and stub every cycle stage
        except the decision/fill step (inputs are faked: the guard test
        exercises the guard, not the indicator math)."""
        monkeypatch.setattr(sim, "_STATE_FILE", tmp_path / "state" / "sim_state.json")
        sim._reset_session_guard()
        monkeypatch.setattr(sim, "_utcnow", lambda: fixed)
        monkeypatch.setattr(sim, "deposit_allowance",
                            AsyncMock(return_value={"deposited": False}))
        monkeypatch.setattr(sim, "_candidate_tickers", AsyncMock(return_value=[]))
        monkeypatch.setattr(sim, "refresh_many",
                            AsyncMock(return_value=([], [])))
        monkeypatch.setattr(sim.settings, "sim_benchmark_enabled", False)
        monkeypatch.setattr(sim.settings, "sim_strategy", "deterministic")
        monkeypatch.setattr(sim, "_benchmark_snapshot", AsyncMock())

        async def fake_valuate():
            return {"cash": 1000.0, "positions": [], "positions_value": 0.0,
                    "total_equity": 1000.0, "allowance_total": 0.0}

        monkeypatch.setattr(sim, "valuate", fake_valuate)

        async def fake_inputs(tickers, held_entries=None, now=None):
            return sim._DecisionInputs(
                f_ts=pd.Timestamp(F_STR), s_ts=pd.Timestamp(S_STR),
                signals={}, prices={}, session_bars={}, ratchets={})

        monkeypatch.setattr(sim, "_decision_inputs", fake_inputs)

    async def test_second_cycle_same_session_is_a_noop(self, mem_db,
                                                       monkeypatch, tmp_path):
        """Exactly one decision/fill cycle per session: the second call bails
        before touching anything, and backfill/reset_sim re-arm the guard."""
        fixed = NOW_F
        self._patch_cycle(monkeypatch, tmp_path, fixed)
        key = fixed.date().isoformat()

        first = await sim.run_cycle()
        assert first.get("skipped") is not True, first
        state = json.loads((tmp_path / "state" / "sim_state.json").read_text())
        assert state["last_session"] == key

        second = await sim.run_cycle()
        assert second == {"skipped": True,
                          "reason": f"session {key} already processed"}

        # reset_sim/backfill clear the guard so the next cycle runs again.
        sim._reset_session_guard()
        third = await sim.run_cycle()
        assert third.get("skipped") is not True, third

    async def test_guard_helpers(self, monkeypatch, tmp_path):
        """The guard helpers on their own: stamping marks the session as
        processed, reset re-arms it."""
        monkeypatch.setattr(sim, "_STATE_FILE", tmp_path / "sim_state.json")
        sim._reset_session_guard()
        assert sim._session_guard_passed("2026-03-09") is True
        sim._stamp_session_guard("2026-03-09")
        assert sim._session_guard_passed("2026-03-09") is False
        assert sim._session_guard_passed("2026-03-10") is True
        sim._reset_session_guard()
        assert sim._session_guard_passed("2026-03-09") is True
        # A missing/unreadable state file behaves like "not stamped".
        monkeypatch.setattr(sim, "_STATE_FILE", tmp_path / "nope" / "x.json")
        assert sim._session_guard_passed("2026-03-09") is True
