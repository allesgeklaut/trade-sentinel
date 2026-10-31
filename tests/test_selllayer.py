"""Tests for the LLM sell layer (app/selllayer.py) — slice 1: consult +
EXECUTE in the live accounts (types A/B/C).

All tests are OFFLINE: every consult uses a mock transport (or the no-backend
short-circuit), and the LLM backend fixture only configures a fake backend so
``llm.current_backend()`` is non-empty — no HTTP is ever made. In-memory
SQLite throughout.

Covered (brief §TESTS):

  1. the A overlay on a canned tape — CONFIRM executes as today, DEFER
     suppresses + persists the re-armed level, the next breach of the re-armed
     level fires, the percent stop is never loosened, the defer cap executes
     at the cap;
  2. the B overlay at a canned rebuild — DEFER retains the name, the displaced
     add is not bought, capacity <= target_n, the MonthlyRebalance row records
     engine picks + executed picks, CONFIRM is bit-identical;
  3. the C overlay — a thesis break -> EXIT creates the unfilled pending row,
     the NEXT cycle fills it at that session's strict open with
     filled_at >= decided_at, HOLD does nothing, the dedupe prevents a second
     consult for the same filing;
  4. fail-safes — no-backend short-circuit (zero transport calls), transport
     exception, wait_for timeout, malformed JSON, refusal;
  5. bit-identical OFF parity — knobs off -> identity overlays, zero ledger
     rows, zero transport calls;
  6. the contamination proof — a backfill run with an exploding mock transport
     makes ZERO consult attempts;
  7. the ledger — append-only rows with the full field set; the additive
     SimPosition ALTER is idempotent (init_db twice);
  8. determinism of the record payloads modulo latency/wall-clock.
"""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from types import SimpleNamespace

import pandas as pd
import pytest
from sqlalchemy import StaticPool, func, inspect, select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app import monthly, selllayer, sim
from app.config import settings
from app.db import (
    Base,
    Candle,
    LlmSellDecision,
    MonthlyAccount,
    MonthlyPosition,
    MonthlyRebalance,
    MonthlyTrade,
    SimAccount,
    SimPosition,
)

# Cycle dates: S = Friday 2026-03-06, F = Monday 2026-03-09.
S_STR = "2026-03-06"
F_STR = "2026-03-09"


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

@pytest.fixture
async def mem_db(monkeypatch):
    """In-memory SQLite for every engine the sell layer touches. selllayer
    resolves ``db.Session`` at call time, so patching ``app.db.Session`` (as
    the existing fixtures do) covers it too."""
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    import app.db as db_mod
    import app.edgar as edgar_mod
    import app.fundamentals as fund_mod
    monkeypatch.setattr(sim, "Session", session_factory)
    monkeypatch.setattr(db_mod, "Session", session_factory)
    monkeypatch.setattr(monthly, "Session", session_factory)
    monkeypatch.setattr(fund_mod, "Session", session_factory)
    monkeypatch.setattr(edgar_mod, "Session", session_factory)

    async with session_factory() as s:
        s.add(SimAccount(id=1, cash=settings.sim_start_cash, last_allowance_month=None))
        s.add(MonthlyAccount(id=1, cash=settings.sim_monthly_start_cash))
        await s.commit()
    yield session_factory
    await engine.dispose()


@pytest.fixture
def llm_backend(monkeypatch, tmp_path):
    """Configure a fake LLM backend so ``llm.current_backend()`` is non-empty
    (the no-backend short-circuit must NOT fire); no HTTP is ever made — the
    tests inject the transport."""
    from app import llm as llm_mod
    monkeypatch.setattr(settings, "llm_backends", json.dumps([
        {"name": "test", "type": "ollama", "url": "http://test:11434", "model": "m1"},
    ]))
    monkeypatch.setattr(llm_mod, "_DEFAULT_STATE_FILE", tmp_path / "llm_state.json")


@pytest.fixture
def no_backend(monkeypatch, tmp_path):
    """No LLM backend configured at all."""
    from app import llm as llm_mod
    monkeypatch.setattr(settings, "llm_backends", "")
    monkeypatch.setattr(settings, "ollama_model", "")
    monkeypatch.setattr(llm_mod, "_DEFAULT_STATE_FILE", tmp_path / "none.json")


def mock_transport(reply: str | Exception, calls: list | None = None):
    """A transport returning a canned reply (or raising), counting calls."""
    async def transport(messages, model_id, temperature, timeout_s):
        if calls is not None:
            calls.append({"messages": messages, "model_id": model_id,
                          "timeout_s": timeout_s})
        if isinstance(reply, Exception):
            raise reply
        return reply
    return transport


def _series(ticker: str, n: int = 230, start: str = "2025-04-01",
            f_open: float = 90.0) -> pd.DataFrame:
    """A per-ticker candle series (time as 'YYYY-MM-DD' strings) ending at S,
    plus F's bar with the given open."""
    dates = pd.bdate_range(start, periods=n)
    rows = []
    for i, d in enumerate(dates):
        c = 100.0 + 0.05 * i
        rows.append({"time": d.strftime("%Y-%m-%d"), "open": c - 0.3,
                     "high": c + 0.5, "low": c - 0.6, "close": c, "volume": 1000.0})
    rows.append({"time": F_STR, "open": f_open, "high": f_open + 1.0,
                 "low": f_open - 1.0, "close": f_open + 0.5, "volume": 1000.0})
    return pd.DataFrame(rows)


def _decision():
    return SimpleNamespace(
        f_ts=pd.Timestamp(F_STR), s_ts=pd.Timestamp(S_STR),
        signals={}, prices={}, session_bars={}, ratchets={})


def _stop_proposal(ticker="X", level=95.0, label="ATR stop", price=90.0) -> dict:
    return {"ticker": ticker, "side": "SELL", "price": price, "shares": None,
            "stop_fire": {"level": level, "label": label},
            "reason": f"{label} standing order: open {price:.2f} <= level {level:.2f}"}


async def _add_sim_position(ticker="X", shares=10.0, avg_cost=100.0,
                            stop_price=85.0, opened_at="2026-01-02",
                            defer_level=None):
    from app.db import Session
    async with Session() as s:
        s.add(SimPosition(ticker=ticker, shares=shares, avg_cost=avg_cost,
                          stop_price=stop_price,
                          opened_at=datetime.strptime(opened_at, "%Y-%m-%d"),
                          stop_defer_level=defer_level,
                          stop_defer_set_at=(datetime.now(UTC) if defer_level else None)))
        await s.commit()


async def _ledger_rows():
    from app.db import Session
    async with Session() as s:
        return (await s.scalars(
            select(LlmSellDecision).order_by(LlmSellDecision.id))).all()


# ---------------------------------------------------------------------------
# 4. Fail-safes of the live consult runner
# ---------------------------------------------------------------------------

class TestConsultFailSafes:
    async def test_no_backend_short_circuits_before_any_call(self, mem_db, no_backend):
        calls: list = []
        rec = await selllayer.consult_live("A:X:2026-03-06", "A", {"ticker": "X"},
                                           transport=mock_transport('{"decision": "DEFER", "rearm_mult": 2.0}', calls))
        assert calls == []                      # ZERO transport calls
        assert rec["decision"] == "CONFIRM"     # fail-safe: the engine's rule stands
        assert rec["fail_safe"] is True and rec["fail_reason"] == "no backend"
        assert rec["backend"] == "" and rec["model"] == ""

    async def test_transport_exception_fails_safe(self, mem_db, llm_backend):
        rec = await selllayer.consult_live(
            "A:X:2026-03-06", "A", {"ticker": "X"},
            transport=mock_transport(RuntimeError("proxy down")))
        assert rec["decision"] == "CONFIRM" and rec["fail_safe"] is True
        assert rec["parsed"] is False
        assert "RuntimeError" in (rec["fail_reason"] or "")
        assert rec["backend"] == "test" and rec["model"] == "m1"

    async def test_wait_for_timeout_fails_safe(self, mem_db, llm_backend):
        async def slow(messages, model_id, temperature, timeout_s):
            await asyncio.sleep(5.0)
            return '{"decision": "DEFER", "rearm_mult": 2.0}'
        rec = await selllayer.consult_live("B:X:2026-03-09", "B", {"ticker": "X"},
                                           transport=slow, timeout_s=0.05)
        assert rec["decision"] == "CONFIRM" and rec["fail_safe"] is True
        assert rec["fail_reason"] == "timeout"

    async def test_malformed_and_refusal_fail_safe(self, mem_db, llm_backend):
        for text in ("not json at all", '{"decision": "MAYBE"}', ""):
            rec = await selllayer.consult_live("C:X:2026-03-09", "C", {"ticker": "X"},
                                               transport=mock_transport(text))
            # type C's fail-safe is HOLD (the live book waits for month-end)
            assert rec["decision"] == "HOLD" and rec["fail_safe"] is True
            assert rec["parsed"] is False

    async def test_valid_defer_parses_with_clamped_rearm(self, mem_db, llm_backend):
        rec = await selllayer.consult_live(
            "A:X:2026-03-06", "A", {"ticker": "X"},
            transport=mock_transport('{"decision": "DEFER", "rearm_mult": 9.0, "rationale": "hold"}'))
        assert rec["decision"] == "DEFER" and rec["parsed"] is True
        assert rec["params"]["rearm_mult"] == 3.0  # clamped to [1.0, 3.0]
        assert rec["rationale"] == "hold"


# ---------------------------------------------------------------------------
# 1. Type A — the sim's standing-stop overlay
# ---------------------------------------------------------------------------

class TestTypeAOverlay:
    async def test_confirm_keeps_proposal_and_logs(self, mem_db, llm_backend):
        await _add_sim_position()
        calls: list = []
        proposals = [_stop_proposal()]
        out = await selllayer.overlay_type_a(
            proposals, _decision(),
            transport=mock_transport('{"decision": "CONFIRM", "rationale": "let it rip"}',
                                     calls),
            series={"X": _series("X")})
        assert out == proposals and len(calls) == 1
        rows = await _ledger_rows()
        assert len(rows) == 1
        r = rows[0]
        assert (r.event_type, r.engine, r.ticker, r.decision) == ("A", "sim", "X", "CONFIRM")
        assert r.parsed is True and r.fail_safe is False
        assert r.prompt_version == "sellshadow/1"
        assert json.loads(r.engine_branch)["action"] == "execute standing stop"
        assert json.loads(r.context)["ticker"] == "X"
        assert r.filled_at is None and r.shares is None  # NULL together until executed

    async def test_defer_suppresses_and_persists_rearmed_level(self, mem_db, llm_backend):
        await _add_sim_position()
        level = 95.0
        out = await selllayer.overlay_type_a(
            [_stop_proposal(level=level)], _decision(),
            transport=mock_transport('{"decision": "DEFER", "rearm_mult": 1.5}'),
            series={"X": _series("X")})
        assert out == []                        # the sale is suppressed
        from app.db import Session
        async with Session() as s:
            pos = await s.scalar(select(SimPosition).where(SimPosition.ticker == "X"))
            # 5A-1 re-arm: level x (1 - (mult-1) x REARM_WIDEN) = 95 x 0.95
            assert pos.stop_defer_level == pytest.approx(level * 0.95)
            assert pos.stop_defer_set_at is not None
            assert pos.stop_price == 85.0      # the frozen percent stop is untouched
        rows = await _ledger_rows()
        assert len(rows) == 1 and rows[0].decision == "DEFER"
        branch = json.loads(rows[0].engine_branch)
        assert branch["action"] == "defer"
        assert branch["rearmed_level"] == pytest.approx(level * 0.95)
        assert branch["rearm_mult"] == 1.5

    async def test_defer_cap_executes_without_consult(self, mem_db, llm_backend, monkeypatch):
        await _add_sim_position(opened_at="2026-01-02")
        # two consecutive A-DEFER rows since the opening = the default cap (2)
        from app.db import Session
        async with Session() as s:
            for i in range(2):
                s.add(LlmSellDecision(
                    decided_at=datetime(2026, 3, 2 + i, 23, 30, tzinfo=UTC),
                    session_date=f"2026-03-0{2 + i}", event_type="A", engine="sim",
                    event_id=f"A:X:2026-03-0{2 + i}", ticker="X", decision="DEFER",
                    rationale="", prompt_version="sellshadow/1", parsed=True,
                    fail_safe=False, context="{}", engine_branch="{}",
                    backend="test", model="m1", latency_ms=1.0))
            await s.commit()
        calls: list = []
        out = await selllayer.overlay_type_a(
            [_stop_proposal()], _decision(),
            transport=mock_transport('{"decision": "DEFER", "rearm_mult": 2.0}', calls),
            series={"X": _series("X")})
        assert out != []                        # the stop executes regardless
        assert calls == []                      # no consult at the cap
        rows = await _ledger_rows()
        assert len(rows) == 3
        cap_row = rows[-1]
        assert cap_row.decision == "CONFIRM" and cap_row.fail_safe is True
        assert cap_row.fail_reason == "defer cap"

    async def test_knob_off_is_identity(self, mem_db, llm_backend, monkeypatch):
        monkeypatch.setattr(settings, "sim_llm_sell_layer", False)
        await _add_sim_position()
        proposals = [_stop_proposal()]
        calls: list = []
        out = await selllayer.overlay_type_a(
            proposals, _decision(),
            transport=mock_transport('{"decision": "DEFER", "rearm_mult": 2.0}', calls),
            series={"X": _series("X")})
        assert out is proposals and calls == []
        assert await _ledger_rows() == []


class TestTypeAStandingStopSemantics:
    """strategy-level: while a defer is active the standing stop is the frozen
    percent stop + the DEFERRED ATR level; the next breach of the re-armed
    level fires; the percent stop is never loosened."""

    def _params(self):
        from app.strategy import StrategyParams
        return StrategyParams(stop_type="percent", stop_pct=15.0, use_atr_stop=True)

    def _run(self, pos, f_open, f_low):
        from app.strategy import propose_trades
        return propose_trades(
            [pos], 10_000.0, {"X": 100.0},
            {"X": {"action": "HOLD", "strength": 0, "reason": "", "snapshot": {}}},
            self._params(),
            session_bars={"X": {"open": f_open, "low": f_low, "close": f_open}})

    def test_rearmed_level_fires_on_next_breach(self):
        pos = {"ticker": "X", "shares": 10.0, "avg_cost": 100.0, "stop_price": 85.0,
               "atr_stop_level": 92.0, "stop_defer_level": 90.25}
        # open 89 <= deferred 90.25 -> stop fires at the open, labeled deferred
        out = self._run(pos, 89.0, 88.0)
        assert len(out) == 1 and out[0]["side"] == "SELL"
        assert out[0]["stop_fire"] == {"level": 90.25, "label": "ATR stop (deferred)"}
        assert out[0]["price"] == 89.0

    def test_between_defer_and_ratchet_does_not_fire(self):
        pos = {"ticker": "X", "shares": 10.0, "avg_cost": 100.0, "stop_price": 85.0,
               "atr_stop_level": 92.0, "stop_defer_level": 90.25}
        # open 91: above the deferred level -> no fire, even though the old
        # ratchet (92) would have fired without the defer
        assert self._run(pos, 91.0, 90.5) == []
        # same input WITHOUT the defer -> the ratchet fires (regression pin)
        pos2 = dict(pos, stop_defer_level=None)
        out2 = self._run(pos2, 91.0, 90.5)
        assert len(out2) == 1 and out2[0]["stop_fire"]["label"] == "ATR stop"

    def test_percent_stop_never_loosened(self):
        # the frozen percent stop (95) is above the deferred ATR level: it
        # still governs and fires — the defer never loosens it
        pos = {"ticker": "X", "shares": 10.0, "avg_cost": 100.0, "stop_price": 95.0,
               "atr_stop_level": 92.0, "stop_defer_level": 90.25}
        out = self._run(pos, 90.0, 89.0)
        assert len(out) == 1
        assert out[0]["stop_fire"] == {"level": 95.0, "label": "Initial stop"}


# ---------------------------------------------------------------------------
# 2. Type B — the monthly rebuild's band-drop overlay
# ---------------------------------------------------------------------------

def _monthly_tape(n_days=400, tickers=("AAA", "BBB", "CCC", "DDD")):
    import numpy as np
    dates = pd.bdate_range("2024-06-03", periods=n_days)
    rng = np.random.default_rng(7)
    close = pd.DataFrame(
        rng.normal(0.0004, 0.012, (n_days, len(tickers))).cumsum(axis=0) + 100,
        index=dates, columns=list(tickers))
    vol = pd.DataFrame(5e6, index=dates, columns=list(tickers))

    def fund_rec(q):
        return {
            "StockholdersEquity": [{"start": None, "end": "2024-06-30",
                                    "filed": "2024-08-14", "val": 1e9}],
            "NetIncomeLoss": [{"start": "2024-07-01", "end": "2025-06-30",
                               "filed": "2025-08-14", "val": 1e8 * q}],
            "NetCashProvidedByUsedInOperatingActivities": [
                {"start": "2024-07-01", "end": "2025-06-30", "filed": "2025-08-14",
                 "val": 5e8}],
            "PaymentsToAcquirePropertyPlantAndEquipment": [
                {"start": "2024-07-01", "end": "2025-06-30", "filed": "2025-08-14",
                 "val": -1e8}],
            "CommonStockSharesOutstanding": [{"start": None, "end": "2024-06-30",
                                              "filed": "2024-08-14", "val": 1e8}],
        }
    fund = {t: fund_rec(10 - i) for i, t in enumerate(tickers)}
    return dates, close, vol, fund


class TestTypeBOverlay:
    async def _setup_book(self, mem_db, held=("AAA", "BBB")):
        from app.db import Session
        async with Session() as s:
            acc = await s.get(MonthlyAccount, 1)
            acc.cash = 100_000.0
            for t in held:
                s.add(MonthlyPosition(ticker=t, shares=100.0, avg_cost=100.0,
                                      opened_at=datetime(2025, 1, 2)))
            await s.commit()

    async def _run_overlay(self, mem_db, transport, target_n=3):
        dates, close, vol, fund = _monthly_tape()
        f_date = dates[-1]
        frame = await asyncio.to_thread(
            monthly.eligible_frame, pd.Timestamp(f_date), close, vol, fund)
        held_before = ["AAA", "BBB"]
        picks = ["AAA", "CCC", "DDD"]          # BBB band-dropped; CCC/DDD are adds
        fill_map = {"AAA": 100.0, "BBB": 99.0, "CCC": 50.0, "DDD": 51.0}
        return await selllayer.overlay_type_b(
            held_before=held_before, picks=picks, fill_map=fill_map, frame=frame,
            close=close, fund=fund, members=list(close.columns),
            me_ts=pd.Timestamp(f_date), f_date=f_date,
            target_n=target_n, hold_band=4, transport=transport), \
            (held_before, picks, f_date)

    async def test_defer_retains_and_displaces_last_add(self, mem_db, llm_backend):
        await self._setup_book(mem_db)
        (executed, detail), (held_before, picks, f_date) = await self._run_overlay(
            mem_db, mock_transport('{"decision": "DEFER", "rationale": "keep it"}'))
        # the last-added new name (DDD) is not bought; BBB is retained
        assert executed == ["AAA", "CCC", "BBB"]
        assert len(executed) <= 3                       # capacity stays target_n
        assert detail["engine_picks"] == picks
        assert detail["executed_picks"] == executed
        assert detail["retained"] == ["BBB"]
        rows = await _ledger_rows()
        assert len(rows) == 1
        r = rows[0]
        assert (r.event_type, r.engine, r.ticker, r.decision) == ("B", "monthly", "BBB", "DEFER")
        assert r.prompt_version == "sellshadow-b/1"
        assert json.loads(r.engine_branch)["action"] == "release at rebuild"
        assert json.loads(r.context)["fill_price"] == 99.0

    async def test_confirm_is_bit_identical(self, mem_db, llm_backend):
        await self._setup_book(mem_db)
        (executed, detail), (_, picks, _) = await self._run_overlay(
            mem_db, mock_transport('{"decision": "CONFIRM"}'))
        assert executed == picks
        assert detail["executed_picks"] == picks
        rows = await _ledger_rows()
        assert len(rows) == 1 and rows[0].decision == "CONFIRM" and rows[0].parsed is True

    async def test_candidate_without_quote_is_not_consulted(self, mem_db, llm_backend):
        await self._setup_book(mem_db)
        dates, close, vol, fund = _monthly_tape()
        f_date = dates[-1]
        frame = await asyncio.to_thread(
            monthly.eligible_frame, pd.Timestamp(f_date), close, vol, fund)
        calls: list = []
        executed, detail = await selllayer.overlay_type_b(
            held_before=["AAA", "BBB"], picks=["AAA", "CCC"],
            fill_map={"AAA": 100.0, "CCC": 50.0},     # BBB has NO quote -> pending
            frame=frame, close=close, fund=fund, members=list(close.columns),
            me_ts=pd.Timestamp(f_date), f_date=f_date, target_n=3, hold_band=4,
            transport=mock_transport('{"decision": "DEFER"}', calls))
        assert executed == ["AAA", "CCC"] and detail is None and calls == []
        assert await _ledger_rows() == []

    async def test_knob_off_is_identity(self, mem_db, llm_backend, monkeypatch):
        monkeypatch.setattr(settings, "sim_monthly_llm_sells", False)
        await self._setup_book(mem_db)
        calls: list = []
        (executed, detail), (_, picks, _) = await self._run_overlay(
            mem_db, mock_transport('{"decision": "DEFER"}', calls))
        assert executed == picks and detail is None and calls == []
        assert await _ledger_rows() == []

    async def _run_with_frame(self, mem_db, frame, transport, hold_band=3):
        """Run the B overlay with a hand-built frame (deterministic qv order)."""
        await self._setup_book(mem_db)
        dates, close, vol, fund = _monthly_tape()
        f_date = dates[-1]
        held_before = ["AAA", "BBB"]
        picks = ["AAA", "CCC", "DDD"]          # BBB is the band-drop candidate
        fill_map = {"AAA": 100.0, "BBB": 99.0, "CCC": 50.0, "DDD": 51.0}
        return await selllayer.overlay_type_b(
            held_before=held_before, picks=picks, fill_map=fill_map, frame=frame,
            close=close, fund=fund, members=list(close.columns),
            me_ts=pd.Timestamp(f_date), f_date=f_date,
            target_n=3, hold_band=hold_band, transport=transport)

    @staticmethod
    def _fake_frame(order) -> pd.DataFrame:
        """A frame whose _qv_order is exactly ``order`` (mom desc drives it)."""
        moms = {t: 0.4 - 0.1 * i for i, t in enumerate(order)}
        return pd.DataFrame({
            "roe": [0.1] * len(order), "p_fcf": [20.0] * len(order),
            "mcap": [1e10] * len(order), "dollar_vol": [5e7] * len(order),
            "mom": [moms[t] for t in order], "price": [100.0] * len(order),
            "vol": [0.2] * len(order), "eligible": [True] * len(order),
        }, index=pd.Index(order, name="ticker"))

    async def test_band_excludes_name_true_when_outside_band(self, mem_db, llm_backend):
        # order AAA > CCC > DDD > BBB, hold_band 3 -> band excludes BBB
        frame = self._fake_frame(["AAA", "CCC", "DDD", "BBB"])
        await self._run_with_frame(mem_db, frame,
                                   mock_transport('{"decision": "CONFIRM"}'),
                                   hold_band=3)
        rows = await _ledger_rows()
        assert len(rows) == 1
        assert json.loads(rows[0].context)["band_excludes_name"] is True

    async def test_band_excludes_name_false_in_capacity_edge(self, mem_db, llm_backend):
        # capacity edge: BBB IS still inside the band (band = all 4) but is
        # not picked (_band_fill truncates band[:target_n]) — the context must
        # not claim the band excluded it.
        frame = self._fake_frame(["AAA", "BBB", "CCC", "DDD"])
        await self._run_with_frame(mem_db, frame,
                                   mock_transport('{"decision": "CONFIRM"}'),
                                   hold_band=4)
        rows = await _ledger_rows()
        assert len(rows) == 1
        assert json.loads(rows[0].context)["band_excludes_name"] is False


class TestTypeBRebalanceIntegration:
    """Full ``run_rebalance`` with the type-B overlay wired in: the audit row
    records engine picks + executed picks, and the retained name is neither
    sold nor displaced."""

    async def test_rebalance_audit_and_retention(self, mem_db, llm_backend, monkeypatch):
        from app.screener import tickers as universe_tickers
        uni = universe_tickers("diversified-plus")
        tickers = uni[:12]
        dates = pd.bdate_range("2024-06-03", periods=400)
        prices = {t: 100.0 + i for i, t in enumerate(tickers)}
        close = pd.DataFrame(prices, index=dates)

        async with mem_db() as s:
            for t in tickers:
                for ts, px in zip(dates, close[t], strict=False):
                    s.add(Candle(ticker=t, timestamp=ts.to_pydatetime(), open=px,
                                 high=px, low=px, close=px, volume=2e6))
            quality = {t: (10 - i) for i, t in enumerate(tickers)}
            for t in tickers:
                q = quality[t]
                s.add(__import__("app.db", fromlist=["Fundamental"]).Fundamental(
                    ticker=t, tag="StockholdersEquity", start=None, end="2024-06-30",
                    filed="2024-08-14", val=1e9, currency="USD"))
                s.add(__import__("app.db", fromlist=["Fundamental"]).Fundamental(
                    ticker=t, tag="NetIncomeLoss", start="2024-07-01", end="2025-06-30",
                    filed="2025-08-14", val=1e8 * q, currency="USD"))
                s.add(__import__("app.db", fromlist=["Fundamental"]).Fundamental(
                    ticker=t, tag="NetCashProvidedByUsedInOperatingActivities",
                    start="2024-07-01", end="2025-06-30", filed="2025-08-14",
                    val=5e8, currency="USD"))
                s.add(__import__("app.db", fromlist=["Fundamental"]).Fundamental(
                    ticker=t, tag="PaymentsToAcquirePropertyPlantAndEquipment",
                    start="2024-07-01", end="2025-06-30", filed="2025-08-14",
                    val=-1e8, currency="USD"))
                s.add(__import__("app.db", fromlist=["Fundamental"]).Fundamental(
                    ticker=t, tag="CommonStockSharesOutstanding", start=None,
                    end="2024-06-30", filed="2024-08-14", val=1e8, currency="shares"))
            await s.commit()

        async def fake_refresh_data(tickers_list):
            return [], {}
        monkeypatch.setattr(monthly, "refresh_data", fake_refresh_data)

        clock = dates[-1].to_pydatetime().replace(hour=23, minute=30, tzinfo=UTC)
        r1 = await monthly.run_rebalance(now=clock)
        assert r1.get("rebalanced") is True
        held = set(r1["picks"])
        assert len(held) == settings.sim_monthly_target_n

        # Force a re-decide with the engine dropping T05 and adding T10.
        dropped = sorted(held)[5]
        added = [t for t in tickers if t not in held][0]
        forced_picks = sorted((held - {dropped}) | {added})

        def fake_pick_portfolio(rebal_date, c, v, f, holdings=None, frame=None):
            return list(forced_picks), frame
        monkeypatch.setattr(monthly, "pick_portfolio", fake_pick_portfolio)

        # The sell layer's transport: DEFER the band-dropped name.
        monkeypatch.setattr(selllayer, "_llm_transport",
                            mock_transport('{"decision": "DEFER", "rationale": "retain"}'))

        r2 = await monthly.run_rebalance(force=True, now=clock)
        assert r2.get("rebalanced") is True
        # executed picks: T10 not bought, the dropped name retained
        assert r2["engine_picks"] == forced_picks
        assert sorted(r2["picks"]) == sorted(held)
        traded = {t["ticker"]: t["side"] for t in r2["trades"]}
        assert dropped not in traded and added not in traded

        from app.db import Session
        async with Session() as s:
            reb = (await s.scalars(select(MonthlyRebalance))).all()
            assert len(reb) == 1
            detail = json.loads(reb[0].sell_layer)
            assert detail["engine_picks"] == forced_picks
            assert sorted(detail["executed_picks"]) == sorted(held)
            assert detail["retained"] == [dropped]
        rows = await _ledger_rows()
        assert len(rows) == 1
        assert (rows[0].event_type, rows[0].ticker, rows[0].decision) == ("B", dropped, "DEFER")


# ---------------------------------------------------------------------------
# 3. Type C — the monthly book's nightly thesis-break step
# ---------------------------------------------------------------------------

def _broken_fund() -> dict:
    """AAA's fundamentals PUBLIC as of 2026-02-02: equity <= 0 (thesis break)."""
    return {"AAA": {
        "StockholdersEquity": [{"start": None, "end": "2025-12-31",
                                "filed": "2026-01-15", "val": -5.0}],
        "NetIncomeLoss": [{"start": "2025-01-01", "end": "2025-12-31",
                           "filed": "2026-01-15", "val": 1e8}],
        "CommonStockSharesOutstanding": [{"start": None, "end": "2025-12-31",
                                          "filed": "2026-01-15", "val": 1e8}],
    }}


def _c_tape():
    # 25 business days from 2026-01-02 covers both nightly sessions
    # (2026-02-02, 2026-02-03) plus the 2026-01-30 month-end.
    dates = pd.bdate_range("2026-01-02", periods=25)
    close = pd.DataFrame({"AAA": 100.0}, index=dates)
    vol = pd.DataFrame(1e6, index=dates, columns=["AAA"])
    return dates, close, vol


class TestTypeC:
    async def _seed(self, mem_db):
        from app.db import Session
        async with Session() as s:
            acc = await s.get(MonthlyAccount, 1)
            acc.cash = 50_000.0
            s.add(MonthlyPosition(ticker="AAA", shares=100.0, avg_cost=100.0,
                                  opened_at=datetime(2026, 1, 5)))
            await s.commit()

    async def test_exit_creates_unfilled_row_then_next_cycle_fills_at_open(
            self, mem_db, llm_backend):
        await self._seed(mem_db)
        dates, close, vol = _c_tape()
        fund = _broken_fund()
        night1 = datetime(2026, 2, 2, 23, 30, tzinfo=UTC)   # Monday
        res = await selllayer.run_type_c_step(
            now=night1, transport=mock_transport('{"decision": "EXIT", "rationale": "broken"}'),
            fund=fund, close=close, vol=vol)
        assert len(res["scans"]) == 1 and res["scans"][0]["decision"] == "EXIT"
        assert res["fills"] == []
        rows = await _ledger_rows()
        assert len(rows) == 1
        r = rows[0]
        assert (r.event_type, r.engine, r.ticker, r.decision) == ("C", "monthly", "AAA", "EXIT")
        assert r.event_id == "C:AAA:2026-01-15"
        assert r.prompt_version == "sellshadow-c/1"
        assert r.filled_at is None and r.price is None      # the pending record
        decided_at = r.decided_at

        # NEXT nightly cycle: the fill step sells at THAT session's strict open.
        from app.db import Session
        async with Session() as s:
            s.add(Candle(ticker="AAA", timestamp=datetime(2026, 2, 3), open=42.0,
                         high=43.0, low=41.0, close=42.5, volume=1000.0))
            await s.commit()
        night2 = datetime(2026, 2, 3, 23, 30, tzinfo=UTC)
        res2 = await selllayer.run_type_c_step(
            now=night2, transport=mock_transport('{"decision": "HOLD"}'),
            fund=fund, close=close, vol=vol)
        assert len(res2["fills"]) == 1
        assert res2["fills"][0]["price"] == 42.0            # the strict exact-date OPEN
        assert res2["scans"] == []                           # dedupe: no second consult

        rows = await _ledger_rows()
        assert len(rows) == 1                                # append-only: no new row
        r = rows[0]
        assert r.filled_at is not None and r.filled_at >= decided_at
        assert r.price == 42.0 and r.shares == 100.0 and r.cash_after is not None
        async with Session() as s:
            assert await s.scalar(select(func.count()).select_from(MonthlyPosition)) == 0
            trades = (await s.scalars(select(MonthlyTrade))).all()
            assert len(trades) == 1 and trades[0].side == "SELL"
            assert trades[0].reason == "thesis-break LLM exit"
            assert trades[0].price == 42.0

    async def test_hold_does_nothing_and_dedupes(self, mem_db, llm_backend):
        await self._seed(mem_db)
        dates, close, vol = _c_tape()
        fund = _broken_fund()
        night1 = datetime(2026, 2, 2, 23, 30, tzinfo=UTC)
        calls: list = []
        res = await selllayer.run_type_c_step(
            now=night1, transport=mock_transport('{"decision": "HOLD"}', calls),
            fund=fund, close=close, vol=vol)
        assert len(res["scans"]) == 1 and res["scans"][0]["decision"] == "HOLD"
        rows = await _ledger_rows()
        assert len(rows) == 1 and rows[0].decision == "HOLD" and rows[0].filled_at is None
        # second night: same filing -> NO second consult
        res2 = await selllayer.run_type_c_step(
            now=datetime(2026, 2, 3, 23, 30, tzinfo=UTC),
            transport=mock_transport('{"decision": "EXIT"}', calls),
            fund=fund, close=close, vol=vol)
        assert res2["scans"] == [] and len(calls) == 1
        rows = await _ledger_rows()
        assert len(rows) == 1 and rows[0].decision == "HOLD"   # unchanged
        from app.db import Session
        async with Session() as s:
            assert await s.scalar(select(func.count()).select_from(MonthlyPosition)) == 1

    async def test_healthy_fundamentals_no_consult(self, mem_db, llm_backend):
        await self._seed(mem_db)
        dates, close, vol = _c_tape()
        _, _, _, fund = _monthly_tape(n_days=20, tickers=("AAA",))
        calls: list = []
        res = await selllayer.run_type_c_step(
            now=datetime(2026, 2, 2, 23, 30, tzinfo=UTC),
            transport=mock_transport('{"decision": "EXIT"}', calls),
            fund=fund, close=close, vol=vol)
        assert res["scans"] == [] and calls == []
        assert await _ledger_rows() == []

    async def test_knob_off_skips(self, mem_db, monkeypatch):
        monkeypatch.setattr(settings, "sim_monthly_llm_sells", False)
        res = await selllayer.run_type_c_step(now=datetime(2026, 2, 2, 23, 30, tzinfo=UTC))
        assert res["skipped"] is True

    async def test_self_computed_frame_uses_universe_and_live_variant(
            self, mem_db, llm_backend, monkeypatch):
        """Defect-1 pin: with NO injected fund/close/vol/frame, the scan's loads
        run over the LIVE UNIVERSE (not the held names) and the mom variant is
        synced from ``monthly._strategy_variant()`` exactly like the rebuild —
        no "raw" pin anywhere."""
        await self._seed(mem_db)
        # monkeypatch restores the default at teardown even though the sell
        # layer assigns settings.sim_daily_core_mom_variant directly.
        monkeypatch.setattr(settings, "sim_daily_core_mom_variant", "raw")

        universe = ["AAA", "BBB", "CCC", "DDD"]
        seen: dict = {}

        monkeypatch.setattr(monthly, "_strategy_universe", lambda: "test-uni")
        monkeypatch.setattr(monthly, "_strategy_variant", lambda: "residual")

        def fake_universe_tickers(name):
            seen["universe_arg"] = name
            return list(universe)
        monkeypatch.setattr(monthly, "universe_tickers", fake_universe_tickers)

        import app.fundamentals as fundamentals_mod

        async def fake_load_fundamentals(tickers):
            seen["fund_members"] = sorted(tickers)
            return _broken_fund() | {t: {"x": []} for t in universe if t != "AAA"}
        monkeypatch.setattr(fundamentals_mod, "load_fundamentals", fake_load_fundamentals)

        dates = pd.bdate_range("2026-01-02", periods=25)  # includes month-end 01-30
        close = pd.DataFrame({t: 100.0 for t in universe}, index=dates)
        vol = pd.DataFrame(1e6, index=dates, columns=universe)

        async def fake_load_frames(tickers, asof, start=None):
            seen["frame_members"] = sorted(tickers)
            return close, vol
        monkeypatch.setattr(monthly, "load_frames", fake_load_frames)

        def fake_eligible_frame(me, c, v, f):
            seen["variant_at_frame"] = settings.sim_daily_core_mom_variant
            return pd.DataFrame({
                "roe": [0.1] * 4, "p_fcf": [20.0] * 4, "mcap": [1e10] * 4,
                "dollar_vol": [5e7] * 4, "mom": [0.4, 0.3, 0.2, 0.1],
                "price": [100.0] * 4, "vol": [0.2] * 4, "eligible": [True] * 4,
            }, index=pd.Index(universe, name="ticker"))
        monkeypatch.setattr(monthly, "eligible_frame", fake_eligible_frame)

        res = await selllayer.run_type_c_step(
            now=datetime(2026, 2, 2, 23, 30, tzinfo=UTC),
            transport=mock_transport('{"decision": "EXIT"}'))
        assert len(res["scans"]) == 1 and res["scans"][0]["decision"] == "EXIT"
        # (i) the loads used the UNIVERSE members, not just the held names
        assert seen["universe_arg"] == "test-uni"
        assert seen["fund_members"] == sorted(universe)          # not ["AAA"]
        assert seen["frame_members"] == sorted(universe)         # not ["AAA"]
        # (ii) the variant was synced from the live selection before frame math
        assert seen["variant_at_frame"] == "residual"
        assert settings.sim_daily_core_mom_variant == "residual"
        # the context really carries universe-frame cross-sectional fields
        rows = await _ledger_rows()
        assert len(rows) == 1
        ctx = json.loads(rows[0].context)
        assert ctx["mom_rank_pct_in_frame"] is not None
        assert ctx["eligible_in_frame"] is True


# ---------------------------------------------------------------------------
# 5. Bit-identical OFF parity at the engine level
# ---------------------------------------------------------------------------

class TestOffParity:
    async def test_off_knobs_make_all_overlays_identity(self, mem_db, monkeypatch):
        monkeypatch.setattr(settings, "sim_llm_sell_layer", False)
        monkeypatch.setattr(settings, "sim_monthly_llm_sells", False)
        proposals = [_stop_proposal()]
        assert await selllayer.overlay_type_a(proposals, _decision()) is proposals
        dates, close, vol, fund = _monthly_tape()
        picks = ["AAA", "CCC"]
        executed, detail = await selllayer.overlay_type_b(
            held_before=["AAA", "BBB"], picks=picks, fill_map={"BBB": 99.0},
            frame=None, close=close, fund=fund, members=list(close.columns),
            me_ts=pd.Timestamp(dates[-1]), f_date=dates[-1],
            target_n=3, hold_band=4)
        assert executed == picks and detail is None
        res = await selllayer.run_type_c_step(now=datetime(2026, 2, 2, tzinfo=UTC))
        assert res["skipped"] is True
        assert await _ledger_rows() == []


# ---------------------------------------------------------------------------
# 6. Contamination proof — backfills make ZERO consult attempts
# ---------------------------------------------------------------------------

class TestContamination:
    async def test_sim_backfill_makes_zero_consult_attempts(self, mem_db, monkeypatch):
        from app import optimize as opt
        from app.analysis import signal_series
        from tests.test_optimize import _gen_candles

        up = signal_series(_gen_candles(100.0, 0.01, n=300, seed=1))
        dn = signal_series(_gen_candles(100.0, -0.01, n=300, seed=2))

        async def fake_load_series(_tickers, start=None):
            return {"UP": up, "DN": dn}
        monkeypatch.setattr(opt, "_load_series", fake_load_series)

        attempts: list = []

        async def exploding(*a, **k):
            attempts.append(1)
            raise AssertionError("sell layer consulted during a backfill")
        monkeypatch.setattr(selllayer, "consult_live", exploding)
        monkeypatch.setattr(selllayer, "_llm_transport", exploding)
        monkeypatch.setattr(selllayer, "overlay_type_a", exploding)
        monkeypatch.setattr(selllayer, "overlay_type_b", exploding)
        monkeypatch.setattr(selllayer, "run_type_c_step", exploding)

        r = await sim.backfill(start="2025-03-01")
        assert r["ok"] is True
        assert attempts == []                    # ZERO consult attempts

    async def test_monthly_backfill_makes_zero_consult_attempts(self, mem_db, monkeypatch):
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
        monkeypatch.setattr(monthly, "_current_month", lambda: "2026-08")

        attempts: list = []

        async def exploding(*a, **k):
            attempts.append(1)
            raise AssertionError("sell layer consulted during a backfill")
        monkeypatch.setattr(selllayer, "consult_live", exploding)
        monkeypatch.setattr(selllayer, "_llm_transport", exploding)
        monkeypatch.setattr(selllayer, "overlay_type_a", exploding)
        monkeypatch.setattr(selllayer, "overlay_type_b", exploding)
        monkeypatch.setattr(selllayer, "run_type_c_step", exploding)

        r = await monthly.backfill(start="2026-07-01")
        assert r["ok"] is True
        assert attempts == []                    # ZERO consult attempts


# ---------------------------------------------------------------------------
# 7. The ledger + the idempotent additive ALTERs
# ---------------------------------------------------------------------------

class TestLedgerAndSchema:
    async def test_record_has_full_field_set_and_is_append_only(self, mem_db, llm_backend):
        rec = await selllayer.consult_live(
            "A:X:2026-03-06", "A", {"ticker": "X", "k": 1},
            transport=mock_transport('{"decision": "CONFIRM"}'))
        row = await selllayer._record_decision(
            rec=rec, engine="sim", session_date=S_STR, ticker="X",
            context={"ticker": "X", "k": 1},
            engine_branch={"action": "execute standing stop"},
            decided_at=datetime(2026, 3, 9, 23, 30, tzinfo=UTC))
        assert row.decided_at == datetime(2026, 3, 9, 23, 30, tzinfo=UTC)  # frozen
        assert row.session_date == S_STR
        assert row.event_id == "A:X:2026-03-06"
        assert row.backend == "test" and row.model == "m1"
        assert row.latency_ms is not None
        assert json.loads(row.context) == {"ticker": "X", "k": 1}
        assert json.loads(row.engine_branch)["action"] == "execute standing stop"
        assert (row.filled_at, row.shares, row.price, row.cash_after) == (None, None, None, None)
        rows = await _ledger_rows()
        assert len(rows) == 1 and rows[0].id == row.id   # appended, never updated

    async def test_record_payloads_deterministic_modulo_wall_clock(self, mem_db, llm_backend):
        ctx = {"ticker": "X", "nested": {"b": 2, "a": 1}}
        branch = {"action": "release at rebuild", "fill_price": 99.0}
        rec = await selllayer.consult_live("B:X:2026-03-09", "B", ctx,
                                           transport=mock_transport('{"decision": "CONFIRM"}'))
        fixed = datetime(2026, 3, 9, 23, 30, tzinfo=UTC)
        r1 = await selllayer._record_decision(rec=rec, engine="monthly", session_date=F_STR,
                                              ticker="X", context=ctx, engine_branch=branch,
                                              decided_at=fixed)
        r2 = await selllayer._record_decision(rec=rec, engine="monthly", session_date=F_STR,
                                              ticker="X", context=ctx, engine_branch=branch,
                                              decided_at=fixed)
        assert r1.context == r2.context          # sorted-keys JSON, identical
        assert r1.engine_branch == r2.engine_branch
        assert (r1.decision, r1.prompt_version, r1.parsed, r1.fail_safe, r1.backend) == \
               (r2.decision, r2.prompt_version, r2.parsed, r2.fail_safe, r2.backend)

    async def test_additive_alters_are_idempotent(self, tmp_path):
        from app import db as db_mod
        engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path}/t.db")
        await db_mod.init_db(engine)
        await db_mod.init_db(engine)             # twice — must not raise
        async with engine.begin() as conn:
            def check(sync_conn):
                insp = inspect(sync_conn)
                cols = {c["name"] for c in insp.get_columns("sim_positions")}
                reb_cols = {c["name"] for c in insp.get_columns("monthly_rebalances")}
                tables = set(insp.get_table_names())
                assert {"stop_defer_level", "stop_defer_set_at"} <= cols
                assert "sell_layer" in reb_cols
                assert "llm_sell_decisions" in tables
            await conn.run_sync(check)
        await engine.dispose()
