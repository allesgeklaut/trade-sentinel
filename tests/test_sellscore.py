"""Tests for the LLM sell scorecard (app/sellscore.py) — slice 2: the
with/without-LLM history on COMPLETED predeclared windows.

All tests are OFFLINE and in-memory (SQLite): the ledger rows, the canned
candles, the SimTrade exit fills and the MonthlyRebalance audit rows are
seeded directly; the scorer is READ-ONLY and makes no LLM/network calls.

Covered (brief §TESTS):

  1. a hand-computed A-DEFER delta (a canned re-stop within the window: the
     deferred exit vs the engine's sold-at-level + cash yield);
  2. a hand-computed B-DEFER delta (the retained name vs the displaced add
     over the canned window) + the declared fallback (equal-weight mean of
     the new adds) when the displaced name is unidentifiable;
  3. a hand-computed C-EXIT delta (the early exit + cash yield vs held-to-
     rebuild) + the no-price-at-rebuild PENDING case;
  4. the completed-window rule: a mid-window event is PENDING and becomes
     scored once the canned data advances past its window;
  5. fail-safe accounting: fail-safe rows count, score by their implied
     branch (CONFIRM/HOLD = the engine's path, delta 0 / no divergence);
  6. the bars appear; determinism (byte-identical modulo the wall-clock);
     read-only (row counts unchanged after a scoring run);
  7. the API endpoint contract (the response shape, the empty-ledger state);
  8. a small UI smoke: `node --check` on index.html's extracted inline
     scripts (the repo's node-subprocess pattern) — the section itself is
     covered by the endpoint contract + manual render.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest
from sqlalchemy import StaticPool, func, select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app import main, monthly, sellscore, sim
from app.db import (
    Base,
    Candle,
    LlmSellDecision,
    MonthlyAccount,
    MonthlyRebalance,
    SimAccount,
    SimTrade,
)
from app.sellscore import TIME_CAP_SESSIONS

NOW = datetime(2026, 10, 31, 12, 0, 0, tzinfo=UTC)
RATE = 0.001          # CostConfig stock one-way 10 bps
YIELD = 0.03          # CostConfig cash_yield_annual
DAYS_PER_YEAR = 252.0


def _cy(w: int) -> float:
    """The COMPOUNDED per-session cash-yield growth over w sessions — the
    reval convention ((1+cy)**(1/252) per session) the bars were measured
    under; the scorer must use the same one."""
    return (1.0 + YIELD) ** (w / DAYS_PER_YEAR)

TAPE_START = "2026-01-02"
TAPE_END = "2026-07-31"


# ---------------------------------------------------------------------------
# Fixtures / canned tape
# ---------------------------------------------------------------------------

@pytest.fixture
async def mem_db(monkeypatch):
    """In-memory SQLite for every engine the scorecard reads. sellscore
    resolves ``db.Session`` and ``monthly.Session`` at call time, so patching
    ``app.db.Session`` / ``app.monthly.Session`` covers it too."""
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    import app.db as db_mod
    monkeypatch.setattr(sim, "Session", session_factory)
    monkeypatch.setattr(db_mod, "Session", session_factory)
    monkeypatch.setattr(monthly, "Session", session_factory)

    async with session_factory() as s:
        s.add(SimAccount(id=1, cash=100_000.0))
        s.add(MonthlyAccount(id=1, cash=100_000.0))
        await s.commit()
    yield session_factory
    await engine.dispose()


def _sessions(end: str = TAPE_END) -> list[str]:
    return [d.strftime("%Y-%m-%d") for d in pd.bdate_range(TAPE_START, end)]


# Distinct, exactly computable open paths per ticker (close = open + drift).
_PATHS = {
    "AAA": (100.0, 0.10, 0.5),
    "BBB": (50.0, 0.05, 0.25),
    "CCC": (30.0, -0.02, 0.1),
    "DDD": (20.0, 0.03, 0.1),
    "EEE": (80.0, -0.04, -0.2),
}


def _open_of(ticker: str, day: str, sessions: list[str] | None = None) -> float:
    sessions = sessions if sessions is not None else _sessions()
    base, slope, _drift = _PATHS[ticker]
    return base + slope * sessions.index(day)


def _close_of(ticker: str, day: str, sessions: list[str] | None = None) -> float:
    sessions = sessions if sessions is not None else _sessions()
    _base, _slope, drift = _PATHS[ticker]
    return _open_of(ticker, day, sessions) + drift


async def _seed_candles(tickers, start: str = TAPE_START, end: str = TAPE_END) -> list[str]:
    """Insert the canned tape (one bar per session per ticker) over
    [start, end]; returns the FULL session calendar [TAPE_START, end]."""
    sessions = [d.strftime("%Y-%m-%d") for d in pd.bdate_range(start, end)]
    all_sessions = _sessions(end)
    from app.db import Session
    async with Session() as s:
        for tk in tickers:
            base, slope, drift = _PATHS[tk]
            for d in sessions:
                i = all_sessions.index(d)
                op = base + slope * i
                s.add(Candle(ticker=tk, timestamp=datetime.strptime(d, "%Y-%m-%d"),
                             open=op, high=op + 1.0, low=op - 1.0, close=op + drift,
                             volume=1e6))
        await s.commit()
    return all_sessions


async def _add_ledger(*, event_type, engine, session_date, ticker, decision,
                      event_id=None, parsed=True, fail_safe=False, fail_reason=None,
                      context=None, engine_branch=None, filled_at=None, price=None,
                      decided_at=None, rationale="r") -> None:
    from app.db import Session
    event_id = event_id or f"{event_type}:{ticker}:{session_date}"
    async with Session() as s:
        s.add(LlmSellDecision(
            decided_at=decided_at or datetime(2026, 10, 30, 23, 0, tzinfo=UTC),
            session_date=session_date, event_type=event_type, engine=engine,
            event_id=event_id, ticker=ticker, decision=decision, rationale=rationale,
            prompt_version="sellshadow/1", parsed=parsed, fail_safe=fail_safe,
            fail_reason=fail_reason, context=json.dumps(context or {}),
            engine_branch=json.dumps(engine_branch or {}),
            filled_at=filled_at, price=price,
            backend="test", model="m1", latency_ms=1.0))
        await s.commit()


async def _add_sim_sell(ticker: str, day: str, price: float) -> None:
    from app.db import Session
    async with Session() as s:
        s.add(SimTrade(ticker=ticker, side="SELL", shares=10.0, price=price,
                       cash_after=0.0, reason="re-armed stop",
                       created_at=datetime.strptime(day, "%Y-%m-%d")))
        await s.commit()


async def _add_rebalance(month: str, day: str, held_before: str, picked: str,
                         sell_layer: dict | None) -> None:
    from app.db import Session
    async with Session() as s:
        s.add(MonthlyRebalance(rebal_month=month,
                               rebal_date=datetime.strptime(day, "%Y-%m-%d"),
                               held_before=held_before, picked=picked, n_new=2,
                               snapshot="{}",
                               sell_layer=json.dumps(sell_layer) if sell_layer else None))
        await s.commit()


async def _table_counts() -> dict[str, int]:
    from app.db import Session
    tables = [Candle, LlmSellDecision, MonthlyRebalance, SimTrade, SimAccount,
              MonthlyAccount]
    async with Session() as s:
        return {t.__tablename__: await s.scalar(select(func.count()).select_from(t))
                for t in tables}


async def _score():
    return await sellscore.score_sell_decisions(now=NOW)


# ---------------------------------------------------------------------------
# 1. Type A — hand-computed A-DEFER delta (re-stop within the window)
# ---------------------------------------------------------------------------

class TestTypeA:
    async def test_defer_delta_hand_computed(self, mem_db):
        sessions = await _seed_candles(["AAA"])
        s_date = "2026-03-06"
        sell_day, sell_px = "2026-03-20", 97.5
        fill_px = 102.0
        await _add_sim_sell("AAA", sell_day, sell_px)
        await _add_ledger(event_type="A", engine="sim", session_date=s_date,
                          ticker="AAA", decision="DEFER",
                          context={"fill_price": fill_px, "proposal_price": fill_px},
                          engine_branch={"action": "defer", "level": 95.0,
                                         "rearmed_level": 90.25})
        # a CONFIRM row on the same window: no divergence, delta 0
        await _add_ledger(event_type="A", engine="sim", session_date=s_date,
                          ticker="AAA", decision="CONFIRM",
                          event_id="A:AAA:confirm",
                          context={"fill_price": fill_px},
                          engine_branch={"action": "execute standing stop",
                                         "level": 95.0, "fill_price": fill_px})
        rep = await _score()
        tA = rep["types"]["A"]
        assert tA["n_rows"] == 2 and tA["n_decided"] == 2 and tA["n_pending"] == 0
        assert tA["n_divergent"] == 1            # only the DEFER diverges
        # independent arithmetic on the canned tape (no sellscore helpers);
        # cash yield compounded per session: (1+cy)**(w/252)
        w = sessions.index(sell_day) - sessions.index(s_date)
        engine_ret = (1.0 - RATE) * _cy(w) - 1.0
        llm_ret = sell_px / fill_px - 1.0 - RATE
        exp = llm_ret - engine_ret
        ev = next(e for e in tA["events"] if e["decision"] == "DEFER")
        assert ev["status"] == "scored" and ev["delta"] == pytest.approx(exp)
        assert ev["window"]["end"] == sell_day and ev["window"]["sessions"] == w
        assert tA["aggregate"]["mean"] == pytest.approx(exp)   # divergent only
        assert tA["aggregate"]["n"] == 1
        conf = next(e for e in tA["events"] if e["decision"] == "CONFIRM")
        assert conf["delta"] == 0.0


# ---------------------------------------------------------------------------
# 2. Type B — hand-computed B-DEFER delta + the declared fallback
# ---------------------------------------------------------------------------

class TestTypeB:
    # Live-data shape (the gate-review bug's regression pin): a rebuild for
    # month-end 2026-03-31 is AUDITED as rebal_month "2026-03" with
    # rebal_date = 2026-03-31 (me_ts), but its fills happen on the NEXT
    # session 2026-04-01, and that is what selllayer writes as the B row's
    # session_date (as_of = f_date). So the B row's session_date is in the
    # calendar month AFTER rebal_month — a month-string join would never match.
    ME1, FILL1 = "2026-03-31", "2026-04-01"   # month-end / its fill session
    ME2, FILL2 = "2026-04-30", "2026-05-01"

    async def _seed_book(self):
        sessions = await _seed_candles(["AAA", "BBB", "CCC", "DDD"])
        await _add_rebalance("2026-03", self.ME1, "AAA,BBB", "AAA,CCC,DDD",
                             {"engine_picks": ["AAA", "CCC", "DDD"],
                              "executed_picks": ["AAA", "CCC", "BBB"],
                              "retained": ["BBB"]})
        await _add_rebalance("2026-04", self.ME2, "AAA,CCC", "AAA,CCC,DDD",
                             {"engine_picks": ["AAA", "CCC", "DDD"],
                              "executed_picks": ["AAA", "CCC", "DDD"]})
        return sessions

    async def test_defer_vs_displaced_add_hand_computed(self, mem_db):
        sessions = await self._seed_book()
        await _add_ledger(event_type="B", engine="monthly", session_date=self.FILL1,
                          ticker="BBB", decision="DEFER")
        rep = await _score()
        tB = rep["types"]["B"]
        assert tB["n_decided"] == 1 and tB["n_divergent"] == 1
        ev = tB["events"][0]
        # window = [2026-04-01 rebuild fill → 2026-05-01 next rebuild fill]
        assert ev["window"]["start"] == self.FILL1 and ev["window"]["end"] == "2026-05-01"
        e_r, x_r = _open_of("BBB", self.FILL1, sessions), _open_of("BBB", "2026-05-01", sessions)
        # the displaced add = engine_picks − executed_picks = the tail add DDD
        e_d, x_d = _open_of("DDD", self.FILL1, sessions), _open_of("DDD", "2026-05-01", sessions)
        # both legs of both branches net of the one-way cost -> costs cancel
        exp = (x_r / e_r - 1.0 - 2 * RATE) - (x_d / e_d - 1.0 - 2 * RATE)
        assert ev["delta"] == pytest.approx(exp)
        assert ev["detail"]["engine_branch"] == "displaced add DDD"

    async def test_regression_exact_displaced_add_not_the_mean_fallback(self, mem_db):
        """The gate-review bug: with >=2 new additions at the rebuild, the exact
        displaced name (DDD) differs from the equal-weight mean of the new adds
        (CCC, DDD) — a dead join would silently score the mean instead."""
        sessions = await self._seed_book()
        await _add_ledger(event_type="B", engine="monthly", session_date=self.FILL1,
                          ticker="BBB", decision="DEFER")
        rep = await _score()
        ev = rep["types"]["B"]["events"][0]
        e_r, x_r = _open_of("BBB", self.FILL1, sessions), _open_of("BBB", "2026-05-01", sessions)
        llm_net = x_r / e_r - 1.0 - 2 * RATE
        e_d, x_d = _open_of("DDD", self.FILL1, sessions), _open_of("DDD", "2026-05-01", sessions)
        e_c, x_c = _open_of("CCC", self.FILL1, sessions), _open_of("CCC", "2026-05-01", sessions)
        exact_delta = llm_net - (x_d / e_d - 1.0 - 2 * RATE)
        mean_delta = llm_net - ((x_c / e_c - 1.0 - 2 * RATE) + (x_d / e_d - 1.0 - 2 * RATE)) / 2.0
        assert exact_delta != pytest.approx(mean_delta)      # the two paths differ here
        assert ev["delta"] == pytest.approx(exact_delta)     # the EXACT path was used
        assert ev["detail"]["engine_branch"] == "displaced add DDD"
        assert "fallback" not in ev["detail"]["engine_branch"]

    async def test_fallback_equal_weight_new_adds(self, mem_db):
        sessions = await self._seed_book()
        # sell_layer carries no displaced name (engine == executed picks) -> the
        # declared fallback: equal-weight mean of the names NEWLY ADDED at the
        # same rebuild (picks − held_before = DDD only)
        await _add_ledger(event_type="B", engine="monthly", session_date=self.FILL2,
                          ticker="BBB", decision="DEFER")
        rep = await _score()
        ev = rep["types"]["B"]["events"][0]
        assert ev["window"]["start"] == self.FILL2
        assert ev["window"]["end"] == "2026-06-01"   # first session after 2026-05-29
        e_r, x_r = _open_of("BBB", self.FILL2, sessions), _open_of("BBB", "2026-06-01", sessions)
        leg = (_open_of("DDD", "2026-06-01", sessions) / _open_of("DDD", self.FILL2, sessions)
               - 1.0 - 2 * RATE)
        exp = (x_r / e_r - 1.0 - 2 * RATE) - leg
        assert ev["delta"] == pytest.approx(exp)
        assert "fallback" in ev["detail"]["engine_branch"]


# ---------------------------------------------------------------------------
# 3. Type C — hand-computed C-EXIT delta + the no-price-at-rebuild PENDING
# ---------------------------------------------------------------------------

class TestTypeC:
    async def test_exit_delta_hand_computed(self, mem_db):
        sessions = await _seed_candles(["EEE"])
        fill_day = "2026-03-16"
        fill_px = _open_of("EEE", fill_day, sessions)
        await _add_ledger(event_type="C", engine="monthly", session_date="2026-03-16",
                          ticker="EEE", decision="EXIT",
                          filled_at=datetime.strptime(fill_day, "%Y-%m-%d"), price=fill_px)
        rep = await _score()
        tC = rep["types"]["C"]
        assert tC["n_decided"] == 1 and tC["n_divergent"] == 1
        ev = tC["events"][0]
        # window = [the EXIT fill 2026-03-16 → the 2026-04-01 rebuild fill]
        assert ev["window"]["start"] == fill_day and ev["window"]["end"] == "2026-04-01"
        w = sessions.index("2026-04-01") - sessions.index(fill_day)
        llm_ret = (1.0 - RATE) * _cy(w) - 1.0
        engine_ret = _open_of("EEE", "2026-04-01", sessions) / fill_px - 1.0 - RATE
        assert ev["delta"] == pytest.approx(llm_ret - engine_ret)

    async def test_no_price_at_next_rebuild_stays_pending(self, mem_db):
        # the calendar runs to 2026-07-31 but EEE has no bars in July: the
        # would-be exit at the 2026-07-01 rebuild has no executable price
        await _seed_candles(["AAA"])
        await _seed_candles(["EEE"], end="2026-06-30")
        fill_day = "2026-06-15"
        fill_px = _open_of("EEE", fill_day)
        await _add_ledger(event_type="C", engine="monthly", session_date="2026-06-12",
                          ticker="EEE", decision="EXIT",
                          filled_at=datetime.strptime(fill_day, "%Y-%m-%d"), price=fill_px)
        rep = await _score()
        tC = rep["types"]["C"]
        assert tC["n_decided"] == 0 and tC["n_pending"] == 1 and tC["n_divergent"] == 0
        d = rep["decisions"][0]
        assert d["status"] == "pending" and d["delta"] is None
        assert d["window"]["end"] == "2026-07-01" and "no executable price" in d["skip_reason"]


class TestTypeD:
    """The 2026-11-04 promotion: D scores through the SAME completed-window
    contract as C — [the EXIT fill → the next month-end rebuild's fill]; the
    WAIT is the counterfactual. HOLD = delta 0."""

    async def test_exit_delta_hand_computed_same_contract_as_c(self, mem_db):
        sessions = await _seed_candles(["EEE"])
        fill_day = "2026-03-16"
        fill_px = _open_of("EEE", fill_day, sessions)
        await _add_ledger(event_type="D", engine="monthly", session_date="2026-03-15",
                          ticker="EEE", decision="EXIT", event_id="D:EEE:2026-03-15:m",
                          filled_at=datetime.strptime(fill_day, "%Y-%m-%d"), price=fill_px)
        rep = await _score()
        tD = rep["types"]["D"]
        assert tD["n_decided"] == 1 and tD["n_divergent"] == 1
        ev = tD["events"][0]
        assert ev["window"]["start"] == fill_day and ev["window"]["end"] == "2026-04-01"
        w = sessions.index("2026-04-01") - sessions.index(fill_day)
        llm_ret = (1.0 - RATE) * _cy(w) - 1.0
        engine_ret = _open_of("EEE", "2026-04-01", sessions) / fill_px - 1.0 - RATE
        assert ev["delta"] == pytest.approx(llm_ret - engine_ret)

    async def test_hold_scores_zero(self, mem_db):
        await _seed_candles(["EEE"])
        await _add_ledger(event_type="D", engine="monthly", session_date="2026-03-13",
                          ticker="EEE", decision="HOLD", event_id="D:EEE:2026-03-13")
        rep = await _score()
        tD = rep["types"]["D"]
        assert tD["n_decided"] == 1 and tD["n_divergent"] == 0
        assert tD["events"][0]["delta"] == 0.0
        assert rep["types"]["C"]["n_rows"] == 0   # never bucketed into C

    async def test_unfilled_exit_stays_pending(self, mem_db):
        await _seed_candles(["EEE"])
        await _add_ledger(event_type="D", engine="monthly", session_date="2026-03-13",
                          ticker="EEE", decision="EXIT", event_id="D:EEE:2026-03-13:m")
        rep = await _score()
        tD = rep["types"]["D"]
        assert tD["n_decided"] == 0 and tD["n_pending"] == 1
        d = rep["decisions"][0]
        assert d["status"] == "pending" and "has not filled yet" in d["skip_reason"]

    async def test_no_offline_reference_bar_is_declared(self, mem_db):
        await _seed_candles(["EEE"])
        await _add_ledger(event_type="D", engine="monthly", session_date="2026-03-13",
                          ticker="EEE", decision="HOLD", event_id="D:EEE:2026-03-13")
        rep = await _score()
        assert "un-backtestable" in rep["reference_bars"]["D"]["source"]


@pytest.mark.parametrize("event_type", ["C", "D"])
async def test_month_end_exit_scores_at_immediately_following_rebuild(mem_db, event_type):
    sessions = await _seed_candles(["EEE"], end="2026-03-31")
    fill_day = "2026-03-31"
    await _add_ledger(event_type=event_type, engine="monthly", session_date="2026-03-30",
                      ticker="EEE", decision="EXIT",
                      filled_at=datetime(2026, 3, 31),
                      price=_open_of("EEE", fill_day, sessions))
    before = await _score()
    assert before["types"][event_type]["n_pending"] == 1
    await _seed_candles(["EEE"], start="2026-04-01", end="2026-04-01")
    report = await _score()
    event = report["types"][event_type]["events"][0]
    assert event["window"] == {"start": fill_day, "end": "2026-04-01", "sessions": 1}


@pytest.mark.parametrize(("anchor", "expected"), [
    ("2026-01-29", "2026-02-02"), ("2026-01-30", "2026-02-02"),
    ("2026-02-02", "2026-03-02"), ("2026-02-27", "2026-03-02"),
])
def test_next_rebuild_respects_month_end_open_boundary(anchor, expected):
    days = pd.bdate_range("2026-01-01", "2026-03-03").strftime("%Y-%m-%d").tolist()
    assert sellscore._next_rebuild_fill(days, anchor) == expected


# ---------------------------------------------------------------------------
# 4. The completed-window rule: mid-window PENDING -> scored once data advances
# ---------------------------------------------------------------------------

class TestCompletedWindowRule:
    async def test_a_defer_pending_then_scored_at_cap(self, mem_db):
        sessions_short = await _seed_candles(["AAA"], end="2026-04-30")
        s_date = "2026-03-06"
        fill_px = 102.0
        await _add_ledger(event_type="A", engine="sim", session_date=s_date,
                          ticker="AAA", decision="DEFER",
                          context={"fill_price": fill_px},
                          engine_branch={"action": "defer", "level": 95.0})
        early = await _score()
        assert early["types"]["A"]["n_decided"] == 0
        assert early["types"]["A"]["n_pending"] == 1
        assert early["decisions"][0]["status"] == "pending"
        assert early["data_as_of"] == "2026-04-30"

        # advance the canned data past the 63-session cap: no exit fill, so
        # the window ends at the cap session, marked at its close
        sessions = await _seed_candles(["AAA"], start="2026-05-01", end=TAPE_END)
        late = await _score()
        tA = late["types"]["A"]
        assert tA["n_decided"] == 1 and tA["n_pending"] == 0
        ev = tA["events"][0]
        d_idx = sessions.index(s_date)
        cap_day = sessions[d_idx + TIME_CAP_SESSIONS]
        assert ev["window"]["end"] == cap_day and ev["window"]["sessions"] == TIME_CAP_SESSIONS
        assert "time cap" in ev["window"]["exit_kind"]
        w = TIME_CAP_SESSIONS
        engine_ret = (1.0 - RATE) * _cy(w) - 1.0
        llm_ret = _close_of("AAA", cap_day, sessions) / fill_px - 1.0 - RATE
        assert ev["delta"] == pytest.approx(llm_ret - engine_ret)
        assert len(sessions_short) < len(sessions)  # the tape really advanced


# ---------------------------------------------------------------------------
# 5. Fail-safe accounting
# ---------------------------------------------------------------------------

class TestFailSafeAccounting:
    async def test_fail_safe_rows_count_and_score_as_engine_branch(self, mem_db):
        await _seed_candles(["AAA", "BBB", "CCC", "DDD", "EEE"])
        await _add_rebalance("2026-03", "2026-03-31", "AAA,BBB", "AAA,CCC,DDD",
                             {"engine_picks": ["AAA", "CCC", "DDD"],
                              "executed_picks": ["AAA", "CCC", "DDD"]})
        # A fail-safe (no backend): implied branch CONFIRM -> delta 0
        await _add_ledger(event_type="A", engine="sim", session_date="2026-03-06",
                          ticker="AAA", decision="CONFIRM", parsed=False,
                          fail_safe=True, fail_reason="no backend",
                          context={"fill_price": 102.0},
                          engine_branch={"action": "execute standing stop",
                                         "level": 95.0, "fill_price": 102.0})
        # B fail-safe: CONFIRM -> delta 0 (session_date = the rebuild's FILL
        # session 2026-04-01; the audit row is rebal_month "2026-03",
        # rebal_date 2026-03-31)
        await _add_ledger(event_type="B", engine="monthly", session_date="2026-04-01",
                          ticker="BBB", decision="CONFIRM", parsed=False,
                          fail_safe=True, fail_reason="timeout")
        # C fail-safe: HOLD -> delta 0
        await _add_ledger(event_type="C", engine="monthly", session_date="2026-03-16",
                          ticker="EEE", decision="HOLD", parsed=False,
                          fail_safe=True, fail_reason="unparsable response")
        rep = await _score()
        for t, n in (("A", 1), ("B", 1), ("C", 1)):
            blk = rep["types"][t]
            assert blk["n_fail_safe"] == n and blk["fail_safe_rate"] == pytest.approx(1.0)
            assert blk["n_decided"] == 1 and blk["n_divergent"] == 0
            assert blk["events"][0]["delta"] == 0.0
            assert blk["aggregate"]["n"] == 0      # no divergent rows -> empty agg
        assert rep["overall"]["n_fail_safe"] == 3
        assert rep["overall"]["fail_safe_rate"] == pytest.approx(1.0)
        assert rep["overall"]["n_divergent"] == 0
        assert all(d["fail_safe"] is True for d in rep["decisions"])


# ---------------------------------------------------------------------------
# 6. Bars, determinism, read-only
# ---------------------------------------------------------------------------

class TestBarsDeterminismReadOnly:
    async def test_bars_determinism_and_read_only(self, mem_db):
        await _seed_candles(["AAA", "BBB", "CCC", "DDD", "EEE"])
        await _add_rebalance("2026-03", "2026-03-31", "AAA,BBB", "AAA,CCC,DDD",
                             {"engine_picks": ["AAA", "CCC", "DDD"],
                              "executed_picks": ["AAA", "CCC", "BBB"],
                              "retained": ["BBB"]})
        await _add_sim_sell("AAA", "2026-03-20", 97.5)
        await _add_ledger(event_type="A", engine="sim", session_date="2026-03-06",
                          ticker="AAA", decision="DEFER",
                          context={"fill_price": 102.0},
                          engine_branch={"action": "defer", "level": 95.0})
        await _add_ledger(event_type="B", engine="monthly", session_date="2026-04-01",
                          ticker="BBB", decision="DEFER")
        await _add_ledger(event_type="C", engine="monthly", session_date="2026-03-16",
                          ticker="EEE", decision="HOLD")
        before = await _table_counts()
        rep1 = await _score()
        rep2 = await _score()
        assert await _table_counts() == before          # read-only: counts unchanged
        assert sellscore._dumps(rep1) == sellscore._dumps(rep2)   # deterministic
        assert rep1["schema"] == "sellscore/1" and rep1["read_only"] is True
        assert rep1["generated_at"] == NOW.isoformat()  # the wall-clock field
        bars = rep1["reference_bars"]
        assert bars["A"]["blanket_defer_pp_per_event"] == pytest.approx(0.0293)
        assert bars["B"]["blanket_retention_mean_pp_per_event"] == pytest.approx(-0.0160)
        assert bars["B"]["n_events_full_window"] == 165
        assert bars["B"]["by_year_mean_pp"]["2023"] == pytest.approx(-0.1330)
        assert bars["C"]["m5_vs_m0_pp_per_year"] == pytest.approx(0.0005)
        assert "COMPLETED" in rep1["completed_window_rule"]
        assert "p-values" in rep1["small_n_caveat"]
        assert rep1["types"]["A"]["reference_bar"] == bars["A"]
        assert rep1["overall"]["n_rows"] == 3 and rep1["overall"]["n_decided"] == 3
        assert rep1["overall"]["n_divergent"] == 2      # A-DEFER + B-DEFER only
        assert rep1["types"]["C"]["aggregate"]["n"] == 0
        by_year = rep1["types"]["A"]["by_year"]
        assert set(by_year) == {"2026"} and by_year["2026"]["n"] == 1


# ---------------------------------------------------------------------------
# 7. The API endpoint contract
# ---------------------------------------------------------------------------

class TestApiContract:
    async def test_empty_ledger_clean_state(self, mem_db):
        await _seed_candles(["AAA"])
        rep = await main.sellscore()
        assert rep["schema"] == "sellscore/1"
        assert rep["decisions"] == []
        assert rep["data_as_of"] == TAPE_END
        for t in ("A", "B", "C"):
            blk = rep["types"][t]
            assert blk["n_rows"] == 0 and blk["n_decided"] == 0
            assert blk["fail_safe_rate"] is None
            assert blk["aggregate"]["n"] == 0
            assert blk["reference_bar"] == rep["reference_bars"][t]
        assert rep["reference_bars"]["B"]["n_events_full_window"] == 165  # bars always

    async def test_response_shape_and_history(self, mem_db):
        await _seed_candles(["AAA"])
        await _add_sim_sell("AAA", "2026-03-20", 97.5)
        await _add_ledger(event_type="A", engine="sim", session_date="2026-03-06",
                          ticker="AAA", decision="DEFER", rationale="x" * 500,
                          context={"fill_price": 102.0},
                          engine_branch={"action": "defer", "level": 95.0})
        await _add_ledger(event_type="A", engine="sim", session_date="2026-03-09",
                          ticker="AAA", decision="CONFIRM", event_id="A:AAA:2",
                          decided_at=datetime(2026, 10, 31, 23, 0, tzinfo=UTC),
                          rationale="x" * 500,
                          context={"fill_price": 103.0},
                          engine_branch={"action": "execute standing stop",
                                         "level": 95.0, "fill_price": 103.0})
        rep = await main.sellscore()
        assert len(rep["decisions"]) == 2
        newest = rep["decisions"][0]                    # newest first
        assert newest["event_id"] == "A:AAA:2" and newest["status"] == "scored"
        for key in ("session_date", "account", "ticker", "event_type", "decision",
                    "parsed", "fail_safe", "fail_reason", "delta", "status", "rationale"):
            assert key in newest
        assert newest["account"] == "sim"
        assert len(newest["rationale"]) == 200          # truncated
        assert rep["types"]["A"]["n_decided"] == 2 and rep["types"]["A"]["n_divergent"] == 1


# ---------------------------------------------------------------------------
# 8. UI smoke: node --check on index.html's inline scripts (repo pattern)
# ---------------------------------------------------------------------------

def test_index_html_inline_scripts_parse():
    """The repo's UI-smoke pattern (subprocess node, as test_portfolio_ui.py
    does for portfolio.js): the scorecard's inline script must parse. The
    section's data contract is covered by TestApiContract above."""
    node = shutil.which("node")
    if node is None:
        pytest.skip("node not available")
    import re
    src = (Path(__file__).resolve().parents[1] / "static" / "index.html").read_text(
        encoding="utf-8")
    blocks = re.findall(r"<script>(.*?)</script>", src, re.S)
    assert blocks, "no inline scripts found"
    assert "loadSellScore" in src and 'id="sellscore-section"' in src
    for block in blocks:
        with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False,
                                         dir="/tmp/opencode") as f:
            f.write(block)
            path = f.name
        try:
            r = subprocess.run([node, "--check", path], capture_output=True, text=True)
            assert r.returncode == 0, r.stderr
        finally:
            Path(path).unlink()


def test_index_html_has_no_duplicate_top_level_functions():
    """A duplicate ``function name(...)`` declaration at the same nesting
    level silently overrides the earlier one — declarations hoist and the
    LAST one wins, with no error and no warning.

    That is exactly how the scorecard died: a later
    ``function fmtPct(v) { return v.toFixed(1) + '%'; }`` (introduced by the
    monthly-alignment commit, long before the scorecard existed) shadowed
    the null-safe ``fmtPct`` declared above it, so ``renderSellScore`` threw
    "Cannot read properties of null (reading 'toFixed')" on every null
    aggregate — an empty ledger, or any type with no completed windows — and
    rendered nothing at all.

    Guard the whole inline script against the class: no two declarations of
    the same name at the same indentation level.
    """
    import collections
    import re

    src = (Path(__file__).resolve().parents[1] / "static" / "index.html").read_text(
        encoding="utf-8")
    blocks = re.findall(r"<script>(.*?)</script>", src, re.S)
    assert blocks, "no inline scripts found"
    for bi, block in enumerate(blocks):
        seen = collections.Counter(
            (m.group(1), m.group(2))
            for m in re.finditer(
                r"^([ \t]*)function\s+([A-Za-z_$][\w$]*)\s*\(", block, re.M)
        )
        dup = {f"{name} (indent {len(indent)})": n
               for (indent, name), n in seen.items() if n > 1}
        assert not dup, (
            f"inline script block {bi} declares the same function more than once at "
            f"the same nesting level: {dup} — hoisting makes the LAST declaration win "
            f"silently, which is how the scorecard's null-safe fmtPct was shadowed")


# ---------------------------------------------------------------------------
# 2b. Type B attribution: capacity_truncated -> capacity_released flag
# ---------------------------------------------------------------------------


class TestTypeBCapacityReleased:
    """When the rebuild's defensive capacity cap releases a DEFER-retained
    name anyway, selllayer records it in the MonthlyRebalance sell_layer JSON
    as ``capacity_truncated``. The scorecard must flag that DEFER on its
    history entry (``capacity_released: true``) and count it top-level
    (``n_capacity_released``) — attribution honesty only; the delta math is
    unchanged (the counterfactual stands)."""

    ME1, FILL1 = "2026-03-31", "2026-04-01"
    ME2, FILL2 = "2026-04-30", "2026-05-01"

    async def _seed_book(self):
        sessions = await _seed_candles(["AAA", "BBB", "CCC", "DDD"])
        # target_n effectively 2: the cap truncates the retained BBB out of
        # executed_picks — exactly the selllayer capacity_truncated edge case.
        await _add_rebalance("2026-03", self.ME1, "AAA,BBB", "AAA,CCC,DDD",
                             {"engine_picks": ["AAA", "CCC", "DDD"],
                              "executed_picks": ["AAA", "CCC"],
                              "retained": ["BBB"], "consulted": 1,
                              "capacity_truncated": ["BBB"]})
        await _add_rebalance("2026-04", self.ME2, "AAA,CCC", "AAA,CCC,DDD",
                             {"engine_picks": ["AAA", "CCC", "DDD"],
                              "executed_picks": ["AAA", "CCC", "DDD"]})
        return sessions

    async def test_capacity_truncated_defer_is_flagged_and_counted(self, mem_db):
        sessions = await self._seed_book()
        await _add_ledger(event_type="B", engine="monthly", session_date=self.FILL1,
                          ticker="BBB", decision="DEFER")
        await _add_ledger(event_type="B", engine="monthly", session_date=self.FILL1,
                          ticker="AAA", decision="CONFIRM", event_id="B:AAA:1")
        rep = await _score()
        # top-level count
        assert rep["n_capacity_released"] == 1
        # the history entries carry the flag
        by_ticker = {d["ticker"]: d for d in rep["decisions"] if d["event_type"] == "B"}
        assert by_ticker["BBB"]["capacity_released"] is True
        assert by_ticker["AAA"]["capacity_released"] is False
        # ... also on the scored per-type events
        ev = next(e for e in rep["types"]["B"]["events"] if e["ticker"] == "BBB")
        assert ev["capacity_released"] is True
        # delta math UNCHANGED: still the exact displaced-add counterfactual
        e_r = _open_of("BBB", self.FILL1, sessions)
        x_r = _open_of("BBB", "2026-05-01", sessions)
        e_d = _open_of("DDD", self.FILL1, sessions)
        x_d = _open_of("DDD", "2026-05-01", sessions)
        exp = (x_r / e_r - 1.0 - 2 * RATE) - (x_d / e_d - 1.0 - 2 * RATE)
        assert ev["delta"] == pytest.approx(exp)
        assert ev["detail"]["engine_branch"] == "displaced add DDD"
        assert rep["types"]["B"]["n_divergent"] == 1

    async def test_no_capacity_truncated_means_zero_released(self, mem_db):
        """Ordinary DEFERs (no capacity_truncated in the rebuild's sell_layer)
        are NOT flagged — the flag is attribution, not a blanket marker."""
        await _seed_candles(["AAA", "BBB", "CCC", "DDD"])
        await _add_rebalance("2026-03", self.ME1, "AAA,BBB", "AAA,CCC,DDD",
                             {"engine_picks": ["AAA", "CCC", "DDD"],
                              "executed_picks": ["AAA", "CCC", "BBB"],
                              "retained": ["BBB"]})
        await _add_rebalance("2026-04", self.ME2, "AAA,CCC", "AAA,CCC,DDD",
                             {"engine_picks": ["AAA", "CCC", "DDD"],
                              "executed_picks": ["AAA", "CCC", "DDD"]})
        await _add_ledger(event_type="B", engine="monthly", session_date=self.FILL1,
                          ticker="BBB", decision="DEFER")
        rep = await _score()
        assert rep["n_capacity_released"] == 0
        ev = rep["types"]["B"]["events"][0]
        assert ev["capacity_released"] is False
