"""Offline tests for the LLM's frozen-decision ledger (session fill contract, rule 7).

The LLM is ALWAYS stubbed — a fake ``httpx.AsyncClient`` exactly like
``tests/test_llm.py`` / ``tests/test_sim.py``, never a network call — and the
database is in-memory SQLite, never the production DB.

Covered here:

* the hybrid review writes one row per LLM decision with the right relation
  (confirmation / change / veto / hold / addition) and the frozen fill of the
  ones that executed;
* a decision whose trade never executed (exec guard / veto / hold) is stored
  as decided-but-unfilled with ``decided_at`` still frozen at parse time;
* the pure-LLM path (relation ``addition``, no deterministic proposal);
* chat ``[[ACTION]]`` rows (mode ``chat``) with and without fills;
* append-only across sessions (earlier rows bit-for-bit unchanged, the
  per-session guard writes nothing on a same-session repeat);
* a ledger write failure is non-fatal for trading;
* a deterministic-only cycle writes NO ledger rows.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime, timedelta

import httpx
import pandas as pd
import pytest
from sqlalchemy import StaticPool, select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app import sim
from app.config import settings
from app.db import (
    Base,
    SimAccount,
    SimLlmDecision,
    SimPosition,
    SimTrade,
)

# A mid-day, DST-safe parse time for the frozen-clock tests.
_T0 = datetime(2026, 3, 2, 14, 0, tzinfo=UTC)

# The hybrid response: one decision per relation this test class needs.
_HYBRID_RESPONSE = (
    "Confirming one, flipping one, vetoing one, holding one and adding one.\n"
    "[\n"
    '  {"ticker": "AAA", "action": "BUY", "reason": "confirm the setup"},\n'
    '  {"ticker": "BBB", "action": "SELL", "reason": "flip it"},\n'
    '  {"ticker": "CCC", "action": "HOLD", "reason": "veto the exit"},\n'
    '  {"ticker": "DDD", "action": "HOLD", "reason": "nothing to do"},\n'
    '  {"ticker": "EEE", "action": "BUY", "reason": "new idea"}\n'
    "]\n"
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

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
    import app.db as db_mod
    monkeypatch.setattr(db_mod, "Session", session_factory)

    async with session_factory() as s:
        s.add(SimAccount(id=1, cash=10000.0, last_allowance_month=None))
        await s.commit()

    yield session_factory

    await engine.dispose()


@pytest.fixture
def guard_state(monkeypatch, tmp_path):
    """Point the per-session guard file at a temp path (never /data)."""
    state_file = tmp_path / "sim_state.json"
    monkeypatch.setattr(sim, "_STATE_FILE", state_file)
    return state_file


@pytest.fixture
def llm_backend(monkeypatch, tmp_path):
    """Configure a fake backend so the LLM paths reach the stubbed HTTP client."""
    monkeypatch.setattr(sim.llm_mod.settings, "llm_backends", json.dumps([
        {"name": "test", "type": "ollama", "url": "http://test:11434", "model": "m1"},
    ]))
    monkeypatch.setattr(sim.llm_mod, "_DEFAULT_STATE_FILE", tmp_path / "llm_state.json")


def _stub_llm(monkeypatch, *contents: str) -> None:
    """Stub ``llm_mod.chat`` by faking ``httpx.AsyncClient`` (no network).

    Responses are served in order; the last one repeats for any surplus call
    (e.g. the optional prose-summary second shot).
    """
    queue = list(contents)

    class FakeResponse:
        status_code = 200

        def json(self):
            text = queue[0] if len(queue) == 1 else queue.pop(0)
            return {"message": {"content": text}}

    class FakeClient:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def post(self, *a, **k):
            return FakeResponse()

    monkeypatch.setattr(httpx, "AsyncClient", FakeClient)


def _freeze_first_call_clock(monkeypatch, parse_time: datetime) -> None:
    """``sim._utcnow`` returns *parse_time* on its first call, later after.

    Makes the freeze testable: ``decided_at`` must be exactly the parse-time
    capture, while the fill/ledger timestamps come from later calls.
    """
    state = {"calls": 0}

    def _now() -> datetime:
        state["calls"] += 1
        if state["calls"] == 1:
            return parse_time
        return parse_time + timedelta(seconds=30 * state["calls"])

    monkeypatch.setattr(sim, "_utcnow", _now)


async def _rows(factory) -> list[dict]:
    """Every ledger row as a plain dict (all columns), ordered by id."""
    async with factory() as s:
        rows = (await s.scalars(
            select(SimLlmDecision).order_by(SimLlmDecision.id))).all()
        return [{c.name: getattr(r, c.name) for c in SimLlmDecision.__table__.columns}
                for r in rows]


def _valuation(cash: float = 10000.0, equity: float = 10000.0) -> dict:
    return {"cash": cash, "positions_value": 0.0, "total_equity": equity,
            "allowance_total": 10000.0, "positions": []}


# ---------------------------------------------------------------------------
# 1. Hybrid cycle: five decisions, five relations, frozen fills
# ---------------------------------------------------------------------------

class TestHybridCycleLedger:
    async def test_five_relations_and_frozen_fills(self, mem_db, monkeypatch,
                                                   llm_backend):
        _freeze_first_call_clock(monkeypatch, _T0)
        _stub_llm(monkeypatch, _HYBRID_RESPONSE)

        async def fake_close(ticker):
            return {"EEE": 25.0}.get(ticker)
        monkeypatch.setattr(sim, "_latest_close", fake_close)

        proposals = [
            {"ticker": "AAA", "side": "BUY", "price": 100.0, "budget": 2000.0,
             "reason": "det BUY AAA", "entry_stop": 95.0},
            {"ticker": "BBB", "side": "BUY", "price": 50.0, "budget": 1000.0,
             "reason": "det BUY BBB", "entry_stop": 47.5},
            {"ticker": "CCC", "side": "SELL", "price": 20.0, "shares": None,
             "reason": "det SELL CCC"},
        ]
        session_bars = {"EEE": {"open": 25.0, "low": 24.0, "close": 26.0}}

        executed, vetoed = await sim._llm_review_proposals(
            _valuation(), proposals, {}, session_bars=session_bars)

        # Execution behaviour is untouched: AAA (confirmed) + EEE (addition).
        assert [(t["ticker"], t["side"]) for t in executed] == [
            ("AAA", "BUY"), ("EEE", "BUY")]
        assert sorted(v["ticker"] for v in vetoed) == ["BBB", "CCC"]

        rows = await _rows(mem_db)
        assert len(rows) == 5
        by_ticker = {r["ticker"]: r for r in rows}

        # Relations + the deterministic proposal they relate to.
        assert by_ticker["AAA"]["relation"] == "confirmation"
        assert by_ticker["AAA"]["det_side"] == "BUY"
        assert by_ticker["AAA"]["det_reason"] == "det BUY AAA"
        assert by_ticker["BBB"]["relation"] == "change"
        assert by_ticker["BBB"]["det_side"] == "BUY"
        assert by_ticker["BBB"]["det_reason"] == "det BUY BBB"
        assert by_ticker["CCC"]["relation"] == "veto"
        assert by_ticker["CCC"]["det_side"] == "SELL"
        assert by_ticker["CCC"]["det_reason"] == "det SELL CCC"
        assert by_ticker["DDD"]["relation"] == "hold"
        assert by_ticker["DDD"]["det_side"] is None
        assert by_ticker["DDD"]["det_reason"] is None
        assert by_ticker["EEE"]["relation"] == "addition"
        assert by_ticker["EEE"]["det_side"] is None
        assert by_ticker["EEE"]["det_reason"] is None

        # Every row: cycle mode, the cycle's session F date, decided_at frozen
        # at PARSE time (never the later ledger-write time).
        for row in rows:
            assert row["mode"] == "cycle"
            assert row["session_date"] == _T0.date().isoformat()
            assert row["decided_at"] == _T0

        # Filled rows carry the frozen fill; decided_at <= filled_at always.
        for ticker, shares, price, cash_after in (
                ("AAA", 20.0, 100.0, 8000.0), ("EEE", 40.0, 25.0, 7000.0)):
            row = by_ticker[ticker]
            assert row["filled_at"] is not None
            assert row["decided_at"] <= row["filled_at"]
            assert row["filled_at"] > row["decided_at"]  # write did not re-stamp
            assert row["shares"] == pytest.approx(shares)
            assert row["price"] == pytest.approx(price)
            assert row["cash_after"] == pytest.approx(cash_after)

        # Veto / hold / change (flip) executed nothing: NULL fill columns.
        for ticker in ("BBB", "CCC", "DDD"):
            row = by_ticker[ticker]
            assert row["filled_at"] is None
            assert row["shares"] is None
            assert row["price"] is None
            assert row["cash_after"] is None


# ---------------------------------------------------------------------------
# 2. A decision whose fill is blocked stays recorded, unfilled
# ---------------------------------------------------------------------------

class TestDecidedButUnfilled:
    async def test_exec_guard_skip_keeps_decided_at_and_null_fill(
            self, mem_db, monkeypatch, llm_backend):
        _freeze_first_call_clock(monkeypatch, _T0)
        _stub_llm(monkeypatch, (
            "The cash is short this session.\n"
            '[{"ticker": "AAA", "action": "BUY", "reason": "confirm"}]\n'))

        # The account can no longer fund the order (exec guard: cash shortfall).
        async with mem_db() as s:
            acc = await s.get(SimAccount, 1)
            acc.cash = 50.0
            await s.commit()

        proposals = [{"ticker": "AAA", "side": "BUY", "price": 100.0,
                      "budget": 5000.0, "reason": "det BUY AAA"}]
        executed, _vetoed = await sim._llm_review_proposals(
            _valuation(), proposals, {}, session_bars={})

        assert executed == []  # the guard skipped the trade
        async with mem_db() as s:
            trades = (await s.scalars(select(SimTrade))).all()
        assert trades == []

        rows = await _rows(mem_db)
        assert len(rows) == 1
        row = rows[0]
        assert row["ticker"] == "AAA"
        assert row["action"] == "BUY"
        assert row["relation"] == "confirmation"
        # Decided — at the parse-time instant, not at the (later) write.
        assert row["decided_at"] == _T0
        # ... never filled: no pending-order persistence, re-taken next cycle.
        assert row["filled_at"] is None
        assert row["shares"] is None
        assert row["price"] is None
        assert row["cash_after"] is None


# ---------------------------------------------------------------------------
# 3. Pure-LLM cycle: LLM-initiated decisions only
# ---------------------------------------------------------------------------

class TestPureLlmCycleLedger:
    async def test_addition_rows_without_deterministic_proposal(
            self, mem_db, monkeypatch, llm_backend):
        _freeze_first_call_clock(monkeypatch, _T0)
        _stub_llm(monkeypatch, (
            "One new position, one name to leave alone.\n"
            "[\n"
            '  {"ticker": "AAA", "action": "BUY", "reason": "fresh idea"},\n'
            '  {"ticker": "BBB", "action": "HOLD", "reason": "no edge"}\n'
            "]\n"))

        async def fake_close(ticker):
            return {"AAA": 100.0, "BBB": 50.0}.get(ticker)
        monkeypatch.setattr(sim, "_latest_close", fake_close)

        executed = await sim._llm_decide(_valuation(), [], {}, pure_llm=True)

        assert [(t["ticker"], t["side"]) for t in executed] == [("AAA", "BUY")]

        rows = await _rows(mem_db)
        assert [r["ticker"] for r in rows] == ["AAA", "BBB"]
        for row in rows:
            # No deterministic proposal exists in this path.
            assert row["relation"] == "addition"
            assert row["det_side"] is None
            assert row["det_reason"] is None
            assert row["mode"] == "cycle"
            assert row["session_date"] == _T0.date().isoformat()
            assert row["decided_at"] == _T0

        filled = rows[0]
        assert filled["action"] == "BUY"
        assert filled["filled_at"] is not None
        assert filled["decided_at"] <= filled["filled_at"]
        assert filled["shares"] == pytest.approx(10.0)  # 1000 budget @ 100
        assert filled["price"] == pytest.approx(100.0)
        assert filled["cash_after"] == pytest.approx(9000.0)

        held = rows[1]
        assert held["filled_at"] is None
        assert held["shares"] is None


# ---------------------------------------------------------------------------
# 4. Chat [[ACTION]] rows
# ---------------------------------------------------------------------------

class TestChatActionLedger:
    async def test_buy_sell_filled_and_unpriced_action_unfilled(
            self, mem_db, monkeypatch, llm_backend):
        _freeze_first_call_clock(monkeypatch, _T0)
        await sim._exec_buy("AAA", 100.0, 3000.0, "seed")  # 30 shares

        async def fake_close(ticker):
            return {"AAA": 100.0, "BBB": 50.0}.get(ticker)  # ZZZ has no price
        monkeypatch.setattr(sim, "_latest_close", fake_close)

        async def mock_context():
            return "mock context"
        monkeypatch.setattr(sim, "_build_sim_chat_context", mock_context)

        _stub_llm(monkeypatch, (
            "Trimming AAA and buying BBB; ZZZ has no quote.\n"
            "[[ACTION]]\n"
            '{"actions": [\n'
            '  {"ticker": "AAA", "action": "SELL", "amount": 500.0, "reason": "trim"},\n'
            '  {"ticker": "BBB", "action": "BUY", "amount": 1000.0, "reason": "diversify"},\n'
            '  {"ticker": "ZZZ", "action": "BUY", "amount": 500.0, "reason": "no quote"}\n'
            "]}\n"
            "[[/ACTION]]"))

        result = await sim.sim_chat([{"role": "user", "content": "rebalance"}])

        # Slice 3D: the turn only PROPOSED — honest envelope, no ledger rows,
        # no new trades; the unpriced ZZZ action is previewed infeasible.
        assert "trades" not in result and "actions_executed" not in result
        assert len(result["proposals"]) == 1
        actions = result["proposals"][0]["actions"]
        assert [a["ticker"] for a in actions] == ["AAA", "BBB", "ZZZ"]
        assert actions[2]["preview"]["feasible"] is False
        assert actions[2]["preview"]["issue"] == "no price"
        assert await _rows(mem_db) == []
        async with mem_db() as s:
            assert len((await s.scalars(select(SimTrade))).all()) == 1  # the seed buy only
        assert "[[ACTION]]" not in result["text"]
        assert result["history"][-1]["role"] == "assistant"

        # Confirming executes the two fillable actions and skips ZZZ with a
        # machine-readable reason; the ledger rows are frozen at confirm time
        # with the PROPOSAL's parse-time decided_at.
        conf = await sim.confirm_chat_proposal(result["proposals"][0]["proposal_id"])
        assert conf["ok"] is True
        assert [(t["ticker"], t["side"]) for t in conf["trades"]] == [
            ("AAA", "SELL"), ("BBB", "BUY")]
        assert conf["trades"][0]["shares"] == pytest.approx(5.0)
        assert conf["trades"][1]["shares"] == pytest.approx(20.0)
        assert conf["skipped"] == [{"ticker": "ZZZ", "reason": "no price"}]
        assert conf["snapshot_taken"] is True

        rows = await _rows(mem_db)
        assert [r["ticker"] for r in rows] == ["AAA", "BBB", "ZZZ"]
        for row in rows:
            assert row["mode"] == "chat"
            assert row["relation"] == "chat_action"
            assert row["det_side"] is None
            assert row["det_reason"] is None
            assert row["session_date"] == sim._chat_session_date()
            assert row["decided_at"] == _T0

        sell, buy, unpriced = rows
        assert sell["action"] == "SELL"
        assert sell["shares"] == pytest.approx(5.0)
        assert sell["price"] == pytest.approx(100.0)
        assert sell["decided_at"] <= sell["filled_at"]
        assert buy["action"] == "BUY"
        assert buy["shares"] == pytest.approx(20.0)
        assert buy["price"] == pytest.approx(50.0)
        assert buy["decided_at"] <= buy["filled_at"]
        # No quote -> the action was skipped: decided, never filled.
        assert unpriced["action"] == "BUY"
        assert unpriced["filled_at"] is None
        assert unpriced["shares"] is None
        assert unpriced["price"] is None
        assert unpriced["cash_after"] is None

    async def test_stream_action_block_writes_a_ledger_row_on_confirm(
            self, mem_db, monkeypatch):
        """``sim_chat_stream`` — the streaming twin of ``sim_chat`` — PROPOSES
        its ``[[ACTION]]`` block in the done event; the ledger row is frozen
        when the proposal is CONFIRMED, with ``decided_at`` still the stream's
        parse time, so no LLM decision is lost by choosing the streamed
        endpoint."""
        _freeze_first_call_clock(monkeypatch, _T0)

        async def mock_context():
            return "mock context"
        monkeypatch.setattr(sim, "_build_sim_chat_context", mock_context)

        async def fake_close(ticker):
            return 100.0
        monkeypatch.setattr(sim, "_latest_close", fake_close)

        raw_text = (
            "Buying.\n[[ACTION]]\n"
            '{"actions": [{"ticker": "AAA", "action": "BUY", "amount": 500.0,'
            ' "reason": "go"}]}\n[[/ACTION]]')

        async def fake_chat_stream(messages):
            yield {"type": "delta", "text": raw_text}
            yield {"type": "done", "text": raw_text, "reasoning": "",
                   "backend": "a", "model": "m1"}

        monkeypatch.setattr(sim.llm_mod, "chat_stream", fake_chat_stream)

        events = []
        async for evt in sim.sim_chat_stream([{"role": "user", "content": "buy"}]):
            events.append(evt)
        done = next(e for e in events if e["type"] == "done")
        # The streamed turn proposes, it does not execute.
        assert "trades" not in done and "actions_executed" not in done
        assert len(done["proposals"]) == 1
        assert done["proposals"][0]["actions"][0]["preview"]["feasible"] is True
        assert await _rows(mem_db) == []  # proposing alone writes no ledger row

        conf = await sim.confirm_chat_proposal(done["proposals"][0]["proposal_id"])
        assert [(t["ticker"], t["side"]) for t in conf["trades"]] == [("AAA", "BUY")]

        rows = await _rows(mem_db)
        assert len(rows) == 1
        assert rows[0]["mode"] == "chat"
        assert rows[0]["relation"] == "chat_action"
        assert rows[0]["ticker"] == "AAA"
        assert rows[0]["decided_at"] == _T0
        assert rows[0]["shares"] == pytest.approx(5.0)
        assert rows[0]["price"] == pytest.approx(100.0)


# ---------------------------------------------------------------------------
# 5. Append-only across sessions; the per-session guard writes nothing new
# ---------------------------------------------------------------------------

class TestAppendOnlyAcrossCycles:
    async def test_second_session_appends_and_guard_skips_repeat(
            self, mem_db, monkeypatch, llm_backend, guard_state):
        day1 = datetime(2026, 3, 2, 23, 30, tzinfo=UTC)
        day2 = datetime(2026, 3, 3, 23, 30, tzinfo=UTC)

        monkeypatch.setattr(settings, "sim_strategy", "llm")
        monkeypatch.setattr(settings, "sim_benchmark_enabled", False)
        monkeypatch.setattr(settings, "sim_llm_fundamentals_context", False)
        monkeypatch.setattr(settings, "sim_block_negative_roe", False)

        async def fake_candidates():
            return ["AAA", "BBB"]
        monkeypatch.setattr(sim, "_candidate_tickers", fake_candidates)

        async def fake_refresh_many(tickers, period, max_age_seconds=None):
            return [], {}
        monkeypatch.setattr(sim, "refresh_many", fake_refresh_many)

        async def fake_news(signals):
            return {}
        monkeypatch.setattr(sim, "_gather_news", fake_news)

        async def fake_decision_inputs(tickers, held_entries=None, now=None):
            return sim._DecisionInputs(
                f_ts=pd.Timestamp(datetime.now(UTC).date()), s_ts=None,
                signals={}, prices={}, session_bars=None, ratchets={})
        monkeypatch.setattr(sim, "_decision_inputs", fake_decision_inputs)

        async def fake_close(ticker):
            return {"AAA": 100.0, "BBB": 50.0}.get(ticker)
        monkeypatch.setattr(sim, "_latest_close", fake_close)

        _stub_llm(
            monkeypatch,
            'First session.\n[{"ticker": "AAA", "action": "BUY", "reason": "first"}]\n',
            'Second session.\n[{"ticker": "BBB", "action": "BUY", "reason": "second"}]\n',
        )

        # --- cycle 1 (session F = day1) ---
        monkeypatch.setattr(sim, "_utcnow", lambda: day1)
        first = await sim.run_cycle()
        assert "skipped" not in first
        rows_after_1 = await _rows(mem_db)
        assert len(rows_after_1) == 1
        assert rows_after_1[0]["ticker"] == "AAA"
        assert rows_after_1[0]["session_date"] == "2026-03-02"

        # --- cycle 2 (next session) appends, never rewrites ---
        monkeypatch.setattr(sim, "_utcnow", lambda: day2)
        second = await sim.run_cycle()
        assert "skipped" not in second
        rows_after_2 = await _rows(mem_db)
        assert len(rows_after_2) == 2
        # The earlier row is bit-for-bit unchanged (every column compared).
        assert rows_after_2[0] == rows_after_1[0]
        assert rows_after_2[1]["ticker"] == "BBB"
        assert rows_after_2[1]["session_date"] == "2026-03-03"
        assert rows_after_2[1]["relation"] == "addition"
        assert rows_after_2[1]["filled_at"] is not None

        # --- same-session repeat: the guard bails out before any decision ---
        third = await sim.run_cycle()
        assert third.get("skipped") is True
        assert third["reason"] == "session 2026-03-03 already processed"
        assert await _rows(mem_db) == rows_after_2


# ---------------------------------------------------------------------------
# 6. A ledger failure is never a trading failure
# ---------------------------------------------------------------------------

class _AddAllBoomSession:
    """Async-session proxy whose ``add_all`` (the ledger's write call) fails.

    Trades use ``s.add(...)``, so this breaks exactly the ledger write and
    nothing else.
    """

    def __init__(self, inner):
        self._inner = inner
        self._session = None

    async def __aenter__(self):
        self._session = await self._inner.__aenter__()

        def _boom(*a, **k):
            raise RuntimeError("sim_llm_decisions is gone")

        self._session.add_all = _boom
        return self._session

    async def __aexit__(self, *exc):
        return await self._inner.__aexit__(*exc)


class _AddAllBoomFactory:
    def __init__(self, factory):
        self._factory = factory

    def __call__(self):
        return _AddAllBoomSession(self._factory())


class TestLedgerFailureIsNonFatal:
    async def test_raising_writer_never_fails_the_confirm(
            self, mem_db, monkeypatch, llm_backend, caplog):
        async def boom(rows):
            raise RuntimeError("ledger writer exploded")
        monkeypatch.setattr(sim, "_ledger_llm_decisions", boom)

        async def mock_context():
            return "mock context"
        monkeypatch.setattr(sim, "_build_sim_chat_context", mock_context)

        async def fake_close(ticker):
            return 100.0
        monkeypatch.setattr(sim, "_latest_close", fake_close)

        _stub_llm(monkeypatch, (
            "Buying.\n[[ACTION]]\n"
            '{"actions": [{"ticker": "AAA", "action": "BUY", "amount": 500.0,'
            ' "reason": "go"}]}\n[[/ACTION]]'))

        # The chat turn only proposes — the ledger failure can only surface at
        # confirm, when the confirmed trades are already committed real state.
        result = await sim.sim_chat([{"role": "user", "content": "buy AAA"}])
        assert len(result["proposals"]) == 1
        assert await _rows(mem_db) == []

        with caplog.at_level(logging.ERROR, logger="trade_sentinel.sim"):
            conf = await sim.confirm_chat_proposal(result["proposals"][0]["proposal_id"])

        assert conf["ok"] is True
        assert [t["ticker"] for t in conf["trades"]] == ["AAA"]
        async with mem_db() as s:
            positions = (await s.scalars(select(SimPosition))).all()
        assert [p.ticker for p in positions] == ["AAA"]  # the trade is real state
        assert await _rows(mem_db) == []  # the ledger write was lost
        assert any(r.levelno >= logging.ERROR and "ledger" in r.getMessage().lower()
                   for r in caplog.records)

    async def test_failing_db_write_is_logged_and_swallowed(
            self, mem_db, monkeypatch, llm_backend, caplog):
        # The REAL writer, with its only DB call (add_all) failing.
        monkeypatch.setattr(sim, "Session", _AddAllBoomFactory(mem_db))

        async def mock_context():
            return "mock context"
        monkeypatch.setattr(sim, "_build_sim_chat_context", mock_context)

        async def fake_close(ticker):
            return 100.0
        monkeypatch.setattr(sim, "_latest_close", fake_close)

        _stub_llm(monkeypatch, (
            "Buying.\n[[ACTION]]\n"
            '{"actions": [{"ticker": "AAA", "action": "BUY", "amount": 500.0,'
            ' "reason": "go"}]}\n[[/ACTION]]'))

        result = await sim.sim_chat([{"role": "user", "content": "buy AAA"}])
        assert len(result["proposals"]) == 1

        with caplog.at_level(logging.ERROR, logger="trade_sentinel.sim"):
            conf = await sim.confirm_chat_proposal(result["proposals"][0]["proposal_id"])

        assert conf["ok"] is True
        assert [t["ticker"] for t in conf["trades"]] == ["AAA"]
        assert any(r.levelno >= logging.ERROR and "ledger write" in r.getMessage().lower()
                   for r in caplog.records)
        # The trades themselves were committed before the failed ledger write.
        async with mem_db() as s:
            trades = (await s.scalars(select(SimTrade))).all()
        assert len(trades) == 1


# ---------------------------------------------------------------------------
# 7. A deterministic-only cycle writes NO ledger rows
# ---------------------------------------------------------------------------

class TestDeterministicCycleWritesNothing:
    async def test_deterministic_cycle_has_no_ledger_rows(
            self, mem_db, monkeypatch, llm_backend, guard_state):
        monkeypatch.setattr(settings, "sim_strategy", "deterministic")
        monkeypatch.setattr(settings, "sim_benchmark_enabled", False)

        async def fake_candidates():
            return []
        monkeypatch.setattr(sim, "_candidate_tickers", fake_candidates)

        async def fake_refresh_many(tickers, period, max_age_seconds=None):
            return [], {}
        monkeypatch.setattr(sim, "refresh_many", fake_refresh_many)

        result = await sim.run_cycle()

        assert "skipped" not in result
        assert result["trades"] == []
        assert await _rows(mem_db) == []
