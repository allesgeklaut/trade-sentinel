"""Tests for the autonomous paper-trading simulation engine (app.sim).

These tests use an in-memory SQLite database so they don't touch the real
data volume.  They cover:

- Account creation and allowance deposit logic
- Valuation with and without positions
- Buy / sell execution and position math (weighted avg cost, FIFO selling)
- Reset
- LLM decision parsing (no network)
"""

from __future__ import annotations

import json
import random
from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest
from sqlalchemy import StaticPool, func, select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app import sim
from app.config import settings
from app.db import (
    Base,
    Candle,
    SimAccount,
    SimPosition,
    SimSnapshot,
    SimTrade,
)

# ---------------------------------------------------------------------------
# Fixtures
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
    # Also patch db.Session in case code imports it directly
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
    """Give the sim account some starting cash."""
    async with mem_db() as s:
        acc = await s.get(SimAccount, 1)
        acc.cash = 10000.0
        await s.commit()
    return mem_db


@pytest.fixture
def llm_backend(monkeypatch):
    """Configure a fake LLM backend so sim.chat/sim_chat reach the mocked
    HTTP client instead of short-circuiting on "No LLM backend configured".

    Without this, the sim-chat and LLM-decide tests only pass when a real
    backend exists in the operator's .env, and several of them "pass" by
    silently hitting the no-backend fallback instead of exercising the mock.
    """
    monkeypatch.setattr(sim.llm_mod.settings, "llm_backends", json.dumps([
        {"name": "test", "type": "ollama", "url": "http://test:11434", "model": "m1"},
    ]))
    monkeypatch.setattr(sim.llm_mod.settings, "llm_state_file",
                        "/tmp/test-llm-state-none.json")


# ---------------------------------------------------------------------------
# Benchmark (DCA control portfolio)
# ---------------------------------------------------------------------------

class TestBenchmark:
    async def test_benchmark_valuate_empty(self, mem_db, monkeypatch):
        """Benchmark valuation with no deposits should be zero."""
        async def mock_close(ticker):
            return 100.0
        monkeypatch.setattr(sim, "_latest_close", mock_close)
        val = await sim.benchmark_valuate()
        assert val["shares"] == 0
        assert val["total_equity"] == 0.0

    async def test_benchmark_deposit_and_buy(self, mem_db, monkeypatch):
        """DCA deposit should buy fractional shares at current price."""
        monkeypatch.setattr(sim, "_current_month", lambda: "2026-01")
        async def mock_close(ticker):
            return 100.0
        monkeypatch.setattr(sim, "_latest_close", mock_close)
        result = await sim._benchmark_deposit_and_buy()
        assert result["deposited"] is True
        assert result["amount"] == settings.sim_monthly_allowance
        assert result["shares"] == settings.sim_monthly_allowance / 100.0

    async def test_benchmark_no_double_deposit(self, mem_db, monkeypatch):
        """Second call in the same month should not deposit again."""
        monkeypatch.setattr(sim, "_current_month", lambda: "2026-01")
        async def mock_close(ticker):
            return 100.0
        monkeypatch.setattr(sim, "_latest_close", mock_close)
        await sim._benchmark_deposit_and_buy()
        result = await sim._benchmark_deposit_and_buy()
        assert result["deposited"] is False

    async def test_benchmark_deposit_next_month(self, mem_db, monkeypatch):
        """A new month should trigger a fresh DCA deposit."""
        monkeypatch.setattr(sim, "_current_month", lambda: "2026-01")
        async def mock_close(ticker):
            return 100.0
        monkeypatch.setattr(sim, "_latest_close", mock_close)
        await sim._benchmark_deposit_and_buy()
        monkeypatch.setattr(sim, "_current_month", lambda: "2026-02")
        result = await sim._benchmark_deposit_and_buy()
        assert result["deposited"] is True
        assert result["month"] == "2026-02"

    async def test_benchmark_valuate_after_deposit(self, mem_db, monkeypatch):
        """Valuation should reflect shares x current price after a deposit."""
        monkeypatch.setattr(sim, "_current_month", lambda: "2026-01")
        async def mock_close(ticker):
            return 100.0
        monkeypatch.setattr(sim, "_latest_close", mock_close)
        await sim._benchmark_deposit_and_buy()
        # Price goes up to 120
        async def mock_close_120(ticker):
            return 120.0
        monkeypatch.setattr(sim, "_latest_close", mock_close_120)
        val = await sim.benchmark_valuate()
        # 1000 / 100 = 10 shares, now worth 120 each = 1200
        assert val["shares"] == 10.0
        assert val["total_equity"] == 1200.0
        assert val["current_price"] == 120.0

    async def test_benchmark_no_price_skips(self, mem_db, monkeypatch):
        """If no price is available, the deposit should be skipped."""
        monkeypatch.setattr(sim, "_current_month", lambda: "2026-01")
        async def mock_close_none(ticker):
            return None
        monkeypatch.setattr(sim, "_latest_close", mock_close_none)
        result = await sim._benchmark_deposit_and_buy()
        assert result["deposited"] is False

    async def test_benchmark_reset(self, mem_db, monkeypatch):
        """Reset should wipe benchmark tables."""
        monkeypatch.setattr(sim, "_current_month", lambda: "2026-01")
        async def mock_close(ticker):
            return 100.0
        monkeypatch.setattr(sim, "_latest_close", mock_close)
        await sim._benchmark_deposit_and_buy()
        await sim.reset_benchmark()
        val = await sim.benchmark_valuate()
        assert val["shares"] == 0
        assert val["total_equity"] == 0.0

    async def test_benchmark_multiple_deposits_avg_cost(self, mem_db, monkeypatch):
        """Multiple DCA deposits at different prices should track weighted avg cost."""
        # Month 1: price 100
        monkeypatch.setattr(sim, "_current_month", lambda: "2026-01")
        async def mock_close_100(ticker):
            return 100.0
        monkeypatch.setattr(sim, "_latest_close", mock_close_100)
        await sim._benchmark_deposit_and_buy()  # 10 shares @ 100
        # Month 2: price 200
        monkeypatch.setattr(sim, "_current_month", lambda: "2026-02")
        async def mock_close_200(ticker):
            return 200.0
        monkeypatch.setattr(sim, "_latest_close", mock_close_200)
        await sim._benchmark_deposit_and_buy()  # 5 shares @ 200

        val = await sim.benchmark_valuate()
        # 15 shares total, avg cost = (10*100 + 5*200) / 15 = 2000/15 ~= 133.33
        assert val["shares"] == 15.0
        assert val["avg_cost"] == pytest.approx(133.33, abs=0.1)


# ---------------------------------------------------------------------------
# Allowance
# ---------------------------------------------------------------------------

class TestAllowance:
    async def test_deposit_on_new_month(self, mem_db, monkeypatch):
        """First deposit should add the monthly allowance and record it."""
        monkeypatch.setattr(sim, "_current_month", lambda: "2026-01")
        result = await sim.deposit_allowance()
        assert result["deposited"] is True
        assert result["amount"] == settings.sim_monthly_allowance

    async def test_no_double_deposit_same_month(self, mem_db, monkeypatch):
        """Second call in the same month should not deposit again."""
        monkeypatch.setattr(sim, "_current_month", lambda: "2026-01")
        await sim.deposit_allowance()
        result = await sim.deposit_allowance()
        assert result["deposited"] is False

    async def test_deposit_next_month(self, mem_db, monkeypatch):
        """A new month should trigger a fresh deposit."""
        monkeypatch.setattr(sim, "_current_month", lambda: "2026-01")
        await sim.deposit_allowance()
        monkeypatch.setattr(sim, "_current_month", lambda: "2026-02")
        result = await sim.deposit_allowance()
        assert result["deposited"] is True
        assert result["month"] == "2026-02"

    async def test_allowance_recorded(self, mem_db, monkeypatch):
        """Allowance records should be queryable."""
        monkeypatch.setattr(sim, "_current_month", lambda: "2026-03")
        await sim.deposit_allowance()
        records = await sim.get_allowances()
        assert len(records) == 1
        assert records[0]["month"] == "2026-03"
        assert records[0]["amount"] == settings.sim_monthly_allowance


# ---------------------------------------------------------------------------
# Valuation
# ---------------------------------------------------------------------------

class TestValuation:
    async def test_empty_portfolio(self, with_cash):
        """Valuation with no positions should just return cash."""
        val = await sim.valuate()
        assert val["cash"] == 10000.0
        assert val["positions_value"] == 0.0
        assert val["total_equity"] == 10000.0
        assert val["positions"] == []

    async def test_with_position(self, with_cash, monkeypatch):
        """Valuation should include position value at current close."""
        # Insert a position
        async with sim.Session() as s:
            s.add(SimPosition(ticker="AAPL", shares=10, avg_cost=150.0))
            await s.commit()

        # Mock _latest_close to return 160
        async def mock_close(ticker):
            return 160.0 if ticker == "AAPL" else None
        monkeypatch.setattr(sim, "_latest_close", mock_close)

        val = await sim.valuate()
        assert val["cash"] == 10000.0
        assert val["positions_value"] == 1600.0  # 10 * 160
        assert val["total_equity"] == 11600.0
        assert len(val["positions"]) == 1
        assert val["positions"][0]["ticker"] == "AAPL"
        assert val["positions"][0]["pnl_pct"] == pytest.approx(6.67, abs=0.1)

    async def test_allowance_total_tracked(self, with_cash, monkeypatch):
        """Allowance total should accumulate across deposits."""
        monkeypatch.setattr(sim, "_current_month", lambda: "2026-01")
        await sim.deposit_allowance()
        monkeypatch.setattr(sim, "_current_month", lambda: "2026-02")
        await sim.deposit_allowance()
        val = await sim.valuate()
        assert val["allowance_total"] == pytest.approx(settings.sim_monthly_allowance * 2)


# ---------------------------------------------------------------------------
# Buy / Sell execution
# ---------------------------------------------------------------------------

class TestExecBuy:
    async def test_basic_buy(self, with_cash):
        """A buy should reduce cash, create a position, and log a trade."""
        t = await sim._exec_buy("AAPL", 150.0, 5000.0, "test buy")
        assert t is not None
        assert t["ticker"] == "AAPL"
        assert t["side"] == "BUY"
        assert t["shares"] == pytest.approx(33.3333, abs=0.001)  # 5000 / 150 fractional
        assert t["price"] == 150.0

        # Check state
        async with sim.Session() as s:
            from sqlalchemy import select as sa_select
            pos = await s.scalar(sa_select(SimPosition).where(SimPosition.ticker == "AAPL"))
            assert pos is not None
            assert pos.shares == pytest.approx(33.3333, abs=0.001)
            assert pos.avg_cost == 150.0

            acc = await s.get(SimAccount, 1)
            assert acc.cash == pytest.approx(10000 - 33.3333 * 150, abs=0.01)

            trades = (await s.scalars(sa_select(SimTrade))).all()
            assert len(trades) == 1

    async def test_buy_insufficient_budget(self, with_cash):
        """A buy with budget under $1 should be skipped."""
        t = await sim._exec_buy("BRK.A", 500_000, 0.5, "too expensive")
        assert t is None

    async def test_buy_zero_price(self, with_cash):
        """A buy at zero price should be skipped."""
        t = await sim._exec_buy("AAPL", 0, 5000, "zero price")
        assert t is None

    async def test_buy_adds_to_existing_position(self, with_cash):
        """A second buy should update the weighted average cost."""
        await sim._exec_buy("AAPL", 100.0, 2000.0, "first buy")  # 20 shares @ 100
        await sim._exec_buy("AAPL", 150.0, 1500.0, "second buy")  # 10 shares @ 150

        async with sim.Session() as s:
            from sqlalchemy import select as sa_select
            pos = await s.scalar(sa_select(SimPosition).where(SimPosition.ticker == "AAPL"))
            assert pos is not None
            assert pos.shares == pytest.approx(30.0, abs=0.001)
            # Weighted avg: (20*100 + 10*150) / 30 = 3500/30 ≈ 116.67
            assert pos.avg_cost == pytest.approx(116.67, abs=0.1)


class TestExecSell:
    async def test_basic_sell(self, with_cash):
        """A sell should increase cash, reduce position, and log a trade."""
        await sim._exec_buy("AAPL", 100.0, 2000.0, "buy")  # 20 shares @ 100

        t = await sim._exec_sell("AAPL", 120.0, None, "sell all")
        assert t is not None
        assert t["side"] == "SELL"
        assert t["shares"] == pytest.approx(20.0, abs=0.001)
        assert t["price"] == 120.0

        async with sim.Session() as s:
            from sqlalchemy import select as sa_select
            pos = await s.scalar(sa_select(SimPosition).where(SimPosition.ticker == "AAPL"))
            assert pos is None  # fully sold, position deleted

            acc = await s.get(SimAccount, 1)
            # cash = 10000 - 2000 (buy) + 2400 (sell 20*120)
            assert acc.cash == pytest.approx(10000 - 2000 + 2400, abs=0.01)

    async def test_sell_no_position(self, with_cash):
        """Selling a ticker with no position should return None."""
        t = await sim._exec_sell("MSFT", 100.0, None, "nothing to sell")
        assert t is None

    async def test_partial_sell(self, with_cash):
        """A partial sell should reduce shares but keep the position."""
        await sim._exec_buy("AAPL", 100.0, 3000.0, "buy")  # 30 shares

        t = await sim._exec_sell("AAPL", 110.0, 10, "partial sell")
        assert t is not None
        assert t["shares"] == pytest.approx(10.0, abs=0.001)

        async with sim.Session() as s:
            from sqlalchemy import select as sa_select
            pos = await s.scalar(sa_select(SimPosition).where(SimPosition.ticker == "AAPL"))
            assert pos is not None
            assert pos.shares == pytest.approx(20.0, abs=0.001)  # 30 - 10


# ---------------------------------------------------------------------------
# Reset
# ---------------------------------------------------------------------------

class TestReset:
    async def test_reset_preserves_history_clears_current_book(
            self, with_cash, monkeypatch, tmp_path):
        """``reset_sim()`` CLOSES the epoch; it does not rewrite it.

        The CURRENT book is cleared and re-seeded (positions gone, cash =
        start, month/week markers None) and a ``RunEpoch`` boundary moves the
        engine to epoch 2, while the closed epoch's ledger — trades, snapshots
        and allowances — survives bit-for-bit in every mapped column. The old
        test expected every table to be wiped, i.e. the destructive reset that
        rewrote the paper book's own record; the epoch boundary in
        app/run_config.py replaces it.

        The DCA benchmark is the documented DERIVED-ARTIFACT exception (a
        comparison curve, not a decision book) and is already covered by
        ``TestBenchmark.test_benchmark_reset`` — not re-asserted here.
        """
        from sqlalchemy import select as sa_select

        from app import run_config
        from app.db import RunEpoch, SimAllowance

        # reset re-arms the per-session guard in the sim state file
        monkeypatch.setattr(sim, "_STATE_FILE", tmp_path / "state.json")

        # Create some state: a filled trade, a snapshot, an allowance and the
        # account markers the reset must clear.
        await sim._exec_buy("AAPL", 100.0, 2000.0, "buy")
        async with sim.Session() as s:
            s.add(SimSnapshot(cash=8000, positions_value=2000,
                              total_equity=10000, allowance_total=0))
            s.add(SimAllowance(amount=1000.0, month="2026-09"))
            acc = await s.get(SimAccount, 1)
            acc.last_allowance_month = "2026-09"
            acc.last_review_week = "2026-W36"
            await s.commit()

        async def dump(model) -> list[dict]:
            """Every mapped column of every row of *model*, in id order."""
            async with sim.Session() as s:
                rows = (await s.scalars(
                    sa_select(model).order_by(model.id))).all()
            cols = [c.name for c in model.__table__.columns]
            return [{c: getattr(r, c) for c in cols} for r in rows]

        tables = (SimTrade, SimSnapshot, SimAllowance)
        before = {m.__name__: await dump(m) for m in tables}

        result = await sim.reset_sim()
        assert result["ok"] is True
        assert result["cash"] == settings.sim_start_cash
        assert result["epoch"] == 2
        assert result["preserved"] == {"trades": 1, "snapshots": 1,
                                       "allowances": 1}

        async with sim.Session() as s:
            pos = await s.scalar(sa_select(SimPosition).where(
                SimPosition.ticker == "AAPL"))
            assert pos is None
            assert (await s.scalars(sa_select(SimPosition))).all() == []
            acc = await s.get(SimAccount, 1)
            assert acc.cash == settings.sim_start_cash
            assert acc.last_allowance_month is None
            assert acc.last_review_week is None
            epochs = (await s.scalars(sa_select(RunEpoch))).all()
            assert [(e.engine, e.epoch) for e in epochs] == [("sim", 2)]
            assert await run_config.current_epoch("sim", session=s) == 2

        # The closed epoch's ledger is preserved bit-for-bit ...
        after = {m.__name__: await dump(m) for m in tables}
        for name, rows in before.items():
            assert after[name] == rows, \
                f"{name}: the reset rewrote the closed epoch's rows"

        # ... and the fresh epoch's curve starts EMPTY (no closed-epoch leak).
        assert await sim.get_equity_curve() == []


# ---------------------------------------------------------------------------
# LLM decision parsing
# ---------------------------------------------------------------------------

class TestParseLLMDecisions:
    def test_valid_json(self):
        content = json.dumps([
            {"ticker": "AAPL", "action": "BUY", "reason": "strong uptrend"},
            {"ticker": "MSFT", "action": "HOLD", "reason": "overbought"},
        ])
        result = sim._parse_llm_decisions(content)
        assert result is not None
        assert len(result) == 2
        assert result[0]["ticker"] == "AAPL"
        assert result[0]["action"] == "BUY"

    def test_markdown_fenced_json(self):
        content = '```json\n[{"ticker":"NVDA","action":"SELL","reason":"bearish"}]\n```'
        result = sim._parse_llm_decisions(content)
        assert result is not None
        assert len(result) == 1
        assert result[0]["ticker"] == "NVDA"
        assert result[0]["action"] == "SELL"

    def test_prose_wrapped_json(self):
        content = 'Here are my decisions:\n[{"ticker":"AMD","action":"BUY","reason":"RSI cross"}]\nThat is all.'
        result = sim._parse_llm_decisions(content)
        assert result is not None
        assert len(result) == 1

    def test_invalid_json_returns_none(self):
        assert sim._parse_llm_decisions("not json at all") is None

    def test_empty_returns_none(self):
        assert sim._parse_llm_decisions("") is None
        assert sim._parse_llm_decisions(None) is None

    def test_action_normalization(self):
        content = json.dumps([
            {"ticker": "A", "action": "buy", "reason": "lowercase"},
            {"ticker": "B", "action": "Sell", "reason": "mixed case"},
        ])
        result = sim._parse_llm_decisions(content)
        assert result is not None
        assert result[0]["action"] == "BUY"
        assert result[1]["action"] == "SELL"

    def test_invalid_action_filtered(self):
        content = json.dumps([
            {"ticker": "A", "action": "MOON", "reason": "invalid"},
            {"ticker": "B", "action": "BUY", "reason": "valid"},
        ])
        result = sim._parse_llm_decisions(content)
        assert result is not None
        assert len(result) == 1
        assert result[0]["ticker"] == "B"

    def test_missing_ticker_filtered(self):
        content = json.dumps([
            {"action": "BUY", "reason": "no ticker"},
        ])
        result = sim._parse_llm_decisions(content)
        # All entries filtered out → None
        assert result is None

    def test_partial_size_fields(self):
        """Optional 'shares'/'amount' fields should be extracted and normalized."""
        content = json.dumps([
            {"ticker": "MDB", "action": "SELL", "amount": 67.43, "reason": "trim"},
            {"ticker": "ANET", "action": "SELL", "shares": 3.5, "reason": "trim"},
            {"ticker": "SPY", "action": "BUY", "amount": 500, "reason": "diversify"},
            {"ticker": "X", "action": "SELL", "shares": 3, "amount": 100, "reason": "both"},
        ])
        result = sim._parse_llm_decisions(content)
        assert result is not None
        assert len(result) == 4
        assert result[0]["amount"] == pytest.approx(67.43)
        assert "shares" not in result[0]
        assert result[1]["shares"] == pytest.approx(3.5)
        assert "amount" not in result[1]
        assert result[2]["amount"] == pytest.approx(500.0)
        assert result[3]["shares"] == pytest.approx(3.0)
        assert result[3]["amount"] == pytest.approx(100.0)

    def test_empty_array_is_valid(self):
        """An explicitly empty JSON array is a valid 'do nothing' decision."""
        content = json.dumps([])
        result = sim._parse_llm_decisions(content)
        # Empty list, not None — distinguishes 'do nothing' from a parse failure
        assert result == []

    def test_invalid_size_fields_ignored(self):
        """Non-positive or non-numeric size fields should be dropped."""
        content = json.dumps([
            {"ticker": "A", "action": "SELL", "amount": -5, "reason": "neg"},
            {"ticker": "B", "action": "SELL", "shares": 0, "reason": "zero"},
            {"ticker": "C", "action": "SELL", "amount": "lots", "reason": "str"},
            {"ticker": "D", "action": "SELL", "amount": True, "reason": "bool"},
        ])
        result = sim._parse_llm_decisions(content)
        assert result is not None
        assert len(result) == 4
        for r in result:
            assert "shares" not in r
            assert "amount" not in r

    def test_prose_brackets_before_json(self):
        """Prose containing '[' must not be mistaken for the decisions array."""
        content = (
            "I reviewed [NVDA] and [AMD]. The setup is mixed.\n"
            '[{"ticker":"NVDA","action":"HOLD","reason":"overbought"}]'
        )
        result = sim._parse_llm_decisions(content)
        assert result is not None
        assert len(result) == 1
        assert result[0]["ticker"] == "NVDA"
        assert result[0]["action"] == "HOLD"

    def test_brackets_inside_json_strings(self):
        """Square brackets inside reason strings must not truncate the array."""
        content = (
            'Summary [note]. [{"ticker":"X","action":"HOLD",'
            '"reason":"see [1] and [2] for detail"}]'
        )
        result = sim._parse_llm_decisions(content)
        assert result is not None
        assert len(result) == 1
        assert result[0]["ticker"] == "X"
        assert result[0]["reason"] == "see [1] and [2] for detail"

    def test_trailing_prose_after_json(self):
        """Prose after the array should not break parsing."""
        content = (
            'Here is my read.\n'
            '[{"ticker":"SPY","action":"SELL","reason":"trim"}]\n'
            "That is all, happy to elaborate."
        )
        result = sim._parse_llm_decisions(content)
        assert result is not None
        assert len(result) == 1
        assert result[0]["ticker"] == "SPY"

    def test_prose_brackets_no_array_returns_none(self):
        """Prose with brackets but no real decisions array → None (fallback)."""
        assert sim._parse_llm_decisions("I reviewed [NVDA] but decided nothing.") is None


class TestBuildLLMContext:
    def test_context_includes_portfolio(self):
        val = {
            "cash": 5000.0,
            "positions_value": 3000.0,
            "total_equity": 8000.0,
            "allowance_total": 10000.0,
            "positions": [
                {"ticker": "AAPL", "shares": 10, "avg_cost": 150.0,
                 "current_price": 160.0, "value": 1600.0, "pnl_pct": 6.67},
            ],
        }
        ctx = sim._build_llm_context(val, [], {})
        assert "5000.00" in ctx
        assert "8000.00" in ctx
        assert "AAPL" in ctx
        assert "10000.00" in ctx

    def test_context_includes_signals(self):
        val = {"cash": 0, "positions_value": 0, "total_equity": 0,
               "allowance_total": 0, "positions": []}
        signals = {
            "NVDA": {"action": "BUY", "strength": 85,
                     "snapshot": {"close": 120.0, "rsi": 55.0, "macd": 0.5}},
        }
        ctx = sim._build_llm_context(val, [], signals)
        assert "NVDA" in ctx
        assert "BUY" in ctx

    def test_context_includes_deterministic_trades(self):
        val = {"cash": 0, "positions_value": 0, "total_equity": 0,
               "allowance_total": 0, "positions": []}
        trades = [
            {"ticker": "AMD", "side": "BUY", "shares": 5, "price": 150.0,
             "reason": "strong momentum"},
        ]
        ctx = sim._build_llm_context(val, trades, {})
        assert "AMD" in ctx
        assert "strong momentum" in ctx

    def test_all_signals_shown_regardless_of_strength(self):
        """The full signal table is always shown — the LLM is the
        decision-maker and needs to see every candidate. (Truncation was
        removed: it starved the LLM of context and caused churn.)"""
        val = {"cash": 0, "positions_value": 0, "total_equity": 0,
               "allowance_total": 0, "positions": []}
        # 40 BUY signals with different strengths, plus weak SELLs.
        signals = {
            f"T{i:02d}": {"action": "BUY", "strength": i,
                           "snapshot": {"close": 100.0, "rsi": 50.0, "macd": 0.0}}
            for i in range(40)
        }
        for i in range(5):
            signals[f"SELL{i:02d}"] = {"action": "SELL", "strength": 10,
                                       "snapshot": {"close": 50.0, "rsi": 30.0, "macd": -0.5}}
        ctx = sim._build_llm_context(val, [], signals)
        # Every ticker appears — no truncation, no "hidden" notice.
        assert "T00" in ctx
        assert "T39" in ctx
        assert "SELL00" in ctx
        assert "SELL04" in ctx
        assert "hidden" not in ctx
        assert "all candidate tickers" in ctx

    def test_context_shows_position_thesis(self):
        """The context must show each position's entry thesis and holding
        date so the LLM can judge 'is the thesis still valid?'"""
        val = {
            "cash": 0, "positions_value": 0, "total_equity": 0,
            "allowance_total": 0,
            "positions": [
                {"ticker": "NVDA", "shares": 5, "avg_cost": 100.0,
                 "current_price": 120.0, "value": 600.0, "pnl_pct": 20.0,
                 "thesis": "AI infrastructure leader, strong uptrend",
                 "buy_date": "2025-06-01"},
            ],
        }
        ctx = sim._build_llm_context(val, [], {}, pure_llm=True)
        assert "AI infrastructure leader" in ctx
        assert "2025-06-01" in ctx
        assert "thesis:" in ctx

    def test_context_truncates_long_thesis(self):
        """Long theses are truncated to 77 chars + ... to keep lines readable."""
        long_thesis = "x" * 100
        val = {
            "cash": 0, "positions_value": 0, "total_equity": 0,
            "allowance_total": 0,
            "positions": [
                {"ticker": "A", "shares": 1, "avg_cost": 10.0,
                 "current_price": 10.0, "value": 10.0, "pnl_pct": 0.0,
                 "thesis": long_thesis, "buy_date": "2025-01-01"},
            ],
        }
        ctx = sim._build_llm_context(val, [], {}, pure_llm=True)
        # The full 100-char thesis should not appear; the truncated version should.
        assert long_thesis not in ctx
        assert "xxx..." in ctx  # truncated to 77 chars + "..."

    def test_context_no_thesis_shows_nothing(self):
        """Positions without a thesis don't show a 'thesis:' line."""
        val = {
            "cash": 0, "positions_value": 0, "total_equity": 0,
            "allowance_total": 0,
            "positions": [
                {"ticker": "A", "shares": 1, "avg_cost": 10.0,
                 "current_price": 10.0, "value": 10.0, "pnl_pct": 0.0,
                 "thesis": "", "buy_date": ""},
            ],
        }
        ctx = sim._build_llm_context(val, [], {}, pure_llm=True)
        assert "thesis:" not in ctx
        assert "since" not in ctx

    def test_context_includes_recent_trades(self):
        """The Recent Trades section lets the LLM see its own recent activity
        so it can avoid round-trips and repeated mistakes."""
        val = {"cash": 0, "positions_value": 0, "total_equity": 0,
               "allowance_total": 0, "positions": []}
        history = [
            {"date": "2025-06-10", "side": "SELL", "ticker": "ASML.AS",
             "shares": 5.0, "price": 100.0, "reason": "momentum rolled over"},
            {"date": "2025-06-08", "side": "BUY", "ticker": "NVDA",
             "shares": 3.0, "price": 120.0, "reason": "pullback in uptrend"},
        ]
        ctx = sim._build_llm_context(val, [], {}, pure_llm=True,
                                     trade_history=history)
        assert "Recent Trades" in ctx
        assert "ASML.AS" in ctx
        assert "2025-06-10" in ctx
        assert "momentum rolled over" in ctx

    def test_context_truncates_trade_history_to_12(self):
        """Only the last 12 trades are shown — the context stays compact."""
        val = {
            "cash": 0, "positions_value": 0, "total_equity": 0,
            "allowance_total": 0, "positions": [],
        }
        history = [
            {"date": f"2025-06-{i:02d}", "side": "BUY", "ticker": f"T{i:02d}",
             "shares": 1.0, "price": 100.0, "reason": "test"}
            for i in range(20)
        ]
        ctx = sim._build_llm_context(val, [], {}, pure_llm=True, trade_history=history)
        assert "Recent Trades" in ctx
        # Only the first 12 (by provided order) are shown.
        assert "T00" in ctx  # first trade shown
        assert "T19" not in ctx  # 20th trade truncated

    def test_context_no_recent_trades_omits_section(self):
        val = {
            "cash": 0, "positions_value": 0, "total_equity": 0,
            "allowance_total": 0, "positions": [],
        }
        ctx = sim._build_llm_context(val, [], {}, pure_llm=True)
        assert "Recent Trades" not in ctx


class TestRunCycleResetsLLMState:
    """run_cycle() must reset all per-cycle LLM state at the start, so a
    deterministic cycle never re-reports a previous LLM cycle's reasoning
    (the frontend's llm_called flag and reasoning panel depend on this)."""

    async def test_llm_state_cleared_before_deterministic_cycle(self, mem_db, monkeypatch, llm_backend):
        # Stale state from a "previous" hybrid cycle.
        sim._last_llm_reasoning = "Some previous cycle's reasoning"
        sim._last_llm_summary = "Previous summary"
        sim._last_llm_decisions = [{"ticker": "OLD", "action": "BUY", "reason": "stale"}]
        sim._last_llm_vetoes = [{"ticker": "OLD", "action": "HOLD", "reason": "stale"}]

        monkeypatch.setattr(settings, "sim_strategy", "deterministic")
        monkeypatch.setattr(settings, "sim_benchmark_enabled", False)

        # Minimal run_cycle setup: no candidates, no benchmark.
        async def mock_candidates():
            return []
        monkeypatch.setattr(sim, "_candidate_tickers", mock_candidates)

        result = await sim.run_cycle()

        assert "skipped" not in result, "cycle should have run"
        assert sim._last_llm_reasoning == ""
        assert sim._last_llm_summary == ""
        assert sim._last_llm_decisions == []
        assert sim._last_llm_vetoes == []

        # The frontend-facing summary must report the LLM as not called.
        summary = sim.get_last_llm_summary()
        assert summary["llm_called"] is False
        assert summary["raw"] == ""


# ---------------------------------------------------------------------------
# Year-long simulation with synthetic market data
# ---------------------------------------------------------------------------


def _gen_synthetic_candles(start_price: float, daily_drift: float,
                           n: int = 500, seed: int = 42) -> list[dict]:
    """Generate *n* daily OHLCV candles with given drift + Gaussian noise.

    Returns a list of dicts in the same shape as ``market.candles()``:
    ``{time, open, high, low, close, volume}``.
    """
    rng = random.Random(seed)
    rows: list[dict] = []
    price = start_price
    base_date = datetime(2025, 1, 1)
    for i in range(n):
        noise = rng.gauss(0, 0.015)  # 1.5% daily volatility
        close = price * (1 + daily_drift + noise)
        open_ = price
        high = max(open_, close) * (1 + abs(rng.gauss(0, 0.004)))
        low = min(open_, close) * (1 - abs(rng.gauss(0, 0.004)))
        vol = 1_000_000 * (1 + abs(rng.gauss(0, 0.2)))
        rows.append({
            "time": (base_date + timedelta(days=i)).strftime("%Y-%m-%d"),
            "open": round(open_, 2),
            "high": round(high, 2),
            "low": round(low, 2),
            "close": round(close, 2),
            "volume": round(vol, 0),
        })
        price = close
    return rows


class TestYearLongSimulation:
    """End-to-end: 1 year of sim cycles on synthetic market data.

    Generates ~500 daily candles for 4 tickers (uptrend, sideways, downtrend,
    + benchmark ETF), progressively reveals them month-by-month, and runs the
    full deterministic strategy + benchmark DCA for 12 monthly cycles.

    No network, no LLM — purely deterministic.
    """

    SYNTHETIC = {
        "BULL": _gen_synthetic_candles(100, 0.0008, 500, seed=1),    # ~+50%/yr
        "FLAT":  _gen_synthetic_candles(100, 0.0,    500, seed=2),   # sideways
        "BEAR":  _gen_synthetic_candles(100, -0.0008, 500, seed=3),  # ~-30%/yr
        "URTH":  _gen_synthetic_candles(100, 0.0004, 500, seed=4),  # mild uptrend
    }
    MONTHS = [f"2026-{m:02d}" for m in range(1, 13)]
    INITIAL_DAYS = 210   # enough for SMA-200 (MIN_CANDLES = 206)
    DAYS_PER_MONTH = 21  # ~21 trading days per month

    # -- sanity of the synthetic data itself --------------------------------

    def test_synthetic_data_sanity(self):
        """Generated data should have valid OHLCV shape and expected trends."""
        for rows in self.SYNTHETIC.values():
            assert len(rows) == 500
            assert all(r["close"] > 0 for r in rows)
            assert all(r["high"] >= r["close"] for r in rows)
            assert all(r["low"] <= r["close"] for r in rows)
            assert all(r["high"] >= r["low"] for r in rows)
            assert all(r["volume"] > 0 for r in rows)
        # BULL should end higher; BEAR should end lower
        assert self.SYNTHETIC["BULL"][-1]["close"] > self.SYNTHETIC["BULL"][0]["close"]
        assert self.SYNTHETIC["BEAR"][-1]["close"] < self.SYNTHETIC["BEAR"][0]["close"]

    # -- the main year-long simulation --------------------------------------

    async def test_one_year_deterministic(self, mem_db, monkeypatch):
        """Run 12 monthly cycles and verify benchmark DCA + bot behavior."""
        from sqlalchemy import select as sa_select

        from app.db import (
            SimBenchmarkSnapshot,
            SimSnapshot,
        )

        # --- progressive data reveal state ---
        reveal = {"day": self.INITIAL_DAYS}
        synthetic = self.SYNTHETIC

        # --- mock candles to return synthetic data up to current reveal ---
        async def mock_candles(ticker, period=None):
            data = synthetic.get(ticker, [])
            return data[: reveal["day"]] if data else []

        monkeypatch.setattr(sim, "candles", mock_candles)

        # _latest_close now reads the last bar via market.latest_close
        # (single-row query); mock it to the same progressive synthetic series.
        async def mock_latest_close(ticker):
            data = synthetic.get(ticker, [])[: reveal["day"]]
            return float(data[-1]["close"]) if data else None

        monkeypatch.setattr(sim, "latest_close", mock_latest_close)

        # --- mock refresh to no-op (no yfinance calls) ---
        async def mock_refresh(ticker, period="2y"):
            pass

        monkeypatch.setattr(sim, "refresh", mock_refresh)

        # --- mock candidate tickers (skip watchlist / universe file) ---
        async def mock_candidates():
            return ["BULL", "FLAT", "BEAR"]

        monkeypatch.setattr(sim, "_candidate_tickers", mock_candidates)

        # --- the candle cache lives in the DB now --------------------------
        # The session fill contract reads decision candles and F's bar from
        # the stored candle cache (one query), not through market.candles().
        # Seed the synthetic series into the temp DB progressively — each
        # cycle reveals one more month of stored sessions.
        seeded = {"n": 0}

        async def seed_candles():
            n = reveal["day"]
            async with mem_db() as s:
                for t, data in synthetic.items():
                    for r in data[seeded["n"]:n]:
                        s.add(Candle(
                            ticker=t,
                            timestamp=datetime.fromisoformat(r["time"]),
                            open=r["open"], high=r["high"], low=r["low"],
                            close=r["close"], volume=r["volume"]))
                await s.commit()
            seeded["n"] = n

        # --- the simulated session clock -----------------------------------
        # Each cycle stands for ONE completed session F: the last revealed
        # candle's date, at the scheduler's 23:30 UTC. The per-session fill
        # guard keys on this date, so the 12 cycles are 12 distinct sessions.
        base_date = datetime(2025, 1, 1)

        def mock_utcnow():
            d = base_date + timedelta(days=reveal["day"] - 1)
            return datetime(d.year, d.month, d.day, 23, 30, tzinfo=UTC)

        monkeypatch.setattr(sim, "_utcnow", mock_utcnow)

        # --- mock _current_month to advance through 12 months ---
        month_iter = iter(self.MONTHS)
        current_month = [next(month_iter)]

        def mock_current_month():
            return current_month[0]

        monkeypatch.setattr(sim, "_current_month", mock_current_month)

        # --- ensure deterministic strategy + our benchmark ticker ---
        monkeypatch.setattr(settings, "sim_strategy", "deterministic")
        monkeypatch.setattr(settings, "sim_benchmark_ticker", "URTH")
        monkeypatch.setattr(settings, "sim_benchmark_enabled", True)
        monkeypatch.setattr(settings, "sim_start_cash", 0.0)

        # --- run 12 monthly cycles ---
        results = []
        for i in range(12):
            if i > 0:
                current_month[0] = next(month_iter)
                reveal["day"] += self.DAYS_PER_MONTH
            await seed_candles()
            result = await sim.run_cycle()
            results.append(result)

        # === Assertions ===

        # 1. Benchmark: one DCA deposit per month (12 total)
        bench_deposits = [r for r in results if r["benchmark"].get("deposited")]
        assert len(bench_deposits) == 12, (
            f"Expected 12 benchmark deposits, got {len(bench_deposits)}"
        )

        # 2. Benchmark account: shares accumulated, cumulative deposits tracked
        bench_val = await sim.benchmark_valuate()
        assert bench_val["shares"] > 0
        assert bench_val["total_equity"] > 0
        assert bench_val["ticker"] == "URTH"

        async with mem_db() as s:
            from app.db import SimBenchmarkAccount
            bench_acc = await s.get(SimBenchmarkAccount, 1)
            assert bench_acc is not None
            assert bench_acc.shares > 0
            # cash field tracks cumulative deposits (12 × monthly allowance)
            assert bench_acc.cash == pytest.approx(
                settings.sim_monthly_allowance * 12
            )
            assert bench_acc.last_allowance_month == self.MONTHS[-1]

        # 3. Sim allowance: 12 monthly deposits recorded (ordered newest-first)
        allowances = await sim.get_allowances()
        assert len(allowances) == 12
        assert allowances[-1]["month"] == "2026-01"
        assert allowances[0]["month"] == "2026-12"
        assert all(a["amount"] == settings.sim_monthly_allowance for a in allowances)

        # 4. Snapshots: 12 sim + 12 benchmark
        async with mem_db() as s:
            sim_snaps = (await s.scalars(sa_select(SimSnapshot))).all()
            bench_snaps = (await s.scalars(sa_select(SimBenchmarkSnapshot))).all()
        assert len(sim_snaps) == 12
        assert len(bench_snaps) == 12

        # 4b. Stored created_at must be tz-aware UTC (Batch 2 convention)
        for snap in sim_snaps:
            assert snap.created_at.tzinfo is not None, \
                "SimSnapshot.created_at must be tz-aware"
            assert snap.created_at.utcoffset() == UTC.utcoffset(None), \
                "SimSnapshot.created_at must be UTC"
        for snap in bench_snaps:
            assert snap.created_at.tzinfo is not None, \
                "SimBenchmarkSnapshot.created_at must be tz-aware"
            assert snap.created_at.utcoffset() == UTC.utcoffset(None), \
                "SimBenchmarkSnapshot.created_at must be UTC"

        # 5. Bot made at least one trade over the year
        trades = await sim.get_trades(limit=500)
        assert len(trades) >= 1, "Bot should have made at least one trade in a year"

        # 6. Bot total equity is positive and allowance tracked
        val = await sim.valuate()
        assert val["total_equity"] > 0
        assert val["allowance_total"] == pytest.approx(
            settings.sim_monthly_allowance * 12
        )

        # 7. No negative cash
        assert val["cash"] >= 0

        # 8. Ledger reconciliation: every open position's shares equal the
        # net of the trade log for its ticker (the end state IS the replay
        # of the trades), and equity = cash + Σ(shares × mark). Under the
        # session fill contract a chandelier stop can legitimately flatten
        # the whole book in a choppy synthetic year, so "has a position" is
        # not an invariant — the ledger matching is.
        from collections import defaultdict
        net_shares: dict[str, float] = defaultdict(float)
        for t in trades:
            net_shares[t["ticker"]] += (t["shares"] if t["side"] == "BUY"
                                        else -t["shares"])
        held = {p["ticker"]: p["shares"] for p in val["positions"]}
        assert held == {t: s for t, s in net_shares.items() if s > 0.0001}
        assert val["total_equity"] == pytest.approx(
            val["cash"] + sum(p["shares"] * p["current_price"]
                              for p in val["positions"]), abs=0.05)

        # 9. Equity curve snapshots show non-decreasing allowance_total
        curve = await sim.get_equity_curve(limit=365)
        assert len(curve) == 12
        assert curve[-1]["allowance_total"] == pytest.approx(
            settings.sim_monthly_allowance * 12
        )


# ---------------------------------------------------------------------------
# Max-positions cap: top-ups of held tickers must not be blocked
# ---------------------------------------------------------------------------

class TestTopUpAtCap:
    """The max-positions cap must block NEW positions but still allow topping
    up tickers already held. Previously the BUY loop used 'break' when at the
    cap, which stopped all buying — including top-ups — leaving cash idle.
    The fix changed 'break' to 'continue' with a 'pos is None' guard.
    """

    async def test_top_up_proposed_for_held_ticker_at_cap(self, with_cash, monkeypatch):
        """When the portfolio is at the max-positions cap, a held ticker with
        a continuing BUY signal should still get a top-up proposal. A new
        ticker with a BUY signal should NOT get a proposal (cap blocks new)."""

        # Seed 10 positions (at the cap) at a low cost basis so they're well
        # below the max_position_pct ceiling and can be topped up.
        for i in range(10):
            await sim._exec_buy(f"HELD{i}", 50.0, 200.0, "seed")  # 4 shares @ 50

        # Mock _latest_close: held tickers at 60 (above entry, room to top up),
        # new ticker at 100.
        async def mock_close(ticker):
            if ticker.startswith("HELD"):
                return 60.0
            return 100.0  # NEW1
        monkeypatch.setattr(sim, "_latest_close", mock_close)

        # Mock signals: all held tickers + NEW1 have BUY signals; the snapshot
        # carries run_5d below the block threshold so BUYs aren't filtered.
        def buy_signal(ticker):
            return {
                "action": "BUY",
                "reason": "BUY (test)",
                "strength": 70,
                "snapshot": {"run_5d": 0.0, "atr_stop": None, "close": 60.0 if ticker.startswith("HELD") else 100.0},
            }
        signals = {f"HELD{i}": buy_signal(f"HELD{i}") for i in range(10)}
        signals["NEW1"] = buy_signal("NEW1")

        # Keep max_positions at the default (10) — the portfolio is at cap.
        monkeypatch.setattr(settings, "sim_max_positions", 10)

        # Valuation with enough cash for top-ups (monthly allowance just landed).
        acc = await sim._account()
        acc.cash = 5000.0
        async with sim.Session() as s:
            await s.commit()
        valuation = await sim.valuate()

        # Contract path: frozen decision inputs — decision closes (S) for
        # held + candidate tickers, F's bars for the held tickers (fills at
        # F's open), no ratchets yet.
        prices = {f"HELD{i}": 60.0 for i in range(10)}
        prices["NEW1"] = 100.0
        session_bars = {f"HELD{i}": {"open": 60.0, "low": 59.0, "close": 60.0}
                        for i in range(10)}
        decision = sim._DecisionInputs(
            f_ts=pd.Timestamp("2026-03-03"), s_ts=pd.Timestamp("2026-03-02"),
            signals=signals, prices=prices, session_bars=session_bars,
            ratchets={f"HELD{i}": None for i in range(10)})
        proposals = await sim._deterministic_propose(valuation, decision=decision)

        # Split proposals by whether the ticker is already held.
        held_tickers = {f"HELD{i}" for i in range(10)}
        top_ups = [p for p in proposals if p["side"] == "BUY" and p["ticker"] in held_tickers]
        new_buys = [p for p in proposals if p["side"] == "BUY" and p["ticker"] not in held_tickers]

        # Top-ups should be proposed for held tickers (the fix).
        assert len(top_ups) > 0, (
            f"no top-up proposals for held tickers (the cap blocked them); "
            f"proposals: {[(p['ticker'], p['side']) for p in proposals]}"
        )

        # NEW1 must NOT be proposed — the cap blocks new positions.
        assert len(new_buys) == 0, (
            f"new ticker proposed at cap: {[(p['ticker'], p['side']) for p in new_buys]}"
        )

    async def test_top_up_executes_and_deploys_cash(self, with_cash, monkeypatch):
        """End-to-end: propose + execute at the cap should actually buy more of
        a held ticker and reduce cash. Calls _deterministic_propose (with a
        pre-built _DecisionInputs) + _execute_proposed directly, mirroring
        the hybrid branch's propose→execute flow."""
        from app.db import SimPosition

        # Seed 10 positions at low cost basis.
        for i in range(10):
            await sim._exec_buy(f"HELD{i}", 50.0, 200.0, "seed")  # 4 shares @ 50

        # Held tickers at 60 (room to top up).
        async def mock_close(ticker):
            return 60.0 if ticker.startswith("HELD") else 100.0
        monkeypatch.setattr(sim, "_latest_close", mock_close)

        def buy_signal(ticker):
            return {
                "action": "BUY", "reason": "BUY (test)", "strength": 70,
                "snapshot": {"run_5d": 0.0, "atr_stop": None, "close": 60.0},
            }
        signals = {f"HELD{i}": buy_signal(f"HELD{i}") for i in range(10)}

        monkeypatch.setattr(settings, "sim_max_positions", 10)

        # Give excess cash so top-ups can deploy it.
        acc = await sim._account()
        acc.cash = 5000.0
        async with sim.Session() as s:
            await s.commit()
        cash_before = acc.cash

        valuation = await sim.valuate()
        # Propose on the contract path with pre-built frozen decision inputs,
        # then execute — same flow as the hybrid branch.
        prices = {f"HELD{i}": 60.0 for i in range(10)}
        session_bars = {f"HELD{i}": {"open": 60.0, "low": 59.0, "close": 60.0}
                        for i in range(10)}
        decision = sim._DecisionInputs(
            f_ts=pd.Timestamp("2026-03-03"), s_ts=pd.Timestamp("2026-03-02"),
            signals=signals, prices=prices, session_bars=session_bars,
            ratchets={f"HELD{i}": None for i in range(10)})
        proposals = await sim._deterministic_propose(valuation, decision=decision)
        trades = await sim._execute_proposed(proposals)

        # BUY trades should exist (top-ups of held tickers).
        buys = [t for t in trades if t["side"] == "BUY"]
        assert len(buys) > 0, "no top-up BUYs executed at cap"

        # Cash should have decreased (deployed into top-ups).
        acc_after = await sim._account()
        assert acc_after.cash < cash_before, (
            f"cash didn't decrease: {cash_before} → {acc_after.cash} (top-ups didn't deploy cash)"
        )

        # Position count should still be 10 (top-ups don't open new positions).
        async with sim.Session() as s:
            from sqlalchemy import select as sa_select
            positions = (await s.scalars(sa_select(SimPosition))).all()
        assert len(positions) == 10, (
            f"position count {len(positions)} != 10 (top-ups opened new positions)"
        )


# ---------------------------------------------------------------------------
# Deterministic engine: entry gate + ATR stop (Phase 1 financial correctness)
# ---------------------------------------------------------------------------

class TestDeterministicEntryAndStops:
    """The live sim must not buy a bearish HOLD through the relaxed fallback,
    and the ATR stop must be a level the price can actually break."""

    def _hold_signal(self, *, net: int, trend_up: bool, dist_above: float,
                     weekly: bool, strength: int = 60,
                     atr_stop: float | None = None, close: float = 100.0) -> dict:
        return {
            "action": "HOLD",
            "strength": strength,
            "reason": f"HOLD (score {net:+d})",
            "snapshot": {
                "close": close,
                "atr14": 2.0,
                "atr_stop": atr_stop,
                "run_5d": 0.0,
                "dist_above": dist_above,
                "trend_up": trend_up,
                "trend_down": not trend_up,
                "weekly_trend_up": weekly,
                "net_score": net,
            },
        }

    async def test_bearish_hold_is_not_bought(self, with_cash, monkeypatch):
        """The audit reproduction: bullish 0 / bearish 65 with a weekly
        downtrend produced a BUY. It must not any more."""
        sig = self._hold_signal(net=-65, trend_up=False, dist_above=-8.0,
                                 weekly=False, strength=65)

        valuation = await sim.valuate()
        decision = sim._DecisionInputs(
            f_ts=pd.Timestamp("2026-03-03"), s_ts=pd.Timestamp("2026-03-02"),
            signals={"BEAR": sig}, prices={"BEAR": 100.0}, session_bars={},
            ratchets={})
        proposals = await sim._deterministic_propose(valuation, decision=decision)
        assert [p for p in proposals if p["side"] == "BUY"] == []

    async def test_near_threshold_hold_is_bought_when_eligible(
        self, with_cash, monkeypatch
    ):
        """A HOLD that misses only the score threshold still enters."""
        sig = self._hold_signal(net=25, trend_up=True, dist_above=5.0,
                                 weekly=True, strength=45)

        valuation = await sim.valuate()
        decision = sim._DecisionInputs(
            f_ts=pd.Timestamp("2026-03-03"), s_ts=pd.Timestamp("2026-03-02"),
            signals={"GOOD": sig}, prices={"GOOD": 100.0},
            session_bars={"GOOD": {"open": 100.0, "low": 99.0, "close": 100.0}},
            ratchets={})
        proposals = await sim._deterministic_propose(valuation, decision=decision)
        buys = [p for p in proposals if p["side"] == "BUY"]
        assert len(buys) == 1
        assert buys[0]["ticker"] == "GOOD"
        assert buys[0]["price"] == 100.0  # fills at F's open (contract)
        assert "relaxed HOLD entry" in buys[0]["reason"]

    async def test_atr_stop_sells_a_broken_position(self, with_cash, monkeypatch):
        """A price below the snapshot's ATR stop must produce a SELL."""
        await sim._exec_buy("AAA", 100.0, 1_000.0, "seed")
        sig = self._hold_signal(net=5, trend_up=True, dist_above=5.0,
                                 weekly=True, strength=10, atr_stop=95.0,
                                 close=90.0)

        # valuate() marks the held position at the latest close.
        async def mock_close(ticker):
            return 90.0
        monkeypatch.setattr(sim, "_latest_close", mock_close)

        valuation = await sim.valuate()
        # Contract path: the ATR level is a standing order for F — F's open
        # 90 <= level 95, so the stop fills AT THE OPEN (not at a stale close).
        decision = sim._DecisionInputs(
            f_ts=pd.Timestamp("2026-03-03"), s_ts=pd.Timestamp("2026-03-02"),
            signals={"AAA": sig}, prices={"AAA": 90.0},
            session_bars={"AAA": {"open": 90.0, "low": 89.0, "close": 90.0}},
            ratchets={"AAA": None})
        proposals = await sim._deterministic_propose(valuation, decision=decision)
        sells = [p for p in proposals if p["side"] == "SELL"]
        assert len(sells) == 1
        assert sells[0]["price"] == 90.0
        assert sells[0]["reason"].startswith("ATR stop standing order")

    async def test_no_buy_below_the_published_atr_level(self, with_cash, monkeypatch):
        """The ATR level is anchored to the market's 20-bar high, so a pullback
        entry can sit below it. Entering there would be stopped out on the
        next cycle (buy -> sell -> buy churn at the same price)."""
        sig = {
            "action": "BUY",
            "strength": 80,
            "reason": "BUY (score +60)",
            "snapshot": {
                "close": 128.46,
                "atr14": 2.0,
                "atr_stop": 128.97,   # above the entry price
                "run_5d": 0.0,
                "dist_above": 8.0,
                "trend_up": True,
                "trend_down": False,
                "weekly_trend_up": True,
                "net_score": 60,
            },
        }

        valuation = await sim.valuate()
        decision = sim._DecisionInputs(
            f_ts=pd.Timestamp("2026-03-03"), s_ts=pd.Timestamp("2026-03-02"),
            signals={"AAA": sig}, prices={"AAA": 128.46},
            session_bars={"AAA": {"open": 128.46, "low": 128.0, "close": 128.46}},
            ratchets={})
        proposals = await sim._deterministic_propose(valuation, decision=decision)
        assert [p for p in proposals if p["side"] == "BUY"] == []

    async def test_no_rebuy_of_a_ticker_sold_in_the_same_cycle(
        self, with_cash, monkeypatch
    ):
        """A ticker stopped out in a cycle must not be bought again in it."""
        await sim._exec_buy("AAA", 100.0, 1_000.0, "seed")
        sig = {
            "action": "BUY",
            "strength": 80,
            "reason": "BUY (score +60)",
            "snapshot": {
                "close": 85.0,
                "atr14": 2.0,
                "atr_stop": 90.0,     # price below the level -> SELL
                "run_5d": 0.0,
                "dist_above": 8.0,
                "trend_up": True,
                "trend_down": False,
                "weekly_trend_up": True,
                "net_score": 60,
            },
        }

        # valuate() marks the held position at the latest close.
        async def mock_close(ticker):
            return 85.0
        monkeypatch.setattr(sim, "_latest_close", mock_close)

        valuation = await sim.valuate()
        decision = sim._DecisionInputs(
            f_ts=pd.Timestamp("2026-03-03"), s_ts=pd.Timestamp("2026-03-02"),
            signals={"AAA": sig}, prices={"AAA": 85.0},
            session_bars={"AAA": {"open": 85.0, "low": 84.0, "close": 85.0}},
            ratchets={"AAA": None})
        proposals = await sim._deterministic_propose(valuation, decision=decision)
        sides = [(p["ticker"], p["side"]) for p in proposals]
        assert ("AAA", "SELL") in sides
        assert ("AAA", "BUY") not in sides, (
            f"the ticker was sold and bought again in the same cycle: {sides}"
        )


class TestTimezoneConvention:
    """Verify that sim timestamps are stored as tz-aware UTC, and the monthly
    allowance deposit is anchored to Europe/Vienna (a calendar concept)."""

    async def test_utcnow_is_tz_aware(self):
        """sim._utcnow() must return a tz-aware UTC datetime."""
        t = sim._utcnow()
        assert t.tzinfo is not None
        assert t.utcoffset() == UTC.utcoffset(None)

    async def test_simtrade_created_at_is_tz_aware_utc(self, with_cash):
        """A logged trade's created_at must be tz-aware UTC after a buy."""
        await sim._exec_buy("AAPL", 100.0, 1000.0, "tz test")
        async with sim.Session() as s:
            from sqlalchemy import select as sa_select
            trade = await s.scalar(sa_select(SimTrade).order_by(SimTrade.created_at.desc()))
            assert trade is not None
            assert trade.created_at.tzinfo is not None
            assert trade.created_at.utcoffset() == UTC.utcoffset(None)

    async def test_current_month_is_vienna_local(self, monkeypatch):
        """_current_month() must return a YYYY-MM string (Vienna calendar).

        This is the one intentional exception to the UTC-everywhere rule:
        the monthly allowance deposit is a calendar-month concept, so it
        must follow the operator's local timezone, not UTC.

        We verify the boundary: 23:30 UTC on 2026-01-31 is 00:30 on
        2026-02-01 in Vienna, so _current_month() must return '2026-02'.
        """
        from datetime import datetime
        from zoneinfo import ZoneInfo

        vienna = ZoneInfo("Europe/Vienna")
        fake_utc = datetime(2026, 1, 31, 23, 30, tzinfo=UTC)
        assert fake_utc.astimezone(vienna).strftime("%Y-%m") == "2026-02"

        # Freeze sim._TZ's "now" by patching datetime.now globally for the call.
        # _current_month calls datetime.now(_TZ); we make _TZ a fake zone where
        # "now" returns our fixed Vienna instant.
        import app.sim as sim_mod
        real_datetime = sim_mod.datetime

        class FakeDateTime(real_datetime):
            @classmethod
            def now(cls, tz=None):
                if tz is not None:
                    return fake_utc.astimezone(tz)
                return fake_utc.replace(tzinfo=None)

        monkeypatch.setattr(sim_mod, "datetime", FakeDateTime)
        month = sim._current_month()
        assert month == "2026-02", \
            f"Expected Vienna-local month '2026-02' at the UTC/Vienna boundary, got '{month}'"

    def test_week_diff(self):
        """_week_diff computes calendar-week distance, across year boundaries."""
        # Same week
        assert sim._week_diff("2026-W03", "2026-W03") == 0
        # One week later
        assert sim._week_diff("2026-W04", "2026-W03") == 1
        # Ten weeks later
        assert sim._week_diff("2026-W13", "2026-W03") == 10
        # Year boundary: week 50 of 2025 -> week 2 of 2026 = 4 weeks
        # (2025-W50 Thu is Dec 11; 2026-W02 Thu is Jan 8; 28 days apart).
        assert sim._week_diff("2026-W02", "2025-W50") == 4
        # Never reviewed: treat None specially in the caller, not here.
        assert sim._week_diff("2026-W01", "2026-W01") == 0


# ---------------------------------------------------------------------------
# Action block parsing (chat-driven trades)
# ---------------------------------------------------------------------------

class TestParseActionBlock:
    """Tests for _parse_action_block — extracts [[ACTION]] JSON from LLM text."""

    def test_no_action_block(self):
        """Text without any [[ACTION]] block should return None."""
        assert sim._parse_action_block("Just a normal chat response.") is None

    def test_empty_text(self):
        assert sim._parse_action_block("") is None
        assert sim._parse_action_block(None) is None

    def test_basic_buy_sell(self):
        """Basic action block with BUY and SELL, no optional size fields."""
        text = (
            'I think you should trim MDB and buy SPY.\n'
            '[[ACTION]]\n'
            '{"actions": [\n'
            '  {"ticker": "MDB", "action": "SELL", "reason": "trim overweight"},\n'
            '  {"ticker": "SPY", "action": "BUY", "reason": "diversify"}\n'
            ']}\n'
            '[[/ACTION]]'
        )
        result = sim._parse_action_block(text)
        assert result is not None
        assert len(result) == 2
        assert result[0]["ticker"] == "MDB"
        assert result[0]["action"] == "SELL"
        assert result[0]["reason"] == "trim overweight"
        assert result[1]["ticker"] == "SPY"
        assert result[1]["action"] == "BUY"
        # No size fields present
        assert "shares" not in result[0]
        assert "amount" not in result[0]
        assert "shares" not in result[1]
        assert "amount" not in result[1]

    def test_partial_sell_with_amount(self):
        """Action block with 'amount' field should be extracted."""
        text = (
            '[[ACTION]]\n'
            '{"actions": [{"ticker": "MDB", "action": "SELL", "amount": 67.43, "reason": "trim"}]}\n'
            '[[/ACTION]]'
        )
        result = sim._parse_action_block(text)
        assert result is not None
        assert len(result) == 1
        assert result[0]["ticker"] == "MDB"
        assert result[0]["action"] == "SELL"
        assert result[0]["amount"] == pytest.approx(67.43)
        assert "shares" not in result[0]

    def test_partial_sell_with_shares(self):
        """Action block with 'shares' field should be extracted."""
        text = (
            '[[ACTION]]\n'
            '{"actions": [{"ticker": "ANET", "action": "SELL", "shares": 3.5, "reason": "trim"}]}\n'
            '[[/ACTION]]'
        )
        result = sim._parse_action_block(text)
        assert result is not None
        assert result[0]["shares"] == pytest.approx(3.5)
        assert "amount" not in result[0]

    def test_partial_buy_with_amount(self):
        """BUY action with 'amount' field should be extracted."""
        text = (
            '[[ACTION]]\n'
            '{"actions": [{"ticker": "SPY", "action": "BUY", "amount": 500.0, "reason": "diversify"}]}\n'
            '[[/ACTION]]'
        )
        result = sim._parse_action_block(text)
        assert result is not None
        assert result[0]["amount"] == pytest.approx(500.0)

    def test_partial_buy_with_shares(self):
        """BUY action with 'shares' field should be extracted."""
        text = (
            '[[ACTION]]\n'
            '{"actions": [{"ticker": "SPY", "action": "BUY", "shares": 2.0, "reason": "diversify"}]}\n'
            '[[/ACTION]]'
        )
        result = sim._parse_action_block(text)
        assert result is not None
        assert result[0]["shares"] == pytest.approx(2.0)

    def test_both_shares_and_amount(self):
        """If both 'shares' and 'amount' are given, both should be extracted."""
        text = (
            '[[ACTION]]\n'
            '{"actions": [{"ticker": "X", "action": "SELL", "shares": 3, "amount": 100, "reason": "both"}]}\n'
            '[[/ACTION]]'
        )
        result = sim._parse_action_block(text)
        assert result is not None
        assert result[0]["shares"] == pytest.approx(3.0)
        assert result[0]["amount"] == pytest.approx(100.0)

    def test_negative_amount_ignored(self):
        """Negative or zero amounts should be silently dropped (not included)."""
        text = (
            '[[ACTION]]\n'
            '{"actions": [{"ticker": "X", "action": "SELL", "amount": -50, "reason": "neg"}]}\n'
            '[[/ACTION]]'
        )
        result = sim._parse_action_block(text)
        assert result is not None
        assert "amount" not in result[0]
        assert "shares" not in result[0]

    def test_zero_shares_ignored(self):
        """Zero shares should be silently dropped."""
        text = (
            '[[ACTION]]\n'
            '{"actions": [{"ticker": "X", "action": "SELL", "shares": 0, "reason": "zero"}]}\n'
            '[[/ACTION]]'
        )
        result = sim._parse_action_block(text)
        assert result is not None
        assert "shares" not in result[0]

    def test_bool_amount_ignored(self):
        """Boolean values (True/False) should not be treated as numbers."""
        text = (
            '[[ACTION]]\n'
            '{"actions": [{"ticker": "X", "action": "SELL", "amount": true, "reason": "bool"}]}\n'
            '[[/ACTION]]'
        )
        result = sim._parse_action_block(text)
        assert result is not None
        assert "amount" not in result[0]

    def test_invalid_json(self):
        """Malformed JSON inside the action block should return None."""
        text = '[[ACTION]]\nnot valid json\n[[/ACTION]]'
        assert sim._parse_action_block(text) is None

    def test_missing_end_tag(self):
        """Missing [[/ACTION]] closing tag should return None."""
        text = '[[ACTION]]\n{"actions": []}'
        assert sim._parse_action_block(text) is None

    def test_empty_actions_list(self):
        """An empty actions list should return None (no valid actions)."""
        text = '[[ACTION]]\n{"actions": []}\n[[/ACTION]]'
        assert sim._parse_action_block(text) is None

    def test_ticker_uppercased(self):
        """Ticker should be uppercased and stripped."""
        text = (
            '[[ACTION]]\n'
            '{"actions": [{"ticker": " aapl ", "action": "buy", "reason": "x"}]}\n'
            '[[/ACTION]]'
        )
        result = sim._parse_action_block(text)
        assert result is not None
        assert result[0]["ticker"] == "AAPL"

    def test_invalid_action_filtered(self):
        """Actions with invalid 'action' values should be filtered out."""
        text = (
            '[[ACTION]]\n'
            '{"actions": [\n'
            '  {"ticker": "A", "action": "HODL", "reason": "typo"},\n'
            '  {"ticker": "B", "action": "BUY", "reason": "valid"}\n'
            ']}\n'
            '[[/ACTION]]'
        )
        result = sim._parse_action_block(text)
        assert result is not None
        assert len(result) == 1
        assert result[0]["ticker"] == "B"

    def test_string_amount_ignored(self):
        """String values for 'amount' should be ignored (not coerced)."""
        text = (
            '[[ACTION]]\n'
            '{"actions": [{"ticker": "X", "action": "SELL", "amount": "100", "reason": "str"}]}\n'
            '[[/ACTION]]'
        )
        result = sim._parse_action_block(text)
        assert result is not None
        assert "amount" not in result[0]

    def test_integer_amount_accepted(self):
        """Integer values (not bool) should be accepted for amount."""
        text = (
            '[[ACTION]]\n'
            '{"actions": [{"ticker": "X", "action": "SELL", "amount": 100, "reason": "int"}]}\n'
            '[[/ACTION]]'
        )
        result = sim._parse_action_block(text)
        assert result is not None
        assert result[0]["amount"] == pytest.approx(100.0)

    def test_use_reserve_flag_parsed(self):
        """use_reserve: true should be passed through on a BUY action."""
        text = (
            '[[ACTION]]\n'
            '{"actions": [{"ticker": "AAPL", "action": "BUY", "use_reserve": true, "reason": "dry powder"}]}\n'
            '[[/ACTION]]'
        )
        result = sim._parse_action_block(text)
        assert result is not None
        assert result[0]["use_reserve"] is True

    def test_use_reserve_false_not_set(self):
        """use_reserve absent or false should not add the flag."""
        text = (
            '[[ACTION]]\n'
            '{"actions": [{"ticker": "AAPL", "action": "BUY", "reason": "normal buy"}]}\n'
            '[[/ACTION]]'
        )
        result = sim._parse_action_block(text)
        assert result is not None
        assert "use_reserve" not in result[0]


# ---------------------------------------------------------------------------
# Chat proposals (sim_chat / sim_chat_stream PROPOSE, confirm executes)
# ---------------------------------------------------------------------------

def _stub_chat_reply(monkeypatch, text: str) -> None:
    """Fake the LLM HTTP reply for a sim chat turn (no network)."""

    class MockResponse:
        status_code = 200

        def raise_for_status(self):
            pass

        def json(self):
            return {"message": {"content": text}}

    class MockClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def post(self, url, json=None):
            return MockResponse()

    monkeypatch.setattr("app.llm.httpx.AsyncClient", lambda **kw: MockClient())


def _mock_chat_context(monkeypatch) -> None:
    """Keep the sim chat context minimal (no signals/news gathering)."""

    async def mock_context():
        return "mock context"

    monkeypatch.setattr(sim, "_build_sim_chat_context", mock_context)


async def _trade_count() -> int:
    async with sim.Session() as s:
        return await s.scalar(select(func.count()).select_from(SimTrade)) or 0


async def _chat_then_confirm(result, trades_before: int) -> dict:
    """The slice-3D contract: a chat turn only PROPOSES (honest envelope — no
    "trades"/"actions_executed" keys, no new trade rows), and the returned
    proposal is what confirm_chat_proposal executes."""
    assert "trades" not in result
    assert "actions_executed" not in result
    assert len(result["proposals"]) == 1
    assert await _trade_count() == trades_before, "a proposing turn must not write trades"
    return await sim.confirm_chat_proposal(result["proposals"][0]["proposal_id"])


class TestSimChatPartialSell:
    """Tests that a chat turn PROPOSES partial SELL sizes (with an accurate
    preview) and that confirming the proposal executes exactly those sizes."""

    async def test_sell_with_amount_trims_position(self, with_cash, monkeypatch, llm_backend):
        """SELL with 'amount' should sell only the equivalent shares, not all."""
        # Setup: buy 100 shares of AAPL at $100 = $10,000
        await sim._exec_buy("AAPL", 100.0, 10000.0, "initial buy")
        trades_before = await _trade_count()

        async def mock_close(ticker):
            return 100.0 if ticker == "AAPL" else None
        monkeypatch.setattr(sim, "_latest_close", mock_close)
        _mock_chat_context(monkeypatch)

        _stub_chat_reply(monkeypatch, (
            'Trimming AAPL by $3000.\n'
            '[[ACTION]]\n'
            '{"actions": [{"ticker": "AAPL", "action": "SELL", "amount": 3000.0, "reason": "trim"}]}\n'
            '[[/ACTION]]'
        ))

        result = await sim.sim_chat([{"role": "user", "content": "trim AAPL by $3000"}])

        # The turn proposed, not executed: the preview estimates 30 shares / $3000.
        action = result["proposals"][0]["actions"][0]
        assert action["ticker"] == "AAPL" and action["action"] == "SELL"
        assert action["amount"] == 3000.0 and action["reason"] == "trim"
        preview = action["preview"]
        assert preview["feasible"] is True
        assert preview["issue"] is None
        assert preview["price"] == 100.0
        assert preview["est_shares"] == pytest.approx(30.0, abs=0.001)
        assert preview["est_value"] == pytest.approx(3000.0)
        assert preview["est_cash_after"] == pytest.approx(3000.0)  # cash was 0 after the buy
        assert preview["est_position_value_after"] == pytest.approx(7000.0)

        conf = await _chat_then_confirm(result, trades_before)
        assert conf["ok"] is True
        assert conf["snapshot_taken"] is True
        assert conf["skipped"] == []
        assert len(conf["trades"]) == 1
        trade = conf["trades"][0]
        assert trade["ticker"] == "AAPL"
        assert trade["side"] == "SELL"
        # $3000 / $100 = 30 shares sold — exactly what the preview estimated
        assert trade["shares"] == pytest.approx(30.0, abs=0.001)

        # Verify the position still exists with ~70 shares remaining
        async with sim.Session() as s:
            pos = await s.scalar(select(SimPosition).where(SimPosition.ticker == "AAPL"))
            assert pos is not None, "Position should still exist after partial sell"
            assert pos.shares == pytest.approx(70.0, abs=0.001)

    async def test_sell_with_shares_trims_position(self, with_cash, monkeypatch, llm_backend):
        """SELL with 'shares' should sell exactly that many shares."""
        await sim._exec_buy("AAPL", 100.0, 10000.0, "initial buy")
        trades_before = await _trade_count()

        async def mock_close(ticker):
            return 100.0 if ticker == "AAPL" else None
        monkeypatch.setattr(sim, "_latest_close", mock_close)
        _mock_chat_context(monkeypatch)

        _stub_chat_reply(monkeypatch, (
            'Selling 25 shares.\n'
            '[[ACTION]]\n'
            '{"actions": [{"ticker": "AAPL", "action": "SELL", "shares": 25.0, "reason": "trim"}]}\n'
            '[[/ACTION]]'
        ))

        result = await sim.sim_chat([{"role": "user", "content": "sell 25 shares of AAPL"}])

        preview = result["proposals"][0]["actions"][0]["preview"]
        assert preview["feasible"] is True
        assert preview["est_shares"] == pytest.approx(25.0, abs=0.001)
        assert preview["est_value"] == pytest.approx(2500.0)

        conf = await _chat_then_confirm(result, trades_before)
        assert len(conf["trades"]) == 1
        assert conf["trades"][0]["shares"] == pytest.approx(25.0, abs=0.001)

        async with sim.Session() as s:
            pos = await s.scalar(select(SimPosition).where(SimPosition.ticker == "AAPL"))
            assert pos is not None
            assert pos.shares == pytest.approx(75.0, abs=0.001)

    async def test_sell_without_size_liquidates_all(self, with_cash, monkeypatch, llm_backend):
        """SELL without 'shares' or 'amount' should sell the entire position (backward compat)."""
        await sim._exec_buy("AAPL", 100.0, 5000.0, "initial buy")  # 50 shares
        trades_before = await _trade_count()

        async def mock_close(ticker):
            return 100.0 if ticker == "AAPL" else None
        monkeypatch.setattr(sim, "_latest_close", mock_close)
        _mock_chat_context(monkeypatch)

        _stub_chat_reply(monkeypatch, (
            'Selling all AAPL.\n'
            '[[ACTION]]\n'
            '{"actions": [{"ticker": "AAPL", "action": "SELL", "reason": "exit"}]}\n'
            '[[/ACTION]]'
        ))

        result = await sim.sim_chat([{"role": "user", "content": "sell all AAPL"}])

        preview = result["proposals"][0]["actions"][0]["preview"]
        assert preview["feasible"] is True
        assert preview["est_shares"] == pytest.approx(50.0, abs=0.001)
        assert preview["est_position_value_after"] == pytest.approx(0.0)

        conf = await _chat_then_confirm(result, trades_before)
        assert len(conf["trades"]) == 1
        assert conf["trades"][0]["shares"] == pytest.approx(50.0, abs=0.001)

        async with sim.Session() as s:
            pos = await s.scalar(select(SimPosition).where(SimPosition.ticker == "AAPL"))
            assert pos is None, "Position should be fully liquidated"

    async def test_sell_amount_exceeding_position_sells_all(self, with_cash, monkeypatch, llm_backend):
        """If 'amount' exceeds the position value, sell the entire position."""
        await sim._exec_buy("AAPL", 100.0, 3000.0, "initial buy")  # 30 shares = $3000
        trades_before = await _trade_count()

        async def mock_close(ticker):
            return 100.0 if ticker == "AAPL" else None
        monkeypatch.setattr(sim, "_latest_close", mock_close)
        _mock_chat_context(monkeypatch)

        _stub_chat_reply(monkeypatch, (
            '[[ACTION]]\n'
            '{"actions": [{"ticker": "AAPL", "action": "SELL", "amount": 99999.0, "reason": "over"}]}\n'
            '[[/ACTION]]'
        ))

        result = await sim.sim_chat([{"role": "user", "content": "sell AAPL"}])

        # The preview clamps to the position: 30 shares, not 999.99.
        preview = result["proposals"][0]["actions"][0]["preview"]
        assert preview["feasible"] is True
        assert preview["est_shares"] == pytest.approx(30.0, abs=0.001)

        conf = await _chat_then_confirm(result, trades_before)
        assert len(conf["trades"]) == 1
        # _exec_sell clamps to pos.shares
        assert conf["trades"][0]["shares"] == pytest.approx(30.0, abs=0.001)


class TestSimChatPartialBuy:
    """Tests that a chat turn PROPOSES partial BUY sizes and that confirming
    the proposal executes the previewed budget."""

    async def test_buy_with_amount(self, with_cash, monkeypatch, llm_backend):
        """BUY with 'amount' should invest at most that many dollars."""
        # Raise max position % so the $2000 amount isn't clamped to $1500
        monkeypatch.setattr(settings, "sim_max_position_pct", 30.0)

        async def mock_close(ticker):
            return 100.0 if ticker == "AAPL" else None
        monkeypatch.setattr(sim, "_latest_close", mock_close)
        _mock_chat_context(monkeypatch)

        _stub_chat_reply(monkeypatch, (
            'Buying $2000 of AAPL.\n'
            '[[ACTION]]\n'
            '{"actions": [{"ticker": "AAPL", "action": "BUY", "amount": 2000.0, "reason": "dip"}]}\n'
            '[[/ACTION]]'
        ))

        result = await sim.sim_chat([{"role": "user", "content": "buy $2000 of AAPL"}])
        trades_before = await _trade_count()

        # cash 10000, min_cash 500, max_pos 3000 → budget = min(9500, 3000, 2000) = 2000
        preview = result["proposals"][0]["actions"][0]["preview"]
        assert preview["feasible"] is True
        assert preview["issue"] is None
        assert preview["est_shares"] == pytest.approx(20.0, abs=0.001)
        assert preview["est_value"] == pytest.approx(2000.0)
        assert preview["est_cash_after"] == pytest.approx(8000.0)
        assert preview["est_position_value_after"] == pytest.approx(2000.0)

        conf = await _chat_then_confirm(result, trades_before)
        assert len(conf["trades"]) == 1
        trade = conf["trades"][0]
        assert trade["ticker"] == "AAPL"
        assert trade["side"] == "BUY"
        # $2000 / $100 = 20 shares
        assert trade["shares"] == pytest.approx(20.0, abs=0.001)

    async def test_buy_with_shares(self, with_cash, monkeypatch, llm_backend):
        """BUY with 'shares' should buy at most that many shares (clamped to budget)."""
        # Raise max position % so the requested 15 shares aren't cash/max-clamped.
        monkeypatch.setattr(settings, "sim_max_position_pct", 30.0)

        async def mock_close(ticker):
            return 100.0 if ticker == "AAPL" else None
        monkeypatch.setattr(sim, "_latest_close", mock_close)
        _mock_chat_context(monkeypatch)

        _stub_chat_reply(monkeypatch, (
            '[[ACTION]]\n'
            '{"actions": [{"ticker": "AAPL", "action": "BUY", "shares": 15.0, "reason": "dip"}]}\n'
            '[[/ACTION]]'
        ))

        result = await sim.sim_chat([{"role": "user", "content": "buy 15 shares of AAPL"}])
        trades_before = await _trade_count()

        # 15 shares * $100 = $1500 budget → 15 shares bought
        preview = result["proposals"][0]["actions"][0]["preview"]
        assert preview["feasible"] is True
        assert preview["est_shares"] == pytest.approx(15.0, abs=0.001)
        assert preview["est_value"] == pytest.approx(1500.0)

        conf = await _chat_then_confirm(result, trades_before)
        assert len(conf["trades"]) == 1
        assert conf["trades"][0]["shares"] == pytest.approx(15.0, abs=0.001)

    async def test_buy_amount_clamped_to_max_budget(self, with_cash, monkeypatch, llm_backend):
        """BUY 'amount' exceeding the risk-limited budget should be clamped (feasible preview)."""
        # Cash = $10,000, min_cash = 5% = $500, so max spend = $9,500
        # max_position = 10% of $10,000 = $1,000
        # Requesting $5,000 should be clamped to $1,000
        async def mock_close(ticker):
            return 100.0 if ticker == "AAPL" else None
        monkeypatch.setattr(sim, "_latest_close", mock_close)
        _mock_chat_context(monkeypatch)

        _stub_chat_reply(monkeypatch, (
            '[[ACTION]]\n'
            '{"actions": [{"ticker": "AAPL", "action": "BUY", "amount": 5000.0, "reason": "too much"}]}\n'
            '[[/ACTION]]'
        ))

        result = await sim.sim_chat([{"role": "user", "content": "buy $5000 of AAPL"}])
        trades_before = await _trade_count()

        preview = result["proposals"][0]["actions"][0]["preview"]
        assert preview["feasible"] is True
        assert preview["est_value"] == pytest.approx(1000.0)  # clamped to max position
        assert preview["est_shares"] == pytest.approx(10.0, abs=0.001)

        conf = await _chat_then_confirm(result, trades_before)
        assert len(conf["trades"]) == 1
        # max_position = 10% of $10,000 = $1,000 → 10 shares at $100
        assert conf["trades"][0]["shares"] == pytest.approx(10.0, abs=0.001)

    async def test_buy_without_amount_uses_max_budget(self, with_cash, monkeypatch, llm_backend):
        """BUY without 'amount' or 'shares' should use the max allowed budget."""
        async def mock_close(ticker):
            return 100.0 if ticker == "AAPL" else None
        monkeypatch.setattr(sim, "_latest_close", mock_close)
        _mock_chat_context(monkeypatch)

        _stub_chat_reply(monkeypatch, (
            '[[ACTION]]\n'
            '{"actions": [{"ticker": "AAPL", "action": "BUY", "reason": "dip"}]}\n'
            '[[/ACTION]]'
        ))

        result = await sim.sim_chat([{"role": "user", "content": "buy AAPL"}])
        trades_before = await _trade_count()

        preview = result["proposals"][0]["actions"][0]["preview"]
        assert preview["feasible"] is True
        assert preview["est_value"] == pytest.approx(1000.0)

        conf = await _chat_then_confirm(result, trades_before)
        assert len(conf["trades"]) == 1
        # max_position = 10% of $10,000 = $1,000 → 10 shares at $100
        assert conf["trades"][0]["shares"] == pytest.approx(10.0, abs=0.001)


class TestSimChatReserveOverride:
    """Tests that a confirmed BUY can spend the cash reserve only with use_reserve."""

    async def test_buy_without_reserve_is_clamped(self, with_cash, monkeypatch, llm_backend):
        """A normal BUY (no use_reserve) cannot spend below the 5% cash floor."""
        # Raise max position so the cash floor is the only clamp in play.
        monkeypatch.setattr(settings, "sim_max_position_pct", 100.0)

        async def mock_close(ticker):
            return 100.0 if ticker == "AAPL" else None
        monkeypatch.setattr(sim, "_latest_close", mock_close)
        _mock_chat_context(monkeypatch)

        _stub_chat_reply(monkeypatch, (
            '[[ACTION]]\n'
            '{"actions": [{"ticker": "AAPL", "action": "BUY", "reason": "dip"}]}\n'
            '[[/ACTION]]'
        ))

        result = await sim.sim_chat([{"role": "user", "content": "buy AAPL"}])
        trades_before = await _trade_count()

        preview = result["proposals"][0]["actions"][0]["preview"]
        assert preview["feasible"] is True
        # min_cash = 5% of $10,000 = $500 → budget = $9,500 → 95 shares @ $100
        assert preview["est_shares"] == pytest.approx(95.0, abs=0.001)
        assert preview["est_cash_after"] == pytest.approx(500.0)

        conf = await _chat_then_confirm(result, trades_before)
        assert len(conf["trades"]) == 1
        assert conf["trades"][0]["shares"] == pytest.approx(95.0, abs=0.001)

    async def test_buy_with_reserve_spends_dry_powder(self, with_cash, monkeypatch, llm_backend):
        """A BUY with use_reserve=true can spend below the cash floor (to $0)."""
        monkeypatch.setattr(settings, "sim_max_position_pct", 100.0)

        async def mock_close(ticker):
            return 100.0 if ticker == "AAPL" else None
        monkeypatch.setattr(sim, "_latest_close", mock_close)
        _mock_chat_context(monkeypatch)

        _stub_chat_reply(monkeypatch, (
            '[[ACTION]]\n'
            '{"actions": [{"ticker": "AAPL", "action": "BUY", "use_reserve": true, "reason": "spend dry powder"}]}\n'
            '[[/ACTION]]'
        ))

        result = await sim.sim_chat([{"role": "user", "content": "spend the dry powder on AAPL"}])
        trades_before = await _trade_count()

        action = result["proposals"][0]["actions"][0]
        assert action["use_reserve"] is True
        preview = action["preview"]
        assert preview["feasible"] is True
        # floor = $0 → budget = $10,000 → 100 shares @ $100
        assert preview["est_shares"] == pytest.approx(100.0, abs=0.001)
        assert preview["est_cash_after"] == pytest.approx(0.0)

        conf = await _chat_then_confirm(result, trades_before)
        assert len(conf["trades"]) == 1
        assert conf["trades"][0]["shares"] == pytest.approx(100.0, abs=0.001)
        assert "[reserve spent]" in conf["trades"][0]["reason"]


class TestSimChatMultipleActions:
    """Tests that one chat turn proposes multiple actions and one confirm
    executes them all."""

    async def test_sell_then_buy(self, with_cash, monkeypatch, llm_backend):
        """Sell one ticker, then buy another — both should execute on confirm."""
        # Raise max position % so the $2000 SPY buy isn't clamped to $1500
        monkeypatch.setattr(settings, "sim_max_position_pct", 30.0)

        # Setup: buy MDB and ANET
        await sim._exec_buy("MDB", 100.0, 5000.0, "initial")  # 50 shares MDB
        await sim._exec_buy("ANET", 50.0, 2000.0, "initial")  # 40 shares ANET
        trades_before = await _trade_count()

        async def mock_close(ticker):
            prices = {"MDB": 100.0, "ANET": 50.0, "SPY": 400.0}
            return prices.get(ticker)
        monkeypatch.setattr(sim, "_latest_close", mock_close)
        _mock_chat_context(monkeypatch)

        _stub_chat_reply(monkeypatch, (
            'Trim MDB by $2000 and buy SPY.\n'
            '[[ACTION]]\n'
            '{"actions": [\n'
            '  {"ticker": "MDB", "action": "SELL", "amount": 2000.0, "reason": "trim"},\n'
            '  {"ticker": "SPY", "action": "BUY", "amount": 2000.0, "reason": "diversify"}\n'
            ']}\n'
            '[[/ACTION]]'
        ))

        result = await sim.sim_chat([{"role": "user", "content": "trim MDB to buy SPY"}])

        proposal = result["proposals"][0]
        assert len(proposal["actions"]) == 2
        sell_preview = proposal["actions"][0]["preview"]
        buy_preview = proposal["actions"][1]["preview"]
        assert sell_preview["feasible"] is True
        assert sell_preview["est_shares"] == pytest.approx(20.0, abs=0.001)
        assert buy_preview["feasible"] is True
        assert buy_preview["est_shares"] == pytest.approx(5.0, abs=0.001)  # $2000 / $400

        conf = await _chat_then_confirm(result, trades_before)
        assert len(conf["trades"]) == 2
        # First trade: SELL MDB $2000 / $100 = 20 shares
        assert conf["trades"][0]["ticker"] == "MDB"
        assert conf["trades"][0]["side"] == "SELL"
        assert conf["trades"][0]["shares"] == pytest.approx(20.0, abs=0.001)
        # Second trade: BUY SPY $2000 / $400 = 5 shares
        assert conf["trades"][1]["ticker"] == "SPY"
        assert conf["trades"][1]["side"] == "BUY"
        assert conf["trades"][1]["shares"] == pytest.approx(5.0, abs=0.001)

    async def test_no_action_block_returns_text_only(self, with_cash, monkeypatch, llm_backend):
        """If the LLM doesn't include an action block, just return the text (no proposal)."""
        _mock_chat_context(monkeypatch)

        llm_response_text = 'I think the portfolio looks good right now. No changes needed.'
        _stub_chat_reply(monkeypatch, llm_response_text)

        result = await sim.sim_chat([{"role": "user", "content": "what do you think?"}])

        assert result["proposals"] == []
        assert "trades" not in result
        assert "actions_executed" not in result
        assert result["text"] == llm_response_text


# ---------------------------------------------------------------------------
# sim_chat_stream — streaming variant
# ---------------------------------------------------------------------------

class TestSimChatStream:
    """Tests for sim.sim_chat_stream: progressive deltas, persist-after-done,
    action-block PROPOSAL after the stream completes, orphan-message avoidance."""

    async def _run_stream(self, new_msgs, fake_llm_events):
        """Drive sim_chat_stream with a mocked llm_mod.chat_stream that yields
        the given list of events. Returns the list of events yielded to the
        caller (the frontend), plus the final persisted chat history."""
        async def fake_chat_stream(messages):
            for evt in fake_llm_events:
                yield evt
        monkeypatch_local = sim.llm_mod.chat_stream
        sim.llm_mod.chat_stream = fake_chat_stream
        try:
            out = []
            async for evt in sim.sim_chat_stream(new_msgs):
                out.append(evt)
        finally:
            sim.llm_mod.chat_stream = monkeypatch_local
        history = await sim.get_chat_history()
        return out, history

    async def test_streams_deltas_then_done(self, with_cash, monkeypatch):
        """Plain text response — deltas arrive, then a done envelope with the
        full text and an empty proposals list. No action block means no proposal."""
        async def mock_context():
            return "mock context"
        monkeypatch.setattr(sim, "_build_sim_chat_context", mock_context)

        fake_events = [
            {"type": "delta", "text": "Hello"},
            {"type": "delta", "text": " world"},
            {"type": "done", "text": "Hello world", "reasoning": "",
             "backend": "a", "model": "m1"},
        ]
        out, history = await self._run_stream(
            [{"role": "user", "content": "hi"}], fake_events,
        )
        deltas = [e for e in out if e["type"] == "delta"]
        dones = [e for e in out if e["type"] == "done"]
        assert [d["text"] for d in deltas] == ["Hello", " world"]
        assert len(dones) == 1
        assert dones[0]["text"] == "Hello world"
        assert dones[0]["proposals"] == []
        assert "trades" not in dones[0]
        assert "actions_executed" not in dones[0]
        # Persistence: user msg + assistant reply both stored AFTER stream done.
        assert len(history) == 2
        assert history[0]["role"] == "user"
        assert history[0]["content"] == "hi"
        assert history[1]["role"] == "assistant"
        assert history[1]["content"] == "Hello world"

    async def test_action_block_proposed_after_done(self, with_cash, monkeypatch):
        """LLM streams raw text containing [[ACTION]]; the done event carries a
        PROPOSAL (with preview) — nothing executes until confirm; done.text is
        the stripped display_text."""
        await sim._exec_buy("AAPL", 100.0, 10000.0, "initial")
        trades_before = await _trade_count()

        async def mock_close(ticker):
            return 100.0 if ticker == "AAPL" else None
        monkeypatch.setattr(sim, "_latest_close", mock_close)

        async def mock_context():
            return "mock context"
        monkeypatch.setattr(sim, "_build_sim_chat_context", mock_context)

        raw_text = (
            'Trimming AAPL.\n'
            '[[ACTION]]\n'
            '{"actions": [{"ticker": "AAPL", "action": "SELL", "amount": 3000.0, "reason": "trim"}]}\n'
            '[[/ACTION]]'
        )
        fake_events = [
            {"type": "delta", "text": raw_text[:10]},
            {"type": "delta", "text": raw_text[10:]},
            {"type": "done", "text": raw_text, "reasoning": "",
             "backend": "a", "model": "m1"},
        ]
        out, history = await self._run_stream(
            [{"role": "user", "content": "trim AAPL by $3000"}], fake_events,
        )
        dones = [e for e in out if e["type"] == "done"]
        assert len(dones) == 1
        done = dones[0]
        # The done event proposes, it does not execute.
        assert "trades" not in done
        assert "actions_executed" not in done
        assert len(done["proposals"]) == 1
        preview = done["proposals"][0]["actions"][0]["preview"]
        assert preview["feasible"] is True
        assert preview["est_shares"] == pytest.approx(30.0, abs=0.001)
        assert await _trade_count() == trades_before  # nothing executed yet
        # done.text is the stripped display_text, NOT the raw action block.
        assert "[[ACTION]]" not in done["text"]
        assert "Trimming AAPL." in done["text"]
        # raw field carries the original full response.
        assert "[[ACTION]]" in done["raw"]
        # Persisted assistant message is the stripped display_text.
        assert history[-1]["role"] == "assistant"
        assert "[[ACTION]]" not in history[-1]["content"]

        # Confirming the streamed proposal executes the previewed sell.
        conf = await sim.confirm_chat_proposal(done["proposals"][0]["proposal_id"])
        assert len(conf["trades"]) == 1
        assert conf["trades"][0]["ticker"] == "AAPL"
        assert conf["trades"][0]["side"] == "SELL"
        assert conf["trades"][0]["shares"] == pytest.approx(30.0, abs=0.001)

    async def test_error_no_text_persists_error_message(self, with_cash, monkeypatch):
        """If the LLM errors before any text, an error-style assistant reply is
        emitted and persisted (so the user's question isn't lost)."""
        async def mock_context():
            return "mock context"
        monkeypatch.setattr(sim, "_build_sim_chat_context", mock_context)

        fake_events = [{"type": "error", "text": "connection refused"}]
        out, history = await self._run_stream(
            [{"role": "user", "content": "hi"}], fake_events,
        )
        dones = [e for e in out if e["type"] == "done"]
        assert len(dones) == 1
        assert "LLM unavailable: connection refused" in dones[0]["text"]
        assert dones[0]["proposals"] == []
        assert dones[0]["error"] == "connection refused"
        assert "trades" not in dones[0]
        assert "actions_executed" not in dones[0]
        # Both user question and error reply persisted.
        assert len(history) == 2
        assert history[0]["role"] == "user"
        assert history[1]["role"] == "assistant"
        assert "LLM unavailable" in history[1]["content"]


class TestSchedulerTick:
    """The scheduler must run each stage independently: a skipped or failed
    stage (sim lock held, monthly skip) must not prevent the daily-core
    deployment from running."""

    @pytest.fixture(autouse=True)
    def _trading_day(self, monkeypatch):
        # Pin the clock to a normal weekday: _scheduler_tick returns early on
        # weekends/NYSE holidays, so without this the tests only pass Mon-Fri
        # (CI failed on a weekend run). 2026-09-21 is a Monday.
        monkeypatch.setattr(sim, "_utcnow", lambda: datetime(2026, 9, 21, 22, 30))

    async def test_daily_core_runs_when_sim_lock_held(self, monkeypatch):
        from app import daily_core, monthly

        called = {"sim": 0, "daily_core": 0}

        async def fake_sim_cycle():
            called["sim"] += 1
            return {"skipped": False, "trades": []}

        async def fake_daily_core_cycle():
            called["daily_core"] += 1
            return {"deployment": {"trades": []}}

        async def fake_monthly_cycle():
            return {"skipped": True, "reason": "not month end"}

        monkeypatch.setattr(sim, "run_cycle", fake_sim_cycle)
        monkeypatch.setattr(daily_core, "run_daily_cycle", fake_daily_core_cycle)
        monkeypatch.setattr(monthly, "run_monthly_cycle", fake_monthly_cycle)

        await sim._run_cycle_lock.acquire()
        try:
            await sim._scheduler_tick()
        finally:
            sim._run_cycle_lock.release()

        assert called["sim"] == 0, "the sim cycle should be skipped while the lock is held"
        assert called["daily_core"] == 1, (
            "daily-core must still run when the sim cycle is skipped for a "
            "held lock — otherwise that day's deployment is silently lost"
        )

    async def test_all_stages_run_when_unlocked(self, monkeypatch):
        from app import daily_core, monthly

        called = {"sim": 0, "daily_core": 0, "monthly": 0}

        async def fake_sim_cycle():
            called["sim"] += 1
            return {"skipped": False, "trades": []}

        async def fake_daily_core_cycle():
            called["daily_core"] += 1
            return {"deployment": {"trades": []}}

        async def fake_monthly_cycle():
            called["monthly"] += 1
            return {"skipped": False, "trades": []}

        monkeypatch.setattr(sim, "run_cycle", fake_sim_cycle)
        monkeypatch.setattr(daily_core, "run_daily_cycle", fake_daily_core_cycle)
        monkeypatch.setattr(monthly, "run_monthly_cycle", fake_monthly_cycle)

        await sim._scheduler_tick()

        assert called == {"sim": 1, "daily_core": 1, "monthly": 1}

    async def test_daily_core_runs_even_if_sim_cycle_raises(self, monkeypatch):
        from app import daily_core, monthly

        called = {"daily_core": 0}

        async def boom():
            raise RuntimeError("sim exploded")

        async def fake_daily_core_cycle():
            called["daily_core"] += 1
            return {"deployment": {"trades": []}}

        async def fake_monthly_cycle():
            return {"skipped": True, "reason": "x"}

        monkeypatch.setattr(sim, "run_cycle", boom)
        monkeypatch.setattr(daily_core, "run_daily_cycle", fake_daily_core_cycle)
        monkeypatch.setattr(monthly, "run_monthly_cycle", fake_monthly_cycle)

        await sim._scheduler_tick()
        assert called["daily_core"] == 1


# ---------------------------------------------------------------------------
# Backfill (deterministic approximation, no LLM calls)
# ---------------------------------------------------------------------------

class TestSimBackfill:
    @pytest.fixture
    def fake_series(self, monkeypatch):
        """Patch optimize._load_series with a tiny deterministic market:
        one strong uptrender + one downtrender so the engine has signals."""
        from app import optimize as opt
        from app.analysis import signal_series
        from tests.test_optimize import _gen_candles

        up = signal_series(_gen_candles(100.0, 0.01, n=300, seed=1))
        dn = signal_series(_gen_candles(100.0, -0.01, n=300, seed=2))

        async def fake_load_series(_tickers, start=None):
            return {"UP": up, "DN": dn}

        monkeypatch.setattr(opt, "_load_series", fake_load_series)
        return {"UP": up, "DN": dn}

    async def test_backfill_replaces_state(self, mem_db, fake_series, monkeypatch):
        """The replay persists trade/snapshot rows with historical stamps,
        converges to the replay's end state, and flags approximation."""
        from sqlalchemy import select

        from app.db import SimAllowance as Al

        r = await sim.backfill(start="2025-03-01")
        assert r["ok"] is True
        assert r["approximation"] is True
        assert r["days"] > 0 and r["snapshots"] == r["days"]

        async def check():
            async with mem_db() as s:
                snaps = (await s.scalars(select(SimSnapshot))).all()
                assert len(snaps) == r["snapshots"]
                days = sorted(x.created_at.strftime("%Y-%m-%d") for x in snaps)
                assert days[0] == "2025-03-01"
                # allowance rows: one per replayed month (+ current month)
                months = sorted(x.month for x in (await s.scalars(
                    select(Al))).all())
                assert months[0] == "2025-03"
                # positions converge to the replay's holdings
                pos = (await s.scalars(select(SimPosition))).all()
                assert len(pos) >= 1
                for p in pos:
                    assert p.shares > 0
                # trades carry the replay day
                trs = (await s.scalars(select(SimTrade))).all()
                for t in trs:
                    assert t.created_at.strftime("%Y-%m") >= "2025-03"
                    assert t.shares > 0 and t.price > 0
                    assert t.reason.startswith("backfill:")
        await check()

    async def test_backfill_positions_carry_entry_anchors(
            self, mem_db, fake_series, monkeypatch, tmp_path):
        """End-state positions must carry the replay's REAL entry anchors:
        ``opened_at`` = the actual first-buy session date (the monotone ATR
        ratchet's anchor — wall-clock now() would let the ratchet see bars
        from before the position existed), and ``stop_price`` = the percent
        stop frozen at the entry fill (a top-up can never loosen it)."""
        from sqlalchemy import select

        from app import optimize as opt

        monkeypatch.setattr(sim, "_STATE_FILE", tmp_path / "sim_state.json")

        r = await sim.backfill(start="2025-03-01")
        assert r["ok"] is True

        async with mem_db() as s:
            pos = (await s.scalars(select(SimPosition))).all()
            trades = (await s.scalars(select(SimTrade)
                                      .order_by(SimTrade.created_at))).all()
        assert pos, "the replay must end with at least one holding"

        # First BUY per ticker, respecting that a full SELL closes the
        # position — a later re-buy is a NEW entry.
        opened: dict[str, str] = {}
        entry_px: dict[str, float] = {}
        live: set[str] = set()
        for tr in trades:
            if tr.side == "BUY":
                if tr.ticker not in live:
                    live.add(tr.ticker)
                    opened[tr.ticker] = tr.created_at.strftime("%Y-%m-%d")
                    entry_px[tr.ticker] = tr.price
            else:
                live.discard(tr.ticker)

        stop_pct = opt._live_sim_params().stop_pct
        for p in pos:
            assert p.opened_at is not None
            anchor = p.opened_at.strftime("%Y-%m-%d") if p.opened_at else ""
            assert anchor == opened[p.ticker], (
                f"{p.ticker}: opened_at {anchor} must be the replay's "
                f"first-buy date {opened[p.ticker]}, not wall-clock now()")
            assert p.stop_price is not None
            assert p.stop_price == pytest.approx(
                entry_px[p.ticker] * (1 - stop_pct / 100), rel=1e-6)

    async def test_backfill_resets_session_guard(
            self, mem_db, fake_series, monkeypatch, tmp_path):
        """The replay replaces the whole portfolio, so it must clear the
        per-session fill guard — the next cycle has to be free to run."""
        monkeypatch.setattr(sim, "_STATE_FILE", tmp_path / "sim_state.json")

        sim._stamp_session_guard("2099-01-01")  # a consumed-looking session
        r = await sim.backfill(start="2025-03-01")
        assert r["ok"] is True
        assert sim._load_state().get("last_session") is None

    async def test_uncovered_current_month_keeps_its_contribution(
            self, mem_db, fake_series, monkeypatch):
        """A backfill run before the current month has candles must restore the
        live deposit (marker + row + cash), not just the row — otherwise the
        account is permanently one contribution short and the next live
        deposit would hit the unique allowance constraint."""
        from sqlalchemy import select

        from app.db import SimAccount
        from app.db import SimAllowance as Al

        monkeypatch.setattr(sim, "_current_month", lambda: "2026-09")

        # Control: no live marker -> the current month is not replayed at all.
        await sim.backfill(start="2025-03-01")
        async with mem_db() as s:
            base_cash = (await s.get(SimAccount, 1)).cash

        # Seed the live deposit marker, then re-run from the same window.
        async def seed():
            async with mem_db() as s:
                acc = await s.get(SimAccount, 1)
                acc.cash = 0.0
                acc.last_allowance_month = "2026-09"
                await s.commit()
        await seed()

        r = await sim.backfill(start="2025-03-01")
        assert r["ok"] is True

        async with mem_db() as s:
            acc = await s.get(SimAccount, 1)
            assert acc.last_allowance_month == "2026-09"
            months = sorted(x.month for x in (await s.scalars(select(Al))).all())
            assert "2026-09" in months
            # the September contribution is actually funded, not just logged
            assert acc.cash == pytest.approx(
                base_cash + settings.sim_monthly_allowance, abs=0.5)

    async def test_backfill_zero_equity_days_not_persisted_negatively(
            self, mem_db, fake_series):
        """The replay starts flat (no cash until the first deposit): early
        snapshots carry 0 equity — the curve must still start at the window
        start, not the first deposit."""
        r = await sim.backfill(start="2025-03-01")
        assert r["ok"] is True
        assert r["start"] == "2025-03-01"

    async def test_no_llm_calls(self, mem_db, fake_series, monkeypatch):
        """The deterministic backfill must never touch the LLM client."""
        from app import llm as llm_mod

        def _explode(*a, **k):
            raise AssertionError("LLM called during deterministic backfill")

        monkeypatch.setattr(llm_mod, "chat", _explode, raising=False)
        monkeypatch.setattr(llm_mod, "chat_stream", _explode, raising=False)
        r = await sim.backfill(start="2025-03-01")
        assert r["ok"] is True

    async def test_benchmark_backfill_replaces_state(self, mem_db, monkeypatch):
        """The DCA replay deposits on each month's first candle, buys at that
        close, and persists daily snapshots + the end-state account."""
        from unittest.mock import patch

        import pandas as pd
        from sqlalchemy import select as sa_select

        from app.db import Candle, SimBenchmarkAccount, SimBenchmarkSnapshot

        dates = pd.bdate_range("2025-03-01", "2025-04-30")
        candles = [
            Candle(ticker="URTH", timestamp=ts.to_pydatetime(), open=100.0,
                   high=101.0, low=99.0, close=float(100 + i * 0.1), volume=1e6)
            for i, ts in enumerate(dates)
        ]

        async def fake_refresh(ticker, period):
            return None

        async def fake_sync_start():
            return None

        # _load candles come from the mem_db (Candle rows inserted above)
        async def seed():
            async with mem_db() as s:
                for c in candles:
                    s.add(c)
                await s.commit()
        await seed()

        monkeypatch.setattr(sim, "_current_month", lambda: "2025-04")
        with patch("app.market.refresh", fake_refresh), \
             patch.object(sim, "_sync_start_date", fake_sync_start):
            r = await sim.backfill_benchmark(start="2025-03-01")
        assert r["ok"] is True
        assert r["n_buys"] == 2  # March + April
        assert r["contributed"] == 2 * settings.sim_monthly_allowance

        async def check():
            async with mem_db() as s:
                acc = await s.get(SimBenchmarkAccount, 1)
                assert acc is not None
                # 2 deposits fully invested; price rose -> equity > contributed
                assert acc.shares > 0
                assert r["final_equity"] > r["contributed"] * 0.99
                snaps = (await s.scalars(sa_select(SimBenchmarkSnapshot))).all()
                assert len(snaps) == r["days"]
                # contributed steps at the first trading day of each month
                by_day = {x.created_at.strftime("%Y-%m-%d"): x.allowance_total for x in snaps}
                assert by_day["2025-03-03"] == settings.sim_monthly_allowance  # first bday
                assert by_day["2025-04-01"] == 2 * settings.sim_monthly_allowance
        await check()

    async def test_benchmark_backfill_marker_not_ahead_of_funded_months(
            self, mem_db, monkeypatch):
        """If the current month has no candle, the marker must stay at the last
        FUNDED month — writing the current month would suppress its next live
        deposit forever (the benchmark ends a contribution behind)."""
        from unittest.mock import patch

        import pandas as pd

        from app.db import Candle, SimBenchmarkAccount

        dates = pd.bdate_range("2025-03-01", "2025-03-31")
        candles = [Candle(ticker="URTH", timestamp=ts.to_pydatetime(), open=100.0,
                          high=101.0, low=99.0, close=100.0, volume=1e6) for ts in dates]

        async def fake_refresh(ticker, period):
            return None

        async def fake_sync_start():
            return None

        async def seed():
            async with mem_db() as s:
                for c in candles:
                    s.add(c)
                await s.commit()
        await seed()

        monkeypatch.setattr(sim, "_current_month", lambda: "2025-04")
        with patch("app.market.refresh", fake_refresh), \
             patch.object(sim, "_sync_start_date", fake_sync_start):
            r = await sim.backfill_benchmark(start="2025-03-01")
        assert r["ok"] is True

        async with mem_db() as s:
            acc = await s.get(SimBenchmarkAccount, 1)
            # March was the last funded month; April had no candle.
            assert acc.last_allowance_month == "2025-03"


# ---------------------------------------------------------------------------
# Nightly shared universe prefetch
# ---------------------------------------------------------------------------

class TestUniversePrefetch:
    async def test_prefetch_uses_configured_provider(self, monkeypatch):
        """The shared prefetch must route through the configured provider
        (auto → Twelve Data), not the bulk Yahoo path."""
        async def fake_candidates():
            return ["AAA", "BBB"]

        captured: dict = {}

        async def fake_refresh_many(tickers, period, **kwargs):
            captured["tickers"] = list(tickers)
            captured["kwargs"] = kwargs
            return list(tickers), []

        monkeypatch.setattr(sim, "_candidate_tickers", fake_candidates)
        monkeypatch.setattr(sim, "refresh_many", fake_refresh_many)

        r = await sim.prefetch_universe()
        assert r["ok"] is True and r["total"] == 2 and r["refreshed"] == 2
        assert captured["tickers"] == ["AAA", "BBB"]
        assert captured["kwargs"].get("use_provider") is True

    async def test_empty_universe_is_a_noop(self, monkeypatch):
        async def empty_candidates():
            return []

        monkeypatch.setattr(sim, "_candidate_tickers", empty_candidates)
        r = await sim.prefetch_universe()
        assert r["ok"] is False


# ---------------------------------------------------------------------------
# Trading-day calendar (US NYSE)
# ---------------------------------------------------------------------------

class TestTradingDayCalendar:
    def test_weekends_are_not_trading_days(self):
        assert sim.is_trading_day(datetime(2026, 9, 19, 22, 30)) is False  # Sat
        assert sim.is_trading_day(datetime(2026, 9, 20, 22, 30)) is False  # Sun

    def test_nyse_holidays_2026(self):
        for d in (
            datetime(2026, 1, 1),   # New Year's Day
            datetime(2026, 1, 19),  # MLK
            datetime(2026, 2, 16),  # Washington's Birthday
            datetime(2026, 4, 3),   # Good Friday
            datetime(2026, 5, 25),  # Memorial Day
            datetime(2026, 6, 19),  # Juneteenth
            datetime(2026, 7, 3),   # July 4 (Sat) observed Friday
            datetime(2026, 9, 7),   # Labor Day
            datetime(2026, 11, 26),  # Thanksgiving
            datetime(2026, 12, 25),  # Christmas
        ):
            assert sim.is_trading_day(d) is False, d

    def test_normal_weekdays_are_trading_days(self):
        assert sim.is_trading_day(datetime(2026, 9, 21)) is True   # Mon
        assert sim.is_trading_day(datetime(2026, 1, 2)) is True    # Fri after NY
        assert sim.is_trading_day(datetime(2026, 7, 6)) is True    # Mon after Jul 4

    async def test_scheduler_tick_skips_a_weekend(self, monkeypatch):
        from app import daily_core, monthly

        monkeypatch.setattr(sim, "_utcnow", lambda: datetime(2026, 9, 19, 22, 30))  # Sat

        async def boom(*a, **k):
            raise AssertionError("a cycle ran on a non-trading day")

        monkeypatch.setattr(monthly, "run_monthly_cycle", boom)
        monkeypatch.setattr(daily_core, "run_daily_cycle", boom)
        await sim._scheduler_tick()  # must return without running anything
