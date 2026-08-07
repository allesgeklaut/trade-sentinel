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

import pytest
from sqlalchemy import StaticPool
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker

from app import sim
from app.config import settings
from app.db import (
    Base,
    SimAccount,
    SimBenchmarkAccount,
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
        assert t["shares"] == 33  # int(5000 // 150)
        assert t["price"] == 150.0

        # Check state
        async with sim.Session() as s:
            from sqlalchemy import select as sa_select
            pos = await s.scalar(sa_select(SimPosition).where(SimPosition.ticker == "AAPL"))
            assert pos is not None
            assert pos.shares == 33
            assert pos.avg_cost == 150.0

            acc = await s.get(SimAccount, 1)
            assert acc.cash == pytest.approx(10000 - 33 * 150)

            trades = (await s.scalars(sa_select(SimTrade))).all()
            assert len(trades) == 1

    async def test_buy_insufficient_budget(self, with_cash):
        """A buy with budget less than one share should be skipped."""
        t = await sim._exec_buy("BRK.A", 500_000, 10000, "too expensive")
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
            assert pos.shares == 30
            # Weighted avg: (20*100 + 10*150) / 30 = 3500/30 ≈ 116.67
            assert pos.avg_cost == pytest.approx(116.67, abs=0.1)


class TestExecSell:
    async def test_basic_sell(self, with_cash):
        """A sell should increase cash, reduce position, and log a trade."""
        await sim._exec_buy("AAPL", 100.0, 2000.0, "buy")  # 20 shares @ 100

        t = await sim._exec_sell("AAPL", 120.0, None, "sell all")
        assert t is not None
        assert t["side"] == "SELL"
        assert t["shares"] == 20
        assert t["price"] == 120.0

        async with sim.Session() as s:
            from sqlalchemy import select as sa_select
            pos = await s.scalar(sa_select(SimPosition).where(SimPosition.ticker == "AAPL"))
            assert pos is None  # fully sold, position deleted

            acc = await s.get(SimAccount, 1)
            # cash = 10000 - 2000 (buy) + 2400 (sell 20*120)
            assert acc.cash == pytest.approx(10000 - 2000 + 2400)

    async def test_sell_no_position(self, with_cash):
        """Selling a ticker with no position should return None."""
        t = await sim._exec_sell("MSFT", 100.0, None, "nothing to sell")
        assert t is None

    async def test_partial_sell(self, with_cash):
        """A partial sell should reduce shares but keep the position."""
        await sim._exec_buy("AAPL", 100.0, 3000.0, "buy")  # 30 shares

        t = await sim._exec_sell("AAPL", 110.0, 10, "partial sell")
        assert t is not None
        assert t["shares"] == 10

        async with sim.Session() as s:
            from sqlalchemy import select as sa_select
            pos = await s.scalar(sa_select(SimPosition).where(SimPosition.ticker == "AAPL"))
            assert pos is not None
            assert pos.shares == 20  # 30 - 10


# ---------------------------------------------------------------------------
# Reset
# ---------------------------------------------------------------------------

class TestReset:
    async def test_reset_wipes_everything(self, with_cash):
        """Reset should clear all tables and reinitialize cash."""
        # Create some state
        await sim._exec_buy("AAPL", 100.0, 2000.0, "buy")
        async with sim.Session() as s:
            s.add(SimSnapshot(cash=8000, positions_value=2000, total_equity=10000, allowance_total=0))
            await s.commit()

        result = await sim.reset_sim()
        assert result["ok"] is True

        async with sim.Session() as s:
            from sqlalchemy import select as sa_select
            pos = await s.scalar(sa_select(SimPosition).where(SimPosition.ticker == "AAPL"))
            assert pos is None
            trades = (await s.scalars(sa_select(SimTrade))).all()
            assert len(trades) == 0
            snaps = (await s.scalars(sa_select(SimSnapshot))).all()
            assert len(snaps) == 0
            acc = await s.get(SimAccount, 1)
            assert acc.cash == settings.sim_start_cash


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