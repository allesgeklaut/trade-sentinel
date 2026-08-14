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

import httpx
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


class TestLLMDecide:
    """Integration tests for _llm_decide: LLM decisions → executed trades.

    Mocks the Ollama HTTP call and _latest_close so no network is needed.
    """

    def _fake_http(self, monkeypatch, content: str):
        """Replace httpx.AsyncClient with a fake returning ``content``."""
        class FakeResponse:
            status_code = 200
            def json(self):
                return {"message": {"content": content}}

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

    async def test_fractional_sell_and_clamped_buy(self, with_cash, monkeypatch):
        """A SELL with 'amount' sells only that value; a BUY is clamped to budget."""
        # Seed a position: 30 shares of AAPL @ 100 = $3000
        await sim._exec_buy("AAPL", 100.0, 3000.0, "seed")

        # LLM returns: sell $500 of AAPL, buy $99999 of MSFT (clamped by cash).
        decisions = json.dumps([
            {"ticker": "AAPL", "action": "SELL", "amount": 500.0, "reason": "trim"},
            {"ticker": "MSFT", "action": "BUY", "amount": 99999.0, "reason": "diversify"},
        ])
        self._fake_http(monkeypatch, decisions)

        # Prices: AAPL @ 100, MSFT @ 50
        async def fake_close(ticker):
            return {"AAPL": 100.0, "MSFT": 50.0}.get(ticker)
        monkeypatch.setattr(sim, "_latest_close", fake_close)

        valuation = {
            "cash": 10000.0, "positions_value": 3000.0, "total_equity": 13000.0,
            "allowance_total": 10000.0, "positions": [],
        }
        executed = await sim._llm_decide(valuation, [], {})

        # Two trades executed
        assert len(executed) == 2

        # Fractional SELL: $500 / $100 = 5 shares sold, 25 remain
        sell = next(t for t in executed if t["ticker"] == "AAPL")
        assert sell["side"] == "SELL"
        assert sell["shares"] == pytest.approx(5.0, abs=0.001)

        # BUY clamped: cash after sell = 10000 + 500 = 10500; min_cash = 5% of
        # equity. Budget = min(cash - min_cash, max_position - current).
        buy = next(t for t in executed if t["ticker"] == "MSFT")
        assert buy["side"] == "BUY"
        # MSFT @ 50 → shares = budget / 50
        assert buy["shares"] > 0

        async with sim.Session() as s:
            from sqlalchemy import select as sa_select
            aapl = await s.scalar(sa_select(SimPosition).where(SimPosition.ticker == "AAPL"))
            assert aapl is not None
            assert aapl.shares == pytest.approx(25.0, abs=0.001)  # 30 - 5

            msft = await s.scalar(sa_select(SimPosition).where(SimPosition.ticker == "MSFT"))
            assert msft is not None
            assert msft.shares > 0

    async def test_unparseable_decision_falls_back(self, with_cash, monkeypatch):
        """A genuinely unparseable response (not an empty array) should fall
        back to the deterministic candidate trades."""
        # LLM returns garbage that isn't a JSON array.
        self._fake_http(monkeypatch, "I am not JSON")

        async def fake_close(ticker):
            return 100.0
        monkeypatch.setattr(sim, "_latest_close", fake_close)

        valuation = {
            "cash": 10000.0, "positions_value": 0.0, "total_equity": 10000.0,
            "allowance_total": 10000.0, "positions": [],
        }
        deterministic = [
            {"ticker": "AAPL", "side": "BUY", "shares": 5, "price": 100.0, "reason": "det"},
        ]
        executed = await sim._llm_decide(valuation, deterministic, {})

        # Fell back: the deterministic trade list is returned as-is (execution
        # of the fallback happens in the caller), not dropped.
        assert len(executed) == 1
        assert executed[0]["ticker"] == "AAPL"
        assert executed[0]["side"] == "BUY"
        assert executed[0] is deterministic[0]

    async def test_empty_decisions_do_not_fall_back(self, with_cash, monkeypatch):
        """An empty [] decision should execute no trades, not fall back to
        deterministic candidates."""
        await sim._exec_buy("AAPL", 100.0, 2000.0, "seed")  # 20 shares

        # LLM returns an empty array → decide to do nothing.
        self._fake_http(monkeypatch, json.dumps([]))

        async def fake_close(ticker):
            return 100.0
        monkeypatch.setattr(sim, "_latest_close", fake_close)

        valuation = {
            "cash": 10000.0, "positions_value": 2000.0, "total_equity": 12000.0,
            "allowance_total": 10000.0, "positions": [],
        }
        executed = await sim._llm_decide(valuation, [], {})

        # No trades executed, no fallback triggered
        assert executed == []

        async with sim.Session() as s:
            from sqlalchemy import select as sa_select
            pos = await s.scalar(sa_select(SimPosition).where(SimPosition.ticker == "AAPL"))
            assert pos is not None  # position untouched
            assert pos.shares == pytest.approx(20.0, abs=0.001)

    async def test_sell_without_size_sells_entire_position(self, with_cash, monkeypatch):
        """A SELL with no size field sells the whole position (default)."""
        await sim._exec_buy("AAPL", 100.0, 2000.0, "seed")  # 20 shares

        decisions = json.dumps([
            {"ticker": "AAPL", "action": "SELL", "reason": "exit"},
        ])
        self._fake_http(monkeypatch, decisions)

        async def fake_close(ticker):
            return 100.0
        monkeypatch.setattr(sim, "_latest_close", fake_close)

        valuation = {
            "cash": 10000.0, "positions_value": 2000.0, "total_equity": 12000.0,
            "allowance_total": 10000.0, "positions": [],
        }
        executed = await sim._llm_decide(valuation, [], {})

        assert len(executed) == 1
        assert executed[0]["ticker"] == "AAPL"
        assert executed[0]["side"] == "SELL"
        assert executed[0]["shares"] == pytest.approx(20.0, abs=0.001)

        async with sim.Session() as s:
            from sqlalchemy import select as sa_select
            pos = await s.scalar(sa_select(SimPosition).where(SimPosition.ticker == "AAPL"))
            assert pos is None  # fully sold


# ---------------------------------------------------------------------------
# Year-long simulation with synthetic market data
# ---------------------------------------------------------------------------

import random
from datetime import datetime, timedelta


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
        for ticker, rows in self.SYNTHETIC.items():
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
            SimAllowance,
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

        # --- mock refresh to no-op (no yfinance calls) ---
        async def mock_refresh(ticker, period="2y"):
            pass

        monkeypatch.setattr(sim, "refresh", mock_refresh)

        # --- mock candidate tickers (skip watchlist / universe file) ---
        async def mock_candidates():
            return ["BULL", "FLAT", "BEAR"]

        monkeypatch.setattr(sim, "_candidate_tickers", mock_candidates)

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
        from datetime import timezone as tz
        for snap in sim_snaps:
            assert snap.created_at.tzinfo is not None, \
                "SimSnapshot.created_at must be tz-aware"
            assert snap.created_at.utcoffset() == tz.utc.utcoffset(None), \
                "SimSnapshot.created_at must be UTC"
        for snap in bench_snaps:
            assert snap.created_at.tzinfo is not None, \
                "SimBenchmarkSnapshot.created_at must be tz-aware"
            assert snap.created_at.utcoffset() == tz.utc.utcoffset(None), \
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

        # 8. Bot has at least one open position
        assert len(val["positions"]) >= 1

        # 9. Equity curve snapshots show non-decreasing allowance_total
        curve = await sim.get_equity_curve(limit=365)
        assert len(curve) == 12
        assert curve[-1]["allowance_total"] == pytest.approx(
            settings.sim_monthly_allowance * 12
        )


# ---------------------------------------------------------------------------
# Timezone convention (Batch 2)
# ---------------------------------------------------------------------------

class TestTimezoneConvention:
    """Verify that sim timestamps are stored as tz-aware UTC, and the monthly
    allowance deposit is anchored to Europe/Vienna (a calendar concept)."""

    async def test_utcnow_is_tz_aware(self):
        """sim._utcnow() must return a tz-aware UTC datetime."""
        from datetime import timezone as tz
        t = sim._utcnow()
        assert t.tzinfo is not None
        assert t.utcoffset() == tz.utc.utcoffset(None)

    async def test_simtrade_created_at_is_tz_aware_utc(self, with_cash):
        """A logged trade's created_at must be tz-aware UTC after a buy."""
        await sim._exec_buy("AAPL", 100.0, 1000.0, "tz test")
        async with sim.Session() as s:
            from sqlalchemy import select as sa_select
            trade = await s.scalar(sa_select(SimTrade).order_by(SimTrade.created_at.desc()))
            assert trade is not None
            assert trade.created_at.tzinfo is not None
            from datetime import timezone as tz
            assert trade.created_at.utcoffset() == tz.utc.utcoffset(None)

    async def test_current_month_is_vienna_local(self, monkeypatch):
        """_current_month() must return a YYYY-MM string (Vienna calendar).

        This is the one intentional exception to the UTC-everywhere rule:
        the monthly allowance deposit is a calendar-month concept, so it
        must follow the operator's local timezone, not UTC.

        We verify the boundary: 23:30 UTC on 2026-01-31 is 00:30 on
        2026-02-01 in Vienna, so _current_month() must return '2026-02'.
        """
        from datetime import datetime, timezone
        from zoneinfo import ZoneInfo

        vienna = ZoneInfo("Europe/Vienna")
        fake_utc = datetime(2026, 1, 31, 23, 30, tzinfo=timezone.utc)
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
# Chat-driven trade execution (sim_chat)
# ---------------------------------------------------------------------------

class TestSimChatPartialSell:
    """Tests that sim_chat honours partial SELL sizes from the action block.

    These tests mock the LLM HTTP call and the portfolio context builder,
    then verify that _exec_sell is called with the correct share count
    (not None, which would liquidate the entire position).
    """

    async def test_sell_with_amount_trims_position(self, with_cash, monkeypatch):
        """SELL with 'amount' should sell only the equivalent shares, not all."""
        # Setup: buy 100 shares of AAPL at $100 = $10,000
        await sim._exec_buy("AAPL", 100.0, 10000.0, "initial buy")
        assert 10000.0 >= 1  # sanity

        # Mock _latest_close to return $100 for AAPL
        async def mock_close(ticker):
            return 100.0 if ticker == "AAPL" else None
        monkeypatch.setattr(sim, "_latest_close", mock_close)

        # Mock _build_sim_chat_context to return minimal context
        async def mock_context():
            return "mock context"
        monkeypatch.setattr(sim, "_build_sim_chat_context", mock_context)

        # Mock the LLM HTTP response to return a partial SELL of $3000
        llm_response_text = (
            'Trimming AAPL by $3000.\n'
            '[[ACTION]]\n'
            '{"actions": [{"ticker": "AAPL", "action": "SELL", "amount": 3000.0, "reason": "trim"}]}\n'
            '[[/ACTION]]'
        )

        class MockResponse:
            status_code = 200
            def raise_for_status(self):
                pass
            def json(self):
                return {"message": {"content": llm_response_text}}

        class MockClient:
            async def __aenter__(self):
                return self
            async def __aexit__(self, *args):
                pass
            async def post(self, url, json=None):
                return MockResponse()

        monkeypatch.setattr(sim.httpx, "AsyncClient", lambda **kw: MockClient())

        result = await sim.sim_chat([{"role": "user", "content": "trim AAPL by $3000"}])

        assert result["actions_executed"] is True
        assert len(result["trades"]) == 1
        trade = result["trades"][0]
        assert trade["ticker"] == "AAPL"
        assert trade["side"] == "SELL"
        # $3000 / $100 = 30 shares sold
        assert trade["shares"] == pytest.approx(30.0, abs=0.001)

        # Verify the position still exists with ~70 shares remaining
        async with sim.Session() as s:
            from sqlalchemy import select as sa_select
            pos = await s.scalar(sa_select(SimPosition).where(SimPosition.ticker == "AAPL"))
            assert pos is not None, "Position should still exist after partial sell"
            assert pos.shares == pytest.approx(70.0, abs=0.001)

    async def test_sell_with_shares_trims_position(self, with_cash, monkeypatch):
        """SELL with 'shares' should sell exactly that many shares."""
        await sim._exec_buy("AAPL", 100.0, 10000.0, "initial buy")

        async def mock_close(ticker):
            return 100.0 if ticker == "AAPL" else None
        monkeypatch.setattr(sim, "_latest_close", mock_close)

        async def mock_context():
            return "mock context"
        monkeypatch.setattr(sim, "_build_sim_chat_context", mock_context)

        llm_response_text = (
            'Selling 25 shares.\n'
            '[[ACTION]]\n'
            '{"actions": [{"ticker": "AAPL", "action": "SELL", "shares": 25.0, "reason": "trim"}]}\n'
            '[[/ACTION]]'
        )

        class MockResponse:
            status_code = 200
            def raise_for_status(self):
                pass
            def json(self):
                return {"message": {"content": llm_response_text}}

        class MockClient:
            async def __aenter__(self):
                return self
            async def __aexit__(self, *args):
                pass
            async def post(self, url, json=None):
                return MockResponse()

        monkeypatch.setattr(sim.httpx, "AsyncClient", lambda **kw: MockClient())

        result = await sim.sim_chat([{"role": "user", "content": "sell 25 shares of AAPL"}])

        assert result["actions_executed"] is True
        assert len(result["trades"]) == 1
        assert result["trades"][0]["shares"] == pytest.approx(25.0, abs=0.001)

        async with sim.Session() as s:
            from sqlalchemy import select as sa_select
            pos = await s.scalar(sa_select(SimPosition).where(SimPosition.ticker == "AAPL"))
            assert pos is not None
            assert pos.shares == pytest.approx(75.0, abs=0.001)

    async def test_sell_without_size_liquidates_all(self, with_cash, monkeypatch):
        """SELL without 'shares' or 'amount' should sell the entire position (backward compat)."""
        await sim._exec_buy("AAPL", 100.0, 5000.0, "initial buy")  # 50 shares

        async def mock_close(ticker):
            return 100.0 if ticker == "AAPL" else None
        monkeypatch.setattr(sim, "_latest_close", mock_close)

        async def mock_context():
            return "mock context"
        monkeypatch.setattr(sim, "_build_sim_chat_context", mock_context)

        llm_response_text = (
            'Selling all AAPL.\n'
            '[[ACTION]]\n'
            '{"actions": [{"ticker": "AAPL", "action": "SELL", "reason": "exit"}]}\n'
            '[[/ACTION]]'
        )

        class MockResponse:
            status_code = 200
            def raise_for_status(self):
                pass
            def json(self):
                return {"message": {"content": llm_response_text}}

        class MockClient:
            async def __aenter__(self):
                return self
            async def __aexit__(self, *args):
                pass
            async def post(self, url, json=None):
                return MockResponse()

        monkeypatch.setattr(sim.httpx, "AsyncClient", lambda **kw: MockClient())

        result = await sim.sim_chat([{"role": "user", "content": "sell all AAPL"}])

        assert result["actions_executed"] is True
        assert len(result["trades"]) == 1
        assert result["trades"][0]["shares"] == pytest.approx(50.0, abs=0.001)

        async with sim.Session() as s:
            from sqlalchemy import select as sa_select
            pos = await s.scalar(sa_select(SimPosition).where(SimPosition.ticker == "AAPL"))
            assert pos is None, "Position should be fully liquidated"

    async def test_sell_amount_exceeding_position_sells_all(self, with_cash, monkeypatch):
        """If 'amount' exceeds the position value, sell the entire position."""
        await sim._exec_buy("AAPL", 100.0, 3000.0, "initial buy")  # 30 shares = $3000

        async def mock_close(ticker):
            return 100.0 if ticker == "AAPL" else None
        monkeypatch.setattr(sim, "_latest_close", mock_close)

        async def mock_context():
            return "mock context"
        monkeypatch.setattr(sim, "_build_sim_chat_context", mock_context)

        llm_response_text = (
            '[[ACTION]]\n'
            '{"actions": [{"ticker": "AAPL", "action": "SELL", "amount": 99999.0, "reason": "over"}]}\n'
            '[[/ACTION]]'
        )

        class MockResponse:
            status_code = 200
            def raise_for_status(self):
                pass
            def json(self):
                return {"message": {"content": llm_response_text}}

        class MockClient:
            async def __aenter__(self):
                return self
            async def __aexit__(self, *args):
                pass
            async def post(self, url, json=None):
                return MockResponse()

        monkeypatch.setattr(sim.httpx, "AsyncClient", lambda **kw: MockClient())

        result = await sim.sim_chat([{"role": "user", "content": "sell AAPL"}])

        assert result["actions_executed"] is True
        assert len(result["trades"]) == 1
        # _exec_sell clamps to pos.shares
        assert result["trades"][0]["shares"] == pytest.approx(30.0, abs=0.001)


class TestSimChatPartialBuy:
    """Tests that sim_chat honours partial BUY sizes from the action block."""

    async def test_buy_with_amount(self, with_cash, monkeypatch):
        """BUY with 'amount' should invest at most that many dollars."""
        # Raise max position % so the $2000 amount isn't clamped to $1500
        monkeypatch.setattr(settings, "sim_max_position_pct", 30.0)

        async def mock_close(ticker):
            return 100.0 if ticker == "AAPL" else None
        monkeypatch.setattr(sim, "_latest_close", mock_close)

        async def mock_context():
            return "mock context"
        monkeypatch.setattr(sim, "_build_sim_chat_context", mock_context)

        llm_response_text = (
            'Buying $2000 of AAPL.\n'
            '[[ACTION]]\n'
            '{"actions": [{"ticker": "AAPL", "action": "BUY", "amount": 2000.0, "reason": "dip"}]}\n'
            '[[/ACTION]]'
        )

        class MockResponse:
            status_code = 200
            def raise_for_status(self):
                pass
            def json(self):
                return {"message": {"content": llm_response_text}}

        class MockClient:
            async def __aenter__(self):
                return self
            async def __aexit__(self, *args):
                pass
            async def post(self, url, json=None):
                return MockResponse()

        monkeypatch.setattr(sim.httpx, "AsyncClient", lambda **kw: MockClient())

        result = await sim.sim_chat([{"role": "user", "content": "buy $2000 of AAPL"}])

        assert result["actions_executed"] is True
        assert len(result["trades"]) == 1
        trade = result["trades"][0]
        assert trade["ticker"] == "AAPL"
        assert trade["side"] == "BUY"
        # $2000 / $100 = 20 shares
        assert trade["shares"] == pytest.approx(20.0, abs=0.001)

    async def test_buy_with_shares(self, with_cash, monkeypatch):
        """BUY with 'shares' should buy at most that many shares (clamped to budget)."""
        # Raise max position % so the requested 15 shares aren't cash/max-clamped.
        monkeypatch.setattr(settings, "sim_max_position_pct", 30.0)

        async def mock_close(ticker):
            return 100.0 if ticker == "AAPL" else None
        monkeypatch.setattr(sim, "_latest_close", mock_close)

        async def mock_context():
            return "mock context"
        monkeypatch.setattr(sim, "_build_sim_chat_context", mock_context)

        llm_response_text = (
            '[[ACTION]]\n'
            '{"actions": [{"ticker": "AAPL", "action": "BUY", "shares": 15.0, "reason": "dip"}]}\n'
            '[[/ACTION]]'
        )

        class MockResponse:
            status_code = 200
            def raise_for_status(self):
                pass
            def json(self):
                return {"message": {"content": llm_response_text}}

        class MockClient:
            async def __aenter__(self):
                return self
            async def __aexit__(self, *args):
                pass
            async def post(self, url, json=None):
                return MockResponse()

        monkeypatch.setattr(sim.httpx, "AsyncClient", lambda **kw: MockClient())

        result = await sim.sim_chat([{"role": "user", "content": "buy 15 shares of AAPL"}])

        assert result["actions_executed"] is True
        assert len(result["trades"]) == 1
        # 15 shares * $100 = $1500 budget → 15 shares bought
        assert result["trades"][0]["shares"] == pytest.approx(15.0, abs=0.001)

    async def test_buy_amount_clamped_to_max_budget(self, with_cash, monkeypatch):
        """BUY 'amount' exceeding the risk-limited budget should be clamped."""
        # Cash = $10,000, min_cash = 5% = $500, so max spend = $9,500
        # max_position = 10% of $10,000 = $1,000
        # Requesting $5,000 should be clamped to $1,000
        async def mock_close(ticker):
            return 100.0 if ticker == "AAPL" else None
        monkeypatch.setattr(sim, "_latest_close", mock_close)

        async def mock_context():
            return "mock context"
        monkeypatch.setattr(sim, "_build_sim_chat_context", mock_context)

        llm_response_text = (
            '[[ACTION]]\n'
            '{"actions": [{"ticker": "AAPL", "action": "BUY", "amount": 5000.0, "reason": "too much"}]}\n'
            '[[/ACTION]]'
        )

        class MockResponse:
            status_code = 200
            def raise_for_status(self):
                pass
            def json(self):
                return {"message": {"content": llm_response_text}}

        class MockClient:
            async def __aenter__(self):
                return self
            async def __aexit__(self, *args):
                pass
            async def post(self, url, json=None):
                return MockResponse()

        monkeypatch.setattr(sim.httpx, "AsyncClient", lambda **kw: MockClient())

        result = await sim.sim_chat([{"role": "user", "content": "buy $5000 of AAPL"}])

        assert result["actions_executed"] is True
        assert len(result["trades"]) == 1
        # max_position = 10% of $10,000 = $1,000 → 10 shares at $100
        assert result["trades"][0]["shares"] == pytest.approx(10.0, abs=0.001)

    async def test_buy_without_amount_uses_max_budget(self, with_cash, monkeypatch):
        """BUY without 'amount' or 'shares' should use the max allowed budget."""
        async def mock_close(ticker):
            return 100.0 if ticker == "AAPL" else None
        monkeypatch.setattr(sim, "_latest_close", mock_close)

        async def mock_context():
            return "mock context"
        monkeypatch.setattr(sim, "_build_sim_chat_context", mock_context)

        llm_response_text = (
            '[[ACTION]]\n'
            '{"actions": [{"ticker": "AAPL", "action": "BUY", "reason": "dip"}]}\n'
            '[[/ACTION]]'
        )

        class MockResponse:
            status_code = 200
            def raise_for_status(self):
                pass
            def json(self):
                return {"message": {"content": llm_response_text}}

        class MockClient:
            async def __aenter__(self):
                return self
            async def __aexit__(self, *args):
                pass
            async def post(self, url, json=None):
                return MockResponse()

        monkeypatch.setattr(sim.httpx, "AsyncClient", lambda **kw: MockClient())

        result = await sim.sim_chat([{"role": "user", "content": "buy AAPL"}])

        assert result["actions_executed"] is True
        assert len(result["trades"]) == 1
        # max_position = 10% of $10,000 = $1,000 → 10 shares at $100
        assert result["trades"][0]["shares"] == pytest.approx(10.0, abs=0.001)


class TestSimChatReserveOverride:
    """Tests that sim_chat can spend the cash reserve only with use_reserve."""

    async def test_buy_without_reserve_is_clamped(self, with_cash, monkeypatch):
        """A normal BUY (no use_reserve) cannot spend below the 5% cash floor."""
        # Raise max position so the cash floor is the only clamp in play.
        monkeypatch.setattr(settings, "sim_max_position_pct", 100.0)

        async def mock_close(ticker):
            return 100.0 if ticker == "AAPL" else None
        monkeypatch.setattr(sim, "_latest_close", mock_close)

        async def mock_context():
            return "mock context"
        monkeypatch.setattr(sim, "_build_sim_chat_context", mock_context)

        llm_response_text = (
            '[[ACTION]]\n'
            '{"actions": [{"ticker": "AAPL", "action": "BUY", "reason": "dip"}]}\n'
            '[[/ACTION]]'
        )

        class MockResponse:
            status_code = 200
            def raise_for_status(self):
                pass
            def json(self):
                return {"message": {"content": llm_response_text}}

        class MockClient:
            async def __aenter__(self):
                return self
            async def __aexit__(self, *args):
                pass
            async def post(self, url, json=None):
                return MockResponse()

        monkeypatch.setattr(sim.httpx, "AsyncClient", lambda **kw: MockClient())

        result = await sim.sim_chat([{"role": "user", "content": "buy AAPL"}])

        assert result["actions_executed"] is True
        assert len(result["trades"]) == 1
        # min_cash = 5% of $10,000 = $500 → budget = $9,500 → 95 shares @ $100
        assert result["trades"][0]["shares"] == pytest.approx(95.0, abs=0.001)

    async def test_buy_with_reserve_spends_dry_powder(self, with_cash, monkeypatch):
        """A BUY with use_reserve=true can spend below the cash floor (to $0)."""
        monkeypatch.setattr(settings, "sim_max_position_pct", 100.0)

        async def mock_close(ticker):
            return 100.0 if ticker == "AAPL" else None
        monkeypatch.setattr(sim, "_latest_close", mock_close)

        async def mock_context():
            return "mock context"
        monkeypatch.setattr(sim, "_build_sim_chat_context", mock_context)

        llm_response_text = (
            '[[ACTION]]\n'
            '{"actions": [{"ticker": "AAPL", "action": "BUY", "use_reserve": true, "reason": "spend dry powder"}]}\n'
            '[[/ACTION]]'
        )

        class MockResponse:
            status_code = 200
            def raise_for_status(self):
                pass
            def json(self):
                return {"message": {"content": llm_response_text}}

        class MockClient:
            async def __aenter__(self):
                return self
            async def __aexit__(self, *args):
                pass
            async def post(self, url, json=None):
                return MockResponse()

        monkeypatch.setattr(sim.httpx, "AsyncClient", lambda **kw: MockClient())

        result = await sim.sim_chat([{"role": "user", "content": "spend the dry powder on AAPL"}])

        assert result["actions_executed"] is True
        assert len(result["trades"]) == 1
        # floor = $0 → budget = $10,000 → 100 shares @ $100
        assert result["trades"][0]["shares"] == pytest.approx(100.0, abs=0.001)


class TestSimChatMultipleActions:
    """Tests that sim_chat correctly processes multiple actions in one response."""

    async def test_sell_then_buy(self, with_cash, monkeypatch):
        """Sell one ticker, then buy another — both should execute."""
        # Raise max position % so the $2000 SPY buy isn't clamped to $1500
        monkeypatch.setattr(settings, "sim_max_position_pct", 30.0)

        # Setup: buy MDB and ANET
        await sim._exec_buy("MDB", 100.0, 5000.0, "initial")  # 50 shares MDB
        await sim._exec_buy("ANET", 50.0, 2000.0, "initial")  # 40 shares ANET

        async def mock_close(ticker):
            prices = {"MDB": 100.0, "ANET": 50.0, "SPY": 400.0}
            return prices.get(ticker)
        monkeypatch.setattr(sim, "_latest_close", mock_close)

        async def mock_context():
            return "mock context"
        monkeypatch.setattr(sim, "_build_sim_chat_context", mock_context)

        llm_response_text = (
            'Trim MDB by $2000 and buy SPY.\n'
            '[[ACTION]]\n'
            '{"actions": [\n'
            '  {"ticker": "MDB", "action": "SELL", "amount": 2000.0, "reason": "trim"},\n'
            '  {"ticker": "SPY", "action": "BUY", "amount": 2000.0, "reason": "diversify"}\n'
            ']}\n'
            '[[/ACTION]]'
        )

        class MockResponse:
            status_code = 200
            def raise_for_status(self):
                pass
            def json(self):
                return {"message": {"content": llm_response_text}}

        class MockClient:
            async def __aenter__(self):
                return self
            async def __aexit__(self, *args):
                pass
            async def post(self, url, json=None):
                return MockResponse()

        monkeypatch.setattr(sim.httpx, "AsyncClient", lambda **kw: MockClient())

        result = await sim.sim_chat([{"role": "user", "content": "trim MDB to buy SPY"}])

        assert result["actions_executed"] is True
        assert len(result["trades"]) == 2
        # First trade: SELL MDB $2000 / $100 = 20 shares
        assert result["trades"][0]["ticker"] == "MDB"
        assert result["trades"][0]["side"] == "SELL"
        assert result["trades"][0]["shares"] == pytest.approx(20.0, abs=0.001)
        # Second trade: BUY SPY $2000 / $400 = 5 shares
        assert result["trades"][1]["ticker"] == "SPY"
        assert result["trades"][1]["side"] == "BUY"
        assert result["trades"][1]["shares"] == pytest.approx(5.0, abs=0.001)

    async def test_no_action_block_returns_text_only(self, with_cash, monkeypatch):
        """If the LLM doesn't include an action block, just return the text."""
        async def mock_context():
            return "mock context"
        monkeypatch.setattr(sim, "_build_sim_chat_context", mock_context)

        llm_response_text = 'I think the portfolio looks good right now. No changes needed.'

        class MockResponse:
            status_code = 200
            def raise_for_status(self):
                pass
            def json(self):
                return {"message": {"content": llm_response_text}}

        class MockClient:
            async def __aenter__(self):
                return self
            async def __aexit__(self, *args):
                pass
            async def post(self, url, json=None):
                return MockResponse()

        monkeypatch.setattr(sim.httpx, "AsyncClient", lambda **kw: MockClient())

        result = await sim.sim_chat([{"role": "user", "content": "what do you think?"}])

        assert result["actions_executed"] is False
        assert result["trades"] == []
        assert result["text"] == llm_response_text
