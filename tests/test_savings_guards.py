"""NaN/inf input guards for the savings tracker (app/savings.py).

Pydantic v2 accepts JSON ``1e999`` -> inf and the ``NaN`` literal through
plain ``float`` request fields (see the *Request models in app/main.py),
and the guards in front of the entry points used to be `<= 0` / `< 0.005`
comparisons that non-finites PASS — one inf deposit poisoned cash inf→NaN
permanently. Every entry point now validates finiteness (reusing the
``_positive_amount`` / new ``_finite_number`` helpers) BEFORE any mutation
and answers in the existing ``{"ok": False, "reason": ...}`` shape.

Covered:

- each entry point (initialize, trueup, add_deposit, add_saveback,
  manual_buy, add_plan, set_config) refuses 1e999/inf and NaN with no
  mutation;
- the end-to-end poisoning pin: an inf deposit refused through the HTTP
  API (raw ``1e999`` JSON body) leaves cash unchanged and books no event;
- finite happy paths still work (positive amounts, true-up delta,
  config update, plan CRUD, position seeding).
"""

from __future__ import annotations

from datetime import UTC, datetime

import httpx
import pytest
from sqlalchemy import StaticPool, select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app import main, savings
from app.db import (
    Base,
    Candle,
    SavingsAccount,
    SavingsEvent,
    SavingsPlan,
)

INF = float("inf")
NAN = float("nan")


# ---------------------------------------------------------------------------
# Fixtures / helpers (same in-memory DB pattern as the other savings tests)
# ---------------------------------------------------------------------------


@pytest.fixture
async def mem_db(monkeypatch):
    """Point savings + the shared db/market modules at one in-memory DB."""
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    import app.db as db_mod
    import app.market as market_mod

    monkeypatch.setattr(db_mod, "Session", session_factory)
    monkeypatch.setattr(savings, "Session", session_factory)
    monkeypatch.setattr(market_mod, "Session", session_factory)
    yield session_factory
    await engine.dispose()


@pytest.fixture
async def client(mem_db):
    transport = httpx.ASGITransport(app=main.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


async def _seed_candle(session_factory, ticker: str, close: float) -> None:
    async with session_factory() as s:
        s.add(Candle(ticker=ticker, timestamp=datetime.now(UTC),
                     open=close, high=close, low=close, close=close, volume=0))
        await s.commit()


async def _cash(mem_db) -> float:
    async with mem_db() as s:
        acc = await s.get(SavingsAccount, 1)
        return float(acc.cash) if acc else float("nan")


async def _events(mem_db, kind: str | None = None) -> list[SavingsEvent]:
    async with mem_db() as s:
        stmt = select(SavingsEvent).order_by(SavingsEvent.id)
        if kind:
            stmt = stmt.where(SavingsEvent.kind == kind)
        return list((await s.scalars(stmt)).all())


# ---------------------------------------------------------------------------
# initialize
# ---------------------------------------------------------------------------


class TestInitializeGuards:
    @pytest.mark.parametrize("bad", [INF, NAN])
    async def test_initialize_refuses_nonfinite_cash_without_mutation(self, mem_db, bad):
        r = await savings.initialize(bad)
        assert r["ok"] is False and "finite" in r["reason"]
        assert await savings.initialized() is False  # nothing was seeded

    @pytest.mark.parametrize("field", ["interest_rate", "monthly_transfer"])
    @pytest.mark.parametrize("bad", [INF, NAN])
    async def test_initialize_refuses_nonfinite_config_fields(self, mem_db, field, bad):
        r = await savings.initialize(100.0, **{field: bad})
        assert r["ok"] is False and "finite" in r["reason"]
        assert await savings.initialized() is False

    async def test_initialize_skips_nonfinite_position_lots(self, mem_db):
        r = await savings.initialize(
            100.0,
            positions=[
                {"ticker": "URTH", "shares": INF, "avg_cost": 100.0},   # skipped
                {"ticker": "SPY", "shares": 2.0, "avg_cost": NAN},       # skipped
                {"ticker": "AAPL", "shares": 1.0, "avg_cost": 50.0},     # kept
            ],
        )
        assert r["ok"] is True
        assert r["invested"] == 50.0  # only the finite lot

    async def test_initialize_finite_happy_path(self, mem_db):
        r = await savings.initialize(250.0, monthly_transfer=100.0, interest_rate=2.5)
        assert r["ok"] is True and r["cash"] == 250.0
        assert await _cash(mem_db) == 250.0


# ---------------------------------------------------------------------------
# trueup / add_deposit / add_saveback / manual_buy / add_plan / set_config
# ---------------------------------------------------------------------------


class TestOperatorActionGuards:
    @pytest.mark.parametrize("bad", [INF, -INF, NAN])
    async def test_trueup_refuses_nonfinite_total(self, mem_db, bad):
        await savings.initialize(100.0)
        r = await savings.trueup(bad)
        assert r["ok"] is False and "finite" in r["reason"]
        assert await _cash(mem_db) == 100.0            # unchanged
        assert await _events(mem_db, "TRUEUP") == []     # no event booked

    @pytest.mark.parametrize("bad", [INF, -INF, NAN, 0.0, -5.0])
    async def test_add_deposit_refuses_unusable_amounts(self, mem_db, bad):
        await savings.initialize(100.0)
        r = await savings.add_deposit(bad)
        assert r["ok"] is False
        assert await _cash(mem_db) == 100.0
        assert await _events(mem_db, "TRANSFER") == []

    @pytest.mark.parametrize("bad", [INF, NAN])
    async def test_add_saveback_refuses_nonfinite_amount(self, mem_db, bad):
        await savings.initialize(100.0)
        await _seed_candle(mem_db, "SPY", 50.0)
        r = await savings.add_saveback(bad, "SPY")
        assert r["ok"] is False
        assert await _events(mem_db, "SAVEBACK") == []

    @pytest.mark.parametrize("bad", [INF, NAN])
    async def test_manual_buy_refuses_nonfinite_amount(self, mem_db, bad):
        await savings.initialize(100.0)
        await _seed_candle(mem_db, "SPY", 50.0)
        r = await savings.manual_buy(bad, "SPY")
        assert r["ok"] is False
        assert await _cash(mem_db) == 100.0
        assert await _events(mem_db, "MANUAL_BUY") == []

    @pytest.mark.parametrize("bad", [INF, NAN])
    async def test_add_plan_refuses_nonfinite_amount(self, mem_db, bad):
        await savings.initialize(100.0)
        r = await savings.add_plan("URTH", bad, 1)
        assert r["ok"] is False
        async with mem_db() as s:
            assert (await s.scalars(select(SavingsPlan))).all() == []

    async def test_add_plan_refuses_nonfinite_day(self, mem_db):
        await savings.initialize(100.0)
        for bad_day in (INF, NAN):
            r = await savings.add_plan("URTH", 100.0, bad_day)
            assert r["ok"] is False
        async with mem_db() as s:
            assert (await s.scalars(select(SavingsPlan))).all() == []

    @pytest.mark.parametrize("field", ["monthly_transfer", "interest_rate"])
    @pytest.mark.parametrize("bad", [INF, NAN])
    async def test_set_config_refuses_nonfinite_values(self, mem_db, field, bad):
        await savings.initialize(100.0, monthly_transfer=10.0, interest_rate=1.0)
        r = await savings.set_config(**{field: bad})
        assert r["ok"] is False and "finite" in r["reason"]
        async with mem_db() as s:
            acc = await s.get(SavingsAccount, 1)
            assert acc.monthly_transfer == 10.0 and acc.interest_rate == 1.0  # unchanged

    async def test_finite_happy_paths(self, mem_db):
        await savings.initialize(100.0)
        await _seed_candle(mem_db, "SPY", 50.0)
        await _seed_candle(mem_db, "EURUSD=X", 1.0)  # US ticker prices 1:1 in €
        assert (await savings.trueup(150.0))["ok"] is True
        assert (await savings.add_deposit(50.0))["ok"] is True
        assert (await savings.manual_buy(30.0, "SPY"))["ok"] is True
        assert (await savings.add_saveback(10.0, "SPY"))["ok"] is True
        assert (await savings.add_plan("SPY", 100.0, 15))["ok"] is True
        assert (await savings.set_config(monthly_transfer=300.0, interest_rate=2.0))["ok"] is True
        v = await savings.valuate()
        assert v["cash"] == 170.0  # 100 + 50 (trueup) + 50 (deposit) - 30 (buy)


# ---------------------------------------------------------------------------
# The end-to-end poisoning pin, through the HTTP API (raw 1e999 / NaN JSON)
# ---------------------------------------------------------------------------


class TestApiPoisoningPin:
    async def test_inf_deposit_via_api_is_refused_and_cash_unchanged(self, client, mem_db):
        r = await client.post("/api/savings/initialize", json={"cash": 100.0})
        assert r.status_code == 200
        # Raw JSON body: 1e999 parses to inf through pydantic's float field.
        r = await client.post("/api/savings/deposit", content=b'{"amount": 1e999}',
                              headers={"content-type": "application/json"})
        assert r.status_code == 422
        assert await _cash(mem_db) == 100.0
        assert await _events(mem_db, "TRANSFER") == []
        # NaN literal likewise.
        r = await client.post("/api/savings/deposit", content=b'{"amount": NaN}',
                              headers={"content-type": "application/json"})
        assert r.status_code == 422
        assert await _cash(mem_db) == 100.0
        # The tracker is not poisoned: a finite deposit still works.
        r = await client.post("/api/savings/deposit", json={"amount": 25.0})
        assert r.status_code == 200
        assert await _cash(mem_db) == 125.0

    async def test_inf_trueup_and_initialize_via_api_are_refused(self, client, mem_db):
        r = await client.post("/api/savings/initialize", content=b'{"cash": 1e999}',
                              headers={"content-type": "application/json"})
        assert r.status_code == 422
        assert await savings.initialized() is False
        await client.post("/api/savings/initialize", json={"cash": 100.0})
        r = await client.post("/api/savings/trueup", content=b'{"actual_cash": 1e999}',
                              headers={"content-type": "application/json"})
        assert r.status_code == 422
        assert await _cash(mem_db) == 100.0
