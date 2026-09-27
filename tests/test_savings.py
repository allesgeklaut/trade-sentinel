"""Tests for the savings tracker (app.savings).

These tests use an in-memory SQLite database so they don't touch the real
data volume. They cover:

- Initialization (seed cash + positions, month markers, mirror seeding)
- EUR pricing (native-EUR suffixes, USD conversion via EURUSD=X,
  non-EUR local listings like .SW)
- Interest accrual math (actual/365, catch-up after downtime) and the
  monthly payout (idempotent, seeded-month skip)
- Monthly transfer deposits (once per month, idempotent)
- Sparplan execution (due-day logic, month dedupe via the event ledger,
  insufficient-cash WARN, day-31 clamp in short months, deferred when the
  price is missing)
- True-up / Saveback / config / plan CRUD
- The daily-core forward mirror (allocation, deferred without a ranking,
  valuation)
- Reset isolation: savings reset never touches the sim tables and
  reset_sim never touches the savings tables
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

import pytest
from sqlalchemy import StaticPool, select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app import savings
from app.config import settings
from app.db import (
    Base,
    Candle,
    DailyCoreRanking,
    SavingsAccount,
    SavingsEvent,
    SavingsMirrorEntry,
    SavingsMirrorPosition,
    SavingsPlan,
    SavingsPosition,
    SimAccount,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
async def mem_db(monkeypatch):
    """Replace the savings engine's database with an in-memory SQLite DB."""
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    session_factory = async_sessionmaker(engine, expire_on_commit=False)

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    import app.db as db_mod

    monkeypatch.setattr(db_mod, "Session", session_factory)
    monkeypatch.setattr(savings, "Session", session_factory)
    # _price_eur_map delegates to market.latest_close, which uses the
    # market module's own Session import — patch that too.
    import app.market as market_mod

    monkeypatch.setattr(market_mod, "Session", session_factory)
    yield session_factory
    await engine.dispose()


async def _seed_candle(
    session_factory, ticker: str, close: float, when: datetime | None = None
) -> None:
    when = when or datetime.now(UTC)
    async with session_factory() as s:
        s.add(
            Candle(
                ticker=ticker,
                timestamp=when,
                open=close,
                high=close,
                low=close,
                close=close,
                volume=0,
            )
        )
        await s.commit()


async def _seed_ranking(session_factory, picks: list[str], age_hours: float = 1.0) -> None:
    async with session_factory() as s:
        s.add(
            DailyCoreRanking(
                id=1,
                ranking_date=datetime.now(UTC) - timedelta(hours=age_hours),
                band=",".join(picks),
                picks=",".join(picks),
            )
        )
        await s.commit()


async def _backdate_plan(session_factory, ticker: str, created: date) -> None:
    """Set every plan row for `ticker` created_at (tz-aware UTC) — the
    created-after-due-day rule keys off it, and tests with synthetic
    calendar dates need plans that 'existed' before those dates."""
    async with session_factory() as s:
        plans = (
            await s.scalars(select(SavingsPlan).where(SavingsPlan.ticker == ticker))
        ).all()
        assert plans, f"no plan for {ticker}"
        for plan in plans:
            plan.created_at = datetime(
                created.year, created.month, created.day, 12, 0, tzinfo=UTC
            )
        await s.commit()


async def _events(mem_db, kind: str | None = None) -> list[SavingsEvent]:
    async with mem_db() as s:
        stmt = select(SavingsEvent)
        if kind:
            stmt = stmt.where(SavingsEvent.kind == kind)
        return list((await s.scalars(stmt)).all())


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------


async def test_initialize_seeds_cash_positions_and_markers(mem_db, monkeypatch):
    monkeypatch.setattr(settings, "savings_interest_rate", 3.0)
    monkeypatch.setattr(settings, "savings_monthly_transfer", 200.0)
    await _seed_ranking(mem_db, ["AAPL", "MSFT", "NVDA"])
    await _seed_candle(mem_db, "AAPL", 100.0)
    await _seed_candle(mem_db, "MSFT", 200.0)
    await _seed_candle(mem_db, "NVDA", 50.0)
    await _seed_candle(mem_db, "EURUSD=X", 1.0)  # US picks price 1:1 in €

    r = await savings.initialize(
        5000.0,
        positions=[{"ticker": "URTH", "shares": 2.0, "avg_cost": 120.0}],
        monthly_transfer=250.0,
        interest_rate=2.25,
    )

    assert r["ok"] is True
    assert r["cash"] == 5000.0
    assert r["invested"] == 240.0  # 2 × 120
    # The seeded €240 mirrors into the (priced) daily-core picks.
    assert r["mirror"]["allocated"] == 240.0
    assert r["mirror"]["picks"] == 3

    val = await savings.valuate()
    assert val["initialized"] is True
    assert val["cash"] == 5000.0
    assert val["interest_rate"] == 2.25
    assert val["monthly_transfer"] == 250.0
    assert val["contributed"] == 5000.0  # seed cash; positions were seeded too
    # The current month is marked done: no transfer/payout for the seed month.
    transfer = await savings.deposit_monthly_transfer()
    assert transfer["deposited"] is False
    payout = await savings.payout_interest()
    assert payout["paid"] is False


async def test_initialize_replaces_previous_state(mem_db):
    await savings.initialize(100.0)
    await savings.add_plan("URTH", 50.0, 1)
    r = await savings.initialize(9000.0)
    assert r["ok"] is True
    plans = await savings.get_plans()
    assert plans == []
    val = await savings.valuate()
    assert val["cash"] == 9000.0
    # Old INIT events were wiped — contributed counts only the new seed.
    assert val["contributed"] == 9000.0


async def test_all_actions_noop_when_uninitialized(mem_db):
    assert await savings.initialized() is False
    assert (await savings.deposit_monthly_transfer())["deposited"] is False
    assert (await savings.payout_interest())["paid"] is False
    assert (await savings.accrue_interest())["accrued"] is False
    assert (await savings.run_sparplans())["executed"] == 0
    assert (await savings.run_savings_cycle())["snapshot"]["snapshotted"] is False
    assert (await savings.set_config(100.0, 1.0))["ok"] is False
    assert (await savings.add_plan("URTH", 50.0, 1))["ok"] is False


# ---------------------------------------------------------------------------
# EUR pricing
# ---------------------------------------------------------------------------


async def test_price_eur_map_native_and_usd(mem_db):
    # Native EUR listing prices 1:1; US name converts via EURUSD=X.
    await _seed_candle(mem_db, "IFX.DE", 30.0)
    await _seed_candle(mem_db, "AAPL", 100.0)
    await _seed_candle(mem_db, "EURUSD=X", 1.25)
    prices = await savings._price_eur_map(["IFX.DE", "AAPL", "NOPE.DE"])
    assert prices["IFX.DE"] == 30.0
    assert prices["AAPL"] == pytest.approx(80.0)  # 100 / 1.25
    assert prices["NOPE.DE"] is None  # no candle


async def test_price_eur_map_swiss_listing_two_hop(mem_db):
    # .SW prices in CHF: convert CHF->USD via USDCHF=X (div), then USD->EUR.
    await _seed_candle(mem_db, "NESN.SW", 80.0)  # CHF
    await _seed_candle(mem_db, "USDCHF=X", 0.8)  # 1 USD = 0.8 CHF
    await _seed_candle(mem_db, "EURUSD=X", 1.6)  # 1 EUR = 1.6 USD
    prices = await savings._price_eur_map(["NESN.SW"])
    # 80 CHF / 0.8 = 100 USD; 100 / 1.6 = 62.5 EUR
    assert prices["NESN.SW"] == pytest.approx(62.5)


async def test_price_eur_map_missing_fx(mem_db):
    await _seed_candle(mem_db, "AAPL", 100.0)  # no EURUSD=X candle
    prices = await savings._price_eur_map(["AAPL"])
    assert prices["AAPL"] is None


# ---------------------------------------------------------------------------
# Interest
# ---------------------------------------------------------------------------


async def test_interest_accrual_daily_and_catchup(mem_db):
    await savings.initialize(3650.0, interest_rate=1.0)
    # First accrual only sets the marker — no invented back-interest.
    r = await savings.accrue_interest()
    assert r["accrued"] is True and r["days"] == 0 and r["amount"] == 0.0

    # Simulate 10 days of downtime by backdating the marker.
    async with mem_db() as s:
        acc = await s.get(SavingsAccount, 1)
        acc.last_accrual_day = (datetime.now(UTC).date() - timedelta(days=10)).isoformat()
        await s.commit()
    r = await savings.accrue_interest()
    assert r["accrued"] is True
    assert r["days"] == 10
    # 3650 € at 1% p.a. / 365 = 0.10/day → 1.00 over 10 days.
    assert r["amount"] == pytest.approx(1.00, abs=1e-6)
    assert r["total_accrued"] == pytest.approx(1.00, abs=1e-6)

    # Same-day re-run is a no-op.
    r2 = await savings.accrue_interest()
    assert r2["accrued"] is False


async def test_interest_accrues_on_cash_only(mem_db):
    await savings.initialize(1000.0, interest_rate=3.65)
    await _seed_candle(mem_db, "URTH", 100.0)
    await _seed_candle(mem_db, "EURUSD=X", 1.0)
    await savings.add_plan("URTH", 800.0, 1)
    await _backdate_plan(mem_db, "URTH", date.today().replace(day=1))
    r = await savings.run_sparplans()
    assert r["executed"] == 1
    # Backdate + accrue: 200 € cash at 3.65% = 0.02/day.
    async with mem_db() as s:
        acc = await s.get(SavingsAccount, 1)
        acc.last_accrual_day = (datetime.now(UTC).date() - timedelta(days=1)).isoformat()
        await s.commit()
    r = await savings.accrue_interest()
    assert r["amount"] == pytest.approx(0.02, abs=1e-6)


async def test_interest_payout_monthly_idempotent(mem_db):
    await savings.initialize(7300.0, interest_rate=5.0)
    async with mem_db() as s:
        acc = await s.get(SavingsAccount, 1)
        acc.last_accrual_day = (datetime.now(UTC).date() - timedelta(days=30)).isoformat()
        await s.commit()
    await savings.accrue_interest()  # 30 × 1.0 = 30 €
    async with mem_db() as s:
        acc = await s.get(SavingsAccount, 1)
        acc.last_interest_month = "2000-01"  # pretend an older month ran
        await s.commit()
    r = await savings.payout_interest()
    assert r["paid"] is True
    assert r["amount"] == pytest.approx(30.0)
    val = await savings.valuate()
    assert val["cash"] == pytest.approx(7330.0)
    assert val["accrued_interest"] == 0.0
    # Second payout this month: no-op.
    r2 = await savings.payout_interest()
    assert r2["paid"] is False
    # Ledger has the payout event.
    kinds = [e.kind for e in await _events(mem_db, "INTEREST_PAYOUT")]
    assert kinds == ["INTEREST_PAYOUT"]


# ---------------------------------------------------------------------------
# Monthly transfer
# ---------------------------------------------------------------------------


async def test_monthly_transfer_deposits_once(mem_db):
    await savings.initialize(100.0, monthly_transfer=500.0)
    r = await savings.deposit_monthly_transfer()
    # The seed month is marked done at initialization.
    assert r["deposited"] is False
    # Pretend the last transfer ran in an older month.
    async with mem_db() as s:
        acc = await s.get(SavingsAccount, 1)
        acc.last_transfer_month = "2000-01"
        await s.commit()
    r = await savings.deposit_monthly_transfer()
    assert r["deposited"] is True
    assert r["amount"] == 500.0
    val = await savings.valuate()
    assert val["cash"] == 600.0
    assert val["contributed"] == 600.0
    # Idempotent within the same month.
    r2 = await savings.deposit_monthly_transfer()
    assert r2["deposited"] is False
    events = await _events(mem_db, "TRANSFER")
    assert len(events) == 1


async def test_zero_transfer_marks_month_without_event(mem_db):
    await savings.initialize(100.0, monthly_transfer=0.0)
    async with mem_db() as s:
        acc = await s.get(SavingsAccount, 1)
        acc.last_transfer_month = "2000-01"
        await s.commit()
    r = await savings.deposit_monthly_transfer()
    assert r["deposited"] is False
    assert (await _events(mem_db, "TRANSFER")) == []


# ---------------------------------------------------------------------------
# Sparplan execution
# ---------------------------------------------------------------------------


async def test_sparplan_executes_on_due_day_only(mem_db):
    await savings.initialize(1000.0)
    await _seed_candle(mem_db, "URTH", 100.0)
    await _seed_candle(mem_db, "EURUSD=X", 1.0)
    await savings.add_plan("URTH", 100.0, 15)
    # The plan "existed" before its due day this month.
    await _backdate_plan(mem_db, "URTH", date(2026, 9, 1))

    # Day 10: not yet due.
    r = await savings.run_sparplans(day=date(2026, 9, 10))
    assert r["executed"] == 0
    # Day 15: due, executes.
    r = await savings.run_sparplans(day=date(2026, 9, 15))
    assert r["executed"] == 1
    assert r["trades"][0]["shares"] == pytest.approx(1.0)
    # Same month again: ledger dedupe blocks a second buy.
    r = await savings.run_sparplans(day=date(2026, 9, 20))
    assert r["executed"] == 0
    # Next month: due again.
    r = await savings.run_sparplans(day=date(2026, 10, 15))
    assert r["executed"] == 1
    val = await savings.valuate()
    assert val["cash"] == pytest.approx(800.0)
    assert len(val["positions"]) == 1
    assert val["positions"][0]["shares"] == pytest.approx(2.0)


async def test_sparplan_created_after_due_day_fires_next_month(mem_db):
    """The exact regression the live box hit: a plan added on the 27th with
    day_of_month=2 must NOT fire retroactively in September — first buy is
    02.10 (mirroring Trade Republic's own Sparplan semantics)."""
    await savings.initialize(1000.0)
    await _seed_candle(mem_db, "URTH", 100.0)
    await _seed_candle(mem_db, "EURUSD=X", 1.0)
    # Created 27.09. — after the 2nd of the current month.
    await savings.add_plan("URTH", 100.0, 2)
    await _backdate_plan(mem_db, "URTH", date(2026, 9, 27))

    # Same month, after the due day: NOT due (created after the 2nd).
    r = await savings.run_sparplans(day=date(2026, 9, 27))
    assert r["executed"] == 0
    r = await savings.run_sparplans(day=date(2026, 9, 30))
    assert r["executed"] == 0
    # Next month on the due day: fires.
    r = await savings.run_sparplans(day=date(2026, 10, 2))
    assert r["executed"] == 1
    # Only one buy total.
    val = await savings.valuate()
    assert val["cash"] == pytest.approx(900.0)


async def test_sparplan_created_before_due_day_catches_up(mem_db):
    """A plan that existed before its due day but was missed (app down on
    the 2nd) still catches up later in the SAME month."""
    await savings.initialize(1000.0)
    await _seed_candle(mem_db, "URTH", 100.0)
    await _seed_candle(mem_db, "EURUSD=X", 1.0)
    await savings.add_plan("URTH", 100.0, 2)
    await _backdate_plan(mem_db, "URTH", date(2026, 9, 1))

    r = await savings.run_sparplans(day=date(2026, 9, 27))
    assert r["executed"] == 1
    # Dedupe still applies within the month.
    r = await savings.run_sparplans(day=date(2026, 9, 28))
    assert r["executed"] == 0


async def test_sparplan_day31_clamps_in_short_month(mem_db):
    await savings.initialize(1000.0)
    await _seed_candle(mem_db, "URTH", 100.0)
    await _seed_candle(mem_db, "EURUSD=X", 1.0)
    await savings.add_plan("URTH", 100.0, 31)
    await _backdate_plan(mem_db, "URTH", date(2026, 2, 1))

    # Feb 28: the day-31 plan is due (clamped to month end).
    r = await savings.run_sparplans(day=date(2026, 2, 28))
    assert r["executed"] == 1


async def test_sparplan_insufficient_cash_warns(mem_db):
    await savings.initialize(50.0)
    await _seed_candle(mem_db, "URTH", 100.0)
    await _seed_candle(mem_db, "EURUSD=X", 1.0)
    await savings.add_plan("URTH", 100.0, 1)
    await _backdate_plan(mem_db, "URTH", date(2026, 9, 1))

    r = await savings.run_sparplans(day=date(2026, 9, 1))
    assert r["executed"] == 0
    warns = await _events(mem_db, "WARN")
    assert len(warns) == 1
    assert "insufficient" in warns[0].note
    # Cash untouched.
    val = await savings.valuate()
    assert val["cash"] == 50.0


async def test_sparplan_deferred_without_price(mem_db):
    await savings.initialize(1000.0)
    await savings.add_plan("NOPE.DE", 100.0, 1)  # no candle
    await _backdate_plan(mem_db, "NOPE.DE", date(2026, 9, 1))

    r = await savings.run_sparplans(day=date(2026, 9, 1))
    assert r["executed"] == 0
    # A candle arrives; the next pass catches up within the month.
    await _seed_candle(mem_db, "NOPE.DE", 25.0)
    r = await savings.run_sparplans(day=date(2026, 9, 5))
    assert r["executed"] == 1
    assert r["trades"][0]["shares"] == pytest.approx(4.0)


# ---------------------------------------------------------------------------
# Mirror
# ---------------------------------------------------------------------------


async def test_mirror_allocates_across_priced_picks(mem_db):
    await _seed_ranking(mem_db, ["AAPL", "MSFT", "NOPE"])
    await _seed_candle(mem_db, "AAPL", 100.0)
    await _seed_candle(mem_db, "MSFT", 50.0)
    await _seed_candle(mem_db, "EURUSD=X", 1.0)
    # NOPE has no candle — allocation spreads over the priced picks only.
    r = await savings._mirror_allocate(300.0, "URTH")
    assert r is not None
    assert r["pick_count"] == 2
    assert r["amount"] == 300.0
    val = await savings._mirror_valuate()
    assert val["allocated"] == 300.0
    assert val["positions_value"] == pytest.approx(300.0)
    by_ticker = {p["ticker"]: p for p in val["positions"]}
    assert by_ticker["AAPL"]["shares"] == pytest.approx(1.5)  # 150/100
    assert by_ticker["MSFT"]["shares"] == pytest.approx(3.0)  # 150/50


async def test_mirror_defers_without_fresh_ranking(mem_db):
    # No ranking row at all.
    r = await savings._mirror_allocate(300.0, "URTH")
    assert r is None
    # Stale ranking (> 4 days old): deferred too.
    await _seed_ranking(mem_db, ["AAPL"], age_hours=24 * 5)
    await _seed_candle(mem_db, "AAPL", 100.0)
    r = await savings._mirror_allocate(300.0, "URTH")
    assert r is None


async def test_mirror_avg_cost_on_second_buy(mem_db):
    await _seed_ranking(mem_db, ["AAPL"])
    await _seed_candle(mem_db, "AAPL", 100.0)
    await _seed_candle(mem_db, "EURUSD=X", 1.0)
    await savings._mirror_allocate(100.0, "URTH")  # 1 share @ 100
    await _seed_candle(mem_db, "AAPL", 200.0)
    await savings._mirror_allocate(100.0, "URTH")  # 0.5 share @ 200
    async with mem_db() as s:
        rows = list((await s.scalars(select(SavingsMirrorPosition))).all())
    assert len(rows) == 1
    assert rows[0].shares == pytest.approx(1.5)
    assert rows[0].avg_cost == pytest.approx((100 + 100) / 1.5)


async def test_mirror_entries_log_source(mem_db):
    await _seed_ranking(mem_db, ["AAPL"])
    await _seed_candle(mem_db, "AAPL", 100.0)
    await _seed_candle(mem_db, "EURUSD=X", 1.0)
    await savings._mirror_allocate(100.0, "URTH")
    async with mem_db() as s:
        rows = list((await s.scalars(select(SavingsMirrorEntry))).all())
    assert len(rows) == 1
    assert rows[0].amount == 100.0
    assert rows[0].source_ticker == "URTH"
    # Breakdown JSON: the picks bought with this entry.
    import json as _json
    bd = _json.loads(rows[0].breakdown)
    assert bd == [["AAPL", 1.0, 100.0]]


async def test_mirror_curve_replays_true_time_series(mem_db):
    """The mirror curve must value each date with that date's close, not
    today's value (no look-ahead, no flat line)."""
    from datetime import timedelta

    await _seed_ranking(mem_db, ["AAPL"])
    base = datetime.now(UTC) - timedelta(days=10)
    # AAPL rallies 100 -> 200 over the window (EURUSD 1:1).
    for i, px in enumerate((100.0, 150.0, 200.0)):
        await _seed_candle(mem_db, "AAPL", px, when=base + timedelta(days=i * 4))
        await _seed_candle(mem_db, "EURUSD=X", 1.0, when=base + timedelta(days=i * 4))
    # Entry on day 0: 1 share @ 100.
    async with mem_db() as s:
        s.add(savings.SavingsMirrorEntry(
            amount=100.0, source_ticker="", breakdown='[["AAPL", 1.0, 100.0]]',
            created_at=base))
        # Snapshots on day 0, 4, 8.
        for i in range(3):
            s.add(savings.SavingsSnapshot(
                cash=0.0, positions_value=0.0, total_equity=0.0, contributed=0.0,
                created_at=base + timedelta(days=i * 4)))
        await s.commit()
    curve = await savings.get_mirror_curve(100)
    assert len(curve) == 3
    assert [c["allocated"] for c in curve] == [100.0, 100.0, 100.0]
    # Valued at each date's OWN close: 100, 150, 200 — a rising line.
    assert [c["value"] for c in curve] == [100.0, 150.0, 200.0]


async def test_mirror_curve_carries_price_forward(mem_db):
    """Dates without a candle for a ticker are marked at the most recent
    prior close, not zero."""
    from datetime import timedelta

    await _seed_ranking(mem_db, ["AAPL"])
    base = datetime.now(UTC) - timedelta(days=10)
    # One candle early; snapshots on three consecutive days after it.
    await _seed_candle(mem_db, "AAPL", 100.0, when=base)
    await _seed_candle(mem_db, "EURUSD=X", 1.0, when=base)
    async with mem_db() as s:
        s.add(savings.SavingsMirrorEntry(
            amount=100.0, source_ticker="", breakdown='[["AAPL", 1.0, 100.0]]',
            created_at=base))
        for i in range(3):
            s.add(savings.SavingsSnapshot(
                cash=0.0, positions_value=0.0, total_equity=0.0, contributed=0.0,
                created_at=base + timedelta(days=i)))
        await s.commit()
    curve = await savings.get_mirror_curve(100)
    assert [c["value"] for c in curve] == [100.0, 100.0, 100.0]


# ---------------------------------------------------------------------------
# True-up, Saveback, config, plans CRUD
# ---------------------------------------------------------------------------


async def test_trueup_adjusts_delta(mem_db):
    await savings.initialize(100.0)
    r = await savings.trueup(150.0)
    assert r["ok"] is True
    assert r["delta"] == 50.0
    val = await savings.valuate()
    assert val["cash"] == 150.0
    # A no-op true-up books nothing.
    r = await savings.trueup(150.0)
    assert r["delta"] == 0.0
    events = await _events(mem_db, "TRUEUP")
    assert len(events) == 1


async def test_trueup_refuses_negative_cash(mem_db):
    await savings.initialize(100.0)
    # An entered total below the tracked positions' worth would drive the
    # tracked cash negative (100 cash − 250 delta) — refused, nothing booked.
    r = await savings.trueup(-50.0)
    assert r["ok"] is False
    assert "negative" in r["reason"]
    val = await savings.valuate()
    assert val["cash"] == 100.0  # untouched
    assert (await _events(mem_db, "TRUEUP")) == []


async def test_saveback_buys_position_and_mirrors(mem_db):
    await savings.initialize(100.0)
    await _seed_ranking(mem_db, ["AAPL"])
    await _seed_candle(mem_db, "SPY", 50.0)
    await _seed_candle(mem_db, "AAPL", 100.0)
    await _seed_candle(mem_db, "EURUSD=X", 1.0)
    r = await savings.add_saveback(10.0, "SPY")
    assert r["ok"] is True
    assert r["shares"] == pytest.approx(0.2)
    val = await savings.valuate()
    assert val["cash"] == 100.0  # saveback is a bonus, not savings cash
    assert val["positions_value"] == pytest.approx(10.0)
    # …and the 10 € mirrored into the daily-core pick.
    mirror = await savings._mirror_valuate()
    assert mirror["allocated"] == 10.0


async def test_config_and_plan_crud(mem_db):
    await savings.initialize(100.0)
    r = await savings.set_config(monthly_transfer=300.0, interest_rate=2.0)
    assert r["ok"] is True and r["monthly_transfer"] == 300.0
    r = await savings.add_plan("URTH", 100.0, 1)
    assert r["ok"] is True
    # Same ticker+day updates instead of duplicating.
    r = await savings.add_plan("URTH", 150.0, 1)
    assert r["ok"] is True
    plans = await savings.get_plans()
    assert len(plans) == 1 and plans[0]["amount"] == 150.0
    # Different day is a separate plan.
    await savings.add_plan("URTH", 50.0, 15)
    plans = await savings.get_plans()
    assert len(plans) == 2
    r = await savings.remove_plan(plans[0]["id"])
    assert r["ok"] is True
    assert len(await savings.get_plans()) == 1
    # Invalid plans rejected.
    assert (await savings.add_plan("", 100.0, 1))["ok"] is False
    assert (await savings.add_plan("URTH", -5.0, 1))["ok"] is False
    assert (await savings.add_plan("URTH", 100.0, 32))["ok"] is False


async def test_remove_position(mem_db):
    await savings.initialize(
        100.0, positions=[{"ticker": "URTH", "shares": 1.0, "avg_cost": 100.0}]
    )
    r = await savings.remove_position("URTH")
    assert r["ok"] is True
    val = await savings.valuate()
    assert val["positions"] == []
    # Removing again: no such position.
    assert (await savings.remove_position("URTH"))["ok"] is False


# ---------------------------------------------------------------------------
# Cycle + snapshot + reset isolation
# ---------------------------------------------------------------------------


async def test_run_cycle_full_pass(mem_db):
    await _seed_ranking(mem_db, ["AAPL"])
    await _seed_candle(mem_db, "URTH", 100.0)
    await _seed_candle(mem_db, "AAPL", 100.0)
    await _seed_candle(mem_db, "EURUSD=X", 1.0)
    await savings.initialize(1000.0, monthly_transfer=200.0)
    # Make both monthly actions due.
    async with mem_db() as s:
        acc = await s.get(SavingsAccount, 1)
        acc.last_transfer_month = "2000-01"
        acc.last_interest_month = "2000-01"
        await s.commit()
    await savings.add_plan("URTH", 100.0, 1)
    await _backdate_plan(mem_db, "URTH", date.today().replace(day=1))
    r = await savings.run_savings_cycle()
    assert r["transfer"]["deposited"] is True
    assert r["payout"]["paid"] is True  # 0 € but marks the month
    assert r["accrual"]["accrued"] is True
    assert r["sparplans"]["executed"] == 1
    assert r["snapshot"]["snapshotted"] is True
    curve = await savings.get_equity_curve()
    assert len(curve) == 1
    # 1000 seed + 200 transfer − 100 sparplan = 1100 cash + 100 positions.
    assert curve[0]["cash"] == pytest.approx(1100.0)
    assert curve[0]["positions_value"] == pytest.approx(100.0)
    assert curve[0]["total_equity"] == pytest.approx(1200.0)
    assert curve[0]["contributed"] == pytest.approx(1200.0)


async def test_cycle_refreshes_candles_before_acting(mem_db, monkeypatch):
    """The cycle must refresh held tickers BEFORE valuing/buying, so a
    weekend pass (or first run after a restart) prices on the most recent
    cached trading day instead of whatever the DB happens to hold."""
    calls: list[list[str]] = []

    async def fake_refresh_many(tickers, period="2y", **kwargs):
        calls.append(list(tickers))
        return list(tickers), []

    from app import market as market_mod
    monkeypatch.setattr(market_mod, "refresh_many", fake_refresh_many)
    await _seed_candle(mem_db, "URTH", 100.0)
    await _seed_candle(mem_db, "EURUSD=X", 1.0)
    await savings.initialize(100.0, positions=[
        {"ticker": "URTH", "shares": 1.0, "avg_cost": 100.0}])
    await savings.add_plan("URTH", 50.0, 1)
    await _backdate_plan(mem_db, "URTH", date.today().replace(day=1))
    r = await savings.run_savings_cycle()
    # The refresh ran and covered the held ticker + EURUSD=X.
    assert r["refresh"]["refreshed"] >= 2
    assert any("EURUSD=X" in c for c in calls)
    assert any("URTH" in c for c in calls)
    # The sparplan priced on the (fake-)refreshed candle path afterwards.
    assert r["sparplans"]["executed"] == 1


async def test_cycle_lock_skips_reentry(mem_db):
    await savings.initialize(100.0)
    # Held across a nested call: the second (non-forced) run skips.
    async with savings._cycle_lock:
        r = await savings.run_savings_cycle()
    assert r.get("skipped") is True


async def test_reset_wipes_tracker_only(mem_db):
    await savings.initialize(
        5000.0, positions=[{"ticker": "URTH", "shares": 1.0, "avg_cost": 100.0}]
    )
    await savings.add_plan("URTH", 50.0, 1)
    # A paper-sim account row must survive the savings reset.
    async with mem_db() as s:
        s.add(SimAccount(id=1, cash=123.0))
        await s.commit()
    r = await savings.reset()
    assert r["ok"] is True
    assert await savings.initialized() is False
    async with mem_db() as s:
        assert (await s.get(SavingsPosition, 1)) is None
        assert (await s.get(SavingsPlan, 1)) is None
        assert (await s.get(SimAccount, 1)) is not None  # untouched


async def test_valuation_falls_back_to_avg_cost_without_candles(mem_db):
    await savings.initialize(
        100.0, positions=[{"ticker": "NOPE.DE", "shares": 2.0, "avg_cost": 25.0}]
    )
    val = await savings.valuate()
    assert val["positions_value"] == 50.0
    assert val["total_equity"] == 150.0
    assert val["positions"][0]["current_price"] == 25.0


async def test_held_tickers_includes_fx_and_plans(mem_db):
    await savings.initialize(
        100.0, positions=[{"ticker": "AAPL", "shares": 1.0, "avg_cost": 100.0}]
    )
    await savings.add_plan("URTH", 50.0, 1)
    await _seed_ranking(mem_db, ["MSFT"])
    await _seed_candle(mem_db, "MSFT", 100.0)
    await _seed_candle(mem_db, "EURUSD=X", 1.0)
    await savings._mirror_allocate(100.0, "AAPL")
    tickers = await savings.held_tickers()
    assert set(tickers) == {"AAPL", "URTH", "MSFT", "EURUSD=X"}


async def test_held_tickers_includes_suffix_fx_pairs(mem_db):
    """A .SW/.L holding can't be priced without its suffix FX pair — without
    it the position silently falls back to avg_cost (review finding #4)."""
    await savings.initialize(
        100.0, positions=[{"ticker": "NESN.SW", "shares": 1.0, "avg_cost": 100.0}]
    )
    tickers = await savings.held_tickers()
    assert "NESN.SW" in tickers
    assert "USDCHF=X" in tickers  # .SW -> CHF -> USD -> EUR
    assert "EURUSD=X" in tickers


# ---------------------------------------------------------------------------
# Review findings: two plans per ticker, pending mirror, deduped WARNs
# ---------------------------------------------------------------------------

async def test_two_plans_same_ticker_both_execute(mem_db):
    """Review finding #1: the dedupe used to be ticker+month keyed, so a
    second plan for the same ticker (day 15) was silently skipped for the
    rest of the month after the first plan (day 1) bought."""
    await savings.initialize(1000.0)
    await _seed_candle(mem_db, "URTH", 100.0)
    await _seed_candle(mem_db, "EURUSD=X", 1.0)
    await savings.add_plan("URTH", 100.0, 1)
    await savings.add_plan("URTH", 200.0, 15)
    await _backdate_plan(mem_db, "URTH", date(2026, 9, 1))

    r = await savings.run_sparplans(day=date(2026, 9, 1))
    assert r["executed"] == 1
    assert r["trades"][0]["amount"] == 100.0
    r = await savings.run_sparplans(day=date(2026, 9, 15))
    assert r["executed"] == 1, "day-15 plan must still fire after the day-1 plan"
    assert r["trades"][0]["amount"] == 200.0
    val = await savings.valuate()
    assert val["cash"] == pytest.approx(700.0)


async def test_plan_marker_blocks_same_month_reexecution(mem_db):
    """The per-plan month marker is the dedupe — a third pass in the same
    month does nothing even though two plans share the ticker."""
    await savings.initialize(1000.0)
    await _seed_candle(mem_db, "URTH", 100.0)
    await _seed_candle(mem_db, "EURUSD=X", 1.0)
    await savings.add_plan("URTH", 100.0, 1)
    await savings.add_plan("URTH", 200.0, 15)
    await _backdate_plan(mem_db, "URTH", date(2026, 9, 1))
    await savings.run_sparplans(day=date(2026, 9, 15))
    r = await savings.run_sparplans(day=date(2026, 9, 20))
    assert r["executed"] == 0
    # Next month, both fire again.
    r = await savings.run_sparplans(day=date(2026, 10, 15))
    assert r["executed"] == 2


async def test_missing_price_warns_once_per_month(mem_db):
    """Review finding #6: a permanently unpriceable plan used to log a WARN
    on every daily pass (up to ~30 identical ledger rows)."""
    await savings.initialize(1000.0)
    await savings.add_plan("NOPE.DE", 100.0, 1)
    await _backdate_plan(mem_db, "NOPE.DE", date(2026, 9, 1))
    for d in (1, 2, 3, 4):
        r = await savings.run_sparplans(day=date(2026, 9, d))
        assert r["executed"] == 0
        assert r.get("deferred") == ["NOPE.DE"]
    warns = await _events(mem_db, "WARN")
    assert len(warns) == 1
    assert "no price" in warns[0].note


async def test_insufficient_cash_warns_once_per_month(mem_db):
    await savings.initialize(50.0)
    await _seed_candle(mem_db, "URTH", 100.0)
    await _seed_candle(mem_db, "EURUSD=X", 1.0)
    await savings.add_plan("URTH", 100.0, 1)
    await _backdate_plan(mem_db, "URTH", date(2026, 9, 1))
    for d in (1, 2, 3):
        r = await savings.run_sparplans(day=date(2026, 9, d))
        assert r["executed"] == 0
    warns = await _events(mem_db, "WARN")
    assert len(warns) == 1, "one WARN per plan per month"
    assert "insufficient" in warns[0].note


async def test_mirror_deferral_is_pending_not_dropped(mem_db):
    """Review finding #2: with no fresh daily-core ranking the mirror
    allocation used to be dropped silently, permanently understating the
    comparison. Deferred € is parked and flushed on a later pass."""
    await savings.initialize(1000.0)
    await _seed_candle(mem_db, "URTH", 100.0)
    await _seed_candle(mem_db, "EURUSD=X", 1.0)
    await savings.add_plan("URTH", 100.0, 1)
    await _backdate_plan(mem_db, "URTH", date(2026, 9, 1))

    # No ranking at all -> the buy happens but the mirror defers.
    r = await savings.run_sparplans(day=date(2026, 9, 1))
    assert r["executed"] == 1
    assert r["trades"][0]["mirror_allocated"] is False
    val = await savings.valuate()
    assert val["mirror_pending"] == pytest.approx(100.0)
    mirror = await savings._mirror_valuate()
    assert mirror["allocated"] == 0
    assert mirror["pending"] == pytest.approx(100.0)

    # Still nothing: a flush attempt leaves it parked.
    f = await savings.flush_pending_mirror()
    assert f["flushed"] == 0.0
    assert f["pending"] == pytest.approx(100.0)

    # A fresh ranking arrives -> the next cycle flushes it into the picks.
    await _seed_ranking(mem_db, ["AAPL", "MSFT"])
    await _seed_candle(mem_db, "AAPL", 100.0)
    await _seed_candle(mem_db, "MSFT", 100.0)
    f = await savings.flush_pending_mirror()
    assert f["flushed"] == pytest.approx(100.0)
    assert f["pending"] == 0.0
    mirror = await savings._mirror_valuate()
    assert mirror["allocated"] == pytest.approx(100.0)
    assert mirror["pending"] == 0.0
    by_ticker = {p["ticker"]: p for p in mirror["positions"]}
    assert by_ticker["AAPL"]["shares"] == pytest.approx(0.5)
    assert by_ticker["MSFT"]["shares"] == pytest.approx(0.5)


async def test_initialize_aggregates_duplicate_positions(mem_db):
    """Review finding #3: SavingsPosition.ticker is UNIQUE, so two lots of
    the same ETF used to blow up the whole init with an IntegrityError."""
    r = await savings.initialize(
        1000.0,
        positions=[
            {"ticker": "URTH", "shares": 1.0, "avg_cost": 100.0},
            {"ticker": "URTH", "shares": 3.0, "avg_cost": 140.0},
        ],
    )
    assert r["invested"] == pytest.approx(520.0)  # 100 + 420
    val = await savings.valuate()
    assert len(val["positions"]) == 1
    pos = val["positions"][0]
    assert pos["shares"] == pytest.approx(4.0)
    assert pos["avg_cost"] == pytest.approx(130.0)  # weighted average


async def test_initialize_rejects_non_numeric_positions(mem_db):
    """Review finding #8: float(None) raised TypeError -> raw 500. Rows with
    unusable values are skipped, the rest still initialize."""
    r = await savings.initialize(
        500.0,
        positions=[
            {"ticker": "URTH", "shares": None, "avg_cost": None},
            {"ticker": "URTH", "shares": 2.0, "avg_cost": 50.0},
        ],
    )
    assert r["ok"] is True
    assert r["invested"] == pytest.approx(100.0)
    val = await savings.valuate()
    assert val["positions"][0]["shares"] == pytest.approx(2.0)
