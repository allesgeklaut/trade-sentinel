"""Tests for the savings DISPOSAL RECORD (Batch 1 of the savings ledger).

When the operator removes a position, the tracker must be able to say WHAT
was disposed and WHEN — without pretending it settled anything. These tests
pin the append-only ``SELL`` row that a removal writes:

- the disposal facts (operator-local date, shares, average cost, flow class,
  and the execution price only when the operator supplied one — never a
  valuation guess), next to the unchanged WARN audit row;
- ``amount=0.0`` inertia: cash, contributed capital, accrued interest and
  the snapshots are exactly what they were before the removal, because the
  proceeds reach the tracker via the next true-up, not through the ledger;
- an invalid price is rejected before ANY mutation (no delete, no WARN, no
  SELL row, no mirror reversal);
- the additive ``init_db`` migration: a pre-columns ``savings_events`` table
  converges to the full column set, legacy rows keep NULL in all five new
  columns (never backfilled), a second startup is a no-op, and a removal on
  the migrated DB writes a fully-populated SELL row;
- one-transaction atomicity: a failure after the SELL row is staged rolls
  the delete, the WARN row, the SELL row, the parked mirror €, the mirror
  reversal and the projection rebuild back — a clean retry then works;
- append-only across a remove -> re-add -> remove cycle: the second disposal
  adds a second row and the first is never mutated.

Every DB here is an in-memory SQLite built from the ORM metadata (or a
hand-shaped legacy table); the real project database is never touched.
"""

from __future__ import annotations

from datetime import UTC, datetime
from zoneinfo import ZoneInfo

import pytest
from sqlalchemy import StaticPool, select, text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

import app.db as db_mod
import app.market as market_mod
from app import savings
from app.config import settings
from app.db import (
    Base,
    Candle,
    DailyCoreRanking,
    SavingsEvent,
    SavingsMirrorEntry,
    SavingsPosition,
)

# The pre-columns savings_events table of a build that predates the disposal
# record: exactly the columns the ORM had before this batch.
_LEGACY_EVENTS_DDL = """
CREATE TABLE savings_events (
    id INTEGER NOT NULL PRIMARY KEY,
    kind VARCHAR(16) NOT NULL,
    amount FLOAT NOT NULL,
    ticker VARCHAR(32),
    note TEXT NOT NULL,
    created_at DATETIME NOT NULL
)
"""

_LEGACY_EVENT_INSERT = (
    "INSERT INTO savings_events (id, kind, amount, ticker, note, created_at) "
    "VALUES (:id, :kind, :amount, :ticker, :note, :created)"
)


# ---------------------------------------------------------------------------
# Fixtures + helpers
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

    monkeypatch.setattr(db_mod, "Session", session_factory)
    monkeypatch.setattr(savings, "Session", session_factory)
    # _price_eur_map delegates to market.latest_close (its own Session import).
    monkeypatch.setattr(market_mod, "Session", session_factory)
    yield session_factory
    await engine.dispose()


def _engine():
    return create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )


def _operator_today() -> str:
    """Today's operator-local calendar date, derived independently of the
    engine (same settings.allowance_tz convention)."""
    return datetime.now(ZoneInfo(settings.allowance_tz)).date().isoformat()


async def _seed_candle(session_factory, ticker: str, close: float) -> None:
    async with session_factory() as s:
        s.add(
            Candle(
                ticker=ticker,
                timestamp=datetime.now(UTC),
                open=close,
                high=close,
                low=close,
                close=close,
                volume=0,
            )
        )
        await s.commit()


async def _seed_ranking(session_factory, picks: list[str]) -> None:
    async with session_factory() as s:
        s.add(DailyCoreRanking(id=1, ranking_date=datetime.now(UTC), band="", picks=",".join(picks)))
        await s.commit()


async def _events(session_factory, kind: str | None = None) -> list[SavingsEvent]:
    async with session_factory() as s:
        stmt = select(SavingsEvent).order_by(SavingsEvent.id)
        if kind:
            stmt = stmt.where(SavingsEvent.kind == kind)
        return list((await s.scalars(stmt)).all())


async def _disposals(session_factory) -> list[SavingsEvent]:
    """The append-only SELL rows, oldest first."""
    return await _events(session_factory, "SELL")


async def _warns(session_factory) -> list[SavingsEvent]:
    return await _events(session_factory, "WARN")


async def _positions(session_factory) -> list[SavingsPosition]:
    async with session_factory() as s:
        return list((await s.scalars(select(SavingsPosition))).all())


async def _mirror_snapshot(session_factory) -> list[tuple]:
    """(id, amount, source_ticker, breakdown, created_at) per mirror entry —
    the immutable part of the mirror history, comparable before/after."""
    async with session_factory() as s:
        rows = (
            await s.scalars(select(SavingsMirrorEntry).order_by(SavingsMirrorEntry.id))
        ).all()
    return [(e.id, e.amount, e.source_ticker, e.breakdown, e.created_at) for e in rows]


def _disposal_row(e: SavingsEvent) -> tuple:
    """Every disposal column of one ledger row, for bit-for-bit comparison."""
    return (
        e.id,
        e.kind,
        e.amount,
        e.ticker,
        e.note,
        e.occurred_on,
        e.shares,
        e.price,
        e.cost_basis,
        e.flow_class,
        e.created_at,
    )


# ---------------------------------------------------------------------------
# 1. The SELL row itself
# ---------------------------------------------------------------------------


async def test_removal_writes_dated_disposal_record_and_keeps_warn(mem_db):
    """A removal appends ONE SELL row describing the disposal, next to the
    existing WARN audit row, and drops the position."""
    await savings.initialize(
        100.0, positions=[{"ticker": "URTH", "shares": 2.0, "avg_cost": 50.0}]
    )

    r = await savings.remove_position("URTH")

    assert r == {"ok": True}
    assert await _positions(mem_db) == []
    sells = await _disposals(mem_db)
    assert len(sells) == 1
    sell = sells[0]
    assert sell.kind == "SELL"
    assert sell.amount == 0.0  # NOT a cash movement
    assert sell.ticker == "URTH"
    assert sell.occurred_on == _operator_today()
    assert len(sell.occurred_on) == 10 and sell.occurred_on[4] == "-"  # YYYY-MM-DD
    assert sell.shares == pytest.approx(2.0)
    assert sell.cost_basis == pytest.approx(50.0)  # SavingsPosition.avg_cost
    assert sell.flow_class == "sale"
    assert sell.price is None  # never invented
    assert "disposed" in sell.note and "true-up" in sell.note
    assert sell.created_at is not None
    # The WARN audit row is still written (existing tests + the UI rely on it).
    warns = await _warns(mem_db)
    assert len(warns) == 1
    assert warns[0].ticker == "URTH"
    assert warns[0].amount == 0.0
    # The disposal record is not a WARN: the two never collide in the dedupe.
    assert [e.kind for e in await _events(mem_db)] == ["INIT", "WARN", "SELL"]


# ---------------------------------------------------------------------------
# 2. Operator-supplied price, and rejection of invalid prices
# ---------------------------------------------------------------------------


async def test_removal_records_operator_supplied_price(mem_db):
    await savings.initialize(
        100.0, positions=[{"ticker": "URTH", "shares": 3.0, "avg_cost": 40.0}]
    )
    assert (await savings.remove_position("urth", price=123.45))["ok"] is True
    sells = await _disposals(mem_db)
    assert len(sells) == 1
    assert sells[0].price == pytest.approx(123.45)
    assert (sells[0].shares, sells[0].cost_basis) == (pytest.approx(3.0), pytest.approx(40.0))


@pytest.mark.parametrize("bad_price", [0, -1, -0.001, float("nan"), float("inf"), float("-inf")])
async def test_removal_rejects_invalid_price_without_touching_anything(mem_db, bad_price):
    """An invalid price must leave NO trace at all: no delete, no WARN, no
    SELL row, no parked-€ release, no mirror reversal, no projection change."""
    await _seed_ranking(mem_db, ["AAPL"])
    await _seed_candle(mem_db, "AAPL", 100.0)
    await _seed_candle(mem_db, "EURUSD=X", 1.0)
    await _seed_candle(mem_db, "SPY", 50.0)
    await savings.initialize(1000.0)
    assert (await savings.manual_buy(30.0, "SPY"))["ok"] is True  # placed mirror €
    async with mem_db() as s:
        s.add(savings.SavingsMirrorPending(amount=7.0, source_ticker="SPY"))
        await s.commit()

    positions_before = [(p.ticker, p.shares, p.avg_cost) for p in await _positions(mem_db)]
    events_before = [_disposal_row(e) for e in await _events(mem_db)]
    mirror_before = await _mirror_snapshot(mem_db)
    mirror_value_before = await savings._mirror_valuate()
    pending_before = await savings._pending_total()
    val_before = await savings.valuate()

    r = await savings.remove_position("SPY", price=bad_price)

    assert r == {"ok": False, "reason": "invalid price"}
    assert [(p.ticker, p.shares, p.avg_cost) for p in await _positions(mem_db)] == positions_before
    assert [_disposal_row(e) for e in await _events(mem_db)] == events_before
    assert await _disposals(mem_db) == []
    assert await _warns(mem_db) == []
    assert await _mirror_snapshot(mem_db) == mirror_before
    assert await savings._mirror_valuate() == mirror_value_before
    assert await savings._pending_total() == pytest.approx(pending_before)
    assert await savings.valuate() == val_before
    # And a VALID price still works afterwards: the rejection left no lock.
    assert (await savings.remove_position("SPY", price=1.0))["ok"] is True


# ---------------------------------------------------------------------------
# 3. amount=0.0 inertia on cash / contributed / snapshots
# ---------------------------------------------------------------------------


async def test_removal_is_inert_for_cash_contributed_and_snapshots(mem_db):
    """The SELL row must not move any money: cash, accrued interest,
    contributed capital and every snapshot field it touches are unchanged."""
    await savings.initialize(
        1000.0, positions=[{"ticker": "URTH", "shares": 2.0, "avg_cost": 100.0}]
    )
    assert (await savings.add_deposit(250.0))["ok"] is True  # a real contribution
    before = await savings.valuate()
    await savings.take_snapshot()
    curve_before = await savings.get_equity_curve()
    assert before["contributed"] == pytest.approx(1250.0)
    assert before["cash"] == pytest.approx(1250.0)
    assert before["positions_value"] == pytest.approx(200.0)

    assert (await savings.remove_position("URTH"))["ok"] is True

    after = await savings.valuate()
    await savings.take_snapshot()
    curve_after = await savings.get_equity_curve()

    assert after["cash"] == before["cash"]  # no proceeds booked
    assert after["accrued_interest"] == before["accrued_interest"]
    assert after["contributed"] == before["contributed"]  # not a contribution
    assert after["positions"] == [] and after["positions_value"] == 0.0
    # Equity only lost the position's value — exactly the pre-batch behavior.
    assert after["total_equity"] == pytest.approx(
        before["total_equity"] - before["positions_value"]
    )
    # Every snapshot row keeps the same cash/contributed pair as before.
    assert {(s["cash"], s["contributed"]) for s in curve_after} == {
        (s["cash"], s["contributed"]) for s in curve_before
    }
    # Direct consumer check: the SELL row is inert in contributed capital
    # (kind filter AND amount > 0 both exclude it).
    sells = await _disposals(mem_db)
    assert len(sells) == 1 and sells[0].amount == 0.0
    contributed_sum = sum(
        e.amount
        for e in await _events(mem_db)
        if e.kind in ("INIT", "TRANSFER") and e.amount > 0
    )
    assert after["contributed"] == pytest.approx(contributed_sum) == pytest.approx(1250.0)


# ---------------------------------------------------------------------------
# 4. Additive migration of a pre-columns savings_events table
# ---------------------------------------------------------------------------


async def test_init_db_migrates_legacy_events_table_then_removal_fills_new_columns(monkeypatch):
    engine = _engine()
    monkeypatch.setattr(db_mod, "engine", engine)
    try:
        async with engine.begin() as conn:
            await conn.execute(text(_LEGACY_EVENTS_DDL))
            await conn.execute(
                text(_LEGACY_EVENT_INSERT),
                {
                    "id": 1,
                    "kind": "INIT",
                    "amount": 500.0,
                    "ticker": None,
                    "note": "legacy init",
                    "created": datetime(2026, 1, 2, 10, 0).strftime(
                        "%Y-%m-%d %H:%M:%S.%f"
                    ),
                },
            )

        await db_mod.init_db()
        await db_mod.init_db()  # ALTERs must no-op, not fail or duplicate data

        async with engine.connect() as conn:
            cols = {
                row[1]
                for row in (await conn.execute(text("PRAGMA table_info(savings_events)"))).all()
            }
            legacy = (await conn.execute(text(
                "SELECT kind, amount, occurred_on, shares, price, cost_basis, flow_class "
                "FROM savings_events WHERE id = 1"))).one()
            count = (await conn.execute(text("SELECT COUNT(*) FROM savings_events"))).scalar()
        assert {"occurred_on", "shares", "price", "cost_basis", "flow_class"} <= cols
        # Never backfilled: that disposal was never observed.
        assert tuple(legacy) == ("INIT", 500.0, None, None, None, None, None)
        assert count == 1

        session_factory = async_sessionmaker(engine, expire_on_commit=False)
        monkeypatch.setattr(db_mod, "Session", session_factory)
        monkeypatch.setattr(savings, "Session", session_factory)
        monkeypatch.setattr(market_mod, "Session", session_factory)
        await savings.initialize(
            100.0, positions=[{"ticker": "URTH", "shares": 1.0, "avg_cost": 60.0}]
        )
        # A row written the OLD way (only the pre-columns named) after the
        # migration: SQLite leaves the new columns NULL. History-safe
        # initialize (Batch 4) no longer wipes the ledger, so the explicit id
        # must continue AFTER the preserved legacy row and the INIT row
        # initialize appended (id 2) — hence id 3.
        async with engine.begin() as conn:
            await conn.execute(
                text(_LEGACY_EVENT_INSERT),
                {
                    "id": 3,
                    "kind": "TRANSFER",
                    "amount": 50.0,
                    "ticker": None,
                    "note": "legacy-shaped transfer",
                    "created": datetime(2026, 1, 3, 10, 0).strftime(
                        "%Y-%m-%d %H:%M:%S.%f"
                    ),
                },
            )

        assert (await savings.remove_position("URTH", price=61.5))["ok"] is True

        sells = await _disposals(session_factory)
        assert len(sells) == 1
        assert (
            sells[0].occurred_on,
            sells[0].shares,
            sells[0].price,
            sells[0].cost_basis,
            sells[0].flow_class,
        ) == (_operator_today(), pytest.approx(1.0), pytest.approx(61.5),
              pytest.approx(60.0), "sale")
        # The new columns are filled on NEW rows only.
        async with engine.connect() as conn:
            old_row = (await conn.execute(text(
                "SELECT occurred_on, shares, price, cost_basis, flow_class "
                "FROM savings_events WHERE id = 3"))).one()
        assert tuple(old_row) == (None, None, None, None, None)
    finally:
        await engine.dispose()


async def test_init_db_on_fresh_db_creates_disposal_columns(monkeypatch):
    """A brand-new DB gets the columns from create_all — the ALTER guard must
    not fail on the table it just created, and re-running stays safe."""
    engine = _engine()
    monkeypatch.setattr(db_mod, "engine", engine)
    try:
        await db_mod.init_db()
        await db_mod.init_db()
        async with engine.connect() as conn:
            cols = {
                row[1]
                for row in (await conn.execute(text("PRAGMA table_info(savings_events)"))).all()
            }
        assert {"occurred_on", "shares", "price", "cost_basis", "flow_class"} <= cols
    finally:
        await engine.dispose()


# ---------------------------------------------------------------------------
# 5. One transaction: a failure after the SELL row rolls everything back
# ---------------------------------------------------------------------------


async def test_removal_rolls_back_disposal_record_on_failure(mem_db, monkeypatch):
    """The delete, WARN row, SELL disposal record, parked-€ release, mirror
    reversal and projection rebuild are ONE transaction. The failure is
    injected AFTER the SELL row was flushed (the mirror reversal flushes the
    whole unit of work), so the rollback has to undo a real INSERT."""
    await _seed_ranking(mem_db, ["AAPL"])
    await _seed_candle(mem_db, "AAPL", 100.0)
    await _seed_candle(mem_db, "EURUSD=X", 1.0)
    await _seed_candle(mem_db, "SPY", 50.0)
    await savings.initialize(1000.0)
    assert (await savings.manual_buy(30.0, "SPY"))["ok"] is True  # placed entry +30
    async with mem_db() as s:
        ranking = await s.get(DailyCoreRanking, 1)
        ranking.ranking_date = datetime(2020, 1, 1)  # stale -> next buy parks
        await s.commit()
    assert (await savings.manual_buy(20.0, "SPY"))["ok"] is True  # parked 20
    entries_before = await _mirror_snapshot(mem_db)
    warns_before = len(await _warns(mem_db))
    assert warns_before == 0
    assert await savings._pending_total() == pytest.approx(20.0)

    original_rebuild = savings._rebuild_mirror_positions
    staged: list[SavingsEvent] = []

    async def boom(session):
        await original_rebuild(session)
        # Prove the failure lands AFTER the SELL row was staged and flushed
        # inside this very transaction (the mirror reversal's flush wrote it).
        staged.extend(
            (await session.scalars(
                select(SavingsEvent).where(SavingsEvent.kind == "SELL"))).all()
        )
        raise RuntimeError("mirror write failed")

    with monkeypatch.context() as m:
        m.setattr(savings, "_rebuild_mirror_positions", boom)
        with pytest.raises(RuntimeError, match="mirror write failed"):
            await savings.remove_position("SPY", price=42.0)

    # The SELL row was really written before the failure…
    assert len(staged) == 1
    assert (staged[0].ticker, staged[0].price, staged[0].flow_class) == ("SPY", 42.0, "sale")
    # …and the rollback undid it together with everything else.
    # Nothing survives the failed removal — not even the disposal record.
    async with mem_db() as s:
        still = await s.scalar(select(SavingsPosition).where(SavingsPosition.ticker == "SPY"))
    assert still is not None
    assert await _disposals(mem_db) == []
    assert await _events(mem_db, "SELL") == []
    assert len(await _warns(mem_db)) == warns_before
    assert await savings._pending_total() == pytest.approx(20.0)
    assert await _mirror_snapshot(mem_db) == entries_before
    mirror_after_failure = await savings._mirror_valuate()
    assert mirror_after_failure["allocated"] == pytest.approx(30.0)
    pos = next(p for p in mirror_after_failure["positions"] if p["ticker"] == "AAPL")
    assert pos["shares"] == pytest.approx(0.3)

    # A clean retry works fully: one SELL row, mirror unwound, pending gone.
    assert (await savings.remove_position("SPY", price=42.0))["ok"] is True
    sells = await _disposals(mem_db)
    assert len(sells) == 1 and sells[0].price == pytest.approx(42.0)
    assert await savings._pending_total() == 0.0
    assert [round(row[1], 2) for row in await _mirror_snapshot(mem_db)] == [30.0, -30.0]
    assert (await savings._mirror_valuate())["allocated"] == pytest.approx(0.0)
    assert len(await _warns(mem_db)) == warns_before + 1


# ---------------------------------------------------------------------------
# 6. Append-only across remove -> re-add -> remove
# ---------------------------------------------------------------------------


async def test_remove_readd_remove_appends_second_disposal_record(mem_db):
    await _seed_candle(mem_db, "URTH", 100.0)
    await _seed_candle(mem_db, "EURUSD=X", 1.0)  # URTH prices 1:1 in €
    await savings.initialize(
        1000.0, positions=[{"ticker": "URTH", "shares": 1.5, "avg_cost": 80.0}]
    )
    assert (await savings.remove_position("URTH", price=95.0))["ok"] is True
    first = (await _disposals(mem_db))[0]
    first_row = _disposal_row(first)
    assert (first.shares, first.cost_basis, first.price) == (
        pytest.approx(1.5), pytest.approx(80.0), pytest.approx(95.0))

    # Re-add: a real manual buy from cash (2 sh @ 100 €, a new cost basis).
    assert (await savings.manual_buy(200.0, "URTH"))["ok"] is True
    assert (await savings.remove_position("URTH"))["ok"] is True

    sells = await _disposals(mem_db)
    assert len(sells) == 2
    # The first disposal record was never mutated (append-only).
    assert _disposal_row(sells[0]) == first_row
    second = sells[1]
    assert second.id > first.id
    assert (second.shares, second.cost_basis, second.price) == (
        pytest.approx(2.0), pytest.approx(100.0), None)
    assert second.flow_class == "sale"
    assert second.amount == 0.0 and first.amount == 0.0
    assert len(await _warns(mem_db)) == 2  # one audit row per removal
