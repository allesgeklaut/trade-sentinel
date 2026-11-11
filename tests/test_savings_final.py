"""Savings tracker — final Phase 2 gaps: epoch identity, split attribution,
partial disposals, settlement linkage, dated tax and future-dated cash guards.

This suite pins the defects the audit's independent review found after Batch 4:

- SAME-DAY EPOCH MEMBERSHIP. ``_snapshot_epoch_clause`` compared ``as_of``
  against the reset's ``occurred_on``, so every snapshot written after a reset
  but measuring the same day (the normal case!) was read as the OLD epoch and
  the new epoch's metrics stayed empty until the next day. Membership is
  CREATION ORDER (``created_at > reset.created_at``), independent of the
  valuation day; metrics group the surviving points by ``as_of`` AFTER that
  filter.
- SPLIT ATTRIBUTION. ``record_split`` appended ONE delta under
  ``source_ticker=ticker`` from the whole mirror projection. The corporate
  action rescales the SECURITY, so the delta is computed PER FUNDING SOURCE
  from the mirror entries outstanding AS OF the split day (later buys are not
  multiplied), each source then releases exactly its scaled shares (no
  residue, no over-release of the other sources or the empty seed source).
- PARTIAL DATED SALES. ``remove_position`` gained ``shares``/``occurred_on``:
  the SELL row records the exact disposed quantity and the prior average cost,
  the remaining lot keeps its average cost, the funding source's mirror book
  is released proportionally, and cash is untouched (settlement is separate).
- REPEAT SETTLEMENT. A second ``settle_sale`` for the same disposal used to
  double-credit the proceeds. The settlement now links the SELL row through
  the persisted UNIQUE ``settlement_of``, infers only the single candidate,
  refuses every repeat, and keeps an explicit ``legacy=True`` opt-in (with an
  explanatory note) for disposals that predate the SELL record.
- DATED TAX. ``TAX`` is a cost inside the tracker (cash out, ``tax`` class),
  never an external flow: it hits the return and never moves contributed.
- FUTURE-DATED BOOK MUTATIONS. A cash movement or a position-changing record
  (SPLIT/CORRECTION/disposal) dated in the future used to mutate the current
  state now while the metrics could not contain it until the next snapshot — a
  fabricated loss (and invented holdings for a future retroactive split).
  Such dates are refused; same-day execution dates stay valid.
- BACK-DATED CHRONOLOGY. There is no lot replay: a historical split/disposal
  is refused when the current epoch holds a later position-changing row for
  that ticker, and a historical disposal is also refused when the funding
  source has a later positive mirror allocation (its negative reversal would
  otherwise replay BEFORE the original buys). Honest refusal instead of
  fabricated history; a seeded holding with no tracked later changes may
  still be adjusted.

Everything runs against an isolated in-memory SQLite database built from the
ORM metadata, with candles seeded directly: no network, no provider calls and
never the production database.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo

import httpx
import pytest
from sqlalchemy import StaticPool, select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

import app.db as db_mod
import app.market as market_mod
from app import main, savings
from app.config import settings
from app.db import (
    Base,
    Candle,
    DailyCoreRanking,
    SavingsEvent,
    SavingsMirrorEntry,
    SavingsPosition,
    SavingsSnapshot,
)

_TZ = ZoneInfo(settings.allowance_tz)


# ---------------------------------------------------------------------------
# Fixtures + helpers
# ---------------------------------------------------------------------------


@pytest.fixture
async def mem_db(monkeypatch):
    """Point every consumer of the DB at one isolated in-memory database."""
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
    monkeypatch.setattr(market_mod, "Session", session_factory)

    yield session_factory
    await engine.dispose()


@pytest.fixture
async def client(mem_db):
    transport = httpx.ASGITransport(app=main.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


def _operator_today() -> str:
    """Today's operator-local date, derived independently of the engine."""
    return datetime.now(_TZ).date().isoformat()


def _future_day() -> str:
    return (datetime.now(_TZ).date() + timedelta(days=1)).isoformat()


def _local_day_start_utc(day: str) -> datetime:
    """Test-side anchor: operator-local midnight of ``day`` in UTC."""
    return datetime.fromisoformat(day).replace(tzinfo=_TZ).astimezone(UTC)


async def _seed_candle(session_factory, ticker: str, close: float,
                       when: datetime | None = None) -> None:
    when = when or datetime.now(UTC)
    async with session_factory() as s:
        s.add(Candle(ticker=ticker, timestamp=when, open=close, high=close,
                     low=close, close=close, volume=0))
        await s.commit()


async def _seed_ranking(session_factory, picks: list[str]) -> None:
    async with session_factory() as s:
        s.add(DailyCoreRanking(id=1, ranking_date=datetime.now(UTC), band="",
                               picks=",".join(picks)))
        await s.commit()


async def _events(session_factory, kind: str | None = None) -> list[SavingsEvent]:
    async with session_factory() as s:
        stmt = select(SavingsEvent).order_by(SavingsEvent.id)
        if kind:
            stmt = stmt.where(SavingsEvent.kind == kind)
        return list((await s.scalars(stmt)).all())


async def _ledger(session_factory) -> list[tuple]:
    """Every ledger row as comparable columns — a 'no mutation' baseline."""
    return [
        (e.id, e.kind, e.amount, e.ticker, e.note, e.created_at, e.occurred_on,
         e.shares, e.price, e.cost_basis, e.flow_class, e.settlement_of)
        for e in await _events(session_factory)
    ]


async def _positions(session_factory) -> list[tuple]:
    async with session_factory() as s:
        rows = (await s.scalars(
            select(SavingsPosition).order_by(SavingsPosition.ticker))).all()
    return [(p.ticker, p.shares, p.avg_cost) for p in rows]


async def _mirror_entries(session_factory) -> list[SavingsMirrorEntry]:
    async with session_factory() as s:
        return list((await s.scalars(
            select(SavingsMirrorEntry).order_by(SavingsMirrorEntry.id))).all())


async def _mirror_snapshot(session_factory) -> list[tuple]:
    return [(e.id, e.amount, e.source_ticker, e.breakdown, e.created_at)
            for e in await _mirror_entries(session_factory)]


async def _snapshot_row(session_factory, *, as_of: str | None, equity: float,
                        created_at: datetime | None = None) -> None:
    async with session_factory() as s:
        s.add(SavingsSnapshot(
            created_at=created_at or datetime.now(UTC),
            cash=equity, positions_value=0.0, accrued_interest=0.0,
            total_equity=equity, contributed=0.0, as_of=as_of))
        await s.commit()


async def _set_flow_day(session_factory, kind: str, day: str) -> None:
    async with session_factory() as s:
        rows = (await s.scalars(
            select(SavingsEvent).where(SavingsEvent.kind == kind))).all()
        assert rows, f"no {kind} event to date"
        for r in rows:
            r.occurred_on = day
        await s.commit()


async def _backdate_entry(session_factory, entry_id: int, when: datetime) -> None:
    async with session_factory() as s:
        entry = await s.get(SavingsMirrorEntry, entry_id)
        entry.created_at = when
        await s.commit()


# ---------------------------------------------------------------------------
# 1. Same-day epoch membership (creation order, not as_of)
# ---------------------------------------------------------------------------


async def test_same_day_snapshot_after_reset_belongs_to_the_new_epoch(mem_db):
    """The normal case: reset, then a snapshot the SAME day. Under the old
    ``as_of > reset.occurred_on`` guard the new point was read as the old
    epoch and the metrics were empty. Membership is creation order."""
    await savings.initialize(1000.0)
    await savings.take_snapshot()          # old epoch point (same local day)
    await savings.reset()
    assert (await savings.initialize(0.0))["ok"] is True
    assert (await savings.add_deposit(300.0))["ok"] is True
    assert (await savings.take_snapshot())["snapshotted"] is True

    data = await savings.metric_series()
    assert [p["date"] for p in data["points"]] == [_operator_today()]
    assert data["points"][0]["total_equity"] == pytest.approx(300.0)
    assert data["points"][0]["contributed"] == pytest.approx(300.0)
    assert data["flows"] == [{"date": _operator_today(), "amount": 300.0}]

    # The mirror curve reads the same epoch clause: the same-day point exists.
    curve = await savings.get_mirror_curve(100)
    assert [p["date"] for p in curve] == [_operator_today()]

    # History is preserved: the pre-reset point is still in the raw curve.
    assert len(await savings.get_equity_curve()) == 2


async def test_epoch_membership_uses_creation_order_whatever_as_of_says(mem_db):
    """A point written AFTER the reset belongs to the new epoch even when its
    ``as_of`` measures a pre-reset day; the surviving points are bucketed by
    ``as_of`` after the membership filter. A legacy NULL row is included too
    (grouped by its created_at day)."""
    await savings.initialize(1000.0)
    await _snapshot_row(mem_db, as_of="2025-12-30", equity=1000.0)
    await savings.reset()
    assert (await savings.initialize(0.0))["ok"] is True

    # Written now, measures a pre-reset day: current epoch by creation order.
    await _snapshot_row(mem_db, as_of="2025-12-31", equity=999.0)
    # Legacy NULL as_of row written now: included, grouped by created_at day.
    await _snapshot_row(mem_db, as_of=None, equity=123.0)

    dates = [p["date"] for p in (await savings.metric_series())["points"]]
    assert dates == ["2025-12-31", _operator_today()]
    assert "2025-12-30" not in dates        # the preserved pre-reset point


# ---------------------------------------------------------------------------
# 2. Split attribution per funding source, as of the split day
# ---------------------------------------------------------------------------


async def test_split_scales_the_mirror_per_funding_source_without_overrelease(mem_db):
    """Seed + REAL_A + REAL_B all feed PICK. The split doubles all three
    (dated at the split's local-day start) and removing REAL_A releases
    exactly REAL_A's scaled shares — REAL_B and the seed keep theirs."""
    await _seed_ranking(mem_db, ["PICK"])
    await _seed_candle(mem_db, "PICK", 25.0)
    await _seed_candle(mem_db, "REAL_A", 100.0)
    await _seed_candle(mem_db, "REAL_B", 200.0)
    await _seed_candle(mem_db, "EURUSD=X", 1.0)
    await savings.initialize(
        1000.0, positions=[{"ticker": "PICK", "shares": 4.0, "avg_cost": 25.0}])
    assert (await savings.manual_buy(50.0, "REAL_A"))["ok"] is True   # 2 PICK sh
    assert (await savings.manual_buy(25.0, "REAL_B"))["ok"] is True   # 1 PICK sh

    entries = await _mirror_entries(mem_db)
    assert [(e.amount, e.source_ticker) for e in entries] == [
        (100.0, ""), (50.0, "REAL_A"), (25.0, "REAL_B")]
    entries_before = await _mirror_snapshot(mem_db)

    r = await savings.record_split("PICK", 2.0)   # today: all entries are <= today
    assert r["mirror_adjusted"] is True
    assert (r["shares"], r["avg_cost"]) == (pytest.approx(8.0), pytest.approx(12.5))

    # ONE 0-€ delta per source, each dated at the split's local-day start.
    entries = await _mirror_entries(mem_db)
    assert len(entries) == 6
    assert (await _mirror_snapshot(mem_db))[:3] == entries_before  # append-only
    deltas = {e.source_ticker: json.loads(e.breakdown) for e in entries[3:]}
    assert deltas == {
        "": [["PICK", 4.0, 0.0]],           # empty seed source is scaled too
        "REAL_A": [["PICK", 2.0, 0.0]],
        "REAL_B": [["PICK", 1.0, 0.0]],
    }
    assert all(e.amount == 0.0 for e in entries[3:])
    assert all(e.created_at == _local_day_start_utc(_operator_today())
               for e in entries[3:])

    # Projection: 14 sh, cost pool still 175 € => avg 12.5.
    async with mem_db() as s:
        row = await s.scalar(
            select(savings.SavingsMirrorPosition).where(
                savings.SavingsMirrorPosition.ticker == "PICK"))
    assert (row.shares, row.avg_cost) == (pytest.approx(14.0), pytest.approx(12.5))
    assert (await savings._mirror_valuate())["allocated"] == pytest.approx(175.0)

    # Removing REAL_A releases its ORIGINAL 2 + scaled 2 shares = 4 @ 12.5.
    assert (await savings.remove_position("REAL_A"))["ok"] is True
    entries = await _mirror_entries(mem_db)
    release = entries[-1]
    assert (release.amount, release.source_ticker) == (-50.0, "REAL_A")
    assert json.loads(release.breakdown) == [["PICK", -4.0, 12.5]]
    async with mem_db() as s:
        row = await s.scalar(
            select(savings.SavingsMirrorPosition).where(
                savings.SavingsMirrorPosition.ticker == "PICK"))
    # Seed 8 + REAL_B 2 remain: no residue, no over-release.
    assert (row.shares, row.avg_cost) == (pytest.approx(10.0), pytest.approx(12.5))

    # REAL_B then releases exactly its scaled 2 shares; the seed stays.
    assert (await savings.remove_position("REAL_B"))["ok"] is True
    async with mem_db() as s:
        row = await s.scalar(
            select(savings.SavingsMirrorPosition).where(
                savings.SavingsMirrorPosition.ticker == "PICK"))
    assert (row.shares, row.avg_cost) == (pytest.approx(8.0), pytest.approx(12.5))
    net_by_source: dict[str, float] = {}
    for e in await _mirror_entries(mem_db):
        net_by_source[e.source_ticker] = net_by_source.get(e.source_ticker, 0.0) + e.amount
    assert net_by_source == {"": pytest.approx(100.0),
                             "REAL_A": pytest.approx(0.0),
                             "REAL_B": pytest.approx(0.0)}


async def test_backdated_split_refused_when_later_position_changes_exist(mem_db):
    """No lot replay: a historical split over a book that changed later would
    invent current holdings (real 6 → 12 while the mirror only has 10).
    Refuse with the clear chronology reason and no trace; the SAME split with
    today's execution date is the normal path and still applies."""
    await _seed_ranking(mem_db, ["PICK"])
    await _seed_candle(mem_db, "PICK", 25.0)
    await _seed_candle(mem_db, "EURUSD=X", 1.0)
    await savings.initialize(1000.0)
    assert (await savings.manual_buy(100.0, "PICK"))["ok"] is True   # 4 sh TODAY
    positions = await _positions(mem_db)
    ledger = await _ledger(mem_db)
    mirror = await _mirror_snapshot(mem_db)
    split_day = (datetime.now(UTC) - timedelta(days=5)).date().isoformat()

    assert await savings.record_split("PICK", 2.0, occurred_on=split_day) == {
        "ok": False, "reason": "later position changes prevent backdated split"}
    assert await _positions(mem_db) == positions
    assert await _ledger(mem_db) == ledger
    assert await _mirror_snapshot(mem_db) == mirror

    # Same-day split: the current book IS the split-date book.
    assert (await savings.record_split("PICK", 2.0))["ok"] is True
    pos = (await savings.valuate())["positions"][0]
    assert (pos["shares"], pos["avg_cost"]) == (pytest.approx(8.0), pytest.approx(12.5))


async def test_backdated_split_scales_seeded_holding_and_only_presplit_mirror_shares(mem_db):
    """The mirror AS-OF rule with an honest real book: real PICK is a seeded,
    untracked pre-split holding (no later PICK event), and the mirror is
    funded from ANOTHER source (FUND_A) with one entry before the split and
    one after. Only the pre-split source shares are multiplied — and the real
    position is asserted, with its total cost invariant."""
    await _seed_ranking(mem_db, ["PICK"])
    await _seed_candle(mem_db, "PICK", 25.0)
    await _seed_candle(mem_db, "FUND_A", 100.0)
    await _seed_candle(mem_db, "EURUSD=X", 1.0)
    await savings.initialize(1000.0)
    # Seeded/untracked pre-split holding: no ledger position event exists.
    async with mem_db() as s:
        s.add(SavingsPosition(ticker="PICK", shares=4.0, avg_cost=25.0))
        await s.commit()
    # Mirror money from ANOTHER source: 100 € 10 days ago, 50 € TODAY.
    assert (await savings.manual_buy(100.0, "FUND_A"))["ok"] is True
    async with mem_db() as s:
        first = (await s.scalars(
            select(SavingsMirrorEntry).order_by(SavingsMirrorEntry.id))).all()[0]
    await _backdate_entry(mem_db, first.id, datetime.now(UTC) - timedelta(days=10))
    assert (await savings.manual_buy(50.0, "FUND_A"))["ok"] is True
    split_day = (datetime.now(UTC) - timedelta(days=5)).date().isoformat()

    r = await savings.record_split("PICK", 2.0, occurred_on=split_day)
    assert r["ok"] is True and r["mirror_adjusted"] is True

    # REAL position: 8 sh @ 12.5, total cost invariant 4×25 = 8×12.5 = 100 €.
    pick = next(p for p in (await savings.valuate())["positions"]
                if p["ticker"] == "PICK")
    assert (pick["shares"], pick["avg_cost"]) == (
        pytest.approx(8.0), pytest.approx(12.5))
    assert pick["shares"] * pick["avg_cost"] == pytest.approx(4.0 * 25.0)

    # Mirror: only the pre-split FUND_A entry (4 sh) is doubled.
    entries = await _mirror_entries(mem_db)
    assert len(entries) == 3
    adjustment = entries[-1]
    assert adjustment.source_ticker == "FUND_A"
    assert json.loads(adjustment.breakdown) == [["PICK", 4.0, 0.0]]
    assert adjustment.created_at == _local_day_start_utc(split_day)
    async with mem_db() as s:
        row = await s.scalar(
            select(savings.SavingsMirrorPosition).where(
                savings.SavingsMirrorPosition.ticker == "PICK"))
    # 4 + 4 (split) + 2 (later buy, NOT doubled) = 10 sh / 150 €.
    assert (row.shares, row.avg_cost) == (pytest.approx(10.0), pytest.approx(15.0))


async def test_split_ratio_that_underflows_the_average_cost_is_refused(mem_db):
    """A pathological ratio must not write a zero (or denormal) avg_cost: the
    guard refuses BEFORE the assignment, leaving no trace."""
    await savings.initialize(
        1000.0, positions=[{"ticker": "URTH", "shares": 1.0, "avg_cost": 5e-300}])
    ledger = await _ledger(mem_db)
    positions = await _positions(mem_db)
    r = await savings.record_split("URTH", 1e308)
    assert r == {"ok": False, "reason": "invalid ratio"}
    assert await _ledger(mem_db) == ledger
    assert await _positions(mem_db) == positions


# ---------------------------------------------------------------------------
# 3. Partial dated sales
# ---------------------------------------------------------------------------


async def test_partial_disposal_records_exact_sell_keeps_lot_and_releases_mirror(mem_db):
    """4 sh @ 25 → dispose 1.5 @ 30: SELL facts exact, 2.5 sh @ 25 remain,
    cash untouched, the source's mirror book released proportionally; the
    settlement is a separate, once-only credit."""
    await _seed_ranking(mem_db, ["PICK"])
    await _seed_candle(mem_db, "PICK", 25.0)
    await _seed_candle(mem_db, "EURUSD=X", 1.0)
    await savings.initialize(1000.0)
    assert (await savings.manual_buy(100.0, "PICK"))["ok"] is True   # 4 sh @ 25
    assert (await savings.valuate())["cash"] == pytest.approx(900.0)
    assert await _events(mem_db, "SETTLEMENT") == []
    entries_before = await _mirror_snapshot(mem_db)
    ledger_before = await _ledger(mem_db)

    r = await savings.remove_position(
        "PICK", 30.0, shares=1.5, occurred_on=_operator_today())
    assert r["ok"] is True
    assert r["shares"] == pytest.approx(1.5)
    assert r["remaining_shares"] == pytest.approx(2.5)
    assert r["avg_cost"] == pytest.approx(25.0)
    assert r["price"] == pytest.approx(30.0)
    assert r["occurred_on"] == _operator_today()
    assert r["mirror_released"] == pytest.approx(37.5)
    disposal_id = r["disposal_event_id"]
    assert isinstance(disposal_id, int)

    # The SELL row: exact disposed quantity, prior avg cost, supplied price.
    sell = (await _events(mem_db, "SELL"))[0]
    assert (sell.kind, sell.amount, sell.ticker, sell.occurred_on) == (
        "SELL", 0.0, "PICK", _operator_today())
    assert (sell.shares, sell.cost_basis, sell.price) == (
        pytest.approx(1.5), pytest.approx(25.0), pytest.approx(30.0))
    assert sell.flow_class == "sale" and sell.settlement_of is None

    # Remaining lot unchanged in average cost; cash untouched (no settlement).
    assert await _positions(mem_db) == [("PICK", pytest.approx(2.5), pytest.approx(25.0))]
    assert (await savings.valuate())["cash"] == pytest.approx(900.0)
    assert await _events(mem_db, "SETTLEMENT") == []

    # Mirror: proportional reversal dated at the explicit day's local start.
    entries = await _mirror_entries(mem_db)
    assert (await _mirror_snapshot(mem_db))[:len(entries_before)] == entries_before
    reversal = entries[-1]
    assert (reversal.amount, reversal.source_ticker) == (-37.5, "PICK")
    assert json.loads(reversal.breakdown) == [["PICK", -1.5, 25.0]]
    assert reversal.created_at == _local_day_start_utc(_operator_today())
    # The ledger is append-only too: the historical rows are bit-for-bit.
    assert (await _ledger(mem_db))[:len(ledger_before)] == ledger_before
    async with mem_db() as s:
        row = await s.scalar(
            select(savings.SavingsMirrorPosition).where(
                savings.SavingsMirrorPosition.ticker == "PICK"))
    assert (row.shares, row.avg_cost) == (pytest.approx(2.5), pytest.approx(25.0))

    # Settlement: credited ONCE, linked to this SELL row.
    paid = await savings.settle_sale("PICK", 45.0)
    assert paid["ok"] is True
    assert paid["settlement_of"] == disposal_id
    assert paid["cash"] == pytest.approx(945.0)
    repeat = await savings.settle_sale("PICK", 45.0)
    assert repeat == {"ok": False, "reason": "already settled"}
    assert (await savings.valuate())["cash"] == pytest.approx(945.0)
    assert len(await _events(mem_db, "SETTLEMENT")) == 1

    # A later FULL removal of the remainder releases the rest of the mirror
    # book and nets the source to exactly zero (no residue).
    assert (await savings.remove_position("PICK"))["ok"] is True
    assert await _positions(mem_db) == []
    assert (await savings._mirror_valuate())["positions"] == []
    assert (await savings._mirror_valuate())["allocated"] == pytest.approx(0.0)
    second_sell = (await _events(mem_db, "SELL"))[-1]
    assert (second_sell.shares, second_sell.cost_basis) == (
        pytest.approx(2.5), pytest.approx(25.0))
    net_shares, net_amount = 0.0, 0.0
    for e in await _mirror_entries(mem_db):
        if e.source_ticker != "PICK":
            continue
        net_amount += e.amount
        for t, sh, _px in json.loads(e.breakdown or "[]"):
            if t == "PICK":
                net_shares += float(sh)
    assert net_amount == pytest.approx(0.0)
    assert net_shares == pytest.approx(0.0)


async def test_partial_disposal_rejections_leave_no_trace(mem_db):
    await _seed_ranking(mem_db, ["PICK"])
    await _seed_candle(mem_db, "PICK", 25.0)
    await _seed_candle(mem_db, "EURUSD=X", 1.0)
    await savings.initialize(1000.0)
    assert (await savings.manual_buy(100.0, "PICK"))["ok"] is True
    ledger = await _ledger(mem_db)
    positions = await _positions(mem_db)
    mirror = await _mirror_snapshot(mem_db)
    cash = (await savings.valuate())["cash"]

    for bad in (0, -1, float("nan"), float("inf"), float("-inf"), "abc"):
        assert await savings.remove_position("PICK", shares=bad) == {
            "ok": False, "reason": "invalid shares"}
    assert await savings.remove_position("PICK", shares=4.5) == {
        "ok": False, "reason": "insufficient shares"}
    assert await savings.remove_position(
        "PICK", shares=1.0, occurred_on="15.01.2026") == {
        "ok": False, "reason": "invalid date"}
    assert await savings.remove_position(
        "PICK", shares=1.0, price=0) == {"ok": False, "reason": "invalid price"}

    assert await _ledger(mem_db) == ledger
    assert await _positions(mem_db) == positions
    assert await _mirror_snapshot(mem_db) == mirror
    assert (await savings.valuate())["cash"] == cash

    # A valid partial still works afterwards (nothing was poisoned).
    ok = await savings.remove_position("PICK", shares=1.0)
    assert ok["ok"] is True and ok["remaining_shares"] == pytest.approx(3.0)


async def test_disposal_quantity_is_exact_not_snapped(mem_db):
    """Only floating-point REPRESENTATION noise may snap to the held amount:
    an ordinary oversell (even the old 1e-9 slack) is refused instead of being
    silently clamped into a disposal the operator never entered."""
    await _seed_ranking(mem_db, ["PICK"])
    await _seed_candle(mem_db, "PICK", 25.0)
    await _seed_candle(mem_db, "EURUSD=X", 1.0)
    await savings.initialize(1000.0)
    assert (await savings.manual_buy(100.0, "PICK"))["ok"] is True   # 4 sh
    ledger = await _ledger(mem_db)
    positions = await _positions(mem_db)
    mirror = await _mirror_snapshot(mem_db)

    assert await savings.remove_position("PICK", shares=4.0 + 1e-9) == {
        "ok": False, "reason": "insufficient shares"}
    assert await _ledger(mem_db) == ledger
    assert await _positions(mem_db) == positions
    assert await _mirror_snapshot(mem_db) == mirror

    # Representation noise (1e-13 on 4 sh) IS a full disposal.
    r = await savings.remove_position("PICK", shares=4.0 + 1e-13)
    assert r["ok"] is True
    assert r["shares"] == pytest.approx(4.0)
    assert r["remaining_shares"] == 0.0
    assert (await _positions(mem_db)) == []
    sell = (await _events(mem_db, "SELL"))[0]
    assert sell.shares == pytest.approx(4.0)


async def test_backdated_disposal_refused_when_later_position_changes_exist(mem_db):
    """A historical disposal over a book that changed later would rewrite
    today's holding; refuse with no mutation. A SAME-DAY date is the normal
    execution date and still applies."""
    await _seed_ranking(mem_db, ["PICK"])
    await _seed_candle(mem_db, "PICK", 25.0)
    await _seed_candle(mem_db, "EURUSD=X", 1.0)
    await savings.initialize(1000.0)
    assert (await savings.manual_buy(100.0, "PICK"))["ok"] is True   # MANUAL_BUY today
    positions = await _positions(mem_db)
    ledger = await _ledger(mem_db)
    mirror = await _mirror_snapshot(mem_db)
    five_days_ago = (datetime.now(UTC) - timedelta(days=5)).date().isoformat()

    assert await savings.remove_position("PICK", shares=1.0,
                                         occurred_on=five_days_ago) == {
        "ok": False, "reason": "later position changes prevent backdated disposal"}
    assert await _positions(mem_db) == positions
    assert await _ledger(mem_db) == ledger
    assert await _mirror_snapshot(mem_db) == mirror

    ok = await savings.remove_position(
        "PICK", shares=1.0, occurred_on=_operator_today())
    assert ok["ok"] is True and ok["remaining_shares"] == pytest.approx(3.0)


async def test_backdated_disposal_refused_when_later_mirror_allocations_exist(mem_db):
    """A source's POSITIVE mirror allocation dated after the disposal would
    sort before the reversal in the curve replay — refuse instead of
    fabricating a short position that never existed."""
    await _seed_ranking(mem_db, ["PICK"])
    await _seed_candle(mem_db, "PICK", 25.0)
    await _seed_candle(mem_db, "EURUSD=X", 1.0)
    await savings.initialize(1000.0)
    async with mem_db() as s:   # seeded position: no position-changing event
        s.add(SavingsPosition(ticker="PICK", shares=4.0, avg_cost=25.0))
        await s.commit()
    assert (await savings._mirror_allocate(100.0, "PICK")) is not None  # entry TODAY
    positions = await _positions(mem_db)
    ledger = await _ledger(mem_db)
    mirror = await _mirror_snapshot(mem_db)
    five_days_ago = (datetime.now(UTC) - timedelta(days=5)).date().isoformat()

    assert await savings.remove_position("PICK", shares=1.0,
                                         occurred_on=five_days_ago) == {
        "ok": False, "reason": "later mirror allocations prevent backdated disposal"}
    assert await _positions(mem_db) == positions
    assert await _ledger(mem_db) == ledger
    assert await _mirror_snapshot(mem_db) == mirror

    # Same-day still works: no allocation can be "later" than today.
    ok = await savings.remove_position(
        "PICK", shares=1.0, occurred_on=_operator_today())
    assert ok["ok"] is True and ok["remaining_shares"] == pytest.approx(3.0)


async def test_partial_disposal_rolls_back_on_failure(mem_db, monkeypatch):
    """The shrunken position, WARN row, SELL record, mirror reversal and
    projection rebuild are ONE transaction."""
    await _seed_ranking(mem_db, ["PICK"])
    await _seed_candle(mem_db, "PICK", 25.0)
    await _seed_candle(mem_db, "EURUSD=X", 1.0)
    await savings.initialize(1000.0)
    assert (await savings.manual_buy(100.0, "PICK"))["ok"] is True
    positions_before = await _positions(mem_db)
    mirror_before = await _mirror_snapshot(mem_db)

    async def boom(session):
        raise RuntimeError("mirror write failed")

    with monkeypatch.context() as m:
        m.setattr(savings, "_rebuild_mirror_positions", boom)
        with pytest.raises(RuntimeError, match="mirror write failed"):
            await savings.remove_position("PICK", shares=1.5)

    assert await _positions(mem_db) == positions_before
    assert await _events(mem_db, "SELL") == []
    assert await _events(mem_db, "WARN") == []
    assert await _mirror_snapshot(mem_db) == mirror_before

    # A clean retry succeeds fully.
    assert (await savings.remove_position("PICK", shares=1.5))["ok"] is True


# ---------------------------------------------------------------------------
# 4. Settlement linkage: one disposal, one settlement
# ---------------------------------------------------------------------------


async def test_repeat_settlement_is_refused_without_crediting_again(mem_db):
    await savings.initialize(1000.0, positions=[
        {"ticker": "URTH", "shares": 2.0, "avg_cost": 50.0}])
    assert (await savings.remove_position("URTH", price=60.0))["ok"] is True
    disposal_id = (await _events(mem_db, "SELL"))[0].id

    first = await savings.settle_sale("URTH", 120.0)
    assert first == {"ok": True, "ticker": "URTH", "proceeds": 120.0,
                     "cash": 1120.0, "occurred_on": _operator_today(),
                     "settlement_of": disposal_id}
    ledger = await _ledger(mem_db)

    # Same call again: refused, no mutation at all (one credit only).
    assert await savings.settle_sale("URTH", 120.0) == {
        "ok": False, "reason": "already settled"}
    assert await savings.settle_sale(
        "URTH", 120.0, settlement_of=disposal_id) == {
        "ok": False, "reason": "already settled"}
    assert (await savings.valuate())["cash"] == pytest.approx(1120.0)
    assert await _ledger(mem_db) == ledger
    assert len(await _events(mem_db, "SETTLEMENT")) == 1


async def test_settlement_inference_requires_exactly_one_unsettled_disposal(mem_db):
    await savings.initialize(1000.0, positions=[
        {"ticker": "AAA", "shares": 1.0, "avg_cost": 10.0},
        {"ticker": "BBB", "shares": 1.0, "avg_cost": 10.0},
    ])
    assert (await savings.remove_position("AAA"))["ok"] is True
    assert (await savings.remove_position("BBB"))["ok"] is True
    aaa_id = (await _events(mem_db, "SELL"))[0].id
    bbb_id = (await _events(mem_db, "SELL"))[1].id
    # A ticker with no unsettled disposal cannot be inferred.
    assert await savings.settle_sale("CCC", 10.0) == {
        "ok": False, "reason": "no disposal to settle"}

    # BBB has exactly one: inference picks it.
    assert (await savings.settle_sale("BBB", 12.0))["settlement_of"] == bbb_id
    # AAA gets a second disposal (a holding re-added directly, test-side), so
    # inference is ambiguous and demands the explicit id.
    async with mem_db() as s:
        s.add(SavingsPosition(ticker="AAA", shares=1.0, avg_cost=10.0))
        await s.commit()
    assert (await savings.remove_position("AAA"))["ok"] is True
    assert await savings.settle_sale("AAA", 12.0) == {
        "ok": False, "reason": "ambiguous disposal — specify settlement_of"}
    # Explicit ids resolve each one; the first one settles exactly once.
    assert (await savings.settle_sale(
        "AAA", 10.0, settlement_of=aaa_id))["settlement_of"] == aaa_id
    second_id = (await _events(mem_db, "SELL"))[-1].id
    assert (await savings.settle_sale(
        "AAA", 10.0, settlement_of=second_id))["settlement_of"] == second_id


async def test_settlement_rejects_bad_references_dates_and_overpriced_proceeds(mem_db):
    await savings.initialize(1000.0, positions=[
        {"ticker": "URTH", "shares": 2.0, "avg_cost": 50.0}])
    assert (await savings.remove_position("URTH", price=60.0))["ok"] is True
    sell_id = (await _events(mem_db, "SELL"))[0].id
    init_id = (await _events(mem_db, "INIT"))[0].id
    ledger = await _ledger(mem_db)

    assert await savings.settle_sale("URTH", 10.0, settlement_of=999) == {
        "ok": False, "reason": "no such disposal"}
    assert await savings.settle_sale("URTH", 10.0, settlement_of=init_id) == {
        "ok": False, "reason": "not a disposal"}
    assert await savings.settle_sale("OTHER", 10.0, settlement_of=sell_id) == {
        "ok": False, "reason": "ticker mismatch"}
    assert await savings.settle_sale(
        "URTH", 10.0, occurred_on=(datetime.now(_TZ).date() - timedelta(days=1)).isoformat(),
        settlement_of=sell_id) == {"ok": False, "reason": "date before disposal"}
    # 2 sh × 60 € = 120 € is the recorded execution value.
    assert await savings.settle_sale("URTH", 120.01, settlement_of=sell_id) == {
        "ok": False, "reason": "proceeds exceed disposal value"}
    assert await _ledger(mem_db) == ledger
    assert (await savings.valuate())["cash"] == pytest.approx(1000.0)

    # The sub-cent tolerance admits the exact execution value.
    assert (await savings.settle_sale("URTH", 120.0, settlement_of=sell_id))["ok"] is True


async def test_legacy_settlement_requires_explicit_opt_in_and_a_note(mem_db):
    await savings.initialize(100.0)
    ledger = await _ledger(mem_db)

    # No SELL row and no explicit opt-in: never silently booked as legacy.
    assert await savings.settle_sale("URTH", 50.0) == {
        "ok": False, "reason": "no disposal to settle"}
    assert await savings.settle_sale("URTH", 50.0, legacy=True) == {
        "ok": False, "reason": "legacy settlement needs a note"}
    assert await _ledger(mem_db) == ledger
    assert (await savings.valuate())["cash"] == pytest.approx(100.0)

    ok = await savings.settle_sale(
        "URTH", 50.0, legacy=True, note="sold before the disposal record existed")
    assert ok["ok"] is True and ok["settlement_of"] is None
    assert (await savings.valuate())["cash"] == pytest.approx(150.0)
    row = (await _events(mem_db, "SETTLEMENT"))[0]
    assert row.settlement_of is None
    assert row.note.startswith("sold before the disposal record existed — ")

    # A repeat is not reclassified as legacy on its own.
    assert await savings.settle_sale("URTH", 50.0) == {
        "ok": False, "reason": "no disposal to settle"}
    # Explicitly linking a disposal while claiming legacy is contradictory.
    assert await savings.settle_sale(
        "URTH", 50.0, legacy=True, note="x", settlement_of=1) == {
        "ok": False, "reason": "legacy settlement cannot link a disposal"}


# ---------------------------------------------------------------------------
# 5. Future-dated cash guards vs record-only events
# ---------------------------------------------------------------------------


async def test_cash_events_refuse_a_future_date_without_mutation(mem_db):
    await savings.initialize(1000.0)
    ledger = await _ledger(mem_db)
    future = _future_day()

    assert await savings.withdraw(10.0, occurred_on=future) == {
        "ok": False, "reason": "future date"}
    assert await savings.add_dividend(10.0, "URTH", occurred_on=future) == {
        "ok": False, "reason": "future date"}
    assert await savings.add_fee(10.0, occurred_on=future) == {
        "ok": False, "reason": "future date"}
    assert await savings.add_tax(10.0, occurred_on=future) == {
        "ok": False, "reason": "future date"}
    assert await savings.settle_sale("URTH", 10.0, occurred_on=future) == {
        "ok": False, "reason": "future date"}

    assert await _ledger(mem_db) == ledger
    assert (await savings.valuate())["cash"] == pytest.approx(1000.0)


async def test_position_mutating_records_refuse_a_future_date(mem_db):
    """SPLIT / CORRECTION / disposal MUTATE the current real position — they
    are not record-only — so a future date is refused with no trace. A
    SAME-DAY date is the normal execution date and still applies."""
    future = _future_day()
    await savings.initialize(1000.0, positions=[
        {"ticker": "URTH", "shares": 2.0, "avg_cost": 50.0}])
    positions = await _positions(mem_db)
    ledger = await _ledger(mem_db)
    mirror = await _mirror_snapshot(mem_db)

    assert await savings.record_split("URTH", 2.0, occurred_on=future) == {
        "ok": False, "reason": "future date"}
    assert await savings.correct_position("URTH", shares=5.0,
                                          occurred_on=future) == {
        "ok": False, "reason": "future date"}
    assert await savings.remove_position("URTH", shares=1.0,
                                         occurred_on=future) == {
        "ok": False, "reason": "future date"}
    assert await _positions(mem_db) == positions
    assert await _ledger(mem_db) == ledger
    assert await _mirror_snapshot(mem_db) == mirror
    assert await _events(mem_db, "SPLIT") == []
    assert await _events(mem_db, "SELL") == []

    # Same-day execution dates are unchanged.
    assert (await savings.record_split(
        "URTH", 2.0, occurred_on=_operator_today()))["ok"] is True
    assert (await savings.correct_position(
        "URTH", shares=5.0, occurred_on=_operator_today()))["ok"] is True
    assert (await savings.remove_position(
        "URTH", shares=1.0, occurred_on=_operator_today()))["ok"] is True


# ---------------------------------------------------------------------------
# 6. Dated tax
# ---------------------------------------------------------------------------


async def test_tax_is_a_dated_cost_inside_the_tracker(client, mem_db):
    assert savings.flow_class_of("TAX", -10.0) == "tax"
    assert savings.flow_class_of("tax", -10.0) == "tax"

    await savings.initialize(0.0)
    await savings.add_deposit(1000.0)
    await _set_flow_day(mem_db, "TRANSFER", "2025-01-01")
    await _snapshot_row(mem_db, as_of="2025-01-01", equity=1000.0)
    assert (await savings.add_tax(
        100.0, occurred_on="2025-01-05", note="capital gains tax"))["ok"] is True
    await _snapshot_row(mem_db, as_of="2025-01-10", equity=900.0)

    row = (await _events(mem_db, "TAX"))[0]
    assert (row.kind, row.amount, row.ticker, row.occurred_on, row.flow_class) == (
        "TAX", -100.0, None, "2025-01-05", "tax")
    assert row.note == "capital gains tax"
    # Cash left, contributed did not: a tax is a cost, not a withdrawal.
    val = await savings.valuate()
    assert val["cash"] == pytest.approx(900.0)
    assert val["contributed"] == pytest.approx(1000.0)
    assert savings.net_contributed(await _events(mem_db)) == pytest.approx(1000.0)

    # It hits the return honestly and is excluded from the external flows.
    twr = (await client.get("/api/savings/metrics/twr")).json()
    assert [p["date"] for p in twr["points"]] == ["2025-01-01", "2025-01-10"]
    assert [p["return_pct"] for p in twr["points"]] == pytest.approx([0.0, -10.0])
    xirr = (await client.get("/api/savings/metrics/xirr")).json()
    assert xirr["flows"] == [{"date": "2025-01-01", "amount": 1000.0}]


async def test_tax_rejections_leave_no_trace(client, mem_db):
    await savings.initialize(1000.0)
    ledger = await _ledger(mem_db)
    for bad in (0, -5, float("nan"), float("inf"), "abc"):
        assert await savings.add_tax(bad) == {
            "ok": False, "reason": "invalid amount"}
    assert await savings.add_tax(5.0, occurred_on="15.01.2026") == {
        "ok": False, "reason": "invalid date"}
    assert await savings.add_tax(1000.01) == {
        "ok": False, "reason": "insufficient cash"}
    assert await _ledger(mem_db) == ledger
    assert (await savings.valuate())["cash"] == pytest.approx(1000.0)


# ---------------------------------------------------------------------------
# 7. HTTP API surface
# ---------------------------------------------------------------------------


async def test_api_sell_endpoint_partial_and_refusals(client, mem_db):
    await _seed_ranking(mem_db, ["PICK"])
    await _seed_candle(mem_db, "PICK", 25.0)
    await _seed_candle(mem_db, "EURUSD=X", 1.0)
    await savings.initialize(1000.0)
    assert (await savings.manual_buy(100.0, "PICK"))["ok"] is True

    r = await client.post("/api/savings/sell",
                          json={"ticker": "pick", "shares": 1.5, "price": 30.0,
                                "occurred_on": _operator_today()})
    assert r.status_code == 200
    body = r.json()
    assert (body["ticker"], body["shares"], body["remaining_shares"]) == (
        "PICK", pytest.approx(1.5), pytest.approx(2.5))
    assert isinstance(body["disposal_event_id"], int)

    # 422: malformed quantity / over-held / bad date.
    assert (await client.post("/api/savings/sell",
                              json={"ticker": "PICK", "shares": 0})).status_code == 422
    r = await client.post("/api/savings/sell",
                          json={"ticker": "PICK", "shares": 9})
    assert (r.status_code, r.json()["detail"]) == (422, "insufficient shares")
    r = await client.post("/api/savings/sell",
                          json={"ticker": "PICK", "shares": 1,
                                "occurred_on": "15.01.2026"})
    assert (r.status_code, r.json()["detail"]) == (422, "invalid date")
    # 404: no such position.
    r = await client.post("/api/savings/sell",
                          json={"ticker": "NOPE", "shares": 1})
    assert (r.status_code, r.json()["detail"]) == (404, "no such position")
    # The DELETE route stays the undated full disposal.
    assert (await client.delete("/api/savings/positions/PICK")).status_code == 200
    assert await _positions(mem_db) == []


async def test_api_sell_and_tax_404_when_uninitialized(client):
    r = await client.post("/api/savings/sell", json={"ticker": "URTH", "shares": 1})
    assert (r.status_code, r.json()["detail"]) == (404, "not initialized")
    r = await client.post("/api/savings/tax", json={"amount": 5.0})
    assert (r.status_code, r.json()["detail"]) == (404, "not initialized")


async def test_api_tax_endpoint_books_the_dated_cost(client, mem_db):
    await savings.initialize(1000.0)
    r = await client.post("/api/savings/tax",
                          json={"amount": 40.0, "occurred_on": "2026-02-03",
                                "note": "withholding"})
    assert r.status_code == 200
    assert (r.json()["cash"], r.json()["occurred_on"]) == (960.0, "2026-02-03")
    row = (await _events(mem_db, "TAX"))[0]
    assert (row.amount, row.occurred_on, row.note) == (-40.0, "2026-02-03", "withholding")

    r = await client.post("/api/savings/tax", json={"amount": 10_000})
    assert (r.status_code, r.json()["detail"]) == (422, "insufficient cash")
    assert (await client.post("/api/savings/tax",
                              json={"amount": 0})).status_code == 422


async def test_api_settle_links_the_disposal_and_accepts_legacy_opt_in(client, mem_db):
    await savings.initialize(1000.0, positions=[
        {"ticker": "URTH", "shares": 1.0, "avg_cost": 100.0}])
    assert (await savings.remove_position("URTH", price=150.0))["ok"] is True
    sell_id = (await _events(mem_db, "SELL"))[0].id

    r = await client.post("/api/savings/settle",
                          json={"ticker": "URTH", "proceeds": 150.0})
    assert r.status_code == 200
    assert (r.json()["settlement_of"], r.json()["cash"]) == (sell_id, 1150.0)
    r = await client.post("/api/savings/settle",
                          json={"ticker": "URTH", "proceeds": 150.0})
    assert (r.status_code, r.json()["detail"]) == (422, "already settled")

    # Legacy: explicit flag + note, no SELL row needed.
    r = await client.post("/api/savings/settle",
                          json={"ticker": "OLD", "proceeds": 5.0,
                                "legacy": True, "note": "pre-record disposal"})
    assert r.status_code == 200 and r.json()["settlement_of"] is None
    assert (await savings.valuate())["cash"] == pytest.approx(1155.0)
