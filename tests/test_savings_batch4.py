"""Savings ledger Batch 4: the remaining dated kinds, sale settlement,
snapshot ``as_of`` and the history-safe reset/initialize.

Batch 1 wrote the dated disposal record, Batch 3 the one flow classifier + net
contributed + the savings TWR/XIRR endpoints. This batch completes the ledger:

- ``flow_class_of`` gains DIVIDEND -> dividend, FEE -> fee, SETTLEMENT ->
  sale_settlement, SPLIT -> corporate_action, CORRECTION -> correction and
  RESET -> audit, keeping the table exhaustive and unknown kinds unknown;
- ``add_dividend`` (investment income: cash in, contributed untouched),
  ``add_fee`` (a cost INSIDE the tracker: excluded from external flows so it
  hits the RETURN), ``settle_sale`` (the second half of a disposal pair —
  excluded from external flows, so the pair nets to the realized outcome) and
  ``correct_position`` (a dated, auditable bookkeeping fix; the mirror is
  untouched);
- ``record_split``: the real position is rescaled with the total cost basis
  invariant, a SPLIT record carries the NEW count/average cost, and a
  mirror-held ticker gets ONE 0-€ adjustment entry dated at the split —
  append-only, source-tagged, so past curve dates replay bit-for-bit and a
  later removal releases exactly the scaled shares and € (no residue);
- ``SavingsSnapshot.as_of``: fresh snapshots stamp the operator-local day they
  MEASURE, every day-grouping reader prefers it and falls back to created_at
  for legacy NULL rows, and ``init_db`` adds the column additively;
- history-safe reset/initialize: a second ``initialize`` is refused with no
  trace, ``reset`` appends a RESET boundary row and PRESERVES every event and
  snapshot bit-for-bit while clearing only current state (positions, plans,
  the mirror book, the account), and everything epoch-shaped (contributed,
  the metric series, the mirror curve) reads only the rows after the boundary.

Everything runs against an isolated in-memory SQLite database built from the
ORM metadata, with candles seeded directly: no network, no provider calls, and
never the production database.
"""

from __future__ import annotations

import json
from datetime import UTC, date, datetime, time, timedelta
from zoneinfo import ZoneInfo

import httpx
import pytest
from sqlalchemy import StaticPool, select, text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

import app.db as db_mod
import app.market as market_mod
from app import daily_core, main, monthly, savings, sim
from app.config import settings
from app.db import (
    Base,
    Candle,
    DailyCoreRanking,
    SavingsAccount,
    SavingsEvent,
    SavingsMirrorEntry,
    SavingsMirrorPending,
    SavingsMirrorPosition,
    SavingsPosition,
    SavingsSnapshot,
)

_TZ = ZoneInfo(settings.allowance_tz)


# ---------------------------------------------------------------------------
# Fixtures + helpers
# ---------------------------------------------------------------------------


@pytest.fixture
async def mem_db(monkeypatch):
    """Point every consumer of the DB at one isolated in-memory database.

    Same fixture contract as tests/test_savings_flows.py: ``app.db`` plus the
    engine modules (savings, market, and the paper sims whose readers the
    shared /api/metrics endpoints use) are patched, so no code path in these
    tests can reach the real database.
    """
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    session_factory = async_sessionmaker(engine, expire_on_commit=False)

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    monkeypatch.setattr(db_mod, "Session", session_factory)
    for mod in (savings, market_mod, sim, monthly, daily_core):
        monkeypatch.setattr(mod, "Session", session_factory)

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


def _local_day(when: datetime) -> str:
    """Operator-local day of an explicit tz-aware timestamp (test-side, no
    engine helper involved)."""
    return when.astimezone(_TZ).date().isoformat()


async def _seed_candle(session_factory, ticker: str, close: float,
                       when: datetime | None = None) -> None:
    when = when or datetime.now(UTC)
    async with session_factory() as s:
        s.add(Candle(ticker=ticker, timestamp=when, open=close, high=close,
                     low=close, close=close, volume=0))
        await s.commit()


async def _seed_ranking(session_factory, picks: list[str]) -> None:
    async with session_factory() as s:
        s.add(DailyCoreRanking(
            id=1, ranking_date=datetime.now(UTC), band="", picks=",".join(picks)))
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
         e.shares, e.price, e.cost_basis, e.flow_class)
        for e in await _events(session_factory)
    ]


async def _snapshots(session_factory) -> list[tuple]:
    """Every snapshot row bit-for-bit, id order."""
    async with session_factory() as s:
        rows = (await s.scalars(
            select(SavingsSnapshot).order_by(SavingsSnapshot.id))).all()
    return [
        (r.id, r.created_at, r.cash, r.positions_value, r.accrued_interest,
         r.total_equity, r.contributed, r.as_of)
        for r in rows
    ]


async def _snapshot_row(session_factory, *, as_of: str | None, equity: float,
                        created_at: datetime | None = None) -> None:
    """One equity point. ``as_of`` is the day it MEASURES (None = legacy row)."""
    async with session_factory() as s:
        s.add(SavingsSnapshot(
            created_at=created_at or datetime.now(UTC),
            cash=equity, positions_value=0.0, accrued_interest=0.0,
            total_equity=equity, contributed=0.0, as_of=as_of))
        await s.commit()


async def _set_flow_day(session_factory, kind: str, day: str) -> None:
    """Give an engine-written flow row its execution day (the API has no date
    argument for deposits; the synthetic tapes need a dated opening flow)."""
    async with session_factory() as s:
        rows = (await s.scalars(
            select(SavingsEvent).where(SavingsEvent.kind == kind))).all()
        assert rows, f"no {kind} event to date"
        for r in rows:
            r.occurred_on = day
        await s.commit()


async def _positions(session_factory) -> list[SavingsPosition]:
    async with session_factory() as s:
        return list((await s.scalars(
            select(SavingsPosition).order_by(SavingsPosition.id))).all())


async def _mirror_entries(session_factory) -> list[SavingsMirrorEntry]:
    async with session_factory() as s:
        return list((await s.scalars(
            select(SavingsMirrorEntry).order_by(SavingsMirrorEntry.id))).all())


async def _mirror_entry_snapshot(session_factory) -> list[tuple]:
    """(id, amount, source_ticker, breakdown, created_at) — the immutable
    part of the mirror history, comparable before/after."""
    return [
        (e.id, e.amount, e.source_ticker, e.breakdown, e.created_at)
        for e in await _mirror_entries(session_factory)
    ]


async def _mirror_rows(session_factory) -> list[SavingsMirrorPosition]:
    async with session_factory() as s:
        return list((await s.scalars(select(SavingsMirrorPosition))).all())


async def _account_row(session_factory) -> SavingsAccount | None:
    async with session_factory() as s:
        return await s.get(SavingsAccount, 1)


# ---------------------------------------------------------------------------
# 1. The classifier: the additions, and the fallback that stays honest
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("kind", "amount", "expected"), [
    ("DIVIDEND", 3.0, "dividend"),
    ("dividend", 3.0, "dividend"),                 # case-insensitive kind
    (" DIVIDEND ", 3.0, "dividend"),               # surrounding whitespace
    ("DIVIDEND", -1.0, "dividend"),                # kind decides, not the sign
    ("FEE", -1.0, "fee"),
    ("fee", 1.0, "fee"),
    ("SETTLEMENT", 600.0, "sale_settlement"),
    ("SPLIT", 0.0, "corporate_action"),
    ("CORRECTION", 0.0, "correction"),
    ("RESET", 0.0, "audit"),
])
def test_classifier_batch4_kinds(kind, amount, expected):
    assert savings.flow_class_of(kind, amount) == expected


@pytest.mark.parametrize(("kind", "amount"), [
    ("NOPE", 0.0),
    ("", 0.0),
    (None, 0.0),
    ("DIVIDENDX", 1.0),
])
def test_classifier_unknown_kinds_stay_unknown(kind, amount):
    assert savings.flow_class_of(kind, amount) == "unknown"


# ---------------------------------------------------------------------------
# 2a. Dividends: investment income, never contributed capital
# ---------------------------------------------------------------------------


async def test_add_dividend_books_exact_dated_event_and_never_raises_contributed(mem_db):
    await savings.initialize(1000.0)
    assert (await savings.valuate())["contributed"] == 1000.0

    r = await savings.add_dividend(12.5, "urth")
    assert r == {"ok": True, "amount": 12.5, "ticker": "URTH",
                 "cash": 1012.5, "occurred_on": _operator_today()}
    # Cash in, contributed untouched: the income is not fresh capital.
    val = await savings.valuate()
    assert val["cash"] == 1012.5
    assert val["contributed"] == 1000.0

    row = (await _events(mem_db, "DIVIDEND"))[0]
    assert (row.kind, row.amount, row.ticker, row.occurred_on, row.flow_class) == (
        "DIVIDEND", 12.5, "URTH", _operator_today(), "dividend")
    # No disposal columns invented on a dividend row.
    assert (row.shares, row.price, row.cost_basis) == (None, None, None)
    assert row.created_at is not None
    # The write-time snapshot agrees with the one classification definition.
    assert savings.flow_class_of(row.kind, row.amount) == row.flow_class

    # An explicit execution date and note are stored exactly.
    r = await savings.add_dividend(7.5, "ifx.de", occurred_on="2026-02-03", note="Q4")
    assert r["ticker"] == "IFX.DE" and r["occurred_on"] == "2026-02-03"
    assert r["cash"] == 1020.0
    row2 = (await _events(mem_db, "DIVIDEND"))[-1]
    assert (row2.amount, row2.occurred_on, row2.note) == (7.5, "2026-02-03", "Q4")

    # The function-level check the design asks for: net contributed counts only
    # the contribution/withdrawal classes, so neither dividend moves it.
    assert savings.net_contributed(await _events(mem_db)) == pytest.approx(1000.0)


@pytest.mark.parametrize("bad", [0.0, -5.0, float("nan"), float("inf"),
                                 float("-inf"), "abc", None])
async def test_add_dividend_rejects_invalid_amount_without_mutation(mem_db, bad):
    await savings.initialize(1000.0)
    ledger = await _ledger(mem_db)
    assert await savings.add_dividend(bad, "URTH") == {
        "ok": False, "reason": "invalid amount"}
    assert (await savings.valuate())["cash"] == 1000.0
    assert await _ledger(mem_db) == ledger


@pytest.mark.parametrize("bad", ["", "   ", None])
async def test_add_dividend_requires_a_ticker(mem_db, bad):
    await savings.initialize(1000.0)
    ledger = await _ledger(mem_db)
    assert await savings.add_dividend(5.0, bad) == {
        "ok": False, "reason": "ticker required"}
    assert (await savings.valuate())["cash"] == 1000.0
    assert await _ledger(mem_db) == ledger


@pytest.mark.parametrize("bad", ["2026-13-01", "01/02/2026", "20260101", "",
                                 "yesterday", "2026-1-1"])
async def test_add_dividend_rejects_invalid_date_without_mutation(mem_db, bad):
    await savings.initialize(1000.0)
    ledger = await _ledger(mem_db)
    assert await savings.add_dividend(5.0, "URTH", occurred_on=bad) == {
        "ok": False, "reason": "invalid date"}
    assert (await savings.valuate())["cash"] == 1000.0
    assert await _ledger(mem_db) == ledger


# ---------------------------------------------------------------------------
# 2b. Fees: a cost inside the tracker, visible in the RETURN
# ---------------------------------------------------------------------------


async def test_add_fee_books_negative_row_and_exact_cash(mem_db):
    await savings.initialize(1000.0)
    r = await savings.add_fee(10.0)
    assert r == {"ok": True, "amount": 10.0, "cash": 990.0,
                 "occurred_on": _operator_today()}
    val = await savings.valuate()
    assert val["cash"] == 990.0
    assert val["contributed"] == 1000.0  # a fee is not a withdrawal

    row = (await _events(mem_db, "FEE"))[0]
    assert (row.kind, row.amount, row.ticker, row.occurred_on, row.flow_class) == (
        "FEE", -10.0, None, _operator_today(), "fee")
    assert savings.flow_class_of(row.kind, row.amount) == row.flow_class

    # An explicit date + note; the sub-cent tolerance admits an exact-cash fee.
    r = await savings.add_fee(990.0, occurred_on="2026-02-03", note="TR FX fee")
    assert r["cash"] == 0.0 and r["occurred_on"] == "2026-02-03"
    row2 = (await _events(mem_db, "FEE"))[-1]
    assert (row2.amount, row2.occurred_on, row2.note) == (-990.0, "2026-02-03", "TR FX fee")


@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf"), "abc"])
async def test_add_fee_rejects_invalid_amount_without_mutation(mem_db, bad):
    await savings.initialize(1000.0)
    ledger = await _ledger(mem_db)
    assert await savings.add_fee(bad) == {"ok": False, "reason": "invalid amount"}
    assert (await savings.valuate())["cash"] == 1000.0
    assert await _ledger(mem_db) == ledger


async def test_add_fee_insufficient_cash_leaves_no_trace(mem_db):
    await savings.initialize(1000.0)
    ledger = await _ledger(mem_db)
    assert await savings.add_fee(1000.01) == {
        "ok": False, "reason": "insufficient cash"}
    assert (await savings.valuate())["cash"] == 1000.0
    assert await _ledger(mem_db) == ledger
    # A rejected fee does not poison the next valid one.
    assert (await savings.add_fee(10.0))["cash"] == 990.0


@pytest.mark.parametrize("bad", ["2026-13-01", "15.01.2026", "20260101", ""])
async def test_add_fee_rejects_invalid_date_without_mutation(mem_db, bad):
    await savings.initialize(1000.0)
    ledger = await _ledger(mem_db)
    assert await savings.add_fee(5.0, occurred_on=bad) == {
        "ok": False, "reason": "invalid date"}
    assert (await savings.valuate())["cash"] == 1000.0
    assert await _ledger(mem_db) == ledger


async def test_fee_hits_the_return_instead_of_contributed(client, mem_db):
    """1000 € funded, 10 € fee, no positions: the TWR day is −1% and
    contributed stays 1000 — a fee is not an external flow."""
    await savings.initialize(0.0)                     # INIT 0 = reconciliation
    await savings.add_deposit(1000.0)
    await _set_flow_day(mem_db, "TRANSFER", "2025-01-01")   # data-prep: opening day
    await _snapshot_row(mem_db, as_of="2025-01-01", equity=1000.0)
    await _snapshot_row(mem_db, as_of="2025-01-10", equity=990.0)
    assert (await savings.add_fee(10.0, occurred_on="2025-01-10"))["ok"] is True

    twr = (await client.get("/api/savings/metrics/twr")).json()
    assert [p["date"] for p in twr["points"]] == ["2025-01-01", "2025-01-10"]
    assert [p["return_pct"] for p in twr["points"]] == pytest.approx([0.0, -1.0])
    assert [p["contributed"] for p in twr["points"]] == pytest.approx([1000.0, 1000.0])
    assert twr["twr_pct"] == pytest.approx(-1.0)
    assert twr["max_drawdown_pct"] == pytest.approx(1.0)  # magnitude, in %

    # The flow schedule proves the exclusion: only the deposit is a flow.
    xirr = (await client.get("/api/savings/metrics/xirr")).json()
    assert xirr["flows"] == [{"date": "2025-01-01", "amount": 1000.0}]


# ---------------------------------------------------------------------------
# 2c. Sale settlement: the second half of a disposal pair
# ---------------------------------------------------------------------------


async def test_settle_sale_books_cash_and_exact_dated_row(mem_db):
    """The proceeds half of a disposal that PREDATES the SELL record.

    Correction policy (independent accounting review): since the
    one-disposal-one-settlement fix, an unmatched settlement is never booked
    silently as legacy. This fixture deliberately has no SELL row, so the
    default call must refuse and the booking needs the explicit ``legacy``
    opt-in plus an explanatory note. The cash/amount/flow assertions are
    unchanged; the exact return now also carries ``settlement_of`` = None.
    """
    await savings.initialize(100.0)

    # No SELL row exists and no opt-in was given: refuse, do not guess.
    assert await savings.settle_sale("URTH", 600.0) == {
        "ok": False, "reason": "no disposal to settle"}
    assert (await savings.valuate())["cash"] == 100.0

    r = await savings.settle_sale(
        "urth", 600.0, legacy=True,
        note="sold before the SELL record existed")
    assert r == {"ok": True, "ticker": "URTH", "proceeds": 600.0,
                 "cash": 700.0, "occurred_on": _operator_today(),
                 "settlement_of": None}
    assert (await savings.valuate())["cash"] == 700.0
    row = (await _events(mem_db, "SETTLEMENT"))[0]
    assert (row.kind, row.amount, row.ticker, row.occurred_on, row.flow_class) == (
        "SETTLEMENT", 600.0, "URTH", _operator_today(), "sale_settlement")
    # Legacy means NO disposal is linked — the opt-in + note is the attribution.
    assert row.settlement_of is None
    assert row.note.startswith("sold before the SELL record existed — ")
    # The note carries the attribution even though no SELL row exists.
    assert "disposal" in row.note and "proceeds" in row.note
    # No prior SELL row is required (legacy disposals predate the record).
    assert await _events(mem_db, "SELL") == []

    r = await savings.settle_sale("URTH", 50.0, occurred_on="2026-02-03",
                                  note="sold 3 sh", legacy=True)
    assert r["cash"] == 750.0 and r["occurred_on"] == "2026-02-03"
    assert r["settlement_of"] is None
    row2 = (await _events(mem_db, "SETTLEMENT"))[-1]
    assert row2.settlement_of is None
    assert (row2.amount, row2.occurred_on) == (50.0, "2026-02-03")
    assert row2.note.startswith("sold 3 sh — ")  # caller note, then the convention
    assert "settlement of the manual disposal" in row2.note
    # Settlement is never contributed capital.
    assert (await savings.valuate())["contributed"] == 100.0


@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf"),
                                 float("-inf"), "abc"])
async def test_settle_sale_rejects_invalid_proceeds_without_mutation(mem_db, bad):
    await savings.initialize(1000.0)
    ledger = await _ledger(mem_db)
    assert await savings.settle_sale("URTH", bad) == {
        "ok": False, "reason": "invalid proceeds"}
    assert (await savings.valuate())["cash"] == 1000.0
    assert await _ledger(mem_db) == ledger


async def test_settle_sale_requires_a_ticker_and_valid_date(mem_db):
    await savings.initialize(100.0)
    ledger = await _ledger(mem_db)
    assert await savings.settle_sale("", 10.0) == {
        "ok": False, "reason": "ticker required"}
    assert await savings.settle_sale("URTH", 10.0, occurred_on="15.01.2026") == {
        "ok": False, "reason": "invalid date"}
    assert (await savings.valuate())["cash"] == 100.0
    assert await _ledger(mem_db) == ledger


async def test_settlement_is_not_an_external_flow_and_reads_as_the_pair_net(client, mem_db):
    """Disposal mark-down + settlement, no external flow in between: the
    cumulative TWR over the pair is the REALIZED outcome (+10%), not a
    fabricated deposit and not a −50% loss.

    Correction policy (independent accounting review): this fixture seeds a
    HISTORIC snapshot tape by hand instead of a real ``remove_position``
    disposal, so there is no SELL row to link. It therefore exercises the
    legitimate legacy path — explicit ``legacy=True`` with an explanatory
    note — and the economic assertions below are unchanged.
    """
    await savings.initialize(0.0)
    await savings.add_deposit(1000.0)
    await _set_flow_day(mem_db, "TRANSFER", "2025-01-01")
    await _snapshot_row(mem_db, as_of="2025-01-01", equity=1000.0)
    await _snapshot_row(mem_db, as_of="2025-02-01", equity=500.0)   # disposal
    await _snapshot_row(mem_db, as_of="2025-03-01", equity=1100.0)  # settled
    assert (await savings.settle_sale(
        "URTH", 600.0, occurred_on="2025-03-01",
        note="disposal predates the SELL record", legacy=True))["ok"] is True
    settlement = (await _events(mem_db, "SETTLEMENT"))[0]
    assert settlement.settlement_of is None   # legacy: nothing to link

    twr = (await client.get("/api/savings/metrics/twr")).json()
    assert [p["return_pct"] for p in twr["points"]] == pytest.approx([0.0, -50.0, 10.0])
    assert [p["contributed"] for p in twr["points"]] == pytest.approx(
        [1000.0, 1000.0, 1000.0])
    assert twr["twr_pct"] == pytest.approx(10.0)   # the pair's net
    assert twr["max_drawdown_pct"] == pytest.approx(50.0)  # the in-flight dip (%)

    xirr = (await client.get("/api/savings/metrics/xirr")).json()
    assert xirr["flows"] == [{"date": "2025-01-01", "amount": 1000.0}]


@pytest.mark.parametrize(("call", "reason"), [
    (lambda: savings.add_dividend(5.0, "URTH"), "not initialized"),
    (lambda: savings.add_fee(5.0), "not initialized"),
    (lambda: savings.settle_sale("URTH", 5.0), "not initialized"),
    (lambda: savings.record_split("URTH", 2.0), "not initialized"),
    (lambda: savings.correct_position("URTH", shares=1.0), "not initialized"),
])
async def test_dated_events_require_an_initialized_tracker(mem_db, call, reason):
    assert await call() == {"ok": False, "reason": reason}
    assert await _events(mem_db) == []


# ---------------------------------------------------------------------------
# 3a. Splits: the real position, the SPLIT record and the cost invariant
# ---------------------------------------------------------------------------


async def test_record_split_adjusts_position_and_books_exact_row(mem_db):
    await savings.initialize(
        1000.0, positions=[{"ticker": "URTH", "shares": 10.0, "avg_cost": 20.0}])
    assert (await savings.valuate())["contributed"] == 1000.0

    r = await savings.record_split("urth", 2.0)
    assert r == {"ok": True, "ticker": "URTH", "ratio": 2.0, "shares": 20.0,
                 "avg_cost": 10.0, "occurred_on": _operator_today(),
                 "mirror_adjusted": False}

    val = await savings.valuate()
    pos = val["positions"][0]
    assert (pos["ticker"], pos["shares"]) == ("URTH", 20.0)
    assert pos["avg_cost"] == pytest.approx(10.0)
    # TOTAL cost basis invariant: 20 sh × 10 € == 10 sh × 20 € (exact numbers).
    assert pos["shares"] * pos["avg_cost"] == pytest.approx(10.0 * 20.0)
    assert val["cash"] == 1000.0
    assert val["contributed"] == 1000.0   # a corporate action is not capital

    row = (await _events(mem_db, "SPLIT"))[0]
    assert (row.kind, row.amount, row.ticker, row.occurred_on, row.flow_class) == (
        "SPLIT", 0.0, "URTH", _operator_today(), "corporate_action")
    # The dated-record columns carry the NEW count / NEW average cost.
    assert (row.shares, row.cost_basis) == (pytest.approx(20.0), pytest.approx(10.0))
    assert row.price is None
    assert "2-for-1" in row.note
    assert savings.flow_class_of(row.kind, row.amount) == row.flow_class

    # A reverse split, with an explicit note. The date is TODAY: a split
    # mutates the CURRENT holdings, so a historical date that precedes a later
    # tracked position change is refused by the honesty guard (it would scale
    # the later shares too). Retroactive mixed-action refusals are covered by
    # the new final suite; here the reverse split stays on the operator's day
    # so the share/cost invariant remains the assertion under test.
    r = await savings.record_split("URTH", 0.5, occurred_on=_operator_today(),
                                   note="reverse split")
    assert r["shares"] == pytest.approx(10.0) and r["avg_cost"] == pytest.approx(20.0)
    row2 = (await _events(mem_db, "SPLIT"))[-1]
    assert (row2.occurred_on, row2.note.startswith("reverse split — ")) == (
        _operator_today(), True)
    assert (row2.shares, row2.cost_basis) == (
        pytest.approx(10.0), pytest.approx(20.0))
    assert ((await savings.valuate())["positions"][0]["shares"]
            * (await savings.valuate())["positions"][0]["avg_cost"]
            == pytest.approx(200.0))


@pytest.mark.parametrize(("ratio", "reason"), [
    (0.0, "invalid ratio"),
    (-2.0, "invalid ratio"),
    (float("nan"), "invalid ratio"),
    (float("inf"), "invalid ratio"),
    (float("-inf"), "invalid ratio"),
    ("abc", "invalid ratio"),
    (1.0, "no-op split"),
    # A denormal ratio overflows avg_cost /= ratio: refused before assignment.
    (5e-324, "invalid ratio"),
])
async def test_record_split_rejections_leave_no_trace(mem_db, ratio, reason):
    await savings.initialize(
        1000.0, positions=[{"ticker": "URTH", "shares": 10.0, "avg_cost": 20.0}])
    ledger = await _ledger(mem_db)
    positions = [(p.ticker, p.shares, p.avg_cost) for p in await _positions(mem_db)]
    assert await savings.record_split("URTH", ratio) == {"ok": False, "reason": reason}
    assert await _ledger(mem_db) == ledger
    assert [(p.ticker, p.shares, p.avg_cost) for p in await _positions(mem_db)] == positions


async def test_record_split_rejects_missing_ticker_position_and_date(mem_db):
    await savings.initialize(
        1000.0, positions=[{"ticker": "URTH", "shares": 10.0, "avg_cost": 20.0}])
    ledger = await _ledger(mem_db)
    assert await savings.record_split("", 2.0) == {
        "ok": False, "reason": "ticker required"}
    assert await savings.record_split("NOPE.DE", 2.0) == {
        "ok": False, "reason": "no such position"}
    assert await savings.record_split("URTH", 2.0, occurred_on="2026-13-01") == {
        "ok": False, "reason": "invalid date"}
    assert await _ledger(mem_db) == ledger


# ---------------------------------------------------------------------------
# 3b. The mirror side: ONE 0-€ entry dated at the split
# ---------------------------------------------------------------------------


async def test_record_split_scales_the_mirror_with_a_zero_euro_dated_entry(mem_db):
    base = datetime.now(UTC) - timedelta(days=10)
    split_day = (datetime.now(UTC) - timedelta(days=5)).date().isoformat()
    post_day = (datetime.now(UTC) - timedelta(days=2)).date().isoformat()
    await _seed_ranking(mem_db, ["URTH"])
    await _seed_candle(mem_db, "URTH", 25.0, when=base)
    await _seed_candle(mem_db, "EURUSD=X", 1.0, when=base)
    await savings.initialize(1000.0)
    assert (await savings.manual_buy(100.0, "URTH"))["ok"] is True  # 4 sh @ 25

    # Data prep (test-side seeding only, no engine behaviour): the whole
    # acquisition is dated at the window's start — the mirror entry, the real
    # MANUAL_BUY event and the position's opened_at. The mirror book already
    # supports back-dated entries, and this keeps the back-dated split honest:
    # the new guard refuses a HISTORICAL split when the tracked position
    # changed AFTER that day (it would rescale later buys too), so there must
    # be no later position change here. The split is then 5 days after the
    # only acquisition, with nothing in between.
    async with mem_db() as s:
        entry = (await s.scalars(
            select(SavingsMirrorEntry).order_by(SavingsMirrorEntry.id))).all()[-1]
        entry.created_at = base
        buy_event = (await s.scalars(
            select(SavingsEvent).where(SavingsEvent.kind == "MANUAL_BUY")
        )).all()[-1]
        buy_event.created_at = base
        buy_event.occurred_on = _local_day(base)
        position = await s.scalar(
            select(SavingsPosition).where(SavingsPosition.ticker == "URTH"))
        position.opened_at = base
        await s.commit()
    await _snapshot_row(mem_db, as_of=_local_day(base), equity=100.0, created_at=base)
    curve_before = await savings.get_mirror_curve(100)
    assert curve_before == [{"date": _local_day(base), "allocated": 100.0,
                             "value": 100.0}]

    r = await savings.record_split("URTH", 2.0, occurred_on=split_day)
    assert r["mirror_adjusted"] is True

    # The real position: 8 sh @ 12.5 €, total cost still 100 €.
    pos = (await savings.valuate())["positions"][0]
    assert (pos["shares"], round(pos["avg_cost"], 10)) == (8.0, 12.5)

    # ONE appended 0-€, source-tagged adjustment, dated at the split.
    entries = await _mirror_entries(mem_db)
    assert len(entries) == 2
    split_entry = entries[-1]
    assert split_entry.amount == 0.0
    assert split_entry.source_ticker == "URTH"
    assert json.loads(split_entry.breakdown) == [["URTH", 4.0, 0.0]]
    # CORRECTED EXPECTATION (independent accounting review): the entry anchors
    # at the START of the split's OPERATOR-LOCAL day converted to UTC — not at
    # 00:00 UTC, which in Europe/Vienna is 02:00 local (summer) and can even
    # fall on the previous local day. Derive it independently of the engine.
    expected_split_at = datetime.combine(
        date.fromisoformat(split_day), time.min, tzinfo=_TZ).astimezone(UTC)
    assert split_entry.created_at == expected_split_at

    # The projection scaled: 8 mirror shares, cost pool (100 €) unchanged.
    rows = await _mirror_rows(mem_db)
    assert len(rows) == 1
    assert (rows[0].ticker, rows[0].shares) == ("URTH", pytest.approx(8.0))
    assert rows[0].avg_cost == pytest.approx(12.5)
    mirror = await savings._mirror_valuate()
    assert mirror["allocated"] == pytest.approx(100.0)

    # The curve: the pre-split date replays bit-for-bit; post-split dates value
    # the SCALED shares at the post-split (halved) close — same 100 €.
    await _seed_candle(mem_db, "URTH", 12.5, when=datetime.fromisoformat(split_day)
                       .replace(tzinfo=UTC, hour=12))
    await _seed_candle(mem_db, "EURUSD=X", 1.0, when=datetime.fromisoformat(split_day)
                       .replace(tzinfo=UTC, hour=12))
    # The current projection values on the latest (post-split) close: 8 × 12.5.
    assert (await savings._mirror_valuate())["positions"][0]["value"] == pytest.approx(100.0)
    await _snapshot_row(mem_db, as_of=split_day, equity=100.0,
                        created_at=datetime.fromisoformat(split_day).replace(tzinfo=UTC))
    await _snapshot_row(mem_db, as_of=post_day, equity=100.0,
                        created_at=datetime.fromisoformat(post_day).replace(tzinfo=UTC))
    curve_after = await savings.get_mirror_curve(100)
    assert curve_after[0] == curve_before[0]   # past date untouched, bit-for-bit
    assert [p["allocated"] for p in curve_after] == [100.0, 100.0, 100.0]
    assert [p["value"] for p in curve_after] == [100.0, 100.0, 100.0]


async def test_record_split_without_a_mirror_position_appends_no_entry(mem_db):
    await _seed_ranking(mem_db, ["AAPL"])
    await _seed_candle(mem_db, "AAPL", 100.0)
    await _seed_candle(mem_db, "EURUSD=X", 1.0)
    await _seed_candle(mem_db, "SPY", 50.0)
    await savings.initialize(1000.0)
    assert (await savings.manual_buy(50.0, "SPY"))["ok"] is True  # mirror -> AAPL
    assert {r.ticker for r in await _mirror_rows(mem_db)} == {"AAPL"}
    entries_before = await _mirror_entry_snapshot(mem_db)

    r = await savings.record_split("SPY", 2.0)
    assert r["mirror_adjusted"] is False
    assert await _mirror_entry_snapshot(mem_db) == entries_before
    pos = (await savings.valuate())["positions"][0]
    assert (pos["ticker"], pos["shares"], pos["avg_cost"]) == ("SPY", 2.0, 25.0)


async def test_split_then_removal_releases_the_scaled_mirror_without_residue(mem_db):
    """Buy -> split -> remove: the source's original buy AND its split delta
    are reversed together — the full (scaled) shares and the right € leave no
    phantom residue behind."""
    await _seed_ranking(mem_db, ["URTH"])
    await _seed_candle(mem_db, "URTH", 25.0)
    await _seed_candle(mem_db, "EURUSD=X", 1.0)
    await savings.initialize(1000.0)
    assert (await savings.manual_buy(100.0, "URTH"))["ok"] is True   # real 4 sh
    assert (await savings.record_split("URTH", 2.0))["mirror_adjusted"] is True
    mirror = await savings._mirror_valuate()
    assert mirror["positions"][0]["shares"] == pytest.approx(8.0)

    assert (await savings.remove_position("URTH"))["ok"] is True

    rows = await _mirror_entries(mem_db)
    assert [round(r.amount, 10) for r in rows] == [100.0, 0.0, -100.0]
    reversal = rows[-1]
    assert reversal.source_ticker == "URTH"
    # Everything the source ever placed: 4 original + 4 split shares, at the
    # post-split average cost (the 0-price split entry carried no cost).
    assert json.loads(reversal.breakdown) == [["URTH", -8.0, 12.5]]
    # No phantom residue: the projection is empty and the allocation is zero.
    mirror = await savings._mirror_valuate()
    assert mirror["positions"] == []
    assert mirror["allocated"] == 0.0
    assert await _positions(mem_db) == []


# ---------------------------------------------------------------------------
# 4. Corrections: dated, auditable, mirror untouched
# ---------------------------------------------------------------------------


async def test_correct_position_sets_fields_and_records_the_old_values(mem_db):
    await savings.initialize(
        1000.0, positions=[{"ticker": "URTH", "shares": 2.0, "avg_cost": 50.0}])

    r = await savings.correct_position("urth", shares=3.0, avg_cost=60.0)
    assert r == {"ok": True, "ticker": "URTH", "shares": 3.0, "avg_cost": 60.0,
                 "occurred_on": _operator_today()}
    pos = (await savings.valuate())["positions"][0]
    assert (pos["shares"], pos["avg_cost"]) == (3.0, 60.0)

    row = (await _events(mem_db, "CORRECTION"))[0]
    assert (row.kind, row.amount, row.ticker, row.occurred_on, row.flow_class) == (
        "CORRECTION", 0.0, "URTH", _operator_today(), "correction")
    assert (row.shares, row.cost_basis) == (pytest.approx(3.0), pytest.approx(60.0))
    # The OLD values are in the note: the overwritten row stays explainable.
    assert row.note == (
        "corrected shares to 3.000000 and avg cost to 60.0000 € "
        "(was 2.000000 sh @ 50.0000 €)")
    assert row.price is None
    # No cash / no contributed move.
    val = await savings.valuate()
    assert val["cash"] == 1000.0 and val["contributed"] == 1000.0

    # Only shares: the untouched field stays NULL in the record.
    r = await savings.correct_position("URTH", shares=4.0, occurred_on="2026-02-03",
                                       note="missed lot")
    assert r["shares"] == 4.0 and r["avg_cost"] == 60.0
    row2 = (await _events(mem_db, "CORRECTION"))[-1]
    assert (row2.shares, row2.cost_basis) == (pytest.approx(4.0), None)
    assert (row2.occurred_on, row2.note.startswith("missed lot — ")) == (
        "2026-02-03", True)
    assert (await savings.valuate())["positions"][0]["avg_cost"] == 60.0

    # Only avg_cost: shares stays NULL in the record.
    r = await savings.correct_position("URTH", avg_cost=42.0)
    assert r["shares"] == 4.0 and r["avg_cost"] == 42.0
    row3 = (await _events(mem_db, "CORRECTION"))[-1]
    assert (row3.shares, row3.cost_basis) == (None, pytest.approx(42.0))


@pytest.mark.parametrize(("kwargs", "reason"), [
    ({}, "nothing to correct"),
    ({"shares": 0.0}, "invalid shares"),
    ({"shares": -1.0}, "invalid shares"),
    ({"shares": float("nan")}, "invalid shares"),
    ({"shares": float("inf")}, "invalid shares"),
    ({"avg_cost": 0.0}, "invalid avg cost"),
    ({"avg_cost": float("nan")}, "invalid avg cost"),
    ({"shares": 1.0, "avg_cost": float("inf")}, "invalid avg cost"),
])
async def test_correct_position_rejections_leave_no_trace(mem_db, kwargs, reason):
    await savings.initialize(
        1000.0, positions=[{"ticker": "URTH", "shares": 2.0, "avg_cost": 50.0}])
    ledger = await _ledger(mem_db)
    assert await savings.correct_position("URTH", **kwargs) == {
        "ok": False, "reason": reason}
    assert await _ledger(mem_db) == ledger
    pos = (await savings.valuate())["positions"][0]
    assert (pos["shares"], pos["avg_cost"]) == (2.0, 50.0)


async def test_correct_position_rejects_missing_ticker_position_and_date(mem_db):
    await savings.initialize(
        1000.0, positions=[{"ticker": "URTH", "shares": 2.0, "avg_cost": 50.0}])
    ledger = await _ledger(mem_db)
    assert await savings.correct_position("", shares=1.0) == {
        "ok": False, "reason": "ticker required"}
    assert await savings.correct_position("NOPE.DE", shares=1.0) == {
        "ok": False, "reason": "no such position"}
    assert await savings.correct_position("URTH", shares=1.0,
                                          occurred_on="20260101") == {
        "ok": False, "reason": "invalid date"}
    assert await _ledger(mem_db) == ledger


# ---------------------------------------------------------------------------
# 5. Snapshot ``as_of``
# ---------------------------------------------------------------------------


async def test_take_snapshot_stamps_the_day_it_measures(mem_db):
    await savings.initialize(0.0)
    assert (await savings.take_snapshot())["snapshotted"] is True
    rows = await _snapshots(mem_db)
    assert len(rows) == 1
    assert rows[0][-1] == _operator_today()   # as_of
    curve = await savings.get_equity_curve()
    assert curve[0]["date"] == _operator_today()


async def test_metric_series_groups_by_as_of_when_present(mem_db):
    """A point written LATE attributes to the day it measures — and two points
    for one day still collapse to that day's last write."""
    now = datetime.now(UTC)
    await _snapshot_row(mem_db, as_of="2025-06-01", equity=1000.0, created_at=now)
    await _snapshot_row(mem_db, as_of="2025-06-01", equity=1010.0,
                        created_at=now + timedelta(seconds=1))
    await _snapshot_row(mem_db, as_of="2025-06-02", equity=1020.0,
                        created_at=now + timedelta(seconds=2))
    data = await savings.metric_series()
    assert [p["date"] for p in data["points"]] == ["2025-06-01", "2025-06-02"]
    assert [p["total_equity"] for p in data["points"]] == [1010.0, 1020.0]
    curve = await savings.get_equity_curve()
    assert [p["date"] for p in curve] == ["2025-06-01", "2025-06-01", "2025-06-02"]


async def test_metric_series_falls_back_to_created_at_for_legacy_rows(mem_db):
    """Legacy NULL ``as_of`` rows (written before the column existed) keep
    their created_at operator-local day — no backfill."""
    when = datetime(2025, 3, 4, 12, 0, tzinfo=UTC)
    await _snapshot_row(mem_db, as_of=None, equity=900.0, created_at=when)
    data = await savings.metric_series()
    assert [p["date"] for p in data["points"]] == ["2025-03-04"]
    curve = await savings.get_equity_curve()
    assert curve[-1]["date"] == "2025-03-04"


async def test_init_db_adds_as_of_to_a_legacy_snapshots_table(monkeypatch):
    """Additive migration, exactly like the savings_events one: a pre-``as_of``
    table converges, legacy rows keep NULL (never backfilled), a second
    startup is a no-op."""
    legacy_ddl = """
    CREATE TABLE savings_snapshots (
        id INTEGER NOT NULL PRIMARY KEY,
        created_at DATETIME NOT NULL,
        cash FLOAT NOT NULL,
        positions_value FLOAT NOT NULL,
        accrued_interest FLOAT NOT NULL,
        total_equity FLOAT NOT NULL,
        contributed FLOAT NOT NULL
    )
    """
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    monkeypatch.setattr(db_mod, "engine", engine)
    try:
        async with engine.begin() as conn:
            await conn.execute(text(legacy_ddl))
            await conn.execute(
                text("INSERT INTO savings_snapshots "
                     "(id, created_at, cash, positions_value, accrued_interest, "
                     "total_equity, contributed) VALUES "
                     "(1, :created, 100, 0, 0, 100, 50)"),
                {"created": datetime(2026, 1, 2, 10, 0).strftime(
                    "%Y-%m-%d %H:%M:%S.%f")},
            )

        await db_mod.init_db()
        await db_mod.init_db()   # the ALTER must no-op

        async with engine.connect() as conn:
            cols = {
                row[1]
                for row in (await conn.execute(
                    text("PRAGMA table_info(savings_snapshots)"))).all()
            }
            legacy = (await conn.execute(text(
                "SELECT id, total_equity, as_of FROM savings_snapshots "
                "WHERE id = 1"))).one()
        assert "as_of" in cols
        assert tuple(legacy) == (1, 100.0, None)   # never backfilled

        session_factory = async_sessionmaker(engine, expire_on_commit=False)
        async with session_factory() as s:
            s.add(SavingsSnapshot(cash=1.0, positions_value=0.0,
                                  accrued_interest=0.0, total_equity=1.0,
                                  contributed=0.0, as_of="2026-01-03"))
            await s.commit()
            rows = (await s.scalars(
                select(SavingsSnapshot).order_by(SavingsSnapshot.id))).all()
        assert [r.as_of for r in rows] == [None, "2026-01-03"]
    finally:
        await engine.dispose()


# ---------------------------------------------------------------------------
# 6. History-safe initialize / reset
# ---------------------------------------------------------------------------


async def test_initialize_twice_is_refused_without_a_trace(mem_db):
    await savings.initialize(
        1000.0, positions=[{"ticker": "URTH", "shares": 2.0, "avg_cost": 50.0}])
    await savings.add_plan("URTH", 25.0, 15)
    await savings.add_deposit(100.0)
    await savings.take_snapshot()
    ledger = await _ledger(mem_db)
    snaps = await _snapshots(mem_db)
    positions = [(p.ticker, p.shares, p.avg_cost) for p in await _positions(mem_db)]
    before = await savings.valuate()

    r = await savings.initialize(9000.0)
    assert r == {"ok": False, "reason": "already initialized"}
    assert await _ledger(mem_db) == ledger
    assert await _snapshots(mem_db) == snaps
    assert [(p.ticker, p.shares, p.avg_cost) for p in await _positions(mem_db)] == positions
    assert await savings.valuate() == before
    assert len(await savings.get_plans()) == 1


async def test_reset_appends_boundary_keeps_history_and_clears_state(mem_db):
    await _seed_ranking(mem_db, ["AAPL"])
    await _seed_candle(mem_db, "AAPL", 100.0)
    await _seed_candle(mem_db, "EURUSD=X", 1.0)
    await _seed_candle(mem_db, "URTH", 50.0)
    await savings.initialize(
        1000.0, positions=[{"ticker": "URTH", "shares": 2.0, "avg_cost": 50.0}])
    assert (await savings.manual_buy(30.0, "URTH"))["ok"] is True
    await savings.add_plan("URTH", 25.0, 15)
    assert (await savings.add_dividend(5.0, "URTH"))["ok"] is True
    await savings.take_snapshot()
    ledger_before = await _ledger(mem_db)
    snaps_before = await _snapshots(mem_db)
    assert [row[1] for row in ledger_before] == ["INIT", "MANUAL_BUY", "DIVIDEND"]
    assert len(snaps_before) == 1

    r = await savings.reset()
    assert r["ok"] is True
    assert r["preserved_events"] == 3
    assert r["preserved_snapshots"] == 1
    assert r["occurred_on"] == _operator_today()

    # (b) Every prior row is preserved bit-for-bit (ids, timestamps, columns).
    ledger_after = await _ledger(mem_db)
    assert ledger_after[:3] == ledger_before
    assert await _snapshots(mem_db) == snaps_before

    # (a) ONE boundary row, dated, amount 0, classified audit.
    assert len(ledger_after) == 4
    boundary = ledger_after[-1]
    assert boundary[1] == "RESET"
    assert boundary[2] == 0.0                       # amount
    assert boundary[3] is None                      # ticker
    assert boundary[6] == _operator_today()         # occurred_on
    assert boundary[10] == "audit"                  # flow_class
    assert boundary[4] == "tracker reset — ledger preserved"

    # (c) Current state is cleared: account, positions, plans, mirror book.
    assert await savings.initialized() is False
    assert await _account_row(mem_db) is None
    assert await savings.valuate() == {"initialized": False}
    assert await _positions(mem_db) == []
    assert await savings.get_plans() == []
    assert await _mirror_entries(mem_db) == []
    assert await _mirror_rows(mem_db) == []
    assert await savings._pending_total() == 0.0
    async with mem_db() as s:
        assert (await s.scalars(select(SavingsMirrorPending))).all() == []


async def test_reset_scopes_contributed_and_keeps_full_history(mem_db):
    await savings.initialize(1000.0)
    await savings.add_deposit(250.0)
    assert (await savings.valuate())["contributed"] == 1250.0
    await savings.take_snapshot()          # a pre-reset curve point
    # Independent baseline: reset is history-safe, so every pre-reset row must
    # survive bit-for-bit after the boundary is appended.
    snaps_before = await _snapshots(mem_db)
    ledger_before = await _ledger(mem_db)
    await savings.reset()

    # A fresh epoch: the new INIT of 0 is reconciliation, and the preserved
    # pre-reset 1250 € must NOT leak into contributed.
    assert (await savings.initialize(0.0))["ok"] is True
    assert (await savings.valuate())["contributed"] == 0.0
    assert (await savings.add_deposit(300.0))["cash"] == 300.0
    assert (await savings.valuate())["contributed"] == 300.0
    assert (await savings.withdraw(120.0))["cash"] == 180.0
    assert (await savings.valuate())["contributed"] == 180.0

    # Pre-reset rows still bit-for-bit: the ledger snapshot point is unchanged
    # (append-only ids/timestamps/columns), the RESET/new epoch only appends.
    assert (await _ledger(mem_db))[:len(ledger_before)] == ledger_before
    assert (await _snapshots(mem_db))[:len(snaps_before)] == snaps_before

    # get_events still returns the FULL preserved history (shape unchanged).
    events = await savings.get_events(500)
    kinds = [e["kind"] for e in events]
    assert set(kinds) == {"INIT", "TRANSFER", "WITHDRAWAL", "RESET"}
    assert kinds.count("INIT") == 2          # the pre-reset seed + the new epoch
    assert set(events[0]) == {"kind", "amount", "ticker", "note", "date"}

    # CORRECTED EXPECTATION (independent accounting review): epoch membership
    # is CREATION ORDER, not the day a point measures. A snapshot written
    # after the reset belongs to the CURRENT epoch even when it measures the
    # reset day itself — ``as_of`` is a valuation day, not an epoch marker. So
    # today's fresh point is NOT clipped: the metrics are [today, tomorrow],
    # both carrying the epoch's 180 € net contributed. The pre-reset same-day
    # point stays out (old epoch) but the raw curve keeps both segments.
    assert (await savings.take_snapshot())["snapshotted"] is True
    today = _operator_today()
    tomorrow = (savings._today_local() + timedelta(days=1)).isoformat()
    today_only = await savings.metric_series()
    assert [p["date"] for p in today_only["points"]] == [today]
    assert today_only["points"][0]["contributed"] == pytest.approx(180.0)
    await _snapshot_row(mem_db, as_of=tomorrow, equity=180.0)
    data = await savings.metric_series()
    assert [p["date"] for p in data["points"]] == [today, tomorrow]
    assert [p["contributed"] for p in data["points"]] == pytest.approx(
        [180.0, 180.0])
    assert data["flows"] == [{"date": today, "amount": 180.0}]
    assert len(await savings.get_equity_curve()) == 3   # preserved + new segment


async def test_reset_twice_uses_the_newest_boundary(mem_db):
    """Only the NEWEST RESET row scopes the current epoch: a second restart
    must not resurrect the first epoch's flows."""
    await savings.initialize(0.0)
    await savings.add_deposit(100.0)
    await savings.reset()
    await savings.initialize(0.0)
    await savings.add_deposit(50.0)
    assert (await savings.valuate())["contributed"] == pytest.approx(50.0)
    await savings.reset()
    await savings.initialize(0.0)
    await savings.add_deposit(7.0)
    assert (await savings.valuate())["contributed"] == pytest.approx(7.0)
    assert [e.kind for e in await _events(mem_db)].count("RESET") == 2


# ---------------------------------------------------------------------------
# 7. End-to-end: the epoch TWR does not cross the reset boundary
# ---------------------------------------------------------------------------


async def test_epoch_twr_does_not_cross_the_reset_boundary(client, mem_db):
    # Pre-reset epoch: a flat flows-only tape (0% TWR).
    await savings.initialize(0.0)
    await savings.add_deposit(1000.0)
    await _set_flow_day(mem_db, "TRANSFER", "2025-01-01")
    await _snapshot_row(mem_db, as_of="2025-01-01", equity=1000.0)
    await _snapshot_row(mem_db, as_of="2025-02-01", equity=1000.0)
    first = (await client.get("/api/savings/metrics/twr")).json()
    assert [p["date"] for p in first["points"]] == ["2025-01-01", "2025-02-01"]
    assert first["twr_pct"] == pytest.approx(0.0)

    await savings.reset()

    # New epoch: its own flat tape, starting AFTER the boundary.
    assert (await savings.initialize(0.0))["ok"] is True
    assert (await savings.add_deposit(300.0))["ok"] is True
    tomorrow = (savings._today_local() + timedelta(days=1)).isoformat()
    day_after = (savings._today_local() + timedelta(days=2)).isoformat()
    await _snapshot_row(mem_db, as_of=tomorrow, equity=300.0)
    await _snapshot_row(mem_db, as_of=day_after, equity=300.0)

    second = (await client.get("/api/savings/metrics/twr")).json()
    assert [p["date"] for p in second["points"]] == [tomorrow, day_after]
    assert second["start"] == tomorrow and second["end"] == day_after
    assert [p["contributed"] for p in second["points"]] == pytest.approx([300.0, 300.0])
    assert second["twr_pct"] == pytest.approx(0.0)
    assert second["max_drawdown_pct"] == pytest.approx(0.0)

    xirr = (await client.get("/api/savings/metrics/xirr")).json()
    assert xirr["flows"] == [{"date": _operator_today(), "amount": 300.0}]
    assert xirr["contributed"] == pytest.approx(300.0)


# ---------------------------------------------------------------------------
# 8. HTTP API
# ---------------------------------------------------------------------------


async def test_api_dated_events_book_and_report_state_errors(client, mem_db):
    await savings.initialize(1000.0)

    r = await client.post("/api/savings/dividend",
                          json={"amount": 12.5, "ticker": "urth", "note": "q4"})
    assert r.status_code == 200
    body = r.json()
    assert (body["ok"], body["ticker"], body["cash"]) == (True, "URTH", 1012.5)
    assert body["occurred_on"] == _operator_today()

    r = await client.post("/api/savings/fee",
                          json={"amount": 2.5, "occurred_on": "2026-01-15"})
    assert r.status_code == 200
    assert (r.json()["cash"], r.json()["occurred_on"]) == (1010.0, "2026-01-15")

    # CORRECTED EXPECTATION (independent accounting review): the default
    # unmatched settlement is REFUSED — no SELL row exists for this fixture,
    # and the endpoint must not silently classify it as legacy (422, and the
    # tracked cash is untouched by the refusal).
    r = await client.post("/api/savings/settle",
                          json={"ticker": "URTH", "proceeds": 90.0})
    assert (r.status_code, r.json()["detail"]) == (422, "no disposal to settle")
    assert (await client.get("/api/savings/status")).json()["cash"] == 1010.0

    # The legacy disposal predates the SELL record: the explicit opt-in plus
    # an explanatory note books the proceeds, with no disposal linked.
    r = await client.post("/api/savings/settle",
                          json={"ticker": "URTH", "proceeds": 90.0,
                                "legacy": True,
                                "note": "disposal predates the SELL record"})
    assert r.status_code == 200
    assert (r.json()["proceeds"], r.json()["cash"]) == (90.0, 1100.0)
    assert r.json()["settlement_of"] is None

    # 422: malformed payload / state rejection (never 500, never a mutation).
    assert (await client.post("/api/savings/dividend",
                              json={"amount": 0, "ticker": "URTH"})).status_code == 422
    assert (await client.post("/api/savings/dividend",
                              json={"amount": 5, "ticker": "bad ticker"})).status_code == 422
    assert (await client.post("/api/savings/fee",
                              json={"amount": 10_000})).status_code == 422
    r = await client.post("/api/savings/fee", json={"amount": 10_000})
    assert r.json()["detail"] == "insufficient cash"
    assert (await client.post("/api/savings/settle",
                              json={"ticker": "URTH", "proceeds": -1})).status_code == 422
    r = await client.post("/api/savings/dividend",
                          json={"amount": 5, "ticker": "URTH",
                                "occurred_on": "15.01.2026"})
    assert (r.status_code, r.json()["detail"]) == (422, "invalid date")

    # split / correct need a real position.
    assert (await client.post("/api/savings/split",
                              json={"ticker": "URTH", "ratio": 2})).status_code == 422
    r = await client.post("/api/savings/split", json={"ticker": "URTH", "ratio": 2})
    assert r.json()["detail"] == "no such position"
    assert (await client.post("/api/savings/correct",
                              json={"ticker": "URTH", "shares": 2})).status_code == 422

    await _seed_candle(mem_db, "URTH", 20.0)
    await _seed_candle(mem_db, "EURUSD=X", 1.0)
    assert (await savings.manual_buy(100.0, "URTH"))["ok"] is True   # 5 sh
    # Same correction policy as the reverse split above: a split mutates the
    # CURRENT position, so a historical date after a later tracked buy is now
    # refused ("later position changes prevent backdated split"). Keep the
    # endpoint's split book-keeping assertion on the operator's day; the
    # retroactive refusal itself is covered by the new final suite.
    r = await client.post("/api/savings/split",
                          json={"ticker": "urth", "ratio": 2.0,
                                "occurred_on": _operator_today()})
    assert r.status_code == 200
    assert (r.json()["shares"], r.json()["avg_cost"]) == (pytest.approx(10.0),
                                                          pytest.approx(10.0))
    r = await client.post("/api/savings/split", json={"ticker": "URTH", "ratio": 1.0})
    assert (r.status_code, r.json()["detail"]) == (422, "no-op split")

    r = await client.post("/api/savings/correct",
                          json={"ticker": "URTH", "shares": 9.0, "note": "missed lot"})
    assert r.status_code == 200 and r.json()["shares"] == pytest.approx(9.0)
    r = await client.post("/api/savings/correct", json={"ticker": "URTH"})
    assert (r.status_code, r.json()["detail"]) == (422, "nothing to correct")


@pytest.mark.parametrize(("path", "payload"), [
    ("/api/savings/dividend", {"amount": 5, "ticker": "URTH"}),
    ("/api/savings/fee", {"amount": 5}),
    ("/api/savings/settle", {"ticker": "URTH", "proceeds": 5}),
    ("/api/savings/split", {"ticker": "URTH", "ratio": 2}),
    ("/api/savings/correct", {"ticker": "URTH", "shares": 2}),
])
async def test_api_dated_event_endpoints_404_when_uninitialized(client, path, payload):
    r = await client.post(path, json=payload)
    assert r.status_code == 404
    assert r.json()["detail"] == "not initialized"


async def test_api_initialize_is_refused_on_a_live_tracker(client):
    r = await client.post("/api/savings/initialize", json={"cash": 100.0})
    assert r.status_code == 200 and r.json()["ok"] is True
    r = await client.post("/api/savings/initialize", json={"cash": 9000.0})
    assert (r.status_code, r.json()["detail"]) == (422, "already initialized")
    status = (await client.get("/api/savings/status")).json()
    assert status["cash"] == 100.0
