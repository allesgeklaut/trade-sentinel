"""Savings ledger Batch 3: flow classification, dated withdrawals, NET
contributed capital, savings TWR/XIRR metrics and the London pence fix.

The audit finding (``docs/ui-strategy-audit-2026-10-21.md`` P1-B) was that a
withdrawal was represented as reconciliation drift and contributed capital
only summed POSITIVE INIT/TRANSFER rows — so taking money out looked like an
investment loss. These tests pin:

- ``flow_class_of`` as THE classification table (every kind/sign, unknown
  kinds stay ``"unknown"``);
- ``_log_event`` writing the write-time ``flow_class`` snapshot for every
  existing caller, with legacy NULL rows classified on read (no backfill);
- ``withdraw``: a dated, signed ``WITHDRAWAL`` outflow that decrements cash,
  validates amount/date/cash BEFORE mutating, and defaults to today
  (operator-local) exactly like the rest of the ledger;
- NET contributed capital: a deposit followed by an equal withdrawal leaves
  contributed at zero and creates NO phantom investment loss (the audit's
  acceptance case), and TRUEUP/interest/rewards never move it;
- the savings metrics endpoints (TWR + dated XIRR): a flat flows-only tape is
  0% TWR with no drawdown, a mid-curve withdrawal never creates a negative
  return day, and XIRR discounts a flow on its ``occurred_on`` date rather
  than the ``created_at`` day it was booked;
- EUR pricing converts London pence BEFORE the GBP pair (close * 0.01 *
  GBPUSD / EURUSD) in both the spot map and the historical series, while
  unit-1.0 venues stay bit-for-bit and a missing pair still falls back to
  None (avg_cost);
- every inline ``<script>`` block in ``static/index.html`` parses
  (``node --check``) — no DOM assertions are fabricated.

Everything runs against an isolated in-memory SQLite database built from the
ORM metadata, with candles seeded directly: no network, no provider calls,
and never the production database.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import httpx
import pytest
from sqlalchemy import StaticPool, select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

import app.db as db_mod
import app.market as market_mod
from app import daily_core, main, monthly, savings, sim
from app.config import settings
from app.db import (
    Base,
    Candle,
    SavingsAccount,
    SavingsEvent,
    SavingsPlan,
    SavingsSnapshot,
)

INDEX_HTML = Path(__file__).resolve().parent.parent / "static" / "index.html"


# ---------------------------------------------------------------------------
# Fixtures + helpers
# ---------------------------------------------------------------------------


@pytest.fixture
async def mem_db(monkeypatch):
    """Point every consumer of the DB at one isolated in-memory database.

    ``app.db`` plus the engine modules (savings, market, and the paper sims
    whose readers the shared ``/api/metrics`` endpoints use) are patched, so
    no code path in these tests can reach the real database.
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
    return datetime.now(ZoneInfo(settings.allowance_tz)).date().isoformat()


async def _seed_candle(session_factory, ticker: str, close: float,
                       when: datetime | None = None) -> None:
    when = when or datetime.now(UTC)
    async with session_factory() as s:
        s.add(Candle(ticker=ticker, timestamp=when, open=close, high=close,
                     low=close, close=close, volume=0))
        await s.commit()


async def _events(session_factory, kind: str | None = None) -> list[SavingsEvent]:
    async with session_factory() as s:
        stmt = select(SavingsEvent).order_by(SavingsEvent.id)
        if kind:
            stmt = stmt.where(SavingsEvent.kind == kind)
        return list((await s.scalars(stmt)).all())


async def _ledger(session_factory) -> list[tuple]:
    """Every ledger row as comparable columns — a 'no mutation' baseline."""
    async with session_factory() as s:
        rows = (await s.scalars(select(SavingsEvent).order_by(SavingsEvent.id))).all()
    return [
        (e.id, e.kind, e.amount, e.ticker, e.note, e.created_at, e.occurred_on,
         e.shares, e.price, e.cost_basis, e.flow_class)
        for e in rows
    ]


async def _backdate_plan(session_factory, ticker: str, days: int) -> None:
    """Move a plan's created_at into the past so it is not skipped by the
    created-after-due-day rule."""
    created = datetime.now(UTC) - timedelta(days=days)
    async with session_factory() as s:
        plans = (
            await s.scalars(select(SavingsPlan).where(SavingsPlan.ticker == ticker))
        ).all()
        assert plans, f"no plan for {ticker}"
        for plan in plans:
            plan.created_at = created
        await s.commit()


async def _snapshot(session_factory, day: date, equity: float) -> None:
    """One daily equity point (12:00 UTC so no timezone can shift the date)."""
    async with session_factory() as s:
        s.add(SavingsSnapshot(
            created_at=datetime(day.year, day.month, day.day, 12, tzinfo=UTC),
            cash=0.0,
            positions_value=equity,
            total_equity=equity,
            contributed=0.0,
        ))
        await s.commit()


async def _flow_event(session_factory, kind: str, amount: float, *,
                      occurred_on: date | None = None,
                      created_at: datetime | None = None) -> None:
    """Append an external-flow ledger row with a NULL flow_class, so the read
    path must classify it (the legacy-row convention)."""
    async with session_factory() as s:
        s.add(SavingsEvent(
            kind=kind,
            amount=amount,
            note="",
            occurred_on=occurred_on.isoformat() if occurred_on else None,
            created_at=created_at or datetime.now(UTC),
        ))
        await s.commit()


def _reference_xirr(flows: list[tuple[date, float]], final_value: float,
                    final_date: date) -> float:
    """Independent bisection on the XIRR definition (deliberately a separate
    implementation, so a bug in ``app.performance.xirr`` cannot confirm
    itself)."""
    start = min([d for d, _ in flows] + [final_date])
    times = [((d - start).days / 365.0) for d, _ in flows]
    times.append((final_date - start).days / 365.0)
    amounts = [-a for _, a in flows] + [final_value]

    def npv(rate: float) -> float:
        return sum(a / (1.0 + rate) ** t for a, t in zip(amounts, times, strict=True))

    lo, hi = -0.9999, 1.0
    while npv(lo) * npv(hi) > 0:
        hi *= 2.0
    for _ in range(300):
        mid = 0.5 * (lo + hi)
        if npv(lo) * npv(mid) <= 0:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


# ---------------------------------------------------------------------------
# 1. The classifier grid
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("kind", "amount", "expected"), [
    # contributions: money in from outside
    ("INIT", 5000.0, "contribution"),
    ("TRANSFER", 250.0, "contribution"),
    ("transfer", 250.0, "contribution"),          # case-insensitive kind
    (" TRANSFER ", 1.0, "contribution"),          # surrounding whitespace
    # pre-withdrawal-era outflow corrections -> reconciliation, not capital
    ("INIT", 0.0, "reconciliation"),
    ("TRANSFER", 0.0, "reconciliation"),
    ("INIT", -1.0, "reconciliation"),
    ("TRANSFER", -20.0, "reconciliation"),
    ("TRANSFER", float("nan"), "reconciliation"),  # unreadable never capital
    # the withdrawal kind owns the money-out flow now
    ("WITHDRAWAL", -300.0, "withdrawal"),
    ("WITHDRAWAL", 300.0, "withdrawal"),          # kind decides, not the sign
    # buys are investments (money stays in the account, it just changes form)
    ("SPARPLAN_BUY", -100.0, "investment"),
    ("MANUAL_BUY", -50.0, "investment"),
    ("SAVEBACK", 10.0, "reward"),
    ("INTEREST_PAYOUT", 3.5, "interest"),
    ("TRUEUP", 12.0, "reconciliation"),
    ("TRUEUP", -12.0, "reconciliation"),
    ("SELL", 0.0, "sale"),
    ("WARN", 0.0, "audit"),
    # Batch 4: the remaining dated kinds are classified, not guessed. The two
    # rows that used to pin DIVIDEND/FEE as "unknown" were the classifier's
    # documented fallback; the kinds now exist (Batch 4) and own their classes.
    ("DIVIDEND", 3.0, "dividend"),
    ("FEE", -1.0, "fee"),
    ("SETTLEMENT", 600.0, "sale_settlement"),
    ("SPLIT", 0.0, "corporate_action"),
    ("CORRECTION", 0.0, "correction"),
    ("RESET", 0.0, "audit"),
    # never a guess dressed up as knowledge
    ("", 0.0, "unknown"),
    (None, 0.0, "unknown"),
])
def test_flow_classifier_grid(kind, amount, expected):
    assert savings.flow_class_of(kind, amount) == expected


def test_classifier_is_deterministic_and_never_empty():
    """Every case maps to a non-empty class string; repeated calls agree."""
    cases = [("INIT", 1.0), ("TRANSFER", -1.0), ("WITHDRAWAL", -1.0),
             ("SPARPLAN_BUY", -1.0), ("SAVEBACK", 1.0), ("INTEREST_PAYOUT", 1.0),
             ("TRUEUP", 1.0), ("SELL", 0.0), ("WARN", 0.0), ("NOPE", 7.0)]
    for kind, amount in cases:
        first = savings.flow_class_of(kind, amount)
        assert isinstance(first, str) and first
        assert savings.flow_class_of(kind, amount) == first


# ---------------------------------------------------------------------------
# 2. Write-time classification by existing callers; legacy rows on read
# ---------------------------------------------------------------------------


async def test_log_event_classifies_every_existing_caller(mem_db):
    """No call site changed: the rows the engine writes now carry the class."""
    await savings.initialize(1000.0)
    await savings.add_deposit(50.0)
    await savings.trueup(1075.0)                  # +25 reconciliation delta
    await _seed_candle(mem_db, "SPY", 50.0)
    await _seed_candle(mem_db, "EURUSD=X", 1.0)
    await savings.add_saveback(10.0, "SPY")
    await savings.manual_buy(20.0, "SPY")

    # Interest payout: seed accrued interest and a previous payout month.
    async with mem_db() as s:
        acc = await s.get(SavingsAccount, 1)
        acc.accrued_interest = 3.0
        acc.last_interest_month = "2000-01"
        await s.commit()
    await savings.payout_interest()

    # One due Sparplan (executes -> investment) and one unpriceable plan
    # (WARN -> audit); both created well before the due day.
    today = savings._today_local()
    await savings.add_plan("SPY", 10.0, today.day)
    await savings.add_plan("NOPE.DE", 10.0, today.day)
    await _backdate_plan(mem_db, "SPY", 40)
    await _backdate_plan(mem_db, "NOPE.DE", 40)
    await savings.run_sparplans(day=today)

    # A manual disposal appends its SELL record (sale).
    await savings.remove_position("SPY")

    classes: dict[str, set[str]] = {}
    for e in await _events(mem_db):
        classes.setdefault(e.kind, set()).add(e.flow_class)
    assert classes == {
        "INIT": {"contribution"},
        "TRANSFER": {"contribution"},
        "TRUEUP": {"reconciliation"},
        "SAVEBACK": {"reward"},
        "MANUAL_BUY": {"investment"},
        "INTEREST_PAYOUT": {"interest"},
        "SPARPLAN_BUY": {"investment"},
        "WARN": {"audit"},
        "SELL": {"sale"},
    }


async def test_legacy_null_flow_class_classifies_on_read(mem_db):
    """Rows written before the column existed keep NULL — no backfill — and
    read through the one classifier: the negative TRANSFER stays
    reconciliation, an old WITHDRAWAL still nets out."""
    async with mem_db() as s:
        s.add(SavingsEvent(kind="INIT", amount=1000.0, note="legacy seed"))
        s.add(SavingsEvent(kind="TRANSFER", amount=-100.0, note="legacy correction"))
        s.add(SavingsEvent(kind="WITHDRAWAL", amount=-250.0, note="legacy withdrawal"))
        s.add(SavingsEvent(kind="TRUEUP", amount=50.0, note="legacy drift"))
        await s.commit()
    rows = await _events(mem_db)
    assert [r.flow_class for r in rows] == [None, None, None, None]
    assert [savings.flow_class_of(r.kind, r.amount) for r in rows] == [
        "contribution", "reconciliation", "withdrawal", "reconciliation"]
    # 1000 contribution − 250 withdrawal; the −100 correction and the +50
    # true-up are reconciliation, never capital.
    assert savings.net_contributed(rows) == pytest.approx(750.0)


# ---------------------------------------------------------------------------
# 3. Dated withdrawals
# ---------------------------------------------------------------------------


async def test_withdraw_decrements_cash_and_books_exact_dated_event(mem_db):
    await savings.initialize(1000.0)
    r = await savings.withdraw(300.0)
    assert r == {"ok": True, "amount": 300.0, "cash": 700.0,
                 "occurred_on": _operator_today()}
    assert (await savings.valuate())["cash"] == 700.0

    rows = await _events(mem_db, "WITHDRAWAL")
    assert len(rows) == 1
    e = rows[0]
    # Signed money-out, operator-local date, its own flow class, verbatim note.
    assert (e.kind, e.amount, e.occurred_on, e.flow_class, e.note) == (
        "WITHDRAWAL", -300.0, _operator_today(), "withdrawal", "")
    assert (e.ticker, e.shares, e.price, e.cost_basis) == (None, None, None, None)
    # The write-time snapshot agrees with the one classification definition.
    assert savings.flow_class_of(e.kind, e.amount) == e.flow_class == "withdrawal"

    # An explicit execution date and note are stored exactly.
    r = await savings.withdraw(50.0, occurred_on="2026-01-15", note="rent")
    assert r["occurred_on"] == "2026-01-15"
    e2 = (await _events(mem_db, "WITHDRAWAL"))[-1]
    assert (e2.amount, e2.occurred_on, e2.note, e2.flow_class) == (
        -50.0, "2026-01-15", "rent", "withdrawal")
    assert (await savings.valuate())["cash"] == 650.0


@pytest.mark.parametrize("bad", [0.0, -5.0, float("nan"), float("inf"),
                                 float("-inf"), "abc"])
async def test_withdraw_rejects_invalid_amount_without_mutation(mem_db, bad):
    await savings.initialize(1000.0)
    before = await _ledger(mem_db)
    assert await savings.withdraw(bad) == {"ok": False, "reason": "invalid amount"}
    assert (await savings.valuate())["cash"] == 1000.0
    assert await _ledger(mem_db) == before


@pytest.mark.parametrize("bad", ["2026-13-01", "01/02/2026", "20260101",
                                 "", "yesterday", "2026-1-1"])
async def test_withdraw_rejects_invalid_date_without_mutation(mem_db, bad):
    await savings.initialize(1000.0)
    before = await _ledger(mem_db)
    r = await savings.withdraw(100.0, occurred_on=bad)
    assert r == {"ok": False, "reason": "invalid date"}
    assert (await savings.valuate())["cash"] == 1000.0
    assert await _ledger(mem_db) == before


async def test_withdraw_insufficient_cash_does_not_mutate(mem_db):
    await savings.initialize(1000.0)
    before = await _ledger(mem_db)
    assert await savings.withdraw(1000.01) == {
        "ok": False, "reason": "insufficient cash"}
    assert (await savings.valuate())["cash"] == 1000.0
    assert await _ledger(mem_db) == before

    # A rejected withdrawal does not poison the next valid one; an exact-cash
    # withdrawal is allowed and leaves cash at zero.
    r = await savings.withdraw(400.0)
    assert r["ok"] is True and r["cash"] == 600.0
    r = await savings.withdraw(600.0)
    assert r["ok"] is True and r["cash"] == 0.0


async def test_withdraw_requires_an_initialized_tracker(mem_db):
    assert await savings.withdraw(10.0) == {"ok": False, "reason": "not initialized"}
    assert await _events(mem_db) == []


async def test_api_withdraw_books_and_reports_state_errors(client, mem_db):
    await savings.initialize(1000.0)
    r = await client.post("/api/savings/withdraw", json={
        "amount": 250.0, "occurred_on": "2026-01-15", "note": "rent"})
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert body["cash"] == 750.0
    assert body["occurred_on"] == "2026-01-15"

    # A state rejection is 422 (matching /api/savings/buy's class); a
    # malformed payload is 422 too; only the missing tracker is 404.
    r = await client.post("/api/savings/withdraw", json={"amount": 10_000.0})
    assert r.status_code == 422 and r.json()["detail"] == "insufficient cash"
    r = await client.post("/api/savings/withdraw", json={"amount": 0})
    assert r.status_code == 422 and r.json()["detail"] == "invalid amount"
    r = await client.post("/api/savings/withdraw",
                          json={"amount": 10, "occurred_on": "15.01.2026"})
    assert r.status_code == 422 and r.json()["detail"] == "invalid date"


async def test_api_withdraw_not_initialized_is_404(client):
    r = await client.post("/api/savings/withdraw", json={"amount": 10})
    assert r.status_code == 404 and r.json()["detail"] == "not initialized"


# ---------------------------------------------------------------------------
# 4. NET contributed capital (the audit's phantom-loss case)
# ---------------------------------------------------------------------------


async def test_deposit_then_equal_withdrawal_has_no_phantom_loss(mem_db):
    await savings.initialize(0.0)  # INIT 0 is a correction, not capital
    assert (await savings.valuate())["contributed"] == 0.0

    await savings.add_deposit(1000.0)
    val = await savings.valuate()
    assert val["contributed"] == 1000.0

    assert (await savings.withdraw(300.0))["ok"] is True
    val = await savings.valuate()
    assert val["cash"] == 700.0
    assert val["total_equity"] == 700.0
    assert val["contributed"] == 700.0  # 1000 in − 300 out
    # The audit's case: equity equals NET contributed, so the withdrawal is
    # not an investment loss. The pre-Batch-3 formula (contributed = 1000)
    # would have reported −30% here.
    assert val["total_equity"] / val["contributed"] * 100 - 100 == pytest.approx(0.0)


async def test_withdrawal_does_not_create_investment_pnl_with_a_position(mem_db):
    await savings.initialize(0.0)
    await savings.add_deposit(1000.0)
    await _seed_candle(mem_db, "SPY", 50.0)
    await _seed_candle(mem_db, "EURUSD=X", 1.0)
    assert (await savings.manual_buy(500.0, "SPY"))["ok"] is True  # 10 sh @ 50

    val = await savings.valuate()
    assert val["cash"] == 500.0
    assert val["positions_value"] == pytest.approx(500.0)
    assert val["total_equity"] == pytest.approx(1000.0)
    assert val["contributed"] == 1000.0

    assert (await savings.withdraw(200.0))["ok"] is True
    val = await savings.valuate()
    # Cash drops by the withdrawal; positions and P&L are untouched and
    # contributed drops by the same 200 — gain stays exactly 0%.
    assert val["cash"] == 300.0
    assert val["positions_value"] == pytest.approx(500.0)
    assert val["total_equity"] == pytest.approx(800.0)
    assert val["contributed"] == 800.0
    assert val["total_equity"] / val["contributed"] * 100 - 100 == pytest.approx(0.0)


async def test_trueup_interest_and_rewards_never_move_net_contributed(mem_db):
    await savings.initialize(1000.0)
    await savings.add_deposit(500.0)
    await savings.withdraw(200.0)
    assert (await savings.trueup(1400.0))["delta"] == 100.0  # drift, not capital
    val = await savings.valuate()
    assert val["cash"] == 1400.0
    assert val["contributed"] == 1300.0  # 1000 + 500 − 200


async def test_snapshot_persists_net_contributed(mem_db):
    await savings.initialize(0.0)
    await savings.add_deposit(1000.0)
    await savings.withdraw(300.0)
    assert (await savings.take_snapshot())["snapshotted"] is True
    curve = await savings.get_equity_curve(10)
    assert len(curve) == 1
    assert curve[-1]["contributed"] == 700.0
    assert curve[-1]["total_equity"] == 700.0


# ---------------------------------------------------------------------------
# 5. Savings as a metrics scope: TWR + dated XIRR
# ---------------------------------------------------------------------------


async def test_savings_metrics_response_shapes_mirror_the_existing_endpoints(client):
    """The savings endpoints return the SAME field sets as the paper ones
    (TWR additionally documents one additive drawdown field)."""
    xirr_body = (await client.get("/api/savings/metrics/xirr")).json()
    twr_body = (await client.get("/api/savings/metrics/twr")).json()
    paper_xirr = (await client.get("/api/metrics/xirr?scope=sim")).json()
    paper_twr = (await client.get("/api/metrics/twr?scope=sim")).json()

    assert set(xirr_body) == set(paper_xirr)
    assert set(twr_body) - {"max_drawdown_pct"} == set(paper_twr)
    assert "max_drawdown_pct" in twr_body
    assert xirr_body["scope"] == twr_body["scope"] == "savings"


async def test_savings_metrics_empty_account_reports_why(client):
    xirr_body = (await client.get("/api/savings/metrics/xirr")).json()
    twr_body = (await client.get("/api/savings/metrics/twr")).json()
    assert xirr_body["ok"] is True and twr_body["ok"] is True
    assert xirr_body["xirr"] is None and xirr_body["points"] == 0
    assert "no equity snapshots" in xirr_body["reason"]
    assert twr_body["twr_pct"] is None and twr_body["points"] == []
    assert twr_body["max_drawdown_pct"] is None
    assert "no equity snapshots" in twr_body["reason"]


async def test_savings_twr_flows_only_tape_is_flat_and_has_no_drawdown(client, mem_db):
    """Deposit then withdraw with no market move: 0% TWR, 0 drawdown, 0 XIRR.
    External flows are not returns."""
    await _snapshot(mem_db, date(2025, 1, 1), 1000.0)
    await _snapshot(mem_db, date(2025, 2, 1), 1500.0)   # +500 deposit
    await _snapshot(mem_db, date(2025, 3, 1), 1000.0)   # −500 withdrawal
    await _flow_event(mem_db, "INIT", 1000.0, occurred_on=date(2025, 1, 1))
    await _flow_event(mem_db, "TRANSFER", 500.0, occurred_on=date(2025, 2, 1))
    await _flow_event(mem_db, "WITHDRAWAL", -500.0, occurred_on=date(2025, 3, 1))

    twr = (await client.get("/api/savings/metrics/twr")).json()
    assert twr["twr_pct"] == pytest.approx(0.0, abs=1e-12)
    assert twr["max_drawdown_pct"] == pytest.approx(0.0, abs=1e-12)
    assert twr["nav_start"] == pytest.approx(1.0)
    assert [p["nav"] for p in twr["points"]] == pytest.approx([1.0, 1.0, 1.0])
    assert [p["return_pct"] for p in twr["points"]] == pytest.approx([0.0, 0.0, 0.0])
    assert [p["contributed"] for p in twr["points"]] == pytest.approx([1000.0, 1500.0, 1000.0])
    assert twr["start"] == "2025-01-01" and twr["end"] == "2025-03-01"

    xirr = (await client.get("/api/savings/metrics/xirr")).json()
    assert xirr["xirr"] == pytest.approx(0.0, abs=1e-12)
    assert xirr["xirr_pct"] == pytest.approx(0.0, abs=1e-12)
    assert xirr["flows"] == [
        {"date": "2025-01-01", "amount": 1000.0},
        {"date": "2025-02-01", "amount": 500.0},
        {"date": "2025-03-01", "amount": -500.0},
    ]
    assert xirr["contributed"] == pytest.approx(1000.0)
    assert xirr["final_value"] == pytest.approx(1000.0)
    assert xirr["final_date"] == "2025-03-01"
    assert xirr["window_days"] == 59
    assert xirr["reason"] is None


async def test_savings_midcurve_withdrawal_is_not_a_negative_return_day(client, mem_db):
    """+20% earned, then a 400 € withdrawal mid-curve: the invested capital
    drops, the NAV does NOT (base = prior equity + flow)."""
    await _snapshot(mem_db, date(2025, 1, 1), 1000.0)
    await _snapshot(mem_db, date(2025, 2, 1), 1200.0)   # +20%, no flow
    await _snapshot(mem_db, date(2025, 3, 1), 800.0)    # −400 withdrawal only
    await _flow_event(mem_db, "INIT", 1000.0, occurred_on=date(2025, 1, 1))
    await _flow_event(mem_db, "WITHDRAWAL", -400.0, occurred_on=date(2025, 3, 1))

    twr = (await client.get("/api/savings/metrics/twr")).json()
    returns = [p["return_pct"] for p in twr["points"]]
    assert min(returns) >= 0.0, f"a withdrawal produced a negative return day: {returns}"
    assert returns == pytest.approx([0.0, 20.0, 20.0])
    assert [p["nav"] for p in twr["points"]] == pytest.approx([1.0, 1.2, 1.2])
    assert [p["contributed"] for p in twr["points"]] == pytest.approx([1000.0, 1000.0, 600.0])
    assert twr["twr_pct"] == pytest.approx(20.0)
    assert twr["max_drawdown_pct"] == pytest.approx(0.0, abs=1e-12)


async def test_savings_xirr_uses_occurred_on_not_created_at(client, mem_db):
    """A withdrawal booked TODAY but executed in 2025 is discounted on the
    execution date (occurred_on); using created_at would drop it entirely."""
    await _snapshot(mem_db, date(2025, 1, 1), 1000.0)
    await _snapshot(mem_db, date(2025, 7, 1), 1200.0)
    await _snapshot(mem_db, date(2026, 1, 1), 900.0)
    await _flow_event(mem_db, "INIT", 1000.0, occurred_on=date(2025, 1, 1),
                      created_at=datetime.now(UTC))
    await _flow_event(mem_db, "WITHDRAWAL", -500.0, occurred_on=date(2025, 7, 1),
                      created_at=datetime.now(UTC))

    body = (await client.get("/api/savings/metrics/xirr")).json()
    assert body["flows"] == [
        {"date": "2025-01-01", "amount": 1000.0},
        {"date": "2025-07-01", "amount": -500.0},
    ]
    assert body["final_value"] == pytest.approx(900.0)
    assert body["final_date"] == "2026-01-01"
    assert body["window_days"] == 365
    assert body["contributed"] == pytest.approx(500.0)
    assert body["reason"] is None
    expected = _reference_xirr(
        [(date(2025, 1, 1), 1000.0), (date(2025, 7, 1), -500.0)],
        900.0, date(2026, 1, 1))
    assert body["xirr"] == pytest.approx(expected, abs=1e-9)

    twr = (await client.get("/api/savings/metrics/twr")).json()
    assert [p["contributed"] for p in twr["points"]] == pytest.approx(
        [1000.0, 500.0, 500.0])


async def test_savings_xirr_analytical_single_deposit(client, mem_db):
    """1000 € in on 2025-01-01, 1100 € on 2026-01-01 = exactly 10%/yr."""
    await _snapshot(mem_db, date(2025, 1, 1), 1000.0)
    await _snapshot(mem_db, date(2026, 1, 1), 1100.0)
    await _flow_event(mem_db, "INIT", 1000.0, occurred_on=date(2025, 1, 1))

    body = (await client.get("/api/savings/metrics/xirr")).json()
    assert body["xirr"] == pytest.approx(0.10, abs=1e-6)
    assert body["xirr_pct"] == pytest.approx(10.0, abs=1e-4)
    assert body["flows"] == [{"date": "2025-01-01", "amount": 1000.0}]
    assert body["points"] == 2


async def test_savings_xirr_excludes_flows_after_the_last_snapshot(client, mem_db):
    """A flow the terminal equity cannot contain (booked after the last
    snapshot) is excluded until a snapshot covers it."""
    await _snapshot(mem_db, date(2025, 1, 1), 1000.0)
    await _flow_event(mem_db, "INIT", 1000.0, occurred_on=date(2025, 1, 1))
    await _flow_event(mem_db, "WITHDRAWAL", -200.0, occurred_on=date(2025, 6, 1))

    body = (await client.get("/api/savings/metrics/xirr")).json()
    assert body["flows"] == [{"date": "2025-01-01", "amount": 1000.0}]
    assert body["final_date"] == "2025-01-01"


async def test_savings_xirr_with_only_post_window_flows_reports_why(client, mem_db):
    await _snapshot(mem_db, date(2025, 1, 1), 1000.0)
    await _flow_event(mem_db, "WITHDRAWAL", -200.0, occurred_on=date(2025, 6, 1))

    body = (await client.get("/api/savings/metrics/xirr")).json()
    assert body["flows"] == []
    assert body["xirr"] is None
    assert "no dated external flows in this window" in body["reason"]


async def test_savings_metrics_limit_truncates_the_equity_window(client, mem_db):
    await _snapshot(mem_db, date(2025, 1, 1), 1000.0)
    await _snapshot(mem_db, date(2025, 2, 1), 1100.0)
    await _snapshot(mem_db, date(2025, 3, 1), 1200.0)
    await _flow_event(mem_db, "INIT", 1000.0, occurred_on=date(2025, 1, 1))

    body = (await client.get("/api/savings/metrics/twr?limit=2")).json()
    assert [p["date"] for p in body["points"]] == ["2025-02-01", "2025-03-01"]
    assert body["start"] == "2025-02-01" and body["end"] == "2025-03-01"
    assert body["twr_pct"] == pytest.approx((1200.0 / 1100.0 - 1.0) * 100.0)
    # The flow schedule is the ledger's, so the opening capital is real
    # (no invented in-window inflow at the truncated window's first point).
    assert body["points"][0]["contributed"] == pytest.approx(1000.0)


# ---------------------------------------------------------------------------
# 6. London pence fix (quote unit BEFORE the FX pair)
# ---------------------------------------------------------------------------


async def test_price_eur_map_london_pence_before_gbp_pair(mem_db):
    await _seed_candle(mem_db, "HSBA.L", 1250.0)   # quoted in pence
    await _seed_candle(mem_db, "GBPUSD=X", 1.25)
    await _seed_candle(mem_db, "EURUSD=X", 1.10)
    prices = await savings._price_eur_map(["HSBA.L"])
    # 1250 GBp = £12.50 -> $15.625 -> €14.2045...
    assert prices["HSBA.L"] == pytest.approx(1250.0 * 0.01 * 1.25 / 1.10)
    # and not the old hundredfold error (£1250)
    assert prices["HSBA.L"] < 100.0


async def test_price_eur_map_unit_one_venues_are_bit_identical(mem_db):
    await _seed_candle(mem_db, "NESN.SW", 80.0)
    await _seed_candle(mem_db, "USDCHF=X", 0.8)
    await _seed_candle(mem_db, "IFX.DE", 30.0)
    await _seed_candle(mem_db, "AAPL", 100.0)
    await _seed_candle(mem_db, "EURUSD=X", 1.60)

    prices = await savings._price_eur_map(["NESN.SW", "IFX.DE", "AAPL"])
    # Exact (no tolerance): the unit factor 1.0 is not multiplied at all, so
    # the arithmetic is bit-for-bit the pre-fix path.
    assert prices["NESN.SW"] == 80.0 / 0.8 / 1.60
    assert prices["IFX.DE"] == 30.0
    assert prices["AAPL"] == 100.0 / 1.60


async def test_price_eur_map_london_without_gbp_pair_falls_back_to_none(mem_db):
    await _seed_candle(mem_db, "HSBA.L", 1250.0)
    await _seed_candle(mem_db, "EURUSD=X", 1.10)  # no GBPUSD=X candle
    prices = await savings._price_eur_map(["HSBA.L"])
    assert prices["HSBA.L"] is None  # caller falls back to avg_cost

    # And a .L holding valuates at its avg_cost, never at zero or a 100x price.
    await savings.initialize(100.0, positions=[
        {"ticker": "HSBA.L", "shares": 10.0, "avg_cost": 12.5}])
    val = await savings.valuate()
    assert val["positions"][0]["current_price"] == 12.5
    assert val["positions_value"] == pytest.approx(125.0)


async def test_eur_close_history_london_pence_per_date(mem_db):
    d1 = datetime(2025, 3, 3, 12, tzinfo=UTC)
    d2 = datetime(2025, 3, 4, 12, tzinfo=UTC)
    for when, px, gbp in ((d1, 1250.0, 1.25), (d2, 1300.0, 1.30)):
        await _seed_candle(mem_db, "HSBA.L", px, when=when)
        await _seed_candle(mem_db, "GBPUSD=X", gbp, when=when)
        await _seed_candle(mem_db, "EURUSD=X", 1.10, when=when)

    out = await savings._eur_close_history(["HSBA.L"])
    assert out["HSBA.L"]["2025-03-03"] == pytest.approx(1250.0 * 0.01 * 1.25 / 1.10)
    assert out["HSBA.L"]["2025-03-04"] == pytest.approx(1300.0 * 0.01 * 1.30 / 1.10)


async def test_eur_close_history_unit_one_venue_unchanged(mem_db):
    when = datetime(2025, 3, 3, 12, tzinfo=UTC)
    await _seed_candle(mem_db, "NESN.SW", 80.0, when=when)
    await _seed_candle(mem_db, "USDCHF=X", 0.8, when=when)
    await _seed_candle(mem_db, "EURUSD=X", 1.10, when=when)

    out = await savings._eur_close_history(["NESN.SW"])
    # Exact: factor 1.0 skips the multiply entirely (bit-identical).
    assert out["NESN.SW"]["2025-03-03"] == 80.0 / 0.8 / 1.10


# ---------------------------------------------------------------------------
# 7. UI: every inline script block parses
# ---------------------------------------------------------------------------


def test_inline_scripts_parse_under_node(tmp_path):
    node = shutil.which("node")
    if not node:
        pytest.skip("node is not installed")
    src = INDEX_HTML.read_text(encoding="utf-8")
    blocks = re.findall(r"<script(?![^>]*\bsrc=)[^>]*>(.*?)</script>", src, re.S)
    assert blocks, "index.html has no inline <script> blocks to check"
    for i, block in enumerate(blocks):
        path = tmp_path / f"inline-script-{i}.js"
        path.write_text(block, encoding="utf-8")
        proc = subprocess.run([node, "--check", str(path)],
                              capture_output=True, text=True, timeout=60)
        assert proc.returncode == 0, f"inline script block {i}: {proc.stderr}"
