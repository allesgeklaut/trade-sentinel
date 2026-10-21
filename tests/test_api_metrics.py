"""Tests for the read-only performance-metrics endpoints (dated XIRR + TWR).

Phase 1 "financial correctness" (audit item: "TWR charts and dated XIRR are
not delivered"):

* ``GET /api/metrics/xirr?scope=...`` — money-weighted annualized return from
  the dated external flows (day-over-day deltas of the cumulative
  contribution) and the terminal equity.
* ``GET /api/metrics/twr?scope=...`` — time-weighted return series on the
  unitized (cash-flow-adjusted) NAV, which must match
  ``app.performance.unitized_nav`` on the same snapshots.

Everything runs against an isolated in-memory SQLite database with synthetic
snapshots: no network, no provider data, no production database.
"""

from __future__ import annotations

from datetime import UTC, date, datetime

import httpx
import pytest
from sqlalchemy import StaticPool, func, select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app import daily_core, main, monthly, sim
from app.db import (
    Base,
    DailyCoreSnapshot,
    MonthlySnapshot,
    SimAllowance,
    SimSnapshot,
)
from app.performance import dated_external_flows, unitized_nav, xirr

ALL_SCOPES = ("daily_core", "monthly", "sim")

_SNAPSHOT_MODEL = {
    "sim": SimSnapshot,
    "monthly": MonthlySnapshot,
    "daily_core": DailyCoreSnapshot,
}


# ---------------------------------------------------------------------------
# fixtures
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

    import app.db as db_mod

    monkeypatch.setattr(db_mod, "Session", session_factory)
    # The snapshot readers use their module-level Session import.
    for mod in (sim, monthly, daily_core):
        monkeypatch.setattr(mod, "Session", session_factory)

    yield session_factory
    await engine.dispose()


@pytest.fixture
async def client(mem_db):
    transport = httpx.ASGITransport(app=main.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


async def seed(session_factory, scope: str, rows: list[tuple[date, float, float]]) -> None:
    """Insert snapshot rows ``(date, total_equity, contributed)`` for a scope."""
    model = _SNAPSHOT_MODEL[scope]
    async with session_factory() as s:
        for day, equity, contributed in rows:
            s.add(model(
                created_at=datetime(day.year, day.month, day.day, 12, tzinfo=UTC),
                cash=0.0,
                positions_value=equity,
                total_equity=equity,
                allowance_total=contributed,
            ))
        await s.commit()


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def reference_xirr(flows: list[tuple[date, float]], final_value: float,
                   final_date: date) -> float:
    """Independent bisection on the same definition, written for the test.

    Deliberately a separate implementation (simple, unoptimised bisection) so
    a bug in ``app.performance.xirr``'s solver or conventions cannot be
    "confirmed" by itself.
    """
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


MIXED_ROWS: list[tuple[date, float, float]] = [
    # contributed: 1000 -> +1000 deposit; 2000 -> +1000 deposit;
    # 1500 -> -500 withdrawal; final equity 1800.
    (date(2025, 1, 1), 1000.0, 1000.0),
    (date(2025, 4, 1), 1500.0, 2000.0),
    (date(2025, 7, 1), 1200.0, 1500.0),
    (date(2025, 12, 31), 1800.0, 1500.0),
]


# ---------------------------------------------------------------------------
# XIRR: analytic and mixed schedules
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("scope", ALL_SCOPES)
async def test_xirr_analytic_single_deposit(scope, client, mem_db):
    """1000 in on 2025-01-01, 1100 on 2026-01-01 (365 days) = exactly 10%/yr."""
    await seed(mem_db, scope, [
        (date(2025, 1, 1), 1000.0, 1000.0),
        (date(2026, 1, 1), 1100.0, 1000.0),
    ])

    r = await client.get(f"/api/metrics/xirr?scope={scope}")
    assert r.status_code == 200
    body = r.json()

    assert body["ok"] is True
    assert body["scope"] == scope
    assert body["points"] == 2
    assert body["xirr"] == pytest.approx(0.10, abs=1e-6)
    assert body["xirr_pct"] == pytest.approx(10.0, abs=1e-4)
    assert body["final_value"] == pytest.approx(1100.0)
    assert body["final_date"] == "2026-01-01"
    assert body["contributed"] == pytest.approx(1000.0)
    assert body["window_days"] == 365
    # Only the initial funding is a flow: the second point changed nothing.
    assert body["flows"] == [{"date": "2025-01-01", "amount": 1000.0}]
    assert body["reason"] is None


async def test_xirr_mixed_deposits_and_withdrawal(client, mem_db):
    """A deposit, a second deposit, a withdrawal, then growth to 1800.

    The rate is solved to tolerance and independently reproduced by the test's
    own bisection; NPV at the returned rate is ~0.
    """
    await seed(mem_db, "daily_core", MIXED_ROWS)

    r = await client.get("/api/metrics/xirr?scope=daily_core")
    body = r.json()

    assert body["flows"] == [
        {"date": "2025-01-01", "amount": 1000.0},
        {"date": "2025-04-01", "amount": 1000.0},
        {"date": "2025-07-01", "amount": -500.0},
    ]
    expected = reference_xirr(
        [(date(2025, 1, 1), 1000.0), (date(2025, 4, 1), 1000.0), (date(2025, 7, 1), -500.0)],
        1800.0, date(2025, 12, 31))
    assert body["xirr"] == pytest.approx(expected, abs=1e-6)
    assert body["xirr"] == pytest.approx(0.2012, abs=1e-3)
    assert body["reason"] is None

    # NPV at the returned annualized rate is zero (the actual/365 definition).
    start = date(2025, 1, 1)
    npv = sum(
        -amount / (1.0 + body["xirr"]) ** (((day - start).days) / 365.0)
        for day, amount in [
            (date(2025, 1, 1), 1000.0),
            (date(2025, 4, 1), 1000.0),
            (date(2025, 7, 1), -500.0),
        ]
    ) + 1800.0 / (1.0 + body["xirr"]) ** ((date(2025, 12, 31) - start).days / 365.0)
    assert npv == pytest.approx(0.0, abs=1e-6)


async def test_deposit_only_schedule_has_zero_xirr(client, mem_db):
    """Two deposits, no growth: the money earned exactly 0%."""
    await seed(mem_db, "monthly", [
        (date(2025, 1, 1), 1000.0, 1000.0),
        (date(2025, 6, 1), 2000.0, 2000.0),
        (date(2025, 12, 1), 2000.0, 2000.0),
    ])

    body = (await client.get("/api/metrics/xirr?scope=monthly")).json()

    assert body["flows"] == [
        {"date": "2025-01-01", "amount": 1000.0},
        {"date": "2025-06-01", "amount": 1000.0},
    ]
    assert body["xirr"] == 0.0          # exactly zero, not a solver artefact
    assert body["xirr_pct"] == 0.0
    assert body["reason"] is None


async def test_total_loss_is_not_annualized(client, mem_db):
    """A wiped-out account has no NPV=0 rate: the answer is undefined, not 0%."""
    await seed(mem_db, "sim", [
        (date(2025, 1, 1), 1000.0, 1000.0),
        (date(2025, 6, 1), 0.0, 1000.0),
    ])

    body = (await client.get("/api/metrics/xirr?scope=sim")).json()

    assert body["xirr"] is None
    assert body["xirr_pct"] is None
    assert body["reason"] and "NPV" in body["reason"]

    # The TWR side reports the honest -100% (a terminal wipeout is a return).
    twr = (await client.get("/api/metrics/twr?scope=sim")).json()
    assert twr["twr_pct"] == pytest.approx(-100.0)


# ---------------------------------------------------------------------------
# XIRR: no-flow and empty-account edges
# ---------------------------------------------------------------------------

async def test_no_external_flows_is_undefined_not_zero(client, mem_db):
    """Equity moved but nothing was ever contributed: no money-weighted rate.

    There is no dated external flow to weigh, so XIRR must be null (0% would
    claim the cash-flow schedule earned nothing). TWR still measures the
    portfolio's own return.
    """
    await seed(mem_db, "sim", [
        (date(2025, 1, 1), 1000.0, 0.0),
        (date(2025, 6, 1), 1200.0, 0.0),
    ])

    body = (await client.get("/api/metrics/xirr?scope=sim")).json()

    assert body["flows"] == []
    assert body["xirr"] is None
    assert body["xirr_pct"] is None
    assert "no dated external flows" in body["reason"]

    twr = (await client.get("/api/metrics/twr?scope=sim")).json()
    assert twr["twr_pct"] == pytest.approx(20.0)


async def test_empty_account_returns_nulls_not_an_error(client, mem_db):
    for scope in ALL_SCOPES:
        r = await client.get(f"/api/metrics/xirr?scope={scope}")
        assert r.status_code == 200
        body = r.json()
        assert body["xirr"] is None
        assert body["points"] == 0
        assert body["final_value"] is None
        assert body["flows"] == []
        assert "no equity snapshots" in body["reason"]

        r = await client.get(f"/api/metrics/twr?scope={scope}")
        assert r.status_code == 200
        body = r.json()
        assert body["points"] == []
        assert body["twr_pct"] is None
        assert body["start"] is None
        assert body["end"] is None
        assert "no equity snapshots" in body["reason"]


async def test_invalid_scope_is_rejected(client, mem_db):
    for path in ("/api/metrics/xirr", "/api/metrics/twr"):
        r = await client.get(f"{path}?scope=bogus")
        assert r.status_code == 422
        assert "invalid scope" in r.json()["detail"]


# ---------------------------------------------------------------------------
# TWR: the series must be the unitized NAV itself
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("scope", ALL_SCOPES)
async def test_twr_series_matches_unitized_nav(scope, client, mem_db):
    rows = [
        (date(2025, 1, 1), 1000.0, 1000.0),
        (date(2025, 2, 1), 1100.0, 1000.0),   # +10%
        (date(2025, 3, 1), 2100.0, 2000.0),   # +1000 deposit, NAV unchanged
        (date(2025, 4, 1), 1680.0, 2000.0),   # -20% on 2100 -> NAV 0.88
    ]
    await seed(mem_db, scope, rows)

    equity = [e for _, e, _ in rows]
    contributed = [c for _, _, c in rows]
    expected_nav = unitized_nav(equity, contributed)

    body = (await client.get(f"/api/metrics/twr?scope={scope}")).json()

    assert [p["nav"] for p in body["points"]] == pytest.approx(expected_nav, abs=1e-12)
    assert [p["date"] for p in body["points"]] == [
        "2025-01-01", "2025-02-01", "2025-03-01", "2025-04-01"]
    assert body["nav_start"] == pytest.approx(1.0)
    assert body["start"] == "2025-01-01"
    assert body["end"] == "2025-04-01"
    # Cumulative TWR of the window: last NAV / first funded NAV - 1.
    assert body["twr_pct"] == pytest.approx((expected_nav[-1] - 1.0) * 100.0, abs=1e-9)
    assert body["twr_pct"] == pytest.approx(-12.0, abs=1e-9)
    assert body["reason"] is None


async def test_twr_is_cash_flow_neutral(client, mem_db):
    """A deposit that changes no portfolio value is not a return."""
    await seed(mem_db, "monthly", [
        (date(2025, 1, 1), 1000.0, 1000.0),
        (date(2025, 2, 1), 1200.0, 1000.0),   # +20% market move
        (date(2025, 3, 1), 2200.0, 2000.0),   # +1000 deposit, no price move
    ])

    body = (await client.get("/api/metrics/twr?scope=monthly")).json()

    assert [p["nav"] for p in body["points"]] == pytest.approx([1.0, 1.2, 1.2])
    assert body["twr_pct"] == pytest.approx(20.0)
    for point in body["points"]:
        assert point["return_pct"] is not None


async def test_intraday_duplicate_snapshots_collapse_to_the_last(client, mem_db):
    """The monthly/daily-core tables can hold two marks for one day; the last
    one wins, matching what the charts plot."""
    await seed(mem_db, "daily_core", [
        (date(2025, 1, 1), 1000.0, 1000.0),
        (date(2025, 2, 1), 1050.0, 1000.0),
        (date(2025, 2, 1), 1100.0, 1000.0),   # post-rebalance mark, same day
    ])

    body = (await client.get("/api/metrics/twr?scope=daily_core")).json()

    assert [p["date"] for p in body["points"]] == ["2025-01-01", "2025-02-01"]
    assert [p["nav"] for p in body["points"]] == pytest.approx([1.0, 1.1])
    assert body["points"][0]["return_pct"] == pytest.approx(0.0)
    assert body["points"][1]["return_pct"] == pytest.approx(10.0)
    assert body["points"][1]["total_equity"] == pytest.approx(1100.0)
    assert body["twr_pct"] == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# read-only guarantee + helper unit edges
# ---------------------------------------------------------------------------

async def test_metrics_endpoints_are_read_only(client, mem_db):
    """No allowance deposit, no snapshot write: calling the metrics must not
    mutate a paper portfolio's history."""
    await seed(mem_db, "sim", [
        (date(2025, 1, 1), 1000.0, 1000.0),
        (date(2026, 1, 1), 1100.0, 1000.0),
    ])

    for scope in ALL_SCOPES:
        await client.get(f"/api/metrics/xirr?scope={scope}")
        await client.get(f"/api/metrics/twr?scope={scope}")

    async with mem_db() as s:
        snapshots = (await s.execute(
            select(func.count()).select_from(SimSnapshot))).scalar_one()
        allowances = (await s.execute(
            select(func.count()).select_from(SimAllowance))).scalar_one()
    assert snapshots == 2
    assert allowances == 0


def test_xirr_helper_rejects_length_mismatch():
    with pytest.raises(ValueError, match="length mismatch"):
        xirr([date(2025, 1, 1)], [1000.0, 500.0], 1100.0, date(2026, 1, 1))
    with pytest.raises(ValueError, match="length mismatch"):
        dated_external_flows([date(2025, 1, 1)], [1000.0, 500.0])


def test_dated_external_flows_skips_invalid_observations():
    """A bad contribution is not a withdrawal: the last valid baseline holds."""
    flows = dated_external_flows(
        [date(2025, 1, 1), date(2025, 2, 1), date(2025, 3, 1), date(2025, 4, 1)],
        [1000.0, float("nan"), 1500.0, 1500.0],
    )
    assert flows == [(date(2025, 1, 1), 1000.0), (date(2025, 3, 1), 500.0)]


def test_xirr_helper_is_undefined_without_a_terminal_value():
    assert xirr([date(2025, 1, 1)], [1000.0], None, date(2026, 1, 1)) is None
    assert xirr([], [], 0.0, date(2026, 1, 1)) is None


async def test_restarted_nav_chain_has_no_single_cumulative_twr(client, mem_db):
    """A wipeout followed by a fresh deposit restarts the NAV at 1.0.

    There is no single honest cumulative TWR across two chains, so the scalar
    is null with a reason while the per-point NAV still shows both.
    """
    await seed(mem_db, "sim", [
        (date(2025, 1, 1), 1000.0, 1000.0),
        (date(2025, 6, 1), 0.0, 1000.0),       # total loss: NAV -> 0
        (date(2025, 7, 1), 1000.0, 2000.0),    # fresh deposit: NAV restarts
    ])

    body = (await client.get("/api/metrics/twr?scope=sim")).json()

    assert [p["nav"] for p in body["points"]] == pytest.approx([1.0, 0.0, 1.0])
    assert body["twr_pct"] is None
    assert body["reason"] and "restarted" in body["reason"]
    assert body["points"][1]["return_pct"] == pytest.approx(-100.0)


async def test_limit_truncates_the_window_consistently(client, mem_db):
    """``limit`` keeps the most recent N days and treats the first kept
    cumulative contribution as the window's opening inflow."""
    await seed(mem_db, "sim", [
        (date(2025, 1, 1), 0.0, 0.0),           # account exists, not funded yet
        (date(2025, 2, 1), 1000.0, 1000.0),     # funding point = window start
        (date(2026, 2, 1), 1100.0, 1000.0),
    ])

    body = (await client.get("/api/metrics/xirr?scope=sim&limit=2")).json()

    assert body["points"] == 2
    assert body["flows"] == [{"date": "2025-02-01", "amount": 1000.0}]
    assert body["final_value"] == pytest.approx(1100.0)
    assert body["window_days"] == 365
    # 1000 -> 1100 over one year with no new money: exactly 10%/yr.
    assert body["xirr"] == pytest.approx(0.10, abs=1e-6)
