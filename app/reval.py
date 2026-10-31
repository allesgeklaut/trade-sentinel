"""Phase 4 (slice 4A) strategy revaluation harness — predeclared arm comparison.

Purpose
-------
``docs/ui-strategy-audit-2026-10-21.md`` Phase 4 (lines 341-360) demands a
comparison of a small PREDECLARED set of strategies under realistic
spread/slippage/FX/cash-yield assumptions, reporting TWR, dated XIRR, proper
NAV drawdown, volatility, excess-return Sharpe, turnover, time in cash,
concentration and benchmark differences — because the prior evidence was not
decisive (zero costs, cash earning zero, Sharpe with an implicit rf=0,
survivorship-biased universes; lines 262-286). This module is that harness,
built on top of the repo's deterministic in-memory research backtests
(``app.optimize``) and the shared metrics (``app.performance``).

It is ANALYSIS-ONLY and OFFLINE-ONLY:

* it never writes the database — ``point_at_database`` opens the target with
  SQLite ``PRAGMA query_only=ON`` so a write attempt fails at the DB level,
  and ``tests/test_reval.py`` proves every table's row counts are unchanged;
* it never calls a market-data provider or an LLM; every engine runs from the
  stored candles/fundamentals of the ``--db`` snapshot.

The predeclared arm set (all over the SAME window, the SAME monthly
contribution schedule and the SAME declared costs unless a per-arm override is
recorded in the report):

1. ``dca``        — broad-market DCA control: each month's contribution buys
                    the benchmark ETF (``settings.sim_benchmark_ticker``,
                    default URTH) on the first trading session of the month at
                    that session's strict exact-date OPEN (no fill -> the cash
                    is carried to the next session), fractional shares, cost
                    class "etf". Implemented in-memory from the candles — it
                    deliberately does NOT call ``sim.backfill_benchmark``
                    (which writes the DB).
2. ``monthly``    — the monthly factor strategy (qv-mom-v1: target_n=10,
                    hold_band=20) via ``optimize._monthly_backtest``. WRAPPER:
                    at zero cost / zero cash yield its results are identical
                    to calling the engine directly (pinned by a test).
3. ``daily_core`` — the SAME factor policy with immediate cash deployment via
                    ``optimize._daily_core_backtest`` (dca="rank", boost=0,
                    target_n=10, hold_band=20, all risk overlays OFF = the
                    frozen clean baseline). Same wrapper-parity requirement.
4. ``sim``        — the technical strategy as an experimental comparator via
                    ``optimize._replay`` in CONTRACT mode with
                    ``optimize._live_sim_params()`` and the new optional
                    ``cost_oneway`` hook (a ``ticker -> one-way rate``
                    callable, applied exactly like ``_daily_core_backtest``'s
                    fee wrapper; default 0.0 keeps the old behaviour).

Controls scaffold: arms of kind ``random_n`` (seeded) and ``equal_weight`` are
accepted by the interface and raise a declared ``NotImplementedError``
("coming in slice 4B") that the CLI reports gracefully. They are NOT
implemented here.

Declared assumptions (all recorded verbatim in the report header)
-----------------------------------------------------------------
* ``CostConfig``: ``stock_oneway_bps`` (default 10) and ``etf_oneway_bps``
  (default 2) are one-way per-fill rates on fill notional; ``fx_oneway_bps``
  (default 15) is added ONLY for fills in non-USD listings (any ticker whose
  suffix is not a known USD notation — ``app.instruments.is_usd_native``);
  ``cash_yield_annual`` (default 0.03) is a POST-HOC declared assumption
  credited on the session cash curve (it is NOT what the live paper accounts
  accrue — they are fee-free and their cash earns 0%).
* Cost plumbing per engine (where the brief and the engines' existing
  conventions conflict, the existing conventions win — each case is also
  listed in the report): ``_daily_core_backtest`` accepts only a flat
  ``cost_oneway``, so the stock leg goes through that wrapper exactly and the
  FX leg is a post-hoc NAV debit on the per-leg trade ledger (exact per fill,
  but it does not reduce investable cash mid-window); ``_monthly_backtest``
  reads a flat ``settings.sim_monthly_cost_oneway``, so the monthly arm
  declares one flat rate = stock + fx * (non-USD share of the universe) on
  swapped notional; ``_replay`` (new hook) and the in-memory ``dca`` arm apply
  the full per-ticker rate at fill time.
* The cash yield is credited as interest on a shadow cash balance that
  compounds at ``(1+cy)**(1/252)`` per session and passes cash-flow changes
  through at par, and is ADDED TO EQUITY FOR METRICS ONLY — the engines' trade
  sizes are unaffected (it measures "what if parked cash sat in a money
  market"). The unitized NAV is re-run on the yield-adjusted equity.
* ``configs_tried`` is caller-managed and recorded (audit requirement: the
  number of configurations tried must be reported with any comparison).

Honesty limits (must be stated by any 4C report built on this)
-------------------------------------------------------------
* SURVIVORSHIP/SELECTION BIAS: universes are current-membership snapshots
  (``screener.universe_path``), so ABSOLUTE comparisons against the broad-
  market DCA control carry selection bias; RELATIVE comparisons across arms
  sharing the same universe are robust to it (all arms here share one
  universe, one session axis and one contribution schedule).
* The cash yield is a post-hoc declared assumption, not an observed accrual.
* The monthly engine is an aggregate-value model (equal-weight basket, marks
  only at fill sessions + the window's last close — its own docstring
  discloses "no intra-period marks"); its session series here is the step
  function of those marks, so its intra-month volatility is understated by
  construction (flagged per-arm in the report as ``marks_only``).
* The monthly arm's FX leg is folded into its declared flat rate (see above),
  not charged per fill.
* The Sharpe's risk-free proxy is the DECLARED cash yield (excess-return
  series), not an observed rate; at ``cash_yield=0`` it degenerates to the old
  implicit rf=0 Sharpe.
* No "winner" language: the report renders numbers and deltas only (audit
  finding #4).

Report: JSON schema ``"reval-report/1"``, sorted keys, caller-supplied
``generated_at``; two runs with identical inputs (including ``generated_at``)
produce byte-identical JSON (pinned by a test). A markdown renderer writes the
summary table + declared assumptions + caveats next to the JSON.

CLI::

    python -m app.reval run --db sqlite+aiosqlite:////tmp/reval/trading.db \
        --start 2024-01-01 --end 2026-09-30 --universe diversified-plus \
        [--universes-dir DIR] [--arms dca,monthly,daily_core,sim] \
        [--cash-yield 0.03] [--stock-bps 10] [--etf-bps 2] [--fx-bps 15] \
        [--out DIR] [--seed N]

``--db`` is required for ``run``: the harness points this process at that
snapshot (read-only) instead of the ambient ``DATABASE_URL`` — mirror of how
``app.optimize``'s CLI uses the ambient engine, with the explicit override.
Library-first: everything is callable as plain functions from tests.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import io
import json
import math
import sys
from contextlib import asynccontextmanager, contextmanager, redirect_stdout
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, UTC
from pathlib import Path
from typing import Any

import pandas as pd

SCHEMA = "reval-report/1"
TRADING_DAYS_PER_YEAR = 252.0
ARM_KINDS = ("dca", "monthly", "daily_core", "sim", "random_n", "equal_weight")
CONTROL_KINDS = ("random_n", "equal_weight")

__all__ = [
    "SCHEMA",
    "CostConfig",
    "ArmSpec",
    "RevalConfig",
    "ArmRun",
    "resolve_universe",
    "point_at_database",
    "universes_override",
    "default_arms",
    "run_reval",
    "run_reval_async",
    "report_json",
    "render_markdown",
    "apply_cash_yield",
    "twr_metrics",
    "xirr_metric",
    "drawdown_metrics",
    "underwater_spell",
    "annualized_vol",
    "excess_sharpe",
    "time_in_cash",
    "concentration_metrics",
    "turnover_metrics",
    "main",
]


# ---------------------------------------------------------------------------
# Configuration (all of it recorded verbatim in the report header)
# ---------------------------------------------------------------------------

@dataclass
class CostConfig:
    """Declared cost/yield assumptions. Rates are ONE-WAY fractions of fill
    notional; ``fx_oneway_bps`` applies only to non-USD listings."""

    stock_oneway_bps: float = 10.0
    etf_oneway_bps: float = 2.0
    fx_oneway_bps: float = 15.0
    cash_yield_annual: float = 0.03

    def rate_for(self, ticker: str, kind: str = "stock") -> float:
        """One-way rate for a fill in ``ticker`` (cost class ``kind``)."""
        from .instruments import is_usd_native
        base = (self.etf_oneway_bps if kind == "etf" else self.stock_oneway_bps) / 1e4
        if not is_usd_native(ticker):
            base += self.fx_oneway_bps / 1e4
        return base


@dataclass
class ArmSpec:
    """One predeclared arm. ``params`` is engine-specific and recorded verbatim."""

    name: str
    kind: str  # dca | monthly | daily_core | sim | random_n | equal_weight
    params: dict[str, Any] = field(default_factory=dict)
    cost: dict[str, float] | None = None  # per-arm bps override; None -> CostConfig


@dataclass
class RevalConfig:
    """The full run configuration. ``configs_tried`` is caller-managed (the
    audit requires the count of configurations tried to travel with the
    report)."""

    db_url: str
    start: str
    end: str
    universe: str
    arms: list[ArmSpec]
    costs: CostConfig = field(default_factory=CostConfig)
    seeds: dict[str, int] = field(default_factory=lambda: {"run": 1})
    configs_tried: int = 1
    contribution: float = 1000.0
    benchmark: str = "URTH"
    start_cash: float = 0.0
    universes_dir: str | None = None
    generated_at: str | None = None
    schema_version: str = SCHEMA


@dataclass
class ArmRun:
    """One arm's assembled result: series + declared cost + engine summary."""

    spec: ArmSpec
    declared_cost: dict[str, Any]
    dates: list[str]
    equity_raw: list[float]
    cash: list[float]
    invested: list[float]
    equity: list[float]          # yield-adjusted (metrics use this)
    interest: list[float]        # credited cash yield per session
    nav: list[float]
    trades: list[dict]
    weights: list[dict[str, float]]
    engine_summary: dict[str, Any]
    metrics: dict[str, Any] = field(default_factory=dict)


def default_arms() -> list[ArmSpec]:
    """The predeclared four-arm set (Phase 4 items 1-4)."""
    return [
        ArmSpec("dca", "dca", {"role": "broad-market DCA control",
                               "benchmark_fill": "first session of month, strict open"}),
        ArmSpec("monthly", "monthly", {"policy": "qv-mom-v1", "target_n": 10,
                                       "hold_band": 20}),
        ArmSpec("daily_core", "daily_core", {"policy": "qv-mom-v1 + immediate deployment",
                                             "dca": "rank", "boost": 0.0, "target_n": 10,
                                             "hold_band": 20, "entry_gate": "none",
                                             "exit_cadence": "monthly",
                                             "crash_pause": False, "overlays": "off"}),
        ArmSpec("sim", "sim", {"mode": "CONTRACT", "params": "_live_sim_params",
                               "role": "technical strategy, experimental comparator"}),
    ]


# ---------------------------------------------------------------------------
# Read-only database / universe resolution
# ---------------------------------------------------------------------------

@asynccontextmanager
async def point_at_database(db_url: str):
    """Point this process at ``db_url`` READ-ONLY for the block.

    Mirrors how ``app.optimize``'s CLI resolves the DB (the ambient engine in
    ``app.db``), with the explicit override: the module-global ``app.db.engine``
    is swapped for a fresh engine built from ``db_url`` and restored on exit
    (``Session()``/``active_engine()`` resolve it at call time). For SQLite
    URLs the connection additionally sets ``PRAGMA query_only=ON``, so ANY
    write — INSERT/UPDATE/DELETE/DDL, even a journal-mode change — fails at
    the database level. The harness never calls ``create_all``/``init_db``.
    """
    from sqlalchemy import event
    from sqlalchemy.ext.asyncio import create_async_engine
    import app.db as db_mod

    eng = create_async_engine(db_url, connect_args={"timeout": 5}, pool_pre_ping=True)
    if db_url.startswith("sqlite"):
        def _query_only(dbapi_conn, _record) -> None:
            cur = dbapi_conn.cursor()
            cur.execute("PRAGMA query_only=ON")
            cur.close()
        event.listen(eng.sync_engine, "connect", _query_only)
    old = db_mod.engine
    db_mod.engine = eng
    db_mod._sessionmakers.pop("prod", None)
    try:
        yield eng
    finally:
        db_mod.engine = old
        db_mod._sessionmakers.pop("prod", None)
        await eng.dispose()


@contextmanager
def universes_override(universes_dir: str | None):
    """Temporarily resolve universe files from ``universes_dir`` first
    (``screener.universe_path`` semantics: the extra dir shadows the repo's
    ``universes/`` dir). ``None`` keeps the ambient resolution."""
    from . import screener
    if universes_dir is None:
        yield
        return
    old = screener._EXTRA_UNIVERSES_DIR
    screener._EXTRA_UNIVERSES_DIR = Path(universes_dir)
    try:
        yield
    finally:
        screener._EXTRA_UNIVERSES_DIR = old


def resolve_universe(name: str, universes_dir: str | None = None) -> dict:
    """Resolve a universe name to its members + sha256 (of the sorted member
    list) + path, via ``screener.universe_path`` semantics."""
    from . import screener
    with universes_override(universes_dir):
        p = screener.universe_path(name)
        if p is None:
            raise ValueError(f"unknown universe {name!r} (looked in the extra dir "
                             f"and the repo universes/)")
        members = screener.tickers(name)
    members_sorted = sorted(members)
    sha = hashlib.sha256("\n".join(members_sorted).encode("utf-8")).hexdigest()
    return {"name": name, "members": members_sorted, "count": len(members_sorted),
            "sha256": sha, "path": str(p)}


@contextmanager
def _settings_overrides(**kw):
    """Temporarily pin ``settings`` fields (the repo's own CLI-override
    pattern, e.g. ``optimize._main`` setting ``sim_daily_core_mom_variant``)."""
    from .config import settings
    old = {k: getattr(settings, k) for k in kw}
    for k, v in kw.items():
        setattr(settings, k, v)
    try:
        yield
    finally:
        for k, v in old.items():
            setattr(settings, k, v)


# ---------------------------------------------------------------------------
# Pure metric functions (mirror the existing app conventions; see docstrings)
# ---------------------------------------------------------------------------

def apply_cash_yield(cash: list[float], equity: list[float],
                     cash_yield_annual: float) -> tuple[list[float], list[float]]:
    """Post-hoc money-market credit on the session cash curve.

    A shadow balance ``S`` compounds at ``g = (1+cy)**(1/252)`` per session and
    passes cash-flow changes through at par: ``S_i = (S_{i-1} + cash_i -
    cash_{i-1}) * g``. The credited interest ``S_i - cash_i`` is ADDED TO
    EQUITY FOR METRICS ONLY (engine trade sizes are unaffected). A constant
    cash balance ``C`` held for ``k`` sessions therefore accrues exactly
    ``C * ((1+cy)**(k/252) - 1)``. At ``cy = 0`` the equity is returned
    unchanged.
    """
    if cash_yield_annual == 0.0:
        return list(equity), [0.0] * len(equity)
    g = (1.0 + cash_yield_annual) ** (1.0 / TRADING_DAYS_PER_YEAR)
    s = 0.0
    prev = 0.0
    out: list[float] = []
    interest: list[float] = []
    for c, e in zip(cash, equity, strict=True):
        s = (s + (c - prev)) * g
        prev = c
        interest.append(s - c)
        out.append(e + (s - c))
    return out, interest


def twr_metrics(dates: list[str], nav: list[float]) -> dict:
    """TWR mirroring ``/api/metrics/twr`` (app/main.py:1254-1323) exactly for
    the cumulative part — chained unitized NAV from ``performance.unitized_nav``,
    base = first funded NAV, null when the chain restarted after a wipeout —
    then annualized over the window with ACTUAL/365 day counts."""
    start_index = next((i for i, v in enumerate(nav) if v > 0.0), None)
    base = nav[start_index] if start_index is not None else None
    zeros = [i for i in range((start_index or 0) + 1, len(nav))
             if nav[i] == 0.0 and nav[i - 1] > 0.0]
    restarted = bool(zeros) and any(v > 0.0 for v in nav[zeros[-1] + 1:])
    cum: float | None = None
    if base is not None and not restarted:
        cum = nav[-1] / base - 1.0
    window_days = 0
    if len(dates) > 1:
        window_days = (date.fromisoformat(dates[-1]) - date.fromisoformat(dates[0])).days
    ann: float | None = None
    if cum is not None:
        if window_days > 0:
            ann = (1.0 + cum) ** (365.0 / window_days) - 1.0
        elif cum == 0.0:
            ann = 0.0
    return {"twr_cumulative": cum, "twr_annualized": ann,
            "window_days": window_days, "nav_restarted": restarted}


def xirr_metric(dates: list[str], invested: list[float],
                final_equity: float | None) -> float | None:
    """Dated XIRR mirroring ``/api/metrics/xirr`` (app/main.py:1180-1251):
    flows = ``performance.dated_external_flows(dates, invested)`` (day-over-day
    deltas of the cumulative contribution series), terminal value = final
    (yield-adjusted) equity on the final date, solved by
    ``performance.xirr`` (ACTUAL/365, Excel XIRR convention)."""
    from .performance import dated_external_flows, xirr
    if not dates or final_equity is None:
        return None
    flows = dated_external_flows(dates, invested)
    if not flows:
        return None
    return xirr([d for d, _ in flows], [a for _, a in flows], final_equity, dates[-1])


def underwater_spell(nav: list[float]) -> dict:
    """Longest underwater spell of the unitized NAV in SESSIONS
    (peak-to-recovery); a spell still open at the window end counts as
    ongoing."""
    if not nav:
        return {"sessions": 0, "ongoing": False}
    peak = nav[0]
    peak_i = 0
    open_i: int | None = None
    best = 0
    best_ongoing = False
    for i, v in enumerate(nav):
        if v > peak:
            peak, peak_i = v, i
        if open_i is not None and v >= peak:
            spell = i - open_i
            if spell > best:
                best, best_ongoing = spell, False
            open_i = None
        elif v < peak and open_i is None:
            open_i = peak_i
    if open_i is not None:
        spell = len(nav) - 1 - open_i
        if spell >= best:
            best, best_ongoing = spell, True
    return {"sessions": best, "ongoing": best_ongoing}


def drawdown_metrics(nav: list[float]) -> dict:
    """Max drawdown on the (yield-adjusted) unitized NAV via
    ``performance.max_drawdown`` (percent, the shared cash-flow-neutral
    definition), plus the longest underwater spell."""
    from .performance import max_drawdown
    return {"max_drawdown_pct": max_drawdown(nav) if nav else 0.0,
            "underwater": underwater_spell(nav)}


def _daily_simple_returns(nav: list[float]) -> list[float]:
    return [nav[i] / nav[i - 1] - 1.0 for i in range(1, len(nav))
            if nav[i] > 0.0 and nav[i - 1] > 0.0]


def annualized_vol(nav: list[float]) -> float | None:
    """Annualized std of daily simple returns of the unitized NAV (x sqrt(252),
    sample std ddof=1 — the same estimator ``optimize._sharpe`` uses)."""
    r = _daily_simple_returns(nav)
    if len(r) < 2:
        return None
    m = sum(r) / len(r)
    var = sum((x - m) ** 2 for x in r) / (len(r) - 1)
    return math.sqrt(var) * math.sqrt(TRADING_DAYS_PER_YEAR)


def excess_sharpe(nav: list[float], cash_yield_annual: float) -> float | None:
    """Excess-return Sharpe: mean(daily unitized return - daily cash yield) /
    std(same excess series) * sqrt(252), where the daily cash yield is
    ``(1+cy)**(1/252) - 1``. The risk-free proxy is the DECLARED cash yield
    (audit finding #5: the old Sharpe assumed an implicit rf=0)."""
    r = _daily_simple_returns(nav)
    if len(r) < 2:
        return None
    rf = (1.0 + cash_yield_annual) ** (1.0 / TRADING_DAYS_PER_YEAR) - 1.0
    ex = [x - rf for x in r]
    m = sum(ex) / len(ex)
    var = sum((x - m) ** 2 for x in ex) / (len(ex) - 1)
    sd = math.sqrt(var)
    if sd == 0.0:
        return 0.0 if m == 0.0 else None
    return m / sd * math.sqrt(TRADING_DAYS_PER_YEAR)


def time_in_cash(cash: list[float], equity: list[float]) -> dict:
    """Mean cash weight (mean over sessions of cash/equity) AND the share of
    sessions fully invested (cash weight <= 5%). Both are reported."""
    ws = [c / e for c, e in zip(cash, equity, strict=True) if e > 0]
    if not ws:
        return {"mean_cash_weight": None, "pct_sessions_fully_invested": None,
                "sessions": 0}
    fully = sum(1 for w in ws if w <= 0.05)
    return {"mean_cash_weight": sum(ws) / len(ws),
            "pct_sessions_fully_invested": fully / len(ws), "sessions": len(ws)}


def concentration_metrics(weights: list[dict[str, float]]) -> dict:
    """Mean/max over sessions of the largest position weight and of the HHI
    (sum of squared position weights)."""
    tops = [max(w.values()) if w else 0.0 for w in weights]
    hhis = [sum(v * v for v in w.values()) for w in weights]
    n = len(weights) or 1
    return {"mean_top_weight": sum(tops) / n, "max_top_weight": max(tops) if tops else 0.0,
            "mean_hhi": sum(hhis) / n, "max_hhi": max(hhis) if hhis else 0.0}


def turnover_metrics(dates: list[str], trades: list[dict], equity: list[float],
                     contrib_by_date: dict[str, float]) -> dict:
    """Annualized one-way turnover, mirroring the research ``avg_turnover``
    convention in ``app/optimize.py``: per calendar month, one-way churn =
    ((buys - fresh contribution) + sells) / 2, relative to the equity at the
    month's first session (the daily-core convention at optimize.py:3613-3627;
    the monthly baseline's slot-swap turn at optimize.py:2737-2752); the
    reported rate is the mean churn over months, annualized x12."""
    if not dates:
        return {"buys_notional": 0.0, "sells_notional": 0.0, "one_way_notional": 0.0,
                "avg_monthly_oneway": 0.0, "annualized": 0.0, "months": 0}
    date_equity = dict(zip(dates, equity, strict=True))
    first_of_month: dict[tuple[int, int], str] = {}
    for d in dates:
        key = (int(d[:4]), int(d[5:7]))
        first_of_month.setdefault(key, d)
    buys_m: dict[tuple[int, int], float] = {}
    sells_m: dict[tuple[int, int], float] = {}
    for t in trades:
        key = (int(t["date"][:4]), int(t["date"][5:7]))
        if t["side"] == "BUY":
            buys_m[key] = buys_m.get(key, 0.0) + float(t["notional"])
        else:
            sells_m[key] = sells_m.get(key, 0.0) + float(t["notional"])
    churns: list[float] = []
    one_way = 0.0
    for key, d0 in sorted(first_of_month.items()):
        eq0 = date_equity.get(d0, 0.0)
        contrib = sum(v for dd, v in contrib_by_date.items()
                      if (int(dd[:4]), int(dd[5:7])) == key)
        b = buys_m.get(key, 0.0)
        s = sells_m.get(key, 0.0)
        one_way += (b + s) / 2.0
        if eq0 > 0:
            churns.append(((max(b - contrib, 0.0) + s) / 2.0) / eq0)
    avg = sum(churns) / len(churns) if churns else 0.0
    return {"buys_notional": sum(buys_m.values()), "sells_notional": sum(sells_m.values()),
            "one_way_notional": one_way, "avg_monthly_oneway": avg,
            "annualized": avg * 12.0, "months": len(churns)}


# ---------------------------------------------------------------------------
# Input loading (read-only, via the swapped engine)
# ---------------------------------------------------------------------------

@dataclass
class _Inputs:
    universe: dict
    members: list[str]
    open_px: pd.DataFrame      # USD-converted strict open frame (factor engines)
    close: pd.DataFrame        # USD-converted close frame
    vol: pd.DataFrame
    raw_close: pd.DataFrame    # RAW listing-currency closes (sim arm marks)
    fund: dict
    bench: dict[str, dict]     # "YYYY-MM-DD" -> {"open", "close"} (raw, benchmark)
    axis: list[str]            # canonical session axis "YYYY-MM-DD"
    contrib_days: set          # pd.Timestamp, first session of each window month
    contrib_by_date: dict[str, float]
    db_info: dict


def _naive_ts(ts) -> pd.Timestamp:
    """Stored candle timestamps read back tz-aware UTC; the frames are naive
    (same normalization ``monthly.load_frames_sync`` applies)."""
    ts = pd.Timestamp(ts)
    if ts.tzinfo is not None:
        ts = ts.tz_localize(None)
    return ts.normalize()


async def _load_inputs(cfg: RevalConfig, uni: dict) -> _Inputs:
    from sqlalchemy import func, select
    from . import fundamentals as fundamentals_mod
    from . import monthly as monthly_mod
    from . import optimize
    from .db import Candle, Session

    members = uni["members"]
    open_px, close, vol = await monthly_mod.load_ohlcv_frames(members)
    fund = await fundamentals_mod.load_fundamentals(members)
    async with Session() as s:
        bench_rows = (await s.execute(
            select(Candle.timestamp, Candle.open, Candle.close)
            .where(Candle.ticker == cfg.benchmark)
            .order_by(Candle.timestamp))).all()
        raw_rows = (await s.execute(
            select(Candle.ticker, Candle.timestamp, Candle.close)
            .where(Candle.ticker.in_(members))
            .order_by(Candle.ticker, Candle.timestamp))).all()
        rng = (await s.execute(
            select(func.min(Candle.timestamp), func.max(Candle.timestamp), func.count())
            .select_from(Candle)
            .where(Candle.ticker.in_(members + [cfg.benchmark])))).one()
    bench = {r.timestamp.strftime("%Y-%m-%d"): {"open": r.open, "close": r.close}
             for r in bench_rows}
    raw_close = pd.DataFrame(
        {t: pd.Series({_naive_ts(r.timestamp): r.close
                       for r in raw_rows if r.ticker == t})
         for t in members}).sort_index()

    lo, hi = pd.Timestamp(cfg.start), pd.Timestamp(cfg.end)
    window = pd.DatetimeIndex([d for d in close.index if lo <= d <= hi])
    bench_days = [pd.Timestamp(d) for d in bench if lo <= pd.Timestamp(d) <= hi]
    axis_ts = sorted(set(window) | set(bench_days))
    axis = [d.strftime("%Y-%m-%d") for d in axis_ts]
    contrib_days = optimize._contribution_days(
        pd.DatetimeIndex(axis_ts), {(d.year, d.month) for d in axis_ts})
    contrib_by_date = {d.strftime("%Y-%m-%d"): cfg.contribution for d in contrib_days}

    path: str | None = None
    if cfg.db_url.startswith("sqlite"):
        tail = cfg.db_url.split("///", 1)[-1] if "///" in cfg.db_url else ":memory:"
        path = ":memory:" if tail == ":memory:" else tail
    db_info = {
        "url": cfg.db_url,
        "path": path if path is not None else ":memory:",
        "candle_range": [rng[0].strftime("%Y-%m-%d") if rng[0] else None,
                         rng[1].strftime("%Y-%m-%d") if rng[1] else None],
        "candle_rows": int(rng[2]),
        "read_only": True,
    }
    return _Inputs(universe=uni, members=members, open_px=open_px, close=close,
                   vol=vol, raw_close=raw_close, fund=fund, bench=bench, axis=axis,
                   contrib_days=contrib_days, contrib_by_date=contrib_by_date,
                   db_info=db_info)


# ---------------------------------------------------------------------------
# Arm runners
# ---------------------------------------------------------------------------

def _arm_costs(arm: ArmSpec, costs: CostConfig) -> CostConfig:
    if arm.cost is None:
        return costs
    return CostConfig(
        stock_oneway_bps=arm.cost.get("stock_oneway_bps", costs.stock_oneway_bps),
        etf_oneway_bps=arm.cost.get("etf_oneway_bps", costs.etf_oneway_bps),
        fx_oneway_bps=arm.cost.get("fx_oneway_bps", costs.fx_oneway_bps),
        cash_yield_annual=arm.cost.get("cash_yield_annual", costs.cash_yield_annual),
    )


def _reindex_to_axis(arm_dates: list[str], series: list[list[float]],
                     axis: list[str]) -> list[list[float]]:
    """Carry each arm's series onto the canonical session axis (forward carry;
    leading sessions before the arm's first point take its first values)."""
    pos = {d: i for i, d in enumerate(arm_dates)}
    out = [[] for _ in series]
    last = [0] * len(series)
    for d in axis:
        i = pos.get(d)
        if i is None:
            j = max((k for k in pos if k <= d), default=None)
            i = pos[j] if j is not None else 0
        for k, s in enumerate(series):
            last[k] = s[i]
            out[k].append(s[i])
    return out


def _run_dca_arm(cfg: RevalConfig, arm: ArmSpec, inputs: _Inputs,
                 costs: CostConfig) -> ArmRun:
    """Broad-market DCA control, in-memory from the benchmark candles.

    Each month's contribution is received on the first session of the month
    and buys the benchmark ETF at that session's strict exact-date OPEN
    (``optimize._fill_buy`` wrapper, cost class "etf"); a session without an
    executable open gets NO fill and the cash is carried to the next session.
    Fractional shares (4dp floor via the shared ``factor_fills`` primitive).
    """
    from .instruments import is_usd_native
    from .optimize import _fill_buy

    rate = costs.etf_oneway_bps / 1e4
    if not is_usd_native(cfg.benchmark):
        rate += costs.fx_oneway_bps / 1e4
    contrib = inputs.contrib_days
    shares = 0.0
    cash = 0.0
    invested = 0.0
    dates: list[str] = []
    equity_raw: list[float] = []
    cash_s: list[float] = []
    invested_s: list[float] = []
    trades: list[dict] = []
    weights: list[dict[str, float]] = []
    last_close: float | None = None
    for d in inputs.axis:
        ts = pd.Timestamp(d)
        if d == inputs.axis[0] and cfg.start_cash:
            cash += cfg.start_cash
            invested += cfg.start_cash
        if ts in contrib:
            cash += cfg.contribution
            invested += cfg.contribution
        bar = inputs.bench.get(d)
        op = bar["open"] if bar else None
        if op is not None and math.isfinite(op) and op > 0 and cash >= 1.0:
            fill = _fill_buy(cash, float(op), round(cash, 2), rate)
            if fill is not None:
                notional, sh, cash = fill
                shares += sh
                trades.append({"date": d, "side": "BUY", "ticker": cfg.benchmark,
                               "notional": notional, "shares": sh, "price": float(op)})
        if bar is not None and bar["close"]:
            last_close = float(bar["close"])
        mark = last_close or 0.0
        eq = cash + shares * mark
        dates.append(d)
        equity_raw.append(eq)
        cash_s.append(cash)
        invested_s.append(invested)
        weights.append({cfg.benchmark: shares * mark / eq} if eq > 0 else {})
    declared = {"mode": "per-ticker at fill (etf class)",
                "etf_oneway_bps": costs.etf_oneway_bps,
                "fx_oneway_bps": costs.fx_oneway_bps if not is_usd_native(cfg.benchmark) else 0.0,
                "benchmark": cfg.benchmark}
    return _assemble(cfg, arm, inputs, costs, declared, dates, equity_raw, cash_s,
                     invested_s, trades, weights,
                     {"fills": len(trades), "shares": shares, "cost_class": "etf"})


async def _run_monthly_arm(cfg: RevalConfig, arm: ArmSpec, inputs: _Inputs,
                           costs: CostConfig) -> ArmRun:
    """Monthly factor arm — a WRAPPER around ``optimize._monthly_backtest``.

    The official results come from one full-window engine call. The session
    series is assembled from the engine's own outputs: prefix calls at each
    fill session give the aggregate mark values (the engine marks only at
    fills + the window's last close — disclosed as ``marks_only``), the picks
    sequence (via the engine's own ``monthly.pick_portfolio`` on the shared
    frame cache) gives the holdings/turns, and the parked-cash schedule
    follows the engine's settle rules. At zero cost/yield the summary is
    identical to a direct ``_monthly_backtest`` call (pinned by a test).

    Cost plumbing (existing convention wins): the engine reads one flat
    ``settings.sim_monthly_cost_oneway``, so this arm declares the flat rate
    ``stock + fx * non_usd_member_share`` on swapped notional.
    """
    from . import monthly as monthly_mod
    from . import optimize
    from .instruments import is_usd_native

    close, vol, open_px = inputs.close, inputs.vol, inputs.open_px
    idx_all = pd.DatetimeIndex(close.index)
    lo, hi = pd.Timestamp(cfg.start), pd.Timestamp(cfg.end)
    idx = idx_all[(idx_all >= lo) & (idx_all <= hi)]
    if not len(idx):
        raise ValueError(f"no sessions in window {cfg.start}..{cfg.end}")
    months = monthly_mod.completed_month_ends(close.index, idx[0], idx[-1])
    fill_days: list[pd.Timestamp] = []
    for me in months:
        later = idx[idx > pd.Timestamp(me)]
        if not len(later):
            break
        fill_days.append(later[0])

    target_n = int(arm.params.get("target_n", 10))
    hold_band = int(arm.params.get("hold_band", 20))
    nonusd = [t for t in inputs.members if not is_usd_native(t)]
    nonusd_share = len(nonusd) / max(len(inputs.members), 1)
    flat_bps = costs.stock_oneway_bps + costs.fx_oneway_bps * nonusd_share
    flat_rate = flat_bps / 1e4

    frame_cache: dict[pd.Timestamp, pd.DataFrame | None] = {}
    with _settings_overrides(sim_monthly_cost_oneway=flat_rate,
                             sim_monthly_target_n=target_n,
                             sim_monthly_hold_band=hold_band,
                             sim_daily_core_mom_variant="raw"):
        with redirect_stdout(io.StringIO()):
            summary = await optimize._monthly_backtest(
                cfg.start, cfg.end, cfg.universe, cfg.contribution, False,
                frame_cache=frame_cache, close_preloaded=(close, vol),
                fund_preloaded=inputs.fund, open_preloaded=open_px)
            if summary is None:
                raise ValueError("monthly backtest returned no summary for the window")
            marks: list[tuple[pd.Timestamp, float]] = []
            for fd in fill_days:
                pre = await optimize._monthly_backtest(
                    cfg.start, fd.strftime("%Y-%m-%d"), cfg.universe, cfg.contribution,
                    False, frame_cache=frame_cache, close_preloaded=(close, vol),
                    fund_preloaded=inputs.fund, open_preloaded=open_px)
                if pre is not None:
                    marks.append((fd, pre.final_value))

        # Picks sequence via the engine's own picker on the shared cache.
        holdings: list[str] = []
        picks_by_fill: dict[pd.Timestamp, list[str]] = {}
        turns: list[float] = []
        for me, fd in zip(months, fill_days, strict=False):
            frame = frame_cache.get(me)
            if frame is not None:
                picks, _ = monthly_mod.pick_portfolio(me, close, vol, inputs.fund,
                                                      holdings, frame)
            else:
                picks = holdings
            adds = len(set(picks) - set(holdings))
            drops = len(set(holdings) - set(picks))
            turns.append((adds + drops) / max(max(len(picks), len(holdings)), 1))
            picks_by_fill[fd] = list(picks)
            holdings = list(picks)

    # Parked-cash schedule (the engine's settle/deploy rules).
    first_days = sorted(optimize._contribution_days(idx, {(d.year, d.month) for d in idx}))
    booked: set[tuple[int, int]] = set()
    parked = 0.0
    mark_cash: dict[pd.Timestamp, float] = {}
    for me, fd in zip(months, fill_days, strict=False):
        for day in first_days:
            key = (day.year, day.month)
            if key not in booked and day <= fd:
                parked += cfg.contribution
                booked.add(key)
        frame = frame_cache.get(me)
        live = frame is not None and bool(frame["eligible"].any())
        if live:
            parked = 0.0
        mark_cash[fd] = parked
    for day in first_days:
        key = (day.year, day.month)
        if key not in booked and day <= idx[-1]:
            parked += cfg.contribution
            booked.add(key)
    mark_cash[idx[-1]] = parked

    mark_value = {fd: v for fd, v in marks}
    mark_value[idx[-1]] = summary.final_value

    # Synthetic per-month legs (the aggregate engine has no per-leg ledger):
    # swapped notional at fill k = turn_k * mark value, split over the legs.
    trades: list[dict] = []
    for k, (_me, fd) in enumerate(zip(months, fill_days, strict=False)):
        picks = picks_by_fill.get(fd, [])
        prev = picks_by_fill.get(fill_days[k - 1], []) if k else []
        adds = [t for t in picks if t not in prev]
        drops = [t for t in prev if t not in picks]
        n_legs = len(adds) + len(drops)
        if n_legs and turns[k] > 0:
            leg = turns[k] * mark_value.get(fd, 0.0) / n_legs
            for t in adds:
                trades.append({"date": fd.strftime("%Y-%m-%d"), "side": "BUY",
                               "ticker": t, "notional": leg, "shares": 0.0, "price": 0.0})
            for t in drops:
                trades.append({"date": fd.strftime("%Y-%m-%d"), "side": "SELL",
                               "ticker": t, "notional": leg, "shares": 0.0, "price": 0.0})

    # Step series over the canonical axis.
    dates: list[str] = []
    equity_raw: list[float] = []
    cash_s: list[float] = []
    invested_s: list[float] = []
    weights: list[dict[str, float]] = []
    mark_ts = sorted(mark_value)
    cash_ts = sorted(mark_cash)
    for d in inputs.axis:
        ts = pd.Timestamp(d)
        mv = [m for m in mark_ts if m <= ts]
        mc = [m for m in cash_ts if m <= ts]
        invested = cfg.contribution * sum(1 for cd in inputs.contrib_days if cd <= ts)
        if mv:
            eq = mark_value[mv[-1]]
            cash = mark_cash[mc[-1]] if mc else 0.0
        else:
            # everything still parked before the first mark (the factor engines
            # start unfunded; a non-zero start_cash is not representable here)
            eq = invested
            cash = invested
        dates.append(d)
        equity_raw.append(eq)
        cash_s.append(cash)
        invested_s.append(invested)
        fd = [m for m in picks_by_fill if m <= ts]
        if fd:
            picks = picks_by_fill[fd[-1]]
            weights.append({t: 1.0 / len(picks) for t in picks} if picks else {})
        else:
            weights.append({})

    declared = {"mode": "flat one-way rate on swapped notional (engine convention: "
                         "settings.sim_monthly_cost_oneway)",
                "oneway_bps": flat_bps,
                "stock_oneway_bps": costs.stock_oneway_bps,
                "fx_oneway_bps_folded": costs.fx_oneway_bps * nonusd_share,
                "non_usd_member_share": nonusd_share,
                "approximation": "FX leg folded into the flat rate (no per-leg ledger "
                                 "in the aggregate engine)"}
    engine_summary = {"irr": summary.irr, "final_value": summary.final_value,
                      "contributed": summary.contributed,
                      "avg_turnover": summary.avg_turnover, "n_trades": summary.n_trades,
                      "window": list(summary.window), "marks_only": True,
                      "marks": [[m.strftime("%Y-%m-%d"), v] for m, v in marks],
                      "replayed_avg_turnover": (sum(turns) / len(turns)) if turns else 0.0,
                      "start_cash_note": ("factor engines start unfunded; start_cash "
                                          "applies to the dca/sim arms only"
                                          if cfg.start_cash else None)}
    return _assemble(cfg, arm, inputs, costs, declared, dates, equity_raw, cash_s,
                     invested_s, trades, weights, engine_summary)


async def _run_daily_core_arm(cfg: RevalConfig, arm: ArmSpec, inputs: _Inputs,
                              costs: CostConfig) -> ArmRun:
    """Daily-core arm (same factor policy, immediate cash deployment) — a
    WRAPPER around ``optimize._daily_core_backtest`` with dca="rank", boost=0,
    target_n/hold_band from the arm params and ALL risk overlays OFF (frozen
    clean baseline).

    Cost plumbing (existing convention wins): the stock leg goes through the
    engine's own flat ``cost_oneway`` wrapper exactly (mirroring how
    ``_daily_core_backtest`` applies it around ``_fill_buy``); the FX leg is a
    post-hoc NAV debit computed per fill from the engine's per-leg trade
    ledger (exact per fill, but not compounded into mid-window cash). The
    session series is reconstructed from that ledger + the engine's EOD mark
    policy (fresh forward-filled close, else cost basis), which reproduces
    the engine's own equity marks.
    """
    from . import monthly as monthly_mod
    from . import optimize
    from . import price_freshness
    from .config import settings
    from .instruments import is_usd_native

    close, vol, open_px = inputs.close, inputs.vol, inputs.open_px
    target_n = int(arm.params.get("target_n", 10))
    hold_band = int(arm.params.get("hold_band", 20))
    stock_rate = costs.stock_oneway_bps / 1e4
    fx_rate = costs.fx_oneway_bps / 1e4

    with _settings_overrides(
            sim_monthly_target_n=target_n, sim_monthly_hold_band=hold_band,
            sim_monthly_rank_boost=float(arm.params.get("boost", 0.0)),
            sim_daily_core_mom_variant="raw", sim_daily_core_lowvol_tilt=False,
            sim_daily_core_vol_weight=False, sim_daily_core_target_vol=0.0,
            sim_daily_core_portfolio_stop=0.0, sim_daily_core_exposure_trend=0,
            sim_daily_core_trailing_stop=0.0, sim_daily_core_basket_trend=0,
            sim_daily_core_basket_confirm=3, sim_daily_core_basket_threshold=0.0,
            sim_daily_core_basket_drawdown=0.0, sim_daily_core_basket_er_min=0.0,
            sim_daily_core_basket_good_times=False, sim_daily_core_basket_arm_sma=200):
        with redirect_stdout(io.StringIO()):
            summary = await optimize._daily_core_backtest(
                cfg.start, cfg.end, cfg.universe, cfg.contribution, False,
                arm.params.get("entry_gate", "none"),
                arm.params.get("exit_cadence", "monthly"),
                bool(arm.params.get("crash_pause", False)),
                arm.params.get("dca", "rank"),
                frame_cache=None, close_preloaded=(close, vol),
                fund_preloaded=inputs.fund, open_preloaded=open_px,
                cost_oneway=stock_rate, with_baseline=False)
    if summary is None:
        raise ValueError("daily-core backtest returned no summary for the window")

    # --- reconstruct the daily series from the engine's trade ledger ------
    idx_all = pd.DatetimeIndex(close.index)
    lo, hi = pd.Timestamp(cfg.start), pd.Timestamp(cfg.end)
    days = idx_all[(idx_all >= lo) & (idx_all <= hi)]
    contrib_days = optimize._contribution_days(days, {(d.year, d.month) for d in days})
    fresh = price_freshness.freshness_mask(close, None, settings.factor_price_max_age_days)
    close_val = close.ffill()

    def eod_mark(t: str, d: pd.Timestamp, cost_basis: dict[str, float]) -> float:
        try:
            is_fresh = bool(fresh.at[d, t])
        except KeyError:
            is_fresh = False
        if not is_fresh:
            return cost_basis.get(t, 0.0) or 0.0
        p = monthly_mod.px_at(close_val, d, t)
        return p if p is not None else 0.0

    trades_by_date: dict[str, list[dict]] = {}
    for t in summary.trades:
        trades_by_date.setdefault(t["date"], []).append(t)

    shares: dict[str, float] = {}
    cost_basis: dict[str, float] = {}
    cash = 0.0
    invested = 0.0
    dates: list[str] = []
    equity_raw: list[float] = []
    cash_s: list[float] = []
    invested_s: list[float] = []
    weights: list[dict[str, float]] = []
    fx_fee_cum = 0.0
    fx_fee: list[float] = []
    for d in days:
        ds = d.strftime("%Y-%m-%d")
        if d in contrib_days:
            cash += cfg.contribution
            invested += cfg.contribution
        # NOTE: the factor engines start unfunded (cash=0, contributions only),
        # so a non-zero start_cash is not representable in them; it applies to
        # the dca and sim arms only (recorded in engine_summary below).
        for tr in trades_by_date.get(ds, []):
            t, notional = tr["ticker"], float(tr["notional"])
            if tr["side"] == "BUY":
                cash -= notional * (1.0 + stock_rate)
                prev = shares.get(t, 0.0)
                shares[t] = prev + float(tr["shares"])
                cost_basis[t] = ((prev * cost_basis.get(t, 0.0) + notional) / shares[t]
                                 if prev > 0 else float(tr["price"]))
            else:
                cash += notional * (1.0 - stock_rate)
                shares.pop(t, None)
                cost_basis.pop(t, None)
            if not is_usd_native(t):
                fx_fee_cum += notional * fx_rate
        eq = cash + sum(shares[t] * eod_mark(t, d, cost_basis) for t in sorted(shares))
        dates.append(ds)
        equity_raw.append(eq)
        cash_s.append(cash)
        invested_s.append(invested)
        fx_fee.append(fx_fee_cum)
        weights.append({t: shares[t] * eod_mark(t, d, cost_basis) / eq
                        for t in sorted(shares)} if eq > 0 else {})

    # Post-hoc FX debit is reported separately and applied to the adjusted
    # equity (engine raw equity stays untouched in ``equity_raw``).
    declared = {"mode": "stock leg flat in engine (cost_oneway wrapper) + FX leg "
                        "post-hoc on the per-leg ledger",
                "stock_oneway_bps": costs.stock_oneway_bps,
                "fx_oneway_bps": costs.fx_oneway_bps,
                "fx_post_hoc_notional": sum(float(t["notional"]) for t in summary.trades
                                            if not is_usd_native(t["ticker"]))}
    engine_summary = {"irr": summary.irr, "final_value": summary.final_value,
                      "contributed": summary.contributed,
                      "avg_turnover": summary.avg_turnover, "n_trades": summary.n_trades,
                      "window": list(summary.window), "sharpe": summary.sharpe,
                      "max_drawdown": summary.max_drawdown, "marks_only": False,
                      "reconstructed_final_equity": equity_raw[-1] if equity_raw else 0.0,
                      "start_cash_note": ("factor engines start unfunded; start_cash "
                                          "applies to the dca/sim arms only"
                                          if cfg.start_cash else None)}
    return _assemble(cfg, arm, inputs, costs, declared, dates, equity_raw, cash_s,
                     invested_s, list(summary.trades), weights, engine_summary,
                     post_hoc_debit=fx_fee)


async def _run_sim_arm(cfg: RevalConfig, arm: ArmSpec, inputs: _Inputs,
                       costs: CostConfig) -> ArmRun:
    """Technical strategy as experimental comparator: ``optimize._replay`` in
    CONTRACT mode (series carry the strict session bars via
    ``optimize._load_series``) with ``optimize._live_sim_params()`` and the
    new ``cost_oneway`` hook fed the per-ticker rate (stock class + FX for
    non-USD listings), applied at fill time exactly like the daily-core fee
    wrapper."""
    from . import optimize

    series = await optimize._load_series(
        inputs.members, start=pd.Timestamp(cfg.start).to_pydatetime())
    if not series:
        raise ValueError("no series with enough candle history for the sim arm")
    params = optimize._live_sim_params()
    params.monthly_allowance = cfg.contribution
    params.start_cash = cfg.start_cash

    def rate_fn(t: str) -> float:
        return costs.rate_for(t, "stock")

    with redirect_stdout(io.StringIO()):
        res = optimize._replay(series, params, start=cfg.start, end=cfg.end,
                               cost_oneway=rate_fn)
    dates = [e["time"] for e in res.equity_curve]
    equity_raw = [float(e["equity"]) for e in res.equity_curve]
    cash_s = [float(c) for c in res.cash_curve]
    # invested: start_cash + allowance on the first session of each month
    # (the replay's own deposit rule), reconstructed over its walk days.
    invested_s: list[float] = []
    cum = cfg.start_cash
    prev_month: str | None = None
    for d in dates:
        if d[:7] != prev_month:
            cum += cfg.contribution
            prev_month = d[:7]
        invested_s.append(cum)

    # Holdings weights from the trade ledger, marked at raw closes carried
    # forward (the replay's own mark convention).
    raw = inputs.raw_close
    shares: dict[str, float] = {}
    trades_by_date: dict[str, list[dict]] = {}
    for t in res.trades:
        trades_by_date.setdefault(t["date"], []).append(t)
    weights: list[dict[str, float]] = []
    last_known: dict[str, float] = {}
    for i, d in enumerate(dates):
        for tr in trades_by_date.get(d, []):
            if tr["side"] == "BUY":
                shares[tr["ticker"]] = shares.get(tr["ticker"], 0.0) + float(tr["shares"])
            else:
                shares[tr["ticker"]] = shares.get(tr["ticker"], 0.0) - float(tr["shares"])
                if shares[tr["ticker"]] <= 0:
                    shares.pop(tr["ticker"], None)
        ts = pd.Timestamp(d)
        for t in list(shares):
            if t in raw.columns and ts in raw[t].index:
                last_known[t] = float(raw[t].at[ts])
        eq = equity_raw[i]
        weights.append({t: shares[t] * last_known.get(t, 0.0) / eq
                        for t in sorted(shares)} if eq > 0 else {})

    declared = {"mode": "per-ticker at fill via _replay's cost_oneway hook",
                "stock_oneway_bps": costs.stock_oneway_bps,
                "fx_oneway_bps": costs.fx_oneway_bps}
    engine_summary = {"final_equity": res.final_equity,
                      "total_return_pct": res.total_return_pct, "sharpe": res.sharpe,
                      "max_drawdown_pct": res.max_drawdown_pct, "n_trades": res.n_trades,
                      "marks_only": False, "contract_mode": True,
                      "params": {"max_positions": params.max_positions,
                                 "stop_type": params.stop_type,
                                 "stop_pct": params.stop_pct,
                                 "use_atr_stop": params.use_atr_stop,
                                 "max_run_5d": params.max_run_5d,
                                 "monthly_allowance": params.monthly_allowance,
                                 "start_cash": params.start_cash}}
    trades = [{"date": t["date"], "side": t["side"], "ticker": t["ticker"],
               "notional": float(t["shares"]) * float(t["price"]),
               "shares": float(t["shares"]), "price": float(t["price"])}
              for t in res.trades]
    return _assemble(cfg, arm, inputs, costs, declared, dates, equity_raw, cash_s,
                     invested_s, trades, weights, engine_summary)


def _assemble(cfg: RevalConfig, arm: ArmSpec, inputs: _Inputs, costs: CostConfig,
              declared: dict, dates: list[str], equity_raw: list[float],
              cash: list[float], invested: list[float], trades: list[dict],
              weights: list[dict[str, float]], engine_summary: dict,
              post_hoc_debit: list[float] | None = None) -> ArmRun:
    """Reindex to the canonical axis, apply the declared post-hoc cash yield
    (and any post-hoc cost debit, e.g. the FX leg), re-run the unitized NAV on
    the adjusted equity, compute metrics."""
    from .performance import unitized_nav
    eq, cash_r, inv = _reindex_to_axis(dates, [equity_raw, cash, invested], inputs.axis)
    debit = None
    if post_hoc_debit:
        debit = _reindex_to_axis(dates, [post_hoc_debit], inputs.axis)[0]
    # weights: carry the holdings forward per axis date
    wpos = {d: i for i, d in enumerate(dates)}
    w_out: list[dict[str, float]] = []
    for d in inputs.axis:
        i = wpos.get(d)
        if i is None:
            j = max((k for k in wpos if k <= d), default=None)
            i = wpos[j] if j is not None else 0
        w_out.append(dict(weights[i]))
    equity_adj, interest = apply_cash_yield(cash_r, eq, costs.cash_yield_annual)
    if debit is not None:
        equity_adj = [e - d for e, d in zip(equity_adj, debit, strict=True)]
    nav = unitized_nav(equity_adj, inv)
    run = ArmRun(spec=arm, declared_cost=declared, dates=list(inputs.axis),
                 equity_raw=eq, cash=cash_r, invested=inv, equity=equity_adj,
                 interest=interest, nav=nav, trades=trades, weights=w_out,
                 engine_summary=engine_summary)
    run.metrics = _arm_metrics(cfg, run, costs, inputs)
    return run


def _arm_metrics(cfg: RevalConfig, run: ArmRun, costs: CostConfig,
                 inputs: _Inputs) -> dict:
    twr = twr_metrics(run.dates, run.nav)
    dd = drawdown_metrics(run.nav)
    to = turnover_metrics(run.dates, run.trades, run.equity, inputs.contrib_by_date)
    return {
        "twr_cumulative": twr["twr_cumulative"],
        "twr_annualized": twr["twr_annualized"],
        "twr_window_days": twr["window_days"],
        "xirr": xirr_metric(run.dates, run.invested,
                           run.equity[-1] if run.equity else None),
        "max_drawdown_pct": dd["max_drawdown_pct"],
        "longest_underwater_sessions": dd["underwater"]["sessions"],
        "underwater_ongoing": dd["underwater"]["ongoing"],
        "vol_annualized": annualized_vol(run.nav),
        "sharpe_excess": excess_sharpe(run.nav, costs.cash_yield_annual),
        "rf_proxy": "declared cash_yield_annual (post-hoc assumption)",
        "rf_annual": costs.cash_yield_annual,
        "mean_cash_weight": time_in_cash(run.cash, run.equity)["mean_cash_weight"],
        "pct_sessions_fully_invested": time_in_cash(run.cash, run.equity)["pct_sessions_fully_invested"],
        "concentration": concentration_metrics(run.weights),
        "turnover": to,
        "final_equity": run.equity[-1] if run.equity else None,
        "final_equity_raw": run.equity_raw[-1] if run.equity_raw else None,
        "n_trades": len(run.trades),
    }


async def _run_arm(cfg: RevalConfig, arm: ArmSpec, inputs: _Inputs) -> ArmRun:
    costs = _arm_costs(arm, cfg.costs)
    if arm.kind in CONTROL_KINDS:
        raise NotImplementedError(
            f"arm {arm.name!r} (kind {arm.kind!r}) is a declared control scaffold "
            f"for slice 4B ({'seeded random-N' if arm.kind == 'random_n' else 'equal-weight'} "
            f"control) — not implemented yet in 4A")
    if arm.kind == "dca":
        return _run_dca_arm(cfg, arm, inputs, costs)
    if arm.kind == "monthly":
        return await _run_monthly_arm(cfg, arm, inputs, costs)
    if arm.kind == "daily_core":
        return await _run_daily_core_arm(cfg, arm, inputs, costs)
    if arm.kind == "sim":
        return await _run_sim_arm(cfg, arm, inputs, costs)
    raise ValueError(f"unknown arm kind {arm.kind!r} (choose from {ARM_KINDS})")


# ---------------------------------------------------------------------------
# Report assembly
# ---------------------------------------------------------------------------

def _jsonable(obj):
    """Recursively convert NaN/inf floats to None so ``allow_nan=False`` JSON
    is always writable and deterministic."""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    return obj


async def run_reval_async(cfg: RevalConfig) -> dict:
    """Run every arm over the window and return the report dict (async)."""
    from .config import settings
    uni = resolve_universe(cfg.universe, cfg.universes_dir)
    with universes_override(cfg.universes_dir):
        async with point_at_database(cfg.db_url):
            inputs = await _load_inputs(cfg, uni)
            arms: dict[str, ArmRun] = {}
            for arm in cfg.arms:
                arms[arm.name] = await _run_arm(cfg, arm, inputs)
    return _build_report(cfg, uni, inputs, arms,
                         ambient={"sim_universe": settings.sim_universe,
                                  "sim_monthly_universe": settings.sim_monthly_universe,
                                  "sim_benchmark_ticker": settings.sim_benchmark_ticker,
                                  "sim_monthly_contribution": settings.sim_monthly_contribution})


def run_reval(cfg: RevalConfig) -> dict:
    """Sync wrapper around :func:`run_reval_async` (CLI convenience)."""
    return asyncio.run(run_reval_async(cfg))


_METRIC_ROWS = [
    ("final_equity", "final equity ($, yield-adj)", "{:.2f}"),
    ("twr_annualized", "TWR annualized", "{:+.4f}"),
    ("twr_cumulative", "TWR cumulative", "{:+.4f}"),
    ("xirr", "XIRR (dated, annualized)", "{:+.4f}"),
    ("max_drawdown_pct", "max drawdown % (unitized NAV)", "{:.2f}"),
    ("longest_underwater_sessions", "longest underwater (sessions)", "{}"),
    ("vol_annualized", "volatility (ann, x sqrt252)", "{:.4f}"),
    ("sharpe_excess", "Sharpe (excess over declared cash yield)", "{:+.3f}"),
    ("mean_cash_weight", "mean cash weight", "{:.4f}"),
    ("pct_sessions_fully_invested", "share sessions fully invested", "{:.3f}"),
    ("turnover_annualized", "one-way turnover (ann, x12)", "{:.4f}"),
    ("n_trades", "trade count", "{}"),
    ("conc_mean_top_weight", "mean largest position weight", "{:.4f}"),
    ("conc_max_top_weight", "max largest position weight", "{:.4f}"),
    ("conc_mean_hhi", "mean HHI", "{:.4f}"),
    ("conc_max_hhi", "max HHI", "{:.4f}"),
]


def _metric_value(metrics: dict, key: str):
    if key == "turnover_annualized":
        return metrics["turnover"]["annualized"]
    if key.startswith("conc_"):
        return metrics["concentration"][key[5:]]
    return metrics[key]


def _build_report(cfg: RevalConfig, uni: dict, inputs: _Inputs,
                  arms: dict[str, ArmRun], ambient: dict) -> dict:
    arms_out: dict[str, Any] = {}
    for name, run in arms.items():
        arms_out[name] = {
            "kind": run.spec.kind,
            "params": run.spec.params,
            "cost_override": run.spec.cost,
            "declared_cost": run.declared_cost,
            "series": {
                "dates": run.dates,
                "equity": run.equity,
                "equity_raw": run.equity_raw,
                "invested": run.invested,
                "cash": run.cash,
                "nav": run.nav,
                "interest_credited": run.interest,
            },
            "metrics": run.metrics,
            "trades": {"count": len(run.trades), "legs": run.trades},
            "engine_summary": run.engine_summary,
        }
    deltas: dict[str, Any] = {}
    if "dca" in arms:
        base = arms["dca"].metrics
        for name, run in arms.items():
            if name == "dca":
                continue
            d: dict[str, Any] = {}
            for key, _label, _fmt in _METRIC_ROWS:
                a, b = _metric_value(run.metrics, key), _metric_value(base, key)
                d[key] = (a - b) if (isinstance(a, (int, float)) and isinstance(b, (int, float))) else None
            deltas[name] = d
    return {
        "schema": SCHEMA,
        "generated_at": cfg.generated_at,
        "window": {"start": cfg.start, "end": cfg.end, "sessions": len(inputs.axis),
                   "sessions_axis": inputs.axis},
        "db": inputs.db_info,
        "config": {
            "universe": uni,
            "arms": [asdict(a) for a in cfg.arms],
            "costs": asdict(cfg.costs),
            "seeds": cfg.seeds,
            "configs_tried": cfg.configs_tried,
            "schema_version": cfg.schema_version,
            "contribution": cfg.contribution,
            "benchmark": cfg.benchmark,
            "start_cash": cfg.start_cash,
            "ambient_settings": ambient,
            "note": "all arms share one universe, one session axis and one "
                    "contribution schedule; costs are the declared CostConfig "
                    "unless a per-arm override is recorded",
        },
        "arms": arms_out,
        "deltas_vs_dca": deltas,
        "caveats": _CAVEATS,
    }


_CAVEATS = [
    "Universes are current-membership snapshots: absolute comparisons vs the "
    "broad-market DCA control carry survivorship/selection bias; relative "
    "comparisons across arms sharing this universe are robust to it.",
    "cash_yield_annual is a post-hoc declared assumption credited on the cash "
    "curve for metrics only; the live paper accounts accrue 0% on cash.",
    "The Sharpe's risk-free proxy is the declared cash yield (excess-return "
    "series); at cash_yield=0 it degenerates to the old implicit rf=0 Sharpe.",
    "The monthly arm is an aggregate equal-weight model marked only at fill "
    "sessions + the window's last close (marks_only); its session series is "
    "the step function of those marks, so intra-month volatility is "
    "understated by construction.",
    "The monthly arm's FX leg is folded into its declared flat rate; the "
    "daily-core arm's FX leg is a post-hoc NAV debit on its trade ledger (not "
    "compounded into mid-window cash).",
    "Numbers and deltas only — no verdict is implied (audit finding #4).",
]


def report_json(report: dict) -> str:
    """Canonical JSON: sorted keys, fixed separators — identical inputs give
    byte-identical output."""
    return json.dumps(_jsonable(report), sort_keys=True, indent=2, allow_nan=False) + "\n"


def render_markdown(report: dict) -> str:
    """Markdown report: declared-assumptions header, caveats, one summary
    table (arms x all metrics) and the deltas-vs-dca table. No winner
    language."""
    cfg = report["config"]
    costs = cfg["costs"]
    uni = cfg["universe"]
    lines: list[str] = []
    lines.append(f"# Reval report ({report['schema']})")
    lines.append("")
    lines.append(f"- generated_at: {report['generated_at']}")
    lines.append(f"- window: {report['window']['start']} .. {report['window']['end']} "
                 f"({report['window']['sessions']} sessions)")
    lines.append(f"- db: {report['db']['url']} (read-only; candles "
                 f"{report['db']['candle_range'][0]}..{report['db']['candle_range'][1]}, "
                 f"{report['db']['candle_rows']} rows)")
    lines.append(f"- universe: {uni['name']} ({uni['count']} members, sha256 "
                 f"{uni['sha256'][:16]}...)")
    lines.append(f"- arms: {', '.join(report['arms'].keys())}")
    lines.append(f"- costs (declared): stock {costs['stock_oneway_bps']:.0f} bps one-way, "
                 f"etf {costs['etf_oneway_bps']:.0f} bps, fx {costs['fx_oneway_bps']:.0f} bps "
                 f"(non-USD listings only), cash yield {costs['cash_yield_annual']:.4f}/yr")
    lines.append(f"- contribution: {cfg['contribution']:.2f}/month on the first session "
                 f"of each month; benchmark: {cfg['benchmark']}; start cash: {cfg['start_cash']:.2f}")
    lines.append(f"- seeds: {cfg['seeds']}; configs_tried: {cfg['configs_tried']}")
    lines.append("")
    lines.append("## Declared assumptions and caveats")
    lines.append("")
    for c in report["caveats"]:
        lines.append(f"- {c}")
    lines.append("")
    arm_names = list(report["arms"].keys())
    lines.append("## Summary (arms x metrics; numbers and deltas only, no verdict)")
    lines.append("")
    header = "| metric | " + " | ".join(arm_names) + " |"
    lines.append(header)
    lines.append("|" + "---|" * (len(arm_names) + 1))
    for key, label, fmt in _METRIC_ROWS:
        cells = []
        for a in arm_names:
            v = _metric_value(report["arms"][a]["metrics"], key)
            cells.append("n/a" if v is None else fmt.format(v))
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    lines.append("")
    if report.get("deltas_vs_dca"):
        lines.append("## Deltas vs the dca arm (same window; each arm under its own declared costs)")
        lines.append("")
        dnames = list(report["deltas_vs_dca"].keys())
        lines.append("| metric | " + " | ".join(dnames) + " |")
        lines.append("|" + "---|" * (len(dnames) + 1))
        for key, label, fmt in _METRIC_ROWS:
            cells = []
            for a in dnames:
                v = report["deltas_vs_dca"][a].get(key)
                cells.append("n/a" if v is None else fmt.format(v))
            lines.append(f"| {label} | " + " | ".join(cells) + " |")
        lines.append("")
    lines.append("Per-arm declared costs and engine summaries are in the JSON "
                 "(`arms.<name>.declared_cost`, `arms.<name>.engine_summary`).")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="app.reval",
        description="Phase 4 slice 4A: predeclared strategy revaluation harness "
                    "(analysis-only, read-only DB, offline)")
    sub = p.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run", help="run the arm comparison and write the report")
    r.add_argument("--db", required=True,
                   help="database URL of the snapshot to read (required, e.g. "
                        "sqlite+aiosqlite:////tmp/opencode/reval/trading.db); "
                        "opened read-only, never the ambient DB implicitly")
    r.add_argument("--start", required=True, help="YYYY-MM-DD inclusive window start")
    r.add_argument("--end", required=True, help="YYYY-MM-DD inclusive window end")
    r.add_argument("--universe", default=None,
                   help="universe name (default: settings.sim_monthly_universe)")
    r.add_argument("--universes-dir", default=None,
                   help="extra universes dir (shadows the repo universes/ dir; "
                        "default: settings.universe_extra_dir)")
    r.add_argument("--arms", default="dca,monthly,daily_core,sim",
                   help="comma-separated arm kinds (default: all four predeclared)")
    r.add_argument("--cash-yield", type=float, default=0.03,
                   help="declared annual cash yield on parked cash (default 0.03)")
    r.add_argument("--stock-bps", type=float, default=10.0,
                   help="stock one-way cost in bps (default 10)")
    r.add_argument("--etf-bps", type=float, default=2.0,
                   help="ETF one-way cost in bps (default 2)")
    r.add_argument("--fx-bps", type=float, default=15.0,
                   help="FX one-way cost in bps on non-USD fills (default 15)")
    r.add_argument("--out", default=".", help="output directory for the report files")
    r.add_argument("--seed", type=int, default=1, help="run seed (recorded in the report)")
    r.add_argument("--configs-tried", type=int, default=1,
                   help="caller-managed count of configurations tried (audit requirement)")
    r.add_argument("--contribution", type=float, default=None,
                   help="monthly contribution in USD (default: settings.sim_monthly_contribution)")
    r.add_argument("--benchmark", default=None,
                   help="benchmark ETF ticker (default: settings.sim_benchmark_ticker)")
    r.add_argument("--start-cash", type=float, default=0.0, help="opening cash (default 0)")
    return p


def _config_from_args(args) -> RevalConfig:
    from .config import settings
    universe = args.universe or settings.sim_monthly_universe
    kind_params = {
        "dca": {"role": "broad-market DCA control"},
        "monthly": {"policy": "qv-mom-v1", "target_n": 10, "hold_band": 20},
        "daily_core": {"policy": "qv-mom-v1 + immediate deployment", "dca": "rank",
                       "boost": 0.0, "target_n": 10, "hold_band": 20,
                       "entry_gate": "none", "exit_cadence": "monthly",
                       "crash_pause": False, "overlays": "off"},
        "sim": {"mode": "CONTRACT", "params": "_live_sim_params"},
        "random_n": {"role": "seeded random-N control (slice 4B)"},
        "equal_weight": {"role": "equal-weight control (slice 4B)"},
    }
    arms = []
    for name in [a.strip() for a in args.arms.split(",") if a.strip()]:
        if name not in ARM_KINDS:
            raise ValueError(f"unknown arm {name!r} (choose from {ARM_KINDS})")
        arms.append(ArmSpec(name, name, dict(kind_params[name])))
    return RevalConfig(
        db_url=args.db, start=args.start, end=args.end, universe=universe, arms=arms,
        costs=CostConfig(stock_oneway_bps=args.stock_bps, etf_oneway_bps=args.etf_bps,
                         fx_oneway_bps=args.fx_bps, cash_yield_annual=args.cash_yield),
        seeds={"run": args.seed}, configs_tried=args.configs_tried,
        contribution=(args.contribution if args.contribution is not None
                      else settings.sim_monthly_contribution),
        benchmark=args.benchmark or settings.sim_benchmark_ticker,
        start_cash=args.start_cash, universes_dir=args.universes_dir,
        generated_at=datetime.now(UTC).isoformat(),
    )


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "run":
        try:
            cfg = _config_from_args(args)
        except ValueError as e:
            print(f"error: {e}", file=sys.stderr)
            return 2
        try:
            report = asyncio.run(run_reval_async(cfg))
        except NotImplementedError as e:
            print(f"error: {e}", file=sys.stderr)
            return 2
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        (out / "reval-report.json").write_text(report_json(report))
        (out / "reval-report.md").write_text(render_markdown(report))
        arms = ", ".join(report["arms"].keys())
        print(f"reval report written to {out}/reval-report.{{json,md}} "
              f"(schema {report['schema']}, arms: {arms}, "
              f"window {report['window']['start']}..{report['window']['end']}, "
              f"{report['window']['sessions']} sessions)")
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
