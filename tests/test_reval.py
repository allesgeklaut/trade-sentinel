"""Tests for the Phase 4 slice 4A revaluation harness (``app/reval.py``).

Covers, on ONE shared synthetic tape (``tests/reval_tape.py`` — the shared
helper for NEW tests only; existing test modules keep their copied tapes):

1. zero-cost/zero-yield WRAPPER PARITY: the ``monthly`` and ``daily_core``
   arms' engine summaries are identical to direct ``optimize._monthly_backtest``
   / ``optimize._daily_core_backtest`` calls with matching params;
2. COST APPLICATION: hand-computed NAV debits (etf bps on the dca arm, stock
   bps through the daily-core engine's ``cost_oneway`` wrapper, +fx bps more on
   non-USD fills via the post-hoc ledger debit);
3. CASH YIELD: a constant cash balance held for k sessions accrues exactly
   ``C * ((1+cy)**(k/252) - 1)``;
4. every metric hand-checked on a fully controlled series;
5. DETERMINISM: two runs with identical inputs -> byte-identical JSON;
6. NO-WRITE PROOF: every table's row count and the table set are unchanged
   after a full run (plus the query_only engine refuses a write outright);
7. the ``sim`` arm trades in CONTRACT mode and ``cost_oneway > 0`` strictly
   reduces final equity vs zero cost on the same tape;
8. CLI smoke: ``python -m app.reval run ...`` completes offline and writes
   JSON + markdown;
9. ``random_n`` / ``equal_weight`` raise the declared NotImplementedError
   (slice 4B) and the CLI reports it gracefully.

All offline: the harness reads only the seeded snapshot DB.
"""

from __future__ import annotations

import io
import json
import math
import os
import sqlite3
import subprocess
import sys
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app import monthly as monthly_mod
from app import optimize, reval
from app.config import settings
from app.db import Base
from app.instruments import is_usd_native

from . import reval_tape  # shared helper for NEW tests only (see its docstring)

REPO_ROOT = Path(__file__).resolve().parent.parent

W_START = "2025-03-03"
W_END = "2026-06-30"
GENERATED_AT = "2026-10-31T00:00:00+00:00"


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

@pytest.fixture
async def tape_env(tmp_path):
    """Seed the shared tape into a temp FILE SQLite DB + write its universe."""
    tape = reval_tape.make_tape(seed=11)
    db_path = tmp_path / "tape.db"
    url = f"sqlite+aiosqlite:///{db_path}"
    engine = create_async_engine(url, connect_args={"timeout": 5})
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    await reval_tape.seed_db(factory, tape)
    udir = tmp_path / "universes"
    reval_tape.universe_file(udir, tape)
    yield SimpleNamespace(url=url, path=db_path, tape=tape, universes_dir=str(udir),
                          universe=reval_tape.UNIVERSE_NAME, bench=reval_tape.BENCH,
                          tickers=reval_tape.TAPE_TICKERS)
    await engine.dispose()


def _cfg(tape_env, arms=None, costs=None, start=W_START, end=W_END, **cost_kw) -> reval.RevalConfig:
    return reval.RevalConfig(
        db_url=tape_env.url, start=start, end=end, universe=tape_env.universe,
        arms=arms if arms is not None else reval.default_arms(),
        costs=costs or reval.CostConfig(**cost_kw),
        seeds={"run": 7}, configs_tried=1, contribution=1000.0,
        benchmark=tape_env.bench, start_cash=0.0,
        universes_dir=tape_env.universes_dir, generated_at=GENERATED_AT,
    )


async def _run_arms(tape_env, cfg: reval.RevalConfig):
    """Library-level run of the configured arms (no report assembly)."""
    with reval.universes_override(tape_env.universes_dir):
        async with reval.point_at_database(tape_env.url):
            uni = reval.resolve_universe(cfg.universe, tape_env.universes_dir)
            inputs = await reval._load_inputs(cfg, uni)
            arms = {}
            for a in cfg.arms:
                arms[a.name] = await reval._run_arm(cfg, a, inputs)
            return inputs, arms


async def _direct_engine_summaries(tape_env, cfg: reval.RevalConfig):
    """Direct engine calls with matching params (the parity baseline)."""
    from app import fundamentals as fundamentals_mod
    with reval.universes_override(tape_env.universes_dir):
        async with reval.point_at_database(tape_env.url):
            members = reval.resolve_universe(cfg.universe, tape_env.universes_dir)["members"]
            open_px, close, vol = await monthly_mod.load_ohlcv_frames(members)
            fund = await fundamentals_mod.load_fundamentals(members)
            with reval._settings_overrides(
                    sim_monthly_cost_oneway=0.0, sim_monthly_target_n=10,
                    sim_monthly_hold_band=20, sim_daily_core_mom_variant="raw",
                    sim_daily_core_lowvol_tilt=False, sim_daily_core_vol_weight=False,
                    sim_daily_core_target_vol=0.0, sim_daily_core_portfolio_stop=0.0,
                    sim_daily_core_exposure_trend=0, sim_daily_core_trailing_stop=0.0,
                    sim_daily_core_basket_trend=0, sim_daily_core_basket_confirm=3,
                    sim_daily_core_basket_threshold=0.0, sim_daily_core_basket_drawdown=0.0,
                    sim_daily_core_basket_er_min=0.0, sim_daily_core_basket_good_times=False,
                    sim_daily_core_basket_arm_sma=200, sim_monthly_rank_boost=0.0):
                with redirect_stdout(io.StringIO()):
                    mb = await optimize._monthly_backtest(
                        cfg.start, cfg.end, cfg.universe, cfg.contribution, False,
                        close_preloaded=(close, vol), fund_preloaded=fund,
                        open_preloaded=open_px)
                    dc = await optimize._daily_core_backtest(
                        cfg.start, cfg.end, cfg.universe, cfg.contribution, False,
                        "none", "monthly", False, "rank",
                        close_preloaded=(close, vol), fund_preloaded=fund,
                        open_preloaded=open_px, cost_oneway=0.0, with_baseline=False)
            return mb, dc, (open_px, close, vol), fund


# ---------------------------------------------------------------------------
# 1. Wrapper parity at zero cost / zero yield
# ---------------------------------------------------------------------------

async def test_wrapper_parity_zero_cost_zero_yield(tape_env):
    """With costs=0 and cash_yield=0 the monthly/daily_core arms must be
    IDENTICAL to direct engine calls with matching params."""
    arms = [reval.ArmSpec("monthly", "monthly",
                          {"policy": "qv-mom-v1", "target_n": 10, "hold_band": 20}),
            reval.ArmSpec("daily_core", "daily_core",
                          {"dca": "rank", "boost": 0.0, "target_n": 10, "hold_band": 20,
                           "entry_gate": "none", "exit_cadence": "monthly",
                           "crash_pause": False, "overlays": "off"})]
    cfg = _cfg(tape_env, arms=arms,
               costs=reval.CostConfig(stock_oneway_bps=0.0, etf_oneway_bps=0.0,
                                      fx_oneway_bps=0.0, cash_yield_annual=0.0))
    _, out = await _run_arms(tape_env, cfg)
    mb, dc, _frames, _fund = await _direct_engine_summaries(tape_env, cfg)

    m = out["monthly"].engine_summary
    assert m["final_value"] == mb.final_value
    assert m["contributed"] == mb.contributed
    assert m["avg_turnover"] == mb.avg_turnover
    assert m["irr"] == mb.irr
    # the picks replay used for the series assembly reproduces the engine's turn
    assert m["replayed_avg_turnover"] == pytest.approx(mb.avg_turnover, rel=1e-12)
    # the assembled series ends exactly where the engine ends
    assert out["monthly"].equity_raw[-1] == mb.final_value
    assert out["monthly"].invested[-1] == mb.contributed
    # zero cash yield -> adjusted == raw equity
    assert out["monthly"].equity == out["monthly"].equity_raw

    d = out["daily_core"].engine_summary
    assert d["final_value"] == dc.final_value
    assert d["contributed"] == dc.contributed
    assert d["avg_turnover"] == dc.avg_turnover
    assert d["n_trades"] == dc.n_trades
    # the per-leg trade ledger is identical to the direct call
    arm_legs = [(t["date"], t["side"], t["ticker"], t["notional"], t["shares"], t["price"])
                for t in out["daily_core"].trades]
    dc_legs = [(t["date"], t["side"], t["ticker"], t["notional"], t["shares"], t["price"])
               for t in dc.trades]
    assert arm_legs == dc_legs
    # the reconstructed daily series reproduces the engine's final mark
    assert d["reconstructed_final_equity"] == pytest.approx(dc.final_value, abs=1e-6)
    assert out["daily_core"].equity_raw[-1] == pytest.approx(dc.final_value, abs=1e-6)


# ---------------------------------------------------------------------------
# 2. Cost application (hand-computed)
# ---------------------------------------------------------------------------

def test_cost_rate_per_ticker():
    costs = reval.CostConfig(stock_oneway_bps=10.0, etf_oneway_bps=2.0, fx_oneway_bps=15.0)
    assert costs.rate_for("T01", "stock") == pytest.approx(0.0010)
    assert costs.rate_for("T01", "etf") == pytest.approx(0.0002)
    # non-USD listings pay stock + fx
    assert costs.rate_for("X01.L", "stock") == pytest.approx(0.0025)
    assert costs.rate_for("X02.DE", "stock") == pytest.approx(0.0025)
    assert not is_usd_native("X01.L") and is_usd_native("URTH")


def test_dca_arm_etf_cost_hand_computed():
    """A $1000 contribution buying at open=100 with etf 2bps must debit
    exactly 2bps of the fill value from the NAV (hand-computed)."""
    axis = ["2026-01-01", "2026-01-02", "2026-01-03"]
    bench = {"2026-01-01": {"open": 100.0, "close": 101.0},
             "2026-01-02": {"open": 100.5, "close": 100.25},
             "2026-01-03": {"open": 100.25, "close": 100.5}}
    inputs = reval._Inputs(
        universe={"name": "u", "members": [], "count": 0, "sha256": "x", "path": "p"},
        members=[], open_px=pd.DataFrame(), close=pd.DataFrame(), vol=pd.DataFrame(),
        raw_close=pd.DataFrame(), fund={}, bench=bench, axis=axis,
        contrib_days={pd.Timestamp("2026-01-01")},
        contrib_by_date={"2026-01-01": 1000.0}, db_info={})
    cfg = reval.RevalConfig(
        db_url="sqlite+aiosqlite:///:memory:", start=axis[0], end=axis[-1],
        universe="u", arms=[], costs=reval.CostConfig(etf_oneway_bps=2.0),
        contribution=1000.0, benchmark="URTH")
    run = reval._run_dca_arm(cfg, reval.ArmSpec("dca", "dca", {}), inputs, cfg.costs)

    rate = 0.0002
    budget = min(1000.0, 1000.0 / (1.0 + rate))     # _fill_buy's fee-reserved cap
    shares = math.floor(budget / 100.0 * 10000) / 10000
    notional = shares * 100.0
    assert run.trades[0]["side"] == "BUY"
    assert run.trades[0]["price"] == 100.0           # strict session OPEN
    assert run.trades[0]["shares"] == shares
    assert run.trades[0]["notional"] == pytest.approx(notional)
    # cash after the fill carries the fee: NAV debit == exactly 2bps of fill
    assert run.cash[0] == pytest.approx(1000.0 - notional * (1.0 + rate), rel=1e-12)
    zero_fee_equity = (1000.0 - notional) + shares * 101.0
    assert zero_fee_equity - run.equity_raw[0] == pytest.approx(notional * rate, rel=1e-12)
    # second session: leftover cash < $1 -> no fill, cash carried
    assert len(run.trades) == 1
    assert run.cash[1] == pytest.approx(run.cash[0])


async def test_daily_core_stock_bps_hand_computed(monkeypatch):
    """Flat canned tape, one deployment month: with cost_oneway=0.001 the
    engine's final value is exactly the zero-cost value minus 10bps of every
    booked leg (hand-computed on a small tape)."""
    tickers = ["A01", "B02", "C03", "D04", "E05", "F06", "G07", "H08"]
    dates = pd.bdate_range("2023-06-01", "2025-02-28")
    close = pd.DataFrame(100.0, index=dates, columns=tickers)
    vol = pd.DataFrame(1e6, index=dates, columns=tickers)

    def canned(order):
        roe = {t: 0.5 - 0.01 * i for i, t in enumerate(order)}
        rows = {t: {"roe": roe.get(t, 0.001), "p_fcf": 20.0, "mcap": 1e10,
                   "dollar_vol": 1e8, "mom": 0.1, "price": 100.0,
                   "vol": 0.2, "eligible": True} for t in tickers}
        return pd.DataFrame(rows).T

    frames = {pd.Timestamp("2024-12-31"): canned(tickers),
              pd.Timestamp("2025-01-31"): canned(tickers)}
    monkeypatch.setattr(monthly_mod, "eligible_frame",
                        lambda d, c, v, f: frames.get(pd.Timestamp(d).normalize()))
    monkeypatch.setattr(settings, "sim_monthly_target_n", 3)
    monkeypatch.setattr(settings, "sim_monthly_hold_band", 6)
    monkeypatch.setattr(settings, "sim_monthly_rank_boost", 0.0)

    async def run(cost):
        with redirect_stdout(io.StringIO()):
            return await optimize._daily_core_backtest(
                "2025-01-02", "2025-02-28", "diversified-plus", 1000.0, False,
                "none", "monthly", False, "rank",
                close_preloaded=(close, vol), fund_preloaded={},
                open_preloaded=close, cost_oneway=cost, with_baseline=False)

    def reconstruct(summary, cost):
        """Hand-rebuild the account from the booked legs: contributions on the
        first session of each window month, then each leg debits/credits
        notional * (1 +/- cost); flat marks at 100."""
        cash = 0.0
        shares: dict[str, float] = {}
        contrib = {pd.Timestamp("2025-01-02"), pd.Timestamp("2025-02-03")}
        by_date: dict[str, list[dict]] = {}
        for t in summary.trades:
            by_date.setdefault(t["date"], []).append(t)
        for d in pd.bdate_range("2025-01-02", "2025-02-28"):
            ds = d.strftime("%Y-%m-%d")
            if d in contrib:
                cash += 1000.0
            for t in by_date.get(ds, []):
                if t["side"] == "BUY":
                    cash -= t["notional"] * (1.0 + cost)
                    shares[t["ticker"]] = shares.get(t["ticker"], 0.0) + t["shares"]
                else:
                    cash += t["notional"] * (1.0 - cost)
                    shares[t["ticker"]] = shares.get(t["ticker"], 0.0) - t["shares"]
        return cash + sum(shares.values()) * 100.0

    zero = await run(0.0)
    costly = await run(0.001)
    assert zero is not None and costly is not None
    assert zero.trades, "the canned tape must produce fills"
    # first-day deployment: 3 legs of one third of the $1000 contribution
    assert zero.trades[0]["price"] == 100.0
    assert zero.trades[0]["notional"] == pytest.approx(1000.0 / 3, abs=0.01)
    # the engine's final value is exactly the ledger reconstruction: every leg
    # debits precisely one-way ``cost`` of its notional on top of the notional
    assert zero.final_value == pytest.approx(reconstruct(zero, 0.0), abs=1e-6)
    assert costly.final_value == pytest.approx(reconstruct(costly, 0.001), abs=1e-6)
    # and the cost strictly reduces the final value
    assert costly.final_value < zero.final_value
    # hand value: flat prices, two $1000 contributions -> zero-cost final 2000
    assert zero.final_value == pytest.approx(2000.0, abs=0.05)


async def test_arm_fx_post_hoc_exact_on_non_usd_legs(tape_env):
    """On the tape, the daily_core arm's FX leg is an exact post-hoc debit:
    (yield-adjusted equity - raw equity) at the end == -fx_rate * sum of
    non-USD fill notionals, and the FX leg does not change the trade set."""
    arm = reval.ArmSpec("daily_core", "daily_core",
                        {"dca": "rank", "boost": 0.0, "target_n": 10, "hold_band": 20,
                         "entry_gate": "none", "exit_cadence": "monthly",
                         "crash_pause": False, "overlays": "off"})
    cfg10 = _cfg(tape_env, arms=[arm], costs=reval.CostConfig(
        stock_oneway_bps=10.0, etf_oneway_bps=2.0, fx_oneway_bps=0.0,
        cash_yield_annual=0.0))
    cfg10fx = _cfg(tape_env, arms=[arm], costs=reval.CostConfig(
        stock_oneway_bps=10.0, etf_oneway_bps=2.0, fx_oneway_bps=15.0,
        cash_yield_annual=0.0))
    _, a10 = await _run_arms(tape_env, cfg10)
    _, a10fx = await _run_arms(tape_env, cfg10fx)
    r10, r10fx = a10["daily_core"], a10fx["daily_core"]
    # the post-hoc FX debit never changes the engine's trade set
    assert r10fx.trades == r10.trades
    nonusd = [t for t in r10fx.trades if not is_usd_native(t["ticker"])]
    assert nonusd, "the tape must produce fills in non-USD listings (X01.L/X02.DE)"
    nonusd_notional = sum(float(t["notional"]) for t in nonusd)
    assert (r10fx.equity[-1] - r10fx.equity_raw[-1]) == pytest.approx(
        -0.0015 * nonusd_notional, rel=1e-9)
    # stock bps strictly reduce final equity vs zero cost
    cfg0 = _cfg(tape_env, arms=[arm], costs=reval.CostConfig(
        stock_oneway_bps=0.0, etf_oneway_bps=2.0, fx_oneway_bps=0.0,
        cash_yield_annual=0.0))
    _, a0 = await _run_arms(tape_env, cfg0)
    assert r10.equity_raw[-1] < a0["daily_core"].equity_raw[-1]


# ---------------------------------------------------------------------------
# 3. Cash yield compounding
# ---------------------------------------------------------------------------

def test_cash_yield_compounding_hand_computed():
    """A constant cash balance C held for k sessions (k accruals) must credit
    exactly C * ((1+cy)**(k/252) - 1)."""
    k = 10
    c = 500.0
    cash = [c] * k          # k sessions of constant cash -> k accrual periods
    equity = [1000.0] * k
    adj, interest = reval.apply_cash_yield(cash, equity, 0.05)
    expected = c * ((1.05) ** (k / 252.0) - 1.0)
    assert interest[-1] == pytest.approx(expected, abs=1e-9)
    assert adj[-1] == pytest.approx(equity[-1] + expected, abs=1e-9)
    # zero yield is a no-op
    adj0, int0 = reval.apply_cash_yield(cash, equity, 0.0)
    assert adj0 == equity and int0 == [0.0] * len(cash)


# ---------------------------------------------------------------------------
# 4. Metrics hand-checked on a controlled series
# ---------------------------------------------------------------------------

def test_metrics_hand_checked():
    dates = ["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04"]
    equity = [100.0, 110.0, 99.0, 108.9]
    invested = [100.0, 100.0, 100.0, 100.0]
    cash = [10.0, 20.0, 0.0, 5.0]
    from app.performance import unitized_nav
    nav = unitized_nav(equity, invested)
    assert nav == pytest.approx([1.0, 1.1, 0.99, 1.089])

    twr = reval.twr_metrics(dates, nav)
    assert twr["twr_cumulative"] == pytest.approx(0.089)
    assert twr["window_days"] == 3
    assert twr["twr_annualized"] == pytest.approx(1.089 ** (365.0 / 3.0) - 1.0)

    x = reval.xirr_metric(dates, invested, equity[-1])
    # single opening flow: xirr == (final/initial)**(365/days) - 1
    assert x == pytest.approx((108.9 / 100.0) ** (365.0 / 3.0) - 1.0, rel=1e-9)

    dd = reval.drawdown_metrics(nav)
    assert dd["max_drawdown_pct"] == pytest.approx(10.0)  # (1.1-0.99)/1.1
    assert dd["underwater"] == {"sessions": 2, "ongoing": True}  # 1.089 < 1.1

    rets = [0.1, -0.1, 0.1]
    mean = sum(rets) / 3
    var = sum((r - mean) ** 2 for r in rets) / 2
    sd = math.sqrt(var)
    assert reval.annualized_vol(nav) == pytest.approx(sd * math.sqrt(252.0))
    assert reval.excess_sharpe(nav, 0.0) == pytest.approx(mean / sd * math.sqrt(252.0))
    rf = 1.05 ** (1.0 / 252.0) - 1.0
    ex = [r - rf for r in rets]
    mex = sum(ex) / 3
    sdex = math.sqrt(sum((e - mex) ** 2 for e in ex) / 2)
    assert reval.excess_sharpe(nav, 0.05) == pytest.approx(mex / sdex * math.sqrt(252.0))

    trades = [{"date": "2026-01-01", "side": "BUY", "ticker": "A", "notional": 500.0},
              {"date": "2026-01-03", "side": "SELL", "ticker": "B", "notional": 200.0}]
    to = reval.turnover_metrics(dates, trades, equity, {"2026-01-01": 100.0})
    # one month, equity at month start 100: ((500-100)+200)/2 / 100 = 3.0
    assert to["avg_monthly_oneway"] == pytest.approx(3.0)
    assert to["annualized"] == pytest.approx(36.0)
    assert to["one_way_notional"] == pytest.approx(350.0)
    assert to["buys_notional"] == pytest.approx(500.0)
    assert to["sells_notional"] == pytest.approx(200.0)

    tic = reval.time_in_cash(cash, equity)
    ws = [c / e for c, e in zip(cash, equity, strict=True)]
    assert tic["mean_cash_weight"] == pytest.approx(sum(ws) / len(ws))
    assert tic["pct_sessions_fully_invested"] == pytest.approx(
        sum(1 for w in ws if w <= 0.05) / len(ws))

    conc = reval.concentration_metrics([{"A": 0.5, "B": 0.3}, {"A": 0.9}])
    assert conc["mean_top_weight"] == pytest.approx(0.7)
    assert conc["max_top_weight"] == pytest.approx(0.9)
    assert conc["mean_hhi"] == pytest.approx((0.25 + 0.09 + 0.81) / 2)
    assert conc["max_hhi"] == pytest.approx(0.81)


# ---------------------------------------------------------------------------
# 5. Determinism
# ---------------------------------------------------------------------------

async def test_determinism_byte_identical(tape_env):
    cfg = _cfg(tape_env, costs=reval.CostConfig(
        stock_oneway_bps=10.0, etf_oneway_bps=2.0, fx_oneway_bps=15.0,
        cash_yield_annual=0.03))
    r1 = await reval.run_reval_async(cfg)
    r2 = await reval.run_reval_async(cfg)
    j1 = reval.report_json(r1)
    j2 = reval.report_json(r2)
    assert j1 == j2
    assert json.loads(j1)["schema"] == "reval-report/1"
    assert set(json.loads(j1)["arms"]) == {"dca", "monthly", "daily_core", "sim"}
    # deltas vs dca exist for every other arm
    assert set(json.loads(j1)["deltas_vs_dca"]) == {"monthly", "daily_core", "sim"}


# ---------------------------------------------------------------------------
# 6. No-write proof
# ---------------------------------------------------------------------------

def _snapshot(path: Path):
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        tables = [r[0] for r in con.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")]
        counts = {t: con.execute(f'SELECT COUNT(*) FROM "{t}"').fetchone()[0]
                  for t in tables}
        return tables, counts
    finally:
        con.close()


async def test_harness_never_writes_the_database(tape_env):
    before = _snapshot(tape_env.path)
    cfg = _cfg(tape_env)  # all four arms, non-zero costs and cash yield
    report = await reval.run_reval_async(cfg)
    assert report["db"]["read_only"] is True
    after = _snapshot(tape_env.path)
    assert after == before
    # and the read-only engine refuses a write outright
    from app.db import Session
    with reval.universes_override(tape_env.universes_dir):
        async with reval.point_at_database(tape_env.url):
            with pytest.raises(Exception, match="readonly|query_only|write"):
                async with Session() as s:
                    await s.execute(text("INSERT INTO watchlist (ticker) VALUES ('ZZZ')"))
                    await s.commit()
    assert _snapshot(tape_env.path) == before


# ---------------------------------------------------------------------------
# 7. Sim arm on the tape: CONTRACT mode, trades, cost strictly reduces equity
# ---------------------------------------------------------------------------

async def test_sim_arm_contract_mode_and_cost(tape_env):
    arm = reval.ArmSpec("sim", "sim", {"mode": "CONTRACT", "params": "_live_sim_params"})
    cfg0 = _cfg(tape_env, arms=[arm], costs=reval.CostConfig(
        stock_oneway_bps=0.0, etf_oneway_bps=0.0, fx_oneway_bps=0.0,
        cash_yield_annual=0.0))
    cfgc = _cfg(tape_env, arms=[arm], costs=reval.CostConfig(
        stock_oneway_bps=10.0, etf_oneway_bps=2.0, fx_oneway_bps=15.0,
        cash_yield_annual=0.0))
    _, a0 = await _run_arms(tape_env, cfg0)
    _, ac = await _run_arms(tape_env, cfgc)
    r0, rc = a0["sim"], ac["sim"]
    assert r0.engine_summary["contract_mode"] is True
    assert r0.trades, "the sim arm must produce trades on the tape"
    assert {t["side"] for t in r0.trades} <= {"BUY", "SELL"}
    assert r0.metrics["final_equity"] > 0
    assert r0.metrics["vol_annualized"] is not None
    assert r0.metrics["twr_annualized"] is not None
    # BUY fills are at the strict session open (contract mode; standing-order
    # stop SELLs may fill at the stop level instead, so only BUYs are pinned)
    opens = {(r["time"], t): r["open"]
             for t, rows in tape_env.tape["candles"].items() for r in rows}
    buys = [t for t in r0.trades if t["side"] == "BUY"]
    assert buys
    for t in buys:
        assert t["price"] == pytest.approx(opens[(t["date"], t["ticker"])])
    # cost_oneway > 0 strictly reduces final equity on the same tape
    assert rc.metrics["final_equity"] < r0.metrics["final_equity"]


# ---------------------------------------------------------------------------
# 8. CLI smoke (offline subprocess)
# ---------------------------------------------------------------------------

async def test_cli_smoke_offline(tape_env, tmp_path):
    out = tmp_path / "out"
    env = dict(os.environ)
    env["DATABASE_URL"] = "sqlite+aiosqlite:////tmp/opencode/reval/ambient_unused.db"
    env["DATABASE_URL_DEV"] = ""
    cmd = [sys.executable, "-m", "app.reval", "run",
           "--db", tape_env.url, "--start", W_START, "--end", W_END,
           "--universe", tape_env.universe, "--universes-dir", tape_env.universes_dir,
           "--arms", "dca,monthly,daily_core,sim",
           "--cash-yield", "0.03", "--stock-bps", "10", "--etf-bps", "2",
           "--fx-bps", "15", "--out", str(out), "--seed", "7"]
    res = subprocess.run(cmd, cwd=REPO_ROOT, env=env, capture_output=True,
                         text=True, timeout=900)
    assert res.returncode == 0, res.stderr[-3000:]
    jpath = out / "reval-report.json"
    mpath = out / "reval-report.md"
    assert jpath.exists() and mpath.exists()
    report = json.loads(jpath.read_text())
    assert report["schema"] == "reval-report/1"
    assert report["db"]["url"] == tape_env.url
    assert report["db"]["read_only"] is True
    assert set(report["arms"]) == {"dca", "monthly", "daily_core", "sim"}
    assert report["config"]["configs_tried"] == 1
    assert report["config"]["seeds"] == {"run": 7}
    md = mpath.read_text()
    assert "Summary" in md and "Deltas vs the dca arm" in md
    assert "winner" not in md.lower()


# ---------------------------------------------------------------------------
# 9. Control scaffolds (slice 4B) raise gracefully
# ---------------------------------------------------------------------------

async def test_control_arms_raise_not_implemented(tape_env):
    for kind in ("random_n", "equal_weight"):
        cfg = _cfg(tape_env, arms=[reval.ArmSpec(kind, kind, {})])
        with pytest.raises(NotImplementedError, match="4B"):
            await reval.run_reval_async(cfg)


def test_cli_reports_control_arms_gracefully(tape_env, tmp_path):
    env = dict(os.environ)
    env["DATABASE_URL"] = "sqlite+aiosqlite:////tmp/opencode/reval/ambient_unused.db"
    env["DATABASE_URL_DEV"] = ""
    cmd = [sys.executable, "-m", "app.reval", "run",
           "--db", tape_env.url, "--start", W_START, "--end", W_END,
           "--universe", tape_env.universe, "--universes-dir", tape_env.universes_dir,
           "--arms", "random_n", "--out", str(tmp_path / "o")]
    res = subprocess.run(cmd, cwd=REPO_ROOT, env=env, capture_output=True,
                         text=True, timeout=300)
    assert res.returncode == 2
    assert "4B" in res.stderr and "not implemented" in res.stderr
    assert "Traceback" not in res.stderr
