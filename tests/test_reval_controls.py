"""Tests for Phase 4 slice 4B (``app/reval.py``): the control arms, the study
driver and the uncertainty layer.

Covers, on the shared synthetic tape (``tests/reval_tape.py`` — extended with
a deliberate one-session gap in ``T07``):

1. random_n determinism (same seed -> byte-identical run) + seed sensitivity
   (different seeds -> different picks);
2. random_n degenerate top-ranked protocol parity: it reproduces the monthly
   arm's picks exactly when the monthly arm's hysteresis band has zero width
   beyond target_n (hold_band == target_n);
3. eligibility identity: both controls' picks at each month-end come from the
   monthly arm's OWN eligible set (one shared frame cache);
4. equal_weight hand-computed on a small canned tape (weights, turnover legs
   and costs flow through the same engine + cost model);
5. the controls' metrics/deltas flow through the shared report pipeline;
6. window derivation (yearly/halves partition the session axis exactly) and
   each window report's DCA regime-context numbers;
7. the fixed-capital case: per-arm conventions (dca buys once at the first
   session's open; the factor engines fund once via the additive
   ``fixed_capital`` hook — parity vs direct engine calls; sim uses
   start_cash=lump with zero allowance) + no monthly contributions + the
   hook's default is bit-identical;
8. the predeclared OFAT grid runs exactly the declared variants and
   ``configs_tried`` is computed correctly;
9. the block bootstrap hand-checked (golden values, constant series, seed
   determinism) + the percentile helper;
10. the study driver: inputs loaded ONCE (counter), byte-identical study JSON,
    no-write proof extended to the full study;
11. CLI study smoke offline (small --seeds) with the wall-clock printed;
12. the sim arm's weights stay finite across the tape's one-session gap (the
    addendum regression), for every arm.

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
import time
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app import fundamentals as fundamentals_mod
from app import monthly as monthly_mod
from app import optimize, reval
from app.db import Base

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


def _cfg(tape_env, arms=None, costs=None, start=W_START, end=W_END, **kw) -> reval.RevalConfig:
    return reval.RevalConfig(
        db_url=tape_env.url, start=start, end=end, universe=tape_env.universe,
        arms=arms if arms is not None else reval.default_arms(),
        costs=costs or reval.CostConfig(),
        seeds={"run": 7}, configs_tried=1, contribution=1000.0,
        benchmark=tape_env.bench, start_cash=0.0,
        universes_dir=tape_env.universes_dir, generated_at=GENERATED_AT, **kw)


async def _run_arms(tape_env, cfg: reval.RevalConfig, *, controls=True, frame_caches=None):
    """Library-level run of the configured arms (no report assembly)."""
    with reval.universes_override(tape_env.universes_dir):
        async with reval.point_at_database(tape_env.url):
            uni = reval.resolve_universe(cfg.universe, tape_env.universes_dir)
            inputs = await reval._load_inputs(cfg, uni)
            arms = {}
            for a in cfg.arms:
                arms[a.name] = await reval._run_arm(cfg, a, inputs,
                                                    frame_caches=frame_caches,
                                                    controls=controls)
            return inputs, arms


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


def _study(tape_env, **kw) -> reval.RevalStudy:
    base = dict(db_url=tape_env.url, start=W_START, end=W_END, universe=tape_env.universe,
                windows=("full",), cases=("contribution",), sensitivity=False, n_seeds=2,
                costs=reval.CostConfig(), contribution=1000.0, fixed_capital=5000.0,
                benchmark=tape_env.bench, start_cash=0.0, seed=13,
                bootstrap_B=200, universes_dir=tape_env.universes_dir,
                generated_at=GENERATED_AT)
    base.update(kw)
    return reval.RevalStudy(**base)


# ---------------------------------------------------------------------------
# 1. random_n determinism + seed sensitivity
# ---------------------------------------------------------------------------

async def test_random_n_determinism_and_seed_sensitivity(tape_env):
    def mk(seed):
        return [reval.ArmSpec("random_n", "random_n", {"seed": seed, "target_n": 10})]
    r1 = await reval.run_reval_async(_cfg(tape_env, arms=mk(101)), controls=True)
    r2 = await reval.run_reval_async(_cfg(tape_env, arms=mk(101)), controls=True)
    r3 = await reval.run_reval_async(_cfg(tape_env, arms=mk(202)), controls=True)
    # same seed -> byte-identical report
    assert reval.report_json(r1) == reval.report_json(r2)
    a1, a3 = r1["arms"]["random_n"], r3["arms"]["random_n"]
    # different seeds -> different picks (and metrics differ)
    assert a1["engine_summary"]["picks_by_month"] != a3["engine_summary"]["picks_by_month"]
    assert a1["metrics"] != a3["metrics"]
    # the protocol is declared in the report
    ctl = a1["engine_summary"]["control"]
    assert ctl["seed"] == 101 and ctl["kind"] == "random_n"
    assert "sha256" in ctl["protocol"] and "sample" in ctl["protocol"]
    # every pick set has exactly target_n names (the tape has 24 eligible-ish members)
    for _me, picks in a1["engine_summary"]["picks_by_month"]:
        assert len(picks) == 10
        assert len(set(picks)) == 10


# ---------------------------------------------------------------------------
# 2. Degenerate top-ranked protocol parity vs the zero-width-band monthly arm
# ---------------------------------------------------------------------------

async def test_random_n_degenerate_top_parity_with_zero_band_monthly(tape_env):
    """``pick="top"`` returns the top-ranked eligible names; the monthly arm
    with a zero-width hysteresis band beyond target_n (hold_band == target_n)
    picks exactly the same names every month — the two runs must be identical."""
    fc: dict = {}
    mcfg = _cfg(tape_env, arms=[reval.ArmSpec("monthly", "monthly",
                                              {"policy": "qv-mom-v1", "target_n": 10,
                                               "hold_band": 10})])
    _, m = await _run_arms(tape_env, mcfg, frame_caches={"raw": fc})
    rcfg = _cfg(tape_env, arms=[reval.ArmSpec("random_n", "random_n",
                                              {"seed": 5, "target_n": 10, "pick": "top"})])
    _, r = await _run_arms(tape_env, rcfg, frame_caches={"raw": fc})
    me = m["monthly"].engine_summary
    re_ = r["random_n"].engine_summary
    assert re_["control"]["degenerate_top_ranked"] is True
    assert re_["picks_by_month"] == me["picks_by_month"]
    assert re_["picks_by_fill"] == me["picks_by_fill"]
    for k in ("final_value", "irr", "avg_turnover", "contributed"):
        assert re_[k] == me[k]
    assert r["random_n"].equity_raw == m["monthly"].equity_raw
    assert r["random_n"].trades == m["monthly"].trades


# ---------------------------------------------------------------------------
# 3. Eligibility identity (shared frame cache)
# ---------------------------------------------------------------------------

async def test_controls_share_the_monthly_arms_eligible_sets(tape_env):
    fc: dict = {}
    mcfg = _cfg(tape_env, arms=[reval.ArmSpec("monthly", "monthly",
                                              {"target_n": 10, "hold_band": 20})])
    _, m = await _run_arms(tape_env, mcfg, frame_caches={"raw": fc})
    keys_after_monthly = set(fc.keys())
    assert keys_after_monthly, "the monthly arm must have filled the shared cache"

    ccfg = _cfg(tape_env, arms=[reval.ArmSpec("random_n", "random_n",
                                              {"seed": 9, "target_n": 10}),
                               reval.ArmSpec("equal_weight", "equal_weight", {})])
    _, c = await _run_arms(tape_env, ccfg, frame_caches={"raw": fc})
    # the controls reused the very same frames (no key was rebuilt/added)
    assert set(fc.keys()) == keys_after_monthly
    for name in ("random_n", "equal_weight"):
        picks = c[name].engine_summary["picks_by_month"]
        assert picks
        for me_str, plist in picks:
            frame = fc[pd.Timestamp(me_str)]
            elig = set(frame[frame["eligible"]].index.tolist())
            assert set(plist) <= elig, f"{name} picked outside the eligible set"
    # equal_weight picks the FULL eligible set of each month-end's frame
    for me_str, plist in c["equal_weight"].engine_summary["picks_by_month"]:
        frame = fc[pd.Timestamp(me_str)]
        elig = sorted(frame[frame["eligible"]].index.tolist())
        assert plist == elig


# ---------------------------------------------------------------------------
# 4. equal_weight hand-computed on a small canned tape
# ---------------------------------------------------------------------------

async def test_equal_weight_hand_computed_small_tape(tape_env):
    """Two month-ends with canned frames ({A,B} then {A,C}), flat-ish prices,
    $1000/month: the engine's aggregate value, the picks, the synthetic legs
    and the weights are all hand-computed."""
    dates = pd.bdate_range("2024-01-02", "2025-03-31")
    tickers = ["A", "B", "C"]
    n = len(dates)
    close = pd.DataFrame(index=dates, columns=tickers, dtype=float)
    close["A"] = 100.0
    close["B"] = 100.0 * (1.001 ** np.arange(n))   # drift between rebalances
    close["C"] = 100.0
    vol = pd.DataFrame(1e6, index=dates, columns=tickers)
    open_px = close.copy()                           # open == close -> clean math

    def frame(elig):
        rows = {}
        for i, t in enumerate(tickers):
            rows[t] = {"roe": 0.5 - 0.01 * i, "p_fcf": 20.0, "mcap": 1e10,
                       "dollar_vol": 1e8, "mom": 0.1, "price": 100.0,
                       "vol": 0.2, "stale": False, "eligible": t in elig}
        return pd.DataFrame(rows).T

    me1, me2 = pd.Timestamp("2025-01-31"), pd.Timestamp("2025-02-28")
    fc = {"raw": {me1: frame({"A", "B"}), me2: frame({"A", "C"})}}
    axis = [d.strftime("%Y-%m-%d") for d in dates
            if pd.Timestamp("2025-01-02") <= d <= pd.Timestamp("2025-03-31")]
    inputs = reval._Inputs(
        universe={"name": tape_env.universe, "members": tickers, "count": 3,
                  "sha256": "x", "path": "p"},
        members=tickers, open_px=open_px, close=close, vol=vol,
        raw_close=close.copy(), fund={}, bench={}, axis=axis,
        contrib_days={pd.Timestamp("2025-01-02"), pd.Timestamp("2025-02-03"),
                      pd.Timestamp("2025-03-03")},
        contrib_by_date={"2025-01-02": 1000.0, "2025-02-03": 1000.0,
                         "2025-03-03": 1000.0},
        db_info={})
    cfg = reval.RevalConfig(
        db_url=tape_env.url, start="2025-01-02", end="2025-03-31",
        universe=tape_env.universe,
        arms=[reval.ArmSpec("equal_weight", "equal_weight",
                            {"role": "equal-weight control"})],
        costs=reval.CostConfig(stock_oneway_bps=10.0, etf_oneway_bps=2.0,
                               fx_oneway_bps=0.0, cash_yield_annual=0.0),
        contribution=1000.0, benchmark="URTH",
        universes_dir=tape_env.universes_dir, generated_at=GENERATED_AT)
    with reval.universes_override(tape_env.universes_dir):
        run = await reval._run_arm(cfg, cfg.arms[0], inputs, frame_caches=fc,
                                   controls=True)

    cost = 0.001                      # flat one-way rate (all-USD members, 10 bps)
    fd1, fd2 = pd.Timestamp("2025-02-03"), pd.Timestamp("2025-03-03")

    def op(t, d):
        return float(open_px.at[d, t])

    r_ab = ((op("A", fd2) / op("A", fd1) - 1.0) + (op("B", fd2) / op("B", fd1) - 1.0)) / 2.0
    invested = 2000.0 * (1.0 - cost)            # fd1: Jan+Feb contributions, turn=1
    invested = invested * (1.0 + r_ab) + 1000.0  # fd2: mark, then the Mar contribution
    invested = invested * (1.0 - cost)           # fd2: turn=1 (the B->C swap)
    expected = invested                          # trailing stretch: A and C flat -> 0
    assert run.engine_summary["final_value"] == pytest.approx(expected, rel=1e-12)
    assert run.equity_raw[-1] == pytest.approx(expected, rel=1e-12)
    assert run.engine_summary["contributed"] == 3000.0
    # picks: the FULL eligible set each month-end
    assert run.engine_summary["picks_by_month"] == [["2025-01-31", ["A", "B"]],
                                                    ["2025-02-28", ["A", "C"]]]
    # equal weights 1/2 per holding; 4 synthetic legs (2 BUY, then BUY + SELL)
    assert all(abs(w - 0.5) < 1e-12 for ws in run.weights for w in ws.values())
    assert len(run.trades) == 4
    assert [t["side"] for t in run.trades] == ["BUY", "BUY", "BUY", "SELL"]
    mark2 = expected
    leg1, leg2 = 1998.0 / 2.0, mark2 / 2.0
    assert [t["notional"] for t in run.trades] == pytest.approx(
        [leg1, leg1, leg2, leg2], rel=1e-12)
    # turnover flows through the shared pipeline (fresh contributions excluded)
    to = run.metrics["turnover"]
    assert to["months"] == 3
    assert to["buys_notional"] == pytest.approx(1998.0 + leg2, rel=1e-12)
    assert to["sells_notional"] == pytest.approx(leg2, rel=1e-12)
    # concentration: two equal holdings -> top weight 0.5 wherever invested
    # (pre-fill sessions carry an empty weight dict, which counts as 0.0)
    conc = run.metrics["concentration"]
    filled = sum(1 for w in run.weights if w)
    assert conc["max_top_weight"] == pytest.approx(0.5)
    assert conc["mean_top_weight"] == pytest.approx(0.5 * filled / len(run.weights))
    assert conc["max_hhi"] == pytest.approx(0.5)
    assert conc["mean_hhi"] == pytest.approx(0.5 * filled / len(run.weights))


# ---------------------------------------------------------------------------
# 5. Controls flow through the shared report pipeline
# ---------------------------------------------------------------------------

async def test_controls_flow_through_the_shared_pipeline(tape_env):
    arms = reval.default_arms() + [
        reval.ArmSpec("random_n", "random_n", {"seed": 42, "target_n": 10}),
        reval.ArmSpec("equal_weight", "equal_weight", {}),
    ]
    rep = await reval.run_reval_async(_cfg(tape_env, arms=arms), controls=True)
    assert set(rep["arms"]) == {"dca", "monthly", "daily_core", "sim",
                                "random_n", "equal_weight"}
    assert set(rep["deltas_vs_dca"]) == {"monthly", "daily_core", "sim",
                                        "random_n", "equal_weight"}
    m0 = rep["arms"]["monthly"]["metrics"]
    for name in ("random_n", "equal_weight"):
        # identical metric structure
        assert set(rep["arms"][name]["metrics"]) == set(m0)
        # same declared cost model as the monthly arm (flat rate on swapped notional)
        assert (rep["arms"][name]["declared_cost"]["oneway_bps"]
                == rep["arms"]["monthly"]["declared_cost"]["oneway_bps"])
        assert rep["arms"][name]["declared_cost"]["mode"] == \
            rep["arms"]["monthly"]["declared_cost"]["mode"]
        # deltas vs dca are computed for the controls too
        assert rep["deltas_vs_dca"][name]["twr_annualized"] is not None
        assert rep["deltas_vs_dca"][name]["sharpe_excess"] is not None


# ---------------------------------------------------------------------------
# 6. Window derivation + regime context in every window report
# ---------------------------------------------------------------------------

def test_study_windows_partition_the_axis_exactly():
    axis = [d.strftime("%Y-%m-%d") for d in reval_tape.sessions()]
    axis = [d for d in axis if W_START <= d <= W_END]
    wins = reval.study_windows(axis, ("full", "yearly", "halves"))

    def sessions(w):
        return [d for d in axis if w["start"] <= d <= w["end"]]

    yearly = [w for w in wins if w["name"].startswith("yearly")]
    halves = [w for w in wins if w["name"].startswith("halves")]
    # yearly windows partition the axis exactly (order, no gaps/overlaps)
    assert [d for w in yearly for d in sessions(w)] == axis
    # halves split by session count exactly
    mid = len(axis) // 2
    assert sessions(halves[0]) == axis[:mid]
    assert sessions(halves[1]) == axis[mid:]
    assert [d for w in halves for d in sessions(w)] == axis
    # full covers everything
    full = [w for w in wins if w["name"] == "full"]
    assert sessions(full[0]) == axis


async def test_each_window_report_embeds_the_dca_regime_context(tape_env):
    study = _study(tape_env, windows=("full", "halves"), cases=("contribution",),
                   n_seeds=2)
    rep = await reval.run_study_async(study)
    assert rep["schema"] == "reval-study/1"
    assert len(rep["runs"]) == 3  # full, halves-1, halves-2
    names = [r["window"]["name"] for r in rep["runs"]]
    assert names == ["full", "halves-1", "halves-2"]
    for run in rep["runs"]:
        rc = run["regime_context"]
        # the DCA arm's realized numbers travel as the regime CONTEXT
        assert rc["dca_twr_cumulative"] is not None
        assert rc["dca_twr_annualized"] is not None
        assert rc["dca_vol_annualized"] is not None
        assert rc["sessions"] > 0
        # the embedded per-window report is a full reval-report/1
        assert run["report"]["schema"] == "reval-report/1"
        assert set(run["report"]["arms"]) == set(reval._STUDY_ARM_ORDER)


# ---------------------------------------------------------------------------
# 7. Fixed-capital case
# ---------------------------------------------------------------------------

async def test_fixed_capital_case_per_arm_conventions(tape_env):
    LUMP = 5000.0

    def fcfg(arms):
        return reval.RevalConfig(
            db_url=tape_env.url, start=W_START, end=W_END, universe=tape_env.universe,
            arms=arms, costs=reval.CostConfig(), seeds={"run": 1}, configs_tried=1,
            contribution=0.0, benchmark=tape_env.bench, start_cash=0.0,
            fixed_capital=LUMP, universes_dir=tape_env.universes_dir,
            generated_at=GENERATED_AT)

    arms = [reval.ArmSpec("dca", "dca", {"role": "x"}),
            reval.ArmSpec("monthly", "monthly", {"target_n": 10, "hold_band": 20}),
            reval.ArmSpec("daily_core", "daily_core",
                          {"dca": "rank", "boost": 0.0, "target_n": 10, "hold_band": 20,
                           "entry_gate": "none", "exit_cadence": "monthly",
                           "crash_pause": False}),
            reval.ArmSpec("sim", "sim", {"mode": "CONTRACT", "params": "_live_sim_params"})]
    cfg = fcfg(arms)
    inputs, out = await _run_arms(tape_env, cfg)
    dca, mon, dc, sim = (out[n] for n in ("dca", "monthly", "daily_core", "sim"))

    # dca: the lump buys the benchmark ONCE at the window's first session open
    assert len(dca.trades) == 1
    first = inputs.axis[0]
    assert dca.trades[0]["date"] == first and dca.trades[0]["side"] == "BUY"
    bench_opens = {r["time"]: r["open"] for r in tape_env.tape["candles"][tape_env.bench]}
    assert dca.trades[0]["price"] == pytest.approx(bench_opens[first])
    assert dca.invested[-1] == LUMP and len(dca.trades) == 1  # no further buys

    # factor engines: funded ONCE (no monthly contributions)
    assert mon.engine_summary["contributed"] == LUMP
    assert dc.engine_summary["contributed"] == LUMP
    assert mon.invested[-1] == LUMP and dc.invested[-1] == LUMP
    assert mon.engine_summary["fixed_capital"] == LUMP
    assert dc.engine_summary["fixed_capital"] == LUMP

    # sim: start_cash = lump, zero monthly allowance
    assert sim.engine_summary["params"]["start_cash"] == LUMP
    assert sim.engine_summary["params"]["monthly_allowance"] == 0.0
    assert sim.invested[-1] == LUMP

    # parity vs direct engine calls with the additive fixed_capital hook
    with reval.universes_override(tape_env.universes_dir):
        async with reval.point_at_database(tape_env.url):
            members = reval.resolve_universe(cfg.universe, tape_env.universes_dir)["members"]
            open_px, close, vol = await monthly_mod.load_ohlcv_frames(members)
            fund = await fundamentals_mod.load_fundamentals(members)
            # the arm's declared flat rate: stock bps + fx bps * non-USD share
            from app.instruments import is_usd_native
            nonusd_share = sum(1 for t in members if not is_usd_native(t)) / len(members)
            flat_rate = (10.0 + 15.0 * nonusd_share) / 1e4
            with reval._settings_overrides(
                    sim_monthly_cost_oneway=flat_rate, sim_monthly_target_n=10,
                    sim_monthly_hold_band=20, sim_daily_core_mom_variant="raw",
                    sim_monthly_rank_boost=0.0, sim_daily_core_lowvol_tilt=False,
                    sim_daily_core_vol_weight=False, sim_daily_core_target_vol=0.0,
                    sim_daily_core_portfolio_stop=0.0, sim_daily_core_exposure_trend=0,
                    sim_daily_core_trailing_stop=0.0, sim_daily_core_basket_trend=0,
                    sim_daily_core_basket_confirm=3, sim_daily_core_basket_threshold=0.0,
                    sim_daily_core_basket_drawdown=0.0, sim_daily_core_basket_er_min=0.0,
                    sim_daily_core_basket_good_times=False, sim_daily_core_basket_arm_sma=200):
                with redirect_stdout(io.StringIO()):
                    mb = await optimize._monthly_backtest(
                        W_START, W_END, cfg.universe, 0.0, False,
                        close_preloaded=(close, vol), fund_preloaded=fund,
                        open_preloaded=open_px, fixed_capital=LUMP)
                    dcs = await optimize._daily_core_backtest(
                        W_START, W_END, cfg.universe, 0.0, False, "none", "monthly",
                        False, "rank", close_preloaded=(close, vol), fund_preloaded=fund,
                        open_preloaded=open_px, cost_oneway=0.001, with_baseline=False,
                        fixed_capital=LUMP)
    assert mb is not None and dcs is not None
    assert mon.engine_summary["final_value"] == mb.final_value
    assert mon.engine_summary["contributed"] == mb.contributed == LUMP
    assert dc.engine_summary["final_value"] == dcs.final_value
    assert dc.engine_summary["contributed"] == dcs.contributed == LUMP


async def test_fixed_capital_hook_default_is_bit_identical(tape_env):
    """The additive optimize hook must be behavior-preserving at its default:
    no fixed_capital arg == fixed_capital=0.0 (bit-identical summaries)."""
    with reval.universes_override(tape_env.universes_dir):
        async with reval.point_at_database(tape_env.url):
            members = reval.resolve_universe(tape_env.universe, tape_env.universes_dir)["members"]
            open_px, close, vol = await monthly_mod.load_ohlcv_frames(members)
            fund = await fundamentals_mod.load_fundamentals(members)
            with reval._settings_overrides(
                    sim_monthly_cost_oneway=0.001, sim_monthly_target_n=10,
                    sim_monthly_hold_band=20, sim_daily_core_mom_variant="raw",
                    sim_monthly_rank_boost=0.0, sim_daily_core_lowvol_tilt=False,
                    sim_daily_core_vol_weight=False, sim_daily_core_target_vol=0.0,
                    sim_daily_core_portfolio_stop=0.0, sim_daily_core_exposure_trend=0,
                    sim_daily_core_trailing_stop=0.0, sim_daily_core_basket_trend=0,
                    sim_daily_core_basket_confirm=3, sim_daily_core_basket_threshold=0.0,
                    sim_daily_core_basket_drawdown=0.0, sim_daily_core_basket_er_min=0.0,
                    sim_daily_core_basket_good_times=False, sim_daily_core_basket_arm_sma=200):
                async def go(**extra):
                    with redirect_stdout(io.StringIO()):
                        mb = await optimize._monthly_backtest(
                            W_START, W_END, tape_env.universe, 1000.0, False,
                            close_preloaded=(close, vol), fund_preloaded=fund,
                            open_preloaded=open_px, **extra)
                        dc = await optimize._daily_core_backtest(
                            W_START, W_END, tape_env.universe, 1000.0, False, "none",
                            "monthly", False, "rank", close_preloaded=(close, vol),
                            fund_preloaded=fund, open_preloaded=open_px,
                            cost_oneway=0.001, with_baseline=False, **extra)
                    return mb, dc
                mb1, dc1 = await go()
                mb2, dc2 = await go(fixed_capital=0.0)
    for a, b in ((mb1, mb2), (dc1, dc2)):
        assert a.final_value == b.final_value
        assert a.irr == b.irr
        assert a.contributed == b.contributed
        assert a.avg_turnover == b.avg_turnover
        assert a.n_trades == b.n_trades
    assert dc1.trades == dc2.trades


# ---------------------------------------------------------------------------
# 8. OFAT grid + configs_tried
# ---------------------------------------------------------------------------

async def test_ofat_grid_and_configs_tried(tape_env):
    study = _study(tape_env, sensitivity=True, n_seeds=2)
    rep = await reval.run_study_async(study)
    assert len(rep["runs"]) == 1
    run = rep["runs"][0]
    names = [o["name"] for o in run["ofat"]]
    assert names == ["stock_bps_5", "stock_bps_20", "cash_yield_0.00", "cash_yield_0.05",
                     "mom_residual", "target_n_20", "hold_band_40"]
    for o in run["ofat"]:
        assert set(o["arms"]) == {"monthly", "daily_core"}  # controls: baseline only
        for a in o["arms"].values():
            assert "twr_annualized" in a["metrics"] and "engine_summary" in a
    # configs_tried = 1 baseline + n_seeds random_n + len(ofat) per window x case
    assert run["configs_tried"] == 1 + 2 + 7
    assert rep["config"]["configs_tried"]["total"] == 1 + 2 + 7
    assert rep["config"]["configs_tried"]["per_run"] == [
        {"window": "full", "case": "contribution", "configs_tried": 10}]
    # fx scales proportionally with stock bps in the declared grid
    grid = {g["name"]: g["overrides"] for g in rep["config"]["ofat_grid"]}
    assert grid["stock_bps_5"]["costs"] == {"stock_oneway_bps": 5.0, "fx_oneway_bps": 7.5}
    assert grid["stock_bps_20"]["costs"] == {"stock_oneway_bps": 20.0, "fx_oneway_bps": 30.0}
    assert grid["mom_residual"]["params"] == {"mom_variant": "residual"}


# ---------------------------------------------------------------------------
# 9. Block bootstrap + percentile helper (hand-checked)
# ---------------------------------------------------------------------------

def test_percentile_linear_interpolation():
    assert reval._percentile([1.0], 95) == 1.0
    assert reval._percentile([1.0, 2.0, 3.0, 4.0], 50) == 2.5
    assert reval._percentile([1.0, 2.0, 3.0, 4.0], 5) == pytest.approx(1.15)
    assert reval._percentile([], 50) is None


def test_bootstrap_constant_series_zero_width():
    rets = [0.001] * 60
    ci = reval.block_bootstrap_ci(rets, reval._stat_vol, block=21, B=200, seed=1)
    assert ci["ci_low"] == ci["ci_high"] == 0.0 and ci["point"] == 0.0
    ci2 = reval.block_bootstrap_ci(rets, lambda r: sum(r) / len(r) * 252.0,
                                   block=21, B=200, seed=1)
    assert ci2["ci_low"] == ci2["ci_high"] == pytest.approx(0.252)


def test_bootstrap_golden_and_deterministic():
    rets = [0.01, -0.02, 0.03, -0.01, 0.02, -0.005]
    def stat(r):
        return sum(r) / len(r) * 252.0

    a = reval.block_bootstrap_ci(rets, stat, block=2, B=10, seed=7)
    b = reval.block_bootstrap_ci(rets, stat, block=2, B=10, seed=7)
    assert a == b  # deterministic for a given seed
    # golden values (hand-computed from the seeded resampling)
    assert a["point"] == pytest.approx(1.05)
    assert a["ci_low"] == pytest.approx(-0.5827500000000001)
    assert a["ci_high"] == pytest.approx(1.53825)
    assert a["B_used"] == 10
    ci_vol = reval.block_bootstrap_ci(rets, reval._stat_vol, block=2, B=10, seed=7)
    assert ci_vol["point"] == pytest.approx(0.3029356367283321)
    assert ci_vol["ci_low"] == pytest.approx(0.22953903777823131)
    assert ci_vol["ci_high"] == pytest.approx(0.3532533690314761)
    # degenerate inputs
    assert reval.block_bootstrap_ci([], stat) is None
    assert reval.block_bootstrap_ci(rets, stat, B=0) is None


# ---------------------------------------------------------------------------
# 10. Study driver: shared inputs (loaded once), determinism, no-write
# ---------------------------------------------------------------------------

async def test_study_driver_shared_inputs_determinism_nowrite(tape_env, monkeypatch):
    calls = {"n": 0}
    orig = reval._load_inputs

    async def counting(cfg, uni):
        calls["n"] += 1
        return await orig(cfg, uni)

    monkeypatch.setattr(reval, "_load_inputs", counting)
    study = _study(tape_env, windows=("full",), cases=("contribution", "fixed"),
                   sensitivity=True, n_seeds=2)
    before = _snapshot(tape_env.path)
    rep1 = await reval.run_study_async(study)
    assert calls["n"] == 1, "the study must load the inputs ONCE over the full span"
    rep2 = await reval.run_study_async(study)
    assert reval.study_json(rep1) == reval.study_json(rep2), \
        "identical inputs -> byte-identical study JSON"
    after = _snapshot(tape_env.path)
    assert after == before, "the study must never write the DB"
    assert rep1["schema"] == "reval-study/1"
    assert len(rep1["runs"]) == 2  # full window x 2 cases
    # configs_tried computed over all runs: 2 x (1 baseline + 2 seeds + 7 ofat)
    assert rep1["config"]["configs_tried"]["total"] == 2 * 10
    # uncertainty layer present in every run
    for run in rep1["runs"]:
        unc = run["uncertainty"]
        assert unc["random_n_seed_dispersion"]["twr_annualized"]["n_seeds"] == 2
        for arm in ("dca", "monthly", "daily_core", "sim"):
            assert f"{arm}.twr_annualized" in unc["arm_percentile_in_random_n"]
        for arm in reval._STUDY_ARM_ORDER:
            b = unc["block_bootstrap"][arm]
            for stat in ("annualized_excess_return", "excess_sharpe", "vol_annualized"):
                ci = b[stat]
                assert ci is not None and ci["ci_low"] <= ci["ci_high"]
    # fixed case conventions are recorded
    fixed_run = [r for r in rep1["runs"] if r["case"] == "fixed"][0]
    assert fixed_run["report"]["config"]["fixed_capital"] == 5000.0
    assert fixed_run["report"]["config"]["contribution"] == 0.0
    assert fixed_run["report"]["arms"]["dca"]["metrics"]["final_equity"] is not None
    # the sim series was loaded once per window (cached across cases)
    # (the study shares inputs; only _load_series is per-window — by design)


# ---------------------------------------------------------------------------
# 11. CLI study smoke (offline subprocess, small --seeds)
# ---------------------------------------------------------------------------

async def test_cli_study_smoke_offline(tape_env, tmp_path):
    out = tmp_path / "study_out"
    env = dict(os.environ)
    env["DATABASE_URL"] = "sqlite+aiosqlite:////tmp/opencode/reval/ambient_unused.db"
    env["DATABASE_URL_DEV"] = ""
    cmd = [sys.executable, "-m", "app.reval", "study",
           "--db", tape_env.url, "--start", W_START, "--end", W_END,
           "--universe", tape_env.universe, "--universes-dir", tape_env.universes_dir,
           "--windows", "full,halves", "--cases", "contribution,fixed",
           "--sensitivity", "ofat", "--seeds", "2", "--bootstrap-B", "200",
           "--fixed-capital", "5000", "--out", str(out)]
    t0 = time.perf_counter()
    res = subprocess.run(cmd, cwd=REPO_ROOT, env=env, capture_output=True,
                         text=True, timeout=1800)
    wall = time.perf_counter() - t0
    assert res.returncode == 0, res.stderr[-3000:]
    assert "wall-clock" in res.stdout
    jpath = out / "reval-study.json"
    mpath = out / "reval-study.md"
    assert jpath.exists() and mpath.exists()
    report = json.loads(jpath.read_text())
    assert report["schema"] == "reval-study/1"
    assert report["db"]["read_only"] is True
    assert len(report["runs"]) == 3 * 2  # 3 windows x 2 cases
    # configs_tried = 6 runs x (1 + 2 seeds + 7 ofat)
    assert report["config"]["configs_tried"]["total"] == 6 * 10
    md = mpath.read_text()
    assert "Reval study" in md and "random_n seed dispersion" in md
    assert "winner" not in md.lower()
    print(f"\n>>> STUDY SMOKE (tape, 3 windows x 2 cases, ofat, seeds=2): "
          f"subprocess wall-clock {wall:.1f}s; CLI printed: "
          f"{[l for l in res.stdout.splitlines() if 'wall-clock' in l]}")


# ---------------------------------------------------------------------------
# 12. Sim arm weights finite across the tape's one-session gap (addendum)
# ---------------------------------------------------------------------------

async def test_sim_arm_weights_finite_across_tape_gap(tape_env):
    gap = tape_env.tape["gap"]
    cfg = _cfg(tape_env, arms=reval.default_arms())
    inputs, out = await _run_arms(tape_env, cfg)
    # the trigger really exists: raw_close's union index puts a NaN cell at
    # the gap date for the gapped ticker (every other member has that session)
    assert pd.isna(inputs.raw_close.at[pd.Timestamp(gap["date"]), gap["ticker"]])
    assert gap["date"] in inputs.axis
    for name, run in out.items():
        assert gap["date"] in run.dates or name != "sim"
        for w in run.weights:
            for v in w.values():
                assert math.isfinite(v), f"{name}: non-finite weight at the gap"
        conc = run.metrics["concentration"]
        assert conc["mean_top_weight"] is not None and math.isfinite(conc["mean_top_weight"]), name
        assert conc["mean_hhi"] is not None and math.isfinite(conc["mean_hhi"]), name
    # the sim arm really held the gapped ticker across the gap window
    sim = out["sim"]
    assert any(gap["ticker"] in w for w in sim.weights)


# ---------------------------------------------------------------------------
# 13. windows the engine itself refuses are recorded, never raised
# ---------------------------------------------------------------------------

async def test_study_records_infeasible_windows(tape_env, monkeypatch):
    """The study driver records windows the ENGINE itself refuses (its own
    momentum-warmup condition) as infeasible entries instead of raising —
    the regression behind the first 4C crash: the real-data yearly-2015
    window (the snapshot starts 2015-01-02, so only ~252 sessions exist by
    year-end, under the 253-session warmup) killed the whole 57-minute study
    with ``ValueError``. On the tape (719 sessions, 2024-01-02..2026-10-02)
    the warmup is raised to 600 sessions: yearly-2025 ends with 522 loaded
    sessions (infeasible) while yearly-2026/full end with 651 (feasible) —
    the same shape as the real 2015 edge, in miniature. The guard uses the
    engine's EXACT condition (sessions in the loaded close index up to the
    window's last day), so a window the guard marks feasible is one the
    engine really runs."""
    from app.config import settings
    monkeypatch.setattr(settings, "sim_monthly_min_history_days", 600)
    study = _study(tape_env, windows=("full", "yearly"), cases=("contribution",),
                   sensitivity=False, n_seeds=1)
    rep = await reval.run_study_async(study)
    by_key = {(r["window"]["name"], r["case"]): r for r in rep["runs"]}
    # the refused window: recorded with the reason, no arms fabricated
    bad = by_key[("yearly-2025", "contribution")]
    assert bad["status"] == "infeasible"
    assert "warmup" in bad["reason"] and "600" in bad["reason"]
    assert "report" not in bad and "uncertainty" not in bad
    # the feasible windows ran the real engine (proves the guard is not
    # stricter than the engine — otherwise these would have raised)
    for name in ("full", "yearly-2026"):
        r = by_key[(name, "contribution")]
        assert r.get("status") is None and "report" in r
    # configs_tried excludes the infeasible window entirely
    assert rep["config"]["configs_tried"]["per_run"] == [
        {"window": "full", "case": "contribution", "configs_tried": 2},
        {"window": "yearly-2026", "case": "contribution", "configs_tried": 2}]
    assert rep["config"]["configs_tried"]["total"] == 4
    assert rep["config"]["infeasible_windows"] == [
        {"window": "yearly-2025", "case": "contribution", "reason": bad["reason"]}]
    # the markdown renders the refusal explicitly, and the study stays
    # byte-identical for identical inputs
    md = reval.render_study_markdown(rep)
    assert "INFEASIBLE" in md and bad["reason"] in md
    rep2 = await reval.run_study_async(study)
    assert reval.study_json(rep) == reval.study_json(rep2)
