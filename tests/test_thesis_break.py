"""Tests for the M5 fundamental thesis-break exit (``optimize._thesis_break_fundamentals``
+ the ``thesis_break_exit`` hook on ``optimize._daily_core_backtest``) and its
M5 variant wiring in ``app/sellshadow.py``.

Self-contained synthetic FACTOR tape (preloaded frames — the engine tests below
never touch a DB): 8 USD tickers on the true US session calendar from
2023-06-01 (momentum warmup) with quarterly NetIncomeLoss entries (the 4-quarter
TTM chain is closed-form), instant equity/shares facts, and OCF/capex for p_fcf.

Tape events (all trigger filings are filed 2025-08-08, a Friday):

* T05 (top ROE, held) gets a Q2'25 NI entry of -8e8 -> post-filing TTM NI
  = -8e8 + 3*2e8 = -2e8 < 0 -> ROE <= 0 (the research doc's canonical case);
* T06 (2nd, held) gets a RESTATEMENT of Q1'25 (same (start,end), later filed)
  at -6e8 -> TTM = -6e8 + 3*1.9e8 = -3e7 < 0 -> ROE <= 0 via restatement;
* T07 (3rd, held) gets ONLY a capex restatement making FCF negative (p_fcf
  -> None) -> F/P-FCF change alone must NOT trigger;
* T08 (4th, held) gets an equity restatement at -1.0 -> equity <= 0 hard gate.

With target_n=4/hold_band=4 pinned, the book is exactly
{T05, T06, T07, T08} from the first valid rebuild (2025-05-01). Under
thesis_break_exit=True the three broken names exit at 2025-08-11's open (the
standing order struck at S=2025-08-08); under the default OFF they hold to the
2025-09-02 month-end rebuild (the 2025-08-29 frame re-ranks them out of the
top-4 band). 2025-09-01 is Labor Day, so the rebuild day is 2025-09-02.

Covers the brief's 8 areas: (1) default-OFF bit-identical parity, (2) the hook
fires at F's open / OFF holds to the rebuild, (3) PIT guard (no look-ahead),
(4) restatement triggers, (5) FCF-only non-trigger, (6) hard gates (equity<=0
engine-level; record/NI/shares-missing branches unit-level), (7) M5 in the
mvariants report (in tests/test_sellshadow.py), (8) determinism + no-write +
CLI smoke (scratch DB /tmp/opencode/reval/m5/scratch.db).

All offline; the DB-backed CLI smoke is the only DB user and it is seeded
locally (the engine-hook tests proper run on preloaded frames, no DB at all).
"""

from __future__ import annotations

import sqlite3
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app import optimize, reval
from app.db import Base, Candle

M5_DIR = Path("/tmp/opencode/reval/m5")
M5_DB = M5_DIR / "scratch.db"
M5_URL = f"sqlite+aiosqlite:///{M5_DB}"
MUNI_NAME = "m5-tape"
GENERATED_AT = "2026-10-31T00:00:00+00:00"

HIST_START = "2023-06-01"
TAPE_END = "2025-09-30"
WIN_START = "2025-01-02"
WIN_END = "2025-09-30"
TRIGGER_FILED = "2025-08-08"   # Friday; first session seeing it is S=2025-08-08
TRIGGER_FILL = "2025-08-11"    # Monday; the standing order fills at its open
REBUILD_DAY = "2025-09-02"     # first session after the 2025-08-29 month-end
TICKERS = [f"T{i:02d}" for i in range(1, 9)]
# quarterly NI per ticker (T05..T08 are the top-4 ROE -> the held book)
QNI = {"T01": 1.00e8, "T02": 1.05e8, "T03": 1.10e8, "T04": 1.15e8,
       "T05": 2.00e8, "T06": 1.90e8, "T07": 1.80e8, "T08": 1.70e8}

QUARTERS = [  # (start, end, filed) — the 4-quarter TTM chain, closed-form
    ("2024-04-01", "2024-06-30", "2024-08-15"),
    ("2024-07-01", "2024-09-30", "2024-11-14"),
    ("2024-10-01", "2024-12-31", "2025-01-10"),
    ("2025-01-01", "2025-03-31", "2025-02-28"),
]


def _sessions() -> pd.DatetimeIndex:
    from app import sim as sim_mod
    dates = pd.bdate_range(HIST_START, TAPE_END)
    return pd.DatetimeIndex(
        [d for d in dates if sim_mod.is_trading_day(d.to_pydatetime())])


def make_m5_tape():
    """The M5 factor tape: candles (open != close) + fundamentals in the
    ``load_fundamentals`` read schema. Returns frames + the fund dict."""
    idx = _sessions()
    n = len(idx)
    close = pd.DataFrame(index=idx, columns=TICKERS, dtype=float)
    open_px = pd.DataFrame(index=idx, columns=TICKERS, dtype=float)
    # identical price paths for every ticker: momentum and P/FCF ranks tie, so
    # the qv-mom order is decided by ROE alone and the held book (top-4) is
    # exactly {T05, T06, T07, T08}
    c = 100.0 * np.exp(0.0004 * np.arange(n))
    c = c * (1.0 + 0.005 * np.sin(2 * np.pi * np.arange(n) / 11.0))
    for t in TICKERS:
        close[t] = np.round(c, 4)
        open_px[t] = np.round(c * (1.0 - 0.001 * (np.arange(n) % 4) / 4.0), 4)
    vol = pd.DataFrame(1e6, index=idx, columns=TICKERS)

    fund: dict[str, dict[str, list[dict]]] = {}
    for t in TICKERS:
        ni = [{"start": s, "end": e, "filed": f, "val": QNI[t]}
              for s, e, f in QUARTERS]
        fund[t] = {
            "StockholdersEquity": [
                {"start": None, "end": "2024-12-31", "filed": "2025-01-10",
                 "val": 5e9}],
            "NetIncomeLoss": ni,
            "CommonStockSharesOutstanding": [
                {"start": None, "end": "2024-12-31", "filed": "2025-01-10",
                 "val": 1e8}],
            "NetCashProvidedByUsedInOperatingActivities": [
                {"start": "2024-01-01", "end": "2024-12-31", "filed": "2025-01-10",
                 "val": 1.0e9}],
            "PaymentsToAcquirePropertyPlantAndEquipment": [
                {"start": "2024-01-01", "end": "2024-12-31", "filed": "2025-01-10",
                 "val": -2e8}],
        }
    # the trigger filings (all filed 2025-08-08)
    fund["T05"]["NetIncomeLoss"].append(
        {"start": "2025-04-01", "end": "2025-06-30", "filed": TRIGGER_FILED,
         "val": -8e8})                                     # ROE <= 0
    fund["T06"]["NetIncomeLoss"].append(
        {"start": "2025-01-01", "end": "2025-03-31", "filed": TRIGGER_FILED,
         "val": -6e8})                                     # restatement -> ROE <= 0
    fund["T07"]["PaymentsToAcquirePropertyPlantAndEquipment"].append(
        {"start": "2024-01-01", "end": "2024-12-31", "filed": TRIGGER_FILED,
         "val": -1.2e9})                                   # FCF < 0 only: NO trigger
    fund["T08"]["StockholdersEquity"].append(
        {"start": None, "end": "2024-12-31", "filed": TRIGGER_FILED,
         "val": -1.0})                                     # equity <= 0
    return {"sessions": idx, "close": close, "open": open_px, "vol": vol,
            "fund": fund, "tickers": list(TICKERS)}


@pytest.fixture(scope="module")
def tape():
    return make_m5_tape()


@pytest.fixture(scope="module")
def udir(tmp_path_factory):
    """Universe file for the tape's tickers (resolved via the extra dir)."""
    d = tmp_path_factory.mktemp("m5-universes")
    (d / f"{MUNI_NAME}.txt").write_text("\n".join(TICKERS) + "\n")
    return str(d)


async def _run(tape, udir, *, start=WIN_START, end=WIN_END, **kw):
    """One daily-core run on preloaded frames (no DB), small book pinned."""
    with reval._settings_overrides(sim_monthly_target_n=4,
                                   sim_monthly_hold_band=4):
        with reval.universes_override(udir):
            return await optimize._daily_core_backtest(
                start, end, MUNI_NAME, 1000.0, False, "none", "monthly", False,
                "monthly",
                close_preloaded=(tape["close"], tape["vol"]),
                fund_preloaded=tape["fund"], open_preloaded=tape["open"],
                cost_oneway=0.0, with_baseline=False, **kw)


# ---------------------------------------------------------------------------
# 1. Default-OFF bit-identical parity
# ---------------------------------------------------------------------------

async def test_default_off_bit_identical(tape, udir):
    """Omitted knob vs thesis_break_exit=False: identical summaries incl. the
    full trade ledgers (the no-arg path must stay bit-identical)."""
    a = await _run(tape, udir)
    b = await _run(tape, udir, thesis_break_exit=False)
    assert a is not None and b is not None
    assert asdict(a) == asdict(b)
    assert a.n_thesis_break == 0 and a.thesis_break_exits == []
    # and the settings default is OFF
    from app.config import settings
    assert settings.sim_daily_core_thesis_break is False


# ---------------------------------------------------------------------------
# 2/4/5/6. The hook fires at F's open; restatement; FCF non-trigger; hard gates
# ---------------------------------------------------------------------------

async def test_hook_fires_at_next_open(tape, udir):
    """ON: T05/T06/T08 break at S=2025-08-08 and fill at 2025-08-11's OPEN;
    T07 (FCF-only) does NOT exit. OFF: all three hold to the 2025-09-02
    month-end rebuild."""
    on = await _run(tape, udir, thesis_break_exit=True)
    off = await _run(tape, udir, thesis_break_exit=False)
    assert on.n_thesis_break == 3
    exits = {(e["ticker"], e["date"], e["decision_date"])
             for e in on.thesis_break_exits}
    assert exits == {("T05", TRIGGER_FILL, TRIGGER_FILED),
                     ("T06", TRIGGER_FILL, TRIGGER_FILED),
                     ("T08", TRIGGER_FILL, TRIGGER_FILED)}
    # the fills are SELL legs at 2025-08-11's strict open
    fills = {t["ticker"]: t for t in on.trades
             if t["side"] == "SELL" and t["date"] == TRIGGER_FILL}
    assert set(fills) == {"T05", "T06", "T08"}
    for tk, tr in fills.items():
        assert tr["price"] == pytest.approx(
            float(tape["open"].loc[pd.Timestamp(TRIGGER_FILL), tk]))
    # T07: FCF deterioration alone is NOT a thesis break — no exit at all
    assert "T07" not in {e["ticker"] for e in on.thesis_break_exits}
    assert not any(t["ticker"] == "T07" and t["side"] == "SELL"
                   and t["date"] <= "2025-09-01" for t in on.trades)
    # OFF: no thesis exits; the three names are released at the rebuild instead
    assert off.n_thesis_break == 0 and off.thesis_break_exits == []
    off_sells = {(t["ticker"], t["date"]) for t in off.trades
                 if t["side"] == "SELL"}
    assert {("T05", REBUILD_DAY), ("T06", REBUILD_DAY),
            ("T08", REBUILD_DAY)} <= off_sells
    assert not any(t["ticker"] in {"T05", "T06", "T08"} and t["side"] == "SELL"
                   and t["date"] < REBUILD_DAY for t in off.trades)


# ---------------------------------------------------------------------------
# 3. PIT guard: a filing with filed > S does not trigger at S (no look-ahead)
# ---------------------------------------------------------------------------

async def test_pit_guard_no_lookahead(tape, udir):
    """Window ending 2025-08-07 (before the trigger filing): ON is identical
    to OFF and fires nothing. Including 2025-08-08 alone must not fill either
    (its decision session S=2025-08-07 predates the filing)."""
    pre_end = "2025-08-07"
    on = await _run(tape, udir, end=pre_end, thesis_break_exit=True)
    off = await _run(tape, udir, end=pre_end, thesis_break_exit=False)
    assert on.n_thesis_break == 0 and asdict(on) == asdict(off)
    # 2025-08-08 session: decision S = 2025-08-07 -> filing not yet public
    on2 = await _run(tape, udir, end=TRIGGER_FILED, thesis_break_exit=True)
    assert on2.n_thesis_break == 0
    assert not any(t["side"] == "SELL" and t["date"] <= TRIGGER_FILED
                   for t in on2.trades)


# ---------------------------------------------------------------------------
# 6b. Unit-level branches of the predeclared condition
# ---------------------------------------------------------------------------

def test_thesis_break_condition_branches():
    f = optimize._thesis_break_fundamentals
    asof = pd.Timestamp("2025-08-11")
    healthy = {
        "StockholdersEquity": [{"start": None, "end": "2024-12-31",
                                "filed": "2025-01-10", "val": 5e9}],
        "NetIncomeLoss": [{"start": s, "end": e, "filed": fl, "val": 1e8}
                          for s, e, fl in QUARTERS],
        "CommonStockSharesOutstanding": [{"start": None, "end": "2024-12-31",
                                          "filed": "2025-01-10", "val": 1e8}],
    }
    assert f(healthy, asof) is False                 # healthy name: no break
    assert f(None, asof) is True                      # record missing
    no_sh = {k: v for k, v in healthy.items()
             if k != "CommonStockSharesOutstanding"}
    assert f(no_sh, asof) is True                     # shares missing
    one_q = {**healthy, "NetIncomeLoss": healthy["NetIncomeLoss"][:1]}
    assert f(one_q, asof) is True                     # TTM NI missing (no chain)
    neg_eq = {**healthy, "StockholdersEquity": [
        {"start": None, "end": "2024-12-31", "filed": "2025-01-10", "val": -1.0}]}
    assert f(neg_eq, asof) is True                    # equity <= 0
    neg_ni = {**healthy, "NetIncomeLoss": [
        {**q, "val": -1e8} for q in healthy["NetIncomeLoss"]]}
    assert f(neg_ni, asof) is True                    # ROE = NI/equity <= 0
    # F/P-FCF missing alone (no OCF/capex at all) must NOT trigger
    no_fcf = {k: v for k, v in healthy.items()
              if k not in ("NetCashProvidedByUsedInOperatingActivities",
                           "PaymentsToAcquirePropertyPlantAndEquipment")}
    assert f(no_fcf, asof) is False
    # filing-date PIT inside the condition: a negative restatement filed AFTER
    # asof is invisible
    future = {**healthy, "NetIncomeLoss": healthy["NetIncomeLoss"] + [
        {"start": "2025-01-01", "end": "2025-03-31", "filed": "2025-12-31",
         "val": -9e9}]}
    assert f(future, asof) is False


# ---------------------------------------------------------------------------
# 8a. Determinism (identical rerun)
# ---------------------------------------------------------------------------

async def test_hook_deterministic(tape, udir):
    a = await _run(tape, udir, thesis_break_exit=True)
    b = await _run(tape, udir, thesis_break_exit=True)
    assert asdict(a) == asdict(b)


# ---------------------------------------------------------------------------
# 8b/8c. No-write proof + CLI smoke on the local scratch DB (offline)
# ---------------------------------------------------------------------------

def _snapshot(path: Path):
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        tables = [r[0] for r in con.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")]
        return {t: con.execute(f'SELECT COUNT(*) FROM "{t}"').fetchone()[0]
                for t in tables}
    finally:
        con.close()


async def _seed_m5_db(tmp_path):
    """Seed the tape (candles + fundamentals) into the fixed M5 scratch DB and
    write its universe file; returns the universes dir."""
    from app.db import Fundamental
    M5_DIR.mkdir(parents=True, exist_ok=True)
    if M5_DB.exists():
        M5_DB.unlink()
    tape = make_m5_tape()
    engine = create_async_engine(M5_URL, connect_args={"timeout": 5})
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    candle_rows = []
    for t in TICKERS:
        for d in tape["sessions"]:
            candle_rows.append({
                "ticker": t, "timestamp": d.to_pydatetime(),
                "open": float(tape["open"].loc[d, t]),
                "high": float(max(tape["open"].loc[d, t],
                                  tape["close"].loc[d, t])) * 1.001,
                "low": float(min(tape["open"].loc[d, t],
                                 tape["close"].loc[d, t])) * 0.999,
                "close": float(tape["close"].loc[d, t]), "volume": 1e6})
    fund_rows = [{"ticker": t, "tag": tag, "start": e["start"], "end": e["end"],
                  "filed": e["filed"], "val": e["val"], "currency": "USD",
                  "source": "edgar"}
                 for t, tags in tape["fund"].items()
                 for tag, entries in tags.items() for e in entries]
    async with factory() as s:
        await s.execute(Candle.__table__.insert(), candle_rows)
        await s.execute(Fundamental.__table__.insert(), fund_rows)
        await s.commit()
    await engine.dispose()
    udir = tmp_path / "m5-universes"
    udir.mkdir(parents=True, exist_ok=True)
    (udir / f"{MUNI_NAME}.txt").write_text("\n".join(TICKERS) + "\n")
    return str(udir)


def _daily_core_args(**over):
    """The daily-core subcommand's parsed-args namespace (built directly:
    argparse's %-check on this Python 3.15 rejects the BASE parser's bare-%
    help strings — a pre-existing incompatibility, verified at the base
    commit — so the smoke drives ``optimize._main`` in-process instead)."""
    from types import SimpleNamespace
    args = dict(command="daily-core", start=WIN_START, end=WIN_END,
                universe=MUNI_NAME, contribution=None, verbose=False,
                dca="monthly", entry_gate="none", exit_cadence="monthly",
                crash_pause=False, mom=None, vol_weight=False, target_vol=None,
                lowvol_tilt=False, portfolio_stop=None, exposure_trend=None,
                trailing_stop=None, basket_trend=None, basket_confirm=None,
                basket_threshold=None, basket_drawdown=None, basket_er_min=None,
                basket_good_times=False, basket_arm_sma=None, thesis_break=False)
    args.update(over)
    return SimpleNamespace(**args)


async def test_engine_no_write_proof(tmp_path):
    """No-write proof on the local scratch DB: seed the tape, run the daily-core
    subcommand path (``optimize._main`` with thesis_break=True) against the
    DB-loaded frames, and check every table's row count is unchanged."""
    import io
    from contextlib import redirect_stdout
    udir = await _seed_m5_db(tmp_path)
    before = _snapshot(M5_DB)
    with reval.universes_override(udir):
        async with reval.point_at_database(M5_URL):
            with reval._settings_overrides(sim_monthly_target_n=4,
                                           sim_monthly_hold_band=4):
                buf = io.StringIO()
                with redirect_stdout(buf):
                    await optimize._main(_daily_core_args(thesis_break=True))
    out = buf.getvalue()
    assert "thesis-break=on" in out
    assert "thesis-break exits: 3" in out
    assert _snapshot(M5_DB) == before  # the run never wrote


async def test_cli_smoke_offline(tmp_path):
    """CLI smoke, offline: the daily-core subcommand path with
    ``thesis_break=True`` fires the hook (3 exits at 2025-08-11) and the
    default OFF run carries no thesis-break output — the no-arg path is
    unchanged. (Driven through ``optimize._main``; see _daily_core_args.)"""
    import io
    from contextlib import redirect_stdout
    udir = await _seed_m5_db(tmp_path)
    with reval.universes_override(udir):
        async with reval.point_at_database(M5_URL):
            with reval._settings_overrides(sim_monthly_target_n=4,
                                           sim_monthly_hold_band=4):
                buf = io.StringIO()
                with redirect_stdout(buf):
                    await optimize._main(_daily_core_args(thesis_break=True))
                assert "thesis-break=on" in buf.getvalue()
                assert "thesis-break exits: 3" in buf.getvalue()
                buf = io.StringIO()
                with redirect_stdout(buf):
                    await optimize._main(_daily_core_args())
                assert "thesis-break" not in buf.getvalue()
