"""Tests for the Phase 5A slice 5A-1 sell-shadow harness (``app/sellshadow.py``).

Self-contained synthetic tape (built in-file — ``tests/reval_tape.py`` belongs
to the 4A suite and is NOT touched here): a hand-computable linear-path tape
where the ATR stop level is closed-form (constant true range -> constant
ATR14), so the extractor and the ratchet self-check are pinned to
hand-computed numbers, not to the code under test.

Covers:
1. EXTRACTOR hand-checked on the controlled tape (known ATR path -> known
   level, known context fields, at-level vs at-open fills);
2. RATCHET SELF-CHECK: recomputed standing level == reason-string level on
   every event (match rate 1.0) and the recompute is bit-identical to the
   signal series' own columns;
3. PROMPT byte-identical determinism + parse_response fail-safety (malformed /
   refusal / empty -> CONFIRM; DEFER rearm outside [1.0, 3.0] -> clamped);
4. COUNTERFACTUAL LEDGER hand-computed on tiny cases (gap-through, at-level,
   time cap, window end, missing-bar retry) + ledger arithmetic on the tape;
5. VARIANT PARITY: BASE via run_variant is bit-identical to a direct
   ``optimize._replay`` call with the same params;
6. NO-WRITE PROOF on the scratch DB (row counts unchanged + query_only
   refuses a write);
7. CLI SMOKE offline on the tape (contexts / defer / variants).

All offline: tests read only the seeded scratch DB.
"""

from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app import optimize, reval, sellshadow
from app.db import Base, Candle

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRATCH_DIR = Path("/tmp/opencode/reval/sell5a")
SCRATCH_DB = SCRATCH_DIR / "scratch.db"
SCRATCH_URL = f"sqlite+aiosqlite:///{SCRATCH_DB}"

TAPE_START = "2024-01-01"
TAPE_END = "2025-12-31"
N = 260
BENCH = "URTH"
UNIVERSE_NAME = "sellshadow-tape"
GENERATED_AT = "2026-10-31T00:00:00+00:00"


# ---------------------------------------------------------------------------
# Self-contained synthetic tape (hand-computable ATR path)
# ---------------------------------------------------------------------------

def _candles(closes, opens=None, spread=0.5):
    """OHLCV rows from explicit closes; open defaults to the previous close
    (no gap); high/low = max/min(open, close) +/- spread, so the true range is
    constant ``|open-close| + 2*spread`` and ATR14 is closed-form."""
    rows = []
    for i, c in enumerate(closes):
        o = opens[i] if opens and opens[i] is not None else (closes[i - 1] if i else c)
        rows.append({
            "time": (pd.Timestamp(TAPE_START) + pd.Timedelta(days=i)).strftime("%Y-%m-%d"),
            "open": round(float(o), 4), "high": round(max(o, c) + spread, 4),
            "low": round(min(o, c) - spread, 4), "close": round(float(c), 4),
            "volume": 1e6,
        })
    return rows


def _series(rows):
    """CONTRACT-mode series: signal series + the strict session bars attached,
    exactly like ``optimize._load_series`` does."""
    ser = optimize._signal_series(rows).copy()
    ser["open"] = [r["open"] for r in rows]
    ser["high"] = [r["high"] for r in rows]
    ser["low"] = [r["low"] for r in rows]
    return ser


def make_tape():
    """The tape: AAA rises +1/day for 220 sessions then crashes 8 (at-level
    stop, hand level = close[S-1] - 2*ATR14 = 318 - 4 = 314); GAP rises
    +0.5/day then gaps down 7% at the open (at-open stop, hand level =
    159 - 2*1.1 = 156.8); four mild fillers; URTH benchmark uptrend."""
    aaa_closes = [100.0 + i for i in range(220)] + [100.0 + 219 - 8.0]
    aaa_closes += [aaa_closes[-1] + 0.3 * (i % 3 - 1) for i in range(N - 221)]
    aaa = _candles(aaa_closes[:N], spread=0.5)

    gap_closes = [50.0 + 0.5 * i for i in range(220)] + [None]
    gap_closes[220] = gap_closes[219] * 0.90
    gap_closes += [gap_closes[-1] * 1.001 for _ in range(N - 221)]
    gap_opens = [None] * N
    gap_opens[220] = gap_closes[219] * 0.93  # gap open below the stop level
    gap = _candles(gap_closes[:N], opens=gap_opens, spread=0.3)

    fillers = {f"F{k:02d}": _candles([80.0 + 0.05 * i + k + 0.02 * (i % 5)
                                      for i in range(N)], spread=0.4)
               for k in range(4)}
    bench = _candles([100.0 + 0.1 * i for i in range(N)], spread=0.4)

    series = {"AAA": _series(aaa), "GAP": _series(gap),
              **{t: _series(r) for t, r in fillers.items()}}
    return {"series": series, "bench": _series(bench), "bench_rows": bench,
            "tickers": ["AAA", "GAP"] + list(fillers)}


def _params():
    p = optimize._live_sim_params()
    p.monthly_allowance = sellshadow.CONTRIBUTION
    p.start_cash = sellshadow.START_CASH
    return p


def _replay_tape(tape):
    params = _params()
    res = optimize._replay(tape["series"], params, start=TAPE_START, end=TAPE_END)
    return params, res


@pytest.fixture(scope="module")
def tape():
    return make_tape()


@pytest.fixture(scope="module")
def replayed(tape):
    return _replay_tape(tape)


# ---------------------------------------------------------------------------
# 1+2. Extractor hand-checked + ratchet self-check
# ---------------------------------------------------------------------------

def test_extractor_hand_checked(tape, replayed):
    params, res = replayed
    ctx = sellshadow.extract_contexts(tape["series"], tape["bench"], res, params,
                                      TAPE_START, TAPE_END)
    assert ctx["schema"] == "stopfire-context/1"
    events = {e["ticker"] + "|" + e["fill_date"]: e for e in ctx["events"]}
    # the tape produces exactly three ATR stop fires (verified by construction)
    assert len(ctx["events"]) == 3

    # --- AAA crash day: at-level fill, hand-computed level ---
    # close[218] = 318, true range = 2.0 constant -> ATR14 = 2.0,
    # level(S) = max(close[199..218]) - 2*2.0 = 318 - 4 = 314.0
    # crash day 220: open = 319, low = 311 - 0.5 = 310.5 <= 314 -> fill AT 314.
    ev = events["AAA|2024-08-08"]
    assert ev["decision_date"] == "2024-08-07"
    assert ev["fill_kind"] == "at_level"
    assert ev["stop_level"] == 314.0
    assert ev["fill_price"] == 314.0
    assert ev["atr14"] == pytest.approx(2.0)
    assert ev["current_close"] == 319.0
    assert ev["sessions_held"] == 20  # first buy on day 200 (fill 299.0), fill day 220
    # entry basis is the blended avg cost over the lot's BUY fills — recompute
    # it independently from the raw trade ledger (PaperPortfolio's blend rule),
    # walking trades in order and stopping at the target SELL
    shares = avg = 0.0
    first_fill = None
    for t in res.trades:
        if t["ticker"] == "AAA" and t["side"] == "SELL" and t["date"] == "2024-08-08":
            break
        if t["ticker"] == "AAA" and t["side"] == "BUY":
            if first_fill is None:
                first_fill = t["price"]
            avg = (shares * avg + t["shares"] * t["price"]) / (shares + t["shares"])
            shares += t["shares"]
    assert ev["entry_price"] == pytest.approx(avg)
    assert ev["selfcheck"]["frozen_pct_stop"] == pytest.approx(first_fill * 0.85)
    assert first_fill == 299.0
    assert ev["pnl_pct"] == pytest.approx((319.0 / avg - 1) * 100)
    assert ev["dist_to_stop_atr"] == pytest.approx((319.0 - 314.0) / 2.0)
    assert ev["selfcheck"]["recomputed_atr_candidate"] == pytest.approx(314.0)
    assert ev["selfcheck"]["level_match"] is True
    assert ev["bench_drawdown_pct"] == pytest.approx(0.0)  # URTH never falls
    assert ev["sessions_to_month_end"] == 24  # tape = calendar days; 2024-08-07 -> 2024-08-31

    # --- GAP gap-down day: at-open fill, hand-computed level ---
    # close[218] = 159, tr = 0.5 + 0.6 = 1.1 -> ATR14 = 1.1,
    # level = 159 - 2*1.1 = 156.8; open[220] = close[219]*0.93 = 159.5*0.93 <= 156.8 -> fill at open
    ev = events["GAP|2024-08-08"]
    assert ev["fill_kind"] == "at_open"
    assert ev["stop_level"] == pytest.approx(156.8)
    assert ev["fill_price"] == pytest.approx(159.5 * 0.93, abs=1e-4)
    assert ev["atr14"] == pytest.approx(1.1)
    assert ev["selfcheck"]["level_match"] is True


def test_ratchet_selfcheck_match_rate_100pct(tape, replayed):
    params, res = replayed
    ctx = sellshadow.extract_contexts(tape["series"], tape["bench"], res, params,
                                      TAPE_START, TAPE_END)
    sc = ctx["selfcheck"]
    assert sc["events"] == len(ctx["events"]) == 3
    assert sc["level_matches"] == sc["events"]
    assert sc["match_rate"] == 1.0
    # the recompute uses the engine's exact pandas formula on the raw bars:
    # bit-identical to the signal series' own atr14/atr_stop columns
    assert sc["recompute_vs_series_column_max_abs_diff"] == 0.0
    # every event's governing level equals the level in the raw reason string
    for ev in ctx["events"]:
        assert ev["selfcheck"]["level_match"] is True
        assert abs(ev["selfcheck"]["recomputed_governing_level"] - ev["stop_level"]) <= 0.005


def test_contexts_json_deterministic(tape, replayed):
    params, res = replayed
    c1 = sellshadow.extract_contexts(tape["series"], tape["bench"], res, params,
                                     TAPE_START, TAPE_END)
    c2 = sellshadow.extract_contexts(tape["series"], tape["bench"], res, params,
                                     TAPE_START, TAPE_END)
    assert sellshadow._dumps(c1) == sellshadow._dumps(c2)
    json.loads(sellshadow._dumps(c1))  # allow_nan=False writable


# ---------------------------------------------------------------------------
# 3. Prompt determinism + parser fail-safety
# ---------------------------------------------------------------------------

def test_render_prompt_byte_identical(tape, replayed):
    params, res = replayed
    ctx = sellshadow.extract_contexts(tape["series"], tape["bench"], res, params,
                                      TAPE_START, TAPE_END)
    ev = ctx["events"][0]
    p1 = sellshadow.render_prompt(ev)
    p2 = sellshadow.render_prompt(dict(reversed(list(ev.items()))))  # key order must not matter
    assert p1 == p2
    assert json.dumps(p1) == json.dumps(p2)  # byte-identical serialization
    assert [m["role"] for m in p1] == ["system", "user"]
    assert f"prompt-version: {sellshadow.PROMPT_VERSION}" in p1[1]["content"]
    assert "CONFIRM" in p1[0]["content"] and "DEFER" in p1[0]["content"]
    assert "fail-safe" in p1[0]["content"].lower() or "FAIL-SAFE" in p1[0]["content"]
    # recorded config constants travel with the module, never called
    assert sellshadow.MODEL_ID == "deepseek-v4.1-flash-mistral"
    assert sellshadow.TEMPERATURE == 0.0


def test_classify_sell_reason():
    c = sellshadow.classify_sell_reason
    assert c("ATR stop standing order: low 1.00 <= level 1.10 — filled at the level") == "atr_stop"
    assert c("ATR stop hit: 1.00 <= 1.10") == "atr_stop"
    assert c("Initial stop: 1.00 <= 1.10") == "initial_stop"
    assert c("Trailing stop: 1.00 <= 1.10 (peak 1.20, -15%)") == "trailing_pct"
    assert c("Portfolio stop: equity 80 is -20.0% from peak 100") == "portfolio_stop"
    assert c("SELL (strength 50)") == "signal"  # replay adapter's signal-SELL stamp
    assert c("SELL signal (strength 50)") == "signal"
    assert c("something else") == "other"


def test_parse_response_fail_safety():
    p = sellshadow.parse_response
    # well-formed
    r = p('{"decision": "CONFIRM", "rationale": "stop stands"}')
    assert r == {"decision": "CONFIRM", "rearm_mult": None, "rationale": "stop stands",
                 "parsed": True, "fail_safe": False}
    r = p('{"decision": "DEFER", "rearm_mult": 1.5, "rationale": "rebound likely"}')
    assert r["decision"] == "DEFER" and r["rearm_mult"] == 1.5 and r["parsed"] is True
    # clamped to [1.0, 3.0]
    assert p('{"decision": "DEFER", "rearm_mult": 9.9}')["rearm_mult"] == 3.0
    assert p('{"decision": "DEFER", "rearm_mult": 0.2}')["rearm_mult"] == 1.0
    assert p('{"decision": "DEFER", "rearm_mult": "2.5"}')["rearm_mult"] == 2.5
    # fail-safe -> CONFIRM
    for bad in ("", "   ", "I cannot help with that.", "no json here",
                '{"decision": "DEFER"}', '{"decision": "DEFER", "rearm_mult": "abc"}',
                '{"decision": "MAYBE"}', "[1, 2, 3]", "{broken",
                '{"decision": "DEFER", "rearm_mult": null}'):
        r = p(bad)
        assert r["decision"] == "CONFIRM" and r["fail_safe"] is True and r["parsed"] is False, bad
    # fenced JSON with surrounding prose still parses
    r = p('Sure! ```json\n{"decision": "DEFER", "rearm_mult": 2.0, "rationale": "x"}\n``` done')
    assert r["decision"] == "DEFER" and r["rearm_mult"] == 2.0 and r["fail_safe"] is False


# ---------------------------------------------------------------------------
# 4. Counterfactual ledger, hand-computed
# ---------------------------------------------------------------------------

def _days(n=80):
    return [(pd.Timestamp("2025-01-01") + pd.Timedelta(days=i)).strftime("%Y-%m-%d")
            for i in range(n)]


def test_defer_forward_hand_computed():
    days = _days()
    fill_idx, level = 10, 100.0
    rearmed = level * (1 - sellshadow.REARM_WIDEN)  # 90.0
    assert rearmed == pytest.approx(90.0)

    # (a1) gap-through: open 85 <= 90 -> fill at the open
    bars = {d: {"open": 95.0, "low": 94.0, "close": 95.5} for d in days}
    bars[days[11]] = {"open": 85.0, "low": 84.0, "close": 86.0}
    r = sellshadow.defer_forward(bars, days, fill_idx, level, len(days))
    assert (r["exit_date"], r["exit_price"], r["exit_kind"]) == (days[11], 85.0, "at_open")

    # (a2) at-level breach: open 95 > 90, low 89 <= 90 -> fill AT the level 90
    bars = {d: {"open": 95.0, "low": 94.0, "close": 95.5} for d in days}
    bars[days[13]] = {"open": 95.0, "low": 89.0, "close": 92.0}
    r = sellshadow.defer_forward(bars, days, fill_idx, level, len(days))
    assert (r["exit_date"], r["exit_price"], r["exit_kind"]) == (days[13], 90.0, "at_level")

    # (b) time cap: no breach within TIME_CAP_SESSIONS -> exit at the cap session's open
    bars = {d: {"open": 95.0, "low": 94.0, "close": 95.5} for d in days}
    cap_day = days[fill_idx + sellshadow.TIME_CAP_SESSIONS]
    bars[cap_day] = {"open": 93.0, "low": 92.0, "close": 93.5}
    r = sellshadow.defer_forward(bars, days, fill_idx, level, len(days))
    assert r["exit_kind"] == "time_cap" and r["exit_date"] == cap_day and r["exit_price"] == 93.0

    # (c) window end before the cap -> exit at the last window session
    hi = fill_idx + 5
    r = sellshadow.defer_forward(bars, days, fill_idx, level, hi)
    assert r["exit_kind"] == "window_end" and r["exit_date"] == days[hi - 1]

    # missing bar on a would-be breach day -> no fill, retry next session
    bars = {d: {"open": 95.0, "low": 94.0, "close": 95.5} for d in days}
    del bars[days[12]]
    bars[days[13]] = {"open": 80.0, "low": 79.0, "close": 81.0}
    r = sellshadow.defer_forward(bars, days, fill_idx, level, len(days))
    assert r["exit_date"] == days[13] and r["exit_price"] == 80.0


def test_ledger_arithmetic_hand_computed(tape, replayed):
    params, res = replayed
    ctx = sellshadow.extract_contexts(tape["series"], tape["bench"], res, params,
                                      TAPE_START, TAPE_END)
    led = sellshadow.build_defer_ledger(tape["series"], res, params, ctx,
                                        TAPE_START, TAPE_END)
    assert led["schema"] == "defer-ledger/1"
    assert led["config"]["rearm_widen"] == 0.10
    assert led["config"]["time_cap_sessions"] == 63
    assert led["selfcheck_match_rate"] == 1.0
    assert len(led["events"]) == 3
    # AAA 2024-08-08: mark 319.0, confirm fill 314.0, rate 10bps (USD stock)
    # confirm_fwd = 314/319 - 1 - 2*0.001 ; defer exits at window end (open 311.0)
    # defer_fwd   = 311/319 - 1 - 2*0.001 ; delta = (311 - 314)/319
    ev = next(e for e in led["events"] if e["ticker"] == "AAA" and e["fill_date"] == "2024-08-08")
    assert ev["oneway_rate"] == pytest.approx(0.001)
    assert ev["confirm_fwd_net"] == pytest.approx(314.0 / 319.0 - 1 - 0.002)
    assert ev["defer_exit_price"] == 311.0 and ev["defer_exit_kind"] == "window_end"
    assert ev["defer_fwd_net"] == pytest.approx(311.0 / 319.0 - 1 - 0.002)
    assert ev["delta_defer_minus_confirm"] == pytest.approx((311.0 - 314.0) / 319.0)
    agg = led["aggregate"]
    assert agg["n"] == 3
    deltas = sorted(e["delta_defer_minus_confirm"] for e in led["events"])
    assert agg["mean"] == pytest.approx(sum(deltas) / 3)
    assert agg["median"] == pytest.approx(deltas[1])
    assert agg["share_positive"] == pytest.approx(sum(1 for d in deltas if d > 0) / 3)
    assert set(led["by_year"]) == {"2024"}
    # deterministic JSON
    assert sellshadow._dumps(led) == sellshadow._dumps(
        sellshadow.build_defer_ledger(tape["series"], res, params, ctx, TAPE_START, TAPE_END))


# ---------------------------------------------------------------------------
# 5. Variant parity: BASE bit-identical to a direct _replay call
# ---------------------------------------------------------------------------

def test_variant_parity_base_bit_identical(tape):
    params = _params()
    costs = reval.CostConfig()
    v = sellshadow.run_variant(tape["series"], "BASE", params, TAPE_START, TAPE_END, costs)
    direct = optimize._replay(tape["series"], params, start=TAPE_START, end=TAPE_END,
                              cost_oneway=lambda t: costs.rate_for(t, "stock"))
    assert v["metrics"]["n_trades"] == direct.n_trades == len(direct.trades)
    assert v["metrics"]["n_sells"] == sum(1 for t in direct.trades if t["side"] == "SELL")
    assert v["final_equity"] == direct.final_equity
    # the variant table's sell-mix sums to the sell count and classifies by reason
    assert sum(v["sell_mix"].values()) == v["metrics"]["n_sells"]
    assert v["sell_mix"].get("atr_stop") == v["metrics"]["n_sells"]  # tape: all ATR stops
    assert v["holding"]["n_closed_lots"] == v["metrics"]["n_sells"]
    d = v["disposition"]
    assert d["RG"] + d["RL"] == v["metrics"]["n_sells"]
    # hand-check the holding median on the tape: lots are 20, 20 and 1 sessions
    assert sorted([20, 20, 1]) == [1, 20, 20]
    assert v["holding"]["median_sessions"] == 20


def test_variants_table_runs_all_five(tape):
    costs = reval.CostConfig()
    out = [sellshadow.run_variant(tape["series"], name, params, TAPE_START, TAPE_END, costs)
           for name, params in sellshadow.variant_params().items()]
    assert [v["name"] for v in out] == ["BASE", "V1_signal_exits", "V2_pct_trail_15",
                                        "V3_initial_atr_only", "V4_portfolio_breaker"]
    assert sellshadow.VARIANTS_CONFIGS_TRIED == 5
    for v in out:
        m = v["metrics"]
        # honest metric pipeline fields are present and finite where expected
        assert m["twr_annualized"] is not None
        assert m["max_drawdown_pct"] is not None
        assert m["vol_annualized"] is not None
        assert m["sharpe_excess"] is not None
        assert m["turnover_annualized"] is not None
        assert m["n_trades"] > 0
    # V1 (no stops at all) must differ from BASE on this tape (BASE stops fired)
    base, v1 = out[0], out[1]
    assert base["sell_mix"].get("atr_stop", 0) > 0
    assert v1["sell_mix"].get("atr_stop", 0) == 0


# ---------------------------------------------------------------------------
# 6+7. No-write proof + CLI smoke on the scratch DB (offline)
# ---------------------------------------------------------------------------

async def _seed_scratch_db(tmp_path):
    """Seed the tape into the fixed scratch DB + write its universe file."""
    SCRATCH_DIR.mkdir(parents=True, exist_ok=True)
    if SCRATCH_DB.exists():
        SCRATCH_DB.unlink()
    tape = make_tape()
    engine = create_async_engine(SCRATCH_URL, connect_args={"timeout": 5})
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    rows = []
    for t, ser in {**tape["series"], BENCH: tape["bench"]}.items():
        for d, o, h, l, c in zip(ser["time"], ser["open"], ser["high"], ser["low"],
                                 ser["close"], strict=True):
            rows.append({"ticker": t, "timestamp": pd.Timestamp(d).to_pydatetime(),
                         "open": float(o), "high": float(h), "low": float(l),
                         "close": float(c), "volume": 1e6})
    async with factory() as s:
        await s.execute(Candle.__table__.insert(), rows)
        await s.commit()
    await engine.dispose()
    udir = tmp_path / "universes"
    udir.mkdir(parents=True, exist_ok=True)
    (udir / f"{UNIVERSE_NAME}.txt").write_text(
        "\n".join(tape["tickers"] + [BENCH]) + "\n")
    return str(udir)


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


async def test_no_write_proof(tmp_path):
    udir = await _seed_scratch_db(tmp_path)
    before = _snapshot(SCRATCH_DB)
    report = await sellshadow.run_contexts(SCRATCH_URL, UNIVERSE_NAME, udir,
                                           TAPE_START, TAPE_END, GENERATED_AT)
    assert report["schema"] == "stopfire-context/1"
    assert report["db"]["read_only"] is True
    assert report["selfcheck"]["match_rate"] == 1.0  # same tape -> 100% again via the DB
    ledger = await sellshadow.run_defer(SCRATCH_URL, UNIVERSE_NAME, udir,
                                         TAPE_START, TAPE_END, GENERATED_AT)
    assert ledger["schema"] == "defer-ledger/1"
    assert _snapshot(SCRATCH_DB) == before
    # the read-only engine refuses a write outright
    from app.db import Session
    with reval.universes_override(udir):
        async with reval.point_at_database(SCRATCH_URL):
            with pytest.raises(Exception, match="readonly|query_only|write"):
                async with Session() as s:
                    await s.execute(text("INSERT INTO watchlist (ticker) VALUES ('ZZZ')"))
                    await s.commit()
    assert _snapshot(SCRATCH_DB) == before


def test_cli_smoke_offline(tmp_path):
    import asyncio
    udir = asyncio.run(_seed_scratch_db(tmp_path))
    before = _snapshot(SCRATCH_DB)
    env = dict(os.environ)
    env["DATABASE_URL"] = "sqlite+aiosqlite:////tmp/opencode/reval/ambient_unused.db"
    env["DATABASE_URL_DEV"] = ""
    for cmd, fname, schema in (
            ("contexts", "stopfire-contexts.json", "stopfire-context/1"),
            ("defer", "defer-ledger.json", "defer-ledger/1"),
            ("variants", "sellshadow-variants.json", "sellshadow-variants/1")):
        out = tmp_path / f"out-{cmd}"
        res = subprocess.run(
            [sys.executable, "-m", "app.sellshadow", cmd,
             "--db", SCRATCH_URL, "--universe", UNIVERSE_NAME, "--universes-dir", udir,
             "--start", TAPE_START, "--end", TAPE_END, "--out", str(out),
             "--generated-at", GENERATED_AT],
            cwd=REPO_ROOT, env=env, capture_output=True, text=True, timeout=600)
        assert res.returncode == 0, res.stderr[-3000:]
        path = out / fname
        assert path.exists()
        doc = json.loads(path.read_text())
        assert doc["schema"] == schema
        assert doc["generated_at"] == GENERATED_AT
        assert doc["db"]["read_only"] is True
        if cmd == "contexts":
            assert doc["selfcheck"]["match_rate"] == 1.0
            assert len(doc["events"]) == 3
        if cmd == "defer":
            assert doc["aggregate"]["n"] == 3
        if cmd == "variants":
            assert doc["config"]["configs_tried"] == 5
            assert [v["name"] for v in doc["variants"]] == [
                "BASE", "V1_signal_exits", "V2_pct_trail_15",
                "V3_initial_atr_only", "V4_portfolio_breaker"]
    assert _snapshot(SCRATCH_DB) == before  # CLI never wrote either
