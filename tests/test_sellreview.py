"""Tests for the Phase 5C slice 5C-A/B (``app/sellreview.py``): the unified
reviewer's prospective consult runner + the type-B band-drop adapter.

Self-contained canned factor tape (built in-file — ``tests/test_sellshadow.py``
is owned by a parallel worker and is NOT imported): 10 USD tickers with
candles + ``{start,end,filed,val}`` fundamentals, seeded into the fixed scratch
DB /tmp/opencode/reval/sell5c/scratch.db. T06 (top ROE) sinks from 2025-11-17
and carries a negative filing dated 2025-12-15, so at the 2025-12-31 month-end
its mcap is below the eligibility floor and it drops out of the band -> one
band-drop release at the 2026-01-02 rebuild (settings pinned to a small book:
target_n=4, hold_band=6). T05 sinks from 2026-02-03 for a likely second drop.

Covers:
1. consult with MOCK transports: valid DEFER (A) parses; malformed / refusal /
   empty -> fail-safe CONFIRM (parsed=False); a raised transport exception ->
   CONFIRM + error class; a timeout -> CONFIRM. The record is deterministic
   given fixed inputs (modulo latency_ms) and NEVER contains the key;
   log_consult appends deterministic JSONL. Type-B consult: parameter-free DEFER.
2. ZERO-network proof: consult(..., transport=mock) never constructs httpx
   state (httpx.AsyncClient monkeypatched to raise).
3. banddrop_contexts on the canned tape: events found, byte-identical rerun,
   self-check match rate 1.0, factor fields present/finite; a no-filings name ->
   days_since_filing None (declared).
4. banddrop_ledger hand-check: the pure per-event math pinned to constants
   (cost convention + the no-additions cash-yield branch), plus a recomputation
   of a real tape event from the primary ledger/frame data.
5. The B prompt renders byte-identical; parse_response_b accepts a
   parameter-free DEFER while sellshadow.parse_response fail-safes it (the A/B
   parser split is real).
6. NO-WRITE proof on the scratch DB (row counts unchanged) + CLI smoke offline
   for both subcommands.

All offline: tests read only the seeded scratch DB; no test or CLI path invokes
the default (real LLM) transport.
"""

from __future__ import annotations

import asyncio
import json
import os
import sqlite3
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app import reval, sellreview, sellshadow
from app.db import Base, Candle

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRATCH_DIR = Path("/tmp/opencode/reval/sell5c")
SCRATCH_DB = SCRATCH_DIR / "scratch.db"
SCRATCH_URL = f"sqlite+aiosqlite:///{SCRATCH_DB}"

BENCH = "URTH"
UNIVERSE_NAME = "sellreview-tape"
GENERATED_AT = "2026-10-31T00:00:00+00:00"

TAPE_START = "2024-01-02"   # candles seeded a year before the window so the
TAPE_END = "2026-06-30"     # month-end frames have >= 253 history days
W_START = "2025-01-02"
W_END = "2026-06-30"
TICKERS = [f"T{i:02d}" for i in range(1, 11)]


# ---------------------------------------------------------------------------
# Self-contained canned factor tape
# ---------------------------------------------------------------------------

def _sessions() -> pd.DatetimeIndex:
    from app import sim as sim_mod
    dates = pd.bdate_range(TAPE_START, TAPE_END)
    return pd.DatetimeIndex(
        [d for d in dates if sim_mod.is_trading_day(d.to_pydatetime())])


def make_tape():
    """10 tickers, piecewise paths, open != close; T06 sinks 0.965/day from
    2025-11-17 (mcap below the 5e9 floor by the 2025-12-31 month-end) and gets
    a negative NI filing filed 2025-12-15; T05 sinks 0.97/day from 2026-02-03.
    Returns row dicts in the ``market.candles()`` shape plus fundamentals in
    the ``load_fundamentals`` read schema."""
    import numpy as np
    idx = _sessions()
    n = len(idx)
    candles: dict[str, list[dict]] = {}
    for i, t in enumerate(TICKERS):
        salt = sum((k + 1) * ord(c) for k, c in enumerate(t)) % 97
        base = 100.0 + 5.0 * i
        drift = 0.0004 + 0.0001 * (salt % 5)
        close = base * np.exp(drift * np.arange(n))
        if t == "T06":
            mask = idx >= pd.Timestamp("2025-11-17")
            close = np.where(mask, base * 0.965 ** np.cumsum(mask), close)
        if t == "T05":
            mask = idx >= pd.Timestamp("2026-02-03")
            close = np.where(mask, base * 0.97 ** np.cumsum(mask), close)
        phase = 0.6 * (i % 7)
        close = close * (1.0 + 0.01 * np.sin(2.0 * np.pi * np.arange(n) / 9.0 + phase))
        close = np.maximum(close, 1.0)
        open_px = close * (1.0 - 0.002 * (np.arange(n) % 5) / 5.0)
        candles[t] = [{"time": d.strftime("%Y-%m-%d"),
                       "open": round(float(o), 4),
                       "high": round(float(max(o, c)) * 1.001, 4),
                       "low": round(float(min(o, c)) * 0.999, 4),
                       "close": round(float(c), 4), "volume": 1e6}
                      for d, o, c in zip(idx, open_px, close, strict=True)]
    # benchmark: steady uptrend, open != close
    import numpy as np
    bclose = 100.0 * np.exp(0.0003 * np.arange(n))
    bopen = bclose * 0.999
    candles[BENCH] = [{"time": d.strftime("%Y-%m-%d"), "open": round(float(o), 4),
                       "high": round(float(c) * 1.001, 4), "low": round(float(c) * 0.999, 4),
                       "close": round(float(c), 4), "volume": 1e7}
                      for d, o, c in zip(idx, bopen, bclose, strict=True)]

    fundamentals: dict[str, dict[str, list[dict]]] = {}
    for i, t in enumerate(TICKERS):
        roe = 0.50 if t == "T06" else 0.45 - 0.02 * i
        ni_entries = [{"start": "2024-01-01", "end": "2024-12-31", "filed": "2025-02-14",
                       "val": 5e9 * roe}]
        if t == "T06":
            # full-year 2025 loss filed 2025-12-15: from the 2025-12-31 frame on,
            # T06's TTM NI (and with the sinking price its mcap) knocks it out of
            # eligibility — it stays IN the frame row (negative ROE) so the context
            # carries its factor fields
            ni_entries.append({"start": "2025-01-01", "end": "2025-12-31",
                               "filed": "2025-12-15", "val": -1e9})
        fundamentals[t] = {
            "StockholdersEquity": [
                {"start": None, "end": "2024-12-31", "filed": "2025-02-14", "val": 5e9}],
            "NetIncomeLoss": ni_entries,
            "CommonStockSharesOutstanding": [
                {"start": None, "end": "2024-12-31", "filed": "2025-02-14", "val": 1e8}],
            "NetCashProvidedByUsedInOperatingActivities": [
                {"start": "2024-01-01", "end": "2024-12-31", "filed": "2025-02-14",
                 "val": 1.0e9}],
            "PaymentsToAcquirePropertyPlantAndEquipment": [
                {"start": "2024-01-01", "end": "2024-12-31", "filed": "2025-02-14",
                 "val": -2e8}],
        }
    return {"sessions": idx, "candles": candles, "fundamentals": fundamentals,
            "tickers": list(TICKERS)}


async def _seed_scratch_db(tmp_path):
    """Seed the tape into the fixed scratch DB + write its universe file."""
    from app.db import Fundamental
    SCRATCH_DIR.mkdir(parents=True, exist_ok=True)
    for suffix in ("", "-shm", "-wal"):
        p = SCRATCH_DB.parent / (SCRATCH_DB.name + suffix)
        if p.exists():
            p.unlink()
    tape = make_tape()
    engine = create_async_engine(SCRATCH_URL, connect_args={"timeout": 5})
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    candle_rows = [{"ticker": t, "timestamp": pd.Timestamp(r["time"]).to_pydatetime(),
                    "open": r["open"], "high": r["high"], "low": r["low"],
                    "close": r["close"], "volume": r["volume"]}
                   for t, rows in tape["candles"].items() for r in rows]
    fund_rows = [{"ticker": t, "tag": tag, "start": e["start"], "end": e["end"],
                  "filed": e["filed"], "val": e["val"], "currency": "USD",
                  "source": "edgar"}
                 for t, tags in tape["fundamentals"].items()
                 for tag, entries in tags.items() for e in entries]
    async with factory() as s:
        await s.execute(Candle.__table__.insert(), candle_rows)
        await s.execute(Fundamental.__table__.insert(), fund_rows)
        await s.commit()
    await engine.dispose()
    udir = tmp_path / "universes"
    udir.mkdir(parents=True, exist_ok=True)
    (udir / f"{UNIVERSE_NAME}.txt").write_text("\n".join(tape["tickers"]) + "\n")
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


def _pin_book(mp):
    """Pin the small-book settings so band drops actually occur."""
    from app.config import settings
    mp.setattr(settings, "sim_monthly_target_n", 4)
    mp.setattr(settings, "sim_monthly_hold_band", 6)


@pytest.fixture(scope="module")
async def ctx_run(tmp_path_factory):
    """One banddrop_contexts run on the canned tape (small-book settings)."""
    mp = pytest.MonkeyPatch()
    _pin_book(mp)
    udir = await _seed_scratch_db(tmp_path_factory.mktemp("sell5c"))
    try:
        report = await sellreview.banddrop_contexts(
            SCRATCH_URL, UNIVERSE_NAME, udir, W_START, W_END, generated_at=GENERATED_AT)
    finally:
        mp.undo()
    return report, udir


@pytest.fixture(scope="module")
async def ledger_run(tmp_path_factory):
    """One banddrop_ledger run on the canned tape (small-book settings)."""
    mp = pytest.MonkeyPatch()
    _pin_book(mp)
    udir = await _seed_scratch_db(tmp_path_factory.mktemp("sell5c-ledger"))
    try:
        report = await sellreview.banddrop_ledger(
            SCRATCH_URL, UNIVERSE_NAME, udir, W_START, W_END, generated_at=GENERATED_AT)
    finally:
        mp.undo()
    return report, udir


# ---------------------------------------------------------------------------
# 1+2. consult with mock transports + zero-network proof
# ---------------------------------------------------------------------------

def _mock_transport(text=None, exc=None):
    async def transport(messages, model_id, temperature, timeout_s):
        if exc is not None:
            raise exc
        return text
    return transport


async def test_consult_mock_transports_and_fail_safe():
    msgs = [{"role": "user", "content": "hi"}]
    # valid type-A DEFER with rearm_mult parses
    r = await sellreview.consult("ev1", "stopfire", sellshadow.PROMPT_VERSION, msgs,
                                 transport=_mock_transport(
                                     '{"decision": "DEFER", "rearm_mult": 1.5, '
                                     '"rationale": "widen"}'))
    assert r["schema"] == "sell-consult/1"
    assert r["decision"] == "DEFER" and r["params"] == {"rearm_mult": 1.5}
    assert r["parsed"] is True and r["fail_safe"] is False
    assert r["error"] is None and r["key_present"] is False
    assert r["model_id"] == sellreview.MODEL_ID and r["temperature"] == sellreview.TEMPERATURE
    # valid CONFIRM
    r = await sellreview.consult("ev2", "stopfire", sellshadow.PROMPT_VERSION, msgs,
                                 transport=_mock_transport('{"decision": "CONFIRM"}'))
    assert r["decision"] == "CONFIRM" and r["params"] == {} and r["parsed"] is True
    # malformed JSON / refusal text / empty -> fail-safe CONFIRM, parsed=False
    for bad in ("{not json", "I cannot answer that.", "", "   "):
        r = await sellreview.consult("ev3", "stopfire", sellshadow.PROMPT_VERSION, msgs,
                                     transport=_mock_transport(bad))
        assert r["decision"] == "CONFIRM" and r["parsed"] is False and r["fail_safe"] is True
        assert r["error"] is None  # parse fail-safes carry no transport error
    # raised transport exception -> CONFIRM + error class recorded
    r = await sellreview.consult("ev4", "stopfire", sellshadow.PROMPT_VERSION, msgs,
                                 transport=_mock_transport(exc=RuntimeError("boom-123")))
    assert r["decision"] == "CONFIRM" and r["parsed"] is False and r["fail_safe"] is True
    assert r["error"] == "boom-123" and r["error_class"] == "RuntimeError"
    # timeout -> CONFIRM
    r = await sellreview.consult("ev5", "stopfire", sellshadow.PROMPT_VERSION, msgs,
                                 transport=_mock_transport(exc=TimeoutError()))
    assert r["decision"] == "CONFIRM" and r["fail_safe"] is True
    assert r["error_class"] == "TimeoutError"
    # type-B consult: parameter-free DEFER routes to parse_response_b
    r = await sellreview.consult("ev6", "banddrop", sellreview.PROMPT_VERSION_B, msgs,
                                 transport=_mock_transport('{"decision": "DEFER", "rationale": "hold"}'))
    assert r["decision"] == "DEFER" and r["params"] == {} and r["parsed"] is True
    assert r["prompt_version"] == "sellshadow-b/1"


async def test_consult_record_deterministic_and_key_never_present(monkeypatch, tmp_path):
    monkeypatch.setenv("LITELLM_API_KEY", "super-secret-test-key-XYZ")
    msgs = [{"role": "user", "content": "ctx"}]
    kwargs = dict(event_id="ev", event_type="banddrop",
                  prompt_version=sellreview.PROMPT_VERSION_B, messages=msgs,
                  transport=_mock_transport('{"decision": "CONFIRM", "rationale": "r"}'))
    r1 = await sellreview.consult(**kwargs)
    r2 = await sellreview.consult(**kwargs)
    strip = lambda r: {k: v for k, v in r.items() if k != "latency_ms"}  # noqa: E731
    assert strip(r1) == strip(r2)          # deterministic given fixed inputs
    assert r1["latency_ms"] >= 0 and r2["latency_ms"] >= 0
    blob = json.dumps(r1) + json.dumps(r2)
    assert "super-secret-test-key-XYZ" not in blob   # the key is NEVER in the record
    assert r1["key_present"] is False       # mock transport: no key is read
    # log_consult appends deterministic JSONL (shadow-consult-log/1, sorted keys)
    log = tmp_path / "consult.jsonl"
    sellreview.log_consult(r1, log)
    sellreview.log_consult(r2, log)
    lines = log.read_text().splitlines()
    assert len(lines) == 2
    for ln in lines:
        obj = json.loads(ln)
        assert obj["schema"] == "shadow-consult-log/1"
        assert ln == json.dumps(obj, sort_keys=True, separators=(",", ":"))
    assert json.loads(lines[0])["event_id"] == "ev"


async def test_consult_never_constructs_httpx_state(monkeypatch):
    """ZERO-network proof: with an injected transport, consult never touches
    httpx (the default transport's only network path)."""
    import httpx

    class _ExplodingClient:
        def __init__(self, *a, **kw):
            raise AssertionError("httpx.AsyncClient constructed during a mock consult")

    monkeypatch.setattr(httpx, "AsyncClient", _ExplodingClient)
    monkeypatch.setattr(sellreview, "_default_transport",
                        _mock_transport(exc=AssertionError("default transport invoked")))
    msgs = [{"role": "user", "content": "x"}]
    for et, pv in (("stopfire", sellshadow.PROMPT_VERSION),
                   ("banddrop", sellreview.PROMPT_VERSION_B)):
        r = await sellreview.consult("e", et, pv, msgs,
                                     transport=_mock_transport('{"decision": "CONFIRM"}'))
        assert r["decision"] == "CONFIRM"


# ---------------------------------------------------------------------------
# 3. banddrop_contexts on the canned tape
# ---------------------------------------------------------------------------

def test_contexts_schema_events_selfcheck(ctx_run):
    report, _ = ctx_run
    assert report["schema"] == "banddrop-contexts/1"
    assert report["prompt_version"] == sellreview.PROMPT_VERSION_B
    assert report["db"]["read_only"] is True
    assert report["generated_at"] == GENERATED_AT
    sc = report["selfcheck"]
    assert sc["events"] >= 1, "the engineered T06 band drop must appear"
    assert sc["match_rate"] == 1.0 and sc["checks_passed"] == sc["events"]
    # the engineered drop: T06 out of the band at the 2025-12-31 month-end,
    # sold at the 2026-01-02 rebuild fill
    t06 = [e for e in report["events"] if e["ticker"] == "T06"]
    assert t06, report["events"]
    ev = t06[0]
    assert ev["decision_date"] == "2025-12-31"
    assert ev["fill_date"] == "2026-01-02"
    assert ev["eligible_in_frame"] is False       # mcap below the floor
    assert ev["band_distance_ranks"] is None       # not ranked (ineligible)
    assert ev["selfcheck"] == {"held_at_prior_session": True,
                               "band_excludes_name": True,
                               "fill_price_matches_ledger": True}


def test_contexts_factor_fields_present_and_finite(ctx_run):
    report, _ = ctx_run
    for ev in report["events"]:
        for key in ("fill_price", "entry_date", "entry_price", "sessions_held",
                    "pnl_pct", "shares", "position_weight", "roe", "mom_12_1",
                    "days_since_filing", "bench_drawdown_pct", "bench_vol_ann_20d",
                    "sessions_to_next_rebalance", "next_rebuild_date"):
            assert key in ev, (key, ev)
        assert ev["fill_price"] > 0 and ev["roe"] is not None and pd.notna(ev["roe"])
        assert ev["mom_12_1"] is not None and pd.notna(ev["mom_12_1"])
        assert "p_fcf" in ev and "mom_rank_pct_in_frame" in ev  # may be None when unavailable
        # the stint entry must be recovered from the ledger (regression: entry popped
        # before read -> empty entry_date)
        assert ev["entry_date"] and ev["entry_date"] < ev["fill_date"]
        assert ev["entry_price"] > 0 and ev["sessions_held"] is not None
        assert ev["sessions_held"] >= 1 and ev["pnl_pct"] is not None
        assert ev["giveback_from_peak_pct"] is not None
        assert ev["position_weight"] is not None and 0 < ev["position_weight"] < 1
        assert ev["days_since_filing"] is not None and ev["days_since_filing"] >= 0
        assert ev["sessions_to_next_rebalance"] >= 1
        assert ev["next_rebuild_date"] > ev["fill_date"]
    # a name with NO filings -> days_since_filing None (declared)
    assert sellreview._days_since_filing({}, "2025-12-31") is None
    assert sellreview._days_since_filing(None, "2025-12-31") is None


async def test_contexts_byte_identical_rerun(tmp_path):
    mp = pytest.MonkeyPatch()
    _pin_book(mp)
    udir = await _seed_scratch_db(tmp_path)
    try:
        r1 = await sellreview.banddrop_contexts(SCRATCH_URL, UNIVERSE_NAME, udir,
                                                W_START, W_END, generated_at=GENERATED_AT)
        r2 = await sellreview.banddrop_contexts(SCRATCH_URL, UNIVERSE_NAME, udir,
                                                W_START, W_END, generated_at=GENERATED_AT)
    finally:
        mp.undo()
    assert sellreview._dumps(r1) == sellreview._dumps(r2)
    json.loads(sellreview._dumps(r1))  # allow_nan=False writable


# ---------------------------------------------------------------------------
# 4. banddrop_ledger hand-check
# ---------------------------------------------------------------------------

def test_retention_counterfactual_hand_computed():
    """Pin the delta math exactly: cost convention + the no-additions
    cash-yield branch."""
    rate = 0.001  # 10 bps one-way
    # additions branch: fill 100 -> retention exit 110; addition X bought at
    # 50, exits at 55 (r_add = 0.10)
    cf = sellreview._retention_counterfactual(
        100.0, 110.0, [("X", 50.0, 55.0)], sessions_window=21, rate=rate)
    assert cf["r_hold"] == pytest.approx(0.10)
    assert cf["r_replacement"] == pytest.approx(0.10)
    assert cf["replacement_kind"] == "equal_weight_new_additions"
    assert cf["defer_fwd_net"] == pytest.approx(0.10 - rate)   # cost at next rebalance
    assert cf["confirm_fwd_net"] == pytest.approx(0.10 - rate)  # cost at the release
    assert cf["delta_defer_minus_confirm"] == pytest.approx(0.0)
    # two additions, equal-weight mean: X 50->55 (+0.10), Y 20->18 (-0.10) -> 0.0
    cf = sellreview._retention_counterfactual(
        100.0, 110.0, [("X", 50.0, 55.0), ("Y", 20.0, 18.0)], sessions_window=21, rate=rate)
    assert cf["r_replacement"] == pytest.approx(0.0)
    assert cf["confirm_fwd_net"] == pytest.approx(-rate)
    assert cf["delta_defer_minus_confirm"] == pytest.approx(0.10)
    # no-additions branch: cash yield over the window (3%/yr, 21 sessions)
    cf = sellreview._retention_counterfactual(100.0, 110.0, [], sessions_window=21, rate=rate)
    assert cf["replacement_kind"] == "cash_yield_no_additions"
    assert cf["r_replacement"] == pytest.approx(0.03 * 21 / 252)
    assert cf["confirm_fwd_net"] == pytest.approx(0.03 * 21 / 252 - rate)
    assert cf["delta_defer_minus_confirm"] == pytest.approx(0.10 - 0.03 * 21 / 252)


def test_ledger_schema_aggregates_and_handcheck(ledger_run, ctx_run):
    report, _ = ledger_run
    contexts, _ = ctx_run
    assert report["schema"] == "banddrop-ledger/1"
    assert report["prompt_version"] == sellreview.PROMPT_VERSION_B
    assert report["db"]["read_only"] is True
    assert report["selfcheck_match_rate"] == 1.0
    assert report["aggregate"]["n"] == len(report["events"]) == len(contexts["events"])
    agg = report["aggregate"]
    for key in ("n", "mean", "median", "p5", "p50", "p95", "share_positive"):
        assert key in agg
    assert 0.0 <= agg["share_positive"] <= 1.0
    assert set(report["by_year"]) == {e["decision_date"][:4] for e in report["events"]}
    assert report["baseline_drift"]["aggregate"]["n"] == agg["n"]
    assert len(report["caveats"]) == 4  # declared verbatim
    # hand-check one real event from the PRIMARY data (ledger + frames), not
    # from sellreview's own helpers: recompute the delta with explicit arithmetic
    ev = report["events"][0]
    ctx_ev = next(e for e in contexts["events"]
                  if e["ticker"] == ev["ticker"] and e["fill_date"] == ev["fill_date"])
    assert ev["fill_price"] == ctx_ev["fill_price"]
    rate = reval.CostConfig().rate_for(ev["ticker"], "stock")
    assert ev["oneway_rate"] == pytest.approx(rate)
    r_hold = ev["retention_exit_price"] / ev["fill_price"] - 1.0
    assert ev["r_hold"] == pytest.approx(r_hold)
    assert ev["defer_fwd_net"] == pytest.approx(r_hold - rate)
    if ev["n_additions"] > 0:
        r_repl = sum(a["fwd_return"] for a in ev["additions"]) / ev["n_additions"]
        assert ev["replacement_kind"] == "equal_weight_new_additions"
    else:
        r_repl = sellreview.CASH_YIELD_ANNUAL * ev["sessions_retention_window"] / 252.0
        assert ev["replacement_kind"] == "cash_yield_no_additions"
    assert ev["r_replacement"] == pytest.approx(r_repl)
    assert ev["confirm_fwd_net"] == pytest.approx(r_repl - rate)
    assert ev["delta_defer_minus_confirm"] == pytest.approx(r_hold - r_repl)


# ---------------------------------------------------------------------------
# 5. The B prompt + the A/B parser split
# ---------------------------------------------------------------------------

def test_prompt_b_byte_identical_and_parse_split():
    ctx = {"ticker": "T06", "decision_date": "2025-12-31", "roe": -0.2}
    m1 = sellreview.render_prompt_b(ctx)
    m2 = sellreview.render_prompt_b(ctx)
    assert m1 == m2  # byte-identical rendering
    assert m1[0]["role"] == "system" and sellreview.PROMPT_VERSION_B in m1[0]["content"]
    assert m1[1]["role"] == "user"
    assert m1[1]["content"].startswith(f"prompt-version: {sellreview.PROMPT_VERSION_B}\n")
    # parse_response_b: parameter-free DEFER accepted
    r = sellreview.parse_response_b('{"decision": "DEFER", "rationale": "hold to next rebalance"}')
    assert r == {"decision": "DEFER", "rationale": "hold to next rebalance",
                 "parsed": True, "fail_safe": False}
    assert "rearm_mult" not in r  # DEFER is parameter-free for B
    r = sellreview.parse_response_b('{"decision": "CONFIRM", "rationale": "release"}')
    assert r["decision"] == "CONFIRM" and r["parsed"] is True and r["fail_safe"] is False
    for bad in ("no json here", "", '{"decision": "MAYBE"}', "[1, 2]", "{broken"):
        r = sellreview.parse_response_b(bad)
        assert r["decision"] == "CONFIRM" and r["parsed"] is False and r["fail_safe"] is True
    # the A/B split is real: type A's parser fail-safes a parameterless DEFER
    ra = sellshadow.parse_response('{"decision": "DEFER", "rationale": "hold"}')
    assert ra["decision"] == "CONFIRM" and ra["parsed"] is False and ra["fail_safe"] is True


# ---------------------------------------------------------------------------
# 6. No-write proof + CLI smoke offline
# ---------------------------------------------------------------------------

async def test_no_write_proof(tmp_path):
    mp = pytest.MonkeyPatch()
    _pin_book(mp)
    udir = await _seed_scratch_db(tmp_path)
    before = _snapshot(SCRATCH_DB)
    try:
        c = await sellreview.banddrop_contexts(SCRATCH_URL, UNIVERSE_NAME, udir,
                                               W_START, W_END, generated_at=GENERATED_AT)
        l = await sellreview.banddrop_ledger(SCRATCH_URL, UNIVERSE_NAME, udir,
                                             W_START, W_END, generated_at=GENERATED_AT)
    finally:
        mp.undo()
    assert c["schema"] == "banddrop-contexts/1" and l["schema"] == "banddrop-ledger/1"
    assert _snapshot(SCRATCH_DB) == before


def test_cli_smoke_offline(tmp_path):
    udir = asyncio.run(_seed_scratch_db(tmp_path))
    before = _snapshot(SCRATCH_DB)
    env = dict(os.environ)
    env["DATABASE_URL"] = "sqlite+aiosqlite:////tmp/opencode/reval/ambient_unused.db"
    env["DATABASE_URL_DEV"] = ""
    # pin the small-book settings for the CLI subprocess via a sitecustomize-style
    # wrapper: run through `python -c` is NOT used; instead the CLI runs with the
    # repo defaults, so run it on the short window where the engineered drop lies
    for cmd, fname, schema in (
            ("contexts", "banddrop-contexts.json", "banddrop-contexts/1"),
            ("ledger", "banddrop-ledger.json", "banddrop-ledger/1")):
        out = tmp_path / f"out-{cmd}"
        res = subprocess.run(
            [sys.executable, "-m", "app.sellreview", cmd,
             "--db", SCRATCH_URL, "--universe", UNIVERSE_NAME, "--universes-dir", udir,
             "--start", W_START, "--end", W_END, "--out", str(out),
             "--generated-at", GENERATED_AT],
            cwd=REPO_ROOT, env=env, capture_output=True, text=True, timeout=600)
        assert res.returncode == 0, res.stderr[-3000:]
        path = out / fname
        assert path.exists()
        doc = json.loads(path.read_text())
        assert doc["schema"] == schema
        assert doc["generated_at"] == GENERATED_AT
        assert doc["db"]["read_only"] is True
    assert _snapshot(SCRATCH_DB) == before  # CLI never wrote either
