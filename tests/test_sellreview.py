"""Tests for the Phase 5C slices 5C-A/B/C (``app/sellreview.py``): the unified
reviewer's prospective consult runner + the type-B band-drop adapter + the
type-C thesis-break adapter + the daily shadow command + the assessor.

Self-contained canned factor tape (built in-file — ``tests/test_sellshadow.py``
is owned by a parallel worker and is NOT imported): 11 USD tickers with
candles + ``{start,end,filed,val}`` fundamentals, seeded into the fixed scratch
DB /tmp/opencode/reval/sell5c/scratch.db. T06 (top ROE) sinks from 2025-11-17
and carries a negative filing dated 2025-12-15, so at the 2025-12-31 month-end
its mcap is below the eligibility floor and it drops out of the band -> one
band-drop release at the 2026-01-02 rebuild (settings pinned to a small book:
target_n=4, hold_band=6). T05 sinks from 2026-02-03 for a likely second drop,
and carries a full-year 2025 loss filed 2026-03-10 -> the M5 thesis-break hook
fires mid-month at the 2026-03-11 fill while T05 is still in the band (the
engineered type-C event).

Covers:
1. consult with MOCK transports: valid DEFER (A) parses; malformed / refusal /
   empty -> fail-safe CONFIRM (parsed=False); a raised transport exception ->
   CONFIRM + error class; a timeout -> CONFIRM. The record is deterministic
   given fixed inputs (modulo latency_ms) and NEVER contains the key;
   log_consult appends deterministic JSONL. Type-B consult: parameter-free DEFER.
   Type-C consult: parameter-free EXIT; failures -> fail-safe HOLD.
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
   parser split is real). The C prompt renders byte-identical; parse_response_c
   accepts parameter-free EXIT and fail-safes everything else to HOLD; the
   A/B/C parser map routes correctly; the three prompts differ only in their
   declared semantics.
6. NO-WRITE proof on the scratch DB (row counts unchanged) + CLI smoke offline
   for all subcommands.
7. thesisbreak_contexts on the canned tape: the engineered T05 mid-month break
   is found (decision 2025-12-10, fill 2025-12-11), self-check 1.0, deterministic,
   factor fields finite; thesisbreak_ledger hand-check (pure constants + a real
   event recomputed from primary data).
8. shadow --dry-run on a seeded live state (sim_positions/monthly_positions):
   detects the day's A/B/C candidates, writes shadow-dryrun/1, makes ZERO
   consult attempts (exploding mock transport + exploding httpx).
9. shadow without --dry-run with a MOCK transport: per-type prompt routing
   (A -> sellshadow/1, B -> -b/1, C -> -c/1), decisions + fail-safes logged to
   the JSONL, shadow-run/1 summary correct.
10. assess: synthetic log + canned tape with HAND-COMPUTED per-type deltas;
    the completed-window rule (mid-window event unscored, scored once the data
    advances); fail-safe accounting; reference bars present; determinism +
    no-write.

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
        if t == "T05":
            # the engineered MID-MONTH thesis break: a full-year 2025 LOSS filed
            # 2026-03-10 — public from session 2026-03-10 on, so the M5 hook fires at
            # the 2026-03-11 fill (ROE = TTM NI/equity <= 0) while T05 is still IN the
            # band (T05's mcap only crosses the eligibility floor at the 2026-03-31
            # frame; T06's own break is public only from 2026-01-02, where the band
            # release fires first). A negative-NI (not negative-equity) trigger keeps
            # the frame's roe column numeric for the type-B contexts.
            ni_entries.append({"start": "2025-01-01", "end": "2025-12-31",
                               "filed": "2026-03-10", "val": -1e9})
        equity_entries = [{"start": None, "end": "2024-12-31", "filed": "2025-02-14",
                           "val": 5e9}]
        fundamentals[t] = {
            "StockholdersEquity": equity_entries,
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


async def test_default_transport_is_the_unified_wire(monkeypatch):
    """The default transport delegates to app.llm.chat — ONE wire for the
    shadow CLI and the live layer (backend selection, key precedence,
    payload discipline), temperature/timeout riding through so the wire
    matches the frozen constant every consult record declares."""
    from app import llm as llm_mod
    seen: dict = {}

    async def fake_chat(messages, *, temperature=None, timeout_s=None):
        seen["temperature"] = temperature
        seen["timeout_s"] = timeout_s
        return {"text": '{"decision": "HOLD"}', "backend": "lan",
                "model": "m", "reasoning": ""}

    monkeypatch.setattr(llm_mod, "chat", fake_chat)
    text = await sellreview._default_transport([], "model-id", 0.0, 12.5)
    assert text == '{"decision": "HOLD"}'
    assert seen == {"temperature": 0.0, "timeout_s": 12.5}


async def test_default_transport_raises_on_no_backend(monkeypatch):
    """A no-backend result raises, so the consult's fail-safe table
    classifies it exactly like the old local transport's errors."""
    from app import llm as llm_mod

    async def fake_chat(messages, *, temperature=None, timeout_s=None):
        return {"text": "No LLM backend configured", "backend": "",
                "model": "", "reasoning": ""}

    monkeypatch.setattr(llm_mod, "chat", fake_chat)
    with pytest.raises(RuntimeError):
        await sellreview._default_transport([], "model-id", 0.0, 5.0)


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


# ---------------------------------------------------------------------------
# 7. The type-C adapter: thesis-break contexts + ledger on the canned tape
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
async def tctx_run(tmp_path_factory):
    """One thesisbreak_contexts run (the M5 run) on the canned tape."""
    mp = pytest.MonkeyPatch()
    _pin_book(mp)
    udir = await _seed_scratch_db(tmp_path_factory.mktemp("sell5cc-tctx"))
    try:
        report = await sellreview.thesisbreak_contexts(
            SCRATCH_URL, UNIVERSE_NAME, udir, W_START, W_END, generated_at=GENERATED_AT)
    finally:
        mp.undo()
    return report, udir


@pytest.fixture(scope="module")
async def tledger_run(tmp_path_factory):
    """One thesisbreak_ledger run on the canned tape."""
    mp = pytest.MonkeyPatch()
    _pin_book(mp)
    udir = await _seed_scratch_db(tmp_path_factory.mktemp("sell5cc-tled"))
    try:
        report = await sellreview.thesisbreak_ledger(
            SCRATCH_URL, UNIVERSE_NAME, udir, W_START, W_END, generated_at=GENERATED_AT)
    finally:
        mp.undo()
    return report, udir


def test_tcontexts_schema_events_selfcheck(tctx_run):
    report, _ = tctx_run
    assert report["schema"] == "thesisbreak-contexts/1"
    assert report["prompt_version"] == sellreview.PROMPT_VERSION_C
    assert report["db"]["read_only"] is True
    assert report["generated_at"] == GENERATED_AT
    sc = report["selfcheck"]
    # the engineered mid-month thesis break: T05's negative-equity fact filed
    # 2026-03-10 is public at S=2026-03-10 -> the M5 hook fills at 2026-03-11
    t05 = [e for e in report["events"] if e["ticker"] == "T05"]
    assert t05, report["events"]
    assert len(report["events"]) == 1, report["events"]  # only T05 breaks mid-month
    ev = t05[0]
    assert ev["decision_date"] == "2026-03-10"
    assert ev["fill_date"] == "2026-03-11"
    assert ev["breaking_filing_date"] == "2026-03-10"
    assert ev["days_since_breaking_filing"] == 0
    assert ev["selfcheck"] == {"held_at_prior_session": True,
                               "engine_marked_thesis_break": True,
                               "condition_holds_at_S": True,
                               "fill_price_matches_ledger": True}
    assert sc["events"] == sc["checks_passed"] == 1 and sc["match_rate"] == 1.0
    # the predeclared condition is declared verbatim in every event + header
    assert "ROE = TTM NI / equity <= 0" in report["config"]["thesis_break_condition"]
    assert "ROE = TTM NI / equity <= 0" in ev["thesis_break_condition"]


def test_tcontexts_factor_fields_finite_and_deterministic(tctx_run):
    report, udir = tctx_run
    for ev in report["events"]:
        for key in ("fill_price", "entry_date", "entry_price", "sessions_held", "pnl_pct",
                    "giveback_from_peak_pct", "shares", "position_weight", "roe",
                    "mom_12_1", "bench_drawdown_pct", "bench_vol_ann_20d",
                    "sessions_to_next_rebuild", "next_rebuild_date",
                    "days_since_breaking_filing"):
            assert key in ev, (key, ev)
        assert ev["fill_price"] > 0 and ev["entry_price"] > 0
        assert ev["entry_date"] < ev["fill_date"] and ev["sessions_held"] >= 1
        assert ev["pnl_pct"] is not None and ev["position_weight"] is not None
        assert 0 < ev["position_weight"] < 1
        assert ev["next_rebuild_date"] > ev["fill_date"]
        assert ev["sessions_to_next_rebuild"] >= 1
        assert ev["roe"] is not None  # T05's frame ROE (negative once the break is public)
    # byte-identical rerun (same universe dir — the report embeds its path)
    async def _rerun():
        mp = pytest.MonkeyPatch()
        _pin_book(mp)
        try:
            return await sellreview.thesisbreak_contexts(
                SCRATCH_URL, UNIVERSE_NAME, udir, W_START, W_END, generated_at=GENERATED_AT)
        finally:
            mp.undo()
    r2 = asyncio.run(_rerun())
    assert sellreview._dumps(report) == sellreview._dumps(r2)


def test_thesisbreak_counterfactual_hand_computed():
    """Pin the EXIT-vs-HOLD delta math exactly: both cost legs + the cash-yield
    deployment of the early proceeds."""
    rate = 0.001  # 10 bps one-way
    w = 21        # sessions to the next rebuild
    # early exit at 100, default exit at 112 -> r_default = 0.12 - rate
    cf = sellreview._thesisbreak_counterfactual(100.0, 112.0, w, rate)
    assert cf["r_early_exit"] == pytest.approx((1 - rate) * (1 + 0.03 * w / 252.0) - 1.0)
    assert cf["r_default_hold"] == pytest.approx(0.12 - rate)
    assert cf["early_fwd_net"] == pytest.approx(cf["r_early_exit"])
    assert cf["default_fwd_net"] == pytest.approx(cf["r_default_hold"])
    assert cf["delta_exit_minus_hold"] == pytest.approx(
        cf["r_early_exit"] - (0.12 - rate))
    # missing default exit price -> the delta is None (declared), early leg kept
    cf = sellreview._thesisbreak_counterfactual(100.0, None, w, rate)
    assert cf["r_default_hold"] is None and cf["delta_exit_minus_hold"] is None
    assert cf["r_early_exit"] == pytest.approx((1 - rate) * (1 + 0.03 * w / 252.0) - 1.0)


def test_tledger_schema_aggregates_and_handcheck(tledger_run, tctx_run):
    report, _ = tledger_run
    contexts, _ = tctx_run
    assert report["schema"] == "thesisbreak-ledger/1"
    assert report["prompt_version"] == sellreview.PROMPT_VERSION_C
    assert report["db"]["read_only"] is True
    assert report["selfcheck_match_rate"] == 1.0
    assert report["aggregate"]["n"] == len(report["events"]) == len(contexts["events"]) == 1
    assert len(report["caveats"]) == 4  # declared verbatim
    assert "M5" in report["config"]["reconciliation"]  # the M5-vs-M0 reconciliation note
    ev = report["events"][0]
    ctx_ev = contexts["events"][0]
    assert ev["ticker"] == ctx_ev["ticker"] == "T05"
    assert ev["fill_price"] == ctx_ev["fill_price"]
    # hand-check from PRIMARY data (the tape's candles), explicit arithmetic
    tape = make_tape()
    rows = {r["time"]: r for r in tape["candles"]["T05"]}
    sessions = [d.strftime("%Y-%m-%d") for d in tape["sessions"]]
    rate = reval.CostConfig().rate_for("T05", "stock")
    fill = rows[ev["fill_date"]]["open"]
    exit_px = rows[ev["next_rebuild_date"]]["open"]
    w = sessions.index(ev["next_rebuild_date"]) - sessions.index(ev["fill_date"])
    assert ev["fill_price"] == pytest.approx(fill)
    assert ev["default_exit_price"] == pytest.approx(exit_px)
    assert ev["sessions_window"] == w
    r_early = (1 - rate) * (1 + sellreview.CASH_YIELD_ANNUAL * w / 252.0) - 1.0
    r_default = exit_px / fill - 1.0 - rate
    assert ev["r_early_exit"] == pytest.approx(r_early)
    assert ev["r_default_hold"] == pytest.approx(r_default)
    assert ev["delta_exit_minus_hold"] == pytest.approx(r_early - r_default)
    agg = report["aggregate"]
    assert agg["mean"] == pytest.approx(r_early - r_default)
    assert set(report["by_year"]) == {"2026"}


# ---------------------------------------------------------------------------
# 8. The C prompt + the A/B/C parser map
# ---------------------------------------------------------------------------

def test_prompt_c_byte_identical_and_parser_map_routes():
    ctx = {"ticker": "T05", "decision_date": "2026-03-10", "roe": -1.75}
    m1 = sellreview.render_prompt_c(ctx)
    m2 = sellreview.render_prompt_c(ctx)
    assert m1 == m2  # byte-identical rendering
    assert m1[0]["role"] == "system" and sellreview.PROMPT_VERSION_C in m1[0]["content"]
    assert "EXIT" in m1[0]["content"] and "HOLD" in m1[0]["content"]
    assert m1[1]["content"].startswith(f"prompt-version: {sellreview.PROMPT_VERSION_C}\n")
    # parse_response_c: parameter-free EXIT parsed
    r = sellreview.parse_response_c('{"decision": "EXIT", "rationale": "equity negative"}')
    assert r == {"decision": "EXIT", "rationale": "equity negative",
                 "parsed": True, "fail_safe": False}
    assert "rearm_mult" not in r  # EXIT is parameter-free for C
    r = sellreview.parse_response_c('{"decision": "HOLD", "rationale": "wait"}')
    assert r["decision"] == "HOLD" and r["parsed"] is True and r["fail_safe"] is False
    # malformed / refusal / empty / unknown -> fail-safe HOLD
    for bad in ("no json here", "", "   ", '{"decision": "MAYBE"}', "[1, 2]", "{broken",
                "I cannot answer that."):
        r = sellreview.parse_response_c(bad)
        assert r["decision"] == "HOLD" and r["parsed"] is False and r["fail_safe"] is True
    # the A/B/C parser map routes correctly
    assert sellreview._parser_for("A") is sellshadow.parse_response
    assert sellreview._parser_for("stopfire")("x") == sellshadow.parse_response("x")
    assert sellreview._parser_for("B")("x") == sellreview.parse_response_b("x")
    assert sellreview._parser_for("thesisbreak")("x") == sellreview.parse_response_c("x")
    assert sellreview._parser_for("C")("x") == sellreview.parse_response_c("x")
    # the C fail-safe on consult failure is HOLD, not CONFIRM
    assert sellreview._fail_safe_for("C", "why")["decision"] == "HOLD"
    assert sellreview._fail_safe_for("B", "why")["decision"] == "CONFIRM"
    assert sellreview._fail_safe_for("A", "why")["decision"] == "CONFIRM"
    # the three frozen prompts differ only in their declared semantics
    pa = sellshadow.render_prompt(ctx)[0]["content"]
    pb = sellreview.render_prompt_b(ctx)[0]["content"]
    pc = sellreview.render_prompt_c(ctx)[0]["content"]
    assert len({pa, pb, pc}) == 3
    for p_, pv in ((pa, sellshadow.PROMPT_VERSION), (pb, sellreview.PROMPT_VERSION_B),
                   (pc, sellreview.PROMPT_VERSION_C)):
        assert f"prompt-version: {pv}" in p_
        assert "FAIL-SAFE" in p_ and "OUTPUT FORMAT" in p_
    # per-type prompt routing
    assert sellreview.prompt_version_for("C") == "sellshadow-c/1"
    assert sellreview.render_prompt_for("C", ctx)[0]["content"] == pc
    assert sellreview.render_prompt_for("B", ctx)[0]["content"] == pb
    assert sellreview.render_prompt_for("A", ctx)[0]["content"] == pa


async def test_consult_type_c_routes_and_fail_safe_hold():
    msgs = [{"role": "user", "content": "hi"}]
    r = await sellreview.consult("evC1", "thesisbreak", sellreview.PROMPT_VERSION_C, msgs,
                                 transport=_mock_transport('{"decision": "EXIT", "rationale": "x"}'))
    assert r["decision"] == "EXIT" and r["params"] == {} and r["parsed"] is True
    assert r["prompt_version"] == "sellshadow-c/1"
    # parameter-free EXIT: even with a spurious rearm_mult it is ignored
    r = await sellreview.consult("evC2", "C", sellreview.PROMPT_VERSION_C, msgs,
                                 transport=_mock_transport('{"decision": "EXIT", "rearm_mult": 9}'))
    assert r["decision"] == "EXIT" and r["params"] == {}
    # malformed -> fail-safe HOLD (the C fail-safe, not CONFIRM)
    r = await sellreview.consult("evC3", "C", sellreview.PROMPT_VERSION_C, msgs,
                                 transport=_mock_transport("refusal"))
    assert r["decision"] == "HOLD" and r["parsed"] is False and r["fail_safe"] is True
    # transport failure -> fail-safe HOLD + error class
    r = await sellreview.consult("evC4", "C", sellreview.PROMPT_VERSION_C, msgs,
                                 transport=_mock_transport(exc=TimeoutError()))
    assert r["decision"] == "HOLD" and r["error_class"] == "TimeoutError"


# ---------------------------------------------------------------------------
# 9+10. The daily shadow command + the assessor (canned tape + live state)
# ---------------------------------------------------------------------------

SHADOW_AS_OF = "2026-04-30"   # a completed month-end in the tape (rebuild 2026-05-01,
SHADOW_NEXT = "2026-05-01"    # next month-end 2026-05-29 -> next rebuild 2026-06-01)


def _tape_px(ticker: str) -> dict[str, dict]:
    return {r["time"]: r for r in make_tape()["candles"][ticker]}


async def _seed_live_state(tmp_path):
    """Seed the live-state tables (sim/monthly positions + accounts) for the
    shadow tests, on top of the canned tape."""
    from app.db import MonthlyAccount, MonthlyPosition, SimAccount, SimPosition
    px1 = _tape_px("T01")
    nxt_open = px1[SHADOW_NEXT]["open"]
    engine = create_async_engine(SCRATCH_URL, connect_args={"timeout": 5})
    factory = async_sessionmaker(engine, expire_on_commit=False)
    async with factory() as s:
        await s.execute(SimAccount.__table__.insert(), [
            {"id": 1, "cash": 5000.0, "last_allowance_month": "2026-05"}])
        await s.execute(SimPosition.__table__.insert(), [
            # stored stop above the next open -> fires (stored-stop path)
            {"ticker": "T01", "shares": 2.0, "avg_cost": px1[SHADOW_AS_OF]["close"],
             "opened_at": pd.Timestamp("2026-01-02").to_pydatetime(),
             "thesis": "BUY", "stop_price": nxt_open + 1.0},
            # stop_price NULL -> declared avg_cost fallback, engineered to fire
            # (avg_cost high enough that the derived stop clears both the next
            # open and the ATR ratchet)
            {"ticker": "T02", "shares": 3.0,
             "avg_cost": 250.0,
             "opened_at": pd.Timestamp("2026-01-02").to_pydatetime(),
             "thesis": "BUY", "stop_price": None},
            # a stop that never fires -> not a candidate
            {"ticker": "T03", "shares": 1.0, "avg_cost": 50.0,
             "opened_at": pd.Timestamp("2026-01-02").to_pydatetime(),
             "thesis": "BUY", "stop_price": 0.5},
        ])
        await s.execute(MonthlyAccount.__table__.insert(), [
            {"id": 1, "cash": 3000.0, "last_allowance_month": "2026-05",
             "last_rebalance_month": "2026-04"}])
        await s.execute(MonthlyPosition.__table__.insert(), [
            {"ticker": "T06", "shares": 1.0, "avg_cost": 20.0,
             "opened_at": pd.Timestamp("2026-01-02").to_pydatetime()},  # out of band -> B; thesis broken -> C
            {"ticker": "T05", "shares": 1.0, "avg_cost": 100.0,
             "opened_at": pd.Timestamp("2026-01-02").to_pydatetime()},  # thesis broken -> C
            {"ticker": "T04", "shares": 1.0, "avg_cost": 100.0,
             "opened_at": pd.Timestamp("2026-01-02").to_pydatetime()},  # healthy -> no event
        ])
        await s.commit()
    await engine.dispose()
    return tmp_path


@pytest.fixture(scope="module")
async def shadow_dry(tmp_path_factory):
    """One shadow dry-run on the canned tape + seeded live state."""
    udir = await _seed_scratch_db(tmp_path_factory.mktemp("sell5cc-shadow"))
    await _seed_live_state(tmp_path_factory.mktemp("sell5cc-shadow-live"))
    return await sellreview.run_shadow(
        SCRATCH_URL, UNIVERSE_NAME, udir, SHADOW_AS_OF, dry_run=True,
        generated_at=GENERATED_AT)


def test_shadow_dryrun_detection_and_zero_consults(shadow_dry, monkeypatch):
    import httpx

    class _ExplodingClient:
        def __init__(self, *a, **kw):
            raise AssertionError("httpx.AsyncClient constructed during a dry-run")

    monkeypatch.setattr(httpx, "AsyncClient", _ExplodingClient)

    async def _exploding_transport(*a, **kw):
        raise AssertionError("consult transport invoked during a dry-run")

    monkeypatch.setattr(sellreview, "_default_transport", _exploding_transport)
    rep = shadow_dry
    assert rep["schema"] == "shadow-dryrun/1" and rep["dry_run"] is True
    assert rep["as_of"] == SHADOW_AS_OF and rep["next_session"] == SHADOW_NEXT
    assert rep["consults_attempted"] == 0
    assert rep["db"]["read_only"] is True
    n = rep["n_candidates"]
    a_tickers = {e["context"]["ticker"] for e in rep["events"] if e["event_type"] == "A"}
    b_tickers = {e["context"]["ticker"] for e in rep["events"] if e["event_type"] == "B"}
    c_tickers = {e["context"]["ticker"] for e in rep["events"] if e["event_type"] == "C"}
    assert a_tickers == {"T01", "T02"}, a_tickers       # T03's stop never fires
    assert "T06" in b_tickers                            # out of the band at the rebuild
    assert c_tickers == {"T06", "T05"}, c_tickers        # both break the thesis at the as-of
    assert n == {"A": len(a_tickers), "B": len(b_tickers), "C": len(c_tickers)}
    assert all(e["would_consult"] is True for e in rep["events"])
    # the declared derivations/sourcing are in the header
    assert "avg_cost" in rep["sourcing"]["type_A"]
    assert "monthly_positions" in rep["sourcing"]["type_B_C_book"]
    # the A event that used the NULL-stop fallback declares it
    t02 = next(e["context"] for e in rep["events"]
               if e["event_type"] == "A" and e["context"]["ticker"] == "T02")
    assert t02["stop_price_stored"] is False and "derived" in t02["derivation"]
    t01 = next(e["context"] for e in rep["events"]
               if e["event_type"] == "A" and e["context"]["ticker"] == "T01")
    assert t01["stop_price_stored"] is True
    assert t01["fill_date"] == SHADOW_NEXT and t01["fill_kind"] == "at_open"
    assert t01["stop_level"] >= t01["fill_price"]  # the next open breaches the level
    # the C event carries the thesis-break condition + the assess inputs
    c_ev = next(e for e in rep["events"]
                if e["event_type"] == "C" and e["context"]["ticker"] == "T05")
    assert "ROE = TTM NI / equity <= 0" in c_ev["context"]["thesis_break_condition"]
    assert c_ev["context"]["assess"]["fill_date"] == SHADOW_NEXT
    assert c_ev["context"]["assess"]["next_rebuild_date"] is not None


async def test_shadow_determinism_and_no_write(tmp_path, monkeypatch):
    import httpx
    monkeypatch.setattr(httpx, "AsyncClient", None)  # any httpx use would fail loudly
    udir = await _seed_scratch_db(tmp_path)
    await _seed_live_state(tmp_path)
    before = _snapshot(SCRATCH_DB)
    r1 = await sellreview.run_shadow(SCRATCH_URL, UNIVERSE_NAME, udir, SHADOW_AS_OF,
                                     dry_run=True, generated_at=GENERATED_AT)
    r2 = await sellreview.run_shadow(SCRATCH_URL, UNIVERSE_NAME, udir, SHADOW_AS_OF,
                                     dry_run=True, generated_at=GENERATED_AT)
    assert sellreview._dumps(r1) == sellreview._dumps(r2)
    assert r1["schema"] == "shadow-dryrun/1"
    assert _snapshot(SCRATCH_DB) == before  # read-only: row counts unchanged


async def test_shadow_mock_transport_routing_and_log(tmp_path):
    """Without --dry-run, a MOCK transport: per-type prompts route correctly,
    decisions + fail-safes are logged to the JSONL, the run summary is correct."""
    udir = await _seed_scratch_db(tmp_path)
    await _seed_live_state(tmp_path)
    seen: list[str] = []

    async def transport(messages, model_id, temperature, timeout_s):
        sys_prompt = messages[0]["content"]
        pv = next(pv for pv in (sellshadow.PROMPT_VERSION, sellreview.PROMPT_VERSION_B,
                                sellreview.PROMPT_VERSION_C)
                  if f"prompt-version: {pv}" in sys_prompt)
        seen.append(pv)
        if pv == sellshadow.PROMPT_VERSION:
            return "not json at all"          # -> fail-safe CONFIRM (A)
        if pv == sellreview.PROMPT_VERSION_B:
            return '{"decision": "DEFER", "rationale": "retain"}'
        return '{"decision": "EXIT", "rationale": "thesis broken"}'

    out = tmp_path / "shadow-out"
    rep = await sellreview.run_shadow(
        SCRATCH_URL, UNIVERSE_NAME, udir, SHADOW_AS_OF, dry_run=False,
        out=out, generated_at=GENERATED_AT, transport=transport)
    assert rep["schema"] == "shadow-run/1" and rep["dry_run"] is False
    n = rep["n_candidates"]
    assert rep["n_decided"] == n  # every candidate was consulted
    # prompt routing: A contexts -> sellshadow/1, B -> -b/1, C -> -c/1
    assert seen.count(sellshadow.PROMPT_VERSION) == n["A"]
    assert seen.count(sellreview.PROMPT_VERSION_B) == n["B"]
    assert seen.count(sellreview.PROMPT_VERSION_C) == n["C"]
    dec = {(d["event_type"], d["decision"], d["fail_safe"]) for d in rep["decisions"]}
    assert ("A", "CONFIRM", True) in dec       # the malformed mock answer fail-safed
    assert ("B", "DEFER", False) in dec
    assert ("C", "EXIT", False) in dec
    assert rep["n_fail_safe"] == {"A": n["A"]}
    # the JSONL log: one line per consult, shadow-consult-log/1, carries the event
    log = out / "consult-log.jsonl"
    lines = [json.loads(ln) for ln in log.read_text().splitlines()]
    assert len(lines) == sum(n.values())
    for obj in lines:
        assert obj["schema"] == "shadow-consult-log/1"
        assert obj["event"]["event_type"] in ("A", "B", "C")
        assert obj["event"]["assess"]["ticker"]
    # the run summary records the log path + the declared determinism exclusions
    assert rep["log_path"] == str(log)
    assert "latency_ms" in rep["determinism"]


def _write_assess_log(path: Path):
    """A synthetic consult log with hand-computable events on the canned tape."""
    tape = make_tape()
    sessions = [d.strftime("%Y-%m-%d") for d in tape["sessions"]]
    px = {t: {r["time"]: r for r in tape["candles"][t]} for t in ("T01", "T04", "T06", "T05")}
    fill_d, nr = "2026-06-01", "2026-06-30"
    w = sessions.index(nr) - sessions.index(fill_d)
    lines = [
        # a malformed line + a wrong-schema line (counted + skipped)
        "this is not json",
        json.dumps({"schema": "other/1", "decision": "EXIT"}),
        # A: DEFER with rearm_mult 1.5 on T01 (fill 2026-06-11, mark = close 06-10)
        {"schema": "shadow-consult-log/1", "event_id": "A:T01:2026-06-10",
         "event_type": "stopfire", "prompt_version": sellshadow.PROMPT_VERSION,
         "decision": "DEFER", "params": {"rearm_mult": 1.5}, "parsed": True,
         "fail_safe": False, "rationale": "r",
         "event": {"event_type": "A", "assess": {
             "ticker": "T01", "decision_date": "2026-06-10", "fill_date": "2026-06-11",
             "stop_level": px["T01"]["2026-06-11"]["open"] * 1.02,
             "fill_price": px["T01"]["2026-06-11"]["open"],
             "current_close": px["T01"]["2026-06-10"]["close"]}}},
        # B: DEFER on T06 (no additions -> cash-yield replacement leg)
        {"schema": "shadow-consult-log/1", "event_id": "B:T06:2026-05-29",
         "event_type": "banddrop", "prompt_version": sellreview.PROMPT_VERSION_B,
         "decision": "DEFER", "params": {}, "parsed": True, "fail_safe": False,
         "rationale": "r",
         "event": {"event_type": "B", "assess": {
             "ticker": "T06", "fill_date": fill_d, "fill_price": px["T06"][fill_d]["open"],
             "next_rebuild_date": nr, "sessions_window": w, "additions": []}}},
        # B: CONFIRM on T04 -> delta 0
        {"schema": "shadow-consult-log/1", "event_id": "B:T04:2026-05-29",
         "event_type": "banddrop", "prompt_version": sellreview.PROMPT_VERSION_B,
         "decision": "CONFIRM", "params": {}, "parsed": True, "fail_safe": False,
         "rationale": "r",
         "event": {"event_type": "B", "assess": {
             "ticker": "T04", "fill_date": fill_d, "fill_price": px["T04"][fill_d]["open"],
             "next_rebuild_date": nr, "sessions_window": w, "additions": []}}},
        # C: EXIT on T05
        {"schema": "shadow-consult-log/1", "event_id": "C:T05:2026-05-29",
         "event_type": "thesisbreak", "prompt_version": sellreview.PROMPT_VERSION_C,
         "decision": "EXIT", "params": {}, "parsed": True, "fail_safe": False,
         "rationale": "r",
         "event": {"event_type": "C", "assess": {
             "ticker": "T05", "decision_date": "2026-05-29", "fill_date": fill_d,
             "fill_price": px["T05"][fill_d]["open"],
             "next_rebuild_date": nr, "sessions_window": w}}},
        # C: fail-safe (parsed=False -> HOLD -> the engine branch, delta 0)
        {"schema": "shadow-consult-log/1", "event_id": "C:T06:2026-05-29",
         "event_type": "thesisbreak", "prompt_version": sellreview.PROMPT_VERSION_C,
         "decision": "HOLD", "params": {}, "parsed": False, "fail_safe": True,
         "rationale": "fail-safe: no JSON object",
         "event": {"event_type": "C", "assess": {
             "ticker": "T06", "decision_date": "2026-05-29", "fill_date": fill_d,
             "fill_price": px["T06"][fill_d]["open"],
             "next_rebuild_date": nr, "sessions_window": w}}},
    ]
    path.write_text("\n".join(ln if isinstance(ln, str) else json.dumps(ln) for ln in lines) + "\n")


def _hand_computed_deltas():
    """Independent arithmetic on the tape (no sellreview/sellshadow helpers)."""
    tape = make_tape()
    sessions = [d.strftime("%Y-%m-%d") for d in tape["sessions"]]
    px = {t: {r["time"]: r for r in tape["candles"][t]} for t in ("T01", "T06", "T05")}
    rate = 0.001  # CostConfig stock one-way 10 bps
    fill_d, nr = "2026-06-01", "2026-06-30"
    w = sessions.index(nr) - sessions.index(fill_d)
    # A: DEFER rearm 1.5 -> rearmed = level * (1 - 0.5*0.10); realized defer exit
    level = px["T01"]["2026-06-11"]["open"] * 1.02
    rearmed = level * (1 - 0.5 * 0.10)
    fill_idx = sessions.index("2026-06-11")
    cap = min(fill_idx + 63, len(sessions) - 1)
    exit_px = None
    for j in range(fill_idx + 1, cap + 1):
        r = px["T01"][sessions[j]]
        if r["open"] <= rearmed:
            exit_px = r["open"]
            break
        if r["low"] <= rearmed:
            exit_px = rearmed
            break
    if exit_px is None:
        exit_px = px["T01"][sessions[cap]]["open"]
    mark = px["T01"]["2026-06-10"]["close"]
    fill_a = px["T01"]["2026-06-11"]["open"]
    delta_a = (exit_px - fill_a) / mark  # round-trip cost cancels in the delta
    # B DEFER on T06: r_hold - cash-yield replacement
    fill_b = px["T06"][fill_d]["open"]
    exit_b = px["T06"][nr]["open"]
    delta_b = (exit_b / fill_b - 1.0) - (0.03 * w / 252.0)
    # C EXIT on T05: r_early - r_default
    fill_c = px["T05"][fill_d]["open"]
    exit_c = px["T05"][nr]["open"]
    r_early = (1 - rate) * (1 + 0.03 * w / 252.0) - 1.0
    r_default = exit_c / fill_c - 1.0 - rate
    delta_c = r_early - r_default
    return {"A": delta_a, "B": delta_b, "C": delta_c}


async def test_assess_hand_computed_deltas_and_bars(tmp_path):
    await _seed_scratch_db(tmp_path)
    log = tmp_path / "consult-log.jsonl"
    _write_assess_log(log)
    before = _snapshot(SCRATCH_DB)
    rep = await sellreview.run_assess([str(log)], SCRATCH_URL, as_of="2026-06-30",
                                      generated_at=GENERATED_AT)
    assert rep["schema"] == "shadow-assessment/1"
    assert rep["db"]["read_only"] is True
    assert rep["log_lines"] == {"total": 7, "malformed_or_skipped": 2, "parsed": 5}
    exp = _hand_computed_deltas()
    tA, tB, tC = rep["types"]["A"], rep["types"]["B"], rep["types"]["C"]
    assert tA["n_decided"] == 1 and tA["n_fail_safe"] == 0
    assert tA["aggregate"]["mean"] == pytest.approx(exp["A"])
    assert tB["n_decided"] == 2 and tB["aggregate"]["n"] == 2
    b_deltas = [e["delta"] for e in tB["events"]]
    assert any(d == pytest.approx(0.0) for d in b_deltas)          # the CONFIRM -> delta 0
    assert any(d == pytest.approx(exp["B"]) for d in b_deltas)      # the DEFER hand-check
    assert tC["n_decided"] == 2 and tC["n_fail_safe"] == 1   # the fail-safe is counted
    c_deltas = [e["delta"] for e in tC["events"]]
    assert any(d == pytest.approx(0.0) for d in c_deltas)    # fail-safe HOLD -> engine branch
    assert any(d == pytest.approx(exp["C"]) for d in c_deltas)  # the EXIT hand-check
    # the fail-safe event is flagged
    fs = [e for e in tC["events"] if e["fail_safe"]]
    assert len(fs) == 1 and fs[0]["decision"] == "HOLD" and fs[0]["delta"] == 0.0
    # the reference bars + the completed-window rule + the small-n caveat
    bars = rep["reference_bars"]
    assert bars["A"]["blanket_defer_pp_per_event"] == pytest.approx(0.0293)
    assert bars["B"]["n_events_full_window"] == 165
    assert bars["C"]["m5_vs_m0_pp_per_year"] == pytest.approx(0.0005)
    assert "COMPLETED" in rep["completed_window_rule"]
    assert "p-values" in rep["small_n_caveat"]
    assert rep["as_of"] == "2026-06-30"
    assert _snapshot(SCRATCH_DB) == before  # read-only
    # determinism modulo the declared wall-clock fields
    rep2 = await sellreview.run_assess([str(log)], SCRATCH_URL, as_of="2026-06-30",
                                       generated_at=GENERATED_AT)
    strip = lambda r: {k: v for k, v in r.items() if k != "wall_clock_s"}  # noqa: E731
    assert sellreview._dumps(strip(rep)) == sellreview._dumps(strip(rep2))


async def test_assess_completed_window_rule(tmp_path):
    """A mid-window event is NOT scored yet; it scores once the canned data
    advances past its predeclared window."""
    await _seed_scratch_db(tmp_path)
    log = tmp_path / "consult-log.jsonl"
    _write_assess_log(log)
    # as-of 2026-06-12: no window completed yet (B/C windows end 2026-06-30; the
    # A defer window ends at fill+63 or the last session; no re-armed breach)
    early = await sellreview.run_assess([str(log)], SCRATCH_URL, as_of="2026-06-12",
                                        generated_at=GENERATED_AT)
    assert all(v["n_decided"] == 0 for v in early["types"].values())
    assert early["types"]["A"]["n_window_not_completed"] == 1
    assert early["types"]["B"]["n_window_not_completed"] == 2
    assert early["types"]["C"]["n_window_not_completed"] == 2
    # as-of 2026-06-30 (the last stored session): all windows completed
    late = await sellreview.run_assess([str(log)], SCRATCH_URL, as_of="2026-06-30",
                                       generated_at=GENERATED_AT)
    assert late["types"]["A"]["n_decided"] == 1
    assert late["types"]["B"]["n_decided"] == 2
    assert late["types"]["C"]["n_decided"] == 2
    assert all(v["n_window_not_completed"] == 0 for v in late["types"].values())


async def test_assess_empty_log_clean_exit(tmp_path):
    await _seed_scratch_db(tmp_path)
    log = tmp_path / "empty.jsonl"
    log.write_text("")
    rep = await sellreview.run_assess([str(log)], SCRATCH_URL, generated_at=GENERATED_AT)
    assert rep["schema"] == "shadow-assessment/1"
    assert rep["log_lines"]["parsed"] == 0
    assert all(v["n_decided"] == 0 and v["aggregate"]["n"] == 0 for v in rep["types"].values())
    assert rep["reference_bars"]["B"]["n_events_full_window"] == 165  # bars still in the header


# ---------------------------------------------------------------------------
# 11. CLI smoke for the new subcommands (offline, read-only)
# ---------------------------------------------------------------------------

def test_cli_smoke_offline_new_subcommands(tmp_path):
    udir = asyncio.run(_seed_scratch_db(tmp_path))
    asyncio.run(_seed_live_state(tmp_path))
    before = _snapshot(SCRATCH_DB)
    env = dict(os.environ)
    env["DATABASE_URL"] = "sqlite+aiosqlite:////tmp/opencode/reval/ambient_unused.db"
    env["DATABASE_URL_DEV"] = ""
    empty_log = tmp_path / "empty.jsonl"
    empty_log.write_text("")
    cases = [
        (["tcontexts", "--db", SCRATCH_URL, "--universe", UNIVERSE_NAME,
          "--universes-dir", udir, "--start", W_START, "--end", W_END,
          "--out", str(tmp_path / "o-tctx"), "--generated-at", GENERATED_AT],
         "thesisbreak-contexts.json", "thesisbreak-contexts/1"),
        (["tledger", "--db", SCRATCH_URL, "--universe", UNIVERSE_NAME,
          "--universes-dir", udir, "--start", W_START, "--end", W_END,
          "--out", str(tmp_path / "o-tled"), "--generated-at", GENERATED_AT],
         "thesisbreak-ledger.json", "thesisbreak-ledger/1"),
        (["shadow", "--db", SCRATCH_URL, "--universe", UNIVERSE_NAME,
          "--universes-dir", udir, "--as-of", SHADOW_AS_OF,
          "--out", str(tmp_path / "o-shadow"), "--generated-at", GENERATED_AT],
         "shadow-dryrun.json", "shadow-dryrun/1"),
        (["assess", "--log", str(empty_log), "--db", SCRATCH_URL,
          "--out", str(tmp_path / "o-assess"), "--generated-at", GENERATED_AT],
         "shadow-assessment.json", "shadow-assessment/1"),
    ]
    for cmd, fname, schema in cases:
        res = subprocess.run([sys.executable, "-m", "app.sellreview", *cmd],
                             cwd=REPO_ROOT, env=env, capture_output=True, text=True, timeout=600)
        assert res.returncode == 0, (cmd, res.stderr[-3000:])
        out_dir = Path(cmd[cmd.index("--out") + 1])
        doc = json.loads((out_dir / fname).read_text())
        assert doc["schema"] == schema
        assert doc["generated_at"] == GENERATED_AT
    assert _snapshot(SCRATCH_DB) == before  # CLI never wrote
