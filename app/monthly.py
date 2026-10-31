"""Monthly qv-mom portfolio: stockstrat's qv-mom-v1 as a separate paper
portfolio with a monthly rebalance under the session fill contract
(app/session.py — normative).

Cadence: the ranking DECISION is the completed month-end's qv-mom frame
(month-end ranking frame, unchanged); the rebalance FILLS land at the FIRST
session after that month-end, at that session's strict exact-date OPEN per
ticker (never a stale or forward-filled close) — one session of cash drag,
which is the faithful option. Legs that cannot fill (no stored bar for the
ticker that session) are retried at the following sessions until the
membership is complete; a newer month-end ranking supersedes a still-pending
one. The audit row keeps the decision date (``rebal_date`` = the month-end)
distinct from the fill date (the trades' ``created_at`` = the fill session).

Strategy rules (deterministic, ported from stockstrat qv-mom-v1):
  * eligibility: mcap > $5B, 20d avg dollar volume > $10M, >=253 trading days
    of valid closes, finite 12-1 momentum
  * score = pct_rank(ROE) + pct_rank(12-1 momentum) + pct_rank(1/P-FCF)
  * total order: score desc -> ROE desc -> P/FCF asc (missing last) -> ticker asc
  * hysteresis: target 10 names; a held name keeps its slot while it stays in
    the top ``hold_band`` of the ranking; new names fill open slots from the top
  * equal weight (1/N of equity), fractional shares
  * no stops, no daily risk rules — the band IS the risk control

Point-in-time discipline (no look-ahead): a rebalance on date d only sees
fundamental facts with end < d and filed <= d; momentum uses closes in
[t-252td, t-21td]; market cap uses the latest share count public at d.

Non-USD listings: closes are converted to USD at the historical FX rate from
the candles cache (FX pairs stored as pseudo-tickers, e.g. EURUSD=X), so
market cap / dollar volume / momentum are comparable across the universe.
"""
import asyncio
import hashlib
import json
import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta, UTC
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from sqlalchemy import func, or_, select

from . import factor_fills
from . import fundamentals as fundamentals_mod
from . import instruments
from . import price_freshness
from . import run_config
from .config import settings
from .db import (Candle, MonthlyAccount, MonthlyAllowance, MonthlyPosition,
                 MonthlyRebalance, MonthlySnapshot, MonthlyTrade, Session)
from .screener import tickers as universe_tickers

logger = logging.getLogger("trade_sentinel.monthly")

# The RunConfig registry's engine key for this portfolio (app/run_config.py).
_ENGINE = "monthly"

# Bumped whenever the MEANING of the config snapshot changes, so a config
# recorded by an older contract is never read as equivalent to a newer one.
_CONFIG_CONTRACT_VERSION = "phase2-v1"


def _strategy_universe() -> str:
    """The runtime-selected qv-mom universe (daily-core state file), falling
    back to the config default. Imported lazily: daily_core imports this module,
    so a module-level import here would be circular."""
    try:
        from . import daily_core
        return daily_core.current_universe()
    except Exception:  # noqa: BLE001 — never let selection break a cycle
        return settings.sim_monthly_universe


def _strategy_variant() -> str:
    """The runtime-selected momentum variant (daily-core state file), falling
    back to the config default. The qv-mom ranking is shared by the monthly and
    daily-core portfolios, so both must resolve it the same way — relying on
    the ambient ``settings`` value made the monthly replay depend on call
    order (whoever set it last won)."""
    try:
        from . import daily_core
        return daily_core.current_variant()
    except Exception:  # noqa: BLE001
        return settings.sim_daily_core_mom_variant


@dataclass
class BackfillPreload:
    """Inputs every qv-mom backfill replay shares for one universe + window.

    ``frames`` is a read-through cache keyed by month-end: the first replay
    that asks for a month computes it, the second reuses it. ``backfill-all``
    builds one of these and hands it to both the monthly and daily-core
    replays, which otherwise load the same fundamentals/frames twice (the
    frames are the expensive part — ~1.6s per month on a 500-name universe).
    ``open`` is the USD-converted strict-open fill series (session fill
    contract rule 2): the replays fill at the fill session's exact open,
    never a stale close.
    """
    tickers: list[str]
    fund: dict[str, dict[str, list[dict]]]
    open: pd.DataFrame
    close: pd.DataFrame
    vol: pd.DataFrame
    frames: dict[pd.Timestamp, pd.DataFrame | None] = field(default_factory=dict)


async def preload_backfill(start: str | None) -> BackfillPreload:
    """Load the inputs shared by the monthly and daily-core backfills for the
    same window: point-in-time fundamentals and the USD open/close/volume
    frames (frames are computed lazily, per month, by the replays themselves)."""
    settings.sim_daily_core_mom_variant = _strategy_variant()
    tickers = universe_tickers(_strategy_universe())
    fund = await fundamentals_mod.load_fundamentals(tickers)
    load_start = ((pd.Timestamp(start) - timedelta(days=_BACKFILL_WARMUP_DAYS)).to_pydatetime()
                  if start else None)
    open_, close, vol = await load_ohlcv_frames(tickers, None, start=load_start)
    return BackfillPreload(tickers=tickers, fund=fund, open=open_, close=close, vol=vol)

# Anchor for the allowance "month" key: the same operator-local timezone the
# sim portfolio uses (settings.allowance_tz), so both portfolios' cumulative
# contributed figures step at the same calendar-month boundary and the two
# equity curves are directly comparable.
_TZ = ZoneInfo(settings.allowance_tz)

# Recent-window bound for "latest price" lookups (see _price_usd_map): long
# enough that a stale feed still resolves, short enough to skip the decades of
# unused history that broad universes carry.
_PRICE_LOOKBACK_DAYS = 400

# Momentum warmup kept ahead of a backfill start when bounding the candle load.
_BACKFILL_WARMUP_DAYS = 600

# Candle-history bound for the LIVE monthly ranking (~4y): ample for the 253d
# momentum warmup, without loading decades of unused history every rebalance.
_LIVE_HISTORY_DAYS = 1460


def _now() -> datetime:
    """The live decision path's clock (UTC) — injectable for tests.

    Every ``datetime.now(UTC)`` on a decision path routes through this hook so
    the golden parity tests (tests/test_cadence_parity.py) can drive the live
    deployment session by session through a fake multi-session market;
    production returns real UTC now."""
    return datetime.now(UTC)


def _live_history_start() -> datetime:
    """Naive-UTC lower bound for the live ranking's candle load. Candle
    timestamps are stored naive, so the bound must be naive too."""
    return _now().replace(tzinfo=None) - timedelta(days=_LIVE_HISTORY_DAYS)


def _current_month() -> str:
    """Operator-local calendar month key (YYYY-MM), matching sim._current_month."""
    return _now().astimezone(_TZ).strftime("%Y-%m")


# ---------------------------------------------------------------------------
# Run configuration (the immutable provenance half of Phase 2)
# ---------------------------------------------------------------------------
#
# RULE (provenance contract, restated on every engine's builder): the snapshot
# must capture EVERY input that changes a decision. A configuration that
# changed a decision but not the config hash is a provenance bug. The field
# list is grep-verified against this module's ``settings.*`` reads, the
# persisted universe/variant selection and the eligibility thresholds
# ``eligible_frame`` reads (the ranking path).
# Deliberately excluded, with rationale:
#   * ``sim_monthly_enabled`` — feature gate (a disabled portfolio runs nothing).
#   * ``market_fresh_seconds`` — data-refresh plumbing; the Candle provenance
#     columns cover the DATA, not the run policy.
#   * ``sim_monthly_cost_oneway`` — NOT read by this live path (research
#     replay only), and the live paper book is fee-free.
#   * ``sim_run_hour``/``sim_run_minute`` and the benchmark settings — timing
#     and comparison artifacts, never a monthly decision input.


def _universe_content(name: str) -> tuple[dict, list[str] | None]:
    """The universe's ACTUAL membership (content + sorted-join sha256).

    A universe file whose membership changed is a different configuration;
    recording only the stem would equate two different universes. Unresolvable
    selections keep None rather than fabricating a membership. Never raises:
    provenance capture must not break a cycle.

    The membership CONTENT is under the single top-level ``universe_members``
    key — the one key ``run_provenance.summarize`` reads to scope its
    candle/fundamental provenance query (exactly one members key per config).
    """
    try:
        members = sorted(universe_tickers(name))
    except Exception:  # noqa: BLE001 — provenance must never break a cycle
        return {"name": name, "members_hash": None}, None
    return (
        {"name": name,
         "members_hash": hashlib.sha256(",".join(members).encode()).hexdigest()},
        members,
    )


def current_config() -> dict:
    """The immutable configuration snapshot for ONE monthly run.

    Effective values (the persisted universe/variant selections) decide when
    set; the configured defaults are captured because they decide when the
    state file leaves the key unset. Tuples are stored as lists for JSON
    stability; no secret or API key is part of a run configuration.
    """
    universe = _strategy_universe()
    universe_meta, universe_members = _universe_content(universe)
    return {
        "contract_version": _CONFIG_CONTRACT_VERSION,
        "engine": _ENGINE,
        "currency": "USD",
        "fill_basis": "session strict exact-date open (app/session.py rule 2), zero fees",
        "fill_basis_note": (
            "the technical sim's historical replay (app/optimize.py) still "
            "fills each decision at that bar's same close — a documented "
            "legacy approximation, not this engine's live contract"),
        "timezone": settings.allowance_tz,
        # --- effective runtime selections (shared daily-core state file) ---
        # EFFECTIVE values only: the raw fallback settings are NOT repeated —
        # a persisted selection shadows them, so recording both would mint a
        # new "configuration tried" for a setting that changed no decision.
        "variant": _strategy_variant(),
        "universe": universe_meta,
        # The single top-level members key ``run_provenance.summarize`` reads
        # to scope its candle/fundamental provenance query.
        "universe_members": universe_members,
        # --- hard-coded decision constants (see the contract bump rule) ---
        "code": {
            "momentum_lookback": 252,
            "momentum_skip": 21,
            "dollar_vol_window": 20,
            "vol_window": 121,
        },
        # --- portfolio construction / funding -----------------------------
        "sim_monthly_target_n": settings.sim_monthly_target_n,
        "sim_monthly_hold_band": settings.sim_monthly_hold_band,
        "sim_monthly_contribution": settings.sim_monthly_contribution,
        "sim_monthly_start_cash": settings.sim_monthly_start_cash,
        # --- eligibility thresholds (read inside eligible_frame) ----------
        "sim_monthly_min_mcap": settings.sim_monthly_min_mcap,
        "sim_monthly_min_dollar_vol": settings.sim_monthly_min_dollar_vol,
        "sim_monthly_min_history_days": settings.sim_monthly_min_history_days,
        # The shared qv-mom score's low-vol tilt (read by _qv_order).
        "sim_daily_core_lowvol_tilt": getattr(settings, "sim_daily_core_lowvol_tilt", False),
        # Marks/ranking price-age policy (app/price_freshness.py).
        "factor_price_max_age_days": settings.factor_price_max_age_days,
        # Fundamentals source: EDGAR-first vs yfinance-only changes the FACTS
        # a decision sees (filed dates / coverage), so it is a decision input.
        "sim_monthly_fundamentals_source": settings.sim_monthly_fundamentals_source,
        # --- LLM sell layer (app/selllayer.py) — provenance: these knobs are
        # decision inputs of the live monthly cycle (types B + C).
        "sim_monthly_llm_sells": settings.sim_monthly_llm_sells,
        "sell_layer_timeout_s": settings.sell_layer_timeout_s,
    }


async def _paper_stamp(config: dict | None = None, *, record: bool = True) -> dict:
    """Resolve the stamp for ONE forward PAPER run: ``(mode, config_id, epoch)``.

    See ``daily_core._paper_stamp``: resolve ONCE per logical run and thread
    it to the writers; the fallback path uses THIS module's ``Session``
    factory so a monkeypatched in-memory DB is never bypassed.
    ``record=False`` defers the RunExecution to the caller's stable input
    point (the rebalance pass records after its refresh/loads and after the
    same-month provenance check).
    """
    cfg = current_config() if config is None else config
    stamp = {
        "mode": "paper",
        "config_id": await run_config.get_or_create(
            _ENGINE, "paper", cfg, session_factory=Session),
        "epoch": await run_config.current_epoch(_ENGINE, session_factory=Session),
    }
    # A stamp scope IS a logical run: record its data provenance exactly once
    # (a run that threads this stamp never records again; a standalone public
    # writer that resolves its own stamp is a small logical run of its own).
    if record:
        await _record_run(stamp, cfg)
    return stamp


async def _record_run(stamp: dict, cfg: dict, *,
                      asof: datetime | None = None,
                      session=None) -> dict:
    """Summarize the run's DATA provenance and append one RunExecution.

    The clock is an explicit ``asof`` or the WALL clock at recording time —
    deliberately NOT the injectable decision clock (``_now``), so recording
    provenance can never consume or perturb a frozen decision/fill timestamp
    (session fill contract rule 7); in production the two coincide. The
    summary is an upper-bounded aggregate of the candle/fundamental cache the
    run could read, unknown provenance preserved — NOT an immutable data
    snapshot and NOT a byte-exact manifest.
    The row records that the run EXECUTED (configuration counts live in
    RunConfig), and it is written after the refresh so it describes the cache
    the decision actually read.
    """
    from . import run_provenance
    clock = datetime.now(UTC) if asof is None else asof
    # A caller-owned session keeps the execution row in the caller's
    # transaction (the helpers never commit it): a backfill's rows and its
    # execution record then commit — or roll back on a late forward-record
    # refusal — atomically.
    scope = {"session": session} if session is not None else {"session_factory": Session}
    summary = await run_provenance.summarize(cfg, asof=clock, **scope)
    # The run's capital basis: the account's frozen opening seed, or an
    # explicit UNKNOWN for a legacy account (never an inferred value).
    if session is not None:
        acc = await session.get(MonthlyAccount, 1)
    else:
        async with Session() as meta_s:
            acc = await meta_s.get(MonthlyAccount, 1)
    opening = None if acc is None else acc.opening_seed
    summary["opening_seed"] = {
        "value": opening,
        "known": opening is not None,
        "basis": ("account_opening_seed" if opening is not None
                  else "conservative_zero_legacy_unknown"),
    }
    await run_config.record_execution(
        _ENGINE, stamp["mode"], stamp["config_id"], stamp["epoch"], summary,
        data_asof=clock, **scope)
    return summary


async def _stamp_is_current(s, stamp: dict) -> bool:
    """True when the stamp's epoch is the CURRENT epoch (read in the caller's
    session, i.e. under the state lock + its transaction)."""
    return await run_config.current_epoch(_ENGINE, session=s) == stamp["epoch"]


async def _resolve_paper_stamp(*, mode: str | None = None,
                               config_id: int | None = None,
                               epoch: int | None = None) -> dict:
    """The stamp a public mutator writes with: explicit kwargs (a logical
    run's frozen stamp) win, else a fresh PAPER stamp. Standalone calls
    (status endpoint, a hand-run executor) are forward paper writes too and
    are stamped ``paper`` by default; all-None/NULL exists only for direct
    model construction, never for an engine write."""
    if mode is not None and config_id is not None and epoch is not None:
        return {"mode": mode, "config_id": config_id, "epoch": epoch}
    fresh = await _paper_stamp()
    return {
        "mode": "paper" if mode is None else mode,
        "config_id": fresh["config_id"] if config_id is None else config_id,
        "epoch": fresh["epoch"] if epoch is None else epoch,
    }


def _config_changed(frozen: dict) -> bool:
    """True when the runtime configuration no longer matches the one this run
    froze at its start (best-effort mid-run guard; see
    ``daily_core._config_changed``)."""
    return run_config.config_hash(current_config()) != run_config.config_hash(frozen)


async def _epoch_seed(s) -> float:
    """The opening SEED capital of the CURRENT epoch — the account's own
    ``opening_seed``, FROZEN when the account was opened/re-seeded.

    A live-setting read would retroactively rewrite the contributed basis when
    ``sim_monthly_start_cash`` changes without a reset (the account still holds
    the old capital). The field is written in the SAME transaction that
    creates/re-seeds the account and by an explicit historical replay
    (backfill sets 0). A LEGACY account (NULL) has an UNKNOWN seed: the
    conservative fallback is 0.0 and ``_record_run`` flags
    ``opening_seed.known = False`` — never fabricate historical funding.
    """
    acc = await s.get(MonthlyAccount, 1)
    seed = None if acc is None else acc.opening_seed
    if seed is not None:
        return float(seed)
    return 0.0


# Serializes rebalances: the month-idempotence check runs before the
# multi-minute refresh_data, so two overlapping invocations (scheduler +
# manual UI run) would both pass the gate and double-deposit the allowance /
# duplicate trades. Mirrors sim._run_cycle_lock.
_rebalance_lock: asyncio.Lock = asyncio.Lock()
# Upper bound on how long the scheduler entry waits for the busy lock before
# giving up with the old "skipped" semantics (a DEV backfill can hold it for
# minutes; a wedged holder should not park the scheduler forever). The manual
# run_rebalance() keeps its own fail-fast skip.
_TICK_LOCK_WAIT_SECONDS = 7200

# Serialises every STATE/CASH/HISTORY mutation against a reset's boundary (the
# stamp/reset TOCTOU): a stamp captured before a reset is validated against the
# CURRENT epoch INSIDE this lock and its transaction, and refuses without
# writing into the new epoch's book. Lock order is ALWAYS rebalance-lock ->
# state-lock (never the reverse), so no nested path can deadlock.
_state_write_lock: asyncio.Lock = asyncio.Lock()


# ---------------------------------------------------------------------------
# Point-in-time fact helpers (ported from stockstrat/strategy.py)
# ---------------------------------------------------------------------------

def _asof_entries(entries: list[dict], asof: pd.Timestamp) -> list[dict]:
    """Entries public as of `asof`, dedup'd per (start,end) keeping the
    most-recently-filed value (restatements replace originals)."""
    a = asof.strftime("%Y-%m-%d")
    best: dict[tuple, dict] = {}
    for e in entries:
        if e["end"] >= a or e["filed"] > a:
            continue
        key = (e.get("start"), e["end"])
        if key not in best or e["filed"] > best[key]["filed"]:
            best[key] = e
    return list(best.values())


def _latest_as_of(entries: list[dict], asof: pd.Timestamp) -> tuple[str, float] | None:
    """Latest instant value (start=None facts) public as of `asof`."""
    best = None
    for e in _asof_entries(entries, asof):
        if e.get("start") is None:  # instant facts only (equity, shares)
            if best is None or e["end"] > best["end"]:
                best = e
    if best is None:
        return None
    return best["end"], float(best["val"])


def _ttm_as_of(entries: list[dict], asof: pd.Timestamp) -> float | None:
    """Trailing-12-month value public as of `asof`.

    Prefers the classic identity TTM = latest FY + latest YTD - prior-year
    YTD; falls back to a chain of 4 consecutive reported quarters (the
    yfinance feed usually has standalone quarters only)."""
    ents = _asof_entries(entries, asof)
    durs: list[tuple[pd.Timestamp, pd.Timestamp, int, float]] = []
    for e in ents:
        if not e.get("start"):
            continue
        s = pd.Timestamp(e["start"])
        end = pd.Timestamp(e["end"])
        days = int((end - s).days)
        if 80 <= days <= 380:
            durs.append((s, end, days, float(e["val"])))
    if not durs:
        return None
    durs.sort(key=lambda x: x[1])
    E = durs[-1][1]  # latest period end public at asof

    # a) a full fiscal year ending at/just before the latest period end
    fys = [d for d in durs if d[2] >= 340]
    fy_at_e = [d for d in fys if d[1] >= E - pd.Timedelta(days=45)]
    if fy_at_e:
        return max(fy_at_e, key=lambda d: d[1])[3]

    # b) FY + YTD - prior-year YTD (genuine year-to-date spans only)
    ytds = [d for d in durs if d[1] >= E - pd.Timedelta(days=7) and d[2] <= 300]
    if ytds:
        y_start, _y_end, y_days, y_val = max(ytds, key=lambda d: d[2])
        fys_before = [d for d in fys if d[1] <= y_start + pd.Timedelta(days=7)]
        if fys_before:
            fy0 = max(fys_before, key=lambda d: d[1])
            if (y_start - fy0[1]).days <= 45:
                pri = [
                    d for d in durs
                    if abs(d[2] - y_days) <= 15
                    and E - pd.Timedelta(days=380) <= d[1] <= E - pd.Timedelta(days=340)
                ]
                if pri:
                    p = max(pri, key=lambda d: d[1])
                    return fy0[3] + y_val - p[3]

    # c) fallback: 4 consecutive quarters (small calendar overlaps tolerated),
    #    gap between periods <= 45d, covering >= 330 days
    qs = [d for d in durs if d[2] <= 100]
    for k in range(len(qs) - 1, -1, -1):
        total, cnt = qs[k][3], 1
        start_cur = qs[k][0]
        j = k - 1
        while j >= 0 and cnt < 4:
            s_j, e_j, _d_j, v_j = qs[j]
            gap = (start_cur - e_j).days
            if -10 <= gap <= 45:
                total += v_j
                start_cur = s_j
                cnt += 1
                j -= 1
            else:
                break
        if cnt == 4 and (qs[k][1] - start_cur).days >= 330:
            return total
    return None


# ---------------------------------------------------------------------------
# Prices / volume / FX from the candle cache
# ---------------------------------------------------------------------------

def _fx_convert_usd(frame: pd.DataFrame) -> pd.DataFrame:
    """Convert a raw {date x ticker} price frame to USD in place-style.

    Same suffix->FX mapping and historical-rate convention as
    ``load_frames_sync`` (the pair's own closes ffilled across the frame
    index: the rate prevailing at each date). FX pseudo-ticker columns are
    dropped afterwards.

    Quote units come from ``app/instruments.py`` and are applied BEFORE the FX
    rate: London (``.L``) closes are quoted in PENCE, so 1250.0 means £12.50
    and becomes 1250.0 * 0.01 * GBPUSD — e.g. $15.625 at 1.25, not the
    100x-inflated $1562.50 a bare pence * GBPUSD would report. Every other
    listing's quote unit is 1.0, so its arithmetic is unchanged.

    Fail-closed currency contract: a KNOWN non-USD suffix is converted only
    when a usable rate (finite and > 0) exists at or before each date. A
    missing pair column, a pair with no prior print, or a NaN/inf/<=0 rate
    leaves that ticker's cells NaN — the native-currency price is NEVER
    passed through as USD. USD tickers (no suffix) and unknown suffixes stay
    unchanged: the latter's currency is explicitly unresolved, and guessing
    would silently misstate exposure."""
    out = frame.copy()
    for t in list(out.columns):
        pm = fundamentals_mod._suffix_fx(t)
        if not pm:
            continue
        if pm[0] not in out.columns:
            out[t] = np.nan
            continue
        rate = out[pm[0]].reindex(out.index).ffill()
        # Invalid prints are not rates: fail closed at those dates instead of
        # treating the local price as USD. Masking AFTER the causal ffill is
        # deliberate: a NaN quote carries the prior valid rate forward (the
        # documented convention), while inf/<=0 never becomes a multiplier.
        rate = rate.where(np.isfinite(rate) & (rate > 0))
        # Quote unit first (pence -> pounds), then the FX rate. The alias keeps
        # the exact previous arithmetic for unit-1.0 venues (.DE/.SW/...).
        unit = instruments.quote_unit_factor(t)
        local = out[t] * unit if unit != 1.0 else out[t]
        if pm[1] == "mul":
            out[t] = local * rate
        else:
            out[t] = local / rate
    drop = [c for c in out.columns if c.endswith("=X")]
    return out.drop(columns=drop)


def load_frames_sync(tickers: list[str], rows: list, asof: datetime | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build (close, volume) frames from candle rows (sync, no DB)."""
    closes: dict[str, dict[pd.Timestamp, float]] = {}
    vols: dict[str, dict[pd.Timestamp, float]] = {}
    for r in rows:
        # candles are stored UTC; strip tzinfo so the frame index is naive and
        # slices with a naive rebalance date work (DateTime re-attaches UTC on load)
        ts = pd.Timestamp(r.timestamp)
        if ts.tzinfo is not None:
            ts = ts.tz_localize(None)
        closes.setdefault(r.ticker, {})[ts] = r.close
        vols.setdefault(r.ticker, {})[ts] = r.volume
    close = pd.DataFrame({t: pd.Series(v).sort_index() for t, v in closes.items()})
    vol = pd.DataFrame({t: pd.Series(v).sort_index() for t, v in vols.items()})

    close = _fx_convert_usd(close)
    vol = vol.drop(columns=[c for c in vol.columns if c not in close.columns])
    # FX-only calendar rows: currency pairs also print on days US equities
    # don't trade (holidays, today-before-tonight's-close). A trailing row
    # without a single stock close would make eligible_frame see every
    # ticker as invalid on the rebalance date — drop rows where the
    # remaining (stock) columns are all NaN. This runs *after* FX
    # conversion: a stock missing the FX-only day stays NaN there, so
    # all-NaN rows identify themselves.
    close = close.dropna(how="all")
    vol = vol.reindex(columns=close.columns).loc[close.index]
    close = close.reindex(columns=[t for t in tickers if t in close.columns])
    vol = vol.reindex(columns=close.columns)
    return close, vol


def load_ohlcv_frames_sync(tickers: list[str], rows: list,
                           asof: datetime | None = None,
                           ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """(open, close, volume) frames from candle rows (sync, no DB).

    ``load_frames_sync`` plus the OPEN series the session fill contract's
    fill rule needs (app/session.py rule 2: fills execute at F's open, read
    from the stored bar with an exact date match). The open frame is NEVER
    forward-filled — a date where a ticker has no stored bar carries NaN
    there, and the executor treats it as "no fill this session". Both price
    series go through the same USD conversion as the closes."""
    opens: dict[str, dict[pd.Timestamp, float]] = {}
    closes: dict[str, dict[pd.Timestamp, float]] = {}
    vols: dict[str, dict[pd.Timestamp, float]] = {}
    for r in rows:
        ts = pd.Timestamp(r.timestamp)
        if ts.tzinfo is not None:
            ts = ts.tz_localize(None)
        if r.open is not None:
            opens.setdefault(r.ticker, {})[ts] = r.open
        closes.setdefault(r.ticker, {})[ts] = r.close
        vols.setdefault(r.ticker, {})[ts] = r.volume

    def _pivot(col: dict[str, dict[pd.Timestamp, float]]) -> pd.DataFrame:
        return pd.DataFrame({t: pd.Series(v).sort_index() for t, v in col.items()})

    close = _fx_convert_usd(_pivot(closes))
    open_ = _fx_convert_usd(_pivot(opens))
    vol = _pivot(vols)
    vol = vol.drop(columns=[c for c in vol.columns if c not in close.columns])
    # Same FX-only-calendar cleanup as load_frames_sync, keyed on the CLOSES.
    close = close.dropna(how="all")
    idx = close.index
    open_ = open_.reindex(index=idx).reindex(columns=close.columns)  # strict, no ffill
    vol = vol.reindex(columns=close.columns).loc[idx]
    close = close.reindex(columns=[t for t in tickers if t in close.columns])
    open_ = open_.reindex(columns=close.columns)
    vol = vol.reindex(columns=close.columns)
    return open_, close, vol


def px_at(frame: pd.DataFrame, d: pd.Timestamp, ticker: str) -> float | None:
    """Scalar close lookup: frame.at[d, ticker] as a float, None when the
    cell is missing or NaN. Shared by the live engines and the backtests so
    the pandas scalar typing quirk is handled in exactly one place."""
    try:
        p = frame.at[d, ticker]
    except KeyError:
        return None
    if pd.isna(p):
        return None
    return float(np.asarray(p).reshape(-1)[0])


async def load_frames(tickers: list[str], asof: datetime | None = None,
                      start: datetime | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(close, volume) DataFrames {date: {ticker: value}} from the candles
    cache, loaded in a thread. Foreign-listing closes are converted to USD at
    the historical FX rate from the FX pseudo-tickers in the same cache.

    ``start`` optionally bounds the history: the strategy only ever needs a
    momentum-warmup window before its decision date, and after the broad
    universes were fetched with full history ~61% of the candle table sits
    before 2015 — loading it on every call is pure waste. Callers that need
    the full series (e.g. a ``start=all`` backfill) pass ``None``."""
    async with Session() as s:
        # Load ONLY the requested tickers (+ the FX pairs their closes need
        # converting), and only the four columns the frame builder reads —
        # full-ORM hydration of millions of rows was the other half of the
        # slowness. Same ticker-filter pattern as _price_usd_map.
        pairs = sorted({pm[0] for t in tickers if (pm := fundamentals_mod._suffix_fx(t))})
        q = (select(Candle.ticker, Candle.timestamp, Candle.close, Candle.volume)
             .where(Candle.ticker.in_(set(tickers) | set(pairs)))
             .order_by(Candle.timestamp))
        if start is not None:
            q = q.where(Candle.timestamp >= start)
        if asof is not None:
            q = q.where(Candle.timestamp <= asof)
        rows = list((await s.execute(q)).all())
    return await asyncio.to_thread(load_frames_sync, tickers, rows, asof)


async def load_ohlcv_frames(tickers: list[str], asof: datetime | None = None,
                            start: datetime | None = None,
                            ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """(open, close, volume) DataFrames from the candles cache — the factor
    portfolios' loader WITH the strict-open fill series (the session fill
    contract's rule 2). Same bounds/conversion as ``load_frames``."""
    async with Session() as s:
        pairs = sorted({pm[0] for t in tickers if (pm := fundamentals_mod._suffix_fx(t))})
        q = (select(Candle.ticker, Candle.timestamp, Candle.open, Candle.close,
                    Candle.volume)
             .where(Candle.ticker.in_(set(tickers) | set(pairs)))
             .order_by(Candle.timestamp))
        if start is not None:
            q = q.where(Candle.timestamp >= start)
        if asof is not None:
            q = q.where(Candle.timestamp <= asof)
        rows = list((await s.execute(q)).all())
    return await asyncio.to_thread(load_ohlcv_frames_sync, tickers, rows, asof)


# ---------------------------------------------------------------------------
# Eligibility + scoring
# ---------------------------------------------------------------------------

def eligible_frame(rebal_date: pd.Timestamp, close: pd.DataFrame, vol: pd.DataFrame,
                   fundamentals: dict[str, dict[str, list[dict]]]) -> pd.DataFrame | None:
    """DataFrame index=ticker, columns: roe, p_fcf, mcap, dollar_vol, mom,
    price, vol (annualized realized), stale, eligible.

    ``stale`` is the price-age policy's flag (``app/price_freshness.py``): a
    ticker whose last valid close is older than
    ``settings.factor_price_max_age_days`` relative to the ranking date is
    marked ``stale=True`` and excluded from ``eligible`` — an indefinitely
    stale listing must not remain silently eligible, however strong its
    fundamentals/momentum. Its other columns keep the last known values so
    consumers can display/report the name; eligibility is the only
    consequence.
    """
    d = pd.Timestamp(rebal_date)
    hist = close.loc[:d]
    if hist.empty or len(hist) < settings.sim_monthly_min_history_days:
        return None
    vhist = vol.loc[:d]

    # 20-day average dollar volume (raw volume * USD close)
    dv20 = (hist * vhist).tail(20).mean()

    valid = hist.notna()
    n_valid = valid.sum()

    c = hist.ffill()
    # 12-1 momentum: close[t-21td] / close[t-252td] - 1. Default "raw";
    # settings.sim_daily_core_mom_variant == "residual" swaps in the
    # Blitz-Huij-Martens residual momentum (mom per unit of idiosyncratic
    # vol) — same `mom` column semantic downstream (higher = better) so the
    # rest of the frame logic is untouched.
    if len(hist) <= 252:
        return None
    if getattr(settings, "sim_daily_core_mom_variant", "raw") == "residual":
        mom = residual_momentum(hist, d)
    else:
        mom_start = c.iloc[-21]
        mom_end = c.iloc[-252]
        mom = mom_start / mom_end - 1.0
    vol_ann = hist.tail(121).pct_change().std() * np.sqrt(252.0)

    rows: dict[str, dict[str, float | None]] = {}
    stale_by_ticker: dict[str, bool] = {}
    for t in hist.columns:
        # Value each ticker at its last VALID close <= asof, not strictly at
        # the asof day's close. The month-end rebalance runs after the US
        # close so behavior there is unchanged (the last valid close IS the
        # asof-day close). But an asof day where a market simply hasn't
        # closed yet (daily-core runs intraday, holidays where one market
        # trades and the other doesn't) must not silently drop every ticker
        # of that market from the ranking — it did: on 2026-09-09 the frame
        # contained only the 11 European names that had already closed,
        # flipping the top-10 from US to European names overnight.
        vpos = valid[t].to_numpy(dtype=bool)
        last_valid = int(np.max(np.nonzero(vpos))) if vpos.any() else -1
        if last_valid < 0:
            continue
        if int(n_valid[t]) < settings.sim_monthly_min_history_days:
            continue
        price = float(c[t].iloc[last_valid])
        # Price-age gate (app/price_freshness.py): the last print must be
        # recent relative to the RANKING date. A months-old listing marks
        # stale and drops out of eligibility below; its row stays so the
        # flag is visible to consumers.
        stale = not price_freshness.is_fresh(
            hist.index[last_valid], d, settings.factor_price_max_age_days)
        rec = fundamentals.get(t)
        if not rec:
            continue

        eq = _latest_as_of(rec.get("StockholdersEquity", []), d)
        ni = _ttm_as_of(rec.get("NetIncomeLoss", []), d)
        ocf = _ttm_as_of(rec.get("NetCashProvidedByUsedInOperatingActivities", []), d)
        capex = _ttm_as_of(rec.get("PaymentsToAcquirePropertyPlantAndEquipment", []), d)
        # share counts live under either tag depending on the filer (dei vs us-gaap)
        sh = _latest_as_of(
            rec.get("CommonStockSharesOutstanding", [])
            + rec.get("EntityCommonStockSharesOutstanding", []),
            d,
        )
        if not eq or eq[1] <= 0 or ni is None or sh is None:
            continue
        stale_by_ticker[t] = bool(stale)
        roe = ni / eq[1]
        mcap = price * sh[1]
        fcf = None
        if ocf is not None and capex is not None:
            fcf = ocf + capex  # capex is canonical: a NEGATIVE cash outflow
        # Capex missing -> FCF missing. The old OCF-only fallback assumed
        # capex == 0, which OVERSTATES FCF for every non-repiler and quietly
        # inflates the value factor (audit P0-A). Ingestion
        # (fundamentals.canonical_capex) owns the negative-outflow sign.
        p_fcf = mcap / fcf if fcf is not None and fcf > 0 else None
        mom_v = float(mom.get(t, np.nan))
        rows[t] = {
            "roe": roe,
            "p_fcf": p_fcf,
            "mcap": mcap,
            "dollar_vol": float(dv20.get(t, 0) or 0),
            "mom": mom_v,
            "price": price,
            "vol": float(vol_ann.get(t, np.nan)),
        }

    if not rows:
        return None
    df = pd.DataFrame(rows).T
    # The `stale` flag is added AFTER the transpose: a bool inside every row
    # dict would make the whole frame object-dtype and break the numeric
    # ufuncs below (np.isfinite(df["mom"])). Kept as a bool column so
    # consumers can act on the price-age policy's signal directly.
    df["stale"] = pd.Series([stale_by_ticker[t] for t in df.index],
                            index=df.index, dtype=bool)
    # The mom > -0.99 gate is a raw-momentum "went to zero" sanity floor
    # (close-ratio - 1). Residual momentum is a scaled t-stat; a -2 value is
    # meaningful signal, not a busted name — apply the floor to raw only.
    raw_mom = getattr(settings, "sim_daily_core_mom_variant", "raw") != "residual"
    df["eligible"] = (
        (df["mcap"] > settings.sim_monthly_min_mcap)
        & (df["dollar_vol"] > settings.sim_monthly_min_dollar_vol)
        & np.isfinite(df["mom"])
        & ((df["mom"] > -0.99) if raw_mom else True)
        & ~df["stale"]
    )
    return df


def quality_snapshot(fund: dict[str, dict[str, list[dict]]], ticker: str,
                     asof: pd.Timestamp, close: float | None = None,
                     ) -> dict[str, float | None] | None:
    """Point-in-time quality metrics for one ticker, as of `asof`.

    Returns {"roe": float|None, "p_fcf": float|None} or None when the ticker
    has no fundamentals at all (ETFs, no-CIK listings, thin coverage — the
    caller treats None as "unknown", NOT as bad quality).

    ROE = TTM net income / latest stockholders' equity (None when equity <= 0
    or NI missing). P/FCF = market cap / TTM FCF, where market cap uses the
    `close` price (pass the USD-converted close so foreign listings compare)
    and TTM FCF = OCF + capex (capex is canonical: a NEGATIVE cash outflow,
    normalized at ingestion). Missing capex -> missing FCF — OCF alone is
    never substituted, because assuming capex == 0 overstates FCF. None
    values mean "not computable", never "bad".

    Reuses the same as-of helpers as the monthly qv-mom scoring, so the daily
    sim sees exactly the numbers the monthly strategy would compute.
    """
    rec = fund.get(ticker)
    if not rec:
        return None

    roe: float | None = None
    eq = _latest_as_of(rec.get("StockholdersEquity", []), asof)
    ni = _ttm_as_of(rec.get("NetIncomeLoss", []), asof)
    if eq and eq[1] > 0 and ni is not None:
        roe = ni / eq[1]

    p_fcf: float | None = None
    ocf = _ttm_as_of(rec.get("NetCashProvidedByUsedInOperatingActivities", []), asof)
    capex = _ttm_as_of(rec.get("PaymentsToAcquirePropertyPlantAndEquipment", []), asof)
    fcf = None
    if ocf is not None and capex is not None:
        fcf = ocf + capex  # capex is canonical: a NEGATIVE cash outflow
    # Capex missing -> FCF missing: OCF alone is NOT a conservative FCF
    # estimate (it assumes capex == 0 and overstates value). Same contract
    # as eligible_frame; the negative sign is owned by ingestion.
    if fcf is not None and fcf > 0 and close is not None and close > 0:
        sh = _latest_as_of(
            rec.get("CommonStockSharesOutstanding", [])
            + rec.get("EntityCommonStockSharesOutstanding", []),
            asof,
        )
        if sh is not None and sh[1] > 0:
            p_fcf = (close * sh[1]) / fcf

    return {"roe": roe, "p_fcf": p_fcf}


def _band_fill(order: list[str], holdings: list[str], target_n: int, hold_band: int) -> list[str]:
    """Hysteresis: held names inside the top `hold_band` of `order` keep their
    slots; remaining slots are filled by the best-ranked names not held. Held
    names that fell out of the band release their slot."""
    band = [t for t in order[:hold_band] if t in holdings]
    picked = list(band[:target_n])
    for t in order:
        if len(picked) >= target_n:
            break
        if t in picked or t in holdings:
            continue  # already kept, or held-but-outside-band -> slot released
        picked.append(t)
    return sorted(picked)


def _deploy_targets(order: list[str], holdings: list[str], target_n: int,
                    hold_band: int, *, refill: bool) -> list[str]:
    """The hysteresis-selected names ONE deployment pass may act on.

    PLANNING semantics: with ``refill`` the return is the POST-RELEASE
    membership — band members kept + open slots filled from the best-ranked
    non-held names — assuming every pending release succeeds. The live
    engine calls this BEFORE its SELL pass so replacements are planned, not
    deferred to the next day; the executor then enforces the ACTUAL
    post-SELL capacity (a failed release leaves its slot occupied, so the
    live BUY loop / backfill ``do_buy`` / replay BUY blocks each guard a
    new-name buy with ``len(shares) < target_n``). Capping the PLAN instead
    would defer every replacement after a successful release to the next
    day.

    With ``refill=False`` the held in-band names are returned WITHOUT
    fills: that is the deferred-release replay between rebuilds, where a
    slot only frees at the next rebuild, so buying a fill would create a
    transient 11th holding while an out-of-band release is still pending.

    Held names inside the top ``hold_band`` of ``order`` keep their slots
    (capped at ``target_n``). ``refill = len(holdings) < target_n`` only
    decides WHETHER fills are planned — it does not cap their count: a
    FAILED release (no usable price) leaves held-but-out-of-band names
    occupying slots while ``refill`` is True (9 held, 8 in-band, 1 pending:
    the plan still queues 2 fills). The ten-slot guarantee comes from the
    executors' hard backstops (the live BUY loop, backfill ``do_buy``,
    replay ``trade()``/BUY blocks), each of which skips a new-name buy once
    ``len(shares) >= target_n``.

    The return is the selected MEMBERSHIP in RANKING order (``order``
    position), never ``_band_fill``'s alphabetical sort: deployment funds
    best-rank first, and with limited cash the top-ranked picks must be the
    ones that get the money.

    Deployment MUST act on this set — not on ``order[:target_n]``. Buying
    the raw top-10 of the ranking while ten held names already occupy the
    slots grew the daily-core book past the target (audit P0-D).
    """
    if refill:
        members = _band_fill(order, holdings, target_n, hold_band)
    else:
        members = [t for t in order[:hold_band] if t in holdings][:target_n]
    member_set = set(members)
    return [t for t in order if t in member_set]


def residual_momentum(close: pd.DataFrame, asof: pd.Timestamp,
                      market: pd.Series | None = None,
                      lookback: int = 252, skip: int = 21) -> pd.Series:
    """Residual momentum per Blitz-Huij-Martens (2011): for each ticker, run
    the daily log returns in [asof-lookback+skip .. asof-skip] through a
    one-factor market regression, then return mean(residuals)/std(residuals)
    * sqrt(n) — momentum per unit of idiosyncratic risk. Empirically ~2x the
    Sharpe of raw 12-1 momentum with roughly half the crash exposure.

    `market` defaults to the equal-weight mean of the frame's own returns.
    Returns a Series indexed by ticker; tickers with <60 valid days are NaN.

    Note: with the default in-frame market on a SMALL universe the common
    drift absorbs into the market leg and the residuals cluster ~0 — the
    method separates names on idiosyncratic VOL (its purpose) rather than
    on raw trend, which is exactly what reduces crash exposure. For the
    production use the frame spans ~100 names, so the market leg behaves
    like a true index.
    """
    hist = close.loc[:asof]
    if len(hist) < lookback + skip:
        return pd.Series({t: float("nan") for t in close.columns}, dtype=float)
    # Window: [asof - (lookback+skip) .. asof - skip] — ends `skip` days
    # before asof (the classic 12-1 skip: excludes the most-recent month to
    # avoid the short-term reversal that contaminates momentum).
    win: pd.DataFrame = hist.iloc[-(lookback + skip + 1):-(skip + 1) if skip else None]
    rets = win.apply(lambda s: np.log(s / s.shift(1))).iloc[1:]
    if market is None:
        mkt = rets.mean(axis=1)
    else:
        m = market.reindex(rets.index).ffill()
        mkt = pd.Series(np.log(m / m.shift(1)), index=m.index).iloc[1:]
        mkt = mkt.reindex(rets.index)
    mv = mkt.to_numpy(dtype=float)
    out: dict[str, float] = {}
    for t in rets.columns:
        r = rets[t].to_numpy(dtype=float)
        mask = np.isfinite(r) & np.isfinite(mv)
        if mask.sum() < 60:
            out[t] = float("nan")
            continue
        r_ = r[mask]
        m_ = mv[mask]
        # OLS with intercept on the RAW market returns: r = a + b*m + e.
        # Momentum lives in E[r]; removing a + b*m leaves the idiosyncratic
        # drift in E[e] = a, so the score is alpha / resid-std * sqrt(n) —
        # the alpha t-statistic (the Blitz-Huij-Martens ranking).
        m_var = float(np.dot(m_ - m_.mean(), m_ - m_.mean()))
        beta = float(np.dot(r_ - r_.mean(), m_ - m_.mean()) / m_var) if m_var > 0 else 0.0
        alpha = float(r_.mean() - beta * m_.mean())
        resid = r_ - (alpha + beta * m_)
        sd = float(resid.std())
        # Ranking = alpha t-statistic: OLS forces mean(resid)=0, so score
        # the ALPHA itself per unit of residual vol, annualized by sqrt(n).
        out[t] = float(alpha / sd * np.sqrt(len(resid))) if sd > 0 else float("nan")
    return pd.Series(out)


def _qv_order(frame: pd.DataFrame, lowvol_tilt: bool | None = None) -> tuple[pd.DataFrame, list[str]]:
    """Score = pct_rank(ROE) + pct_rank(momentum) + pct_rank(1 / P-FCF if
    available) [+ pct_rank(-vol) when the low-vol tilt is on]. Explicit total
    order (unique ranking => unique portfolio):
      score desc -> higher ROE -> lower P/FCF (missing last) -> ticker asc."""
    if lowvol_tilt is None:
        lowvol_tilt = settings.sim_daily_core_lowvol_tilt
    elig = frame[frame["eligible"]].copy()
    if elig.empty:
        return elig, []
    elig["roe_rank"] = elig["roe"].rank(pct=True)
    elig["mom_rank"] = elig["mom"].rank(pct=True)
    # P/FCF: lower is better; missing values ranked last (worst)
    has_fcf = elig["p_fcf"].notna()
    elig["value_rank"] = np.nan
    if has_fcf.any():
        elig.loc[has_fcf, "value_rank"] = 1 - elig.loc[has_fcf, "p_fcf"].rank(pct=True)
    n_with = int(has_fcf.sum())
    if n_with >= max(5, int(0.05 * len(elig))):
        elig["score"] = (
            elig["roe_rank"].fillna(0)
            + elig["mom_rank"].fillna(0)
            + elig["value_rank"].fillna(0.0)
        )
    else:
        # degenerate: no value data anywhere -> quality+momentum only
        elig["score"] = elig["roe_rank"].fillna(0) + elig["mom_rank"].fillna(0)
    if lowvol_tilt:
        # pct_rank(-vol): the LOWEST-vol names rank highest. Finite-missing
        # vol stays neutral at 0 (can't penalise what we can't measure).
        vol_rank = (-elig["vol"]).rank(pct=True)
        elig["score"] = elig["score"] + vol_rank.where(np.isfinite(elig["vol"]), 0.0)

    order = (
        elig.assign(_pf=elig["p_fcf"].fillna(np.inf), _tk=elig.index)
        .sort_values(
            ["score", "roe", "_pf", "_tk"],
            ascending=[False, False, True, True],
            kind="mergesort",
        )
        .index.tolist()
    )
    return elig, order


def pick_portfolio(rebal_date: pd.Timestamp, close: pd.DataFrame, vol: pd.DataFrame,
                   fundamentals: dict[str, dict[str, list[dict]]],
                   holdings: list[str] | None = None,
                   frame: pd.DataFrame | None = None) -> tuple[list[str], pd.DataFrame | None]:
    """qv-mom-v1 picker: top target_n by score with hysteresis.
    Returns (sorted picks, eligibility frame)."""
    holdings = list(holdings or [])
    if frame is None:
        frame = eligible_frame(rebal_date, close, vol, fundamentals)
    if frame is None or not frame["eligible"].any():
        return holdings, frame
    _elig, order = _qv_order(frame)
    picks = _band_fill(order, holdings, settings.sim_monthly_target_n,
                       settings.sim_monthly_hold_band)
    return picks, frame


# ---------------------------------------------------------------------------
# Rebalance scheduling
# ---------------------------------------------------------------------------

def month_last_trading_day(now: datetime) -> datetime:
    """Last US trading day of `now`'s month (weekday, non-NYSE-holiday).

    Walks back over weekends AND NYSE holidays so a month whose last weekday
    is a holiday (e.g. 2027-05-31 Memorial Day) rebalances on the prior actual
    trading day, instead of being skipped when the scheduler's trading-day
    guard suppresses that holiday's tick."""
    from .sim import is_trading_day  # local: sim imports monthly at module load
    if now.month == 12:
        last = datetime(now.year + 1, 1, 1, tzinfo=now.tzinfo) - timedelta(days=1)
    else:
        last = datetime(now.year, now.month + 1, 1, tzinfo=now.tzinfo) - timedelta(days=1)
    while not is_trading_day(last):
        last = last - timedelta(days=1)
    return last


def is_rebalance_day(today: datetime | None = None) -> bool:
    """True when `today` is the last US trading day of its month (weekday,
    non-NYSE-holiday)."""
    now = today or datetime.now(UTC)
    return now.date() == month_last_trading_day(now).date()


def completed_month_ends(index, start=None, end=None) -> list[pd.Timestamp]:
    """Calendar month-ends actually stored in ``index``, within [start, end].

    A month-end is the scheduler's last US trading day of its calendar month
    (weekday, non-NYSE-holiday — ``month_last_trading_day``), NOT the last
    stored candle of that month: a frame ending inside a running month has
    NO month-end, and recognizing one by "a later candle exists" breaks
    future-data invariance (appending February candles flipped January-31's
    membership). The scheduled date counts only when its own candle is
    stored, so a data hole at the true month-end is never papered over by
    the last mid-month bar. Every path keys frames on these dates — the
    research backtests, the parameter sweep and both backfills.
    """
    idx = pd.DatetimeIndex(index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    idx = idx.normalize()
    if not len(idx):
        return []

    def _naive(ts) -> pd.Timestamp:
        ts = pd.Timestamp(ts)
        return ts.tz_localize(None) if ts.tzinfo is not None else ts

    lo = _naive(start) if start is not None else idx[0]
    hi = _naive(end) if end is not None else idx[-1]
    out: list[pd.Timestamp] = []
    for ym in sorted({(d.year, d.month) for d in idx}):
        me = pd.Timestamp(month_last_trading_day(
            datetime(int(ym[0]), int(ym[1]), 1))).normalize()
        if me in idx and lo <= me <= hi:
            out.append(me)
    return out


# ---------------------------------------------------------------------------
# Monthly portfolio engine
# ---------------------------------------------------------------------------

async def monthly_valuate(now: datetime | None = None) -> dict[str, Any]:
    """Value the monthly portfolio (same shape as sim.valuate).

    ``now`` is the valuation's clock: marks use closes <= now (the cycle's
    own session F).

    Additive staleness keys (price-age policy, ``app/price_freshness.py``):
    each position carries ``"stale": bool`` — True iff the mark came from the
    cost-basis fallback (no usable market mark: no quote, no FX, or a print
    older than ``settings.factor_price_max_age_days``) — and the summary
    carries ``"stale_marks": [tickers...]`` sorted. Existing keys/shapes are
    unchanged.

    EPOCH-SCOPED contributed total ("seed capital + monthly allowances"):
    ``allowance_total`` = the epoch's opening SEED (the epoch's FROZEN
    ``sim_monthly_start_cash`` — see ``_epoch_seed``) + the CURRENT epoch's
    monthly deposits. A nonzero seed must be counted or the contributed
    baseline is 0 on the seed-only stretch (undefined XIRR, phantom gain on
    the seed); the seed is constant per epoch so contribution-adjusted return
    differences cancel, and it is FROZEN: changing the setting without a reset
    does not rewrite an account that already holds the old capital. A reset
    re-seeds with the then-current setting, so the fresh book's baseline
    restarts with its own seed."""
    if now is None:
        now = _now()
    async with Session() as s:
        acc = await s.get(MonthlyAccount, 1)
        if acc is None:
            acc = MonthlyAccount(id=1, cash=settings.sim_monthly_start_cash,
                                 last_allowance_month=None,
                                 opening_seed=settings.sim_monthly_start_cash)
            s.add(acc)
            await s.commit()
        positions = (await s.scalars(select(MonthlyPosition).order_by(MonthlyPosition.ticker))).all()
        cur = await run_config.current_epoch(_ENGINE, session=s)
        deposits = (await s.scalar(
            select(func.sum(MonthlyAllowance.amount)).where(
                run_config.epoch_clause(MonthlyAllowance, cur)))) or 0.0
        seed = await _epoch_seed(s)

    positions_value = 0.0
    pos_list: list[dict[str, Any]] = []
    stale_marks: list[str] = []
    px_map = await _price_usd_map([p.ticker for p in positions], asof=now)
    for p in positions:
        price = px_map.get(p.ticker)
        # Price-age policy (app/price_freshness.py): no usable market mark
        # (no quote, no FX, or a stale print) -> the cost basis is the mark
        # and the position is FLAGGED, never silently valued at an old price.
        stale = price is None
        if stale:
            price = p.avg_cost
            stale_marks.append(p.ticker)
        value = p.shares * price
        positions_value += value
        pnl_pct = ((price - p.avg_cost) / p.avg_cost * 100) if p.avg_cost > 0 else 0.0
        pos_list.append({
            "ticker": p.ticker,
            "shares": round(p.shares, 4),
            "avg_cost": round(p.avg_cost, 4),
            "current_price": round(price, 4),
            "value": round(value, 2),
            "pnl_pct": round(pnl_pct, 2),
            "opened_at": p.opened_at.strftime("%Y-%m-%d") if p.opened_at else "",
            "stale": stale,
        })
    return {
        "cash": round(acc.cash, 2),
        "positions": pos_list,
        "positions_value": round(positions_value, 2),
        "total_equity": round(acc.cash + positions_value, 2),
        # "seed capital + monthly allowances": the epoch's external capital
        # basis (see the docstring). The pre-existing key set stays frozen —
        # the composition is documented, not expanded with new keys.
        "allowance_total": round(seed + float(deposits), 2),
        "stale_marks": sorted(stale_marks),
    }


def _fx_close_series(rows: list, asof: pd.Timestamp | None = None,
                     field: str = "close") -> pd.Series:
    """FX pair values from candle rows as a naive-UTC-indexed Series.

    ``field`` is the candle column to read, and it MUST match the column the
    caller's query selected: the CLOSE conversion reads pair closes, the OPEN
    conversion reads pair opens. Making it explicit prevents the previous
    ``_open_usd_map`` defect, where an OPEN-only query was fed to a close
    reader and every foreign pair row raised ``AttributeError``."""
    pairs: dict[pd.Timestamp, float] = {}
    for r in rows:
        ts = pd.Timestamp(r.timestamp)
        if ts.tzinfo is not None:
            ts = ts.tz_localize(None)
        if asof is None or ts <= asof:
            pairs[ts] = float(getattr(r, field))
    if not pairs:
        return pd.Series(dtype=float)
    return pd.Series(pairs, dtype=float).sort_index()


def _convert_usd(price: float, ts: pd.Timestamp, fx: pd.Series, mode: str) -> float | None:
    """Local price -> USD at the FX rate prevailing at `ts`, or None.

    Fail-closed currency contract: returns None when the source price is not
    finite and positive, when the FX series is absent/empty, when there is no
    stored rate at or before `ts` (NEVER a future rate — the lookup is
    causal), when the rate itself is not finite and positive, or when the
    rate print at/before `ts` is older than
    ``settings.factor_price_max_age_days`` RELATIVE TO THE PRICE'S OWN `ts`
    (price-age policy, ``app/price_freshness.py``). Known non-USD tickers are
    never booked at their native price as if it were USD.

    The forward carry is intentional and causal: the most recent prior valid
    rate applies until a newer print arrives — but only within the age limit.
    An ancient FX print cannot convert anything (it would book a currency
    move the market never priced), so it fails closed to None exactly like
    the "no prior rate" case.
    """
    p = float(price)
    if not math.isfinite(p) or p <= 0:
        return None
    if fx is None or fx.empty:
        return None
    rate = fx
    idx = pd.DatetimeIndex(rate.index)
    if idx.tz is not None:  # DateTime round-trips attach UTC; compare naive
        rate = rate.copy()
        rate.index = idx.tz_localize(None)
        idx = pd.DatetimeIndex(rate.index)
    at = pd.Timestamp(ts)
    if at.tzinfo is not None:
        at = at.tz_localize(None)
    # The prevailing print = the last non-NaN rate at or before `at` (the old
    # Series.asof semantics) — kept explicit so we can also read the print's
    # OWN timestamp for the age check.
    values = np.asarray(rate.to_numpy(dtype=float)).reshape(-1)
    have = ~np.isnan(values)
    if not have.any():
        return None
    rate_idx = idx[have]
    pos = int(rate_idx.searchsorted(at, side="right")) - 1
    if pos < 0:
        return None
    if not price_freshness.is_fresh(rate_idx[pos], at,
                                    settings.factor_price_max_age_days):
        return None
    rf = float(values[have][pos])
    if not math.isfinite(rf) or rf <= 0:
        return None
    return p * rf if mode == "mul" else p / rf


async def _price_usd_map(tickers: list[str],
                         asof: datetime | pd.Timestamp | None = None,
                         ) -> dict[str, float | None]:
    """{ticker: latest USD-converted close} for the given tickers.

    The strategy decides on USD-converted closes (load_frames_sync), so
    execution and valuation must book the same converted price — a raw
    local-currency close booked as USD would misstate exposure by the FX
    rate and ignore FX moves in pnl/equity. Same suffix->FX mapping.

    Fail-closed: a known non-USD ticker with no usable FX rate at/before its
    close date maps to None (never its native price); a non-finite or
    non-positive source close also maps to None. USD (no suffix) and unknown
    suffixes keep their raw close — the latter is explicitly unresolved.

    PRICE-AGE POLICY (``app/price_freshness.py``): a close older than
    ``settings.factor_price_max_age_days`` relative to the map's valuation
    clock maps to None — a stale print is not a mark, so the callers'
    existing avg_cost fallback takes over (and they flag the position stale).
    The clock is ``asof`` when given; with ``asof=None`` it is the newest row
    timestamp in the query result (today's "latest stored close" semantics
    generalized to "as of the latest stored print"). The FX rate converting a
    close must itself be within the limit of that close's timestamp, else the
    conversion fails closed (see ``_convert_usd``).

    ``asof`` bounds the valuation's clock (marks use data <= that session —
    the live engines pass the cycle's own session F so a valuation can never
    see a bar after F, rule 1/3 of the contract); None keeps the historical
    behavior (the latest stored close).
    """
    from sqlalchemy import select as _select

    wanted = set(tickers)
    pairs = sorted({pm[0] for t in tickers if (pm := fundamentals_mod._suffix_fx(t))})
    # Only the recent tail is needed for a "latest close": held tickers with
    # full history (broad universes) otherwise hydrate ~16k rows each on every
    # valuation, which dominated the status endpoints.
    since = _now().replace(tzinfo=None) - timedelta(days=_PRICE_LOOKBACK_DAYS)
    bounds = [Candle.timestamp >= since]
    ts_asof: pd.Timestamp | None = None
    if asof is not None:
        ts_asof = pd.Timestamp(asof)
        if ts_asof.tzinfo is not None:
            ts_asof = ts_asof.tz_localize(None)
        # Parenthesised comparison: the old inline conditional bound `<=` to
        # only the tz-aware branch, so a naive asof appended a bare datetime
        # to the WHERE list (SQLAlchemy ArgumentError).
        bounds.append(Candle.timestamp <= ts_asof.to_pydatetime())
    async with Session() as s:
        rows = (await s.execute(
            _select(Candle.ticker, Candle.timestamp, Candle.close)
            .where(Candle.ticker.in_(wanted | set(pairs)), *bounds)
            .order_by(Candle.timestamp))).all()
    pair_close = {p: _fx_close_series([r for r in rows if r.ticker == p])
                  for p in pairs}
    latest: dict[str, tuple[float, pd.Timestamp]] = {}
    newest: pd.Timestamp | None = None
    for r in rows:  # timestamp ASC — later assignments win
        ts = pd.Timestamp(r.timestamp)
        if ts.tzinfo is not None:
            ts = ts.tz_localize(None)
        if newest is None or ts > newest:
            newest = ts
        if r.ticker in wanted:
            latest[r.ticker] = (float(r.close), ts)
    # The valuation clock: the caller's asof, or (asof=None) the newest print
    # in the result — "latest stored close" is only meaningful relative to
    # something, and the market's own newest print IS that clock.
    clock = ts_asof if ts_asof is not None else newest
    max_age = settings.factor_price_max_age_days
    out: dict[str, float | None] = {}
    for t in tickers:
        px_ts = latest.get(t)
        if px_ts is None:
            out[t] = None
            continue
        price, ts = px_ts
        if not math.isfinite(price) or price <= 0:
            out[t] = None  # not a usable price — never a 0/NaN/inf mark
            continue
        if clock is not None and not price_freshness.is_fresh(ts, clock, max_age):
            # Stale print: not a mark. None hands the callers to their
            # existing cost-basis fallback (avg_cost), visibly flagged.
            out[t] = None
            continue
        pm = fundamentals_mod._suffix_fx(t)
        if pm:
            price = _convert_usd(price, ts, pair_close.get(pm[0], pd.Series(dtype=float)), pm[1])
        out[t] = price
    return out


async def _price(ticker: str) -> float | None:
    """Latest USD-converted close for `ticker` from the candle cache."""
    return (await _price_usd_map([ticker])).get(ticker)


async def _open_usd_map(tickers: list[str], d: datetime | pd.Timestamp,
                        ) -> dict[str, float | None]:
    """{ticker: USD-converted OPEN of the session exactly on `d`} — the fill
    price of the session fill contract (app/session.py rule 2).

    Strict exact-date stored bar per ticker: a ticker without a stored bar
    on `d` maps to None — a missing quote is not an executable price, and
    the executor must give that leg NO fill (it is retried at the next
    session). NEVER a stale or forward-filled close. The FX conversion
    reuses _price_usd_map's mapping and `_convert_usd` unchanged, so the
    decision (USD closes) and the execution (USD opens) book the same
    currency — but it reads the pair's OPEN (the rate prevailing at the fill
    session's open), never the pair's close, and it fails closed (None) when
    a known non-USD ticker has no usable prior rate. Non-finite/non-positive
    opens also map to None.
    """
    from sqlalchemy import select as _select

    wanted = set(tickers)
    ts_d = pd.Timestamp(d)
    if ts_d.tzinfo is not None:
        ts_d = ts_d.tz_localize(None)
    ts_d = ts_d.normalize()
    pairs = sorted({pm[0] for t in tickers if (pm := fundamentals_mod._suffix_fx(t))})
    since = (ts_d - timedelta(days=_PRICE_LOOKBACK_DAYS)).to_pydatetime()
    async with Session() as s:
        rows = (await s.execute(
            _select(Candle.ticker, Candle.timestamp, Candle.open)
            .where(Candle.ticker.in_(wanted | set(pairs)),
                   Candle.timestamp >= since,
                   Candle.timestamp <= ts_d.to_pydatetime())
            .order_by(Candle.timestamp))).all()
    pair_open = {p: _fx_close_series([r for r in rows if r.ticker == p],
                                     field="open")
                 for p in pairs}
    at_d: dict[str, tuple[float, pd.Timestamp]] = {}
    for r in rows:  # exact date match only — no ffill, no stale close
        if r.ticker not in wanted or r.open is None:
            continue
        ts = pd.Timestamp(r.timestamp)
        if ts.tzinfo is not None:
            ts = ts.tz_localize(None)
        if ts == ts_d:
            at_d[r.ticker] = (float(r.open), ts)
    out: dict[str, float | None] = {}
    for t in tickers:
        hit = at_d.get(t)
        if hit is None:
            out[t] = None
            continue
        px, ts = hit
        if not math.isfinite(px) or px <= 0:
            out[t] = None  # not an executable open — never a 0/NaN/inf fill
            continue
        pm = fundamentals_mod._suffix_fx(t)
        if pm:
            px = _convert_usd(px, ts, pair_open.get(pm[0], pd.Series(dtype=float)), pm[1])
        out[t] = px
    return out


async def deposit_allowance(*, mode: str | None = None, config_id: int | None = None,
                            epoch: int | None = None) -> dict[str, Any]:
    """Deposit the monthly allowance if a new operator-local month has begun.

    Called from the daily scheduler and /api/monthly/status to fund the
    account at the START of the month (mirroring sim.deposit_allowance), and
    from run_rebalance() as a no-op fallback when the daily path missed a
    month. Deposits at the start of the month — not at the month-end
    rebalance — keeps the cumulative "contributed" figure in lockstep with the
    sim portfolio's so the two equity curves compare like-for-like; the
    month-end rebalance then invests whatever cash has accumulated (including
    this deposit). Idempotent per (month, epoch).

    Forward PAPER record: stamped with the logical run's frozen stamp when one
    is threaded (run_rebalance), else with a freshly resolved paper stamp
    (the status endpoint calls this standalone).

    EPOCH-AWARE GUARD: the month marker is current STATE and a reset clears
    it, so it cannot prove the month was funded in THIS epoch (the closed
    epoch's row of the same month is preserved on purpose). Before inserting,
    the guard verifies no allowance row exists for the month under the current
    epoch — a marker that lagged its row heals instead of duplicating, while a
    post-reset same-month deposit writes its own new-epoch row.
    """
    month = _current_month()
    async with _state_write_lock:
        return await _deposit_allowance_locked(month, mode=mode,
                                               config_id=config_id, epoch=epoch)


async def _deposit_allowance_locked(month: str, *, mode: str | None,
                                    config_id: int | None,
                                    epoch: int | None) -> dict[str, Any]:
    stamp = await _resolve_paper_stamp(mode=mode, config_id=config_id, epoch=epoch)
    async with Session() as s:
        if not await _stamp_is_current(s, stamp):
            acc = await s.get(MonthlyAccount, 1)
            logger.warning("Monthly deposit refused: stale epoch stamp %s",
                           stamp["epoch"])
            return {"deposited": False, "month": month,
                    "cash": acc.cash if acc is not None else None,
                    "refused": "stale epoch stamp"}
        acc = await s.get(MonthlyAccount, 1)
        if acc is None:
            acc = MonthlyAccount(id=1, cash=settings.sim_monthly_start_cash,
                                 last_allowance_month=None,
                                 opening_seed=settings.sim_monthly_start_cash)
            s.add(acc)
        if acc.last_allowance_month == month:
            return {"deposited": False, "month": month, "cash": acc.cash}
        exists = await s.scalar(
            select(func.count()).select_from(MonthlyAllowance).where(
                MonthlyAllowance.month == month,
                run_config.epoch_clause(MonthlyAllowance, stamp["epoch"])))
        if exists:
            acc.last_allowance_month = month
            await s.commit()
            logger.info("Monthly allowance for %s already deposited (epoch %s)",
                        month, stamp["epoch"])
            return {"deposited": False, "month": month, "cash": acc.cash}
        acc.cash += settings.sim_monthly_contribution
        acc.last_allowance_month = month
        s.add(MonthlyAllowance(amount=settings.sim_monthly_contribution, month=month,
                               mode=stamp["mode"], config_id=stamp["config_id"],
                               epoch=stamp["epoch"]))
        await s.commit()
        logger.info("Monthly allowance deposited: %.2f for %s -> cash %.2f",
                    settings.sim_monthly_contribution, month, acc.cash)
        return {"deposited": True, "amount": settings.sim_monthly_contribution,
                "month": month, "cash": acc.cash}


async def _exec_buy(ticker: str, price: float, budget: float, reason: str, *,
                    mode: str | None = None, config_id: int | None = None,
                    epoch: int | None = None) -> dict | None:
    async with _state_write_lock:
        stamp = await _resolve_paper_stamp(mode=mode, config_id=config_id, epoch=epoch)
        return await _exec_buy_locked(ticker, price, budget, reason, stamp)


async def _exec_buy_locked(ticker: str, price: float, budget: float, reason: str,
                           stamp: dict) -> dict | None:
    async with Session() as s:
        if not await _stamp_is_current(s, stamp):
            logger.warning("Monthly BUY %s refused: stale epoch stamp", ticker)
            return None
        acc = await s.get(MonthlyAccount, 1)
        if acc is None:
            return None
        # Sizing + the exact-cash guard are the shared zero-fee primitive
        # (app/factor_fills.py): 4dp-floored shares, no fee, and a leg whose
        # booked cost exceeds the EXACT cash is dropped — never resized.
        fill = factor_fills.buy_fill(acc.cash, budget, price)
        if fill is None:
            return None
        acc.cash = fill.cash_after
        pos = await s.scalar(select(MonthlyPosition).where(MonthlyPosition.ticker == ticker))
        if pos:
            total_shares = pos.shares + fill.shares
            pos.avg_cost = (pos.shares * pos.avg_cost + fill.notional) / total_shares
            pos.shares = total_shares
        else:
            s.add(MonthlyPosition(ticker=ticker, shares=fill.shares, avg_cost=price))
        s.add(MonthlyTrade(ticker=ticker, side="BUY", shares=fill.shares, price=price,
                           cash_after=acc.cash, reason=reason, mode=stamp["mode"],
                           config_id=stamp["config_id"], epoch=stamp["epoch"]))
        await s.commit()
        return {"ticker": ticker, "side": "BUY", "shares": fill.shares, "price": price,
                "cost": round(fill.notional, 2), "reason": reason}


async def _exec_sell(ticker: str, price: float, reason: str, *,
                     mode: str | None = None, config_id: int | None = None,
                     epoch: int | None = None) -> dict | None:
    async with _state_write_lock:
        stamp = await _resolve_paper_stamp(mode=mode, config_id=config_id, epoch=epoch)
        return await _exec_sell_locked(ticker, price, reason, stamp)


async def _exec_sell_locked(ticker: str, price: float, reason: str,
                            stamp: dict) -> dict | None:
    async with Session() as s:
        if not await _stamp_is_current(s, stamp):
            logger.warning("Monthly SELL %s refused: stale epoch stamp", ticker)
            return None
        pos = await s.scalar(select(MonthlyPosition).where(MonthlyPosition.ticker == ticker))
        if pos is None:
            return None
        acc = await s.get(MonthlyAccount, 1)
        if acc is None:
            return None
        # The SELL goes through the shared primitive BEFORE any state change:
        # an invalid/zero/NaN quote yields no fill, so a position is never
        # deleted for $0 (or negative) proceeds and no trade is booked.
        fill = factor_fills.sell_fill(acc.cash, pos.shares, price)
        if fill is None:
            return None
        acc.cash = fill.cash_after
        s.add(MonthlyTrade(ticker=ticker, side="SELL", shares=fill.shares, price=price,
                           cash_after=acc.cash, reason=reason, mode=stamp["mode"],
                           config_id=stamp["config_id"], epoch=stamp["epoch"]))
        await s.delete(pos)
        await s.commit()
        return {"ticker": ticker, "side": "SELL", "shares": fill.shares,
                "price": price, "proceeds": round(fill.notional, 2), "reason": reason}


def _snapshot_json(frame: pd.DataFrame | None) -> str:
    if frame is None:
        return "{}"
    cols = ["roe", "p_fcf", "mcap", "dollar_vol", "mom", "price", "vol", "eligible"]
    return frame[cols].to_json(orient="index") or "{}"


async def refresh_holdings() -> dict[str, Any]:
    """Refresh candle data (+ FX pairs) for the currently held tickers so the
    daily valuation/snapshot uses current prices. No rebalance, no deposits.
    Mirrors the /api/monthly/refresh endpoint logic."""
    from .market import refresh
    async with Session() as s:
        held = [p.ticker for p in
                (await s.scalars(select(MonthlyPosition))).all()]
    pairs = sorted({pm[0] for t in held if (pm := fundamentals_mod._suffix_fx(t))})
    refreshed: list[str] = []
    errors: list[str] = []
    for t in held + pairs:
        try:
            await refresh(t, "2y")
            refreshed.append(t)
        except Exception as e:
            errors.append(f"{t}: {e}")
    return {"refreshed": refreshed, "errors": errors}


async def take_snapshot() -> dict[str, Any]:
    """Record one equity-curve snapshot of the current valuation.

    Called daily by the scheduler so the Monthly equity curve moves every day
    (previously snapshots were only written after month-end rebalances, so the
    curve sat flat between rebalances). The month-end rebalance also snapshots
    post-execution, so on rebalance days the curve shows both the daily mark
    and the post-trade state.

    Skipped while a rebalance is running: the rebalance writes its own
    authoritative post-trade snapshot, and a daily mark racing it mid-execution
    (cash moved, buys incomplete) would record a distorted equity point.

    Uses a lock-guarded body rather than a locked()-then-valuate check:
    checking `locked()` and valuating outside the lock races a manual
    rebalance that starts in between, and the mark would then read
    mid-execution state (cash already moved, buys incomplete). Holding the
    lock for the whole snapshot makes it atomic with respect to the
    rebalance instead. (asyncio.Lock.acquire()'s uncontended fast path sets
    the flag synchronously, so no other task can slip in between the
    `locked()` check and the acquire — no await happens between them.)
    """
    if _rebalance_lock.locked():
        return {"skipped": True, "reason": "rebalance in progress"}
    async with _rebalance_lock, _state_write_lock:
        stamp = await _resolve_paper_stamp()
        async with Session() as s:
            if not await _stamp_is_current(s, stamp):
                logger.warning("Monthly snapshot refused: stale epoch stamp")
                return {"snapshotted": False, "refused": "stale epoch stamp"}
        post = await monthly_valuate()
        async with Session() as s:
            if not await _stamp_is_current(s, stamp):
                logger.warning("Monthly snapshot refused: stale epoch stamp")
                return {"snapshotted": False, "refused": "stale epoch stamp"}
            s.add(MonthlySnapshot(cash=post["cash"], positions_value=post["positions_value"],
                                  total_equity=post["total_equity"],
                                  allowance_total=post["allowance_total"],
                                  mode=stamp["mode"], config_id=stamp["config_id"],
                                  epoch=stamp["epoch"]))
            await s.commit()
        logger.info("Monthly daily snapshot: equity %.2f (cash %.2f, positions %.2f)",
                    post["total_equity"], post["cash"], post["positions_value"])
        return {"snapshotted": True, "total_equity": post["total_equity"]}


async def refresh_data(tickers: list[str]) -> tuple[list[str], dict[str, str]]:
    """Refresh candles (+ FX pairs) and fundamentals for the universe.

    Candles go through the bounded-concurrency ``refresh_many`` batch with the
    same freshness window as the daily sim / daily-core, so a rebalance reuses
    the nightly universe prefetch (and any recent fetch) instead of re-pulling
    the whole ~500-name universe from Yahoo one ticker at a time. The deep
    (10y) candle cache is populated separately by the screener's deep-load and
    kept in the DB; this only tops up the ~2y the live ranking reads.

    Fundamentals source priority: SEC EDGAR for US filers (real filed dates,
    history to ~2009), yfinance fallback for CIK-less listings (ETFs, European
    exchanges). Same split as the stockstrat research pipeline."""
    from . import edgar
    from .market import refresh_many
    wanted = list(tickers) + sorted({pm[0] for t in tickers if (pm := fundamentals_mod._suffix_fx(t))})
    # Yahoo only (not the metered provider): the universe-sized batch path.
    _, refresh_errors = await refresh_many(
        wanted, "2y", max_age_seconds=settings.market_fresh_seconds)
    fund_status: dict[str, str]
    if settings.sim_monthly_fundamentals_source == "edgar":
        fund_status = await edgar.refresh_universe_mixed(tickers=tickers)
    else:
        fund_status = await fundamentals_mod.refresh_universe(tickers=tickers)
    return refresh_errors, fund_status


def _prev_trading_day(today: datetime) -> datetime | None:
    """The previous NYSE trading day strictly before ``today`` (calendar walk,
    capped at two weeks so an unusable calendar can never spin). Naive-UTC:
    the caller's clock may carry tzinfo, but every downstream comparison is
    against naive stored timestamps."""
    from .sim import is_trading_day
    ts = pd.Timestamp(today)
    if ts.tzinfo is not None:
        ts = ts.tz_localize(None)
    day = (ts.normalize() - pd.Timedelta(days=1)).to_pydatetime()
    for _ in range(14):
        if is_trading_day(day):
            return day
        day = day - timedelta(days=1)
    return None


def _candidate_month_end(s) -> datetime | None:
    """ME <= S (the previous completed session) by the session calendar."""
    ts = pd.Timestamp(s)
    if ts.tzinfo is not None:
        ts = ts.tz_localize(None)
    d = ts.normalize()
    me_this = pd.Timestamp(
        month_last_trading_day(datetime(d.year, d.month, 1))).normalize()
    if me_this <= d:
        return me_this.to_pydatetime()
    prev_y, prev_m = (d.year - 1, 12) if d.month == 1 else (d.year, d.month - 1)
    return pd.Timestamp(month_last_trading_day(
        datetime(prev_y, prev_m, 1))).to_pydatetime()


def _split_csv(col: str | None) -> list[str]:
    return [t for t in (col or "").split(",") if t]


async def _candle_stored_on(tickers: list[str], d) -> bool:
    """True when ANY of ``tickers`` has a stored bar exactly on ``d`` — the
    live twin of the backfill's ``me in idx`` completion test (a data hole at
    the decision day is never papered over)."""
    ts = pd.Timestamp(d)
    if ts.tzinfo is not None:
        ts = ts.tz_localize(None)
    async with Session() as s:
        hit = await s.scalar(
            select(func.count()).select_from(Candle)
            .where(Candle.ticker.in_(list(tickers)),
                   Candle.timestamp == ts.to_pydatetime()))
    return bool(hit)


async def _pending_rebuild(now: datetime) -> tuple[str, pd.Timestamp] | None:
    """The pending monthly rebuild: ``(decision month key, month-end date)``
    when the month-end before ``now`` has a decision that has not been fully
    executed yet — the first session after the previous month-end, with
    catch-up for any later session while it is still pending. None when
    there is no completed month-end behind us, its candle is missing (a data
    hole — never papered over), or the rebuild is already complete."""
    s = _prev_trading_day(now)
    if s is None:
        return None
    me = _candidate_month_end(s)
    if me is None:
        return None
    tickers = universe_tickers(_strategy_universe())
    if not await _candle_stored_on(tickers, me):
        return None
    key = pd.Timestamp(me).strftime("%Y-%m")
    async with Session() as s2:
        acc = await s2.get(MonthlyAccount, 1)
        cur = await run_config.current_epoch(_ENGINE, session=s2)
        # EPOCH-SCOPED decision lookup: a closed epoch's audit row for the same
        # month is preserved history, not evidence that THIS epoch's rebuild
        # ran (the rebuilt UNIQUE(rebal_month, epoch) lets both coexist).
        row = await s2.scalar(
            select(MonthlyRebalance).where(
                MonthlyRebalance.rebal_month == key,
                run_config.epoch_clause(MonthlyRebalance, cur)))
        held = {p.ticker for p in await s2.scalars(select(MonthlyPosition))}
    if acc is not None and acc.last_rebalance_month == key:
        return None  # this decision's legs are all filled (or terminally unfillable)
    if row is not None and held == set(_split_csv(row.picked)):
        return None  # acted on; membership complete (marker write lagged)
    return key, pd.Timestamp(me).normalize()


async def run_rebalance(force: bool = False,
                        now: datetime | None = None) -> dict[str, Any]:
    """One monthly rebuild leg pass under the session fill contract
    (adapted from app/session.py):

      * the DECISION is the completed month-end's qv-mom ranking — the
        eligibility frame is taken as of the last completed month-end <= S
        (the previous completed session), never a newer frame;
      * the FILLS execute at the current session's strict exact-date OPEN
        per ticker (``_open_usd_map``) — a missing quote gives that leg no
        fill and it is retried at the next session;
      * the rebuild normally lands on the FIRST session of the month (one
        session of cash drag); ``force`` re-runs it any day;
      * audit rows keep the decision trading day (``rebal_date`` = the
        month-end) distinct from the fill day (each trade's ``created_at``).

    Clock discipline (session fill contract rules 1/3): the RANKING and the
    ORDER SIZING both read marks known at the previous completed session S.
    F's close never sizes an order — it only reports the end-of-session
    valuation/snapshot, after the fills.

    ``now`` is injectable so the parity tests can drive time; None = now."""
    # Non-blocking acquire: the idempotence gate below runs before the
    # multi-minute data refresh, so a queued second invocation would pass
    # the gate too and double-deposit / duplicate trades.
    if _rebalance_lock.locked():
        return {"skipped": True, "reason": "already running"}
    async with _rebalance_lock:
        return await _run_rebalance_locked(force, now)


async def _run_rebalance_locked(force: bool, now: datetime | None) -> dict[str, Any]:
    if now is None:
        now = _now()
    f_ts = pd.Timestamp(now)
    f_date = (f_ts.tz_localize(None) if f_ts.tzinfo is not None else f_ts).normalize()
    # S — the previous COMPLETED session — is the clock for every decision
    # input (app/session.py rules 1/3): the ranking is taken through the
    # month-end <= S, and the marks that SIZE the orders must also be S-known.
    # Resolve it here for BOTH paths: only the force branch used to bind it,
    # which left the normal _pending_rebuild path with no S to size against.
    s_prev = _prev_trading_day(now)
    if not force:
        pending = await _pending_rebuild(now)
        if pending is None:
            return {"skipped": True, "reason": "no pending month-end rebuild"}
        month, me_ts = pending
    else:
        me = _candidate_month_end(s_prev) if s_prev is not None else None
        if me is None:
            return {"skipped": True, "reason": "no completed month-end yet"}
        month = pd.Timestamp(me).strftime("%Y-%m")
        me_ts = pd.Timestamp(me).normalize()
    # Both branches gate on S existing (no S -> no decision), so an executable
    # pass always has one; never substitute the month-end (S can be later).
    assert s_prev is not None
    s_ts = pd.Timestamp(s_prev).normalize()

    # ONE paper stamp for the whole rebuild pass: the deposit, the audit row,
    # the trades and the post-trade snapshot all carry the same configuration
    # (resolving per leg could stamp two configs into one logical run). The
    # RunExecution is deferred to the stable input point below.
    cfg = current_config()
    stamp = await _paper_stamp(cfg, record=False)

    # Same-month provenance policy: the audit row for this month (this epoch)
    # records the configuration of the pass that DECIDED it. A pass running
    # under a different configuration — or an un-stamped legacy row, whose
    # provenance cannot be proven — must NOT rewrite that content under the
    # new stamp (silent relabelling of old history) and must not append a
    # second row for the month (UNIQUE(rebal_month, epoch)). Refuse BEFORE the
    # deposit and any decision/ledger write; a reset (new epoch) re-decides
    # the month under the current configuration. A BACKFILL row is the one
    # exception: it is synthetic and replaceable by policy (the live paper
    # pass supersedes it in the same epoch and the row is re-stamped paper).
    async with Session() as s:
        existing_reb = await s.scalar(
            select(MonthlyRebalance).where(
                MonthlyRebalance.rebal_month == month,
                run_config.epoch_clause(MonthlyRebalance, stamp["epoch"])))
    if (existing_reb is not None and existing_reb.mode != "backfill"
            and existing_reb.config_id != stamp["config_id"]):
        return {"skipped": True, "rebal_month": month,
                "reason": ("a rebuild row for this month already belongs to a "
                           "different run configuration — reset (new epoch) to "
                           "re-decide it under the current configuration")}

    allowance = await deposit_allowance(**stamp)
    tickers = universe_tickers(_strategy_universe())
    # The ranking is shared with daily-core: resolve the runtime variant here
    # too, rather than depending on whoever set settings last.
    settings.sim_daily_core_mom_variant = _strategy_variant()
    refresh_errors, fund_status = await refresh_data(tickers)

    close, vol = await load_frames(tickers, None, start=_live_history_start())
    fund = await fundamentals_mod.load_fundamentals(tickers)
    # The decision frame is the COMPLETED month-end's (rule 1): eligible_frame
    # truncates every candle to sessions <= the decision day, so F's bar —
    # and anything after the month-end — can never inform it.
    frame = await asyncio.to_thread(eligible_frame, me_ts, close, vol, fund)
    if frame is None or not bool(frame["eligible"].any()):
        # Terminal for the month: mark the decision attempted so ordinary
        # sessions stop retrying a dead frame (matches the backfill's
        # live_months filter — ineligible months are funded, not rebuilt).
        async with Session() as s:
            acc = await s.get(MonthlyAccount, 1)
            if acc is not None:
                acc.last_rebalance_month = month
            await s.commit()
        return {"skipped": True, "reason": "no eligible frame",
                "rebal_month": month, "decision_date": str(me_ts.date()),
                "refresh_errors": refresh_errors, "fundamentals": fund_status}

    async with Session() as s:
        held_rows = (await s.scalars(select(MonthlyPosition))).all()
    held_before = sorted(p.ticker for p in held_rows)
    picks, _ = await asyncio.to_thread(
        pick_portfolio, me_ts, close, vol, fund, held_before, frame)
    # Fill prices: the session's STRICT exact-date OPEN, USD-converted (never
    # a stale/forward-filled close). Moved UP from the SELL loop: the query is
    # read-only, and the LLM sell layer's type-B overlay needs the executable
    # fill prices to identify its candidates (names without a quote stay
    # pending as today — not consulted).
    fill_map = await _open_usd_map(sorted(set(picks) | set(held_before)), f_date)

    # Mid-run configuration guard: the awaits above (refresh, frame math) give
    # an operator time to change the universe/variant. This pass decided under
    # `cfg`; executing or stamping under a changed configuration would
    # misattribute the run, so refuse BEFORE the first decision write and let
    # the next pass run entirely under the new configuration.
    if _config_changed(cfg):
        return {"skipped": True,
                "reason": "run configuration changed mid-run — retry under the new config"}

    # Admitted: the frame, decision and config gates all passed and the pass is
    # about to write. Stable input point — the refresh above wrote the rows the
    # decision reads — so ONE provenance row is recorded here (never for a
    # read poll or a refused/terminal pass).
    await _record_run(stamp, cfg)

    # --- LLM sell layer, type B: consult per band-drop candidate AFTER
    # pick_portfolio, BEFORE the SELL loop. CONFIRM -> as today; DEFER -> the
    # name is retained one more month and the new name it displaced is not
    # bought (capacity stays target_n). Fail-safe: knob off / no backend /
    # any failure -> executed_picks == picks (bit-identical OFF).
    from . import selllayer
    executed_picks, sell_detail = await selllayer.overlay_type_b(
        held_before=held_before, picks=picks, fill_map=fill_map, frame=frame,
        close=close, fund=fund, members=tickers, me_ts=me_ts, f_date=f_date,
        target_n=int(settings.sim_monthly_target_n),
        hold_band=int(settings.sim_monthly_hold_band))
    picks = executed_picks
    if sell_detail is not None:
        logger.info("Monthly sell layer B: engine picks %s -> executed %s",
                    sell_detail["engine_picks"], sell_detail["executed_picks"])

    async with Session() as s:
        cur = await run_config.current_epoch(_ENGINE, session=s)
        # EPOCH-SCOPED: a closed epoch's audit row for the same month is
        # preserved history (rebuilt UNIQUE(rebal_month, epoch)), never an
        # update target — updating it would rewrite a preserved record.
        existing = await s.scalar(
            select(MonthlyRebalance).where(
                MonthlyRebalance.rebal_month == month,
                run_config.epoch_clause(MonthlyRebalance, cur)))
        reb = existing or MonthlyRebalance(rebal_month=month, rebal_date=me_ts.to_pydatetime())
        reb.held_before = ",".join(held_before)
        # "picked" stays the EXECUTED picks (what the book actually holds after
        # the pass); the sell layer's audit JSON carries the engine picks AND
        # the executed picks as additive JSON keys so the difference is
        # auditable (NULL when the overlay did not consult anyone).
        reb.picked = ",".join(picks)
        reb.n_new = len(set(picks) - set(held_before))
        reb.snapshot = _snapshot_json(frame)
        reb.sell_layer = (json.dumps(sell_detail, sort_keys=True)
                          if sell_detail is not None else None)
        if existing is None:
            reb.mode = stamp["mode"]
            reb.config_id = stamp["config_id"]
            reb.epoch = stamp["epoch"]
            s.add(reb)
        elif existing.mode == "backfill":
            # Synthetic row superseded by this live pass in the same epoch:
            # the content becomes the paper decision, so the row is re-stamped
            # honestly as paper (a backfill row is replaceable by policy).
            existing.mode = stamp["mode"]
            existing.config_id = stamp["config_id"]
            existing.epoch = stamp["epoch"]
        # else: a same-config paper catch-up. Refresh the CONTENT
        # (held_before/picks/n_new/snapshot) but NEVER rewrite the FIRST
        # pass's provenance stamp — the row records the run that made the
        # decision (the config check above guarantees it is the same config).
        await s.commit()

    trades: list[dict] = []
    # (fill_map was computed above, before the sell layer's type-B overlay —
    # the same strict exact-date OPEN map. A ticker without an F bar gets no
    # fill; that leg is retried at the next session.)
    # SELLs first: held names no longer picked (frees cash for the BUYs).
    pending_fill = False
    for t in held_before:
        if t in picks:
            continue
        px = fill_map.get(t)
        if not px:
            pending_fill = True  # no executable quote this session
            continue
        r = await _exec_sell(t, px,
                             f"monthly rebuild: out of top band (fill {f_date.date()})",
                             **stamp)
        if r:
            trades.append(r)
    # BUYs: equal weight 1/N of total equity (top-ups included). The ORDER
    # SIZING marks are S-known closes — never F's close (rules 1/3: decision
    # inputs are data <= S; app/session.py). The CASH is real: post-deposit
    # and post-F-open-SELL, exactly what the account holds at this point. F's
    # close only reports the end-of-session state (below).
    valuation = await monthly_valuate(now=s_ts)
    weight = valuation["total_equity"] / max(len(picks), 1)
    for t in picks:
        px = fill_map.get(t)
        if not px:
            if t not in {p["ticker"] for p in valuation["positions"]}:
                pending_fill = True  # an unfilled OPEN leg is pending work
            continue
        held_pos = next((p for p in valuation["positions"] if p["ticker"] == t), None)
        current_value = held_pos["value"] if held_pos else 0.0
        budget = weight - current_value
        if budget < 1:
            continue
        r = await _exec_buy(t, px, budget,
                            f"monthly rebuild: top-{settings.sim_monthly_target_n} qv-mom pick",
                            **stamp)
        if r:
            trades.append(r)

    # End-of-session reporting at F's close (rule 3) — never a decision input.
    post = await monthly_valuate(now=now)
    async with Session() as s:
        s.add(MonthlySnapshot(cash=post["cash"], positions_value=post["positions_value"],
                              total_equity=post["total_equity"],
                              allowance_total=post["allowance_total"],
                              mode=stamp["mode"], config_id=stamp["config_id"],
                              epoch=stamp["epoch"]))
        held_after = {p.ticker for p in await s.scalars(select(MonthlyPosition))}
        acc = await s.get(MonthlyAccount, 1)
        complete = (not pending_fill) and held_after == set(picks)
        if acc is not None and complete:
            acc.last_rebalance_month = month
        await s.commit()

    return {"rebalanced": True, "month": month, "allowance": allowance,
            "decision_date": str(me_ts.date()), "picks": picks,
            # the sell layer's engine picks (before the type-B overlay) when it
            # consulted anyone, else None — the audit row carries both lists
            "engine_picks": (sell_detail or {}).get("engine_picks"),
            "held_before": held_before, "trades": trades,
            "complete": complete,
            "valuation": post, "refresh_errors": refresh_errors,
            "fundamentals": fund_status}


async def run_monthly_cycle(now: datetime | None = None) -> dict[str, Any]:
    """Scheduler entry: fills the pending monthly rebuild.

    The trigger is the session fill contract's, not ``is_rebalance_day``:
    the fill session is the FIRST session after the completed month-end (the
    month-end tick is the decision, the next open is the execution), and any
    later session fills it too while the rebuild is still pending (catch-up).
    Legs that cannot fill retry at the following sessions; a newer month-end
    ranking supersedes a pending one.

    Concurrency: a DEV backfill holding the lock must DELAY the month-end
    rebalance, not skip it — a missed month has no catch-up. The free path
    delegates to run_rebalance() (asyncio.Lock's uncontended acquire sets the
    flag synchronously, so no other task can slip in between this check and
    run_rebalance's own acquire). The busy path waits, bounded by
    ``_TICK_LOCK_WAIT_SECONDS``, and holds the lock through the rebalance
    body. Only a holder wedged past the bound skips (error-logged). The
    manual /api/monthly/run path calls run_rebalance() directly and keeps
    its fail-fast answer.
    """
    if not settings.sim_monthly_enabled:
        return {"skipped": True, "reason": "monthly portfolio disabled"}
    # --- LLM sell layer, type C (app/selllayer.py): a NEW nightly step that
    # runs EVERY trading night (not only pending-rebuild nights), BEFORE the
    # rebuild step (declared order): first the FILL step (unfilled C-EXIT
    # ledger rows for still-held names sell at THIS session's strict exact-date
    # open), then the SCAN step (held names whose newly available PIT
    # fundamentals break the predeclared thesis are consulted). Fail-safe:
    # any failure is logged and never breaks the rebuild below.
    c_result: dict[str, Any] = {"skipped": True, "reason": "sell layer off"}
    if settings.sim_monthly_llm_sells:
        try:
            from . import selllayer
            c_result = await selllayer.run_type_c_step(now=now)
        except Exception as e:  # noqa: BLE001 — never break the rebuild
            logger.error("Monthly sell-layer type-C step failed: %s", e, exc_info=True)
            c_result = {"error": f"{type(e).__name__}: {e}"}
    if not _rebalance_lock.locked():
        result = await run_rebalance(now=now)
    else:
        logger.warning("Monthly scheduler: tick delayed — waiting up to %ds for the "
                       "rebalance lock (held by a backfill or manual run)",
                       _TICK_LOCK_WAIT_SECONDS)
        try:
            async with asyncio.timeout(_TICK_LOCK_WAIT_SECONDS):
                await _rebalance_lock.acquire()
        except TimeoutError:
            logger.error("Monthly scheduler: rebalance lock still held after %ds — "
                         "skipping this tick", _TICK_LOCK_WAIT_SECONDS)
            result = {"skipped": True, "reason": "already running"}
        else:
            try:
                result = await _run_rebalance_locked(False, now)
            finally:
                _rebalance_lock.release()
    # The type-C result is logged (never attached to the pinned result dict).
    if c_result.get("fills") or c_result.get("scans"):
        logger.info("Monthly sell layer C: %d fill(s), %d scan(s) — %s",
                    len(c_result.get("fills", [])), len(c_result.get("scans", [])),
                    c_result.get("as_of"))
    return result


# ---------------------------------------------------------------------------
# Backfill (synthetic history, same contract as daily_core.backfill)
# ---------------------------------------------------------------------------

async def _sync_start_date() -> str | None:
    """Earliest daily-sim snapshot date — the monthly backfill's default
    start so the three equity curves cover the same window (same rule as
    daily_core._sync_start_date; the daily-sim curve is the longest-running
    live one, so it anchors the shared window).

    Deliberately NOT epoch-clipped: a fresh backfill may start inside a CLOSED
    epoch's window to model a shared historical experiment. It only READS that
    history; the replay's own rows are stamped into the current epoch, so
    nothing in a closed epoch is rewritten."""
    from .db import SimSnapshot
    async with Session() as s:
        d = await s.scalar(select(func.min(SimSnapshot.created_at)))
    return d.strftime("%Y-%m-%d") if d else None


async def backfill(start: str | None = None,
                   *, preload: BackfillPreload | None = None) -> dict[str, Any]:
    """Replay the monthly qv-mom strategy over historical data and REPLACE
    the portfolio state with the replay's end state.

    Wipes account/positions/trades/allowances/snapshots/rebalances, then
    walks EVERY stored session from `start` to today as the live engine's
    cycle stream (the session fill contract, app/session.py):

      * the allowance is deposited on each calendar month's FIRST stored
        session, unconditionally (the live engine funds the account monthly
        even when no ranking exists — only the rebuild needs a frame);
      * the pending rebuild decision is recomputed every session — the
        newest completed month-end <= S (the previous stored session) whose
        own candle is stored — so a newer month-end supersedes a
        still-pending older one, and a data hole at a decision day is never
        papered over;
      * the decision's legs execute at the CURRENT session's strict
        exact-date OPEN per ticker (never a stale or forward-filled close):
        fills land on the FIRST session after the decision month-end (one
        session of cash drag), and a leg without an executable quote this
        session stays pending and is retried at the following sessions;
      * the audit row is written at the FIRST pass with ``rebal_date`` = the
        decision month-end — distinct from each trade's ``created_at`` (the
        fill session);
      * ORDER SIZING marks the previous completed stored session S's closes
        (rules 1/3): F's own close never sizes a leg, it only values the
        end-of-session snapshot (which the replay records at F's close).

    No fees: the replay mirrors the LIVE engine, whose
    `_exec_buy`/`_exec_sell` charge no commission (the paper portfolios are
    fee-free by design — see docs/brokers.md), so a backfill's end state
    matches what the live engine would hold. Materialized as real
    trade/rebalance/snapshot rows the UI can render.

    CAPITAL BASIS CAVEAT: the replay's baseline is cash = 0 plus its own
    monthly deposits, and its snapshot rows carry that explicit contributed
    total (which excludes a configured ``start_cash`` seed). The live
    valuation still reports the epoch's configured seed as contributed
    capital (see ``monthly_valuate``); with the default ``start_cash = 0`` the
    two agree exactly.

    ``start`` semantics (mirrors daily_core.backfill):
      - explicit "YYYY-MM-DD": replay from that day
      - None (default): synched with the other sims — starts on the earliest
        snapshot date of the daily sim / daily-core portfolios so all three
        equity curves cover the same window
      - "all": the full stored history (2017+, first month with an eligible
        frame) — the long-view replay

    Paper-portfolio convenience, not a live track record. Point-in-time
    discipline: a decision on month-end ME only sees fundamentals public by
    then (same EDGAR-first fact store as the backtest) and candles <= ME;
    fills see only the strict same-session bar. A month only rebalances on
    its ACTUAL US session calendar month-end (never the last stored candle
    of a still-running month).

    FORWARD-RECORD PROTECTION (the audit's "do not overwrite a forward
    record"): if the CURRENT epoch already holds a forward paper row, the
    replay is REFUSED before any mutation. A replay regenerates the book's
    ORIGIN; a "bounded re-backfill" (regenerate only the pre-paper synthetic
    segment) was considered and REJECTED because the seam books a spurious
    external flow (``allowance_total`` jumps between the two segments'
    baselines) and the paper book's origin would no longer match the synthetic
    run preceding it. Reset first — a reset preserves the old history as a
    closed epoch.

    The replay's rows are stamped ``mode="backfill"`` in the current epoch; a
    RE-backfill replaces only its own rows (mode NULL or ``backfill`` in the
    same epoch) — never a paper row and never a closed epoch's history. The
    refusal is FAIL-CLOSED: ``has_paper_rows`` treats a legacy NULL-mode row
    as a protected forward record (its mode is unknown, so it cannot be proven
    synthetic) — a legacy book must be reset once before it can be backfilled.
    The NULL in the wipe predicate is defensive only (rows written by
    pre-stamping backfills in an epoch that passed the guard).
    """
    from sqlalchemy import delete as sa_delete
    from .db import (MonthlyAccount as Acc, MonthlyPosition as Pos,
                     MonthlyTrade as Tr, MonthlyAllowance as Al,
                     MonthlySnapshot as Sn, MonthlyRebalance as Rb)

    async with _rebalance_lock:
        # --- forward-record guard: BEFORE any mutation ---------------------
        cur = await run_config.current_epoch(_ENGINE, session_factory=Session)
        if await run_config.has_paper_rows(_ENGINE, cur, session_factory=Session):
            return {"ok": False,
                    "error": "a forward paper record exists for this epoch — "
                             "backfill would overwrite the live book's origin; "
                             "reset first (reset preserves history as a closed epoch)"}
        # The replay's OWN configuration snapshot — never a paper stamp (a
        # backfill must not create a paper config row while it prepares).
        cfg = current_config()
        stamp = {
            "mode": "backfill",
            "config_id": await run_config.get_or_create(
                _ENGINE, "backfill", cfg, session_factory=Session),
            "epoch": cur,
        }
        # Recorded ONLY for a backfill that will actually run (the refusal
        # guard above already returned). ``asof`` = now is the run's DATA upper
        # bound; per-decision point-in-time discipline lives inside the replay.
        if start is None:
            # synched with the other sims: the earliest daily-sim snapshot
            start = await _sync_start_date()
        elif start == "all":
            start = None  # full stored history
        start_note = start or "first eligible month"

        settings.sim_daily_core_mom_variant = _strategy_variant()
        if preload is not None:
            tickers, fund = preload.tickers, preload.fund
            open_px, close, vol = preload.open, preload.close, preload.vol
        else:
            tickers = universe_tickers(_strategy_universe())
            fund = await fundamentals_mod.load_fundamentals(tickers)
            load_start = ((pd.Timestamp(start) - timedelta(days=_BACKFILL_WARMUP_DAYS)).to_pydatetime()
                          if start else None)
            open_px, close, vol = await load_ohlcv_frames(tickers, None, start=load_start)
        if not fund:
            return {"ok": False, "error": "no fundamentals loaded"}
        if close.empty:
            return {"ok": False, "error": "no candle data"}

        full_idx = pd.DatetimeIndex(close.index).normalize()
        idx = full_idx
        if start:
            idx = idx[idx >= pd.Timestamp(start)]
        if idx.empty:
            return {"ok": False, "error": f"no trading days since {start_note}"}
        full_set = set(full_idx)

        # COMPLETED month-ends only, by the US session calendar (same rule
        # as the research backtests — completed_month_ends): a frame ending
        # inside a running month (e.g. the current partial month) has no
        # month-end, so the replay no longer fabricates a rebuild on its
        # last candle.
        months = completed_month_ends(close.index, idx[0], idx[-1])

        # eligibility frames per month-end (plus the month-end before the
        # window so a mid-month start ranks against the prior frame) — the
        # prior month-end is keyed off the WINDOW START's month,
        # independent of the rebalance list.
        prior_month_ends: list[pd.Timestamp] = []
        d0 = idx[0]
        prev_y, prev_m = (d0.year - 1, 12) if d0.month == 1 else (d0.year, d0.month - 1)
        pme_sched = pd.Timestamp(month_last_trading_day(datetime(prev_y, prev_m, 1)))
        prev_idx = full_idx[full_idx <= pme_sched]
        if len(prev_idx):
            prior_month_ends.append(prev_idx[-1])
        shared = preload.frames if preload is not None else {}
        frame_cache: dict[pd.Timestamp, pd.DataFrame | None] = {}
        for m in sorted(set(prior_month_ends) | set(months)):
            if m in shared:
                frame_cache[m] = shared[m]
                continue
            frame_cache[m] = await asyncio.to_thread(eligible_frame, m, close, vol, fund)
            shared[m] = frame_cache[m]
        live_months = [m for m in months
                       if (f := frame_cache.get(m)) is not None
                       and bool(f["eligible"].any())]
        if not live_months:
            return {"ok": False, "error": "no month with eligible names"}

        # contribution timing: first trading day of EVERY calendar month in the
        # replayed window (not just the eligible ones — the live engine deposits
        # monthly even when no ranking exists; only the rebuild needs a frame).
        contrib_days: dict[str, pd.Timestamp] = {}
        for m in sorted({(d.year, d.month) for d in idx}):
            key = f"{m[0]}-{m[1]:02d}"
            month_days = idx[(idx.year == m[0]) & (idx.month == m[1])]
            if len(month_days):
                contrib_days[key] = month_days[0]
        # valuation closes: forward-filled so missing candles carry the last
        # known close (never value a held position at 0) — MARKS ONLY. The
        # fill prices read the STRICT open frame (never ffilled).
        close_val = close.ffill()

        def px_of(t: str, d: pd.Timestamp) -> float | None:
            return px_at(close_val, d, t)

        def fill_px_of(t: str, d: pd.Timestamp) -> float | None:
            # Session fill contract rule 2 (app/session.py): the strict
            # exact-date OPEN per ticker — a stored bar on `d` or NOTHING (a
            # missing quote is not an executable price; the leg is retried
            # at the next session, never filled from a stale close).
            return px_at(open_px, d, t)

        # --- in-memory session-by-session replay (the live engine's cycle
        # stream) ---
        # Real share/cash bookkeeping (not the optimize backtest's
        # aggregate-value shortcut) so the trade log can carry actual
        # shares/prices: a pending decision's legs trade at the fill
        # session's open, marks value the book at that session's close.
        # Between passes the portfolio is untouched (the monthly engine
        # trades only when a month-end decision is pending).
        shares: dict[str, float] = {}
        cost_basis: dict[str, float] = {}   # true average cost, as _exec_buy books it
        opened_at: dict[str, str] = {}      # ticker -> actual first-buy date
        cash = 0.0
        contributed = 0.0
        deposited_months: set[str] = set()  # months the replay actually funded
        marker: str | None = None           # the replay's last_rebalance_month
        audit: dict[str, dict[str, Any]] = {}  # month key -> first-pass record
        rebalance_rows: list[tuple[str, str, str, str, int]] = []
        trade_rows: list[tuple[str, str, str, float, float, float]] = []
        snaps: list[tuple[str, float, float]] = []

        locs = full_idx.get_indexer(idx)  # S = the stored session before each walk day
        for i, d in enumerate(idx):
            # 1) deposit (unconditional — see contrib_days above).
            key = f"{d.year}-{d.month:02d}"
            if contrib_days.get(key) == d:
                cash += settings.sim_monthly_contribution
                contributed += settings.sim_monthly_contribution
                deposited_months.add(key)

            # 2) pending decision, recomputed every session (the live
            #    _pending_rebuild state machine): the newest month-end <= S
            #    with its own candle stored. A newer month-end supersedes a
            #    still-pending older one; a hole at the decision day means
            #    NO decision this session (never papered over).
            li = locs[i]
            s_prev = full_idx[li - 1] if li > 0 else None
            pending: tuple[str, pd.Timestamp] | None = None
            if s_prev is not None:
                me = _candidate_month_end(s_prev.to_pydatetime())
                me_ts = pd.Timestamp(me).normalize() if me is not None else None
                if me_ts is not None and me_ts in full_set:
                    mkey = me_ts.strftime("%Y-%m")
                    if marker != mkey:
                        row = audit.get(mkey)
                        if not (row is not None
                                and set(shares) == set(row["picked"])):
                            pending = (mkey, me_ts)  # supersede / first sight

            # 3) the decision's leg pass at THIS session's strict opens.
            if pending is not None:
                mkey, me_ts = pending
                frame = frame_cache.get(me_ts)
                if frame is None and me_ts not in frame_cache:
                    # A restricted `start` can surface a month-end before the
                    # precomputed set: compute it on demand — the live engine
                    # would rank it from the same candles.
                    frame = await asyncio.to_thread(eligible_frame, me_ts, close, vol, fund)
                    frame_cache[me_ts] = frame
                    shared[me_ts] = frame
                if frame is None or not bool(frame["eligible"].any()):
                    # Dead frame: terminal for the month (the live engine
                    # marks the decision attempted so ordinary sessions stop
                    # retrying) — no audit row, no trades.
                    marker = mkey
                else:
                    held_before = sorted(shares)
                    picks, _ = await asyncio.to_thread(
                        pick_portfolio, me_ts, close, vol, fund, held_before, frame)
                    if mkey not in audit:  # audit row: the FIRST pass only
                        audit[mkey] = {"picked": picks}
                        rebalance_rows.append(
                            (mkey, me_ts.strftime("%Y-%m-%d"),
                             ",".join(held_before), ",".join(picks),
                             len(set(picks) - set(held_before))))
                    fill_map = {t: fill_px_of(t, d)
                                for t in sorted(set(picks) | set(held_before))}
                    pending_fill = False
                    # SELLs first: held names no longer picked (frees cash
                    # for the BUYs). No fallback price: a leg without a bar
                    # this session is retried next session.
                    for t in held_before:
                        if t in picks:
                            continue
                        px = fill_map.get(t)
                        if px is None or px <= 0:
                            pending_fill = True  # no executable quote this session
                            continue
                        # Shared zero-fee SELL primitive: it confirms the leg
                        # is executable BEFORE the position is popped, so an
                        # invalid quote/share count can never silently remove
                        # a holding (the leg simply stays pending instead).
                        sell = factor_fills.sell_fill(cash, shares[t], px)
                        if sell is None:
                            pending_fill = True  # not executable — retry later
                            continue
                        shares.pop(t)
                        cash = sell.cash_after
                        cost_basis.pop(t, None)
                        opened_at.pop(t, None)
                        trade_rows.append((d.strftime("%Y-%m-%d"), "SELL", t,
                                           sell.notional, sell.shares, px))
                    # BUYs: equal weight 1/N of total equity; the leg fills
                    # at F's open (rule 2). Mirrors the live engine exactly:
                    # ONE pre-trade valuation whose MARKS are S-known closes
                    # (px at the previous completed stored session `s_prev` —
                    # rules 1/3, never F's close), with the same avg-cost
                    # fallback, per-position values rounded to the cents it
                    # reports and total equity rounded once. F's close below
                    # values the end-of-session snapshot only.
                    pos_value = 0.0
                    pos_values: dict[str, float] = {}
                    for t in sorted(shares):  # live reads positions ticker-ordered
                        price = px_of(t, s_prev)
                        if price is None:
                            price = cost_basis.get(t, 0.0)  # live: avg_cost fallback
                        v = shares[t] * price
                        pos_value += v
                        pos_values[t] = round(v, 2)
                    weight = round(cash + pos_value, 2) / max(len(picks), 1)
                    for t in picks:
                        px = fill_map.get(t)
                        if px is None or px <= 0:
                            if t not in shares:
                                pending_fill = True  # an unfilled OPEN leg is pending work
                            continue
                        budget = weight - pos_values.get(t, 0.0)
                        if budget < 1:
                            continue
                        # Shared zero-fee BUY primitive (mirrors live
                        # _exec_buy): 4dp floor, sub-minimum and exact-cash
                        # guards included — a leg the account cannot afford
                        # is skipped, never resized down to the cash.
                        buy = factor_fills.buy_fill(cash, budget, px)
                        if buy is None:
                            continue  # live _exec_buy booked no fill
                        if t in shares:
                            new_sh = shares[t] + buy.shares
                            cost_basis[t] = (shares[t] * cost_basis[t] + buy.notional) / new_sh
                            shares[t] = new_sh
                        else:
                            shares[t] = buy.shares
                            cost_basis[t] = px
                            opened_at[t] = d.strftime("%Y-%m-%d")
                        cash = buy.cash_after
                        trade_rows.append((d.strftime("%Y-%m-%d"), "BUY", t,
                                           buy.notional, buy.shares, px))
                    # The pass is complete when no leg waited on a quote and
                    # the membership matches the plan — the marker is set
                    # only then; an incomplete pass stays pending (its legs
                    # retry at the following sessions' opens).
                    if not pending_fill and set(shares) == set(picks):
                        marker = mkey

            # 4) the day's snapshot at marks (deposit → pass → snapshot).
            eq = cash + sum(sh * (px_of(t, d) or 0.0) for t, sh in shares.items())
            snaps.append((d.strftime("%Y-%m-%d"), eq, contributed))

        # --- persist the end state: ONE transaction under the state lock ---
        # The forward-record guard, the deletions, the inserts, the account
        # re-seed/markers, the RunExecution and the commit are ONE transaction
        # serialized with standalone writers by ``_state_write_lock``: a
        # writer either committed BEFORE it (the guard refuses) or runs AFTER
        # the lock releases (its paper row lands on the new book and is never
        # erased) — it can never slip into the gap. The long replay/loader work
        # above stays OUTSIDE the lock; only the final commit block is held.
        replay_months = sorted(deposited_months)
        current_month = _current_month()
        async with _state_write_lock:
            async with Session() as s:
                if await run_config.has_paper_rows(_ENGINE, cur, session=s):
                    await s.rollback()
                    return {"ok": False,
                            "error": "a forward paper record exists for this epoch — "
                                     "backfill would overwrite the live book's origin; "
                                     "reset first (reset preserves history as a closed epoch)"}
                acc = await s.get(Acc, 1)
                live_allowance_month = None if acc is None else acc.last_allowance_month
                live_rebalance_month = None if acc is None else acc.last_rebalance_month
                # The live engine may already have deposited the CURRENT month
                # while the replay window has no candle for it yet — re-create
                # that contribution (same rule as daily_core.backfill), inside
                # the transaction so the basis cannot be observed half-applied.
                if (live_allowance_month == current_month
                        and current_month not in replay_months):
                    cash += settings.sim_monthly_contribution
                    contributed += settings.sim_monthly_contribution
                    replay_months = sorted(set(replay_months) | {current_month})
                # Only this epoch's replaceable rows go: mode backfill, or
                # legacy NULL-mode rows (cannot be proven forward). A paper row
                # would have been refused above; a CLOSED epoch's history is
                # never touched. Positions are current STATE, cleared outright.
                for tbl in (Tr, Al, Sn, Rb):
                    await s.execute(sa_delete(tbl).where(
                        run_config.epoch_clause(tbl, cur),
                        or_(tbl.mode.is_(None), tbl.mode == "backfill")))
                await s.execute(sa_delete(Pos))
                if acc is None:
                    acc = Acc(id=1, opening_seed=0.0)
                    s.add(acc)
                # converge to the live engine's state shape: cash = replay
                # cash, positions with their true average cost and first-buy
                # date — the same bookkeeping the live _exec_buy performs.
                acc.cash = cash
                # The replay's capital basis is cash = 0 + its own deposits.
                acc.opening_seed = 0.0
                # Appended in THIS transaction, after the in-tx forward guard
                # and the account seed: a refusal/failure records no execution.
                await _record_run(stamp, cfg, session=s)
                for t, sh in shares.items():
                    opened = opened_at.get(t) or idx[-1].strftime("%Y-%m-%d")
                    s.add(Pos(ticker=t, shares=sh, avg_cost=cost_basis.get(t, 0.0),
                              opened_at=pd.Timestamp(
                                  f"{opened} 16:00:00+00:00").to_pydatetime()))
                for td, side, t, notional, sh, pr in trade_rows:
                    s.add(Tr(ticker=t, side=side, shares=round(sh, 6),
                             price=round(pr, 6), cash_after=0.0,
                             reason=f"backfill {side.lower()} ${notional:,.0f}",
                             mode=stamp["mode"], config_id=stamp["config_id"],
                             epoch=stamp["epoch"],
                             created_at=pd.Timestamp(
                                 f"{td} 16:00:00+00:00").to_pydatetime()))
                for m_key in replay_months:
                    s.add(Al(amount=settings.sim_monthly_contribution, month=m_key,
                             mode=stamp["mode"], config_id=stamp["config_id"],
                             epoch=stamp["epoch"]))
                for m_key, td, held, picked, n_new in rebalance_rows:
                    s.add(Rb(rebal_month=m_key,
                             rebal_date=pd.Timestamp(
                                 f"{td} 16:00:00+00:00").to_pydatetime(),
                             held_before=held, picked=picked, n_new=n_new,
                             snapshot="", mode=stamp["mode"],
                             config_id=stamp["config_id"], epoch=stamp["epoch"]))
                # Never let the allowance marker lag the newest inserted row,
                # or the next live deposit would duplicate that month.
                last_month = max(live_allowance_month or "",
                                 replay_months[-1] if replay_months else "")
                acc.last_allowance_month = last_month or None
                # The rebalance marker: the replay's own completion state,
                # never regressed below what the live engine had recorded —
                # _pending_rebuild gates on it, so a month the replay could
                # not complete stays unmarked and the live engine retries it.
                final_marker = max(live_rebalance_month or "", marker or "")
                acc.last_rebalance_month = final_marker or None
                for sd, eq, contrib_at_day in snaps:
                    s.add(Sn(cash=0.0, positions_value=eq, total_equity=eq,
                             allowance_total=round(contrib_at_day, 2),
                             mode=stamp["mode"], config_id=stamp["config_id"],
                             epoch=stamp["epoch"],
                             created_at=pd.Timestamp(
                                 f"{sd} 16:00:00+00:00").to_pydatetime()))
                await s.commit()

        return {"ok": True, "months": len(live_months), "start": str(idx[0].date()),
                "end": str(idx[-1].date()), "requested_start": start_note,
                "contributed": round(contributed, 2),
                "final_equity": round(snaps[-1][1], 2) if snaps else 0.0,
                "trades": len(trade_rows), "rebalances": len(rebalance_rows),
                "snapshots": len(snaps)}


# ---------------------------------------------------------------------------
# Read APIs
# ---------------------------------------------------------------------------

async def get_trades(limit: int = 100) -> list[dict]:
    """Raw trade-log export (all epochs, newest first).

    Deliberately NOT epoch-clipped: the audit/history listing cannot feed a
    decision or the curve (this engine has no decision-feeding trade read).
    """
    async with Session() as s:
        rows = (await s.scalars(select(MonthlyTrade)
                                .order_by(MonthlyTrade.created_at.desc()).limit(limit))).all()
    return [{"ticker": r.ticker, "side": r.side, "shares": r.shares, "price": r.price,
             "cash_after": round(r.cash_after, 2), "reason": r.reason,
             "date": r.created_at.isoformat()} for r in rows]


async def get_equity_curve(limit: int = 365) -> list[dict]:
    """The CURRENT epoch's equity curve, oldest first.

    Epoch-scoped at the SQL level and the limit applies AFTER the clip: a
    reset restarts the curve, so closed-epoch rows stay in the table
    (audit/history) but never leak into the chart. Each row carries its
    provenance additively (``mode``/``config_id``/``epoch``; JSON-safe None on
    legacy rows).
    """
    async with Session() as s:
        cur = await run_config.current_epoch(_ENGINE, session=s)
        rows = (await s.scalars(select(MonthlySnapshot)
                                .where(run_config.epoch_clause(MonthlySnapshot, cur))
                                .order_by(MonthlySnapshot.created_at.desc())
                                .limit(limit))).all()
    return [{"date": r.created_at.strftime("%Y-%m-%d"), "cash": r.cash,
             "positions_value": r.positions_value, "total_equity": r.total_equity,
             "allowance_total": r.allowance_total,
             "mode": r.mode, "config_id": r.config_id, "epoch": r.epoch}
            for r in reversed(rows)]


async def get_rebalances(limit: int = 24) -> list[dict]:
    """Raw rebalance-audit export (all epochs, newest first).

    Deliberately NOT epoch-clipped (audit/history); the DECISION-feeding
    rebalance reads (``_pending_rebuild`` and the writer's existing-row
    lookup) DO get the epoch clause.
    """
    async with Session() as s:
        rows = (await s.scalars(select(MonthlyRebalance)
                                .order_by(MonthlyRebalance.created_at.desc()).limit(limit))).all()
    return [{"month": r.rebal_month, "date": r.rebal_date.strftime("%Y-%m-%d"),
             "held_before": r.held_before.split(",") if r.held_before else [],
             "picked": r.picked.split(",") if r.picked else [],
             "n_new": r.n_new} for r in rows]


async def get_allowances() -> list[dict]:
    """The CURRENT epoch's deposit ledger, newest month first (epoch-scoped:
    no read may count the lifetime after a reset; closed epochs' rows stay in
    the table for audit). ``mode``/``config_id``/``epoch`` are additive keys.
    """
    async with Session() as s:
        cur = await run_config.current_epoch(_ENGINE, session=s)
        rows = (await s.scalars(
            select(MonthlyAllowance)
            .where(run_config.epoch_clause(MonthlyAllowance, cur))
            .order_by(MonthlyAllowance.month.desc()))).all()
    return [{"amount": r.amount, "month": r.month, "mode": r.mode,
             "config_id": r.config_id, "epoch": r.epoch} for r in rows]


async def reset_monthly() -> dict[str, Any]:
    """Close the current epoch and re-seed the book — HISTORY IS PRESERVED.

    The money ledger survives: trades, allowances, snapshots and rebalances
    are NOT deleted. One ``RunEpoch`` boundary records the reset (with a note
    counting what was preserved), the current-epoch positions are cleared and
    the account is re-seeded (cash = start, both month markers None), all in
    ONE transaction under the engine lock — a crash can never leave a boundary
    without the re-seed or a half-cleared book. The equity curve restarts at
    the new epoch (``get_equity_curve`` clips to it); preserved rows keep
    their epoch and remain auditable.

    CAPITAL BASIS: the new epoch is an INDEPENDENT paper capital baseline. Its
    contribution basis is the configured ``start_cash`` seed (EXTERNAL capital
    the paper book is opened with — not a real transfer) plus the epoch's own
    monthly deposits; a deposit for a month the closed epoch already funded is
    a NEW allocation in the new ledger (the UNIQUE(month, epoch) lets both
    rows coexist), never a move of the old epoch's money. Epoch metrics never
    count the lifetime. Market data, fundamentals, the screener and the
    universe selection are left intact (only portfolio *state* is cleared).

    Returns the new epoch and the preserved row counts additively; existing
    keys are unchanged.
    """
    from sqlalchemy import delete as sa_delete
    from .db import (MonthlyAccount as Acc, MonthlyPosition as Pos,
                     MonthlyTrade as Tr, MonthlyAllowance as Al,
                     MonthlySnapshot as Sn, MonthlyRebalance as Rb)

    async with _rebalance_lock, _state_write_lock:
        async with Session() as s:
            # ONE transaction: boundary + counts + positions + re-seed. The
            # helpers take the caller's session and never commit behind us.
            cur = await run_config.current_epoch(_ENGINE, session=s)
            preserved = {
                "trades": (await s.scalar(
                    select(func.count()).select_from(Tr)
                    .where(run_config.epoch_clause(Tr, cur)))) or 0,
                "snapshots": (await s.scalar(
                    select(func.count()).select_from(Sn)
                    .where(run_config.epoch_clause(Sn, cur)))) or 0,
                "allowances": (await s.scalar(
                    select(func.count()).select_from(Al)
                    .where(run_config.epoch_clause(Al, cur)))) or 0,
                "rebalances": (await s.scalar(
                    select(func.count()).select_from(Rb)
                    .where(run_config.epoch_clause(Rb, cur)))) or 0,
            }
            note = (f"reset — {preserved['trades']} trades, "
                    f"{preserved['snapshots']} snapshots, "
                    f"{preserved['allowances']} allowances, "
                    f"{preserved['rebalances']} rebalances preserved "
                    f"as epoch {cur}")
            new_epoch = await run_config.record_boundary(_ENGINE, note, session=s)
            await s.execute(sa_delete(Pos))
            acc = await s.get(Acc, 1)
            if acc is None:
                acc = Acc(id=1, cash=settings.sim_monthly_start_cash,
                          last_allowance_month=None,
                          opening_seed=settings.sim_monthly_start_cash)
                s.add(acc)
            else:
                acc.cash = settings.sim_monthly_start_cash
                acc.last_allowance_month = None
                acc.last_rebalance_month = None
                # The new epoch's ACTUAL opening capital (external seed).
                acc.opening_seed = settings.sim_monthly_start_cash
            await s.commit()
    logger.info("Monthly reset: cash=%.2f epoch=%s (preserved %s)",
                settings.sim_monthly_start_cash, new_epoch, preserved)
    return {"ok": True, "cash": settings.sim_monthly_start_cash,
            "epoch": new_epoch, "preserved": preserved}
