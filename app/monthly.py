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
import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta, UTC
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from sqlalchemy import func, select

from . import fundamentals as fundamentals_mod
from .config import settings
from .db import (Candle, MonthlyAccount, MonthlyAllowance, MonthlyPosition,
                 MonthlyRebalance, MonthlySnapshot, MonthlyTrade, Session)
from .screener import tickers as universe_tickers

logger = logging.getLogger("trade_sentinel.monthly")


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


# Serializes rebalances: the month-idempotence check runs before the
# multi-minute refresh_data, so two overlapping invocations (scheduler +
# manual UI run) would both pass the gate and double-deposit the allowance /
# duplicate trades. Mirrors sim._run_cycle_lock.
_rebalance_lock: asyncio.Lock = asyncio.Lock()


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
    dropped; untouched columns (USD tickers, pairs without a listed rate)
    pass through unchanged."""
    out = frame.copy()
    for t in list(out.columns):
        suffix = t[t.rfind("."):] if "." in t else None
        pm = fundamentals_mod.SUFFIX_FX.get(suffix) if suffix else None
        if not pm or pm[0] not in out.columns:
            continue
        rate = out[pm[0]].reindex(out.index).ffill()
        if pm[1] == "mul":
            out[t] = out[t] * rate
        else:
            out[t] = out[t] / rate
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
    price, vol (annualized realized), eligible."""
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
    # The mom > -0.99 gate is a raw-momentum "went to zero" sanity floor
    # (close-ratio - 1). Residual momentum is a scaled t-stat; a -2 value is
    # meaningful signal, not a busted name — apply the floor to raw only.
    raw_mom = getattr(settings, "sim_daily_core_mom_variant", "raw") != "residual"
    df["eligible"] = (
        (df["mcap"] > settings.sim_monthly_min_mcap)
        & (df["dollar_vol"] > settings.sim_monthly_min_dollar_vol)
        & np.isfinite(df["mom"])
        & ((df["mom"] > -0.99) if raw_mom else True)
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
    own session F)."""
    if now is None:
        now = _now()
    async with Session() as s:
        acc = await s.get(MonthlyAccount, 1)
        if acc is None:
            acc = MonthlyAccount(id=1, cash=settings.sim_monthly_start_cash,
                                 last_allowance_month=None)
            s.add(acc)
            await s.commit()
        positions = (await s.scalars(select(MonthlyPosition).order_by(MonthlyPosition.ticker))).all()
        allowance_total = (await s.scalar(select(func.sum(MonthlyAllowance.amount)))) or 0.0

    positions_value = 0.0
    pos_list: list[dict[str, Any]] = []
    px_map = await _price_usd_map([p.ticker for p in positions], asof=now)
    for p in positions:
        price = px_map.get(p.ticker)
        if price is None:
            price = p.avg_cost
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
        })
    return {
        "cash": round(acc.cash, 2),
        "positions": pos_list,
        "positions_value": round(positions_value, 2),
        "total_equity": round(acc.cash + positions_value, 2),
        "allowance_total": round(float(allowance_total), 2),
    }


def _fx_close_series(rows: list, asof: pd.Timestamp | None = None) -> pd.Series:
    """FX pair closes from candle rows as a naive-UTC-indexed Series."""
    pairs: dict[pd.Timestamp, float] = {}
    for r in rows:
        ts = pd.Timestamp(r.timestamp)
        if ts.tzinfo is not None:
            ts = ts.tz_localize(None)
        if asof is None or ts <= asof:
            pairs[ts] = float(r.close)
    if not pairs:
        return pd.Series(dtype=float)
    return pd.Series(pairs, dtype=float).sort_index()


def _convert_usd(price: float, ts: pd.Timestamp, fx: pd.Series, mode: str) -> float:
    """Local close -> USD at the FX rate prevailing at `ts`."""
    if fx is None or fx.empty:
        return price  # no FX data — documented fallback: assume USD
    rate = fx
    idx = pd.DatetimeIndex(rate.index)
    if idx.tz is not None:  # DateTime round-trips attach UTC; compare naive
        rate = rate.copy()
        rate.index = idx.tz_localize(None)
    r = rate.asof(ts)
    if r is None:
        return price
    rf = float(np.asarray(r).reshape(-1)[0])
    if np.isnan(rf) or rf == 0:
        return price
    return price * rf if mode == "mul" else price / rf


async def _price_usd_map(tickers: list[str],
                         asof: datetime | pd.Timestamp | None = None,
                         ) -> dict[str, float | None]:
    """{ticker: latest USD-converted close} for the given tickers.

    The strategy decides on USD-converted closes (load_frames_sync), so
    execution and valuation must book the same converted price — a raw
    local-currency close booked as USD would misstate exposure by the FX
    rate and ignore FX moves in pnl/equity. Same suffix->FX mapping.

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
    if asof is not None:
        bounds.append(Candle.timestamp <= pd.Timestamp(asof).tz_localize(None)
                      if pd.Timestamp(asof).tzinfo is not None
                      else pd.Timestamp(asof).to_pydatetime())
    async with Session() as s:
        rows = (await s.execute(
            _select(Candle.ticker, Candle.timestamp, Candle.close)
            .where(Candle.ticker.in_(wanted | set(pairs)), *bounds)
            .order_by(Candle.timestamp))).all()
    pair_close = {p: _fx_close_series([r for r in rows if r.ticker == p])
                  for p in pairs}
    latest: dict[str, tuple[float, pd.Timestamp]] = {}
    for r in rows:  # timestamp ASC — later assignments win
        if r.ticker in wanted:
            ts = pd.Timestamp(r.timestamp)
            if ts.tzinfo is not None:
                ts = ts.tz_localize(None)
            latest[r.ticker] = (float(r.close), ts)
    out: dict[str, float | None] = {}
    for t in tickers:
        px_ts = latest.get(t)
        if px_ts is None:
            out[t] = None
            continue
        price, ts = px_ts
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
    currency."""
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
    pair_open = {p: _fx_close_series([r for r in rows if r.ticker == p])
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
        pm = fundamentals_mod._suffix_fx(t)
        if pm:
            px = _convert_usd(px, ts, pair_open.get(pm[0], pd.Series(dtype=float)), pm[1])
        out[t] = px
    return out


async def deposit_allowance() -> dict[str, Any]:
    """Deposit the monthly allowance if a new operator-local month has begun.

    Called from the daily scheduler and /api/monthly/status to fund the
    account at the START of the month (mirroring sim.deposit_allowance), and
    from run_rebalance() as a no-op fallback when the daily path missed a
    month. Deposits at the start of the month — not at the month-end
    rebalance — keeps the cumulative "contributed" figure in lockstep with the
    sim portfolio's so the two equity curves compare like-for-like; the
    month-end rebalance then invests whatever cash has accumulated (including
    this deposit). Idempotent per operator-local month.
    """
    month = _current_month()
    async with Session() as s:
        acc = await s.get(MonthlyAccount, 1)
        if acc is None:
            acc = MonthlyAccount(id=1, cash=settings.sim_monthly_start_cash, last_allowance_month=None)
            s.add(acc)
        if acc.last_allowance_month == month:
            return {"deposited": False, "month": month, "cash": acc.cash}
        acc.cash += settings.sim_monthly_contribution
        acc.last_allowance_month = month
        s.add(MonthlyAllowance(amount=settings.sim_monthly_contribution, month=month))
        await s.commit()
        logger.info("Monthly allowance deposited: %.2f for %s -> cash %.2f",
                    settings.sim_monthly_contribution, month, acc.cash)
        return {"deposited": True, "amount": settings.sim_monthly_contribution,
                "month": month, "cash": acc.cash}


def share_size(budget: float, price: float) -> float:
    """Fractional-share size toward `budget` at `price`, floored to 4dp.

    The ONE sizing formula for the factor engines: the live ``_exec_buy`` and
    both persisted replays must size identical budgets identically, or the
    golden parity (tests/test_cadence_parity.py) breaks. Returns 0.0 when the
    leg is not executable (no positive price / sub-dollar budget)."""
    if price <= 0 or budget < 1:
        return 0.0
    return math.floor(budget / price * 10000) / 10000


async def _exec_buy(ticker: str, price: float, budget: float, reason: str) -> dict | None:
    shares = share_size(budget, price)
    if shares < 0.0001:
        return None
    cost = shares * price
    async with Session() as s:
        acc = await s.get(MonthlyAccount, 1)
        if acc is None or acc.cash < cost:
            return None
        acc.cash -= cost
        pos = await s.scalar(select(MonthlyPosition).where(MonthlyPosition.ticker == ticker))
        if pos:
            total_shares = pos.shares + shares
            pos.avg_cost = (pos.shares * pos.avg_cost + cost) / total_shares
            pos.shares = total_shares
        else:
            s.add(MonthlyPosition(ticker=ticker, shares=shares, avg_cost=price))
        s.add(MonthlyTrade(ticker=ticker, side="BUY", shares=shares, price=price,
                           cash_after=acc.cash, reason=reason))
        await s.commit()
        return {"ticker": ticker, "side": "BUY", "shares": shares, "price": price,
                "cost": round(cost, 2), "reason": reason}


async def _exec_sell(ticker: str, price: float, reason: str) -> dict | None:
    async with Session() as s:
        pos = await s.scalar(select(MonthlyPosition).where(MonthlyPosition.ticker == ticker))
        if pos is None:
            return None
        proceeds = pos.shares * price
        acc = await s.get(MonthlyAccount, 1)
        if acc is None:
            return None
        acc.cash += proceeds
        s.add(MonthlyTrade(ticker=ticker, side="SELL", shares=pos.shares, price=price,
                           cash_after=acc.cash, reason=reason))
        await s.delete(pos)
        await s.commit()
        return {"ticker": ticker, "side": "SELL", "shares": pos.shares,
                "price": price, "proceeds": round(proceeds, 2), "reason": reason}


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
    async with _rebalance_lock:
        post = await monthly_valuate()
        async with Session() as s:
            s.add(MonthlySnapshot(cash=post["cash"], positions_value=post["positions_value"],
                                  total_equity=post["total_equity"],
                                  allowance_total=post["allowance_total"]))
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
        row = await s2.scalar(
            select(MonthlyRebalance).where(MonthlyRebalance.rebal_month == key))
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
    if not force:
        pending = await _pending_rebuild(now)
        if pending is None:
            return {"skipped": True, "reason": "no pending month-end rebuild"}
        month, me_ts = pending
    else:
        s = _prev_trading_day(now)
        me = _candidate_month_end(s) if s is not None else None
        if me is None:
            return {"skipped": True, "reason": "no completed month-end yet"}
        month = pd.Timestamp(me).strftime("%Y-%m")
        me_ts = pd.Timestamp(me).normalize()

    allowance = await deposit_allowance()
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

    async with Session() as s:
        existing = await s.scalar(select(MonthlyRebalance).where(MonthlyRebalance.rebal_month == month))
        reb = existing or MonthlyRebalance(rebal_month=month, rebal_date=me_ts.to_pydatetime())
        reb.held_before = ",".join(held_before)
        reb.picked = ",".join(picks)
        reb.n_new = len(set(picks) - set(held_before))
        reb.snapshot = _snapshot_json(frame)
        if not existing:
            s.add(reb)
        await s.commit()

    trades: list[dict] = []
    # Fill prices: the session's STRICT exact-date OPEN, USD-converted (never
    # a stale/forward-filled close). A ticker without an F bar gets no fill;
    # that leg is retried at the next session (pending_fill keeps the rebuild
    # open, so the catch-up re-takes it at the next session's open).
    fill_map = await _open_usd_map(sorted(set(picks) | set(held_before)), f_date)
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
                             f"monthly rebuild: out of top band (fill {f_date.date()})")
        if r:
            trades.append(r)
    # BUYs: equal weight 1/N of total equity (top-ups included); the weight is
    # marked at F's close (rule 3), the leg fills at F's open (rule 2).
    valuation = await monthly_valuate(now=now)
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
                            f"monthly rebuild: top-{settings.sim_monthly_target_n} qv-mom pick")
        if r:
            trades.append(r)

    post = await monthly_valuate(now=now)
    async with Session() as s:
        s.add(MonthlySnapshot(cash=post["cash"], positions_value=post["positions_value"],
                              total_equity=post["total_equity"],
                              allowance_total=post["allowance_total"]))
        held_after = {p.ticker for p in await s.scalars(select(MonthlyPosition))}
        acc = await s.get(MonthlyAccount, 1)
        complete = (not pending_fill) and held_after == set(picks)
        if acc is not None and complete:
            acc.last_rebalance_month = month
        await s.commit()

    return {"rebalanced": True, "month": month, "allowance": allowance,
            "decision_date": str(me_ts.date()), "picks": picks,
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
    ranking supersedes a pending one."""
    if not settings.sim_monthly_enabled:
        return {"skipped": True, "reason": "monthly portfolio disabled"}
    return await run_rebalance(now=now)


# ---------------------------------------------------------------------------
# Backfill (synthetic history, same contract as daily_core.backfill)
# ---------------------------------------------------------------------------

async def _sync_start_date() -> str | None:
    """Earliest daily-sim snapshot date — the monthly backfill's default
    start so the three equity curves cover the same window (same rule as
    daily_core._sync_start_date; the daily-sim curve is the longest-running
    live one, so it anchors the shared window)."""
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
        fill session).

    No fees: the replay mirrors the LIVE engine, whose
    `_exec_buy`/`_exec_sell` charge no commission (the paper portfolios are
    fee-free by design — see docs/brokers.md), so a backfill's end state
    matches what the live engine would hold. Materialized as real
    trade/rebalance/snapshot rows the UI can render.

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
    """
    from sqlalchemy import delete as sa_delete
    from .db import (MonthlyAccount as Acc, MonthlyPosition as Pos,
                     MonthlyTrade as Tr, MonthlyAllowance as Al,
                     MonthlySnapshot as Sn, MonthlyRebalance as Rb)

    async with _rebalance_lock:
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

        # --- wipe + reset (preserve the live month markers like daily-core) ---
        live_allowance_month: str | None = None
        live_rebalance_month: str | None = None
        async with Session() as s:
            acc0 = await s.get(Acc, 1)
            if acc0 is not None:
                live_allowance_month = acc0.last_allowance_month
                live_rebalance_month = acc0.last_rebalance_month
            for tbl in (Tr, Pos, Al, Sn, Rb):
                await s.execute(sa_delete(tbl))
            acc = await s.get(Acc, 1)
            if acc is None:
                acc = Acc(id=1)
                s.add(acc)
            acc.cash = 0.0
            acc.last_allowance_month = None
            await s.commit()

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
                        sh = shares.pop(t)
                        proceeds = sh * px
                        cash += proceeds
                        cost_basis.pop(t, None)
                        opened_at.pop(t, None)
                        trade_rows.append((d.strftime("%Y-%m-%d"), "SELL", t,
                                           proceeds, sh, px))
                    # BUYs: equal weight 1/N of total equity marked at F's
                    # close (rule 3); the leg fills at F's open (rule 2).
                    # Mirrors the live engine exactly: ONE pre-trade
                    # valuation, per-position values rounded to the cents it
                    # reports, total equity rounded once.
                    pos_value = 0.0
                    pos_values: dict[str, float] = {}
                    for t in sorted(shares):  # live reads positions ticker-ordered
                        price = px_of(t, d)
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
                        sh = share_size(budget, px)
                        if sh < 0.0001:
                            continue  # live _exec_buy: sub-minimum leg → no fill
                        cost = sh * px
                        if cash < cost:
                            continue  # live _exec_buy: insufficient cash → no fill
                        if t in shares:
                            new_sh = shares[t] + sh
                            cost_basis[t] = (shares[t] * cost_basis[t] + cost) / new_sh
                            shares[t] = new_sh
                        else:
                            shares[t] = sh
                            cost_basis[t] = px
                            opened_at[t] = d.strftime("%Y-%m-%d")
                        cash -= cost
                        trade_rows.append((d.strftime("%Y-%m-%d"), "BUY", t,
                                           cost, sh, px))
                    # The pass is complete when no leg waited on a quote and
                    # the membership matches the plan — the marker is set
                    # only then; an incomplete pass stays pending (its legs
                    # retry at the following sessions' opens).
                    if not pending_fill and set(shares) == set(picks):
                        marker = mkey

            # 4) the day's snapshot at marks (deposit → pass → snapshot).
            eq = cash + sum(sh * (px_of(t, d) or 0.0) for t, sh in shares.items())
            snaps.append((d.strftime("%Y-%m-%d"), eq, contributed))

        # --- persist the end state ---
        # allowance rows: exactly the months the replay funded (ineligible
        # months included) — matches daily_core.backfill.
        replay_months = sorted(deposited_months)
        # The live engine may already have deposited the CURRENT month while
        # the replay window has no candle for it yet — the wipe destroyed the
        # allowance row, so re-create it (same rule as daily_core.backfill).
        # `_current_month()` (operator-local) matches deposit_allowance.
        current_month = _current_month()
        if live_allowance_month == current_month and current_month not in replay_months:
            cash += settings.sim_monthly_contribution
            contributed += settings.sim_monthly_contribution
            replay_months = sorted(set(replay_months) | {current_month})
        async with Session() as s:
            acc = await s.get(Acc, 1)
            assert acc is not None  # created in the wipe step above
            # converge to the live engine's state shape: cash = replay cash,
            # positions with their true average cost and first-buy date —
            # the same bookkeeping the live _exec_buy performs.
            acc.cash = cash
            for t, sh in shares.items():
                opened = opened_at.get(t) or idx[-1].strftime("%Y-%m-%d")
                s.add(Pos(ticker=t, shares=sh, avg_cost=cost_basis.get(t, 0.0),
                          opened_at=pd.Timestamp(f"{opened} 16:00:00+00:00").to_pydatetime()))
            for td, side, t, notional, sh, pr in trade_rows:
                s.add(Tr(ticker=t, side=side, shares=round(sh, 6), price=round(pr, 6),
                         cash_after=0.0,
                         reason=f"backfill {side.lower()} ${notional:,.0f}",
                         created_at=pd.Timestamp(f"{td} 16:00:00+00:00").to_pydatetime()))
            for m_key in replay_months:
                s.add(Al(amount=settings.sim_monthly_contribution, month=m_key))
            for m_key, td, held, picked, n_new in rebalance_rows:
                s.add(Rb(rebal_month=m_key, rebal_date=pd.Timestamp(f"{td} 16:00:00+00:00").to_pydatetime(),
                         held_before=held, picked=picked, n_new=n_new,
                         snapshot=""))
            # Never let the allowance marker lag the newest inserted row, or
            # the next live deposit would duplicate that month.
            last_month = max(live_allowance_month or "",
                             replay_months[-1] if replay_months else "")
            acc.last_allowance_month = last_month or None
            # the rebalance marker: the replay's own completion state, never
            # regressed below what the live engine had already recorded —
            # _pending_rebuild gates on it, so a month the replay could not
            # complete stays unmarked and the live engine retries it.
            final_marker = max(live_rebalance_month or "", marker or "")
            acc.last_rebalance_month = final_marker or None
            for sd, eq, contrib_at_day in snaps:
                s.add(Sn(cash=0.0, positions_value=eq, total_equity=eq,
                         allowance_total=round(contrib_at_day, 2),
                         created_at=pd.Timestamp(f"{sd} 16:00:00+00:00").to_pydatetime()))
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
    async with Session() as s:
        rows = (await s.scalars(select(MonthlyTrade)
                                .order_by(MonthlyTrade.created_at.desc()).limit(limit))).all()
    return [{"ticker": r.ticker, "side": r.side, "shares": r.shares, "price": r.price,
             "cash_after": round(r.cash_after, 2), "reason": r.reason,
             "date": r.created_at.isoformat()} for r in rows]


async def get_equity_curve(limit: int = 365) -> list[dict]:
    async with Session() as s:
        rows = (await s.scalars(select(MonthlySnapshot)
                                .order_by(MonthlySnapshot.created_at.desc()).limit(limit))).all()
    return [{"date": r.created_at.strftime("%Y-%m-%d"), "cash": r.cash,
             "positions_value": r.positions_value, "total_equity": r.total_equity,
             "allowance_total": r.allowance_total} for r in reversed(rows)]


async def get_rebalances(limit: int = 24) -> list[dict]:
    async with Session() as s:
        rows = (await s.scalars(select(MonthlyRebalance)
                                .order_by(MonthlyRebalance.created_at.desc()).limit(limit))).all()
    return [{"month": r.rebal_month, "date": r.rebal_date.strftime("%Y-%m-%d"),
             "held_before": r.held_before.split(",") if r.held_before else [],
             "picked": r.picked.split(",") if r.picked else [],
             "n_new": r.n_new} for r in rows]


async def get_allowances() -> list[dict]:
    async with Session() as s:
        rows = (await s.scalars(select(MonthlyAllowance).order_by(MonthlyAllowance.month.desc()))).all()
    return [{"amount": r.amount, "month": r.month} for r in rows]


async def reset_monthly() -> dict[str, Any]:
    """Wipe the monthly portfolio and re-initialize it at start cash.

    Clears trades, positions, allowances, snapshots and rebalances, and resets
    the account markers so the next deposit/rebalance starts clean. Market
    data, fundamentals, the screener and the universe selection are left
    intact (only portfolio *state* is cleared)."""
    from sqlalchemy import delete as sa_delete
    from .db import (MonthlyAccount as Acc, MonthlyPosition as Pos,
                     MonthlyTrade as Tr, MonthlyAllowance as Al,
                     MonthlySnapshot as Sn, MonthlyRebalance as Rb)

    async with _rebalance_lock:
        async with Session() as s:
            for tbl in (Tr, Pos, Al, Sn, Rb):
                await s.execute(sa_delete(tbl))
            acc = await s.get(Acc, 1)
            if acc is None:
                acc = Acc(id=1, cash=settings.sim_monthly_start_cash,
                          last_allowance_month=None)
                s.add(acc)
            else:
                acc.cash = settings.sim_monthly_start_cash
                acc.last_allowance_month = None
                acc.last_rebalance_month = None
            await s.commit()
    logger.info("Monthly reset: cash=%.2f", settings.sim_monthly_start_cash)
    return {"ok": True, "cash": settings.sim_monthly_start_cash}
