"""Monthly qv-mom portfolio: stockstrat's qv-mom-v1 as a separate paper
portfolio with a monthly rebalance (last trading day of each month).

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

# Anchor for the allowance "month" key: the same operator-local timezone the
# sim portfolio uses (settings.allowance_tz), so both portfolios' cumulative
# contributed figures step at the same calendar-month boundary and the two
# equity curves are directly comparable.
_TZ = ZoneInfo(settings.allowance_tz)


def _current_month() -> str:
    """Operator-local calendar month key (YYYY-MM), matching sim._current_month."""
    return datetime.now(_TZ).strftime("%Y-%m")


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

    for t in list(close.columns):
        suffix = t[t.rfind("."):] if "." in t else None
        pm = fundamentals_mod.SUFFIX_FX.get(suffix) if suffix else None
        if not pm or pm[0] not in close.columns:
            continue
        rate = close[pm[0]].reindex(close.index).ffill()
        if pm[1] == "mul":
            close[t] = close[t] * rate
        else:
            close[t] = close[t] / rate
    drop = [c for c in close.columns if c.endswith("=X")]
    close = close.drop(columns=drop)
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


async def load_frames(tickers: list[str], asof: datetime | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(close, volume) DataFrames {date: {ticker: value}} from the candles
    cache, loaded in a thread. Foreign-listing closes are converted to USD at
    the historical FX rate from the FX pseudo-tickers in the same cache."""
    async with Session() as s:
        q = select(Candle).order_by(Candle.timestamp)
        if asof is not None:
            q = q.where(Candle.timestamp <= asof)
        rows = list((await s.scalars(q)).all())
    return await asyncio.to_thread(load_frames_sync, tickers, rows, asof)


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
    # 12-1 momentum: close[t-21td] / close[t-252td] - 1
    if len(hist) <= 252:
        return None
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
            fcf = ocf + capex  # capex is conventionally negative
        elif ocf is not None:
            fcf = ocf  # conservative fallback: OCF alone
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
    df["eligible"] = (
        (df["mcap"] > settings.sim_monthly_min_mcap)
        & (df["dollar_vol"] > settings.sim_monthly_min_dollar_vol)
        & np.isfinite(df["mom"])
        & (df["mom"] > -0.99)
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
    and TTM FCF = OCF + capex (capex conventionally negative; falls back to
    OCF alone). None values mean "not computable", never "bad".

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
        fcf = ocf + capex  # capex is conventionally negative
    elif ocf is not None:
        fcf = ocf  # conservative fallback: OCF alone
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


def _qv_order(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Score = pct_rank(ROE) + pct_rank(momentum) + pct_rank(1 / P-FCF if
    available). Explicit total order (unique ranking => unique portfolio):
      score desc -> higher ROE -> lower P/FCF (missing last) -> ticker asc."""
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
    """Last weekday of `now`'s month (holiday approximation, documented)."""
    if now.month == 12:
        last = datetime(now.year + 1, 1, 1, tzinfo=now.tzinfo) - timedelta(days=1)
    else:
        last = datetime(now.year, now.month + 1, 1, tzinfo=now.tzinfo) - timedelta(days=1)
    while last.weekday() >= 5:
        last = last - timedelta(days=1)
    return last


def is_rebalance_day(today: datetime | None = None) -> bool:
    """True when `today` is the last US trading day of its month.

    Heuristic: the last calendar day of the month walked back to a weekday.
    A holiday on the final weekday is accepted as an approximation for paper
    trading (documented caveat)."""
    now = today or datetime.now(UTC)
    return now.weekday() < 5 and now.date() == month_last_trading_day(now).date()


# ---------------------------------------------------------------------------
# Monthly portfolio engine
# ---------------------------------------------------------------------------

async def monthly_valuate() -> dict[str, Any]:
    """Value the monthly portfolio (same shape as sim.valuate)."""
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
    px_map = await _price_usd_map([p.ticker for p in positions])
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


async def _price_usd_map(tickers: list[str]) -> dict[str, float | None]:
    """{ticker: latest USD-converted close} for the given tickers.

    The strategy decides on USD-converted closes (load_frames_sync), so
    execution and valuation must book the same converted price — a raw
    local-currency close booked as USD would misstate exposure by the FX
    rate and ignore FX moves in pnl/equity. Same suffix->FX mapping."""
    from sqlalchemy import select as _select

    wanted = set(tickers)
    pairs = sorted({pm[0] for t in tickers if (pm := fundamentals_mod._suffix_fx(t))})
    async with Session() as s:
        rows = (await s.scalars(
            _select(Candle).where(Candle.ticker.in_(wanted | set(pairs)))
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


async def _exec_buy(ticker: str, price: float, budget: float, reason: str) -> dict | None:
    if price <= 0 or budget < 1:
        return None
    shares = math.floor(budget / price * 10000) / 10000
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

    Fundamentals source priority: SEC EDGAR for US filers (real filed dates,
    history to ~2009), yfinance fallback for CIK-less listings (ETFs, European
    exchanges). Same split as the stockstrat research pipeline."""
    from . import edgar
    from .market import refresh
    refresh_errors: list[str] = []
    wanted = list(tickers) + sorted({pm[0] for t in tickers if (pm := fundamentals_mod._suffix_fx(t))})
    for t in wanted:
        try:
            await refresh(t, "10y")
        except Exception as e:
            refresh_errors.append(f"{t}: {e}")
    fund_status: dict[str, str]
    if settings.sim_monthly_fundamentals_source == "edgar":
        fund_status = await edgar.refresh_universe_mixed(tickers=tickers)
    else:
        fund_status = await fundamentals_mod.refresh_universe(tickers=tickers)
    return refresh_errors, fund_status


async def run_rebalance(force: bool = False) -> dict[str, Any]:
    """One rebalance decision + execution (runs on the last trading day of the
    month, or manually). Deposits the allowance first (no-op if the daily
    scheduler already funded the account this month), refreshes data, picks
    with hysteresis, then executes SELLs (freed slots) and equal-weight BUYs.
    Idempotent per month unless ``force``."""
    # Non-blocking acquire: the idempotence gate below runs before the
    # multi-minute data refresh, so a queued second invocation would pass
    # the gate too and double-deposit / duplicate trades.
    if _rebalance_lock.locked():
        return {"skipped": True, "reason": "already running"}
    async with _rebalance_lock:
        return await _run_rebalance_locked(force)


async def _run_rebalance_locked(force: bool) -> dict[str, Any]:
    month = datetime.now(UTC).strftime("%Y-%m")
    async with Session() as s:
        acc = await s.get(MonthlyAccount, 1)
        if acc is not None and acc.last_rebalance_month == month and not force:
            return {"skipped": True, "reason": f"already rebalanced {month}"}

    allowance = await deposit_allowance()
    tickers = universe_tickers(settings.sim_monthly_universe)
    refresh_errors, fund_status = await refresh_data(tickers)

    close, vol = await load_frames(tickers, None)
    fund = await fundamentals_mod.load_fundamentals(tickers)
    # tz-naive date: candle timestamps are stored naive (UTC), so the slice
    # index must be naive too (mixing tz-aware would raise in pandas).
    d = pd.Timestamp(datetime.now(UTC).replace(tzinfo=None).date())
    frame = await asyncio.to_thread(eligible_frame, d, close, vol, fund)
    if frame is None:
        return {"skipped": True, "reason": "no eligible frame",
                "refresh_errors": refresh_errors, "fundamentals": fund_status}

    async with Session() as s:
        held_rows = (await s.scalars(select(MonthlyPosition))).all()
    held_before = sorted(p.ticker for p in held_rows)
    picks, _ = await asyncio.to_thread(pick_portfolio, d, close, vol, fund, held_before, frame)

    async with Session() as s:
        existing = await s.scalar(select(MonthlyRebalance).where(MonthlyRebalance.rebal_month == month))
        reb = existing or MonthlyRebalance(rebal_month=month, rebal_date=d.to_pydatetime())
        reb.held_before = ",".join(held_before)
        reb.picked = ",".join(picks)
        reb.n_new = len(set(picks) - set(held_before))
        reb.snapshot = _snapshot_json(frame)
        if not existing:
            s.add(reb)
        acc = await s.get(MonthlyAccount, 1)
        if acc is not None:
            acc.last_rebalance_month = month
        await s.commit()

    trades: list[dict] = []
    # USD-converted prices for everything we might touch (held + picked):
    # the decision ran on FX-converted closes, so trades must book the same
    # USD prices — one DB pass, shared by SELLs and BUYs below.
    price_map = await _price_usd_map(sorted(set(picks) | set(held_before)))
    # SELLs first: held names no longer picked (frees cash for the BUYs).
    # A held name without candles falls back to avg_cost (the valuation
    # convention) — never sell at 0.
    async with Session() as s:
        costs = {p.ticker: p.avg_cost for p in
                 (await s.scalars(select(MonthlyPosition))).all()}
    for t in held_before:
        if t not in picks:
            price = price_map.get(t)
            if not price:
                price = costs.get(t) or 0.0
            r = await _exec_sell(t, price, "monthly rebalance: out of top band")
            if r:
                trades.append(r)
    # BUYs: equal weight 1/N of total equity (top-ups included)
    valuation = await monthly_valuate()
    weight = valuation["total_equity"] / max(len(picks), 1)
    for t in picks:
        price = price_map.get(t)
        if not price:
            continue
        held_pos = next((p for p in valuation["positions"] if p["ticker"] == t), None)
        current_value = held_pos["value"] if held_pos else 0.0
        budget = weight - current_value
        if budget < 1:
            continue
        r = await _exec_buy(t, price, budget,
                            f"monthly rebalance: top-{settings.sim_monthly_target_n} qv-mom pick")
        if r:
            trades.append(r)

    post = await monthly_valuate()
    async with Session() as s:
        s.add(MonthlySnapshot(cash=post["cash"], positions_value=post["positions_value"],
                              total_equity=post["total_equity"],
                              allowance_total=post["allowance_total"]))
        await s.commit()

    return {"rebalanced": True, "month": month, "allowance": allowance,
            "picks": picks, "held_before": held_before, "trades": trades,
            "valuation": post, "refresh_errors": refresh_errors,
            "fundamentals": fund_status}


async def run_monthly_cycle() -> dict[str, Any]:
    """Scheduler entry: only acts on the last trading day of the month."""
    if not settings.sim_monthly_enabled:
        return {"skipped": True, "reason": "monthly portfolio disabled"}
    if not is_rebalance_day():
        return {"skipped": True, "reason": "not last trading day of month"}
    return await run_rebalance()


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
