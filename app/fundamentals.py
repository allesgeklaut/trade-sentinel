"""Fundamentals fetcher for the monthly qv-mom portfolio.

Quarterly + annual statements from yfinance for the whole universe, stored in
the ``fundamentals`` table as XBRL-style facts (tag / start / end / filed /
val) so the scoring math in app/monthly.py can treat US and European listings
identically.

Point-in-time approximation (documented trade-off, mirrors stockstrat):
  * every period gets ``filed = period_end + FILED_LAG_DAYS`` — the real
    filing date is not in the yfinance feed
  * values are today's restated view yfinance serves, frozen at fetch time
  * coverage is typically ~4-5 years, so names simply become eligible later
  * statement values are converted to USD at the period-end FX rate, matching
    the USD-converted prices used for momentum/market-cap

Tags (SEC XBRL names): NetIncomeLoss, StockholdersEquity,
CommonStockSharesOutstanding, NetCashProvidedByUsedInOperatingActivities,
PaymentsToAcquirePropertyPlantAndEquipment.
"""
import asyncio
import logging
from datetime import datetime, timezone

import pandas as pd
from sqlalchemy import func

from .config import settings
from .db import Fundamental, Session
from .screener import tickers as universe_tickers

logger = logging.getLogger(__name__)

FILED_LAG_DAYS = 45   # conservative assumption: most filers report within ~6 weeks
QUARTER_DAYS = 91     # nominal quarter length for duration facts
YEAR_DAYS = 365       # nominal fiscal-year length for annual duration facts

# Yahoo exchange suffix -> FX pair: "XXXUSD=X" multiplies local->USD,
# "USDXXX=X" divides (local currency units per USD).
SUFFIX_FX = {
    ".DE": ("EURUSD=X", "mul"), ".AS": ("EURUSD=X", "mul"), ".MC": ("EURUSD=X", "mul"),
    ".PA": ("EURUSD=X", "mul"), ".BR": ("EURUSD=X", "mul"), ".HE": ("EURUSD=X", "mul"),
    ".SW": ("USDCHF=X", "div"), ".ST": ("USDSEK=X", "div"), ".L": ("GBPUSD=X", "mul"),
}

STALE_AFTER_DAYS = 90  # refetch fundamentals quarterly

# yfinance statement row names -> (XBRL tag, duration?, convert to USD?, span days)
_FIELD_MAP_QUARTERLY = {
    "Net Income": ("NetIncomeLoss", True, True, QUARTER_DAYS),
    "Net Income Common Stockholders": ("NetIncomeLoss", True, True, QUARTER_DAYS),
    "Operating Cash Flow": ("NetCashProvidedByUsedInOperatingActivities", True, True, QUARTER_DAYS),
    "Capital Expenditure": ("PaymentsToAcquirePropertyPlantAndEquipment", True, True, QUARTER_DAYS),
    "Stockholders Equity": ("StockholdersEquity", False, True, YEAR_DAYS),
    "Total Equity Gross Minority Interest": ("StockholdersEquity", False, True, YEAR_DAYS),
    "Ordinary Shares Number": ("CommonStockSharesOutstanding", False, False, YEAR_DAYS),
    "Share Issued": ("CommonStockSharesOutstanding", False, False, YEAR_DAYS),
    "Ordinary Shares": ("CommonStockSharesOutstanding", False, False, YEAR_DAYS),
}
# Annual statements reuse the same row names; annual duration facts get
# YEAR_DAYS spans so the TTM builder can prefer them as FY anchors.
_ANNUAL_SPAN_OVERRIDE = {
    "NetIncomeLoss": YEAR_DAYS,
    "NetCashProvidedByUsedInOperatingActivities": YEAR_DAYS,
    "PaymentsToAcquirePropertyPlantAndEquipment": YEAR_DAYS,
}


def _suffix_fx(ticker: str) -> tuple[str, str] | None:
    suffix = ticker[ticker.rfind("."):] if "." in ticker else None
    return SUFFIX_FX.get(suffix) if suffix else None


def _fx_factor(end: pd.Timestamp, rate: pd.Series | None, mode: str) -> float:
    """USD conversion factor for a value booked at period `end`."""
    if rate is None or rate.empty:
        return 1.0  # no FX data -> assume USD (documented fallback)
    end = pd.Timestamp(end)
    if pd.isna(end):
        return 1.0
    if end.tzinfo is not None:
        end = end.tz_localize(None)
    # rate index may be tz-aware (candle timestamps round-trip as UTC);
    # compare in the same convention (naive UTC)
    idx = pd.DatetimeIndex(rate.index)
    if idx.tz is not None:
        rate = rate.copy()
        rate.index = idx.tz_localize(None)
    r_val = rate.asof(end)
    r = float(r_val)  # type: ignore[arg-type]
    if pd.isna(r) or r == 0.0:
        return 1.0
    return r if mode == "mul" else 1.0 / r


def fetch_rec(symbol: str, fx_rate: pd.Series | None = None, fx_mode: str = "mul") -> tuple[list[dict], str | None]:
    """Fetch one ticker's fundamentals as fact records (sync, network).

    Returns (facts, note) — note is None on success, an explanatory string
    for ETFs/context-only symbols.
    """
    import yfinance as yf

    tk = yf.Ticker(symbol)
    try:
        info = tk.info or {}
    except Exception:  # noqa: BLE001 - delisted/invalid symbols
        info = {}
    qtype = info.get("quoteType")
    if qtype is not None and qtype != "EQUITY":
        return [], f"quoteType={qtype}: prices only"

    quarterly = (tk.quarterly_income_stmt, tk.quarterly_balance_sheet, tk.quarterly_cashflow)
    annual = (tk.income_stmt, tk.balance_sheet, tk.cash_flow)

    facts: list[dict] = []
    # quarterly values by (tag, end) for the mislabeled-column guard below
    qmap: dict[tuple[str, str], float] = {}

    def add_row(row_name: str, series, span_override: int | None):
        if series is None:
            return
        tag, duration, convert, span_days = _FIELD_MAP_QUARTERLY[row_name]
        if span_override:
            span_days = span_override
        for end, v in series.items():
            if pd.isna(v):
                continue
            v_f = float(v) if not hasattr(v, "item") else float(v.item())
            end_ts = pd.Timestamp(end)
            if pd.isna(end_ts):
                continue
            start = (end_ts - pd.Timedelta(days=span_days)).strftime("%Y-%m-%d") if duration else None
            end_str = end_ts.strftime("%Y-%m-%d")
            if convert:
                factor = _fx_factor(end_ts, fx_rate, fx_mode)  # type: ignore[arg-type]
            else:
                factor = 1.0
            val = v_f * factor
            # some feeds serve quarterly-scale columns in the "annual"
            # statement; if the value equals the quarterly value at the same
            # end, it is NOT a fiscal-year figure — skip it so the TTM
            # builder never mistakes a quarter for an FY
            if duration and span_days > 200:
                qv = qmap.get((tag, end_str))
                if qv is not None and abs(val - qv) / max(abs(qv), 1e-9) < 0.10:
                    continue
            facts.append({
                "tag": tag,
                "start": start,
                "end": end_str,
                "filed": (end_ts + pd.Timedelta(days=FILED_LAG_DAYS)).strftime("%Y-%m-%d"),
                "val": val,
                "currency": "USD" if convert else "shares",
            })
            if duration and span_days <= 200:
                qmap[(tag, end_str)] = val

    for df in quarterly:
        if df is None:
            continue
        for row_name in df.index:
            if row_name in _FIELD_MAP_QUARTERLY:
                add_row(row_name, df.loc[row_name].dropna(), None)
    for df in annual:
        if df is None:
            continue
        for row_name in df.index:
            if row_name in _FIELD_MAP_QUARTERLY:
                tag = _FIELD_MAP_QUARTERLY[row_name][0]
                add_row(row_name, df.loc[row_name].dropna(), _ANNUAL_SPAN_OVERRIDE.get(tag))

    return facts, None


async def _fx_rates_from_candles(pairs: list[str]) -> dict[str, pd.Series]:
    """Historical FX closes from the candles cache ({pair: Series})."""
    from sqlalchemy import select

    from .db import Candle
    out: dict[str, pd.Series] = {}
    async with Session() as sess:
        rows = (await sess.scalars(
            select(Candle).where(Candle.ticker.in_(set(pairs))).order_by(Candle.timestamp)
        )).all() if pairs else []
    for pair in set(pairs):
        out[pair] = pd.Series(dtype=float)
    for r in rows:
        out.setdefault(r.ticker, pd.Series(dtype=float))
        out[r.ticker][pd.Timestamp(r.timestamp)] = r.close
    return {k: v.dropna() for k, v in out.items()}


async def refresh_universe(universe: str | None = None, tickers: list[str] | None = None,
                           force: bool = False) -> dict[str, str]:
    """Fetch fundamentals for the universe and store them in the DB.

    Skips tickers whose facts were refreshed within STALE_AFTER_DAYS unless
    ``force``. Returns {ticker: status} where status is ok|skipped|context-only|failed.
    """
    from sqlalchemy import select

    from sqlalchemy.dialects.sqlite import insert as sqlite_insert

    if tickers is None:
        tickers = universe_tickers(universe or settings.sim_monthly_universe)
    tickers = list(tickers)
    pairs = sorted({pm[0] for t in tickers if (pm := _suffix_fx(t))})
    rates = await _fx_rates_from_candles(pairs)

    cutoff = datetime.now(timezone.utc).timestamp() - STALE_AFTER_DAYS * 86400
    statuses: dict[str, str] = {}

    # which tickers need a (re)fetch?
    async with Session() as s:
        rows = (await s.execute(
            select(Fundamental.ticker, func.max(Fundamental.updated_at))
            .where(Fundamental.ticker.in_(tickers))
            .group_by(Fundamental.ticker)
        )).all()
    last_refresh: dict[str, datetime] = {t: ts for t, ts in rows}

    for t in sorted(tickers):
        pair_mode = _suffix_fx(t)
        rate = rates.get(pair_mode[0]) if pair_mode else None
        mode = pair_mode[1] if pair_mode else "mul"
        try:
            latest = last_refresh.get(t)
            if not force and latest is not None and latest.timestamp() > cutoff:
                statuses[t] = "skipped"
                continue
            # network fetch in a thread; DB I/O stays async
            facts, note = await asyncio.to_thread(fetch_rec, t, rate, mode)
            if not facts:
                statuses[t] = note or "failed"
                continue
            async with Session() as s:
                for f in facts:
                    stmt = sqlite_insert(Fundamental).values(ticker=t, source="yfinance", **f)
                    stmt = stmt.on_conflict_do_update(
                        index_elements=["ticker", "tag", "start", "end", "filed", "source"],
                        set_={"val": stmt.excluded.val,
                              "currency": stmt.excluded.currency,
                              "updated_at": stmt.excluded.updated_at},
                    )
                    await s.execute(stmt)
                await s.commit()
            statuses[t] = "ok"
        except Exception as e:  # noqa: BLE001
            logger.warning("fundamentals %s failed: %s", t, e)
            statuses[t] = f"failed: {e}"
    return statuses


def _load_fundamentals_rows(tickers: list[str], rows: list) -> dict[str, dict[str, list[dict]]]:
    """Build {ticker: {tag: [fact, ...]}} from Fundamental ORM rows.

    When a ticker has facts from both sources (edgar + yfinance), EDGAR wins:
    its filed dates are real and its values as-reported. Dropping the yfinance
    rows for those tickers avoids mixing a restated view into a point-in-time
    series."""
    by_ticker_source: dict[str, dict[str, list]] = {}
    for r in rows:
        by_ticker_source.setdefault(r.ticker, {}).setdefault(getattr(r, "source", "yfinance"), []).append(r)
    out: dict[str, dict[str, list[dict]]] = {}
    for t, by_src in by_ticker_source.items():
        chosen = by_src.get("edgar") or by_src.get("yfinance") or []
        for r in chosen:
            out.setdefault(t, {}).setdefault(r.tag, []).append({
                "start": r.start, "end": r.end, "filed": r.filed, "val": r.val,
            })
    return out


async def load_fundamentals(tickers: list[str]) -> dict[str, dict[str, list[dict]]]:
    """Load cached facts as {ticker: {tag: [ {start,end,filed,val}, ... ]}} —
    the exact schema the stockstrat strategy math (_asof_entries etc.) expects."""
    from sqlalchemy import select

    async with Session() as s:
        rows = (await s.scalars(
            select(Fundamental).where(Fundamental.ticker.in_(tickers))
            .order_by(Fundamental.ticker, Fundamental.tag, Fundamental.end)
        )).all()
    return await asyncio.to_thread(_load_fundamentals_rows, tickers, rows)