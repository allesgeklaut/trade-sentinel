from datetime import datetime
import asyncio
import logging
import httpx
import numpy as np
import pandas as pd
import yfinance as yf
from sqlalchemy import select
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from .db import Candle, Session
from .config import settings

logger = logging.getLogger("trade_sentinel.market")


def provider():
    value=settings.market_data_provider.lower().strip()
    if value not in {"yfinance","twelvedata"}: raise ValueError("MARKET_DATA_PROVIDER must be yfinance or twelvedata")
    if value=="twelvedata" and not settings.twelve_data_api_key: raise ValueError("TWELVE_DATA_API_KEY is required when MARKET_DATA_PROVIDER=twelvedata")
    return value

# Range presets: key → (yfinance period, twelvedata outputsize)
RANGES = {
    "6m": ("6mo", 126),
    "2y": ("2y", 730),
    "5y": ("5y", 1825),
    "10y": ("10y", 3650),
    "max": ("max", 5000),
}

def yahoo_history(ticker, period="2y"):
    # auto_adjust=True: split/dividend-adjusted OHLC. The monthly qv-mom
    # strategy needs adjusted closes for 12-1 momentum (a raw close series
    # fakes a crash at every ex-dividend/split date); the daily technical
    # engine uses the same series, so indicators are consistent everywhere.
    # hide_exceptions=False (global yfinance config) is the non-deprecated
    # replacement for raise_errors=True: fetch failures raise instead of
    # returning None.
    yf.config.debug.hide_exceptions = False
    frame=yf.Ticker(ticker).history(period=period,interval="1d",auto_adjust=True,actions=False)
    if frame.empty: raise ValueError(f"No Yahoo Finance daily data for {ticker}")
    frame=frame.replace([np.inf,-np.inf],np.nan).dropna(subset=["Open","High","Low","Close"])
    if frame.empty: raise ValueError(f"No valid Yahoo Finance daily data for {ticker}")
    out = []
    for ts, row in frame.iterrows():
        assert isinstance(ts, pd.Timestamp), f"unexpected index type {type(ts)}"
        out.append({
            "timestamp": ts.to_pydatetime().replace(tzinfo=None),
            "open": float(row["Open"]), "high": float(row["High"]),
            "low": float(row["Low"]), "close": float(row["Close"]),
            "volume": float(row.get("Volume", 0) or 0),
        })
    return out
def yahoo_info(ticker):
    """Fetch a human-readable company name for a single ticker."""
    t = yf.Ticker(ticker)
    try:
        fi = t.fast_info
        name = getattr(fi, "short_name", None) or getattr(fi, "long_name", None)
        if name:
            return name
    except Exception:
        pass
    try:
        i = t.info
        return i.get("shortName") or i.get("longName") or ""
    except Exception:
        return ""

def yahoo_search(q):
    quotes=yf.Search(q,max_results=8,news_count=0,lists_count=0,enable_fuzzy_query=True,raise_errors=True).quotes
    return [{"symbol":x.get("symbol"),"name":x.get("shortname") or x.get("longname") or x.get("symbol"),"exchange":x.get("exchDisp") or x.get("exchange",""),"country":x.get("region", ""),"type":x.get("quoteType","")} for x in quotes if x.get("symbol")]
async def info(ticker):
    if provider()=="yfinance": return await asyncio.to_thread(yahoo_info,ticker)
    async with httpx.AsyncClient(timeout=10) as c:
        data=(await c.get("https://api.twelvedata.com/profile",params={"symbol":ticker,"apikey":settings.twelve_data_api_key})).json()
    return data.get("name","") if data.get("status")!="error" else ""

async def search(q):
    if provider()=="yfinance": return await asyncio.to_thread(yahoo_search,q)
    async with httpx.AsyncClient(timeout=10) as c: data=(await c.get("https://api.twelvedata.com/symbol_search",params={"symbol":q,"outputsize":8,"apikey":settings.twelve_data_api_key})).json()
    if data.get("status")=="error": raise ValueError(data.get("message","Symbol search failed"))
    return [{"symbol":x.get("symbol"),"name":x.get("instrument_name",x.get("symbol")),"exchange":x.get("exchange",""),"country":x.get("country",""),"type":x.get("instrument_type","")} for x in data.get("data",[])]
async def refresh(ticker, period="2y"):
    if period not in RANGES: raise ValueError(f"Unsupported range: {period}")
    yf_period, td_output = RANGES[period]
    if provider()=="yfinance": values=await asyncio.to_thread(yahoo_history,ticker,yf_period)
    else:
        async with httpx.AsyncClient(timeout=20) as c: data=(await c.get("https://api.twelvedata.com/time_series",params={"symbol":ticker,"interval":"1day","outputsize":td_output,"apikey":settings.twelve_data_api_key})).json()
        if data.get("status")=="error" or "values" not in data: raise ValueError(data.get("message","market-data response had no candles"))
        values=[{"timestamp":datetime.fromisoformat(x["datetime"]),"open":float(x["open"]),"high":float(x["high"]),"low":float(x["low"]),"close":float(x["close"]),"volume":float(x.get("volume") or 0)} for x in data["values"]]
    async with Session() as s:
        if values:
            stmt = sqlite_insert(Candle).values(
                [{"ticker": ticker, **x} for x in values]
            )
            stmt = stmt.on_conflict_do_update(
                index_elements=["ticker", "timestamp"],
                set_={
                    "open": stmt.excluded.open,
                    "high": stmt.excluded.high,
                    "low": stmt.excluded.low,
                    "close": stmt.excluded.close,
                    "volume": stmt.excluded.volume,
                },
            )
            await s.execute(stmt)
        await s.commit()
async def candles(ticker, period=None):
    async with Session() as s:
        rows=(await s.scalars(select(Candle).where(Candle.ticker==ticker).order_by(Candle.timestamp))).all()
    # Filter by period if requested (slice the last N rows)
    if period:
        count = PERIOD_COUNTS.get(period)  # ~252 trading days/year
        if count: rows = rows[-count:]
    return [{"time":r.timestamp.strftime("%Y-%m-%d"),"open":r.open,"high":r.high,"low":r.low,"close":r.close,"volume":r.volume} for r in rows]


# Chart-slice sizes per range preset (~252 trading days/year). Shared by the
# API layer (dashboard endpoint) so both sides agree on how much to trim.
PERIOD_COUNTS = {"6m":126,"2y":504,"5y":1260,"10y":2520,"max":5000}


async def refresh_many(tickers, period="2y", *, concurrency: int = 4, on_result=None, work=None):
    """Refresh candle data for many tickers with bounded concurrency.

    Input is deduplicated while preserving order. Returns ``(refreshed,
    errors)`` where errors are "TICKER: message" strings and ``refreshed``
    lists the tickers that completed without error, in input order (not
    completion order). ``on_result(ticker, ok, error_or_None)`` fires after
    each ticker completes — success or failure — so callers can update
    progress UI while the batch is still running. Pass ``work`` to run a
    custom per-ticker coroutine instead of :func:`refresh` (used by the
    screener to bundle per-symbol scoring with the fetch).
    """
    ordered = list(dict.fromkeys(tickers))
    semaphore = asyncio.Semaphore(max(1, concurrency))
    errors: list[str] = []
    ok_flags: dict[str, bool] = {}

    async def _one(ticker: str) -> None:
        async with semaphore:
            err: str | None = None
            try:
                if work is not None:
                    await work(ticker)
                else:
                    await refresh(ticker, period)
            except Exception as e:
                err = str(e)
                errors.append(f"{ticker}: {e}")
                logger.warning("refresh %s failed: %s", ticker, e)
            ok_flags[ticker] = err is None
        if on_result is not None:
            on_result(ticker, err is None, err)

    await asyncio.gather(*(_one(t) for t in ordered))
    refreshed = [t for t in ordered if ok_flags.get(t)]
    return refreshed, errors
