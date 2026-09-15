from datetime import datetime
import asyncio
import logging
from pathlib import Path
import httpx
import numpy as np
import pandas as pd
import yfinance as yf
from sqlalchemy import select
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from .db import Candle, Session
from .config import settings

logger = logging.getLogger("trade_sentinel.market")


def _twelve_key() -> str:
    """Twelve Data API key: TWELVE_DATA_API_KEY, else the file named by
    TWELVE_DATA_API_KEY_FILE (a read-only secret mount — the preferred source,
    so the key is never committed to the repo or duplicated in .env)."""
    key=(settings.twelve_data_api_key or "").strip()
    if key: return key
    path=(settings.twelve_data_api_key_file or "").strip()
    if not path: return ""
    try:
        return Path(path).read_text().strip()
    except OSError as e:
        logger.warning("Could not read Twelve Data key file %s: %s",path,e)
        return ""

def provider():
    """Primary market-data source.

    ``MARKET_DATA_PROVIDER``:
      - ``yfinance``  → always Yahoo Finance (no key needed)
      - ``twelvedata``→ Twelve Data (a key is required)
      - ``auto`` (default) → Twelve Data when a key is configured, else yfinance

    When Twelve Data is primary, :func:`refresh`/:func:`info`/:func:`search`
    fall back to yfinance per request if Twelve Data has no data for that
    ticker (unknown symbol, rate/plan limit, network error).
    """
    value=settings.market_data_provider.lower().strip() or "auto"
    if value not in {"auto","yfinance","twelvedata"}: raise ValueError("MARKET_DATA_PROVIDER must be auto, yfinance or twelvedata")
    if value=="yfinance": return "yfinance"
    if not _twelve_key():
        if value=="twelvedata": raise ValueError("TWELVE_DATA_API_KEY is required when MARKET_DATA_PROVIDER=twelvedata")
        return "yfinance"
    return "twelvedata"

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
async def _twelve_history(ticker: str, outputsize: int) -> list[dict]:
    """Twelve Data daily OHLCV, split+dividend adjusted.

    ``adjust=all`` matches yfinance's ``auto_adjust=True`` so the 12-1
    momentum (which assumes adjusted closes) stays consistent across sources.
    Raises ValueError on an API error, missing candles or malformed payloads
    so callers can fall back to yfinance.
    """
    async with httpx.AsyncClient(timeout=20) as c:
        data=(await c.get("https://api.twelvedata.com/time_series",params={
            "symbol":ticker,"interval":"1day","outputsize":outputsize,
            "adjust":"all","apikey":_twelve_key()})).json()
    if not isinstance(data, dict) or data.get("status")=="error" or "values" not in data:
        msg = data.get("message","market-data response had no candles") if isinstance(data, dict) else "malformed market-data response"
        raise ValueError(msg)
    values=[{"timestamp":datetime.fromisoformat(x["datetime"]),"open":float(x["open"]),"high":float(x["high"]),"low":float(x["low"]),"close":float(x["close"]),"volume":float(x.get("volume") or 0)} for x in data["values"]]
    if not values: raise ValueError(f"No Twelve Data daily data for {ticker}")
    return values

def _twelve_profile(ticker: str) -> str:
    with httpx.Client(timeout=10) as c:
        data=c.get("https://api.twelvedata.com/profile",params={"symbol":ticker,"apikey":_twelve_key()}).json()
    return data.get("name","") if isinstance(data, dict) and data.get("status")!="error" else ""

def _twelve_search(q: str) -> list[dict]:
    with httpx.Client(timeout=10) as c:
        data=c.get("https://api.twelvedata.com/symbol_search",params={"symbol":q,"outputsize":8,"apikey":_twelve_key()}).json()
    if not isinstance(data, dict) or data.get("status")=="error":
        raise ValueError(data.get("message","Symbol search failed") if isinstance(data, dict) else "malformed symbol search response")
    return [{"symbol":x.get("symbol"),"name":x.get("instrument_name",x.get("symbol")),"exchange":x.get("exchange",""),"country":x.get("country",""),"type":x.get("instrument_type","")} for x in data.get("data",[])]

async def info(ticker):
    if provider()=="yfinance": return await asyncio.to_thread(yahoo_info,ticker)
    try:
        name=await asyncio.to_thread(_twelve_profile,ticker)
        if name: return name
    except Exception as e:
        logger.warning("twelvedata profile failed for %s (%s) — falling back to yfinance",ticker,e)
    return await asyncio.to_thread(yahoo_info,ticker)

async def search(q):
    if provider()=="yfinance": return await asyncio.to_thread(yahoo_search,q)
    try:
        return await asyncio.to_thread(_twelve_search,q)
    except Exception as e:
        logger.warning("twelvedata symbol search failed for %r (%s) — falling back to yfinance",q,e)
        return await asyncio.to_thread(yahoo_search,q)

async def refresh(ticker, period="2y"):
    if period not in RANGES: raise ValueError(f"Unsupported range: {period}")
    yf_period, td_output = RANGES[period]
    if provider()=="yfinance":
        values=await asyncio.to_thread(yahoo_history,ticker,yf_period)
    else:
        try:
            values=await _twelve_history(ticker,td_output)
        except Exception as e:
            logger.warning("twelvedata refresh failed for %s (%s) — falling back to yfinance",ticker,e)
            values=await asyncio.to_thread(yahoo_history,ticker,yf_period)
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


async def latest_close(ticker: str) -> float | None:
    """Most recent cached close for a ticker, or None.

    A single-row query: valuation only needs the last bar, and loading a
    broad-universe ticker's entire history (decades) to read its final value
    was what made /api/sim/status slow.
    """
    async with Session() as s:
        row = await s.scalar(select(Candle.close).where(Candle.ticker == ticker)
                             .order_by(Candle.timestamp.desc()).limit(1))
    return float(row) if row is not None else None


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
