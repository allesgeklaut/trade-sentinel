from datetime import datetime
from functools import partial
import asyncio
import httpx
import pandas as pd
import numpy as np
import yfinance as yf
from sqlalchemy import select
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from .db import Candle, Session
from .config import settings

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
    frame=yf.Ticker(ticker).history(period=period,interval="1d",auto_adjust=True,actions=False,raise_errors=True)
    if frame.empty: raise ValueError(f"No Yahoo Finance daily data for {ticker}")
    frame=frame.replace([np.inf,-np.inf],np.nan).dropna(subset=["Open","High","Low","Close"])
    if frame.empty: raise ValueError(f"No valid Yahoo Finance daily data for {ticker}")
    return [{"timestamp":x.Index.to_pydatetime().replace(tzinfo=None),"open":float(x.Open),"high":float(x.High),"low":float(x.Low),"close":float(x.Close),"volume":float(x.Volume or 0)} for x in frame.itertuples()]
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
        period_counts = {"6m":126,"2y":504,"5y":1260,"10y":2520,"max":5000}  # ~252 trading days/year
        count = period_counts.get(period)
        if count: rows = rows[-count:]
    return [{"time":r.timestamp.strftime("%Y-%m-%d"),"open":r.open,"high":r.high,"low":r.low,"close":r.close,"volume":r.volume} for r in rows]