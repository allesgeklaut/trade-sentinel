from datetime import datetime
from functools import partial
import asyncio
import httpx
import pandas as pd
import numpy as np
import yfinance as yf
from sqlalchemy import select
from .db import Candle, Session
from .config import settings

def provider():
    value=settings.market_data_provider.lower().strip()
    if value not in {"yfinance","twelvedata"}: raise ValueError("MARKET_DATA_PROVIDER must be yfinance or twelvedata")
    if value=="twelvedata" and not settings.twelve_data_api_key: raise ValueError("TWELVE_DATA_API_KEY is required when MARKET_DATA_PROVIDER=twelvedata")
    return value

def yahoo_history(ticker):
    frame=yf.Ticker(ticker).history(period="2y",interval="1d",auto_adjust=False,actions=False,raise_errors=True)
    if frame.empty: raise ValueError(f"No Yahoo Finance daily data for {ticker}")
    frame=frame.replace([np.inf,-np.inf],np.nan).dropna(subset=["Open","High","Low","Close"])
    if frame.empty: raise ValueError(f"No valid Yahoo Finance daily data for {ticker}")
    return [{"timestamp":x.Index.to_pydatetime().replace(tzinfo=None),"open":float(x.Open),"high":float(x.High),"low":float(x.Low),"close":float(x.Close),"volume":float(x.Volume or 0)} for x in frame.itertuples()]
def yahoo_search(q):
    quotes=yf.Search(q,max_results=8,news_count=0,lists_count=0,enable_fuzzy_query=True,raise_errors=True).quotes
    return [{"symbol":x.get("symbol"),"name":x.get("shortname") or x.get("longname") or x.get("symbol"),"exchange":x.get("exchDisp") or x.get("exchange",""),"country":x.get("region", ""),"type":x.get("quoteType","")} for x in quotes if x.get("symbol")]
async def search(q):
    if provider()=="yfinance": return await asyncio.to_thread(yahoo_search,q)
    async with httpx.AsyncClient(timeout=10) as c: data=(await c.get("https://api.twelvedata.com/symbol_search",params={"symbol":q,"outputsize":8,"apikey":settings.twelve_data_api_key})).json()
    if data.get("status")=="error": raise ValueError(data.get("message","Symbol search failed"))
    return [{"symbol":x.get("symbol"),"name":x.get("instrument_name",x.get("symbol")),"exchange":x.get("exchange",""),"country":x.get("country",""),"type":x.get("instrument_type","")} for x in data.get("data",[])]
async def refresh(ticker):
    if provider()=="yfinance": values=await asyncio.to_thread(yahoo_history,ticker)
    else:
        async with httpx.AsyncClient(timeout=20) as c: data=(await c.get("https://api.twelvedata.com/time_series",params={"symbol":ticker,"interval":"1day","outputsize":365,"apikey":settings.twelve_data_api_key})).json()
        if data.get("status")=="error" or "values" not in data: raise ValueError(data.get("message","market-data response had no candles"))
        values=[{"timestamp":datetime.fromisoformat(x["datetime"]),"open":float(x["open"]),"high":float(x["high"]),"low":float(x["low"]),"close":float(x["close"]),"volume":float(x.get("volume") or 0)} for x in data["values"]]
    async with Session() as s:
        for x in values:
            row=await s.scalar(select(Candle).where(Candle.ticker==ticker,Candle.timestamp==x["timestamp"]))
            if row:
                for key in ["open","high","low","close","volume"]: setattr(row,key,x[key])
            else: s.add(Candle(ticker=ticker,**x))
        await s.commit()
async def candles(ticker):
    async with Session() as s:
        rows=(await s.scalars(select(Candle).where(Candle.ticker==ticker).order_by(Candle.timestamp))).all()
    return [{"time":r.timestamp.strftime("%Y-%m-%d"),"open":r.open,"high":r.high,"low":r.low,"close":r.close,"volume":r.volume} for r in rows]
