from datetime import datetime
import httpx
from sqlalchemy import select
from .db import Candle, Session
from .config import settings
async def refresh(ticker:str):
    if not settings.twelve_data_api_key: raise ValueError("TWELVE_DATA_API_KEY is not configured")
    url="https://api.twelvedata.com/time_series"
    p={"symbol":ticker,"interval":"1day","outputsize":365,"apikey":settings.twelve_data_api_key}
    async with httpx.AsyncClient(timeout=20) as c: data=(await c.get(url,params=p)).json()
    if data.get("status")=="error" or "values" not in data: raise ValueError(data.get("message","market-data response had no candles"))
    async with Session() as s:
        for x in data["values"]:
            ts=datetime.fromisoformat(x["datetime"])
            existing=await s.scalar(select(Candle).where(Candle.ticker==ticker, Candle.timestamp==ts))
            vals={"open":float(x["open"]),"high":float(x["high"]),"low":float(x["low"]),"close":float(x["close"]),"volume":float(x.get("volume") or 0)}
            if existing:
                for k,v in vals.items(): setattr(existing,k,v)
            else: s.add(Candle(ticker=ticker,timestamp=ts,**vals))
        await s.commit()
async def candles(ticker:str):
    async with Session() as s:
        rows=(await s.scalars(select(Candle).where(Candle.ticker==ticker).order_by(Candle.timestamp))).all()
        return [{"time":r.timestamp.strftime("%Y-%m-%d"),"open":r.open,"high":r.high,"low":r.low,"close":r.close,"volume":r.volume} for r in rows]
