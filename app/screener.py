from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
from sqlalchemy import delete, select
from .db import ScreenerResult, Session
from .market import candles, refresh

def universe_names(): return sorted(p.stem for p in Path("universes").glob("*.txt"))
def tickers(name):
    p=Path("universes")/f"{name}.txt"
    if not p.exists() or "/" in name or ".." in name: raise ValueError("Unknown universe")
    return [x.strip().upper() for x in p.read_text().splitlines() if x.strip() and not x.startswith("#")]
def score(rows):
    if len(rows)<65: return None
    d=pd.DataFrame(rows); c=d.close; vol=d.volume
    sma50=c.rolling(50).mean().iloc[-1]; sma200=c.rolling(200).mean().iloc[-1] if len(d)>=200 else None
    r20=(c.iloc[-1]/c.iloc[-21]-1)*100; r60=(c.iloc[-1]/c.iloc[-61]-1)*100
    delta=c.diff(); up=delta.clip(lower=0).ewm(alpha=1/14,adjust=False).mean(); down=(-delta.clip(upper=0)).ewm(alpha=1/14,adjust=False).mean(); rsi=float((100-(100/(1+up/down))).iloc[-1])
    rv=float(vol.iloc[-1]/vol.tail(21).iloc[:-1].mean()) if vol.tail(21).iloc[:-1].mean()>0 else 1
    above50=c.iloc[-1]>sma50; above200=(sma200 is not None and c.iloc[-1]>sma200); aligned=(sma200 is not None and sma50>sma200)
    trend="BULLISH" if above50 and above200 and aligned else "UPTREND" if above50 else "NEUTRAL"
    value=(35 if trend=="BULLISH" else 18 if trend=="UPTREND" else 0)+min(max(r20,0),20)+min(max(r60,0),20)/2+min(max((rv-1)*10,0),10)+(10 if 50<=rsi<=70 else 3 if 45<=rsi<75 else 0)
    return dict(score=round(value,2),trend=trend,return_20d=round(r20,2),return_60d=round(r60,2),rsi=round(rsi,2),relative_volume=round(rv,2),close=round(float(c.iloc[-1]),2))
async def run(name):
    # Preserve order while preventing duplicate symbols from violating the
    # (universe, ticker) database constraint.
    symbols=list(dict.fromkeys(tickers(name))); results=[]
    for symbol in symbols:
        try:
            await refresh(symbol); out=score(await candles(symbol))
            if out: results.append((symbol,out))
        except Exception: continue
    async with Session() as s:
        await s.execute(delete(ScreenerResult).where(ScreenerResult.universe==name))
        for symbol,x in results: s.add(ScreenerResult(universe=name,ticker=symbol,updated_at=datetime.now(timezone.utc),**x))
        await s.commit()
    return {"universe":name,"processed":len(symbols),"ranked":len(results)}
async def results(name):
    async with Session() as s:
        rows=(await s.scalars(select(ScreenerResult).where(ScreenerResult.universe==name).order_by(ScreenerResult.score.desc()))).all()
        return [{"ticker":r.ticker,"score":r.score,"trend":r.trend,"return_20d":r.return_20d,"return_60d":r.return_60d,"rsi":r.rsi,"relative_volume":r.relative_volume,"close":r.close,"updated_at":r.updated_at.isoformat()} for r in rows]