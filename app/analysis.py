import json, pandas as pd
from sqlalchemy import desc, select
from .db import Signal, Session

def compute(rows):
    if len(rows) < 16: raise ValueError("Need at least 16 daily candles")
    d=pd.DataFrame(rows); c=d.close
    d["sma20"]=c.rolling(20).mean(); d["sma50"]=c.rolling(50).mean(); d["sma200"]=c.rolling(200).mean()
    delta=c.diff(); up=delta.clip(lower=0).ewm(alpha=1/14,adjust=False).mean(); down=(-delta.clip(upper=0)).ewm(alpha=1/14,adjust=False).mean(); d["rsi"]=100-(100/(1+up/down))
    d["macd"]=c.ewm(span=12,adjust=False).mean()-c.ewm(span=26,adjust=False).mean(); d["macd_signal"]=d.macd.ewm(span=9,adjust=False).mean()
    prev=c.shift(); tr=pd.concat([d.high-d.low,(d.high-prev).abs(),(d.low-prev).abs()],axis=1).max(axis=1); d["atr14"]=tr.rolling(14).mean()
    x=d.iloc[-1]; bullish=x.close>x.sma50 and x.sma50>x.sma200 and x.sma50>d.sma50.iloc[-6]
    recovery=x.rsi>50 and d.rsi.iloc[-2]<=50 and x.macd>x.macd_signal
    bearish=x.close<x.sma50 and x.sma50<x.sma200
    action="BUY" if bullish and recovery else "SELL" if bearish else "HOLD"
    reason=("Bullish trend alignment and RSI recovery with MACD confirmation" if action=="BUY" else "Bearish trend break: price below SMA-50 and SMA-50 below SMA-200" if action=="SELL" else "No complete entry or exit setup; wait for the defined rules")
    def norm(v):
        try: v=float(v)
        except Exception: return None
        return None if pd.isna(v) else round(v,2)
    snap={k:norm(x[k]) for k in ["close","sma20","sma50","sma200","rsi","macd","macd_signal","atr14"]}
    return {"action":action,"reason":reason,"snapshot":snap,"candles":rows}
async def persist(ticker,result):
    async with Session() as s: s.add(Signal(ticker=ticker,action=result["action"],reason=result["reason"],snapshot=json.dumps(result["snapshot"]))); await s.commit()
async def history(ticker):
    async with Session() as s:
        r=(await s.scalars(select(Signal).where(Signal.ticker==ticker).order_by(desc(Signal.created_at)).limit(20))).all()
        return [{"at":x.created_at.isoformat(),"action":x.action,"reason":x.reason,"snapshot":json.loads(x.snapshot)} for x in r]
