import json, httpx
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from sqlalchemy import select
from .config import settings
from .db import Watchlist, Session, init_db
from .market import refresh, candles
from .analysis import compute, persist, history
@asynccontextmanager
async def lifespan(app):
    await init_db()
    async with Session() as s:
        for t in settings.watchlist.split(','): s.add(Watchlist(ticker=t.strip().upper()))
        try: await s.commit()
        except: await s.rollback()
    yield
app=FastAPI(title="Trade Sentinel",lifespan=lifespan)
@app.get('/healthz')
async def health(): return {"ok":True,"paper_trading":settings.paper_trading}
@app.get('/api/watchlist')
async def watchlist():
    async with Session() as s: return [x.ticker for x in (await s.scalars(select(Watchlist).order_by(Watchlist.ticker))).all()]
@app.post('/api/watchlist/{ticker}')
async def add(ticker:str):
    async with Session() as s: s.add(Watchlist(ticker=ticker.upper())); await s.commit()
    return {"ticker":ticker.upper()}
@app.delete('/api/watchlist/{ticker}')
async def remove(ticker:str):
    async with Session() as s:
        x=await s.get(Watchlist,ticker.upper())
        if x: await s.delete(x); await s.commit()
    return {"ok":True}
@app.post('/api/refresh/{ticker}')
async def fetch(ticker:str):
    try: await refresh(ticker.upper()); return {"ok":True}
    except Exception as e: raise HTTPException(400,str(e))
@app.get('/api/dashboard/{ticker}')
async def dashboard(ticker:str):
    rows=await candles(ticker.upper())
    try: r=compute(rows); await persist(ticker.upper(),r); r['history']=await history(ticker.upper()); return r
    except ValueError as e: raise HTTPException(400,str(e))
@app.post('/api/explain/{ticker}')
async def explain(ticker:str):
    rows=await candles(ticker.upper())
    try: r=compute(rows)
    except ValueError as e: raise HTTPException(400,str(e))
    prompt=f"You are a cautious stock-research assistant. Explain this deterministic research signal in <=100 words. Never issue a recommendation or promise. Ticker {ticker.upper()}; signal {r['action']}; reason {r['reason']}; indicators {json.dumps(r['snapshot'])}. Mention it needs independent research."
    try:
        async with httpx.AsyncClient(timeout=45) as c: out=(await c.post(settings.ollama_url.rstrip('/')+'/api/generate',json={"model":settings.ollama_model,"prompt":prompt,"stream":False})).json()
        return {"text":out.get('response','No Ollama response'),"model":settings.ollama_model}
    except Exception as e: return {"text":f"Ollama unavailable: {e}","model":settings.ollama_model}
app.mount('/',StaticFiles(directory='static',html=True),name='static')
