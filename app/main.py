import json, httpx
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Query
from fastapi.staticfiles import StaticFiles
from sqlalchemy import select
from pydantic import BaseModel
from .config import settings
from .db import Watchlist, Session, init_db
from .market import refresh, candles, search, provider
from .analysis import compute, persist, history
from .screener import universe_names, run, results
@asynccontextmanager
async def lifespan(app):
    await init_db()
    async with Session() as s:
        for t in settings.watchlist.split(','):
            if not await s.get(Watchlist,t.strip().upper()): s.add(Watchlist(ticker=t.strip().upper()))
        await s.commit()
    yield
app=FastAPI(title="Trade Sentinel",lifespan=lifespan)
@app.get('/healthz')
async def health(): return {"ok":True,"paper_trading":settings.paper_trading,"market_data_provider":provider()}
@app.get('/api/watchlist')
async def watchlist():
    async with Session() as s: return [x.ticker for x in (await s.scalars(select(Watchlist).order_by(Watchlist.ticker))).all()]
@app.post('/api/watchlist/{ticker}')
async def add(ticker:str):
    ticker=ticker.upper()
    async with Session() as s:
        if not await s.get(Watchlist,ticker): s.add(Watchlist(ticker=ticker)); await s.commit()
    return {"ticker":ticker}
@app.delete('/api/watchlist/{ticker}')
async def remove(ticker:str):
    async with Session() as s:
        x=await s.get(Watchlist,ticker.upper())
        if x: await s.delete(x); await s.commit()
    return {"ok":True}
@app.get('/api/symbols')
async def symbols(q:str=Query(min_length=2,max_length=80)):
    try: return await search(q)
    except ValueError as e: raise HTTPException(400,str(e))
    except Exception as e: raise HTTPException(502,f"{provider()} symbol search failed: {e}")
@app.post('/api/refresh/{ticker}')
async def fetch(ticker:str):
    try: await refresh(ticker.upper()); return {"ok":True}
    except Exception as e: raise HTTPException(400,str(e))
@app.get('/api/dashboard/{ticker}')
async def dashboard(ticker:str):
    rows=await candles(ticker.upper())
    try: r=compute(rows); await persist(ticker.upper(),r); r['history']=await history(ticker.upper()); return r
    except ValueError as e: raise HTTPException(400,str(e))
@app.get('/api/screener/universes')
async def universes(): return universe_names()
@app.post('/api/screener/run/{universe}')
async def screen_run(universe:str):
    try: return await run(universe)
    except ValueError as e: raise HTTPException(404,str(e))
@app.get('/api/screener/{universe}')
async def screen_results(universe:str):
    try:
        if universe not in universe_names(): raise ValueError('Unknown universe')
        return await results(universe)
    except ValueError as e: raise HTTPException(404,str(e))
@app.post('/api/explain/{ticker}')
async def explain(ticker:str):
    rows=await candles(ticker.upper())
    try: r=compute(rows)
    except ValueError as e: raise HTTPException(400,str(e))
    prompt=f"You are a cautious stock-research assistant. Explain this deterministic research signal in <=100 words. Never issue a recommendation or promise. Ticker {ticker.upper()}; signal {r['action']}; reason {r['reason']}; indicators {json.dumps(r['snapshot'])}. Mention it needs independent research."
    try:
        timeout = httpx.Timeout(
            connect=10.0,
            read=settings.ollama_timeout_seconds,
            write=30.0,
            pool=10.0,
        )
        async with httpx.AsyncClient(timeout=timeout) as c: out=(await c.post(settings.ollama_url.rstrip('/')+'/api/generate',json={"model":settings.ollama_model,"prompt":prompt,"stream":False})).json()
        return {"text":out.get('response','No Ollama response'),"model":settings.ollama_model}
    except Exception as e: return {"text":f"Ollama unavailable: {e}","model":settings.ollama_model}

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: list[ChatMessage]

async def _stock_context(ticker: str) -> str:
    """Build a system prompt with deterministic stock data so the LLM has facts to chat about."""
    rows = await candles(ticker)
    r = compute(rows)
    return (
        f"You are a trading expert. You are chatting with a user about {ticker}. "
        f"Use the deterministic research data below as your only source of facts. "
        f"Ticker: {ticker}\n"
        f"Signal: {r['action']}\n"
        f"Reason: {r['reason']}\n"
        f"Indicators: {json.dumps(r['snapshot'])}\n"
    )

@app.post('/api/chat/{ticker}')
async def chat(ticker: str, req: ChatRequest):
    """Multi-turn chat about a stock. Pass the full message history; the system context is rebuilt each turn."""
    ticker = ticker.upper()
    if not req.messages:
        raise HTTPException(400, "messages must not be empty")
    # Sanitise: keep only the last 20 messages, only role/content, only known roles
    history = []
    for m in req.messages[-20:]:
        if m.role in ('user', 'assistant') and m.content.strip():
            history.append({"role": m.role, "content": m.content})
    if not history:
        raise HTTPException(400, "no valid messages")
    system_prompt = await _stock_context(ticker)
    try:
        timeout = httpx.Timeout(
            connect=10.0,
            read=settings.ollama_timeout_seconds,
            write=30.0,
            pool=10.0,
        )
        async with httpx.AsyncClient(timeout=timeout) as c:
            out = (await c.post(
                settings.ollama_url.rstrip('/') + '/api/chat',
                json={
                    "model": settings.ollama_model,
                    "messages": [{"role": "system", "content": system_prompt}] + history,
                    "stream": False,
                },
            )).json()
        text = out.get('message', {}).get('content', '') or 'No Ollama response'
        return {"text": text, "model": settings.ollama_model}
    except Exception as e:
        return {"text": f"Ollama unavailable: {e}", "model": settings.ollama_model}

app.mount('/', StaticFiles(directory='static', html=True), name='static')
