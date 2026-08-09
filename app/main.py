import json, httpx
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI, HTTPException, Query
from fastapi.staticfiles import StaticFiles
from sqlalchemy import select
from pydantic import BaseModel
from .config import settings
from .db import Watchlist, Session, init_db
from .market import refresh, candles, search, info, provider
from .analysis import compute, persist, history, MIN_CANDLES
from .screener import universe_names, run, results
from . import sim

_STATIC_DIR = Path(__file__).resolve().parent.parent / "static"

@asynccontextmanager
async def lifespan(app):
    await init_db()
    async with Session() as s:
        for t in settings.watchlist.split(','):
            if not await s.get(Watchlist,t.strip().upper()): s.add(Watchlist(ticker=t.strip().upper()))
        await s.commit()
    if settings.sim_enabled:
        sim.start_scheduler()
    yield
    if settings.sim_enabled:
        sim.stop_scheduler()
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
@app.get('/api/info/{ticker}')
async def ticker_info(ticker:str):
    try: return {"ticker":ticker.upper(),"name":await info(ticker.upper())}
    except Exception: return {"ticker":ticker.upper(),"name":""}

@app.get('/api/symbols')
async def symbols(q:str=Query(min_length=2,max_length=80)):
    try: return await search(q)
    except ValueError as e: raise HTTPException(400,str(e))
    except Exception as e: raise HTTPException(502,f"{provider()} symbol search failed: {e}")
@app.post('/api/refresh/{ticker}')
async def fetch(ticker:str, period:str=None):
    try: await refresh(ticker.upper(), period); return {"ok":True}
    except Exception as e: raise HTTPException(400,str(e))
@app.get('/api/dashboard/{ticker}')
async def dashboard(ticker:str, period:str=None):
    # Always fetch the full cached dataset — indicators need >=206 candles
    all_rows = await candles(ticker.upper())
    candles_for_chart = list(all_rows)
    # Slice candles for the chart based on the selected range
    if period:
        period_counts = {"6m":126,"2y":504,"5y":1260,"10y":2520,"max":5000}
        count = period_counts.get(period)
        if count: candles_for_chart = candles_for_chart[-count:]
    try:
        r = compute(all_rows)
        await persist(ticker.upper(), r)
        r['history'] = await history(ticker.upper())
    except ValueError:
        # Not enough candles for full analysis — return chart with placeholder signal
        r = {
            "action": "N/A",
            "reason": f"Insufficient data for analysis ({len(all_rows)} candles, need {MIN_CANDLES}). Chart shown for reference only.",
            "snapshot": {},
            "strength": 0,
            "candles": [],
        }
        r['history'] = []
    r['candles'] = candles_for_chart
    return r
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
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: list[ChatMessage]

async def _stock_context(ticker: str) -> str:
    """Build a system prompt with deterministic stock data so the LLM has facts to chat about."""
    rows = await candles(ticker)
    try:
        r = compute(rows)
        return (
            f"You are a trading expert. You are chatting with a user about {ticker}. "
            f"Use the deterministic research data below as your only source of facts. "
            f"Ticker: {ticker}\n"
            f"Signal: {r['action']} (strength {r['strength']}/100)\n"
            f"Reason: {r['reason']}\n"
            f"Indicators: {json.dumps(r['snapshot'])}\n"
            f"Decision rules: BUY when close > SMA-50 > SMA-200, SMA-50 rising over 6 days, "
            f"RSI in a fresh cross above 50 or rising in the 50-70 band (not overbought <75), "
            f"MACD > MACD signal, and volume surge (>1.25× 20-day average). "
            f"SELL on early exit (close < SMA-50, MACD bearish, RSI breaks below 50) or "
            f"bearish trend (close < SMA-50 < SMA-200, SMA-50 falling). Otherwise HOLD."
        )
    except ValueError:
        return (
            f"You are a trading expert. You are chatting with a user about {ticker}. "
            f"There is currently insufficient historical data for a full technical analysis "
            f"({len(rows)} candles available, need {MIN_CANDLES}). "
            f"Be transparent about this limitation. You can discuss what you know, "
            f"but do not fabricate indicator values or signals."
        )

@app.post('/api/chat/{ticker}')
async def chat(ticker: str, req: ChatRequest):
    """Multi-turn chat about a stock. Pass the full message history; the system context is rebuilt each turn."""
    ticker = ticker.upper()
    if not req.messages:
        raise HTTPException(400, "messages must not be empty")
    # Sanitise: keep only the last 20 messages, only role/content, only known roles
    history_msgs = []
    for m in req.messages[-20:]:
        if m.role in ('user', 'assistant') and m.content.strip():
            history_msgs.append({"role": m.role, "content": m.content})
    if not history_msgs:
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
                    "messages": [{"role": "system", "content": system_prompt}] + history_msgs,
                    "stream": False,
                },
            )).json()
        text = out.get('message', {}).get('content', '') or 'No Ollama response'
        return {"text": text, "model": settings.ollama_model}
    except Exception as e:
        return {"text": f"Ollama unavailable: {e}", "model": settings.ollama_model}

# =====================================================================
# Autonomous paper-trading simulation endpoints
# =====================================================================

@app.get('/api/sim/status')
async def sim_status():
    """Portfolio snapshot: cash, positions, equity, P&L."""
    val = await sim.valuate()
    allowance_result = await sim.deposit_allowance()  # ensures account exists
    return {**val, "sim_enabled": settings.sim_enabled, "sim_strategy": settings.sim_strategy,
            "sim_universe": settings.sim_universe,
            "benchmark_enabled": settings.sim_benchmark_enabled,
            "benchmark_ticker": settings.sim_benchmark_ticker}

@app.get('/api/sim/trades')
async def sim_trades(limit: int = Query(default=100, ge=1, le=500)):
    """Trade log (most recent first)."""
    return await sim.get_trades(limit)

@app.get('/api/sim/equity')
async def sim_equity(limit: int = Query(default=365, ge=1, le=1000)):
    """Equity-curve snapshots for charting (oldest-first)."""
    return await sim.get_equity_curve(limit)

@app.get('/api/sim/allowances')
async def sim_allowances():
    """Monthly allowance deposit history."""
    return await sim.get_allowances()

@app.get('/api/sim/benchmark')
async def sim_benchmark():
    """DCA benchmark portfolio status + equity curve."""
    val = await sim.benchmark_valuate()
    curve = await sim.get_benchmark_equity_curve(365)
    return {**val, "equity_curve": curve}

@app.post('/api/sim/run')
async def sim_run():
    """Manually trigger a sim decision cycle."""
    try:
        return await sim.run_cycle()
    except Exception as e:
        raise HTTPException(500, f"Sim cycle failed: {e}")

@app.get('/api/sim/reasoning')
async def sim_reasoning():
    """Return the raw LLM reasoning text from the most recent sim cycle."""
    return {"reasoning": sim._last_llm_reasoning}

@app.post('/api/sim/reset')
async def sim_reset():
    """Wipe all sim tables and restart with start cash."""
    return await sim.reset_sim()

app.mount('/', StaticFiles(directory=str(_STATIC_DIR), html=True), name='static')