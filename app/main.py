import json, logging
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
from .screener import universe_names, run, results, refresh_incremental, load_deep_history
from . import sim
from . import news as news_mod
from . import llm as llm_mod

_STATIC_DIR = Path(__file__).resolve().parent.parent / "static"
logger = logging.getLogger("trade_sentinel.main")

@asynccontextmanager
async def lifespan(app):
    await init_db()
    async with Session() as s:
        for t in settings.watchlist.split(','):
            ticker = t.strip().upper()
            if not ticker:
                continue
            if not await s.get(Watchlist, ticker):
                s.add(Watchlist(ticker=ticker))
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
    except Exception as e:
        logger.debug("info lookup failed for %s: %s", ticker.upper(), e)
        return {"ticker":ticker.upper(),"name":""}

@app.get('/api/symbols')
async def symbols(q:str=Query(min_length=2,max_length=80)):
    try: return await search(q)
    except ValueError as e: raise HTTPException(400,str(e))
    except Exception as e: raise HTTPException(502,f"{provider()} symbol search failed: {e}")
@app.post('/api/refresh/{ticker}')
async def fetch(ticker:str, period:str=None):
    try: await refresh(ticker.upper(), period); return {"ok":True}
    except Exception as e: raise HTTPException(400,str(e))

@app.post('/api/refresh-watchlist')
async def refresh_watchlist(period: str = "2y"):
    """Refresh candle data for every watchlist ticker. Returns per-ticker status."""
    async with Session() as s:
        tickers = [x.ticker for x in (await s.scalars(select(Watchlist).order_by(Watchlist.ticker))).all()]
    refreshed, errors = [], []
    for t in tickers:
        try:
            await refresh(t, period); refreshed.append(t)
        except Exception as e:
            errors.append(f"{t}: {e}"); logger.warning("watchlist refresh %s failed: %s", t, e)
    return {"refreshed": refreshed, "errors": errors, "total": len(tickers)}

@app.post('/api/sim/refresh')
async def sim_refresh(period: str = "2y"):
    """Refresh candle data for all sim holdings + benchmark so valuations use live prices."""
    tickers = await sim.held_tickers()
    refreshed, errors = [], []
    for t in tickers:
        try:
            await refresh(t, period); refreshed.append(t)
        except Exception as e:
            errors.append(f"{t}: {e}"); logger.warning("sim refresh %s failed: %s", t, e)
    return {"refreshed": refreshed, "errors": errors, "total": len(tickers)}
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
async def list_universes(): return universe_names()
@app.post('/api/screener/run/{universe}')
async def screen_run(universe:str):
    try: return await run(universe)
    except ValueError as e: raise HTTPException(404,str(e))
@app.post('/api/screener/refresh/{universe}')
async def screen_refresh(universe: str):
    """Incrementally refresh candle data — only fetches tickers with missing or stale data."""
    try: return await refresh_incremental(universe)
    except ValueError as e: raise HTTPException(404, str(e))
@app.post('/api/screener/load_deep/{universe}')
async def screen_load_deep(universe: str, period: str = '10y'):
    """Fetch deep history (default 10y) for all tickers — for optimization/backtest."""
    try: return await load_deep_history(universe, period)
    except ValueError as e: raise HTTPException(404, str(e))
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
    headlines = await news_mod.search_ticker_news(ticker)
    news_section = ""
    if headlines:
        news_lines = "\n".join(f"- {h['title']}" for h in headlines[:3])
        news_section = f"\nRecent news (supplementary context):\n{news_lines}\n"
    try:
        r = compute(rows)
        return (
            f"You are a trading expert. You are chatting with a user about {ticker}. "
            f"Use the deterministic research data below as your only source of facts. "
            f"Ticker: {ticker}\n"
            f"Signal: {r['action']} (strength {r['strength']}/100)\n"
            f"Reason: {r['reason']}\n"
            f"Indicators: {json.dumps(r['snapshot'])}\n"
            f"Decision rules: a weighted technical score (net_score, range -100..+100) "
            f"combines trend alignment (close vs SMA-50 vs SMA-200, gated by ADX>25 so "
            f"choppy markets earn less), SMA-50 slope, MACD histogram momentum (rising "
            f"histogram, not just crossover), RSI pullback entry (RSI rising from 40-55 "
            f"rewarded, RSI>70 overbought penalized), distance from SMA-200, and volume. "
            f"BUY when net_score >= +40 AND close > SMA-50 > SMA-200 AND price is >2% "
            f"above SMA-200 AND the weekly trend is up (weekly close > weekly SMA-50). "
            f"SELL when net_score <= -40 AND close < SMA-50 < SMA-200 AND price is >2% "
            f"below SMA-200. Otherwise HOLD. Volume surge is a bonus, not a requirement."
            f"{news_section}"
        )
    except ValueError:
        return (
            f"You are a trading expert. You are chatting with a user about {ticker}. "
            f"There is currently insufficient historical data for a full technical analysis "
            f"({len(rows)} candles available, need {MIN_CANDLES}). "
            f"Be transparent about this limitation. You can discuss what you know, "
            f"but do not fabricate indicator values or signals."
            f"{news_section}"
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
        out = await llm_mod.chat([{"role": "system", "content": system_prompt}] + history_msgs)
        return {"text": out["text"], "model": out["model"] or await llm_mod.current_model_label()}
    except Exception as e:
        logger.warning("Chat LLM call failed for %s: %s", ticker, e)
        return {"text": f"LLM unavailable: {e}", "model": await llm_mod.current_model_label()}

# =====================================================================
# LLM backend / model management
# =====================================================================

class LLMSelectRequest(BaseModel):
    backend: str
    model: str | None = None

@app.get('/api/llm/status')
async def llm_status():
    """List all configured LLM backends, the models each serves, and the active one."""
    backends = await llm_mod.list_backends()
    active = await llm_mod.current_backend()
    return {
        "backends": backends,
        "active_backend": active.get("name", ""),
        "active_model": active.get("model", ""),
    }

@app.post('/api/llm/select')
async def llm_select(req: LLMSelectRequest):
    """Switch the active LLM backend and optionally the model within it
    (persisted across restarts)."""
    try:
        match = await llm_mod.select_backend(req.backend, req.model)
    except ValueError as e:
        raise HTTPException(404, str(e))
    return {
        "ok": True,
        "backend": match["name"],
        "model": match["model"],
    }

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
    """Return structured LLM reasoning summary from the most recent sim cycle."""
    return sim.get_last_llm_summary()

@app.get('/api/sim/raw_reasoning')
async def sim_raw_reasoning():
    """Return the raw LLM reasoning text (for debugging/full text view)."""
    return {"reasoning": sim.get_last_llm_reasoning()}

@app.post('/api/sim/chat')
async def sim_chat_endpoint(req: ChatRequest):
    """Interactive chat with the sim portfolio manager LLM.

    The server keeps the full conversation history in the DB, so the client
    only needs to send the new user message for this turn. If the LLM proposes
    actions, they are executed immediately.
    """
    if not req.messages:
        raise HTTPException(400, "messages must not be empty")
    # Send only this turn's new messages to sim_chat — it merges them with
    # the persisted history itself.
    new_msgs = []
    for m in req.messages[-1:]:
        if m.role in ('user', 'assistant') and m.content.strip():
            new_msgs.append({"role": m.role, "content": m.content})
    if not new_msgs:
        raise HTTPException(400, "no valid messages")
    return await sim.sim_chat(new_msgs)


@app.get('/api/sim/chat/history')
async def sim_chat_history():
    """Return the persisted sim portfolio-manager conversation history."""
    return await sim.get_chat_history()


@app.delete('/api/sim/chat/history')
async def sim_chat_history_clear():
    """Clear the persisted sim portfolio-manager conversation history."""
    await sim.clear_chat_history()
    return {"ok": True}


@app.post('/api/sim/reset')
async def sim_reset():
    """Wipe all sim tables and restart with start cash."""
    return await sim.reset_sim()

app.mount('/', StaticFiles(directory=str(_STATIC_DIR), html=True), name='static')