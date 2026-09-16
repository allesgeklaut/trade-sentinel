import asyncio, json, logging, re
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, UTC
from pathlib import Path
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy import select
from pydantic import BaseModel
from .config import settings
from .db import Watchlist, Session, init_db
from .market import refresh, refresh_many, refresh_yfinance, candles, search, info, provider, PERIOD_COUNTS
from .analysis import compute, persist, history, MIN_CANDLES
from .screener import universe_names, run, results, refresh_incremental, load_deep_history, get_screener_progress, ScreenerBusy
from . import sim
from . import news as news_mod
from . import llm as llm_mod

_STATIC_DIR = Path(__file__).resolve().parent.parent / "static"
logger = logging.getLogger("trade_sentinel.main")
_WATCHLIST_REFRESH_PERIOD = "2y"
_watchlist_prefetch_task: asyncio.Task | None = None

# Uvicorn installs no handlers for the root logger, so app loggers
# ("trade_sentinel.*") emit nothing at INFO: scheduler runs, allowance
# deposits and cycle completions were all invisible in `docker compose logs`.
# Route app logs to stdout at INFO (uvicorn.access stays as configured by
# uvicorn itself). Idempotent: guard so --reload / test re-imports don't
# duplicate handlers.
if not logging.getLogger().handlers:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    # httpx logs every request URL at INFO, which would echo provider API keys
    # (and other query params) into the container logs. We use header auth, but
    # keep this muted as defense-in-depth.
    logging.getLogger("httpx").setLevel(logging.WARNING)

async def _watchlist_tickers() -> list[str]:
    async with Session() as s:
        return [x.ticker for x in (await s.scalars(select(Watchlist).order_by(Watchlist.ticker))).all()]


async def prefetch_watchlist() -> dict:
    """Fetch the watchlist's candles into the cache via the configured provider.

    App-level and nightly: the Dashboard then opens straight from the DB with no
    network fetch. Shares the Twelve Data limiter/budget with the universe
    prefetch, and skips tickers fetched within ``market_fresh_seconds`` — so the
    usual overlap (mega-caps also in the S&P 500) costs nothing extra.
    """
    tickers = await _watchlist_tickers()
    if not tickers:
        return {"ok": True, "total": 0, "refreshed": 0, "errors": []}
    refreshed, errors = await refresh_many(
        tickers, _WATCHLIST_REFRESH_PERIOD, use_provider=True,
        max_age_seconds=settings.market_fresh_seconds)
    logger.info("Watchlist prefetch: %d/%d fetched, %d errors",
                len(refreshed), len(tickers), len(errors))
    return {"ok": True, "total": len(tickers), "refreshed": len(refreshed), "errors": errors}


async def _watchlist_prefetch_loop() -> None:
    """Run :func:`prefetch_watchlist` nightly at ``sim_run_hour`` (UTC).

    Deliberately independent of ``SIM_ENABLED`` / ``SIM_UNIVERSE_PREFETCH``: the
    watchlist is a Dashboard feature, not a sim feature. Scheduled after the
    universe prefetch window so the two don't contend for the rate limiter.
    Skipped on weekends/NYSE holidays (EOD data doesn't change then).
    """
    while True:
        now = datetime.now(UTC)
        target = now.replace(hour=settings.sim_run_hour, minute=settings.sim_run_minute,
                             second=0, microsecond=0)
        if target <= now:
            target += timedelta(days=1)
        wait_seconds = (target - now).total_seconds()
        logger.info("Watchlist prefetch: next run at %s (in %.0f seconds)", target, wait_seconds)
        await asyncio.sleep(wait_seconds)
        if not sim.is_trading_day(datetime.now(UTC)):
            logger.info("Watchlist prefetch: %s is not a trading day — skipping",
                        datetime.now(UTC).date())
            continue
        try:
            await prefetch_watchlist()
        except Exception as e:
            logger.error("Watchlist prefetch failed: %s", e, exc_info=True)


@asynccontextmanager
async def lifespan(app):
    global _watchlist_prefetch_task
    await init_db()
    async with Session() as s:
        for t in settings.watchlist.split(','):
            ticker = t.strip().upper()
            if not ticker:
                continue
            if not await s.get(Watchlist, ticker):
                s.add(Watchlist(ticker=ticker))
        await s.commit()
    if settings.watchlist_prefetch:
        _watchlist_prefetch_task = asyncio.create_task(_watchlist_prefetch_loop())
    if settings.sim_enabled:
        sim.start_scheduler()
    yield
    if _watchlist_prefetch_task and not _watchlist_prefetch_task.done():
        _watchlist_prefetch_task.cancel()
    _watchlist_prefetch_task = None
    if settings.sim_enabled:
        sim.stop_scheduler()
app=FastAPI(title="Trade Sentinel",lifespan=lifespan)
_TICKER_RE = re.compile(r"^[A-Z0-9.\-^=]{1,32}$")

@app.middleware("http")
async def security_headers(request, call_next):
    response = await call_next(request)
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("X-Frame-Options", "DENY")
    response.headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
    return response

@app.get('/healthz')
async def health(): return {"ok":True,"paper_trading":settings.paper_trading,"market_data_provider":provider()}
@app.get('/api/watchlist')
async def watchlist():
    return await _watchlist_tickers()
@app.post('/api/watchlist/{ticker}')
async def add(ticker:str):
    ticker=ticker.upper()
    if not _TICKER_RE.match(ticker):
        raise HTTPException(422, f"Invalid ticker symbol: {ticker!r}")
    async with Session() as s:
        if not await s.get(Watchlist,ticker): s.add(Watchlist(ticker=ticker)); await s.commit()
    # Best-effort initial fetch so the new ticker charts immediately instead of
    # waiting for a pull-to-refresh or the nightly prefetch. One Yahoo call.
    try:
        await refresh_yfinance(ticker, _WATCHLIST_REFRESH_PERIOD)
        fetched = True
    except Exception as e:
        fetched = False
        logger.warning("watchlist add: initial fetch failed for %s: %s", ticker, e)
    return {"ticker":ticker,"fetched":fetched}
@app.delete('/api/watchlist/{ticker}')
async def remove(ticker:str):
    ticker=ticker.upper()
    if not _TICKER_RE.match(ticker):
        raise HTTPException(422, f"Invalid ticker symbol: {ticker!r}")
    async with Session() as s:
        x=await s.get(Watchlist,ticker)
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
    except ValueError as e: raise HTTPException(400,str(e)) from e
    except Exception as e: raise HTTPException(502,f"{provider()} symbol search failed: {e}") from e
@app.post('/api/refresh/{ticker}')
async def fetch(ticker:str, period:str|None=None):
    try: await refresh(ticker.upper(), period or "2y"); return {"ok":True}
    except Exception as e: raise HTTPException(400,str(e)) from e

@app.post('/api/refresh-watchlist')
async def refresh_watchlist(period: str = "2y"):
    """Refresh candle data for every watchlist ticker. Returns per-ticker status.

    Uses the bulk Yahoo path (``refresh_many`` default), NOT the metered
    provider: pull-to-refresh must be fast and free.
    """
    tickers = await _watchlist_tickers()
    refreshed, errors = await refresh_many(tickers, period)
    return {"refreshed": refreshed, "errors": errors, "total": len(tickers)}

@app.post('/api/sim/refresh')
async def sim_refresh(period: str = "2y"):
    """Refresh candle data for all sim holdings + benchmark so valuations use live prices.

    Pull-to-refresh is a user gesture and must feel fast: this uses the free
    Yahoo bulk path (unmetered), NOT the rate-limited provider. With the 8/min
    Twelve Data plan, >8 holdings made one pull take a full minute.
    """
    tickers = await sim.held_tickers()
    refreshed, errors = await refresh_many(tickers, period)
    return {"refreshed": refreshed, "errors": errors, "total": len(tickers)}
@app.get('/api/dashboard/{ticker}')
async def dashboard(ticker:str, period:str|None=None):
    # Always fetch the full cached dataset — indicators need >=206 candles
    all_rows = await candles(ticker.upper())
    candles_for_chart = list(all_rows)
    # Slice candles for the chart based on the selected range
    if period:
        count = PERIOD_COUNTS.get(period)
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
    except ScreenerBusy as e: raise HTTPException(409, str(e)) from e
    except ValueError as e: raise HTTPException(404,str(e)) from e
@app.post('/api/screener/refresh/{universe}')
async def screen_refresh(universe: str):
    """Incrementally refresh candle data — only fetches tickers with missing or stale data."""
    try: return await refresh_incremental(universe)
    except ScreenerBusy as e: raise HTTPException(409, str(e)) from e
    except ValueError as e: raise HTTPException(404, str(e)) from e
@app.post('/api/screener/load_deep/{universe}')
async def screen_load_deep(universe: str, period: str = '10y'):
    """Fetch deep history (default 10y) for all tickers — for optimization/backtest."""
    try: return await load_deep_history(universe, period)
    except ScreenerBusy as e: raise HTTPException(409, str(e)) from e
    except ValueError as e: raise HTTPException(404, str(e)) from e
@app.get('/api/screener/status')
async def screener_status():
    """Current/last screener operation progress for the frontend poller.

    Returns {running, op, universe, current, done, total, started_at,
    updated_at, error}. Polled by the screener buttons while an update /
    refresh / deep-load is in flight.
    """
    return get_screener_progress()

@app.get('/api/screener/{universe}')
async def screen_results(universe:str):
    try:
        if universe not in universe_names(): raise ValueError('Unknown universe')
        return await results(universe)
    except ValueError as e: raise HTTPException(404,str(e)) from e
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


async def _sse_event(data: dict) -> str:
    """Serialize ``data`` as a single Server-Sent Events ``data:`` line."""
    return "data: " + json.dumps(data) + "\n\n"


@app.post('/api/chat/{ticker}/stream')
async def chat_stream(ticker: str, req: ChatRequest):
    """SSE stream of a multi-turn chat about a stock.

    Emits ``data: {"type":"delta","text":"..."}`` chunks as the LLM produces
    tokens, then a final ``data: {"type":"done","text":full,"model":...}``.
    """
    ticker = ticker.upper()
    if not req.messages:
        raise HTTPException(400, "messages must not be empty")
    history_msgs = []
    for m in req.messages[-20:]:
        if m.role in ('user', 'assistant') and m.content.strip():
            history_msgs.append({"role": m.role, "content": m.content})
    if not history_msgs:
        raise HTTPException(400, "no valid messages")
    system_prompt = await _stock_context(ticker)

    async def gen():
        try:
            async for evt in llm_mod.chat_stream(
                [{"role": "system", "content": system_prompt}] + history_msgs
            ):
                yield await _sse_event(evt)
        except Exception as e:
            yield await _sse_event({"type": "error", "text": f"{type(e).__name__}: {e}"})

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # disable proxy buffering (nginx)
        },
    )

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
        raise HTTPException(404, str(e)) from e
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
    await sim.deposit_allowance()  # ensures account exists
    return {**val, "sim_enabled": settings.sim_enabled, "sim_strategy": settings.sim_strategy,
            "sim_universe": sim._universe(),
            "benchmark_enabled": settings.sim_benchmark_enabled,
            "benchmark_ticker": settings.sim_benchmark_ticker}

@app.get('/api/sim/trades')
async def sim_trades(limit: int = Query(default=100, ge=1, le=500)):
    """Trade log (most recent first)."""
    return await sim.get_trades(limit)

@app.get('/api/sim/equity')
async def sim_equity(limit: int = Query(default=365, ge=1, le=12000)):
    """Equity-curve snapshots for charting (oldest-first)."""
    return await sim.get_equity_curve(limit)

@app.get('/api/sim/allowances')
async def sim_allowances():
    """Monthly allowance deposit history."""
    return await sim.get_allowances()

@app.get('/api/sim/benchmark')
async def sim_benchmark(limit: int = Query(default=365, ge=1, le=12000)):
    """DCA benchmark portfolio status + equity curve."""
    val = await sim.benchmark_valuate()
    curve = await sim.get_benchmark_equity_curve(limit)
    return {**val, "equity_curve": curve}

@app.post('/api/sim/run')
async def sim_run():
    """Manually trigger a sim decision cycle.

    Fire-and-forget: the cycle runs as a background asyncio task and this
    endpoint returns immediately with {started: true/false}. The cycle
    survives browser disconnects because it's not tied to the HTTP request.
    Track progress via /api/sim/run-status.
    """
    result = sim.start_run_cycle_background()
    if not result.get("started"):
        raise HTTPException(409, result.get("reason", "already running"))
    return result

@app.get('/api/sim/run-status')
async def sim_run_status():
    """Current/last run-cycle progress for the frontend status poller.

    Returns {running, stage, detail, started_at, updated_at, error}.
    Polled every ~1s by the Run Bot Now button while a cycle is in flight.
    """
    return sim.get_run_progress()

@app.post('/api/sim/backfill')
async def sim_backfill(start: str | None = Query(default=None)):
    """Backfill the daily sim with a DETERMINISTIC synthetic history.

    Replays the engine's own rules (no LLM calls — the hybrid live strategy's
    LLM overlay cannot be replayed) over stored candles/fundamentals and
    REPLACES the portfolio state with the replay's end state. An
    approximation by design; the response carries approximation: true.
    """
    from . import sim
    r = await sim.backfill(start)
    if r.get("skipped"):
        raise HTTPException(409, r.get("reason", "skipped"))
    if not r.get("ok"):
        raise HTTPException(400, r.get("error", "backfill failed"))
    return r

@app.post('/api/sim/backfill-benchmark')
async def sim_backfill_benchmark(start: str | None = Query(default=None)):
    """Backfill the DCA benchmark curve (URTH $1000/month) over historical
    candles. The twin of the other backfills so all four curves can cover
    the same window. ``start``: date | "all" | default = synced."""
    from . import sim
    r = await sim.backfill_benchmark(start)
    if not r.get("ok"):
        raise HTTPException(400, r.get("error", "backfill failed"))
    return r

@app.post('/api/backfill-all')
async def backfill_all(start: str | None = Query(default=None)):
    """Backfill ALL FOUR portfolios over the SAME window — the shared-range
    button. Resolves the start ONCE (explicit "YYYY-MM-DD", "all" for the
    full history, or default = the earliest snapshot across the other
    portfolios so the curves stay synced), then runs every backfill with
    that same window:

      daily sim (deterministic approximation) · monthly qv-mom ·
      daily-core (selected variant) · DCA benchmark (URTH).

    Each backfill replaces its portfolio's state. Runs the slowest first
    (daily sim ~minutes) so an early failure doesn't leave the others
    wiped-then-unfilled; results are reported per portfolio. This is a
    long operation (several minutes) — the caller gets everything in one
    response when it finishes.
    """
    from . import daily_core, monthly, sim

    # Resolve the shared window ONCE. The per-backfill defaults would each
    # pick their own sync date (and wipe their own snapshots mid-run), so
    # the coordinator pins one start for everyone.
    #
    # "all" is passed through as the literal "all" — each backfill maps it to
    # ITS own full stored history (None internally). Passing None here would
    # mean "synced window" to every backfill, so the full-history button would
    # silently replay the short synced range instead (the bug this fixes).
    if start == "all":
        effective = "all"
        start_note = "full stored history"
    elif start is None:
        resolved = await daily_core._sync_start_date()
        effective = resolved
        start_note = resolved or "first eligible window"
    else:
        effective = start
        start_note = start

    # The monthly and daily-core replays run on the same universe, window and
    # momentum variant, so they share one preload: the fundamentals and the
    # per-month-end eligibility frames (the expensive part) are built once and
    # reused. daily_sim runs first (slowest, and it doesn't use qv-mom frames).
    # Runs are sequential and each backfill acquires its own lock, so the
    # nightly cycle can't interleave with the multi-portfolio wipe/replay.
    out: dict[str, dict] = {}
    errors: dict[str, str] = {}
    preload: dict = {}

    async def _monthly_backfill():
        # preload_backfill takes a real date or None (= load everything);
        # backfill() takes "all" as its full-history sentinel.
        preload_start = None if effective == "all" else effective
        preload["v"] = await monthly.preload_backfill(preload_start)
        return await monthly.backfill(effective, preload=preload["v"])

    async def _daily_core_backfill():
        return await daily_core.backfill(effective, preload=preload.get("v"))

    for name, fn in (
        ("daily_sim", lambda: sim.backfill(effective)),
        ("monthly", _monthly_backfill),
        ("daily_core", _daily_core_backfill),
        ("benchmark", lambda: sim.backfill_benchmark(effective)),
    ):
        try:
            out[name] = await fn()
        except Exception as e:
            logger.exception("backfill-all: %s failed", name)
            errors[name] = str(e)

    return {"ok": not errors, "start": start_note,
            "requested_start": start or "synced (earliest of the other sims)",
            "results": out, "errors": errors}

@app.get('/api/sim/reasoning')
async def sim_reasoning():
    """Return structured LLM reasoning summary from the most recent sim cycle."""
    return sim.get_last_llm_summary()

@app.get('/api/sim/raw_reasoning')
async def sim_raw_reasoning():
    """Return the raw LLM reasoning text (for debugging/full text view)."""
    return {"reasoning": sim.get_last_llm_reasoning()}

# =====================================================================
# Monthly qv-mom portfolio endpoints (separate paper portfolio)
# =====================================================================

@app.get('/api/monthly/status')
async def monthly_status():
    """Monthly portfolio snapshot: cash, positions, equity, config."""
    from . import monthly
    from . import daily_core
    # Deposit the allowance when a new month has begun (start-of-month, same
    # timing as the sim portfolio) so viewing the tab reflects the deposit
    # immediately instead of waiting for the nightly scheduler pass. Skipped
    # when the monthly portfolio is disabled — a tab visit must not fund it.
    if settings.sim_monthly_enabled:
        await monthly.deposit_allowance()
    val = await monthly.monthly_valuate()
    return {**val,
            "sim_monthly_enabled": settings.sim_monthly_enabled,
            "sim_monthly_universe": daily_core.current_universe(),
            "sim_monthly_contribution": settings.sim_monthly_contribution,
            "sim_monthly_target_n": settings.sim_monthly_target_n,
            "next_rebalance": monthly.month_last_trading_day(datetime.now(UTC)).strftime("%Y-%m-%d")}

@app.get('/api/monthly/trades')
async def monthly_trades(limit: int = Query(default=100, ge=1, le=500)):
    """Monthly portfolio trade log (most recent first)."""
    from . import monthly
    return await monthly.get_trades(limit)

@app.get('/api/monthly/equity')
async def monthly_equity(limit: int = Query(default=365, ge=1, le=12000)):
    """Monthly portfolio equity-curve snapshots (oldest-first)."""
    from . import monthly
    return await monthly.get_equity_curve(limit)

@app.post('/api/monthly/backfill')
async def monthly_backfill(start: str | None = Query(default=None)):
    """Backfill the monthly portfolio with synthetic history.

    Replays the qv-mom strategy over stored candles/fundamentals to today
    and REPLACES the portfolio state with the replay's end state — the
    monthly twin of /api/dailycore/backfill, so all three equity curves can
    cover the same (full) window for comparison.

    ``start``: "YYYY-MM-DD" to replay from that date, "all" for the full
    stored history, or omit (default) to synch with the daily sim's
    earliest snapshot.
    """
    from . import monthly
    return await monthly.backfill(start)


@app.get('/api/monthly/rebalances')
async def monthly_rebalances(limit: int = Query(default=24, ge=1, le=120)):
    """Rebalance decision log: held before, picked, #new per month."""
    from . import monthly
    return await monthly.get_rebalances(limit)

@app.post('/api/monthly/run')
async def monthly_run():
    """Manually trigger a monthly rebalance (idempotent per month)."""
    from . import monthly
    result = await monthly.run_rebalance()
    if result.get("skipped"):
        return result
    return result

@app.post('/api/monthly/refresh')
async def monthly_refresh():
    """Refresh candle data for the monthly portfolio's holdings (+ FX pairs)
    so valuation/equity use current prices, without running a rebalance."""
    from . import monthly
    from sqlalchemy import select
    async with monthly.Session() as s:
        held = [p.ticker for p in (await s.scalars(select(monthly.MonthlyPosition))).all()]
    pairs = sorted({pm[0] for t in held if (pm := monthly.fundamentals_mod._suffix_fx(t))})
    # Held-only pull-to-refresh: free Yahoo bulk path (fast, unmetered). FX pairs
    # are Yahoo symbols (EURUSD=X) anyway, so the provider path just 404'd them.
    refreshed, errors = await refresh_many(held + pairs, "2y")
    return {"refreshed": refreshed, "errors": errors}

@app.get('/api/dailycore/status')
async def daily_core_status():
    """Daily-core portfolio status: valuation + config + today's ranking.

    Reads the stored daily ranking (computed once per day by the cycle);
    recomputes on demand only when no fresh one exists (first run after
    deploy, or the cycle hasn't fired yet today).
    """
    from . import daily_core
    if settings.sim_daily_core_enabled:
        await daily_core.deposit_allowance()
    val = await daily_core.valuate()
    band, picks, _frame = await daily_core.compute_targets()
    return {
        "valuation": val,
        "sim_daily_core_enabled": settings.sim_daily_core_enabled,
        "band": band,
        "picks": picks,
        "ranking_date": await daily_core.get_ranking_date(),
        # rank map {ticker: 1-based rank} so the UI can show each holding's
        # position in the ranking, not just the bare band list.
        "rank_map": {t: i + 1 for i, t in enumerate(band)},
        # runtime-selectable momentum variant + the options (UI dropdown)
        "mom_variant": daily_core.current_variant(),
        "mom_variants": daily_core.STRATEGY_VARIANTS,
        "protection": daily_core.current_protection(),
        "protections": daily_core.PROTECTION_MODES,
        "target_vol": daily_core.current_target_vol(),
        "target_vols": daily_core.TARGET_VOL_MODES,
        # runtime-selectable qv-mom universe + the options (UI dropdown)
        "universe": daily_core.current_universe(),
        "universes": universe_names(),
    }

@app.post('/api/dailycore/strategy')
async def daily_core_strategy(req: dict):
    """Select the daily-core momentum variant at runtime (persisted across
    restarts). Body: {"variant": "raw" | "residual"}.

    The choice drives the NEXT ranking recompute (nightly cycle or a manual
    "Run Cycle Now") and every future backfill. It does NOT rewrite stored
    history: re-run the backfill after switching to see the new variant's
    synthetic track record.
    """
    from . import daily_core
    variant = (req or {}).get("variant", "")
    try:
        daily_core.set_variant(variant)
    except ValueError as e:
        raise HTTPException(422, str(e)) from e
    # Invalidate the stored ranking: it was computed under the previous
    # variant, and the next cycle must recompute under the new one.
    from .db import DailyCoreRanking
    from sqlalchemy import delete as sa_delete
    async with Session() as s:
        await s.execute(sa_delete(DailyCoreRanking))
        await s.commit()
    return {"ok": True, "variant": variant,
            "label": daily_core.STRATEGY_VARIANTS[variant]}

@app.post('/api/dailycore/protection')
async def daily_core_protection(req: dict):
    """Select the daily-core protection overlay at runtime (persisted).
    Body: {"mode": "none" | "trend200" | "gradient200" | "gradient100" | "gradient50"}.

    The mode drives the NEXT deployment pass and every future backfill. With a
    gradient mode a confirmed negative basket slope (armed only in good times)
    cashes out the book; the engine re-enters on its own rule, no cooldown.
    """
    from . import daily_core
    mode = (req or {}).get("mode", "")
    try:
        daily_core.set_protection(mode)
    except ValueError as e:
        raise HTTPException(422, str(e)) from e
    return {"ok": True, "mode": mode, "label": daily_core.PROTECTION_MODES[mode]}

@app.post('/api/dailycore/targetvol')
async def daily_core_targetvol(req: dict):
    """Set the daily-core target-volatility control at runtime (persisted).
    Body: {"mode": "off" | "0.10" | "0.15" | "0.20" | "0.25"}.

    When the portfolio's own 21d realized vol exceeds the target, new cash is
    parked instead of deployed (never sells, never leverages). Drives the next
    deployment pass and every future backfill.
    """
    from . import daily_core
    mode = (req or {}).get("mode", "")
    try:
        daily_core.set_target_vol(mode)
    except ValueError as e:
        raise HTTPException(422, str(e)) from e
    return {"ok": True, "mode": mode, "label": daily_core.TARGET_VOL_MODES[mode]}

@app.post('/api/sim/universe')
async def sim_universe(req: dict):
    """Set the universe for ALL sims at runtime (persisted).
    Body: {"universe": "<name from /api/screener/universes>"}.

    The single Dashboard control writes this. It drives the daily sim, the
    monthly qv-mom portfolio and the daily-core portfolio, plus every future
    backfill. It does NOT rewrite stored history — re-run a backfill after
    switching to see the new universe's track record.
    """
    from . import daily_core
    name = (req or {}).get("universe", "")
    try:
        daily_core.set_universe(name)
    except ValueError as e:
        raise HTTPException(422, str(e)) from e
    # The stored daily-core ranking was computed on the previous universe;
    # drop it so the next cycle recomputes on the new one.
    from .db import DailyCoreRanking
    from sqlalchemy import delete as sa_delete
    async with Session() as s:
        await s.execute(sa_delete(DailyCoreRanking))
        await s.commit()
    return {"ok": True, "universe": name}

@app.get('/api/dailycore/trades')
async def daily_core_trades(limit: int = Query(default=100, ge=1, le=500)):
    from . import daily_core
    return await daily_core.get_trades(limit)

@app.get('/api/dailycore/equity')
async def daily_core_equity(limit: int = Query(default=365, ge=1, le=12000)):
    from . import daily_core
    return await daily_core.get_equity_curve(limit)

@app.post('/api/dailycore/run')
async def daily_core_run():
    """Manually trigger one daily-core cycle (idempotent; lock-guarded)."""
    from . import daily_core
    return await daily_core.run_daily_cycle()

@app.post('/api/dailycore/refresh')
async def daily_core_refresh():
    """Refresh candle data for the daily-core portfolio's holdings (+ FX
    pairs) so valuation/equity use current prices — the pull-to-refresh
    path, mirroring /api/monthly/refresh. The universe-wide refresh stays
    in the nightly cycle."""
    from . import daily_core
    from .db import DailyCorePosition
    from sqlalchemy import select
    async with Session() as s:
        held = [p.ticker for p in (await s.scalars(select(DailyCorePosition))).all()]
    pairs = sorted({pm[0] for t in held
                    if (pm := daily_core.monthly_mod.fundamentals_mod._suffix_fx(t))})
    # Held-only pull-to-refresh: free Yahoo bulk path (fast, unmetered). FX pairs
    # are Yahoo symbols (EURUSD=X) anyway, so the provider path just 404'd them.
    refreshed, errors = await refresh_many(held + pairs, "2y")
    return {"refreshed": refreshed, "errors": errors, "total": len(refreshed) + len(errors)}

@app.post('/api/dailycore/backfill')
async def daily_core_backfill(start: str | None = Query(default=None)):
    """Backfill the daily-core portfolio with synthetic history.

    Replays the winning strategy (qv-mom ranking + daily rank deployment)
    over stored candles/fundamentals to today and REPLACES the portfolio
    state with the replay's end state.

    ``start``: "YYYY-MM-DD" to replay from that date, "all" for the full
    stored history, or omit (default) to synch with the other sims — the
    replay starts on the earliest snapshot date of the daily sim / monthly
    portfolios so all three equity curves cover the same window.
    """
    from . import daily_core
    r = await daily_core.backfill(start)
    if not r.get("ok"):
        raise HTTPException(400, r.get("error", "backfill failed"))
    return r

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


@app.post('/api/sim/chat/stream')
async def sim_chat_stream(req: ChatRequest):
    """SSE stream of an interactive chat with the sim portfolio manager.

    Same semantics as ``POST /api/sim/chat`` — server persists history, can
    execute trades from ``[[ACTION]]`` blocks — but streams the LLM text
    incrementally.

    Events:
      - ``data: {"type":"delta","text":"..."}`` — raw LLM chunks (action block
        included; the frontend swaps it for the stripped display_text on done).
      - ``data: {"type":"done","text":display_text,"trades":[...],
         "actions_executed":bool,"raw":full_raw}`` — final envelope.

    Persistence happens AFTER the LLM stream completes, so a refresh
    mid-stream does not leave an orphan user question in the DB.
    """
    if not req.messages:
        raise HTTPException(400, "messages must not be empty")
    new_msgs = []
    for m in req.messages[-1:]:
        if m.role in ('user', 'assistant') and m.content.strip():
            new_msgs.append({"role": m.role, "content": m.content})
    if not new_msgs:
        raise HTTPException(400, "no valid messages")

    async def gen():
        try:
            async for evt in sim.sim_chat_stream(new_msgs):
                yield await _sse_event(evt)
        except Exception as e:
            yield await _sse_event({"type": "error", "text": f"{type(e).__name__}: {e}"})

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


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

@app.get('/', include_in_schema=False)
async def index() -> FileResponse:
    """Serve the SPA entry with no-cache so browsers revalidate on each
    deploy. Starlette's StaticFiles only sets ETag/Last-Modified; without
    Cache-Control the browser heuristically caches index.html and keeps
    running the previous JS after a redeploy."""
    resp = FileResponse(_STATIC_DIR / "index.html")
    resp.headers["Cache-Control"] = "no-cache"
    return resp

app.mount('/', StaticFiles(directory=str(_STATIC_DIR), html=True), name='static')
