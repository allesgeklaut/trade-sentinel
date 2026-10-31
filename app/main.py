import asyncio, json, logging, math, re
from contextlib import asynccontextmanager
from datetime import date, datetime, timedelta, UTC
from pathlib import Path
from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy import func, select
from pydantic import BaseModel
from .config import settings
from .db import (Watchlist, Session, init_db, current_db, use_db, dev_enabled,
                 dev_db_exists, prod_db_exists, clone_prod_to_dev)
from .market import refresh, refresh_many, refresh_yfinance, candles, search, info, provider, PERIOD_COUNTS
from .analysis import compute, persist, history, MIN_CANDLES
from .screener import universe_names, run, results, refresh_incremental, load_deep_history, get_screener_progress, ScreenerBusy
from . import sim
from . import news as news_mod
from . import llm as llm_mod
from . import run_config
from . import universe_sync

_STATIC_DIR = Path(__file__).resolve().parent.parent / "static"
logger = logging.getLogger("trade_sentinel.main")
_WATCHLIST_REFRESH_PERIOD = "2y"
_watchlist_prefetch_task: asyncio.Task | None = None
_universe_prefetch_task: asyncio.Task | None = None
_universe_sync_task: asyncio.Task | None = None
_savings_loop_task: asyncio.Task | None = None

# Serialises the four-portfolio backfill: concurrent clicks (or a click plus a
# retry) would otherwise each queue all four multi-minute replays.
_backfill_all_lock = asyncio.Lock()

# Set while /api/db/clone swaps the DEV file; observable via /api/db/status so
# other clients/scripts see an in-flight clone. The clone holds
# _backfill_all_lock (so backfill-all/reset-all 409 while it runs) and the
# guarded DEV backfills/runs 409 while it is set (a clone swaps only the DEV
# file — PROD requests are unaffected).
_db_cloning = False


# ---------------------------------------------------------------------------
# PROD/DEV database-switch guards.
#
# The guards are FastAPI DEPENDENCIES, not in-body calls: dependencies run on
# every HTTP request (routing resolves db-select middleware first, so
# current_db() is the cookie-selected database) while a DIRECT Python call of
# the endpoint function bypasses them entirely — test/maintenance callers keep
# the plain functions usable.
#
# UNCONDITIONAL on PROD: with the feature disabled there is no DEV database at
# all, so NO database accepts the history-rewriting endpoints below — PROD is
# append-only by design and a 403 is the honest answer.
# ---------------------------------------------------------------------------

def _require_dev_db(action: str):
    """403 history-rewriting actions unless the request is DEV-routed."""
    async def _guard() -> None:
        if current_db() != "dev":
            raise HTTPException(
                403, f"{action} is disabled on the PROD database — switch to DEV")
    return _guard


def _require_prod_db(action: str):
    """403 real-money (savings) actions unless the request is PROD-routed:
    the DEV copy of the savings data is a stale clone, so writes there would
    only mislead."""
    async def _guard() -> None:
        if current_db() != "prod":
            raise HTTPException(
                403, f"{action} is only available on the PROD database")
    return _guard


async def _refuse_during_clone() -> None:
    """409 DEV-routed backfills/runs while a PROD→DEV clone is swapping the
    dev file: an op racing the swap could read a half-replaced database or
    write into an orphaned inode. A clone only swaps DEV — PROD requests are
    unaffected and never see this guard."""
    if current_db() == "dev" and _db_cloning:
        raise HTTPException(409, "a PROD→DEV clone is running — try again shortly")


def _validated_start(start: str | None) -> str | None:
    """Backfill ``start``: None, "all", or an ISO date. Reject anything else
    with 422 here rather than a raw 500 from the backfill's pd.Timestamp."""
    if start is None or start == "all":
        return start
    try:
        date.fromisoformat(start)
    except ValueError as e:
        raise HTTPException(422, f"invalid start: {start!r} (use YYYY-MM-DD or 'all')") from e
    return start

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


async def _universe_prefetch_loop() -> None:
    """Prefetch the shared universe ``sim_prefetch_lead_minutes`` before the sim
    run, then rescore the screener from those fresh candles.

    App-level, NOT inside the sim scheduler: the Dashboard's screener table and
    charts need fresh candles even when ``SIM_ENABLED`` is false. Gated by
    ``sim_universe_prefetch``; the screener rescore only runs after a successful
    prefetch (never re-stamps a stale cache as fresh).
    """
    while True:
        now = datetime.now(UTC)
        lead = max(0, settings.sim_prefetch_lead_minutes)
        target = now.replace(hour=settings.sim_run_hour, minute=settings.sim_run_minute,
                             second=0, microsecond=0) - timedelta(minutes=lead)
        if target <= now:
            target += timedelta(days=1)
        wait_seconds = (target - now).total_seconds()
        logger.info("Universe prefetch: next run at %s (in %.0f seconds)", target, wait_seconds)
        await asyncio.sleep(wait_seconds)
        if not sim.is_trading_day(datetime.now(UTC)):
            logger.info("Universe prefetch: %s is not a trading day — skipping",
                        datetime.now(UTC).date())
            continue
        try:
            r = await sim.prefetch_universe()
        except Exception as e:
            logger.error("Universe prefetch failed: %s", e, exc_info=True)
            continue  # don't rescore stale candles
        if not r.get("ok"):
            logger.warning("Universe prefetch: %s", r.get("reason"))
            continue
        if settings.screener_auto_rescore:
            try:
                from . import screener
                res = await screener.rescore(r["universe"])
                logger.info("Screener auto-rescore: %s", res)
            except Exception as e:
                logger.error("Screener auto-rescore failed: %s", e, exc_info=True)


async def _universe_sync_loop() -> None:
    """Refresh the generated S&P 500 universe weekly at the configured UTC
    weekday/hour. App-level, independent of ``SIM_ENABLED``; the nightly
    universe prefetch picks up any membership change on its next run."""
    while True:
        now = datetime.now(UTC)
        days_ahead = (settings.universe_sync_weekday - now.weekday()) % 7
        # % 24: an out-of-range config value must not raise here and kill the
        # task (the scheduling math runs outside the try below).
        target = (now + timedelta(days=days_ahead)).replace(
            hour=settings.universe_sync_hour % 24, minute=0, second=0, microsecond=0)
        if target <= now:
            target += timedelta(days=7)
        wait_seconds = (target - now).total_seconds()
        logger.info("Universe sync: next run at %s (in %.0f seconds)", target, wait_seconds)
        await asyncio.sleep(wait_seconds)
        try:
            r = await universe_sync.sync_sp500()
            logger.info("Universe sync: %s -> %d tickers (+%d/-%d)",
                        r["universe"], r["count"], len(r["added"]), len(r["removed"]))
        except Exception as e:
            logger.error("Universe sync failed: %s", e, exc_info=True)


async def _savings_loop() -> None:
    """Daily savings-tracker pass at ``savings_snapshot_hour`` (UTC).

    App-level, independent of ``SIM_ENABLED`` — the real-money account must
    keep accruing interest even when the paper sims are off. Runs EVERY
    calendar day (interest accrues on calendar days, not NYSE days; the
    sims already refreshed candles at ``sim_run_hour``). No-ops until the
    tracker is initialized; the cycle's stages are all idempotent, so a
    restart mid-day just catches up.
    """
    from . import savings
    while True:
        now = datetime.now(UTC)
        target = now.replace(hour=settings.savings_snapshot_hour % 24,
                             minute=settings.savings_snapshot_minute % 60,
                             second=0, microsecond=0)
        if target <= now:
            target += timedelta(days=1)
        wait_seconds = (target - now).total_seconds()
        logger.info("Savings loop: next run at %s (in %.0f seconds)", target, wait_seconds)
        await asyncio.sleep(wait_seconds)
        if not settings.savings_enabled:
            continue
        try:
            if not await savings.initialized():
                logger.info("Savings loop: tracker not initialized — skipping pass")
                continue
            r = await savings.run_savings_cycle()
            if r.get("skipped"):
                # A manual "Run Now" or an operator action holding
                # account_lock is mid-pass — every stage is idempotent and
                # the markers catch up on the next nightly pass.
                logger.info("Savings cycle skipped: %s", r.get("reason"))
                continue
            logger.info("Savings cycle: refresh=%d transfer=%s payout=%s accrual=%s sparplans=%d",
                        r["refresh"]["refreshed"], r["transfer"].get("deposited"),
                        r["payout"].get("paid"), r["accrual"].get("accrued"),
                        r["sparplans"].get("executed", 0))
        except Exception as e:
            logger.error("Savings cycle failed: %s", e, exc_info=True)


async def _dev_is_empty() -> bool:
    """True when DEV has no watchlist rows yet (a fresh, schema-only DEV).

    Lifespan/status run outside any request context, where a plain Session()
    is PROD — the use_db("dev") wrapper is what points this query at DEV.
    """
    with use_db("dev"):
        async with Session() as s:
            return (await s.scalars(select(func.count()).select_from(Watchlist))).one() == 0


@asynccontextmanager
async def lifespan(app):
    global _watchlist_prefetch_task, _universe_prefetch_task, _universe_sync_task
    global _savings_loop_task
    # init_db() FIRST: it creates/validates BOTH schemas (PROD and, when the
    # switch is on, DEV), so the queries below always have their tables.
    await init_db()
    async with Session() as s:
        for t in settings.watchlist.split(','):
            ticker = t.strip().upper()
            if not ticker:
                continue
            if not await s.get(Watchlist, ticker):
                s.add(Watchlist(ticker=ticker))
        await s.commit()
    # Seed DEV with a full copy of PROD (candles + portfolio state) whenever
    # DEV is missing OR still empty and PROD has data. A greenfield first
    # boot creates both schemas empty (no clone); the moment the settings
    # watchlist lands in PROD the empty DEV is cloned, so the sandbox starts
    # with the same candles, watchlist and portfolio state. init_db() may
    # have created an empty dev file first — the clone replaces it (dispose +
    # os.replace handle that). A failed seed never blocks startup: DEV is
    # disposable, PROD is not.
    if dev_enabled() and prod_db_exists() and (not dev_db_exists() or await _dev_is_empty()):
        try:
            logger.info("DEV database missing/empty — seeding it from PROD")
            await clone_prod_to_dev()
        except Exception as e:  # noqa: BLE001 — startup must survive a failed seed
            logger.error("DEV auto-seed failed (continuing with an empty DEV db): %s", e)
    if settings.watchlist_prefetch:
        _watchlist_prefetch_task = asyncio.create_task(_watchlist_prefetch_loop())
    if settings.sim_universe_prefetch:
        _universe_prefetch_task = asyncio.create_task(_universe_prefetch_loop())
    if settings.universe_sync_enabled:
        _universe_sync_task = asyncio.create_task(_universe_sync_loop())
    if settings.savings_enabled:
        _savings_loop_task = asyncio.create_task(_savings_loop())
    if settings.sim_enabled:
        sim.start_scheduler()
    yield
    if _watchlist_prefetch_task and not _watchlist_prefetch_task.done():
        _watchlist_prefetch_task.cancel()
    _watchlist_prefetch_task = None
    if _universe_prefetch_task and not _universe_prefetch_task.done():
        _universe_prefetch_task.cancel()
    _universe_prefetch_task = None
    if _universe_sync_task and not _universe_sync_task.done():
        _universe_sync_task.cancel()
    _universe_sync_task = None
    if _savings_loop_task and not _savings_loop_task.done():
        _savings_loop_task.cancel()
    _savings_loop_task = None
    if settings.sim_enabled:
        sim.stop_scheduler()
app=FastAPI(title="Trade Sentinel",lifespan=lifespan)
_TICKER_RE = re.compile(r"^[A-Z0-9.\-^=]{1,32}$")

@app.middleware("http")
async def db_select_middleware(request, call_next):
    """Route this request to the PROD or DEV database via the ts_db cookie.

    Unknown/missing values fall through to PROD (the ContextVar default), and
    DEV is only honoured when the dev engine is configured — so disabling
    DATABASE_URL_DEV instantly pins every client back to PROD. ContextVars are
    task-local: the cookie selection reaches exactly this request's handling
    (routing, dependencies, endpoint) and can never redirect the background
    schedulers, which run outside any request context.
    """
    if dev_enabled() and request.cookies.get("ts_db") == "dev":
        with use_db("dev"):
            return await call_next(request)
    return await call_next(request)

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
@app.post('/api/screener/sync')
async def screener_sync():
    """Refresh the generated S&P 500 universe from the IVV holdings CSV.

    Returns the change diff ({count, added, removed, changed, as_of}). One
    upstream request; no candle fetch. Newly added names appear in the screener
    after the next Update/Refresh (or the nightly prefetch + auto-rescore).
    """
    try:
        return await universe_sync.sync_sp500()
    except Exception as e:
        raise HTTPException(502, f"universe sync failed: {e}") from e
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

class ChatProposalRequest(BaseModel):
    proposal_id: str

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

@app.post('/api/sim/run', dependencies=[Depends(_refuse_during_clone)])
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

@app.post('/api/sim/backfill', dependencies=[Depends(_require_dev_db("Backfill")),
                                            Depends(_refuse_during_clone)])
async def sim_backfill(start: str | None = Query(default=None)):
    """Backfill the daily sim with a DETERMINISTIC synthetic history.

    Replays the engine's own rules (no LLM calls — the hybrid live strategy's
    LLM overlay cannot be replayed) over stored candles/fundamentals and
    REPLACES the portfolio state with the replay's end state. An
    approximation by design; the response carries approximation: true.
    """
    from . import sim
    r = await sim.backfill(_validated_start(start))
    if r.get("skipped"):
        raise HTTPException(409, r.get("reason", "skipped"))
    if not r.get("ok"):
        raise HTTPException(400, r.get("error", "backfill failed"))
    return r

@app.post('/api/sim/backfill-benchmark',
          dependencies=[Depends(_require_dev_db("Benchmark backfill")),
                        Depends(_refuse_during_clone)])
async def sim_backfill_benchmark(start: str | None = Query(default=None)):
    """Backfill the DCA benchmark curve (URTH $1000/month) over historical
    candles. The twin of the other backfills so all four curves can cover
    the same window. ``start``: date | "all" | default = synced."""
    from . import sim
    r = await sim.backfill_benchmark(_validated_start(start))
    if not r.get("ok"):
        raise HTTPException(400, r.get("error", "backfill failed"))
    return r

@app.post('/api/backfill-all', dependencies=[Depends(_require_dev_db("Backfill-all"))])
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
    start = _validated_start(start)
    if _backfill_all_lock.locked():
        raise HTTPException(409, "a backfill-all is already running")
    async with _backfill_all_lock:
        return await _backfill_all_locked(start)


async def _backfill_all_locked(start: str | None) -> dict:
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
    # Each replay holds its own cycle lock while it runs, so a scheduled cycle
    # can't interleave *within* a replay; the four run sequentially and this
    # coordinator holds no lock across them, so a nightly tick can still fire
    # between two replays.
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
        except Exception:
            logger.exception("backfill-all: %s failed", name)
            # Don't return internal exception text to the client; the detail
            # is in the server log.
            errors[name] = f"{name} backfill failed (see server logs)"

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

@app.post('/api/monthly/backfill',
          dependencies=[Depends(_require_dev_db("Monthly backfill")),
                        Depends(_refuse_during_clone)])
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
    return await monthly.backfill(_validated_start(start))


@app.get('/api/monthly/rebalances')
async def monthly_rebalances(limit: int = Query(default=24, ge=1, le=120)):
    """Rebalance decision log: held before, picked, #new per month."""
    from . import monthly
    return await monthly.get_rebalances(limit)

@app.post('/api/monthly/run', dependencies=[Depends(_refuse_during_clone)])
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

# =====================================================================
# LLM sell scorecard (slice 2): the with/without-LLM history
# =====================================================================

@app.get('/api/sellscore')
async def sellscore():
    """The LLM sell scorecard: per-type with/without-LLM delta aggregates on
    COMPLETED predeclared forward windows, the reference bars, and the recent
    decision history (last 200 ledger rows). Compute-on-demand and read-only;
    no cache state. Spans both paper accounts (sim type A + monthly B/C)."""
    from . import sellscore as sellscore_mod
    return await sellscore_mod.score_sell_decisions()


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
        "gate": daily_core.current_gate(),
        "gates": daily_core.GATE_MODES,
        "gradient": daily_core.current_gradient(),
        "gradients": daily_core.GRADIENT_MODES,
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

@app.post('/api/dailycore/gate')
async def daily_core_gate(req: dict):
    """Set the daily-core DEPLOYMENT GATE at runtime (persisted).
    Body: {"mode": "off" | "200" | "100" | "50"}.

    While the market index is below its N-day SMA the gate parks new buys
    (contributions stay in cash). It does NOT sell. Drives the next cycle and
    every future backfill.
    """
    from . import daily_core
    mode = (req or {}).get("mode", "")
    try:
        daily_core.set_gate(mode)
    except ValueError as e:
        raise HTTPException(422, str(e)) from e
    return {"ok": True, "mode": mode, "label": daily_core.GATE_MODES[mode]}

@app.post('/api/dailycore/gradient')
async def daily_core_gradient(req: dict):
    """Set the daily-core GRADIENT CASH-OUT arm at runtime (persisted).
    Body: {"mode": "off" | "always" | "200" | "100" | "50"}.

    On a confirmed negative slope of the target basket the book is sold to
    cash; "always" fires ungated, an SMA value arms it only in good times
    (market above that SMA). Re-entry is unconditional (no cooldown). Drives
    the next cycle and every future backfill.
    """
    from . import daily_core
    mode = (req or {}).get("mode", "")
    try:
        daily_core.set_gradient(mode)
    except ValueError as e:
        raise HTTPException(422, str(e)) from e
    return {"ok": True, "mode": mode, "label": daily_core.GRADIENT_MODES[mode]}

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

@app.post('/api/dailycore/run', dependencies=[Depends(_refuse_during_clone)])
async def daily_core_run():
    """Manually trigger one daily-core cycle (idempotent; lock-guarded)."""
    from . import daily_core
    # wait=False: an HTTP request must not hang behind a running backfill —
    # the manual path keeps the immediate "already running" answer.
    return await daily_core.run_daily_cycle(wait=False)

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

@app.post('/api/dailycore/backfill',
          dependencies=[Depends(_require_dev_db("Daily-core backfill")),
                        Depends(_refuse_during_clone)])
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
    r = await daily_core.backfill(_validated_start(start))
    if not r.get("ok"):
        raise HTTPException(400, r.get("error", "backfill failed"))
    return r

# =====================================================================
# Portfolio performance metrics (read-only): dated XIRR + TWR
#
# The audit's outstanding Phase 1 item — "TWR charts and dated XIRR are not
# delivered" — needs two numbers the tabs did not have:
#
#   * XIRR — the money-weighted annualized return of the account's actual
#     cash-flow schedule (deposits/withdrawals + terminal value, actual/365).
#   * TWR  — the time-weighted return of the unitized (cash-flow-adjusted)
#     NAV, so the weeks BEFORE a deposit weigh the same as the weeks after.
#
# Both are read-only: unlike the per-tab status endpoints they never deposit
# an allowance or write a snapshot, so calling them cannot mutate a paper
# portfolio's history.
# =====================================================================

_METRIC_SCOPES = ("daily_core", "monthly", "sim")


def _validated_scope(scope: str) -> str:
    if scope not in _METRIC_SCOPES:
        raise HTTPException(
            422, f"invalid scope: {scope!r} (use one of {', '.join(_METRIC_SCOPES)})")
    return scope


def _finite_or_none(value) -> float | None:
    """JSON-safe float: NaN/Infinity/None/non-numeric -> None.

    Starlette serialises with ``allow_nan=False``, so an invalid stored
    observation must never reach the response as NaN.
    """
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


async def _metric_snapshot_points(scope: str, limit: int) -> list[dict]:
    """Per-day snapshot points for one paper portfolio, oldest-first.

    Reuses the existing equity-curve readers (``sim``/``monthly``/
    ``daily_core`` ``get_equity_curve``) rather than duplicating their
    queries.  Intraday duplicates — the monthly and daily-core portfolios
    write one mark per day plus one after a rebalance — collapse to that
    day's LAST snapshot, the same rule the charts use.
    """
    _validated_scope(scope)
    if scope == "sim":
        rows = await sim.get_equity_curve(limit)
    elif scope == "monthly":
        from . import monthly
        rows = await monthly.get_equity_curve(limit)
    else:
        from . import daily_core
        rows = await daily_core.get_equity_curve(limit)

    by_day: dict[str, dict] = {}
    for row in rows:
        day = row.get("date") or (row.get("at") or "")[:10]
        if not day:
            continue
        by_day[day] = {
            "date": day,
            "total_equity": row.get("total_equity"),
            "contributed": row.get("allowance_total"),
        }
    return [by_day[day] for day in sorted(by_day)]


@app.get('/api/metrics/xirr')
async def metrics_xirr(scope: str = Query(default='daily_core'),
                       limit: int = Query(default=12000, ge=1, le=12000)):
    """Dated money-weighted annualized return (XIRR) for one paper portfolio.

    ``scope``: ``daily_core`` | ``monthly`` | ``sim``.

    Flows are the day-over-day deltas of the account's cumulative
    contribution (``allowance_total``, the same series the drawdown uses);
    the terminal value is the latest ``total_equity`` on the latest snapshot
    date.  Actual/365 day counts, solved by bisection on NPV=0 — see
    ``app.performance.xirr``.

    ``xirr`` is a FRACTION (0.10 = +10%/yr); ``xirr_pct`` is the same number
    in percent for display.  Both are null when the money-weighted return is
    undefined (empty account, no dated external flows, no positive terminal
    value); ``reason`` then says why.  ``window_days`` reports the span the
    rate annualizes (a handful of days annualizes into noise — the UI only
    shows XIRR once the window is at least a month).
    ``limit`` truncates the window.  The first point's cumulative
    contribution counts as the window's opening inflow — the convention the
    cash-flow series supports.  A window that starts mid-history cannot see
    market gains made before the window, so the default (full history) is the
    meaningful measurement; a shorter ``limit`` is for inspecting a slice,
    with that caveat.

    Caveat: the contribution ledger is the only dated record of external
    money.  Capital seeded into an account WITHOUT a contribution row cannot
    be dated, so it would inflate the rate; the shipped start-cash defaults
    are 0 and the backfills record their start cash as a contribution.
    """
    from .performance import dated_external_flows, xirr

    scope = _validated_scope(scope)
    points = await _metric_snapshot_points(scope, limit)
    flows = dated_external_flows(
        [p["date"] for p in points], [p["contributed"] for p in points])

    final_date = points[-1]["date"] if points else None
    final_value = _finite_or_none(points[-1]["total_equity"]) if points else None
    window_days = None
    if len(points) > 1:
        first = date.fromisoformat(points[0]["date"])
        last = date.fromisoformat(points[-1]["date"])
        window_days = (last - first).days
    rate: float | None = None
    reason: str | None = None
    if not points:
        reason = "no equity snapshots for this scope yet"
    elif final_value is None:
        reason = "latest snapshot has no usable total_equity"
    elif not flows:
        reason = "no dated external flows (the contributed total never changed)"
    else:
        rate = xirr([d for d, _ in flows], [a for _, a in flows], final_value, final_date)
        if rate is None:
            reason = ("no NPV=0 solution — e.g. no positive terminal value "
                      "(a total loss cannot be annualized)")

    return {
        "ok": True,
        "scope": scope,
        "xirr": rate,
        "xirr_pct": None if rate is None else rate * 100.0,
        "final_value": final_value,
        "final_date": final_date,
        "window_days": window_days,
        "contributed": _finite_or_none(points[-1]["contributed"]) if points else None,
        "points": len(points),
        "flows": [{"date": d.isoformat(), "amount": a} for d, a in flows],
        "reason": reason,
    }


@app.get('/api/metrics/twr')
async def metrics_twr(scope: str = Query(default='daily_core'),
                      limit: int = Query(default=12000, ge=1, le=12000)):
    """Time-weighted return (unitized NAV) series for one paper portfolio.

    ``nav`` is the chained cash-flow-adjusted NAV from
    ``app.performance.unitized_nav``: 1.0 at the first funded point, and a
    deposit or withdrawal that leaves the portfolio value unchanged leaves
    the NAV unchanged.  ``return_pct`` is that point's cumulative return since
    the chain start (null before funding).

    ``twr_pct`` is the cumulative TWR of the returned window (last NAV /
    first funded NAV - 1).  It is null — with a ``reason`` — for an empty or
    never-funded account, and when the NAV chain restarted after a wipeout
    inside the window (then no single cumulative number is honest; the
    per-point ``nav`` still shows both chains).  ``scope``/``limit`` behave
    as on ``/api/metrics/xirr``.
    """
    from .performance import unitized_nav

    scope = _validated_scope(scope)
    points = await _metric_snapshot_points(scope, limit)
    nav = unitized_nav([p["total_equity"] for p in points],
                       [p["contributed"] for p in points]) if points else []

    start_index = next((i for i, v in enumerate(nav) if v > 0.0), None)
    base = nav[start_index] if start_index is not None else None

    # A wipeout (NAV 0 on a positive base) followed by another positive NAV
    # means the chain restarted at 1.0; a terminal wipeout is a real -100%.
    zeros = [i for i in range((start_index or 0) + 1, len(nav))
             if nav[i] == 0.0 and nav[i - 1] > 0.0]
    restarted = bool(zeros) and any(v > 0.0 for v in nav[zeros[-1] + 1:])

    twr_pct: float | None = None
    reason: str | None = None
    if not points:
        reason = "no equity snapshots for this scope yet"
    elif base is None:
        reason = "the account was never funded (no positive equity point)"
    elif restarted:
        reason = ("the unitized NAV hit zero (total loss) inside this window and "
                  "restarted — no single cumulative TWR is meaningful")
    else:
        twr_pct = (nav[-1] / base - 1.0) * 100.0

    series = []
    for i, (point, value) in enumerate(zip(points, nav, strict=True)):
        return_pct: float | None = None
        if (base is not None and start_index is not None
                and i >= start_index and math.isfinite(value)):
            return_pct = (value / base - 1.0) * 100.0
        series.append({
            "date": point["date"],
            "nav": _finite_or_none(value),
            "return_pct": return_pct,
            "total_equity": _finite_or_none(point["total_equity"]),
            "contributed": _finite_or_none(point["contributed"]),
        })

    return {
        "ok": True,
        "scope": scope,
        "points": series,
        "twr_pct": twr_pct,
        "nav_start": base,
        "start": points[start_index]["date"] if start_index is not None else None,
        "end": points[-1]["date"] if points else None,
        "reason": reason,
    }


# =====================================================================
# Savings tracker endpoints (real money, EUR — no backfill, no reset-all)
# =====================================================================

class SavingsInitRequest(BaseModel):
    cash: float
    positions: list[dict] = []
    monthly_transfer: float | None = None
    interest_rate: float | None = None

class SavingsTrueupRequest(BaseModel):
    actual_cash: float  # TR cash total incl. unpaid interest, positions excluded

class SavingsDepositRequest(BaseModel):
    amount: float

class SavingsWithdrawRequest(BaseModel):
    amount: float
    occurred_on: str | None = None  # "YYYY-MM-DD"; omitted = today (operator-local)
    note: str = ""

class SavingsDividendRequest(BaseModel):
    amount: float                   # EUR received (money IN to the tracked cash)
    ticker: str                     # the paying holding
    occurred_on: str | None = None  # "YYYY-MM-DD"; omitted = today (operator-local)
    note: str = ""

class SavingsFeeRequest(BaseModel):
    amount: float                   # EUR paid from the tracked cash
    occurred_on: str | None = None  # "YYYY-MM-DD"; omitted = today (operator-local)
    note: str = ""

class SavingsSettleRequest(BaseModel):
    ticker: str                     # the disposed holding the proceeds belong to
    proceeds: float                 # EUR now in the tracked cash
    occurred_on: str | None = None  # "YYYY-MM-DD"; omitted = today (operator-local)
    note: str = ""
    # The SELL row this settlement closes (optional: inferred when exactly one
    # unsettled disposal exists for the ticker). UNIQUE per disposal.
    settlement_of: int | None = None
    # Explicit opt-in for a disposal that predates the SELL record; requires a
    # non-empty explanatory note.
    legacy: bool = False

class SavingsTaxRequest(BaseModel):
    amount: float                   # EUR paid from the tracked cash
    occurred_on: str | None = None  # "YYYY-MM-DD"; omitted = today (operator-local)
    note: str = ""

class SavingsSellRequest(BaseModel):
    ticker: str
    shares: float                    # quantity disposed (> 0, <= held); omitted = full via DELETE
    price: float | None = None       # execution price per share in EUR (never invented)
    occurred_on: str | None = None   # "YYYY-MM-DD"; omitted = today (operator-local)

class SavingsSplitRequest(BaseModel):
    ticker: str
    ratio: float                    # new shares per old share (> 0, not exactly 1.0)
    occurred_on: str | None = None  # "YYYY-MM-DD"; omitted = today (operator-local)
    note: str = ""

class SavingsCorrectRequest(BaseModel):
    ticker: str
    shares: float | None = None      # NEW share count (omitted = not corrected)
    avg_cost: float | None = None    # NEW per-share average cost in EUR
    occurred_on: str | None = None   # "YYYY-MM-DD"; omitted = today (operator-local)
    note: str = ""

class SavingsSavebackRequest(BaseModel):
    amount: float
    ticker: str

class SavingsBuyRequest(BaseModel):
    amount: float
    ticker: str

class SavingsConfigRequest(BaseModel):
    monthly_transfer: float | None = None
    interest_rate: float | None = None

class SavingsPlanRequest(BaseModel):
    ticker: str
    amount: float
    day_of_month: int

@app.get('/api/savings/status')
async def savings_status():
    """Savings tab status: valuation, config, plans, events, mirror."""
    from . import savings
    return await savings.status()

@app.get('/api/savings/equity')
async def savings_equity(limit: int = Query(default=365, ge=1, le=12000)):
    """Savings equity curve (oldest-first), EUR."""
    from . import savings
    return await savings.get_equity_curve(limit)

@app.get('/api/savings/mirror')
async def savings_mirror(limit: int = Query(default=365, ge=1, le=12000)):
    """Mirror portfolio: valuation + allocated-over-time curve."""
    from . import savings
    return {**await savings._mirror_valuate(),
            "curve": await savings.get_mirror_curve(limit)}

@app.get('/api/savings/metrics/xirr')
async def savings_metrics_xirr(limit: int = Query(default=12000, ge=1, le=12000)):
    """Dated money-weighted annualized return (XIRR) of the savings account.

    Same response shape/conventions as ``/api/metrics/xirr`` — ``scope`` is
    ``savings``, and the flows are the event ledger's DATED external capital
    (contributions in, withdrawals out, dated ``occurred_on`` when recorded
    else the operator-local ``created_at`` day) instead of the day-over-day
    deltas of a snapshot column. The equity window is the
    ``SavingsSnapshot`` curve; the terminal value is its latest point.

    ``xirr`` is a FRACTION (``0.10`` = +10%/yr) and null (with a ``reason``)
    when the money-weighted return is undefined. ``window_days`` reports the
    span of the equity window. Because the flow schedule comes from the
    ledger (not from the window's first contribution), a truncated ``limit``
    does not invent an opening inflow — but it does drop equity points
    before the window, so the full history remains the meaningful run.
    """
    from . import savings
    from .performance import xirr

    data = await savings.metric_series(limit)
    points = data["points"]
    flows = [(date.fromisoformat(f["date"]), f["amount"]) for f in data["flows"]]

    final_date = points[-1]["date"] if points else None
    final_value = _finite_or_none(points[-1]["total_equity"]) if points else None
    window_days = None
    if len(points) > 1:
        window_days = (date.fromisoformat(points[-1]["date"])
                       - date.fromisoformat(points[0]["date"])).days
    rate: float | None = None
    reason: str | None = None
    if not points:
        reason = "no equity snapshots for this savings account yet"
    elif final_value is None:
        reason = "latest snapshot has no usable total_equity"
    elif not flows:
        reason = ("no dated external flows in this window (no contribution or "
                  "withdrawal at or before the last snapshot)")
    else:
        rate = xirr([d for d, _ in flows], [a for _, a in flows], final_value, final_date)
        if rate is None:
            reason = ("no NPV=0 solution — e.g. no positive terminal value "
                      "(a total loss cannot be annualized)")

    return {
        "ok": True,
        "scope": "savings",
        "xirr": rate,
        "xirr_pct": None if rate is None else rate * 100.0,
        "final_value": final_value,
        "final_date": final_date,
        "window_days": window_days,
        "contributed": _finite_or_none(points[-1]["contributed"]) if points else None,
        "points": len(points),
        "flows": [{"date": d.isoformat(), "amount": a} for d, a in flows],
        "reason": reason,
    }

@app.get('/api/savings/metrics/twr')
async def savings_metrics_twr(limit: int = Query(default=12000, ge=1, le=12000)):
    """Time-weighted return (unitized NAV) series of the savings account.

    Same response shape/conventions as ``/api/metrics/twr`` (``scope`` is
    ``savings``): ``nav`` is the chained, cash-flow-adjusted NAV from
    ``app.performance.unitized_nav`` over the ``SavingsSnapshot`` curve with
    the ledger's cumulative NET contributed series; a deposit or withdrawal
    that leaves the value unchanged leaves the NAV unchanged. ``return_pct``
    is the cumulative return since the chain start.

    Additive vs the paper-portfolio endpoint: ``max_drawdown_pct`` is the
    cash-flow-neutral (peak-to-trough of the same unitized NAV) maximum
    drawdown from ``app.performance.max_drawdown`` — withdrawals and deposits
    are flows, so they never appear as drawdown.
    """
    from . import savings
    from .performance import max_drawdown, unitized_nav

    data = await savings.metric_series(limit)
    points = data["points"]
    nav = unitized_nav([p["total_equity"] for p in points],
                       [p["contributed"] for p in points]) if points else []

    start_index = next((i for i, v in enumerate(nav) if v > 0.0), None)
    base = nav[start_index] if start_index is not None else None

    zeros = [i for i in range((start_index or 0) + 1, len(nav))
             if nav[i] == 0.0 and nav[i - 1] > 0.0]
    restarted = bool(zeros) and any(v > 0.0 for v in nav[zeros[-1] + 1:])

    twr_pct: float | None = None
    reason: str | None = None
    if not points:
        reason = "no equity snapshots for this savings account yet"
    elif base is None:
        reason = "the account was never funded (no positive equity point)"
    elif restarted:
        reason = ("the unitized NAV hit zero (total loss) inside this window and "
                  "restarted — no single cumulative TWR is meaningful")
    else:
        twr_pct = (nav[-1] / base - 1.0) * 100.0

    series = []
    for i, (point, value) in enumerate(zip(points, nav, strict=True)):
        return_pct: float | None = None
        if (base is not None and start_index is not None
                and i >= start_index and math.isfinite(value)):
            return_pct = (value / base - 1.0) * 100.0
        series.append({
            "date": point["date"],
            "nav": _finite_or_none(value),
            "return_pct": return_pct,
            "total_equity": _finite_or_none(point["total_equity"]),
            "contributed": _finite_or_none(point["contributed"]),
        })

    return {
        "ok": True,
        "scope": "savings",
        "points": series,
        "twr_pct": twr_pct,
        "nav_start": base,
        "start": points[start_index]["date"] if start_index is not None else None,
        "end": points[-1]["date"] if points else None,
        "max_drawdown_pct": max_drawdown(nav) if nav else None,
        "reason": reason,
    }

@app.post('/api/savings/initialize',
          dependencies=[Depends(_require_prod_db("Savings initialize"))])
async def savings_initialize(req: SavingsInitRequest):
    """Seed a FRESH tracker: savings cash + current holdings.

    Non-destructive by contract. An already initialized tracker is refused
    with 422 ``already initialized`` BEFORE anything is touched — the operator's
    restart path is ``/api/savings/reset``, which preserves the event ledger
    and the equity curve, clears the current state, and lets this endpoint seed
    a fresh epoch. Pass positions as [{ticker, shares, avg_cost}] (EUR);
    repeated tickers are aggregated into one weighted-average lot.
    """
    from . import savings
    if req.cash < 0:
        raise HTTPException(422, "cash must be >= 0")
    for p in req.positions:
        ticker = str(p.get("ticker") or "").strip().upper()
        if not _TICKER_RE.match(ticker):
            raise HTTPException(422, f"invalid ticker symbol: {ticker!r}")
        try:
            sh, ac = float(p["shares"]), float(p["avg_cost"])
        except (KeyError, ValueError, TypeError) as e:
            raise HTTPException(422, "position needs numeric shares/avg_cost") from e
        if sh <= 0 or ac <= 0:
            raise HTTPException(422, "position shares/avg_cost must be > 0")
    async with savings.account_lock:
        r = await savings.initialize(
            req.cash, req.positions, req.monthly_transfer, req.interest_rate)
    if not r.get("ok"):
        raise HTTPException(422, r.get("reason", "initialize failed"))
    return r

@app.post('/api/savings/reset',
          dependencies=[Depends(_require_prod_db("Savings reset"))])
async def savings_reset():
    """Start a new epoch: clear current state, KEEP the ledger and the curve.

    History-safe: the events and snapshots are preserved bit-for-bit and a
    ``RESET`` boundary row is appended; contributed capital and the metrics
    series read only the new epoch's rows. Positions/plans and the picks
    mirror (the comparison artifact) are cleared and the tracker is
    uninitialized again, so the setup form returns. The paper sims are never
    touched.
    """
    from . import savings
    async with savings.account_lock:
        return await savings.reset()

@app.post('/api/savings/trueup',
          dependencies=[Depends(_require_prod_db("Savings true-up"))])
async def savings_trueup(req: SavingsTrueupRequest):
    """Reconcile the savings cash with the real TR balance (delta booked)."""
    from . import savings
    async with savings.account_lock:
        r = await savings.trueup(req.actual_cash)
    if not r.get("ok"):
        raise HTTPException(422, r.get("reason", "true-up failed"))
    return r

@app.post('/api/savings/config',
          dependencies=[Depends(_require_prod_db("Savings config"))])
async def savings_config(req: SavingsConfigRequest):
    """Update interest rate / monthly transfer (persisted, runtime-editable)."""
    from . import savings
    async with savings.account_lock:
        r = await savings.set_config(req.monthly_transfer, req.interest_rate)
    if not r.get("ok"):
        raise HTTPException(409, r.get("reason", "not initialized"))
    return r

@app.post('/api/savings/deposit',
          dependencies=[Depends(_require_prod_db("Savings deposit"))])
async def savings_deposit(req: SavingsDepositRequest):
    """Book a one-off cash deposit: fresh money moved into the TR cash —
    a contribution (raises NET contributed capital). True-up stays the
    reconciliation tool for drift."""
    from . import savings
    async with savings.account_lock:
        r = await savings.add_deposit(req.amount)
    if not r.get("ok"):
        raise HTTPException(422, r.get("reason", "deposit failed"))
    return r

@app.post('/api/savings/withdraw',
          dependencies=[Depends(_require_prod_db("Savings withdrawal"))])
async def savings_withdraw(req: SavingsWithdrawRequest):
    """Book a dated cash withdrawal: fresh money moved OUT of the TR cash.

    The mirror of ``/api/savings/deposit``: the tracked cash drops and a
    ``WITHDRAWAL`` event (signed negative, dated ``occurred_on``) lowers NET
    contributed capital — it is an external flow, never an investment loss.
    Refusals: 422 for a malformed amount/date or for insufficient tracked
    cash (matching ``/api/savings/buy``'s state-rejection class); 404 for an
    uninitialized tracker (the resource does not exist) — a larger
    real-world gap is true-up's reconciliation.
    """
    from . import savings
    async with savings.account_lock:
        r = await savings.withdraw(req.amount, req.occurred_on, req.note)
    return _savings_event_result(r, "withdrawal failed")


def _savings_event_result(result: dict, action: str) -> dict:
    """Shared status convention for the dated savings ledger events.

    422 for a malformed payload or a state rejection (matching
    ``/api/savings/buy``'s class); 404 only for an uninitialized tracker —
    the resource does not exist, so nothing could have been booked.
    """
    if not result.get("ok"):
        reason = result.get("reason", action)
        raise HTTPException(404 if reason == "not initialized" else 422, reason)
    return result


@app.post('/api/savings/dividend',
          dependencies=[Depends(_require_prod_db("Savings dividend"))])
async def savings_dividend(req: SavingsDividendRequest):
    """Book a cash dividend received into the tracked TR cash.

    A dividend is INVESTMENT INCOME, not fresh capital: the cash rises and a
    ``DIVIDEND`` event (signed positive, dated ``occurred_on``, flow_class
    ``dividend``) is appended, so NET contributed capital is untouched and the
    income shows up in the return. Refusals: 422 malformed amount/date or an
    invalid ticker symbol; 404 for an uninitialized tracker.
    """
    from . import savings
    ticker = req.ticker.strip().upper()
    if not _TICKER_RE.match(ticker):
        raise HTTPException(422, f"invalid ticker symbol: {ticker!r}")
    async with savings.account_lock:
        r = await savings.add_dividend(req.amount, ticker, req.occurred_on, req.note)
    return _savings_event_result(r, "dividend failed")


@app.post('/api/savings/fee',
          dependencies=[Depends(_require_prod_db("Savings fee"))])
async def savings_fee(req: SavingsFeeRequest):
    """Book a fee PAID from the tracked cash (e.g. TR's FX fee).

    The cash drops and a ``FEE`` event (signed negative, flow_class ``fee``)
    is appended. A fee is a cost INSIDE the tracker, not an external flow: it
    is never netted against contributed capital and hits the return honestly.
    Refusals: 422 malformed amount/date or ``insufficient cash``; 404 for an
    uninitialized tracker.
    """
    from . import savings
    async with savings.account_lock:
        r = await savings.add_fee(req.amount, req.occurred_on, req.note)
    return _savings_event_result(r, "fee failed")


@app.post('/api/savings/tax',
          dependencies=[Depends(_require_prod_db("Savings tax"))])
async def savings_tax(req: SavingsTaxRequest):
    """Book a tax PAID from the tracked cash.

    The cash drops and a ``TAX`` event (signed negative, flow_class ``tax``)
    is appended. A tax is a cost INSIDE the tracker, not an external flow: it
    is never netted against contributed capital and hits the return honestly.
    Refusals: 422 malformed/future date or ``insufficient cash``; 404 for an
    uninitialized tracker.
    """
    from . import savings
    async with savings.account_lock:
        r = await savings.add_tax(req.amount, req.occurred_on, req.note)
    return _savings_event_result(r, "tax failed")


@app.post('/api/savings/settle',
          dependencies=[Depends(_require_prod_db("Savings settlement"))])
async def savings_settle(req: SavingsSettleRequest):
    """Book the settlement half of a manual disposal: proceeds are now cash.

    The counterpart of the SELL disposal record: the tracked cash rises and a
    ``SETTLEMENT`` event (signed positive, flow_class ``sale_settlement``) is
    appended, dated ``occurred_on`` and LINKED to the SELL row it closes
    (``settlement_of`` — inferred when exactly one unsettled disposal exists
    for the ticker). It is NOT an external flow — the pair reads honestly
    (temporary mark-down while the money was in flight, then the proceeds
    booked inside the account; the net over the pair is the realized outcome).
    One disposal can never be settled twice (the link is UNIQUE). A disposal
    that predates the SELL record needs the explicit ``legacy`` opt-in and an
    explanatory note. Refusals: 422 malformed proceeds/date, invalid ticker,
    already-settled/ambiguous linkage or an over-priced settlement; 404
    uninitialized tracker.
    """
    from . import savings
    ticker = req.ticker.strip().upper()
    if not _TICKER_RE.match(ticker):
        raise HTTPException(422, f"invalid ticker symbol: {ticker!r}")
    async with savings.account_lock:
        r = await savings.settle_sale(
            ticker, req.proceeds, req.occurred_on, req.note,
            settlement_of=req.settlement_of, legacy=req.legacy)
    return _savings_event_result(r, "settlement failed")


@app.post('/api/savings/split',
          dependencies=[Depends(_require_prod_db("Savings split"))])
async def savings_split(req: SavingsSplitRequest):
    """Record a share split (a corporate action) on a real holding.

    Adjusts the position in place — shares × ratio, average cost ÷ ratio, the
    total cost basis unchanged — appends a ``SPLIT`` record (amount 0, the NEW
    share count and average cost, flow_class ``corporate_action``) and, for
    every funding source with mirror shares of that ticker outstanding as of
    the split day, appends ONE 0-€ per-source adjustment dated at the split
    (so past curve dates replay unchanged and each source releases exactly its
    scaled shares later).
    Refusals: 422 invalid ratio / no-op split (ratio 1.0) / no such position /
    malformed or future date / ``later position changes prevent backdated
    split`` (no lot replay); 404 uninitialized tracker.
    """
    from . import savings
    ticker = req.ticker.strip().upper()
    if not _TICKER_RE.match(ticker):
        raise HTTPException(422, f"invalid ticker symbol: {ticker!r}")
    async with savings.account_lock:
        r = await savings.record_split(ticker, req.ratio, req.occurred_on, req.note)
    return _savings_event_result(r, "split failed")


@app.post('/api/savings/correct',
          dependencies=[Depends(_require_prod_db("Savings correction"))])
async def savings_correct(req: SavingsCorrectRequest):
    """Record a dated bookkeeping correction of a real holding.

    Sets the supplied fields (shares and/or average cost) and appends a
    ``CORRECTION`` record carrying the NEW values (the untouched field stays
    NULL) plus the old ones in the note — so the change is auditable even
    though the position row is overwritten. No cash moves, contributed capital
    is untouched and the picks mirror is NOT rescaled (a corporate action is
    ``/api/savings/split``). A correction mutates the current position, so a
    future date is refused. Refusals: 422 nothing to correct / invalid value /
    invalid or future date / no such position; 404 uninitialized tracker.
    """
    from . import savings
    ticker = req.ticker.strip().upper()
    if not _TICKER_RE.match(ticker):
        raise HTTPException(422, f"invalid ticker symbol: {ticker!r}")
    async with savings.account_lock:
        r = await savings.correct_position(
            ticker, req.shares, req.avg_cost, req.occurred_on, req.note)
    return _savings_event_result(r, "correction failed")

@app.post('/api/savings/saveback',
          dependencies=[Depends(_require_prod_db("Savings saveback"))])
async def savings_saveback(req: SavingsSavebackRequest):
    """Book a Saveback payout TR invested into a Sparplan asset."""
    from . import savings
    ticker = req.ticker.strip().upper()
    if not _TICKER_RE.match(ticker):
        raise HTTPException(422, f"invalid ticker symbol: {ticker!r}")
    async with savings.account_lock:
        r = await savings.add_saveback(req.amount, ticker)
    if not r.get("ok"):
        raise HTTPException(422, r.get("reason", "saveback failed"))
    return r

@app.post('/api/savings/buy',
          dependencies=[Depends(_require_prod_db("Savings buy"))])
async def savings_manual_buy(req: SavingsBuyRequest):
    """Record a one-off buy paid from the tracked savings cash (deducts
    cash, mirrors into daily-core). For TR's actual Saveback bonus use
    /api/savings/saveback — that money comes from nowhere."""
    from . import savings
    ticker = req.ticker.strip().upper()
    if not _TICKER_RE.match(ticker):
        raise HTTPException(422, f"invalid ticker symbol: {ticker!r}")
    async with savings.account_lock:
        r = await savings.manual_buy(req.amount, ticker)
    if not r.get("ok"):
        raise HTTPException(422, r.get("reason", "buy failed"))
    return r

@app.get('/api/savings/plans')
async def savings_plans():
    from . import savings
    return await savings.get_plans()

@app.post('/api/savings/plans',
          dependencies=[Depends(_require_prod_db("Savings plan"))])
async def savings_add_plan(req: SavingsPlanRequest):
    from . import savings
    ticker = req.ticker.strip().upper()
    if not _TICKER_RE.match(ticker):
        raise HTTPException(422, f"invalid ticker symbol: {ticker!r}")
    async with savings.account_lock:
        r = await savings.add_plan(ticker, req.amount, req.day_of_month)
    if not r.get("ok"):
        raise HTTPException(422, r.get("reason", "invalid plan"))
    return r

@app.delete('/api/savings/plans/{plan_id}',
          dependencies=[Depends(_require_prod_db("Savings plan removal"))])
async def savings_remove_plan(plan_id: int):
    from . import savings
    async with savings.account_lock:
        r = await savings.remove_plan(plan_id)
    if not r.get("ok"):
        raise HTTPException(404, r.get("reason", "no such plan"))
    return r

@app.delete('/api/savings/positions/{ticker}',
          dependencies=[Depends(_require_prod_db("Savings position removal"))])
async def savings_remove_position(ticker: str):
    """Remove a position manually (sold in TR — proceeds arrive via true-up)."""
    from . import savings
    async with savings.account_lock:
        r = await savings.remove_position(ticker.strip().upper())
    if not r.get("ok"):
        raise HTTPException(404, r.get("reason", "no such position"))
    return r

@app.post('/api/savings/sell',
          dependencies=[Depends(_require_prod_db("Savings sell"))])
async def savings_sell(req: SavingsSellRequest):
    """Dispose of a specific quantity of one holding (partial sale record).

    Records the EXACT disposed shares as a dated ``SELL`` row (prior average
    cost, execution price only when supplied, amount=0 — no cash moves), keeps
    the remaining shares at their average cost, and releases the funding
    source's picks-mirror allocation proportionally. The response carries
    ``disposal_event_id`` for linking the later settlement. A full quantity
    works too; the DELETE route remains the undated full-disposal path. A
    future date is refused (a disposal mutates the current position), and a
    BACK-DATED quantity refuses when later position changes or later mirror
    allocations would make the reversal fabricate history. Refusals: 422
    invalid quantity/price/date or more shares than held, 422 backdated
    chronology conflicts; 404 uninitialized tracker or no such position.
    """
    from . import savings
    ticker = req.ticker.strip().upper()
    if not _TICKER_RE.match(ticker):
        raise HTTPException(422, f"invalid ticker symbol: {ticker!r}")
    async with savings.account_lock:
        r = await savings.remove_position(
            ticker, req.price, shares=req.shares, occurred_on=req.occurred_on)
    if not r.get("ok"):
        reason = r.get("reason", "sell failed")
        raise HTTPException(
            404 if reason in ("not initialized", "no such position") else 422, reason)
    return r

@app.post('/api/savings/refresh',
          dependencies=[Depends(_require_prod_db("Savings refresh"))])
async def savings_refresh():
    """Pull-to-refresh: fetch candles for held tickers + EURUSD=X so the
    tab valuates on the most recent close (on weekends: Friday's). Free
    Yahoo bulk path; skips tickers within the freshness window."""
    from . import savings
    r = await savings.refresh_data()
    return {"refreshed": r["refreshed"], "errors": r["errors"],
            "total": len(r["refreshed"]) + len(r["errors"])}

@app.post('/api/savings/run',
          dependencies=[Depends(_require_prod_db("Savings run"))])
async def savings_run():
    """Manually trigger one savings pass (transfer/payout/accrue/sparplans/
    snapshot — all idempotent)."""
    from . import savings
    return await savings.run_savings_cycle()

@app.post('/api/sim/chat')
async def sim_chat_endpoint(req: ChatRequest):
    """Interactive chat with the sim portfolio manager LLM.

    The server keeps the full conversation history in the DB, so the client
    only needs to send the new user message for this turn. If the LLM proposes
    actions (an ``[[ACTION]]`` block), they are NOT executed here — the turn
    returns them as a pending proposal (with a per-action preview), and they
    only execute when the user explicitly confirms via
    ``POST /api/sim/chat/confirm``.
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

    Same semantics as ``POST /api/sim/chat`` — server persists history, and an
    ``[[ACTION]]`` block is only PROPOSED (executed solely via
    ``POST /api/sim/chat/confirm`` after explicit user confirmation) — but
    streams the LLM text incrementally.

    Events:
      - ``data: {"type":"delta","text":"..."}`` — raw LLM chunks (action block
        included; the frontend swaps it for the stripped display_text on done).
      - ``data: {"type":"done","text":display_text,"proposals":[...],
         "raw":full_raw,"history":[...]}`` — final envelope.

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


@app.post('/api/sim/chat/confirm', dependencies=[Depends(_refuse_during_clone)])
async def sim_chat_confirm(req: ChatProposalRequest):
    """Execute a pending chat proposal the user explicitly confirmed.

    The chat turn itself only PROPOSED the actions (no trades, no ledger, no
    snapshot). This endpoint re-validates them against FRESH state (same
    per-action validation the proposal preview used), executes them on the sim
    paper book, takes a snapshot if anything traded, and freezes the LLM
    decision ledger with the proposal's parse-time decided_at.

    404 when the proposal id is unknown or expired; 409 when the run
    configuration changed since the proposal (the pending entry is dropped).
    Like ``/api/sim/run``, DEV-routed confirms answer 409 while a PROD→DEV
    clone is running; PROD confirms proceed.
    """
    try:
        return await sim.confirm_chat_proposal(req.proposal_id)
    except sim.ProposalNotFound as e:
        raise HTTPException(404, str(e)) from e
    except sim.ProposalConfigChanged as e:
        raise HTTPException(409, str(e)) from e


@app.post('/api/sim/chat/reject')
async def sim_chat_reject(req: ChatProposalRequest):
    """Drop a pending chat proposal without executing anything (no DB writes).

    404 when the proposal id is unknown or expired.
    """
    try:
        return await sim.reject_chat_proposal(req.proposal_id)
    except sim.ProposalNotFound as e:
        raise HTTPException(404, str(e)) from e


# ---------------------------------------------------------------------------
# Live run configuration (read-only)
# ---------------------------------------------------------------------------

@app.get('/api/run-config')
async def run_config_live():
    """The CURRENT (live) effective configuration of every engine, plus its
    current epoch.

    Read-only, no guards. Each value is that engine's own ``current_config()``
    dict plus ``epoch`` (from the run-config registry) and ``engine``. sim's
    ``current_config()`` is async because its universe can resolve from the
    DB — so on a dev-routed request it follows the cookie, per the PROD/DEV
    semantics; monthly's and daily-core's are sync.
    """
    from . import daily_core, monthly
    sim_cfg = await sim.current_config()
    return {
        "sim": {
            **sim_cfg,
            "engine": "sim",
            "epoch": await run_config.current_epoch("sim", session_factory=sim.Session),
        },
        "monthly": {
            **monthly.current_config(),
            "engine": "monthly",
            "epoch": await run_config.current_epoch("monthly",
                                                     session_factory=monthly.Session),
        },
        "daily_core": {
            **daily_core.current_config(),
            "engine": "daily_core",
            "epoch": await run_config.current_epoch("daily_core",
                                                     session_factory=daily_core.Session),
        },
    }


@app.post('/api/sim/reset', dependencies=[Depends(_require_dev_db("Reset"))])
async def sim_reset():
    """Wipe all sim tables and restart with start cash."""
    return await sim.reset_sim()


@app.post('/api/sim/reset-all', dependencies=[Depends(_require_dev_db("Reset-all"))])
async def sim_reset_all():
    """Wipe ALL FOUR paper portfolios and restart each at its start cash.

    Clears trades, positions, allowances and equity history for the daily sim,
    monthly, daily-core and DCA benchmark portfolios. Market data,
    fundamentals, the screener and the stored rankings/universe selection are
    left intact — only portfolio state is cleared. Refused while a backfill is
    running so the reset can't race the multi-minute replays.
    """
    from . import daily_core, monthly
    if _backfill_all_lock.locked():
        raise HTTPException(409, "a backfill is already running — wait for it to finish")
    async with _backfill_all_lock:
        return {
            "ok": True,
            "results": {
                # reset_sim also clears the DCA benchmark tables.
                "daily_sim": await sim.reset_sim(),
                "monthly": await monthly.reset_monthly(),
                "daily_core": await daily_core.reset_daily_core(),
            },
        }

# ---------------------------------------------------------------------------
# PROD/DEV database switch endpoints
# ---------------------------------------------------------------------------

@app.get('/api/db/status')
async def db_status():
    """Which database this request is talking to, and the DEV sandbox state."""
    return {"active": current_db(),
            "dev_enabled": dev_enabled(),
            "dev_seeded": dev_enabled() and dev_db_exists() and not await _dev_is_empty(),
            "cloning": _db_cloning}

@app.post('/api/db/clone')
async def db_clone():
    """Overwrite DEV with a fresh copy of PROD (candles + portfolio state).

    Serialized with backfill-all/reset-all (the shared lock) so the dev file
    can't be swapped while those are mid-run, and refused while any portfolio
    backfill/manual run holds its lock (the clone would swap the file under
    it). 409s when DEV is disabled entirely.
    """
    global _db_cloning
    if _backfill_all_lock.locked():
        raise HTTPException(409, "another backfill, reset, or clone is already running — "
                                 "wait for it to finish")
    if not dev_enabled():
        raise HTTPException(409, "the DEV database is disabled (DATABASE_URL_DEV is unset)")
    from . import sim, monthly, daily_core
    async with _backfill_all_lock:
        # Re-check inside the lock: a DEV backfill/manual run may have started
        # between the unlocked check and the acquire (the runs do not take
        # _backfill_all_lock). This narrows — not closes — the in-flight race
        # documented in docs/dev-prod-db.md.
        if (sim._run_cycle_lock.locked() or monthly._rebalance_lock.locked()
                or daily_core._cycle_lock.locked()):
            raise HTTPException(409, "a backfill or run is already running — wait for it to finish")
        _db_cloning = True
        try:
            logger.info("DB clone: copying PROD onto DEV")
            r = await clone_prod_to_dev()
            logger.info("DB clone: DEV refreshed (%d bytes)", r["size_bytes"])
            return r
        finally:
            _db_cloning = False


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
