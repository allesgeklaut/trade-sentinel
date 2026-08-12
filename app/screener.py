from datetime import datetime, timezone, timedelta
from pathlib import Path
import logging
import pandas as pd
from sqlalchemy import delete, func, select
from .analysis import compute
from .db import Candle, ScreenerResult, Session
from .market import candles, refresh

_UNIVERSES_DIR = Path(__file__).resolve().parent.parent / "universes"
logger = logging.getLogger("trade_sentinel.screener")


def universe_names(): return sorted(p.stem for p in _UNIVERSES_DIR.glob("*.txt"))
def tickers(name):
    p=_UNIVERSES_DIR/f"{name}.txt"
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
            await refresh(symbol)
            rows = await candles(symbol)
            out = score(rows)
            if not out: continue
            # Attach BUY/SELL/HOLD signal from the full analysis engine.
            # candles() already returned ~2y of data from refresh(); compute()
            # needs >=206 rows. New IPOs with insufficient history get "N/A".
            try:
                r = compute(rows)
                out["action"] = r["action"]
                out["strength"] = r["strength"]
            except ValueError:
                out["action"] = "N/A"
                out["strength"] = None
            results.append((symbol, out))
        except Exception as e:
            logger.warning("screener skip %s: %s", symbol, e)
            continue
    async with Session() as s:
        await s.execute(delete(ScreenerResult).where(ScreenerResult.universe==name))
        for symbol,x in results: s.add(ScreenerResult(universe=name,ticker=symbol,updated_at=datetime.now(timezone.utc),**x))
        await s.commit()
    return {"universe":name,"processed":len(symbols),"ranked":len(results)}
async def results(name):
    async with Session() as s:
        rows=(await s.scalars(select(ScreenerResult).where(ScreenerResult.universe==name).order_by(ScreenerResult.score.desc()))).all()
        return [{"ticker":r.ticker,"score":r.score,"trend":r.trend,"return_20d":r.return_20d,"return_60d":r.return_60d,"rsi":r.rsi,"relative_volume":r.relative_volume,"close":r.close,"updated_at":r.updated_at.isoformat(),"action":r.action,"strength":r.strength} for r in rows]


async def refresh_incremental(name: str, max_age_days: int = 3) -> dict:
    """Incrementally refresh candle data for a universe.

    Only fetches tickers that are missing from the DB or whose latest candle
    is older than *max_age_days*. Already-fresh tickers are skipped to avoid
    redundant network calls. Does NOT re-run the screener ranking.
    """
    symbols = list(dict.fromkeys(tickers(name)))
    refreshed = 0
    skipped = 0
    errors: list[str] = []
    cutoff = datetime.now(timezone.utc) - timedelta(days=max_age_days)

    for symbol in symbols:
        try:
            async with Session() as s:
                latest = await s.scalar(
                    select(func.max(Candle.timestamp)).where(Candle.ticker == symbol)
                )
            if latest is not None and latest.replace(tzinfo=timezone.utc) >= cutoff:
                skipped += 1
                continue
            await refresh(symbol, "2y")
            refreshed += 1
        except Exception as e:
            errors.append(f"{symbol}: {e}")
            logger.warning("incremental refresh skip %s: %s", symbol, e)

    logger.info("Incremental refresh %s: %d refreshed, %d skipped, %d errors",
                name, refreshed, skipped, len(errors))
    return {"universe": name, "refreshed": refreshed, "skipped": skipped,
            "errors": errors, "total": len(symbols)}


async def load_deep_history(name: str, period: str = "10y") -> dict:
    """Fetch deep candle history for all tickers in a universe.

    Used to backfill data for walk-forward optimization and the regime filter.
    Fetches *period* (default 10y) for every ticker, overwriting existing
    candles via upsert. Does NOT re-run the screener ranking.
    """
    symbols = list(dict.fromkeys(tickers(name)))
    loaded = 0
    errors: list[str] = []

    for symbol in symbols:
        try:
            await refresh(symbol, period)
            loaded += 1
        except Exception as e:
            errors.append(f"{symbol}: {e}")
            logger.warning("deep load skip %s: %s", symbol, e)

    logger.info("Deep load %s (%s): %d loaded, %d errors", name, period, loaded, len(errors))
    return {"universe": name, "period": period, "loaded": loaded,
            "errors": errors, "total": len(symbols)}