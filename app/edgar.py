"""SEC EDGAR fundamentals for the monthly qv-mom portfolio.

Companyfacts (https://data.sec.gov/api/xbrl/companyfacts/CIK##########.json)
serves every dated XBRL fact a US filer ever reported: period start/end, the
real filing date, restatements per filing, history back to ~2009. Stored in
the same ``fundamentals`` table as the yfinance facts (source='edgar'), the
strategy math treats them identically — but the point-in-time discipline is
real: a rebalance sees exactly what was public by its close.

Ticker -> CIK mapping from sec.gov/files/company_tickers.json, cached in the
``sec_ciks`` table. Tickers without a CIK (ETFs, foreign listings) fall back
to the yfinance fetcher, mirroring stockstrat's split.

SEC fair-access: max ~10 req/s — we fetch companyfacts once per ticker at
~8/s with retries, then cache; refreshes are rare (facts accumulate slowly).
"""
import asyncio
import json
import logging
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone

from sqlalchemy import func, select
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

from .config import settings
from .db import Fundamental, SecCik, Session
from .screener import tickers as universe_tickers

logger = logging.getLogger("trade_sentinel.edgar")

# Primary XBRL tags the scoring math consumes (instant = equity/shares,
# duration = TTM-built income/cashflow items).
US_GAAP_TAGS = [
    "StockholdersEquity",
    "NetIncomeLoss",
    "NetCashProvidedByUsedInOperatingActivities",
    "PaymentsToAcquirePropertyPlantAndEquipment",
    "CommonStockSharesOutstanding",
]
DEI_TAGS = [
    "EntityCommonStockSharesOutstanding",
]
# Some issuers switch income-statement tags over time (e.g. Booking Holdings
# moved to NetIncomeLossAvailableToCommonStockholdersBasic), leaving holes in
# the primary tags. Fill from these alternates, deduped by (end, start).
TAG_FALLBACKS = {
    "NetIncomeLoss": [
        "NetIncomeLossAvailableToCommonStockholdersBasic",
        "ProfitLoss",
    ],
    "NetCashProvidedByUsedInOperatingActivities": [
        "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations",
    ],
    "StockholdersEquity": [
        "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
    ],
}
ALL_TAGS = sorted(set(US_GAAP_TAGS) | set(DEI_TAGS)
                  | {a for alts in TAG_FALLBACKS.values() for a in alts})

UA = {"User-Agent": "trade-sentinel paper-trading research (private deployment)",
      "Accept": "*/*"}

STALE_AFTER_DAYS = 90


def _http_get(url: str, retries: int = 4, backoff: float = 2.0) -> bytes:
    last = None
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=30) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code == 404:
                raise  # permanent: no companyfacts for this CIK — don't retry
            last = e
            time.sleep(backoff * (i + 1))
        except Exception as e:  # noqa: BLE001 - network errors vary
            last = e
            time.sleep(backoff * (i + 1))
    raise RuntimeError(f"GET {url} failed after {retries} tries: {last}")


# ---------------------------------------------------------------------------
# CIK mapping
# ---------------------------------------------------------------------------

async def ensure_cik_map(tickers: list[str], force: bool = False) -> dict[str, int]:
    """Return {ticker: cik} with cik=0 for tickers confirmed to have no CIK.
    Uses the cached sec_ciks table; downloads the SEC map only when tickers
    are missing from it (or force=True)."""
    want = set(tickers)
    async with Session() as s:
        rows = (await s.scalars(
            select(SecCik).where(SecCik.ticker.in_(want)))).all()
    have = {r.ticker: r.cik for r in rows}
    missing = sorted(want - set(have))
    if missing and not force:
        # fill from the official map (single ~1MB download)
        def _sync() -> dict[str, int]:
            sec = json.loads(_http_get("https://www.sec.gov/files/company_tickers.json"))
            out: dict[str, int] = {}
            by_ticker = {}
            for v in sec.values():
                by_ticker[v["ticker"].upper()] = (int(v["cik_str"]), v["title"])
            for t in missing:
                hit = None
                for cand in (t, t.replace("-", "."), t.replace(".", "-")):
                    if cand in by_ticker:
                        hit = cand
                        break
                if hit:
                    cik, name = by_ticker[hit]
                    out[t] = cik
                else:
                    out[t] = 0  # known no-CIK
            return out

        resolved = await asyncio.to_thread(_sync)
        async with Session() as s:
            for t, cik in resolved.items():
                stmt = sqlite_insert(SecCik).values(ticker=t, cik=cik, name="")
                stmt = stmt.on_conflict_do_update(index_elements=["ticker"],
                                                  set_={"cik": cik})
                await s.execute(stmt)
            await s.commit()
        have.update(resolved)
    return {t: have.get(t, 0) for t in want}


# ---------------------------------------------------------------------------
# companyfacts -> fact rows
# ---------------------------------------------------------------------------

def _parse_companyfacts(facts: dict) -> dict[str, list[dict]]:
    """Extract the needed XBRL tags from a companyfacts payload as
    {tag: [{start,end,filed,val}, ...]} with fallback-tag gap filling."""
    rec: dict[str, list[dict]] = {}
    gaap = facts.get("facts", {}).get("us-gaap", {})
    dei = facts.get("facts", {}).get("dei", {})
    for tagname in ALL_TAGS:
        d = gaap.get(tagname) or dei.get(tagname)
        if not d:
            continue
        entries: dict[tuple, dict] = {}
        for _unit, vlist in d.get("units", {}).items():
            for v in vlist:
                end = v.get("end")
                if not end:
                    continue
                entry = {"start": v.get("start"), "end": end,
                         "filed": v["filed"], "val": v["val"]}
                # keep every filed version; strategy dedups by (start,end)
                # preferring the latest filed <= asof (restatements)
                entries[(end, entry["start"], entry["filed"])] = entry
        rec[tagname] = sorted(entries.values(), key=lambda e: (e["end"], e["filed"]))

    # fill primary-tag holes from fallback tags (periods the primary lacks)
    for tag, alts in TAG_FALLBACKS.items():
        primary = rec.get(tag, [])
        keys = {(e["end"], e.get("start")) for e in primary}
        for alt in alts:
            for e in rec.pop(alt, []) if alt in rec else []:
                if (e["end"], e.get("start")) not in keys:
                    primary.append(e)
                    keys.add((e["end"], e.get("start")))
        if primary:
            rec[tag] = sorted(primary, key=lambda e: (e["end"], e["filed"]))
    return rec


def fetch_companyfacts(cik: int) -> dict[str, list[dict]]:
    """Fetch one CIK's companyfacts and parse it (sync, network)."""
    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"
    try:
        payload = json.loads(_http_get(url))
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return {}  # ETF/trust with a CIK but no companyfacts
        raise
    return _parse_companyfacts(payload)


async def refresh_edgar(tickers: list[str] | None = None, force: bool = False,
                        progress=None) -> dict[str, str]:
    """Fetch EDGAR companyfacts for the universe into the fundamentals table
    (source='edgar'). Skips tickers refreshed within STALE_AFTER_DAYS unless
    force. Returns {ticker: status}."""
    from sqlalchemy import select as _select

    if tickers is None:
        tickers = universe_tickers(settings.sim_monthly_universe)
    tickers = list(tickers)
    ciks = await ensure_cik_map(tickers)
    cutoff = datetime.now(timezone.utc).timestamp() - STALE_AFTER_DAYS * 86400

    async with Session() as s:
        rows = (await s.execute(
            _select(Fundamental.ticker, func.max(Fundamental.updated_at))
            .where(Fundamental.ticker.in_(tickers), Fundamental.source == "edgar")
            .group_by(Fundamental.ticker))).all()
    last_refresh = {t: ts for t, ts in rows}

    statuses: dict[str, str] = {}
    if force:
        todo = [t for t in sorted(tickers) if t in ciks and ciks[t]]
    else:
        # refetch tickers with no EDGAR rows yet, or whose newest row is older
        # than STALE_AFTER_DAYS (same staleness rule as the yfinance path)
        todo = [t for t in sorted(tickers) if t in ciks and ciks[t]
                and (t not in last_refresh
                     or last_refresh[t].timestamp() <= cutoff)]

    fetched = 0
    for t in todo:
        try:
            facts = await asyncio.to_thread(fetch_companyfacts, ciks[t])
            if not facts:
                statuses[t] = "no-companyfacts"
                continue
            count = 0
            async with Session() as s:
                for tag, entries in facts.items():
                    for e in entries:
                        stmt = sqlite_insert(Fundamental).values(
                            ticker=t, tag=tag, start=e.get("start"),
                            end=e["end"], filed=e["filed"], val=float(e["val"]),
                            currency="shares" if tag.endswith("SharesOutstanding") else "USD",
                            source="edgar")
                        stmt = stmt.on_conflict_do_update(
                            index_elements=["ticker", "tag", "start", "end", "filed", "source"],
                            set_={"val": stmt.excluded.val,
                                  "updated_at": stmt.excluded.updated_at})
                        await s.execute(stmt)
                        count += 1
                await s.commit()
            statuses[t] = f"ok({count})"
            fetched += 1
            if progress:
                progress(t, fetched, len(todo))
        except Exception as e:  # noqa: BLE001
            logger.warning("edgar %s failed: %s", t, e)
            statuses[t] = f"failed: {e}"
        # SEC fair-access: stay under ~10 req/s
        await asyncio.sleep(0.12)

    # tickers with a CIK that were not refetched this pass are fresh (skipped)
    for t in sorted(tickers):
        if t not in statuses and t in ciks and ciks[t]:
            statuses[t] = "skipped"
    for t in sorted(tickers):
        if t not in statuses:
            statuses[t] = "no-cik (yfinance fallback)"
    return statuses


async def refresh_universe_mixed(universe: str | None = None,
                                 tickers: list[str] | None = None,
                                 force: bool = False) -> dict[str, str]:
    """EDGAR first for CIK tickers, yfinance fallback for the rest.
    Mirrors stockstrat's fetch pipeline. Returns {ticker: status}."""
    from . import fundamentals as fundamentals_mod

    if tickers is None:
        tickers = universe_tickers(universe or settings.sim_monthly_universe)
    tickers = list(tickers)
    ciks = await ensure_cik_map(tickers)
    edgar_targets = [t for t in tickers if ciks.get(t)]
    yf_targets = [t for t in tickers if not ciks.get(t)]

    statuses: dict[str, str] = {}
    if edgar_targets:
        ed = await refresh_edgar(edgar_targets, force=force)
        statuses.update({t: f"edgar:{s}" for t, s in ed.items()})
    if yf_targets:
        yf = await fundamentals_mod.refresh_universe(tickers=yf_targets, force=force)
        statuses.update({t: f"yfinance:{s}" for t, s in yf.items()})
    return statuses