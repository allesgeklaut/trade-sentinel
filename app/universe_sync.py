"""Auto-refresh universe files from index/ETF holdings.

Currently only the S&P 500: the iShares Core S&P 500 ETF (IVV) holdings CSV is
effectively the index membership itself (IVV fully replicates), and iShares
publishes it daily. Each sync writes a generated ``sp500.txt`` into the writable
universe dir (``/data/universes`` in the container), which shadows the repo
bootstrap list — see ``screener.universe_path``.

Design notes:
  * One HTTP request per sync: no Yahoo / Twelve Data / EDGAR provider load.
  * The generated file is written atomically; a fetch/parse failure leaves the
    previous list untouched (never truncate to an empty universe).
  * Current members only -> NOT survivorship-free (same caveat as the repo
    list): a backfill ranks on today's membership, not point-in-time MSCI/S&P
    membership. Documented in the README.
"""
import asyncio
import csv
import io
import logging
import os
import re
import time
import urllib.error
import urllib.request
from datetime import UTC, datetime

from . import screener
from .config import settings

logger = logging.getLogger("trade_sentinel.universe_sync")

SP500_UNIVERSE = "sp500"

# US share-class symbols use '-' (BRK-B); the holdings CSV writes them with a
# space ("BRK B") or a dot. Normalize both, then validate defensively.
_SPACE_DOT_RE = re.compile(r"[ .]")
_SYMBOL_RE = re.compile(r"^[A-Z0-9.\-]{1,10}$")

# Sanity floor: a real S&P 500 list is ~500 names. A much smaller parse means
# the CSV shape changed or the download was truncated — never overwrite with it.
_MIN_CONSTITUENTS = 400

# Exchange values that mean "no live listing": residual stubs of acquired or
# delisted names linger in the holdings file marked "Equity" (e.g. HOLOGIC
# after its buyout: weight 0.00, price $0.01, "NO MARKET"). Dropping them keeps
# non-constituents out of the screener/sim universe.
_EXCHANGE_PLACEHOLDERS = {"", "-", "NO MARKET", "NO MARKET (E.G. UNLISTED)", "IFLL"}

_UA = {"User-Agent": "Mozilla/5.0 (trade-sentinel research)", "Accept": "*/*"}

_sync_lock: asyncio.Lock = asyncio.Lock()


def _http_get(url: str, retries: int = 3, backoff: float = 2.0) -> bytes:
    last = None
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers=_UA)
            with urllib.request.urlopen(req, timeout=30) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            last = e
            if i < retries - 1:
                time.sleep(backoff * (i + 1))
        except Exception as e:  # noqa: BLE001 - network errors vary
            last = e
            if i < retries - 1:
                time.sleep(backoff * (i + 1))
    raise RuntimeError(f"GET {url} failed after {retries} tries: {last}")


def _normalize_symbol(raw: str) -> str | None:
    sym = _SPACE_DOT_RE.sub("-", (raw or "").strip().upper())
    return sym if _SYMBOL_RE.match(sym) else None


def parse_constituents(payload: bytes) -> tuple[list[str], str | None]:
    """Parse an iShares holdings CSV into (sorted tickers, holdings-as-of).

    Keeps only ``Asset Class == "Equity"`` rows (drops the cash / money-market /
    futures lines) and normalizes each ticker to the Yahoo convention. Raises
    ValueError when the expected header row or columns are missing.
    """
    text = payload.decode("utf-8-sig", errors="replace")
    rows = list(csv.reader(io.StringIO(text)))

    hdr_i = next((i for i, r in enumerate(rows) if r and r[0].strip() == "Ticker"), None)
    if hdr_i is None:
        raise ValueError("holdings CSV: no 'Ticker' header row found")
    header = [c.strip() for c in rows[hdr_i]]
    try:
        col = {name: header.index(name) for name in ("Ticker", "Asset Class")}
    except ValueError as e:
        raise ValueError(f"holdings CSV: missing expected column ({e})") from e
    # Optional guards: applied only when the columns exist, so the parser still
    # works if iShares drops or reorders them.
    ex_col = header.index("Exchange") if "Exchange" in header else None
    wt_col = header.index("Weight (%)") if "Weight (%)" in header else None

    # "Fund Holdings as of" lives in the preamble, e.g. Fund Holdings as of,"Sep 15, 2026"
    as_of = None
    for r in rows[:hdr_i]:
        if r and r[0].strip().startswith("Fund Holdings as of"):
            as_of = (r[1].strip() if len(r) > 1 else "") or None
            break

    needed = max(col.values())

    def _is_constituent(r: list[str]) -> bool:
        if r[col["Asset Class"]].strip() != "Equity":
            return False
        if ex_col is not None and len(r) > ex_col:
            if r[ex_col].strip().upper() in _EXCHANGE_PLACEHOLDERS:
                return False
        if wt_col is not None and len(r) > wt_col:
            try:
                if float(r[wt_col].strip().replace(",", "")) <= 0:
                    return False
            except ValueError:
                pass  # unexpected format: don't drop a name on that basis
        return True

    symbols: set[str] = set()
    for r in rows[hdr_i + 1:]:
        if len(r) <= needed or not r[0].strip():
            continue
        if not _is_constituent(r):
            continue
        sym = _normalize_symbol(r[col["Ticker"]])
        if sym:
            symbols.add(sym)
    if not symbols:
        raise ValueError("holdings CSV: no equity tickers parsed")
    return sorted(symbols), as_of


def _render(symbols: list[str], as_of: str | None, url: str) -> str:
    now = datetime.now(UTC).isoformat(timespec="seconds")
    lines = [
        "# S&P 500 current constituents — AUTO-GENERATED, do not edit by hand.",
        "# Source: iShares Core S&P 500 ETF (IVV) holdings CSV",
        f"#   {url}",
        f"# Retrieved: {now}",
    ]
    if as_of:
        lines.append(f"# Fund holdings as of: {as_of}")
    lines += [
        f"# {len(symbols)} tickers. Current members only -> NOT survivorship-free.",
        '# "S&P 500" is a trademark of S&P Dow Jones Indices. Research universe',
        "# only: inclusion is not a recommendation.",
    ]
    return "\n".join(lines) + "\n\n" + "\n".join(symbols) + "\n"


async def sync_sp500() -> dict:
    """Fetch the current S&P 500 membership and rewrite the generated universe.

    Returns {universe, as_of, count, added, removed, changed, path, url}. Raises
    on fetch/parse failure, leaving any existing file untouched.
    """
    async with _sync_lock:
        url = settings.sp500_holdings_url
        payload = await asyncio.to_thread(_http_get, url)
        symbols, as_of = parse_constituents(payload)
        if len(symbols) < _MIN_CONSTITUENTS:
            raise ValueError(
                f"refusing to write {SP500_UNIVERSE}: only {len(symbols)} "
                f"tickers parsed (expected >= {_MIN_CONSTITUENTS})"
            )

        try:
            old = set(screener.tickers(SP500_UNIVERSE))
        except ValueError:
            old = set()
        new = set(symbols)
        added = sorted(new - old)
        removed = sorted(old - new)

        path = screener.extra_universes_dir() / f"{SP500_UNIVERSE}.txt"
        path.parent.mkdir(parents=True, exist_ok=True)
        body = _render(symbols, as_of, url)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(body, encoding="utf-8")
        os.replace(tmp, path)  # atomic: readers see either the old or new file

        logger.info("Universe sync %s: %d tickers (as of %s), +%d/-%d",
                    SP500_UNIVERSE, len(symbols), as_of or "?", len(added), len(removed))
        return {
            "universe": SP500_UNIVERSE,
            "as_of": as_of,
            "count": len(symbols),
            "added": added,
            "removed": removed,
            "changed": bool(added or removed),
            "path": str(path),
            "url": url,
        }
