"""The session fill contract — the single place it is documented.

FROZEN CONTRACT (do not redesign without orchestrator sign-off):

The scheduler cycle runs once per session at 23:30 UTC (``sim_run_hour``),
after session F's bar is complete and cached. Session F is the cycle's own
session (the run date); S is the previous *completed* session.

  1. Decision inputs are strictly data <= S's close: every candle handed to
     the indicator engine is truncated to sessions <= S. F's bar never
     informs the decision, so mutating or appending F's candle cannot change
     what is proposed.
  2. Fills execute at F's OPEN — strict same-session bar per ticker: the
     fill price is read from the stored bar with an exact date match
     (``open_at`` — never forward-filled, never a stale close; a missing
     quote is not an executable price). If F has no bar for a ticker, there
     is NO fill — the decision is simply re-taken at the next cycle. A
     per-session guard makes a second cycle within the same session a
     no-op, so a re-taken decision cannot double-fill.
  3. Marks/valuation use F's close (the completed session bar).
  4. A stop level known at S's close is a standing order for F: fill at
     F's open when open <= level, else at the level when F's low <= level,
     else no stop fill this session.
  5. The percent stop is frozen at entry for the live sim: it is persisted
     on the position at entry (``SimPosition.stop_price``) and never
     re-anchored to a blended average cost — a top-up cannot loosen it.
  6. The ATR stop is a position-specific monotone ratchet:
     ``stop_t = max(candidate level over all completed bars since entry)``
     — it can only tighten, never loosen. It is recomputed per cycle from
     the entry anchor (``SimPosition.opened_at`` = the actual first-buy
     session date) so it needs no persistence.
  7. No pending-order persistence: the deterministic decision is recomputed
     at F's cycle from frozen inputs and is bit-identical on a re-run; an
     unfilled proposal is simply dropped and re-taken next session. The
     LLM's frozen-decision ledger is a Phase 2 item — for now the LLM path
     uses the same data <= S inputs and fills at F's open, but its
     *reasoning* is timestamped to the cycle that produced it (an honest,
     documented limitation, not a replayable artifact).

Manual cycles outside the scheduled window: F's open is stable intraday, so
open fills stay correct, but F's *low* is only final after the session
completes — a manual cycle before the close may trigger a stop fill on a
not-yet-final low. The scheduled 23:30 UTC run is the contract path.

Prices here are RAW cached candle values in the listing's own currency (the
live sim ledger is raw-price, matching ``market.candles``); the factor
portfolios' FX-converting loader remains ``monthly.load_frames``.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import pandas as pd
from sqlalchemy import select

# In-function imports of ``db`` (Session/Candle) on purpose: tests patch
# ``app.db.Session`` per-fixture, and a call-time import sees the patch.


class SessionFrames(NamedTuple):
    """Aligned per-field frames from one candle query.

    All five frames share one normalized naive date index (the union of the
    stored sessions); a ticker without a bar on a date carries NaN there.
    ``high``/``low`` back the ATR/ADX indicators and the stop fill rule
    (4); ``close`` is the decision/mark series; ``open`` is the fill price.
    """

    open: pd.DataFrame
    high: pd.DataFrame
    low: pd.DataFrame
    close: pd.DataFrame
    volume: pd.DataFrame

    @property
    def index(self) -> pd.DatetimeIndex:
        """The union session index (any frame's index — they are aligned)."""
        return pd.DatetimeIndex(self.close.index)


def _norm_date(d) -> pd.Timestamp:
    """Normalize a date-like to a naive midnight ``pd.Timestamp``."""
    ts = pd.Timestamp(d)
    if ts.tzinfo is not None:
        ts = ts.tz_localize(None)
    return ts.normalize()


def prev_session(index, asof) -> pd.Timestamp | None:
    """Previous completed session S strictly before *asof*, from a frame index.

    Primary source is the stored candle index — the cache is the ground
    truth of which sessions have completed bars. When the index carries no
    bar before *asof* (e.g. a warmup-bounded load over an empty cache),
    fall back to the NYSE calendar (``sim.is_trading_day``, lazy import —
    ``app.sim`` sits above this module): walk back over non-trading days
    from the day before *asof*, capped at two weeks so an unusable calendar
    can never spin. ``monthly.month_last_trading_day`` is the month-end
    variant of the same idea and is used by the factor schedulers, not
    here. Returns ``None`` when no prior session can be established — the
    caller must then take no decision (and make no fill).
    """
    idx = pd.DatetimeIndex(pd.to_datetime(pd.Index(index))).normalize()
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    target = _norm_date(asof)
    earlier = idx[idx < target]
    if len(earlier):
        return pd.Timestamp(earlier.max())
    # Fallback: calendar walk over non-trading days (holiday clusters etc.).
    from .sim import is_trading_day  # lazy: avoids the import cycle

    day = (target - pd.Timedelta(days=1)).date()
    for _ in range(14):
        if is_trading_day(pd.Timestamp(day).to_pydatetime()):
            return pd.Timestamp(day)
        day = day - pd.Timedelta(days=1)
    return None


def open_at(frame: pd.DataFrame, d, ticker: str) -> float | None:
    """Strict exact-date cell lookup on a session frame — no ffill, ever.

    Reads ``frame.at[d, ticker]`` and returns it as a float, or ``None``
    when the date is absent or the cell is NaN. The strictness is the
    point: a missing quote is not an executable price, and a stale or
    forward-filled close must never fill a trade. Works for any of the
    aligned frames (open / high / low / close / volume) — the fill rule
    uses it for the open and the stop-trigger low.
    """
    if frame is None or not len(frame.index):
        return None
    try:
        v = frame.at[_norm_date(d), ticker]
    except KeyError:
        return None
    if pd.isna(v):
        return None
    return float(np.asarray(v).reshape(-1)[0])


async def load_open_frames(tickers: list[str], asof=None,
                           start=None) -> SessionFrames:
    """One query returning aligned open/high/low/close/volume frames.

    ``monthly.load_frames`` returns close+volume only; the fill contract
    needs the open (fill price), the low (stop trigger) and the high/low
    (ATR/ADX indicator math), so this loader selects all five fields in a
    single query and pivots them to shared-index frames. Bounds are
    inclusive on both ends and use the same midnight-naive convention as
    the stored timestamps. Prices are raw cached values — see the module
    docstring for the FX note.
    """
    from .db import Candle, Session

    async with Session() as s:
        q = (select(Candle.ticker, Candle.timestamp, Candle.open, Candle.high,
                    Candle.low, Candle.close, Candle.volume)
             .where(Candle.ticker.in_(list(tickers)))
             .order_by(Candle.ticker, Candle.timestamp))
        if start is not None:
            q = q.where(Candle.timestamp >= _norm_date(start).to_pydatetime())
        if asof is not None:
            q = q.where(Candle.timestamp <= _norm_date(asof).to_pydatetime())
        rows = list((await s.execute(q)).all())

    fields: dict[str, dict[str, dict[pd.Timestamp, float]]] = {
        k: {} for k in ("open", "high", "low", "close", "volume")}
    for r in rows:
        ts = pd.Timestamp(r.timestamp)
        if ts.tzinfo is not None:
            ts = ts.tz_localize(None)
        ts = ts.normalize()
        for k in fields:
            val = getattr(r, k)
            if val is not None:
                fields[k].setdefault(r.ticker, {})[ts] = float(val)

    def _frame(col: dict[str, dict[pd.Timestamp, float]]) -> pd.DataFrame:
        if not col:
            return pd.DataFrame(index=pd.DatetimeIndex([], name="timestamp"))
        return pd.DataFrame({t: pd.Series(v).sort_index()
                             for t, v in col.items()})

    f_open, f_high, f_low, f_close, f_vol = (
        _frame(fields["open"]), _frame(fields["high"]),
        _frame(fields["low"]), _frame(fields["close"]),
        _frame(fields["volume"]))
    # Align every frame on the union index so exact-date lookups and the
    # per-ticker row rebuild see the same calendar.
    union = f_open.index.union(f_close.index).union(f_high.index) \
        .union(f_low.index).union(f_vol.index).sort_values()
    f_open = f_open.reindex(union)
    f_high = f_high.reindex(union)
    f_low = f_low.reindex(union)
    f_close = f_close.reindex(union)
    f_vol = f_vol.reindex(union)
    return SessionFrames(open=f_open, high=f_high, low=f_low,
                         close=f_close, volume=f_vol)


def rows_for(frames: SessionFrames, ticker: str) -> list[dict]:
    """Rebuild ``analysis.compute``-shaped OHLCV rows for one ticker.

    Drops dates where the ticker has no close — the bar does not exist for
    that ticker and is never forward-filled. Output rows are oldest-first
    with ``time`` as "YYYY-MM-DD", exactly what ``analysis.signal_series``
    and ``analysis.compute`` consume.
    """
    if ticker not in frames.close.columns:
        return []
    out: list[dict] = []
    for ts in frames.close.index:
        close = frames.close.at[ts, ticker]
        if pd.isna(close):
            continue
        out.append({
            "time": ts.strftime("%Y-%m-%d"),
            "open": open_at(frames.open, ts, ticker),
            "high": open_at(frames.high, ts, ticker),
            "low": open_at(frames.low, ts, ticker),
            "close": float(np.asarray(close).reshape(-1)[0]),
            "volume": open_at(frames.volume, ts, ticker) or 0.0,
        })
    return out
