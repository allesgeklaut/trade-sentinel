"""Price-age policy: a stale print is NOT a mark.

THE POLICY (one definition, every consumer):

    A stored price print older than ``settings.factor_price_max_age_days``
    calendar days (default 10 — daily bars; ~7 sessions, which covers the
    US/EU holiday gaps including Thanksgiving week) is not a current mark.
    A missing quote is not a valid executable price either, so a forward-
    filled or carried-forward series is only usable while the print BEHIND
    the cell is that young.

What consumers do with a stale print — never silently, and always the
existing fail-closed behaviour of their site:

  * MARKS (live ``daily_core.valuate`` / ``monthly.monthly_valuate``, the
    persisted ``daily_core.backfill`` marks, the research
    ``optimize._daily_core_backtest`` marks): the cell is treated exactly
    like a missing close — the site's cost-basis (avg_cost) fallback takes
    over, and the position is flagged ``stale`` (never a back-dated price
    posing as today's mark).
  * RANKING (``monthly.eligible_frame``): the name is excluded from
    eligibility (``eligible`` False, ``stale`` True) — an indefinitely stale
    listing must not stay silently eligible.
  * FX (``monthly._convert_usd``): a rate print older than the limit
    RELATIVE TO THE PRICE'S OWN TIMESTAMP does not convert anything — the
    amount stays None, like the existing "no prior rate" failure.

This module is deliberately pure: no DB, no I/O, no engine imports (only
``config`` for the default limit), so the live engines, the persisted
replays and the research replay share exactly one definition and it is
unit-testable in isolation.

Timestamp convention: inputs are normalized to naive wall-clock timestamps
by DROPPING the tz (``tz_localize(None)``), which is what the monthly
price/FX lookups already do — candle timestamps round-trip with a UTC tz
attached and the frames are naive, so dropping is lossless here and never
shifts a date.
"""
from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from .config import settings


def _naive(ts: datetime | pd.Timestamp | str | None) -> pd.Timestamp:
    """Timestamp -> naive ``pd.Timestamp`` (tz dropped, NaT for None/NaT)."""
    if ts is None:
        return pd.NaT
    t = pd.Timestamp(ts)
    if pd.isna(t):
        return pd.NaT
    if t.tzinfo is not None:
        t = t.tz_localize(None)
    return t


def is_fresh(print_ts: datetime | pd.Timestamp | None,
             asof_ts: datetime | pd.Timestamp | None,
             max_age_days: int = settings.factor_price_max_age_days) -> bool:
    """True iff ``0 <= (asof_ts - print_ts) <= max_age_days`` calendar days.

    A print exactly ``max_age_days`` old (to the microsecond) is still
    fresh; anything older is not. A FUTURE print (age < 0) is never fresh:
    it is not a known-as-of mark. A missing/NaT print or asof is not fresh.
    Both arguments are normalized to naive wall-clock time, so a tz-aware
    asof and a naive print compare without shifting a day.
    """
    p = _naive(print_ts)
    a = _naive(asof_ts)
    if pd.isna(p) or pd.isna(a):
        return False
    age = a - p
    return timedelta(0) <= age <= timedelta(days=max_age_days)


def freshness_mask(close: pd.DataFrame,
                   asof_ts: datetime | pd.Timestamp | None = None,
                   max_age_days: int = settings.factor_price_max_age_days,
                   ) -> pd.DataFrame:
    """Boolean mask aligned to ``close``: is the print behind a cell fresh?

    ``mask.loc[d, t]`` is True iff the LAST ACTUAL print of ticker ``t`` at
    or before ``d`` (ignoring NaN — a forward-filled cell inherits its
    source print) exists and is no older than ``max_age_days`` **relative to
    ``d`` itself**. That is the question the replay mark helpers ask of a
    forward-filled cell: is this value backed by a fresh print?

    So a NaN cell IS True when the prior print is fresh (the ffilled value is
    usable), and False when there is no prior print at all or when the prior
    print is stale. Rows after ``asof_ts`` (when given) are False — the mask
    never rules on data outside the caller's clock. Computed once for a whole
    frame with last-valid-index arithmetic (a running max over the raw
    non-NaN cells, no ffill), it is O(rows x tickers). One column at a time,
    so the working set is O(rows) regardless of the universe size.

    The frame's index must be datetime-ordered ascending (the loader
    contract); an unsorted frame is sorted internally and the result is
    returned in the input's row order.
    """
    if close is None or close.empty:
        return pd.DataFrame(False, index=getattr(close, "index", None),
                            columns=getattr(close, "columns", None))

    frame = close
    idx = pd.DatetimeIndex(frame.index)
    if not idx.is_monotonic_increasing:
        frame = frame.sort_index()
        idx = pd.DatetimeIndex(frame.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)

    row_ns = idx.to_numpy(dtype="datetime64[ns]")
    n_rows = len(row_ns)
    row_no = np.arange(n_rows)
    horizon_ns = _naive(asof_ts).to_datetime64() if asof_ts is not None else None
    max_age = float(max_age_days)

    columns: dict = {}
    for col in frame.columns:
        values = frame[col].to_numpy(dtype=float)
        # Row number of the most recent ACTUAL print at or before each row: a
        # running maximum of "this row's number where the cell is a real
        # print, else -1". (A plain cumsum would count prints but not locate
        # them; ffill would erase the print dates.)
        last_row = np.maximum.accumulate(np.where(~np.isnan(values), row_no, -1))
        has = last_row >= 0
        last_ns = np.where(has, row_ns[np.maximum(last_row, 0)],
                           np.datetime64("NaT", "ns"))
        age_days = (row_ns - last_ns) / np.timedelta64(1, "D")
        mask = (has
                & np.isfinite(age_days)
                & (age_days >= 0.0)
                & (age_days <= max_age))
        if horizon_ns is not None:
            mask &= row_ns <= horizon_ns
        columns[col] = mask

    out = pd.DataFrame(columns, index=frame.index)
    if frame is not close:
        out = out.reindex(index=close.index, columns=close.columns)
    return out
