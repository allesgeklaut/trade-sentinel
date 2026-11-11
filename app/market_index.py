"""Synthetic equal-weight market gate index — the ONE shared market-overlay series.

Every live/replay/research protection overlay (the deployment gate and the
gradient "good times" arm) compares a market series against its N-day SMA.
That series must be an economic return index, not the average of raw nominal
prices: ``close.ffill().mean(axis=1)`` weighs a $1,000 share 100x more than a
$10 share, so a stock split or a different listing currency can flip the
signal without changing any investor's return (audit P1-A).

``equal_weight_return_index`` builds a daily-rebalanced, equal-weighted
arithmetic simple-return index chained from 100:

  * Prices are validated first: non-finite, non-positive and missing cells
    are unavailable. No unit conversion happens — the index is scale-free by
    construction because only price RATIOS enter.
  * A name's last known valid close is carried forward for return assignment
    only. A temporarily missing quote therefore contributes a ZERO return
    that day (it stays in the cross-section and does not silently shift
    weight to the names that happen to have a quote). Nothing is ever
    back-filled: a name has no weight until its first observed valid close.
  * The period return is the simple average of the per-name simple returns
    over names that have both a valid prior close and a valid current close.
    A first observed price contributes no return until a prior price exists;
    a period with no usable pair contributes no return (the index is flat),
    so absent data is never converted into invented performance.
  * The index starts at 100.0 on the first row (there is no prior row) and is
    chained multiplicatively afterwards.

The result is strictly causal: the value at any date depends only on that
date and earlier rows, so appending future candles cannot change history, and
multiplying an individual column by any positive constant cannot change the
index (or the gate outcome derived from it).

This is a SYNTHETIC UNIVERSE GATE, not an investable historical-membership
benchmark. The frame's columns are whoever the caller loaded; the helper has
no point-in-time constituent history, no survivorship correction and no
corporate-action/delisting calendar. In particular, a name that stops quoting
contributes zero returns until it reappears (its long-gap move is attributed
on the reappearance session). Delisting handling and a maximum stale-quote
horizon are separate Phase 2 work and deliberately NOT modelled here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["BASE", "equal_weight_return_index"]

BASE = 100.0
_NAME = "equal_weight_return_index"


def equal_weight_return_index(close: pd.DataFrame, *, base: float = BASE) -> pd.Series:
    """Daily-rebalanced equal-weight simple-return index chained from *base*.

    Pure and causal: no I/O, no state, no back-fill. See the module docstring
    for the construction rules and the synthetic-gate caveat.

    Empty input (no rows) returns an empty float Series. A frame with rows but
    no usable prices (all NaN / non-finite / non-positive, or no columns)
    returns a flat *base* series: an all-unavailable period invents no return.
    """
    if close is None:
        return pd.Series(dtype="float64", name=_NAME)
    index = pd.Index(close.index)
    if len(index) == 0:
        return pd.Series(dtype="float64", index=index, name=_NAME)
    if close.shape[1] == 0:
        return pd.Series(float(base), index=index, dtype="float64", name=_NAME)

    # Validate, don't transform: keep only finite, strictly positive prices.
    # Everything else is "no quote" and becomes NaN (ffill will carry a
    # previously known valid close over it, if there is one).
    px = close.apply(pd.to_numeric, errors="coerce").astype("float64")
    arr = px.to_numpy(dtype="float64", copy=True)
    arr[~np.isfinite(arr) | (arr <= 0.0)] = np.nan
    valid = pd.DataFrame(arr, index=index, columns=px.columns)

    # Forward carry is CAUSAL and only defines a current price for a name
    # whose prior close is already known; leading NaN stays NaN (no bfill).
    carried = valid.ffill()
    with np.errstate(divide="ignore", invalid="ignore"):
        prev = carried.shift(1)
        # Both sides are either a finite positive carried close or NaN, so
        # the ratio is finite where a name has prior+current prices and NaN
        # otherwise (first observation, or a period before inception).
        rets = (carried / prev) - 1.0

    # Names with a valid pair are averaged with equal weight; a temporarily
    # missing quote is carried flat above, so it counts as a zero return
    # instead of being dropped. An all-unavailable row means "no return".
    period = rets.mean(axis=1, skipna=True)
    period = period.where(np.isfinite(period), 0.0)

    values = float(base) * (1.0 + period.to_numpy(dtype="float64")).cumprod()
    return pd.Series(values, index=index, dtype="float64", name=_NAME)
