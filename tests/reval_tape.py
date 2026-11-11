"""Shared synthetic market tape for the Phase-4 reval harness tests (slice 4A).

This is the NEW-tests-only shared helper for ``tests/test_reval.py`` — the
existing test modules keep their in-file "copied tape, do not import" convention
and are untouched. Everything here is deterministic from ``seed``: no network,
no provider, no DB writes beyond the explicit seeding helper.

The tape contains:

* ``TAPE_TICKERS`` — N equity tickers with piecewise regime paths (real rank
  flips at month-ends) plus a per-ticker phase-shifted sine wiggle, and an
  OPEN != CLOSE on every bar (modelled on ``tests/test_cadence_parity.py``);
  two of them are NON-USD listings (``X01.L`` quoted in pence, ``X02.DE`` in
  EUR) so the FX cost leg is exercised;
* the FX pseudo-tickers those listings need for the USD conversion
  (``GBPUSD=X``, ``EURUSD=X``) — ``monthly.load_ohlcv_frames`` fails closed
  without them, so the tape must carry them;
* ``BENCH`` — the benchmark ETF the DCA control buys (its own path, open !=
  close, USD-native);
* synthetic fundamentals with filed dates (all-positive ROE decaying by ticker
  index so the qv-mom ranking is stable yet distinct per name), in the exact
  schema ``fundamentals.load_fundamentals`` reads from the ``fundamentals``
  table;
* a synthetic universe file (``UNIVERSE_NAME``) listing the equity tickers;
* a deliberate ONE-SESSION GAP: ``GAP_TICKER`` (T07) is missing the candle on
  the session nearest ``GAP_ANCHOR`` (2025-10-15) that every other member
  has — the slice-4B addendum's alignment-robustness regression trigger
  (``tape["gap"]`` records it).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from app import sim as sim_mod
from app.db import Candle, Fundamental

TAPE_TICKERS = ([f"T{i:02d}" for i in range(1, 23)] + ["X01.L", "X02.DE"])
BENCH = "URTH"
FX_PAIRS = ["GBPUSD=X", "EURUSD=X"]
UNIVERSE_NAME = "reval-tape"
TAPE_START = "2024-01-02"
TAPE_END = "2026-10-02"

# Slice-4B addendum: one member is deliberately MISSING one candle on a date
# that every other member HAS (mid-window). pandas aligns the per-ticker
# Series in ``reval._load_inputs``'s raw_close on the UNION of date indexes, so
# this gap produces a NaN cell for GAP_TICKER on GAP_DATE — the sim arm's
# weight reconstruction must carry the last KNOWN finite close there instead
# of poisoning the weights (regression test in tests/test_reval_controls.py).
GAP_TICKER = "T07"
GAP_ANCHOR = "2025-10-15"   # the tape session nearest to this date gets the gap

# Per-ticker drift regimes: (ticker, start date, daily factor). Two names
# surge / sink at fixed dates so month-end frames genuinely re-rank. T01 is a
# clean strong uptrend so the TECHNICAL sim arm gets BUY signals on the tape.
_REGIMES = [
    ("T05", "2025-12-31", 1.005),   # momentum takeover
    ("T06", "2026-01-30", 0.995),   # sinks out of the band
    ("T02", "2026-04-01", 0.99),    # held name falls out of the band
]
_STRONG_DRIFT = {"T01": 0.008}  # clean uptrend for the sim arm's BUY signals


def sessions() -> pd.DatetimeIndex:
    """The tape's session calendar: business days that are US trading days."""
    dates = pd.bdate_range(TAPE_START, TAPE_END)
    return pd.DatetimeIndex(
        [d for d in dates if sim_mod.is_trading_day(d.to_pydatetime())])


def _ticker_salt(ticker: str) -> int:
    """Stable per-ticker salt (built-in ``hash()`` is process-randomized and
    would make the tape non-deterministic across runs)."""
    return sum((i + 1) * ord(c) for i, c in enumerate(ticker)) % 100003


def _path(ticker: str, idx: pd.DatetimeIndex, base: float, seed: int,
          salt: int) -> pd.DataFrame:
    """Piecewise-regime base path + phase-shifted sine wiggle; open != close."""
    rng = np.random.default_rng(seed + salt)
    n = len(idx)
    drift = _STRONG_DRIFT.get(ticker, 0.0004 + 0.00013 * (salt % 7))  # distinct drifts
    steps = np.arange(n)
    close = base * np.exp(drift * steps)
    for t, start, factor in _REGIMES:
        if t != ticker:
            continue
        mask = idx >= pd.Timestamp(start)
        k = np.cumsum(mask)  # 0 before the regime starts
        close = np.where(mask, base * factor ** k * np.exp(drift * steps), close)
    phase = 0.7 * (salt % 8)
    wig = 1.0 + 0.012 * np.sin(2 * np.pi * np.arange(n) / 9.0 + phase)
    close = close * wig * (1.0 + rng.normal(0.0, 0.002, size=n))
    close = np.maximum(close, base * 0.05)
    open_px = close * (1.0 - 0.002 * (np.arange(n) % 5) / 5.0)
    high = np.maximum(open_px, close) * 1.002
    low = np.minimum(open_px, close) * 0.998
    return pd.DataFrame({"open": open_px, "high": high, "low": low,
                         "close": close, "volume": 1e6 * (1 + 0.1 * rng.random(n))},
                        index=idx)


def make_tape(seed: int = 11) -> dict:
    """The full deterministic tape.

    Returns ``{"sessions", "candles": {ticker: [row-dicts]}, "fundamentals":
    {ticker: {tag: [entry-dicts]}}, "universe_text": str}`` — row dicts in the
    shape ``market.candles()`` produces (``time``/``open``/``high``/``low``/
    ``close``/``volume``), fundamentals entries in the ``load_fundamentals``
    read schema (``start``/``end``/``filed``/``val``).
    """
    idx = sessions()
    candles: dict[str, list[dict]] = {}
    for i, t in enumerate(TAPE_TICKERS):
        # .L quotes pence: keep the USD-converted mcap above the 5e9 eligibility
        # floor (4200p = GBP 42 = ~$52.5 x 1e8 shares ~ 5.2e9)
        base = 4200.0 + 90.0 * i if t.endswith(".L") else (60.0 + 11.0 * i)
        df = _path(t, idx, base, seed, _ticker_salt(t))
        candles[t] = [{"time": d.strftime("%Y-%m-%d"),
                       "open": round(float(r.open), 4), "high": round(float(r.high), 4),
                       "low": round(float(r.low), 4), "close": round(float(r.close), 4),
                       "volume": round(float(r.volume), 1)}
                      for d, r in df.iterrows()]
    # Benchmark ETF: steady uptrend with its own wiggle, open != close, USD.
    df = _path(BENCH, idx, 100.0, seed + 1, _ticker_salt(BENCH))
    candles[BENCH] = [{"time": d.strftime("%Y-%m-%d"),
                       "open": round(float(r.open), 4), "high": round(float(r.high), 4),
                       "low": round(float(r.low), 4), "close": round(float(r.close), 4),
                       "volume": round(float(r.volume), 1)}
                      for d, r in df.iterrows()]
    # FX pseudo-tickers: near-constant rates (the conversion only needs a
    # usable causal rate per date).
    for j, pair in enumerate(FX_PAIRS):
        rate = 1.25 + 0.05 * j
        candles[pair] = [{"time": d.strftime("%Y-%m-%d"), "open": rate, "high": rate * 1.001,
                          "low": rate * 0.999, "close": rate,
                          "volume": 1e7} for d in idx]

    fundamentals: dict[str, dict[str, list[dict]]] = {}
    for i, t in enumerate(TAPE_TICKERS):
        roe = 0.50 - 0.018 * i  # distinct, positive ROE per name (ranking input)
        fundamentals[t] = {
            "StockholdersEquity": [
                {"start": None, "end": "2023-12-31", "filed": "2024-02-14", "val": 5e9},
                {"start": None, "end": "2024-06-30", "filed": "2024-08-14", "val": 5.2e9},
            ],
            "NetIncomeLoss": [
                {"start": "2023-01-01", "end": "2023-12-31", "filed": "2024-02-14",
                 "val": 5e9 * roe},
                {"start": "2024-01-01", "end": "2024-12-31", "filed": "2025-02-14",
                 "val": 5.2e9 * roe},
            ],
            "CommonStockSharesOutstanding": [
                {"start": None, "end": "2023-12-31", "filed": "2024-02-14", "val": 1e8},
                {"start": None, "end": "2024-06-30", "filed": "2024-08-14", "val": 1e8},
            ],
            "NetCashProvidedByUsedInOperatingActivities": [
                {"start": "2023-01-01", "end": "2023-12-31", "filed": "2024-02-14", "val": 1.0e9},
                {"start": "2024-01-01", "end": "2024-12-31", "filed": "2025-02-14", "val": 1.1e9},
            ],
            "PaymentsToAcquirePropertyPlantAndEquipment": [
                {"start": "2023-01-01", "end": "2023-12-31", "filed": "2024-02-14", "val": -2e8},
                {"start": "2024-01-01", "end": "2024-12-31", "filed": "2025-02-14", "val": -2.2e8},
            ],
        }

    universe_text = (
        "# synthetic reval tape universe (tests only)\n"
        + "\n".join(TAPE_TICKERS) + "\n"
    )
    # the deliberate one-session gap (deterministic: nearest tape session to
    # GAP_ANCHOR that other members have)
    gap_ts = min(idx, key=lambda d: abs((d - pd.Timestamp(GAP_ANCHOR)).days))
    gap_date = gap_ts.strftime("%Y-%m-%d")
    candles[GAP_TICKER] = [r for r in candles[GAP_TICKER] if r["time"] != gap_date]
    return {"sessions": idx, "candles": candles,
            "fundamentals": fundamentals, "universe_text": universe_text,
            "tickers": list(TAPE_TICKERS), "bench": BENCH, "fx_pairs": list(FX_PAIRS),
            "gap": {"ticker": GAP_TICKER, "date": gap_date}}


def universe_file(directory: str | Path, tape: dict) -> Path:
    """Write the tape's universe file into ``directory``; returns its path."""
    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{UNIVERSE_NAME}.txt"
    p.write_text(tape["universe_text"])
    return p


async def seed_db(session_factory, tape: dict) -> None:
    """Insert the tape's candles + fundamentals through ``session_factory``."""
    candle_rows = []
    for t, rows in tape["candles"].items():
        for r in rows:
            candle_rows.append({
                "ticker": t, "timestamp": pd.Timestamp(r["time"]).to_pydatetime(),
                "open": r["open"], "high": r["high"], "low": r["low"],
                "close": r["close"], "volume": r["volume"],
            })
    fund_rows = []
    for t, tags in tape["fundamentals"].items():
        for tag, entries in tags.items():
            for e in entries:
                fund_rows.append({"ticker": t, "tag": tag, "start": e["start"],
                                  "end": e["end"], "filed": e["filed"], "val": e["val"],
                                  "currency": "USD", "source": "edgar"})
    async with session_factory() as s:
        await s.execute(Candle.__table__.insert(), candle_rows)
        await s.execute(Fundamental.__table__.insert(), fund_rows)
        await s.commit()
