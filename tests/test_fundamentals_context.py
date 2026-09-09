"""Tests for fundamentals-in-daily-basis features:

- ``monthly.quality_snapshot``: point-in-time ROE / P-FCF math, missing-data
  semantics (None = unknown, never bad), TTM building from quarterly facts.
- ``strategy.propose_trades`` quality guard: blocks known non-positive ROE,
  passes missing fundamentals (ETFs like GLD), respects the off-switch, and
  also guards the relaxed-HOLD fallback.
- ``sim._build_llm_context``: ROE / P-FCF columns render when the quality map
  is provided, missing values render as "-", table unchanged when off.
"""

from __future__ import annotations

import pandas as pd
import pytest

from app import sim
from app.monthly import quality_snapshot
from app.strategy import StrategyParams, propose_trades


# pandas-stubs types Timestamp constructors as returning Timestamp | NaTType,
# which trips pyright against quality_snapshot's `asof: pd.Timestamp` — same
# pre-existing noise pattern as app/monthly.py. One helper keeps it in one place.
def _ts(s: str) -> pd.Timestamp:
    return pd.Timestamp(s)  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# quality_snapshot
# ---------------------------------------------------------------------------

def _facts(ni: list[dict], eq: list[dict], ocf: list[dict] | None = None,
           capex: list[dict] | None = None, shares: list[dict] | None = None,
           dei_shares: list[dict] | None = None) -> dict[str, list[dict]]:
    rec: dict[str, list[dict]] = {
        "NetIncomeLoss": ni,
        "StockholdersEquity": eq,
    }
    if ocf is not None:
        rec["NetCashProvidedByUsedInOperatingActivities"] = ocf
    if capex is not None:
        rec["PaymentsToAcquirePropertyPlantAndEquipment"] = capex
    if shares is not None:
        rec["CommonStockSharesOutstanding"] = shares
    if dei_shares is not None:
        rec["EntityCommonStockSharesOutstanding"] = dei_shares
    return rec


class TestQualitySnapshot:
    def test_basic_roe_and_p_fcf(self):
        fund = {
            "AAPL": _facts(
                ni=[{"start": "2025-01-01", "end": "2025-12-31",
                     "filed": "2026-01-30", "val": 100.0}],
                eq=[{"start": None, "end": "2025-12-31",
                     "filed": "2026-01-30", "val": 400.0}],
                ocf=[{"start": "2025-01-01", "end": "2025-12-31",
                      "filed": "2026-01-30", "val": 120.0}],
                capex=[{"start": "2025-01-01", "end": "2025-12-31",
                        "filed": "2026-01-30", "val": -20.0}],
                shares=[{"start": None, "end": "2025-12-31",
                         "filed": "2026-01-30", "val": 10.0}],
            ),
        }
        asof = _ts("2026-06-30")
        q = quality_snapshot(fund, "AAPL", asof, close=50.0)
        assert q is not None
        assert q["roe"] == pytest.approx(100.0 / 400.0)
        # mcap = 50 * 10 = 500; FCF = 120 + (-20) = 100
        assert q["p_fcf"] == pytest.approx(500.0 / 100.0)

    def test_no_fundamentals_returns_none(self):
        # GLD-style: ETF with no facts at all — unknown, not bad.
        assert quality_snapshot({}, "GLD", _ts("2026-06-30"), close=90.0) is None

    def test_negative_equity_roe_is_none(self):
        fund = {"BAD": _facts(
            ni=[{"start": "2025-01-01", "end": "2025-12-31",
                 "filed": "2026-01-30", "val": -50.0}],
            eq=[{"start": None, "end": "2025-12-31",
                 "filed": "2026-01-30", "val": -100.0}],  # negative equity
        )}
        q = quality_snapshot(fund, "BAD", _ts("2026-06-30"))
        assert q is not None
        assert q["roe"] is None  # not computable, NOT reported as <= 0
        assert q["p_fcf"] is None

    def test_negative_roe_is_reported(self):
        fund = {"LOSS": _facts(
            ni=[{"start": "2025-01-01", "end": "2025-12-31",
                 "filed": "2026-01-30", "val": -50.0}],
            eq=[{"start": None, "end": "2025-12-31",
                 "filed": "2026-01-30", "val": 400.0}],
        )}
        q = quality_snapshot(fund, "LOSS", _ts("2026-06-30"))
        assert q is not None
        assert q["roe"] == pytest.approx(-50.0 / 400.0)

    def test_roe_none_when_ni_missing(self):
        fund = {"NI": _facts(
            ni=[],
            eq=[{"start": None, "end": "2025-12-31",
                 "filed": "2026-01-30", "val": 400.0}],
        )}
        q = quality_snapshot(fund, "NI", _ts("2026-06-30"))
        assert q is not None
        assert q["roe"] is None

    def test_p_fcf_none_when_no_price(self):
        fund = {"NOPX": _facts(
            ni=[{"start": "2025-01-01", "end": "2025-12-31",
                 "filed": "2026-01-30", "val": 100.0}],
            eq=[{"start": None, "end": "2025-12-31",
                 "filed": "2026-01-30", "val": 400.0}],
            ocf=[{"start": "2025-01-01", "end": "2025-12-31",
                  "filed": "2026-01-30", "val": 120.0}],
            shares=[{"start": None, "end": "2025-12-31",
                     "filed": "2026-01-30", "val": 10.0}],
        )}
        q = quality_snapshot(fund, "NOPX", _ts("2026-06-30"), close=None)
        assert q is not None
        assert q["roe"] == pytest.approx(0.25)
        assert q["p_fcf"] is None

    def test_p_fcf_none_when_fcf_negative(self):
        fund = {"BURN": _facts(
            ni=[{"start": "2025-01-01", "end": "2025-12-31",
                 "filed": "2026-01-30", "val": 100.0}],
            eq=[{"start": None, "end": "2025-12-31",
                 "filed": "2026-01-30", "val": 400.0}],
            ocf=[{"start": "2025-01-01", "end": "2025-12-31",
                  "filed": "2026-01-30", "val": 10.0}],
            capex=[{"start": "2025-01-01", "end": "2025-12-31",
                    "filed": "2026-01-30", "val": -50.0}],  # FCF = -40
            shares=[{"start": None, "end": "2025-12-31",
                     "filed": "2026-01-30", "val": 10.0}],
        )}
        q = quality_snapshot(fund, "BURN", _ts("2026-06-30"), close=50.0)
        assert q is not None
        assert q["p_fcf"] is None

    def test_filed_after_asof_is_invisible(self):
        # Point-in-time discipline: a fact filed after `asof` must not be seen.
        fund = {"LATE": _facts(
            ni=[{"start": "2025-01-01", "end": "2025-12-31",
                 "filed": "2026-06-01", "val": 100.0}],
            eq=[{"start": None, "end": "2025-12-31",
                 "filed": "2026-01-30", "val": 400.0}],
        )}
        q = quality_snapshot(fund, "LATE", _ts("2026-03-01"))
        assert q is not None
        assert q["roe"] is None  # NI not yet public on 2026-03-01


# ---------------------------------------------------------------------------
# propose_trades quality guard
# ---------------------------------------------------------------------------

def _pos(ticker: str) -> dict:
    return {"ticker": ticker, "shares": 0, "avg_cost": 0.0, "stop_price": None}


def _buy_signal(ticker: str) -> dict:
    return {"action": "BUY", "strength": 80,
            "snapshot": {"close": 100.0, "rsi": 50.0, "macd": 0.0,
                         "run_5d": 0.0, "dist_above": 0.0}}


def _valuation() -> dict:
    return {"cash": 10_000.0, "positions_value": 0.0, "total_equity": 10_000.0,
            "allowance_total": 10_000.0, "positions": []}


_BASE_PARAMS = StrategyParams(
    min_cash_pct=5.0, max_position_pct=50.0, max_positions=10,
    stop_type="none", use_atr_stop=False, max_run_5d=0.0, min_run_5d=0.0,
    max_dist_above=0.0,
)
_GUARD_PARAMS = StrategyParams(**{**_BASE_PARAMS.__dict__, "block_negative_roe": True})


class TestBlockNegativeRoe:
    def test_blocks_known_negative_roe(self):
        quality = {"LOSS": {"roe": -0.12, "p_fcf": None}}
        out = propose_trades(
            [], 10_000.0, {"LOSS": 100.0}, {"LOSS": _buy_signal("LOSS")},
            _GUARD_PARAMS, quality_of=lambda t: quality.get(t),
        )
        assert out == []

    def test_blocks_zero_roe(self):
        quality = {"ZERO": {"roe": 0.0, "p_fcf": None}}
        out = propose_trades(
            [], 10_000.0, {"ZERO": 100.0}, {"ZERO": _buy_signal("ZERO")},
            _GUARD_PARAMS, quality_of=lambda t: quality.get(t),
        )
        assert out == []

    def test_missing_fundamentals_stay_neutral(self):
        # GLD-style: quality_of returns None -> must NOT block.
        out = propose_trades(
            [], 10_000.0, {"GLD": 100.0}, {"GLD": _buy_signal("GLD")},
            _GUARD_PARAMS, quality_of=lambda t: None,
        )
        assert len(out) == 1 and out[0]["side"] == "BUY"

    def test_unknown_ticker_stays_neutral(self):
        quality = {"OTHER": {"roe": -1.0, "p_fcf": None}}
        out = propose_trades(
            [], 10_000.0, {"NVDA": 100.0}, {"NVDA": _buy_signal("NVDA")},
            _GUARD_PARAMS, quality_of=lambda t: quality.get(t),
        )
        assert len(out) == 1 and out[0]["side"] == "BUY"

    def test_positive_roe_passes(self):
        quality = {"GOOD": {"roe": 0.25, "p_fcf": 20.0}}
        out = propose_trades(
            [], 10_000.0, {"GOOD": 100.0}, {"GOOD": _buy_signal("GOOD")},
            _GUARD_PARAMS, quality_of=lambda t: quality.get(t),
        )
        assert len(out) == 1 and out[0]["side"] == "BUY"

    def test_guard_off_by_default(self):
        # block_negative_roe defaults False: same negative-ROE map, BUY passes.
        assert _BASE_PARAMS.block_negative_roe is False
        quality = {"LOSS": {"roe": -0.12, "p_fcf": None}}
        out = propose_trades(
            [], 10_000.0, {"LOSS": 100.0}, {"LOSS": _buy_signal("LOSS")},
            _BASE_PARAMS, quality_of=lambda t: quality.get(t),
        )
        assert len(out) == 1 and out[0]["side"] == "BUY"

    def test_relaxed_hold_fallback_also_guarded(self):
        # The relaxed fallback (HOLD with high strength) routes through the
        # same _entry_ok guard — negative-ROE names must be blocked there too.
        quality = {"LOSS": {"roe": -0.5, "p_fcf": None}}
        hold_sig = {**_buy_signal("LOSS"), "action": "HOLD", "strength": 60}
        out = propose_trades(
            [], 10_000.0, {"LOSS": 100.0}, {"LOSS": hold_sig},
            _GUARD_PARAMS, quality_of=lambda t: quality.get(t),
        )
        assert out == []

    def test_relaxed_hold_fallback_passes_when_quality_ok(self):
        quality = {"OK": {"roe": 0.3, "p_fcf": None}}
        hold_sig = {**_buy_signal("OK"), "action": "HOLD", "strength": 60}
        out = propose_trades(
            [], 10_000.0, {"OK": 100.0}, {"OK": hold_sig},
            _GUARD_PARAMS, quality_of=lambda t: quality.get(t),
        )
        assert len(out) == 1 and out[0]["side"] == "BUY"


# ---------------------------------------------------------------------------
# _build_llm_context fundamentals columns
# ---------------------------------------------------------------------------

def _val() -> dict:
    return {"cash": 0, "positions_value": 0, "total_equity": 0,
            "allowance_total": 0, "positions": []}


class TestContextQualityColumns:
    SIGNALS = {
        "GOOD": {"action": "BUY", "strength": 80,
                 "snapshot": {"close": 120.0, "rsi": 55.0, "macd": 0.5}},
        "LOSS": {"action": "HOLD", "strength": 30,
                 "snapshot": {"close": 20.0, "rsi": 40.0, "macd": -0.1}},
        "GLD": {"action": "HOLD", "strength": 10,
                "snapshot": {"close": 90.0, "rsi": 50.0, "macd": 0.0}},
    }
    QUALITY = {
        "GOOD": {"roe": 0.25, "p_fcf": 33.3},
        "LOSS": {"roe": -0.4, "p_fcf": None},
        # GLD absent: no fundamentals (ETF).
    }

    def test_columns_rendered_when_quality_given(self):
        ctx = sim._build_llm_context(_val(), [], self.SIGNALS, quality=self.QUALITY)
        assert "roe%" in ctx and "p/fcf" in ctx
        assert "25.0" in ctx          # GOOD ROE 0.25 -> 25.0%
        assert "33.3" in ctx          # GOOD P/FCF
        assert "-40.0" in ctx         # LOSS ROE -0.4 -> -40.0%

    def test_missing_values_render_as_dash(self):
        ctx = sim._build_llm_context(_val(), [], self.SIGNALS, quality=self.QUALITY)
        # LOSS has no P/FCF, GLD has neither — both must show "-", not 0.
        lines = [ln for ln in ctx.splitlines() if ln.startswith("LOSS") or ln.startswith("GLD")]
        assert len(lines) == 2
        assert all("      -" in ln for ln in lines)
        assert not any("  0.0" in ln.split("      -")[0][-8:] and "p/fcf" in ln
                       for ln in lines)

    def test_table_unchanged_when_quality_none(self):
        ctx_off = sim._build_llm_context(_val(), [], self.SIGNALS)
        assert "roe%" not in ctx_off
        assert "p/fcf" not in ctx_off

    def test_quality_none_omits_columns_entirely(self):
        # Passing quality=None must be byte-identical to the old behaviour.
        ctx_old = sim._build_llm_context(_val(), [], self.SIGNALS)
        ctx_new = sim._build_llm_context(_val(), [], self.SIGNALS, quality=None)
        assert ctx_old == ctx_new
