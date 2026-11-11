"""Instrument quote metadata (``app/instruments.py``) — the one suffix owner.

Pinned here:

* the suffix->FX map was MOVED, not copied: ``instruments.suffix_fx`` and
  ``fundamentals._suffix_fx`` agree on every entry and on unknown / no-suffix /
  dash-form tickers, and ``from app.fundamentals import SUFFIX_FX`` is the same
  live mapping (``app/savings.py`` imports it there);
* quote units: ``.L`` closes are PENCE (factor 0.01), everything else 1.0 —
  and the price path applies the unit BEFORE the FX rate, so 1250p at
  GBPUSD 1.25 is $15.625, not the 100x-inflated $1562.50;
* USD-native listings: no suffix, or an explicit US share-class suffix
  (``.A``/``.B``/``.C``). ``BRK.B`` therefore ingests converted facts at factor
  1.0 again — now as KNOWN USD, not by accident — while unknown foreign
  suffixes (``NOVO-B.CO``) keep failing closed;
* invariance: per-listing quote-unit rescaling cannot move the equal-weight
  market index (it consumes returns), and a frame without ``.L`` names goes
  through numerically unchanged arithmetic.

Offline: ``yfinance`` is an in-process stub (the ``sys.modules`` pattern from
``tests/test_fundamentals_fx.py``) — no provider/network call, no DB.
"""

from __future__ import annotations

import logging
import sys
import types

import numpy as np
import pandas as pd
import pytest

from app import fundamentals, instruments, monthly
from app.market_index import equal_weight_return_index

Q1 = "2024-03-31"
EU_RATE = 1.25  # EURUSD=X: 1 EUR = 1.25 USD
CHF_RATE = 0.80  # USDCHF=X: 1 USD = 0.80 CHF
EQUITY_TAG = "StockholdersEquity"
SHARES_TAG = "CommonStockSharesOutstanding"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _frame(rows: dict[str, dict[str, float]]) -> pd.DataFrame:
    """yfinance statement shape: line items (index) x period ends (columns)."""
    return pd.DataFrame(rows).T if rows else pd.DataFrame()


def _rate_series(points: list[tuple[str, float]]) -> pd.Series:
    idx = pd.DatetimeIndex([pd.Timestamp(d) for d, _v in points])
    return pd.Series([v for _d, v in points], index=idx, dtype=float)


def _install_yf(monkeypatch) -> None:
    """Replace yfinance with an in-process stub carrying one quarter of facts.

    Same pattern as ``tests/test_fundamentals_fx.py``: only the quarterly
    statements carry data, so the annual/quarterly duplicate guard stays out
    of the fixture. No network is reachable through the stub.
    """
    income = _frame({"Net Income": {Q1: 1000.0}})
    cashflow = _frame({
        "Operating Cash Flow": {Q1: 500.0},
        "Capital Expenditure": {Q1: -200.0},
    })
    balance = _frame({
        "Stockholders Equity": {Q1: 5000.0},
        "Ordinary Shares Number": {Q1: 100.0},
    })
    empty = pd.DataFrame()
    ticker = types.SimpleNamespace(
        info={"quoteType": "EQUITY"},
        quarterly_income_stmt=income,
        quarterly_balance_sheet=balance,
        quarterly_cashflow=cashflow,
        income_stmt=empty,
        balance_sheet=empty,
        cash_flow=empty,
    )
    fake = types.ModuleType("yfinance")
    fake.Ticker = lambda symbol: ticker  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "yfinance", fake)


def _by_end(facts: list[dict], tag: str) -> dict[str, float]:
    return {f["end"]: f["val"] for f in facts if f["tag"] == tag}


def _converted(facts: list[dict]) -> list[dict]:
    return [f for f in facts if f["currency"] == "USD"]


def _price_frame(values: dict[str, list[float]], days) -> pd.DataFrame:
    return pd.DataFrame(values, index=pd.DatetimeIndex(days))


# ---------------------------------------------------------------------------
# 1. The suffix->FX map: moved, not copied
# ---------------------------------------------------------------------------

class TestSuffixFxMap:
    def test_fundamentals_reexport_is_the_same_live_mapping(self):
        from app.fundamentals import SUFFIX_FX  # savings.py's import path

        assert SUFFIX_FX is instruments.SUFFIX_FX
        assert fundamentals.SUFFIX_FX is instruments.SUFFIX_FX

    @pytest.mark.parametrize("suffix", sorted(instruments.SUFFIX_FX))
    def test_every_entry_agrees_through_both_lookups(self, suffix):
        ticker = f"XYZ{suffix}"
        assert instruments.suffix_fx(ticker) == instruments.SUFFIX_FX[suffix]
        assert fundamentals._suffix_fx(ticker) == instruments.SUFFIX_FX[suffix]

    @pytest.mark.parametrize(
        "ticker",
        ["AAPL", "BRK-B", "BRK.B", "XYZ.MI", "NOVO-B.CO", "ABC.VI", "SAP.DE",
         "NESN.SW", "SHEL.L", "XYZ", ""],
    )
    def test_edge_tickers_agree(self, ticker):
        assert fundamentals._suffix_fx(ticker) == instruments.suffix_fx(ticker)

    def test_no_suffix_and_unknown_suffix_are_none(self):
        assert instruments.suffix_fx("AAPL") is None
        assert instruments.suffix_fx("BRK-B") is None
        assert instruments.suffix_fx("BRK.B") is None  # share class, not FX
        assert instruments.suffix_fx("XYZ.MI") is None
        assert instruments.suffix_fx("NOVO-B.CO") is None
        assert instruments.suffix_fx("") is None

    def test_london_maps_to_gbpusd_mul(self):
        # The pence quoted below needs exactly this pair to become USD.
        assert instruments.SUFFIX_FX[".L"] == ("GBPUSD=X", "mul")

    def test_map_is_identity_stable_between_importers(self):
        # A copy would drift silently; identity proves one owner.
        assert fundamentals._suffix_fx.__module__ == "app.fundamentals"
        assert instruments.SUFFIX_FX is fundamentals.SUFFIX_FX


# ---------------------------------------------------------------------------
# 2. Quote units: explicit pence metadata, 1.0 everywhere else
# ---------------------------------------------------------------------------

class TestQuoteUnits:
    @pytest.mark.parametrize("ticker", ["SHEL.L", "VOD.L", "XYZ.L"])
    def test_london_closes_are_pence(self, ticker):
        assert instruments.quote_unit_factor(ticker) == 0.01

    @pytest.mark.parametrize(
        "ticker",
        ["SAP.DE", "ASML.AS", "NESN.SW", "VOLV-B.ST", "AAPL", "BRK-B", "BRK.B",
         "XYZ.MI", "NOVO-B.CO", "ABC.VI", ""],
    )
    def test_everything_else_is_one(self, ticker):
        assert instruments.quote_unit_factor(ticker) == 1.0

    def test_only_london_has_a_non_unit_quote(self):
        assert instruments.QUOTE_UNIT_FACTORS == {".L": 0.01}

    def test_quote_units_are_a_subset_of_the_fx_map(self):
        # A unit can only scale a listing whose currency conversion is known;
        # it must never be attached to a pass-through (USD/unknown) suffix.
        assert set(instruments.QUOTE_UNIT_FACTORS) <= set(instruments.SUFFIX_FX)


# ---------------------------------------------------------------------------
# 3. USD-native notation: no suffix, or a US share class
# ---------------------------------------------------------------------------

class TestUsdNative:
    @pytest.mark.parametrize("ticker", ["BRK.B", "BRK.A", "BRK.C", "AAPL", "BRK-B"])
    def test_usd_listings_are_native(self, ticker):
        assert instruments.is_usd_native(ticker) is True

    @pytest.mark.parametrize(
        "ticker",
        ["SAP.DE", "XYZ.MI", "NOVO-B.CO", "NESN.SW", "SHEL.L", "VOLV-B.ST"],
    )
    def test_foreign_and_unknown_suffixes_are_not_usd(self, ticker):
        assert instruments.is_usd_native(ticker) is False

    def test_trailing_dot_is_not_usd(self):
        # A degenerate suffix is unrecognised, never a currency code.
        assert instruments.is_usd_native("ABC.") is False
        assert instruments.suffix_fx("ABC.") is None

    def test_share_class_notations(self):
        assert instruments.USD_NATIVE_DOT_SUFFIXES == frozenset({".A", ".B", ".C"})

    def test_share_classes_are_not_currency_codes(self):
        # Disjointness is the honesty condition: a suffix cannot both denote a
        # US share class and demand an FX pair.
        assert instruments.USD_NATIVE_DOT_SUFFIXES.isdisjoint(instruments.SUFFIX_FX)


# ---------------------------------------------------------------------------
# 4. Price path: unit first, then FX, fail closed without a rate
# ---------------------------------------------------------------------------

class TestPricePath:
    def test_l_pence_close_is_scaled_then_converted(self):
        days = pd.bdate_range("2026-01-05", periods=2)
        raw = _price_frame(
            {"GBPUSD=X": [1.25, 1.25], "SHEL.L": [1250.0, 1300.0]}, days)

        out = monthly._fx_convert_usd(raw)

        # 1250p = £12.50 -> $15.625 at 1.25 (and NOT $1562.50)
        assert out["SHEL.L"].iloc[0] == 15.625
        assert out["SHEL.L"].iloc[1] == pytest.approx(1300.0 * 0.01 * 1.25)
        assert "GBPUSD=X" not in out.columns

    def test_l_without_the_pair_column_fails_closed(self):
        days = pd.bdate_range("2026-01-05", periods=2)
        raw = _price_frame({"SHEL.L": [1250.0, 1300.0]}, days)

        out = monthly._fx_convert_usd(raw)

        assert out["SHEL.L"].isna().all()

    def test_l_with_no_rate_prior_to_a_date_fails_closed_there(self):
        d0, d1 = pd.bdate_range("2026-01-05", periods=2)
        raw = _price_frame(
            {"GBPUSD=X": [np.nan, 1.25], "SHEL.L": [1250.0, 1260.0]}, [d0, d1])

        out = monthly._fx_convert_usd(raw)

        assert pd.isna(out["SHEL.L"].iloc[0])
        assert out["SHEL.L"].iloc[1] == pytest.approx(1260.0 * 0.01 * 1.25)

    def test_known_suffix_without_a_quote_unit_converts_exactly_as_before(self):
        days = pd.bdate_range("2026-01-05", periods=2)
        raw = _price_frame(
            {"EURUSD=X": [1.25, 1.50], "SAP.DE": [100.0, 110.0]}, days)

        out = monthly._fx_convert_usd(raw)

        # unit 1.0: literally the previous `close * rate` arithmetic
        assert out["SAP.DE"].tolist() == [125.0, 165.0]
        assert "EURUSD=X" not in out.columns

    def test_usd_and_unknown_suffix_columns_are_bit_identical(self):
        days = pd.bdate_range("2026-01-05", periods=3)
        raw = _price_frame({
            "EURUSD=X": [1.25, np.nan, 1.10],
            "AAPL": [10.0, 11.0, np.nan],
            "XYZ.MI": [7.0, np.nan, 9.0],
        }, days)

        out = monthly._fx_convert_usd(raw)

        # raw pass-through: their currency is theirs (USD) or unresolved
        assert out["AAPL"].equals(raw["AAPL"])
        assert out["XYZ.MI"].equals(raw["XYZ.MI"])
        np.testing.assert_array_equal(
            out["AAPL"].to_numpy(), np.array([10.0, 11.0, np.nan]))

    def test_div_mode_applies_the_quote_unit_before_dividing(self, monkeypatch):
        days = pd.bdate_range("2026-01-05", periods=1)
        raw = _price_frame({"USDCHF=X": [CHF_RATE], "NESN.SW": [100.0]}, days)

        # real .SW metadata: unit 1.0 -> the unchanged division
        assert monthly._fx_convert_usd(raw)["NESN.SW"].iloc[0] == pytest.approx(125.0)

        # inject a non-unit quote for the div-mode venue: the unit must scale
        # the price BEFORE the rate divides it (100 * 0.5 CHF / 0.80).
        monkeypatch.setitem(instruments.QUOTE_UNIT_FACTORS, ".SW", 0.5)
        converted = monthly._fx_convert_usd(raw)["NESN.SW"].iloc[0]
        assert converted == pytest.approx(50.0 / CHF_RATE)
        assert converted == pytest.approx(62.5)


# ---------------------------------------------------------------------------
# 5. Statement ingest policy through the real fetch_rec
# ---------------------------------------------------------------------------

class TestFundamentalsPolicy:
    @pytest.mark.parametrize("symbol", ["BRK.B", "BRK.C"])
    def test_us_share_class_ingests_converted_facts_at_factor_one(
            self, monkeypatch, symbol):
        """BRK.B is a USD listing: its dot is a share class, not a currency.

        Before the instruments owner the unknown-dot-suffix rule skipped every
        converted fact for it (currency "unresolved"). Now it is KNOWN USD and
        its facts ingest unchanged, exactly like a no-suffix listing.
        """
        _install_yf(monkeypatch)
        assert instruments.is_usd_native(symbol) is True
        assert instruments.suffix_fx(symbol) is None  # no FX pair to convert

        facts, note = fundamentals.fetch_rec(symbol)
        aapl, _ = fundamentals.fetch_rec("AAPL")

        assert note is None
        assert facts == aapl  # values unchanged: factor 1.0 both
        assert _by_end(facts, "NetIncomeLoss")[Q1] == 1000.0
        assert _by_end(facts, EQUITY_TAG)[Q1] == 5000.0
        assert _by_end(facts, SHARES_TAG)[Q1] == 100.0
        assert {f["currency"] for f in facts} == {"USD", "shares"}

    def test_unknown_foreign_suffix_still_skips_converted_facts(
            self, monkeypatch, caplog):
        """NOVO-B.CO has an unknown suffix and is NOT USD: fail closed."""
        _install_yf(monkeypatch)
        caplog.set_level(logging.INFO, logger="app.fundamentals")
        assert instruments.is_usd_native("NOVO-B.CO") is False
        assert instruments.suffix_fx("NOVO-B.CO") is None

        facts, note = fundamentals.fetch_rec("NOVO-B.CO")

        assert note is None
        assert _converted(facts) == []
        assert {f["tag"] for f in facts} == {SHARES_TAG}
        assert [f["val"] for f in facts] == [100.0]
        messages = [r.getMessage() for r in caplog.records
                    if r.name == "app.fundamentals"]
        assert any("NOVO-B.CO" in m and "skipped 4" in m for m in messages), messages

    def test_known_suffix_still_converts_at_the_pair_rate(self, monkeypatch):
        _install_yf(monkeypatch)

        facts, note = fundamentals.fetch_rec(
            "SAP.DE", _rate_series([("2024-01-02", EU_RATE)]), "mul")

        assert note is None
        assert _by_end(facts, "NetIncomeLoss")[Q1] == pytest.approx(1000.0 * EU_RATE)
        assert _by_end(facts, "NetCashProvidedByUsedInOperatingActivities")[Q1] == \
            pytest.approx(500.0 * EU_RATE)
        assert _by_end(facts, "PaymentsToAcquirePropertyPlantAndEquipment")[Q1] == \
            pytest.approx(-200.0 * EU_RATE)
        assert _by_end(facts, SHARES_TAG)[Q1] == 100.0

    def test_share_class_listing_is_not_treated_as_its_currency_code(
            self, monkeypatch, caplog):
        """A stray rate must not touch a USD-native share-class listing: the
        conversion branch is never entered (asserted through the skip log)."""
        _install_yf(monkeypatch)
        caplog.set_level(logging.INFO, logger="app.fundamentals")

        facts, _ = fundamentals.fetch_rec(
            "BRK.B", _rate_series([("2024-01-02", 2.0)]), "mul")

        assert _by_end(facts, "NetIncomeLoss")[Q1] == 1000.0
        assert not any("skipped" in r.getMessage() for r in caplog.records)


# ---------------------------------------------------------------------------
# 6. Invariance: the market index and .L-free frames cannot see the change
# ---------------------------------------------------------------------------

class TestInvariance:
    def test_equal_weight_index_is_invariant_to_a_quote_unit_rescale(self):
        """The gate index consumes RETURNS, so quoting one member in pence
        (x100) instead of pounds cannot move it — the economic reason the
        price-path fix is safe for every existing (L-free) universe."""
        days = pd.bdate_range("2026-01-05", periods=3)
        base = _price_frame({"AAA": [100.0, 105.0, 100.0],
                             "BBB": [10.0, 5.0, 6.0]}, days)
        pence = base.copy()
        pence["BBB"] = base["BBB"] * 100.0

        np.testing.assert_array_equal(
            equal_weight_return_index(base).to_numpy(),
            equal_weight_return_index(pence).to_numpy(),
        )

    def test_price_path_without_l_names_keeps_the_previous_arithmetic(self):
        """A frame with no .L names must not see the change at all: the output
        is bit-identical to the documented pre-change arithmetic, and the gate
        index built from it is unchanged."""
        days = pd.bdate_range("2026-01-05", periods=4)
        raw = _price_frame({
            "EURUSD=X": [1.10, np.nan, 1.25, np.nan],
            "USDCHF=X": [CHF_RATE, CHF_RATE, np.nan, 0.90],
            "SAP.DE": [100.0, np.nan, 120.0, 130.0],
            "NESN.SW": [50.0, 55.0, 60.0, np.nan],
            "AAPL": [10.0, 11.0, 12.0, 13.0],
        }, days)
        expected = raw.copy()
        for ticker, pair, mode in (("SAP.DE", "EURUSD=X", "mul"),
                                   ("NESN.SW", "USDCHF=X", "div")):
            rate = expected[pair].ffill()
            rate = rate.where(np.isfinite(rate) & (rate > 0))
            expected[ticker] = (expected[ticker] * rate if mode == "mul"
                                else expected[ticker] / rate)
        expected = expected.drop(columns=["EURUSD=X", "USDCHF=X"])

        out = monthly._fx_convert_usd(raw)

        pd.testing.assert_frame_equal(out, expected[out.columns])
        np.testing.assert_array_equal(
            equal_weight_return_index(out).to_numpy(),
            equal_weight_return_index(expected[out.columns]).to_numpy(),
        )
