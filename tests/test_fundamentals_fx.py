"""Fail-closed USD conversion for the fundamentals ingest.

Policy authority: ``app.monthly._fx_convert_usd`` (the factor-price path),
which the audit already verified. The rule it enforces — and which these tests
pin for the statement ingest — is:

    a foreign-currency number is converted only with a genuinely usable rate,
    or it does not exist at all; it is NEVER relabelled USD by assuming a 1.0
    factor.

Before this change ``fundamentals._fx_factor`` returned a fabricated 1.0 in
every missing-rate case (no pair, no prior print, NaN/0 rate), silently
booking a German company's EUR NetIncome / OCF / CAPEX / equity as USD. Now
those facts are skipped: they are simply absent, so the ticker is not eligible
on factors that need them (the fundamentals analog of the price path's NaN).

Cases pinned here:
  * ``_fx_factor`` unit grid — usable prior print (mul + div), no series, a
    period before the first print, and NaN/inf/<=0 prints all return None;
  * ``fetch_rec`` with a KNOWN non-USD suffix (SAP.DE / NESN.SW): exact
    ``value * rate`` (or ``/ rate``) with the canonical capex sign preserved,
    share counts unconverted;
  * missing/empty rate series -> zero converted facts ingested, shares kept,
    the skip logged at INFO;
  * a rate series that starts after an older period end: that period is
    skipped, the newer one converts (per-period, not all-or-nothing);
  * an invalid print stops being a multiplier for periods at/after it;
  * an UNKNOWN suffix is explicitly unresolved (converted facts skipped);
  * a no-suffix USD listing is unchanged (factor 1.0) and ignores stray rates;
  * ``refresh_universe`` end-to-end: no usable rate -> no converted rows in
    the DB and an honest status (never "ok" when nothing at all ingests).

Offline: ``yfinance`` is an in-process stub, ``_fx_rates_from_candles`` is
monkeypatched at the driver level, and the DB is a fresh in-memory SQLite.
"""

from __future__ import annotations

import logging
import sys
import types

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from app import fundamentals
from app.db import Base, Fundamental
from app.fundamentals import CAPEX_TAG, OCF_TAG, _fx_factor

EQUITY_TAG = "StockholdersEquity"
SHARES_TAG = "CommonStockSharesOutstanding"

Q1, Q2 = "2024-03-31", "2024-06-30"
EU_RATE = 1.25          # EURUSD=X: 1 EUR = 1.25 USD
CHF_RATE = 0.80         # USDCHF=X: 1 USD = 0.80 CHF -> factor 1/0.80


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _frame(rows: dict[str, dict[str, float]]) -> pd.DataFrame:
    """yfinance statement shape: line items (index) x period ends (columns)."""
    return pd.DataFrame(rows).T if rows else pd.DataFrame()


def _series(points: list[tuple[str, float]], tz: str | None = None) -> pd.Series:
    """Rate series indexed by date (optionally tz-aware, like stored candles)."""
    idx = pd.DatetimeIndex([pd.Timestamp(d) for d, _v in points])
    if tz is not None:
        idx = pd.DatetimeIndex([pd.Timestamp(d, tz=tz) for d, _v in points])
    return pd.Series([v for _d, v in points], index=idx, dtype=float)


def _install_yf(monkeypatch, *, income=None, cashflow=None, balance=None) -> None:
    """Replace the yfinance module with an in-process stub (no network).

    Only the quarterly statements carry data; the annual ones are empty, which
    keeps the annual/quarterly duplicate guard out of these fixtures.
    """
    empty = pd.DataFrame()
    ticker = types.SimpleNamespace(
        info={"quoteType": "EQUITY"},
        quarterly_income_stmt=income if income is not None else empty,
        quarterly_balance_sheet=balance if balance is not None else empty,
        quarterly_cashflow=cashflow if cashflow is not None else empty,
        income_stmt=empty,
        balance_sheet=empty,
        cash_flow=empty,
    )
    fake = types.ModuleType("yfinance")
    fake.Ticker = lambda symbol: ticker  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "yfinance", fake)


def _one_quarter(capex_raw: float = -200.0):
    """One quarter: NetIncome 1000, OCF 500, capex -200, equity 5000, 100 sh."""
    income = _frame({"Net Income": {Q1: 1000.0}})
    cashflow = _frame({
        "Operating Cash Flow": {Q1: 500.0},
        "Capital Expenditure": {Q1: capex_raw},
    })
    balance = _frame({
        "Stockholders Equity": {Q1: 5000.0},
        "Ordinary Shares Number": {Q1: 100.0},
    })
    return income, cashflow, balance


def _two_quarter_income():
    return _frame({"Net Income": {Q1: 1000.0, Q2: 1100.0}})


def _by_end(facts: list[dict], tag: str) -> dict[str, float]:
    return {f["end"]: f["val"] for f in facts if f["tag"] == tag}


def _converted(facts: list[dict]) -> list[dict]:
    return [f for f in facts if f["currency"] == "USD"]


@pytest.fixture
async def fund_db(monkeypatch):
    """Fresh in-memory SQLite with the app's Session factories redirected.

    Never the production DB: both the module-level binding used by
    ``fundamentals`` and the one in ``app.db`` point at this engine.
    """
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    import app.db as db_mod

    monkeypatch.setattr(db_mod, "Session", session_factory)
    monkeypatch.setattr(fundamentals, "Session", session_factory)
    yield session_factory
    await engine.dispose()


# ---------------------------------------------------------------------------
# 1. _fx_factor: fail closed, no fabricated 1.0
# ---------------------------------------------------------------------------

class TestFxFactorGrid:
    def test_mul_mode_returns_the_prior_print(self):
        rate = _series([("2024-01-02", 1.10), ("2024-02-01", EU_RATE)])
        assert _fx_factor(pd.Timestamp(Q1), rate, "mul") == pytest.approx(EU_RATE)

    def test_div_mode_returns_the_reciprocal(self):
        rate = _series([("2024-01-02", CHF_RATE)])
        assert _fx_factor(pd.Timestamp(Q1), rate, "div") == pytest.approx(1.0 / CHF_RATE)
        assert _fx_factor(pd.Timestamp(Q1), rate, "div") == pytest.approx(1.25)

    def test_missing_series_returns_none(self):
        end = pd.Timestamp(Q1)
        assert _fx_factor(end, None, "mul") is None
        assert _fx_factor(end, pd.Series(dtype=float), "mul") is None

    def test_period_before_the_first_print_returns_none(self):
        rate = _series([("2024-05-01", EU_RATE)])
        # 2024-03-31 predates the first stored rate: no prior print -> no fact.
        assert _fx_factor(pd.Timestamp(Q1), rate, "mul") is None
        assert _fx_factor(pd.Timestamp("2024-05-01"), rate, "mul") == \
            pytest.approx(EU_RATE)

    def test_nat_period_end_returns_none(self):
        rate = _series([("2024-01-02", EU_RATE)])
        assert _fx_factor(pd.NaT, rate, "mul") is None

    @pytest.mark.parametrize("bad", [float("inf"), float("-inf"), 0.0, -1.5])
    def test_invalid_print_at_the_asof_point_returns_none(self, bad):
        rate = _series([("2024-01-02", bad)])
        assert _fx_factor(pd.Timestamp(Q1), rate, "mul") is None
        assert _fx_factor(pd.Timestamp(Q1), rate, "div") is None

    def test_only_nan_prints_returns_none(self):
        # asof skips NaN prints, so with no valid print at all there is no
        # prior rate to use -> fail closed (never the fabricated 1.0).
        rate = _series([("2024-01-02", float("nan")), ("2024-02-01", float("nan"))])
        assert _fx_factor(pd.Timestamp(Q1), rate, "mul") is None

    def test_nan_print_carries_the_prior_valid_rate_forward(self):
        # Same convention as the price path's ffill-then-mask: a missing
        # print is not a rate, the last valid print stays in force.
        rate = _series([("2024-01-02", EU_RATE), ("2024-02-01", float("nan"))])
        assert _fx_factor(pd.Timestamp(Q1), rate, "mul") == pytest.approx(EU_RATE)

    def test_no_lookahead_uses_the_last_print_at_or_before_the_end(self):
        rate = _series([("2024-01-02", 1.10), ("2024-06-01", 9.90)])
        # A future print must never be backfilled into an earlier period.
        assert _fx_factor(pd.Timestamp(Q1), rate, "mul") == pytest.approx(1.10)

    def test_end_equal_to_the_print_date_uses_that_print(self):
        rate = _series([(Q1, EU_RATE)])
        assert _fx_factor(pd.Timestamp(Q1), rate, "mul") == pytest.approx(EU_RATE)

    def test_tz_aware_index_is_compared_as_naive_utc(self):
        rate = _series([("2024-01-02", EU_RATE)], tz="UTC")
        assert _fx_factor(pd.Timestamp(Q1), rate, "mul") == pytest.approx(EU_RATE)
        # ... and a period end carrying tz info is localized before comparing.
        assert _fx_factor(pd.Timestamp(Q1, tz="UTC"), rate, "mul") == \
            pytest.approx(EU_RATE)
        assert _fx_factor(pd.Timestamp("2023-12-31", tz="UTC"), rate, "mul") is None


# ---------------------------------------------------------------------------
# 2. fetch_rec with a known suffix and a usable rate: exact conversion
# ---------------------------------------------------------------------------

class TestKnownSuffixConverts:
    @pytest.mark.parametrize("capex_raw", [-200.0, 200.0])
    def test_converted_facts_are_value_times_rate(self, monkeypatch, capex_raw):
        income, cashflow, balance = _one_quarter(capex_raw=capex_raw)
        _install_yf(monkeypatch, income=income, cashflow=cashflow, balance=balance)
        rate = _series([("2024-01-02", EU_RATE)])

        facts, note = fundamentals.fetch_rec("SAP.DE", rate, "mul")
        assert note is None

        assert _by_end(facts, "NetIncomeLoss")[Q1] == pytest.approx(1000.0 * EU_RATE)
        assert _by_end(facts, OCF_TAG)[Q1] == pytest.approx(500.0 * EU_RATE)
        assert _by_end(facts, EQUITY_TAG)[Q1] == pytest.approx(5000.0 * EU_RATE)
        # canonical capex sign survives the conversion: always an outflow
        capex = _by_end(facts, CAPEX_TAG)[Q1]
        assert capex == pytest.approx(-200.0 * EU_RATE)
        assert capex < 0

        # share counts carry no currency and are never multiplied
        shares = _by_end(facts, SHARES_TAG)[Q1]
        assert shares == 100.0
        share_facts = [f for f in facts if f["tag"] == SHARES_TAG]
        assert [f["currency"] for f in share_facts] == ["shares"]
        assert all(f["currency"] == "USD" for f in _converted(facts))

    def test_div_suffix_divides_by_the_rate(self, monkeypatch):
        income, cashflow, balance = _one_quarter()
        _install_yf(monkeypatch, income=income, cashflow=cashflow, balance=balance)
        rate = _series([("2024-01-02", CHF_RATE)])

        facts, note = fundamentals.fetch_rec("NESN.SW", rate, "div")
        assert note is None
        assert _by_end(facts, "NetIncomeLoss")[Q1] == pytest.approx(1000.0 / CHF_RATE)
        assert _by_end(facts, "NetIncomeLoss")[Q1] == pytest.approx(1250.0)
        assert _by_end(facts, OCF_TAG)[Q1] == pytest.approx(500.0 / CHF_RATE)
        assert _by_end(facts, CAPEX_TAG)[Q1] == pytest.approx(-200.0 / CHF_RATE)


# ---------------------------------------------------------------------------
# 3. fetch_rec fail-closed: no usable rate -> no converted fact
# ---------------------------------------------------------------------------

class TestFailClosed:
    @pytest.mark.parametrize("rate", [None, pd.Series(dtype=float)])
    def test_no_rate_skips_every_converted_fact(self, monkeypatch, caplog, rate):
        income, cashflow, balance = _one_quarter()
        _install_yf(monkeypatch, income=income, cashflow=cashflow, balance=balance)
        caplog.set_level(logging.INFO, logger="app.fundamentals")

        facts, note = fundamentals.fetch_rec("SAP.DE", rate, "mul")
        # the (facts, note) contract is unchanged: no note for FX skips
        assert note is None

        # ZERO converted facts: no EUR number was relabelled USD.
        assert _converted(facts) == []
        assert {f["tag"] for f in facts} == {SHARES_TAG}
        assert {f["currency"] for f in facts} == {"shares"}
        assert not any(f["val"] == 1000.0 for f in facts)  # the EUR NetIncome
        assert not any(f["val"] == 5000.0 for f in facts)  # the EUR equity

        messages = [r.getMessage() for r in caplog.records
                    if r.name == "app.fundamentals" and r.levelno == logging.INFO]
        skipped = [m for m in messages if "SAP.DE" in m and "skipped 4" in m]
        assert skipped, f"expected an INFO skip log, got {messages!r}"

    def test_no_shares_means_no_facts_at_all(self, monkeypatch):
        # Income only: with no FX every fact is skipped, so the ticker yields
        # nothing rather than a USD-labelled EUR number.
        _install_yf(monkeypatch, income=_one_quarter()[0])
        facts, note = fundamentals.fetch_rec("SAP.DE", None, "mul")
        assert facts == []
        assert note is None

    def test_rate_starting_after_the_period_end_is_per_period(self, monkeypatch):
        _install_yf(monkeypatch, income=_two_quarter_income())
        # First print on 2024-05-01: Q1 (ends 03-31) has no prior rate, Q2 does.
        rate = _series([("2024-05-01", EU_RATE)])

        facts, note = fundamentals.fetch_rec("SAP.DE", rate, "mul")
        assert note is None
        ni = _by_end(facts, "NetIncomeLoss")
        assert Q1 not in ni
        assert ni[Q2] == pytest.approx(1100.0 * EU_RATE)

    @pytest.mark.parametrize("bad", [0.0, -1.5, float("inf"), float("-inf")])
    def test_invalid_rate_never_becomes_a_multiplier(self, monkeypatch, bad):
        """A print before Q1 is valid; the bad print lands between Q1 and Q2.
        Q1 converts with the valid rate, Q2 fails closed — the invalid print
        is not a multiplier for periods at/after it (and is not backfilled
        into the period before it either)."""
        _install_yf(monkeypatch, income=_two_quarter_income())
        rate = _series([("2024-01-02", 1.10), ("2024-04-01", bad)])

        facts, note = fundamentals.fetch_rec("SAP.DE", rate, "mul")
        assert note is None
        ni = _by_end(facts, "NetIncomeLoss")
        assert ni[Q1] == pytest.approx(1000.0 * 1.10)
        assert Q2 not in ni

    def test_nan_print_carries_the_prior_valid_rate_forward(self, monkeypatch):
        """A NaN print is a missing observation, not a rate — the same
        convention as the price path's ffill-then-mask: the last valid print
        remains the prevailing rate. (An inf/<=0 print is different: see
        above, it fails closed.)"""
        _install_yf(monkeypatch, income=_two_quarter_income())
        rate = _series([("2024-01-02", 1.10), ("2024-04-01", float("nan"))])

        facts, note = fundamentals.fetch_rec("SAP.DE", rate, "mul")
        assert note is None
        ni = _by_end(facts, "NetIncomeLoss")
        assert ni[Q1] == pytest.approx(1000.0 * 1.10)
        assert ni[Q2] == pytest.approx(1100.0 * 1.10)


# ---------------------------------------------------------------------------
# 4. fetch_rec with an unknown suffix: currency explicitly unresolved
# ---------------------------------------------------------------------------

class TestUnknownSuffix:
    def test_unknown_suffix_converted_facts_are_skipped(self, monkeypatch, caplog):
        """A dot-suffix not in SUFFIX_FX is NOT known USD: its currency is
        unresolved, so its converted facts fail closed exactly like a known
        suffix whose pair is missing (old behaviour: 1.0 "assume USD")."""
        assert fundamentals._suffix_fx("XYZ.MI") is None
        income, cashflow, balance = _one_quarter()
        _install_yf(monkeypatch, income=income, cashflow=cashflow, balance=balance)
        caplog.set_level(logging.INFO, logger="app.fundamentals")

        facts, note = fundamentals.fetch_rec("XYZ.MI")
        assert note is None
        assert _converted(facts) == []
        assert {f["tag"] for f in facts} == {SHARES_TAG}
        assert [f["val"] for f in facts] == [100.0]

        messages = [r.getMessage() for r in caplog.records if r.name == "app.fundamentals"]
        assert any("XYZ.MI" in m and "skipped 4" in m for m in messages), messages

    def test_unknown_suffix_is_unresolved_even_if_a_rate_is_supplied(self, monkeypatch):
        # No pair exists for an unknown suffix, so a rate argument cannot
        # certify the currency: the driver never supplies one, and one passed
        # by mistake must not silently convert a possibly-USD statement.
        assert "." in "XYZ.MI" and fundamentals._suffix_fx("XYZ.MI") is None
        _install_yf(monkeypatch, income=_one_quarter()[0])
        facts, _ = fundamentals.fetch_rec("XYZ.MI", _series([("2024-01-02", EU_RATE)]))
        assert facts == []


# ---------------------------------------------------------------------------
# 5. No suffix: USD-native, unchanged
# ---------------------------------------------------------------------------

class TestUsdNative:
    def test_no_suffix_ingests_at_factor_one(self, monkeypatch, caplog):
        income, cashflow, balance = _one_quarter()
        _install_yf(monkeypatch, income=income, cashflow=cashflow, balance=balance)
        caplog.set_level(logging.INFO, logger="app.fundamentals")

        facts, note = fundamentals.fetch_rec("AAPL")
        assert note is None
        assert _by_end(facts, "NetIncomeLoss")[Q1] == 1000.0
        assert _by_end(facts, OCF_TAG)[Q1] == 500.0
        assert _by_end(facts, CAPEX_TAG)[Q1] == -200.0
        assert _by_end(facts, EQUITY_TAG)[Q1] == 5000.0
        assert _by_end(facts, SHARES_TAG)[Q1] == 100.0
        assert not any("skipped" in r.getMessage() for r in caplog.records)

    def test_no_suffix_ignores_a_stray_rate(self, monkeypatch):
        # USD listings never enter the conversion branch: the residual 1.0
        # fallback is only reachable here, where 1.0 is the true rate.
        _install_yf(monkeypatch, income=_one_quarter()[0])
        facts, _ = fundamentals.fetch_rec("AAPL", _series([("2024-01-02", 2.0)]))
        assert _by_end(facts, "NetIncomeLoss")[Q1] == 1000.0


# ---------------------------------------------------------------------------
# 6. refresh_universe end-to-end (real fetch_rec + real DB writes)
# ---------------------------------------------------------------------------

async def _empty_rates(pairs):  # noqa: ANN001 - matches _fx_rates_from_candles
    """The candle cache holds no FX prints for the requested pairs."""
    return {p: pd.Series(dtype=float) for p in pairs}


def _usd_rates(rate: float):
    async def _rates(pairs):  # noqa: ANN001 - matches _fx_rates_from_candles
        return {p: _series([("2024-01-02", rate)]) for p in pairs}

    return _rates


async def _rows(fund_db, ticker: str) -> list[Fundamental]:
    async with fund_db() as s:
        return list((await s.scalars(
            select(Fundamental).where(Fundamental.ticker == ticker))).all())


class TestRefreshUniverseDriver:
    async def test_known_suffix_without_rate_stores_no_converted_facts(
            self, fund_db, monkeypatch):
        income, cashflow, balance = _one_quarter()
        _install_yf(monkeypatch, income=income, cashflow=cashflow, balance=balance)
        monkeypatch.setattr(fundamentals, "_fx_rates_from_candles", _empty_rates)

        statuses = await fundamentals.refresh_universe(tickers=["SAP.DE", "AAPL"])
        # SAP.DE still has its share count -> the fetch itself succeeded.
        assert statuses == {"SAP.DE": "ok", "AAPL": "ok"}

        sap = await _rows(fund_db, "SAP.DE")
        assert {r.tag for r in sap} == {SHARES_TAG}
        assert all(r.currency == "shares" for r in sap)
        assert all(r.val == 100.0 for r in sap)
        # nothing was booked as USD-by-assumption from the EUR statements
        assert not any(r.currency == "USD" for r in sap)
        assert not any(r.val in (1000.0, 500.0, -200.0, 5000.0) for r in sap)

        # AAPL is USD-native and unaffected by the missing FX pair
        aapl = await _rows(fund_db, "AAPL")
        assert len(aapl) == 5
        assert all(r.currency in ("USD", "shares") for r in aapl)
        assert [r.val for r in aapl if r.tag == "NetIncomeLoss"] == [1000.0]

    async def test_known_suffix_without_rate_and_no_shares_is_failed(
            self, fund_db, monkeypatch):
        # Every fact of this stub is a converted one: with no FX nothing at all
        # can be ingested, and the status must say so instead of "ok".
        _install_yf(monkeypatch, income=_one_quarter()[0])
        monkeypatch.setattr(fundamentals, "_fx_rates_from_candles", _empty_rates)

        statuses = await fundamentals.refresh_universe(tickers=["SAP.DE"])
        assert statuses["SAP.DE"] != "ok"
        assert statuses["SAP.DE"] == "failed"
        assert await _rows(fund_db, "SAP.DE") == []

    async def test_known_suffix_with_rate_converts_and_stores(
            self, fund_db, monkeypatch):
        income, cashflow, balance = _one_quarter()
        _install_yf(monkeypatch, income=income, cashflow=cashflow, balance=balance)
        monkeypatch.setattr(fundamentals, "_fx_rates_from_candles", _usd_rates(EU_RATE))

        statuses = await fundamentals.refresh_universe(tickers=["SAP.DE"])
        assert statuses == {"SAP.DE": "ok"}

        sap = await _rows(fund_db, "SAP.DE")
        by_tag = {r.tag: r.val for r in sap}
        assert by_tag["NetIncomeLoss"] == pytest.approx(1000.0 * EU_RATE)
        assert by_tag[OCF_TAG] == pytest.approx(500.0 * EU_RATE)
        assert by_tag[CAPEX_TAG] == pytest.approx(-200.0 * EU_RATE)
        assert by_tag[EQUITY_TAG] == pytest.approx(5000.0 * EU_RATE)
        assert by_tag[SHARES_TAG] == 100.0
        assert all(np.isfinite(r.val) for r in sap)
