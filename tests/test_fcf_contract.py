"""Contract tests for the canonical free-cash-flow sign (audit P0-A).

The strategy math in ``app/monthly.py`` computes ``FCF = OCF + capex`` and
assumes capex is a NEGATIVE cash outflow. The raw providers disagree:

* SEC ``us-gaap:PaymentsToAcquirePropertyPlantAndEquipment`` reports the
  expenditure as a POSITIVE number (Apple FY2025: +12,715,000,000).
* yfinance's "Capital Expenditure" cash-flow row is already a NEGATIVE outflow.

Before the fix, an SEC-sourced ticker therefore had capex ADDED to operating
cash flow: Apple's true FCF of 98,767,000,000 was scored as 124,197,000,000.

Canonical contract exercised here (see ``app/fundamentals.py``):

    capex is a NEGATIVE cash outflow in every source and at every stage —
    when parsed, when stored and when read back.

The tests cover: the sign rule itself (SEC positive, SEC already negative,
Yahoo negative, Yahoo positive), its idempotence, legacy rows that were
stored with the raw positive SEC number before the fix, and the real
ingest -> store -> load -> ``quality_snapshot`` pipeline on a genuine SEC
Apple FY2025 payload.

The second half of the file covers the non-finite hardening: NaN and +/-inf
are not usable values (SQLite stores NaN as NULL, which breaks the NOT NULL
constraint on ``fundamentals.val`` and rolls back the whole ticker's insert
batch). A non-finite fact must be skipped on its own - never coerced to 0,
never allowed to abort its valid siblings - on the way in (SEC parse, Yahoo
fetch), on the way into the DB and on the way out (the loader).

No network access: the EDGAR fetch is monkeypatched and yfinance is replaced
by an in-process fake.
"""

from __future__ import annotations

import sys
import types

import pandas as pd
import pytest
from sqlalchemy import StaticPool, select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app import edgar, fundamentals
from app.db import Base, Fundamental, SecCik
from app.fundamentals import CAPEX_TAG, OCF_TAG, canonical_capex, finite_float
from app.monthly import quality_snapshot

# ---------------------------------------------------------------------------
# Genuine SEC figures — Apple FY2025 (period 2024-09-29 .. 2025-09-27,
# filed 2025-10-31), from the SEC companyfacts API:
#   https://data.sec.gov/api/xbrl/companyfacts/CIK0000320193.json
# The audit verified these numbers against the filing itself.
# ---------------------------------------------------------------------------
APPLE_OCF = 111_482_000_000            # NetCashProvidedByUsedInOperatingActivities
APPLE_CAPEX_RAW = 12_715_000_000      # PaymentsToAcquirePropertyPlantAndEquipment (SEC: positive)
APPLE_FCF = 98_767_000_000            # OCF - capex, the economically correct FCF
APPLE_FCF_BUGGY = 124_197_000_000     # what the old OCF + positive-capex math produced
APPLE_START, APPLE_END, APPLE_FILED = "2024-09-29", "2025-09-27", "2025-10-31"

# Illustrative (NOT asserted as a genuine filing figure) market data used to
# turn FCF into a P/FCF ratio through the real pipeline.
ILLUSTRATIVE_PRICE = 250.0
ILLUSTRATIVE_SHARES = 15_000_000_000.0


def _ts(s: str) -> pd.Timestamp:
    return pd.Timestamp(s)  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
async def mem_db(monkeypatch):
    """In-memory SQLite with the app's Session factories pointed at it."""
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
    monkeypatch.setattr(edgar, "Session", session_factory)

    yield session_factory
    await engine.dispose()


async def _no_sleep(_):
    return None


def _apple_companyfacts() -> dict:
    """A companyfacts payload carrying Apple's genuine FY2025 cash-flow
    numbers, with capex in SEC's raw (positive) convention."""
    return {
        "facts": {
            "us-gaap": {
                "StockholdersEquity": {"units": {"USD": [
                    {"start": None, "end": APPLE_END, "filed": APPLE_FILED, "val": 56_950_000_000},
                ]}},
                "NetIncomeLoss": {"units": {"USD": [
                    {"start": APPLE_START, "end": APPLE_END, "filed": APPLE_FILED,
                     "val": 112_010_000_000},
                ]}},
                "NetCashProvidedByUsedInOperatingActivities": {"units": {"USD": [
                    {"start": APPLE_START, "end": APPLE_END, "filed": APPLE_FILED,
                     "val": APPLE_OCF},
                ]}},
                # SEC reports the expenditure as a POSITIVE number
                "PaymentsToAcquirePropertyPlantAndEquipment": {"units": {"USD": [
                    {"start": APPLE_START, "end": APPLE_END, "filed": APPLE_FILED,
                     "val": APPLE_CAPEX_RAW},
                ]}},
                "CommonStockSharesOutstanding": {"units": {"shares": [
                    {"start": None, "end": APPLE_END, "filed": APPLE_FILED,
                     "val": ILLUSTRATIVE_SHARES},
                ]}},
            },
            "dei": {},
        }
    }


# ---------------------------------------------------------------------------
# The sign rule
# ---------------------------------------------------------------------------

class TestCanonicalCapexRule:
    def test_sec_positive_becomes_negative_outflow(self):
        assert canonical_capex(APPLE_CAPEX_RAW, "edgar") == -APPLE_CAPEX_RAW

    def test_sec_already_negative_is_untouched(self):
        assert canonical_capex(-APPLE_CAPEX_RAW, "edgar") == -APPLE_CAPEX_RAW

    def test_yahoo_negative_is_not_negated_again(self):
        # The audit's explicit warning: do NOT blindly negate Yahoo capex.
        assert canonical_capex(-12_715_000_000.0, "yfinance") == -12_715_000_000.0

    def test_yahoo_positive_is_flipped(self):
        assert canonical_capex(12_715_000_000.0, "yfinance") == -12_715_000_000.0

    def test_zero_stays_zero(self):
        assert canonical_capex(0.0, "edgar") == 0.0
        assert canonical_capex(0.0, "yfinance") == 0.0

    def test_repeat_normalization_is_idempotent(self):
        for raw, source in ((APPLE_CAPEX_RAW, "edgar"), (-APPLE_CAPEX_RAW, "edgar"),
                            (-APPLE_CAPEX_RAW, "yfinance"), (APPLE_CAPEX_RAW, "yfinance")):
            once = canonical_capex(raw, source)
            assert canonical_capex(once, source) == once
            assert canonical_capex(canonical_capex(once, source), source) == once
            assert once <= 0  # canonical form is always an outflow

    def test_sec_and_yahoo_converge_on_one_canonical_value(self):
        # Same economic fact, opposite raw conventions -> identical canonical
        # value, hence identical FCF (audit acceptance criterion).
        assert canonical_capex(APPLE_CAPEX_RAW, "edgar") == canonical_capex(
            -APPLE_CAPEX_RAW, "yfinance")


# ---------------------------------------------------------------------------
# SEC parse path (fresh records)
# ---------------------------------------------------------------------------

class TestSecParseSign:
    def test_parse_normalizes_positive_sec_capex(self):
        rec = edgar._parse_companyfacts(_apple_companyfacts())
        capex = rec[CAPEX_TAG]
        assert len(capex) == 1
        assert capex[0]["val"] == -APPLE_CAPEX_RAW
        assert rec[OCF_TAG][0]["val"] == APPLE_OCF
        # the period/filed metadata is untouched by the sign fix
        assert capex[0]["start"] == APPLE_START
        assert capex[0]["end"] == APPLE_END
        assert capex[0]["filed"] == APPLE_FILED

    def test_parse_leaves_already_negative_sec_capex_alone(self):
        payload = _apple_companyfacts()
        payload["facts"]["us-gaap"][CAPEX_TAG]["units"]["USD"][0]["val"] = -APPLE_CAPEX_RAW
        rec = edgar._parse_companyfacts(payload)
        assert rec[CAPEX_TAG][0]["val"] == -APPLE_CAPEX_RAW

    def test_parsed_facts_give_the_correct_fcf(self):
        rec = edgar._parse_companyfacts(_apple_companyfacts())
        fcf = rec[OCF_TAG][0]["val"] + rec[CAPEX_TAG][0]["val"]
        assert fcf == APPLE_FCF
        assert fcf != APPLE_FCF_BUGGY  # the old "OCF + positive capex" result

    def test_parse_is_idempotent_over_repeated_runs(self):
        payload = _apple_companyfacts()
        first = edgar._parse_companyfacts(payload)
        second = edgar._parse_companyfacts(payload)
        assert first[CAPEX_TAG][0]["val"] == second[CAPEX_TAG][0]["val"] == -APPLE_CAPEX_RAW

    def test_non_capex_tags_keep_their_raw_sign(self):
        rec = edgar._parse_companyfacts(_apple_companyfacts())
        assert rec[OCF_TAG][0]["val"] == APPLE_OCF       # inflow stays positive
        assert rec["NetIncomeLoss"][0]["val"] == 112_010_000_000


# ---------------------------------------------------------------------------
# Yahoo ingest path (fresh records, no network)
# ---------------------------------------------------------------------------

def _yf_frame(rows: dict[str, dict[str, float]]) -> pd.DataFrame:
    """yfinance-shaped statement frame: line items (index) x period ends
    (columns)."""
    return pd.DataFrame(rows).T


def _install_fake_yfinance(monkeypatch, capex_annual: float,
                            capex_quarterly: float) -> None:
    """Replace the yfinance module with an in-process fake.

    ``capex_annual``/``capex_quarterly`` are the RAW "Capital Expenditure"
    values the feed serves, so a test can pick the provider's convention.
    """
    q_cashflow = _yf_frame({
        "Operating Cash Flow": {"2025-06-30": 20_000_000_000.0},
        "Capital Expenditure": {"2025-06-30": capex_quarterly},
    })
    a_cashflow = _yf_frame({
        "Operating Cash Flow": {"2025-09-30": float(APPLE_OCF)},
        "Capital Expenditure": {"2025-09-30": capex_annual},
    })
    q_income = _yf_frame({"Net Income": {"2025-06-30": 20_000_000_000.0}})
    a_income = _yf_frame({"Net Income": {"2025-09-30": 95_000_000_000.0}})
    empty = _yf_frame({})

    class _FakeTicker:
        info = {"quoteType": "EQUITY"}
        quarterly_income_stmt = q_income
        quarterly_balance_sheet = empty
        quarterly_cashflow = q_cashflow
        income_stmt = a_income
        balance_sheet = empty
        cash_flow = a_cashflow

    fake = types.ModuleType("yfinance")
    fake.Ticker = lambda symbol: _FakeTicker()  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "yfinance", fake)


class TestYahooIngestSign:
    def test_negative_yahoo_capex_is_stored_unchanged(self, monkeypatch):
        _install_fake_yfinance(monkeypatch,
                               capex_annual=-float(APPLE_CAPEX_RAW),
                               capex_quarterly=-2_000_000_000.0)
        facts, note = fundamentals.fetch_rec("AAPL")
        assert note is None
        capex = [f for f in facts if f["tag"] == CAPEX_TAG]
        assert capex, "expected capex facts"
        assert all(f["val"] < 0 for f in capex)
        annual = [f for f in capex if f["end"] == "2025-09-30"][0]
        assert annual["val"] == -float(APPLE_CAPEX_RAW)

    def test_positive_yahoo_capex_is_flipped(self, monkeypatch):
        _install_fake_yfinance(monkeypatch,
                               capex_annual=float(APPLE_CAPEX_RAW),
                               capex_quarterly=2_000_000_000.0)
        facts, note = fundamentals.fetch_rec("AAPL")
        assert note is None
        capex = [f for f in facts if f["tag"] == CAPEX_TAG]
        assert all(f["val"] < 0 for f in capex)
        annual = [f for f in capex if f["end"] == "2025-09-30"][0]
        assert annual["val"] == -float(APPLE_CAPEX_RAW)

    def test_yahoo_and_sec_yield_the_same_fcf(self, monkeypatch):
        # Same economics, opposite raw conventions -> the same FCF.
        _install_fake_yfinance(monkeypatch,
                               capex_annual=-float(APPLE_CAPEX_RAW),
                               capex_quarterly=-2_000_000_000.0)
        facts, _ = fundamentals.fetch_rec("AAPL")
        ocf = [f for f in facts if f["tag"] == OCF_TAG and f["end"] == "2025-09-30"][0]
        capex = [f for f in facts if f["tag"] == CAPEX_TAG and f["end"] == "2025-09-30"][0]
        assert ocf["val"] + capex["val"] == APPLE_FCF


# ---------------------------------------------------------------------------
# Stored rows: legacy raw-positive SEC rows and canonical rows
# ---------------------------------------------------------------------------

class TestStoredRows:
    async def test_legacy_stored_sec_row_is_corrected_on_read(self, mem_db):
        """A row written before the sign fix still holds the raw positive SEC
        expenditure; reading it must yield the canonical negative outflow."""
        async with mem_db() as s:
            s.add(Fundamental(ticker="AAPL", tag=CAPEX_TAG, start=APPLE_START,
                              end=APPLE_END, filed=APPLE_FILED,
                              val=float(APPLE_CAPEX_RAW), currency="USD",
                              source="edgar"))
            s.add(Fundamental(ticker="AAPL", tag=OCF_TAG, start=APPLE_START,
                              end=APPLE_END, filed=APPLE_FILED,
                              val=float(APPLE_OCF), currency="USD", source="edgar"))
            await s.commit()

        fund = await fundamentals.load_fundamentals(["AAPL"])
        capex = fund["AAPL"][CAPEX_TAG][0]["val"]
        ocf = fund["AAPL"][OCF_TAG][0]["val"]
        assert capex == -float(APPLE_CAPEX_RAW)
        assert ocf == float(APPLE_OCF)
        assert ocf + capex == APPLE_FCF

    async def test_canonical_stored_row_is_not_inverted_twice(self, mem_db):
        async with mem_db() as s:
            s.add(Fundamental(ticker="AAPL", tag=CAPEX_TAG, start=APPLE_START,
                              end=APPLE_END, filed=APPLE_FILED,
                              val=-float(APPLE_CAPEX_RAW), currency="USD",
                              source="edgar"))
            await s.commit()

        fund = await fundamentals.load_fundamentals(["AAPL"])
        assert fund["AAPL"][CAPEX_TAG][0]["val"] == -float(APPLE_CAPEX_RAW)

    async def test_legacy_yahoo_row_with_positive_capex_is_corrected(self, mem_db):
        async with mem_db() as s:
            s.add(Fundamental(ticker="AAA", tag=CAPEX_TAG, start="2024-01-01",
                              end="2024-12-31", filed="2025-02-14",
                              val=2e8, currency="USD", source="yfinance"))
            await s.commit()

        fund = await fundamentals.load_fundamentals(["AAA"])
        assert fund["AAA"][CAPEX_TAG][0]["val"] == -2e8

    async def test_reading_twice_does_not_drift(self, mem_db):
        """The load path is a pure function of the stored row: loading twice
        yields the same canonical value (no cumulative inversion)."""
        async with mem_db() as s:
            s.add(Fundamental(ticker="AAPL", tag=CAPEX_TAG, start=APPLE_START,
                              end=APPLE_END, filed=APPLE_FILED,
                              val=float(APPLE_CAPEX_RAW), currency="USD",
                              source="edgar"))
            await s.commit()

        first = await fundamentals.load_fundamentals(["AAPL"])
        second = await fundamentals.load_fundamentals(["AAPL"])
        assert first["AAPL"][CAPEX_TAG][0]["val"] == second["AAPL"][CAPEX_TAG][0]["val"]
        assert first["AAPL"][CAPEX_TAG][0]["val"] == -float(APPLE_CAPEX_RAW)

    async def test_non_capex_tags_are_never_sign_flipped(self, mem_db):
        async with mem_db() as s:
            s.add(Fundamental(ticker="AAPL", tag=OCF_TAG, start=APPLE_START,
                              end=APPLE_END, filed=APPLE_FILED,
                              val=float(APPLE_OCF), currency="USD", source="edgar"))
            await s.commit()

        fund = await fundamentals.load_fundamentals(["AAPL"])
        assert fund["AAPL"][OCF_TAG][0]["val"] == float(APPLE_OCF)


# ---------------------------------------------------------------------------
# The real pipeline: ingest -> store -> load -> quality_snapshot
# ---------------------------------------------------------------------------

class TestQualitySnapshotPipeline:
    async def test_sec_payload_through_the_real_pipeline(self, mem_db, monkeypatch):
        """parse -> refresh_edgar (real write path) -> load_fundamentals ->
        quality_snapshot must produce Apple's true FCF of 98,767,000,000."""
        async with mem_db() as s:
            s.add(SecCik(ticker="AAPL", cik=320193))
            await s.commit()

        monkeypatch.setattr(edgar, "fetch_companyfacts",
                            lambda cik: edgar._parse_companyfacts(_apple_companyfacts()))
        monkeypatch.setattr(edgar.asyncio, "sleep", _no_sleep)

        statuses = await edgar.refresh_edgar(["AAPL"])
        assert statuses["AAPL"].startswith("ok(")

        # stored canonical
        async with mem_db() as s:
            rows = (await s.scalars(select(Fundamental).where(
                Fundamental.ticker == "AAPL",
                Fundamental.tag == CAPEX_TAG))).all()
            assert len(rows) == 1
            assert rows[0].val == -float(APPLE_CAPEX_RAW)
            assert rows[0].source == "edgar"

        fund = await fundamentals.load_fundamentals(["AAPL"])
        capex = fund["AAPL"][CAPEX_TAG][0]["val"]
        ocf = fund["AAPL"][OCF_TAG][0]["val"]
        assert (ocf, capex) == (float(APPLE_OCF), -float(APPLE_CAPEX_RAW))
        assert ocf + capex == APPLE_FCF

        q = quality_snapshot(fund, "AAPL", _ts("2025-12-31"),
                             close=ILLUSTRATIVE_PRICE)
        assert q is not None
        expected_p_fcf = (ILLUSTRATIVE_PRICE * ILLUSTRATIVE_SHARES) / APPLE_FCF
        assert q["p_fcf"] == pytest.approx(expected_p_fcf)
        # the old bug produced FCF = OCF + positive capex -> a smaller P/FCF
        buggy_p_fcf = (ILLUSTRATIVE_PRICE * ILLUSTRATIVE_SHARES) / APPLE_FCF_BUGGY
        assert q["p_fcf"] != pytest.approx(buggy_p_fcf)

    async def test_legacy_row_and_fresh_pipeline_agree(self, mem_db, monkeypatch):
        """A pre-fix stored row and a freshly ingested one must yield the same
        economics — the loader corrects the legacy row, so no migration and no
        divergent history."""
        async with mem_db() as s:
            s.add(SecCik(ticker="AAPL", cik=320193))
            # legacy rows as written before the sign fix: raw positive SEC capex
            s.add(Fundamental(ticker="AAPL", tag=CAPEX_TAG, start=APPLE_START,
                              end=APPLE_END, filed=APPLE_FILED,
                              val=float(APPLE_CAPEX_RAW), currency="USD",
                              source="edgar"))
            s.add(Fundamental(ticker="AAPL", tag=OCF_TAG, start=APPLE_START,
                              end=APPLE_END, filed=APPLE_FILED,
                              val=float(APPLE_OCF), currency="USD", source="edgar"))
            s.add(Fundamental(ticker="AAPL", tag="CommonStockSharesOutstanding",
                              start=None, end=APPLE_END, filed=APPLE_FILED,
                              val=ILLUSTRATIVE_SHARES, currency="shares",
                              source="edgar"))
            await s.commit()

        # the legacy history is corrected on read, not left divergent
        legacy = await fundamentals.load_fundamentals(["AAPL"])
        assert legacy["AAPL"][CAPEX_TAG][0]["val"] == -float(APPLE_CAPEX_RAW)
        assert (legacy["AAPL"][OCF_TAG][0]["val"]
                + legacy["AAPL"][CAPEX_TAG][0]["val"]) == APPLE_FCF
        legacy_q = quality_snapshot(legacy, "AAPL", _ts("2025-12-31"),
                                    close=ILLUSTRATIVE_PRICE)
        assert legacy_q is not None
        assert legacy_q["p_fcf"] == pytest.approx(
            (ILLUSTRATIVE_PRICE * ILLUSTRATIVE_SHARES) / APPLE_FCF)

        # re-running the refresh overwrites the legacy row with the canonical
        # value (self-healing, no data migration)
        monkeypatch.setattr(edgar, "fetch_companyfacts",
                            lambda cik: edgar._parse_companyfacts(_apple_companyfacts()))
        monkeypatch.setattr(edgar.asyncio, "sleep", _no_sleep)
        await edgar.refresh_edgar(["AAPL"], force=True)

        async with mem_db() as s:
            rows = (await s.scalars(select(Fundamental).where(
                Fundamental.ticker == "AAPL",
                Fundamental.tag == CAPEX_TAG))).all()
            assert len(rows) == 1
            assert rows[0].val == -float(APPLE_CAPEX_RAW)

        fund = await fundamentals.load_fundamentals(["AAPL"])
        assert fund["AAPL"][CAPEX_TAG][0]["val"] == -float(APPLE_CAPEX_RAW)
        assert fund["AAPL"][OCF_TAG][0]["val"] + fund["AAPL"][CAPEX_TAG][0]["val"] == APPLE_FCF

    async def test_quality_snapshot_is_point_in_time_on_the_normalized_facts(
            self, mem_db, monkeypatch):
        """The normalized fact is still point-in-time: invisible before it was
        filed, visible afterwards."""
        async with mem_db() as s:
            s.add(SecCik(ticker="AAPL", cik=320193))
            await s.commit()

        monkeypatch.setattr(edgar, "fetch_companyfacts",
                            lambda cik: edgar._parse_companyfacts(_apple_companyfacts()))
        monkeypatch.setattr(edgar.asyncio, "sleep", _no_sleep)
        await edgar.refresh_edgar(["AAPL"])
        fund = await fundamentals.load_fundamentals(["AAPL"])

        early = quality_snapshot(fund, "AAPL", _ts("2025-10-30"),
                                 close=ILLUSTRATIVE_PRICE)
        assert early is not None
        assert early["p_fcf"] is None  # not public yet -> unknown, not bad

        late = quality_snapshot(fund, "AAPL", _ts("2025-12-31"),
                                close=ILLUSTRATIVE_PRICE)
        assert late is not None
        assert late["p_fcf"] == pytest.approx(
            (ILLUSTRATIVE_PRICE * ILLUSTRATIVE_SHARES) / APPLE_FCF)


# ---------------------------------------------------------------------------
# The stored-row contract is what the strategy math consumes
# ---------------------------------------------------------------------------

def test_strategy_math_contract_is_additive_with_negative_capex():
    """Document the contract the monthly math relies on: FCF = OCF + capex
    with capex <= 0. If this ever changes, the monthly module must change
    with it."""
    ocf = float(APPLE_OCF)
    capex = canonical_capex(float(APPLE_CAPEX_RAW), "edgar")
    assert capex <= 0
    assert ocf + capex == APPLE_FCF


def test_fresh_and_legacy_rows_need_no_schema_change():
    """The fundamentals table already carries the provider identity in
    ``source``; the contract needs no new column."""
    fields = Fundamental.__table__.columns
    assert "source" in fields
    assert "tag" in fields
    assert not any(c.name == "capex_sign" for c in fields)


# ===========================================================================
# Non-finite hardening (NaN / +/-inf)
#
# SQLite stores NaN as NULL, which violates the NOT NULL constraint on
# fundamentals.val and rolls back the whole ticker's insert batch - one bad
# fact must never take the ticker's other, valid facts down with it.
# ===========================================================================

class TestNonFiniteCapexRule:
    def test_canonical_capex_rejects_nan(self):
        with pytest.raises(ValueError):
            canonical_capex(float("nan"), "edgar")

    def test_canonical_capex_rejects_infinities(self):
        for bad in (float("inf"), float("-inf")):
            with pytest.raises(ValueError):
                canonical_capex(bad, "yfinance")

    def test_canonical_capex_keeps_the_finite_sign_unchanged(self):
        assert canonical_capex(12_715_000_000.0, "edgar") == -12_715_000_000.0
        assert canonical_capex(-12_715_000_000.0, "yfinance") == -12_715_000_000.0
        assert canonical_capex(0.0, "edgar") == 0.0

    def test_finite_float_rejects_non_finite_and_non_numeric(self):
        assert finite_float(1.5) == 1.5
        assert finite_float(float("nan")) is None
        assert finite_float(float("inf")) is None
        assert finite_float(float("-inf")) is None
        assert finite_float(None) is None
        assert finite_float("not-a-number") is None


def _companyfacts_with_bad_capex() -> dict:
    """A companyfacts payload whose capex facts are NaN / +inf / -inf plus one
    genuinely valid period; the other tags are valid throughout."""
    payload = _apple_companyfacts()
    payload["facts"]["us-gaap"][CAPEX_TAG]["units"]["USD"] = [
        {"start": APPLE_START, "end": APPLE_END, "filed": APPLE_FILED,
         "val": float("nan")},
        {"start": "2023-09-25", "end": "2024-09-28", "filed": "2024-11-01",
         "val": float("inf")},
        {"start": "2022-09-25", "end": "2023-09-30", "filed": "2023-11-03",
         "val": float("-inf")},
        # the one good period, stored in SEC's raw positive convention
        {"start": "2021-09-26", "end": "2022-09-24", "filed": "2022-11-01",
         "val": APPLE_CAPEX_RAW},
    ]
    return payload


class TestSecNonFiniteCapex:
    def test_parse_drops_non_finite_capex_and_keeps_other_tags(self):
        rec = edgar._parse_companyfacts(_companyfacts_with_bad_capex())
        capex = rec[CAPEX_TAG]
        # only the valid period survives, in the canonical sign
        assert len(capex) == 1
        assert capex[0]["end"] == "2022-09-24"
        assert capex[0]["val"] == -float(APPLE_CAPEX_RAW)
        # the other tags are untouched
        assert rec[OCF_TAG][0]["val"] == APPLE_OCF
        assert rec["StockholdersEquity"][0]["val"] == 56_950_000_000
        assert rec["CommonStockSharesOutstanding"][0]["val"] == ILLUSTRATIVE_SHARES

    async def test_refresh_keeps_the_ticker_when_one_capex_fact_is_bad(
            self, mem_db, monkeypatch):
        """The regression this hardening exists for: one NaN capex fact must
        not roll back the ticker's whole insert batch."""
        async with mem_db() as s:
            s.add(SecCik(ticker="AAPL", cik=320193))
            await s.commit()

        monkeypatch.setattr(
            edgar, "fetch_companyfacts",
            lambda cik: edgar._parse_companyfacts(_companyfacts_with_bad_capex()))
        monkeypatch.setattr(edgar.asyncio, "sleep", _no_sleep)

        statuses = await edgar.refresh_edgar(["AAPL"])
        assert statuses["AAPL"].startswith("ok(")  # not "failed: ..."

        async with mem_db() as s:
            rows = (await s.scalars(select(Fundamental).where(
                Fundamental.ticker == "AAPL"))).all()
        by_tag: dict[str, list[Fundamental]] = {}
        for r in rows:
            by_tag.setdefault(r.tag, []).append(r)

        # the bad capex facts were never stored ...
        assert len(by_tag.get(CAPEX_TAG, [])) == 1
        assert by_tag[CAPEX_TAG][0].val == -float(APPLE_CAPEX_RAW)
        # ... and the ticker's valid facts were not rolled back with them
        assert by_tag[OCF_TAG][0].val == APPLE_OCF
        assert by_tag["StockholdersEquity"][0].val == 56_950_000_000
        assert by_tag["CommonStockSharesOutstanding"][0].val == ILLUSTRATIVE_SHARES

    async def test_refresh_drops_a_non_finite_non_capex_fact(self, mem_db, monkeypatch):
        """Same protection for the other tags: one bad fact is skipped, the
        rest of the ticker is stored."""
        payload = _apple_companyfacts()
        payload["facts"]["us-gaap"][OCF_TAG]["units"]["USD"][0]["val"] = float("nan")
        async with mem_db() as s:
            s.add(SecCik(ticker="AAPL", cik=320193))
            await s.commit()

        monkeypatch.setattr(edgar, "fetch_companyfacts",
                            lambda cik: edgar._parse_companyfacts(payload))
        monkeypatch.setattr(edgar.asyncio, "sleep", _no_sleep)

        statuses = await edgar.refresh_edgar(["AAPL"])
        assert statuses["AAPL"].startswith("ok(")

        async with mem_db() as s:
            rows = (await s.scalars(select(Fundamental).where(
                Fundamental.ticker == "AAPL"))).all()
        by_tag: dict[str, list[Fundamental]] = {}
        for r in rows:
            by_tag.setdefault(r.tag, []).append(r)
        assert OCF_TAG not in by_tag                       # the NaN fact
        assert by_tag["StockholdersEquity"][0].val == 56_950_000_000
        assert by_tag[CAPEX_TAG][0].val == -float(APPLE_CAPEX_RAW)


def _install_fake_yfinance_full(monkeypatch, capex_annual: float,
                                ocf_annual: float = float(APPLE_OCF)) -> None:
    """A complete yfinance-shaped statement set (income, balance sheet, cash
    flow) so the whole pipeline can be exercised: the raw capex convention is
    the caller's choice."""
    q_cashflow = _yf_frame({
        "Operating Cash Flow": {"2025-06-30": 20_000_000_000.0},
        "Capital Expenditure": {"2025-06-30": -2_000_000_000.0},
    })
    a_cashflow = _yf_frame({
        "Operating Cash Flow": {"2025-09-30": ocf_annual},
        "Capital Expenditure": {"2025-09-30": capex_annual},
    })
    q_income = _yf_frame({
        "Net Income": {"2025-06-30": 20_000_000_000.0},
        "Ordinary Shares Number": {"2025-06-30": ILLUSTRATIVE_SHARES},
    })
    a_income = _yf_frame({
        "Net Income": {"2025-09-30": 95_000_000_000.0},
        "Ordinary Shares Number": {"2025-09-30": ILLUSTRATIVE_SHARES},
    })
    q_balance = _yf_frame({"Stockholders Equity": {"2025-06-30": 56_950_000_000.0}})
    a_balance = _yf_frame({"Stockholders Equity": {"2025-09-30": 56_950_000_000.0}})

    class _FakeTicker:
        info = {"quoteType": "EQUITY"}
        quarterly_income_stmt = q_income
        quarterly_balance_sheet = q_balance
        quarterly_cashflow = q_cashflow
        income_stmt = a_income
        balance_sheet = a_balance
        cash_flow = a_cashflow

    fake = types.ModuleType("yfinance")
    fake.Ticker = lambda symbol: _FakeTicker()  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "yfinance", fake)


class TestYahooNonFiniteCapex:
    def test_fetch_skips_non_finite_capex_but_keeps_valid_facts(self, monkeypatch):
        _install_fake_yfinance_full(monkeypatch, capex_annual=float("nan"))
        facts, note = fundamentals.fetch_rec("AAPL")
        assert note is None
        capex = [f for f in facts if f["tag"] == CAPEX_TAG]
        # the annual (NaN) capex fact is gone, the valid quarterly one stays
        assert all(f["end"] != "2025-09-30" for f in capex)
        assert [f["val"] for f in capex] == [-2_000_000_000.0]
        # the other facts are unaffected
        ocf = [f for f in facts if f["tag"] == OCF_TAG and f["end"] == "2025-09-30"]
        assert len(ocf) == 1 and ocf[0]["val"] == float(APPLE_OCF)

    def test_fetch_skips_infinite_capex(self, monkeypatch):
        """``pd.isna`` catches NaN but NOT +/-inf, so the feed guard has to be
        an explicit finiteness check."""
        for bad in (float("inf"), float("-inf")):
            _install_fake_yfinance_full(monkeypatch, capex_annual=bad)
            facts, note = fundamentals.fetch_rec("AAPL")
            assert note is None
            capex = [f for f in facts if f["tag"] == CAPEX_TAG]
            assert all(f["end"] != "2025-09-30" for f in capex)
            assert [f["val"] for f in capex] == [-2_000_000_000.0]
            ocf = [f for f in facts if f["tag"] == OCF_TAG and f["end"] == "2025-09-30"]
            assert len(ocf) == 1 and ocf[0]["val"] == float(APPLE_OCF)

    def test_fetch_skips_infinite_non_capex_fact(self, monkeypatch):
        """The feed guard is a finiteness check, not a NaN check: an infinite
        operating cash flow must not become a fact either."""
        _install_fake_yfinance_full(monkeypatch,
                                    capex_annual=-float(APPLE_CAPEX_RAW),
                                    ocf_annual=float("inf"))
        facts, note = fundamentals.fetch_rec("AAPL")
        assert note is None
        ocf = [f for f in facts if f["tag"] == OCF_TAG]
        assert all(f["end"] != "2025-09-30" for f in ocf)
        assert [f["val"] for f in ocf] == [20_000_000_000.0]
        # the capex facts of the same ticker are unaffected
        capex = [f for f in facts if f["tag"] == CAPEX_TAG]
        assert [f["val"] for f in capex if f["end"] == "2025-09-30"] == [-float(APPLE_CAPEX_RAW)]

    async def test_yahoo_end_to_end_refresh_store_load_quality_snapshot(
            self, mem_db, monkeypatch):
        """The full Yahoo path: fetch -> refresh_universe (real DB write) ->
        load_fundamentals -> quality_snapshot, with the stored capex asserted
        to be a negative outflow tagged source='yfinance'."""
        _install_fake_yfinance_full(monkeypatch,
                                    capex_annual=-float(APPLE_CAPEX_RAW))

        statuses = await fundamentals.refresh_universe(tickers=["AAPL"])
        assert statuses["AAPL"] == "ok"

        # the stored row: canonical sign, yfinance source
        async with mem_db() as s:
            rows = (await s.scalars(select(Fundamental).where(
                Fundamental.ticker == "AAPL",
                Fundamental.tag == CAPEX_TAG))).all()
        annual = [r for r in rows if r.end == "2025-09-30"]
        assert len(annual) == 1
        assert annual[0].val == -float(APPLE_CAPEX_RAW)
        assert annual[0].val < 0
        assert annual[0].source == "yfinance"
        # every stored capex fact is a negative outflow
        assert all(r.val < 0 for r in rows)
        assert all(r.source == "yfinance" for r in rows)

        fund = await fundamentals.load_fundamentals(["AAPL"])
        loaded_capex = [f for f in fund["AAPL"][CAPEX_TAG] if f["end"] == "2025-09-30"]
        assert len(loaded_capex) == 1
        assert loaded_capex[0]["val"] == -float(APPLE_CAPEX_RAW)
        loaded_ocf = [f for f in fund["AAPL"][OCF_TAG] if f["end"] == "2025-09-30"]
        assert len(loaded_ocf) == 1
        assert loaded_ocf[0]["val"] == float(APPLE_OCF)

        q = quality_snapshot(fund, "AAPL", _ts("2025-12-31"),
                             close=ILLUSTRATIVE_PRICE)
        assert q is not None
        assert q["p_fcf"] == pytest.approx(
            (ILLUSTRATIVE_PRICE * ILLUSTRATIVE_SHARES) / APPLE_FCF)
        assert q["p_fcf"] != pytest.approx(
            (ILLUSTRATIVE_PRICE * ILLUSTRATIVE_SHARES) / APPLE_FCF_BUGGY)

    async def test_yahoo_end_to_end_with_infinite_capex(self, mem_db, monkeypatch):
        """+/-inf must not reach the DB either (SQLite stores inf happily, so
        nothing would complain until the value poisons the FCF math)."""
        _install_fake_yfinance_full(monkeypatch, capex_annual=float("inf"))

        statuses = await fundamentals.refresh_universe(tickers=["AAPL"])
        assert statuses["AAPL"] == "ok"

        async with mem_db() as s:
            rows = (await s.scalars(select(Fundamental).where(
                Fundamental.ticker == "AAPL"))).all()
        assert all(finite_float(r.val) is not None for r in rows)
        capex_rows = [r for r in rows if r.tag == CAPEX_TAG]
        assert all(r.end != "2025-09-30" for r in capex_rows)
        assert all(r.val < 0 for r in capex_rows)
        # the ticker's other facts are stored
        ocf_rows = [r for r in rows if r.tag == OCF_TAG and r.end == "2025-09-30"]
        assert len(ocf_rows) == 1 and ocf_rows[0].val == float(APPLE_OCF)

    async def test_yahoo_end_to_end_with_non_finite_capex(self, mem_db, monkeypatch):
        """A NaN capex from the feed must not cost the ticker its other facts."""
        _install_fake_yfinance_full(monkeypatch, capex_annual=float("nan"))

        statuses = await fundamentals.refresh_universe(tickers=["AAPL"])
        assert statuses["AAPL"] == "ok"

        async with mem_db() as s:
            rows = (await s.scalars(select(Fundamental).where(
                Fundamental.ticker == "AAPL"))).all()
        by_tag: dict[str, list[Fundamental]] = {}
        for r in rows:
            by_tag.setdefault(r.tag, []).append(r)

        # no NaN/inf capex row reached the DB ...
        assert all(finite_float(r.val) is not None for r in rows)
        # ... the valid facts of the same ticker were stored anyway
        annual_ocf = [r for r in by_tag[OCF_TAG] if r.end == "2025-09-30"]
        assert len(annual_ocf) == 1
        assert annual_ocf[0].val == float(APPLE_OCF)
        assert by_tag["StockholdersEquity"][0].val == 56_950_000_000
        assert by_tag["CommonStockSharesOutstanding"][0].val == ILLUSTRATIVE_SHARES

        fund = await fundamentals.load_fundamentals(["AAPL"])
        loaded_ocf = [f for f in fund["AAPL"][OCF_TAG] if f["end"] == "2025-09-30"]
        assert len(loaded_ocf) == 1
        assert loaded_ocf[0]["val"] == float(APPLE_OCF)
        assert fund["AAPL"]["StockholdersEquity"][0]["val"] == 56_950_000_000
        # the load must not raise, and the snapshot must still be computable
        q = quality_snapshot(fund, "AAPL", _ts("2025-12-31"),
                             close=ILLUSTRATIVE_PRICE)
        assert q is not None


class TestLoaderNonFinite:
    async def test_loader_drops_an_invalid_stored_capex_fact(self, mem_db):
        """A legacy row holding +/-inf (SQLite can store it) must not poison
        the valid facts of the same ticker."""
        async with mem_db() as s:
            s.add(Fundamental(ticker="AAPL", tag=CAPEX_TAG, start=APPLE_START,
                              end=APPLE_END, filed=APPLE_FILED,
                              val=float("inf"), currency="USD", source="edgar"))
            s.add(Fundamental(ticker="AAPL", tag=OCF_TAG, start=APPLE_START,
                              end=APPLE_END, filed=APPLE_FILED,
                              val=float(APPLE_OCF), currency="USD", source="edgar"))
            s.add(Fundamental(ticker="AAPL", tag="StockholdersEquity", start=None,
                              end=APPLE_END, filed=APPLE_FILED,
                              val=56_950_000_000.0, currency="USD", source="edgar"))
            s.add(Fundamental(ticker="AAPL", tag="CommonStockSharesOutstanding",
                              start=None, end=APPLE_END, filed=APPLE_FILED,
                              val=ILLUSTRATIVE_SHARES, currency="shares",
                              source="edgar"))
            await s.commit()

        fund = await fundamentals.load_fundamentals(["AAPL"])
        assert CAPEX_TAG not in fund["AAPL"]              # the invalid fact
        assert fund["AAPL"][OCF_TAG][0]["val"] == float(APPLE_OCF)
        assert fund["AAPL"]["StockholdersEquity"][0]["val"] == 56_950_000_000
        assert fund["AAPL"]["CommonStockSharesOutstanding"][0]["val"] == ILLUSTRATIVE_SHARES

    async def test_loader_drops_an_invalid_stored_ocf_fact(self, mem_db):
        async with mem_db() as s:
            s.add(Fundamental(ticker="AAPL", tag=OCF_TAG, start=APPLE_START,
                              end=APPLE_END, filed=APPLE_FILED,
                              val=float("inf"), currency="USD", source="edgar"))
            s.add(Fundamental(ticker="AAPL", tag=CAPEX_TAG, start=APPLE_START,
                              end=APPLE_END, filed=APPLE_FILED,
                              val=-float(APPLE_CAPEX_RAW), currency="USD",
                              source="edgar"))
            await s.commit()

        fund = await fundamentals.load_fundamentals(["AAPL"])
        assert OCF_TAG not in fund["AAPL"]
        assert fund["AAPL"][CAPEX_TAG][0]["val"] == -float(APPLE_CAPEX_RAW)

    async def test_loader_keeps_other_tickers_when_one_row_is_invalid(self, mem_db):
        async with mem_db() as s:
            s.add(Fundamental(ticker="BAD", tag=OCF_TAG, start=APPLE_START,
                              end=APPLE_END, filed=APPLE_FILED,
                              val=float("-inf"), currency="USD", source="edgar"))
            s.add(Fundamental(ticker="GOOD", tag=OCF_TAG, start=APPLE_START,
                              end=APPLE_END, filed=APPLE_FILED,
                              val=float(APPLE_OCF), currency="USD", source="edgar"))
            await s.commit()

        fund = await fundamentals.load_fundamentals(["BAD", "GOOD"])
        assert fund["GOOD"][OCF_TAG][0]["val"] == float(APPLE_OCF)
        assert OCF_TAG not in fund.get("BAD", {})
