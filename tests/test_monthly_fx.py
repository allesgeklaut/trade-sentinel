"""Currency-correctness tests for the monthly USD conversion paths.

Phase 2 batch: the fill path must convert foreign OPENs with the pair's OPEN
(and closes with the pair's CLOSE), known non-USD tickers must fail closed
when FX is missing/invalid/not yet available, ``.VI`` maps to EURUSD, USD
tickers stay untouched, and no future rate may leak into an earlier result.

Fixtures are real temporary SQLite databases seeded with synthetic candle
rows (no network, no production DB).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app import fundamentals, monthly
from app.db import Base, Candle


def _business_days(n: int = 4) -> list[pd.Timestamp]:
    """Recent business days: the scalar maps bound their candle query to a
    400-day lookback from ``now``, so fixtures must stay inside it."""
    end = pd.Timestamp.now().normalize() - pd.Timedelta(days=3)
    return list(pd.bdate_range(end=end, periods=n))


def _candle(ticker: str, d, open_: float, close: float) -> Candle:
    return Candle(
        ticker=ticker,
        timestamp=pd.Timestamp(d).to_pydatetime(),
        open=float(open_),
        high=float(max(open_, close)),
        low=float(min(open_, close)),
        close=float(close),
        volume=1e6,
    )


async def _seed(session_factory, rows: list[Candle]) -> None:
    async with session_factory() as s:
        s.add_all(rows)
        await s.commit()


def _frame(values: dict[str, list[float]], days) -> pd.DataFrame:
    return pd.DataFrame(values, index=pd.DatetimeIndex(days))


@pytest.fixture
async def fx_db(tmp_path, monkeypatch):
    """File-backed temporary SQLite DB with monthly's Session hooked to it."""
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'fx.sqlite'}")
    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    import app.db as db_mod

    monkeypatch.setattr(monthly, "Session", session_factory)
    monkeypatch.setattr(db_mod, "Session", session_factory)
    yield session_factory
    await engine.dispose()


# ---------------------------------------------------------------------------
# OPEN vs CLOSE: regression for the _open_usd_map AttributeError
# ---------------------------------------------------------------------------

async def test_open_conversion_reads_pair_open_not_close(fx_db):
    """Regression: _open_usd_map selected Candle.open but read row.close for
    the FX pair -> AttributeError on every foreign fill. OPEN conversion must
    use the pair's OPEN and the stock's OPEN, even when open != close."""
    d0, d1 = _business_days(2)
    await _seed(fx_db, [
        _candle("EURUSD=X", d0, 1.10, 1.30),
        _candle("EURUSD=X", d1, 1.20, 1.40),
        _candle("SAP.DE", d0, 100.0, 110.0),
        _candle("SAP.DE", d1, 200.0, 220.0),
        _candle("AAPL", d1, 50.0, 60.0),
    ])

    out = await monthly._open_usd_map(["SAP.DE", "AAPL"], d1)
    # open*open — NOT open*close (280), close*open (264) or close*close (308)
    assert out["SAP.DE"] == pytest.approx(200.0 * 1.20)
    assert out["AAPL"] == pytest.approx(50.0)  # USD open unchanged


async def test_close_conversion_reads_pair_close_not_open(fx_db):
    """The valuation/decision path keeps using stock CLOSE x pair CLOSE."""
    d0, d1 = _business_days(2)
    await _seed(fx_db, [
        _candle("EURUSD=X", d0, 1.10, 1.30),
        _candle("EURUSD=X", d1, 1.20, 1.40),
        _candle("SAP.DE", d0, 100.0, 110.0),
        _candle("SAP.DE", d1, 200.0, 220.0),
        _candle("AAPL", d1, 50.0, 60.0),
    ])

    out = await monthly._price_usd_map(["SAP.DE", "AAPL"], asof=d1)
    assert out["SAP.DE"] == pytest.approx(220.0 * 1.40)
    assert out["AAPL"] == pytest.approx(60.0)


# ---------------------------------------------------------------------------
# Fail closed: missing / invalid / no-prior FX
# ---------------------------------------------------------------------------

async def test_missing_fx_fails_closed_never_native_as_usd(fx_db):
    d0, d1 = _business_days(2)
    await _seed(fx_db, [
        _candle("SAP.DE", d0, 100.0, 100.0),
        _candle("SAP.DE", d1, 101.0, 102.0),
    ])

    assert await monthly._price_usd_map(["SAP.DE"], asof=d1) == {"SAP.DE": None}
    assert await monthly._open_usd_map(["SAP.DE"], d1) == {"SAP.DE": None}
    # No USD price exists -> the frame has no usable row (never 102 "USD").
    close, _vol = await monthly.load_frames(["SAP.DE"], asof=d1)
    assert close.empty
    conv = monthly._fx_convert_usd(
        _frame({"SAP.DE": [100.0, 102.0]}, [d0, d1]))
    assert conv["SAP.DE"].isna().all()


@pytest.mark.parametrize("bad_rate", [0.0, -1.5, float("inf")])
async def test_invalid_fx_rate_fails_closed(fx_db, bad_rate):
    d0, d1 = _business_days(2)
    await _seed(fx_db, [
        _candle("EURUSD=X", d0, bad_rate, bad_rate),
        _candle("EURUSD=X", d1, bad_rate, bad_rate),
        _candle("SAP.DE", d0, 100.0, 100.0),
        _candle("SAP.DE", d1, 110.0, 110.0),
    ])

    assert await monthly._price_usd_map(["SAP.DE"], asof=d1) == {"SAP.DE": None}
    assert await monthly._open_usd_map(["SAP.DE"], d1) == {"SAP.DE": None}
    conv = monthly._fx_convert_usd(
        _frame({"EURUSD=X": [bad_rate, bad_rate],
                "SAP.DE": [100.0, 110.0]}, [d0, d1]))
    assert conv["SAP.DE"].isna().all()


async def test_fx_prior_to_instrument_inception_is_carried_forward(fx_db):
    """A rate printed before the instrument's first bar is the prevailing
    rate (causal forward carry), not a reason to fail."""
    d0, d1, _d2 = _business_days(3)
    await _seed(fx_db, [
        _candle("EURUSD=X", d0, 1.10, 1.10),
        _candle("SAP.DE", d1, 100.0, 100.0),
    ])

    assert (await monthly._price_usd_map(["SAP.DE"], asof=d1))["SAP.DE"] == \
        pytest.approx(110.0)
    assert (await monthly._open_usd_map(["SAP.DE"], d1))["SAP.DE"] == \
        pytest.approx(110.0)


async def test_missing_later_fx_print_carries_the_prior_rate_forward(fx_db):
    """Causal forward carry is preserved: once a rate exists, a missing later
    pair print keeps the prior rate until a new print arrives (only the FIRST
    rate is a fail-closed boundary)."""
    d0, d1, d2 = _business_days(3)
    await _seed(fx_db, [
        _candle("EURUSD=X", d0, 1.10, 1.10),
        _candle("SAP.DE", d0, 100.0, 100.0),
        _candle("SAP.DE", d1, 110.0, 110.0),
        _candle("SAP.DE", d2, 120.0, 120.0),
    ])
    assert (await monthly._price_usd_map(["SAP.DE"], asof=d2))["SAP.DE"] == \
        pytest.approx(132.0)
    assert (await monthly._open_usd_map(["SAP.DE"], d2))["SAP.DE"] == \
        pytest.approx(132.0)

    conv = monthly._fx_convert_usd(_frame(
        {"EURUSD=X": [1.10, np.nan, np.nan], "SAP.DE": [100.0, 110.0, 120.0]},
        [d0, d1, d2]))
    assert conv["SAP.DE"].tolist() == pytest.approx([110.0, 121.0, 132.0])


async def test_fx_after_instrument_inception_fails_closed_until_first_rate(fx_db):
    """A stock bar BEFORE the first stored FX print has no prior rate: None.
    The later rate must never be backfilled into the earlier date."""
    d0, d1, d2 = _business_days(3)
    await _seed(fx_db, [
        _candle("EURUSD=X", d2, 1.50, 1.60),
        _candle("SAP.DE", d0, 100.0, 100.0),
        _candle("SAP.DE", d1, 110.0, 110.0),
        _candle("SAP.DE", d2, 120.0, 120.0),
    ])

    assert (await monthly._price_usd_map(["SAP.DE"], asof=d1))["SAP.DE"] is None
    assert (await monthly._open_usd_map(["SAP.DE"], d1))["SAP.DE"] is None
    assert (await monthly._price_usd_map(["SAP.DE"], asof=d2))["SAP.DE"] == \
        pytest.approx(120.0 * 1.60)
    assert (await monthly._open_usd_map(["SAP.DE"], d2))["SAP.DE"] == \
        pytest.approx(120.0 * 1.50)

    conv = monthly._fx_convert_usd(
        _frame({"EURUSD=X": [np.nan, np.nan, 1.60],
                "SAP.DE": [100.0, 110.0, 120.0]}, [d0, d1, d2]))
    assert conv["SAP.DE"].iloc[:2].isna().all()
    assert conv["SAP.DE"].iloc[2] == pytest.approx(192.0)


# ---------------------------------------------------------------------------
# Suffix mapping: .VI added; USD and unknown suffixes untouched
# ---------------------------------------------------------------------------

def test_vi_maps_to_eurusd():
    assert fundamentals.SUFFIX_FX[".VI"] == ("EURUSD=X", "mul")
    assert fundamentals._suffix_fx("ABC.VI") == ("EURUSD=X", "mul")


async def test_vi_prices_convert_with_eurusd(fx_db):
    d0 = _business_days(1)[0]
    await _seed(fx_db, [
        _candle("EURUSD=X", d0, 1.25, 1.30),
        _candle("ABC.VI", d0, 10.0, 12.0),
    ])

    assert (await monthly._price_usd_map(["ABC.VI"], asof=d0))["ABC.VI"] == \
        pytest.approx(12.0 * 1.30)
    assert (await monthly._open_usd_map(["ABC.VI"], d0))["ABC.VI"] == \
        pytest.approx(10.0 * 1.25)


async def test_usd_and_unknown_suffix_prices_unchanged(fx_db):
    """USD tickers (no suffix) are never converted; unknown suffixes stay
    explicitly unresolved rather than guessed."""
    d0 = _business_days(1)[0]
    await _seed(fx_db, [
        _candle("EURUSD=X", d0, 1.25, 1.30),
        _candle("AAPL", d0, 200.0, 210.0),
        _candle("XYZ.MI", d0, 40.0, 44.0),
    ])

    out = await monthly._price_usd_map(["AAPL", "XYZ.MI"], asof=d0)
    assert out["AAPL"] == pytest.approx(210.0)
    assert out["XYZ.MI"] == pytest.approx(44.0)
    opens = await monthly._open_usd_map(["AAPL", "XYZ.MI"], d0)
    assert opens["AAPL"] == pytest.approx(200.0)
    assert opens["XYZ.MI"] == pytest.approx(40.0)


def test_frame_drops_pair_columns_and_keeps_usd(fx_db):
    days = _business_days(2)
    conv = monthly._fx_convert_usd(pd.DataFrame(
        {"EURUSD=X": [1.2, 1.3], "AAPL": [200.0, 210.0]},
        index=pd.DatetimeIndex(days)))
    assert "EURUSD=X" not in conv.columns
    assert conv["AAPL"].tolist() == [200.0, 210.0]


# ---------------------------------------------------------------------------
# Scalar/frame parity and no look-ahead
# ---------------------------------------------------------------------------

async def test_scalar_and_frame_paths_agree(fx_db):
    days = _business_days(4)
    eur_open = [1.10, 1.12, 1.08, 1.15]
    eur_close = [1.11, 1.13, 1.09, 1.16]
    stock_open = [100.0, 102.0, 99.0, 105.0]
    stock_close = [101.0, 103.0, 98.0, 106.0]
    rows: list[Candle] = []
    for d, eo, ec, so, sc in zip(days, eur_open, eur_close,
                                 stock_open, stock_close, strict=True):
        rows.append(_candle("EURUSD=X", d, eo, ec))
        rows.append(_candle("SAP.DE", d, so, sc))
        rows.append(_candle("AAPL", d, so + 50.0, sc + 50.0))
    await _seed(fx_db, rows)

    for d in days:
        px = await monthly._price_usd_map(["SAP.DE", "AAPL"], asof=d)
        op = await monthly._open_usd_map(["SAP.DE", "AAPL"], d)
        open_frame, close_frame, _vol = await monthly.load_ohlcv_frames(
            ["SAP.DE", "AAPL"], asof=d)
        for t in ("SAP.DE", "AAPL"):
            assert px[t] is not None
            assert op[t] is not None
            # scalar map and the frame loader book the same USD value
            assert px[t] == pytest.approx(monthly.px_at(close_frame, d, t))
            assert op[t] == pytest.approx(monthly.px_at(open_frame, d, t))


def test_frame_conversion_ignores_future_rows():
    days = pd.DatetimeIndex(pd.bdate_range("2026-01-05", periods=5))
    base = pd.DataFrame(
        {"EURUSD=X": [1.10, 1.20, 1.30, 1.40, 1.50],
         "SAP.DE": [100.0, 101.0, 102.0, 103.0, 104.0]},
        index=days)
    prefix = monthly._fx_convert_usd(base.iloc[:3])

    full = monthly._fx_convert_usd(base)
    pd.testing.assert_series_equal(full["SAP.DE"].iloc[:3], prefix["SAP.DE"])

    # Mutating/appending future data cannot change an earlier converted value.
    mutated = base.copy()
    mutated.loc[days[4], "EURUSD=X"] = 9.0
    mutated.loc[days[4], "SAP.DE"] = 999.0
    out = monthly._fx_convert_usd(mutated)
    pd.testing.assert_series_equal(out["SAP.DE"].iloc[:3], prefix["SAP.DE"])


async def test_scalar_maps_ignore_future_rows(fx_db):
    d0, d1, d2, d3 = _business_days(4)
    await _seed(fx_db, [
        _candle("EURUSD=X", d0, 1.1, 1.1),
        _candle("EURUSD=X", d1, 1.2, 1.2),
        _candle("SAP.DE", d0, 100.0, 100.0),
        _candle("SAP.DE", d1, 110.0, 110.0),
    ])
    before_px = await monthly._price_usd_map(["SAP.DE"], asof=d1)
    before_op = await monthly._open_usd_map(["SAP.DE"], d1)
    assert before_px == {"SAP.DE": pytest.approx(132.0)}
    assert before_op == {"SAP.DE": pytest.approx(132.0)}

    # Append future rows (relative to the query's asof) with wildly different
    # prices/rates: the earlier result must be bit-identical.
    await _seed(fx_db, [
        _candle("EURUSD=X", d2, 5.0, 5.0),
        _candle("EURUSD=X", d3, 6.0, 6.0),
        _candle("SAP.DE", d2, 500.0, 500.0),
        _candle("SAP.DE", d3, 600.0, 600.0),
    ])
    assert await monthly._price_usd_map(["SAP.DE"], asof=d1) == before_px
    assert await monthly._open_usd_map(["SAP.DE"], d1) == before_op


# ---------------------------------------------------------------------------
# Scalar source-price validation
# ---------------------------------------------------------------------------

def test_convert_usd_rejects_invalid_scalar_prices():
    fx = pd.Series([1.2], index=pd.DatetimeIndex(["2026-01-05"]))
    ts = pd.Timestamp("2026-01-06")
    for bad in (0.0, -1.0, float("nan"), float("inf"), float("-inf")):
        assert monthly._convert_usd(bad, ts, fx, "mul") is None
    assert monthly._convert_usd(100.0, ts, fx, "mul") == pytest.approx(120.0)


async def test_maps_reject_invalid_source_prices(fx_db):
    d0 = _business_days(1)[0]
    await _seed(fx_db, [
        _candle("EURUSD=X", d0, 1.2, 1.2),
        _candle("AAPL", d0, 0.0, -5.0),
        _candle("SAP.DE", d0, float("inf"), 100.0),
    ])

    assert (await monthly._price_usd_map(["AAPL"], asof=d0))["AAPL"] is None
    assert (await monthly._open_usd_map(["AAPL"], d0))["AAPL"] is None
    assert (await monthly._open_usd_map(["SAP.DE"], d0))["SAP.DE"] is None
    # the same ticker's valid close still converts
    assert (await monthly._price_usd_map(["SAP.DE"], asof=d0))["SAP.DE"] == \
        pytest.approx(120.0)
