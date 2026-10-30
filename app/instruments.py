"""Instrument quote metadata: the ONE owner of ticker-suffix conventions.

A ticker's Yahoo suffix is not just a listing venue — it carries facts the
strategy math must not guess:

  * **quote unit** (``QUOTE_UNIT_FACTORS``): the unit the venue quotes its
    prices in. London (``.L``) quotes equities in PENCE, not pounds, so a
    cached close of 1250.0 means £12.50. Every other venue modelled here
    quotes in whole currency units. This is explicit metadata, not a guess: a
    venue appears only when its quote unit is known to differ from 1.0, and
    the unit is applied BEFORE the FX rate (pence * 0.01 * GBPUSD).
  * **currency** (``SUFFIX_FX``): the FX pair that converts that listing
    currency to USD, and whether the pair multiplies (``XXXUSD=X``) or divides
    (``USDXXX=X``).
  * **USD-native notation** (``USD_NATIVE_DOT_SUFFIXES``): a dot suffix that
    denotes a US SHARE CLASS, not a foreign exchange. ``BRK.B`` is Berkshire's
    class-B common stock quoted in USD; its dot is not a currency code, so its
    statements are USD as they stand.

Consumers (do not re-implement these rules anywhere else):

  * ``app.monthly._fx_convert_usd`` — the price path: quote unit first, then
    the FX rate; fail closed (NaN) when no usable rate exists for a known
    non-USD suffix.
  * ``app.fundamentals.fetch_rec`` — the statement path: factor 1.0 for a
    USD-native listing, a real conversion for a known suffix, and fail closed
    (converted facts skipped) for anything else.
  * ``app.savings`` / ``app.main`` / ``app.monthly`` — the FX pair set a
    universe needs (``suffix_fx(t)[0]``).

Known LIMITATION (recorded, never guessed)
------------------------------------------
Reporting currency is NOT distinguished from trading currency: statement
values are assumed to be denominated in the suffix's currency. An issuer that
reports in USD while its listing quotes in GBp/GBP (or reports in EUR while
the local listing quotes something else) is not modelled — its facts would be
converted (or passed through) as if they were in the listing currency. There
is no reporting-currency metadata source in this codebase, so the gap is
recorded as open. It must stay open: an unresolved currency is never
"fixed" by silently assuming one (that is exactly the dishonesty the
fail-closed currency policy removed).
"""

from __future__ import annotations

# Yahoo exchange suffix -> FX pair: "XXXUSD=X" multiplies local->USD,
# "USDXXX=X" divides (local currency units per USD).
SUFFIX_FX: dict[str, tuple[str, str]] = {
    ".DE": ("EURUSD=X", "mul"), ".AS": ("EURUSD=X", "mul"), ".MC": ("EURUSD=X", "mul"),
    ".PA": ("EURUSD=X", "mul"), ".BR": ("EURUSD=X", "mul"), ".HE": ("EURUSD=X", "mul"),
    ".VI": ("EURUSD=X", "mul"),
    ".SW": ("USDCHF=X", "div"), ".ST": ("USDSEK=X", "div"), ".L": ("GBPUSD=X", "mul"),
}

# Suffix -> multiplier from the LISTED quote unit to the currency unit. London
# equities are quoted in pence (1/100 GBP), so a raw close must be scaled by
# 0.01 before GBPUSD=X is applied. Listed explicitly per venue: every other
# venue is 1.0 (whole units), never a guessed conversion.
QUOTE_UNIT_FACTORS: dict[str, float] = {".L": 0.01}

# Dot suffixes that are US share-class notations (BRK.B, BRK.A, ...), not
# foreign-exchange codes. A listing carrying one of these is quoted, and
# reports, in USD.
USD_NATIVE_DOT_SUFFIXES: frozenset[str] = frozenset({".A", ".B", ".C"})


def _suffix(ticker: str) -> str | None:
    """The ticker's dot suffix as written (``".L"``), or None without a dot.

    Same convention every suffix consumer already used:
    ``ticker[ticker.rfind("."):]`` when the ticker contains a dot. A trailing
    dot therefore yields ``"."`` — an unrecognised suffix, never USD.
    """
    return ticker[ticker.rfind("."):] if "." in ticker else None


def suffix_fx(ticker: str) -> tuple[str, str] | None:
    """``(FX pair, mode)`` for the ticker's listing currency, or None.

    None means "not a known non-USD suffix": the listing is either USD-native
    (see :func:`is_usd_native`) or its currency is unresolved. Callers must
    never read None as "USD" — they decide with :func:`is_usd_native`.
    """
    suffix = _suffix(ticker)
    return SUFFIX_FX.get(suffix) if suffix else None


def quote_unit_factor(ticker: str) -> float:
    """Multiplier from the listed quote unit to the currency unit (default 1.0).

    Explicit quote-unit metadata, not a guess: ``.L`` closes are pence and
    factor 0.01; every other suffix — including no suffix and an unknown
    suffix — quotes in whole currency units and factor 1.0.
    """
    suffix = _suffix(ticker)
    return QUOTE_UNIT_FACTORS.get(suffix, 1.0) if suffix else 1.0


def is_usd_native(ticker: str) -> bool:
    """True iff the ticker is a KNOWN USD listing.

    USD-native means: no dot at all (``AAPL``, ``BRK-B``) or a dot suffix that
    is a US share-class notation (``BRK.B``, ``BRK.A``, ``BRK.C``) — a share
    class, not a currency. Any other dot suffix is NOT known USD: its currency
    is unresolved and its converted facts must fail closed.
    """
    suffix = _suffix(ticker)
    return suffix is None or suffix in USD_NATIVE_DOT_SUFFIXES
