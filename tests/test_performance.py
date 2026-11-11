"""Tests for the shared cash-flow-neutral performance metrics.

Phase 1 "financial correctness" (audit finding P0-B): the drawdown must be
measured on a cash-flow-adjusted (unitized / time-weighted) NAV, not on
``equity / cumulative_contributions``.

The JS counterpart lives in ``static/index.html`` (``mddOf``); the parity
test at the bottom runs the *shipped* JavaScript through node and compares
it with the Python implementation.
"""

from __future__ import annotations

import json
import math
import shutil
import subprocess
from pathlib import Path

import pytest

from app.performance import funded_start_index, max_drawdown, unitized_nav

REPO_ROOT = Path(__file__).resolve().parent.parent
INDEX_HTML = REPO_ROOT / "static" / "index.html"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _old_ratio_drawdown(equity: list[float], invested: list[float]) -> float:
    """The WRONG convention that was replaced: peak-to-trough of equity/contributed.

    Kept in the test suite as the regression witness: it reports a drawdown
    for a deposit that never lost money.
    """
    peak = -math.inf
    mdd = 0.0
    for e, inv in zip(equity, invested, strict=True):
        ratio = e / inv if inv > 0 else 0.0
        if ratio > peak:
            peak = ratio
        if peak > 0:
            mdd = max(mdd, (peak - ratio) / peak)
    return mdd * 100


# ---------------------------------------------------------------------------
# the headline case: a deposit is not a loss
# ---------------------------------------------------------------------------

def test_deposit_without_price_move_is_not_a_drawdown():
    """Audit P0-B reproduction: 1000 -> 1200 -> deposit 1000 (no price move).

    Equity 2200, contributions 2000. The old contributed-ratio metric called
    this an 8.33% drawdown although not a dollar was lost.
    """
    equity = [1000.0, 1200.0, 2200.0]
    invested = [1000.0, 1000.0, 2000.0]
    assert _old_ratio_drawdown(equity, invested) == pytest.approx(8.3333, abs=1e-3)
    assert max_drawdown(equity, invested) == pytest.approx(0.0, abs=1e-12)


def test_deposit_after_a_gain_does_not_create_a_drawdown():
    equity = [1000.0, 1500.0, 2500.0]
    invested = [1000.0, 1000.0, 2000.0]
    assert max_drawdown(equity, invested) == pytest.approx(0.0, abs=1e-12)


def test_deposit_between_two_losses_is_still_a_real_drawdown():
    """The deposit does not hide the loss: -20%, a deposit, then -20% again.

    The NAV keeps compounding (1.0 -> 0.8 -> 0.8 -> 0.64), so the peak-to-
    trough drawdown is 36% — the two losses compound, and the deposit in
    between is not a loss itself.
    """
    equity = [1000.0, 800.0, 1800.0, 1440.0]
    invested = [1000.0, 1000.0, 2000.0, 2000.0]
    nav = unitized_nav(equity, invested)
    assert nav == pytest.approx([1.0, 0.8, 0.8, 0.64], abs=1e-12)
    assert max_drawdown(equity, invested) == pytest.approx(36.0, abs=1e-9)


# ---------------------------------------------------------------------------
# real losses
# ---------------------------------------------------------------------------

def test_actual_loss_is_reported():
    equity = [1000.0, 1200.0, 900.0]
    invested = [1000.0, 1000.0, 1000.0]
    assert max_drawdown(equity, invested) == pytest.approx(25.0, abs=1e-9)


def test_loss_before_a_deposit_is_not_double_counted():
    """-30% before the deposit, nothing after: one 30% drawdown, not more."""
    equity = [1000.0, 700.0, 1700.0]
    invested = [1000.0, 1000.0, 2000.0]
    # NAV: 1.0 -> 0.7 -> 1700 / (700 + 1000) = 1.0
    assert max_drawdown(equity, invested) == pytest.approx(30.0, abs=1e-9)


def test_peak_to_trough_uses_the_peak_not_the_start():
    """100 -> 150 -> 120: the drawdown is from the 150 peak (-20%), not -40%."""
    equity = [100.0, 150.0, 120.0]
    invested = [100.0, 100.0, 100.0]
    assert max_drawdown(equity, invested) == pytest.approx(20.0, abs=1e-9)


# ---------------------------------------------------------------------------
# withdrawals (negative flows)
# ---------------------------------------------------------------------------

def test_withdrawal_of_the_profit_is_not_a_drawdown():
    """Withdraw 200 out of 1200 -> 1000 left; nothing was lost."""
    equity = [1000.0, 1200.0, 1000.0]
    invested = [1000.0, 1000.0, 800.0]
    assert max_drawdown(equity, invested) == pytest.approx(0.0, abs=1e-12)


def test_withdrawal_plus_a_real_loss():
    equity = [1000.0, 1200.0, 1000.0, 900.0]
    invested = [1000.0, 1000.0, 800.0, 800.0]
    # NAV: 1.0 -> 1.2 -> 1.2 -> 1.08  => 10% from the 1.2 peak
    assert max_drawdown(equity, invested) == pytest.approx(10.0, abs=1e-9)


def test_full_withdrawal_is_not_a_loss():
    """Withdrawing everything leaves 0 equity but no loss: the base is 0, so
    the day has no computable return and the NAV is carried forward."""
    equity = [1000.0, 0.0]
    invested = [1000.0, 0.0]
    assert max_drawdown(equity, invested) == pytest.approx(0.0, abs=1e-12)


# ---------------------------------------------------------------------------
# total loss / degenerate values
# ---------------------------------------------------------------------------

def test_total_loss_is_a_100pct_drawdown_not_skipped():
    """A wipeout must NOT be silently skipped (the old JS skipped v <= 0)."""
    assert max_drawdown([1000.0, 0.0], [1000.0, 1000.0]) == pytest.approx(100.0, abs=1e-12)


def test_total_loss_in_the_middle_of_the_curve():
    equity = [1000.0, 0.0, 1000.0]
    invested = [1000.0, 1000.0, 1000.0]
    assert max_drawdown(equity, invested) == pytest.approx(100.0, abs=1e-12)


def test_chain_restarts_after_a_wipeout_but_the_drawdown_stays():
    """A fresh deposit after a wipeout restarts the NAV at 1 — the 100%
    drawdown was already recorded and must not be forgotten."""
    equity = [1000.0, 0.0, 1000.0]
    invested = [1000.0, 1000.0, 2000.0]
    nav = unitized_nav(equity, invested)
    assert nav[0] == pytest.approx(1.0)
    assert nav[1] == pytest.approx(0.0)
    assert nav[2] == pytest.approx(1.0)          # restarted chain
    assert max_drawdown(equity, invested) == pytest.approx(100.0, abs=1e-12)


def test_never_funded_curve_has_no_drawdown():
    assert max_drawdown([0.0, 0.0], [0.0, 0.0]) == 0.0


def test_empty_curve_has_no_drawdown():
    assert max_drawdown([], []) == 0.0
    assert unitized_nav([], []) == []


# ---------------------------------------------------------------------------
# startup handling
# ---------------------------------------------------------------------------

def test_startup_before_the_first_funding_is_skipped():
    """The account existed but held nothing yet: the chain starts at the
    first funded point (NAV = 1), not at the empty beginning."""
    equity = [0.0, 0.0, 1000.0, 1200.0]
    invested = [0.0, 0.0, 1000.0, 1000.0]
    nav = unitized_nav(equity, invested)
    assert nav[:2] == [0.0, 0.0]
    assert nav[2] == pytest.approx(1.0)
    assert nav[3] == pytest.approx(1.2)
    assert max_drawdown(equity, invested) == pytest.approx(0.0, abs=1e-12)
    assert funded_start_index(equity) == 2


def test_funded_start_index_of_an_unfunded_curve():
    assert funded_start_index([0.0, 0.0]) == -1


# ---------------------------------------------------------------------------
# no flows / plain equity drawdown
# ---------------------------------------------------------------------------

def test_without_contributions_it_is_the_plain_equity_drawdown():
    equity = [1000.0, 1200.0, 900.0, 950.0]
    assert max_drawdown(equity) == pytest.approx(25.0, abs=1e-9)


def test_constant_contributions_equal_the_plain_equity_drawdown():
    equity = [1000.0, 1200.0, 900.0, 950.0]
    assert max_drawdown(equity, [1000.0] * 4) == pytest.approx(max_drawdown(equity), abs=1e-9)


def test_plain_equity_drawdown_handles_a_wipeout():
    assert max_drawdown([1000.0, 0.0]) == pytest.approx(100.0, abs=1e-12)


# ---------------------------------------------------------------------------
# length mismatches are errors, never a silent truncation
# ---------------------------------------------------------------------------

def test_length_mismatch_raises_instead_of_truncating():
    with pytest.raises(ValueError, match="length mismatch"):
        max_drawdown([1000.0, 1200.0, 2200.0], [1000.0, 1000.0])
    with pytest.raises(ValueError, match="length mismatch"):
        unitized_nav([1000.0, 1200.0], [1000.0, 1000.0, 2000.0])


def test_length_mismatch_is_not_silently_zip_truncated():
    """A shorter curve would have produced a different number if truncated."""
    equity = [1000.0, 1200.0, 2200.0]
    invested = [1000.0, 1000.0]
    with pytest.raises(ValueError):
        max_drawdown(equity, invested)


# ---------------------------------------------------------------------------
# the NAV helper itself
# ---------------------------------------------------------------------------

def test_unitized_nav_chains_the_daily_returns():
    equity = [1000.0, 1100.0, 990.0]
    invested = [1000.0, 1000.0, 1000.0]
    # 1.0 -> 1.1 -> 1.1 * 0.9
    assert unitized_nav(equity, invested) == pytest.approx([1.0, 1.1, 0.99], abs=1e-12)


def test_unitized_nav_is_aligned_with_the_input_curve():
    equity = [0.0, 1000.0, 1200.0]
    invested = [0.0, 1000.0, 1000.0]
    nav = unitized_nav(equity, invested)
    assert len(nav) == len(equity)


def test_unitized_nav_without_contributions_tracks_the_equity_shape():
    equity = [1000.0, 1200.0, 900.0]
    nav = unitized_nav(equity)
    assert nav == pytest.approx([1.0, 1.2, 0.9], abs=1e-12)


# ---------------------------------------------------------------------------
# invalid observations (None / non-numeric / NaN / Infinity)
# ---------------------------------------------------------------------------

def test_invalid_middle_observation_does_not_fabricate_a_loss():
    """A NaN in the middle of the curve must be SKIPPED, not turned into a
    0-value point — a 0-value NAV is a total loss and would fabricate a
    100% drawdown out of a data glitch."""
    equity = [1000.0, 1200.0, float("nan"), 900.0]
    invested = [1000.0] * 4
    nav = unitized_nav(equity, invested)
    assert nav[2] == pytest.approx(1.2)      # carried forward, NOT 0.0
    assert 0.0 not in nav[1:]                # no fake wipeout anywhere
    # The last VALID baseline is kept: 1200 -> 900 is still a 25% loss.
    assert max_drawdown(equity, invested) == pytest.approx(25.0, abs=1e-9)


def test_null_observation_is_invalid_not_zero():
    """None is an invalid observation (Python None -> NaN), not a 0-value."""
    equity = [1000.0, None, 1200.0, 900.0]  # type: ignore[list-item]
    invested = [1000.0] * 4
    nav = unitized_nav(equity, invested)  # type: ignore[arg-type]
    assert nav[1] == pytest.approx(1.0)
    assert max_drawdown(equity, invested) == pytest.approx(25.0, abs=1e-9)  # type: ignore[arg-type]


def test_infinity_observation_is_invalid():
    equity = [1000.0, float("inf"), 1200.0, 900.0]
    invested = [1000.0] * 4
    nav = unitized_nav(equity, invested)
    assert nav[1] == pytest.approx(1.0)
    assert max_drawdown(equity, invested) == pytest.approx(25.0, abs=1e-9)


def test_non_numeric_observation_is_invalid():
    equity = [1000.0, "abc", 1200.0, 900.0]  # type: ignore[list-item]
    invested = [1000.0] * 4
    nav = unitized_nav(equity, invested)  # type: ignore[arg-type]
    assert nav[1] == pytest.approx(1.0)
    assert max_drawdown(equity, invested) == pytest.approx(25.0, abs=1e-9)  # type: ignore[arg-type]


def test_numeric_string_is_a_valid_observation():
    """float("1200") is a valid number in Python and in JS (Number("1200"))."""
    equity = [1000.0, "1200", 900.0]  # type: ignore[list-item]
    invested = [1000.0] * 3
    assert max_drawdown(equity, invested) == pytest.approx(25.0, abs=1e-9)  # type: ignore[arg-type]


def test_invalid_contribution_observation_carries_the_nav():
    """A bad contribution cannot be turned into a flow: the NAV carries
    forward and the drawdown stays at the last valid value."""
    equity = [1000.0, 1200.0, 1100.0]
    invested = [1000.0, 1000.0, float("nan")]
    nav = unitized_nav(equity, invested)
    assert nav[2] == pytest.approx(1.2)
    assert max_drawdown(equity, invested) == pytest.approx(0.0, abs=1e-12)


def test_invalid_observation_before_the_start_is_skipped():
    equity = [float("nan"), 1000.0, 1200.0]
    invested = [float("nan"), 1000.0, 1000.0]
    nav = unitized_nav(equity, invested)
    assert nav[0] == 0.0            # the chain had not started yet
    assert nav[1] == pytest.approx(1.0)
    assert max_drawdown(equity, invested) == pytest.approx(0.0, abs=1e-12)


def test_null_before_the_start_is_skipped():
    equity = [None, 1000.0, 1200.0]  # type: ignore[list-item]
    invested = [None, 1000.0, 1000.0]  # type: ignore[list-item]
    nav = unitized_nav(equity, invested)  # type: ignore[arg-type]
    assert nav[0] == 0.0
    assert nav[1] == pytest.approx(1.0)
    assert max_drawdown(equity, invested) == pytest.approx(0.0, abs=1e-12)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# JS / Python parity — runs the SHIPPED mddOf() from static/index.html
# ---------------------------------------------------------------------------

_PARITY_CASES: list[tuple[str, list[float], list[float]]] = [
    ("deposit only", [1000.0, 1200.0, 2200.0], [1000.0, 1000.0, 2000.0]),
    ("real loss", [1000.0, 1200.0, 900.0], [1000.0, 1000.0, 1000.0]),
    ("withdrawal", [1000.0, 1200.0, 1000.0], [1000.0, 1000.0, 800.0]),
    ("withdrawal then loss", [1000.0, 1200.0, 1000.0, 900.0],
     [1000.0, 1000.0, 800.0, 800.0]),
    ("total loss", [1000.0, 0.0], [1000.0, 1000.0]),
    ("total loss then deposit", [1000.0, 0.0, 1000.0], [1000.0, 1000.0, 2000.0]),
    ("full withdrawal", [1000.0, 0.0], [1000.0, 0.0]),
    ("withdrawal bigger than equity", [1000.0, 100.0], [1000.0, 0.0]),
    ("startup skip", [0.0, 1000.0, 1200.0], [0.0, 1000.0, 1000.0]),
    ("never funded", [0.0, 0.0], [0.0, 0.0]),
    ("single point", [1000.0], [1000.0]),
    ("mixed flows", [1000.0, 1500.0, 1500.0, 1200.0, 2000.0],
     [1000.0, 1000.0, 2000.0, 2000.0, 2500.0]),
    ("flat curve", [1000.0, 1000.0, 1000.0], [1000.0, 1000.0, 1000.0]),
    ("rising curve", [1000.0, 1100.0, 1210.0], [1000.0, 1000.0, 1000.0]),
    # Invalid observations: they must be skipped on both sides, never turned
    # into a 0-value point (that would fake a 100% drawdown).
    ("nan in the middle", [1000.0, 1200.0, float("nan"), 900.0], [1000.0] * 4),
    ("null in the middle", [1000.0, None, 1200.0, 900.0], [1000.0] * 4),
    ("infinity in the middle", [1000.0, float("inf"), 1200.0, 900.0], [1000.0] * 4),
    ("non-numeric in the middle", [1000.0, "abc", 1200.0, 900.0], [1000.0] * 4),
    ("numeric string in the middle", [1000.0, "1200", 900.0], [1000.0] * 3),
    ("nan in the contributions", [1000.0, 1200.0, 1100.0],
     [1000.0, 1000.0, float("nan")]),
    ("nan before the start", [float("nan"), 1000.0, 1200.0],
     [float("nan"), 1000.0, 1000.0]),
    ("null before the start", [None, 1000.0, 1200.0], [None, 1000.0, 1000.0]),
]


def _encode_json_value(v):
    """JSON cannot carry NaN/Infinity — encode them as markers.

    ``null`` is a native JSON value and stays ``null`` (the JS side must
    treat it as invalid, like Python's ``None``).
    """
    if v is None:
        return None
    if isinstance(v, float):
        if math.isnan(v):
            return "__NaN__"
        if math.isinf(v):
            return "__Infinity__" if v > 0 else "__-Infinity__"
    return v


def _extract_js_function(html: str, name: str) -> str:
    """Return the source of ``function name(...)`` from an HTML file."""
    marker = f"function {name}("
    if marker not in html:
        raise AssertionError(f"{name}() not found in the HTML")
    start = html.index(marker)
    i = html.index("{", start)
    depth = 0
    quote: str | None = None
    in_line_comment = False
    in_block_comment = False
    while i < len(html):
        ch = html[i]
        nxt = html[i + 1] if i + 1 < len(html) else ""
        if in_line_comment:
            if ch == "\n":
                in_line_comment = False
        elif in_block_comment:
            if ch == "*" and nxt == "/":
                in_block_comment = False
                i += 1
        elif quote:
            if ch == "\\":
                i += 1
            elif ch == quote:
                quote = None
        elif ch == "/" and nxt == "/":
            in_line_comment = True
            i += 1
        elif ch == "/" and nxt == "*":
            in_block_comment = True
            i += 1
        elif ch in "\"'`":
            quote = ch
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return html[start:i + 1]
        i += 1
    raise AssertionError(f"unbalanced braces while extracting {name}()")


def test_shipped_mddof_does_not_use_the_equity_over_contributed_ratio():
    """Guard against reintroducing the old (wrong) metric in the UI."""
    src = _extract_js_function(INDEX_HTML.read_text(encoding="utf-8"), "mddOf")
    assert "/ p.contributed" not in src
    assert "value / p.contributed" not in src


def test_shipped_mddof_checks_finiteness_explicitly():
    """The falsy coercion `Number(x) || 0` turns NaN/null into a 0-value
    point -> a fake 100% drawdown. The shipped function must use an
    explicit finite check instead."""
    src = _extract_js_function(INDEX_HTML.read_text(encoding="utf-8"), "mddOf")
    code = "\n".join(line.split("//")[0] for line in src.splitlines())  # drop comments
    assert "Number.isFinite" in code
    assert "|| 0" not in code


def test_js_and_python_agree(tmp_path):
    """Run the shipped mddOf() under node and compare with the Python metric."""
    node = shutil.which("node")
    if not node:
        pytest.skip("node is not installed")

    src = _extract_js_function(INDEX_HTML.read_text(encoding="utf-8"), "mddOf")
    driver = tmp_path / "mdd_parity.js"
    driver.write_text(
        src
        + "\n"
        + "const cases = JSON.parse(require('fs').readFileSync(0, 'utf8'));\n"
        + "const decode = v => {\n"
        + "    if (v === '__NaN__') return NaN;\n"
        + "    if (v === '__Infinity__') return Infinity;\n"
        + "    if (v === '__-Infinity__') return -Infinity;\n"
        + "    return v;\n"
        + "};\n"
        + "const out = cases.map(c => mddOf(c.points.map(p => "
        + "({ value: decode(p.value), contributed: decode(p.contributed) }))));\n"
        + "process.stdout.write(JSON.stringify(out));\n",
        encoding="utf-8",
    )
    payload = [
        {
            "points": [
                {"value": _encode_json_value(e), "contributed": _encode_json_value(i)}
                for e, i in zip(eq, inv, strict=True)
            ],
        }
        for _, eq, inv in _PARITY_CASES
    ]
    proc = subprocess.run(
        [node, str(driver)],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    js_values = json.loads(proc.stdout)

    assert len(js_values) == len(_PARITY_CASES)
    for (name, equity, invested), js_value in zip(_PARITY_CASES, js_values, strict=True):
        py_value = max_drawdown(equity, invested)
        assert js_value == pytest.approx(py_value, abs=1e-9), (
            f"JS/Python mismatch for {name!r}: js={js_value} py={py_value}")
