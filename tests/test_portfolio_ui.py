"""Independent regression tests for the Phase-3B shared paper comparison module.

The tests exercise ``static/portfolio.js`` through Node's CommonJS entry point
(and the browser global it is required to publish).  They never import the
FastAPI app, never open a database and never touch the network: the Node runner
replaces ``fetch`` with a tripwire and all expected values are hand-computed in
``tests/fixtures/portfolio_ui_cases.json``.

The immutable Phase-3A browser fixture is used only to reconstruct *inputs* for
the captured legacy golden pins; the expected numbers were recorded from the
real legacy ``load*Equity`` functions during development and are committed in
the JSON fixture, so running these tests does not depend on
``/tmp/opencode/phase3b-baseline``.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "static" / "portfolio.js"
INDEX = ROOT / "static" / "index.html"
CASES_PATH = Path(__file__).resolve().parent / "fixtures" / "portfolio_ui_cases.json"
FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "ui_phase3.json"

CASES = json.loads(CASES_PATH.read_text(encoding="utf-8"))
FIXTURE = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))

# ---------------------------------------------------------------------------
# Digest re-pin history (source_guards + fixture payload pins, which live in
# tests/fixtures/portfolio_ui_cases.json):
#   - pinned at Phase 3A;
#   - re-pinned 2026-10-31 (in portfolio_ui_cases.json itself) after the
#     Phase-3C DB-switch UI and the Phase 3D/3E/3F slices (assistant drawer,
#     Record-transaction drawer, CURRENT live-config strip) changed the
#     guarded index.html Savings block and the ui_phase3.json fixture
#     document.
#   - Intactness evidence for the 2026-10-31 re-pin: mddOf and withLiveToday
#     digests are UNCHANGED from the 3A pins (byte-identical), the golden
#     overlay pins in test_legacy_golden_* still pass (comparison math
#     behavior), and `node --check` passes on both extracted inline scripts.
#     Only the Savings block digest moved, exactly where slice 3E retired the
#     prompt() buttons / ✕ remove-position anti-pattern.
#   - 2026-10-31 (dead-code pass, part 2): index.html's dead withLiveToday
#     copy was retired (portfolio.js's own withLiveToday is the live one and
#     is untouched), so its source_guards entry was REMOVED from
#     portfolio_ui_cases.json — a guard whose start marker no longer exists
#     cannot be pinned. Intactness evidence: the mddOf and Savings block
#     digests are UNCHANGED (they do not span the deleted function), the
#     golden overlay pins still pass, node --check passes on both inline
#     scripts.
#   - Amended later the same day (orchestrator, after the populated gate
#     caught two real 3E defects): the type select's onchange now updates
#     svTxnType before re-rendering (stale type kept the old field set), and
#     submitSavingsTxn re-renders BEFORE writing its success message (the
#     re-render clears the status line, wiping the confirmation). Savings
#     block digest re-pinned again; same intactness evidence.
#   - 2026-10-31 (declutter slice): the Savings render template's mirror
#     explanation was collapsed into a details disclosure (live mirror stats
#     stay visible); savings block re-pinned. Intactness evidence: mddOf and
#     withLiveToday digests unchanged, golden overlay pins pass, node --check
#     passes on both inline scripts.
#   - 2026-10-31 (dead-code pass, part 2): the withLiveToday guard entry was
#     removed together with the function (see the entry above); mddOf and
#     Savings digests unchanged.
#   - 2026-11-01 (type-D manual news-consult panel in the Monthly sub-view:
#     new #news-consult-section HTML in #monthly-view, the
#     news-consult-ticker select population in loadMonthlyStatus, and
#     consultNews/renderNewsConsult beside runMonthly; plus the retired-
#     hybrid wording in backfillSim's confirm()). NO guarded block spans
#     those edits, so NO digest moved — re-pin verified, not needed:
#     mddOf sha256 237c6ebde46f9af4c6078aec5511afa57805396a78dbcf462d2ba94c9e0641dd
#     and savings_block_all_functions_and_math sha256
#     77f1c31f0cfcaf6e6e13f386f27a56f2bfebecdb8f2ab3bbea76f9858b71b048 are
#     UNCHANGED (byte-identical, recomputed after the edit). Intactness
#     evidence: `node --check` passes on both extracted inline scripts of
#     static/index.html, the golden overlay pins in test_legacy_golden_*
#     still pass (comparison math behavior), and portfolio.js's withLiveToday
#     is untouched (index.html's retired copy was already gone at 3A+).
#   - 2026-11-01 (review-fix slice): the news-consult JS block is pinned as a
#     THIRD source_guards entry, news_consult_block_renderNewsConsult_and_
#     consultNews — boundaries: from the "// --- Type-D material-news consult
#     (one real agent run per click) ---" comment through the end of
#     consultNews (end marker "        async function backfillMonthly() {",
#     the same start-of-next-section extraction style as the Savings block),
#     sha256 177089c9dc256776fd1519f6f078aced6e5a6ed2578a7ef4ef6a9bf5131ee831.
#     The pinned block carries the review fixes: consultNews's catch now
#     renders an honest bounded error state (truncated 200-char technical
#     detail, dim) instead of the raw assertion, renderNewsConsult uses a
#     string citation's own words as its title, and the turns/tool_calls/lat
#     meta interpolations are escapeHtml-wrapped. Intactness evidence for
#     this re-pin: the two EXISTING pins are byte-identical — mddOf sha256
#     237c6ebde46f9af4c6078aec5511afa57805396a78dbcf462d2ba94c9e0641dd and
#     savings_block_all_functions_and_math sha256
#     77f1c31f0cfcaf6e6e13f386f27a56f2bfebecdb8f2ab3bbea76f9858b71b048
#     recomputed UNCHANGED after the edit; `node --check` passes on both
#     extracted inline scripts of static/index.html; the golden overlay pins
#     in test_legacy_golden_* still pass.
#   - 2026-11-01 (entry #2, type-D scope extension + run-all slice): the
#     news-consult JS block moved again — runNewsScan/renderNewsScan (the
#     "Run all (N)" background scan: POST news/scan, poll news/scan/status
#     every ~3s, verdict-chip summary) were added inside the pinned block,
#     and loadMonthlyStatus's picker population (OUTSIDE the pinned block)
#     now merges /api/monthly/status + /api/dailycore/status into the union
#     with book tags (M / DC / M+DC) and labels the scan button "Run all (N)".
#     The pinned block was re-pinned ONLY: sha256
#     0a9372ad0b18c1b669c611fd2e598474225d375d7607c5a0684a783d34eb1844 (was
#     177089c9dc256776fd1519f6f078aced6e5a6ed2578a7ef4ef6a9bf5131ee831).
#     Intactness evidence: the mddOf sha256
#     237c6ebde46f9af4c6078aec5511afa57805396a78dbcf462d2ba94c9e0641dd and the
#     savings_block_all_functions_and_math sha256
#     77f1c31f0cfcaf6e6e13f386f27a56f2bfebecdb8f2ab3bbea76f9858b71b048 are
#     UNCHANGED (byte-identical, recomputed after the edit); `node --check`
#     passes on both extracted inline scripts of static/index.html; the
#     golden overlay pins in test_legacy_golden_* still pass.
#   - 2026-11-01 (entry #3, news-consult panel moved to paper-workspace
#     level): the #news-consult-section HTML block moved out of
#     #monthly-view to a sibling of #sellscore-section (pure HTML move, no
#     JS/CSS change — all handlers address elements by id). NO digest
#     moved — all three source_guards are byte-identical, recomputed after
#     the move: mddOf sha256
#     237c6ebde46f9af4c6078aec5511afa57805396a78dbcf462d2ba94c9e0641dd,
#     savings_block_all_functions_and_math sha256
#     77f1c31f0cfcaf6e6e13f386f27a56f2bfebecdb8f2ab3bbea76f9858b71b048,
#     news_consult_block_renderNewsConsult_and_consultNews sha256
#     0a9372ad0b18c1b669c611fd2e598474225d375d7607c5a0684a783d34eb1844.
#     Intactness evidence: `node --check` passes on both extracted inline
#     scripts of static/index.html; the golden overlay pins in
#     test_legacy_golden_* still pass.
# ---------------------------------------------------------------------------

RESULT_PREFIX = "__PORTFOLIO_UI_RESULT__"

NODE_RUNNER = r"""
"use strict";
const fs = require("fs");

const modulePath = process.argv[2];
const payload = JSON.parse(fs.readFileSync(0, "utf8"));
const RESULT_PREFIX = "__PORTFOLIO_UI_RESULT__";

// A pure transform must not reach the network. Any attempt is a hard failure.
global.window = global;
global.fetch = function () {
    throw new Error("network access attempted by a pure portfolio module");
};
if (typeof global.XMLHttpRequest === "undefined") {
    global.XMLHttpRequest = function () {
        throw new Error("XMLHttpRequest constructed by a pure portfolio module");
    };
}

function revive(value) {
    if (Array.isArray(value)) return value.map(revive);
    if (value && typeof value === "object") {
        if (Object.prototype.hasOwnProperty.call(value, "$nonfinite")) {
            const tag = value.$nonfinite;
            if (tag === "NaN") return NaN;
            if (tag === "Infinity") return Infinity;
            if (tag === "-Infinity") return -Infinity;
            throw new Error("bad nonfinite sentinel: " + String(tag));
        }
        const out = {};
        for (const key of Object.keys(value)) out[key] = revive(value[key]);
        return out;
    }
    return value;
}

function snapshot(value) {
    // v8.serialize is not content-stable for arrays whose backing store can
    // change on read; a canonical content string is a faithful immutability
    // check (including NaN / ±Infinity / -0).
    function canonical(item) {
        if (typeof item === "number") {
            if (Number.isNaN(item)) return "NaN";
            if (item === Infinity) return "Infinity";
            if (item === -Infinity) return "-Infinity";
            if (Object.is(item, -0)) return "-0";
            return String(item);
        }
        if (Array.isArray(item)) return "[" + item.map(canonical).join(",") + "]";
        if (item && typeof item === "object") {
            return "{" + Object.keys(item)
                .map((key) => JSON.stringify(key) + ":" + canonical(item[key]))
                .join(",") + "}";
        }
        return JSON.stringify(item);
    }
    return canonical(value);
}

let loaded = null;
let loadError = null;
try {
    loaded = require(modulePath);
} catch (error) {
    loadError = String((error && error.stack) || error);
}

function resolveBuilder() {
    if (loaded && typeof loaded.buildComparisonSeries === "function") {
        return { fn: loaded.buildComparisonSeries, ctx: loaded };
    }
    if (loaded && loaded.default && typeof loaded.default.buildComparisonSeries === "function") {
        return { fn: loaded.default.buildComparisonSeries, ctx: loaded.default };
    }
    const globalApi = globalThis.PaperPortfolioUI;
    if (globalApi && typeof globalApi.buildComparisonSeries === "function") {
        return { fn: globalApi.buildComparisonSeries, ctx: globalApi };
    }
    return null;
}

const globalApi = globalThis.PaperPortfolioUI;
const globalExportsBuilder = !!globalApi && typeof globalApi.buildComparisonSeries === "function";
let builder = null;
if (!loadError) {
    builder = resolveBuilder();
    if (!builder) loadError = "static/portfolio.js does not export buildComparisonSeries";
}

async function runOne(item) {
    const input = revive(item.input);
    const before = snapshot(input);
    try {
        const output = await Promise.resolve(builder.fn.call(builder.ctx, input));
        const inputUnchanged = snapshot(input) === before;
        let outputAliasSafe = true;
        if (output && Array.isArray(output.series) && output.series.length
            && Array.isArray(output.series[0].data) && output.series[0].data.length) {
            const aliasBefore = snapshot(input);
            try {
                const point = output.series[0].data[0];
                const original = { time: point.time, value: point.value };
                point.time = "1900-01-01";
                point.value = -1;
                outputAliasSafe = snapshot(input) === aliasBefore;
                point.time = original.time;
                point.value = original.value;
            } catch (_) {
                outputAliasSafe = true;
            }
        }
        return {
            name: item.name,
            threw: null,
            output: output,
            input_unchanged: inputUnchanged,
            output_alias_safe: outputAliasSafe,
        };
    } catch (error) {
        return {
            name: item.name,
            threw: String((error && error.stack) || error),
            output: null,
            input_unchanged: snapshot(input) === before,
            output_alias_safe: true,
        };
    }
}

(async () => {
    const results = [];
    const golden = [];
    const positions = [];
    if (!loadError) {
        for (const item of payload.cases) results.push(await runOne(item));
        for (const item of payload.golden_inputs) golden.push(await runOne(item));
        const apiObject = (loaded && typeof loaded.renderPositionsHtml === "function")
            ? loaded
            : (globalThis.PaperPortfolioUI
                && typeof globalThis.PaperPortfolioUI.renderPositionsHtml === "function"
                ? globalThis.PaperPortfolioUI : null);
        for (const item of (payload.positions_cases || [])) {
            const record = { name: item.name, html: null, threw: null };
            try {
                if (!apiObject) throw new Error("renderPositionsHtml is not exported");
                record.html = apiObject.renderPositionsHtml(
                    revive(item.positions || []),
                    revive(item.options || {}),
                );
            } catch (error) {
                record.threw = String((error && error.stack) || error);
            }
            positions.push(record);
        }
    }
    process.stdout.write(RESULT_PREFIX + JSON.stringify({
        loaded: !loadError,
        load_error: loadError,
        global_exports_builder: globalExportsBuilder,
        commonjs_exports_builder: !!(loaded && typeof loaded.buildComparisonSeries === "function"),
        results: results,
        golden: golden,
        positions: positions,
    }) + "\n");
})().catch((error) => {
    process.stderr.write(String((error && error.stack) || error) + "\n");
    process.exit(1);
});
"""


def _run_node(cases: dict, golden_inputs: list[dict], tmp_dir: Path) -> dict:
    """Run the CommonJS module once for every case in a single Node process."""
    runner = tmp_dir / "portfolio_ui_node_runner.js"
    runner.write_text(NODE_RUNNER, encoding="utf-8")
    payload = {
        "cases": cases,
        "golden_inputs": golden_inputs,
        "positions_cases": CASES.get("positions_cases", []),
    }
    completed = subprocess.run(
        ["node", str(runner), str(MODULE)],
        input=json.dumps(payload),
        text=True,
        capture_output=True,
        timeout=120,
        check=False,
    )
    if completed.returncode != 0:
        return {
            "loaded": False,
            "load_error": (
                f"Node runner failed (exit {completed.returncode}):\n{completed.stderr.strip()}"
            ),
            "results": [],
            "golden": [],
        }
    for line in reversed(completed.stdout.splitlines()):
        if line.startswith(RESULT_PREFIX):
            return json.loads(line[len(RESULT_PREFIX) :])
    return {
        "loaded": False,
        "load_error": (
            "Node runner produced no result line.\n"
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        ),
        "results": [],
        "golden": [],
    }


def _live_accounts_from_fixture(document: dict) -> dict:
    """Map the committed fixture HTTP payloads to the module's account shape.

    This mapping is intentionally independent of the module: it mirrors the
    documented endpoints (equity/status/benchmark) and nothing else.
    """
    portfolios = document["modes"]["populated"]["portfolios"]
    return {
        "sim": {
            "snapshots": copy.deepcopy(portfolios["sim"]["equity"]),
            "live": copy.deepcopy(portfolios["sim"]["status"]),
        },
        "monthly": {
            "snapshots": copy.deepcopy(portfolios["monthly"]["equity"]),
            "live": copy.deepcopy(portfolios["monthly"]["status"]),
        },
        "dailycore": {
            "snapshots": copy.deepcopy(portfolios["daily_core"]["equity"]),
            "live": copy.deepcopy(portfolios["daily_core"]["status"]),
        },
        "benchmark": {
            "snapshots": copy.deepcopy(portfolios["sim"]["benchmark_equity"]),
            "live": copy.deepcopy(portfolios["sim"]["benchmark"]),
        },
    }


def _golden_inputs(document: dict) -> list[dict]:
    golden = document["legacy_golden"]
    today = golden["golden_inputs"]["today"]
    accounts = _live_accounts_from_fixture(FIXTURE)
    inputs = []
    for selected, mode in golden["golden_inputs"]["selected_modes"]:
        inputs.append(
            {
                "name": f"{selected}:{mode}",
                "input": {
                    "selected": selected,
                    "mode": mode,
                    "accounts": copy.deepcopy(accounts),
                    "today": today,
                },
            }
        )
    return inputs


@pytest.fixture(scope="session")
def module_payload(tmp_path_factory: pytest.TempPathFactory) -> dict:
    if not MODULE.is_file():
        return {
            "loaded": False,
            "load_error": (
                "DEPENDENCY PENDING: static/portfolio.js has not been delivered by the "
                "frontend worker yet. The tests fail (they do not skip/xfail) so the "
                "missing contract implementation cannot be mistaken for a pass."
            ),
            "results": [],
            "golden": [],
        }
    tmp_dir = tmp_path_factory.mktemp("portfolio_ui")
    payload = _run_node(CASES["cases"], _golden_inputs(CASES), tmp_dir)
    payload["_by_name"] = {entry["name"]: entry for entry in payload.get("results", [])}
    payload["_golden_by_name"] = {entry["name"]: entry for entry in payload.get("golden", [])}
    payload["_positions_by_name"] = {entry["name"]: entry for entry in payload.get("positions", [])}
    return payload


def _require_loaded(payload: dict) -> None:
    assert payload.get("loaded"), payload.get("load_error") or "module did not load"


def _assert_close(actual: object, expected: object, label: str) -> None:
    assert isinstance(actual, (int, float)) and not isinstance(actual, bool), (
        f"{label}: expected a number, got {actual!r}"
    )
    assert math.isfinite(float(actual)), f"{label}: non-finite value {actual!r}"
    assert math.isclose(float(actual), float(expected), rel_tol=1e-9, abs_tol=1e-9), (
        f"{label}: expected {expected!r}, got {actual!r}"
    )


def _assert_series_payload(actual: list[dict], expected: list[dict], primary_id: str) -> None:
    assert isinstance(actual, list), f"series must be a list, got {type(actual).__name__}"
    actual_by_id = {}
    for item in actual:
        assert isinstance(item, dict), f"series entry must be an object, got {item!r}"
        ident = item.get("id")
        assert isinstance(ident, str) and ident, f"series id must be a non-empty string: {item!r}"
        assert ident not in actual_by_id, f"duplicate series id {ident!r}"
        actual_by_id[ident] = item
        assert isinstance(item.get("label"), str) and item["label"].strip(), (
            f"{ident}: label must be a non-empty string"
        )
        assert isinstance(item.get("color"), str) and item["color"].strip(), (
            f"{ident}: color must be a non-empty string"
        )
    expected_ids = {item["id"] for item in expected}
    assert set(actual_by_id) == expected_ids, (
        f"series ids mismatch: expected {sorted(expected_ids)}, got {sorted(actual_by_id)}"
    )

    primaries = [item for item in actual if item.get("primary") is True]
    assert len(primaries) == 1, f"exactly one primary series required, got {len(primaries)}"
    assert primaries[0]["id"] == primary_id, (
        f"primary series must be {primary_id!r}, got {primaries[0]['id']!r}"
    )
    for item in actual:
        if item["id"] != primary_id:
            assert item.get("primary") is False, (
                f"overlay {item['id']!r} must not be marked primary"
            )

    for expected_series in expected:
        ident = expected_series["id"]
        item = actual_by_id[ident]
        expected_data = expected_series["data"]
        data = item.get("data")
        assert isinstance(data, list), f"{ident}: data must be a list"
        assert [point.get("time") for point in data] == [
            point["time"] for point in expected_data
        ], f"{ident}: time axis mismatch"
        assert len(data) == len(expected_data), f"{ident}: data length mismatch"
        previous = None
        for point, expected_point in zip(data, expected_data, strict=True):
            assert isinstance(point, dict), f"{ident}: data point must be an object"
            assert set(point) >= {"time", "value"}, f"{ident}: data point keys {point!r}"
            time_value = point["time"]
            assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", str(time_value)), (
                f"{ident}: time must be UTC YYYY-MM-DD, got {time_value!r}"
            )
            if previous is not None:
                assert time_value > previous, f"{ident}: times must be strictly ascending"
            previous = time_value
            _assert_close(point["value"], expected_point["value"], f"{ident}@{time_value}")


def _assert_contributed(actual: object, expected: list[dict]) -> None:
    assert isinstance(actual, list), "contributed must be a list"
    assert len(actual) == len(expected), (
        f"contributed length: expected {len(expected)}, got {len(actual)}"
    )
    for point, expected_point in zip(actual, expected, strict=True):
        assert isinstance(point, dict), "contributed point must be an object"
        assert point.get("time") == expected_point["time"]
        _assert_close(point.get("value"), expected_point["value"], f"contributed@{point['time']}")


def _warnings_text(output: dict) -> str:
    parts = []
    for key in ("warnings", "warning"):
        value = output.get(key)
        if value is not None:
            parts.append(json.dumps(value, sort_keys=True))
    return "\n".join(parts)


@pytest.mark.parametrize("case", CASES["cases"], ids=[case["name"] for case in CASES["cases"]])
def test_explicit_contract_case(case: dict, module_payload: dict) -> None:
    _require_loaded(module_payload)
    result = module_payload["_by_name"].get(case["name"])
    assert result is not None, f"Node runner returned no result for {case['name']}"
    assert result["threw"] is None, f"{case['name']} threw:\n{result['threw']}"
    output = result["output"]
    assert isinstance(output, dict), f"{case['name']}: output must be an object"
    expected = case["expected"]

    assert output.get("empty") is expected["empty"], (
        f"{case['name']}: empty expected {expected['empty']}, got {output.get('empty')!r}"
    )
    if expected["empty"]:
        assert output.get("series") == [], f"{case['name']}: empty output must have no series"
        assert output.get("contributed") == [], (
            f"{case['name']}: empty output must have no contributed baseline"
        )
        assert output.get("asOf") is None, f"{case['name']}: empty output asOf must be null"
        return

    assert output.get("asOf") == expected["asOf"], (
        f"{case['name']}: asOf expected {expected['asOf']!r}, got {output.get('asOf')!r}"
    )
    _assert_series_payload(output.get("series"), expected["series"], expected["primary"])
    _assert_contributed(output.get("contributed"), expected["contributed"])

    warning_text = _warnings_text(output)
    if expected.get("warnings_min"):
        assert warning_text, f"{case['name']}: expected a pct-denominator warning"
        pattern = expected.get("warning_contains")
        if pattern:
            assert re.search(pattern, warning_text), (
                f"{case['name']}: warning does not explain the omitted pct points: {warning_text}"
            )

    # The pct baseline is always the flat 100 line; USD is the primary's own
    # contributed total. This is the "not TWR" / "not mixed units" guard.
    primary_series = next(item for item in output["series"] if item.get("primary") is True)
    assert len(output["contributed"]) == len(primary_series["data"]), (
        f"{case['name']}: contributed baseline must align with the primary curve"
    )
    if case["input"]["mode"] == "pct":
        for point in output["contributed"]:
            _assert_close(point["value"], 100.0, f"{case['name']} contributed@{point['time']}")


def test_all_inputs_are_not_mutated(module_payload: dict) -> None:
    _require_loaded(module_payload)
    dirty = [
        entry["name"]
        for entry in [*module_payload["results"], *module_payload["golden"]]
        if entry.get("input_unchanged") is not True
    ]
    assert not dirty, f"buildComparisonSeries mutated its inputs for: {dirty}"
    aliased = [
        entry["name"]
        for entry in [*module_payload["results"], *module_payload["golden"]]
        if entry.get("output_alias_safe") is not True
    ]
    assert not aliased, (
        f"buildComparisonSeries exposed input-backed arrays/objects in its output for: {aliased}"
    )


def test_module_exports_commonjs_and_browser_global(module_payload: dict) -> None:
    _require_loaded(module_payload)
    assert module_payload.get("commonjs_exports_builder") is True, (
        "static/portfolio.js must export buildComparisonSeries via module.exports"
    )
    assert module_payload.get("global_exports_builder") is True, (
        "static/portfolio.js must publish globalThis.PaperPortfolioUI.buildComparisonSeries"
    )


def test_fixture_payload_matches_accepted_phase3a_baseline() -> None:
    provenance = CASES["legacy_golden"]["provenance"]
    digest = hashlib.sha256(FIXTURE_PATH.read_bytes()).hexdigest()
    assert digest == provenance["fixture_sha256"], (
        "tests/fixtures/ui_phase3.json changed from the accepted Phase-3A baseline; "
        "the frontend worker must not edit it"
    )


def test_legacy_golden_primary_and_overlay_values(module_payload: dict) -> None:
    _require_loaded(module_payload)
    golden = CASES["legacy_golden"]
    by_name = module_payload["_golden_by_name"]

    def series_by_id(output: dict, ident: str) -> dict:
        matches = [item for item in output["series"] if item.get("id") == ident]
        assert len(matches) == 1, f"expected exactly one {ident!r} series, got {len(matches)}"
        return matches[0]

    def point_at(item: dict, time_value: str) -> dict:
        matches = [point for point in item["data"] if point.get("time") == time_value]
        assert len(matches) == 1, f"{item['id']}: missing/duplicated point at {time_value}"
        return matches[0]

    for pin in golden["pins"]:
        key = f"{pin['selected']}:{pin['mode']}"
        result = by_name[key]
        assert result["threw"] is None, f"{key} threw:\n{result['threw']}"
        output = result["output"]
        assert output["empty"] is False, f"{key}: golden fixture must not be empty"
        assert output["asOf"] == golden["golden_inputs"]["today"], (
            f"{key}: asOf expected today's date, got {output['asOf']!r}"
        )
        item = series_by_id(output, pin["selected"])
        assert item.get("primary") is True, f"{key}: selected series must be primary"
        _assert_close(
            point_at(item, pin["time"])["value"],
            pin["value"],
            f"golden {pin['selected']}/{pin['mode']}@{pin['time']}",
        )

    for pin in golden["overlay_pins"]:
        for key, result in by_name.items():
            selected, mode = key.split(":", 1)
            if mode != pin["mode"]:
                continue
            output = result["output"]
            if not any(item.get("id") == pin["id"] for item in output["series"]):
                # An account can only be an overlay when it has recorded history;
                # in the golden fixture every account does.
                pytest.fail(f"{key}: expected overlay {pin['id']!r} in the golden output")
            item = series_by_id(output, pin["id"])
            if pin["id"] == selected:
                continue
            assert item.get("primary") is False, f"{key}: {pin['id']} must be an overlay"
            _assert_close(
                point_at(item, pin["time"])["value"],
                pin["value"],
                f"golden overlay {pin['id']}/{pin['mode']}@{pin['time']} ({key})",
            )


def test_legacy_golden_contributed_baseline_and_units(module_payload: dict) -> None:
    _require_loaded(module_payload)
    last_allowance = {
        "sim": FIXTURE["modes"]["populated"]["portfolios"]["sim"]["status"]["allowance_total"],
        "monthly": FIXTURE["modes"]["populated"]["portfolios"]["monthly"]["status"][
            "allowance_total"
        ],
        "dailycore": FIXTURE["modes"]["populated"]["portfolios"]["daily_core"]["status"][
            "valuation"
        ]["allowance_total"],
    }
    for key, result in module_payload["_golden_by_name"].items():
        selected, mode = key.split(":", 1)
        output = result["output"]
        assert output["empty"] is False, f"{key}: golden fixture must not be empty"
        primary = next(item for item in output["series"] if item.get("primary") is True)
        assert len(output["contributed"]) == len(primary["data"]), (
            f"{key}: contributed baseline must align with the primary curve"
        )
        if mode == "pct":
            for point in output["contributed"]:
                _assert_close(point["value"], 100.0, f"{key} contributed@{point['time']}")
            # The benchmark's last pct point is the documented presentation fix:
            # the legacy sim/DCA overlay had no live mark (46.4333...).
            if any(item.get("id") == "benchmark" for item in output["series"]):
                benchmark = next(item for item in output["series"] if item["id"] == "benchmark")
                _assert_close(
                    benchmark["data"][-1]["value"],
                    133.98,
                    f"{key} benchmark live mark",
                )
        else:
            assert output["contributed"][-1]["value"] == last_allowance[selected], (
                f"{key}: USD baseline last point must be the primary's own contributed total"
            )
            for item in output["series"]:
                assert item["data"], f"{key}/{item['id']}: USD series is empty"
                assert min(point["value"] for point in item["data"]) > 100.0, (
                    f"{key}/{item['id']}: USD series looks like a percentage, not actual equity"
                )


def test_phase3a_helpers_and_savings_block_byte_identical() -> None:
    assert INDEX.is_file(), "static/index.html is missing"
    source = INDEX.read_text(encoding="utf-8")
    guards = CASES["source_guards"]["functions"]
    for guard in guards:
        start_marker = guard["start_marker"]
        end_marker = guard["end_marker"]
        start = source.find(start_marker)
        assert start >= 0, (
            f"{guard['name']} start marker not found in static/index.html; the accepted "
            "Phase-3A helper/Savings block must remain byte-identical "
            "(frontend may not move, rename or rewrite it)"
        )
        end = source.find(end_marker, start)
        assert end >= 0, f"{guard['name']} end marker not found in static/index.html"
        block = source[start:end]
        digest = hashlib.sha256(block.encode("utf-8")).hexdigest()
        assert digest == guard["sha256"], (
            f"{guard['name']} changed from the accepted Phase-3A baseline:\n"
            f"expected sha256 {guard['sha256']}\n"
            f"got      sha256 {digest}\n"
            "The contract requires mddOf and all Savings functions/math "
            "to remain byte-identical (the retired index.html withLiveToday copy "
            "no longer has a guard entry; portfolio.js's copy is the live one)."
        )


def test_no_runtime_baseline_path_is_required() -> None:
    """The committed tests must be reproducible without the TEMP baseline copy."""
    test_source = Path(__file__).read_text(encoding="utf-8")
    baseline_marker = "/tmp/opencode" + "/phase3b-baseline/static/index.html"
    assert baseline_marker not in test_source, (
        "the committed test must not read the development-only TEMP baseline at runtime"
    )
    # The provenance string is documentation, not a runtime dependency.
    provenance = CASES["legacy_golden"]["provenance"]["capture_method"]
    assert "/tmp/opencode" in provenance


def _cells(fragment: str, tag: str) -> list[str]:
    return [
        re.sub(r"<[^>]+>", "", cell)
        for cell in re.findall(rf"<{tag}[^>]*>(.*?)</{tag}>", fragment, flags=re.S)
    ]


@pytest.mark.parametrize(
    "case",
    CASES.get("positions_cases", []),
    ids=[case["name"] for case in CASES.get("positions_cases", [])],
)
def test_shared_positions_renderer(case: dict, module_payload: dict) -> None:
    """The shared renderer must keep the legacy columns/formulas/escaping."""
    _require_loaded(module_payload)
    result = module_payload["_positions_by_name"].get(case["name"])
    assert result is not None, f"Node runner returned no positions result for {case['name']}"
    assert result["threw"] is None, f"{case['name']} threw:\n{result['threw']}"
    html = result["html"]
    assert isinstance(html, str)
    if "expected_html" in case:
        assert html == case["expected_html"]
        return
    rows = re.findall(r"<tr>(.*?)</tr>", html, flags=re.S)
    assert rows, f"{case['name']}: renderer produced no table rows"
    headers = _cells(rows[0], "th")
    assert headers == case["expected_headers"], f"{case['name']}: header order changed: {headers!r}"
    body = rows[1:]
    assert len(body) == len(case["expected_rows"]), (
        f"{case['name']}: expected {len(case['expected_rows'])} data rows, got {len(body)}"
    )
    for index, (row, expected_cells, expected_class) in enumerate(
        zip(body, case["expected_rows"], case["expected_row_classes"], strict=True)
    ):
        cells = _cells(row, "td")
        assert cells == expected_cells, (
            f"{case['name']} row {index}: expected {expected_cells!r}, got {cells!r}"
        )
        assert f'class="{expected_class}"' in row, (
            f"{case['name']} row {index}: expected class={expected_class!r} (P&L colour)"
        )
    if case.get("forbid_weight_column"):
        assert "Weight" not in headers
        assert "Rank" not in headers
