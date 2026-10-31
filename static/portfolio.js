/*
 * static/portfolio.js — Phase 3B shared paper-portfolio comparison + renderers.
 *
 * One pure, DOM-free module used by the Paper workspace. It is loaded before
 * the page's inline scripts and is also requireable from Node (offline tests):
 *
 *     window.PaperPortfolioUI           // browser global
 *     module.exports = require(...)     // CommonJS (no DOM access at load)
 *
 * CONTRACT — buildComparisonSeries({ selected, mode, accounts, today })
 *   selected : 'sim' | 'monthly' | 'dailycore'   (primary curve)
 *   mode     : 'pct' | 'usd'
 *   accounts : { sim?, monthly?, dailycore?, benchmark? } where each entry is
 *              { snapshots: Array, live: Object|null }
 *              - snapshots rows use `date` (monthly/daily-core) or `at`
 *                (sim/benchmark) plus `total_equity` and `allowance_total`.
 *              - live may be a raw status ({total_equity, allowance_total}) or
 *                a raw daily-core status ({valuation:{...}, ...}) whose
 *                valuation is flattened. No other source is consulted: today's
 *                point is appended/replaced only from this payload, never
 *                fabricated from current settings.
 *   today    : optional UTC 'YYYY-MM-DD' (tests pass it for determinism).
 *
 * Returns, without renaming any contract field:
 *   {
 *     series: [{ id, label, color, data: [{time:'YYYY-MM-DD', value}], primary }],
 *     contributed: [{ time, value }],   // selected account's baseline only
 *     asOf: string|null,                // primary's last plotted UTC day
 *     empty: boolean                    // primary has nothing plottable
 *   }
 *   plus `warnings: string[]` (extra, not part of the pinned shape).
 *
 * PRESENTATION RULES (intentional, disclosed in the Phase 3B report):
 *   - ONE daily UTC axis: every input row is collapsed to its UTC day, the LAST
 *     input row for that day wins, series are sorted ascending. The old Daily
 *     Sim chart plotted the raw benchmark curve with intraday/second timestamps
 *     and no live-today point; that is standardized here (no number changes).
 *   - % mode is gain on contributions: equity / own contributed total × 100
 *     (100 = break-even), NOT a time-weighted (TWR) return. Funding dates and
 *     amounts differ, so no line is an "alpha winner".
 *   - A % point is emitted only when the recorded contribution is a finite
 *     positive number. Zero / negative / missing contributions are omitted and
 *     reported in `warnings`; they are NEVER rendered as 0% or as raw USD.
 *   - USD mode plots actual equity for every curve on the same visible USD
 *     axis; the contributed baseline is the selected account's own recorded
 *     contributed series.
 *   - A primary account with no recorded snapshots returns an empty result
 *     (empty=true, series=[]) even if a live payload exists: recorded history
 *     is required, never a fabricated live-only curve. Missing overlay data is
 *     omitted, never zero-filled, and does not make the primary empty.
 *
 * renderPositionsHtml(positions, options) is the shared positions table
 * renderer (unchanged columns/formulas/colours/empty copy per account):
 *   options = { emptyText, showRank, showWeight, rankMap, totalEquity }
 *   - showRank adds the Daily-Core rank column (rankMap default '—').
 *   - showWeight adds the Daily-Core weight column (position value /
 *     total_equity, cash included, 1 decimal).
 *   - The Daily-Core band table is NOT absorbed here (account-specific view).
 */
(function (globalScope, factory) {
    'use strict';
    const api = factory();
    if (typeof module === 'object' && module !== null && module.exports) {
        module.exports = api;
    }
    if (globalScope && typeof globalScope === 'object') {
        globalScope.PaperPortfolioUI = api;
    }
})(typeof globalThis !== 'undefined' ? globalThis : null, function () {
    'use strict';

    const ACCOUNT_ORDER = ['sim', 'monthly', 'dailycore'];
    const SERIES_META = {
        sim: { label: 'Daily Sim (Bot)', color: '#42ff72' },
        monthly: { label: 'Monthly', color: '#4d9fff' },
        dailycore: { label: 'Daily-Core', color: '#b48cff' },
        benchmark: { label: 'DCA benchmark (derived)', color: '#ff9f43' },
    };
    const CONTRIBUTED_COLOR = '#7aaa81';
    const PERCENT_BASIS_NOTE =
        'equity / own contributions × 100 (100 baseline) — NOT TWR; funding differs by account';

    function finiteNumber(value) {
        if (typeof value === 'number') return Number.isFinite(value) ? value : null;
        if (typeof value === 'string') {
            const text = value.trim();
            if (!text) return null;
            const number = Number(text);
            return Number.isFinite(number) ? number : null;
        }
        return null;
    }

    // Normalize a row date / ISO timestamp to a UTC 'YYYY-MM-DD' day, or null.
    // Date-only strings are validated as real calendar days (2026-02-31 is
    // rejected instead of rolling into March); timestamps are converted to
    // their UTC day so the axis is timezone-stable.
    function utcDay(value) {
        if (value instanceof Date) {
            if (Number.isNaN(value.getTime())) return null;
            return value.toISOString().slice(0, 10);
        }
        if (typeof value !== 'string') return null;
        const text = value.trim();
        if (!text) return null;
        if (/^\d{4}-\d{2}-\d{2}$/.test(text)) {
            const parsed = new Date(text + 'T00:00:00Z');
            if (Number.isNaN(parsed.getTime())) return null;
            return parsed.toISOString().slice(0, 10) === text ? text : null;
        }
        // Naive ISO datetimes (no Z / offset) are treated as UTC so the UTC
        // day is deterministic regardless of the host timezone.
        let normalized = text;
        if (/^\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}(:\d{2}(\.\d+)?)?$/.test(text)) {
            normalized = text.replace(' ', 'T') + 'Z';
        }
        const parsed = new Date(normalized);
        if (Number.isNaN(parsed.getTime())) return null;
        return parsed.toISOString().slice(0, 10);
    }

    function todayUtc(value) {
        const day = utcDay(value);
        if (day) return day;
        return new Date().toISOString().slice(0, 10);
    }

    // Last input row per UTC day wins; invalid dates / non-finite equity are
    // skipped (tolerated, never coerced to zero).
    function normalizeSnapshots(snapshots) {
        const byDay = new Map();
        if (!Array.isArray(snapshots)) return [];
        for (const row of snapshots) {
            if (!row || typeof row !== 'object') continue;
            const day = utcDay(row.date !== undefined && row.date !== null && row.date !== ''
                ? row.date
                : row.at);
            if (!day) continue;
            const value = finiteNumber(row.total_equity);
            if (value === null) continue;
            byDay.set(day, {
                time: day,
                value,
                contributed: finiteNumber(row.allowance_total),
            });
        }
        return Array.from(byDay.values())
            .sort((a, b) => (a.time < b.time ? -1 : a.time > b.time ? 1 : 0));
    }

    // Flatten a raw status / valuation payload into one live point.
    function livePoint(live) {
        if (!live || typeof live !== 'object') return null;
        const source = (live.valuation && typeof live.valuation === 'object')
            ? live.valuation
            : live;
        const value = finiteNumber(source.total_equity);
        if (value === null) return null;
        return { value, contributed: finiteNumber(source.allowance_total) };
    }

    // Append/replace TODAY using the provided live payload only. If the live
    // payload has no valid equity nothing is appended (no last-mark guess).
    function withLiveToday(points, live, today) {
        const liveValue = livePoint(live);
        if (!liveValue || !points.length) return points;
        const last = points[points.length - 1];
        if (last.time === today) {
            const copy = points.slice();
            copy[copy.length - 1] = {
                time: today,
                value: liveValue.value,
                contributed: liveValue.contributed,
            };
            return copy;
        }
        if (last.time < today) {
            return points.concat([{
                time: today,
                value: liveValue.value,
                contributed: liveValue.contributed,
            }]);
        }
        return points;   // future-dated recorded point: do not invent "today"
    }

    function buildSeries(entry, mode, today, id) {
        const meta = SERIES_META[id] || { label: id, color: '#7aaa81' };
        const points = withLiveToday(
            normalizeSnapshots(entry && entry.snapshots),
            entry && entry.live,
            today);
        const data = [];
        const contributed = [];
        let omittedPercent = 0;
        for (const point of points) {
            if (mode === 'pct') {
                if (point.contributed !== null && point.contributed > 0) {
                    data.push({ time: point.time, value: point.value / point.contributed * 100 });
                    contributed.push({ time: point.time, value: 100 });
                } else {
                    // Zero / negative / missing contribution: % is undefined.
                    // Keep the point out of both the curve and the baseline.
                    omittedPercent += 1;
                }
            } else {
                data.push({ time: point.time, value: point.value });
                if (point.contributed !== null) {
                    contributed.push({ time: point.time, value: point.contributed });
                }
            }
        }
        return { meta, points, data, contributed, omittedPercent };
    }

    function buildComparisonSeries(options) {
        const opts = options && typeof options === 'object' ? options : {};
        const selected = ACCOUNT_ORDER.indexOf(opts.selected) >= 0 ? opts.selected : 'sim';
        const mode = opts.mode === 'usd' ? 'usd' : 'pct';
        const accounts = (opts.accounts && typeof opts.accounts === 'object') ? opts.accounts : {};
        const today = todayUtc(opts.today);
        const warnings = [];

        const primary = buildSeries(accounts[selected], mode, today, selected);
        if (!primary.data.length) {
            if (!Array.isArray(accounts[selected] && accounts[selected].snapshots)
                || !accounts[selected].snapshots.length) {
                warnings.push(SERIES_META[selected].label
                    + ': no recorded snapshots — the comparison is unavailable rather than fabricated from a live valuation.');
            } else if (mode === 'pct' && !primary.data.length) {
                warnings.push(SERIES_META[selected].label
                    + ': % is undefined for every recorded point (zero, negative or missing contributions); '
                    + 'switch to USD to see actual equity. Points are not shown as 0%.');
            } else {
                warnings.push(SERIES_META[selected].label
                    + ': no plottable recorded points in this range — curve unavailable (not zero-filled).');
            }
            return {
                series: [],
                contributed: [],
                asOf: null,
                empty: true,
                warnings,
            };
        }

        if (primary.omittedPercent) {
            warnings.push(SERIES_META[selected].label + ': ' + primary.omittedPercent
                + ' point(s) omitted from the % view — % is undefined where the recorded contribution was '
                + 'zero, negative or missing; they are not shown as 0%.');
        }
        const primaryNoLive = primary.points.length > 0
            && livePoint(accounts[selected] && accounts[selected].live) === null;
        if (primaryNoLive) {
            warnings.push(SERIES_META[selected].label + ': latest recorded mark '
                + primary.data[primary.data.length - 1].time + ' — no live valuation supplied.');
        }

        const series = [{
            id: selected,
            label: SERIES_META[selected].label,
            color: SERIES_META[selected].color,
            data: primary.data,
            primary: true,
            kind: 'account',
        }];

        // Other paper accounts first, then the derived DCA benchmark. Missing
        // or empty overlays are omitted (never zero-filled) and never make the
        // primary empty.
        const overlayIds = ACCOUNT_ORDER.filter(id => id !== selected).concat(['benchmark']);
        for (const id of overlayIds) {
            if (!accounts[id]) continue;
            const built = buildSeries(accounts[id], mode, today, id);
            if (!built.data.length) {
                warnings.push(SERIES_META[id].label
                    + ': no plottable recorded points in this range — curve unavailable (omitted, not zero-filled).');
                continue;
            }
            if (built.omittedPercent) {
                warnings.push(SERIES_META[id].label + ': ' + built.omittedPercent
                    + ' point(s) omitted from the % view — % is undefined where the recorded contribution was '
                    + 'zero, negative or missing; they are not shown as 0%.');
            }
            if (livePoint(accounts[id] && accounts[id].live) === null) {
                warnings.push(SERIES_META[id].label + ': latest recorded mark '
                    + built.data[built.data.length - 1].time + ' — recorded snapshots only, no live valuation.');
            }
            series.push({
                id,
                label: SERIES_META[id].label,
                color: SERIES_META[id].color,
                data: built.data,
                primary: false,
                kind: id === 'benchmark' ? 'benchmark' : 'account',
            });
        }

        return {
            series,
            contributed: primary.contributed,
            asOf: primary.data[primary.data.length - 1].time,
            empty: false,
            warnings,
        };
    }

    function escapeHtml(value) {
        return String(value === null || value === undefined ? '' : value)
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;')
            .replace(/'/g, '&#39;');
    }

    /*
     * Shared positions table. Columns, formulas, formatting, colour classes and
     * per-account empty copy match the three legacy renderers exactly:
     *   standard: Ticker | Shares | Avg Cost | Price | Value | P&L%
     *   Daily-Core adds Rank after Ticker and Weight after Value.
     * Only the ticker is escaped (same as the legacy renderers); numeric fields
     * keep their toFixed formatting.
     */
    function renderPositionsHtml(positions, options) {
        const opts = options && typeof options === 'object' ? options : {};
        const rows = Array.isArray(positions) ? positions : [];
        const emptyText = opts.emptyText || 'No open positions.';
        if (!rows.length) return '<div class="sim-empty">' + emptyText + '</div>';
        const showRank = opts.showRank === true;
        const showWeight = opts.showWeight === true;
        const rankMap = (opts.rankMap && typeof opts.rankMap === 'object') ? opts.rankMap : {};
        const totalEquity = finiteNumber(opts.totalEquity);
        const currency = typeof opts.currency === 'string' && opts.currency ? opts.currency : '$';

        let header = '<th>Ticker</th>';
        if (showRank) header += '<th>Rank</th>';
        header += '<th>Shares</th><th>Avg Cost</th><th>Price</th><th>Value</th>';
        if (showWeight) header += '<th>Weight</th>';
        header += '<th>P&L%</th>';

        const body = rows.map(position => {
            const pnl = position.pnl_pct;
            const cls = pnl > 0 ? 'buy' : pnl < 0 ? 'sell' : '';
            const tradeCell = '<td>' + escapeHtml(position.ticker) + '</td>'
                + (showRank ? '<td>' + (rankMap[position.ticker] || '—') + '</td>' : '')
                + '<td>' + position.shares.toFixed(4) + '</td>'
                + '<td>' + currency + position.avg_cost.toFixed(2) + '</td>'
                + '<td>' + currency + position.current_price.toFixed(2) + '</td>'
                + '<td>' + currency + position.value.toFixed(2) + '</td>';
            const weightCell = showWeight
                ? '<td>' + (totalEquity !== null && totalEquity > 0
                    ? (position.value / totalEquity * 100).toFixed(1) + '%'
                    : '0.0%') + '</td>'
                : '';
            return '<tr>' + tradeCell + weightCell + '<td class="' + cls + '">'
                + (pnl >= 0 ? '+' : '') + pnl.toFixed(2) + '%</td></tr>';
        }).join('');

        return '<table><tr>' + header + '</tr>' + body + '</table>';
    }

    return {
        ACCOUNT_ORDER: ACCOUNT_ORDER.slice(),
        SERIES_META,
        CONTRIBUTED_COLOR,
        PERCENT_BASIS_NOTE,
        buildComparisonSeries,
        renderPositionsHtml,
        // Exposed for parity checks (pure helpers; no DOM).
        normalizeSnapshots,
        utcDay,
    };
});
