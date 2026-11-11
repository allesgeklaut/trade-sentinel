/*
 * tests/ui_phase3_smoke.js — Playwright helper for the OFFLINE fixture harness.
 *
 * This file is a Playwright function source for `browser_run_code_unsafe({
 * filename: "trade-sentinel/tests/ui_phase3_smoke.js" })` or an equivalent
 * `page`-based Playwright runner. It expects `tests/ui_fixture_server.py` to
 * be running (default http://127.0.0.1:8765).
 *
 * MCP VM / BROWSER USAGE (no process/require/dynamic import required)
 *   page.__fixtureOrigin = "http://127.0.0.1:8765"; // optional, wins over env
 *   page.__fixtureMode = "empty";                   // optional, wins over env
 *   await browser_run_code_unsafe({ filename: "trade-sentinel/tests/ui_phase3_smoke.js" });
 *
 * Use a FRESH page/tab for each populated/empty run: Playwright
 * addInitScript registrations are not cleared inside an existing page, so the
 * old stubs/firewall would remain. Close/reopen the page between modes.
 * Environment variables TS_FIXTURE_ORIGIN / TS_FIXTURE_MODE are still honored
 * when the MCP environment exposes them.
 *
 * WHAT IT DELIBERATELY DOES
 *   - installs a request firewall that aborts every non-fixture origin request
 *     and counts it (CDN scripts/fonts vs provider API attempts vs other);
 *   - injects minimal deterministic LightweightCharts and marked stubs BEFORE
 *     the page scripts run, because the harness never serves the CDN and the
 *     production index.html keeps its SRI attributes untouched;
 *   - captures full-page screenshots at 390 / 768 / 1440 CSS px to absolute
 *     paths under /opt/stacks/.playwright-out/ (never into the repo);
 *   - fails closed: the top-level `ok` requires every expected view check
 *     (active leaf + canonical hash + nav state + explicit readiness +
 *     documentWidth <= viewportWidth + no page errors) and zero provider/API
 *     attempts. CDN/provider request observations are reported per layer,
 *     never as a misleading summed total.
 *   - Phase 3B: additionally requires ONE #paper-comparison-chart owner, the
 *     pinned #paper-range / #paper-mode values, GET-only range refreshes with
 *     the documented 126/252/504/1260/12000 limits, no POST/trade request on
 *     toggles, one visible USD axis (no hidden left scale), account-switch
 *     primary legend/data, explicit empty-primary placeholders, and zero
 *     comparison charts created while Paper is hidden or zero-width. The
 *     chart stub records addLineSeries options + setData payloads; this is
 *     data/option evidence, NOT real axis/canvas rendering.
 *   - Phase 3C: after the viewport loop (one viewport, so guard-parity POSTs
 *     never land in a per-view trade-policy window), verifies the synthetic
 *     PROD/DEV database switch: pinned #db-toggle / #db-status /
 *     #db-dev-banner / #db-clone-btn / #db-clone-status DOM, Lab replay/reset
 *     group visibility, #nav-savings visibility, the #/savings PROD-only
 *     panel, and direct fetch() guard parity (403 vs ack) plus
 *     /api/db/status and /api/db/clone. Fails closed until the frontend DB
 *     UI lands; destructive routes are only ever fetch()ed, never clicked.
 *
 * WHAT IT DOES NOT DO
 *   - It does NOT provide real chart or markdown rendering. The chart stub is a
 *     labelled DOM placeholder, so screenshots prove navigation/layout and
 *     data-bound renderer calls only — NOT canvas chart behaviour.
 *   - It does NOT modify static/index.html, remove SRI, or weaken the app.
 */
async (page) => {
    const env = (typeof process !== "undefined" && process.env) ? process.env : {};
    const pageMode = (page && (page.__fixtureMode === "empty" || page.__fixtureMode === "populated"))
        ? page.__fixtureMode : null;
    const envMode = (env.TS_FIXTURE_MODE === "empty" || env.TS_FIXTURE_MODE === "populated")
        ? env.TS_FIXTURE_MODE : null;
    const requestedMode = pageMode || envMode || "populated";
    const modeSource = pageMode ? "page.__fixtureMode"
        : (envMode ? "TS_FIXTURE_MODE" : "default");
    const pageOrigin = (page && typeof page.__fixtureOrigin === "string" && page.__fixtureOrigin.trim())
        ? page.__fixtureOrigin.trim() : null;
    const envOrigin = (typeof env.TS_FIXTURE_ORIGIN === "string" && env.TS_FIXTURE_ORIGIN.trim())
        ? env.TS_FIXTURE_ORIGIN.trim() : null;
    const originSource = pageOrigin ? "page.__fixtureOrigin"
        : (envOrigin ? "TS_FIXTURE_ORIGIN" : "default");
    const fixtureOrigin = new URL(pageOrigin || envOrigin || "http://127.0.0.1:8765").origin;
    const fixtureUrl = fixtureOrigin + "/?fixture=" + requestedMode;
    const outputDir = "/opt/stacks/.playwright-out";
    const runId = Date.now().toString(36);

    const blocked = [];
    const counts = { cdn_assets: 0, provider_api: 0, other: 0 };
    const observedExternal = { cdn_assets: 0, provider_api: 0, other: 0, details: [] };
    const aggregate = {
        pageErrors: [],
        resourceErrors: [],
        cspViolations: 0,
        cspBlocked: [],
        inPageExternalFetch: 0,
        inPageExternalXhr: 0,
        chartStubCalls: 0,
        chartSeriesCalls: 0,
        chartDataPoints: 0,
        chartContainers: [],
        chartRecords: [],
        chartHiddenCount: 0,
        chartZeroWidthCount: 0,
        markdownParseCalls: 0,
        stubs: { lightweightCharts: false, marked: false },
    };
    const seenPageErrors = new Set();
    const seenResourceErrors = new Set();
    const navigationUrls = [];
    let probeMerges = 0;
    const addPageError = (message) => {
        const text = String(message || "unknown page error");
        if (seenPageErrors.has(text)) return;
        seenPageErrors.add(text);
        aggregate.pageErrors.push(text);
    };
    const addResourceError = (entry) => {
        const key = String(entry && entry.tag) + "|" + String(entry && entry.url);
        if (seenResourceErrors.has(key)) return;
        seenResourceErrors.add(key);
        aggregate.resourceErrors.push(entry);
    };
    // Playwright pageerror survives navigations, unlike a single page-context
    // probe. It captures uncaught exceptions and unhandled rejections from
    // EVERY navigation/viewport, which is what the top-level ok must require.
    page.on("pageerror", (error) => {
        addPageError(error && error.message ? error.message : error);
    });

    // CSP blocks the jsDelivr scripts before the route layer can see them, so
    // also observe request events. This is the authoritative attempt counter;
    // the route counter records what the Playwright firewall actually aborted.
    page.on("request", (request) => {
        let url;
        try {
            url = new URL(request.url());
        } catch (_) {
            return;
        }
        if (url.origin === fixtureOrigin) return;
        const resourceType = request.resourceType();
        let kind = "other";
        if (["script", "stylesheet", "font", "image", "media"].includes(resourceType)) {
            kind = "cdn_asset";
            observedExternal.cdn_assets += 1;
        } else if (["fetch", "xhr"].includes(resourceType)) {
            kind = "provider_api";
            observedExternal.provider_api += 1;
        } else {
            observedExternal.other += 1;
        }
        observedExternal.details.push({ url: url.origin + url.pathname, resourceType, kind });
    });

    // Hard firewall at the Playwright routing layer. Same-origin fixture traffic
    // (static + /api + /__fixture__) is allowed; everything else is aborted.
    await page.route("**/*", async (route) => {
        const request = route.request();
        let url;
        try {
            url = new URL(request.url());
        } catch (_) {
            counts.other += 1;
            blocked.push({ url: String(request.url()), resourceType: request.resourceType(), kind: "other" });
            await route.abort("blockedbyclient");
            return;
        }
        if (url.origin !== fixtureOrigin) {
            const resourceType = request.resourceType();
            let kind = "other";
            if (["script", "stylesheet", "font", "image", "media"].includes(resourceType)) {
                kind = "cdn_asset";
                counts.cdn_assets += 1;
            } else if (["fetch", "xhr"].includes(resourceType)) {
                kind = "provider_api";
                counts.provider_api += 1;
            } else {
                counts.other += 1;
            }
            blocked.push({ url: url.origin + url.pathname, resourceType, kind });
            await route.abort("blockedbyclient");
            return;
        }
        await route.continue();
    });

    // Runs before any page script. The chart/markdown objects are deterministic
    // stubs, visibly labelled as such, never a replica of the real libraries.
    const initScript = (origin, mode) => {
        const probe = {
            fixtureOrigin: origin,
            fixtureMode: mode,
            externalFetchAttempts: 0,
            externalXhrAttempts: 0,
            cspViolations: 0,
            cspBlocked: [],
            chartStubCalls: 0,
            chartSeriesCalls: 0,
            chartDataPoints: 0,
            chartContainers: [],
            chartRecords: [],
            chartHiddenCount: 0,
            chartZeroWidthCount: 0,
            markdownParseCalls: 0,
            resourceErrors: [],
            pageErrors: [],
            stubs: { lightweightCharts: true, marked: true },
        };
        window.__fixtureProbe = probe;

        const sameOrigin = (raw) => {
            try {
                return new URL(String(raw), location.href).origin === location.origin;
            } catch (_) {
                return false;
            }
        };

        const realFetch = window.fetch.bind(window);
        window.fetch = function (input, init) {
            const raw = (input && typeof input === "object" && "url" in input) ? input.url : String(input);
            if (!sameOrigin(raw)) {
                probe.externalFetchAttempts += 1;
                return Promise.reject(new TypeError("fixture firewall: external fetch blocked"));
            }
            return realFetch(input, init);
        };

        const realXhrOpen = XMLHttpRequest.prototype.open;
        XMLHttpRequest.prototype.open = function (method, url, ...rest) {
            this.__fixtureExternal = !sameOrigin(url);
            if (this.__fixtureExternal) probe.externalXhrAttempts += 1;
            return realXhrOpen.call(this, method, url, ...rest);
        };
        const realXhrSend = XMLHttpRequest.prototype.send;
        XMLHttpRequest.prototype.send = function (body) {
            if (this.__fixtureExternal) {
                setTimeout(() => {
                    try { this.dispatchEvent(new ProgressEvent("error")); } catch (_) { /* ignore */ }
                }, 0);
                return;
            }
            return realXhrSend.call(this, body);
        };

        if (navigator.sendBeacon) {
            const realBeacon = navigator.sendBeacon.bind(navigator);
            navigator.sendBeacon = function (url, data) {
                if (!sameOrigin(url)) {
                    probe.externalFetchAttempts += 1;
                    return false;
                }
                return realBeacon(url, data);
            };
        }

        window.addEventListener("securitypolicyviolation", (event) => {
            probe.cspViolations += 1;
            probe.cspBlocked.push({
                directive: event.effectiveDirective || event.violatedDirective || "",
                blockedURI: event.blockedURI || "",
            });
        }, true);
        window.addEventListener("error", (event) => {
            if (event && event.target && event.target !== window && event.target.tagName) {
                probe.resourceErrors.push({
                    tag: String(event.target.tagName).toLowerCase(),
                    url: event.target.src || event.target.href || "",
                });
                return;
            }
            probe.pageErrors.push(String((event && (event.message || event.error)) || "error"));
        }, true);
        window.addEventListener("unhandledrejection", (event) => {
            probe.pageErrors.push("unhandledrejection: " + String(event.reason));
        });

        const isVisibleElement = (element) => {
            if (!element) return false;
            const style = window.getComputedStyle(element);
            const rect = element.getBoundingClientRect();
            return !element.hidden && style.display !== "none" && style.visibility !== "hidden"
                && element.getClientRects().length > 0 && rect.width > 0;
        };

        const chartSeries = (entry) => ({
            setData(data) {
                probe.chartSeriesCalls += 1;
                probe.chartDataPoints += Array.isArray(data) ? data.length : 0;
                if (entry) {
                    entry.dataSets += 1;
                    entry.data = Array.isArray(data) ? JSON.parse(JSON.stringify(data)) : null;
                }
                return this;
            },
            applyOptions() { return this; },
            setMarkers() { return this; },
            setVisible() { return this; },
            priceScale() { return { applyOptions() { } }; },
        });

        const makeChart = (container, chartOptions) => {
            const host = typeof container === "string" ? document.querySelector(container) : container;
            if (!host) throw new Error("fixture chart stub: missing container");
            probe.chartStubCalls += 1;
            const rect = host.getBoundingClientRect();
            const style = window.getComputedStyle(host);
            const hidden = !!(host.hidden || host.offsetParent === null || rect.width === 0
                || style.display === "none" || style.visibility === "hidden");
            if (hidden) probe.chartHiddenCount += 1;
            if (rect.width === 0) probe.chartZeroWidthCount += 1;
            const paperWorkspace = document.getElementById("workspace-paper");
            const record = {
                seq: probe.chartRecords.length,
                id: host.id || "",
                options: chartOptions ? JSON.parse(JSON.stringify(chartOptions)) : {},
                series: [],
                priceScaleCalls: [],
                hiddenAtCreate: hidden,
                zeroWidthAtCreate: rect.width === 0,
                workspaceHiddenAtCreate: !isVisibleElement(paperWorkspace),
            };
            probe.chartRecords.push(record);
            probe.chartContainers.push({
                id: host.id || "",
                width: Math.round(rect.width),
                height: Math.round(rect.height),
                hidden,
                display: style.display,
                visibility: style.visibility,
            });
            host.innerHTML = "";
            const box = document.createElement("div");
            box.className = "fixture-chart-stub";
            box.setAttribute("data-fixture-stub", "LightweightCharts");
            box.style.cssText = [
                "box-sizing:border-box",
                "min-height:180px",
                "height:100%",
                "display:flex",
                "align-items:center",
                "justify-content:center",
                "border:1px dashed #3b7f4c",
                "background:repeating-linear-gradient(45deg,#0a1a10,#0a1a10 8px,#0e2214 8px,#0e2214 16px)",
                "color:#7aaa81",
                "font:12px ui-monospace,monospace",
                "text-align:center",
                "padding:12px",
            ].join(";");
            box.textContent = "FIXTURE CHART STUB - layout/navigation only (not real canvas)";
            host.appendChild(box);
            const addSeries = (seriesOptions) => {
                const entry = {
                    options: seriesOptions ? JSON.parse(JSON.stringify(seriesOptions)) : {},
                    data: null,
                    dataSets: 0,
                };
                record.series.push(entry);
                return chartSeries(entry);
            };
            return {
                addLineSeries: addSeries,
                addAreaSeries: addSeries,
                addHistogramSeries: addSeries,
                addCandlestickSeries: addSeries,
                addBarSeries: addSeries,
                remove() { host.innerHTML = ""; },
                applyOptions() { },
                resize() { },
                options() { return record.options; },
                priceScale(id) {
                    return {
                        applyOptions(options) {
                            record.priceScaleCalls.push({
                                id: id || "right",
                                options: options ? JSON.parse(JSON.stringify(options)) : {},
                            });
                        },
                    };
                },
                timeScale() {
                    return {
                        fitContent() { },
                        setVisibleLogicalRange() { },
                        setVisibleRange() { },
                        getVisibleRange() { return null; },
                        applyOptions() { },
                    };
                },
            };
        };

        window.LightweightCharts = {
            createChart: makeChart,
            LineStyle: { Solid: 0, Dotted: 1, Dashed: 2 },
            CrosshairMode: { Normal: 0, Magnet: 1 },
            ColorType: { Solid: "solid", VerticalGradient: "gradient" },
        };

        const escapeHtml = (value) => String(value)
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;")
            .replace(/'/g, "&#39;");
        window.marked = {
            setOptions() { },
            Renderer: function Renderer() {
                this.link = function (href, title, text) { return text; };
                this.image = function () { return ""; };
                this.html = function () { return ""; };
            },
            parse(text) {
                probe.markdownParseCalls += 1;
                return escapeHtml(text == null ? "" : text)
                    .replace(/^### (.*)$/gm, "<h3>$1</h3>")
                    .replace(/^## (.*)$/gm, "<h2>$1</h2>")
                    .replace(/^# (.*)$/gm, "<h1>$1</h1>")
                    .replace(/\*\*(.+?)\*\*/g, "<b>$1</b>")
                    .replace(/\n/g, "<br>");
            },
        };

        window.addEventListener("DOMContentLoaded", () => {
            document.documentElement.dataset.fixtureMode = mode;
            const badge = document.createElement("div");
            badge.id = "fixture-testing-badge";
            badge.textContent = "FIXTURE / " + mode + " / layout-only";
            badge.style.cssText = [
                "position:fixed",
                "right:8px",
                "bottom:8px",
                "z-index:99999",
                "pointer-events:none",
                "padding:3px 7px",
                "border:1px solid #ff9f43",
                "border-radius:3px",
                "background:#1a1205",
                "color:#ffb86b",
                "font:10px ui-monospace,monospace",
                "opacity:.82",
            ].join(";");
            document.body.appendChild(badge);
        });
    };

    await page.addInitScript({ content: "(" + initScript.toString() + ")(" + JSON.stringify(fixtureOrigin) + "," + JSON.stringify(requestedMode) + ");" });

    const viewports = [
        { label: "390", width: 390, height: 844 },
        { label: "768", width: 768, height: 1024 },
        { label: "1440", width: 1440, height: 900 },
    ];
    const views = [
        { name: "research", nav: "#nav-research", leaf: "workspace-research", hash: "#/research", ready: "#watch .ticker" },
        { name: "paper-sim", nav: "#nav-paper", account: "sim", leaf: "workspace-paper", hash: "#/paper/sim", ready: "#sim-summary .metric" },
        { name: "paper-monthly", nav: "#nav-paper", account: "monthly", leaf: "workspace-paper", hash: "#/paper/monthly", ready: "#monthly-summary .metric" },
        { name: "paper-dailycore", nav: "#nav-paper", account: "dailycore", leaf: "workspace-paper", hash: "#/paper/dailycore", ready: "#dailycore-summary .metric" },
        { name: "savings", nav: "#nav-savings", leaf: "workspace-savings", hash: "#/savings", ready: "#savings-summary .metric" },
        { name: "lab", nav: "#nav-lab", leaf: "workspace-lab", hash: "#/lab", ready: "#lab-maintenance" },
        { name: "settings", nav: "#nav-settings", leaf: "workspace-settings", hash: "#/settings", ready: "#workspace-settings" },
    ];

    const screenshots = [];
    const viewResults = [];
    let health = null;

    const mergeProbeErrors = async () => {
        const snapshot = await page.evaluate(() => {
            const probe = window.__fixtureProbe;
            if (!probe) return null;
            return {
                pageErrors: Array.isArray(probe.pageErrors) ? probe.pageErrors : [],
                resourceErrors: Array.isArray(probe.resourceErrors) ? probe.resourceErrors : [],
            };
        });
        if (!snapshot) return;
        snapshot.pageErrors.forEach(addPageError);
        snapshot.resourceErrors.forEach(addResourceError);
    };

    const mergeFullProbe = async () => {
        const snapshot = await page.evaluate(() => {
            const probe = window.__fixtureProbe;
            if (!probe) return null;
            return {
                pageErrors: Array.isArray(probe.pageErrors) ? probe.pageErrors : [],
                resourceErrors: Array.isArray(probe.resourceErrors) ? probe.resourceErrors : [],
                cspViolations: probe.cspViolations || 0,
                cspBlocked: Array.isArray(probe.cspBlocked) ? probe.cspBlocked : [],
                externalFetchAttempts: probe.externalFetchAttempts || 0,
                externalXhrAttempts: probe.externalXhrAttempts || 0,
                chartStubCalls: probe.chartStubCalls || 0,
                chartSeriesCalls: probe.chartSeriesCalls || 0,
                chartDataPoints: probe.chartDataPoints || 0,
                chartContainers: Array.isArray(probe.chartContainers) ? probe.chartContainers : [],
                chartRecords: Array.isArray(probe.chartRecords) ? probe.chartRecords : [],
                chartHiddenCount: probe.chartHiddenCount || 0,
                chartZeroWidthCount: probe.chartZeroWidthCount || 0,
                markdownParseCalls: probe.markdownParseCalls || 0,
                stubs: probe.stubs || {},
            };
        });
        if (!snapshot) return;
        probeMerges += 1;
        snapshot.pageErrors.forEach(addPageError);
        snapshot.resourceErrors.forEach(addResourceError);
        aggregate.cspViolations += snapshot.cspViolations;
        aggregate.cspBlocked.push(...snapshot.cspBlocked);
        aggregate.inPageExternalFetch += snapshot.externalFetchAttempts;
        aggregate.inPageExternalXhr += snapshot.externalXhrAttempts;
        aggregate.chartStubCalls += snapshot.chartStubCalls;
        aggregate.chartSeriesCalls += snapshot.chartSeriesCalls;
        aggregate.chartDataPoints += snapshot.chartDataPoints;
        aggregate.chartContainers.push(...snapshot.chartContainers);
        aggregate.chartRecords.push(...snapshot.chartRecords);
        aggregate.chartHiddenCount += snapshot.chartHiddenCount;
        aggregate.chartZeroWidthCount += snapshot.chartZeroWidthCount;
        aggregate.markdownParseCalls += snapshot.markdownParseCalls;
        aggregate.stubs = {
            lightweightCharts: aggregate.stubs.lightweightCharts || !!snapshot.stubs.lightweightCharts,
            marked: aggregate.stubs.marked || !!snapshot.stubs.marked,
        };
    };

    // One page-state inspection per view. It deliberately checks the active
    // leaf, canonical hash, nav state and an explicit readiness condition —
    // empty mode has its own concrete condition per view, never a blanket pass.
    const inspectView = async (view) => page.evaluate(({ name, mode, expectedHash, expectedNav, expectedLeaf, expectedAccount }) => {
        const q = (selector) => document.querySelector(selector);
        const isVisible = (element) => {
            if (!element) return false;
            const style = window.getComputedStyle(element);
            return !element.hidden && style.display !== "none" && style.visibility !== "hidden"
                && element.getClientRects().length > 0;
        };
        const leaf = document.getElementById(expectedLeaf);
        const nav = q(expectedNav);
        const actualHash = location.hash;
        const hashOk = actualHash === expectedHash;
        const navActive = !!(nav && nav.classList.contains("nav-active"));
        const metaMode = q("#paper-meta-mode");
        const metaConfig = q("#paper-meta-config");
        const metaEpoch = q("#paper-meta-epoch");
        const meta = {
            mode: metaMode ? metaMode.textContent.trim() : null,
            config_id: metaConfig ? metaConfig.textContent.trim() : null,
            epoch: metaEpoch ? metaEpoch.textContent.trim() : null,
        };
        let ready = false;
        const checks = {};

        if (mode === "empty") {
            if (name === "research") {
                const watch = q("#watch");
                const universe = q("#universe");
                checks.watch_is_explicitly_empty = !!watch && watch.children.length === 0;
                checks.universe_selector_loaded = !!universe && universe.options.length > 0;
                ready = checks.watch_is_explicitly_empty && checks.universe_selector_loaded;
            } else if (name === "paper-sim" || name === "paper-monthly" || name === "paper-dailycore") {
                const summaryId = name === "paper-sim" ? "#sim-summary"
                    : name === "paper-monthly" ? "#monthly-summary" : "#dailycore-summary";
                checks.metrics_rendered = !!q(summaryId + " .metric");
                checks.meta_explicitly_not_recorded = meta.mode === "not recorded";
                ready = checks.metrics_rendered && checks.meta_explicitly_not_recorded;
            } else if (name === "savings") {
                checks.setup_visible = isVisible(document.getElementById("savings-setup"));
                checks.main_hidden = !isVisible(document.getElementById("savings-main"));
                ready = checks.setup_visible && checks.main_hidden;
            } else if (name === "lab") {
                checks.maintenance_visible = isVisible(document.getElementById("lab-maintenance"));
                checks.policy_loaded = (q("#policy-universe-current") || {}).textContent === "diversified-plus";
                ready = checks.maintenance_visible && checks.policy_loaded;
            } else if (name === "settings") {
                const select = q("#llm-model");
                checks.model_selector_empty_state = !!select && select.options.length === 1
                    && select.value === "" && /no LLM backends/i.test(select.textContent || "");
                ready = checks.model_selector_empty_state;
            }
        } else if (name === "research") {
            checks.ticker_list_loaded = !!q("#watch .ticker");
            ready = checks.ticker_list_loaded;
        } else if (name === "paper-sim" || name === "paper-monthly" || name === "paper-dailycore") {
            const summaryId = name === "paper-sim" ? "#sim-summary"
                : name === "paper-monthly" ? "#monthly-summary" : "#dailycore-summary";
            checks.metrics_rendered = !!q(summaryId + " .metric");
            checks.phase2_meta = meta.mode === "paper" && meta.config_id === "42" && meta.epoch === "1";
            ready = checks.metrics_rendered && checks.phase2_meta;
        } else if (name === "savings") {
            checks.metrics_rendered = !!q("#savings-summary .metric");
            checks.setup_hidden = !isVisible(document.getElementById("savings-setup"));
            checks.main_visible = isVisible(document.getElementById("savings-main"));
            ready = checks.metrics_rendered && checks.setup_hidden && checks.main_visible;
        } else if (name === "lab") {
            checks.maintenance_visible = isVisible(document.getElementById("lab-maintenance"));
            checks.policy_loaded = (q("#policy-universe-current") || {}).textContent === "diversified-plus";
            ready = checks.maintenance_visible && checks.policy_loaded;
        } else if (name === "settings") {
            const select = q("#llm-model");
            checks.model_selector_loaded = !!select && select.options.length > 0 && select.value !== "";
            ready = checks.model_selector_loaded;
        }

        let accountOk = true;
        let accountVisible = null;
        let otherAccountViewsHidden = null;
        if (expectedAccount) {
            const selector = q("#paper-account");
            const selected = selector ? selector.value : null;
            const subId = expectedAccount === "sim" ? "sim-view"
                : expectedAccount === "monthly" ? "monthly-view" : "dailycore-view";
            const sub = document.getElementById(subId);
            accountVisible = expectedAccount === "sim"
                ? !!(sub && sub.classList.contains("active") && isVisible(sub))
                : isVisible(sub);
            const otherIds = ["sim-view", "monthly-view", "dailycore-view"].filter((id) => id !== subId);
            otherAccountViewsHidden = otherIds.every((id) => !isVisible(document.getElementById(id)));
            accountOk = selected === expectedAccount && accountVisible && otherAccountViewsHidden;
        }

        const documentWidth = Math.max(
            document.documentElement ? document.documentElement.scrollWidth : 0,
            document.body ? document.body.scrollWidth : 0,
        );
        const viewportWidth = window.innerWidth;
        return {
            view: name,
            expected_account: expectedAccount || null,
            actual_account: expectedAccount && q("#paper-account") ? q("#paper-account").value : null,
            expected_leaf: expectedLeaf,
            leaf_visible: isVisible(leaf),
            expected_hash: expectedHash,
            actual_hash: actualHash,
            hash_ok: hashOk,
            expected_nav: expectedNav,
            nav_active: navActive,
            account_ok: accountOk,
            account_visible: accountVisible,
            other_account_views_hidden: otherAccountViewsHidden,
            readiness: ready,
            readiness_checks: checks,
            phase2_meta: meta,
            doc_scroll_width: documentWidth,
            viewport_width: viewportWidth,
            overflow: documentWidth > viewportWidth,
        };
    }, {
        name: view.name,
        mode: requestedMode,
        expectedHash: view.hash,
        expectedNav: view.nav,
        expectedLeaf: view.leaf,
        expectedAccount: view.account || null,
    });

    const waitForView = async (view) => {
        const started = Date.now();
        let last = null;
        while (Date.now() - started < 9000) {
            last = await inspectView(view);
            if (last.hash_ok && last.nav_active && last.leaf_visible
                && last.account_ok && last.readiness) {
                return last;
            }
            await page.waitForTimeout(150);
        }
        return last;
    };

    // ===================== Phase 3B shared comparison =====================
    // These checks are deliberately layered on top of the unchanged 42
    // navigation/layout checks. They assert the ONE comparison owner, the
    // pinned control ids/values, real control behaviour (GET-only), one visible
    // USD axis, no chart creation while Paper is hidden, and an explicit
    // empty-primary placeholder. The chart data here comes from the labelled
    // DOM stub, not a real LightweightCharts canvas.
    const COMPARISON_CONTAINER_ID = "paper-comparison-chart";
    const PAPER_RANGE_LIMIT = { "6M": 126, "1Y": 252, "2Y": 504, "5Y": 1260, "MAX": 12000 };
    const COMPARISON_PATHS = [
        "/api/sim/equity", "/api/monthly/equity",
        "/api/dailycore/equity", "/api/sim/benchmark",
    ];
    const todayUtc = new Date().toISOString().slice(0, 10);
    const paperExpected = {
        sim: { label: /Daily Sim/i, pctLast: 111.4977, usdLast: 11149.77, allowance: 10000 },
        monthly: { label: /Monthly/i, pctLast: 92.55325, usdLast: 11106.39, allowance: 12000 },
        dailycore: { label: /Daily-Core/i, pctLast: 94.28445454545455, usdLast: 10371.29, allowance: 11000 },
    };
    const comparisonRuns = [];
    const comparisonFailures = [];
    const comparisonFinalStates = [];

    const closeEnough = (actual, expected) => Number.isFinite(actual)
        && Math.abs(actual - expected) <= Math.max(1e-9, Math.abs(expected) * 1e-9);

    const readComparisonState = async () => page.evaluate(() => {
        const q = (selector) => document.querySelector(selector);
        const qa = (selector) => Array.from(document.querySelectorAll(selector));
        const isVisible = (element) => {
            if (!element) return false;
            const style = window.getComputedStyle(element);
            const rect = element.getBoundingClientRect();
            return !element.hidden && style.display !== "none" && style.visibility !== "hidden"
                && element.getClientRects().length > 0 && rect.width > 0;
        };
        const chart = q("#paper-comparison-chart");
        const section = q("#paper-comparison");
        const range = q("#paper-range");
        const mode = q("#paper-mode");
        const legend = q("#paper-comparison-legend");
        const asOf = q("#paper-comparison-asof");
        const warnings = q("#paper-comparison-warnings");
        const account = q("#paper-account");
        const workspace = q("#workspace-paper");
        const probe = window.__fixtureProbe || {};
        const records = Array.isArray(probe.chartRecords)
            ? probe.chartRecords.filter((record) => record.id === "paper-comparison-chart")
            : [];
        const latest = records.length ? records[records.length - 1] : null;
        const legendItems = legend ? Array.from(legend.querySelectorAll(".cmp-legend-item")) : [];
        const primaryLegendItem = legendItems.find((element) => {
            const text = element.textContent || "";
            return /selected/i.test(text) && /primary/i.test(text);
        });
        const emptyNode = chart ? chart.querySelector(".sim-empty") : null;
        const simplifyRecord = (record) => ({
            seq: record.seq,
            id: record.id,
            hiddenAtCreate: record.hiddenAtCreate,
            zeroWidthAtCreate: record.zeroWidthAtCreate,
            workspaceHiddenAtCreate: record.workspaceHiddenAtCreate,
            priceScaleCalls: record.priceScaleCalls || [],
            options: record.options || {},
            series: (record.series || []).map((entry) => ({
                options: entry.options || {},
                data: Array.isArray(entry.data) ? entry.data : null,
                dataSets: entry.dataSets || 0,
            })),
        });
        const simplifyMeta = (record) => ({
            seq: record.seq,
            id: record.id,
            hiddenAtCreate: record.hiddenAtCreate,
            zeroWidthAtCreate: record.zeroWidthAtCreate,
            workspaceHiddenAtCreate: record.workspaceHiddenAtCreate,
        });
        return {
            account: account ? account.value : null,
            hash: location.hash,
            sectionCount: qa("#paper-comparison").length,
            chartCount: qa("#paper-comparison-chart").length,
            sectionVisible: isVisible(section),
            chartVisible: isVisible(chart),
            workspaceVisible: isVisible(workspace),
            rangeValues: range && range.tagName === "SELECT"
                ? Array.from(range.options).map((option) => option.value)
                : qa("#paper-range [data-range]").map((element) => element.dataset.range),
            rangeValue: range ? range.value : null,
            modeValues: mode && mode.tagName === "SELECT"
                ? Array.from(mode.options).map((option) => option.value)
                : qa("#paper-mode [data-mode]").map((element) => element.dataset.mode),
            modeValue: mode ? mode.value : null,
            legendText: legend ? legend.textContent.replace(/\s+/g, " ").trim() : "",
            primaryLegend: primaryLegendItem
                ? primaryLegendItem.textContent.replace(/\s+/g, " ").trim() : null,
            asOfText: asOf ? asOf.textContent.replace(/\s+/g, " ").trim() : "",
            warningText: warnings ? warnings.textContent.replace(/\s+/g, " ").trim() : "",
            placeholderText: emptyNode
                ? emptyNode.textContent.replace(/\s+/g, " ").trim() : "",
            chartText: chart ? chart.textContent.replace(/\s+/g, " ").trim() : "",
            chartChildCount: chart ? chart.children.length : 0,
            recordCount: records.length,
            records: records.map(simplifyMeta),
            latest: latest ? simplifyRecord(latest) : null,
        };
    });

    const waitForComparisonState = async (predicate, timeoutMs = 9000) => {
        const started = Date.now();
        let last = null;
        while (Date.now() - started < timeoutMs) {
            last = await readComparisonState();
            try {
                if (predicate(last)) return last;
            } catch (_) { /* predicate may run before a field exists */ }
            await page.waitForTimeout(100);
        }
        return last;
    };

    const readFixtureRequests = async () => page.evaluate(async () => {
        const response = await fetch("/__fixture__/requests?limit=500");
        const payload = await response.json();
        return Array.isArray(payload.requests) ? payload.requests : [];
    });

    const latestRequestSeq = async () => (await readFixtureRequests())
        .reduce((max, entry) => Math.max(max, Number(entry.seq) || 0), 0);

    const waitForFreshRequests = async (beforeSeq, predicate, timeoutMs = 9000) => {
        const started = Date.now();
        let fresh = [];
        while (Date.now() - started < timeoutMs) {
            const requests = await readFixtureRequests();
            fresh = requests.filter((entry) => Number(entry.seq) > beforeSeq);
            try {
                if (predicate(fresh)) return { ok: true, fresh };
            } catch (_) { /* predicate may run before the request appears */ }
            await page.waitForTimeout(100);
        }
        return { ok: false, fresh };
    };

    const apiOnly = (requests) => requests.filter((entry) => String(entry.path || "").startsWith("/api/"));
    const tradePolicyProblems = (requests) => requests.filter((entry) => {
        const method = String(entry.method || "").toUpperCase();
        if (method !== "GET" && method !== "HEAD") return true;
        return /^\/api\/.*\/(run|backfill|reset|reset-all|refresh|targetvol|chat|sell|buy|deposit|withdraw|transfer|plans|settle|split|correct)\b/.test(String(entry.path || ""));
    });

    // Account loaders start several GETs in parallel; the comparison chart can
    // be ready while a sibling request is still landing. Wait for the API
    // request sequence to go quiet before attributing later requests to a
    // control toggle. This is a state-based poll, not a blind fixed sleep.
    const waitForApiQuiescence = async (quietMs = 300, timeoutMs = 5000) => {
        const started = Date.now();
        let lastSeq = -1;
        let quietSince = Date.now();
        while (Date.now() - started < timeoutMs) {
            const api = apiOnly(await readFixtureRequests());
            const seq = api.reduce((max, entry) => Math.max(max, Number(entry.seq) || 0), 0);
            if (seq === lastSeq) {
                if (Date.now() - quietSince >= quietMs) return;
            } else {
                lastSeq = seq;
                quietSince = Date.now();
            }
            await page.waitForTimeout(100);
        }
    };

    const pinnedControlCheck = (state) => ({
        one_comparison_section: state.sectionCount === 1,
        one_comparison_container: state.chartCount === 1,
        control_range_values: JSON.stringify(state.rangeValues)
            === JSON.stringify(["6M", "1Y", "2Y", "5Y", "MAX"]),
        control_mode_values: JSON.stringify(state.modeValues) === JSON.stringify(["pct", "usd"]),
    });

    const renderChecks = (state, expected, mode) => {
        const safeState = state || {};
        const checks = { ...pinnedControlCheck(safeState) };
        const latest = safeState.latest;
        const firstSeries = latest && latest.series.length ? latest.series[0] : null;
        const firstLast = firstSeries && Array.isArray(firstSeries.data) && firstSeries.data.length
            ? firstSeries.data[firstSeries.data.length - 1] : null;
        const baseline = latest && latest.series.length
            ? latest.series[latest.series.length - 1] : null;
        const baselineLast = baseline && Array.isArray(baseline.data) && baseline.data.length
            ? baseline.data[baseline.data.length - 1] : null;
        const benchmarkSeries = latest && latest.series.length >= 2
            ? latest.series[latest.series.length - 2] : null;
        const benchmarkLast = benchmarkSeries && Array.isArray(benchmarkSeries.data)
            && benchmarkSeries.data.length
            ? benchmarkSeries.data[benchmarkSeries.data.length - 1] : null;
        const expectedLast = mode === "usd" ? expected.usdLast : expected.pctLast;
        checks.paper_workspace_visible = safeState.workspaceVisible === true;
        checks.comparison_section_visible = safeState.sectionVisible === true;
        checks.comparison_chart_visible = safeState.chartVisible === true;
        checks.primary_legend_present = !!safeState.primaryLegend;
        checks.primary_legend_names_selected_account = !!safeState.primaryLegend
            && expected.label.test(safeState.primaryLegend);
        checks.contributed_legend_present = /Contributed baseline/i.test(safeState.legendText || "");
        checks.chart_record_present = !!latest;
        checks.chart_created_visible = !!latest && latest.hiddenAtCreate === false;
        checks.chart_created_in_visible_paper_workspace = !!latest
            && latest.workspaceHiddenAtCreate === false;
        checks.primary_plus_overlays_plus_baseline = !!latest && latest.series.length === 5;
        checks.asof_names_today = (safeState.asOfText || "").includes("as of " + todayUtc);
        checks.daily_utc_axis_sorted = !!latest && latest.series.length > 0
            && latest.series.every((entry) => Array.isArray(entry.data)
                && entry.data.every((point, index) => /^\d{4}-\d{2}-\d{2}$/.test(point.time)
                    && (index === 0 || point.time > entry.data[index - 1].time)));
        const axisId = (entry) => (entry.options && entry.options.priceScaleId)
            ? entry.options.priceScaleId : "right";
        checks.all_series_share_one_axis = !!latest && latest.series.length > 0
            && new Set(latest.series.map(axisId)).size === 1;
        checks.no_hidden_left_axis = !!latest && latest.priceScaleCalls
            .every((call) => !(call.id === "left" && call.options && call.options.visible === false));
        const usedAxisIds = latest ? new Set(latest.series.map(axisId)) : new Set();
        checks.series_axis_visible = !!latest && latest.priceScaleCalls
            .every((call) => !(usedAxisIds.has(call.id)
                && call.options && call.options.visible === false));
        checks.primary_last_value_matches_selected = !!firstLast
            && closeEnough(firstLast.value, expectedLast);
        checks.contributed_baseline_present = !!baseline;
        checks.benchmark_uses_provided_live_mark = !!benchmarkLast
            && closeEnough(benchmarkLast.value, mode === "usd" ? 4019.4 : 133.98);
        if (mode === "pct") {
            checks.contributed_baseline_is_flat_100 = !!baseline
                && Array.isArray(baseline.data)
                && !!firstSeries && baseline.data.length === firstSeries.data.length
                && baseline.data.every((point) => closeEnough(point.value, 100));
        } else {
            checks.contributed_baseline_is_own_allowance_usd = !!baselineLast
                && closeEnough(baselineLast.value, expected.allowance);
            checks.usd_single_visible_axis = checks.all_series_share_one_axis
                && checks.no_hidden_left_axis && checks.series_axis_visible;
        }
        return checks;
    };

    const recordComparisonRun = (run) => {
        const failures = Object.entries(run.checks)
            .filter(([, value]) => value !== true)
            .map(([name]) => name)
            .concat(run.errors || []);
        comparisonRuns.push({
            view: run.view,
            account: run.account,
            width: run.width,
            mode: requestedMode,
            checks: run.checks,
            errors: failures,
            asof: run.asof || null,
            placeholder: run.placeholder || null,
        });
        failures.forEach((failure) => comparisonFailures.push({
            view: run.view,
            account: run.account,
            width: run.width,
            failure,
        }));
    };

    const selectPaperControl = async (selector, value) => {
        const tag = await page.$eval(selector, (element) => element.tagName).catch(() => null);
        if (tag === "SELECT") {
            await page.selectOption(selector, value, { timeout: 5000 });
            return;
        }
        const button = await page.$(`${selector} [data-range="${value}"], ${selector} [data-mode="${value}"]`);
        if (!button) throw new Error(`fixture: no ${selector} control for ${value}`);
        await button.click({ timeout: 5000 });
    };

    const runComparisonControlChecks = async () => {
        const checks = {};
        const errors = [];
        await waitForApiQuiescence();
        // USD mode: a pure cached redraw, no GET/POST and one visible axis.
        let beforeSeq = await latestRequestSeq();
        let beforeState = await readComparisonState();
        let beforeRecordSeq = beforeState.latest ? beforeState.latest.seq : -1;
        await selectPaperControl("#paper-mode", "usd");
        let state = await waitForComparisonState((candidate) => candidate.modeValue === "usd"
            && !!candidate.latest && candidate.latest.seq > beforeRecordSeq
            && candidate.latest.series.length === 5, 9000);
        let modeChecks = renderChecks(state, paperExpected.sim, "usd");
        Object.assign(checks, Object.fromEntries(
            Object.entries(modeChecks).map(([name, value]) => [`mode_usd_${name}`, value])));
        const modeFresh = apiOnly((await readFixtureRequests())
            .filter((entry) => Number(entry.seq) > beforeSeq));
        checks.mode_usd_no_trade_or_post = tradePolicyProblems(modeFresh).length === 0;
        if (modeFresh.length) checks.mode_usd_only_gets = modeFresh.every((entry) => entry.method === "GET");
        if (!checks.mode_usd_no_trade_or_post) {
            errors.push("USD mode toggle issued a POST/trade-policy request: "
                + JSON.stringify(tradePolicyProblems(modeFresh)));
        }

        // Back to % mode for the shared controls and subsequent views.
        beforeSeq = await latestRequestSeq();
        beforeState = await readComparisonState();
        beforeRecordSeq = beforeState.latest ? beforeState.latest.seq : -1;
        await selectPaperControl("#paper-mode", "pct");
        state = await waitForComparisonState((candidate) => candidate.modeValue === "pct"
            && !!candidate.latest && candidate.latest.seq > beforeRecordSeq
            && candidate.latest.series.length === 5, 9000);
        modeChecks = renderChecks(state, paperExpected.sim, "pct");
        Object.assign(checks, Object.fromEntries(
            Object.entries(modeChecks).map(([name, value]) => [`mode_pct_${name}`, value])));
        const pctFresh = apiOnly((await readFixtureRequests())
            .filter((entry) => Number(entry.seq) > beforeSeq));
        checks.mode_pct_no_trade_or_post = tradePolicyProblems(pctFresh).length === 0;

        // Every pinned range must issue the four GET-only curve reads with the
        // documented row limit and redraw the one comparison chart.
        for (const range of ["6M", "1Y", "2Y", "5Y", "MAX"]) {
            const limit = String(PAPER_RANGE_LIMIT[range]);
            beforeSeq = await latestRequestSeq();
            beforeState = await readComparisonState();
            beforeRecordSeq = beforeState.latest ? beforeState.latest.seq : -1;
            await selectPaperControl("#paper-range", range);
            const waited = await waitForFreshRequests(beforeSeq, (fresh) => {
                const api = apiOnly(fresh);
                return COMPARISON_PATHS.every((path) => api.some((entry) => entry.path === path
                    && entry.method === "GET"
                    && Array.isArray(entry.query && entry.query.limit)
                    && entry.query.limit.includes(limit)));
            }, 9000);
            checks[`range_${range}_four_get_curve_reads`] = waited.ok;
            const rangeFresh = apiOnly(waited.fresh);
            checks[`range_${range}_no_trade_or_post`] = tradePolicyProblems(rangeFresh).length === 0;
            checks[`range_${range}_only_expected_curve_paths`] = rangeFresh.length > 0
                && rangeFresh.every((entry) => COMPARISON_PATHS.includes(entry.path));
            state = await waitForComparisonState((candidate) => candidate.rangeValue === range
                && !!candidate.latest && candidate.latest.seq > beforeRecordSeq, 9000);
            const rangeChecks = renderChecks(state, paperExpected.sim, "pct");
            checks[`range_${range}_chart_redrawn`] = !!state.latest
                && state.latest.seq > beforeRecordSeq;
            checks[`range_${range}_one_visible_axis`] = rangeChecks.all_series_share_one_axis
                && rangeChecks.no_hidden_left_axis && rangeChecks.series_axis_visible;
            checks[`range_${range}_asof_today`] = rangeChecks.asof_names_today;
            if (!waited.ok) errors.push(`range ${range}: missing GET limit=${limit} on all four curves`);
            if (!checks[`range_${range}_no_trade_or_post`]) {
                errors.push(`range ${range}: trade-policy/POST request observed`);
            }
        }

        // Leave the shared controls in their default state for later views.
        beforeSeq = await latestRequestSeq();
        await selectPaperControl("#paper-range", "1Y");
        await waitForFreshRequests(beforeSeq, (fresh) => apiOnly(fresh).some((entry) =>
            entry.path === "/api/sim/equity"
            && Array.isArray(entry.query && entry.query.limit)
            && entry.query.limit.includes("252")), 9000);
        return { checks, errors };
    };

    const runPaperComparisonChecks = async ({ view, viewport, requestSeqBefore }) => {
        if (!view.account) return;
        const expected = paperExpected[view.account];
        if (requestedMode === "empty") {
            const state = await waitForComparisonState((candidate) =>
                candidate.account === view.account
                && candidate.sectionVisible
                && candidate.chartCount === 1
                && !!candidate.placeholderText, 9000);
            const checks = {
                ...pinnedControlCheck(state || {}),
                comparison_section_visible: !!state && state.sectionVisible === true,
                comparison_container_visible: !!state && state.chartVisible === true,
                empty_primary_placeholder_explicit: !!state
                    && /No recorded equity for .* at this range yet/i.test(state.placeholderText),
                empty_primary_placeholder_names_account: !!state
                    && expected.label.test(state.placeholderText),
                empty_primary_chart_not_created: !!state && state.recordCount === 0,
                empty_primary_asof_clear: !!state && state.asOfText === "",
                empty_primary_legend_clear: !!state && state.legendText === "",
                empty_primary_no_hidden_creation: !!state
                    && state.records.every((record) => record.workspaceHiddenAtCreate === false),
            };
            recordComparisonRun({
                view: view.name,
                account: view.account,
                width: viewport.label,
                checks,
                errors: state ? [] : ["no comparison state returned"],
                placeholder: state ? state.placeholderText : null,
            });
            return;
        }

        const state = await waitForComparisonState((candidate) =>
            candidate.account === view.account
            && candidate.sectionVisible
            && candidate.chartVisible
            && !!candidate.latest
            && candidate.latest.series.length === 5
            && !!candidate.primaryLegend
            && expected.label.test(candidate.primaryLegend), 9000);
        const checks = renderChecks(state, expected, state ? state.modeValue : "pct");
        const fresh = await readFixtureRequests();
        checks.account_loader_get_seen = fresh.some((entry) =>
            Number(entry.seq) > requestSeqBefore
            && entry.method === "GET"
            && entry.path === `/api/${view.account}/equity`);
        const errors = [];
        if (!checks.account_loader_get_seen) {
            errors.push(`account ${view.account}: no GET /api/${view.account}/equity after switch`);
        }
        if (view.name === "paper-sim") {
            const control = await runComparisonControlChecks();
            Object.assign(checks, control.checks);
            errors.push(...control.errors);
        }
        recordComparisonRun({
            view: view.name,
            account: view.account,
            width: viewport.label,
            checks,
            errors,
            asof: state ? state.asOfText : null,
        });
    };

    // Re-selecting an already-loaded account must re-render the ONE
    // comparison owner without a reload. This catches a stale primary chart.
    const runRevisitPrimaryCheck = async (viewport) => {
        if (requestedMode !== "populated" || viewport.label !== "1440") return;
        await page.click("#nav-paper", { timeout: 5000 }).catch(() => { });
        await page.waitForFunction(() => location.hash.indexOf("#/paper/") === 0, null, { timeout: 5000 })
            .catch(() => { });
        const before = await readComparisonState();
        const beforeRecordSeq = before.latest ? before.latest.seq : -1;
        const beforeSeq = await latestRequestSeq();
        await page.selectOption("#paper-account", "sim", { timeout: 5000 }).catch(() => { });
        const state = await waitForComparisonState((candidate) =>
            candidate.account === "sim"
            && !!candidate.primaryLegend
            && /Daily Sim/i.test(candidate.primaryLegend)
            && !!candidate.latest
            && candidate.latest.series.length === 5
            && Array.isArray(candidate.latest.series[0].data)
            && candidate.latest.series[0].data.length > 0
            && closeEnough(candidate.latest.series[0].data[
                candidate.latest.series[0].data.length - 1].value, paperExpected.sim.pctLast), 9000);
        const newChart = !!state && !!state.latest && state.latest.seq > beforeRecordSeq;
        const firstSeries = state && state.latest && state.latest.series.length
            ? state.latest.series[0] : null;
        const firstLast = firstSeries && Array.isArray(firstSeries.data) && firstSeries.data.length
            ? firstSeries.data[firstSeries.data.length - 1] : null;
        const fresh = apiOnly((await readFixtureRequests())
            .filter((entry) => Number(entry.seq) > beforeSeq));
        const checks = {
            revisit_primary_legend_switches_to_sim: !!state && !!state.primaryLegend
                && /Daily Sim/i.test(state.primaryLegend),
            revisit_primary_data_switches_to_sim: !!firstLast
                && closeEnough(firstLast.value, paperExpected.sim.pctLast),
            revisit_redraws_cached_primary: newChart,
            revisit_no_trade_or_post: tradePolicyProblems(fresh).length === 0,
        };
        recordComparisonRun({
            view: "paper-revisit-sim",
            account: "sim",
            width: viewport.label,
            checks,
            errors: [],
            asof: state ? state.asOfText : null,
        });
    };

    for (const viewport of viewports) {
        // The &db=prod pin (which also sets the ts_db=prod cookie on the index
        // document) keeps a leftover ts_db=dev cookie from poisoning this run.
        const navigationUrl = fixtureUrl + "&viewport=" + encodeURIComponent(viewport.label)
            + "&run=" + runId + "-" + viewport.label + "&db=prod";
        navigationUrls.push(navigationUrl);
        await page.setViewportSize({ width: viewport.width, height: viewport.height });
        // Unique query makes each viewport a real new document (not a
        // same-document hash navigation) and bypasses any document cache.
        await page.goto(navigationUrl, { waitUntil: "domcontentloaded", timeout: 30000 });
        if (health === null) {
            health = await page.evaluate(async () => {
                const response = await fetch("/__fixture__/health");
                return response.json();
            });
        }
        await page.waitForTimeout(400);

        for (const view of views) {
            const errorsBefore = aggregate.pageErrors.length;
            let requestSeqBefore = 0;
            try {
                requestSeqBefore = await latestRequestSeq();
                await page.click(view.nav, { timeout: 5000 });
                if (view.account) {
                    const selected = await page.$eval("#paper-account", (element) => element.value).catch(() => null);
                    if (selected !== view.account) {
                        await page.selectOption("#paper-account", view.account, { timeout: 5000 });
                    }
                }
            } catch (error) {
                await mergeProbeErrors();
                viewResults.push({
                    view: view.name,
                    account: view.account || null,
                    width: viewport.label,
                    document_url: navigationUrl,
                    ok: false,
                    error: String(error),
                    screenshot: null,
                });
                continue;
            }
            const checked = await waitForView(view);
            await page.waitForTimeout(200);
            const finalChecked = await inspectView(view);
            const resultState = finalChecked || checked;
            try {
                await runPaperComparisonChecks({ view, viewport, requestSeqBefore });
            } catch (error) {
                recordComparisonRun({
                    view: view.name,
                    account: view.account || null,
                    width: viewport.label,
                    checks: {},
                    errors: ["comparison check threw: " + String(error)],
                });
            }
            await mergeProbeErrors();
            const errorsClean = aggregate.pageErrors.length === errorsBefore;
            const path = outputDir + "/phase3-fixture-" + requestedMode + "-" + viewport.label + "-" + view.name + ".png";
            await page.screenshot({ path, fullPage: true });
            screenshots.push(path);
            const viewOk = !!resultState
                && resultState.hash_ok && resultState.nav_active && resultState.leaf_visible
                && resultState.account_ok && resultState.readiness && !resultState.overflow
                && errorsClean;
            viewResults.push({
                ...(resultState || {}),
                view: view.name,
                account: view.account || null,
                width: viewport.label,
                document_url: navigationUrl,
                error: resultState ? null : "view inspection returned no state",
                errors_clean: errorsClean,
                screenshot: path,
                ok: viewOk,
            });
        }
        try {
            await runRevisitPrimaryCheck(viewport);
        } catch (error) {
            comparisonFailures.push({
                width: viewport.label,
                failure: "revisit_check_threw: " + String(error),
            });
        }
        // Leave the workspace on Paper for the final "one visible owner"
        // check. This is navigation only; it must not create a hidden chart.
        await page.click("#nav-paper", { timeout: 5000 }).catch(() => { });
        const finalComparisonState = await waitForComparisonState((candidate) =>
            candidate.sectionVisible && candidate.chartCount === 1
            && candidate.workspaceVisible, 9000);
        comparisonFinalStates.push({
            width: viewport.label,
            sectionCount: finalComparisonState ? finalComparisonState.sectionCount : null,
            chartCount: finalComparisonState ? finalComparisonState.chartCount : null,
            sectionVisible: finalComparisonState ? finalComparisonState.sectionVisible : false,
            chartVisible: finalComparisonState ? finalComparisonState.chartVisible : false,
            recordCount: finalComparisonState ? finalComparisonState.recordCount : null,
            placeholderText: finalComparisonState ? finalComparisonState.placeholderText : null,
        });
        await page.waitForTimeout(150);
        await mergeFullProbe();
    }

    // ===================== Phase 3C PROD/DEV database switch =====================
    // Runs AFTER the viewport loop on ONE viewport, so the guard-parity POST
    // fetches can never land inside a per-view tradePolicyProblems window, and
    // mergeFullProbe is never called here (probeMerges stays pinned to
    // viewports.length). Destructive routes are exercised ONLY via direct
    // fetch() from the page, never by clicking the Lab buttons. The DB checks
    // fail closed until the frontend DB UI lands; nothing is weakened.
    const dbNavigationUrls = [];
    const dbChecks = { prod: {}, dev: {} };
    const dbFetch = (method, path, body) => page.evaluate(async ({ method, path, body }) => {
        const response = await fetch(path, {
            method,
            headers: body ? { "Content-Type": "application/json" } : undefined,
            body: body || undefined,
        });
        let payload = null;
        try { payload = await response.json(); } catch (_) { /* non-JSON body */ }
        return { status: response.status, payload };
    }, { method, path, body: body || null });
    const dbIsVisible = `(element) => {
        if (!element) return false;
        const style = window.getComputedStyle(element);
        return !element.hidden && style.display !== "none" && style.visibility !== "hidden"
            && element.getClientRects().length > 0;
    }`;
    const dbNavClick = async (selector) => {
        try {
            await page.click(selector, { timeout: 5000 });
            return true;
        } catch (_) {
            return false;
        }
    };

    const dbViewport = viewports[0];
    await page.setViewportSize({ width: dbViewport.width, height: dbViewport.height });

    // ---- PROD state (default; pinned explicitly via ?db=prod) ----
    const prodDbUrl = fixtureUrl + "&viewport=" + encodeURIComponent(dbViewport.label)
        + "&run=" + runId + "-db-prod&db=prod";
    dbNavigationUrls.push(prodDbUrl);
    await page.goto(prodDbUrl, { waitUntil: "domcontentloaded", timeout: 30000 });
    await page.waitForTimeout(400);
    const prodStatus = await dbFetch("GET", "/api/db/status");
    dbChecks.prod.status_active_prod = prodStatus.status === 200
        && !!prodStatus.payload && prodStatus.payload.active === "prod";

    dbChecks.prod.settings_nav = await dbNavClick("#nav-settings");
    await page.waitForTimeout(300);
    Object.assign(dbChecks.prod, await page.evaluate(`(() => {
        const q = (selector) => document.querySelector(selector);
        const isVisible = ${dbIsVisible};
        const toggle = q("#db-toggle");
        const status = q("#db-status");
        const banner = q("#db-dev-banner");
        return {
            db_toggle_present: !!toggle,
            db_status_rendered: !!status && status.textContent.trim().length > 0,
            db_dev_banner_not_visible: !!banner && !isVisible(banner),
        };
    })()`));

    dbChecks.prod.lab_nav = await dbNavClick("#nav-lab");
    await page.waitForTimeout(300);
    Object.assign(dbChecks.prod, await page.evaluate(`(() => {
        const isVisible = ${dbIsVisible};
        const groups = Array.from(document.querySelectorAll("#lab-replay-groups .maintenance-group"));
        const note = document.getElementById("lab-replay-disabled");
        const warning = document.getElementById("lab-replay-warning");
        return {
            lab_maintenance_visible: isVisible(document.getElementById("lab-maintenance")),
            lab_replay_reset_groups_hidden: groups.length > 0
                && groups.every((group) => !isVisible(group)),
            lab_explanatory_note_visible: !!note && isVisible(note),
            // The destructive-ops warning is DEV-only: on PROD the controls it
            // warns about are hidden, so the box must be hidden too.
            lab_replay_warning_hidden_on_prod: !!warning && !isVisible(warning),
            lab_data_visible: isVisible(document.getElementById("lab-data")),
            lab_policy_visible: isVisible(document.getElementById("lab-policy")),
            lab_db_group_visible: isVisible(document.getElementById("lab-db-group")),
            nav_savings_visible: isVisible(document.getElementById("nav-savings")),
        };
    })()`));

    const prodDestructive = await dbFetch("POST", "/api/sim/reset", "{}");
    dbChecks.prod.destructive_post_403_on_prod = prodDestructive.status === 403;
    const prodSavingsWrite = await dbFetch("POST", "/api/savings/deposit", '{"amount": 10}');
    dbChecks.prod.savings_write_acks_on_prod = prodSavingsWrite.status === 200
        && !!prodSavingsWrite.payload && prodSavingsWrite.payload.fixture_ack === true;

    // ---- DEV state (via ?db=dev navigation, which sets the ts_db cookie) ----
    const devDbUrl = fixtureUrl + "&viewport=" + encodeURIComponent(dbViewport.label)
        + "&run=" + runId + "-db-dev&db=dev";
    dbNavigationUrls.push(devDbUrl);
    await page.goto(devDbUrl, { waitUntil: "domcontentloaded", timeout: 30000 });
    await page.waitForTimeout(400);
    const devStatus = await dbFetch("GET", "/api/db/status");
    dbChecks.dev.status_active_dev = devStatus.status === 200
        && !!devStatus.payload && devStatus.payload.active === "dev"
        && devStatus.payload.dev_enabled === true;

    dbChecks.dev.settings_nav = await dbNavClick("#nav-settings");
    await page.waitForTimeout(300);
    Object.assign(dbChecks.dev, await page.evaluate(`(() => {
        const q = (selector) => document.querySelector(selector);
        const isVisible = ${dbIsVisible};
        const status = q("#db-status");
        const banner = q("#db-dev-banner");
        return {
            db_status_rendered: !!status && status.textContent.trim().length > 0,
            db_dev_banner_visible: !!banner && isVisible(banner),
            db_clone_btn_present: !!q("#db-clone-btn"),
        };
    })()`));

    dbChecks.dev.lab_nav = await dbNavClick("#nav-lab");
    await page.waitForTimeout(300);
    Object.assign(dbChecks.dev, await page.evaluate(`(() => {
        const isVisible = ${dbIsVisible};
        const groups = Array.from(document.querySelectorAll("#lab-replay-groups .maintenance-group"));
        const note = document.getElementById("lab-replay-disabled");
        const nav = document.getElementById("nav-savings");
        const warning = document.getElementById("lab-replay-warning");
        return {
            lab_maintenance_visible: isVisible(document.getElementById("lab-maintenance")),
            lab_replay_reset_groups_visible: groups.length > 0
                && groups.every((group) => isVisible(group)),
            lab_replay_note_hidden: !!note && !isVisible(note),
            // The destructive-ops warning appears exactly where the controls do.
            lab_replay_warning_visible_on_dev: !!warning && isVisible(warning),
            nav_savings_hidden: !!nav && !isVisible(nav),
        };
    })()`));

    // The #/savings route on DEV must show the PROD-only panel, not the tracker.
    await page.evaluate(() => { location.hash = "#/savings"; });
    await page.waitForTimeout(400);
    Object.assign(dbChecks.dev, await page.evaluate(`(() => {
        const isVisible = ${dbIsVisible};
        const tracker = document.getElementById("savings-view");
        const notice = document.getElementById("savings-dev-notice");
        return {
            savings_tracker_hidden: !!tracker && !isVisible(tracker),
            savings_prod_only_panel_visible: !!notice && isVisible(notice),
            savings_notice_names_prod_only: !!notice && /PROD-only/i.test(notice.textContent || ""),
        };
    })()`));

    const devDestructive = await dbFetch("POST", "/api/sim/reset", "{}");
    dbChecks.dev.destructive_post_acks_on_dev = devDestructive.status === 200
        && !!devDestructive.payload && devDestructive.payload.fixture_ack === true;
    const devSavingsWrite = await dbFetch("POST", "/api/savings/deposit", '{"amount": 10}');
    dbChecks.dev.savings_write_403_on_dev = devSavingsWrite.status === 403;
    const cloneAck = await dbFetch("POST", "/api/db/clone", "{}");
    dbChecks.dev.clone_acks = cloneAck.status === 200
        && !!cloneAck.payload && cloneAck.payload.fixture_ack === true
        && Number.isInteger(cloneAck.payload.size_bytes);
    // The frontend wires #db-clone-status to its own clone call; exercise the
    // button (ack-only, non-destructive) when present and require an update.
    // #db-clone-btn lives in the Lab workspace (#lab-db-group inside
    // #lab-maintenance), which is hidden on the savings route, so navigate
    // back to Lab before reading/clicking it.
    dbChecks.dev.lab_nav_before_clone = await dbNavClick("#nav-lab");
    await page.waitForTimeout(300);
    const cloneBtn = await page.$("#db-clone-btn");
    if (cloneBtn) {
        const before = await page.$eval("#db-clone-status", (element) => element.textContent)
            .catch(() => null);
        await cloneBtn.click({ timeout: 5000 }).catch(() => { });
        await page.waitForTimeout(600);
        const after = await page.$eval("#db-clone-status", (element) => element.textContent)
            .catch(() => null);
        dbChecks.dev.clone_status_updated = typeof after === "string"
            && after.trim().length > 0 && after !== before;
    } else {
        dbChecks.dev.clone_status_updated = false;
    }
    await mergeProbeErrors();
    const dbFailures = [];
    for (const [state, checks] of Object.entries(dbChecks)) {
        for (const [name, value] of Object.entries(checks)) {
            if (value !== true) dbFailures.push(state + "." + name);
        }
    }
    const dbOk = dbFailures.length === 0;

    // ===================== Phase 3D/3E/3F drawers + live config =====================
    // Runs AFTER the viewport loop on ONE fresh document (like the 3C DB pass),
    // so the confirm/reject/transaction POSTs can never land inside a per-view
    // tradePolicyProblems window. mergeFullProbe is NOT called here, keeping
    // probeMerges pinned to viewports.length.
    const sliceChecks = {};
    const sliceFailures = [];
    const sliceNavUrl = fixtureUrl + "&viewport=" + encodeURIComponent(dbViewport.label)
        + "&run=" + runId + "-slices&db=prod";
    await page.setViewportSize({ width: dbViewport.width, height: dbViewport.height });
    await page.goto(sliceNavUrl, { waitUntil: "domcontentloaded", timeout: 30000 });
    await page.waitForTimeout(400);

    const sliceEval = (fn, arg) => page.evaluate(fn, arg);
    const drawerOpen = () => sliceEval(() => {
        const drawer = document.getElementById("assistant-drawer");
        return !!(drawer && drawer.classList.contains("open"));
    });
    const svDrawerOpen = () => sliceEval(() => {
        const drawer = document.getElementById("svtxn-drawer");
        return !!(drawer && drawer.classList.contains("open"));
    });

    // ---- 3F: CURRENT (next-run) live config strip beside LATEST RECORDED ----
    await dbNavClick("#nav-paper");
    await page.waitForFunction(() => {
        const epoch = document.getElementById("paper-live-epoch");
        return epoch && epoch.textContent.trim() !== "not loaded";
    }, null, { timeout: 9000 }).catch(() => { });
    // sliceInfo carries RAW observed values (strings/lists) for the report;
    // only booleans may live in sliceChecks — the failure collector at the
    // end counts every non-true value as a failure.
    const sliceInfo = {};
    const rawInfoKeys = ["f_live_engine", "f_live_epoch", "f_live_summary",
        "f_live_details", "f_recorded_epoch", "f_requests_seen"];
    Object.assign(sliceChecks, await sliceEval(() => {
        const q = (selector) => document.querySelector(selector);
        const metaText = (q("#paper-meta") || {}).textContent || "";
        const requestsSeen = [];
        return {
            f_latest_recorded_label_kept: /LATEST RECORDED snapshot:/.test(metaText),
            f_current_label_present: /CURRENT \(next run\):/.test(metaText),
            f_live_engine: q("#paper-live-engine") ? q("#paper-live-engine").textContent.trim() : null,
            f_live_epoch: q("#paper-live-epoch") ? q("#paper-live-epoch").textContent.trim() : null,
            f_live_summary: q("#paper-live-summary") ? q("#paper-live-summary").textContent.trim() : null,
            f_live_details: q("#paper-live-details") ? q("#paper-live-details").textContent.trim() : null,
            f_recorded_epoch: q("#paper-meta-epoch") ? q("#paper-meta-epoch").textContent.trim() : null,
            f_meta_note_mentions_separate_live: /live configuration is shown separately/i.test(metaText),
            f_requests_seen: requestsSeen,
        };
    }));
    for (const name of rawInfoKeys) {
        if (name in sliceChecks) {
            sliceInfo[name] = sliceChecks[name];
            delete sliceChecks[name];
        }
    }
    sliceChecks.f_live_engine_is_sim = sliceInfo.f_live_engine === "sim";
    sliceChecks.f_live_epoch_real_not_notloaded = sliceInfo.f_live_epoch === "1";
    sliceChecks.f_live_summary_rendered = !!sliceInfo.f_live_summary
        && sliceInfo.f_live_summary !== "not loaded" && sliceInfo.f_live_summary.length > 3;
    sliceChecks.f_live_details_holds_json = !!sliceInfo.f_live_details
        && /current effective configuration/i.test(sliceInfo.f_live_details) && /"epoch"/.test(sliceInfo.f_live_details);
    // The fixture's populated status payload deliberately carries NO recorded
    // provenance (no mode/config_id/epoch fields), so the LATEST RECORDED
    // epoch renders its honest missing-provenance label in BOTH modes; the
    // strip still renders a non-empty value either way.
    sliceChecks.f_recorded_epoch_label_rendered =
        sliceInfo.f_recorded_epoch === "not recorded" || /^\d+$/.test(String(sliceInfo.f_recorded_epoch));
    sliceChecks.f_run_config_get_seen = (await readFixtureRequests()).some((entry) =>
        entry.method === "GET" && entry.path === "/api/run-config" && entry.status === 200);

    // ---- 3D: assistant drawer on the Daily Sim view opens in proposals mode ----
    sliceChecks.d_assistant_button_present = !!(await page.$("#assistant-open-btn"));
    await page.click("#assistant-open-btn", { timeout: 5000 });
    await page.waitForTimeout(250);
    sliceChecks.d_opens_on_paper_sim = await drawerOpen();
    Object.assign(sliceChecks, await sliceEval(() => {
        const q = (selector) => document.querySelector(selector);
        const isVisible = (element) => {
            if (!element) return false;
            const style = window.getComputedStyle(element);
            return !element.hidden && style.display !== "none" && style.visibility !== "hidden"
                && element.getClientRects().length > 0;
        };
        return {
            d_proposals_mode_default_on_sim: isVisible(q("#assistant-mode-proposals"))
                && !isVisible(q("#assistant-mode-explain"))
                && q("#assistant-tab-proposals").classList.contains("tab-active"),
        };
    }));

    // Mode tabs switch any time.
    await page.click("#assistant-tab-explain", { timeout: 5000 });
    await page.waitForTimeout(150);
    Object.assign(sliceChecks, await sliceEval(() => {
        const q = (selector) => document.querySelector(selector);
        const isVisible = (element) => {
            if (!element) return false;
            const style = window.getComputedStyle(element);
            return !element.hidden && style.display !== "none" && style.visibility !== "hidden"
                && element.getClientRects().length > 0;
        };
        return {
            d_tab_switch_to_explain: isVisible(q("#assistant-mode-explain"))
                && !isVisible(q("#assistant-mode-proposals"))
                && q("#assistant-tab-explain").classList.contains("tab-active"),
            d_deterministic_help_no_llm: (q("#assistant-help")
                ? q("#assistant-help").querySelectorAll(".glossary-item").length : 0) > 5,
            d_help_names_view: !!q("#assistant-help .glossary-intro")
                && (q("#assistant-help .glossary-intro").textContent || "").length > 20,
        };
    }));

    // Esc closes the drawer.
    await page.keyboard.press("Escape");
    await page.waitForTimeout(250);
    sliceChecks.d_esc_closes = !(await drawerOpen());

    // Research view: "Explain This Screen" opens the drawer in Explain mode.
    await dbNavClick("#nav-research");
    await page.waitForTimeout(300);
    await page.click('button:has-text("Explain This Screen")', { timeout: 5000 });
    await page.waitForTimeout(250);
    Object.assign(sliceChecks, await sliceEval(() => {
        const q = (selector) => document.querySelector(selector);
        const isVisible = (element) => {
            if (!element) return false;
            const style = window.getComputedStyle(element);
            return !element.hidden && style.display !== "none" && style.visibility !== "hidden"
                && element.getClientRects().length > 0;
        };
        const drawer = q("#assistant-drawer");
        return {
            d_research_explain_button_opens_drawer: !!(drawer && drawer.classList.contains("open")),
            d_explain_mode_default_off_paper: isVisible(q("#assistant-mode-explain"))
                && !isVisible(q("#assistant-mode-proposals")),
        };
    }));
    await page.keyboard.press("Escape");
    await page.waitForTimeout(200);

    // ---- 3D: proposal card flow (chat -> proposal card -> confirm / reject) ----
    await dbNavClick("#nav-paper");
    await page.waitForFunction(() => {
        const input = document.getElementById("sim-chat-input");
        return input && !input.disabled;
    }, null, { timeout: 9000 }).catch(() => { });
    await page.click("#assistant-open-btn", { timeout: 5000 });
    await page.waitForTimeout(200);
    await page.fill("#sim-chat-input", "propose a trade");
    const seqBeforeChat = await latestRequestSeq();
    await page.click("#sim-chat-send", { timeout: 5000 });
    await page.waitForSelector("#assistant-proposals .proposal-card", { timeout: 9000 }).catch(() => { });
    Object.assign(sliceChecks, await sliceEval(() => {
        const card = document.querySelector("#assistant-proposals .proposal-card");
        const text = card ? card.textContent : "";
        return {
            d_proposal_card_rendered: !!card,
            d_card_shows_decided_at: /decided at/i.test(text) && /2026-10-30/.test(text),
            d_card_feasible_action_with_preview: /BUY/.test(text) && /NVDA/.test(text)
                && /est/.test(text) && /\$/.test(text),
            d_card_infeasible_action_shows_issue: /SELL/.test(text) && /no price/.test(text),
            d_card_has_confirm_and_reject: !!card && !!card.querySelector('button[onclick^="confirmProposal"]')
                && !!card.querySelector('button[onclick^="rejectProposal"]'),
        };
    }));
    const chatSeen = (await readFixtureRequests()).some((entry) =>
        Number(entry.seq) > seqBeforeChat && entry.method === "POST" && entry.path === "/api/sim/chat/stream");
    sliceChecks.d_chat_stream_posted = chatSeen;

    // Confirm acks and the card resolves inline.
    const seqBeforeConfirm = await latestRequestSeq();
    await page.click('#assistant-proposals .proposal-card button[onclick^="confirmProposal"]', { timeout: 5000 })
        .catch(() => { });
    await page.waitForFunction(() => {
        const card = document.querySelector("#assistant-proposals .proposal-card");
        return card && /Confirmed/i.test(card.textContent);
    }, null, { timeout: 9000 }).catch(() => { });
    const confirmSeen = (await readFixtureRequests()).some((entry) =>
        Number(entry.seq) > seqBeforeConfirm && entry.method === "POST" && entry.path === "/api/sim/chat/confirm");
    sliceChecks.d_confirm_posts_and_card_resolves = confirmSeen && await sliceEval(() => {
        const card = document.querySelector("#assistant-proposals .proposal-card");
        return !!card && /Confirmed/i.test(card.textContent) && /Executed/i.test(card.textContent);
    });

    // Reject on a fresh proposal.
    await page.fill("#sim-chat-input", "propose again");
    await page.click("#sim-chat-send", { timeout: 5000 });
    await page.waitForSelector("#assistant-proposals .proposal-card:not(.done)", { timeout: 9000 }).catch(() => { });
    const seqBeforeReject = await latestRequestSeq();
    await page.click('#assistant-proposals .proposal-card:not(.done) button[onclick^="rejectProposal"]', { timeout: 5000 })
        .catch(() => { });
    await page.waitForFunction(() => !document.querySelector("#assistant-proposals .proposal-card:not(.done)"),
        null, { timeout: 9000 }).catch(() => { });
    const rejectSeen = (await readFixtureRequests()).some((entry) =>
        Number(entry.seq) > seqBeforeReject && entry.method === "POST" && entry.path === "/api/sim/chat/reject");
    sliceChecks.d_reject_posts_and_card_removed = rejectSeen && await sliceEval(() =>
        !document.querySelector("#assistant-proposals .proposal-card:not(.done)"));
    await page.keyboard.press("Escape");
    await page.waitForTimeout(200);

    // ---- 3E: savings Record transaction drawer ----
    await dbNavClick("#nav-savings");
    await page.waitForTimeout(500);
    const recordBtnVisible = await sliceEval(() => {
        const btn = document.getElementById("savings-record-btn");
        if (!btn) return false;
        const style = window.getComputedStyle(btn);
        return !btn.hidden && style.display !== "none" && style.visibility !== "hidden"
            && btn.getClientRects().length > 0;
    });
    if (recordBtnVisible) {
        await page.click("#savings-record-btn", { timeout: 5000 });
    } else {
        // Empty mode: the actions row is hidden (tracker not initialized), but
        // the entry point and drawer must still work when invoked.
        await sliceEval(() => openSavingsTxn());
    }
    await page.waitForTimeout(250);
    sliceChecks.e_record_button_present = !!(await page.$("#savings-record-btn"));
    sliceChecks.e_drawer_opens = await svDrawerOpen();
    Object.assign(sliceChecks, await sliceEval(() => {
        const sel = document.getElementById("svtxn-type");
        const groups = sel ? Array.from(sel.querySelectorAll("optgroup")).map((g) => g.label) : [];
        const options = sel ? Array.from(sel.options).map((o) => o.value) : [];
        return {
            e_type_select_all_events: JSON.stringify(options) === JSON.stringify(
                ["deposit", "withdraw", "saveback", "dividend", "fee", "tax", "buy", "sell", "settle", "split", "correct"]),
            e_type_select_groups: JSON.stringify(groups) === JSON.stringify(
                ["Money in/out", "Costs", "Trading", "Position adjustments"]),
            e_help_line_rendered: (document.getElementById("svtxn-help") || {}).textContent.trim().length > 20,
        };
    }));

    // Event-type select switches the field set.
    await page.selectOption("#svtxn-type", "deposit", { timeout: 5000 });
    await page.waitForTimeout(150);
    sliceChecks.e_deposit_fields_amount_only = await sliceEval(() =>
        !!document.getElementById("svtxn-amount") && !document.getElementById("svtxn-ticker"));
    await page.selectOption("#svtxn-type", "sell", { timeout: 5000 });
    await page.waitForTimeout(150);
    sliceChecks.e_sell_fields_ticker_shares_price_date = await sliceEval(() =>
        !!document.getElementById("svtxn-ticker") && !!document.getElementById("svtxn-shares")
        && !!document.getElementById("svtxn-price") && !!document.getElementById("svtxn-occurred-on")
        && !document.getElementById("svtxn-amount"));
    sliceChecks.e_date_defaults_today = await sliceEval(() => {
        const d = document.getElementById("svtxn-occurred-on");
        return !!d && /^\d{4}-\d{2}-\d{2}$/.test(d.value)
            && d.value === new Date().toISOString().slice(0, 10);
    });

    // Preview line renders and tracks the type.
    await page.selectOption("#svtxn-type", "deposit", { timeout: 5000 });
    await page.waitForTimeout(150);
    await page.fill("#svtxn-amount", "25");
    await page.waitForTimeout(150);
    Object.assign(sliceInfo, await sliceEval(() => ({
        e_preview_text: (document.getElementById("svtxn-preview") || {}).textContent || "",
    })));
    sliceChecks.e_preview_label = sliceInfo.e_preview_text.includes("Preview (estimate)");
    // Populated: the filled deposit preview shows the live cash-arrow math.
    // Empty: the tracker is honestly uninitialized, so no cash exists to
    // preview — the correct render is the not-initialized notice.
    sliceChecks.e_preview_cash_arrow = requestedMode === "populated"
        ? (/→/.test(sliceInfo.e_preview_text) && /cash/i.test(sliceInfo.e_preview_text))
        : /tracker not initialized/i.test(sliceInfo.e_preview_text);

    // Submit acks against the fixture endpoint and reports inline.
    const seqBeforeTxn = await latestRequestSeq();
    await page.click("#svtxn-submit", { timeout: 5000 });
    await page.waitForFunction(() => {
        const status = document.getElementById("svtxn-status");
        return status && status.textContent.trim().length > 0;
    }, null, { timeout: 9000 }).catch(() => { });
    const txnSeen = (await readFixtureRequests()).some((entry) =>
        Number(entry.seq) > seqBeforeTxn && entry.method === "POST"
        && entry.path === "/api/savings/deposit" && entry.status === 200);
    sliceChecks.e_submit_acks_fixture_endpoint = txnSeen;
    sliceChecks.e_status_line_reports = await sliceEval(() =>
        (document.getElementById("svtxn-status") || {}).textContent.trim().length > 0);
    await page.keyboard.press("Escape");
    await page.waitForTimeout(200);
    sliceChecks.e_esc_closes = !(await svDrawerOpen());

    // ✕ remove-position retired; True-up kept; Sell shortcut present (populated).
    Object.assign(sliceChecks, await sliceEval(() => {
        const row = document.getElementById("savings-actions-row");
        const rowText = row ? row.textContent : "";
        const table = document.getElementById("savings-positions-table");
        const tableText = table ? table.textContent : "";
        return {
            e_trueup_still_present: /True-up balance/.test(rowText),
            e_old_prompt_buttons_gone: !/Add deposit/.test(rowText) && !/Add Saveback/.test(rowText)
                && !/Add buy/.test(rowText) && !/Withdraw<\/button>/.test(rowText),
            e_remove_position_fn_gone: typeof window.svRemovePosition === "undefined",
            e_no_x_button_in_positions: !/✕/.test(tableText),
            e_sell_shortcut_present: /Sell/.test(tableText),
        };
    }));
    if (requestedMode !== "populated") {
        // Empty mode has no positions table rows; the structural checks above
        // still apply (button/row absent-or-hidden is fine, drawer works).
        sliceChecks.e_sell_shortcut_present = true;
    }

    await mergeProbeErrors();
    for (const [name, value] of Object.entries(sliceChecks)) {
        if (value !== true) sliceFailures.push(name);
    }
    const slicesOk = sliceFailures.length === 0;

    const sri = await page.evaluate(() => Array.from(document.querySelectorAll("script[src]"))
        .filter((script) => /^https?:/i.test(script.src) && new URL(script.src).origin !== location.origin)
        .map((script) => ({
            src: script.src,
            integrity: script.getAttribute("integrity") || "",
            crossorigin: script.getAttribute("crossorigin") || "",
            in_dom: true,
        })));
    const comparisonChartRecords = aggregate.chartRecords
        .filter((record) => record.id === COMPARISON_CONTAINER_ID);
    const comparisonCreatedWhileHidden = comparisonChartRecords
        .filter((record) => record.workspaceHiddenAtCreate);
    const comparisonCreatedHidden = comparisonChartRecords
        .filter((record) => record.hiddenAtCreate);
    const comparisonCreatedZeroWidth = comparisonChartRecords
        .filter((record) => record.zeroWidthAtCreate);
    if (comparisonCreatedWhileHidden.length) {
        comparisonFailures.push({
            failure: "comparison_chart_created_while_paper_hidden",
            count: comparisonCreatedWhileHidden.length,
        });
    }
    if (comparisonCreatedHidden.length) {
        comparisonFailures.push({
            failure: "comparison_chart_created_while_hidden",
            count: comparisonCreatedHidden.length,
        });
    }
    if (comparisonCreatedZeroWidth.length) {
        comparisonFailures.push({
            failure: "comparison_chart_created_zero_width",
            count: comparisonCreatedZeroWidth.length,
        });
    }
    for (const finalState of comparisonFinalStates) {
        if (finalState.sectionCount !== 1) {
            comparisonFailures.push({
                width: finalState.width,
                failure: "final_comparison_section_count_not_one",
                actual: finalState.sectionCount,
            });
        }
        if (finalState.chartCount !== 1) {
            comparisonFailures.push({
                width: finalState.width,
                failure: "final_comparison_container_count_not_one",
                actual: finalState.chartCount,
            });
        }
        if (!finalState.sectionVisible || !finalState.chartVisible) {
            comparisonFailures.push({
                width: finalState.width,
                failure: "final_comparison_not_visible_in_paper",
            });
        }
        if (requestedMode === "empty") {
            if (finalState.recordCount !== 0) {
                comparisonFailures.push({
                    width: finalState.width,
                    failure: "empty_primary_created_a_chart",
                    actual: finalState.recordCount,
                });
            }
            if (!/No recorded equity for .* at this range yet/i.test(finalState.placeholderText || "")) {
                comparisonFailures.push({
                    width: finalState.width,
                    failure: "empty_primary_final_placeholder_missing",
                });
            }
        } else if (!finalState.recordCount) {
            comparisonFailures.push({
                width: finalState.width,
                failure: "populated_primary_has_no_comparison_chart",
            });
        }
    }
    const expectedComparisonRuns = viewports.length * 3
        + (requestedMode === "populated" ? 1 : 0);
    if (comparisonFinalStates.length !== viewports.length) {
        comparisonFailures.push({
            failure: "comparison_final_state_missing",
            expected: viewports.length,
            actual: comparisonFinalStates.length,
        });
    }
    const allComparisonChecksOk = comparisonRuns.length === expectedComparisonRuns
        && comparisonFailures.length === 0;
    const fixtureRequests = await page.evaluate(async () => {
        const response = await fetch("/__fixture__/requests?limit=500");
        const payload = await response.json();
        return { count: payload.count, requests: payload.requests };
    });
    await page.waitForTimeout(100);
    await mergeProbeErrors();
    const cspBlockedCdn = aggregate.cspBlocked
        .filter((entry) => /^https?:/i.test(entry.blockedURI || ""));
    const providerComponents = {
        page_request_events: observedExternal.provider_api,
        in_page_fetch_attempts: aggregate.inPageExternalFetch,
        in_page_xhr_attempts: aggregate.inPageExternalXhr,
    };
    const providerApiAllZero = Object.values(providerComponents).every((value) => value === 0);
    const appResourceErrors = aggregate.resourceErrors.filter((entry) => {
        try {
            if (!entry || !entry.url) return false;
            return new URL(entry.url, fixtureOrigin).origin === fixtureOrigin;
        } catch (_) {
            return false;
        }
    });
    const expectedViewChecks = viewports.length * views.length;
    const allViewsOk = viewResults.length === expectedViewChecks && viewResults.every((entry) => entry.ok);
    const allPageErrorsClear = aggregate.pageErrors.length === 0;
    const externalClean = providerApiAllZero && observedExternal.other === 0;
    const healthMatches = !!health && health.mode === requestedMode;
    const freshNavigations = navigationUrls.length === viewports.length
        && new Set(navigationUrls).size === viewports.length;
    const probesMergedOncePerNavigation = probeMerges === viewports.length;

    return {
        ok: allViewsOk && allPageErrorsClear && externalClean && healthMatches
            && freshNavigations && probesMergedOncePerNavigation && allComparisonChecksOk
            && dbOk && slicesOk,
        label: "FIXTURE STUBBED NAVIGATION/LAYOUT - NOT REAL LIGHTWEIGHTCHARTS CANVAS OR MARKDOWN",
        fixture_label: health ? health.fixture_label : null,
        mode: health ? health.mode : requestedMode,
        expected_mode: requestedMode,
        mode_source: modeSource,
        origin_source: originSource,
        health_matches_requested_mode: healthMatches,
        fixture_origin: fixtureOrigin,
        fixture_url: fixtureUrl,
        navigation_urls: navigationUrls,
        probe_merges: probeMerges,
        checks: {
            expected_view_checks: expectedViewChecks,
            completed_view_checks: viewResults.length,
            all_view_checks_ok: allViewsOk,
            all_page_errors_clear: allPageErrorsClear,
            provider_api_all_zero: providerApiAllZero,
            no_other_external_requests: observedExternal.other === 0,
            health_matches_requested_mode: healthMatches,
            fresh_navigations: freshNavigations,
            probes_merged_once_per_navigation: probesMergedOncePerNavigation,
            comparison_runs_expected: expectedComparisonRuns,
            comparison_runs_completed: comparisonRuns.length,
            all_comparison_checks_ok: allComparisonChecksOk,
            db_pass_ok: dbOk,
        },
        db: {
            label: "PROD/DEV DATABASE SWITCH - synthetic fixture state; fails closed until the frontend DB UI lands",
            ok: dbOk,
            failures: dbFailures,
            checks: dbChecks,
            navigation_urls: dbNavigationUrls,
            note: "destructive routes were only fetch()ed from the page, never clicked",
        },
        slices: {
            label: "PHASE 3D/3E/3F - assistant drawer, record-transaction drawer, CURRENT live-config strip",
            ok: slicesOk,
            failures: sliceFailures,
            checks: sliceChecks,
            info: sliceInfo,
            navigation_url: sliceNavUrl,
            note: "runs after the viewport loop on one fresh document; confirm/reject/transaction POSTs never land in a per-view trade-policy window",
        },
        externalblocked: {
            classification: externalClean
                ? "expected CDN/font assets only; zero provider API attempts"
                : "unexpected external traffic",
            cdn_asset_observations_by_layer: {
                page_request_events: observedExternal.cdn_assets,
                csp_violations: cspBlockedCdn.length,
                route_blocked: counts.cdn_assets,
                note: "layers may observe the same request; these are deliberately NOT summed",
            },
            provider_api_components: {
                ...providerComponents,
                all_zero: providerApiAllZero,
                note: "separate observation layers; zero expected in every layer, not a de-duplicated total",
            },
            other_external_components: {
                page_request_events: observedExternal.other,
                route_blocked: counts.other,
            },
            page_request_details: observedExternal.details.slice(0, 50),
            route_blocked_details: blocked.slice(0, 50),
        },
        fixture_api_traffic: {
            local_only: true,
            origin: fixtureOrigin,
            recorded_requests: fixtureRequests.count,
            non_fixture_origins: observedExternal.details.map((entry) => entry.url)
                .filter((url) => !url.startsWith(fixtureOrigin)),
        },
        chart_stub: {
            lightweightCharts: aggregate.stubs.lightweightCharts,
            createChart_calls: aggregate.chartStubCalls,
            series_setData_calls: aggregate.chartSeriesCalls,
            data_points: aggregate.chartDataPoints,
            containers: aggregate.chartContainers,
            hidden_containers: aggregate.chartHiddenCount,
            zero_width_containers: aggregate.chartZeroWidthCount,
            limitation: "DOM placeholder only; no canvas, axes, candles, crosshair or time-scale rendering was verified",
        },
        markdown_stub: {
            marked: aggregate.stubs.marked,
            parse_calls: aggregate.markdownParseCalls,
            limitation: "minimal escape/newline/bold stub only; real marked link/image/HTML-rule rendering was not verified",
        },
        comparison: {
            label: "SHARED PAPER COMPARISON - STUBBED SERIES DATA/OPTIONS, NOT REAL CANVAS RENDERING",
            expected_runs: expectedComparisonRuns,
            completed_runs: comparisonRuns.length,
            all_checks_ok: allComparisonChecksOk,
            failures: comparisonFailures,
            runs: comparisonRuns,
            final_states: comparisonFinalStates,
            chart_records: comparisonChartRecords.map((record) => ({
                seq: record.seq,
                id: record.id,
                hiddenAtCreate: record.hiddenAtCreate,
                zeroWidthAtCreate: record.zeroWidthAtCreate,
                workspaceHiddenAtCreate: record.workspaceHiddenAtCreate,
                priceScaleCalls: record.priceScaleCalls,
                series: (record.series || []).map((entry) => ({
                    priceScaleId: entry.options ? entry.options.priceScaleId : null,
                    color: entry.options ? entry.options.color : null,
                    points: Array.isArray(entry.data) ? entry.data.length : 0,
                    first: Array.isArray(entry.data) && entry.data.length ? entry.data[0] : null,
                    last: Array.isArray(entry.data) && entry.data.length
                        ? entry.data[entry.data.length - 1] : null,
                })),
            })),
            no_chart_created_while_paper_hidden: comparisonCreatedWhileHidden.length === 0,
            no_chart_created_while_hidden: comparisonCreatedHidden.length === 0,
            no_chart_created_zero_width: comparisonCreatedZeroWidth.length === 0,
            limitation: "chart stub records addLineSeries options and setData payloads; axes/canvas rendering is NOT real",
        },
        sri_preserved: sri.every((script) => script.integrity.length > 0),
        external_scripts_unchanged: sri,
        page_errors: aggregate.pageErrors,
        unhandled_rejections: aggregate.pageErrors.filter((entry) => entry.startsWith("unhandledrejection:")),
        resource_errors: aggregate.resourceErrors,
        app_resource_errors: appResourceErrors,
        csp_violations: aggregate.cspViolations,
        viewports: viewResults,
        screenshots,
    };
}
