# UI and financial-strategy audit

Date: 2026-10-21. Reviewed checkout: `ba37afa`.

Status addendum (2026-10-21): Phase 1 corrections are **code-complete and
independently verified** on `feature/phase-1-financial-correctness`, with
acceptance still pending — see "Phase 1 implementation status" at the end of
this document. The findings below are unchanged and describe the reviewed
checkout `ba37afa`; their file/line references are intentionally left as
written.

## Bottom line

Trade Sentinel is useful as a research terminal and paper laboratory. Its best
direction is **fewer first-class strategies, clearer comparisons, and stronger
financial correctness**, not more indicators or more LLM discretion.

The quality/value/momentum family has a defensible economic rationale. The current
implementation and historical results are not strong enough to establish a reliable
investable edge. Several reproducible defects affect rankings, risk metrics, and
the relationship between backtests and forward paper portfolios.

Recommended product focus:

- Keep broad-market DCA as the indispensable control.
- Treat Monthly and Daily-Core as variants of one factor strategy, not independent
  sources of diversification. Prefer simplicity until corrected comparisons justify
  a more complicated deployment policy.
- Keep the technical strategy as an experiment, not the flagship recommendation.
- Keep AI explanations; put AI trading decisions and cash-out overlays in the lab.
- Keep real-account Savings explicitly separate from every simulation.

This is a software/methodology assessment, not personal investment advice or a
recommendation to buy the benchmark ticker.

## Scope and verification

- Reviewed the frontend, strategy/execution modules, fundamentals ingestion,
  research optimizer, Savings accounting, tests, README, and experiment report.
- Opened all five views in an isolated local app with an empty temporary database
  and scheduled jobs disabled. Inspected desktop at 1440px and mobile at 390px.
  The deployed service and its populated database were not available here.
- Existing suite: **523 passed in 56.51 seconds**.
- Separate synthetic invariant checks reproduced the issues listed below; they
  used pure functions and an in-memory database, not user portfolio data.
- Verified the SEC capital-expenditure convention against Apple's FY2025
  companyfacts, not just a hypothetical input.
- No strategy, app behavior, production data, or configuration was changed. This
  document is the review deliverable. No new historical performance run was done.

Passing software tests is useful evidence of stability, not proof that financial
definitions or mocked provider conventions are correct.

## 1. Strategy ratings

Subjective scores out of 10. **Design** assesses the financial idea; **current
trust** assesses this implementation/evidence, not expected returns or suitability.
Historical alpha confidence is low for every active strategy pending corrections.

| Approach | Design | Current trust | Assessment |
|---|---:|---:|---|
| Broad, low-cost equity DCA control | 9 | 6 | Strong simple baseline for long-horizon equity exposure; still has substantial market risk. Needs the same realistic pricing, currency, cash-flow and cost conventions as competitors. |
| Monthly quality/value/momentum | 7 | 4 | Sensible factor combination, liquidity filters and turnover-reducing hold band. Ten stocks, ROE-only quality and no sector constraints leave considerable concentration risk. FCF and price/share consistency need correction. |
| Daily-Core factor variant | 7 | 4 | Putting available cash to work without arbitrary month-end delay is reasonable. Added benefit is unproven after accounting for model differences, cash interest and execution costs. Live holdings can exceed the advertised target. |
| Technical Daily Sim | 4 | 2 | Trend/momentum ideas are plausible, but many inputs repeat the same price information and weights are hand-tuned. Bearish HOLD fallback and ineffective ATR stop undermine the stated rules. |
| LLM/hybrid trade decisions | 3 | 2 | No established incremental edge. Short, repeatedly examined windows, model instability and historical knowledge contamination weaken retrospective evidence. Better as a read-only explainer or shadow reviewer. |
| SMA deployment gate | 5 | 3 | Transparent trend filter, but can miss rebounds. This implementation's market proxy is not an equal-weight return index. Test independently rather than call it protection by default. |
| Gradient/all-cash overlays | 3 | 2 | Substantial whipsaw and turnover risk, many tuning choices, and fragile evidence across universes. Experimental only until independently validated. |
| Contribution-only volatility gate | 6 | 4 | A reasonable restraint on new risk, but it cannot deliver a portfolio volatility target because it never reduces existing exposure. Rename to describe what it actually controls. |

Savings is a ledger, not a trading strategy. It is useful, but withdrawals, sale
accounting and the mirror comparison need improvement before treating its reported
gain as comprehensive investment performance.

Quality and momentum literature supports investigating these concepts, not the
specific ten-stock formula, its parameter choices, or claims of future excess
returns. In particular, no-stop equity investing is not inherently unsound; the
problem is calling a rank-retention band a complete risk-management system.

## 2. UI redundancies and information architecture

| Current UI | Problem | Recommendation |
|---|---|---|
| Daily Sim, Monthly Sim and Daily-Core each repeat summary cards, comparisons, positions, logs and range controls | Too much navigation for the same portfolio workflow; nearly identical rendering and data-loading paths can drift | One **Paper Portfolios** workspace with portfolio selector and one comparison view; retain separate underlying accounts |
| Monthly and Daily-Core presented as peer strategies | Shared factor exposures can look like independent diversification | One factor-strategy family with named, versioned deployment variants; keep Monthly as an explicit control |
| Screener score, technical signal strength and portfolio ranking | Different objectives/horizons look interchangeable | Label them “technical discovery score”, “technical rule score”, and “factor portfolio rank”; display horizon and reason. Share raw indicator calculations, not unlike scoring formulas |
| “Ask local LLM”, persistent AI explanation, glossary modal, portfolio reasoning and manager chat | Explanations compete for screen space; only one chat can execute paper trades | Contextual assistant drawer with **Explain** and **paper proposal** modes; keep deterministic help available without an LLM. Require preview/confirmation before chat-generated trades |
| Update, Refresh, Load 10y, Sync SP500, per-ticker refresh, pull-to-refresh | Operator must understand internal data plumbing | One “Update data” action with clear scope/progress; put deep-history and membership maintenance in a Data/Lab panel. Refresh stale data and rescore together |
| Research screener's universe dropdown changes every sim | Browsing candidates unexpectedly changes portfolio policy | Separate research filter from portfolio universe; portfolio changes need explicit scope, confirmation and a new strategy version |
| Per-tab backfills plus Backfill All and resets mixed with normal actions | Global/destructive operations appear inside one portfolio; backfill replaces forward history | Move to **Lab / Maintenance**; create separate experiment runs rather than overwrite a forward record |
| Header model picker on every view | Expensive/advanced implementation choice is always prominent | Settings/assistant control; show an unobtrusive model badge when an AI answer is displayed |
| Savings true-up/deposit/buy/saveback/remove controls | Some are genuinely different accounting events, not redundant buttons | One “Record transaction” entry point with explicit event type, amount, date, fees and preview; keep reconciliation separate |

Proposed navigation:

```text
Research          watchlist, stock detail, screener, contextual explanation
Paper Portfolios  comparison, selected account, holdings, decisions, activity
Savings           real EUR ledger, plans, reconciliation, clearly labeled mirror
Lab               strategy variants, backtests, validation, data maintenance
Settings          model/provider, scheduling, currency, environment and safeguards
```

Keep the terminal aesthetic if desired; the main issue is hierarchy, not theme.
Show equity, properly defined return, drawdown and benchmark difference first;
collapse logs, detailed indicators and long strategy prose.

### Observed layout issues

- Daily-Core at 1440px produced a **1575px-wide document**. The unwrapped actions
  row pushed Backfill All and Reset All beyond the viewport
  (`static/index.html:1251-1255`, `1696-1718`).
- At 390px there was no document-wide horizontal overflow in the empty-data view,
  but chart legends/mode buttons wrapped awkwardly and a tall empty chart pushed
  useful content far down the page. Use responsive toolbars and compact empty states.
- Add persistent labels for strategy controls instead of relying on tooltips and
  the selected option. Replace internal field names with readable labels/units.
- Distinguish backfilled, forward-paper, stale and reconciled data visibly. Do not
  use identical styling for model changes, ordinary refreshes and destructive actions.
- Populated table/chart behavior still needs testing against a safe copy of real
  data; this audit did not claim to validate that unavailable deployment state.

## 3. Financial correctness findings

### P0-A: SEC capital expenditure increases FCF instead of reducing it

`app/edgar.py:172-206` preserves raw SEC values. The
`PaymentsToAcquirePropertyPlantAndEquipment` tag is a positive expenditure in the
verified Apple filing. Both `eligible_frame` and `quality_snapshot` add this value
to OCF, assuming negative capex (`app/monthly.py:374-393`, `449-458`).

Apple FY2025 SEC facts: OCF **$111.482bn**, capex **+$12.715bn**. Correct FCF is
**$98.767bn**; current arithmetic produces **$124.197bn**. That changes the value
ranking. Existing tests largely supply negative capex, including the EDGAR fixture.

Normalize provider/tag-specific signs at ingestion; test against recorded real
provider payloads. Do not blindly negate already-negative Yahoo capex. Missing
capex should not silently make OCF a supposedly conservative FCF estimate.

### P0-B: “Max DD” is not cash-flow-neutral drawdown

`app/optimize.py:277-316` and `static/index.html:3070-3084` calculate drawdown on
equity divided by cumulative contributions. This is not a time-weighted NAV.

Reproduction: invest 1000, grow to 1200, then deposit 1000 with no price move.
Equity is 2200, contributions 2000. The app reports **8.33% drawdown** despite
there being no investment loss after the peak.

Chain cash-flow-adjusted returns into a unitized NAV; calculate drawdown from that
NAV. Show actual profit and dated money-weighted XIRR separately. “vs DCA” is a
benchmark performance difference, not risk-adjusted alpha. Align comparison windows
and currency explicitly; dividing by contributions does not repair different dates.
Use one dollar axis for dollar comparisons, not a hidden independently scaled axis
(`static/index.html:4082-4093`).

### P0-C: Technical fallback can buy bearish names; ATR stop cannot fire

- `strength_for("HOLD", bullish, bearish)` returns the larger score
  (`app/analysis.py:318-324`). The relaxed fallback buys high-strength HOLDs without
  checking bullishness (`app/strategy.py:256-265`). A synthetic HOLD with bullish=0,
  bearish=65 and weekly trend down generated a **BUY**. Thus the fallback can also
  bypass the trend checks that prevented a strict BUY.
- ATR stop is `current_close - 2 * ATR` (`app/analysis.py:220-221`). The strategy then
  asks whether that same close is below the stop (`app/strategy.py:205-212`), which
  is impossible for nonnegative ATR. The separate percent stop still exists; this
  finding does not mean all exits are absent.

Make any fallback explicitly bullish and subject to the same eligibility/risk
rules, or remove it. If retaining an ATR stop, define prior-known, entry-frozen or
high-water-mark ratcheting semantics and a realistic gap/fill convention.

### P0-D: Daily-Core does not reliably hold the advertised ten picks

The picker retains held ranks 11-20, but deployment buys ranks 1-10 rather than the
hysteresis-adjusted `picks` (`app/daily_core.py:763-796`). In an in-memory replay,
ten valid retained positions plus a new monthly cash balance became **11 positions**
after one cycle, despite a target of ten and a ten-name `picks` list.

Choose one contract: exactly ten hysteresis-selected holdings, or a larger evolving
band portfolio. Enforce and display the chosen contract consistently in live runs,
replays and the Savings mirror. Equal-weight *buy targets* are also not hard weight
limits: neither monthly top-ups nor Daily-Core routinely trim appreciated winners.

### P0-E: Optimizer look-ahead and live/replay policy differences

- The optimizer builds frames for window month-ends, then `month_for` falls back to
  the first month-end when no prior frame exists (`app/optimize.py:2652-2676`,
  `2781-2797`). With a first ranking dated **2025-01-31**, a synthetic run bought on
  **2025-01-01**. This boundary case leaks future ranking information. The persisted
  Daily-Core backfill has a different, prior-month-frame implementation.
- Live Daily-Core recomputes rankings and releases out-of-band holdings daily;
  its stored-history replay uses monthly frames and month-end releases. Those are
  different policies even if some historical windows happened to give similar results.
- Strategies generally calculate using a day's completed close and also fill at
  that close. A decision made after seeing the final bar needs a later attainable
  execution price; same-close fills are not executable merely because they are paper.
- Live/backfill factor accounts are fee-free, while the research optimizer has a
  default one-way 10bp cost. Comparisons need an explicit shared execution model.

Use a single decision kernel and explicit observation/fill timestamps. Load the
last available pre-window ranking; never substitute a future one. Add golden
fixtures where daily versus monthly facts/ranks deliberately change.

### P1-A: The market gate uses average price levels, not equal-weight returns

`close.ffill().mean(axis=1)` drives the market SMA in live/replay overlays
(`app/daily_core.py:390`, `app/optimize.py:2873-2877`). A high nominal share price
gets more influence. Changing a stock's price units can change the signal without
changing its economic return.

Synthetic example: AAA 100→105, BBB 10→5 gives the proxy a 0% return. Rescaling
BBB prices by 100, with identical percentage returns, gives **-45%**. Neither is
the equal-weight return of -22.5%.

Use a specified investable benchmark or a properly chained, point-in-time return
index. Treat any results involving this gate as provisional until rerun.

### P1-B: Savings needs transactions, not removal plus true-up

Withdrawals are represented as reconciliation drift, while contributed capital
only sums positive INIT/TRANSFER events (`app/savings.py:752-754`, `829-836`). A
withdrawal can therefore look like an investment loss. Removing a sold holding
does not book proceeds or realized P&L; a later true-up is expected to repair cash
(`1066-1089`). Removing its mirror entries also changes the reconstructed mirror
history rather than recording a dated exit.

Add dated withdrawal, partial/full sale, dividend, fee, tax and correction events.
Preserve history and distinguish investment P&L, rewards, external flows and
reconciliation. Use actual execution quantities/prices for real holdings.

The mirror is also **not the Daily-Core strategy**: it equally splits each new
amount across stored picks (`175-211`), whereas Daily-Core tops up by rank and has
its own exits/overlays. Label it “equal-allocation picks mirror” or implement an
actual same-cash-flow strategy account. Do not present it as a full strategy test.

### P1-C: Price, currency and data provenance require stronger contracts

- Dividend/split-adjusted historical closes are useful for return signals but are
  also multiplied by as-of historical share counts for market cap. Those quantities
  need the same adjustment basis; raw/split/dividend-adjusted series are not
  interchangeable (`app/market.py:130-150`, `app/monthly.py:386-393`). Real-share
  ledgers also need explicit corporate actions/dividend accounting.
- Currency inferred from ticker suffix is incomplete: `.VI` is not in the monthly
  FX mapping; London quote pence and reporting-versus-trading currency need explicit
  metadata. Missing FX can currently fall back to treating a price as USD.
- Forward-filled or average-cost fallback prices need stale flags and age limits.
  A missing quote is not a valid executable price, and an indefinitely stale listing
  should not remain silently eligible.
- Fundamentals can be cached for 90 days. A filed-date filter prevents some future
  information use; it does not ensure the newest public filing has been fetched.
  Yahoo fallback uses restated facts and an assumed 45-day filing lag, not true
  point-in-time history.

## 4. Why the reported strategy evidence is not yet decisive

1. Current index membership and hand-curated universes create survivorship/selection
   bias; the README acknowledges this. Historical membership and delisting returns
   are needed for claims about an investable broad-universe edge.
2. Repeatedly choosing overlays/prompts using the same four “OOS” windows turns those
   windows into research data. Keep a genuinely untouched holdout and prospective
   paper record; record the number of configurations tried.
3. `residual_momentum` is a custom one-factor daily alpha/residual-vol score. The
   cited Blitz-Huij-Martens method uses a three-factor regression over a longer
   estimation window and standardized residual formation returns, explicitly
   excluding alpha. The custom method may be useful, but is not that replication.
4. The experiment report says residual has the highest Sharpe of six arms in all
   four windows, but its own table shows other arms ahead in several windows
   (`docs/llm-strategy-experiments.md:314-345`). Beating the raw control is not the
   same as winning every comparison. Remove unconditional “winner” labels.
5. Zero commissions do not imply zero spread, slippage, FX costs or tax drag. Cash
   earns zero in the paper comparison; that can exaggerate the cost of parking cash.
   Existing Sharpe calculations assume a zero risk-free rate; disclose that or use
   an explicit excess-return series.
6. Selecting the deterministic model's worst historical trades is a diagnostic LLM
   benchmark, not unbiased evidence of improvement. A model can also know historical
   outcomes from training despite point-in-time indicator inputs. Require prospective
   shadow decisions and compare every eligible decision, including model failures.

## 5. Improvement plan, in dependency order

### Phase 1 — Restore financial truth (P0)

Scope: capex sign, cash-flow-neutral metrics, bullish fallback/ATR semantics,
holdings contract, optimizer boundary look-ahead, and executable fill timing.

Acceptance criteria:

- Real SEC and Yahoo fixtures produce the same economically correct FCF convention.
- Deposit/withdrawal-only events cause no NAV return or drawdown.
- A bearish HOLD cannot enter through fallback; an intended stop is demonstrably
  triggerable and never loosens silently.
- A target-ten strategy cannot accidentally create an eleventh holding.
- Changing data filed after date D cannot change decisions on or before D.
- Live kernel and historical replay produce identical decisions on an identical
  ordered input stream, with fees and fill timing explicitly selected.

Do not select new “winning” parameters until this phase is complete.

### Phase 2 — Consolidate the portfolio model and accounting (P1)

- Share execution/ledger, valuation, cash-flow and metric primitives; keep strategy
  definitions/configuration separate. Start with Monthly/Daily-Core duplication.
- Give each run an immutable configuration/version, data provenance and mode:
  historical experiment versus forward paper. Do not overwrite a forward record.
- Fix market-index construction, instrument currencies/corporate actions and stale
  price handling. Route every decision path through the same hard risk checks.
- Repair Savings transactions and label/repair the mirror before judging real
  picks versus model performance.
- Keep the technical screener and factor ranking as distinct views of shared inputs;
  do not collapse different investment horizons into one opaque mega-score.

Acceptance: one definition of return/drawdown across backend/UI; reconciled ledgers;
no history rewritten by resets, configuration changes or a sold-position removal.

### Phase 3 — Simplify the UI (P1, partly parallel with Phase 2)

- Build the Research / Paper Portfolios / Savings / Lab navigation above.
- One portfolio comparator; one shared chart/table/activity implementation.
- Separate research universe browsing from strategy policy changes.
- Add clearly scoped Update Data; move deep-load/backfill/reset to the Lab.
- Put detailed controls and AI into drawers, retaining an explicit paper-trade
  confirmation step. Fix the desktop action-row overflow and mobile empty states.

Acceptance: no horizontal page overflow at 390/768/1440px with populated fixtures;
one obvious refresh action per scope; account mode/currency/configuration always
visible; all important controls labeled and keyboard accessible.

### Phase 4 — Revalidate, then prune strategies (P2)

Compare only a small predeclared set initially:

1. Broad-market DCA, with matching cash flows and currency.
2. Monthly factor strategy.
3. The same factor policy with immediate deployment of available cash.
4. Technical strategy as an experimental comparator.

Run broad point-in-time universes, random-N and simple equal-weight controls,
realistic spread/slippage/FX/cash-yield assumptions, multiple regimes, contribution
and fixed-capital cases, and parameter sensitivity. Report TWR, dated XIRR,
proper NAV drawdown, volatility, excess-return Sharpe, turnover, time in cash,
concentration and benchmark differences with uncertainty estimates.

Only then test a small number of overlays against the frozen baseline. Keep AI
trading in prospective shadow mode. Archive a variant from the main UI when it
adds complexity without a robust net-of-cost benefit; retain reproducible research.
Several months of forward paper results are useful operational evidence, not a
substitute for multi-regime validation.

### Proposed implementation sequence

1. Regression fixtures and corrections for Phase 1.
2. Shared ledger/decision kernel plus immutable experiment runs.
3. New portfolio comparison UI and scoped controls.
4. Corrected comparison report and decision on which variants remain prominent.

Each should be independently reviewed. No commits, deployment, data migration,
strategy switch or deletion is authorized by this audit alone.

## Phase 1 implementation status — 2026-10-21 (code complete, not yet accepted)

**Status date: 2026-10-21.** Branch: `feature/phase-1-financial-correctness`
(base commit `ba37afa`; commit `888c919` plus three uncommitted follow-up
batches). Phase 1 of the plan in section 5 is **code-complete** and
**independently verified** (see the checkpoint below). The acceptance criteria
of section 5 are met in code apart from the honest residual list further down;
formal acceptance is a separate decision, so this note does not declare Phase 1
accepted. The findings above remain the **original state** of the reviewed
checkout (`ba37afa`); their line references are deliberately left as written
and are not rewritten to the branch.

No commit, deployment, data migration, strategy switch, or new parameter
selection is authorized by this note, and no production data was changed.

### Batch workstreams — implemented and independently verified

| Workstream | Where the batch stands | State |
|---|---|---|
| Source-safe negative capex, including already-cached SEC facts | `app/fundamentals.py` (`canonical_capex`, `CAPEX_TAG`/`OCF_TAG`), `app/edgar.py` (parse-time flip), legacy cached rows corrected on read with no DB migration, non-finite facts skipped instead of ranked | Code implemented — independently verified |
| Missing capex must not become zero capex | `app/monthly.py` — absent capex yields an **unknown** FCF; OCF alone is never substituted, because assuming capex = 0 overstates FCF | Code implemented — independently verified |
| Unitized, cash-flow-neutral drawdown (Python + shipped JS + optimizer) | `app/performance.py` (shared unitized-NAV drawdown), `static/index.html` `mddOf` on the same convention, `app/optimize.py` `_max_drawdown` reduced to a wrapper over the shared module | Code implemented — independently verified |
| Bullish-only, finite fallback behind the strict trend guards | `app/analysis.py` `entry_eligible` (shared by the strict BUY and the fallback), `app/strategy.py` `_fallback_eligible` (finite positive net score, fails closed on missing snapshot fields) | Code implemented — independently verified |
| Sector room clamp | `app/strategy.py` — the BUY budget is clamped to the sector's remaining room, not only pre-checked | Code implemented — independently verified |
| Prior-known ATR reference, entry guard, no same-cycle SELL/BUY churn | `app/analysis.py` — the reference level is built from the previous completed bar only; `app/strategy.py` — an entry must be above the published level, and a ticker sold in a cycle is not bought back in the same cycle | Code implemented — independently verified |
| Actual ten picks, rank-first (live / backfill / research) | `app/monthly.py` `_deploy_targets` returns the hysteresis membership in ranking order; `app/daily_core.py` deploys those picks in the live and the backfill path instead of the raw top-N band | Code implemented — independently verified |
| Prior-window ranking, shared completed-month-end calendar, replay adapter | `app/optimize.py` — `month_for` uses the pre-window month-end frame and never the first in-window one; `monthly.completed_month_ends` is the shared calendar for backfill, research and sweep; the replay adapter supplies `net_score` / `trend_up` / `trend_down` / `weekly_trend_up` | Code implemented — independently verified |
| Frozen session fill contract | `app/session.py` — one documented, frozen contract: decision inputs are data ≤ the previous completed session S; fills use F's strict exact-date open (never forward-filled; a missing quote is no fill and the decision is re-taken next session); a stop level known at S's close is a standing order for F (open through the level → the open, else the low → the level, else nothing); a per-session guard makes a second cycle in the same session a no-op | Code implemented — independently verified |
| Entry-frozen percent stop and monotone ATR ratchet (sim) | `app/db.py` adds a nullable `SimPosition.stop_price` in place (additive `ALTER TABLE`, no migration; legacy rows keep the documented avg-cost fallback); `app/sim.py` persists the stop at entry and never re-anchors it on a top-up; `sim._atr_ratchets` takes the max prior-known ATR candidate over every completed bar since `opened_at`, so it can only tighten; the backfills write the actual first-buy session into `opened_at` instead of wall-clock now | Code implemented — independently verified |
| Strict-open USD fills and deferred month-end rebuild (monthly + daily-core) | `app/monthly.py` fills at the session's exact-date open converted to USD (`_open_usd_map`); a pending month-end rebuild is deferred to the first session of the next month with per-leg retry and an audit `rebal_date` kept distinct from the fill day; `app/daily_core.py` runs the documented month-end band policy with an injectable `_now()` clock; both backfills stamp `opened_at` with the actual first-buy date | Code implemented — independently verified |
| Research replay on the same contract | `app/optimize.py` — decisions on S closes, with the ATR entry gate judged against the price actually paid (F's open); the research-only cost parameter is explicit (`cost_oneway`, default `settings.sim_monthly_cost_oneway` = 10 bps one-way, documented as not the live fee model) and the fee is reserved inside the spend so a full-cash buy cannot drive cash negative | Code implemented — independently verified |
| TWR series and dated XIRR, honest UI chips | `app/performance.py` supplies the unitized NAV series and dated external flows; `GET /api/metrics/twr` returns the TWR series and `GET /api/metrics/xirr` a dated money-weighted XIRR; the Daily-Core, Monthly and Sim tabs read both chips and label the `%` view as gain on contributed capital, explicitly not a TWR | Code implemented — independently verified |
| Golden live/backfill and research/backfill parity | `tests/test_cadence_parity.py` drives the live engines session by session against the persisted backfill on one synthetic tape with real month-end rank flips and asserts identical trade sets plus `opened_at` = the actual first-buy date; `tests/test_replay_fills.py` pins the research replay to the live backfill over the same kind of tape with `open != close`, including that mutating F's own bar cannot change F's fills | Code implemented — independently verified |

### Verification checkpoint

The final independent integration run over the whole Phase 1 code scope
reported **783 tests passed in 103.02 s**, a clean `ruff` check over `app/` and
`tests/`, and a clean `git diff --check`. The run used local and synthetic
fixtures with temporary databases only — **no external data pull**. The two
final-review defects listed below were fixed with regression tests before that
run, and the follow-up batches above carry their own regression modules under
`tests/`.

### Final-review defects — fixed and covered by regression tests

- **The ten-slot contract could reach eleven in backfill/research.** Nine held
  names with one out-of-band name still awaiting release could take a helper
  fill and end the cycle with two new names. Fixed with hard capacity
  execution guards: the executor checks the actual free capacity after the
  SELLs, so a failed release keeps its slot and only a genuinely free slot can
  be filled.
- **The optimizer's future dead-month trim could remove prior decisions.**
  Trimming a later month with no eligible names could drop January's
  decisions because February's eligibility was used. Fixed with causal
  replay: a dead month is no longer trimmed on future-frame eligibility, and
  the dead month still deploys on its prior frame.
- Regression tests: `TestPendingReleaseExecutionGuard`,
  `TestDeadMonthCausality` and `TestMonthlyBacktestCalendarSettlement`
  (leading and trailing funding holes, parked-vs-invested money).

### Phase 1 status

The code scope of Phase 1 is **complete and independently verified**: the
acceptance criteria in section 5 are met in code, with this honest residual
list. The residuals are documented exclusions, not silent gaps, and none of
them reopens the corrected decision/fill path for the four strategies that
Phase 4 compares.

- **Savings mirror and chat trade paths keep legacy same-close fills.** The
  mirror allocates at the latest available close, and `sim.sim_chat` books
  chat-initiated trades at `_latest_close`; neither is one of the four
  research strategies, so neither is wired to the session contract.
- **The LLM decision ledger is deferred to Phase 2.** Frozen `decided_at` /
  `filled_at` per order is a Phase 2 accounting item (recorded in
  `app/session.py`, rule 7). The LLM path already uses data ≤ S and fills at
  F's open, but its reasoning is timestamped to the cycle that produced it.
- **Published performance and experiment numbers remain stale** until they are
  re-run on the corrected engine; that re-run is Phase 4 evidence work, not
  Phase 1.
- **Live `created_at` / `opened_at` are write-time on manual runs.** Only the
  backfills stamp the actual session / first-buy date; a manual cycle books
  the wall-clock write time.

### Notes and limits of this batch

- The per-bar ATR reference is **prior-known and reachable** — today's close
  can break a level that was already known before today's bar closed — but it
  is recomputed per bar and can move *down* as well as up. That snapshot value
  is therefore an indicator candidate, **not a ratchet**, and must not be
  described as one. The monotone behaviour lives in the caller-computed
  position-specific ratchet (the max candidate level over every completed bar
  since `opened_at`), which can only tighten. The percent stop is now frozen at
  entry and persisted on the position (`SimPosition.stop_price`) in the live
  sim and the replay alike, so a top-up cannot loosen it; legacy rows without
  the column keep the documented avg-cost fallback.
- **No new "winning" parameter or strategy selection** until the whole of
  Phase 1 is complete and accepted.
- **The previously published performance and experiment results are
  provisional and stale.** They were produced by the pre-correction engine
  (wrong capex sign, contribution-ratio drawdown, band-based deployment,
  future-frame ranking), so every published number must be re-run with the
  corrected engine before it is quoted, compared, or used to choose a strategy.
- The chart "%" view remains **gain on contributed capital**
  (equity / contributed − 1), **not** a time-weighted return. The labels now
  say so; the metric itself is unchanged.

## References

- [SEC Apple companyfacts](https://data.sec.gov/api/xbrl/companyfacts/CIK0000320193.json)
  — FY2025 OCF/capex example, period 2024-09-29 to 2025-09-27, filed 2025-10-31.
- [Portfolio Performance: time-weighted return](https://help.portfolio-performance.info/en/concepts/performance/time-weighted/)
  — cash-flow-adjusted daily return convention.
- [Blitz, Huij, Martens: Residual Momentum (2011)](https://pure.eur.nl/ws/files/46882404/ResidualMomentum-2011.pdf)
  — original estimator, especially methodology and alpha exclusion.
- [Asness, Frazzini, Pedersen: Quality Minus Junk](https://doi.org/10.1007/s11142-018-9470-2)
  — quality includes multiple profitability, growth and safety measures, not ROE alone.
- `README.md` and `docs/llm-strategy-experiments.md` — project research claims and
  stated limitations; reported historical performance was not independently rerun.
