# UI and financial-strategy audit

Date: 2026-10-21. Reviewed checkout: `ba37afa`.

Status addendum (2026-10-21): Phase 1 corrections are **code-complete and
independently verified** on `feature/phase-1-financial-correctness`, with
acceptance still pending — see "Phase 1 implementation status" at the end of
this document. The findings below are unchanged and describe the reviewed
checkout `ba37afa`; their file/line references are intentionally left as
written.

Subsequent verification (2026-10-30) **reopened a Phase 1 causal acceptance
gap**: factor order sizing read F's close before filling at F's open. The
October 21 checkpoint below remains a historical test record, not proof of
that missing invariant. All three engines were repaired the same day and
independently verified — sizing reads S, fills read F's open, reporting reads
F's close — with funded, nonempty-book mutation regressions; see the Phase 2
progress addendum.

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

**Status: complete and independently verified, 2026-10-30.** See
[final acceptance](#phase-2-final-acceptance--2026-10-30) for evidence and limits.
Commit/merge authorization is separate from acceptance; this does not start
Phase 3 or validate performance.

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
(base commit `ba37afa`; commits `888c919`, `4a9ac8d`, `93ef048` and
`fffda7e`). Phase 1 of the plan in section 5 is **code-complete** and
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

At the October 21 checkpoint, the code scope of Phase 1 was reported as
**complete and independently verified**: the
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

## Phase 2 implementation progress — 2026-10-30

Work continues on `feature/phase-1-financial-correctness` after the four
Phase 1 commits above. The whole roadmap is to be completed before merging;
new Phase 2 changes remain uncommitted unless separately authorized.

The first bounded implementation batch corrects the synthetic universe market
gate: chain equal-weight daily simple returns rather than average nominal
share prices, using one causal helper in live, persisted replay and research.
Per-listing price-unit rescaling must not change the index or its gate signal.
This does not turn the gate into a survivorship-free investable benchmark or
resolve stale-price and delisting policy. Independent verification passed
19 focused tests, scoped lint and a separate loop-reference check over a
random missing-price tape, including rescaling and prefix-causality probes.

A separate currency batch fixes the factor open lookup reading a nonexistent
close field from FX open rows and makes known foreign listings fail closed
when a valid prior FX rate is unavailable. It also adds Vienna's EUR mapping
and fixes a naive-as-of query bound. Independent verification passed 82
Monthly/FX tests and scoped lint. Full instrument metadata, pence scaling,
corporate-action accounting and explicit price-age limits remain open.

A fundamentals-ingestion batch extends the same fail-closed currency contract
to statement facts: the fabricated 1.0 "no FX data, assume USD" factor is
gone, and a converted fact (income, OCF, capex, equity) is only booked when a
usable rate exists at or before its period end — a missing pair, a
pre-history period, or an invalid (NaN/inf/negative) rate skips that fact
instead of storing a local-currency value as USD. Unknown dot-suffixes are
now explicitly unresolved (their converted facts skip rather than being
guessed USD); share counts are currency-free and always ingest. Independent
verification passed 33 offline tests plus an orchestrator probe of every
fail-closed path, with the ingestion driver left untouched. One honest
consequence: dot-form share-class tickers such as `BRK.B` in the small
us-large-cap universe (the S&P 500 file uses the dash form `BRK-B` and is
unaffected) now skip converted facts with an INFO log, and the genuinely
unmapped `NOVO-B.CO` stops being stored at a roughly sevenfold-wrong USD
value. Resolving share-class suffixes belongs to explicit instrument
metadata, not to per-path guessing — the batch after this one provides it.

A follow-up batch consolidates instrument quote metadata under one owner,
`app/instruments.py`: the suffix→FX map (moved verbatim, re-exported from
`fundamentals` so every existing importer keeps working), explicit
pence-scaled London quotes (`.L` closes are quoted in pence — a raw close
must be scaled by 0.01 before GBPUSD is applied, a latent hundredfold error
no shipped universe exercises but any custom one could), and an explicit
USD-native share-class set (`.A`/`.B`/`.C`, e.g. `BRK.B` — a notation, not a
currency code) so dot-form US share classes ingest fundamentals as
explicitly known-USD rather than accidentally. The factor price path
applies the unit factor before FX; unknown foreign suffixes remain raw and
explicitly unresolved; reporting-versus-trading currency remains explicitly
unmodelled and documented as open. Independent verification passed 210
scoped tests plus an orchestrator probe (pence conversion exact at
1250p → $15.625, `.DE`/`.SW` bit-identical to the previous arithmetic, `.L`
without its pair fails closed to NaN, re-export identity holds), and the
`BRK.B` consequence above is resolved: its converted facts ingest again,
now as explicitly known-USD. The Savings tracker's own EUR pricing still
multiplies a `.L` close by GBPUSD without the unit — the same latent error,
scheduled into the next Savings batch.

The price-age policy batch gives staleness one definition
(`app/price_freshness.py`): a print older than
`factor_price_max_age_days` (default 10 calendar days, rationale documented)
is not a current mark. Marks — live valuations and both persisted replays —
treat a stale cell exactly like a missing close, so the existing cost-basis
fallback takes over and the position is flagged `stale` (per-position and
`stale_marks`, additive keys, no shape change); ranking excludes a stale
listing explicitly (`eligible_frame` gains a `stale` column; the flag is
visible, the exclusion is the only consequence); an FX rate print older
than the limit no longer converts anything. The research replay gained the
cost-basis bookkeeping its marks needed to mirror the persisted replay, and
its final valuation now uses the same gated EOD mark. `_exec_buy` gained
the same hard-slot backstop the replay and research paths already have
(inert in normal flow — the live planner caps first; structural
equivalence). Independent verification passed 417 scoped tests, an
orchestrator probe of every policy boundary (10-day edge fresh, 11 days
stale, future print stale, ffill-backed cells expiring exactly, and a
strong-fundamentals listing with a month-old print explicitly stale and
ineligible while a fresh name ranks), and all five sizing reproducers
remain invariant under the F-close mutation. Two pre-existing Monthly
fixtures that seeded a hard-coded September bar against the real wall clock
had their valuation clocks pinned to the seeded session — a documented
expectation update (the assertions themselves are unchanged); they were
time bombs under any age policy. The audit's "indefinitely stale listing
stays silently eligible" gap is closed.

The read-only Savings review confirmed that position removal deletes booked
mirror entries and rewrites earlier comparison points. Its first implementation
batch retains those entries, appends a dated reversal and makes removal plus
mirror updates atomic. Cash semantics remain unchanged; this is not yet a
settled-sale ledger. A disjoint label-only batch makes clear that the mirror
is an equal-allocation picks comparison, not the full Daily-Core strategy.
The backend repair passed independent source review, 64 Savings tests and
scoped lint, including an injected post-write failure that proves the removal
and mirror book roll back together. Fixed-candle curve points strictly before
the removal day remain unchanged; the removal day's daily aggregate can change.
Reset/reinitialization are still destructive and are not covered by this repair.
The label-only change has been independently reviewed: the diff changes only
display strings/comments, and both inline JavaScript blocks pass `node
--check`. The backend history repair is verified as described above; no responsive or
rendered-page acceptance is claimed by this label check.

A follow-up batch gives the ledger its disposal record: `SavingsEvent`
gains nullable `occurred_on`/`shares`/`price`/`cost_basis`/`flow_class`
columns through an additive, non-backfilling migration (legacy removals were
never observed, so nothing is fabricated for them), and `remove_position`
appends a dated SELL row in the same transaction as the position delete, the
WARN audit row and the mirror reversal. The row is a record of facts, not
settlement: `amount` stays 0.0 — cash and contributed are untouched, the
proceeds still arrive via the next true-up — and the execution price is
stored only when the operator supplied one (an invalid price rejects the
whole removal before anything is touched). Independent verification passed
77 Savings tests plus an orchestrator probe of rejection-leaves-no-trace,
row exactness and cash/contributed inertia, and the injected-failure
rollback now covers the SELL row too. Flow classification of the wider
ledger, dated withdrawals, sale settlement and a non-destructive initialize
remain the open Savings batches; the disposal columns are not yet surfaced
through the events API.

Batch 3 adds the flow taxonomy, dated withdrawals and Savings metrics. One
classifier (`flow_class_of`) maps every event kind to its flow class —
contribution, withdrawal, investment, reward, interest, reconciliation,
sale, audit — auto-stamped on every new row and applied on read to legacy
rows (one definition, no backfill migration). `withdraw` books a dated,
signed-negative WITHDRAWAL event under an explicit cash guard, and every
contributed computation becomes NET (contributions − withdrawals): a
deposit-then-withdraw tape that the old metric reported as a −40% loss now
reports exactly 0% — the orchestrator verified that contrast directly.
Savings gains the same TWR/XIRR endpoints as the other tabs (key-identical
to `/api/metrics/*`, reusing the generic performance functions untouched,
plus one additive drawdown field), a flat flows-only tape pins at 0% TWR
with zero drawdown, and the tab shows the same chips with net-contributed
wording plus a Withdraw control. The tracker's own `.L` pricing now applies
the pence unit before GBPUSD (1250 GBp at 1.25/1.10 prices at
14.2045 € exactly; unit-1.0 venues bit-identical). Independent verification
passed 139 Savings tests plus the orchestrator probes; two integration
corrections were made at acceptance: the withdraw endpoint's
insufficient-cash rejection is 422 (matching `/api/savings/buy`'s class;
404 remains for the uninitialized tracker) and the `SavingsEvent` docstring
now records the WITHDRAWAL kind and the one-definition flow classification.

The LLM decision ledger (Phase 1's deferred rule-7 item) is implemented:
`SimLlmDecision` append-only rows for every LLM decision — hybrid review,
pure-LLM cycle, and both chat paths (including the streamed variant) — with
`decided_at` frozen at parse time, the decision's relation to the
deterministic proposal (one shared classifier, equivalence-checked on 500
randomised cases against the old bucket logic), and the frozen fill
(`filled_at` plus shares/price/cash_after) written once when the trade
executed; a decided-but-unfilled decision keeps NULL fill columns, which is
the honest record under rule 7's no-pending-order semantics. Ledger failures
are logged and can never fail a cycle. Independent verification passed 172
scoped tests, confirmed zero edits to any execution path, and the full suite
passes with network sockets disabled — no LLM or provider call anywhere.
session.py rule 7 now records the ledger instead of deferring it.

A further bounded provenance batch adds nullable candle source, requested
adjustment basis and ingestion time, retaining unknown provenance on legacy
rows and recording the actual provider after fallbacks. It is not an immutable
experiment data snapshot, and migration verification uses temporary databases
only; the running database is not being migrated in this work. Independent
verification passed 51 market/provenance tests and scoped lint. Review caught
and covered a compatibility regression: omitted volume must still default to
zero, while an explicitly invalid volume must not be silently coerced. A
second review finding is fixed and independently verified: the three extra
bound parameters per row lower SQLite's single-statement batch ceiling from
4,680 to 3,276 rows (independently reproduced: 3,276 bars stored, 3,277
failed), so ten-year and max-range refreshes could fail outright with zero
bars stored. The chunked single-commit upsert — 2,000 rows per chunk, the
bound documented against the 32,766-parameter cap and guarded against future
schema growth by a test — restores 3,650- and 5,000-bar stores with every
row tagged, mid-chunk failures roll back the whole batch, and 55
market/provenance tests pass.

Shared execution arithmetic now has pure zero-fee fill primitives and Monthly
live/backfill adoption. Independent verification passed 121 primitive/Monthly
tests and scoped lint, preserving floor-four sizing, the exact-cash guard and
valid-input operation order while rejecting invalid fills before mutation.
Daily-Core's adoption is complete and independently verified in the same way:
all four fill sites (live `_exec_buy`/`_exec_sell` and the backfill
`do_buy`/`do_sell`) call the shared primitives, a 200,000-case arithmetic
differential found zero mismatches on valid inputs, and the scoped suites
(153 tests including the golden cadence parity) pass unchanged. The adoption
also hardens the live executors: the old code raised on NaN budgets, and a
NaN or infinite SELL price could commit NaN cash or delete a position while
persisting an infinite-proceeds trade — those inputs now book no fill and
mutate nothing, pinned by 33 new tests. The invalid-input delta is a
disclosed behavior change from "raises/corrupts" to "no fill", not a
silent one. Research adoption is complete as well, verified the same way:
`_fill_buy` and the replay's SELL arm are thin fee-aware wrappers over the
primitive `buy_fill`/`sell_fill` (the fee comes out of cash only, sizes are
never re-floored), a grid comparison found the wrappers bit-identical to the
old formulas at every fee level, and the wrapper's fee-inclusive guard was
proven NON-redundant — real inputs exist where the zero-fee primitive
accepts a leg whose fee-inclusive total exceeds the cash by one ULP, which
only the retained guard rejects. The research sizing reproducer remains
invariant under the F-close mutation and all five parity suites pass
(184 scoped tests). With all three engines on the shared arithmetic and the
per-ticker risk checks structurally identical, the shared-execution item's
code scope is complete. A separate
read-only review confirmed, and the
orchestrator independently reproduced, pre-fill use of F-close position values
in factor order sizing. Changing only a held name's F close from 100 to 140
changed live Monthly and Daily-Core F-open buys from 5/10 shares to 4/11,
with identical F opens and history through S. Research and both persisted
replays share the defect, so cross-engine parity passes despite it. This
reopens Phase 1's causal acceptance criterion rather than being treated as an
optional enhancement. Repairs are bounded by engine: sizing uses S marks and
actual cash, execution uses F open, and end-of-session reporting remains F
close. The Monthly engine's repair is complete and independently verified:
both reproducers show identical F-day trades under the F-close mutation (live
and persisted replay) with F-close reporting intact, and 128 scoped tests pass
with clean lint. The research replay's repair is verified the same way: its
reproducer is invariant under the mutation, all four confirmed sizing sites
plus the target-vol cap read S, 152 scoped tests pass including both golden
parity suites, and the three existing research test files are unchanged. The
Daily-Core clock repair is verified the same way: both of its reproducers are
invariant under the mutation with F-close reporting intact, its sizing
valuation and post-buy refresh read S, its target-vol history is bounded to
sessions before the cycle, and the contract docstring (rules 1/3) now states
the sizing/reporting split explicitly. With all three engines repaired, the
reopened causal gap is closed in code: every reproducer is invariant, both
golden parity suites are green, and the full integrated suite passes (933
tests, clean ruff, clean diff check). Formal acceptance remains a separate
decision.

At the subsequent offline checkpoint, the Savings Batch 4 implementation adds
dated dividends, fees, sale settlements, splits and corrections; snapshots
carry an optional `as_of` date, reinitialization refuses to overwrite a live
tracker, and reset preserves events/snapshots behind an appended boundary.
Independent verification passes 239 Savings tests and the full 1,314-test
suite with outbound connections blocked and **zero connection attempts**;
all five sizing reproducers remain invariant. Review identified same-day reset
clipping, split-source attribution, missing partial-sale/tax entry points and
stale reset UI wording; the follow-up repairs are now independently verified
with **263 Savings tests, zero connection attempts**, plus an orchestrator
end-to-end probe. Partial sales preserve the remaining quantity and average
cost, append exact disposal facts and proportionally release the mirror;
settlement credits cash exactly once through a unique disposal link. Unlinked
legacy settlements require explicit acknowledgement and an explanatory note.
Splits append one adjustment per funding source, measured at the split date;
later buys are not multiplied. Unsupported retroactive actions that would
require replaying later position changes refuse instead of inventing holdings;
future-dated mutations also refuse. Tax is a dated internal expense, not an
external withdrawal. Snapshot epoch identity follows creation order, while
`as_of` controls valuation-day grouping, so same-day post-reset metrics work.
The correction tests preserve the original cash/share/return invariants and
document each intentional expectation change.

The per-execution provenance summarizer independently passes **29 tests with
zero connection attempts**. It records bounded cached candle/fundamental
source, adjustment-basis and ingestion-time summaries for the actual universe
and required FX pairs. Unknown provenance stays explicitly unknown; unresolved
membership never widens to the entire cache. This is an observation of cached
inputs, **not an immutable dataset snapshot or a content-replay guarantee**.
Immutable run configurations, mode/epoch stamping and protection of forward
paper records are in final integration: the independent schema/adoption gate
passes **53 tests with zero connection attempts**. A separate synthetic probe
confirms all three paper engines refuse unknown legacy history before mutation,
preserve it bit-for-bit on reset, and credit a new epoch's same-month allowance
exactly once. Migration default parity, same-month rebalance provenance and
opening-capital stability under configuration changes remain under final review.
The account-schema follow-up independently passes **39 tests with zero network
attempts**: nullable `opening_seed` records actual newly opened capital without
inventing a value for legacy accounts. The first full integrated run passes
1,415 tests and exposes four obsolete destructive-history expectations plus one
decision-clock regression (the unchanged nine-test ledger suite subsequently
passes after the application fix). A separate review also reproduces a stale
writer/reset epoch race; fencing those writes and freezing the actual opening
capital are still required before closure. This is not yet the full integrated
acceptance gate. No production migration,
provider refresh, strategy switch, commit or new performance claim is part of
this checkpoint.

## Phase 2 final acceptance — 2026-10-30

**Phase 2 is complete.** This acceptance supersedes the intermediate checkpoints
above; their failures and open-item descriptions remain as the review history.
The final integrated gate is independently run on the released working tree:

```text
1435 passed in 247.68s (0:04:07)
NETWORK_CONNECTION_ATTEMPTS: []
SOURCE_FILES_CHANGED_DURING_GATE: []
```

Outbound IPv4/IPv6 `connect` and `connect_ex` were blocked and counted. Tests
used synthetic fixtures and temporary/in-memory SQLite, not the production
database. The final source-hash manifest still matches the accepted tree.
`ruff check app tests`, `git diff --check` and `node --check` for both inline
JavaScript blocks pass. All five sizing reproducers also pass in **separate
fresh interpreters**, with parsed base/mutated trade tuples asserted equal and
zero network attempts; reporting still reflects the changed F-close marks.

### Accepted integration

- Shared fill/accounting, currency/FX, market-index and price-age primitives
  are integrated without merging technical and factor strategies into one score.
  Exact-cash, eligibility and slot backstops remain in the execution paths.
- Immutable `RunConfig` registrations capture effective settings, actual universe
  membership/hash, currency/fill conventions and a version contract. Cache
  identity follows that configuration; unknown ranking provenance invalidates
  the derived cache. A conflicting same-month paper rebalance refuses rather
  than silently re-stamping its earlier decision.
- `RunExecution` separately records admitted runs and bounded cached-input
  provenance. A registered configuration is not evidence an experiment ran;
  read-only ranking polls, terminal data skips and refused replays do not count
  as executions. Replay rows and their execution record commit atomically.
- All three paper engines protect unknown legacy and forward-paper records
  before replay mutation. Resets atomically append an epoch boundary, preserve
  historical ledgers and reseed current state. Decision/metric readers scope the
  current epoch; raw history exports intentionally retain closed epochs.
- Reset and final replay-persistence transactions share short state-write locks
  with standalone writers. Inside-transaction epoch fencing rejects stale
  allowance, trade, ranking and snapshot writes. Snapshot valuation cannot cross
  a reset. Independent event-barrier verification proves a queued writer cannot
  enter its body before replay commits, then credits cash exactly once afterwards.
- Actual `opening_seed` is persisted when an account opens, resets or replays.
  Changing settings does not invent funding; a zero-seed factor replay remains
  zero-seed when paper trading starts. New epochs have independent imaginary
  capital baselines, not transfers from the closed paper book.
- Savings acceptance includes exact partial/full disposal facts, linked
  exactly-once settlement, dated tax/expense/income events, source-correct split
  adjustments, append-only mirror releases and same-day post-reset metrics.
  The independent end-to-end probe reconciles quantities, cost, cash, mirror
  allocations and preserved history.

Independent focused evidence includes 39 schema/migration tests, 29 cache
provenance tests, 263 Savings tests and 46 adoption/sizing-clock/unchanged-ledger
tests. Existing tests were corrected only for the intentional history-preserving
reset/replay policy, explicit legacy settlement, operator-local dates and
known-matching ranking-cache fixtures. Original financial/golden parity assertions
remain; preservation and refusal assertions were added, not skipped or weakened.
The decision-clock regression found during integration was fixed in application
code, with the original ledger tests unchanged.

### Explicit limits and next phases

Legacy seed/source metadata remains unknown where it was never observed: seed
NULL uses a conservative zero fallback and is marked unknown in run provenance,
not inferred from current settings. Cached-input summaries are **not immutable
dataset snapshots or content-replay guarantees**. The singleton ranking and DCA
benchmark remain documented derived-artifact reset exceptions, not immutable
decision ledgers. Savings mirror restatement remains day-granularity; ambiguous
retroactive actions requiring later-position replay refuse. Reporting-versus-
trading currency and additional signal-staleness gating are not introduced here.

Verification performed no provider refresh, LLM call, production migration or
strategy switch. Commits and merges require separate explicit authorization.
Published performance claims remain stale until Phase 4
revalidation. Phase 3 UI simplification and Phase 4 research validation/pruning
remain the next roadmap work; Phase 2 acceptance is not a claim of investment
edge or live-provider correctness.

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
