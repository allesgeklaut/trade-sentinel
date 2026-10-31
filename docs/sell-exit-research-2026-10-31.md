# Sell-side / exit-decision research — 2026-10-31

Owner's hypothesis: **"what all these engines do well is buying, but they all
struggle with selling."** Mandate: research first (papers and publications),
including whether an LLM could help the sell decision specifically. The owner
authorized a model for the eventual sell-decider experiments — **`deepseek-v4.1-flash-mistral`**
(the exact id as it appears in the live hosting's model dropdown; served by
their **Mistral** serving — not ollama; the `deepseek-v4-flash` string in
`app/llm.py`'s docstring example is coincidental). This document is
the research deliverable. **No engine change follows from it without separate
approval** — the decision gate is at the end.

Method: ground the hypothesis in our own data first (the full-window reval
ledgers, 2015-01-02..2026-10-30, sp500 universe, CONTRACT fills, default costs,
classic scoring — the same evidence base as Phase 4A), then map the literature
onto what the data shows, then design the experiments. All numbers below are
measured, not asserted; scripts and raw outputs live under
`/tmp/opencode/reval/` (`excess_drift_analysis.py`, `disposition_analysis.py`,
`post_trade_drift.json`, `excess_drift.json`, `sim_reason_drift.json` — the
last from `/tmp/opencode/reval-4a-wt/sim_reason_drift.py`, a direct
`_replay` rerun to recover per-trade sell reasons, which the reval report
ledger does not carry).

## 1. What our own data says

Three lenses on the same ledgers: **raw forward drift** (what happened after
each trade), **excess drift** (the traded name minus the average universe
member over the SAME dates — this kills the market-timing confound, because
sells cluster in rebounds), and **Odean's PGR/PLR disposition measure**
(how readily an engine realizes gains vs losses).

### 1.1 The sim's entire exit stack is one rule: the ATR trailing stop

Over 11.75 years and 6,686 trades, **all 2,773 sells are ATR trailing-stop
standing orders** — zero signal sells, zero percent-stop sells, zero
portfolio-breaker sells. The sell-signal path in the technical strategy is
effectively dead code: the ATR ratchet is tighter than any signal exit and
always fires first.

Consequences, measured on the closed positions (3,630 lots):

| quantity | value |
|---|---|
| holding period, mean / median | 16.1 / **6 sessions** (~1 week median) |
| held ≤ 1 month / ≤ 1 quarter | 84% / 98% |
| stops firing ≤ 5 sessions after entry | **48%** |
| giveback from holding-period peak, median / mean | **3.1% / 3.4%** |
| sold within 5% / 15% of the peak | 80% / 99% |

The trail sits ~3–5% below the peak, so **ordinary pullbacks inside uptrends
trigger exits**. The strategy can never hold a winner for the months the
momentum literature says the factor needs (§2.3), and it re-buys after the
whipsaw, paying one-way costs both ways — the 22×/yr turnover measured in
Phase 4A is almost entirely this loop.

### 1.2 The raw "sells give up upside" impression is market timing, not selection

| arm · side | raw fwd drift +63 sessions | excess vs same-date universe | universe baseline at those dates |
|---|---|---|---|
| sim · BUY | +3.77% | **−0.87%** (median −1.64%) | +4.78% |
| sim · SELL | +4.61% | **−0.78%** (median −1.59%) | **+5.11%** |
| monthly · SELL | (n/a, synthetic ledger → close-based −1.04%) | **−1.04%** | +4.42% |
| daily_core · SELL | +2.53% | **−0.98%** | +4.43% |
| daily_core · BUY | +5.59% | +0.72% (median −0.60%) | +4.49% |
| monthly · BUY | (close-based −0.10%) | −0.10% (median −1.93%) | +4.77% |

Readings:

- The sim's sell-date universe baseline (+5.11%/quarter) is the **highest of
  any trade type**: its stops fire into rebounds — exactly the momentum-crash
  regime the literature warns about. The raw +4.61% "sells give up upside"
  number is almost entirely market beta at those dates; the selection-level
  truth is −0.78% (sold names slightly *lag* the average member afterwards).
- The sim has **no selection edge on either side** (BUY −0.87%, SELL −0.78%):
  the raw positive post-buy drift was also market beta. Its decade result
  (−2.2%/yr, maxDD 41%, Sharpe −0.24) is what no-edge selection plus 22×
  cost churn plus whipsaw realization compounds to.
- **The factor arms' exits have edge**: names they drop (band exits)
  subsequently *lag* the same-date universe average by ~1%/quarter, at both
  the monthly and daily-core cadence. Their selling is not the problem.

### 1.3 Disposition: the sim is a loss-churning machine, not a naive profit-taker

Odean (1998) counts, on each sell day, positions realizing gains (RG) vs
losses (RL) and still-held paper gains (PG) vs losses (PL):
PGR = RG/(RG+PG), PLR = RL/(RL+PL). Classic retail disposition effect is
PGR ≫ PLR (sell winners too readily). Measured on our ledgers:

| arm | PGR | PLR | reading |
|---|---|---|---|
| sim (1,382 sell-days) | **0.078** | **0.360** | realizes losses **4.6×** more readily than gains (RG 843 / RL 2,063) |
| daily_core (94 sell-days) | 0.137 | 0.180 | near-neutral, mild reverse |

So the engines do **not** exhibit the retail bias the owner's phrasing
suggests ("selling too early" in the disposition sense). The sim's failure
mode is the opposite one: a stop stack that **crystallizes drawdowns into
realized losses during the whipsaws and rebounds where those drawdowns would
have recovered** — the behavior Kaminski & Lo's framework predicts will hurt
when post-stop drift is positive, which is exactly what §1.2 measures here.

### 1.4 Refined diagnosis

- **Factor arms (monthly / daily_core): selling is fine; risk-control is the
  gap.** Name-level exits have edge and no disposition problem. What they
  lack is any *portfolio-level* de-risking — the hold band is their only exit
  (the audit already noted a rank-retention band is not a risk-management
  system).
- **Sim (technical): the sell stack IS the failure**, through one mechanism:
  a ~3% ATR trail that (a) realizes ordinary pullbacks as losses into
  rebounds (PGR/PLR 4.6× loss-skew, sell-date baseline +5.1%), (b) prevents
  holding winners past a median 6 sessions, (c) generates the 22×/yr
  turnover whose cost drag Phase 4A measured (~2.2%/yr at 10 bps).

The owner's hypothesis is therefore **confirmed for the sim and refuted for
the factor arms** — and in both cases the productive frontier is the same:
risk management at the *portfolio* level, exit discipline at the *regime*
level, not per-name stop tuning.

## 2. What the literature says

### 2.1 Stop-loss rules mostly do not add value — and ours is the worst case for them

- **Kaminski & Lo (2014), "When do stop-loss rules stop losses?"**
  (J. Financial Markets 18): an analytical framework for the value of
  stop rules on an arbitrary strategy. The rule's benefit hinges on the
  post-stop drift: stops help when the underlying tends to keep falling
  after the stop; they hurt when it reverts. At longer sampling frequencies
  some policies can improve return while cutting volatility — the benefit is
  regime-dependent, not universal.
- **Clare, Seaton, Smith & Thomas (2013)** ("trading can damage your wealth"):
  on the S&P 500, stop-loss rules do not add value to trend rules; monthly
  decision cadence beats daily; the punchline we should take literally:
  **"a change of trend is the best stop loss."**
- **Dai, Li & Zhang (2021)** (trailing stops): trailing stops can reduce risk
  but with givebacks whose size depends on trail width; tighter trails trade
  risk reduction for return giveback — the exact trade our 3.1% median
  giveback is making.
- Time-based exit heuristics (deadline-to-recover, N-day progress checks)
  fail in every controlled test we found (e.g. the Tickfloor holding-clock
  series: 0/3 survive multiple-testing correction, all negative vs
  benchmark).

Our data matches this literature uncomfortably well: the sim's stops fire
into positive-drift regimes (sell-date baseline +5.1%/quarter) — the case
where Kaminski & Lo's framework says stops destroy value — and its exits are
price-triggered, not trend-triggered, which is the specific structure Clare
et al. found valueless.

### 2.2 Disposition

- **Odean (1998)**: retail investors realize winners ~1.7× more readily than
  losers (PGR 0.148 vs PLR 0.098 in his sample), and the winners they sell
  subsequently outperform the losers they keep. The bias is expensive.
- **Shefrin & Statman (1985)**: the prospect-theory origin of the effect.

Our engines are free of this bias (§1.3) — the useful takeaway is
methodological: PGR/PLR is a cheap, well-established instrument for
auditing any future exit module (including the LLM one) for disposition
drift.

### 2.3 Momentum's exit problem is a *risk* problem — manage it at the portfolio level

- **Research Affiliates, "Can Momentum Investing Be Saved?"**: momentum's
  buy side is the easy half; profits reverse precisely when discipline is
  hardest; sell discipline (rank-based, rule-bound) is what saves momentum.
- **Barroso & Santa-Clara (2015), "Momentum has its moments"** (JFE 116):
  momentum's risk is highly predictable (OOS R² ~58%); **scaling exposure to
  realized volatility eliminates the crashes**: Sharpe 0.53→0.97, maxDD
  −96.7%→−45.2%, excess kurtosis 18.2→2.7 — and turnover barely moves
  (74%→75%/month), so the risk management is nearly free after costs.
- **Daniel & Moskowitz (2016), "Momentum crashes"** (JFE 122): crashes happen
  in panic states (post-decline, high vol) and are contemporaneous with
  *rebounds* — the losers crash up. A dynamic strategy scaling on forecast
  mean and variance roughly doubles Sharpe vs static momentum. The rebound
  structure is exactly where our sim's stops concentrate their damage
  (§1.2).
- Cost-aware exit work (e.g. Tajaddini 2015; the Novy-Marx/Velikov turnover
  literature): strategies that are *fearful of transaction costs* net out
  better than naive ones — i.e. fewer, more deliberate exits.

### 2.4 LLMs: the consistent demonstrated strength is risk management, not alpha

- **StockBench (2025, arXiv:2510.02209)** — contamination-free multi-month
  trading benchmark: **most LLM agents fail to beat buy-and-hold on return,
  but every tested model beat it on maximum drawdown.** Static financial
  knowledge does not transfer to trading; downside-risk behavior is where
  LLM agents demonstrably help.
- **FinCon (NeurIPS 2024)**: manager-analyst hierarchy with a dual-level
  risk-control component — CVaR-triggered within-episode risk aversion plus
  episodic belief updates. The architecture that works treats the LLM as a
  *risk reviewer*, not a signal generator.
- **TradingAgents (2024/25)**: multi-agent debate with a dedicated risk
  management team; gains are modest and evaluation is mostly
  contamination-suspect.
- **Profit Mirage (2025, arXiv:2510.07920)**: leakage metrics show
  LLM-financial agents rely on memorized patterns (prediction-change scores
  >0.69 across methods) — generalization is the exception.
- **"The Alpha Illusion" (2026 position paper)**: LLM trading results are
  not deployment evidence without temporal-integrity, friction, and
  counterfactual tests.
- In-repo: `docs/llm-strategy-experiments.md` (prior hybrid experiments,
  contamination caveats); the audit rated LLM/hybrid decisions 3/2 —
  "better as a read-only explainer or shadow reviewer."

The external evidence therefore lines up with the owner's intuition in a
specific way: an LLM's plausible edge is in the **risk/sell review** — the
one function where even benchmarks show consistent LLM value — provided it
runs in shadow mode with honest fills, costs, and counterfactual tracking
(the Alpha Illusion conditions), because return generation is where LLMs
demonstrably fail.

## 3. Candidate improvements, in evidence order

Each is a predeclared experiment, offline in the Phase 4 harness, before
anything touches a live engine. The LLM decider is *one* candidate among
these, not the centerpiece — a cheap deterministic change that captures the
same gain wins over the LLM by construction (fewer moving parts, no
contamination surface, zero call cost).

**Sim exit-rule variants** (all measurable with `_replay` + the honest
metric pipeline; none needs an LLM):

1. **Trend-change exits instead of price stops** — Clare et al.'s direct
   recommendation; matches §1.2's finding that price-triggered exits fire
   into rebounds.
2. **Wider trail** (larger ATR multiple / percent floor) so the trail sits
   below ordinary-pullback range (currently ~3%; the giveback distribution
   in §1.1 sizes the change).
3. **Entry grace window** — no stop inside the first N sessions except a
   crash gap (48% of stops fire within 5 sessions of entry; most of those
   are whipsaws, not trend failures).
4. **Monthly stop cadence** — evaluate standing stops at month-ends only
   (the factor cadence), trading intra-month gap risk for churn reduction;
   Clare et al. found monthly beats daily for the rules themselves.
5. **Portfolio-level de-risking instead of per-name stops** — vol-targeted
   exposure (Barroso & Santa-Clara) or drawdown-gated deployment; the
   literature's best-supported mechanism, and it addresses the sim's
   41% maxDD directly rather than one position at a time.

**Factor arms** (their name-level selling already has edge; the gap is
portfolio-level):

6. **Vol-targeting at rebalance** (scale gross exposure by realized vol) —
   the strongest risk-adjusted-return evidence in the entire literature
   scan, nearly free in turnover.
7. **Drawdown circuit breaker** — the daily-core engine already carries a
   `crash_pause` overlay that the frozen baseline deliberately disabled;
   it can be evaluated as-is in the study driver without new code.

**LLM sell-decider** — §4.

## 4. The LLM sell-decider: experiment design (for approval)

**Research question.** Given the same position context the engine already
computes, can an LLM improve the sim's exit decisions — measured against
(a) the deterministic stop it would override and (b) the best cheap
deterministic variant from §3 — net of honest CONTRACT fills and costs?

**Where it sits.** Shadow mode only, per the roadmap's standing rule (the
LLM cannot be replayed faithfully; prospective evidence only; never
auto-executed). Two consult points:

- **Stop-fire review**: whenever the ATR stop would fire, the LLM receives
  the position context and returns CONFIRM or DEFER (+ a re-arm rule: wider
  stop level or time limit). The live engine still executes the stop; both
  branches' forward outcomes are tracked.
- **Monthly portfolio risk review**: at each month-end the LLM sees the
  book (positions, P&L, thesis, time held) and benchmark regime, and
  returns a risk posture note (no trade authority) — the FinCon-style
  risk-reviewer role that StockBench evidence supports.

**Context provided** (PIT only, all from data the engine already computes;
no news ingestion in v1 — keeps the contamination surface minimal and the
prompts replayable-in-form): entry reason + strength, sessions held, P&L
vs basis and vs peak, distance to trail, ATR14, benchmark drawdown/vol
regime, position weight. Prompts frozen and versioned; model called at
temperature 0; refusals/malformed outputs count as CONFIRM (fail-safe to
the existing rule).

**Authorized model.** **`deepseek-v4.1-flash-mistral`** (exact id, as it
appears in the live hosting's model dropdown), served by the owner's
**Mistral** serving behind the LAN LiteLLM proxy at
`http://192.168.0.46:4000` (OpenAI-compatible; the repo's LLM layer selects
backends via the `LLM_BACKENDS` JSON). Verified from the offline sandbox on
2026-10-31: the proxy's health endpoint answers, the sandbox's key store
(`/opt/secrets/litellm.key`) authenticates, the model id is present in the
proxy's `/v1/models` list, and a 1-token smoke completion at temperature 0
returned cleanly (`finish_reason: stop`). The live server's `.env` remains
the authority for the production wiring at deploy time.

**Contamination controls** (from the Profit Mirage / Alpha Illusion /
in-repo findings): evaluation on post-cutoff data only (and declared as
such whatever the cutoff turns out to be); no future data in any prompt;
model version + every prompt/response logged verbatim; the counterfactual
ledger is computed by the same pipeline as everything else in this audit
(CONTRACT fills, declared costs, unitized-NAV metrics).

**Primary metrics.** Per-stop-fire: the deferred branch's realized outcome
vs the stop branch (excess drift at the engine's own re-entry points, net
of round-trip costs). Portfolio-level: TWR/Sharpe/maxDD of the
LLM-influenced policy vs the stop policy vs §3's best deterministic
variant; PGR/PLR of the LLM-influenced book (disposition drift check).

**What would count as success.** The LLM branch beats the stop AND the best
deterministic variant on net excess drift with a dispersion that survives
the 4B bootstrap treatment, across regimes (the yearly windows), with no
disposition drift. Anything less → the deterministic variant wins and the
LLM stays a read-only explainer.

**Cost/latency bound.** Consults only on stop fires (~1/day in the sim)
plus a monthly review — a bounded call volume logged with token counts.

## 4a. The monthly-buys strategies: an intra-month sell mechanism (5B — predeclared)

Owner's extension (2026-10-31): a strategy combining the monthly
fundamental (qv-mom) buys with a mechanism to sell during the month at the
right time. Measured rationale: the factor arms' exits HAVE edge (dropped
names subsequently lag the universe by ~1%/quarter, §1.2) but the monthly
cadence realizes that edge up to 4–7 weeks late — and the sim evidence sets
the boundary: tight per-name price stops destroy value (§1, §3), so the
promising mechanisms are prompt rank-based release and portfolio-level
scaling. The daily-core engine already carries the needed knobs (the audit
rated their added benefit "unproven" — they were never run through the
honest pipeline).

The predeclared one-variable-at-a-time set (same window/universe/costs,
CONTRACT fills, the 4A metric pipeline, no tuning; M0 = the reference):

- **M0 baseline**: the monthly arm (band-drops wait for the month-end
  rebalance).
- **M1** `exit_cadence="daily"`: a band-dropped holding is released the
  NEXT day — the core of the request; captures the measured exit edge
  ~2–6 weeks earlier.
- **M2** M1 + `entry_gate="not-crash"` (a buy-timing guard on the monthly
  picks).
- **M3** vol targeting: `target_vol` 0.20 / 0.25 (the literature's
  best-supported risk mechanism, Barroso & Santa-Clara — scales EXISTING
  exposure, fixing the audit's finding against the contribution-only
  volatility gate).
- **M4** catastrophe trailing stop 25–30% per name (skeptical prior: the
  sim proved tight stops destroy; a wide guard on a 10-name,
  ~9-month-hold book is a different animal — one test).
- **M5** fundamental thesis-break exit: when a holding's newly FILED
  fundamentals would knock it out of the eligible frame (e.g. ROE ≤ 0),
  exit at the next open. The most fundamental-consistent intra-month sell —
  exit when the REASON breaks, not when the price wiggles. Needs an
  additive engine hook (filing-day eligibility re-check); deferred until
  4B lands (same hook window as the sim's entry-grace/cadence hooks).

Deliberately out of the set: the `basket_*` all-cash market-timing
overlays (audit 3/2 — whipsaw, fragile evidence) and any tight per-name
stop (the sim's measured failure). M1–M4 are expressible with existing
engine knobs (zero engine changes); configs are frozen in this section
before any run. Evaluation: full window 2015→2026 plus yearly windows,
against 4B's controls (random_n / equal_weight) and bootstrap CIs; if a
variant wins, it becomes a UI strategy variant candidate and a natural
replacement for whatever 4D prunes.

**Results (2026-10-31, full window 2015-01-02..2026-10-30, 2,977
sessions, sp500 snapshot, $1,000/month, flat 10bps + FX post-hoc, cash
yield 3%, `configs_tried=7` — `monthly-variants/1`):**

- **M0**: TWR +17.87%/yr, maxDD 35.1%, Sharpe 0.75, 1.31×/yr, 165 sells
  (all band releases at the rebuild, median hold 105 sessions, PGR 0.093
  / PLR 0.141). Parity vs the monthly arm: `contributed` exactly equal,
  IRR 1.0pp apart (20.3% vs 21.3%, within the engine's own 5pp
  equivalence tolerance).
- **M1 and M2: bit-identical no-ops on real data.** All 165 band
  releases already fill on the first session after the month-end rebuild
  (release lag max 0 in both cadences), and the per-name not-crash gate
  never bound once in 141 months. Confirmed at the engine level: the SELL
  pass only runs `if order:` and the band only changes at month-end
  rebuilds — a daily exit cadence can only change the RETRY timing of
  failed rebuild fills, never create intra-month exits. **Mechanism
  correction to this section's premise: the ~1%/quarter exit edge (§1.2)
  is delayed by monthly DETECTION, not monthly response — cadence cannot
  fix it. The owner's intra-month sell therefore rides on the
  intra-month event sources: M5's filing-day thesis-break re-check and
  the 5C type-C alarms.**
- **M3a (`target_vol=0.20`): the decade's winner — better on every axis**
  (+18.34%/yr, maxDD 33.3%, Sharpe 0.79, turnover 1.25×); M3b (0.25)
  close behind (+18.18%, 33.4%). Vol targeting is the only predeclared
  overlay that improved the book — the candidate UI variant pending the
  4C controls/bootstrap context.
- **M4a/M4b (25%/30% catastrophe trails): a dead end** — +16.68%/+17.31%
  with WORSE drawdowns (36.4%/36.0%) and higher turnover. Per-name price
  stops are now measured value-destroying on BOTH engines (the sim's
  ~21pp/yr gap; the factor book's −1.2pp with worse DD) —
  contra-indicated platform-wide, matching the skeptical prior.
- **M5 (thesis-break, run 2026-10-31 after the engine hook landed)**:
  the interpretation pin, frozen before the run: the thesis breaks when
  the fundamentals record is missing, OR equity ≤ 0, OR TTM NI missing, OR
  shares missing, OR **ROE = TTM NI / equity ≤ 0** — evaluated on the
  fundamentals public as of the decision session S (filing-date PIT via
  `monthly._asof_entries`, restatements replace originals); F/P-FCF
  changes alone do NOT trigger; price-side gates are NOT re-checked
  intraday; the break fills at the next session's open; re-entry
  unchanged. **Results (full window, `configs_tried=8`): 7 breaks in 141
  months** (BA 2017-07 and 2018-07, SBUX 2019-01, HCA 2021-08, DELL
  2022-03, COR 2022-11, DVA 2025-05), response lag 1 session on all 7,
  the monthly-detection delay avoided median 5 / max 21 sessions — but
  **portfolio-neutral**: TWR +17.92% vs M0's +17.87%, identical maxDD /
  Sharpe / turnover. The events are real but rare, and the month-end
  frames' own hard gates catch these names at the next rebuild anyway —
  the hook buys about a week of exit timing, seven times a decade. The
  deterministic intra-month program is therefore fully measured: M1/M2
  no-ops, M3a the only winner, M4 harmful, M5 neutral — the monthly
  factor book's exits are already near-optimal at monthly cadence. The
  platform's intra-month exit edge lives on the sim side (V1) and in the
  reviewer's SELECTIVITY (5C) — whose type-C event source is now built,
  with a known candidate density (~0.6 thesis-breaks/year + the type-B
  band-drops).

## 4b. Unification: one LLM sell reviewer, per-engine event adapters (5C — design)

The §4 sell-decider generalizes from "sim stop fires" to a single shadow
reviewer over CANDIDATE EXIT EVENTS from every engine, sharing one context
schema family, one prompt/versioning discipline, one fail-safe (CONFIRM =
the engine's own rule stands), and one counterfactual-ledger pattern:

- **Event type A — sim ATR stop fires**: CONFIRM the stop / DEFER with a
  wider re-arm (the 5A-1 machinery, built and gated).
- **Event type B — monthly band-drop events**: at each rebalance a name
  falls out of the hysteresis band; the reviewer CONFIRMs the release or
  DEFERs retention to the next rebalance. The measured bar is HIGH:
  blanket retention of dropped names is negative-edge (−1.0%/quarter
  excess, §1.2), so deferral must find the specific subset where it wins —
  the mirror image of the sim case. **Measured (2026-10-31, full window,
  `banddrop-ledger/1`, 165 events, self-check 165/165): blanket retention
  mean −1.60pp/event (median −1.03pp, 44.2% positive) — and the by-year
  selectivity map is the reviewer's assignment: retention WINS in
  bear/choppy years (2022 +3.18pp mean, 62% positive; 2020 +1.32pp) and
  LOSES catastrophically when fresh additions rip (2023 −13.30pp mean,
  only 9% positive; 2016 −2.85pp; 2018 −2.60pp; 2026-partial −9.20pp).
  The dropped names' raw ~1-month drift is ~flat (+0.04pp/event) — the
  negative edge is the REPLACEMENTS outperforming the retained names:
  the ranking works; the reviewer must find the events where the specific
  retained name beats its replacement. The B prompt is frozen
  (`sellshadow-b/1`, parameter-free DEFER) and the machinery is committed
  (`app/sellreview.py`).**
- **Event type C — intra-month thesis-break candidates for the factor
  book**: the M5 filing-day hook's predeclared condition (record missing
  OR equity ≤ 0 OR TTM NI missing OR shares missing OR ROE ≤ 0; F/P-FCF
  alone does NOT trigger; price-side gates excluded — the M-set verdict
  contra-indicates per-name price alarms platform-wide); the reviewer
  proposes an early exit the monthly cadence would have delayed. The live
  book still waits for month-end unless a policy survives validation.
  **Measured (2026-10-31, full window, `thesisbreak-ledger/1`, 7 events,
  self-check 7/7): the early exit vs the month-end wait is a FAT-TAIL
  SELECTION problem — mean +0.80pp/event but median −0.66pp, only 42.9%
  positive. The blanket rule wins on average only because 2 of 7 events
  are large (DELL 2022-03 +9.27pp dodged a −9.3% slide; DVA 2025-05
  +4.99pp); most thesis-break dips BOUNCE by month-end (BA 2017-07
  −2.60pp, COR 2022-11 −3.61pp, SBUX 2019-01 −2.18pp, HCA 2021-08
  −0.66pp; BA 2018-07 +0.41pp). The per-event ledger RECONCILES with
  M5's portfolio-level verdict (+0.05pp/yr ≈ 7 events × ~+0.8pp × ~10%
  position weight). A reviewer that separates "broken and falling" from
  "transient dip" beats both blanket rules; at ~0.6 candidates/year the
  C side accumulates slowly live (declared). The C prompt is frozen
  (`sellshadow-c/1`, parameter-free EXIT, fail-safe HOLD per the rule
  above) and the machinery is committed (`app/sellreview.py`).**

Contexts stay PIT-only and engine-appropriate: the factor-book events add
fundamental fields (ROE, F/P-FCF, 12-1 momentum rank, band distance, days
since the last filing) to the price/regime fields the 5A-1 schema already
carries. Prompts are versioned per event type (`sellshadow/1` = A; B/C
adapters follow). The evaluation discipline is unchanged: prospective
shadow mode only, no LLM calls on history, fail-safe to the engine's own
rule, contamination controls per §4 — and the LLM must beat BOTH the
engine default AND the best deterministic variant (V1/V2 for the sim; the
M-set winner for the factor book), net of honest fills and costs. A cheap
deterministic rule that captures the gain wins by construction.

## 5. Decision gate (nothing below happens without approval)

Proposed sequencing (each step gated, owner approval between phases):

- **5A-1** (committed — `app/sellshadow.py`, the context extractor with the
  2,906/2,906 bit-identical ratchet self-check, the frozen prompt +
  fail-safe parser, the defer ledger, the five deterministic exit
  variants): full-window verdict recorded — V1 signal exits +18.5%/yr vs
  BASE −2.2%/yr (same buys — the entire gap is the exit rule); V2 the
  drawdown-averse pick (27.4% maxDD); the crude portfolio breaker a dead
  end; blanket deferral +2.93pp/event (positive 9 of 12 years, 61% of
  defers never re-stopping within a quarter) — the LLM's bar is beating
  the blanket rule with selectivity AND beating V1/V2 at portfolio level.
- **5B** (committed — the M-set runner + full-window verdicts): M1/M2 are
  bit-identical no-ops on real data (the band only changes at month-end
  rebuilds — cadence cannot create intra-month exits; the per-name
  not-crash gate never bound), M3a vol-targeting is the decade's winner
  (+18.34%/yr, maxDD 33.3%, better on every axis), M4 catastrophe trails
  a dead end (worse return AND drawdown — per-name price stops
  contra-indicated platform-wide). The owner's intra-month sell rides on
  M5's filing-day hook and the 5C type-C alarms.
- **5C** (committed — the unified reviewer, complete): the per-engine
  event adapters (A via 5A-1, B and C above) + the daily `shadow`
  command (`python -m app.sellreview shadow` — detection-only by
  default; the deployed command accumulates the prospective evidence log)
  + the `assess` scorer (completed-window rule, realized per-event deltas
  vs the reference bars, fail-safe accounting — the LLM is graded ONLY on
  the forward log). The owner signed off the `sellshadow/1` prompt design
  and the 5C start; the engineering smoke (three canned-synthetic-context
  consults through the live proxy, temperature 0) passed clean — correct
  per-type decisions, rationales citing the context fields, no key
  material logged. Grading remains prospective-only per §4.
- In parallel, the §3 deterministic exit variants belong in the Phase 4C
  study runs.

No change to any live engine behavior is in scope for this research phase.

## References

- Kaminski, K.M. & Lo, A.W. (2014). "When do stop-loss rules stop losses?"
  Journal of Financial Markets 18, 234–254.
- Clare, A., Seaton, J., Smith, P.N. & Thomas, S. (2013). "Breaking into the
  blackbox: trend following, stop losses and the frequency of trading"
  (Cass/CUBS working paper 2012/11; the "trading can damage your wealth"
  result).
- Dai, M., Li, P. & Zhang, J. (2021). "Risk reduction using trailing
  stop-loss rules" (International Journal of Theoretical & Applied Finance).
- Shefrin, H. & Statman, M. (1985). "The disposition to sell winners too
  early and ride losers too long." Journal of Finance 40.
- Odean, T. (1998). "Are investors reluctant to realize their losses?"
  Journal of Finance 53.
- Barroso, P. & Santa-Clara, P. (2015). "Momentum has its moments."
  Journal of Financial Economics 116, 111–120.
- Daniel, K. & Moskowitz, T. (2016). "Momentum crashes." Journal of
  Financial Economics 122, 221–247.
- Research Affiliates. "Can momentum investing be saved?"
- FinCon: Yang et al. (2024). "FinCon: A synthesized LLM multi-agent system
  with conceptual verbal reinforcement" (NeurIPS 2024).
- TradingAgents: Xiao et al. (2024). arXiv:2412.20138.
- StockBench (2025). "Can LLM agents trade stocks profitably in real-world
  markets?" arXiv:2510.02209.
- Profit Mirage (2025). "Revisiting information leakage in LLM-based
  financial agents." arXiv:2510.07920.
- "The Alpha Illusion" (2026, position paper).
- Internal: `docs/ui-strategy-audit-2026-10-21.md` (Phase 4 spec, finding #2
  configs-tried, the LLM 3/2 rating), `docs/llm-strategy-experiments.md`
  (prior LLM experiments + contamination caveats).
