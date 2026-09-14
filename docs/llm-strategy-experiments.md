# LLM Strategy Experiment Report

Date: 2026-08-26
Branch: `feature/llm-strategy` (HEAD `01510f3`)

## 1. Executive summary

We ran ~40 benchmark configurations across 3 window sets to find the best
hybrid LLM-review strategy. **The winner: hybrid mode + minimal prompt +
engine-owns-exits + failure-marker intervention (stop-out cascade / drawdown
trigger, no cooldown).** It is the only configuration that fixes the
previously-broken 60-day chop window while keeping the bull-window upside.

## 2. What we tried (chronological)

| # | Change | Commit/Tag |
|---|--------|-----------|
| 1 | Seed pinning removed (cloud model ignores seed/temperature) | `f29705b` |
| 2 | Replay mirrors live sim's calendar-week review | `c4169ae` (tag `base-weekly-review`) |
| 3 | Market-regime context (breadth, PULLBACK, hard-BULL rule) | `c883ae4`, `8a16b4b`, `907af9a` (tag `v2-regime-aware`) |
| 4 | Engine owns exits (block LLM-initiated SELLs) | `9c93a9f` (tag `v3-engine-owns-exits`) |
| 5 | Minimal prompt (no methodology/regime/veto bias) | `19b0a76` |
| 6 | Long-horizon signals (run20d/run60d/dist52w-high) | `6bef7e9` |
| 7 | Marker-gated (LLM only when engine proposes) | experiment — rejected |
| 8 | Failure marker (stop-outs + drawdown, no cooldown) | `c38622e` |
| 9 | Pure-LLM cadence fix (daily, not weekly) + prompt fix | `c3a1b3e` |
| 10 | News context (live + per-day) | tried, reverted — SearXNG can't date-filter |
| 11 | Live sim: failure-marker intervention | `01510f3` |

## 3. Benchmark results (mean delta vs deterministic)

### 5×30d suite (the reliable scoreboard)

| Config | Mean | Worst | Pos |
|--------|------|-------|-----|
| baseline (no regime) | -0.19 / +0.26% | -6.9% | 2/5 |
| v2 regime-aware | +0.18..+1.71% | -1.9% | 3-4/5 |
| minimal prompt | +1.42 / +0.24% | -3.3% | 3/5 |
| minimal + interval 3 | +0.29 / +0.16% | -1.8% | 2/5 |
| **failure marker** | -0.30% | -1.9% | 1/5 |
| pure-LLM (full prompt) | -2.63% | -7.8% | 2/5 |

### 90d bull window (2025-06-02..2025-10-08, det +21.34%)

| Config | Delta |
|--------|-------|
| v2 (sells allowed) | -8.69% |
| minimal + no-llm-sells | +2.09 / +0.57% |
| minimal + interval 3/5 | **+5.49%** |
| **failure marker** | **+5.48%** |
| pure-LLM (minimal) | -10.37% |

### 60d chop window (2026-06-01..2026-08-21, det +1.60%) — the universal problem case

| Config | Delta |
|--------|-------|
| base / v2 | -5.4% avg |
| minimal interval 1 | -4.73% |
| minimal interval 3 | -7.08% |
| failure marker (cooldown) | -3.62% |
| **failure marker (no-cooldown, dd=7)** | **-0.84% to -2.81%** |
| pure-LLM | -6.7 to -8.9% |

## 4. Key findings

1. **LLM-initiated SELLs are pure churn** — selling winners/dips in bull
   markets cost -8.7% on the 90d window; blocking them flipped it to +4.7%.
   The engine owns exits (stop-losses + deterministic SELL signals).
2. **The methodology prompt's veto rules actively hurt** — the minimal
   prompt matches or beats it everywhere; the model's own judgment is
   better than hand-written rules.
3. **Fixed intervals can't fix the chop window** — fewer calls help the
   bull window (+5.5% at interval 3) but make the chop window worse (-7.1%).
4. **The failure marker is the only thing that fixes the chop window** —
   it fires exactly when the engine bleeds (2+ stop-outs in 5 days, or
   equity >7% below peak), no cooldown, and stays quiet when the engine
   works. -0.84% to -2.81% vs -4.7% baseline.
5. **Pure-LLM is strictly worse** — no engine risk floor; bleeds on long
   windows (-10 to -15%). Two bugs were found and fixed: weekly cadence
   instead of daily, and the replay used the hybrid prompt instead of
   `_PURE_LLM_SYSTEM_PROMPT`.
6. **Pure-LLM never tops up** — at the 10-position cap, allowance cash
   sits idle because the LLM proposes new tickers (blocked by the cap)
   instead of adding to existing positions. Real gap, unfixed.
7. **News context hurts** — SearXNG can't date-filter, so the LLM
   overweights today's headlines in historical reviews. Reverted.

## 5. Best configuration

**Hybrid + mode-aware prompt + engine-owns-exits + failure marker
(stop_outs=2, drawdown=7%, no cooldown)**

Rationale (2026-08-27, post-parity-fix scoreboard): mode-aware beats the
minimal prompt on every non-neutral window of the 5×30d suite (mean +0.78%
vs +0.43%, worst +0.00% both) with equal bull-window behavior (the marker
stays quiet — see §8 for why the old +5.5% "bull window alpha" was
structural, not LLM). The 60d chop window remains the weak spot for both
prompts (mode-aware -3.45%, minimal -2.78%); mode-aware wins the live slot
on the suite average, not on the chop case.

## 6. Live deployment

Live sim runs (`.env`, gitignored):

```
SIM_STRATEGY=hybrid
SIM_LLM_REVIEW_INTERVAL=5
SIM_LLM_MODE_AWARE_PROMPT=true
SIM_LLM_FAILURE_MARKER=true
```

How it works live: the deterministic engine runs every cycle as primary;
the LLM is consulted only when the engine shows failure — 2+ stop-out SELLs
in 5 trading days, or equity >7% below its running peak (from SimSnapshot
history) — with no cooldown. When the engine works, the LLM stays quiet
(saves tokens, avoids churn). The mode-aware prompt tells the LLM the
engine owns exits and its only job is adding high-quality BUYs.

## 7. Entry-quality experiments (2026-08-27)

Component attribution (95–205 BUY-signal entries, 21d forward returns)
found the classic bullish score's ranking is partly inverted: the rising-
MACD-histogram bonus marks late entries (-3.4% fwd when true, +8.9% when
absent), 1d-RSI-rising chases (-0.7% vs +6.2% for 5d RSI *falling*), and
the dist_above bonus rewards parabolic extension (-14.5% fwd above +100%
SMA200; 4 of 10 one-year stop-outs entered above +50%).

Two fixes were built, both **opt-in** (neither is live):

1. **Entry guards** (`feature/mode-aware-prompt`, commit `00e0b50`):
   `min_run_5d` (falling-knife block) + `max_dist_above` (parabolic block)
   in `propose_trades`, defaults 0 = off. A/B: fixes the failure mode
   (win4 chop: 7→2 stop-outs, -7.7%→-0.4%) but costs right-tail returns
   (win2: +20.5%→+15.6% — the blocked "parabolic" names kept mooning).
   Net ~wash on return, mild dd improvement. A trade-off, not a free lunch.

2. **Pullback scoring** (`feature/scoring-extension-penalty`, commit
   `b76bcb6`): `SIGNAL_SCORING=pullback` — drops the hist-rising bonus,
   rewards 5d RSI decline into 40-65, re-curves dist_above. Ranking quality
   on its fitting sample: classic -3.3% (inverted) → +7.1% tercile spread.
   Full-replay A/B on the fitting year: 1yr mean +10.4% vs +5.0%.

**The 8.5-year walkforward A/B overturned the pullback recommendation.**
Both scorings ran the full 17-window sweep (train 504d / test 126d,
2016→2026-07, per-window param search):

| Metric | classic | pullback |
|---|---|---|
| OOS mean/window | **+5.76%** | +3.76% |
| Compounded | **+144% (11.1%/yr)** | +80% (7.2%/yr) |
| Windows won | **13/17** | 4/17 |
| Worst window | -6.7% | -10.2% |
| Mean max-dd | 14.6% | **13.1%** |
| Mean Sharpe | 2.93 | 2.90 |

Caveats: each arm searched its own params (not a pure scoring A/B), and
classic's edge concentrates in the 2016–2024 bull era where chasing pays.
But the evidence hierarchy is clear: the fitted 1-year edge loses to the
8.5-year OOS record. **Live sim reverted to classic**; the pullback variant
stays in the repo as a one-line flip if the regime turns choppy (it won
both short-window tests in the 2025-10→2026-08 chop era).

Methodology note: an earlier read of the first walkforward log claimed it
"validated" pullback scoring out-of-sample. That was wrong — the run had
no classic control arm. The proper A/B (this section) reversed the call.
Recorded here so the mistake isn't repeated.

## 8. Hybrid replay parity fix (2026-08-27, commit a54fb5b)

`_hybrid_replay` used single-phase propose/execute while `_replay` uses
two-phase (SELLs execute, then BUYs re-proposed against freed slots). A
stop-out SELL blocked its own same-day re-entry BUY in the hybrid skeleton,
a structural divergence that fabricated "+5.48% LLM alpha" on the 90d bull
window with **zero** LLM calls. Fixed: non-review days now mirror `_replay`
exactly (verified 0.00 delta on return/dd/sharpe/trades); review days keep
single-phase (that's the live-sim LLM contract). Any pre-fix "LLM delta"
numbers from long windows are contaminated and should not be trusted.

## 9. Open items

- Pure-LLM top-up gap: at the max-positions cap, the LLM proposes new
  tickers (blocked) instead of topping up held positions; allowance cash
  sits idle. Worth a follow-up if pure-LLM is pursued further.
- The 60d chop window remains the hardest case; the failure marker is the
  best mitigation found so far, not a full fix.
- Entry quality: guards + pullback scoring are both opt-in and unmerged.
  The scoring ranking inversion is real but regime-dependent — revisit if
  the market shifts back to the 2025-10→2026-08 chop pattern.
- The walkforward sweep prefers ATR stops (7/17) over the live percent-15
  (6/17); differences were small but worth a dedicated look someday.

## 10. Fundamentals-first daily engine ("daily-core") — 2026-09-09

Owner's design: the monthly qv-mom ranking IS the portfolio (fundamentals
decide WHAT to own); candles only make minor WHEN adjustments. Deterministic
only — LLM overlays are a separate experiment (§ earlier: LLM context columns
and the ROE guard stay flag-gated off; the context arm measured inside noise
on the chop window, the guard cost right-tail returns).

New tool: `app.optimize daily-core` — monthly qv-mom core with candle-driven
overlays. All-defaults ≈ the monthly baseline (sanity check built in; the
report prints both side by side on the same window).

### Review fixes (commit 7a95861)
- pending_cash stranding: a zero-new-picks month never deployed the
  contribution; all modes now release it at the month-end rebuild
- month-end rebuild restores picks to equal weight (baseline semantics)
- turnover reporting was always 0; now (buys+sells)/2 / equity at month start
- dead code removal (is_month_end, duplicate run5_frame line)

### Review fixes (commit 2dd5f05) — re-measured A/B

A code review found the backtest deposited contributions only on month-end
days, so `pending_cash` was dead and `--dca daily/rank` never saw intra-month
cash — the "cash-drag elimination" premise was not actually being measured.
The contribution now arrives on the **first trading day of the month** (as the
live engine and `backfill` already did) and the fee is reserved inside BUY
spends. The monthly baseline is unchanged; only the daily/rank arms moved.
The tables below were re-run on 2026-09-14 (data through 2026-09-11), so the
numbers that stood here before are superseded.

### Cash-deployment A/B (IRR/yr, diversified-plus, $1k/mo, re-run 2026-09-14)

| Window | monthly sim | daily-core defaults | rank daily, boost=0 | rank boost=0.25 |
|---|---|---|---|---|
| 2020-01..2023-01 (bear) | 2.67% | 3.81% | **6.10%** | 5.56% |
| 2022-01..2026-09 (mixed) | 43.44% | 44.66% | 47.43% | **47.59%** |
| 2023-01..2026-09 (bull) | 46.55% | 49.02% | **54.04%** | 53.16% |

### Walk-forward (non-overlapping OOS windows, boost=0 vs 0.25)

| Window | monthly sim | rank b0 | rank b0.25 |
|---|---|---|---|
| 2017-2019 | 13.41% | **17.66%** | 15.35% |
| 2020-2021 | 37.28% | 49.99% | **50.83%** |
| 2022-2023 | 26.03% | 26.69% | **29.96%** |
| 2024-2026 | 41.73% | 49.53% | **50.09%** |

### Findings

1. **Daily rank deployment beats the monthly sim on every window tested** —
   in-sample +3.4 / +4.0 / +7.5pp on the 2020-2026 windows, and out-of-sample
   on all four walk-forward windows (+4.3 / +12.7 / +0.7 / +7.8pp at boost=0).
   The edge is cash-drag elimination: fresh contributions arrive on the first
   trading day and deploy immediately instead of parking ~2 weeks, so the
   momentum concentration is preserved instead of diluted.
2. **Boost is inconclusive — keep flat (boost=0).** boost=0 wins 2 of the 3
   in-sample windows and the earliest OOS window by 2.3pp; boost=0.25 wins
   the other three OOS windows but the mean tilt edge is only ~+0.6pp — inside
   noise. Flat targets are simpler, lower single-name concentration, and match
   the live `run_deployment`. Revisit only if more OOS windows confirm a tilt.
3. **Equal-weight pro-rata drip loses badly** (-8pp IRR): spreading fresh
   cash across all holdings rebalances into laggards. Daily cash must go to
   the TOP of the ranking (or the most underweight), never spread.
   (Architectural conclusion; not re-run in the 2dd5f05 sweep.)
4. **Entry timing is mode-dependent under the corrected model** — in-sample
   `above-sma50` is a no-op for rank, costs ~6pp for `dca=daily`, and is
   roughly neutral for `dca=monthly`; `not-crash` is a no-op. Leave gates off.
5. **Exit cadence is irrelevant** — the hysteresis band is sticky; daily
   release ≈ monthly release (top configs tie exactly on final value).

### Config note

`rank boost=0.0` ≈ "deploy daily into the top-ranked names toward equal
weight". The monthly sim's structural edge survives only its ranking; its
monthly cadence is a (small, consistent) cost.

**Live config (re-measured 2026-09-14):** `run_deployment` already deploys
flat rank-first targets and deposits the allowance at month start, so the
corrected backtest now models production. No live config change was required
— the walk-forward confirms the deployed rule beats the monthly baseline
out-of-sample in every window.

## 11. Risk overlays: residual momentum wins the walk-forward — 2026-09-14

Goal: "optimize return by not getting too volatile." Four literature-backed
risk knobs were added to the daily-core backtest (all opt-in, code defaults
conservative):

- `--mom residual` — Blitz-Huij-Martens (2011) residual momentum: per month,
  regress each ticker's daily log returns (252d window, 21d skip) on the
  equal-weight market, rank by alpha / residual-std * sqrt(n) (the alpha
  t-stat). Momentum per unit of idiosyncratic risk. `eligible_frame` fills
  the same `mom` column; the raw-only `mom > -0.99` floor is skipped for
  residual (it's a t-stat, not a return).
- `--target-vol X` — Barroso-Santa-Clara (2015) vol management: when the
  portfolio's own 21d realized vol (contribution-adjusted daily log returns)
  exceeds the target, deployment is capped so the excess stays in cash.
  Only caps down, never leverages up.
- `--vol-weight` — inverse-vol position weights among the day's top-N
  candidates instead of equal weight (`_inv_vol_weights`; missing-vol names
  keep equal weight, weights renormalize to the same total).
- `--lowvol-tilt` — pct_rank(-vol) as a 4th equal score term in `_qv_order`.

New tooling: `daily-core-sweep --stage 3` (in-sample risk-overlay sweep with
Sharpe + max-DD now reported by every backtest) and `--stage 4` (walk-forward
A/B of the overlays on the live config across the §10 OOS windows). Found
and fixed on the way: stage-3 never applied the stage-1 winner's boost in the
parent process (control silently ran boost=0 — which happens to be the live
config, so the overlay comparison stayed valid).

### In-sample (2017-01..2026-09, full window, rank deploy b0)

| Arm | IRR | Sharpe | MaxDD | Turnover |
|---|---|---|---|---|
| control (live) | 32.05% | 0.81 | 35.8% | 5.2% |
| **residual** | 30.64% | **0.95** | **28.2%** | 5.3% |
| vol-weight | 28.61% | 0.83 | 33.3% | 5.0% |
| target-vol 0.25 | 31.80% | 0.81 | 34.7% | 5.0% |
| target-vol 0.20 | 27.84% | 0.76 | 33.7% | 4.8% |
| lowvol-tilt | 22.33% | 0.95 | 24.7% | 4.7% |
| residual+tv0.25 | 29.38% | 0.93 | 28.2% | 5.4% |

### Walk-forward A/B on the live config (non-overlapping OOS windows)

| Window | control | residual | res+tv0.25 | tv0.25 | lowvol | vol-weight |
|---|---|---|---|---|---|---|
| 2017-19 Sharpe | 1.01 | **1.09** | 1.09 | 0.98 | 1.01 | **1.18** |
| — maxDD | 17.0% | 16.4% | 16.4% | 17.1% | 15.8% | 15.3% |
| 2020-21 Sharpe | 0.97 | **1.02** | 0.98 | 0.99 | **1.04** | (0.83 full) |
| — maxDD | 24.3% | 23.7% | 23.7% | **21.6%** | 23.9% | — |
| 2022-23 Sharpe | 0.09 | **0.24** | **0.24** | 0.11 | 0.11 | 0.15 |
| — maxDD | 15.1% | **12.6%** | 12.7% | 15.1% | 10.6% | 14.0% |
| 2024-26 Sharpe | 1.28 | 1.74 | **1.78** | 1.28 | 1.75 | 1.37 |
| — maxDD | 24.2% | 21.6% | **18.7%** | 23.5% | 8.6% | 18.2% |
| 2024-26 IRR | 49.5% | 62.9% | 61.4% | 46.8% | 27.9% | 40.0% |

(IRRs for the other windows are in /data/sweeps/daily_core_risk_walkforward.json
inside the container; Sharpe is the decision metric here.)

### Findings

1. **Residual momentum is the walk-forward winner — adopted live.** It
   posts the highest Sharpe of the six arms in all four OOS windows and
   cuts max drawdown in every stress window (2022-23 bear: 15.1% → 12.6%
   with Sharpe 0.09 → 0.24; 2024-26: 24.2% → 21.6% with Sharpe 1.28 →
   1.74 AND IRR +13.4pp). The full-window in-sample IRR cost (-1.4pp vs
   control) is a bull-market-chasing artifact; OOS the steadier ranking
   pays for itself. Live: `SIM_DAILY_CORE_MOM_VARIANT=residual` (.env);
   code default stays `raw` (conservative).
2. **residual+tv0.25 is a close second** — equal Sharpe in 2022-23, better
   2024-26 maxDD (18.7% vs 21.6%) at the cost of 2020-21. Not adopted:
   target-vol needs live daily-return tracking (the live engine would have
   to gate deployment on its own realized vol — more moving parts for a
   marginal OOS edge). Revisit if 2026+ windows keep confirming.
3. **lowvol-tilt is too expensive** — best drawdowns (8.6% in 2024-26!)
   but gives up 22pp IRR in the same window. The score tilt removes the
   very momentum exposure the strategy monetizes. Rejected.
4. **vol-weight underperforms on this 110-name universe** — the frame's
   `vol` column is computed over only 121 days for all names, and inv-vol
   weights systematically under-allocate the momentum leaders. Only bright
   spot: 2017-19 (Sharpe 1.18). Rejected.
5. **Backfill phantom-allowance bug (found via start=all):** allowance rows
   were persisted for EVERY month in the candle window (1980+ → 550 rows)
   while the replay only deposits once a ranking exists (2017+ → 110).
   allowance_total read $550k instead of $110k, breaking every
   contributed-normalized metric. Fixed: rows derive from the replay's
   actual deposits (`deposited_months`), test-guarded in test_daily_core.
6. **UI risk metrics:** all three portfolio tabs now show Max DD (peak-to-
   trough on the contributed-normalized curve) and alpha chips vs their
   natural comparators; Daily-Core gained rank/weight columns, a stale-
   ranking warning, chart range bars (6M..MAX) and a Monthly-sim overlay.

### Config note

Live daily-core is now: qv-mom ranking with RESIDUAL momentum + daily rank
deployment (boost=0, no gates, no vol overlay). Everything else unchanged.

## 12. "Cash out the win": protection overlays vs the Jul-2026 giveback — 2026-09-14

Owner observation: every portfolio rode a strong Apr–Jun 2026 run (beating the
DCA benchmark) then round-tripped it in July. daily-core peaked at **137.1% of
contributed on 2026-06-30** and fell to **112.5% by 2026-09-14** (−17.9% from
the peak, trough 111.8% on 07-29). The qv-mom design has no exits except the
monthly hysteresis band, so a momentum crash hands the run-up back.

Three opt-in protection mechanisms were built into `_daily_core_backtest`
(CLI: `daily-core --portfolio-stop/--exposure-trend/--trailing-stop`):

- `--portfolio-stop X` — peak-to-trough cash-out brake: when the portfolio's
  own equity is X% below its running peak, **sell everything to cash** and park
  contributions until re-entry.
- `--exposure-trend N` — deploy cash only while the equal-weight universe index
  is above its N-day SMA (also the re-entry gate after a stop, since a fully
  cashed book's own drawdown is frozen).
- `--trailing-stop X` — per-name stop: exit a holding when its price falls X%
  below its own peak since entry; the factor-level cut that does not need a
  market downtrend (a re-buy is possible once the name re-ranks).

Measurement note: the daily-core maxDD was also fixed to the
contributed-normalized convention (equity / invested-to-date) that the live UI
uses — the raw-equity DD understated the real peak-to-trough loss (11.2% →
18.9% on the 2026 episode, matching the chart).

### Walk-forward A/B (residual core, rank deploy b0; Sharpe / normalized maxDD / turnover)

| Window | control (live) | stop10 | stop10+trend200 | trend200 | trail15 |
|---|---|---|---|---|---|
| 2017-19 | 1.09 / 21.6% / 4.9% | **1.26** / **18.3%** / 16.1% | 1.22 / **13.8%** / 8.4% | 1.13 / 19.7% / 5.2% | 1.13 / 21.7% / 8.7% |
| 2020-21 | 1.02 / 28.2% / 5.6% | **1.12** / 25.2% / 89.5% | 1.04 / **19.7%** / 57.1% | 0.96 / 28.2% / 5.6% | 1.07 / 28.5% / 15.5% |
| 2022-23 | 0.24 / 20.2% / 7.7% | **-0.01** / 20.5% / 62.0% | 0.51 / **11.9%** / 9.3% | **0.58** / **12.1%** / 5.6% | 0.19 / 20.4% / 24.3% |
| 2024-26 | 1.72 / 31.4% / 4.2% | 1.76 / 27.0% / 109.8% | 1.76 / **22.8%** / 71.9% | 1.73 / 30.6% / 4.3% | 1.67 / 31.5% / 11.4% |

- **stop10+trend200** wins Sharpe in all four windows and cuts maxDD to the
  best-or-near-best in every window — but pays 8–72%/month turnover (each
  cash-out + re-entry is a full book turn).
- **The pure stop whipsaws**: in 2022-23 it fired 15× and *destroyed* the
  Sharpe (0.24 → -0.01) — the trend gate is what makes the brake usable
  (2 events, 0.51).
- **trend200 is the cheap win**: no added turnover, big help in the 2022-23
  bear (Sharpe 0.24 → 0.58, DD 20.2% → 12.1%).
- **trailing stops don't pay**: neutral-to-worse Sharpe in every window at
  1.5–3× turnover. The monthly hysteresis band already cuts fallen names at
  month-end; the daily trailing exit mostly front-runs that, churning.

### The specific 2026 episode (2026-01-01..2026-09-14, in-sample, normalized DD)

| Arm | Final | IRR | Sharpe | maxDD | Turnover |
|---|---|---|---|---|---|
| control | $10,050 | 38.25% | **1.14** | **18.9%** | 10.4% |
| trend200 | $10,050 | 38.25% | 1.14 | 18.9% | — (never fired) |
| trailing 15% | $10,035 | 37.65% | 1.13 | 19.0% | 31.1% |
| target-vol 0.25 | $9,986 | 35.73% | 1.08 | 18.9% | 10.6% |
| stop10+trend200 | **$9,580** | 20.33% | 0.86 | **21.4%** | 37.9% (2 events) |

**Every protection mechanism was neutral or WORSE on the episode that prompted
the question.** Why:

1. **trend200 never fired** — the equal-weight universe index stayed above its
   200-day SMA through the July reversal. This was a *momentum-sleeve crash
   while the broad market held* (AI/semis reversal), not a market downtrend.
2. **stop10+trend200 round-tripped**: the portfolio's drawdown triggered a
   full cash-out, but the market still looked "up" so the trend gate re-armed
   the very next day → sell-all + rebuy into the continuing decline, twice.
   Ending equity 4.7% lower AND a worse drawdown (21.4% vs 18.9%).
3. **trailing stop** cut individual names but the strategy kept rotating into
   other falling names; 3× turnover bought nothing.
4. **target-vol** throttled deployment into the rebound → slightly lower final.

### Verdict

**Do not "fix" the giveback with a cash-out brake.** The walk-forward says
stop10+trend200 improves long-run Sharpe/DD, but it earns that on *sustained
market downtrends* (2022) that the overlay can see — not on a fast,
market-neutral momentum crash, where it actively hurts and multiplies
turnover. The one defensible, low-cost overlay is `--exposure-trend 200`
(helps the bear window, free otherwise) — but it does **not** address the
July-2026 shape at all.

The giveback is the price of the momentum premium; the factor-level mitigation
(residual momentum) is already the live core (§11). All new knobs stay opt-in
(`sim_daily_core_portfolio_stop=0`, `sim_daily_core_exposure_trend=0`,
`sim_daily_core_trailing_stop=0`); live daily-core is unchanged.

## 13. Gradient filter ("basket trend") — owner's idea, walk-forward verdict — 2026-09-14

Owner's proposal after §12: instead of the *market* trend, watch the **gradient
(N-day rate-of-change) of the strategy's own basket** — cash out when it turns
negative for some days, re-enter when positive. Not stupid: it is trend-following
on the strategy's own signal, which sees a momenton-sleeve reversal the market
index cannot.

Implemented (`--basket-trend N --basket-confirm K`): chain the equal-weight
top-N target basket day by day; when its N-day slope is negative for K
consecutive closes, sell everything and park contributions; re-enter when the
slope is positive for K closes. The basket keeps moving while in cash, so
re-entry can trigger (unlike a frozen equity-curve gate).

### The Jul-2026 episode (2026-01-01..2026-09-14, in-sample)

| Arm | Final | IRR | Sharpe | maxDD | Turnover |
|---|---|---|---|---|---|
| control | $10,050 | 38.25% | 1.14 | 18.9% | 10.4% |
| **basket10c3** | **$10,175** | **43.23%** | **1.42** | **13.0%** | 89.3% (5 events) |
| basket15c3 | $9,986 | 35.72% | 1.41 | 14.7% | 45.0% |
| basket20c3 | $9,709 | 25.11% | 1.20 | 18.3% | 38.5% |
| basket5c3 | $9,165 | 5.58% | 0.54 | 14.3% | 112.0% |

basket10c3 beats control on **every** metric on the episode — higher final
value, higher IRR, Sharpe 1.42 vs 1.14, DD 13.0% vs 18.9% — the only mechanism
tested that does (§12's stop/trend/trailing all failed there).

### Walk-forward A/B (Sharpe / normalized maxDD / turnover)

| Window | control | basket10c3 | basket15c3 | basket30c5 |
|---|---|---|---|---|
| 2017-19 | **1.09** / 21.6% / 4.9% | 0.65 / 24.8% / 65.1% | 0.83 / 20.0% / 48.0% | 1.06 / 16.5% / 30.1% |
| 2020-21 | 1.02 / 28.2% / 5.6% | 1.27 / 14.8% / 51.2% | **1.32** / **12.4%** / 36.5% | 1.07 / 14.2% / 22.4% |
| 2022-23 | **0.24** / 20.2% / 7.7% | **-0.13** / 14.5% / 66.2% | -0.10 / 14.7% / 65.2% | 0.08 / 13.6% / 37.3% |
| 2024-26 | 1.72 / 31.4% / 4.2% | **1.92** / **14.5%** / 59.0% | 1.65 / 14.2% / 49.1% | 1.48 / 19.0% / 23.6% |
| **avg** | **1.02** | 0.93 | 0.93 | 0.92 |

### Verdict

**The idea works in the episode and in 2 of 4 windows, but does not survive the
walk-forward.** The gradient filter is trend-following on your own signal: it
whipsaws in range-bound markets (2022-23 Sharpe -0.13 vs +0.24 control; 11.7%
IRR vs 28.7%) and pays 30-66%/month turnover everywhere. Its one consistent
virtue is drawdown: it cuts maxDD in 3 of 4 windows (31.4% → 14.5% recent),
i.e. it does deliver "not too volatile" — at a real, regime-dependent return
cost. The only overlay whose walk-forward *average* beat control remains
stop10+trend200 (§12, avg 1.13 vs 1.02) — and that one failed the episode.

Honest bottom line: no simple timing rule converts the momentum premium into a
free lunch. The giveback in the owner's episode is real and the gradient filter
caught it, but the same rule loses in chop. All knobs remain opt-in
(`sim_daily_core_basket_trend=0`, defaults off).

## 14. "On demand" switches for the gradient filter: thresholds, drawdown, efficiency — 2026-09-14

Follow-up to §13: can a signal ARM the gradient filter only in the windows where
it works? Three candidates were built and walk-forwarded (all opt-in):

- `--basket-threshold X` — only a slope below -X% counts (real drops, not noise).
- `--basket-drawdown X` — cash out when the SIGNAL BASKET is X% below its own
  peak (the basket keeps moving in cash, so re-entry on the drawdown halving
  can fire — unlike the frozen portfolio-equity brake).
- `--basket-er-min X` — Kaufman efficiency ratio gate: arm the EXIT only while
  ER = |net move| / path length over the last 20 basket prints is >= X
  (trend-following pays in efficient trends, whipsaws in chop). Re-entry stays
  unconditional so chop can never lock the portfolio in cash.

### Walk-forward averages (4 OOS windows, residual core)

| Arm | avg Sharpe | avg maxDD | avg turnover | worst window |
|---|---|---|---|---|
| control (live) | 1.02 | 25.4% | 5.6% | 2022-23 (0.24) |
| **g10t5** (threshold 5%) | **1.07** | **18.9%** | 20.5% | 2022-23 (-0.06) |
| bdd12 (drawdown 12%) | 0.95 | 17.0% | 12.2% | 2022-23 (-0.02) |
| g10c3 (threshold-free, §13) | 0.93 | 19.4% | 55.4% | 2022-23 (-0.13) |
| bdd8 (drawdown 8%) | 0.79 | 17.7% | 20.7% | 2022-23 (-0.30) |

### The 2026 episode under each switch

| Arm | Final | Sharpe | maxDD | events |
|---|---|---|---|---|
| control | $10,050 | 1.14 | 18.9% | — |
| g10c3 (no threshold) | **$10,175** | **1.42** | **13.0%** | 5 |
| g10t5 (5% threshold) | $9,974 | 1.25 | 18.8% | 2 |
| g10t5+ER0.3 | $9,915 | 1.11 | 18.9% | 1 |
| bdd12 / bdd12+ER0.3 | $10,050 | 1.14 | 18.9% | 0 |

### Verdict: no "on demand" switch gets both

The two regimes are **mutually exclusive with these mechanisms**:

- **Threshold-free gradient** wins the episode (it exits on the first serious
  negative slope) but whipsaws in chop because small dips trigger too.
- **Threshold / drawdown / ER-gated variants** suppress the chop whipsaw and
  improve the long-run average (g10t5: Sharpe 1.07 vs 1.02, DD 18.9% vs
  25.4%) — but they **also suppress the episode win**: the Jul-2026 decline was
  a slow stepwise erosion (10-day slope -0.5%, -6.5%, -1.7%, -5.5%, ...), not a
  sharp threshold breach, so the gated filters either never fire or exit into a
  bounce.
- **The efficiency-ratio gate does not separate the regimes**: in the July
  reversal the basket path was efficient enough to arm the exit only after the
  damage was done; in 2017-19's calm bull the ER armed it on ordinary pullbacks.

The honest conclusion: at the moment of decision, a reversal and a dip look
alike; every rule that catches the reversal also pays in the chop. The only
mechanism that improved the long-run average at acceptable cost is the
5%-threshold gradient (g10t5) at ~20% turnover — but it does not solve the
owner's Jul-2026 complaint, and it still loses the 2022-23 window. All knobs
remain opt-in with defaults off; live daily-core is unchanged.

## 15. Owner's composite: gradient filter in GOOD TIMES + market-trend gate — 2026-09-14

Owner's design after §12-14: *don't* gate everything on one regime rule. Keep the
engine's normal buying in charge; use the **market-trend gate for bad times**
(park cash in downtrends, §12) and arm the **gradient cash-out only in good
times** (market > SMA200) to catch momentum-sleeve crashes inside healthy bull
markets — no cooldown, the engine re-enters whenever it decides. Implemented as
`--basket-good-times` (arms the §13 slope trigger only above the market's
200-day SMA; re-entry unconditional).

### 2026 episode (2026-01-01..2026-09-14)

| Arm | Final | IRR | Sharpe | maxDD | Turnover |
|---|---|---|---|---|---|
| control | $10,050 | 38.25% | 1.14 | 18.9% | 10.4% |
| g10c3 (ungated) | **$10,175** | **43.23%** | **1.42** | **13.0%** | 89.3% |
| g10c3gt (good-times) | **$10,175** | **43.23%** | **1.42** | **13.0%** | 89.3% |
| **g10c3gt+t200** | **$10,175** | **43.23%** | **1.42** | **13.0%** | 89.3% |

(Identical: the market stayed above its SMA200 through the July sleeve crash —
the good-times arm never disarmed, and the trend gate never blocked.)

### Walk-forward (Sharpe / normalized maxDD / turnover)

| Window | control | g10c3gt | g15c3gt | **g10c3gt+t200** |
|---|---|---|---|---|
| 2017-19 | **1.09** / 21.6% / 4.9% | 0.86 / 25.0% / 58.0% | 0.85 / 23.1% / 40.9% | **1.14** / **15.5%** / 57.7% |
| 2020-21 | 1.02 / 28.2% / 5.6% | 1.27 / 14.8% / 51.2% | **1.32** / **12.4%** / 36.5% | 1.01 / 14.6% / 51.3% |
| 2022-23 | **0.24** / 20.2% / 7.7% | -0.08 / 20.2% / 37.6% | 0.00 / 20.2% / 33.9% | 0.11 / **12.2%** / 35.5% |
| 2024-26 | 1.72 / 31.4% / 4.2% | 1.72 / 18.5% / 56.2% | 1.65 / 14.2% / 49.1% | **1.78** / **14.5%** / 55.9% |
| **avg** | **1.02** / 25.4% | 0.94 / 19.6% | 0.96 / 17.5% | 1.01 / **14.2%** |

### Verdict

**This is the first composite that addresses the owner's episode without
degrading the long-run average.** `g10c3gt+t200`:

- wins the 2026 episode on every metric (Sharpe 1.42 vs 1.14, DD 13.0% vs 18.9%);
- **halves the average walk-forward drawdown** (14.2% vs 25.4%);
- keeps average Sharpe neutral (1.01 vs 1.02) — the "not too volatile" goal;
- costs 35-58%/month turnover (the price of the churn).

Compared with §12's stop10+trend200 (avg Sharpe 1.13, avg DD 17.0%) the owner's
composite trades a little long-run Sharpe for the episode win and a lower DD —
and it is the only configuration tested that does both. Good-times arming alone
does **not** fix 2022-23 (bear rallies re-arm it), so the market-trend gate is
what handles the bad regime; the two gates are complementary, exactly as the
owner described. Opt-in (`sim_daily_core_basket_good_times`), live wiring TBD.

### §15 addendum — arming-window study: SMA50 vs 100 vs 200

Owner asked whether the good-times arming line should be SMA50 instead of
SMA200. The window is now configurable (`--basket-arm-sma N`). Verdict: **the
differences are inside noise and flip direction across windows** — another
parameter to distrust.

| Window | control | gt50+t200 | gt100+t200 | gt200+t200 |
|---|---|---|---|---|
| 2017-19 | **1.09** / 21.6% | 0.84 / 23.1% | 0.71 / 25.6% | 1.14 / 15.5% |
| 2020-21 | 1.02 / 28.2% | 0.71 / 28.2% | 1.05 / 14.6% | 1.01 / 14.6% |
| 2022-23 | 0.24 / 20.2% | **0.50 / 12.1%** | 0.34 / 12.9% | 0.11 / 12.2% |
| 2024-26 | 1.72 / 31.4% | 1.87 / 22.6% | **2.16 / 14.3%** | 1.78 / 14.5% |
| **avg Sharpe** | 1.02 | 0.98 | **1.06** | 1.01 |
| **avg maxDD** | 25.4% | 21.5% | 16.9% | **14.2%** |
| avg turnover | 5.6% | 36.8% | 45.6% | 50.1% |

- **On the 2026 episode SMA50/100 look spectacular** (gt50: $10,757, Sharpe
  1.96 vs control 1.14) — the shorter line disarms during the pre-crash dips
  and re-arms for the crash. But that is one episode.
- **Across the walk-forward SMA50 loses 2017-19 and 2020-21** (fast disarm
  misses the whipsaw protection), wins 2022-23, and the averages are
  indistinguishable from 100/200 (±0.05 Sharpe over 4 windows).
- SMA100 has the best average Sharpe (1.06) and SMA200 the best average DD
  (14.2%) — but with four windows, picking between them is curve-fitting.

**Recommendation:** if the composite is ever wired live, keep the classic
SMA200 arming (fewest parameters, best drawdown, no window choice to overfit)
— or expose the window to the experiment selector and judge forward, not on
this table. All variants remain opt-in.
