# Phase 4C — the strategy-audit study: skill vs selection bias (2015–2026)

**Status date: 2026-10-31.** The Phase 4 flagship study — the multi-regime,
controls-and-uncertainty revalidation the audit predeclared ("Revalidate, then
prune strategies") — ran to completion on the read-only snapshot. This report
is the analysis of that study and the input to the Phase 4D prune decision.
No winner language; numbers and predeclared bars only (audit finding #4).

- Study: `python -m app.reval study` (the Phase 4B predeclared driver), schema
  `reval-study/1`, artifacts `/tmp/opencode/reval/study4c/reval-study.{json,md}`
  (JSON 73 MB, MD 3,144 lines).
- Data: `/tmp/opencode/reval/trading_reval.db` (read-only, `PRAGMA
  query_only=ON`), sp500 universe (503 members), span 2015-01-02 .. 2026-10-30
  (2,977 sessions). Row counts + `quick_check` re-verified unchanged after the
  run (candles 1,581,262, ok).
- Grid: windows `full` + 11 `yearly` + 2 `halves` (2015-01-02..2020-11-27 and
  2020-11-30..2026-10-30) × cases `contribution` ($1,000/month) and `fixed`
  ($12,000 lump at the window's first session) = 30 window×case runs,
  **2 infeasible** (yearly-2015, both cases: momentum warmup — 252 loaded
  sessions < 253 required; the engine's honest refusal, recorded as
  infeasible, contributing no arms/configs/uncertainty — the 4696052 fix
  working exactly as designed; the overlapping full/halves windows still cover
  that period).
- Controls and uncertainty: `random_n` 100 seeds (per month-end:
  `sha256(seed|month_end)`-derived, process-independent, 10 random eligible
  names), `equal_weight` (the full eligible set, the selection bar), circular
  block bootstrap B=2000 (block 21 sessions) on annualized excess return /
  excess Sharpe / vol, the predeclared 7-variant OFAT grid (stock bps {5,20},
  cash yield {0.00,0.05}, mom_residual, target_n 20, hold_band 40 —
  monthly/daily_core arms only), `configs_tried` **3,024** recorded per
  window×case and in total (audit finding #2).
- Costs (declared): stock 10 bps one-way on fill notional, ETF 2 bps, FX +15 bps
  on non-USD listings, cash yield 3%/yr credited on the cash curve for metrics
  only (the live accounts accrue 0%). Wall-clock 106m51s, EXIT=0.

## 1. The decade verdict (full window, 2015–2026)

| arm | TWR ann | XIRR | max DD % | excess Sharpe | turnover (1-way, ann) | longest underwater | pct in random_n | bootstrap 95% CI (ann excess) |
|---|---|---|---|---|---|---|---|---|
| **monthly** (qv-mom-v1) | **+18.29%** | +21.03% | 26.36%¹ | **+0.814** | 1.38× | 398 sess | **98** | **[+4.98%, +26.50%]** |
| **daily_core** (immediate deployment) | +17.85% | +20.02% | 35.09% | +0.745 | 1.31× | 501 sess | **98** | **[+4.94%, +26.56%]** |
| **sim** (technical) | **−2.17%** | −2.52% | **41.19%** | −0.239 | **22.03×** | **2,207 sess** | **0** | [−13.98%, +6.07%] |
| equal_weight (selection bar) | +14.89% | +15.74% | 27.70% | +0.763 | 0.08× | 396 sess | n/a² | [+3.22%, +20.85%] |
| dca (broad-market control) | +11.10% | +12.96% | 34.02% | +0.531 | 0.00× | 492 sess | 37 | [+0.32%, +17.44%] |
| random_n (seed mean) | +12.01% | +12.78% | 36.67% | +0.519 | 10.50× | 736 sess | — | [−1.20%, +18.67%] |

random_n seed dispersion (TWR ann): p5 +7.76%, p50 +12.10%, **p95 +15.88%**.

¹ The monthly arm is `marks_only` (a step NAV of month-end fill marks) — its
26.36% maxDD **understates intra-month drawdown by construction**. The same
factor policy marked daily is the daily_core arm's 35.09% — reconciled
exactly by the 5B M0 run (the daily-core engine at the monthly cadence:
maxDD 35.1%). **The flagship's honest maxDD is ~35%, not 26%.**
² equal_weight holds the full eligible set; it is not placed in the
10-name random_n distribution (it IS the universe bar).

**The four verdicts the study was built to deliver:**

1. **The monthly factor book has SKILL, not selection luck.** +18.29%/yr sits
   at the **98th percentile** of the random-placement distribution — above the
   entire seed p95 (+15.88%), with the bootstrap CI excluding zero
   decisively, and robust across all 7 OFAT variants (below). It clears the
   equal_weight selection bar by +3.4pp/yr on the decade.
2. **Immediate deployment (daily_core) also has skill** — the same 98th
   percentile, CI excluding zero — but it is NOT an improvement on the
   monthly book: −0.44pp/yr on the decade, the same honest maxDD (~35.1%;
   the monthly arm's 26.4% is the step-NAV view¹), and a 2022 whipsaw the
   monthly cadence avoided (§2).
3. **The technical sim has NO skill — it is worse than random.** The 0th
   percentile on the full window (both cases), −2.17%/yr vs the random-n mean
   +12.01%/yr (a −14.2pp/yr gap), the **worst drawdown of all six arms
   (41.19%, worse than the unhedged DCA control's 34.02%** — the stops did
   not protect on the axis they exist for), 22×/yr one-way turnover (6,686
   trades; ~2.2pp/yr pure cost drag before selection quality), and a
   2,207-session continuous underwater spell — **8.8 years below water**.
   Its CI [−13.98%, +6.07%] cannot exclude zero, let alone the bars.
4. **The controls did their job.** The DCA control lands at the 37th
   percentile of random placement (+11.10% vs seed mean +12.01%) — the
   absolute-vs-dca comparisons carry the survivorship caveat (below), but the
   sandwich (dca < random_n < factor arms) is what skill-vs-selection-bias
   evidence is supposed to look like.

## 2. The regime map (yearly windows, contribution case)

| window | dca | monthly | daily_core | sim | equal_weight | pct: monthly / daily_core / sim |
|---|---|---|---|---|---|---|
| 2016 | +9.2% | +20.1% | +21.6% | −3.5% | +20.8% | 65 / 76 / 0 |
| 2017 | +21.3% | +21.2% | **+34.3%** | +10.5% | +17.6% | 81 / **100** / 16 |
| 2018 | **−9.1%** | **−14.8%** | −12.1% | −26.3% | −8.6% | **27 / 44 / 0** |
| 2019 | +27.4% | +19.3% | **+31.8%** | +14.6% | +18.2% | 74 / **99** / 51 |
| 2020 | +15.3% | +16.6% | +14.9% | +0.0% | **+20.6%** | 50 / 41 / 3 |
| 2021 | +21.4% | +18.5% | +17.2% | +16.6% | **+26.3%** | **24 / 22 / 17** |
| 2022 | **−18.9%** | **−4.2%** | −13.0% | −24.1% | −5.3% | 66 / **16** / 0 |
| 2023 | +24.4% | +29.7% | **+44.1%** | +6.9% | +13.8% | 98 / **100** / 27 |
| 2024 | +19.4% | +32.0% | **+50.3%** | −1.8% | +13.7% | 98 / **100** / 3 |
| 2025 | +22.6% | +10.9% | +23.4% | +6.3% | +11.5% | 60 / **97** / 34 |
| 2026 (partial, to 10-30) | +13.3% | **+32.9%** | +23.6% | +8.1% | +16.4% | 90 / 77 / 29 |
| halves-1 (2015..2020-11) | +9.4% | +13.5% | +14.4% | −2.1% | +14.5% | 80 / 86 / 0 |
| halves-2 (2020-11..2026) | +12.6% | **+22.6%** | +22.9% | −3.0% | +14.3% | **98 / 98 / 0** |

Where the skill lives — and where it does not:

- **The skill is regime-concentrated.** The decade placement is carried by
  the strong/dislocating momentum years: 2023 and 2024 (monthly 98th,
  daily_core 100th both years), 2026-partial (monthly 90th), halves-2
  (98/98). In the halves-1 era (2015..2020-11) the placements are 80/86 —
  above the random median, inside the seed band.
- **The 2022 defensive win belongs to the monthly cadence.** monthly −4.2%
  (XIRR **+4.6%**; daily_core also finished in the black money-weighted at
  +3.4% — the only two risk-arm years that did) vs dca −18.9% and
  daily_core −13.0% TWR. The month-end hysteresis de-risked; **immediate
  deployment whipsawed** (daily_core at the 16th percentile — its worst
  relative year). This is the sharpest measured distinction between the
  two cadences in the study.
- **2018 is the study's hardest honesty fact: both factor arms lost MORE
  than the DCA control** (monthly −14.8% and daily_core −12.1% vs dca
  −9.1%; monthly at the 27th percentile). The momentum crash year is real;
  a decade aggregate at the 98th percentile does not remove it.
- **2020–2021: selection HURT.** equal_weight beat both factor arms
  (2020: +20.6% vs +16.6%/+14.9%; 2021: +26.3% vs +18.5%/+17.2%, with the
  factor arms at the 24th/22nd percentiles — breadth years, momentum
  unfriendly). In 2025 the monthly book badly trailed the control
  (+10.9% vs dca +22.6%) while daily_core printed its 97th percentile.
- **The sim never has a good year.** Best yearly placement: the 51st
  percentile (2019 — a coin flip vs random). Below the random median in 27 of
  28 window×case placements; at the 0th percentile in 11 of 28.

## 3. Robustness

- **OFAT (full window, monthly arm TWR ann):** stock_bps 5 → +18.47%,
  stock_bps 20 → +17.95% (−0.52pp across the 5→20 bps cost span);
  cash_yield 0 → +18.03%, cash_yield 5% → +18.48%; **mom_residual →
  +18.92%** (residualizing the momentum leg leaves the skill intact — the
  selection carries more than momentum); target_n 20 → +17.29%;
  hold_band 40 → +18.19%. **Every variant stays above the random p95
  (+15.88%).** daily_core shows the same shape (17.01%–18.36% across
  variants). The verdict is not parameter-fragile.
- **Bootstrap:** the full-window CIs exclude zero for monthly
  [+4.98%, +26.50%] and daily_core [+4.94%, +26.56%]; the sim's straddles
  zero. Yearly-window CIs are wide (single-year data; the windows are
  declared views of overlapping data, not independent samples) — the
  placement bars, not the yearly CIs, carry the regime conclusions.
- **The fixed case:** identical placements (full window: monthly 98 /
  daily_core 98 / sim 0 in BOTH cases; TWR +18.30% / +17.37% / −2.00%).
  The funding schedule does not change any verdict (within-case comparisons
  only, as declared).

## 4. Honesty limits (declared, verbatim from the study header)

- Windows overlap by construction (full vs yearly vs halves are views of
  the same data, not independent samples); the uncertainty layer does not
  de-correlate them.
- The monthly-cadence arms (incl. both controls) are `marks_only` — a step
  NAV of fill marks — so their intra-month volatility/drawdown is
  understated by construction; the daily_core arm is the honest daily-marks
  view of the same policy. Bootstrap CIs on the marks_only arms inherit
  that understatement.
- The fixed case changes the funding schedule, not just the amount:
  XIRR/TWR compare differently ACROSS cases than WITHIN one.
- **Survivorship/selection bias:** the universes are current-membership
  snapshots — absolute comparisons vs the broad-market DCA control are
  biased (up); **relative comparisons across arms sharing this universe are
  robust to it.** This applies to the equal_weight bar too.
- cash_yield is a post-hoc declared assumption credited for metrics only;
  the Sharpe's risk-free proxy is that same declared yield.
- yearly-2015 is infeasible for the factor arms (momentum warmup: 252
  sessions < 253) — an honest refusal, not a hidden zero.
- The sim arm runs the live sim parameters as an experimental comparator
  (the audit's arm 4); this study grades the composite arm, not a
  decomposition of its signal vs its stops (the 5A-1 sell-side study
  measures the stop machinery separately).
- No significance tests beyond the declared percentile/bootstrap bars; the
  yearly windows are single views with wide CIs.

## 5. The input to Phase 4D (the prune — recommendation, owner decides)

The predeclared prune rule: *"Archive a variant from the main UI when it
adds complexity without a robust net-of-cost benefit; retain reproducible
research."* Against the 4C evidence:

1. **KEEP the monthly factor book (qv-mom-v1) as the flagship.** The only
   arm with decade skill (98th percentile, CI>0, OFAT-robust) AND a 2022
   defense (−4.2% vs the control's −18.9%, XIRR positive). The 2018/2021
   regime facts belong in its UI context, not in a footnote: it can and did
   lose to the broad control in a momentum crash and to equal-weight in
   breadth years.
2. **ARCHIVE the standalone daily-core variant from the main UI**
   (recommendation): the immediate-deployment policy adds a second
   factor-strategy surface whose decade delta vs the monthly book is
   negative (−0.44pp/yr, worse honest maxDD) and whose distinguishing
   behavior is the 2022 whipsaw (16th percentile) — complexity without a
   robust net-of-cost benefit. The ENGINE stays (it is the platform the 5B
   M-set and the 5C reviewer machinery run on, and 5B measured M3a
   target_vol=0.20 as the decade's winner at +18.34%/yr / 33.3% maxDD —
   an engine-config decision the owner may take separately, not a UI-variant
   question).
3. **ARCHIVE the technical sim from the main UI** (recommendation): the
   study's most decisive negative — 0th percentile, −14.2pp/yr vs random
   placement, the worst drawdown of all six arms, 22×/yr turnover, 8.8
   years continuously underwater. The engine and its research path stay
   (the 5A/5C sell-reviewer machinery rides on it; the audit's rule is
   archive-from-UI, not delete).
4. **AI stays in prospective shadow mode** (predeclared; now built — the 5C
   reviewer, the `shadow` command, the `assess` scorer — awaiting the
   owner's deploy; graded only on the forward log).
5. **Keep the controls visible in the comparison UI:** dca, random_n and
   equal_weight are the bars that make the flagship's skill claim honest;
   2018 and 2021 are the years that keep it humble.

## 6. Reproducibility

The study is deterministic (byte-identical for identical inputs modulo the
printed wall-clock; seeds 1..100 for random_n, the bootstrap's deterministic
seed, B=2000, block 21). The artifact set is
`/tmp/opencode/reval/study4c/reval-study.{json,md}` (schema `reval-study/1`,
`configs_tried` 3,024); the driver is `app/reval.py` (`python -m app.reval
study --db … --windows full,yearly,halves --cases contribution,fixed
--sensitivity ofat --seeds 100`), committed through Phase 4B + the
infeasible-window fix (4696052).
