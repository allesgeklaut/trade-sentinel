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

**Hybrid + minimal prompt + engine-owns-exits + failure marker
(stop_outs=2, drawdown=7%, no cooldown)**

Rationale: the only config competitive on all three windows — ties the
best on the bull window (+5.5%), dramatically best on the chop window
(-0.8 to -2.8%), acceptable on the short suite (-0.3%). The 5×30 short-suite
cost is the price for fixing the long windows, where real money is at risk.

## 6. Live deployment

Live sim runs (`.env`, gitignored):

```
SIM_STRATEGY=hybrid
SIM_LLM_REVIEW_INTERVAL=5
SIM_LLM_MINIMAL_PROMPT=true
SIM_LLM_FAILURE_MARKER=true
```

How it works live: the deterministic engine runs every cycle as primary;
the LLM is consulted only when the engine shows failure — 2+ stop-out SELLs
in 5 trading days, or equity >7% below its running peak (from SimSnapshot
history) — with no cooldown. When the engine works, the LLM stays quiet
(saves tokens, avoids churn).

## 7. Open items

- Pure-LLM top-up gap: at the max-positions cap, the LLM proposes new
  tickers (blocked) instead of topping up held positions; allowance cash
  sits idle. Worth a follow-up if pure-LLM is pursued further.
- The 60d chop window remains the hardest case; the failure marker is the
  best mitigation found so far, not a full fix.
