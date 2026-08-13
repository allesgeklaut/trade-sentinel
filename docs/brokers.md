# Live broker options

Trade Sentinel is currently **paper-only**: the simulation engine and the
deterministic strategy never send orders to a real broker. This document
summarises the brokers that expose a usable REST API, evaluated from the
perspective of an operator in **Austria** running this app.

There is **no broker that offers all three** of: a clean REST API, EU-listed
equities, *and* Austrian automatic capital-gains-tax (KESt) deduction. Every
choice is a tradeoff between those three axes.

## Summary

| Broker            | API          | US stocks | EU stocks | Commission                          | AT tax (KESt) | Verdict for this app                  |
| ----------------- | ------------ | :--------: | :-------: | ----------------------------------- | :-----------: | ------------------------------------- |
| **Alpaca**        | REST (clean) | yes        | **no**    | $0 (SEC/FINRA fees on sells only)   | self-declare   | Best if live universe is **US-only**  |
| **XTB**           | xStation REST | yes       | yes       | €0 < €100k/mo, 0.2% (min €10) above  | self-declare   | Best for the **mixed US/EU** universe |
| **Interactive Brokers** | REST / FIX / TWS | yes | yes   | ~0.1–0.5% per exchange tier         | self-declare   | Best coverage, but infra-heavy        |
| Flatex AT / Direktanlage | **none**  | yes        | yes       | low, KESt included at source         | **automatic**  | Steuereinfach, but no API at all      |
| DEGIRO / Trade Republic / Scalable | **none (official)** | yes | yes | low | self-declare | No official trading API; unofficial scrapers break |

## Alpaca

<https://alpaca.markets> — API-first US broker, the closest match to this app's
current architecture (`httpx` async REST, $0 commission, fractional shares).

**Fees:**
- Commission: **$0** on US-listed stocks/ETFs.
- SEC fee: $22.90 per $1,000,000 of sell principal (≈0.00229%), rounded up to
  the nearest penny per fill.
- FINRA TAF: $0.000119 per share sold, rounded up, capped at $5.95 per sell.
- Net effect on a $1,500 sell: ~$0.03. Effectively zero for a research sim.

**Pros:**
- Clean REST API; drops straight into the existing `httpx` async style.
- Austria is an accepted country (non-US resident accounts supported).
- The legacy Pattern Day Trader (PDT) rule was **retired in June 2026** and
  replaced by a real-time intraday margin framework, so the old "3 day trades
  in 5 days" limit no longer applies.
- $0 commission, so the fee model in the sim stays accurate as-is.

**Cons / blockers:**
- **US-exchange-listed equities only.** No EU listings. Confirmed by Alpaca
  staff on their forum: "Alpaca supports US exchanges only." Tickers such as
  `IFX.DE`, `ASML.AS`, `OMV.VI` from the `global-large-cap` universe **cannot be
  traded** on Alpaca. They can remain paper/research-only, but a live order
  for them would fail.
- USD account only — FX conversion costs apply on deposits/withdrawals (not on
  trades themselves).
- Not steuereinfach: no Austrian KESt deduction. Capital gains and US
  withholding tax on dividends must be self-declared in the yearly
  Steuererklärung (Quellensteuer relief via the Doppelbesteuerungsabkommen is
  manual).

**Use Alpaca if** the live-executable universe is restricted to US names.

## XTB

<https://www.xtb.com> — Polish brokerage, EU-regulated and passported into
Austria. Offers real stocks and ETFs (not just CFDs) alongside CFDs/forex.

**Fees:**
- Stocks/ETFs (OMI instruments): **€0 commission up to €100,000/month
  turnover**; 0.2% (minimum €10) above that threshold.
- Below €100k/mo turnover (the common case for a single-user research app),
  trading is effectively free.
- Currency conversion fee applies if the instrument is in a currency other than
  the account currency.

**Pros:**
- Official **xStation REST API** (HTTP, no local process to keep running).
- Lists **both US and EU equities/ETFs** — matches the existing
  `global-large-cap` universe including `IFX.DE` and other European names.
- Austria supported; EU-regulated (Polish entity, passported).
- No PDT-style limits for a cash account.

**Cons:**
- Not steuereinfach: no automatic Austrian KESt. Same self-declaration regime as
  Alpaca. The broker's annual report makes the declaration doable, just not
  automatic.
- The €10 minimum on the 0.2% tier would materially affect a strategy that
  makes many small trades once the €100k/month threshold is crossed — at that
  point the sim's fee model would need to account for it.
- Smaller US-stock subset than Alpaca/IBKR.

**Use XTB if** keeping EU names executable matters. It is the only API broker in
this list that covers both US and EU exchanges.

## Interactive Brokers (IBKR)

<https://www.interactivebrokers.com> — broadest market coverage of any broker,
institutional-grade.

**Fees:**
- Tiered or fixed pricing per exchange; typically ~0.1–0.5% depending on the
  venue and plan. Minimum per-order fees apply on some exchanges.
- Inactivity fee if trading volume is too low (waivable with conditions).

**Pros:**
- Widest instrument coverage of any broker (US, EU, Asia, options, forex).
- REST (Client Portal API), FIX, and TWS/Gateway SDKs.

**Cons:**
- **Infrastructure-heavy.** The classic API requires the IB Gateway or Trader
  Workstation process to be running locally — awkward in a containerised Docker
  setup like this app's. The newer Client Portal REST API is more
  cloud-friendly but still finicky and less mature than Alpaca's or XTB's.
- Not steuereinfach (self-declare).
- Overkill for a single-user research app.

**Use IBKR if** you need the widest possible market coverage and can absorb the
operational overhead of keeping the Gateway running.

## Austrian-domiciled brokers (Flatex AT, Direktanlage, Erste, …)

These are **steuereinfach**: they deduct Austrian KESt at source, handle
Quellensteuer relief, and report to the Finanzamt, so the operator does
essentially nothing at tax time.

**The blocker:** none of them expose an official trading API. Flatex/Degiro
(same group) explicitly states it offers no API. Trade Republic and Scalable
Capital likewise have no official trading API; the community-built connectors
rely on reverse-engineered web sessions and WAF bypasses that break
regularly.

**Verdict:** unusable for a programmatic app, despite the tax convenience.

## Decision guide

```
Is the live universe US-only?
  yes → Alpaca (simplest, cheapest, cleanest API)
  no  → Is keeping EU tickers executable important?
          yes → XTB (only API broker covering both US + EU)
          no  → Alpaca, and keep EU names paper-only
```

No option is steuereinfach. Self-declaration in the Austrian Steuererklärung
is unavoidable for any broker with an API. The differentiator is therefore
**which exchanges are executable**, not the tax treatment.

## Note on the sim's fee model

The current simulation charges **no commission and no fees** — a reasonable
approximation for Alpaca (where fees are ~0.002% of sells only, i.e. a few
cents per year on a typical paper portfolio).

If you ever route live orders through XTB above its €100k/month threshold, or
through IBKR, the fee plumbing would start to matter (e.g. XTB's €10 minimum
ticket fee on small trades is ~1000× Alpaca's sell-side cost). At that point a
config-driven fee model — `sim_commission_pct`, `sim_sec_fee_per_million`,
`sim_finra_taf_per_share`, `sim_finra_taf_cap`, defaulting to the real Alpaca
numbers — would be worth adding. For the current paper/Alpaca-shaped sim it is
not.