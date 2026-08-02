# Trade Sentinel MVP

A self-hosted, paper-only stock research dashboard. Market data is switched globally with **one environment variable**; all providers normalize historical daily OHLCV data into the same local SQLite cache, so charts, signals, autocomplete, and screening use the selected backend consistently.

## Choose a provider

```dotenv
# Default: free/best-effort US + international coverage, including Yahoo symbols such as IFX.DE and OMV.VI
MARKET_DATA_PROVIDER=yfinance

# Alternative: requires an API key; Basic coverage is mainly US equities/ETFs, forex and crypto
# MARKET_DATA_PROVIDER=twelvedata
# TWELVE_DATA_API_KEY=your_key
```

`yfinance` uses Yahoo Finance's public endpoints through the `yfinance` library. It enables global ticker search and mixed US/EU screeners without a data key, but it is not an official market-data API: cache aggressively, throttle manual screener runs, and treat it as EOD/best-effort research data. Do not use it for execution.

## Run

```bash
cp .env.example .env
nano .env
docker compose up -d --build
```

Open `http://SERVER:8010`. Altering `MARKET_DATA_PROVIDER` requires a restart: `docker compose up -d --build`.

## Autocomplete and screener

Autocomplete uses the selected provider. With `yfinance`, it supports fuzzy company/ticker lookup and returns canonical Yahoo symbols such as `IFX.DE`, `ASML.AS`, and `OMV.VI`. The `global-large-cap` universe mixes US, German, Dutch, French, Swiss, and Vienna listings. Press **Update** manually after markets close; it downloads and caches about two years of daily candles for each symbol, then ranks trend alignment, 20/60-day momentum, RSI, and relative volume.

No live broker or order API exists. Signals and rankings are research tools, not financial advice.

## AI / space universe

`universes/global-large-cap.txt` is now an AI and advanced-technology research universe. It includes semiconductors, AI infrastructure/platforms, application/automation names, space/connectivity companies, selected European listings, and recent IPOs such as CoreWeave (`CRWV`), Figma (`FIG`), Circle (`CRCL`), Chime (`CHYM`) and eToro (`ETOR`). SpaceX is listed as `SPCX`; use provider autocomplete to confirm current symbol availability before a screen run. A company being included only makes it a candidate for a rule-based research screen, not an investment recommendation.

## Fix: duplicate screener symbols

The screener deduplicates a universe while preserving its order before downloading data and saving results. This prevents duplicate entries such as a symbol classified in both AI and space groups from violating SQLite's `(universe, ticker)` uniqueness constraint.
