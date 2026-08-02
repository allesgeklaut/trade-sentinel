# Trade Sentinel MVP

Self-hosted, paper-first stock research dashboard using Twelve Data only. It caches daily candles in SQLite, computes transparent signals, provides Twelve Data ticker/company autocomplete, and screens a local editable universe for sustained trends. It has no broker credentials or execution routes.

## Run
```bash
cp .env.example .env
# Set TWELVE_DATA_API_KEY and a container-reachable OLLAMA_URL
docker compose up -d --build
```
Open `http://SERVER:8010`.

## Ticker autocomplete
Type two or more characters in the add field. The frontend debounces remote Twelve Data `/symbol_search` requests by 300 ms, then displays the provider's canonical symbol, instrument name and exchange. Select a match before adding it to avoid ambiguous tickers.

## Local screener
`universes/us-large-cap.txt` is an editable 41-symbol starter universe. Add `atx.txt` or `xetra.txt`, one provider symbol per line, then rebuild the image. **Update** fetches daily history sequentially, writes it into SQLite, and ranks entries by trend alignment, 20/60-day momentum, RSI, and relative volume. It consumes one Twelve Data time-series request per symbol, so run it once after market close (not repeatedly) on the free quota.

The screener intentionally is not scheduled in this MVP: manually update it initially so data-credit consumption remains fully visible and predictable. Add a scheduler only after selecting the universe size and market-close time you want.

Signals/screener rankings are research tools, not financial advice. Validate in a backtest and paper mode before any live execution.
