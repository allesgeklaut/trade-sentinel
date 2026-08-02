# Trade Sentinel MVP

Self-hosted, paper-first stock research dashboard. It fetches daily candles from Twelve Data, caches them in SQLite, computes transparent trend/RSI/MACD/ATR signals, stores analysis history, and can ask a local Ollama model to explain an already-computed signal. It deliberately does **not** place live orders.

## Run

```bash
cp .env.example .env
# Set TWELVE_DATA_API_KEY and your reachable OLLAMA_URL
# For Linux Docker engines, use your host LAN IP or add host-gateway mapping.
docker compose up -d --build
```

Open `http://SERVER:8010`. The first refresh needs the Twelve Data key. Add or remove tickers in the UI. Data lives in the named volume `trade_sentinel_data`.

## Safety model

- `PAPER_TRADING=true` is mandatory for this MVP; no broker credentials or execution routes exist.
- Signals are deterministic, versioned and persisted with their indicator snapshot.
- LLM output is explanatory only; it receives no credentials and cannot execute actions.
- Keep the service behind your normal Cloudflare Access/OIDC setup; it has no built-in auth.

## Signal policy

`BUY` means bullish SMA alignment plus an RSI recovery; `SELL` means a bearish trend break; all other states are `HOLD`. These are research signals, not financial advice. Backtest and paper-trade before considering any real execution.

## API

- `GET /api/watchlist`
- `POST /api/watchlist/{ticker}` / `DELETE /api/watchlist/{ticker}`
- `POST /api/refresh/{ticker}`
- `GET /api/dashboard/{ticker}`
- `POST /api/explain/{ticker}`
- `GET /healthz`
