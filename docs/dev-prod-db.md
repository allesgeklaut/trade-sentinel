# PROD/DEV database switch

Trade Sentinel can run **two SQLite databases side by side**: **PROD**
(`/data/trading.db`) keeps the live paper-portfolio and savings history, while
**DEV** (`/data/trading_dev.db`) is a disposable sandbox for backfills and
strategy experiments. Both live on the same `/data` volume of the same
`docker-compose` service — no second container, no second mount. The selection
is **per browser** (a cookie), so a DEV session can rebuild and re-run the
paper portfolios while the scheduled jobs and every other client keep
operating on the live PROD book.

## Enabling

Point `DATABASE_URL_DEV` at a file in `/data` (see `.env.example`):

```dotenv
DATABASE_URL_DEV=sqlite+aiosqlite:////data/trading_dev.db
```

The feature is **disabled by default**: an empty/unset value means no dev
engine exists at all and every request is served by PROD. With the dev engine
configured and the dev file missing, the app **auto-seeds DEV from a full copy
of PROD on startup** (before `init_db()` runs), so the sandbox starts with the
same candles and portfolio state instead of an empty schema.

## The GUI switch

The header toggle writes the `ts_db` cookie (`prod` or `dev` — one year,
`path=/`) and reloads the page. Server-side, `db_select_middleware` routes
each request by that cookie:

- missing/unknown cookie, or `DATABASE_URL_DEV` unset → **PROD**;
- `ts_db=dev` with the dev engine configured → **DEV**.

The cookie makes the choice **per browser**: another client (or an incognito
window) still talks to PROD. The UI follows the active DB returned by
`GET /api/db/status`:

- **PROD selected** — the backfill/reset controls are hidden behind a note
  ("Backfills & resets are disabled on PROD — switch to DEV"); the amber
  sandbox banner is hidden.
- **DEV selected** — an amber *DEV SANDBOX* banner appears with a **Clone
  PROD → DEV** button, the backfill/reset controls become available, and the
  **Savings tab is hidden** (the DEV copy of real-money data is stale).

## What runs where

| Action | PROD | DEV |
| --- | --- | --- |
| Backfills — `POST /api/sim/backfill`, `/api/sim/backfill-benchmark`, `/api/backfill-all`, `/api/monthly/backfill`, `/api/dailycore/backfill` | **403** | yes |
| Resets — `POST /api/sim/reset`, `/api/sim/reset-all` | **403** | yes |
| Savings writes — initialize, reset, true-up, config, deposit, saveback, buy, plan add/remove, position removal, refresh, run | yes | **403** |
| Savings reads — GET status/equity/mirror/plans | yes | yes |
| Background schedulers — sim cycles, universe prefetch/sync, savings interest/Sparplan pass | **always** | never |
| Manual actions — Run Bot Now, rebalances, refresh buttons | selected DB | selected DB |
| Daily-core strategy state | `/data/daily_core_state.json` | `/data/daily_core_state_dev.json` |
| LLM backend state | `/data/llm_state.json` (shared) | shared |

**Why the guards are split this way.** The live history is the product:
rewriting it (backfill) or wiping it (reset) is safe only on the disposable
copy, so every history-rewriting endpoint answers 403 on PROD. The savings
tracker is the opposite — it mirrors **real money** — so its writes are
PROD-only and the DEV copy is read-only; savings GETs are unguarded.

**Schedulers never follow the cookie.** The background jobs (daily/weekly sim
cycles, the nightly prefetch and universe sync, the savings interest/Sparplan
pass) start in the app's lifespan and run outside any request context, so they
always operate on PROD. A browser selecting DEV cannot starve or redirect
them. Manual triggers (Run Bot Now, a rebalance, a refresh) run inside the
request and follow the selected DB.

**Daily-core state is per-DB, LLM state is shared.** The persisted daily-core
strategy choice (momentum variant plus the protection-overlay/gradient chain)
lives in a JSON state file read by both the scheduler and backfills; it is
selected per active DB, so DEV experiments can never change what the live PROD
book runs. The LLM backend/model selection (`llm_state.json`) stays shared —
the LLM choice is application-global, not per-database.

## Cloning

`POST /api/db/clone` (or the banner's **Clone PROD → DEV** button) overwrites
DEV with a fresh copy of PROD, **including the full candle history and the
portfolio state**. Use it when the sandbox went stale: the schedulers advance
PROD every day, so a DEV copy from last week lacks the newest candles, trades
and snapshots. Cloning also discards whatever the last DEV experiment left
behind.

- **Atomic**: the copy uses SQLite's online backup API (safe while PROD is
  live under WAL), lands in a temp file and is swapped in with `os.replace`;
  the dev engine is disposed around the swap and stale `-wal`/`-shm` sidecars
  are removed so no pooled connection or leftover sidecar can pair with the
  new file.
- **Serialised**: clone shares the backfill lock with **Backfill All** and
  **Reset All** — it answers **409** while one of those is running, and they
  answer 409 while a clone is running. It also answers 409 when DEV is
  disabled entirely.
- **Not copied**: the per-DB daily-core state file stays as-is on the DEV
  side (the clone copies the database, not the `*.json` state files).
- **Observable**: `GET /api/db/status` reports `cloning: true` for the
  duration of the clone (the UI shows "Cloning…").

## Status endpoint

`GET /api/db/status` returns the state of the switch for the calling client:

```json
{"active": "prod", "dev_enabled": true, "dev_seeded": true, "cloning": false}
```

- `active` — the DB that served this request (`prod` or `dev`, from the cookie).
- `dev_enabled` — whether `DATABASE_URL_DEV` is configured at all.
- `dev_seeded` — whether the dev database file exists on disk.
- `cloning` — whether a clone is currently running.

## Rollback / disabling

- Unset (or empty) `DATABASE_URL_DEV` and restart: there is no dev engine, the
  middleware only ever serves PROD, and every client is pinned back instantly.
  The DEV button remains in the UI but DEV requests fall through to PROD, and
  `GET /api/db/status` reports `dev_enabled: false` (as does `POST
  /api/db/clone`, with a 409).
- The dev database file itself is **left on disk**; nothing deletes it
  automatically. Remove `/data/trading_dev.db` (plus any `-wal`/`-shm`
  sidecars) to reclaim the space — with the feature still enabled, the next
  start auto-seeds a fresh DEV copy from PROD.
- Nothing on the PROD side needs undoing: DEV experiments never wrote to PROD
  (the guards above), and the schedulers kept writing to PROD throughout, so
  disabling the switch changes nothing about the live history.
