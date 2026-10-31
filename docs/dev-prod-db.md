# PROD/DEV database switch

Trade Sentinel can run **two SQLite databases side by side**: **PROD**
(`/data/trading.db`) keeps the live paper-portfolio, run-config/epoch and
savings history; **DEV** (`/data/trading_dev.db`) is a disposable sandbox for
backfills and strategy experiments. Both live on the same `/data` volume of
the same `docker-compose` service — no second container, no second mount. The
selection is **per browser** (the `ts_db` cookie), so a DEV session can
rebuild and re-run the paper portfolios while the scheduled jobs and every
other client keep operating on the live PROD book.

## The invariants (what this feature guarantees)

1. **PROD books can never be history-rewritten via the API.** The seven
   destructive endpoints — `POST /api/sim/backfill`, `/api/sim/backfill-benchmark`,
   `/api/backfill-all`, `/api/monthly/backfill`, `/api/dailycore/backfill`,
   `/api/sim/reset`, `/api/sim/reset-all` — answer **403** unless the request
   is dev-routed. This is **unconditional**: with the feature *disabled*
   there is no DEV database at all, so **no database accepts them** — PROD is
   append-only by design (resets advance an epoch, backfills are refused
   outright) and a 403 is the honest answer in every configuration.
2. **Savings (real money) writes are PROD-only.** All 20 mutating savings
   endpoints — initialize, reset, true-up, config, deposit, withdraw,
   dividend, fee, tax, settle, split, correct, saveback, buy, plan add,
   plan removal, position removal, sell, refresh, run — answer **403** when
   the request is dev-routed: the DEV copy of real-money data is a stale
   clone, so writes there would only mislead. Savings GETs (including
   `GET /api/savings/plans`) are unguarded and read whichever DB is selected.
3. **Schedulers always operate on PROD.** The background jobs (the sim
   scheduler tick, the monthly daily-mark loop, the universe prefetch/sync,
   the watchlist prefetch, the savings pass) run outside any request
   context, so a browser cookie can never redirect them. A DEV experiment
   holding a portfolio lock may **delay** a scheduled tick, but never
   silently drop it (see below).
4. **DEV is disposable.** It is auto-seeded from PROD on startup and
   re-clonable at any time. A misconfigured `DATABASE_URL_DEV` can never
   endanger PROD: a non-file URL (`:memory:`, a non-SQLite backend) or a
   path that resolves to the PROD file disables the DEV engine at startup
   with an **error log** and the whole feature is off.
5. **Per-browser selection via the `ts_db` cookie.** Missing/unknown cookie
   values — or the feature being off — always mean PROD.
6. **Phase 2 accounting is per database.** Run-config/epoch reads and writes
   resolve the active session at call time, so a dev-routed request's
   run-config registry, epochs and session state hit DEV, never PROD.
7. **Off by default; disabling pins everything to PROD instantly.** An
   empty/unset `DATABASE_URL_DEV` means there is no dev engine, the
   middleware only ever serves PROD, and `GET /api/db/status` reports
   `dev_enabled: false`.

## Enabling

Point `DATABASE_URL_DEV` at a file in `/data` (see `.env.example`):

```dotenv
DATABASE_URL_DEV=sqlite+aiosqlite:////data/trading_dev.db
```

With the dev engine configured, the app **auto-seeds DEV from a full copy of
PROD on startup** whenever DEV is missing **or still empty** (no watchlist
rows) and PROD has data. `init_db()` runs first (both schemas exist) and the
settings watchlist is seeded into PROD before the clone check, so a
greenfield first boot creates both DBs empty and clones as soon as PROD has
the seed tickers — the sandbox starts with the same candles, watchlist and
portfolio state. A failed seed is error-logged and never blocks startup.

## The GUI

- **Settings → Database card**: the `db-toggle` switch writes the `ts_db`
  cookie (`prod` or `dev`, long-lived, `path=/`) and reloads; the
  `db-status` line shows the state from `GET /api/db/status`.
- **Global banner** (`db-dev-banner`): the amber *DEV SANDBOX* strip shown
  whenever the active database is dev.
- **Lab → clone group**: the `db-clone-btn` button triggers
  `POST /api/db/clone` and `db-clone-status` shows its progress/result
  (including the "cloning" state reported by `/api/db/status`).

Server-side, `db_select_middleware` routes each request by the cookie: the
cookie makes the choice **per browser** — another client (or an incognito
window) still talks to PROD. The guards above are what the UI reflects: the
backfill/reset controls only make sense on DEV, the savings tab is
read-only on DEV.

## What runs where

| Action | PROD | DEV |
| --- | --- | --- |
| Backfills — `POST /api/sim/backfill`, `/api/sim/backfill-benchmark`, `/api/backfill-all`, `/api/monthly/backfill`, `/api/dailycore/backfill` | **403** | yes |
| Resets — `POST /api/sim/reset`, `/api/sim/reset-all` | **403** | yes |
| Savings writes (the 20 mutating endpoints above) | yes | **403** |
| Savings reads — status/equity/mirror/plans/metrics | yes | yes |
| Manual runs — `POST /api/sim/run`, `/api/monthly/run`, `/api/dailycore/run` | yes | yes (follow the cookie) |
| Refreshes — `sim/refresh`, `monthly/refresh`, `dailycore/refresh`, `screener/*`, `watchlist/*`, `refresh/*` | selected DB | selected DB |
| Runtime strategy config — `dailycore/{strategy,gate,gradient,targetvol}`, `sim/universe` | selected DB | selected DB (per-DB state file) |
| Chat / LLM — `chat/*`, `sim/chat/*` (incl. confirm/reject and the history DELETE), `llm/select` | selected DB for chat history; LLM state is shared | same |
| Background schedulers — sim cycles, monthly marks/rebalances, daily-core cycles, universe prefetch/sync, watchlist prefetch, savings pass | **always** | never |

**The guards are FastAPI dependencies, not in-body checks.** They run on
every HTTP request; calling the endpoint function directly from Python
(maintenance scripts, tests) bypasses them by design.

**Schedulers never follow the cookie, and a DEV experiment never silently
drops a scheduled tick.** The background jobs start in the app's lifespan and
run outside any request context, so they always operate on PROD. They do
share the portfolio locks with DEV work:

- a scheduled **monthly rebalance** finding `monthly._rebalance_lock` held
  (a DEV backfill or manual run) **waits** for it, bounded by
  `_TICK_LOCK_WAIT_SECONDS` (~2 h), instead of skipping — a missed month-end
  has no catch-up;
- a scheduled **daily-core cycle** likewise waits (bounded); only the manual
  `POST /api/dailycore/run` keeps its immediate "already running" answer
  (`wait=False`);
- a scheduled **sim cycle** finding `sim._run_cycle_lock` held skips that
  wake (so the tick returns promptly and the daily-core stage still runs)
  but schedules a **fire-and-forget deferred retry**: after an initial
  ~60 s delay it waits for the lock (bounded, ~2 h) and then runs the very
  same cycle — *delayed, never dropped*. Only a holder wedged past the
  bound gives up, and that is **error-logged**;
- the daily **monthly mark** (`take_snapshot`) skipped because a rebalance
  holds the lock is likewise deferred by a bounded retry — a delayed mark
  beats a hole in the equity curve.

**Per-DB state files.** The sim's per-session fill guard
(`/data/sim_state.json`) and the daily-core strategy choice
(`/data/daily_core_state.json`) are **per database** (`*_dev.json` variants
when dev-routed): a DEV run stamping the shared file would otherwise make
the PROD tick silently skip a session — and a DEV experiment must never
change what the live PROD book runs. The LLM backend/model selection
(`/data/llm_state.json`) stays shared — the LLM choice is
application-global, not per-database.

## Cloning

`POST /api/db/clone` (or the Lab's **Clone PROD → DEV** button) overwrites
DEV with a fresh copy of PROD, **including the full candle history, the
run-config/epoch registry and the portfolio state**. Use it when the sandbox
went stale: the schedulers advance PROD every day, so a DEV copy from last
week lacks the newest candles, trades and snapshots. Cloning also discards
whatever the last DEV experiment left behind.

- **Atomic**: the copy uses SQLite's online backup API (safe while PROD is
  live under WAL), lands in a temp file next to the DEV file and is swapped
  in with `os.replace`; the dev engine is disposed around the swap and
  stale `-wal`/`-shm` sidecars of the OLD file are removed so no pooled
  connection or leftover sidecar can pair with the new file. A failed clone
  leaves the OLD dev file untouched and no half-written temp behind.
- **Serialised**: the clone holds the shared backfill-all lock, so
  backfill-all/reset-all answer **409** while it runs and it answers 409
  while one of those is running. It also answers **409** ("a backfill or run
  is already running") while any portfolio lock (`sim._run_cycle_lock`,
  `monthly._rebalance_lock`, `daily_core._cycle_lock`) is held. Conversely,
  the four individual backfills and the three manual runs answer **409**
  ("a PROD→DEV clone is running") while the clone flag is set — for
  DEV-routed requests only: a clone swaps the DEV file and nothing else, so
  PROD requests are unaffected. A clone with the feature disabled answers
  **409**.
- **Residual hole**: an operation that was already in flight when a clone
  starts can still race the swap (the lock check narrows, but cannot close,
  it). This is single-operator tooling — don't start a clone while a run you
  care about is mid-flight, and if a DEV result ever looks corrupted,
  re-clone: DEV is disposable and recoverable from PROD at any time.
- **Not copied**: the per-DB state files stay as-is on the DEV side (the
  clone copies the database, not the `*.json` state files).
- **Observable**: `GET /api/db/status` reports `cloning: true` for the
  duration (the UI's `db-clone-status` shows "Cloning…").

## Status endpoint

`GET /api/db/status` returns the state of the switch for the calling client:

```json
{"active": "prod", "dev_enabled": true, "dev_seeded": true, "cloning": false}
```

- `active` — the DB that served this request (`prod` or `dev`, from the
  cookie).
- `dev_enabled` — whether `DATABASE_URL_DEV` is configured and usable.
- `dev_seeded` — the DEV file exists **and has data** (watchlist rows). A
  fresh schema-only DEV file reports `false` until the auto-seed (or a
  re-clone) fills it.
- `cloning` — whether a clone is currently running.

## Rollback / disabling

- Unset (or empty) `DATABASE_URL_DEV` and restart: there is no dev engine,
  the middleware only ever serves PROD, and every client is pinned back
  instantly (a `ts_db=dev` cookie simply falls through to PROD).
  `GET /api/db/status` reports `dev_enabled: false` (as does
  `POST /api/db/clone`, with a 409).
- The dev database file itself is **left on disk**; nothing deletes it
  automatically. Remove `/data/trading_dev.db` (plus any `-wal`/`-shm`
  sidecars and the `sim_state_dev.json` / `daily_core_state_dev.json` state
  files) to reclaim the space — with the feature still enabled, the next
  start auto-seeds a fresh DEV copy from PROD.
- Nothing on the PROD side needs undoing: DEV experiments never wrote to
  PROD (the guards above), and the schedulers kept writing to PROD
  throughout, so disabling the switch changes nothing about the live
  history.
