from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    database_url: str = "sqlite+aiosqlite:////data/trading.db"
    # auto = Twelve Data when TWELVE_DATA_API_KEY is set, else yfinance.
    # yfinance / twelvedata force one source (twelvedata requires a key).
    market_data_provider: str = "auto"
    twelve_data_api_key: str = ""
    # Path to a file holding the bare key (single source of truth, e.g. a
    # read-only /opt/secrets mount). Used when TWELVE_DATA_API_KEY is empty,
    # mirroring LITELLM_API_KEY_FILE for the LLM backends.
    twelve_data_api_key_file: str = ""
    # Twelve Data Basic plan: 8 credits/min and 800/day. The limiter paces
    # requests to max_per_min and falls back to yfinance once the day's budget
    # is spent — so a broad universe prefetch can never trip 429s or overrun.
    twelve_data_max_per_min: int = 8
    twelve_data_daily_budget: int = 750
    watchlist: str = "AAPL,MSFT,NVDA,IFX.DE"

    ollama_url: str = "http://host.docker.internal:11434"
    ollama_model: str = ""  # set via .env, e.g. "qwen3:32b"
    ollama_timeout_seconds: float = 1200.0

    # Multi-backend LLM config. JSON list of {"name", "type", "url", "model"}
    # where type is "ollama" (native /api/chat) or "openai" (OpenAI-compatible
    # /v1/chat/completions, e.g. llama.cpp llama-server). When unset, the
    # legacy OLLAMA_URL / OLLAMA_MODEL settings are used as a single backend.
    llm_backends: str = ""
    llm_state_file: str = "/data/llm_state.json"
    # Qwen3 thinking budget applied to OpenAI-compatible backends (llama-server).
    # One of: low | medium | high | xhigh. Empty = leave server default untouched.
    llm_reasoning_effort: str = ""

    # --- SearXNG news search (optional) -------------------------------
    searxng_url: str = ""  # e.g. "http://your-server-ip:8081"; empty = disabled
    searxng_timeout: float = 10.0

    paper_trading: bool = True

    # --- Signal scoring variant -------------------------------------------
    # "classic" = original weights (momentum-chasing: rewards 1d RSI rising +
    # rising MACD histogram + extension). "pullback" = measured v2 weights:
    # drop the hist-rising bonus, reward 5d RSI *falling* (pullback entries,
    # +6.2% fwd vs -0.7% chasing), re-curve dist_above (sweet spot 2-50%,
    # penalty >80%). Component attribution on 205 year-long entries showed
    # the classic score's ranking is inverted (top tercile -3.3% vs bottom
    # +4.7% fwd); pullback scoring fixes it (+7.1% spread) and wins the
    # bull90 holdout (+12.3% vs +9.9%).
    signal_scoring: str = "classic"

    # Timezone anchor for the calendar-month allowance deposits of BOTH
    # paper portfolios (sim + monthly qv-mom). Sharing one anchor means their
    # cumulative "contributed" figures step at the same moment, so the two
    # equity curves are directly comparable.
    allowance_tz: str = "Europe/Vienna"

    # --- Autonomous paper-trading simulation -----------------------------
    sim_enabled: bool = True
    sim_monthly_allowance: float = 1000.0
    sim_start_cash: float = 0.0
    sim_universe: str = "diversified-plus"
    sim_strategy: str = "deterministic"  # deterministic | llm | hybrid
    sim_max_position_pct: float = 10.0
    sim_min_cash_pct: float = 5.0
    sim_max_positions: int = 10
    sim_stop_pct: float = 15.0
    sim_max_run_5d: float = 12.0  # block BUYs after a 5-day run-up > this % (0 = disabled)
    # Entry guards — opt-in (0 = disabled). Measured rationale in strategy.py:
    # min_run_5d=-15 blocks falling-knife entries (pooled fwd -30%),
    # max_dist_above=80 blocks parabolic entries (pooled fwd -13% at >100%).
    # A/B verdict: fixes the stop-out cascade (win4: 7->2 stop-outs, -7.7->-0.4%)
    # but costs right-tail returns in strong trends (win2: +20.5->+15.6%) —
    # keep OFF until the scoring-side fix is evaluated on its own branch.
    sim_min_run_5d: float = 0.0
    sim_max_dist_above: float = 0.0
    sim_llm_review_interval: int = 1  # consult the LLM every N cycles (1=daily, 5=weekly)
    sim_llm_minimal_prompt: bool = False  # use the minimal system prompt (no methodology/regime rules)
    sim_llm_mode_aware_prompt: bool = False  # use the mode-aware minimal prompt (engine owns exits; LLM adds BUYs only)
    sim_llm_failure_marker: bool = False  # hybrid: consult the LLM only on engine failure (stop-out cascade / drawdown)
    # Fundamentals context (daily sim): show point-in-time ROE % and P/FCF per
    # candidate in the LLM's signals table. Missing fundamentals render as "-"
    # (neutral); the table is omitted entirely when off (zero prompt change).
    sim_llm_fundamentals_context: bool = False
    # Quality guard (daily sim): block BUY entries for names with KNOWN
    # non-positive ROE. Missing fundamentals stay neutral (ETFs like GLD,
    # thin coverage) — only known-bad data blocks. Opt-in until the replay
    # A/B verdict (bull-window right-tail cost is the known risk class).
    sim_block_negative_roe: bool = False
    # Daily-core rank deployment: boost factor for the #1-ranked name's target
    # weight, decaying linearly to 1.0 at the band edge (rank N gets exactly
    # the equal weight). 0.5 = top name may hold 1.5x the equal weight while
    # the 10th holds 1.0x. Only used by the daily-core backtest --dca rank.
    # Default 0.0 = flat equal-weight targets, the measured winner (experiment
    # doc §10); the live run_deployment is flat too.
    sim_monthly_rank_boost: float = 0.0

    # --- Daily-core risk overlays (all opt-in, measured via daily-core-sweep) ---
    # mom_variant: "raw" = classic 12-1 close/close momentum. "residual" =
    # Blitz-Huij-Martens residual momentum: rank by the residuals of each
    # stock's 12-1 daily log returns regressed on the market's, scaled by
    # their std-dev (momentum per unit of idiosyncratic vol). Literature:
    # ~2x Sharpe and roughly half the crash risk of raw momentum.
    sim_daily_core_mom_variant: str = "raw"
    # target_vol (annualized, 0 = off): when the portfolio's own 21d realized
    # vol exceeds this, hold (realized/target - 1) of the equity in cash.
    # Barroso-Santa-Clara vol management, capped so bull markets stay ~fully
    # invested — it never scales UP past 100%, only down.
    sim_daily_core_target_vol: float = 0.0
    # vol_weight: position targets proportional to 1/realized-vol (risk parity)
    # instead of equal weight. Independent of mom_variant.
    sim_daily_core_vol_weight: bool = False
    # lowvol_tilt: adds pct_rank(-vol) as a 4th equal term in the qv-mom
    # score. The classic defensive tilt — expect lower vol AND lower return;
    # kept for completeness (the sweep decides).
    sim_daily_core_lowvol_tilt: bool = False
    # portfolio_stop_pct (0 = off): peak-to-trough circuit breaker — when the
    # strategy's own equity is this % below its running peak, sell everything
    # to cash and park contributions until the market trend recovers. The
    # "don't give the win back" brake. Wired live only if the walk-forward
    # says the avoided drawdown beats the missed rebound (§12).
    sim_daily_core_portfolio_stop: float = 0.0
    # exposure_trend_days (0 = off): deploy cash only while the equal-weight
    # universe index is above its N-day SMA (Faber-style). Also the re-entry
    # gate after a portfolio-stop trigger.
    sim_daily_core_exposure_trend: int = 0
    # trailing_stop_pct (0 = off): per-name trailing stop — exit a holding when
    # its price falls this fraction below its own peak since entry. Targets
    # momentum-sleeve crashes the market-trend brakes cannot see (§12).
    sim_daily_core_trailing_stop: float = 0.0
    # basket_trend_days (0 = off): "gradient filter" — N-day rate-of-change of
    # the strategy's own target basket (equal-weight top-N candidates). Cash
    # out when the gradient stays negative for basket_confirm_days, re-enter
    # when it stays positive. §12-13.
    sim_daily_core_basket_trend: int = 0
    sim_daily_core_basket_confirm: int = 3
    # basket_threshold (fraction, 0 = any negative slope): the gradient must
    # be BELOW -threshold (a real drawdown, not noise) to count toward the
    # cash-out streak. Re-entry stays on any positive slope for the same
    # confirm streak — sell on deep drops, re-enter on the recovery. This is
    # the "on demand" switch: small dips in calm markets no longer trigger.
    sim_daily_core_basket_threshold: float = 0.0
    # basket_drawdown (fraction, 0 = off): cash out when the target basket is
    # this far below its own running peak; re-enter when the drawdown halves
    # (built-in hysteresis — one event per real drawdown, no slope whipsaw).
    # The basket keeps moving in cash, so re-entry can trigger.
    sim_daily_core_basket_drawdown: float = 0.0
    # basket_er_min (0 = off): Kaufman efficiency ratio gate — only ARM the
    # gradient/drawdown cash-out when the basket's recent path is efficient
    # (|net move| / path length above this). Trend-following pays in
    # efficient trends and whipsaws in chop; the ER is the classic
    # distinguisher. §13.
    sim_daily_core_basket_er_min: float = 0.0
    # basket_good_times (bool): arm the gradient cash-out ONLY while the
    # market is above its 200-day SMA — "gradient filter only in good times".
    # Catches momentum-sleeve crashes in healthy bull markets (Jul 2026) while
    # the §12 market-trend gate handles bad times; no cooldown, the normal
    # deployment re-enters as soon as the signal allows. §15.
    sim_daily_core_basket_good_times: bool = False
    # basket_arm_sma (default 200): the market SMA window that defines "good
    # times" for arming the gradient cash-out. 50 = only strong uptrends,
    # 200 = the classic bull/bear line. §15.
    sim_daily_core_basket_arm_sma: int = 200
    sim_run_hour: int = 22
    sim_run_minute: int = 30
    # Nightly shared universe prefetch: all sims share one universe, so fetch it
    # ONCE before the cycles and let them read the DB instead of each pulling it.
    # Uses the configured provider (auto → Twelve Data, rate-limited/budgeted)
    # with a yfinance fallback per ticker. Starts `sim_prefetch_lead_minutes`
    # before sim_run_hour; the cycles skip tickers fetched within
    # `market_fresh_seconds`.
    sim_universe_prefetch: bool = True
    sim_prefetch_lead_minutes: int = 90
    market_fresh_seconds: int = 21600  # 6h: "already fetched, skip the re-pull"
    # Nightly watchlist prefetch (app-level, NOT sim-level): fetch the watchlist
    # with the configured provider each night so the Dashboard opens straight
    # from the DB. Runs independently of SIM_ENABLED / SIM_UNIVERSE_PREFETCH, at
    # sim_run_hour (the app's nightly refresh time). Overlaps are skipped via
    # MARKET_FRESH_SECONDS; the Twelve Data limiter paces/falls back.
    watchlist_prefetch: bool = True

    # --- Benchmark (DCA control portfolio) --------------------------------
    sim_benchmark_enabled: bool = True
    sim_benchmark_ticker: str = "URTH"  # iShares MSCI World ETF

    # --- Monthly qv-mom portfolio (separate paper portfolio) ---------------
    # stockstrat qv-mom-v1: top-10 quality-value-momentum, monthly rebalance
    # on the last trading day, hysteresis band, equal weight. Research record:
    # stockstrat backtest 2016-11..2026-08 on diversified-plus (CAGR 29.9%)
    # and S&P 500 (CAGR 20.4%); on the small curated universe the result sat
    # inside the random-10 band — the edge is demonstrated on broad universes.
    sim_monthly_enabled: bool = True
    sim_monthly_universe: str = "diversified-plus"
    sim_monthly_contribution: float = 1000.0
    sim_monthly_start_cash: float = 0.0
    sim_monthly_target_n: int = 10          # portfolio size
    sim_monthly_hold_band: int = 20         # hysteresis: held names stay while in top-N of the ranking
    sim_monthly_min_mcap: float = 5e9       # USD
    sim_monthly_min_dollar_vol: float = 1e7  # USD, 20-day avg of close*volume
    sim_monthly_min_history_days: int = 253  # trading days of valid closes required
    sim_monthly_cost_oneway: float = 0.0010  # paper friction, one-way bps on notional swapped
    sim_monthly_fundamentals_source: str = "edgar"  # edgar (US filers) + yfinance fallback | yfinance-only

    # --- Daily-core portfolio (qv-mom core + daily cash deployment) ---------
    # §10 A/B winner: monthly qv-mom ranking decides WHAT to own (same
    # top-N hysteresis, no stops); candles only deploy cash daily into the
    # top-ranked names toward equal weight. Measured +3.4..+7.5pp IRR over
    # the monthly sim on 2020-2026 windows (cash-drag elimination); positive
    # out-of-sample on all four walk-forward windows (see §10).
    sim_daily_core_enabled: bool = True

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
    )

settings = Settings()
