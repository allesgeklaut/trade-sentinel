from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    database_url: str = "sqlite+aiosqlite:////data/trading.db"
    market_data_provider: str = "yfinance"
    twelve_data_api_key: str = ""
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
    sim_monthly_rank_boost: float = 0.5
    sim_run_hour: int = 22
    sim_run_minute: int = 30

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
    # top-ranked names toward equal weight. Measured +3..+5.5pp IRR over
    # the monthly sim on 2020-2026 windows (cash-drag elimination).
    sim_daily_core_enabled: bool = True

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
    )

settings = Settings()
