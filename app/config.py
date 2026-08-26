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
    searxng_url: str = ""  # e.g. "http://192.168.0.46:8081"; empty = disabled
    searxng_timeout: float = 10.0

    paper_trading: bool = True

    # --- Autonomous paper-trading simulation -----------------------------
    sim_enabled: bool = True
    sim_monthly_allowance: float = 1000.0
    sim_start_cash: float = 0.0
    sim_universe: str = "global-large-cap"
    sim_strategy: str = "deterministic"  # deterministic | llm | hybrid
    sim_max_position_pct: float = 10.0
    sim_min_cash_pct: float = 5.0
    sim_max_positions: int = 10
    sim_stop_pct: float = 15.0
    sim_max_run_5d: float = 12.0  # block BUYs after a 5-day run-up > this % (0 = disabled)
    sim_llm_review_interval: int = 1  # consult the LLM every N cycles (1=daily, 5=weekly)
    sim_llm_minimal_prompt: bool = False  # use the minimal system prompt (no methodology/regime rules)
    sim_llm_failure_marker: bool = False  # hybrid: consult the LLM only on engine failure (stop-out cascade / drawdown)
    sim_run_hour: int = 22
    sim_run_minute: int = 30

    # --- Benchmark (DCA control portfolio) --------------------------------
    sim_benchmark_enabled: bool = True
    sim_benchmark_ticker: str = "URTH"  # iShares MSCI World ETF

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
    )

settings = Settings()