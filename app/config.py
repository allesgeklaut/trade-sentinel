from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    database_url: str = "sqlite+aiosqlite:////data/trading.db"
    market_data_provider: str = "yfinance"
    twelve_data_api_key: str = ""
    watchlist: str = "AAPL,MSFT,NVDA,IFX.DE"

    ollama_url: str = "http://host.docker.internal:11434"
    ollama_model: str = ""  # set via .env, e.g. "qwen3:32b"
    ollama_timeout_seconds: float = 180.0

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