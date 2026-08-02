from pydantic_settings import BaseSettings
class Settings(BaseSettings):
    database_url: str = "sqlite+aiosqlite:////data/trading.db"
    twelve_data_api_key: str = ""
    watchlist: str = "AAPL,MSFT,NVDA,VOO"
    ollama_url: str = "http://host.docker.internal:11434"
    ollama_model: str = "qwen3.6:27b-q3_K_M"
    refresh_minutes: int = 60
    paper_trading: bool = True
    max_notional_per_order: float = 1000
    max_daily_loss: float = 100
    class Config: env_file = ".env"
settings = Settings()
