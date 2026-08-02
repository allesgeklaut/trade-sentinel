from pydantic_settings import BaseSettings
class Settings(BaseSettings):
    database_url: str = "sqlite+aiosqlite:////data/trading.db"
    market_data_provider: str = "yfinance"
    twelve_data_api_key: str = ""
    watchlist: str = "AAPL,MSFT,NVDA,VOO,IFX.DE"
    ollama_url: str = "http://host.docker.internal:11434"
    ollama_model: str = "qwen3.6:27b-q3_K_M"
    paper_trading: bool = True
    class Config: env_file = ".env"
settings=Settings()
