# app/config.py
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from pathlib import Path

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(Path(__file__).resolve().parent.parent / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    APP_ENV: str = Field(default="dev")
    LOG_LEVEL: str = Field(default="INFO")

    PG_DSN: str
    OPENAI_API_KEY: str | None = None
    EMBED_MODEL: str = Field(default="text-embedding-3-small")

    # These exist only if you added retries:
    LLM_MODEL_RERANK: str = Field(default="gpt-4o-mini")
    LLM_MODEL_ANSWER: str = Field(default="gpt-4o-mini")
    LLM_TIMEOUT_S: int = Field(default=20)
    LLM_MAX_RETRIES: int = Field(default=3)

    # Redis (if using Redis sessions)
    REDIS_URL: str = Field(default="redis://localhost:6379/0")

settings = Settings()
