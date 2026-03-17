from functools import lru_cache
from typing import List
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Google Gemini - FREE at https://aistudio.google.com/app/apikey
    GEMINI_API_KEY: str = ""
    GEMINI_MODEL: str = "gemini-2.5-flash"
    EMBEDDING_MODEL: str = "models/text-embedding-004"

    # Database
    DATABASE_URL: str = "sqlite+aiosqlite:///./codex.db"

    # CORS
    ALLOWED_ORIGINS: str = "*"

    # Limits
    MAX_FILE_SIZE_MB: int = 10
    MAX_REPO_SIZE_MB: int = 200
    MAX_FILES_PER_REPO: int = 3000
    CHUNK_SIZE_TOKENS: int = 800
    CHUNK_OVERLAP_TOKENS: int = 100
    TOP_K_RETRIEVAL: int = 6

    @property
    def async_database_url(self) -> str:
        url = self.DATABASE_URL
        if url.startswith("postgres://"):
            url = url.replace("postgres://", "postgresql+asyncpg://", 1)
        elif url.startswith("postgresql://"):
            url = url.replace("postgresql://", "postgresql+asyncpg://", 1)
        return url

    @property
    def is_sqlite(self) -> bool:
        return "sqlite" in self.async_database_url

    @property
    def cors_origins(self) -> List[str]:
        if self.ALLOWED_ORIGINS == "*":
            return ["*"]
        return [o.strip() for o in self.ALLOWED_ORIGINS.split(",")]


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
