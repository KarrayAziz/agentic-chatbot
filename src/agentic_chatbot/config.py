"""Typed application configuration loaded from the environment."""

from pathlib import Path
from typing import Literal

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Environment-backed settings shared by application components.

    API keys are optional in Phase 0 so the foundation can start without external
    services. Future phases can validate a key when its integration is used.
    """

    google_api_key: SecretStr | None = None
    langsmith_tracing: bool = False
    langsmith_api_key: SecretStr | None = None
    langsmith_project: str = "agentic-chatbot"
    langsmith_endpoint: str = "https://api.smith.langchain.com"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


def load_settings() -> Settings:
    """Load a fresh settings object from environment variables and `.env`."""

    return Settings()
