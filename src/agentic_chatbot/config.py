"""Typed application configuration loaded from the environment."""

from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Environment-backed settings shared by application components.

    The Gemini key remains optional while settings are parsed, which keeps config
    importable in tests. The model factory validates it before creating a client.
    """

    google_api_key: SecretStr | None = None
    gemini_model: str = "gemini-2.5-flash"
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
    """Load `.env` into the process, then return typed application settings.

    Loading values into the process environment lets LangChain and LangGraph's
    built-in LangSmith integration discover the standard ``LANGSMITH_*`` values.
    Existing shell environment variables take precedence over `.env` values.
    """

    load_dotenv(PROJECT_ROOT / ".env", override=False)
    return Settings()
