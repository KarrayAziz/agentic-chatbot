"""Tests for environment-backed application settings."""

from agentic_chatbot.config import Settings


def test_settings_have_safe_defaults() -> None:
    settings = Settings(_env_file=None)

    assert settings.google_api_key is None
    assert settings.gemini_model == "gemini-2.5-flash"
    assert settings.tavily_api_key is None
    assert settings.checkpoint_db_path.name == "langgraph_checkpoints.sqlite"
    assert settings.conversation_db_path.name == "conversations.sqlite"
    assert settings.conversation_db_path != settings.checkpoint_db_path
    assert settings.langsmith_tracing is False
    assert settings.langsmith_project == "agentic-chatbot"
    assert settings.log_level == "INFO"


def test_settings_read_environment_variables(monkeypatch) -> None:
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")

    settings = Settings(_env_file=None)

    assert settings.google_api_key is not None
    assert settings.google_api_key.get_secret_value() == "test-key"
    assert settings.langsmith_tracing is True
    assert settings.log_level == "DEBUG"
