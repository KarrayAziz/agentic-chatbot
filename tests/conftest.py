"""Shared test isolation from developer-machine credentials and tracing."""

from collections.abc import Iterator

import pytest
from pytest import MonkeyPatch


@pytest.fixture(autouse=True)
def isolate_application_environment(monkeypatch: MonkeyPatch) -> Iterator[None]:
    """Keep tests deterministic and prevent accidental network tracing."""
    for variable in (
        "GOOGLE_API_KEY",
        "GEMINI_MODEL",
        "GEMINI_EMBEDDING_MODEL",
        "TAVILY_API_KEY",
        "LANGSMITH_API_KEY",
        "LANGSMITH_PROJECT",
        "LOG_LEVEL",
    ):
        monkeypatch.delenv(variable, raising=False)
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    yield
