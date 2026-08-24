"""Tests for Tavily search configuration without making API calls."""

import pytest

from agentic_chatbot.config import Settings
from agentic_chatbot.tools.web_search import create_web_search_tool


def test_web_search_requires_tavily_api_key() -> None:
    settings = Settings(_env_file=None, tavily_api_key=None)

    with pytest.raises(ValueError, match="TAVILY_API_KEY is required"):
        create_web_search_tool(settings)


def test_web_search_tool_has_bounded_results(monkeypatch) -> None:
    monkeypatch.setenv("TAVILY_API_KEY", "test-key")
    settings = Settings(_env_file=None)

    search = create_web_search_tool(settings)

    assert search.name == "tavily_search"
    assert search.max_results == 5
    assert search.search_depth == "basic"
