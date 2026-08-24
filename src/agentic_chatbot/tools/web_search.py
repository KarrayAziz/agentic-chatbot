"""Tavily web-search tool construction."""

from langchain_core.tools import BaseTool
from langchain_tavily import TavilySearch

from agentic_chatbot.config import Settings


def create_web_search_tool(settings: Settings) -> BaseTool:
    """Create a concise Tavily search tool using the configured API key."""

    if (
        settings.tavily_api_key is None
        or not settings.tavily_api_key.get_secret_value().strip()
    ):
        raise ValueError(
            "TAVILY_API_KEY is required for web search. Add it to your .env file."
        )

    return TavilySearch(
        max_results=5,
        search_depth="basic",
        include_answer=False,
        include_raw_content=False,
        topic="general",
    )
