"""Tools available to the chatbot agent."""

from langchain_core.tools import BaseTool

from agentic_chatbot.config import Settings
from agentic_chatbot.tools.calculator import calculator
from agentic_chatbot.tools.paper_trading import paper_buy_stock
from agentic_chatbot.tools.weather import get_current_weather
from agentic_chatbot.tools.web_search import create_web_search_tool


def build_tools(settings: Settings) -> list[BaseTool]:
    """Build the complete tool list that is bound to Gemini and ToolNode."""

    return [
        calculator,
        get_current_weather,
        create_web_search_tool(settings),
        paper_buy_stock,
    ]


__all__ = ["build_tools", "calculator", "get_current_weather", "paper_buy_stock"]
