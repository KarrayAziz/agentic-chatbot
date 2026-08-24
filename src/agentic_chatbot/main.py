"""Executable entry point for the Gemini chatbot."""

import logging

from agentic_chatbot.cli import run_chat_cli
from agentic_chatbot.config import load_settings
from agentic_chatbot.graph import build_chat_graph
from agentic_chatbot.logging_config import configure_logging
from agentic_chatbot.model import create_gemini_model

LOGGER = logging.getLogger(__name__)


def main() -> None:
    """Configure the application, build the graph, and start the chat loop."""

    settings = load_settings()
    configure_logging(settings.log_level)

    try:
        model = create_gemini_model(settings)
    except ValueError as error:
        LOGGER.error("%s", error)
        raise SystemExit(2) from error

    graph = build_chat_graph(model)
    run_chat_cli(graph)


if __name__ == "__main__":
    main()
