"""Executable entry point for the Gemini chatbot."""

import argparse
import logging
from collections.abc import Sequence
from uuid import UUID, uuid4

from agentic_chatbot.cli import run_chat_cli
from agentic_chatbot.config import load_settings
from agentic_chatbot.graph import build_chat_graph
from agentic_chatbot.logging_config import configure_logging
from agentic_chatbot.model import create_gemini_model
from agentic_chatbot.persistence import open_sqlite_checkpointer
from agentic_chatbot.tools import build_tools

LOGGER = logging.getLogger(__name__)


def _uuid_argument(value: str) -> str:
    """Validate and normalize a conversation thread UUID."""

    try:
        return str(UUID(value))
    except ValueError as error:
        raise argparse.ArgumentTypeError("thread ID must be a valid UUID") from error


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Agentic AI chatbot.")
    parser.add_argument(
        "--thread-id",
        type=_uuid_argument,
        help="Resume an existing conversation UUID; omit to create a new one.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Configure the application, build the graph, and start the chat loop."""

    args = _parse_args(argv)
    settings = load_settings()
    configure_logging(settings.log_level)
    thread_id = args.thread_id or str(uuid4())

    try:
        model = create_gemini_model(settings)
        tools = build_tools(settings)
    except ValueError as error:
        LOGGER.error("%s", error)
        raise SystemExit(2) from error

    with open_sqlite_checkpointer(settings.checkpoint_db_path) as checkpointer:
        graph = build_chat_graph(model, tools, checkpointer=checkpointer)
        run_chat_cli(graph, thread_id)


if __name__ == "__main__":
    main()
