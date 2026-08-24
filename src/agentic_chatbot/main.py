"""Executable entry point for the Gemini chatbot."""

import argparse
import logging
from collections.abc import Sequence
from uuid import UUID, uuid4

from agentic_chatbot.cli import run_chat_cli
from agentic_chatbot.config import Settings, load_settings
from agentic_chatbot.conversations import open_conversation_repository
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
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument(
        "--thread-id",
        type=_uuid_argument,
        help="Resume an existing conversation UUID; omit to create a new one.",
    )
    actions.add_argument(
        "--list-conversations",
        action="store_true",
        help="List saved conversation metadata and exit.",
    )
    actions.add_argument(
        "--rename",
        nargs=2,
        metavar=("THREAD_ID", "TITLE"),
        help="Rename a saved conversation and exit.",
    )
    actions.add_argument(
        "--delete",
        type=_uuid_argument,
        metavar="THREAD_ID",
        help="Delete conversation metadata and its LangGraph checkpoints.",
    )
    return parser.parse_args(argv)


def _handle_management_command(args: argparse.Namespace, settings: Settings) -> bool:
    """Run a metadata management command, returning whether one was handled."""

    if args.list_conversations:
        with open_conversation_repository(settings.conversation_db_path) as repository:
            conversations = repository.list_all()
        if not conversations:
            print("No saved conversations.")
        else:
            for conversation in conversations:
                print(
                    f"{conversation.thread_id} | {conversation.title} | "
                    f"updated {conversation.updated_at}"
                )
        return True

    if args.rename:
        try:
            thread_id = _uuid_argument(args.rename[0])
        except argparse.ArgumentTypeError as error:
            LOGGER.error("%s", error)
            raise SystemExit(2) from error
        with open_conversation_repository(settings.conversation_db_path) as repository:
            try:
                conversation = repository.rename(thread_id, args.rename[1])
            except (KeyError, ValueError) as error:
                LOGGER.error("%s", error.args[0])
                raise SystemExit(2) from error
        print(f"Renamed conversation {thread_id} to: {conversation.title}")
        return True

    if args.delete:
        thread_id = args.delete
        with open_conversation_repository(settings.conversation_db_path) as repository:
            if repository.get(thread_id) is None:
                LOGGER.error("Conversation '%s' does not exist.", thread_id)
                raise SystemExit(2)

            # Use LangGraph's public API; never edit its internal SQLite tables.
            with open_sqlite_checkpointer(settings.checkpoint_db_path) as checkpointer:
                checkpointer.delete_thread(thread_id)
            repository.delete(thread_id)
        print(f"Deleted conversation {thread_id} and its checkpoints.")
        return True

    return False


def main(argv: Sequence[str] | None = None) -> None:
    """Configure the application, build the graph, and start the chat loop."""

    args = _parse_args(argv)
    settings = load_settings()
    configure_logging(settings.log_level)

    if _handle_management_command(args, settings):
        return

    thread_id = args.thread_id or str(uuid4())

    try:
        model = create_gemini_model(settings)
        tools = build_tools(settings)
    except ValueError as error:
        LOGGER.error("%s", error)
        raise SystemExit(2) from error

    with open_conversation_repository(settings.conversation_db_path) as repository:
        if repository.get(thread_id) is None:
            repository.create(thread_id)

        with open_sqlite_checkpointer(settings.checkpoint_db_path) as checkpointer:
            graph = build_chat_graph(model, tools, checkpointer=checkpointer)
            run_chat_cli(
                graph,
                thread_id,
                on_user_message=lambda message: repository.record_user_message(
                    thread_id, message
                ),
            )


if __name__ == "__main__":
    main()
