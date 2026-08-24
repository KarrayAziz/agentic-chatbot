"""LangGraph checkpoint persistence resources."""

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver

from agentic_chatbot.config import PROJECT_ROOT


def resolve_checkpoint_path(path: Path) -> Path:
    """Resolve configured relative paths from the project root."""

    return path if path.is_absolute() else PROJECT_ROOT / path


@contextmanager
def open_sqlite_checkpointer(path: Path) -> Iterator[SqliteSaver]:
    """Open a persistent SQLite checkpointer and close it on application exit."""

    database_path = resolve_checkpoint_path(path)
    database_path.parent.mkdir(parents=True, exist_ok=True)
    with SqliteSaver.from_conn_string(str(database_path)) as checkpointer:
        yield checkpointer
