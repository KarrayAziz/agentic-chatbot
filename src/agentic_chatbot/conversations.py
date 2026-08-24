"""Application-owned conversation metadata storage."""

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from agentic_chatbot.config import PROJECT_ROOT

DEFAULT_TITLE = "New conversation"
MAX_TITLE_LENGTH = 60


@dataclass(frozen=True, slots=True)
class Conversation:
    """Application metadata for one LangGraph thread."""

    thread_id: str
    title: str
    created_at: str
    updated_at: str


def title_from_message(message: str) -> str:
    """Create a deterministic title from the first non-empty user message."""

    normalized = " ".join(message.split())
    if not normalized:
        return DEFAULT_TITLE
    if len(normalized) <= MAX_TITLE_LENGTH:
        return normalized
    return f"{normalized[: MAX_TITLE_LENGTH - 3].rstrip()}..."


class ConversationRepository:
    """CRUD operations for the application-owned conversations table."""

    def __init__(self, connection: sqlite3.Connection) -> None:
        self.connection = connection
        self.connection.row_factory = sqlite3.Row
        self._setup()

    def _setup(self) -> None:
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS conversations (
                thread_id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        self.connection.commit()

    @staticmethod
    def _from_row(row: sqlite3.Row) -> Conversation:
        return Conversation(
            thread_id=row["thread_id"],
            title=row["title"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    def create(self, thread_id: str, title: str = DEFAULT_TITLE) -> Conversation:
        now = datetime.now(UTC).isoformat()
        self.connection.execute(
            """
            INSERT INTO conversations (thread_id, title, created_at, updated_at)
            VALUES (?, ?, ?, ?)
            """,
            (thread_id, title, now, now),
        )
        self.connection.commit()
        return Conversation(thread_id, title, now, now)

    def get(self, thread_id: str) -> Conversation | None:
        row = self.connection.execute(
            """
            SELECT thread_id, title, created_at, updated_at
            FROM conversations
            WHERE thread_id = ?
            """,
            (thread_id,),
        ).fetchone()
        return self._from_row(row) if row is not None else None

    def list_all(self) -> list[Conversation]:
        rows = self.connection.execute(
            """
            SELECT thread_id, title, created_at, updated_at
            FROM conversations
            ORDER BY updated_at DESC, created_at DESC
            """
        ).fetchall()
        return [self._from_row(row) for row in rows]

    def record_user_message(self, thread_id: str, message: str) -> Conversation:
        """Set the initial title once and refresh the activity timestamp."""

        now = datetime.now(UTC).isoformat()
        generated_title = title_from_message(message)
        cursor = self.connection.execute(
            """
            UPDATE conversations
            SET title = CASE WHEN title = ? THEN ? ELSE title END,
                updated_at = ?
            WHERE thread_id = ?
            """,
            (DEFAULT_TITLE, generated_title, now, thread_id),
        )
        self.connection.commit()
        if cursor.rowcount == 0:
            raise KeyError(f"Conversation '{thread_id}' does not exist.")
        return self._require(thread_id)

    def rename(self, thread_id: str, title: str) -> Conversation:
        title = " ".join(title.split())
        if not title:
            raise ValueError("Conversation title cannot be empty.")
        if len(title) > MAX_TITLE_LENGTH:
            raise ValueError(
                f"Conversation title cannot exceed {MAX_TITLE_LENGTH} characters."
            )

        now = datetime.now(UTC).isoformat()
        cursor = self.connection.execute(
            """
            UPDATE conversations SET title = ?, updated_at = ? WHERE thread_id = ?
            """,
            (title, now, thread_id),
        )
        self.connection.commit()
        if cursor.rowcount == 0:
            raise KeyError(f"Conversation '{thread_id}' does not exist.")
        return self._require(thread_id)

    def delete(self, thread_id: str) -> bool:
        cursor = self.connection.execute(
            "DELETE FROM conversations WHERE thread_id = ?",
            (thread_id,),
        )
        self.connection.commit()
        return cursor.rowcount > 0

    def _require(self, thread_id: str) -> Conversation:
        conversation = self.get(thread_id)
        if conversation is None:
            raise KeyError(f"Conversation '{thread_id}' does not exist.")
        return conversation


def resolve_conversation_db_path(path: Path) -> Path:
    """Resolve configured relative paths from the project root."""

    return path if path.is_absolute() else PROJECT_ROOT / path


@contextmanager
def open_conversation_repository(path: Path) -> Iterator[ConversationRepository]:
    """Open the application metadata database for a bounded operation."""

    database_path = resolve_conversation_db_path(path)
    database_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(database_path)
    try:
        yield ConversationRepository(connection)
    finally:
        connection.close()
