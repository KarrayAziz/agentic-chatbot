"""Tests for application-owned conversation metadata storage."""

from pathlib import Path
from uuid import uuid4

from agentic_chatbot.conversations import (
    DEFAULT_TITLE,
    MAX_TITLE_LENGTH,
    open_conversation_repository,
    title_from_message,
)


def test_repository_crud_persists_across_reopen(tmp_path: Path) -> None:
    database_path = tmp_path / "conversations.sqlite"
    thread_id = str(uuid4())

    with open_conversation_repository(database_path) as repository:
        created = repository.create(thread_id)
        assert created.title == DEFAULT_TITLE

        titled = repository.record_user_message(
            thread_id, "  Explain   LangGraph checkpoints  "
        )
        assert titled.title == "Explain LangGraph checkpoints"

        renamed = repository.rename(thread_id, "Persistence lesson")
        assert renamed.title == "Persistence lesson"
        assert renamed.updated_at >= created.updated_at

    with open_conversation_repository(database_path) as repository:
        conversations = repository.list_all()
        assert [conversation.thread_id for conversation in conversations] == [
            thread_id
        ]
        assert conversations[0].title == "Persistence lesson"
        assert repository.delete(thread_id) is True
        assert repository.get(thread_id) is None


def test_only_first_message_generates_title(tmp_path: Path) -> None:
    database_path = tmp_path / "conversations.sqlite"
    thread_id = str(uuid4())

    with open_conversation_repository(database_path) as repository:
        repository.create(thread_id)
        repository.record_user_message(thread_id, "First question")
        conversation = repository.record_user_message(thread_id, "Second question")

    assert conversation.title == "First question"


def test_title_generation_is_deterministic_and_bounded() -> None:
    message = "A   very long question " * 10

    title = title_from_message(message)

    assert len(title) == MAX_TITLE_LENGTH
    assert title.endswith("...")
