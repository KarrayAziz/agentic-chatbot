"""Tests for persistent, thread-isolated LangGraph checkpoints."""

from pathlib import Path
from typing import Any
from uuid import uuid4

from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage

from agentic_chatbot.graph import build_chat_graph
from agentic_chatbot.persistence import open_sqlite_checkpointer
from agentic_chatbot.tools.calculator import calculator


class CheckpointFakeModel(FakeMessagesListChatModel):
    """Fake model that accepts the graph's tool binding."""

    def bind_tools(self, tools: Any, **kwargs: Any) -> "CheckpointFakeModel":
        return self


def _config(thread_id: str) -> dict[str, dict[str, str]]:
    return {"configurable": {"thread_id": thread_id}}


def test_messages_survive_checkpointer_reopen(tmp_path: Path) -> None:
    database_path = tmp_path / "langgraph_checkpoints.sqlite"
    thread_id = str(uuid4())

    with open_sqlite_checkpointer(database_path) as checkpointer:
        graph = build_chat_graph(
            CheckpointFakeModel(responses=[AIMessage("I will remember.")]),
            [calculator],
            checkpointer=checkpointer,
        )
        graph.invoke(
            {"messages": [HumanMessage("My name is Aziz.")]},
            config=_config(thread_id),
        )

    assert database_path.exists()

    with open_sqlite_checkpointer(database_path) as checkpointer:
        graph = build_chat_graph(
            CheckpointFakeModel(responses=[AIMessage("Your name is Aziz.")]),
            [calculator],
            checkpointer=checkpointer,
        )
        result = graph.invoke(
            {"messages": [HumanMessage("What is my name?")]},
            config=_config(thread_id),
        )

    assert [message.content for message in result["messages"]] == [
        "My name is Aziz.",
        "I will remember.",
        "What is my name?",
        "Your name is Aziz.",
    ]


def test_different_thread_ids_have_separate_state(tmp_path: Path) -> None:
    database_path = tmp_path / "langgraph_checkpoints.sqlite"
    first_thread_id = str(uuid4())
    second_thread_id = str(uuid4())

    with open_sqlite_checkpointer(database_path) as checkpointer:
        graph = build_chat_graph(
            CheckpointFakeModel(
                responses=[AIMessage("First reply"), AIMessage("Second reply")]
            ),
            [calculator],
            checkpointer=checkpointer,
        )
        first_result = graph.invoke(
            {"messages": [HumanMessage("First thread message")]},
            config=_config(first_thread_id),
        )
        second_result = graph.invoke(
            {"messages": [HumanMessage("Second thread message")]},
            config=_config(second_thread_id),
        )

    assert [message.content for message in first_result["messages"]] == [
        "First thread message",
        "First reply",
    ]
    assert [message.content for message in second_result["messages"]] == [
        "Second thread message",
        "Second reply",
    ]
