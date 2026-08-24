"""Tests for application wiring without making an external model call."""

from contextlib import nullcontext
from uuid import UUID

from langchain_core.language_models.fake_chat_models import FakeListChatModel

import agentic_chatbot.main as main_module


def test_main_builds_graph_and_starts_cli(monkeypatch) -> None:
    fake_model = FakeListChatModel(responses=["unused"])
    fake_tools = []
    fake_graph = object()
    fake_checkpointer = object()
    received_conversations = []

    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    monkeypatch.setattr(main_module, "create_gemini_model", lambda settings: fake_model)
    monkeypatch.setattr(main_module, "build_tools", lambda settings: fake_tools)
    monkeypatch.setattr(
        main_module,
        "open_sqlite_checkpointer",
        lambda path: nullcontext(fake_checkpointer),
    )
    monkeypatch.setattr(
        main_module,
        "build_chat_graph",
        lambda model, tools, checkpointer: fake_graph,
    )
    monkeypatch.setattr(
        main_module,
        "run_chat_cli",
        lambda graph, thread_id: received_conversations.append((graph, thread_id)),
    )

    main_module.main([])

    assert len(received_conversations) == 1
    assert received_conversations[0][0] is fake_graph
    assert str(UUID(received_conversations[0][1])) == received_conversations[0][1]
