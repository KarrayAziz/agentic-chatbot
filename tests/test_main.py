"""Tests for application wiring without making an external model call."""

from langchain_core.language_models.fake_chat_models import FakeListChatModel

import agentic_chatbot.main as main_module


def test_main_builds_graph_and_starts_cli(monkeypatch) -> None:
    fake_model = FakeListChatModel(responses=["unused"])
    received_graphs = []

    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    monkeypatch.setattr(main_module, "create_gemini_model", lambda settings: fake_model)
    monkeypatch.setattr(
        main_module,
        "run_chat_cli",
        lambda graph: received_graphs.append(graph),
    )

    main_module.main()

    assert len(received_graphs) == 1
