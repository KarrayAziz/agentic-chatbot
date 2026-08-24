"""Tests for the in-memory CLI conversation loop."""

from langchain_core.language_models.fake_chat_models import FakeListChatModel

from agentic_chatbot.cli import run_chat_cli
from agentic_chatbot.graph import build_chat_graph


def test_cli_prints_response_and_quits() -> None:
    graph = build_chat_graph(FakeListChatModel(responses=["A test response."]))
    user_inputs = iter(["Hello", "quit"])
    output: list[str] = []

    run_chat_cli(
        graph,
        input_fn=lambda prompt: next(user_inputs),
        output_fn=output.append,
    )

    assert "Assistant: A test response." in output
    assert output[-1] == "Goodbye!"
