"""Tests for the in-memory CLI conversation loop."""

from langchain_core.messages import AIMessage

from agentic_chatbot.cli import run_chat_cli


class StubGraph:
    """Return a fixed assistant response without invoking a model."""

    def invoke(self, state):
        return {"messages": [*state["messages"], AIMessage("A test response.")]}


def test_cli_prints_response_and_quits() -> None:
    user_inputs = iter(["Hello", "quit"])
    output: list[str] = []

    run_chat_cli(
        StubGraph(),
        input_fn=lambda prompt: next(user_inputs),
        output_fn=output.append,
    )

    assert "Assistant: A test response." in output
    assert output[-1] == "Goodbye!"
