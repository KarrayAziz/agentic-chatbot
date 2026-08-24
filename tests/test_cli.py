"""Tests for the in-memory CLI conversation loop."""

from typing import Any

from langchain_core.language_models.fake_chat_models import (
    FakeMessagesListChatModel,
    GenericFakeChatModel,
)
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver

from agentic_chatbot.cli import run_chat_cli
from agentic_chatbot.graph import build_chat_graph
from agentic_chatbot.tools.calculator import calculator
from agentic_chatbot.tools.paper_trading import paper_buy_stock


class StreamingFakeModel(GenericFakeChatModel):
    """Fake model that streams text chunks and accepts tool binding."""

    def bind_tools(self, tools: Any, **kwargs: Any) -> "StreamingFakeModel":
        return self


class ToolCallingFakeModel(FakeMessagesListChatModel):
    """Fake model that preserves structured tool calls."""

    def bind_tools(self, tools: Any, **kwargs: Any) -> "ToolCallingFakeModel":
        return self


def test_cli_streams_response_incrementally_without_final_duplicate() -> None:
    model = StreamingFakeModel(messages=iter(["LangGraph streams progressively."]))
    graph = build_chat_graph(model, [calculator], checkpointer=InMemorySaver())
    user_inputs = iter(["Hello", "quit"])
    output_lines: list[str] = []
    streamed_writes: list[str] = []
    recorded_messages: list[str] = []

    run_chat_cli(
        graph,
        "00000000-0000-0000-0000-000000000001",
        on_user_message=recorded_messages.append,
        input_fn=lambda prompt: next(user_inputs),
        output_fn=output_lines.append,
        write_fn=streamed_writes.append,
    )

    rendered = "".join(streamed_writes)
    assert rendered == "Assistant: LangGraph streams progressively.\n"
    assert len(streamed_writes) > 3
    assert rendered.count("LangGraph streams progressively.") == 1
    assert recorded_messages == ["Hello"]
    assert output_lines[1] == (
        "Conversation ID: 00000000-0000-0000-0000-000000000001"
    )
    assert output_lines[-1] == "Goodbye!"


def test_cli_displays_paper_trade_and_resumes_after_rejection() -> None:
    tool_request = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "paper_buy_stock",
                "args": {"ticker": "NVDA", "quantity": 3},
                "id": "paper-order-1",
                "type": "tool_call",
            }
        ],
    )
    model = ToolCallingFakeModel(
        responses=[tool_request, AIMessage("The simulated purchase was rejected.")]
    )
    graph = build_chat_graph(
        model, [paper_buy_stock], checkpointer=InMemorySaver()
    )
    user_inputs = iter(["Paper buy 3 shares of NVDA", "reject", "quit"])
    output_lines: list[str] = []
    streamed_writes: list[str] = []

    run_chat_cli(
        graph,
        "00000000-0000-0000-0000-000000000002",
        input_fn=lambda prompt: next(user_inputs),
        output_fn=output_lines.append,
        write_fn=streamed_writes.append,
    )

    assert "Approval required — PAPER TRADING ONLY" in output_lines
    assert "Action: paper_buy_stock" in output_lines
    assert "Ticker: NVDA" in output_lines
    assert "Quantity: 3" in output_lines
    assert "".join(streamed_writes) == (
        "Assistant: The simulated purchase was rejected.\n"
    )
    assert output_lines[-1] == "Goodbye!"
