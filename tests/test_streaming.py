"""Tests for sanitized LangGraph-to-client streaming events."""

from typing import Any

from langchain_core.language_models.fake_chat_models import (
    FakeMessagesListChatModel,
    GenericFakeChatModel,
)
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

from agentic_chatbot.graph import build_chat_graph
from agentic_chatbot.streaming import stream_agent_events
from agentic_chatbot.tools.calculator import calculator
from agentic_chatbot.tools.paper_trading import paper_buy_stock

THREAD_ID = "00000000-0000-0000-0000-000000000075"


class StreamingFakeModel(GenericFakeChatModel):
    """Fake streaming model that accepts bound tools."""

    def bind_tools(self, tools: Any, **kwargs: Any) -> "StreamingFakeModel":
        return self


class ToolCallingFakeModel(FakeMessagesListChatModel):
    """Fake model that returns structured tool calls without external APIs."""

    def bind_tools(self, tools: Any, **kwargs: Any) -> "ToolCallingFakeModel":
        return self


def test_normal_response_is_incremental_and_not_duplicated() -> None:
    model = StreamingFakeModel(messages=iter(["Streaming works."]))
    graph = build_chat_graph(model, [calculator], checkpointer=InMemorySaver())

    events = list(
        stream_agent_events(graph, [HumanMessage("Hello")], THREAD_ID)
    )
    chunks = [event.content for event in events if event.type == "assistant_chunk"]

    assert len(chunks) > 1
    assert "".join(chunk for chunk in chunks if chunk) == "Streaming works."
    assert [event.type for event in events].count("complete") == 1
    assert events[-1].type == "complete"


def test_calculator_runs_then_final_answer_streams() -> None:
    tool_request = AIMessage(
        "",
        tool_calls=[
            {
                "name": "calculator",
                "args": {"expression": "6 * 7"},
                "id": "calculation-1",
                "type": "tool_call",
            }
        ],
    )
    model = ToolCallingFakeModel(
        responses=[tool_request, AIMessage("The answer is 42.")]
    )
    graph = build_chat_graph(model, [calculator], checkpointer=InMemorySaver())

    events = list(
        stream_agent_events(
            graph,
            [HumanMessage("Calculate six times seven")],
            THREAD_ID,
        )
    )

    assert [event.type for event in events] == [
        "tool_started",
        "tool_finished",
        "assistant_chunk",
        "complete",
    ]
    assert events[0].tool == "calculator"
    assert events[1].tool == "calculator"
    assert events[2].content == "The answer is 42."
    assert "expression" not in "".join(str(event.as_dict()) for event in events)


def test_same_thread_restores_history_for_later_stream() -> None:
    model = StreamingFakeModel(messages=iter(["First answer.", "Second answer."]))
    graph = build_chat_graph(model, [calculator], checkpointer=InMemorySaver())

    list(stream_agent_events(graph, [HumanMessage("First")], THREAD_ID))
    list(stream_agent_events(graph, [HumanMessage("Second")], THREAD_ID))
    snapshot = graph.get_state({"configurable": {"thread_id": THREAD_ID}})

    assert [message.text for message in snapshot.values["messages"]] == [
        "First",
        "First answer.",
        "Second",
        "Second answer.",
    ]


def test_interrupt_emits_pending_approval_without_completion() -> None:
    tool_request = AIMessage(
        "",
        tool_calls=[
            {
                "name": "paper_buy_stock",
                "args": {"ticker": "AAPL", "quantity": 5},
                "id": "paper-order-1",
                "type": "tool_call",
            }
        ],
    )
    model = ToolCallingFakeModel(responses=[tool_request])
    graph = build_chat_graph(
        model,
        [paper_buy_stock],
        checkpointer=InMemorySaver(),
    )

    events = list(
        stream_agent_events(
            graph,
            [HumanMessage("Paper buy 5 AAPL shares")],
            THREAD_ID,
        )
    )

    assert [event.type for event in events] == [
        "tool_started",
        "pending_approval",
    ]
    assert events[-1].approval is not None
    assert events[-1].approval["ticker"] == "AAPL"
    assert not any(event.type == "tool_finished" for event in events)
    assert not any(event.type == "complete" for event in events)
