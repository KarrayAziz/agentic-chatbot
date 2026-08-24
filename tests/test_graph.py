"""Tests for the explicit chatbot tool loop."""

from typing import Any

from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from agentic_chatbot.graph import AGENT_NODE, TOOLS_NODE, build_chat_graph
from agentic_chatbot.tools.calculator import calculator


class ToolCallingFakeModel(FakeMessagesListChatModel):
    """Fake model that accepts tool binding and returns predefined messages."""

    def bind_tools(self, tools: Any, **kwargs: Any) -> "ToolCallingFakeModel":
        return self


def test_graph_has_explicit_agent_and_tool_loop_structure() -> None:
    model = ToolCallingFakeModel(responses=[AIMessage(content="Hello!")])
    graph = build_chat_graph(model, [calculator])
    drawable_graph = graph.get_graph()

    assert set(drawable_graph.nodes) == {
        "__start__",
        AGENT_NODE,
        TOOLS_NODE,
        "__end__",
    }
    edges = {(edge.source, edge.target) for edge in drawable_graph.edges}
    assert ("__start__", AGENT_NODE) in edges
    assert (AGENT_NODE, TOOLS_NODE) in edges
    assert (AGENT_NODE, "__end__") in edges
    assert (TOOLS_NODE, AGENT_NODE) in edges


def test_graph_routes_tool_call_through_tool_node_and_back_to_model() -> None:
    tool_request = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "calculator",
                "args": {"expression": "837 * 92"},
                "id": "calculation-1",
                "type": "tool_call",
            }
        ],
    )
    final_answer = AIMessage(content="837 × 92 is 77,004.")
    model = ToolCallingFakeModel(responses=[tool_request, final_answer])
    graph = build_chat_graph(model, [calculator])

    result = graph.invoke(
        {"messages": [HumanMessage(content="What is 837 * 92?")]}
    )

    assert len(result["messages"]) == 4
    assert result["messages"][1].tool_calls[0]["name"] == "calculator"
    assert isinstance(result["messages"][2], ToolMessage)
    assert "77004" in result["messages"][2].content
    assert isinstance(result["messages"][-1], AIMessage)
    assert result["messages"][-1].content == "837 × 92 is 77,004."


def test_graph_ends_without_tools_for_normal_conversation() -> None:
    model = ToolCallingFakeModel(
        responses=[AIMessage(content="Hello! I am doing well.")]
    )
    graph = build_chat_graph(model, [calculator])

    result = graph.invoke({"messages": [HumanMessage(content="How are you?")]})

    assert len(result["messages"]) == 2
    assert not any(isinstance(message, ToolMessage) for message in result["messages"])
