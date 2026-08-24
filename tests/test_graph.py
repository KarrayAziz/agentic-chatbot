"""Tests for the explicit chatbot graph."""

from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage, HumanMessage

from agentic_chatbot.graph import GEMINI_NODE, build_chat_graph


def test_graph_has_explicit_start_gemini_end_structure() -> None:
    graph = build_chat_graph(FakeListChatModel(responses=["Hello!"]))
    drawable_graph = graph.get_graph()

    assert set(drawable_graph.nodes) == {"__start__", GEMINI_NODE, "__end__"}
    assert {(edge.source, edge.target) for edge in drawable_graph.edges} == {
        ("__start__", GEMINI_NODE),
        (GEMINI_NODE, "__end__"),
    }


def test_gemini_node_appends_model_response_to_messages() -> None:
    model = FakeListChatModel(responses=["Hello from fake Gemini!"])
    graph = build_chat_graph(model)
    history = [
        HumanMessage(content="Remember that my name is Aziz."),
        AIMessage(content="I will remember."),
        HumanMessage(content="What is my name?"),
    ]

    result = graph.invoke({"messages": history})

    assert len(result["messages"]) == 4
    assert result["messages"][:-1] == history
    assert isinstance(result["messages"][-1], AIMessage)
    assert result["messages"][-1].content == "Hello from fake Gemini!"
