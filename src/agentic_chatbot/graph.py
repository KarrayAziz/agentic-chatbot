"""Explicit LangGraph definition for the Gemini chatbot."""

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.graph.state import CompiledStateGraph

GEMINI_NODE = "gemini"


def build_chat_graph(model: BaseChatModel) -> CompiledStateGraph:
    """Build and compile the one-node conversational graph.

    ``MessagesState`` uses LangGraph's message reducer. Returning only the new AI
    message therefore appends it to the existing conversation instead of replacing
    the conversation.
    """

    def call_gemini(state: MessagesState) -> dict[str, list[BaseMessage]]:
        response = model.invoke(state["messages"])
        return {"messages": [response]}

    graph_builder = StateGraph(MessagesState)
    graph_builder.add_node(GEMINI_NODE, call_gemini)
    graph_builder.add_edge(START, GEMINI_NODE)
    graph_builder.add_edge(GEMINI_NODE, END)

    return graph_builder.compile()
