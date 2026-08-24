"""Explicit LangGraph tool-calling loop for the Gemini chatbot."""

from collections.abc import Sequence

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.tools import BaseTool
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode, tools_condition

AGENT_NODE = "agent"
TOOLS_NODE = "tools"


def build_chat_graph(
    model: BaseChatModel, tools: Sequence[BaseTool]
) -> CompiledStateGraph:
    """Build and compile the explicit model → tools → model loop.

    ``MessagesState`` uses LangGraph's message reducer. Returning only the new AI
    message therefore appends it to the existing conversation instead of replacing
    the conversation.
    """

    tool_list = list(tools)
    model_with_tools = model.bind_tools(tool_list)

    def call_gemini(state: MessagesState) -> dict[str, list[BaseMessage]]:
        response = model_with_tools.invoke(state["messages"])
        return {"messages": [response]}

    graph_builder = StateGraph(MessagesState)
    graph_builder.add_node(AGENT_NODE, call_gemini)
    graph_builder.add_node(
        TOOLS_NODE,
        ToolNode(tool_list, handle_tool_errors=True),
    )
    graph_builder.add_edge(START, AGENT_NODE)
    graph_builder.add_conditional_edges(
        AGENT_NODE,
        tools_condition,
        {TOOLS_NODE: TOOLS_NODE, END: END},
    )
    graph_builder.add_edge(TOOLS_NODE, AGENT_NODE)

    return graph_builder.compile()
