"""Terminal interface for the chatbot."""

import sys
from collections.abc import Callable
from typing import Any

from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph.state import CompiledStateGraph

from agentic_chatbot.graph import AGENT_NODE


def _message_text(message: BaseMessage) -> str:
    """Return text blocks while ignoring tool calls and other structured blocks."""

    return str(message.text)


def _write_to_stdout(text: str) -> None:
    """Write a chunk immediately, without adding a newline."""

    sys.stdout.write(text)
    sys.stdout.flush()


def stream_assistant_response(
    graph: CompiledStateGraph,
    messages: list[BaseMessage],
    thread_id: str,
    write_fn: Callable[[str], None],
) -> list[BaseMessage]:
    """Stream agent text and return the graph's final message state.

    ``messages`` mode supplies model chunks for progressive display. ``values``
    mode supplies complete state snapshots, whose last value becomes the history
    used for the next CLI turn.
    """

    final_state: dict[str, Any] | None = None
    wrote_streamed_text = False

    for part in graph.stream(
        {"messages": messages},
        config={"configurable": {"thread_id": thread_id}},
        stream_mode=["messages", "values"],
        version="v2",
    ):
        if part["type"] == "values":
            final_state = part["data"]
            continue
        if part["type"] != "messages":
            continue

        message_chunk, metadata = part["data"]
        if metadata.get("langgraph_node") != AGENT_NODE:
            continue

        text = _message_text(message_chunk)
        if text:
            write_fn(text)
            wrote_streamed_text = True

    if final_state is None:
        raise RuntimeError("Graph streaming completed without a final state.")

    final_messages = final_state["messages"]
    if not wrote_streamed_text and final_messages:
        fallback_text = _message_text(final_messages[-1])
        if fallback_text:
            write_fn(fallback_text)

    return final_messages


def run_chat_cli(
    graph: CompiledStateGraph,
    thread_id: str,
    *,
    input_fn: Callable[[str], str] = input,
    output_fn: Callable[[str], None] = print,
    write_fn: Callable[[str], None] = _write_to_stdout,
) -> None:
    """Run a streaming conversation persisted under one LangGraph thread."""

    output_fn("Gemini chatbot ready. Type 'exit' or 'quit' to stop.")
    output_fn(f"Conversation ID: {thread_id}")

    while True:
        try:
            user_text = input_fn("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            output_fn("\nGoodbye!")
            return

        if user_text.lower() in {"exit", "quit"}:
            output_fn("Goodbye!")
            return
        if not user_text:
            continue

        write_fn("Assistant: ")
        stream_assistant_response(
            graph,
            [HumanMessage(content=user_text)],
            thread_id,
            write_fn,
        )
        write_fn("\n")
