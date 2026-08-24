"""Terminal interface for the chatbot."""

import sys
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command

from agentic_chatbot.graph import AGENT_NODE
from agentic_chatbot.hitl import paper_approval_from_interrupts


@dataclass(frozen=True, slots=True)
class StreamResult:
    """Visible output and any approval request produced by one graph run."""

    messages: list[BaseMessage]
    approval: dict[str, Any] | None
    wrote_assistant_text: bool


def _message_text(message: BaseMessage) -> str:
    """Return text blocks while ignoring tool calls and other structured blocks."""

    return str(message.text)


def _write_to_stdout(text: str) -> None:
    """Write a chunk immediately, without adding a newline."""

    sys.stdout.write(text)
    sys.stdout.flush()


def stream_assistant_response(
    graph: CompiledStateGraph,
    graph_input: list[BaseMessage] | Command,
    thread_id: str,
    write_fn: Callable[[str], None],
) -> StreamResult:
    """Stream agent text and surface a paper-trade approval interrupt.

    ``messages`` mode supplies model chunks for progressive display. ``values``
    mode supplies complete state snapshots, whose last value becomes the history
    used for the next CLI turn. Interrupts are carried on v2 stream parts.
    """

    final_state: dict[str, Any] | None = None
    approval: dict[str, Any] | None = None
    wrote_streamed_text = False

    for part in graph.stream(
        {"messages": graph_input} if isinstance(graph_input, list) else graph_input,
        config={"configurable": {"thread_id": thread_id}},
        stream_mode=["messages", "values"],
        version="v2",
    ):
        approval = approval or paper_approval_from_interrupts(part.get("interrupts", ()))
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
            if not wrote_streamed_text:
                write_fn("Assistant: ")
            write_fn(text)
            wrote_streamed_text = True

    if final_state is None:
        raise RuntimeError("Graph streaming completed without a final state.")

    final_messages = final_state["messages"]
    if not wrote_streamed_text and final_messages:
        fallback_text = _message_text(final_messages[-1])
        if fallback_text:
            write_fn("Assistant: ")
            write_fn(fallback_text)
            wrote_streamed_text = True

    return StreamResult(final_messages, approval, wrote_streamed_text)


def get_pending_approval(
    graph: CompiledStateGraph, thread_id: str
) -> dict[str, Any] | None:
    """Read a restart-safe pending approval from the latest checkpoint."""

    snapshot = graph.get_state({"configurable": {"thread_id": thread_id}})
    return paper_approval_from_interrupts(snapshot.interrupts)


def _ask_for_approval(
    approval: dict[str, Any],
    input_fn: Callable[[str], str],
    output_fn: Callable[[str], None],
) -> str | None:
    """Display the pending paper action and collect approve/reject."""

    output_fn("Approval required — PAPER TRADING ONLY")
    output_fn("Action: paper_buy_stock")
    output_fn(f"Ticker: {approval['ticker']}")
    output_fn(f"Quantity: {approval['quantity']}")

    while True:
        try:
            decision = input_fn("Approve or reject? [approve/reject]: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            output_fn("\nApproval remains pending. Goodbye!")
            return None

        if decision in {"approve", "reject"}:
            return decision
        if decision in {"exit", "quit"}:
            output_fn("Approval remains pending. Goodbye!")
            return None
        output_fn("Please enter 'approve' or 'reject'.")


def _run_until_complete(
    graph: CompiledStateGraph,
    graph_input: list[BaseMessage] | Command,
    thread_id: str,
    input_fn: Callable[[str], str],
    output_fn: Callable[[str], None],
    write_fn: Callable[[str], None],
) -> bool:
    """Stream, collect approvals, and resume until complete or user exit."""

    next_input = graph_input
    while True:
        result = stream_assistant_response(graph, next_input, thread_id, write_fn)
        if result.wrote_assistant_text:
            write_fn("\n")
        if result.approval is None:
            return True

        decision = _ask_for_approval(result.approval, input_fn, output_fn)
        if decision is None:
            return False
        next_input = Command(resume=decision)


def run_chat_cli(
    graph: CompiledStateGraph,
    thread_id: str,
    *,
    on_user_message: Callable[[str], None] | None = None,
    input_fn: Callable[[str], str] = input,
    output_fn: Callable[[str], None] = print,
    write_fn: Callable[[str], None] = _write_to_stdout,
) -> None:
    """Run a streaming conversation persisted under one LangGraph thread."""

    output_fn("Gemini chatbot ready. Type 'exit' or 'quit' to stop.")
    output_fn(f"Conversation ID: {thread_id}")

    pending_approval = get_pending_approval(graph, thread_id)
    if pending_approval is not None:
        decision = _ask_for_approval(pending_approval, input_fn, output_fn)
        if decision is None:
            return
        if not _run_until_complete(
            graph,
            Command(resume=decision),
            thread_id,
            input_fn,
            output_fn,
            write_fn,
        ):
            return

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

        if on_user_message is not None:
            on_user_message(user_text)
        if not _run_until_complete(
            graph,
            [HumanMessage(content=user_text)],
            thread_id,
            input_fn,
            output_fn,
            write_fn,
        ):
            return
