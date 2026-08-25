"""Sanitized streaming events shared by the CLI and HTTP API."""

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, Literal

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command

from agentic_chatbot.graph import AGENT_NODE, TOOLS_NODE
from agentic_chatbot.hitl import paper_approval_from_interrupts


StreamEventType = Literal[
    "assistant_chunk",
    "tool_started",
    "tool_finished",
    "pending_approval",
    "complete",
    "error",
]
PUBLIC_APPROVAL_FIELDS = (
    "kind",
    "action",
    "trade_mode",
    "ticker",
    "quantity",
    "question",
)


@dataclass(frozen=True, slots=True)
class AgentStreamEvent:
    """One safe, client-facing event produced during a graph execution."""

    type: StreamEventType
    content: str | None = None
    tool: str | None = None
    approval: dict[str, Any] | None = None
    message: str | None = None

    def as_dict(self) -> dict[str, Any]:
        """Return a compact transport payload without empty fields."""

        payload: dict[str, Any] = {"type": self.type}
        for field_name in ("content", "tool", "approval", "message"):
            value = getattr(self, field_name)
            if value is not None:
                payload[field_name] = value
        return payload


def message_text(message: BaseMessage) -> str:
    """Return text blocks while ignoring tool calls and structured blocks."""

    return str(message.text)


def stream_agent_events(
    graph: CompiledStateGraph,
    graph_input: list[BaseMessage] | dict[str, Any] | Command,
    thread_id: str,
) -> Iterator[AgentStreamEvent]:
    """Translate LangGraph v2 stream parts into a small, safe event protocol."""

    if isinstance(graph_input, list):
        invocation_input: dict[str, Any] | Command = {"messages": graph_input}
    else:
        invocation_input = graph_input

    final_state: dict[str, Any] | None = None
    approval: dict[str, Any] | None = None
    wrote_assistant_text = False

    for part in graph.stream(
        invocation_input,
        config={"configurable": {"thread_id": thread_id}},
        stream_mode=["messages", "updates", "values"],
        version="v2",
    ):
        approval = approval or paper_approval_from_interrupts(
            part.get("interrupts", ())
        )

        if part["type"] == "messages":
            message_chunk, metadata = part["data"]
            if metadata.get("langgraph_node") != AGENT_NODE:
                continue
            text = message_text(message_chunk)
            if text:
                wrote_assistant_text = True
                yield AgentStreamEvent("assistant_chunk", content=text)
            continue

        if part["type"] == "updates":
            update = part["data"]
            agent_update = update.get(AGENT_NODE)
            if agent_update:
                for message in agent_update.get("messages", []):
                    if isinstance(message, AIMessage):
                        for tool_call in message.tool_calls:
                            tool_name = tool_call.get("name")
                            if tool_name:
                                yield AgentStreamEvent(
                                    "tool_started", tool=str(tool_name)
                                )

            tools_update = update.get(TOOLS_NODE)
            if tools_update:
                for message in tools_update.get("messages", []):
                    if isinstance(message, ToolMessage) and message.name:
                        yield AgentStreamEvent(
                            "tool_finished", tool=message.name
                        )
            continue

        if part["type"] == "values":
            final_state = part["data"]

    if final_state is None:
        raise RuntimeError("Graph streaming completed without a final state.")

    # Some test or non-streaming model implementations may emit only a complete
    # message. Preserve a useful fallback without duplicating streamed content.
    if not wrote_assistant_text:
        final_answer = next(
            (
                message_text(message)
                for message in reversed(final_state.get("messages", []))
                if isinstance(message, AIMessage) and message_text(message)
            ),
            None,
        )
        if final_answer:
            yield AgentStreamEvent("assistant_chunk", content=final_answer)

    if approval is not None:
        public_approval = {
            field: approval[field]
            for field in PUBLIC_APPROVAL_FIELDS
            if field in approval
        }
        yield AgentStreamEvent("pending_approval", approval=public_approval)
    else:
        yield AgentStreamEvent("complete")
