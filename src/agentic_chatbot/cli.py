"""Terminal interface for the chatbot."""

from collections.abc import Callable

from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph.state import CompiledStateGraph


def _message_text(message: BaseMessage) -> str:
    """Return a displayable text representation of a model message."""

    if isinstance(message.content, str):
        return message.content
    return "\n".join(
        str(block.get("text", block)) if isinstance(block, dict) else str(block)
        for block in message.content
    )


def run_chat_cli(
    graph: CompiledStateGraph,
    *,
    input_fn: Callable[[str], str] = input,
    output_fn: Callable[[str], None] = print,
) -> None:
    """Run a conversation, retaining messages in memory for this process only."""

    messages: list[BaseMessage] = []
    output_fn("Gemini chatbot ready. Type 'exit' or 'quit' to stop.")

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

        messages.append(HumanMessage(content=user_text))
        result = graph.invoke({"messages": messages})
        messages = result["messages"]
        output_fn(f"Assistant: {_message_text(messages[-1])}")
