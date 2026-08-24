"""Tests for the in-memory CLI conversation loop."""

from typing import Any

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel

from agentic_chatbot.cli import run_chat_cli
from agentic_chatbot.graph import build_chat_graph
from agentic_chatbot.tools.calculator import calculator


class StreamingFakeModel(GenericFakeChatModel):
    """Fake model that streams text chunks and accepts tool binding."""

    def bind_tools(self, tools: Any, **kwargs: Any) -> "StreamingFakeModel":
        return self


def test_cli_streams_response_incrementally_without_final_duplicate() -> None:
    model = StreamingFakeModel(messages=iter(["LangGraph streams progressively."]))
    graph = build_chat_graph(model, [calculator])
    user_inputs = iter(["Hello", "quit"])
    output_lines: list[str] = []
    streamed_writes: list[str] = []

    run_chat_cli(
        graph,
        input_fn=lambda prompt: next(user_inputs),
        output_fn=output_lines.append,
        write_fn=streamed_writes.append,
    )

    rendered = "".join(streamed_writes)
    assert rendered == "Assistant: LangGraph streams progressively.\n"
    assert len(streamed_writes) > 3
    assert rendered.count("LangGraph streams progressively.") == 1
    assert output_lines[-1] == "Goodbye!"
