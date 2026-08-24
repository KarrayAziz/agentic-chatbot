"""Tests for restart-safe human approval of paper stock purchases."""

from pathlib import Path
from typing import Any
from uuid import uuid4

from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from agentic_chatbot.cli import get_pending_approval
from agentic_chatbot.graph import build_chat_graph
from agentic_chatbot.persistence import open_sqlite_checkpointer
from agentic_chatbot.tools.paper_trading import paper_buy_stock
import agentic_chatbot.tools.paper_trading as paper_trading_module


class PaperTradeFakeModel(FakeMessagesListChatModel):
    """Fake model that accepts tool binding and returns predefined messages."""

    def bind_tools(self, tools: Any, **kwargs: Any) -> "PaperTradeFakeModel":
        return self


def _config(thread_id: str) -> dict[str, dict[str, str]]:
    return {"configurable": {"thread_id": thread_id}}


def _paper_trade_request() -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[
            {
                "name": "paper_buy_stock",
                "args": {"ticker": "aapl", "quantity": 5},
                "id": "paper-order-1",
                "type": "tool_call",
            }
        ],
    )


def test_interrupted_trade_persists_without_execution(
    tmp_path: Path, monkeypatch
) -> None:
    database_path = tmp_path / "checkpoints.sqlite"
    thread_id = str(uuid4())
    executions: list[tuple[str, int]] = []
    monkeypatch.setattr(
        paper_trading_module,
        "_execute_paper_trade",
        lambda ticker, quantity: executions.append((ticker, quantity)),
    )

    with open_sqlite_checkpointer(database_path) as checkpointer:
        graph = build_chat_graph(
            PaperTradeFakeModel(responses=[_paper_trade_request()]),
            [paper_buy_stock],
            checkpointer=checkpointer,
        )
        list(
            graph.stream(
                {"messages": [HumanMessage("Paper buy 5 shares of AAPL")]},
                config=_config(thread_id),
                stream_mode=["messages", "values"],
                version="v2",
            )
        )
        approval = graph.get_state(_config(thread_id)).interrupts[0].value

    assert approval == {
        "kind": "approval",
        "action": "paper_buy_stock",
        "trade_mode": "paper",
        "ticker": "AAPL",
        "quantity": 5,
        "question": "Approve this simulated paper stock purchase?",
    }
    assert executions == []

    with open_sqlite_checkpointer(database_path) as checkpointer:
        reopened_graph = build_chat_graph(
            PaperTradeFakeModel(responses=[AIMessage("unused")]),
            [paper_buy_stock],
            checkpointer=checkpointer,
        )
        reopened_approval = get_pending_approval(reopened_graph, thread_id)

    assert reopened_approval == approval
    assert executions == []


def test_approval_executes_paper_trade_and_returns_to_agent(monkeypatch) -> None:
    executions: list[tuple[str, int]] = []

    def fake_execute(ticker: str, quantity: int) -> dict[str, Any]:
        executions.append((ticker, quantity))
        return {"status": "executed", "ticker": ticker, "quantity": quantity}

    monkeypatch.setattr(paper_trading_module, "_execute_paper_trade", fake_execute)
    model = PaperTradeFakeModel(
        responses=[_paper_trade_request(), AIMessage("The paper trade was executed.")]
    )
    graph = build_chat_graph(
        model, [paper_buy_stock], checkpointer=InMemorySaver()
    )
    config = _config(str(uuid4()))

    graph.invoke(
        {"messages": [HumanMessage("Paper buy 5 shares of AAPL")]}, config=config
    )
    assert executions == []

    result = graph.invoke(Command(resume="approve"), config=config)

    assert executions == [("AAPL", 5)]
    tool_result = next(
        message for message in result["messages"] if isinstance(message, ToolMessage)
    )
    assert "executed" in tool_result.content
    assert result["messages"][-1].content == "The paper trade was executed."


def test_rejection_skips_execution_and_returns_rejection_to_agent(monkeypatch) -> None:
    executions: list[tuple[str, int]] = []
    monkeypatch.setattr(
        paper_trading_module,
        "_execute_paper_trade",
        lambda ticker, quantity: executions.append((ticker, quantity)),
    )
    model = PaperTradeFakeModel(
        responses=[_paper_trade_request(), AIMessage("The paper trade was rejected.")]
    )
    graph = build_chat_graph(
        model, [paper_buy_stock], checkpointer=InMemorySaver()
    )
    config = _config(str(uuid4()))

    graph.invoke(
        {"messages": [HumanMessage("Paper buy 5 shares of AAPL")]}, config=config
    )
    result = graph.invoke(Command(resume="reject"), config=config)

    assert executions == []
    tool_result = next(
        message for message in result["messages"] if isinstance(message, ToolMessage)
    )
    assert "rejected" in tool_result.content
    assert result["messages"][-1].content == "The paper trade was rejected."
