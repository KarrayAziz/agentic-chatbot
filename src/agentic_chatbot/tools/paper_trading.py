"""Human-approved, simulated stock trading tool.

This module never connects to a brokerage and never moves real money.
"""

import re
from typing import Any, Literal

from langchain_core.tools import tool
from langgraph.types import interrupt

_TICKER_PATTERN = re.compile(r"^[A-Z][A-Z0-9.-]{0,9}$")
MAX_PAPER_QUANTITY = 1_000_000


def _execute_paper_trade(ticker: str, quantity: int) -> dict[str, Any]:
    """Create a deterministic result for an approved paper-only trade."""

    return {
        "status": "executed",
        "trade_mode": "paper",
        "action": "buy",
        "ticker": ticker,
        "quantity": quantity,
        "message": (
            f"PAPER TRADE executed: simulated purchase of {quantity} "
            f"share(s) of {ticker}. No real order was placed and no money was used."
        ),
    }


@tool
def paper_buy_stock(ticker: str, quantity: int) -> dict[str, Any]:
    """Simulate buying stock after human approval; use only for paper trades.

    Use this when the user asks to buy shares, make a simulated stock purchase,
    or place a paper-trading buy order. It never uses a brokerage or real money.
    """

    normalized_ticker = ticker.strip().upper()
    if not _TICKER_PATTERN.fullmatch(normalized_ticker):
        raise ValueError("Ticker must be 1-10 letters, numbers, dots, or hyphens.")
    if isinstance(quantity, bool) or not 1 <= quantity <= MAX_PAPER_QUANTITY:
        raise ValueError(
            f"Quantity must be between 1 and {MAX_PAPER_QUANTITY:,} shares."
        )

    decision: Literal["approve", "reject"] = interrupt(
        {
            "kind": "approval",
            "action": "paper_buy_stock",
            "trade_mode": "paper",
            "ticker": normalized_ticker,
            "quantity": quantity,
            "question": "Approve this simulated paper stock purchase?",
        }
    )

    if decision != "approve":
        return {
            "status": "rejected",
            "trade_mode": "paper",
            "action": "buy",
            "ticker": normalized_ticker,
            "quantity": quantity,
            "message": (
                "PAPER TRADE rejected by the user. No order was placed and "
                "no money was used."
            ),
        }

    return _execute_paper_trade(normalized_ticker, quantity)
