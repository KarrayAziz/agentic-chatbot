"""Shared helpers for paper-trade approval interrupts."""

from collections.abc import Iterable
from typing import Any

from langgraph.types import Interrupt


def paper_approval_from_interrupts(
    interrupts: Iterable[Interrupt],
) -> dict[str, Any] | None:
    """Return the first recognized paper-trade approval payload."""

    for pending in interrupts:
        value = pending.value
        if isinstance(value, dict) and value.get("action") == "paper_buy_stock":
            return value
    return None
