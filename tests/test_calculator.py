"""Tests for the safe calculator tool."""

import pytest

from agentic_chatbot.tools.calculator import calculator


def test_calculator_handles_normal_arithmetic() -> None:
    result = calculator.invoke({"expression": "837 * 92 + (10 / 2)"})

    assert result == {"expression": "837 * 92 + (10 / 2)", "result": 77009.0}


@pytest.mark.parametrize(
    "expression",
    [
        "__import__('os').getcwd()",
        "sum([1, 2])",
        "2 ** 1000",
        "1 / 0",
    ],
)
def test_calculator_rejects_unsafe_or_excessive_expressions(expression: str) -> None:
    with pytest.raises(ValueError):
        calculator.invoke({"expression": expression})
