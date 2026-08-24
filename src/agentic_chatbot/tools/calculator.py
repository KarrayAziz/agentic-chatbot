"""A deliberately small and safe arithmetic evaluator."""

import ast
import math
import operator
from collections.abc import Callable

from langchain_core.tools import tool

Number = int | float
BinaryOperation = Callable[[Number, Number], Number]
UnaryOperation = Callable[[Number], Number]

MAX_EXPRESSION_LENGTH = 200
MAX_POWER = 100
MAX_INTEGER_BITS = 4096

BINARY_OPERATIONS: dict[type[ast.operator], BinaryOperation] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
UNARY_OPERATIONS: dict[type[ast.unaryop], UnaryOperation] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def _validate_result(value: Number) -> Number:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("Calculation result must be a real number.")
    if isinstance(value, int) and value.bit_length() > MAX_INTEGER_BITS:
        raise ValueError("Calculation result is too large.")
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Calculation result must be finite.")
    return value


def _evaluate(node: ast.AST) -> Number:
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise ValueError("Only integer and decimal numbers are allowed.")
        return _validate_result(node.value)

    if isinstance(node, ast.UnaryOp) and type(node.op) in UNARY_OPERATIONS:
        return _validate_result(UNARY_OPERATIONS[type(node.op)](_evaluate(node.operand)))

    if isinstance(node, ast.BinOp) and type(node.op) in BINARY_OPERATIONS:
        left = _evaluate(node.left)
        right = _evaluate(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > MAX_POWER:
            raise ValueError(f"Exponents must be between -{MAX_POWER} and {MAX_POWER}.")
        try:
            return _validate_result(BINARY_OPERATIONS[type(node.op)](left, right))
        except ZeroDivisionError as error:
            raise ValueError("Division by zero is not allowed.") from error
        except OverflowError as error:
            raise ValueError("Calculation result is too large.") from error

    raise ValueError("Expression contains an unsupported operation.")


@tool
def calculator(expression: str) -> dict[str, str | Number]:
    """Calculate normal arithmetic using numbers, parentheses, +, -, *, /, //, %, or **.

    Use this tool whenever an exact arithmetic result is needed. The expression
    must contain arithmetic only; names, functions, and Python code are rejected.
    """

    expression = expression.strip()
    if not expression:
        raise ValueError("Expression cannot be empty.")
    if len(expression) > MAX_EXPRESSION_LENGTH:
        raise ValueError("Expression is too long.")

    try:
        parsed = ast.parse(expression, mode="eval")
    except SyntaxError as error:
        raise ValueError("Expression is not valid arithmetic.") from error

    return {"expression": expression, "result": _evaluate(parsed.body)}
