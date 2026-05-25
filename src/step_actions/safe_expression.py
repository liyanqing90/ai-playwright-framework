from __future__ import annotations

import ast
import operator
from typing import Any, Mapping


class SafeExpressionError(ValueError):
    pass


_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

_UNARY_OPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
    ast.Not: operator.not_,
}

_COMPARE_OPS = {
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.In: lambda left, right: left in right,
    ast.NotIn: lambda left, right: left not in right,
}


def safe_eval_expression(
    expression: str, names: Mapping[str, Any] | None = None
) -> Any:
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise SafeExpressionError(f"表达式语法错误: {expression}") from exc
    evaluator = _Evaluator(names or {})
    return evaluator.evaluate(tree.body)


class _Evaluator:
    def __init__(self, names: Mapping[str, Any]):
        self.names = dict(names)

    def evaluate(self, node: ast.AST) -> Any:
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Name):
            return self._eval_name(node)
        if isinstance(node, ast.List):
            return [self.evaluate(item) for item in node.elts]
        if isinstance(node, ast.Tuple):
            return tuple(self.evaluate(item) for item in node.elts)
        if isinstance(node, ast.UnaryOp):
            return self._eval_unary(node)
        if isinstance(node, ast.BinOp):
            return self._eval_binop(node)
        if isinstance(node, ast.BoolOp):
            return self._eval_boolop(node)
        if isinstance(node, ast.Compare):
            return self._eval_compare(node)
        if isinstance(node, ast.Call):
            return self._eval_call(node)
        raise SafeExpressionError(f"不支持的表达式节点: {type(node).__name__}")

    def _eval_name(self, node: ast.Name) -> Any:
        if node.id in self.names:
            return self.names[node.id]
        constants = {"True": True, "False": False, "None": None}
        if node.id in constants:
            return constants[node.id]
        raise SafeExpressionError(f"表达式包含未允许的名称: {node.id}")

    def _eval_unary(self, node: ast.UnaryOp) -> Any:
        op = _UNARY_OPS.get(type(node.op))
        if op is None:
            raise SafeExpressionError(f"不支持的一元运算: {type(node.op).__name__}")
        return op(self.evaluate(node.operand))

    def _eval_binop(self, node: ast.BinOp) -> Any:
        op = _BIN_OPS.get(type(node.op))
        if op is None:
            raise SafeExpressionError(f"不支持的二元运算: {type(node.op).__name__}")
        return op(self.evaluate(node.left), self.evaluate(node.right))

    def _eval_boolop(self, node: ast.BoolOp) -> Any:
        if isinstance(node.op, ast.And):
            result = True
            for value in node.values:
                result = self.evaluate(value)
                if not result:
                    return result
            return result
        if isinstance(node.op, ast.Or):
            result = False
            for value in node.values:
                result = self.evaluate(value)
                if result:
                    return result
            return result
        raise SafeExpressionError(f"不支持的布尔运算: {type(node.op).__name__}")

    def _eval_compare(self, node: ast.Compare) -> bool:
        left = self.evaluate(node.left)
        for op_node, comparator in zip(node.ops, node.comparators):
            op = _COMPARE_OPS.get(type(op_node))
            if op is None:
                raise SafeExpressionError(f"不支持的比较运算: {type(op_node).__name__}")
            right = self.evaluate(comparator)
            if not op(left, right):
                return False
            left = right
        return True

    def _eval_call(self, node: ast.Call) -> Any:
        if not isinstance(node.func, ast.Name):
            raise SafeExpressionError("表达式只允许调用白名单函数")
        func = self._eval_name(node.func)
        if not callable(func):
            raise SafeExpressionError(f"表达式名称不可调用: {node.func.id}")
        args = [self.evaluate(arg) for arg in node.args]
        kwargs = {kw.arg: self.evaluate(kw.value) for kw in node.keywords if kw.arg}
        return func(*args, **kwargs)
