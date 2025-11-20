#!/usr/bin/env python3
"""Translate an extracted Egglog expression back into an MLIR snippet.

This utility runs `egglog-experimental` on the given `.egg` file, parses the
resulting expression (supporting `LayoutBP (ArrayBP<N> ...)`,
`LayoutBS (ArrayBS<N> ...)`, or `LayoutCPU (Array<N> ...)`), and emits a textual
MLIR function that realises the same computation.  The mapping uses custom
dialect operations such as `pimbp.addi` / `pimbs.addi` to mirror the BP/BS
operators in the Egglog model, while CPU-only layouts lower to `arith.*`
operations directly.

The conversion deliberately keeps the implementation simple: it performs a
tree-to-DAG lowering with value memoisation, assumes integer element types, and
reuses constants when possible.  If more precise typing is required, extend the
`TYPE_TO_MLIR` and operator cases below.  The generated MLIR currently emits a
function with no explicit arguments; constants that Egglog used to stand in for
arguments remain materialised as literal values in the output.  If you need to
reintroduce SSA arguments, post-process the emitted IR with your preferred
rewriting pipeline.
"""

from __future__ import annotations

import argparse
import dataclasses
import itertools
import pathlib
import re
import subprocess
import sys
from typing import Dict, Iterable, Iterator, List, Optional, Tuple, Union

Token = str

TOKEN_RE = re.compile(r"\(|\)|[^\s()]+")


def tokenize(text: str) -> Iterator[Token]:
    for match in TOKEN_RE.finditer(text):
        yield match.group(0)


@dataclasses.dataclass(frozen=True)
class Atom:
    value: str


@dataclasses.dataclass(frozen=True)
class Node:
    op: str
    args: Tuple["Expr", ...]


Expr = Union[Atom, Node]


class ParseError(RuntimeError):
    pass


class Parser:
    def __init__(self, tokens: Iterable[Token]):
        self._tokens = list(tokens)
        self._index = 0

    def _peek(self) -> Optional[Token]:
        if self._index >= len(self._tokens):
            return None
        return self._tokens[self._index]

    def _next(self) -> Token:
        token = self._peek()
        if token is None:
            raise ParseError("unexpected end of input")
        self._index += 1
        return token

    def parse_expr(self) -> Expr:
        token = self._peek()
        if token is None:
            raise ParseError("unexpected end while parsing expression")
        if token == "(":
            self._next()  # consume "("
            head = self._next()
            args: List[Expr] = []
            while True:
                nxt = self._peek()
                if nxt is None:
                    raise ParseError(f"unterminated list for {head!r}")
                if nxt == ")":
                    self._next()  # consume ")"
                    break
                args.append(self.parse_expr())
            return Node(head, tuple(args))
        if token == ")":
            raise ParseError("unexpected ')'")
        return Atom(self._next())


def node_key(expr: Expr) -> Tuple:
    if isinstance(expr, Atom):
        return ("atom", expr.value)
    return (expr.op, tuple(node_key(arg) for arg in expr.args))


TYPE_TO_MLIR: Dict[str, str] = {
    "cpu_i32": "i32",
    "cpu_i64": "i64",
    "bp_i32": "i32",
    "bp_i64": "i64",
    "bs_i32": "i32",
    "bs_i64": "i64",
}

MLIR_TO_CPU_TYPE: Dict[str, str] = {
    "i32": "cpu_i32",
    "i64": "cpu_i64",
}


def bp_to_cpu_type(bp_type: str) -> str:
    if bp_type == "bp_i32":
        return "cpu_i32"
    if bp_type == "bp_i64":
        return "cpu_i64"
    raise ValueError(f"unhandled BP type {bp_type}")


def cpu_to_bp_type(cpu_type: str) -> str:
    if cpu_type == "cpu_i32":
        return "bp_i32"
    if cpu_type == "cpu_i64":
        return "bp_i64"
    raise ValueError(f"unhandled CPU type {cpu_type}")


def cpu_to_bs_type(cpu_type: str) -> str:
    if cpu_type == "cpu_i32":
        return "bs_i32"
    if cpu_type == "cpu_i64":
        return "bs_i64"
    raise ValueError(f"unhandled CPU type {cpu_type}")


def bp_to_bs_type(bp_type: str) -> str:
    if bp_type == "bp_i32":
        return "bs_i32"
    if bp_type == "bp_i64":
        return "bs_i64"
    raise ValueError(f"unhandled BP type {bp_type}")


def bs_to_bp_type(bs_type: str) -> str:
    if bs_type == "bs_i32":
        return "bp_i32"
    if bs_type == "bs_i64":
        return "bp_i64"
    raise ValueError(f"unhandled BS type {bs_type}")


def bs_to_cpu_type(bs_type: str) -> str:
    if bs_type == "bs_i32":
        return "cpu_i32"
    if bs_type == "bs_i64":
        return "cpu_i64"
    raise ValueError(f"unhandled BS type {bs_type}")


class MLIRBuilder:
    def __init__(self, arg_specs: Optional[List[Tuple[str, str]]] = None) -> None:
        self.lines: List[str] = []
        self._value_counter = itertools.count()
        self._memo: Dict[Tuple[Tuple, Optional[str]], Tuple[str, str]] = {}
        self._cpu_constants: Dict[Tuple[int, str], str] = {}
        self._bp_constants: Dict[Tuple[int, str], str] = {}
        self._bs_constants: Dict[Tuple[int, str], str] = {}
        self._arg_specs: List[Tuple[str, str]] = arg_specs or []
        self._arg_order: List[int] = []
        self._arg_names: Dict[int, str] = {}
        self._arg_mlir_types: Dict[int, str] = {}
        self._arg_type_tags: Dict[int, str] = {}
        self._arg_values: Dict[int, str] = {}

    def new_value(self, prefix: str) -> str:
        return f"%{prefix}{next(self._value_counter)}"

    def type_to_mlir(self, type_tag: str) -> str:
        try:
            return TYPE_TO_MLIR[type_tag]
        except KeyError as exc:
            raise ValueError(f"unknown type tag {type_tag}") from exc

    def get_constant(self, domain: str, value: int) -> Tuple[str, str]:
        if domain.startswith("cpu_"):
            cache = self._cpu_constants
            op_name = "arith.constant"
            type_tag = domain
        elif domain.startswith("bp_"):
            cache = self._bp_constants
            op_name = "pimbp.constant"
            type_tag = domain
        elif domain.startswith("bs_"):
            cache = self._bs_constants
            op_name = "pimbs.constant"
            type_tag = domain
        else:
            raise ValueError(f"unhandled constant domain {domain}")

        key = (value, type_tag)
        if key in cache:
            return cache[key], type_tag

        mlir_type = self.type_to_mlir(type_tag)
        value_name = self.new_value("c")
        self.lines.append(f"{value_name} = {op_name} {value} : {mlir_type}")
        cache[key] = value_name
        return value_name, type_tag

    def _ensure_argument(self, index: int) -> Tuple[str, str]:
        if index not in self._arg_values:
            if index < len(self._arg_specs):
                name, mlir_type = self._arg_specs[index]
            else:
                name, mlir_type = (f"arg{index}", "i32")
            name = name.lstrip("%")
            type_tag = MLIR_TO_CPU_TYPE.get(mlir_type)
            if type_tag is None:
                raise ValueError(f"unsupported MLIR type for argument: {mlir_type}")
            self._arg_names[index] = name
            self._arg_mlir_types[index] = mlir_type
            self._arg_type_tags[index] = type_tag
            self._arg_values[index] = f"%{name}"
            self._arg_order.append(index)
        return self._arg_values[index], self._arg_type_tags[index]
    def _promote_argument_type(self, index: int, new_type: str) -> None:
        if index not in self._arg_values:
            return
        current = self._arg_type_tags[index]
        if current == new_type:
            return
        if current == "cpu_i32" and new_type == "cpu_i64":
            self._arg_type_tags[index] = new_type
            self._arg_mlir_types[index] = TYPE_TO_MLIR[new_type]
        elif current == "cpu_i64" and new_type == "cpu_i32":
            return
        else:
            raise ValueError(
                f"cannot promote CpuArg {index} from {current} to {new_type}"
            )

    def list_arguments(self) -> List[Tuple[str, str]]:
        return [
            (self._arg_names[idx], self._arg_mlir_types[idx])
            for idx in sorted(self._arg_order)
        ]

    def emit(self, expr: Expr, expected: Optional[str] = None) -> Tuple[str, str]:
        memo_key = (node_key(expr), expected)
        if memo_key in self._memo:
            return self._memo[memo_key]

        if isinstance(expr, Atom):
            raise ValueError(f"unexpected atom {expr.value!r} without operator")

        op = expr.op

        if op == "Const":
            if not expr.args or not isinstance(expr.args[0], Atom):
                raise ValueError("Const expects a numeric atom argument")
            value = int(expr.args[0].value)
            domain = expected or "cpu_i32"
            result = self.get_constant(domain, value)
            self._memo[memo_key] = result
            return result
        if op == "CpuArg":
            if not expr.args or not isinstance(expr.args[0], Atom):
                raise ValueError("CpuArg expects an integer atom argument")
            index = int(expr.args[0].value)
            value, type_tag = self._ensure_argument(index)
            if expected and expected != type_tag:
                self._promote_argument_type(index, expected)
                value, type_tag = self._ensure_argument(index)
            result = (value, type_tag)
            self._memo[memo_key] = result
            return result

        handler = getattr(self, f"_emit_{op}", None)
        if handler is None:
            raise NotImplementedError(f"no translation rule for {op}")
        result = handler(expr, expected)
        self._memo[memo_key] = result
        return result

    # ------------------------------------------------------------------
    # Helper emission primitives
    # ------------------------------------------------------------------

    def _binary(
        self,
        expr: Node,
        mlir_op: str,
        result_hint: Optional[str] = None,
        lhs_hint: Optional[str] = None,
        rhs_hint: Optional[str] = None,
    ) -> Tuple[str, str]:
        lhs, lhs_type = self.emit(expr.args[0], lhs_hint)
        rhs, rhs_type = self.emit(expr.args[1], rhs_hint)
        if lhs_type != rhs_type:
            raise ValueError(
                f"{mlir_op} expects matching operand types, got {lhs_type} and {rhs_type}"
            )
        result_type = lhs_type
        if result_hint and result_hint == lhs_type:
            result_type = result_hint
        elif result_hint and result_hint != lhs_type:
            result_type = lhs_type
        res = self.new_value(mlir_op.split(".")[-1])
        mlir_ty = self.type_to_mlir(result_type)
        self.lines.append(f"{res} = {mlir_op} {lhs}, {rhs} : {mlir_ty}")
        return res, result_type

    def _unary(
        self,
        expr: Node,
        mlir_op: str,
        result_type: str,
        operand_type: str,
        *,
        needs_cast_annotation: bool = False,
    ) -> Tuple[str, str]:
        (arg,) = expr.args
        operand, _ = self.emit(arg, operand_type)
        res = self.new_value(mlir_op.split(".")[-1])
        res_ty = self.type_to_mlir(result_type)
        if needs_cast_annotation:
            src_ty = self.type_to_mlir(operand_type)
            self.lines.append(f"{res} = {mlir_op} {operand} : {src_ty} to {res_ty}")
        else:
            self.lines.append(f"{res} = {mlir_op} {operand} : {res_ty}")
        return res, result_type

    # ------------------------------------------------------------------
    # CPU ops (arith.*)
    # ------------------------------------------------------------------

    def _emit_ExtSI(self, expr: Node, expected: Optional[str]) -> Tuple[str, str]:
        result_type = "cpu_i64"
        return self._unary(expr, "arith.extsi", result_type, "cpu_i32", needs_cast_annotation=True)

    def _emit_Trunc(self, expr: Node, expected: Optional[str]) -> Tuple[str, str]:
        result_type = "cpu_i32"
        return self._unary(expr, "arith.trunci", result_type, "cpu_i64", needs_cast_annotation=True)

    def _emit_Mul(self, expr: Node, expected: Optional[str]) -> Tuple[str, str]:
        return self._binary(expr, "arith.muli", "cpu_i64", "cpu_i64", "cpu_i64")

    def _emit_Rem(self, expr: Node, expected: Optional[str]) -> Tuple[str, str]:
        return self._binary(expr, "arith.remsi", "cpu_i64", "cpu_i64", "cpu_i64")

    def _emit_Add(self, expr: Node, expected: Optional[str]) -> Tuple[str, str]:
        return self._binary(expr, "arith.addi", None, None, None)

    def _emit_Sub(self, expr: Node, expected: Optional[str]) -> Tuple[str, str]:
        return self._binary(expr, "arith.subi", None, None, None)

    def _emit_ShrU(self, expr: Node, expected: Optional[str]) -> Tuple[str, str]:
        return self._binary(expr, "arith.shrui", None, None, None)

    def _emit_ShrS(self, expr: Node, expected: Optional[str]) -> Tuple[str, str]:
        return self._binary(expr, "arith.shrsi", None, None, None)

    def _emit_And(self, expr: Node, expected: Optional[str]) -> Tuple[str, str]:
        return self._binary(expr, "arith.andi", None, None, None)

    def _emit_Xor(self, expr: Node, expected: Optional[str]) -> Tuple[str, str]:
        return self._binary(expr, "arith.xori", None, None, None)

    def _emit_Or(self, expr: Node, expected: Optional[str]) -> Tuple[str, str]:
        return self._binary(expr, "arith.ori", None, None, None)

    # ------------------------------------------------------------------
    # BP ops (pimbp.*)
    # ------------------------------------------------------------------

    def _emit_ExtSIBP(self, expr: Node, expected: Optional[str]) -> Tuple[str, str]:
        result_type = "bp_i64"
        return self._unary(expr, "pimbp.extsi", result_type, "bp_i32", needs_cast_annotation=True)

    def _emit_TruncBP(self, expr: Node, expected: Optional[str]) -> Tuple[str, str]:
        result_type = "bp_i32"
        return self._unary(expr, "pimbp.trunci", result_type, "bp_i64", needs_cast_annotation=True)

    def _emit_MulBP(self, expr: Node, expected: Optional[str]) -> Tuple[str, str]:
        return self._binary(expr, "pimbp.muli", "bp_i64", "bp_i64", "bp_i64")

    def _emit_RemBP(self, expr: Node, expected: Optional[str]) -> Tuple[str, str]:
        return self._binary(expr, "pimbp.remsi", "bp_i64", "bp_i64", "bp_i64")

    def _emit_AddBP(self, expr: Node, expected: Optional[str]) -> Tuple[str, str]:
        return self._binary(expr, "pimbp.addi", None, None, None)

    def _emit_SubBP(self, expr: Node, expected: Optional[str]) -> Tuple[str, str]:
        return self._binary(expr, "pimbp.subi", None, None, None)

    def _emit_ShrUBP(self, expr: Node, expected: Optional[str]) -> Tuple[str, str]:
        return self._binary(expr, "pimbp.shrui", None, None, None)

    def _emit_ShrSBP(self, expr: Node, expected: Optional[str]) -> Tuple[str, str]:
        return self._binary(expr, "pimbp.shrsi", None, None, None)

    def _emit_AndBP(self, expr: Node, expected: Optional[str]) -> Tuple[str, str]:
        return self._binary(expr, "pimbp.andi", None, None, None)

    def _emit_XorBP(self, expr: Node, expected: Optional[str]) -> Tuple[str, str]:
        return self._binary(expr, "pimbp.xori", None, None, None)

    def _emit_OrBP(self, expr: Node, expected: Optional[str]) -> Tuple[str, str]:
        return self._binary(expr, "pimbp.ori", None, None, None)

    # ------------------------------------------------------------------
    # BS ops (pimbs.*)
    # ------------------------------------------------------------------

    def _emit_ExtSIBS(self, expr: Node, expected: Optional[str]) -> Tuple[str, str]:
        result_type = "bs_i64"
        return self._unary(expr, "pimbs.extsi", result_type, "bs_i32", needs_cast_annotation=True)

    def _emit_TruncBS(self, expr: Node, expected: Optional[str]) -> Tuple[str, str]:
        result_type = "bs_i32"
        return self._unary(expr, "pimbs.trunci", result_type, "bs_i64", needs_cast_annotation=True)

    def _emit_MulBS(self, expr: Node, expected: Optional[str]) -> Tuple[str, str]:
        return self._binary(expr, "pimbs.muli", "bs_i64", "bs_i64", "bs_i64")

    def _emit_RemBS(self, expr: Node, expected: Optional[str]) -> Tuple[str, str]:
        return self._binary(expr, "pimbs.remsi", "bs_i64", "bs_i64", "bs_i64")

    def _emit_AddBS(self, expr: Node, expected: Optional[str]) -> Tuple[str, str]:
        return self._binary(expr, "pimbs.addi", None, None, None)

    def _emit_SubBS(self, expr: Node, expected: Optional[str]) -> Tuple[str, str]:
        return self._binary(expr, "pimbs.subi", None, None, None)

    def _emit_ShrUBS(self, expr: Node, expected: Optional[str]) -> Tuple[str, str]:
        return self._binary(expr, "pimbs.shrui", None, None, None)

    def _emit_ShrSBS(self, expr: Node, expected: Optional[str]) -> Tuple[str, str]:
        return self._binary(expr, "pimbs.shrsi", None, None, None)

    def _emit_AndBS(self, expr: Node, expected: Optional[str]) -> Tuple[str, str]:
        return self._binary(expr, "pimbs.andi", None, None, None)

    def _emit_XorBS(self, expr: Node, expected: Optional[str]) -> Tuple[str, str]:
        return self._binary(expr, "pimbs.xori", None, None, None)

    def _emit_OrBS(self, expr: Node, expected: Optional[str]) -> Tuple[str, str]:
        return self._binary(expr, "pimbs.ori", None, None, None)

    # ------------------------------------------------------------------
    # Casts and layout helpers
    # ------------------------------------------------------------------

    def _emit_LoadCPUToBP(self, expr: Node, expected: Optional[str]) -> Tuple[str, str]:
        (arg,) = expr.args
        operand, operand_type = self.emit(arg)
        if not operand_type.startswith("cpu_"):
            raise ValueError("LoadCPUToBP expects a CPU operand")
        result_type = cpu_to_bp_type(operand_type)
        res = self.new_value("bpfromcpu")
        mlir_ty = self.type_to_mlir(operand_type)
        self.lines.append(f"{res} = pimbp.from_cpu {operand} : {mlir_ty}")
        return res, result_type

    def _emit_StoreBPToCPU(self, expr: Node, expected: Optional[str]) -> Tuple[str, str]:
        (arg,) = expr.args
        operand, operand_type = self.emit(arg)
        if not operand_type.startswith("bp_"):
            raise ValueError("StoreBPToCPU expects a BP operand")
        result_type = bp_to_cpu_type(operand_type)
        res = self.new_value("bptocpu")
        mlir_ty = self.type_to_mlir(result_type)
        self.lines.append(f"{res} = pimbp.to_cpu {operand} : {mlir_ty}")
        return res, result_type

    def _emit_LoadCPUToBS(self, expr: Node, expected: Optional[str]) -> Tuple[str, str]:
        (arg,) = expr.args
        operand, operand_type = self.emit(arg)
        if not operand_type.startswith("cpu_"):
            raise ValueError("LoadCPUToBS expects a CPU operand")
        result_type = cpu_to_bs_type(operand_type)
        res = self.new_value("bsfromcpu")
        mlir_ty = self.type_to_mlir(operand_type)
        self.lines.append(f"{res} = pimbs.from_cpu {operand} : {mlir_ty}")
        return res, result_type

    def _emit_LoadBP(self, expr: Node, expected: Optional[str]) -> Tuple[str, str]:
        (arg,) = expr.args
        operand, operand_type = self.emit(arg)
        if not operand_type.startswith("cpu_"):
            raise ValueError("LoadBP expects a CPU operand")
        result_type = cpu_to_bp_type(operand_type)
        res = self.new_value("bpload")
        mlir_ty = self.type_to_mlir(operand_type)
        self.lines.append(f"{res} = pimbp.from_cpu {operand} : {mlir_ty}")
        return res, result_type

    def _emit_LoadConstantBP(self, expr: Node, expected: Optional[str]) -> Tuple[str, str]:
        (arg,) = expr.args
        desired = expected or "bp_i32"
        return self.emit(arg, desired)

    def _emit_LoadConstantBS(self, expr: Node, expected: Optional[str]) -> Tuple[str, str]:
        (arg,) = expr.args
        desired = expected or "bs_i32"
        return self.emit(arg, desired)

    def _emit_LoadBS(self, expr: Node, expected: Optional[str]) -> Tuple[str, str]:
        (arg,) = expr.args
        operand, operand_type = self.emit(arg)
        if not operand_type.startswith("cpu_"):
            raise ValueError("LoadBS expects a CPU operand")
        result_type = cpu_to_bs_type(operand_type)
        res = self.new_value("bsfromcpu")
        mlir_ty = self.type_to_mlir(operand_type)
        self.lines.append(f"{res} = pimbs.from_cpu {operand} : {mlir_ty}")
        return res, result_type

    def _emit_StoreBSToCPU(self, expr: Node, expected: Optional[str]) -> Tuple[str, str]:
        (arg,) = expr.args
        operand, operand_type = self.emit(arg)
        if not operand_type.startswith("bs_"):
            raise ValueError("StoreBSToCPU expects a BS operand")
        result_type = bs_to_cpu_type(operand_type)
        res = self.new_value("bstocpu")
        mlir_ty = self.type_to_mlir(result_type)
        self.lines.append(f"{res} = pimbs.to_cpu {operand} : {mlir_ty}")
        return res, result_type

    def _emit_CastBPtoBS(self, expr: Node, expected: Optional[str]) -> Tuple[str, str]:
        (arg,) = expr.args
        operand, operand_type = self.emit(arg)
        if not operand_type.startswith("bp_"):
            raise ValueError("CastBPtoBS expects a BP operand")
        result_type = bp_to_bs_type(operand_type)
        res = self.new_value("bp2bs")
        mlir_ty = self.type_to_mlir(result_type)
        self.lines.append(f"{res} = pimbs.cast_from_bp {operand} : {mlir_ty}")
        return res, result_type

    def _emit_CastBStoBP(self, expr: Node, expected: Optional[str]) -> Tuple[str, str]:
        (arg,) = expr.args
        operand, operand_type = self.emit(arg)
        if not operand_type.startswith("bs_"):
            raise ValueError("CastBStoBP expects a BS operand")
        result_type = bs_to_bp_type(operand_type)
        res = self.new_value("bs2bp")
        mlir_ty = self.type_to_mlir(result_type)
        self.lines.append(f"{res} = pimbp.cast_from_bs {operand} : {mlir_ty}")
        return res, result_type

    # ------------------------------------------------------------------
    # Layout-only combinators
    # ------------------------------------------------------------------

    def _emit_ArrayBP2(self, expr: Node, expected: Optional[str]) -> Tuple[str, str]:
        raise ValueError("ArrayBP* should be handled at the top-level")

    def _emit_ArrayBP4(self, expr: Node, expected: Optional[str]) -> Tuple[str, str]:
        raise ValueError("ArrayBP* should be handled at the top-level")

    def _emit_ArrayBS2(self, expr: Node, expected: Optional[str]) -> Tuple[str, str]:
        raise ValueError("ArrayBS* should be handled at the top-level")

    def _emit_ArrayBS4(self, expr: Node, expected: Optional[str]) -> Tuple[str, str]:
        raise ValueError("ArrayBS* should be handled at the top-level")

    def _emit_LayoutBP(self, expr: Node, expected: Optional[str]) -> Tuple[str, str]:
        raise ValueError("LayoutBP should be handled at the top-level")

    def _emit_LayoutBS(self, expr: Node, expected: Optional[str]) -> Tuple[str, str]:
        raise ValueError("LayoutBS should be handled at the top-level")


def run_egglog(path: str) -> str:
    try:
        result = subprocess.run(
            ["egglog-experimental", path],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        sys.stderr.write(exc.stdout)
        sys.stderr.write(exc.stderr)
        raise
    return result.stdout.strip()


def build_mlir(expr: Expr, arg_specs: Optional[List[Tuple[str, str]]] = None) -> str:
    builder = MLIRBuilder(arg_specs)
    if not isinstance(expr, Node) or expr.op not in {"LayoutBP", "LayoutBS", "LayoutCPU"}:
        raise ValueError("expected top-level LayoutBP/LayoutBS/LayoutCPU expression")

    if len(expr.args) != 1 or not isinstance(expr.args[0], Node):
        raise ValueError(f"{expr.op} expects a single array expression")

    layout_kind = expr.op  # remember for later
    array_node = expr.args[0]

    if array_node.op.startswith("ArrayBP"):
        try:
            length = int(array_node.op[len("ArrayBP") :])
        except ValueError as exc:
            raise ValueError(f"unrecognised ArrayBP arity: {array_node.op}") from exc
        if length != len(array_node.args):
            raise ValueError(
                f"{array_node.op} expected {length} elements, found {len(array_node.args)}"
            )
        return_names: List[str] = []
        result_type_tags: List[str] = []
        for elem in array_node.args:
            value, value_type = builder.emit(elem)
            name = builder.new_value("ret")
            cpu_type_tag = bp_to_cpu_type(value_type)
            mlir_ty = builder.type_to_mlir(cpu_type_tag)
            builder.lines.append(f"{name} = pimbp.to_cpu {value} : {mlir_ty}")
            return_names.append(name)
            result_type_tags.append(builder.type_to_mlir(cpu_type_tag))
    elif array_node.op.startswith("ArrayBS"):
        try:
            length = int(array_node.op[len("ArrayBS") :])
        except ValueError as exc:
            raise ValueError(f"unrecognised ArrayBS arity: {array_node.op}") from exc
        if length != len(array_node.args):
            raise ValueError(
                f"{array_node.op} expected {length} elements, found {len(array_node.args)}"
            )
        return_names = []
        result_type_tags = []
        for elem in array_node.args:
            value, value_type = builder.emit(elem)
            name = builder.new_value("ret")
            cpu_type_tag = bs_to_cpu_type(value_type)
            mlir_ty = builder.type_to_mlir(cpu_type_tag)
            builder.lines.append(f"{name} = pimbs.to_cpu {value} : {mlir_ty}")
            return_names.append(name)
            result_type_tags.append(builder.type_to_mlir(cpu_type_tag))
    elif array_node.op.startswith("Array"):
        try:
            length = int(array_node.op[len("Array") :])
        except ValueError as exc:
            raise ValueError(f"unrecognised Array arity: {array_node.op}") from exc
        if length != len(array_node.args):
            raise ValueError(
                f"{array_node.op} expected {length} elements, found {len(array_node.args)}"
            )
        return_names = []
        result_type_tags = []
        for elem in array_node.args:
            value, value_type = builder.emit(elem)
            return_names.append(value)
            result_type_tags.append(builder.type_to_mlir(value_type))
    else:
        raise ValueError(f"unsupported array constructor {array_node.op}")

    return_tuple = ", ".join(return_names)
    result_types_list = result_type_tags
    mlir_return_types = ", ".join(result_types_list)
    arg_entries = builder.list_arguments()
    arg_sig = ", ".join(f"%{name}: {typ}" for name, typ in arg_entries)
    return_line = f"return {return_tuple} : {mlir_return_types}"
    body_lines = "\n    ".join(builder.lines + [return_line])
    return_sig = f"({mlir_return_types})"
    if arg_sig:
        func_header = f"  func.func @pim_butterfly({arg_sig}) -> {return_sig} {{"
    else:
        func_header = f"  func.func @pim_butterfly() -> {return_sig} {{"
    mlir = (
        "module {\n"
        f"{func_header}\n"
        f"    {body_lines}\n"
        "  }\n"
        "}\n"
    )
    return mlir


def main(argv: List[str]) -> None:
    parser = argparse.ArgumentParser(description="Convert Egglog extraction to MLIR")
    parser.add_argument("egg_path", help="Path to the .egg program to run")
    parser.add_argument("-o", "--output", type=argparse.FileType("w"), default=sys.stdout)
    parser.add_argument(
        "--arg",
        action="append",
        default=[],
        help="Override argument name/type in order, e.g. --arg u:i32",
    )
    parser.add_argument(
        "--raw-expr",
        action="store_true",
        help="Treat the input file as a raw extracted expression (skip egglog execution).",
    )
    args = parser.parse_args(argv)

    if args.raw_expr:
        extract_output = pathlib.Path(args.egg_path).read_text().strip()
    else:
        extract_output = run_egglog(args.egg_path)
    expr = Parser(tokenize(extract_output)).parse_expr()
    raw_specs = args.arg if args.arg else ["u:i32", "v:i32", "tw:i32"]
    arg_specs: List[Tuple[str, str]] = []
    for spec in raw_specs:
        if not spec:
            continue
        if ":" in spec:
            name, typ = spec.split(":", 1)
        else:
            name, typ = spec, "i32"
        name = name.strip().lstrip("%")
        typ = typ.strip() or "i32"
        if not name:
            raise ValueError("argument name cannot be empty")
        arg_specs.append((name, typ))

    mlir_text = build_mlir(expr, arg_specs)
    args.output.write(mlir_text)


if __name__ == "__main__":
    main(sys.argv[1:])
