#!/usr/bin/env python3
"""Translate the butterfly snippet MLIR into its Egglog encoding."""
from __future__ import annotations

import argparse
import pathlib
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

# ----------------------------------------------------------------------------
# Opcode configuration
# ----------------------------------------------------------------------------

# (mlir_op, cpu_ctor, bp_ctor, bs_ctor, arity)
OP_VARIANTS: Sequence[Tuple[str, str, str, str, int]] = [
    ("load", "LoadMemRefCPU", "LoadMemRefBP", "LoadMemRefBS", 1),
    ("store", "WriteMemRefCPU", "WriteMemRefBP", "WriteMemRefBS", 2),
    ("extsi", "ExtSI", "ExtSIBP", "ExtSIBS", 1),
    ("muli", "Mul", "MulBP", "MulBS", 2),
    ("remsi", "Rem", "RemBP", "RemBS", 2),
    ("divsi", "Div", "DivBP", "DivBS", 2),
    ("trunci", "Trunc", "TruncBP", "TruncBS", 1),
    ("addi", "Add", "AddBP", "AddBS", 2),
    ("subi", "Sub", "SubBP", "SubBS", 2),
    ("shrui", "ShrU", "ShrUBP", "ShrUBS", 2),
    ("shrsi", "ShrS", "ShrSBP", "ShrSBS", 2),
    ("andi", "And", "AndBP", "AndBS", 2),
    ("xori", "Xor", "XorBP", "XorBS", 2),
    ("ori", "Or", "OrBP", "OrBS", 2),
]

OP_MAP: Dict[str, Tuple[str, str, str]] = {name: (cpu, bp, bs) for name, cpu, bp, bs, _ in OP_VARIANTS}
OP_ARITY: Dict[str, int] = {name: arity for name, _, _, _, arity in OP_VARIANTS}

ARRAY_LENGTH = 2  # NTT forward snippet returns 16 values

TOP_EXPR_CONSTRUCTORS: Sequence[Tuple[str, Sequence[str]]] = [
    (f"Array{ARRAY_LENGTH}", tuple("Expr" for _ in range(ARRAY_LENGTH)))
] + [(f"Get{i}", ("Expr",)) for i in range(ARRAY_LENGTH)] + [
    ("LoadCPUToBP", ("Expr",)),
    ("LoadCPUToBS", ("Expr",)),
    ("LoadConstantBP", ("Expr",)),
    ("LoadConstantBS", ("Expr",)),
    ("LoadMemRefCPU", ("Expr",)),
    ("LoadMemRefBP", ("Expr",)),
    ("LoadMemRefBS", ("Expr",)),
    ("WriteMemRefCPU", ("Expr",)),
    ("WriteMemRefBP", ("Expr",)),
    ("WriteMemRefBS", ("Expr",)),
    ("StoreBPToCPU", ("Expr",)),
    ("StoreBSToCPU", ("Expr",)),
]

TRAILING_EXPR_CONSTRUCTORS: Sequence[Tuple[str, Sequence[str]]] = [
    (f"ArrayBP{ARRAY_LENGTH}", tuple("Expr" for _ in range(ARRAY_LENGTH))),
    (f"ArrayBS{ARRAY_LENGTH}", tuple("Expr" for _ in range(ARRAY_LENGTH))),
    ("CastBPtoBS", ("Expr",)),
    ("CastBStoBP", ("Expr",)),
]

ARRAY_NAME = "classical_array"

ARRAY_LENGTH = 2
TOP_EXPR_CONSTRUCTORS: Sequence[Tuple[str, Sequence[str]]] = []
TRAILING_EXPR_CONSTRUCTORS: Sequence[Tuple[str, Sequence[str]]] = []
ARRAY_RESULT_ACCESSORS: Sequence[str] = ()
ARRAY_RESULT_ACCESSORS: Sequence[str] = tuple(f"Get{i}" for i in range(ARRAY_LENGTH))

SCALE_FACTOR = 100000

LOAD_COSTS: Dict[str, int] = {
    "LoadCPUToBP": 100000,
    "LoadCPUToBS": 6400000,
    "LoadConstantBP": 0,
    "LoadConstantBS": 0,
}

CPU_COSTS: Dict[str, int] = {
    "LoadMemRefCPU": 44 * SCALE_FACTOR,
    "WriteMemRefCPU": 44 * SCALE_FACTOR,
    "ExtSI": 1 * SCALE_FACTOR,
    "Trunc": 1 * SCALE_FACTOR,
    "Add": 1 * SCALE_FACTOR,
    "Sub": 1 * SCALE_FACTOR,
    "And": 1 * SCALE_FACTOR,
    "Xor": 1 * SCALE_FACTOR,
    "Or": 1 * SCALE_FACTOR,
    "ShrU": 1 * SCALE_FACTOR,
    "ShrS": 1 * SCALE_FACTOR,
    "Mul": 3 * SCALE_FACTOR,
    "Rem": 19 * SCALE_FACTOR,
    # Integer division (Div) measured at ~19 cycles per op
    "Div": 19 * SCALE_FACTOR,
}

_BP_COST_BASE: Dict[str, int] = {
    "ExtSIBP": 0,
    "MulBP": 6600000,
    "DivBP": 6600000,
    "RemBP": 0,
    "TruncBP": 0,
    "AddBP": 100000,
    "SubBP": 200000,
    "ShrUBP": 6400000,
    "AndBP": 100000,
    "XorBP": 100000,
    "OrBP": 100000,
    "ShrSBP": 6400000,
}

_BS_COST_BASE: Dict[str, int] = {
    "ExtSIBS": 0,
    "MulBS": 409600000,
    "DivBS": 409600000,
    "RemBS": 0,
    "TruncBS": 0,
    "AddBS": 6400000,
    "SubBS": 6400000,
    "ShrUBS": 0,
    "AndBS": 6400000,
    "XorBS": 6400000,
    "OrBS": 6400000,
    "ShrSBS": 0,
}

STORE_COSTS: Dict[str, int] = {
    "StoreBPToCPU": 1 * SCALE_FACTOR,
    "StoreBSToCPU": 16 * SCALE_FACTOR,
}

CAST_COSTS: Dict[str, int] = {
    "CastBPtoBS": 6600000,
    "CastBStoBP": 6600000,
}

BP_COSTS: Dict[str, int] = {}
BS_COSTS: Dict[str, int] = {}

LAYOUT_MODE = "hybrid"
ALLOWED_LAYOUTS: Set[str] = {"cpu", "bp", "bs"}
PRIMARY_LAYOUT_KIND = "cpu"
ACTIVE_LAYOUTS: Set[str] = set(ALLOWED_LAYOUTS)
TAIL_SUFFIX = ""

def _layout_ctor(kind: str) -> str:
    return {
        "cpu": "LayoutCPU",
        "bp": "LayoutBP",
        "bs": "LayoutBS",
    }[kind]

def _ensure_fn(kind: str) -> str:
    return {
        "cpu": "EnsureCPU",
        "bp": "EnsureBP",
        "bs": "EnsureBS",
    }[kind]

def _rebuild_tail_suffix() -> None:
    global TAIL_SUFFIX
    extractor = f"{ARRAY_NAME}_layout"
    TAIL_SUFFIX = f"""(run-schedule (saturate (run)))
(extract {extractor})
"""

def configure_layout_mode(mode: str) -> None:
    global LAYOUT_MODE, ALLOWED_LAYOUTS, PRIMARY_LAYOUT_KIND, ACTIVE_LAYOUTS
    normalized = mode.lower()
    if normalized == "hybrid":
        allowed = {"cpu", "bp", "bs"}
        primary = "cpu"
    elif normalized == "cpu":
        allowed = {"cpu"}
        primary = "cpu"
    elif normalized == "bp":
        allowed = {"bp"}
        primary = "bp"
    elif normalized == "bs":
        allowed = {"bs"}
        primary = "bs"
    else:
        raise ValueError(f"unsupported layout mode: {mode}")
    LAYOUT_MODE = normalized
    ALLOWED_LAYOUTS = set(allowed)
    PRIMARY_LAYOUT_KIND = primary
    ACTIVE_LAYOUTS = set(allowed)
    ACTIVE_LAYOUTS.add(primary)
    _rebuild_tail_suffix()

def _layout_active(kind: str) -> bool:
    return kind in ACTIVE_LAYOUTS


def configure_array_length(length: int) -> None:
    global ARRAY_LENGTH, TOP_EXPR_CONSTRUCTORS, TRAILING_EXPR_CONSTRUCTORS
    global ARRAY_RESULT_ACCESSORS, BP_COSTS, BS_COSTS

    ARRAY_LENGTH = length
    TOP_EXPR_CONSTRUCTORS = [
        (f"Array{ARRAY_LENGTH}", tuple("Expr" for _ in range(ARRAY_LENGTH)))
    ] + [(f"Get{i}", ("Expr",)) for i in range(ARRAY_LENGTH)] + [
        ("LoadCPUToBP", ("Expr",)),
        ("LoadCPUToBS", ("Expr",)),
        ("LoadConstantBP", ("Expr",)),
        ("LoadConstantBS", ("Expr",)),
        ("StoreBPToCPU", ("Expr",)),
        ("StoreBSToCPU", ("Expr",)),
    ]

    TRAILING_EXPR_CONSTRUCTORS = [
        (f"ArrayBP{ARRAY_LENGTH}", tuple("Expr" for _ in range(ARRAY_LENGTH))),
        (f"ArrayBS{ARRAY_LENGTH}", tuple("Expr" for _ in range(ARRAY_LENGTH))),
        ("CastBPtoBS", ("Expr",)),
        ("CastBStoBP", ("Expr",)),
    ]

    ARRAY_RESULT_ACCESSORS = tuple(f"Get{i}" for i in range(ARRAY_LENGTH))

    BP_COSTS = dict(_BP_COST_BASE)
    BP_COSTS[f"ArrayBP{ARRAY_LENGTH}"] = 0
    BS_COSTS = dict(_BS_COST_BASE)
    BS_COSTS[f"ArrayBS{ARRAY_LENGTH}"] = 0
    _rebuild_tail_suffix()


configure_layout_mode(LAYOUT_MODE)
configure_array_length(ARRAY_LENGTH)

def header_array_rewrites() -> str:
    element_symbols = " ".join(f"a{j}" for j in range(ARRAY_LENGTH))
    return "\n".join(
        f"(rewrite (Get{i} (Array{ARRAY_LENGTH} {element_symbols})) a{i})"
        for i in range(ARRAY_LENGTH)
    )


def build_header_template() -> str:
    return f"""(with-dynamic-cost
(datatype Expr
{{expr_lines}}))

(datatype Layout
(LayoutCPU Expr)
(LayoutBP Expr)
(LayoutBS Expr))

(constructor EnsureCPU (Layout) Expr)
(constructor EnsureBP (Layout) Expr)
(constructor EnsureBS (Layout) Expr)

(rewrite (EnsureCPU (LayoutCPU x)) x :subsume)
(rewrite (EnsureCPU (LayoutBP x)) (StoreBPToCPU x) :subsume)
(rewrite (EnsureCPU (LayoutBS x)) (StoreBSToCPU x) :subsume)
(rewrite (EnsureBP (LayoutBP x)) x :subsume)
(rewrite (EnsureBP (LayoutCPU x)) (LoadCPUToBP x) :subsume)
(rewrite (EnsureBP (LayoutBS x)) (CastBStoBP x) :subsume)
(rewrite (EnsureBS (LayoutBS x)) x :subsume)
(rewrite (EnsureBS (LayoutCPU x)) (LoadCPUToBS x) :subsume)
(rewrite (EnsureBS (LayoutBP x)) (CastBPtoBS x) :subsume)

{header_array_rewrites()}
"""

@dataclass
class Constant:
    name: str
    value: str


@dataclass
class Operation:
    name: str
    op: str
    operands: List[str]


@dataclass
class OperationEmission:
    name: str
    block: str
    cpu_ctor: str
    cpu_expr: str
    bp_ctor: str
    bp_expr: str
    bs_ctor: str
    bs_expr: str


@dataclass
class ClassicalArrayEmission:
    block: str
    array_bp_expr: Optional[str]
    array_bs_expr: Optional[str]
    store_bp_terms: Sequence[str]
    store_bs_terms: Sequence[str]
    cast_bp_to_bs_terms: Sequence[str]
    cast_bs_to_bp_terms: Sequence[str]


def expr_constructor_lines() -> Iterable[str]:
    yield "(Const i64)"
    yield "(CpuArg i64)"
    for _, cpu, _, _, arity in OP_VARIANTS:
        sig = " ".join(["Expr"] * arity)
        yield f"({cpu} {sig})"
    for name, args in TOP_EXPR_CONSTRUCTORS:
        yield f"({name} {' '.join(args)})"
    for _, _, bp, bs, arity in OP_VARIANTS:
        sig = " ".join(["Expr"] * arity)
        yield f"({bp} {sig})"
        yield f"({bs} {sig})"
    for name, args in TRAILING_EXPR_CONSTRUCTORS:
        yield f"({name} {' '.join(args)})"


def append_unique(seq: List[str], value: str) -> None:
    if value not in seq:
        seq.append(value)


def build_header() -> str:
    expr_lines = "\n".join(expr_constructor_lines())
    return build_header_template().format(expr_lines=expr_lines)


def sanitize_name(name: str) -> str:
    if not name:
        return name
    if name[0].isdigit():
        return f"v{name}"
    return name.replace("-", "_")


def parse_mlir(path: pathlib.Path) -> Tuple[List[str], List[Constant], List[Operation], List[str]]:
    text = path.read_text()
    lines = [line.strip() for line in text.splitlines()]

    func_match = re.search(r"func\.func\s+@[^(]+\((.*?)\)\s*->", text, re.S)
    if not func_match:
        raise ValueError("failed to locate function signature")
    args_part = func_match.group(1).replace("\n", " ")
    arg_names: List[str] = []
    if args_part:
        for part in args_part.split(","):
            name_part = part.split(":")[0].strip()
            if not name_part:
                continue
            arg_names.append(name_part.lstrip("%"))

    constants: List[Constant] = []
    operations: List[Operation] = []
    name_map: Dict[str, str] = {}

    const_re = re.compile(r"%([\w\d_-]+) = arith\.constant (-?\d+) :")
    op_re = re.compile(r"%([\w\d_]+) = arith\.([a-z]+) (.*)")
    load_re = re.compile(r"%([\w\d_]+) = affine\.load (.*)")
    store_re = re.compile(r"affine\.store %([\w\d_]+), (.*)")
    
    store_counter = 0  # Counter for generating unique names for store operations

    for line in lines:
        # Handle affine.store (doesn't start with %)
        store_match = store_re.match(line.strip())
        if store_match:
            value_name = store_match.group(1)
            rest = store_match.group(2)
            # Extract the memref being stored to (e.g., %arg0[0] -> arg0)
            memref_match = re.match(r"%([\w\d_]+)\[", rest)
            if memref_match:
                memref_name = memref_match.group(1)
                # Generate a unique name for this store operation
                store_name = f"store{store_counter}"
                store_counter += 1
                # Treat store as a binary operation: store(value, memref)
                sanitized_value = name_map.get(value_name, value_name)
                operations.append(Operation(store_name, "store", [sanitized_value, memref_name]))
            continue
        
        if not line.startswith("%"):
            continue
        const_match = const_re.match(line)
        if const_match:
            original = const_match.group(1)
            sanitized = sanitize_name(original)
            name_map[original] = sanitized
            constants.append(Constant(sanitized, const_match.group(2)))
            continue
        load_match = load_re.match(line)
        if load_match:
            original_name = load_match.group(1)
            sanitized_name = sanitize_name(original_name)
            name_map[original_name] = sanitized_name
            # Extract the memref being loaded from (e.g., %arg0[8] -> arg0)
            rest = load_match.group(2)
            memref_match = re.match(r"%([\w\d_]+)\[", rest)
            if memref_match:
                memref_name = memref_match.group(1)
                # Treat load as a unary operation on the memref
                operations.append(Operation(sanitized_name, "load", [memref_name]))
            continue
        op_match = op_re.match(line)
        if op_match:
            original_name, op, rest = op_match.group(1), op_match.group(2), op_match.group(3)
            sanitized_name = sanitize_name(original_name)
            name_map[original_name] = sanitized_name
            rest = rest.split(" : ")[0]
            operands_raw = [tok.strip()[1:] for tok in rest.split(",")]
            operands = [name_map.get(operand, operand) for operand in operands_raw]
            operations.append(Operation(sanitized_name, op, operands))

    return_match = re.search(r"return\s+(.*?)\s*:", text, re.S)
    return_values: List[str] = []
    if return_match:
        raw_returns = return_match.group(1)
        for token in raw_returns.replace("\n", " ").split(","):
            token = token.strip()
            if not token:
                continue
            original = token.lstrip("%")
            return_values.append(name_map.get(original, original))

    return arg_names, constants, operations, return_values


def operand_layout(name: str, arg_layout_map: Dict[str, str]) -> str:
    return arg_layout_map.get(name, f"{sanitize_name(name)}_layout")


def emit_constant(
    const: Constant,
    load_constant_bp_terms: List[str],
    load_constant_bs_terms: List[str],
) -> str:
    lines = [f"(let {const.name} (Const {const.value}))"]
    layout_exprs = {
        "cpu": const.name,
        "bp": f"(LoadConstantBP {const.name})",
        "bs": f"(LoadConstantBS {const.name})",
    }
    primary_ctor = _layout_ctor(PRIMARY_LAYOUT_KIND)
    lines.append(f"(let {const.name}_layout ({primary_ctor} {layout_exprs[PRIMARY_LAYOUT_KIND]}))")
    for kind in ACTIVE_LAYOUTS:
        if kind == PRIMARY_LAYOUT_KIND:
            continue
        lines.append(f"(union {const.name}_layout ({_layout_ctor(kind)} {layout_exprs[kind]}))")
    if _layout_active("bp"):
        append_unique(load_constant_bp_terms, const.name)
    if _layout_active("bs"):
        append_unique(load_constant_bs_terms, const.name)
    return "\n".join(lines)


def emit_operation(op: Operation, arg_layout_map: Dict[str, str]) -> OperationEmission:
    cpu_ctor, bp_ctor, bs_ctor = OP_MAP[op.op]
    arity = OP_ARITY[op.op]
    if len(op.operands) != arity:
        raise ValueError(f"unexpected operand count for {op.op}: {op.operands}")
    cpu_args = " ".join(f"(EnsureCPU {operand_layout(arg, arg_layout_map)})" for arg in op.operands)
    bp_args = " ".join(f"(EnsureBP {operand_layout(arg, arg_layout_map)})" for arg in op.operands)
    bs_args = " ".join(f"(EnsureBS {operand_layout(arg, arg_layout_map)})" for arg in op.operands)
    lines = [
        f"(let {op.name} ({cpu_ctor} {cpu_args}))",
        f"(let {op.name}_bp ({bp_ctor} {bp_args}))",
        f"(let {op.name}_bs ({bs_ctor} {bs_args}))",
    ]
    layout_names = {
        "cpu": op.name,
        "bp": f"{op.name}_bp",
        "bs": f"{op.name}_bs",
    }
    primary_ctor = _layout_ctor(PRIMARY_LAYOUT_KIND)
    lines.append(f"(let {op.name}_layout ({primary_ctor} {layout_names[PRIMARY_LAYOUT_KIND]}))")
    for kind in ACTIVE_LAYOUTS:
        if kind == PRIMARY_LAYOUT_KIND:
            continue
        lines.append(
            f"(union {op.name}_layout ({_layout_ctor(kind)} {layout_names[kind]}))"
        )
    block = "\n".join(lines)
    cpu_expr = f"({cpu_ctor} {cpu_args})"
    bp_expr = f"({bp_ctor} {bp_args})"
    bs_expr = f"({bs_ctor} {bs_args})"
    return OperationEmission(
        name=op.name,
        block=block,
        cpu_ctor=cpu_ctor,
        cpu_expr=cpu_expr,
        bp_ctor=bp_ctor,
        bp_expr=bp_expr,
        bs_ctor=bs_ctor,
        bs_expr=bs_expr,
    )


def emit_classical_array_block(
    components: Sequence[str],
    available_symbols: Sequence[str],
) -> ClassicalArrayEmission:
    if not components:
        raise ValueError("expected at least one return value to form classical array")
    if len(components) != ARRAY_LENGTH:
        raise ValueError(
            f"expected exactly {ARRAY_LENGTH} return values, got {len(components)}"
        )

    array_name = ARRAY_NAME
    result_accessors = [
        (f"result{idx}", accessor)
        for idx, accessor in enumerate(ARRAY_RESULT_ACCESSORS[: len(components)])
    ]

    def ensure(kind: str) -> str:
        fn = _ensure_fn(kind)
        return " ".join(f"({fn} {name}_layout)" for name in components)

    array_exprs = {
        "cpu": f"(Array{ARRAY_LENGTH} {ensure('cpu')})",
        "bp": f"(ArrayBP{ARRAY_LENGTH} {ensure('bp')})",
        "bs": f"(ArrayBS{ARRAY_LENGTH} {ensure('bs')})",
    }

    lines = [
        f"(let {array_name} {array_exprs['cpu']})",
        f"(let {array_name}_bp {array_exprs['bp']})",
        f"(let {array_name}_bs {array_exprs['bs']})",
    ]
    layout_names = {
        "cpu": array_name,
        "bp": f"{array_name}_bp",
        "bs": f"{array_name}_bs",
    }
    lines.append(
        f"(let {array_name}_layout ({_layout_ctor(PRIMARY_LAYOUT_KIND)} {layout_names[PRIMARY_LAYOUT_KIND]}))"
    )
    for kind in ACTIVE_LAYOUTS:
        if kind == PRIMARY_LAYOUT_KIND:
            continue
        lines.append(
            f"(union {array_name}_layout ({_layout_ctor(kind)} {layout_names[kind]}))"
        )
    for result_name, accessor in result_accessors:
        lines.append(
            f"(let {result_name} ({accessor} (EnsureCPU {array_name}_layout)))"
        )
    block = "\n".join(lines)

    array_bp_expr = array_exprs["bp"] if _layout_active("bp") else None
    array_bs_expr = array_exprs["bs"] if _layout_active("bs") else None
    available_set = set(available_symbols)
    component_bp_terms: List[str] = []
    component_bs_terms: List[str] = []
    if _layout_active("bp"):
        component_bp_terms = [
            f"{name}_bp" for name in components if name in available_set
        ]
    if _layout_active("bs"):
        component_bs_terms = [
            f"{name}_bs" for name in components if name in available_set
        ]
    cast_bp_terms = tuple(component_bp_terms) if _layout_active("bp") else tuple()
    cast_bs_terms = tuple(component_bs_terms) if _layout_active("bs") else tuple()
    store_bp_terms = (
        tuple(component_bp_terms + [f"{array_name}_bp"])
        if _layout_active("bp")
        else tuple()
    )
    store_bs_terms = (
        tuple(component_bs_terms + [f"{array_name}_bs"])
        if _layout_active("bs")
        else tuple()
    )

    return ClassicalArrayEmission(
        block=block,
        array_bp_expr=array_bp_expr,
        array_bs_expr=array_bs_expr,
        store_bp_terms=store_bp_terms,
        store_bs_terms=store_bs_terms,
        cast_bp_to_bs_terms=cast_bp_terms,
        cast_bs_to_bp_terms=cast_bs_terms,
    )


def emit_argument_block(
    arg_names: Sequence[str],
    load_cpu_to_bp_terms: List[str],
    load_cpu_to_bs_terms: List[str],
) -> str:
    lines: List[str] = []
    for index, name in enumerate(arg_names):
        lines.append(f"(let {name} (CpuArg {index}))")
        layout_exprs = {
            "cpu": name,
            "bp": f"(LoadCPUToBP {name})",
            "bs": f"(LoadCPUToBS {name})",
        }
        lines.append(
            f"(let {name}_layout ({_layout_ctor(PRIMARY_LAYOUT_KIND)} {layout_exprs[PRIMARY_LAYOUT_KIND]}))"
        )
        for kind in ACTIVE_LAYOUTS:
            if kind == PRIMARY_LAYOUT_KIND:
                continue
            lines.append(
                f"(union {name}_layout ({_layout_ctor(kind)} {layout_exprs[kind]}))"
            )
        if _layout_active("bp"):
            append_unique(load_cpu_to_bp_terms, name)
        if _layout_active("bs"):
            append_unique(load_cpu_to_bs_terms, name)
    return "\n".join(lines)


def translate_mlir_to_egg(source_mlir: pathlib.Path) -> str:
    arg_names, constants, operations, return_values = parse_mlir(source_mlir)
    arg_layout_map = {mlir: f"{mlir}_layout" for mlir in arg_names}

    load_cpu_to_bp_terms: List[str] = []
    load_cpu_to_bs_terms: List[str] = []
    load_constant_bp_terms: List[str] = []
    load_constant_bs_terms: List[str] = []

    sections: List[str] = [
        build_header(),
        "",
        emit_argument_block(arg_names, load_cpu_to_bp_terms, load_cpu_to_bs_terms),
    ]

    constant_blocks = [
        emit_constant(const, load_constant_bp_terms, load_constant_bs_terms)
        for const in constants
    ]
    if constant_blocks:
        sections.append("")
        sections.append("\n".join(constant_blocks))

    op_emissions = [emit_operation(op, arg_layout_map) for op in operations]
    if op_emissions:
        sections.append("")
        sections.append("\n".join(em.block for em in op_emissions))

    classical = emit_classical_array_block(
        return_values, [em.name for em in op_emissions]
    )
    sections.append(classical.block)

    cpu_cost_lines: List[str] = []
    for em in op_emissions:
        cost = CPU_COSTS.get(em.cpu_ctor)
        if cost is not None:
            cpu_cost_lines.append(f"(set-cost {em.cpu_expr} {cost})")

    bp_cost_lines: List[str] = []
    if _layout_active("bp"):
        bp_load_cost = LOAD_COSTS.get("LoadCPUToBP")
        if bp_load_cost is not None:
            for term in load_cpu_to_bp_terms:
                bp_cost_lines.append(f"(set-cost (LoadCPUToBP {term}) {bp_load_cost})")
        constant_bp_cost = LOAD_COSTS.get("LoadConstantBP")
        if constant_bp_cost is not None:
            for term in load_constant_bp_terms:
                bp_cost_lines.append(
                    f"(set-cost (LoadConstantBP {term}) {constant_bp_cost})"
                )
        for em in op_emissions:
            cost = BP_COSTS.get(em.bp_ctor)
            if cost is not None:
                bp_cost_lines.append(f"(set-cost {em.bp_expr} {cost})")
        bp_array_cost = BP_COSTS.get(f"ArrayBP{ARRAY_LENGTH}")
        if bp_array_cost is not None and classical.array_bp_expr is not None:
            bp_cost_lines.append(f"(set-cost {classical.array_bp_expr} {bp_array_cost})")

    bs_cost_lines: List[str] = []
    if _layout_active("bs"):
        bs_load_cost = LOAD_COSTS.get("LoadCPUToBS")
        if bs_load_cost is not None:
            for term in load_cpu_to_bs_terms:
                bs_cost_lines.append(f"(set-cost (LoadCPUToBS {term}) {bs_load_cost})")
        constant_bs_cost = LOAD_COSTS.get("LoadConstantBS")
        if constant_bs_cost is not None:
            for term in load_constant_bs_terms:
                bs_cost_lines.append(
                    f"(set-cost (LoadConstantBS {term}) {constant_bs_cost})"
                )
        for em in op_emissions:
            cost = BS_COSTS.get(em.bs_ctor)
            if cost is not None:
                bs_cost_lines.append(f"(set-cost {em.bs_expr} {cost})")
        bs_array_cost = BS_COSTS.get(f"ArrayBS{ARRAY_LENGTH}")
        if bs_array_cost is not None and classical.array_bs_expr is not None:
            bs_cost_lines.append(f"(set-cost {classical.array_bs_expr} {bs_array_cost})")

    store_bp_terms: List[str] = []
    store_bs_terms: List[str] = []
    if _layout_active("bp"):
        for em in op_emissions:
            append_unique(store_bp_terms, f"{em.name}_bp")
        for term in classical.store_bp_terms:
            append_unique(store_bp_terms, term)
    if _layout_active("bs"):
        for em in op_emissions:
            append_unique(store_bs_terms, f"{em.name}_bs")
        for term in classical.store_bs_terms:
            append_unique(store_bs_terms, term)

    cast_bp_terms: List[str] = []
    cast_bs_terms: List[str] = []
    if _layout_active("bp"):
        for term in store_bp_terms:
            append_unique(cast_bp_terms, term)
        for term in classical.cast_bp_to_bs_terms:
            append_unique(cast_bp_terms, term)
    if _layout_active("bs"):
        for term in store_bs_terms:
            append_unique(cast_bs_terms, term)
        for term in classical.cast_bs_to_bp_terms:
            append_unique(cast_bs_terms, term)

    store_cost_lines: List[str] = []
    bp_store_cost = STORE_COSTS.get("StoreBPToCPU")
    if bp_store_cost is not None and _layout_active("bp"):
        for term in store_bp_terms:
            store_cost_lines.append(f"(set-cost (StoreBPToCPU {term}) {bp_store_cost})")
    bs_store_cost = STORE_COSTS.get("StoreBSToCPU")
    if bs_store_cost is not None and _layout_active("bs"):
        for term in store_bs_terms:
            store_cost_lines.append(f"(set-cost (StoreBSToCPU {term}) {bs_store_cost})")

    cast_cost_lines: List[str] = []
    bp_cast_cost = CAST_COSTS.get("CastBPtoBS")
    if bp_cast_cost is not None and _layout_active("bp"):
        for term in cast_bp_terms:
            cast_cost_lines.append(f"(set-cost (CastBPtoBS {term}) {bp_cast_cost})")
    bs_cast_cost = CAST_COSTS.get("CastBStoBP")
    if bs_cast_cost is not None and _layout_active("bs"):
        for term in cast_bs_terms:
            cast_cost_lines.append(f"(set-cost (CastBStoBP {term}) {bs_cast_cost})")

    if cpu_cost_lines:
        sections.append("")
        sections.append("\n".join(cpu_cost_lines))

    if bp_cost_lines:
        sections.append("")
        sections.append("\n".join(bp_cost_lines))

    if bs_cost_lines:
        sections.append("")
        sections.append("\n".join(bs_cost_lines))

    if store_cost_lines:
        sections.append("")
        sections.append("\n".join(store_cost_lines))

    if cast_cost_lines:
        sections.append("")
        sections.append("\n".join(cast_cost_lines))

    sections.append("")
    sections.append(TAIL_SUFFIX.rstrip())

    return "\n".join(sections) + "\n"


def main(argv: List[str]) -> None:
    parser = argparse.ArgumentParser(description="Translate the butterfly snippet MLIR to Egglog")
    parser.add_argument("mlir_path", type=pathlib.Path, help="path to MLIR input")
    parser.add_argument(
        "--array-length",
        type=int,
        default=ARRAY_LENGTH,
        help="number of results expected (default: %(default)s)",
    )
    parser.add_argument(
        "--layout-mode",
        choices=["hybrid", "cpu", "bp", "bs"],
        default=LAYOUT_MODE,
        help="layout exposure: hybrid (default), cpu-only, bp-only, or bs-only",
    )
    parser.add_argument("-o", "--output", type=pathlib.Path)
    args = parser.parse_args(argv)

    configure_layout_mode(args.layout_mode)
    configure_array_length(args.array_length)

    egg_text = translate_mlir_to_egg(args.mlir_path.resolve())
    if args.output:
        args.output.write_text(egg_text)
    else:
        print(egg_text, end="")


if __name__ == "__main__":
    main(__import__("sys").argv[1:])
