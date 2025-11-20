#!/usr/bin/env python3
"""Standalone kernel cost capture pipeline.

Given a module containing `scop.stmt` functions (e.g., from stage-split output),
extract a single function, convert its terminal store into a return value, pass
the snippet through `mlir_to_egg.py -> egglog`, and record the cost breakdown as
JSON without touching the main `run_ntt_pipeline.sh` flow.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import pathlib
import re
import subprocess
import sys
import textwrap
from typing import Dict, List, Optional, Tuple, Union

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
# In the minimal_release layout, this file lives under REPO_ROOT/src/.
REPO_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import analyze_egg_cost  # type: ignore  # pylint: disable=wrong-import-position
import mlir_to_egg  # type: ignore  # pylint: disable=wrong-import-position


STORE_RE = re.compile(
    r"(?:affine|memref)\.store\s+%(?P<value>[\w\d_]+),.*:.*memref<(?P<memref>[^>]+)>"
)
FUNC_RE = re.compile(
    r"(?P<indent>\s*)func\.func\s+(?P<visibility>private\s+)?@(?P<name>[\w\d_]+)\("
    r"(?P<args>[^)]*)\)(?P<tail>.*)",
    re.S,
)


class KernelExtractionError(RuntimeError):
    pass


def _extract_function_block(module_text: str, func_name: str) -> str:
    pattern = re.compile(rf"func\.func\s+[^\{{]*@{re.escape(func_name)}\b", re.S)
    match = pattern.search(module_text)
    if not match:
        raise KernelExtractionError(f"function @{func_name} not found")

    search_idx = match.end()
    while True:
        brace_start = module_text.find("{", search_idx)
        if brace_start == -1:
            raise KernelExtractionError(f"malformed function @{func_name}: missing body")
        prefix = module_text[search_idx:brace_start]
        if "attributes" in prefix:
            brace_end = module_text.find("}", brace_start)
            if brace_end == -1:
                raise KernelExtractionError(f"unterminated attributes for @{func_name}")
            search_idx = brace_end + 1
            continue
        break

    depth = 0
    idx = brace_start
    while idx < len(module_text):
        char = module_text[idx]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return module_text[match.start() : idx + 1]
        idx += 1
    raise KernelExtractionError(f"unterminated body for @{func_name}")


def _convert_store_to_return(func_text: str) -> Tuple[str, str]:
    lines = func_text.splitlines()
    store_idx: Optional[int] = None
    store_value: Optional[str] = None
    element_type: Optional[str] = None

    # Find the LAST affine.store (not the first) to handle functions with multiple stores
    for idx, line in enumerate(lines):
        match = STORE_RE.search(line)
        if not match:
            continue
        store_idx = idx
        store_value = match.group("value")
        memref_desc = match.group("memref")
        # Element type is the suffix after the last "x".
        element_type = memref_desc.rsplit("x", 1)[-1].strip()
        # Don't break - continue to find the last store

    if store_idx is None or store_value is None or element_type is None:
        raise KernelExtractionError("expected at least one affine/memref.store to convert into return")

    # Don't delete the store - keep it for cost modeling
    # Instead, we'll modify the return to return the stored value

    return_replaced = False
    for idx in range(len(lines) - 1, -1, -1):
        stripped = lines[idx].strip()
        if not stripped.startswith("return"):
            continue
        indent = lines[idx][: len(lines[idx]) - len(lines[idx].lstrip())]
        lines[idx] = f"{indent}return %{store_value} : {element_type}"
        return_replaced = True
        break

    if not return_replaced:
        raise KernelExtractionError("function missing terminal return to replace")

    header_line = lines[0]
    header_match = FUNC_RE.match(header_line)
    if not header_match:
        raise KernelExtractionError("failed to parse function header for return insertion")

    indent = header_match.group("indent") or ""
    visibility = ""  # drop `private` to satisfy mlir_to_egg parser expectations
    name = header_match.group("name")
    args = header_match.group("args")
    tail = header_match.group("tail") or ""
    if "->" not in tail:
        updated_header = f"{indent}func.func {visibility}@{name}({args}) -> ({element_type}){tail}"
    else:
        updated_header = header_line
    lines[0] = updated_header
    return "\n".join(lines), element_type


def _wrap_module(func_text: str) -> str:
    dedented = textwrap.dedent(func_text).strip("\n")
    indented = textwrap.indent(dedented, "  ")
    return f"module {{\n{indented}\n}}\n"


def _run_egglog(egglog_binary: str, egg_program: pathlib.Path) -> str:
    proc = subprocess.run(
        [egglog_binary, str(egg_program)],
        check=True,
        capture_output=True,
        text=True,
    )
    return proc.stdout.strip()


def _write_text(path: pathlib.Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def _infer_return_count(func_text: str) -> int:
    match = re.search(r"return\s+(?P<body>[^:]+)\s*:", func_text)
    if not match:
        return 0
    body = match.group("body").strip()
    if not body:
        return 0
    return len([token.strip() for token in body.split(",") if token.strip()])


TYPE_BITS_RE = re.compile(r"i(?P<bits>\d+)$")
SEXP_TOKEN_RE = re.compile(r"\(|\)|[^\s()]+")
WRAPPER_OPS = {"LayoutCPU", "LayoutBP", "LayoutBS", "EnsureCPU", "EnsureBP", "EnsureBS"}


def _infer_bitwidth(type_str: Optional[str]) -> Optional[int]:
    if not type_str:
        return None
    match = TYPE_BITS_RE.match(type_str.strip())
    if not match:
        return None
    return int(match.group("bits"))


class LayoutUsage:
    __slots__ = ("peak_bp", "hold_bp", "peak_bs", "hold_bs", "bs_unknown")

    def __init__(self, peak_bp: int = 0, hold_bp: int = 0, peak_bs: int = 0, hold_bs: int = 0, bs_unknown: bool = False) -> None:
        self.peak_bp = peak_bp
        self.hold_bp = hold_bp
        self.peak_bs = peak_bs
        self.hold_bs = hold_bs
        self.bs_unknown = bs_unknown


def _tokenize_sexpr(text: str) -> List[str]:
    return [match.group(0) for match in SEXP_TOKEN_RE.finditer(text)]


def _parse_sexpr(text: str) -> Union[str, List[Union[str, list]]]:
    tokens = _tokenize_sexpr(text)
    if not tokens:
        return ""
    idx = 0

    def parse() -> Union[str, List[Union[str, list]]]:
        nonlocal idx
        if idx >= len(tokens):
            raise ValueError("unexpected end of tokens")
        token = tokens[idx]
        idx += 1
        if token == "(":
            if idx >= len(tokens):
                raise ValueError("missing operator after '('")
            op = tokens[idx]
            idx += 1
            node: List[Union[str, list]] = [op]
            while idx < len(tokens) and tokens[idx] != ")":
                node.append(parse())
            if idx >= len(tokens):
                raise ValueError("missing closing ')'")
            idx += 1  # consume ')'
            return node
        if token == ")":
            raise ValueError("unexpected ')'")
        return token

    expr = parse()
    if idx != len(tokens):
        raise ValueError("extra tokens after parsing S-expression")
    return expr


def _node_layout_kind(op: str) -> Optional[str]:
    if op in WRAPPER_OPS:
        return None
    if op.startswith("Store"):
        return None
    if op.startswith("ArrayBP") or op.endswith("BP"):
        return "bp"
    if op.startswith("ArrayBS") or op.endswith("BS"):
        return "bs"
    return None


def _accumulate_children(metrics: List[LayoutUsage], attr: str) -> Tuple[int, int]:
    current = 0
    peak = 0
    for child in metrics:
        child_peak = child.peak_bp if attr == "bp" else child.peak_bs
        child_hold = child.hold_bp if attr == "bp" else child.hold_bs
        peak = max(peak, current + child_peak)
        current += child_hold
        peak = max(peak, current)
    return peak, current


def _analyze_layout_usage(node: Union[str, List[Union[str, list]]], bitwidth: Optional[int]) -> LayoutUsage:
    if not isinstance(node, list):
        return LayoutUsage()
    if not node:
        return LayoutUsage()
    op = node[0]
    children = node[1:]
    if op in WRAPPER_OPS and children:
        # Treat wrapper as alias to its first child
        return _analyze_layout_usage(children[0], bitwidth)

    child_metrics = [_analyze_layout_usage(child, bitwidth) for child in children]
    usage = LayoutUsage()
    usage.bs_unknown = any(child.bs_unknown for child in child_metrics)

    bp_peak_children, bp_hold_children = _accumulate_children(child_metrics, "bp")
    bs_peak_children, bs_hold_children = _accumulate_children(child_metrics, "bs")

    layout_kind = _node_layout_kind(op)
    node_bp_hold = 1 if layout_kind == "bp" else 0
    node_bs_hold = 0
    if layout_kind == "bs":
        if bitwidth is None:
            usage.bs_unknown = True
            node_bs_hold = 0
        else:
            node_bs_hold = bitwidth

    usage.peak_bp = max(bp_peak_children, bp_hold_children, bp_hold_children + node_bp_hold, node_bp_hold)
    usage.hold_bp = node_bp_hold

    usage.peak_bs = max(bs_peak_children, bs_hold_children, bs_hold_children + node_bs_hold, node_bs_hold)
    usage.hold_bs = node_bs_hold

    return usage


def _compute_pim_storage_from_expr(expr_text: str, bitwidth: Optional[int], parallelism: int) -> Dict[str, Optional[int]]:
    expr_text = expr_text.strip()
    def _default_storage() -> Dict[str, Optional[int]]:
        bp_cols = bitwidth * parallelism if bitwidth is not None else None
        return {
            "bitwidth": bitwidth,
            "parallelism": parallelism,
            "pim": {
                "bp": {"rows": 0, "columns": bp_cols},
                "bs": {"rows": None if bitwidth is None else 0, "columns": parallelism},
            },
        }
    if not expr_text:
        return _default_storage()
    try:
        expr = _parse_sexpr(expr_text)
    except ValueError:
        return _default_storage()
    usage = _analyze_layout_usage(expr, bitwidth)
    bp_rows = usage.peak_bp
    if bitwidth is None and (usage.peak_bs > 0 or usage.bs_unknown):
        bs_rows: Optional[int] = None
    else:
        bs_rows = usage.peak_bs
    return {
        "bitwidth": bitwidth,
        "parallelism": parallelism,
        "pim": {
            "bp": {
                "rows": bp_rows,
                "columns": bitwidth * parallelism if bitwidth is not None else None,
            },
            "bs": {
                "rows": bs_rows,
                "columns": 1 * parallelism,
            },
        },
    }


def _dedup_loadmemrefcpu_for_cpu_layout(
    expr_text: str, cost_breakdown: Dict[str, object]
) -> None:
    """
    Deduplicate LoadMemRefCPU for CPU layout:
    only the first load per CpuArg pays full cost.
    """
    text = expr_text.strip()
    if not text:
        return
    try:
        expr = _parse_sexpr(text)
    except ValueError:
        return

    counts = cost_breakdown.get("counts", {})
    if not isinstance(counts, dict):
        return
    cpu_counts = counts.get("cpu", {})
    if not isinstance(cpu_counts, dict):
        return
    load_count = int(cpu_counts.get("LoadMemRefCPU", 0))
    if load_count <= 1:
        return

    # Collect CpuArg ids used in LoadMemRefCPU operations
    unique_args: Set[str] = set()

    def _walk(node: Union[str, List[Union[str, list]]]) -> None:
        if not isinstance(node, list) or not node:
            return
        op = node[0]
        if op == "LoadMemRefCPU" and len(node) >= 2:
            arg = node[1]
            if isinstance(arg, list) and arg and arg[0] == "CpuArg" and len(arg) >= 2:
                unique_args.add(repr(arg[1]))
        for child in node[1:]:
            _walk(child)

    _walk(expr)
    if not unique_args:
        return

    unique_count = len(unique_args)
    if unique_count <= 0 or unique_count >= load_count:
        return

    per_op_cost = mlir_to_egg.CPU_COSTS.get("LoadMemRefCPU", 0)
    if per_op_cost <= 0:
        return

    total_load_cost = load_count * per_op_cost
    dedup_load_cost = unique_count * per_op_cost
    delta = total_load_cost - dedup_load_cost

    totals = cost_breakdown.get("totals", {})
    if isinstance(totals, dict):
        cpu_total = int(totals.get("cpu", 0))
        new_cpu_total = cpu_total - delta
        if new_cpu_total < 0:
            new_cpu_total = 0
        totals["cpu"] = new_cpu_total
        cost_breakdown["totals"] = totals

        grand = int(cost_breakdown.get("grand_total", 0))
        grand -= delta
        if grand < 0:
            grand = 0
        cost_breakdown["grand_total"] = grand

    cpu_counts["LoadMemRefCPU"] = unique_count
    counts["cpu"] = cpu_counts
    cost_breakdown["counts"] = counts


def capture_kernel_cost(
    source_mlir: pathlib.Path,
    func_name: str,
    output_dir: pathlib.Path,
    parallelism_label: str,
    array_length: Optional[int],
    layout_mode: str,
    egglog_binary: str,
    cpu_count: int,
) -> Dict[str, object]:
    module_text = source_mlir.read_text()
    function_block = _extract_function_block(module_text, func_name)
    converted_block, element_type = _convert_store_to_return(function_block)
    snippet_text = _wrap_module(converted_block)

    prefix = f"p{parallelism_label}_"
    snippet_path = output_dir / f"{prefix}snippet.mlir"
    _write_text(snippet_path, snippet_text)
    bitwidth = _infer_bitwidth(element_type)

    inferred_returns = _infer_return_count(converted_block)
    effective_array_length = array_length or inferred_returns or 1

    mlir_to_egg.configure_layout_mode(layout_mode)
    mlir_to_egg.configure_array_length(effective_array_length)
    egg_text = mlir_to_egg.translate_mlir_to_egg(snippet_path)

    egg_path = output_dir / f"{prefix}program.egg"
    _write_text(egg_path, egg_text)

    extraction = _run_egglog(egglog_binary, egg_path)
    extracted_path = output_dir / f"{prefix}extracted.txt"
    _write_text(extracted_path, extraction + "\n")

    storage_stats = _compute_pim_storage_from_expr(extraction, bitwidth, int(parallelism_label))

    cost_breakdown = analyze_egg_cost.compute_cost_from_file(
        extracted_path, array_length=effective_array_length
    )
    
    # For CPU layout, treat only the first LoadMemRefCPU
    # per CpuArg as paying full cost; duplicate loads are free.
    if layout_mode == "cpu":
        _dedup_loadmemrefcpu_for_cpu_layout(extraction, cost_breakdown)
    
    # Also analyze the .egg file to count store operations that may have been optimized away
    # from the extracted expression
    egg_text = egg_path.read_text()
    store_counts = {}
    store_costs_from_egg = {}
    
    # Count WriteMemRef operations in the .egg file
    for line in egg_text.splitlines():
        line = line.strip()
        if line.startswith("(let store"):
            # Extract store operation: (let store0 (WriteMemRefCPU ...))
            if "WriteMemRefCPU" in line:
                store_counts["WriteMemRefCPU"] = store_counts.get("WriteMemRefCPU", 0) + 1
            elif "WriteMemRefBP" in line:
                store_counts["WriteMemRefBP"] = store_counts.get("WriteMemRefBP", 0) + 1
            elif "WriteMemRefBS" in line:
                store_counts["WriteMemRefBS"] = store_counts.get("WriteMemRefBS", 0) + 1
    
    # Calculate store costs based on layout mode
    # Separate CPU stores (WriteMemRefCPU) from PIM stores (WriteMemRefBP/BS)
    cpu_store_cost = 0
    pim_store_cost = 0
    
    for op_name, count in store_counts.items():
        cost = mlir_to_egg.CPU_COSTS.get(op_name, 0)
        if cost > 0:
            store_costs_from_egg[op_name] = cost * count
            if op_name == "WriteMemRefCPU":
                # CPU store: add to CPU work
                cpu_store_cost += cost * count
            else:
                # PIM store (WriteMemRefBP/BS): add to store work
                pim_store_cost += cost * count
    
    # Add CPU store costs to CPU totals
    if cpu_store_cost > 0:
        cost_breakdown["totals"]["cpu"] = cost_breakdown["totals"].get("cpu", 0) + cpu_store_cost
        cost_breakdown["grand_total"] = cost_breakdown.get("grand_total", 0) + cpu_store_cost
        if "counts" not in cost_breakdown:
            cost_breakdown["counts"] = {}
        if "cpu" not in cost_breakdown["counts"]:
            cost_breakdown["counts"]["cpu"] = {}
        cost_breakdown["counts"]["cpu"]["WriteMemRefCPU"] = store_counts.get("WriteMemRefCPU", 0)
    
    # Add PIM store costs to store totals
    if pim_store_cost > 0:
        cost_breakdown["totals"]["store"] = cost_breakdown["totals"].get("store", 0) + pim_store_cost
        cost_breakdown["grand_total"] = cost_breakdown.get("grand_total", 0) + pim_store_cost
        if "counts" not in cost_breakdown:
            cost_breakdown["counts"] = {}
        if "store" not in cost_breakdown["counts"]:
            cost_breakdown["counts"]["store"] = {}
        for op_name in ["WriteMemRefBP", "WriteMemRefBS"]:
            if op_name in store_counts:
                cost_breakdown["counts"]["store"][op_name] = store_counts[op_name]

    # Scale all costs by parallelism to get total cost for all instances
    parallelism = int(parallelism_label)
    totals = cost_breakdown.get("totals", {})
    counts = cost_breakdown.get("counts", {})
    
    # Scale all totals by parallelism
    cpu_total_scaled = int(totals.get("cpu", 0)) * parallelism
    bp_total_scaled = int(totals.get("bp", 0)) * parallelism
    bs_total_scaled = int(totals.get("bs", 0)) * parallelism
    load_total_scaled = int(totals.get("load", 0)) * parallelism
    store_total_scaled = int(totals.get("store", 0)) * parallelism
    cast_total_scaled = int(totals.get("cast", 0)) * parallelism
    
    # Recalculate grand total
    grand_total = (cpu_total_scaled + bp_total_scaled + bs_total_scaled + 
                   load_total_scaled + store_total_scaled + cast_total_scaled)
    
    # Update cost_breakdown with scaled values
    cost_breakdown["totals"] = {
        "cpu": cpu_total_scaled,
        "bp": bp_total_scaled,
        "bs": bs_total_scaled,
        "load": load_total_scaled,
        "store": store_total_scaled,
        "cast": cast_total_scaled,
    }
    cost_breakdown["grand_total"] = grand_total
    cost_breakdown["parallelism_scaling_applied"] = True
    cost_breakdown["per_instance_totals"] = {
        "cpu": int(totals.get("cpu", 0)),
        "bp": int(totals.get("bp", 0)),
        "bs": int(totals.get("bs", 0)),
        "load": int(totals.get("load", 0)),
        "store": int(totals.get("store", 0)),
        "cast": int(totals.get("cast", 0)),
    }

    timestamp = _dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

    result = {
        "kernel": func_name,
        "source_mlir": str(source_mlir),
        "snippet_mlir": str(snippet_path),
        "egg_program": str(egg_path),
        "extracted_expr": str(extracted_path),
        "element_type": element_type,
        "array_length": effective_array_length,
        "layout_mode": layout_mode,
        "cost_breakdown": cost_breakdown,
        "generated_at": timestamp,
        "storage": storage_stats,
        "cpu_resources": {
            "assumed_count": cpu_count,
            "total_cost": cpu_total_scaled,
        },
    }

    cost_path = output_dir / f"{prefix}cost.json"
    _write_text(cost_path, json.dumps(result, indent=2, sort_keys=True))
    return result


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capture per-kernel Egglog costs without touching existing pipelines.")
    parser.add_argument(
        "--source-mlir",
        type=pathlib.Path,
        required=True,
        help="Path to the module containing scop.stmt functions.",
    )
    parser.add_argument(
        "--function",
        required=True,
        help="Function name (e.g., S0) to extract from the module.",
    )
    parser.add_argument(
        "--array-length",
        type=int,
        default=None,
        help="Override number of return values (defaults to inferred count or 1).",
    )
    parser.add_argument(
        "--layout-mode",
        choices=["hybrid", "cpu", "bp", "bs"],
        default="hybrid",
        help="Layout exposure passed to mlir_to_egg.py",
    )
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        default=REPO_ROOT / "out" / "kernel_costs",
        help="Directory for all intermediate artifacts.",
    )
    parser.add_argument(
        "--parallelism-label",
        type=str,
        default="1",
        help="Parallelism tag (e.g., 1, 2) used in directory names (default: 1).",
    )
    parser.add_argument(
        "--cpu-count",
        type=int,
        default=1,
        help="Assumed number of CPU units contributing to the cost model (default: 1).",
    )
    parser.add_argument(
        "--workload",
        type=str,
        default="unknown",
        help="Top-level workload/benchmark name used in the output directory (default: unknown).",
    )
    parser.add_argument(
        "--egglog-binary",
        default="egglog-experimental",
        help="egglog executable to run (default: %(default)s)",
    )
    return parser.parse_args(argv)


def main(argv: List[str]) -> None:
    args = parse_args(argv)
    base_dir: pathlib.Path = args.output_dir
    workload_dir = base_dir / args.workload
    kernel_dir = workload_dir / args.function
    kernel_dir.mkdir(parents=True, exist_ok=True)

    result = capture_kernel_cost(
        source_mlir=args.source_mlir.resolve(),
        func_name=args.function,
        output_dir=kernel_dir.resolve(),
        parallelism_label=args.parallelism_label,
        array_length=args.array_length,
        layout_mode=args.layout_mode,
        egglog_binary=args.egglog_binary,
        cpu_count=args.cpu_count,
    )
    result["parallelism_label"] = args.parallelism_label
    result["workload"] = args.workload
    # Results are persisted on disk; no need to spam stdout.


if __name__ == "__main__":
    main(sys.argv[1:])
