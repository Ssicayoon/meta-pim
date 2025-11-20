#!/usr/bin/env python3
"""Offline cost analysis for Egglog extracted expressions."""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Tuple, Union

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(SCRIPT_DIR))

import mlir_to_egg


Category = str
SExpr = Union[str, int, list]


def _symbol(node: Any) -> str | None:
    return node if isinstance(node, str) else None


def _update_cost(
    category: Category,
    op: str,
    mapping: Mapping[str, int],
    counts: Dict[Category, Counter],
    totals: Dict[Category, int],
) -> None:
    if op not in mapping:
        return
    counts[category][op] += 1
    totals[category] += mapping[op]


def analyse_expr(expr: SExpr, counts: Dict[Category, Counter], totals: Dict[Category, int]) -> None:
    if not isinstance(expr, list):
        return

    iterator = iter(expr)
    head = next(iterator, None)
    op = _symbol(head)
    if op is None:
        # descend into children anyway
        for child in iterator:
            analyse_expr(child, counts, totals)
        return

    _update_cost("cpu", op, mlir_to_egg.CPU_COSTS, counts, totals)
    _update_cost("bp", op, mlir_to_egg.BP_COSTS, counts, totals)
    _update_cost("bs", op, mlir_to_egg.BS_COSTS, counts, totals)
    _update_cost("load", op, mlir_to_egg.LOAD_COSTS, counts, totals)
    _update_cost("store", op, mlir_to_egg.STORE_COSTS, counts, totals)
    _update_cost("cast", op, mlir_to_egg.CAST_COSTS, counts, totals)

    for child in iterator:
        analyse_expr(child, counts, totals)


def _tokenise(text: str) -> Iterable[str]:
    buf = []
    for ch in text:
        if ch in "()":
            if buf:
                yield "".join(buf)
                buf = []
            yield ch
        elif ch.isspace():
            if buf:
                yield "".join(buf)
                buf = []
        else:
            buf.append(ch)
    if buf:
        yield "".join(buf)


def _parse_sexpr(text: str) -> SExpr:
    tokens = list(_tokenise(text))
    idx = 0

    def parse_inner() -> SExpr:
        nonlocal idx
        if idx >= len(tokens):
            raise ValueError("unexpected end of input")
        tok = tokens[idx]
        idx += 1
        if tok == "(":
            items: list = []
            while idx < len(tokens) and tokens[idx] != ")":
                items.append(parse_inner())
            if idx >= len(tokens):
                raise ValueError("missing closing ')'")
            idx += 1  # consume ')'
            return items
        if tok == ")":
            raise ValueError("unexpected ')'")
        if tok.lstrip("-").isdigit():
            return int(tok)
        return tok

    result = parse_inner()
    if idx != len(tokens):
        raise ValueError("extra tokens after parsing expression")
    return result


def format_report(counts: Dict[Category, Counter], totals: Dict[Category, int]) -> str:
    lines = []
    categories = [
        ("cpu", "CPU"),
        ("bp", "BP"),
        ("bs", "BS"),
        ("load", "Loads"),
        ("store", "Stores"),
        ("cast", "Casts"),
    ]
    grand_total = sum(totals.values())
    lines.append(f"Total estimated cost: {grand_total}")
    for key, title in categories:
        total = totals.get(key, 0)
        lines.append(f"{title}: {total}")
        details = counts.get(key)
        if not details:
            continue
        for op, count in details.most_common():
            cost_per = (
                mlir_to_egg.CPU_COSTS.get(op)
                or mlir_to_egg.BP_COSTS.get(op)
                or mlir_to_egg.BS_COSTS.get(op)
                or mlir_to_egg.LOAD_COSTS.get(op)
                or mlir_to_egg.STORE_COSTS.get(op)
                or mlir_to_egg.CAST_COSTS.get(op)
                or 0
            )
            lines.append(f"  - {op}: count={count} cost_each={cost_per} subtotal={count * cost_per}")
    return "\n".join(lines)


def compute_cost_from_expr(expr_text: str, array_length: int | None = None) -> Dict[str, Any]:
    if array_length is not None:
        mlir_to_egg.configure_array_length(array_length)

    text = expr_text.strip()
    if not text:
        raise ValueError("empty expression text")

    parsed_expr = _parse_sexpr(text)
    counts: Dict[Category, Counter] = defaultdict(Counter)
    totals: Dict[Category, int] = defaultdict(int)
    analyse_expr(parsed_expr, counts, totals)
    grand_total = sum(totals.values())
    return {
        "grand_total": grand_total,
        "totals": dict(totals),
        "counts": {cat: dict(counter) for cat, counter in counts.items()},
    }


def compute_cost_from_file(path: Path, array_length: int | None = None) -> Dict[str, Any]:
    return compute_cost_from_expr(path.read_text(), array_length=array_length)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute cost breakdown for an Egglog extracted expression.")
    parser.add_argument("expr_path", type=Path, help="path to the extracted expression (text file)")
    parser.add_argument(
        "--array-length",
        type=int,
        default=mlir_to_egg.ARRAY_LENGTH,
        help="array length used during lowering (default: %(default)s)",
    )
    parser.add_argument("--json", action="store_true", help="emit JSON instead of human-readable text")
    args = parser.parse_args()

    mlir_to_egg.configure_array_length(args.array_length)

    result = compute_cost_from_file(args.expr_path, array_length=args.array_length)

    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        totals = defaultdict(int, result["totals"])
        counts = {cat: Counter(data) for cat, data in result["counts"].items()}
        print(format_report(counts, totals))


if __name__ == "__main__":
    main()
