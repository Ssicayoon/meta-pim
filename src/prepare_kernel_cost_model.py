#!/usr/bin/env python3
"""Detect kernel bitwidth from MLIR and refresh the PIM cost model.

Typical usage:
  python3 src/prepare_kernel_cost_model.py \
      --source-mlir experiments/<benchmark>/<kernel>_scop.mlir --function S0

Optionally pass --bitwidth to override detection (e.g., when testing synthetic
inputs). Every update rewrites the cost tables in mlir_to_egg.py and records the
parameters in out/kernel_costs/cost_model_update.json.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import re
import sys
from pathlib import Path
from typing import Dict
import math

SCALE_FACTOR = 100000
# In the minimal_release layout, this file lives under REPO_ROOT/src/.
REPO_ROOT = Path(__file__).resolve().parents[1]
MLIR_TO_EGG = REPO_ROOT / "src" / "mlir_to_egg.py"

sys.path.insert(0, str(MLIR_TO_EGG.parent))
import mlir_to_egg  # type: ignore  # pylint: disable=wrong-import-position


class BitwidthDetectionError(RuntimeError):
    pass


FUNC_HEADER_RE = re.compile(
    r"func\.func\s+(?:private\s+)?@(?P<name>[\w\d_]+)\(",
    re.S,
)


def extract_function_block(text: str, func_name: str) -> str:
    start_match = re.search(
        rf"func\.func\s+[^\{{]*@{re.escape(func_name)}\b", text
    )
    if not start_match:
        raise BitwidthDetectionError(f"function @{func_name} not found")

    idx = start_match.end()
    while True:
        brace = text.find("{", idx)
        if brace == -1:
            raise BitwidthDetectionError(f"missing body for @{func_name}")
        pre = text[idx:brace]
        if "attributes" in pre:
            closing = text.find("}", brace)
            if closing == -1:
                raise BitwidthDetectionError(f"unterminated attributes for @{func_name}")
            idx = closing + 1
            continue
        break

    depth = 0
    end = brace
    while end < len(text):
        if text[end] == "{":
            depth += 1
        elif text[end] == "}":
            depth -= 1
            if depth == 0:
                return text[start_match.start() : end + 1]
        end += 1
    raise BitwidthDetectionError(f"unterminated body for @{func_name}")


STORE_TYPE_RE = re.compile(r"memref<[^>]*x(?P<type>i\d+)>")
RETURN_TYPE_RE = re.compile(r"return\s+.+:\s*(?P<type>i\d+)")
TYPE_BITS_RE = re.compile(r"i(?P<bits>\d+)")


def infer_bitwidth(func_block: str) -> int:
    store_match = STORE_TYPE_RE.search(func_block)
    if store_match:
        return parse_type_to_bits(store_match.group("type"))
    return_match = RETURN_TYPE_RE.search(func_block)
    if return_match:
        return parse_type_to_bits(return_match.group("type"))
    raise BitwidthDetectionError("failed to infer bitwidth (no memref/return types)")


def parse_type_to_bits(type_str: str) -> int:
    type_str = type_str.strip()
    match = TYPE_BITS_RE.fullmatch(type_str)
    if not match:
        raise BitwidthDetectionError(f"unsupported type format: {type_str}")
    return int(match.group("bits"))


def format_dict(name: str, mapping: Dict[str, int]) -> str:
    body_lines = [f'    "{key}": {value},' for key, value in mapping.items()]
    body = "\n".join(body_lines)
    return f"{name}: Dict[str, int] = {{\n{body}\n}}\n"


def replace_section(text: str, name: str, new_block: str) -> str:
    pattern = re.compile(rf"{name}: Dict\[str, int\] = \{{.*?\}}\n", re.S)
    new_text, count = pattern.subn(new_block, text, count=1)
    if count != 1:
        raise RuntimeError(f"failed to update {name}")
    return new_text


def _scale_cost(value: int, parallelism: int) -> int:
    if value == 0:
        return 0
    scaled = math.ceil((value / parallelism) * SCALE_FACTOR)
    return max(1, scaled)


def compute_bp_costs(bitwidth: int, shift_bits: int, parallelism: int) -> Dict[str, int]:
    base = {
        "ExtSIBP": 0,
        "MulBP": bitwidth + 2,
        "DivBP": bitwidth + 2,
        "RemBP": 0,
        "TruncBP": 0,
        "AddBP": 1,
        "SubBP": 2,
        "ShrUBP": shift_bits,
        "AndBP": 1,
        "XorBP": 1,
        "OrBP": 1,
        "ShrSBP": shift_bits,
    }
    return {k: _scale_cost(v, parallelism) for k, v in base.items()}


def compute_bs_costs(bitwidth: int, parallelism: int) -> Dict[str, int]:
    logic_cost = bitwidth
    base = {
        "ExtSIBS": 0,
        "MulBS": bitwidth * bitwidth,
        "DivBS": bitwidth * bitwidth,
        "RemBS": 0,
        "TruncBS": 0,
        "AddBS": logic_cost,
        "SubBS": logic_cost,
        "ShrUBS": 0,
        "AndBS": logic_cost,
        "XorBS": logic_cost,
        "OrBS": logic_cost,
        "ShrSBS": 0,
    }
    return {k: _scale_cost(v, parallelism) for k, v in base.items()}


def compute_cast_costs(bitwidth: int) -> Dict[str, int]:
    cost = bitwidth + 2
    return {
        "CastBPtoBS": cost * SCALE_FACTOR,
        "CastBStoBP": cost * SCALE_FACTOR,
    }


def compute_load_costs(bitwidth: int) -> Dict[str, int]:
    return {
        "LoadCPUToBP": 1 * SCALE_FACTOR,
        "LoadCPUToBS": bitwidth * SCALE_FACTOR,
        "LoadConstantBP": 0,
        "LoadConstantBS": 0,
    }



def update_cost_tables(
    bitwidth: int,
    shift_bits: int,
    parallelism: int,
    target: Path,
) -> Dict[str, object]:
    bp_costs = compute_bp_costs(bitwidth, shift_bits, parallelism)
    bs_costs = compute_bs_costs(bitwidth, parallelism)
    cast_costs = compute_cast_costs(bitwidth)
    load_costs = compute_load_costs(bitwidth)

    text = target.read_text()
    bp_snippet = format_dict("_BP_COST_BASE", bp_costs)
    bs_snippet = format_dict("_BS_COST_BASE", bs_costs)
    cast_snippet = format_dict("CAST_COSTS", cast_costs)
    load_snippet = format_dict("LOAD_COSTS", load_costs)

    text = replace_section(text, "_BP_COST_BASE", bp_snippet)
    text = replace_section(text, "_BS_COST_BASE", bs_snippet)
    text = replace_section(text, "CAST_COSTS", cast_snippet)
    text = replace_section(text, "LOAD_COSTS", load_snippet)
    target.write_text(text)

    # Verify the rewrite succeeded before moving on.
    rewritten = target.read_text()
    for snippet, name in [
        (bp_snippet, "_BP_COST_BASE"),
        (bs_snippet, "_BS_COST_BASE"),
        (cast_snippet, "CAST_COSTS"),
        (load_snippet, "LOAD_COSTS"),
    ]:
        if snippet not in rewritten:
            raise RuntimeError(f"failed to update {name} in {target}")

    payload = {
        "target": str(target),
        "bitwidth": bitwidth,
        "shift_bits": shift_bits,
        "parallelism": parallelism,
        "bp_costs": bp_costs,
        "bs_costs": bs_costs,
        "cast_costs": cast_costs,
        "load_costs": load_costs,
        "cpu_costs": dict(mlir_to_egg.CPU_COSTS),
    }
    return payload


def write_log(log_dir: Path, payload: Dict[str, object]) -> Path:
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = _dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
    enriched = dict(payload)
    enriched["timestamp"] = timestamp
    path = log_dir / "cost_model_update.json"
    path.write_text(json.dumps(enriched, indent=2, sort_keys=True))
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Infer kernel bitwidth and update mlir_to_egg cost tables."
    )
    parser.add_argument(
        "--source-mlir",
        type=Path,
        help="Stage-split MLIR file containing scop.stmt functions.",
        default=None,
    )
    parser.add_argument(
        "--function",
        help="Function name to analyze (e.g., S0).",
        default=None,
    )
    parser.add_argument(
        "--bitwidth",
        type=int,
        default=None,
        help="Override detected precision p (skips MLIR parsing).",
    )
    parser.add_argument(
        "--shift-bits",
        type=int,
        default=None,
        help="Override BP shift latency (defaults to detected bitwidth).",
    )
    parser.add_argument(
        "--parallelism",
        type=int,
        default=1,
        help="Number of independent kernel instances assumed to run concurrently (default: 1).",
    )
    parser.add_argument(
        "--parallelism-label",
        type=str,
        default=None,
        help="Label stored with the cost model (defaults to the parallelism value).",
    )
    parser.add_argument(
        "--workload",
        type=str,
        default="unknown",
        help="Workload/benchmark name for organizing per-kernel logs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "out" / "kernel_costs",
        help="Base directory for kernel logs (default: out/kernel_costs).",
    )
    parser.add_argument(
        "--target",
        type=Path,
        default=MLIR_TO_EGG,
        help="mlir_to_egg.py path to update.",
    )
    args = parser.parse_args()
    if args.bitwidth is None:
        if args.source_mlir is None or args.function is None:
            parser.error("provide --bitwidth or both --source-mlir and --function")
    return args


def main() -> None:
    args = parse_args()
    if args.parallelism <= 0:
        raise SystemExit("--parallelism must be >= 1")
    if args.parallelism_label:
        parallelism_label = args.parallelism_label
    else:
        parallelism_label = str(args.parallelism)
    if args.bitwidth is not None:
        bitwidth = args.bitwidth
    else:
        module_text = args.source_mlir.read_text()
        func_block = extract_function_block(module_text, args.function)
        bitwidth = infer_bitwidth(func_block)
    shift_bits = args.shift_bits if args.shift_bits is not None else bitwidth
    payload = update_cost_tables(
        bitwidth=bitwidth,
        shift_bits=shift_bits,
        parallelism=args.parallelism,
        target=args.target,
    )
    kernel_dir = args.output_dir / args.workload / args.function
    kernel_dir.mkdir(parents=True, exist_ok=True)
    kernel_payload = dict(payload)
    kernel_payload["parallelism_label"] = parallelism_label
    kernel_log_path = kernel_dir / f"p{parallelism_label}_cost_model.json"
    kernel_log_path.write_text(json.dumps(kernel_payload, indent=2, sort_keys=True))
    if args.source_mlir and args.function:
        print(
            f"Detected bitwidth p={bitwidth} (shift_bits={shift_bits}) "
            f"for @{args.function} in {args.source_mlir}"
        )
    else:
        print(f"Applied bitwidth override p={bitwidth} (shift_bits={shift_bits})")
    print(
        f"Updated {args.target} for parallelism={args.parallelism} "
        f"(label={parallelism_label}); log: {kernel_log_path}"
    )


if __name__ == "__main__":
    sys.exit(main())
