#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def parse_stage_signature_and_store_indices(stage_mlir: str) -> Tuple[str, List[Tuple[str, str]], Dict[str, int]]:
    """
    Returns:
    - memref_type: the memref type string of %arg0 (e.g., 'memref<?x16xi64>')
    - scalars: list of (name, mlir_type) for scalar args in order (arg1..)
    - s_to_col: map 'Sx' -> column index k extracted from affine.store to %arg0[0, k]
    """
    # Find any S0 function as template for signature
    func_sig_re = re.compile(
        r"func\.func\s+private\s+@S0\(\%arg0:\s*([^,]+),\s*(.*?)\)\s+attributes\s*\{scop\.stmt\}",
        re.DOTALL,
    )
    m = func_sig_re.search(stage_mlir)
    if not m:
        raise RuntimeError("Failed to locate S0 signature in stage MLIR")
    memref_type = m.group(1).strip()
    scalar_part = m.group(2)
    # Parse scalar args "%arg1: i64, %arg2: i64, ..."
    scalars: List[Tuple[str, str]] = []
    for seg in scalar_part.split(","):
        seg = seg.strip()
        if not seg:
            continue
        name_type = seg.split(":")
        if len(name_type) != 2:
            continue
        name = name_type[0].strip()
        typ = name_type[1].strip()
        scalars.append((name, typ))

    # Extract store indices per statement
    s_to_col: Dict[str, int] = {}
    func_block_re = re.compile(
        r"func\.func\s+private\s+@(S\d+)\([^\)]*\)\s+attributes\s*\{scop\.stmt\}\s*\{(.*?)\}",
        re.DOTALL,
    )
    for fm in func_block_re.finditer(stage_mlir):
        sname = fm.group(1)
        body = fm.group(2)
        store_m = re.search(r"affine\.store\s+%[a-zA-Z0-9_]+,\s*%arg0\[\s*0\s*,\s*(\d+)\s*\]\s*:\s*memref", body)
        if store_m:
            s_to_col[sname] = int(store_m.group(1))
    return memref_type, scalars, s_to_col


def derive_with_pim_path(cost_path: str, repo_root: Path) -> Path:
    # cost_path example: out/kernel_costs/bp/gemm/S0/p1_cost.json
    rel = Path(cost_path)
    mlir_name = rel.name.replace("_cost.json", "_with_pim.mlir")
    return (repo_root / rel.parent / mlir_name).resolve()


def extract_snippet_function(snippet_mlir: str) -> Tuple[str, str, str, List[str]]:
    """
    Find the only function in snippet MLIR and return (name, param_sig, ret_type, body_lines).
    """
    m = re.search(r"func\.func\s+@([a-zA-Z0-9_]+)\((.*?)\)\s*->\s*\((.*?)\)\s*\{\s*(.*?)\s*\}\s*", snippet_mlir, re.DOTALL)
    if not m:
        raise RuntimeError("Failed to locate function in with-PIM MLIR")
    fname = m.group(1)
    params = m.group(2).strip()
    ret = m.group(3).strip()
    body = m.group(4)
    # Strip trailing 'return ...' and keep the SSA that is returned
    lines = [ln.rstrip() for ln in body.splitlines() if ln.strip()]
    return fname, params, ret, lines


def emit_function(new_name: str, params: str, ret_ty: str, body_lines: List[str]) -> str:
    return "\n".join(
        [
            f"  func.func @{new_name}({params}) -> ({ret_ty}) {{",
            *[f"    {ln}" for ln in body_lines],
            "  }",
        ]
    )


def replace_statement_bodies_with_calls(
    stage_mlir: str,
    scalars: List[Tuple[str, str]],
    schedule_entries: List[Tuple[str, int, Path]],
    s_to_col: Dict[str, int],
) -> str:
    """
    Keep original entry and function signatures intact.
    For each @Sx, replace function body with:
      %r = func.call @pim_Sx(%arg1..%argN) : (i64, ... x N) -> (i64)
      affine.store %r, %arg0[0, col] : memref<?x16xi64>
      return
    Also append definitions of @pim_Sx at end of module.
    """
    # Map sname -> path for quick lookup
    s_to_path: Dict[str, Path] = {s: p for s, _, p in schedule_entries}
    scalar_types = ", ".join(t for _, t in scalars)
    scalar_names = ", ".join(name for name, _ in scalars)

    def repl_fn(m: re.Match) -> str:
        # Groups: 1) full header, 2) S-name, 3) signature, 4) closing brace
        sname = m.group(2)
        signature = m.group(3)  # inside parens
        col = s_to_col.get(sname)
        if col is None:
            # If no store column known, leave original body unchanged
            return m.group(0)
        body = []
        body.append(f"  func.func private @{sname}({signature}) attributes {{scop.stmt}} {{")
        # Call uses scalar args only (skip %arg0 memref)
        if scalar_types:
            body.append(
                f"    %r0 = func.call @pim_{sname}({scalar_names}) : ({scalar_types}) -> (i64)"
            )
        else:
            body.append("    %r0 = func.call @pim_{sname}() : () -> (i64)")
        # Store back to memref %arg0
        body.append(f"    affine.store %r0, %arg0[0, {col}] : memref<?x16xi64>")
        body.append("    return")
        body.append("  }")
        return "\n".join(body)

    # Replace each @Sx body
    func_block_re = re.compile(
        r"(func\.func\s+private\s+@(S\d+)\(([^)]*)\)\s+attributes\s*\{scop\.stmt\}\s*\{)(.*?)(^\s*\})",
        re.DOTALL | re.MULTILINE,
    )
    new_text = func_block_re.sub(lambda m: repl_fn(m), stage_mlir)

    # Append @pim_Sx function definitions
    appended: List[str] = []
    for sname, _, mlir_path in schedule_entries:
        snippet_text = read_text(mlir_path)
        _, params, ret_ty, body_lines = extract_snippet_function(snippet_text)
        appended.append(emit_function(f"pim_{sname}", params, ret_ty, body_lines))

    # Insert appended before final closing brace of module
    if appended:
        new_text = new_text.rstrip()
        if new_text.endswith("}"):
            new_text = new_text[:-1] + "\n" + "\n".join(appended) + "\n}\n"
        else:
            new_text = new_text + "\n" + "\n".join(appended) + "\n"
    return new_text


def build_metadata(schedule: Dict) -> Tuple[List[Dict], List[int], Dict[int, Dict]]:
    """Return (statements_list, group_order, group_info)."""
    kernels = schedule.get("kernels", [])
    statements: List[Dict] = []
    group_order: List[int] = []
    group_info: Dict[int, Dict[str, List[int]]] = {}

    for kernel in kernels:
        name = kernel.get("name")
        if not name:
            continue
        bundles = kernel.get("bundles") or []
        if not bundles:
            continue
        parallelism = int(bundles[0].get("parallelism", 1))
        group_id = int(kernel.get("group_id", 0))

        if group_id not in group_info:
            group_info[group_id] = {"indices": []}
            group_order.append(group_id)

        stmt_index = len(statements)
        group_info[group_id]["indices"].append(stmt_index)
        statements.append({"name": name, "parallelism": parallelism})

    return statements, group_order, group_info


def write_metadata_files(schedule: Dict, output_dir: Path) -> None:
    statements, group_order, group_info = build_metadata(schedule)
    target_p = schedule.get("target_P")
    if target_p is None:
        # fallback: sum of parallelism for first group
        target_p = sum(stmt["parallelism"] for stmt in statements)

    metadata = {
        "workload": schedule.get("workload"),
        "target_P": schedule.get("target_P"),
        "groups": [
            {
                "group_id": gid,
                "statements": [
                    {
                        "name": statements[idx]["name"],
                        "parallelism": statements[idx]["parallelism"],
                    }
                    for idx in group_info[gid]["indices"]
                ],
            }
            for gid in group_order
        ],
        "statements": statements,
    }

    json_path = output_dir / "scheduled_kernel_metadata.json"
    json_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"[assemble] Wrote metadata JSON to {json_path}")

    statement_count = len(statements)
    group_ids = group_order
    group_offsets: List[int] = []
    group_counts: List[int] = []
    for gid in group_ids:
        indices = group_info[gid]["indices"]
        if not indices:
            group_offsets.append(0)
            group_counts.append(0)
            continue
        group_offsets.append(indices[0])
        group_counts.append(len(indices))

    names_array = ", ".join(f"\"{stmt['name']}\"" for stmt in statements)
    parallel_array = ", ".join(str(stmt["parallelism"]) for stmt in statements)
    header_lines = [
        "#ifndef SCHEDULED_KERNEL_METADATA_H",
        "#define SCHEDULED_KERNEL_METADATA_H",
        "",
        "#include <stddef.h>",
        "",
        f"static const int kScheduleGroupCount = {len(group_ids)};",
        f"static const int kScheduleStatementCount = {statement_count};",
        f"static const int kScheduleTargetParallelism = {int(target_p)};",
        "",
        "static const int kScheduleGroupIds[] = {" + ", ".join(str(gid) for gid in group_ids) + "};",
        "static const int kScheduleGroupStatementCounts[] = {" + ", ".join(str(cnt) for cnt in group_counts) + "};",
        "static const int kScheduleGroupStatementOffsets[] = {" + ", ".join(str(off) for off in group_offsets) + "};",
        "static const char *kScheduleStatementNames[] = {" + names_array + "};",
        "static const int kScheduleStatementParallelism[] = {" + parallel_array + "};",
        "",
        "#endif  // SCHEDULED_KERNEL_METADATA_H",
        "",
    ]
    header_path = output_dir / "scheduled_kernel_metadata.h"
    header_path.write_text("\n".join(header_lines), encoding="utf-8")
    print(f"[assemble] Wrote metadata header to {header_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Assemble scheduled kernel MLIR from schedule.json")
    ap.add_argument("--schedule", required=True, help="Path to schedule.json")
    ap.add_argument("--source-mlir", required=True, help="Stage MLIR used by scheduler (for signature/store indices)")
    ap.add_argument("--output", required=True, help="Output MLIR path (scheduled kernel)")
    ap.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[1]), help="Repository root")
    ap.add_argument("--generate-missing", action="store_true", help="Generate missing with-PIM MLIRs via script")
    ap.add_argument("--generate-metadata", action="store_true", help="Emit metadata artifacts for pthread driver")
    args = ap.parse_args()

    repo_root = Path(args.repo_root).resolve()
    schedule_path = Path(args.schedule).resolve()
    source_mlir_path = Path(args.source_mlir).resolve()
    output_path = Path(args.output).resolve()

    schedule = json.loads(schedule_path.read_text(encoding="utf-8"))
    kernels = schedule.get("kernels", [])
    if not kernels:
        raise RuntimeError("No kernels in schedule")

    stage_mlir = read_text(source_mlir_path)
    memref_type, scalars, s_to_col = parse_stage_signature_and_store_indices(stage_mlir)

    schedule_entries: List[Tuple[str, int, Path]] = []
    for k in kernels:
        sname = k["name"]
        bundle = k["bundles"][0]
        cost_path = bundle["cost_path"]
        with_pim = derive_with_pim_path(cost_path, repo_root)
        if not with_pim.exists():
            if args.generate_missing:
                # Best-effort generation by calling the Egglog→MLIR PIM generator
                os.system(str(repo_root / "scripts" / "egg_to_pim_mlir.sh"))
            if not with_pim.exists():
                raise FileNotFoundError(f"Missing with-PIM MLIR: {with_pim}")
        schedule_entries.append((sname, int(bundle["parallelism"]), with_pim))

    program_mlir = replace_statement_bodies_with_calls(stage_mlir, scalars, schedule_entries, s_to_col)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(program_mlir, encoding="utf-8")
    print(f"[assemble] Wrote {output_path}")

    if args.generate_metadata:
        write_metadata_files(schedule, output_path.parent)


if __name__ == "__main__":
    main()
