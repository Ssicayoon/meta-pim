#!/usr/bin/env python3
"""Rewrite PIM dialect operations (pimbp/pimbs) into intrinsic function calls.

This keeps track of each `pimbp.*`/`pimbs.*` operation and converts it to a `func.call`
of a synthetic intrinsic such as `@llvm.pimbp.addi.i32` or `@llvm.pimbs.addi.i32`.  Later lowering steps
(`--convert-func-to-llvm`) will turn these into LLVM calls, and the downstream
toolchain (or gem5) can provide implementations for the intrinsics or further
rewrite them to inline assembly.
"""

from __future__ import annotations

import argparse
import pathlib
import re
from typing import Dict, Iterable, List, Tuple


Declaration = Tuple[Tuple[str, ...], str]  # (arg_types, result_type)


BINARY_OPS: Dict[str, str] = {}
SHIFT_OPS: Dict[str, str] = {}
CAST_OPS: Dict[str, str] = {}
CPU_XFER_OPS: Dict[str, str] = {}

for prefix in ("pimbp", "pimbs"):
    base = f"llvm.{prefix}"
    BINARY_OPS.update({
        f"{prefix}.addi": f"{base}.addi",
        f"{prefix}.subi": f"{base}.subi",
        f"{prefix}.andi": f"{base}.andi",
        f"{prefix}.xori": f"{base}.xori",
        f"{prefix}.ori": f"{base}.ori",
        f"{prefix}.muli": f"{base}.muli",
        f"{prefix}.remsi": f"{base}.remsi",
    })
    SHIFT_OPS.update({
        f"{prefix}.shrui": f"{base}.shrui",
        f"{prefix}.shrsi": f"{base}.shrsi",
    })
    CAST_OPS.update({
        f"{prefix}.extsi": f"{base}.extsi",
        f"{prefix}.trunci": f"{base}.trunci",
    })
    CPU_XFER_OPS.update({
        f"{prefix}.from_cpu": f"{base}.from_cpu",
        f"{prefix}.to_cpu": f"{base}.to_cpu",
    })


def intrinsic_name(base: str, *types: str) -> str:
    suffix = ".".join(types)
    return f"{base}.{suffix}" if suffix else base


def build_call(
    indent: str,
    result: str,
    intrinsic: str,
    operands: Iterable[str],
    arg_types: Tuple[str, ...],
    result_type: str,
) -> str:
    arg_list = ", ".join(operands)
    arg_sig = ", ".join(arg_types)
    arg_tuple = f"({arg_sig})" if arg_sig else "()"
    return (
        f"{indent}{result} = func.call @{intrinsic}"
        f"({arg_list}) : ({arg_sig}) -> {result_type}"
        if arg_sig
        else f"{indent}{result} = func.call @{intrinsic}() : () -> {result_type}"
    )


def rewrite_lines(lines: List[str]) -> Tuple[List[str], Dict[str, Declaration]]:
    new_lines: List[str] = []
    declarations: Dict[str, Declaration] = {}

    binary_re = re.compile(
        r"(?P<indent>\s*)(?P<result>%[\w\d_.]+)\s*=\s*(?P<op>pimb[ps]\.\w+)\s+"
        r"(?P<operands>[^:]+)\s*:\s*(?P<type>\S+)"
    )
    cast_re = re.compile(
        r"(?P<indent>\s*)(?P<result>%[\w\d_.]+)\s*=\s*(?P<op>pimb[ps]\.(?:extsi|trunci))\s+"
        r"(?P<operand>[^:]+)\s*:\s*(?P<src>\S+)\s+to\s+(?P<dst>\S+)"
    )
    cpu_re = re.compile(
        r"(?P<indent>\s*)(?P<result>%[\w\d_.]+)\s*=\s*(?P<op>pimb[ps]\.(?:from_cpu|to_cpu))\s+"
        r"(?P<operand>[^:]+)\s*:\s*(?P<type>\S+)"
    )
    const_re = re.compile(
        r"(?P<indent>\s*)(?P<result>%[\w\d_.]+)\s*=\s*pimb[ps]\.constant\s+"
        r"(?P<value>-?\d+)\s*:\s*(?P<type>\S+)"
    )

    for line in lines:
        stripped = line.strip()
        if not stripped.startswith("%"):
            new_lines.append(line)
            continue

        const_match = const_re.match(line)
        if const_match:
            indent = const_match.group("indent")
            res = const_match.group("result")
            value = const_match.group("value")
            typ = const_match.group("type")
            new_lines.append(f"{indent}{res} = arith.constant {value} : {typ}")
            continue

        cast_match = cast_re.match(line)
        if cast_match:
            indent = cast_match.group("indent")
            res = cast_match.group("result")
            op = cast_match.group("op")
            operand = cast_match.group("operand").strip()
            src = cast_match.group("src")
            dst = cast_match.group("dst")
            base = CAST_OPS[op]
            name = intrinsic_name(base, src, dst)
            declarations[name] = ((src,), dst)
            new_lines.append(
                f"{indent}{res} = func.call @{name}({operand}) : ({src}) -> {dst}"
            )
            continue

        cpu_match = cpu_re.match(line)
        if cpu_match:
            indent = cpu_match.group("indent")
            res = cpu_match.group("result")
            op = cpu_match.group("op")
            operand = cpu_match.group("operand").strip()
            typ = cpu_match.group("type")
            base = CPU_XFER_OPS[op]
            name = intrinsic_name(base, typ)
            if op.endswith("from_cpu"):
                declarations[name] = ((typ,), typ)
                new_lines.append(
                    f"{indent}{res} = func.call @{name}({operand}) : ({typ}) -> {typ}"
                )
            else:  # to_cpu
                declarations[name] = ((typ,), typ)
                new_lines.append(
                    f"{indent}{res} = func.call @{name}({operand}) : ({typ}) -> {typ}"
                )
            continue

        binary_match = binary_re.match(line)
        if binary_match:
            indent = binary_match.group("indent")
            res = binary_match.group("result")
            op = binary_match.group("op")
            operands = [tok.strip() for tok in binary_match.group("operands").split(",")]
            typ = binary_match.group("type")

            if op in BINARY_OPS:
                base = BINARY_OPS[op]
                name = intrinsic_name(base, typ)
                declarations[name] = ((typ, typ), typ)
                operand_list = ", ".join(operands)
                new_lines.append(
                    f"{indent}{res} = func.call @{name}({operand_list}) : ({typ}, {typ}) -> {typ}"
                )
                continue

            if op in SHIFT_OPS:
                base = SHIFT_OPS[op]
                name = intrinsic_name(base, typ)
                declarations[name] = ((typ, typ), typ)
                operand_list = ", ".join(operands)
                new_lines.append(
                    f"{indent}{res} = func.call @{name}({operand_list}) : ({typ}, {typ}) -> {typ}"
                )
                continue

        # Default: leave line unchanged
        new_lines.append(line)

    return new_lines, declarations


def insert_declarations(lines: List[str], declarations: Dict[str, Declaration]) -> List[str]:
    if not declarations:
        return lines

    decl_lines = []
    for name, (arg_types, result_type) in sorted(declarations.items()):
        arg_sig = ", ".join(arg_types)
        decl = f"  func.func private @{name}({arg_sig}) -> {result_type}"
        decl_lines.append(decl)

    final_lines: List[str] = []
    inserted = False
    for line in lines:
        final_lines.append(line)
        if not inserted and line.strip() == "module {":
            final_lines.extend(decl_lines)
            inserted = True
    return final_lines


def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=pathlib.Path, help="Input MLIR file with PIM ops")
    parser.add_argument(
        "-o", "--output", type=pathlib.Path, required=True, help="Output MLIR file"
    )
    args = parser.parse_args(argv)

    lines = args.input.read_text().splitlines()
    rewritten, decls = rewrite_lines(lines)
    final_lines = insert_declarations(rewritten, decls)
    args.output.write_text("\n".join(final_lines) + "\n")


if __name__ == "__main__":
    main()
