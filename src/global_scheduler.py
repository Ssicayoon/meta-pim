#!/usr/bin/env python3
"""Advanced resource-aware scheduler with global bundle-combination search.

This tool scans per-statement cost artifacts under out/kernel_costs/<layout>/<workload>/<S*>/p*_cost.json,
retains at most two bundle combinations per statement for each target P, explores the cartesian
product across statements, simulates CPU/PIM/bandwidth constraints, and emits schedule.json/markdown.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import itertools
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
import time
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple


# ---------------------------
# Helpers from simple scheduler
# ---------------------------

def _normalize_index(expr: str) -> str:
    return re.sub(r"\s+", "", expr)


def _parse_linear_affine(expr: str) -> Optional[Tuple[int, int]]:
    """
    解析简单的线性仿射表达式: a*i + b
    返回 (a, b) 或 None
    
    支持的格式:
    - a*i + b
    - a*i - b
    - i + b
    - i - b
    - a*i
    - i
    - 常量
    """
    expr = expr.strip()
    
    # 尝试解析为常量
    try:
        const = int(expr)
        return (0, const)
    except ValueError:
        pass
    
    # 匹配 a*var + b 或 a*var - b
    match = re.match(r'(\d+)\*(\w+)\s*([+\-])\s*(\d+)', expr)
    if match:
        coeff = int(match.group(1))
        op = match.group(3)
        const = int(match.group(4))
        if op == '-':
            const = -const
        return (coeff, const)
    
    # 匹配 var + b 或 var - b
    match = re.match(r'(\w+)\s*([+\-])\s*(\d+)', expr)
    if match:
        op = match.group(2)
        const = int(match.group(3))
        if op == '-':
            const = -const
        return (1, const)
    
    # 匹配 a*var
    match = re.match(r'(\d+)\*(\w+)', expr)
    if match:
        coeff = int(match.group(1))
        return (coeff, 0)
    
    # 匹配单个变量
    if re.match(r'^\w+$', expr):
        return (1, 0)
    
    return None


def _gcd_test(idx1: str, idx2: str) -> bool:
    """
    GCD 测试判断两个线性仿射索引是否可能冲突
    返回 True 表示可能冲突，False 表示肯定不冲突
    """
    linear1 = _parse_linear_affine(idx1)
    linear2 = _parse_linear_affine(idx2)
    
    if linear1 is None or linear2 is None:
        # 无法解析，保守假设可能冲突
        return True
    
    a1, b1 = linear1
    a2, b2 = linear2
    
    # 两个都是常量
    if a1 == 0 and a2 == 0:
        return b1 == b2
    
    # 其中一个是常量，另一个是变量
    if a1 == 0 or a2 == 0:
        # 常量和变量可能冲突（保守假设）
        return True
    
    # 两个都是线性表达式
    # GCD 测试: 如果 gcd(a1, a2) 不能整除 (b2 - b1)，则无依赖
    g = math.gcd(abs(a1), abs(a2))
    diff = abs(b2 - b1)
    
    if diff % g != 0:
        # GCD 不能整除差值，肯定无依赖
        return False
    
    # 可能有依赖
    return True


def can_indices_conflict(mem1: str, idx1: str, mem2: str, idx2: str) -> bool:
    """
    判断两个内存访问是否可能冲突
    
    Args:
        mem1, mem2: memref 名称
        idx1, idx2: 索引表达式（可以是多维的，如 "0,1" 或 "i,j"）
    
    Returns:
        True 表示可能冲突，False 表示肯定不冲突
    """
    # 不同的 memref，肯定不冲突
    if mem1 != mem2:
        return False
    
    # 规范化索引表达式
    idx1_norm = _normalize_index(idx1)
    idx2_norm = _normalize_index(idx2)
    
    # 完全相同的表达式，肯定冲突
    if idx1_norm == idx2_norm:
        return True
    
    # 处理多维索引：按逗号分割
    dims1 = idx1_norm.split(',')
    dims2 = idx2_norm.split(',')
    
    # 维度不同，无法判断（保守返回 True）
    if len(dims1) != len(dims2):
        return True
    
    # 逐维度比较
    for d1, d2 in zip(dims1, dims2):
        d1 = d1.strip()
        d2 = d2.strip()
        
        # 完全相同的维度表达式
        if d1 == d2:
            continue
        
        # 尝试解析为常量
        try:
            const1 = int(d1)
            const2 = int(d2)
            # 如果任意一个维度的常量不同，则肯定不冲突
            if const1 != const2:
                return False
            # 如果相同，继续检查下一个维度
            continue
        except ValueError:
            pass
        
        # 如果有任意一个维度无法确定是否相同，使用 GCD 测试
        affine1 = _parse_linear_affine(d1)
        affine2 = _parse_linear_affine(d2)
        if affine1 and affine2:
            if not _gcd_test(d1, d2):
                return False
        else:
            # 无法解析，保守返回 True
            return True
    
    # 所有维度都可能冲突
    return True


def parse_stage_functions(stage_path: Path) -> Dict[str, Dict[str, List[Tuple[str, str]]]]:
    text = stage_path.read_text()
    functions: Dict[str, Dict[str, List[Tuple[str, str]]]] = {}
    pattern = re.compile(
        r"\s*func\.func\s+private\s+@(?P<name>\w+)\s*\((?P<header>.*?)\)\s*(?:attributes\s+\{.*?\}\s*)?(?P<body>\{)",
        re.S,
    )
    idx = 0
    while True:
        match = pattern.search(text, idx)
        if not match:
            break
        name = match.group("name")
        body_start = match.end("body")
        depth = 1
        pos = body_start
        while pos < len(text) and depth > 0:
            if text[pos] == "{":
                depth += 1
            elif text[pos] == "}":
                depth -= 1
            pos += 1
        body = text[body_start : pos - 1]
        idx = pos
        stores = re.findall(
            r"(?:affine|memref)\.store\s+%[\w\d_]+,\s+%(?P<mem>\w+)\[(?P<idx>[^\]]+)\]",
            body,
        )
        loads = re.findall(
            r"(?:affine|memref)\.load\s+%(?P<mem>\w+)\[(?P<idx>[^\]]+)\]",
            body,
        )
        store_list = [(mem, _normalize_index(idx_expr)) for mem, idx_expr in stores]
        load_list = [(mem, _normalize_index(idx_expr)) for mem, idx_expr in loads]
        functions[name] = {
            "stores": store_list,
            "loads": load_list,
        }
    return functions


def build_dependencies(
    kernels: List[str], stage_info: Dict[str, Dict[str, List[Tuple[str, str]]]]
) -> Dict[str, Set[str]]:
    deps: Dict[str, Set[str]] = {k: set() for k in kernels}
    have_stage_info = bool(stage_info)
    
    # 第一步：识别哪些内存位置在某个 statement 之前已经被写入
    # 我们按照 kernels 的顺序（即 MLIR 文件中的顺序）来判断
    def is_written_before(mem: str, idx: str, before_stmt: str) -> bool:
        """判断某个内存位置在 before_stmt 之前是否被写入过"""
        before_idx = kernels.index(before_stmt)
        for i in range(before_idx):
            s_info = stage_info.get(kernels[i])
            if not s_info:
                continue
            for write_mem, write_idx in s_info.get("stores", []):
                if can_indices_conflict(write_mem, write_idx, mem, idx):
                    return True
        return False
    
    for dst in kernels:
        dst_info = stage_info.get(dst)
        if not dst_info:
            idx_d = kernels.index(dst)
            if idx_d > 0:
                deps[dst].add(kernels[idx_d - 1])
            continue
        dst_loads = dst_info.get("loads", [])
        dst_stores = dst_info.get("stores", [])
        for src in kernels:
            if src == dst:
                continue
            src_info = stage_info.get(src)
            if not src_info:
                continue
            src_stores = src_info.get("stores", [])
            src_loads = src_info.get("loads", [])
            
            # RAW: dst 读取 src 写入的位置
            for mem_dst, idx_dst in dst_loads:
                # 如果 dst 读取的内存在它之前没有被写入过，说明是读取输入数据
                # 这种情况下不应该依赖于之后的 statement 的写入
                if not is_written_before(mem_dst, idx_dst, dst):
                    continue
                
                for mem_src, idx_src in src_stores:
                    if can_indices_conflict(mem_src, idx_src, mem_dst, idx_dst):
                        deps[dst].add(src)
                        break
                if src in deps[dst]:
                    break
            
            # WAW: dst 写入 src 写入的位置
            if src not in deps[dst]:
                for mem_dst, idx_dst in dst_stores:
                    for mem_src, idx_src in src_stores:
                        if can_indices_conflict(mem_src, idx_src, mem_dst, idx_dst):
                            deps[dst].add(src)
                            break
                    if src in deps[dst]:
                        break
            
            # WAR: dst 写入 src 读取的位置
            # 注意：WAR 依赖只有在 src 在 dst 之前执行时才有意义
            # 如果 dst 在 src 之前，那么 dst 的写入不会影响 src 的读取
            if src not in deps[dst]:
                src_idx = kernels.index(src)
                dst_idx = kernels.index(dst)
                
                # 只有当 src 在 dst 之前时，才检查 WAR 依赖
                if src_idx < dst_idx:
                    for mem_dst, idx_dst in dst_stores:
                        for mem_src, idx_src in src_loads:
                            # 如果 src 读取的内存在 src 之前没有被写入过，说明是读取输入数据
                            # 这种情况下不应该产生 WAR 依赖
                            if not is_written_before(mem_src, idx_src, src):
                                continue
                            
                            if can_indices_conflict(mem_src, idx_src, mem_dst, idx_dst):
                                deps[dst].add(src)
                                break
                        if src in deps[dst]:
                            break
        
        # Fallback: 只有当 statement 没有 load/store 信息时，才添加顺序依赖
        # 这样可以避免为完全独立的 statements（如 GEMM）添加不必要的依赖
        if not deps[dst] and have_stage_info:
            # 检查是否有 load/store 信息
            has_memory_ops = (dst_info and 
                            (dst_info.get("loads") or dst_info.get("stores")))
            # 如果没有内存操作信息，则添加顺序依赖以保持安全
            if not has_memory_ops:
                idx_d = kernels.index(dst)
                if idx_d > 0:
                    deps[dst].add(kernels[idx_d - 1])
    return deps


def topo_order(graph: Dict[str, Set[str]]) -> List[str]:
    indegree: Dict[str, int] = {node: 0 for node in graph}
    for deps in graph.values():
        for pred in deps:
            indegree[pred] = indegree.get(pred, 0)
    for node, preds in graph.items():
        for pred in preds:
            indegree[node] += 1
    ready = [node for node, deg in indegree.items() if deg == 0]
    order: List[str] = []
    while ready:
        node = ready.pop(0)
        order.append(node)
        for succ in graph:
            if node in graph[succ]:
                indegree[succ] -= 1
                if indegree[succ] == 0:
                    ready.append(succ)
    # If we have a cycle (incomplete order), fall back to sorted node list
    if len(order) < len(graph):
        order = sorted(graph.keys())
    return order


def find_parallel_groups(kernels: List[str], deps: Dict[str, Set[str]]) -> List[List[str]]:
    groups: List[List[str]] = []
    remaining = set(kernels)
    while remaining:
        group: List[str] = []
        for k in list(remaining):
            if deps.get(k, set()) & remaining:
                continue
            group.append(k)
        if not group:
            group.append(remaining.pop())
        else:
            for k in group:
                remaining.remove(k)
        groups.append(group)
    return groups

# ---------------------------
# Data structures
# ---------------------------

@dataclass
class CostEntry:
    statement: str
    layout: str
    parallelism: int
    totals_cpu: float
    totals_bp: float
    totals_bs: float
    totals_load: float
    totals_store: float
    totals_cast: float
    pim_rows_bp: Optional[int]
    pim_cols_bp: Optional[int]
    pim_rows_bs: Optional[int]
    pim_cols_bs: Optional[int]
    bitwidth: int
    cost_path: Path

    @property
    def total_cpu_work(self) -> float:
        """Total CPU work across all instances (already scaled by parallelism in p*_cost.json)
        
        Note: WriteMemRefCPU operations are moved to store category in scan_costs().
        For CPU layout, WriteMemRefCPU is CPU work (computation happens on CPU).
        For PIM layouts (BP/BS/Hybrid), WriteMemRefCPU is data movement, not CPU work.
        """
        cpu_work = float(self.totals_cpu)
        # Only for CPU layout: add store cost (WriteMemRefCPU is CPU work)
        if self.layout == "cpu":
            cpu_work += float(self.totals_store)
        return cpu_work

    @property
    def total_pim_work(self) -> float:
        """Total PIM work across all instances (already scaled by parallelism in p*_cost.json)"""
        return float(
            (self.totals_bp or 0)
            + (self.totals_bs or 0)
            + (self.totals_load or 0)
            + (self.totals_store or 0)
            + (self.totals_cast or 0)
        )

    def rows_cols_for_mode(self, mode: str) -> Tuple[Optional[int], Optional[int]]:
        if mode == "bp":
            return self.pim_rows_bp, self.pim_cols_bp
        if mode == "bs":
            return self.pim_rows_bs, self.pim_cols_bs
        if mode == "cpu":
            # CPU mode: no PIM resources, return 0 rows and 0 cols
            return 0, 0
        if mode == "hybrid":
            # Hybrid mode: sum rows and cols from both BP and BS
            # Only count modes where rows > 0 (indicating actual usage)
            total_rows = 0
            total_cols = 0
            
            # Check BP mode
            bp_rows = self.pim_rows_bp or 0
            bp_cols = self.pim_cols_bp or 0
            if bp_rows > 0:
                total_rows += bp_rows
                total_cols += bp_cols
            
            # Check BS mode
            bs_rows = self.pim_rows_bs or 0
            bs_cols = self.pim_cols_bs or 0
            if bs_rows > 0:
                total_rows += bs_rows
                total_cols += bs_cols
            
            return total_rows, total_cols
        return None, None

    def calculate_noc_traffic(self, mode: str) -> Dict[str, float]:
        """
        Calculate NoC traffic breakdown for this cost entry.
        
        Key insights:
        1. Each Load/Store operation transfers ONE ROW of data
        2. rows and cols in storage.pim are TOTAL across all k instances
        3. totals (cycles) already include all k instances
        4. Migration is based on TOTAL rows, not per-instance
        
        Args:
            mode: PIM mode ("bp", "bs", "hybrid", or "cpu")
        
        Returns:
            Dict with keys: load_bytes, store_bytes, cast_bytes, migration_bytes, total_bytes
        """
        rows, cols = self.rows_cols_for_mode(mode)
        rows = rows or 0
        cols = cols or 0
        bitwidth = self.bitwidth
        
        # rows and cols are ALREADY totals across all k instances
        # Each operation transfers 1 row of data
        bytes_per_row = cols * (bitwidth / 8.0)
        
        # Load traffic: CPU → PIM
        if mode == "bs":
            load_cost_per_op = bitwidth * 100000  # LoadCPUToBS
        else:
            load_cost_per_op = 100000  # LoadCPUToBP
        
        load_op_count = self.totals_load / load_cost_per_op if load_cost_per_op > 0 else 0
        load_bytes = load_op_count * bytes_per_row
        
        # Store traffic: PIM → CPU
        if mode == "bs":
            store_cost_per_op = bitwidth * 100000  # StoreBSToCPU
        else:
            store_cost_per_op = 100000  # StoreBPToCPU
        
        store_op_count = self.totals_store / store_cost_per_op if store_cost_per_op > 0 else 0
        store_bytes = store_op_count * bytes_per_row
        
        # Cast traffic: BP ↔ BS conversion (transfers entire array)
        cast_cost_per_op = (bitwidth + 2) * 100000  # CastBPtoBS / CastBStoBP
        cast_op_count = self.totals_cast / cast_cost_per_op if cast_cost_per_op > 0 else 0
        cast_bytes = cast_op_count * rows * bytes_per_row
        
        # Migration traffic: row overflow (rows > 64)
        # rows is ALREADY the total across all k instances
        # So we DON'T multiply by k again!
        migration_bytes = 0.0
        if rows > 64:
            extra_rows = rows - 64
            # bytes_per_row already uses total cols (across all k instances)
            migration_bytes = extra_rows * bytes_per_row
        
        return {
            "load_bytes": load_bytes,
            "store_bytes": store_bytes,
            "cast_bytes": cast_bytes,
            "migration_bytes": migration_bytes,
            "total_bytes": load_bytes + store_bytes + cast_bytes + migration_bytes,
        }


@dataclass(frozen=True)
class Bundle:
    layout: str
    parallelism: int
    entry: CostEntry
    mode: str  # "bp" or "bs" for PIM footprint; may be equal to layout for PIM layouts; "cpu" possible


@dataclass
class BundleUse:
    bundle: Bundle
    count: int  # number of times this bundle is used in a combination


# ---------------------------
# Scanning
# ---------------------------

def _safe_get(d: Dict, *keys, default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def analyze_unique_loads(extracted_file: Path) -> Tuple[int, int, float]:
    """
    Analyze the extracted egglog expression to count unique loads.
    
    Returns:
        (total_loads, unique_args, dedup_factor)
    """
    import re
    
    if not extracted_file.exists():
        return 0, 0, 1.0
    
    try:
        expr = extracted_file.read_text()
        
        # 提取所有的 CpuArg (这些会被转换为 LoadCPUToBP)
        cpu_args = re.findall(r'CpuArg (\d+)', expr)
        
        if not cpu_args:
            return 0, 0, 1.0
        
        unique_args = set(cpu_args)
        total_loads = len(cpu_args)
        unique_count = len(unique_args)
        
        # Deduplication factor: 实际需要的 loads / 当前的 loads
        # 理想情况下，每个 unique arg 只需要 load 一次
        dedup_factor = total_loads / unique_count if unique_count > 0 else 1.0
        
        return total_loads, unique_count, dedup_factor
    except Exception as e:
        return 0, 0, 1.0


def scan_costs(
    cost_root: Path, layouts: Sequence[str], workload: str, default_bitwidth: int
) -> Tuple[List[str], Dict[str, List[CostEntry]]]:
    statement_to_entries: Dict[str, List[CostEntry]] = {}
    found_statements: Set[str] = set()
    for layout in layouts:
        layout_dir = cost_root / layout / workload
        if not layout_dir.exists():
            continue
        for statement_dir in layout_dir.iterdir():
            if not statement_dir.is_dir():
                continue
            statement = statement_dir.name
            found_statements.add(statement)
            for pfile in statement_dir.glob("p*_cost.json"):
                try:
                    data = json.loads(pfile.read_text())
                except Exception:
                    continue
                totals = data.get("cost_breakdown", {}).get("totals", {}) or {}
                storage = data.get("storage", {}) or {}
                pim = storage.get("pim", {}) or {}
                bitwidth = int(storage.get("bitwidth") or default_bitwidth)
                parallelism = int(storage.get("parallelism") or 1)
                
                # Extract load count for deduplication
                counts = data.get("cost_breakdown", {}).get("counts", {})
                load_counts = counts.get("load", {})
                load_cpu_to_bp_count = load_counts.get("LoadCPUToBP", 0)
                load_cpu_to_bs_count = load_counts.get("LoadCPUToBS", 0)
                
                # Extract CPU cost and check if it contains WriteMemRefCPU
                cpu_cost = float(totals.get("cpu", 0))
                store_cost = float(totals.get("store", 0))
                load_cost = float(totals.get("load", 0))
                
                # Load Deduplication: 精确分析 egglog 表达式中的 unique arguments
                # 问题：Egglog没有做CSE，导致每次使用一个参数都重新load
                # 例如：CpuArg 2 被使用4次，但只需要load一次
                # 
                # 解决方案：分析 extracted.txt，统计 unique CpuArg，计算精确的 dedup factor
                original_load_cost = load_cost
                
                # 找到对应的 extracted.txt 文件
                extracted_file = pfile.parent / f"p{parallelism}_extracted.txt"
                total_loads, unique_args, dedup_factor = analyze_unique_loads(extracted_file)
                
                if dedup_factor > 1.0 and total_loads > 0:
                    # 精确的 deduplication：只保留 unique args 的 load
                    load_cost = load_cost / dedup_factor
                    print(f"  [Load Dedup] {statement}/{layout}/P{parallelism}: "
                          f"{total_loads} loads → {unique_args} unique args "
                          f"(factor: {dedup_factor:.2f}x, cost: {original_load_cost:,} → {load_cost:,.0f})")
                
                # Check if there's a WriteMemRefCPU operation in the counts
                cpu_counts = counts.get("cpu", {})
                writemem_count = cpu_counts.get("WriteMemRefCPU", 0)
                
                if writemem_count > 0:
                    # Get per-instance totals to calculate the actual WriteMemRefCPU cost
                    per_instance = data.get("cost_breakdown", {}).get("per_instance_totals", {})
                    per_instance_cpu = per_instance.get("cpu", 0)
                    
                    # The per_instance_cpu should be the WriteMemRefCPU cost for a single instance
                    # Total cost = per_instance_cpu * parallelism
                    writemem_total_cost = per_instance_cpu * parallelism
                    
                    # Verify this matches the total CPU cost
                    if abs(writemem_total_cost - cpu_cost) < 1:
                        # Move WriteMemRefCPU cost from CPU to store
                        cpu_cost = 0
                        store_cost = store_cost + writemem_total_cost
                        
                        # Print info for debugging
                        print(f"  [{statement}/{layout}/P{parallelism}] Moved WriteMemRefCPU cost: {writemem_total_cost:,} from CPU to store")
                    else:
                        # If there's a mismatch, use the calculated value but warn
                        print(f"  ⚠️  [{statement}/{layout}/P{parallelism}] CPU cost mismatch: expected {writemem_total_cost:,}, got {cpu_cost:,}")
                        cpu_cost = max(0, cpu_cost - writemem_total_cost)
                        store_cost = store_cost + writemem_total_cost
                
                entry = CostEntry(
                    statement=statement,
                    layout=layout,
                    parallelism=parallelism,
                    totals_cpu=cpu_cost,
                    totals_bp=float(totals.get("bp", 0)),
                    totals_bs=float(totals.get("bs", 0)),
                    totals_load=load_cost,  # Use deduplicated load cost
                    totals_store=store_cost,
                    totals_cast=float(totals.get("cast", 0)),
                    pim_rows_bp=_safe_get(pim, "bp", "rows", default=None),
                    pim_cols_bp=_safe_get(pim, "bp", "columns", default=None),
                    pim_rows_bs=_safe_get(pim, "bs", "rows", default=None),
                    pim_cols_bs=_safe_get(pim, "bs", "columns", default=None),
                    bitwidth=bitwidth,
                    cost_path=pfile,
                )
                statement_to_entries.setdefault(statement, []).append(entry)
    statements = sorted(found_statements)
    return statements, statement_to_entries


# ---------------------------
# Combination generation
# ---------------------------

def build_bundles_for_statement(entries: List[CostEntry]) -> List[Bundle]:
    bundles: List[Bundle] = []
    for e in entries:
        # Determine PIM mode based on layout
        # For BS layout, use BS mode; for BP, use BP mode
        # For Hybrid, use "hybrid" mode (will check both BP and BS)
        # For CPU, use "cpu" mode (no PIM resources needed)
        if e.layout == "bs":
            mode = "bs"
        elif e.layout == "bp":
            mode = "bp"
        elif e.layout == "hybrid":
            mode = "hybrid"
        elif e.layout == "cpu":
            mode = "cpu"
        else:  # other unknown layouts
            # Fallback: prefer bp if available, else bs
            mode = "bp" if e.pim_cols_bp is not None else "bs"
        bundles.append(Bundle(layout=e.layout, parallelism=e.parallelism, entry=e, mode=mode))
    # Sort by parallelism descending for stable enumeration
    bundles.sort(key=lambda b: b.parallelism, reverse=True)
    return bundles


def enumerate_combinations_sum_to_P(
    bundles: List[Bundle], target_p: int
) -> List[List[Bundle]]:
    # Fallback naive enumerator (kept for reference; replaced by DP-based top-K in retain_top2_per_statement).
    results: List[List[Bundle]] = []
    def rec(start_idx: int, remaining: int, current: List[Bundle]) -> None:
        if remaining == 0:
            results.append(list(current))
            return
        if remaining < 0:
            return
        for i in range(start_idx, len(bundles)):
            b = bundles[i]
            if b.parallelism > remaining:
                continue
            current.append(b)
            rec(i, remaining - b.parallelism, current)
            current.pop()
    rec(0, target_p, [])
    return results


# ---------------------------
# Scoring and pruning (Top-2)
# ---------------------------

def score_combination_proxy(
    combo: List[Bundle],
    cpu_cores: int,
    arrays_total: int,
    fifo_bw_bytes_per_cycle: int,
) -> Tuple[float, Dict]:
    total_score = 0.0
    arrays_penalty = 0
    migration_penalty = 0.0

    for b in combo:
        k = b.parallelism
        e = b.entry
        # Note: totals in p*_cost.json are already scaled by parallelism
        total_cpu_work = e.total_cpu_work
        total_pim_work = e.total_pim_work
        # CPU time: total work divided by available cores (capped by parallelism)
        t_cpu = 0.0
        if total_cpu_work > 0:
            effective_cores = min(k, cpu_cores)
            t_cpu = total_cpu_work / max(1, effective_cores)
        # PIM time: total work (already accounts for all instances)
        t_pim = total_pim_work
        # Row overflow migration proxy
        rows, cols = e.rows_cols_for_mode(b.mode)
        rows = rows or 0
        cols = cols or 0
        bw = fifo_bw_bytes_per_cycle or 32
        bitwidth = max(1, int(e.bitwidth))
        mig = 0.0
        if rows > 64:
            extra_rows = rows - 64
            # Migration cost for all k instances
            bytes_total = extra_rows * cols * (bitwidth / 8.0) * k
            mig = max(bytes_total / bw, float(extra_rows * k))
        # Column pressure proxy: waves if per-instance columns exceed capacity
        # Note: capacity uses arrays_total * 512 columns in the whole system
        column_wave_penalty = 0
        if arrays_total > 0:
            system_cols = arrays_total * 512
            if cols > 0:
                column_wave_penalty = math.ceil(cols / system_cols)
        total_score += t_cpu + t_pim + mig + column_wave_penalty
        arrays_penalty += column_wave_penalty
        migration_penalty += mig

    diagnostics = {
        "arrays_penalty": arrays_penalty,
        "migration_penalty": migration_penalty,
    }
    return total_score, diagnostics


def retain_topk_per_statement(
    bundles: List[Bundle],
    target_p: int,
    cpu_cores: int,
    arrays_total: int,
    fifo_bw_bytes_per_cycle: int,
    topk: int,
) -> List[List[Bundle]]:
    # Dynamic programming to track top-K combos per sum; uses linear per-bundle proxy score.
    # Precompute per-bundle base score
    per_bundle_score: List[float] = []
    for b in bundles:
        s, _ = score_combination_proxy([b], cpu_cores, arrays_total, fifo_bw_bytes_per_cycle)
        per_bundle_score.append(s)
    K = max(1, int(topk))
    # dp[s] = list of up to K entries: (score, combo_signature_tuple, combo_bundles_list)
    dp: List[List[Tuple[float, Tuple[int, ...], List[Bundle]]]] = [[] for _ in range(target_p + 1)]
    # signature is counts per bundle index; for memory reduce, we store tuple of counts
    zero_sig = tuple([0] * len(bundles))
    dp[0] = [(0.0, zero_sig, [])]
    for s in range(target_p + 1):
        if not dp[s]:
            continue
        for idx, b in enumerate(bundles):
            nxt = s + b.parallelism
            if nxt > target_p:
                continue
            for cur_score, cur_sig, cur_list in dp[s]:
                new_sig = list(cur_sig)
                new_sig[idx] += 1
                new_score = cur_score + per_bundle_score[idx]
                new_list = cur_list + [b]
                # Insert into dp[nxt] maintaining top-K by (score, len(list))
                inserted = False
                # Avoid duplicates by signature
                existing_sigs = {sig for _, sig, _ in dp[nxt]}
                new_sig_t = tuple(new_sig)
                if new_sig_t in existing_sigs:
                    continue
                candidate = (new_score, new_sig_t, new_list)
                dp[nxt].append(candidate)
                dp[nxt].sort(key=lambda x: (x[0], len(x[2])))
                if len(dp[nxt]) > K:
                    dp[nxt] = dp[nxt][:K]
    retained: List[List[Bundle]] = [entry[2] for entry in dp[target_p]]
    return retained


# ---------------------------
# Global assignment simulation
# ---------------------------

@dataclass
class StatementDecision:
    statement: str
    combo: List[Bundle]


@dataclass
class SimBundleResult:
    layout: str
    p: int
    rows: Optional[int]
    cols: Optional[int]
    arrays_needed: int
    rounds: int
    mig_bytes: float
    t_cpu: float
    t_pim: float
    t_mig: float
    cost_path: str
    noc_load_bytes: float = 0.0
    noc_store_bytes: float = 0.0
    noc_cast_bytes: float = 0.0
    noc_migration_bytes: float = 0.0
    noc_total_bytes: float = 0.0


def _simulate_cpu_timeline(
    cpu_tasks: List[Tuple[str, int, float, int]]
) -> Tuple[float, Dict[Tuple[str, int], float]]:
    """
    cpu_tasks: list of (statement, task_id, work_units, core_cap)
        work_units = c_cpu * parallelism
        core_cap = parallelism (max cores usable)
    Returns:
        total_time, per_task_finish_time keyed by (statement, task_id)
    """
    if not cpu_tasks:
        return 0.0, {}
    time_now = 0.0
    # Remaining work per task
    remaining: Dict[Tuple[str, int], float] = {}
    caps: Dict[Tuple[str, int], int] = {}
    for stmt, tid, work, cap in cpu_tasks:
        key = (stmt, tid)
        remaining[key] = float(work)
        caps[key] = int(max(1, cap))
    finished: Dict[Tuple[str, int], float] = {}
    active: Set[Tuple[str, int]] = set(remaining.keys())
    while active:
        # Allocate cores fairly up to cap, across active tasks
        # First pass: give 1 core to each active up to caps and total cores
        allocations: Dict[Tuple[str, int], float] = {k: 0.0 for k in active}
        cores_left = cpu_cores_global  # will be set by outer scope
        # Greedy round-robin up to caps
        while cores_left > 0:
            progressed = False
            for k in list(active):
                if allocations[k] < caps[k]:
                    allocations[k] += 1.0
                    cores_left -= 1
                    progressed = True
                    if cores_left == 0:
                        break
            if not progressed:
                break
        # If no cores were allocated (shouldn't happen), break to avoid div0
        if sum(allocations.values()) <= 0:
            break
        # Compute time to next completion
        deltas: List[float] = []
        for k in active:
            a = allocations[k]
            if a <= 0:
                continue
            deltas.append(remaining[k] / a)
        if not deltas:
            break
        delta = min(deltas)
        time_now += delta
        # Advance, mark finished
        to_remove: List[Tuple[str, int]] = []
        for k in active:
            a = allocations[k]
            if a <= 0:
                continue
            remaining[k] -= a * delta
            if remaining[k] <= 1e-9:
                finished[k] = time_now
                to_remove.append(k)
        for k in to_remove:
            active.remove(k)
    return time_now, finished


def simulate_statement_block(
    decision: StatementDecision,
    cpu_cores: int,
    arrays_total: int,
    fifo_bw_bytes_per_cycle: int,
    fifo_cursor: float,
) -> Tuple[float, float, List[SimBundleResult], Dict[str, float]]:
    """
    Returns: (block_time, new_fifo_cursor, bundle_results, penalties)
    """
    t_cpu_block = 0.0
    t_pim_block = 0.0
    t_mig_block = 0.0
    fifo_length_this_block = 0.0
    bundle_results: List[SimBundleResult] = []
    arrays_total = max(1, arrays_total)
    bw = fifo_bw_bytes_per_cycle or 32

    arrays_penalty_rounds = 0
    mig_penalty_time = 0.0

    for b in decision.combo:
        e = b.entry
        k = b.parallelism
        # Note: totals in p*_cost.json are already scaled by parallelism
        total_cpu_work = e.total_cpu_work
        total_pim_work = e.total_pim_work

        t_cpu = 0.0
        if total_cpu_work > 0:
            effective_cores = min(k, cpu_cores)
            t_cpu = total_cpu_work / max(1, effective_cores)
        t_pim = total_pim_work

        rows, cols = e.rows_cols_for_mode(b.mode)
        rows = rows or 0
        cols = cols or 0
        instances_per_array = max(1, 512 // max(1, cols)) if cols > 0 else 512
        arrays_needed = math.ceil(k / instances_per_array) if k > 0 else 0
        rounds = math.ceil(arrays_needed / arrays_total) if arrays_total > 0 else 1
        t_pim_eff = t_pim * max(1, rounds)

        mig_bytes = 0.0
        t_mig = 0.0
        if rows > 64:
            extra_rows = rows - 64
            # rows and cols are already totals, don't multiply by k
            mig_bytes = extra_rows * cols * (e.bitwidth / 8.0)
            t_mig = max(mig_bytes / bw, extra_rows * 1.0)
            fifo_length_this_block += mig_bytes / bw
            mig_penalty_time += t_mig

        arrays_penalty_rounds = max(arrays_penalty_rounds, rounds)

        # Calculate NoC traffic for this bundle
        noc_traffic = e.calculate_noc_traffic(b.mode)

        t_cpu_block += t_cpu
        t_pim_block += t_pim_eff
        t_mig_block += t_mig

        bundle_results.append(
            SimBundleResult(
                layout=b.layout,
                p=k,
                rows=rows,
                cols=cols,
                arrays_needed=arrays_needed,
                rounds=rounds,
                mig_bytes=mig_bytes,
                t_cpu=t_cpu,
                t_pim=t_pim_eff,
                t_mig=t_mig,
                cost_path=str(e.cost_path),
                noc_load_bytes=noc_traffic["load_bytes"],
                noc_store_bytes=noc_traffic["store_bytes"],
                noc_cast_bytes=noc_traffic["cast_bytes"],
                noc_migration_bytes=noc_traffic["migration_bytes"],
                noc_total_bytes=noc_traffic["total_bytes"],
            )
        )

    fifo_wait_increment = fifo_cursor  # tasks of this block must wait existing FIFO length
    new_fifo_cursor = fifo_cursor + fifo_length_this_block
    t_block = t_cpu_block + t_pim_block + t_mig_block + fifo_wait_increment
    penalties = {
        "cpu_time": t_cpu_block,
        "pim_time": t_pim_block,
        "migration_time": t_mig_block,
        "fifo_wait": fifo_wait_increment,
        "arrays_rounds": float(arrays_penalty_rounds),
    }
    return t_block, new_fifo_cursor, bundle_results, penalties


def simulate_group_block(
    group: List[str],
    selection_by_stmt: Dict[str, List[Bundle]],
    cpu_cores: int,
    arrays_total: int,
    fifo_bw_bytes_per_cycle: int,
    fifo_cursor: float,
) -> Tuple[float, float, Dict[str, List[SimBundleResult]], List[Tuple[float, int, float]], float]:
    """Simulate one parallel group: CPU cores are shared across statements in the group.
    PIM/migration still use per-bundle estimates and are summed (no array sharing yet).
    Returns (group_duration, new_fifo_cursor, per_stmt_results, pim_timeline, avg_utilization).
    
    pim_timeline: List of (time, columns_used, utilization_percent)
    avg_utilization: Time-weighted average PIM utilization percentage
    """
    # Prepare CPU tasks
    global cpu_cores_global  # used by _simulate_cpu_timeline
    cpu_cores_global = cpu_cores
    cpu_tasks: List[Tuple[str, int, float, int]] = []
    bundle_index_map: Dict[Tuple[str, int], Tuple[str, int]] = {}
    # Precompute per-bundle PIM/migration and collect CPU work units
    per_stmt_results: Dict[str, List[SimBundleResult]] = {}
    pim_sum = 0.0
    mig_sum = 0.0
    fifo_add = 0.0

    tmp_results: Dict[Tuple[str, int], SimBundleResult] = {}
    # First pass: compute per-bundle arrays_needed and collect totals for group sharing
    total_arrays_needed = 0
    for stmt in group:
        bundles = selection_by_stmt.get(stmt, [])
        idx = 0
        for b in bundles:
            e = b.entry
            k = b.parallelism
            # Note: totals in p*_cost.json are already scaled by parallelism
            total_cpu_work = e.total_cpu_work
            total_pim_work = e.total_pim_work
            rows, cols = e.rows_cols_for_mode(b.mode)
            rows = rows or 0
            cols = cols or 0
            instances_per_array = max(1, 512 // max(1, cols)) if cols > 0 else 512
            arrays_needed = math.ceil(k / instances_per_array) if k > 0 else 0
            total_arrays_needed += arrays_needed
            # Placeholder rounds; will be overwritten after group rounds computed
            rounds = 1
            t_pim_eff = total_pim_work

            mig_bytes = 0.0
            t_mig = 0.0
            if rows > 64:
                extra_rows = rows - 64
                # rows and cols are already totals, don't multiply by k
                mig_bytes = extra_rows * cols * (e.bitwidth / 8.0)
                t_mig = max(mig_bytes / max(1, fifo_bw_bytes_per_cycle), extra_rows * 1.0)
                fifo_add += mig_bytes / max(1, fifo_bw_bytes_per_cycle)

            # Calculate NoC traffic for this bundle
            noc_traffic = e.calculate_noc_traffic(b.mode)

            key = (stmt, idx)
            tmp = SimBundleResult(
                layout=b.layout,
                p=k,
                rows=rows,
                cols=cols,
                arrays_needed=arrays_needed,
                rounds=rounds,
                mig_bytes=mig_bytes,
                t_cpu=0.0,
                t_pim=t_pim_eff,
                t_mig=t_mig,
                cost_path=str(e.cost_path),
                noc_load_bytes=noc_traffic["load_bytes"],
                noc_store_bytes=noc_traffic["store_bytes"],
                noc_cast_bytes=noc_traffic["cast_bytes"],
                noc_migration_bytes=noc_traffic["migration_bytes"],
                noc_total_bytes=noc_traffic["total_bytes"],
            )
            tmp_results[key] = tmp
            if total_cpu_work > 0 and k > 0:
                # work_units is already total work across all instances
                work_units = total_cpu_work
                core_cap = k
                cpu_tasks.append((stmt, idx, work_units, core_cap))
            mig_sum += t_mig
            idx += 1

    # Build per-instance list for PIM timeline with mixed columns
    # Note: total_pim_work is already scaled by parallelism, so divide by k to get per-instance work
    instances: List[Tuple[int, float, Tuple[str, int]]] = []
    for stmt in group:
        bundles = selection_by_stmt.get(stmt, [])
        for idx, b in enumerate(bundles):
            e = b.entry
            k = max(1, b.parallelism)
            # Skip PIM timeline for CPU-only bundles
            if e.total_pim_work <= 0:
                continue
            cols_total = (e.rows_cols_for_mode(b.mode)[1] or 0)
            per_inst_cols = int(math.ceil(cols_total / k)) if cols_total > 0 else 0
            per_inst_cols = max(1, per_inst_cols)
            # Divide total work by k to get per-instance work
            t_unit = e.total_pim_work / k if k > 0 else 0.0
            for _ in range(k):
                instances.append((per_inst_cols, t_unit, (stmt, idx)))

    # Initialize arrays pool with heaps for O(log N) placement
    import heapq  # local import to avoid top clutter
    INF = float("inf")

    class _ArrayState:
        __slots__ = ("free_cols", "events", "gen")
        def __init__(self) -> None:
            self.free_cols = 512
            self.events: List[Tuple[float, int]] = []  # (finish_time, cols_used)
            self.gen = 0

    arrays: List[_ArrayState] = [_ArrayState() for _ in range(max(1, arrays_total))]
    # Heaps: free_heap by -free_cols; event_heap by next_finish_time
    free_heap: List[Tuple[int, int, int]] = []  # (-free_cols, gen, idx)
    event_heap: List[Tuple[float, int, int]] = []  # (next_finish, gen, idx)
    for i, a in enumerate(arrays):
        heapq.heappush(free_heap, (-a.free_cols, a.gen, i))
        next_ft = a.events[0][0] if a.events else INF
        heapq.heappush(event_heap, (next_ft, a.gen, i))

    def _refresh_free(i: int) -> None:
        a = arrays[i]
        heapq.heappush(free_heap, (-a.free_cols, a.gen, i))

    def _refresh_event(i: int) -> None:
        a = arrays[i]
        next_ft = a.events[0][0] if a.events else INF
        heapq.heappush(event_heap, (next_ft, a.gen, i))

    def _pop_valid_free() -> int:
        while free_heap:
            neg_free, gen, i = free_heap[0]
            if gen != arrays[i].gen or -neg_free != arrays[i].free_cols:
                heapq.heappop(free_heap)
                continue
            return i
        return 0

    def _pop_valid_event() -> int:
        while event_heap:
            ft, gen, i = event_heap[0]
            current_ft = arrays[i].events[0][0] if arrays[i].events else INF
            if gen != arrays[i].gen or ft != current_ft:
                heapq.heappop(event_heap)
                continue
            return i
        return 0

    def _ensure_fit(i: int, w: int) -> float:
        a = arrays[i]
        time_cursor = 0.0
        while a.free_cols < w:
            if not a.events:
                break
            ft, cols = heapq.heappop(a.events)
            time_cursor = ft
            a.free_cols += cols
        if time_cursor > 0.0:
            a.gen += 1
            _refresh_free(i)
            _refresh_event(i)
        return time_cursor

    def _place(i: int, w: int, dur: float) -> Tuple[float, float]:
        """Place an instance on array i. Returns (start_time, finish_time)."""
        a = arrays[i]
        start_offset = 0.0
        if a.free_cols < w:
            start_offset = _ensure_fit(i, w)
        finish = start_offset + dur
        a.free_cols -= w
        heapq.heappush(a.events, (finish, w))
        a.gen += 1
        _refresh_free(i)
        _refresh_event(i)
        return start_offset, finish

    # Sort instances by columns desc (FFD)
    instances.sort(key=lambda x: x[0], reverse=True)
    bundle_to_finish: Dict[Tuple[str, int], float] = {}
    
    # Track PIM utilization timeline: list of (time, columns_used)
    pim_utilization_events: List[Tuple[float, int, str]] = []  # (time, delta_cols, event_type)
    
    debug_place_count = 0
    for w, dur, key in instances:
        # Try best free array first
        i_free = _pop_valid_free()
        if arrays[i_free].free_cols >= w:
            start_time, finish = _place(i_free, w, dur)
            array_used = i_free
        else:
            # No array can fit now; pick the array that frees earliest and place there
            i_ev = _pop_valid_event()
            start_time, finish = _place(i_ev, w, dur)
            array_used = i_ev
        
        # Debug: print first few placements
        if False and debug_place_count < 20:  # Disabled
            print(f"DEBUG place {debug_place_count}: array={array_used}, w={w}, dur={dur}, start={start_time}, finish={finish}, free_cols_after={arrays[array_used].free_cols}")
            debug_place_count += 1
        
        # Record utilization events
        pim_utilization_events.append((start_time, w, "start"))
        pim_utilization_events.append((finish, -w, "end"))
        
        prev = bundle_to_finish.get(key, 0.0)
        if finish > prev:
            bundle_to_finish[key] = finish
    
    # Set per-bundle t_pim as its max finish; group PIM time as max across bundles
    pim_group_time = 0.0
    for (stmt, idx), res in tmp_results.items():
        t_fin = bundle_to_finish.get((stmt, idx), 0.0)
        res.t_pim = t_fin
        res.rounds = 1
        if t_fin > pim_group_time:
            pim_group_time = t_fin

    # Run CPU timeline for the group
    cpu_time_group, finishes = _simulate_cpu_timeline(cpu_tasks)
    # Debug: print CPU timeline results
    if False:  # Set to True for debugging
        print(f"DEBUG: cpu_time_group = {cpu_time_group:,.0f}")
        print(f"DEBUG: len(finishes) = {len(finishes)}")
        print(f"DEBUG: len(cpu_tasks) = {len(cpu_tasks)}")
        if len(finishes) > 0:
            first_key = list(finishes.keys())[0]
            print(f"DEBUG: first finish: {first_key} -> {finishes[first_key]:,.0f}")
    # Attribute per-bundle t_cpu by finished time (time from group start)
    for (stmt, idx), t_finish in finishes.items():
        if (stmt, idx) in tmp_results:
            tmp_results[(stmt, idx)].t_cpu = t_finish
            # For CPU-only bundles, also set t_pim to 0 (it may have been set incorrectly)
            if tmp_results[(stmt, idx)].t_pim == 0 or tmp_results[(stmt, idx)].layout == "cpu":
                tmp_results[(stmt, idx)].t_pim = 0

    # Compose per-statement lists
    for (stmt, idx), res in tmp_results.items():
        per_stmt_results.setdefault(stmt, []).append(res)

    # Calculate PIM utilization timeline from events
    # Sort events by time
    pim_utilization_events.sort(key=lambda x: x[0])
    
    # Debug: print event count
    if False:  # Disabled
        print(f"DEBUG pim_utilization_events: {len(pim_utilization_events)} events")
        # Print first 20 events
        for i, (t, delta, etype) in enumerate(pim_utilization_events[:20]):
            print(f"  {i}: t={t}, delta={delta}, type={etype}")
    
    # Build timeline: list of (time, columns_used, utilization_percent)
    pim_timeline: List[Tuple[float, int, float]] = []
    current_cols = 0
    max_cols = arrays_total * 512  # Total available columns (512 columns per array)
    
    for time, delta_cols, event_type in pim_utilization_events:
        current_cols += delta_cols
        utilization = (current_cols / max_cols * 100) if max_cols > 0 else 0
        pim_timeline.append((time, current_cols, utilization))
    
    # Calculate average utilization
    if pim_timeline and len(pim_timeline) > 1:
        # Merge events at the same time point and integrate utilization over time
        # First, consolidate events at the same time (keep the last state at each time)
        consolidated = []
        i = 0
        while i < len(pim_timeline):
            time_point = pim_timeline[i][0]
            # Find all events at this time point
            j = i
            while j < len(pim_timeline) and pim_timeline[j][0] == time_point:
                j += 1
            # Keep the last state at this time point
            consolidated.append(pim_timeline[j - 1])
            i = j
        
        # Now integrate utilization over time
        total_util_time = 0.0
        for i in range(len(consolidated) - 1):
            time_start = consolidated[i][0]
            time_end = consolidated[i + 1][0]
            util = consolidated[i][2]
            duration = time_end - time_start
            if duration > 0:
                total_util_time += duration * util
        
        # Use the actual timeline span as denominator
        timeline_span = consolidated[-1][0] - consolidated[0][0]
        avg_utilization = total_util_time / timeline_span if timeline_span > 0 else 0
        
        # Replace pim_timeline with consolidated for better sampling
        pim_timeline = consolidated
    else:
        avg_utilization = 0.0

    fifo_wait_increment = fifo_cursor
    # Group duration: overlap CPU and PIM compute; add migration and FIFO wait
    group_duration = max(cpu_time_group, pim_group_time) + mig_sum + fifo_wait_increment
    new_fifo_cursor = fifo_cursor + fifo_add
    
    # Store PIM timeline in per_stmt_results metadata (attach to first statement)
    # We'll pass this through a different mechanism
    return group_duration, new_fifo_cursor, per_stmt_results, pim_timeline, avg_utilization


# ---------------------------
# Output rendering
# ---------------------------

def write_schedule_json(
    output_path: Path,
    workload: str,
    target_p: int,
    cpu_cores: int,
    arrays_total: int,
    fifo_bw: int,
    topo: List[str],
    decisions: Dict[str, List[SimBundleResult]],
    starts_finishes: Dict[str, Tuple[float, float]],
    total_duration: float,
    fifo_total_wait: float,
    pim_timelines: List[Tuple[int, List[Tuple[float, int, float]], float]] = None,
) -> None:
    # Statements starting at the same time belong to the same parallel group.
    start_to_group: Dict[float, int] = {}
    next_group_id = 0
    kernels = []
    for stmt in topo:
        if stmt not in decisions:
            continue
        start, finish = starts_finishes.get(stmt, (0.0, 0.0))
        if start not in start_to_group:
            start_to_group[start] = next_group_id
            next_group_id += 1
        group_id = start_to_group[start]
        bundles = [
            {
                "layout": br.layout,
                "parallelism": br.p,
                "rows": br.rows,
                "columns": br.cols,
                "arrays_needed": br.arrays_needed,
                "rounds": br.rounds,
                "mig_bytes": br.mig_bytes,
                "t_cpu": br.t_cpu,
                "t_pim": br.t_pim,
                "t_mig": br.t_mig,
                "cost_path": br.cost_path,
                "noc_traffic": {
                    "load_bytes": br.noc_load_bytes,
                    "store_bytes": br.noc_store_bytes,
                    "cast_bytes": br.noc_cast_bytes,
                    "migration_bytes": br.noc_migration_bytes,
                    "total_bytes": br.noc_total_bytes,
                },
            }
            for br in decisions[stmt]
        ]
        kernels.append(
            {
                "name": stmt,
                "group_id": group_id,
                "bundles": bundles,
                "start": start,
                "finish": finish,
            }
        )
    # Calculate total NoC traffic across all kernels
    total_noc_load = 0.0
    total_noc_store = 0.0
    total_noc_cast = 0.0
    total_noc_migration = 0.0
    for stmt in topo:
        if stmt not in decisions:
            continue
        for br in decisions[stmt]:
            total_noc_load += br.noc_load_bytes
            total_noc_store += br.noc_store_bytes
            total_noc_cast += br.noc_cast_bytes
            total_noc_migration += br.noc_migration_bytes
    
    total_noc_traffic = total_noc_load + total_noc_store + total_noc_cast + total_noc_migration
    
    result = {
        "workload": workload,
        "target_P": target_p,
        "cpu_cores": cpu_cores,
        "arrays_total": arrays_total,
        "fifo_bw_bytes_per_cycle": fifo_bw,
        "generated_at": _dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
        "total_duration": total_duration,
        "total_cost": total_duration * max(1, target_p),
        "kernels": kernels,
        "bandwidth_fifo": {
            "total_wait": fifo_total_wait,
        },
        "noc_traffic": {
            "total_bytes": total_noc_traffic,
            "load_bytes": total_noc_load,
            "store_bytes": total_noc_store,
            "cast_bytes": total_noc_cast,
            "migration_bytes": total_noc_migration,
        },
    }
    
    # Add PIM utilization information if available
    if pim_timelines:
        pim_util_data = []
        total_avg_util = 0.0
        for group_id, timeline, avg_util in pim_timelines:
            # Sample timeline at key points (to avoid huge JSON files)
            sampled_timeline = []
            if timeline:
                # Always include first and last points
                sampled_timeline.append({
                    "time": timeline[0][0],
                    "columns_used": timeline[0][1],
                    "utilization_percent": round(timeline[0][2], 2)
                })
                # Sample middle points (max 100 points per group)
                step = max(1, len(timeline) // 100)
                for i in range(step, len(timeline) - 1, step):
                    sampled_timeline.append({
                        "time": timeline[i][0],
                        "columns_used": timeline[i][1],
                        "utilization_percent": round(timeline[i][2], 2)
                    })
                if len(timeline) > 1:
                    sampled_timeline.append({
                        "time": timeline[-1][0],
                        "columns_used": timeline[-1][1],
                        "utilization_percent": round(timeline[-1][2], 2)
                    })
            
            pim_util_data.append({
                "group_id": group_id,
                "avg_utilization_percent": round(avg_util, 2),
                "timeline": sampled_timeline
            })
            total_avg_util += avg_util
        
        overall_avg_util = total_avg_util / len(pim_timelines) if pim_timelines else 0.0
        result["pim_utilization"] = {
            "overall_avg_percent": round(overall_avg_util, 2),
            "per_group": pim_util_data
        }
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True))


def write_schedule_md(output_path: Path, workload: str, total_duration: float, target_p: int, topo: List[str], starts_finishes: Dict[str, Tuple[float, float]]) -> None:
    lines: List[str] = []
    lines.append(f"# Schedule for {workload}")
    lines.append("")
    lines.append(f"- Total duration: {total_duration:.3f}")
    lines.append(f"- Total cost (duration × P): {total_duration * max(1, target_p):.3f}")
    lines.append("")
    lines.append("## Timeline")
    for stmt in topo:
        s, f = starts_finishes.get(stmt, (0.0, 0.0))
        lines.append(f"- {stmt}: start={s:.3f}, finish={f:.3f}, duration={f - s:.3f}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines))


# ---------------------------
# Main flow
# ---------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Advanced bundle-combination scheduler.")
    parser.add_argument("--workload", type=str, required=True, help="Workload name (e.g., gemm).")
    parser.add_argument(
        "--layouts",
        type=str,
        default=None,
        help="Comma-separated list among {hybrid,bp,bs,cpu}. Default: all present under cost-root.",
    )
    parser.add_argument(
        "--cost-root",
        type=Path,
        default=Path("out/kernel_costs"),
        help="Root of cost artifacts.",
    )
    parser.add_argument(
        "--stage-mlir",
        type=Path,
        default=None,
        help="Optional stage MLIR to infer dependencies.",
    )
    parser.add_argument(
        "--target-p",
        type=str,
        default=None,
        help="Comma-separated P values (e.g., 1,2,4,...). If omitted, infer intersection across statements.",
    )
    parser.add_argument("--cpu-cores", type=int, default=64)
    parser.add_argument("--arrays-per-slice", type=int, default=64)
    parser.add_argument("--num-slices", type=int, default=8)
    parser.add_argument("--fifo-bytes-per-cycle", type=int, default=32)
    parser.add_argument("--default-bitwidth", type=int, default=32)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: out/kernel_costs/advanced/<workload>/).",
    )
    parser.add_argument(
        "--show-progress",
        action="store_true",
        help="Print progress while evaluating global combinations.",
    )
    parser.add_argument(
        "--use-tqdm",
        action="store_true",
        help="Use tqdm progress bar if available (falls back to print-based progress).",
    )
    parser.add_argument(
        "--per-statement-topk",
        type=int,
        default=1,
        help="Retain top-K bundle combinations per statement before global search (default: 1).",
    )
    parser.add_argument(
        "--per-layout-top1",
        action="store_true",
        help="For each statement, keep exactly one best combo per layout (no cross-layout mixing within a statement).",
    )
    parser.add_argument(
        "--force-sequential",
        action="store_true",
        help="Force sequential execution by constructing chain dependencies (S0 -> S1 -> S2 -> ...).",
    )
    parser.add_argument(
        "--progress-interval",
        type=int,
        default=100,
        help="Progress print interval in number of combinations (default: 100).",
    )
    args = parser.parse_args()

    if args.layouts is None:
        # discover available layouts under cost-root
        possible = ["hybrid", "bp", "bs", "cpu"]
        layouts = [l for l in possible if (args.cost_root / l / args.workload).exists()]
    else:
        layouts = [s.strip() for s in args.layouts.split(",") if s.strip()]
    if not layouts:
        raise SystemExit("No layouts found or specified.")

    # Scan inputs
    statements, statement_entries = scan_costs(
        args.cost_root, layouts, args.workload, args.default_bitwidth
    )
    if not statements:
        raise SystemExit("No statements discovered.")

    # Determine P candidates
    if args.target_p:
        p_candidates = [int(x) for x in args.target_p.split(",") if x.strip()]
    else:
        # intersection of available p values across statements
        per_stmt_ps: List[Set[int]] = []
        for s in statements:
            ps = {e.parallelism for e in statement_entries.get(s, [])}
            per_stmt_ps.append(ps)
        if not per_stmt_ps:
            raise SystemExit("No parallelism values available.")
        intersection = set.intersection(*per_stmt_ps) if per_stmt_ps else set()
        if not intersection:
            # fallback: union, sorted
            intersection = set.union(*per_stmt_ps)
        inferred = sorted(intersection)
        # Limit default evaluation set to reduce runtime: choose up to 8 largest values plus 1 if present
        if len(inferred) > 9:
            largest = sorted(inferred)[-8:]
            if 1 in inferred and 1 not in largest:
                p_candidates = [1] + largest
            else:
                p_candidates = largest
        else:
            p_candidates = inferred

    # Dependencies and parallel groups
    stage_info: Dict[str, Dict[str, List[Tuple[str, str]]]] = {}
    if args.stage_mlir:
        if args.stage_mlir.exists():
            stage_info = parse_stage_functions(args.stage_mlir)
        else:
            raise FileNotFoundError(f"stage MLIR not found: {args.stage_mlir}")
    
    # Build dependencies
    if args.force_sequential:
        # Force sequential execution by constructing chain dependencies
        # S0 -> S1 -> S2 -> ... -> S(n-1)
        deps = {statements[0]: set()}
        for i in range(1, len(statements)):
            deps[statements[i]] = {statements[i-1]}
        print("Force-sequential mode: constructed chain dependencies")
    else:
        # Normal dependency analysis
        deps = build_dependencies(statements, stage_info)
    
    topo = topo_order(deps)
    groups = find_parallel_groups(topo, deps)

    arrays_total = args.num_slices * args.arrays_per_slice
    fifo_bw = args.fifo_bytes_per_cycle

    best_total = float("inf")
    best_output: Optional[Dict] = None
    per_p_outputs: Dict[int, Dict] = {}

    for target_p in p_candidates:
        # Build top-2 combos per statement
        per_stmt_top2: Dict[str, List[List[Bundle]]] = {}
        feasible = True
        for s in topo:
            bundles = build_bundles_for_statement(statement_entries.get(s, []))
            if not bundles:
                feasible = False
                break
            if args.per_layout_top1:
                # For each layout present for this statement, keep best combo that uses only that layout
                layout_names = sorted({b.layout for b in bundles})
                per_layout_candidates: List[List[Bundle]] = []
                for lname in layout_names:
                    filtered = [b for b in bundles if b.layout == lname]
                    if not filtered:
                        continue
                    best_one = retain_topk_per_statement(
                        filtered,
                        target_p,
                        args.cpu_cores,
                        arrays_total,
                        fifo_bw,
                        1,
                    )
                    if best_one:
                        per_layout_candidates.append(best_one[0])
                topk = per_layout_candidates
            else:
                topk = retain_topk_per_statement(
                    bundles,
                    target_p,
                    args.cpu_cores,
                    arrays_total,
                    fifo_bw,
                    args.per_statement_topk,
                )
            if not topk:
                feasible = False
                break
            per_stmt_top2[s] = topk
        if not feasible:
            continue

        # Global product
        choices_lists = [per_stmt_top2[s] for s in topo]
        fifo_cursor = 0.0
        current_time = 0.0
        starts_finishes: Dict[str, Tuple[float, float]] = {}
        per_stmt_bundle_results: Dict[str, List[SimBundleResult]] = {}
        fifo_total_wait_accum = 0.0

        # Enumerate all choices; evaluate group-by-group with CPU timeline sharing
        # Progress setup
        total_combos = 1
        for lst in choices_lists:
            total_combos *= max(1, len(lst))
        combo_counter = 0
        t_start = time.perf_counter()
        last_print = t_start
        # Optional tqdm progress bar
        pbar = None
        if args.show_progress and args.use_tqdm:
            try:
                from tqdm import tqdm as _tqdm  # type: ignore
                pbar = _tqdm(total=total_combos, desc=f"P={target_p}")
            except Exception:
                pbar = None
        # Track best for this specific P
        best_cost_p = float("inf")
        best_output_p: Optional[Dict] = None
        for choice in itertools.product(*choices_lists):
            combo_counter += 1
            if pbar is not None:
                pbar.update(1)
            elif args.show_progress:
                now = time.perf_counter()
                if (combo_counter % max(1, args.progress_interval) == 0) or (combo_counter == total_combos) or (now - last_print >= 0.5):
                    elapsed = now - t_start
                    rate = combo_counter / elapsed if elapsed > 0 else 0.0
                    remaining = (total_combos - combo_counter) / rate if rate > 0 else float("inf")
                    pct = combo_counter * 100.0 / max(1, total_combos)
                    print(f"[P={target_p}] {combo_counter}/{total_combos} ({pct:.2f}%), "
                          f"elapsed {elapsed:.3f}s, rate {rate:.1f}/s, eta {remaining:.3f}s")
                    last_print = now
            # evaluate this combination
            fifo_cursor_local = 0.0
            current_time_local = 0.0
            starts_finishes_local: Dict[str, Tuple[float, float]] = {}
            bundle_results_local: Dict[str, List[SimBundleResult]] = {}
            fifo_wait_sum_local = 0.0

            # Build selection lookup
            selection_by_stmt: Dict[str, List[Bundle]] = {}
            for s, combo in zip(topo, choice):
                selection_by_stmt[s] = combo

            # Simulate per parallel group
            all_pim_timelines: List[Tuple[int, List[Tuple[float, int, float]], float]] = []  # (group_id, timeline, avg_util)
            for group_id, group in enumerate(groups):
                start = current_time_local
                fifo_wait_before = fifo_cursor_local
                g_time, fifo_cursor_local, g_results, pim_timeline, avg_util = simulate_group_block(
                    group,
                    selection_by_stmt,
                    args.cpu_cores,
                    arrays_total,
                    fifo_bw,
                    fifo_cursor_local,
                )
                current_time_local += g_time
                # Record per statement results and uniform group start/finish
                for s in group:
                    starts_finishes_local[s] = (start, current_time_local)
                    if s in g_results:
                        bundle_results_local[s] = g_results[s]
                fifo_wait_sum_local += fifo_wait_before
                # Adjust timeline to global time (add group start offset)
                adjusted_timeline = [(t + start, cols, util) for t, cols, util in pim_timeline]
                all_pim_timelines.append((group_id, adjusted_timeline, avg_util))
            
            # Track best for this specific P
            candidate_cost = current_time_local * max(1, int(target_p))
            if candidate_cost < best_cost_p:
                best_cost_p = candidate_cost
                best_output_p = {
                    "target_p": target_p,
                    "best_duration": current_time_local,
                    "starts_finishes": starts_finishes_local,
                    "bundle_results": bundle_results_local,
                    "fifo_total_wait": fifo_wait_sum_local,
                    "pim_timelines": all_pim_timelines,
                }
        if pbar is not None:
            pbar.close()
        elif args.show_progress:
            total_elapsed = time.perf_counter() - t_start
            final_rate = combo_counter / total_elapsed if total_elapsed > 0 else 0.0
            print(f"[P={target_p}] done {combo_counter}/{total_combos} in {total_elapsed:.3f}s "
                  f"({final_rate:.1f}/s)")
        # Save per-P best result and update global best
        if best_output_p is not None:
            payload = {
                "best_cost": best_cost_p,
                **best_output_p,
            }
            per_p_outputs[int(target_p)] = payload
            if best_cost_p < best_total:
                best_total = best_cost_p
                best_output = payload

    if not best_output:
        raise SystemExit("No feasible combination found for any P.")

    output_dir = args.output_dir or (Path("out/kernel_costs/advanced") / args.workload)
    json_out = output_dir / "schedule.json"
    md_out = output_dir / "schedule.md"

    # Write per-P outputs into subdirectories
    for p_val, payload in sorted(per_p_outputs.items()):
        p_dir = output_dir / f"P{p_val}"
        write_schedule_json(
            p_dir / "schedule.json",
            args.workload,
            int(payload["target_p"]),
            args.cpu_cores,
            arrays_total,
            fifo_bw,
            topo,
            payload["bundle_results"],
            payload["starts_finishes"],
            float(payload["best_duration"]),
            float(payload["fifo_total_wait"]),
            payload.get("pim_timelines"),
        )
        write_schedule_md(
            p_dir / "schedule.md",
            args.workload,
            float(payload["best_duration"]),
            int(payload["target_p"]),
            topo,
            payload["starts_finishes"],
        )

    write_schedule_json(
        json_out,
        args.workload,
        int(best_output["target_p"]),
        args.cpu_cores,
        arrays_total,
        fifo_bw,
        topo,
        best_output["bundle_results"],
        best_output["starts_finishes"],
        float(best_output["best_duration"]),
        float(best_output["fifo_total_wait"]),
        best_output.get("pim_timelines"),
    )
    write_schedule_md(
        md_out,
        args.workload,
        float(best_output["best_duration"]),
        int(best_output["target_p"]),
        topo,
        best_output["starts_finishes"],
    )
    print(f"Wrote schedule to {json_out} and {md_out}")


if __name__ == "__main__":
    main()
