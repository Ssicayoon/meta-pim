#!/usr/bin/env bash
set -euo pipefail

# Pipeline: C -> MLIR -> Affine -> ISL (for load-compute-store splitting)

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# Default output directory (can be overridden by 4th argument)
DEFAULT_OUT_DIR="$REPO_ROOT/experiments/gemm"

# Tool paths (Polygeist toolchain).
# Allow override via environment:
#   METAPIM_POLYGEIST_BASE  – Polygeist source/build root (default: $REPO_ROOT/third_party/Polygeist)
#   METAPIM_LLVM_BUILD_DIR  – LLVM build dir with mlir-opt (default: $METAPIM_POLYGEIST_BASE/llvm-project/build)
POLYGEIST_BASE="${METAPIM_POLYGEIST_BASE:-"$REPO_ROOT/third_party/Polygeist"}"
LLVM_BASE="${METAPIM_LLVM_BUILD_DIR:-"$POLYGEIST_BASE/llvm-project/build"}"

CGEIST="$POLYGEIST_BASE/build/bin/cgeist"
POLYGEIST_OPT="$POLYGEIST_BASE/build/bin/polygeist-opt"
POLYMER_OPT="$POLYGEIST_BASE/build/bin/polymer-opt"
MLIR_OPT="$LLVM_BASE/bin/mlir-opt"

log() {
  echo "[Stage Split Pipeline] $1"
}

# Parse arguments
C_FILE="${1:-$REPO_ROOT/benchmark/gemm/gemm.c}"
FUNCTION_NAME="${2:-gemm}"
# OUTPUT_PREFIX controls the basename of generated MLIR files:
#   <prefix>_raw.mlir, <prefix>_affine.mlir, <prefix>_scop.mlir
OUTPUT_PREFIX="${3:-gemm}"
OUT_DIR="${4:-$DEFAULT_OUT_DIR}"

mkdir -p "$OUT_DIR"

if [ ! -f "$C_FILE" ]; then
  echo "Error: C file not found: $C_FILE"
  exit 1
fi

log "Step 1: C -> MLIR (using cgeist)"
"$CGEIST" "$C_FILE" \
  -O0 -S --function="$FUNCTION_NAME" \
  -o "$OUT_DIR/${OUTPUT_PREFIX}_raw.mlir"

log "Step 2: MLIR -> Affine Dialect (with memref->affine conversion)"
"$POLYGEIST_OPT" "$OUT_DIR/${OUTPUT_PREFIX}_raw.mlir" \
  --raise-scf-to-affine \
  --affine-cfg \
  --canonicalize \
  -o "$OUT_DIR/${OUTPUT_PREFIX}_affine.mlir"

log "Step 3: Extract SCoP statements"
# Extract scop.stmt functions using polymer-opt
# Note: Remove module attributes that polymer-opt doesn't support
sed 's/module attributes {.*} {/module {/' "$OUT_DIR/${OUTPUT_PREFIX}_affine.mlir" > "$OUT_DIR/${OUTPUT_PREFIX}_affine_clean.mlir"
"$POLYMER_OPT" "$OUT_DIR/${OUTPUT_PREFIX}_affine_clean.mlir" \
  -extract-scop-stmt \
  -o "$OUT_DIR/${OUTPUT_PREFIX}_scop.mlir"

log "Step 4: Verify SCoP extraction"
if grep -q "scop.stmt" "$OUT_DIR/${OUTPUT_PREFIX}_scop.mlir"; then
  log "✓ Successfully extracted SCoP statements"
  grep "scop.stmt" "$OUT_DIR/${OUTPUT_PREFIX}_scop.mlir" | head -5
else
  log "✗ Failed to extract SCoP statements"
  exit 1
fi

log "Pipeline completed"
log "Output files:"
log "  - Raw MLIR: $OUT_DIR/${OUTPUT_PREFIX}_raw.mlir"
log "  - Affine MLIR: $OUT_DIR/${OUTPUT_PREFIX}_affine.mlir"
log "  - SCoP extracted: $OUT_DIR/${OUTPUT_PREFIX}_scop.mlir"
