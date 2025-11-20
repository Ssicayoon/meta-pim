#!/usr/bin/env bash
set -euo pipefail

# Run the final stage of the PIM pipeline:
#   schedule.json + stage MLIR -> PIM binary -> gem5 run.
#
# Usage:
#   ./scripts/run_schedule_to_gem5.sh <schedule.json> <stage_mlir> [out_dir]
#
REPO="$(cd "$(dirname "$0")/.." && pwd)"

SCHEDULE_JSON="${1:-"$REPO/out/kernel_costs/advanced/gemm/sweeps/bp/P1/schedule.json"}"
STAGE_MLIR="${2:-"$REPO/experiments/gemm/gemm_scop.mlir"}"
# Allow override of LLVM / gem5 paths via environment:
#   METAPIM_LLVM_BUILD_DIR – LLVM build dir (default: $REPO/third_party/Polygeist/llvm-project/build)
#   METAPIM_GEM5_ROOT      – gem5 root (default: $REPO/gem5-pim/gem5)
OUT_DIR="${3:-"$REPO/out/scheduled"}"
WORKLOAD="${4:-gemm}"

LLVM_BASE="${METAPIM_LLVM_BUILD_DIR:-"$REPO/third_party/Polygeist/llvm-project/build"}"
LLVM_BIN="$LLVM_BASE/bin"
LLVM_CONFIG="$LLVM_BIN/llvm-config"
CLANG="$LLVM_BIN/clang"
CLANGXX="$LLVM_BIN/clang++"
OPT="$LLVM_BIN/opt"
LLC="$LLVM_BIN/llc"
MLIR_OPT="$LLVM_BIN/mlir-opt"
MLIR_TRANSLATE="$LLVM_BIN/mlir-translate"

mkdir -p "$OUT_DIR"

echo "[schedule] Assemble kernel from schedule"
python3 "$REPO/src/schedule_to_pim_kernel.py" \
  --schedule "$SCHEDULE_JSON" \
  --source-mlir "$STAGE_MLIR" \
  --output "$OUT_DIR/scheduled_kernel.mlir" \
  --generate-missing \
  --generate-metadata

echo "[schedule] Lower PIM ops to intrinsics"
python3 "$REPO/src/lower_pim_to_intrinsics.py" \
  "$OUT_DIR/scheduled_kernel.mlir" \
  -o "$OUT_DIR/scheduled_kernel_intrinsic.mlir"

echo "[schedule] Inline calls after intrinsics lowering"
"$MLIR_OPT" "$OUT_DIR/scheduled_kernel_intrinsic.mlir" \
  -inline \
  -o "$OUT_DIR/scheduled_kernel_restored.mlir"

echo "[schedule] Lower to LLVM IR"
"$MLIR_OPT" "$OUT_DIR/scheduled_kernel_restored.mlir" \
  -lower-affine \
  --convert-to-llvm \
  --convert-arith-to-llvm \
  --convert-func-to-llvm \
  --reconcile-unrealized-casts \
  -o "$OUT_DIR/scheduled_kernel_intrinsic_llvm.mlir"
"$MLIR_TRANSLATE" "$OUT_DIR/scheduled_kernel_intrinsic_llvm.mlir" --mlir-to-llvmir \
  -o "$OUT_DIR/scheduled_kernel_intrinsic.ll"

echo "[schedule] Inline intrinsics to asm via LLVM pass"
LLVM_INCLUDE=$("$LLVM_CONFIG" --includedir)
LLVM_BUILD_INCLUDE="$LLVM_BASE/include"
"$CLANGXX" -fPIC -shared -I"$LLVM_INCLUDE" -I"$LLVM_BUILD_INCLUDE" "$REPO/src/llvm_passes/LowerPIMIntrinsics.cpp" \
  -o "$OUT_DIR/LowerPIMIntrinsics.scheduled.so"
"$OPT" -S -load-pass-plugin="$OUT_DIR/LowerPIMIntrinsics.scheduled.so" -passes=lower-pim-intrinsics \
  "$OUT_DIR/scheduled_kernel_intrinsic.ll" -o "$OUT_DIR/scheduled_kernel_inline.ll"
"$LLC" -filetype=obj "$OUT_DIR/scheduled_kernel_inline.ll" -o "$OUT_DIR/scheduled_kernel.o"

echo "[schedule] Build driver"
GEM5_ROOT="${METAPIM_GEM5_ROOT:-"$REPO/gem5-pim/gem5"}"
GEM5_INCLUDE="$GEM5_ROOT/include"
M5_LIB="$GEM5_ROOT/util/m5/build/x86/out/libm5.a"
DRIVER_C="$REPO/experiments/$WORKLOAD/gem5_driver.c"
"$CLANG" "$DRIVER_C" "$OUT_DIR/scheduled_kernel.o" \
  -pthread \
  -I"$OUT_DIR" \
  -I"$GEM5_INCLUDE" \
  -DGEM5_M5OPS \
  "$M5_LIB" \
  -o "$OUT_DIR/scheduled_kernel_pim.out"

echo "[schedule] Run in gem5"
mkdir -p "$OUT_DIR/m5out"
"$GEM5_ROOT/build/X86/gem5.opt" \
  --outdir "$OUT_DIR/m5out" \
  "$GEM5_ROOT/configs/example/gem5_library/x86-pim-runner.py" \
  --binary "$OUT_DIR/scheduled_kernel_pim.out" \
  > "$OUT_DIR/gem5_run.log"

echo "[schedule] Done."
