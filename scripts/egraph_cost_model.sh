#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")"/.. && pwd)"
SOURCE_MLIR="$REPO_ROOT/experiments/gemm/gemm_scop.mlir"
WORKLOAD="gemm"
PARALLELISMS="1 2 4 8 16 32 64 128 256 512 1024 2048 4096 8192 16384 32768 65536"
# PARALLELISMS="1 2 4 8 16 32 64 128"
LAYOUT_MODES="hybrid bp bs cpu"
# LAYOUT_MODES="hybrid"
ARRAY_LENGTH="1"
CPU_COUNT="1"
OUTPUT_DIR="$REPO_ROOT/experiments/gemm/kernel_costs"
# Allow override via METAPIM_EGGLOG_BIN; otherwise default to egglog-experimental.
EGGLOG_BIN="${METAPIM_EGGLOG_BIN:-egglog-experimental}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --source-mlir)
      SOURCE_MLIR="$2"
      shift 2
      ;;
    --workload)
      WORKLOAD="$2"
      shift 2
      ;;
    --parallelisms)
      PARALLELISMS="$2"
      shift 2
      ;;
    --layout-modes)
      LAYOUT_MODES="$2"
      shift 2
      ;;
    --array-length)
      ARRAY_LENGTH="$2"
      shift 2
      ;;
    --cpu-count)
      CPU_COUNT="$2"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --egglog-binary)
      EGGLOG_BIN="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

if [[ ! -f "$SOURCE_MLIR" ]]; then
  echo "Error: cannot find $SOURCE_MLIR" >&2
  exit 1
fi

FUNCS=$(rg -o '@S[0-9]+' "$SOURCE_MLIR" | sort -u | tr -d '@')
if [[ -z "$FUNCS" ]]; then
  echo "Error: no functions matching @S* found in $SOURCE_MLIR" >&2
  exit 1
fi

for PAR in $PARALLELISMS; do
  echo "=== Parallelism $PAR ==="
  for FUNC in $FUNCS; do
    echo "--- Processing $FUNC ---"
    for LAYOUT in $LAYOUT_MODES; do
      echo ">>> Layout $LAYOUT"
      LAYOUT_OUTPUT_DIR="$OUTPUT_DIR/$LAYOUT"
      
      # # Check if cost.json already exists
      # COST_JSON="$LAYOUT_OUTPUT_DIR/$WORKLOAD/$FUNC/p${PAR}_cost.json"
      # if [[ -f "$COST_JSON" ]]; then
      #   echo "    [SKIP] $COST_JSON already exists"
      #   continue
      # fi
      
      python3 "$REPO_ROOT/src/prepare_kernel_cost_model.py" \
        --source-mlir "$SOURCE_MLIR" \
        --function "$FUNC" \
        --parallelism "$PAR" \
        --workload "$WORKLOAD" \
        --output-dir "$LAYOUT_OUTPUT_DIR"

      sleep 0.1

      CAPTURE_CMD=(python3 "$REPO_ROOT/src/kernel_cost_capture.py" \
        --source-mlir "$SOURCE_MLIR" \
        --function "$FUNC" \
        --workload "$WORKLOAD" \
        --parallelism-label "$PAR" \
        --layout-mode "$LAYOUT" \
        --cpu-count "$CPU_COUNT" \
        --output-dir "$LAYOUT_OUTPUT_DIR" \
        --egglog-binary "$EGGLOG_BIN")
      if [[ -n "$ARRAY_LENGTH" ]]; then
        CAPTURE_CMD+=(--array-length "$ARRAY_LENGTH")
      fi
      "${CAPTURE_CMD[@]}"
    done
  done
done
