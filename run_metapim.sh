#!/usr/bin/env bash
set -euo pipefail

# One-shot Meta-PIM pipeline: C kernel -> Polyhedral front-end -> Egglog cost
# model -> Global scheduler -> PIM kernel -> gem5 (PIM instructions).
#
# Usage:
#   ./run_metapim.sh
#
# External dependencies (not included in this minimal tree):
#   - Polygeist toolchain under third_party/Polygeist
#   - Egglog binary (egglog-experimental)
#   - gem5 with PIM extensions under gem5-pim/gem5

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"

# Default workload and tuning knobs:
#   WORKLOAD        – logical name / directory under benchmark/ and experiments/
#   TARGET_P        – target parallelism (per-stage P), default 1
#   LAYOUTS         – space- or comma-separated layouts (e.g., \"bp bs\")
WORKLOAD="${WORKLOAD:-gemm}"
TARGET_P="${TARGET_P:-1}"
LAYOUTS="${LAYOUTS:-bp}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --workload)
      WORKLOAD="$2"
      shift 2
      ;;
    --target-p)
      TARGET_P="$2"
      shift 2
      ;;
    --layouts)
      LAYOUTS="$2"
      shift 2
      ;;
    --help|-h)
      echo "Usage: $0 [--workload NAME] [--target-p P] [--layouts \"cpu bp bs hybrid\"]"
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      echo "Usage: $0 [--workload NAME] [--target-p P] [--layouts \"cpu bp bs hybrid\"]" >&2
      exit 1
      ;;
  esac
done

# Derive paths from workload:
SRC_C="$REPO_ROOT/benchmark/$WORKLOAD/$WORKLOAD.c"
EXPERIMENT_DIR="$REPO_ROOT/experiments/$WORKLOAD"
STAGE_MLIR="$EXPERIMENT_DIR/${WORKLOAD}_scop.mlir"
COST_ROOT="$EXPERIMENT_DIR/kernel_costs"

# Normalise layouts:
# - LAYOUTS_SPACES: space-separated (for shell loops, egraph_cost_model.sh)
# - LAYOUTS_CSV:    comma-separated (for global_scheduler.py)
LAYOUTS_SPACES="${LAYOUTS//,/ }"
LAYOUTS_CSV="${LAYOUTS// /,}"
# Use the first layout to name output directories
FIRST_LAYOUT=$(printf '%s\n' $LAYOUTS_SPACES | head -n 1)
SCHED_DIR="$REPO_ROOT/out/${WORKLOAD}_pim_schedule/${FIRST_LAYOUT}_P${TARGET_P}"
RUN_OUT_DIR="$REPO_ROOT/out/${WORKLOAD}_pim_run/${FIRST_LAYOUT}_P${TARGET_P}"

echo "╔════════════════════════════════════════════════════════════╗"
echo "║  Meta-PIM pipeline (real_minimal_release)                 ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

echo "[1/4] Polyhedral front-end: C -> MLIR (scop)..."
bash "$REPO_ROOT/scripts/polysat_frontend.sh" \
  "$SRC_C" \
  gemm \
  "$WORKLOAD" \
  "$EXPERIMENT_DIR"
echo ""

echo "[2/4] E-graph optimizer: generating kernel costs with Egglog..."
bash "$REPO_ROOT/scripts/egraph_cost_model.sh" \
  --source-mlir "$STAGE_MLIR" \
  --workload "$WORKLOAD" \
  --output-dir "$COST_ROOT" \
  --array-length 1 \
  --parallelisms "$TARGET_P" \
  --layout-modes "$LAYOUTS_SPACES"
echo ""

echo "[3/4] Global scheduler: layouts=$LAYOUTS_CSV, P=$TARGET_P..."
mkdir -p "$SCHED_DIR"
python3 "$REPO_ROOT/src/global_scheduler.py" \
  --workload "$WORKLOAD" \
  --stage-mlir "$STAGE_MLIR" \
  --target-p "$TARGET_P" \
  --layouts "$LAYOUTS_CSV" \
  --cost-root "$COST_ROOT" \
  --output-dir "$SCHED_DIR" \
  --per-statement-topk 1
echo ""

SCHEDULE_JSON="$SCHED_DIR/schedule.json"
if [[ ! -f "$SCHEDULE_JSON" ]]; then
  echo "Error: schedule.json not found at $SCHEDULE_JSON" >&2
  exit 1
fi

echo "[4/4] PIM back-end: lowering to binary and running gem5..."
bash "$REPO_ROOT/scripts/schedule_to_gem5.sh" \
  "$SCHEDULE_JSON" \
  "$STAGE_MLIR" \
  "$RUN_OUT_DIR" \
  "$WORKLOAD"

echo ""
echo "✅ Full PIM GEMM pipeline completed."
echo "Artifacts under:"
echo "  - Schedule:   $SCHEDULE_JSON"
echo "  - PIM binary: $RUN_OUT_DIR/scheduled_kernel_pim.out"
echo "  - gem5 out:   $RUN_OUT_DIR/m5out"
