#!/usr/bin/env bash
set -euo pipefail

# Generate "with PIM" MLIR for each statement's p*_extracted.txt under
# <kernel_cost_root>/{bp,bs,cpu,hybrid}/gemm/S*/.
#
# Usage:
#   bash scripts/egg_to_pim_mlir.sh
#

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# Allow override via METAPIM_KERNEL_COST_ROOT; default to experiments/gemm/kernel_costs
BASE_OUT="${METAPIM_KERNEL_COST_ROOT:-"$REPO_ROOT/experiments/gemm/kernel_costs"}"
DRIVER="${REPO_ROOT}/src/egg_to_mlir.py"

if [[ ! -f "${DRIVER}" ]]; then
  echo "egg_to_mlir.py not found: ${DRIVER}" >&2
  exit 1
fi

layouts=(bp bs cpu hybrid)

for layout in "${layouts[@]}"; do
  root="${BASE_OUT}/${layout}/gemm"
  if [[ ! -d "${root}" ]]; then
    continue
  fi
  # Iterate statements (e.g., S0, S1, ...)
  find "${root}" -maxdepth 1 -mindepth 1 -type d | sort | while read -r stmt_dir; do
    # For all available parallelisms: p*_extracted.txt
    shopt -s nullglob
    for in_txt in "${stmt_dir}"/p*_extracted.txt; do
      out_mlir="${in_txt%_extracted.txt}_with_pim.mlir"
      echo "Generating with-PIM MLIR: ${in_txt} -> ${out_mlir}"
      # Detect max CpuArg index and pass arg0..argN as i64
      max_idx="$(grep -o 'CpuArg [0-9]\+' "$in_txt" | awk '{print $2}' | sort -n | tail -1 || true)"
      args=()
      if [[ -n "${max_idx:-}" ]]; then
        for ((i=0; i<=max_idx; i++)); do
          args+=(--arg "arg${i}:i64")
        done
      fi
      python3 "$DRIVER" "$in_txt" --raw-expr "${args[@]}" -o "$out_mlir"
    done
    shopt -u nullglob
  done
done

echo "Done."
