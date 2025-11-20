Meta-PIM: An End-to-End Compilation Framework for Configurable Processing-in-Memory
===================================================================================

This repository is the **official open-source code release** for the Meta-PIM
compilation framework. Meta-PIM is a universal, end-to-end compilation system
for configurable Processing-in-Memory (PIM) that integrates four stages:
LLM-guided source transformation, a PolySat polyhedral front-end, an
E-graph–based layout and instruction optimizer, and a global scheduler with
PIM back-end. Together with a co-designed NoC-based PIM architecture, this
pipeline delivers speedups and energy reductions over CPU and bit-serial PIM
baselines across diverse workloads.

Architecture
------------

The overall Meta-PIM architecture (compiler + hardware) is summarized in:

![Meta-PIM architecture](./metapim_architecture.png)

This figure shows a four-stage compilation pipeline:

1. **LLM Transform** – converts irregular, affine-unfriendly C code into
   affine-friendly kernels.
2. **PolySat front-end** – performs dependency-aware clustering and tiling
   using polyhedral analysis.
3. **E-graph Optimizer** – explores CPU/BP/BS/hybrid layouts and PIM-specific
   instruction mappings via equality saturation.
4. **Global Scheduler + PIM back-end** – coordinates resource-aware placement
   across CPU/PIM and lowers PIM ops to a concrete ISA on the NCP hardware.

Entry point
-----------

- `run_metapim.sh` – end-to-end Meta-PIM pipeline script.

Quick usage
-----------

1. Set environment variables to point at your local toolchain (example paths assume the full repo is at `/path/to/egg-pim/egg-pim`):

   ```bash
   export METAPIM_POLYGEIST_BASE=/path/to/egg-pim/egg-pim/third_party/Polygeist
   export METAPIM_LLVM_BUILD_DIR=/path/to/egg-pim/egg-pim/third_party/Polygeist/llvm-project/build
   export METAPIM_GEM5_ROOT=/path/to/egg-pim/egg-pim/gem5-pim/gem5
   export METAPIM_EGGLOG_BIN=egglog-experimental
   ```

2. Run the default pipeline (using the default benchmark and P=1, layout=bp):

   ```bash
   ./run_metapim.sh
   ```

3. Run with a different parallelism `P` and layouts (space- or comma-separated):

   ```bash
   ./run_metapim.sh --target-p 64 --layouts "bp bs"
   # or
   TARGET_P=128 LAYOUTS="cpu,bp,bs" ./run_metapim.sh
   ```

Key outputs (per benchmark `<benchmark>` and kernel `<kernel>`):

- SCoP MLIR: `experiments/<benchmark>/<kernel>_scop.mlir`
- Kernel costs: `experiments/<benchmark>/kernel_costs/<layout>/<kernel>/S*/pP_cost.json`
  where `S*` are per-statement kernels (e.g., `S0`, `S1`, ...).
- Schedule: `out/<kernel>_pim_schedule/<layout>_PP/schedule.json`
- PIM binary + gem5 logs: `out/<kernel>_pim_run/<layout>_PP/`

Component mapping
-----------------

Below we map each stage in the architecture figure to the concrete files in
this release.

**Component 1 – LLM Transform (C → affine-friendly C)**

- Inputs:
  - `llm_transform/original_<benchmark>_affine_unfriendly.c`  
    Example affine-unfriendly C benchmark with dynamic allocation,
    data-dependent strides, and branches.
- LLM prompt:
  - `llm_transform/<kernel>_llm_prompt.md`  
    Describes the four transformations: static allocation, affine loop
    normalisation, control-flow linearisation, and dependency elimination.
- Output (fed into the rest of the pipeline):
  - `benchmark/<benchmark>/<benchmark>.c`, `benchmark/<benchmark>/<benchmark>.h`  
    Affine-friendly benchmark C/H files. The hot function inside (the
    *kernel* for this benchmark) is what PolySat subsequently extracts
    and labels as `@S*` statements.

**Component 2 – PolySat front-end (MLIR + polyhedral analysis)**

- `scripts/polysat_frontend.sh`  
  C → MLIR → affine → SCoP extraction using cgeist + polygeist-opt + polymer.
- Key artifacts:
  - `experiments/<benchmark>/<kernel>_raw.mlir`
  - `experiments/<benchmark>/<kernel>_affine.mlir`
  - `experiments/<benchmark>/<kernel>_scop.mlir`

**Component 3 – E-graph Optimizer (layouts + PIM instruction mapping)**

- Cost modelling and expression extraction:
  - `scripts/egraph_cost_model.sh`
  - `src/mlir_to_egg.py`
  - `src/prepare_kernel_cost_model.py`
  - `src/kernel_cost_capture.py`
  - `src/analyze_egg_cost.py`
- Egglog expression → PIM MLIR:
  - `src/egg_to_mlir.py`
  - `scripts/egg_to_pim_mlir.sh`
- Key artifacts:
  - `experiments/<benchmark>/kernel_costs/<layout>/<kernel>/S*/pP_cost.json`
  - `experiments/<benchmark>/kernel_costs/<layout>/<kernel>/S*/pP_extracted.txt`
  - `experiments/<benchmark>/kernel_costs/<layout>/<kernel>/S*/pP_with_pim.mlir`

**Component 4 – Global Scheduler + PIM back-end**

- Global scheduler (bundle selection + dependency-aware grouping):
  - `src/global_scheduler.py`
  - Outputs `schedule.json` under:
    - `out/<kernel>_pim_schedule/<layout>_PP/`
- Schedule → PIM kernel MLIR + metadata:
  - `src/schedule_to_pim_kernel.py`
  - Artifacts:
    - `out/<kernel>_pim_run/<layout>_PP/scheduled_kernel.mlir`
    - `out/<kernel>_pim_run/<layout>_PP/scheduled_kernel_metadata.{json,h}`
- PIM back-end (intrinsics lowering + NCP ISA + gem5):
  - `src/lower_pim_to_intrinsics.py`
  - `src/llvm_passes/LowerPIMIntrinsics.cpp`
  - `scripts/schedule_to_gem5.sh`
  - `experiments/<benchmark>/gem5_driver.c`

External dependencies (to be integrated):

- Polygeist toolchain under `third_party/Polygeist` (official upstream Polygeist, not modified here).
- Egglog binary (invoked as `egglog-experimental`; use the official Egglog release).
- gem5 with PIM extensions under `gem5-pim/gem5` (Meta-PIM PIM extensions, to be added/integrated into this release).

You can also run the Meta-PIM flow with an unmodified upstream gem5 for CPU-only evaluation; without the PIM extensions, gem5 cannot execute the PIM instructions, so the PIM-lowering stages (e.g., lowering to PIM intrinsics and ISA) should be disabled or skipped.

For details of the individual stages, refer to the scripts under `scripts/`
and the source files under `src/`.