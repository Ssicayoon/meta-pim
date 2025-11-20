Experiments
===========

This directory contains per-benchmark experiment folders.

- Each subdirectory (for example, `gemm/`) corresponds to one benchmark.
- Inside each benchmark directory there is a driver (for example, `gem5_driver.c`), which wraps the benchmark kernel so it can be run inside gem5.
- For each benchmark, SCoP MLIR files and kernel cost artifacts produced by the Meta-PIM pipeline (for example, `<kernel>_scop.mlir` and `kernel_costs/`) are also stored under the corresponding subdirectory.
