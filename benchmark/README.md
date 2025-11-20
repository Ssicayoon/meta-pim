Benchmarks
==========

This directory collects the workloads targeted by the Meta-PIM compiler
framework. Each subdirectory corresponds to a benchmark and contains the
LLM-regularised kernel(s) that feed into the PolySat front-end.

Current contents
----------------

- `gemm/` – Scalarised 4×4 GEMM kernel (`gemm.c`, `gemm.h`), used by the
  end-to-end pipeline in this release.

More workloads (e.g., NTT, FFT, convolution, BlackScholes) will be added
to this directory over time and will reuse the same Meta-PIM compilation
pipeline.

