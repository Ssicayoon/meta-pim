LLM Transform stage (Component 1)
=================================

In the full Meta-PIM pipeline, the LLM Transform is the first component
that converts *irregular* C into the affine, scalarised kernels that the
rest of the compiler consumes (PolySat, E-graph Optimizer, Global Scheduler).

This directory documents how to apply the four LLM-guided transformations
described in the paper so that an LLM can transform an *original* (possibly
irregular) benchmark implementation into an affine benchmark C file of the
form:

  benchmark/<benchmark>/<benchmark>.c   (defining the kernel function `<kernel>`)

The four transformations are:

1. Static memory allocation (no `malloc` / dynamic sizes).
2. Affine loop normalization (constant steps, no data-dependent bounds).
3. Control-flow linearization (remove `if` branches via arithmetic).
4. Dependency elimination via precomputation (lookup tables, etc.).

The prompt template in `gemm_llm_prompt.md` can be used with an LLM to
perform this source-to-source rewrite, starting from an affine-unfriendly
benchmark such as `original_<benchmark>_affine_unfriendly.c`. The output is
expected to be close to `benchmark/<benchmark>/<benchmark>.c` (which defines
the kernel function `<kernel>`), and this benchmark file is what the rest of
the Meta-PIM pipeline uses as its starting point.
