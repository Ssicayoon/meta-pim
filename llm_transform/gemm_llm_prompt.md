LLM Transform Prompt: GEMM (irregular → affine kernel)
======================================================

**Goal.** Rewrite the *original* GEMM implementation into a small affine,
scalarised kernel that matches the structure of `gemm` in
`benchmark/gemm/gemm.c` (4×4 tile, scalar arguments, no dynamic allocation).

You (the LLM) are the *Component (1) LLM Transform* in the Meta-PIM
compiler pipeline, sitting before PolySat, the E-graph Optimizer, and
the Global Scheduler.

Input format
------------

- I will paste an *original* C implementation of GEMM that may contain:
  - Heap allocations (`malloc`, `free`) and pointer arithmetic.
  - Nested loops with variable bounds and/or variable strides.
  - Conditionals (`if` / `else`) inside the hot loop.
  - Loop-carried dependencies (e.g., accumulators, twiddle updates).

- You must emit a *single C translation unit* that defines:
  - A small, scalarised GEMM kernel named `gemm` with signature:

    ```c
    struct Gemm4x4Result {
        int64_t c00; int64_t c01; int64_t c02; int64_t c03;
        int64_t c10; int64_t c11; int64_t c12; int64_t c13;
        int64_t c20; int64_t c21; int64_t c22; int64_t c23;
        int64_t c30; int64_t c31; int64_t c32; int64_t c33;
    };

    struct Gemm4x4Result gemm(
        int64_t alpha, int64_t beta,
        int64_t c00, int64_t c01, int64_t c02, int64_t c03,
        int64_t c10, int64_t c11, int64_t c12, int64_t c13,
        int64_t c20, int64_t c21, int64_t c22, int64_t c23,
        int64_t c30, int64_t c31, int64_t c32, int64_t c33,
        int64_t a00, int64_t a01, int64_t a02, int64_t a03,
        int64_t a10, int64_t a11, int64_t a12, int64_t a13,
        int64_t a20, int64_t a21, int64_t a22, int64_t a23,
        int64_t a30, int64_t a31, int64_t a32, int64_t a33,
        int64_t b00, int64_t b01, int64_t b02, int64_t b03,
        int64_t b10, int64_t b11, int64_t b12, int64_t b13,
        int64_t b20, int64_t b21, int64_t b22, int64_t b23,
        int64_t b30, int64_t b31, int64_t b32, int64_t b33);
    ```


Required transformations
------------------------

Apply the following four transformations, as described in the Meta-PIM paper.

1. **Static memory allocation.**
   - Eliminate all dynamic allocation (`malloc`, `calloc`, `realloc`, `free`)
     and pointer-based heap arrays.
   - Replace them with either:
     - Explicit scalar parameters (for small tiles, e.g., 4×4 GEMM), or
     - Static arrays with fixed bounds (e.g., `double A[4][4]`).
   - Ensure the resulting kernel uses only scalar arguments and/or static
     arrays on the stack, so that the MLIR front-end (cgeist) produces
     static `memref` types.

2. **Affine loop normalization.**
   - Remove non-affine or variable-step loops (e.g., `for (len = 2; len <= n; len <<= 1)`).
   - For GEMM, fully unroll the 4×4 tile into straight-line scalar code:
     - No `for` loops remain in the final `gemm` body.
     - All indices become scalar variables (`a00`, `a01`, ..., `b33`).
   - The result must be affine/analyzable by PolySat because there are no
     remaining data-dependent loop bounds or strides.

3. **Control-flow linearization.**
   - Remove `if`/`else` branches from the hot computation:
     - Rewrite simple branches as arithmetic (e.g., `c ? a : b` →
       `c*a + (1-c)*b`), or hoist them out if they select between
       different kernels.
   - The final `gemm` kernel should have no conditional branches inside
     the arithmetic core; it should be purely straight-line arithmetic.

4. **Dependency elimination / precomputation.**
   - Eliminate loop-carried dependencies that block parallelization:
     - For GEMM, this typically means choosing a small tile (4×4) and
       computing it independently using only its inputs.
   - Ensure that the kernel is a pure function of its scalar inputs
     (`alpha`, `beta`, C tile, A tile, B tile) and has no hidden state.

Style and constraints
---------------------

- Use only standard C (C99 or later) and `int64_t` for integer types.
- Do not introduce any external dependencies, macros, or inline assembly.
- Do not allocate memory dynamically or use global mutable state.
- The body of `gemm` should be a sequence of scalar arithmetic operations
  that computes the 4×4 output tile according to the standard GEMM formula:

  \[
      C' = \beta C + \alpha A B
  \]

  where `A`, `B`, and `C` are 4×4 matrices and `C'` is the result.

Output format
-------------

- Emit a single self-contained C file containing:
  - The `#include <stdint.h>` directive.
  - The definition of `struct Gemm4x4Result`.
  - The implementation of `gemm` as specified above.
- Do not include any comments describing the original code; focus on the
  transformed kernel.

Example usage
-------------

The final `gemm` kernel you produce will be compiled by the Meta-PIM pipeline
into MLIR, passed through PolySat, the E-graph Optimizer, and the Global
Scheduler, and finally lowered to PIM instructions and executed inside gem5.
