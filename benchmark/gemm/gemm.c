/*
 * Scalarised 4x4 GEMM kernel derived from the PolyBench/C implementation
 * (polybench-c-3.2/linear-algebra/kernels/gemm/gemm.c).
 *
 * All loops and array accesses are unrolled so the function operates purely
 * on scalar values. This form is convenient for MLIR lowering passes that
 * expect straight-line arithmetic instead of memref operations.
 */

#include "gemm.h"

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
        int64_t b30, int64_t b31, int64_t b32, int64_t b33)
{
    struct Gemm4x4Result out;

    int64_t r;

    r = beta * c00;
    r += alpha * a00 * b00;
    r += alpha * a01 * b10;
    r += alpha * a02 * b20;
    r += alpha * a03 * b30;
    out.c00 = r;

    r = beta * c01;
    r += alpha * a00 * b01;
    r += alpha * a01 * b11;
    r += alpha * a02 * b21;
    r += alpha * a03 * b31;
    out.c01 = r;

    r = beta * c02;
    r += alpha * a00 * b02;
    r += alpha * a01 * b12;
    r += alpha * a02 * b22;
    r += alpha * a03 * b32;
    out.c02 = r;

    r = beta * c03;
    r += alpha * a00 * b03;
    r += alpha * a01 * b13;
    r += alpha * a02 * b23;
    r += alpha * a03 * b33;
    out.c03 = r;

    r = beta * c10;
    r += alpha * a10 * b00;
    r += alpha * a11 * b10;
    r += alpha * a12 * b20;
    r += alpha * a13 * b30;
    out.c10 = r;

    r = beta * c11;
    r += alpha * a10 * b01;
    r += alpha * a11 * b11;
    r += alpha * a12 * b21;
    r += alpha * a13 * b31;
    out.c11 = r;

    r = beta * c12;
    r += alpha * a10 * b02;
    r += alpha * a11 * b12;
    r += alpha * a12 * b22;
    r += alpha * a13 * b32;
    out.c12 = r;

    r = beta * c13;
    r += alpha * a10 * b03;
    r += alpha * a11 * b13;
    r += alpha * a12 * b23;
    r += alpha * a13 * b33;
    out.c13 = r;

    r = beta * c20;
    r += alpha * a20 * b00;
    r += alpha * a21 * b10;
    r += alpha * a22 * b20;
    r += alpha * a23 * b30;
    out.c20 = r;

    r = beta * c21;
    r += alpha * a20 * b01;
    r += alpha * a21 * b11;
    r += alpha * a22 * b21;
    r += alpha * a23 * b31;
    out.c21 = r;

    r = beta * c22;
    r += alpha * a20 * b02;
    r += alpha * a21 * b12;
    r += alpha * a22 * b22;
    r += alpha * a23 * b32;
    out.c22 = r;

    r = beta * c23;
    r += alpha * a20 * b03;
    r += alpha * a21 * b13;
    r += alpha * a22 * b23;
    r += alpha * a23 * b33;
    out.c23 = r;

    r = beta * c30;
    r += alpha * a30 * b00;
    r += alpha * a31 * b10;
    r += alpha * a32 * b20;
    r += alpha * a33 * b30;
    out.c30 = r;

    r = beta * c31;
    r += alpha * a30 * b01;
    r += alpha * a31 * b11;
    r += alpha * a32 * b21;
    r += alpha * a33 * b31;
    out.c31 = r;

    r = beta * c32;
    r += alpha * a30 * b02;
    r += alpha * a31 * b12;
    r += alpha * a32 * b22;
    r += alpha * a33 * b32;
    out.c32 = r;

    r = beta * c33;
    r += alpha * a30 * b03;
    r += alpha * a31 * b13;
    r += alpha * a32 * b23;
    r += alpha * a33 * b33;
    out.c33 = r;

    return out;
}
