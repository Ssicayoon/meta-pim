#ifndef GEMM_H
#define GEMM_H

#include <stdint.h>

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

#endif /* GEMM_H */
