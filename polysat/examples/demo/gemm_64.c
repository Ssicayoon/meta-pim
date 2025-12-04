// General Matrix Multiplication (GEMM)
// C = alpha * A * B + beta * C
// This is one of the most important kernels in linear algebra and deep learning

#define N 64

// Standard GEMM: C = A * B
void gemm_standard(double A[N][N], double B[N][N], double C[N][N]) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            for (int k = 0; k < N; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}
