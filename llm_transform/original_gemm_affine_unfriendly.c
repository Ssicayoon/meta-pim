#include <stdint.h>
#include <stdlib.h>

/*
 * Original kernel example (affine-unfriendly, pre-LLM Transform).
 *
 * This code intentionally contains patterns that are inconvenient for
 * direct affine / polyhedral analysis:
 *   - dynamic memory allocation via malloc/free,
 *   - data-dependent loop strides,
 *   - a conditional branch inside the hot loop.
 *
 * The LLM Transform (Component 1) is expected to take a function like
 * this and rewrite it into a small affine, scalarised kernel matching
 * the affine-friendly version used by the Meta-PIM pipeline.
 */

void kernel_affine_unfriendly(int64_t n,
                              int64_t alpha,
                              int64_t beta) {
    int64_t *A = (int64_t *)malloc(n * n * sizeof(int64_t));
    int64_t *B = (int64_t *)malloc(n * n * sizeof(int64_t));
    int64_t *C = (int64_t *)malloc(n * n * sizeof(int64_t));

    if (!A || !B || !C) {
        free(A);
        free(B);
        free(C);
        return;
    }

    // Initialise with some values.
    for (int64_t i = 0; i < n; ++i) {
        for (int64_t j = 0; j < n; ++j) {
            A[i * n + j] = i + j + 1;
            B[i * n + j] = (i - j);
            C[i * n + j] = (i + 1) * (j + 1);
        }
    }

    // Affine-unfriendly pattern: data-dependent inner stride and a conditional branch.
    for (int64_t i = 0; i < n; ++i) {
        for (int64_t j = 0; j < n; ) {
            int64_t sum = 0;
            for (int64_t k = 0; k < n; ++k) {
                int64_t a = A[i * n + k];
                int64_t b = B[k * n + j];
                // Simple conditional to break affine form.
                if ((j ^ k) & 1) {
                    sum += alpha * a * b;
                } else {
                    sum += alpha * a * b + 1;
                }
            }
            C[i * n + j] = beta * C[i * n + j] + sum;

            // Data-dependent j increment (non-constant stride).
            int64_t step = 1 + (j & 1);
            j += step;
        }
    }

    // In a real application, C would be returned or stored; here we just free.
    free(A);
    free(B);
    free(C);
}

