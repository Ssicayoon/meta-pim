/**
 * Optimized GEMM 64x64x64 - EXACT ISL Schedule from PolySat
 *
 * Schedule rank_0001, Cost: 0.026
 *
 * EXACT ISL Schedule Structure (no approximations):
 * 1. mark: "parallel" (outer)
 * 2. mark: "parallel" (inner)
 * 3. schedule: [(i0-(i0)mod 32), (i1-(i1)mod 32), (i2-(i2)mod 32)]
 *              // 32x32x32 outer tiles
 * 4. schedule: [(-1*((i0)mod 16)+(i0)mod 32),
 *               (-1*((i1)mod 16)+(i1)mod 32),
 *               (-1*((i2)mod 8)+(i2)mod 32)]
 *              // 16x16x8 inner tiles
 * 5. schedule: [(0), (0), (0)]
 *              // separator
 * 6. schedule: [((i0)mod 16), ((i1)mod 16), ((i2)mod 8)]
 *              // point loops
 * 7. schedule: L1[(i1)]
 *              // original j loop
 * 8. schedule: L0[(i2)]
 *              // original k loop
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define N 64
#define WARMUP_RUNS 3
#define BENCHMARK_RUNS 10

// Original baseline GEMM for reference
void gemm_baseline(float A[N][N], float B[N][N], float C[N][N]) {
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      for (int k = 0; k < N; k++) {
        C[i][j] += A[i][k] * B[k][j];
      }
    }
  }
}

// Optimized GEMM following the best ISL schedule
void gemm_optimized(float A[N][N], float B[N][N], float C[N][N]) {
// Level 1: Outer 3D tiling (32x32x32) - PARALLEL
// ISL: [(i0 - (i0) mod 32), (i1 - (i1) mod 32), (i2 - (i2) mod 32)]
#pragma omp parallel for collapse(2)
  for (int ii = 0; ii < N; ii += 32) {     // tile_i0_32
    for (int jj = 0; jj < N; jj += 32) {   // tile_i1_32
      for (int kk = 0; kk < N; kk += 32) { // tile_k_32

        // Level 2: Inner tiling (16x16x8)
        // ISL: [(-1*((i0) mod 16) + (i0) mod 32),
        //       (-1*((i1) mod 16) + (i1) mod 32),
        //       (-1*((i2) mod 8) + (i2) mod 32)]
        // This creates 4 sub-tiles of size [0-15], [16-31] for i,j
        // and 4 sub-tiles of size [0-7], [8-15], [16-23], [24-31] for k
        for (int iii = ii; iii < ii + 32; iii += 16) {
          for (int jjj = jj; jjj < jj + 32; jjj += 16) {
            for (int kkk = kk; kkk < kk + 32; kkk += 8) { // FIXED: 8 not 16!

              // Level 3: Zero padding schedules (constant 0)
              // ISL: [(0), (0), (0)]
              // This is just a no-op separator in the schedule tree

              // Level 4: Point loops within tiles
              // ISL: [((i0) mod 16), ((i1) mod 16), ((i2) mod 8)]
              for (int i = iii; i < iii + 16 && i < N; i++) {
                for (int j = jjj; j < jjj + 16 && j < N; j++) {
                  // Innermost k-loop: mod 8, not mod 16!
                  for (int k = kkk; k < kkk + 8 && k < N; k++) {
                    C[i][j] += A[i][k] * B[k][j];
                  }
                }
              }
            }
          }
        }
      }
    }
  }
}

// Timing helper functions
static inline double get_time_seconds() {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
}

static inline double get_time_ms() { return get_time_seconds() * 1000.0; }

// Calculate statistics
void calculate_stats(double *times, int n, double *mean, double *min,
                     double *max, double *stddev) {
  *mean = 0.0;
  *min = times[0];
  *max = times[0];

  for (int i = 0; i < n; i++) {
    *mean += times[i];
    if (times[i] < *min)
      *min = times[i];
    if (times[i] > *max)
      *max = times[i];
  }
  *mean /= n;

  // Calculate standard deviation
  *stddev = 0.0;
  for (int i = 0; i < n; i++) {
    double diff = times[i] - *mean;
    *stddev += diff * diff;
  }
  *stddev = sqrt(*stddev / n);
}

// Test harness with performance measurement
int main() {
  printf("╔════════════════════════════════════════════════════════════╗\n");
  printf("║  GEMM 64×64×64 Performance Benchmark                      ║\n");
  printf("║  Baseline vs PolySat Optimized Schedule                   ║\n");
  printf("╚════════════════════════════════════════════════════════════╝\n\n");

  // Allocate matrices
  float(*A)[N] = malloc(sizeof(float[N][N]));
  float(*B)[N] = malloc(sizeof(float[N][N]));
  float(*C_baseline)[N] = malloc(sizeof(float[N][N]));
  float(*C_optimized)[N] = malloc(sizeof(float[N][N]));

  if (!A || !B || !C_baseline || !C_optimized) {
    fprintf(stderr, "Memory allocation failed\n");
    return 1;
  }

  // Initialize matrices with random-like values
  printf("Initializing matrices...\n");
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      A[i][j] = (float)(i * N + j) / (N * N);
      B[i][j] = (float)(i - j + N) / (N * N);
      C_baseline[i][j] = 0.0f;
      C_optimized[i][j] = 0.0f;
    }
  }

  printf("Matrix size: %d × %d\n", N, N);
  printf("Total operations: %d MACs (multiply-accumulate)\n", N * N * N);
  printf("Memory per matrix: %.2f KB\n\n", (N * N * sizeof(float)) / 1024.0);

  // Arrays to store timing results
  double baseline_times[BENCHMARK_RUNS];
  double optimized_times[BENCHMARK_RUNS];

  // ========== Baseline Performance ==========
  printf("─────────────────────────────────────────────────────────────\n");
  printf("Benchmarking BASELINE implementation...\n");
  printf("  Warmup runs: %d\n", WARMUP_RUNS);

  // Warmup
  for (int i = 0; i < WARMUP_RUNS; i++) {
    memset(C_baseline, 0, sizeof(float[N][N]));
    gemm_baseline(A, B, C_baseline);
  }

  // Actual benchmark
  printf("  Benchmark runs: %d\n", BENCHMARK_RUNS);
  for (int i = 0; i < BENCHMARK_RUNS; i++) {
    memset(C_baseline, 0, sizeof(float[N][N]));

    double start = get_time_ms();
    gemm_baseline(A, B, C_baseline);
    double end = get_time_ms();

    baseline_times[i] = end - start;
    printf("    Run %2d: %.4f ms\n", i + 1, baseline_times[i]);
  }

  // Calculate baseline statistics
  double baseline_mean, baseline_min, baseline_max, baseline_stddev;
  calculate_stats(baseline_times, BENCHMARK_RUNS, &baseline_mean, &baseline_min,
                  &baseline_max, &baseline_stddev);

  printf("\n  Statistics:\n");
  printf("    Mean:   %.4f ms\n", baseline_mean);
  printf("    Min:    %.4f ms\n", baseline_min);
  printf("    Max:    %.4f ms\n", baseline_max);
  printf("    StdDev: %.4f ms\n", baseline_stddev);
  printf("    GFLOPS: %.2f\n",
         (2.0 * N * N * N / 1e9) / (baseline_mean / 1000.0));

  // ========== Optimized Performance ==========
  printf("\n─────────────────────────────────────────────────────────────\n");
  printf("Benchmarking OPTIMIZED implementation (PolySat schedule)...\n");
  printf("  Warmup runs: %d\n", WARMUP_RUNS);

  // Warmup
  for (int i = 0; i < WARMUP_RUNS; i++) {
    memset(C_optimized, 0, sizeof(float[N][N]));
    gemm_optimized(A, B, C_optimized);
  }

  // Actual benchmark
  printf("  Benchmark runs: %d\n", BENCHMARK_RUNS);
  for (int i = 0; i < BENCHMARK_RUNS; i++) {
    memset(C_optimized, 0, sizeof(float[N][N]));

    double start = get_time_ms();
    gemm_optimized(A, B, C_optimized);
    double end = get_time_ms();

    optimized_times[i] = end - start;
    printf("    Run %2d: %.4f ms\n", i + 1, optimized_times[i]);
  }

  // Calculate optimized statistics
  double optimized_mean, optimized_min, optimized_max, optimized_stddev;
  calculate_stats(optimized_times, BENCHMARK_RUNS, &optimized_mean,
                  &optimized_min, &optimized_max, &optimized_stddev);

  printf("\n  Statistics:\n");
  printf("    Mean:   %.4f ms\n", optimized_mean);
  printf("    Min:    %.4f ms\n", optimized_min);
  printf("    Max:    %.4f ms\n", optimized_max);
  printf("    StdDev: %.4f ms\n", optimized_stddev);
  printf("    GFLOPS: %.2f\n",
         (2.0 * N * N * N / 1e9) / (optimized_mean / 1000.0));

  // ========== Correctness Verification ==========
  printf("\n─────────────────────────────────────────────────────────────\n");
  printf("Verifying correctness...\n");

  int errors = 0;
  float max_error = 0.0f;
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      float diff = fabsf(C_baseline[i][j] - C_optimized[i][j]);
      if (diff > max_error)
        max_error = diff;
      if (diff > 1e-4) {
        if (errors < 5) {
          printf("  Mismatch at [%d][%d]: baseline=%.6f, optimized=%.6f, "
                 "diff=%.6e\n",
                 i, j, C_baseline[i][j], C_optimized[i][j], diff);
        }
        errors++;
      }
    }
  }

  if (errors == 0) {
    printf("  ✓ All %d elements match (max error: %.2e)\n", N * N, max_error);
  } else {
    printf("  ✗ Found %d mismatches (max error: %.2e)\n", errors, max_error);
  }

  // ========== Performance Summary ==========
  printf("\n╔════════════════════════════════════════════════════════════╗\n");
  printf("║  Performance Summary                                       ║\n");
  printf("╚════════════════════════════════════════════════════════════╝\n\n");

  printf("Baseline (naive ijk):     %.4f ms (± %.4f ms)\n", baseline_mean,
         baseline_stddev);
  printf("Optimized (PolySat):      %.4f ms (± %.4f ms)\n", optimized_mean,
         optimized_stddev);
  printf("────────────────────────────────────────────────────────────\n");
  printf("Speedup:                  %.2fx\n", baseline_mean / optimized_mean);
  printf("Time reduction:           %.2f%%\n",
         100.0 * (baseline_mean - optimized_mean) / baseline_mean);
  printf("Performance gain:         %.2f GFLOPS\n",
         (2.0 * N * N * N / 1e9) * (1.0 / (optimized_mean / 1000.0) -
                                    1.0 / (baseline_mean / 1000.0)));

  printf("\nNote: This is measured on local CPU, not NCP hardware.\n");
  printf(
      "      The PolySat cost model predicts 159.82x on NCP architecture.\n");
  printf(
      "      Actual CPU speedup reflects cache/parallelism optimizations.\n");

  // Cleanup
  free(A);
  free(B);
  free(C_baseline);
  free(C_optimized);

  return errors > 0 ? 1 : 0;
}
