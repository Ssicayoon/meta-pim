// Gem5 driver for measuring GEMM kernel performance
// Uses gem5 m5ops to mark regions of interest
// All performance metrics come from gem5 stats.txt
#include <inttypes.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef GEM5_M5OPS
#  include <gem5/m5ops.h>
#else
// Stub implementations when not using gem5
static inline void m5_work_begin(uint64_t workid, uint64_t threadid) { (void)workid; (void)threadid; }
static inline void m5_work_end(uint64_t workid, uint64_t threadid) { (void)workid; (void)threadid; }
static inline void m5_dump_stats(uint64_t ns_delay, uint64_t ns_period) { (void)ns_delay; (void)ns_period; }
static inline void m5_reset_stats(uint64_t ns_delay, uint64_t ns_period) { (void)ns_delay; (void)ns_period; }
#endif

// MLIR memref descriptor for 2D array
typedef struct {
  int64_t *allocated;
  int64_t *aligned;
  int64_t offset;
  int64_t sizes[2];
  int64_t strides[2];
} memref2d_i64;

// Kernel inputs
typedef struct {
  int64_t alpha;
  int64_t beta;
  int64_t c[16];
  int64_t a[16];
  int64_t b[16];
} KernelInputs;

// Worker thread context
typedef struct {
  int thread_id;
  const KernelInputs *inputs;
  memref2d_i64 memref;
} WorkerContext;

// External kernel function (compiled from MLIR)
extern void gemm(
  int64_t arg0,  int64_t arg1,  int64_t arg2,  int64_t arg3,
  int64_t arg4,  int64_t arg5,  int64_t arg6,  int64_t arg7,
  int64_t arg8,  int64_t arg9,  int64_t arg10, int64_t arg11,
  int64_t arg12, int64_t arg13, int64_t arg14, int64_t arg15,
  int64_t arg16, int64_t arg17, int64_t arg18, int64_t arg19,
  int64_t arg20, int64_t arg21, int64_t arg22, int64_t arg23,
  int64_t arg24, int64_t arg25, int64_t arg26, int64_t arg27,
  int64_t arg28, int64_t arg29, int64_t arg30, int64_t arg31,
  int64_t arg32, int64_t arg33, int64_t arg34, int64_t arg35,
  int64_t arg36, int64_t arg37, int64_t arg38, int64_t arg39,
  int64_t arg40, int64_t arg41, int64_t arg42, int64_t arg43,
  int64_t arg44, int64_t arg45, int64_t arg46, int64_t arg47,
  int64_t arg48, int64_t arg49, memref2d_i64 arg50);

static void init_inputs(KernelInputs *inputs,
                        const int64_t C[4][4],
                        const int64_t A[4][4],
                        const int64_t B[4][4],
                        int64_t alpha,
                        int64_t beta) {
  inputs->alpha = alpha;
  inputs->beta = beta;
  for (int idx = 0; idx < 16; ++idx) {
    int r = idx / 4;
    int c = idx % 4;
    inputs->c[idx] = C[r][c];
    inputs->a[idx] = A[r][c];
    inputs->b[idx] = B[r][c];
  }
}

static void init_memref(memref2d_i64 *dst, int64_t *buffer) {
  dst->allocated = buffer;
  dst->aligned = buffer;
  dst->offset = 0;
  dst->sizes[0] = 1;
  dst->sizes[1] = 16;
  dst->strides[0] = 16;
  dst->strides[1] = 1;
}

// Note: We don't use rdtsc for performance measurement
// All performance metrics come from gem5 stats.txt
// rdtsc only measures CPU instruction cycles, not memory/cache latency

static void invoke_kernel(const KernelInputs *inputs, memref2d_i64 memref) {
  gemm(
    inputs->alpha, inputs->beta,
    inputs->c[0],  inputs->c[1],  inputs->c[2],  inputs->c[3],
    inputs->c[4],  inputs->c[5],  inputs->c[6],  inputs->c[7],
    inputs->c[8],  inputs->c[9],  inputs->c[10], inputs->c[11],
    inputs->c[12], inputs->c[13], inputs->c[14], inputs->c[15],
    inputs->a[0],  inputs->a[1],  inputs->a[2],  inputs->a[3],
    inputs->a[4],  inputs->a[5],  inputs->a[6],  inputs->a[7],
    inputs->a[8],  inputs->a[9],  inputs->a[10], inputs->a[11],
    inputs->a[12], inputs->a[13], inputs->a[14], inputs->a[15],
    inputs->b[0],  inputs->b[1],  inputs->b[2],  inputs->b[3],
    inputs->b[4],  inputs->b[5],  inputs->b[6],  inputs->b[7],
    inputs->b[8],  inputs->b[9],  inputs->b[10], inputs->b[11],
    inputs->b[12], inputs->b[13], inputs->b[14], inputs->b[15],
    memref);
}

static void *worker_thread(void *arg) {
  WorkerContext *ctx = (WorkerContext *)arg;
  
  // Mark kernel execution region for gem5 stats
  // gem5 will track all performance metrics (cycles, cache misses, etc.)
  m5_work_begin(0, ctx->thread_id);
  
  invoke_kernel(ctx->inputs, ctx->memref);
  
  m5_work_end(0, ctx->thread_id);
  
  return NULL;
}

int main(int argc, char *argv[]) {
  // HARDCODED_PARALLELISM: This value will be modified by the build script
  // Do not manually edit this line - it's automatically updated during compilation
  int parallelism = 1;  // PARALLELISM_PLACEHOLDER
  
  if (parallelism < 1 || parallelism > 256) {
    fprintf(stderr, "Parallelism must be between 1 and 256\n");
    return 1;
  }
  
  printf("Running GEMM with parallelism = %d (SIMD mode)\n", parallelism);
  
  const int M = 4, N = 4, K = 4;
  int64_t alpha = 32412;
  int64_t beta = 2123;
  
  // Allocate contexts and buffers
  pthread_t *threads = (pthread_t *)malloc(parallelism * sizeof(pthread_t));
  WorkerContext *contexts = (WorkerContext *)malloc(parallelism * sizeof(WorkerContext));
  int64_t (*buffers)[16] = (int64_t (*)[16])malloc(parallelism * 16 * sizeof(int64_t));
  
  // Allocate per-thread input data (SIMD mode: each thread has different data)
  KernelInputs *inputs_array = (KernelInputs *)malloc(parallelism * sizeof(KernelInputs));
  
  // Allocate reference results for verification
  int64_t (*Cref_array)[M][N] = (int64_t (*)[M][N])malloc(parallelism * sizeof(int64_t[M][N]));
  
  if (!threads || !contexts || !buffers || !inputs_array || !Cref_array) {
    fprintf(stderr, "Memory allocation failed\n");
    return 1;
  }
  
  // Initialize per-thread data (each thread gets unique input)
  printf("Initializing per-thread data...\n");
  for (int t = 0; t < parallelism; ++t) {
    // Each thread uses different input data (offset by thread_id)
    int64_t A[M][K], B[K][N], C[M][N];
    
    for (int i = 0; i < M; ++i) {
      for (int k = 0; k < K; ++k) {
        A[i][k] = (int64_t)(t * 1000 + i * K + k + 1);
      }
    }
    for (int k = 0; k < K; ++k) {
      for (int j = 0; j < N; ++j) {
        B[k][j] = (int64_t)(t * 1000 + k * N + j + 1);
      }
    }
    for (int i = 0; i < M; ++i) {
      for (int j = 0; j < N; ++j) {
        C[i][j] = (int64_t)(t * 1000 + (i + 1) * (j + 1));
      }
    }
    
    // Compute reference result for this thread
    memcpy(Cref_array[t], C, sizeof(C));
    for (int i = 0; i < M; ++i) {
      for (int j = 0; j < N; ++j) {
        int64_t cij = beta * Cref_array[t][i][j];
        for (int kk = 0; kk < K; ++kk) {
          cij += alpha * A[i][kk] * B[kk][j];
        }
        Cref_array[t][i][j] = cij;
      }
    }
    
    // Initialize inputs for this thread
    init_inputs(&inputs_array[t], C, A, B, alpha, beta);
    
    // Initialize context
    contexts[t].thread_id = t;
    contexts[t].inputs = &inputs_array[t];
    memset(buffers[t], 0, 16 * sizeof(int64_t));
    init_memref(&contexts[t].memref, buffers[t]);
  }
  
  // Flush cache to ensure cold start
  // Allocate and touch a large buffer to evict all cached data
  printf("Flushing cache...\n");
  size_t cache_flush_size = 32 * 1024 * 1024;  // 32 MB (larger than typical L3 cache)
  char *cache_flush_buffer = (char *)malloc(cache_flush_size);
  if (cache_flush_buffer) {
    // Touch every cache line to evict existing data
    for (size_t i = 0; i < cache_flush_size; i += 64) {
      cache_flush_buffer[i] = (char)i;
    }
    // Prevent compiler from optimizing away the loop
    volatile char sink = cache_flush_buffer[cache_flush_size - 1];
    (void)sink;
  }
  
  // Reset Gem5 stats before measurement
  printf("\n[Gem5] Resetting stats...\n");
  m5_reset_stats(0, 0);
  
  // Create threads
  for (int i = 0; i < parallelism; ++i) {
    int rc = pthread_create(&threads[i], NULL, worker_thread, &contexts[i]);
    if (rc != 0) {
      fprintf(stderr, "pthread_create failed for thread %d (rc=%d)\n", i, rc);
      return 1;
    }
  }
  
  // Wait for all threads
  for (int i = 0; i < parallelism; ++i) {
    pthread_join(threads[i], NULL);
  }
  
  // Dump Gem5 stats after measurement
  printf("[Gem5] Dumping stats...\n");
  m5_dump_stats(0, 0);
  
  // Verify results (each thread against its own reference)
  int mismatch_found = 0;
  for (int t = 0; t < parallelism; ++t) {
    int64_t *aligned = contexts[t].memref.aligned;
    for (int idx = 0; idx < 16; ++idx) {
      int r = idx / 4;
      int c = idx % 4;
      if (aligned[idx] != Cref_array[t][r][c]) {
        fprintf(stderr,
                "Thread %d mismatch at (%d,%d): got %" PRId64 " expected %" PRId64 "\n",
                t, r, c, aligned[idx], Cref_array[t][r][c]);
        mismatch_found = 1;
      }
    }
  }
  
  // Print results
  printf("\n");
  printf("╔════════════════════════════════════════════════════════════════╗\n");
  printf("║  GEMM Execution Complete (P=%d)                                \n", parallelism);
  printf("╚════════════════════════════════════════════════════════════════╝\n");
  printf("\n");
  printf("Performance metrics are available in gem5 stats.txt:\n");
  printf("  - Total cycles:       simTicks\n");
  printf("  - Cache misses:       dcache.overall_misses\n");
  printf("  - Memory bandwidth:   mem_ctrl.bytes_read/written\n");
  printf("  - CPU utilization:    cpu.numCycles\n");
  printf("\n");
  printf("To extract metrics, run:\n");
  printf("  grep 'simTicks\\|dcache\\|mem_ctrl' m5out/stats.txt\n");
  printf("\n");
  
  if (mismatch_found) {
    printf("❌ Test FAILED with mismatches\n");
  } else {
    printf("✅ Test PASSED\n");
  }
  
  // Write results to JSON for easy parsing
  FILE *fp = fopen("gem5_results.json", "w");
  if (fp) {
    fprintf(fp, "{\n");
    fprintf(fp, "  \"parallelism\": %d,\n", parallelism);
    fprintf(fp, "  \"test_passed\": %s,\n", mismatch_found ? "false" : "true");
    fprintf(fp, "  \"note\": \"Performance metrics are in gem5 stats.txt\"\n");
    fprintf(fp, "}\n");
    fclose(fp);
    printf("\nResults written to gem5_results.json\n");
  }
  
  // Cleanup
  free(threads);
  free(contexts);
  free(buffers);
  free(inputs_array);
  free(Cref_array);
  if (cache_flush_buffer) {
    free(cache_flush_buffer);
  }
  
  return mismatch_found ? 1 : 0;
}
