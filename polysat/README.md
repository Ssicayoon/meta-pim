# PolySat Artifact

PolySat is an automated framework for exploring and optimizing polyhedral schedules using equality saturation. It combines the precision of the polyhedral model with the search capabilities of e-graphs to discover high-performance loop transformations tailored for specific hardware architectures (e.g., Near-Data Processing).

## Prerequisites & Setup

### Environment Variables

PolySat requires **Polygeist** for C-to-MLIR compilation. Configure the environment before running:

1. **Copy the template**:
   ```bash
   cp .env.template .env
   ```

2. **Edit `.env`** (setup the POLYGEIST path and LLVM path):

3. **Source the environment** before running any examples:
   ```bash
   source .env
   ```

> **Note**: You must run `source .env` in each new shell session before using PolySat.

## Pipeline Design

The PolySat pipeline consists of four main stages:

1.  **Extraction**: 
    - Uses **Polygeist** to compile C kernels into MLIR.
    - Extracts the iteration domain and initial schedule into **ISL** (Integer Set Library) format.
    - Performs dependency analysis to ensure transformation legality.

2.  **Exploration**:
    - Represents the schedule space using an **E-graph**.
    - Applies rewrite rules (tiling, parallelism, reordering) to generate a vast space of equivalent schedules.
    - Uses **Equality Saturation** to efficiently manage the search space.

3.  **Cost Modeling**:
    - Evaluates candidates using a hardware-aware cost model.
    - Supports custom architectures (e.g., NCP) by modeling data movement, cache locality, and parallelism.

4.  **Code Generation**:
    - Selects the optimal schedule based on the cost model.
    - Generates optimized C code that implements the chosen schedule structure.

## Usage Example: GEMM Optimization

This artifact includes a complete demonstration of optimizing a General Matrix Multiplication (GEMM) kernel for an NCP (Near-Memory Computing) architecture.

**Important**: Make sure you've configured and sourced your environment first (see [Prerequisites & Setup](#prerequisites--setup)).

### 1. Generate Baseline Files
First, extract the baseline schedule and access patterns from the C source:

```bash
source .env  # Load environment variables
cargo run --example generate_gemm_64 # examples/demo/gemm_64.c
```

This creates the initial ISL files in `polysat_schedules/gemm_standard_64_isl/`.

### 2. Run Exploration
Run the PolySat explorer to find the optimal schedule:

```bash
source .env  # Load environment variables
cargo run --example explore_gemm_ncp_aware
```

This process will:
- Load the baseline schedule.
- Explore thousands of valid transformations (tiling, permutation, etc.).
- Rank schedules using the NCP-aware cost model.
- Output the best schedules to `output/gemm_standard_64/`.

### 3. Verify Results
The best schedule is analyzed and converted into a C implementation for verification.

- **Code**: See `examples/demo/polysat_opt_gemm_64.c` for the generated C implementation.

To verify correctness and measure local speedup:

```bash
# Compile the generated benchmark
clang -O3 -march=native examples/demo/polysat_opt_gemm_64.c -o polysat_opt_gemm_64 -lm 
# -fopenmp if you are using gcc instead of clang
# Run the benchmark
./polysat_opt_gemm_64

```

## Directory Structure

- `src/`: Core PolySat implementation (e-graph rules, cost models, ISL bindings).
- `examples/`: End-to-end optimization scripts (e.g., GEMM).
- `examples/demo/`: Input C Kernels and output optimized C file
- `benchmarks/`: Input C kernels.
- `polysat_schedules/`: Intermediate ISL extraction results.
