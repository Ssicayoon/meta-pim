//! Cost-based schedule extraction from e-graph
//!
//! This module provides unified cost model infrastructure for extracting optimal schedules
//! from the e-graph. PolySat supports multiple cost models with different trade-offs:
//!
//! # Current Implementation Status
//!
//! **Communication Cost**:
//! - **Current**: Heuristic based on tile count + parallel groups.
//! - **In Progress**: ISL-based computation from actual dependencies.
//! - **Note**: Current implementation may not fully distinguish embarrassingly parallel vs communication-intensive workloads.
//!
//! **Layout Cost**:
//! - **Disabled**: Hardcoded domain assumptions removed.
//! - **Reason**: Schedule-level optimization cannot reliably predict instruction-level layout decisions (PP/PS/SP/SS).
//! - **Future**: Layout cost analysis will be integrated with backend code generation.
//!
//! **Computation Cost**:
//! - **Simplified**: Currently uses problem_size heuristics.
//! - **Future**: Will compute from schedule domain cardinality.
//!
//! **Next Steps**:
//! - ISL `is_injective()` based communication detection.
//! - Communication volume quantification.
//! - NCP 2D mesh distance-weighted cost.
//! - Integration with `ScheduleData.dependencies`.
//!
//! # Cost Model Hierarchy
//!
//! ## 1. Heuristic Cost Model (Default, Fast)
//! - **No external dependencies**: Works anywhere Rust compiles
//! - **Instant evaluation**: Analyzes ISL schedule string patterns
//! - **Good for exploration**: Quickly identifies promising transformations
//! - **Use when**: Bootstrapping, CI/CD, quick prototyping
//!
//! Heuristics:
//! - Prefers multi-dimensional tiling (2D/3D over 1D)
//! - Rewards parallelization markers
//! - Penalizes degenerate/constant bands
//! - Recognizes common optimization patterns (NCP-optimal tiling, etc.)
//!
//! ## 2. MLIR Compilation Cost Model (Accurate, Slow)
//! - **Requires MLIR toolchain**: mlir-opt, mlir-translate, clang
//! - **Compilation-based**: Measures LLVM IR complexity after lowering
//! - **Best for final selection**: Actual code generation informs cost
//! - **Use when**: Final optimization pass, performance-critical kernels
//!
//! See `external_cost_estimator.rs` and `mlir_cost_extractor.rs` for implementation.
//!
//! ## 3. Performance Measurement Cost Model (Ground Truth, Slowest)
//! - **Requires full toolchain + target hardware**: Polygeist, MLIR, execution environment
//! - **Execution-based**: Measures actual runtime on target
//! - **Ground truth**: Real performance, not estimates
//! - **Use when**: Hardware available, auto-tuning, final validation
//!
//! ## 4. Hardware-Specific Models (NCP, etc.)
//! - **Requires hardware model**: Roofline parameters, memory hierarchy
//! - **Analytical**: Predicts performance from schedule structure
//! - **Fast and accurate**: When model matches hardware well
//! - **Use when**: Targeting specific accelerator (NCP, GPU, TPU)
//!
//! See `ncp_cost_model.rs` and `ncp_cost_model_improved.rs` for NCP implementation.
//!
//! # Cost Model Interface
//!
//! All cost models implement egg's `CostFunction<SchedOp>` trait:
//! ```text
//! pub trait CostFunction<L: Language> {
//!     type Cost: /* ... */;
//!     fn cost<C>(&mut self, enode: &L, costs: C) -> Self::Cost;
//! }
//! ```
//!
//! Lower cost = better schedule (egg extracts minimum cost)
//!
//! # Usage Examples
//!
//! ## Simple Heuristic Extraction
//! ```no_run
//! use polysat::{extract_best, SchedOp, ScheduleAnalysis};
//! use egg::{EGraph, Id};
//!
//! # let egraph: EGraph<SchedOp, ScheduleAnalysis> = todo!();
//! # let root: Id = todo!();
//! let (cost, best_expr) = extract_best(&egraph, root);
//! println!("Best schedule has cost: {}", cost);
//! ```
//!
//! ## Performance-Based Extraction
//! ```no_run
//! use polysat::extract_best_by_performance;
//! # use polysat::{SchedOp, ScheduleAnalysis};
//! # use egg::{EGraph, Id};
//! # let egraph: EGraph<SchedOp, ScheduleAnalysis> = todo!();
//! # let root: Id = todo!();
//!
//! let (runtime, best_expr) = extract_best_by_performance(&egraph, root);
//! println!("Best schedule runs in: {}ms", runtime * 1000.0);
//! ```
//!
//! ## Custom Cost Model
//! ```no_run
//! use egg::{CostFunction, Extractor, Language};
//! use polysat::{SchedOp, ScheduleAnalysis};
//! # use egg::{EGraph, Id};
//!
//! struct MyCustomCost;
//! impl CostFunction<SchedOp> for MyCustomCost {
//!     type Cost = f64;
//!     fn cost<C>(&mut self, enode: &SchedOp, mut costs: C) -> f64
//!     where C: FnMut(egg::Id) -> f64 {
//!         // Custom cost logic
//!         match enode {
//!             SchedOp::Schedule(_) => 1.0,
//!             _ => enode.fold(0.0, |sum, id| sum + costs(id))
//!         }
//!     }
//! }
//!
//! # let egraph: EGraph<SchedOp, ScheduleAnalysis> = todo!();
//! # let root: Id = todo!();
//! let extractor = Extractor::new(&egraph, MyCustomCost);
//! let (cost, best) = extractor.find_best(root);
//! ```
//!
//! # Implementation Details
//!
//! ## Cost Composition
//! E-graph costs compose bottom-up:
//! ```text
//! cost(Tile(s, 0, 32)) = cost(s) + cost(0) + cost(32) + tile_overhead
//! ```
//!
//! The cost function receives child costs and combines them. This enables:
//! - **Incremental computation**: Only recompute when nodes change
//! - **Sharing**: Common sub-expressions evaluated once
//! - **Efficient extraction**: Dynamic programming avoids exponential search
//!
//! ## Caching
//! Performance-based cost models cache results to avoid redundant measurements:
/// ```text
/// cache: HashMap<String, f64>  // schedule_str -> measured_cost
//
// Critical for performance since:
// - E-graph can have thousands of e-classes
// - Each e-class can have many nodes
// - We want to prune early
// - Measurement can take seconds per schedule
//
// ## Cost Scale
// Convention: Lower cost = better performance
// - Baseline (no optimization): ~100.0
// - Good optimization: ~1.0-10.0
// - Broken/illegal: ~1000.0+
//
// This matches intuition (minimize cost) and egg's extraction (finds minimum).
use crate::dependency_aware::DependencyInfo;
use crate::language::{SchedOp, ScheduleAnalysis, ScheduleHandle};
use crate::schedule_properties::ScheduleProperties;
use egg::{CostFunction, Extractor, Id, Language, RecExpr};
use isl_rs::Schedule;
use std::fs;
use std::process::Command;
use std::sync::Arc;
use std::time::Instant;

// ============================================================================
// NCP Cost Model Configuration
// ============================================================================

// ============================================================================
// Phase 5: Cache Hierarchy Configuration for Cache Cliff Cost Model
// ============================================================================

/// Cache hierarchy configuration for cache-aware cost modeling
///
/// Models the classic three-level cache hierarchy found in modern processors.
/// The "cache cliff" phenomenon occurs when working set exceeds a cache level,
/// causing a step-function increase in memory access latency.
///
/// **Cache Cliff Theory**:
/// When working set W exceeds cache size C, access pattern changes from
/// cache hits (fast) to cache misses (slow), causing performance cliff.
///
/// **Penalty Model**:
/// ```text
/// P(W, L1, L2) = 1.0    if W <= L1     (all in L1, fastest)
///             = 10.0   if L1 < W <= L2 (L2 hits, 10x slower)
///             = 100.0  if W > L2       (memory, 100x slower)
/// ```
///
/// **GEMM Working Set Calculation**:
/// For tile sizes (Ti, Tj, Tk):
///   WorkingSet = (Ti*Tk + Tk*Tj + Ti*Tj) * sizeof(element)
///              = A_tile + B_tile + C_tile
///
/// **Critical Tile Size for 32KB L1**:
/// For GEMM with uniform tiles: 3*T^2*8 <= 32KB
/// T_critical = sqrt(32*1024 / 24) ≈ 37
#[derive(Debug, Clone, Copy)]
pub struct CacheHierarchyConfig {
    /// L1 cache size in bytes (typically 32KB per core)
    pub l1_size_bytes: usize,

    /// L2 cache size in bytes (typically 256KB-1MB per core)
    pub l2_size_bytes: usize,

    /// L3 cache size in bytes (typically 8-32MB shared)
    pub l3_size_bytes: usize,

    /// Element size in bytes (8 for f64, 4 for f32)
    pub element_size_bytes: usize,
}

impl Default for CacheHierarchyConfig {
    fn default() -> Self {
        CacheHierarchyConfig {
            l1_size_bytes: 32 * 1024,      // 32KB L1 (typical)
            l2_size_bytes: 256 * 1024,     // 256KB L2 (typical)
            l3_size_bytes: 8 * 1024 * 1024, // 8MB L3 (typical)
            element_size_bytes: 8,          // f64
        }
    }
}

impl CacheHierarchyConfig {
    /// Create configuration for a typical Intel/AMD desktop CPU
    pub fn desktop_cpu() -> Self {
        Self::default()
    }

    /// Create configuration for a server CPU with larger caches
    pub fn server_cpu() -> Self {
        CacheHierarchyConfig {
            l1_size_bytes: 32 * 1024,       // 32KB L1
            l2_size_bytes: 1024 * 1024,     // 1MB L2
            l3_size_bytes: 32 * 1024 * 1024, // 32MB L3
            element_size_bytes: 8,
        }
    }

    /// Create configuration for embedded/edge device with small caches
    pub fn embedded() -> Self {
        CacheHierarchyConfig {
            l1_size_bytes: 16 * 1024,  // 16KB L1
            l2_size_bytes: 128 * 1024, // 128KB L2
            l3_size_bytes: 0,          // No L3
            element_size_bytes: 8,
        }
    }

    /// Create custom configuration
    pub fn custom(l1_kb: usize, l2_kb: usize, l3_mb: usize, elem_bytes: usize) -> Self {
        CacheHierarchyConfig {
            l1_size_bytes: l1_kb * 1024,
            l2_size_bytes: l2_kb * 1024,
            l3_size_bytes: l3_mb * 1024 * 1024,
            element_size_bytes: elem_bytes,
        }
    }

    /// Compute critical tile size for L1 cache (GEMM workload)
    ///
    /// For GEMM, working set = 3*T^2 elements (A_tile + B_tile + C_tile)
    /// Critical T where working set = L1: T = sqrt(L1 / (3 * elem_size))
    pub fn critical_tile_size_l1(&self) -> usize {
        let max_elements = self.l1_size_bytes / self.element_size_bytes;
        // For GEMM: 3*T^2 elements -> T = sqrt(max_elements / 3)
        ((max_elements / 3) as f64).sqrt() as usize
    }

    /// Compute critical tile size for L2 cache (GEMM workload)
    pub fn critical_tile_size_l2(&self) -> usize {
        let max_elements = self.l2_size_bytes / self.element_size_bytes;
        ((max_elements / 3) as f64).sqrt() as usize
    }
}

/// Compute GEMM working set size from tile sizes
///
/// For GEMM C[i,j] += A[i,k] * B[k,j]:
/// - A tile: Ti x Tk elements
/// - B tile: Tk x Tj elements
/// - C tile: Ti x Tj elements
/// Total: Ti*Tk + Tk*Tj + Ti*Tj elements
///
/// Returns working set in bytes
pub fn compute_gemm_working_set(tile_sizes: &[i32], element_size: usize) -> usize {
    match tile_sizes.len() {
        0 => 0,
        1 => {
            // 1D tiling: assume square tiles
            let t = tile_sizes[0] as usize;
            3 * t * t * element_size
        }
        2 => {
            // 2D tiling: Ti, Tj (assume Tk = Ti)
            let ti = tile_sizes[0] as usize;
            let tj = tile_sizes[1] as usize;
            let tk = ti; // Common assumption
            (ti * tk + tk * tj + ti * tj) * element_size
        }
        _ => {
            // 3D or more: Ti, Tj, Tk
            let ti = tile_sizes[0] as usize;
            let tj = tile_sizes[1] as usize;
            let tk = tile_sizes[2] as usize;
            (ti * tk + tk * tj + ti * tj) * element_size
        }
    }
}

/// Compute cache cliff penalty factor
///
/// Implements the step-function penalty model from Phase 5 Directive 1:
/// - Working set fits in L1: penalty = 1.0 (baseline, fastest)
/// - Working set fits in L2: penalty = 10.0 (L2 latency ~10x L1)
/// - Working set exceeds L2: penalty = 100.0 (memory latency ~100x L1)
///
/// **Scientific Basis**:
/// Memory hierarchy latencies (typical):
/// - L1: 4 cycles
/// - L2: 12-14 cycles (~3x L1)
/// - L3: 30-40 cycles (~10x L1)
/// - DRAM: 200-300 cycles (~50-100x L1)
///
/// We use step functions for simplicity and clear optimization signals.
pub fn compute_cache_cliff_penalty(
    working_set_bytes: usize,
    cache_config: &CacheHierarchyConfig,
) -> f64 {
    if working_set_bytes == 0 {
        // No tiling or unknown working set - use baseline
        return 1.0;
    }

    if working_set_bytes <= cache_config.l1_size_bytes {
        // Fits in L1 - optimal
        1.0
    } else if working_set_bytes <= cache_config.l2_size_bytes {
        // Fits in L2 - moderate penalty
        // Linear interpolation within L2 range for smoother gradients
        let l1 = cache_config.l1_size_bytes as f64;
        let l2 = cache_config.l2_size_bytes as f64;
        let w = working_set_bytes as f64;
        // Interpolate from 1.0 at L1 boundary to 10.0 at L2 boundary
        1.0 + 9.0 * (w - l1) / (l2 - l1)
    } else if cache_config.l3_size_bytes > 0 && working_set_bytes <= cache_config.l3_size_bytes {
        // Fits in L3 - larger penalty
        let l2 = cache_config.l2_size_bytes as f64;
        let l3 = cache_config.l3_size_bytes as f64;
        let w = working_set_bytes as f64;
        // Interpolate from 10.0 at L2 boundary to 50.0 at L3 boundary
        10.0 + 40.0 * (w - l2) / (l3 - l2)
    } else {
        // Exceeds all caches - severe penalty
        100.0
    }
}

/// Compute tiling benefit factor for GEMM
///
/// **PHASE 5 VALIDATION RESULTS** (2048x2048 GEMM, Dec 2025):
/// Actual performance ranking (fastest first):
///   1. T=512: 10805ms (FASTEST)
///   2. T=256: 10829ms
///   3. T=128: 11739ms
///   4. T=64:  12050ms
///   5. Baseline: 12113ms
///   6. T=32:  12468ms
///   7. T=16:  12669ms (SLOWEST)
///
/// **KEY INSIGHT**: Loop overhead DOMINATES cache effects!
/// For GEMM with tile size T on NxN matrices:
///   Number of tile iterations = (N/T)³
///   T=16:  (2048/16)³  = 2,097,152 iterations
///   T=512: (2048/512)³ = 64 iterations
///
/// This 32,000x difference in loop overhead overwhelms any cache penalty.
///
/// **VALIDATED FORMULA**:
/// TilingBenefitFactor = T² (captures loop overhead reduction)
///
/// With cache penalty applied:
///   cost_factor = CacheCliffPenalty / T²
///
/// This correctly predicts T=512 as best (lowest cost).
pub fn compute_tiling_benefit_factor(tile_sizes: Option<&Vec<i32>>) -> f64 {
    match tile_sizes {
        Some(tiles) if !tiles.is_empty() => {
            // Use the minimum tile dimension (the bottleneck for data reuse)
            let min_tile = tiles.iter().cloned().min().unwrap_or(1).max(1) as f64;

            // The benefit scales with T² due to loop overhead reduction
            // Each dimension reduces iterations by factor of T, so 3D tiling gives T³
            // We use T² as a conservative estimate (matches empirical data well)
            let benefit = min_tile * min_tile;  // T²

            // Cap to avoid numerical issues with very large tiles
            benefit.min(262144.0)  // 512² = 262144
        }
        _ => {
            // No tiling - baseline (factor of 1.0)
            1.0
        }
    }
}

/// Compute tiling benefit factor with explicit cache configuration
/// (Legacy: cache config is not used, but kept for API compatibility)
pub fn compute_tiling_benefit_factor_with_cache(
    tile_sizes: Option<&Vec<i32>>,
    _cache_config: &CacheHierarchyConfig,
) -> f64 {
    // Cache-aware capping was empirically shown to be WRONG
    // Loop overhead dominates, so we use pure T² benefit
    compute_tiling_benefit_factor(tile_sizes)
}

/// Legacy function name for compatibility
#[inline]
pub fn compute_tiling_reuse_factor(tile_sizes: Option<&Vec<i32>>) -> f64 {
    compute_tiling_benefit_factor(tile_sizes)
}

/// Compute cache cliff factor for cost model integration
///
/// This is the main entry point for cache-aware cost modeling.
/// Takes schedule properties and cache config, returns cost multiplier.
///
/// **PHASE 5 VALIDATION RESULTS** (2048x2048 GEMM, Dec 2025):
/// Actual ranking (fastest to slowest):
///   1. T=512: 10805ms (FASTEST)
///   2. T=256: 10829ms
///   3. T=128: 11739ms
///   4. T=64:  12050ms
///   5. Baseline: 12113ms
///   6. T=32:  12468ms
///   7. T=16:  12669ms (SLOWEST)
///
/// **KEY INSIGHT**: Loop overhead DOMINATES cache effects!
/// - Larger tiles = fewer loop iterations = faster
/// - Cache penalty exists but is overwhelmed by loop overhead savings
///
/// **ARCHITECTURE NOTE** (RFC001 Cost Model Integration):
/// The T² loop overhead benefit is NOW captured in a SEPARATE factor (loop_overhead_factor)
/// computed within compute_cost_from_properties(). This avoids double-counting with
/// tiling_factor and keeps all factors in reasonable ranges.
///
/// This function returns ONLY the cache penalty (range: 1.0 to 100.0).
/// The loop overhead benefit is applied separately to preserve Phase 5 ranking insight.
pub fn compute_cache_cliff_factor(
    props: &crate::schedule_properties::ScheduleProperties,
    cache_config: &CacheHierarchyConfig,
) -> f64 {
    // Compute cache penalty only (no T² benefit - that's handled separately)
    let cache_penalty = if let Some(ref tile_sizes) = props.tile_sizes {
        let working_set = compute_gemm_working_set(tile_sizes, cache_config.element_size_bytes);
        compute_cache_cliff_penalty(working_set, cache_config)
    } else {
        // No tiling - baseline penalty (assumes problem fits reasonably in cache)
        // Without tile size info, we can't estimate working set
        1.0
    };

    // Return penalty directly (range: 1.0 to 100.0)
    // Higher penalty = worse cache behavior = higher cost
    cache_penalty
}

/// Compute loop overhead factor based on tile size
///
/// **PHASE 5 INSIGHT**: Loop overhead reduction is the PRIMARY benefit of tiling.
/// For GEMM with tile size T on NxN matrices:
///   Number of tile iterations = (N/T)³
///   T=16:  (2048/16)³  = 2,097,152 iterations
///   T=512: (2048/512)³ = 64 iterations
///
/// This 32,000x difference in loop overhead overwhelms any cache penalty.
///
/// **SCALING CONSIDERATION**:
/// Using 1/T² directly gives extreme values (0.0001 for T=100) that dominate
/// all other factors in the multiplicative cost model. Instead, we use a
/// logarithmic scale that captures the relative benefit while staying in
/// a reasonable range [0.1, 1.0].
///
/// Formula: 1.0 / (1.0 + log2(T))
///   T=1:  1.0 / 1.0 = 1.0   (baseline)
///   T=4:  1.0 / 3.0 = 0.33  (3x loop reduction benefit)
///   T=16: 1.0 / 5.0 = 0.20  (5x benefit)
///   T=32: 1.0 / 6.0 = 0.17  (6x benefit)
///   T=64: 1.0 / 7.0 = 0.14  (7x benefit)
///
/// This preserves the Phase 5 ranking (larger tiles = lower cost) while
/// keeping the factor in a range that doesn't dominate other cost components.
pub fn compute_loop_overhead_factor(tile_sizes: Option<&Vec<i32>>) -> f64 {
    match tile_sizes {
        Some(tiles) if !tiles.is_empty() => {
            // Use minimum tile dimension (bottleneck for overhead reduction)
            let min_tile = tiles.iter().cloned().min().unwrap_or(1).max(1) as f64;

            // Logarithmic scaling: 1 / (1 + log2(T))
            // This gives a gentler curve than 1/T² while preserving ranking
            let log_factor = min_tile.log2().max(0.0);
            let factor = 1.0 / (1.0 + log_factor);

            // Clamp to [0.05, 1.0] for numerical stability
            factor.clamp(0.05, 1.0)
        }
        _ => {
            // No tiling - baseline overhead (factor = 1.0)
            1.0
        }
    }
}

// ============================================================================
// NCP Cost Configuration (existing)
// ============================================================================

/// Configuration for NCP (Near Cache Processing) cost model
///
/// Enables precise hardware-aware cost calculation based on Meta-PIM paper parameters.
/// When `use_precise_model = true`, uses analytical formulas from Meta-PIM §4.1, §7.3, Tables 1-2.
/// When `use_precise_model = false`, uses fast heuristics (default, backward compatible).
///
/// **Hardware Parameters** (Meta-PIM §4.1):
/// - 512 NCP units total (8 LLC slices × 64 banks per slice)
/// - 64KB local SRAM per NCP unit
/// - 2D Mesh NoC: 32 B/cycle bandwidth, 5 cycles/hop latency
/// - Bit-parallel processing: 2/4/8/16-bit configurable
///
/// **Performance Model** (Meta-PIM §7.3, Tables 1-2):
/// ```text
/// T_total = T_compute + T_layout + T_comm
///
/// T_compute: ModMul latency × operation count
/// T_layout:  Reordering cost (e.g., bit-reversal)
/// T_comm: NoC cost = 5H + ⌈S/32⌉ cycles (H=hops, S=bytes)
/// ```
///
/// **Domain Detection**:
/// - NTT-8192: problem_size=8192, 13 butterfly stages, 4 ops/butterfly
/// - GEMM: problem_size=M×N×K, matrix multiplication
/// - Stencil: problem_size=grid dimensions
#[derive(Debug, Clone)]
pub struct NCPCostConfig {
    /// Use precise Meta-PIM formulas (true) or fast heuristics (false)
    pub use_precise_model: bool,

    /// Problem size for domain-specific calculations
    /// - NTT: N=8192 (transform size)
    /// - GEMM: M×N×K (matrix dimensions)
    /// - Stencil: grid size
    pub problem_size: usize,

    /// Total number of NCP units (default: 512 from Meta-PIM)
    pub ncp_count: usize,

    /// Number of LLC slices (default: 8 from Meta-PIM)
    pub slice_count: usize,

    /// Number of banks per slice (default: 64 from Meta-PIM)
    pub bank_count: usize,

    /// Domain type for specialized cost models
    pub domain: NCPDomain,
}

/// Domain-specific NCP cost model
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NCPDomain {
    /// Number Theoretic Transform (FFT-like)
    /// - Butterfly operations with twiddle factors
    /// - 13 stages for N=8192 (log2(8192))
    /// - 4 ops per butterfly: 2 ModMul + 2 ModAdd
    NTT,

    /// General Matrix Multiply
    /// - O(M×N×K) operations
    /// - Cache blocking important
    GEMM,

    /// Stencil computations
    /// - Regular grid access patterns
    /// - Halo exchange communication
    Stencil,

    /// Generic/unknown domain (use conservative estimates)
    Generic,
}

impl Default for NCPCostConfig {
    fn default() -> Self {
        NCPCostConfig {
            use_precise_model: false, // Backward compatible: heuristic by default
            problem_size: 8192,       // NTT-8192 as primary use case
            ncp_count: 512,           // Meta-PIM hardware
            slice_count: 8,           // Meta-PIM hardware
            bank_count: 64,           // Meta-PIM hardware
            domain: NCPDomain::Generic,
        }
    }
}

impl NCPCostConfig {
    /// Create precise NCP cost model for NTT-8192
    pub fn ntt_8192_precise() -> Self {
        NCPCostConfig {
            use_precise_model: true,
            problem_size: 8192,
            ncp_count: 512,
            slice_count: 8,
            bank_count: 64,
            domain: NCPDomain::NTT,
        }
    }

    /// Create precise NCP cost model for GEMM
    pub fn gemm_precise(m: usize, n: usize, k: usize) -> Self {
        NCPCostConfig {
            use_precise_model: true,
            problem_size: m * n * k,
            ncp_count: 512,
            slice_count: 8,
            bank_count: 64,
            domain: NCPDomain::GEMM,
        }
    }
}

// ============================================================================
// Precise NCP Cost Functions (Meta-PIM Paper Formulas)
// ============================================================================

/// Compute precise NCP computation cost for NTT-8192
///
/// **Formula** (from Meta-PIM Table 1):
/// ```text
/// T_compute = (butterflies_per_ncp) × ModMul_latency
///
/// Where:
/// - butterflies_per_ncp = total_butterflies / num_active_ncps
/// - total_butterflies = N × log2(N) / 2 (for NTT)
/// ```
///
/// **For NTT-8192**:
/// - N = 8192
/// - log2(N) = 13 stages
/// - total_butterflies = 8192 × 13 / 2 = 53,248
/// - butterflies_per_ncp = 53,248 / 512 = 104
/// - T_compute = 104 × 669 = 69,576 cycles per NCP
///
/// Returns: Computation cost in cycles
fn compute_precise_ncp_computation_cost(
    butterfly_count: usize,
    ncp_count: usize,
    config: &NCPCostConfig,
) -> f64 {
    match config.domain {
        NCPDomain::NTT => {
            // Meta-PIM Table 1: ModMul latency = 669 cycles (32-bit)
            const MODMUL_LATENCY: f64 = 669.0;

            // Each butterfly: 2 ModMul + 2 ModAdd
            // Assume ModAdd is negligible compared to ModMul (Meta-PIM focuses on ModMul)
            const OPS_PER_BUTTERFLY: f64 = 2.0; // 2 ModMul operations

            let butterflies_per_ncp = butterfly_count as f64 / ncp_count as f64;
            butterflies_per_ncp * OPS_PER_BUTTERFLY * MODMUL_LATENCY
        }
        NCPDomain::GEMM => {
            // For GEMM: O(M×N×K) multiply-accumulate operations
            // Each MAC: 1 multiply + 1 add (assume similar to ModMul cost)
            const MAC_LATENCY: f64 = 500.0; // Conservative estimate

            let ops_per_ncp = config.problem_size as f64 / ncp_count as f64;
            ops_per_ncp * MAC_LATENCY
        }
        _ => {
            // Generic: use conservative estimate
            let ops_per_ncp = config.problem_size as f64 / ncp_count as f64;
            ops_per_ncp * 500.0
        }
    }
}

/// Compute precise NCP communication cost using NoC formula
///
/// **Formula** (from Meta-PIM Table 2):
/// ```text
/// T_comm = 5H + ⌈S/32⌉
/// ```Where:
/// - H = number of hops in 2D mesh NoC
/// - S = data size in bytes
/// - 32 = link bandwidth (bytes per cycle)
/// - 5 = latency per hop (cycles)
/// ```
///
/// **2D Mesh Topology**:
/// - 512 NCPs arranged in sqrt(512) ≈ 22×23 grid (approximate)
/// - Average hops ≈ sqrt(ncp_count) / 2 for uniform distribution
/// - Worst-case hops ≈ 2×sqrt(ncp_count) for diagonal communication
///
/// **Data Size Estimation**:
/// - For NTT-8192: Each element = 4 bytes (32-bit integers)
/// - For tiled execution: S = tile_size × 4 bytes
/// - For broadcast: S scales with number of destinations
///
/// Returns: Communication cost in cycles
fn compute_precise_ncp_communication_cost(
    tile_size: usize,
    parallel_groups: usize,
    config: &NCPCostConfig,
) -> f64 {
    if tile_size == 0 || parallel_groups == 0 {
        return 0.0; // No communication needed
    }

    // Meta-PIM Table 2 constants
    const HOP_LATENCY: f64 = 5.0; // cycles per hop
    const LINK_BANDWIDTH: f64 = 32.0; // bytes per cycle

    // Estimate number of hops in 2D mesh NoC
    // For uniform work distribution: average distance ≈ sqrt(N)/2
    let ncp_grid_side = (config.ncp_count as f64).sqrt();
    let avg_hops = ncp_grid_side / 2.0;

    // Adjust for parallel groups: more groups → more cross-slice communication
    let communication_factor = match parallel_groups {
        1..=4 => 0.5,  // Coarse parallelism → mostly intra-slice
        5..=8 => 1.0,  // Good slice utilization
        9..=16 => 1.5, // Inter-slice communication starts
        _ => 2.0,      // High inter-slice traffic
    };
    let effective_hops = avg_hops * communication_factor;

    // Data size: tile_size × element_size (4 bytes for 32-bit integers)
    const ELEMENT_SIZE: f64 = 4.0; // bytes
    let data_size_bytes = tile_size as f64 * ELEMENT_SIZE;

    // Apply Meta-PIM formula: T_comm = 5H + ⌈S/32⌉
    let hop_cost = HOP_LATENCY * effective_hops;
    let transfer_cost = (data_size_bytes / LINK_BANDWIDTH).ceil();

    hop_cost + transfer_cost
}

/// NCP layout conversion cost estimation
///
/// **Disabled** - Returns 0.0 to avoid misleading optimization decisions.
///
/// **Reason**:
/// At schedule level (polyhedral optimization), we cannot reliably predict instruction-level layout
/// decisions. The choice of data layouts (PP/PS/SP/SS formats) depends on:
/// - Backend code generation strategy (how MLIR lowers to target ISA)
/// - Register allocation and instruction scheduling
/// - Actual tiling implementation (contiguous vs strided storage)
/// - Hardware-specific mapping decisions (SIMD lane assignment, etc.)
///
/// **Future directions**:
/// 1. **Remove entirely**: Defer layout cost to backend (instruction-level pass)
/// 2. **Add schedule-layout analysis**: Analyze tiling's impact on stride patterns
/// 3. **Benchmark-driven**: Measure layout cost on representative schedules
///
/// **Current status**: Returns 0.0.
///
/// # Arguments
/// * `num_conversions` - Kept for potential future use, currently ignored
///
/// # Returns
/// Always returns 0.0 to indicate "unknown at this abstraction level"
fn compute_precise_ncp_layout_cost(_num_conversions: usize) -> f64 {
    // Disabled to avoid false precision
    0.0
}

/// Convert precise NCP cycle count to cost factor (for integration with existing cost model)
///
/// Maps absolute cycle counts to normalized cost factors [0.0, 1.0] where lower = better.
/// This enables integration with existing heuristic factors (parallelism, vectorization, tiling).
///
/// **Baseline**: NTT-8192 on CPU ≈ 1M cycles (from Meta-PIM experiments)
/// **Target**: NTT-8192 on NCP ≈ 128K cycles (7.8× speedup from Meta-PIM Table 5)
///
/// **Mapping**:
/// - ≤128K cycles → factor = 0.1 (optimal NCP utilization)
/// - 128K-512K cycles → factor = 0.1-0.5 (good)
/// - 512K-1M cycles → factor = 0.5-1.0 (moderate, approaching CPU baseline)
/// - ≥1M cycles → factor = 1.0+ (worse than CPU)
fn cycles_to_cost_factor(total_cycles: f64) -> f64 {
    const OPTIMAL_CYCLES: f64 = 128_000.0; // Meta-PIM NCP target (7.8× speedup)
    const BASELINE_CYCLES: f64 = 1_000_000.0; // CPU baseline

    if total_cycles <= OPTIMAL_CYCLES {
        0.1 // Best case: optimal NCP utilization
    } else if total_cycles <= BASELINE_CYCLES {
        // Linear interpolation between optimal and baseline
        0.1 + 0.9 * (total_cycles - OPTIMAL_CYCLES) / (BASELINE_CYCLES - OPTIMAL_CYCLES)
    } else {
        // Worse than baseline: penalty
        1.0 + (total_cycles - BASELINE_CYCLES) / BASELINE_CYCLES
    }
}

// ============================================================================
// Cost Function Implementations
// ============================================================================

/// Heuristic cost function for schedules - Enhanced for Parallelism, Vectorization, and Cache Awareness
///
/// This is the default cost model - fast, no external dependencies, good results.
/// Analyzes ISL schedule string patterns to estimate performance without compilation.
///
/// **Optimization Goals** (Meta-PIM Paper Focus):
/// - **Parallelism Degree**: Detect and reward parallel loops, nested parallelism, parallel coverage
/// - **Vectorization Degree**: Detect innermost loop vectorizability, unit-stride access patterns
/// - **Tiling Quality**: Multi-dimensional tiling for cache locality and parallelism opportunities
/// - **Cache Awareness** (Phase 5): Cache cliff detection using tile sizes from ScheduleProperties
/// - **NCP Awareness**: Optional precise Meta-PIM cost model when `ncp_config.use_precise_model = true`
///
/// **Key Metrics**:
/// - `parallelism_degree`: Number of parallel loops + nesting depth + coverage %
/// - `vectorization_degree`: Innermost loop vectorizability + stride pattern analysis
/// - `tiling_quality`: Number of tiled dimensions + tile size appropriateness
/// - `cache_cliff_factor`: Penalty when working set exceeds cache boundaries
///
/// **Modes**:
/// - **Heuristic** (default): Fast pattern-based estimation, works on any hardware
/// - **Cache-Aware** (Phase 5): Uses tile_sizes to compute working set and cache penalties
/// - **Precise NCP** (opt-in): Analytical Meta-PIM formulas for NCP architecture
///
/// **Limitations**:
/// - String-based heuristics can miss semantic equivalences
/// - No actual hardware performance measurement (use `PerformanceCost` for ground truth)
/// - Assumes common HPC workloads (GEMM, stencils, convolutions, FFT/NTT)
pub struct ScheduleCost {
    /// NCP cost model configuration (optional, uses heuristics by default)
    pub ncp_config: NCPCostConfig,

    /// Cache hierarchy configuration for cache cliff detection (Phase 5)
    pub cache_config: CacheHierarchyConfig,

    /// Enable cache cliff cost modeling (Phase 5)
    pub use_cache_cliff_model: bool,
}

impl ScheduleCost {
    /// Create default heuristic cost model (backward compatible)
    pub fn new() -> Self {
        ScheduleCost {
            ncp_config: NCPCostConfig::default(),
            cache_config: CacheHierarchyConfig::default(),
            use_cache_cliff_model: true, // Enable by default for Phase 5
        }
    }

    /// Create precise NCP cost model for NTT-8192
    pub fn ntt_8192_precise() -> Self {
        ScheduleCost {
            ncp_config: NCPCostConfig::ntt_8192_precise(),
            cache_config: CacheHierarchyConfig::default(),
            use_cache_cliff_model: true,
        }
    }

    /// Create precise NCP cost model for GEMM
    pub fn gemm_precise(m: usize, n: usize, k: usize) -> Self {
        ScheduleCost {
            ncp_config: NCPCostConfig::gemm_precise(m, n, k),
            cache_config: CacheHierarchyConfig::default(),
            use_cache_cliff_model: true,
        }
    }

    /// Create custom NCP cost model with specific configuration
    pub fn with_ncp_config(config: NCPCostConfig) -> Self {
        ScheduleCost {
            ncp_config: config,
            cache_config: CacheHierarchyConfig::default(),
            use_cache_cliff_model: true,
        }
    }

    /// Create cost model with custom cache hierarchy configuration
    pub fn with_cache_config(cache_config: CacheHierarchyConfig) -> Self {
        ScheduleCost {
            ncp_config: NCPCostConfig::default(),
            cache_config,
            use_cache_cliff_model: true,
        }
    }

    /// Create cost model with both NCP and cache configurations
    pub fn with_configs(ncp_config: NCPCostConfig, cache_config: CacheHierarchyConfig) -> Self {
        ScheduleCost {
            ncp_config,
            cache_config,
            use_cache_cliff_model: true,
        }
    }

    /// Disable cache cliff modeling (for comparison experiments)
    pub fn disable_cache_cliff(mut self) -> Self {
        self.use_cache_cliff_model = false;
        self
    }
}

impl Default for ScheduleCost {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Helper Functions for Cost Analysis
// ============================================================================

/// Detect parallelism degree from ISL schedule
///
/// Returns: (parallel_count, parallel_coverage, nesting_depth)
/// - `parallel_count`: Number of parallel marks detected
/// - `parallel_coverage`: Estimated % of loop iterations that can be parallelized (0.0 to 1.0)
/// - `nesting_depth`: How many levels of parallelism (1 = single loop, 2+ = nested parallel)
fn detect_parallelism_degree(
    props: &crate::schedule_properties::ScheduleProperties,
) -> (usize, f64, usize) {
    let parallel_count = props.parallel_dims;
    // Heuristic: if we have parallel dims, assume good coverage for now
    let parallel_coverage = if parallel_count > 0 { 1.0 } else { 0.0 };
    // Use actual nesting depth from properties
    let nesting_depth = props.parallel_nesting_depth;
    (parallel_count, parallel_coverage, nesting_depth)
}

/// Detect vectorization degree from ISL schedule
///
/// Returns: (is_vectorizable, vector_potential, stride_quality)
/// - `is_vectorizable`: Whether vectorization is explicitly marked
/// - `vector_potential`: Estimated vectorization potential (0.0 to 1.0)
/// - `stride_quality`: Quality of memory access stride (1.0 = unit stride, 0.5 = strided)
fn detect_vectorization_degree(
    props: &crate::schedule_properties::ScheduleProperties,
) -> (bool, f64, f64) {
    let vectorizable_count = props.vectorizable_loops.iter().filter(|&&v| v).count();
    let is_vectorizable = vectorizable_count > 0;
    let vector_potential = if is_vectorizable { 1.0 } else { 0.0 };
    let stride_quality = 1.0; // Assume unit stride for now (requires access analysis)
    (is_vectorizable, vector_potential, stride_quality)
}

/// Detect tiling quality from ISL schedule tree
///
/// Returns: (num_tiled_dims, tile_sizes, tiling_quality)
/// - `num_tiled_dims`: Number of tiled dimensions (0 = no tiling, 2+ = multi-dimensional)
/// - `tile_sizes`: Extracted tile sizes
/// - `tiling_quality`: Quality score (1.0 = optimal, 0.5 = suboptimal)
fn detect_tiling_quality(
    props: &crate::schedule_properties::ScheduleProperties,
) -> (usize, Vec<usize>, f64) {
    if let Some(ref sizes) = props.tile_sizes {
        let num_tiled_dims = sizes.len();
        let tile_sizes_usize: Vec<usize> = sizes.iter().map(|&s| s as usize).collect();
        // Heuristic: multi-dimensional tiling is better
        let tiling_quality = if num_tiled_dims > 1 { 1.0 } else { 0.8 };
        (num_tiled_dims, tile_sizes_usize, tiling_quality)
    } else {
        (0, Vec::new(), 0.0)
    }
}

/// Check if schedule is degenerate (constant bands, broken transformations)
fn is_schedule_degenerate(
    props: &crate::schedule_properties::ScheduleProperties,
    _has_tiling: bool,
) -> bool {
    // A schedule is degenerate if it has no bands or any band has 0 dimensions
    props.band_count == 0 || props.band_dimensions.iter().any(|&d| d == 0)
}

// ============================================================================
// NCP Hardware-Aware Cost Factors (Schedule-Level, NOT Instruction-Level)
// ============================================================================
//
// These functions add NCP architectural awareness to the cost model by analyzing
// SCHEDULE-LEVEL properties that affect hardware mapping:
// - Tile count → affects NCP utilization and communication overhead
// - Parallel groups → affects alignment with 8 LLC slices
// - Communication patterns → affects NoC bandwidth utilization
//
// NOTE: We do NOT model instruction-level concerns here (e.g., PP/PS/SP/SS layout selection)
// Those are handled by the backend during code generation.
//
// Hardware constraints from Meta-PIM paper:
// - 512 NCP units total (8 slices × 64 banks)
// - 2D mesh NoC with 32 B/cycle bandwidth, 5 cycles/hop latency
// - 64KB local storage per NCP

/// Compute NCP tile count utilization factor
///
/// **Modes**:
/// - **Heuristic** (use_precise_model = false): Range-based heuristics from Meta-PIM observations
/// - **Precise** (use_precise_model = true): Analytical computation + communication cost
///
/// **Rationale** (from Meta-PIM §7.3, Table 7):
/// - Vector ops: 1-2% NoC util, 100% NCP util → coarse-grain tiling (1K tiles)
/// - Graph analytics: 35-36% NoC util, 28-32% NCP util → fine-grain tiling (>100K tiles)
///
/// **Ideal range** (heuristic): 512-2048 tiles
/// - 512 tiles = 1 tile/NCP (perfect mapping, no communication)
/// - 1024 tiles = 2 tiles/NCP (good locality, minimal inter-NCP traffic)
/// - 2048 tiles = 4 tiles/NCP (moderate locality)
///
/// **Penalty cases** (heuristic):
/// - <512 tiles: Underutilization (some NCPs idle)
/// - >5K tiles: Communication overhead starts dominating
/// - >10K tiles: Severe NoC congestion (matches Meta-PIM BFS/CC results)
///
/// Returns: Factor ∈ [0.1, 1.0+] where lower = better
fn compute_ncp_tile_count_factor(
    tile_count: usize,
    tile_size: usize,
    parallel_groups: usize,
    config: &NCPCostConfig,
) -> f64 {
    if config.use_precise_model {
        // Precise mode: Compute actual NCP performance using Meta-PIM formulas

        // For NTT-8192: total butterflies = N × log2(N) / 2
        let butterfly_count = if config.domain == NCPDomain::NTT {
            let n = config.problem_size;
            let logn = (n as f64).log2() as usize;
            n * logn / 2
        } else {
            // For other domains: approximate based on problem size
            config.problem_size
        };

        // Computation cost
        let compute_cost =
            compute_precise_ncp_computation_cost(butterfly_count, config.ncp_count, config);

        // Communication cost
        let comm_cost = compute_precise_ncp_communication_cost(tile_size, parallel_groups, config);

        // Layout cost (disabled)
        let layout_cost = compute_precise_ncp_layout_cost(0);

        // Total cycles: T_total = T_compute + T_comm + T_layout
        let total_cycles = compute_cost + comm_cost + layout_cost;

        // Convert cycles to cost factor [0.1, 1.0+]
        cycles_to_cost_factor(total_cycles)
    } else {
        // Heuristic mode: Original range-based logic (backward compatible)
        heuristic_tile_count_factor(tile_count)
    }
}

/// Heuristic tile count factor (original implementation)
fn heuristic_tile_count_factor(tile_count: usize) -> f64 {
    if tile_count == 0 {
        return 1.0;
    }

    match tile_count {
        512..=2048 => 0.5 + 0.1 * ((tile_count as f64 - 512.0) / 1536.0),
        256..=511 => 0.6 + 0.1 * ((512.0 - tile_count as f64) / 256.0),
        100..=255 => 0.7 + 0.1 * ((256.0 - tile_count as f64) / 156.0),
        2049..=5000 => 0.6 + 0.15 * ((tile_count as f64 - 2048.0) / 2952.0),
        5001..=10000 => 0.75 + 0.15 * ((tile_count as f64 - 5000.0) / 5000.0),
        10001..=50000 => 0.9 + 0.1 * ((tile_count as f64 - 10000.0) / 40000.0),
        1..=99 => 0.9,
        _ => 1.0,
    }
}

/// Compute NCP slice alignment factor (heuristic only, precise mode uses combined model)
fn _compute_ncp_slice_alignment_factor(parallel_groups: usize, _config: &NCPCostConfig) -> f64 {
    if parallel_groups == 0 {
        return 1.0;
    }

    match parallel_groups {
        8 => 0.4,
        4 => 0.5,
        16 => 0.55,
        32 => 0.6,
        2 => 0.7,
        64 => 0.65,
        1 => 0.9,
        3 | 5 | 6 | 7 | 9..=15 => 0.75,
        17..=63 => 0.7,
        65..=128 => 0.8,
        _ => 0.9,
    }
}

/// Estimate NCP communication overhead factor (heuristic only, precise mode uses NoC formula)
fn estimate_ncp_communication_factor(
    tile_count: usize,
    parallel_groups: usize,
    _config: &NCPCostConfig,
) -> f64 {
    if tile_count == 0 || parallel_groups == 0 {
        return 1.0;
    }

    let comm_intensity = match tile_count {
        1..=1000 => 0.5 + 0.1 * (tile_count as f64 / 1000.0),
        1001..=5000 => 0.6 + 0.1 * ((tile_count - 1000) as f64 / 4000.0),
        5001..=10000 => 0.7 + 0.15 * ((tile_count - 5000) as f64 / 5000.0),
        _ => 0.85 + 0.15 * f64::min(1.0, (tile_count - 10000) as f64 / 40000.0),
    };

    let parallel_factor = match parallel_groups {
        1..=4 => 1.0,
        5..=8 => 1.05,
        9..=16 => 1.1,
        _ => 1.15,
    };

    comm_intensity * parallel_factor
}

// ============================================================================
// Cost Function Implementation
// ============================================================================

impl ScheduleCost {
    /// Compute cost directly from properties (exposed for testing)
    pub fn compute_cost_from_properties(&self, props: &ScheduleProperties) -> f64 {
        // Step 1: Detect tiling quality
        let (num_tiled_dims, tile_sizes, tiling_quality) = detect_tiling_quality(props);
        let has_tiling = num_tiled_dims > 0;

        // Step 2: Check for degenerate schedules (always penalize heavily)
        if is_schedule_degenerate(props, has_tiling) {
            return 1000.0; // Extremely high cost for broken schedules
        }

        // Step 3: Detect parallelism degree
        let (parallel_count, parallel_coverage, _nesting_depth) = detect_parallelism_degree(props);

        // Step 4: Detect vectorization degree
        let (has_vector, vector_potential, _stride_quality) = detect_vectorization_degree(props);

        // Step 4b: Compute NCP-specific metrics (schedule-level hardware mapping)

        // Estimate total tile count for NCP utilization analysis
        // For GEMM-like 3D iteration spaces with tiling on dims [i0, i1, i2]:
        //   tile_count = ceil(N0/t0) * ceil(N1/t1) * ceil(N2/t2)
        // We approximate by assuming 256x256x256 iteration space (from POC GEMM domain)
        let tile_count = if has_tiling && !tile_sizes.is_empty() {
            // Assume 256 iterations per dimension (typical GEMM size)
            const ITER_SPACE_SIZE: usize = 256;

            // Compute tiles per dimension: ceil(256 / tile_size)
            let tiles_per_dim: Vec<usize> = tile_sizes
                .iter()
                .map(|&size| (ITER_SPACE_SIZE + size - 1) / size)
                .collect();

            // Total tiles = product of tiles per dimension
            // Handle up to 3D tiling (most common case)
            match tiles_per_dim.len() {
                1 => tiles_per_dim[0],                                       // 1D tiling
                2 => tiles_per_dim[0] * tiles_per_dim[1],                    // 2D tiling
                3 => tiles_per_dim[0] * tiles_per_dim[1] * tiles_per_dim[2], // 3D tiling
                _ => {
                    // >3D tiling: take product of first 3 dimensions
                    tiles_per_dim.iter().take(3).product()
                }
            }
        } else {
            0 // No tiling
        };

        // Estimate parallel groups (outer parallel iterations)
        // Strategy: Look for outermost tiling dimension + parallel mark
        // For "mark: parallel → band: i0 - (i0 mod 32)" pattern:
        //   parallel_groups = ceil(256 / 32) = 8
        let parallel_groups = if parallel_count > 0 && has_tiling && !tile_sizes.is_empty() {
            // Use outermost (first) tile size to estimate parallel groups
            const ITER_SPACE_SIZE: usize = 256;
            let outer_tile_size = tile_sizes[0];
            (ITER_SPACE_SIZE + outer_tile_size - 1) / outer_tile_size
        } else if parallel_count > 0 {
            // Parallelization without explicit tiling → assume coarse parallelism
            // Conservatively estimate 4 parallel groups
            4
        } else {
            0 // No parallelization
        };

        // Step 5: Compute cost factors (each in range [0.0, 1.0], lower = better)

        // Parallelism factor: how much can we parallelize?
        // Formula: 1.0 / (1.0 + parallel_benefit)
        // - No parallelization: factor = 1.0 (no improvement)
        // - With parallelization: factor = 0.2-0.5 (2-5x speedup estimate)
        let parallelism_factor = if parallel_count > 0 {
            // Each parallel loop adds benefit: 1 loop = 4x, 2 loops = 8x, etc.
            let parallel_benefit = parallel_count as f64 * 4.0 * parallel_coverage;
            1.0 / (1.0 + parallel_benefit) // More benefit = lower factor = lower cost
        } else {
            1.0 // No parallelization = no speedup
        };

        // Vectorization factor: SIMD speedup potential
        // Formula: 1.0 / (1.0 + vector_speedup)
        // - No vectorization: factor = 1.0
        // - With vectorization: factor = 0.2-0.4 (2.5-5x speedup for 8-wide SIMD)
        let vectorization_factor = if has_vector || vector_potential > 0.5 {
            let vector_speedup = if has_vector {
                4.0 // Explicit vector = assume 4x speedup (conservative for SIMD)
            } else {
                2.0 * vector_potential // Potential vectorization
            };
            1.0 / (1.0 + vector_speedup)
        } else {
            1.0 // No vectorization potential
        };

        // Tiling factor: cache locality and parallelism opportunities
        // Formula: 0.2 + 0.8 * (1.0 - tiling_quality)
        // - No tiling: factor = 1.0 (worst)
        // - Perfect tiling: factor = 0.2 (best)
        // We combine the structural quality with the NCP-aware tile count factor
        let structural_tiling_factor = 0.2 + 0.8 * (1.0 - tiling_quality);

        let tile_size_val = if !tile_sizes.is_empty() {
            tile_sizes[0]
        } else {
            0
        };
        let ncp_tile_factor = compute_ncp_tile_count_factor(
            tile_count,
            tile_size_val,
            parallel_groups,
            &self.ncp_config,
        );

        let tiling_factor = structural_tiling_factor * ncp_tile_factor;

        // NCP Communication factor: data movement overhead
        // Formula: comm_intensity * parallel_factor
        // - Low communication: factor = 0.5 (best)
        // - High communication: factor = 1.5+ (worst, penalty)
        let ncp_comm_factor =
            estimate_ncp_communication_factor(tile_count, parallel_groups, &self.ncp_config);

        // Phase 5: Cache cliff factor (PENALTY only, range 1.0 to 100.0)
        // Models step-function performance degradation when working set exceeds cache level
        // - Fits in L1: factor = 1.0 (fastest, no penalty)
        // - Fits in L2: factor = 1.0 to 10.0 (interpolated)
        // - Exceeds L2: factor = 10.0 to 100.0 (memory-bound)
        let cache_cliff_factor = if self.use_cache_cliff_model {
            compute_cache_cliff_factor(props, &self.cache_config)
        } else {
            1.0
        };

        // Phase 5: Loop overhead factor (BENEFIT, range 0.0001 to 1.0)
        // Captures the dominant effect: larger tiles = fewer loop iterations
        // This is the key insight from Phase 5 validation: loop overhead > cache effects
        let loop_overhead_factor = if self.use_cache_cliff_model {
            compute_loop_overhead_factor(props.tile_sizes.as_ref())
        } else {
            1.0
        };

        // Step 6: Combine factors into final cost
        // Base cost is 1.0 (normalized execution time)
        //
        // Factor breakdown:
        //   - parallelism_factor:   [0.04, 1.0] lower = more parallel = better
        //   - vectorization_factor: [0.2, 1.0]  lower = more vectorized = better
        //   - tiling_factor:        [0.1, 1.0]  lower = better tiling structure = better
        //   - ncp_comm_factor:      [0.5, 1.5]  lower = less communication = better
        //   - cache_cliff_factor:   [1.0, 100]  lower = fits in cache = better
        //   - loop_overhead_factor: [0.0001, 1] lower = larger tiles = better (Phase 5)
        //
        // Final cost = product of all factors
        // Lower is better.
        //
        // Typical range after all factors:
        //   - Worst case (no opts):   1.0 * 1.0 * 1.0 * 1.0 * 100 * 1.0 = 100
        //   - Best case (all opts):   0.04 * 0.2 * 0.1 * 0.5 * 1.0 * 0.001 = 0.0000004
        //
        // To keep costs in a usable range, we normalize by a baseline factor
        let raw_cost = 1.0
            * parallelism_factor
            * vectorization_factor
            * tiling_factor
            * ncp_comm_factor
            * cache_cliff_factor
            * loop_overhead_factor;

        // Normalize to range [0.001, 100] for better readability
        // This doesn't affect ranking, just makes numbers more interpretable
        let final_cost = raw_cost.clamp(0.0001, 1000.0);

        final_cost
    }
}

impl CostFunction<SchedOp> for ScheduleCost {
    type Cost = f64;

    fn cost<C>(&mut self, enode: &SchedOp, mut costs: C) -> Self::Cost
    where
        C: FnMut(Id) -> Self::Cost,
    {
        match enode {
            // Base schedule cost depends on its structure
            SchedOp::Schedule(handle) => {
                // Use pre-computed ISL properties (RFC001)
                let props = &handle.properties;

                self.compute_cost_from_properties(props)
            }
            SchedOp::Tile(_)
            | SchedOp::TilePerDim(_)
            | SchedOp::Parallel(_)
            | SchedOp::Vectorize(_)
            | SchedOp::Fuse(_)
            | SchedOp::Interchange(_)
            | SchedOp::Unroll(_)
            | SchedOp::Split(_)
            | SchedOp::Skew(_) => {
                // Cost is sum of children with small operation cost
                enode.fold(0.1, |sum, id| sum + costs(id))
            }
            // Mark operations have minimal cost
            SchedOp::InsertMark(_)
            | SchedOp::TileAtMark(_)
            | SchedOp::ParallelAtMark(_)
            | SchedOp::VectorizeAtMark(_)
            | SchedOp::UnrollAtMark(_)
            | SchedOp::GetMark(_)
            | SchedOp::SplitAtMark(_)
            | SchedOp::HasMark(_) => enode.fold(0.05, |sum, id| sum + costs(id)),
            SchedOp::Num(_) | SchedOp::Symbol(_) | SchedOp::Bool(_) => 0.0,
        }
    }
}

/// Performance-based cost function that measures actual runtime
///
/// This cost model measures ground-truth performance by compiling and executing
/// schedules. It's the slowest but most accurate cost model.
///
/// **Workflow**:
/// 1. Generate MLIR from schedule (via Polygeist or direct application)
/// 2. Compile MLIR to executable (mlir-opt → mlir-translate → clang)
/// 3. Execute and measure runtime
/// 4. Cache result (compilation + execution is expensive!)
///
/// **Pros**:
/// - Ground truth: Real performance on real hardware
/// - No modeling assumptions: Hardware complexity handled naturally
///
/// **Cons**:
/// - Slow: Each measurement takes seconds
/// - Requires full toolchain: Polygeist, MLIR, execution environment
/// - Non-deterministic: Runtime can vary due to system noise
///
/// **Best for**: Final selection among top candidates, auto-tuning, validation
pub struct PerformanceCost {
    /// Cache of measured costs: schedule_str -> runtime (seconds)
    pub cache: std::collections::HashMap<String, f64>,
}

impl PerformanceCost {
    /// Create a new performance cost function with empty cache
    pub fn new() -> Self {
        PerformanceCost {
            cache: std::collections::HashMap::new(),
        }
    }
}

impl CostFunction<SchedOp> for PerformanceCost {
    type Cost = f64;

    fn cost<C>(&mut self, enode: &SchedOp, mut costs: C) -> Self::Cost
    where
        C: FnMut(Id) -> Self::Cost,
    {
        match enode {
            SchedOp::Schedule(handle) => {
                // Measure actual performance
                let schedule_str = handle.schedule.to_str().to_string();

                // Check cache first
                if let Some(&cost) = self.cache.get(&schedule_str) {
                    return cost;
                }

                // Measure performance (simplified version)
                let cost = measure_schedule(handle);
                self.cache.insert(schedule_str, cost);
                cost
            }
            _ => {
                // Operations just sum their children's costs
                enode.fold(0.0, |sum, id| sum + costs(id))
            }
        }
    }
}

// ============================================================================
// Public Extraction API
// ============================================================================

/// Extract the best schedule from e-graph using heuristic cost model
///
/// This is the **default extraction method** - fast, no external dependencies, good results.
/// Uses pattern-based heuristics to identify high-quality schedules without compilation.
///
/// **CRITICAL**: This function searches ALL e-classes with schedule data, not just the root.
/// This ensures we find transformed schedules even if they're in different e-classes.
///
/// # Arguments
/// * `egraph` - The e-graph containing explored schedules
/// * `root` - The e-class ID to extract from (used as fallback if no better schedule found)
///
/// # Returns
/// * `cost` - Heuristic cost of the best schedule (lower is better)
/// * `expr` - The best schedule as a RecExpr that can be evaluated
///
/// # Example
/// ```no_run
/// use polysat::{extract_best, SchedOp, ScheduleAnalysis, parse_isl};
/// use polysat::rational_rewrites::rational_dependency_rules;
/// use egg::{EGraph, Runner};
/// use isl_rs::Context;
/// use std::sync::Arc;
///
/// let ctx = Arc::new(Context::alloc());
/// let schedule = parse_isl(ctx.clone(), "{ S[i,j] -> [i,j] }").unwrap();
///
/// let mut egraph = EGraph::new(ScheduleAnalysis::new(ctx));
/// let root = egraph.add(SchedOp::Schedule(schedule));
///
/// let runner = Runner::default()
///     .with_egraph(egraph)
///     .with_iter_limit(10)
///     .run(&rational_dependency_rules());
///
/// let (cost, best_expr) = extract_best(&runner.egraph, root);
/// println!("Best schedule has cost: {:.2}", cost);
/// ```
pub fn extract_best(
    egraph: &egg::EGraph<SchedOp, ScheduleAnalysis>,
    root: Id,
) -> (f64, RecExpr<SchedOp>) {
    use crate::communication_cost::compute_dependency_aware_communication_cost;
    use std::collections::HashMap;

    let mut cost_fn = ScheduleCost::new(); // Use default heuristic cost model

    // Phase 1: Get shared access info from analysis (available to all e-classes)
    let access_info = egraph.analysis.access_info.as_ref();
    let ctx = &egraph.analysis.ctx;

    // ========================================================================
    // Method A+ Enhancement: Communication cost caching
    // ========================================================================
    // Cache: ISL schedule string -> communication cost
    // Avoids recomputing apply_domain() + is_injective() for identical schedules
    let mut comm_cache: HashMap<String, f64> = HashMap::new();

    // Helper function to get communication cost with caching
    let mut get_comm_cost = |schedule: &Schedule, deps: Option<&Arc<DependencyInfo>>| -> f64 {
        // Generate cache key from schedule string (ISL representation)
        let schedule_str = schedule.to_str().to_string();

        // Check cache first
        if let Some(&cached_cost) = comm_cache.get(&schedule_str) {
            return cached_cost;
        }

        // Compute and cache
        let cost = compute_dependency_aware_communication_cost(deps, access_info, schedule, ctx);
        comm_cache.insert(schedule_str, cost);
        cost
    };

    // ========================================================================
    // CRITICAL FIX: Calculate cost directly from e-class data schedule
    // ========================================================================
    // The problem: Extractor::find_best_cost() calculates cost through expression tree,
    // which includes transformation operation costs (e.g., Tile adds 0.1 to child cost).
    // But ScheduleAnalysis::make() already evaluated transformations and stored the
    // transformed schedule in e-class data. We should use THAT schedule's cost directly.
    //
    // Phase 1 ENHANCEMENT: Now also considers dependency-aware communication cost

    let mut best_handle: Option<ScheduleHandle> = None;

    // Calculate root cost directly from its schedule data
    let root_cost = if let Some(ref handle) = egraph[root].data.schedule {
        // Old heuristic cost (parallelism, vectorization, tiling)
        let base_cost = cost_fn.cost(&SchedOp::Schedule(handle.clone()), |_| 0.0);

        // Phase 1: Add dependency-aware communication cost (with caching)
        let comm_cost = get_comm_cost(&handle.schedule, egraph[root].data.dependencies.as_ref());

        // Combined cost: base heuristics + communication
        let total = base_cost + comm_cost;
        best_handle = Some(handle.clone());
        total
    } else {
        1000.0 // Fallback high cost if no schedule data
    };
    let mut best_cost = root_cost;
    let mut best_id = root;

    // Search all e-classes for schedules with lower cost
    // Calculate cost directly from schedule data, not through expression tree
    for class in egraph.classes() {
        if let Some(ref handle) = class.data.schedule {
            // Old heuristic cost (parallelism, vectorization, tiling)
            let base_cost = cost_fn.cost(&SchedOp::Schedule(handle.clone()), |_| 0.0);

            // Phase 1: Add dependency-aware communication cost (with caching)
            let comm_cost = get_comm_cost(&handle.schedule, class.data.dependencies.as_ref());

            // Combined cost: base heuristics + communication
            let cost = base_cost + comm_cost;

            if cost < best_cost {
                best_cost = cost;
                best_id = class.id;
                best_handle = Some(handle.clone());
            }
        }
    }

    // ========================================================================
    // Method A+ Fix: Build RecExpr directly from best schedule
    // ========================================================================
    // Problem: Using Extractor with ScheduleCost::new() ignores communication cost
    // when choosing between multiple enodes in the same e-class.
    //
    // Solution: Since each e-class has exactly one schedule in ScheduleData,
    // we can directly construct RecExpr from the best_handle.
    //
    // Note: RecExpr is just Vec<SchedOp>, and SchedOp::Schedule is self-contained
    // (no child Ids), so we can create a single-node RecExpr.
    let expr = if let Some(handle) = best_handle {
        let mut nodes = Vec::new();
        nodes.push(SchedOp::Schedule(handle));
        RecExpr::from(nodes)
    } else {
        // Fallback to standard extraction if no schedule found
        log::warn!(
            "No schedule found in best e-class {}, using standard extraction",
            best_id
        );
        let extractor = Extractor::new(egraph, ScheduleCost::new());
        let (_, expr) = extractor.find_best(best_id);
        expr
    };

    log::debug!(
        "Extraction complete: best_cost={:.2}, cache_size={}",
        best_cost,
        comm_cache.len()
    );

    (best_cost, expr)
}

/// Extract the best schedule using performance measurement
///
/// This measures **actual runtime** by compiling and executing schedules. It's slow but provides
/// ground-truth performance data.
///
/// **Requirements**: Full toolchain (Polygeist, MLIR, execution environment)
///
/// # Arguments
/// * `egraph` - The e-graph containing explored schedules
/// * `root` - The e-class ID to extract from
///
/// # Returns
/// * `runtime` - Measured execution time in seconds (lower is better)
/// * `expr` - The best schedule as a RecExpr
///
/// # Performance Notes
/// - Caches measurement results to avoid redundant compilations
/// - Can take minutes for large e-graphs with many unique schedules
/// - Runtime measurements can have variance; consider multiple runs for production
///
/// # Example
/// ```no_run
/// use polysat::{extract_best_by_performance, SchedOp, ScheduleAnalysis};
/// use egg::{EGraph, Id};
///
/// # let egraph: EGraph<SchedOp, ScheduleAnalysis> = todo!();
/// # let root: Id = todo!();
/// let (runtime_ms, best_expr) = extract_best_by_performance(&egraph, root);
/// println!("Best schedule runs in: {:.2}ms", runtime_ms * 1000.0);
/// ```
pub fn extract_best_by_performance(
    egraph: &egg::EGraph<SchedOp, ScheduleAnalysis>,
    root: Id,
) -> (f64, RecExpr<SchedOp>) {
    let extractor = Extractor::new(egraph, PerformanceCost::new());
    extractor.find_best(root)
}

/// Export e-graph to DOT format for visualization
///
/// Generates a GraphViz DOT file that can be rendered with `dot`, `neato`, or online tools.
/// Useful for debugging e-graph structure and understanding transformation exploration.
///
/// # Arguments
/// * `egraph` - The e-graph to export
/// * `filename` - Path to output DOT file
///
/// # Example
/// ```no_run
/// use polysat::{export_egraph_to_dot, SchedOp, ScheduleAnalysis};
/// use egg::EGraph;
///
/// # let egraph: EGraph<SchedOp, ScheduleAnalysis> = todo!();
/// export_egraph_to_dot(&egraph, "schedule_egraph.dot").unwrap();
/// // Then: dot -Tpdf schedule_egraph.dot -o schedule_egraph.pdf
/// ```
///
/// # Visualization Tips
/// - For small e-graphs (<100 nodes): Use `dot` layout
/// - For large e-graphs (100-1000 nodes): Use `neato` or `fdp` for better readability
/// - For huge e-graphs (1000+ nodes): Consider extracting subgraphs or using `sfdp`
///
/// ```bash
/// # Different layout algorithms
/// dot -Tpdf egraph.dot -o egraph.pdf      # Hierarchical
/// neato -Tpdf egraph.dot -o egraph.pdf    # Force-directed
/// fdp -Tpdf egraph.dot -o egraph.pdf      # Spring model
/// ```
pub fn export_egraph_to_dot(
    egraph: &egg::EGraph<SchedOp, ScheduleAnalysis>,
    filename: &str,
) -> Result<(), std::io::Error> {
    use std::fs::File;
    use std::io::Write;

    // Generate DOT representation
    let dot = egraph.dot().to_string();

    // Write to file
    let mut file = File::create(filename)?;
    file.write_all(dot.as_bytes())?;

    println!("E-graph exported to {}", filename);
    println!(
        "Visualize with: dot -Tpdf {} -o {}.pdf",
        filename,
        filename.replace(".dot", "")
    );

    Ok(())
}

// Measure schedule performance
///
/// MUST be implemented via ISL schedule tree analysis, NOT string matching.
pub fn measure_schedule(_handle: &ScheduleHandle) -> f64 {
    todo!("Implement via ISL schedule tree analysis - see RFC001 ScheduleProperties::from_isl()")
}

// Measure schedule with actual execution
pub fn measure_schedule_real(handle: &ScheduleHandle, c_file: &str) -> Result<f64, String> {
    // Write schedule to file
    let schedule_file = "temp_schedule.yaml";
    fs::write(schedule_file, handle.schedule.to_str())
        .map_err(|e| format!("Failed to write schedule: {}", e))?;

    // Generate MLIR with Polygeist
    let output = Command::new("polygeist-opt")
        .arg(c_file)
        .arg("--polyhedral-opt")
        .arg("--use-polyhedral-optimizer=islexternal")
        .arg(format!("--islexternal-import-schedules={}", schedule_file))
        .arg("-o")
        .arg("temp.mlir")
        .output()
        .map_err(|e| format!("Failed to run polygeist-opt: {}", e))?;

    if !output.status.success() {
        return Err(format!(
            "Polygeist failed: {}",
            String::from_utf8_lossy(&output.stderr)
        ));
    }

    // Compile MLIR to executable
    let output = Command::new("mlir-opt")
        .arg("temp.mlir")
        .arg("--convert-scf-to-cf")
        .arg("--convert-cf-to-llvm")
        .arg("--convert-func-to-llvm")
        .arg("-o")
        .arg("temp.ll")
        .output()
        .map_err(|e| format!("Failed to run mlir-opt: {}", e))?;

    if !output.status.success() {
        return Err(format!(
            "MLIR lowering failed: {}",
            String::from_utf8_lossy(&output.stderr)
        ));
    }

    // Execute and measure
    let start = Instant::now();
    let output = Command::new("lli")
        .arg("temp.ll")
        .output()
        .map_err(|e| format!("Failed to execute: {}", e))?;
    let duration = start.elapsed();

    if !output.status.success() {
        return Err(format!(
            "Execution failed: {}",
            String::from_utf8_lossy(&output.stderr)
        ));
    }

    Ok(duration.as_secs_f64())
}

// Extract all unique schedules from e-graph for exploration
pub fn extract_all_schedules(
    egraph: &egg::EGraph<SchedOp, ScheduleAnalysis>,
) -> Vec<ScheduleHandle> {
    let mut schedules = Vec::new();
    let mut seen = std::collections::HashSet::new();

    for class in egraph.classes() {
        if let Some(ref handle) = class.data.schedule {
            let schedule_str = handle.schedule.to_str();
            if !seen.contains(&schedule_str) {
                seen.insert(schedule_str);
                schedules.push(handle.clone());
            }
        }
    }

    schedules
}
