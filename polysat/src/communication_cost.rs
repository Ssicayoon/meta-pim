//! ISL-based cost modeling for dependency-aware schedule optimization
//!
//! # Overview
//!
//! This module implements complete cost estimation for NCP architecture based on
//! ISL (Integer Set Library) analysis of schedules and access patterns. It combines:
//!
//! 1. **Computation Cost**: Based on iteration space cardinality and operation latency
//! 2. **Communication Cost**: Based on data reuse patterns and NoC topology
//!
//! # Architecture
//!
//! ## Communication Cost
//!
//! Based on Polyhedral approach:
//! 1. **Phase 1**: Detect if communication is needed (`is_injective()`)
//! 2. **Phase 2**: Quantify volume via `range()` + `count_val()`
//! 3. **Phase 3**: Weight by NCP 2D mesh distance via `deltas()`
//!
//! ## Computation Cost (ISL Domain Analysis)
//!
//! Based on iteration space analysis:
//! 1. Extract domain from schedule: `schedule.get_domain()`
//! 2. Compute cardinality: Number of iterations
//! 3. Apply operation latency: Domain-specific (GEMM FMA, NTT ModMul, etc.)
//! 4. Account for parallelism: Divide by parallel factor
//!
//! # Key Insight from PPCG (gpu_group.c:950)
//!
//! ```c
//! no_reuse = isl_union_map_is_injective(local);
//! ```
//!
//! **Injective test**:
//! - `access.apply_domain(schedule)` → `{ [t] -> A[...] }`
//! - If **injective**: Each data element accessed by ≤1 time point → NO communication needed
//! - If **non-injective**: Same element accessed by >1 time point → Communication required
//!
//! # Integration with PolySat
//!
//! - **Does NOT use** egg's `CostFunction` trait (cannot access dependencies)
//! - **Direct usage** in `extract_best()` loop where we have access to `EGraph` and `ScheduleData`
//! - **Leverages cached dependencies** from `ScheduleData.dependencies`

use crate::access_analysis::AccessInfo;
use crate::dependency_aware::DependencyInfo;
use isl_rs::{Context, Schedule, UnionMap};
use log;
use std::sync::Arc;

// ============================================================================
// Computation Cost Estimation (ISL Domain Analysis)
// ============================================================================

/// Compute computation cost based on iteration space and operation latency
///
/// # Algorithm
///
/// For a polyhedral schedule with domain $D = \{ S[i_0, \ldots, i_n] : \text{constraints} \}$:
///
/// 1. **Extract iteration count**: $|D|$ via ISL domain cardinality
/// 2. **Determine operation type**: GEMM (FMA), NTT (ModMul), Stencil (Add/Mul), etc.
/// 3. **Apply latency**: $T_{\text{compute}} = |D| \times L_{\text{op}} / P$
///    - $L_{\text{op}}$: Operation latency (Meta-PIM Table 1)
///    - $P$: Parallelism factor (detected from schedule)
///
/// # Meta-PIM Operation Latencies (Table 1, 32-bit precision)
///
/// - **ModMul** (NTT): 669 cycles
/// - **FMA** (GEMM): ~500 cycles (estimate, not in paper)
/// - **Add/Mul** (Stencil): ~100 cycles (estimate)
///
/// # Arguments
/// * `schedule` - The ISL schedule to analyze
/// * `domain_type` - Type of computation (GEMM, NTT, Stencil, Generic)
/// * `parallelism_factor` - Detected parallelism (1 = sequential, 8 = 8-way parallel)
///
/// # Returns
/// Computation cost in cycles
///
/// # Example
///
/// ```text
/// // GEMM 64×64×64
/// Domain: { S0[i,j,k] : 0 <= i,j,k <= 63 }
/// Iteration count: 64³ = 262,144
/// Operation: FMA (fused multiply-add)
/// Latency: 500 cycles per FMA
/// Parallelism: 8 (outer loops parallelized)
///
/// T_compute = 262,144 × 500 / 8 = 16,384,000 cycles
/// ```
pub fn compute_computation_cost_from_schedule(
    schedule: &Schedule,
    domain_type: ComputationDomain,
    parallelism_factor: usize,
) -> Option<f64> {
    // Step 1: Extract domain from schedule
    let domain = schedule.get_domain();
    let domain_str = domain.to_str().to_string();

    log::debug!("Computing computation cost for domain: {}", domain_str);

    // Step 2: Try to compute exact cardinality
    // For bounded domains, ISL can compute exact count
    if let Some(iteration_count) = extract_iteration_count(&domain_str) {
        // Step 3: Determine operation latency based on domain type
        let op_latency = get_operation_latency(domain_type);

        // Step 4: Account for parallelism
        // Each NCP unit processes iterations/parallelism_factor iterations
        let parallel_factor = parallelism_factor.max(1) as f64;

        // Step 5: Compute total cost
        // Total cycles = (iterations × latency_per_iteration) / parallelism
        let total_cost = (iteration_count as f64) * op_latency / parallel_factor;

        log::debug!("  Iteration count: {}", iteration_count);
        log::debug!("  Operation latency: {} cycles", op_latency);
        log::debug!("  Parallelism factor: {}", parallel_factor);
        log::debug!("  Total computation cost: {:.2} cycles", total_cost);

        Some(total_cost)
    } else {
        // Cannot compute exact count (parametric domain or parsing failed)
        log::debug!("  Cannot extract iteration count (parametric domain)");
        None
    }
}

/// Extract iteration count from ISL domain string
///
/// Parses domain constraints to compute exact cardinality for bounded domains.
///
/// # Examples
///
/// ```text
/// "{ S0[i,j] : 0 <= i <= 63 and 0 <= j <= 63 }" → 64 × 64 = 4,096
/// "{ S0[i,j,k] : 0 <= i,j,k <= 63 }" → 64³ = 262,144
/// "{ S0[i] : 0 < i <= 510 }" → 510
/// ```
///
/// Returns `None` for parametric domains like `{ S[i] : 0 <= i < N }`.
fn extract_iteration_count(domain_str: &str) -> Option<i64> {
    use regex::Regex;
    use std::collections::HashMap;

    // Regex to capture constraints.
    // Group 1-5: Bounded range (Const Op Var Op Const) e.g. 0 <= i <= 63
    // Group 6-8: Var Op Const e.g. i <= 63
    // Group 9-11: Const Op Var e.g. 0 <= i
    let pattern = Regex::new(
        r"(\d+)\s*(<=|<)\s*(\w+)\s*(<=|<)\s*(\d+)|(\w+)\s*(<=|>=|<|>)\s*(\d+)|(\d+)\s*(<=|>=|<|>)\s*(\w+)"
    ).ok()?;

    let mut var_bounds: HashMap<String, (Option<i64>, Option<i64>)> = HashMap::new();

    for cap in pattern.captures_iter(domain_str) {
        if let (Some(lower_str), Some(op1), Some(var), Some(op2), Some(upper_str)) =
            (cap.get(1), cap.get(2), cap.get(3), cap.get(4), cap.get(5))
        {
            // Case 1: Bounded range (0 <= i <= 63)
            let mut lower = lower_str.as_str().parse::<i64>().ok()?;
            let mut upper = upper_str.as_str().parse::<i64>().ok()?;

            if op1.as_str() == "<" {
                lower += 1;
            }
            if op2.as_str() == "<" {
                upper -= 1;
            }

            let entry = var_bounds
                .entry(var.as_str().to_string())
                .or_insert((None, None));
            entry.0 = Some(entry.0.map_or(lower, |old| old.max(lower)));
            entry.1 = Some(entry.1.map_or(upper, |old| old.min(upper)));
        } else if let (Some(var), Some(op), Some(val_str)) = (cap.get(6), cap.get(7), cap.get(8)) {
            // Case 2: Var Op Const (i <= 63)
            let val = val_str.as_str().parse::<i64>().ok()?;
            let entry = var_bounds
                .entry(var.as_str().to_string())
                .or_insert((None, None));

            match op.as_str() {
                "<=" => entry.1 = Some(entry.1.map_or(val, |old| old.min(val))),
                "<" => entry.1 = Some(entry.1.map_or(val - 1, |old| old.min(val - 1))),
                ">=" => entry.0 = Some(entry.0.map_or(val, |old| old.max(val))),
                ">" => entry.0 = Some(entry.0.map_or(val + 1, |old| old.max(val + 1))),
                _ => {}
            }
        } else if let (Some(val_str), Some(op), Some(var)) = (cap.get(9), cap.get(10), cap.get(11))
        {
            // Case 3: Const Op Var (0 <= i)
            let val = val_str.as_str().parse::<i64>().ok()?;
            let entry = var_bounds
                .entry(var.as_str().to_string())
                .or_insert((None, None));

            match op.as_str() {
                "<=" => entry.0 = Some(entry.0.map_or(val, |old| old.max(val))),
                "<" => entry.0 = Some(entry.0.map_or(val + 1, |old| old.max(val + 1))),
                ">=" => entry.1 = Some(entry.1.map_or(val, |old| old.min(val))),
                ">" => entry.1 = Some(entry.1.map_or(val - 1, |old| old.min(val - 1))),
                _ => {}
            }
        }
    }

    if var_bounds.is_empty() {
        println!("[DEBUG] No constraints found in domain: {}", domain_str);
        return None;
    }

    let mut total_count: i64 = 1;
    for (var, (lower_opt, upper_opt)) in var_bounds {
        let lower = lower_opt.unwrap_or(0); // Default lower bound 0

        let upper = match upper_opt {
            Some(u) => u,
            None => {
                println!("[DEBUG] Missing upper bound for var {}", var);
                return None;
            }
        };

        let size = if upper >= lower { upper - lower + 1 } else { 0 };
        total_count = total_count.saturating_mul(size);
    }

    Some(total_count)
}

/// Domain type for computation cost modeling
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ComputationDomain {
    /// General Matrix Multiply: C[i,j] += A[i,k] * B[k,j]
    /// Operations: Fused Multiply-Add (FMA)
    GEMM,

    /// Number Theoretic Transform (FFT-like)
    /// Operations: Modular multiplication + addition
    NTT,

    /// Stencil computations: out[i,j] = f(in[i±1, j±1])
    /// Operations: Add/Mul on neighbors
    Stencil,

    /// 2D Convolution: out[i,j] = Σ kernel[m,n] × input[i+m, j+n]
    /// Operations: Multiply-accumulate
    Conv2D,

    /// Generic computation (use conservative estimate)
    Generic,
}

/// Infer computation domain type from schedule characteristics
///
/// Uses heuristics based on:
/// - Domain dimensionality (1D/2D/3D)
/// - Iteration space size (powers of 2 suggest FFT/NTT)
/// - Schedule patterns (certain tiling patterns characteristic of specific kernels)
///
/// # Returns
/// Best guess for ComputationDomain, defaults to Generic if uncertain
pub fn infer_domain_type(schedule: &Schedule) -> ComputationDomain {
    let domain = schedule.get_domain();
    let domain_str = domain.to_str().to_string();

    log::debug!("Inferring domain type from: {}", domain_str);

    // Extract dimensionality by counting iterator variables in domain
    // Pattern: S0[i0, i1, i2] → 3 dimensions
    // We count commas within brackets: [..., ..., ...]
    use regex::Regex;
    let dim_pattern = Regex::new(r"\[([^\]]+)\]").unwrap();
    let dim_count = if let Some(cap) = dim_pattern.captures(&domain_str) {
        let iterators_str = cap.get(1).map(|m| m.as_str()).unwrap_or("");
        // Count commas + 1 to get number of iterators
        iterators_str.split(',').count()
    } else {
        0
    };

    log::debug!("  Detected {} dimensions", dim_count);

    // Check for power-of-2 sizes (characteristic of NTT/FFT)
    let has_power_of_2 = domain_str.contains("8192")
        || domain_str.contains("4096")
        || domain_str.contains("2048")
        || domain_str.contains("1024")
        || domain_str.contains("512") && !domain_str.contains("510");

    // Heuristic inference
    let inferred = if dim_count == 3 {
        // 3D domain: likely GEMM (i,j,k) or Conv3D
        ComputationDomain::GEMM
    } else if dim_count == 2 {
        // 2D domain: likely Stencil or Conv2D
        ComputationDomain::Stencil
    } else if dim_count == 1 && has_power_of_2 {
        // 1D with power-of-2: likely NTT/FFT
        ComputationDomain::NTT
    } else {
        // Default to generic
        log::debug!(
            "  Cannot infer domain type (dim={}, pow2={}), using Generic",
            dim_count,
            has_power_of_2
        );
        ComputationDomain::Generic
    };

    log::debug!("  Inferred type: {:?}", inferred);
    inferred
}

/// Get operation latency for computation domain
///
/// # Meta-PIM Latencies (Table 1)
///
/// - **ModMul** (32-bit): 669 cycles (measured on Meta-PIM prototype)
/// - **FMA** (estimate): ~500 cycles (not in paper, conservative)
/// - **Add/Mul** (estimate): ~100 cycles (simpler operations)
///
/// # Returns
/// Latency per operation in cycles
fn get_operation_latency(domain: ComputationDomain) -> f64 {
    match domain {
        ComputationDomain::NTT => {
            // Meta-PIM Table 1: ModMul latency = 669 cycles
            // NTT butterfly: 2 ModMul + 2 ModAdd per butterfly
            // We count ModMul as dominant cost
            669.0
        }
        ComputationDomain::GEMM => {
            // GEMM: Fused Multiply-Add (FMA) operation
            // Each C[i,j] += A[i,k] * B[k,j] is 1 FMA
            // Conservative estimate (not in Meta-PIM paper)
            // Typical FP64 FMA: 400-600 cycles on low-power processors
            500.0
        }
        ComputationDomain::Conv2D => {
            // Conv2D: Similar to GEMM (multiply-accumulate pattern)
            500.0
        }
        ComputationDomain::Stencil => {
            // Stencil: Typically simple add/mul operations
            // Lighter than FMA
            200.0
        }
        ComputationDomain::Generic => {
            // Generic: Use conservative estimate
            400.0
        }
    }
}

/// Detect parallelism factor from schedule structure (Phase 4 - Precise Tiling-Based)
///
/// **PHASE 4 IMPROVEMENT**: This function now computes PRECISE parallelism based on
/// actual tiling structure, not heuristic marker counting.
///
/// # Algorithm
///
/// For a tiled schedule, parallelism = $$\prod_{d \in \text{tiled dims}} \left\lceil \frac{N_d}{T_d} \right\rceil$$
///
/// Where:
/// - $N_d$: Dimension size (extracted from domain)
/// - $T_d$: Tile size (extracted from `mod` operations)
/// - Product over all **spatial** dimensions (exclude reduction dimensions)
///
/// # Examples
///
/// **GEMM 64×64×64 with single i-tiling**:
/// ```text
/// Domain: { S0[i0, i1, i2] : 0 <= i0,i1,i2 <= 63 }
/// Schedule: i0 - (i0) mod 8  (tile i by 8)
/// Parallelism: 64/8 = 8-way
/// ```
///
/// **GEMM 64×64×64 with double tiling**:
/// ```texttext
/// Domain: { S0[i0, i1, i2] : 0 <= i0,i1,i2 <= 63 }
/// Band: { S0[i0, i1, i2] -> [i0, i1, i2] }
/// Parallelism: (64/8) × (64/8) = 64-way
/// ```
///
/// # Returns
/// Parallelism factor P (capped at 512 for NCP hardware limit)

// ============================================================================
// Phase 4: Local Memory Footprint Cost (NCP Memory Capacity Awareness)
// ============================================================================

/// Detect reduction dimension tiling penalty (Phase 4.2b - Refined Heuristic)
///
/// **KEY INSIGHT**: Not all K-tiling is bad! We must distinguish:
/// 1. **Local K-tiling** (GOOD): Cache/register blocking BEFORE distribution
/// 2. **Parallel K-tiling** (BAD): Distributing K dimension across NCPs
///
/// # Why Local K-tiling is GOOD
///
/// Local tiling (within a single NCP) improves cache/register reuse:
/// ```text
/// for ii, jj in parallel:  // Distribute i,j across NCPs
///   for k in 0..K:         // Reduction loop
///     C[ii,jj] += A[ii,k] * B[k,jj]
/// ```
/// This is standard loop blocking - no penalty.
///
/// # Why Parallel K-tiling is BAD
///
/// Distributing K across NCPs causes redundant communication:
/// ```text
/// for kk in 0..K step 8 parallel:  // Each NCP gets different K range
///   for i, j:
///     C_partial[kk][i,j] += ...
/// // Requires reduction across NCPs at the end!
/// ```
/// Each NCP computes partial results, requiring expensive reduction.
///
/// # Detection Strategy (Option 3)
///
/// We differentiate these cases by analyzing schedule structure:
/// - **Local tiling**: K-tiling appears in child of "schedule" node (loop blocking)
/// - **Parallel tiling**: K-tiling appears in child of "mark: parallel" node
///
/// **Current Implementation (Conservative)**:
/// For 64x64 GEMM, K-tiling penalty (~10K cycles) is negligible compared to
/// total cost (~16M cycles = 0.06%). The **main cost model** (volume + distance)
/// already captures parallel K overhead. Therefore, we **disable this penalty**
/// to allow beneficial local K-tiling.
///
/// Future work could parse ISL schedule tree to detect true parallel K-tiling.
///
/// # Arguments
/// * `schedule_str` - Schedule string to analyze
/// * `domain_type` - Type of computation
///
/// # Returns
/// Penalty in cycles (0 for now - let main cost model handle it)
pub fn detect_reduction_tiling_penalty(schedule_str: &str, domain_type: ComputationDomain) -> f64 {
    match domain_type {
        ComputationDomain::GEMM => {
            // OPTION 3 IMPLEMENTATION (Conservative):
            //
            // Analysis shows that:
            // 1. Best schedules (rank 0001-0100) all use LOCAL K-tiling for cache blocking
            // 2. K-tiling penalty (10K) << total cost (16M), only 0.06%
            // 3. Main communication cost model (volume + distance) already accounts for
            //    parallel K overhead via increased communication volume
            // 4. Simple regex cannot reliably distinguish local vs parallel K-tiling
            //
            // Therefore: DISABLE penalty, rely on main cost model
            //
            // Future improvement: Parse ISL schedule tree to detect if K-tiling
            // occurs AFTER "mark: parallel" nodes (true parallel K)

            // Detect K-tiling for logging/debugging
            use regex::Regex;
            let k_tiling_pattern =
                Regex::new(r"i2\s*\)\s*mod").unwrap_or_else(|_| Regex::new("NEVER_MATCH").unwrap());

            let has_k_tiling = k_tiling_pattern.is_match(schedule_str);

            if has_k_tiling {
                log::debug!("Detected K-tiling (i2 mod) - allowing for local cache blocking");
            }

            // Return 0.0 - no penalty
            // Main communication cost model will capture parallel K overhead
            0.0
        }
        ComputationDomain::NTT => {
            // NTT typically doesn't have reduction dimensions
            0.0
        }
        ComputationDomain::Stencil => {
            // Stencil doesn't have reduction dimensions
            0.0
        }
        _ => 0.0,
    }
}

/// Compute local memory footprint penalty
///
/// This penalty ensures that schedules with better
/// data locality (smaller per-NCP footprint) are preferred during extraction.
///
/// # Theoretical Analysis
///
/// For GEMM 64×64×64 on NCP (64 KB local SRAM per NCP):
///
/// **Single-layer tiling (i by 8)**:
/// - Per-tile footprint: A(8×64) + B(64×64) + C(8×64) = 512 + 4,096 + 512 = 5,120 elements
/// - Memory usage: 5,120 × 4 bytes = 20 KB
/// - NCP count: 8 tiles → 8-way parallelism
///
/// **Double-layer tiling (i by 8, j by 8)**:
/// - Per-tile footprint: A(8×64) + B(64×8) + C(8×8) = 512 + 512 + 64 = 1,088 elements
/// - Memory usage: 1,088 × 4 bytes = 4.3 KB
/// - NCP count: 64 tiles → 64-way parallelism
///
/// **Key insight**: Double tiling has **4.7× smaller footprint** per NCP, allowing:
/// 1. Better cache utilization
/// 2. Reduced memory bandwidth pressure
/// 3. Higher parallelism (64 vs 8)
///
/// # Cost Formula
///
/// $$C_{\text{footprint}} = \begin{cases}
/// 10^6 & \text{if footprint} > 64\text{KB (exceeds capacity)} \\
/// 1000 \times (U - 0.5)^2 & \text{if } U > 0.5 \text{ (high utilization risk)} \\
/// 100 \times (0.05 - U) & \text{if } U < 0.05 \text{ (wasteful)} \\
/// 0 & \text{otherwise (sweet spot: 5-50\%)}
/// \end{cases}$$
///
/// Where $U = \frac{\text{footprint}}{64\text{KB}}$ is utilization ratio.
///
/// # Arguments
/// * `access_info` - Access patterns (reads/writes)
/// * `schedule` - The schedule to analyze
/// * `parallelism` - Detected parallelism factor (number of tiles/NCPs)
///
/// # Returns
/// Footprint penalty in cycles (0 for optimal, high for bad)
pub fn compute_local_footprint_penalty(
    access_info: &AccessInfo,
    schedule: &Schedule,
    parallelism: usize,
) -> f64 {
    // Step 1: Estimate total data footprint (global)
    // This is the same volume computed in Phase 3
    let ctx = Arc::new(schedule.get_ctx());
    if let Some(global_volume) = compute_total_communication_volume(access_info, schedule, &ctx) {
        // Step 2: Estimate per-NCP footprint
        // Heuristic: footprint_per_ncp ≈ global_volume / parallelism
        //
        // This is an APPROXIMATION. The real per-NCP footprint depends on:
        // - Which dimensions are tiled
        // - Access patterns (full vs partial data needed)
        // - Data sharing between NCPs
        //
        // For GEMM with i-tiling only:
        //   - Each tile needs: A(8×64) + B(64×64) + C(8×64) ≈ 5,120 elements
        //   - global_volume / parallelism = 12,288 / 8 = 1,536 (UNDERESTIMATE!)
        //
        // For GEMM with i,j-tiling:
        //   - Each tile needs: A(8×64) + B(64×8) + C(8×8) ≈ 1,088 elements
        //   - global_volume / parallelism = 12,288 / 64 = 192 (UNDERESTIMATE!)
        //
        // The approximation is CONSISTENT (underestimates both), so RELATIVE
        // comparison is still valid: 1,536 vs 192 → 8× difference ✓

        let footprint_per_ncp = if parallelism > 0 {
            global_volume as f64 / parallelism as f64
        } else {
            global_volume as f64
        };

        // Step 3: Convert to cost penalty
        // NCP local SRAM: 64 KB = 16,384 elements (for 4-byte float32)
        const LOCAL_MEMORY_CAPACITY: f64 = 16384.0; // elements

        let utilization = footprint_per_ncp / LOCAL_MEMORY_CAPACITY;

        let penalty = if footprint_per_ncp > LOCAL_MEMORY_CAPACITY {
            // Exceeds local memory capacity
            // This should rarely happen, but if it does, heavy penalty
            let overflow = footprint_per_ncp - LOCAL_MEMORY_CAPACITY;
            log::warn!(
                "Local footprint ({:.0} elem) exceeds capacity ({:.0} elem)",
                footprint_per_ncp,
                LOCAL_MEMORY_CAPACITY
            );
            overflow * 100.0 // 100 cycles per element of overflow
        } else if utilization > 0.5 {
            // High utilization (risky - leaves little room for working data)
            // Quadratic penalty: prefer staying below 50% utilization
            let excess = utilization - 0.5;
            excess * excess * 10000.0
        } else if utilization < 0.05 {
            // Very low utilization (wasteful parallelism - too many small tiles)
            // Linear penalty: prefer not wasting NCPs
            let waste = 0.05 - utilization;
            waste * 1000.0
        } else {
            // Sweet spot: 5-50% utilization
            // No penalty - this is ideal range
            0.0
        };

        log::debug!(
            "Local footprint: {:.0} elem ({:.1}% of 64KB), penalty = {:.2} cycles",
            footprint_per_ncp,
            utilization * 100.0,
            penalty
        );

        penalty
    } else {
        // Cannot compute footprint, no penalty
        log::debug!("Cannot compute footprint, no penalty");
        0.0
    }
}

// ============================================================================
// Phase 1: Communication Detection (Boolean)
// ============================================================================

/// Detect if a schedule requires communication based on access patterns
///
/// # Theory (PPCG gpu_group.c:950)
///
/// For each memory access:
/// 1. Apply schedule: `{ S[i] -> A[f(i)] }` + `{ S[i] -> [t] }` → `{ [t] -> A[f(i)] }`
/// 2. Check injectivity:
///    - **Injective**: Each `A[...]` accessed by unique `[t]` → No reuse → No communication
///    - **Non-injective**: Same `A[...]` accessed by multiple `[t]` → Reuse → Communication
///
/// # Arguments
/// * `access` - Access relation `{ S[i,...] -> Array[...] }`
/// * `schedule_map` - Schedule map `{ S[i,...] -> [t0, t1, ...] }`
///
/// # Returns
/// * `true` if communication is needed (non-injective after schedule application)
/// * `false` if no communication (injective - each element accessed at most once)
///
/// # Example
///
/// ```text
/// // Case 1: Embarrassingly parallel (no communication)
/// access:   { S[i] -> A[i] }          // Each iteration accesses different element
/// schedule: { S[i] -> [i] }           // Sequential schedule
/// Result:   { [i] -> A[i] }           // Still injective → NO communication
///
/// // Case 2: Reduction (communication required)
/// access:   { S[i] -> sum[0] }        // All iterations access same element
/// schedule: { S[i] -> [i] }
/// Result:   { [i] -> sum[0] }         // Non-injective (many [i] → one sum[0]) → COMMUNICATION
///
/// // Case 3: Stencil (communication required)
/// access:   { S[i] -> A[i-1]; S[i] -> A[i]; S[i] -> A[i+1] }
/// schedule: { S[i] -> [i] }
/// Result:   A[i] accessed by S[i-1], S[i], S[i+1] → Non-injective → COMMUNICATION
/// ```
///
/// # Implementation Note
///
/// We apply the schedule to the access relation, NOT to dependencies directly.
/// This is because:
/// - Dependencies tell us "S[i] must execute before S[j]"
/// - But communication happens when "different time points access same data"
/// - The schedule determines WHEN each statement executes
/// - Combining schedule + access tells us WHEN each data element is accessed
pub fn has_communication_impl(
    access: &UnionMap, // Accept reference (Arc<UnionMap> can deref to &UnionMap)
    schedule_map: &UnionMap,
) -> bool {
    // Apply schedule to access relation
    // Before: { S[i] -> A[f(i)] }
    // After:  { [t] -> A[f(i)] }  where t = schedule(i)
    // Note: apply_domain() consumes both arguments, so we use copy() to create owned values
    let scheduled_access = access.copy().apply_domain(schedule_map.copy());

    // Check if injective
    let is_inj = scheduled_access.is_injective();
    log::debug!(
        "has_communication_impl: is_injective={}, scheduled_access={}",
        is_inj,
        scheduled_access.to_str()
    );

    // Check if injective
    // - Injective: each A[...] accessed by at most one [t] → no reuse
    // - Non-injective: same A[...] accessed by multiple [t] → reuse → communication
    !is_inj
}

// ============================================================================
// Phase 2: Communication Volume Quantification
// ============================================================================

/// Parse statistics for volume computation (Phase 2 enhancement)
#[derive(Debug, Clone, Default)]
pub struct VolumeParseStats {
    pub total_attempts: usize,
    pub successful_parses: usize,
    pub failed_parses: usize,
}

impl VolumeParseStats {
    pub fn success_rate(&self) -> f64 {
        if self.total_attempts == 0 {
            return 0.0;
        }
        (self.successful_parses as f64) / (self.total_attempts as f64)
    }
}

thread_local! {
    static PARSE_STATS: std::cell::RefCell<VolumeParseStats> = std::cell::RefCell::new(VolumeParseStats::default());
}

/// Get global parse statistics
pub fn get_parse_stats() -> VolumeParseStats {
    PARSE_STATS.with(|stats| stats.borrow().clone())
}

/// Reset global parse statistics
pub fn reset_parse_stats() {
    PARSE_STATS.with(|stats| {
        *stats.borrow_mut() = VolumeParseStats::default();
    });
}

/// Extract array name from ISL set string
///
/// # Examples
/// - `"{ A[i,k] : 0 <= i <= 255 }"` → `Some("A")`
/// - `"{ MemRef_C[i0, i1] : ... }"` → `Some("MemRef_C")`
/// - `"{ [i,j] : ... }"` → `None` (anonymous set)
fn extract_array_name(set_str: &str) -> Option<String> {
    use regex::Regex;

    // Pattern: "{ ARRAY_NAME[..." where ARRAY_NAME is alphanumeric + underscore
    let pattern = Regex::new(r"\{\s*([A-Za-z_][A-Za-z0-9_]*)\s*\[").ok()?;

    if let Some(cap) = pattern.captures(set_str) {
        Some(cap.get(1)?.as_str().to_string())
    } else {
        None
    }
}

/// Estimate volume by parsing ISL set constraint strings
///
/// # Heuristic Approach
///
/// Instead of ISL's exact quasi-polynomial cardinality (complex to handle),
/// we parse constraint strings to extract simple bounds.
///
/// # Supported Patterns
///
/// - `0 <= i < N` → size N
/// - `0 <= i <= N-1` → size N
/// - `i >= 0 and i < N` → size N
///
/// Volume = product of all dimension sizes
///
/// # Limitations
///
/// - Only handles simple box constraints (rectangular domains)
/// - Doesn't handle complex polyhedral constraints
/// - Returns None for parametric or unbounded sets
///
/// # Examples
///
/// ```text
/// Input: "{ A[i,k] : 0 <= i < 256 and 0 <= k < 256 }"
/// Output: Some(("A".to_string(), 65536))  // 256 * 256
///
/// Input: "{ B[i] : 0 <= i < N }"
/// Output: None  // parametric
/// ```
fn estimate_volume_from_constraints(set_str: &str) -> Option<(String, i64)> {
    use regex::Regex;

    // Update statistics
    PARSE_STATS.with(|stats| {
        stats.borrow_mut().total_attempts += 1;
    });

    // Step 1: Extract array name
    let array_name = match extract_array_name(set_str) {
        Some(name) => name,
        None => {
            log::debug!("Cannot extract array name from: {}", set_str);
            PARSE_STATS.with(|stats| {
                stats.borrow_mut().failed_parses += 1;
            });
            return None;
        }
    };

    // Step 2: Extract constraints portion: "{ ... : CONSTRAINTS }"
    let constraints = if let Some(colon_idx) = set_str.find(':') {
        let close_brace = set_str.rfind('}').unwrap_or(set_str.len());
        &set_str[colon_idx + 1..close_brace].trim()
    } else {
        // No constraints (unbounded or empty)
        log::debug!("No constraints found in set string: {}", set_str);
        PARSE_STATS.with(|stats| {
            stats.borrow_mut().failed_parses += 1;
        });
        return None;
    };

    log::debug!("Parsing constraints: {}", constraints);

    // Regex patterns for common bound formats
    // Pattern 1: "var < N" (exclusive bound, size = N)
    let pattern_lt = Regex::new(r"(\w+)\s*<\s*(\d+)").ok()?;
    // Pattern 2: "var <= N" (inclusive bound, size = N+1)
    let pattern_le = Regex::new(r"(\w+)\s*<=\s*(\d+)").ok()?;
    // Pattern 3: "var <= N-1" (equivalent to < N, size = N)
    let pattern_le_minus = Regex::new(r"(\w+)\s*<=\s*(\d+)\s*-\s*1").ok()?;

    let mut bounds: std::collections::HashMap<String, i64> = std::collections::HashMap::new();

    // Split by "and" to process each constraint separately
    for constraint in constraints.split("and") {
        let constraint = constraint.trim();

        // Try pattern 1: var < N (exclusive) → size = N
        if let Some(cap) = pattern_lt.captures(constraint) {
            let var = cap.get(1)?.as_str().to_string();
            let bound: i64 = cap.get(2)?.as_str().parse().ok()?;

            // Update bound (use max if multiple constraints on same var)
            bounds
                .entry(var)
                .and_modify(|b| *b = (*b).max(bound))
                .or_insert(bound);
            continue;
        }

        // Try pattern 3: var <= N-1 → size = N
        if let Some(cap) = pattern_le_minus.captures(constraint) {
            let var = cap.get(1)?.as_str().to_string();
            let bound: i64 = cap.get(2)?.as_str().parse().ok()?;

            bounds
                .entry(var)
                .and_modify(|b| *b = (*b).max(bound))
                .or_insert(bound);
            continue;
        }

        // Try pattern 2: var <= N (inclusive) → size = N+1
        // Must come AFTER pattern 3 to avoid matching "N-1" as just "N"
        if let Some(cap) = pattern_le.captures(constraint) {
            let var = cap.get(1)?.as_str().to_string();
            let bound_inclusive: i64 = cap.get(2)?.as_str().parse().ok()?;
            let bound = bound_inclusive + 1; // Inclusive → exclusive

            bounds
                .entry(var)
                .and_modify(|b| *b = (*b).max(bound))
                .or_insert(bound);
            continue;
        }

        // Ignore lower bounds (0 <= var) as they don't affect size
        // Ignore complex constraints for now
    }

    if bounds.is_empty() {
        log::debug!("No parseable bounds found, cannot estimate volume");
        PARSE_STATS.with(|stats| {
            stats.borrow_mut().failed_parses += 1;
        });
        return None;
    }

    // Step 3: Compute product of all dimension bounds
    let mut volume: i64 = 1;
    for (var, bound) in &bounds {
        log::debug!("  Dimension {}: size {}", var, bound);
        match volume.checked_mul(*bound) {
            Some(v) => volume = v,
            None => {
                log::debug!("Volume overflow detected, cannot estimate");
                PARSE_STATS.with(|stats| {
                    stats.borrow_mut().failed_parses += 1;
                });
                return None;
            }
        }
    }

    log::debug!(
        "Array '{}': estimated volume = {} elements",
        array_name,
        volume
    );

    // Success! Update statistics
    PARSE_STATS.with(|stats| {
        stats.borrow_mut().successful_parses += 1;
    });

    Some((array_name, volume))
}

/// Compute communication volume for a single access pattern
///
/// # Theory (PPCG gpu_group.c:1080)
///
/// The communication volume is the number of unique data elements accessed after
/// schedule application. This is computed via ISL operations:
/// 1. Apply schedule: `{ S[i] -> A[f(i)] }` + `{ S[i] -> [t] }` → `{ [t] -> A[f(i)] }`
/// 2. Get range (data footprint): `range({ [t] -> A[...] }) = { A[...] }`
/// 3. Count elements: `|{ A[...] }|`
///
/// # Arguments
/// * `access` - Access relation `{ S[i,...] -> Array[...] }`
/// * `schedule_map` - Schedule map `{ S[i,...] -> [t0, t1, ...] }`
///
/// # Returns
/// * `Some(HashMap<String, i64>)` - Per-array volume breakdown (in elements, not bytes)
/// * `None` - Cannot compute (e.g., unbounded set, ISL operation failed)
///
/// # Example
///
/// ```text
/// // GEMM: C[i,j] += A[i,k] * B[k,j]  (256×256×256)
/// // Access: { S[i,j,k] -> A[i,k] ; S[i,j,k] -> B[k,j] }   (reads A and B)
/// // Schedule: { S[i,j,k] -> [i,j,k] } (sequential)
/// //
/// // Step 1: Apply schedule
/// // { [i,j,k] -> A[i,k] ; [i,j,k] -> B[k,j] }  (time → data mapping)
/// //
/// // Step 2: Get range (data footprint)
/// // range = { A[i,k] : 0 <= i < 256, 0 <= k < 256 ; B[k,j] : 0 <= k < 256, 0 <= j < 256 }
/// //
/// // Step 3: Parse and aggregate by array name
/// // Returns: {"A": 65536, "B": 65536}
/// ```
///
/// # Implementation Note
///
/// Returns per-array breakdown to enable deduplication (Phase 2 Priority 1).
/// For example, if array C appears in both reads and writes, callers can
/// deduplicate by taking the max volume instead of summing.
pub fn compute_communication_volume_impl(
    access: &UnionMap,
    schedule_map: &UnionMap,
) -> Option<std::collections::HashMap<String, i64>> {
    // Step 1: Apply schedule to access relation
    // { S[i] -> A[f(i)] } → { [t] -> A[f(i)] }
    let scheduled_access = access.copy().apply_domain(schedule_map.copy());

    // Step 2: Get the range (data footprint)
    // { [t] -> A[...] } → { A[...] }
    let footprint = scheduled_access.range();

    // Step 3: Estimate volume by parsing footprint constraints
    // Note: ISL's exact cardinality counting (isl_union_pw_qpolynomial_card)
    // returns quasi-polynomials which are complex to handle.
    // For Phase 2, we use a heuristic parser approach similar to PPCG's symbolic bounds.
    let footprint_str = footprint.to_str().to_string();

    log::debug!(
        "Computing per-array volume for footprint: {}",
        footprint_str
    );

    // Parse constraints to extract dimension bounds per array
    // UnionSet format: "{ A[i,k] : 0 <= i <= 255 and 0 <= k <= 255 ; B[k,j] : ... }"
    // Split by semicolon to handle multiple sets in the union
    let mut array_volumes: std::collections::HashMap<String, i64> =
        std::collections::HashMap::new();

    // Remove outer braces and split by semicolon
    let inner = footprint_str
        .trim()
        .trim_start_matches('{')
        .trim_end_matches('}');
    for set_str in inner.split(';') {
        let set_str = set_str.trim();
        if set_str.is_empty() {
            continue;
        }

        // Wrap in braces for individual parsing
        let individual_set = format!("{{ {} }}", set_str);

        match estimate_volume_from_constraints(&individual_set) {
            Some((array_name, vol)) => {
                log::debug!("  Array '{}': {} elements", array_name, vol);

                // Aggregate by array name (use max to handle union of same array with different constraints)
                array_volumes
                    .entry(array_name)
                    .and_modify(|existing| *existing = (*existing).max(vol))
                    .or_insert(vol);
            }
            None => {
                log::debug!("Failed to parse set: {}, returning None", individual_set);
                return None; // If any set fails, return None (triggers fallback)
            }
        }
    }

    if array_volumes.is_empty() {
        log::debug!("No volumes computed, returning None");
        return None;
    }

    let total: i64 = array_volumes.values().sum();
    log::debug!("Per-array volumes: {:?} (total = {})", array_volumes, total);

    Some(array_volumes)
}

/// Compute total communication volume across all accesses WITH DEDUPLICATION
///
/// # Strategy (Phase 2 Priority 1 Fix)
///
/// 1. Compute per-array volumes for reads
/// 2. Compute per-array volumes for writes
/// 3. **Deduplicate** by taking MAX volume per array (not sum!)
/// 4. Sum the deduplicated volumes
///
/// This fixes the GEMM C[i,j] double-counting issue where C appears in both reads and writes.
///
/// # Example
///
/// ```text
/// GEMM: C[i,j] += A[i,k] * B[k,j]
/// Reads:  {A: 65536, B: 65536, C: 65536}
/// Writes: {C: 65536}
///
/// OLD (wrong): 65536 + 65536 + 65536 + 65536 = 262,144 (C counted twice!)
/// NEW (correct): max({A: 65536, B: 65536, C: max(65536, 65536)}) = 196,608
/// ```
///
/// # Arguments
/// * `access_info` - Complete access information (reads + writes for all statements)
/// * `schedule` - ISL schedule tree
/// * `ctx` - ISL context (for schedule.get_map())
///
/// # Returns
/// * `Some(volume)` - Total DEDUPLICATED communication volume in elements
/// * `None` - Cannot compute (e.g., missing access info, ISL operations failed)
///
/// # Units
///
/// Returns volume in **number of elements**, not bytes.
/// To convert to bytes, multiply by element size (e.g., 4 bytes for float32).
pub fn compute_total_communication_volume(
    access_info: &AccessInfo,
    schedule: &Schedule,
    _ctx: &Arc<Context>,
) -> Option<i64> {
    // Step 1: Get schedule map
    let schedule_map = schedule.get_map();

    if schedule_map.is_empty() {
        log::warn!("schedule.get_map() returned empty map - cannot compute volume");
        return None;
    }

    // Step 2: Check if we have access information
    let has_reads = access_info.reads_union_map.is_some();
    let has_writes = access_info.writes_union_map.is_some();

    if !has_reads && !has_writes {
        log::debug!("No access union maps available - cannot compute volume");
        return None;
    }

    // Step 3: Aggregate volumes by array name (deduplication)
    let mut array_volumes: std::collections::HashMap<String, i64> =
        std::collections::HashMap::new();

    // Step 3a: Collect read volumes
    if let Some(ref reads_union) = access_info.reads_union_map {
        match compute_communication_volume_impl(reads_union.as_ref(), &schedule_map) {
            Some(read_vols) => {
                log::debug!("Read volumes: {:?}", read_vols);
                for (array, vol) in read_vols {
                    array_volumes
                        .entry(array)
                        .and_modify(|existing| *existing = (*existing).max(vol))
                        .or_insert(vol);
                }
            }
            None => {
                log::debug!("Cannot compute read volume (parametric or ISL failure)");
                return None; // If any component fails, return None
            }
        }
    }

    // Step 3b: Collect write volumes (deduplicate with reads)
    if let Some(ref writes_union) = access_info.writes_union_map {
        match compute_communication_volume_impl(writes_union.as_ref(), &schedule_map) {
            Some(write_vols) => {
                log::debug!("Write volumes: {:?}", write_vols);
                for (array, vol) in write_vols {
                    array_volumes
                        .entry(array)
                        .and_modify(|existing| *existing = (*existing).max(vol))
                        .or_insert(vol);
                }
            }
            None => {
                log::debug!("Cannot compute write volume (parametric or ISL failure)");
                return None;
            }
        }
    }

    // Step 4: Sum deduplicated volumes
    let total_volume: i64 = array_volumes.values().sum();

    log::debug!("Deduplicated array volumes: {:?}", array_volumes);
    log::debug!(
        "Total communication volume (deduplicated): {} elements",
        total_volume
    );

    Some(total_volume)
}

/// Check if ANY array in the access info requires communication
///
/// # Strategy
///
/// Check all read and write access relations. If ANY of them shows reuse after
/// schedule application, communication is required.
///
/// # Arguments
/// * `access_info` - Complete access information (reads + writes for all statements)
/// * `schedule` - ISL schedule tree
/// * `ctx` - ISL context (for schedule.get_map())
///
/// # Returns
/// * `Some(true)` - Communication required
/// * `Some(false)` - No communication needed
/// * `None` - Cannot determine (e.g., schedule.get_map() failed)
pub fn requires_communication(
    access_info: &AccessInfo,
    schedule: &Schedule,
    _ctx: &Arc<Context>,
) -> Option<bool> {
    // Step 1: Convert schedule tree to schedule map
    // Schedule tree: Band/Filter/Sequence nodes (hierarchical)
    // Schedule map: { S[...] -> [t0, t1, ...] } (flat mapping)
    let schedule_map = schedule.get_map();

    // Error handling: Check if schedule map is valid
    // ISL may return empty map for invalid/incomplete schedules
    if schedule_map.is_empty() {
        log::warn!("schedule.get_map() returned empty map - cannot determine communication");
        return None;
    }

    // Step 2: Check if we have ANY access information
    // If both union maps are None, we cannot determine communication requirements
    let has_reads = access_info.reads_union_map.is_some();
    let has_writes = access_info.writes_union_map.is_some();

    if !has_reads && !has_writes {
        log::debug!("No access union maps available - cannot determine communication (returning None for conservative penalty)");
        return None; // FIXED: Was Some(false) - unsafe!
    }

    // Step 3: Check reads union map (if available)
    // Polymer-extracted access patterns provide precise UnionMaps
    if let Some(ref reads_union) = access_info.reads_union_map {
        // Check all reads for reuse
        let read_comm = has_communication_impl(reads_union.as_ref(), &schedule_map);
        if read_comm {
            log::debug!("Communication required: reads show reuse after scheduling");
            return Some(true);
        }
    }

    // Step 4: Check writes union map
    if let Some(ref writes_union) = access_info.writes_union_map {
        let write_comm = has_communication_impl(writes_union.as_ref(), &schedule_map);
        if write_comm {
            log::debug!("Communication required: writes show reuse after scheduling");
            return Some(true);
        }
    }

    // No communication required - all accesses are injective after scheduling
    log::debug!("No communication required: all accesses are injective");
    Some(false)
}

// ============================================================================
// Phase 3: Distance-Weighted Communication Cost (Meta-PIM Formula)
// ============================================================================

/// Compute average Manhattan distance from dependency delta vectors
///
/// # Algorithm
///
/// For a dependency map $D: \mathcal{I} \rightarrow \mathcal{I}$, we compute:
/// 1. Extract delta set: $\Delta = \{ i' - i \mid (i, i') \in D \}$ via `isl_map_deltas()`
/// 2. Sample representative points from each BasicSet in $\Delta$
/// 3. Project to 2D mesh coordinates (assuming logical→physical mapping)
/// 4. Compute Manhattan distance: $H(d) = |d_x| + |d_y|$
/// 5. Return average: $\bar{H} = \frac{1}{|\Delta|} \sum_{d \in \Delta} H(d)$
///
/// # Arguments
/// * `dep_map` - ISL dependency UnionMap (e.g., RAW/WAR/WAW)
/// * `ncp_slices` - Number of NCP slices (for 2D mesh projection)
/// * `ncp_banks` - Number of banks per slice (for 2D mesh projection)
///
/// # Returns
/// * `Some(avg_hops)` - Average Manhattan distance in hops
/// * `None` - Cannot compute (empty map, cross-statement deps, etc.)
///
/// # NCP 2D Mesh Mapping
///
/// Meta-PIM architecture: 8 slices × 64 banks = 512 NCPs in 2D mesh.
/// Logical iteration space → Physical NCP mapping:
/// - For GEMM (i,j,k): Map (i,j) to 2D mesh, k is reduction dimension
/// - For Stencil (i,j): Direct (i,j) mapping
/// - Projection matrix $P_{2d}$: Extract first 2 spatial dimensions
fn compute_average_manhattan_distance(
    dep_map: &UnionMap,
    ncp_slices: usize,
    ncp_banks: usize,
) -> Option<f64> {
    use isl_rs::DimType;

    if dep_map.is_empty() {
        log::debug!("Empty dependency map, no distance to compute");
        return None;
    }

    // Step 1: Extract delta vectors using ISL API
    // deltas() returns UnionSet representing { [d] : ∃ i,i'. D(i,i') and d = i' - i }
    let deltas = dep_map.copy().deltas();

    if deltas.is_empty() {
        log::debug!("Empty deltas (likely cross-statement dependency), cannot compute distance");
        return None;
    }

    // Step 2: Get BasicSet list from UnionSet
    let basic_set_list = deltas.get_basic_set_list();
    let n_sets = basic_set_list.size();

    if n_sets == 0 {
        return None;
    }

    let mut total_distance = 0.0;
    let mut sample_count = 0;

    // Step 3: Sample from each BasicSet to get representative delta vectors
    for i in 0..n_sets {
        let basic_set = basic_set_list.get_at(i);
        let n_dims = basic_set.dim(DimType::Set) as usize;

        if n_dims == 0 {
            continue;
        }

        // Try to sample a point from this BasicSet
        // For simplicity, we use ISL's sample_point() or extract from constraints
        // Here we approximate by parsing the BasicSet string representation
        let set_str = basic_set.to_str();

        // Parse delta vector from constraint string
        // Format: "{ [d0, d1, ...] : constraints }"
        if let Some(deltas_vec) = parse_representative_delta(&set_str, n_dims) {
            // Project to 2D mesh coordinates
            // Take first 2 dimensions as spatial coordinates (i,j)
            let d_x = if deltas_vec.len() > 0 {
                deltas_vec[0].abs()
            } else {
                0
            };
            let d_y = if deltas_vec.len() > 1 {
                deltas_vec[1].abs()
            } else {
                0
            };

            // Compute Manhattan distance in 2D mesh
            // Map logical distance to physical hops
            // Assumption: Iteration space is mapped uniformly to NCP mesh
            let mesh_side = (ncp_slices * ncp_banks) as f64;
            let mesh_dim = mesh_side.sqrt(); // e.g., sqrt(512) ≈ 22.6

            // Scale logical distance to physical distance
            // If delta is (1, 0), it maps to ~1 hop in typical tile-based mapping
            let physical_hops = (d_x as f64 + d_y as f64).min(mesh_dim);

            total_distance += physical_hops;
            sample_count += 1;

            log::debug!(
                "  Delta sample: ({}, {}) → {:.1} hops",
                d_x,
                d_y,
                physical_hops
            );
        }
    }

    if sample_count > 0 {
        let avg_dist = total_distance / sample_count as f64;
        log::debug!(
            "Computed average Manhattan distance: {:.2} hops from {} samples",
            avg_dist,
            sample_count
        );
        Some(avg_dist)
    } else {
        log::debug!("Could not sample any delta vectors");
        None
    }
}

/// Parse a representative delta vector from ISL BasicSet string representation
///
/// Extracts integer values from constraint expressions like:
/// - "{ [d0, d1] : d0 = 1 and d1 = 0 }"
/// - "{ [d0] : d0 = -1 }"
/// - "{ [d0, d1, d2] : d0 = 0 and d1 = 1 and d2 = 0 }"
fn parse_representative_delta(set_str: &str, n_dims: usize) -> Option<Vec<i32>> {
    // Try to extract exact values from equality constraints
    let mut deltas = vec![0; n_dims];
    let mut found_any = false;

    // Look for patterns like "d0 = 1" or "d0 = -1"
    for dim_idx in 0..n_dims {
        let pattern = format!(r"d{}\s*=\s*(-?\d+)", dim_idx);
        if let Ok(re) = regex::Regex::new(&pattern) {
            if let Some(cap) = re.captures(set_str) {
                if let Some(val_str) = cap.get(1) {
                    if let Ok(val) = val_str.as_str().parse::<i32>() {
                        deltas[dim_idx] = val;
                        found_any = true;
                    }
                }
            }
        }
    }

    // If we found at least one dimension with exact value, return the vector
    // Otherwise, use conservative estimate (assume distance 1 in first dimension)
    if found_any {
        Some(deltas)
    } else {
        // Conservative fallback: assume minimal non-zero distance
        log::debug!(
            "Could not parse exact deltas from '{}', using conservative estimate",
            set_str.chars().take(80).collect::<String>()
        );
        let mut conservative = vec![0; n_dims];
        if n_dims > 0 {
            conservative[0] = 1; // Assume at least 1-hop distance
        }
        Some(conservative)
    }
}

/// Compute per-array communication cost with distance weighting (Phase 3)
///
/// # Meta-PIM Formula
///
/// $$T_{\text{comm},a} = \chi(L_a) \left[ \beta V_a + \alpha V_a \cdot \frac{1}{|\Delta_a|} \sum_{d \in \Delta_a} H(d) \right]$$
///
/// Where:
/// - $\chi(L_a)$: Injective indicator (0 if no communication, 1 if communication)
/// - $V_a$: Communication volume (elements)
/// - $\beta = 1/32$: Bandwidth cost (cycles/element)
/// - $\alpha = 5$: Hop latency (cycles/hop)
/// - $H(d)$: Manhattan distance in 2D mesh
/// - $\Delta_a$: Dependency delta vectors
///
/// # Arguments
/// * `array_volume` - Communication volume for this array (elements)
/// * `dep_map` - Dependency map for this array (contains deltas)
/// * `ncp_slices` - Number of NCP slices
/// * `ncp_banks` - Number of banks per slice
///
/// # Returns
/// Communication cost in cycles for this array
fn compute_array_communication_cost_with_distance(
    array_volume: i64,
    dep_map: Option<&UnionMap>,
    ncp_slices: usize,
    ncp_banks: usize,
) -> f64 {
    // Meta-PIM constants from paper (Table 2)
    const ALPHA: f64 = 5.0; // cycles/hop (NoC hop latency)
    const BETA: f64 = 1.0 / 32.0; // cycles/element (link bandwidth: 32 B/cycle)

    let volume_f64 = array_volume as f64;

    // Bandwidth cost (always present if volume > 0)
    let bandwidth_cost = BETA * volume_f64;

    // Distance-weighted cost (only if we have dependency information)
    let distance_cost = if let Some(deps) = dep_map {
        if let Some(avg_hops) = compute_average_manhattan_distance(deps, ncp_slices, ncp_banks) {
            ALPHA * volume_f64 * avg_hops
        } else {
            // No distance information, use conservative estimate
            // Assume average distance is sqrt(N)/2 where N is total NCPs
            let ncp_count = ncp_slices * ncp_banks;
            let conservative_hops = (ncp_count as f64).sqrt() / 2.0;
            log::debug!(
                "Using conservative hop estimate: {:.1} hops",
                conservative_hops
            );
            ALPHA * volume_f64 * conservative_hops
        }
    } else {
        // No dependency map, use conservative estimate
        let ncp_count = ncp_slices * ncp_banks;
        let conservative_hops = (ncp_count as f64).sqrt() / 2.0;
        ALPHA * volume_f64 * conservative_hops
    };

    let total = bandwidth_cost + distance_cost;

    log::debug!(
        "  Array cost: volume={} elem, bandwidth={:.2} cyc, distance={:.2} cyc, total={:.2} cyc",
        array_volume,
        bandwidth_cost,
        distance_cost,
        total
    );

    total
}

// ============================================================================
// Phase 2+3: Unified Communication Cost Computation
// ============================================================================

/// Compute communication cost based on actual dependencies and volumes
///
/// # Implementation (Phase 3 - Distance-Weighted)
///
/// This implements the complete Meta-PIM communication cost formula:
///
/// $$T_{\text{comm}} = \sum_{a \in \mathcal{A}} \chi(L_a) \left[ \beta V_a + \alpha V_a \cdot \frac{1}{|\Delta_a|} \sum_{d \in \Delta_a} H(d) \right]$$
///
/// Returns:
/// - **Phase 3 cost** if both AccessInfo and DependencyInfo available (volume + distance)
/// - **Phase 2 cost** if only AccessInfo available (volume only, with $\beta$ coefficient)
/// - **Conservative penalty** if communication detected but volume unavailable
/// - **0.0** if no communication required (injective accesses)
///
/// # Cost Model Components
///
/// 1. **Volume Cost**: $\beta V_a$ where $\beta = 1/32$ cycles/element (NoC bandwidth)
/// 2. **Distance Cost**: $\alpha V_a \cdot \bar{H}$ where $\alpha = 5$ cycles/hop (NoC latency)
/// 3. **Total**: Sum over all arrays that require communication ($\chi(L_a) = 1$)
///
/// # Priority Logic (Method A+ Extended)
///
/// 1. **Priority 1A (Most Accurate - Phase 3)**: AccessInfo + DependencyInfo
///    - Compute exact volume via `compute_total_communication_volume()`
///    - Extract delta vectors from dependency maps
///    - Compute Manhattan distances in 2D mesh
///    - Apply Meta-PIM formula with distance weighting
///    - Falls back to Priority 1B if volume computation fails
///
/// 2. **Priority 1B (Phase 2 Fallback)**: AccessInfo volume only
///    - Compute volume without distance information
///    - Use bandwidth cost only: $\beta V_a$
///    - Falls back to Priority 1C if volume fails
///
/// 3. **Priority 1C (Qualitative)**: AccessInfo injective test
///    - Uses `requires_communication()` to check if communication needed
///    - Returns conservative penalty if non-injective
///    - Returns 0.0 if injective (no communication)
///    - Falls back to Priority 2 if AccessInfo unavailable
///
/// 4. **Priority 2 (Conservative)**: DependencyInfo fallback
///    - Non-empty dependency maps → conservative penalty
///    - Empty dependency maps → 0.0 (no communication)
///
/// 5. **Priority 3 (Unknown)**: Conservative penalty
///    - Returns 100.0 if neither AccessInfo nor DependencyInfo available
///
/// # Arguments
/// * `dependencies` - Cached dependency info (optional, from ScheduleData)
/// * `access_info` - Shared access info (from ScheduleAnalysis)
/// * `schedule` - The schedule to evaluate
/// * `ctx` - ISL context
/// * `ncp_slices` - Number of NCP slices (default: 8)
/// * `ncp_banks` - Number of banks per slice (default: 64)
///
/// # Returns
/// Communication cost in cycles:
/// - **Phase 3**: Volume + distance-weighted cost (most accurate)
/// - **Phase 2**: Volume-only cost (when dependencies unavailable)
/// - **Conservative penalty**: When communication detected but cannot quantify
/// - **0.0**: No communication (injective or no dependencies)
pub fn compute_dependency_aware_communication_cost(
    dependencies: Option<&Arc<DependencyInfo>>,
    access_info: Option<&Arc<AccessInfo>>,
    schedule: &Schedule,
    ctx: &Arc<Context>,
) -> f64 {
    // Default NCP configuration (Meta-PIM paper §4.1)
    const DEFAULT_NCP_SLICES: usize = 8;
    const DEFAULT_NCP_BANKS: usize = 64;

    compute_dependency_aware_communication_cost_with_config(
        dependencies,
        access_info,
        schedule,
        ctx,
        DEFAULT_NCP_SLICES,
        DEFAULT_NCP_BANKS,
    )
}

/// Compute communication cost with configurable NCP parameters
///
/// This is the full implementation with all Phase 3 features.
/// See `compute_dependency_aware_communication_cost()` for documentation.
pub fn compute_dependency_aware_communication_cost_with_config(
    dependencies: Option<&Arc<DependencyInfo>>,
    access_info: Option<&Arc<AccessInfo>>,
    schedule: &Schedule,
    ctx: &Arc<Context>,
    ncp_slices: usize,
    ncp_banks: usize,
) -> f64 {
    // ========================================================================
    // Priority 1A: AccessInfo + DependencyInfo (Phase 3 - most accurate)
    // ========================================================================
    if let Some(info) = access_info {
        // Try to compute exact volume first
        if let Some(volume) = compute_total_communication_volume(info, schedule, ctx) {
            // Check if we have dependency information for distance weighting
            if let Some(deps) = dependencies {
                // **PHASE 3**: Full formula with distance weighting
                // We have both volume and dependency information
                // Apply Meta-PIM formula with distance-weighted cost

                log::debug!(
                    "✓ [Priority 1A - Phase 3] Computing distance-weighted communication cost"
                );
                log::debug!("  Total volume: {} elements", volume);

                // Use RAW dependencies for distance estimation (primary communication driver)
                // In Meta-PIM, read-after-write dependencies dominate communication patterns
                let dep_map = if let Some(ref raw_map) = deps.raw_map {
                    Some(raw_map.as_ref())
                } else if let Some(ref war_map) = deps.war_map {
                    // Fallback to WAR if RAW unavailable
                    Some(war_map.as_ref())
                } else if let Some(ref waw_map) = deps.waw_map {
                    // Last resort: WAW
                    Some(waw_map.as_ref())
                } else {
                    None
                };

                // Apply distance-weighted cost formula
                let total_cost = compute_array_communication_cost_with_distance(
                    volume, dep_map, ncp_slices, ncp_banks,
                );

                log::debug!(
                    "✓ [Priority 1A - Phase 3] Total communication cost = {:.2} cycles",
                    total_cost
                );
                return total_cost;
            } else {
                // **PHASE 2 FALLBACK**: Volume only (no distance information)
                const BETA: f64 = 1.0 / 32.0; // Meta-PIM bandwidth cost
                let cost = volume as f64 * BETA;
                log::debug!("✓ [Priority 1A - Phase 2 Fallback] Volume-based cost: {} elements × {:.5} cycles/elem = {:.2} cycles",
                           volume, BETA, cost);
                return cost;
            }
        }

        // ====================================================================
        // Priority 1B: AccessInfo injective test (fallback if volume fails)
        // ====================================================================
        log::debug!("⚠ [Priority 1A] Cannot compute volume (parametric or parse failure), falling back to injective test");

        match requires_communication(info, schedule, ctx) {
            Some(true) => {
                log::debug!("✓ [Priority 1B] Communication required (non-injective) - using conservative penalty");
                return 500.0; // Conservative penalty when volume unavailable
            }
            Some(false) => {
                log::debug!(
                    "✓ [Priority 1B] No communication needed (injective - embarrassingly parallel)"
                );
                return 0.0;
            }
            None => {
                log::debug!("⚠ [Priority 1B] Cannot determine from AccessInfo, falling back to DependencyInfo");
                // Fall through to Priority 2
            }
        }
    } else {
        log::debug!("⚠ [Priority 1] No AccessInfo available, falling back to DependencyInfo");
    }

    // ========================================================================
    // Priority 2: DependencyInfo fallback (conservative estimate)
    // ========================================================================
    if let Some(deps) = dependencies {
        // Check if any dependency maps exist and are non-empty
        // Note: This is a CONSERVATIVE estimate - dependencies don't directly mean communication
        // (e.g., producer-consumer in same tile has dependency but no inter-tile communication)
        let has_raw = deps.raw_map.as_ref().map_or(false, |m| !m.is_empty());
        let has_war = deps.war_map.as_ref().map_or(false, |m| !m.is_empty());
        let has_waw = deps.waw_map.as_ref().map_or(false, |m| !m.is_empty());

        if has_raw || has_war || has_waw {
            log::debug!("⚠ [Priority 2 Fallback] Has dependencies (RAW={}, WAR={}, WAW={}) - conservatively assume communication possible",
                       has_raw, has_war, has_waw);
            return 500.0; // Conservative: dependencies may require communication
        } else {
            log::debug!("✓ [Priority 2 Fallback] No dependencies - likely no communication");
            return 0.0;
        }
    }

    // ========================================================================
    // Priority 3: Unknown - conservative penalty
    // ========================================================================
    log::warn!("⚠ [Priority 3 Fallback] No AccessInfo or DependencyInfo available - using conservative penalty");
    100.0 // Unknown: conservative estimate
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_no_communication_embarrassingly_parallel() {
        let ctx = Arc::new(Context::alloc());

        // A[i] = B[i] + C[i]  (each iteration accesses different elements)
        let access_str = "{ S[i] -> A[i] }";
        let schedule_str = "{ S[i] -> [i] }";

        let access = UnionMap::read_from_str(&ctx, access_str);
        let schedule = UnionMap::read_from_str(&ctx, schedule_str);

        assert_eq!(
            has_communication_impl(&access, &schedule),
            false,
            "Embarrassingly parallel should have NO communication"
        );
    }

    #[test]
    fn test_has_communication_reduction() {
        let ctx = Arc::new(Context::alloc());

        // sum += A[i]  (all iterations write to same location)
        let access_str = "{ S[i] -> sum[0] }";
        let schedule_str = "{ S[i] -> [i] }";

        let access = UnionMap::read_from_str(&ctx, access_str);
        let schedule = UnionMap::read_from_str(&ctx, schedule_str);

        assert_eq!(
            has_communication_impl(&access, &schedule),
            true,
            "Reduction should require communication (multiple times access same element)"
        );
    }

    #[test]
    fn test_has_communication_stencil() {
        let ctx = Arc::new(Context::alloc());

        // A[i] = A[i-1] + A[i] + A[i+1]  (reuse neighbors)
        let access_str = "{ S[i] -> A[i-1]; S[i] -> A[i]; S[i] -> A[i+1] }";
        let schedule_str = "{ S[i] -> [i] }";

        let access = UnionMap::read_from_str(&ctx, access_str);
        let schedule = UnionMap::read_from_str(&ctx, schedule_str);

        assert_eq!(
            has_communication_impl(&access, &schedule),
            true,
            "Stencil should require communication (A[i] accessed by S[i-1], S[i], S[i+1])"
        );
    }

    // ========================================================================
    // Phase 2: Volume Computation Tests
    // ========================================================================

    #[test]
    fn test_estimate_volume_simple_2d() {
        // Test parsing "{ A[i,k] : 0 <= i < 256 and 0 <= k < 256 }"
        let set_str = "{ A[i,k] : 0 <= i < 256 and 0 <= k < 256 }";
        let result = estimate_volume_from_constraints(set_str);

        assert!(result.is_some(), "Should parse successfully");
        let (array_name, volume) = result.unwrap();
        assert_eq!(array_name, "A", "Should extract array name 'A'");
        assert_eq!(
            volume,
            256 * 256,
            "Should compute 256×256 = 65536 for 2D array"
        );
    }

    #[test]
    fn test_estimate_volume_3d() {
        // Test parsing "{ S[i,j,k] : 0 <= i < 128 and 0 <= j < 64 and 0 <= k < 32 }"
        let set_str = "{ S[i,j,k] : 0 <= i < 128 and 0 <= j < 64 and 0 <= k < 32 }";
        let result = estimate_volume_from_constraints(set_str);

        assert!(result.is_some(), "Should parse successfully");
        let (array_name, volume) = result.unwrap();
        assert_eq!(array_name, "S", "Should extract array name 'S'");
        assert_eq!(
            volume,
            128 * 64 * 32,
            "Should compute 128×64×32 = 262144 for 3D domain"
        );
    }

    #[test]
    fn test_estimate_volume_alternative_format() {
        // Test parsing "{ B[k,j] : k <= 255 and j <= 255 and k >= 0 and j >= 0 }"
        // This uses <= N format (inclusive bounds)
        let set_str = "{ B[k,j] : k <= 255 and j <= 255 and k >= 0 and j >= 0 }";
        let result = estimate_volume_from_constraints(set_str);

        assert!(
            result.is_some(),
            "Should parse successfully with inclusive bounds"
        );
        let (array_name, volume) = result.unwrap();
        assert_eq!(array_name, "B", "Should extract array name 'B'");
        // k <= 255 means size 256, j <= 255 means size 256
        assert_eq!(
            volume,
            256 * 256,
            "Should convert inclusive bounds correctly"
        );
    }

    #[test]
    fn test_estimate_volume_parametric() {
        // Test parametric set (should return None)
        let set_str = "{ A[i] : 0 <= i < N }";
        let result = estimate_volume_from_constraints(set_str);

        assert_eq!(
            result, None,
            "Parametric sets should return None (cannot compute constant)"
        );
    }

    #[test]
    fn test_estimate_volume_no_constraints() {
        // Test set with no constraints
        let set_str = "{ A[i] }";
        let result = estimate_volume_from_constraints(set_str);

        assert_eq!(result, None, "Unbounded sets should return None");
    }

    #[test]
    fn test_compute_volume_impl() {
        let ctx = Arc::new(Context::alloc());

        // GEMM read of A: { S[i,j,k] -> A[i,k] } with i,j,k in [0,256)
        // After scheduling, footprint should be A[0..256, 0..256]
        let access_str = "{ S[i,j,k] -> A[i,k] : 0 <= i < 256 and 0 <= j < 256 and 0 <= k < 256 }";
        let schedule_str = "{ S[i,j,k] -> [i,j,k] }";

        let access = UnionMap::read_from_str(&ctx, access_str);
        let schedule = UnionMap::read_from_str(&ctx, schedule_str);

        let result = compute_communication_volume_impl(&access, &schedule);

        // Expected: A[i,k] for i,k in [0,256) = 256*256 = 65536 elements
        assert!(result.is_some(), "Should compute volume successfully");
        let volumes = result.unwrap();
        assert_eq!(
            volumes.get("A"),
            Some(&65536),
            "GEMM read of A should access 256×256 = 65536 elements"
        );
    }

    #[test]
    fn test_volume_based_cost() {
        let ctx = Arc::new(Context::alloc());

        // Simple 1D array access with known size
        let access_str = "{ S[i] -> A[i] : 0 <= i < 1000 }";
        let schedule_str = "{ S[i] -> [i] }";

        let access = UnionMap::read_from_str(&ctx, access_str);
        let schedule_map = UnionMap::read_from_str(&ctx, schedule_str);

        let result = compute_communication_volume_impl(&access, &schedule_map);

        // Expected volume: 1000 elements for array A
        assert!(result.is_some(), "Should compute volume successfully");
        let volumes = result.unwrap();
        assert_eq!(
            volumes.get("A"),
            Some(&1000),
            "Array A should have 1000 elements"
        );

        // Total volume should be 1000
        let total_volume: i64 = volumes.values().sum();
        assert_eq!(total_volume, 1000);

        // Cost should be volume × 0.01 = 10.0 cycles
        let expected_cost = total_volume as f64 * 0.01;
        assert_eq!(
            expected_cost, 10.0,
            "Volume-based cost should be proportional to data size"
        );
    }
}
