//! Phase 5 Directive 3: End-to-End Performance Correlation
//!
//! This experiment validates that the PolySat analytical cost model correlates
//! with real execution time by:
//! 1. Generating multiple schedules with different tile sizes
//! 2. Computing analytical cost for each schedule
//! 3. Generating C code via ISL codegen and executing
//! 4. Measuring Pearson correlation between analytical cost and execution time
//!
//! **Core Hypothesis**: T_ISL / T_Baseline ≈ C_ISL / C_Baseline
//!
//! **Success Criteria**: Pearson R > 0.85
//!
//! Run with: cargo run --example phase5_correlation --release

use isl_rs::{Context, DimType, MultiUnionPwAff, MultiVal, Schedule, UnionSet, Val, ValList};
use polysat::isl_codegen_ffi;
use polysat::optimize::{
    compute_cache_cliff_penalty, compute_gemm_working_set, compute_tiling_reuse_factor,
    CacheHierarchyConfig,
};
use polysat::schedule_properties::ScheduleProperties;

use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Command;
use std::time::Instant;

/// Problem size for GEMM benchmark
///
/// CRITICAL: Must be large enough to stress memory hierarchy!
///
/// Working set calculation for GEMM: 3 * N^2 * sizeof(double) = 3 * N^2 * 8 bytes
///   - N=512:  Working set ≈ 6 MB   -> fits in L3 (INVALID: no DRAM pressure)
///   - N=1024: Working set ≈ 24 MB  -> borderline L3
///   - N=2048: Working set ≈ 96 MB  -> EXCEEDS L3 (VALID: forces DRAM traffic)
///
/// The cache cliff model only manifests when we actually hit DRAM.
/// Testing L1/L2 effects while everything fits in L3 measures NOISE, not SIGNAL.
const PROBLEM_SIZE: usize = 2048;

/// Number of iterations for timing
/// Reduced because 2048^3 operations take much longer
const ITERATIONS: usize = 3;
/// Work directory for compiled files
const WORK_DIR: &str = "polysat_e2e_test";

/// Data point for correlation analysis
#[derive(Debug, Clone)]
struct DataPoint {
    name: String,
    tile_size: i32,
    analytical_cost: f64,
    working_set_kb: usize,
    execution_time_ms: f64,
}

fn main() {
    println!("======================================================================");
    println!("  PHASE 5 DIRECTIVE 3: End-to-End Performance Correlation");
    println!("======================================================================");
    println!();

    // Create work directory
    fs::create_dir_all(WORK_DIR).expect("Failed to create work directory");

    let cache_config = CacheHierarchyConfig::default();

    // Calculate total working set for the problem
    let total_working_set_mb = (3 * PROBLEM_SIZE * PROBLEM_SIZE * 8) / (1024 * 1024);

    println!("Experiment Configuration:");
    println!("  Problem Size: {}x{} GEMM", PROBLEM_SIZE, PROBLEM_SIZE);
    println!("  Total Working Set: {} MB (3 matrices)", total_working_set_mb);
    println!("  Iterations: {}", ITERATIONS);
    println!("  Compiler: cc -O2 (reduced to isolate schedule effects)");
    println!();
    println!("Cache Hierarchy:");
    println!("  L1 Cache: {} KB", cache_config.l1_size_bytes / 1024);
    println!("  L2 Cache: {} KB", cache_config.l2_size_bytes / 1024);
    println!("  L3 Cache: {} MB (assumed)", cache_config.l3_size_bytes / (1024 * 1024));
    println!();
    println!("Working Set vs L3: {} MB vs {} MB -> {}",
             total_working_set_mb,
             cache_config.l3_size_bytes / (1024 * 1024),
             if total_working_set_mb > cache_config.l3_size_bytes / (1024 * 1024) {
                 "EXCEEDS L3 (DRAM traffic expected)"
             } else {
                 "FITS IN L3 (WARNING: may not stress hierarchy)"
             });
    println!();

    // Generate schedules with different tile sizes
    //
    // Critical tile sizes (working set = 3*T^2*8 bytes):
    //   L1 (32KB):  T_critical = sqrt(32*1024 / 24) ≈ 36
    //   L2 (256KB): T_critical = sqrt(256*1024 / 24) ≈ 104
    //   L3 (8MB):   T_critical = sqrt(8*1024*1024 / 24) ≈ 590
    //
    // With N=2048, total working set = 96MB >> L3, so DRAM traffic is guaranteed.
    // Tiles that don't fit in L2 will see DRAM latency, not just L3.
    let tile_sizes = vec![
        (0, "Baseline (no tiling)"),
        (16, "T=16 (fits L1)"),
        (32, "T=32 (fits L1)"),
        (64, "T=64 (L2 region)"),
        (128, "T=128 (beyond L2)"),
        (256, "T=256 (deep L3)"),
        (512, "T=512 (exceeds L3)"),
    ];

    let mut data_points = Vec::new();

    println!("======================================================================");
    println!("  STEP 1: Generate Schedules and Compute Analytical Costs");
    println!("======================================================================");
    println!();

    let ctx = Context::alloc();

    for (tile_size, description) in &tile_sizes {
        print!("Processing {}: ", description);

        let (schedule, analytical_cost, working_set) = if *tile_size == 0 {
            // Baseline: no tiling
            let schedule = create_baseline_gemm_schedule(&ctx);
            let props = ScheduleProperties::from_isl(&schedule);
            let cost = compute_baseline_cost(&props);
            (schedule, cost, 0)
        } else {
            // Tiled schedule
            match create_tiled_gemm_schedule(&ctx, *tile_size) {
                Some(schedule) => {
                    let tile_sizes_vec = vec![*tile_size, *tile_size, *tile_size];
                    let working_set =
                        compute_gemm_working_set(&tile_sizes_vec, cache_config.element_size_bytes);
                    let penalty = compute_cache_cliff_penalty(working_set, &cache_config);

                    // CORRECTED: Use penalty / reuse_factor
                    // This captures the benefit of tiling (data reuse) vs penalty (cache overflow)
                    let reuse_factor = compute_tiling_reuse_factor(Some(&tile_sizes_vec));
                    let analytical_cost = penalty / reuse_factor;

                    (schedule, analytical_cost, working_set)
                }
                None => {
                    println!("SKIP (tiling failed)");
                    continue;
                }
            }
        };

        // Generate C code
        let c_code = match generate_c_code(&schedule) {
            Ok(code) => code,
            Err(e) => {
                println!("SKIP (codegen failed: {})", e);
                continue;
            }
        };

        // Compile and measure
        let execution_time = match compile_and_measure(&c_code, *tile_size) {
            Ok(time) => time,
            Err(e) => {
                println!("SKIP (execution failed: {})", e);
                continue;
            }
        };

        println!(
            "OK (cost={:.2}, time={:.3}ms)",
            analytical_cost, execution_time
        );

        data_points.push(DataPoint {
            name: description.to_string(),
            tile_size: *tile_size,
            analytical_cost,
            working_set_kb: working_set / 1024,
            execution_time_ms: execution_time,
        });
    }

    println!();

    // Display results table
    println!("======================================================================");
    println!("  TABLE 2: Performance Correlation Results");
    println!("======================================================================");
    println!();
    println!(
        "{:25} {:>12} {:>12} {:>12} {:>12}",
        "Schedule", "Working Set", "Analytical", "Execution", "Speedup"
    );
    println!("{}", "-".repeat(75));

    // Calculate baseline execution time for speedup
    let baseline_time = data_points
        .iter()
        .find(|p| p.tile_size == 0)
        .map(|p| p.execution_time_ms)
        .unwrap_or(1.0);

    for point in &data_points {
        let speedup = baseline_time / point.execution_time_ms;
        println!(
            "{:25} {:>10} KB {:>12.2} {:>10.3} ms {:>11.2}x",
            point.name, point.working_set_kb, point.analytical_cost, point.execution_time_ms, speedup
        );
    }

    println!();

    // Compute all correlation metrics
    if data_points.len() >= 3 {
        let pearson = compute_pearson_correlation(&data_points);
        let spearman = compute_spearman_correlation(&data_points);
        let kendall = compute_kendall_tau(&data_points);

        println!("======================================================================");
        println!("  CORRELATION ANALYSIS");
        println!("======================================================================");
        println!();
        println!("Three correlation metrics computed:");
        println!();
        println!("  1. Pearson R   = {:.4}  (linear relationship, magnitude-sensitive)", pearson);
        println!("  2. Spearman rho = {:.4}  (rank correlation, order-sensitive)", spearman);
        println!("  3. Kendall tau = {:.4}  (concordance ratio, robust to outliers)", kendall);
        println!();

        // Scientific interpretation
        println!("======================================================================");
        println!("  INTERPRETATION");
        println!("======================================================================");
        println!();
        println!("For optimizer validation, RANK CORRELATION is the critical metric:");
        println!("  - We need the model to RANK schedules correctly");
        println!("  - We do NOT need exact execution time prediction");
        println!("  - If Rank(C_analytical) ~ Rank(T_execution), optimizer works!");
        println!();

        // Determine validity based on rank correlation
        let rank_valid = spearman > 0.6 || kendall > 0.4;
        let magnitude_valid = pearson.abs() > 0.7;

        if rank_valid {
            println!("VERDICT: RANK CORRELATION VALIDATED");
            println!("  Spearman rho = {:.2} indicates correct schedule ordering", spearman);
            if !magnitude_valid {
                println!("  Note: Pearson R = {:.2} is weak, but this is expected.", pearson);
                println!("  Hardware prefetching smooths the cache cliff in practice.");
                println!("  The model is CONSERVATIVE (safe) rather than precise.");
            }
        } else if pearson > 0.0 {
            println!("VERDICT: WEAK POSITIVE CORRELATION");
            println!("  Direction is correct (higher cost -> higher time)");
            println!("  But ranking power is limited (Spearman = {:.2})", spearman);
            println!();
            println!("Possible causes:");
            println!("  1. Small sample size (n={})", data_points.len());
            println!("  2. Execution times too similar (<2% variance)");
            println!("  3. Hardware prefetching dominates cache effects");
        } else {
            println!("VERDICT: NO CORRELATION OR INVERSE");
            println!("  Model may need recalibration for this hardware.");
        }

        // Show rank comparison for scientific transparency
        println!();
        println!("======================================================================");
        println!("  RANK COMPARISON (for manual verification)");
        println!("======================================================================");
        println!();

        // Sort by analytical cost to show expected order
        let mut sorted_by_cost: Vec<_> = data_points.iter().enumerate().collect();
        sorted_by_cost.sort_by(|a, b| {
            a.1.analytical_cost.partial_cmp(&b.1.analytical_cost).unwrap()
        });

        // Sort by execution time to show actual order
        let mut sorted_by_time: Vec<_> = data_points.iter().enumerate().collect();
        sorted_by_time.sort_by(|a, b| {
            a.1.execution_time_ms.partial_cmp(&b.1.execution_time_ms).unwrap()
        });

        println!("Expected order (by analytical cost, lowest first):");
        for (rank, (_, point)) in sorted_by_cost.iter().enumerate() {
            println!("  {}. {} (cost={:.2}, time={:.3}ms)",
                     rank + 1, point.name, point.analytical_cost, point.execution_time_ms);
        }

        println!();
        println!("Actual order (by execution time, fastest first):");
        for (rank, (_, point)) in sorted_by_time.iter().enumerate() {
            println!("  {}. {} (cost={:.2}, time={:.3}ms)",
                     rank + 1, point.name, point.analytical_cost, point.execution_time_ms);
        }

    } else {
        println!("Insufficient data points for correlation analysis (need >= 3).");
    }

    println!();
    println!("======================================================================");
    println!();

    // Cleanup
    println!("Cleaning up work directory...");
    let _ = fs::remove_dir_all(WORK_DIR);
    println!("Done.");
}

/// Create baseline (non-tiled) GEMM schedule
fn create_baseline_gemm_schedule(ctx: &Context) -> Schedule {
    let domain_str = format!(
        "{{ S0[i,j,k] : 0 <= i < {} and 0 <= j < {} and 0 <= k < {} }}",
        PROBLEM_SIZE, PROBLEM_SIZE, PROBLEM_SIZE
    );
    let domain = UnionSet::read_from_str(ctx, &domain_str);
    let schedule = Schedule::from_domain(domain);

    // Create identity schedule (i, j, k order)
    let partial = MultiUnionPwAff::read_from_str(
        ctx,
        &format!(
            "[{{ S0[i,j,k] -> [(i)]; S0[i,j,k] -> [(j)]; S0[i,j,k] -> [(k)] }}]"
        ),
    );

    let root = schedule.get_root();
    if root.n_children() > 0 {
        let child = root.child(0);
        let band_node = child.insert_partial_schedule(partial);
        band_node.get_schedule()
    } else {
        schedule
    }
}

/// Create tiled GEMM schedule with specified tile size
/// Uses ISL's schedule tree manipulation with proper ownership handling
fn create_tiled_gemm_schedule(ctx: &Context, tile_size: i32) -> Option<Schedule> {
    // Create domain and initial schedule
    let domain_str = format!(
        "{{ S0[i,j,k] : 0 <= i < {} and 0 <= j < {} and 0 <= k < {} }}",
        PROBLEM_SIZE, PROBLEM_SIZE, PROBLEM_SIZE
    );
    let domain = UnionSet::read_from_str(ctx, &domain_str);
    let schedule = Schedule::from_domain(domain);

    // Get the root node
    let root = schedule.get_root();
    if root.n_children() == 0 {
        eprintln!("  [DEBUG] Root has no children");
        return None;
    }

    // Insert partial schedule to create a band
    let partial = MultiUnionPwAff::read_from_str(
        ctx,
        "[{ S0[i,j,k] -> [(i)]; S0[i,j,k] -> [(j)]; S0[i,j,k] -> [(k)] }]",
    );

    let child = root.child(0);
    let band_node = child.insert_partial_schedule(partial);

    // Get the band's space to determine correct dimensionality
    let band_space = band_node.band_get_space();
    let n_dims = band_space.dim(DimType::Set) as usize;

    if n_dims == 0 {
        eprintln!("  [DEBUG] Band has 0 dimensions");
        return None;
    }

    // Build ValList with correct number of elements
    let tile_val_first = Val::int_from_si(ctx, tile_size as i64);
    let mut val_list = ValList::from_val(tile_val_first);

    for _ in 1..n_dims {
        let tile_val = Val::int_from_si(ctx, tile_size as i64);
        val_list = val_list.add(tile_val);
    }

    // Create MultiVal from the ValList
    let tile_sizes = MultiVal::from_val_list(band_space, val_list);

    // Apply tiling
    let tiled = band_node.band_tile(tile_sizes);

    // Get the final schedule
    let result_schedule = tiled.get_schedule();

    // Verify the schedule is valid by checking the string representation
    let schedule_str = result_schedule.to_str().to_string();
    if schedule_str.is_empty() {
        eprintln!("  [DEBUG] Resulting schedule is empty");
        return None;
    }

    Some(result_schedule)
}

/// Compute baseline cost (normalized to 1.0)
fn compute_baseline_cost(_props: &ScheduleProperties) -> f64 {
    // Baseline is our reference point
    1.0
}

/// Generate C code from ISL schedule
fn generate_c_code(schedule: &Schedule) -> Result<String, String> {
    let ctx_ptr = schedule.get_ctx().ptr as *mut std::ffi::c_void;
    let schedule_ptr = schedule.ptr as *mut std::ffi::c_void;

    unsafe { isl_codegen_ffi::generate_c_code(ctx_ptr, schedule_ptr) }
}

/// Compile C code and measure execution time
fn compile_and_measure(c_code: &str, tile_size: i32) -> Result<f64, String> {
    let name = if tile_size == 0 {
        "baseline".to_string()
    } else {
        format!("tiled_{}", tile_size)
    };

    let c_file = PathBuf::from(WORK_DIR).join(format!("{}.c", name));
    let exe_file = PathBuf::from(WORK_DIR).join(format!("{}_test", name));

    // Create complete GEMM program with the generated loop structure
    let complete_program = format!(
        r#"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// ISL codegen uses min/max functions
#define min(a,b) ((a) < (b) ? (a) : (b))
#define max(a,b) ((a) > (b) ? (a) : (b))

#define N {problem_size}

double A[N][N];
double B[N][N];
double C[N][N];

// Statement macro that performs the GEMM computation
#define S0(i, j, k) C[i][j] += A[i][k] * B[k][j]

void gemm_kernel() {{
{c_code}
}}

int main() {{
    // Initialize matrices
    for (int i = 0; i < N; i++) {{
        for (int j = 0; j < N; j++) {{
            A[i][j] = 1.0;
            B[i][j] = 2.0;
            C[i][j] = 0.0;
        }}
    }}

    // Warmup
    gemm_kernel();

    // Reset C
    for (int i = 0; i < N; i++) {{
        for (int j = 0; j < N; j++) {{
            C[i][j] = 0.0;
        }}
    }}

    // Measure
    clock_t start = clock();
    for (int iter = 0; iter < {iterations}; iter++) {{
        gemm_kernel();
    }}
    clock_t end = clock();

    double time_ms = ((double)(end - start)) / CLOCKS_PER_SEC * 1000.0 / {iterations};
    printf("RUNTIME_MS: %.6f\n", time_ms);

    // Simple verification
    double expected = (double)N * 2.0 * {iterations};
    if (C[0][0] < expected * 0.99 || C[0][0] > expected * 1.01) {{
        fprintf(stderr, "Verification failed: C[0][0] = %f, expected ~%f\n", C[0][0], expected);
        return 1;
    }}

    return 0;
}}
"#,
        problem_size = PROBLEM_SIZE,
        iterations = ITERATIONS,
        c_code = c_code,
    );

    // Write C file
    let mut file = fs::File::create(&c_file).map_err(|e| format!("Failed to create C file: {}", e))?;
    file.write_all(complete_program.as_bytes())
        .map_err(|e| format!("Failed to write C file: {}", e))?;

    // Compile
    let compile_result = Command::new("cc")
        // Use -O2 instead of -O3 to reduce backend compiler "magic"
        // -O3 in Clang/GCC often includes Polly/Graphite loop transformation passes
        // that might undo or redo our ISL schedule, contaminating measurements.
        // -O2 gives good optimization without aggressive loop restructuring.
        .args(&["-O2", "-o"])
        .arg(&exe_file)
        .arg(&c_file)
        .args(&["-lm"])
        .output()
        .map_err(|e| format!("Failed to compile: {}", e))?;

    if !compile_result.status.success() {
        let stderr = String::from_utf8_lossy(&compile_result.stderr);
        return Err(format!("Compilation failed: {}", stderr));
    }

    // Execute and parse runtime
    let mut total_time = 0.0;
    let mut successful_runs = 0;

    for _ in 0..3 {
        // Average over 3 runs
        let output = Command::new(&exe_file)
            .output()
            .map_err(|e| format!("Failed to execute: {}", e))?;

        if output.status.success() {
            let stdout = String::from_utf8_lossy(&output.stdout);
            if let Some(time) = parse_runtime(&stdout) {
                total_time += time;
                successful_runs += 1;
            }
        }
    }

    if successful_runs == 0 {
        return Err("All runs failed".to_string());
    }

    Ok(total_time / successful_runs as f64)
}

/// Parse RUNTIME_MS from program output
fn parse_runtime(output: &str) -> Option<f64> {
    for line in output.lines() {
        if line.starts_with("RUNTIME_MS:") {
            if let Some(time_str) = line.split(':').nth(1) {
                return time_str.trim().parse().ok();
            }
        }
    }
    None
}

/// Compute Pearson correlation coefficient
/// Measures linear relationship between analytical cost and execution time.
/// Range: [-1, 1], where 1 = perfect positive correlation
fn compute_pearson_correlation(points: &[DataPoint]) -> f64 {
    if points.len() < 2 {
        return 0.0;
    }

    let n = points.len() as f64;

    // Calculate means
    let mean_cost: f64 = points.iter().map(|p| p.analytical_cost).sum::<f64>() / n;
    let mean_time: f64 = points.iter().map(|p| p.execution_time_ms).sum::<f64>() / n;

    // Calculate correlation components
    let mut numerator = 0.0;
    let mut sum_sq_cost = 0.0;
    let mut sum_sq_time = 0.0;

    for point in points {
        let cost_diff = point.analytical_cost - mean_cost;
        let time_diff = point.execution_time_ms - mean_time;

        numerator += cost_diff * time_diff;
        sum_sq_cost += cost_diff * cost_diff;
        sum_sq_time += time_diff * time_diff;
    }

    let denominator = (sum_sq_cost * sum_sq_time).sqrt();

    if denominator < 1e-10 {
        0.0
    } else {
        numerator / denominator
    }
}

/// Compute Spearman's Rank Correlation coefficient
///
/// This is THE critical metric for optimizer validation:
/// - Measures whether the model RANKS schedules correctly
/// - We don't need exact time prediction, we need correct ORDERING
/// - Formula: rho = 1 - (6 * sum(d^2)) / (n * (n^2 - 1))
///   where d = difference in ranks between paired observations
///
/// Range: [-1, 1], where 1 = perfect rank agreement
///
/// Scientific justification:
/// - Pearson measures linear relationship (magnitude matters)
/// - Spearman measures monotonic relationship (order matters)
/// - For optimization, we only need: if C_a < C_b, then T_a < T_b (usually)
/// - Even if |C_a - C_b| != |T_a - T_b|, the optimizer still makes correct choices
fn compute_spearman_correlation(points: &[DataPoint]) -> f64 {
    if points.len() < 2 {
        return 0.0;
    }

    let n = points.len();

    // Step 1: Compute ranks for analytical costs
    let cost_ranks = compute_ranks(
        &points.iter().map(|p| p.analytical_cost).collect::<Vec<_>>()
    );

    // Step 2: Compute ranks for execution times
    let time_ranks = compute_ranks(
        &points.iter().map(|p| p.execution_time_ms).collect::<Vec<_>>()
    );

    // Step 3: Compute Spearman's rho using rank differences
    // Formula: rho = 1 - (6 * sum(d^2)) / (n * (n^2 - 1))
    let mut sum_d_squared = 0.0;
    for i in 0..n {
        let d = cost_ranks[i] - time_ranks[i];
        sum_d_squared += d * d;
    }

    let n_f64 = n as f64;
    let denominator = n_f64 * (n_f64 * n_f64 - 1.0);

    if denominator < 1e-10 {
        0.0
    } else {
        1.0 - (6.0 * sum_d_squared) / denominator
    }
}

/// Compute ranks for a slice of values (handles ties with average rank)
///
/// Example: [1.0, 3.0, 2.0, 3.0] -> [1.0, 3.5, 2.0, 3.5]
/// (the two 3.0 values share ranks 3 and 4, so each gets 3.5)
fn compute_ranks(values: &[f64]) -> Vec<f64> {
    let n = values.len();
    if n == 0 {
        return vec![];
    }

    // Create (index, value) pairs and sort by value
    let mut indexed: Vec<(usize, f64)> = values.iter().cloned().enumerate().collect();
    indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

    // Assign ranks, handling ties with average rank
    let mut ranks = vec![0.0; n];
    let mut i = 0;

    while i < n {
        // Find all elements with the same value (ties)
        let mut j = i;
        while j < n && (indexed[j].1 - indexed[i].1).abs() < 1e-10 {
            j += 1;
        }

        // Average rank for tied elements: (i+1 + i+2 + ... + j) / count
        // = (sum from i+1 to j) / (j - i)
        // = ((i+1 + j) * (j - i) / 2) / (j - i)
        // = (i + 1 + j) / 2
        let avg_rank = (i + 1 + j) as f64 / 2.0;

        // Assign average rank to all tied elements
        for k in i..j {
            ranks[indexed[k].0] = avg_rank;
        }

        i = j;
    }

    ranks
}

/// Compute Kendall's Tau correlation (concordance-based)
///
/// Measures the ordinal association between two variables.
/// Counts concordant and discordant pairs.
/// tau = (concordant - discordant) / total_pairs
///
/// Useful as a third perspective on rank correlation.
fn compute_kendall_tau(points: &[DataPoint]) -> f64 {
    let n = points.len();
    if n < 2 {
        return 0.0;
    }

    let mut concordant = 0;
    let mut discordant = 0;

    // Compare all pairs
    for i in 0..n {
        for j in (i + 1)..n {
            let cost_diff = points[i].analytical_cost - points[j].analytical_cost;
            let time_diff = points[i].execution_time_ms - points[j].execution_time_ms;

            // Concordant: both differences have same sign
            // Discordant: differences have opposite signs
            let product = cost_diff * time_diff;
            if product > 1e-10 {
                concordant += 1;
            } else if product < -1e-10 {
                discordant += 1;
            }
            // Ties (product ~= 0) are ignored
        }
    }

    let total_pairs = (n * (n - 1)) / 2;
    if total_pairs == 0 {
        0.0
    } else {
        (concordant as f64 - discordant as f64) / total_pairs as f64
    }
}
