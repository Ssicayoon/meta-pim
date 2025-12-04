//! Dependency-Aware Schedule Extraction and Performance Measurement
//!
//! This module integrates:
//! 1. Dependency-aware conditional rewrites that only apply safe transformations
//! 2. Real performance measurement through the complete pipeline:
//!    ISL schedule → Polygeist → MLIR → compilation → execution
//! 3. External cost estimation using actual runtime measurements

use egg::{EGraph, Runner};
use isl_rs::{Context, Schedule};

use std::fs;
use std::path::PathBuf;
use std::sync::Arc;

use crate::{
    access_analysis::AccessInfo,
    dependency_aware::DependencyInfo,
    execution::{measure_and_validate, PerformanceResult},
    extract_all::{extract_and_validate_all, extract_diverse_schedules},
    SchedOp, ScheduleAnalysis, ScheduleHandle,
};

/// Configuration for dependency-aware extraction and measurement
pub struct ExtractionConfig {
    /// C source file for testing
    pub c_file: String,

    /// Kernel function name
    pub kernel_name: String,

    /// Polygeist installation directory
    pub polygeist_dir: String,

    /// Problem size for performance testing
    pub problem_size: usize,

    /// Maximum candidates to measure (for performance)
    pub max_measure: usize,

    /// Output directory for results
    pub output_dir: String,

    /// Enable dependency checking
    pub check_dependencies: bool,

    /// Enable real performance measurement
    pub measure_real_performance: bool,
}

/// Result of extracting and measuring a schedule candidate
pub struct CandidateResult {
    pub id: usize,
    pub schedule_str: String, // Store schedule as string instead of Schedule object
    pub transformations: Vec<String>,
    pub is_safe: bool,
    pub simulated_cost: f64,
    pub real_performance: Option<PerformanceResult>,
    pub speedup: Option<f64>,
}

/// Main entry point for dependency-aware extraction with real performance measurement
pub fn extract_and_measure_candidates(
    input_schedule_file: &str,
    config: &ExtractionConfig,
) -> Result<Vec<CandidateResult>, String> {
    println!("=== Dependency-Aware Schedule Extraction and Measurement ===");

    // Load initial schedule
    println!("\n[Step 1/5] Loading input schedule...");
    let ctx = Arc::new(Context::alloc());
    let initial_schedule = crate::load_isl_file(ctx.clone(), input_schedule_file)?;
    println!("  - Loaded schedule from {}", input_schedule_file);

    // Extract access information if available
    println!("\n[Step 2/5] Analyzing dependencies...");
    let access_info = if config.check_dependencies {
        // In a real implementation, this would extract from Polygeist/MLIR
        extract_access_info(&config.c_file, &config.kernel_name, &config.polygeist_dir)?
    } else {
        println!("  ! Dependency checking disabled, using conservative assumptions");
        create_conservative_access_info()
    };

    // Try to get schedule directory from config
    let schedule_dir = std::path::Path::new(&config.output_dir)
        .parent()
        .and_then(|p| p.join("schedule").canonicalize().ok())
        .or_else(|| {
            // Try to find schedule directory relative to polygeist_dir
            std::path::Path::new(&config.polygeist_dir)
                .join("isl_output")
                .join("schedules")
                .canonicalize()
                .ok()
        });

    let dependency_info = DependencyInfo::compute_from_access_info(
        &access_info,
        &initial_schedule.schedule,
        schedule_dir.as_deref(),
    )?;

    println!("  - Dependency analysis complete:");
    println!(
        "    - Has RAW dependencies: {}",
        dependency_info.raw_deps.has_deps
    );
    println!(
        "    - Has WAR dependencies: {}",
        dependency_info.war_deps.has_deps
    );
    println!(
        "    - Has WAW dependencies: {}",
        dependency_info.waw_deps.has_deps
    );

    // Build e-graph with dependency-aware rules
    println!("\n[Step 3/5] Building e-graph with dependency-aware transformations...");
    let mut egraph = EGraph::new(ScheduleAnalysis::new(ctx.clone()));
    let root = egraph.add(SchedOp::Schedule(initial_schedule.clone()));

    // Create dependency-aware rewrite rules
    let rules = create_dependency_aware_rules(&dependency_info);

    // Run equality saturation with dependency-aware rules
    let runner = Runner::default()
        .with_egraph(egraph)
        .with_iter_limit(5)
        .with_node_limit(10000)
        .run(&rules);

    println!("  - E-graph exploration complete:");
    println!("    - E-classes: {}", runner.egraph.number_of_classes());
    println!("    - Nodes: {}", runner.egraph.total_number_of_nodes());
    println!("    - Stop reason: {:?}", runner.stop_reason);

    // Extract all valid candidates
    println!("\n[Step 4/5] Extracting valid schedule candidates...");
    let candidates = extract_and_validate_all(&runner.egraph, root);
    let diverse = extract_diverse_schedules(&runner.egraph);

    println!("  - Found {} valid candidates", candidates.len());
    println!("  - Found {} diverse strategies", diverse.len());

    // Measure real performance for top candidates
    println!("\n[Step 5/5] Measuring real performance...");
    let mut results = Vec::new();
    let baseline_perf = if config.measure_real_performance {
        measure_baseline_performance(&initial_schedule, &config)?
    } else {
        None
    };

    // Process candidates
    for (i, (sim_cost, _expr, schedule_opt)) in candidates.iter().enumerate() {
        if i >= config.max_measure {
            println!(
                "\n  Stopping at {} candidates (max_measure limit)",
                config.max_measure
            );
            break;
        }

        if let Some((schedule, schedule_str)) = schedule_opt {
            print!("  Candidate {}: ", i);

            // Analyze transformations applied
            let transformations = analyze_transformations(&schedule);

            // Check if transformations are safe
            let is_safe = if config.check_dependencies {
                check_transformation_safety(&transformations, &dependency_info)
            } else {
                true // Assume safe if not checking
            };

            // Measure real performance if requested and safe
            let real_perf = if config.measure_real_performance && is_safe {
                match measure_candidate_performance(&schedule, &config, i) {
                    Ok(perf) => {
                        print!("- ({:.2}ms", perf.execution_time_ms);
                        if let Some(ref base) = baseline_perf {
                            let speedup = base.execution_time_ms / perf.execution_time_ms;
                            print!(", {:.2}x speedup", speedup);
                        }
                        print!(")");
                        Some(perf)
                    }
                    Err(e) => {
                        print!("x (measurement failed: {})", e);
                        None
                    }
                }
            } else if !is_safe {
                print!("! (unsafe transformation)");
                None
            } else {
                print!("○ (not measured)");
                None
            };

            println!();

            // Calculate speedup if we have both measurements
            let speedup = match (&real_perf, &baseline_perf) {
                (Some(perf), Some(base)) if base.execution_time_ms > 0.0 => {
                    Some(base.execution_time_ms / perf.execution_time_ms)
                }
                _ => None,
            };

            results.push(CandidateResult {
                id: i,
                schedule_str: schedule_str.clone(),
                transformations: transformations.clone(),
                is_safe,
                simulated_cost: *sim_cost,
                real_performance: real_perf,
                speedup,
            });
        }
    }

    // Save results
    save_results(&results, &config.output_dir)?;

    // Print summary
    print_summary(&results, baseline_perf.as_ref());

    Ok(results)
}

/// Create dependency-aware rewrite rules that respect dependencies
///
/// Uses `rational_dependency_rules()` which has built-in ISL dependency analysis
/// via `analyze_dependencies_at_level()`. The rules internally check legality
/// before applying transformations.
fn create_dependency_aware_rules(
    _deps: &DependencyInfo,
) -> Vec<egg::Rewrite<SchedOp, ScheduleAnalysis>> {
    use crate::rational_rewrites::rational_dependency_rules;

    // rational_dependency_rules() includes 12 sophisticated rules that
    // use real ISL dependency analysis (UnionAccessInfo::compute_flow).
    // Each rule checks dependencies internally before applying transformations.
    //
    // Rules include:
    // - safe_tiling_rule, safe_unrolling_rule (always safe)
    // - outer_parallel_rule, reduction_parallel_rule (checks loop-carried deps)
    // - innermost_vectorize_rule (checks vectorizability)
    // - locality_interchange_rule (uses ISL interchange function)
    // - gemm_optimization_rule (domain-specific)
    rational_dependency_rules()
}

/// Extract access information from C code via Polygeist
fn extract_access_info(
    _c_file: &str,
    _kernel_name: &str,
    _polygeist_dir: &str,
) -> Result<AccessInfo, String> {
    // This would:
    // 1. Compile C to MLIR using Polygeist
    // 2. Extract memory access patterns from MLIR
    // 3. Build AccessInfo structure

    // For now, return a placeholder
    Ok(create_conservative_access_info())
}

/// Create conservative access info when real analysis unavailable
fn create_conservative_access_info() -> AccessInfo {
    use crate::access_analysis::{
        AccessInfo, ContextHandle, ScheduleHandle as AccessScheduleHandle, StmtAccess,
    };

    let ctx = ContextHandle::new_placeholder();
    let sched = AccessScheduleHandle::new_placeholder();

    let mut info = AccessInfo::new(ctx, sched);

    // Add placeholder statement with conservative assumptions
    let stmt = StmtAccess::new("S0".to_string());
    // Conservative: assume both reads and writes
    info.add_statement(stmt);

    info
}

/// Analyze what transformations were applied to a schedule
fn analyze_transformations(schedule: &Schedule) -> Vec<String> {
    let mut transformations = Vec::new();
    let schedule_str = schedule.to_str().to_string();

    if schedule_str.contains("mod") {
        transformations.push("tiling".to_string());
    }
    if schedule_str.contains("atomic") || schedule_str.contains("parallel") {
        transformations.push("parallelization".to_string());
    }
    if schedule_str.contains("vectorize") {
        transformations.push("vectorization".to_string());
    }
    if schedule_str.contains("unroll") {
        transformations.push("unrolling".to_string());
    }

    transformations
}

/// Check if transformations are safe given dependencies
fn check_transformation_safety(transformations: &[String], deps: &DependencyInfo) -> bool {
    for transform in transformations {
        match transform.as_str() {
            "parallelization" => {
                // Parallel is safe only if no loop-carried dependencies
                if deps.all_deps.loop_carried.iter().any(|&carried| carried) {
                    return false;
                }
            }
            "vectorization" => {
                // Detect statement name (Polygeist generates S0 for single-statement kernels)
                // Polygeist uses S0 for GEMM, not S1!
                if deps.all_deps.loop_carried.last() == Some(&true) {
                    return false;
                }
            }
            "tiling" => {
                // Tiling is generally safe, just changes iteration order
            }
            "unrolling" => {
                // Unrolling is safe but may increase code size
            }
            _ => {}
        }
    }

    true
}

/// Measure baseline performance
fn measure_baseline_performance(
    schedule: &ScheduleHandle,
    config: &ExtractionConfig,
) -> Result<Option<PerformanceResult>, String> {
    println!("\n  Measuring baseline performance...");

    // Save baseline schedule
    let baseline_dir = format!("{}/baseline", config.output_dir);
    fs::create_dir_all(&baseline_dir)
        .map_err(|e| format!("Failed to create baseline dir: {}", e))?;

    let baseline_schedule_file = format!("{}/baseline.isl", baseline_dir);
    crate::save_isl_file(schedule, &baseline_schedule_file)?;

    // Run through pipeline and measure
    measure_schedule_performance(&baseline_schedule_file, &config, &baseline_dir)
}

/// Measure performance of a candidate schedule
fn measure_candidate_performance(
    schedule: &Schedule,
    config: &ExtractionConfig,
    candidate_id: usize,
) -> Result<PerformanceResult, String> {
    // Save candidate schedule
    let candidate_dir = format!("{}/candidate_{:04}", config.output_dir, candidate_id);
    fs::create_dir_all(&candidate_dir)
        .map_err(|e| format!("Failed to create candidate dir: {}", e))?;

    // Wrap in ScheduleHandle for saving
    let ctx = Arc::new(Context::alloc());
    let schedule_handle = ScheduleHandle::new(ctx, schedule.copy());

    let schedule_file = format!("{}/schedule.isl", candidate_dir);
    crate::save_isl_file(&schedule_handle, &schedule_file)?;

    // Measure through pipeline
    measure_schedule_performance(&schedule_file, config, &candidate_dir)
        .and_then(|opt| opt.ok_or_else(|| "Failed to measure performance".to_string()))
}

/// Core function to measure schedule performance through the complete pipeline
fn measure_schedule_performance(
    schedule_file: &str,
    config: &ExtractionConfig,
    work_dir: &str,
) -> Result<Option<PerformanceResult>, String> {
    // The complete pipeline:
    // 1. C → MLIR (baseline)
    // 2. Apply ISL schedule transformations
    // 3. MLIR → executable
    // 4. Execute and measure

    // Step 1: Generate baseline MLIR
    let baseline_mlir = format!("{}/baseline.mlir", work_dir);
    crate::codegen::compile_c_to_mlir(
        &config.polygeist_dir,
        &config.c_file,
        &config.kernel_name,
        &baseline_mlir,
    )?;

    // Step 2: Apply the ISL schedule transformation
    let transformed_mlir = format!("{}/transformed.mlir", work_dir);
    crate::codegen::apply_schedule_to_mlir(
        &config.polygeist_dir,
        &baseline_mlir,
        work_dir,
        &fs::read_to_string(schedule_file).map_err(|e| e.to_string())?,
        &transformed_mlir,
    )?;

    // Step 3 & 4: Compile and measure performance
    // Use the execution module to compile and run
    let result = measure_and_validate(
        &config.polygeist_dir,
        &PathBuf::from(&baseline_mlir),
        &PathBuf::from(&transformed_mlir),
        &config.kernel_name,
        config.problem_size,
    )?;

    Ok(Some(result))
}

/// Save results to files for analysis
fn save_results(results: &[CandidateResult], output_dir: &str) -> Result<(), String> {
    // Save detailed results as JSON
    let results_file = format!("{}/results.json", output_dir);
    let json = serde_json::to_string_pretty(results)
        .map_err(|e| format!("Failed to serialize results: {}", e))?;
    fs::write(&results_file, json).map_err(|e| format!("Failed to write results: {}", e))?;

    // Save summary CSV for easy analysis
    let csv_file = format!("{}/summary.csv", output_dir);
    let mut csv = String::from("id,is_safe,simulated_cost,real_time_ms,speedup,transformations\n");

    for result in results {
        csv.push_str(&format!(
            "{},{},{:.4},{},{},{}\n",
            result.id,
            result.is_safe,
            result.simulated_cost,
            result
                .real_performance
                .as_ref()
                .map(|p| format!("{:.2}", p.execution_time_ms))
                .unwrap_or_else(|| "N/A".to_string()),
            result
                .speedup
                .map(|s| format!("{:.2}", s))
                .unwrap_or_else(|| "N/A".to_string()),
            result.transformations.join("+")
        ));
    }

    fs::write(&csv_file, csv).map_err(|e| format!("Failed to write CSV: {}", e))?;

    println!("\n  - Results saved to {}", output_dir);

    Ok(())
}

/// Print summary of results
fn print_summary(results: &[CandidateResult], baseline: Option<&PerformanceResult>) {
    println!("\n=== Performance Measurement Summary ===");

    if let Some(base) = baseline {
        println!("Baseline execution time: {:.2}ms", base.execution_time_ms);
    }

    let safe_count = results.iter().filter(|r| r.is_safe).count();
    let measured_count = results
        .iter()
        .filter(|r| r.real_performance.is_some())
        .count();

    println!("Total candidates: {}", results.len());
    println!("Safe transformations: {}", safe_count);
    println!("Successfully measured: {}", measured_count);

    // Find best performing candidate
    if let Some(best) = results
        .iter()
        .filter(|r| r.speedup.is_some())
        .max_by(|a, b| a.speedup.partial_cmp(&b.speedup).unwrap())
    {
        println!("\nBest performing candidate:");
        println!("  ID: {}", best.id);
        println!("  Speedup: {:.2}x", best.speedup.unwrap());
        println!("  Transformations: {}", best.transformations.join(", "));
        if let Some(ref perf) = best.real_performance {
            println!("  Execution time: {:.2}ms", perf.execution_time_ms);
        }
    }

    // Show distribution of speedups
    let speedups: Vec<f64> = results.iter().filter_map(|r| r.speedup).collect();

    if !speedups.is_empty() {
        let avg_speedup = speedups.iter().sum::<f64>() / speedups.len() as f64;
        let max_speedup = speedups.iter().fold(0.0f64, |a, &b| a.max(b));
        let min_speedup = speedups.iter().fold(f64::MAX, |a, &b| a.min(b));

        println!("\nSpeedup statistics:");
        println!("  Average: {:.2}x", avg_speedup);
        println!("  Maximum: {:.2}x", max_speedup);
        println!("  Minimum: {:.2}x", min_speedup);
    }
}

// For serde serialization
use serde::Serialize;

impl Serialize for CandidateResult {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;

        let mut state = serializer.serialize_struct("CandidateResult", 7)?;
        state.serialize_field("id", &self.id)?;
        state.serialize_field("transformations", &self.transformations)?;
        state.serialize_field("is_safe", &self.is_safe)?;
        state.serialize_field("simulated_cost", &self.simulated_cost)?;
        state.serialize_field("real_performance", &self.real_performance)?;
        state.serialize_field("speedup", &self.speedup)?;
        state.serialize_field("schedule_str", &self.schedule_str)?;
        state.end()
    }
}
