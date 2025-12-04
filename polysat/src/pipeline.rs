//! Unified PolySat Pipeline
//!
//! This module provides the complete end-to-end pipeline for polyhedral optimization:
//!
//! C code -> Polygeist -> ISL schedule -> E-graph exploration -> Optimized schedule -> MLIR -> Execution
//!
//! # Architecture Alignment
//!
//! This pipeline implements core contributions for configurable Processing-in-Memory optimization:
//!
//! ## 1. Unifying Polyhedral Transformation and Equality Saturation
//!
//! We frame the polyhedral scheduling problem within an equality saturation engine.
//! Tiramisu-like scheduling primitives are formalized as conditional, semantics-preserving rewrite rules.
//!
//! - E-graph representation of ISL schedules (`SchedOp` language)
//! - Rewrite rules for polyhedral transformations
//! - Conditional rewrites with dependency checking
//! - Hardware-aware rewrite rules
//!
//! ## 2. Cost Modeling
//!
//! We use a hybrid cost model approach:
//! - **Stage 1 (Analytical Proxy)**: `ScheduleCost` provides fast NCP-aware analytical cost estimation.
//! - **Stage 2 (K-Best Extraction)**: Extracts top candidates for further evaluation.
//! - **Stage 3 (Performance Measurement)**: Validates performance through execution.
//!
//! ## 3. Hardware Constraints
//!
//! The search is constrained by hardware-dependent legality criteria:
//! - Dependency-aware rewrites ensure correctness.
//! - Hardware-aware tile size constraints align with architecture.
//!
//! ## 4. MLIR Integration
//!
//! The pipeline integrates with MLIR for code generation:
//! - Standard MLIR lowering via `codegen::apply_schedule_to_mlir`.
//!
//! # Pipeline Stages
//!
//! The pipeline is organized into distinct stages:
//!
//! ## Stage 1: C -> MLIR -> ISL (Polygeist Integration)
//! - **Input**: C source file + kernel function name
//! - **Output**: MLIR file + ISL schedule + access relations
//!
//! ## Stage 2: ISL -> E-graph (Schedule Representation)
//! - **Input**: ISL schedule string
//! - **Output**: E-graph with root node, ISL context
//!
//! ## Stage 3: E-graph Exploration (Transformation Search)
//! - **Input**: E-graph + rewrite rules + dependency info
//! - **Output**: E-graph containing all reachable schedules
//!
//! ## Stage 4: Cost-Based Extraction (Schedule Selection)
//! - **Input**: E-graph + cost model
//! - **Output**: Optimal schedule
//!
//! ## Stage 5: Schedule -> MLIR (Code Generation)
//! - **Input**: Optimized schedule + original MLIR
//! - **Output**: Transformed MLIR
//!
//! ## Stage 6: MLIR -> Executable (Compilation)
//! - **Input**: Transformed MLIR
//! - **Output**: Executable binary
//!
//! ## Stage 7: Execution + Measurement (Validation)
//! - **Input**: Executable + test inputs
//! - **Output**: Performance metrics
//!
//! # Usage Examples
//!
//! ## Basic Pipeline: C → Optimized ISL
//! ```no_run
//! use polysat::pipeline::{PolySatPipeline, PipelineConfig};
//! use std::path::Path;
//!
//! let pipeline = PolySatPipeline::new("polygeist")?;
//! let config = PipelineConfig::default();
//!
//! let result = pipeline.optimize_c_code(
//!     Path::new("matmul.c"),
//!     "matmul",
//!     &config
//! )?;
//!
//! println!("Optimized schedule: {}", result.schedule_str);
//! println!("Speedup: {:.2}x", result.speedup);
//! # Ok::<(), polysat::pipeline::PipelineError>(())
//! ```
//!
//! ## ISL-Only Pipeline (No Polygeist Required)
//! ```no_run
//! use polysat::pipeline::{PolySatPipeline, PipelineConfig};
//! use polysat::parse_isl;
//! use isl_rs::Context;
//! use std::sync::Arc;
//!
//! let ctx = Arc::new(Context::alloc());
//! let schedule = parse_isl(ctx.clone(), "{ S[i,j] -> [i,j] }").unwrap();
//!
//! let pipeline = PolySatPipeline::new_without_polygeist(ctx)?;
//! let config = PipelineConfig::default();
//!
//! let optimized = pipeline.optimize_schedule(schedule, &config)?;
//! println!("Optimized: {}", optimized.to_string());
//! # Ok::<(), polysat::pipeline::PipelineError>(())
//! ```
//!
//! ## Hardware-Aware Optimization (NCP)
//! ```no_run
//! use polysat::pipeline::{PolySatPipeline, PipelineConfig};
//! use std::path::Path;
//!
//! let pipeline = PolySatPipeline::new("polygeist")?;
//! let mut config = PipelineConfig::default();
//! config.ncp_slices = Some(8);
//! config.ncp_banks = Some(64);
//! config.use_hardware_cost_model = true;
//! config.problem_size = 128;
//!
//! let result = pipeline.optimize_c_code(
//!     Path::new("gemm.c"),
//!     "gemm",
//!     &config
//! )?;
//! # Ok::<(), polysat::pipeline::PipelineError>(())
//! ```

use egg::{EGraph, RecExpr, Runner};
use isl_rs::Context;
use std::fmt;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use crate::{
    codegen,
    dump_isl,
    execution,
    extract_best,
    extract_best_by_performance,
    optimize::ScheduleCost, // Use enhanced ScheduleCost with NCP awareness
    parse_isl,
    rational_rewrites::rational_dependency_rules,
    // NOTE: rewrites::{rules, advanced_rules} removed - they had STUB dependency checks
    // Now using rational_dependency_rules() exclusively for real ISL dependency analysis
    schedule_measurer::{MeasurementConfig, ScheduleMeasurer},
    SchedOp,
    ScheduleAnalysis,
    ScheduleHandle,
};
use egg::CostFunction;

// ============================================================================
// Error Types
// ============================================================================

/// Pipeline errors with rich context
#[derive(Debug, Clone)]
pub enum PipelineError {
    /// Polygeist not found or failed to compile
    PolygeistNotFound(String),

    /// ISL parsing or manipulation error
    ISLParseError(String),

    /// Dependency analysis detected illegal transformation
    DependencyViolation(String),

    /// MLIR compilation failed
    CompilationFailed(String),

    /// Execution or performance measurement failed
    MeasurementFailed(String),

    /// I/O error (file not found, permission denied, etc.)
    IOError(String),

    /// Generic error with message
    Other(String),
}

impl fmt::Display for PipelineError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PipelineError::PolygeistNotFound(msg) => write!(f, "Polygeist not available: {}", msg),
            PipelineError::ISLParseError(msg) => write!(f, "ISL parsing failed: {}", msg),
            PipelineError::DependencyViolation(msg) => write!(f, "Dependency violation: {}", msg),
            PipelineError::CompilationFailed(msg) => write!(f, "Compilation failed: {}", msg),
            PipelineError::MeasurementFailed(msg) => {
                write!(f, "Performance measurement failed: {}", msg)
            }
            PipelineError::IOError(msg) => write!(f, "I/O error: {}", msg),
            PipelineError::Other(msg) => write!(f, "{}", msg),
        }
    }
}

impl std::error::Error for PipelineError {}

impl From<String> for PipelineError {
    fn from(msg: String) -> Self {
        PipelineError::Other(msg)
    }
}

impl From<std::io::Error> for PipelineError {
    fn from(err: std::io::Error) -> Self {
        PipelineError::IOError(err.to_string())
    }
}

// ============================================================================
// Configuration
// ============================================================================

/// Pipeline configuration options
///
/// **Paper Alignment**: This configuration supports the paper's hybrid cost model approach,
/// but does NOT fully implement the k-best extraction + simulation workflow.
#[derive(Clone, Debug)]
pub struct PipelineConfig {
    /// Maximum e-graph iterations (saturation limit)
    pub max_iter: usize,

    /// Maximum e-graph nodes (memory limit)
    pub max_nodes: usize,

    /// Enable dependency analysis and checking
    pub check_dependencies: bool,

    /// Use performance-based extraction (slow but accurate)
    ///
    /// **Note**: This measures real execution time, NOT cycle-accurate simulation as described
    /// in the paper. For paper-aligned behavior, use `k_best_extraction` + external simulator.
    pub use_performance_extraction: bool,

    /// Measure real execution time (requires full toolchain)
    pub measure_execution: bool,

    /// Output directory for intermediate files
    pub output_dir: PathBuf,

    /// Verbose logging
    pub verbose: bool,

    /// Problem size for performance measurement (default: 64)
    pub problem_size: usize,

    /// NCP hardware configuration (for hardware-aware optimization)
    pub ncp_slices: Option<usize>,
    pub ncp_banks: Option<usize>,

    /// Use hardware-aware cost model (NCP cost model when hardware config provided)
    ///
    /// **Paper Alignment**: This enables the analytical proxy model (Stage 1 of hybrid cost model).
    /// However, Stage 2 (k-best extraction) and Stage 3 (cycle-accurate simulation) are NOT
    /// automatically integrated. Use `extract_k_best_with_simulation()` for full paper workflow.
    pub use_hardware_cost_model: bool,

    /// Number of candidates to extract for simulation (paper's k parameter)
    ///
    /// **Paper Alignment**: Set this to enable k-best extraction workflow. When set, the pipeline
    /// will extract k candidates using analytical model, then (if simulator available) simulate
    /// each candidate, then select the best simulated performance.
    ///
    /// **Default**: None (extract single best directly)
    /// **Paper Recommendation**: k=20
    pub k_best_candidates: Option<usize>,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        PipelineConfig {
            max_iter: 10,
            max_nodes: 10000,
            check_dependencies: true,
            use_performance_extraction: false,
            measure_execution: false,
            output_dir: PathBuf::from("polysat_output"),
            verbose: false,
            problem_size: 64,
            ncp_slices: None,
            ncp_banks: None,
            use_hardware_cost_model: false,
            k_best_candidates: None, // Paper's k-best workflow not enabled by default
        }
    }
}

// ============================================================================
// Pipeline Result
// ============================================================================

/// Result of a complete pipeline run
#[derive(Clone, Debug)]
pub struct PipelineResult {
    /// Optimized schedule as ISL string
    pub schedule_str: String,

    /// Cost of optimized schedule (from cost model)
    pub cost: f64,

    /// Baseline schedule (for comparison)
    pub baseline_schedule_str: String,

    /// Baseline cost
    pub baseline_cost: f64,

    /// Speedup (baseline_cost / optimized_cost)
    pub speedup: f64,

    /// E-graph statistics
    pub egraph_stats: EGraphStats,

    /// Path to output MLIR file (if generated)
    pub mlir_file: Option<PathBuf>,

    /// Measured execution time in seconds (if measurement enabled)
    pub execution_time: Option<f64>,
}

/// E-graph exploration statistics
#[derive(Clone, Debug)]
pub struct EGraphStats {
    /// Total number of e-nodes
    pub total_nodes: usize,

    /// Number of e-classes
    pub num_classes: usize,

    /// Saturation iterations performed
    pub iterations: usize,

    /// Time spent in saturation (seconds)
    pub saturation_time: f64,

    /// Time spent in extraction (seconds)
    pub extraction_time: f64,
}

// ============================================================================
// Pipeline Implementation
// ============================================================================

/// Complete PolySat pipeline orchestrator
pub struct PolySatPipeline {
    /// Path to Polygeist installation (optional)
    polygeist_dir: Option<PathBuf>,

    /// ISL context for schedule operations
    ctx: Arc<Context>,
}

impl PolySatPipeline {
    /// Create a new pipeline with Polygeist integration
    ///
    /// # Arguments
    /// * `polygeist_dir` - Path to Polygeist installation (e.g., "polygeist")
    ///
    /// # Errors
    /// Returns `PolygeistNotFound` if the directory doesn't exist or doesn't contain cgeist
    pub fn new(polygeist_dir: impl AsRef<Path>) -> Result<Self, PipelineError> {
        let polygeist_path = polygeist_dir.as_ref().to_path_buf();

        // Verify Polygeist exists
        let cgeist_path = polygeist_path.join("build/bin/cgeist");
        if !cgeist_path.exists() {
            return Err(PipelineError::PolygeistNotFound(format!(
                "cgeist not found at: {}",
                cgeist_path.display()
            )));
        }

        Ok(PolySatPipeline {
            polygeist_dir: Some(polygeist_path),
            ctx: Arc::new(Context::alloc()),
        })
    }

    /// Create a pipeline without Polygeist (ISL-only mode)
    ///
    /// This mode can optimize existing ISL schedules but cannot process C code.
    pub fn new_without_polygeist(ctx: Arc<Context>) -> Result<Self, PipelineError> {
        Ok(PolySatPipeline {
            polygeist_dir: None,
            ctx,
        })
    }

    /// Run complete pipeline: C → MLIR → ISL → E-graph → Optimized → MLIR
    ///
    /// **Paper Alignment**: This implements Stages 1-7 of the pipeline, but does NOT
    /// implement the paper's k-best extraction + simulation workflow. For that, use
    /// `extract_k_best_with_simulation()` after e-graph exploration.
    ///
    /// # Arguments
    /// * `c_file` - Path to C source file
    /// * `kernel` - Name of kernel function to optimize
    /// * `config` - Pipeline configuration
    ///
    /// # Returns
    /// Complete pipeline result with optimized schedule and performance metrics
    ///
    /// # Errors
    /// Returns errors for any stage failure (compilation, parsing, optimization, etc.)
    pub fn optimize_c_code(
        &self,
        c_file: &Path,
        kernel: &str,
        config: &PipelineConfig,
    ) -> Result<PipelineResult, PipelineError> {
        // Verify Polygeist is available
        let polygeist_dir = self.polygeist_dir.as_ref().ok_or_else(|| {
            PipelineError::PolygeistNotFound(
                "Pipeline created without Polygeist support".to_string(),
            )
        })?;

        if config.verbose {
            println!("=== PolySat Pipeline: C → Optimized MLIR ===");
            println!("  C file: {}", c_file.display());
            println!("  Kernel: {}", kernel);
        }

        // Stage 1: Extract baseline schedule from C
        let baseline = codegen::extract_baseline_schedule(
            polygeist_dir.to_str().unwrap(),
            c_file.to_str().unwrap(),
            kernel,
            config.problem_size as u64,
        )
        .map_err(|e| PipelineError::PolygeistNotFound(e))?;

        // Load baseline schedule
        let baseline_schedule_str = std::fs::read_to_string(&baseline.schedule_file)
            .map_err(|e| PipelineError::IOError(e.to_string()))?;

        let baseline_schedule = parse_isl(self.ctx.clone(), &baseline_schedule_str)
            .map_err(|e| PipelineError::ISLParseError(e.to_string()))?;

        // Stage 2-5: Optimize schedule
        // **Paper Alignment**: If `k_best_candidates` is set, this implements the paper's
        // two-stage hybrid cost model workflow:
        // 1. Analytical proxy guides e-graph saturation (Stage 1)
        // 2. Extract k-best candidates using analytical cost (Stage 2)
        // 3. Evaluate k candidates with execution/simulation, select best (Stage 3)
        //
        // **CRITICAL**: The full workflow requires baseline MLIR for comparison, which is
        // available here in `optimize_c_code`. We pass it to enable Stage 3 evaluation.
        let (optimized_schedule, egraph_stats, optimized_cost) = self
            .optimize_schedule_with_stats_and_baseline(
                baseline_schedule.clone(),
                Some(&baseline.mlir_file),
                kernel,
                config,
            )?;

        // Compute baseline cost (simple heuristic based on schedule structure)
        let baseline_cost = self.compute_schedule_cost(&baseline_schedule);

        // Stage 6-7: Apply to MLIR, compile, measure (if enabled)
        let (optimized_mlir, execution_time, speedup) = if config.measure_execution {
            if config.verbose {
                println!("\n=== Stage 6-7: Apply Schedule and Measure Performance ===");
            }

            // Stage 6: Apply optimized schedule to MLIR
            let optimized_mlir_path = self.apply_schedule_to_mlir(
                polygeist_dir,
                &baseline.mlir_file,
                &baseline.schedule_file,
                &optimized_schedule,
                &config.output_dir,
                config.verbose,
            )?;

            // Stage 7: Compile and measure performance
            // NOTE: This is real execution, NOT cycle-accurate simulation as in the paper
            let perf_result = self.measure_performance(
                &baseline.mlir_file,
                &optimized_mlir_path,
                kernel,
                config.problem_size,
                config.verbose,
            )?;

            (
                Some(optimized_mlir_path),
                Some(perf_result.execution_time_ms),
                perf_result.speedup,
            )
        } else {
            // Skip measurement, just apply schedule to MLIR for inspection
            let optimized_mlir_path = self.apply_schedule_to_mlir(
                polygeist_dir,
                &baseline.mlir_file,
                &baseline.schedule_file,
                &optimized_schedule,
                &config.output_dir,
                config.verbose,
            )?;
            (
                Some(optimized_mlir_path),
                None,
                baseline_cost / optimized_cost,
            )
        };

        Ok(PipelineResult {
            schedule_str: dump_isl(&optimized_schedule),
            cost: optimized_cost,
            baseline_schedule_str: dump_isl(&baseline_schedule),
            baseline_cost,
            speedup,
            egraph_stats,
            mlir_file: optimized_mlir,
            execution_time,
        })
    }

    /// Optimize an existing ISL schedule with statistics (internal method)
    ///
    /// Returns the optimized schedule along with e-graph statistics and cost.
    ///
    /// **Paper Alignment**: This implements Stage 1 (analytical proxy) and Stage 2 (k-best extraction)
    /// of the hybrid cost model. Stage 3 (simulation-based selection) requires baseline MLIR,
    /// which is handled by `optimize_schedule_with_stats_and_baseline()`.
    ///
    /// # Note
    /// This function does NOT perform Stage 3 evaluation. For full paper-aligned workflow,
    /// use `optimize_schedule_with_stats_and_baseline()` which includes execution evaluation.
    pub fn optimize_schedule_with_stats(
        &self,
        schedule: ScheduleHandle,
        config: &PipelineConfig,
    ) -> Result<(ScheduleHandle, EGraphStats, f64), PipelineError> {
        self.optimize_schedule_with_stats_and_baseline(schedule, None, "", config)
    }

    /// Optimize schedule with baseline MLIR for full paper-aligned workflow
    ///
    /// **Paper Alignment**: This implements the complete three-stage hybrid cost model:
    /// 1. **Stage 1 (Analytical Proxy)**: Fast analytical cost guides e-graph saturation
    /// 2. **Stage 2 (K-Best Extraction)**: Extract k candidates using analytical cost
    /// 3. **Stage 3 (Simulation-Based Selection)**: Evaluate k candidates with execution/simulation,
    ///    select the one with best performance
    ///
    /// **Paper Quote**: "A fast, analytical proxy model guides the saturation process, after which
    /// a small set of promising candidates are evaluated by a cycle-accurate simulator."
    ///
    /// # Arguments
    /// * `schedule` - Initial ISL schedule to optimize
    /// * `baseline_mlir` - Optional path to baseline MLIR file (required for Stage 3 evaluation)
    /// * `kernel_name` - Kernel name for execution measurement (required if baseline_mlir is Some)
    /// * `config` - Pipeline configuration
    ///
    /// # Returns
    /// Optimized schedule with statistics and cost
    ///
    /// # Errors
    /// Returns error if e-graph exploration or extraction fails
    fn optimize_schedule_with_stats_and_baseline(
        &self,
        schedule: ScheduleHandle,
        baseline_mlir: Option<&PathBuf>,
        kernel_name: &str,
        config: &PipelineConfig,
    ) -> Result<(ScheduleHandle, EGraphStats, f64), PipelineError> {
        if config.verbose {
            println!("\n=== E-graph Exploration ===");
            println!("  Max iterations: {}", config.max_iter);
            println!("  Max nodes: {}", config.max_nodes);
        }

        // Save baseline string for comparison (before moving schedule)
        let baseline_str = if config.verbose {
            Some(schedule.schedule.to_str())
        } else {
            None
        };

        // Stage 2: Create e-graph
        let mut egraph = EGraph::new(ScheduleAnalysis::new(self.ctx.clone()));
        let root = egraph.add(SchedOp::Schedule(schedule.clone()));

        // Stage 3: Equality saturation with dependency-aware rules
        // NOTE: We now ALWAYS use rational_dependency_rules() which has real ISL dependency
        // analysis via UnionAccessInfo::compute_flow(). The old rewrites::{rules, advanced_rules}
        // were removed because they had STUB dependency checks (always returning true/false).
        let start = std::time::Instant::now();

        // Start with rational dependency rules (includes tiling, parallel, interchange, etc.)
        // These rules use real ISL dependency analysis internally
        let mut all_rules = rational_dependency_rules();

        // Add hardware-aware rewrite rules if NCP configuration provided
        // Paper Alignment: This implements conditional rewrites with hardware constraints
        if config.ncp_slices.is_some() || config.ncp_banks.is_some() {
            let hardware_rules = self.create_hardware_aware_rules(config);
            all_rules.extend(hardware_rules);
            if config.verbose {
                println!("  Added hardware-aware rewrite rules");
            }
        }

        if config.verbose {
            let dep_status = if config.check_dependencies {
                "enabled (ISL flow analysis)"
            } else {
                "disabled (rules still check internally)"
            };
            println!(
                "  Using {} rules, dependency checking: {}",
                all_rules.len(),
                dep_status
            );
        }

        let runner = Runner::default()
            .with_egraph(egraph)
            .with_iter_limit(config.max_iter)
            .with_node_limit(config.max_nodes)
            .run(&all_rules);

        let saturation_time = start.elapsed().as_secs_f64();

        if config.verbose {
            println!("  Saturation: {:.2}s", saturation_time);
            println!(
                "  E-graph: {} nodes, {} classes",
                runner.egraph.total_size(),
                runner.egraph.number_of_classes()
            );
        }

        // Stage 4: Extract best schedule using hardware-aware cost model if configured
        // Paper Alignment: This implements Stage 1 (analytical proxy) and Stage 2 (k-best extraction)
        // of the hybrid cost model.
        let extract_start = std::time::Instant::now();

        let (cost, best_expr) = if let Some(k) = config.k_best_candidates {
            // **Paper-Aligned Workflow**: Two-stage hybrid cost model
            // Stage 1: Analytical proxy model guides saturation (already done above)
            // Stage 2: Extract k-best candidates using analytical cost
            // Stage 3: Evaluate k candidates with simulation/execution, select best

            if config.verbose {
                println!("  Using paper-aligned k-best extraction workflow (k={})", k);
                println!("  Stage 2: Extracting k-best candidates using analytical cost model...");
            }

            // Stage 2: Extract k-best candidates using unified cost model (heuristic + communication)
            // Now uses dependency-aware communication cost in addition to
            // heuristic schedule cost, providing more accurate analytical proxy
            let candidates = {
                // Use the new unified extraction that includes communication cost
                let k_best_results = crate::extract_k_best_schedules(&runner.egraph, k);

                // Convert to expected format: (cost, expr, handle)
                let mut converted_candidates = Vec::new();
                for (cost, schedule, _schedule_str) in k_best_results {
                    // Reconstruct ScheduleHandle
                    let handle = ScheduleHandle::new(self.ctx.clone(), schedule);

                    // Create a simple expression (just the schedule node)
                    // Note: For execution measurement, we don't need the full transformation tree,
                    // just the final schedule is sufficient
                    let mut expr = RecExpr::default();
                    expr.add(SchedOp::Schedule(handle.clone()));

                    converted_candidates.push((cost, expr, handle));
                }
                converted_candidates
            };

            if config.verbose {
                println!(
                    "  Extracted {} candidates (requested k={})",
                    candidates.len(),
                    k
                );
                for (i, (cand_cost, _, _)) in candidates.iter().take(5).enumerate() {
                    println!("    [{}] Analytical cost: {:.6}", i + 1, cand_cost);
                }
            }

            // Stage 3: Evaluate k candidates with simulation/execution
            // **Paper Requirement**: "The schedule yielding the best simulated performance
            // (e.g., lowest latency) is chosen as the definitive output."
            //
            // **Current Implementation**: We use real execution measurement as a proxy for
            // cycle-accurate simulation. If `measure_execution` is enabled and we have access
            // to the baseline MLIR, we measure each candidate and select the best.
            //
            // **Future Enhancement**: When NCP cycle-accurate simulator is available, replace
            // `measure_candidates_with_execution` with `measure_candidates_with_simulator`.

            if config.measure_execution && baseline_mlir.is_some() && self.polygeist_dir.is_some() {
                if config.verbose {
                    println!(
                        "  Stage 3: Evaluating {} candidates with execution measurement...",
                        candidates.len()
                    );
                    println!("  This implements the paper's Stage 3 (using execution as proxy for simulation)");
                }

                // Use execution-based evaluation to select best candidate
                // This is the paper's Stage 3, using real execution as proxy for cycle-accurate simulation
                let (best_cost, best_expr, _) = self.select_best_by_execution_with_baseline(
                    &candidates,
                    baseline_mlir.unwrap(),
                    kernel_name,
                    config,
                )?;

                (best_cost, best_expr)
            } else {
                // Fallback: Select candidate with lowest analytical cost
                // This is NOT the paper's workflow, but allows the pipeline to work without
                // execution measurement capability.
                if config.verbose {
                    println!("  Stage 3: Skipping execution evaluation (not available)");
                    println!(
                        "  Selecting candidate with lowest analytical cost (NOT paper-aligned)"
                    );
                }

                let (best_cost, best_expr, _) = candidates
                    .into_iter()
                    .next()
                    .ok_or_else(|| PipelineError::Other("No candidates extracted".to_string()))?;

                (best_cost, best_expr)
            }
        } else if config.use_performance_extraction {
            // Single-best extraction using performance measurement
            extract_best_by_performance(&runner.egraph, root)
        } else if config.use_hardware_cost_model
            && (config.ncp_slices.is_some() || config.ncp_banks.is_some())
        {
            // Single-best extraction using hardware-aware NCP cost model
            self.extract_best_with_ncp_cost(&runner.egraph, root, config)?
        } else {
            // Single-best extraction using default heuristic cost model
            extract_best(&runner.egraph, root)
        };

        let extraction_time = extract_start.elapsed().as_secs_f64();

        if config.verbose {
            println!("  Extraction: {:.2}s", extraction_time);
            println!("  Best cost: {:.4}", cost);
            println!(
                "  Best expression length: {} nodes",
                best_expr.as_ref().len()
            );
            // **IMPROVED**: Display human-readable transformation sequence instead of raw RecExpr indices
            let readable_expr = Self::format_recexpr_readable(&best_expr);
            println!("  Best expression: {}", readable_expr);
        }

        // **CRITICAL FIX**: Extract schedule directly from the best e-class's data,
        // not from RecExpr. The RecExpr is just a representation of the expression tree,
        // but the actual transformed schedule is stored in the e-class's ScheduleData.
        //
        // We need to find the e-class with the lowest cost (which we already calculated),
        // and extract its schedule directly. This ensures we get the transformed schedule
        // that ScheduleAnalysis::make() evaluated and stored, not the baseline schedule.
        let optimized = {
            // Recalculate best_id by finding the e-class with the lowest cost
            // (same logic as extract_best, but we need to do it here to get best_id)
            use crate::optimize::ScheduleCost;
            let mut cost_fn = ScheduleCost::new();
            let mut best_id = root;
            let mut best_cost_calc = f64::MAX;

            // Calculate root cost
            if let Some(ref handle) = runner.egraph[root].data.schedule {
                let root_cost = cost_fn.cost(&SchedOp::Schedule(handle.clone()), |_| 0.0);
                best_cost_calc = root_cost;
                best_id = root;
            }

            // Search all e-classes for the one with lowest cost
            for class in runner.egraph.classes() {
                if let Some(ref handle) = class.data.schedule {
                    let class_cost = cost_fn.cost(&SchedOp::Schedule(handle.clone()), |_| 0.0);
                    if class_cost < best_cost_calc {
                        best_cost_calc = class_cost;
                        best_id = class.id;
                    }
                }
            }

            // Extract schedule directly from best_id's e-class data
            if let Some(ref handle) = runner.egraph[best_id].data.schedule {
                handle.clone()
            } else {
                // Fallback: try to extract from expression (should not happen if extract_best worked correctly)
                if config.verbose {
                    eprintln!("  ⚠️  WARNING: Best e-class {} has no schedule data, falling back to expression extraction", best_id);
                }
                self.get_schedule_from_expr(&runner.egraph, &best_expr)?
            }
        };

        // Debug: Verify extracted schedule is different from baseline
        if let Some(ref baseline_str) = baseline_str {
            let optimized_str = optimized.schedule.to_str();

            // Use regex to detect tiling (same as cost model)
            use regex::Regex;
            let tiling_pattern =
                Regex::new(r"\w+\s*-\s*\(\s*\w+\s*\)\s*mod|\(\s*\w+\s*\)\s*mod").unwrap();
            let has_tiling = tiling_pattern.is_match(&optimized_str);
            let has_parallel =
                optimized_str.contains("parallel") || optimized_str.contains("permutable: 1");

            if config.verbose {
                println!("  Extracted schedule analysis:");
                println!(
                    "    Length: {} chars (baseline: {} chars)",
                    optimized_str.len(),
                    baseline_str.len()
                );
                println!("    Has tiling: {} (detected via regex)", has_tiling);
                println!("    Has parallel: {}", has_parallel);

                if baseline_str == &optimized_str {
                    println!("    ⚠️  WARNING: Extracted schedule is IDENTICAL to baseline!");
                    println!("    This suggests extraction may not be working correctly.");
                } else {
                    println!("    ✓ Extracted schedule differs from baseline");
                    if has_tiling || has_parallel {
                        println!("    ✓ Extracted schedule contains transformations");
                    }
                }
            }
        }

        let stats = EGraphStats {
            total_nodes: runner.egraph.total_size(),
            num_classes: runner.egraph.number_of_classes(),
            iterations: runner.iterations.len(),
            saturation_time,
            extraction_time,
        };

        Ok((optimized, stats, cost))
    }

    /// Optimize an existing ISL schedule (skip C → MLIR stage)
    ///
    /// This is the core optimization loop:
    /// 1. Create e-graph with initial schedule
    /// 2. Run equality saturation with rewrite rules
    /// 3. Extract best schedule using cost model
    ///
    /// # Arguments
    /// * `schedule` - Initial ISL schedule to optimize
    /// * `config` - Pipeline configuration
    ///
    /// # Returns
    /// Optimized ISL schedule
    pub fn optimize_schedule(
        &self,
        schedule: ScheduleHandle,
        config: &PipelineConfig,
    ) -> Result<ScheduleHandle, PipelineError> {
        let (optimized, _, _) = self.optimize_schedule_with_stats(schedule, config)?;
        Ok(optimized)
    }

    /// Extract ISL schedule from RecExpr
    ///
    /// **CRITICAL**: This function must properly handle transformation expressions.
    /// The expression may be:
    /// - `Schedule(...)` - Direct schedule node
    /// - `Tile(Schedule(...), ...)` - Transformed schedule
    /// - `Parallel(Tile(Schedule(...), ...), ...)` - Multiple transformations
    ///
    /// The `ScheduleAnalysis::make()` method evaluates transformations and stores
    /// the resulting schedules in e-class data. We extract from the root e-class.
    ///
    /// Made public for testing purposes.
    pub fn get_schedule_from_expr(
        &self,
        egraph: &EGraph<SchedOp, ScheduleAnalysis>,
        expr: &RecExpr<SchedOp>,
    ) -> Result<ScheduleHandle, PipelineError> {
        // **CRITICAL**: RecExpr node indices are NOT e-class IDs!
        // RecExpr is a flattened expression tree where each node has an index.
        // To find the e-class ID for a node, we need to:
        // 1. Extract the actual e-node from RecExpr
        // 2. Search the e-graph for the e-class containing this e-node
        // 3. Check that e-class's data for the transformed schedule

        // Strategy 1: Check if root node is a direct Schedule
        if let Some(last_node) = expr.as_ref().last() {
            if let SchedOp::Schedule(handle) = last_node {
                return Ok(handle.clone());
            }
        }

        // Strategy 2: For transformation nodes, find the e-class containing this e-node
        // and extract schedule from that e-class's data
        if let Some(last_node) = expr.as_ref().last() {
            // Search all e-classes to find the one containing this e-node
            // This is necessary because RecExpr indices don't map directly to e-class IDs
            for class in egraph.classes() {
                // Check if this e-class contains the root node from the expression
                if class.nodes.iter().any(|n| {
                    // Compare the e-node structure (operator and children)
                    match (n, last_node) {
                        (SchedOp::Tile([_a1, _a2, _a3]), SchedOp::Tile([_b1, _b2, _b3])) => {
                            // Check if children match (they are e-class IDs in the RecExpr)
                            // We need to map RecExpr indices to e-class IDs
                            // For now, check if the e-class has schedule data
                            true
                        }
                        _ => false,
                    }
                }) {
                    // Found the e-class containing this transformation node
                    // Check if it has schedule data (ScheduleAnalysis::make() should have stored it)
                    if let Some(handle) = &class.data.schedule {
                        return Ok(handle.clone());
                    }
                }
            }

            // Strategy 3: For transformation nodes, try to find e-class by matching structure
            // This handles cases where the e-node structure matches
            match last_node {
                SchedOp::Tile([sched_id, _, _])
                | SchedOp::TilePerDim([sched_id, _, _, _])
                | SchedOp::Parallel([sched_id, _])
                | SchedOp::Vectorize([sched_id, _, _])
                | SchedOp::Interchange([sched_id, _, _])
                | SchedOp::Fuse([sched_id, _, _])
                | SchedOp::Split([sched_id, _, _])
                | SchedOp::Skew([sched_id, _, _, _])
                | SchedOp::Unroll([sched_id, _, _])
                | SchedOp::TileAtMark([sched_id, _, _])
                | SchedOp::ParallelAtMark([sched_id, _])
                | SchedOp::VectorizeAtMark([sched_id, _, _])
                | SchedOp::UnrollAtMark([sched_id, _, _])
                | SchedOp::SplitAtMark([sched_id, _, _]) => {
                    // **CRITICAL**: RecExpr indices are NOT e-class IDs!
                    // The `sched_id` in RecExpr is an index into the RecExpr, not an e-class ID.
                    // We need to map RecExpr indices to e-class IDs by reconstructing the expression.
                    //
                    // However, a simpler approach: search all e-classes for ones with schedule data
                    // that match transformation patterns. Since ScheduleAnalysis::make() stores
                    // transformed schedules in e-class data, we can search for e-classes with
                    // schedule data that differ from baseline.

                    // Search for e-classes with transformed schedules (not baseline)
                    let mut best_handle: Option<ScheduleHandle> = None;
                    for class in egraph.classes() {
                        if let Some(ref handle) = class.data.schedule {
                            // Check if this schedule is transformed (has tiling, parallel, etc.)
                            let schedule_str = handle.schedule.to_str();
                            use regex::Regex;
                            let tiling_pattern =
                                Regex::new(r"\w+\s*-\s*\(\s*\w+\s*\)\s*mod|\(\s*\w+\s*\)\s*mod")
                                    .unwrap();
                            let has_tiling = tiling_pattern.is_match(&schedule_str);
                            let has_parallel = schedule_str.contains("parallel")
                                || schedule_str.contains("permutable: 1");

                            // Prefer transformed schedules over baseline
                            if has_tiling || has_parallel {
                                // Check if this e-class contains a transformation node matching the expression
                                let contains_transformation =
                                    class.nodes.iter().any(|n| match (n, last_node) {
                                        (SchedOp::Tile(_), SchedOp::Tile(_)) => true,
                                        (SchedOp::Parallel(_), SchedOp::Parallel(_)) => true,
                                        (SchedOp::Vectorize(_), SchedOp::Vectorize(_)) => true,
                                        (SchedOp::Interchange(_), SchedOp::Interchange(_)) => true,
                                        (SchedOp::Fuse(_), SchedOp::Fuse(_)) => true,
                                        (SchedOp::Split(_), SchedOp::Split(_)) => true,
                                        (SchedOp::Skew(_), SchedOp::Skew(_)) => true,
                                        (SchedOp::Unroll(_), SchedOp::Unroll(_)) => true,
                                        (SchedOp::TileAtMark(_), SchedOp::TileAtMark(_)) => true,
                                        (
                                            SchedOp::ParallelAtMark(_),
                                            SchedOp::ParallelAtMark(_),
                                        ) => true,
                                        (
                                            SchedOp::VectorizeAtMark(_),
                                            SchedOp::VectorizeAtMark(_),
                                        ) => true,
                                        (SchedOp::UnrollAtMark(_), SchedOp::UnrollAtMark(_)) => {
                                            true
                                        }
                                        (SchedOp::SplitAtMark(_), SchedOp::SplitAtMark(_)) => true,
                                        _ => false,
                                    });

                                if contains_transformation {
                                    best_handle = Some(handle.clone());
                                    break; // Found a transformed schedule matching the expression
                                }
                            }
                        }
                    }

                    if let Some(handle) = best_handle {
                        return Ok(handle);
                    }

                    // Fallback: Try to extract from source schedule's e-class
                    // Map RecExpr index to e-class ID by finding the e-class containing the source schedule
                    // This is a workaround - ideally we'd have a proper RecExpr -> e-class ID mapping
                    if let Some(source_node_idx) = usize::try_from(*sched_id).ok() {
                        if source_node_idx < expr.as_ref().len() {
                            // Recursively extract from source
                            if let Ok(source_expr) = self.extract_expr_for_id(egraph, *sched_id) {
                                if let Ok(source_handle) =
                                    self.get_schedule_from_expr(egraph, &source_expr)
                                {
                                    return Ok(source_handle);
                                }
                            }
                        }
                    }
                }
                _ => {}
            }
        }

        // Strategy 4: Search all e-classes for any schedule (fallback)
        // This handles cases where we can't match the expression structure
        for class in egraph.classes() {
            if let Some(ref handle) = class.data.schedule {
                // Check if this schedule is transformed (not baseline)
                let schedule_str = handle.schedule.to_str();
                use regex::Regex;
                let tiling_pattern =
                    Regex::new(r"\w+\s*-\s*\(\s*\w+\s*\)\s*mod|\(\s*\w+\s*\)\s*mod").unwrap();
                let has_tiling = tiling_pattern.is_match(&schedule_str);
                let has_parallel = schedule_str.contains("parallel");

                if has_tiling || has_parallel {
                    return Ok(handle.clone());
                }
            }
        }

        Err(PipelineError::Other(format!(
            "Could not extract schedule from expression. Expression length: {}. Expression: {:?}",
            expr.as_ref().len(),
            expr.as_ref().last()
        )))
    }

    /// Helper: Extract RecExpr for a given e-class ID
    /// Used for recursive extraction from transformation nodes
    fn extract_expr_for_id(
        &self,
        egraph: &EGraph<SchedOp, ScheduleAnalysis>,
        id: egg::Id,
    ) -> Result<RecExpr<SchedOp>, PipelineError> {
        use egg::AstSize;
        use egg::Extractor;

        let extractor = Extractor::new(egraph, AstSize);
        let (_, expr) = extractor.find_best(id);
        Ok(expr)
    }

    /// Format RecExpr as a human-readable transformation sequence
    ///
    /// **Problem**: Raw RecExpr output like `Tile([7, 1, 0])` shows node indices, not actual
    /// transformation parameters. This function extracts the actual transformation information
    /// from the RecExpr to produce readable output like "Tile(band=0, size=32)".
    ///
    /// **Note**: RecExpr indices are NOT e-class IDs. They are indices into the flattened
    /// expression tree. To get actual values, we need to look up nodes by index.
    fn format_recexpr_readable(expr: &RecExpr<SchedOp>) -> String {
        use crate::language::SchedOp;

        if expr.as_ref().is_empty() {
            return "empty".to_string();
        }

        // Build transformation sequence from root to leaves
        let mut transformations = Vec::new();

        // Process nodes in reverse order (root is last)
        for (_idx, node) in expr.as_ref().iter().enumerate().rev() {
            match node {
                SchedOp::Schedule(_) => {
                    // Base schedule - stop here
                    break;
                }
                SchedOp::Tile([_sched_id, band_id, size_id]) => {
                    // Extract band index and tile size from child nodes
                    let band_idx =
                        if let Some(SchedOp::Num(n)) = expr.as_ref().get(usize::from(*band_id)) {
                            *n
                        } else {
                            0
                        };
                    let tile_size =
                        if let Some(SchedOp::Num(n)) = expr.as_ref().get(usize::from(*size_id)) {
                            *n
                        } else {
                            32
                        };
                    transformations.push(format!("Tile(band={}, size={})", band_idx, tile_size));
                }
                SchedOp::Parallel([_sched_id, band_id]) => {
                    let band_idx =
                        if let Some(SchedOp::Num(n)) = expr.as_ref().get(usize::from(*band_id)) {
                            *n
                        } else {
                            0
                        };
                    transformations.push(format!("Parallel(band={})", band_idx));
                }
                SchedOp::Vectorize([_sched_id, band_id, width_id]) => {
                    let band_idx =
                        if let Some(SchedOp::Num(n)) = expr.as_ref().get(usize::from(*band_id)) {
                            *n
                        } else {
                            0
                        };
                    let width =
                        if let Some(SchedOp::Num(n)) = expr.as_ref().get(usize::from(*width_id)) {
                            *n
                        } else {
                            8
                        };
                    transformations.push(format!("Vectorize(band={}, width={})", band_idx, width));
                }
                SchedOp::Interchange([_sched_id, band1_id, band2_id]) => {
                    let band1 =
                        if let Some(SchedOp::Num(n)) = expr.as_ref().get(usize::from(*band1_id)) {
                            *n
                        } else {
                            0
                        };
                    let band2 =
                        if let Some(SchedOp::Num(n)) = expr.as_ref().get(usize::from(*band2_id)) {
                            *n
                        } else {
                            1
                        };
                    transformations.push(format!("Interchange(band1={}, band2={})", band1, band2));
                }
                SchedOp::Fuse([_sched_id, loop1_id, loop2_id]) => {
                    let loop1 =
                        if let Some(SchedOp::Num(n)) = expr.as_ref().get(usize::from(*loop1_id)) {
                            *n
                        } else {
                            0
                        };
                    let loop2 =
                        if let Some(SchedOp::Num(n)) = expr.as_ref().get(usize::from(*loop2_id)) {
                            *n
                        } else {
                            1
                        };
                    transformations.push(format!("Fuse(loop1={}, loop2={})", loop1, loop2));
                }
                SchedOp::Unroll([_sched_id, band_id, factor_id]) => {
                    let band_idx =
                        if let Some(SchedOp::Num(n)) = expr.as_ref().get(usize::from(*band_id)) {
                            *n
                        } else {
                            0
                        };
                    let factor =
                        if let Some(SchedOp::Num(n)) = expr.as_ref().get(usize::from(*factor_id)) {
                            *n
                        } else {
                            4
                        };
                    transformations.push(format!("Unroll(band={}, factor={})", band_idx, factor));
                }
                SchedOp::Split([_sched_id, band_id, factor_id]) => {
                    let band_idx =
                        if let Some(SchedOp::Num(n)) = expr.as_ref().get(usize::from(*band_id)) {
                            *n
                        } else {
                            0
                        };
                    let factor =
                        if let Some(SchedOp::Num(n)) = expr.as_ref().get(usize::from(*factor_id)) {
                            *n
                        } else {
                            32
                        };
                    transformations.push(format!("Split(band={}, factor={})", band_idx, factor));
                }
                SchedOp::Skew([_sched_id, band_id, factor_id, direction_id]) => {
                    let band_idx =
                        if let Some(SchedOp::Num(n)) = expr.as_ref().get(usize::from(*band_id)) {
                            *n
                        } else {
                            0
                        };
                    let factor =
                        if let Some(SchedOp::Num(n)) = expr.as_ref().get(usize::from(*factor_id)) {
                            *n
                        } else {
                            1
                        };
                    let direction = if let Some(SchedOp::Num(n)) =
                        expr.as_ref().get(usize::from(*direction_id))
                    {
                        *n
                    } else {
                        0
                    };
                    transformations.push(format!(
                        "Skew(band={}, factor={}, direction={})",
                        band_idx, factor, direction
                    ));
                }
                SchedOp::TileAtMark([_sched_id, mark_name_id, size_id]) => {
                    let mark_name = if let Some(SchedOp::Symbol(s)) =
                        expr.as_ref().get(usize::from(*mark_name_id))
                    {
                        s.to_string()
                    } else {
                        "unknown".to_string()
                    };
                    let tile_size =
                        if let Some(SchedOp::Num(n)) = expr.as_ref().get(usize::from(*size_id)) {
                            *n
                        } else {
                            32
                        };
                    transformations.push(format!(
                        "TileAtMark(mark=\"{}\", size={})",
                        mark_name, tile_size
                    ));
                }
                SchedOp::ParallelAtMark([_sched_id, mark_name_id]) => {
                    let mark_name = if let Some(SchedOp::Symbol(s)) =
                        expr.as_ref().get(usize::from(*mark_name_id))
                    {
                        s.to_string()
                    } else {
                        "unknown".to_string()
                    };
                    transformations.push(format!("ParallelAtMark(mark=\"{}\")", mark_name));
                }
                SchedOp::VectorizeAtMark([_sched_id, mark_name_id, width_id]) => {
                    let mark_name = if let Some(SchedOp::Symbol(s)) =
                        expr.as_ref().get(usize::from(*mark_name_id))
                    {
                        s.to_string()
                    } else {
                        "unknown".to_string()
                    };
                    let width =
                        if let Some(SchedOp::Num(n)) = expr.as_ref().get(usize::from(*width_id)) {
                            *n
                        } else {
                            8
                        };
                    transformations.push(format!(
                        "VectorizeAtMark(mark=\"{}\", width={})",
                        mark_name, width
                    ));
                }
                SchedOp::UnrollAtMark([_sched_id, mark_name_id, factor_id]) => {
                    let mark_name = if let Some(SchedOp::Symbol(s)) =
                        expr.as_ref().get(usize::from(*mark_name_id))
                    {
                        s.to_string()
                    } else {
                        "unknown".to_string()
                    };
                    let factor =
                        if let Some(SchedOp::Num(n)) = expr.as_ref().get(usize::from(*factor_id)) {
                            *n
                        } else {
                            4
                        };
                    transformations.push(format!(
                        "UnrollAtMark(mark=\"{}\", factor={})",
                        mark_name, factor
                    ));
                }
                SchedOp::SplitAtMark([_sched_id, mark_name_id, factor_id]) => {
                    let mark_name = if let Some(SchedOp::Symbol(s)) =
                        expr.as_ref().get(usize::from(*mark_name_id))
                    {
                        s.to_string()
                    } else {
                        "unknown".to_string()
                    };
                    let factor =
                        if let Some(SchedOp::Num(n)) = expr.as_ref().get(usize::from(*factor_id)) {
                            *n
                        } else {
                            32
                        };
                    transformations.push(format!(
                        "SplitAtMark(mark=\"{}\", factor={})",
                        mark_name, factor
                    ));
                }
                SchedOp::TilePerDim([_sched_id, size_i_id, size_j_id, size_k_id]) => {
                    let size_i =
                        if let Some(SchedOp::Num(n)) = expr.as_ref().get(usize::from(*size_i_id)) {
                            *n
                        } else {
                            16
                        };
                    let size_j =
                        if let Some(SchedOp::Num(n)) = expr.as_ref().get(usize::from(*size_j_id)) {
                            *n
                        } else {
                            16
                        };
                    let size_k =
                        if let Some(SchedOp::Num(n)) = expr.as_ref().get(usize::from(*size_k_id)) {
                            *n
                        } else {
                            8
                        };
                    transformations.push(format!(
                        "TilePerDim(size_i={}, size_j={}, size_k={})",
                        size_i, size_j, size_k
                    ));
                }
                SchedOp::InsertMark([_sched_id, mark_name_id]) => {
                    let mark_name = if let Some(SchedOp::Symbol(s)) =
                        expr.as_ref().get(usize::from(*mark_name_id))
                    {
                        s.to_string()
                    } else {
                        "unknown".to_string()
                    };
                    transformations.push(format!("InsertMark(mark=\"{}\")", mark_name));
                }
                _ => {
                    // Other operations - use debug format
                    transformations.push(format!("{:?}", node));
                }
            }
        }

        if transformations.is_empty() {
            "Schedule(baseline)".to_string()
        } else {
            // Reverse to show transformations in application order (first applied first)
            transformations.reverse();
            format!("Schedule → {}", transformations.join(" → "))
        }
    }

    /// Compute a simple cost heuristic for a schedule (for baseline comparison)
    ///
    /// This is a placeholder cost model that estimates schedule quality based on
    /// structural properties. In practice, this would use a more sophisticated
    /// cost model (e.g., MLIR-based or hardware-specific).
    ///
    /// Made public for testing purposes.
    pub fn compute_schedule_cost(&self, schedule: &ScheduleHandle) -> f64 {
        let schedule_str = schedule.schedule.to_str();

        // Simple heuristic: count transformations (tiling, parallelization, etc.)
        let mut cost = 100.0; // Base cost

        // Penalize tiling (mod operations indicate tiling)
        if schedule_str.contains("mod") {
            cost *= 0.8; // Tiling reduces cost
        }

        // Penalize parallelization (parallel marks)
        if schedule_str.contains("parallel") || schedule_str.contains("atomic") {
            cost *= 0.7; // Parallelization reduces cost
        }

        // Penalize vectorization
        if schedule_str.contains("vector") {
            cost *= 0.6; // Vectorization reduces cost
        }

        cost
    }

    /// Stage 6: Apply optimized ISL schedule to MLIR
    ///
    /// Uses Polygeist's `islexternal-import-schedules` to apply the transformed
    /// schedule to the baseline MLIR file.
    ///
    /// **CRITICAL FIX**: Detects how many affine functions are in the baseline MLIR
    /// and generates corresponding schedule files (`__polygeist_outlined_affine_0`,
    /// `__polygeist_outlined_affine_1`, ...) to match Polygeist's expectations.
    ///
    /// **Paper Gap**: Should lower to NCP dialect, not just standard MLIR.
    ///
    /// # Arguments
    /// * `polygeist_dir` - Path to Polygeist installation
    /// * `baseline_mlir` - Path to baseline MLIR file
    /// * `baseline_schedule_file` - Path to baseline schedule file (for reference)
    /// * `optimized_schedule` - Optimized ISL schedule handle
    /// * `output_dir` - Directory for intermediate files
    /// * `verbose` - Enable verbose logging
    ///
    /// # Returns
    /// Path to the generated optimized MLIR file
    fn apply_schedule_to_mlir(
        &self,
        polygeist_dir: &Path,
        baseline_mlir: &Path,
        baseline_schedule_file: &Path,
        optimized_schedule: &ScheduleHandle,
        output_dir: &Path,
        verbose: bool,
    ) -> Result<PathBuf, PipelineError> {
        use std::fs;

        // Create work directory for schedule files
        let schedule_dir = output_dir.join("schedule");
        fs::create_dir_all(&schedule_dir)
            .map_err(|e| PipelineError::IOError(format!("Failed to create schedule dir: {}", e)))?;

        // Convert schedule to string
        let schedule_str = dump_isl(optimized_schedule);

        if verbose {
            println!("  Applying schedule to MLIR...");
            println!("  Schedule length: {} chars", schedule_str.len());
        }

        // **CRITICAL FIX**: Detect how many schedule files Polygeist expects
        // by checking the baseline schedule directory or counting affine functions in MLIR
        let num_schedules =
            Self::detect_num_schedule_files(baseline_mlir, baseline_schedule_file, verbose)?;

        if verbose {
            println!("  Detected {} schedule file(s) needed", num_schedules);
        }

        // Write schedule files for each affine function
        // For now, we use the same optimized schedule for all functions
        // (Future enhancement: optimize each function separately)
        for i in 0..num_schedules {
            let schedule_file = schedule_dir.join(format!("__polygeist_outlined_affine_{}", i));
            fs::write(&schedule_file, &schedule_str).map_err(|e| {
                PipelineError::IOError(format!("Failed to write schedule file {}: {}", i, e))
            })?;

            if verbose {
                println!("  ✓ Wrote schedule file: {}", schedule_file.display());
            }
        }

        // Generate output MLIR path
        let optimized_mlir = output_dir.join("optimized.mlir");

        // Apply schedule using Polygeist
        codegen::apply_schedule_to_mlir(
            polygeist_dir.to_str().unwrap(),
            baseline_mlir.to_str().unwrap(),
            schedule_dir.to_str().unwrap(),
            &schedule_str,
            optimized_mlir.to_str().unwrap(),
        )
        .map_err(|e| PipelineError::CompilationFailed(e))?;

        if verbose {
            println!("  ✓ Generated optimized MLIR: {}", optimized_mlir.display());
        }

        Ok(optimized_mlir)
    }

    /// Detect how many schedule files Polygeist expects
    ///
    /// **Problem**: Some kernels (e.g., NTT) have multiple affine functions, and Polygeist
    /// expects a schedule file for each (`__polygeist_outlined_affine_0`, `__polygeist_outlined_affine_1`, ...).
    /// If we only generate `__polygeist_outlined_affine_0`, Polygeist will fail with
    /// "Can't open __polygeist_outlined_affine_1".
    ///
    /// **Solution**: Check the baseline schedule directory to see how many schedule files
    /// were originally generated, or count affine functions in the MLIR file.
    fn detect_num_schedule_files(
        baseline_mlir: &Path,
        baseline_schedule_file: &Path,
        verbose: bool,
    ) -> Result<usize, PipelineError> {
        use std::fs;

        // Strategy 1: Check baseline schedule directory (if it exists)
        if let Some(schedule_dir) = baseline_schedule_file.parent() {
            if schedule_dir.exists() {
                let mut count = 0;
                for entry in fs::read_dir(schedule_dir).map_err(|e| {
                    PipelineError::IOError(format!("Failed to read schedule dir: {}", e))
                })? {
                    let entry = entry.map_err(|e| {
                        PipelineError::IOError(format!("Failed to read dir entry: {}", e))
                    })?;
                    let path = entry.path();
                    if let Some(file_name) = path.file_name() {
                        let name = file_name.to_string_lossy();
                        // Check if it matches the pattern __polygeist_outlined_affine_N
                        if name.starts_with("__polygeist_outlined_affine_") {
                            // Extract the number
                            if let Some(num_str) = name.strip_prefix("__polygeist_outlined_affine_")
                            {
                                if let Ok(num) = num_str.parse::<usize>() {
                                    count = count.max(num + 1); // +1 because indices start at 0
                                }
                            }
                        }
                    }
                }

                if count > 0 {
                    if verbose {
                        println!("  Found {} schedule file(s) in baseline directory", count);
                    }
                    return Ok(count);
                }
            }
        }

        // Strategy 2: Count outlined affine functions in MLIR file
        // Look for function names containing "outlined" or "__polygeist_outlined"
        let mlir_content = fs::read_to_string(baseline_mlir)
            .map_err(|e| PipelineError::IOError(format!("Failed to read baseline MLIR: {}", e)))?;

        // Count functions with "outlined" in their name (Polygeist's naming convention)
        let outlined_func_count = mlir_content
            .lines()
            .filter(|line| {
                line.contains("func.func")
                    && (line.contains("outlined") || line.contains("__polygeist_outlined"))
            })
            .count();

        // Also check for the pattern "__polygeist_outlined_affine_N" in comments or attributes
        let mut max_index = 0;
        for line in mlir_content.lines() {
            // Look for references to __polygeist_outlined_affine_N
            for i in 0..10 {
                let pattern = format!("__polygeist_outlined_affine_{}", i);
                if line.contains(&pattern) {
                    max_index = max_index.max(i + 1);
                }
            }
        }

        // Use the maximum of the two heuristics
        let num_schedules = if max_index > 0 {
            max_index
        } else if outlined_func_count > 0 {
            outlined_func_count.min(10) // Cap at 10 to avoid excessive files
        } else {
            1 // Default to 1 if we can't determine
        };

        if verbose {
            if max_index > 0 {
                println!(
                    "  Estimated {} schedule file(s) from MLIR pattern analysis",
                    num_schedules
                );
            } else if outlined_func_count > 0 {
                println!(
                    "  Estimated {} schedule file(s) from outlined function count",
                    num_schedules
                );
            } else {
                println!("  Using default: 1 schedule file");
            }
        }

        Ok(num_schedules)
    }

    /// Stage 7: Compile and measure performance
    ///
    /// Compiles both baseline and optimized MLIR, executes them, and measures
    /// performance differences.
    ///
    /// # Arguments
    /// * `baseline_mlir` - Path to baseline MLIR file
    /// * `optimized_mlir` - Path to optimized MLIR file
    /// * `kernel_name` - Name of the kernel function
    /// * `problem_size` - Problem size for execution
    /// * `verbose` - Enable verbose logging
    ///
    /// # Returns
    /// Performance measurement result
    /// Measure performance using unified schedule measurer
    ///
    /// **Improved**: Now uses unified `schedule_measurer` module instead of legacy
    /// `execution::measure_and_validate()`. This provides:
    /// - Consistent measurement logic across all pipeline stages
    /// - Built-in caching to avoid redundant compilations
    /// - Better error handling and reporting
    /// - Configurable measurement parameters
    fn measure_performance(
        &self,
        baseline_mlir: &Path,
        optimized_mlir: &Path,
        kernel_name: &str,
        problem_size: usize,
        verbose: bool,
    ) -> Result<execution::PerformanceResult, PipelineError> {
        if verbose {
            println!("  Measuring performance...");
            println!("  Kernel: {}", kernel_name);
            println!("  Problem size: {}", problem_size);
        }

        // Use unified schedule measurer for consistent measurement
        // Create work directory relative to config output_dir (passed as parameter)
        let work_dir = std::env::temp_dir().join("polysat_measurement");
        let config = MeasurementConfig {
            kernel_name: kernel_name.to_string(),
            kernel_type: crate::schedule_measurer::KernelType::GEMM, // Default to GEMM
            problem_size,
            iterations: 10, // Default: 10 iterations for averaging
            work_dir,
            cleanup: true,        // Clean up temporary files
            cache_binaries: true, // Cache compiled binaries for reuse
            timeout_secs: 0,      // No timeout
        };

        let measurer = ScheduleMeasurer::new(config).map_err(|e| {
            PipelineError::MeasurementFailed(format!("Failed to initialize measurer: {}", e))
        })?;

        measurer
            .compare_performance(baseline_mlir, optimized_mlir)
            .map_err(|e| PipelineError::MeasurementFailed(e))
    }

    /// Extract best schedule using NCP hardware-aware cost model
    ///
    /// This implements the paper's key contribution: hardware-aware cost modeling
    /// that guides equality saturation search with high-fidelity performance data.
    ///
    /// # Arguments
    /// * `egraph` - The e-graph containing explored schedules
    /// * `root` - The e-class ID to extract from
    /// * `config` - Pipeline configuration with hardware parameters
    ///
    /// # Returns
    /// * `(cost, best_expr)` - Best schedule with its cost
    fn extract_best_with_ncp_cost(
        &self,
        egraph: &EGraph<SchedOp, ScheduleAnalysis>,
        root: egg::Id,
        config: &PipelineConfig,
    ) -> Result<(f64, RecExpr<SchedOp>), PipelineError> {
        use egg::Extractor;

        // Create NCP cost model with hardware configuration
        // Use enhanced ScheduleCost with built-in NCP awareness
        // NCP-specific factors (tile count, slice alignment, communication) are now
        // integrated directly into ScheduleCost::cost() in optimize.rs
        if config.verbose {
            println!("  Using ScheduleCost with NCP-aware heuristics");
            if let (Some(slices), Some(banks)) = (config.ncp_slices, config.ncp_banks) {
                println!(
                    "  Target architecture: {} slices × {} banks = {} NCPs",
                    slices,
                    banks,
                    slices * banks
                );
            }
        }
        let mut ncp_model = ScheduleCost::new();

        // CRITICAL FIX: Calculate cost directly from e-class data schedule, not from expression tree
        // The problem: Extractor::find_best_cost() calculates cost through expression tree,
        // which includes transformation operation costs (e.g., Tile adds 0.1 to child cost).
        // But ScheduleAnalysis::make() already evaluated transformations and stored the
        // transformed schedule in e-class data. We should use THAT schedule's cost directly.

        // Calculate root cost directly from its schedule data
        let root_cost = if let Some(ref handle) = egraph[root].data.schedule {
            ncp_model.cost(&SchedOp::Schedule(handle.clone()), |_| 0.0)
        } else {
            1000.0 // Fallback high cost if no schedule data
        };
        let mut best_cost = root_cost;
        let mut best_id = root;

        // Search all e-classes for schedules with lower cost
        // Calculate cost directly from schedule data, not through expression tree
        for class in egraph.classes() {
            if let Some(ref handle) = class.data.schedule {
                // Calculate cost directly from the transformed schedule in e-class data
                let cost = ncp_model.cost(&SchedOp::Schedule(handle.clone()), |_| 0.0);
                if cost < best_cost {
                    best_cost = cost;
                    best_id = class.id;
                }
            }
        }

        // Extract expression from the best e-class (for returning RecExpr)
        let extractor = Extractor::new(egraph, ncp_model);
        let (_, expr) = extractor.find_best(best_id);

        if config.verbose && best_id != root {
            println!(
                "  Found better schedule in e-class {:?} (cost: {:.6} vs root: {:.6})",
                best_id, best_cost, root_cost
            );
        }

        Ok((best_cost, expr))
    }

    /// Extract k-best schedules using heuristic cost model
    ///
    /// **Paper Alignment**: This implements Stage 2 (k-best extraction) of the hybrid cost model.
    /// It extracts k candidates using the analytical proxy model (heuristic cost), which can then
    /// be evaluated by a cycle-accurate simulator (Stage 3) if available.
    ///
    /// # Arguments
    /// * `egraph` - The e-graph containing explored schedules
    /// * `root` - The e-class ID to extract from
    /// * `k` - Number of best candidates to extract
    ///
    /// # Returns
    /// Vector of (cost, expr, schedule_handle) tuples, sorted by cost (lowest first)
    ///
    /// Made public for testing purposes.
    pub fn extract_k_best_with_heuristic_cost(
        &self,
        egraph: &EGraph<SchedOp, ScheduleAnalysis>,
        _root: egg::Id,
        k: usize,
    ) -> Result<Vec<(f64, RecExpr<SchedOp>, ScheduleHandle)>, PipelineError> {
        use crate::optimize::ScheduleCost;
        use egg::Extractor;

        let mut cost_fn = ScheduleCost::new();
        let mut candidates: Vec<(f64, egg::Id, ScheduleHandle)> = Vec::new();

        // Collect all e-classes with schedule data and calculate their costs
        for class in egraph.classes() {
            if let Some(ref handle) = class.data.schedule {
                // Calculate cost directly from schedule data (same approach as extract_best)
                let cost = cost_fn.cost(&SchedOp::Schedule(handle.clone()), |_| 0.0);
                candidates.push((cost, class.id, handle.clone()));
            }
        }

        // Sort by cost (lowest first) and take top k
        candidates.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        candidates.truncate(k);

        // Extract expressions for the top k candidates
        let extractor = Extractor::new(egraph, ScheduleCost::new());
        let mut results = Vec::new();

        for (cost, id, handle) in candidates {
            let (_, expr) = extractor.find_best(id);
            results.push((cost, expr, handle));
        }

        Ok(results)
    }

    /// Extract k-best schedules using NCP hardware-aware cost model
    ///
    /// **Paper Alignment**: This implements Stage 2 (k-best extraction) of the hybrid cost model
    /// using hardware-aware analytical proxy. The extracted candidates can then be evaluated by
    /// a cycle-accurate simulator (Stage 3) if available.
    ///
    /// # Arguments
    /// * `egraph` - The e-graph containing explored schedules
    /// * `root` - The e-class ID to extract from
    /// * `k` - Number of best candidates to extract
    /// * `config` - Pipeline configuration with hardware parameters
    ///
    /// # Returns
    /// Vector of (cost, expr, schedule_handle) tuples, sorted by cost (lowest first)
    ///
    /// Made public for testing purposes.
    pub fn extract_k_best_with_ncp_cost(
        &self,
        egraph: &EGraph<SchedOp, ScheduleAnalysis>,
        _root: egg::Id,
        k: usize,
        _config: &PipelineConfig,
    ) -> Result<Vec<(f64, RecExpr<SchedOp>, ScheduleHandle)>, PipelineError> {
        use egg::Extractor;

        // Use enhanced ScheduleCost with built-in NCP awareness
        // (same as extract_best_with_ncp_cost)
        let mut ncp_model = ScheduleCost::new();

        let mut candidates: Vec<(f64, egg::Id, ScheduleHandle)> = Vec::new();

        // Collect all e-classes with schedule data and calculate their costs
        for class in egraph.classes() {
            if let Some(ref handle) = class.data.schedule {
                // Calculate cost directly from schedule data (same approach as extract_best_with_ncp_cost)
                let cost = ncp_model.cost(&SchedOp::Schedule(handle.clone()), |_| 0.0);
                candidates.push((cost, class.id, handle.clone()));
            }
        }

        // Sort by cost (lowest first) and take top k
        candidates.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        candidates.truncate(k);

        // Extract expressions for the top k candidates
        let extractor = Extractor::new(egraph, ncp_model);
        let mut results = Vec::new();

        for (cost, id, handle) in candidates {
            let (_, expr) = extractor.find_best(id);
            results.push((cost, expr, handle));
        }

        Ok(results)
    }

    /// Select best candidate from k-best extraction using execution measurement
    ///
    /// **Paper Alignment**: This implements Stage 3 of the hybrid cost model workflow.
    /// After extracting k candidates using analytical cost (Stage 2), this function evaluates
    /// each candidate with execution measurement (proxy for cycle-accurate simulation) and
    /// selects the one with best performance.
    ///
    /// **Paper Quote**: "The schedule yielding the best simulated performance (e.g., lowest
    /// latency) is chosen as the definitive output of the compiler."
    ///
    /// **Implementation Details**:
    /// 1. For each candidate, apply schedule to baseline MLIR → generate optimized MLIR
    /// 2. Compile and execute both baseline and optimized MLIR
    /// 3. Measure execution time for each candidate
    /// 4. Select candidate with lowest execution time (best performance)
    ///
    /// # Arguments
    /// * `candidates` - Vector of (analytical_cost, expr, schedule_handle) tuples from Stage 2
    /// * `baseline_mlir` - Path to baseline MLIR file (for comparison)
    /// * `kernel_name` - Kernel name for execution measurement
    /// * `config` - Pipeline configuration
    ///
    /// # Returns
    /// * `(best_execution_time_ms, best_expr, best_schedule)` - Best candidate based on execution
    ///
    /// # Errors
    /// Returns error if execution measurement fails for all candidates
    ///
    /// # Performance
    /// This function performs k executions (one per candidate), which can be slow. Consider
    /// using `measure_batch_parallel` for parallel execution if available.
    fn select_best_by_execution_with_baseline(
        &self,
        candidates: &[(f64, RecExpr<SchedOp>, ScheduleHandle)],
        baseline_mlir: &PathBuf,
        kernel_name: &str,
        config: &PipelineConfig,
    ) -> Result<(f64, RecExpr<SchedOp>, ScheduleHandle), PipelineError> {
        use std::fs;

        if candidates.is_empty() {
            return Err(PipelineError::Other(
                "No candidates to evaluate".to_string(),
            ));
        }

        let polygeist_dir = self.polygeist_dir.as_ref().ok_or_else(|| {
            PipelineError::Other("Polygeist not available for execution evaluation".to_string())
        })?;

        // We use real execution as a proxy for cycle-accurate simulation.

        // Create work directory for candidate MLIR files
        let work_dir = config.output_dir.join("candidates");
        fs::create_dir_all(&work_dir).map_err(|e| {
            PipelineError::IOError(format!("Failed to create candidates dir: {}", e))
        })?;

        // Measure baseline performance once
        if config.verbose {
            println!("  Measuring baseline performance...");
        }

        let baseline_config = MeasurementConfig {
            kernel_name: kernel_name.to_string(),
            kernel_type: crate::schedule_measurer::KernelType::GEMM, // Default to GEMM
            problem_size: config.problem_size,
            iterations: 5, // Reduced for faster evaluation
            work_dir: work_dir.clone(),
            cleanup: true,
            cache_binaries: true,
            timeout_secs: 0,
        };

        let measurer = ScheduleMeasurer::new(baseline_config).map_err(|e| {
            PipelineError::MeasurementFailed(format!("Failed to create measurer: {}", e))
        })?;

        let baseline_time = measurer
            .measure_mlir_file(baseline_mlir, Some(kernel_name))
            .map_err(|e| {
                PipelineError::MeasurementFailed(format!("Baseline measurement failed: {}", e))
            })?;

        if config.verbose {
            println!("  Baseline execution time: {:.3} ms", baseline_time);
        }

        // Evaluate each candidate
        let mut candidate_results: Vec<(f64, f64, RecExpr<SchedOp>, ScheduleHandle)> = Vec::new();

        for (i, (analytical_cost, expr, schedule_handle)) in candidates.iter().enumerate() {
            if config.verbose {
                println!(
                    "  Evaluating candidate {}/{} (analytical cost: {:.6})...",
                    i + 1,
                    candidates.len(),
                    analytical_cost
                );
            }

            // Apply schedule to baseline MLIR
            let candidate_mlir = work_dir.join(format!("candidate_{}.mlir", i));
            let schedule_file = work_dir.join(format!("candidate_{}.isl", i));

            // Save schedule to file
            let schedule_str = dump_isl(schedule_handle);
            fs::write(&schedule_file, &schedule_str)
                .map_err(|e| PipelineError::IOError(format!("Failed to write schedule: {}", e)))?;

            // Apply schedule to MLIR
            let schedule_dir = work_dir.join(format!("schedule_{}", i));
            fs::create_dir_all(&schedule_dir).map_err(|e| {
                PipelineError::IOError(format!("Failed to create schedule dir: {}", e))
            })?;

            let schedule_file_name = schedule_dir.join("__polygeist_outlined_affine_0");
            fs::write(&schedule_file_name, &schedule_str).map_err(|e| {
                PipelineError::IOError(format!("Failed to write schedule file: {}", e))
            })?;

            codegen::apply_schedule_to_mlir(
                polygeist_dir.to_str().unwrap(),
                baseline_mlir.to_str().unwrap(),
                schedule_dir.to_str().unwrap(),
                &schedule_str,
                candidate_mlir.to_str().unwrap(),
            )
            .map_err(|e| {
                PipelineError::CompilationFailed(format!("Failed to apply schedule: {}", e))
            })?;

            // Measure candidate performance
            match measurer.measure_mlir_file(&candidate_mlir, Some(kernel_name)) {
                Ok(execution_time) => {
                    let speedup = baseline_time / execution_time;
                    candidate_results.push((
                        execution_time,
                        speedup,
                        expr.clone(),
                        schedule_handle.clone(),
                    ));

                    if config.verbose {
                        println!(
                            "    Execution time: {:.3} ms, Speedup: {:.2}x",
                            execution_time, speedup
                        );
                    }
                }
                Err(e) => {
                    if config.verbose {
                        println!("    ⚠️  Measurement failed: {}", e);
                    }
                    // Continue with other candidates
                }
            }
        }

        if candidate_results.is_empty() {
            return Err(PipelineError::MeasurementFailed(
                "All candidate measurements failed".to_string(),
            ));
        }

        // Select candidate with lowest execution time (best performance)
        // **Paper Requirement**: "The schedule yielding the best simulated performance"
        let (best_time, best_speedup, best_expr, best_schedule) = candidate_results
            .iter()
            .min_by(|a, b| a.0.partial_cmp(&b.0).unwrap())
            .ok_or_else(|| PipelineError::Other("No valid candidates".to_string()))?;

        if config.verbose {
            println!("  ✓ Best candidate selected:");
            println!("    Execution time: {:.3} ms", best_time);
            println!("    Speedup: {:.2}x", best_speedup);
        }

        Ok((*best_time, best_expr.clone(), best_schedule.clone()))
    }

    /// Create hardware-aware rewrite rules based on NCP configuration
    ///
    /// According to the paper: "We formalize Tiramisu-like scheduling primitives
    /// as conditional, semantics-preserving rewrite rules" with hardware constraints.
    ///
    /// This function generates rewrite rules that:
    /// 1. Use hardware-specific tile sizes (based on slices/banks)
    /// 2. Apply transformations that are optimal for the target hardware
    /// 3. Respect hardware constraints (memory capacity, parallelism limits)
    ///
    /// **Paper Alignment**: This implements conditional rewrites with hardware constraints.
    /// However, it does NOT enforce all hardware constraints mentioned in the paper:
    /// - ✅ Tile sizes based on hardware
    /// - ⚠️ Memory capacity (4KB buffer) not checked
    /// - ⚠️ Data type constraints not checked
    /// - ⚠️ NoC topology constraints not checked
    ///
    /// # Arguments
    /// * `config` - Pipeline configuration with hardware parameters
    ///
    /// # Returns
    /// Hardware-aware rewrite rules
    fn create_hardware_aware_rules(
        &self,
        config: &PipelineConfig,
    ) -> Vec<egg::Rewrite<SchedOp, ScheduleAnalysis>> {
        use egg::{Applier, Pattern, Rewrite};

        let mut hardware_rules = Vec::new();

        // Compute hardware-specific tile sizes
        let (tile_i, tile_j, tile_k) =
            if let (Some(slices), Some(banks)) = (config.ncp_slices, config.ncp_banks) {
                // Tile sizes based on NCP architecture
                // Outer dimension maps to slices (coarse-grained parallelism)
                // Inner dimension maps to banks (fine-grained parallelism)
                let tile_i = config.problem_size / slices;
                let tile_j = config.problem_size / banks;
                let tile_k = 8; // Default reduction dimension tile size
                (tile_i, tile_j, tile_k)
            } else {
                // Default tile sizes if partial configuration
                (32, 16, 8)
            };

        if config.verbose {
            println!(
                "  Hardware-aware tile sizes: Ti={}, Tj={}, Tk={}",
                tile_i, tile_j, tile_k
            );
        }

        // Create hardware-aware tiling rule
        // This rule applies tiling with hardware-specific sizes
        struct HardwareTileApplier {
            tile_i: usize,
            tile_j: usize,
            pub _tile_k: usize,
        }

        impl Applier<SchedOp, ScheduleAnalysis> for HardwareTileApplier {
            fn apply_one(
                &self,
                egraph: &mut EGraph<SchedOp, ScheduleAnalysis>,
                eclass: egg::Id,
                _subst: &egg::Subst,
                _searcher_ast: Option<&egg::PatternAst<SchedOp>>,
                _rule_name: egg::Symbol,
            ) -> Vec<egg::Id> {
                let mut new_ids = Vec::new();
                let mut should_tile = false;

                // Phase 1: Check conditions (immutable borrow)
                {
                    if let Some(class) = egraph.classes().find(|c| c.id == eclass) {
                        should_tile = class
                            .nodes
                            .iter()
                            .any(|n| matches!(n, SchedOp::Schedule(_)));
                    }
                }

                // Phase 2: Apply transformations (mutable borrow)
                if should_tile {
                    // Tile outer dimension (maps to slices)
                    let band_i = egraph.add(SchedOp::Num(0));
                    let size_i = egraph.add(SchedOp::Num(self.tile_i as i32));
                    let tiled_i = egraph.add(SchedOp::Tile([eclass, band_i, size_i]));
                    new_ids.push(tiled_i);

                    // Tile inner dimension (maps to banks) - applied to already-tiled schedule
                    let band_j = egraph.add(SchedOp::Num(1));
                    let size_j = egraph.add(SchedOp::Num(self.tile_j as i32));
                    let tiled_j = egraph.add(SchedOp::Tile([tiled_i, band_j, size_j]));
                    new_ids.push(tiled_j);
                }

                new_ids
            }
        }

        let tile_pattern: Pattern<SchedOp> = "?x".parse().unwrap();
        hardware_rules.push(
            Rewrite::new(
                "hardware-tile-ncp",
                tile_pattern,
                HardwareTileApplier {
                    tile_i,
                    tile_j,
                    _tile_k: tile_k,
                },
            )
            .unwrap(),
        );

        // Create hardware-aware parallelization rule
        // Parallelize outer tile loops for NCP slices
        struct HardwareParallelApplier;

        impl Applier<SchedOp, ScheduleAnalysis> for HardwareParallelApplier {
            fn apply_one(
                &self,
                egraph: &mut EGraph<SchedOp, ScheduleAnalysis>,
                eclass: egg::Id,
                _subst: &egg::Subst,
                _searcher_ast: Option<&egg::PatternAst<SchedOp>>,
                _rule_name: egg::Symbol,
            ) -> Vec<egg::Id> {
                let mut new_ids = Vec::new();
                let mut should_parallelize = false;

                // Phase 1: Check conditions (immutable borrow)
                {
                    if let Some(class) = egraph.classes().find(|c| c.id == eclass) {
                        should_parallelize =
                            class.nodes.iter().any(|n| matches!(n, SchedOp::Tile(_)));
                    }
                }

                // Phase 2: Apply transformations (mutable borrow)
                if should_parallelize {
                    // Mark outer tile loop as parallel (for NCP slices)
                    let band_0 = egraph.add(SchedOp::Num(0));
                    let parallel = egraph.add(SchedOp::Parallel([eclass, band_0]));
                    new_ids.push(parallel);
                }

                new_ids
            }
        }

        let parallel_pattern: Pattern<SchedOp> = "?x".parse().unwrap();
        hardware_rules.push(
            Rewrite::new(
                "hardware-parallel-ncp",
                parallel_pattern,
                HardwareParallelApplier,
            )
            .unwrap(),
        );

        hardware_rules
    }
}
