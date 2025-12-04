//! PolySat CLI - Unified Entry Point
//!
//! This is the **single, unified CLI entry point** for PolySat. It uses the
//! `pipeline.rs` module which implements the complete end-to-end workflow
//! aligned with the Meta-PIM paper requirements.
//!
//! **Paper Alignment**: This CLI provides access to the complete polyhedral +
//! equality saturation framework:
//! - E-graph exploration with conditional rewrites
//! - Hardware-aware cost model (analytical proxy)
//! - K-best extraction workflow
//! - Performance measurement (real execution as proxy for cycle-accurate simulation)
//!
//! # Usage
//!
//! ## End-to-End Optimization (C → Optimized MLIR)
//! ```bash
//! cargo run --bin polysat -- optimize-c \
//!   --c-file matmul.c \
//!   --kernel matmul \
//!   --polygeist POLYGEIST_DIR\
//!   --measure-execution \
//!   --verbose
//! ```
//!
//! ## ISL Schedule Optimization (No Polygeist Required)
//! ```bash
//! cargo run --bin polysat -- optimize-isl \
//!   --input schedule.isl \
//!   --output optimized.isl \
//!   --max-iter 10 \
//!   --check-dependencies
//! ```
//!
//! ## Hardware-Aware Optimization (NCP)
//! ```bash
//! cargo run --bin polysat -- optimize-c \
//!   --c-file gemm.c \
//!   --kernel gemm \
//!   --polygeist POLYGEIST_DIR\
//!   --ncp-slices 8 \
//!   --ncp-banks 64 \
//!   --use-hardware-cost-model \
//!   --k-best-candidates 20 \
//!   --measure-execution
//! ```

use clap::{Parser, Subcommand};
use polysat::pipeline::{PipelineConfig, PolySatPipeline};
use regex::Regex;
use std::path::PathBuf;

#[derive(Parser)]
#[clap(name = "polysat")]
#[clap(about = "PolySat - Polyhedral Schedule Optimization with Equality Saturation")]
#[clap(version = "1.0")]
struct Args {
    #[clap(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Optimize C code end-to-end (C → MLIR → ISL → E-graph → Optimized MLIR)
    ///
    /// This command runs the complete pipeline from C source code to optimized MLIR,
    /// implementing the full polyhedral + equality saturation workflow from the Meta-PIM paper.
    ///
    /// **Paper Alignment**: Implements Stages 1-7 of the pipeline:
    /// - Stage 1: C → MLIR → ISL (Polygeist)
    /// - Stage 2-3: E-graph exploration with conditional rewrites
    /// - Stage 4: Cost-based extraction (analytical proxy or k-best)
    /// - Stage 5-6: Schedule → MLIR (code generation)
    /// - Stage 7: Performance measurement (if enabled)
    OptimizeC {
        /// C source file containing the kernel
        #[clap(long = "c-file", value_name = "FILE")]
        c_file: PathBuf,

        /// Kernel function name to optimize
        #[clap(long = "kernel", short = 'k', value_name = "NAME")]
        kernel: String,

        /// Path to Polygeist installation
        #[clap(long = "polygeist", short = 'p', default_value = "polygeist")]
        polygeist_dir: String,

        /// Maximum e-graph iterations
        #[clap(long = "max-iter", default_value = "10")]
        max_iter: usize,

        /// Maximum e-graph nodes
        #[clap(long = "max-nodes", default_value = "10000")]
        max_nodes: usize,

        /// Enable dependency checking
        #[clap(long = "check-dependencies", default_value = "true")]
        check_dependencies: bool,

        /// Use hardware-aware cost model (requires --ncp-slices and --ncp-banks)
        #[clap(long = "use-hardware-cost-model")]
        use_hardware_cost_model: bool,

        /// Number of NCP slices (for hardware-aware optimization)
        #[clap(long = "ncp-slices", value_name = "NUM")]
        ncp_slices: Option<usize>,

        /// Number of NCP banks per slice (for hardware-aware optimization)
        #[clap(long = "ncp-banks", value_name = "NUM")]
        ncp_banks: Option<usize>,

        /// Extract k-best candidates using analytical cost, then measure (paper's hybrid cost model)
        #[clap(long = "k-best-candidates", value_name = "K")]
        k_best_candidates: Option<usize>,

        /// Measure real execution time (requires MLIR toolchain)
        #[clap(long = "measure-execution")]
        measure_execution: bool,

        /// Problem size for performance measurement
        #[clap(long = "problem-size", default_value = "64")]
        problem_size: usize,

        /// Output directory for results
        #[clap(long = "output", short = 'o', default_value = "polysat_output")]
        output_dir: PathBuf,

        /// Enable verbose output
        #[clap(long = "verbose", short = 'v')]
        verbose: bool,
    },

    /// Optimize an existing ISL schedule (ISL-only mode, no Polygeist required)
    ///
    /// This command optimizes an ISL schedule using equality saturation without
    /// requiring Polygeist or C source code. Useful for schedule-only optimization.
    OptimizeIsl {
        /// Input ISL schedule file
        #[clap(long = "input", short = 'i', value_name = "FILE")]
        input: PathBuf,

        /// Output ISL schedule file
        #[clap(long = "output", short = 'o', value_name = "FILE")]
        output: PathBuf,

        /// Maximum e-graph iterations
        #[clap(long = "max-iter", default_value = "10")]
        max_iter: usize,

        /// Maximum e-graph nodes
        #[clap(long = "max-nodes", default_value = "10000")]
        max_nodes: usize,

        /// Enable dependency checking
        #[clap(long = "check-dependencies", default_value = "true")]
        check_dependencies: bool,

        /// Use performance-based extraction (requires C source)
        #[clap(long = "use-performance-extraction")]
        use_performance_extraction: bool,

        /// C source file (required if --use-performance-extraction)
        /// NOTE: Performance extraction not yet supported in ISL-only mode
        #[clap(long = "c-file", value_name = "FILE")]
        _c_file: Option<PathBuf>,

        /// Output directory for intermediate files
        #[clap(long = "output-dir", default_value = "polysat_output")]
        output_dir: PathBuf,

        /// Enable verbose output
        #[clap(long = "verbose", short = 'v')]
        verbose: bool,
    },
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    match args.command {
        Commands::OptimizeC {
            c_file,
            kernel,
            polygeist_dir,
            max_iter,
            max_nodes,
            check_dependencies,
            use_hardware_cost_model,
            ncp_slices,
            ncp_banks,
            k_best_candidates,
            measure_execution,
            problem_size,
            output_dir,
            verbose,
        } => {
            optimize_c_code_command(
                c_file,
                kernel,
                polygeist_dir,
                max_iter,
                max_nodes,
                check_dependencies,
                use_hardware_cost_model,
                ncp_slices,
                ncp_banks,
                k_best_candidates,
                measure_execution,
                problem_size,
                output_dir,
                verbose,
            )?;
        }

        Commands::OptimizeIsl {
            input,
            output,
            max_iter,
            max_nodes,
            check_dependencies,
            use_performance_extraction,
            _c_file,
            output_dir,
            verbose,
        } => {
            optimize_isl_command(
                input,
                output,
                max_iter,
                max_nodes,
                check_dependencies,
                use_performance_extraction,
                output_dir,
                verbose,
            )?;
        }
    }

    Ok(())
}

/// Optimize C code end-to-end using the unified pipeline
fn optimize_c_code_command(
    c_file: PathBuf,
    kernel: String,
    polygeist_dir: String,
    max_iter: usize,
    max_nodes: usize,
    check_dependencies: bool,
    use_hardware_cost_model: bool,
    ncp_slices: Option<usize>,
    ncp_banks: Option<usize>,
    k_best_candidates: Option<usize>,
    measure_execution: bool,
    problem_size: usize,
    output_dir: PathBuf,
    verbose: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    use std::fs;

    // Create output directory
    fs::create_dir_all(&output_dir)
        .map_err(|e| format!("Failed to create output directory: {}", e))?;

    // Initialize pipeline with Polygeist
    let pipeline = PolySatPipeline::new(&polygeist_dir)
        .map_err(|e| format!("Failed to initialize pipeline: {}", e))?;

    // Build configuration
    let config = PipelineConfig {
        max_iter,
        max_nodes,
        check_dependencies,
        use_performance_extraction: false, // Use analytical cost by default
        measure_execution,
        output_dir: output_dir.clone(),
        verbose,
        problem_size,
        ncp_slices,
        ncp_banks,
        use_hardware_cost_model,
        k_best_candidates,
    };

    // Print header
    if verbose {
        println!("\n==================================================================");
        println!("         PolySat: End-to-End Optimization Pipeline              ");
        println!("==================================================================\n");
        println!("Configuration:");
        println!("  C file: {}", c_file.display());
        println!("  Kernel: {}", kernel);
        println!("  Polygeist: {}", polygeist_dir);
        println!("  E-graph: {} iterations, {} nodes", max_iter, max_nodes);
        println!("  Dependency checking: {}", check_dependencies);
        if use_hardware_cost_model {
            println!(
                "  Hardware-aware: {} slices × {} banks",
                ncp_slices.unwrap_or(0),
                ncp_banks.unwrap_or(0)
            );
        }
        if let Some(k) = k_best_candidates {
            println!("  K-best extraction: k={}", k);
        }
        println!("  Performance measurement: {}", measure_execution);
        println!("  Output: {}", output_dir.display());
        println!();
    }

    // Run pipeline
    let result = pipeline
        .optimize_c_code(&c_file, &kernel, &config)
        .map_err(|e| format!("Pipeline failed: {}", e))?;

    // Print results
    println!("\n==================================================================");
    println!("                      Optimization Results                      ");
    println!("==================================================================\n");

    println!("E-graph Statistics:");
    println!("  Total nodes: {}", result.egraph_stats.total_nodes);
    println!("  E-classes: {}", result.egraph_stats.num_classes);
    println!("  Iterations: {}", result.egraph_stats.iterations);
    println!(
        "  Saturation time: {:.2}s",
        result.egraph_stats.saturation_time
    );
    println!(
        "  Extraction time: {:.2}s",
        result.egraph_stats.extraction_time
    );

    println!("\nCost Analysis:");
    println!("  Baseline cost: {:.4}", result.baseline_cost);
    println!("  Optimized cost: {:.4}", result.cost);
    println!("  Speedup (cost-based): {:.2}x", result.speedup);

    // **CRITICAL**: Analyze tiling in optimized schedule
    // This verifies that the optimization actually produced tiled schedules
    let tiling_pattern = Regex::new(r"\w+\s*-\s*\(\s*\w+\s*\)\s*mod|\(\s*\w+\s*\)\s*mod").unwrap();
    let has_tiling = tiling_pattern.is_match(&result.schedule_str);
    let has_parallel =
        result.schedule_str.contains("parallel") || result.schedule_str.contains("permutable: 1");

    // Extract tile sizes from schedule string
    let tile_size_pattern = Regex::new(r"mod\s+(\d+)").unwrap();
    let mut tile_sizes = Vec::new();
    for cap in tile_size_pattern.captures_iter(&result.schedule_str) {
        if let Ok(size) = cap[1].parse::<usize>() {
            tile_sizes.push(size);
        }
    }

    println!("\nTiling Analysis:");
    println!("  Has tiling: {}", if has_tiling { "YES" } else { "NO" });
    println!(
        "  Has parallelization: {}",
        if has_parallel { "YES" } else { "NO" }
    );
    if !tile_sizes.is_empty() {
        println!("  Detected tile sizes: {:?}", tile_sizes);
    }

    // Verify schedule transformation
    if result.schedule_str == result.baseline_schedule_str {
        println!("  ! WARNING: Optimized schedule is IDENTICAL to baseline!");
        println!("     This suggests extraction may not be working correctly.");
    } else {
        println!("  - Optimized schedule differs from baseline");
        if has_tiling || has_parallel {
            println!("  - Optimized schedule contains transformations");
        }
    }

    if let Some(exec_time) = result.execution_time {
        println!("\nPerformance Measurement:");
        println!("  Execution time: {:.3} ms", exec_time * 1000.0);
        if result.speedup > 1.0 {
            println!(
                "  - Performance improvement achieved: {:.2}x speedup",
                result.speedup
            );
        } else if result.speedup < 1.0 {
            println!(
                "  ! Performance regression: {:.2}x slowdown",
                result.speedup
            );
        } else {
            println!("  ! No performance change");
        }
    }

    if let Some(ref mlir_file) = result.mlir_file {
        println!("\nOutput Files:");
        println!("  Optimized MLIR: {}", mlir_file.display());

        // Save optimized schedule
        let schedule_file = output_dir.join("optimized_schedule.isl");
        std::fs::write(&schedule_file, &result.schedule_str)
            .map_err(|e| format!("Failed to save schedule: {}", e))?;
        println!("  Optimized schedule: {}", schedule_file.display());
    }

    println!("\nOptimization complete!");

    Ok(())
}

/// Optimize ISL schedule using the unified pipeline (ISL-only mode)
fn optimize_isl_command(
    input: PathBuf,
    output: PathBuf,
    max_iter: usize,
    max_nodes: usize,
    check_dependencies: bool,
    use_performance_extraction: bool,
    output_dir: PathBuf,
    verbose: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    use isl_rs::Context;
    use polysat::load_isl_file;
    use polysat::save_isl_file;
    use std::fs;
    use std::sync::Arc;

    // Create output directory
    fs::create_dir_all(&output_dir)
        .map_err(|e| format!("Failed to create output directory: {}", e))?;

    // Load input schedule
    let ctx = Arc::new(Context::alloc());
    let schedule = load_isl_file(ctx.clone(), &input.to_string_lossy())
        .map_err(|e| format!("Failed to load schedule: {}", e))?;

    // Initialize pipeline without Polygeist (ISL-only mode)
    let pipeline = PolySatPipeline::new_without_polygeist(ctx.clone())
        .map_err(|e| format!("Failed to initialize pipeline: {}", e))?;

    // Build configuration
    let config = PipelineConfig {
        max_iter,
        max_nodes,
        check_dependencies,
        use_performance_extraction,
        measure_execution: false, // ISL-only mode doesn't support measurement
        output_dir,
        verbose,
        problem_size: 64,
        ncp_slices: None,
        ncp_banks: None,
        use_hardware_cost_model: false,
        k_best_candidates: None,
    };

    if verbose {
        println!("\n==================================================================");
        println!("         PolySat: ISL Schedule Optimization                    ");
        println!("==================================================================\n");
        println!("Configuration:");
        println!("  Input: {}", input.display());
        println!("  Output: {}", output.display());
        println!("  E-graph: {} iterations, {} nodes", max_iter, max_nodes);
        println!("  Dependency checking: {}", check_dependencies);
        println!("  Performance extraction: {}", use_performance_extraction);
        println!();
    }

    // Optimize schedule
    let optimized = pipeline
        .optimize_schedule(schedule, &config)
        .map_err(|e| format!("Optimization failed: {}", e))?;

    // Create output directory if it doesn't exist
    if let Some(parent) = output.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| format!("Failed to create output directory: {}", e))?;
    }

    // Save output
    save_isl_file(&optimized, output.to_str().ok_or("Invalid output path")?)
        .map_err(|e| format!("Failed to save schedule: {}", e))?;

    if verbose {
        println!("Schedule optimized and saved to {}", output.display());
    }

    Ok(())
}
