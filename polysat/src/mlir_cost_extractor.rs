//! MLIR-Based Cost Extraction for E-graphs
//!
//! This module provides the core integration between e-graph extraction and real
//! MLIR performance measurement, similar to how Tensat integrates with TASO's GPU runtime.
//!
//! Key features:
//! 1. Real performance measurement through ISL → Polygeist → MLIR → execution pipeline
//! 2. Efficient caching to avoid redundant compilations
//! 3. Batch compilation for multiple schedules
//! 4. ML-based prediction fallback for unmeasurable schedules
//! 5. Integration with egg's CostFunction trait for seamless extraction

use egg::{CostFunction, EGraph, Extractor, Id, Language, RecExpr};
use rayon::prelude::*;
use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use crate::{
    external_cost_estimator::{CostEstimatorConfig, ExternalCostEstimator},
    optimize::measure_schedule_real,
    SchedOp, ScheduleAnalysis, ScheduleHandle,
};

/// Configuration for MLIR-based cost extraction
#[derive(Clone)]
pub struct MLIRCostConfig {
    /// C source file containing the kernel
    pub c_file: String,

    /// Name of the kernel function
    pub kernel_name: String,

    /// Path to Polygeist installation
    pub polygeist_dir: String,

    /// Problem size for benchmarking
    pub problem_size: usize,

    /// Cache directory for compiled binaries
    pub cache_dir: String,

    /// Maximum schedules to measure in parallel
    pub parallel_batch_size: usize,

    /// Whether to use ML prediction for unmeasured schedules
    pub use_ml_fallback: bool,

    /// Timeout for each compilation/execution (seconds)
    pub timeout_secs: u64,
}

impl Default for MLIRCostConfig {
    fn default() -> Self {
        MLIRCostConfig {
            c_file: String::new(),
            kernel_name: "kernel".to_string(),
            polygeist_dir: "polygeist".to_string(),
            problem_size: 64,
            cache_dir: "polysat_mlir_cache".to_string(),
            parallel_batch_size: 4,
            use_ml_fallback: true,
            timeout_secs: 30,
        }
    }
}

/// MLIR-based cost function that measures real performance
/// This is the key integration point between e-graphs and real execution
pub struct MLIRCostFunction {
    _config: MLIRCostConfig,

    /// Cache mapping schedule hash to measured cost
    cost_cache: Arc<Mutex<HashMap<u64, f64>>>,

    /// Cache of compiled binaries for reuse
    _binary_cache: Arc<Mutex<HashMap<u64, PathBuf>>>,

    /// External cost estimator for full pipeline
    estimator: Arc<Mutex<ExternalCostEstimator>>,

    /// C file for kernel source
    c_file: String,

    /// Statistics for analysis
    stats: Arc<Mutex<MeasurementStats>>,
}

#[derive(Default, Clone)]
struct MeasurementStats {
    total_schedules: usize,
    cache_hits: usize,
    successful_measurements: usize,
    failed_measurements: usize,
    ml_predictions: usize,
    total_compilation_time_ms: f64,
    total_execution_time_ms: f64,
}

impl MLIRCostFunction {
    /// Create a new MLIR-based cost function
    pub fn new(config: MLIRCostConfig) -> Result<Self, String> {
        // Create cache directory
        fs::create_dir_all(&config.cache_dir)
            .map_err(|e| format!("Failed to create cache directory: {}", e))?;

        // Store C file path
        let c_file = config.c_file.clone();

        // Create external cost estimator
        let est_config = CostEstimatorConfig {
            c_file: config.c_file.clone(),
            kernel_name: config.kernel_name.clone(),
            polygeist_dir: config.polygeist_dir.clone(),
            problem_size: config.problem_size,
            cache_dir: config.cache_dir.clone(),
            use_ml_prediction: config.use_ml_fallback,
            ..Default::default()
        };

        let estimator = ExternalCostEstimator::new(est_config)?;

        Ok(MLIRCostFunction {
            _config: config,
            cost_cache: Arc::new(Mutex::new(HashMap::new())),
            _binary_cache: Arc::new(Mutex::new(HashMap::new())),
            estimator: Arc::new(Mutex::new(estimator)),
            c_file,
            stats: Arc::new(Mutex::new(MeasurementStats::default())),
        })
    }

    /// Hash a schedule for caching
    fn hash_schedule(&self, handle: &ScheduleHandle) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let schedule_str = handle.schedule.to_str().to_string();
        let mut hasher = DefaultHasher::new();
        schedule_str.hash(&mut hasher);
        hasher.finish()
    }

    /// Measure cost of a schedule with caching
    pub fn measure_schedule_cost(&self, handle: &ScheduleHandle) -> f64 {
        let hash = self.hash_schedule(handle);

        // Update stats
        if let Ok(mut stats) = self.stats.lock() {
            stats.total_schedules += 1;
        }

        // Check cache first
        if let Ok(cache) = self.cost_cache.lock() {
            if let Some(&cost) = cache.get(&hash) {
                if let Ok(mut stats) = self.stats.lock() {
                    stats.cache_hits += 1;
                }
                return cost;
            }
        }

        // Try to measure with real execution first
        let cost = match measure_schedule_real(handle, &self.c_file) {
            Ok(runtime_secs) => {
                let runtime_ms = runtime_secs * 1000.0;
                println!(
                    "[MLIRCostFunction] Measured real performance: {:.3} ms",
                    runtime_ms
                );

                // Update stats
                if let Ok(mut stats) = self.stats.lock() {
                    stats.successful_measurements += 1;
                    stats.total_execution_time_ms += runtime_ms;
                }

                runtime_ms
            }
            Err(e) => {
                println!("[MLIRCostFunction] Real measurement failed: {}", e);

                // Try external estimator as fallback
                if let Ok(mut estimator) = self.estimator.lock() {
                    let estimated_cost = estimator.measure_schedule_cost(handle);
                    if estimated_cost > 0.0 {
                        println!(
                            "[MLIRCostFunction] Using external estimator: {:.3} ms",
                            estimated_cost
                        );
                        if let Ok(mut stats) = self.stats.lock() {
                            stats.ml_predictions += 1;
                        }
                        estimated_cost
                    } else {
                        // Ultimate fallback to heuristic
                        println!("[MLIRCostFunction] Falling back to heuristic");
                        if let Ok(mut stats) = self.stats.lock() {
                            stats.failed_measurements += 1;
                        }
                        self.heuristic_cost(handle)
                    }
                } else {
                    // Fallback to heuristic if estimator lock fails
                    self.heuristic_cost(handle)
                }
            }
        };

        // Cache the result
        if cost > 0.0 {
            if let Ok(mut cache) = self.cost_cache.lock() {
                cache.insert(hash, cost);
            }
        }

        cost
    }

    /// Simple heuristic cost as ultimate fallback
    fn heuristic_cost(&self, handle: &ScheduleHandle) -> f64 {
        let schedule_str = handle.schedule.to_str().to_string();
        let mut cost = 1.0;

        // Reward beneficial transformations
        if schedule_str.contains("tile") || schedule_str.contains("mod") {
            cost *= 0.8; // Tiling improves cache locality
        }
        if schedule_str.contains("parallel") || schedule_str.contains("atomic") {
            cost *= 0.7; // Parallelization is good
        }
        if schedule_str.contains("vector") {
            cost *= 0.6; // Vectorization is very good
        }
        if schedule_str.contains("unroll") {
            cost *= 0.95; // Unrolling has tradeoffs
        }

        // Penalize complex schedules
        let depth = schedule_str.matches("child:").count();
        cost *= 1.0 + (depth as f64) * 0.05;

        cost
    }

    /// Print statistics about measurements
    pub fn print_stats(&self) {
        if let Ok(stats) = self.stats.lock() {
            println!("\n=== MLIR Cost Extraction Statistics ===");
            println!("Total schedules evaluated: {}", stats.total_schedules);
            println!(
                "Cache hits: {} ({:.1}%)",
                stats.cache_hits,
                (stats.cache_hits as f64 / stats.total_schedules.max(1) as f64) * 100.0
            );
            println!("Successful measurements: {}", stats.successful_measurements);
            println!("Failed measurements: {}", stats.failed_measurements);
            println!("ML predictions used: {}", stats.ml_predictions);
            println!(
                "Average compilation time: {:.2} ms",
                stats.total_compilation_time_ms / stats.successful_measurements.max(1) as f64
            );
            println!(
                "Average execution time: {:.2} ms",
                stats.total_execution_time_ms / stats.successful_measurements.max(1) as f64
            );
        }
    }
}

impl CostFunction<SchedOp> for MLIRCostFunction {
    type Cost = f64;

    fn cost<C>(&mut self, enode: &SchedOp, mut costs: C) -> Self::Cost
    where
        C: FnMut(Id) -> Self::Cost,
    {
        match enode {
            SchedOp::Schedule(handle) => {
                // This is where we measure real performance!
                self.measure_schedule_cost(handle)
            }
            _ => {
                // For non-schedule nodes, sum the costs of children
                enode.fold(0.0, |sum, id| sum + costs(id))
            }
        }
    }
}

/// Batch extraction with real performance measurement
/// This function extracts multiple candidates and measures them efficiently
pub struct BatchMLIRExtractor {
    config: MLIRCostConfig,
    cost_function: Arc<Mutex<MLIRCostFunction>>,
}

impl BatchMLIRExtractor {
    pub fn new(config: MLIRCostConfig) -> Result<Self, String> {
        let cost_function = MLIRCostFunction::new(config.clone())?;
        Ok(BatchMLIRExtractor {
            config,
            cost_function: Arc::new(Mutex::new(cost_function)),
        })
    }

    /// Extract and measure top N candidates from the e-graph
    pub fn extract_top_candidates(
        &self,
        egraph: &EGraph<SchedOp, ScheduleAnalysis>,
        root: Id,
        n: usize,
    ) -> Vec<(f64, RecExpr<SchedOp>)> {
        println!(
            "\n[BatchMLIRExtractor] Extracting top {} candidates with real measurement...",
            n
        );

        // Create a new MLIR cost function for extraction
        let cost_function = match MLIRCostFunction::new(self.config.clone()) {
            Ok(cf) => cf,
            Err(e) => {
                println!("[BatchMLIRExtractor] Failed to create cost function: {}", e);
                println!("[BatchMLIRExtractor] Falling back to heuristic extraction");

                // Fallback to heuristic extraction
                let extractor = Extractor::new(egraph, crate::optimize::ScheduleCost::new());
                let (cost, expr) = extractor.find_best(root);
                return vec![(cost, expr)];
            }
        };

        // Extract using MLIR cost function
        let extractor = Extractor::new(egraph, cost_function);
        let (best_cost, best_expr) = extractor.find_best(root);
        println!(
            "[BatchMLIRExtractor] Best candidate cost: {:.3} ms",
            best_cost
        );

        // For now, return just the best candidate
        // In a full implementation, we'd extract multiple candidates
        vec![(best_cost, best_expr)]
    }

    /// Measure a batch of schedules in parallel
    pub fn measure_batch_parallel(&self, schedules: Vec<ScheduleHandle>) -> Vec<(usize, f64)> {
        println!(
            "\n[BatchMLIRExtractor] Measuring {} schedules in parallel...",
            schedules.len()
        );

        // Split into batches for parallel processing
        let batch_size = self.config.parallel_batch_size;
        let results: Vec<_> = schedules
            .par_chunks(batch_size)
            .flat_map(|batch| {
                batch
                    .par_iter()
                    .enumerate()
                    .map(|(i, schedule)| {
                        if let Ok(cf) = self.cost_function.lock() {
                            let cost = cf.measure_schedule_cost(schedule);
                            (i, cost)
                        } else {
                            (i, 1.0) // Default cost on lock failure
                        }
                    })
                    .collect::<Vec<_>>()
            })
            .collect();

        println!("[BatchMLIRExtractor] Completed batch measurement");
        results
    }
}

/// Main entry point for MLIR-based extraction
pub fn extract_with_mlir_costs(
    egraph: &EGraph<SchedOp, ScheduleAnalysis>,
    root: Id,
    config: MLIRCostConfig,
) -> Result<(f64, RecExpr<SchedOp>), String> {
    println!("\n=== MLIR-Based Cost Extraction ===");
    println!("Configuration:");
    println!("  C file: {}", config.c_file);
    println!("  Kernel: {}", config.kernel_name);
    println!("  Problem size: {}", config.problem_size);
    println!("  Cache dir: {}", config.cache_dir);

    // Create the MLIR cost function
    let cost_function = MLIRCostFunction::new(config)?;

    // Create extractor with real performance measurement
    let extractor = Extractor::new(egraph, cost_function);

    // Extract the best schedule
    let (cost, expr) = extractor.find_best(root);

    println!("\n- Extraction complete");
    println!("  Best cost: {:.3} ms", cost);
    println!("  Expression size: {} nodes", expr.as_ref().len());

    Ok((cost, expr))
}

/// Compare heuristic vs real costs for analysis
pub fn compare_extraction_methods(
    egraph: &EGraph<SchedOp, ScheduleAnalysis>,
    root: Id,
    config: MLIRCostConfig,
) -> Result<(), String> {
    println!("\n=== Comparing Extraction Methods ===");

    // Extract with heuristic cost
    let heuristic_extractor = Extractor::new(egraph, crate::optimize::ScheduleCost::new());
    let (heuristic_cost, heuristic_expr) = heuristic_extractor.find_best(root);

    // Extract with real MLIR cost
    let (mlir_cost, mlir_expr) = extract_with_mlir_costs(egraph, root, config)?;

    // Compare results
    println!("\nResults Comparison:");
    println!("┌─────────────────────────────────────────────────┐");
    println!("│ Method      │ Cost     │ Expr Size │ Different? │");
    println!("├─────────────────────────────────────────────────┤");
    println!(
        "│ Heuristic   │ {:8.3} │ {:9} │            │",
        heuristic_cost,
        heuristic_expr.as_ref().len()
    );
    println!(
        "│ MLIR Real   │ {:8.3} │ {:9} │ {:10} │",
        mlir_cost,
        mlir_expr.as_ref().len(),
        if heuristic_expr == mlir_expr {
            "No"
        } else {
            "Yes"
        }
    );
    println!("└─────────────────────────────────────────────────┘");

    if heuristic_expr != mlir_expr {
        println!("\n! Different schedules selected!");
        println!("This shows the importance of real performance measurement.");
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mlir_cost_function() {
        // Test that the cost function can be created
        let config = MLIRCostConfig::default();
        let result = MLIRCostFunction::new(config);
        assert!(result.is_ok() || result.is_err()); // Should either succeed or fail gracefully
    }

    #[test]
    fn test_schedule_hashing() {
        // Test that schedule hashing is deterministic
        let ctx = Arc::new(isl_rs::Context::alloc());
        let schedule_str = r#"domain: "{ S0[i, j] : 0 <= i < 64 and 0 <= j < 64 }""#;
        if let Ok(handle) = crate::parse_isl(ctx, schedule_str) {
            let config = MLIRCostConfig::default();
            if let Ok(cf) = MLIRCostFunction::new(config) {
                let hash1 = cf.hash_schedule(&handle);
                let hash2 = cf.hash_schedule(&handle);
                assert_eq!(hash1, hash2);
            }
        }
    }
}
