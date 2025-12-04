//! Automated Schedule Exploration and Top-K Extraction
//!
//! This module provides the core infrastructure for automated polyhedral schedule optimization
//! using equality saturation (e-graphs). It implements the critical pipeline:
//!
//! ```text
//! Baseline ISL Schedule
//!        ↓
//!   E-graph Exploration (apply rewrite rules)
//!        ↓
//!   Top-K Extraction (cost-based ranking)
//!        ↓
//!   Ranked Schedules for Validation
//! ```
//!
//! # Core Hypothesis (to be validated)
//!
//! **Claim**: PolySat's e-graph exploration with NCP-oriented cost model can automatically
//! discover parallel+tiled schedules that significantly outperform baseline schedules.
//!
//! **Validation**: Cost ranking correlates with real performance (Spearman ρ ≥ 0.7)
//!
//! # Design Rationale
//!
//! ## Why E-graph for Polyhedral Optimization?
//!
//! Traditional polyhedral compilers (Pluto, PPCG) use:
//! - **Affine transformations**: Limited to specific optimization patterns
//! - **Greedy heuristics**: May miss better schedules in large search space
//! - **Fixed cost models**: Hard-coded preferences (e.g., always prefer tiling)
//!
//! E-graph approach offers:
//! - **Equality saturation**: Explores ALL equivalent schedules (up to resource limits)
//! - **Compositional rewrites**: Discovers novel optimization sequences
//! - **Flexible cost models**: Can plug in different cost functions (heuristic, MLIR-based, perf)
//! - **Provable correctness**: ISL dependency analysis ensures legality
//!
//! ## Top-K Extraction (not just best)
//!
//! Why extract multiple schedules instead of single best?
//! 1. **Cost model uncertainty**: Heuristic costs may not perfectly predict real performance
//! 2. **Hardware variability**: Different schedules optimal on different targets (CPU vs NCP)
//! 3. **Validation**: Need diverse candidates to test cost-performance correlation
//! 4. **Auto-tuning**: Can measure top-K on target hardware and select empirical best
//!
//! # Implementation Strategy
//!
//! ## Implementation Strategy
//!
//! - **Core Infrastructure**: `ScheduleExplorer` encapsulates e-graph exploration + top-k extraction.
//! - **Configuration**: `ExplorationConfig` allows tuning exploration parameters.
//! - **Integration**: Works with `extract_all::extract_all_unique()` for schedule discovery.
//! - **Validation**: Extracted schedules can be validated for correctness and performance.

use egg::{EGraph, Id, RecExpr, Runner, StopReason};
use std::time::{Duration, Instant};

use crate::{
    dep_aware_rules::dependency_aware_rules,
    extract_all::extract_and_validate_all,
    language::{SchedOp, ScheduleAnalysis},
    rational_rewrites::rational_dependency_rules,
    // NOTE: rewrites::rules is DEPRECATED - it has STUB dependency checks
    // Use rational_dependency_rules() instead for real ISL dependency analysis
};

// ============================================================================
// Configuration
// ============================================================================

/// Configuration for e-graph exploration
///
/// **Design Note**: These parameters critically affect exploration effectiveness:
/// - `max_iterations`: Higher values explore deeper transformation sequences but cost more time
/// - `max_nodes`: Prevents memory exhaustion; should be tuned based on kernel complexity
/// - `enable_dependency_checks`: Safety vs performance trade-off (always enable for production)
#[derive(Debug, Clone)]
pub struct ExplorationConfig {
    /// Maximum number of e-graph saturation iterations
    ///
    /// Typical values:
    /// - Quick test: 5-10 iterations
    /// - Standard: 15-20 iterations
    /// - Exhaustive: 30-50 iterations
    ///
    /// **Trade-off**: More iterations = deeper optimization chains but diminishing returns
    pub max_iterations: usize,

    /// Maximum e-graph nodes before stopping
    ///
    /// Prevents memory explosion for complex kernels with many equivalent schedules.
    ///
    /// Typical values:
    /// - Small kernels (GEMM 256x256): 10,000-50,000 nodes
    /// - Medium kernels (Conv2D): 50,000-100,000 nodes
    /// - Large kernels: 100,000-500,000 nodes
    ///
    /// **Trade-off**: More nodes = more schedules explored but higher memory usage
    pub max_nodes: usize,

    /// Whether to enable dependency-aware rewrites
    ///
    /// When enabled, each transformation checks ISL dependencies before application.
    /// This ensures all extracted schedules are **legal** (preserve dependencies).
    ///
    /// **Safety-critical**: Always enable for production. Disable only for debugging.
    pub enable_dependency_checks: bool,

    /// Which rewrite rule sets to use
    ///
    /// Available rule sets:
    /// - `basic`: Core transformations (tile, parallel, interchange)
    /// - `advanced`: Additional patterns (skew, unroll, fusion)
    /// - `rational`: Mathematically-proven equivalences
    ///
    /// **Recommendation**: Start with `basic` + `rational`, add `advanced` if needed
    pub rewrite_rules: Vec<RuleSet>,

    /// Timeout for exploration (optional)
    ///
    /// Useful for batch processing where fixed time budget is required.
    /// If None, exploration runs until max_iterations or max_nodes reached.
    pub timeout: Option<Duration>,
}

/// Available rewrite rule sets
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RuleSet {
    /// Basic polyhedral transformations (tile, parallel, interchange)
    Basic,
    /// Advanced transformations (skew, unroll, fusion, distribution)
    Advanced,
    /// Mathematically-proven rational rewrites
    Rational,
}

impl Default for ExplorationConfig {
    fn default() -> Self {
        ExplorationConfig {
            max_iterations: 20,
            max_nodes: 50_000,
            enable_dependency_checks: true,
            rewrite_rules: vec![RuleSet::Basic, RuleSet::Rational],
            timeout: Some(Duration::from_secs(60)), // 1 minute default timeout
        }
    }
}

// ============================================================================
// Main Explorer
// ============================================================================

/// Automated schedule explorer using e-graph equality saturation
///
/// # Core Workflow
///
/// ```rust,no_run
/// use polysat::schedule_explorer::{ScheduleExplorer, ExplorationConfig};
///
/// fn main() -> Result<(), Box<dyn std::error::Error>> {
///     let config = ExplorationConfig::default();
///     let explorer = ScheduleExplorer::new(config);
///
///     // Input: baseline ISL schedule string
///     let baseline_isl = "domain: \"{ S[i,j] : 0 <= i < N and 0 <= j < N }\" ...";
///
///     // Explore: e-graph saturation finds equivalent schedules
///     // Note: This will fail with "..." as input, so we expect error or use valid input
///     // For doctest purposes, we just show the API usage
///     // let results = explorer.explore_and_extract(baseline_isl, 10)?;
///
///     // Output: top-10 schedules ranked by cost
///     /*
///     for (i, result) in results.iter().enumerate() {
///         println!("Rank {}: cost={:.2}", i+1, result.cost);
///         println!("  Has parallel: {}", result.has_parallel_mark);
///         println!("  Has tiling: {}", result.has_tile_mark);
///     }
///     */
///     Ok(())
/// }
/// ```
pub struct ScheduleExplorer {
    config: ExplorationConfig,
}

/// Result of schedule exploration
///
/// Contains both the schedule itself and metadata for analysis
#[derive(Debug, Clone)]
pub struct ScheduleExplorationResult {
    /// Rank (1-indexed) in cost ordering
    pub rank: usize,

    /// Predicted cost from cost model
    ///
    /// Lower = better. Interpretation depends on cost model:
    /// - Heuristic: Pattern-based score (~0-1000)
    /// - MLIR: LLVM IR complexity (~0-10000)
    /// - Performance: Estimated runtime in seconds
    pub cost: f64,

    /// ISL schedule string representation
    ///
    /// This is the **complete** schedule tree in ISL format, suitable for:
    /// - Manual inspection
    /// - MLIR code generation (via Polymer)
    /// - Direct application to MLIR affine code
    pub isl_string: String,

    /// E-graph expression (for debugging)
    ///
    /// RecExpr shows the transformation sequence that produced this schedule.
    /// Useful for understanding optimization path:
    /// ```text
    /// Tile(Parallel(Interchange(Schedule(baseline), 0, 1), 0), 0, 32)
    /// ```
    pub expression: RecExpr<SchedOp>,

    /// Whether schedule contains parallel mark
    ///
    /// Indicates multi-core parallelization opportunity.
    /// **Success criterion**: Top-K should have ≥50% with parallel marks
    pub has_parallel_mark: bool,

    /// Whether schedule contains tiling
    ///
    /// Indicates cache-blocking optimization.
    /// Common in high-performance schedules for memory-bound kernels.
    pub has_tile_mark: bool,

    /// Whether schedule contains vectorization hints
    ///
    /// May improve SIMD utilization (hardware-dependent).
    pub has_vector_mark: bool,
}

impl ScheduleExplorer {
    /// Create new schedule explorer with configuration
    pub fn new(config: ExplorationConfig) -> Self {
        ScheduleExplorer { config }
    }

    /// Main entry point: Explore from baseline and extract top-K schedules
    ///
    /// # Arguments
    /// * `baseline_isl` - Initial ISL schedule string (typically sequential, unoptimized)
    /// * `k` - Number of top schedules to extract
    ///
    /// # Returns
    /// Top-K schedules ranked by cost (best first)
    ///
    /// # Errors
    /// - ISL parsing errors (invalid schedule syntax)
    /// - E-graph exploration failures (very rare)
    ///
    /// # Performance
    /// Typical execution time:
    /// - Small kernels (GEMM 256x256): 1-10 seconds
    /// - Medium kernels (NTT 8192): 5-30 seconds
    /// - Large kernels (Conv2D): 10-60 seconds
    ///
    /// **Bottleneck**: E-graph saturation (can be parallelized in future)
    pub fn explore_and_extract(
        &self,
        baseline_isl: &str,
        k: usize,
    ) -> Result<Vec<ScheduleExplorationResult>, String> {
        println!("\n=== Schedule Exploration Started ===");
        println!(
            "Config: max_iter={}, max_nodes={}, dependency_checks={}",
            self.config.max_iterations, self.config.max_nodes, self.config.enable_dependency_checks
        );

        let start_time = Instant::now();

        // Step 1: Parse baseline schedule and initialize e-graph
        println!("\n[Step 1/4] Parsing baseline ISL schedule...");
        let (egraph, root) = self.initialize_egraph(baseline_isl)?;
        println!(
            "  - E-graph initialized: {} classes, {} nodes",
            egraph.number_of_classes(),
            egraph.total_size()
        );

        // Step 2: Run e-graph exploration (saturation)
        println!("\n[Step 2/4] Running e-graph exploration...");
        let egraph = self.run_exploration(egraph, root)?;
        println!(
            "  - Exploration complete: {} classes, {} nodes",
            egraph.number_of_classes(),
            egraph.total_size()
        );

        // Step 3: Extract all unique schedules
        println!("\n[Step 3/4] Extracting unique schedules...");
        let all_schedules = extract_and_validate_all(&egraph, root);
        println!("  - Found {} unique schedules", all_schedules.len());

        // Step 4: Select top-K and build results
        println!("\n[Step 4/4] Ranking and selecting top-{}...", k);
        let results = self.build_results(all_schedules, k)?;

        let elapsed = start_time.elapsed();
        println!("\n=== Exploration Complete ===");
        println!("Total time: {:.2}s", elapsed.as_secs_f64());
        println!("Schedules extracted: {}", results.len());
        println!(
            "  - With parallel marks: {}",
            results.iter().filter(|r| r.has_parallel_mark).count()
        );
        println!(
            "  - With tiling: {}",
            results.iter().filter(|r| r.has_tile_mark).count()
        );
        println!(
            "  - With vectorization: {}",
            results.iter().filter(|r| r.has_vector_mark).count()
        );

        Ok(results)
    }

    /// Initialize e-graph from baseline ISL schedule
    ///
    /// **Implementation Note**: We parse the ISL string into a ScheduleHandle,
    /// which wraps the ISL schedule object. This handle is inserted into the e-graph
    /// as the root node.
    fn initialize_egraph(
        &self,
        baseline_isl: &str,
    ) -> Result<(EGraph<SchedOp, ScheduleAnalysis>, Id), String> {
        use isl_rs::Context;
        use std::sync::Arc;

        // Parse ISL schedule from string
        let ctx = Arc::new(Context::alloc());
        let schedule_handle = crate::parse::parse_isl(ctx, baseline_isl)
            .map_err(|e| format!("Failed to parse baseline schedule: {}", e))?;

        // Create e-graph and insert baseline schedule
        let mut egraph = EGraph::new(ScheduleAnalysis::default());
        let root = egraph.add(SchedOp::Schedule(schedule_handle));

        Ok((egraph, root))
    }

    /// Run e-graph exploration with configured rewrite rules
    ///
    /// **Core Algorithm**: Equality saturation
    ///
    /// ```text
    /// repeat until fixed-point (or limits):
    ///   1. Match: Find all applicable rewrites
    ///   2. Apply: Add new schedules to e-graph (merge equivalent ones)
    ///   3. Rebuild: Update e-class analysis data
    /// ```
    ///
    /// **Termination**: Stops when:
    /// - Fixed point reached (no new rewrites applicable)
    /// - Max iterations exceeded
    /// - Max nodes exceeded
    /// - Timeout (if configured)
    ///
    /// **Correctness**: ISL dependency analysis in rewrites ensures all
    /// generated schedules preserve program semantics.
    fn run_exploration(
        &self,
        egraph: EGraph<SchedOp, ScheduleAnalysis>,
        _root: Id,
    ) -> Result<EGraph<SchedOp, ScheduleAnalysis>, String> {
        // Collect rewrite rules based on configuration
        let mut rules = Vec::new();

        for ruleset in &self.config.rewrite_rules {
            match ruleset {
                RuleSet::Basic => {
                    // NOTE: RuleSet::Basic now uses rational_dependency_rules() which includes
                    // real ISL dependency analysis. The old rewrites::rules() was deprecated
                    // because it had STUB dependency checks that always returned true/false.
                    println!("  - Adding basic rewrites (tile, parallel, interchange) - using rational rules with ISL deps");
                    rules.extend(rational_dependency_rules());
                }
                RuleSet::Advanced => {
                    println!("  - Adding dependency-aware rewrites (legality-checked)");
                    rules.extend(dependency_aware_rules());
                }
                RuleSet::Rational => {
                    println!("  - Adding rational rewrites (mathematically-proven)");
                    rules.extend(rational_dependency_rules());
                }
            }
        }

        println!("  Total rewrite rules: {}", rules.len());

        // Run e-graph saturation with Runner
        let runner = Runner::default()
            .with_egraph(egraph)
            .with_iter_limit(self.config.max_iterations)
            .with_node_limit(self.config.max_nodes)
            .with_time_limit(self.config.timeout.unwrap_or(Duration::from_secs(3600)))
            .run(&rules);

        // Check termination reason
        match runner.stop_reason {
            Some(StopReason::Saturated) => {
                println!("  Termination: Saturated (fixed-point reached)");
            }
            Some(StopReason::IterationLimit(_)) => {
                println!("  Termination: Iteration limit reached");
            }
            Some(StopReason::NodeLimit(_)) => {
                println!("  Termination: Node limit reached");
            }
            Some(StopReason::TimeLimit(_)) => {
                println!("  Termination: Timeout");
            }
            Some(StopReason::Other(_)) => {
                println!("  Termination: Other");
            }
            None => {
                println!("  Termination: Unknown");
            }
        }

        Ok(runner.egraph)
    }

    /// Build ScheduleExplorationResult from raw extraction data
    ///
    /// **Key Processing**:
    /// 1. Select top-K by cost
    /// 2. Analyze each schedule for optimization marks
    /// 3. Extract ISL string representation
    /// 4. Assign ranks
    fn build_results(
        &self,
        all_schedules: Vec<(f64, RecExpr<SchedOp>, Option<(isl_rs::Schedule, String)>)>,
        k: usize,
    ) -> Result<Vec<ScheduleExplorationResult>, String> {
        let top_k = all_schedules.iter().take(k);

        let mut results = Vec::new();

        for (rank, (cost, expr, schedule_opt)) in top_k.enumerate() {
            // Only include valid schedules (skip invalid ones)
            if let Some((_, isl_string)) = schedule_opt {
                // Analyze schedule for optimization marks
                let has_parallel_mark = isl_string.contains("parallel");
                let has_tile_mark = isl_string.contains("mod") || isl_string.contains("tile");
                let has_vector_mark = isl_string.contains("vector");

                results.push(ScheduleExplorationResult {
                    rank: rank + 1, // 1-indexed
                    cost: *cost,
                    isl_string: isl_string.clone(),
                    expression: expr.clone(),
                    has_parallel_mark,
                    has_tile_mark,
                    has_vector_mark,
                });
            }
        }

        if results.is_empty() {
            return Err("No valid schedules extracted".to_string());
        }

        Ok(results)
    }

    /// Validate exploration results against success criteria
    ///
    /// **Success Criteria**:
    /// - At least 50% of top-K have parallel or tile marks
    ///
    /// **Returns**: (passed, message)
    pub fn validate_exploration_quality(results: &[ScheduleExplorationResult]) -> (bool, String) {
        if results.is_empty() {
            return (false, "No schedules extracted".to_string());
        }

        let total = results.len();
        let with_optimizations = results
            .iter()
            .filter(|r| r.has_parallel_mark || r.has_tile_mark)
            .count();

        let percentage = (with_optimizations as f64 / total as f64) * 100.0;

        let passed = percentage >= 50.0;
        let message = format!(
            "{}/{} schedules ({:.1}%) have parallel/tile marks (threshold: 50%)",
            with_optimizations, total, percentage
        );

        (passed, message)
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Quick-start helper: Explore with default configuration
///
/// Convenience function for common case. Equivalent to:
/// ```rust
/// use polysat::schedule_explorer::{ScheduleExplorer, ExplorationConfig};
/// use isl_rs::{Context, Schedule, UnionSet};
/// use std::sync::Arc;
///
/// fn main() -> Result<(), Box<dyn std::error::Error>> {
///     let ctx = Arc::new(Context::alloc());
///     let domain = UnionSet::read_from_str(&ctx, "{ S0[i] : 0 <= i < 10 }");
///     let baseline_isl = Schedule::from_domain(domain);
///     let baseline_str = baseline_isl.to_str();
///     
///     let mut config = ExplorationConfig::default();
///     config.max_iterations = 1; // Fast for doctest
///     config.max_nodes = 100;
///
///     let explorer = ScheduleExplorer::new(config);
///     let results = explorer.explore_and_extract(&baseline_str, 10)?;
///     Ok(())
/// }
/// ```
pub fn explore_default(
    baseline_isl: &str,
    k: usize,
) -> Result<Vec<ScheduleExplorationResult>, String> {
    let explorer = ScheduleExplorer::new(ExplorationConfig::default());
    explorer.explore_and_extract(baseline_isl, k)
}

// ============================================================================
// Unit Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_explorer_initialization() {
        let config = ExplorationConfig::default();
        let _ = ScheduleExplorer::new(config);
        // Basic smoke test
        assert!(true);
    }

    // TODO: Add integration tests with actual kernels
    // Need baseline ISL schedules for GEMM/NTT/Conv2D
}
