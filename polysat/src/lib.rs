//! PolySat: Polyhedral Schedule Optimization using Equality Saturation
//!
//! This library provides a complete pipeline for optimizing polyhedral schedules:
//! 1. Parse ISL schedules from Polygeist/isl output
//! 2. Apply dependency-aware transformations via e-graph exploration
//! 3. Extract optimal schedules using cost models
//! 4. Generate optimized MLIR code
//!
//! # Core Pipeline Flow
//! ```text
//! C code → Polygeist → ISL schedule → PolySat e-graph → Optimized schedule → MLIR
//!           ↓              ↓                ↓                    ↓
//!       MLIR + ISL    Access Patterns  Dependency-Aware    Cost Model
//!                                      Transformations     Selection
//! ```
//!
//! # Module Organization
//!
//! ## Core E-graph Infrastructure
//! - [`language`]: Schedule operations DSL and ISL integration
//! - [`optimize`]: Cost-based extraction with unified cost model interface
//! - [`rational_rewrites`]: Dependency-aware transformation rules (ISL flow analysis)
//!
//! ## Pipeline Components
//! - [`parse`]: ISL schedule I/O operations
//! - [`codegen`]: Polygeist integration (C → MLIR → ISL)
//! - [`mlir_compiler`]: MLIR compilation and execution
//! - [`execution`]: Performance measurement and validation
//!
//! ## Dependency Analysis (ISL Flow-Based)
//! - [`dependency_aware`]: Main dependency checker using ISL UnionAccessInfo
//! - [`extract_access`]: Access relation extraction from Polygeist output
//! - [`access_analysis`]: Access information data structures
//! - [`dep_aware_rules`]: Dependency-aware rewrite rules
//!
//! ## Cost Models (Unified Interface)
//! - [`optimize`]: Simple heuristic cost model (default, no external deps)
//! - [`external_cost_estimator`]: MLIR compilation-based cost model
//! - [`mlir_cost_extractor`]: MLIR cost extraction utilities
//! - NCP cost model: Integrated directly into [`optimize::ScheduleCost`]
//!
//! ## Advanced Features (Optional/Alternative Approaches)
//! - [`rational_rewrites`]: Alternative rational rewrite system
//! - [`extract_all`]: Exhaustive candidate extraction from e-graph
//! - [`dep_aware_extractor`]: Dependency-aware extraction strategy
//! - [`transformations`]: High-level polyhedral transformations (interchange, skew, etc.)
//! - [`tile_per_dimension`]: Per-dimension tiling implementation
//! - [`tile_separate_bands`]: Separate band tiling for Polygeist schedules
//! - [`band_fusion`]: Band fusion transformations
//! - [`schedule_serializer`]: Schedule serialization
//! - [`isl_printer_ffi`]: Custom ISL printer FFI
//! - [`isl_block_printer`]: Block-style ISL schedule printing
//! - [`polygeist_import`]: Polygeist/Polymer export format parsing

// ============================================================================
// Core E-graph Infrastructure
// ============================================================================

pub mod language; // Schedule operations DSL
pub mod optimize; // Cost-based extraction
pub mod schedule_properties; // RFC001: ISL-based schedule property extraction

// ============================================================================
// Access Pattern Analysis (Ground Truth Support)
// ============================================================================

pub mod polymer_access_reader; // Polymer ground-truth access file parser
                               // Reads --islexternal-dump-accesses output
                               // NOTE: rewrites module REMOVED - it had STUB dependency checks (is_interchange_safe always true,
                               // has_loop_carried_dependencies always false). Use rational_rewrites::rational_dependency_rules()
                               // which provides real ISL dependency analysis via UnionAccessInfo::compute_flow().
pub mod schedule_explorer; // Automated exploration + top-k extraction

// ============================================================================
// Pipeline Components
// ============================================================================

pub mod codegen; // Polygeist integration
pub mod execution; // Performance measurement (legacy, use schedule_measurer)
pub mod mlir_compiler; // MLIR compilation
pub mod parse; // ISL I/O
pub mod pipeline; // Unified pipeline orchestration
pub mod schedule_measurer; // Unified schedule performance measurement

// ============================================================================
// Dependency Analysis (ISL Flow-Based)
// ============================================================================

pub mod access_analysis; // Access info structures
pub mod communication_cost;
pub mod dep_aware_rules; // Dependency-aware rewrites
pub mod dependency_aware; // Main dependency checker (uses ISL UnionAccessInfo::compute_flow)
pub mod extract_access; // Access relation extraction
pub mod polygeist_import; // Polygeist export parsing // Phase 1: ISL-based communication volume analysis

// ============================================================================
// Cost Models (Unified Interface)
// ============================================================================
// Note: Simple heuristic in optimize.rs is default (no external dependencies)
// MLIR-based and NCP models are optional for specialized use cases

pub mod external_cost_estimator; // MLIR compilation-based
pub mod mlir_cost_extractor; // MLIR cost extraction
                             // NOTE: NCP cost model is integrated directly into optimize.rs::ScheduleCost

// ============================================================================
// Advanced Features (Optional/Alternative Approaches)
// ============================================================================

pub mod band_fusion; // Band fusion
pub mod dep_aware_extractor; // Dependency-aware extraction
pub mod extract_all; // Exhaustive candidate extraction
pub mod isl_block_printer; // Block-style ISL printing
pub mod isl_codegen_ffi;
pub mod rational_rewrites; // Alternative rewrite system
pub mod schedule_serializer;
pub mod tile_per_dimension; // Per-dimension tiling
pub mod tile_separate_bands; // Separate band tiling
pub mod transformations; // High-level transformations (interchange, skew, fuse, etc.) // Custom ISL printer FFI

pub use codegen::{apply_schedule_to_mlir, extract_baseline_schedule, BaselineResult};
pub use language::{mark_parallel, tile_schedule, SchedOp, ScheduleAnalysis, ScheduleHandle};
pub use mlir_compiler::{compile_and_execute, ExecutionResult};
pub use optimize::{
    export_egraph_to_dot, extract_best, extract_best_by_performance, measure_schedule,
    NCPCostConfig, NCPDomain, ScheduleCost,
};
pub use parse::{
    create_schedule_with_bands, dump_isl, load_isl_file, parse_isl, save_isl_file, save_isl_string,
    ParseError,
};
// NOTE: rewrites::rules export REMOVED - use rational_rewrites::rational_dependency_rules instead
pub use schedule_measurer::{
    compare_mlir_simple, measure_mlir_simple, KernelType, MeasurementConfig, ScheduleMeasurer,
};

// Exports for unified pipeline
pub use pipeline::{EGraphStats, PipelineConfig, PipelineError, PipelineResult, PolySatPipeline};

// Exports for dependency-aware polyhedral optimization
pub use access_analysis::{
    AccessInfo, ArrayInfo, ContextHandle, DataType, MemoryLayout,
    ScheduleHandle as AccessScheduleHandle, StmtAccess,
};
pub use band_fusion::{apply_2d_band_tile, create_2d_band_schedule, fuse_consecutive_bands};
pub use communication_cost::{
    compute_computation_cost_from_schedule, compute_dependency_aware_communication_cost,
    compute_local_footprint_penalty, compute_total_communication_volume,
    detect_reduction_tiling_penalty, get_parse_stats, infer_domain_type, reset_parse_stats,
    ComputationDomain, VolumeParseStats,
};
pub use dep_aware_extractor::{extract_and_measure_candidates, CandidateResult, ExtractionConfig};
pub use dep_aware_rules::dependency_aware_rules;
pub use dependency_aware::{
    DependencyAwareEGraph, DependencyInfo, DependencySet, SafeTransformer, TransformParams,
    TransformSuggestion,
};
pub use external_cost_estimator::{CostEstimatorConfig, ExternalCostEstimator};
pub use extract_access::{
    extract_isl_accesses_for_pattern, extract_isl_accesses_with_fallback, AccessExtractor,
};
pub use extract_all::{
    extract_and_validate_all, extract_diverse_schedules, extract_k_best_schedules,
    validate_schedule, AllCandidatesExtractor,
};
pub use mlir_cost_extractor::{
    compare_extraction_methods, extract_with_mlir_costs, BatchMLIRExtractor, MLIRCostConfig,
    MLIRCostFunction,
};
pub use polygeist_import::{AccessRelation, PolymerExport};
pub use polymer_access_reader::{read_polymer_access_file, PolymerAccessInfo};
pub use rational_rewrites::{rational_dependency_rules, LoopDependencyInfo};
// Disabled exports - see note above about ISL FFI issues
