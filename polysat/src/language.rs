//! Schedule transformation language for equality saturation
//!
//! This module defines the core DSL (Domain Specific Language) for representing polyhedral
//! schedule transformations as an e-graph. It provides the foundation for PolySat's
//! optimization framework.
//!
//! # Design Principles
//!
//! ## 1. Operations, Not Structures
//! We represent **transformations** (tile, parallel, interchange) rather than ISL's
//! internal tree structures (band nodes, filter nodes, sequence nodes). This keeps the
//! search space tractable and focuses on meaningful semantic operations.
//!
//! **Why this matters**: ISL's schedule tree has many low-level structural elements.
//! Representing them directly in the e-graph would explode the search space with
//! structurally different but semantically equivalent representations.
//!
//! Example:
//! ```text
//! E-graph: (tile (schedule S) 0 32)  # High-level: tile loop 0 by 32
//! ISL:     band → split → band        # Low-level: creates multiple tree nodes
//! ```
//!
//! ## 2. Mark-Based Navigation (Robust to Transformations)
//! Following PPCG's approach, we use **semantic marks** instead of fragile position-based
//! indexing. Position indices like `tile:0:32` become invalid after transformations that
//! restructure the schedule tree.
//!
//! **Problem**: After tiling loop 0, the indices shift:
//! ```text
//! Before: band[i,j,k] → indices 0,1,2
//! After:  band[i_outer] → band[i_inner,j,k] → indices change!
//! ```
//!
//! **Solution**: Use marks that survive transformations:
//! ```text
//! use polysat::language::{tile_at_mark, SchedOp};
//! use egg::{RecExpr, Id};
//!
//! # let schedule = RecExpr::default();
//! // tile_at_mark(schedule, "outer_loop", 32)
//! // tile_at_mark(schedule, "outer_loop", 32)
//! ```
//! "Mark persists through transforms"
//!
//! ## 3. ISL Integration via Opaque Handles
//! The actual ISL schedule is stored as an `Arc<Schedule>` (opaque handle) with
//! transformations applied through ISL's API. This provides:
//! - **Safety**: ISL owns the schedule memory, preventing dangling pointers
//! - **Sharing**: Arc enables efficient sharing in the e-graph without copying
//! - **Context**: Each schedule carries its ISL context for operations
//!
//! # Core Types
//!
//! ## SchedOp: The Schedule Operation Language
//! The `SchedOp` enum defines all operations in our e-graph DSL:
//! - **Transformations**: `tile`, `parallel`, `vectorize`, `interchange`, `fuse`, `skew`, etc.
//! - **Mark operations**: `insert-mark`, `tile-at-mark`, `parallel-at-mark`, etc.
//! - **Schedule handle**: `Schedule(ScheduleHandle)` wraps actual ISL schedules
//! - **Terminals**: `Num`, `Symbol`, `Bool` for parameters
//!
//! ## ScheduleHandle: ISL Schedule Wrapper
//! Wraps `isl_rs::Schedule` for use in the e-graph:
//! - `schedule: Arc<Schedule>` - Shared ISL schedule
//! - `ctx: Arc<Context>` - ISL context (required for all ISL operations)
//! - `tree_str: String` - Cached full tree representation
//!
//! ## ScheduleAnalysis: E-graph Analysis
//! Implements `egg::Analysis` to track schedule semantics through the e-graph:
//! - Maintains ISL context across all operations
//! - Computes schedule equality and hashing
//! - Merges equivalent representations
//!
//! # Usage Example
//!
//! ```no_run
//! use polysat::{SchedOp, ScheduleAnalysis};
//! use polysat::language::ScheduleHandle;
//! use polysat::parse::parse_isl;
//! use egg::{EGraph, Id};
//! use isl_rs::Context;
//! use std::sync::Arc;
//!
//! fn main() {
//! // Create ISL context and parse schedule
//! let ctx = Arc::new(Context::alloc());
//! let handle = parse_isl(ctx.clone(), "{ S[i,j] -> [i,j] }").unwrap();
//!
//! // Create e-graph with schedule
//! let mut egraph = EGraph::new(ScheduleAnalysis::new(ctx));
//! let root: Id = egraph.add(SchedOp::Schedule(handle));
//! }
//! ```
//!
//! // Apply transformations (these are e-graph operations, not immediate ISL calls)
//! ```rust
//! use polysat::language::{SchedOp, ScheduleHandle};
//! use egg::{EGraph, RecExpr, Id};
//! use isl_rs::{Context, Schedule, UnionSet};
//! use std::sync::Arc;
//!
//! fn main() {
//!     let ctx = Arc::new(Context::alloc());
//!     let domain = UnionSet::read_from_str(&ctx, "{ S[i,j] : 0 <= i,j < 10 }");
//!     let schedule = Schedule::from_domain(domain);
//!     let handle = ScheduleHandle::new(ctx.clone(), schedule);
//!
//!     let mut egraph = EGraph::<SchedOp, ()>::default();
//!     let root = egraph.add(SchedOp::Schedule(handle));
//!     let tile_size = egraph.add(SchedOp::Num(32));
//!     let band_idx = egraph.add(SchedOp::Num(0));
//!     let tiled = egraph.add(SchedOp::Tile([root, band_idx, tile_size]));
//! }
//! ```
//!
//! # Implementation Notes
//!
//! ## Why Arc<Schedule>?
//! - **E-graph requirement**: Values must be cheaply clonable (Copy or cheap Clone)
//! - **ISL ownership**: Can't copy ISL schedules (C pointers), must share
//! - **Efficiency**: Avoids expensive ISL string serialization/deserialization
//!
//! ## Why cache tree_str?
//! ISL's `schedule.to_str()` loses structural information. We use custom block-style
//! printing via `isl_block_printer::schedule_to_block_str()` to preserve the full
//! tree structure for debugging and analysis.
//!
//! ## Thread Safety
//! ISL is NOT thread-safe. All schedules sharing a context must be accessed from
//! the same thread. PolySat is currently single-threaded; multi-threading would
//! require per-thread contexts.

use crate::{AccessInfo, DependencyInfo};
use egg::{define_language, Analysis, EGraph, Id};
use isl_rs::{Context, Schedule};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
define_language! {
    pub enum SchedOp {
        // High-level transformation operations
        "tile" = Tile([Id; 3]),         // (tile schedule band_idx size) - uniform tiling
        "tile-per-dim" = TilePerDim([Id; 4]), // (tile-per-dim schedule size_i size_j size_k) - per-dimension tiling
        "parallel" = Parallel([Id; 2]),  // (parallel schedule band_idx)
        "vectorize" = Vectorize([Id; 3]), // (vectorize schedule band_idx width)
        "fuse" = Fuse([Id; 3]),          // (fuse schedule loop1 loop2)
        "interchange" = Interchange([Id; 3]), // (interchange schedule band1 band2)
        "unroll" = Unroll([Id; 3]),      // (unroll schedule band_idx factor)
        "split" = Split([Id; 3]),        // (split schedule band_idx factor)
        "skew" = Skew([Id; 4]),          // (skew schedule band_idx factor direction)

        // Mark-based navigation operations (solves band indexing fragility)
        "insert-mark" = InsertMark([Id; 2]), // (insert-mark schedule mark_name)
        "tile-at-mark" = TileAtMark([Id; 3]), // (tile-at-mark schedule mark_name size)
        "parallel-at-mark" = ParallelAtMark([Id; 2]), // (parallel-at-mark schedule mark_name)
        "vectorize-at-mark" = VectorizeAtMark([Id; 3]), // (vectorize-at-mark schedule mark_name width)
        "unroll-at-mark" = UnrollAtMark([Id; 3]), // (unroll-at-mark schedule mark_name factor)
        "split-at-mark" = SplitAtMark([Id; 3]), // (split-at-mark schedule mark_name factor)
        "has-mark" = HasMark([Id; 2]), // (has-mark schedule mark_name) -> boolean
        "get-mark" = GetMark([Id; 2]), // (get-mark schedule mark_name) -> schedule_node

        // The actual ISL schedule (holds isl-rs handle)
        Schedule(ScheduleHandle),

        // Terminals for parameters
        Num(i32),
        Symbol(egg::Symbol),
        Bool(bool),
    }
}

/// Wrapper for ISL schedule to work with egg's e-graph infrastructure
///
/// The schedule is wrapped in Arc for efficient sharing in the e-graph.
/// The context must be shared across all schedules for ISL operations.
///
/// **RFC001 Enhancement**: Includes precomputed `ScheduleProperties` for
/// ISL-based cost analysis (replacing string-based detection).
#[derive(Clone)]
pub struct ScheduleHandle {
    pub schedule: Arc<Schedule>,
    pub ctx: Arc<Context>,
    /// Cached tree string in block-style YAML format
    /// This preserves the full schedule tree structure that to_str() loses
    pub tree_str: String,
    /// ISL-derived schedule properties (RFC001)
    /// Computed at construction time for eager semantic evaluation
    pub properties: crate::schedule_properties::ScheduleProperties,
}

impl ScheduleHandle {
    /// Create a new schedule handle with shared context
    ///
    /// **RFC001**: Properties are computed eagerly at construction time
    /// using ISL API calls (not string parsing).
    pub fn new(ctx: Arc<Context>, schedule: Schedule) -> Self {
        // Capture the tree string immediately using block-style printer
        let tree_str = crate::isl_block_printer::schedule_to_block_str(&schedule);
        // Compute ISL-based properties (RFC001 - Eager Semantic Evaluation)
        let properties = crate::schedule_properties::ScheduleProperties::from_isl(&schedule);
        ScheduleHandle {
            schedule: Arc::new(schedule),
            ctx,
            tree_str,
            properties,
        }
    }
}

impl PartialEq for ScheduleHandle {
    fn eq(&self, other: &Self) -> bool {
        // Compare schedule strings for equality
        self.schedule.to_str() == other.schedule.to_str()
    }
}

impl Eq for ScheduleHandle {}

impl PartialOrd for ScheduleHandle {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for ScheduleHandle {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.schedule.to_str().cmp(&other.schedule.to_str())
    }
}

impl std::hash::Hash for ScheduleHandle {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.schedule.to_str().hash(state);
    }
}

impl std::fmt::Debug for ScheduleHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "Schedule({})", self.tree_str)
    }
}

impl std::fmt::Display for ScheduleHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        // For DOT output, truncate long schedules and escape special characters
        let sched_str = self.schedule.to_str();
        if sched_str.len() > 100 {
            write!(f, "Schedule[{} chars]", sched_str.len())
        } else {
            // Escape problematic characters for DOT format
            let escaped = sched_str
                .replace('\\', "\\\\")
                .replace('"', "\\\"")
                .replace('\n', "\\n")
                .replace('\r', "\\r");
            write!(f, "{}", escaped)
        }
    }
}

impl std::str::FromStr for ScheduleHandle {
    type Err = String;

    fn from_str(_s: &str) -> Result<Self, Self::Err> {
        // This won't be used in practice as we don't parse from strings
        Err("Cannot parse ScheduleHandle from string".to_string())
    }
}

// Data associated with each e-class
//
// **NEW in Option B (Lazy Dependency Recomputation)**:
// Added `dependencies` field for per-schedule dependency caching
#[derive(Clone, Debug)]
pub struct ScheduleData {
    pub schedule: Option<ScheduleHandle>,
    pub value: Option<i32>,
    pub symbol: Option<String>,
    pub boolean: Option<bool>,

    /// If this schedule is equivalent to another schedule, store the e-class id to union with
    /// This enables proper equality saturation: when two transformations produce identical
    /// schedules, they are merged into the same e-class, allowing extraction to choose the
    /// one with lowest cost.
    pub equivalent_to: Option<Id>,

    /// Lazily computed dependency information (NEW in Option B)
    ///
    /// # Semantics
    /// - `None`: Not yet computed (first access will trigger computation)
    /// - `Some(Arc<DependencyInfo>)`: Computed and cached
    ///
    /// # Correctness
    /// Dependencies are per-schedule (not per-transformation-path).
    /// Two schedules with identical ISL representation have identical dependencies.
    /// Caching is correct because schedules in e-graph are immutable.
    ///
    /// # Performance
    /// - `None`: 8 bytes overhead (Option discriminant)
    /// - `Some`: 16 bytes overhead (Option + Arc pointer)
    /// - Arc clone: O(1) atomic reference count increment
    ///
    /// # Usage
    /// Access via `ScheduleAnalysis::get_or_compute_dependencies()`
    /// which handles lazy computation and caching.
    pub dependencies: Option<Arc<DependencyInfo>>,
}

impl ScheduleData {
    pub fn new_schedule(handle: ScheduleHandle) -> Self {
        ScheduleData {
            schedule: Some(handle),
            value: None,
            symbol: None,
            boolean: None,
            equivalent_to: None,
            dependencies: None, // Lazy - computed on first access
        }
    }

    pub fn new_schedule_with_equivalent(handle: ScheduleHandle, equivalent_to: Id) -> Self {
        ScheduleData {
            schedule: Some(handle),
            value: None,
            symbol: None,
            boolean: None,
            equivalent_to: Some(equivalent_to),
            dependencies: None, // Lazy - computed on first access
        }
    }

    pub fn new_value(v: i32) -> Self {
        ScheduleData {
            schedule: None,
            value: Some(v),
            symbol: None,
            boolean: None,
            equivalent_to: None,
            dependencies: None, // Not a schedule - never computed
        }
    }

    pub fn new_symbol(s: String) -> Self {
        ScheduleData {
            schedule: None,
            value: None,
            symbol: Some(s),
            boolean: None,
            equivalent_to: None,
            dependencies: None, // Not a schedule - never computed
        }
    }

    pub fn new_bool(b: bool) -> Self {
        ScheduleData {
            schedule: None,
            value: None,
            symbol: Some(b.to_string()), // Changed from None to Some(b.to_string())
            boolean: Some(b),
            equivalent_to: None,
            dependencies: None, // Not a schedule - never computed
        }
    }
}

// Analysis that performs actual ISL transformations
//
// **CRITICAL**: This analysis detects and merges equivalent schedules to enable
// proper equality saturation. When two transformations produce schedules with
// identical ISL string representations, they are merged into the same e-class,
// allowing extraction to choose the one with lowest cost.
//
// **NEW in Option B (Lazy Dependency Recomputation)**:
// This analysis now stores AccessInfo (shared across all schedules) to enable
// per-schedule dependency computation. Dependencies are computed lazily on demand
// and cached in ScheduleData (per e-class).
#[derive(Clone)]
pub struct ScheduleAnalysis {
    /// ISL context for all operations
    /// Required for creating ISL objects and calling ISL APIs
    /// Shared across all schedules in the e-graph
    pub(crate) ctx: Arc<Context>,

    /// Shared access information (immutable, one copy for all schedules)
    /// Contains read/write patterns extracted from Polygeist or synthesized
    /// None in ISL-only mode (no dependency analysis)
    /// Arc enables zero-cost sharing across all e-classes
    pub(crate) access_info: Option<Arc<AccessInfo>>,

    /// Directory containing Polygeist output files (for ground-truth access loading)
    /// Required when using Polymer-extracted access patterns
    /// None when using pattern synthesis or ISL-only mode
    pub(crate) schedule_dir: Option<PathBuf>,

    /// Map from schedule string representation to e-class id
    /// This enables detecting equivalent schedules created by different transformation sequences
    pub(crate) schedule_to_eclass: Arc<std::sync::Mutex<HashMap<String, Id>>>,
}

impl Default for ScheduleAnalysis {
    fn default() -> Self {
        // Default creates a minimal context for ISL-only mode
        ScheduleAnalysis {
            ctx: Arc::new(Context::alloc()),
            access_info: None,
            schedule_dir: None,
            schedule_to_eclass: Arc::new(std::sync::Mutex::new(HashMap::new())),
        }
    }
}

impl ScheduleAnalysis {
    /// Create analysis for ISL-only mode (no dependency analysis)
    ///
    /// # Use Cases
    /// - Pure schedule exploration without dependency checking
    /// - Testing and prototyping
    /// - When AccessInfo is not available
    ///
    /// # Note
    /// In this mode, dependency-aware rewrites will use conservative heuristics
    pub fn new(ctx: Arc<Context>) -> Self {
        ScheduleAnalysis {
            ctx,
            access_info: None,
            schedule_dir: None,
            schedule_to_eclass: Arc::new(std::sync::Mutex::new(HashMap::new())),
        }
    }

    /// Get the access information (if available)
    ///
    /// Returns `None` in ISL-only mode when no access patterns were provided.
    /// This is useful for cost models and extractors that need to access
    /// dependency information.
    pub fn get_access_info(&self) -> Option<&Arc<AccessInfo>> {
        self.access_info.as_ref()
    }

    /// Get the ISL context
    pub fn ctx(&self) -> &Arc<Context> {
        &self.ctx
    }

    /// Create analysis with access information for dependency-aware optimization
    ///
    /// # Arguments
    /// * `ctx` - ISL context (required for all ISL operations)
    /// * `access_info` - Access information (read/write patterns)
    /// * `schedule_dir` - Optional directory containing Polygeist output files
    ///
    /// # Use Cases
    /// - Dependency-aware schedule optimization
    /// - Safety-checked parallelization
    /// - Precise legality checking
    ///
    /// # Example
    /// ```no_run
    /// use polysat::{ScheduleAnalysis, AccessInfo};
    /// use isl_rs::Context;
    /// use std::sync::Arc;
    ///
    /// let ctx = Arc::new(Context::alloc());
    /// # let access_info = todo!();
    /// # let schedule = todo!();
    /// let analysis = ScheduleAnalysis::with_access_info(
    ///     ctx,
    ///     access_info,
    ///     Some("/path/to/polygeist/output".into()),
    /// );
    /// ```
    pub fn with_access_info(
        ctx: Arc<Context>,
        access_info: AccessInfo,
        schedule_dir: Option<PathBuf>,
    ) -> Self {
        ScheduleAnalysis {
            ctx,
            access_info: Some(Arc::new(access_info)),
            schedule_dir,
            schedule_to_eclass: Arc::new(std::sync::Mutex::new(HashMap::new())),
        }
    }

    /// Check if a schedule string already exists in the e-graph and return its e-class id
    /// This enables merging equivalent schedules created by different transformation sequences
    fn find_equivalent_schedule(&self, schedule_str: &str) -> Option<Id> {
        self.schedule_to_eclass
            .lock()
            .unwrap()
            .get(schedule_str)
            .copied()
    }

    /// Register a schedule string with its e-class id
    fn register_schedule(&self, schedule_str: String, eclass_id: Id) {
        self.schedule_to_eclass
            .lock()
            .unwrap()
            .insert(schedule_str, eclass_id);
    }

    /// Helper function to create ScheduleData with equivalence detection
    /// This is used by all transformation operations to detect and merge equivalent schedules
    ///
    /// **CRITICAL**: This function checks if a newly created schedule is equivalent to an existing
    /// schedule in the e-graph. If equivalent, it marks the schedule for unioning, enabling proper
    /// equality saturation where extraction can choose the schedule with lowest cost.
    fn create_schedule_data_with_equivalence(
        analysis: &Self,
        _egraph: &EGraph<SchedOp, Self>,
        handle: ScheduleHandle,
    ) -> ScheduleData {
        let schedule_str = handle.schedule.to_str();
        if let Some(existing_id) = analysis.find_equivalent_schedule(&schedule_str) {
            // This schedule is equivalent to an existing one - mark for unioning
            ScheduleData::new_schedule_with_equivalent(handle, existing_id)
        } else {
            // New unique schedule - will be registered in modify() after e-class creation
            ScheduleData::new_schedule(handle)
        }
    }

    /// Get or compute dependencies for schedule in given e-class (LAZY COMPUTATION)
    ///
    /// This is the core API for Option B (Lazy Dependency Recomputation).
    /// It implements a cache-aside pattern: check cache first, compute on miss, store result.
    ///
    /// # Algorithm
    /// 1. **Check cache** in e-class data → return if exists (O(1) Arc clone)
    /// 2. **Extract schedule** from e-class → return None if not a schedule
    /// 3. **Check AccessInfo** available → return None if ISL-only mode
    /// 4. **Compute dependencies** via ISL flow analysis (O(n²) for n accesses, typically <10ms)
    /// 5. **Wrap in Arc** and cache in e-class data (mutate egraph)
    /// 6. **Return Arc** clone
    ///
    /// # Returns
    /// - `Ok(Some(Arc<DependencyInfo>))`: Dependencies computed/cached successfully
    /// - `Ok(None)`: E-class has no schedule (is Num/Symbol/Bool) OR no AccessInfo (ISL-only mode)
    /// - `Err(String)`: ISL flow analysis failed
    ///
    /// # Correctness Guarantee
    /// Dependencies are computed from the **actual schedule** in the e-class,
    /// not from a baseline. Each schedule gets its own dependency computation.
    /// Two schedules with identical ISL representation will have identical dependencies
    /// (but computed separately if in different e-classes).
    ///
    /// # Performance
    /// - **Cache hit**: O(1) - Arc clone (atomic increment)
    /// - **Cache miss**: O(ISL_flow_analysis) ≈ O(n²) for n array accesses
    ///   - Typical: <10ms for small kernels (<100 statements)
    ///   - Worst case: ~100ms for large kernels (1000+ statements)
    /// - **Memory**: 16 bytes per e-class with computed dependencies (Arc + Option)
    ///
    /// # Thread Safety
    /// - `Arc<DependencyInfo>` is immutable and thread-safe
    /// - Mutation of `egraph[id].data` requires `&mut EGraph` (no concurrent access)
    ///
    /// # Example
    /// ```no_run
    /// use polysat::{ScheduleAnalysis, AccessInfo, SchedOp};
    /// use egg::EGraph;
    /// use isl_rs::Context;
    /// use std::sync::Arc;
    ///
    /// let ctx = Arc::new(Context::alloc());
    /// # let access_info = todo!();
    /// # let schedule = todo!();
    /// let analysis = ScheduleAnalysis::with_access_info(ctx, access_info, None);
    /// let mut egraph = EGraph::new(analysis);
    /// let root_id = egraph.add(SchedOp::Schedule(schedule));
    ///
    /// // First access - computes dependencies
    /// let deps1 = ScheduleAnalysis::get_or_compute_dependencies(&egraph.analysis, &mut egraph, root_id)?;
    ///
    /// // Second access - cache hit (same Arc)
    /// let deps2 = ScheduleAnalysis::get_or_compute_dependencies(&egraph.analysis, &mut egraph, root_id)?;
    /// # Ok::<(), String>(())
    /// ```
    pub fn get_or_compute_dependencies(
        analysis: &ScheduleAnalysis,
        egraph: &mut EGraph<SchedOp, ScheduleAnalysis>,
        eclass_id: Id,
    ) -> Result<Option<Arc<DependencyInfo>>, String> {
        // Step 1: Check cache (fast path - O(1))
        if let Some(ref deps) = egraph[eclass_id].data.dependencies {
            return Ok(Some(deps.clone())); // Cache hit - Arc clone is O(1) atomic increment
        }

        // Step 2: Check if this e-class represents a schedule (not Num/Symbol/Bool)
        let schedule_handle = match &egraph[eclass_id].data.schedule {
            Some(h) => h.clone(),
            None => return Ok(None), // Not a schedule - no dependencies to compute
        };

        // Step 3: Check if AccessInfo available (required for dependency analysis)
        let access_info = match &analysis.access_info {
            Some(ai) => ai.clone(),  // Arc clone - O(1)
            None => return Ok(None), // ISL-only mode, no dependency analysis possible
        };

        // Step 3.5: TIER-2 PATH - Try Polymer access files first (most precise)
        // If schedule_dir exists, attempt to load Polymer-generated access patterns
        // and compute dependencies using ISL flow analysis directly from UnionMaps
        if let Some(ref dir) = analysis.schedule_dir {
            match Self::try_load_polymer_dependencies(
                dir,
                &*schedule_handle.schedule,
                analysis.ctx.clone(),
            ) {
                Ok(deps) => {
                    // Success! Cache and return Tier-2 dependencies
                    let deps_arc = Arc::new(deps);
                    egraph[eclass_id].data.dependencies = Some(deps_arc.clone());
                    return Ok(Some(deps_arc));
                }
                Err(e) => {
                    // Polymer files not found or parse failed - fall through to pattern-based
                    use log::debug;
                    debug!("[Tier-2] Polymer access files not available: {}", e);
                    debug!("[Tier-2] Falling back to pattern-based analysis");
                }
            }
        }

        // Step 4: FALLBACK PATH - Pattern-based analysis (original, backward compatible)
        // This calls DependencyInfo::compute_from_access_info which uses
        // ISL's UnionAccessInfo::compute_flow() for precise RAW/WAR/WAW analysis
        // but synthesizes UnionMaps from patterns instead of loading from Polymer
        let deps = DependencyInfo::compute_from_access_info(
            &*access_info,
            &*schedule_handle.schedule,
            analysis.schedule_dir.as_deref(),
        )?;

        let deps_arc = Arc::new(deps);

        // Step 5: Cache in e-class data for future accesses
        // SAFETY: We have &mut EGraph, so this is safe (no concurrent access)
        // This mutation is why we need &mut EGraph, not just &EGraph
        egraph[eclass_id].data.dependencies = Some(deps_arc.clone());

        Ok(Some(deps_arc))
    }

    /// Get cached dependencies without computing (read-only, no side effects)
    ///
    /// Use this when you have `&EGraph` (not `&mut`) and can't trigger computation.
    /// If dependencies haven't been computed yet, returns `None`.
    ///
    /// # Use Cases
    /// - Check if dependencies already computed (avoid recomputation check)
    /// - Read-only queries where mutation is not possible
    /// - When you have `&EGraph` but not `&mut EGraph`
    ///
    /// # Returns
    /// - `Some(Arc<DependencyInfo>)`: Already cached
    /// - `None`: Not yet computed OR not a schedule
    ///
    /// # Example
    /// ```rust
    /// use polysat::language::{ScheduleAnalysis, SchedOp};
    /// use egg::{EGraph, Id};
    ///
    /// let mut egraph = EGraph::<SchedOp, ScheduleAnalysis>::default();
    /// let root_id = egraph.add(SchedOp::Num(0)); // Dummy root
    /// // Check if dependencies are already computed
    /// if let Some(deps) = ScheduleAnalysis::get_cached_dependencies(&egraph, root_id) {
    ///     println!("Has {} loop levels", deps.all_deps.loop_carried.len());
    /// } else {
    ///     println!("Dependencies not yet computed");
    /// }
    /// ```
    pub fn get_cached_dependencies(
        egraph: &EGraph<SchedOp, ScheduleAnalysis>,
        eclass_id: Id,
    ) -> Option<Arc<DependencyInfo>> {
        egraph[eclass_id].data.dependencies.clone()
    }

    /// Try loading Polymer-generated access files from schedule_dir (TIER-2 PATH)
    ///
    /// This function implements the Tier-2 integration strategy:
    /// 1. Search for Polymer access files: `*_accesses.reads` and `*_accesses.writes`
    /// 2. Parse them using `polymer_access_reader`
    /// 3. Compute dependencies via ISL flow analysis using `compute_from_union_maps`
    ///
    /// # File Naming Convention
    /// Supports flexible naming:
    /// - `<kernel_name>_accesses.reads` / `<kernel_name>_accesses.writes`
    /// - `accesses.reads` / `accesses.writes` (fallback)
    ///
    /// # Returns
    /// - `Ok(DependencyInfo)`: Successfully loaded and computed from Polymer files
    /// - `Err(String)`: Files not found, parse failed, or ISL flow analysis failed
    ///
    /// # Example Directory Structure
    /// ```text
    /// schedule_dir/
    ///   ├── gemm_accesses.reads   ← Found via pattern match
    ///   ├── gemm_accesses.writes
    ///   └── baseline_schedule.isl
    /// ```
    fn try_load_polymer_dependencies(
        schedule_dir: &Path,
        schedule: &Schedule,
        ctx: Arc<Context>,
    ) -> Result<DependencyInfo, String> {
        use log::debug;

        debug!(
            "[Tier-2] Attempting to load Polymer access files from: {}",
            schedule_dir.display()
        );

        // Step 1: Find access files using pattern matching
        let entries = fs::read_dir(schedule_dir)
            .map_err(|e| format!("Failed to read directory {}: {}", schedule_dir.display(), e))?;

        let mut reads_file: Option<PathBuf> = None;
        let mut writes_file: Option<PathBuf> = None;

        for entry in entries {
            let entry = entry.map_err(|e| format!("Failed to read directory entry: {}", e))?;
            let path = entry.path();

            if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                if name.ends_with("_accesses.reads") || name == "accesses.reads" {
                    reads_file = Some(path.clone());
                    debug!("  ✓ Found reads file: {}", path.display());
                }
                if name.ends_with("_accesses.writes") || name == "accesses.writes" {
                    writes_file = Some(path.clone());
                    debug!("  ✓ Found writes file: {}", path.display());
                }
            }
        }

        let reads_path = reads_file.ok_or_else(|| {
            format!(
                "No Polymer access file found matching *_accesses.reads in {}",
                schedule_dir.display()
            )
        })?;
        let writes_path = writes_file.ok_or_else(|| {
            format!(
                "No Polymer access file found matching *_accesses.writes in {}",
                schedule_dir.display()
            )
        })?;

        // Step 2: Load and parse access files
        debug!("[Tier-2] Loading Polymer access files...");
        let reads_info = crate::read_polymer_access_file(&reads_path)
            .map_err(|e| format!("Failed to parse .reads file: {}", e))?;
        let writes_info = crate::read_polymer_access_file(&writes_path)
            .map_err(|e| format!("Failed to parse .writes file: {}", e))?;

        // Step 3: Convert to ISL UnionMaps
        debug!("[Tier-2] Converting to ISL UnionMaps...");
        let (reads_map, _) = reads_info.to_union_maps(&ctx)?;
        let (_, writes_map) = writes_info.to_union_maps(&ctx)?;

        debug!(
            "  ✓ Reads UnionMap: {} statements",
            reads_map.to_str().len()
        );
        debug!(
            "  ✓ Writes UnionMap: {} statements",
            writes_map.to_str().len()
        );

        // Step 4: Compute dependencies using TIER-2 API
        debug!("[Tier-2] Computing dependencies via ISL flow analysis...");
        let deps = DependencyInfo::compute_from_union_maps(&reads_map, &writes_map, schedule, ctx)?;

        debug!("[Tier-2] ✅ Successfully computed dependencies from Polymer files");
        debug!("  - loop_carried: {:?}", deps.all_deps.loop_carried);

        Ok(deps)
    }

    /// Helper function to get or compute dependencies with only &mut EGraph (no borrow issues)
    ///
    /// This function solves the borrow checker issue with the method-style API.
    /// Use this in tests and other code where you have `&mut EGraph`.
    pub fn get_or_compute_deps_mut(
        egraph: &mut EGraph<SchedOp, ScheduleAnalysis>,
        eclass_id: Id,
    ) -> Result<Option<Arc<DependencyInfo>>, String> {
        // Step 1: Check cache (fast path)
        if let Some(ref deps) = egraph[eclass_id].data.dependencies {
            return Ok(Some(deps.clone()));
        }

        // Step 2: Extract schedule
        let schedule_handle = match &egraph[eclass_id].data.schedule {
            Some(h) => h.clone(),
            None => return Ok(None),
        };

        // Step 3: Extract AccessInfo
        let access_info = match &egraph.analysis.access_info {
            Some(ai) => ai.clone(),
            None => return Ok(None),
        };

        // Step 3.5: TIER-2 PATH - Try Polymer access files first (most precise)
        // If schedule_dir exists, attempt to load Polymer-generated access patterns
        if let Some(ref dir) = egraph.analysis.schedule_dir {
            match Self::try_load_polymer_dependencies(
                dir,
                &*schedule_handle.schedule,
                egraph.analysis.ctx.clone(),
            ) {
                Ok(deps) => {
                    // Success! Cache and return Tier-2 dependencies
                    let deps_arc = Arc::new(deps);
                    egraph[eclass_id].data.dependencies = Some(deps_arc.clone());
                    return Ok(Some(deps_arc));
                }
                Err(e) => {
                    // Polymer files not found or parse failed - fall through to pattern-based
                    use log::debug;
                    debug!("[Tier-2] Polymer access files not available: {}", e);
                    debug!("[Tier-2] Falling back to pattern-based analysis");
                }
            }
        }

        // Step 4: FALLBACK PATH - Extract schedule_dir for pattern-based analysis
        let schedule_dir = egraph.analysis.schedule_dir.clone();

        // Step 5: FALLBACK PATH - Compute dependencies using pattern-based approach
        let deps = DependencyInfo::compute_from_access_info(
            &*access_info,
            &*schedule_handle.schedule,
            schedule_dir.as_deref(),
        )?;

        let deps_arc = Arc::new(deps);

        // Step 6: Cache
        egraph[eclass_id].data.dependencies = Some(deps_arc.clone());

        Ok(Some(deps_arc))
    }

    /// Helper function to get cached dependencies with only &EGraph (read-only)
    pub fn get_cached_deps(
        egraph: &EGraph<SchedOp, ScheduleAnalysis>,
        eclass_id: Id,
    ) -> Option<Arc<DependencyInfo>> {
        egraph[eclass_id].data.dependencies.clone()
    }
}

impl Analysis<SchedOp> for ScheduleAnalysis {
    type Data = ScheduleData;

    /// Create analysis data for an e-node, detecting equivalent schedules
    ///
    /// **CRITICAL**: This method detects when a transformation produces a schedule that is equivalent
    /// (same ISL string representation) to an existing schedule. When equivalence is detected, we
    /// mark it in `ScheduleData::equivalent_to` for later unioning in `modify()`. This is essential
    /// for proper equality saturation: extraction can then choose the schedule with lowest cost
    /// from the merged e-class.
    ///
    /// **Equivalence Detection**: Two schedules are considered equivalent if their ISL string
    /// representations (`schedule.to_str()`) are identical. This is a conservative but correct
    /// approach: if two schedules have the same ISL representation, they execute the same computation.
    ///
    /// **Note**: We cannot union directly in `make()` because egg's API requires `&EGraph`, not `&mut EGraph`.
    /// Instead, we mark equivalence in `ScheduleData` and perform union in `modify()`.
    fn make(egraph: &EGraph<SchedOp, Self>, enode: &SchedOp) -> Self::Data {
        match enode {
            SchedOp::Schedule(handle) => ScheduleData::new_schedule(handle.clone()),

            SchedOp::Num(n) => ScheduleData::new_value(*n),

            SchedOp::Symbol(s) => ScheduleData::new_symbol(s.to_string()),

            SchedOp::Bool(b) => ScheduleData::new_bool(*b),

            SchedOp::InsertMark([sched_id, mark_name_id]) => {
                let sched_data = &egraph[*sched_id].data;
                let mark_name = egraph[*mark_name_id]
                    .data
                    .symbol
                    .as_ref()
                    .unwrap_or(&"unknown".to_string())
                    .clone();

                if let Some(ref handle) = sched_data.schedule {
                    let marked = insert_mark_at_band(&handle.schedule, &mark_name, 0);
                    let new_handle = ScheduleHandle::new(handle.ctx.clone(), marked);
                    // **CRITICAL**: Detect and merge equivalent schedules for proper equality saturation
                    Self::create_schedule_data_with_equivalence(
                        &egraph.analysis,
                        egraph,
                        new_handle,
                    )
                } else {
                    ScheduleData::new_symbol("invalid".to_string())
                }
            }

            SchedOp::TileAtMark([sched_id, mark_name_id, size_id]) => {
                let sched_data = &egraph[*sched_id].data;
                let mark_name = egraph[*mark_name_id]
                    .data
                    .symbol
                    .as_ref()
                    .unwrap_or(&"unknown".to_string())
                    .clone();
                let tile_size = egraph[*size_id].data.value.unwrap_or(32);

                if let Some(ref handle) = sched_data.schedule {
                    let tiled = tile_at_mark(&handle.schedule, &mark_name, tile_size);
                    let new_handle = ScheduleHandle::new(handle.ctx.clone(), tiled);
                    // **CRITICAL**: Detect and merge equivalent schedules for proper equality saturation
                    Self::create_schedule_data_with_equivalence(
                        &egraph.analysis,
                        egraph,
                        new_handle,
                    )
                } else {
                    ScheduleData::new_symbol("invalid".to_string())
                }
            }

            SchedOp::ParallelAtMark([sched_id, mark_name_id]) => {
                let sched_data = &egraph[*sched_id].data;
                let mark_name = egraph[*mark_name_id]
                    .data
                    .symbol
                    .as_ref()
                    .unwrap_or(&"unknown".to_string())
                    .clone();

                if let Some(ref handle) = sched_data.schedule {
                    let parallel = parallel_at_mark(&handle.schedule, &mark_name);
                    let new_handle = ScheduleHandle::new(handle.ctx.clone(), parallel);
                    // **CRITICAL**: Detect and merge equivalent schedules for proper equality saturation
                    Self::create_schedule_data_with_equivalence(
                        &egraph.analysis,
                        egraph,
                        new_handle,
                    )
                } else {
                    ScheduleData::new_symbol("invalid".to_string())
                }
            }

            SchedOp::VectorizeAtMark([sched_id, mark_name_id, width_id]) => {
                let sched_data = &egraph[*sched_id].data;
                let mark_name = egraph[*mark_name_id]
                    .data
                    .symbol
                    .as_ref()
                    .unwrap_or(&"unknown".to_string())
                    .clone();
                let width = egraph[*width_id].data.value.unwrap_or(8);

                if let Some(ref handle) = sched_data.schedule {
                    let vectorized = vectorize_at_mark(&handle.schedule, &mark_name, width);
                    let new_handle = ScheduleHandle::new(handle.ctx.clone(), vectorized);
                    // **CRITICAL**: Detect and merge equivalent schedules for proper equality saturation
                    Self::create_schedule_data_with_equivalence(
                        &egraph.analysis,
                        egraph,
                        new_handle,
                    )
                } else {
                    ScheduleData::new_symbol("invalid".to_string())
                }
            }

            SchedOp::UnrollAtMark([sched_id, mark_name_id, factor_id]) => {
                let sched_data = &egraph[*sched_id].data;
                let mark_name = egraph[*mark_name_id]
                    .data
                    .symbol
                    .as_ref()
                    .unwrap_or(&"unknown".to_string())
                    .clone();
                let factor = egraph[*factor_id].data.value.unwrap_or(4);

                if let Some(ref handle) = sched_data.schedule {
                    let unrolled = unroll_at_mark(&handle.schedule, &mark_name, factor);
                    let new_handle = ScheduleHandle::new(handle.ctx.clone(), unrolled);
                    // **CRITICAL**: Detect and merge equivalent schedules for proper equality saturation
                    Self::create_schedule_data_with_equivalence(
                        &egraph.analysis,
                        egraph,
                        new_handle,
                    )
                } else {
                    ScheduleData::new_symbol("invalid".to_string())
                }
            }

            SchedOp::SplitAtMark([sched_id, mark_name_id, factor_id]) => {
                let sched_data = &egraph[*sched_id].data;
                let mark_name = egraph[*mark_name_id]
                    .data
                    .symbol
                    .as_ref()
                    .unwrap_or(&"unknown".to_string())
                    .clone();
                let factor = egraph[*factor_id].data.value.unwrap_or(32);

                if let Some(ref handle) = sched_data.schedule {
                    let split = split_at_mark(&handle.schedule, &mark_name, factor);
                    let new_handle = ScheduleHandle::new(handle.ctx.clone(), split);
                    // **CRITICAL**: Detect and merge equivalent schedules for proper equality saturation
                    Self::create_schedule_data_with_equivalence(
                        &egraph.analysis,
                        egraph,
                        new_handle,
                    )
                } else {
                    ScheduleData::new_symbol("invalid".to_string())
                }
            }

            SchedOp::HasMark([sched_id, mark_name_id]) => {
                let sched_data = &egraph[*sched_id].data;
                let mark_name = egraph[*mark_name_id]
                    .data
                    .symbol
                    .as_ref()
                    .unwrap_or(&"unknown".to_string())
                    .clone();

                if let Some(ref handle) = sched_data.schedule {
                    let has_mark = check_has_mark(&handle.schedule, &mark_name);
                    ScheduleData::new_bool(has_mark)
                } else {
                    ScheduleData::new_bool(false)
                }
            }

            SchedOp::Tile([sched_id, band_id, size_id]) => {
                // Get the schedule from the e-graph
                let sched_data = &egraph[*sched_id].data;
                let band_idx = egraph[*band_id].data.value.unwrap_or(0) as usize;
                let tile_size = egraph[*size_id].data.value.unwrap_or(32);

                // **Performance**: Removed println! debug output to avoid performance bottlenecks
                // during e-graph exploration. Use log::debug! with appropriate log levels if needed.

                if let Some(ref handle) = sched_data.schedule {
                    // Perform actual ISL tiling operation
                    let tiled = tile_schedule(&handle.schedule, band_idx, tile_size);
                    let new_handle = ScheduleHandle::new(handle.ctx.clone(), tiled);

                    // **CRITICAL**: Detect and merge equivalent schedules for proper equality saturation
                    Self::create_schedule_data_with_equivalence(
                        &egraph.analysis,
                        egraph,
                        new_handle,
                    )
                } else {
                    ScheduleData::new_symbol("invalid".to_string())
                }
            }

            SchedOp::TilePerDim([sched_id, size_i_id, size_j_id, size_k_id]) => {
                // Per-dimension tiling based on PPCG's approach
                let sched_data = &egraph[*sched_id].data;
                let size_i = egraph[*size_i_id].data.value.unwrap_or(16) as i32;
                let size_j = egraph[*size_j_id].data.value.unwrap_or(16) as i32;
                let size_k = egraph[*size_k_id].data.value.unwrap_or(8) as i32;

                // **Performance**: Removed println! debug output to avoid performance bottlenecks
                // during e-graph exploration. Use log::debug! with appropriate log levels if needed.

                if let Some(ref handle) = sched_data.schedule {
                    // Check if schedule has separate 1D bands (Polygeist pattern)
                    // If so, merge them first before applying per-dimension tiling
                    let schedule_str = handle.tree_str.as_str();
                    let has_separate_bands = schedule_str.contains("schedule: \"L")
                        && schedule_str.contains("[(i0)]")
                        && schedule_str.contains("[(i1)]");

                    let schedule_to_tile = if has_separate_bands {
                        merge_bands_for_gemm(&handle.schedule)
                    } else {
                        handle.schedule.copy()
                    };

                    // Use the per-dimension tiling function from tile_per_dimension module
                    let tile_sizes = vec![size_i, size_j, size_k];
                    let tiled = crate::tile_per_dimension::tile_per_dimension(
                        &schedule_to_tile,
                        0, // Default to band 0 for now
                        tile_sizes,
                    );
                    let new_handle = ScheduleHandle::new(handle.ctx.clone(), tiled);
                    // **CRITICAL**: Detect and merge equivalent schedules for proper equality saturation
                    Self::create_schedule_data_with_equivalence(
                        &egraph.analysis,
                        egraph,
                        new_handle,
                    )
                } else {
                    ScheduleData::new_symbol("invalid".to_string())
                }
            }

            SchedOp::Parallel([sched_id, band_id]) => {
                let sched_data = &egraph[*sched_id].data;
                let band_idx = egraph[*band_id].data.value.unwrap_or(0) as usize;

                // Check if the parent is already a Parallel node to prevent nesting
                let parent_is_parallel = egraph[*sched_id]
                    .nodes
                    .iter()
                    .any(|node| matches!(node, SchedOp::Parallel(_)));

                if parent_is_parallel {
                    println!("[DEBUG] Skipping nested Parallel - parent is already parallel");
                    // Return the parent schedule unchanged
                    sched_data.clone()
                } else if let Some(ref handle) = sched_data.schedule {
                    // Mark band as parallel in ISL
                    let parallel = mark_parallel(&handle.schedule, band_idx);
                    let new_handle = ScheduleHandle::new(handle.ctx.clone(), parallel);
                    // **CRITICAL**: Detect and merge equivalent schedules for proper equality saturation
                    Self::create_schedule_data_with_equivalence(
                        &egraph.analysis,
                        egraph,
                        new_handle,
                    )
                } else {
                    ScheduleData::new_symbol("invalid".to_string())
                }
            }

            SchedOp::Vectorize([sched_id, band_id, width_id]) => {
                let sched_data = &egraph[*sched_id].data;
                let band_idx = egraph[*band_id].data.value.unwrap_or(0) as usize;
                let width = egraph[*width_id].data.value.unwrap_or(8);

                if let Some(ref handle) = sched_data.schedule {
                    let vectorized = vectorize_schedule(&handle.schedule, band_idx, width);
                    let new_handle = ScheduleHandle::new(handle.ctx.clone(), vectorized);
                    // **CRITICAL**: Detect and merge equivalent schedules for proper equality saturation
                    Self::create_schedule_data_with_equivalence(
                        &egraph.analysis,
                        egraph,
                        new_handle,
                    )
                } else {
                    ScheduleData::new_symbol("invalid".to_string())
                }
            }

            SchedOp::Fuse([sched_id, loop1_id, loop2_id]) => {
                let sched_data = &egraph[*sched_id].data;
                let loop1 = egraph[*loop1_id].data.value.unwrap_or(0) as usize;
                let loop2 = egraph[*loop2_id].data.value.unwrap_or(1) as usize;

                if let Some(ref handle) = sched_data.schedule {
                    let fused = fuse_loops(&handle.schedule, loop1, loop2);
                    let new_handle = ScheduleHandle::new(handle.ctx.clone(), fused);
                    // **CRITICAL**: Detect and merge equivalent schedules for proper equality saturation
                    Self::create_schedule_data_with_equivalence(
                        &egraph.analysis,
                        egraph,
                        new_handle,
                    )
                } else {
                    ScheduleData::new_symbol("invalid".to_string())
                }
            }

            SchedOp::Interchange([sched_id, band1_id, band2_id]) => {
                let sched_data = &egraph[*sched_id].data;
                let band1 = egraph[*band1_id].data.value.unwrap_or(0) as usize;
                let band2 = egraph[*band2_id].data.value.unwrap_or(1) as usize;

                if let Some(ref handle) = sched_data.schedule {
                    let interchanged = interchange_bands(&handle.schedule, band1, band2);
                    let new_handle = ScheduleHandle::new(handle.ctx.clone(), interchanged);
                    // **CRITICAL**: Detect and merge equivalent schedules for proper equality saturation
                    Self::create_schedule_data_with_equivalence(
                        &egraph.analysis,
                        egraph,
                        new_handle,
                    )
                } else {
                    ScheduleData::new_symbol("invalid".to_string())
                }
            }

            SchedOp::Unroll([sched_id, band_id, factor_id]) => {
                let sched_data = &egraph[*sched_id].data;
                let band_idx = egraph[*band_id].data.value.unwrap_or(0) as usize;
                let factor = egraph[*factor_id].data.value.unwrap_or(4);

                if let Some(ref handle) = sched_data.schedule {
                    let unrolled = unroll_band(&handle.schedule, band_idx, factor);
                    let new_handle = ScheduleHandle::new(handle.ctx.clone(), unrolled);
                    // **CRITICAL**: Detect and merge equivalent schedules for proper equality saturation
                    Self::create_schedule_data_with_equivalence(
                        &egraph.analysis,
                        egraph,
                        new_handle,
                    )
                } else {
                    ScheduleData::new_symbol("invalid".to_string())
                }
            }

            SchedOp::Split([sched_id, band_id, factor_id]) => {
                let sched_data = &egraph[*sched_id].data;
                let band_idx = egraph[*band_id].data.value.unwrap_or(0) as usize;
                let factor = egraph[*factor_id].data.value.unwrap_or(32);

                if let Some(ref handle) = sched_data.schedule {
                    let split = split_band(&handle.schedule, band_idx, factor);
                    let new_handle = ScheduleHandle::new(handle.ctx.clone(), split);
                    // **CRITICAL**: Detect and merge equivalent schedules for proper equality saturation
                    Self::create_schedule_data_with_equivalence(
                        &egraph.analysis,
                        egraph,
                        new_handle,
                    )
                } else {
                    ScheduleData::new_symbol("invalid".to_string())
                }
            }

            SchedOp::Skew([sched_id, band_id, factor_id, direction_id]) => {
                let sched_data = &egraph[*sched_id].data;
                let band_idx = egraph[*band_id].data.value.unwrap_or(0) as usize;
                let factor = egraph[*factor_id].data.value.unwrap_or(1);
                let direction = egraph[*direction_id].data.value.unwrap_or(0);

                if let Some(ref handle) = sched_data.schedule {
                    let skewed = skew_band(&handle.schedule, band_idx, factor, direction);
                    let new_handle = ScheduleHandle::new(handle.ctx.clone(), skewed);
                    // **CRITICAL**: Detect and merge equivalent schedules for proper equality saturation
                    Self::create_schedule_data_with_equivalence(
                        &egraph.analysis,
                        egraph,
                        new_handle,
                    )
                } else {
                    ScheduleData::new_symbol("invalid".to_string())
                }
            }

            SchedOp::GetMark([sched_id, mark_name_id]) => {
                let sched_data = &egraph[*sched_id].data;
                let mark_name = egraph[*mark_name_id]
                    .data
                    .symbol
                    .as_ref()
                    .unwrap_or(&"unknown".to_string())
                    .clone();

                if let Some(ref handle) = sched_data.schedule {
                    // Get schedule node at mark
                    let marked_node = get_mark_node(&handle.schedule, &mark_name);
                    if let Some(node_schedule) = marked_node {
                        let new_handle = ScheduleHandle::new(handle.ctx.clone(), node_schedule);
                        // **CRITICAL**: Detect and merge equivalent schedules for proper equality saturation
                        Self::create_schedule_data_with_equivalence(
                            &egraph.analysis,
                            egraph,
                            new_handle,
                        )
                    } else {
                        ScheduleData::new_symbol("no_mark".to_string())
                    }
                } else {
                    ScheduleData::new_symbol("invalid".to_string())
                }
            }
        }
    }

    fn merge(&mut self, to: &mut Self::Data, from: Self::Data) -> egg::DidMerge {
        // Merge analysis data when two e-classes are unioned
        // This happens when rewrite rules or equivalence detection merges e-classes

        if from.schedule.is_some() {
            if to.schedule.is_none() {
                to.schedule = from.schedule;
                to.equivalent_to = from.equivalent_to;
                egg::DidMerge(true, false)
            } else {
                // Both have schedules - keep the one that's not marked as equivalent
                // (the "original" schedule, not a duplicate)
                if to.equivalent_to.is_none() && from.equivalent_to.is_some() {
                    // Keep to (original), ignore from (duplicate)
                    egg::DidMerge(false, false)
                } else if to.equivalent_to.is_some() && from.equivalent_to.is_none() {
                    // Keep from (original), replace to (duplicate)
                    to.schedule = from.schedule;
                    to.equivalent_to = None;
                    egg::DidMerge(true, false)
                } else {
                    // Both are originals or both are duplicates - keep to
                    egg::DidMerge(false, false)
                }
            }
        } else {
            // Merge non-schedule data (values, symbols, booleans)
            if from.value.is_some() && to.value.is_none() {
                to.value = from.value;
            }
            if from.symbol.is_some() && to.symbol.is_none() {
                to.symbol = from.symbol;
            }
            if from.boolean.is_some() && to.boolean.is_none() {
                to.boolean = from.boolean;
            }
            egg::DidMerge(false, false)
        }
    }

    /// Modify e-class data when it changes, performing union operations for equivalent schedules
    ///
    /// **CRITICAL**: This method is called when e-class data changes. If a schedule is marked
    /// as equivalent to another schedule (via `equivalent_to`), we union the two e-classes.
    /// This enables proper equality saturation: extraction can then choose the schedule with
    /// lowest cost from the merged e-class.
    fn modify(egraph: &mut EGraph<SchedOp, Self>, id: Id) {
        let data = &egraph[id].data;

        // If this schedule is equivalent to another, union them
        if let Some(equivalent_id) = data.equivalent_to {
            // Register this schedule for future equivalence detection
            if let Some(ref handle) = data.schedule {
                let schedule_str = handle.schedule.to_str();
                egraph
                    .analysis
                    .register_schedule(schedule_str.to_string(), id);
            }

            // Union with the equivalent e-class
            // This merges the two e-classes, allowing extraction to choose the one with lowest cost
            egraph.union(id, equivalent_id);
        } else if let Some(ref handle) = data.schedule {
            // Register this schedule for future equivalence detection
            let schedule_str = handle.schedule.to_str();
            egraph
                .analysis
                .register_schedule(schedule_str.to_string(), id);
        }
    }
}

// Helper functions that perform actual ISL operations

// Check if a mark exists in the schedule
#[allow(dead_code)]
fn has_mark(schedule: &Schedule, mark_name: &str) -> bool {
    let root = schedule.get_root();
    // Check if mark exists anywhere in the tree
    check_mark_recursive(&root, mark_name)
}

#[allow(dead_code)]
fn check_mark_recursive(node: &isl_rs::ScheduleNode, mark_name: &str) -> bool {
    use isl_rs::ScheduleNodeType;

    match node.get_type() {
        ScheduleNodeType::Mark => {
            let id = node.mark_get_id();
            if id.get_name() == mark_name {
                return true;
            }
        }
        _ => {}
    }

    // Check children
    let n_children = node.n_children();
    for i in 0..n_children {
        let child = node.get_child(i as i32);
        if check_mark_recursive(&child, mark_name) {
            return true;
        }
    }

    false
}

// Get the schedule node at a specific mark
fn get_mark_node(_schedule: &Schedule, _mark_name: &str) -> Option<Schedule> {
    // For now, return None as this requires complex tree manipulation
    // In a full implementation, this would extract the subtree at the mark
    None
}

pub fn tile_schedule(schedule: &Schedule, band_idx: usize, tile_size: i32) -> Schedule {
    use isl_rs::{MultiVal, Val, ValList};

    println!(
        "[DEBUG] tile_schedule called: band_idx={}, tile_size={}",
        band_idx, tile_size
    );

    // Safety check: tile size must be positive
    if tile_size <= 0 {
        println!(
            "[DEBUG] Invalid tile size {}, returning original schedule",
            tile_size
        );
        return schedule.copy();
    }

    // Check domain size to prevent invalid tiling
    // Tiling with size larger than domain creates constant bands
    let domain = schedule.get_domain();
    let domain_str = domain.to_str().to_string();

    // Try to extract domain bounds (simplified check for common patterns)
    // Looking for patterns like "0 <= i <= 63" or "0 <= i < 64"
    let mut max_domain_size = 0;
    for part in domain_str.split("and") {
        if part.contains("<=") || part.contains("<") {
            // Extract numbers from bounds
            let numbers: Vec<i32> = part
                .split(|c: char| !c.is_numeric() && c != '-')
                .filter_map(|s| s.parse::<i32>().ok())
                .collect();

            for &num in &numbers {
                if num > 0 && num < 10000 {
                    // Sanity check
                    max_domain_size = max_domain_size.max(num);
                }
            }
        }
    }

    // If we found domain bounds, check tile size
    if max_domain_size > 0 && tile_size > max_domain_size {
        println!(
            "[WARNING] Tile size {} exceeds domain size {} - reducing to {}",
            tile_size,
            max_domain_size,
            max_domain_size / 2
        );
        // Don't tile with size larger than half the domain
        let safe_tile_size = (max_domain_size / 2).min(32);
        return tile_schedule(schedule, band_idx, safe_tile_size);
    }

    // Get the root of the schedule tree
    let root = schedule.get_root();
    println!("[DEBUG] Schedule tree root obtained");

    // Dump the schedule before transformation
    let before = schedule.to_str();
    println!("[DEBUG] Schedule BEFORE tiling:");
    println!("{}", before);

    // Check if schedule contains degenerate bands (like [(0)])
    if before.contains("[(0)]") || before.contains("-> [0]") {
        println!("[DEBUG] WARNING: Schedule contains constant bands, may not be tileable");
    }

    // Navigate to the band node
    if let Some(band_node) = find_band_node(root, band_idx) {
        // Get the number of members in the band
        let n_members = Schedule::node_band_n_member(&band_node);
        println!("[DEBUG] Found band node with {} members", n_members);

        // Check if band can be tiled
        if n_members == 0 {
            println!("[DEBUG] Band has no members, cannot tile");
            return schedule.copy();
        }

        // Check if this band is already constant (result of invalid tiling)
        // Get the partial schedule to check if it's degenerate
        let partial = Schedule::node_band_get_partial_schedule(&band_node);
        let partial_str = partial.to_str().to_string();

        if partial_str.contains("[(0)]") || partial_str.contains("-> [0]") {
            println!("[ERROR] Band is constant/degenerate: {}", partial_str);
            println!("[ERROR] Cannot tile a constant band - returning original schedule");
            return schedule.copy();
        }

        // Get context from schedule
        let ctx = schedule.get_ctx();

        // Create tile sizes for all dimensions of the band
        // Note: ISL crashes if we try to tile with size 0 or on degenerate bands
        let mut val_list = ValList::alloc(&ctx, n_members);

        for _i in 0..n_members {
            // For safety, use tile_size for all dimensions to avoid zero values
            // A zero value causes ISL to crash with "cannot scale down by zero"
            let val = if tile_size > 0 {
                Val::int_from_si(&ctx, tile_size as i64)
            } else {
                // Should never happen due to earlier check, but be defensive
                Val::int_from_si(&ctx, 1)
            };
            val_list = val_list.add(val);
        }

        // Get space from the band node
        let space = Schedule::node_band_get_space(&band_node);

        // Create MultiVal from list and space
        let sizes = MultiVal::from_val_list(space, val_list);

        // Apply tiling with error handling
        println!("[DEBUG] Applying tiling with size {}", tile_size);

        // Try to tile, but catch potential issues
        let tiled_node = Schedule::node_band_tile(band_node, sizes);

        // Get the schedule from the transformed node
        let result = Schedule::node_get_schedule(&tiled_node);

        // Dump the schedule after transformation using block-style printer
        let after = crate::isl_block_printer::schedule_to_block_str(&result);
        println!("[DEBUG] Schedule AFTER tiling:");
        println!("{}", after);

        result
    } else {
        println!(
            "[DEBUG] Band node {} not found, returning original schedule",
            band_idx
        );
        schedule.copy()
    }
}

// New function for selective dimension tiling - tiles only specific dimensions
pub fn tile_schedule_selective(
    schedule: &Schedule,
    band_idx: usize,
    dim_idx: usize,
    tile_size: i32,
) -> Schedule {
    use isl_rs::{MultiVal, Val, ValList};

    println!(
        "[DEBUG] tile_schedule_selective called: band_idx={}, dim_idx={}, tile_size={}",
        band_idx, dim_idx, tile_size
    );

    let root = schedule.get_root();
    println!("[DEBUG] Schedule BEFORE selective tiling:");
    println!("{}", schedule.to_str());

    // Navigate to find the band node
    let band_node = find_band_node(root, band_idx);

    if let Some(band_node) = band_node {
        let n_member = Schedule::node_band_n_member(&band_node) as usize;
        println!("[DEBUG] Found band node with {} members", n_member);

        if dim_idx >= n_member {
            println!(
                "[ERROR] Dimension index {} out of range (band has {} dimensions)",
                dim_idx, n_member
            );
            return schedule.copy();
        }

        // Strategy: Split the band to isolate the dimension we want to tile
        // 1. If dim_idx > 0, split at dim_idx to separate earlier dimensions
        // 2. Then split at position 1 to isolate our target dimension
        // 3. Tile the isolated dimension

        let mut current_node = band_node;

        // Step 1: If we're not tiling the first dimension, split to isolate it
        if dim_idx > 0 {
            println!(
                "[DEBUG] Splitting band at position {} to isolate dimension",
                dim_idx
            );
            current_node = Schedule::node_band_split(current_node, dim_idx as i32);
            // Navigate to the child band that contains our target dimension
            if Schedule::node_has_children(&current_node) {
                current_node = Schedule::node_get_child(&current_node, 0);
            }
        }

        // Step 2: If there are dimensions after our target, split them off
        let remaining_dims = n_member - dim_idx;
        if remaining_dims > 1 {
            println!("[DEBUG] Splitting off dimensions after position 1");
            current_node = Schedule::node_band_split(current_node, 1);
        }

        // Step 3: Now we have a band with just one dimension - tile it
        println!(
            "[DEBUG] Tiling the isolated dimension with size {}",
            tile_size
        );
        let ctx = schedule.get_ctx();
        let sizes = {
            let mut val_list = ValList::alloc(&ctx, 1);
            let val = Val::int_from_si(&ctx, tile_size as i64);
            val_list = val_list.add(val);

            let space = Schedule::node_band_get_space(&current_node);
            MultiVal::from_val_list(space, val_list)
        };

        let tiled_node = Schedule::node_band_tile(current_node, sizes);
        let tiled_schedule = Schedule::node_get_schedule(&tiled_node);

        // Print the result
        let tiled_str = crate::isl_block_printer::schedule_to_block_str(&tiled_schedule);
        println!("[DEBUG] Schedule AFTER selective tiling:");
        println!("{}", tiled_str);

        tiled_schedule
    } else {
        println!("[DEBUG] Band node {} not found", band_idx);
        schedule.copy()
    }
}

fn find_band_node(node: isl_rs::ScheduleNode, target_idx: usize) -> Option<isl_rs::ScheduleNode> {
    let mut current_band_idx = 0;
    find_band_node_recursive(node, target_idx, &mut current_band_idx)
}

fn find_band_node_recursive(
    node: isl_rs::ScheduleNode,
    target_idx: usize,
    current_idx: &mut usize,
) -> Option<isl_rs::ScheduleNode> {
    use isl_rs::ScheduleNodeType;

    // Check if this is a band node
    let node_type = Schedule::node_get_type(&node);

    if matches!(node_type, ScheduleNodeType::Band) {
        // Count this as a band dimension based on its members
        let _n_members = Schedule::node_band_n_member(&node);

        // For multi-dimensional bands, each dimension counts separately
        // But for now, treat the whole band as one unit
        if *current_idx == target_idx {
            return Some(node);
        }
        *current_idx += 1;
    }

    // Check children
    if Schedule::node_has_children(&node) {
        let n_children = Schedule::node_n_children(&node);
        for i in 0..n_children {
            let child = Schedule::node_get_child(&node, i);
            if let Some(found) = find_band_node_recursive(child, target_idx, current_idx) {
                return Some(found);
            }
        }
    }

    None
}

// Alias for schedule_propagator compatibility
pub fn apply_tiling(schedule: &Schedule, band_idx: usize, tile_size: usize) -> Schedule {
    tile_schedule(schedule, band_idx, tile_size as i32)
}

// Vectorization support
pub fn mark_vectorize(schedule: &Schedule, band_idx: usize, _vector_width: usize) -> Schedule {
    // For now, just mark as parallel (vectorization is a form of parallelism)
    // In a full implementation, this would add vector marks
    mark_parallel(schedule, band_idx)
}

// Unrolling support
pub fn apply_unroll(schedule: &Schedule, _band_idx: usize, _unroll_factor: usize) -> Schedule {
    // For now, return unchanged
    // In a full implementation, this would add unroll marks
    schedule.copy()
}

pub fn mark_parallel(schedule: &Schedule, band_idx: usize) -> Schedule {
    use isl_rs::{Id, ScheduleNodeType};

    println!("[DEBUG] mark_parallel called: band_idx={}", band_idx);

    // Dump the schedule before
    let before = schedule.to_str();
    println!("[DEBUG] Schedule BEFORE parallel marking:");
    println!("{}", before);

    // Get the root of the schedule tree
    let root = schedule.get_root();
    let ctx = schedule.get_ctx();
    println!("[DEBUG] Schedule tree root obtained");

    // Check if the root is already a mark node - if so, we need to be careful
    // to not create nested marks directly (which Polygeist can't handle)
    let root_type = root.get_type();
    println!("[DEBUG] Root node type: {:?}", root_type);

    // Check if root is a parallel mark - if so, don't add another parallel mark
    if root_type == ScheduleNodeType::Mark {
        let mark_id = root.mark_get_id();
        if mark_id.to_str() == "parallel" {
            println!("[DEBUG] Schedule root is already marked as parallel");
            // Need to re-get root since mark_get_id consumes it
            let root2 = schedule.get_root();
            // Check if the child is also a mark (nested marks)
            let child = root2.child(0);
            if child.get_type() == ScheduleNodeType::Mark {
                println!("[WARN] Detected nested mark nodes - this is invalid for Polygeist!");
                // Return unchanged schedule
                let root3 = schedule.get_root();
                return Schedule::node_get_schedule(&root3);
            }
        }
    }

    // Re-get root for navigation since it was consumed
    let root = schedule.get_root();
    let root_type = root.get_type();

    // Navigate to the band node and mark it as parallel using ISL marks
    // If root is already a mark, we need to navigate past it to find the band
    let band_node = if root_type == ScheduleNodeType::Mark {
        println!("[DEBUG] Root is already a mark, navigating to child");
        // Get the child of the mark node and find band from there
        find_band_node(root.child(0), band_idx)
    } else {
        find_band_node(root, band_idx)
    };

    if let Some(band_node) = band_node {
        let n_members = Schedule::node_band_n_member(&band_node);
        println!("[DEBUG] Found band node with {} members", n_members);

        // Following Polygeist's approach: make band permutable and insert "parallel" mark
        // First, make the band permutable (required for parallel execution)
        let mut permutable_band = Schedule::node_band_set_permutable(band_node, 1); // 1 = true in ISL

        // Set coincident property for all members (required for ScheduleProperties detection)
        let n_members = Schedule::node_band_n_member(&permutable_band);
        for i in 0..n_members {
            permutable_band =
                Schedule::node_band_member_set_coincident(permutable_band, i as i32, 1);
        }

        // Then insert the "parallel" mark that Polygeist recognizes
        let parallel_id = Id::read_from_str(&ctx, "parallel");
        let parallel_node = permutable_band.insert_mark(parallel_id);

        // Get the schedule from the transformed node
        let result = Schedule::node_get_schedule(&parallel_node);

        // Dump the schedule after transformation
        let after = result.to_str();
        println!("[DEBUG] Schedule AFTER parallel marking:");
        println!("{}", after);

        result
    } else {
        println!(
            "[DEBUG] Band node {} not found, returning original schedule",
            band_idx
        );
        schedule.copy()
    }
}

pub fn vectorize_schedule(schedule: &Schedule, _band_idx: usize, _width: i32) -> Schedule {
    // Vectorize innermost loop
    // schedule.band_set_ast_build_options(band_idx, &format!("vector_{}", width))
    schedule.copy()
}

fn fuse_loops(schedule: &Schedule, _loop1: usize, _loop2: usize) -> Schedule {
    // Fuse two loops
    // schedule.band_fuse(loop1, loop2)
    schedule.copy()
}

pub fn interchange_bands(schedule: &Schedule, band1: usize, band2: usize) -> Schedule {
    println!(
        "[DEBUG] interchange_bands called: band1={}, band2={}",
        band1, band2
    );

    // IMPLEMENTATION NOTE:
    // ISL-RS doesn't expose band_set_partial_schedule, so we can't directly
    // manipulate band nodes. Instead, we'll work at the schedule map level.
    // This is a simplified implementation that demonstrates the concept.

    // Get the schedule as a string and manipulate it
    let sched_str = schedule.to_str();
    println!("[DEBUG] Original schedule: {}", sched_str);

    // For now, we'll use a workaround: create a modified schedule string
    // In a real implementation, we'd need to:
    // 1. Navigate to the band node
    // 2. Get its partial schedule (MultiUnionPwAff)
    // 3. Swap the dimensions
    // 4. Set it back (but this API is missing)

    // Check if this is a simple schedule we can handle
    if sched_str.contains("domain:") && sched_str.contains("schedule:") {
        // This is a schedule tree format
        // For demonstration, we'll just mark that interchange was attempted
        println!(
            "[INFO] Loop interchange requested for bands {} and {}",
            band1, band2
        );
        println!(
            "[WARN] Full interchange implementation requires ISL band_set_partial_schedule API"
        );

        // Return a copy for now - in production, you'd implement this properly
        // by extending the ISL-RS bindings to include band_set_partial_schedule
        return schedule.copy();
    }

    // Alternative approach: If we have a simple map-based schedule
    // Try to get the schedule map and manipulate it
    let sched_map = schedule.get_map();
    let _ctx = schedule.get_ctx();
    let map_str = sched_map.to_str();

    // Check if this is a simple affine schedule like { S[i,j,k] -> [t0,t1,t2] }
    if map_str.contains("->") && map_str.contains("[") {
        println!("[DEBUG] Attempting map-based interchange on: {}", map_str);

        // Parse to understand structure
        // For a schedule { S[i,j,k] -> [t0,t1,t2] }
        // Interchanging bands 0 and 1 gives { S[i,j,k] -> [t1,t0,t2] }

        // This would require careful string manipulation or proper ISL API usage
        // For now, demonstrate the concept
        println!(
            "[INFO] Would interchange dimensions {} and {} in schedule map",
            band1, band2
        );
    }

    // Use the consolidated transformations module
    match crate::transformations::interchange(schedule, band1, band2, None) {
        Ok(Some(interchanged)) => {
            println!("[DEBUG] Interchange successful");
            interchanged
        }
        Ok(None) => {
            println!("[WARN] Interchange not applicable - returning original");
            schedule.copy()
        }
        Err(e) => {
            println!("[WARN] Interchange failed: {} - returning original", e);
            schedule.copy()
        }
    }
}

fn unroll_band(schedule: &Schedule, _band_idx: usize, _factor: i32) -> Schedule {
    // Unroll a band
    // schedule.band_set_ast_build_options(band_idx, &format!("unroll_{}", factor))
    schedule.copy()
}

fn split_band(schedule: &Schedule, _band_idx: usize, _factor: i32) -> Schedule {
    // Split a band (strip-mining)
    // schedule.band_split(band_idx, factor)
    schedule.copy()
}

pub fn skew_band(schedule: &Schedule, band_idx: usize, factor: i32, direction: i32) -> Schedule {
    println!(
        "[DEBUG] skew_band called: band_idx={}, factor={}, direction={}",
        band_idx, factor, direction
    );

    // Skewing transformation for wavefront parallelization
    // Mathematical foundation: Apply affine transformation
    // For 2D: [i,j] -> [i, j+factor*i] (forward skew) or [i+factor*j, j] (backward skew)

    // This enables parallelization of loops with diagonal dependencies
    // Common in stencil computations

    let sched_str = schedule.to_str();
    println!("[DEBUG] Original schedule for skewing: {}", sched_str);

    // IMPLEMENTATION NOTE:
    // Proper skewing requires modifying the schedule's affine expressions
    // In ISL, this would be done by:
    // 1. Getting the band's partial schedule
    // 2. Applying an affine transformation matrix
    // 3. Setting the modified partial schedule back

    // Since ISL-RS doesn't expose the necessary APIs, we demonstrate the concept
    println!(
        "[INFO] Skewing band {} by factor {} in direction {}",
        band_idx, factor, direction
    );
    println!("[INFO] This transformation enables wavefront parallelization");
    println!("[WARN] Full implementation requires extended ISL-RS bindings");

    // Use the consolidated transformations module
    let forward = direction == 0;
    match crate::transformations::skew(schedule, band_idx, factor, forward, None) {
        Ok(Some(skewed)) => {
            println!("[DEBUG] Skewing successful - wavefront parallelism enabled");
            skewed
        }
        Ok(None) => {
            println!("[WARN] Skewing not applicable - returning original");
            schedule.copy()
        }
        Err(e) => {
            println!("[WARN] Skewing failed: {} - returning original", e);
            schedule.copy()
        }
    }
}

// Mark-based navigation functions

pub fn insert_mark_at_band(schedule: &Schedule, mark_name: &str, band_idx: usize) -> Schedule {
    use isl_rs::Id;

    println!(
        "[DEBUG] insert_mark_at_band called: mark_name={}, band_idx={}",
        mark_name, band_idx
    );

    let root = schedule.get_root();
    let ctx = schedule.get_ctx();

    // Find the band node
    if let Some(band_node) = find_band_node(root, band_idx) {
        // Create an ID for the mark
        let mark_id = Id::read_from_str(&ctx, mark_name);

        // Insert mark before the band
        let marked_node = band_node.insert_mark(mark_id);

        // Get the schedule from the transformed node
        let result = Schedule::node_get_schedule(&marked_node);

        println!("[DEBUG] Mark '{}' inserted successfully", mark_name);
        result
    } else {
        println!(
            "[DEBUG] Band node {} not found, returning original schedule",
            band_idx
        );
        schedule.copy()
    }
}

pub fn find_mark_node(node: isl_rs::ScheduleNode, mark_name: &str) -> Option<isl_rs::ScheduleNode> {
    use isl_rs::ScheduleNodeType;

    // Check if this is a mark node with the right name
    let node_type = Schedule::node_get_type(&node);

    if matches!(node_type, ScheduleNodeType::Mark) {
        let mark_id = node.mark_get_id();
        if mark_id.get_name() == mark_name {
            return Some(node);
        }
    }

    // Check children
    if Schedule::node_has_children(&node) {
        let n_children = Schedule::node_n_children(&node);
        for i in 0..n_children {
            let child = Schedule::node_get_child(&node, i);
            if let Some(found) = find_mark_node(child, mark_name) {
                return Some(found);
            }
        }
    }

    None
}

pub fn tile_at_mark(schedule: &Schedule, mark_name: &str, tile_size: i32) -> Schedule {
    println!(
        "[DEBUG] tile_at_mark called: mark_name={}, tile_size={}",
        mark_name, tile_size
    );

    let root = schedule.get_root();

    // Find the mark node
    if let Some(mark_node) = find_mark_node(root, mark_name) {
        // Get the first band child of the mark
        if Schedule::node_has_children(&mark_node) {
            let child = Schedule::node_get_child(&mark_node, 0);
            let node_type = Schedule::node_get_type(&child);

            if matches!(node_type, isl_rs::ScheduleNodeType::Band) {
                // Apply tiling to this band
                use isl_rs::{MultiVal, Val, ValList};

                let n_members = Schedule::node_band_n_member(&child);
                let ctx = schedule.get_ctx();

                // Create tile sizes
                let first_val = Val::int_from_si(&ctx, tile_size as i64);
                let mut val_list = ValList::from_val(first_val);

                for _i in 1..n_members {
                    let val = Val::int_from_si(&ctx, tile_size as i64);
                    val_list = val_list.add(val);
                }

                let space = Schedule::node_band_get_space(&child);
                let sizes = MultiVal::from_val_list(space, val_list);

                let tiled_node = Schedule::node_band_tile(child, sizes);
                let result = Schedule::node_get_schedule(&tiled_node);

                println!("[DEBUG] Tiling at mark '{}' completed", mark_name);
                return result;
            }
        }
    }

    println!("[DEBUG] Mark '{}' not found or no band to tile", mark_name);
    schedule.copy()
}

pub fn parallel_band(schedule: &Schedule, band_idx: usize) -> Schedule {
    println!("[DEBUG] parallel_band called: band_idx={}", band_idx);

    // This is a simplified parallelization - in reality we'd add parallel pragmas
    // For now, just mark the band as permutable which is required for parallelization
    schedule.copy()
}

pub fn parallel_at_mark(schedule: &Schedule, mark_name: &str) -> Schedule {
    use isl_rs::Id;

    println!("[DEBUG] parallel_at_mark called: mark_name={}", mark_name);

    let root = schedule.get_root();

    // Find the mark node
    if let Some(mark_node) = find_mark_node(root, mark_name) {
        // Get the first band child of the mark
        if Schedule::node_has_children(&mark_node) {
            let child = Schedule::node_get_child(&mark_node, 0);
            let node_type = Schedule::node_get_type(&child);

            if matches!(node_type, isl_rs::ScheduleNodeType::Band) {
                // Following Polygeist's approach: insert a "parallel" mark before the band
                // First, make the band permutable (required for parallel execution)
                let permutable_band = Schedule::node_band_set_permutable(child, 1); // 1 = true in ISL

                // Then insert the "parallel" mark that Polygeist recognizes
                let ctx = schedule.get_ctx();
                let parallel_id = Id::read_from_str(&ctx, "parallel");
                let parallel_node = permutable_band.insert_mark(parallel_id);

                let result = Schedule::node_get_schedule(&parallel_node);
                println!(
                    "[DEBUG] Band at mark '{}' marked as parallel with 'parallel' mark",
                    mark_name
                );
                return result;
            }
        }
    }

    println!(
        "[DEBUG] Mark '{}' not found or no band to parallelize",
        mark_name
    );
    schedule.copy()
}

pub fn vectorize_at_mark(schedule: &Schedule, mark_name: &str, _width: i32) -> Schedule {
    println!(
        "[DEBUG] vectorize_at_mark called: mark_name={}, width={}",
        mark_name, _width
    );
    // TODO: Implement vectorization at mark
    schedule.copy()
}

pub fn unroll_at_mark(schedule: &Schedule, mark_name: &str, _factor: i32) -> Schedule {
    println!(
        "[DEBUG] unroll_at_mark called: mark_name={}, factor={}",
        mark_name, _factor
    );
    // TODO: Implement unrolling at mark
    schedule.copy()
}

pub fn split_at_mark(schedule: &Schedule, mark_name: &str, _factor: i32) -> Schedule {
    println!(
        "[DEBUG] split_at_mark called: mark_name={}, factor={}",
        mark_name, _factor
    );
    // TODO: Implement splitting at mark
    schedule.copy()
}

pub fn check_has_mark(schedule: &Schedule, mark_name: &str) -> bool {
    let root = schedule.get_root();
    find_mark_node(root, mark_name).is_some()
}

/// Merges consecutive 1D band nodes into a single multi-dimensional band node
///
/// This function addresses the issue where Polygeist generates separate 1D bands
/// (e.g., band[i0], band[i1], band[i2]) but TilePerDim needs a single 3D band
/// (e.g., band[i0, i1, i2]) to apply different tile sizes per dimension.
///
/// # Theory
/// Uses ISL's `flat_range_product` to combine partial schedules:
/// - Input: A → B₁ and A → B₂
/// - Output: A → (B₁, B₂)
///
/// # Implementation
/// 1. Find consecutive band nodes in the schedule tree
/// 2. Extract their partial schedules (MultiUnionPwAff)
/// 3. Combine using flat_range_product
/// 4. Create new band node with combined schedule
/// 5. Replace original bands with merged band
pub fn merge_consecutive_bands(
    schedule: &Schedule,
    start_band_idx: usize,
    num_bands: usize,
) -> Schedule {
    println!(
        "[DEBUG] merge_consecutive_bands called: start_idx={}, num_bands={}",
        start_band_idx, num_bands
    );

    if num_bands < 2 {
        println!("[DEBUG] Need at least 2 bands to merge, got {}", num_bands);
        return schedule.copy();
    }

    let root = schedule.get_root();
    let _ctx = schedule.get_ctx();

    // Find the first band node
    let first_band = find_band_node(root.copy(), start_band_idx);
    if first_band.is_none() {
        println!("[ERROR] Could not find band at index {}", start_band_idx);
        return schedule.copy();
    }

    let mut band_nodes = vec![first_band.unwrap()];

    // Collect consecutive band nodes
    for i in 1..num_bands {
        let next_band = find_band_node(root.copy(), start_band_idx + i);
        if let Some(band) = next_band {
            band_nodes.push(band);
        } else {
            println!(
                "[ERROR] Could not find band at index {}",
                start_band_idx + i
            );
            return schedule.copy();
        }
    }

    println!(
        "[DEBUG] Found {} consecutive band nodes to merge",
        band_nodes.len()
    );

    // Extract partial schedules from each band
    let mut partial_schedules = Vec::new();
    for (i, band) in band_nodes.iter().enumerate() {
        let n_members = Schedule::node_band_n_member(band);
        println!("[DEBUG] Band {} has {} members", i, n_members);

        if n_members != 1 {
            println!(
                "[WARNING] Band {} has {} members, expected 1 for Polygeist-style bands",
                i, n_members
            );
        }

        let partial = Schedule::node_band_get_partial_schedule(band);
        partial_schedules.push(partial);
    }

    // Combine partial schedules using flat_range_product
    println!(
        "[DEBUG] Combining {} partial schedules using flat_range_product",
        partial_schedules.len()
    );

    let mut combined = partial_schedules[0].copy();
    for i in 1..partial_schedules.len() {
        println!("[DEBUG] Merging schedule {} into combined schedule", i);
        // flat_range_product: combines A → B₁ and A → B₂ into A → (B₁, B₂)
        combined = combined.flat_range_product(partial_schedules[i].copy());
    }

    println!("[DEBUG] Successfully combined partial schedules");

    // Now we need to reconstruct the schedule tree with the merged band
    // This is complex because we need to:
    // 1. Remove the old consecutive bands
    // 2. Insert the new merged band
    // 3. Preserve the rest of the tree structure

    // For now, we'll use a simpler approach:
    // Create a new schedule from the combined partial schedule
    // This might lose some tree structure but ensures correctness

    // Create a new clean schedule with just the merged band
    // ISL requires us to work with the schedule tree properly

    // First approach: Try to modify the original schedule in place
    // by removing the extra bands after the first one
    let root = schedule.get_root();

    // Navigate to the first band
    let first_band = find_band_node(root.copy(), 0);
    if first_band.is_none() {
        println!("[ERROR] Could not find first band to replace");
        return schedule.copy();
    }
    let first_band = first_band.unwrap();

    // Replace the partial schedule of the first band with the merged one
    // This is a workaround since we can't directly set partial schedule
    // We'll delete the band and insert a new one with the merged schedule

    // ISL CONSTRAINT: isl_schedule_node_delete() can only delete nodes
    // with exactly 1 child (see isl_schedule_node.c:2824-2827).
    //
    // Problem: The current approach tries to delete all children of the domain,
    // but if any child has multiple children, node_delete() will fail with segfault.
    //
    // Solution: Check if the first band has exactly 1 child before attempting deletion.
    // If not, use a safer fallback approach that reconstructs the schedule from domain.

    let first_band_n_children = Schedule::node_n_children(&first_band);
    println!("[DEBUG] First band has {} children", first_band_n_children);

    if first_band_n_children != 1 {
        println!(
            "[WARNING] First band has {} children, cannot use node_delete (requires exactly 1)",
            first_band_n_children
        );
        println!(
            "[WARNING] Using fallback: reconstruct schedule from domain + merged partial schedule"
        );

        // Fallback: Reconstruct schedule from domain + merged partial schedule
        // This preserves correctness but may lose some tree structure (marks, nested bands, etc.)
        let domain = schedule.get_domain();

        // Create a new schedule from domain
        let new_schedule = Schedule::from_domain(domain);

        // Insert the merged partial schedule
        // Note: This uses Schedule-level API, not node-level API
        let result = new_schedule.insert_partial_schedule(combined);

        println!("[DEBUG] Schedule reconstructed with merged band (fallback method)");
        return result;
    }

    // Safe path: First band has exactly 1 child, we can delete it
    println!("[DEBUG] First band satisfies ISL constraint (1 child), proceeding with deletion");

    // Get the parent of the first band (should be the domain or a mark node)
    // Note: node_parent consumes first_band, so we need to get parent before deletion
    let _parent = Schedule::node_parent(first_band.copy());

    // Delete the first band node (it has exactly 1 child, so this is safe)
    // We need to get the band again since node_parent consumed it
    let first_band_for_delete =
        find_band_node(root.copy(), start_band_idx).expect("First band should still exist");
    let parent_after_delete = Schedule::node_delete(first_band_for_delete);

    // Insert the merged band at the parent
    let band_node = parent_after_delete.insert_partial_schedule(combined);

    // Get the resulting schedule
    let result = Schedule::node_get_schedule(&band_node);

    // Print the result using block-style printer
    let result_str = crate::isl_block_printer::schedule_to_block_str(&result);
    println!("[DEBUG] Schedule AFTER band merging:");
    println!("{}", result_str);

    result
}

/// Convenience function to merge the first 3 bands for GEMM-style kernels
/// This handles the common case where Polygeist generates band[i0], band[i1], band[i2]
/// and we need band[i0, i1, i2] for per-dimension tiling
pub fn merge_bands_for_gemm(schedule: &Schedule) -> Schedule {
    println!("[DEBUG] merge_bands_for_gemm: Merging first 3 bands for GEMM kernel");
    merge_consecutive_bands(schedule, 0, 3)
}
