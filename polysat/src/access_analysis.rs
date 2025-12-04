//! Access analysis module for PolySat
//! Provides data structures and functionality for tracking memory access patterns
//! and dependencies in polyhedral schedules.

use isl_rs::UnionMap;
use std::collections::HashMap;
use std::sync::Arc;

// ============================================================================
// Validation Diagnostics
// ============================================================================

/// Validation warnings collected during access pattern analysis
///
/// These warnings help diagnose incomplete or problematic access information,
/// particularly useful for cost models to make informed decisions when
/// volume computation or dependency analysis fails.
///
/// **Rationale**:
/// - Preserves complete ISL relation strings for maximum information
/// - Allows cost models to distinguish between "missing data" vs "parse failure"
/// - Enables better error messages and fallback strategies
///
/// **Example**:
/// ```rust
/// use polysat::access_analysis::{AccessInfo, ValidationWarning, ContextHandle, ScheduleHandle};
///
/// let access_info = AccessInfo::new(ContextHandle::new_placeholder(), ScheduleHandle::new_placeholder()); // Dummy
/// for warning in &access_info.validation_warnings {
///     match warning {
///         ValidationWarning::WriteOnlyArray { array_name, .. } => {
///             println!("Warning: Array {} is written but never read", array_name);
///         }
///         _ => {}
///     }
/// }
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ValidationWarning {
    /// Array is written but never read (potential missing reduction read)
    ///
    /// **Common Cause**: GEMM-style reduction where accumulator array (e.g., C[i,j])
    /// is missing from the reads set. This breaks ISL flow analysis as it cannot
    /// detect the cross-iteration dependency.
    ///
    /// **Example**: `C[i,j] += A[i,k] * B[k,j]` should have C in both reads and writes.
    WriteOnlyArray {
        /// Name of the array (e.g., "C")
        array_name: String,

        /// Complete ISL write relation for diagnostic purposes
        /// Example: "{ S0[i, j, k] -> C[i, j] }"
        write_relation: String,
    },

    /// Array is read but never written (read-only, usually expected)
    ///
    /// This is typically **not a problem** (e.g., input arrays like A, B in GEMM),
    /// but included for completeness and to help validate access info integrity.
    ReadOnlyArray {
        /// Name of the array (e.g., "A", "B")
        array_name: String,

        /// Complete ISL read relation
        read_relation: String,
    },

    /// Read-modify-write pattern detected (reduction/accumulation)
    ///
    /// This is a **positive validation**: indicates a legitimate reduction pattern.
    /// Cost models can use this to identify kernels that require accumulator
    /// communication and special scheduling constraints.
    ///
    /// **Example**: GEMM reduction on C[i,j], stencil updates
    ReadModifyWrite {
        /// Name of the array with RMW pattern (e.g., "C")
        array_name: String,

        /// Complete ISL read relation
        reads_relation: String,

        /// Complete ISL write relation
        writes_relation: String,
    },

    /// Empty access maps detected
    ///
    /// **Critical Issue**: Either reads or writes (or both) are completely empty,
    /// making dependency analysis impossible. This usually indicates a problem
    /// in the Polymer extraction or manual specification.
    EmptyAccessMaps {
        /// true if reads are empty
        reads_empty: bool,

        /// true if writes are empty
        writes_empty: bool,
    },
}

impl ValidationWarning {
    /// Get a human-readable description of the warning
    pub fn description(&self) -> String {
        match self {
            ValidationWarning::WriteOnlyArray { array_name, .. } => {
                format!(
                    "Array '{}' is write-only (potential missing reduction read)",
                    array_name
                )
            }
            ValidationWarning::ReadOnlyArray { array_name, .. } => {
                format!(
                    "Array '{}' is read-only (expected for input arrays)",
                    array_name
                )
            }
            ValidationWarning::ReadModifyWrite { array_name, .. } => {
                format!(
                    "Array '{}' has read-modify-write pattern (reduction detected)",
                    array_name
                )
            }
            ValidationWarning::EmptyAccessMaps {
                reads_empty,
                writes_empty,
            } => match (reads_empty, writes_empty) {
                (true, true) => "Both reads and writes are empty".to_string(),
                (true, false) => "Reads are empty (write-only kernel)".to_string(),
                (false, true) => "Writes are empty (read-only kernel)".to_string(),
                (false, false) => unreachable!("EmptyAccessMaps with both non-empty"),
            },
        }
    }

    /// Get the severity level of this warning
    pub fn severity(&self) -> ValidationSeverity {
        match self {
            ValidationWarning::EmptyAccessMaps { .. } => ValidationSeverity::Error,
            ValidationWarning::WriteOnlyArray { .. } => ValidationSeverity::Warning,
            ValidationWarning::ReadOnlyArray { .. } => ValidationSeverity::Info,
            ValidationWarning::ReadModifyWrite { .. } => ValidationSeverity::Info,
        }
    }
}

/// Severity level for validation warnings
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ValidationSeverity {
    /// Informational message (no action needed)
    Info,

    /// Warning (may affect accuracy, but not fatal)
    Warning,

    /// Error (likely to cause analysis failure)
    Error,
}

/// Core structure representing access information for a polyhedral program
///
/// **Integration with ISL Flow Analysis**:
/// This structure now supports storing actual ISL UnionMaps for precise dependency analysis.
/// When `reads_union_map` and `writes_union_map` are populated (e.g., from Polymer access files),
/// the dependency analysis can use ISL's exact flow analysis instead of pattern matching.
///
/// **Analysis Order**:
/// 1. **Exact Flow Analysis**: Use `reads_union_map` + `writes_union_map` -> ISL flow analysis
/// 2. **Pattern Synthesis**: Pattern-based synthesis from schedule string -> ISL deltas
/// 3. **Conservative Fallback**: Conservative analysis based on `stmt_accesses`
#[derive(Clone)]
pub struct AccessInfo {
    /// The ISL schedule tree (opaque handle)
    pub schedule: ScheduleHandle,

    /// Per-statement access information
    pub stmt_accesses: HashMap<String, StmtAccess>,

    /// Array information (dimensions, types, etc.)
    pub arrays: HashMap<String, ArrayInfo>,

    /// ISL context for operations (opaque handle)
    pub ctx: ContextHandle,

    // ========================================================================
    // Integration: Actual ISL UnionMaps for Precise Flow Analysis
    // ========================================================================
    /// Actual ISL read access UnionMap
    ///
    /// When available (e.g., loaded from Polymer's `*_accesses.reads` file),
    /// this enables precise ISL flow analysis via `UnionAccessInfo::compute_flow()`.
    ///
    /// **Format**: `{ S[i,j] -> A[i,k]; S[i,j] -> B[k,j] }`
    ///
    /// **None** when:
    /// - No Polymer access files available
    /// - Using pattern synthesis (Tier-1)
    /// - ISL-only mode without access info
    ///
    /// **Arc Wrapper**: ISL objects don't implement Clone, so we use Arc for zero-cost sharing.
    pub reads_union_map: Option<Arc<UnionMap>>,

    /// Actual ISL write access UnionMap
    ///
    /// When available (e.g., loaded from Polymer's `*_accesses.writes` file),
    /// this enables precise ISL flow analysis.
    ///
    /// **Format**: `{ S[i,j] -> C[i,j] }`
    ///
    /// **None** when not using Tier-2 analysis.
    ///
    /// **Arc Wrapper**: ISL objects don't implement Clone, so we use Arc for zero-cost sharing.
    pub writes_union_map: Option<Arc<UnionMap>>,

    // ========================================================================
    // INPUT VALIDATION: Diagnostic Information
    // ========================================================================
    /// **Validation warnings** collected during access pattern analysis
    ///
    /// Populated by:
    /// - `validate_access_patterns()` in dependency_aware.rs (Tier-2 path)
    /// - `populate_from_polymer_files()` when loading access data
    ///
    /// **Purpose**:
    /// - Help cost models distinguish between "missing data" vs "parse failure"
    /// - Enable better error messages for users
    /// - Support intelligent fallback strategies (e.g., conservative volume estimates)
    ///
    /// **Usage Example** (in cost model):
    /// ```rust
    /// use polysat::access_analysis::{AccessInfo, ValidationWarning, ContextHandle, ScheduleHandle};
    ///
    /// let access_info = AccessInfo::new(ContextHandle::new_placeholder(), ScheduleHandle::new_placeholder()); // Dummy
    /// let write_only_arrays: Vec<_> = access_info.validation_warnings.iter()
    ///     .filter_map(|w| match w {
    ///         ValidationWarning::WriteOnlyArray { array_name, .. } => Some(array_name),
    ///         _ => None,
    ///     })
    ///     .collect();
    ///
    /// if !write_only_arrays.is_empty() {
    ///     log::warn!("Volume estimate may be inaccurate: missing reads for {:?}",
    ///                write_only_arrays);
    ///     // Fallback to conservative estimate
    /// }
    /// ```
    ///
    /// **Design Note**: Warnings are accumulated (not replaced) to preserve all
    /// diagnostic information from different analysis stages.
    pub validation_warnings: Vec<ValidationWarning>,
}

impl std::fmt::Debug for AccessInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AccessInfo")
            .field("stmt_accesses", &self.stmt_accesses)
            .field("arrays", &self.arrays)
            .field("has_tier2_reads", &self.reads_union_map.is_some())
            .field("has_tier2_writes", &self.writes_union_map.is_some())
            .field("validation_warnings_count", &self.validation_warnings.len())
            .finish()
    }
}

/// Opaque handle to ISL schedule
#[derive(Clone)]
pub struct ScheduleHandle {
    // Internal pointer to ISL schedule
    pub(crate) _inner: Arc<()>, // Placeholder - in real implementation would hold isl_schedule*
}

impl ScheduleHandle {
    /// Create a new placeholder schedule handle
    pub fn new_placeholder() -> Self {
        ScheduleHandle {
            _inner: Arc::new(()),
        }
    }
}

impl std::fmt::Debug for ScheduleHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "ScheduleHandle")
    }
}

/// Opaque handle to ISL context
#[derive(Clone)]
pub struct ContextHandle {
    pub(crate) _inner: Arc<()>, // Placeholder - in real implementation would hold isl_ctx*
}

impl ContextHandle {
    /// Create a new placeholder context handle
    pub fn new_placeholder() -> Self {
        ContextHandle {
            _inner: Arc::new(()),
        }
    }
}

impl std::fmt::Debug for ContextHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "ContextHandle")
    }
}

/// Access information for a single statement
#[derive(Debug, Clone)]
pub struct StmtAccess {
    /// Statement identifier
    pub stmt_id: String,

    /// Iteration domain of the statement
    pub domain: DomainHandle,

    /// Read access relations (iteration space -> memory space)
    pub reads: AccessMapHandle,

    /// Write access relations (iteration space -> memory space)
    pub writes: AccessMapHandle,

    /// May-read relations (for conditional accesses)
    pub may_reads: Option<AccessMapHandle>,

    /// May-write relations (for conditional accesses)
    pub may_writes: Option<AccessMapHandle>,
}

/// Opaque handle to ISL union set (domain)
#[derive(Clone)]
pub struct DomainHandle {
    pub(crate) _inner: Arc<()>, // Placeholder
}

impl std::fmt::Debug for DomainHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "DomainHandle")
    }
}

/// Opaque handle to ISL union map (access relations)
#[derive(Clone)]
pub struct AccessMapHandle {
    pub(crate) _inner: Arc<()>, // Placeholder
}

impl std::fmt::Debug for AccessMapHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "AccessMapHandle")
    }
}

/// Information about an array
#[derive(Debug, Clone)]
pub struct ArrayInfo {
    /// Array name
    pub name: String,

    /// Number of dimensions
    pub dimensions: usize,

    /// Size of each dimension (if known statically)
    pub sizes: Vec<Option<i64>>,

    /// Element type (for cost modeling)
    pub element_type: DataType,

    /// Layout information (row-major, column-major, etc.)
    pub layout: MemoryLayout,
}

/// Data types for array elements
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DataType {
    Float32,
    Float64,
    Int32,
    Int64,
    Bool,
    Custom(usize), // Custom type with size in bytes
}

/// Memory layout patterns
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryLayout {
    RowMajor,
    ColumnMajor,
    Tiled { tile_size: [usize; 2] },
    Custom,
}

impl StmtAccess {
    /// Create a new statement access with the given ID
    pub fn new(stmt_id: String) -> Self {
        StmtAccess {
            stmt_id,
            domain: DomainHandle {
                _inner: Arc::new(()),
            },
            reads: AccessMapHandle {
                _inner: Arc::new(()),
            },
            writes: AccessMapHandle {
                _inner: Arc::new(()),
            },
            may_reads: None,
            may_writes: None,
        }
    }

    /// Check if this statement has any read accesses
    pub fn has_reads(&self) -> bool {
        // In real implementation, would check if access maps are empty
        true // Placeholder
    }

    /// Check if this statement has any write accesses
    pub fn has_writes(&self) -> bool {
        // In real implementation, would check if access maps are empty
        true // Placeholder
    }

    /// Get all read accesses (must + may)
    pub fn all_reads(&self) -> AccessMapHandle {
        // In real implementation, would union must and may reads
        self.reads.clone()
    }

    /// Get all write accesses (must + may)
    pub fn all_writes(&self) -> AccessMapHandle {
        // In real implementation, would union must and may writes
        self.writes.clone()
    }
}

impl AccessInfo {
    /// Create new access info with the given context and schedule
    pub fn new(ctx: ContextHandle, schedule: ScheduleHandle) -> Self {
        AccessInfo {
            schedule,
            stmt_accesses: HashMap::new(),
            arrays: HashMap::new(),
            ctx,
            reads_union_map: None,
            writes_union_map: None,
            validation_warnings: Vec::new(),
        }
    }

    /// Add access information for a statement
    pub fn add_statement(&mut self, stmt: StmtAccess) {
        self.stmt_accesses.insert(stmt.stmt_id.clone(), stmt);
    }

    /// Add array information
    pub fn add_array(&mut self, array: ArrayInfo) {
        self.arrays.insert(array.name.clone(), array);
    }

    /// Collect all read accesses across all statements
    pub fn collect_all_reads(&self) -> AccessMapHandle {
        // In real implementation, would union all read maps
        AccessMapHandle {
            _inner: Arc::new(()),
        }
    }

    /// Collect all write accesses across all statements
    pub fn collect_all_writes(&self) -> AccessMapHandle {
        // In real implementation, would union all write maps
        AccessMapHandle {
            _inner: Arc::new(()),
        }
    }

    /// Get the iteration domain for all statements
    pub fn get_domain(&self) -> DomainHandle {
        // In real implementation, would union all statement domains
        DomainHandle {
            _inner: Arc::new(()),
        }
    }

    /// Check if accesses are valid for the current schedule
    pub fn validate_accesses(&self) -> Result<(), String> {
        // Placeholder validation
        Ok(())
    }

    /// Clone with a new schedule but same access information
    pub fn with_schedule(&self, new_schedule: ScheduleHandle) -> Self {
        AccessInfo {
            schedule: new_schedule,
            stmt_accesses: self.stmt_accesses.clone(),
            arrays: self.arrays.clone(),
            ctx: self.ctx.clone(),
            reads_union_map: self.reads_union_map.clone(),
            writes_union_map: self.writes_union_map.clone(),
            validation_warnings: self.validation_warnings.clone(),
        }
    }

    // ========================================================================
    // P1.2 INTEGRATION: Polymer Access File Integration
    // ========================================================================

    /// **[P1.2]** Populate UnionMaps from Polymer access files
    ///
    /// This method bridges the gap between Polymer-generated access files and
    /// the dependency analysis pipeline, enabling precise flow analysis.
    ///
    /// # Workflow
    /// ```text
    /// Polymer files (*.reads, *.writes)
    ///     |
    ///     v
    /// polymer_access_reader::read_polymer_access_file()
    ///     |
    ///     v
    /// PolymerAccessInfo::to_union_maps()
    ///     |
    ///     v
    /// AccessInfo.{reads,writes}_union_map
    ///     |
    ///     v
    /// DependencyInfo::compute_from_access_info()
    /// ```
    ///
    /// # Arguments
    /// * `reads_file` - Path to Polymer reads file (e.g., `gemm_accesses.reads`)
    /// * `writes_file` - Path to Polymer writes file (e.g., `gemm_accesses.writes`)
    /// * `ctx` - ISL context (must match schedule's context for flow analysis)
    ///
    /// # Returns
    /// * `Ok(())` - UnionMaps successfully populated
    /// * `Err(msg)` - File not found, parsing error, or ISL conversion failure
    ///
    /// # Example
    /// ```rust,ignore
    /// use polysat::{AccessInfo, ContextHandle, AccessScheduleHandle};
    /// use std::path::Path;
    /// use std::sync::Arc;
    /// use isl_rs::Context;
    ///
    /// let ctx = Arc::new(Context::alloc());
    /// let mut access_info = AccessInfo::new(
    ///     ContextHandle::new_placeholder(),
    ///     AccessScheduleHandle::new_placeholder()
    /// );
    ///
    /// // Populate from Polymer files
    /// access_info.populate_from_polymer_files(
    ///     Path::new("tests/test_data/gemm_256/gemm_accesses.reads"),
    ///     Path::new("tests/test_data/gemm_256/gemm_accesses.writes"),
    ///     &ctx
    /// )?;
    ///
    /// // Now compute_from_access_info() will use Tier-2 path
    /// let deps = DependencyInfo::compute_from_access_info(&access_info, &schedule, None)?;
    /// ```
    ///
    /// # Integration Status
    /// - Polymer reader implemented (`src/polymer_access_reader.rs`)
    /// - Flow analysis path exists in `compute_from_access_info` (checks UnionMaps first)
    /// - Verified with real GEMM 256x256 data
    /// - This method completes the integration
    pub fn populate_from_polymer_files(
        &mut self,
        reads_file: &std::path::Path,
        writes_file: &std::path::Path,
        ctx: &Arc<isl_rs::Context>,
    ) -> Result<(), String> {
        use crate::polymer_access_reader;

        // Read and parse reads file
        let reads_info =
            polymer_access_reader::read_polymer_access_file(reads_file).map_err(|e| {
                format!(
                    "Failed to read Polymer reads file {}: {}",
                    reads_file.display(),
                    e
                )
            })?;

        // Read and parse writes file
        let writes_info =
            polymer_access_reader::read_polymer_access_file(writes_file).map_err(|e| {
                format!(
                    "Failed to read Polymer writes file {}: {}",
                    writes_file.display(),
                    e
                )
            })?;

        // Convert to ISL UnionMaps
        let (reads_umap, _) = reads_info
            .to_union_maps(ctx)
            .map_err(|e| format!("Failed to convert reads to UnionMap: {}", e))?;

        let (_, writes_umap) = writes_info
            .to_union_maps(ctx)
            .map_err(|e| format!("Failed to convert writes to UnionMap: {}", e))?;

        // Populate fields (enables precise ISL flow analysis)
        self.reads_union_map = Some(Arc::new(reads_umap));
        self.writes_union_map = Some(Arc::new(writes_umap));

        log::info!("Populated UnionMaps from Polymer files");
        log::debug!(
            "  Reads:  {}",
            self.reads_union_map.as_ref().unwrap().to_str()
        );
        log::debug!(
            "  Writes: {}",
            self.writes_union_map.as_ref().unwrap().to_str()
        );

        Ok(())
    }

    /// **[P1.2]** Factory method: Create AccessInfo with Polymer data pre-loaded
    ///
    /// Convenience wrapper around `new()` + `populate_from_polymer_files()`.
    ///
    /// # Example
    /// ```rust,ignore
    /// let access_info = AccessInfo::from_polymer_files(
    ///     Path::new("test_data/gemm_256"),
    ///     "gemm_accesses",
    ///     &ctx,
    ///     ContextHandle::new_placeholder(),
    ///     AccessScheduleHandle::new_placeholder()
    /// )?;
    /// ```
    pub fn from_polymer_files(
        polymer_dir: &std::path::Path,
        base_name: &str,
        ctx: &Arc<isl_rs::Context>,
        ctx_handle: ContextHandle,
        schedule_handle: ScheduleHandle,
    ) -> Result<Self, String> {
        let mut access_info = Self::new(ctx_handle, schedule_handle);

        let reads_file = polymer_dir.join(format!("{}.reads", base_name));
        let writes_file = polymer_dir.join(format!("{}.writes", base_name));

        access_info.populate_from_polymer_files(&reads_file, &writes_file, ctx)?;

        Ok(access_info)
    }
}

impl DataType {
    /// Get the size in bytes for this data type
    pub fn size_bytes(&self) -> usize {
        match self {
            DataType::Float32 => 4,
            DataType::Float64 => 8,
            DataType::Int32 => 4,
            DataType::Int64 => 8,
            DataType::Bool => 1,
            DataType::Custom(size) => *size,
        }
    }

    /// Check if this type supports vectorization
    pub fn supports_vectorization(&self) -> bool {
        matches!(
            self,
            DataType::Float32 | DataType::Float64 | DataType::Int32 | DataType::Int64
        )
    }
}

impl ArrayInfo {
    /// Calculate the total size in bytes
    pub fn total_bytes(&self) -> Option<usize> {
        let mut total = 1usize;
        for size in &self.sizes {
            match size {
                Some(s) if *s > 0 => total *= *s as usize,
                _ => return None, // Dynamic or invalid size
            }
        }
        Some(total * self.element_type.size_bytes())
    }

    /// Check if array has static (compile-time known) dimensions
    pub fn has_static_dimensions(&self) -> bool {
        self.sizes.iter().all(|s| s.is_some() && s.unwrap() > 0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stmt_access_creation() {
        let stmt = StmtAccess::new("S0".to_string());
        assert_eq!(stmt.stmt_id, "S0");
    }

    #[test]
    fn test_array_info_size_calculation() {
        let array = ArrayInfo {
            name: "A".to_string(),
            dimensions: 2,
            sizes: vec![Some(100), Some(200)],
            element_type: DataType::Float32,
            layout: MemoryLayout::RowMajor,
        };

        assert_eq!(array.total_bytes(), Some(100 * 200 * 4));
        assert!(array.has_static_dimensions());
    }

    #[test]
    fn test_data_type_properties() {
        assert_eq!(DataType::Float32.size_bytes(), 4);
        assert_eq!(DataType::Float64.size_bytes(), 8);
        assert!(DataType::Float32.supports_vectorization());
        assert!(!DataType::Bool.supports_vectorization());
    }
}
