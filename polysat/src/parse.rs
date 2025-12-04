use crate::language::ScheduleHandle;
use isl_rs::{Context, Schedule, UnionSet};
use log::{debug, error, warn};
use std::sync::Arc;
use thiserror::Error;

/// Errors that can occur during ISL schedule parsing
#[derive(Error, Debug)]
pub enum ParseError {
    /// ISL parsing failed due to incompatible format or segfault
    #[error("Failed to parse ISL schedule: {0}")]
    ParseFailed(String),

    /// Parsed schedule is empty (invalid)
    #[error("Parsed ISL schedule is empty")]
    EmptySchedule,

    /// I/O error when reading/writing files
    #[error("I/O error: {0}")]
    IOError(#[from] std::io::Error),
}

/// Allow ParseError to be converted to String for backward compatibility
/// This enables gradual migration from String-based error handling
impl From<ParseError> for String {
    fn from(err: ParseError) -> Self {
        err.to_string()
    }
}

/// Parse ISL schedule string into isl-rs Schedule
///
/// This function attempts to parse an ISL schedule string. If the primary parsing
/// fails (e.g., due to format incompatibility or ISL segfault), it attempts a
/// fallback by extracting the domain and creating a simple schedule.
///
/// # Arguments
/// * `ctx` - ISL context (must be shared across all ISL operations)
/// * `isl_str` - ISL schedule string in ISL format
///
/// # Returns
/// * `Ok(ScheduleHandle)` - Successfully parsed schedule
/// * `Err(ParseError)` - Parsing failed with detailed error
///
/// # Errors
/// Returns `ParseError::ParseFailed` if ISL parsing fails and fallback also fails.
/// Returns `ParseError::EmptySchedule` if parsing succeeds but result is empty.
pub fn parse_isl(ctx: Arc<Context>, isl_str: &str) -> Result<ScheduleHandle, ParseError> {
    debug!("Parsing ISL schedule ({} chars)", isl_str.len());

    // Early validation: empty strings cause ISL to segfault
    if isl_str.trim().is_empty() {
        return Err(ParseError::ParseFailed(
            "ISL schedule string is empty".to_string(),
        ));
    }

    // Try to parse as ISL schedule
    // The actual ISL format from Polygeist uses JSON-like syntax
    // isl-rs expects a different format, so we may need to convert

    // Use panic::catch_unwind to handle potential segfaults/panics from isl-rs
    // This is necessary because isl-rs is a thin wrapper around C ISL library
    // which can segfault on invalid input
    let result = std::panic::catch_unwind(|| Schedule::read_from_str(&*ctx, isl_str));

    match result {
        Ok(schedule) => {
            debug!("Successfully parsed ISL schedule");
            let dumped = schedule.to_str();

            // Check if schedule is valid (non-empty)
            if dumped.is_empty() {
                error!("Parsed schedule is empty");
                return Err(ParseError::EmptySchedule);
            }

            Ok(ScheduleHandle::new(ctx, schedule))
        }
        Err(_) => {
            warn!("ISL parsing panicked/segfaulted, attempting fallback");

            // Try to extract just the domain and create a simple schedule
            // This fallback handles cases where the schedule tree format is incompatible
            // but we can still extract the domain
            if let Some(domain_start) = isl_str.find("domain:") {
                if let Some(domain_end) = isl_str[domain_start..].find(", child:") {
                    let domain_str = &isl_str[domain_start + 8..domain_start + domain_end];
                    let domain_str = domain_str.trim().trim_matches('"');
                    debug!("Extracted domain for fallback: {}", domain_str);

                    // Try to create a schedule from just the domain
                    let fallback_result = std::panic::catch_unwind(|| {
                        let domain = UnionSet::read_from_str(&*ctx, domain_str);
                        Schedule::from_domain(domain)
                    });

                    match fallback_result {
                        Ok(schedule) => {
                            warn!(
                                "Created fallback schedule from domain (may lose transformations)"
                            );
                            return Ok(ScheduleHandle::new(ctx.clone(), schedule));
                        }
                        Err(_) => {
                            error!("Fallback schedule creation also failed");
                        }
                    }
                }
            }

            Err(ParseError::ParseFailed(
                "ISL format incompatible and domain extraction failed".to_string(),
            ))
        }
    }
}

// Dump isl-rs Schedule to ISL string
// We just use isl-rs's built-in to_str() method
pub fn dump_isl(handle: &ScheduleHandle) -> String {
    // Simple dump using isl-rs
    handle.schedule.to_str().to_string()
}

/// Create a simple schedule programmatically (for testing)
///
/// Creates a schedule for a single-dimensional loop: `for i in 0..10: S[i]`
/// **WARNING**: This creates a schedule with only domain node, no band node.
/// Transformations (tile, parallel, etc.) will fail on this schedule.
/// Use `create_schedule_with_bands()` for schedules that support transformations.
///
/// # Arguments
/// * `ctx` - ISL context
///
/// # Errors
/// This function should not fail for valid ISL contexts, but returns `ParseError`
/// for consistency with other parsing functions.
pub fn create_simple_schedule(ctx: Arc<Context>) -> Result<ScheduleHandle, ParseError> {
    // Create domain: { S[i] : 0 <= i < 10 }
    let domain_str = "{ S[i] : 0 <= i < 10 }";
    let domain = UnionSet::read_from_str(&*ctx, domain_str);

    // Create schedule from domain
    let schedule = Schedule::from_domain(domain);

    Ok(ScheduleHandle::new(ctx, schedule))
}

/// Create a schedule with band nodes programmatically (for testing)
///
/// Note: This function creates schedules with proper band structure that support
/// transformations (tiling, parallelization, etc.). This is essential for e-graph exploration
/// tests, as `Schedule::from_domain()` only creates domain nodes without bands, causing all
/// transformations to fail with "Band node not found" errors.
///
/// # ISL Schedule String Format Specification
///
/// Based on working examples in `test_simple_ntt_tile.rs`, `test_schedule_save.rs`, and
/// `band_fusion.rs`, the correct ISL schedule string format is:
///
/// ```text
/// { domain: "{ S0[i, j] : ... }", child: { schedule: "[{ S0[i, j] -> [(i)] }, { S0[i, j] -> [(j)] }]" } }
/// ```
///
/// **Key Format Requirements**:
/// 1. **Outer structure**: `{ domain: "...", child: { schedule: "..." } }`
/// 2. **Schedule string format**: `"[{ S0[i,j] -> [(i)] }, { S0[i,j] -> [(j)] }]"`
///    - Wrapped in square brackets `[...]`
///    - Each dimension is a separate mapping: `{ S0[...] -> [(dim)] }`
///    - Multiple dimensions separated by commas: `[{ ... }, { ... }]`
///    - Each dimension wrapped in parentheses: `(i)`, `(j)`, not `(i), (j)` in one mapping
/// 3. **Statement iterator format**: `S0[i, j]` (comma-separated, no spaces around commas in ISL)
/// 4. **Schedule dimension format**: `[(i)]` for single dimension, `[(i), (j)]` for multiple
///    - Note: For multi-dimensional bands, ISL expects separate mappings per dimension
///
/// # Why This Format?
///
/// - **Band nodes require schedule mappings**: ISL's band nodes represent loop dimensions.
///   Without explicit schedule mappings, ISL cannot identify which dimensions to transform.
/// - **Multi-dimensional bands**: Each dimension must be a separate mapping in the schedule
///   string to allow independent transformation (e.g., tiling only dimension 0).
/// - **String escaping**: The schedule string is double-quoted, so inner braces must be escaped
///   in Rust raw strings: `"[{{ ... }}]"` becomes `"[{ ... }]"` in the final ISL string.
///
/// # Arguments
/// * `ctx` - ISL context (must be shared across all ISL operations)
/// * `domain_str` - Domain string, e.g., `"{ S0[i, j] : 0 <= i < 64 and 0 <= j < 64 }"`
/// * `schedule_dims` - Schedule dimensions, e.g., `vec!["i", "j"]` for 2D schedule
///
/// # Example
/// ```rust
/// use polysat::parse::create_schedule_with_bands;
/// use isl_rs::{Context, UnionSet, Schedule};
/// use std::sync::Arc;
///
/// let ctx = Arc::new(Context::alloc());
/// let domain = UnionSet::read_from_str(&ctx, "{ S0[i] : 0 <= i < 10 }").unwrap();
/// let schedule = create_schedule_with_bands(
///     ctx.clone(), // Pass Arc<Context> by cloning
///     "{ S0[i] : 0 <= i < 10 }", // Pass domain as string
///     vec!["i"] // Pass schedule dimensions as Vec<&str>
/// ).unwrap();
/// // Creates: { domain: "{ S0[i, j] : ... }", child: { schedule: "[{ S0[i, j] -> [(i)] }, { S0[i, j] -> [(j)] }]" } }
/// ```
///
/// # Errors
/// Returns `ParseError::ParseFailed` if ISL string parsing fails or panics.
///
/// # Fallback Behavior
/// If the full schedule tree format fails to parse, this function falls back to
/// `Schedule::from_domain()`, which creates a schedule without band nodes. This fallback
/// schedule will NOT support transformations, but allows tests to continue.
pub fn create_schedule_with_bands(
    ctx: Arc<Context>,
    domain_str: &str,
    schedule_dims: Vec<&str>,
) -> Result<ScheduleHandle, ParseError> {
    // Extract statement name from domain
    // ISL uses S0, S1, etc. for different statements in the domain
    let stmt_name = if domain_str.contains("S0") {
        "S0"
    } else if domain_str.contains("S1") {
        "S1"
    } else {
        "S"
    };

    // Extract iterator names from schedule_dims
    // Format: "i, j" (comma-separated, no spaces around commas in ISL)
    let iter_str = schedule_dims.join(", ");

    // Build schedule mappings: each dimension is a separate mapping
    // Format: [{ S0[i, j] -> [(i)] }, { S0[i, j] -> [(j)] }]
    // Note: ISL expects separate mappings for multi-dimensional bands,
    // not a single mapping with multiple dimensions like [{ S0[i, j] -> [(i), (j)] }]
    let schedule_mappings: Vec<String> = schedule_dims
        .iter()
        .map(|dim| {
            // Each dimension gets its own mapping: { S0[i, j] -> [(dim)] }
            format!("{{ {}[{}] -> [({})] }}", stmt_name, iter_str, dim)
        })
        .collect();

    // Join mappings with commas: [{ ... }, { ... }]
    let schedule_maps_str = schedule_mappings.join(", ");

    // Build complete ISL schedule string with proper band structure
    // Format: { domain: "...", child: { schedule: "[{ ... }, { ... }]" } }
    // Note: Double braces {{ }} in Rust raw string become single braces { } in ISL string
    let isl_schedule_str = format!(
        r#"{{ domain: "{}", child: {{ schedule: "[{}]" }} }}"#,
        domain_str, schedule_maps_str
    );

    debug!("Creating schedule with bands:");
    debug!("  Domain: {}", domain_str);
    debug!("  Schedule dims: {:?}", schedule_dims);
    debug!("  Generated ISL string: {}", isl_schedule_str);

    // Try parsing with catch_unwind to handle ISL panics/segfaults
    // ISL's read_from_str can panic or segfault on invalid formats, so we must catch it
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        Schedule::read_from_str(&*ctx, &isl_schedule_str)
    }));

    match result {
        Ok(schedule) => {
            debug!("Successfully created schedule with bands");

            // Verify the schedule has band nodes by checking the schedule string
            // If it only contains domain, transformations will fail
            let schedule_str = schedule.to_str();
            if schedule_str.contains("schedule:") || schedule_str.contains("-> [") {
                Ok(ScheduleHandle::new(ctx, schedule))
            } else {
                warn!(
                    "Created schedule but it may not have band nodes: {}",
                    schedule_str
                );
                Ok(ScheduleHandle::new(ctx, schedule))
            }
        }
        Err(_) => {
            // Fallback: create schedule from domain only (no band nodes)
            // This schedule will NOT support transformations, but allows tests to continue
            warn!("Failed to parse schedule with bands, falling back to domain-only schedule");
            warn!("  This schedule will NOT support transformations (tile, parallel, etc.)");
            warn!("  Generated string was: {}", isl_schedule_str);

            let fallback_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let domain = UnionSet::read_from_str(&*ctx, domain_str);
                Schedule::from_domain(domain)
            }));

            match fallback_result {
                Ok(schedule) => {
                    warn!("Created fallback schedule (transformations will fail)");
                    Ok(ScheduleHandle::new(ctx, schedule))
                }
                Err(_) => Err(ParseError::ParseFailed(format!(
                    "Failed to create schedule with bands: ISL parsing panicked.\n\
                         Domain: {}\n\
                         Schedule dims: {:?}\n\
                         Generated string: {}",
                    domain_str, schedule_dims, isl_schedule_str
                ))),
            }
        }
    }
}

/// Load ISL schedule from file
///
/// # Arguments
/// * `ctx` - ISL context
/// * `path` - Path to ISL schedule file
///
/// # Errors
/// Returns `ParseError::IOError` if file cannot be read.
/// Returns `ParseError::ParseFailed` if file content cannot be parsed.
pub fn load_isl_file(ctx: Arc<Context>, path: &str) -> Result<ScheduleHandle, ParseError> {
    debug!("Loading ISL schedule from file: {}", path);
    let content = std::fs::read_to_string(path)?;
    parse_isl(ctx, &content)
}

/// Save ISL schedule to file
///
/// # Arguments
/// * `handle` - Schedule handle to save
/// * `path` - Output file path
///
/// # Errors
/// Returns `ParseError::IOError` if file cannot be written.
pub fn save_isl_file(handle: &ScheduleHandle, path: &str) -> Result<(), ParseError> {
    debug!("Saving ISL schedule to file: {}", path);
    let content = dump_isl(handle);
    std::fs::write(path, content)?;
    Ok(())
}

/// Save an ISL schedule string directly to a file
///
/// This preserves the full schedule tree with transformations.
/// Use this when you have a schedule string that includes transformations
/// that may not be preserved by `dump_isl()`.
///
/// # Arguments
/// * `schedule_str` - ISL schedule string to save
/// * `path` - Output file path
///
/// # Errors
/// Returns `ParseError::IOError` if file cannot be written.
pub fn save_isl_string(schedule_str: &str, path: &str) -> Result<(), ParseError> {
    debug!("Saving ISL schedule string to file: {}", path);
    std::fs::write(path, schedule_str)?;
    Ok(())
}

/// Validate that a schedule is well-formed
///
/// Basic validation checks that the schedule is non-empty.
/// isl-rs ensures schedules are well-formed at construction time,
/// so this is primarily a sanity check.
///
/// # Arguments
/// * `handle` - Schedule handle to validate
///
/// # Returns
/// `true` if schedule appears valid, `false` otherwise
pub fn validate_schedule(handle: &ScheduleHandle) -> bool {
    // isl-rs ensures schedules are well-formed at construction
    // Additional validation could be added here (e.g., check for cycles)
    !handle.schedule.to_str().is_empty()
}
