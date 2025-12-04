//! Polymer Access File Reader
//!
//! Reads ground-truth access patterns from Polymer's ISL dump files.
//! Polymer's `--islexternal-dump-accesses` generates YAML-like files with:
//! - Domain constraints
//! - Read/write access relations in ISL format
//!
//! # Format Example
//! ```yaml
//! domain: "{ S0[i0, i1, i2] : 0 <= i0 <= 255 and 0 <= i1 <= 255 and 0 <= i2 <= 255 }"
//! accesses:
//!   - S0:
//!       reads:
//!         - "{ [i0, i1, i2] -> A1[o0, o1] : o0 = i0 and o1 = i1 }"
//!         - "{ [i0, i1, i2] -> A2[o0, o1] : o0 = i0 and o1 = i2 }"
//!       writes:
//!         - "{ [i0, i1, i2] -> A1[o0, o1] : o0 = i0 and o1 = i1 }"
//! ```
//!
//! # Polymer Array Naming
//! Polymer uses positional indices (A1, A2, A3...) not semantic names (C, A, B).
//! This module provides both raw parsing and semantic mapping capabilities.

use isl_rs::{Context, UnionMap};
use std::collections::HashMap;
use std::fs;
use std::path::Path;
use std::sync::Arc;

/// Parsed access information from Polymer dump file
#[derive(Debug, Clone)]
pub struct PolymerAccessInfo {
    /// Domain constraints (e.g., "{ S0[i0,i1,i2] : 0 <= i0 <= 255 ... }")
    pub domain: String,

    /// Read access relations per statement
    /// Key: Statement name (e.g., "S0")
    /// Value: Vector of ISL read relations
    pub reads_by_stmt: HashMap<String, Vec<String>>,

    /// Write access relations per statement
    pub writes_by_stmt: HashMap<String, Vec<String>>,
}

impl PolymerAccessInfo {
    /// Convert to ISL UnionMap for reads and writes
    ///
    /// # Arguments
    /// * `ctx` - ISL context for parsing
    ///
    /// # Returns
    /// Tuple of (read_union_map, write_union_map)
    pub fn to_union_maps(&self, ctx: &Arc<Context>) -> Result<(UnionMap, UnionMap), String> {
        // Combine all read relations across all statements
        let mut read_relations = Vec::new();
        for stmt_reads in self.reads_by_stmt.values() {
            read_relations.extend(stmt_reads.iter().cloned());
        }

        // Combine all write relations
        let mut write_relations = Vec::new();
        for stmt_writes in self.writes_by_stmt.values() {
            write_relations.extend(stmt_writes.iter().cloned());
        }

        // Create union maps
        // ISL union map format: "{ relation1; relation2; relation3 }"
        // NOTE: Individual relations already have "{ ... }" braces from Polymer dump,
        // but we need to strip them and combine into a single union map
        let reads_str = if read_relations.is_empty() {
            "{ }".to_string()
        } else {
            // Strip outer braces from each relation and rejoin
            let stripped: Vec<String> = read_relations
                .iter()
                .map(|rel| {
                    let trimmed = rel.trim();
                    if trimmed.starts_with('{') && trimmed.ends_with('}') {
                        trimmed[1..trimmed.len() - 1].trim().to_string()
                    } else {
                        trimmed.to_string()
                    }
                })
                .collect();
            format!("{{ {} }}", stripped.join("; "))
        };

        let writes_str = if write_relations.is_empty() {
            "{ }".to_string()
        } else {
            // Strip outer braces from each relation and rejoin
            let stripped: Vec<String> = write_relations
                .iter()
                .map(|rel| {
                    let trimmed = rel.trim();
                    if trimmed.starts_with('{') && trimmed.ends_with('}') {
                        trimmed[1..trimmed.len() - 1].trim().to_string()
                    } else {
                        trimmed.to_string()
                    }
                })
                .collect();
            format!("{{ {} }}", stripped.join("; "))
        };

        let reads = UnionMap::read_from_str(ctx, &reads_str);
        let writes = UnionMap::read_from_str(ctx, &writes_str);

        Ok((reads, writes))
    }

    /// Get array names used in access patterns
    ///
    /// Extracts array names like "A1", "A2", "A3" from Polymer format.
    /// Returns sorted unique array names.
    pub fn get_array_names(&self) -> Vec<String> {
        use regex::Regex;

        let mut array_names = std::collections::HashSet::new();
        let array_re = Regex::new(r"->\s*(\w+)\[").unwrap();

        // Extract from reads
        for stmt_reads in self.reads_by_stmt.values() {
            for read_rel in stmt_reads {
                for cap in array_re.captures_iter(read_rel) {
                    array_names.insert(cap[1].to_string());
                }
            }
        }

        // Extract from writes
        for stmt_writes in self.writes_by_stmt.values() {
            for write_rel in stmt_writes {
                for cap in array_re.captures_iter(write_rel) {
                    array_names.insert(cap[1].to_string());
                }
            }
        }

        let mut names: Vec<String> = array_names.into_iter().collect();
        names.sort();
        names
    }
}

/// Read Polymer access file from path
///
/// # File Format
/// Polymer generates YAML-like format:
/// ```yaml
/// domain: "<isl_domain>"
/// accesses:
///   - <stmt_name>:
///       reads:
///         - "<isl_relation>"
///       writes:
///         - "<isl_relation>"
/// ```
///
/// # Arguments
/// * `path` - Path to Polymer access dump file
///
/// # Returns
/// Parsed `PolymerAccessInfo` containing domain, reads, and writes
///
/// # Errors
/// Returns error if:
/// - File doesn't exist
/// - Format is invalid
/// - ISL relations are malformed
pub fn read_polymer_access_file(path: &Path) -> Result<PolymerAccessInfo, String> {
    // Read file content
    let content = fs::read_to_string(path).map_err(|e| {
        format!(
            "Failed to read Polymer access file {}: {}",
            path.display(),
            e
        )
    })?;

    // Parse domain
    let domain = extract_domain(&content)?;

    // Parse accesses section
    let (reads_by_stmt, writes_by_stmt) = parse_accesses_section(&content)?;

    Ok(PolymerAccessInfo {
        domain,
        reads_by_stmt,
        writes_by_stmt,
    })
}

/// Extract domain string from Polymer access file
///
/// Format: `domain: "{ S0[i0,i1,i2] : constraints }"`
fn extract_domain(content: &str) -> Result<String, String> {
    // Find line starting with "domain:"
    for line in content.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with("domain:") {
            // Extract quoted string after "domain:"
            let after_domain = trimmed.strip_prefix("domain:").unwrap().trim();

            // Handle both quoted and unquoted domains
            if after_domain.starts_with('"') {
                // Quoted: domain: "{ ... }"
                let start = after_domain
                    .find('"')
                    .ok_or("Missing opening quote in domain")?;
                let end = after_domain
                    .rfind('"')
                    .ok_or("Missing closing quote in domain")?;
                if start >= end {
                    return Err("Invalid domain quotes".to_string());
                }
                return Ok(after_domain[start + 1..end].to_string());
            } else {
                // Unquoted: domain: { ... }
                return Ok(after_domain.to_string());
            }
        }
    }

    Err("No domain found in Polymer access file".to_string())
}

/// Parse accesses section containing reads and writes per statement
///
/// Format:
/// ```yaml
/// accesses:
///   - S0:
///       reads:
///         - "relation1"
///         - "relation2"
///       writes:
///         - "relation3"
/// ```
fn parse_accesses_section(
    content: &str,
) -> Result<(HashMap<String, Vec<String>>, HashMap<String, Vec<String>>), String> {
    let mut reads_by_stmt = HashMap::new();
    let mut writes_by_stmt = HashMap::new();

    // Simple state machine parser for YAML-like format
    let mut in_accesses = false;
    let mut current_stmt: Option<String> = None;
    let mut in_reads = false;
    let mut in_writes = false;

    for line in content.lines() {
        let trimmed = line.trim();

        // Check if entering accesses section
        if trimmed.starts_with("accesses:") {
            in_accesses = true;
            continue;
        }

        if !in_accesses {
            continue;
        }

        // Check for statement name (e.g., "- S0:")
        if trimmed.starts_with("- ") && trimmed.ends_with(':') {
            let stmt_name = trimmed[2..trimmed.len() - 1].trim().to_string();
            current_stmt = Some(stmt_name.clone());
            reads_by_stmt
                .entry(stmt_name.clone())
                .or_insert_with(Vec::new);
            writes_by_stmt.entry(stmt_name).or_insert_with(Vec::new);
            in_reads = false;
            in_writes = false;
            continue;
        }

        // Check for reads/writes section
        if trimmed == "reads:" {
            in_reads = true;
            in_writes = false;
            continue;
        }

        if trimmed == "writes:" {
            in_reads = false;
            in_writes = true;
            continue;
        }

        // Parse ISL relation line (starts with "- ")
        if trimmed.starts_with("- \"") || trimmed.starts_with("- {") {
            if let Some(ref stmt) = current_stmt {
                // Extract relation string
                let relation = if trimmed.starts_with("- \"") {
                    // Quoted format: - "{ ... }"
                    let start = trimmed.find('"').ok_or("Missing quote in relation")?;
                    let end = trimmed.rfind('"').ok_or("Missing closing quote")?;
                    trimmed[start + 1..end].to_string()
                } else {
                    // Unquoted format: - { ... }
                    trimmed[2..].trim().to_string()
                };

                if in_reads {
                    reads_by_stmt.get_mut(stmt).unwrap().push(relation);
                } else if in_writes {
                    writes_by_stmt.get_mut(stmt).unwrap().push(relation);
                }
            }
        }
    }

    Ok((reads_by_stmt, writes_by_stmt))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_parse_gemm_access_file() {
        // Create temporary file with Polymer GEMM access format
        let mut temp_file = NamedTempFile::new().unwrap();
        writeln!(temp_file, r#"domain: "{{ S0[i0, i1, i2] : 0 <= i0 <= 255 and 0 <= i1 <= 255 and 0 <= i2 <= 255 }}""#).unwrap();
        writeln!(temp_file, "accesses:").unwrap();
        writeln!(temp_file, "  - S0:").unwrap();
        writeln!(temp_file, "      reads:").unwrap();
        writeln!(
            temp_file,
            r#"        - "{{ [i0, i1, i2] -> A1[o0, o1] : o0 = i0 and o1 = i1 }}""#
        )
        .unwrap();
        writeln!(
            temp_file,
            r#"        - "{{ [i0, i1, i2] -> A2[o0, o1] : o0 = i0 and o1 = i2 }}""#
        )
        .unwrap();
        writeln!(
            temp_file,
            r#"        - "{{ [i0, i1, i2] -> A3[o0, o1] : o0 = i2 and o1 = i1 }}""#
        )
        .unwrap();
        writeln!(temp_file, "      writes:").unwrap();
        writeln!(
            temp_file,
            r#"        - "{{ [i0, i1, i2] -> A1[o0, o1] : o0 = i0 and o1 = i1 }}""#
        )
        .unwrap();
        temp_file.flush().unwrap();

        // Parse file
        let access_info = read_polymer_access_file(temp_file.path()).unwrap();

        // Verify domain
        assert!(access_info.domain.contains("S0[i0, i1, i2]"));

        // Verify reads
        assert_eq!(access_info.reads_by_stmt.len(), 1);
        let s0_reads = &access_info.reads_by_stmt["S0"];
        assert_eq!(s0_reads.len(), 3); // A1, A2, A3 reads

        // Verify writes
        assert_eq!(access_info.writes_by_stmt.len(), 1);
        let s0_writes = &access_info.writes_by_stmt["S0"];
        assert_eq!(s0_writes.len(), 1); // A1 write

        // Verify array names
        let arrays = access_info.get_array_names();
        assert_eq!(arrays, vec!["A1", "A2", "A3"]);
    }

    /// Integration Test: Verify Polymer reader with real GEMM 256x256 data
    ///
    /// This test validates that the Polymer access file parser correctly handles
    /// ground-truth access patterns from actual Polygeist/Polymer output.
    ///
    /// **Test Data**: `tests/test_data/gemm_256/gemm_accesses.{reads,writes}`
    /// - Generated by: Polygeist with `--islexternal-dump-accesses`
    /// - Kernel: GEMM C[i,j] += A[i,k] * B[k,j]
    /// - Domain: S0[i,j,k] with 0 <= i,j,k < 256
    ///
    /// **Verification**:
    /// 1. Parse reads file -> extract A[i,k] and B[k,j] access patterns
    /// 2. Parse writes file -> extract C[i,j] access pattern
    /// 3. Convert to ISL UnionMaps -> verify ISL parsing succeeds
    /// 4. Validate UnionMap string representation
    #[test]
    fn test_real_gemm_256_polymer_data() {
        use std::path::PathBuf;

        // Path to real Polymer access files
        let test_data_dir =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/test_data/gemm_256");

        let reads_file = test_data_dir.join("gemm_accesses.reads");
        let writes_file = test_data_dir.join("gemm_accesses.writes");

        // **Step 1**: Parse reads file
        let reads_info =
            read_polymer_access_file(&reads_file).expect("Failed to parse GEMM reads file");

        // Verify domain (should be 3D: i, j, k)
        assert!(
            reads_info.domain.contains("S0[i, j, k]"),
            "Domain should be S0[i,j,k], got: {}",
            reads_info.domain
        );
        assert!(
            reads_info.domain.contains("0 <= i < 256"),
            "Domain should have i bound [0,256), got: {}",
            reads_info.domain
        );

        // Verify S0 statement exists
        assert_eq!(
            reads_info.reads_by_stmt.len(),
            1,
            "Should have exactly 1 statement"
        );
        assert!(
            reads_info.reads_by_stmt.contains_key("S0"),
            "Should have S0 statement"
        );

        // Verify GEMM read accesses: A[i,k] and B[k,j]
        let s0_reads = &reads_info.reads_by_stmt["S0"];
        assert_eq!(
            s0_reads.len(),
            2,
            "GEMM should have 2 read accesses (A and B), got: {:?}",
            s0_reads
        );

        // Check A[i,k] access pattern
        let has_a_access = s0_reads.iter().any(|rel| rel.contains("A[i, k]"));
        assert!(
            has_a_access,
            "Should have A[i,k] read access in: {:?}",
            s0_reads
        );

        // Check B[k,j] access pattern
        let has_b_access = s0_reads.iter().any(|rel| rel.contains("B[k, j]"));
        assert!(
            has_b_access,
            "Should have B[k,j] read access in: {:?}",
            s0_reads
        );

        // **Step 2**: Parse writes file
        let writes_info =
            read_polymer_access_file(&writes_file).expect("Failed to parse GEMM writes file");

        // Verify write access: C[i,j]
        let s0_writes = &writes_info.writes_by_stmt["S0"];
        assert_eq!(
            s0_writes.len(),
            1,
            "GEMM should have 1 write access (C), got: {:?}",
            s0_writes
        );

        let has_c_access = s0_writes.iter().any(|rel| rel.contains("C[i, j]"));
        assert!(
            has_c_access,
            "Should have C[i,j] write access in: {:?}",
            s0_writes
        );

        // **Step 3**: Convert to ISL UnionMaps (verify ISL parsing)
        let ctx = Arc::new(Context::alloc());

        let (reads_umap, _writes_umap_reads) = reads_info
            .to_union_maps(&ctx)
            .expect("Failed to convert reads to UnionMap");

        let (_reads_umap_writes, writes_umap) = writes_info
            .to_union_maps(&ctx)
            .expect("Failed to convert writes to UnionMap");

        // **Step 4**: Verify UnionMap string representation
        let reads_str = reads_umap.to_str();
        let writes_str = writes_umap.to_str();

        // Reads should contain both A and B accesses
        assert!(
            reads_str.contains("A") && reads_str.contains("B"),
            "Reads UnionMap should reference both A and B arrays, got: {}",
            reads_str
        );

        // Writes should contain C access
        assert!(
            writes_str.contains("C"),
            "Writes UnionMap should reference C array, got: {}",
            writes_str
        );

        // Verify UnionMaps are not empty
        assert!(!reads_umap.is_empty(), "Reads UnionMap should not be empty");
        assert!(
            !writes_umap.is_empty(),
            "Writes UnionMap should not be empty"
        );

        println!("P1.1: Successfully parsed and converted real GEMM 256x256 Polymer data");
        println!("  Reads:  {}", reads_str);
        println!("  Writes: {}", writes_str);
    }

    #[test]
    fn test_extract_domain() {
        let content = r#"domain: "{ S0[i0, i1, i2] : 0 <= i0 <= 255 }"
accesses:
  - S0:"#;

        let domain = extract_domain(content).unwrap();
        assert_eq!(domain, "{ S0[i0, i1, i2] : 0 <= i0 <= 255 }");
    }
}
