//! Module for importing Polygeist/Polymer ISL exports into PolySat
//! Handles parsing of schedule trees and access relations from Polymer's output format

use std::collections::HashMap;
use std::fs;
use std::path::Path;
use std::sync::Arc;

use serde::{Deserialize, Serialize};

use crate::access_analysis::{
    AccessInfo, AccessMapHandle, ArrayInfo, ContextHandle, DataType, DomainHandle, MemoryLayout,
    ScheduleHandle, StmtAccess,
};

/// Structure representing Polymer's export format
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct PolymerExport {
    /// Schedule information
    pub schedule: ScheduleExport,

    /// Access relations for all statements
    pub access_relations: Vec<AccessRelation>,

    /// Optional: Array metadata
    #[serde(default)]
    pub arrays: Vec<ArrayMetadata>,

    /// Optional: Additional context constraints
    #[serde(default)]
    pub context: Option<String>,
}

/// Schedule export format
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ScheduleExport {
    /// ISL schedule tree in string format
    pub tree: String,

    /// Union of iteration domains
    pub domain: String,

    /// Optional: Schedule constraints
    #[serde(default)]
    pub constraints: Option<String>,
}

/// Access relation for a single array access
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct AccessRelation {
    /// Statement identifier
    pub stmt_id: String,

    /// Access type: "read" or "write"
    #[serde(rename = "type")]
    pub r#type: String,

    /// Array being accessed
    pub array: String,

    /// ISL relation string: iteration space -> array space
    pub relation: String,

    /// Optional: Whether this is a may-access (conditional)
    #[serde(default)]
    pub is_may_access: bool,
}

/// Array metadata information
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ArrayMetadata {
    /// Array name
    pub name: String,

    /// Number of dimensions
    pub dimensions: usize,

    /// Size of each dimension (None for dynamic)
    pub sizes: Vec<Option<i64>>,

    /// Element type as string
    #[serde(default = "default_element_type")]
    pub element_type: String,

    /// Memory layout hint
    #[serde(default = "default_layout")]
    pub layout: String,
}

fn default_element_type() -> String {
    "float32".to_string()
}

fn default_layout() -> String {
    "row_major".to_string()
}

impl PolymerExport {
    /// Load from a JSON file
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self, String> {
        let content =
            fs::read_to_string(path).map_err(|e| format!("Failed to read file: {}", e))?;

        serde_json::from_str(&content).map_err(|e| format!("Failed to parse JSON: {}", e))
    }

    /// Load from ISL text files (alternative format)
    pub fn from_isl_files(schedule_file: &Path, access_file: &Path) -> Result<Self, String> {
        let schedule_content = fs::read_to_string(schedule_file)
            .map_err(|e| format!("Failed to read schedule file: {}", e))?;

        let access_content = fs::read_to_string(access_file)
            .map_err(|e| format!("Failed to read access file: {}", e))?;

        // Parse ISL format files
        Self::parse_isl_format(&schedule_content, &access_content)
    }

    /// Parse ISL text format (used by some Polymer configurations)
    fn parse_isl_format(schedule_str: &str, access_str: &str) -> Result<Self, String> {
        // Extract schedule tree
        let tree = schedule_str.trim().to_string();

        // Parse access relations (format: stmt_name -> array[indices])
        let mut access_relations = Vec::new();

        for line in access_str.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }

            // Parse lines like: "S0 read A { S0[i,j] -> A[i,k] }"
            if let Some((stmt_part, rest)) = line.split_once(' ') {
                let parts: Vec<&str> = rest.splitn(3, ' ').collect();
                if parts.len() >= 3 {
                    access_relations.push(AccessRelation {
                        stmt_id: stmt_part.to_string(),
                        r#type: parts[0].to_string(),
                        array: parts[1].to_string(),
                        relation: parts[2].to_string(),
                        is_may_access: false,
                    });
                }
            }
        }

        // Extract domain from schedule
        let domain = Self::extract_domain_from_schedule(&tree)?;

        Ok(PolymerExport {
            schedule: ScheduleExport {
                tree,
                domain,
                constraints: None,
            },
            access_relations,
            arrays: Vec::new(),
            context: None,
        })
    }

    /// Extract domain from schedule tree string
    fn extract_domain_from_schedule(schedule_str: &str) -> Result<String, String> {
        // Look for domain specification in schedule
        if let Some(start) = schedule_str.find("domain:") {
            let domain_part = &schedule_str[start + 7..];
            if let Some(end) = domain_part.find('}') {
                return Ok(format!("{}}}", &domain_part[..=end]));
            }
        }

        // If no explicit domain, try to extract from schedule structure
        if schedule_str.contains('{') && schedule_str.contains('}') {
            return Ok(schedule_str.to_string());
        }

        Err("Could not extract domain from schedule".to_string())
    }

    /// Convert to PolySat AccessInfo structure
    pub fn to_polysat_access_info(&self) -> Result<AccessInfo, String> {
        // Create opaque handles for ISL objects
        let ctx = ContextHandle {
            _inner: Arc::new(()),
        };
        let schedule = ScheduleHandle {
            _inner: Arc::new(()),
        };

        // Create AccessInfo
        let mut access_info = AccessInfo::new(ctx, schedule);

        // Group access relations by statement
        let mut stmt_map: HashMap<String, StmtAccess> = HashMap::new();

        for rel in &self.access_relations {
            let stmt = stmt_map
                .entry(rel.stmt_id.clone())
                .or_insert_with(|| StmtAccess::new(rel.stmt_id.clone()));

            // In real implementation, would parse ISL relation
            // For now, using placeholder handles
            match rel.r#type.as_str() {
                "read" => {
                    if rel.is_may_access {
                        stmt.may_reads = Some(AccessMapHandle {
                            _inner: Arc::new(()),
                        });
                    } else {
                        stmt.reads = AccessMapHandle {
                            _inner: Arc::new(()),
                        };
                    }
                }
                "write" => {
                    if rel.is_may_access {
                        stmt.may_writes = Some(AccessMapHandle {
                            _inner: Arc::new(()),
                        });
                    } else {
                        stmt.writes = AccessMapHandle {
                            _inner: Arc::new(()),
                        };
                    }
                }
                _ => return Err(format!("Unknown access type: {}", rel.r#type)),
            }
        }

        // In real implementation, would parse domain
        // For now, assign placeholder domains
        for (_stmt_name, stmt) in stmt_map.iter_mut() {
            stmt.domain = DomainHandle {
                _inner: Arc::new(()),
            };
        }

        // Add all statements to AccessInfo
        for stmt in stmt_map.into_values() {
            access_info.add_statement(stmt);
        }

        // Process array metadata
        for array_meta in &self.arrays {
            let array_info = ArrayInfo {
                name: array_meta.name.clone(),
                dimensions: array_meta.dimensions,
                sizes: array_meta.sizes.clone(),
                element_type: Self::parse_element_type(&array_meta.element_type),
                layout: Self::parse_layout(&array_meta.layout),
            };
            access_info.add_array(array_info);
        }

        // Validate the imported access info
        access_info.validate_accesses()?;

        Ok(access_info)
    }

    /// Parse element type string to DataType enum
    fn parse_element_type(type_str: &str) -> DataType {
        match type_str.to_lowercase().as_str() {
            "float32" | "f32" | "float" => DataType::Float32,
            "float64" | "f64" | "double" => DataType::Float64,
            "int32" | "i32" | "int" => DataType::Int32,
            "int64" | "i64" | "long" => DataType::Int64,
            "bool" | "boolean" => DataType::Bool,
            _ => DataType::Float32, // Default
        }
    }

    /// Parse layout string to MemoryLayout enum
    fn parse_layout(layout_str: &str) -> MemoryLayout {
        match layout_str.to_lowercase().as_str() {
            "row_major" | "row-major" | "c" => MemoryLayout::RowMajor,
            "column_major" | "column-major" | "fortran" => MemoryLayout::ColumnMajor,
            _ if layout_str.starts_with("tiled") => {
                // Parse tiled layout like "tiled_32x32"
                if let Some(sizes) = layout_str.strip_prefix("tiled_") {
                    if let Some((w, h)) = sizes.split_once('x') {
                        if let (Ok(width), Ok(height)) = (w.parse(), h.parse()) {
                            return MemoryLayout::Tiled {
                                tile_size: [width, height],
                            };
                        }
                    }
                }
                MemoryLayout::Custom
            }
            _ => MemoryLayout::Custom,
        }
    }

    /// Export AccessInfo back to Polymer format
    pub fn from_access_info(access_info: &AccessInfo) -> Self {
        let mut access_relations = Vec::new();
        let mut arrays = Vec::new();

        // Convert statement accesses
        for (stmt_id, stmt) in &access_info.stmt_accesses {
            // Convert read accesses
            // In real implementation, would check if access map is empty
            access_relations.push(AccessRelation {
                stmt_id: stmt_id.clone(),
                r#type: "read".to_string(),
                array: "unknown".to_string(), // Placeholder
                relation: "placeholder".to_string(),
                is_may_access: false,
            });

            // Convert may-read accesses
            if let Some(_may_reads) = &stmt.may_reads {
                access_relations.push(AccessRelation {
                    stmt_id: stmt_id.clone(),
                    r#type: "read".to_string(),
                    array: "unknown".to_string(), // Placeholder
                    relation: "placeholder".to_string(),
                    is_may_access: true,
                });
            }

            // Convert write accesses
            // In real implementation, would check if access map is empty
            access_relations.push(AccessRelation {
                stmt_id: stmt_id.clone(),
                r#type: "write".to_string(),
                array: "unknown".to_string(), // Placeholder
                relation: "placeholder".to_string(),
                is_may_access: false,
            });

            // Convert may-write accesses
            if let Some(_may_writes) = &stmt.may_writes {
                access_relations.push(AccessRelation {
                    stmt_id: stmt_id.clone(),
                    r#type: "write".to_string(),
                    array: "unknown".to_string(), // Placeholder
                    relation: "placeholder".to_string(),
                    is_may_access: true,
                });
            }
        }

        // Convert array metadata
        for (name, array) in &access_info.arrays {
            arrays.push(ArrayMetadata {
                name: name.clone(),
                dimensions: array.dimensions,
                sizes: array.sizes.clone(),
                element_type: Self::format_element_type(array.element_type),
                layout: Self::format_layout(array.layout),
            });
        }

        PolymerExport {
            schedule: ScheduleExport {
                tree: "placeholder_schedule".to_string(), // Placeholder
                domain: "placeholder_domain".to_string(), // Placeholder
                constraints: None,
            },
            access_relations,
            arrays,
            context: None,
        }
    }

    /// Extract array name from an access relation
    #[allow(dead_code)]
    fn extract_array_name(_access_map: &AccessMapHandle) -> String {
        // In real implementation, would extract from ISL map
        "unknown".to_string()
    }

    /// Format DataType for export
    fn format_element_type(dtype: DataType) -> String {
        match dtype {
            DataType::Float32 => "float32".to_string(),
            DataType::Float64 => "float64".to_string(),
            DataType::Int32 => "int32".to_string(),
            DataType::Int64 => "int64".to_string(),
            DataType::Bool => "bool".to_string(),
            DataType::Custom(size) => format!("custom_{}", size),
        }
    }

    /// Format MemoryLayout for export
    fn format_layout(layout: MemoryLayout) -> String {
        match layout {
            MemoryLayout::RowMajor => "row_major".to_string(),
            MemoryLayout::ColumnMajor => "column_major".to_string(),
            MemoryLayout::Tiled { tile_size: [w, h] } => format!("tiled_{}x{}", w, h),
            MemoryLayout::Custom => "custom".to_string(),
        }
    }

    /// Save to JSON file
    pub fn save_to_file<P: AsRef<Path>>(&self, path: P) -> Result<(), String> {
        let json = serde_json::to_string_pretty(self)
            .map_err(|e| format!("Failed to serialize: {}", e))?;

        fs::write(path, json).map_err(|e| format!("Failed to write file: {}", e))?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_element_type() {
        assert!(matches!(
            PolymerExport::parse_element_type("float32"),
            DataType::Float32
        ));
        assert!(matches!(
            PolymerExport::parse_element_type("double"),
            DataType::Float64
        ));
        assert!(matches!(
            PolymerExport::parse_element_type("int"),
            DataType::Int32
        ));
    }

    #[test]
    fn test_parse_layout() {
        assert!(matches!(
            PolymerExport::parse_layout("row_major"),
            MemoryLayout::RowMajor
        ));
        assert!(matches!(
            PolymerExport::parse_layout("column-major"),
            MemoryLayout::ColumnMajor
        ));

        if let MemoryLayout::Tiled { tile_size } = PolymerExport::parse_layout("tiled_32x64") {
            assert_eq!(tile_size, [32, 64]);
        } else {
            panic!("Expected tiled layout");
        }
    }

    #[test]
    fn test_polymer_export_roundtrip() {
        let export = PolymerExport {
            schedule: ScheduleExport {
                tree: "{ domain: \"{ S[i,j]: 0 <= i,j < 100 }\" }".to_string(),
                domain: "{ S[i,j]: 0 <= i,j < 100 }".to_string(),
                constraints: None,
            },
            access_relations: vec![
                AccessRelation {
                    stmt_id: "S".to_string(),
                    r#type: "read".to_string(),
                    array: "A".to_string(),
                    relation: "{ S[i,j] -> A[i,j] }".to_string(),
                    is_may_access: false,
                },
                AccessRelation {
                    stmt_id: "S".to_string(),
                    r#type: "write".to_string(),
                    array: "B".to_string(),
                    relation: "{ S[i,j] -> B[j,i] }".to_string(),
                    is_may_access: false,
                },
            ],
            arrays: vec![
                ArrayMetadata {
                    name: "A".to_string(),
                    dimensions: 2,
                    sizes: vec![Some(100), Some(100)],
                    element_type: "float32".to_string(),
                    layout: "row_major".to_string(),
                },
                ArrayMetadata {
                    name: "B".to_string(),
                    dimensions: 2,
                    sizes: vec![Some(100), Some(100)],
                    element_type: "float32".to_string(),
                    layout: "column_major".to_string(),
                },
            ],
            context: None,
        };

        // Test serialization
        let json = serde_json::to_string(&export).unwrap();
        let parsed: PolymerExport = serde_json::from_str(&json).unwrap();

        assert_eq!(parsed.access_relations.len(), 2);
        assert_eq!(parsed.arrays.len(), 2);
        assert_eq!(parsed.schedule.domain, export.schedule.domain);
    }
}
