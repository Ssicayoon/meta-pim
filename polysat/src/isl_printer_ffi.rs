// Direct FFI bindings to ISL printer functions for proper schedule tree serialization
// This bypasses the limitations of isl-rs bindings

use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_int, c_void};

// Opaque types for ISL objects
#[repr(C)]
struct isl_ctx { _private: [u8; 0] }
#[repr(C)]
struct isl_printer { _private: [u8; 0] }
#[repr(C)]
struct isl_schedule { _private: [u8; 0] }

// ISL printer output formats
#[repr(C)]
enum isl_format {
    ISL_FORMAT_ISL = 0,
    ISL_FORMAT_POLYLIB = 1,
    ISL_FORMAT_POLYLIB_CONSTRAINTS = 2,
    ISL_FORMAT_OMEGA = 3,
    ISL_FORMAT_C = 4,
    ISL_FORMAT_LATEX = 5,
    ISL_FORMAT_EXT_POLYLIB = 6,
}

// ISL YAML styles
#[repr(C)]
enum isl_yaml_style {
    ISL_YAML_STYLE_BLOCK = 0,
    ISL_YAML_STYLE_FLOW = 1,
}

// FFI function declarations
extern "C" {
    // Printer creation and destruction
    fn isl_printer_to_str(ctx: *mut isl_ctx) -> *mut isl_printer;
    fn isl_printer_free(printer: *mut isl_printer) -> *mut isl_printer;
    fn isl_printer_get_str(printer: *mut isl_printer) -> *const c_char;
    
    // Printer configuration
    fn isl_printer_set_output_format(printer: *mut isl_printer, output_format: c_int) -> *mut isl_printer;
    fn isl_printer_set_yaml_style(printer: *mut isl_printer, yaml_style: c_int) -> *mut isl_printer;
    
    // Schedule printing
    fn isl_printer_print_schedule(printer: *mut isl_printer, schedule: *mut isl_schedule) -> *mut isl_printer;
    
    // Schedule reference counting
    fn isl_schedule_copy(schedule: *mut isl_schedule) -> *mut isl_schedule;
}

/// Print an ISL schedule in YAML tree format using direct FFI
/// This preserves the full tree structure including all transformations
pub unsafe fn print_schedule_tree_ffi(ctx_ptr: *mut c_void, schedule_ptr: *mut c_void) -> Result<String, String> {
    // Cast to proper types
    let ctx = ctx_ptr as *mut isl_ctx;
    let schedule = schedule_ptr as *mut isl_schedule;
    
    if ctx.is_null() || schedule.is_null() {
        return Err("Null context or schedule pointer".to_string());
    }
    
    // Create a string printer
    let printer = isl_printer_to_str(ctx);
    if printer.is_null() {
        return Err("Failed to create ISL printer".to_string());
    }
    
    // Set YAML block style for tree-like output
    let printer = isl_printer_set_yaml_style(printer, isl_yaml_style::ISL_YAML_STYLE_BLOCK as c_int);
    let printer = isl_printer_set_output_format(printer, isl_format::ISL_FORMAT_ISL as c_int);
    
    // Print the schedule
    let printer = isl_printer_print_schedule(printer, schedule);
    
    // Get the string result
    let c_str = isl_printer_get_str(printer);
    let result = if c_str.is_null() {
        Err("Failed to get string from printer".to_string())
    } else {
        let rust_str = CStr::from_ptr(c_str).to_string_lossy().to_string();
        Ok(rust_str)
    };
    
    // Free the printer
    isl_printer_free(printer);
    
    result
}

/// Alternative approach: Use ISL's internal schedule dump function
pub fn dump_schedule_internal(schedule: &isl_rs::Schedule) -> String {
    // Since we can't easily get raw pointers from isl-rs Schedule,
    // we need to work with what we have.
    
    // The issue is that schedule.to_str() already calls some internal ISL function
    // that doesn't preserve the tree. We need to find another way.
    
    // For now, let's try to reconstruct the tree from the schedule object
    reconstruct_tree_from_schedule(schedule)
}

/// Reconstruct the tree structure from a schedule
fn reconstruct_tree_from_schedule(schedule: &isl_rs::Schedule) -> String {
    use isl_rs::ScheduleNodeType;
    
    // Get the root node
    let root = schedule.get_root();
    
    // Start building the tree representation
    let mut result = String::new();
    result.push_str("{ ");
    
    // Try to get the domain first
    // The schedule should have a domain at the root or close to it
    let domain = schedule.get_domain();
    let domain_str = domain.to_str();
    if !domain_str.is_empty() {
        result.push_str(&format!("domain: \"{}\", child: {{ ", domain_str));
    }
    
    // Get the map representation - this might contain band information
    let map = schedule.get_map();
    let map_str = map.to_str();
    
    // Parse the map to extract band structure
    // Maps typically look like: { S[i,j] -> [i,j] } or with transformations: { S[i,j] -> [(i - i mod 32), (i mod 32), j] }
    if map_str.contains("mod") {
        // This indicates tiling has been applied
        result.push_str("schedule: \"");
        result.push_str(&extract_band_from_map(&map_str));
        result.push_str("\"");
    } else {
        // Basic schedule without transformations
        result.push_str("schedule: \"");
        result.push_str(&map_str);
        result.push_str("\"");
    }
    
    // Close the tree
    if !domain_str.is_empty() {
        result.push_str(" }");
    }
    result.push_str(" }");
    
    result
}

/// Extract band information from a map string
fn extract_band_from_map(map_str: &str) -> String {
    // Try to extract the scheduling dimensions from the map
    // Example: { S[i,j] -> [(i - i mod 32), (i mod 32), j] } 
    // Should become something like: L0[{ S[i,j] -> [(i - i mod 32), (i mod 32), j] }]
    
    if let Some(arrow_pos) = map_str.find("->") {
        let after_arrow = &map_str[arrow_pos + 2..];
        if let Some(bracket_start) = after_arrow.find('[') {
            let schedule_part = &after_arrow[bracket_start..];
            return format!("L0[{{ S -> {} }}]", schedule_part);
        }
    }
    
    // Fallback: return the map as-is
    map_str.to_string()
}