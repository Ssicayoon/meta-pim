// Direct FFI bindings to ISL AST generation functions
// This is necessary because isl-rs does not expose isl_ast_build functionality

use std::ffi::CStr;
use std::os::raw::{c_char, c_int, c_void};

// Opaque types
#[repr(C)]
#[allow(non_camel_case_types)]
pub struct isl_ctx {
    _private: [u8; 0],
}
#[repr(C)]
#[allow(non_camel_case_types)]
pub struct isl_ast_build {
    _private: [u8; 0],
}
#[repr(C)]
#[allow(non_camel_case_types)]
pub struct isl_ast_node {
    _private: [u8; 0],
}
#[repr(C)]
#[allow(non_camel_case_types)]
pub struct isl_printer {
    _private: [u8; 0],
}
#[repr(C)]
#[allow(non_camel_case_types)]
pub struct isl_schedule {
    _private: [u8; 0],
}

#[allow(clashing_extern_declarations)]
extern "C" {
    // Context
    #[allow(dead_code)]
    fn isl_ctx_alloc() -> *mut isl_ctx;
    #[allow(dead_code)]
    fn isl_ctx_free(ctx: *mut isl_ctx);

    // AST Build
    fn isl_ast_build_alloc(ctx: *mut isl_ctx) -> *mut isl_ast_build;
    fn isl_ast_build_free(build: *mut isl_ast_build) -> *mut isl_ast_build;
    fn isl_ast_build_node_from_schedule(
        build: *mut isl_ast_build,
        schedule: *mut isl_schedule,
    ) -> *mut isl_ast_node;

    // Printer
    fn isl_printer_to_str(ctx: *mut isl_ctx) -> *mut isl_printer;
    fn isl_printer_free(printer: *mut isl_printer) -> *mut isl_printer;
    fn isl_printer_get_str(printer: *mut isl_printer) -> *const c_char;
    fn isl_printer_set_output_format(
        printer: *mut isl_printer,
        output_format: c_int,
    ) -> *mut isl_printer;
    fn isl_printer_print_ast_node(
        printer: *mut isl_printer,
        node: *mut isl_ast_node,
    ) -> *mut isl_printer;

    // Schedule (needed for copy)
    fn isl_schedule_copy(schedule: *mut isl_schedule) -> *mut isl_schedule;

    // AST Node
    fn isl_ast_node_free(node: *mut isl_ast_node) -> *mut isl_ast_node;
}

pub const ISL_FORMAT_C: c_int = 4;

/// Generate C code from an ISL schedule using direct FFI
pub unsafe fn generate_c_code(
    ctx_ptr: *mut c_void,
    schedule_ptr: *mut c_void,
) -> Result<String, String> {
    let ctx = ctx_ptr as *mut isl_ctx;
    let schedule = schedule_ptr as *mut isl_schedule;

    if ctx.is_null() || schedule.is_null() {
        return Err("Null context or schedule pointer".to_string());
    }

    // 1. Create AST build
    let build = isl_ast_build_alloc(ctx);
    if build.is_null() {
        return Err("Failed to allocate isl_ast_build".to_string());
    }

    // 2. Generate AST node from schedule
    // We must copy the schedule because isl_ast_build_node_from_schedule consumes it
    let schedule_copy = isl_schedule_copy(schedule);
    let ast_node = isl_ast_build_node_from_schedule(build, schedule_copy);

    // Free the build object as it's no longer needed (it doesn't own the node)
    isl_ast_build_free(build);

    if ast_node.is_null() {
        return Err("Failed to generate AST node".to_string());
    }

    // 3. Print AST node to string (as C code)
    let printer = isl_printer_to_str(ctx);
    if printer.is_null() {
        isl_ast_node_free(ast_node);
        return Err("Failed to allocate printer".to_string());
    }

    let printer = isl_printer_set_output_format(printer, ISL_FORMAT_C);
    let printer = isl_printer_print_ast_node(printer, ast_node);

    let c_str = isl_printer_get_str(printer);
    let result = if c_str.is_null() {
        Err("Failed to get string from printer".to_string())
    } else {
        let rust_str = CStr::from_ptr(c_str).to_string_lossy().to_string();
        Ok(rust_str)
    };

    // Cleanup
    isl_printer_free(printer);
    isl_ast_node_free(ast_node);

    result
}
