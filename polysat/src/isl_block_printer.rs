// FFI bindings to get ISL schedule strings in block (tree) format
// This preserves the full schedule tree structure that to_str() loses

use isl_rs::Schedule;
use libc::uintptr_t;
use std::ffi::CStr;
use std::os::raw::{c_char, c_int};

// ISL printer is an opaque pointer type
type IslPrinter = uintptr_t;
type IslCtx = uintptr_t;
type IslSchedule = uintptr_t;

#[repr(C)]
#[derive(Debug, Clone, Copy)]
enum IslYamlStyle {
    Block = 0,
    #[allow(dead_code)]
    Flow = 1,
}

extern "C" {
    // Printer creation and configuration
    fn isl_printer_to_str(ctx: IslCtx) -> IslPrinter;
    fn isl_printer_set_yaml_style(p: IslPrinter, yaml_style: c_int) -> IslPrinter;
    fn isl_printer_free(printer: IslPrinter) -> IslPrinter;
    fn isl_printer_get_str(printer: IslPrinter) -> *mut c_char;

    // Schedule printing
    fn isl_printer_print_schedule(p: IslPrinter, schedule: IslSchedule) -> IslPrinter;

    // Get context from schedule
    fn isl_schedule_get_ctx(schedule: IslSchedule) -> IslCtx;
}

/// Get the schedule as a block-style YAML string that preserves tree structure
///
/// Performance Note: This function is called frequently during e-graph exploration
/// (once per ScheduleHandle creation). All debug output has been removed to avoid
/// performance bottlenecks. Use `log::debug!` with appropriate log levels if debugging
/// is needed.
pub fn schedule_to_block_str(schedule: &Schedule) -> String {
    unsafe {
        // Access the raw schedule pointer from isl-rs
        // The Schedule struct has a `ptr: uintptr_t` field
        let schedule_ptr = schedule.ptr;

        // Get the context
        let ctx_ptr = isl_schedule_get_ctx(schedule_ptr);

        // Create a string printer
        let mut printer = isl_printer_to_str(ctx_ptr);

        // Set to block style (preserves tree structure)
        printer = isl_printer_set_yaml_style(printer, IslYamlStyle::Block as c_int);

        // Print the schedule
        printer = isl_printer_print_schedule(printer, schedule_ptr);

        // Get the string
        let c_str = isl_printer_get_str(printer);
        let result = if c_str.is_null() {
            // Fallback to simple to_str() if block printer fails
            schedule.to_str().to_string()
        } else {
            CStr::from_ptr(c_str).to_string_lossy().into_owned()
        };

        // Clean up (returns null on success)
        isl_printer_free(printer);

        result
    }
}

/// Alternative: Use isl_schedule_dump to capture output (for debugging)
pub fn schedule_dump_to_string(schedule: &Schedule) -> String {
    // This would require capturing stderr, which is complex
    // For now, just use the FFI approach above
    schedule_to_block_str(schedule)
}
