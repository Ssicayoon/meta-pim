//! Codegen Viability Tests for RFC001 Phase 3
//!
//! This file verifies that ScheduleHandles can be lowered to valid C code
//! using ISL's AST generation and printing facilities.

use isl_rs::{Context, MultiUnionPwAff, MultiVal, Schedule, UnionSet, Val, ValList};
use polysat::isl_codegen_ffi;

use std::io::Write;
use std::process::Command;
use tempfile::NamedTempFile;

// Helper to verify C code compilation
fn verify_c_compilation(c_code: &str) {
    // Wrap the code in a function and add dummy macros for statement instances
    // We assume statement macros like S0, S1, etc. might be present.
    // A simple way is to define a catch-all or just specific ones if we know them.
    // For these tests, we know S0 is used.
    let wrapped_code = format!(
        r#"
#define S0(x) (void)(x)
void kernel() {{
{}
}}
"#,
        c_code
    );

    // Create a temporary file for the C code
    let mut file = NamedTempFile::new().expect("Failed to create temp file");
    write!(file, "{}", wrapped_code).expect("Failed to write C code");

    // Attempt to compile with cc (default C compiler)
    // -x c: treat input as C
    // -c: compile only (don't link)
    // -o /dev/null: discard output
    // -: read from stdin (if we piped, but here we use file path)
    let output = Command::new("cc")
        .arg("-x")
        .arg("c")
        .arg("-c")
        .arg("-o")
        .arg("/dev/null")
        .arg(file.path())
        .output()
        .expect("Failed to execute C compiler");

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        panic!("C compilation failed:\n{}", stderr);
    }
}

// Helper to create a tiled schedule (copied from equivalence.rs for independence)
fn create_tiled_16(ctx: &Context) -> Option<Schedule> {
    // Step 1: Create schedule from domain
    let domain = UnionSet::read_from_str(ctx, "{ S0[i] : 0 <= i <= 255 }");
    let schedule = Schedule::from_domain(domain);

    // Step 2: Insert identity partial schedule to create a band node
    let partial = MultiUnionPwAff::read_from_str(ctx, "[{ S0[i] -> [(i)] }]");
    let root = schedule.get_root();

    if root.n_children() == 0 {
        return None;
    }
    let child = root.child(0);
    let band_node = child.insert_partial_schedule(partial);

    // Step 3: Create tile size and apply tiling
    let band_space = band_node.band_get_space();
    let tile_val = Val::int_from_si(ctx, 16);
    let tile_sizes = MultiVal::from_val_list(band_space, ValList::from_val(tile_val));

    // Apply tiling
    let tiled = band_node.band_tile(tile_sizes);
    Some(tiled.get_schedule())
}

#[test]
fn test_codegen_tiled_schedule() {
    let ctx = Context::alloc();
    let schedule = create_tiled_16(&ctx).expect("Failed to create tiled schedule");

    // Generate C code using our FFI bindings
    // We need raw pointers for the FFI
    let ctx_ptr = schedule.get_ctx().ptr as *mut std::ffi::c_void;
    let schedule_ptr = schedule.ptr as *mut std::ffi::c_void;

    let c_code = unsafe {
        isl_codegen_ffi::generate_c_code(ctx_ptr, schedule_ptr).expect("Failed to generate C code")
    };

    println!("Generated C Code:\n{}", c_code);

    // Assertions
    assert!(!c_code.is_empty(), "Generated code should not be empty");
    assert!(c_code.contains("for ("), "Code should contain for loops");
    assert!(
        c_code.contains("int "),
        "Code should contain integer declarations"
    );

    // Check for tiling structure (nested loops)
    // We expect at least two loops for a 1D tiled schedule
    let for_count = c_code.matches("for (").count();
    assert!(
        for_count >= 2,
        "Tiled schedule should have at least 2 loops (outer + inner)"
    );

    // Verify compilation
    verify_c_compilation(&c_code);
}

#[test]
fn test_codegen_structural_vectorization_schedule() {
    let ctx = Context::alloc();

    // Create a simple 1D schedule
    let domain = UnionSet::read_from_str(&ctx, "{ S0[i] : 0 <= i <= 127 }");
    let schedule = Schedule::from_domain(domain);
    let partial = MultiUnionPwAff::read_from_str(&ctx, "[{ S0[i] -> [(i)] }]");
    let root = schedule.get_root();
    let child = root.child(0);
    let band_node = child.insert_partial_schedule(partial);
    let schedule = band_node.get_schedule();

    // Apply vectorize (width 4)
    let vectorized = polysat::transformations::vectorize(&schedule, 0, 4, None)
        .unwrap()
        .expect("Vectorization not applicable");

    // Generate C code
    let ctx_ptr = vectorized.get_ctx().ptr as *mut std::ffi::c_void;
    let schedule_ptr = vectorized.ptr as *mut std::ffi::c_void;

    let c_code = unsafe {
        isl_codegen_ffi::generate_c_code(ctx_ptr, schedule_ptr).expect("Failed to generate C code")
    };

    println!("Generated Vectorized Code:\n{}", c_code);

    // Assertions
    assert!(!c_code.is_empty());
    assert!(c_code.contains("for ("));

    // Note: Since we only did strip-mining, we don't expect explicit vector intrinsics yet
    // unless ISL auto-detects it or we set build options.
    // But we should see the strip-mined structure (loops with step 4 or mod 4 logic).
    // ISL AST generation usually simplifies loop bounds, so we might see:
    // for (int c0 = 0; c0 <= 127; c0 += 4)
    //   for (int c1 = c0; c1 <= c0 + 3; c1 += 1)

    // Let's check for the loop structure
    let for_count = c_code.matches("for (").count();
    assert!(
        for_count >= 2,
        "Vectorized (strip-mined) schedule should have loops"
    );

    // Verify compilation
    verify_c_compilation(&c_code);
}
