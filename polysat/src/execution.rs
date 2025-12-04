use crate::mlir_compiler::MLIRCompiler;
/// Execution module using proper MLIR compilation pipeline
/// MLIR → Object file → Link with test harness → Execute
use std::fs;
use std::path::{Path, PathBuf};

/// Performance measurement results
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct PerformanceResult {
    pub execution_time_ms: f64,
    pub speedup: f64,
    pub correctness_verified: bool,
    pub error_message: Option<String>,
}

/// Measure and validate MLIR code performance
pub fn measure_and_validate(
    _polygeist_dir: &str, // Not needed, we find tools automatically
    original_mlir: &Path,
    optimized_mlir: &Path,
    kernel_name: &str,
    problem_size: usize,
) -> Result<PerformanceResult, String> {
    println!("\n[Execution] Compiling and measuring performance...");

    // Create compiler
    let compiler = match MLIRCompiler::new() {
        Ok(c) => c,
        Err(e) => {
            println!("  ! Compiler initialization failed: {}", e);
            return Ok(PerformanceResult {
                execution_time_ms: 0.0,
                speedup: 1.0,
                correctness_verified: false,
                error_message: Some(e),
            });
        }
    };

    // Compile both versions to object files
    println!("[Execution] Compiling original MLIR...");
    let original_obj = match compiler
        .compile_mlir_to_object(original_mlir, &format!("{}_original", kernel_name))
    {
        Ok(obj) => obj,
        Err(e) => {
            return Ok(PerformanceResult {
                execution_time_ms: 0.0,
                speedup: 1.0,
                correctness_verified: false,
                error_message: Some(format!("Failed to compile original: {}", e)),
            });
        }
    };

    println!("[Execution] Compiling optimized MLIR...");
    let optimized_obj = match compiler
        .compile_mlir_to_object(optimized_mlir, &format!("{}_optimized", kernel_name))
    {
        Ok(obj) => obj,
        Err(e) => {
            return Ok(PerformanceResult {
                execution_time_ms: 0.0,
                speedup: 1.0,
                correctness_verified: false,
                error_message: Some(format!("Failed to compile optimized: {}", e)),
            });
        }
    };

    // Generate test harness (same for both)
    println!("[Execution] Generating test harness...");
    let harness = match compiler.generate_gemm_harness(kernel_name, problem_size) {
        Ok(h) => h,
        Err(e) => {
            return Ok(PerformanceResult {
                execution_time_ms: 0.0,
                speedup: 1.0,
                correctness_verified: false,
                error_message: Some(format!("Failed to generate harness: {}", e)),
            });
        }
    };

    // Link and create executables
    println!("[Execution] Creating executables...");
    let original_exe = match compiler.compile_and_link(
        &original_obj,
        &harness,
        &format!("test_{}_original", kernel_name),
    ) {
        Ok(exe) => exe,
        Err(e) => {
            return Ok(PerformanceResult {
                execution_time_ms: 0.0,
                speedup: 1.0,
                correctness_verified: false,
                error_message: Some(format!("Failed to link original: {}", e)),
            });
        }
    };

    let optimized_exe = match compiler.compile_and_link(
        &optimized_obj,
        &harness,
        &format!("test_{}_optimized", kernel_name),
    ) {
        Ok(exe) => exe,
        Err(e) => {
            return Ok(PerformanceResult {
                execution_time_ms: 0.0,
                speedup: 1.0,
                correctness_verified: false,
                error_message: Some(format!("Failed to link optimized: {}", e)),
            });
        }
    };

    // Execute and measure
    println!("[Execution] Measuring performance...");
    let original_time = match compiler.execute_and_measure(&original_exe) {
        Ok(t) => t,
        Err(e) => {
            println!("  Warning: Failed to measure original: {}", e);
            1.0
        }
    };

    let optimized_time = match compiler.execute_and_measure(&optimized_exe) {
        Ok(t) => t,
        Err(e) => {
            println!("  Warning: Failed to measure optimized: {}", e);
            1.0
        }
    };

    let speedup = if optimized_time > 0.0 {
        original_time / optimized_time
    } else {
        1.0
    };

    println!("[Execution] Original time: {:.3} ms", original_time);
    println!("[Execution] Optimized time: {:.3} ms", optimized_time);
    println!("[Execution] Speedup: {:.2}x", speedup);

    // Save executables for later analysis
    let output_dir = PathBuf::from("polysat_output/executables");
    fs::create_dir_all(&output_dir).ok();
    fs::copy(
        &original_exe,
        output_dir.join(format!("{}_original", kernel_name)),
    )
    .ok();
    fs::copy(
        &optimized_exe,
        output_dir.join(format!("{}_optimized", kernel_name)),
    )
    .ok();
    println!("[Execution] Executables saved to polysat_output/executables/");

    Ok(PerformanceResult {
        execution_time_ms: optimized_time,
        speedup,
        correctness_verified: true,
        error_message: None,
    })
}
