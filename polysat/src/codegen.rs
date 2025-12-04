use crate::execution::{measure_and_validate, PerformanceResult};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Arc;

/// Interface with Polygeist for complete C -> MLIR -> ISL -> PolySat -> MLIR pipeline
/// Pipeline: C -> Polygeist -> ISL schedule -> PolySat transformations -> transformed ISL -> Polygeist -> optimized MLIR

/// Helper function to compile C to MLIR
pub fn compile_c_to_mlir(
    polygeist_dir: &str,
    c_file: &str,
    kernel_name: &str,
    output_mlir: &str,
) -> Result<(), String> {
    // Get absolute path for polygeist_dir if relative
    let polygeist_path = if polygeist_dir.starts_with("/") {
        polygeist_dir.to_string()
    } else {
        let cwd = std::env::current_dir()
            .map_err(|e| format!("Failed to get current directory: {}", e))?;
        cwd.join(polygeist_dir).to_string_lossy().to_string()
    };

    let cgeist = format!("{}/build/bin/cgeist", polygeist_path);

    let output = Command::new(&cgeist)
        .arg(c_file)
        .arg(format!("-function={}", kernel_name))
        .arg("-raise-scf-to-affine")
        .arg("-memref-fullrank")
        .arg("-S")
        .arg("-o")
        .arg(output_mlir)
        .output()
        .map_err(|e| format!("Failed to run cgeist: {}", e))?;

    if !output.status.success() {
        return Err(format!(
            "cgeist failed: {}",
            String::from_utf8_lossy(&output.stderr)
        ));
    }

    Ok(())
}

/// Convert C to MLIR using cgeist (alias for backwards compatibility)
pub fn c_to_mlir(
    polygeist_dir: &str,
    c_file: &str,
    kernel_name: &str,
    output_mlir: &str,
) -> Result<(), String> {
    compile_c_to_mlir(polygeist_dir, c_file, kernel_name, output_mlir)
}

pub struct BaselineResult {
    pub mlir_file: PathBuf,
    pub schedule_file: PathBuf,
    pub access_file: PathBuf,
}

/// Helper to extract just the baseline schedule
pub fn extract_baseline_schedule(
    polygeist_dir: &str,
    c_file: &str,
    kernel: &str,
    shape: u64,
) -> Result<BaselineResult, String> {
    let output_dir = format!("polysat_schedules/{}_{}_isl", kernel, shape).to_string();
    fs::create_dir_all(&output_dir)
        .map_err(|e| format!("Failed to create output directory: {}", e))?;

    // Compile C to MLIR
    let mlir_file = format!("{}/baseline.mlir", &output_dir);
    compile_c_to_mlir(polygeist_dir, c_file, kernel, &mlir_file)?;

    // Extract ISL schedule and accesses
    let (schedule_str, access_path) = extract_isl_schedule(polygeist_dir, &mlir_file, &output_dir)?;
    let schedule_file = format!("{}/baseline_schedule.isl", output_dir);
    fs::write(&schedule_file, schedule_str)
        .map_err(|e| format!("Failed to write schedule file: {}", e))?;

    Ok(BaselineResult {
        mlir_file: PathBuf::from(mlir_file),
        schedule_file: PathBuf::from(schedule_file),
        access_file: access_path,
    })
}

/// Extract ISL schedule and access relations from MLIR using polygeist-opt
pub fn extract_isl_schedule(
    polygeist_dir: &str,
    mlir_file: &str,
    schedule_dir: &str,
) -> Result<(String, PathBuf), String> {
    // Create schedule directory
    fs::create_dir_all(schedule_dir)
        .map_err(|e| format!("Failed to create schedule directory: {}", e))?;

    // Get absolute path for polygeist_dir if relative
    let polygeist_path = if polygeist_dir.starts_with("/") {
        polygeist_dir.to_string()
    } else {
        let cwd = std::env::current_dir()
            .map_err(|e| format!("Failed to get current directory: {}", e))?;
        cwd.join(polygeist_dir).to_string_lossy().to_string()
    };

    let polygeist_opt = format!("{}/build/bin/polygeist-opt", polygeist_path);

    // 1. Extract Schedule (Standard ISL)
    let output_schedule = Command::new(&polygeist_opt)
        .arg(mlir_file)
        .arg("--polyhedral-opt")
        .arg("--use-polyhedral-optimizer=islexternal")
        .arg(format!("--islexternal-dump-schedules={}", schedule_dir))
        .arg("-o")
        .arg("/dev/null")
        .output()
        .map_err(|e| format!("Failed to run polygeist-opt for schedule: {}", e))?;

    if !output_schedule.status.success() {
        return Err(format!(
            "polygeist-opt schedule extraction failed: {}",
            String::from_utf8_lossy(&output_schedule.stderr)
        ));
    }

    // Read the generated schedule file
    let schedule_file = Path::new(schedule_dir).join("__polygeist_outlined_affine_0");
    if !schedule_file.exists() {
        return Err("No schedule file generated".to_string());
    }

    let schedule_str = fs::read_to_string(&schedule_file)
        .map_err(|e| format!("Failed to read schedule file: {}", e))?;

    // 2. Extract Access Relations (YAML-like format)
    // We run polygeist-opt again with --islexternal-dump-accesses
    // This overwrites the previous dump file, so we must have read it already!
    // Wait, actually it might overwrite __polygeist_outlined_affine_0.
    // So we should rename the first one or read it into memory (which we did).

    let output_access = Command::new(&polygeist_opt)
        .arg(mlir_file)
        .arg("--polyhedral-opt")
        .arg("--use-polyhedral-optimizer=islexternal")
        .arg(format!("--islexternal-dump-schedules={}", schedule_dir)) // Still needed to trigger dump logic
        .arg(format!("--islexternal-dump-accesses={}", schedule_dir)) // Enable access dump
        .arg("-o")
        .arg("/dev/null")
        .output()
        .map_err(|e| format!("Failed to run polygeist-opt for accesses: {}", e))?;

    if !output_access.status.success() {
        return Err(format!(
            "polygeist-opt access extraction failed: {}",
            String::from_utf8_lossy(&output_access.stderr)
        ));
    }

    // The access dump also goes to __polygeist_outlined_affine_0 (or similar)
    // We should rename it to accesses.yaml to avoid confusion
    let access_file_src = Path::new(schedule_dir).join("__polygeist_outlined_affine_0");
    let access_file_dst = Path::new(schedule_dir).join("accesses.yaml");

    if access_file_src.exists() {
        fs::rename(&access_file_src, &access_file_dst)
            .map_err(|e| format!("Failed to save access file: {}", e))?;
    } else {
        return Err("No access file generated".to_string());
    }

    // Restore the original schedule file from memory
    fs::write(&schedule_file, &schedule_str)
        .map_err(|e| format!("Failed to restore schedule file: {}", e))?;

    Ok((schedule_str, access_file_dst))
}

/// Apply transformed ISL schedule to MLIR using Polygeist's islexternal
pub fn apply_schedule_to_mlir(
    polygeist_dir: &str,
    mlir_file: &str,
    schedule_dir: &str,
    transformed_schedule: &str,
    output_mlir: &str,
) -> Result<(), String> {
    // Write transformed schedule with the expected name
    let schedule_file = Path::new(schedule_dir).join("__polygeist_outlined_affine_0");
    fs::write(&schedule_file, transformed_schedule)
        .map_err(|e| format!("Failed to write transformed schedule: {}", e))?;

    // Get absolute path for polygeist_dir if relative
    let polygeist_path = if polygeist_dir.starts_with("/") {
        polygeist_dir.to_string()
    } else {
        let cwd = std::env::current_dir()
            .map_err(|e| format!("Failed to get current directory: {}", e))?;
        cwd.join(polygeist_dir).to_string_lossy().to_string()
    };

    let polygeist_opt = format!("{}/build/bin/polygeist-opt", polygeist_path);

    let output = Command::new(&polygeist_opt)
        .arg(mlir_file)
        .arg("--polyhedral-opt")
        .arg("--use-polyhedral-optimizer=islexternal")
        .arg(format!("--islexternal-import-schedules={}", schedule_dir))
        .arg("-o")
        .arg(output_mlir)
        .output()
        .map_err(|e| format!("Failed to run polygeist-opt: {}", e))?;

    if !output.status.success() {
        return Err(format!(
            "polygeist-opt import failed: {}",
            String::from_utf8_lossy(&output.stderr)
        ));
    }

    Ok(())
}

/// Complete automatic pipeline: C -> MLIR -> ISL -> PolySat -> transformed ISL -> optimized MLIR
pub fn run_complete_pipeline(
    polygeist_dir: &str,
    c_file: &str,
    kernel_name: &str,
    transformations: Vec<(String, Option<usize>, Option<i32>)>, // (transform_type, band_idx, size)
    output_base: &str,
) -> Result<PipelineResult, String> {
    // Create output directory structure
    let output_dir = Path::new(output_base);
    fs::create_dir_all(&output_dir)
        .map_err(|e| format!("Failed to create output directory: {}", e))?;

    let stages_dir = output_dir.join("stages");
    fs::create_dir_all(&stages_dir)
        .map_err(|e| format!("Failed to create stages directory: {}", e))?;

    // Stage 1: C -> MLIR
    println!("[Stage 1/4] Converting C to MLIR...");
    let original_mlir = stages_dir.join("1_original.mlir");
    c_to_mlir(
        polygeist_dir,
        c_file,
        kernel_name,
        original_mlir.to_str().unwrap(),
    )?;
    println!("  - Generated MLIR: {}", original_mlir.display());

    // Stage 2: Extract ISL schedule
    println!("[Stage 2/4] Extracting ISL schedule...");
    let schedule_dir = stages_dir.join("2_isl_schedule");
    let (original_schedule, access_file) = extract_isl_schedule(
        polygeist_dir,
        original_mlir.to_str().unwrap(),
        schedule_dir.to_str().unwrap(),
    )?;
    println!(
        "  - Extracted ISL schedule ({} chars)",
        original_schedule.len()
    );
    println!("  - Extracted access relations: {}", access_file.display());

    // Save original schedule for reference
    let original_schedule_file = output_dir.join("original_schedule.isl");
    fs::write(&original_schedule_file, &original_schedule)
        .map_err(|e| format!("Failed to save original schedule: {}", e))?;

    // Stage 3: Apply PolySat transformations
    println!("[Stage 3/4] Applying PolySat transformations...");
    let mut transformed_schedule = original_schedule.clone();

    for (i, (transform, band_idx, size)) in transformations.iter().enumerate() {
        println!(
            "  - Applying {}{}{}",
            transform,
            band_idx
                .map(|b| format!(" at band {}", b))
                .unwrap_or_default(),
            size.map(|s| format!(" with size {}", s))
                .unwrap_or_default()
        );

        // Apply transformation using our PolySat language
        transformed_schedule =
            apply_polysat_transformation(&transformed_schedule, transform, *band_idx, *size)?;

        // Save intermediate result
        let intermediate_file = stages_dir.join(format!("3_transformed_{}.isl", i + 1));
        fs::write(&intermediate_file, &transformed_schedule)
            .map_err(|e| format!("Failed to save intermediate schedule: {}", e))?;
    }

    // Save final transformed schedule
    let transformed_schedule_file = output_dir.join("transformed_schedule.isl");
    fs::write(&transformed_schedule_file, &transformed_schedule)
        .map_err(|e| format!("Failed to save transformed schedule: {}", e))?;
    println!("  - Transformations complete");

    // Stage 4: Apply transformed schedule to generate optimized MLIR
    println!("[Stage 4/4] Generating optimized MLIR...");
    let import_dir = stages_dir.join("4_import");
    fs::create_dir_all(&import_dir)
        .map_err(|e| format!("Failed to create import directory: {}", e))?;

    let optimized_mlir = output_dir.join("optimized.mlir");
    apply_schedule_to_mlir(
        polygeist_dir,
        original_mlir.to_str().unwrap(),
        import_dir.to_str().unwrap(),
        &transformed_schedule,
        optimized_mlir.to_str().unwrap(),
    )?;
    println!("  - Generated optimized MLIR: {}", optimized_mlir.display());

    // Create summary
    let summary = PipelineResult {
        c_file: c_file.to_string(),
        kernel_name: kernel_name.to_string(),
        output_dir: output_dir.to_path_buf(),
        original_mlir: original_mlir.clone(),
        optimized_mlir: optimized_mlir.clone(),
        original_schedule: original_schedule_file,
        transformed_schedule: transformed_schedule_file,
        transformations: transformations.clone(),
    };

    // Stage 5: Measure performance and validate correctness
    println!("\n[Stage 5/5] Measuring performance and validating correctness...");
    let performance = measure_and_validate(
        polygeist_dir,
        &original_mlir,
        &optimized_mlir,
        kernel_name,
        64, // Default problem size for GEMM
    )
    .unwrap_or_else(|e| {
        println!("  ! Performance measurement failed: {}", e);
        PerformanceResult {
            execution_time_ms: 0.0,
            speedup: 1.0,
            correctness_verified: false,
            error_message: Some(e),
        }
    });

    if performance.correctness_verified {
        println!("  - Correctness verified");
        println!("  - Speedup: {:.2}x", performance.speedup);
    }

    // Update summary with performance data
    let summary = summary;

    // Write summary JSON with performance data
    let summary_with_perf = serde_json::json!({
        "c_file": summary.c_file,
        "kernel_name": summary.kernel_name,
        "output_dir": summary.output_dir,
        "original_mlir": summary.original_mlir,
        "optimized_mlir": summary.optimized_mlir,
        "original_schedule": summary.original_schedule,
        "transformed_schedule": summary.transformed_schedule,
        "transformations": summary.transformations,
        "performance": performance,
    });

    let summary_file = output_dir.join("summary.json");
    let summary_json = serde_json::to_string_pretty(&summary_with_perf)
        .map_err(|e| format!("Failed to serialize summary: {}", e))?;
    fs::write(&summary_file, summary_json)
        .map_err(|e| format!("Failed to write summary: {}", e))?;

    println!("\nPipeline complete! Results in: {}", output_dir.display());

    Ok(summary)
}

/// Apply a PolySat transformation to an ISL schedule
fn apply_polysat_transformation(
    schedule_str: &str,
    transform: &str,
    band_idx: Option<usize>,
    size: Option<i32>,
) -> Result<String, String> {
    use crate::language::{mark_parallel, tile_schedule, vectorize_schedule};
    use crate::parse::parse_isl;
    use isl_rs::Context;

    // Create ISL context
    let ctx = Arc::new(Context::alloc());

    // Parse the schedule
    let schedule_handle = parse_isl(ctx, schedule_str)?;
    let schedule = &*schedule_handle.schedule;

    // Apply transformation
    let transformed = match transform {
        "tile" => {
            let band = band_idx.unwrap_or(0);
            let tile_size = size.unwrap_or(32);
            tile_schedule(schedule, band, tile_size)
        }
        "parallel" => {
            let band = band_idx.unwrap_or(0);
            mark_parallel(schedule, band)
        }
        "vectorize" => {
            let band = band_idx.unwrap_or(0);
            let width = size.unwrap_or(8);
            vectorize_schedule(schedule, band, width)
        }
        _ => return Err(format!("Unknown transformation: {}", transform)),
    };

    // Return the transformed schedule as string
    Ok(transformed.to_str().to_string())
}

#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub struct PipelineResult {
    pub c_file: String,
    pub kernel_name: String,
    pub output_dir: PathBuf,
    pub original_mlir: PathBuf,
    pub optimized_mlir: PathBuf,
    pub original_schedule: PathBuf,
    pub transformed_schedule: PathBuf,
    pub transformations: Vec<(String, Option<usize>, Option<i32>)>,
}
