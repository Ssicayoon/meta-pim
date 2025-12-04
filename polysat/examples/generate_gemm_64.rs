use polysat::codegen::extract_baseline_schedule;
use std::path::PathBuf;

fn main() {
    // Try to get POLYGEIST_DIR from environment, fall back to default
    let polygeist_dir = std::env::var("POLYGEIST_DIR").unwrap_or_else(|_| {
        println!("Warning: POLYGEIST_DIR not set, using default 'polygeist'");
        "polygeist".to_string()
    });

    let polygeist_dir = PathBuf::from(polygeist_dir);
    let shape: u64 = 64;
    let c_file = PathBuf::from(format!("examples/demo/gemm_{}.c", shape));
    let kernel_name = "gemm_standard";

    println!(
        "Generating baseline schedule and accesses for {}...",
        c_file.display()
    );
    println!("Using Polygeist directory: {}", polygeist_dir.display());

    match extract_baseline_schedule(
        polygeist_dir.to_str().unwrap(),
        c_file.to_str().unwrap(),
        kernel_name,
        shape,
    ) {
        Ok(result) => {
            println!("Success!");
            println!("Schedule file: {:?}", result.schedule_file);
            println!("Access file: {:?}", result.access_file);
        }
        Err(e) => {
            eprintln!("Error: {}", e);
            std::process::exit(1);
        }
    }
}
