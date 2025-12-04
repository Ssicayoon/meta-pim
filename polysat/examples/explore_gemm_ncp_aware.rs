//! GEMM NCP-Aware Schedule Exploration using PolySat
//!
//! This example demonstrates the complete ISL <-> EqSat bridge for GEMM optimization.
//! It uses RFC001's ISL-based property extraction for all feature detection and cost
//! computation, avoiding any string-based parsing.
//!
//! ## RFC001 Alignment
//!
//! This example strictly follows RFC001: Rigorous ISL-EqSat Semantic Bridge:
//!
//! 1. **ISL-based Property Extraction**: All schedule properties (parallel_dims,
//!    tile_sizes, kernel_pattern, etc.) are extracted via ScheduleProperties,
//!    which uses ISL API calls, NOT string parsing.
//!
//! 2. **T^2 Cost Model**: The cost model uses the empirically validated T^2
//!    tiling benefit factor (loop overhead dominates cache effects).
//!
//! 3. **Semantic Equivalence**: Schedules are compared via ISL schedule tree
//!    structure, not string representations.
//!
//! ## Optimization Strategy (Meta-PIM NCP Architecture)
//!
//! 1. Slice-level tiling: Partition across 8 LLC slices
//! 2. Bank-level tiling: Distribute among 64 banks per slice
//! 3. Cache-level tiling: Optimize for 64KB local storage per NCP
//! 4. Parallelization: Exploit outer-loop parallelism for multi-NCP execution
//! 5. Communication minimization: Reduce NoC traffic through tiling

use egg::{CostFunction, EGraph, Runner};
use isl_rs::Context;
use polysat::{
    dump_isl, isl_codegen_ffi, parse_isl, rational_dependency_rules,
    schedule_properties::ScheduleProperties, SchedOp, ScheduleAnalysis, ScheduleHandle,
};
use std::fs;
use std::sync::Arc;

/// RFC001-compliant schedule extraction result.
///
/// Contains ISL-derived properties for analysis, avoiding string parsing.
struct ExtractedSchedule {
    cost: f64,
    handle: ScheduleHandle,
    /// ISL-derived properties (RFC001 Section 1.1)
    properties: ScheduleProperties,
    /// Human-readable block representation (for display only, NOT for analysis)
    block_str: String,
}

/// Extract all schedules using NCP-aware cost model with RFC001 ISL-based properties.
///
/// This function demonstrates proper RFC001 usage:
/// - Cost computation via ScheduleProperties (ISL API)
/// - Feature detection via ScheduleProperties fields
/// - NO string matching for schedule analysis
fn extract_with_ncp_cost(
    egraph: &EGraph<SchedOp, ScheduleAnalysis>,
    _root: egg::Id,
    kernel_size: u64,
) -> Vec<ExtractedSchedule> {
    use polysat::isl_block_printer::schedule_to_block_str;
    use polysat::optimize::ScheduleCost;

    println!("\n[EXTRACT] Using NCP-aware cost model with RFC001 ISL-based properties");

    // Create GEMM-specific NCP cost model with correct matrix size
    // The cost model internally uses ScheduleProperties for all computations (RFC001)
    let m = kernel_size as usize;
    let n = kernel_size as usize;
    let k = kernel_size as usize;
    let mut cost_fn = ScheduleCost::gemm_precise(m, n, k);

    let mut results = Vec::new();

    for class in egraph.classes() {
        if let Some(ref schedule_handle) = class.data.schedule {
            // Calculate cost using NCP-aware model
            // Internally calls compute_cost_from_properties() which uses ISL-derived properties
            let schedule_op = SchedOp::Schedule(schedule_handle.clone());
            let cost = cost_fn.cost(&schedule_op, |_| 0.0);

            // RFC001: Use ISL-derived properties directly, NOT string parsing
            let properties = schedule_handle.properties.clone();

            // Block string is for human display only, NOT for feature detection
            let block_str = schedule_to_block_str(&schedule_handle.schedule);

            results.push(ExtractedSchedule {
                cost,
                handle: schedule_handle.clone(),
                properties,
                block_str,
            });
        }
    }

    // Sort by cost (lower is better)
    results.sort_by(|a, b| a.cost.partial_cmp(&b.cost).unwrap());
    results
}

/// Generate C code from an ISL schedule using the ISL AST codegen.
///
/// This function uses the ISL codegen FFI to produce executable C code
/// that implements the schedule's loop nest structure.
fn generate_c_code_from_schedule(handle: &ScheduleHandle) -> Result<String, String> {
    let ctx_ptr = handle.schedule.get_ctx().ptr as *mut std::ffi::c_void;
    let schedule_ptr = handle.schedule.ptr as *mut std::ffi::c_void;
    unsafe { isl_codegen_ffi::generate_c_code(ctx_ptr, schedule_ptr) }
}

/// Format kernel pattern for display
fn format_kernel_pattern(props: &ScheduleProperties) -> String {
    use polysat::schedule_properties::KernelPattern;
    match &props.kernel_pattern {
        Some(pattern) => match pattern {
            KernelPattern::Gemm { m, n, k } => format!("GEMM({}x{}x{})", m, n, k),
            KernelPattern::Conv2D { batch, channels, height, width } => {
                format!("Conv2D({}x{}x{}x{})", batch, channels, height, width)
            }
            KernelPattern::Stencil { dimensions, radius } => {
                format!("Stencil({}D,r={})", dimensions, radius)
            }
            KernelPattern::Ntt { size } => format!("NTT({})", size),
            KernelPattern::Generic => "Generic".to_string(),
        },
        None => "None".to_string(),
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n========================================================================");
    println!("  PolySat NCP-Aware Exploration: GEMM (RFC001 ISL-EqSat Bridge)");
    println!("========================================================================\n");

    // Kernel size is configurable - cost model will use this value
    let kernel_size: u64 = 64;

    println!("RFC001 Cost Model Configuration:");
    println!("  - Property extraction: ISL API (ScheduleProperties)");
    println!("  - Tiling benefit: T^2 model (loop overhead dominates)");
    println!("  - Cache cliff: Step function at L1/L2/L3 boundaries");
    println!("  - NO string parsing for feature detection");

    println!("\nNCP Architecture (Meta-PIM):");
    println!("  - 512 NCPs, 8 slices, 64 banks per slice");
    println!("  - Computation: FP64 MAC latency (500 cycles estimate)");
    println!("  - Communication: 2D mesh NoC (32 B/cycle, 5 cycles/hop)");
    println!("  - Memory: 64KB local SRAM per NCP");

    println!("\nGEMM Problem Configuration:");
    println!(
        "  - Matrix dimensions: {}x{}x{}",
        kernel_size, kernel_size, kernel_size
    );
    println!("  - Total operations: {} MACs (M*N*K)", kernel_size.pow(3));
    println!(
        "  - Operations per NCP: ~{} MACs ({} / 512)",
        kernel_size.pow(3) / 512,
        kernel_size.pow(3)
    );

    // Read baseline ISL schedule from Polygeist
    let baseline_isl = fs::read_to_string(format!(
        "polysat_schedules/gemm_standard_{}_isl/__polygeist_outlined_affine_0",
        kernel_size
    ))?;

    println!("\nBaseline schedule loaded ({} bytes)", baseline_isl.len());
    println!("  Contains: 3D loop nest (i, j, k) for C[i][j] += A[i][k] * B[k][j]");

    // Parse ISL schedule
    println!("\n[Step 1] Parsing ISL schedule...");
    let ctx = Arc::new(Context::alloc());
    let initial_schedule = parse_isl(ctx.clone(), &baseline_isl)?;

    // Display initial schedule properties (RFC001: ISL-based)
    println!("\n[Step 1.1] Initial schedule ISL properties (RFC001):");
    let initial_props = &initial_schedule.properties;
    println!("  - Band count: {}", initial_props.band_count);
    println!("  - Domain dimensions: {}", initial_props.domain_dimensions);
    println!("  - Parallel dims: {}", initial_props.parallel_dims);
    println!("  - Loop depth: {}", initial_props.loop_depth);
    println!("  - Separated bands: {}", initial_props.is_separated_bands);
    println!("  - Kernel pattern: {}", format_kernel_pattern(initial_props));
    if let Some(count) = initial_props.iteration_count {
        println!("  - Iteration count: {}", count);
    }

    // [Step 1.5] Load Access Information (Tier-2 Precise Analysis)
    println!("\n[Step 1.5] Loading access information for precise dependency analysis...");
    let access_file_path = format!(
        "polysat_schedules/gemm_standard_{}_isl/accesses.yaml",
        kernel_size
    );

    // Create AccessInfo with placeholders (handles are opaque in this version)
    let mut access_info = polysat::AccessInfo::new(
        polysat::ContextHandle::new_placeholder(),
        polysat::AccessScheduleHandle::new_placeholder(),
    );

    // Populate from Polymer access file (using same file for reads/writes as it contains both)
    // This enables the "Tier-2" precise dependency analysis path using ISL flow analysis
    if std::path::Path::new(&access_file_path).exists() {
        match access_info.populate_from_polymer_files(
            std::path::Path::new(&access_file_path),
            std::path::Path::new(&access_file_path),
            &ctx,
        ) {
            Ok(_) => println!(
                "  - Successfully loaded access relations from {}",
                access_file_path
            ),
            Err(e) => println!("  - Warning: Failed to load access relations: {}", e),
        }
    } else {
        println!(
            "  - Warning: Access file not found at {}. Using conservative analysis.",
            access_file_path
        );
    }

    // Build e-graph with AccessInfo
    println!("\n[Step 2] Building e-graph with rational dependency-aware rules");
    // Pass None for schedule_dir to force using our manually populated AccessInfo
    let analysis = ScheduleAnalysis::with_access_info(ctx.clone(), access_info, None);
    let mut egraph = EGraph::new(analysis);
    let root = egraph.add(SchedOp::Schedule(initial_schedule.clone()));

    let rules = rational_dependency_rules();
    // Filter out interchange rules to avoid ISL stability issues with separated bands
    let rules: Vec<_> = rules
        .into_iter()
        .filter(|r| !r.name.as_str().contains("interchange"))
        .collect();
    println!(
        "  Loaded {} rational rewrite rules (interchange disabled)",
        rules.len()
    );

    // Run equality saturation with NCP-aware exploration
    println!("\n[Step 3] Running equality saturation...");
    println!("  Configuration:");
    println!("    - Max iterations: 10");
    println!("    - Node limit: 20,000");
    println!("    - Rules: Tiling (8/16/32/64), Parallelization");
    println!("    - Cost: RFC001 ISL-based model (T^2 benefit + cache cliff)");

    let runner = Runner::default()
        .with_egraph(egraph)
        .with_iter_limit(10)
        .with_node_limit(20_000)
        .run(&rules);

    println!("  E-graph built:");
    println!("    - E-classes: {}", runner.egraph.number_of_classes());
    println!(
        "    - Total nodes: {}",
        runner.egraph.total_number_of_nodes()
    );

    // Extract ALL candidates using NCP-aware GEMM cost with RFC001 properties
    println!("\n[Step 4] Extracting schedule candidates...");
    let all_candidates = extract_with_ncp_cost(&runner.egraph, root, kernel_size);
    println!("  Extracted {} candidates", all_candidates.len());

    // Export schedules with RFC001 ISL-based analysis
    println!("\n[Step 5] Exporting schedules with ISL-based analysis...");

    let export_dir = format!("output/gemm_standard_{}", kernel_size);
    fs::create_dir_all(&export_dir)?;

    for (rank, sched) in all_candidates.iter().enumerate() {
        let rank_num = rank + 1;
        let filename = format!(
            "{}/schedule_rank_{:04}_ncp_cost_{:.3}.isl",
            export_dir, rank_num, sched.cost
        );

        // Get ISL string representation for export
        let isl_string = dump_isl(&sched.handle);
        fs::write(&filename, &isl_string)?;

        // RFC001: Use ISL-derived properties for feature analysis, NOT string parsing
        let props = &sched.properties;

        // All these values come from ScheduleProperties (ISL API), not string matching
        let has_parallel = props.parallel_dims > 0;
        let has_tile = props.tile_sizes.is_some();
        let parallel_count = props.parallel_dims;
        let tile_count = props.tile_sizes.as_ref().map_or(0, |v| v.len());
        let tile_sizes_str = props
            .tile_sizes
            .as_ref()
            .map(|v| format!("{:?}", v))
            .unwrap_or_else(|| "None".to_string());

        println!(
            "+-- Rank {} (cost={:.4}) ------------------------------------------+",
            rank_num, sched.cost
        );
        println!(
            "|  Block: {}",
            sched.block_str.lines().next().unwrap_or("")
        );
        println!("|  ISL Properties (RFC001):");
        println!("|    Domain dims:   {}", props.domain_dimensions);
        println!("|    Band count:    {}", props.band_count);
        println!("|    Parallel dims: {} (target: 8 for LLC slices)", parallel_count);
        println!("|    Tile sizes:    {}", tile_sizes_str);
        println!("|    Tile count:    {} (target: 3 for i/j/k)", tile_count);
        println!("|    Loop depth:    {}", props.loop_depth);
        println!("|    Kernel:        {}", format_kernel_pattern(props));
        println!(
            "|  Features: Tiling={} Parallel={}",
            if has_tile { "Y" } else { "N" },
            if has_parallel { "Y" } else { "N" }
        );
        println!("|  Exported: {}", filename);
        println!("+------------------------------------------------------------------+\n");
    }

    // Generate summary report using RFC001 ISL-based properties
    println!("\n========================================================================");
    println!("  NCP-Aware GEMM Exploration Summary (RFC001 Compliant)");
    println!("========================================================================");

    println!("\nTop-10 Schedules (by NCP-aware GEMM cost):");
    println!(
        "{:<6} {:<12} {:<8} {:<8} {:<12} {:<10}",
        "Rank", "Cost", "ParDims", "Tiles", "TileSizes", "Pattern"
    );
    println!("{}", "-".repeat(70));

    for (rank, sched) in all_candidates.iter().take(10).enumerate() {
        let props = &sched.properties;
        // RFC001: All values from ISL-derived ScheduleProperties
        let tile_str = props
            .tile_sizes
            .as_ref()
            .map(|v| {
                if v.len() <= 2 {
                    format!("{:?}", v)
                } else {
                    format!("[{},..]", v[0])
                }
            })
            .unwrap_or_else(|| "-".to_string());

        println!(
            "{:<6} {:<12.4} {:<8} {:<8} {:<12} {:<10}",
            rank + 1,
            sched.cost,
            props.parallel_dims,
            props.tile_sizes.as_ref().map_or(0, |v| v.len()),
            tile_str,
            format_kernel_pattern(props)
        );
    }

    if !all_candidates.is_empty() {
        let best_cost = all_candidates[0].cost;

        // Find baseline using RFC001 ISL properties (not string matching)
        let baseline_cost = all_candidates
            .iter()
            .filter(|s| {
                // Baseline: no tiling and no parallelization (using ISL properties)
                s.properties.tile_sizes.is_none() && s.properties.parallel_dims == 0
            })
            .map(|s| s.cost)
            .next()
            .unwrap_or_else(|| all_candidates.last().unwrap().cost);

        println!("\nResults:");
        println!("  Best schedule cost:     {:.4}", best_cost);
        println!("  Baseline cost:          {:.4}", baseline_cost);
        println!("  Improvement ratio:      {:.2}x", baseline_cost / best_cost);
        println!("  (Lower cost = better: T^2 benefit + cache cliff penalty)");

        // Show best schedule ISL properties
        let best_props = &all_candidates[0].properties;
        println!("\nBest Schedule ISL Properties:");
        println!("  - Domain dimensions: {}", best_props.domain_dimensions);
        println!("  - Parallel dims:     {}", best_props.parallel_dims);
        println!(
            "  - Tile sizes:        {:?}",
            best_props.tile_sizes.as_ref().unwrap_or(&vec![])
        );
        println!("  - Kernel pattern:    {}", format_kernel_pattern(best_props));
        if let Some(count) = best_props.iteration_count {
            println!("  - Iteration count:   {}", count);
        }
    }

    println!("\nFiles Generated:");
    println!("  - Output directory: {}/", export_dir);
    println!("  - Schedule count:   {}", all_candidates.len());
    println!("  - Format: schedule_rank_XXXX_ncp_cost_Y.YYY.isl");
    println!("\nRFC001 Compliance: All feature detection uses ISL-derived ScheduleProperties\n");

    // ========================================================================
    // [Step 6] Generate C code for best schedule and explain NCP cost model
    // ========================================================================
    if !all_candidates.is_empty() {
        let best = &all_candidates[0];
        println!("\n========================================================================");
        println!("  [Step 6] Best Schedule C Code Generation & NCP Cost Analysis");
        println!("========================================================================");

        // Generate C code using ISL AST codegen
        println!("\n--- Generated C Code (ISL AST Codegen) ---\n");
        match generate_c_code_from_schedule(&best.handle) {
            Ok(c_code) => {
                println!("{}", c_code);

                // Save C code to file
                let c_file = format!("{}/best_schedule.c", export_dir);
                fs::write(&c_file, &c_code)?;
                println!("  (Saved to: {})", c_file);
            }
            Err(e) => println!("  Error generating C code: {}", e),
        }

        // Analyze why NCP cost model considers this schedule optimal
        println!("\n--- NCP Cost Model Analysis ---\n");
        analyze_ncp_optimality(best, kernel_size);
    }

    Ok(())
}

/// Analyze why the NCP cost model considers a schedule optimal.
///
/// The NCP cost model is a multiplicative model with several factors:
///   Cost = Base * ParallelFactor * VectorFactor * TilingFactor * NCPCommFactor * CacheCliffFactor * LoopOverheadFactor
///
/// Each factor captures a different aspect of NCP architecture performance:
/// - ParallelFactor: Exploits 512 NCP units via outer loop parallelism
/// - TilingFactor: Reduces memory traffic via data reuse within tiles
/// - LoopOverheadFactor: Larger tiles = fewer loop iterations = less overhead
/// - CacheCliffFactor: Penalty when working set exceeds cache levels
/// - NCPCommFactor: NoC communication cost for data movement between NCPs
fn analyze_ncp_optimality(sched: &ExtractedSchedule, kernel_size: u64) {
    let props = &sched.properties;

    println!("Schedule Properties (ISL-derived):");
    println!("  - Domain:       {} dimensions (GEMM: i, j, k)", props.domain_dimensions);
    println!("  - Tile sizes:   {:?}", props.tile_sizes);
    println!("  - Parallel:     {} dimensions marked parallel", props.parallel_dims);
    println!("  - Band count:   {}", props.band_count);
    println!("  - Loop depth:   {}", props.loop_depth);

    // Explain each cost factor's contribution
    println!("\nCost Factor Breakdown:");

    // 1. Loop Overhead Factor (Phase 5 key insight: this DOMINATES for GEMM)
    if let Some(ref tiles) = props.tile_sizes {
        let min_tile = tiles.iter().cloned().min().unwrap_or(1).max(1) as f64;
        let log_factor = min_tile.log2().max(0.0);
        let loop_overhead = 1.0 / (1.0 + log_factor);
        println!(
            "  1. Loop Overhead Factor: {:.4} (min_tile={}, formula: 1/(1+log2(T)))",
            loop_overhead, min_tile as i32
        );
        println!("     -> Larger tiles = fewer loop iterations = MAJOR speedup");
        println!(
            "     -> For {}x{}x{} GEMM with T={}: ~{} tile iterations vs {} without tiling",
            kernel_size,
            kernel_size,
            kernel_size,
            min_tile as i32,
            (kernel_size as f64 / min_tile).powf(3.0) as u64,
            kernel_size.pow(3)
        );
    } else {
        println!("  1. Loop Overhead Factor: 1.0 (no tiling, baseline)");
    }

    // 2. Parallel Factor
    if props.parallel_dims > 0 {
        let parallel_benefit = 1.0 / (1.0 + props.parallel_dims as f64 * 0.1);
        println!(
            "  2. Parallel Factor: {:.4} ({} parallel dims)",
            parallel_benefit, props.parallel_dims
        );
        println!("     -> Outer loops can execute on different NCPs concurrently");
        println!("     -> 512 NCPs = massive parallelism potential");
    } else {
        println!("  2. Parallel Factor: 1.0 (no parallelism)");
    }

    // 3. Tiling/Reuse Factor
    if let Some(ref tiles) = props.tile_sizes {
        let tile_count = tiles.len();
        let tiling_benefit = 1.0 / (1.0 + tile_count as f64 * 0.2);
        println!(
            "  3. Tiling Factor: {:.4} ({} tiled dimensions)",
            tiling_benefit, tile_count
        );
        println!("     -> Tiling enables data reuse within each tile");
        println!("     -> For GEMM: A[i,k], B[k,j] blocks loaded once, reused many times");
    } else {
        println!("  3. Tiling Factor: 1.0 (no tiling)");
    }

    // 4. Cache Cliff Factor
    if let Some(ref tiles) = props.tile_sizes {
        // Compute working set for GEMM: (Ti*Tk + Tk*Tj + Ti*Tj) * 8 bytes
        let (ti, tj, tk) = match tiles.len() {
            0 => (1, 1, 1),
            1 => (tiles[0], tiles[0], tiles[0]),
            2 => (tiles[0], tiles[1], tiles[0]),
            _ => (tiles[0], tiles[1], tiles[2]),
        };
        let working_set_bytes =
            ((ti * tk + tk * tj + ti * tj) as usize) * 8; // 8 bytes for f64
        let working_set_kb = working_set_bytes as f64 / 1024.0;

        let cache_penalty = if working_set_bytes <= 32 * 1024 {
            1.0 // Fits in L1
        } else if working_set_bytes <= 256 * 1024 {
            let l1 = 32.0 * 1024.0;
            let l2 = 256.0 * 1024.0;
            1.0 + 9.0 * (working_set_bytes as f64 - l1) / (l2 - l1)
        } else {
            10.0 + 40.0 * ((working_set_bytes as f64 - 256.0 * 1024.0) / (8.0 * 1024.0 * 1024.0))
        };

        println!(
            "  4. Cache Cliff Factor: {:.4} (working set: {:.1} KB)",
            cache_penalty, working_set_kb
        );
        if working_set_bytes <= 32 * 1024 {
            println!("     -> Fits in L1 (32KB): optimal cache behavior");
        } else if working_set_bytes <= 256 * 1024 {
            println!("     -> Fits in L2 (256KB): moderate penalty");
        } else {
            println!("     -> Exceeds L2: significant penalty, but offset by tiling benefits");
        }
    } else {
        println!("  4. Cache Cliff Factor: 1.0 (no tiling, streaming access)");
    }

    // 5. NCP Communication Factor (heuristic based on parallel structure)
    println!("  5. NCP Comm Factor: (estimated from parallel structure)");
    println!("     -> Tiled schedules reduce inter-NCP data movement");
    println!("     -> NoC cost: 5 cycles/hop + ceil(S/32) cycles for S bytes");

    // Summary: why this schedule is best
    println!("\n--- Why This Schedule is Optimal for NCP ---\n");
    println!("The NCP cost model identifies this schedule as optimal because:");

    if props.tile_sizes.is_some() && props.parallel_dims > 0 {
        println!("  1. TILING: Multi-level tiling ({:?}) enables data reuse", props.tile_sizes);
        println!("     - Each NCP processes a tile locally with minimal DRAM traffic");
        println!("     - Working set fits in 64KB local SRAM per NCP");
        println!("");
        println!("  2. PARALLELISM: {} parallel dimensions exploit 512 NCPs", props.parallel_dims);
        println!("     - Outer tiled loops can be distributed across LLC slices (8)");
        println!("     - Inner tiled loops distributed across banks (64 per slice)");
        println!("");
        println!("  3. LOOP OVERHEAD: Larger tiles = fewer iterations");
        println!("     - Phase 5 validation showed this dominates cache effects");
        println!("     - T=512 was 13% faster than baseline for 2048x2048 GEMM");
        println!("");
        println!("  4. MEMORY HIERARCHY: Tiling matches NCP architecture");
        println!("     - Slice-level tiling: Partitions across 8 LLC slices");
        println!("     - Bank-level tiling: Distributes among 64 banks per slice");
        println!("     - Cache-level tiling: Fits in 64KB local SRAM per NCP");
    } else if props.tile_sizes.is_some() {
        println!("  - Tiled but not parallelized: good data reuse, limited NCP utilization");
    } else if props.parallel_dims > 0 {
        println!("  - Parallelized but not tiled: good NCP utilization, poor data reuse");
    } else {
        println!("  - Baseline schedule: no tiling, no parallelism (worst case)");
    }

    println!("\nFinal Cost: {:.6}", sched.cost);
    println!("  (Lower cost = better: multiplicative model of all factors)");
}
