use crate::language::{SchedOp, ScheduleAnalysis, ScheduleHandle};
use egg::{EGraph, Id, Language, RecExpr};
use isl_rs::Schedule;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

/// Evaluate an expression to reconstruct the schedule from the e-graph
/// This function walks through the expression tree and applies transformations
/// Returns both the schedule and its string representation with transformations preserved
fn evaluate_expression(
    _egraph: &EGraph<SchedOp, ScheduleAnalysis>,
    expr: &RecExpr<SchedOp>,
) -> Option<(ScheduleHandle, String)> {
    // We need to evaluate the expression bottom-up
    let mut values: HashMap<Id, ScheduleHandle> = HashMap::new();
    let mut transformations: Vec<String> = Vec::new();

    println!(
        "[EVAL DEBUG] Evaluating expression with {} nodes",
        expr.as_ref().len()
    );

    for (i, node) in expr.as_ref().iter().enumerate() {
        let id = Id::from(i);

        match node {
            SchedOp::Schedule(handle) => {
                println!("[EVAL DEBUG] Node {}: Schedule", i);
                values.insert(id, handle.clone());
            }
            SchedOp::Symbol(_) | SchedOp::Num(_) => {
                // These are just values, not schedules - skip them
            }
            SchedOp::Tile([sched_id, band_id, size_id]) => {
                // Handle band-index-based tiling (used by our actual rules)
                if let Some(input_schedule) = values.get(sched_id) {
                    // Get band index
                    let band_idx =
                        if let Some(SchedOp::Num(n)) = expr.as_ref().get(usize::from(*band_id)) {
                            *n as usize
                        } else {
                            0 // Default to first band
                        };

                    // Get tile size
                    let tile_size =
                        if let Some(SchedOp::Num(n)) = expr.as_ref().get(usize::from(*size_id)) {
                            *n
                        } else {
                            32 // Default tile size
                        };

                    // Apply the tiling transformation using band index
                    transformations.push(format!("tile:{}:{}", band_idx, tile_size));
                    println!(
                        "[EVAL DEBUG] Applying tile_band: band_idx={}, size={}",
                        band_idx, tile_size
                    );

                    // BUG FIX #3: Propagate errors instead of silent fallback
                    // PREVIOUS BEHAVIOR: On tiling failure, silently returned original schedule,
                    // causing user to think transformation succeeded when it actually failed.
                    // NEW BEHAVIOR: On failure, skip this candidate entirely by continuing to next node.
                    // This gives honest extraction results - we only return schedules that actually transformed.
                    let tiled = match crate::transformations::tile(
                        &input_schedule.schedule,
                        band_idx,
                        tile_size,
                    ) {
                        Ok(Some(t)) => t,
                        Ok(None) => {
                            println!("[EVAL ERROR] Tiling not applicable");
                            continue;
                        }
                        Err(e) => {
                            println!("[EVAL ERROR] Tiling failed: {}", e);
                            println!(
                                "[EVAL ERROR] Skipping this candidate (no fallback to baseline)"
                            );
                            // Don't insert into values - this node evaluation failed
                            // The e-graph will have other candidates to try
                            continue; // Skip to next node instead of returning baseline
                        }
                    };

                    // Debug: check if tiling actually changed the schedule
                    let original_str = input_schedule.schedule.to_str();
                    let tiled_str = tiled.to_str();
                    if original_str == tiled_str {
                        println!("[EVAL DEBUG WARNING] Tiling did NOT change the schedule!");
                        println!(
                            "[EVAL DEBUG] Original: {}",
                            original_str.lines().next().unwrap_or("")
                        );
                        println!(
                            "[EVAL DEBUG] Tiled: {}",
                            tiled_str.lines().next().unwrap_or("")
                        );
                    } else {
                        println!("[EVAL DEBUG] Tiling successfully transformed the schedule");
                        println!(
                            "[EVAL DEBUG] Tiled result: {}",
                            tiled_str.lines().next().unwrap_or("")
                        );
                    }

                    let new_handle = ScheduleHandle::new(input_schedule.ctx.clone(), tiled);

                    // DEBUG: Check if the handle preserves the tree
                    let handle_str = new_handle.schedule.to_str();
                    println!(
                        "[EVAL DEBUG] Handle string after storing: {}",
                        handle_str.lines().next().unwrap_or("")
                    );

                    values.insert(id, new_handle);
                }
            }
            SchedOp::TileAtMark([sched_id, mark_id, size_id]) => {
                // Handle mark-based tiling (if marks are present)
                if let Some(input_schedule) = values.get(sched_id) {
                    // Get mark name - it should be in our values map or in the expression
                    let mark_name = if let Some(SchedOp::Symbol(s)) =
                        expr.as_ref().get(usize::from(*mark_id))
                    {
                        s.to_string()
                    } else {
                        // Try to get from computed values
                        "L0".to_string() // Default mark
                    };

                    // Get tile size - should be in the expression
                    let tile_size =
                        if let Some(SchedOp::Num(n)) = expr.as_ref().get(usize::from(*size_id)) {
                            *n
                        } else {
                            32 // Default tile size
                        };

                    // Apply the tiling transformation
                    transformations.push(format!("tile_mark:{}:{}", mark_name, tile_size));
                    println!(
                        "[EVAL DEBUG] Applying tile_at_mark: mark='{}', size={}",
                        mark_name, tile_size
                    );
                    let tiled = crate::language::tile_at_mark(
                        &input_schedule.schedule,
                        &mark_name,
                        tile_size,
                    );

                    // Debug: check if tiling actually changed the schedule
                    let original_str = input_schedule.schedule.to_str();
                    let tiled_str = tiled.to_str();
                    if original_str == tiled_str {
                        println!(
                            "[EVAL DEBUG WARNING] Tiling at mark did NOT change the schedule!"
                        );
                        println!(
                            "[EVAL DEBUG] Original: {}",
                            original_str.lines().next().unwrap_or("")
                        );
                        println!(
                            "[EVAL DEBUG] Tiled: {}",
                            tiled_str.lines().next().unwrap_or("")
                        );
                    } else {
                        println!(
                            "[EVAL DEBUG] Tiling at mark successfully transformed the schedule"
                        );
                    }

                    let new_handle = ScheduleHandle::new(input_schedule.ctx.clone(), tiled);

                    // DEBUG: Check if the handle preserves the tree
                    let handle_str = new_handle.schedule.to_str();
                    println!(
                        "[EVAL DEBUG] Handle string after storing: {}",
                        handle_str.lines().next().unwrap_or("")
                    );

                    values.insert(id, new_handle);
                }
            }
            SchedOp::Parallel([sched_id, band_id]) => {
                // Handle band-index-based parallelization
                if let Some(input_schedule) = values.get(sched_id) {
                    let band_idx =
                        if let Some(SchedOp::Num(n)) = expr.as_ref().get(usize::from(*band_id)) {
                            *n as usize
                        } else {
                            0 // Default to first band
                        };

                    println!("[EVAL DEBUG] Applying parallel_band: band_idx={}", band_idx);
                    transformations.push(format!("parallel:{}", band_idx));
                    // For parallelization, we can use the parallel_band function from language module
                    // which should handle the band properly
                    let parallel =
                        crate::language::parallel_band(&input_schedule.schedule, band_idx);
                    let new_handle = ScheduleHandle::new(input_schedule.ctx.clone(), parallel);
                    values.insert(id, new_handle);
                }
            }
            SchedOp::Interchange([sched_id, band1_id, band2_id]) => {
                // Handle band-index-based interchange
                if let Some(input_schedule) = values.get(sched_id) {
                    let band1 =
                        if let Some(SchedOp::Num(n)) = expr.as_ref().get(usize::from(*band1_id)) {
                            *n as usize
                        } else {
                            0
                        };

                    let band2 =
                        if let Some(SchedOp::Num(n)) = expr.as_ref().get(usize::from(*band2_id)) {
                            *n as usize
                        } else {
                            1
                        };

                    println!(
                        "[EVAL DEBUG] Applying interchange_bands: band1={}, band2={}",
                        band1, band2
                    );
                    let interchanged = match crate::transformations::interchange(
                        &input_schedule.schedule,
                        band1,
                        band2,
                        None,
                    ) {
                        Ok(Some(i)) => i,
                        Ok(None) => {
                            println!("[EVAL ERROR] Interchange not applicable");
                            continue;
                        }
                        Err(e) => {
                            println!("[EVAL ERROR] Interchange failed: {}", e);
                            println!("[EVAL ERROR] Skipping this candidate");
                            continue;
                        }
                    };
                    let new_handle = ScheduleHandle::new(input_schedule.ctx.clone(), interchanged);
                    values.insert(id, new_handle);
                }
            }
            SchedOp::Skew([sched_id, band_id, factor_id, direction_id]) => {
                // Handle skewing transformation
                if let Some(input_schedule) = values.get(sched_id) {
                    let band_idx =
                        if let Some(SchedOp::Num(n)) = expr.as_ref().get(usize::from(*band_id)) {
                            *n as usize
                        } else {
                            0
                        };

                    let factor =
                        if let Some(SchedOp::Num(n)) = expr.as_ref().get(usize::from(*factor_id)) {
                            *n
                        } else {
                            1
                        };

                    let direction = if let Some(SchedOp::Num(n)) =
                        expr.as_ref().get(usize::from(*direction_id))
                    {
                        *n
                    } else {
                        0
                    };

                    println!(
                        "[EVAL DEBUG] Applying skew_band: band_idx={}, factor={}, direction={}",
                        band_idx, factor, direction
                    );
                    // Use the transformations module (with error propagation - Bug Fix #3)
                    let forward = direction == 0;
                    let skewed = match crate::transformations::skew(
                        &input_schedule.schedule,
                        band_idx,
                        factor,
                        forward,
                        None,
                    ) {
                        Ok(Some(s)) => s,
                        Ok(None) => {
                            println!("[EVAL ERROR] Skewing not applicable");
                            continue;
                        }
                        Err(e) => {
                            println!("[EVAL ERROR] Skewing failed: {}", e);
                            println!("[EVAL ERROR] Skipping this candidate");
                            continue;
                        }
                    };
                    let new_handle = ScheduleHandle::new(input_schedule.ctx.clone(), skewed);
                    values.insert(id, new_handle);
                }
            }
            SchedOp::ParallelAtMark([sched_id, mark_id]) => {
                if let Some(input_schedule) = values.get(sched_id) {
                    let mark_name = if let Some(SchedOp::Symbol(s)) =
                        expr.as_ref().get(usize::from(*mark_id))
                    {
                        s.to_string()
                    } else {
                        "L0".to_string() // Default mark
                    };

                    let parallel =
                        crate::language::parallel_at_mark(&input_schedule.schedule, &mark_name);
                    let new_handle = ScheduleHandle::new(input_schedule.ctx.clone(), parallel);
                    values.insert(id, new_handle);
                }
            }
            SchedOp::InsertMark([sched_id, mark_id]) => {
                if let Some(input_schedule) = values.get(sched_id) {
                    let mark_name = if let Some(SchedOp::Symbol(s)) =
                        expr.as_ref().get(usize::from(*mark_id))
                    {
                        s.to_string()
                    } else {
                        "L0".to_string() // Default mark
                    };

                    let marked = crate::language::insert_mark_at_band(
                        &input_schedule.schedule,
                        &mark_name,
                        0,
                    );
                    let new_handle = ScheduleHandle::new(input_schedule.ctx.clone(), marked);
                    values.insert(id, new_handle);
                }
            }
            SchedOp::TilePerDim([sched_id, size_i_id, size_j_id, size_k_id]) => {
                // BUG FIX #1: Handle per-dimension tiling (Ti, Tj, Tk can be different)
                // This is critical for NCP-aware optimization where optimal GEMM uses Ti=16, Tj=16, Tk=8
                // to fit within 4KB buffer constraint: 16x8 + 8x16 + 16x16 = 512 doubles = 4KB
                if let Some(input_schedule) = values.get(sched_id) {
                    // Extract tile sizes from e-graph nodes
                    let size_i =
                        if let Some(SchedOp::Num(n)) = expr.as_ref().get(usize::from(*size_i_id)) {
                            *n
                        } else {
                            16 // Default Ti
                        };

                    let size_j =
                        if let Some(SchedOp::Num(n)) = expr.as_ref().get(usize::from(*size_j_id)) {
                            *n
                        } else {
                            16 // Default Tj
                        };

                    let size_k =
                        if let Some(SchedOp::Num(n)) = expr.as_ref().get(usize::from(*size_k_id)) {
                            *n
                        } else {
                            8 // Default Tk (smaller for optimal NCP memory usage)
                        };

                    transformations.push(format!("tile_per_dim:{}:{}:{}", size_i, size_j, size_k));
                    println!(
                        "[EVAL DEBUG] Applying tile_per_dimension: Ti={}, Tj={}, Tk={}",
                        size_i, size_j, size_k
                    );

                    // Check if this schedule has separate 1D bands (Polygeist pattern)
                    // If so, use tile_separate_bands which handles each band individually
                    let schedule_str = input_schedule.schedule.to_str();
                    let has_separate_bands = schedule_str.contains("schedule: \"L")
                        && schedule_str.contains("[(i0)]")
                        && schedule_str.contains("child:")
                        && schedule_str.contains("schedule:");

                    let tiled = if has_separate_bands {
                        println!("[EVAL DEBUG] Detected separate bands, using tile_separate_bands");
                        // Use tile_separate_bands for Polygeist's band[i0], band[i1], band[i2] structure
                        crate::tile_separate_bands::tile_separate_bands(
                            &input_schedule.schedule,
                            vec![size_i, size_j, size_k],
                        )
                    } else {
                        println!(
                            "[EVAL DEBUG] Single multi-D band detected, using tile_per_dimension"
                        );
                        // Use standard per-dimension tiling for multi-dimensional bands
                        crate::tile_per_dimension::tile_per_dimension(
                            &input_schedule.schedule,
                            0, // Band index (default to first band)
                            vec![size_i, size_j, size_k],
                        )
                    };

                    // Verify tiling actually changed the schedule
                    let original_str = input_schedule.schedule.to_str();
                    let tiled_str = tiled.to_str();
                    if original_str == tiled_str {
                        println!("[EVAL DEBUG WARNING] TilePerDim did NOT change the schedule!");
                        println!("[EVAL DEBUG] This may indicate dimensionality mismatch or invalid band");
                    } else {
                        println!("[EVAL DEBUG] TilePerDim successfully transformed the schedule");
                        if tiled_str.contains("mod") {
                            println!("[EVAL DEBUG] - Schedule contains tiling expressions (mod operations)");
                        }
                    }

                    let new_handle = ScheduleHandle::new(input_schedule.ctx.clone(), tiled);
                    values.insert(id, new_handle);
                }
            }
            _ => {
                // For other nodes, we might need to handle them or skip
                println!("[EVAL DEBUG] Node {}: Unhandled variant {:?}", i, node);
            }
        }
    }

    // Return the value of the root node (last node in expression)
    if let Some(last_id) = expr.as_ref().len().checked_sub(1) {
        if let Some(schedule) = values.get(&Id::from(last_id)) {
            println!(
                "[EVAL DEBUG] Returning schedule with {} transformations",
                transformations.len()
            );
            for t in &transformations {
                println!("[EVAL DEBUG]   - {}", t);
            }

            // Use block-style printer to preserve the schedule tree structure
            // ISL's to_str() loses the tree after transformations, so we need block-style YAML
            let schedule_str = crate::isl_block_printer::schedule_to_block_str(&schedule.schedule);
            println!(
                "[EVAL DEBUG] Schedule after transformations (tree format): {}",
                schedule_str.lines().next().unwrap_or("")
            );

            Some((schedule.clone(), schedule_str))
        } else {
            None
        }
    } else {
        None
    }
}

/// Extract ALL possible schedule candidates from the e-graph
/// This is crucial for understanding what transformations the e-graph actually contains
pub struct AllCandidatesExtractor<'a> {
    egraph: &'a EGraph<SchedOp, ScheduleAnalysis>,
    /// Cache to avoid recomputing the same e-class multiple times
    cache: HashMap<Id, Vec<RecExpr<SchedOp>>>,
    /// Limit to prevent exponential explosion
    max_candidates: usize,
}

impl<'a> AllCandidatesExtractor<'a> {
    pub fn new(egraph: &'a EGraph<SchedOp, ScheduleAnalysis>) -> Self {
        Self {
            egraph,
            cache: HashMap::new(),
            max_candidates: 1000, // Default limit
        }
    }

    pub fn with_limit(mut self, limit: usize) -> Self {
        self.max_candidates = limit;
        self
    }

    /// Extract all possible candidates rooted at the given e-class
    /// Returns a vector of (cost, expression) pairs
    pub fn extract_all(&mut self, root: Id) -> Vec<(f64, RecExpr<SchedOp>)> {
        // Use iterative approach to avoid stack overflow
        let candidates = self.extract_candidates_limited(root, 10); // Limit depth

        // Calculate costs for each candidate
        let mut results = Vec::new();
        for expr in candidates {
            let cost = self.calculate_cost(&expr);
            results.push((cost, expr));
        }

        // Sort by cost for easier analysis
        results.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        results
    }

    /// Extract candidates with depth limit to avoid stack overflow
    fn extract_candidates_limited(
        &mut self,
        eclass: Id,
        max_depth: usize,
    ) -> Vec<RecExpr<SchedOp>> {
        if max_depth == 0 {
            // At max depth, just return a simple placeholder
            let mut expr = RecExpr::default();
            expr.add(SchedOp::Num(0));
            return vec![expr];
        }

        let canonical = self.egraph.find(eclass);

        // Check cache first
        if let Some(cached) = self.cache.get(&canonical) {
            return cached.clone();
        }

        // Get the e-class
        let class = match self.egraph.classes().find(|c| c.id == canonical) {
            Some(c) => c,
            None => return vec![],
        };

        let mut all_candidates = Vec::new();
        let mut seen_schedules = HashSet::new();

        // Process ALL nodes to get full diversity
        for node in &class.nodes {
            if all_candidates.len() >= self.max_candidates / 10 {
                // Soft limit per e-class to avoid explosion
                break;
            }

            // Extract with reduced depth
            let node_candidates = self.extract_node_limited(node.clone(), max_depth - 1);

            // Filter out duplicates based on the schedule handle
            for candidate in node_candidates {
                // Check if this is actually a unique schedule
                if let Some(sched_handle) = self.get_schedule_handle(&candidate) {
                    if seen_schedules.insert(sched_handle) {
                        all_candidates.push(candidate);
                    }
                } else {
                    // Not a schedule node, include it anyway for composition
                    all_candidates.push(candidate);
                }
            }
        }

        // Cache the results
        self.cache.insert(canonical, all_candidates.clone());
        all_candidates
    }

    /// Extract candidates for a node with depth limit
    fn extract_node_limited(&mut self, node: SchedOp, max_depth: usize) -> Vec<RecExpr<SchedOp>> {
        match &node {
            // Leaf nodes - just return the node itself
            SchedOp::Schedule(_) | SchedOp::Num(_) | SchedOp::Symbol(_) | SchedOp::Bool(_) => {
                let mut expr = RecExpr::default();
                expr.add(node);
                vec![expr]
            }

            // For transformation nodes, just include the node if it contains a schedule
            SchedOp::Tile([child, _, _])
            | SchedOp::Parallel([child, _])
            | SchedOp::Interchange([child, _, _])
            | SchedOp::Split([child, _, _])
            | SchedOp::Fuse([child, _, _])
            | SchedOp::Vectorize([child, _, _])
            | SchedOp::Unroll([child, _, _]) => {
                // Get the child's candidates (which should include Schedule nodes)
                let child_candidates = self.extract_candidates_limited(*child, max_depth);

                // For simplicity, just return the first child candidate
                // In a full implementation, we'd build the full expression
                child_candidates.into_iter().take(1).collect()
            }

            _ => {
                // For other nodes, return empty
                vec![]
            }
        }
    }

    /// Recursively extract all possible expressions for an e-class
    #[allow(dead_code)]
    fn extract_candidates_for_class(&mut self, eclass: Id) -> Vec<RecExpr<SchedOp>> {
        // Check cache first
        let canonical = self.egraph.find(eclass);
        if let Some(cached) = self.cache.get(&canonical) {
            return cached.clone();
        }

        // Get the e-class
        let class = match self.egraph.classes().find(|c| c.id == canonical) {
            Some(c) => c,
            None => return vec![],
        };

        let mut all_candidates = Vec::new();

        // For each node in the e-class, generate all possible expressions
        for node in &class.nodes {
            // Early termination if we have too many candidates
            if all_candidates.len() >= self.max_candidates {
                eprintln!("[EXTRACT] Hit candidate limit at e-class {:?}", canonical);
                break;
            }

            let node_candidates = self.extract_candidates_for_node(node.clone());
            all_candidates.extend(node_candidates);
        }

        // Cache the results
        self.cache.insert(canonical, all_candidates.clone());
        all_candidates
    }

    /// Generate all possible expressions for a specific e-node
    #[allow(dead_code)]
    fn extract_candidates_for_node(&mut self, node: SchedOp) -> Vec<RecExpr<SchedOp>> {
        match &node {
            // Leaf nodes - just return the node itself
            SchedOp::Schedule(_) | SchedOp::Num(_) | SchedOp::Symbol(_) | SchedOp::Bool(_) => {
                let mut expr = RecExpr::default();
                expr.add(node);
                vec![expr]
            }

            // Unary operations - recursively get candidates for child
            SchedOp::Parallel([child, band])
            | SchedOp::Vectorize([child, band, _])
            | SchedOp::Unroll([child, band, _]) => {
                let child_candidates = self.extract_candidates_for_class(*child);
                let band_candidates = self.extract_candidates_for_class(*band);

                let mut results = Vec::new();

                // Limit combinations to prevent explosion
                let max_combinations = 100;
                let mut count = 0;

                for child_expr in &child_candidates {
                    for band_expr in &band_candidates {
                        if count >= max_combinations {
                            break;
                        }

                        // Build combined expression
                        let mut expr = child_expr.clone();
                        let band_root = expr.add_expr(band_expr);

                        // Add the current node with updated children
                        let new_node = match &node {
                            SchedOp::Parallel([_, _]) => {
                                SchedOp::Parallel([Id::from(expr.as_ref().len() - 1), band_root])
                            }
                            SchedOp::Vectorize([_, _, width]) => {
                                let width_expr = self.extract_candidates_for_class(*width);
                                if let Some(w) = width_expr.first() {
                                    let width_root = expr.add_expr(w);
                                    SchedOp::Vectorize([
                                        Id::from(expr.as_ref().len() - 2),
                                        band_root,
                                        width_root,
                                    ])
                                } else {
                                    continue;
                                }
                            }
                            SchedOp::Unroll([_, _, factor]) => {
                                let factor_expr = self.extract_candidates_for_class(*factor);
                                if let Some(f) = factor_expr.first() {
                                    let factor_root = expr.add_expr(f);
                                    SchedOp::Unroll([
                                        Id::from(expr.as_ref().len() - 2),
                                        band_root,
                                        factor_root,
                                    ])
                                } else {
                                    continue;
                                }
                            }
                            _ => unreachable!(),
                        };

                        expr.add(new_node);
                        results.push(expr);
                        count += 1;
                    }
                }

                results
            }

            // TilePerDim - special case with 4 parameters
            SchedOp::TilePerDim([child, size_i, size_j, size_k]) => {
                let child_candidates = self.extract_candidates_for_class(*child);
                let size_i_candidates = self.extract_candidates_for_class(*size_i);
                let size_j_candidates = self.extract_candidates_for_class(*size_j);
                let size_k_candidates = self.extract_candidates_for_class(*size_k);

                let mut results = vec![];
                let mut count = 0;

                for child_expr in child_candidates.iter().take(10) {
                    if count >= 10 {
                        break;
                    }

                    for (si_expr, sj_expr, sk_expr) in size_i_candidates
                        .iter()
                        .zip(size_j_candidates.iter())
                        .zip(size_k_candidates.iter())
                        .map(|((a, b), c)| (a, b, c))
                        .take(5)
                    {
                        if count >= 10 {
                            break;
                        }

                        let mut expr = RecExpr::default();
                        let child_root = expr.add_expr(child_expr);
                        let si_root = expr.add_expr(si_expr);
                        let sj_root = expr.add_expr(sj_expr);
                        let sk_root = expr.add_expr(sk_expr);

                        expr.add(SchedOp::TilePerDim([child_root, si_root, sj_root, sk_root]));
                        results.push(expr);
                        count += 1;
                    }
                }

                results
            }

            // Binary/Ternary operations
            SchedOp::Tile([child, band, size]) | SchedOp::Split([child, band, size]) => {
                let child_candidates = self.extract_candidates_for_class(*child);
                let band_candidates = self.extract_candidates_for_class(*band);
                let size_candidates = self.extract_candidates_for_class(*size);

                let mut results = Vec::new();
                let max_combinations = 50;
                let mut count = 0;

                for child_expr in &child_candidates {
                    for band_expr in &band_candidates {
                        for size_expr in &size_candidates {
                            if count >= max_combinations {
                                break;
                            }

                            let mut expr = child_expr.clone();
                            let band_root = expr.add_expr(band_expr);
                            let size_root = expr.add_expr(size_expr);

                            let new_node = match &node {
                                SchedOp::Tile([_, _, _]) => SchedOp::Tile([
                                    Id::from(expr.as_ref().len() - 2),
                                    band_root,
                                    size_root,
                                ]),
                                SchedOp::Split([_, _, _]) => SchedOp::Split([
                                    Id::from(expr.as_ref().len() - 2),
                                    band_root,
                                    size_root,
                                ]),
                                _ => unreachable!(),
                            };

                            expr.add(new_node);
                            results.push(expr);
                            count += 1;
                        }
                    }
                }

                results
            }

            SchedOp::Interchange([child, b1, b2]) | SchedOp::Fuse([child, b1, b2]) => {
                let child_candidates = self.extract_candidates_for_class(*child);
                let b1_candidates = self.extract_candidates_for_class(*b1);
                let b2_candidates = self.extract_candidates_for_class(*b2);

                let mut results = Vec::new();
                let max_combinations = 50;
                let mut count = 0;

                for child_expr in &child_candidates {
                    for b1_expr in &b1_candidates {
                        for b2_expr in &b2_candidates {
                            if count >= max_combinations {
                                break;
                            }

                            let mut expr = child_expr.clone();
                            let b1_root = expr.add_expr(b1_expr);
                            let b2_root = expr.add_expr(b2_expr);

                            let new_node = match &node {
                                SchedOp::Interchange([_, _, _]) => SchedOp::Interchange([
                                    Id::from(expr.as_ref().len() - 2),
                                    b1_root,
                                    b2_root,
                                ]),
                                SchedOp::Fuse([_, _, _]) => SchedOp::Fuse([
                                    Id::from(expr.as_ref().len() - 2),
                                    b1_root,
                                    b2_root,
                                ]),
                                _ => unreachable!(),
                            };

                            expr.add(new_node);
                            results.push(expr);
                            count += 1;
                        }
                    }
                }

                results
            }

            // Mark-based operations
            SchedOp::InsertMark([child, mark])
            | SchedOp::ParallelAtMark([child, mark])
            | SchedOp::HasMark([child, mark])
            | SchedOp::GetMark([child, mark]) => {
                let child_candidates = self.extract_candidates_for_class(*child);
                let mark_candidates = self.extract_candidates_for_class(*mark);

                let mut results = Vec::new();
                for child_expr in &child_candidates {
                    for mark_expr in &mark_candidates {
                        let mut expr = child_expr.clone();
                        let mark_root = expr.add_expr(mark_expr);

                        let new_node = match &node {
                            SchedOp::InsertMark([_, _]) => {
                                SchedOp::InsertMark([Id::from(expr.as_ref().len() - 1), mark_root])
                            }
                            SchedOp::ParallelAtMark([_, _]) => SchedOp::ParallelAtMark([
                                Id::from(expr.as_ref().len() - 1),
                                mark_root,
                            ]),
                            SchedOp::HasMark([_, _]) => {
                                SchedOp::HasMark([Id::from(expr.as_ref().len() - 1), mark_root])
                            }
                            SchedOp::GetMark([_, _]) => {
                                SchedOp::GetMark([Id::from(expr.as_ref().len() - 1), mark_root])
                            }
                            _ => unreachable!(),
                        };

                        expr.add(new_node);
                        results.push(expr);
                    }
                }

                results
            }

            SchedOp::TileAtMark([child, mark, size])
            | SchedOp::VectorizeAtMark([child, mark, size])
            | SchedOp::UnrollAtMark([child, mark, size])
            | SchedOp::SplitAtMark([child, mark, size]) => {
                let child_candidates = self.extract_candidates_for_class(*child);
                let mark_candidates = self.extract_candidates_for_class(*mark);
                let size_candidates = self.extract_candidates_for_class(*size);

                let mut results = Vec::new();
                let max_combinations = 30;
                let mut count = 0;

                for child_expr in &child_candidates {
                    for mark_expr in &mark_candidates {
                        for size_expr in &size_candidates {
                            if count >= max_combinations {
                                break;
                            }

                            let mut expr = child_expr.clone();
                            let mark_root = expr.add_expr(mark_expr);
                            let size_root = expr.add_expr(size_expr);

                            let new_node = match &node {
                                SchedOp::TileAtMark([_, _, _]) => SchedOp::TileAtMark([
                                    Id::from(expr.as_ref().len() - 2),
                                    mark_root,
                                    size_root,
                                ]),
                                SchedOp::VectorizeAtMark([_, _, _]) => SchedOp::VectorizeAtMark([
                                    Id::from(expr.as_ref().len() - 2),
                                    mark_root,
                                    size_root,
                                ]),
                                SchedOp::UnrollAtMark([_, _, _]) => SchedOp::UnrollAtMark([
                                    Id::from(expr.as_ref().len() - 2),
                                    mark_root,
                                    size_root,
                                ]),
                                SchedOp::SplitAtMark([_, _, _]) => SchedOp::SplitAtMark([
                                    Id::from(expr.as_ref().len() - 2),
                                    mark_root,
                                    size_root,
                                ]),
                                _ => unreachable!(),
                            };

                            expr.add(new_node);
                            results.push(expr);
                            count += 1;
                        }
                    }
                }

                results
            }
            SchedOp::Skew(_) => {
                // Skewing transformation
                Vec::new()
            }
        }
    }

    /// Get the schedule handle from an expression if it exists
    fn get_schedule_handle(&self, expr: &RecExpr<SchedOp>) -> Option<u64> {
        // Check if the root node is a Schedule node
        if let Some(last) = expr.as_ref().last() {
            if let SchedOp::Schedule(handle) = last {
                // Use the schedule's memory address as a unique identifier
                let ptr = Arc::as_ptr(&handle.schedule) as *const _ as u64;
                return Some(ptr);
            }
        }
        None
    }

    /// Calculate a simple cost for an expression (can be replaced with actual cost model)
    fn calculate_cost(&self, expr: &RecExpr<SchedOp>) -> f64 {
        let mut cost = 0.0;

        for node in expr.as_ref() {
            cost += match node {
                SchedOp::Schedule(_) => 1.0,
                SchedOp::Tile(_) => 0.8,        // Tiling is generally good
                SchedOp::Parallel(_) => 0.7,    // Parallelization is good
                SchedOp::Vectorize(_) => 0.6,   // Vectorization is very good
                SchedOp::Interchange(_) => 0.9, // Interchange can be good
                SchedOp::Split(_) => 0.85,
                SchedOp::Fuse(_) => 0.85,
                SchedOp::Unroll(_) => 0.95, // Unrolling has tradeoffs
                _ => 1.0,
            };
        }

        cost
    }
}

/// Validate that a schedule is semantically correct
pub fn validate_schedule(schedule: &Schedule) -> Result<(), String> {
    // Get the schedule tree representation
    let tree_str = schedule.to_str();

    // Basic validation checks
    // 1. Check that the schedule has a domain
    if !tree_str.contains("domain:") {
        return Err("Schedule missing domain".to_string());
    }

    // 2. Check that the schedule tree is well-formed
    // This would ideally use ISL's internal validation
    // For now, we just check basic structure

    // 3. Try to get the schedule map - this will fail if schedule is invalid
    let map = schedule.get_map();
    if map.is_empty() {
        return Err("Schedule produces empty map".to_string());
    }

    Ok(())
}

/// Extract and validate all schedules directly from e-graph's ScheduleData
pub fn extract_and_validate_all(
    egraph: &EGraph<SchedOp, ScheduleAnalysis>,
    _root: Id,
) -> Vec<(f64, RecExpr<SchedOp>, Option<(Schedule, String)>)> {
    println!("\n[EXTRACT] Extracting all schedule candidates from e-graph");
    println!(
        "[EXTRACT] E-graph has {} e-classes, {} nodes",
        egraph.number_of_classes(),
        egraph.total_number_of_nodes()
    );

    let mut results: Vec<(f64, RecExpr<SchedOp>, Option<(Schedule, String)>)> = Vec::new();
    let mut valid_count = 0;
    let mut invalid_count = 0;
    let mut no_schedule_count = 0;

    // Iterate through ALL e-classes to find those with valid schedules
    for class in egraph.classes() {
        // Debug: show what's in each e-class
        println!("[DEBUG] E-class {:?}:", class.id);
        println!("  Has schedule: {}", class.data.schedule.is_some());
        println!("  Has value: {:?}", class.data.value);
        println!("  Has symbol: {:?}", class.data.symbol);
        println!("  Nodes in class: {}", class.nodes.len());
        for (i, node) in class.nodes.iter().take(3).enumerate() {
            println!("    Node {}: {:?}", i, node);
        }

        // Check if this e-class has a schedule in its data
        if let Some(ref schedule_handle) = class.data.schedule {
            // We found an e-class with an evaluated schedule!
            // Extract the actual expression from e-graph instead of creating a simple one
            // Use the standard extraction to get the full expression tree
            let extractor = egg::Extractor::new(egraph, egg::AstSize);
            let (_, best_expr) = extractor.find_best(class.id);

            // Reconstruct the schedule by evaluating the extracted expression
            // This ensures we get the fully transformed schedule, not just the initial one
            let (schedule, tree_string) = if let Some((reconstructed, schedule_str)) =
                evaluate_expression(egraph, &best_expr)
            {
                println!("[EXTRACT DEBUG] Successfully reconstructed schedule from expression");
                // Use the schedule_str from evaluate_expression which was created with block printer
                (reconstructed.schedule.copy(), schedule_str)
            } else {
                // Fallback to the schedule in the e-class data
                println!("[EXTRACT DEBUG] Failed to reconstruct, using fallback schedule");
                // Get the tree string using block printer for the fallback case
                let fallback_str =
                    crate::isl_block_printer::schedule_to_block_str(&schedule_handle.schedule);
                (schedule_handle.schedule.copy(), fallback_str)
            };

            // Debug: Check what schedule we're actually extracting
            if class.nodes.iter().any(|n| matches!(n, SchedOp::Tile(_))) {
                println!("[EXTRACT DEBUG] E-class {:?} has Tile node", class.id);
                // Use serializer to preserve tree structure
                let serialized_debug =
                    crate::schedule_serializer::serialize_schedule_tree(&schedule);
                println!(
                    "[EXTRACT DEBUG] Schedule: {}",
                    serialized_debug.lines().next().unwrap_or("")
                );
            }

            // Validate the schedule
            match validate_schedule(&schedule) {
                Ok(()) => {
                    valid_count += 1;
                    // Calculate cost using NCP-aware ScheduleCost + Communication Cost (Phase 3)
                    // Extract dependencies and access info from e-class data and egraph analysis
                    let dependencies = class.data.dependencies.as_ref();
                    let access_info = egraph.analysis.access_info.as_ref();

                    // Infer domain type from schedule for computation cost calculation
                    let domain_type = crate::communication_cost::infer_domain_type(&schedule);

                    let cost = calculate_schedule_cost_with_comm(
                        schedule_handle,
                        dependencies,
                        access_info,
                        domain_type,
                    );

                    // Print details about valid schedules
                    println!(
                        "[EXTRACT] Valid schedule in e-class {:?} (cost={:.2}):",
                        class.id, cost
                    );
                    let tree_str = schedule.to_str();
                    for line in tree_str.lines().take(5) {
                        println!("  {}", line);
                    }

                    // Use the tree_string we already captured from the ScheduleHandle
                    // This preserves the full tree structure including transformations
                    let schedule_str = tree_string.clone();
                    println!(
                        "[EXTRACT DEBUG] Serialized schedule for saving (first line): {}",
                        schedule_str.lines().next().unwrap_or("")
                    );
                    println!(
                        "[EXTRACT DEBUG] Full schedule_str length: {} chars",
                        schedule_str.len()
                    );
                    if schedule_str.len() < 200 {
                        println!("[EXTRACT DEBUG] WARNING: Schedule string seems truncated!");
                        println!("[EXTRACT DEBUG] Full content: {}", schedule_str);
                    }

                    // Debug: verify we have transformations in the string
                    if schedule_str.contains("mod") {
                        println!("[EXTRACT DEBUG] Schedule string contains tiling transformations");
                    } else {
                        println!("[EXTRACT DEBUG] Schedule string has NO transformations");
                    }

                    // Now push to results after we're done using schedule
                    // IMPORTANT: We store both the schedule and its string representation
                    // because the schedule object loses its tree structure when passed around
                    let schedule_string: String = schedule_str;
                    results.push((cost, best_expr.clone(), Some((schedule, schedule_string))));
                }
                Err(e) => {
                    invalid_count += 1;
                    eprintln!(
                        "[VALIDATE] Schedule in e-class {:?} invalid: {}",
                        class.id, e
                    );
                    results.push((999.0, best_expr, None));
                }
            }
        } else {
            no_schedule_count += 1;
        }
    }

    println!("\n[EXTRACT] Summary:");
    println!("  Total e-classes: {}", egraph.number_of_classes());
    println!(
        "  E-classes with schedules: {}",
        valid_count + invalid_count
    );
    println!("  Valid schedules: {}", valid_count);
    println!("  Invalid schedules: {}", invalid_count);
    println!("  E-classes without schedules: {}", no_schedule_count);

    // Sort by cost
    results.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

    // Deduplicate schedules based on their string representation
    // This ensures we only keep unique schedules
    let mut seen_schedules = HashSet::new();
    let mut unique_results = Vec::new();
    let total_before_dedup = results.len();

    for (cost, expr, schedule_opt) in results.into_iter() {
        if let Some((ref _sched, ref schedule_str)) = schedule_opt {
            // Use the schedule string as the unique identifier
            // This captures the full tree structure with transformations
            if seen_schedules.insert(schedule_str.clone()) {
                // Debug: show what unique schedules we're keeping
                if schedule_str.contains("mod") {
                    println!(
                        "[EXTRACT] Keeping unique schedule with tiling (cost: {})",
                        cost
                    );
                } else if schedule_str.contains("parallel") {
                    println!(
                        "[EXTRACT] Keeping unique schedule with parallelization (cost: {})",
                        cost
                    );
                } else if schedule_str.contains("child:") {
                    println!(
                        "[EXTRACT] Keeping unique structured schedule (cost: {})",
                        cost
                    );
                } else {
                    println!(
                        "[EXTRACT] Keeping unique baseline schedule (cost: {})",
                        cost
                    );
                }

                unique_results.push((cost, expr, schedule_opt));
            } else {
                println!("[EXTRACT] Skipping duplicate schedule");
            }
        } else {
            // Keep invalid schedules for debugging
            unique_results.push((cost, expr, schedule_opt));
        }
    }

    println!("\n[EXTRACT] Deduplication Summary:");
    println!("  Total candidates: {}", total_before_dedup);
    println!("  Unique schedules: {}", unique_results.len());

    unique_results
}

/// Calculate cost based on schedule characteristics using complete NCP cost model
///
/// # Cost Model (Complete - Computation + Communication)
///
/// This function computes the total execution cost of a schedule by combining:
/// 1. **Computation cost**: ISL domain analysis × operation latency / parallelism
/// 2. **Communication cost**: ISL-based dependency-aware analysis (Phase 3)
///
/// ## Computation Cost Formula
///
/// $$T_{\text{compute}} = \frac{|D| \times L_{\text{op}}}{P}$$
///
/// Where:
/// - $|D|$: Iteration space cardinality (extracted from schedule domain)
/// - $L_{\text{op}}$: Operation latency (GEMM FMA ~500 cyc, NTT ModMul ~669 cyc)
/// - $P$: Parallelism factor (detected from schedule markers)
///
/// ## Communication Cost Formula (Phase 3)
///
/// $$T_{\text{comm}} = \sum_{a} \chi(L_a) \left[ \beta V_a + \alpha V_a \cdot \bar{H} \right]$$
///
/// Where:
/// - $\beta = 1/32$ cycles/element (NoC bandwidth)
/// - $\alpha = 5$ cycles/hop (NoC latency)
/// - $V_a$: Communication volume per array
/// - $\bar{H}$: Average Manhattan distance in 2D mesh
///
/// ## Total Cost
///
/// $$T_{\text{total}} = T_{\text{compute}} + T_{\text{comm}}$$
///
/// # Arguments
/// * `schedule_handle` - The schedule to cost
/// * `dependencies` - Optional cached dependency information (for comm cost)
/// * `access_info` - Optional access information (for comm volume)
/// * `domain_type` - Type of computation (GEMM, NTT, Stencil, etc.)
///
/// # Returns
/// Total cost = computation_cost + communication_cost (in cycles)
///
/// # Example (GEMM 64×64×64)
///
/// ```text
/// Computation: 262,144 iters × 500 cyc/FMA / 8 parallel = 16,384,000 cyc
/// Communication: 12,288 elem × 1/32 cyc/elem = 384 cyc
/// Total: 16,384,384 cyc
/// ```
fn calculate_schedule_cost_with_comm(
    schedule_handle: &ScheduleHandle,
    dependencies: Option<&Arc<crate::dependency_aware::DependencyInfo>>,
    access_info: Option<&Arc<crate::access_analysis::AccessInfo>>,
    domain_type: crate::communication_cost::ComputationDomain,
) -> f64 {
    use crate::communication_cost::{
        compute_computation_cost_from_schedule, compute_dependency_aware_communication_cost,
        compute_local_footprint_penalty,
    };

    let schedule_str = schedule_handle.schedule.to_str().to_string();

    // Step 1: Detect parallelism (Phase 4 - precise tiling-based)
    // This now computes ACTUAL parallelism from tiling structure using ISL properties
    let parallelism = schedule_handle.properties.parallelism_factor();

    // Step 2: Compute computation cost (ISL domain-based)
    // This calculates the actual computational work based on iteration space
    let comp_cost =
        compute_computation_cost_from_schedule(&schedule_handle.schedule, domain_type, parallelism)
            .unwrap_or_else(|| {
                // Fallback if domain cardinality cannot be computed
                // Use heuristic based on schedule complexity
                log::warn!("Cannot compute exact computation cost, using heuristic fallback");
                println!("[DEBUG] Fallback used for schedule: {}", schedule_str);
                1000.0 // Conservative estimate
            });

    // DEBUG: Print parallelism and cost
    if parallelism > 1 {
        println!(
            "[DEBUG] Parallelism: {}, Comp Cost: {:.2}",
            parallelism, comp_cost
        );
    }

    // Step 3: Compute communication cost (Phase 3 - dependency-aware)
    // This uses ISL-based analysis to compute actual communication requirements
    let comm_cost = if access_info.is_some() || dependencies.is_some() {
        compute_dependency_aware_communication_cost(
            dependencies,
            access_info,
            &schedule_handle.schedule,
            &schedule_handle.ctx,
        )
    } else {
        // No access/dependency information available, skip communication cost
        0.0
    };

    // Step 4: Compute local footprint penalty (Phase 4.2)
    // This penalizes schedules with large per-NCP memory footprint
    // This makes double-layer tiling (8x8) preferred over single-layer (8x1)
    let footprint_penalty = if let Some(info) = access_info {
        compute_local_footprint_penalty(info, &schedule_handle.schedule, parallelism)
    } else {
        0.0
    };

    // Step 5: Detect reduction dimension tiling penalty (Phase 4.2b)
    // Penalizes k-tiling in GEMM which causes redundant communication
    let reduction_penalty =
        crate::communication_cost::detect_reduction_tiling_penalty(&schedule_str, domain_type);

    // Step 6: Combine costs (Phase 4 - Complete Model)
    // T_total = T_compute + T_comm + C_footprint + C_reduction
    let total_cost = comp_cost + comm_cost + footprint_penalty + reduction_penalty;

    log::debug!("Schedule cost breakdown: compute={:.2}, comm={:.2}, footprint={:.2}, reduction={:.2}, total={:.2}",
               comp_cost, comm_cost, footprint_penalty, reduction_penalty, total_cost);

    total_cost
}

/// Calculate cost based on schedule characteristics using NCP-aware ScheduleCost
/// (Legacy wrapper for backward compatibility - does not include communication cost)
#[allow(dead_code)]
fn calculate_schedule_cost(schedule_handle: &ScheduleHandle) -> f64 {
    use crate::optimize::ScheduleCost;
    use egg::CostFunction;

    // Use the enhanced ScheduleCost which includes NCP-aware factors:
    // - Generic factors: parallelism, vectorization, tiling
    // - NCP-specific factors: tile count (512 NCPs), slice alignment (8 slices),
    //   communication overhead (2D mesh NoC)
    let schedule_op = SchedOp::Schedule(schedule_handle.clone());

    let mut cost_fn = ScheduleCost::new();
    cost_fn.cost(&schedule_op, |_| 0.0)
}

/// Extract k-best schedules based on combined heuristic + communication cost
///
/// # Algorithm (K-Best Extraction)
///
/// This implements the paper's hybrid cost model approach:
/// 1. Compute analytical cost (heuristic + communication) for all valid schedules
/// 2. Extract top-k candidates sorted by analytical cost
/// 3. Return candidates for potential real hardware measurement
///
/// # Arguments
/// * `egraph` - The e-graph containing explored schedules
/// * `k` - Number of best candidates to extract
///
/// # Returns
/// Vector of (cost, schedule) tuples, sorted by cost (best first)
///
/// # Usage in Pipeline
///
/// ```rust
/// use polysat::extract_all::extract_k_best_schedules;
/// use polysat::language::{SchedOp, ScheduleAnalysis, ScheduleHandle};
/// use egg::EGraph;
///
/// let egraph = EGraph::<SchedOp, ScheduleAnalysis>::default();
/// // ... populate egraph ...
/// let candidates = extract_k_best_schedules(&egraph, 20);
///
/// // Mock measurement function
/// fn measure_on_ncp_hardware(sched: &ScheduleHandle) -> f64 { 0.0 }
///
/// let mut best_perf = f64::INFINITY;
/// let mut best_schedule = None;
///
/// for (_cost, schedule, _schedule_str) in candidates {
///     // In a real scenario, you'd convert `schedule` (isl::schedule::Schedule)
///     // into a ScheduleHandle or similar structure expected by your measurement function.
///     // For this example, we'll create a dummy ScheduleHandle.
///     let ctx = egraph.analysis.ctx().clone();
///     let dummy_handle = ScheduleHandle::new(ctx, schedule);
///     let real_perf = measure_on_ncp_hardware(&dummy_handle);
///     if real_perf < best_perf {
///         best_schedule = Some(dummy_handle);
///         best_perf = real_perf;
///     }
/// }
/// ```
pub fn extract_k_best_schedules(
    egraph: &EGraph<SchedOp, ScheduleAnalysis>,
    k: usize,
) -> Vec<(f64, Schedule, String)> {
    println!("\n[K-BEST] Extracting top {} schedule candidates", k);

    let mut all_candidates = Vec::new();

    // Iterate through all e-classes to find valid schedules
    for class in egraph.classes() {
        if let Some(ref schedule_handle) = class.data.schedule {
            let schedule = schedule_handle.schedule.copy();

            // Validate schedule
            if validate_schedule(&schedule).is_err() {
                continue;
            }

            // Compute cost with complete model (computation + communication)
            let dependencies = class.data.dependencies.as_ref();
            let access_info = egraph.analysis.get_access_info();

            // Infer domain type from schedule
            let domain_type = crate::communication_cost::infer_domain_type(&schedule);

            let cost = calculate_schedule_cost_with_comm(
                schedule_handle,
                dependencies,
                access_info,
                domain_type,
            );

            // Get schedule string representation (for deduplication and storage)
            let schedule_str = crate::isl_block_printer::schedule_to_block_str(&schedule);

            all_candidates.push((cost, schedule, schedule_str));
        }
    }

    // Deduplicate based on schedule string
    let mut seen_schedules = HashSet::new();
    let mut unique_candidates = Vec::new();

    for (cost, schedule, schedule_str) in all_candidates {
        if seen_schedules.insert(schedule_str.clone()) {
            unique_candidates.push((cost, schedule, schedule_str));
        }
    }

    // Sort by cost (ascending - lower is better)
    unique_candidates.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    // Take top-k
    let k_best: Vec<(f64, Schedule, String)> = unique_candidates.into_iter().take(k).collect();

    println!("[K-BEST] Extracted {} candidates:", k_best.len());
    for (i, (cost, _, _)) in k_best.iter().enumerate() {
        println!("  Rank {}: cost={:.2}", i + 1, cost);
    }

    k_best
}

/// Extract diverse schedules using different cost models
pub fn extract_diverse_schedules(
    egraph: &EGraph<SchedOp, ScheduleAnalysis>,
) -> Vec<(String, Option<Schedule>)> {
    let mut diverse_schedules = Vec::new();

    println!("\n[DIVERSE] Extracting schedules with different optimization strategies");

    // Strategy 1: Find the most parallel schedule
    let mut most_parallel = None;
    let mut max_parallel_score = 0;

    // Strategy 2: Find the most tiled schedule
    let mut most_tiled = None;
    let mut max_tile_score = 0;

    // Strategy 3: Find the most vectorized schedule
    let mut most_vectorized = None;
    let mut max_vector_score = 0;

    // Strategy 4: Find the simplest valid schedule
    let mut simplest = None;
    let mut min_depth = usize::MAX;

    // Strategy 5: Find the most aggressive (combined optimizations) schedule
    let mut most_aggressive = None;
    let mut max_combined_score = 0;

    for class in egraph.classes() {
        if let Some(ref schedule_handle) = class.data.schedule {
            let schedule = schedule_handle.schedule.copy();

            // Skip invalid schedules
            if validate_schedule(&schedule).is_err() {
                continue;
            }

            let tree_str = schedule.to_str();

            // Count different transformation types
            let parallel_count = tree_str.matches("parallel").count();
            let tile_count = tree_str.matches("tile").count();
            let vector_count = tree_str.matches("vectorize").count();
            let depth = tree_str.matches("child:").count();
            let combined = parallel_count + tile_count * 2 + vector_count * 3;

            // Update best schedules for each strategy
            if parallel_count > max_parallel_score {
                max_parallel_score = parallel_count;
                most_parallel = Some(schedule.copy());
            }

            if tile_count > max_tile_score {
                max_tile_score = tile_count;
                most_tiled = Some(schedule.copy());
            }

            if vector_count > max_vector_score {
                max_vector_score = vector_count;
                most_vectorized = Some(schedule.copy());
            }

            if depth < min_depth && depth > 0 {
                min_depth = depth;
                simplest = Some(schedule.copy());
            }

            if combined > max_combined_score {
                max_combined_score = combined;
                most_aggressive = Some(schedule.copy());
            }
        }
    }

    // Collect results
    if let Some(sched) = most_parallel {
        println!(
            "[DIVERSE] Found parallel-optimized schedule (score: {})",
            max_parallel_score
        );
        diverse_schedules.push(("parallel_optimized".to_string(), Some(sched)));
    }

    if let Some(sched) = most_tiled {
        println!(
            "[DIVERSE] Found tile-optimized schedule (score: {})",
            max_tile_score
        );
        diverse_schedules.push(("tile_optimized".to_string(), Some(sched)));
    }

    if let Some(sched) = most_vectorized {
        println!(
            "[DIVERSE] Found vector-optimized schedule (score: {})",
            max_vector_score
        );
        diverse_schedules.push(("vector_optimized".to_string(), Some(sched)));
    }

    if let Some(sched) = simplest {
        println!("[DIVERSE] Found simplest schedule (depth: {})", min_depth);
        diverse_schedules.push(("simplest".to_string(), Some(sched)));
    }

    if let Some(sched) = most_aggressive {
        println!(
            "[DIVERSE] Found most aggressive schedule (score: {})",
            max_combined_score
        );
        diverse_schedules.push(("most_aggressive".to_string(), Some(sched)));
    }

    println!(
        "[DIVERSE] Extracted {} diverse schedules",
        diverse_schedules.len()
    );

    diverse_schedules
}

// Helper trait to add expressions to RecExpr
#[allow(dead_code)]
trait RecExprExt {
    fn add_expr(&mut self, other: &RecExpr<SchedOp>) -> Id;
}

impl RecExprExt for RecExpr<SchedOp> {
    fn add_expr(&mut self, other: &RecExpr<SchedOp>) -> Id {
        let base = self.as_ref().len();

        for node in other.as_ref() {
            let mut new_node = node.clone();
            // Update child IDs to account for offset
            new_node.for_each_mut(|id| *id = Id::from(Into::<usize>::into(*id) + base));
            self.add(new_node);
        }

        Id::from(self.as_ref().len() - 1)
    }
}
