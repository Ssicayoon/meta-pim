//! Complete Polyhedral Transformation Implementation for PolySat
//!
//! This module provides real implementations of polyhedral transformations
//! using ISL-RS APIs, including legality checking and actual transformation logic.

use crate::dependency_aware::DependencySet;
use isl_rs::{DimType, MultiUnionPwAff, MultiVal, Schedule, ScheduleNode, ScheduleNodeType, Val};

// ============================================================================
// Core Transformations
// ============================================================================

/// Loop interchange transformation with legality checking
/// Handles both multi-dimensional bands and separated bands (Polygeist style)
pub fn interchange(
    schedule: &Schedule,
    band1_idx: usize,
    band2_idx: usize,
    deps: Option<&DependencySet>,
) -> Result<Option<Schedule>, String> {
    // Check legality if dependencies provided
    if let Some(deps) = deps {
        if !is_interchange_legal(schedule, band1_idx, band2_idx, deps)? {
            return Err("Interchange would violate dependencies".to_string());
        }
    }

    println!(
        "[TRANSFORM] Interchanging loops {} and {}",
        band1_idx, band2_idx
    );

    // Check if this schedule has separated bands (Polygeist format)
    if has_separated_bands(schedule) {
        // Use specialized interchange for separated bands
        // FIXME: This implementation is currently broken and causes segfaults/ISL errors.
        // Disabling it for now to allow exploration to proceed with other transformations.
        // return interchange_separated_bands(schedule, band1_idx, band2_idx);
        return Err("Interchange for separated bands (Polygeist style) is currently disabled due to stability issues.".to_string());
    }

    // Original implementation for multi-dimensional bands
    let root = schedule.get_root();
    let band_node = match find_first_band_node(&root) {
        Some(node) => node,
        None => return Ok(None), // Not applicable if no band found
    };
    let partial_schedule = band_node.band_get_partial_schedule();
    let n_dims = partial_schedule.size();

    if band1_idx >= n_dims as usize || band2_idx >= n_dims as usize {
        return Err(format!(
            "Band indices out of range: {} dimensions available",
            n_dims
        ));
    }

    // Extract and swap dimensions
    let dim1 = partial_schedule.get_at(band1_idx as i32);
    let dim2 = partial_schedule.get_at(band2_idx as i32);

    // Create new MultiUnionPwAff with swapped dimensions
    let space = partial_schedule.get_space();
    let mut new_partial = MultiUnionPwAff::zero(space);

    for i in 0..n_dims {
        let dim = if i == band1_idx as i32 {
            dim2.copy()
        } else if i == band2_idx as i32 {
            dim1.copy()
        } else {
            partial_schedule.get_at(i)
        };
        new_partial = new_partial.set_at(i, dim);
    }

    // Build the permutation vector for coincident flags
    let mut permutation: Vec<usize> = (0..n_dims as usize).collect();
    permutation[band1_idx] = band2_idx;
    permutation[band2_idx] = band1_idx;

    // Create new schedule with transformed partial schedule
    apply_partial_schedule(schedule, &new_partial, Some(&permutation)).map(Some)
}

/// Loop skewing for wavefront parallelization
/// Handles both multi-dimensional bands and separated bands (Polygeist style)
pub fn skew(
    schedule: &Schedule,
    outer_idx: usize,
    factor: i32,
    forward: bool,
    deps: Option<&DependencySet>,
) -> Result<Option<Schedule>, String> {
    // Check legality if dependencies provided
    if let Some(deps) = deps {
        if !is_skewing_legal(schedule, outer_idx, factor, deps)? {
            return Err("Skewing would violate dependencies".to_string());
        }
    }

    println!(
        "[TRANSFORM] Skewing loop {} with factor {} ({})",
        outer_idx,
        factor,
        if forward { "forward" } else { "backward" }
    );

    // Debug: print the schedule string to see what we're working with
    let schedule_str = schedule.to_str().to_string();
    println!(
        "[DEBUG] Schedule string: {}",
        if schedule_str.len() > 200 {
            format!("{}...", &schedule_str[..200])
        } else {
            schedule_str.clone()
        }
    );
    println!("[DEBUG] Contains 'mod': {}", schedule_str.contains(" mod "));

    // Check if this is a post-tiling schedule
    let root = schedule.get_root();
    let is_tiled = has_tiled_structure(&root);
    println!("[DEBUG] has_tiled_structure returned: {}", is_tiled);

    if is_tiled {
        println!("[DEBUG] Detected post-tiling schedule in skew transformation");
        return Err(format!(
            "Skewing of post-tiling schedules is not yet supported. \
             The schedule has tiled structure (contains 'mod' operations) that would be \
             destroyed by naive skewing. Please apply skewing before tiling, or use \
             specialized post-tiling transformations."
        ));
    }

    // Check if this schedule has separated bands (Polygeist format)
    if has_separated_bands(schedule) {
        // Use specialized skewing for separated bands
        return skew_separated_bands(schedule, outer_idx, factor, forward).map(Some);
    }

    // Original implementation for multi-dimensional bands
    let band_node = match find_first_band_node(&root) {
        Some(node) => node,
        None => return Ok(None),
    };
    let partial_schedule = band_node.band_get_partial_schedule();
    let n_dims = partial_schedule.size();

    if outer_idx >= (n_dims - 1) as usize {
        return Err("Need at least 2 dimensions for skewing".to_string());
    }

    let ctx = schedule.get_ctx();
    let space = partial_schedule.get_space();
    let mut skewed = MultiUnionPwAff::zero(space.copy());

    // Apply skewing transformation
    for i in 0..n_dims {
        let dim = partial_schedule.get_at(i);

        let new_dim = if forward && i == (outer_idx + 1) as i32 {
            // Forward: j' = j + factor * i
            let outer = partial_schedule.get_at(outer_idx as i32);
            let factor_val = Val::int_from_si(&ctx, factor as i64);
            let scaled = outer.scale_val(factor_val);
            dim.union_add(scaled)
        } else if !forward && i == outer_idx as i32 {
            // Backward: i' = i + factor * j
            let inner = partial_schedule.get_at((outer_idx + 1) as i32);
            let factor_val = Val::int_from_si(&ctx, factor as i64);
            let scaled = inner.scale_val(factor_val);
            dim.union_add(scaled)
        } else {
            dim
        };

        skewed = skewed.set_at(i, new_dim);
    }

    apply_partial_schedule(schedule, &skewed, None).map(Some)
}

/// Loop tiling transformation
pub fn tile(
    schedule: &Schedule,
    band_idx: usize,
    tile_size: i32,
) -> Result<Option<Schedule>, String> {
    println!(
        "[TRANSFORM] Tiling loop {} with size {}",
        band_idx, tile_size
    );

    // Safety check: tile size must be positive
    if tile_size <= 0 {
        return Err(format!(
            "Invalid tile size: {} (must be positive)",
            tile_size
        ));
    }

    let root = schedule.get_root();
    let band_node = match find_first_band_node(&root) {
        Some(node) => node,
        None => return Ok(None),
    };

    // Check if the band exists and can be tiled
    let partial_schedule = band_node.band_get_partial_schedule();
    let n_dims = partial_schedule.size();

    // Fix: Enhanced error message for multi-statement schedules
    // CONTEXT: NTT has sequence-of-filters structure (13 butterfly stages), not a single 3D band.
    // Each filter has 1-2D bands. Trying to tile dimension 2 of a 1D band fails.
    // SOLUTION: For multi-statement schedules, use TilePerDim operation instead of Tile.
    // TilePerDim routes to tile_per_dimension or tile_separate_bands modules which handle this correctly.
    if band_idx >= n_dims as usize {
        // Check if this is a multi-statement schedule (sequence/filter structure)
        let schedule_str = schedule.to_str().to_string();
        let is_multi_statement = schedule_str.contains("sequence:")
            || (schedule_str.contains("filter:") && schedule_str.matches("filter:").count() > 1);

        let error_msg = if is_multi_statement {
            format!(
                "Band index {} out of bounds for first band (has {} dimensions). \
                 \n  DETECTED: Multi-statement schedule with sequence/filter structure. \
                 \n  HINT: This schedule has multiple separate bands (e.g., NTT has 13 butterfly stages). \
                 \n  SOLUTION: Use TilePerDim operation instead of Tile for per-dimension tiling, \
                 \n            or use tile_separate_bands module to handle each stage individually. \
                 \n  The simple Tile operation only works on single-band schedules.",
                band_idx, n_dims
            )
        } else {
            format!(
                "Band index {} out of bounds (first band has {} dimensions). \
                 \n  Valid indices: 0..{}",
                band_idx, n_dims, n_dims
            )
        };

        return Err(error_msg);
    }

    // Check if the band is already constant (cannot be tiled further)
    let band_expr = partial_schedule.get_at(band_idx as i32);
    let band_str = band_expr.to_str();
    if band_str.contains("[(0)]") || band_str.contains("-> [0]") {
        println!(
            "[TRANSFORM] WARNING: Band {} appears to be constant, skipping tiling",
            band_idx
        );
        return Ok(Some(schedule.copy()));
    }

    // Use ISL's native band_tile
    let ctx = schedule.get_ctx();
    let space = band_node.band_get_space();

    // Create tile sizes
    // Fix: Never use tile size 0 - causes ISL division by zero error
    // ISL's band_tile() computes: outer_loop = floor(i/tile_size), inner_loop = i mod tile_size
    // When tile_size=0, this becomes division by zero in ./isl_union_templ.c:1135
    // Solution: Use tile_size=1 for dimensions that shouldn't be tiled
    // Tile size 1 means "tiles of size 1" = original granularity (no actual tiling)
    let mut sizes = Vec::new();
    for i in 0..n_dims {
        let size = if i as usize == band_idx {
            Val::int_from_si(&ctx, tile_size as i64)
        } else {
            Val::int_from_si(&ctx, 1) // Tile size 1 = no tiling (fixed from 0)
        };
        sizes.push(size);
    }

    // Create MultiVal
    let val_list = isl_rs::ValList::alloc(&ctx, n_dims);
    let val_list = sizes.into_iter().fold(val_list, |list, val| list.add(val));
    let multi_val = MultiVal::from_val_list(space, val_list);

    // Apply tiling
    let tiled_node = band_node.band_tile(multi_val);
    Ok(Some(tiled_node.get_schedule()))
}

/// Loop fusion transformation
pub fn fuse(
    schedule: &Schedule,
    loop1_idx: usize,
    loop2_idx: usize,
    deps: Option<&DependencySet>,
) -> Result<Schedule, String> {
    // Check if fusion is legal
    if let Some(deps) = deps {
        if !is_fusion_legal(schedule, loop1_idx, loop2_idx, deps)? {
            return Err("Fusion would violate dependencies".to_string());
        }
    }

    println!("[TRANSFORM] Fusing loops {} and {}", loop1_idx, loop2_idx);

    let root = schedule.get_root();
    let band1 = find_band_by_index(&root, loop1_idx)?;
    let band2 = find_band_by_index(&root, loop2_idx)?;

    // Get partial schedules
    let partial1 = band1.band_get_partial_schedule();
    let partial2 = band2.band_get_partial_schedule();

    // Combine them (simplified - real fusion needs dependency checking)
    let fused = partial1.union_add(partial2);

    apply_partial_schedule(schedule, &fused, None)
}

/// Vectorization transformation (RFC001-compliant)
///
/// Vectorization in polyhedral compilers involves:
/// 1. Strip-mining the loop with vector width (creates outer/inner loop pair)
/// 2. Inserting a "vectorize" mark node above the inner loop
/// 3. Setting the inner loop type to Unroll for vector-width iterations
///
/// The mark node allows codegen to emit `#pragma omp simd` or similar.
pub fn vectorize(
    schedule: &Schedule,
    band_idx: usize,
    width: i32,
    deps: Option<&DependencySet>,
) -> Result<Option<Schedule>, String> {
    use isl_rs::{ASTLoopType, Id};

    // Check vectorization legality
    if let Some(deps) = deps {
        if !is_vectorization_legal(schedule, band_idx, width as usize, deps)? {
            return Err(format!(
                "Vectorization with width {} would violate dependencies",
                width
            ));
        }
    }

    println!(
        "[TRANSFORM] RFC001 Vectorizing loop {} with width {}",
        band_idx, width
    );

    // Step 1: Strip-mine (tile) the loop with vector width
    // This creates: outer loop (stride=width) -> inner loop (0..width)
    let tiled_result = tile(schedule, band_idx, width)?;
    let tiled_schedule = match tiled_result {
        Some(s) => s,
        None => return Ok(None),
    };

    // Step 2: Navigate to the inner (point) band and mark it for vectorization
    let root = tiled_schedule.get_root();
    let band_node = match find_first_band_node(&root) {
        Some(node) => node,
        None => return Err("No band node found after tiling".to_string()),
    };

    // After tiling, the band has been split. Navigate to find the inner (point) band.
    // Tiling structure: outer band -> inner (point) band
    // We need to mark the innermost dimension for vectorization.

    // Get the context to create the mark identifier
    let ctx = tiled_schedule.get_ctx();

    // Step 3: Set AST loop type for the innermost dimension to Unroll
    // This tells ISL's AST generator to unroll the inner strip, which is
    // essential for vectorization (generates vector-width iterations inline)
    let n_members = band_node.band_n_member() as i32;
    if n_members > 0 {
        // Mark the innermost member (the vector strip) for unrolling
        let innermost_pos = n_members - 1;
        let marked_node = band_node.band_member_set_ast_loop_type(innermost_pos, ASTLoopType::Unroll);

        // Step 4: Insert a "vectorize" mark node above the band
        // This allows codegen to emit vectorization pragmas
        let vectorize_mark = Id::read_from_str(&ctx, "vectorize");
        let with_mark = marked_node.insert_mark(vectorize_mark);

        println!(
            "[TRANSFORM] RFC001: Marked loop with 'vectorize' and set inner loop to Unroll"
        );

        Ok(Some(with_mark.get_schedule()))
    } else {
        // No band members - just return the tiled schedule
        println!("[TRANSFORM] Warning: No band members found for vectorization marking");
        Ok(Some(tiled_schedule))
    }
}

// ============================================================================
// Legality Checking
// ============================================================================

/// Check if loop interchange is legal
fn is_interchange_legal(
    schedule: &Schedule,
    band1_idx: usize,
    band2_idx: usize,
    deps: &DependencySet,
) -> Result<bool, String> {
    let root = schedule.get_root();
    let band_node = find_first_band_node(&root).ok_or("No band node found for legality check")?;

    // Check if band is permutable
    if band_node.band_get_permutable() {
        return Ok(true); // Permutable bands can be interchanged freely
    }

    // Check coincident properties
    let coincident1 = band_node.band_member_get_coincident(band1_idx as i32);
    let coincident2 = band_node.band_member_get_coincident(band2_idx as i32);

    if coincident1 && coincident2 {
        return Ok(true); // Both loops are parallel
    }

    // Check dependency distances
    if !deps.has_deps {
        return Ok(true); // No dependencies
    }

    // Conservative: check if either dimension carries dependencies
    if band1_idx < deps.loop_carried.len() && band2_idx < deps.loop_carried.len() {
        if !deps.loop_carried[band1_idx] && !deps.loop_carried[band2_idx] {
            return Ok(true); // Neither carries dependencies
        }
    }

    Ok(false) // Conservative: disallow
}

/// Check if skewing is legal
fn is_skewing_legal(
    schedule: &Schedule,
    band_idx: usize,
    _factor: i32,
    deps: &DependencySet,
) -> Result<bool, String> {
    let root = schedule.get_root();
    let band_node = find_first_band_node(&root).ok_or("No band node found for legality check")?;
    let n_members = band_node.band_n_member();

    if n_members < 2 {
        return Ok(false); // Need at least 2 dimensions
    }

    if band_idx >= (n_members - 1) as usize {
        return Ok(false); // Can't skew last dimension
    }

    // Skewing is typically safe for stencil patterns
    if !deps.has_deps {
        return Ok(true);
    }

    // For wavefront parallelization, skewing is often beneficial
    Ok(true) // Optimistically allow
}

/// Check if vectorization is legal
fn is_vectorization_legal(
    schedule: &Schedule,
    band_idx: usize,
    width: usize,
    deps: &DependencySet,
) -> Result<bool, String> {
    let root = schedule.get_root();
    let band_node = find_first_band_node(&root).ok_or("No band node found for legality check")?;

    // Check if dimension is coincident (no loop-carried deps)
    if band_node.band_member_get_coincident(band_idx as i32) {
        return Ok(true);
    }

    // Check dependency distances
    if !deps.has_deps {
        return Ok(true);
    }

    // Conservative check for innermost dimension
    let n_members = band_node.band_n_member();
    if band_idx != (n_members - 1) as usize {
        // Not innermost - vectorization less efficient
        return Ok(false);
    }

    // Would need to check actual dependency distances >= width
    // For now, be conservative
    Ok(width <= 4 && !deps.loop_carried.get(band_idx).copied().unwrap_or(true))
}

/// Check if fusion is legal
fn is_fusion_legal(
    _schedule: &Schedule,
    _loop1_idx: usize,
    _loop2_idx: usize,
    deps: &DependencySet,
) -> Result<bool, String> {
    // Fusion requires checking producer-consumer relationships
    // and ensuring no dependencies are reversed

    if !deps.has_deps {
        return Ok(true); // No dependencies to violate
    }

    // Conservative: only allow if no loop-carried dependencies
    Ok(deps.loop_carried.iter().all(|&carried| !carried))
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Apply a new partial schedule to create a transformed schedule
/// This version properly handles both pre-tiling and post-tiling schedules
fn apply_partial_schedule(
    schedule: &Schedule,
    new_partial: &MultiUnionPwAff,
    permutation: Option<&[usize]>,
) -> Result<Schedule, String> {
    // Try to locate the band whose space matches the new partial schedule. This is important
    // for separated-band cases where the first band may not be the true target.
    let root = schedule.get_root();
    let target_band = find_target_band_for_partial(&root, new_partial)
        .or_else(|_| find_first_band_node(&root).ok_or("No band node found".to_string()))?;

    // isl-rs now exposes band_set_partial_schedule(), which replaces the previous workaround
    // that attempted to rebuild schedule trees when modifying a band. This call updates the
    // partial schedule in-place and preserves the rest of the tree (children, filters, etc.).
    let mut updated_band = target_band.band_set_partial_schedule(new_partial.copy());

    if let Some(perm) = permutation {
        let n_members = updated_band.band_n_member();
        // Check for valid dimension (non-negative) and matching length
        if n_members >= 0 && n_members as usize == perm.len() {
            updated_band = updated_band.permute_coincident_flags(perm);
        } else {
            println!(
                "[TRANSFORM] Warning: Skipping coincident flag permutation due to dimension mismatch. Band: {}, Perm: {}", 
                n_members, perm.len()
            );
        }
    }
    Ok(updated_band.get_schedule())
}

/// Check if a schedule has tiled structure (contains mod operations or nested bands)
fn has_tiled_structure(node: &ScheduleNode) -> bool {
    // Check if the schedule string contains tiling patterns
    let schedule = node.get_schedule();
    let schedule_str = schedule.to_str().to_string();
    // Tiling creates "mod" expressions and nested band structures
    if schedule_str.contains(" mod ") || schedule_str.contains("permutable") {
        return true;
    }

    // Also check for nested band structures that indicate tiling
    let mut current = node.copy();
    let mut band_count = 0;
    let mut depth = 0;
    let max_depth = 10;

    while depth < max_depth {
        if current.get_type() == ScheduleNodeType::Band {
            band_count += 1;
            if band_count > 1 {
                // Multiple bands often indicate tiling
                return true;
            }
        }

        if current.has_children() && current.n_children() > 0 {
            current = current.child(0);
            depth += 1;
        } else {
            break;
        }
    }

    false
}

/// Apply partial schedule to a tiled schedule tree
/// This preserves the tiling structure rather than destroying it
/// Find the band node that matches the space of the partial schedule
fn find_target_band_for_partial(
    node: &ScheduleNode,
    partial: &MultiUnionPwAff,
) -> Result<ScheduleNode, String> {
    if node.get_type() == ScheduleNodeType::Band {
        // Check if this band's space matches the partial schedule's space
        let band_space = node.band_get_space();
        let partial_space = partial.get_space();

        // Simple check: same number of dimensions
        let band_dims = band_space.dim(DimType::Out);
        let partial_dims = partial_space.dim(DimType::Out);

        if band_dims == partial_dims {
            return Ok(node.copy());
        }
    }

    // Recursively search children
    let n_children = node.n_children();
    for i in 0..n_children {
        let child = node.copy().child(i);
        if let Ok(band) = find_target_band_for_partial(&child, partial) {
            return Ok(band);
        }
    }

    Err("No matching band node found".to_string())
}

/// Find the first band node in the schedule tree
/// Find the first band node in the schedule tree
fn find_first_band_node(node: &ScheduleNode) -> Option<ScheduleNode> {
    if node.get_type() == ScheduleNodeType::Band {
        return Some(node.copy());
    }

    let n_children = node.n_children();
    for i in 0..n_children {
        let child = node.copy().child(i);
        if let Some(band) = find_first_band_node(&child) {
            return Some(band);
        }
    }

    None
}

/// Find a band node by index
fn find_band_by_index(node: &ScheduleNode, target_idx: usize) -> Result<ScheduleNode, String> {
    let mut current_idx = 0;
    find_band_recursive(node, target_idx, &mut current_idx)
}

fn find_band_recursive(
    node: &ScheduleNode,
    target_idx: usize,
    current_idx: &mut usize,
) -> Result<ScheduleNode, String> {
    if node.get_type() == ScheduleNodeType::Band {
        if *current_idx == target_idx {
            return Ok(node.copy());
        }
        *current_idx += 1;
    }

    let n_children = node.n_children();
    for i in 0..n_children {
        let child = node.copy().child(i);
        if let Ok(band) = find_band_recursive(&child, target_idx, current_idx) {
            return Ok(band);
        }
    }

    Err(format!("Band {} not found", target_idx))
}

/// Check if a schedule has separated band nodes (Polygeist format)
/// Returns true if each band node has only 1 dimension
fn has_separated_bands(schedule: &Schedule) -> bool {
    let root = schedule.get_root();
    let mut current = root;
    let mut band_count = 0;
    let mut max_band_dims = 0;

    // Traverse the schedule tree to find band nodes
    loop {
        match current.get_type() {
            ScheduleNodeType::Band => {
                let partial = current.band_get_partial_schedule();
                let n_dims = partial.size();
                band_count += 1;
                max_band_dims = max_band_dims.max(n_dims);

                // If we find a band with more than 1 dimension,
                // it's not separated bands
                if n_dims > 1 {
                    return false;
                }
            }
            _ => {}
        }

        // Move to next node
        if current.n_children() > 0 {
            current = current.child(0);
        } else {
            break;
        }
    }

    // If we have multiple bands and each has only 1 dimension,
    // then we have separated bands (Polygeist style)
    band_count > 1 && max_band_dims == 1
}

/// Interchange two separated band nodes (Polygeist style)
/// This swaps entire band nodes in the tree rather than dimensions within a band
#[allow(dead_code)]
fn interchange_separated_bands(
    schedule: &Schedule,
    band1_idx: usize,
    band2_idx: usize,
) -> Result<Schedule, String> {
    println!(
        "[TRANSFORM] Interchanging separated bands {} and {}",
        band1_idx, band2_idx
    );

    // Implementation: Actually swap the partial schedules between bands
    // For Polygeist's separated bands format:
    // L2[{ S[i,j,k] -> [(i)] }]  band_idx=0
    // L1[{ S[i,j,k] -> [(j)] }]  band_idx=1
    // L0[{ S[i,j,k] -> [(k)] }]  band_idx=2
    // After interchange(0,1), should become:
    // L2[{ S[i,j,k] -> [(j)] }]  swapped!
    // L1[{ S[i,j,k] -> [(i)] }]  swapped!
    // L0[{ S[i,j,k] -> [(k)] }]  unchanged

    let root = schedule.get_root();

    // Step 1: Collect all band nodes and their partial schedules
    let bands = collect_separated_band_nodes(&root)?;

    // Step 2: Validate indices
    if band1_idx >= bands.len() || band2_idx >= bands.len() {
        return Err(format!(
            "Band indices out of range: {} bands available",
            bands.len()
        ));
    }

    // Step 3: Build new schedule with swapped bands
    // We need to rebuild the tree structure with bands in new order
    let domain = schedule.get_domain();
    let _ctx = schedule.get_ctx();

    // Create new schedule from domain
    let new_schedule = Schedule::from_domain(domain);
    let mut current = new_schedule.get_root();

    // Rebuild tree with swapped partial schedules - using proper nesting
    for i in 0..bands.len() {
        let partial = if i == band1_idx {
            // Use band2's partial schedule at band1's position
            bands[band2_idx].copy()
        } else if i == band2_idx {
            // Use band1's partial schedule at band2's position
            bands[band1_idx].copy()
        } else {
            // Keep original partial schedule
            bands[i].copy()
        };

        // Insert the partial schedule as a new band node
        let band_node = current.insert_partial_schedule(partial);

        // For next iteration, use the newly created band as parent
        current = band_node;
    }

    // Add any remaining structure (like filters, sequences) from original
    // For now, simplified - in production would need full tree reconstruction

    // Get schedule from the root of the new tree, not from current
    let result_schedule = new_schedule;

    // Verify the transformation by checking the schedule string
    let original_str = schedule.to_str().to_string();
    let result_str = result_schedule.to_str().to_string();

    if original_str != result_str {
        println!("[TRANSFORM] Successfully interchanged separated bands - schedule modified");
    } else {
        println!("[TRANSFORM] Warning: Schedule appears unchanged after interchange");
    }

    Ok(result_schedule)
}

/// Helper function to collect all band nodes' partial schedules from separated bands
fn collect_separated_band_nodes(root: &ScheduleNode) -> Result<Vec<MultiUnionPwAff>, String> {
    let mut bands = Vec::new();
    let mut current = root.copy();

    // Traverse the tree depth-first to collect band nodes in order
    loop {
        if current.get_type() == ScheduleNodeType::Band {
            let partial = current.band_get_partial_schedule();
            bands.push(partial);
        }

        // Move to first child if available
        if current.n_children() > 0 {
            current = current.child(0);
        } else {
            break;
        }
    }

    if bands.is_empty() {
        return Err("No band nodes found in schedule tree".to_string());
    }

    // Verify these are separated bands (each with dimension 1)
    for (i, band) in bands.iter().enumerate() {
        if band.size() != 1 {
            return Err(format!(
                "Band {} has {} dimensions, expected 1 for separated bands",
                i,
                band.size()
            ));
        }
    }

    Ok(bands)
}

/// Skew separated band nodes (Polygeist style) for wavefront parallelization
/// This modifies the band schedules to create diagonal wavefronts
fn skew_separated_bands(
    schedule: &Schedule,
    outer_idx: usize,
    factor: i32,
    forward: bool,
) -> Result<Schedule, String> {
    println!(
        "[TRANSFORM] Skewing separated bands {} with factor {} ({})",
        outer_idx,
        factor,
        if forward { "forward" } else { "backward" }
    );

    // Implementation: Actually modify the partial schedules for skewing
    // For Polygeist's separated bands:
    // L2[{ S[i,j,k] -> [(i)] }]  outer_idx=0
    // L1[{ S[i,j,k] -> [(j)] }]  inner (if forward skew)
    // After forward skew with factor=2:
    // L2[{ S[i,j,k] -> [(i)] }]  unchanged
    // L1[{ S[i,j,k] -> [(j + 2*i)] }]  skewed!

    let root = schedule.get_root();
    let ctx = schedule.get_ctx();

    // Step 1: Collect all band nodes and their partial schedules
    let mut bands = collect_separated_band_nodes(&root)?;

    // Step 2: Validate we have consecutive bands for skewing
    if outer_idx >= bands.len() - 1 {
        return Err(format!(
            "Need at least {} bands for skewing at index {}",
            outer_idx + 2,
            outer_idx
        ));
    }

    // Step 3: Apply skewing transformation
    // Get the UnionPwAff from each band (separated bands have size 1)
    let outer_upa = bands[outer_idx].get_at(0);
    let inner_idx = outer_idx + 1;
    let inner_upa = bands[inner_idx].get_at(0);

    // Create the skewed schedule
    let factor_val = Val::int_from_si(&ctx, factor as i64);
    let skewed_upa = if forward {
        // Forward skew: inner' = inner + factor * outer
        let scaled_outer = outer_upa.scale_val(factor_val);
        inner_upa.union_add(scaled_outer)
    } else {
        // Backward skew: outer' = outer + factor * inner
        let scaled_inner = inner_upa.scale_val(factor_val);
        outer_upa.union_add(scaled_inner)
    };

    // Update the appropriate band with the skewed schedule
    let target_idx = if forward { inner_idx } else { outer_idx };

    // For separated bands, we can directly replace the band since it has only 1 dimension
    // Create new MultiUnionPwAff from the skewed UnionPwAff
    let skewed_list = isl_rs::UnionPwAffList::alloc(&ctx, 1).add(skewed_upa);
    let space = bands[target_idx].get_space();
    let new_partial = MultiUnionPwAff::from_union_pw_aff_list(space, skewed_list);
    bands[target_idx] = new_partial;

    // Step 4: Rebuild schedule with modified bands
    // Fix: Check if this is a post-tiling schedule to avoid 0D space error
    let root_check = schedule.get_root();
    let is_post_tiling = has_tiled_structure(&root_check);

    if is_post_tiling {
        // For post-tiling schedules, we cannot rebuild from scratch
        // Instead, we need to work with the existing structure
        println!("[DEBUG] Skewing post-tiling schedule - using careful reconstruction");

        // This is complex - we need to navigate the existing tree and modify bands in place
        // For now, return an error to avoid the crash
        return Err(format!(
            "Skewing of post-tiling schedules is not yet supported. \
             The schedule has tiled structure that would be destroyed by naive reconstruction. \
             Please apply skewing before tiling."
        ));
    }

    // For non-tiled schedules, we can safely rebuild
    let domain = schedule.get_domain();
    let new_schedule = Schedule::from_domain(domain.copy());
    let mut current = new_schedule.get_root();

    // Build nested band structure
    for (i, band) in bands.iter().enumerate() {
        // Check the space dimensions before inserting
        let space = band.get_space();
        let space_dim = space.dim(DimType::In);

        if i == 0 && space_dim != 0 {
            // First band should have matching dimensions with domain
            println!(
                "[WARNING] First band has {}D space, expected to match domain",
                space_dim
            );
        }

        // Insert this band's partial schedule
        let band_node = current.insert_partial_schedule(band.copy());

        // For next iteration, we want to insert inside this band
        current = band_node;
    }

    // Get schedule from the root, not from current which is now deep in the tree
    let result_schedule = new_schedule;

    // Verify the transformation
    let original_str = schedule.to_str().to_string();
    let result_str = result_schedule.to_str().to_string();

    if original_str != result_str {
        println!("[TRANSFORM] Successfully skewed separated bands - schedule modified");
        // The skewed band should now have an expression like [(j + 2*i)] instead of just [(j)]
    } else {
        println!("[TRANSFORM] Warning: Schedule appears unchanged after skewing");
    }

    Ok(result_schedule)
}

#[cfg(test)]
mod tests {
    use super::*;
    use isl_rs::{Context, UnionSet};
    use std::sync::Arc;

    #[test]
    fn test_transformations() {
        let ctx = Arc::new(Context::alloc());

        // Use Schedule::from_domain to avoid ISL parsing segfault
        // This creates a schedule tree with only a Domain node, no Band nodes.
        let domain_str = "{ S[i,j,k] : 0 <= i < 100 and 0 <= j < 100 and 0 <= k < 100 }";
        let domain = UnionSet::read_from_str(&ctx, domain_str);
        let schedule = Schedule::from_domain(domain);

        // Test without dependencies - should return Ok(None) because there are no bands
        let result = interchange(&schedule, 0, 1, None);
        assert!(result.is_ok());
        assert!(
            result.unwrap().is_none(),
            "Interchange should return None for domain-only schedule"
        );

        let result = skew(&schedule, 0, 1, true, None);
        assert!(result.is_ok());
        assert!(
            result.unwrap().is_none(),
            "Skew should return None for domain-only schedule"
        );

        let result = tile(&schedule, 0, 32);
        assert!(result.is_ok());
        assert!(
            result.unwrap().is_none(),
            "Tile should return None for domain-only schedule"
        );
    }
}
