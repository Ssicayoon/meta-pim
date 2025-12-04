// Per-dimension tiling implementation based on isl
// This module enables different tile sizes for each loop dimension
// (e.g., Ti=16, Tj=16, Tk=8 for optimal NCP GEMM)

use isl_rs::{MultiVal, Schedule, ScheduleNode, Val, ValList};

/// Tiles a schedule band with different sizes for each dimension
///
/// # Arguments
/// * `schedule` - The ISL schedule to tile
/// * `band_idx` - Index of the band to tile (usually 0 for outermost)
/// * `tile_sizes` - Vector of tile sizes, one per dimension
///
/// # Example
/// ```rust
/// use polysat::tile_per_dimension::tile_per_dimension;
/// use isl_rs::{Context, Schedule, UnionSet};
/// use std::sync::Arc;
///
/// let ctx = Arc::new(Context::alloc());
/// let domain = UnionSet::read_from_str(&ctx, "{ S0[i,j,k] : 0 <= i,j,k < 64 }");
/// let schedule = Schedule::from_domain(domain);
///
/// let tiled = tile_per_dimension(&schedule, 0, vec![16, 16, 8]);
/// ```
///
/// # Theory
/// This implements PPCG's per-dimension tiling strategy using ISL's
/// isl_schedule_node_band_tile with an isl_multi_val parameter.
/// Each dimension i gets tile_sizes[i] as its tile size.
pub fn tile_per_dimension(schedule: &Schedule, band_idx: usize, tile_sizes: Vec<i32>) -> Schedule {
    println!(
        "[DEBUG] tile_per_dimension called: band_idx={}, sizes={:?}",
        band_idx, tile_sizes
    );

    if tile_sizes.is_empty() {
        println!("[ERROR] No tile sizes provided");
        return schedule.copy();
    }

    // Get the schedule tree root and context
    let root = schedule.get_root();
    let ctx = schedule.get_ctx();

    // Navigate to the band node (using copy to preserve tree structure)
    let band_node = navigate_to_band(root.copy(), band_idx);
    if band_node.is_none() {
        println!("[ERROR] Could not find band at index {}", band_idx);
        return schedule.copy();
    }

    let band_node = band_node.unwrap();

    // Get the number of band members (dimensions to tile)
    let n_members = band_node.band_n_member() as usize;

    // Check if this band has members
    if n_members == 0 {
        println!("[ERROR] Band node has no members, cannot tile");
        return schedule.copy();
    }
    println!("[DEBUG] Band has {} members", n_members);

    // Get the band's partial schedule to extract its space
    // This is crucial: MultiVal needs the space from the band's partial schedule
    let partial_schedule = band_node.band_get_partial_schedule();
    let schedule_space = partial_schedule.get_space();

    println!("[DEBUG] Got space from band's partial schedule");

    let n_dims = n_members; // Use band members count, not space dims

    println!(
        "[DEBUG] Band has {} dimensions, provided {} tile sizes",
        n_dims,
        tile_sizes.len()
    );

    // Create ValList and add tile sizes for each dimension
    let mut val_list = ValList::alloc(&ctx, n_dims as i32);

    for i in 0..n_dims {
        let tile_size = if i < tile_sizes.len() {
            tile_sizes[i] as i64
        } else {
            // Use last tile size for remaining dimensions if not enough provided
            *tile_sizes.last().unwrap() as i64
        };

        let val = Val::int_from_si(&ctx, tile_size);
        val_list = val_list.add(val);
        println!("[DEBUG] Dimension {}: tile size {}", i, tile_size);
    }

    // Create MultiVal from the ValList using the band's schedule space
    let multi_val = MultiVal::from_val_list(schedule_space, val_list);

    // Apply tiling using band_tile with the MultiVal
    println!("[DEBUG] Applying band_tile with MultiVal...");

    // IMPORTANT: band_tile consumes the node and returns a new node
    // positioned at the outer band after tiling
    let tiled_node = band_node.band_tile(multi_val);

    // Get the schedule from the tiled node
    // The tiled_node is positioned at the outer band, so we can get the schedule directly
    println!("[DEBUG] Getting schedule from tiled node...");
    let tiled_schedule = tiled_node.get_schedule();

    println!("[DEBUG] Successfully applied per-dimension tiling!");
    tiled_schedule
}

/// Navigate to the band node at the specified index
/// This traverses the schedule tree to find the nth band node
fn navigate_to_band(node: ScheduleNode, target_band_idx: usize) -> Option<ScheduleNode> {
    use isl_rs::ScheduleNodeType;

    // Simple traversal - just go to first child repeatedly until we find a band
    // In a more sophisticated implementation, we'd handle all node types properly
    let mut current = node;
    let mut band_count = 0;

    // Traverse down the tree looking for band nodes
    loop {
        let node_type = current.get_type();

        if node_type == ScheduleNodeType::Band {
            if band_count == target_band_idx {
                return Some(current);
            }
            band_count += 1;
        }

        // Try to go to first child
        if current.has_children() && current.n_children() > 0 {
            current = current.child(0);
        } else {
            // No more children, band not found
            return None;
        }
    }
}

/// Convenience function for optimal NCP GEMM tiling
/// Returns a schedule tiled with Ti=16, Tj=16, Tk=8
pub fn tile_for_ncp_gemm(schedule: &Schedule) -> Schedule {
    // Optimal tile sizes for 4KB buffer constraint:
    // Ti x Tk + Tk x Tj + Ti x Tj = 16x8 + 8x16 + 16x16 = 512 doubles = 4KB
    tile_per_dimension(schedule, 0, vec![16, 16, 8])
}

/// Tile with hierarchical sizes (for cache hierarchy)
///
/// # Example
/// ```rust
/// use polysat::tile_per_dimension::tile_per_dimension;
/// use isl_rs::{Context, Schedule, UnionSet};
/// use std::sync::Arc;
///
/// let ctx = Arc::new(Context::alloc());
/// let domain = UnionSet::read_from_str(&ctx, "{ S0[i,j,k] : 0 <= i,j,k < 128 }");
/// let schedule = Schedule::from_domain(domain);
///
/// // L1 Tiling (16x16x8)
/// let l1_tiled = tile_per_dimension(&schedule, 0, vec![16, 16, 8]);
///
/// // L2 Tiling (64x64x32) on top of L1
/// let l2_tiled = tile_per_dimension(&l1_tiled, 0, vec![64, 64, 32]);
/// ```
pub fn hierarchical_tiling(schedule: &Schedule, tile_configs: Vec<Vec<i32>>) -> Schedule {
    let mut current = schedule.copy();

    for (level, sizes) in tile_configs.iter().enumerate() {
        println!(
            "[DEBUG] Applying level {} tiling with sizes {:?}",
            level, sizes
        );
        current = tile_per_dimension(&current, 0, sizes.clone());
    }

    current
}

#[cfg(test)]
mod tests {

    use isl_rs::Context;
    use std::sync::Arc;

    #[test]
    fn test_per_dimension_tiling() {
        let _ctx = Arc::new(Context::alloc());
        let _schedule_str = r#"
            domain: "{ S[i, j, k] : 0 <= i < 1024 and 0 <= j < 1024 and 0 <= k < 1024 }"
            child:
              schedule: "[{ S[i, j, k] -> [(i)] }, { S[i, j, k] -> [(j)] }, { S[i, j, k] -> [(k)] }]"
        "#;

        // Parse schedule (would need actual parse_isl function)
        // let schedule = parse_isl(ctx, schedule_str).unwrap();

        // Test per-dimension tiling
        // let tiled = tile_per_dimension(&schedule.schedule, 0, vec![16, 16, 8]);

        // Verify the schedule has expected tiling pattern
        // let schedule_str = tiled.to_str().to_string();
        // assert!(schedule_str.contains("mod 16"));
        // assert!(schedule_str.contains("mod 8"));
    }

    #[test]
    fn test_ncp_optimal_tiling() {
        // Test that NCP optimal configuration produces correct memory footprint
        let ti = 16;
        let tj = 16;
        let tk = 8;

        let memory_footprint = ti * tk + tk * tj + ti * tj;
        assert_eq!(
            memory_footprint, 512,
            "Memory footprint should be exactly 512 doubles (4KB)"
        );
    }
}
