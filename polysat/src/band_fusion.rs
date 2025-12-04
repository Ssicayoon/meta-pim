// Band fusion module for creating multi-dimensional bands
// Essential for generating 2D parallel structures in MLIR

use isl_rs::{MultiVal, Schedule, ScheduleNode, ScheduleNodeType, Val, ValList};

/// Fuse two consecutive bands into a single multi-dimensional band
/// This is the key to getting proper 2D scf.parallel generation
pub fn fuse_consecutive_bands(schedule: &Schedule) -> Result<Schedule, String> {
    let root = schedule.get_root();

    // Find first band node
    let first_band = find_first_band(&root)?;

    // Check if there's a second band immediately following
    // We need to use a copy since first_child() consumes the node
    let first_band_copy = first_band.copy();
    let has_second_band = if first_band_copy.has_children() {
        let child = first_band_copy.first_child();

        // Handle potential intermediate nodes (marks, etc)
        if child.get_type() == ScheduleNodeType::Band {
            true
        } else if child.has_children() {
            let grandchild = child.first_child();
            grandchild.get_type() == ScheduleNodeType::Band
        } else {
            false
        }
    } else {
        false
    };

    if !has_second_band {
        return Err("No second band found for fusion".to_string());
    }

    // Now we have two bands - attempt fusion
    // ISL's band_sink can help push bands down and potentially merge them
    let sunk = first_band.band_sink();

    // Check if the sink operation created a multi-dimensional band
    if sunk.get_type() == ScheduleNodeType::Band {
        let n_members = sunk.band_n_member();
        if n_members >= 2 {
            return Ok(sunk.get_schedule());
        }
    }

    Err("Could not fuse bands".to_string())
}

/// Alternative: Create a single 2D band from the start
/// This directly constructs the desired schedule structure
pub fn create_2d_band_schedule(
    schedule: &Schedule,
    tile_i: i32,
    tile_j: i32,
) -> Result<Schedule, String> {
    let ctx = schedule.get_ctx();
    let root = schedule.get_root();

    // Get the domain
    let domain = root.domain_get_domain();

    // Create a 2D partial schedule that tiles both dimensions at once
    // This is the key insight: instead of nested tiles, we create a single 2D tile

    // Build schedule string that creates 2D band directly
    // Note: ISL requires proper multi-dimensional schedule syntax
    // We need to create a proper 2-dimensional band
    let schedule_str = format!(
        r#"{{ 
            domain: "{}", 
            child: {{ 
                schedule: "[{{ S0[i0, i1] -> [(i0 - (i0) mod {})] }}, {{ S0[i0, i1] -> [(i1 - (i1) mod {})] }}]",
                permutable: 1,
                child: {{
                    schedule: "[{{ S0[i0, i1] -> [((i0) mod {})] }}, {{ S0[i0, i1] -> [((i1) mod {})] }}]"
                }}
            }}
        }}"#,
        domain.to_str(),
        tile_i,
        tile_j,
        tile_i,
        tile_j
    );

    // Parse the new schedule
    // Note: ISL-rs read_from_str doesn't return Result, it panics on error
    // We'll wrap it in a catch_unwind if needed, but for now assume it succeeds
    Ok(Schedule::read_from_str(&ctx, &schedule_str))
}

/// Helper to find first band node
fn find_first_band(node: &ScheduleNode) -> Result<ScheduleNode, String> {
    let mut current = node.copy();

    while current.get_type() != ScheduleNodeType::Band && current.has_children() {
        current = current.first_child();
    }

    if current.get_type() == ScheduleNodeType::Band {
        Ok(current)
    } else {
        Err("No band found".to_string())
    }
}

/// Apply band member set tile to create proper 2D tiling
/// This uses the proper ISL API for multi-dimensional tiling
pub fn apply_2d_band_tile(schedule: &Schedule, tile_sizes: &[i32]) -> Result<Schedule, String> {
    let root = schedule.get_root();
    let band = find_first_band(&root)?;

    // Check if band has multiple members
    let n_members = band.band_n_member() as usize;
    if n_members < 2 {
        return Err("Band doesn't have multiple members for 2D tiling".to_string());
    }

    // Create MultiVal for tile sizes
    let ctx = schedule.get_ctx();
    let space = band.band_get_space();
    let mut val_list = ValList::alloc(&ctx, tile_sizes.len() as i32);

    for &size in tile_sizes.iter() {
        let val = Val::int_from_si(&ctx, size as i64);
        val_list = val_list.add(val);
    }

    let multi_val = MultiVal::from_val_list(space, val_list);

    // Apply band_tile with multi-dimensional sizes
    let tiled = band.band_tile(multi_val);

    Ok(tiled.get_schedule())
}
