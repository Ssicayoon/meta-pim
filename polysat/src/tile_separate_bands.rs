// Tile separate 1D bands with different sizes
// This handles the case where Polygeist generates band[i0], band[i1], band[i2] separately

use isl_rs::*;
use std::vec::Vec;

/// Tiles multiple separate 1D bands with different tile sizes
///
/// This function handles the common Polygeist pattern where each loop dimension
/// is represented as a separate 1D band node in the schedule tree.
///
/// # Arguments
/// * `schedule` - The schedule containing separate 1D bands
/// * `tile_sizes` - Vector of tile sizes, one per band
///
/// # Returns
/// A schedule with each band tiled by its corresponding size
pub fn tile_separate_bands(schedule: &Schedule, tile_sizes: Vec<i32>) -> Schedule {
    println!(
        "[DEBUG] tile_separate_bands called with sizes: {:?}",
        tile_sizes
    );

    if tile_sizes.is_empty() {
        println!("[ERROR] No tile sizes provided");
        return schedule.copy();
    }

    let root = schedule.get_root();
    let ctx = schedule.get_ctx();

    // Navigate through the tree and tile each band
    let mut current_node = root;
    let mut band_count = 0;

    // Find and tile each band in sequence
    current_node = tile_bands_recursively(current_node, &tile_sizes, &mut band_count, &ctx);

    // Get the final schedule
    let result = current_node.get_schedule();

    // Print result
    let result_str = crate::isl_block_printer::schedule_to_block_str(&result);
    println!("[DEBUG] Schedule after tiling separate bands:");
    println!("{}", result_str);

    result
}

fn tile_bands_recursively(
    node: ScheduleNode,
    tile_sizes: &[i32],
    band_idx: &mut usize,
    ctx: &Context,
) -> ScheduleNode {
    let node_type = node.get_type();

    match node_type {
        isl_rs::ScheduleNodeType::Band => {
            // Found a band node - tile it with the appropriate size
            if *band_idx < tile_sizes.len() {
                let tile_size = tile_sizes[*band_idx];
                println!("[DEBUG] Tiling band {} with size {}", band_idx, tile_size);

                // Get the band's space to create the MultiVal
                let band_space = node.band_get_space();
                let n_members = node.band_n_member() as usize;

                // Create a ValList with the tile size
                let mut val_list = ValList::alloc(ctx, n_members as i32);
                for _ in 0..n_members {
                    let val = Val::int_from_si(ctx, tile_size as i64);
                    val_list = val_list.add(val);
                }

                // Create MultiVal from the ValList
                let multi_val = MultiVal::from_val_list(band_space, val_list);

                // Apply tiling
                let tiled_node = node.band_tile(multi_val);

                *band_idx += 1;

                // Continue recursively to handle any child bands
                // For simplicity, we'll just process the first child if it exists
                // This matches the typical Polygeist structure
                if tiled_node.has_children() && tiled_node.n_children() > 0 {
                    let child = tiled_node.child(0);
                    let processed_child = tile_bands_recursively(child, tile_sizes, band_idx, ctx);
                    // Return the processed tree starting from this node
                    // Note: We can't replace children directly, so we return the processed child's parent
                    return processed_child.parent();
                }

                tiled_node
            } else {
                println!(
                    "[WARNING] More bands than tile sizes, skipping band {}",
                    band_idx
                );
                node
            }
        }
        _ => {
            // Not a band node, recurse to children
            if node.has_children() {
                let n_children = node.n_children();

                // Process first child (main path for Polygeist schedules)
                if n_children > 0 {
                    let child = node.child(0);
                    return tile_bands_recursively(child, tile_sizes, band_idx, ctx);
                }
            }
            node
        }
    }
}

/// Convenience function for GEMM-style kernels with 3 separate bands
pub fn tile_gemm_separate_bands(schedule: &Schedule, ti: i32, tj: i32, tk: i32) -> Schedule {
    println!(
        "[DEBUG] tile_gemm_separate_bands: Ti={}, Tj={}, Tk={}",
        ti, tj, tk
    );
    tile_separate_bands(schedule, vec![ti, tj, tk])
}
