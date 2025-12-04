// Schedule tree bindings for ISL
//
//! Tree-level ISL bindings for loop transformation operations.
//!
//! # Purpose
//!
//! This module provides **low-level** bindings to ISL's `isl_schedule_tree` API,
//! specifically to enable band partial schedule modifications that are
//! **not available** at the `schedule_node` level.
//!
//! **⚠️ Most users should NOT use this module directly.** Instead, use the
//! high-level API: [`ScheduleNode::band_set_partial_schedule()`](../schedule_node/struct.ScheduleNode.html#method.band_set_partial_schedule)
//!
//! # Background: Why This Module Exists
//!
//! ## The ISL API Design Problem
//!
//! ISL provides two levels of schedule manipulation APIs:
//!
//! 1. **Node-level API** (`isl_schedule_node_*`) - User-friendly, works with positions in schedule
//!    - ✅ Has: `band_get_partial_schedule`, `band_member_set_coincident`, etc.
//!    - ❌ Missing: `band_set_partial_schedule` (the critical transformation API!)
//!
//! 2. **Tree-level API** (`isl_schedule_tree_*`) - Lower-level, works with detached trees
//!    - ✅ Has: `band_set_partial_schedule`, `band_get_partial_schedule`, etc.
//!    - ⚠️ Problem: Requires manual extraction and grafting to use with schedules
//!
//! **This asymmetry** means loop transformations (interchange, skew, permutation) cannot
//! be implemented using only the public node-level API. We must bridge between levels.
//!
//! ## The Solution: Node ↔ Tree Bridge
//!
//! To implement `ScheduleNode::band_set_partial_schedule()`, we must:
//!
//! ```text
//! +---------------------------------------------------------------+
//! | User calls: node.band_set_partial_schedule(new_schedule)     |
//! +---------------------------------------------------------------+
//!                               |
//!                               v
//! +---------------------------------------------------------------+
//! | 1. Extract tree from node                                    |
//! |    tree = isl_schedule_node_get_tree(node)    [PRIVATE API]  |
//! +---------------------------------------------------------------+
//!                               |
//!                               v
//! +---------------------------------------------------------------+
//! | 2. Modify at tree level                                      |
//! |    new_tree = isl_schedule_tree_band_set_partial_schedule(   |
//! |        tree, new_schedule)                    [PUBLIC API]   |
//! +---------------------------------------------------------------+
//!                               |
//!                               v
//! +---------------------------------------------------------------+
//! | 3. Graft modified tree back                                  |
//! |    new_node = isl_schedule_node_graft_tree(   [PRIVATE API]  |
//! |        node, new_tree)                                       |
//! +---------------------------------------------------------------+
//!                               |
//!                               v
//! +---------------------------------------------------------------+
//! | Return new node with transformed schedule                    |
//! +---------------------------------------------------------------+
//! ```
//!
//! # Private API Usage
//!
//! This module uses two **private ISL APIs** (not in public headers but exported):
//!
//! - `isl_schedule_node_get_tree` - Extract tree from node
//! - `isl_schedule_node_graft_tree` - Graft tree back to node
//!
//! **Why this is acceptable**:
//!
//! 1. **Functions exist and are exported** - Compiled into ISL library
//! 2. **Stable API** - Used by mature tools (PPCG, Pluto) for years
//! 3. **No alternative** - Only way to implement `band_set_partial_schedule`
//! 4. **Standard practice** - Polyhedral compilers universally use these
//! 5. **Safe** - Properly wrapped with Rust's ownership semantics
//!
//! **Alternative considered and rejected**:
//! - ❌ Delete node + insert new schedule - **Loses children and metadata**
//! - ❌ Wait for ISL to add node-level API - **Unlikely to happen (10+ years)**
//! - ❌ Fork ISL to add API - **Unmaintainable**
//!
//! # Memory Management
//!
//! This module follows ISL's reference counting semantics:
//!
//! ## Ownership Transfer Pattern
//!
//! ```ignore
//! // ISL functions typically use one of three ownership patterns:
//!
//! // 1. __isl_take: Function takes ownership (caller loses access)
//! tree2 = isl_schedule_tree_band_set_partial_schedule(tree1, schedule);
//! // ↑ tree1 and schedule are consumed, tree2 is owned by caller
//!
//! // 2. __isl_give: Function returns new object (caller owns)
//! tree = isl_schedule_node_get_tree(node);
//! // ↑ node unchanged, tree owned by caller
//!
//! // 3. __isl_keep: Function borrows (no ownership change)
//! int n = isl_schedule_tree_band_n_member(tree);
//! // ↑ tree unchanged, still owned by caller
//! ```
//!
//! ## Rust Wrapper Implementation
//!
//! We use the `should_free_on_drop` flag to manage ownership:
//!
//! ```ignore
//! pub struct ScheduleTree {
//!     pub ptr: uintptr_t,
//!     pub should_free_on_drop: bool,  // ← Ownership flag
//! }
//!
//! // When transferring to ISL:
//! tree.do_not_free_on_drop();  // Transfer ownership - we don't free
//! let ptr = tree.ptr;
//! let result = unsafe { isl_func(ptr) };  // ISL now owns ptr
//!
//! // When receiving from ISL:
//! let ptr = unsafe { isl_func(...) };
//! ScheduleTree::from_ptr(ptr)  // We own ptr, will free on drop
//! ```
//!
//! # API Overview
//!
//! ## Core Transformation APIs
//!
//! - [`band_set_partial_schedule()`](#method.band_set_partial_schedule) - Modify loop schedule (interchange, etc.)
//! - [`band_get_partial_schedule()`](#method.band_get_partial_schedule) - Read current schedule
//!
//! ## Coincident Flag APIs (Parallelism Metadata)
//!
//! - [`band_member_get_coincident()`](#method.band_member_get_coincident) - Check if dimension is parallel
//! - [`band_member_set_coincident()`](#method.band_member_set_coincident) - Mark dimension as parallel/sequential
//! - [`band_n_member()`](#method.band_n_member) - Get number of loop dimensions
//!
//! ## Internal APIs
//!
//! - [`from_ptr()`](#method.from_ptr) - Create from raw pointer (with ownership)
//! - [`from_ptr_unowned()`](#method.from_ptr_unowned) - Create from raw pointer (without ownership)
//! - [`do_not_free_on_drop()`](#method.do_not_free_on_drop) - Transfer ownership to ISL
//!
//! # Usage Patterns
//!
//! ## ❌ Don't Use Directly
//!
//! ```ignore
//! // BAD: Manual tree manipulation
//! let tree = node.get_tree();
//! let partial = tree.band_get_partial_schedule();
//! let new_partial = /* modify */;
//! let new_tree = tree.band_set_partial_schedule(new_partial);
//! let new_node = node.graft_tree(new_tree);
//! ```
//!
//! ## ✅ Use High-Level API Instead
//!
//! ```ignore
//! // GOOD: Use ScheduleNode wrapper
//! let partial = node.band_get_partial_schedule();
//! let new_partial = /* modify */;
//! let new_node = node.band_set_partial_schedule(new_partial);
//! ```
//!
//! ## When Tree-Level API is Appropriate
//!
//! Only use `ScheduleTree` directly when implementing **new transformation types**:
//!
//! ```ignore
//! // Example: Custom band transformation not in ScheduleNode API
//! impl ScheduleNode {
//!     pub fn custom_complex_transformation(self) -> ScheduleNode {
//!         let tree = self.get_tree();
//!
//!         // Multiple tree-level operations
//!         let tree = tree.some_tree_operation();
//!         let tree = tree.another_tree_operation();
//!         let tree = tree.band_set_partial_schedule(schedule);
//!
//!         self.graft_tree(tree)
//!     }
//! }
//! ```
//!
//! # See Also
//!
//! - **Main API**: [`ScheduleNode`](../schedule_node/struct.ScheduleNode.html) - User-facing schedule manipulation
//! - **High-level transformation**: [`band_set_partial_schedule()`](../schedule_node/struct.ScheduleNode.html#method.band_set_partial_schedule)
//! - **Tests**: `tests/test_band_transformation.rs` - Integration tests demonstrating usage
//!
//! # Examples
//!
//! See [`ScheduleNode::band_set_partial_schedule()`](../schedule_node/struct.ScheduleNode.html#method.band_set_partial_schedule)
//! for comprehensive examples of loop interchange, permutation, and other transformations.

use super::MultiUnionPwAff;
use libc::uintptr_t;

/// Wraps `isl_schedule_tree`.
///
/// Represents a node in ISL's schedule tree structure.
/// Schedule trees are the internal representation used by ISL to store schedules.
///
/// # Memory Management
/// - Follows ISL's reference counting semantics
/// - Automatically freed when dropped (if should_free_on_drop is true)
pub struct ScheduleTree {
    pub ptr: uintptr_t,
    pub should_free_on_drop: bool,
}

extern "C" {
    /// Frees an isl_schedule_tree
    fn isl_schedule_tree_free(tree: uintptr_t);

    /// Copies an isl_schedule_tree (increments reference count)
    fn isl_schedule_tree_copy(tree: uintptr_t) -> uintptr_t;

    /// Gets the partial schedule from a band tree
    /// Returns: __isl_give isl_multi_union_pw_aff *
    fn isl_schedule_tree_band_get_partial_schedule(tree: uintptr_t) -> uintptr_t;

    /// Sets the partial schedule of a band tree
    /// Takes ownership of both tree and schedule
    /// Returns: __isl_give isl_schedule_tree *
    fn isl_schedule_tree_band_set_partial_schedule(
        tree: uintptr_t, schedule: uintptr_t,
    ) -> uintptr_t;

    /// Gets coincident flag for a band member
    /// Returns: isl_bool (0=false, 1=true, -1=error)
    fn isl_schedule_tree_band_member_get_coincident(tree: uintptr_t, pos: i32) -> i32;

    /// Sets coincident flag for a band member
    /// Takes ownership of tree
    /// Returns: __isl_give isl_schedule_tree *
    fn isl_schedule_tree_band_member_set_coincident(
        tree: uintptr_t, pos: i32, coincident: i32,
    ) -> uintptr_t;

    /// Gets the number of members in a band tree
    /// Returns: isl_size (>= 0 on success, -1 on error)
    fn isl_schedule_tree_band_n_member(tree: uintptr_t) -> i32;
}

impl ScheduleTree {
    /// Creates a ScheduleTree from a raw pointer with ownership
    pub(crate) fn from_ptr(ptr: uintptr_t) -> Self {
        ScheduleTree {
            ptr,
            should_free_on_drop: true,
        }
    }

    /// Creates a ScheduleTree from a raw pointer without taking ownership
    pub(crate) fn from_ptr_unowned(ptr: uintptr_t) -> Self {
        ScheduleTree {
            ptr,
            should_free_on_drop: false,
        }
    }

    /// Prevents this tree from being freed when dropped
    /// Used when transferring ownership to ISL
    pub(crate) fn do_not_free_on_drop(&mut self) {
        self.should_free_on_drop = false;
    }

    /// Wraps `isl_schedule_tree_band_set_partial_schedule`.
    ///
    /// Sets the partial schedule of a band tree to the given MultiUnionPwAff.
    /// This is the core operation for implementing loop transformations.
    ///
    /// # Memory Management
    /// - Takes ownership of both `self` (the tree) and `schedule`
    /// - ISL takes ownership of both pointers via FFI
    /// - Returns a new ScheduleTree owned by Rust
    ///
    /// # Example
    /// ```ignore
    /// // Get tree from node, modify partial schedule, graft back
    /// let tree = node.get_tree();
    /// let partial = tree.band_get_partial_schedule();
    /// // ... modify partial schedule (interchange dimensions, etc.) ...
    /// let new_tree = tree.band_set_partial_schedule(new_partial);
    /// let new_node = node.graft_tree(new_tree);
    /// ```
    pub fn band_set_partial_schedule(self, schedule: MultiUnionPwAff) -> ScheduleTree {
        let mut tree = self;
        tree.do_not_free_on_drop(); // Transfer ownership to ISL
        let tree_ptr = tree.ptr;

        let mut schedule = schedule;
        schedule.do_not_free_on_drop(); // Transfer ownership to ISL
        let schedule_ptr = schedule.ptr;

        let result_ptr =
            unsafe { isl_schedule_tree_band_set_partial_schedule(tree_ptr, schedule_ptr) };

        ScheduleTree::from_ptr(result_ptr)
    }

    /// Wraps `isl_schedule_tree_band_get_partial_schedule`.
    ///
    /// Gets the partial schedule of a band tree.
    pub fn band_get_partial_schedule(&self) -> MultiUnionPwAff {
        let result_ptr = unsafe { isl_schedule_tree_band_get_partial_schedule(self.ptr) };

        MultiUnionPwAff {
            ptr: result_ptr,
            should_free_on_drop: true,
        }
    }

    /// Wraps `isl_schedule_tree_band_n_member`.
    ///
    /// Returns the number of dimensions in the band.
    pub fn band_n_member(&self) -> i32 {
        unsafe { isl_schedule_tree_band_n_member(self.ptr) }
    }

    /// Wraps `isl_schedule_tree_band_member_get_coincident`.
    ///
    /// Gets the coincident flag for a specific band dimension.
    pub fn band_member_get_coincident(&self, pos: i32) -> bool {
        let result = unsafe { isl_schedule_tree_band_member_get_coincident(self.ptr, pos) };
        result == 1
    }

    /// Wraps `isl_schedule_tree_band_member_set_coincident`.
    ///
    /// Sets the coincident flag for a specific band dimension.
    /// Takes ownership of self and returns a new tree.
    pub fn band_member_set_coincident(self, pos: i32, coincident: i32) -> ScheduleTree {
        let mut tree = self;
        tree.do_not_free_on_drop(); // Transfer ownership to ISL
        let tree_ptr = tree.ptr;

        let result_ptr =
            unsafe { isl_schedule_tree_band_member_set_coincident(tree_ptr, pos, coincident) };

        ScheduleTree::from_ptr(result_ptr)
    }
}

impl Drop for ScheduleTree {
    fn drop(&mut self) {
        if self.should_free_on_drop {
            unsafe {
                isl_schedule_tree_free(self.ptr);
            }
        }
    }
}

impl Clone for ScheduleTree {
    fn clone(&self) -> Self {
        let new_ptr = unsafe { isl_schedule_tree_copy(self.ptr) };
        ScheduleTree::from_ptr(new_ptr)
    }
}
