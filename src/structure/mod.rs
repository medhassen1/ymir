//! Structure template instancing.
//!
//! A structure — a village house, a dungeon room — is stored once as a template
//! and stamped into the world at each of its placements, with a rotation and an
//! offset. Templates may nest: a village is a template whose instances are
//! themselves templates. Each resolved template is packed into its own buffer
//! and handed to the stamper by pointer, so a placement that repeats a template
//! fifty times stamps from one copy of its cells.
//!
//! A resolved template is also a useful positioning anchor: a village's houses
//! are placed relative to the village origin, not to the world origin directly,
//! and a later placement can be measured against any template the region has
//! already resolved. [`CellArena`] is where those cells live. It is a bump
//! arena, and templates resolve depth first, so it is rewound as the recursion
//! unwinds: a nested template aggregates its children's cells into its own list
//! as they come back, which makes the space the finished subtree was staged in
//! dead the moment that template returns. Rewinding is what keeps a village of
//! fifty houses from staging fifty houses' worth of cells at once.
//!
//! Anchors themselves are held by an [`AnchorRing`] of fixed capacity, so what
//! the region's closing pass costs — and how many templates it keeps
//! addressable — is bounded by the ring rather than by how many templates the
//! region carries.

use crate::budget;
use crate::common::*;
use crate::graft;
use crate::parse::Region;
use crate::reader::Cursor;
use crate::format::NESTED_TEMPLATE;

/// A template smaller than this resolved to nothing useful and is released
/// before the stamping pass rather than being carried through it.
pub const MIN_TEMPLATE_CELLS: usize = 3;

/// One cell of a resolved template.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Cell {
    /// Block state to stamp.
    pub state: u16,
    /// Vertical offset from the instance origin.
    pub dy: i16,
}

/// Where and how a template is stamped.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Placement {
    /// World X of the instance origin.
    pub x: i16,
    /// World Y of the instance origin.
    pub y: i16,
    /// World Z of the instance origin.
    pub z: i16,
    /// Quarter turns about the vertical axis.
    pub rot: u8,
}

/// Cells one [`CellArena`] block holds before it is replaced.
///
/// An ordinary region's whole template list stages into a single block: the
/// arena is rewound as nesting unwinds, so only the templates actually live at
/// a given moment occupy it. A region carrying many large top-level templates
/// outgrows it, which is what the replacement path exists for.
const BLOCK_CELLS: usize = 512;

/// Positioning anchors the ring keeps addressable at once.
///
/// Four is enough for a placement to be measured against a handful of recently
/// resolved templates while keeping the closing pass's cost — and the cells it
/// holds on to — independent of the region's template count.
const ANCHOR_SLOTS: usize = 4;

/// A point in a [`CellArena`] to rewind back to.
#[derive(Clone, Copy)]
struct Mark {
    /// Which block the mark was taken in.
    generation: usize,
    /// Where the bump cursor stood.
    used: usize,
}

/// A bump arena for staged template cells, rewound as recursion unwinds.
///
/// Templates resolve depth first, and a nested template aggregates its
/// children's cells into its own list as they come back — so the space a
/// finished subtree was staged in is dead the moment that template returns.
/// [`CellArena::mark`] and [`CellArena::rewind`] are how that space comes back:
/// the bump cursor returns to where the nested template started, and the next
/// template stages over it.
///
/// Because nothing below the cursor outlives the frame that staged it, the
/// arena holds exactly one block: a block with no room for the next request is
/// released and replaced rather than chained. A replacement at least doubles
/// the block, so a long template list takes a logarithmic number of them
/// rather than one per template.
struct CellArena {
    /// The block being staged into.
    block: Box<[Cell]>,
    /// Cells of `block` already staged.
    used: usize,
    /// Bumped on every replacement, so a mark taken in a block that is gone
    /// can be told apart from one taken in the block now current.
    generation: usize,
}

impl CellArena {
    fn new() -> CellArena {
        CellArena {
            block: vec![Cell { state: 0, dy: 0 }; BLOCK_CELLS].into_boxed_slice(),
            used: 0,
            generation: 0,
        }
    }

    /// Where the bump cursor stands, for a later [`CellArena::rewind`].
    fn mark(&self) -> Mark {
        Mark { generation: self.generation, used: self.used }
    }

    /// Return the bump cursor to `mark`.
    ///
    /// A mark taken in a block that has since been replaced names space that
    /// no longer exists. The replacement block started empty and has only
    /// staged what came after the mark, so there is nothing to give back and
    /// the rewind does nothing.
    fn rewind(&mut self, mark: Mark) {
        if mark.generation == self.generation {
            self.used = mark.used;
        }
    }

    /// Stage `cells` into the arena and return a pointer to where they landed.
    ///
    /// The returned pointer's `cells.len()` entries are contiguous, and address
    /// live memory for as long as the block holding them is the arena's own.
    fn stage(&mut self, cells: &[Cell]) -> *const Cell {
        if self.used + cells.len() > self.block.len() {
            let cap = cells.len().max(self.block.len() * 2);
            self.block = vec![Cell { state: 0, dy: 0 }; cap].into_boxed_slice();
            self.used = 0;
            self.generation += 1;
        }
        let start = self.used;
        self.block[start..start + cells.len()].copy_from_slice(cells);
        self.used += cells.len();
        // SAFETY: the copy above wrote `cells.len()` entries starting at
        // `start`, so `start` is an index inside this allocation.
        unsafe { self.block.as_ptr().add(start) }
    }
}

/// A staged template kept as a positioning anchor for the region's closing
/// pass to fold — see [`instance_region`].
struct Anchor {
    ptr: *const Cell,
    len: usize,
}

/// A fixed-capacity ring of positioning anchors.
///
/// Registering an anchor takes the next slot in round-robin order, displacing
/// whatever was registered [`ANCHOR_SLOTS`] templates ago. Bounding the ring is
/// what keeps the pass's anchor state — and the closing fold's cost — flat over
/// a region with many templates, instead of growing one entry per template the
/// way a plain list would.
struct AnchorRing {
    slots: [Option<Anchor>; ANCHOR_SLOTS],
    next: usize,
}

impl AnchorRing {
    fn new() -> AnchorRing {
        AnchorRing { slots: Default::default(), next: 0 }
    }

    /// Register `anchor` as the newest positioning anchor.
    fn register(&mut self, anchor: Anchor) {
        self.slots[self.next] = Some(anchor);
        self.next = (self.next + 1) % ANCHOR_SLOTS;
    }

    /// The anchors the ring currently holds, in slot order.
    fn anchors(&self) -> impl Iterator<Item = &Anchor> {
        self.slots.iter().flatten()
    }
}

/// Where a template resolve stages what it produces: the arena the cells land
/// in and the ring the resulting anchors are registered with.
struct Staging {
    arena: CellArena,
    anchors: AnchorRing,
}

impl Staging {
    fn new() -> Staging {
        Staging { arena: CellArena::new(), anchors: AnchorRing::new() }
    }

    /// Stage a resolved template's cells and register them as an anchor.
    ///
    /// A template that resolved to fewer than [`MIN_TEMPLATE_CELLS`] cells
    /// contributed nothing worth positioning against, so it is neither staged
    /// nor registered.
    fn record(&mut self, cells: &[Cell]) {
        if cells.len() < MIN_TEMPLATE_CELLS {
            return;
        }
        let ptr = self.arena.stage(cells);
        self.anchors.register(Anchor { ptr, len: cells.len() });
    }
}

/// Decode one template's cells, recursing through nested template lists.
///
/// Every template that resolves — a village's houses as much as the village
/// itself — is staged into `stage` and registered there as a positioning
/// anchor, since a later placement may be measured against any of them.
fn decode_template(
    c: &mut Cursor,
    depth: u32,
    budget: &mut usize,
    stage: &mut Staging,
) -> Result<(Vec<Cell>, Placement), Status> {
    if depth > MAX_TEMPLATE_DEPTH {
        return Err(Status::DepthExceeded);
    }
    if *budget == 0 {
        return Err(Status::Truncated);
    }
    *budget -= 1;

    let template_id = c.u16();
    let x = c.i16();
    let y = c.i16();
    let z = c.i16();
    let rot = c.u8();
    let cell_count = c.u16() as usize;
    if !c.ok {
        return Err(Status::Truncated);
    }
    let placement = Placement { x, y, z, rot: rot & 3 };

    if template_id == NESTED_TEMPLATE {
        // A nested template contributes its children's cells, shifted by its
        // own origin, which is how a village aggregates its houses.
        let entry = stage.arena.mark();
        let mut cells = Vec::new();
        for _ in 0..cell_count.min(32) {
            let (child, child_at) = decode_template(c, depth + 1, budget, stage)?;
            for cell in child {
                cells.push(Cell {
                    state: cell.state,
                    dy: cell.dy.wrapping_add(child_at.y),
                });
            }
        }
        // Every child has been aggregated into `cells`, so the space the
        // subtree was staged in is dead: give it back before this template's
        // own aggregate is staged over it.
        stage.arena.rewind(entry);
        stage.record(&cells);
        return Ok((cells, placement));
    }

    let mut cells = Vec::with_capacity(cell_count.min(256));
    for _ in 0..cell_count.min(256) {
        let state = c.u16();
        let dy = c.i16();
        if !c.ok {
            return Err(Status::Truncated);
        }
        cells.push(Cell { state, dy });
    }
    stage.record(&cells);
    Ok((cells, placement))
}

/// Validate the region's structure section without stamping anything.
///
/// Called by [`crate::verify`], which checks nesting independently of the
/// columns because a template list can be malformed on its own. The staging a
/// resolve writes through is local to this call and released with it: nothing
/// validation stages outlives the check.
pub fn validate(region: &Region) -> Status {
    let data = region.slice(region.strc);
    if data.is_empty() {
        return Status::Ok;
    }
    let mut c = Cursor::new(data);
    let count = c.u16() as usize;
    if !c.ok {
        return Status::Truncated;
    }
    let mut budget = 256usize;
    let mut stage = Staging::new();
    let depth_cap = (region.depth as u32).min(MAX_TEMPLATE_DEPTH);
    for _ in 0..count.min(MAX_STACK) {
        match decode_template(&mut c, MAX_TEMPLATE_DEPTH - depth_cap, &mut budget, &mut stage) {
            Ok(_) => {}
            Err(Status::Truncated) => break,
            Err(e) => return e,
        }
    }
    Status::Ok
}

/// Instance every structure template into the region and fold a digest.
///
/// Each template is resolved in turn and stamped immediately from the cells it
/// resolved to, so an ordinary placement is folded the moment it comes back.
/// Templates that resolved to fewer than [`MIN_TEMPLATE_CELLS`] cells
/// contributed nothing and are never stamped at all. Resolving also stages each
/// template into the region's [`CellArena`] and registers it with the
/// [`AnchorRing`]; once every template has had its turn, the region's closing
/// pass folds the anchors the ring still holds, measuring a nested placement's
/// position against the template it nested under.
pub fn instance_region(region: &Region, n: usize) -> u64 {
    let data = region.slice(region.strc);
    if data.is_empty() {
        return 0;
    }
    let mut c = Cursor::new(data);
    let count = c.u16() as usize;
    if !c.ok {
        return 0;
    }

    let mut budget = 256usize;
    let depth_cap = (region.depth as u32).min(MAX_TEMPLATE_DEPTH);
    let start_depth = MAX_TEMPLATE_DEPTH - depth_cap;

    let mut stage = Staging::new();
    // One band-occupancy table shared across every placement in the region,
    // sized like the other bump-packing stores: from the region's own rebuild
    // pressure rather than a fixed guess.
    let mut marks = vec![0u8; budget::store_bytes(region, n)];
    let mut acc = 0xffu64 ^ (n as u64);

    for _ in 0..count.min(MAX_STACK) {
        let (cells, placement) = match decode_template(&mut c, start_depth, &mut budget, &mut stage)
        {
            Ok(t) => t,
            Err(_) => break,
        };
        if cells.len() < MIN_TEMPLATE_CELLS {
            // Resolved to nothing useful; never stamped.
            continue;
        }

        // Stamp from the cells this template resolved to, which this scope
        // owns for the whole call.
        acc = acc.wrapping_mul(0x100000001b3)
            ^ graft::stamp(cells.as_ptr(), cells.len(), placement, region.seed, &mut marks);
    }

    // Close out the pass by folding in every anchor the ring holds, measuring
    // a nested placement's position against the template it nested under.
    for a in stage.anchors.anchors() {
        acc = acc.wrapping_mul(0x100000001b3) ^ graft::fold_parent(a.ptr, a.len);
    }
    acc
}

#[cfg(test)]
mod tests {
    use super::*;

    fn template_bytes(id: u16, cells: &[(u16, i16)]) -> Vec<u8> {
        let mut v = Vec::new();
        v.extend_from_slice(&id.to_be_bytes());
        v.extend_from_slice(&1i16.to_be_bytes()); // x
        v.extend_from_slice(&2i16.to_be_bytes()); // y
        v.extend_from_slice(&3i16.to_be_bytes()); // z
        v.push(1); // rot
        v.extend_from_slice(&(cells.len() as u16).to_be_bytes());
        for &(state, dy) in cells {
            v.extend_from_slice(&state.to_be_bytes());
            v.extend_from_slice(&dy.to_be_bytes());
        }
        v
    }

    #[test]
    fn decodes_a_leaf_template() {
        let bytes = template_bytes(1, &[(5, 0), (6, 1)]);
        let mut c = Cursor::new(&bytes);
        let mut budget = 16usize;
        let mut stage = Staging::new();
        let (cells, at) = decode_template(&mut c, 0, &mut budget, &mut stage).unwrap();
        assert_eq!(cells, vec![Cell { state: 5, dy: 0 }, Cell { state: 6, dy: 1 }]);
        assert_eq!(at, Placement { x: 1, y: 2, z: 3, rot: 1 });
    }

    #[test]
    fn nested_template_aggregates_children() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&NESTED_TEMPLATE.to_be_bytes());
        bytes.extend_from_slice(&0i16.to_be_bytes());
        bytes.extend_from_slice(&0i16.to_be_bytes());
        bytes.extend_from_slice(&0i16.to_be_bytes());
        bytes.push(0);
        bytes.extend_from_slice(&2u16.to_be_bytes()); // two children
        bytes.extend_from_slice(&template_bytes(1, &[(7, 0)]));
        bytes.extend_from_slice(&template_bytes(2, &[(8, 0)]));

        let mut c = Cursor::new(&bytes);
        let mut budget = 16usize;
        let mut stage = Staging::new();
        let (cells, _) = decode_template(&mut c, 0, &mut budget, &mut stage).unwrap();
        // Each child's cells are shifted by that child's own y origin (2).
        assert_eq!(cells.len(), 2);
        assert_eq!(cells[0].state, 7);
        assert_eq!(cells[0].dy, 2);
    }

    #[test]
    fn depth_budget_stops_runaway_nesting() {
        let mut bytes = Vec::new();
        for _ in 0..12 {
            bytes.extend_from_slice(&NESTED_TEMPLATE.to_be_bytes());
            bytes.extend_from_slice(&0i16.to_be_bytes());
            bytes.extend_from_slice(&0i16.to_be_bytes());
            bytes.extend_from_slice(&0i16.to_be_bytes());
            bytes.push(0);
            bytes.extend_from_slice(&1u16.to_be_bytes());
        }
        let mut c = Cursor::new(&bytes);
        let mut budget = 256usize;
        let mut stage = Staging::new();
        assert_eq!(
            decode_template(&mut c, 0, &mut budget, &mut stage),
            Err(Status::DepthExceeded)
        );
    }

    #[test]
    fn work_budget_stops_wide_nesting() {
        let bytes = template_bytes(1, &[(1, 0)]);
        let mut c = Cursor::new(&bytes);
        let mut budget = 0usize;
        let mut stage = Staging::new();
        assert_eq!(
            decode_template(&mut c, 0, &mut budget, &mut stage),
            Err(Status::Truncated)
        );
    }

    #[test]
    fn truncated_template_is_reported() {
        let bytes = [0u8, 1, 0];
        let mut c = Cursor::new(&bytes);
        let mut budget = 16usize;
        let mut stage = Staging::new();
        assert_eq!(
            decode_template(&mut c, 0, &mut budget, &mut stage),
            Err(Status::Truncated)
        );
    }

    #[test]
    fn rotation_is_masked_to_quarter_turns() {
        let mut bytes = template_bytes(1, &[(1, 0)]);
        bytes[8] = 7; // rot byte
        let mut c = Cursor::new(&bytes);
        let mut budget = 16usize;
        let mut stage = Staging::new();
        let (_, at) = decode_template(&mut c, 0, &mut budget, &mut stage).unwrap();
        assert_eq!(at.rot, 3);
    }

    fn cells(n: usize) -> Vec<Cell> {
        (0..n).map(|i| Cell { state: i as u16 + 1, dy: 0 }).collect()
    }

    #[test]
    fn arena_stage_writes_are_readable_back() {
        let mut arena = CellArena::new();
        let a = arena.stage(&cells(4));
        let b = arena.stage(&cells(2));
        // SAFETY: the block has not been replaced (two tiny stages, far below
        // its size), so both pointers still address the cells just staged.
        unsafe {
            assert_eq!((*a.add(3)).state, 4);
            assert_eq!((*b.add(1)).state, 2);
        }
    }

    #[test]
    fn rewind_returns_the_cursor_to_the_mark() {
        let mut arena = CellArena::new();
        arena.stage(&cells(4));
        let mark = arena.mark();
        arena.stage(&cells(8));
        assert_eq!(arena.used, 12);
        arena.rewind(mark);
        assert_eq!(arena.used, 4, "the subtree's space must come back");
        // The next stage lands exactly where the rewound subtree started.
        let after = arena.stage(&cells(2));
        // SAFETY: the block has not been replaced, so this pointer addresses
        // the cells just staged.
        unsafe {
            assert_eq!((*after).state, 1);
        }
        assert_eq!(arena.used, 6);
    }

    #[test]
    fn rewinding_into_a_replaced_block_does_nothing() {
        let mut arena = CellArena::new();
        arena.stage(&cells(4));
        let mark = arena.mark();
        // Outgrow the block, which replaces it and resets the cursor.
        arena.stage(&cells(BLOCK_CELLS));
        assert_eq!(arena.generation, 1);
        assert_eq!(arena.used, BLOCK_CELLS);
        arena.rewind(mark);
        assert_eq!(arena.used, BLOCK_CELLS, "a mark from a released block gives nothing back");
    }

    #[test]
    fn arena_replaces_its_block_once_outgrown() {
        let mut arena = CellArena::new();
        arena.stage(&cells(BLOCK_CELLS - 2));
        assert_eq!(arena.generation, 0);
        arena.stage(&cells(5));
        assert_eq!(arena.generation, 1, "the exhausted block must be replaced");
        assert!(arena.block.len() >= BLOCK_CELLS * 2, "a replacement at least doubles");
    }

    #[test]
    fn anchor_ring_holds_only_its_newest_entries() {
        let mut ring = AnchorRing::new();
        let staged = cells(4);
        for len in 1..=(ANCHOR_SLOTS + 2) {
            ring.register(Anchor { ptr: staged.as_ptr(), len });
        }
        assert_eq!(ring.anchors().count(), ANCHOR_SLOTS);
        let mut lens: Vec<usize> = ring.anchors().map(|a| a.len).collect();
        lens.sort_unstable();
        assert_eq!(lens, vec![3, 4, 5, 6]);
    }

    #[test]
    fn resolving_a_template_registers_it_as_an_anchor() {
        let bytes = template_bytes(1, &[(5, 0), (6, 1), (7, 2), (8, 3)]);
        let mut c = Cursor::new(&bytes);
        let mut budget = 16usize;
        let mut stage = Staging::new();
        decode_template(&mut c, 0, &mut budget, &mut stage).unwrap();
        assert_eq!(stage.anchors.anchors().count(), 1);
        assert_eq!(stage.anchors.anchors().next().expect("one anchor").len, 4);
    }

    #[test]
    fn a_template_too_small_to_position_against_is_not_registered() {
        let bytes = template_bytes(1, &[(5, 0)]);
        let mut c = Cursor::new(&bytes);
        let mut budget = 16usize;
        let mut stage = Staging::new();
        decode_template(&mut c, 0, &mut budget, &mut stage).unwrap();
        assert_eq!(stage.anchors.anchors().count(), 0);
    }
}
