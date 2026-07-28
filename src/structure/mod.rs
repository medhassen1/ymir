//! Structure template instancing.
//!
//! A structure — a village house, a dungeon room — is stored once as a template
//! and stamped into the world at each of its placements, with a rotation and an
//! offset. Templates may nest: a village is a template whose instances are
//! themselves templates. Each resolved template is packed into its own buffer
//! and handed to the stamper by pointer, so a placement that repeats a template
//! fifty times stamps from one copy of its cells.
//!
//! A large enough template is also a useful positioning anchor: a village's
//! houses are placed relative to the village origin, not to the world origin
//! directly. [`CellArena`] gives the region a home for every decoded
//! template's cells that outlives any single template's own turn through the
//! instancing loop — built once per region, it appends each template's cells
//! into fixed-size chunks and hands back a pointer into them instead of an
//! owned buffer. Because the arena lives for the whole pass, its memory is
//! bounded separately from any one template's lifetime: once the accumulated
//! cell count crosses a threshold, [`CellArena::compact`] drops the oldest
//! chunks, the way a structure cache reclaims cold pages instead of growing
//! without bound on a region with many templates.

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

/// Decode one template's cells, recursing through nested template lists.
fn decode_template(
    c: &mut Cursor,
    depth: u32,
    budget: &mut usize,
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
        let mut cells = Vec::new();
        for _ in 0..cell_count.min(32) {
            let (child, child_at) = decode_template(c, depth + 1, budget)?;
            for cell in child {
                cells.push(Cell {
                    state: cell.state,
                    dy: cell.dy.wrapping_add(child_at.y),
                });
            }
        }
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
    Ok((cells, placement))
}

/// Validate the region's structure section without stamping anything.
///
/// Called by [`crate::verify`], which checks nesting independently of the
/// columns because a template list can be malformed on its own.
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
    let depth_cap = (region.depth as u32).min(MAX_TEMPLATE_DEPTH);
    for _ in 0..count.min(MAX_STACK) {
        match decode_template(&mut c, MAX_TEMPLATE_DEPTH - depth_cap, &mut budget) {
            Ok(_) => {}
            Err(Status::Truncated) => break,
            Err(e) => return e,
        }
    }
    Status::Ok
}

/// Cells held by one [`CellArena`] chunk.
///
/// An ordinary template's decoded cells fit with room to spare, so a typical
/// commit never has to look past the chunk it lands in.
const ARENA_CHUNK_CELLS: usize = 128;

/// Accumulated resident cells across an arena's live chunks that triggers
/// [`CellArena::compact`].
///
/// An ordinary region — a modest number of templates, most small houses —
/// never approaches this. A region built from many large templates does,
/// which is exactly the case the bound exists to catch: without it, a
/// region-lifetime arena would keep every template's cells resident for the
/// whole pass no matter how many templates the region carries.
const COMPACT_THRESHOLD_CELLS: usize = 1024;

/// A template at or above this many cells is large enough to be worth keeping
/// as a positioning anchor for later, nested placements.
const RETAIN_MIN_CELLS: usize = 48;

/// A region-lifetime arena for committed template cells.
///
/// Built once per region rather than once per template, so a pointer handed
/// out while decoding one template stays valid while later templates are
/// decoded — which is what lets a retained parent template (see
/// [`instance_region`]) be read again well after its own turn through the
/// loop has passed. Cells are appended into fixed-size chunks, each stored as
/// an exact-sized boxed slice; a chunk with no room left for the next commit
/// is left as-is and a fresh one takes over, so a single commit is never
/// split across two chunks.
struct CellArena {
    /// Chunks holding committed template cells, oldest first.
    chunks: Vec<Box<[Cell]>>,
    /// Cells already written into the last chunk.
    used: usize,
    /// Cells held across all currently resident chunks.
    resident: usize,
}

impl CellArena {
    fn new() -> CellArena {
        CellArena { chunks: Vec::new(), used: 0, resident: 0 }
    }

    /// Commit `cells` into the arena and return a pointer to where they
    /// landed.
    ///
    /// If what remains of the current chunk cannot hold `cells`, a fresh
    /// chunk takes over first, so the returned pointer's `cells.len()`
    /// entries are always contiguous — addressing live memory for as long as
    /// the chunk backing them stays resident (see [`CellArena::compact`]).
    fn commit(&mut self, cells: &[Cell]) -> *const Cell {
        let len = cells.len();
        let fits_current = self.chunks.last().is_some_and(|c| self.used + len <= c.len());
        if !fits_current {
            let cap = len.max(ARENA_CHUNK_CELLS);
            let filler = Cell { state: 0, dy: 0 };
            self.chunks.push(vec![filler; cap].into_boxed_slice());
            self.used = 0;
            self.resident += cap;
        }
        let chunk = self.chunks.last_mut().expect("a chunk was just ensured above");
        chunk[self.used..self.used + len].copy_from_slice(cells);
        // SAFETY: `chunk` is a live `Box<[Cell]>` at least `self.used + len`
        // entries long — either it already fit `cells` past `self.used`, or a
        // chunk sized to hold at least `cells` was just pushed — so this
        // offset and the `len` entries from it lie inside the allocation.
        let ptr = unsafe { chunk.as_ptr().add(self.used) };
        self.used += len;
        ptr
    }

    /// Drop the oldest resident chunks until the arena's accumulated cells
    /// fall back to `threshold`, or only the chunk currently being written to
    /// is left.
    ///
    /// This is the arena's memory bound: left unchecked, a region-lifetime
    /// arena would keep every template's cells resident for the whole pass no
    /// matter how many templates the region carries. The chunk currently
    /// being written to is never dropped, since the next commit needs
    /// somewhere to land.
    fn compact(&mut self, threshold: usize) {
        while self.resident > threshold && self.chunks.len() > 1 {
            let oldest = self.chunks.remove(0);
            self.resident -= oldest.len();
        }
    }
}

/// A template retained past its own turn through the instancing loop, for the
/// region's closing pass to read as a nested placement's positioning anchor.
struct RetainedParent {
    ptr: *const Cell,
    len: usize,
}

/// Instance every structure template into the region and fold a digest.
///
/// Each template's cells are committed into a region-wide [`CellArena`] as
/// they are decoded and stamped immediately, so an ordinary placement is
/// folded the moment it is resolved. A template large enough to anchor nested
/// placements against is kept as a retained parent. Templates that resolved to
/// fewer than [`MIN_TEMPLATE_CELLS`] cells contributed nothing and are never
/// committed at all. Once every template has had its turn, the region's
/// closing pass folds every retained parent once more, measuring a nested
/// placement's position against the template it nested under.
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

    let mut arena = CellArena::new();
    let mut retained: Vec<RetainedParent> = Vec::new();
    // One band-occupancy table shared across every placement in the region,
    // sized like the other bump-packing stores: from the region's own rebuild
    // pressure rather than a fixed guess.
    let mut marks = vec![0u8; budget::store_bytes(region, n)];
    let mut acc = 0xffu64 ^ (n as u64);

    for _ in 0..count.min(MAX_STACK) {
        let (cells, placement) = match decode_template(&mut c, start_depth, &mut budget) {
            Ok(t) => t,
            Err(_) => break,
        };
        if cells.len() < MIN_TEMPLATE_CELLS {
            // Resolved to nothing useful; never committed or stamped.
            continue;
        }

        // Commit this template's cells into the region's arena. Only past
        // this point is there a pointer stable enough to retain past this
        // template's own scope.
        let ptr = arena.commit(&cells);

        // Stamp the template immediately while its span is still fresh off
        // the commit, so an ordinary placement is folded the moment it is
        // resolved rather than carried past the point its storage might be
        // reclaimed.
        acc = acc.wrapping_mul(0x100000001b3)
            ^ graft::stamp(ptr, cells.len(), placement, region.seed, &mut marks);

        // A template large enough to anchor nested placements against is kept
        // as a retained parent for the region's closing pass to read.
        if cells.len() >= RETAIN_MIN_CELLS {
            retained.push(RetainedParent { ptr, len: cells.len() });
        }

        // Bound the arena's resident memory now that this template's cells
        // are safely committed.
        arena.compact(COMPACT_THRESHOLD_CELLS);
    }

    // Close out the pass by folding in every retained parent template once,
    // measuring a nested placement's position against the template it
    // nested under.
    for r in &retained {
        acc = acc.wrapping_mul(0x100000001b3) ^ graft::fold_parent(r.ptr, r.len);
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
        let (cells, at) = decode_template(&mut c, 0, &mut budget).unwrap();
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
        let (cells, _) = decode_template(&mut c, 0, &mut budget).unwrap();
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
        assert_eq!(decode_template(&mut c, 0, &mut budget), Err(Status::DepthExceeded));
    }

    #[test]
    fn work_budget_stops_wide_nesting() {
        let bytes = template_bytes(1, &[(1, 0)]);
        let mut c = Cursor::new(&bytes);
        let mut budget = 0usize;
        assert_eq!(decode_template(&mut c, 0, &mut budget), Err(Status::Truncated));
    }

    #[test]
    fn truncated_template_is_reported() {
        let bytes = [0u8, 1, 0];
        let mut c = Cursor::new(&bytes);
        let mut budget = 16usize;
        assert_eq!(decode_template(&mut c, 0, &mut budget), Err(Status::Truncated));
    }

    #[test]
    fn rotation_is_masked_to_quarter_turns() {
        let mut bytes = template_bytes(1, &[(1, 0)]);
        bytes[8] = 7; // rot byte
        let mut c = Cursor::new(&bytes);
        let mut budget = 16usize;
        let (_, at) = decode_template(&mut c, 0, &mut budget).unwrap();
        assert_eq!(at.rot, 3);
    }

    fn cells(n: usize) -> Vec<Cell> {
        (0..n).map(|i| Cell { state: i as u16 + 1, dy: 0 }).collect()
    }

    #[test]
    fn cell_arena_commit_writes_are_readable_back() {
        let mut arena = CellArena::new();
        let a = arena.commit(&cells(4));
        let b = arena.commit(&cells(2));
        // SAFETY: neither chunk has been compacted away, so both pointers
        // still address the cells just committed.
        unsafe {
            assert_eq!((*a.add(3)).state, 4);
            assert_eq!((*b.add(1)).state, 2);
        }
    }

    #[test]
    fn cell_arena_starts_a_new_chunk_once_the_current_one_is_full() {
        let mut arena = CellArena::new();
        arena.commit(&cells(ARENA_CHUNK_CELLS - 2));
        assert_eq!(arena.chunks.len(), 1);
        arena.commit(&cells(5));
        assert_eq!(arena.chunks.len(), 2);
    }

    #[test]
    fn cell_arena_compact_drops_oldest_chunks_once_over_threshold() {
        let mut arena = CellArena::new();
        for _ in 0..6 {
            arena.commit(&cells(ARENA_CHUNK_CELLS));
        }
        assert_eq!(arena.chunks.len(), 6);
        arena.compact(3 * ARENA_CHUNK_CELLS);
        assert!(arena.chunks.len() < 6, "compact must drop some chunks");
        assert!(arena.resident <= 3 * ARENA_CHUNK_CELLS);
    }

    #[test]
    fn cell_arena_compact_never_drops_the_last_chunk() {
        let mut arena = CellArena::new();
        arena.commit(&cells(4));
        arena.compact(0);
        assert_eq!(arena.chunks.len(), 1, "the chunk being written to must survive");
    }
}
