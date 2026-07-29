//! Template stamping.
//!
//! [`crate::structure`] resolves and stages templates; this module stamps them.
//! A staged template arrives as a `(pointer, length)` view over its packed cells
//! plus the placement to apply, so a template placed many times is read from one
//! buffer instead of being cloned per placement.

use crate::structure::{Cell, Placement};

/// Vertical bands a column's stamped cells are bucketed into.
///
/// A structure's cells can sit at any absolute Y, but only the low byte of
/// that position selects a band here: a village's foundations overlap each
/// other far more often than they repeat exactly 256 blocks apart, so folding
/// a band's contribution once the first time it's touched is what keeps a
/// dozen overlapping placements from folding the same footprint over and over.
pub const BANDS: usize = 256;

/// Stamp one staged template and fold a digest of the cells it writes.
///
/// `marks` is a per-column band-occupancy table: a non-zero entry means an
/// earlier placement in this column already stamped that band, so this call
/// folds a given band into the digest at most once no matter how many cells
/// (from this template or an earlier one) land in it. `marks` may hold fewer
/// than [`BANDS`] elements — the region sizes it from its own rebuild
/// pressure — so a slot is only ever touched once it has been checked against
/// the table actually passed in, not the nominal band count.
///
/// SAFETY: `cells`/`len` must name a live packed template for the call.
pub fn stamp(cells: *const Cell, len: usize, at: Placement, seed: u32, marks: &mut [u8]) -> u64 {
    if cells.is_null() || len == 0 {
        return 0;
    }
    // SAFETY: the caller keeps this template's committed buffer alive for the
    // duration of the call, so this view names live cells.
    let view = unsafe { std::slice::from_raw_parts(cells, len) };

    let mut acc = placement_seed(at, seed);
    for (i, cell) in view.iter().enumerate() {
        let y = cell.dy.wrapping_add(at.y);
        acc = acc.rotate_left(7) ^ (cell.state as u64);
        acc = acc.wrapping_add((y as i64 as u64) ^ (i as u64));

        // The cell's placement-relative slot: where in the column's vertical
        // band table this cell's absolute position falls. Bands wrap at 256
        // by design (see `BANDS`), so only the low byte of the position
        // matters here.
        let raw = at.y as i32 + cell.dy as i32;
        let slot = raw as u8 as usize;
        if slot < marks.len() {
            // SAFETY: the guard above checks `slot` against `marks.len()`
            // directly, the real bound of the slice handed to this call.
            let mark = unsafe { marks.get_unchecked_mut(slot) };
            if *mark == 0 {
                *mark = 1;
                acc ^= (slot as u64).wrapping_mul(0x9e3779b1);
            }
        }
    }
    acc
}

/// Stamp every staged template in order, sharing one band table across the
/// whole column so overlap folds only once regardless of which placement
/// touches a band first.
pub fn stamp_all(
    staged: &[(*const Cell, usize, Placement)],
    seed: u32,
    chunks: usize,
    marks: &mut [u8],
) -> u64 {
    let mut acc = 0xffu64 ^ (chunks as u64);
    for &(cells, len, at) in staged {
        acc = acc.wrapping_mul(0x100000001b3) ^ stamp(cells, len, at, seed, marks);
    }
    acc
}

/// Fold a digest of a positioning anchor's cells.
///
/// [`crate::structure::instance_region`] stages every template's resolved
/// cells into a region-lifetime arena and registers them as a positioning
/// anchor, so the region's closing pass can fold them again once every
/// template has had its own turn — the way a village's houses are placed
/// relative to the village origin. This is that pass's read: `cells`/`len`
/// name the staged span directly, walked through raw pointer arithmetic rather
/// than a bounds-checked index.
///
/// SAFETY: `cells` must address at least `len` live [`Cell`] values for the
/// call.
pub fn fold_parent(cells: *const Cell, len: usize) -> u64 {
    if cells.is_null() || len == 0 {
        return 0;
    }
    let mut acc = 0x2545f491u64;
    // SAFETY: per this function's contract, the caller guarantees `cells`
    // addresses at least `len` live cells.
    unsafe {
        for i in 0..len {
            let c = *cells.add(i);
            acc = acc.rotate_left(11) ^ (c.state as u64) ^ ((c.dy as i64 as u64) << 16);
        }
    }
    acc
}

/// Fold a placement against the structure it is positioned relative to.
///
/// A region's structures are not placed independently: a well is dug beside the
/// village it belongs to, not at an absolute coordinate, so a placement carries
/// meaning only against the structure the region led with. `lead`/`len` name
/// that structure's staged cells and `at` is where this placement landed.
///
/// SAFETY: `lead` must address at least `len` live [`Cell`] values for the
/// call.
pub fn fold_relative(lead: *const Cell, len: usize, at: Placement) -> u64 {
    if lead.is_null() || len == 0 {
        return 0;
    }
    let mut acc = placement_seed(at, len as u32);
    // SAFETY: per this function's contract, the caller guarantees `lead`
    // addresses at least `len` live cells.
    unsafe {
        for i in 0..len {
            let c = *lead.add(i);
            let offset = c.dy.wrapping_sub(at.y);
            acc = acc.rotate_left(13) ^ (c.state as u64) ^ ((offset as i64 as u64) << 8);
        }
    }
    acc
}

/// The digest seed for a placement: its origin, rotation and the world seed.
fn placement_seed(at: Placement, seed: u32) -> u64 {
    let mut h = seed as u64 ^ 0x9e3779b97f4a7c15;
    h = h.wrapping_mul(0x100000001b3) ^ (at.x as i64 as u64);
    h = h.wrapping_mul(0x100000001b3) ^ (at.z as i64 as u64);
    h = h.rotate_left(at.rot as u32 * 8 + 1);
    h
}

#[cfg(test)]
mod relative_tests {
    use super::*;

    fn at(y: i16) -> Placement {
        Placement { x: 3, y, z: 5, rot: 1 }
    }

    #[test]
    fn fold_relative_reflects_the_lead_and_the_placement() {
        let lead = [Cell { state: 7, dy: 2 }, Cell { state: 9, dy: -1 }];
        let base = fold_relative(lead.as_ptr(), lead.len(), at(0));
        assert_ne!(base, fold_relative(lead.as_ptr(), lead.len(), at(4)));
        let other = [Cell { state: 7, dy: 2 }, Cell { state: 9, dy: -2 }];
        assert_ne!(base, fold_relative(other.as_ptr(), other.len(), at(0)));
    }

    #[test]
    fn fold_relative_of_nothing_is_zero() {
        let lead = [Cell { state: 1, dy: 0 }];
        assert_eq!(fold_relative(std::ptr::null(), 1, at(0)), 0);
        assert_eq!(fold_relative(lead.as_ptr(), 0, at(0)), 0);
    }
}

/// Rotate a template-local offset by a placement's quarter turns.
pub fn rotate_offset(dx: i16, dz: i16, rot: u8) -> (i16, i16) {
    match rot & 3 {
        0 => (dx, dz),
        1 => (-dz, dx),
        2 => (-dx, -dz),
        _ => (dz, -dx),
    }
}

/// The vertical span a staged template covers, as `(min_dy, max_dy)`.
pub fn vertical_span(cells: &[Cell]) -> (i16, i16) {
    if cells.is_empty() {
        return (0, 0);
    }
    let mut lo = i16::MAX;
    let mut hi = i16::MIN;
    for c in cells {
        lo = lo.min(c.dy);
        hi = hi.max(c.dy);
    }
    (lo, hi)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cells(n: usize) -> Vec<Cell> {
        (0..n).map(|i| Cell { state: i as u16 + 1, dy: i as i16 }).collect()
    }

    fn place() -> Placement {
        Placement { x: 1, y: 2, z: 3, rot: 0 }
    }

    #[test]
    fn stamp_reads_the_staged_cells() {
        let c = cells(4);
        let mut marks = [0u8; BANDS];
        let a = stamp(c.as_ptr(), c.len(), place(), 7, &mut marks);
        let d = cells(3);
        let mut marks = [0u8; BANDS];
        assert_ne!(a, stamp(d.as_ptr(), d.len(), place(), 7, &mut marks));
    }

    #[test]
    fn stamp_of_null_or_empty_is_zero() {
        let c = cells(4);
        let mut marks = [0u8; BANDS];
        assert_eq!(stamp(std::ptr::null(), 4, place(), 0, &mut marks), 0);
        assert_eq!(stamp(c.as_ptr(), 0, place(), 0, &mut marks), 0);
    }

    #[test]
    fn stamp_respects_a_marks_table_shorter_than_bands() {
        // A cell landing at slot 5 must be ignored rather than read out of
        // bounds when `marks` holds fewer than `BANDS` elements.
        let c = [Cell { state: 1, dy: 5 }];
        let mut marks = vec![0u8; 3];
        let at = Placement { x: 0, y: 0, z: 0, rot: 0 };
        // Must not panic or read past `marks`.
        let _ = stamp(c.as_ptr(), c.len(), at, 0, &mut marks);
        assert!(marks.iter().all(|&m| m == 0), "slot 5 is out of range for a 3-element table");
    }

    #[test]
    fn placement_and_seed_both_matter() {
        let c = cells(4);
        let mut marks = [0u8; BANDS];
        let a = stamp(c.as_ptr(), c.len(), place(), 7, &mut marks);
        let mut marks = [0u8; BANDS];
        assert_ne!(a, stamp(c.as_ptr(), c.len(), place(), 8, &mut marks));
        let moved = Placement { x: 9, ..place() };
        let mut marks = [0u8; BANDS];
        assert_ne!(a, stamp(c.as_ptr(), c.len(), moved, 7, &mut marks));
    }

    #[test]
    fn a_repeated_band_folds_only_once() {
        // Two cells landing in the same band contribute that band's mixing
        // term only on the first touch.
        let c = [Cell { state: 1, dy: 0 }, Cell { state: 2, dy: 0 }];
        let mut marks_once = [0u8; BANDS];
        let once = stamp(c.as_ptr(), 1, place(), 7, &mut marks_once);
        let mut marks_twice = [0u8; BANDS];
        let twice = stamp(c.as_ptr(), 2, place(), 7, &mut marks_twice);
        // The second cell still folds its state and index, just not the band
        // mixing term again, so the two digests differ...
        assert_ne!(once, twice);
        // ...but the band is marked exactly once either way.
        assert_eq!(marks_once.iter().filter(|&&m| m != 0).count(), 1);
        assert_eq!(marks_twice.iter().filter(|&&m| m != 0).count(), 1);
    }

    #[test]
    fn rotation_cycles_through_four_turns() {
        assert_eq!(rotate_offset(1, 0, 0), (1, 0));
        assert_eq!(rotate_offset(1, 0, 1), (0, 1));
        assert_eq!(rotate_offset(1, 0, 2), (-1, 0));
        assert_eq!(rotate_offset(1, 0, 3), (0, -1));
        // Four turns returns to the start.
        assert_eq!(rotate_offset(1, 0, 4), (1, 0));
    }

    #[test]
    fn vertical_span_brackets_the_cells() {
        let c = [Cell { state: 1, dy: -3 }, Cell { state: 2, dy: 5 }];
        assert_eq!(vertical_span(&c), (-3, 5));
        assert_eq!(vertical_span(&[]), (0, 0));
    }

    #[test]
    fn stamp_all_is_order_sensitive() {
        let a = cells(2);
        let b = cells(3);
        let fwd: Vec<_> = vec![
            (a.as_ptr(), a.len(), place()),
            (b.as_ptr(), b.len(), place()),
        ];
        let rev: Vec<_> = vec![
            (b.as_ptr(), b.len(), place()),
            (a.as_ptr(), a.len(), place()),
        ];
        let mut marks = [0u8; BANDS];
        let f = stamp_all(&fwd, 1, 1, &mut marks);
        let mut marks = [0u8; BANDS];
        let r = stamp_all(&rev, 1, 1, &mut marks);
        assert_ne!(f, r);
    }

    #[test]
    fn fold_parent_is_deterministic() {
        let c = cells(4);
        assert_eq!(fold_parent(c.as_ptr(), c.len()), fold_parent(c.as_ptr(), c.len()));
    }

    #[test]
    fn fold_parent_reflects_contents() {
        let a = cells(4);
        let b = cells(5);
        assert_ne!(fold_parent(a.as_ptr(), a.len()), fold_parent(b.as_ptr(), b.len()));
    }

    #[test]
    fn fold_parent_of_null_or_empty_is_zero() {
        let c = cells(4);
        assert_eq!(fold_parent(std::ptr::null(), 4), 0);
        assert_eq!(fold_parent(c.as_ptr(), 0), 0);
    }
}
