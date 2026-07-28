//! Template stamping.
//!
//! [`crate::structure`] resolves and stages templates; this module stamps them.
//! A staged template arrives as a `(pointer, length)` view over its packed cells
//! plus the placement to apply, so a template placed many times is read from one
//! buffer instead of being cloned per placement.

use crate::structure::{Cell, Placement};

/// Stamp one staged template and fold a digest of the cells it writes.
///
/// SAFETY: `cells`/`len` must name a live packed template for the call.
pub fn stamp(cells: *const Cell, len: usize, at: Placement, seed: u32) -> u64 {
    if cells.is_null() || len == 0 {
        return 0;
    }
    // SAFETY (claimed): the stager keeps every template's buffer alive for the
    // whole stamping pass, so this view names live cells.
    let view = unsafe { std::slice::from_raw_parts(cells, len) };

    let mut acc = placement_seed(at, seed);
    for (i, cell) in view.iter().enumerate() {
        let y = cell.dy.wrapping_add(at.y);
        acc = acc.rotate_left(7) ^ (cell.state as u64);
        acc = acc.wrapping_add((y as i64 as u64) ^ (i as u64));
    }
    acc
}

/// Stamp every staged template in order.
pub fn stamp_all(staged: &[(*const Cell, usize, Placement)], seed: u32, chunks: usize) -> u64 {
    let mut acc = 0xffu64 ^ (chunks as u64);
    for &(cells, len, at) in staged {
        acc = acc.wrapping_mul(0x100000001b3) ^ stamp(cells, len, at, seed);
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
        let a = stamp(c.as_ptr(), c.len(), place(), 7);
        let d = cells(3);
        assert_ne!(a, stamp(d.as_ptr(), d.len(), place(), 7));
    }

    #[test]
    fn stamp_of_null_or_empty_is_zero() {
        let c = cells(4);
        assert_eq!(stamp(std::ptr::null(), 4, place(), 0), 0);
        assert_eq!(stamp(c.as_ptr(), 0, place(), 0), 0);
    }

    #[test]
    fn placement_and_seed_both_matter() {
        let c = cells(4);
        let a = stamp(c.as_ptr(), c.len(), place(), 7);
        assert_ne!(a, stamp(c.as_ptr(), c.len(), place(), 8));
        let moved = Placement { x: 9, ..place() };
        assert_ne!(a, stamp(c.as_ptr(), c.len(), moved, 7));
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
        assert_ne!(stamp_all(&fwd, 1, 1), stamp_all(&rev, 1, 1));
    }
}
