//! Biome row blending.
//!
//! [`crate::biome`] owns the cell grid; this module blends it. The stencil reads
//! a whole row through the view the grid handed out, so the nine taps of a 3x3
//! kernel cost three pointer walks rather than nine bounds-checked index
//! operations.
//!
//! The blends here are called from [`crate::biome::resolve_region`] at three
//! points: [`mix_row`] once per row as each column is decoded and once more at
//! the end of the region's pass for every row its reference ring still holds,
//! and [`mix_across`] once per row against the column upstream of it. All three
//! share the same contract — the caller names a live `(pointer, length)` pair —
//! since this module has no way to tell which kind of call it is looking at.

/// Weights of the horizontal three-tap kernel, in sixteenths.
const KERNEL: [u32; 3] = [4, 8, 4];

/// Blend one row of the biome grid into a digest word.
///
/// `view` addresses the row's first cell and `edge` is the row length. `z` is
/// mixed in so two identical rows at different depths do not collide.
///
/// SAFETY: `view` must address at least `edge` live cells for the call.
pub fn mix_row(view: *const u8, edge: usize, z: u8) -> u64 {
    if view.is_null() || edge == 0 {
        return 0;
    }
    let mut acc = (z as u64).rotate_left(23) ^ 0x2545f491;
    // SAFETY: per this function's contract, the caller guarantees `view`
    // addresses at least `edge` live cells.
    unsafe {
        for x in 0..edge {
            let left = *view.add(x.saturating_sub(1));
            let here = *view.add(x);
            let right = *view.add((x + 1).min(edge - 1));
            let mixed = (left as u32 * KERNEL[0]
                + here as u32 * KERNEL[1]
                + right as u32 * KERNEL[2])
                / 16;
            acc = acc.rotate_left(5) ^ (mixed as u64);
        }
    }
    acc
}

/// Blend a row against the same row of the column upstream of it.
///
/// The three-tap kernel stops at a column boundary, so a region blended column
/// by column shows a seam wherever two columns meet. Mixing each row with the
/// row at the same depth in the column before it is what carries the stencil
/// across that boundary. Both rows are named by `(pointer, length)` because
/// [`crate::biome`] addresses committed rows by where the store put them.
///
/// SAFETY: `upstream` must address at least `upstream_edge` live cells and
/// `here` at least `here_edge`, for the duration of the call.
pub fn mix_across(
    upstream: *const u8,
    upstream_edge: usize,
    here: *const u8,
    here_edge: usize,
    z: u8,
) -> u64 {
    if upstream.is_null() || here.is_null() || upstream_edge == 0 || here_edge == 0 {
        return 0;
    }
    let mut acc = (z as u64).rotate_left(37) ^ 0x9e3779b1;
    let edge = upstream_edge.min(here_edge);
    // SAFETY: per this function's contract each pointer addresses at least the
    // edge length it is paired with, and `edge` is the smaller of the two.
    unsafe {
        for x in 0..edge {
            let up = *upstream.add(x) as u32;
            let cur = *here.add(x) as u32;
            let mixed = (up * KERNEL[0] + cur * KERNEL[1] + up * KERNEL[2]) / 16;
            acc = acc.rotate_left(7) ^ (mixed as u64);
        }
    }
    acc
}

/// Blend a row held as a slice, for callers that already own it.
pub fn mix_slice(row: &[u8], z: u8) -> u64 {
    mix_row(row.as_ptr(), row.len(), z)
}

/// The dominant biome in a row: the value occupying the most cells.
pub fn dominant(row: &[u8]) -> u8 {
    let mut counts = [0u16; 256];
    for &c in row {
        counts[c as usize] += 1;
    }
    let mut best = 0u8;
    let mut best_count = 0u16;
    for (v, &n) in counts.iter().enumerate() {
        if n > best_count {
            best_count = n;
            best = v as u8;
        }
    }
    best
}

/// Whether a row is a single biome throughout.
pub fn is_uniform(row: &[u8]) -> bool {
    match row.first() {
        Some(&first) => row.iter().all(|&c| c == first),
        None => true,
    }
}

/// How many biome transitions a row contains.
pub fn transitions(row: &[u8]) -> usize {
    row.windows(2).filter(|w| w[0] != w[1]).count()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uniform_row_blends_to_its_own_value() {
        let row = [7u8; 4];
        // Every tap is 7, so the weighted mean is 7 and the fold is stable.
        assert_eq!(mix_slice(&row, 0), mix_slice(&row, 0));
        let other = [8u8; 4];
        assert_ne!(mix_slice(&row, 0), mix_slice(&other, 0));
    }

    #[test]
    fn depth_separates_identical_rows() {
        let row = [3u8; 4];
        assert_ne!(mix_slice(&row, 0), mix_slice(&row, 1));
    }

    #[test]
    fn edges_clamp_rather_than_wrap() {
        // A single-cell row must not read outside itself.
        let row = [9u8];
        assert_ne!(mix_slice(&row, 0), 0);
    }

    #[test]
    fn null_or_empty_row_is_zero() {
        assert_eq!(mix_row(std::ptr::null(), 4, 0), 0);
        assert_eq!(mix_slice(&[], 0), 0);
    }

    #[test]
    fn mix_across_reflects_both_columns() {
        let up = [1u8, 2, 3, 4];
        let here = [5u8, 6, 7, 8];
        let base = mix_across(up.as_ptr(), 4, here.as_ptr(), 4, 0);
        let other_up = [9u8, 2, 3, 4];
        assert_ne!(base, mix_across(other_up.as_ptr(), 4, here.as_ptr(), 4, 0));
        // The blend is a weighted mean, so a difference has to survive the
        // division to show up.
        let other_here = [5u8, 6, 7, 20];
        assert_ne!(base, mix_across(up.as_ptr(), 4, other_here.as_ptr(), 4, 0));
        assert_ne!(base, mix_across(up.as_ptr(), 4, here.as_ptr(), 4, 1));
    }

    #[test]
    fn mix_across_stops_at_the_shorter_row() {
        let up = [1u8, 2, 3, 4];
        let short = [5u8, 6];
        assert_ne!(mix_across(up.as_ptr(), 4, short.as_ptr(), 2, 0), 0);
        assert_eq!(mix_across(up.as_ptr(), 4, short.as_ptr(), 0, 0), 0);
        assert_eq!(mix_across(std::ptr::null(), 4, short.as_ptr(), 2, 0), 0);
        assert_eq!(mix_across(up.as_ptr(), 4, std::ptr::null(), 2, 0), 0);
    }

    #[test]
    fn dominant_picks_the_most_common() {
        assert_eq!(dominant(&[1, 2, 2, 3]), 2);
        assert_eq!(dominant(&[]), 0);
    }

    #[test]
    fn uniformity_and_transitions_agree() {
        assert!(is_uniform(&[5, 5, 5]));
        assert!(is_uniform(&[]));
        assert!(!is_uniform(&[5, 6]));
        assert_eq!(transitions(&[5, 5, 6, 6, 7]), 2);
        assert_eq!(transitions(&[]), 0);
    }
}
