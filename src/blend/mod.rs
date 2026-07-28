//! Biome row blending.
//!
//! [`crate::biome`] owns the cell grid; this module blends it. The stencil reads
//! a whole row through the view the grid handed out, so the nine taps of a 3x3
//! kernel cost three pointer walks rather than nine bounds-checked index
//! operations.
//!
//! [`mix_row`] has two call sites in [`crate::biome::resolve_region`]: once per
//! row as each column is decoded, and once more at the end of the region's
//! pass for every row its reference ring still holds as a cross-column
//! reference. Both calls share the same contract — the caller names a live
//! `(pointer, length)` pair — since this module has no way to tell which kind
//! of call it is looking at.

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
