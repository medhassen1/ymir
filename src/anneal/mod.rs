//! Relight folding.
//!
//! [`crate::relight`] owns the column buffers; this module folds them. The
//! anneal pass smooths a merged run of columns so that a light discontinuity at
//! a merge boundary does not show up as a visible seam.

use crate::relight::ColumnBuf;

/// Fold a merged run of relit columns into a digest.
pub fn fold_columns(run: &[ColumnBuf]) -> u64 {
    let mut acc = 0xffu64;
    for buf in run {
        let mut h = buf.len() as u64;
        for &l in buf.levels() {
            h = h.rotate_left(3) ^ (l as u64);
        }
        acc = acc.wrapping_mul(0x100000001b3) ^ h;
    }
    acc
}

/// Smooth a column's levels in place, averaging each with its neighbours.
///
/// A merge boundary leaves a step in the light; one anneal pass softens it.
pub fn smooth(levels: &mut [u8]) {
    if levels.len() < 3 {
        return;
    }
    let original = levels.to_vec();
    for i in 1..original.len() - 1 {
        let sum = original[i - 1] as u32 + original[i] as u32 * 2 + original[i + 1] as u32;
        levels[i] = (sum / 4) as u8;
    }
}

/// The largest step between adjacent levels, which is what a seam looks like.
pub fn max_step(levels: &[u8]) -> u8 {
    levels.windows(2).map(|w| w[0].abs_diff(w[1])).max().unwrap_or(0)
}

/// The mean level across a column, in sixteenths.
pub fn mean_level(levels: &[u8]) -> u32 {
    if levels.is_empty() {
        return 0;
    }
    let sum: u32 = levels.iter().map(|&l| l as u32).sum();
    (sum * 16) / levels.len() as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fold_is_order_sensitive() {
        let a = ColumnBuf::pack(&[1, 2, 3]);
        let b = ColumnBuf::pack(&[4, 5, 6]);
        let fwd = fold_columns(&[a.view(), b.view()]);
        let rev = fold_columns(&[b.view(), a.view()]);
        assert_ne!(fwd, rev);
    }

    #[test]
    fn fold_of_empty_run_is_the_seed() {
        assert_eq!(fold_columns(&[]), 0xff);
    }

    #[test]
    fn smooth_softens_a_step() {
        let mut levels = [0u8, 0, 12, 12, 12];
        let before = max_step(&levels);
        smooth(&mut levels);
        assert!(max_step(&levels) < before);
    }

    #[test]
    fn smooth_leaves_endpoints_alone() {
        let mut levels = [3u8, 9, 3];
        smooth(&mut levels);
        assert_eq!(levels[0], 3);
        assert_eq!(levels[2], 3);
    }

    #[test]
    fn smooth_ignores_short_columns() {
        let mut levels = [1u8, 9];
        smooth(&mut levels);
        assert_eq!(levels, [1, 9]);
    }

    #[test]
    fn summaries_agree_with_the_data() {
        assert_eq!(max_step(&[1, 5, 2]), 4);
        assert_eq!(max_step(&[]), 0);
        assert_eq!(mean_level(&[4, 8]), 6 * 16);
        assert_eq!(mean_level(&[]), 0);
    }
}
