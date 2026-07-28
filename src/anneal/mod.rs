//! Relight folding.
//!
//! [`crate::relight`] owns the region-lifetime slab ring; this module folds
//! what it committed. The anneal pass smooths a merged run of columns so that
//! a light discontinuity at a merge boundary does not show up as a visible
//! seam.

/// Fold one column's committed levels into a digest word.
///
/// `ptr`/`count` name a column's packed levels, however they are currently
/// held — resolved fresh from the ring or read back from a retained span.
///
/// SAFETY: `ptr` must address at least `count` live levels for the call.
pub fn fold_one(ptr: *const u8, count: usize) -> u64 {
    let mut h = count as u64;
    if !ptr.is_null() && count != 0 {
        // SAFETY: guaranteed by the precondition documented above.
        let levels = unsafe { std::slice::from_raw_parts(ptr, count) };
        for &l in levels {
            h = h.rotate_left(3) ^ (l as u64);
        }
    }
    h
}

/// Fold a run of columns' `(pointer, count)` spans into a digest, in order.
///
/// SAFETY: for every `(ptr, count)` pair, `ptr` must address at least `count`
/// live levels.
pub fn fold_columns(run: &[(*const u8, usize)]) -> u64 {
    let mut acc = 0xffu64;
    for &(ptr, count) in run {
        acc = acc.wrapping_mul(0x100000001b3) ^ fold_one(ptr, count);
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
        let a = [1u8, 2, 3];
        let b = [4u8, 5, 6];
        let fwd = fold_columns(&[(a.as_ptr(), a.len()), (b.as_ptr(), b.len())]);
        let rev = fold_columns(&[(b.as_ptr(), b.len()), (a.as_ptr(), a.len())]);
        assert_ne!(fwd, rev);
    }

    #[test]
    fn fold_of_empty_run_is_the_seed() {
        assert_eq!(fold_columns(&[]), 0xff);
    }

    #[test]
    fn fold_walks_only_the_paired_count() {
        // A count shorter than the buffer's own length only folds the prefix.
        let a = [1u8, 2, 3, 4, 5];
        let short = fold_columns(&[(a.as_ptr(), 2)]);
        let full = fold_columns(&[(a.as_ptr(), a.len())]);
        assert_ne!(short, full);
    }

    #[test]
    fn fold_one_treats_null_as_the_count_only() {
        assert_eq!(fold_one(std::ptr::null(), 0), 0);
        assert_eq!(fold_one(std::ptr::null(), 4), 4);
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
