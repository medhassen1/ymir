//! Fold a resident section view into the rebuild digest.
//!
//! Split out from [`crate::seccache`] so the folding arithmetic can be tested
//! and tuned without touching the cache's storage discipline. The functions here
//! take plain slices and never own anything.

/// Fold a digest of a resident column view.
///
/// `blocks` is the column's decoded block count, mixed in so that two columns
/// with the same sampled units but different densities do not collide.
pub fn fold_view(units: &[u32], blocks: usize) -> u64 {
    let mut acc = (blocks as u64).wrapping_mul(0x9e3779b1).wrapping_add(0x2545f491);
    for &u in units {
        acc = acc.rotate_left(5) ^ (u as u64);
    }
    acc
}

/// Fold a run of views into one digest, in order.
pub fn fold_run(views: &[&[u32]]) -> u64 {
    let mut acc = 0xffu64;
    for v in views {
        acc = acc.wrapping_mul(0x100000001b3) ^ fold_view(v, v.len());
    }
    acc
}

/// The number of distinct section tags present in a packed unit stream.
///
/// Used by the inspector to report how many section shapes a column mixes.
pub fn tag_variety(units: &[u32]) -> usize {
    let mut seen: Vec<u32> = Vec::new();
    for &u in units {
        let tag = u >> 16;
        if !seen.contains(&tag) {
            seen.push(tag);
        }
    }
    seen.len()
}

/// The mean block state across a packed unit stream, ignoring air.
pub fn mean_state(units: &[u32]) -> u32 {
    let mut sum = 0u64;
    let mut count = 0u64;
    for &u in units {
        let state = u & 0xffff;
        if state != 0 {
            sum += state as u64;
            count += 1;
        }
    }
    sum.checked_div(count).unwrap_or(0) as u32
}

/// Fold a resident column view walked as `units` packed entries.
///
/// The caller already knows how many units its column contributed, so the span
/// is walked from the buffer's start rather than re-deriving its length. That
/// keeps the fold to one pass over contiguous memory for a whole region.
///
/// SAFETY: `cursor` must address at least `units` live entries for the call.
pub fn fold_span(cursor: *const u32, units: usize, blocks: usize) -> u64 {
    if cursor.is_null() || units == 0 {
        return fold_view(&[], blocks);
    }
    // SAFETY: `cursor` names the cache's resident buffer for this column and
    // `units` is the entry count that column packed into it.
    let span = unsafe { std::slice::from_raw_parts(cursor, units) };
    fold_view(span, blocks)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fold_view_is_deterministic() {
        let u = [1u32, 2, 3];
        assert_eq!(fold_view(&u, 9), fold_view(&u, 9));
    }

    #[test]
    fn fold_view_separates_block_count() {
        let u = [1u32, 2, 3];
        assert_ne!(fold_view(&u, 9), fold_view(&u, 10));
    }

    #[test]
    fn fold_view_is_order_sensitive() {
        assert_ne!(fold_view(&[1, 2], 2), fold_view(&[2, 1], 2));
    }

    #[test]
    fn empty_view_still_folds() {
        assert_ne!(fold_view(&[], 0), 0);
    }

    #[test]
    fn tag_variety_counts_distinct_high_halves() {
        let units = [0x0001_0005, 0x0001_0006, 0x0002_0005];
        assert_eq!(tag_variety(&units), 2);
        assert_eq!(tag_variety(&[]), 0);
    }

    #[test]
    fn mean_state_ignores_air() {
        let units = [0x0000_0000, 0x0000_000a, 0x0000_0014];
        assert_eq!(mean_state(&units), 15);
        assert_eq!(mean_state(&[0, 0]), 0);
    }

    #[test]
    fn fold_run_chains_views() {
        let a: &[u32] = &[1, 2];
        let b: &[u32] = &[3];
        assert_ne!(fold_run(&[a, b]), fold_run(&[b, a]));
    }
}
