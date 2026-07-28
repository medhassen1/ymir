//! Tick firing.
//!
//! [`crate::tick`] owns the queue; this module fires it. Each entry is ranked
//! against its round's anchor entry, which arrives by pointer so the ranking
//! loop does not re-index the queue per comparison.

use crate::tick::TickEntry;

/// Fire the due ticks and fold a digest of what ran.
///
/// `anchor` names the entry every rank is measured against; `slack` is the
/// per-drain work allowance, which bounds how many ticks one drain may fire.
///
/// SAFETY: `anchor` must point at a live queue entry for the duration of the
/// call.
pub fn fire(entries: &[TickEntry], anchor: *const TickEntry, slack: usize) -> u64 {
    if anchor.is_null() {
        return 0;
    }
    // SAFETY: per this function's contract, the caller guarantees `anchor`
    // points at a live queue entry for the duration of the call.
    let base = unsafe { *anchor };

    let mut acc = (base.at_tick as u64).wrapping_mul(0x9e3779b1) ^ (base.priority as u64);
    for (i, e) in entries.iter().enumerate() {
        if i >= slack {
            break;
        }
        let rank = rank_against(e, &base);
        acc = acc.rotate_left(5) ^ rank;
        acc = acc.wrapping_add(e.target as u64);
    }
    acc
}

/// How far `e` sits from the anchor in the ordering.
fn rank_against(e: &TickEntry, base: &TickEntry) -> u64 {
    let dt = (e.at_tick as i64 - base.at_tick as i64).unsigned_abs();
    let dp = (e.priority as i64 - base.priority as i64).unsigned_abs();
    dt.wrapping_mul(31).wrapping_add(dp).wrapping_add(e.sub_order as u64)
}

/// Fold a digest of a retained round's fired entries, read from the region's
/// entry pool.
///
/// [`crate::tick::drain_region`] drains the due queue in rounds, committing
/// each round's entries into a region-lifetime pool, and a busy round is kept
/// as a retained cross-round anchor so the region's closing pass can rank it
/// again once every round has fired. This is that pass's read: `ptr`/`len`
/// name the committed span directly, walked through raw pointer arithmetic
/// rather than a bounds-checked index.
///
/// SAFETY: `ptr` must address at least `len` live [`TickEntry`] values for the
/// call.
pub fn fold_anchor(ptr: *const TickEntry, len: usize) -> u64 {
    if ptr.is_null() || len == 0 {
        return 0;
    }
    let mut acc = 0x2545f491u64;
    // SAFETY: per this function's contract, the caller guarantees `ptr`
    // addresses at least `len` live entries.
    unsafe {
        for i in 0..len {
            let e = *ptr.add(i);
            acc = acc.rotate_left(9) ^ (e.at_tick as u64) ^ ((e.target as u64) << 8) ^ (e.priority as u64);
        }
    }
    acc
}

/// The span of due times across a set of entries, as `(earliest, latest)`.
pub fn time_span(entries: &[TickEntry]) -> (u32, u32) {
    if entries.is_empty() {
        return (0, 0);
    }
    let mut lo = u32::MAX;
    let mut hi = 0u32;
    for e in entries {
        lo = lo.min(e.at_tick);
        hi = hi.max(e.at_tick);
    }
    (lo, hi)
}

/// How many distinct priorities appear in a set of entries.
pub fn priority_variety(entries: &[TickEntry]) -> usize {
    let mut seen = [false; 256];
    let mut count = 0usize;
    for e in entries {
        if !seen[e.priority as usize] {
            seen[e.priority as usize] = true;
            count += 1;
        }
    }
    count
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entry(at: u32, pri: u8, sub: u8) -> TickEntry {
        TickEntry { at_tick: at, target: sub as u16, priority: pri, sub_order: sub }
    }

    #[test]
    fn fire_ranks_against_the_anchor() {
        let e = [entry(1, 0, 0), entry(2, 0, 1)];
        let a = fire(&e, e.as_ptr(), 16);
        let other = [entry(9, 0, 0), entry(2, 0, 1)];
        assert_ne!(a, fire(&other, other.as_ptr(), 16));
    }

    #[test]
    fn fire_of_null_anchor_is_zero() {
        let e = [entry(1, 0, 0)];
        assert_eq!(fire(&e, std::ptr::null(), 16), 0);
    }

    #[test]
    fn slack_bounds_the_work() {
        let e: Vec<TickEntry> = (0..10).map(|i| entry(i, 0, i as u8)).collect();
        let few = fire(&e, e.as_ptr(), 2);
        let many = fire(&e, e.as_ptr(), 10);
        assert_ne!(few, many);
        // Slack beyond the queue length behaves like the full queue.
        assert_eq!(many, fire(&e, e.as_ptr(), 100));
    }

    #[test]
    fn rank_is_symmetric_in_distance() {
        let base = entry(10, 5, 0);
        let below = entry(8, 5, 0);
        let above = entry(12, 5, 0);
        assert_eq!(rank_against(&below, &base), rank_against(&above, &base));
    }

    #[test]
    fn time_span_brackets_the_entries() {
        let e = [entry(7, 0, 0), entry(3, 0, 1), entry(9, 0, 2)];
        assert_eq!(time_span(&e), (3, 9));
        assert_eq!(time_span(&[]), (0, 0));
    }

    #[test]
    fn priority_variety_counts_distinct_levels() {
        let e = [entry(0, 1, 0), entry(0, 1, 1), entry(0, 4, 2)];
        assert_eq!(priority_variety(&e), 2);
        assert_eq!(priority_variety(&[]), 0);
    }

    #[test]
    fn fold_anchor_is_deterministic() {
        let e = [entry(1, 0, 0), entry(2, 0, 1)];
        assert_eq!(fold_anchor(e.as_ptr(), e.len()), fold_anchor(e.as_ptr(), e.len()));
    }

    #[test]
    fn fold_anchor_reflects_contents() {
        let a = [entry(1, 0, 0), entry(2, 0, 1)];
        let b = [entry(1, 0, 0), entry(9, 0, 1)];
        assert_ne!(fold_anchor(a.as_ptr(), a.len()), fold_anchor(b.as_ptr(), b.len()));
    }

    #[test]
    fn fold_anchor_of_null_or_empty_is_zero() {
        let e = [entry(1, 0, 0)];
        assert_eq!(fold_anchor(std::ptr::null(), 4), 0);
        assert_eq!(fold_anchor(e.as_ptr(), 0), 0);
    }
}
