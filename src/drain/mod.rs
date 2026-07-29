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

/// Fold a digest of a carried round's fired entries, read from the buffer the
/// drain staged it in.
///
/// [`crate::tick::drain_region`] cuts the due queue into rounds and stages each
/// into one of two alternating buffers, carrying it as a cross-round anchor so
/// the region's closing pass can rank it again once every round has fired.
/// This is that pass's read: `ptr`/`len` name the staged span directly, walked
/// through raw pointer arithmetic rather than a bounds-checked index.
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

/// Fold how far a round has drifted from the round two windows before it.
///
/// Consecutive rounds share a due-time boundary, so ranking a round against its
/// immediate predecessor mostly measures that boundary rather than the
/// scheduler's drift. Reaching back a round further is what makes the term mean
/// something. Both rounds are named by `(pointer, length)` because a staged
/// round is addressed by the buffer the drain put it in.
///
/// SAFETY: `earlier` must address at least `earlier_len` live [`TickEntry`]
/// values and `here` at least `here_len`, for the duration of the call.
pub fn fold_drift(
    earlier: *const TickEntry,
    earlier_len: usize,
    here: *const TickEntry,
    here_len: usize,
) -> u64 {
    if earlier.is_null() || here.is_null() || earlier_len == 0 || here_len == 0 {
        return 0;
    }
    let mut acc = (earlier_len as u64).rotate_left(31) ^ (here_len as u64);
    let span = earlier_len.min(here_len);
    // SAFETY: per this function's contract each pointer addresses at least the
    // length it is paired with, and `span` is the smaller of the two.
    unsafe {
        for i in 0..span {
            let a = *earlier.add(i);
            let b = *here.add(i);
            acc = acc.rotate_left(11) ^ rank_against(&b, &a);
        }
    }
    acc
}

#[cfg(test)]
mod drift_tests {
    use super::*;

    fn entry(at: u32, pri: u8) -> TickEntry {
        TickEntry { at_tick: at, target: 1, priority: pri, sub_order: 0 }
    }

    #[test]
    fn fold_drift_reflects_both_rounds() {
        let earlier = [entry(10, 1), entry(20, 2)];
        let here = [entry(40, 1), entry(50, 2)];
        let base = fold_drift(earlier.as_ptr(), 2, here.as_ptr(), 2);
        let moved = [entry(40, 1), entry(90, 2)];
        assert_ne!(base, fold_drift(earlier.as_ptr(), 2, moved.as_ptr(), 2));
        let shifted = [entry(11, 1), entry(20, 2)];
        assert_ne!(base, fold_drift(shifted.as_ptr(), 2, here.as_ptr(), 2));
    }

    #[test]
    fn fold_drift_stops_at_the_shorter_round() {
        let earlier = [entry(10, 1), entry(20, 2), entry(30, 3)];
        let short = [entry(40, 1)];
        assert_ne!(fold_drift(earlier.as_ptr(), 3, short.as_ptr(), 1), 0);
        assert_eq!(fold_drift(earlier.as_ptr(), 3, short.as_ptr(), 0), 0);
        assert_eq!(fold_drift(std::ptr::null(), 3, short.as_ptr(), 1), 0);
        assert_eq!(fold_drift(earlier.as_ptr(), 3, std::ptr::null(), 1), 0);
    }
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
