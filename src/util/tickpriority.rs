//! Deterministic ordering for the scheduled tick queue.
//!
//! Block updates scheduled for the same future tick must still resolve in a
//! fixed, repeatable order. Each [`TickEntry`] carries a `sub_order`
//! counter so "equal tick, equal priority" ties break by insertion order.

use std::cmp::{Ordering, Reverse};

/// One scheduled tick: fire `target` at tick `at_tick`, resolving ties
/// against other entries at the same tick by `priority` (higher first) and
/// then by `sub_order` (lower — i.e. earlier-scheduled — first).
#[derive(Debug, Clone)]
pub struct TickEntry<T> {
    /// The world tick at which this entry becomes eligible to fire.
    pub at_tick: u64,
    /// Higher priority entries at the same tick fire first.
    pub priority: i32,
    /// A monotonic counter assigned at scheduling time, breaking ties
    /// between entries with equal `at_tick` and `priority` deterministically.
    pub sub_order: u64,
    /// What this entry acts on (a block position, an entity id, ...).
    pub target: T,
}

impl<T> TickEntry<T> {
    /// The ordering key: earlier tick first, then higher priority first,
    /// then lower `sub_order` first. `target` never participates in
    /// ordering, so entries are compared purely on scheduling metadata.
    fn key(&self) -> (u64, Reverse<i32>, u64) {
        (self.at_tick, Reverse(self.priority), self.sub_order)
    }
}

impl<T> PartialEq for TickEntry<T> {
    fn eq(&self, other: &Self) -> bool {
        self.key() == other.key()
    }
}

impl<T> Eq for TickEntry<T> {}

impl<T> PartialOrd for TickEntry<T> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<T> Ord for TickEntry<T> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.key().cmp(&other.key())
    }
}

/// Compare two entries by the same rule as [`TickEntry`]'s `Ord` impl,
/// exposed as a free function for callers that want a comparator to pass
/// to a sort routine without naming the trait method.
pub fn compare_ticks<T>(a: &TickEntry<T>, b: &TickEntry<T>) -> Ordering {
    a.cmp(b)
}

/// The index at which `entry` should be inserted into an already-sorted
/// `queue` to keep it sorted, choosing the first such index (so equal keys
/// are inserted after any existing equal entries, preserving their
/// relative order).
fn lower_bound<T>(queue: &[TickEntry<T>], entry: &TickEntry<T>) -> usize {
    let mut lo = 0usize;
    let mut hi = queue.len();
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        // SAFETY: the loop invariant `lo < hi` together with `hi` starting
        // at `queue.len()` and only ever decreasing means `mid` (which is
        // always `>= lo` and `< hi`) satisfies `mid < queue.len()`.
        let mid_entry = unsafe { queue.get_unchecked(mid) };
        if mid_entry < entry {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    lo
}

/// Insert `entry` into `queue`, which must already be sorted by
/// [`TickEntry`]'s ordering, keeping it sorted.
pub fn insert_sorted<T>(queue: &mut Vec<TickEntry<T>>, entry: TickEntry<T>) {
    let idx = lower_bound(queue, &entry);
    queue.insert(idx, entry);
}

/// Pop the front entry if it is due (`at_tick <= current_tick`), leaving
/// the queue untouched and returning `None` otherwise.
pub fn pop_due<T>(queue: &mut Vec<TickEntry<T>>, current_tick: u64) -> Option<TickEntry<T>> {
    if queue.is_empty() {
        return None;
    }
    // SAFETY: the `is_empty` check above guarantees `queue` has at least
    // one element, so index 0 is valid.
    let front_due = unsafe { queue.get_unchecked(0) }.at_tick <= current_tick;
    if front_due {
        Some(queue.remove(0))
    } else {
        None
    }
}

const PRIORITY_BANDS: [&str; 5] = ["lowest", "low", "normal", "high", "highest"];

/// A human-readable band name for a priority value, for logs and debug
/// tooling: priorities outside `[-2, 2]` clamp to the nearest band.
pub fn priority_band_label(priority: i32) -> &'static str {
    let idx = ((priority.clamp(-2, 2) + 2) as usize).min(PRIORITY_BANDS.len() - 1);
    // SAFETY: `priority.clamp(-2, 2) + 2` is in `0..=4`, and the extra
    // `.min(PRIORITY_BANDS.len() - 1)` makes that bound explicit at the
    // type level too; either way `idx` is in `0..PRIORITY_BANDS.len()`.
    unsafe { PRIORITY_BANDS.get_unchecked(idx) }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entry(at_tick: u64, priority: i32, sub_order: u64) -> TickEntry<u32> {
        TickEntry { at_tick, priority, sub_order, target: 0 }
    }

    #[test]
    fn earlier_tick_always_sorts_first_regardless_of_priority() {
        let earlier = entry(10, -100, 0);
        let later = entry(11, 100, 0);
        assert!(earlier < later);
    }

    #[test]
    fn equal_tick_orders_by_priority_descending() {
        let high = entry(5, 10, 0);
        let low = entry(5, 1, 1);
        assert!(high < low);
        assert_eq!(compare_ticks(&high, &low), Ordering::Less);
    }

    #[test]
    fn equal_tick_and_priority_breaks_ties_by_sub_order_deterministically() {
        let first = entry(5, 3, 0);
        let second = entry(5, 3, 1);
        assert!(first < second);
        // The comparison is a pure function of the fields, so repeating it
        // (as a fresh scheduler run would) always agrees.
        assert!(first < second);
        assert_eq!(compare_ticks(&second, &first), Ordering::Greater);
    }

    #[test]
    fn insert_sorted_maintains_order_for_a_mixed_batch() {
        let mut queue: Vec<TickEntry<u32>> = Vec::new();
        let batch = [
            entry(5, 0, 2),
            entry(3, 0, 0),
            entry(5, 10, 1),
            entry(3, 5, 3),
            entry(4, 0, 4),
        ];
        for e in batch {
            insert_sorted(&mut queue, e);
        }
        for w in queue.windows(2) {
            assert!(w[0] <= w[1], "queue not sorted: {:?} then {:?}", w[0], w[1]);
        }
        assert_eq!(queue.len(), 5);
    }

    #[test]
    fn pop_due_only_returns_entries_at_or_before_current_tick() {
        let mut queue: Vec<TickEntry<u32>> = Vec::new();
        insert_sorted(&mut queue, entry(10, 0, 0));
        insert_sorted(&mut queue, entry(20, 0, 1));
        assert!(pop_due(&mut queue, 5).is_none());
        let popped = pop_due(&mut queue, 10).expect("tick 10 entry should be due");
        assert_eq!(popped.at_tick, 10);
        assert!(pop_due(&mut queue, 15).is_none());
        let popped2 = pop_due(&mut queue, 25).expect("tick 20 entry should be due");
        assert_eq!(popped2.at_tick, 20);
        assert!(queue.is_empty());
    }

    #[test]
    fn priority_band_label_clamps_at_the_extremes() {
        assert_eq!(priority_band_label(0), "normal");
        assert_eq!(priority_band_label(2), "highest");
        assert_eq!(priority_band_label(-2), "lowest");
        assert_eq!(priority_band_label(1000), "highest");
        assert_eq!(priority_band_label(-1000), "lowest");
    }
}
