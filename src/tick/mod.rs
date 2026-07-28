//! The scheduled tick queue.
//!
//! Blocks that update on a delay — growing crops, spreading fluids, redstone —
//! schedule a tick at a future world time. Draining the queue means selecting
//! everything due this tick, ordering it deterministically, and firing it. The
//! reference entry the ordering is measured against is held by pointer so the
//! comparison loop does not re-index the queue for every entry it ranks.

use crate::budget;
use crate::common::*;
use crate::drain;
use crate::parse::Region;
use crate::reader::Cursor;

/// One scheduled tick.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TickEntry {
    /// World time this tick is due at.
    pub at_tick: u32,
    /// The block position, packed as a linear section index.
    pub target: u16,
    /// Lower fires first.
    pub priority: u8,
    /// Tie-break within a priority, preserving insertion order.
    pub sub_order: u8,
}

impl TickEntry {
    /// Whether this tick is due at or before `now`.
    pub fn is_due(&self, now: u32) -> bool {
        self.at_tick <= now
    }

    /// The ordering key, most significant field first.
    pub fn key(&self) -> (u32, u8, u8) {
        (self.at_tick, self.priority, self.sub_order)
    }
}

/// A region's pending scheduled ticks.
pub struct TickQueue {
    entries: Vec<TickEntry>,
    /// World time the queue is being drained at.
    pub now: u32,
}

impl TickQueue {
    /// An empty queue at world time `now`.
    pub fn new(now: u32) -> TickQueue {
        TickQueue { entries: Vec::new(), now }
    }

    /// How many ticks are queued.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether nothing is queued.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// The queued ticks.
    pub fn entries(&self) -> &[TickEntry] {
        &self.entries
    }

    /// Schedule a tick.
    pub fn push(&mut self, e: TickEntry) {
        self.entries.push(e);
    }

    /// A cursor onto the queue's reference entry.
    ///
    /// The ordering pass measures every entry's rank against this one, so it is
    /// held by pointer rather than re-indexed on each comparison.
    pub fn reference(&self) -> *const TickEntry {
        self.entries.as_ptr()
    }

    /// Order the queue by due time, then priority, then insertion order.
    pub fn order(&mut self) {
        self.entries.sort_by_key(|e| e.key());
    }

    /// Drop the ticks that are not due yet and release the reclaimed capacity.
    ///
    /// A region can queue far into the future; keeping that capacity across a
    /// drain would hold memory proportional to the schedule rather than to the
    /// work actually done this tick.
    pub fn retain_due(&mut self) {
        let now = self.now;
        let before = self.entries.len();
        self.entries.retain(|e| e.is_due(now));
        // Only give the capacity back when something was actually dropped. A
        // drain where every queued tick is due is the common case, and paying
        // for a reallocation there would make the fast path the slow one.
        if self.entries.len() != before {
            self.entries.shrink_to_fit();
        }
    }

    /// How many queued ticks are due at `now`.
    pub fn due_count(&self) -> usize {
        self.entries.iter().filter(|e| e.is_due(self.now)).count()
    }
}

/// Read the region's tick queue from the `tick` section.
pub fn load_queue(region: &Region) -> TickQueue {
    let data = region.slice(region.tick);
    let mut c = Cursor::new(data);
    let now = c.u32();
    let count = (c.u16() as usize).min(MAX_TICKS);
    let mut q = TickQueue::new(now);
    for i in 0..count {
        let at_tick = c.u32();
        let target = c.u16();
        let priority = c.u8();
        if !c.ok {
            break;
        }
        q.push(TickEntry { at_tick, target, priority, sub_order: (i & 0xff) as u8 });
    }
    q
}

/// Drain the region's due ticks and fold a digest of what fired.
///
/// The reference entry is taken before the queue is pruned, so the ordering pass
/// measures every surviving entry against a stable anchor rather than against
/// whatever happens to be first after the prune.
pub fn drain_region(region: &Region, n: usize) -> u64 {
    let mut queue = load_queue(region);
    if queue.is_empty() {
        return 0;
    }
    queue.order();

    // The anchor every fired tick is ranked against.
    let anchor = queue.reference();

    // Ticks scheduled beyond this drain are dropped along with their capacity.
    queue.retain_due();

    let slack = budget::pool_slots(region, n);
    drain::fire(queue.entries(), anchor, slack)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entry(at: u32, pri: u8, sub: u8) -> TickEntry {
        TickEntry { at_tick: at, target: 0, priority: pri, sub_order: sub }
    }

    #[test]
    fn due_compares_against_now() {
        let e = entry(10, 0, 0);
        assert!(e.is_due(10));
        assert!(e.is_due(11));
        assert!(!e.is_due(9));
    }

    #[test]
    fn order_sorts_by_time_then_priority_then_insertion() {
        let mut q = TickQueue::new(100);
        q.push(entry(5, 2, 1));
        q.push(entry(5, 1, 0));
        q.push(entry(4, 9, 0));
        q.order();
        assert_eq!(q.entries()[0].at_tick, 4);
        assert_eq!(q.entries()[1].priority, 1);
        assert_eq!(q.entries()[2].priority, 2);
    }

    #[test]
    fn order_is_stable_for_equal_keys() {
        let mut q = TickQueue::new(100);
        for sub in 0..5u8 {
            q.push(entry(1, 0, sub));
        }
        q.order();
        let subs: Vec<u8> = q.entries().iter().map(|e| e.sub_order).collect();
        assert_eq!(subs, vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn retain_due_drops_future_ticks() {
        let mut q = TickQueue::new(10);
        q.push(entry(5, 0, 0));
        q.push(entry(50, 0, 1));
        assert_eq!(q.due_count(), 1);
        q.retain_due();
        assert_eq!(q.len(), 1);
        assert_eq!(q.entries()[0].at_tick, 5);
    }

    #[test]
    fn retain_due_can_empty_the_queue() {
        let mut q = TickQueue::new(0);
        q.push(entry(5, 0, 0));
        q.retain_due();
        assert!(q.is_empty());
    }

    #[test]
    fn empty_queue_reports_no_work() {
        let q = TickQueue::new(0);
        assert!(q.is_empty());
        assert_eq!(q.due_count(), 0);
        assert_eq!(q.len(), 0);
    }
}
