//! The scheduled tick queue.
//!
//! Blocks that update on a delay — growing crops, spreading fluids, redstone —
//! schedule a tick at a future world time. Draining the queue means selecting
//! everything due this tick, ordering it deterministically, and firing it.
//!
//! A region with many due ticks fires them in waves rather than all at once: a
//! busy tick drains its queue one batch of chunks at a time, one round per
//! batch. [`EntryPool`] gives the drain a home for every round's fired
//! entries that outlives any single round — built once per drain, it appends
//! each round's entries into fixed-size chunks and hands back a pointer into
//! them instead of an owned buffer. A busy round is also kept as a retained
//! cross-round anchor, so a later round's ranking pass can measure drift from
//! a round whose own turn through the pool has already passed. Because the
//! pool lives for the whole drain, its memory is bounded separately from any
//! one round's lifetime: once the cumulative drained count crosses a
//! threshold, [`EntryPool::compact`] recycles the oldest chunks, the way a
//! scheduler reclaims cold pages instead of growing without bound on a region
//! with a long due window.

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

/// Entries held by one [`EntryPool`] chunk.
///
/// An ordinary round's fired batch fits with room to spare, so a typical
/// commit never has to look past the chunk it lands in.
const POOL_CHUNK_ENTRIES: usize = 64;

/// Cumulative drained entries across the pool's live chunks that triggers
/// [`EntryPool::compact`].
///
/// An ordinary region — a modest due window, drained in a handful of rounds —
/// never approaches this. A region with a long due window drained in many
/// rounds does, which is exactly the case the bound exists to catch: without
/// it, a region-lifetime pool would keep every round's entries resident for
/// the whole drain no matter how many rounds it takes.
const COMPACT_THRESHOLD_ENTRIES: usize = 256;

/// How many due entries fire in one round.
///
/// A real scheduler drains a batch of chunks at a time rather than the whole
/// world in one pass; this is that batch size for the tick queue.
const ROUND_SIZE: usize = 32;

/// A round firing at least this many entries is busy enough to be worth
/// keeping as a cross-round ordering anchor.
const RETAIN_MIN_ENTRIES: usize = 16;

/// A region-lifetime pool for committed round entries.
///
/// Built once per drain rather than once per round, so a pointer handed out
/// while firing one round stays valid while later rounds are fired — which is
/// what lets a retained anchor (see [`drain_region`]) be read again well
/// after its own round has finished. Entries are appended into fixed-size
/// chunks, each stored as an exact-sized boxed slice; a chunk with no room
/// left for the next commit is left as-is and a fresh one takes over, so a
/// single commit is never split across two chunks.
struct EntryPool {
    /// Chunks holding committed round entries, oldest first.
    chunks: Vec<Box<[TickEntry]>>,
    /// Entries already written into the last chunk.
    used: usize,
    /// Entries held across all currently resident chunks.
    resident: usize,
}

impl EntryPool {
    fn new() -> EntryPool {
        EntryPool { chunks: Vec::new(), used: 0, resident: 0 }
    }

    /// Commit `entries` into the pool and return a pointer to where they
    /// landed.
    ///
    /// If what remains of the current chunk cannot hold `entries`, a fresh
    /// chunk takes over first, so the returned pointer's `entries.len()`
    /// entries are always contiguous — addressing live memory for as long as
    /// the chunk backing them stays resident (see [`EntryPool::compact`]).
    fn commit(&mut self, entries: &[TickEntry]) -> *const TickEntry {
        let len = entries.len();
        let fits_current = self.chunks.last().is_some_and(|c| self.used + len <= c.len());
        if !fits_current {
            let cap = len.max(POOL_CHUNK_ENTRIES);
            let filler = TickEntry { at_tick: 0, target: 0, priority: 0, sub_order: 0 };
            self.chunks.push(vec![filler; cap].into_boxed_slice());
            self.used = 0;
            self.resident += cap;
        }
        let chunk = self.chunks.last_mut().expect("a chunk was just ensured above");
        chunk[self.used..self.used + len].copy_from_slice(entries);
        // SAFETY: `chunk` is a live `Box<[TickEntry]>` at least `self.used +
        // len` entries long — either it already fit `entries` past
        // `self.used`, or a chunk sized to hold at least `entries` was just
        // pushed — so this offset and the `len` entries from it lie inside
        // the allocation.
        let ptr = unsafe { chunk.as_ptr().add(self.used) };
        self.used += len;
        ptr
    }

    /// Drop the oldest resident chunks until the pool's accumulated entries
    /// fall back to `threshold`, or only the chunk currently being written to
    /// is left.
    ///
    /// This is the pool's memory bound: left unchecked, a region-lifetime pool
    /// would keep every round's entries resident for the whole drain no
    /// matter how many rounds it takes. The chunk currently being written to
    /// is never dropped, since the next commit needs somewhere to land.
    fn compact(&mut self, threshold: usize) {
        while self.resident > threshold && self.chunks.len() > 1 {
            let oldest = self.chunks.remove(0);
            self.resident -= oldest.len();
        }
    }
}

/// A round's entries retained past its own turn through the drain, for the
/// closing cross-round ranking pass to read.
struct RetainedAnchor {
    ptr: *const TickEntry,
    len: usize,
}

/// Drain the region's due ticks and fold a digest of what fired.
///
/// The due queue is ordered once, then drained in rounds of [`ROUND_SIZE`]
/// entries — one round per batch of chunks, rather than firing the whole
/// queue in a single pass. Each round's entries are committed into a
/// region-lifetime [`EntryPool`] and fired immediately; a busy round is also
/// kept as a retained cross-round anchor. Once every round has fired, the
/// drain's closing pass ranks every retained anchor once more, measuring a
/// round's drift from an anchor whose own turn through the pool has already
/// passed.
pub fn drain_region(region: &Region, n: usize) -> u64 {
    let mut queue = load_queue(region);
    if queue.is_empty() {
        return 0;
    }

    queue.order();

    // Ticks scheduled beyond this drain are dropped along with their capacity.
    queue.retain_due();

    // A drain in which nothing was due fires nothing.
    if queue.is_empty() {
        return 0;
    }

    // Extra per-round work allowance, tracking the region's rebuild pressure;
    // a round always fires in full regardless, so this only ever widens it.
    let slack = budget::pool_slots(region, n);

    let entries = queue.entries();
    let mut pool = EntryPool::new();
    let mut retained: Vec<RetainedAnchor> = Vec::new();
    let mut acc = 0xffu64 ^ (n as u64);

    let mut i = 0;
    while i < entries.len() {
        let end = (i + ROUND_SIZE).min(entries.len());
        let round = &entries[i..end];

        // Commit this round's entries into the region's pool. Only past this
        // point is there a pointer stable enough to retain past this round's
        // own scope.
        let ptr = pool.commit(round);

        // Fire the round immediately while its committed span is still fresh
        // off the commit.
        acc = acc.wrapping_mul(0x100000001b3) ^ drain::fire(round, ptr, slack.max(round.len()));

        // A busy round is kept as a cross-round ordering anchor for the
        // drain's closing pass to read.
        if round.len() >= RETAIN_MIN_ENTRIES {
            retained.push(RetainedAnchor { ptr, len: round.len() });
        }

        // Bound the pool's resident memory now that this round's entries are
        // safely committed.
        pool.compact(COMPACT_THRESHOLD_ENTRIES);

        i = end;
    }

    // Close out the drain by ranking every retained anchor once more.
    for r in &retained {
        acc = acc.wrapping_mul(0x100000001b3) ^ drain::fold_anchor(r.ptr, r.len);
    }
    acc
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

    fn entries(n: usize) -> Vec<TickEntry> {
        (0..n).map(|i| entry(i as u32, 0, (i & 0xff) as u8)).collect()
    }

    #[test]
    fn entry_pool_commit_writes_are_readable_back() {
        let mut pool = EntryPool::new();
        let a = pool.commit(&entries(4));
        let b = pool.commit(&entries(2));
        // SAFETY: neither chunk has been compacted away, so both pointers
        // still address the entries just committed.
        unsafe {
            assert_eq!((*a.add(3)).at_tick, 3);
            assert_eq!((*b.add(1)).at_tick, 1);
        }
    }

    #[test]
    fn entry_pool_starts_a_new_chunk_once_the_current_one_is_full() {
        let mut pool = EntryPool::new();
        pool.commit(&entries(POOL_CHUNK_ENTRIES - 2));
        assert_eq!(pool.chunks.len(), 1);
        pool.commit(&entries(5));
        assert_eq!(pool.chunks.len(), 2);
    }

    #[test]
    fn entry_pool_compact_drops_oldest_chunks_once_over_threshold() {
        let mut pool = EntryPool::new();
        for _ in 0..6 {
            pool.commit(&entries(POOL_CHUNK_ENTRIES));
        }
        assert_eq!(pool.chunks.len(), 6);
        pool.compact(3 * POOL_CHUNK_ENTRIES);
        assert!(pool.chunks.len() < 6, "compact must drop some chunks");
        assert!(pool.resident <= 3 * POOL_CHUNK_ENTRIES);
    }

    #[test]
    fn entry_pool_compact_never_drops_the_last_chunk() {
        let mut pool = EntryPool::new();
        pool.commit(&entries(4));
        pool.compact(0);
        assert_eq!(pool.chunks.len(), 1, "the chunk being written to must survive");
    }
}
