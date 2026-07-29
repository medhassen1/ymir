//! The scheduled tick queue.
//!
//! Blocks that update on a delay — growing crops, spreading fluids, redstone —
//! schedule a tick at a future world time. Draining the queue means selecting
//! everything due this tick, ordering it deterministically, and firing it.
//!
//! A region with a long due window fires its ticks in waves rather than all at
//! once: the drain cuts the ordered queue into rounds, one per due-time window
//! (see [`WINDOW_TICKS`]), and fires a round at a time. Rounds are staged
//! through [`RoundBuffers`], a pair of buffers the drain alternates between, so
//! the round being fired and the round before it are both addressable without
//! copying either out — which is what lets a cross-round anchor measure drift
//! from a round whose own turn has already passed. Alternating between two
//! buffers is also what bounds the drain's memory: a region with a hundred
//! rounds holds two of them, not a hundred.
//!
//! The anchors themselves live in a [`CarryRing`] of fixed capacity, so what
//! the closing ranking pass costs is bounded by the ring rather than by how
//! many rounds the drain took.

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

/// Entries one round fires, at most.
///
/// A real scheduler drains a bounded batch at a time rather than the whole
/// world in one pass; this is that batch size for the tick queue.
const ROUND_SIZE: usize = 32;

/// The due-time window one round covers.
///
/// The queue is ordered by due time, so ticks due within the same window are
/// contiguous and fire together. Cutting rounds on the window boundary rather
/// than purely on a count is what keeps a tick's wave-mates in its own round: a
/// region with a long due window drains in waves instead of in one pass.
const WINDOW_TICKS: u32 = 256;

/// How much larger than the round taking it a reused side may be.
///
/// Rounds within a factor of this of each other share a buffer, which is the
/// point of keeping two. A side holding very much more than the round needs is
/// re-taken instead, so a drain that opened on a wide due window does not carry
/// that buffer for every narrow round after it.
const SIDE_SLACK: usize = 4;

/// Cross-round anchors the carry ring keeps addressable at once.
///
/// Four is enough for the closing pass to rank against a handful of recent
/// rounds while keeping its cost — and the rounds it holds on to —
/// independent of how many rounds the drain took.
const CARRY_SLOTS: usize = 4;

/// The pair of buffers the drain alternates rounds between.
///
/// Staging a round takes the side the previous round did not, so the round
/// being fired and the round before it are both addressable. A side is
/// reallocated only when the incoming round does not fit the buffer already
/// there; a round that fits reuses it, which is the point of keeping two
/// buffers rather than allocating one per round.
struct RoundBuffers {
    sides: [Option<Box<[TickEntry]>>; 2],
    /// The side the next round stages into.
    flip: usize,
    /// How many buffers the pair has had to take.
    taken: usize,
}

impl RoundBuffers {
    fn new() -> RoundBuffers {
        RoundBuffers { sides: [None, None], flip: 0, taken: 0 }
    }

    /// Which side the next [`RoundBuffers::stage`] will write to.
    fn next_side(&self) -> usize {
        self.flip
    }

    /// Stage `round` into the next side and return a pointer to where it
    /// landed.
    ///
    /// The returned pointer addresses `round.len()` live entries for as long as
    /// the buffer holding them is still the one on that side.
    fn stage(&mut self, round: &[TickEntry]) -> *const TickEntry {
        let side = self.flip;
        self.flip ^= 1;
        let held = self.sides[side].as_ref().map_or(0, |b| b.len());
        // A side is reused when what is there fits the round without being so
        // much larger than it that the drain would carry the widest window it
        // ever saw for the rest of the pass.
        let reusable = held >= round.len() && held <= round.len() * SIDE_SLACK;
        if reusable {
            let buffer = self.sides[side].as_mut().expect("the side was just found reusable");
            buffer[..round.len()].copy_from_slice(round);
        } else {
            // Nothing on this side yet, or what is there is the wrong size for
            // the incoming round: the side takes a buffer sized to this round
            // and releases whatever it displaces.
            self.sides[side] = Some(round.to_vec().into_boxed_slice());
            self.taken += 1;
        }
        let buffer = self.sides[side].as_ref().expect("the side holds a buffer either way");
        buffer.as_ptr()
    }

    /// The first entry of whatever `side` currently holds, or null if that side
    /// has never been staged into.
    fn side_base(&self, side: usize) -> *const TickEntry {
        match self.sides.get(side).and_then(Option::as_ref) {
            Some(buffer) => buffer.as_ptr(),
            None => std::ptr::null(),
        }
    }
}

/// A round staged into [`RoundBuffers`] and carried past its own turn, for the
/// drain's closing cross-round ranking pass to read.
///
/// The closing pass walks the ring exactly once, so it takes the address the
/// buffers gave out rather than going back through them per round.
struct CarriedRound {
    ptr: *const TickEntry,
    len: usize,
}

/// A round the drain measures a later round's drift against, named by the side
/// it was staged on: the drain already knows which side it wrote, so carrying
/// that costs nothing over carrying the address it was given.
#[derive(Clone, Copy)]
struct StagedRound {
    side: usize,
    len: usize,
}

/// A fixed-capacity ring of cross-round anchors.
///
/// Carrying a round takes the next slot in round-robin order, displacing
/// whatever was carried [`CARRY_SLOTS`] rounds ago. Bounding the ring is what
/// keeps the drain's cross-round state — and the closing pass's cost — flat
/// over a region with a long due window, instead of growing one entry per round
/// the way a plain list would.
struct CarryRing {
    slots: [Option<CarriedRound>; CARRY_SLOTS],
    next: usize,
}

impl CarryRing {
    fn new() -> CarryRing {
        CarryRing { slots: Default::default(), next: 0 }
    }

    /// Carry `round` as the newest cross-round anchor.
    fn carry(&mut self, round: CarriedRound) {
        self.slots[self.next] = Some(round);
        self.next = (self.next + 1) % CARRY_SLOTS;
    }

    /// The anchors the ring currently holds, in slot order.
    fn rounds(&self) -> impl Iterator<Item = &CarriedRound> {
        self.slots.iter().flatten()
    }
}

/// Fill, in entries, at which the drain considers a round well used.
///
/// A round around this size is the scheduler keeping pace: enough work to be
/// worth the round's overhead, not so much that it is running late.
const CADENCE_TARGET: usize = ROUND_SIZE / 2;

/// How far a round's fill may drift from [`CADENCE_TARGET`], accumulated over
/// the rounds fired so far, before the drain changes its window width.
const CADENCE_STEP: i32 = ROUND_SIZE as i32;

/// The due-time window the drain is currently cutting rounds on.
///
/// A fixed window suits a queue whose work is spread evenly and nothing else. A
/// drain that keeps firing near-empty rounds is running ahead of its work and
/// paying a round's overhead for a handful of ticks, so it widens the window to
/// gather more; one that keeps hitting the entry cap is running late and
/// narrows it to stay responsive. The width therefore depends on every round
/// fired before it, not on the round about to be cut.
struct Cadence {
    width: u32,
    drift: i32,
}

impl Cadence {
    fn new() -> Cadence {
        Cadence { width: WINDOW_TICKS, drift: 0 }
    }

    /// Fold a fired round's fill into the cadence.
    fn settle(&mut self, fired: usize) {
        self.drift += CADENCE_TARGET as i32 - fired as i32;
        if self.drift > CADENCE_STEP {
            self.width = (self.width * 2).min(WINDOW_TICKS * 16);
            self.drift = 0;
        } else if self.drift < -CADENCE_STEP {
            self.width = (self.width / 2).max(1);
            self.drift = 0;
        }
    }
}

/// How far the round starting at `entries[i]` reaches.
///
/// A round runs to the end of the due-time window its first entry falls in, or
/// to [`ROUND_SIZE`] entries, whichever comes first. `width` is the window the
/// drain's [`Cadence`] has settled on. The queue is ordered by due time before
/// this is called, so a window's entries are contiguous.
fn round_end(entries: &[TickEntry], i: usize, width: u32) -> usize {
    let width = width.max(1);
    let window = entries[i].at_tick / width;
    let cap = (i + ROUND_SIZE).min(entries.len());
    let mut end = i + 1;
    while end < cap && entries[end].at_tick / width == window {
        end += 1;
    }
    end
}

/// Drain the region's due ticks and fold a digest of what fired.
///
/// The due queue is ordered once, then cut into rounds — one per due-time
/// window, capped at [`ROUND_SIZE`] entries — rather than fired in a single
/// pass. How wide that window is is not fixed: the drain's [`Cadence`] widens
/// it while rounds keep coming up thin and narrows it while they keep hitting
/// the cap, so where a round ends depends on every round fired before it. Each round is staged into the drain's [`RoundBuffers`] and fired
/// immediately, and carried in the [`CarryRing`] as a cross-round anchor. Once
/// every round has fired, the drain's closing pass ranks the anchors the ring
/// still holds, measuring a round's drift from one whose own turn has already
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
    let mut buffers = RoundBuffers::new();
    let mut carried = CarryRing::new();
    // The two rounds behind the one being fired, so drift can be measured
    // against the round a window further back than the immediate predecessor.
    let mut one_back: Option<StagedRound> = None;
    let mut two_back: Option<StagedRound> = None;
    let mut acc = 0xffu64 ^ (n as u64);

    let mut cadence = Cadence::new();
    let mut i = 0;
    while i < entries.len() {
        let end = round_end(entries, i, cadence.width);
        let round = &entries[i..end];
        cadence.settle(round.len());

        // Stage this round into the drain's buffers. Only past this point is
        // there a pointer stable enough to carry past this round's own turn.
        let side = buffers.next_side();
        let ptr = buffers.stage(round);

        // Fire the round immediately while its staged span is still fresh off
        // the flip.
        acc = acc.wrapping_mul(0x100000001b3) ^ drain::fire(round, ptr, slack.max(round.len()));

        // Measure this round's drift from the round two windows back, resolved
        // against whatever that side holds now.
        if let Some(p) = two_back.as_ref() {
            acc = acc.wrapping_mul(0x9e3779b97f4a7c15)
                ^ drain::fold_drift(buffers.side_base(p.side), p.len, ptr, round.len());
        }
        two_back = one_back.take();
        one_back = Some(StagedRound { side, len: round.len() });

        // Carry the round as the newest cross-round anchor.
        carried.carry(CarriedRound { ptr, len: round.len() });

        i = end;
    }

    // Close out the drain by ranking every anchor the ring holds.
    for r in carried.rounds() {
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
    fn a_round_stops_at_the_window_boundary() {
        let e = vec![entry(0, 0, 0), entry(1, 0, 1), entry(WINDOW_TICKS, 0, 2)];
        assert_eq!(
            round_end(&e, 0, WINDOW_TICKS),
            2,
            "the third entry is in the next window"
        );
        assert_eq!(round_end(&e, 2, WINDOW_TICKS), 3);
    }

    #[test]
    fn a_round_stops_at_the_batch_size() {
        let e = entries(ROUND_SIZE + 8);
        assert_eq!(round_end(&e, 0, WINDOW_TICKS), ROUND_SIZE);
    }

    #[test]
    fn round_buffers_stage_writes_are_readable_back() {
        let mut buffers = RoundBuffers::new();
        let first = entries(4);
        let second = entries(2);
        let a = buffers.stage(&first);
        let b = buffers.stage(&second);
        // SAFETY: the two rounds staged into opposite sides, so neither buffer
        // has been displaced and both pointers still address what was staged.
        unsafe {
            assert_eq!((*a.add(3)).at_tick, 3);
            assert_eq!((*b.add(1)).at_tick, 1);
        }
    }

    /// Keeping two buffers only pays if rounds reuse them. A drain of many
    /// rounds that all fit must take two buffers and no more, however long the
    /// due queue is.
    #[test]
    fn round_buffers_stop_allocating_once_both_sides_are_wide_enough() {
        let mut buffers = RoundBuffers::new();
        let wide = entries(ROUND_SIZE);
        buffers.stage(&wide);
        buffers.stage(&wide);
        assert_eq!(buffers.taken, 2, "each side takes one buffer to start");
        // Rounds within the slack band of what the sides hold reuse them.
        for len in (ROUND_SIZE / SIDE_SLACK)..=ROUND_SIZE {
            buffers.stage(&entries(len));
        }
        assert_eq!(
            buffers.taken, 2,
            "rounds inside the slack band must reuse the pair, not replace it"
        );
    }

    /// The other half of the same contract: a side holding very much more than
    /// the round needs is re-taken, so a drain that opened on a wide window
    /// does not carry that buffer through every narrow round after it.
    #[test]
    fn a_side_far_wider_than_its_round_is_re_taken() {
        let mut buffers = RoundBuffers::new();
        let wide = entries(ROUND_SIZE);
        buffers.stage(&wide);
        buffers.stage(&wide);
        let before = buffers.taken;
        buffers.stage(&entries(1));
        assert_eq!(buffers.taken, before + 1, "a far-oversized side must be re-taken");
        assert_eq!(buffers.sides[0].as_ref().expect("staged").len(), 1);
    }

    #[test]
    fn round_buffers_alternate_sides() {
        let mut buffers = RoundBuffers::new();
        let round = entries(4);
        let a = buffers.stage(&round);
        let b = buffers.stage(&round);
        assert_ne!(a, b, "consecutive rounds must land on different sides");
        let c = buffers.stage(&round);
        assert_eq!(a, c, "the third round returns to the first side");
    }

    #[test]
    fn a_round_that_fits_reuses_the_buffer_already_there() {
        let mut buffers = RoundBuffers::new();
        let wide = entries(8);
        let narrow = entries(3);
        let a = buffers.stage(&wide);
        buffers.stage(&wide);
        let c = buffers.stage(&narrow);
        assert_eq!(a, c, "a shorter round must not reallocate the side");
        assert_eq!(
            buffers.sides[0].as_ref().expect("the side holds a buffer").len(),
            8,
            "the buffer keeps the size it was allocated at"
        );
    }
}
