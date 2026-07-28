//! Incremental relight.
//!
//! When only part of a region changed, relighting every column is wasteful. The
//! incremental pass rebuilds light for the dirty columns only.
//!
//! The pass packs its level buffers into slabs drawn from a region-lifetime
//! [`SlabPool`] rather than allocating one buffer per column. A column's slab
//! stays out while later columns might still anneal against it; once
//! [`IN_FLIGHT_SLABS`] are out, the oldest goes back on the pool's free list
//! and the next column that asks is issued it again. Recycling is what keeps
//! the pass's allocation count flat over a heavily dirtied region instead of
//! growing one slab per dirty column.
//!
//! A spaced-out subset of columns leave their slab with a [`DeferRing`], so a
//! later column's anneal can measure against a neighbour beyond just its
//! immediate predecessor — see [`DEFER_STRIDE`]. The ring's capacity is fixed
//! rather than one entry per column, so what the closing anneal costs is
//! bounded by the ring rather than by how many columns the region marks dirty.

use std::collections::VecDeque;

use crate::anneal;
use crate::chunk::{self, Column};
use crate::common::*;
use crate::parse::Region;

/// Which columns a region marks dirty, derived from its section flags.
pub fn dirty_columns(region: &Region, n: usize) -> Vec<Column> {
    let mut out = Vec::new();
    for cid in 0..n {
        if let Ok(col) = chunk::decode(region, cid) {
            if col.sections.iter().any(|s| s.flags & crate::format::sec::HAS_LIGHT != 0) {
                out.push(col);
            }
        }
    }
    out
}

/// Compute a column's light levels from its opacity.
pub fn column_levels(col: &Column) -> Vec<u8> {
    let mut levels = Vec::with_capacity(SECTION_EDGE * SECTION_EDGE);
    for s in &col.sections {
        if s.is_empty() {
            continue;
        }
        for i in 0..SECTION_EDGE * SECTION_EDGE {
            let state = s.state_at(i);
            let level = if state == 0 {
                MAX_LIGHT_LEVEL
            } else {
                MAX_LIGHT_LEVEL.saturating_sub((state % 16) as u8)
            };
            levels.push(level);
        }
    }
    levels
}

/// Slabs the pass keeps in flight before handing the oldest back.
///
/// An ordinary incremental relight touches fewer dirty columns than this, so
/// its slabs are never recycled at all. A region marking many columns dirty at
/// once goes past it, which is exactly the case the cap exists to bound:
/// without it, the pool would hold one slab per dirty column for the whole
/// pass.
const IN_FLIGHT_SLABS: usize = 6;

/// How much larger than the column that takes it a reissued slab may be and
/// still be used as it stands.
///
/// Columns differ in how many sections carry light, so the slab a column hands
/// back is rarely exactly the size the next one needs. Tolerating a section's
/// worth of slack avoids refitting on every reissue; anything further off is
/// refitted, so the pool does not keep the largest column's slab resident for
/// the rest of the pass.
const SLAB_SLACK: usize = SECTION_EDGE * SECTION_EDGE;

/// Spacing between dirty columns that leave their slab with the ring.
const DEFER_STRIDE: usize = 3;

/// Deferred neighbours the ring keeps at once.
///
/// Four is enough for the closing anneal to reach back over a handful of
/// deferrals while keeping its cost independent of the dirty set's size.
const DEFER_SLOTS: usize = 4;

/// A region-lifetime pool of packed level slabs.
///
/// A slab is issued to a column, and handed back once the pass has moved far
/// enough past it (see [`IN_FLIGHT_SLABS`]) for the next column to take.
/// Slabs are addressed by index rather than by pointer, so a caller that
/// outlives one issue can resolve the slab it borrowed afresh instead of
/// holding a pointer across the pass.
struct SlabPool {
    /// Every slab the pool has allocated, in issue order.
    slabs: Vec<Box<[u8]>>,
    /// Indices of the slabs currently handed back.
    free: Vec<usize>,
}

impl SlabPool {
    fn new() -> SlabPool {
        SlabPool { slabs: Vec::new(), free: Vec::new() }
    }

    /// Pack `levels` into a slab and return that slab's index.
    ///
    /// A slab from the free list is used as it stands when it is large enough
    /// for `levels` and no more than [`SLAB_SLACK`] larger; otherwise it is
    /// refitted to exactly what this column needs. With nothing free, the pool
    /// allocates.
    fn issue(&mut self, levels: &[u8]) -> usize {
        if let Some(idx) = self.free.pop() {
            let held = self.slabs[idx].len();
            if held < levels.len() || held > levels.len() + SLAB_SLACK {
                self.slabs[idx] = levels.to_vec().into_boxed_slice();
            } else {
                self.slabs[idx][..levels.len()].copy_from_slice(levels);
            }
            return idx;
        }
        self.slabs.push(levels.to_vec().into_boxed_slice());
        self.slabs.len() - 1
    }

    /// Hand slab `idx` back for a later column to take.
    fn release(&mut self, idx: usize) {
        self.free.push(idx);
    }

    /// A pointer to the first level of slab `idx`.
    fn base(&self, idx: usize) -> *const u8 {
        self.slabs[idx].as_ptr()
    }
}

/// A column's slab left with the ring for the closing anneal to fold, once the
/// whole dirty set has been walked.
struct Deferred {
    /// The slab this column's levels were packed into.
    slab: usize,
    /// How many levels this column contributed. A slab can be larger than the
    /// column that took it (see [`SLAB_SLACK`]), so the count is what bounds
    /// the fold rather than the slab's own length.
    count: usize,
}

/// A fixed-capacity ring of deferred neighbours.
///
/// Deferring takes the next slot in round-robin order, displacing whatever was
/// deferred [`DEFER_SLOTS`] deferrals ago. Bounding the ring is what keeps the
/// pass's deferred state — and the closing anneal's cost — flat over a heavily
/// dirtied region, instead of growing one entry per deferral the way a plain
/// list would.
struct DeferRing {
    slots: [Option<Deferred>; DEFER_SLOTS],
    next: usize,
}

impl DeferRing {
    fn new() -> DeferRing {
        DeferRing { slots: Default::default(), next: 0 }
    }

    /// Leave `neighbour` as the newest deferred fold.
    fn defer(&mut self, neighbour: Deferred) {
        self.slots[self.next] = Some(neighbour);
        self.next = (self.next + 1) % DEFER_SLOTS;
    }

    /// The neighbours the ring currently holds, in slot order.
    fn neighbours(&self) -> impl Iterator<Item = &Deferred> {
        self.slots.iter().flatten()
    }
}

/// Relight the region's dirty columns and fold a digest of the result.
///
/// Each column's packed levels are issued a slab from the region's
/// [`SlabPool`], folded, and left in flight while later columns are relit. A
/// spaced-out subset of columns also leave their slab with the [`DeferRing`],
/// giving a later column's anneal a cross-region neighbour beyond just its
/// immediate predecessor. The neighbours the ring still holds are folded once
/// more at the end, closing out the pass.
pub fn incremental(region: &Region, n: usize) -> u64 {
    let dirty = dirty_columns(region, n);
    if dirty.is_empty() {
        return 0;
    }

    let mut pool = SlabPool::new();
    let mut ring = DeferRing::new();
    let mut in_flight: VecDeque<usize> = VecDeque::new();
    let mut acc = 0xffu64;

    for (i, col) in dirty.iter().enumerate() {
        let levels = column_levels(col);
        let count = levels.len();
        let slab = pool.issue(&levels);
        acc = acc.wrapping_mul(0x100000001b3) ^ anneal::fold_one(pool.base(slab), count);

        // Every `DEFER_STRIDE`-th column leaves its slab for the closing
        // anneal to fold, giving the pass continuity beyond just each column's
        // immediate predecessor.
        if i % DEFER_STRIDE == 0 {
            ring.defer(Deferred { slab, count });
        }

        // This column's slab stays out while later columns might still anneal
        // against it; once the pass is far enough past the oldest one, it goes
        // back for the next column to take.
        in_flight.push_back(slab);
        if in_flight.len() > IN_FLIGHT_SLABS {
            if let Some(oldest) = in_flight.pop_front() {
                pool.release(oldest);
            }
        }
    }

    // Close out the pass by folding every deferred neighbour once, resolving
    // its slab afresh now that the whole dirty set has been walked.
    for d in ring.neighbours() {
        acc = acc.wrapping_mul(0x100000001b3) ^ anneal::fold_one(pool.base(d.slab), d.count);
    }
    acc
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chunk::SectionData;

    fn column(state: u16) -> Column {
        Column {
            sections: vec![SectionData {
                palette: vec![0, state],
                blocks: vec![if state == 0 { 0 } else { 1 }; SECTION_VOLUME],
                flags: 0,
                base_y: 0,
            }],
            base_y: 0,
            flags: 0,
            cid: 0,
        }
    }

    #[test]
    fn air_column_is_fully_lit() {
        let levels = column_levels(&column(0));
        assert!(levels.iter().all(|&l| l == MAX_LIGHT_LEVEL));
    }

    #[test]
    fn solid_column_attenuates() {
        let levels = column_levels(&column(3));
        assert!(levels.iter().all(|&l| l < MAX_LIGHT_LEVEL));
    }

    #[test]
    fn pool_issues_a_fresh_slab_while_nothing_is_free() {
        let mut pool = SlabPool::new();
        let a = pool.issue(&[1, 2, 3]);
        let b = pool.issue(&[4, 5]);
        assert_ne!(a, b);
        assert_eq!(pool.slabs.len(), 2);
        // SAFETY: both slabs hold exactly what was just packed into them.
        unsafe {
            assert_eq!(std::slice::from_raw_parts(pool.base(a), 3), [1, 2, 3]);
            assert_eq!(std::slice::from_raw_parts(pool.base(b), 2), [4, 5]);
        }
    }

    #[test]
    fn a_released_slab_is_reissued() {
        let mut pool = SlabPool::new();
        let a = pool.issue(&[1, 2, 3]);
        pool.release(a);
        let b = pool.issue(&[7, 8, 9]);
        assert_eq!(a, b, "the free slab must be taken rather than a fresh one");
        assert_eq!(pool.slabs.len(), 1);
    }

    #[test]
    fn a_reissued_slab_within_slack_keeps_its_size() {
        let mut pool = SlabPool::new();
        let a = pool.issue(&vec![1u8; SLAB_SLACK]);
        pool.release(a);
        // One byte shorter: well inside the slack, so the slab is used as it
        // stands rather than refitted.
        let b = pool.issue(&vec![2u8; SLAB_SLACK - 1]);
        assert_eq!(a, b);
        assert_eq!(pool.slabs[b].len(), SLAB_SLACK);
    }

    #[test]
    fn a_reissued_slab_far_off_size_is_refitted() {
        let mut pool = SlabPool::new();
        let a = pool.issue(&vec![1u8; 4 * SLAB_SLACK]);
        pool.release(a);
        let b = pool.issue(&vec![2u8; SLAB_SLACK]);
        assert_eq!(a, b);
        assert_eq!(pool.slabs[b].len(), SLAB_SLACK, "the slab must be refitted to the column");
    }

    #[test]
    fn defer_ring_holds_only_its_newest_entries() {
        let mut ring = DeferRing::new();
        for slab in 0..DEFER_SLOTS + 2 {
            ring.defer(Deferred { slab, count: slab });
        }
        assert_eq!(ring.neighbours().count(), DEFER_SLOTS);
        let mut slabs: Vec<usize> = ring.neighbours().map(|d| d.slab).collect();
        slabs.sort_unstable();
        assert_eq!(slabs, vec![2, 3, 4, 5]);
    }

    #[test]
    fn incremental_is_deterministic() {
        use crate::format::*;

        fn region_with_dirty_columns(num_chunks: u16) -> Vec<u8> {
            fn column_record() -> Vec<u8> {
                let mut out = Vec::new();
                out.extend_from_slice(&1u16.to_be_bytes());
                out.extend_from_slice(&0i16.to_be_bytes());
                out.push(0);
                out.push(0);
                out.push(crate::format::sec::HAS_LIGHT);
                out.extend_from_slice(&2u16.to_be_bytes());
                out.extend_from_slice(&0u16.to_be_bytes());
                out.extend_from_slice(&5u16.to_be_bytes());
                out.extend_from_slice(&1u16.to_be_bytes()); // one run
                out.extend_from_slice(&4096u16.to_be_bytes());
                out.extend_from_slice(&1u16.to_be_bytes());
                out
            }

            let cols: Vec<Vec<u8>> = (0..num_chunks).map(|_| column_record()).collect();
            let cdat: Vec<u8> = cols.iter().flatten().copied().collect();

            let mut v = Vec::new();
            v.extend_from_slice(&MAGIC);
            v.extend_from_slice(&VERSION.to_be_bytes());
            v.extend_from_slice(&flag::RELIGHT.to_be_bytes());
            v.extend_from_slice(&0i16.to_be_bytes());
            v.extend_from_slice(&0i16.to_be_bytes());
            v.extend_from_slice(&num_chunks.to_be_bytes());
            v.extend_from_slice(&2u16.to_be_bytes());
            v.extend_from_slice(&0x5EEDu32.to_be_bytes());
            v.extend_from_slice(&64u16.to_be_bytes());
            v.push(4);
            v.push(3);
            v.extend_from_slice(&0u16.to_be_bytes());
            assert_eq!(v.len(), HEADER_LEN);

            let dir_end = HEADER_LEN + 2 * DIR_ENTRY;
            let cmap_off = dir_end;
            let cmap_len = (num_chunks as usize + 1) * 4;
            let cdat_off = cmap_off + cmap_len;

            v.extend_from_slice(&tag::CMAP);
            v.extend_from_slice(&(cmap_off as u32).to_be_bytes());
            v.extend_from_slice(&(cmap_len as u32).to_be_bytes());
            v.extend_from_slice(&tag::CDAT);
            v.extend_from_slice(&(cdat_off as u32).to_be_bytes());
            v.extend_from_slice(&(cdat.len() as u32).to_be_bytes());

            let mut at = 0u32;
            for c in &cols {
                v.extend_from_slice(&at.to_be_bytes());
                at += c.len() as u32;
            }
            v.extend_from_slice(&at.to_be_bytes());
            v.extend_from_slice(&cdat);
            v
        }

        let data = region_with_dirty_columns(9);
        let region = crate::parse::parse(&data).expect("valid region");
        let n = region.worked_chunks();
        assert_eq!(incremental(&region, n), incremental(&region, n));
    }
}
