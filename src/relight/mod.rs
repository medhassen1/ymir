//! Incremental relight.
//!
//! When only part of a region changed, relighting every column is wasteful. The
//! incremental pass rebuilds light for the dirty columns only.
//!
//! The pass keeps its packed level buffers in a region-lifetime [`SlabRing`]
//! rather than one allocation per column: built once for the whole incremental
//! pass, it recycles a fixed set of slabs round-robin, the way a double- (or
//! N-fold-) buffered renderer reuses a small pool of frames instead of
//! allocating a fresh one every time. A handful of columns, spaced across the
//! dirty set, keep their slab's span registered in a deferred fold list so a
//! later column's anneal can measure against a neighbour beyond just its
//! immediate predecessor — see [`RETAIN_STRIDE`] in [`incremental`]. Because
//! the ring is a fixed size, a dirty set larger than it wraps back to the
//! first slab and overwrites it, the way a ring buffer always has, no matter
//! how many columns still hold that slab's span in the deferred list.

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

/// Slabs held by the ring at once.
///
/// An ordinary incremental relight touches only a few dirty columns, well
/// under this, so an ordinary pass never wraps. A region marking many columns
/// dirty at once does, which is exactly the case the ring exists to bound:
/// without a fixed size, a region-lifetime pool of level slabs would grow
/// without bound on a heavily-dirtied region.
const RING_SIZE: usize = 8;

/// Spacing between dirty columns that register a retained slab span.
///
/// The first column has no earlier neighbour to anneal against, so retention
/// starts at the first multiple of the stride past it.
const RETAIN_STRIDE: usize = 3;

/// A ring of reusable level slabs, region-lifetime across the whole
/// incremental pass.
///
/// The first [`RING_SIZE`] columns each claim a fresh slot; every commit after
/// that overwrites the next slot in ring order, dropping whatever slab was
/// resident there and replacing it with a freshly allocated, exact-sized one —
/// the way a fixed-size ring buffer always recycles its oldest entry rather
/// than growing.
struct SlabRing {
    slots: Vec<Box<[u8]>>,
    next: usize,
}

impl SlabRing {
    fn new() -> SlabRing {
        SlabRing { slots: Vec::new(), next: 0 }
    }

    /// Commit `levels` into the ring and return a pointer to where they
    /// landed.
    ///
    /// The returned pointer addresses `levels.len()` live bytes for as long as
    /// the slot backing it has not yet been recycled for a later column (see
    /// [`SlabRing::commit`]'s wrap behaviour above).
    fn commit(&mut self, levels: &[u8]) -> *const u8 {
        let slab: Box<[u8]> = levels.to_vec().into_boxed_slice();
        if self.slots.len() < RING_SIZE {
            self.slots.push(slab);
            self.next = self.slots.len() % RING_SIZE;
            // SAFETY: the slab was just pushed and is exactly `levels.len()`
            // bytes long.
            self.slots.last().unwrap().as_ptr()
        } else {
            let idx = self.next;
            self.slots[idx] = slab;
            self.next = (idx + 1) % RING_SIZE;
            // SAFETY: the slab was just written into `slots[idx]` and is
            // exactly `levels.len()` bytes long.
            self.slots[idx].as_ptr()
        }
    }
}

/// A column's slab span retained past its own turn through [`incremental`]'s
/// main loop, for the closing anneal fold to read once the whole dirty set has
/// been walked.
struct RetainedSlab {
    ptr: *const u8,
    count: usize,
}

/// Relight the region's dirty columns and fold a digest of the result.
///
/// Each column's packed levels are committed into a region-wide [`SlabRing`]
/// rather than freed with the column: a spaced-out subset of columns keep
/// their slab's span registered in a retained list for the rest of the pass,
/// giving a later column's anneal a cross-region neighbour beyond just its
/// immediate predecessor. The retained spans are folded once more at the end,
/// closing out the pass.
pub fn incremental(region: &Region, n: usize) -> u64 {
    let dirty = dirty_columns(region, n);
    if dirty.is_empty() {
        return 0;
    }

    let mut ring = SlabRing::new();
    let mut retained: Vec<RetainedSlab> = Vec::new();
    let mut acc = 0xffu64;

    for (i, col) in dirty.iter().enumerate() {
        let levels = column_levels(col);
        let count = levels.len();
        let ptr = ring.commit(&levels);
        acc = acc.wrapping_mul(0x100000001b3) ^ anneal::fold_one(ptr, count);

        // Every `RETAIN_STRIDE`-th column past the first keeps its slab's span
        // alive as a neighbour for a later column's anneal to read, giving the
        // pass continuity beyond just each column's immediate predecessor.
        if i > 0 && i % RETAIN_STRIDE == 0 {
            retained.push(RetainedSlab { ptr, count });
        }
    }

    // Close out the pass by folding in every retained slab once, now that the
    // whole dirty set has gone through the ring.
    for r in &retained {
        acc = acc.wrapping_mul(0x100000001b3) ^ anneal::fold_one(r.ptr, r.count);
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
    fn ring_commit_writes_are_readable_back() {
        let mut ring = SlabRing::new();
        let a = ring.commit(&[1, 2, 3]);
        let b = ring.commit(&[4, 5]);
        // SAFETY: the ring has not wrapped (only two commits, well under
        // `RING_SIZE`), so neither slot has been overwritten.
        unsafe {
            assert_eq!(std::slice::from_raw_parts(a, 3), [1, 2, 3]);
            assert_eq!(std::slice::from_raw_parts(b, 2), [4, 5]);
        }
    }

    #[test]
    fn ring_reuses_slots_only_after_wrapping() {
        let mut ring = SlabRing::new();
        for _ in 0..RING_SIZE {
            ring.commit(&[9]);
        }
        assert_eq!(ring.slots.len(), RING_SIZE, "the ring must fill before it wraps");
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
