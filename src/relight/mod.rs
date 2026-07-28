//! Incremental relight.
//!
//! When only part of a region changed, relighting every column is wasteful. The
//! incremental pass rebuilds light for the dirty columns only, and merges
//! adjacent columns whose light footprints match — a wall of identical blocks
//! lights identically, so the second column reuses the first's buffer instead of
//! recomputing and re-allocating it.

use std::alloc::{alloc, dealloc, Layout};

use crate::anneal;
use crate::chunk::{self, Column};
use crate::common::*;
use crate::parse::Region;

/// A hand-managed buffer of per-block light levels for one column.
///
/// Exactly one handle owns the allocation and releases it on drop; the rest are
/// views. A merge transfers ownership rather than copying, which is the whole
/// point of merging in the first place.
pub struct ColumnBuf {
    ptr: *mut u8,
    len: usize,
    owner: bool,
}

impl ColumnBuf {
    /// Allocate and fill a buffer from `levels`.
    pub fn pack(levels: &[u8]) -> ColumnBuf {
        let len = levels.len().max(1);
        let layout = Layout::array::<u8>(len).unwrap();
        // SAFETY: `len >= 1`, so the layout is non-zero-sized; the owning handle
        // releases it in `Drop`.
        let ptr = unsafe { alloc(layout) };
        if ptr.is_null() {
            return ColumnBuf { ptr: std::ptr::null_mut(), len: 0, owner: false };
        }
        for (i, &l) in levels.iter().enumerate() {
            // SAFETY: `i < levels.len() <= len`, the allocated byte count.
            unsafe { *ptr.add(i) = l };
        }
        ColumnBuf { ptr, len, owner: true }
    }

    /// A non-owning view of the same buffer.
    pub fn view(&self) -> ColumnBuf {
        ColumnBuf { ptr: self.ptr, len: self.len, owner: false }
    }

    /// Take over this buffer's storage for a merged column, so the merged
    /// column keeps the source allocation instead of re-packing it.
    ///
    /// SAFETY: the caller drops the original handle without freeing, leaving
    /// this one as the sole owner.
    pub fn adopt(&self) -> ColumnBuf {
        ColumnBuf { ptr: self.ptr, len: self.len, owner: true }
    }

    /// The packed light levels.
    pub fn levels(&self) -> &[u8] {
        if self.ptr.is_null() {
            return &[];
        }
        // SAFETY: `ptr`/`len` describe a live buffer held by an owning handle.
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }

    /// The number of levels held.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether the buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// A cheap summary used to decide whether two columns may merge.
    pub fn footprint(&self) -> u32 {
        let mut h = self.len as u32;
        for &l in self.levels() {
            h = h.rotate_left(3) ^ (l as u32);
        }
        h
    }
}

impl Drop for ColumnBuf {
    fn drop(&mut self) {
        if self.owner && !self.ptr.is_null() {
            let layout = Layout::array::<u8>(self.len).unwrap();
            // SAFETY: the owner frees exactly the allocation `pack` made, under
            // the layout it was made with.
            unsafe { dealloc(self.ptr, layout) }
        }
    }
}

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

/// Relight the region's dirty columns, merging matching neighbours, and fold a
/// digest of the result.
///
/// Adjacent columns whose light footprints agree are merged: the merged column
/// adopts the earlier buffer's storage rather than packing its own.
pub fn incremental(region: &Region, n: usize) -> u64 {
    let dirty = dirty_columns(region, n);
    if dirty.is_empty() {
        return 0;
    }

    let run: Vec<ColumnBuf> = dirty.iter().map(|c| ColumnBuf::pack(&column_levels(c))).collect();

    let mut merged: Vec<ColumnBuf> = Vec::with_capacity(run.len());
    let mut i = 0usize;
    while i < run.len() {
        let mergeable = i + 1 < run.len()
            && run[i].len() == run[i + 1].len()
            && !run[i].is_empty()
            && run[i].footprint() == run[i + 1].footprint();
        if mergeable {
            // The merged column takes over the earlier buffer's storage.
            merged.push(run[i].adopt());
            i += 2;
        } else {
            merged.push(run[i].view());
            i += 1;
        }
    }

    anneal::fold_columns(&merged)
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
    fn pack_and_read_round_trip() {
        let b = ColumnBuf::pack(&[1, 2, 3]);
        assert_eq!(b.levels(), &[1, 2, 3]);
        assert_eq!(b.len(), 3);
        assert!(!b.is_empty());
    }

    #[test]
    fn view_shares_the_same_bytes() {
        let b = ColumnBuf::pack(&[4, 5]);
        let v = b.view();
        assert_eq!(v.levels(), b.levels());
    }

    #[test]
    fn footprint_separates_different_light() {
        let a = ColumnBuf::pack(&[1, 2, 3]);
        let b = ColumnBuf::pack(&[3, 2, 1]);
        assert_ne!(a.footprint(), b.footprint());
    }

    #[test]
    fn footprint_matches_for_identical_light() {
        let a = ColumnBuf::pack(&[7; 8]);
        let b = ColumnBuf::pack(&[7; 8]);
        assert_eq!(a.footprint(), b.footprint());
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
}
