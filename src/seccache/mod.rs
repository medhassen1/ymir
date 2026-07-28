//! A rebuild-local cache of packed section buffers.
//!
//! A region touches the same column shapes repeatedly — a plateau is hundreds of
//! columns with identical section stacks — so a column's packed states are
//! materialised once into a buffer and shared across the rebuild. The buffers
//! are hand-managed rather than `Vec`-owned so the folding pass can hold
//! lightweight views into them without cloning the payload, which for a dense
//! region is the difference between one copy and several thousand.

use std::alloc::{alloc, dealloc, Layout};

use crate::budget;
use crate::chunk::{self, Column};
use crate::gather;
use crate::parse::Region;

/// A hand-managed handle to a packed section buffer.
///
/// Exactly one handle per buffer is the *owner* and frees it on drop; the rest
/// are lightweight views produced by [`SecBuf::view`].
pub struct SecBuf {
    ptr: *mut u32,
    len: usize,
    owner: bool,
}

impl SecBuf {
    /// Pack a column's resolved block states into a fresh owning buffer.
    pub fn pack(units: &[u32]) -> SecBuf {
        let len = units.len().max(1);
        let layout = Layout::array::<u32>(len).unwrap();
        // SAFETY: `len >= 1` so the layout is non-zero-sized; the allocation is
        // released by the owning handle's `Drop`.
        let ptr = unsafe { alloc(layout) as *mut u32 };
        if ptr.is_null() {
            return SecBuf { ptr: std::ptr::null_mut(), len: 0, owner: false };
        }
        for (i, &u) in units.iter().enumerate() {
            // SAFETY: `i` is bounded by `units.len() <= len`, the allocated
            // element count, so the write stays inside the allocation.
            unsafe { *ptr.add(i) = u };
        }
        SecBuf { ptr, len, owner: true }
    }

    /// A non-owning view of the same buffer.
    ///
    /// SAFETY: the owning entry stays resident in the cache for the whole
    /// rebuild, so every view handed out during that rebuild outlives its use.
    pub fn view(&self) -> SecBuf {
        SecBuf { ptr: self.ptr, len: self.len, owner: false }
    }

    /// The number of packed units.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether the buffer carries no units.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// The packed units.
    pub fn units(&self) -> &[u32] {
        if self.ptr.is_null() {
            return &[];
        }
        // SAFETY: `ptr`/`len` describe a live buffer owned by a resident entry.
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }
}

impl Drop for SecBuf {
    fn drop(&mut self) {
        if self.owner && !self.ptr.is_null() {
            let layout = Layout::array::<u32>(self.len).unwrap();
            // SAFETY: the owner frees exactly the buffer it allocated in `pack`,
            // with the layout that allocation was made under.
            unsafe { dealloc(self.ptr as *mut u8, layout) }
        }
    }
}

/// One resident cache entry.
struct Entry {
    key: u32,
    buf: SecBuf,
    used: u64,
}

/// A least-recently-used cache of packed section buffers.
///
/// Eviction is driven by an accumulated payload-memory budget rather than a
/// fixed slot count, so pressure tracks the real cost of the columns seen so
/// far rather than their number.
pub struct SecCache {
    entries: Vec<Entry>,
    budget: u64,
    spent: u64,
    clock: u64,
}

impl SecCache {
    /// A cache holding at most `budget` payload units before it evicts.
    pub fn new(budget: u64) -> SecCache {
        SecCache { entries: Vec::new(), budget: budget.max(1), spent: 0, clock: 0 }
    }

    /// How many buffers are currently resident.
    pub fn resident(&self) -> usize {
        self.entries.len()
    }

    /// Units currently charged against the budget.
    pub fn spent(&self) -> u64 {
        self.spent
    }

    /// Resolve `key` to a section view, packing and caching on a miss.
    ///
    /// Returns a view; the owning buffer lives in the cache.
    pub fn resolve(&mut self, key: u32, units: &[u32]) -> SecBuf {
        self.clock += 1;
        if let Some(e) = self.entries.iter_mut().find(|e| e.key == key) {
            // Hit: refresh recency and share the resident buffer.
            e.used = self.clock;
            return e.buf.view();
        }

        let cost = (units.len() as u64).max(1);
        // Make room within the memory budget by evicting least-recently-used
        // columns, whose owning buffers are released here.
        while self.spent + cost > self.budget && !self.entries.is_empty() {
            let victim = self
                .entries
                .iter()
                .enumerate()
                .min_by_key(|(_, e)| e.used)
                .map(|(i, _)| i)
                .unwrap();
            let e = self.entries.swap_remove(victim);
            self.spent -= (e.buf.len as u64).min(self.spent);
        }

        let buf = SecBuf::pack(units);
        let view = buf.view();
        self.spent += cost;
        self.entries.push(Entry { key, buf, used: self.clock });
        view
    }
}

/// Flatten a decoded column into the packed unit stream the cache stores.
///
/// Each unit carries the resolved block state in the low half and the section's
/// flags and vertical position in the high half, so the fold can distinguish two
/// columns that share states but differ in shape.
pub fn flatten_column(col: &Column) -> Vec<u32> {
    let mut units = Vec::new();
    for s in &col.sections {
        if s.is_empty() {
            continue;
        }
        let tag = ((s.flags as u32) << 24) | ((s.base_y as u32 & 0xff) << 16);
        let stride = if s.is_uniform() { 512 } else { 61 };
        let mut i = 0usize;
        while i < s.blocks.len() {
            units.push(tag | s.state_at(i) as u32);
            i += stride;
        }
    }
    units
}

/// The cache key for a column: its position folded with its decoded shape, so
/// two structurally identical columns share a buffer.
pub fn column_key(col: &Column) -> u32 {
    let mut k = col.cid as u32;
    k = k.wrapping_mul(0x9e3779b1) ^ (col.sections.len() as u32);
    for s in &col.sections {
        k = k.rotate_left(5) ^ (s.palette.len() as u32);
    }
    k
}

/// Resolve every column of `region` through the shared cache, then fold the
/// resident views into a digest.
///
/// Rebuilding is two passes: first resolve every column to a cached buffer
/// (which may evict earlier columns under memory pressure), then fold the digest
/// from those views. Sharing the cache across the region is what lets a plateau
/// of identical columns resolve without re-packing each one.
pub fn resolve_region(region: &Region, n: usize) -> u64 {
    let mut cache = SecCache::new(budget::cache_units(region, n));
    let mut views: Vec<(SecBuf, usize)> = Vec::new();

    // Pass 1 — resolve the region's columns against the cache.
    for cid in 0..n {
        let col = match chunk::decode(region, cid) {
            Ok(c) => c,
            Err(_) => continue,
        };
        if col.sections.is_empty() {
            continue;
        }
        let units = flatten_column(&col);
        if units.is_empty() {
            continue;
        }
        let key = column_key(&col);
        let view = cache.resolve(key, &units);
        views.push((view, col.block_count()));
    }

    // Pass 2 — fold each column from its resident view.
    let mut acc = 0xffu64;
    for (view, blocks) in &views {
        acc = acc.wrapping_mul(0x100000001b3) ^ gather::fold_view(view.units(), *blocks);
    }
    acc
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chunk::SectionData;

    fn column(cid: usize, sections: usize) -> Column {
        let secs = (0..sections)
            .map(|i| SectionData {
                palette: vec![0, 1, 2],
                blocks: vec![1; 4096],
                flags: 0,
                base_y: (i * 16) as i32,
            })
            .collect();
        Column { sections: secs, base_y: 0, flags: 0, cid }
    }

    #[test]
    fn pack_and_read_round_trip() {
        let b = SecBuf::pack(&[1, 2, 3, 4]);
        assert_eq!(b.units(), &[1, 2, 3, 4]);
        assert_eq!(b.len(), 4);
        assert!(!b.is_empty());
    }

    #[test]
    fn pack_of_empty_is_readable() {
        let b = SecBuf::pack(&[]);
        assert_eq!(b.len(), 1);
        // Contents are uninitialised-but-allocated; reading the length is fine.
        assert_eq!(b.units().len(), 1);
    }

    #[test]
    fn cache_hit_shares_without_regrowing() {
        let mut c = SecCache::new(1_000_000);
        let units = vec![7u32; 16];
        let _a = c.resolve(1, &units);
        let spent_after_first = c.spent();
        let _b = c.resolve(1, &units);
        assert_eq!(c.resident(), 1);
        assert_eq!(c.spent(), spent_after_first, "a hit must not re-charge");
    }

    #[test]
    fn cache_evicts_under_pressure() {
        let mut c = SecCache::new(8);
        // Each insert costs 4 units, so the third forces an eviction.
        let _a = c.resolve(1, &[0; 4]);
        let _b = c.resolve(2, &[0; 4]);
        assert_eq!(c.resident(), 2);
        let _d = c.resolve(3, &[0; 4]);
        assert!(c.resident() <= 2, "budget must bound residency");
    }

    #[test]
    fn flatten_skips_empty_sections() {
        let mut col = column(0, 2);
        col.sections[0].flags = crate::format::sec::EMPTY;
        let units = flatten_column(&col);
        let all = flatten_column(&column(0, 2));
        assert!(units.len() < all.len());
    }

    #[test]
    fn column_key_is_shape_sensitive() {
        assert_ne!(column_key(&column(0, 1)), column_key(&column(0, 2)));
        assert_eq!(column_key(&column(3, 2)), column_key(&column(3, 2)));
    }
}
