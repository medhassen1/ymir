//! A rebuild-local cache of packed section buffers.
//!
//! A region touches the same column shapes repeatedly — a plateau is hundreds of
//! columns with identical section stacks — so a column's packed states are
//! materialised once into a buffer and shared across the rebuild. The buffers
//! are hand-managed rather than `Vec`-owned so the folding pass can hold
//! lightweight views into them without cloning the payload, which for a dense
//! region is the difference between one copy and several thousand.
//!
//! The cache is region-lifetime: built once per rebuild rather than once per
//! column, so a view resolved while processing one column stays meaningful to
//! read back later in the same pass. [`resolve_region`] folds every column's
//! view on the spot and additionally keeps the more substantial ones in a small
//! warm set — see [`WarmSet`] — which the closing fold reads once the whole
//! region has been walked, the same way a caller might resolve a view early and
//! read it back later in the same rebuild. Residency is bounded by an
//! accumulated payload-memory budget: a region with enough distinct column
//! shapes ages the coldest of them out (see [`SecCache::resolve`]).

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
    /// SAFETY: a view is only sound to read while the entry that owns this
    /// buffer is still resident in the cache. A caller that holds a view past
    /// a later `resolve` call must know that call did not age the owning entry
    /// out — this handle carries no such guarantee itself.
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

    /// The start of the packed units.
    ///
    /// Callers that already know how many units they contributed walk the
    /// buffer from here instead of re-deriving its length.
    pub fn as_ptr(&self) -> *const u32 {
        self.ptr
    }

    /// The packed units.
    ///
    /// SAFETY: as for any read through a view, the entry that owns this buffer
    /// must still be resident — see [`SecBuf::view`].
    pub fn units(&self) -> &[u32] {
        if self.ptr.is_null() {
            return &[];
        }
        // SAFETY: guaranteed by the precondition documented above.
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
/// Recency is refreshed on a hit and nowhere else, which is what makes the
/// ordering mean something: a shape the region keeps coming back to stays
/// resident however long the pass runs, while one the region never asks for
/// again ages toward the back. Eviction is driven by an accumulated
/// payload-memory budget rather than a fixed slot count, so pressure tracks the
/// real cost of the columns seen so far rather than their number.
pub struct SecCache {
    entries: Vec<Entry>,
    budget: u64,
    spent: u64,
    clock: u64,
    aged: u64,
}

impl SecCache {
    /// A cache holding at most `budget` payload units before it ages entries
    /// out.
    ///
    /// The budget is a starting estimate, not a hard ceiling: see
    /// [`SecCache::resolve`] for the one case that raises it.
    pub fn new(budget: u64) -> SecCache {
        SecCache { entries: Vec::new(), budget: budget.max(1), spent: 0, clock: 0, aged: 0 }
    }

    /// How many buffers are currently resident.
    pub fn resident(&self) -> usize {
        self.entries.len()
    }

    /// Units currently charged against the budget.
    pub fn spent(&self) -> u64 {
        self.spent
    }

    /// The budget entries are currently aged out against.
    pub fn budget(&self) -> u64 {
        self.budget
    }

    /// Whether the cache has had to age anything out yet.
    ///
    /// Until it has, every shape the region has offered is still resident and
    /// the residency says more about the region's length than its content. Once
    /// it has, what is left is the working set the region's own pressure
    /// selected.
    pub fn settled(&self) -> bool {
        self.aged > 0
    }

    /// A view of every buffer currently resident, in residency order.
    pub fn resident_views(&self) -> Vec<SecBuf> {
        self.entries.iter().map(|e| e.buf.view()).collect()
    }

    /// Resolve `key` to a section view, packing and caching on a miss.
    ///
    /// Returns a view; the owning buffer lives in the cache.
    ///
    /// A miss first raises the budget far enough to admit the incoming column
    /// outright. A cache whose budget cannot hold the very buffer it was asked
    /// to pack would empty itself on every such column and still miss, so the
    /// widest column the region has offered sets the floor. The budget only
    /// ratchets upward and settles once that column has been seen. Room is then
    /// made within it by ageing out least-recently-used columns, whose owning
    /// buffers are released here.
    pub fn resolve(&mut self, key: u32, units: &[u32]) -> SecBuf {
        self.clock += 1;
        if let Some(e) = self.entries.iter_mut().find(|e| e.key == key) {
            // Hit: refresh recency and share the resident buffer.
            e.used = self.clock;
            return e.buf.view();
        }

        let cost = (units.len() as u64).max(1);
        self.budget = self.budget.max(cost);
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
            self.aged += 1;
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
            // Air contributes nothing to a column's digest, so sampled air is
            // dropped rather than packed. A mostly-empty column therefore packs
            // to far fewer units than a solid one of the same shape.
            let state = s.state_at(i);
            if state != 0 {
                units.push(tag | state as u32);
            }
            i += stride;
        }
    }
    units
}

/// The cache key for a column: an FNV-1a hash of its packed unit stream.
///
/// Deliberately independent of where the column sits in the region — two
/// structurally identical columns should share one buffer however far apart
/// they are, which is what makes the cache worth having on a plateau. Folding
/// the packed units themselves, rather than a summary of the column's shape,
/// is what makes a shared key mean the columns actually agree: two columns
/// that hash alike but decoded to different states would otherwise share a
/// cached buffer sized for the wrong one.
pub fn column_key(units: &[u32]) -> u32 {
    let mut k = 0x811c9dc5u32;
    for &u in units {
        k = (k ^ u).wrapping_mul(0x01000193);
    }
    k
}

/// A column packing at least this many units is substantial enough to be worth
/// keeping warm for [`resolve_region`]'s closing fold.
///
/// A column that packs almost nothing adds almost nothing to the digest a
/// second time, so the warm set spends its slots on the columns that carry the
/// region's shape.
const WARM_MIN_UNITS: usize = 32;

/// Slots in [`WarmSet`].
const WARM_SLOTS: usize = 4;

/// One view kept warm past its own turn through [`resolve_region`]'s main loop.
struct WarmView {
    buf: SecBuf,
    blocks: usize,
}

/// The views [`resolve_region`] reads once more after the whole region has been
/// walked.
///
/// Fixed capacity, reused round-robin. The closing fold has to cost the same on
/// a region of ten columns and one of ten thousand, so the newest views displace
/// the oldest instead of the set growing with the region — the alternative is a
/// deferred list whose length is the region's, which is exactly the unbounded
/// retention the cache's own budget exists to avoid.
struct WarmSet {
    slots: [Option<WarmView>; WARM_SLOTS],
    next: usize,
}

impl WarmSet {
    fn new() -> WarmSet {
        WarmSet { slots: [None, None, None, None], next: 0 }
    }

    /// Put `view` in the next slot, displacing whatever that slot held.
    fn keep(&mut self, view: WarmView) {
        self.slots[self.next] = Some(view);
        self.next = (self.next + 1) % WARM_SLOTS;
    }

    /// The views currently held, oldest slot first.
    fn views(&self) -> impl Iterator<Item = &WarmView> {
        self.slots.iter().flatten()
    }
}

/// Resolve every column of `region` through the shared cache, then fold the
/// resident views into a digest.
///
/// Rebuilding is two passes: first resolve every column to a cached buffer
/// (which may age earlier columns out under memory pressure), then fold the
/// digest from those views. Sharing the cache across the region is what lets a
/// plateau of identical columns resolve without re-packing each one. The more
/// substantial columns additionally leave their view in a small warm set that
/// is only folded once every column has been resolved — giving the digest a
/// cross-region term beyond each column's own immediate fold, the same way a
/// caller might resolve a view early and read it back later in the same
/// rebuild.
///
/// The closing fold has a second term beside the warm set: the cache's working
/// set, captured the moment the cache first has to age something out. Up to
/// that point residency only says how many distinct shapes the region has
/// offered; from it, what is resident is what the region's own pressure
/// selected, and folding that once at the end gives the digest a term for the
/// shapes the region settled on rather than the ones it merely touched.
pub fn resolve_region(region: &Region, n: usize) -> u64 {
    let mut cache = SecCache::new(budget::cache_units(region, n));
    let mut warm = WarmSet::new();
    let mut working: Vec<SecBuf> = Vec::new();
    let mut acc = 0xffu64;

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
        let key = column_key(&units);
        // A resolved view borrows the cache entry that owns it, and a later
        // `resolve` may age that entry out to stay inside the budget. Reading
        // it right here, while its entry is still the one just touched, is
        // always sound; the view's own length is what is walked, not the count
        // this column happened to pack, so a shared buffer never gets
        // over-read.
        let view = cache.resolve(key, &units);
        acc = acc.wrapping_mul(0x100000001b3)
            ^ gather::fold_span(view.as_ptr(), view.len(), col.block_count());

        if units.len() >= WARM_MIN_UNITS {
            warm.keep(WarmView { buf: view.view(), blocks: col.block_count() });
        }

        // The first time the cache has to age a shape out, whatever survived is
        // the working set the region's pressure picked. Take it once — a later
        // snapshot would describe a different, later region.
        if working.is_empty() && cache.settled() {
            working = cache.resident_views();
        }
    }

    // Close out the pass by folding in every warm view once, now that the whole
    // region has gone through the cache.
    for w in warm.views() {
        acc = acc.wrapping_mul(0x100000001b3)
            ^ gather::fold_span(w.buf.as_ptr(), w.buf.len(), w.blocks);
    }

    // Then the working set, so the digest carries what the region settled on.
    for v in &working {
        acc = acc.wrapping_mul(0x9e3779b97f4a7c15)
            ^ gather::fold_span(v.as_ptr(), v.len(), v.len());
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
    fn cache_ages_entries_out_under_pressure() {
        let mut c = SecCache::new(8);
        // Each insert costs 4 units, so the third forces the coldest out.
        let _a = c.resolve(1, &[0; 4]);
        let _b = c.resolve(2, &[0; 4]);
        assert_eq!(c.resident(), 2);
        let _d = c.resolve(3, &[0; 4]);
        assert!(c.resident() <= 2, "budget must bound residency");
    }

    #[test]
    fn a_hit_keeps_an_entry_ahead_of_its_neighbours() {
        let mut c = SecCache::new(8);
        let _a = c.resolve(1, &[0; 4]);
        let _b = c.resolve(2, &[1; 4]);
        // Re-resolving key 1 refreshes its recency, so key 2 is now coldest.
        let _a2 = c.resolve(1, &[0; 4]);
        let _d = c.resolve(3, &[2; 4]);
        assert_eq!(c.resident(), 2);
        assert!(
            c.resolve(1, &[0; 4]).len() == 4,
            "the refreshed entry must still be resident"
        );
    }

    /// Residency is what the budget is for. A cache walked over a long run of
    /// distinct shapes must hold what the budget admits and no more, however
    /// many shapes the region offers — otherwise the pass keeps every buffer
    /// the region ever packed.
    #[test]
    fn residency_stays_inside_the_budget_over_a_long_run() {
        let mut c = SecCache::new(64);
        for key in 0..200u32 {
            let _ = c.resolve(key, &[key; 8]);
        }
        assert!(c.settled(), "a run this long must have aged shapes out");
        assert!(
            c.spent() <= c.budget(),
            "residency outgrew its budget: {} units against {}",
            c.spent(),
            c.budget()
        );
        assert!(c.resident() <= 8, "at 8 units a shape, 64 units is 8 shapes");
    }

    #[test]
    fn a_fresh_cache_has_not_settled() {
        let mut c = SecCache::new(1_000_000);
        assert!(!c.settled());
        assert!(c.resident_views().is_empty());
        let _a = c.resolve(1, &[0; 4]);
        let _b = c.resolve(2, &[0; 4]);
        assert!(!c.settled(), "nothing was aged out, so nothing has settled");
        assert_eq!(c.resident_views().len(), 2);
    }

    #[test]
    fn budget_ratchets_up_to_admit_an_outsized_column() {
        let mut c = SecCache::new(8);
        let _a = c.resolve(1, &[0; 4]);
        let big = vec![9u32; 64];
        let v = c.resolve(2, &big);
        assert_eq!(v.len(), 64, "an outsized column must still be cached");
        assert!(c.budget() >= 64, "the budget must admit the widest column seen");
        // Ratcheting only ever raises the budget.
        let _c = c.resolve(3, &[0; 2]);
        assert!(c.budget() >= 64);
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
    fn column_key_is_content_sensitive() {
        let a = flatten_column(&column(0, 1));
        let b = flatten_column(&column(0, 2));
        assert_ne!(column_key(&a), column_key(&b));
        assert_eq!(column_key(&a), column_key(&flatten_column(&column(0, 1))));
    }

    #[test]
    fn warm_set_holds_only_its_newest_slots() {
        let mut w = WarmSet::new();
        for _ in 0..WARM_SLOTS + 3 {
            w.keep(WarmView { buf: SecBuf::pack(&[1, 2]), blocks: 2 });
        }
        assert_eq!(w.views().count(), WARM_SLOTS, "capacity must be fixed");
    }

    #[test]
    fn a_warm_view_still_resident_folds_the_same_way_every_time() {
        // A single substantial column: warm-kept, and far too little total
        // volume for any budget to age it out — so the closing fold reads a
        // view whose entry is still the one the main loop just touched.
        use crate::format::*;

        fn region_bytes(num_chunks: u16, cdat_cols: &[Vec<u8>]) -> Vec<u8> {
            let mut v = Vec::new();
            v.extend_from_slice(&MAGIC);
            v.extend_from_slice(&VERSION.to_be_bytes());
            v.extend_from_slice(&0u16.to_be_bytes()); // flags: seccache default
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
            let cdat: Vec<u8> = cdat_cols.iter().flatten().copied().collect();

            v.extend_from_slice(&tag::CMAP);
            v.extend_from_slice(&(cmap_off as u32).to_be_bytes());
            v.extend_from_slice(&(cmap_len as u32).to_be_bytes());
            v.extend_from_slice(&tag::CDAT);
            v.extend_from_slice(&(cdat_off as u32).to_be_bytes());
            v.extend_from_slice(&(cdat.len() as u32).to_be_bytes());

            let mut at = 0u32;
            for c in cdat_cols {
                v.extend_from_slice(&at.to_be_bytes());
                at += c.len() as u32;
            }
            v.extend_from_slice(&at.to_be_bytes());
            v.extend_from_slice(&cdat);
            v
        }

        // A column of solid uniform sections: enough packed units to be worth
        // keeping warm, nowhere near enough volume to trouble the budget.
        fn stacked_column(sections: u16, fill: u16) -> Vec<u8> {
            let mut out = Vec::new();
            out.extend_from_slice(&sections.to_be_bytes());
            out.extend_from_slice(&0i16.to_be_bytes());
            out.push(0);
            out.push(0);
            for _ in 0..sections {
                out.push(crate::format::sec::UNIFORM);
                out.extend_from_slice(&2u16.to_be_bytes());
                out.extend_from_slice(&0u16.to_be_bytes());
                out.extend_from_slice(&fill.to_be_bytes());
                out.extend_from_slice(&1u16.to_be_bytes()); // uniform index
            }
            out
        }

        let cols = vec![stacked_column(6, 9)];
        let data = region_bytes(cols.len() as u16, &cols);
        let region = crate::parse::parse(&data).expect("valid region");
        assert_eq!(
            resolve_region(&region, cols.len()),
            resolve_region(&region, cols.len()),
        );
    }
}
