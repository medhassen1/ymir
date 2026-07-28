//! Palette repacking.
//!
//! A section's palette accumulates entries as the world is edited — break a
//! block and its state may leave the section without leaving the palette. The
//! repack pass drops entries nothing refers to and narrows the index width to
//! the smallest that still addresses what remains, which is where most of a
//! region's on-disk savings come from.
//!
//! A region's repack pass compresses one section's table at a time, but the
//! entries are committed into a region-lifetime [`PaletteArena`] rather than
//! freed with the section: built once per region rather than once per
//! section, it appends every section's compressed entries into fixed-size
//! chunks and hands back a pointer into them instead of an owned buffer. A
//! handful of sections, spaced across the region, keep their span registered
//! in a retained list for the rest of the pass. Because the arena lives for
//! the whole pass, its memory is bounded separately from any one section's
//! lifetime: once the accumulated entry count crosses a threshold,
//! [`PaletteArena::compact`] drops the oldest chunks, the way a buffer pool
//! reclaims cold pages instead of growing without bound on a region with many
//! distinct palettes.

use crate::chunk::{self, Column};
use crate::common::*;
use crate::parse::Region;
use crate::repack;

/// A section's palette during repacking.
pub struct PaletteTable {
    entries: Vec<u16>,
    /// Index width in bits, always wide enough for `entries`.
    pub width: u8,
}

impl PaletteTable {
    /// A table over `entries` at the narrowest width that addresses them.
    pub fn new(entries: Vec<u16>) -> PaletteTable {
        let width = width_for(entries.len());
        PaletteTable { entries, width }
    }

    /// The palette entries.
    pub fn entries(&self) -> &[u16] {
        &self.entries
    }

    /// How many entries the palette holds.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the palette is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// A compact bit-packed fingerprint of this table's entries, used by the
    /// repack digest instead of folding each full 16-bit entry.
    ///
    /// Packed at the table's own width rather than a full 16 bits per entry,
    /// since the fingerprint only needs to distinguish tables that would
    /// compress differently, not reproduce their exact values.
    pub fn pack_bytes(&self) -> Vec<u8> {
        // A packed palette always occupies at least one byte on disk, even when
        // its indices are narrow enough to fit inside a single one.
        let mut out = vec![0u8; bytes_for(self.entries.len(), self.width).max(1)];
        let mut bit = 0usize;
        for &e in &self.entries {
            for b in 0..self.width {
                let byte = bit / 8;
                if byte >= out.len() {
                    break;
                }
                if (e >> b) & 1 != 0 {
                    out[byte] |= 1 << (bit % 8);
                }
                bit += 1;
            }
        }
        out
    }

    /// Drop the entries no block index refers to, and renarrow the width.
    ///
    /// `used` is the set of palette indices the section's blocks actually name.
    /// A palette whose entries are all referenced is already minimal and is left
    /// exactly as it stands — rebuilding it would cost an allocation and buy
    /// nothing, and most sections in a settled region are in that state.
    pub fn compress(&mut self, used: &[bool]) {
        let all_referenced =
            used.len() >= self.entries.len() && used.iter().take(self.entries.len()).all(|&u| u);
        if all_referenced {
            return;
        }
        let mut kept = Vec::with_capacity(self.entries.len());
        for (i, &e) in self.entries.iter().enumerate() {
            if used.get(i).copied().unwrap_or(false) {
                kept.push(e);
            }
        }
        if kept.is_empty() {
            kept.push(0);
        }
        self.entries = kept;
        self.width = width_for(self.entries.len());
    }

    /// How much narrower the palette became, in bits.
    pub fn savings_from(&self, original_width: u8) -> u8 {
        original_width.saturating_sub(self.width)
    }
}

/// The narrowest index width that addresses `count` entries.
pub fn width_for(count: usize) -> u8 {
    let mut w = 1u8;
    while (1usize << w) < count.max(2) && w < MAX_PALETTE_BITS {
        w += 1;
    }
    w
}

/// The number of bytes `count` entries occupy at `width` bits each, rounding
/// up to a whole byte.
///
/// Agrees with [`crate::repack::packed_bytes`] by construction: both round the
/// same bit count up the same way, so a table's fingerprint buffer is always
/// exactly as long as the byte count the fold is told to read.
pub fn bytes_for(count: usize, width: u8) -> usize {
    let bits = count * width as usize;
    bits.div_ceil(8)
}

/// Which palette indices a section's blocks actually name.
pub fn used_indices(blocks: &[u16], palette_len: usize) -> Vec<bool> {
    let mut used = vec![false; palette_len];
    for &b in blocks {
        if (b as usize) < palette_len {
            used[b as usize] = true;
        }
    }
    used
}

/// Entries held by one [`PaletteArena`] chunk.
///
/// An ordinary section's compressed palette — a handful of block states —
/// fits with room to spare, so an ordinary commit never has to look past the
/// chunk it lands in.
const ARENA_CHUNK_UNITS: usize = 256;

/// Accumulated resident entries across an arena's live chunks that triggers
/// [`PaletteArena::compact`].
///
/// An ordinary region's repack pass — a modest number of sections, most
/// carrying a small palette — never approaches this. A region built from many
/// sections with wide, distinct palettes does, which is exactly the case the
/// bound exists to catch: without it, a region-lifetime arena would keep
/// every section's compressed entries resident for the whole pass no matter
/// how many sections the region carries.
const COMPACT_THRESHOLD_UNITS: usize = 1024;

/// Spacing between sections that register a retained palette span.
///
/// The first section has no earlier neighbour to share against, so retention
/// starts at the first multiple of the stride past it.
const RETAIN_STRIDE: usize = 8;

/// A region-lifetime arena for compressed palette entries.
///
/// Built once per region rather than once per section, so a span handed out
/// while repacking one section stays valid while later sections are repacked
/// — which is what lets a retained palette span (see [`repack_region`]) be
/// read again well after its own section has finished. Entries are appended
/// into fixed-size chunks, each stored as an exact-sized boxed slice; a chunk
/// with no room left for the next commit is left as-is and a fresh one takes
/// over, so a single section's table is never split across two chunks.
struct PaletteArena {
    /// Chunks holding committed entries, oldest first.
    chunks: Vec<Box<[u16]>>,
    /// Entries already written into the last chunk.
    used: usize,
    /// Entries held across all currently resident chunks.
    resident: usize,
}

impl PaletteArena {
    fn new() -> PaletteArena {
        PaletteArena { chunks: Vec::new(), used: 0, resident: 0 }
    }

    /// Commit `entries` into the arena and return a pointer to where they
    /// landed.
    ///
    /// If what remains of the current chunk cannot hold `entries`, a fresh
    /// chunk takes over first, so the returned pointer's `entries.len()`
    /// values are always contiguous — addressing live memory for as long as
    /// the chunk backing them stays resident (see [`PaletteArena::compact`]).
    fn commit(&mut self, entries: &[u16]) -> *const u16 {
        let len = entries.len();
        let fits_current = self.chunks.last().is_some_and(|c| self.used + len <= c.len());
        if !fits_current {
            let cap = len.max(ARENA_CHUNK_UNITS);
            self.chunks.push(vec![0u16; cap].into_boxed_slice());
            self.used = 0;
            self.resident += cap;
        }
        let chunk = self.chunks.last_mut().expect("a chunk was just ensured above");
        chunk[self.used..self.used + len].copy_from_slice(entries);
        // SAFETY: `chunk` is a live `Box<[u16]>` at least `self.used + len`
        // entries long — either it already fit `entries` past `self.used`, or
        // a chunk sized to hold at least `entries` was just pushed — so this
        // offset and the `len` entries from it lie inside the allocation.
        let ptr = unsafe { chunk.as_ptr().add(self.used) };
        self.used += len;
        ptr
    }

    /// Drop the oldest resident chunks until the arena's accumulated entries
    /// fall back to `threshold`, or only the chunk currently being written to
    /// is left.
    ///
    /// This is the arena's memory bound: left unchecked, a region-lifetime
    /// arena would keep every section's entries resident for the whole pass
    /// no matter how many sections the region carries. The chunk currently
    /// being written to is never dropped, since the next commit needs
    /// somewhere to land.
    fn compact(&mut self, threshold: usize) {
        while self.resident > threshold && self.chunks.len() > 1 {
            let oldest = self.chunks.remove(0);
            self.resident -= oldest.len();
        }
    }
}

/// A section's compressed palette span retained past its own turn through
/// [`repack_region`]'s main loop, for the closing fold to read once the whole
/// region has been walked.
struct RetainedPalette {
    ptr: *const u16,
    len: usize,
    width: u8,
    savings: u8,
}

/// Bit-pack entries read from a raw span into their fingerprint bytes.
///
/// Mirrors [`PaletteTable::pack_bytes`], but reads its entries through a
/// pointer rather than an owned slice, for the region-lifetime arena path.
///
/// SAFETY: `ptr` must address at least `len` live entries for the call.
fn pack_bytes_from(ptr: *const u16, len: usize, width: u8) -> Vec<u8> {
    let mut out = vec![0u8; bytes_for(len, width).max(1)];
    if ptr.is_null() || len == 0 {
        return out;
    }
    // SAFETY: guaranteed by the precondition documented above.
    let entries = unsafe { std::slice::from_raw_parts(ptr, len) };
    let mut bit = 0usize;
    for &e in entries {
        for b in 0..width {
            let byte = bit / 8;
            if byte >= out.len() {
                break;
            }
            if (e >> b) & 1 != 0 {
                out[byte] |= 1 << (bit % 8);
            }
            bit += 1;
        }
    }
    out
}

/// Repack every palette in a column and fold a digest of the savings.
///
/// The table is compressed first, then bit-packed into its fingerprint form,
/// and the fold reads that fingerprint through one pointer rather than
/// re-borrowing the table for each of the several passes repacking makes over
/// it.
pub fn repack_column(col: &Column) -> u64 {
    let mut acc = 0xffu64 ^ (col.cid as u64);
    for s in &col.sections {
        if s.palette.is_empty() {
            continue;
        }
        let original_width = width_for(s.palette.len());
        let mut table = PaletteTable::new(s.palette.clone());

        // Drop the entries this section stopped referring to.
        let used = used_indices(&s.blocks, table.len());
        table.compress(&used);

        // The compressed table's bit-packed fingerprint, and how many bytes
        // a table of this length and width occupies.
        let packed = table.pack_bytes();
        let count = repack::packed_bytes(table.len(), table.width);

        acc = acc.wrapping_mul(0x100000001b3)
            ^ repack::fold_entries(packed.as_ptr(), count, table.savings_from(original_width));
    }
    acc
}

/// Repack every palette in the region and fold a digest of the result.
///
/// Each section's compressed entries are committed into a region-wide
/// [`PaletteArena`] rather than freed with the section: a spaced-out subset of
/// sections keep their span registered in a retained list for the rest of the
/// pass, giving the digest a cross-region term beyond each section's own
/// immediate fold. The retained spans are folded once more at the end,
/// closing out the pass.
pub fn repack_region(region: &Region, n: usize) -> u64 {
    let mut arena = PaletteArena::new();
    let mut retained: Vec<RetainedPalette> = Vec::new();
    let mut acc = 0xffu64;
    let mut sidx = 0usize;

    for cid in 0..n {
        let col = match chunk::decode(region, cid) {
            Ok(c) => c,
            Err(_) => continue,
        };
        if col.sections.is_empty() {
            continue;
        }

        // Each column folds its own sections starting from a fresh seed,
        // mirroring `repack_column`, then that result folds into the region
        // digest — the arena and its retained spans are the only things a
        // column shares with the rest of the pass.
        let mut inner = 0xffu64 ^ (col.cid as u64);
        for s in &col.sections {
            if s.palette.is_empty() {
                continue;
            }
            let original_width = width_for(s.palette.len());
            let mut table = PaletteTable::new(s.palette.clone());

            // Drop the entries this section stopped referring to.
            let used = used_indices(&s.blocks, table.len());
            table.compress(&used);
            let width = table.width;
            let savings = table.savings_from(original_width);

            // Commit this section's finished entries into the region's
            // arena. Only past this point is there a pointer stable enough to
            // retain past this section's own scope.
            let ptr = arena.commit(table.entries());
            let len = table.len();

            let packed = pack_bytes_from(ptr, len, width);
            let count = repack::packed_bytes(len, width);
            inner = inner.wrapping_mul(0x100000001b3)
                ^ repack::fold_entries(packed.as_ptr(), count, savings);

            // Every `RETAIN_STRIDE`-th section past the first keeps its span
            // alive as a reference for the rest of the pass, giving the
            // region continuity beyond just each section's immediate use.
            if sidx > 0 && sidx % RETAIN_STRIDE == 0 {
                retained.push(RetainedPalette { ptr, len, width, savings });
            }
            sidx += 1;

            // Bound the arena's resident memory now that this section's
            // entries are safely committed.
            arena.compact(COMPACT_THRESHOLD_UNITS);
        }

        acc = acc.wrapping_mul(0x100000001b3) ^ inner;
    }

    // Close out the pass by folding in every retained span once.
    for r in &retained {
        let packed = pack_bytes_from(r.ptr, r.len, r.width);
        let count = repack::packed_bytes(r.len, r.width);
        acc = acc.wrapping_mul(0x100000001b3) ^ repack::fold_entries(packed.as_ptr(), count, r.savings);
    }

    acc
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn width_for_is_monotone_and_bounded() {
        assert_eq!(width_for(0), 1);
        assert_eq!(width_for(2), 1);
        assert_eq!(width_for(3), 2);
        assert_eq!(width_for(5), 3);
        assert!(width_for(usize::MAX) <= MAX_PALETTE_BITS);
    }

    #[test]
    fn used_indices_marks_only_referenced_entries() {
        let used = used_indices(&[0, 2, 2], 4);
        assert_eq!(used, vec![true, false, true, false]);
    }

    #[test]
    fn used_indices_ignores_out_of_range_blocks() {
        let used = used_indices(&[9, 0], 2);
        assert_eq!(used, vec![true, false]);
    }

    #[test]
    fn compress_drops_unreferenced_entries() {
        let mut t = PaletteTable::new(vec![10, 20, 30, 40]);
        assert_eq!(t.len(), 4);
        t.compress(&[true, false, true, false]);
        assert_eq!(t.entries(), &[10, 30]);
        assert_eq!(t.width, width_for(2));
    }

    #[test]
    fn compress_never_empties_the_palette() {
        let mut t = PaletteTable::new(vec![7, 8]);
        t.compress(&[false, false]);
        assert_eq!(t.len(), 1);
        assert!(!t.is_empty());
    }

    #[test]
    fn savings_report_the_narrowing() {
        let mut t = PaletteTable::new(vec![1, 2, 3, 4, 5]);
        let original = t.width;
        t.compress(&[true, true, false, false, false]);
        assert!(t.savings_from(original) > 0);
        assert_eq!(t.savings_from(0), 0);
    }

    #[test]
    fn bytes_for_rounds_up() {
        assert_eq!(bytes_for(8, 1), 1);
        assert_eq!(bytes_for(9, 1), 2);
        assert_eq!(bytes_for(4096, 4), 2048);
        assert_eq!(bytes_for(0, 12), 0);
    }

    #[test]
    fn bytes_for_and_packed_bytes_always_agree() {
        // Restored: the two helpers must never disagree, whole-byte or not —
        // a mismatch here used to size a table's fingerprint buffer smaller
        // than the byte count the fold was told to read.
        for (count, width) in [(8u16 as usize, 1u8), (9, 1), (4096, 4), (0, 12), (5, 3), (17, 7)] {
            assert_eq!(bytes_for(count, width), crate::repack::packed_bytes(count, width));
        }
    }

    #[test]
    fn pack_bytes_matches_its_own_size() {
        let t = PaletteTable::new(vec![1, 2, 3, 4, 5]);
        let packed = t.pack_bytes();
        assert_eq!(packed.len(), bytes_for(t.len(), t.width).max(1));
    }

    #[test]
    fn arena_commit_writes_are_readable_back() {
        let mut arena = PaletteArena::new();
        let a = arena.commit(&[1, 2, 3, 4]);
        let b = arena.commit(&[5, 6]);
        // SAFETY: neither chunk has been compacted away, so both pointers
        // still address the entries just committed.
        unsafe {
            assert_eq!(std::slice::from_raw_parts(a, 4), [1, 2, 3, 4]);
            assert_eq!(std::slice::from_raw_parts(b, 2), [5, 6]);
        }
    }

    #[test]
    fn arena_starts_a_new_chunk_once_the_current_one_is_full() {
        let mut arena = PaletteArena::new();
        let filler = vec![0u16; ARENA_CHUNK_UNITS - 4];
        arena.commit(&filler);
        assert_eq!(arena.chunks.len(), 1);
        arena.commit(&[1, 2, 3, 4, 5]);
        assert_eq!(arena.chunks.len(), 2);
    }

    #[test]
    fn compact_leaves_a_small_arena_untouched() {
        let mut arena = PaletteArena::new();
        arena.commit(&[1, 2, 3]);
        arena.compact(COMPACT_THRESHOLD_UNITS);
        assert_eq!(arena.chunks.len(), 1);
    }

    #[test]
    fn compact_drops_oldest_chunks_once_over_threshold() {
        let mut arena = PaletteArena::new();
        for _ in 0..6 {
            arena.commit(&vec![0u16; ARENA_CHUNK_UNITS]);
        }
        assert_eq!(arena.chunks.len(), 6);
        arena.compact(3 * ARENA_CHUNK_UNITS);
        assert!(arena.chunks.len() < 6, "compact must drop some chunks");
        assert!(arena.resident <= 3 * ARENA_CHUNK_UNITS);
    }

    #[test]
    fn compact_never_drops_the_last_chunk() {
        let mut arena = PaletteArena::new();
        arena.commit(&[1, 2, 3]);
        arena.compact(0);
        assert_eq!(arena.chunks.len(), 1, "the chunk being written to must survive");
    }

    #[test]
    fn pack_bytes_from_matches_the_owned_path() {
        let entries = [1u16, 2, 3, 4, 5];
        let width = width_for(entries.len());
        let mut arena = PaletteArena::new();
        let ptr = arena.commit(&entries);
        let via_ptr = pack_bytes_from(ptr, entries.len(), width);

        let t = PaletteTable::new(entries.to_vec());
        let via_owned = t.pack_bytes();
        assert_eq!(via_ptr, via_owned);
    }

    #[test]
    fn repack_region_is_deterministic() {
        use crate::format::*;

        fn region_with_columns(num_chunks: u16) -> Vec<u8> {
            fn column_record() -> Vec<u8> {
                let mut out = Vec::new();
                out.extend_from_slice(&1u16.to_be_bytes());
                out.extend_from_slice(&0i16.to_be_bytes());
                out.push(0);
                out.push(0);
                out.push(0); // not uniform, not empty
                out.extend_from_slice(&3u16.to_be_bytes()); // palette len
                for v in [10u16, 20, 30] {
                    out.extend_from_slice(&v.to_be_bytes());
                }
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
            v.extend_from_slice(&flag::PALETTE.to_be_bytes());
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

        let data = region_with_columns(9);
        let region = crate::parse::parse(&data).expect("valid region");
        let n = region.worked_chunks();
        assert_eq!(repack_region(&region, n), repack_region(&region, n));
    }
}
