//! Palette repacking.
//!
//! A section's palette accumulates entries as the world is edited — break a
//! block and its state may leave the section without leaving the palette. The
//! repack pass drops entries nothing refers to and narrows the index width to
//! the smallest that still addresses what remains, which is where most of a
//! region's on-disk savings come from.

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

    /// A raw view of the entry array, as a `(pointer, count)` pair.
    ///
    /// The folding pass reads entries through this rather than re-borrowing the
    /// table for each of the several passes repacking makes over it.
    pub fn raw(&self) -> (*const u16, usize) {
        (self.entries.as_ptr(), self.entries.len())
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

/// Repack every palette in a column and fold a digest of the savings.
///
/// The raw entry view is taken before the palette is compressed, so the fold
/// reads the table through one pointer across both halves of the pass.
pub fn repack_column(col: &Column) -> u64 {
    let mut acc = 0xffu64 ^ (col.cid as u64);
    for s in &col.sections {
        if s.palette.is_empty() {
            continue;
        }
        let original_width = width_for(s.palette.len());
        let mut table = PaletteTable::new(s.palette.clone());

        // The entry view used by the fold below.
        let (entries, count) = table.raw();

        // Drop the entries this section stopped referring to.
        let used = used_indices(&s.blocks, table.len());
        table.compress(&used);

        acc = acc.wrapping_mul(0x100000001b3)
            ^ repack::fold_entries(entries, count, table.savings_from(original_width));
    }
    acc
}

/// Repack every palette in the region and fold a digest of the result.
pub fn repack_region(region: &Region, n: usize) -> u64 {
    let mut acc = 0xffu64;
    for cid in 0..n {
        let col = match chunk::decode(region, cid) {
            Ok(c) => c,
            Err(_) => continue,
        };
        if col.sections.is_empty() {
            continue;
        }
        acc = acc.wrapping_mul(0x100000001b3) ^ repack_column(&col);
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
}
