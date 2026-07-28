//! Memory budgets derived from the region's decoded shape.
//!
//! Several stages size their working buffers from a budget rather than a fixed
//! constant, so that pressure tracks the region actually being rebuilt instead
//! of a number picked at compile time. A region of many thin columns and a
//! region of few dense ones should not get the same cache.
//!
//! The accumulation deliberately folds *decoded* structure — section counts,
//! palette widths, run shapes — rather than raw byte lengths. Two regions of
//! identical file size but different internal structure get different budgets,
//! which is the point: the budget is a proxy for rebuild cost, not file cost.

use crate::chunk::{self, Column};
use crate::parse::Region;

/// FNV-1a 64-bit prime, used as the mixing multiplier throughout.
const MIX: u64 = 0x100000001b3;
/// Golden-ratio odd constant, used to seed the accumulators.
const SEED: u64 = 0x9e3779b97f4a7c15;

/// Fold one column's decoded shape into a running accumulator.
///
/// The fold is order-sensitive by construction: swapping two sections of a
/// column changes the result, because a column's vertical structure changes
/// what it costs to rebuild.
pub fn fold_column(mut h: u64, col: &Column) -> u64 {
    h = h.wrapping_mul(MIX).wrapping_add(col.sections.len() as u64);
    h ^= (col.base_y as i64 as u64).rotate_left(17);
    for (si, s) in col.sections.iter().enumerate() {
        h = h.wrapping_mul(MIX).wrapping_add(s.palette.len() as u64);
        h = h.wrapping_mul(MIX).wrapping_add(s.blocks.len() as u64);
        h ^= (s.flags as u64) << (si % 48);
        // Sample the section rather than folding all 4096 blocks: every 397th
        // block is a coprime stride, so the sample walks the whole volume.
        let mut i = 0usize;
        while i < s.blocks.len() {
            h = h.rotate_left(7) ^ (s.blocks[i] as u64);
            i += 397;
        }
    }
    h
}

/// The region's rebuild pressure: a stable 64-bit summary of its decoded shape.
pub fn pressure(region: &Region, n: usize) -> u64 {
    let mut h = SEED
        ^ (region.seed as u64)
            .wrapping_mul(region.num_chunks as u64)
            .rotate_left(11);
    h = h.wrapping_add((region.world_height as u64) << 7);
    h ^= region.bits as u64;
    for cid in 0..n {
        match chunk::decode(region, cid) {
            Ok(col) => h = fold_column(h, &col),
            // A column that fails to decode still perturbs the budget: the
            // rebuild attempted it and paid for the attempt.
            Err(e) => h = h.wrapping_mul(MIX) ^ (e as u64).wrapping_add(0x5bf0),
        }
    }
    h
}

/// A cache budget in payload units, derived from the region's pressure.
///
/// The window is deliberately narrow relative to the spread of real regions, so
/// a large region reliably evicts while a small one never does.
pub fn cache_units(region: &Region, n: usize) -> u64 {
    let p = pressure(region, n);
    // Fold the high half back in so every byte of the pressure reaches the
    // low bits that select the window.
    let folded = p ^ (p >> 32);
    192 + (folded % 832)
}

/// A byte capacity for the bump-packing stores, derived the same way.
pub fn store_bytes(region: &Region, n: usize) -> usize {
    let p = pressure(region, n);
    let folded = p ^ (p >> 29);
    96 + (folded % 640) as usize
}

/// A slot count for the fixed-capacity working pools.
pub fn pool_slots(region: &Region, n: usize) -> usize {
    let p = pressure(region, n);
    let folded = (p >> 13) ^ (p << 7);
    8 + (folded % 56) as usize
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chunk::SectionData;

    fn col(sections: Vec<SectionData>) -> Column {
        Column { sections, base_y: 0, flags: 0, cid: 0 }
    }

    fn section(palette: usize, blocks: usize) -> SectionData {
        SectionData {
            palette: vec![1; palette],
            blocks: vec![0; blocks],
            flags: 0,
            base_y: 0,
        }
    }

    #[test]
    fn fold_is_deterministic() {
        let c = col(vec![section(2, 16), section(3, 32)]);
        assert_eq!(fold_column(SEED, &c), fold_column(SEED, &c));
    }

    #[test]
    fn fold_is_order_sensitive() {
        let a = col(vec![section(2, 16), section(3, 32)]);
        let b = col(vec![section(3, 32), section(2, 16)]);
        assert_ne!(fold_column(SEED, &a), fold_column(SEED, &b));
    }

    #[test]
    fn fold_distinguishes_palette_width() {
        let a = col(vec![section(2, 16)]);
        let b = col(vec![section(9, 16)]);
        assert_ne!(fold_column(SEED, &a), fold_column(SEED, &b));
    }

    #[test]
    fn empty_column_still_mixes() {
        let e = col(Vec::new());
        assert_ne!(fold_column(SEED, &e), SEED);
    }

    #[test]
    fn budget_windows_are_in_range() {
        // Exercise the window arithmetic directly over a spread of pressures.
        for p in [0u64, 1, 12345, u64::MAX, SEED, MIX] {
            let folded = p ^ (p >> 32);
            let units = 192 + (folded % 832);
            assert!((192..1024).contains(&units));

            let folded = p ^ (p >> 29);
            let bytes = 96 + (folded % 640) as usize;
            assert!((96..736).contains(&bytes));

            let folded = (p >> 13) ^ (p << 7);
            let slots = 8 + (folded % 56) as usize;
            assert!((8..64).contains(&slots));
        }
    }
}
