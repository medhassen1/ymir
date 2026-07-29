//! Light attenuation and level folding.
//!
//! Split out from [`crate::light`] so the falloff rule can be tuned without
//! touching the search. The attenuation step reads the frontier's origin node
//! through the cursor the search holds, which keeps the six-neighbour loop free
//! of repeated arena indexing.

use crate::light::LightNode;

/// Levels lost crossing one block, per face.
///
/// Vertical faces cost slightly more than horizontal ones, which is what gives
/// torch light its flattened falloff.
const FACE_COST: [u8; 6] = [1, 1, 2, 1, 1, 1];

/// Falloff curve over propagation depth.
///
/// A source lights a fairly uniform bubble close in and only tapers in the
/// outer shell, which this models directly rather than deriving purely from
/// [`FACE_COST`]: the first several dozen steps hold at full brightness, and
/// the curve eases down after that.
pub const FALLOFF_CURVE: [u8; 200] = build_falloff_curve();

const fn build_falloff_curve() -> [u8; 200] {
    let mut table = [0u8; 200];
    let mut i = 0;
    while i < 200 {
        table[i] = if i < 64 { 255 } else { (199 - i) as u8 };
        i += 1;
    }
    table
}

/// The falloff curve's value at `depth`, or `0` once the search has run
/// deeper than the curve models.
///
/// `depth` counts steps from the emitter along one branch of the search.
pub fn falloff_at(depth: u8, table: &[u8]) -> u8 {
    // Every depth under the table's length names a live curve entry, so the
    // lookup only proceeds once that has been confirmed.
    if depth < table.len() as u8 {
        // SAFETY: `depth` was just checked against `table.len()` above.
        unsafe { *table.get_unchecked(depth as usize) }
    } else {
        0
    }
}

/// The level arriving at a neighbour across `face`, measured from the origin
/// node named by `origin`.
///
/// SAFETY: `origin` must point at a live frontier node for the duration of the
/// call.
pub fn attenuate(origin: *const LightNode, face: u8) -> u8 {
    if origin.is_null() {
        return 0;
    }
    // SAFETY: `origin` names the frontier node the search is currently
    // expanding, which stays live for the whole expansion step.
    let level = unsafe { (*origin).level };
    let cost = FACE_COST[(face as usize) % 6];
    level.saturating_sub(cost)
}

/// Fold a digest of a column frontier committed to the region's
/// [`crate::light`] node arena.
///
/// [`crate::light::propagate_region`] commits every column's expanded frontier
/// into a region-lifetime arena and offers it as a cross-column bleed source,
/// so the region's closing pass can fold it again once every column has had its
/// own turn. This is that pass's read: `ptr`/`len` name the committed span
/// directly, walked through raw pointer arithmetic rather than a bounds-checked
/// index.
///
/// SAFETY: `ptr` must address at least `len` live [`LightNode`] values for the
/// call.
pub fn fold_span(ptr: *const LightNode, len: usize) -> u64 {
    if ptr.is_null() || len == 0 {
        return 0;
    }
    let mut acc = 0x2545f491u64;
    // SAFETY: per this function's contract, the caller guarantees `ptr`
    // addresses at least `len` live nodes.
    unsafe {
        for i in 0..len {
            let node = *ptr.add(i);
            acc = acc.rotate_left(5)
                ^ (node.level as u64)
                ^ ((node.depth as u64) << 8)
                ^ ((node.x as u64) << 16)
                ^ ((node.y as u64) << 24)
                ^ ((node.z as u64) << 32);
        }
    }
    acc
}

/// Fold how far a column's frontier has drifted from the region's seed.
///
/// Light propagated column by column accumulates error: each column is solved
/// against its own emitters, so a region lit in pieces ends up with the pieces
/// disagreeing at their boundaries. Folding every frontier against the one the
/// region started from gives the digest a term for that drift. Both sides are
/// `(pointer, length)` spans into [`crate::light`]'s node arena, walked through
/// raw pointer arithmetic because a committed frontier is named by where the
/// arena put it.
///
/// SAFETY: `seed` must address at least `seed_len` live [`LightNode`] values
/// and `here` at least `here_len`, for the duration of the call.
pub fn fold_drift(
    seed: *const LightNode,
    seed_len: usize,
    here: *const LightNode,
    here_len: usize,
) -> u64 {
    if seed.is_null() || here.is_null() || seed_len == 0 || here_len == 0 {
        return 0;
    }
    let mut acc = (seed_len as u64).rotate_left(29) ^ (here_len as u64);
    // SAFETY: per this function's contract each pointer addresses at least the
    // length it is paired with.
    unsafe {
        for i in 0..seed_len {
            let a = *seed.add(i);
            // The two frontiers are rarely the same depth; the drift term walks
            // the seed and wraps around this column's own nodes.
            let b = *here.add(i % here_len);
            acc = acc.rotate_left(7)
                ^ ((a.level as u64) << 8)
                ^ (b.level as u64)
                ^ ((a.depth ^ b.depth) as u64) << 16;
        }
    }
    acc
}

/// Fold a digest of one column's light levels.
pub fn fold_levels(levels: &[u8]) -> u64 {
    let mut acc = (levels.len() as u64).wrapping_mul(0x9e3779b1);
    for (i, &l) in levels.iter().enumerate() {
        if l != 0 {
            acc = acc.rotate_left(5) ^ ((l as u64) << (i % 48));
        }
    }
    acc
}

/// How many blocks received any light at all.
pub fn lit_count(levels: &[u8]) -> usize {
    levels.iter().filter(|&&l| l > 0).count()
}

/// The brightest level anywhere in the volume.
pub fn peak_level(levels: &[u8]) -> u8 {
    levels.iter().copied().max().unwrap_or(0)
}

/// The mean level over the lit blocks only, in sixteenths.
pub fn mean_lit(levels: &[u8]) -> u32 {
    let lit: Vec<u8> = levels.iter().copied().filter(|&l| l > 0).collect();
    if lit.is_empty() {
        return 0;
    }
    let sum: u32 = lit.iter().map(|&l| l as u32).sum();
    (sum * 16) / lit.len() as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    fn node(level: u8) -> LightNode {
        LightNode { x: 1, y: 2, z: 3, level, depth: 0 }
    }

    #[test]
    fn attenuate_costs_one_on_horizontal_faces() {
        let n = node(10);
        assert_eq!(attenuate(&n, 0), 9);
        assert_eq!(attenuate(&n, 1), 9);
        assert_eq!(attenuate(&n, 5), 9);
    }

    #[test]
    fn attenuate_costs_two_downward() {
        let n = node(10);
        assert_eq!(attenuate(&n, 2), 8);
    }

    #[test]
    fn attenuate_saturates_at_zero() {
        let n = node(1);
        assert_eq!(attenuate(&n, 2), 0);
        let z = node(0);
        assert_eq!(attenuate(&z, 0), 0);
    }

    #[test]
    fn attenuate_of_null_is_dark() {
        assert_eq!(attenuate(std::ptr::null(), 0), 0);
    }

    #[test]
    fn fold_levels_ignores_dark_blocks() {
        let dark = vec![0u8; 32];
        let mut one = dark.clone();
        one[7] = 4;
        assert_ne!(fold_levels(&dark), fold_levels(&one));
    }

    #[test]
    fn summaries_agree_with_the_data() {
        let levels = [0u8, 4, 8, 0, 12];
        assert_eq!(lit_count(&levels), 3);
        assert_eq!(peak_level(&levels), 12);
        assert_eq!(mean_lit(&levels), (4 + 8 + 12) * 16 / 3);
        assert_eq!(peak_level(&[]), 0);
        assert_eq!(mean_lit(&[0, 0]), 0);
    }

    #[test]
    fn falloff_at_reads_the_curve() {
        let table = [7u8, 6, 5];
        assert_eq!(falloff_at(0, &table), 7);
        assert_eq!(falloff_at(2, &table), 5);
    }

    #[test]
    fn falloff_at_past_the_table_is_zero() {
        let table = [7u8, 6, 5];
        assert_eq!(falloff_at(3, &table), 0);
        assert_eq!(falloff_at(255, &table), 0);
    }

    #[test]
    fn fold_span_is_deterministic() {
        let nodes = [node(4), node(9)];
        assert_eq!(fold_span(nodes.as_ptr(), nodes.len()), fold_span(nodes.as_ptr(), nodes.len()));
    }

    #[test]
    fn fold_span_reflects_contents() {
        let a = [node(4), node(9)];
        let b = [node(4), node(10)];
        assert_ne!(fold_span(a.as_ptr(), a.len()), fold_span(b.as_ptr(), b.len()));
    }

    #[test]
    fn fold_span_of_null_or_empty_is_zero() {
        let nodes = [node(4)];
        assert_eq!(fold_span(std::ptr::null(), 4), 0);
        assert_eq!(fold_span(nodes.as_ptr(), 0), 0);
    }

    #[test]
    fn fold_drift_reflects_both_frontiers() {
        let seed = [node(9), node(8)];
        let here = [node(4), node(3)];
        let base = fold_drift(seed.as_ptr(), 2, here.as_ptr(), 2);
        let other = [node(4), node(2)];
        assert_ne!(base, fold_drift(seed.as_ptr(), 2, other.as_ptr(), 2));
        let moved = [node(7), node(8)];
        assert_ne!(base, fold_drift(moved.as_ptr(), 2, here.as_ptr(), 2));
    }

    #[test]
    fn fold_drift_wraps_a_shorter_column() {
        let seed = [node(9), node(8), node(7)];
        let short = [node(4)];
        // The seed is what is walked; a shorter column is read round-robin and
        // never past its own end.
        assert_ne!(fold_drift(seed.as_ptr(), 3, short.as_ptr(), 1), 0);
        assert_eq!(fold_drift(seed.as_ptr(), 3, short.as_ptr(), 0), 0);
        assert_eq!(fold_drift(std::ptr::null(), 3, short.as_ptr(), 1), 0);
    }

    #[test]
    fn falloff_curve_is_full_length() {
        assert_eq!(FALLOFF_CURVE.len(), 200);
        assert_eq!(FALLOFF_CURVE[0], 255);
        assert_eq!(FALLOFF_CURVE[199], 0);
    }
}
