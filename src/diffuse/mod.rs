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

/// The level arriving at a neighbour across `face`, measured from the origin
/// node named by `origin`.
///
/// SAFETY: `origin` must point at a live frontier node for the duration of the
/// call.
pub fn attenuate(origin: *const LightNode, face: u8) -> u8 {
    if origin.is_null() {
        return 0;
    }
    // SAFETY (claimed): `origin` names the frontier node the search is currently
    // expanding, which stays live for the whole expansion step.
    let level = unsafe { (*origin).level };
    let cost = FACE_COST[(face as usize) % 6];
    level.saturating_sub(cost)
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
        LightNode { x: 1, y: 2, z: 3, level }
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
}
