//! Worley (cellular) noise: F1/F2 distances to a hashed grid of feature
//! points.
//!
//! Where Perlin and simplex noise are smooth everywhere, Worley noise has
//! sharp cell boundaries — it is what `ymir` reaches for when generating
//! things that should look like *regions* rather than gradients: cave
//! chamber layouts, ore vein clusters, biome cell maps, or cracked-stone
//! textures. Each unit cell of the integer grid gets exactly one
//! pseudo-random feature point (derived from the cell coordinates and a
//! seed, so no state needs to be stored anywhere), and F1/F2 are the
//! distances from a query point to the nearest and second-nearest of
//! those feature points.

/// Selects how distance to a feature point is measured.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DistanceMetric {
    /// Ordinary straight-line distance.
    Euclidean,
    /// Taxicab distance; cheaper, and produces angular, faceted cells.
    Manhattan,
}

/// Hashes a grid cell `(x, y)` plus a `seed` into a well-mixed 64-bit
/// value, using the finalizer from MurmurHash3's 128-bit variant (chosen
/// here purely as a strong avalanche mix, not for MurmurHash compatibility).
fn hash_cell(x: i32, y: i32, seed: u64) -> u64 {
    let mut h = (x as u32 as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    h ^= (y as u32 as u64).wrapping_mul(0xC2B2_AE3D_27D4_EB4F);
    h ^= seed.wrapping_mul(0x1656_67B1_9E37_79F9);
    h ^= h >> 33;
    h = h.wrapping_mul(0xFF51_AFD7_ED55_8CCD);
    h ^= h >> 33;
    h = h.wrapping_mul(0xC4CE_B9FE_1A85_EC53);
    h ^= h >> 33;
    h
}

/// A seeded Worley/cellular noise sampler. Holds nothing but the seed:
/// feature points are derived on demand from cell coordinates, so a
/// sampler is cheap to create and to clone.
#[derive(Debug, Clone, Copy)]
pub struct WorleyNoise {
    seed: u64,
}

impl WorleyNoise {
    /// Builds a sampler for the given seed.
    pub fn new(seed: u64) -> Self {
        WorleyNoise { seed }
    }

    /// Returns the single pseudo-random feature point that lives inside
    /// grid cell `(cx, cy)`, in world coordinates.
    fn feature_point(&self, cx: i32, cy: i32) -> (f64, f64) {
        let h = hash_cell(cx, cy, self.seed);
        let hx = (h & 0xFFFF_FFFF) as u32;
        let hy = (h >> 32) as u32;
        let fx = hx as f64 / u32::MAX as f64;
        let fy = hy as f64 / u32::MAX as f64;
        (cx as f64 + fx, cy as f64 + fy)
    }

    /// Returns the distance to the nearest feature point (`F1`).
    pub fn f1(&self, x: f64, y: f64, metric: DistanceMetric) -> f64 {
        self.f1_f2(x, y, metric).0
    }

    /// Returns both the nearest (`F1`) and second-nearest (`F2`) feature
    /// point distances. Since every cell has exactly one feature point and
    /// it always lies within its own cell, scanning the query point's
    /// cell plus its 8 neighbors is always sufficient to find both.
    pub fn f1_f2(&self, x: f64, y: f64, metric: DistanceMetric) -> (f64, f64) {
        let cx = x.floor() as i32;
        let cy = y.floor() as i32;

        let mut dists = [0.0f64; 9];
        let mut n = 0;
        for dy in -1..=1 {
            for dx in -1..=1 {
                let (fx, fy) = self.feature_point(cx + dx, cy + dy);
                let d = match metric {
                    DistanceMetric::Euclidean => ((fx - x).powi(2) + (fy - y).powi(2)).sqrt(),
                    DistanceMetric::Manhattan => (fx - x).abs() + (fy - y).abs(),
                };
                dists[n] = d;
                n += 1;
            }
        }

        let mut f1 = f64::INFINITY;
        let mut f2 = f64::INFINITY;
        for i in 0..dists.len() {
            // SAFETY: `dists` has the fixed length 9 (one entry per
            // neighbor cell written by the loop above) and `i` ranges over
            // `0..dists.len()`, so the index is always in bounds.
            let d = unsafe { *dists.get_unchecked(i) };
            if d < f1 {
                f2 = f1;
                f1 = d;
            } else if d < f2 {
                f2 = d;
            }
        }
        (f1, f2)
    }

    /// Fills `out` with `F1` distances along a scanline starting at
    /// `x_start`, stepping by `step`, at fixed `y`. This is the batch
    /// entry point a chunk generator uses to fill a full row of a cave or
    /// biome cell map at once.
    pub fn f1_row(&self, y: f64, x_start: f64, step: f64, metric: DistanceMetric, out: &mut [f64]) {
        for i in 0..out.len() {
            let x = x_start + i as f64 * step;
            let v = self.f1(x, y, metric);
            // SAFETY: `i` ranges over `0..out.len()`, so the index is
            // always in bounds for `out`.
            unsafe {
                *out.get_unchecked_mut(i) = v;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const POINTS: [(f64, f64); 6] = [
        (0.3, 0.7),
        (5.5, -3.2),
        (-1.1, 8.9),
        (100.25, 100.25),
        (0.0, 0.0),
        (-50.0, 50.0),
    ];

    #[test]
    fn deterministic_for_same_seed() {
        let a = WorleyNoise::new(42);
        let b = WorleyNoise::new(42);
        for &(x, y) in &POINTS {
            assert_eq!(
                a.f1_f2(x, y, DistanceMetric::Euclidean),
                b.f1_f2(x, y, DistanceMetric::Euclidean)
            );
        }
    }

    #[test]
    fn f1_never_exceeds_f2() {
        let n = WorleyNoise::new(7);
        for &(x, y) in &POINTS {
            let (f1, f2) = n.f1_f2(x, y, DistanceMetric::Euclidean);
            assert!(f1 <= f2, "F1 {f1} exceeded F2 {f2} at ({x}, {y})");
            let (f1m, f2m) = n.f1_f2(x, y, DistanceMetric::Manhattan);
            assert!(f1m <= f2m, "Manhattan F1 {f1m} exceeded F2 {f2m} at ({x}, {y})");
        }
    }

    #[test]
    fn manhattan_is_never_smaller_than_euclidean() {
        // Taxicab distance dominates straight-line distance between the
        // same pair of points, for every metric pair evaluated here.
        let n = WorleyNoise::new(3);
        for &(x, y) in &POINTS {
            let e = n.f1(x, y, DistanceMetric::Euclidean);
            let m = n.f1(x, y, DistanceMetric::Manhattan);
            assert!(m >= e - 1e-12, "Manhattan {m} was smaller than Euclidean {e}");
        }
    }

    #[test]
    fn different_seeds_change_the_feature_layout() {
        let a = WorleyNoise::new(1);
        let b = WorleyNoise::new(2);
        let mut any_diff = false;
        for &(x, y) in &POINTS {
            if (a.f1(x, y, DistanceMetric::Euclidean) - b.f1(x, y, DistanceMetric::Euclidean)).abs() > 1e-9 {
                any_diff = true;
                break;
            }
        }
        assert!(any_diff, "two different seeds produced identical F1 fields");
    }

    #[test]
    fn f1_row_matches_direct_calls() {
        let n = WorleyNoise::new(9001);
        let mut out = [0.0; 12];
        n.f1_row(3.25, -2.0, 0.5, DistanceMetric::Euclidean, &mut out);
        for (i, &v) in out.iter().enumerate() {
            let x = -2.0 + i as f64 * 0.5;
            assert_eq!(v, n.f1(x, 3.25, DistanceMetric::Euclidean));
        }
    }

    #[test]
    fn distances_are_finite_and_nonnegative_on_cell_boundaries() {
        let n = WorleyNoise::new(55);
        for (x, y) in [(0i32, 0i32), (1, 1), (-3, 4), (10, -10)] {
            let (f1, f2) = n.f1_f2(x as f64, y as f64, DistanceMetric::Euclidean);
            assert!(f1.is_finite() && f1 >= 0.0);
            assert!(f2.is_finite() && f2 >= 0.0);
        }
    }
}
