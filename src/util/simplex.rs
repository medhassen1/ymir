//! 2D and 3D simplex noise (Gustavson's formulation of Perlin's simplex
//! algorithm). Where classic Perlin noise costs `2^n` corner evaluations
//! per sample in `n` dimensions, simplex noise only ever touches `n + 1`
//! corners and has no directional grid artifacts. `ymir` reaches for this
//! for cave density fields and biome-blend weights.

/// Skew factor for mapping 2D square coordinates onto the simplex grid.
const F2: f64 = 0.366_025_403_784_438_6; // 0.5 * (sqrt(3) - 1)
/// Unskew factor for mapping 2D simplex coordinates back to square space.
const G2: f64 = 0.211_324_865_405_187_1; // (3 - sqrt(3)) / 6
/// Skew factor for 3D.
const F3: f64 = 1.0 / 3.0;
/// Unskew factor for 3D.
const G3: f64 = 1.0 / 6.0;

/// The 12 gradient directions (cube edge midpoints) shared by the 2D and
/// 3D samplers, per Gustavson's reference implementation.
const GRAD3: [[f64; 3]; 12] = [
    [1.0, 1.0, 0.0], [-1.0, 1.0, 0.0], [1.0, -1.0, 0.0], [-1.0, -1.0, 0.0],
    [1.0, 0.0, 1.0], [-1.0, 0.0, 1.0], [1.0, 0.0, -1.0], [-1.0, 0.0, -1.0],
    [0.0, 1.0, 1.0], [0.0, -1.0, 1.0], [0.0, 1.0, -1.0], [0.0, -1.0, -1.0],
];

#[inline]
fn dot2(g: [f64; 3], x: f64, y: f64) -> f64 {
    g[0] * x + g[1] * y
}

#[inline]
fn dot3(g: [f64; 3], x: f64, y: f64, z: f64) -> f64 {
    g[0] * x + g[1] * y + g[2] * z
}

/// One step of a small SplitMix64-style mixer, used only to seed this
/// module's permutation shuffle, kept self-contained rather than
/// depending on [`crate::util::splitmix64`].
fn mix_step(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// A seeded 2D/3D simplex noise sampler, backed by a 512-entry
/// permutation table (a `0..256` permutation duplicated once).
#[derive(Debug, Clone)]
pub struct SimplexNoise {
    perm: [u8; 512],
}

impl SimplexNoise {
    /// Builds a sampler whose permutation table is a deterministic
    /// shuffle of `0..256` driven by `seed`.
    pub fn new(seed: u64) -> Self {
        let mut p = [0u8; 256];
        for (i, slot) in p.iter_mut().enumerate() {
            *slot = i as u8;
        }
        let mut state = seed;
        for i in (1..256).rev() {
            let r = mix_step(&mut state);
            let j = (r % (i as u64 + 1)) as usize;
            p.swap(i, j);
        }
        let mut perm = [0u8; 512];
        for (idx, slot) in perm.iter_mut().enumerate() {
            *slot = p[idx & 255];
        }
        SimplexNoise { perm }
    }

    /// Looks up the permutation table at `i`, wrapping via a bitmask.
    #[inline]
    fn hash(&self, i: i32) -> i32 {
        let idx = (i & 0x1FF) as usize;
        // SAFETY: `idx` is masked with `0x1FF` (511), so it is always in
        // `0..512`, matching `self.perm`'s fixed length of 512.
        unsafe { *self.perm.get_unchecked(idx) as i32 }
    }

    /// Samples 2D simplex noise at `(x, y)`, nominally in `[-1, 1]`.
    pub fn noise2(&self, x: f64, y: f64) -> f64 {
        let s = (x + y) * F2;
        let i = (x + s).floor();
        let j = (y + s).floor();
        let t = (i + j) * G2;
        let x0 = x - (i - t);
        let y0 = y - (j - t);
        let (i1, j1) = if x0 > y0 { (1, 0) } else { (0, 1) };
        let x1 = x0 - i1 as f64 + G2;
        let y1 = y0 - j1 as f64 + G2;
        let x2 = x0 - 1.0 + 2.0 * G2;
        let y2 = y0 - 1.0 + 2.0 * G2;

        let ii = (i as i32) & 255;
        let jj = (j as i32) & 255;
        let gi0 = (self.hash(ii + self.hash(jj)) % 12) as usize;
        let gi1 = (self.hash(ii + i1 + self.hash(jj + j1)) % 12) as usize;
        let gi2 = (self.hash(ii + 1 + self.hash(jj + 1)) % 12) as usize;

        let n0 = corner_contribution2(0.5, x0, y0, GRAD3[gi0]);
        let n1 = corner_contribution2(0.5, x1, y1, GRAD3[gi1]);
        let n2 = corner_contribution2(0.5, x2, y2, GRAD3[gi2]);
        70.0 * (n0 + n1 + n2)
    }

    /// Samples 3D simplex noise at `(x, y, z)`, nominally in `[-1, 1]`.
    pub fn noise3(&self, x: f64, y: f64, z: f64) -> f64 {
        let s = (x + y + z) * F3;
        let i = (x + s).floor();
        let j = (y + s).floor();
        let k = (z + s).floor();
        let t = (i + j + k) * G3;
        let x0 = x - (i - t);
        let y0 = y - (j - t);
        let z0 = z - (k - t);

        let (i1, j1, k1, i2, j2, k2) = if x0 >= y0 {
            if y0 >= z0 {
                (1, 0, 0, 1, 1, 0)
            } else if x0 >= z0 {
                (1, 0, 0, 1, 0, 1)
            } else {
                (0, 0, 1, 1, 0, 1)
            }
        } else if y0 < z0 {
            (0, 0, 1, 0, 1, 1)
        } else if x0 < z0 {
            (0, 1, 0, 0, 1, 1)
        } else {
            (0, 1, 0, 1, 1, 0)
        };

        let x1 = x0 - i1 as f64 + G3;
        let y1 = y0 - j1 as f64 + G3;
        let z1 = z0 - k1 as f64 + G3;
        let x2 = x0 - i2 as f64 + 2.0 * G3;
        let y2 = y0 - j2 as f64 + 2.0 * G3;
        let z2 = z0 - k2 as f64 + 2.0 * G3;
        let x3 = x0 - 1.0 + 3.0 * G3;
        let y3 = y0 - 1.0 + 3.0 * G3;
        let z3 = z0 - 1.0 + 3.0 * G3;

        let ii = (i as i32) & 255;
        let jj = (j as i32) & 255;
        let kk = (k as i32) & 255;
        let gi0 = (self.hash(ii + self.hash(jj + self.hash(kk))) % 12) as usize;
        let gi1 = (self.hash(ii + i1 + self.hash(jj + j1 + self.hash(kk + k1))) % 12) as usize;
        let gi2 = (self.hash(ii + i2 + self.hash(jj + j2 + self.hash(kk + k2))) % 12) as usize;
        let gi3 = (self.hash(ii + 1 + self.hash(jj + 1 + self.hash(kk + 1))) % 12) as usize;

        let n0 = corner_contribution3(0.6, x0, y0, z0, GRAD3[gi0]);
        let n1 = corner_contribution3(0.6, x1, y1, z1, GRAD3[gi1]);
        let n2 = corner_contribution3(0.6, x2, y2, z2, GRAD3[gi2]);
        let n3 = corner_contribution3(0.6, x3, y3, z3, GRAD3[gi3]);
        32.0 * (n0 + n1 + n2 + n3)
    }

    /// Samples 2D noise at each `xs[i]` for a fixed `y`, writing results
    /// into `out`. Lets a caller precompute an irregular column of
    /// x-coordinates (e.g. LOD-adjusted sample spacing) once and reuse it.
    ///
    /// # Panics
    /// Panics if `xs.len() != out.len()`.
    pub fn sample_row(&self, y: f64, xs: &[f64], out: &mut [f64]) {
        assert_eq!(xs.len(), out.len(), "xs and out must have the same length");
        for i in 0..xs.len() {
            // SAFETY: `i` ranges over `0..xs.len()`, and `out.len() ==
            // xs.len()` was just asserted, so both `get_unchecked` and
            // `get_unchecked_mut` are in bounds for every `i`.
            unsafe {
                let sample = self.noise2(*xs.get_unchecked(i), y);
                *out.get_unchecked_mut(i) = sample;
            }
        }
    }
}

/// Shared 2D corner-contribution formula: a smooth falloff from the
/// corner, zero once the sample point is farther than `radius` away.
#[inline]
fn corner_contribution2(radius: f64, dx: f64, dy: f64, grad: [f64; 3]) -> f64 {
    let t = radius - dx * dx - dy * dy;
    if t < 0.0 { 0.0 } else { (t * t) * (t * t) * dot2(grad, dx, dy) }
}

/// Shared 3D corner-contribution formula.
#[inline]
fn corner_contribution3(radius: f64, dx: f64, dy: f64, dz: f64, grad: [f64; 3]) -> f64 {
    let t = radius - dx * dx - dy * dy - dz * dz;
    if t < 0.0 { 0.0 } else { (t * t) * (t * t) * dot3(grad, dx, dy, dz) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn noise2_is_zero_at_the_origin_for_any_seed() {
        // At (0, 0) the nearest corner's offset is exactly zero (its
        // gradient dot product vanishes) and the other two corners fall
        // outside the simplex's radius, so the sum is zero regardless of
        // the permutation table.
        for seed in [0u64, 1, 123456, 0xABCD] {
            assert_eq!(SimplexNoise::new(seed).noise2(0.0, 0.0), 0.0);
        }
    }

    #[test]
    fn deterministic_for_same_seed() {
        let a = SimplexNoise::new(2024);
        let b = SimplexNoise::new(2024);
        for i in 0..25 {
            let t = i as f64 * 0.29;
            assert_eq!(a.noise2(t, -t * 0.5), b.noise2(t, -t * 0.5));
            assert_eq!(a.noise3(t, t * 0.3, -t), b.noise3(t, t * 0.3, -t));
        }
    }

    #[test]
    fn stays_within_a_sane_bound() {
        let n = SimplexNoise::new(4);
        for i in 0..400 {
            let t = i as f64 * 0.057;
            assert!(n.noise2(t, t * 1.7 - 3.0).abs() <= 1.2);
            assert!(n.noise3(t, t * 0.4, -t * 0.9 + 2.0).abs() <= 1.5);
        }
    }

    #[test]
    fn different_seeds_diverge() {
        let a = SimplexNoise::new(10);
        let b = SimplexNoise::new(20);
        let mut any_diff = false;
        for i in 1..30 {
            let t = i as f64 * 0.13;
            if (a.noise2(t, t) - b.noise2(t, t)).abs() > 1e-9 {
                any_diff = true;
                break;
            }
        }
        assert!(any_diff, "two different seeds produced identical noise fields");
    }

    #[test]
    fn sample_row_matches_direct_calls() {
        let n = SimplexNoise::new(88);
        let xs: Vec<f64> = (0..10).map(|i| i as f64 * 0.31 - 1.0).collect();
        let mut out = vec![0.0; xs.len()];
        n.sample_row(2.5, &xs, &mut out);
        for (i, &x) in xs.iter().enumerate() {
            assert_eq!(out[i], n.noise2(x, 2.5));
        }
    }
}
