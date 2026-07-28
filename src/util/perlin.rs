//! Classic 3D Perlin (gradient) noise.
//!
//! Terrain height fields, cave density, biome blending, and ore vein
//! placement in a voxel world all want a smooth, deterministic function of
//! world-space coordinates that looks organic rather than blocky. Perlin
//! noise is the workhorse for that: seeded once per world, it can be
//! sampled at any `(x, y, z)` — including fractional coordinates used by
//! [`crate::util::fbm`] for multi-octave terrain — and always returns the
//! same value for the same input.

/// Smooths a value in `[0, 1]` with Ken Perlin's improved fade curve
/// (`6t^5 - 15t^4 + 10t^3`), which has zero first and second derivatives
/// at both endpoints so noise tiles together without visible seams.
#[inline]
fn fade(t: f64) -> f64 {
    t * t * t * (t * (t * 6.0 - 15.0) + 10.0)
}

/// Linearly interpolates between `a` and `b` by `t` in `[0, 1]`.
#[inline]
fn lerp(t: f64, a: f64, b: f64) -> f64 {
    a + t * (b - a)
}

/// Computes the dot product of a pseudo-random gradient direction (chosen
/// by the low 4 bits of `hash`) with the offset vector `(x, y, z)`, per
/// Ken Perlin's 2002 "improved noise" reference gradient set.
#[inline]
fn grad(hash: i32, x: f64, y: f64, z: f64) -> f64 {
    let h = hash & 15;
    let u = if h < 8 { x } else { y };
    let v = if h < 4 {
        y
    } else if h == 12 || h == 14 {
        x
    } else {
        z
    };
    (if h & 1 == 0 { u } else { -u }) + (if h & 2 == 0 { v } else { -v })
}

/// One step of a small SplitMix64-style mixer, used only to seed this
/// module's permutation shuffle. Kept private and self-contained rather
/// than depending on [`crate::util::splitmix64`].
fn mix_step(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// A seeded classic-Perlin-noise sampler. Holds a 512-entry permutation
/// table (a 0..256 permutation duplicated once) so lattice-corner lookups
/// never need to wrap the index by hand.
#[derive(Debug, Clone)]
pub struct PerlinNoise {
    perm: [u8; 512],
}

impl PerlinNoise {
    /// Builds a sampler whose permutation table is a deterministic
    /// shuffle of `0..256` driven by `seed`. Every seed produces a valid
    /// permutation (a bijection on `0..256`), so gradient selection never
    /// degenerates.
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
        PerlinNoise { perm }
    }

    /// Looks up the permutation table at `i`, wrapping via a bitmask
    /// rather than a branch. `i` may be as large as roughly `510` during
    /// lattice-corner accumulation (two chained lookups of values already
    /// bounded by `0..256`), which the 512-entry table absorbs directly.
    #[inline]
    fn hash(&self, i: i32) -> i32 {
        let idx = (i & 0x1FF) as usize;
        // SAFETY: `idx` is masked with `0x1FF` (511), so it is always in
        // `0..512`, matching `self.perm`'s fixed length of 512.
        unsafe { *self.perm.get_unchecked(idx) as i32 }
    }

    /// Samples 3D Perlin noise at `(x, y, z)`. The result is smooth and
    /// deterministic, nominally in `[-1, 1]` (classic Perlin noise can
    /// slightly overshoot that range in 3D, but not by much).
    pub fn noise(&self, x: f64, y: f64, z: f64) -> f64 {
        let xf0 = x.floor();
        let yf0 = y.floor();
        let zf0 = z.floor();
        let xi = (xf0 as i32) & 255;
        let yi = (yf0 as i32) & 255;
        let zi = (zf0 as i32) & 255;
        let xf = x - xf0;
        let yf = y - yf0;
        let zf = z - zf0;
        let u = fade(xf);
        let v = fade(yf);
        let w = fade(zf);

        let a = self.hash(xi) + yi;
        let aa = self.hash(a) + zi;
        let ab = self.hash(a + 1) + zi;
        let b = self.hash(xi + 1) + yi;
        let ba = self.hash(b) + zi;
        let bb = self.hash(b + 1) + zi;

        lerp(
            w,
            lerp(
                v,
                lerp(
                    u,
                    grad(self.hash(aa), xf, yf, zf),
                    grad(self.hash(ba), xf - 1.0, yf, zf),
                ),
                lerp(
                    u,
                    grad(self.hash(ab), xf, yf - 1.0, zf),
                    grad(self.hash(bb), xf - 1.0, yf - 1.0, zf),
                ),
            ),
            lerp(
                v,
                lerp(
                    u,
                    grad(self.hash(aa + 1), xf, yf, zf - 1.0),
                    grad(self.hash(ba + 1), xf - 1.0, yf, zf - 1.0),
                ),
                lerp(
                    u,
                    grad(self.hash(ab + 1), xf, yf - 1.0, zf - 1.0),
                    grad(self.hash(bb + 1), xf - 1.0, yf - 1.0, zf - 1.0),
                ),
            ),
        )
    }

    /// Fills `out` with noise samples over a `width`-by-`height` grid on
    /// the `y = y0` plane, stepping by `step` in both world-space axes.
    /// This is the batch entry point a chunk generator would use to fill
    /// an entire heightmap row without a virtual call per sample.
    ///
    /// # Panics
    /// Panics if `out.len() != width * height`.
    // Each of `x0`/`y0`/`z0`, `width`/`height`, `step`, and `out` is an
    // independently meaningful part of "fill this grid of world-space
    // samples"; bundling them into a params struct would just move the
    // same seven values one level of indirection away from the call site.
    #[allow(clippy::too_many_arguments)]
    pub fn sample_plane(&self, x0: f64, z0: f64, y0: f64, width: usize, height: usize, step: f64, out: &mut [f64]) {
        assert_eq!(out.len(), width * height, "output buffer must hold width*height samples");
        for row in 0..height {
            for col in 0..width {
                let idx = row * width + col;
                let sample = self.noise(x0 + col as f64 * step, y0, z0 + row as f64 * step);
                // SAFETY: `idx = row * width + col` with `row < height` and
                // `col < width` implies `idx < height * width`, which was
                // just asserted to equal `out.len()` above, for every
                // iteration of these two loops.
                unsafe {
                    *out.get_unchecked_mut(idx) = sample;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_at_integer_lattice_points() {
        for seed in [0u64, 1, 42, 0xDEAD_BEEF] {
            let n = PerlinNoise::new(seed);
            for (x, y, z) in [(0i32, 0i32, 0i32), (3, -2, 7), (-5, 5, -5), (100, 0, -100)] {
                let v = n.noise(x as f64, y as f64, z as f64);
                assert!(v.abs() < 1e-9, "seed {seed} lattice point ({x},{y},{z}) gave {v}");
            }
        }
    }

    #[test]
    fn deterministic_for_same_seed() {
        let a = PerlinNoise::new(777);
        let b = PerlinNoise::new(777);
        for i in 0..20 {
            let t = i as f64 * 0.37;
            assert_eq!(a.noise(t, t * 1.3, t * 0.7), b.noise(t, t * 1.3, t * 0.7));
        }
    }

    #[test]
    fn different_seeds_diverge() {
        let a = PerlinNoise::new(1);
        let b = PerlinNoise::new(2);
        let mut any_diff = false;
        for i in 1..30 {
            let t = i as f64 * 0.11;
            if (a.noise(t, t, t) - b.noise(t, t, t)).abs() > 1e-9 {
                any_diff = true;
                break;
            }
        }
        assert!(any_diff, "two different seeds produced identical noise fields");
    }

    #[test]
    fn output_stays_within_a_sane_bound() {
        let n = PerlinNoise::new(9);
        for i in 0..500 {
            let t = i as f64 * 0.083;
            let v = n.noise(t, -t * 0.5, t * 2.0 + 1.0);
            assert!(v.abs() <= 1.5, "noise value {v} out of expected bound");
        }
    }

    #[test]
    fn sample_plane_matches_direct_calls() {
        let n = PerlinNoise::new(55);
        let (width, height) = (6usize, 4usize);
        let mut out = vec![0.0; width * height];
        n.sample_plane(1.5, -2.5, 3.0, width, height, 0.25, &mut out);
        for row in 0..height {
            for col in 0..width {
                let expected = n.noise(1.5 + col as f64 * 0.25, 3.0, -2.5 + row as f64 * 0.25);
                assert_eq!(out[row * width + col], expected);
            }
        }
    }

    #[test]
    #[should_panic]
    fn sample_plane_panics_on_wrong_buffer_size() {
        let n = PerlinNoise::new(1);
        let mut out = vec![0.0; 3];
        n.sample_plane(0.0, 0.0, 0.0, 4, 4, 1.0, &mut out);
    }
}
