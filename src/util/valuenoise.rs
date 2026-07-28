//! Value noise: interpolated pseudo-random values at integer lattice
//! points, in 2D and 3D.
//!
//! Value noise is cheaper than gradient (Perlin) noise — one hashed
//! scalar per lattice corner instead of a gradient dot product — at the
//! cost of slightly blockier low-frequency structure. `ymir` uses it for
//! coarse, cheap fields where that trade-off is worth it: biome
//! temperature/humidity maps and other large-scale parameters that get
//! blended smoothly rather than sampled at high frequency.

/// Selects the fade curve used to blend between lattice corners.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Interpolation {
    /// `3t^2 - 2t^3`: cheap, continuous first derivative.
    Smoothstep,
    /// `6t^5 - 15t^4 + 10t^3`: continuous first *and* second derivative
    /// (Ken Perlin's improved fade curve), at the cost of two more
    /// multiplications per axis.
    Quintic,
}

impl Interpolation {
    #[inline]
    fn fade(self, t: f64) -> f64 {
        match self {
            Interpolation::Smoothstep => t * t * (3.0 - 2.0 * t),
            Interpolation::Quintic => t * t * t * (t * (t * 6.0 - 15.0) + 10.0),
        }
    }
}

#[inline]
fn lerp(t: f64, a: f64, b: f64) -> f64 {
    a + t * (b - a)
}

fn mix_step(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// A seeded value-noise sampler backed by a 256-entry permutation table.
#[derive(Debug, Clone)]
pub struct ValueNoise {
    perm: [u8; 256],
}

impl ValueNoise {
    /// Builds a sampler whose permutation table is a deterministic
    /// shuffle of `0..256` driven by `seed`.
    pub fn new(seed: u64) -> Self {
        let mut perm = [0u8; 256];
        for (i, slot) in perm.iter_mut().enumerate() {
            *slot = i as u8;
        }
        let mut state = seed;
        for i in (1..256).rev() {
            let r = mix_step(&mut state);
            let j = (r % (i as u64 + 1)) as usize;
            perm.swap(i, j);
        }
        ValueNoise { perm }
    }

    /// Looks up the permutation table at `i`, wrapping the index into
    /// `0..256` with a bitmask.
    #[inline]
    fn perm_at(&self, i: i32) -> i32 {
        let idx = (i & 255) as usize;
        // SAFETY: `idx` is masked with `255`, so it is always in `0..256`,
        // matching `self.perm`'s fixed length of 256.
        unsafe { *self.perm.get_unchecked(idx) as i32 }
    }

    /// Returns the raw hashed value in `[-1, 1]` assigned to lattice point
    /// `(ix, iy)`. Exposed publicly since callers sometimes want the raw
    /// per-column lattice value (e.g. as a cheap per-chunk-column tag)
    /// without any interpolation.
    pub fn lattice_value2(&self, ix: i32, iy: i32) -> f64 {
        let h = self.perm_at(self.perm_at(ix) + iy);
        (h as f64 / 255.0) * 2.0 - 1.0
    }

    /// The 3D counterpart of [`Self::lattice_value2`].
    pub fn lattice_value3(&self, ix: i32, iy: i32, iz: i32) -> f64 {
        let h = self.perm_at(self.perm_at(self.perm_at(ix) + iy) + iz);
        (h as f64 / 255.0) * 2.0 - 1.0
    }

    /// Samples 2D value noise at `(x, y)`, bilinearly interpolating the
    /// four surrounding lattice corners with the chosen fade curve.
    pub fn noise2(&self, x: f64, y: f64, interp: Interpolation) -> f64 {
        let x0 = x.floor();
        let y0 = y.floor();
        let ix = x0 as i32;
        let iy = y0 as i32;
        let u = interp.fade(x - x0);
        let v = interp.fade(y - y0);

        let c00 = self.lattice_value2(ix, iy);
        let c10 = self.lattice_value2(ix + 1, iy);
        let c01 = self.lattice_value2(ix, iy + 1);
        let c11 = self.lattice_value2(ix + 1, iy + 1);

        lerp(v, lerp(u, c00, c10), lerp(u, c01, c11))
    }

    /// Samples 3D value noise at `(x, y, z)`, trilinearly interpolating
    /// the eight surrounding lattice corners.
    pub fn noise3(&self, x: f64, y: f64, z: f64, interp: Interpolation) -> f64 {
        let x0 = x.floor();
        let y0 = y.floor();
        let z0 = z.floor();
        let (ix, iy, iz) = (x0 as i32, y0 as i32, z0 as i32);
        let u = interp.fade(x - x0);
        let v = interp.fade(y - y0);
        let w = interp.fade(z - z0);

        let c000 = self.lattice_value3(ix, iy, iz);
        let c100 = self.lattice_value3(ix + 1, iy, iz);
        let c010 = self.lattice_value3(ix, iy + 1, iz);
        let c110 = self.lattice_value3(ix + 1, iy + 1, iz);
        let c001 = self.lattice_value3(ix, iy, iz + 1);
        let c101 = self.lattice_value3(ix + 1, iy, iz + 1);
        let c011 = self.lattice_value3(ix, iy + 1, iz + 1);
        let c111 = self.lattice_value3(ix + 1, iy + 1, iz + 1);

        let top = lerp(v, lerp(u, c000, c100), lerp(u, c010, c110));
        let bottom = lerp(v, lerp(u, c001, c101), lerp(u, c011, c111));
        lerp(w, top, bottom)
    }

    /// Samples 2D noise at each `xs[i]` for a fixed `y`, writing results
    /// into `out`.
    ///
    /// # Panics
    /// Panics if `xs.len() != out.len()`.
    pub fn sample_row(&self, y: f64, xs: &[f64], interp: Interpolation, out: &mut [f64]) {
        assert_eq!(xs.len(), out.len(), "xs and out must have the same length");
        for i in 0..xs.len() {
            // SAFETY: `i` ranges over `0..xs.len()`, and `out.len() ==
            // xs.len()` was just asserted, so both accesses are in bounds
            // for every `i`.
            unsafe {
                let sample = self.noise2(*xs.get_unchecked(i), y, interp);
                *out.get_unchecked_mut(i) = sample;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn noise2_at_integer_coordinates_matches_the_raw_lattice_value() {
        // At an integer coordinate the fractional part is 0, so both fade
        // curves return exactly 0 and interpolation collapses to the
        // corner's own lattice value.
        let n = ValueNoise::new(123);
        for interp in [Interpolation::Smoothstep, Interpolation::Quintic] {
            for (x, y) in [(0i32, 0i32), (3, -2), (-7, 9)] {
                let expected = n.lattice_value2(x, y);
                assert_eq!(n.noise2(x as f64, y as f64, interp), expected);
            }
        }
    }

    #[test]
    fn lattice_values_stay_in_unit_range() {
        let n = ValueNoise::new(9);
        for ix in -20..20 {
            for iy in -20..20 {
                let v = n.lattice_value2(ix, iy);
                assert!((-1.0..=1.0).contains(&v));
            }
        }
    }

    #[test]
    fn interpolated_noise_stays_in_unit_range() {
        let n = ValueNoise::new(4242);
        for i in 0..300 {
            let t = i as f64 * 0.037 - 5.0;
            let a = n.noise2(t, -t, Interpolation::Smoothstep);
            let b = n.noise3(t, -t, t * 0.5, Interpolation::Quintic);
            assert!((-1.0..=1.0).contains(&a));
            assert!((-1.0..=1.0).contains(&b));
        }
    }

    #[test]
    fn deterministic_for_same_seed() {
        let a = ValueNoise::new(55);
        let b = ValueNoise::new(55);
        for i in 0..40 {
            let t = i as f64 * 0.21;
            assert_eq!(
                a.noise2(t, t * 0.5, Interpolation::Quintic),
                b.noise2(t, t * 0.5, Interpolation::Quintic)
            );
        }
    }

    #[test]
    fn different_seeds_diverge() {
        let a = ValueNoise::new(1);
        let b = ValueNoise::new(2);
        assert_ne!(a.lattice_value2(5, 5), b.lattice_value2(5, 5));
    }

    #[test]
    fn sample_row_matches_direct_calls() {
        let n = ValueNoise::new(3);
        let xs: Vec<f64> = (0..8).map(|i| i as f64 * 0.4 - 1.0).collect();
        let mut out = vec![0.0; xs.len()];
        n.sample_row(1.0, &xs, Interpolation::Smoothstep, &mut out);
        for (i, &x) in xs.iter().enumerate() {
            assert_eq!(out[i], n.noise2(x, 1.0, Interpolation::Smoothstep));
        }
    }
}
