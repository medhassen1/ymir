//! A 16x16 column heightfield: storage, statistics, slope, and smoothing.
//!
//! World generation needs a cheap ground-height summary per chunk before
//! blocks are placed: this stores that grid with min/max/mean, slope,
//! bilinear sampling, and a box-blur pass for erosion-style smoothing.

use std::mem::MaybeUninit;

/// Columns per heightfield edge.
pub const SECTION_EDGE: usize = 16;

const CELLS: usize = SECTION_EDGE * SECTION_EDGE;

/// A flat `SECTION_EDGE x SECTION_EDGE` grid of column heights.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Heightfield {
    heights: [u16; CELLS],
}

impl Heightfield {
    /// A heightfield with every column at height 0.
    pub fn new() -> Self {
        Heightfield { heights: [0; CELLS] }
    }

    #[inline]
    fn index(x: usize, z: usize) -> usize {
        z * SECTION_EDGE + x
    }

    /// Height of column `(x, z)`. Panics if either coordinate is out of range.
    pub fn get(&self, x: usize, z: usize) -> u16 {
        assert!(x < SECTION_EDGE && z < SECTION_EDGE, "column out of range");
        let idx = Self::index(x, z);
        // SAFETY: the assert above guarantees `x < SECTION_EDGE` and
        // `z < SECTION_EDGE`, so `idx = z * SECTION_EDGE + x
        // < SECTION_EDGE * SECTION_EDGE = CELLS`, matching `heights`'s length.
        unsafe { *self.heights.get_unchecked(idx) }
    }

    /// Set the height of column `(x, z)`. Panics if out of range.
    pub fn set(&mut self, x: usize, z: usize, height: u16) {
        assert!(x < SECTION_EDGE && z < SECTION_EDGE, "column out of range");
        let idx = Self::index(x, z);
        // SAFETY: same reasoning as `get`: the assert bounds `x` and `z`,
        // so `idx` is within `0..CELLS`, matching `heights`'s length.
        unsafe {
            *self.heights.get_unchecked_mut(idx) = height;
        }
    }

    /// Lowest column height in the field.
    pub fn min(&self) -> u16 {
        self.heights.iter().copied().min().unwrap_or(0)
    }

    /// Highest column height in the field.
    pub fn max(&self) -> u16 {
        self.heights.iter().copied().max().unwrap_or(0)
    }

    /// Mean column height across the whole field.
    pub fn mean(&self) -> f32 {
        let sum: u32 = self.heights.iter().map(|&h| h as u32).sum();
        sum as f32 / CELLS as f32
    }

    /// The `(dx, dz)` gradient of height at column `(x, z)`, estimated by
    /// central differences against clamped neighbours (edge columns fall
    /// back to a one-sided difference).
    pub fn gradient(&self, x: usize, z: usize) -> [f32; 2] {
        assert!(x < SECTION_EDGE && z < SECTION_EDGE, "column out of range");
        let x0 = x.saturating_sub(1);
        let x1 = (x + 1).min(SECTION_EDGE - 1);
        let z0 = z.saturating_sub(1);
        let z1 = (z + 1).min(SECTION_EDGE - 1);
        let dx = (self.get(x1, z) as f32 - self.get(x0, z) as f32) / (x1 - x0).max(1) as f32;
        let dz = (self.get(x, z1) as f32 - self.get(x, z0) as f32) / (z1 - z0).max(1) as f32;
        [dx, dz]
    }

    /// Slope magnitude at `(x, z)`: the length of the gradient vector.
    pub fn slope(&self, x: usize, z: usize) -> f32 {
        let g = self.gradient(x, z);
        (g[0] * g[0] + g[1] * g[1]).sqrt()
    }

    /// Bilinearly sample the height at fractional coordinates, clamped to
    /// the field's extent.
    pub fn sample_bilinear(&self, x: f32, z: f32) -> f32 {
        let max_coord = (SECTION_EDGE - 1) as f32;
        let x = x.clamp(0.0, max_coord);
        let z = z.clamp(0.0, max_coord);
        let x0 = x.floor() as usize;
        let z0 = z.floor() as usize;
        let x1 = (x0 + 1).min(SECTION_EDGE - 1);
        let z1 = (z0 + 1).min(SECTION_EDGE - 1);
        let fx = x - x0 as f32;
        let fz = z - z0 as f32;

        let h00 = self.get(x0, z0) as f32;
        let h10 = self.get(x1, z0) as f32;
        let h01 = self.get(x0, z1) as f32;
        let h11 = self.get(x1, z1) as f32;

        let top = h00 + (h10 - h00) * fx;
        let bottom = h01 + (h11 - h01) * fx;
        top + (bottom - top) * fz
    }

    /// A single box-blur smoothing pass (each column becomes the mean of
    /// itself and its up-to-4 orthogonal neighbours), the kind of erosion
    /// approximation used to soften a raw noise-generated heightfield.
    pub fn smooth(&self) -> Heightfield {
        let mut out: MaybeUninit<[u16; CELLS]> = MaybeUninit::uninit();
        let base = out.as_mut_ptr() as *mut u16;
        for z in 0..SECTION_EDGE {
            for x in 0..SECTION_EDGE {
                let mut sum = self.get(x, z) as u32;
                let mut count = 1u32;
                if x > 0 {
                    sum += self.get(x - 1, z) as u32;
                    count += 1;
                }
                if x + 1 < SECTION_EDGE {
                    sum += self.get(x + 1, z) as u32;
                    count += 1;
                }
                if z > 0 {
                    sum += self.get(x, z - 1) as u32;
                    count += 1;
                }
                if z + 1 < SECTION_EDGE {
                    sum += self.get(x, z + 1) as u32;
                    count += 1;
                }
                let idx = Self::index(x, z);
                let averaged = (sum / count) as u16;
                // SAFETY: `idx = z * SECTION_EDGE + x` with `x, z` both in
                // `0..SECTION_EDGE` (the two enclosing `for` loops), so
                // `idx` is in `0..CELLS`, the exact size of the array being
                // built, and each `idx` is visited exactly once as `(x, z)`
                // sweeps every cell — so this write cannot go out of bounds
                // or double-write a slot.
                unsafe {
                    base.add(idx).write(averaged);
                }
            }
        }
        // SAFETY: the nested loop above wrote every index in `0..CELLS`
        // exactly once (one write per `(x, z)` pair, covering the full
        // `SECTION_EDGE x SECTION_EDGE` grid), so `out` is fully initialized.
        let heights = unsafe { out.assume_init() };
        Heightfield { heights }
    }
}

impl Default for Heightfield {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn get_set_round_trip() {
        let mut hf = Heightfield::new();
        hf.set(3, 5, 42);
        assert_eq!(hf.get(3, 5), 42);
        assert_eq!(hf.get(0, 0), 0);
    }

    #[test]
    fn min_max_mean_on_known_values() {
        let mut hf = Heightfield::new();
        hf.set(0, 0, 10);
        hf.set(1, 0, 20);
        assert_eq!(hf.min(), 0);
        assert_eq!(hf.max(), 20);
        let expected_mean = (10.0 + 20.0) / CELLS as f32;
        assert!((hf.mean() - expected_mean).abs() < 1e-6);
    }

    #[test]
    fn flat_field_has_zero_gradient_and_slope_everywhere() {
        let mut hf = Heightfield::new();
        for z in 0..SECTION_EDGE {
            for x in 0..SECTION_EDGE {
                hf.set(x, z, 64);
            }
        }
        assert_eq!(hf.gradient(8, 8), [0.0, 0.0]);
        assert_eq!(hf.slope(8, 8), 0.0);
    }

    #[test]
    fn linear_ramp_has_constant_gradient_in_x() {
        let mut hf = Heightfield::new();
        for z in 0..SECTION_EDGE {
            for x in 0..SECTION_EDGE {
                hf.set(x, z, (x * 2) as u16);
            }
        }
        // Interior columns see a centered difference of (2*(x+1) - 2*(x-1)) / 2 = 2.
        let g = hf.gradient(5, 5);
        assert!((g[0] - 2.0).abs() < 1e-5);
        assert!((g[1] - 0.0).abs() < 1e-5);
    }

    #[test]
    fn bilinear_sample_matches_grid_points_and_interpolates_between() {
        let mut hf = Heightfield::new();
        hf.set(2, 2, 0);
        hf.set(3, 2, 10);
        hf.set(2, 3, 0);
        hf.set(3, 3, 10);
        assert!((hf.sample_bilinear(2.0, 2.0) - 0.0).abs() < 1e-5);
        assert!((hf.sample_bilinear(3.0, 2.0) - 10.0).abs() < 1e-5);
        assert!((hf.sample_bilinear(2.5, 2.5) - 5.0).abs() < 1e-5);
    }

    #[test]
    fn smoothing_a_single_spike_lowers_its_neighbours_and_itself() {
        let mut hf = Heightfield::new();
        hf.set(8, 8, 100);
        let smoothed = hf.smooth();
        // The spike's own value averages with its 4 neighbours (all 0), so
        // it must drop below its original height...
        assert!(smoothed.get(8, 8) < 100);
        // ...while an orthogonal neighbour, previously 0, picks up some of
        // the spike's height.
        assert!(smoothed.get(7, 8) > 0);
        // A far-away column untouched by the spike's neighbourhood stays 0.
        assert_eq!(smoothed.get(0, 0), 0);
    }
}
