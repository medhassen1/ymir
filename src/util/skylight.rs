//! Sky-light column propagation and per-column surface profiles.
//!
//! Sky light enters a column at full strength and is worn down by whatever
//! it passes through, plus the `SECTION_EDGE`-wide surface profile a
//! relight pass queries per column without rescanning the section.

/// Blocks per chunk-section edge; also the width of a [`SkyProfile`] row.
pub const SECTION_EDGE: usize = 16;

/// Highest sky-light level, held by any block with a clear view of the sky.
pub const MAX_LIGHT_LEVEL: u8 = 15;

/// Propagate full-strength sky light down a column of binary opacity flags
/// (`true` = fully opaque), top-first. Light stays at [`MAX_LIGHT_LEVEL`]
/// until the first opaque block, which — like everything below it — gets 0.
pub fn propagate_full(opacity: &[bool]) -> Vec<u8> {
    let mut out = Vec::with_capacity(opacity.len());
    let mut blocked = false;
    for &opaque in opacity {
        if blocked || opaque {
            out.push(0);
            blocked = true;
        } else {
            out.push(MAX_LIGHT_LEVEL);
        }
    }
    out
}

/// Depth (0-based, from the top of the slice) of the first opaque block in
/// a column, or `None` if the column has a clear view of the sky.
pub fn first_opaque_depth(opacity: &[bool]) -> Option<usize> {
    opacity.iter().position(|&opaque| opaque)
}

/// Propagate sky light down a column with partial occlusion: `opacity[y]`
/// is how many levels that block removes from the light passing through it
/// (0 = fully transparent, [`MAX_LIGHT_LEVEL`] = fully opaque). Light
/// accumulates occlusion top-down and saturates at 0 once fully absorbed.
pub fn propagate_occluded(opacity: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(opacity.len());
    let mut level = MAX_LIGHT_LEVEL;
    for &op in opacity {
        level = level.saturating_sub(op);
        out.push(level);
    }
    out
}

/// Same rule as [`propagate_occluded`], specialized to a fixed-width row of
/// exactly `SECTION_EDGE` blocks — the common case of relighting one
/// vertical strip of a section without touching the heap.
pub fn propagate_occluded_row(opacity: &[u8; SECTION_EDGE]) -> [u8; SECTION_EDGE] {
    let mut out = [0u8; SECTION_EDGE];
    let mut level = MAX_LIGHT_LEVEL;
    for (i, &op) in opacity.iter().enumerate() {
        level = level.saturating_sub(op);
        // SAFETY: `i` comes from enumerating `opacity`, a `[u8; SECTION_EDGE]`
        // array, so `i` ranges over `0..SECTION_EDGE` — the same length as
        // `out` — and is always a valid index into it.
        unsafe {
            *out.get_unchecked_mut(i) = level;
        }
    }
    out
}

/// A per-column sky-light surface summary for one `SECTION_EDGE`-wide row:
/// how deep the first opaque block sits in each column, and the light level
/// immediately above that surface.
pub struct SkyProfile {
    /// Depth of the first opaque block in each column, or the column's full
    /// length if it is open to the sky.
    surface_depth: [usize; SECTION_EDGE],
    /// Sky light level immediately above each column's surface.
    surface_light: [u8; SECTION_EDGE],
}

impl SkyProfile {
    /// Build a profile from `SECTION_EDGE` independent opacity columns.
    pub fn build(columns: &[Vec<bool>; SECTION_EDGE]) -> Self {
        let mut depth = [0usize; SECTION_EDGE];
        let mut light = [0u8; SECTION_EDGE];
        for (i, column) in columns.iter().enumerate() {
            let d = first_opaque_depth(column).unwrap_or(column.len());
            depth[i] = d;
            // A column open to the sky (or whose surface is below the top
            // block) still sees full sky light right at its surface; only a
            // column that is opaque at depth 0 sees none.
            light[i] = if d == 0 { 0 } else { MAX_LIGHT_LEVEL };
        }
        SkyProfile {
            surface_depth: depth,
            surface_light: light,
        }
    }

    /// Surface depth of column `x`. Panics if `x >= SECTION_EDGE`.
    pub fn depth_at(&self, x: usize) -> usize {
        assert!(x < SECTION_EDGE, "column index out of range");
        // SAFETY: the assert above guarantees `x < SECTION_EDGE`, which is
        // exactly the length of `surface_depth`.
        unsafe { *self.surface_depth.get_unchecked(x) }
    }

    /// Surface light level of column `x`. Panics if `x >= SECTION_EDGE`.
    pub fn light_at(&self, x: usize) -> u8 {
        assert!(x < SECTION_EDGE, "column index out of range");
        // SAFETY: the assert above guarantees `x < SECTION_EDGE`, which is
        // exactly the length of `surface_light`.
        unsafe { *self.surface_light.get_unchecked(x) }
    }

    /// Mean surface depth across the row, a cheap horizon estimate.
    pub fn mean_depth(&self) -> f32 {
        self.surface_depth.iter().sum::<usize>() as f32 / SECTION_EDGE as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn propagate_full_stops_at_first_opaque_block() {
        let column = [false, false, true, false, false];
        let light = propagate_full(&column);
        assert_eq!(light, vec![MAX_LIGHT_LEVEL, MAX_LIGHT_LEVEL, 0, 0, 0]);
    }

    #[test]
    fn first_opaque_depth_is_none_for_open_column() {
        let column = [false, false, false];
        assert_eq!(first_opaque_depth(&column), None);
        let column2 = [false, true, false];
        assert_eq!(first_opaque_depth(&column2), Some(1));
    }

    #[test]
    fn propagate_occluded_transparent_column_stays_full() {
        let opacity = [0u8; 6];
        let light = propagate_occluded(&opacity);
        assert!(light.iter().all(|&l| l == MAX_LIGHT_LEVEL));
    }

    #[test]
    fn propagate_occluded_accumulates_and_saturates() {
        let opacity = [1u8, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1];
        let light = propagate_occluded(&opacity);
        // Levels decrease monotonically until they hit 0, then never go
        // negative (saturating_sub), and never increase.
        let mut prev = MAX_LIGHT_LEVEL;
        for &l in &light {
            assert!(l <= prev);
            prev = l;
        }
        assert_eq!(*light.last().unwrap(), 0);
    }

    #[test]
    fn propagate_occluded_row_matches_slice_version() {
        let opacity: [u8; SECTION_EDGE] = std::array::from_fn(|i| (i % 3) as u8);
        let row_result = propagate_occluded_row(&opacity);
        let slice_result = propagate_occluded(&opacity);
        assert_eq!(row_result.to_vec(), slice_result);
    }

    #[test]
    fn sky_profile_reports_correct_depth_and_light() {
        let mut columns: [Vec<bool>; SECTION_EDGE] = Default::default();
        columns[0] = vec![true, false, false]; // opaque right at the top
        columns[1] = vec![false, false, true]; // open, opaque deeper down
        columns[2] = vec![false, false, false]; // fully open
        let profile = SkyProfile::build(&columns);
        assert_eq!(profile.depth_at(0), 0);
        assert_eq!(profile.light_at(0), 0);
        assert_eq!(profile.depth_at(1), 2);
        assert_eq!(profile.light_at(1), MAX_LIGHT_LEVEL);
        assert_eq!(profile.depth_at(2), 3);
        assert_eq!(profile.light_at(2), MAX_LIGHT_LEVEL);
    }

    #[test]
    fn sky_profile_mean_depth_averages_all_columns() {
        let columns: [Vec<bool>; SECTION_EDGE] = std::array::from_fn(|i| vec![i == 0; 1]);
        let profile = SkyProfile::build(&columns);
        // Column 0 has depth 0 (opaque immediately); every other column has
        // depth 1 (a single transparent block, open below).
        let expected = (0.0 + 1.0 * (SECTION_EDGE as f32 - 1.0)) / SECTION_EDGE as f32;
        assert!((profile.mean_depth() - expected).abs() < 1e-6);
    }
}
