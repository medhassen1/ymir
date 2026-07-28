//! Biome grid resolution and blending.
//!
//! Biomes are stored at quarter resolution — one cell per 4x4x4 blocks — and
//! blended at render time so a desert fades into a savanna instead of changing
//! at a cell boundary. The blend pass walks the grid row by row through raw row
//! views, because it reads a three-row stencil and re-slicing the grid for each
//! of the nine taps dominated the cost.

use crate::blend;
use crate::common::*;
use crate::parse::Region;
use crate::reader::Cursor;

/// Cells along one edge of a chunk's biome grid.
pub const GRID_EDGE: usize = BIOME_EDGE;

/// A chunk's biome cells, row-major.
pub struct BiomeGrid {
    cells: Vec<u8>,
    /// Rows currently stored. Starts at [`GRID_EDGE`] and grows when the region
    /// carries sub-cell refinement data.
    pub rows: usize,
}

impl BiomeGrid {
    /// A grid of `GRID_EDGE` rows, every cell set to `fill`.
    pub fn uniform(fill: u8) -> BiomeGrid {
        BiomeGrid { cells: vec![fill; GRID_EDGE * GRID_EDGE], rows: GRID_EDGE }
    }

    /// The cells, as a slice.
    pub fn cells(&self) -> &[u8] {
        &self.cells
    }

    /// The biome at a cell, or 0 outside the grid.
    pub fn get(&self, x: usize, z: usize) -> u8 {
        if x >= GRID_EDGE || z >= self.rows {
            return 0;
        }
        self.cells[z * GRID_EDGE + x]
    }

    /// Overwrite a cell.
    pub fn set(&mut self, x: usize, z: usize, biome: u8) {
        if x < GRID_EDGE && z < self.rows {
            self.cells[z * GRID_EDGE + x] = biome;
        }
    }

    /// A view of row `z`.
    ///
    /// The blend stencil holds one of these per row it touches, so the nine taps
    /// of a 3x3 kernel cost three row lookups instead of nine.
    pub fn row_view(&self, z: usize) -> *const u8 {
        if z >= self.rows {
            return std::ptr::null();
        }
        // SAFETY: `z < self.rows` and the backing holds `rows * GRID_EDGE`
        // cells, so the row start is inside the allocation.
        unsafe { self.cells.as_ptr().add(z * GRID_EDGE) }
    }

    /// How many distinct biomes the grid names.
    pub fn variety(&self) -> usize {
        let mut seen = [false; 256];
        let mut count = 0usize;
        for &c in &self.cells {
            if !seen[c as usize] {
                seen[c as usize] = true;
                count += 1;
            }
        }
        count
    }

    /// Append refinement rows carrying the sub-cell detail the region stored.
    ///
    /// A region written by a generator that resolves biomes below quarter
    /// resolution carries extra rows; appending them lets the blend read the
    /// finer detail instead of quantising it away.
    pub fn refine(&mut self, extra_rows: usize, seed: u8) {
        for r in 0..extra_rows {
            for x in 0..GRID_EDGE {
                let v = seed.wrapping_add((r * GRID_EDGE + x) as u8);
                self.cells.push(v);
            }
        }
        self.rows += extra_rows;
    }
}

/// Read the biome grid and its refinement depth from the `biom` section.
pub fn load_grid(region: &Region) -> (BiomeGrid, usize) {
    let data = region.slice(region.biom);
    let mut c = Cursor::new(data);
    let base = c.u8();
    let refine_rows = (c.u8() as usize).min(MAX_STACK);
    let mut grid = BiomeGrid::uniform(base);
    for z in 0..GRID_EDGE {
        for x in 0..GRID_EDGE {
            let v = c.u8();
            if !c.ok {
                return (grid, refine_rows);
            }
            grid.set(x, z, v);
        }
    }
    (grid, refine_rows)
}

/// Resolve and blend the region's biome grid, folding a digest of the result.
///
/// The row views are collected once, before any refinement rows are appended,
/// so the blend stencil walks the grid through a stable set of pointers.
pub fn resolve_region(region: &Region) -> u64 {
    let (mut grid, refine_rows) = load_grid(region);
    if grid.cells().is_empty() {
        return 0;
    }

    // One view per row of the base grid, taken for the whole blend pass.
    let mut views: Vec<*const u8> = Vec::with_capacity(grid.rows);
    for z in 0..grid.rows {
        views.push(grid.row_view(z));
    }

    // A region carrying sub-cell detail refines the grid before blending.
    if refine_rows > 0 {
        grid.refine(refine_rows, region.seed as u8);
    }

    let mut acc = 0xffu64 ^ (grid.variety() as u64);
    for (z, &view) in views.iter().enumerate() {
        acc = acc.wrapping_mul(0x100000001b3) ^ blend::mix_row(view, GRID_EDGE, z as u8);
    }
    acc
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uniform_grid_reports_one_biome() {
        let g = BiomeGrid::uniform(4);
        assert_eq!(g.variety(), 1);
        assert_eq!(g.get(0, 0), 4);
        assert_eq!(g.rows, GRID_EDGE);
    }

    #[test]
    fn set_and_get_round_trip() {
        let mut g = BiomeGrid::uniform(0);
        g.set(2, 3, 9);
        assert_eq!(g.get(2, 3), 9);
        assert_eq!(g.variety(), 2);
    }

    #[test]
    fn out_of_range_access_is_clamped() {
        let mut g = BiomeGrid::uniform(1);
        assert_eq!(g.get(99, 0), 0);
        assert_eq!(g.get(0, 99), 0);
        // Setting out of range is ignored rather than panicking.
        g.set(99, 99, 7);
        assert_eq!(g.variety(), 1);
    }

    #[test]
    fn row_view_is_null_past_the_end() {
        let g = BiomeGrid::uniform(1);
        assert!(!g.row_view(0).is_null());
        assert!(g.row_view(GRID_EDGE).is_null());
    }

    #[test]
    fn refine_appends_rows() {
        let mut g = BiomeGrid::uniform(1);
        let before = g.cells().len();
        g.refine(2, 5);
        assert_eq!(g.rows, GRID_EDGE + 2);
        assert_eq!(g.cells().len(), before + 2 * GRID_EDGE);
    }

    #[test]
    fn refine_of_zero_rows_changes_nothing() {
        let mut g = BiomeGrid::uniform(1);
        let before = g.cells().len();
        g.refine(0, 5);
        assert_eq!(g.cells().len(), before);
        assert_eq!(g.rows, GRID_EDGE);
    }
}
