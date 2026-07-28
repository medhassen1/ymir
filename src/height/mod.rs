//! Surface heightmap rebuild.
//!
//! The heightmap records, for every column of a chunk, the world Y of the
//! highest non-air block. It drives sky-light seeding and terrain queries, so it
//! is rebuilt whenever a region's blocks change. Columns are stored in one flat
//! array and the profiler reads them through a cursor, because the smoothing
//! pass touches each column several times and re-indexing dominated the profile.

use crate::chunk::{self, linear_index, Column};
use crate::common::*;
use crate::parse::Region;
use crate::profile;
use crate::reader::Cursor;

/// Columns along one chunk edge.
pub const MAP_EDGE: usize = SECTION_EDGE;
/// Columns in one chunk's heightmap.
pub const MAP_AREA: usize = MAP_EDGE * MAP_EDGE;

/// A chunk's surface heightmap.
pub struct HeightMap {
    columns: Vec<u16>,
    /// World Y the map is measured from.
    pub base_y: i32,
}

impl HeightMap {
    /// A flat map at `base_y` with every column empty.
    pub fn flat(base_y: i32) -> HeightMap {
        HeightMap { columns: vec![0u16; MAP_AREA], base_y }
    }

    /// The number of columns held.
    pub fn len(&self) -> usize {
        self.columns.len()
    }

    /// Whether the map holds no columns.
    pub fn is_empty(&self) -> bool {
        self.columns.is_empty()
    }

    /// The recorded height of a column.
    pub fn get(&self, x: usize, z: usize) -> u16 {
        self.columns.get(z * MAP_EDGE + x).copied().unwrap_or(0)
    }

    /// Record a column's height, keeping the greater of the two.
    pub fn raise(&mut self, x: usize, z: usize, y: u16) {
        if let Some(slot) = self.columns.get_mut(z * MAP_EDGE + x) {
            if y > *slot {
                *slot = y;
            }
        }
    }

    /// A cursor onto the column array.
    ///
    /// The smoothing and profiling passes read columns through this rather than
    /// bounds-checking each of the four neighbours they touch per column.
    pub fn cursor(&self) -> *const u16 {
        self.columns.as_ptr()
    }

    /// The columns, as a slice.
    pub fn columns(&self) -> &[u16] {
        &self.columns
    }

    /// The tallest column recorded.
    pub fn peak(&self) -> u16 {
        self.columns.iter().copied().max().unwrap_or(0)
    }

    /// Extend the map upward so it can record heights in a taller world.
    ///
    /// A region whose columns reach past the current span needs more resolution
    /// than the flat map allocated, so the array grows to carry the overflow
    /// rows the taller sections occupy.
    pub fn extend_span(&mut self, extra_rows: usize) {
        let want = MAP_AREA + extra_rows * MAP_EDGE;
        if want > self.columns.len() {
            self.columns.resize(want, 0);
        }
    }
}

/// Scan one column's sections and record the highest solid block per position.
pub fn scan_column(col: &Column) -> HeightMap {
    let mut map = HeightMap::flat(col.base_y);
    for s in &col.sections {
        if s.is_empty() {
            continue;
        }
        for y in 0..SECTION_EDGE {
            for z in 0..SECTION_EDGE {
                for x in 0..SECTION_EDGE {
                    if s.state_at(linear_index(x, y, z)) != 0 {
                        let world_y = s.base_y + y as i32;
                        let rel = (world_y - col.base_y).max(0) as u16;
                        map.raise(x, z, rel);
                    }
                }
            }
        }
    }
    map
}

/// How many extra rows a column's sections need beyond the flat map.
///
/// A column whose stack reaches past one section's worth of height records its
/// overflow in additional rows, so the profiler can distinguish a tall spire
/// from a clipped one.
pub fn overflow_rows(col: &Column, world_height: usize) -> usize {
    let span = col.sections.len() * SECTION_EDGE;
    if span <= SECTION_EDGE {
        return 0;
    }
    let over = (span - SECTION_EDGE) / SECTION_EDGE;
    over.min(world_height / SECTION_EDGE).min(MAX_STACK)
}

/// Rebuild the heightmap for every column and fold a digest of the result.
///
/// The profile cursor is taken once per column, before the map is extended for
/// the column's overflow rows, so the smoothing pass reads every neighbour
/// through one pointer.
pub fn rebuild_region(region: &Region, n: usize) -> u64 {
    // The `hgts` section supplies a per-region bias applied to every column.
    let bias = {
        let mut c = Cursor::new(region.slice(region.hgts));
        c.i16() as i32
    };

    let mut acc = 0xffu64 ^ (bias as i64 as u64);
    for cid in 0..n {
        let col = match chunk::decode(region, cid) {
            Ok(c) => c,
            Err(_) => continue,
        };
        if col.sections.is_empty() {
            continue;
        }
        let mut map = scan_column(&col);

        // The profile reads neighbouring columns through this cursor.
        let cursor = map.cursor();

        // Tall columns record their overflow in extra rows before profiling.
        let extra = overflow_rows(&col, region.world_height);
        if extra > 0 {
            map.extend_span(extra);
        }

        acc = acc.wrapping_mul(0x100000001b3)
            ^ profile::slope_sum(cursor, MAP_EDGE, bias);
    }
    acc
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chunk::SectionData;

    fn column_with(sections: usize, solid: bool) -> Column {
        let secs = (0..sections)
            .map(|i| SectionData {
                palette: vec![0, 1],
                blocks: vec![if solid { 1 } else { 0 }; SECTION_VOLUME],
                flags: 0,
                base_y: (i * 16) as i32,
            })
            .collect();
        Column { sections: secs, base_y: 0, flags: 0, cid: 0 }
    }

    #[test]
    fn flat_map_starts_empty() {
        let m = HeightMap::flat(0);
        assert_eq!(m.len(), MAP_AREA);
        assert_eq!(m.peak(), 0);
        assert!(!m.is_empty());
    }

    #[test]
    fn raise_keeps_the_greater_height() {
        let mut m = HeightMap::flat(0);
        m.raise(1, 2, 10);
        m.raise(1, 2, 4);
        assert_eq!(m.get(1, 2), 10);
        m.raise(1, 2, 12);
        assert_eq!(m.get(1, 2), 12);
    }

    #[test]
    fn get_out_of_range_is_zero() {
        let m = HeightMap::flat(0);
        assert_eq!(m.get(99, 99), 0);
    }

    #[test]
    fn scan_records_the_top_of_a_solid_column() {
        let col = column_with(2, true);
        let m = scan_column(&col);
        // Two sections of solid blocks: the top is at relative y 31.
        assert_eq!(m.get(0, 0), 31);
        assert_eq!(m.peak(), 31);
    }

    #[test]
    fn scan_of_air_records_nothing() {
        let col = column_with(1, false);
        let m = scan_column(&col);
        assert_eq!(m.peak(), 0);
    }

    #[test]
    fn overflow_rows_tracks_stack_height() {
        assert_eq!(overflow_rows(&column_with(1, true), 256), 0);
        assert_eq!(overflow_rows(&column_with(3, true), 256), 2);
        // Bounded by the world height.
        assert_eq!(overflow_rows(&column_with(9, true), 32), 2);
    }

    #[test]
    fn extend_span_only_grows() {
        let mut m = HeightMap::flat(0);
        m.extend_span(2);
        assert_eq!(m.len(), MAP_AREA + 2 * MAP_EDGE);
        m.extend_span(1);
        assert_eq!(m.len(), MAP_AREA + 2 * MAP_EDGE, "extend must not shrink");
    }
}
