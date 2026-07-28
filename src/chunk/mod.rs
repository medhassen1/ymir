//! Decode a `cdat` record into a chunk column of stacked sections.
//!
//! A column is a vertical stack of 16x16x16 sections. Each section carries a
//! block-state palette and a run-length encoded array of palette indices, which
//! is how a mostly-uniform world compresses well: a section of solid stone is
//! one run, and a section of air is flagged empty and carries none at all.
//!
//! Every rebuild stage starts here, so this module is the widest fan-in point in
//! the crate. It deliberately does no interpretation of the block states it
//! decodes — that belongs to [`crate::palette`] and the stages above it.

use crate::common::*;
use crate::format::{sec, INHERIT_PALETTE};
use crate::parse::{chunk_bytes, Region};
use crate::reader::Cursor;

/// One decoded 16x16x16 section of a column.
#[derive(Clone)]
pub struct SectionData {
    /// Block-state ids this section's indices refer into.
    pub palette: Vec<u16>,
    /// Expanded palette indices, one per block, in `y*256 + z*16 + x` order.
    pub blocks: Vec<u16>,
    /// Per-section flags; see [`crate::format::sec`].
    pub flags: u8,
    /// World Y of this section's lowest block.
    pub base_y: i32,
}

impl SectionData {
    /// Whether the mesher can skip this section entirely.
    pub fn is_empty(&self) -> bool {
        self.flags & sec::EMPTY != 0 || self.blocks.is_empty()
    }

    /// Whether every block in the section is the same state.
    pub fn is_uniform(&self) -> bool {
        self.flags & sec::UNIFORM != 0
    }

    /// The palette index at a linear position, or 0 past the end.
    pub fn index_at(&self, linear: usize) -> u16 {
        self.blocks.get(linear).copied().unwrap_or(0)
    }

    /// The block state at a linear position, resolved through the palette.
    pub fn state_at(&self, linear: usize) -> u16 {
        let idx = self.index_at(linear) as usize;
        self.palette.get(idx).copied().unwrap_or(0)
    }

    /// How many blocks in this section are not air (state 0).
    pub fn solid_count(&self) -> usize {
        (0..self.blocks.len()).filter(|&i| self.state_at(i) != 0).count()
    }
}

/// A decoded chunk column.
pub struct Column {
    /// Sections from lowest to highest.
    pub sections: Vec<SectionData>,
    /// World Y of the bottom of the column.
    pub base_y: i32,
    /// Column-level flags carried from the record header.
    pub flags: u8,
    /// The column's index within the region.
    pub cid: usize,
}

impl Column {
    /// Total blocks decoded across every section.
    pub fn block_count(&self) -> usize {
        self.sections.iter().map(|s| s.blocks.len()).sum()
    }

    /// The section covering world height `y`, if the column reaches it.
    pub fn section_for_y(&self, y: i32) -> Option<&SectionData> {
        self.sections
            .iter()
            .find(|s| y >= s.base_y && y < s.base_y + SECTION_EDGE as i32)
    }

    /// Whether any section carries per-block light nibbles.
    pub fn has_light(&self) -> bool {
        self.sections.iter().any(|s| s.flags & sec::HAS_LIGHT != 0)
    }
}

/// Decode chunk column `cid` of `region`.
///
/// An unpopulated column (one whose `cmap` entry is empty) decodes to a column
/// with no sections rather than an error, because a sparse region is normal.
pub fn decode(region: &Region, cid: usize) -> Result<Column, Status> {
    let body = chunk_bytes(region, cid);
    if body.is_empty() {
        return Ok(Column { sections: Vec::new(), base_y: 0, flags: 0, cid });
    }

    let mut c = Cursor::new(body);
    let stack_len = c.u16() as usize;
    let base_y = c.i16() as i32;
    let flags = c.u8();
    let _reserved = c.u8();
    if !c.ok {
        return Err(Status::Truncated);
    }
    if stack_len > MAX_STACK {
        return Err(Status::Malformed);
    }

    let mut sections = Vec::with_capacity(clamp_stack(stack_len));
    // A section flagged SHARED_PALETTE inherits whatever the previous section
    // in the column established, which is how a tall run of one material avoids
    // repeating its palette on every section.
    let mut carried: Vec<u16> = Vec::new();

    for si in 0..stack_len {
        let sec_flags = c.u8();
        let palette_len = c.u16() as usize;
        if !c.ok {
            return Err(Status::Truncated);
        }

        let palette = if palette_len == INHERIT_PALETTE as usize {
            if carried.is_empty() {
                return Err(Status::BadPalette);
            }
            carried.clone()
        } else {
            if palette_len == 0 || palette_len > MAX_PALETTE {
                return Err(Status::BadPalette);
            }
            let mut p = Vec::with_capacity(palette_len);
            for _ in 0..palette_len {
                p.push(c.u16());
            }
            if !c.ok {
                return Err(Status::Truncated);
            }
            carried = p.clone();
            p
        };

        let section_base = base_y + (si * SECTION_EDGE) as i32;

        if sec_flags & sec::EMPTY != 0 {
            sections.push(SectionData {
                palette,
                blocks: Vec::new(),
                flags: sec_flags,
                base_y: section_base,
            });
            continue;
        }

        let blocks = if sec_flags & sec::UNIFORM != 0 {
            // A uniform section stores one index and implies the whole volume.
            let idx = c.u16();
            if !c.ok {
                return Err(Status::Truncated);
            }
            if idx as usize >= palette.len() {
                return Err(Status::BadPalette);
            }
            vec![idx; SECTION_VOLUME]
        } else {
            expand_runs(&mut c, palette.len())?
        };

        sections.push(SectionData { palette, blocks, flags: sec_flags, base_y: section_base });
    }

    Ok(Column { sections, base_y, flags, cid })
}

/// Expand a section's run-length encoded palette indices.
///
/// Runs are `(count, index)` pairs. The expansion stops at [`SECTION_VOLUME`]
/// blocks; a section that encodes fewer is tail-padded with index 0, which is
/// how a partially generated section round-trips.
fn expand_runs(c: &mut Cursor, palette_len: usize) -> Result<Vec<u16>, Status> {
    let run_count = c.u16() as usize;
    if !c.ok {
        return Err(Status::Truncated);
    }
    if run_count > SECTION_VOLUME {
        return Err(Status::Malformed);
    }

    let mut blocks: Vec<u16> = Vec::with_capacity(SECTION_VOLUME);
    for _ in 0..run_count {
        let count = c.u16() as usize;
        let index = c.u16();
        if !c.ok {
            return Err(Status::Truncated);
        }
        if index as usize >= palette_len {
            return Err(Status::BadPalette);
        }
        let room = SECTION_VOLUME - blocks.len();
        let take = count.min(room);
        for _ in 0..take {
            blocks.push(index);
        }
        if blocks.len() == SECTION_VOLUME {
            break;
        }
    }
    while blocks.len() < SECTION_VOLUME {
        blocks.push(0);
    }
    Ok(blocks)
}

/// Decode every column of `region`, skipping those that fail, up to the work cap.
///
/// Stages that need the whole region rather than one column start here.
pub fn decode_all(region: &Region) -> Vec<Column> {
    let n = region.worked_chunks();
    let mut out = Vec::with_capacity(n);
    for cid in 0..n {
        if let Ok(col) = decode(region, cid) {
            out.push(col);
        }
    }
    out
}

/// The linear index of a block within a section.
#[inline]
pub fn linear_index(x: usize, y: usize, z: usize) -> usize {
    (y * SECTION_EDGE + z) * SECTION_EDGE + x
}

/// The `(x, y, z)` of a linear index within a section.
#[inline]
pub fn from_linear(linear: usize) -> (usize, usize, usize) {
    let x = linear % SECTION_EDGE;
    let z = (linear / SECTION_EDGE) % SECTION_EDGE;
    let y = linear / (SECTION_EDGE * SECTION_EDGE);
    (x, y, z)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn linear_index_round_trips() {
        for &(x, y, z) in &[(0, 0, 0), (15, 15, 15), (3, 7, 11), (15, 0, 1)] {
            let l = linear_index(x, y, z);
            assert!(l < SECTION_VOLUME);
            assert_eq!(from_linear(l), (x, y, z));
        }
    }

    #[test]
    fn section_helpers_resolve_through_palette() {
        let s = SectionData {
            palette: vec![0, 42, 7],
            blocks: vec![0, 1, 2, 1],
            flags: 0,
            base_y: 0,
        };
        assert_eq!(s.state_at(0), 0);
        assert_eq!(s.state_at(1), 42);
        assert_eq!(s.state_at(2), 7);
        // Out of range reads are clamped rather than panicking.
        assert_eq!(s.index_at(99), 0);
        assert_eq!(s.solid_count(), 3);
    }

    #[test]
    fn empty_flag_reports_empty() {
        let s = SectionData {
            palette: vec![0],
            blocks: vec![0; SECTION_VOLUME],
            flags: sec::EMPTY,
            base_y: 0,
        };
        assert!(s.is_empty());
    }

    #[test]
    fn expand_runs_pads_to_volume() {
        // one run of 4 blocks of index 1, palette of length 2
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&1u16.to_be_bytes()); // run_count
        bytes.extend_from_slice(&4u16.to_be_bytes()); // count
        bytes.extend_from_slice(&1u16.to_be_bytes()); // index
        let mut c = Cursor::new(&bytes);
        let blocks = expand_runs(&mut c, 2).unwrap();
        assert_eq!(blocks.len(), SECTION_VOLUME);
        assert_eq!(&blocks[..4], &[1, 1, 1, 1]);
        assert_eq!(blocks[4], 0);
    }

    #[test]
    fn expand_runs_clamps_overlong_run() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&1u16.to_be_bytes());
        bytes.extend_from_slice(&0xffffu16.to_be_bytes()); // count far past volume
        bytes.extend_from_slice(&0u16.to_be_bytes());
        let mut c = Cursor::new(&bytes);
        let blocks = expand_runs(&mut c, 1).unwrap();
        assert_eq!(blocks.len(), SECTION_VOLUME);
    }

    #[test]
    fn expand_runs_rejects_index_past_palette() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&1u16.to_be_bytes());
        bytes.extend_from_slice(&1u16.to_be_bytes());
        bytes.extend_from_slice(&5u16.to_be_bytes()); // index 5 into a 2-entry palette
        let mut c = Cursor::new(&bytes);
        assert_eq!(expand_runs(&mut c, 2), Err(Status::BadPalette));
    }

    #[test]
    fn column_section_for_y() {
        let col = Column {
            sections: vec![
                SectionData { palette: vec![0], blocks: vec![], flags: 0, base_y: 0 },
                SectionData { palette: vec![0], blocks: vec![], flags: 0, base_y: 16 },
            ],
            base_y: 0,
            flags: 0,
            cid: 0,
        };
        assert!(col.section_for_y(5).is_some());
        assert_eq!(col.section_for_y(20).map(|s| s.base_y), Some(16));
        assert!(col.section_for_y(100).is_none());
    }
}
