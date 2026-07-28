//! Structure template instancing.
//!
//! A structure — a village house, a dungeon room — is stored once as a template
//! and stamped into the world at each of its placements, with a rotation and an
//! offset. Templates may nest: a village is a template whose instances are
//! themselves templates. Each resolved template is packed into its own buffer
//! and handed to the stamper by pointer, so a placement that repeats a template
//! fifty times stamps from one copy of its cells.

use crate::common::*;
use crate::graft;
use crate::parse::Region;
use crate::reader::Cursor;
use crate::format::NESTED_TEMPLATE;

/// A template smaller than this resolved to nothing useful and is released
/// before the stamping pass rather than being carried through it.
pub const MIN_TEMPLATE_CELLS: usize = 3;

/// One cell of a resolved template.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Cell {
    /// Block state to stamp.
    pub state: u16,
    /// Vertical offset from the instance origin.
    pub dy: i16,
}

/// Where and how a template is stamped.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Placement {
    /// World X of the instance origin.
    pub x: i16,
    /// World Y of the instance origin.
    pub y: i16,
    /// World Z of the instance origin.
    pub z: i16,
    /// Quarter turns about the vertical axis.
    pub rot: u8,
}

/// A staged template: its cells, named by pointer, plus where to stamp them.
type Staged = (*const Cell, usize, Placement);

/// Decode one template's cells, recursing through nested template lists.
fn decode_template(
    c: &mut Cursor,
    depth: u32,
    budget: &mut usize,
) -> Result<(Vec<Cell>, Placement), Status> {
    if depth > MAX_TEMPLATE_DEPTH {
        return Err(Status::DepthExceeded);
    }
    if *budget == 0 {
        return Err(Status::Truncated);
    }
    *budget -= 1;

    let template_id = c.u16();
    let x = c.i16();
    let y = c.i16();
    let z = c.i16();
    let rot = c.u8();
    let cell_count = c.u16() as usize;
    if !c.ok {
        return Err(Status::Truncated);
    }
    let placement = Placement { x, y, z, rot: rot & 3 };

    if template_id == NESTED_TEMPLATE {
        // A nested template contributes its children's cells, shifted by its
        // own origin, which is how a village aggregates its houses.
        let mut cells = Vec::new();
        for _ in 0..cell_count.min(32) {
            let (child, child_at) = decode_template(c, depth + 1, budget)?;
            for cell in child {
                cells.push(Cell {
                    state: cell.state,
                    dy: cell.dy.wrapping_add(child_at.y),
                });
            }
        }
        return Ok((cells, placement));
    }

    let mut cells = Vec::with_capacity(cell_count.min(256));
    for _ in 0..cell_count.min(256) {
        let state = c.u16();
        let dy = c.i16();
        if !c.ok {
            return Err(Status::Truncated);
        }
        cells.push(Cell { state, dy });
    }
    Ok((cells, placement))
}

/// Validate the region's structure section without stamping anything.
///
/// Called by [`crate::verify`], which checks nesting independently of the
/// columns because a template list can be malformed on its own.
pub fn validate(region: &Region) -> Status {
    let data = region.slice(region.strc);
    if data.is_empty() {
        return Status::Ok;
    }
    let mut c = Cursor::new(data);
    let count = c.u16() as usize;
    if !c.ok {
        return Status::Truncated;
    }
    let mut budget = 256usize;
    let depth_cap = (region.depth as u32).min(MAX_TEMPLATE_DEPTH);
    for _ in 0..count.min(MAX_STACK) {
        match decode_template(&mut c, MAX_TEMPLATE_DEPTH - depth_cap, &mut budget) {
            Ok(_) => {}
            Err(Status::Truncated) => break,
            Err(e) => return e,
        }
    }
    Status::Ok
}

/// Instance every structure template into the region and fold a digest.
///
/// Templates are staged first — each packed into its own buffer and recorded by
/// pointer — and stamped in a second pass, so a template placed many times is
/// resolved once. Templates that resolved to fewer than [`MIN_TEMPLATE_CELLS`]
/// cells contributed nothing and their buffers are released before the stamp.
pub fn instance_region(region: &Region, n: usize) -> u64 {
    let data = region.slice(region.strc);
    if data.is_empty() {
        return 0;
    }
    let mut c = Cursor::new(data);
    let count = c.u16() as usize;
    if !c.ok {
        return 0;
    }

    let mut budget = 256usize;
    let depth_cap = (region.depth as u32).min(MAX_TEMPLATE_DEPTH);
    let start_depth = MAX_TEMPLATE_DEPTH - depth_cap;

    let mut storage: Vec<Box<[Cell]>> = Vec::new();
    let mut staged: Vec<Staged> = Vec::new();

    for _ in 0..count.min(MAX_STACK) {
        let (cells, placement) = match decode_template(&mut c, start_depth, &mut budget) {
            Ok(t) => t,
            Err(_) => break,
        };
        let packed = cells.into_boxed_slice();
        // Stage the template by pointer; the buffer itself is kept alive in
        // `storage` for the duration of the stamping pass.
        staged.push((packed.as_ptr(), packed.len(), placement));
        storage.push(packed);
    }

    // Release the buffers of templates that resolved to nothing useful, so a
    // region full of degenerate placements does not hold their storage through
    // the stamp.
    storage.retain(|b| b.len() >= MIN_TEMPLATE_CELLS);

    graft::stamp_all(&staged, region.seed, n)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn template_bytes(id: u16, cells: &[(u16, i16)]) -> Vec<u8> {
        let mut v = Vec::new();
        v.extend_from_slice(&id.to_be_bytes());
        v.extend_from_slice(&1i16.to_be_bytes()); // x
        v.extend_from_slice(&2i16.to_be_bytes()); // y
        v.extend_from_slice(&3i16.to_be_bytes()); // z
        v.push(1); // rot
        v.extend_from_slice(&(cells.len() as u16).to_be_bytes());
        for &(state, dy) in cells {
            v.extend_from_slice(&state.to_be_bytes());
            v.extend_from_slice(&dy.to_be_bytes());
        }
        v
    }

    #[test]
    fn decodes_a_leaf_template() {
        let bytes = template_bytes(1, &[(5, 0), (6, 1)]);
        let mut c = Cursor::new(&bytes);
        let mut budget = 16usize;
        let (cells, at) = decode_template(&mut c, 0, &mut budget).unwrap();
        assert_eq!(cells, vec![Cell { state: 5, dy: 0 }, Cell { state: 6, dy: 1 }]);
        assert_eq!(at, Placement { x: 1, y: 2, z: 3, rot: 1 });
    }

    #[test]
    fn nested_template_aggregates_children() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&NESTED_TEMPLATE.to_be_bytes());
        bytes.extend_from_slice(&0i16.to_be_bytes());
        bytes.extend_from_slice(&0i16.to_be_bytes());
        bytes.extend_from_slice(&0i16.to_be_bytes());
        bytes.push(0);
        bytes.extend_from_slice(&2u16.to_be_bytes()); // two children
        bytes.extend_from_slice(&template_bytes(1, &[(7, 0)]));
        bytes.extend_from_slice(&template_bytes(2, &[(8, 0)]));

        let mut c = Cursor::new(&bytes);
        let mut budget = 16usize;
        let (cells, _) = decode_template(&mut c, 0, &mut budget).unwrap();
        // Each child's cells are shifted by that child's own y origin (2).
        assert_eq!(cells.len(), 2);
        assert_eq!(cells[0].state, 7);
        assert_eq!(cells[0].dy, 2);
    }

    #[test]
    fn depth_budget_stops_runaway_nesting() {
        let mut bytes = Vec::new();
        for _ in 0..12 {
            bytes.extend_from_slice(&NESTED_TEMPLATE.to_be_bytes());
            bytes.extend_from_slice(&0i16.to_be_bytes());
            bytes.extend_from_slice(&0i16.to_be_bytes());
            bytes.extend_from_slice(&0i16.to_be_bytes());
            bytes.push(0);
            bytes.extend_from_slice(&1u16.to_be_bytes());
        }
        let mut c = Cursor::new(&bytes);
        let mut budget = 256usize;
        assert_eq!(decode_template(&mut c, 0, &mut budget), Err(Status::DepthExceeded));
    }

    #[test]
    fn work_budget_stops_wide_nesting() {
        let bytes = template_bytes(1, &[(1, 0)]);
        let mut c = Cursor::new(&bytes);
        let mut budget = 0usize;
        assert_eq!(decode_template(&mut c, 0, &mut budget), Err(Status::Truncated));
    }

    #[test]
    fn truncated_template_is_reported() {
        let bytes = [0u8, 1, 0];
        let mut c = Cursor::new(&bytes);
        let mut budget = 16usize;
        assert_eq!(decode_template(&mut c, 0, &mut budget), Err(Status::Truncated));
    }

    #[test]
    fn rotation_is_masked_to_quarter_turns() {
        let mut bytes = template_bytes(1, &[(1, 0)]);
        bytes[8] = 7; // rot byte
        let mut c = Cursor::new(&bytes);
        let mut budget = 16usize;
        let (_, at) = decode_template(&mut c, 0, &mut budget).unwrap();
        assert_eq!(at.rot, 3);
    }
}
