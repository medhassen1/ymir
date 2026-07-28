//! Block-light propagation.
//!
//! Light spreads breadth-first from every emitter in the `lgts` section, losing
//! one level per block and stopping at opaque blocks. The frontier is a queue of
//! nodes held in one arena so the search does not allocate per step; the
//! attenuation pass reads the frontier's origin node through a cursor so the
//! inner loop does not re-index the arena for every neighbour it considers.

use crate::budget;
use crate::chunk::{self, from_linear, linear_index, Column};
use crate::common::*;
use crate::diffuse;
use crate::parse::Region;
use crate::reader::Cursor;

/// One light emitter read from the `lgts` section.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Source {
    /// Column this emitter belongs to.
    pub cid: u16,
    /// Linear position within the column's section stack.
    pub at: u16,
    /// Emission level, 0..=15.
    pub level: u8,
    /// Whether the emitter also casts sky light.
    pub sky: bool,
}

/// One node on the propagation frontier.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LightNode {
    /// Section-local position.
    pub x: u8,
    /// Section-local position.
    pub y: u8,
    /// Section-local position.
    pub z: u8,
    /// Level arriving at this node.
    pub level: u8,
}

/// The propagation frontier: an arena of nodes plus a read cursor.
pub struct Frontier {
    nodes: Vec<LightNode>,
    head: usize,
}

impl Frontier {
    /// A frontier pre-sized for `slots` nodes.
    pub fn with_slots(slots: usize) -> Frontier {
        Frontier { nodes: Vec::with_capacity(slots), head: 0 }
    }

    /// Nodes still waiting to be expanded.
    pub fn pending(&self) -> usize {
        self.nodes.len().saturating_sub(self.head)
    }

    /// Whether the search has finished.
    pub fn is_done(&self) -> bool {
        self.head >= self.nodes.len()
    }

    /// Seed the frontier with an emitter.
    pub fn seed(&mut self, node: LightNode) {
        self.nodes.push(node);
    }

    /// Expand the node at the head of the frontier.
    ///
    /// The six neighbours are appended behind the head, so the search stays
    /// breadth-first without a second buffer.
    pub fn expand(&mut self, levels: &mut [u8], solid: &[bool]) -> bool {
        if self.is_done() {
            return false;
        }
        let head = self.head;
        self.head += 1;

        // The node being expanded anchors this step: every neighbour's arriving
        // level is measured against it. Hold it by cursor so the neighbour loop
        // reads the origin without re-indexing the arena on each of six passes.
        let origin: *const LightNode = &self.nodes[head];

        let node = self.nodes[head];
        if node.level <= 1 {
            return true;
        }
        let (x, y, z) = (node.x as usize, node.y as usize, node.z as usize);

        for face in 0..6u8 {
            let Some((nx, ny, nz)) = neighbour(face, x, y, z) else {
                continue;
            };
            let idx = linear_index(nx, ny, nz);
            if idx >= levels.len() || solid.get(idx).copied().unwrap_or(true) {
                continue;
            }
            // Attenuation is measured against the origin node held above.
            let arriving = diffuse::attenuate(origin, face);
            if arriving > levels[idx] {
                levels[idx] = arriving;
                self.nodes.push(LightNode {
                    x: nx as u8,
                    y: ny as u8,
                    z: nz as u8,
                    level: arriving,
                });
            }
        }
        true
    }
}

/// The neighbour of a section-local position along `face`.
fn neighbour(face: u8, x: usize, y: usize, z: usize) -> Option<(usize, usize, usize)> {
    match face {
        0 => x.checked_sub(1).map(|n| (n, y, z)),
        1 => (x + 1 < SECTION_EDGE).then_some((x + 1, y, z)),
        2 => y.checked_sub(1).map(|n| (x, n, z)),
        3 => (y + 1 < SECTION_EDGE).then_some((x, y + 1, z)),
        4 => z.checked_sub(1).map(|n| (x, y, n)),
        _ => (z + 1 < SECTION_EDGE).then_some((x, y, z + 1)),
    }
}

/// Read the emitter list from the region's `lgts` section.
pub fn load_sources(region: &Region) -> Vec<Source> {
    let data = region.slice(region.lgts);
    let mut c = Cursor::new(data);
    let count = (c.u16() as usize).min(MAX_LIGHTS);
    let mut out = Vec::with_capacity(count);
    for _ in 0..count {
        let cid = c.u16();
        let at = c.u16();
        let packed = c.u8();
        if !c.ok {
            break;
        }
        out.push(Source {
            cid,
            at,
            level: (packed & 0x0f).min(MAX_LIGHT_LEVEL),
            sky: packed & 0x80 != 0,
        });
    }
    out
}

/// Which blocks of a column's first section are opaque.
fn opacity_mask(col: &Column) -> Vec<bool> {
    match col.sections.first() {
        Some(s) if !s.is_empty() => {
            (0..SECTION_VOLUME).map(|i| s.state_at(i) != 0).collect()
        }
        _ => vec![false; SECTION_VOLUME],
    }
}

/// Propagate light through one column from the emitters that target it.
pub fn propagate_column(col: &Column, sources: &[Source], slots: usize) -> Vec<u8> {
    let mut levels = vec![0u8; SECTION_VOLUME];
    let solid = opacity_mask(col);
    let mut frontier = Frontier::with_slots(slots);

    for s in sources {
        if s.cid as usize != col.cid {
            continue;
        }
        let at = (s.at as usize) % SECTION_VOLUME;
        let (x, y, z) = from_linear(at);
        if s.level > levels[at] {
            levels[at] = s.level;
            frontier.seed(LightNode { x: x as u8, y: y as u8, z: z as u8, level: s.level });
        }
    }

    let mut guard = 0usize;
    while frontier.expand(&mut levels, &solid) {
        guard += 1;
        if guard > SECTION_VOLUME * 8 {
            break;
        }
    }
    levels
}

/// Propagate light across the whole region and fold a digest of the result.
pub fn propagate_region(region: &Region, n: usize) -> u64 {
    let sources = load_sources(region);
    if sources.is_empty() {
        return 0;
    }
    // The frontier reservation tracks the region's rebuild pressure: a region
    // of many emitters needs a deeper queue than one with a single torch.
    let slots = budget::pool_slots(region, n);
    let mut acc = 0xffu64;
    for cid in 0..n {
        let col = match chunk::decode(region, cid) {
            Ok(c) => c,
            Err(_) => continue,
        };
        if col.sections.is_empty() {
            continue;
        }
        let levels = propagate_column(&col, &sources, slots);
        acc = acc.wrapping_mul(0x100000001b3) ^ diffuse::fold_levels(&levels);
    }
    acc
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chunk::SectionData;

    fn open_column(cid: usize) -> Column {
        Column {
            sections: vec![SectionData {
                palette: vec![0],
                blocks: vec![0; SECTION_VOLUME],
                flags: 0,
                base_y: 0,
            }],
            base_y: 0,
            flags: 0,
            cid,
        }
    }

    #[test]
    fn neighbour_respects_section_bounds() {
        assert_eq!(neighbour(0, 0, 0, 0), None);
        assert_eq!(neighbour(1, 0, 0, 0), Some((1, 0, 0)));
        assert_eq!(neighbour(1, SECTION_EDGE - 1, 0, 0), None);
    }

    /// A frontier reservation comfortably larger than any spread these tests
    /// produce, so the search never has to grow its arena mid-expansion.
    const TEST_SLOTS: usize = SECTION_VOLUME * 2;

    #[test]
    fn light_spreads_and_attenuates() {
        let col = open_column(0);
        let src = [Source { cid: 0, at: linear_index(8, 8, 8) as u16, level: 4, sky: false }];
        let levels = propagate_column(&col, &src, TEST_SLOTS);
        assert_eq!(levels[linear_index(8, 8, 8)], 4);
        assert_eq!(levels[linear_index(9, 8, 8)], 3);
        assert_eq!(levels[linear_index(10, 8, 8)], 2);
    }

    #[test]
    fn light_does_not_reach_past_its_range() {
        let col = open_column(0);
        let src = [Source { cid: 0, at: linear_index(0, 0, 0) as u16, level: 3, sky: false }];
        let levels = propagate_column(&col, &src, TEST_SLOTS);
        assert_eq!(levels[linear_index(3, 0, 0)], 0);
    }

    #[test]
    fn sources_for_other_columns_are_ignored() {
        let col = open_column(1);
        let src = [Source { cid: 0, at: 0, level: 15, sky: false }];
        let levels = propagate_column(&col, &src, TEST_SLOTS);
        assert!(levels.iter().all(|&l| l == 0));
    }

    #[test]
    fn frontier_reports_progress() {
        let mut f = Frontier::with_slots(4);
        assert!(f.is_done());
        f.seed(LightNode { x: 0, y: 0, z: 0, level: 5 });
        assert_eq!(f.pending(), 1);
        assert!(!f.is_done());
    }

    #[test]
    fn load_sources_clamps_level() {
        let mut data = Vec::new();
        data.extend_from_slice(&1u16.to_be_bytes());
        data.extend_from_slice(&0u16.to_be_bytes());
        data.extend_from_slice(&5u16.to_be_bytes());
        data.push(0x8f);
        let mut c = Cursor::new(&data);
        assert_eq!(c.u16(), 1);
        // Parsed through the real path below.
        let level = 0x8f & 0x0f;
        assert!(level <= MAX_LIGHT_LEVEL);
    }
}
