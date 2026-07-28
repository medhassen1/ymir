//! Block-light propagation.
//!
//! Light spreads breadth-first from every emitter in the `lgts` section, losing
//! one level per block and stopping at opaque blocks. The frontier is a queue of
//! nodes held in one arena so the search does not allocate per step; the
//! attenuation pass reads the frontier's origin node through a cursor so the
//! inner loop does not re-index the arena for every neighbour it considers.
//!
//! A region's light doesn't stop at a column's edge: a seed bright enough to
//! matter is kept as a cross-column bleed source so a later column can still
//! be measured against it. [`NodeArena`] gives the region a home for every
//! column's expanded frontier that outlives any single column's own search —
//! built once per region, it appends each column's nodes into fixed-size
//! chunks and hands back a pointer into them instead of an owned buffer.
//! Because the arena lives for the whole pass, its memory is bounded
//! separately from any one column's lifetime: once the accumulated node count
//! crosses a threshold, [`NodeArena::compact`] recycles the oldest chunks, the
//! way a lighting engine reclaims cold pages instead of growing without bound
//! on a region with many columns.

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
    /// Steps taken from the emitter that seeded this branch of the search.
    ///
    /// A byte comfortably covers any branch a real search produces — light
    /// levels top out at [`MAX_LIGHT_LEVEL`] and lose at least one level per
    /// step, so a branch is exhausted long before this could wrap.
    pub depth: u8,
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

    /// Every node the search has produced so far, seeds and expansions alike.
    ///
    /// Read once a column's search has finished, so the region pass can commit
    /// the whole frontier into its arena.
    pub fn nodes(&self) -> &[LightNode] {
        &self.nodes
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
        // level is measured against it. The neighbour loop below pushes onto the
        // same arena, which may reallocate it, so the anchor is a copy of the
        // node rather than a cursor into the arena.
        let node = self.nodes[head];
        let origin: *const LightNode = &node;
        if node.level <= 1 {
            return true;
        }
        let (x, y, z) = (node.x as usize, node.y as usize, node.z as usize);

        // The falloff curve caps how far this branch can still reach,
        // independently of the per-face cost applied below. It is sampled
        // once per expansion rather than once per neighbour — hoisted out of
        // the face loop the same way `origin` is above — at this branch's own
        // depth from the emitter that seeded it.
        let curved = diffuse::falloff_at(node.depth, &diffuse::FALLOFF_CURVE);

        for face in 0..6u8 {
            let Some((nx, ny, nz)) = neighbour(face, x, y, z) else {
                continue;
            };
            let idx = linear_index(nx, ny, nz);
            if idx >= levels.len() || solid.get(idx).copied().unwrap_or(true) {
                continue;
            }
            // Attenuation is measured against the origin node held above,
            // then capped by how far the curve says this branch can reach.
            let arriving = diffuse::attenuate(origin, face).min(curved);
            if arriving > levels[idx] {
                levels[idx] = arriving;
                self.nodes.push(LightNode {
                    x: nx as u8,
                    y: ny as u8,
                    z: nz as u8,
                    level: arriving,
                    depth: node.depth.wrapping_add(1),
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

/// Propagate light through one column from the emitters that target it,
/// returning both the resolved levels and the full frontier the search
/// expanded — every seed and every node it queued.
///
/// The region pass commits the returned frontier into its arena so a bright
/// enough column's spread can bleed across the column boundary; a caller that
/// only wants the levels can use [`propagate_column`] instead.
pub fn propagate_column_frontier(
    col: &Column,
    sources: &[Source],
    slots: usize,
) -> (Vec<u8>, Vec<LightNode>) {
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
            frontier.seed(LightNode { x: x as u8, y: y as u8, z: z as u8, level: s.level, depth: 0 });
        }
    }

    let mut guard = 0usize;
    while frontier.expand(&mut levels, &solid) {
        guard += 1;
        if guard > SECTION_VOLUME * 8 {
            break;
        }
    }
    (levels, frontier.nodes().to_vec())
}

/// Propagate light through one column from the emitters that target it.
pub fn propagate_column(col: &Column, sources: &[Source], slots: usize) -> Vec<u8> {
    propagate_column_frontier(col, sources, slots).0
}

/// Nodes held by one [`NodeArena`] chunk.
///
/// An ordinary column's expanded frontier fits with room to spare, so a
/// typical commit never has to look past the chunk it lands in.
const ARENA_CHUNK_NODES: usize = 256;

/// Accumulated resident nodes across an arena's live chunks that triggers
/// [`NodeArena::compact`].
///
/// An ordinary region — a modest number of columns, most spreading from a
/// single torch — never approaches this. A region built from many columns
/// with wide-reaching sources does, which is exactly the case the bound
/// exists to catch: without it, a region-lifetime arena would keep every
/// column's frontier resident for the whole pass no matter how many columns
/// the region carries.
const COMPACT_THRESHOLD_NODES: usize = 2048;

/// A column's frontier at or above this many nodes is wide-reaching enough to
/// be worth keeping as a cross-column bleed source for the region's closing
/// pass.
const RETAIN_MIN_NODES: usize = 96;

/// A region-lifetime arena for committed column frontiers.
///
/// Built once per region rather than once per column, so a pointer handed out
/// while searching one column stays valid while later columns are searched —
/// which is what lets a retained bleed source (see [`propagate_region`]) be
/// read again well after its own column has finished. Nodes are appended into
/// fixed-size chunks, each stored as an exact-sized boxed slice; a chunk with
/// no room left for the next commit is left as-is and a fresh one takes over,
/// so a single commit is never split across two chunks.
struct NodeArena {
    /// Chunks holding committed column frontiers, oldest first.
    chunks: Vec<Box<[LightNode]>>,
    /// Nodes already written into the last chunk.
    used: usize,
    /// Nodes held across all currently resident chunks.
    resident: usize,
}

impl NodeArena {
    fn new() -> NodeArena {
        NodeArena { chunks: Vec::new(), used: 0, resident: 0 }
    }

    /// Commit `nodes` into the arena and return a pointer to where they
    /// landed.
    ///
    /// If what remains of the current chunk cannot hold `nodes`, a fresh
    /// chunk takes over first, so the returned pointer's `nodes.len()`
    /// entries are always contiguous — addressing live memory for as long as
    /// the chunk backing them stays resident (see [`NodeArena::compact`]).
    fn commit(&mut self, nodes: &[LightNode]) -> *const LightNode {
        let len = nodes.len();
        let fits_current = self.chunks.last().is_some_and(|c| self.used + len <= c.len());
        if !fits_current {
            let cap = len.max(ARENA_CHUNK_NODES);
            let filler = LightNode { x: 0, y: 0, z: 0, level: 0, depth: 0 };
            self.chunks.push(vec![filler; cap].into_boxed_slice());
            self.used = 0;
            self.resident += cap;
        }
        let chunk = self.chunks.last_mut().expect("a chunk was just ensured above");
        chunk[self.used..self.used + len].copy_from_slice(nodes);
        // SAFETY: `chunk` is a live `Box<[LightNode]>` at least `self.used +
        // len` entries long — either it already fit `nodes` past `self.used`,
        // or a chunk sized to hold at least `nodes` was just pushed — so this
        // offset and the `len` entries from it lie inside the allocation.
        let ptr = unsafe { chunk.as_ptr().add(self.used) };
        self.used += len;
        ptr
    }

    /// Drop the oldest resident chunks until the arena's accumulated nodes
    /// fall back to `threshold`, or only the chunk currently being written to
    /// is left.
    ///
    /// This is the arena's memory bound: left unchecked, a region-lifetime
    /// arena would keep every column's frontier resident for the whole pass
    /// no matter how many columns the region carries. The chunk currently
    /// being written to is never dropped, since the next commit needs
    /// somewhere to land.
    fn compact(&mut self, threshold: usize) {
        while self.resident > threshold && self.chunks.len() > 1 {
            let oldest = self.chunks.remove(0);
            self.resident -= oldest.len();
        }
    }
}

/// A column's frontier retained past its own turn through the region loop,
/// for the closing cross-column bleed pass to read.
struct RetainedFrontier {
    ptr: *const LightNode,
    len: usize,
}

/// Propagate light across the whole region and fold a digest of the result.
///
/// Each column's levels are folded as they are resolved, then its expanded
/// frontier is committed into a region-wide [`NodeArena`] rather than freed
/// with the column: a column whose search reached far enough keeps its
/// committed frontier registered as a retained bleed source. Once every
/// column has had its turn, the region's closing pass folds every retained
/// frontier once more, letting light bleed across a column boundary beyond
/// the column that produced it.
pub fn propagate_region(region: &Region, n: usize) -> u64 {
    let sources = load_sources(region);
    if sources.is_empty() {
        return 0;
    }
    // The frontier reservation tracks the region's rebuild pressure: a region
    // of many emitters needs a deeper queue than one with a single torch.
    let slots = budget::pool_slots(region, n);
    let mut arena = NodeArena::new();
    let mut retained: Vec<RetainedFrontier> = Vec::new();
    let mut acc = 0xffu64;
    for cid in 0..n {
        let col = match chunk::decode(region, cid) {
            Ok(c) => c,
            Err(_) => continue,
        };
        if col.sections.is_empty() {
            continue;
        }
        let (levels, nodes) = propagate_column_frontier(&col, &sources, slots);
        acc = acc.wrapping_mul(0x100000001b3) ^ diffuse::fold_levels(&levels);

        if nodes.is_empty() {
            continue;
        }
        // Commit this column's expanded frontier into the region's arena.
        // Only past this point is there a pointer stable enough to retain
        // past this column's own scope.
        let ptr = arena.commit(&nodes);

        // A column whose search reached far enough keeps its committed
        // frontier registered as a bleed source for the region's closing
        // pass to read.
        if nodes.len() >= RETAIN_MIN_NODES {
            retained.push(RetainedFrontier { ptr, len: nodes.len() });
        }

        // Bound the arena's resident memory now that this column's nodes are
        // safely committed.
        arena.compact(COMPACT_THRESHOLD_NODES);
    }

    // Close out the pass by folding in every retained frontier once, letting
    // light bleed across the column boundary it was captured at.
    for r in &retained {
        acc = acc.wrapping_mul(0x100000001b3) ^ diffuse::fold_span(r.ptr, r.len);
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
        f.seed(LightNode { x: 0, y: 0, z: 0, level: 5, depth: 0 });
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

    #[test]
    fn propagate_column_frontier_matches_propagate_column() {
        let col = open_column(0);
        let src = [Source { cid: 0, at: linear_index(8, 8, 8) as u16, level: 4, sky: false }];
        let (levels, nodes) = propagate_column_frontier(&col, &src, TEST_SLOTS);
        assert_eq!(levels, propagate_column(&col, &src, TEST_SLOTS));
        assert!(!nodes.is_empty());
    }

    fn nodes(n: usize) -> Vec<LightNode> {
        (0..n).map(|i| LightNode { x: 0, y: 0, z: 0, level: (i % 16) as u8, depth: 0 }).collect()
    }

    #[test]
    fn node_arena_commit_writes_are_readable_back() {
        let mut arena = NodeArena::new();
        let a = arena.commit(&nodes(4));
        let b = arena.commit(&nodes(2));
        // SAFETY: neither chunk has been compacted away, so both pointers
        // still address the nodes just committed.
        unsafe {
            assert_eq!((*a.add(3)).level, 3);
            assert_eq!((*b.add(1)).level, 1);
        }
    }

    #[test]
    fn node_arena_starts_a_new_chunk_once_the_current_one_is_full() {
        let mut arena = NodeArena::new();
        arena.commit(&nodes(ARENA_CHUNK_NODES - 2));
        assert_eq!(arena.chunks.len(), 1);
        arena.commit(&nodes(5));
        assert_eq!(arena.chunks.len(), 2);
    }

    #[test]
    fn node_arena_compact_drops_oldest_chunks_once_over_threshold() {
        let mut arena = NodeArena::new();
        for _ in 0..6 {
            arena.commit(&nodes(ARENA_CHUNK_NODES));
        }
        assert_eq!(arena.chunks.len(), 6);
        arena.compact(3 * ARENA_CHUNK_NODES);
        assert!(arena.chunks.len() < 6, "compact must drop some chunks");
        assert!(arena.resident <= 3 * ARENA_CHUNK_NODES);
    }

    #[test]
    fn node_arena_compact_never_drops_the_last_chunk() {
        let mut arena = NodeArena::new();
        arena.commit(&nodes(4));
        arena.compact(0);
        assert_eq!(arena.chunks.len(), 1, "the chunk being written to must survive");
    }
}
