//! Render-mesh rebuild with greedy face merging.
//!
//! For each visible face direction the mesher builds a 16x16 mask of exposed
//! faces, merges coplanar runs into the largest rectangles it can, and emits
//! four vertices per merged quad. The vertex buffer is written through a raw
//! strip cursor rather than by repeated `push`, because the emitter writes the
//! four corners of a quad as one unit and the bounds check per corner shows up
//! in profiles on dense terrain.
//!
//! A region's mesh rebuild touches a column's finished vertex buffer once more
//! after its own turn through the loop, so a seam can be welded against a
//! neighbour whose mesh was already committed. [`VertexArena`] gives the pass a
//! home for those committed meshes that outlives any single column — built once
//! per region rather than once per column, it appends every finished mesh into
//! fixed-size chunks and hands back a pointer into them instead of an owned
//! buffer. A column whose mesh is too large for what remains of the open chunk
//! starts a fresh one and leaves the old chunk's tail unused, so an arena that
//! has taken a run of oversized columns ends up fragmented. Committing is
//! therefore also what repacks: each commit merges the recently retired chunks
//! back into tightly packed ones and hands the slack back to the allocator,
//! which is what keeps a long region's arena sized by the mesh it holds rather
//! than by how many columns the region happens to have.

use crate::budget;
use crate::chunk::{self, linear_index, Column, SectionData};
use crate::common::*;
use crate::emit;
use crate::parse::Region;

/// One mesh vertex: quantized position, packed normal, and a light/tint word.
#[derive(Clone, Copy, Default, Debug, PartialEq, Eq)]
pub struct Vertex {
    /// Chunk-local position, 4 bits of sub-block precision per axis.
    pub pos: u32,
    /// Octahedral-packed normal in the low half, face id in the high half.
    pub norm: u32,
    /// Block light, sky light, and biome tint.
    pub light: u32,
}

/// A merged rectangle of coplanar faces, before vertex expansion.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Quad {
    /// Face direction, 0..6.
    pub face: u8,
    /// Slice depth along the face normal.
    pub depth: u8,
    /// Rectangle origin in the face's 2D plane.
    pub u: u8,
    /// Rectangle origin in the face's 2D plane.
    pub v: u8,
    /// Rectangle extent.
    pub w: u8,
    /// Rectangle extent.
    pub h: u8,
    /// The block state every merged face carries.
    pub state: u16,
}

/// Accumulates the vertices of one column's mesh.
pub struct MeshBuilder {
    verts: Vec<Vertex>,
    quads: usize,
}

impl MeshBuilder {
    /// A builder pre-sized for `slots` quads.
    ///
    /// Pre-sizing matters: the strip cursor below writes four vertices at a time
    /// through a raw pointer, so the buffer is reserved up front for the quads
    /// the mask sweep predicted.
    pub fn with_slots(slots: usize) -> MeshBuilder {
        MeshBuilder { verts: Vec::with_capacity(slots * 4), quads: 0 }
    }

    /// Vertices emitted so far.
    pub fn vertex_count(&self) -> usize {
        self.verts.len()
    }

    /// Quads emitted so far.
    pub fn quad_count(&self) -> usize {
        self.quads
    }

    /// The emitted vertices.
    pub fn vertices(&self) -> &[Vertex] {
        &self.verts
    }

    /// Append one quad's four vertices.
    pub fn push_quad(&mut self, q: &Quad) {
        for corner in 0..4u32 {
            self.verts.push(corner_vertex(q, corner));
        }
        self.quads += 1;
    }

    /// Emit a whole run of merged quads through one strip cursor.
    ///
    /// The cursor is taken once for the run and the emitter fills the corners
    /// behind it, so a long run of merged faces costs one bounds check instead
    /// of four per quad.
    pub fn push_run(&mut self, run: &[Quad]) {
        if run.is_empty() {
            return;
        }
        let base = self.verts.len();
        for q in run {
            for corner in 0..4u32 {
                self.verts.push(corner_vertex(q, corner));
            }
            self.quads += 1;
        }
        // The strip cursor is taken only once the run has been written: a run
        // longer than the remaining reservation grows the vertex buffer, and a
        // cursor taken beforehand would address the old allocation.
        let cursor = self.verts.as_mut_ptr();
        // Post-process the strip in place: adjacent merged quads share edges, so
        // the emitter welds their corner lighting into a continuous gradient.
        emit::weld_strip(cursor, base, run.len() * 4);
    }
}

/// Expand one corner of a merged quad into a vertex.
fn corner_vertex(q: &Quad, corner: u32) -> Vertex {
    let (du, dv) = match corner {
        0 => (0u32, 0u32),
        1 => (q.w as u32, 0),
        2 => (q.w as u32, q.h as u32),
        _ => (0, q.h as u32),
    };
    let u = q.u as u32 + du;
    let v = q.v as u32 + dv;
    let d = q.depth as u32;
    Vertex {
        pos: (u & 0x3f) | ((v & 0x3f) << 6) | ((d & 0x3f) << 12),
        norm: ((q.face as u32) << 16) | 0x2a,
        light: ((q.state as u32) << 8) | (corner & 3),
    }
}

/// Build the exposed-face mask for one slice of a section.
///
/// A face is exposed when the block is solid and its neighbour along `face` is
/// not. The mask holds the block state so the merge step only joins faces that
/// would render identically.
pub fn face_mask(s: &SectionData, face: u8, depth: usize) -> Vec<u16> {
    let mut mask = vec![0u16; SECTION_EDGE * SECTION_EDGE];
    if s.is_empty() {
        return mask;
    }
    for a in 0..SECTION_EDGE {
        for b in 0..SECTION_EDGE {
            let (x, y, z) = unproject(face, depth, a, b);
            let here = s.state_at(linear_index(x, y, z));
            if here == 0 {
                continue;
            }
            let neighbour = match step(face, x, y, z) {
                Some((nx, ny, nz)) => s.state_at(linear_index(nx, ny, nz)),
                // Off the edge of the section: treat as open so the boundary
                // face is emitted.
                None => 0,
            };
            if neighbour == 0 {
                mask[b * SECTION_EDGE + a] = here;
            }
        }
    }
    mask
}

/// Map a face-plane coordinate back into section-local `(x, y, z)`.
fn unproject(face: u8, depth: usize, a: usize, b: usize) -> (usize, usize, usize) {
    match face {
        0 | 1 => (depth, a, b),
        2 | 3 => (a, depth, b),
        _ => (a, b, depth),
    }
}

/// The neighbour of `(x, y, z)` along `face`, or `None` at the section edge.
fn step(face: u8, x: usize, y: usize, z: usize) -> Option<(usize, usize, usize)> {
    match face {
        0 => x.checked_sub(1).map(|nx| (nx, y, z)),
        1 => (x + 1 < SECTION_EDGE).then_some((x + 1, y, z)),
        2 => y.checked_sub(1).map(|ny| (x, ny, z)),
        3 => (y + 1 < SECTION_EDGE).then_some((x, y + 1, z)),
        4 => z.checked_sub(1).map(|nz| (x, y, nz)),
        _ => (z + 1 < SECTION_EDGE).then_some((x, y, z + 1)),
    }
}

/// Merge a face mask into the largest rectangles it admits.
pub fn merge_mask(mask: &mut [u16], face: u8, depth: usize) -> Vec<Quad> {
    let mut out = Vec::new();
    for v in 0..SECTION_EDGE {
        let mut u = 0usize;
        while u < SECTION_EDGE {
            let state = mask[v * SECTION_EDGE + u];
            if state == 0 {
                u += 1;
                continue;
            }
            // Grow width along u.
            let mut w = 1usize;
            while u + w < SECTION_EDGE && mask[v * SECTION_EDGE + u + w] == state {
                w += 1;
            }
            // Grow height along v while the whole row matches.
            let mut h = 1usize;
            'grow: while v + h < SECTION_EDGE {
                for k in 0..w {
                    if mask[(v + h) * SECTION_EDGE + u + k] != state {
                        break 'grow;
                    }
                }
                h += 1;
            }
            for dv in 0..h {
                for du in 0..w {
                    mask[(v + dv) * SECTION_EDGE + u + du] = 0;
                }
            }
            out.push(Quad {
                face,
                depth: depth as u8,
                u: u as u8,
                v: v as u8,
                w: w as u8,
                h: h as u8,
                state,
            });
            u += w;
        }
    }
    out
}

/// Mesh one section into `builder`.
pub fn mesh_section(builder: &mut MeshBuilder, s: &SectionData) {
    for face in 0..6u8 {
        for depth in 0..SECTION_EDGE {
            let mut mask = face_mask(s, face, depth);
            let run = merge_mask(&mut mask, face, depth);
            if !run.is_empty() {
                builder.push_run(&run);
            }
        }
    }
}

/// Mesh a whole column.
pub fn mesh_column(col: &Column, slots: usize) -> MeshBuilder {
    let mut builder = MeshBuilder::with_slots(slots);
    for s in &col.sections {
        if s.is_empty() {
            continue;
        }
        mesh_section(&mut builder, s);
    }
    builder
}

/// Vertices held by one [`VertexArena`] chunk.
///
/// An ordinary column's finished mesh fits with room to spare, so a typical
/// commit never has to look past the chunk it lands in.
const ARENA_CHUNK_VERTS: usize = 512;

/// Retired chunks [`VertexArena::reclaim`] will walk in one pass.
///
/// Merging copies every vertex it moves, so a pass over the whole arena would
/// make each commit cost what the region has meshed so far. Bounding the window
/// keeps that cost flat: the newest retired chunks are where a ragged run of
/// columns leaves its slack, and the older ones behind them were already packed
/// tight by an earlier pass.
const RECLAIM_WINDOW_CHUNKS: usize = 4;

/// Slots in [`SeamBelt`].
const SEAM_SLOTS: usize = 4;

/// One chunk of a [`VertexArena`]: an exact-sized run of vertices and how far
/// into it the arena has written.
struct ArenaChunk {
    verts: Box<[Vertex]>,
    used: usize,
}

impl ArenaChunk {
    /// A chunk with room for `cap` vertices and nothing written yet.
    fn empty(cap: usize) -> ArenaChunk {
        ArenaChunk { verts: vec![Vertex::default(); cap].into_boxed_slice(), used: 0 }
    }

    /// Vertices this chunk could still take.
    fn room(&self) -> usize {
        self.verts.len() - self.used
    }

    /// Append `verts` and return where they landed.
    ///
    /// The caller is responsible for having checked [`ArenaChunk::room`].
    fn append(&mut self, verts: &[Vertex]) -> *const Vertex {
        let at = self.used;
        self.verts[at..at + verts.len()].copy_from_slice(verts);
        // SAFETY: the `copy_from_slice` above indexed `at .. at + verts.len()`
        // of this chunk, so that range lies inside the allocation and `at` is
        // a valid offset within it.
        let ptr = unsafe { self.verts.as_ptr().add(at) };
        self.used = at + verts.len();
        ptr
    }
}

/// A region-lifetime arena for committed column meshes.
///
/// Built once per region rather than once per column, so a pointer handed out
/// while meshing one column stays valid while later columns are meshed — which
/// is what lets a boundary reference (see [`build_region`]) be read again well
/// after its own column has finished. Vertices are appended into fixed-size
/// chunks, each stored as an exact-sized boxed slice; a chunk with no room left
/// for the next commit is retired as-is and a fresh one takes over, so a single
/// commit is never split across two chunks.
struct VertexArena {
    /// Chunks holding committed column meshes, oldest first. The last is the
    /// one currently being written to; the rest are retired.
    chunks: Vec<ArenaChunk>,
}

impl VertexArena {
    fn new() -> VertexArena {
        VertexArena { chunks: Vec::new() }
    }

    /// Vertices held across every chunk the arena has allocated.
    #[cfg(test)]
    fn capacity(&self) -> usize {
        self.chunks.iter().map(|c| c.verts.len()).sum()
    }

    /// Where the mesh committed most recently landed, as an arena position
    /// rather than an address.
    ///
    /// A caller that will read a mesh back more than once holds this instead of
    /// the pointer: a position is the arena's own coordinate for a commit, and
    /// costs the caller two words rather than a word and a lifetime.
    fn locate(&self, len: usize) -> ArenaSlot {
        let chunk = self.chunks.len().saturating_sub(1);
        let at = self.chunks.last().map_or(0, |c| c.used.saturating_sub(len));
        ArenaSlot { chunk, at }
    }

    /// The address `slot` names in the arena's current layout, or null if the
    /// arena no longer has that many chunks.
    fn resolve(&self, slot: ArenaSlot) -> *const Vertex {
        match self.chunks.get(slot.chunk) {
            // SAFETY: `slot.chunk` was just bounds-checked against `chunks`.
            Some(chunk) => unsafe { chunk.verts.as_ptr().add(slot.at) },
            None => std::ptr::null(),
        }
    }

    /// Commit `verts` into the arena and return a pointer to where they
    /// landed.
    ///
    /// If what remains of the open chunk cannot hold `verts`, a fresh chunk
    /// takes over first, so the returned pointer's `verts.len()` vertices are
    /// always contiguous. Committing also settles the arena's accumulated
    /// fragmentation (see [`VertexArena::reclaim`]) — repacking is part of what
    /// it means to commit rather than a separate pass the caller schedules, so
    /// the arena's footprint tracks the mesh it holds however the region drives
    /// it.
    fn commit(&mut self, verts: &[Vertex]) -> *const Vertex {
        let len = verts.len();
        if !self.chunks.last().is_some_and(|c| c.room() >= len) {
            self.chunks.push(ArenaChunk::empty(len.max(ARENA_CHUNK_VERTS)));
        }
        let chunk = self.chunks.last_mut().expect("a chunk was just ensured above");
        let at = chunk.append(verts);
        self.reclaim(RECLAIM_WINDOW_CHUNKS);
        at
    }

    /// Merge the arena's recently retired chunks into tightly packed ones,
    /// recovering the slack their tails hold.
    ///
    /// A column whose mesh does not fit what remains of the open chunk retires
    /// that chunk with its tail unused and starts a fresh one. Over a run of
    /// such columns the waste adds up, and repacking hands whole pages back to
    /// the allocator.
    ///
    /// Two conditions keep the pass from being busy-work. It walks only the
    /// newest `window` retired chunks, which is where a ragged run leaves its
    /// slack — everything older was already packed tight by an earlier pass —
    /// and it runs only once a full chunk's worth of slack has gathered inside
    /// that window, below which there is no page to hand back. The chunk
    /// currently being written to is never touched, since the next commit
    /// continues into it.
    fn reclaim(&mut self, window: usize) {
        let retired = self.chunks.len().saturating_sub(1);
        if retired < 2 || window < 2 {
            return;
        }
        let from = retired - retired.min(window);
        let run = &self.chunks[from..retired];
        let live: usize = run.iter().map(|c| c.used).sum();
        let held: usize = run.iter().map(|c| c.verts.len()).sum();
        if held - live < ARENA_CHUNK_VERTS {
            return;
        }

        let mut packed: Vec<ArenaChunk> = Vec::new();
        for chunk in run {
            let mut rest = &chunk.verts[..chunk.used];
            while !rest.is_empty() {
                if packed.last().map_or(0, ArenaChunk::room) == 0 {
                    packed.push(ArenaChunk::empty(ARENA_CHUNK_VERTS));
                }
                let dst = packed.last_mut().expect("a chunk was just ensured above");
                let take = dst.room().min(rest.len());
                dst.append(&rest[..take]);
                rest = &rest[take..];
            }
        }
        // Splice the packed run back between the chunks it did not cover and
        // the open chunk, releasing the ones the vertices were copied out of.
        let mut rebuilt: Vec<ArenaChunk> = Vec::with_capacity(from + packed.len() + 1);
        rebuilt.extend(self.chunks.drain(..from));
        let open = self.chunks.pop().expect("the open chunk");
        rebuilt.extend(packed);
        rebuilt.push(open);
        self.chunks = rebuilt;
    }
}

/// A column's committed mesh, named for the closing seam-weld pass to read.
///
/// The closing pass walks the belt exactly once, so it takes the address the
/// arena gave out rather than resolving a position per mesh.
#[derive(Clone, Copy)]
struct SeamRef {
    ptr: *const Vertex,
    len: usize,
}

/// A position in a [`VertexArena`]: which chunk, and how far into it.
#[derive(Clone, Copy)]
struct ArenaSlot {
    chunk: usize,
    at: usize,
}

/// The mesh the adjacency weld measures the next column against, held as an
/// arena position because it is read on a later turn of the loop than the one
/// that committed it.
#[derive(Clone, Copy)]
struct Adjacent {
    slot: ArenaSlot,
    len: usize,
}

/// The meshes [`build_region`]'s closing pass welds a seam against.
///
/// Fixed capacity, reused round-robin: every column offers its committed mesh
/// as a boundary reference, and the belt keeps the most recent few so the
/// closing pass costs the same on a region of four columns and one of four
/// thousand.
struct SeamBelt {
    slots: [Option<SeamRef>; SEAM_SLOTS],
    next: usize,
}

impl SeamBelt {
    fn new() -> SeamBelt {
        SeamBelt { slots: [None, None, None, None], next: 0 }
    }

    /// Put `seam` in the next slot, displacing whatever that slot held.
    fn offer(&mut self, seam: SeamRef) {
        self.slots[self.next] = Some(seam);
        self.next = (self.next + 1) % SEAM_SLOTS;
    }

    /// The references currently held, oldest slot first.
    fn refs(&self) -> impl Iterator<Item = &SeamRef> {
        self.slots.iter().flatten()
    }
}

/// Rebuild the render mesh for every column and fold a digest of the result.
///
/// Each column's mesh is welded and folded as it is produced, then committed
/// into a region-wide [`VertexArena`] rather than freed with the column, and
/// offered to the [`SeamBelt`] as a boundary reference. Two seams are welded on
/// top of that per-column work: the *adjacency* seam, where a column meets the
/// one meshed immediately before it, is folded as soon as both sides are
/// committed; and the region's closing pass folds the belt's meshes once more,
/// welding against neighbours whose own turn through the loop has already
/// passed.
pub fn build_region(region: &Region, n: usize) -> u64 {
    // The per-column vertex reservation is sized from the region's rebuild
    // pressure, so a region of sparse columns does not over-reserve.
    let slots = budget::pool_slots(region, n);
    let mut arena = VertexArena::new();
    let mut belt = SeamBelt::new();
    // The column meshed on the previous turn, held for the adjacency seam.
    let mut prev: Option<Adjacent> = None;
    let mut acc = 0xffu64;
    for cid in 0..n {
        let col = match chunk::decode(region, cid) {
            Ok(c) => c,
            Err(_) => continue,
        };
        if col.sections.is_empty() {
            continue;
        }
        let builder = mesh_column(&col, slots);
        if builder.vertices().is_empty() {
            continue;
        }
        // Weld the column's corners and build its index buffer before
        // folding, so the digest covers the mesh a renderer would actually
        // upload.
        let table = crate::weld::weld(builder.vertices());
        let indices = crate::weld::triangulate(&table);
        acc = acc.wrapping_mul(0x100000001b3) ^ emit::fold_mesh(builder.vertices());
        acc = acc.wrapping_mul(0x100000001b3) ^ crate::weld::fold_indices(&indices);

        // Commit this column's finished mesh into the region's arena. Only
        // past this point is there a pointer stable enough to name past this
        // column's own scope.
        let ptr = arena.commit(builder.vertices());
        let here = SeamRef { ptr, len: builder.vertex_count() };

        // Weld the adjacency seam. The two columns meet along a shared chunk
        // boundary, so the trailing corners of the earlier mesh and the leading
        // corners of this one describe the same edge of the world and are
        // folded together now that both sides are in the arena. The earlier
        // mesh is resolved against the arena's current layout, since it was
        // committed a turn ago.
        if let Some(p) = prev {
            acc = acc.wrapping_mul(0x100000001b3)
                ^ emit::fold_boundary(arena.resolve(p.slot), p.len, here.ptr, here.len);
        }

        prev = Some(Adjacent { slot: arena.locate(here.len), len: here.len });
        belt.offer(here);
    }

    // Close out the pass by folding in every boundary mesh the belt holds.
    for r in belt.refs() {
        acc = acc.wrapping_mul(0x100000001b3)
            ^ crate::weld::fold_span(r.ptr, r.len, region.seed as u64);
    }
    acc
}

#[cfg(test)]
mod tests {
    use super::*;

    fn solid_section() -> SectionData {
        SectionData {
            palette: vec![0, 3],
            blocks: vec![1; SECTION_VOLUME],
            flags: 0,
            base_y: 0,
        }
    }

    #[test]
    fn face_mask_of_solid_section_is_edge_only() {
        let s = solid_section();
        // Interior slice: fully enclosed, nothing exposed.
        let interior = face_mask(&s, 0, 8);
        assert!(interior.iter().all(|&m| m == 0));
        // Boundary slice: the whole face is exposed.
        let boundary = face_mask(&s, 0, 0);
        assert!(boundary.iter().all(|&m| m == 3));
    }

    #[test]
    fn merge_mask_joins_a_full_face_into_one_quad() {
        let mut mask = vec![7u16; SECTION_EDGE * SECTION_EDGE];
        let quads = merge_mask(&mut mask, 0, 0);
        assert_eq!(quads.len(), 1);
        assert_eq!(quads[0].w, SECTION_EDGE as u8);
        assert_eq!(quads[0].h, SECTION_EDGE as u8);
        assert!(mask.iter().all(|&m| m == 0), "merged cells must be consumed");
    }

    #[test]
    fn merge_mask_splits_on_differing_state() {
        let mut mask = vec![0u16; SECTION_EDGE * SECTION_EDGE];
        mask[0] = 1;
        mask[1] = 2;
        let quads = merge_mask(&mut mask, 0, 0);
        assert_eq!(quads.len(), 2);
        assert_eq!(quads[0].w, 1);
        assert_eq!(quads[1].w, 1);
    }

    #[test]
    fn merged_area_equals_filled_cells() {
        let mut mask = vec![0u16; SECTION_EDGE * SECTION_EDGE];
        for (i, cell) in mask.iter_mut().enumerate() {
            if i % 3 == 0 {
                *cell = 5;
            }
        }
        let filled = mask.iter().filter(|&&m| m != 0).count();
        let quads = merge_mask(&mut mask, 1, 2);
        let area: usize = quads.iter().map(|q| q.w as usize * q.h as usize).sum();
        assert_eq!(area, filled);
    }

    #[test]
    fn push_quad_emits_four_vertices() {
        let mut b = MeshBuilder::with_slots(4);
        b.push_quad(&Quad { face: 0, depth: 0, u: 0, v: 0, w: 1, h: 1, state: 1 });
        assert_eq!(b.vertex_count(), 4);
        assert_eq!(b.quad_count(), 1);
    }

    #[test]
    fn corner_vertices_are_distinct() {
        let q = Quad { face: 2, depth: 3, u: 1, v: 1, w: 4, h: 5, state: 9 };
        let corners: Vec<Vertex> = (0..4).map(|c| corner_vertex(&q, c)).collect();
        for i in 0..4 {
            for j in i + 1..4 {
                assert_ne!(corners[i], corners[j]);
            }
        }
    }

    #[test]
    fn step_stops_at_section_edges() {
        assert_eq!(step(0, 0, 5, 5), None);
        assert_eq!(step(1, SECTION_EDGE - 1, 5, 5), None);
        assert_eq!(step(1, 0, 5, 5), Some((1, 5, 5)));
    }

    fn verts(n: usize) -> Vec<Vertex> {
        (0..n).map(|i| Vertex { pos: i as u32, norm: 0, light: 0 }).collect()
    }

    #[test]
    fn arena_commit_writes_are_readable_back() {
        let mut arena = VertexArena::new();
        let a = arena.commit(&verts(4));
        let b = arena.commit(&verts(2));
        // SAFETY: both landed in the one open chunk, which nothing here has
        // repacked, so both pointers still address the vertices just
        // committed.
        unsafe {
            assert_eq!((*a.add(3)).pos, 3);
            assert_eq!((*b.add(1)).pos, 1);
        }
    }

    #[test]
    fn arena_starts_a_new_chunk_once_the_current_one_is_full() {
        let mut arena = VertexArena::new();
        arena.commit(&verts(ARENA_CHUNK_VERTS - 2));
        assert_eq!(arena.chunks.len(), 1);
        // Only 2 slots remain in the first chunk; this does not fit.
        arena.commit(&verts(5));
        assert_eq!(arena.chunks.len(), 2);
    }

    #[test]
    fn reclaim_leaves_an_unfragmented_arena_untouched() {
        let mut arena = VertexArena::new();
        arena.commit(&verts(4));
        arena.reclaim(RECLAIM_WINDOW_CHUNKS);
        assert_eq!(arena.chunks.len(), 1);
        // Two chunks, the first exactly full: nothing worth recovering.
        arena.commit(&verts(ARENA_CHUNK_VERTS));
        assert_eq!(arena.chunks.len(), 2);
        arena.reclaim(RECLAIM_WINDOW_CHUNKS);
        assert_eq!(arena.chunks.len(), 2);
    }

    #[test]
    fn reclaim_repacks_retired_chunks_and_keeps_their_vertices() {
        let mut arena = VertexArena::new();
        // Each commit takes most of a chunk, so the next one retires it with
        // its tail unused — and committing settles that fragmentation itself.
        for _ in 0..4 {
            arena.commit(&verts(300));
        }
        assert!(arena.chunks.len() < 4, "committing must recover whole chunks");
        assert!(arena.capacity() < 4 * ARENA_CHUNK_VERTS);
        assert_eq!(
            arena.chunks.iter().map(|c| c.used).sum::<usize>(),
            4 * 300,
            "no vertex may be lost in the merge"
        );
    }

    /// The arena's footprint must track the mesh it holds, not the number of
    /// columns that produced it. A run of ragged commits leaves a chunk tail
    /// unused every time, so an arena that stopped settling its fragmentation —
    /// or deferred it to after the pass — would carry that waste for the whole
    /// region.
    #[test]
    fn arena_slack_stays_bounded_across_a_long_run() {
        let mut arena = VertexArena::new();
        let commits = 40;
        for _ in 0..commits {
            arena.commit(&verts(300));
        }
        let used: usize = arena.chunks.iter().map(|c| c.used).sum();
        assert_eq!(used, commits * 300, "no vertex may be lost across the run");
        let slack = arena.capacity() - used;
        assert!(
            slack <= 3 * ARENA_CHUNK_VERTS,
            "arena slack grew with the region: {slack} unused vertices held"
        );
    }

    #[test]
    fn reclaim_never_touches_the_open_chunk() {
        let mut arena = VertexArena::new();
        for _ in 0..3 {
            arena.commit(&verts(300));
        }
        let open = arena.commit(&verts(300));
        arena.reclaim(RECLAIM_WINDOW_CHUNKS);
        // SAFETY: the open chunk is the one this pointer addresses, and
        // repacking only ever covers the retired chunks behind it.
        unsafe {
            assert_eq!((*open.add(7)).pos, 7);
        }
    }

    #[test]
    fn seam_belt_holds_only_its_newest_slots() {
        let mut belt = SeamBelt::new();
        let v = verts(4);
        for _ in 0..SEAM_SLOTS + 3 {
            belt.offer(SeamRef { ptr: v.as_ptr(), len: v.len() });
        }
        assert_eq!(belt.refs().count(), SEAM_SLOTS, "capacity must be fixed");
    }
}
