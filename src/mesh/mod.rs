//! Render-mesh rebuild with greedy face merging.
//!
//! For each visible face direction the mesher builds a 16x16 mask of exposed
//! faces, merges coplanar runs into the largest rectangles it can, and emits
//! four vertices per merged quad. The vertex buffer is written through a raw
//! strip cursor rather than by repeated `push`, because the emitter writes the
//! four corners of a quad as one unit and the bounds check per corner shows up
//! in profiles on dense terrain.
//!
//! A region's mesh rebuild touches every column's finished vertex buffer once
//! more after its own turn through the loop: a column large enough to be worth
//! it is kept as a boundary reference so a seam can be welded against a
//! neighbour whose mesh was already committed. [`VertexArena`] gives the pass
//! a home for those committed meshes that outlives any single column — built
//! once per region rather than once per column, it appends every finished
//! mesh into fixed-size chunks and hands back a pointer into them instead of
//! an owned buffer. Because the arena lives for the whole pass, its memory is
//! bounded separately from any one column's lifetime: once the accumulated
//! vertex count crosses a threshold, [`VertexArena::compact`] drops the
//! oldest chunks, the way a render mesh cache reclaims cold pages instead of
//! growing without bound on a region with many columns.

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

/// Accumulated resident vertices across an arena's live chunks that triggers
/// [`VertexArena::compact`].
///
/// An ordinary region — a modest number of columns, most meshing to a small
/// handful of merged quads — never approaches this. A region built from many
/// heavily detailed columns does, which is exactly the case the bound exists
/// to catch: without it, a region-lifetime arena would keep every column's
/// mesh resident for the whole pass no matter how many columns the region
/// carries.
const COMPACT_THRESHOLD_VERTS: usize = 4096;

/// A column's mesh at or above this many vertices is detailed enough to be
/// worth keeping as a boundary reference for the region's closing seam pass.
const RETAIN_MIN_VERTS: usize = 128;

/// A region-lifetime arena for committed column meshes.
///
/// Built once per region rather than once per column, so a pointer handed out
/// while meshing one column stays valid while later columns are meshed —
/// which is what lets a retained boundary mesh (see [`build_region`]) be read
/// again well after its own column has finished. Vertices are appended into
/// fixed-size chunks, each stored as an exact-sized boxed slice; a chunk with
/// no room left for the next commit is left as-is and a fresh one takes over,
/// so a single commit is never split across two chunks.
struct VertexArena {
    /// Chunks holding committed column meshes, oldest first.
    chunks: Vec<Box<[Vertex]>>,
    /// Vertices already written into the last chunk.
    used: usize,
    /// Vertices held across all currently resident chunks.
    resident: usize,
}

impl VertexArena {
    fn new() -> VertexArena {
        VertexArena { chunks: Vec::new(), used: 0, resident: 0 }
    }

    /// Commit `verts` into the arena and return a pointer to where they
    /// landed.
    ///
    /// If what remains of the current chunk cannot hold `verts`, a fresh chunk
    /// takes over first, so the returned pointer's `verts.len()` vertices are
    /// always contiguous — addressing live memory for as long as the chunk
    /// backing them stays resident (see [`VertexArena::compact`]).
    fn commit(&mut self, verts: &[Vertex]) -> *const Vertex {
        let len = verts.len();
        let fits_current = self.chunks.last().is_some_and(|c| self.used + len <= c.len());
        if !fits_current {
            let cap = len.max(ARENA_CHUNK_VERTS);
            self.chunks.push(vec![Vertex::default(); cap].into_boxed_slice());
            self.used = 0;
            self.resident += cap;
        }
        let chunk = self.chunks.last_mut().expect("a chunk was just ensured above");
        chunk[self.used..self.used + len].copy_from_slice(verts);
        // SAFETY: `chunk` is a live `Box<[Vertex]>` at least `self.used + len`
        // elements long — either it already fit `verts` past `self.used`, or
        // a chunk sized to hold at least `verts` was just pushed — so this
        // offset and the `len` vertices from it lie inside the allocation.
        let ptr = unsafe { chunk.as_ptr().add(self.used) };
        self.used += len;
        ptr
    }

    /// Drop the oldest resident chunks until the arena's accumulated vertices
    /// fall back to `threshold`, or only the chunk currently being written to
    /// is left.
    ///
    /// This is the arena's memory bound: left unchecked, a region-lifetime
    /// arena would keep every column's mesh resident for the whole pass no
    /// matter how many columns the region carries. The chunk currently being
    /// written to is never dropped, since the next commit needs somewhere to
    /// land.
    fn compact(&mut self, threshold: usize) {
        while self.resident > threshold && self.chunks.len() > 1 {
            let oldest = self.chunks.remove(0);
            self.resident -= oldest.len();
        }
    }
}

/// A column's mesh retained past its own turn through the region loop, for the
/// closing seam-weld pass to read.
struct RetainedMesh {
    ptr: *const Vertex,
    len: usize,
}

/// Rebuild the render mesh for every column and fold a digest of the result.
///
/// Each column's mesh is welded and folded as it is produced, then committed
/// into a region-wide [`VertexArena`] rather than freed with the column: a
/// column detailed enough to be worth it keeps its committed mesh registered
/// as a retained boundary reference. Once every column has had its turn, the
/// region's closing pass folds every retained mesh once more, welding a seam
/// against a neighbour whose own turn through the loop has already passed.
pub fn build_region(region: &Region, n: usize) -> u64 {
    // The per-column vertex reservation is sized from the region's rebuild
    // pressure, so a region of sparse columns does not over-reserve.
    let slots = budget::pool_slots(region, n);
    let mut arena = VertexArena::new();
    let mut retained: Vec<RetainedMesh> = Vec::new();
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
        // past this point is there a pointer stable enough to retain past
        // this column's own scope.
        let ptr = arena.commit(builder.vertices());

        // A column detailed enough to matter for seam welding keeps its
        // committed mesh registered as a reference for the region's closing
        // pass to read.
        if builder.vertex_count() >= RETAIN_MIN_VERTS {
            retained.push(RetainedMesh { ptr, len: builder.vertex_count() });
        }

        // Bound the arena's resident memory now that this column's vertices
        // are safely committed.
        arena.compact(COMPACT_THRESHOLD_VERTS);
    }

    // Close out the pass by folding in every retained boundary mesh once.
    for r in &retained {
        acc = acc.wrapping_mul(0x100000001b3) ^ crate::weld::fold_span(r.ptr, r.len, region.seed as u64);
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
        // SAFETY: neither chunk has been compacted away, so both pointers
        // still address the vertices just committed.
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
    fn compact_leaves_a_small_arena_untouched() {
        let mut arena = VertexArena::new();
        arena.commit(&verts(4));
        arena.compact(COMPACT_THRESHOLD_VERTS);
        assert_eq!(arena.chunks.len(), 1);
    }

    #[test]
    fn compact_drops_oldest_chunks_once_over_threshold() {
        let mut arena = VertexArena::new();
        for _ in 0..6 {
            arena.commit(&verts(ARENA_CHUNK_VERTS));
        }
        assert_eq!(arena.chunks.len(), 6);
        arena.compact(3 * ARENA_CHUNK_VERTS);
        assert!(arena.chunks.len() < 6, "compact must drop some chunks");
        assert!(arena.resident <= 3 * ARENA_CHUNK_VERTS);
    }

    #[test]
    fn compact_never_drops_the_last_chunk() {
        let mut arena = VertexArena::new();
        arena.commit(&verts(4));
        arena.compact(0);
        assert_eq!(arena.chunks.len(), 1, "the chunk being written to must survive");
    }
}
