//! Vertex welding and index-buffer generation.
//!
//! Greedy meshing emits four independent vertices per quad, but adjacent quads
//! very often share corners — a merged wall face meets its neighbour on an exact
//! position/normal/light match. Welding folds those duplicates into one vertex
//! and rewrites the quads as indices, which is what makes a chunk mesh fit in a
//! sensible vertex buffer.

use crate::mesh::Vertex;

/// A welded vertex set plus the remap from original corner order to unique ids.
pub struct WeldTable {
    /// The deduplicated vertices.
    pub unique: Vec<Vertex>,
    /// One entry per original vertex, naming its index in `unique`.
    pub remap: Vec<u32>,
}

impl WeldTable {
    /// How many unique vertices survived welding.
    pub fn unique_count(&self) -> usize {
        self.unique.len()
    }

    /// How many vertices went in.
    pub fn source_count(&self) -> usize {
        self.remap.len()
    }

    /// How many duplicates the weld collapsed.
    pub fn collapsed(&self) -> usize {
        self.remap.len().saturating_sub(self.unique.len())
    }

    /// The fraction of vertices removed, in percent.
    pub fn ratio_percent(&self) -> u32 {
        if self.remap.is_empty() {
            return 0;
        }
        (self.collapsed() * 100 / self.remap.len()) as u32
    }
}

/// Weld a vertex stream, collapsing exact duplicates.
///
/// Two vertices weld only when position, normal and light all agree, so a shared
/// corner between differently lit faces is correctly kept apart.
pub fn weld(verts: &[Vertex]) -> WeldTable {
    let mut unique: Vec<Vertex> = Vec::new();
    let mut remap: Vec<u32> = Vec::with_capacity(verts.len());
    for v in verts {
        match unique.iter().position(|u| u == v) {
            Some(i) => remap.push(i as u32),
            None => {
                unique.push(*v);
                remap.push((unique.len() - 1) as u32);
            }
        }
    }
    WeldTable { unique, remap }
}

/// Build the triangle index buffer for a welded quad stream.
///
/// Each quad becomes two triangles — corners `0,1,2` and `0,2,3` — so the buffer
/// holds six indices per quad. It is filled through a raw cursor because the six
/// writes per quad are contiguous and the per-index bounds check showed up in
/// profiles on dense chunk meshes.
pub fn triangulate(table: &WeldTable) -> Vec<u32> {
    // Reserve from the source quad count. Welding collapses duplicate corners,
    // so the unique vertex count is a lower bound on the quads written and
    // reserving from it would under-size the buffer.
    let source_quads = table.remap.len() / 4;
    let mut indices: Vec<u32> = Vec::with_capacity(source_quads * 6);
    if source_quads == 0 {
        return indices;
    }

    let cursor = indices.as_mut_ptr();
    let mut written = 0usize;

    // Walk the original quad stream so every emitted quad keeps its winding.
    // SAFETY: the buffer was reserved for the quads this stream contains, so the
    // six writes per quad stay inside the reservation.
    unsafe {
        for q in 0..source_quads {
            let a = table.remap[q * 4];
            let b = table.remap[q * 4 + 1];
            let c = table.remap[q * 4 + 2];
            let d = table.remap[q * 4 + 3];
            *cursor.add(written) = a;
            *cursor.add(written + 1) = b;
            *cursor.add(written + 2) = c;
            *cursor.add(written + 3) = a;
            *cursor.add(written + 4) = c;
            *cursor.add(written + 5) = d;
            written += 6;
        }
        indices.set_len(written);
    }
    indices
}

/// Fold a digest of a column mesh retained in the region's [`crate::mesh`]
/// vertex arena.
///
/// [`crate::mesh::build_region`] commits every column's finished mesh into a
/// region-lifetime arena, and a column whose mesh is large enough is kept as a
/// retained boundary reference so the region's closing seam pass can fold it
/// again once every column has had its own turn through the loop. This is
/// that pass's read: `ptr`/`len` name the committed span directly, walked
/// through raw pointer arithmetic rather than a bounds-checked index, so nothing
/// here depends on `Vec`'s own capacity bookkeeping.
///
/// SAFETY: `ptr` must address at least `len` live [`Vertex`] values for the
/// call.
pub fn fold_span(ptr: *const Vertex, len: usize, seed: u64) -> u64 {
    if ptr.is_null() || len == 0 {
        return 0;
    }
    let mut acc = seed ^ 0x2545f491;
    // SAFETY: per this function's contract, the caller guarantees `ptr`
    // addresses at least `len` live vertices.
    unsafe {
        for i in 0..len {
            let v = *ptr.add(i);
            acc = acc.rotate_left(5) ^ (v.pos as u64) ^ ((v.norm as u64) << 3) ^ (v.light as u64);
        }
    }
    acc
}

/// Fold a digest of an index buffer.
pub fn fold_indices(indices: &[u32]) -> u64 {
    let mut acc = (indices.len() as u64).wrapping_mul(0x9e3779b1);
    for &i in indices {
        acc = acc.rotate_left(5) ^ (i as u64);
    }
    acc
}

/// The highest vertex id an index buffer refers to.
pub fn max_index(indices: &[u32]) -> u32 {
    indices.iter().copied().max().unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn vertex(n: u32) -> Vertex {
        Vertex { pos: n, norm: n * 2, light: n * 3 }
    }

    /// Four corners per quad, all distinct — nothing welds away.
    fn distinct_quads(n: usize) -> Vec<Vertex> {
        (0..n * 4).map(|i| vertex(i as u32)).collect()
    }

    #[test]
    fn weld_collapses_exact_duplicates() {
        let verts = vec![vertex(1), vertex(1), vertex(2)];
        let t = weld(&verts);
        assert_eq!(t.unique_count(), 2);
        assert_eq!(t.source_count(), 3);
        assert_eq!(t.collapsed(), 1);
        assert_eq!(t.remap, vec![0, 0, 1]);
    }

    #[test]
    fn weld_keeps_distinct_vertices_apart() {
        let verts = distinct_quads(2);
        let t = weld(&verts);
        assert_eq!(t.unique_count(), 8);
        assert_eq!(t.collapsed(), 0);
        assert_eq!(t.ratio_percent(), 0);
    }

    #[test]
    fn weld_of_nothing_is_empty() {
        let t = weld(&[]);
        assert_eq!(t.unique_count(), 0);
        assert_eq!(t.source_count(), 0);
        assert_eq!(t.ratio_percent(), 0);
        assert!(triangulate(&t).is_empty());
    }

    #[test]
    fn triangulate_emits_six_indices_per_quad() {
        let t = weld(&distinct_quads(3));
        let idx = triangulate(&t);
        assert_eq!(idx.len(), 18);
        // First quad: 0,1,2, 0,2,3
        assert_eq!(&idx[..6], &[0, 1, 2, 0, 2, 3]);
    }

    #[test]
    fn triangulate_indices_stay_in_range() {
        let t = weld(&distinct_quads(4));
        let idx = triangulate(&t);
        assert!(max_index(&idx) < t.unique_count() as u32);
    }

    #[test]
    fn fold_indices_is_order_sensitive() {
        assert_ne!(fold_indices(&[0, 1, 2]), fold_indices(&[2, 1, 0]));
        assert_eq!(max_index(&[]), 0);
    }

    #[test]
    fn fold_span_is_deterministic() {
        let verts = distinct_quads(2);
        assert_eq!(
            fold_span(verts.as_ptr(), verts.len(), 7),
            fold_span(verts.as_ptr(), verts.len(), 7)
        );
    }

    #[test]
    fn fold_span_reflects_seed_and_contents() {
        let verts = distinct_quads(2);
        assert_ne!(fold_span(verts.as_ptr(), verts.len(), 7), fold_span(verts.as_ptr(), verts.len(), 8));
        let other = distinct_quads(3);
        assert_ne!(
            fold_span(verts.as_ptr(), verts.len(), 7),
            fold_span(other.as_ptr(), other.len(), 7)
        );
    }

    #[test]
    fn fold_span_of_null_or_empty_is_zero() {
        let verts = distinct_quads(1);
        assert_eq!(fold_span(std::ptr::null(), 4, 7), 0);
        assert_eq!(fold_span(verts.as_ptr(), 0, 7), 0);
    }
}
