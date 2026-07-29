//! Vertex-strip post-processing and mesh folding.
//!
//! The mesher writes a run of merged quads into the vertex buffer and then hands
//! the strip to [`weld_strip`], which walks the freshly written corners and
//! blends their lighting so that adjacent merged quads meet without a seam.
//! Working through the strip cursor keeps the weld a single forward pass over
//! contiguous memory.

use crate::mesh::Vertex;

/// Weld the corner lighting of a freshly written vertex strip.
///
/// `cursor` addresses the vertex buffer, `base` is the index the strip starts
/// at, and `count` is how many vertices it covers. Corners are visited four at a
/// time; each quad's trailing pair inherits a blend of the leading pair so a run
/// of merged faces shades continuously.
///
/// SAFETY: `cursor` must address a vertex buffer with at least `base + count`
/// initialised slots for the duration of the call.
pub fn weld_strip(cursor: *mut Vertex, base: usize, count: usize) {
    if cursor.is_null() || count == 0 {
        return;
    }
    // SAFETY: per this function's contract, `cursor` addresses at least
    // `base + count` initialised slots, so every `cursor.add(base + i)` below
    // stays inside that allocation.
    unsafe {
        let mut i = 0usize;
        while i + 3 < count {
            let a = *cursor.add(base + i);
            let b = *cursor.add(base + i + 1);
            let blend = ((a.light & 0xffff_ff00) + (b.light & 0xffff_ff00)) / 2;
            let c = cursor.add(base + i + 2);
            let d = cursor.add(base + i + 3);
            (*c).light = ((*c).light & 0xff) | (blend & 0xffff_ff00);
            (*d).light = ((*d).light & 0xff) | (blend & 0xffff_ff00);
            i += 4;
        }
    }
}

/// How many corners of each side an adjacency seam covers.
///
/// Two columns meet along one quad's worth of edge, so the seam is the trailing
/// quad of the earlier mesh against the leading quad of the later one.
const SEAM_CORNERS: usize = 4;

/// Fold the seam between two column meshes that meet along a shared boundary.
///
/// `left` is the mesh committed first and `right` the one committed against it.
/// The fold reads the trailing corners of `left` and the leading corners of
/// `right` — the two ends that describe the same edge of the world — so a
/// region's digest reflects how its columns join, not just what each contains.
/// Both sides are walked through raw pointer arithmetic because a committed
/// mesh is named by where it landed rather than by an owned buffer.
///
/// SAFETY: `left` must address at least `left_len` live [`Vertex`] values and
/// `right` at least `right_len`, for the duration of the call.
pub fn fold_boundary(
    left: *const Vertex,
    left_len: usize,
    right: *const Vertex,
    right_len: usize,
) -> u64 {
    if left.is_null() || right.is_null() {
        return 0;
    }
    let take = SEAM_CORNERS.min(left_len).min(right_len);
    if take == 0 {
        return 0;
    }
    let mut acc = ((left_len ^ right_len) as u64).wrapping_mul(0x9e3779b1);
    // SAFETY: per this function's contract both sides address at least their
    // stated vertex counts, and `take` is bounded by both, so the trailing
    // `take` corners of `left` and the leading `take` of `right` are in range.
    unsafe {
        for i in 0..take {
            let a = *left.add(left_len - take + i);
            let b = *right.add(i);
            acc = acc.rotate_left(9) ^ (a.pos as u64) ^ ((a.light as u64) << 5);
            acc = acc.rotate_left(3) ^ (b.pos as u64) ^ ((b.light as u64) << 1);
        }
    }
    acc
}

/// Fold a digest of a finished mesh.
pub fn fold_mesh(verts: &[Vertex]) -> u64 {
    let mut acc = (verts.len() as u64).wrapping_mul(0x9e3779b1);
    for v in verts {
        acc = acc.rotate_left(7) ^ (v.pos as u64);
        acc = acc.rotate_left(3) ^ (v.norm as u64);
        acc = acc.wrapping_add(v.light as u64);
    }
    acc
}

/// The axis-aligned extent of a mesh in packed position space.
pub fn mesh_extent(verts: &[Vertex]) -> (u32, u32) {
    let mut lo = u32::MAX;
    let mut hi = 0u32;
    for v in verts {
        lo = lo.min(v.pos);
        hi = hi.max(v.pos);
    }
    if verts.is_empty() {
        (0, 0)
    } else {
        (lo, hi)
    }
}

/// How many distinct faces a mesh touches, from the packed normal words.
pub fn face_variety(verts: &[Vertex]) -> usize {
    let mut seen = 0u8;
    for v in verts {
        let face = (v.norm >> 16) & 0x7;
        if face < 6 {
            seen |= 1 << face;
        }
    }
    seen.count_ones() as usize
}

#[cfg(test)]
mod tests {
    use super::*;

    fn strip(n: usize) -> Vec<Vertex> {
        (0..n)
            .map(|i| Vertex {
                pos: i as u32,
                norm: ((i as u32 % 6) << 16) | 0x2a,
                light: 0x0000_1100 | (i as u32 & 3),
            })
            .collect()
    }

    #[test]
    fn weld_blends_trailing_corners() {
        let mut v = strip(4);
        v[0].light = 0x0000_1000;
        v[1].light = 0x0000_3000;
        let before_low = v[2].light & 0xff;
        weld_strip(v.as_mut_ptr(), 0, 4);
        // Trailing corners take the blend, keeping their low corner index byte.
        assert_eq!(v[2].light & 0xffff_ff00, 0x0000_2000);
        assert_eq!(v[2].light & 0xff, before_low);
        assert_eq!(v[3].light & 0xffff_ff00, 0x0000_2000);
    }

    #[test]
    fn weld_ignores_partial_quads() {
        let mut v = strip(3);
        let before = v.clone();
        weld_strip(v.as_mut_ptr(), 0, 3);
        assert_eq!(v, before, "a partial quad must be left alone");
    }

    #[test]
    fn weld_on_empty_or_null_is_a_noop() {
        let mut v = strip(4);
        let before = v.clone();
        weld_strip(v.as_mut_ptr(), 0, 0);
        assert_eq!(v, before);
        weld_strip(std::ptr::null_mut(), 0, 4);
    }

    #[test]
    fn fold_boundary_reads_both_ends_of_the_seam() {
        let left = strip(8);
        let right = strip(8);
        let base = fold_boundary(left.as_ptr(), left.len(), right.as_ptr(), right.len());
        // Changing a corner outside the seam must not move the digest.
        let mut untouched = left.clone();
        untouched[0].pos ^= 0xff;
        assert_eq!(
            base,
            fold_boundary(untouched.as_ptr(), untouched.len(), right.as_ptr(), right.len())
        );
        // Changing the trailing corner of the left side must.
        let mut moved = left.clone();
        moved[7].pos ^= 0xff;
        assert_ne!(
            base,
            fold_boundary(moved.as_ptr(), moved.len(), right.as_ptr(), right.len())
        );
        // As must changing the leading corner of the right side.
        let mut other = right.clone();
        other[0].light ^= 0xff00;
        assert_ne!(
            base,
            fold_boundary(left.as_ptr(), left.len(), other.as_ptr(), other.len())
        );
    }

    #[test]
    fn fold_boundary_clamps_to_the_shorter_side() {
        let left = strip(8);
        let short = strip(2);
        // A two-vertex mesh has no full quad to offer, so only what both sides
        // carry is folded — and nothing is read past either end.
        assert_ne!(
            fold_boundary(left.as_ptr(), left.len(), short.as_ptr(), short.len()),
            0
        );
        assert_eq!(fold_boundary(left.as_ptr(), left.len(), short.as_ptr(), 0), 0);
        assert_eq!(fold_boundary(std::ptr::null(), 4, short.as_ptr(), 2), 0);
        assert_eq!(fold_boundary(left.as_ptr(), 4, std::ptr::null(), 2), 0);
    }

    #[test]
    fn fold_mesh_is_order_sensitive() {
        let a = strip(8);
        let mut b = a.clone();
        b.swap(0, 7);
        assert_ne!(fold_mesh(&a), fold_mesh(&b));
    }

    #[test]
    fn mesh_extent_brackets_positions() {
        let v = strip(5);
        let (lo, hi) = mesh_extent(&v);
        assert_eq!(lo, 0);
        assert_eq!(hi, 4);
        assert_eq!(mesh_extent(&[]), (0, 0));
    }

    #[test]
    fn face_variety_counts_distinct_faces() {
        assert_eq!(face_variety(&strip(6)), 6);
        assert_eq!(face_variety(&strip(2)), 2);
        assert_eq!(face_variety(&[]), 0);
    }
}
