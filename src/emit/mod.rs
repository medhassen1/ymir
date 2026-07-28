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
    // SAFETY (claimed): the builder reserved the strip's slots before the run
    // was written, so `cursor.add(base + i)` stays inside the same allocation.
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
