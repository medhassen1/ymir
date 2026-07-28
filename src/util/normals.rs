//! Vertex and face normal computation, plus octahedral normal encoding.
//!
//! The mesher needs per-face and area-weighted smooth normals; shipping a
//! full normal per vertex is expensive at chunk scale, so this module also
//! packs a unit vector into one octahedral-encoded `u32`.

/// Compute the (unnormalized-then-normalized) normal of a single triangle
/// given its three corner positions, using the right-hand winding rule.
pub fn face_normal(a: [f32; 3], b: [f32; 3], c: [f32; 3]) -> [f32; 3] {
    let u = sub(b, a);
    let v = sub(c, a);
    normalize(cross(u, v))
}

/// Compute the face normal for a triangle given by indices into a shared
/// position buffer. Indices are bounds-checked once, up front.
pub fn face_normal_indexed(positions: &[[f32; 3]], tri: [u32; 3]) -> [f32; 3] {
    let len = positions.len();
    assert!(
        tri.iter().all(|&i| (i as usize) < len),
        "triangle index out of range"
    );
    // SAFETY: the assert above confirmed all three of `tri`'s indices are
    // strictly less than `positions.len()`, so each `get_unchecked` call
    // reads a valid, initialized element of `positions`.
    let (a, b, c) = unsafe {
        (
            *positions.get_unchecked(tri[0] as usize),
            *positions.get_unchecked(tri[1] as usize),
            *positions.get_unchecked(tri[2] as usize),
        )
    };
    face_normal(a, b, c)
}

/// Compute area-weighted smooth vertex normals for an indexed triangle mesh.
/// Each triangle contributes its (unnormalized) cross-product normal to all
/// three corners, so larger triangles pull harder on the shared vertex
/// normal before the final per-vertex normalize.
pub fn smooth_normals(positions: &[[f32; 3]], indices: &[[u32; 3]]) -> Vec<[f32; 3]> {
    let mut acc = vec![[0.0f32; 3]; positions.len()];
    for tri in indices {
        if tri.iter().any(|&i| i as usize >= positions.len()) {
            continue;
        }
        let a = positions[tri[0] as usize];
        let b = positions[tri[1] as usize];
        let c = positions[tri[2] as usize];
        let weighted = cross(sub(b, a), sub(c, a));
        for &idx in tri {
            let slot = &mut acc[idx as usize];
            slot[0] += weighted[0];
            slot[1] += weighted[1];
            slot[2] += weighted[2];
        }
    }
    for n in acc.iter_mut() {
        *n = normalize(*n);
    }
    acc
}

/// Flatten a slice of 3-float positions into a contiguous `&[f32]`, useful
/// for handing raw vertex data to a buffer-upload routine that wants one
/// flat float stream instead of an array of triples.
pub fn flatten_positions(positions: &[[f32; 3]]) -> &[f32] {
    let len = positions.len() * 3;
    // SAFETY: `[f32; 3]` has the same size (12 bytes) and alignment (4) as
    // three consecutive `f32`s with no padding between or within elements —
    // Rust guarantees arrays are laid out as their element type repeated
    // `N` times contiguously. `positions.as_ptr()` is therefore also a
    // valid, correctly aligned `*const f32` pointing at `len` initialized
    // `f32` values, and the returned slice borrows from `positions` so it
    // cannot outlive the backing allocation.
    unsafe { std::slice::from_raw_parts(positions.as_ptr() as *const f32, len) }
}

/// A fixed two-entry sign lookup, used by the octahedral codec below. `idx`
/// is always exactly 0 or 1 (a `bool` cast to `usize`), so this never
/// touches the table out of bounds; it exists to avoid a branch in a
/// function called twice per encode and twice per decode.
#[inline]
fn sign_lut(x: f32) -> f32 {
    const LUT: [f32; 2] = [-1.0, 1.0];
    let idx = (x >= 0.0) as usize;
    // SAFETY: `idx` is the result of casting a `bool` to `usize`, so its
    // only possible values are 0 and 1, and `LUT` has exactly 2 elements —
    // the index is always in bounds.
    unsafe { *LUT.get_unchecked(idx) }
}

/// Encode a (near-)unit vector to the octahedral `[-1, 1]^2` plane. See
/// Cigolle et al., "A Survey of Efficient Representations for Independent
/// Unit Vectors". Input need not be perfectly normalized.
pub fn octahedral_encode(n: [f32; 3]) -> [f32; 2] {
    let l1 = n[0].abs() + n[1].abs() + n[2].abs();
    let inv = if l1 > 0.0 { 1.0 / l1 } else { 0.0 };
    let mut p = [n[0] * inv, n[1] * inv];
    if n[2] < 0.0 {
        p = [
            (1.0 - p[1].abs()) * sign_lut(p[0]),
            (1.0 - p[0].abs()) * sign_lut(p[1]),
        ];
    }
    p
}

/// Decode a point produced by [`octahedral_encode`] back to a unit vector.
pub fn octahedral_decode(p: [f32; 2]) -> [f32; 3] {
    let mut n = [p[0], p[1], 1.0 - p[0].abs() - p[1].abs()];
    if n[2] < 0.0 {
        let old_x = n[0];
        n[0] = (1.0 - n[1].abs()) * sign_lut(old_x);
        n[1] = (1.0 - old_x.abs()) * sign_lut(n[1]);
    }
    normalize(n)
}

/// Pack a unit vector into a `u32`: two 16-bit signed-normalized lanes over
/// the octahedral plane, X in the low 16 bits and Y in the high 16 bits.
pub fn pack_octahedral(n: [f32; 3]) -> u32 {
    let p = octahedral_encode(n);
    let qx = snorm16(p[0]) as u16;
    let qy = snorm16(p[1]) as u16;
    (qx as u32) | ((qy as u32) << 16)
}

/// Inverse of [`pack_octahedral`]: unpack a `u32` back to a unit vector.
pub fn unpack_octahedral(bits: u32) -> [f32; 3] {
    let qx = (bits & 0xFFFF) as u16 as i16;
    let qy = ((bits >> 16) & 0xFFFF) as u16 as i16;
    octahedral_decode([dequantize_snorm16(qx), dequantize_snorm16(qy)])
}

#[inline]
fn snorm16(x: f32) -> i16 {
    (x.clamp(-1.0, 1.0) * i16::MAX as f32).round() as i16
}

#[inline]
fn dequantize_snorm16(q: i16) -> f32 {
    (q as f32 / i16::MAX as f32).clamp(-1.0, 1.0)
}

fn sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn normalize(v: [f32; 3]) -> [f32; 3] {
    let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if len <= f32::EPSILON {
        return [0.0, 0.0, 0.0];
    }
    [v[0] / len, v[1] / len, v[2] / len]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq3(a: [f32; 3], b: [f32; 3], eps: f32) -> bool {
        (0..3).all(|i| (a[i] - b[i]).abs() <= eps)
    }

    #[test]
    fn face_normal_of_xy_triangle_points_along_z() {
        let n = face_normal([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]);
        assert!(approx_eq3(n, [0.0, 0.0, 1.0], 1e-5));
    }

    #[test]
    fn face_normal_indexed_matches_direct_computation() {
        let positions = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let direct = face_normal(positions[0], positions[1], positions[2]);
        let indexed = face_normal_indexed(&positions, [0, 1, 2]);
        assert!(approx_eq3(direct, indexed, 1e-6));
    }

    #[test]
    fn smooth_normals_average_two_coplanar_triangles() {
        // A quad split into two triangles sharing an edge; both faces are
        // coplanar, so every vertex normal should equal the shared face
        // normal exactly (same weight direction, only magnitude differs).
        let positions = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ];
        let indices = [[0, 1, 2], [0, 2, 3]];
        let normals = smooth_normals(&positions, &indices);
        for n in normals {
            assert!(approx_eq3(n, [0.0, 0.0, 1.0], 1e-5));
        }
    }

    #[test]
    fn flatten_positions_preserves_component_order() {
        let positions = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let flat = flatten_positions(&positions);
        assert_eq!(flat, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn octahedral_round_trip_axis_aligned_vectors() {
        let axes = [
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
        ];
        for n in axes {
            let bits = pack_octahedral(n);
            let back = unpack_octahedral(bits);
            assert!(approx_eq3(back, n, 0.01), "n={n:?} back={back:?}");
        }
    }

    #[test]
    fn octahedral_round_trip_general_direction_stays_unit_and_close() {
        let n = normalize([0.3, 0.7, -0.4]);
        let bits = pack_octahedral(n);
        let back = unpack_octahedral(bits);
        let len = (back[0] * back[0] + back[1] * back[1] + back[2] * back[2]).sqrt();
        assert!((len - 1.0).abs() < 1e-4);
        assert!(approx_eq3(back, n, 0.01));
    }
}
