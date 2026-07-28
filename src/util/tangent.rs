//! Per-triangle tangent-space (tangent, bitangent, handedness) computation.
//!
//! Normal-mapped meshes need a tangent basis per vertex. This module derives
//! it from triangle positions and UVs, orthonormalizes against the normal,
//! and folds handedness into `w` so the bitangent is `cross(n, t) * w`.

/// Compute the tangent for one triangle, given its three positions, three
/// UVs, and its (already unit-length) face normal. The result is `[x, y, z,
/// w]` where `w` is `+1.0` or `-1.0` and encodes the handedness of the UV
/// mapping (mirrored UVs flip it).
pub fn compute_tangent(
    positions: [[f32; 3]; 3],
    uvs: [[f32; 2]; 3],
    normal: [f32; 3],
) -> [f32; 4] {
    let edge1 = sub3(positions[1], positions[0]);
    let edge2 = sub3(positions[2], positions[0]);
    let duv1 = sub2(uvs[1], uvs[0]);
    let duv2 = sub2(uvs[2], uvs[0]);

    let denom = duv1[0] * duv2[1] - duv2[0] * duv1[1];
    let f = if denom.abs() > 1e-12 { 1.0 / denom } else { 0.0 };

    let raw_tangent = scale3(
        sub3(scale3(edge1, duv2[1]), scale3(edge2, duv1[1])),
        f,
    );
    let raw_bitangent = scale3(
        sub3(scale3(edge2, duv1[0]), scale3(edge1, duv2[0])),
        f,
    );

    // Gram-Schmidt: remove any component of the raw tangent along the
    // normal, then renormalize.
    let t_minus_n = sub3(raw_tangent, scale3(normal, dot3(normal, raw_tangent)));
    let tangent = normalize3(t_minus_n);

    let handedness = sign_from_dot(cross3(normal, tangent), raw_bitangent);
    [tangent[0], tangent[1], tangent[2], handedness]
}

/// Compute the tangent for a triangle addressed by index into shared
/// position and UV buffers, and a precomputed per-vertex (or per-face)
/// normal buffer of the same length.
pub fn tangent_indexed(
    positions: &[[f32; 3]],
    uvs: &[[f32; 2]],
    normals: &[[f32; 3]],
    tri: [u32; 3],
) -> [f32; 4] {
    let len = positions.len();
    assert!(len == uvs.len() && len == normals.len(), "buffer length mismatch");
    assert!(tri.iter().all(|&i| (i as usize) < len), "index out of range");
    // SAFETY: the asserts above establish that `positions`, `uvs`, and
    // `normals` all have length `len`, and that every index in `tri` is
    // strictly less than `len`. Each `get_unchecked` therefore reads a
    // valid, initialized element of its respective slice.
    let (p, u, n) = unsafe {
        let p = [
            *positions.get_unchecked(tri[0] as usize),
            *positions.get_unchecked(tri[1] as usize),
            *positions.get_unchecked(tri[2] as usize),
        ];
        let u = [
            *uvs.get_unchecked(tri[0] as usize),
            *uvs.get_unchecked(tri[1] as usize),
            *uvs.get_unchecked(tri[2] as usize),
        ];
        let n = *normals.get_unchecked(tri[0] as usize);
        (p, u, n)
    };
    compute_tangent(p, u, n)
}

/// Build the orthonormal `[tangent, bitangent, normal]` basis matrix (rows)
/// from a tangent-with-handedness and a normal, reconstructing the
/// bitangent as `cross(normal, tangent) * w` rather than storing it.
pub fn orthonormal_basis(tangent_w: [f32; 4], normal: [f32; 3]) -> [[f32; 3]; 3] {
    let tangent = [tangent_w[0], tangent_w[1], tangent_w[2]];
    let bitangent = scale3(cross3(normal, tangent), tangent_w[3]);
    let mut out: std::mem::MaybeUninit<[[f32; 3]; 3]> = std::mem::MaybeUninit::uninit();
    let base = out.as_mut_ptr() as *mut [f32; 3];
    let rows = [tangent, bitangent, normal];
    for (i, row) in rows.into_iter().enumerate() {
        // SAFETY: `base` points at the first of 3 `[f32; 3]` slots carved
        // out of `out`'s own storage, and `i` ranges over `0..3` (the
        // length of `rows`), so `base.add(i)` never leaves that array. Each
        // slot is written exactly once, before `assume_init` runs below.
        unsafe { base.add(i).write(row) };
    }
    // SAFETY: the loop above wrote all 3 rows (indices 0, 1, 2) through
    // `base`, which points into `out`'s storage, so every byte of `out` is
    // now initialized and `assume_init` is valid.
    unsafe { out.assume_init() }
}

/// Sign of `dot(cross(n, t), b)`, read from a 2-entry table rather than a
/// branch: `idx` is the boolean "is non-negative" cast to `usize`, so it is
/// always exactly 0 or 1.
#[inline]
fn sign_from_dot(cross_nt: [f32; 3], b: [f32; 3]) -> f32 {
    const SIGN: [f32; 2] = [-1.0, 1.0];
    let idx = (dot3(cross_nt, b) >= 0.0) as usize;
    // SAFETY: `idx` comes from casting a `bool` to `usize`, so it is either
    // 0 or 1, and `SIGN` has exactly 2 elements — always in bounds.
    unsafe { *SIGN.get_unchecked(idx) }
}

fn sub3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn sub2(a: [f32; 2], b: [f32; 2]) -> [f32; 2] {
    [a[0] - b[0], a[1] - b[1]]
}

fn scale3(a: [f32; 3], s: f32) -> [f32; 3] {
    [a[0] * s, a[1] * s, a[2] * s]
}

fn dot3(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn cross3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn normalize3(v: [f32; 3]) -> [f32; 3] {
    let len = dot3(v, v).sqrt();
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
    fn tangent_of_axis_aligned_quad_points_along_u() {
        // A quad in the XZ plane (normal +Y) with UVs aligned to X/Z should
        // produce a tangent pointing along +X.
        let positions = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]];
        let uvs = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
        let normal = [0.0, 1.0, 0.0];
        let t = compute_tangent(positions, uvs, normal);
        assert!(approx_eq3([t[0], t[1], t[2]], [1.0, 0.0, 0.0], 1e-4));
    }

    #[test]
    fn tangent_is_orthogonal_to_normal() {
        let positions = [[0.0, 0.0, 0.0], [2.0, 0.3, 0.0], [0.5, 0.0, 1.7]];
        let uvs = [[0.0, 0.0], [1.0, 0.2], [0.1, 1.0]];
        let normal = [0.0, 0.0, 1.0];
        let t = compute_tangent(positions, uvs, normal);
        let dot = t[0] * normal[0] + t[1] * normal[1] + t[2] * normal[2];
        assert!(dot.abs() < 1e-4);
    }

    #[test]
    fn tangent_is_unit_length() {
        let positions = [[0.0, 0.0, 0.0], [2.0, 0.3, 0.0], [0.5, 0.0, 1.7]];
        let uvs = [[0.0, 0.0], [1.0, 0.2], [0.1, 1.0]];
        let normal = [0.0, 0.0, 1.0];
        let t = compute_tangent(positions, uvs, normal);
        let len = (t[0] * t[0] + t[1] * t[1] + t[2] * t[2]).sqrt();
        assert!((len - 1.0).abs() < 1e-4);
    }

    #[test]
    fn mirrored_uvs_flip_handedness() {
        let positions = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]];
        let normal = [0.0, 1.0, 0.0];
        let uvs_normal = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
        let uvs_mirrored = [[1.0, 0.0], [0.0, 0.0], [1.0, 1.0]];
        let t1 = compute_tangent(positions, uvs_normal, normal);
        let t2 = compute_tangent(positions, uvs_mirrored, normal);
        // Handedness is always +/-1.0, and mirroring the UV winding must
        // flip its sign relative to the unmirrored mapping.
        assert!(t1[3] == 1.0 || t1[3] == -1.0);
        assert_eq!(t2[3], -t1[3]);
    }

    #[test]
    fn tangent_indexed_matches_direct_computation() {
        let positions = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]];
        let uvs = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
        let normals = [[0.0, 1.0, 0.0]; 3];
        let direct = compute_tangent(positions, uvs, normals[0]);
        let indexed = tangent_indexed(&positions, &uvs, &normals, [0, 1, 2]);
        assert_eq!(direct, indexed);
    }

    #[test]
    fn orthonormal_basis_rows_are_mutually_perpendicular() {
        let positions = [[0.0, 0.0, 0.0], [2.0, 0.3, 0.0], [0.5, 0.0, 1.7]];
        let uvs = [[0.0, 0.0], [1.0, 0.2], [0.1, 1.0]];
        let normal = [0.0, 0.0, 1.0];
        let t = compute_tangent(positions, uvs, normal);
        let basis = orthonormal_basis(t, normal);
        for i in 0..3 {
            for j in (i + 1)..3 {
                let d = dot3(basis[i], basis[j]);
                assert!(d.abs() < 1e-3, "rows {i} and {j} not orthogonal: {d}");
            }
        }
    }
}
