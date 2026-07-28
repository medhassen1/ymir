//! A column-major 4x4 `f32` matrix for chunk placement, camera view/projection,
//! and frustum culling.
//!
//! Voxel meshes are built once in chunk-local space and then repositioned
//! per frame, so the engine needs cheap composition of translation, scale,
//! and rotation, plus the camera matrices that turn world space into clip
//! space for culling and rendering.

use crate::util::vec3::Vec3;

/// A 4x4 matrix stored column-major: `cols[c][r]` is the entry at column
/// `c`, row `r`. This matches the layout GPU APIs expect for a `mat4`
/// uniform, so [`Mat4::as_flat`] can hand the raw floats to an upload
/// routine without rearranging them.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Mat4 {
    pub cols: [[f32; 4]; 4],
}

impl Mat4 {
    /// The 4x4 identity matrix.
    pub fn identity() -> Mat4 {
        Mat4 {
            cols: [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        }
    }

    /// A matrix that translates by `t`.
    pub fn from_translation(t: Vec3) -> Mat4 {
        let mut m = Mat4::identity();
        m.cols[3] = [t.x, t.y, t.z, 1.0];
        m
    }

    /// A matrix that scales each axis independently by `s`.
    pub fn from_scale(s: Vec3) -> Mat4 {
        Mat4 {
            cols: [
                [s.x, 0.0, 0.0, 0.0],
                [0.0, s.y, 0.0, 0.0],
                [0.0, 0.0, s.z, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        }
    }

    /// A matrix that rotates `angle` radians about the X axis (right-handed).
    pub fn rotation_x(angle: f32) -> Mat4 {
        let (s, c) = angle.sin_cos();
        Mat4 {
            cols: [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, c, s, 0.0],
                [0.0, -s, c, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        }
    }

    /// A matrix that rotates `angle` radians about the Y axis (right-handed).
    pub fn rotation_y(angle: f32) -> Mat4 {
        let (s, c) = angle.sin_cos();
        Mat4 {
            cols: [
                [c, 0.0, -s, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [s, 0.0, c, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        }
    }

    /// A matrix that rotates `angle` radians about the Z axis (right-handed).
    pub fn rotation_z(angle: f32) -> Mat4 {
        let (s, c) = angle.sin_cos();
        Mat4 {
            cols: [
                [c, s, 0.0, 0.0],
                [-s, c, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        }
    }

    /// Reads the entry at row `r`, column `c`.
    ///
    /// # Panics
    /// Panics if `r >= 4` or `c >= 4`.
    pub fn get(&self, r: usize, c: usize) -> f32 {
        assert!(r < 4 && c < 4, "Mat4 index out of range: ({r}, {c})");
        // SAFETY: the assert above guarantees `c < 4` and `r < 4`, and
        // `self.cols` has 4 columns of 4 elements each, so both unchecked
        // accesses are in bounds.
        unsafe { *self.cols.get_unchecked(c).get_unchecked(r) }
    }

    /// Standard matrix product `self * o` (applies `o` first, then `self`).
    pub fn multiply(&self, o: &Mat4) -> Mat4 {
        let cols: [[f32; 4]; 4] = std::array::from_fn(|c| {
            std::array::from_fn(|r| (0..4).map(|k| self.cols[k][r] * o.cols[c][k]).sum())
        });
        Mat4 { cols }
    }

    /// Transforms a point: applies rotation, scale, and translation (`w = 1`).
    pub fn transform_point(&self, p: Vec3) -> Vec3 {
        let c = &self.cols;
        Vec3 {
            x: c[0][0] * p.x + c[1][0] * p.y + c[2][0] * p.z + c[3][0],
            y: c[0][1] * p.x + c[1][1] * p.y + c[2][1] * p.z + c[3][1],
            z: c[0][2] * p.x + c[1][2] * p.y + c[2][2] * p.z + c[3][2],
        }
    }

    /// Transforms a direction: applies rotation and scale only, ignoring
    /// translation (`w = 0`), e.g. for transforming a face normal.
    pub fn transform_direction(&self, d: Vec3) -> Vec3 {
        let c = &self.cols;
        Vec3 {
            x: c[0][0] * d.x + c[1][0] * d.y + c[2][0] * d.z,
            y: c[0][1] * d.x + c[1][1] * d.y + c[2][1] * d.z,
            z: c[0][2] * d.x + c[1][2] * d.y + c[2][2] * d.z,
        }
    }

    /// The transpose of `self`.
    pub fn transpose(&self) -> Mat4 {
        let cols: [[f32; 4]; 4] = std::array::from_fn(|c| std::array::from_fn(|r| self.cols[r][c]));
        Mat4 { cols }
    }

    /// The inverse of `self`, computed by general 4x4 cofactor expansion, or
    /// `None` if `self` is singular (determinant within `f32::EPSILON` of 0).
    pub fn inverse(&self) -> Option<Mat4> {
        let m = self.as_flat();
        let mut inv = [0.0f32; 16];

        inv[0] = m[5] * m[10] * m[15] - m[5] * m[11] * m[14] - m[9] * m[6] * m[15]
            + m[9] * m[7] * m[14] + m[13] * m[6] * m[11] - m[13] * m[7] * m[10];
        inv[4] = -m[4] * m[10] * m[15] + m[4] * m[11] * m[14] + m[8] * m[6] * m[15]
            - m[8] * m[7] * m[14] - m[12] * m[6] * m[11] + m[12] * m[7] * m[10];
        inv[8] = m[4] * m[9] * m[15] - m[4] * m[11] * m[13] - m[8] * m[5] * m[15]
            + m[8] * m[7] * m[13] + m[12] * m[5] * m[11] - m[12] * m[7] * m[9];
        inv[12] = -m[4] * m[9] * m[14] + m[4] * m[10] * m[13] + m[8] * m[5] * m[14]
            - m[8] * m[6] * m[13] - m[12] * m[5] * m[10] + m[12] * m[6] * m[9];

        inv[1] = -m[1] * m[10] * m[15] + m[1] * m[11] * m[14] + m[9] * m[2] * m[15]
            - m[9] * m[3] * m[14] - m[13] * m[2] * m[11] + m[13] * m[3] * m[10];
        inv[5] = m[0] * m[10] * m[15] - m[0] * m[11] * m[14] - m[8] * m[2] * m[15]
            + m[8] * m[3] * m[14] + m[12] * m[2] * m[11] - m[12] * m[3] * m[10];
        inv[9] = -m[0] * m[9] * m[15] + m[0] * m[11] * m[13] + m[8] * m[1] * m[15]
            - m[8] * m[3] * m[13] - m[12] * m[1] * m[11] + m[12] * m[3] * m[9];
        inv[13] = m[0] * m[9] * m[14] - m[0] * m[10] * m[13] - m[8] * m[1] * m[14]
            + m[8] * m[2] * m[13] + m[12] * m[1] * m[10] - m[12] * m[2] * m[9];

        inv[2] = m[1] * m[6] * m[15] - m[1] * m[7] * m[14] - m[5] * m[2] * m[15]
            + m[5] * m[3] * m[14] + m[13] * m[2] * m[7] - m[13] * m[3] * m[6];
        inv[6] = -m[0] * m[6] * m[15] + m[0] * m[7] * m[14] + m[4] * m[2] * m[15]
            - m[4] * m[3] * m[14] - m[12] * m[2] * m[7] + m[12] * m[3] * m[6];
        inv[10] = m[0] * m[5] * m[15] - m[0] * m[7] * m[13] - m[4] * m[1] * m[15]
            + m[4] * m[3] * m[13] + m[12] * m[1] * m[7] - m[12] * m[3] * m[5];
        inv[14] = -m[0] * m[5] * m[14] + m[0] * m[6] * m[13] + m[4] * m[1] * m[14]
            - m[4] * m[2] * m[13] - m[12] * m[1] * m[6] + m[12] * m[2] * m[5];

        inv[3] = -m[1] * m[6] * m[11] + m[1] * m[7] * m[10] + m[5] * m[2] * m[11]
            - m[5] * m[3] * m[10] - m[9] * m[2] * m[7] + m[9] * m[3] * m[6];
        inv[7] = m[0] * m[6] * m[11] - m[0] * m[7] * m[10] - m[4] * m[2] * m[11]
            + m[4] * m[3] * m[10] + m[8] * m[2] * m[7] - m[8] * m[3] * m[6];
        inv[11] = -m[0] * m[5] * m[11] + m[0] * m[7] * m[9] + m[4] * m[1] * m[11]
            - m[4] * m[3] * m[9] - m[8] * m[1] * m[7] + m[8] * m[3] * m[5];
        inv[15] = m[0] * m[5] * m[10] - m[0] * m[6] * m[9] - m[4] * m[1] * m[10]
            + m[4] * m[2] * m[9] + m[8] * m[1] * m[6] - m[8] * m[2] * m[5];

        let det = m[0] * inv[0] + m[1] * inv[4] + m[2] * inv[8] + m[3] * inv[12];
        if det.abs() < f32::EPSILON {
            return None;
        }
        let inv_det = 1.0 / det;
        let cols: [[f32; 4]; 4] = std::array::from_fn(|c| std::array::from_fn(|r| inv[c * 4 + r] * inv_det));
        Some(Mat4 { cols })
    }

    /// A right-handed perspective projection with the given vertical field
    /// of view (radians), aspect ratio (width / height), and near/far
    /// clip distances, mapping depth to `[-1, 1]` (OpenGL-style clip space).
    pub fn perspective(fovy_radians: f32, aspect: f32, near: f32, far: f32) -> Mat4 {
        let f = 1.0 / (fovy_radians * 0.5).tan();
        let range_inv = 1.0 / (near - far);
        Mat4 {
            cols: [
                [f / aspect, 0.0, 0.0, 0.0],
                [0.0, f, 0.0, 0.0],
                [0.0, 0.0, (near + far) * range_inv, -1.0],
                [0.0, 0.0, 2.0 * near * far * range_inv, 0.0],
            ],
        }
    }

    /// A right-handed orthographic projection over the given box, mapping
    /// depth to `[-1, 1]` (OpenGL-style clip space).
    pub fn orthographic(left: f32, right: f32, bottom: f32, top: f32, near: f32, far: f32) -> Mat4 {
        let rl = 1.0 / (right - left);
        let tb = 1.0 / (top - bottom);
        let fn_ = 1.0 / (far - near);
        Mat4 {
            cols: [
                [2.0 * rl, 0.0, 0.0, 0.0],
                [0.0, 2.0 * tb, 0.0, 0.0],
                [0.0, 0.0, -2.0 * fn_, 0.0],
                [-(right + left) * rl, -(top + bottom) * tb, -(far + near) * fn_, 1.0],
            ],
        }
    }

    /// Views the 16 entries as a flat, column-major `&[f32; 16]`, e.g. to
    /// upload the matrix to a GPU uniform buffer without rearranging it.
    pub fn as_flat(&self) -> &[f32; 16] {
        // SAFETY: `[[f32; 4]; 4]` and `[f32; 16]` have identical size (64
        // bytes) and alignment (4 bytes): Rust arrays have no padding
        // between elements, so 4 consecutive `[f32; 4]`s occupy exactly the
        // same bytes as 16 consecutive `f32`s. `self.cols` is `#[repr(C)]`
        // and `self` outlives the returned reference, so this reinterpret
        // cast is valid.
        unsafe { &*(self.cols.as_ptr() as *const [f32; 16]) }
    }

    /// Builds a matrix from a flat, column-major array of 16 floats — the
    /// inverse of [`Mat4::as_flat`], e.g. for reading a matrix back that
    /// was serialized or received from a GPU-side buffer as raw floats.
    pub fn from_flat(flat: [f32; 16]) -> Mat4 {
        // SAFETY: `flat` is an owned, fully-initialized `[f32; 16]` local
        // value. As in `as_flat`, `[f32; 16]` and `[[f32; 4]; 4]` have
        // identical size and alignment with no inter-element padding on
        // either side, so transmuting between them preserves every bit and
        // cannot read or write out of bounds.
        let cols = unsafe { std::mem::transmute::<[f32; 16], [[f32; 4]; 4]>(flat) };
        Mat4 { cols }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f32 = 1e-4;

    fn approx_eq_mat(a: &Mat4, b: &Mat4) {
        for (col_a, col_b) in a.cols.iter().zip(b.cols.iter()) {
            for (va, vb) in col_a.iter().zip(col_b.iter()) {
                assert!((va - vb).abs() < EPS, "mismatch: {va} != {vb}");
            }
        }
    }

    fn approx_eq_vec(a: Vec3, b: Vec3) {
        assert!((a.x - b.x).abs() < EPS);
        assert!((a.y - b.y).abs() < EPS);
        assert!((a.z - b.z).abs() < EPS);
    }

    #[test]
    fn identity_multiply_is_identity() {
        let m = Mat4::from_translation(Vec3::new(1.0, 2.0, 3.0));
        approx_eq_mat(&m.multiply(&Mat4::identity()), &m);
        approx_eq_mat(&Mat4::identity().multiply(&m), &m);
    }

    #[test]
    fn translation_moves_points_not_directions() {
        let t = Mat4::from_translation(Vec3::new(5.0, 0.0, 0.0));
        approx_eq_vec(t.transform_point(Vec3::ZERO), Vec3::new(5.0, 0.0, 0.0));
        approx_eq_vec(t.transform_direction(Vec3::new(1.0, 0.0, 0.0)), Vec3::new(1.0, 0.0, 0.0));
    }

    #[test]
    fn rotation_z_quarter_turn_matches_known_result() {
        let r = Mat4::rotation_z(std::f32::consts::FRAC_PI_2);
        approx_eq_vec(r.transform_point(Vec3::new(1.0, 0.0, 0.0)), Vec3::new(0.0, 1.0, 0.0));
    }

    #[test]
    fn transpose_round_trip() {
        let m = Mat4::rotation_x(0.7).multiply(&Mat4::from_scale(Vec3::new(2.0, 3.0, 4.0)));
        approx_eq_mat(&m.transpose().transpose(), &m);
    }

    #[test]
    fn inverse_of_invertible_matrix_is_a_true_inverse_and_singular_has_none() {
        let m = Mat4::from_translation(Vec3::new(1.0, -2.0, 3.0))
            .multiply(&Mat4::rotation_y(0.3))
            .multiply(&Mat4::from_scale(Vec3::new(2.0, 1.0, 0.5)));
        let inv = m.inverse().expect("matrix should be invertible");
        approx_eq_mat(&m.multiply(&inv), &Mat4::identity());

        let singular = Mat4::from_scale(Vec3::new(0.0, 1.0, 1.0));
        assert!(singular.inverse().is_none());
    }

    #[test]
    fn as_flat_matches_column_major_layout_and_from_flat_round_trips() {
        let m = Mat4::from_translation(Vec3::new(1.0, 2.0, 3.0));
        let flat = m.as_flat();
        assert!((flat[12] - 1.0).abs() < EPS);
        assert!((flat[13] - 2.0).abs() < EPS);
        assert!((flat[14] - 3.0).abs() < EPS);
        assert!((m.get(0, 3) - 1.0).abs() < EPS);
        approx_eq_mat(&Mat4::from_flat(*flat), &m);
    }

    #[test]
    fn perspective_and_orthographic_map_near_plane_to_minus_w_or_minus_one() {
        // For a perspective matrix, clip.w == -eye.z, and the near plane
        // maps to clip.z / clip.w == -1 before the (external) perspective
        // divide.
        let p = Mat4::perspective(std::f32::consts::FRAC_PI_2, 1.0, 0.1, 100.0);
        let near_z_row = p.cols[2][2] * -0.1 + p.cols[3][2];
        let near_w_row = p.cols[2][3] * -0.1 + p.cols[3][3];
        assert!((near_z_row / near_w_row - (-1.0)).abs() < EPS);

        let o = Mat4::orthographic(-1.0, 1.0, -1.0, 1.0, 0.0, 10.0);
        approx_eq_vec(o.transform_point(Vec3::new(0.0, 0.0, 0.0)), Vec3::new(0.0, 0.0, -1.0));
    }
}
