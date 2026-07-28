//! A unit quaternion for rotating cameras, entities, and structure
//! instances without the gimbal lock or drift that accumulated Euler
//! angles suffer from.
//!
//! Structures pasted into the world (and the camera rig following the
//! player) need smooth, composable rotation; quaternions compose cheaply
//! via [`Quat::multiply`] and interpolate cleanly via [`Quat::slerp`],
//! which a raw [`crate::util::mat4::Mat4`] cannot do without care.

use std::mem::MaybeUninit;

use crate::util::vec3::Vec3;

/// A quaternion `x*i + y*j + z*k + w`. Most operations assume it is unit
/// length; [`Quat::normalize`] restores that after accumulated error.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Quat {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub w: f32,
}

impl Quat {
    /// The identity rotation (no rotation).
    pub fn identity() -> Quat {
        Quat { x: 0.0, y: 0.0, z: 0.0, w: 1.0 }
    }

    /// Builds a rotation of `angle` radians about `axis`. `axis` need not be
    /// normalized; the zero vector yields the identity rotation.
    pub fn from_axis_angle(axis: Vec3, angle: f32) -> Quat {
        let axis = axis.normalize();
        if axis == Vec3::ZERO {
            return Quat::identity();
        }
        let (s, c) = (angle * 0.5).sin_cos();
        Quat { x: axis.x * s, y: axis.y * s, z: axis.z * s, w: c }
    }

    /// Builds a rotation from Euler angles (radians), applied in
    /// roll (Z) - pitch (X) - yaw (Y) order: `yaw * pitch * roll`.
    pub fn from_euler(pitch_x: f32, yaw_y: f32, roll_z: f32) -> Quat {
        let (sx, cx) = (pitch_x * 0.5).sin_cos();
        let (sy, cy) = (yaw_y * 0.5).sin_cos();
        let (sz, cz) = (roll_z * 0.5).sin_cos();
        // yaw (Y) * pitch (X) * roll (Z), expanded directly to avoid three
        // separate quaternion multiplies.
        Quat {
            x: sx * cy * cz + cx * sy * sz,
            y: cx * sy * cz - sx * cy * sz,
            z: cx * cy * sz - sx * sy * cz,
            w: cx * cy * cz + sx * sy * sz,
        }
    }

    /// Hamilton product `self * o`: applies `o` first, then `self`.
    pub fn multiply(self, o: Quat) -> Quat {
        Quat {
            x: self.w * o.x + self.x * o.w + self.y * o.z - self.z * o.y,
            y: self.w * o.y - self.x * o.z + self.y * o.w + self.z * o.x,
            z: self.w * o.z + self.x * o.y - self.y * o.x + self.z * o.w,
            w: self.w * o.w - self.x * o.x - self.y * o.y - self.z * o.z,
        }
    }

    /// The conjugate: for a unit quaternion, this is also the inverse
    /// rotation.
    pub fn conjugate(self) -> Quat {
        Quat { x: -self.x, y: -self.y, z: -self.z, w: self.w }
    }

    /// The squared length of the quaternion as a 4-vector.
    pub fn length_squared(self) -> f32 {
        self.x * self.x + self.y * self.y + self.z * self.z + self.w * self.w
    }

    /// A unit-length quaternion representing the same rotation as `self`,
    /// or [`Quat::identity`] if `self` is (numerically) the zero quaternion.
    pub fn normalize(self) -> Quat {
        let len_sq = self.length_squared();
        if len_sq <= f32::EPSILON * f32::EPSILON {
            return Quat::identity();
        }
        let inv_len = 1.0 / len_sq.sqrt();
        Quat { x: self.x * inv_len, y: self.y * inv_len, z: self.z * inv_len, w: self.w * inv_len }
    }

    /// Spherical linear interpolation from `self` (at `t = 0`) to `o` (at
    /// `t = 1`), taking the shorter arc. Falls back to normalized linear
    /// interpolation when the two quaternions are nearly parallel, since
    /// the slerp formula is numerically unstable there.
    pub fn slerp(self, o: Quat, t: f32) -> Quat {
        let mut dot = self.x * o.x + self.y * o.y + self.z * o.z + self.w * o.w;
        let mut o = o;
        if dot < 0.0 {
            // Take the shorter arc: negating a quaternion yields the same
            // rotation, but the interpolation path differs.
            o = Quat { x: -o.x, y: -o.y, z: -o.z, w: -o.w };
            dot = -dot;
        }
        if dot > 0.9995 {
            let lerped = Quat {
                x: self.x + (o.x - self.x) * t,
                y: self.y + (o.y - self.y) * t,
                z: self.z + (o.z - self.z) * t,
                w: self.w + (o.w - self.w) * t,
            };
            return lerped.normalize();
        }
        let theta_0 = dot.acos();
        let theta = theta_0 * t;
        let (sin_theta, sin_theta_0) = (theta.sin(), theta_0.sin());
        let s0 = theta.cos() - dot * sin_theta / sin_theta_0;
        let s1 = sin_theta / sin_theta_0;
        Quat {
            x: self.x * s0 + o.x * s1,
            y: self.y * s0 + o.y * s1,
            z: self.z * s0 + o.z * s1,
            w: self.w * s0 + o.w * s1,
        }
    }

    /// Rotates `v` by this quaternion (assumed unit length).
    pub fn rotate_vec3(self, v: Vec3) -> Vec3 {
        let q_vec = Vec3::new(self.x, self.y, self.z);
        let t = q_vec.cross(v) * 2.0;
        v + t * self.w + q_vec.cross(t)
    }

    /// Reads one component by index (`0 -> x, 1 -> y, 2 -> z, 3 -> w`).
    ///
    /// # Panics
    /// Panics if `index >= 4`.
    pub fn get(self, index: usize) -> f32 {
        assert!(index < 4, "Quat component index out of range: {index}");
        let arr = self.as_array();
        // SAFETY: the assert above guarantees `index < 4`, and `arr` has
        // exactly 4 elements, so this access is in bounds.
        unsafe { *arr.get_unchecked(index) }
    }

    /// Views the four fields as a contiguous slice, in `x, y, z, w` order.
    pub fn as_slice(&self) -> &[f32] {
        // SAFETY: `Quat` is `#[repr(C)]` with exactly four `f32` fields and
        // no padding, so a pointer to `self` is a valid, aligned pointer to
        // the start of a 4-element `f32` array; the returned slice borrows
        // `self` and cannot outlive it.
        unsafe { std::slice::from_raw_parts(self as *const Quat as *const f32, 4) }
    }

    /// Converts to a plain `[x, y, z, w]` array.
    pub fn as_array(self) -> [f32; 4] {
        let mut out = [MaybeUninit::<f32>::uninit(); 4];
        out[0].write(self.x);
        out[1].write(self.y);
        out[2].write(self.z);
        out[3].write(self.w);
        // SAFETY: every element of `out` was initialized immediately above
        // via `MaybeUninit::write`, so calling `assume_init` on each is sound.
        unsafe { [out[0].assume_init(), out[1].assume_init(), out[2].assume_init(), out[3].assume_init()] }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f32 = 1e-4;

    fn approx_eq_vec(a: Vec3, b: Vec3) {
        assert!((a.x - b.x).abs() < EPS, "{a:?} != {b:?}");
        assert!((a.y - b.y).abs() < EPS, "{a:?} != {b:?}");
        assert!((a.z - b.z).abs() < EPS, "{a:?} != {b:?}");
    }

    fn approx_eq_quat(a: Quat, b: Quat) {
        assert!((a.x - b.x).abs() < EPS && (a.y - b.y).abs() < EPS && (a.z - b.z).abs() < EPS && (a.w - b.w).abs() < EPS, "{a:?} != {b:?}");
    }

    #[test]
    fn identity_rotates_nothing() {
        let v = Vec3::new(1.0, 2.0, 3.0);
        approx_eq_vec(Quat::identity().rotate_vec3(v), v);
    }

    #[test]
    fn quarter_turn_about_z_matches_known_result() {
        let q = Quat::from_axis_angle(Vec3::new(0.0, 0.0, 1.0), std::f32::consts::FRAC_PI_2);
        approx_eq_vec(q.rotate_vec3(Vec3::new(1.0, 0.0, 0.0)), Vec3::new(0.0, 1.0, 0.0));
    }

    #[test]
    fn conjugate_undoes_rotation() {
        let q = Quat::from_axis_angle(Vec3::new(1.0, 1.0, 0.0), 1.2);
        let v = Vec3::new(0.3, -0.7, 2.0);
        approx_eq_vec(q.conjugate().rotate_vec3(q.rotate_vec3(v)), v);
    }

    #[test]
    fn multiply_composes_rotations() {
        let q1 = Quat::from_axis_angle(Vec3::new(0.0, 0.0, 1.0), std::f32::consts::FRAC_PI_2);
        let q2 = Quat::from_axis_angle(Vec3::new(0.0, 0.0, 1.0), std::f32::consts::FRAC_PI_2);
        let combined = q2.multiply(q1);
        let full_turn = Quat::from_axis_angle(Vec3::new(0.0, 0.0, 1.0), std::f32::consts::PI);
        approx_eq_vec(combined.rotate_vec3(Vec3::new(1.0, 0.0, 0.0)), full_turn.rotate_vec3(Vec3::new(1.0, 0.0, 0.0)));
    }

    #[test]
    fn slerp_endpoints_match_inputs_and_midpoint_stays_unit_length() {
        let a = Quat::identity();
        let b = Quat::from_axis_angle(Vec3::new(0.0, 1.0, 0.0), 1.0);
        approx_eq_quat(a.slerp(b, 0.0), a);
        approx_eq_quat(a.slerp(b, 1.0), b);

        let c = Quat::from_axis_angle(Vec3::new(1.0, 0.0, 0.0), 2.0);
        let mid = a.slerp(c, 0.5);
        assert!((mid.length_squared() - 1.0).abs() < 1e-3);
    }

    #[test]
    fn from_euler_single_axis_matches_from_axis_angle() {
        approx_eq_quat(Quat::from_euler(0.6, 0.0, 0.0), Quat::from_axis_angle(Vec3::new(1.0, 0.0, 0.0), 0.6));
        approx_eq_quat(Quat::from_euler(0.0, 0.6, 0.0), Quat::from_axis_angle(Vec3::new(0.0, 1.0, 0.0), 0.6));
        approx_eq_quat(Quat::from_euler(0.0, 0.0, 0.6), Quat::from_axis_angle(Vec3::new(0.0, 0.0, 1.0), 0.6));
    }

    #[test]
    fn normalize_and_array_round_trip() {
        let q = Quat { x: 2.0, y: 0.0, z: 0.0, w: 0.0 };
        let n = q.normalize();
        assert!((n.length_squared() - 1.0).abs() < EPS);
        assert_eq!(q.as_array(), [2.0, 0.0, 0.0, 0.0]);
        assert!((q.get(0) - 2.0).abs() < EPS);
        assert_eq!(q.as_slice(), &[2.0, 0.0, 0.0, 0.0]);
    }
}
