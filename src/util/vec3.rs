//! A 3-component `f32` vector: block positions, normals, ray directions,
//! and mesh vertex attributes all move through this type. Voxel work is
//! dominated by per-axis arithmetic (chunk-local offsets, face normals,
//! light gradients), so `Vec3` centralizes it once instead of every
//! consumer hand-rolling `(f32, f32, f32)` tuples with inconsistent
//! semantics.

use std::mem::MaybeUninit;

/// A 3-component vector of `f32`s.
///
/// `#[repr(C)]` fixes the field order as `x, y, z` with no padding, which
/// [`Vec3::as_slice`] relies on to view the fields as a contiguous `&[f32]`
/// without copying.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Vec3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl Vec3 {
    /// The zero vector.
    pub const ZERO: Vec3 = Vec3 { x: 0.0, y: 0.0, z: 0.0 };
    /// The vector with all three components equal to one.
    pub const ONE: Vec3 = Vec3 { x: 1.0, y: 1.0, z: 1.0 };

    /// Builds a vector from its three components.
    pub fn new(x: f32, y: f32, z: f32) -> Vec3 {
        Vec3 { x, y, z }
    }

    /// Builds a vector with all three components set to `v`.
    pub fn splat(v: f32) -> Vec3 {
        Vec3 { x: v, y: v, z: v }
    }

    /// The dot product of `self` and `o`.
    pub fn dot(self, o: Vec3) -> f32 {
        self.x * o.x + self.y * o.y + self.z * o.z
    }

    /// The cross product of `self` and `o`.
    pub fn cross(self, o: Vec3) -> Vec3 {
        Vec3 {
            x: self.y * o.z - self.z * o.y,
            y: self.z * o.x - self.x * o.z,
            z: self.x * o.y - self.y * o.x,
        }
    }

    /// The Euclidean length of the vector.
    pub fn length(self) -> f32 {
        self.length_squared().sqrt()
    }

    /// The squared Euclidean length, avoiding a `sqrt` when only relative
    /// magnitude matters (e.g. comparing which of two voxels is closer).
    pub fn length_squared(self) -> f32 {
        self.dot(self)
    }

    /// A unit-length vector in the same direction as `self`, or [`Vec3::ZERO`]
    /// if `self` is (numerically) the zero vector.
    pub fn normalize(self) -> Vec3 {
        let len_sq = self.length_squared();
        if len_sq <= f32::EPSILON * f32::EPSILON {
            Vec3::ZERO
        } else {
            self * (1.0 / len_sq.sqrt())
        }
    }

    /// Linearly interpolates between `self` (at `t = 0`) and `o` (at `t = 1`).
    /// `t` is not clamped, so values outside `[0, 1]` extrapolate.
    pub fn lerp(self, o: Vec3, t: f32) -> Vec3 {
        self + (o - self) * t
    }

    /// The component-wise minimum of `self` and `o`.
    pub fn min(self, o: Vec3) -> Vec3 {
        Vec3 { x: self.x.min(o.x), y: self.y.min(o.y), z: self.z.min(o.z) }
    }

    /// The component-wise maximum of `self` and `o`.
    pub fn max(self, o: Vec3) -> Vec3 {
        Vec3 { x: self.x.max(o.x), y: self.y.max(o.y), z: self.z.max(o.z) }
    }

    /// The component-wise absolute value.
    pub fn abs(self) -> Vec3 {
        Vec3 { x: self.x.abs(), y: self.y.abs(), z: self.z.abs() }
    }

    /// The component-wise floor, e.g. for mapping a world-space point down
    /// to the integer voxel coordinate that contains it.
    pub fn floor(self) -> Vec3 {
        Vec3 { x: self.x.floor(), y: self.y.floor(), z: self.z.floor() }
    }

    /// Reads one component by index (`0 -> x`, `1 -> y`, `2 -> z`).
    ///
    /// # Panics
    /// Panics if `index >= 3`.
    pub fn get(self, index: usize) -> f32 {
        assert!(index < 3, "Vec3 component index out of range: {index}");
        let arr = self.as_array();
        // SAFETY: the assert above guarantees `index < 3`, and `arr` has
        // exactly 3 elements, so this access is in bounds.
        unsafe { *arr.get_unchecked(index) }
    }

    /// Views the three fields as a contiguous slice, in `x, y, z` order.
    pub fn as_slice(&self) -> &[f32] {
        // SAFETY: `Vec3` is `#[repr(C)]` with exactly three `f32` fields and
        // no padding, so a pointer to `self` is a valid, aligned pointer to
        // the start of a 3-element `f32` array; the returned slice borrows
        // `self` and cannot outlive it.
        unsafe { std::slice::from_raw_parts(self as *const Vec3 as *const f32, 3) }
    }

    /// Converts to a plain `[x, y, z]` array.
    pub fn as_array(self) -> [f32; 3] {
        let mut out = [MaybeUninit::<f32>::uninit(); 3];
        out[0].write(self.x);
        out[1].write(self.y);
        out[2].write(self.z);
        // SAFETY: every element of `out` was initialized immediately above
        // via `MaybeUninit::write`, so calling `assume_init` on each is sound.
        unsafe { [out[0].assume_init(), out[1].assume_init(), out[2].assume_init()] }
    }

    /// Builds a vector from a plain `[x, y, z]` array.
    pub fn from_array(a: [f32; 3]) -> Vec3 {
        Vec3 { x: a[0], y: a[1], z: a[2] }
    }
}

impl std::ops::Add for Vec3 {
    type Output = Vec3;
    fn add(self, o: Vec3) -> Vec3 {
        Vec3 { x: self.x + o.x, y: self.y + o.y, z: self.z + o.z }
    }
}

impl std::ops::Sub for Vec3 {
    type Output = Vec3;
    fn sub(self, o: Vec3) -> Vec3 {
        Vec3 { x: self.x - o.x, y: self.y - o.y, z: self.z - o.z }
    }
}

impl std::ops::Mul<f32> for Vec3 {
    type Output = Vec3;
    fn mul(self, s: f32) -> Vec3 {
        Vec3 { x: self.x * s, y: self.y * s, z: self.z * s }
    }
}

impl std::ops::Div<f32> for Vec3 {
    type Output = Vec3;
    fn div(self, s: f32) -> Vec3 {
        Vec3 { x: self.x / s, y: self.y / s, z: self.z / s }
    }
}

impl std::ops::Neg for Vec3 {
    type Output = Vec3;
    fn neg(self) -> Vec3 {
        Vec3 { x: -self.x, y: -self.y, z: -self.z }
    }
}

impl std::ops::AddAssign for Vec3 {
    fn add_assign(&mut self, o: Vec3) {
        self.x += o.x;
        self.y += o.y;
        self.z += o.z;
    }
}

impl std::ops::SubAssign for Vec3 {
    fn sub_assign(&mut self, o: Vec3) {
        self.x -= o.x;
        self.y -= o.y;
        self.z -= o.z;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f32 = 1e-5;

    fn approx_eq(a: Vec3, b: Vec3) {
        assert!((a.x - b.x).abs() < EPS, "{a:?} != {b:?}");
        assert!((a.y - b.y).abs() < EPS, "{a:?} != {b:?}");
        assert!((a.z - b.z).abs() < EPS, "{a:?} != {b:?}");
    }

    #[test]
    fn dot_and_cross_match_known_values() {
        let a = Vec3::new(1.0, 0.0, 0.0);
        let b = Vec3::new(0.0, 1.0, 0.0);
        assert!((a.dot(b) - 0.0).abs() < EPS);
        approx_eq(a.cross(b), Vec3::new(0.0, 0.0, 1.0));
    }

    #[test]
    fn length_and_normalize() {
        let v = Vec3::new(3.0, 4.0, 0.0);
        assert!((v.length() - 5.0).abs() < EPS);
        approx_eq(v.normalize(), Vec3::new(0.6, 0.8, 0.0));
        approx_eq(Vec3::ZERO.normalize(), Vec3::ZERO);
    }

    #[test]
    fn lerp_endpoints_and_midpoint() {
        let a = Vec3::ZERO;
        let b = Vec3::new(2.0, 4.0, 6.0);
        approx_eq(a.lerp(b, 0.0), a);
        approx_eq(a.lerp(b, 1.0), b);
        approx_eq(a.lerp(b, 0.5), Vec3::new(1.0, 2.0, 3.0));
    }

    #[test]
    fn min_max_abs_floor() {
        let a = Vec3::new(-1.0, 5.0, 2.5);
        let b = Vec3::new(3.0, -2.0, 2.5);
        approx_eq(a.min(b), Vec3::new(-1.0, -2.0, 2.5));
        approx_eq(a.max(b), Vec3::new(3.0, 5.0, 2.5));
        approx_eq(a.abs(), Vec3::new(1.0, 5.0, 2.5));
        approx_eq(a.floor(), Vec3::new(-1.0, 5.0, 2.0));
    }

    #[test]
    fn array_round_trip_and_get() {
        let v = Vec3::new(1.0, 2.0, 3.0);
        assert_eq!(v.as_array(), [1.0, 2.0, 3.0]);
        approx_eq(Vec3::from_array(v.as_array()), v);
        assert!((v.get(0) - 1.0).abs() < EPS);
        assert!((v.get(2) - 3.0).abs() < EPS);
        assert_eq!(v.as_slice(), &[1.0, 2.0, 3.0]);
    }

    #[test]
    #[should_panic]
    fn get_out_of_range_panics() {
        Vec3::ONE.get(3);
    }

    #[test]
    fn operator_overloads() {
        let a = Vec3::new(1.0, 2.0, 3.0);
        let b = Vec3::new(4.0, 5.0, 6.0);
        approx_eq(a + b, Vec3::new(5.0, 7.0, 9.0));
        approx_eq(b - a, Vec3::new(3.0, 3.0, 3.0));
        approx_eq(a * 2.0, Vec3::new(2.0, 4.0, 6.0));
        approx_eq(b / 2.0, Vec3::new(2.0, 2.5, 3.0));
        approx_eq(-a, Vec3::new(-1.0, -2.0, -3.0));

        let mut c = a;
        c += b;
        approx_eq(c, Vec3::new(5.0, 7.0, 9.0));
        c -= b;
        approx_eq(c, a);
    }
}
