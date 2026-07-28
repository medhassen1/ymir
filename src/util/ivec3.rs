//! A 3-component `i32` vector for exact voxel-grid coordinates.
//!
//! World and local block positions, chunk coordinates, and section offsets
//! must never suffer float rounding, so the engine keeps them as `IVec3`
//! everywhere until the moment they are needed for rendering or physics,
//! at which point [`IVec3::to_vec3`] promotes to the floating-point [`Vec3`].

use std::mem::MaybeUninit;

use crate::util::vec3::Vec3;

/// A 3-component vector of `i32`s.
///
/// `#[repr(C)]` fixes the field order as `x, y, z` with no padding, which
/// [`IVec3::as_slice`] relies on to view the fields as a contiguous
/// `&[i32]` without copying.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct IVec3 {
    pub x: i32,
    pub y: i32,
    pub z: i32,
}

impl IVec3 {
    /// The zero vector.
    pub const ZERO: IVec3 = IVec3 { x: 0, y: 0, z: 0 };

    /// Builds a vector from its three components.
    pub fn new(x: i32, y: i32, z: i32) -> IVec3 {
        IVec3 { x, y, z }
    }

    /// Builds a vector with all three components set to `v`.
    pub fn splat(v: i32) -> IVec3 {
        IVec3 { x: v, y: v, z: v }
    }

    /// The component-wise minimum of `self` and `o`.
    pub fn min(self, o: IVec3) -> IVec3 {
        IVec3 { x: self.x.min(o.x), y: self.y.min(o.y), z: self.z.min(o.z) }
    }

    /// The component-wise maximum of `self` and `o`.
    pub fn max(self, o: IVec3) -> IVec3 {
        IVec3 { x: self.x.max(o.x), y: self.y.max(o.y), z: self.z.max(o.z) }
    }

    /// The Manhattan (L1) distance between `self` and `o`, widened to `i64`
    /// since the sum of three `i32` absolute differences can overflow `i32`.
    pub fn manhattan(self, o: IVec3) -> i64 {
        let dx = (self.x as i64 - o.x as i64).abs();
        let dy = (self.y as i64 - o.y as i64).abs();
        let dz = (self.z as i64 - o.z as i64).abs();
        dx + dy + dz
    }

    /// The Chebyshev (L-infinity) distance between `self` and `o`: the
    /// number of chunk-grid steps a king-move traversal would need.
    pub fn chebyshev(self, o: IVec3) -> i32 {
        let dx = (self.x - o.x).abs();
        let dy = (self.y - o.y).abs();
        let dz = (self.z - o.z).abs();
        dx.max(dy).max(dz)
    }

    /// Widens to a floating-point [`Vec3`], e.g. to place a block's center
    /// in world space.
    pub fn to_vec3(self) -> Vec3 {
        Vec3::new(self.x as f32, self.y as f32, self.z as f32)
    }

    /// Reads one component by index (`0 -> x`, `1 -> y`, `2 -> z`).
    ///
    /// # Panics
    /// Panics if `index >= 3`.
    pub fn get(self, index: usize) -> i32 {
        assert!(index < 3, "IVec3 component index out of range: {index}");
        let arr = self.as_array();
        // SAFETY: the assert above guarantees `index < 3`, and `arr` has
        // exactly 3 elements, so this access is in bounds.
        unsafe { *arr.get_unchecked(index) }
    }

    /// Views the three fields as a contiguous slice, in `x, y, z` order.
    pub fn as_slice(&self) -> &[i32] {
        // SAFETY: `IVec3` is `#[repr(C)]` with exactly three `i32` fields
        // and no padding, so a pointer to `self` is a valid, aligned
        // pointer to the start of a 3-element `i32` array; the returned
        // slice borrows `self` and cannot outlive it.
        unsafe { std::slice::from_raw_parts(self as *const IVec3 as *const i32, 3) }
    }

    /// Converts to a plain `[x, y, z]` array.
    pub fn as_array(self) -> [i32; 3] {
        let mut out = [MaybeUninit::<i32>::uninit(); 3];
        out[0].write(self.x);
        out[1].write(self.y);
        out[2].write(self.z);
        // SAFETY: every element of `out` was initialized immediately above
        // via `MaybeUninit::write`, so calling `assume_init` on each is sound.
        unsafe { [out[0].assume_init(), out[1].assume_init(), out[2].assume_init()] }
    }
}

impl std::ops::Add for IVec3 {
    type Output = IVec3;
    fn add(self, o: IVec3) -> IVec3 {
        IVec3 { x: self.x + o.x, y: self.y + o.y, z: self.z + o.z }
    }
}

impl std::ops::Sub for IVec3 {
    type Output = IVec3;
    fn sub(self, o: IVec3) -> IVec3 {
        IVec3 { x: self.x - o.x, y: self.y - o.y, z: self.z - o.z }
    }
}

impl std::ops::Mul<i32> for IVec3 {
    type Output = IVec3;
    fn mul(self, s: i32) -> IVec3 {
        IVec3 { x: self.x * s, y: self.y * s, z: self.z * s }
    }
}

impl std::ops::Neg for IVec3 {
    type Output = IVec3;
    fn neg(self) -> IVec3 {
        IVec3 { x: -self.x, y: -self.y, z: -self.z }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn min_max_component_wise() {
        let a = IVec3::new(-1, 5, 2);
        let b = IVec3::new(3, -2, 2);
        assert_eq!(a.min(b), IVec3::new(-1, -2, 2));
        assert_eq!(a.max(b), IVec3::new(3, 5, 2));
    }

    #[test]
    fn manhattan_and_chebyshev_distance() {
        let a = IVec3::new(0, 0, 0);
        let b = IVec3::new(3, -4, 5);
        assert_eq!(a.manhattan(b), 12);
        assert_eq!(a.chebyshev(b), 5);
    }

    #[test]
    fn manhattan_does_not_overflow_i32() {
        let a = IVec3::splat(i32::MIN);
        let b = IVec3::splat(i32::MAX);
        // Each axis difference is ~2^32, and the i32 sum would overflow;
        // widening to i64 must carry the true magnitude through.
        assert_eq!(a.manhattan(b), 3 * (i32::MAX as i64 - i32::MIN as i64));
    }

    #[test]
    fn to_vec3_widens_components() {
        let v = IVec3::new(1, -2, 3).to_vec3();
        assert!((v.x - 1.0).abs() < f32::EPSILON);
        assert!((v.y + 2.0).abs() < f32::EPSILON);
        assert!((v.z - 3.0).abs() < f32::EPSILON);
    }

    #[test]
    fn array_round_trip_and_get() {
        let v = IVec3::new(7, 8, 9);
        assert_eq!(v.as_array(), [7, 8, 9]);
        assert_eq!(v.get(0), 7);
        assert_eq!(v.get(2), 9);
        assert_eq!(v.as_slice(), &[7, 8, 9]);
    }

    #[test]
    #[should_panic]
    fn get_out_of_range_panics() {
        IVec3::ZERO.get(3);
    }

    #[test]
    fn operator_overloads() {
        let a = IVec3::new(1, 2, 3);
        let b = IVec3::new(4, 5, 6);
        assert_eq!(a + b, IVec3::new(5, 7, 9));
        assert_eq!(b - a, IVec3::new(3, 3, 3));
        assert_eq!(a * 2, IVec3::new(2, 4, 6));
        assert_eq!(-a, IVec3::new(-1, -2, -3));
    }
}
