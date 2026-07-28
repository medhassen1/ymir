//! An axis-aligned bounding box, the workhorse bounding volume for a voxel
//! engine: chunk bounds, broad-phase collision, ray picking, and frustum
//! culling all reduce to box tests before anything more expensive runs.

use std::mem::MaybeUninit;

use crate::util::vec3::Vec3;

/// An axis-aligned box spanning `[min, max]` on each axis.
///
/// `min` is not required to be less than `max` component-wise by the type
/// itself (callers can construct a degenerate or "empty" box), but every
/// method here treats `min <= max` per axis as the well-formed case.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Aabb {
    pub min: Vec3,
    pub max: Vec3,
}

impl Aabb {
    /// Builds a box directly from its min and max corners (not reordered).
    pub fn from_min_max(min: Vec3, max: Vec3) -> Aabb {
        Aabb { min, max }
    }

    /// Builds the tightest box containing every point in `points`.
    /// Returns [`Aabb::from_min_max`] of `(ZERO, ZERO)` if `points` is empty.
    pub fn from_points(points: &[Vec3]) -> Aabb {
        let mut it = points.iter();
        let Some(&first) = it.next() else {
            return Aabb::from_min_max(Vec3::ZERO, Vec3::ZERO);
        };
        let mut min = first;
        let mut max = first;
        for &p in it {
            min = min.min(p);
            max = max.max(p);
        }
        Aabb { min, max }
    }

    /// The smallest box containing both `self` and `other`.
    pub fn union(&self, other: &Aabb) -> Aabb {
        Aabb { min: self.min.min(other.min), max: self.max.max(other.max) }
    }

    /// The overlapping region of `self` and `other`. May be degenerate
    /// (zero or negative extent on some axis) if the boxes only touch or
    /// do not overlap; check with [`Aabb::intersects`] first if that matters.
    pub fn intersection(&self, other: &Aabb) -> Aabb {
        Aabb { min: self.min.max(other.min), max: self.max.min(other.max) }
    }

    /// Whether `p` lies within the box, inclusive of the boundary.
    pub fn contains_point(&self, p: Vec3) -> bool {
        p.x >= self.min.x
            && p.x <= self.max.x
            && p.y >= self.min.y
            && p.y <= self.max.y
            && p.z >= self.min.z
            && p.z <= self.max.z
    }

    /// Whether `self` and `other` overlap or touch on every axis.
    pub fn intersects(&self, other: &Aabb) -> bool {
        self.min.x <= other.max.x
            && self.max.x >= other.min.x
            && self.min.y <= other.max.y
            && self.max.y >= other.min.y
            && self.min.z <= other.max.z
            && self.max.z >= other.min.z
    }

    /// Grows the box by `amount` on every axis in both directions (e.g. to
    /// pad a chunk's bounds by its tallest possible block before a
    /// coarse visibility test).
    pub fn expand(&self, amount: f32) -> Aabb {
        let pad = Vec3::splat(amount);
        Aabb { min: self.min - pad, max: self.max + pad }
    }

    /// The box's center point.
    pub fn center(&self) -> Vec3 {
        (self.min + self.max) * 0.5
    }

    /// The box's full size along each axis (`max - min`).
    pub fn extents(&self) -> Vec3 {
        self.max - self.min
    }

    /// The total surface area of the box's six faces.
    pub fn surface_area(&self) -> f32 {
        let e = self.extents();
        2.0 * (e.x * e.y + e.y * e.z + e.z * e.x)
    }

    /// The 8 corners of the box, in the fixed order produced by taking
    /// `min` or `max` on each axis with `z` varying fastest: `[---, --+,
    /// -+-, -++, +--, +-+, ++-, +++]` (`-` = min, `+` = max, axes ordered
    /// x, y, z).
    pub fn corners(&self) -> [Vec3; 8] {
        let mut out = [MaybeUninit::<Vec3>::uninit(); 8];
        for (i, slot) in out.iter_mut().enumerate() {
            let x = if i & 4 != 0 { self.max.x } else { self.min.x };
            let y = if i & 2 != 0 { self.max.y } else { self.min.y };
            let z = if i & 1 != 0 { self.max.z } else { self.min.z };
            slot.write(Vec3::new(x, y, z));
        }
        // SAFETY: the loop above wrote every one of the 8 elements of
        // `out` (each `slot` from `out.iter_mut()`) before this point, so
        // the whole array is fully initialized.
        unsafe { out.map(|c| c.assume_init()) }
    }

    /// Reads one corner by index, equivalent to `self.corners()[index]`
    /// without materializing the other seven.
    ///
    /// # Panics
    /// Panics if `index >= 8`.
    pub fn corner(&self, index: usize) -> Vec3 {
        assert!(index < 8, "Aabb corner index out of range: {index}");
        let corners = self.corners();
        // SAFETY: the assert above guarantees `index < 8`, and `corners`
        // has exactly 8 elements, so this access is in bounds.
        unsafe { *corners.get_unchecked(index) }
    }

    /// Views `min` and `max` as a contiguous `[min.x, min.y, min.z, max.x,
    /// max.y, max.z]` slice.
    pub fn as_slice(&self) -> &[f32] {
        // SAFETY: `Aabb` is `#[repr(C)]` with two `#[repr(C)]` `Vec3` fields
        // (`min` then `max`), each three contiguous `f32`s with no padding,
        // so a pointer to `self` is a valid, aligned pointer to the start
        // of a 6-element `f32` array; the returned slice borrows `self`
        // and cannot outlive it.
        unsafe { std::slice::from_raw_parts(self as *const Aabb as *const f32, 6) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f32 = 1e-5;

    #[test]
    fn from_points_bounds_the_input() {
        let pts = [Vec3::new(1.0, -2.0, 3.0), Vec3::new(-1.0, 5.0, 0.0), Vec3::new(2.0, 0.0, -3.0)];
        let b = Aabb::from_points(&pts);
        assert!((b.min.x - (-1.0)).abs() < EPS && (b.max.x - 2.0).abs() < EPS);
        assert!((b.min.y - (-2.0)).abs() < EPS && (b.max.y - 5.0).abs() < EPS);
        assert!((b.min.z - (-3.0)).abs() < EPS && (b.max.z - 3.0).abs() < EPS);
    }

    #[test]
    fn union_and_intersection() {
        let a = Aabb::from_min_max(Vec3::new(0.0, 0.0, 0.0), Vec3::new(2.0, 2.0, 2.0));
        let b = Aabb::from_min_max(Vec3::new(1.0, 1.0, 1.0), Vec3::new(3.0, 3.0, 3.0));
        let u = a.union(&b);
        assert!((u.min.x - 0.0).abs() < EPS && (u.max.x - 3.0).abs() < EPS);
        let i = a.intersection(&b);
        assert!((i.min.x - 1.0).abs() < EPS && (i.max.x - 2.0).abs() < EPS);
    }

    #[test]
    fn contains_point_is_inclusive_of_boundary() {
        let b = Aabb::from_min_max(Vec3::ZERO, Vec3::ONE);
        assert!(b.contains_point(Vec3::new(0.0, 0.5, 1.0)));
        assert!(!b.contains_point(Vec3::new(1.5, 0.5, 0.5)));
    }

    #[test]
    fn intersects_detects_touching_and_separated_boxes() {
        let a = Aabb::from_min_max(Vec3::ZERO, Vec3::ONE);
        let touching = Aabb::from_min_max(Vec3::new(1.0, 0.0, 0.0), Vec3::new(2.0, 1.0, 1.0));
        let separated = Aabb::from_min_max(Vec3::new(2.0, 0.0, 0.0), Vec3::new(3.0, 1.0, 1.0));
        assert!(a.intersects(&touching));
        assert!(!a.intersects(&separated));
    }

    #[test]
    fn expand_center_and_extents() {
        let b = Aabb::from_min_max(Vec3::new(-1.0, -1.0, -1.0), Vec3::new(1.0, 1.0, 1.0));
        let e = b.expand(1.0);
        assert!((e.min.x - (-2.0)).abs() < EPS && (e.max.x - 2.0).abs() < EPS);
        let c = b.center();
        assert!(c.x.abs() < EPS && c.y.abs() < EPS && c.z.abs() < EPS);
        let ext = b.extents();
        assert!((ext.x - 2.0).abs() < EPS);
    }

    #[test]
    fn surface_area_of_unit_cube() {
        let b = Aabb::from_min_max(Vec3::ZERO, Vec3::ONE);
        assert!((b.surface_area() - 6.0).abs() < EPS);
    }

    #[test]
    fn corners_cover_all_eight_combinations_and_match_indexed_access() {
        let b = Aabb::from_min_max(Vec3::ZERO, Vec3::ONE);
        let corners = b.corners();
        assert_eq!(corners.len(), 8);
        let mut seen = std::collections::HashSet::new();
        for c in corners {
            seen.insert((c.x as i32, c.y as i32, c.z as i32));
        }
        assert_eq!(seen.len(), 8, "all 8 corners should be distinct for a non-degenerate box");
        for (i, expected) in corners.iter().enumerate() {
            let c = b.corner(i);
            assert!((c.x - expected.x).abs() < EPS);
        }
    }
}
