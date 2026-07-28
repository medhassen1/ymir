//! Infinite planes in the `normal . p + d = 0` form.
//!
//! Frustum culling, clipping a mesh against a chunk boundary, and slicing
//! a structure template to fit a build volume all reduce to "which side of
//! this plane is this point on" and "where do these planes meet". Storing
//! a plane as `(normal, d)` rather than three points keeps every one of
//! those tests a single dot product.

use crate::util::vec3::Vec3;

/// Distance below which [`Plane::classify`] treats a point as lying on
/// the plane rather than strictly in front of or behind it.
pub const CLASSIFY_EPSILON: f32 = 1e-5;

/// Which side of a plane a point falls on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Side {
    /// In the half-space the normal points into.
    Front,
    /// In the half-space the normal points away from.
    Back,
    /// Within [`CLASSIFY_EPSILON`] of the plane itself.
    On,
}

/// A plane defined by a unit `normal` and offset `d`, satisfying
/// `normal.dot(p) + d == 0` for every point `p` on the plane.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Plane {
    pub normal: Vec3,
    pub d: f32,
}

impl Plane {
    /// Builds a plane through `point` with the given (not necessarily
    /// normalized) `normal`.
    pub fn from_point_normal(point: Vec3, normal: Vec3) -> Plane {
        let n = normal.normalize();
        Plane { normal: n, d: -n.dot(point) }
    }

    /// Builds a plane through three points, with the normal following the
    /// right-hand rule from `a -> b` to `a -> c` (counter-clockwise
    /// winding, viewed from the side the normal points toward).
    pub fn from_points(a: Vec3, b: Vec3, c: Vec3) -> Plane {
        let normal = (b - a).cross(c - a).normalize();
        Self::from_point_normal(a, normal)
    }

    /// The signed distance from `point` to this plane: positive on the
    /// side `normal` points into, negative on the other side, `0` on the
    /// plane.
    pub fn signed_distance(&self, point: Vec3) -> f32 {
        self.normal.dot(point) + self.d
    }

    /// The closest point on this plane to `point`.
    pub fn project_point(&self, point: Vec3) -> Vec3 {
        point - self.normal * self.signed_distance(point)
    }

    /// Classifies `point` as [`Side::Front`], [`Side::Back`], or
    /// [`Side::On`] (within [`CLASSIFY_EPSILON`]).
    pub fn classify(&self, point: Vec3) -> Side {
        let dist = self.signed_distance(point);
        if dist > CLASSIFY_EPSILON {
            Side::Front
        } else if dist < -CLASSIFY_EPSILON {
            Side::Back
        } else {
            Side::On
        }
    }

    /// The single point where three planes all meet, or `None` if any
    /// pair is parallel (no unique intersection point exists).
    pub fn intersect_three_planes(a: &Plane, b: &Plane, c: &Plane) -> Option<Vec3> {
        let bc = b.normal.cross(c.normal);
        let denom = a.normal.dot(bc);
        if denom.abs() < f32::EPSILON {
            return None;
        }
        let ca = c.normal.cross(a.normal);
        let ab = a.normal.cross(b.normal);
        let p = (bc * -a.d + ca * -b.d + ab * -c.d) / denom;
        Some(p)
    }

    /// Computes [`signed_distance`](Plane::signed_distance) for many
    /// points against this plane in one pass, writing directly into a
    /// pre-sized buffer instead of growing a `Vec` one push at a time.
    pub fn signed_distances_batch(&self, points: &[Vec3]) -> Vec<f32> {
        let mut out: Vec<f32> = Vec::with_capacity(points.len());
        let ptr = out.as_mut_ptr();
        for (i, &p) in points.iter().enumerate() {
            let dist = self.signed_distance(p);
            // SAFETY: `ptr` comes from `Vec::with_capacity(points.len())`,
            // so it has room for `points.len()` elements; `i` ranges over
            // `0..points.len()` (the enumeration of `points`), so
            // `ptr.add(i)` stays within that reserved capacity, and each
            // index is written exactly once before `set_len` runs below.
            unsafe {
                ptr.add(i).write(dist);
            }
        }
        // SAFETY: the loop above wrote every index `0..points.len()`
        // exactly once, so `out`'s first `points.len()` elements are all
        // initialized, and that length does not exceed the reserved
        // capacity.
        unsafe {
            out.set_len(points.len());
        }
        out
    }

    /// Classifies many points against this plane in one pass, writing
    /// directly into a pre-sized buffer instead of growing a `Vec` one
    /// push at a time. Used to split a mesh's vertex list into
    /// front/back/on groups without re-testing each vertex individually.
    pub fn classify_batch(&self, points: &[Vec3]) -> Vec<Side> {
        let mut out: Vec<Side> = Vec::with_capacity(points.len());
        let ptr = out.as_mut_ptr();
        for (i, &p) in points.iter().enumerate() {
            let side = self.classify(p);
            // SAFETY: identical reasoning to `signed_distances_batch`:
            // `ptr` has room for `points.len()` elements and `i` ranges
            // over `0..points.len()`.
            unsafe {
                ptr.add(i).write(side);
            }
        }
        // SAFETY: every index `0..points.len()` was written exactly once
        // above, matching the reserved capacity.
        unsafe {
            out.set_len(points.len());
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f32 = 1e-4;

    #[test]
    fn from_point_normal_gives_zero_distance_on_the_plane() {
        let plane = Plane::from_point_normal(Vec3::new(0.0, 5.0, 0.0), Vec3::new(0.0, 1.0, 0.0));
        assert!(plane.signed_distance(Vec3::new(10.0, 5.0, -3.0)).abs() < EPS);
        assert!((plane.signed_distance(Vec3::new(0.0, 6.0, 0.0)) - 1.0).abs() < EPS);
    }

    #[test]
    fn from_points_normal_is_orthogonal_to_both_edges() {
        let a = Vec3::new(0.0, 0.0, 0.0);
        let b = Vec3::new(1.0, 0.0, 0.0);
        let c = Vec3::new(0.0, 1.0, 0.0);
        let plane = Plane::from_points(a, b, c);
        assert!(plane.normal.dot(b - a).abs() < EPS);
        assert!(plane.normal.dot(c - a).abs() < EPS);
        assert!((plane.normal.length() - 1.0).abs() < EPS);
    }

    #[test]
    fn classify_matches_the_sign_of_the_distance() {
        let plane = Plane::from_point_normal(Vec3::ZERO, Vec3::new(0.0, 0.0, 1.0));
        assert_eq!(plane.classify(Vec3::new(0.0, 0.0, 5.0)), Side::Front);
        assert_eq!(plane.classify(Vec3::new(0.0, 0.0, -5.0)), Side::Back);
        assert_eq!(plane.classify(Vec3::new(3.0, -2.0, 0.0)), Side::On);
    }

    #[test]
    fn project_point_lands_exactly_on_the_plane() {
        let plane = Plane::from_point_normal(Vec3::new(1.0, 0.0, 0.0), Vec3::new(1.0, 1.0, 0.0));
        let p = Vec3::new(10.0, -4.0, 7.0);
        let projected = plane.project_point(p);
        assert!(plane.signed_distance(projected).abs() < EPS);
    }

    #[test]
    fn intersect_three_planes_recovers_a_known_corner() {
        let x = Plane::from_point_normal(Vec3::new(1.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0));
        let y = Plane::from_point_normal(Vec3::new(0.0, 2.0, 0.0), Vec3::new(0.0, 1.0, 0.0));
        let z = Plane::from_point_normal(Vec3::new(0.0, 0.0, 3.0), Vec3::new(0.0, 0.0, 1.0));
        let p = Plane::intersect_three_planes(&x, &y, &z).unwrap();
        assert!((p.x - 1.0).abs() < EPS);
        assert!((p.y - 2.0).abs() < EPS);
        assert!((p.z - 3.0).abs() < EPS);
    }

    #[test]
    fn parallel_planes_have_no_unique_intersection() {
        let a = Plane::from_point_normal(Vec3::ZERO, Vec3::new(0.0, 1.0, 0.0));
        let b = Plane::from_point_normal(Vec3::new(0.0, 5.0, 0.0), Vec3::new(0.0, 1.0, 0.0));
        let c = Plane::from_point_normal(Vec3::ZERO, Vec3::new(0.0, 0.0, 1.0));
        assert!(Plane::intersect_three_planes(&a, &b, &c).is_none());
    }

    #[test]
    fn batch_helpers_match_their_scalar_counterparts() {
        let plane = Plane::from_point_normal(Vec3::new(0.0, 0.0, 2.0), Vec3::new(0.0, 0.0, 1.0));
        let points = [Vec3::new(0.0, 0.0, 5.0), Vec3::new(0.0, 0.0, 2.0), Vec3::new(0.0, 0.0, -1.0)];
        let distances = plane.signed_distances_batch(&points);
        let sides = plane.classify_batch(&points);
        for (i, &p) in points.iter().enumerate() {
            assert!((distances[i] - plane.signed_distance(p)).abs() < EPS);
            assert_eq!(sides[i], plane.classify(p));
        }
    }
}
