//! A parametric ray, `origin + t * dir`, for picking the voxel under the
//! cursor, casting visibility rays for lighting, and line-of-sight checks
//! between entities.
//!
//! Everything here answers "does this ray hit that shape, and where" —
//! the [`Aabb`] test in particular is the hot path for descending an
//! octree or chunk grid during picking.

use std::mem::MaybeUninit;

use crate::util::aabb::Aabb;
use crate::util::vec3::Vec3;

/// A ray with an origin point and a direction. `dir` is not required to be
/// normalized; callers that need `t` in world-space units should normalize
/// it first.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Ray {
    pub origin: Vec3,
    pub dir: Vec3,
}

impl Ray {
    /// Builds a ray from an origin and direction.
    pub fn new(origin: Vec3, dir: Vec3) -> Ray {
        Ray { origin, dir }
    }

    /// The point at parameter `t`: `origin + dir * t`.
    pub fn at(&self, t: f32) -> Vec3 {
        self.origin + self.dir * t
    }

    /// Slab-method intersection against an axis-aligned box.
    ///
    /// Returns `Some((t_min, t_max))` — the entry and exit parameters,
    /// clamped so `t_min >= 0` — if the ray hits the box ahead of its
    /// origin, or `None` if it misses or the box is entirely behind it.
    pub fn intersect_aabb(&self, aabb: &Aabb) -> Option<(f32, f32)> {
        let o = self.origin.as_array();
        let d = self.dir.as_array();
        let lo = aabb.min.as_array();
        let hi = aabb.max.as_array();

        let mut t_near = [MaybeUninit::<f32>::uninit(); 3];
        let mut t_far = [MaybeUninit::<f32>::uninit(); 3];
        for (i, (near_slot, far_slot)) in t_near.iter_mut().zip(t_far.iter_mut()).enumerate() {
            // SAFETY: `i` ranges over `0..3` (the length of `t_near`, which
            // `enumerate` drives this loop over), and `o`, `d`, `lo`, `hi`
            // are all `[f32; 3]`, so every unchecked read here is in bounds.
            let (oi, di, loi, hii) = unsafe {
                (*o.get_unchecked(i), *d.get_unchecked(i), *lo.get_unchecked(i), *hi.get_unchecked(i))
            };
            if di.abs() < f32::EPSILON {
                // Parallel to this axis's slab: a hit requires the origin
                // to already lie within the slab.
                if oi < loi || oi > hii {
                    return None;
                }
                near_slot.write(f32::NEG_INFINITY);
                far_slot.write(f32::INFINITY);
            } else {
                let inv_d = 1.0 / di;
                let mut t0 = (loi - oi) * inv_d;
                let mut t1 = (hii - oi) * inv_d;
                if t0 > t1 {
                    std::mem::swap(&mut t0, &mut t1);
                }
                near_slot.write(t0);
                far_slot.write(t1);
            }
        }
        // SAFETY: the loop above wrote every element of both `t_near` and
        // `t_far` (one write per iteration, over all 3 elements), so both
        // arrays are fully initialized.
        let (t_near, t_far) = unsafe {
            (
                [t_near[0].assume_init(), t_near[1].assume_init(), t_near[2].assume_init()],
                [t_far[0].assume_init(), t_far[1].assume_init(), t_far[2].assume_init()],
            )
        };

        let t_min = t_near.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let t_max = t_far.iter().copied().fold(f32::INFINITY, f32::min);
        if t_max < t_min || t_max < 0.0 {
            None
        } else {
            Some((t_min.max(0.0), t_max))
        }
    }

    /// Intersection with the plane `{ p : p.dot(normal) == d }`. Returns
    /// `None` if the ray is parallel to the plane or the hit is behind the
    /// origin.
    pub fn intersect_plane(&self, normal: Vec3, d: f32) -> Option<f32> {
        let denom = normal.dot(self.dir);
        if denom.abs() < f32::EPSILON {
            return None;
        }
        let t = (d - normal.dot(self.origin)) / denom;
        if t < 0.0 {
            None
        } else {
            Some(t)
        }
    }

    /// Intersection with a sphere of the given `center` and `radius`.
    /// Returns the nearest non-negative hit parameter, if any.
    pub fn intersect_sphere(&self, center: Vec3, radius: f32) -> Option<f32> {
        let m = self.origin - center;
        let b = m.dot(self.dir);
        let c = m.length_squared() - radius * radius;
        // Ray origin outside the sphere and pointing away from it: no hit.
        if c > 0.0 && b > 0.0 {
            return None;
        }
        let a = self.dir.length_squared();
        if a < f32::EPSILON {
            return None;
        }
        let discriminant = b * b - a * c;
        if discriminant < 0.0 {
            return None;
        }
        let sqrt_disc = discriminant.sqrt();
        let t0 = (-b - sqrt_disc) / a;
        let t1 = (-b + sqrt_disc) / a;
        let t = if t0 >= 0.0 { t0 } else { t1 };
        if t < 0.0 {
            None
        } else {
            Some(t)
        }
    }

    /// Möller-Trumbore triangle intersection. Returns the hit parameter
    /// `t` (with `t >= 0`) if the ray hits the triangle `(v0, v1, v2)`
    /// (either winding), or `None` for a miss, a parallel ray, or a hit
    /// behind the origin.
    pub fn intersect_triangle(&self, v0: Vec3, v1: Vec3, v2: Vec3) -> Option<f32> {
        let edge1 = v1 - v0;
        let edge2 = v2 - v0;
        let pvec = self.dir.cross(edge2);
        let det = edge1.dot(pvec);
        if det.abs() < f32::EPSILON {
            return None;
        }
        let inv_det = 1.0 / det;
        let tvec = self.origin - v0;
        let u = tvec.dot(pvec) * inv_det;
        if !(0.0..=1.0).contains(&u) {
            return None;
        }
        let qvec = tvec.cross(edge1);
        let v = self.dir.dot(qvec) * inv_det;
        if v < 0.0 || u + v > 1.0 {
            return None;
        }
        let t = edge2.dot(qvec) * inv_det;
        if t < 0.0 {
            None
        } else {
            Some(t)
        }
    }

    /// Views `origin` and `dir` as a contiguous `[ox, oy, oz, dx, dy, dz]`
    /// slice.
    pub fn as_slice(&self) -> &[f32] {
        // SAFETY: `Ray` is `#[repr(C)]` with two `#[repr(C)]` `Vec3` fields
        // (`origin` then `dir`), each three contiguous `f32`s with no
        // padding, so a pointer to `self` is a valid, aligned pointer to
        // the start of a 6-element `f32` array; the returned slice borrows
        // `self` and cannot outlive it.
        unsafe { std::slice::from_raw_parts(self as *const Ray as *const f32, 6) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f32 = 1e-4;

    #[test]
    fn at_follows_the_parametric_line() {
        let r = Ray::new(Vec3::new(1.0, 0.0, 0.0), Vec3::new(0.0, 1.0, 0.0));
        let p = r.at(2.0);
        assert!((p.x - 1.0).abs() < EPS && (p.y - 2.0).abs() < EPS);
    }

    #[test]
    fn aabb_hit_from_outside_and_miss() {
        let b = Aabb::from_min_max(Vec3::new(-1.0, -1.0, -1.0), Vec3::new(1.0, 1.0, 1.0));
        let hit = Ray::new(Vec3::new(-5.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0));
        let (t_min, t_max) = hit.intersect_aabb(&b).expect("should hit the box");
        assert!((t_min - 4.0).abs() < EPS && (t_max - 6.0).abs() < EPS);

        let miss = Ray::new(Vec3::new(-5.0, 5.0, 0.0), Vec3::new(1.0, 0.0, 0.0));
        assert!(miss.intersect_aabb(&b).is_none());
    }

    #[test]
    fn aabb_ray_starting_inside_clamps_t_min_to_zero() {
        let b = Aabb::from_min_max(Vec3::new(-1.0, -1.0, -1.0), Vec3::new(1.0, 1.0, 1.0));
        let r = Ray::new(Vec3::ZERO, Vec3::new(1.0, 0.0, 0.0));
        let (t_min, t_max) = r.intersect_aabb(&b).expect("origin inside the box should hit");
        assert!(t_min.abs() < EPS);
        assert!((t_max - 1.0).abs() < EPS);
    }

    #[test]
    fn plane_intersection_matches_known_distance() {
        let r = Ray::new(Vec3::new(0.0, 5.0, 0.0), Vec3::new(0.0, -1.0, 0.0));
        let t = r.intersect_plane(Vec3::new(0.0, 1.0, 0.0), 0.0).expect("should hit the plane");
        assert!((t - 5.0).abs() < EPS);

        let parallel = Ray::new(Vec3::new(0.0, 5.0, 0.0), Vec3::new(1.0, 0.0, 0.0));
        assert!(parallel.intersect_plane(Vec3::new(0.0, 1.0, 0.0), 0.0).is_none());
    }

    #[test]
    fn sphere_intersection_from_outside_and_miss() {
        let r = Ray::new(Vec3::new(-5.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0));
        let t = r.intersect_sphere(Vec3::ZERO, 1.0).expect("should hit the sphere");
        assert!((t - 4.0).abs() < EPS);

        let miss = Ray::new(Vec3::new(-5.0, 5.0, 0.0), Vec3::new(1.0, 0.0, 0.0));
        assert!(miss.intersect_sphere(Vec3::ZERO, 1.0).is_none());
    }

    #[test]
    fn triangle_intersection_hits_center_and_misses_outside() {
        let v0 = Vec3::new(-1.0, -1.0, 0.0);
        let v1 = Vec3::new(1.0, -1.0, 0.0);
        let v2 = Vec3::new(0.0, 1.0, 0.0);
        let hit = Ray::new(Vec3::new(0.0, -0.3, -5.0), Vec3::new(0.0, 0.0, 1.0));
        let t = hit.intersect_triangle(v0, v1, v2).expect("should hit the triangle");
        assert!((t - 5.0).abs() < EPS);

        let miss = Ray::new(Vec3::new(10.0, 10.0, -5.0), Vec3::new(0.0, 0.0, 1.0));
        assert!(miss.intersect_triangle(v0, v1, v2).is_none());
    }

    #[test]
    fn as_slice_matches_origin_then_dir() {
        let r = Ray::new(Vec3::new(1.0, 2.0, 3.0), Vec3::new(4.0, 5.0, 6.0));
        assert_eq!(r.as_slice(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }
}
