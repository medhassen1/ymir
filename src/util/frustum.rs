//! A view frustum, extracted from a combined view-projection matrix, for
//! culling chunks and meshes that cannot possibly be visible before
//! spending time building or drawing them.
//!
//! Testing a chunk's [`Aabb`] against six planes is far cheaper than
//! rendering it, so this is run once per chunk per frame ahead of the
//! mesh upload, and is the main reason a large loaded radius stays
//! affordable.

use std::mem::MaybeUninit;

use crate::util::aabb::Aabb;
use crate::util::mat4::Mat4;
use crate::util::vec3::Vec3;

/// How a box relates to a frustum.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Containment {
    /// The box lies entirely outside at least one plane: definitely not visible.
    Outside,
    /// The box straddles at least one plane: partially visible, needs finer testing.
    Intersect,
    /// The box lies entirely inside all six planes: fully visible.
    Inside,
}

/// Six frustum planes, each stored as `[a, b, c, d]` with unit-length
/// normal `(a, b, c)`, such that a point `p` is on the inside of the plane
/// when `a*p.x + b*p.y + c*p.z + d >= 0`.
///
/// Plane order is `[left, right, bottom, top, near, far]`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Frustum {
    pub planes: [[f32; 4]; 6],
}

fn normalize_plane(p: [f32; 4]) -> [f32; 4] {
    let len = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt();
    if len <= f32::EPSILON {
        return p;
    }
    let inv_len = 1.0 / len;
    [p[0] * inv_len, p[1] * inv_len, p[2] * inv_len, p[3] * inv_len]
}

impl Frustum {
    /// Extracts the six frustum planes from a combined view-projection
    /// matrix (Gribb-Hartmann method), assuming an OpenGL-style clip space
    /// where the visible volume is `-w <= x, y, z <= w`.
    pub fn from_view_proj(vp: &Mat4) -> Frustum {
        // Row `r` of `vp` treated as a row-major matrix, i.e. the
        // coefficients clip.{x,y,z,w} is a linear combination of.
        let row = |r: usize| [vp.get(r, 0), vp.get(r, 1), vp.get(r, 2), vp.get(r, 3)];
        let (row0, row1, row2, row3) = (row(0), row(1), row(2), row(3));

        let add = |a: [f32; 4], b: [f32; 4]| [a[0] + b[0], a[1] + b[1], a[2] + b[2], a[3] + b[3]];
        let sub = |a: [f32; 4], b: [f32; 4]| [a[0] - b[0], a[1] - b[1], a[2] - b[2], a[3] - b[3]];

        let raw = [
            add(row3, row0), // left
            sub(row3, row0), // right
            add(row3, row1), // bottom
            sub(row3, row1), // top
            add(row3, row2), // near
            sub(row3, row2), // far
        ];
        let planes = raw.map(normalize_plane);
        Frustum { planes }
    }

    /// Whether `p` lies on the inside of every plane (inclusive of the boundary).
    pub fn contains_point(&self, p: Vec3) -> bool {
        self.planes.iter().all(|pl| pl[0] * p.x + pl[1] * p.y + pl[2] * p.z + pl[3] >= 0.0)
    }

    /// Classifies an [`Aabb`] against the frustum using the standard
    /// positive/negative-vertex test: for each plane, the box's most
    /// forward corner along the plane normal must clear it, or the whole
    /// box is [`Containment::Outside`]; if only the box's most backward
    /// corner fails, the box straddles that plane.
    pub fn intersects_aabb(&self, aabb: &Aabb) -> Containment {
        let mut intersecting = false;
        for pl in &self.planes {
            let normal = Vec3::new(pl[0], pl[1], pl[2]);
            let d = pl[3];

            let p_vertex = Vec3::new(
                if normal.x >= 0.0 { aabb.max.x } else { aabb.min.x },
                if normal.y >= 0.0 { aabb.max.y } else { aabb.min.y },
                if normal.z >= 0.0 { aabb.max.z } else { aabb.min.z },
            );
            if normal.dot(p_vertex) + d < 0.0 {
                return Containment::Outside;
            }

            let n_vertex = Vec3::new(
                if normal.x >= 0.0 { aabb.min.x } else { aabb.max.x },
                if normal.y >= 0.0 { aabb.min.y } else { aabb.max.y },
                if normal.z >= 0.0 { aabb.min.z } else { aabb.max.z },
            );
            if normal.dot(n_vertex) + d < 0.0 {
                intersecting = true;
            }
        }
        if intersecting {
            Containment::Intersect
        } else {
            Containment::Inside
        }
    }

    /// Reads one plane by index in `[left, right, bottom, top, near, far]`
    /// order.
    ///
    /// # Panics
    /// Panics if `index >= 6`.
    pub fn plane(&self, index: usize) -> [f32; 4] {
        assert!(index < 6, "Frustum plane index out of range: {index}");
        // SAFETY: the assert above guarantees `index < 6`, and
        // `self.planes` has exactly 6 elements, so this access is in bounds.
        unsafe { *self.planes.get_unchecked(index) }
    }

    /// Views the six planes as a contiguous `&[f32]` of 24 floats, 4 per
    /// plane in `[left, right, bottom, top, near, far]` order.
    pub fn as_slice(&self) -> &[f32] {
        // SAFETY: `self.planes` is a `[[f32; 4]; 6]`, a plain array of
        // arrays with no padding between or within elements, so a pointer
        // to its first element is a valid, aligned pointer to the start of
        // a contiguous 24-element `f32` array; the returned slice borrows
        // `self` and cannot outlive it.
        unsafe { std::slice::from_raw_parts(self.planes.as_ptr() as *const f32, 24) }
    }

    /// Copies the six (already-normalized) planes into an owned array.
    pub fn as_array(&self) -> [[f32; 4]; 6] {
        let mut out = [MaybeUninit::<[f32; 4]>::uninit(); 6];
        for (slot, plane) in out.iter_mut().zip(self.planes.iter()) {
            slot.write(*plane);
        }
        // SAFETY: the loop above wrote one element of `out` per plane in
        // `self.planes` (6 total, matched pairwise by `zip`), so every
        // element of `out` was initialized before this point.
        unsafe { out.map(|p| p.assume_init()) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f32 = 1e-3;

    fn camera_frustum() -> Frustum {
        // A symmetric perspective camera at the origin looking down -Z.
        let proj = Mat4::perspective(std::f32::consts::FRAC_PI_2, 1.0, 1.0, 100.0);
        Frustum::from_view_proj(&proj)
    }

    #[test]
    fn planes_are_unit_normalized() {
        let f = camera_frustum();
        for pl in f.planes {
            let len = (pl[0] * pl[0] + pl[1] * pl[1] + pl[2] * pl[2]).sqrt();
            assert!((len - 1.0).abs() < EPS, "plane normal not unit length: {len}");
        }
    }

    #[test]
    fn contains_point_inside_and_outside_the_view_volume() {
        let f = camera_frustum();
        assert!(f.contains_point(Vec3::new(0.0, 0.0, -10.0)));
        assert!(!f.contains_point(Vec3::new(0.0, 0.0, 10.0))); // behind the camera
        assert!(!f.contains_point(Vec3::new(0.0, 0.0, -1000.0))); // past the far plane
    }

    #[test]
    fn aabb_fully_inside_is_classified_inside() {
        let f = camera_frustum();
        let box_in_view = Aabb::from_min_max(Vec3::new(-0.5, -0.5, -5.5), Vec3::new(0.5, 0.5, -4.5));
        assert_eq!(f.intersects_aabb(&box_in_view), Containment::Inside);
    }

    #[test]
    fn aabb_fully_outside_is_classified_outside() {
        let f = camera_frustum();
        let box_behind = Aabb::from_min_max(Vec3::new(-0.5, -0.5, 50.0), Vec3::new(0.5, 0.5, 51.0));
        assert_eq!(f.intersects_aabb(&box_behind), Containment::Outside);
    }

    #[test]
    fn aabb_straddling_a_plane_is_classified_intersecting() {
        let f = camera_frustum();
        // Straddles the near/far range: spans from well before the near
        // plane to well past the far plane.
        let straddling = Aabb::from_min_max(Vec3::new(-0.1, -0.1, -1000.0), Vec3::new(0.1, 0.1, 5.0));
        assert_eq!(f.intersects_aabb(&straddling), Containment::Intersect);
    }

    #[test]
    fn plane_accessor_matches_the_planes_array() {
        let f = camera_frustum();
        for (i, expected) in f.planes.iter().enumerate() {
            assert_eq!(f.plane(i), *expected);
        }
        assert_eq!(f.as_slice().len(), 24);
        assert_eq!(f.as_array(), f.planes);
    }

    #[test]
    #[should_panic]
    fn plane_out_of_range_panics() {
        camera_frustum().plane(6);
    }
}
